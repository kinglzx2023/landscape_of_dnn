"""BERT baseline fine-tuning on MRPC with geometric weight metrics."""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification, BertTokenizer


def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


RANDOM_SEED = 42
seed_everything(RANDOM_SEED)

# Hyperparameters
BATCH_SIZE = 96
LEARNING_RATE = 2e-5
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.00
LAMBDA_VAR = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "/root/BERT/mrpc/result"
LOCAL_DATASET_PATH = "/root/autodl-tmp/glue_mrpc_local"
LOCAL_MODEL_PATH = "/root/autodl-tmp/bert-base-uncased-local"

MATRIX_SUFFIXES = {
    "query": "attention.self.query.weight",
    "key": "attention.self.key.weight",
    "value": "attention.self.value.weight",
    "attn_out": "attention.output.dense.weight",
    "intermediate": "intermediate.dense.weight",
    "output": "output.dense.weight",
}

PII_TARGET_LAYER = 5

os.makedirs(RESULT_PATH, exist_ok=True)


def load_mrpc_dataset():
    """Load MRPC from disk or download from Hugging Face."""
    print("Loading MRPC dataset...")
    try:
        dataset = load_from_disk(LOCAL_DATASET_PATH)
        print(f"Loaded dataset from {LOCAL_DATASET_PATH}")
    except Exception:
        print("Local dataset not found; downloading MRPC from Hugging Face...")
        dataset = load_dataset("glue", "mrpc")
        dataset.save_to_disk(LOCAL_DATASET_PATH)
        print(f"Dataset saved to {LOCAL_DATASET_PATH}")
    return dataset


def load_bert_tokenizer():
    """Load BERT tokenizer from local cache or download once."""
    print("Loading BERT tokenizer and model...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Local model not found; downloading BERT-base-uncased...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_temp = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        model_temp.save_pretrained(LOCAL_MODEL_PATH)
        del model_temp
        print(f"Model and tokenizer saved to {LOCAL_MODEL_PATH}")
    else:
        print(f"Using cached model at {LOCAL_MODEL_PATH}")

    return BertTokenizer.from_pretrained(LOCAL_MODEL_PATH)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["sentence1"],
        examples["sentence2"],
        padding="max_length",
        truncation=True,
        max_length=128,
    )


def prepare_dataloaders(dataset, tokenizer):
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
    )
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader, eval_dataloader


def compute_cosine_similarity_means(matrix):
    """Return mean absolute row-wise and column-wise cosine similarity."""
    matrix = matrix.detach().float()
    if matrix.dim() > 2:
        matrix = matrix.reshape(-1, matrix.shape[-1])

    num_rows = matrix.shape[0]
    if num_rows > 1:
        if num_rows > 5000:
            indices = torch.randperm(num_rows)[:5000]
            sub_matrix = matrix[indices]
            norm_row = F.normalize(sub_matrix, p=2, dim=1)
            sim_row = torch.mm(norm_row, norm_row.t()).abs()
            row_out = (sim_row.mean() * 5000 - 1.0) / 4999.0
        else:
            norm_row = F.normalize(matrix, p=2, dim=1)
            sim_row = torch.mm(norm_row, norm_row.t()).abs()
            row_out = (sim_row.mean() * num_rows - 1.0) / (num_rows - 1.0)
    else:
        row_out = torch.tensor(0.0)

    num_cols = matrix.shape[1]
    if num_cols > 1:
        norm_col = F.normalize(matrix, p=2, dim=0)
        sim_col = torch.mm(norm_col.t(), norm_col).abs()
        col_out = (sim_col.mean() * num_cols - 1.0) / (num_cols - 1.0)
    else:
        col_out = torch.tensor(0.0)

    return row_out.abs().item(), col_out.abs().item()


def calculate_geometric_metrics(model, compute_spectral=False):
    """Aggregate geometric metrics across all encoder layers."""
    metrics_accum = {
        k: {"TraceLog": 0.0, "Vol": 0.0, "Var": 0.0, "SpecNorm": 0.0, "count": 0}
        for k in MATRIX_SUFFIXES
    }
    pii_metrics = {}
    total_variance_penalty = torch.tensor(0.0, device=DEVICE)

    for name, param in model.named_parameters():
        if not name.startswith("bert.encoder.layer."):
            continue

        matrix_type = None
        for k, suffix in MATRIX_SUFFIXES.items():
            if name.endswith(suffix):
                matrix_type = k
                break

        if matrix_type is None:
            continue

        norms_sq = torch.sum(param**2, dim=1)
        norms_sq_safe = torch.clamp(norms_sq, min=1e-6)

        trace = torch.sum(norms_sq_safe)
        trace_log = torch.log(trace)
        vol = torch.sum(torch.log(norms_sq_safe))
        var = torch.var(norms_sq_safe, unbiased=False)

        metrics_accum[matrix_type]["TraceLog"] += trace_log.item()
        metrics_accum[matrix_type]["Vol"] += vol.item()
        metrics_accum[matrix_type]["Var"] += var.item()

        if compute_spectral:
            spec_norm = torch.linalg.matrix_norm(param, ord=2)
            metrics_accum[matrix_type]["SpecNorm"] += spec_norm.item()

            if name.startswith(f"bert.encoder.layer.{PII_TARGET_LAYER}."):
                _, col_sim = compute_cosine_similarity_means(param)
                pii_metrics[f"{matrix_type}_PII"] = col_sim

        metrics_accum[matrix_type]["count"] += 1
        total_variance_penalty += var

    metrics = {}
    for k, vals in metrics_accum.items():
        count = vals["count"]
        if count > 0:
            metrics[f"{k}_TraceLog"] = vals["TraceLog"] / count
            metrics[f"{k}_Vol"] = vals["Vol"] / count
            metrics[f"{k}_Var"] = vals["Var"] / count
            if compute_spectral:
                metrics[f"{k}_SpecNorm"] = vals["SpecNorm"] / count

    metrics.update(pii_metrics)
    return metrics, total_variance_penalty


def compute_accuracy(model, dataloader):
    """Compute classification accuracy on a dataloader."""
    model.eval()
    correct_pred = 0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            _, predicted_labels = torch.max(outputs.logits, dim=1)
            correct_pred += (predicted_labels == labels).sum().item()
            total_samples += labels.size(0)

    model.train()
    return correct_pred / total_samples * 100


def get_experiment_mode():
    if WEIGHT_DECAY > 0.0 and LAMBDA_VAR == 0.0:
        return "L2_Only"
    if WEIGHT_DECAY == 0.0 and LAMBDA_VAR > 0.0:
        return "Var_Only"
    if WEIGHT_DECAY > 0.0 and LAMBDA_VAR > 0.0:
        return "L2_and_Var"
    return "Baseline"


def run_experiment():
    mode_str = get_experiment_mode()
    exp_name = f"BERT_{mode_str}"

    print(f"\n{'=' * 70}")
    print(f"Experiment: {exp_name}")
    print("Model: BERT-base | Metrics averaged over all encoder layers")
    print(f"PII tracking layer: {PII_TARGET_LAYER}")
    print(f"{'=' * 70}")

    model = BertForSequenceClassification.from_pretrained(
        LOCAL_MODEL_PATH, num_labels=2
    ).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    history = {"Epoch": [], "Train_Loss": [], "Val_Accuracy": []}
    pii_history = {"Epoch": []}
    for k in MATRIX_SUFFIXES:
        history[f"{k}_Vol"] = []
        history[f"{k}_TraceLog"] = []
        history[f"{k}_Var"] = []
        history[f"{k}_SpecNorm"] = []
        pii_history[f"{k}_PII"] = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            loss = outputs.loss

            _, total_var = calculate_geometric_metrics(model, compute_spectral=False)
            if LAMBDA_VAR > 0.0:
                loss += LAMBDA_VAR * total_var

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        val_acc = compute_accuracy(model, eval_dataloader)

        with torch.no_grad():
            final_metrics, _ = calculate_geometric_metrics(model, compute_spectral=True)

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Loss: {avg_train_loss:.4f} | Val accuracy: {val_acc:5.2f}%"
        )
        for k in MATRIX_SUFFIXES:
            print(
                f"  {k.upper():<12}: Var = {final_metrics[f'{k}_Var']:.6f} | "
                f"SpecNorm = {final_metrics[f'{k}_SpecNorm']:.4f} | "
                f"log(Trace) = {final_metrics[f'{k}_TraceLog']:.4f} | "
                f"PII (L{PII_TARGET_LAYER}) = {final_metrics[f'{k}_PII']:.5f}"
            )

        history["Epoch"].append(epoch + 1)
        pii_history["Epoch"].append(epoch + 1)
        history["Train_Loss"].append(avg_train_loss)
        history["Val_Accuracy"].append(val_acc)

        for k in MATRIX_SUFFIXES:
            history[f"{k}_Vol"].append(final_metrics[f"{k}_Vol"])
            history[f"{k}_TraceLog"].append(final_metrics[f"{k}_TraceLog"])
            history[f"{k}_Var"].append(final_metrics[f"{k}_Var"])
            history[f"{k}_SpecNorm"].append(final_metrics[f"{k}_SpecNorm"])
            pii_history[f"{k}_PII"].append(final_metrics[f"{k}_PII"])

    df = pd.DataFrame(history)
    pii_df = pd.DataFrame(pii_history)

    print(f"\nExperiment complete: {exp_name}")
    corr_path = os.path.join(RESULT_PATH, f"{exp_name}_correlation.txt")
    with open(corr_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment mode: {mode_str}\n")
        f.write("Target layer: average of all encoder layers\n\n")
        f.write("Pearson correlations (Volume vs log(Trace)):\n")
        f.write("-" * 65 + "\n")

        for k in MATRIX_SUFFIXES:
            corr_vol_trace_log = df[f"{k}_Vol"].corr(df[f"{k}_TraceLog"])
            print(f"  {k.upper():<12} Vol vs log(Trace) correlation: {corr_vol_trace_log:+.4f}")
            f.write(f"{k.upper()}:\n")
            f.write(f"  - Vol vs log(Trace): {corr_vol_trace_log:.6f}\n\n")

    csv_path = os.path.join(RESULT_PATH, f"{exp_name}_history.csv")
    pii_csv_path = os.path.join(RESULT_PATH, f"{exp_name}_PII_history.csv")
    df.to_csv(csv_path, index=False)
    pii_df.to_csv(pii_csv_path, index=False)

    print(
        f"\nResults saved to:\n"
        f"  {csv_path}\n"
        f"  {pii_csv_path}\n"
        f"  {corr_path}"
    )


if __name__ == "__main__":
    dataset = load_mrpc_dataset()
    tokenizer = load_bert_tokenizer()
    train_dataloader, eval_dataloader = prepare_dataloaders(dataset, tokenizer)
    run_experiment()
