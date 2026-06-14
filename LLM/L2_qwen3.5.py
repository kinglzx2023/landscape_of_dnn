"""Fine-tune Qwen3.5-0.8B on Wikitext-2 with L2 (weight decay) regularization."""

import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup


def seed_everything(seed: int = 42) -> None:
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

BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
LAMBDA_VAR = 0.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "/root/Qwen3.5-0.8B"
LOCAL_DATASET_PATH = "/root/autodl-tmp/"
LOCAL_MODEL_PATH = "/root/autodl-tmp/Qwen3.5-0.8B"
BLOCK_SIZE = 128
PII_TARGET_LAYER = 10

os.makedirs(RESULT_PATH, exist_ok=True)

MATRIX_SUFFIXES = {
    "in_proj_qkv": "linear_attn.in_proj_qkv.weight",
    "in_proj_z": "linear_attn.in_proj_z.weight",
    "in_proj_b": "linear_attn.in_proj_b.weight",
    "in_proj_a": "linear_attn.in_proj_a.weight",
    "attn_out": "linear_attn.out_proj.weight",
    "mlp_gate": "mlp.gate_proj.weight",
    "mlp_up": "mlp.up_proj.weight",
    "mlp_down": "mlp.down_proj.weight",
}

NO_DECAY_PARAM_NAMES = (
    "bias",
    "input_layernorm.weight",
    "post_attention_layernorm.weight",
    "norm.weight",
)


def load_wikitext_dataset():
    """Load Wikitext-2 from disk or download and cache it locally."""
    print("Checking dataset...")
    try:
        dataset = load_from_disk(LOCAL_DATASET_PATH)
        print("Loaded wikitext-2 from local cache.")
    except Exception:
        print("Local dataset not found; downloading from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        dataset.save_to_disk(LOCAL_DATASET_PATH)
        print("Dataset downloaded and saved locally.")
    return dataset


def build_dataloaders(dataset):
    """Tokenize and chunk the dataset into training and validation loaders."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenized = dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
        remove_columns=["text"],
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated[list(examples.keys())[0]])
        total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
        result = {
            k: [tokens[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
            for k, tokens in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(group_texts, batched=True)
    lm_datasets.set_format("torch")

    train_loader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        lm_datasets["validation"],
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    return train_loader, eval_loader


def compute_cosine_similarity_means(matrix):
    """Compute mean row/column cosine similarity for a weight matrix."""
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


def compute_variance_penalty(model):
    """Compute variance penalty over target weight matrices (keeps gradients)."""
    total_variance_penalty = torch.tensor(0.0, device=DEVICE)
    valid_suffixes = tuple(MATRIX_SUFFIXES.values())

    for name, param in model.named_parameters():
        if name.startswith("model.layers.") and name.endswith(valid_suffixes):
            norms_sq = torch.sum(param**2, dim=1)
            norms_sq_safe = torch.clamp(norms_sq, min=1e-6)
            total_variance_penalty += torch.var(norms_sq_safe, unbiased=False)

    return total_variance_penalty


@torch.no_grad()
def calculate_geometric_metrics(model, compute_spectral=False):
    """Compute geometric metrics without building a gradient graph."""
    metrics_accum = {
        k: {"TraceLog": 0.0, "Vol": 0.0, "Var": 0.0, "SpecNorm": 0.0, "count": 0}
        for k in MATRIX_SUFFIXES
    }
    pii_metrics = {}

    for name, param in model.named_parameters():
        if not name.startswith("model.layers."):
            continue

        matrix_type = None
        for key, suffix in MATRIX_SUFFIXES.items():
            if name.endswith(suffix):
                matrix_type = key
                break

        if matrix_type is None:
            continue

        norms_sq = torch.sum(param**2, dim=1)
        norms_sq_safe = torch.clamp(norms_sq, min=1e-6)

        trace_log = torch.log(torch.sum(norms_sq_safe))
        vol = torch.sum(torch.log(norms_sq_safe))
        var = torch.var(norms_sq_safe, unbiased=False)

        metrics_accum[matrix_type]["TraceLog"] += trace_log.item()
        metrics_accum[matrix_type]["Vol"] += vol.item()
        metrics_accum[matrix_type]["Var"] += var.item()

        if compute_spectral:
            neuron_norms = torch.linalg.vector_norm(param, ord=2, dim=1)
            metrics_accum[matrix_type]["SpecNorm"] += torch.max(neuron_norms).item()

            if name.startswith(f"model.layers.{PII_TARGET_LAYER}."):
                row_sim, _ = compute_cosine_similarity_means(param)
                pii_metrics[f"{matrix_type}_PII"] = row_sim

        metrics_accum[matrix_type]["count"] += 1

    metrics = {}
    for key, vals in metrics_accum.items():
        count = vals["count"]
        if count == 0:
            continue
        metrics[f"{key}_TraceLog"] = vals["TraceLog"] / count
        metrics[f"{key}_Vol"] = vals["Vol"] / count
        metrics[f"{key}_Var"] = vals["Var"] / count
        if compute_spectral:
            metrics[f"{key}_SpecNorm"] = vals["SpecNorm"] / count

    metrics.update(pii_metrics)
    return metrics


def compute_perplexity(model, dataloader):
    """Evaluate average loss and perplexity on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_steps = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()
            total_steps += 1

    avg_loss = total_loss / total_steps
    try:
        ppl = math.exp(avg_loss)
    except OverflowError:
        ppl = float("inf")

    model.train()
    return avg_loss, ppl


def resolve_experiment_mode():
    """Derive experiment label from regularization hyperparameters."""
    if WEIGHT_DECAY > 0.0 and LAMBDA_VAR == 0.0:
        return "L2_Only"
    if WEIGHT_DECAY == 0.0 and LAMBDA_VAR > 0.0:
        return "Var_Only"
    if WEIGHT_DECAY > 0.0 and LAMBDA_VAR > 0.0:
        return "L2_and_Var"
    return "Baseline"


def run_experiment():
    """Run fine-tuning and log geometric metrics."""
    mode_str = resolve_experiment_mode()
    exp_name = f"Qwen3.5-0.8B_{mode_str}"

    print(f"\n{'=' * 70}")
    print(f"Starting experiment: {exp_name}")
    print("Model: Qwen3.5-0.8B (Linear Attn) | Dataset: Wikitext-2")
    print(f"PII tracking layer: {PII_TARGET_LAYER}")
    print(f"{'=' * 70}")

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in NO_DECAY_PARAM_NAMES)
            ],
            "weight_decay": WEIGHT_DECAY,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in NO_DECAY_PARAM_NAMES)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

    total_steps = len(train_dataloader) * NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    print(f"Total optimization steps: {total_steps} | Warmup steps: {warmup_steps}")

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = {"Epoch": [], "Train_Loss": [], "Val_Loss": [], "Val_PPL": []}
    pii_history = {"Epoch": []}
    for key in MATRIX_SUFFIXES:
        history[f"{key}_Vol"] = []
        history[f"{key}_TraceLog"] = []
        history[f"{key}_Var"] = []
        history[f"{key}_SpecNorm"] = []
        pii_history[f"{key}_PII"] = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if LAMBDA_VAR > 0.0:
                    loss = loss + LAMBDA_VAR * compute_variance_penalty(model)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        val_loss, val_ppl = compute_perplexity(model, eval_dataloader)
        final_metrics = calculate_geometric_metrics(model, compute_spectral=True)

        current_lr = scheduler.get_last_lr()[0]
        print(
            f"\nEp [{epoch + 1}/{NUM_EPOCHS}] | LR: {current_lr:.2e} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val PPL: {val_ppl:7.2f}"
        )

        for key in MATRIX_SUFFIXES:
            print(
                f"  -> {key.upper():<12}: Var = {final_metrics.get(f'{key}_Var', 0.0):.6f} | "
                f"SpecNorm Proxy = {final_metrics.get(f'{key}_SpecNorm', 0.0):.4f} | "
                f"log(Trace) = {final_metrics.get(f'{key}_TraceLog', 0.0):.4f} | "
                f"PII (L{PII_TARGET_LAYER}) = {final_metrics.get(f'{key}_PII', 0.0):.5f}"
            )

        history["Epoch"].append(epoch + 1)
        pii_history["Epoch"].append(epoch + 1)
        history["Train_Loss"].append(avg_train_loss)
        history["Val_Loss"].append(val_loss)
        history["Val_PPL"].append(val_ppl)

        for key in MATRIX_SUFFIXES:
            history[f"{key}_Vol"].append(final_metrics.get(f"{key}_Vol", 0.0))
            history[f"{key}_TraceLog"].append(final_metrics.get(f"{key}_TraceLog", 0.0))
            history[f"{key}_Var"].append(final_metrics.get(f"{key}_Var", 0.0))
            history[f"{key}_SpecNorm"].append(final_metrics.get(f"{key}_SpecNorm", 0.0))
            pii_history[f"{key}_PII"].append(final_metrics.get(f"{key}_PII", 0.0))

    df = pd.DataFrame(history)
    pii_df = pd.DataFrame(pii_history)

    print(f"\nExperiment finished: {exp_name}")

    corr_path = os.path.join(RESULT_PATH, f"{exp_name}_correlation.txt")
    with open(corr_path, "w", encoding="utf-8") as f:
        f.write("========================================\n")
        f.write(" Experiment Configuration / Hyperparameters\n")
        f.write("========================================\n")
        f.write(f"Experiment Mode: {mode_str}\n")
        f.write("Model:           Qwen3.5-0.8B (Linear Attn)\n")
        f.write("Dataset:         Wikitext-2\n")
        f.write(f"Random Seed:     {RANDOM_SEED}\n")
        f.write(f"Batch Size:      {BATCH_SIZE}\n")
        f.write(f"Learning Rate:   {LEARNING_RATE}\n")
        f.write(f"Num Epochs:      {NUM_EPOCHS}\n")
        f.write(f"Weight Decay:    {WEIGHT_DECAY}\n")
        f.write(f"Lambda Var:      {LAMBDA_VAR}\n")
        f.write(f"Warmup Steps:    {warmup_steps} (out of {total_steps} total steps)\n")
        f.write(f"Max Seq Length:  {BLOCK_SIZE} (Tokenization limit)\n")
        f.write("Gradient Clip:   1.0\n")
        f.write(f"PII Target Lyr:  {PII_TARGET_LAYER}\n")
        f.write("========================================\n\n")
        f.write("Target Layer: Average of All Encoder Layers\n\n")
        f.write("Pearson Correlations (Vol vs log(Trace) Only):\n")
        f.write("-" * 65 + "\n")

        for key in MATRIX_SUFFIXES:
            vol_col = f"{key}_Vol"
            trace_log_col = f"{key}_TraceLog"
            if df[vol_col].nunique() > 1 and df[trace_log_col].nunique() > 1:
                corr_vol_trace_log = df[vol_col].corr(df[trace_log_col])
            else:
                corr_vol_trace_log = float("nan")

            print(f"--> {key.upper():<12} Vol-log(Trace) correlation: {corr_vol_trace_log:+.4f}")
            f.write(f"{key.upper()}:\n")
            f.write(f"  - Vol vs log(Trace): {corr_vol_trace_log:.6f}\n\n")

    csv_path = os.path.join(RESULT_PATH, f"{exp_name}_history.csv")
    pii_csv_path = os.path.join(RESULT_PATH, f"{exp_name}_PII_history.csv")
    df.to_csv(csv_path, index=False)
    pii_df.to_csv(pii_csv_path, index=False)

    print(
        f"\nResults saved to:\n"
        f"- Global metrics: {csv_path}\n"
        f"- Layer PII metrics: {pii_csv_path}\n"
        f"- Config and correlations: {corr_path}"
    )


dataset = load_wikitext_dataset()
train_dataloader, eval_dataloader = build_dataloaders(dataset)

if __name__ == "__main__":
    run_experiment()
