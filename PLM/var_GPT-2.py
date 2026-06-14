"""GPT-2 fine-tuning with variance regularization on WikiText-2."""

import math
import os
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset, load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer


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
BATCH_SIZE = 22
BLOCK_SIZE = 512
LEARNING_RATE = 5e-5
NUM_EPOCHS = 50
WEIGHT_DECAY = 0.00
LAMBDA_VAR = 0.0001

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULT_PATH = "/root/GPT-2/result"
LOCAL_DATASET_PATH = "/root/autodl-tmp/"
LOCAL_MODEL_PATH = "/root/autodl-tmp/gpt2-local"

TARGET_LAYER_IDX = 5
TARGET_MATRICES = {
    "qkv_attn": f"transformer.h.{TARGET_LAYER_IDX}.attn.c_attn.weight",
    "attn_out": f"transformer.h.{TARGET_LAYER_IDX}.attn.c_proj.weight",
    "mlp_in": f"transformer.h.{TARGET_LAYER_IDX}.mlp.c_fc.weight",
    "mlp_out": f"transformer.h.{TARGET_LAYER_IDX}.mlp.c_proj.weight",
}

os.makedirs(RESULT_PATH, exist_ok=True)


def load_wikitext_dataset():
    """Load WikiText-2 from disk or download from Hugging Face."""
    print("Loading WikiText-2 dataset...")
    try:
        dataset = load_from_disk(LOCAL_DATASET_PATH)
        print(f"Loaded dataset from {LOCAL_DATASET_PATH}")
    except Exception:
        print("Local dataset not found; downloading WikiText-2 from Hugging Face...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        dataset.save_to_disk(LOCAL_DATASET_PATH)
        print(f"Dataset saved to {LOCAL_DATASET_PATH}")
    return dataset


def load_gpt2_tokenizer():
    """Load GPT-2 tokenizer from local cache or download once."""
    print("Loading GPT-2 tokenizer and model...")
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Local model not found; downloading GPT-2...")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model_temp = GPT2LMHeadModel.from_pretrained("gpt2")
        os.makedirs(LOCAL_MODEL_PATH, exist_ok=True)
        tokenizer.save_pretrained(LOCAL_MODEL_PATH)
        model_temp.save_pretrained(LOCAL_MODEL_PATH)
        del model_temp
        print(f"Model and tokenizer saved to {LOCAL_MODEL_PATH}")
    else:
        print(f"Using cached model at {LOCAL_MODEL_PATH}")

    tokenizer = GPT2Tokenizer.from_pretrained(LOCAL_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples):
    """Concatenate and chunk tokenized text for causal language modeling."""
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // BLOCK_SIZE) * BLOCK_SIZE
    result = {
        k: [t[i : i + BLOCK_SIZE] for i in range(0, total_length, BLOCK_SIZE)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def prepare_dataloaders(dataset, tokenizer):
    tokenized_datasets = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        desc=f"Grouping texts into {BLOCK_SIZE}-token blocks",
    )
    lm_datasets.set_format("torch")

    train_dataloader = DataLoader(
        lm_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        lm_datasets["validation"],
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
    )
    return train_dataloader, eval_dataloader


def calculate_geometric_metrics(model):
    """Compute Trace, Volume, and variance metrics for target weight matrices."""
    metrics = {}
    total_variance_penalty = torch.tensor(0.0, device=DEVICE)

    for name, param in model.named_parameters():
        if name not in TARGET_MATRICES.values():
            continue

        matrix_type = next(k for k, v in TARGET_MATRICES.items() if v == name)
        # GPT-2 Conv1D weights are [in_features, out_features]; sum over dim=0
        norms_sq = torch.sum(param**2, dim=0)
        norms_sq_safe = torch.clamp(norms_sq, min=1e-6)

        trace = torch.sum(norms_sq_safe)
        vol = torch.sum(torch.log(norms_sq_safe))
        var = torch.var(norms_sq_safe, unbiased=False)

        metrics[f"{matrix_type}_Trace"] = trace.item()
        metrics[f"{matrix_type}_Vol"] = vol.item()
        metrics[f"{matrix_type}_Var"] = var.item()
        total_variance_penalty += var

    return metrics, total_variance_penalty


def compute_perplexity(model, dataloader):
    """Compute validation perplexity."""
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
    return ppl


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
    exp_name = f"GPT2_{mode_str}"

    print(f"\n{'=' * 70}")
    print(f"Experiment: {exp_name}")
    print(f"Model: GPT-2 | Target layer: {TARGET_LAYER_IDX}")
    print("Task: Language modeling (WikiText-2) | Metric: Perplexity")
    print(f"{'=' * 70}")

    model = GPT2LMHeadModel.from_pretrained(LOCAL_MODEL_PATH).to(DEVICE)

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

    history = {"Epoch": [], "Train_Loss": [], "Val_PPL": []}
    for k in TARGET_MATRICES:
        history[f"{k}_Vol"] = []
        history[f"{k}_Trace"] = []
        history[f"{k}_Var"] = []

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            _, total_var = calculate_geometric_metrics(model)
            if LAMBDA_VAR > 0.0:
                loss += LAMBDA_VAR * total_var

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dataloader)
        val_ppl = compute_perplexity(model, eval_dataloader)

        with torch.no_grad():
            final_metrics, _ = calculate_geometric_metrics(model)

        print(
            f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}] | "
            f"Train loss: {avg_train_loss:.4f} | Val PPL: {val_ppl:7.2f}"
        )
        for k in TARGET_MATRICES:
            print(
                f"  {k.upper():<12}: Var = {final_metrics[f'{k}_Var']:.6f} | "
                f"Vol = {final_metrics[f'{k}_Vol']:.2f} | "
                f"Trace = {final_metrics[f'{k}_Trace']:.2f}"
            )

        history["Epoch"].append(epoch + 1)
        history["Train_Loss"].append(avg_train_loss)
        history["Val_PPL"].append(val_ppl)
        for k, v in final_metrics.items():
            history[k].append(v)

    df = pd.DataFrame(history)

    print(f"\nExperiment complete: {exp_name}")
    corr_path = os.path.join(RESULT_PATH, f"{exp_name}_correlation.txt")
    with open(corr_path, "w", encoding="utf-8") as f:
        f.write(f"Experiment mode: {mode_str}\n")
        f.write(f"Target layer: {TARGET_LAYER_IDX}\n\n")
        f.write("Pearson correlations (Volume vs Trace):\n")
        f.write("-" * 40 + "\n")
        for k in TARGET_MATRICES:
            vol_col = f"{k}_Vol"
            trace_col = f"{k}_Trace"
            correlation = df[vol_col].corr(df[trace_col])
            print(f"  {k.upper():<12} Volume vs Trace correlation: {correlation:.4f}")
            f.write(f"{k.upper():<15}: {correlation:.6f}\n")

    csv_path = os.path.join(RESULT_PATH, f"{exp_name}_history.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to:\n  {csv_path}\n  {corr_path}")


if __name__ == "__main__":
    dataset = load_wikitext_dataset()
    tokenizer = load_gpt2_tokenizer()
    train_dataloader, eval_dataloader = prepare_dataloaders(dataset, tokenizer)
    run_experiment()
