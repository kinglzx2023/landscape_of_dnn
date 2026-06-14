"""
MLP volume analysis on MNIST with AdamW and L2 weight decay.

Trains a multi-layer perceptron and records per-epoch metrics:
  - PII (pairwise inner-product index) from column cosine similarity
  - Log-volume, Gram-matrix rank, and diagonal statistics of hidden weights

Outputs three text files under the result directory:
  *_volume.txt, *_acc.txt, *_diag.txt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from scipy.spatial.distance import cosine
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Hyperparameters (defaults match the paper experiments)
# ---------------------------------------------------------------------------
DEFAULT_DATA_ROOT = ""
DEFAULT_RESULT_DIR = ""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an MLP on MNIST and log volume / PII metrics (AdamW + L2)."
    )
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--result-dir", type=str, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--input-size", type=int, default=784)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-size", type=int, default=500)
    parser.add_argument("--num-hidden-layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--epoch-interval", type=int, default=5,
                        help="Record volume metrics every N epochs.")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--init-method", type=str, default="xavier",
                        choices=["xavier", "kaiming", "normal", "uniform",
                                 "orthogonal", "constant"])
    parser.add_argument("--gpu", type=int, default=1,
                        help="CUDA device index; use -1 for CPU.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def get_device(gpu: int) -> torch.device:
    if gpu >= 0 and torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def build_experiment_name(args: argparse.Namespace) -> str:
    return (
        f"L2_{args.init_method}_AdamW"
        f"_B_{args.batch_size}"
        f"_epoch_{args.num_epochs}"
        f"_lr_{args.learning_rate}"
        f"_width_{args.hidden_size}"
        f"_seed_{args.random_seed}"
        f"_layers_{args.num_hidden_layers + 1}"
        f"_L2_{args.weight_decay}"
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def cosine_similarity_matrix_column(matrix: np.ndarray) -> np.ndarray:
    num_columns = matrix.shape[1]
    similarity = np.zeros((num_columns, num_columns))
    for i in range(num_columns):
        for j in range(i, num_columns):
            value = 1.0 - cosine(matrix[:, i], matrix[:, j])
            similarity[i, j] = value
            similarity[j, i] = value
    return np.abs(similarity)


def mean_pii(similarity_matrix: np.ndarray) -> float:
    """Mean pairwise inner-product index (PII) for a similarity matrix."""
    n = similarity_matrix.shape[0]
    matrix_mean = similarity_matrix.mean()
    return abs((matrix_mean - (1.0 / n)) * (n / (n - 1)))


def compute_volume_metrics(weight: torch.Tensor) -> dict:
    matrix = weight.detach().cpu().numpy()
    gram_matrix = matrix.T @ matrix
    diagonal = np.diag(gram_matrix)

    log_diag = np.log(diagonal[diagonal > 0])
    return {
        "diag_sum": diagonal.sum(),
        "diag_mean": diagonal.mean(),
        "diag_var": diagonal.var(),
        "log_volume": log_diag.sum(),
        "rank": np.linalg.matrix_rank(gram_matrix),
        "diagonal": diagonal,
    }


def log_layer_metrics(
    param_name: str,
    weight: torch.Tensor,
    experiment_name: str,
    volume_file,
    diag_file,
) -> None:
    similarity = cosine_similarity_matrix_column(weight.detach().cpu().numpy())
    pii = round(mean_pii(similarity), 6)
    metrics = compute_volume_metrics(weight)

    print(f"Parameter name: {param_name}")
    print(f"Parameter shape: {tuple(weight.shape)}")
    print(f"PII: {pii}  ")
    print(
        f"Diag_sum: {metrics['diag_sum']}  "
        f"V: {metrics['log_volume']}  "
        f"Rank: {metrics['rank']}  "
    )
    print("=" * 50)

    volume_file.write(f"{param_name}  ")
    volume_file.write(f"PII,{pii},")
    volume_file.write(
        f"sum_mean_var,{metrics['diag_sum']},{metrics['diag_mean']},"
        f"{metrics['diag_var']},log_V,{metrics['log_volume']},"
        f"Rank,{metrics['rank']},"
    )
    diag_file.write(
        f"{experiment_name},"
        + ",".join(f"{x:.5g}" for x in metrics["diagonal"])
        + ","
    )


def log_hidden_layers(
    model: nn.Module,
    experiment_name: str,
    volume_file,
    diag_file,
    num_hidden_layers: int,
) -> None:
    target_names = {f"linearH.{i}.weight" for i in range(num_hidden_layers)}
    for name, param in model.named_parameters():
        if name in target_names:
            log_layer_metrics(name, param, experiment_name, volume_file, diag_file)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def initialize_weights(model: nn.Module, init_type: str = "xavier") -> None:
    for module in model.modules():
        if not isinstance(module, nn.Linear):
            continue
        if init_type == "xavier":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "kaiming":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight, a=-0.1, b=0.1)
        elif init_type == "orthogonal":
            nn.init.orthogonal_(module.weight)
        elif init_type == "constant":
            nn.init.constant_(module.weight, 0.05)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class FeedforwardNeuralNetwork(nn.Module):
    """MLP: input -> ReLU -> (Linear + ReLU) x N -> output."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        num_hidden_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        self.r = nn.ReLU()
        self.linearH = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)]
        )
        self.out = nn.Linear(hidden_size, num_classes)
        # Kept for parameter parity with the original experiment script.
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.r(self.linear(x))
        for layer in self.linearH:
            x = self.r(layer(x))
        return self.out(x)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------
def build_dataloaders(data_root: str, batch_size: int):
    transform = transforms.ToTensor()
    train_set = datasets.MNIST(
        root=data_root, train=True, download=True, transform=transform
    )
    test_set = datasets.MNIST(
        root=data_root, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_set, train_loader, test_loader


@torch.no_grad()
def compute_accuracy(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for features, targets in data_loader:
        features = features.view(-1, 28 * 28).to(device)
        targets = targets.to(device)
        logits = model(features)
        predicted = logits.argmax(dim=1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()
    model.train()
    return correct / total * 100.0


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    num_epochs: int,
    dataset_len: int,
    batch_size: int,
) -> float:
    last_loss = 0.0
    for step, (images, labels) in enumerate(train_loader, start=1):
        images = images.view(-1, 28 * 28).to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if step % 40 == 0:
            print(
                f"Epoch: [{epoch + 1}/{num_epochs}], "
                f"Step: [{step}/{dataset_len // batch_size}], "
                f"Loss: {last_loss:.4f} "
            )
    return last_loss


def main():
    args = parse_args()
    set_seed(args.random_seed)
    device = get_device(args.gpu)

    result_dir = Path(args.result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = build_experiment_name(args)

    volume_path = result_dir / f"{experiment_name}_volume.txt"
    acc_path = result_dir / f"{experiment_name}_acc.txt"
    diag_path = result_dir / f"{experiment_name}_diag.txt"

    _, train_loader, test_loader = build_dataloaders(args.data_root, args.batch_size)

    model = FeedforwardNeuralNetwork(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        num_classes=10,
        num_hidden_layers=args.num_hidden_layers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    initialize_weights(model, init_type=args.init_method)

    with (
        open(volume_path, "w", encoding="utf-8") as volume_file,
        open(acc_path, "w", encoding="utf-8") as acc_file,
        open(diag_path, "w", encoding="utf-8") as diag_file,
    ):
        volume_file.write("epoch:0  ")
        diag_file.write("epoch:0  ")
        log_hidden_layers(
            model, experiment_name, volume_file, diag_file, args.num_hidden_layers
        )
        volume_file.write("\n")
        diag_file.write("\n")

        for epoch in range(args.num_epochs):
            loss_value = train_one_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                num_epochs=args.num_epochs,
                dataset_len=len(train_loader.dataset),
                batch_size=args.batch_size,
            )

            test_accuracy = round(
                compute_accuracy(model, test_loader, device), 4
            )
            loss_out = round(loss_value, 4)
            print(f"{test_accuracy}  {loss_out}")
            acc_file.write(f"{test_accuracy}  {loss_out}\n")

            if (epoch + 1) % args.epoch_interval == 0:
                volume_file.write(f"epoch:{epoch}  ")
                diag_file.write(f"epoch:{epoch}  ")
                log_hidden_layers(
                    model,
                    experiment_name,
                    volume_file,
                    diag_file,
                    args.num_hidden_layers,
                )
                volume_file.write("\n")
                diag_file.write("\n")


if __name__ == "__main__":
    main()
