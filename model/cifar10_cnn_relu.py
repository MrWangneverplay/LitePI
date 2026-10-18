"""Plaintext ReLU twin of the HE-friendly CIFAR-10 CNN (LitePI, Phase 1).

Route A design is UNCHANGED from model/cifar10_cnn_enc.py (grayscale): the
3-channel RGB image is collapsed to one grayscale channel by a fixed linear
combination (ITU-R BT.601 luma weights) *in the plaintext domain*, then
zero-mean/unit-variance normalized with the same fixed statistics. The ONLY
difference from the square/poly2 twin is the activation: square/poly2 -> ReLU.
ReLU cannot be evaluated homomorphically, so this is a plaintext-only baseline
for the E1 "Plain-ReLU" row and the plaintext reference for the future Hybrid
mode (ReLU twin + per-layer MPC ReLU).

Structure (unchanged from the square twin):
    RGB(3x32x32) -> gray + z-score (plaintext) -> (1, 32, 32)
      -> Conv(1->12, 5x5, s=3)      # 12 x 10 x 10 = 1200
      -> ReLU
      -> Linear(1200 -> 128)
      -> ReLU
      -> Linear(128 -> 10)

Training config (same as the square twin):
    40 epochs / AdamW (wd 1e-4) + CosineAnnealingLR / lr 1e-3 / batch 128 / seed 73
    acceptance: plaintext test accuracy >= 0.63

Artifacts (per 实验方案.md Phase 1):
    datas/cifar10_relu_cnn_train.pt / .log    (seed 73 main checkpoint)

Multi-seed mode (Phase 1 acceptance: 5 seeds, mean +- std):
    JSON per seed -> results/raw_results/E1/cifar10/plain_relu/seed{s}.json

Run:
    python model/cifar10_cnn_relu.py                   # load checkpoint if present
    python model/cifar10_cnn_relu.py --train           # force retraining (writes pt + log)
    python model/cifar10_cnn_relu.py --num-seeds 5     # seeds 73..77, mean+-std, JSON output
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "datas"
RAW_ROOT = PROJECT_ROOT / "results" / "raw_results" / "E1" / "cifar10" / "plain_relu"

MODEL_PATH = DATA_DIR / "cifar10_relu_cnn_train.pt"
LOG_SAVE_PATH = DATA_DIR / "cifar10_relu_cnn_train.log"

# Network geometry for the 32x32 grayscale input (same as the square twin).
IMG_SIZE = 32
KERNEL_SIZE = 5  # (32-5)/3+1 = 10 -> 10x10 windows
STRIDE = 3
WINDOWS_NB = 100  # im2col windows (10x10)
N_CHANNELS = 12
FLAT_SIZE = N_CHANNELS * WINDOWS_NB  # 1200, flattened conv output size
HIDDEN = 128
ACCEPT_ACC = 0.63  # Phase 1 acceptance threshold
METHOD = "plain_relu"

# ITU-R BT.601 luma weights: RGB -> gray in the plaintext domain.
GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)
GRAY_MEAN = 0.2989 * 0.4914 + 0.5870 * 0.4822 + 0.1140 * 0.4465   # ~0.4808
GRAY_STD = 0.2989 * 0.2470 + 0.5870 * 0.2435 + 0.1140 * 0.2616    # ~0.2465


def to_gray(batch: torch.Tensor) -> torch.Tensor:
    """Collapse an (B, 3, H, W) RGB batch to (B, 1, H, W) grayscale (plaintext side)."""
    r, g, b = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3]
    return GRAY_WEIGHTS[0] * r + GRAY_WEIGHTS[1] * g + GRAY_WEIGHTS[2] * b


def preprocess(batch: torch.Tensor) -> torch.Tensor:
    """Full plaintext-side input pipeline: RGB -> gray -> z-score (fixed stats)."""
    return (to_gray(batch) - GRAY_MEAN) / GRAY_STD


class ReluCifarNet(nn.Module):
    """CIFAR-10 CNN (grayscale) with ReLU activations (plaintext twin of the
    HE-friendly net). Identical structure to HeCifarNet, only ReLU instead of
    square/poly2, so the two can be compared directly."""

    def __init__(self, hidden: int = HIDDEN, output: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, N_CHANNELS, kernel_size=KERNEL_SIZE, stride=STRIDE)
        self.fc1 = nn.Linear(FLAT_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.relu(self.conv1(inputs))
        inputs = inputs.view(-1, FLAT_SIZE)
        inputs = torch.relu(self.fc1(inputs))
        return self.fc2(inputs)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size: int, num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    # Augmentation happens on RGB in the PIL domain; grayscale collapse and
    # normalization are done inside preprocess() so both sides always agree.
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(IMG_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR), train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=str(DATASET_DIR), train=False, download=True, transform=test_transform
    )
    pin = torch.cuda.is_available()
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=pin),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=pin),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for inputs, targets in loader:
            inputs, targets = preprocess(inputs).to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    return total_loss / total, correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict[str, list[float]]:
    """Same optimizer recipe as the square twin: AdamW (wd 1e-4) + CosineAnnealing."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_acc)
        line = (
            f"Epoch {epoch:2d}/{epochs}  Train Loss: {train_loss:.6f}  "
            f"Train Acc: {train_acc:.4f}  Test Loss: {test_loss:.6f}  "
            f"Test Acc: {test_acc:.4f}"
        )
        print(line)
        history.setdefault("lines", []).append(line)

    return history


def get_git_sha() -> str | None:
    """git rev-parse HEAD for result provenance (实验方案.md 红线①)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def save_record(seed: int, extra: dict) -> Path:
    """Write {seed}.json under results/raw_results/E1/cifar10/plain_relu/ (红线①)."""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "exp_id": "E1",
        "dataset": "cifar10",
        "method": METHOD,
        "seed": seed,
        "git_sha": get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(extra)
    path = RAW_ROOT / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 (grayscale) ReLU twin (plaintext): identical architecture "
                    "to the HE square/poly2 net, only the activation differs (LitePI Phase 1)"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=73, help="First training seed")
    parser.add_argument("--num-seeds", type=int, default=1,
                        help="Train N consecutive seeds (73..73+N-1): mean+-std + per-seed JSON")
    parser.add_argument("--train", action="store_true",
                        help="Force retraining (single-seed mode only; writes pt + log)")
    return parser.parse_args()


def train_one(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    seed: int,
) -> tuple[dict, dict]:
    """Train a fresh model with a fixed seed; returns (history, accuracy summary)."""
    torch.manual_seed(seed)
    start = time.perf_counter()
    history = train(model, train_loader, test_loader, device, epochs, lr)
    train_time_s = time.perf_counter() - start
    model.eval()
    return history, {
        "train_acc_final": history["train_accuracy"][-1],
        "test_acc_final": history["test_accuracy"][-1],
        "test_acc_best": max(history["test_accuracy"]),
        "train_time_s": round(train_time_s, 1),
    }


def acceptance_report(acc: float, seed: int) -> dict:
    passed = acc >= ACCEPT_ACC
    print(f"[seed {seed}] Plaintext ReLU test accuracy: {acc:.4f} "
          f"(acceptance >= {ACCEPT_ACC:.4f}: {'PASS' if passed else 'FAIL'})")
    return {"threshold": ACCEPT_ACC, "passed": passed}


def main() -> None:
    args = parse_args()
    device = get_device()
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = load_data(args.batch_size)
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    print(f"Using device: {device}  seeds={seeds}  "
          f"train={len(train_loader.dataset)}  test={len(test_loader.dataset)}")

    results = []
    if args.num_seeds == 1:
        seed = args.seed
        torch.manual_seed(seed)
        model = ReluCifarNet().to(device)
        needs_train = args.train or not MODEL_PATH.exists()
        if not needs_train:
            try:
                model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
                print(f"Loaded weights from: {MODEL_PATH}")
            except (RuntimeError, KeyError) as exc:
                print(f"Stored weights incompatible with current architecture ({exc}); "
                      f"retraining...")
                needs_train = True

        if needs_train:
            history, summary = train_one(model, train_loader, test_loader, device,
                                         args.epochs, args.lr, seed)
            torch.save(model.state_dict(), MODEL_PATH)
            lines = [f"Using device: {device}", f"Activation: ReLU", *history["lines"]]
            LOG_SAVE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"Model weights saved to: {MODEL_PATH}")
        else:
            model.eval()
            _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
            summary = {
                "train_acc_final": None,
                "test_acc_final": test_acc,
                "test_acc_best": None,
                "train_time_s": None,
            }
            acceptance_report(test_acc, seed)

        accept = acceptance_report(summary["test_acc_final"], seed)
        record = {
            "hyperparams": {
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "optimizer": "AdamW", "weight_decay": 1e-4, "scheduler": "CosineAnnealingLR",
            },
            "device": str(device),
            "accuracy": summary,
            "acceptance": accept,
            "client_metrics": "n/a - plaintext training only (Phase 1)",
        }
        path = save_record(seed, record)
        print(f"JSON -> {path}")
        results.append(summary["test_acc_final"])
    else:
        for seed in seeds:
            # Seed BEFORE constructing the model: otherwise the multi-seed run
            # shares one initialization and the reported std is meaningless.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = ReluCifarNet().to(device)
            _, summary = train_one(model, train_loader, test_loader, device,
                                   args.epochs, args.lr, seed)
            accept = acceptance_report(summary["test_acc_final"], seed)
            record = {
                "hyperparams": {
                    "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "optimizer": "AdamW", "weight_decay": 1e-4,
                    "scheduler": "CosineAnnealingLR",
                },
                "device": str(device),
                "accuracy": summary,
                "acceptance": accept,
                "client_metrics": "n/a - plaintext training only (Phase 1)",
            }
            path = save_record(seed, record)
            print(f"JSON -> {path}")
            results.append(summary["test_acc_final"])

    accs = torch.tensor(results)
    mean, std = float(accs.mean()), float(accs.std(unbiased=True))
    print(f"\n=== seeds {seeds[0]}..{seeds[-1]}  test acc "
          f"mean={mean:.4f}  std={std:.4f}  (n={len(seeds)}) ===")


if __name__ == "__main__":
    main()
