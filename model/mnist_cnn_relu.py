"""Plaintext ReLU twin of the HE-friendly MNIST CNN (LitePI experiment, Phase 1).

The architecture is IDENTICAL to model/mnist_cnn_enc.py (and the poly2 twin);
the ONLY difference is the activation: square/poly2 -> ReLU. A ReLU network
cannot be evaluated homomorphically, so this script is a plaintext-only
baseline measuring the accuracy of the exact same architecture without the HE
activation constraint. It is the E1 "Plain-ReLU" row and the plaintext
reference model for the future Hybrid mode (ReLU twin + per-layer MPC ReLU).

Structure (unchanged from the square twin):
    input (1, 28, 28)
      -> Conv2d(1->4, 7x7, stride 3)      # 4 x 8 x 8
      -> ReLU
      -> flatten 256
      -> Linear(256 -> 64)
      -> ReLU
      -> Linear(64 -> 10)

Training config (same hyper-parameters / seed as the square twin):
    10 epochs / Adam / lr 1e-3 / batch 64 / seed 73
    acceptance: plaintext test accuracy >= 0.975

Artifacts (per 实验方案.md Phase 1):
    datas/mnist_relu_cnn_train.pt / .log   (seed 73 main checkpoint)

Multi-seed mode (Phase 1 acceptance: 5 seeds, mean +- std):
    JSON per seed -> results/raw_results/E1/mnist/plain_relu/seed{s}.json

Run:
    python model/mnist_cnn_relu.py                     # load datas/mnist_relu_cnn_train.pt if present
    python model/mnist_cnn_relu.py --train             # force retraining (writes pt + log)
    python model/mnist_cnn_relu.py --num-seeds 5       # seeds 73..77, mean+-std, JSON output
    python model/mnist_cnn_relu.py --seed 1 --num-seeds 5
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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
DATA_DIR = PROJECT_ROOT / "datas"
RAW_ROOT = PROJECT_ROOT / "results" / "raw_results" / "E1" / "mnist" / "plain_relu"

MODEL_PATH = DATA_DIR / "mnist_relu_cnn_train.pt"
LOG_SAVE_PATH = DATA_DIR / "mnist_relu_cnn_train.log"

# Same network geometry as the square twin (mnist_cnn_enc.py).
FLAT_SIZE = 256  # 4 channels * 8 * 8 (28x28 input, stride-3 7x7 kernel)
ACCEPT_ACC = 0.975  # Phase 1 acceptance threshold
METHOD = "plain_relu"  # results/method tag used in the raw-results tree


class ReluMnistNet(nn.Module):
    """MNIST CNN with ReLU activations (plaintext twin of the HE-friendly net).

    Identical structure to HeMnistNet in mnist_cnn_enc.py; only the activation
    differs (square -> ReLU), so the two can be compared directly.
    """

    def __init__(self, hidden: int = 64, output: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3)
        self.fc1 = nn.Linear(FLAT_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.relu(self.conv1(inputs))
        inputs = inputs.view(-1, FLAT_SIZE)
        inputs = torch.relu(self.fc1(inputs))
        return self.fc2(inputs)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        root=str(DATASET_DIR), train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=str(DATASET_DIR), train=False, download=True, transform=transform
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
) -> tuple[float, float, float]:
    """One pass; returns (loss, accuracy, correct)."""
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(is_training):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
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

    return total_loss / total, correct / total, correct


def train(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict[str, list[float]]:
    """Same optimizer recipe as the square twin: Adam, lr 1e-3, no weight decay."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        test_loss, test_acc, _ = run_epoch(model, test_loader, criterion, device)
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


def base_record(seed: int) -> dict:
    return {
        "exp_id": "E1",
        "dataset": "mnist",
        "method": METHOD,
        "seed": seed,
        "git_sha": get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def save_record(seed: int, extra: dict) -> Path:
    """Write {seed}.json under results/raw_results/E1/mnist/plain_relu/ (红线①)."""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    record = base_record(seed)
    record.update(extra)
    path = RAW_ROOT / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST ReLU twin (plaintext): identical architecture to the HE "
                    "square/poly2 net, only the activation differs (LitePI Phase 1)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
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
    """Train a fresh model with a fixed seed; returns (history, summary)."""
    torch.manual_seed(seed)
    start = time.perf_counter()
    history = train(model, train_loader, test_loader, device, epochs, lr)
    train_time_s = time.perf_counter() - start
    model.eval()
    summary = {
        "train_acc_final": history["train_accuracy"][-1],
        "test_acc_final": history["test_accuracy"][-1],
        "test_acc_best": max(history["test_accuracy"]),
        "train_time_s": round(train_time_s, 1),
    }
    return history, summary


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
        # Single-seed path (checkpointing, mirrors the _enc/_poly2 scripts).
        seed = args.seed
        torch.manual_seed(seed)
        model = ReluMnistNet().to(device)
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
            _, test_acc, _ = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
            summary = {
                "train_acc_final": None,
                "test_acc_final": test_acc,
                "test_acc_best": None,
                "train_time_s": None,
                "source": "loaded",
            }
            acceptance_report(test_acc, seed)

        accept = acceptance_report(summary["test_acc_final"], seed)
        record = {
            "hyperparams": {
                "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                "optimizer": "Adam", "weight_decay": 0.0, "scheduler": None,
            },
            "device": str(device),
            "accuracy": summary,
            "acceptance": accept,
            "client_metrics": "n/a - plaintext training only (Phase 1)",
            "source": summary.get("source", "retrained"),
        }
        path = save_record(seed, record)
        print(f"JSON -> {path}")
    else:
        # Multi-seed path: N fresh trainings, per-seed JSON + mean+-std (红线②).
        for seed in seeds:
            # Seed BEFORE constructing the model: otherwise the initial weights do not
            # depend on `seed` and the multi-seed run is neither independent nor reproducible.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = ReluMnistNet().to(device)
            _, summary = train_one(model, train_loader, test_loader, device,
                                   args.epochs, args.lr, seed)
            accept = acceptance_report(summary["test_acc_final"], seed)
            record = {
                "hyperparams": {
                    "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
                    "optimizer": "Adam", "weight_decay": 0.0, "scheduler": None,
                },
                "device": str(device),
                "accuracy": summary,
                "acceptance": accept,
                "client_metrics": "n/a - plaintext training only (Phase 1)",
                "source": "retrained",
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
