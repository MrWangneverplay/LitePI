"""Plaintext ReLU twin of the HE-friendly UCI HAR CNN (LitePI, Phase 1).

Input/design are UNCHANGED from model/uci_har_cnn_enc.py: raw Inertial Signals
windows (128 time steps x 9 signal channels, NOT the 561 precomputed features),
per-channel z-score normalization with statistics cached in
datas/uci_har_he_stats.npz (shared with the HE twins). The ONLY difference
from the square/poly2 twin is the activation: square/poly2 -> ReLU. ReLU
cannot be evaluated homomorphically, so this is a plaintext-only baseline for
the E1 "Plain-ReLU" row and the plaintext reference for the future Hybrid mode
(ReLU twin + per-layer MPC ReLU).

Structure (unchanged from the square twin):
    input (1, 128, 9)
      -> Conv2d(1->12, kernel=(3, 9), stride=(2, 1))    # 12 x 63 x 1
      -> ReLU
      -> flatten 756
      -> Linear(756 -> 128) -> ReLU -> Dropout(0.3, training only)
      -> Linear(128 -> 6)

Training config (same as the square twin):
    30 epochs / AdamW (wd 1e-4) + CosineAnnealingLR / lr 1e-3 / batch 128 / seed 73
    acceptance: plaintext test accuracy >= 0.89

Artifacts (per 实验方案.md Phase 1):
    datas/uci_har_relu_cnn_train.pt / .log    (seed 73 main checkpoint)

Multi-seed mode (Phase 1 acceptance: 5 seeds, mean +- std):
    JSON per seed -> results/raw_results/E1/uci_har/plain_relu/seed{s}.json

Run:
    python model/uci_har_cnn_relu.py                   # load checkpoint if present
    python model/uci_har_cnn_relu.py --train           # force retraining (writes pt + log)
    python model/uci_har_cnn_relu.py --num-seeds 5     # seeds 73..77, mean+-std, JSON output
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
DATA_BASE = DATASET_DIR / "UCI_HAR" / "UCI HAR Dataset"
DATA_DIR = PROJECT_ROOT / "datas"
RAW_ROOT = PROJECT_ROOT / "results" / "raw_results" / "E1" / "uci_har" / "plain_relu"

MODEL_PATH = DATA_DIR / "uci_har_relu_cnn_train.pt"
LOG_SAVE_PATH = DATA_DIR / "uci_har_relu_cnn_train.log"
STATS_PATH = DATA_DIR / "uci_har_he_stats.npz"  # shared with the HE twins (方案: 加载此统计)

# Network geometry for a (128 time x 9 channel) window (same as the square twin).
TIME_SIZE = 128
N_SIGNALS = 9
KERNEL_SHAPE = (3, N_SIGNALS)  # kernel width covers ALL channels: cross-channel time conv
STRIDE = 2
WINDOWS_NB = (TIME_SIZE - KERNEL_SHAPE[0]) // STRIDE + 1  # 63 time positions
N_CHANNELS = 12        # fallback widens to 16 if accuracy < 70%
HIDDEN = 128           # fallback widens to 192 if accuracy < 70%
OUTPUT = 6             # WALKING ... LAYING
DROP_PROB = 0.3        # plaintext-training-only dropout between the FC layers
FALLBACK_THRESHOLD = 0.70  # inherited from _enc.py: widen the net if far below target
ACCEPT_ACC = 0.89      # Phase 1 acceptance threshold
METHOD = "plain_relu"

SIGNAL_NAMES = [
    "body_acc_x", "body_acc_y", "body_acc_z",
    "body_gyro_x", "body_gyro_y", "body_gyro_z",
    "total_acc_x", "total_acc_y", "total_acc_z",
]


def _missing_message() -> str:
    return (
        "UCI HAR Inertial Signals not found under:\n"
        f"  {DATA_BASE}\n"
        "Please download the dataset and unzip it there:\n"
        "  Kaggle : https://www.kaggle.com/datasets/quanganh2001/uci-har-dataset\n"
        "  UCI ML : https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones\n"
        "Expected layout:\n"
        f"  {DATA_BASE / 'train' / 'Inertial Signals' / 'body_acc_x_train.txt'}  (+8 more signals)\n"
        f"  {DATA_BASE / 'test' / 'Inertial Signals' / 'body_acc_x_test.txt'}   (+8 more signals)"
    )


def load_raw_windows(split: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one split: 9 signal txt files (n x 128) -> X (n, 128, 9), labels 0..5."""
    split_dir = DATA_BASE / split
    signal_files = [
        split_dir / "Inertial Signals" / f"{name}_{split}.txt" for name in SIGNAL_NAMES
    ]
    if not split_dir.exists() or not all(f.exists() for f in signal_files):
        raise FileNotFoundError(_missing_message())
    signals = [np.loadtxt(f, dtype=np.float32) for f in signal_files]
    n = signals[0].shape[0]
    x = np.stack(signals, axis=-1)  # (n, 128, 9)
    y = np.loadtxt(split_dir / f"y_{split}.txt", dtype=np.int64) - 1  # 1..6 -> 0..5
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"signal rows ({x.shape[0]}) != label rows ({y.shape[0]}) in {split}")
    return x, y


def get_channel_stats(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std (9 scalars), shared with the HE twins via the npz cache."""
    if STATS_PATH.exists():
        cached = np.load(STATS_PATH)
        return cached["mean"].astype(np.float32), cached["std"].astype(np.float32)
    mean = x_train.mean(axis=(0, 1))
    std = np.maximum(x_train.std(axis=(0, 1)), 1e-8)
    np.savez(STATS_PATH, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score, applied in the plaintext domain."""
    return ((x - mean) / std).astype(np.float32)


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    x_train, y_train = load_raw_windows("train")
    x_test, y_test = load_raw_windows("test")
    mean, std = get_channel_stats(x_train)
    x_train_n, x_test_n = normalize(x_train, mean, std), normalize(x_test, mean, std)
    train_ds = TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test_n), torch.from_numpy(y_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )


class ReluUciHarNet(nn.Module):
    """UCI HAR net with ReLU activations (plaintext twin of the HE-friendly net).

    Identical structure to HeUciHarNet (channels/hidden geometry included), only
    ReLU instead of square/poly2. Dropout is active while training only and
    disappears in eval mode.
    """

    def __init__(self, channels: int = N_CHANNELS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=KERNEL_SHAPE, stride=(STRIDE, 1))
        self.fc1 = nn.Linear(channels * WINDOWS_NB, hidden)
        self.drop = nn.Dropout(p=DROP_PROB)
        self.fc2 = nn.Linear(hidden, OUTPUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = torch.relu(self.conv1(inputs))       # (B, C, 63, 1)
        inputs = inputs.view(-1, inputs.shape[1] * WINDOWS_NB)
        inputs = torch.relu(self.fc1(inputs))
        inputs = self.drop(inputs)
        return self.fc2(inputs)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            # inputs: (B, 128, 9) -> model expects (B, 1, 128, 9)
            inputs, targets = inputs.unsqueeze(1).to(device), targets.to(device)
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
    """Write {seed}.json under results/raw_results/E1/uci_har/plain_relu/ (红线①)."""
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    record = {
        "exp_id": "E1",
        "dataset": "uci_har",
        "method": METHOD,
        "seed": seed,
        "git_sha": get_git_sha(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(extra)
    path = RAW_ROOT / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _save_checkpoint(path: Path, model: nn.Module) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "channels": model.conv1.out_channels,
            "hidden": model.fc1.out_features,
            "activation": "relu",
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UCI HAR ReLU twin (plaintext): identical architecture to the HE "
                    "square/poly2 net, only the activation differs (LitePI Phase 1)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
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

    try:
        train_loader, test_loader = load_data(args.batch_size)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    print(f"Using device: {device}  seeds={seeds}  "
          f"train={len(train_loader.dataset)}  test={len(test_loader.dataset)}  "
          f"(per-channel stats: {STATS_PATH})")

    results = []
    if args.num_seeds == 1:
        # Single-seed path (checkpointing, mirrors the _enc/_poly2 scripts).
        seed = args.seed
        torch.manual_seed(seed)
        model = ReluUciHarNet().to(device)
        needs_train = args.train or not MODEL_PATH.exists()
        if not needs_train:
            try:
                ckpt = torch.load(MODEL_PATH, map_location=device)
                model = ReluUciHarNet(ckpt["channels"], ckpt["hidden"]).to(device)
                model.load_state_dict(ckpt["state_dict"])
                print(f"Loaded weights from: {MODEL_PATH}")
            except (RuntimeError, KeyError, ValueError) as exc:
                print(f"Stored weights incompatible ({exc}); retraining...")
                needs_train = True

        if needs_train:
            history, summary = train_one(model, train_loader, test_loader, device,
                                         args.epochs, args.lr, seed)
            _save_checkpoint(MODEL_PATH, model)
            _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
            print(f"Model weights saved to: {MODEL_PATH}")

            # Acceptance fallback inherited from _enc.py: if far below target,
            # widen channels 12->16 / hidden 128->192 and retrain.
            log_lines = [f"Using device: {device}", f"Activation: ReLU", *history["lines"]]
            if test_acc <= FALLBACK_THRESHOLD:
                print(f"Test accuracy {test_acc:.4f} <= {FALLBACK_THRESHOLD:.2f}: widening "
                      f"channels 12->16, hidden 128->192, epochs {args.epochs}->{max(60, args.epochs)} "
                      f"and retraining...")
                log_lines.append(
                    f"Test accuracy {test_acc:.4f} below {FALLBACK_THRESHOLD:.2f}; retrying with "
                    f"channels=16, hidden=192, epochs={max(60, args.epochs)}"
                )
                fallback_epochs = max(60, args.epochs)
                model = ReluUciHarNet(channels=16, hidden=192).to(device)
                history, summary = train_one(model, train_loader, test_loader, device,
                                             fallback_epochs, args.lr, seed)
                _save_checkpoint(MODEL_PATH, model)
                _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
                summary["test_acc_final"] = test_acc
                summary["test_acc_best"] = max(summary["test_acc_best"], test_acc)
                log_lines.append(f"Fallback retrained; final Test Acc: {test_acc:.4f}")
                print(f"Fallback weights saved to: {MODEL_PATH}")
            LOG_SAVE_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
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
            "architecture": f"conv({model.conv1.out_channels}ch)->relu->fc({model.fc1.out_features})->relu->fc({OUTPUT})",
            "accuracy": summary,
            "acceptance": accept,
            "client_metrics": "n/a - plaintext training only (Phase 1)",
        }
        path = save_record(seed, record)
        print(f"JSON -> {path}")
        results.append(summary["test_acc_final"])
    else:
        # Multi-seed path: N fresh trainings, per-seed JSON + mean+-std (红线②).
        for seed in seeds:
            # Seed BEFORE constructing the model: otherwise the initial weights do not
            # depend on `seed` and the multi-seed run is neither independent nor reproducible.
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            model = ReluUciHarNet().to(device)
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
                "architecture": f"conv({model.conv1.out_channels}ch)->relu->fc({model.fc1.out_features})->relu->fc({OUTPUT})",
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
