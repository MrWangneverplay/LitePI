"""Encrypted UCI HAR inference demo using TenSEAL (CKKS scheme).

Input: the raw Inertial Signals windows (128 time steps x 9 signal channels)
       of the classic "Human Activity Recognition Using Smartphones" dataset,
       NOT the 561 precomputed feature vectors.

Dataset download:
    1. Kaggle: https://www.kaggle.com/datasets/quanganh2001/uci-har-dataset
       (or UCI ML repo: https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones)
    2. Unzip so that this directory exists:
         datasets/UCI_HAR/UCI HAR Dataset/
           - train/Inertial Signals/{body_acc,total_acc,body_gyro}_{x,y,z}_train.txt
           - train/y_train.txt
           - test/Inertial Signals/{body_acc,total_acc,body_gyro}_{x,y,z}_test.txt
           - test/y_test.txt
    3. Run:
         python model/uci_har_cnn_enc.py                 # train if needed, then encrypt & predict
         python model/uci_har_cnn_enc.py --train         # force retraining
         python model/uci_har_cnn_enc.py --num-samples 5 # encrypt & predict 5 windows

HE-friendly network (only additions and multiplications are homomorphically
computable, so ReLU/MaxPool are replaced by the square activation):

    input  (1, 128, 9)  time=128 rows, channel=9 cols
      -> Conv2d(1->12, kernel=(3,9), stride=(2,1))      # cross-channel time conv, 63 windows
      -> x^2
      -> flatten 12*63 = 756
      -> Linear(756 -> 128) -> x^2
      -> Linear(128 -> 6)

Slot geometry check (CKKS slots = 4096 with poly_modulus_degree=8192):
    im2col windows      = (128-3)//2+1 = 63   (column direction has only 1 window)
    one window uses     = next_pow2(3*9=27) = 32 slots
    encoded length      = 63 * 32 = 2016 <= 4096  OK -> single ciphertext
    packed conv output  = 12 * 63 = 756 <= 4096   OK
    multiplicative depth = conv + square + fc + square + fc = 5 (7 primes -> 6 mults)

IMPORTANT: TenSEAL 0.3.x's python wrapper `ts.im2col_encoding` swaps
(kernel_n_rows, kernel_n_cols) before calling the C++ backend. For a
non-square 3x9 kernel this must be compensated in the call by passing (9, 3)
(see encrypted_predict); otherwise the C++ side builds 240 windows and the
encoded vector (7680 slots) no longer fits a single ciphertext, disabling
conv2d_im2col / matmul.

Multiplicative depth 5 matches the MNIST/CIFAR demos, so the exact same CKKS
parameter set is reused.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
DATA_BASE = DATASET_DIR / "UCI_HAR" / "UCI HAR Dataset"
RESULTS_DIR = PROJECT_ROOT / "datas"
HE_MODEL_PATH = RESULTS_DIR / "uci_har_he_cnn_train.pt"
LOG_SAVE_PATH = RESULTS_DIR / "uci_har_he_cnn_train.log"
STATS_PATH = RESULTS_DIR / "uci_har_he_stats.npz"

# CKKS parameters (same as the MNIST/CIFAR demos): 7 primes -> 6 mults deep.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 21, 21, 21, 21, 40]
GLOBAL_SCALE = 2 ** 21

# Network geometry for a (128 time x 9 channel) window.
TIME_SIZE = 128
N_SIGNALS = 9
KERNEL_SHAPE = (3, N_SIGNALS)  # kernel width covers ALL channels: cross-channel time conv
STRIDE = 2
OUT_SIZE = (TIME_SIZE - KERNEL_SHAPE[0]) // STRIDE + 1  # 63 time positions
WINDOWS_NB = OUT_SIZE  # 63 im2col windows (column direction collapses to 1)
N_CHANNELS = 12        # fallback widens to 16 if accuracy < 70%
HIDDEN = 128           # fallback widens to 192 if accuracy < 70%
OUTPUT = 6             # WALKING ... LAYING
FLAT_SIZE = N_CHANNELS * WINDOWS_NB  # 756
DROP_PROB = 0.3        # plaintext-training-only dropout between the FC layers
MIN_TEST_ACC = 0.70    # acceptance threshold requested by the user

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
    """Per-channel mean/std (9 scalars), cached to npz after the first run."""
    if STATS_PATH.exists():
        cached = np.load(STATS_PATH)
        return cached["mean"].astype(np.float32), cached["std"].astype(np.float32)
    mean = x_train.mean(axis=(0, 1))
    std = np.maximum(x_train.std(axis=(0, 1)), 1e-8)
    np.savez(STATS_PATH, mean=mean.astype(np.float32), std=std.astype(np.float32))
    return mean.astype(np.float32), std.astype(np.float32)


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Per-channel z-score. Applied in the plaintext domain (client side), so the
    plaintext model and the encrypted pipeline see bit-identical inputs."""
    return ((x - mean) / std).astype(np.float32)


def load_data(
    batch_size: int,
) -> tuple[DataLoader, DataLoader, TensorDataset, np.ndarray, np.ndarray]:
    x_train, y_train = load_raw_windows("train")
    x_test, y_test = load_raw_windows("test")
    mean, std = get_channel_stats(x_train)
    x_train_n, x_test_n = normalize(x_train, mean, std), normalize(x_test, mean, std)
    train_ds = TensorDataset(torch.from_numpy(x_train_n), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(x_test_n), torch.from_numpy(y_test))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        test_ds,
        mean,
        std,
    )


class HeUciHarNet(nn.Module):
    """HE-friendly UCI HAR net: Conv -> Square -> FC -> Square -> FC.

    The square activation replaces ReLU/MaxPool, which cannot be evaluated
    exactly on ciphertexts; the whole model stays 5 multiplications deep.
    Dropout is only active while training (plaintext) and disappears in
    eval mode, so it does not affect the encrypted forward pass.
    """

    def __init__(self, channels: int = N_CHANNELS, hidden: int = HIDDEN) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=KERNEL_SHAPE, stride=(STRIDE, 1))
        self.fc1 = nn.Linear(channels * WINDOWS_NB, hidden)
        self.drop = nn.Dropout(p=DROP_PROB)
        self.fc2 = nn.Linear(hidden, OUTPUT)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)          # (B, C, 63, 1)
        inputs = inputs * inputs
        inputs = inputs.view(-1, inputs.shape[1] * WINDOWS_NB)
        inputs = self.fc1(inputs)
        inputs = inputs * inputs
        inputs = self.drop(inputs)
        return self.fc2(inputs)


class EncConvNet:
    """PyTorch-like model running on ciphertexts with TenSEAL operations.

    - .conv2d_im2col() runs one convolution on an im2col-encoded ciphertext.
    - .square_() squares the encrypted vector in place (activation).
    - .mm() vector-matrix multiplication (Halevi-Shoup diagonal method).
    - `+ bias` adds a plaintext vector to the ciphertext.
    """

    def __init__(self, torch_nn: HeUciHarNet) -> None:
        channels = torch_nn.conv1.out_channels
        self.channels = channels
        self.conv1_weight = torch_nn.conv1.weight.data.view(channels, *KERNEL_SHAPE).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()  # (C*63, hidden)
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()  # (hidden, 6)
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x: ts.CKKSVector, windows_nb: int) -> ts.CKKSVector:
        channels = [
            enc_x.conv2d_im2col(kernel, windows_nb) + bias
            for kernel, bias in zip(self.conv1_weight, self.conv1_bias)
        ]
        # pack the channel outputs (C x 63) into one flattened ciphertext
        enc_x = ts.CKKSVector.pack_vectors(channels)
        enc_x.square_()
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x.square_()
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs) -> ts.CKKSVector:
        return self.forward(*args, **kwargs)


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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(test_accuracy)
        line = (
            f"Epoch {epoch:2d}/{epochs}  "
            f"Train Loss: {train_loss:.6f}  "
            f"Test Loss: {test_loss:.6f}  Test Acc: {test_accuracy:.4f}"
        )
        print(line)
        history.setdefault("lines", []).append(line)

    return history


def create_context() -> ts.Context:
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    context.global_scale = GLOBAL_SCALE
    context.generate_galois_keys()  # required by pack_vectors / mm / conv2d_im2col
    return context


def encrypted_predict(
    context: ts.Context, enc_model: EncConvNet, window: torch.Tensor
) -> list[float]:
    """Encode + encrypt one normalized (128, 9) window, run the encrypted forward pass."""
    # NOTE: TenSEAL 0.3.x `ts.im2col_encoding` swaps (kernel_n_rows,
    # kernel_n_cols) before calling the C++ backend (square kernels in the
    # MNIST/CIFAR demos hide this quirk). Passing (9, 3) therefore produces a
    # 3(time) x 9(channel) kernel window: (128-3)//2+1 = 63 windows, each padded
    # to 32 slots -> 63*32 = 2016 <= 4096 slots, i.e. a single ciphertext that
    # keeps conv2d_im2col / mm available.
    enc_x, windows_nb = ts.im2col_encoding(
        context, window.numpy().tolist(), KERNEL_SHAPE[1], KERNEL_SHAPE[0], STRIDE
    )
    enc_output = enc_model(enc_x, windows_nb)
    return enc_output.decrypt()


def _save_checkpoint(path: Path, model: nn.Module) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "channels": model.conv1.out_channels,
            "hidden": model.fc1.out_features,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypted UCI HAR inference with TenSEAL (raw Inertial Signals)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Force retraining")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Windows to encrypt & predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(73)
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        train_loader, test_loader, test_ds, mean, std = load_data(args.batch_size)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(f"Using device: {device}  (train={len(train_loader.dataset)}, "
          f"test={len(test_loader.dataset)} samples, per-channel stats cached)")

    # 1) Train or load the HE-friendly model ---------------------------------
    model = HeUciHarNet().to(device)
    needs_train = args.train or not HE_MODEL_PATH.exists()
    if not needs_train:
        try:
            ckpt = torch.load(HE_MODEL_PATH, map_location=device)
            model = HeUciHarNet(ckpt["channels"], ckpt["hidden"]).to(device)
            model.load_state_dict(ckpt["state_dict"])
            print(f"Loaded weights from: {HE_MODEL_PATH}")
        except (RuntimeError, KeyError, ValueError) as exc:  # e.g. old architecture
            print(f"Stored weights incompatible ({exc}); retraining...")
            needs_train = True

    if needs_train:
        history = train(model, train_loader, test_loader, device, args.epochs, args.lr)
        _save_checkpoint(HE_MODEL_PATH, model)
        _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"Model weights saved to: {HE_MODEL_PATH}")
        log_lines = [f"Using device: {device}", *history["lines"]]

        # Acceptance: plaintext test accuracy must be > 70% (fallback widens the net)
        if test_acc <= MIN_TEST_ACC:
            print(f"Test accuracy {test_acc:.4f} <= {MIN_TEST_ACC:.2f}: widening "
                  f"channels 12->16, hidden 128->192, epochs {args.epochs}->{max(60, args.epochs)} and retraining...")
            log_lines.append(
                f"Test accuracy {test_acc:.4f} below {MIN_TEST_ACC:.2f} threshold; retrying with "
                f"channels=16, hidden=192, epochs={max(60, args.epochs)}"
            )
            fallback_epochs = max(60, args.epochs)
            model = HeUciHarNet(channels=16, hidden=192).to(device)
            history = train(model, train_loader, test_loader, device, fallback_epochs, args.lr)
            _save_checkpoint(HE_MODEL_PATH, model)
            _, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
            log_lines.append(f"Fallback retrained; final Test Acc: {test_acc:.4f}")
            print(f"Fallback weights saved to: {HE_MODEL_PATH}")
        else:
            log_lines.append(f"Acceptance passed: plaintext test accuracy {test_acc:.4f} > {MIN_TEST_ACC:.2f}")

        LOG_SAVE_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    model.eval()

    # 2) Plaintext reference accuracy ----------------------------------------
    _, plain_accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    passed = plain_accuracy > MIN_TEST_ACC
    print(f"Plaintext model test accuracy: {plain_accuracy:.4f} "
          f"({'PASS > ' + str(MIN_TEST_ACC) if passed else 'BELOW ' + str(MIN_TEST_ACC) + ' threshold'})")

    # 3) Encrypted inference demo ---------------------------------------------
    context = create_context()
    enc_model = EncConvNet(model)

    print("\n--- Encrypted inference (client encrypts, server runs on ciphertexts) ---")
    indices = random.sample(range(len(test_ds)), args.num_samples)
    agree = 0
    for index in indices:
        window, label = test_ds[index]  # normalized (128, 9) window

        # plaintext reference prediction
        with torch.no_grad():
            plain_logits = model(window.unsqueeze(0).unsqueeze(0).to(device)).cpu()
        plain_pred = int(plain_logits.argmax(dim=1).item())

        # encrypted prediction (weights stay plaintext on the server side)
        start = time.perf_counter()
        enc_logits = encrypted_predict(context, enc_model, window)
        elapsed = time.perf_counter() - start
        enc_pred = int(max(range(OUTPUT), key=lambda i: enc_logits[i]))

        match = "OK" if plain_pred == enc_pred else "MISMATCH"
        agree += plain_pred == enc_pred
        print(
            f"sample #{index:5d}  label={label}  plaintext={plain_pred}  "
            f"encrypted={enc_pred}  [{match}]  ({elapsed:.1f}s)"
        )

    print(f"\nPlaintext vs encrypted agreement: {agree}/{args.num_samples}")
    print(f"Context secrets kept local, only public parts would be shared: "
          f"{context.is_private()}")


if __name__ == "__main__":
    main()
