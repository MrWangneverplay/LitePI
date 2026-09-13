"""Encrypted CIFAR-10 inference demo using TenSEAL (CKKS) - Route A (grayscale).

Route A design:
    The 3-channel RGB image is collapsed to ONE grayscale channel by a fixed
    linear combination (ITU-R BT.601 luma weights, equivalent to a fixed 1x1x3
    convolution) *in the plaintext domain* before encryption, then normalized
    to zero-mean/unit-variance (same statistics, so both plaintext training and
    encrypted inference see identical inputs). The HE network is:

        RGB(3x32x32) --to_gray + normalize (plaintext)--> Gray(1x32x32)
          -> im2col_encoding(kernel 5x5, stride 3) -> 10x10 = 100 windows, one ciphertext
          -> Conv(1->12, 5x5, s=3)   -> 12 channels x 10x10 = 1200
          -> square activation        (ReLU/MaxPool are not HE-computable)
          -> FC(1200 -> 128)
          -> square activation
          -> FC(128 -> 10)

Why v2 (vs v1: 8x8/s4, 8 ch, hidden 64):
    - v1 collapsed 32x32 straight to 7x7 with a huge 8x8 stride-4 kernel,
      throwing away most spatial detail -> only ~48% accuracy after 20 epochs
      while still underfitting. v2 uses a finer 5x5 stride-3 kernel (10x10 map,
      2x more resolution) and wider layers (12 ch, hidden 128).
    - Input is normalized (gray mean/std) to help convergence.
    - Multiplicative depth is unchanged at 5, so the CKKS parameters below
      (7 primes -> 6 mults) still fit.

Run:
    python model/cifar10_cnn_enc.py                  # train if needed, then encrypt & predict
    python model/cifar10_cnn_enc.py --train          # force retraining
    python model/cifar10_cnn_enc.py --num-samples 5  # encrypt & predict 5 images
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "datas"
HE_MODEL_PATH = RESULTS_DIR / "cifar10_he_cnn_train.pt"
LOG_SAVE_PATH = RESULTS_DIR / "cifar10_he_cnn_train.log"

# CKKS parameters (same as the MNIST demo): 7 primes -> 6 multiplications deep.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 21, 21, 21, 21, 40]
GLOBAL_SCALE = 2 ** 21

# Network geometry for the 32x32 grayscale input.
IMG_SIZE = 32
KERNEL_SHAPE = (5, 5)                       # (32-5)/3+1 = 10 -> 10x10 windows
STRIDE = 3
OUT_SIZE = (IMG_SIZE - KERNEL_SHAPE[0]) // STRIDE + 1   # 10
WINDOWS_NB = OUT_SIZE * OUT_SIZE            # 100 im2col windows
N_CHANNELS = 12                             # 12 x 100 = 1200 slots after packing
FLAT_SIZE = N_CHANNELS * WINDOWS_NB         # 1200, flattened conv output size
HIDDEN = 128                                # hidden width of fc1/fc2

# ITU-R BT.601 luma weights: RGB -> gray in the plaintext domain.
# Equivalent to a fixed 1x1x3 convolution (no encryption involved).
GRAY_WEIGHTS = (0.2989, 0.5870, 0.1140)
# Grayscale statistics of CIFAR-10 (luma-weighted RGB mean/std, values in 0..1).
GRAY_MEAN = 0.2989 * 0.4914 + 0.5870 * 0.4822 + 0.1140 * 0.4465   # ~0.4808
GRAY_STD = 0.2989 * 0.2470 + 0.5870 * 0.2435 + 0.1140 * 0.2616    # ~0.2465


def to_gray(batch: torch.Tensor) -> torch.Tensor:
    """Collapse an (B, 3, H, W) RGB batch to (B, 1, H, W) grayscale.

    Applied in the plaintext domain, before encryption, so both the plaintext
    model and the encrypted pipeline see exactly the same grayscale input.
    """
    r, g, b = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3]
    return GRAY_WEIGHTS[0] * r + GRAY_WEIGHTS[1] * g + GRAY_WEIGHTS[2] * b


def normalize_gray(batch: torch.Tensor) -> torch.Tensor:
    """Zero-mean / unit-variance normalization on the grayscale input.

    Also plaintext-side preprocessing. Using fixed dataset statistics keeps the
    plaintext and encrypted forward passes bit-consistent.
    """
    return (batch - GRAY_MEAN) / GRAY_STD


def preprocess(batch: torch.Tensor) -> torch.Tensor:
    """Full plaintext-side input pipeline: RGB -> gray -> normalize."""
    return normalize_gray(to_gray(batch))


class HeCifarNet(nn.Module):
    """HE-friendly CIFAR-10 CNN (grayscale): Conv -> Square -> FC -> Square -> FC.

    Only additions and multiplications are homomorphically computable, so the
    square activation replaces ReLU/MaxPool and the model stays 5 mults deep.
    """

    def __init__(self, hidden: int = HIDDEN, output: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, N_CHANNELS, kernel_size=KERNEL_SHAPE[0], stride=STRIDE)
        self.fc1 = nn.Linear(FLAT_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)
        inputs = inputs * inputs
        inputs = inputs.view(-1, FLAT_SIZE)
        inputs = self.fc1(inputs)
        inputs = inputs * inputs
        return self.fc2(inputs)


class EncConvNet:
    """PyTorch-like model running on ciphertexts with TenSEAL operations.

    - .conv2d_im2col() runs one convolution on an im2col-encoded ciphertext.
    - .square_() squares the encrypted vector in place (activation).
    - .mm() vector-matrix multiplication (Halevi-Shoup diagonal method).
    - `+ bias` adds a plaintext vector to the ciphertext.
    """

    def __init__(self, torch_nn: HeCifarNet) -> None:
        # conv1.weight: (12, 1, 5, 5) -> one 5x5 kernel per output channel
        self.conv1_weight = torch_nn.conv1.weight.data.view(N_CHANNELS, *KERNEL_SHAPE).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()  # (FLAT_SIZE, hidden)
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()  # (hidden, 10)
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x: ts.CKKSVector, windows_nb: int) -> ts.CKKSVector:
        # N_CHANNELS output channels: each is one conv on the shared im2col ciphertext.
        channels = [
            enc_x.conv2d_im2col(kernel, windows_nb) + bias
            for kernel, bias in zip(self.conv1_weight, self.conv1_bias)
        ]
        # pack the channel outputs (12 x 100) into one flattened ciphertext (1200)
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


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

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
    context: ts.Context, enc_model: EncConvNet, image: torch.Tensor
) -> list[float]:
    """Encode + encrypt a single RGB image, run the encrypted forward pass.

    The grayscale collapse + normalization happen here on the plaintext (client
    side), exactly as during training, so the ciphertext matches the plaintext.
    """
    gray = preprocess(image.unsqueeze(0))[0]  # (1, IMG_SIZE, IMG_SIZE)
    enc_x, windows_nb = ts.im2col_encoding(
        context, gray.view(IMG_SIZE, IMG_SIZE).tolist(),
        KERNEL_SHAPE[0], KERNEL_SHAPE[1], STRIDE,
    )
    enc_output = enc_model(enc_x, windows_nb)
    return enc_output.decrypt()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypted CIFAR-10 inference with TenSEAL (Route A: grayscale)"
    )
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Force retraining")
    parser.add_argument("--num-samples", type=int, default=3,
                        help="Images to encrypt & predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(73)
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = load_data(args.batch_size)

    # 1) Train or load the HE-friendly model ---------------------------------
    model = HeCifarNet().to(device)
    needs_train = args.train or not HE_MODEL_PATH.exists()
    if not needs_train:
        try:
            model.load_state_dict(torch.load(HE_MODEL_PATH, map_location=device))
            print(f"Loaded weights from: {HE_MODEL_PATH}")
        except (RuntimeError, KeyError) as exc:  # e.g. old v1 architecture
            print(f"Stored weights incompatible with current architecture ({exc}); "
                  f"retraining...")
            needs_train = True

    if needs_train:
        history = train(model, train_loader, test_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), HE_MODEL_PATH)
        lines = [f"Using device: {device}", *history["lines"]]
        LOG_SAVE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Model weights saved to: {HE_MODEL_PATH}")
    model.eval()

    # 2) Plaintext reference accuracy ----------------------------------------
    _, plain_accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Plaintext (grayscale) model test accuracy: {plain_accuracy:.4f}")

    # 3) Encrypted inference demo ---------------------------------------------
    context = create_context()
    enc_model = EncConvNet(model)

    print("\n--- Encrypted inference (client encrypts, server runs on ciphertexts) ---")
    indices = random.sample(range(len(test_loader.dataset)), args.num_samples)
    agree = 0
    for index in indices:
        image, label = test_loader.dataset[index]  # RGB tensor (3, 32, 32)

        # plaintext reference prediction (same preprocessing)
        with torch.no_grad():
            plain_logits = model(preprocess(image.unsqueeze(0)).to(device)).cpu()
        plain_pred = int(plain_logits.argmax(dim=1).item())

        # encrypted prediction (weights stay plaintext on the server side)
        start = time.perf_counter()
        enc_logits = encrypted_predict(context, enc_model, image)
        elapsed = time.perf_counter() - start
        enc_pred = int(max(range(10), key=lambda i: enc_logits[i]))

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
