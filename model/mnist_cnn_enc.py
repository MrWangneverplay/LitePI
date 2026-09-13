"""Encrypted MNIST inference demo using TenSEAL (CKKS scheme).

Pipeline:
    1. Train (or load) a HE-friendly CNN: conv -> square -> fc -> square -> fc.
       ReLU/MaxPool are not homomorphically computable, so the square activation
       is used instead; the network is only 6 multiplications deep so it fits
       the CKKS multiplicative budget.
    2. Client encrypts a test image under CKKS (private context, never shared).
    3. Server runs the whole forward pass on ciphertexts, while the model
       weights stay in plaintext (data usable but invisible).
    4. Client decrypts the output and compares with the plaintext prediction.

Run:
    python model/mnist_cnn_enc.py                 # train if needed, then encrypt & predict
    python model/mnist_cnn_enc.py --train         # force retraining
    python model/mnist_cnn_enc.py --num-samples 5 # encrypt & predict 5 images
"""

import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import tenseal as ts

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "datas"
HE_MODEL_PATH = RESULTS_DIR / "mnist_he_cnn_train.pt"
LOG_SAVE_PATH = RESULTS_DIR / "mnist_he_cnn_train.log"

# CKKS parameters (following TenSEAL Tutorial 4):
# 7 primes -> 6 multiplications deep, which matches the model depth exactly.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 21, 21, 21, 21, 40]
GLOBAL_SCALE = 2 ** 21
KERNEL_SHAPE = (7, 7)
STRIDE = 3
FLAT_SIZE = 256  # 4 channels * 8 * 8 (28x28 input, stride-3 7x7 kernel)


class HeMnistNet(nn.Module):
    """HE-friendly MNIST CNN: Conv -> Square -> FC -> Square -> FC.

    The square activation replaces ReLU/MaxPool, which cannot be evaluated
    exactly on ciphertexts.
    """

    def __init__(self, hidden: int = 64, output: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3)
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

    - .conv2d_im2col() performs one convolution on an im2col-encoded ciphertext.
    - .mm() performs vector-matrix multiplication (Halevi-Shoup diagonal method).
    - .square_() squares the encrypted vector in place.
    - `+ bias` adds a plaintext vector (bias) to the ciphertext.
    """

    def __init__(self, torch_nn: HeMnistNet) -> None:
        self.conv1_weight = torch_nn.conv1.weight.data.view(4, 7, 7).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

    def forward(self, enc_x: ts.CKKSVector, windows_nb: int) -> ts.CKKSVector:
        channels = [
            enc_x.conv2d_im2col(kernel, windows_nb) + bias
            for kernel, bias in zip(self.conv1_weight, self.conv1_bias)
        ]
        # pack the 4 channel outputs into one flattened ciphertext
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
) -> tuple[float, float]:
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
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "test_loss": [], "test_accuracy": []}

    for epoch in range(1, epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
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
    """Encode + encrypt a single 28x28 image, run the encrypted forward pass."""
    enc_x, windows_nb = ts.im2col_encoding(
        context, image.view(28, 28).tolist(), KERNEL_SHAPE[0], KERNEL_SHAPE[1], STRIDE
    )
    enc_output = enc_model(enc_x, windows_nb)
    return enc_output.decrypt()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encrypted MNIST inference with TenSEAL")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--train", action="store_true", help="Force retraining")
    parser.add_argument("--num-samples", type=int, default=3, help="Images to encrypt & predict")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(73)
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = load_data(args.batch_size)

    # 1) Train or load the HE-friendly model ---------------------------------
    model = HeMnistNet().to(device)
    if args.train or not HE_MODEL_PATH.exists():
        history = train(model, train_loader, test_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), HE_MODEL_PATH)
        lines = [f"Using device: {device}", *history["lines"]]
        LOG_SAVE_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Model weights saved to: {HE_MODEL_PATH}")
    else:
        model.load_state_dict(torch.load(HE_MODEL_PATH, map_location=device))
        print(f"Loaded weights from: {HE_MODEL_PATH}")
    model.eval()

    # 2) Plaintext reference accuracy ----------------------------------------
    _, plain_accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Plaintext model test accuracy: {plain_accuracy:.4f}")

    # 3) Encrypted inference demo ---------------------------------------------
    context = create_context()
    enc_model = EncConvNet(model)

    print("\n--- Encrypted inference (client encrypts, server runs on ciphertexts) ---")
    indices = random.sample(range(len(test_loader.dataset)), args.num_samples)
    agree = 0
    for index in indices:
        image, label = test_loader.dataset[index]

        # plaintext reference prediction
        with torch.no_grad():
            plain_logits = model(image.unsqueeze(0).to(device)).cpu()
        plain_pred = int(plain_logits.argmax(dim=1).item())

        # encrypted prediction (weights stay plaintext on the server side)
        enc_logits = encrypted_predict(context, enc_model, image)
        enc_pred = int(max(range(10), key=lambda i: enc_logits[i]))

        match = "OK" if plain_pred == enc_pred else "MISMATCH"
        agree += plain_pred == enc_pred
        print(
            f"sample #{index:5d}  label={label}  plaintext={plain_pred}  "
            f"encrypted={enc_pred}  [{match}]"
        )

    print(f"\nPlaintext vs encrypted agreement: {agree}/{args.num_samples}")
    print(f"Context secrets kept local, only public parts would be shared: "
          f"{context.is_private()}")


if __name__ == "__main__":
    main()
