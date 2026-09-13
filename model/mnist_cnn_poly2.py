"""Encrypted MNIST training with the HE-friendly quadratic activation

        p(x) = 0.0625 + 0.5*x + 0.5*x^2,

plus a plaintext ReLU baseline of the EXACT same architecture for comparison.

Why this activation:
    A pure square cannot mimic the identity-like behaviour of a ReLU, but any
    extra multiplicative depth would break the CKKS budget. p(x) adds a
    constant and a linear term at ZERO extra multiplicative depth: the x^2 part
    costs the single ciphertext-ciphertext multiplication of the old square,
    while +0.5*x + 0.0625 are plaintext multiplications/additions. The model
    therefore stays 5 multiplications deep and the CKKS parameter set is reused
    unchanged.

    On ciphertexts the polynomial is evaluated through the identity
        p(x) = 0.0625 + 0.5*x + 0.5*x^2 = 0.5*(x + 0.5)^2 - 0.0625
    which needs one squaring and no ciphertext copy.

Run:
    python model/mnist_cnn_poly2.py                       # poly2 HE: train + encrypt & predict
    python model/mnist_cnn_poly2.py --activation relu     # ReLU plaintext baseline
    python model/mnist_cnn_poly2.py --train --num-samples 5

Derived from model/mnist_cnn_enc.py (square activation); the square-based
script and its saved artifacts are left untouched.
"""

import argparse
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

# p(x) = A0 + A1*x + A2*x^2  (A0 + 0.5*(x+0.5)^2*(2*A2) identity used on ciphertexts)
A0, A1, A2 = 0.0625, 0.5, 0.5
# p(x) = A0 + A1*x + A2*x^2 = A2*(x + C)^2 + D. On ciphertexts each activation
# only computes (x + C)^2 (one squaring, same depth as the old x^2 net); the A2
# factor and D constant are folded into the *next* linear layer's weights/bias
# (see EncConvNet), so no extra rescale level is consumed.
C = A1 / (2 * A2)                  # 0.5
D = A0 - A1 * A1 / (4 * A2)        # -0.0625

# CKKS parameters (following TenSEAL Tutorial 4):
# 7 primes -> 6 multiplications deep, which matches the model depth exactly.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 21, 21, 21, 21, 40]
GLOBAL_SCALE = 2 ** 21
KERNEL_SHAPE = (7, 7)
STRIDE = 3
FLAT_SIZE = 256  # 4 channels * 8 * 8 (28x28 input, stride-3 7x7 kernel)


def act_poly2(x: torch.Tensor) -> torch.Tensor:
    """Plaintext evaluation of p(x) = A0 + A1*x + A2*x^2 (exact user formula)."""
    return A0 + A1 * x + A2 * x * x


class HeMnistNet(nn.Module):
    """MNIST CNN with selectable activation: 'poly2' (HE-friendly) or 'relu'
    (plaintext baseline). Structure identical in both cases so that the two can
    be compared directly; only the activation differs."""

    def __init__(self, activation: str = "poly2", hidden: int = 64, output: int = 10) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3)
        self.fc1 = nn.Linear(FLAT_SIZE, hidden)
        self.fc2 = nn.Linear(hidden, output)

    def _act(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.relu(x)
        return act_poly2(x)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = self.conv1(inputs)
        inputs = self._act(inputs)
        inputs = inputs.view(-1, FLAT_SIZE)
        inputs = self.fc1(inputs)
        inputs = self._act(inputs)
        return self.fc2(inputs)


def enc_act(x: ts.CKKSVector) -> ts.CKKSVector:
    """Ciphertext half of the activation: returns (x + C)^2.

    Combined with the folded linear layer (A2*W, bias + D*W.sum(axis=1)) this
    evaluates exactly p(x) = A0 + A1*x + A2*x^2 of the plaintext forward:
    W*p(u) + b = (A2*W)*(u+C)^2 + (b + D*rowsum(W)).
    Costs exactly one ciphertext-ciphertext multiplication (one rescale level),
    identical to the old x^2 activation; no plaintext multiply after squaring,
    which would blow the rescale budget / trigger 'scale out of bounds'.
    """
    x = x + C              # plaintext add, free
    return x * x           # one multiplicative level


class EncConvNet:
    """PyTorch-like model running on ciphertexts with TenSEAL operations.

    - .conv2d_im2col() performs one convolution on an im2col-encoded ciphertext.
    - .mm() performs vector-matrix multiplication (Halevi-Shoup diagonal method).
    - enc_act() applies the quadratic activation (one squaring).
    - `+ bias` adds a plaintext vector (bias) to the ciphertext.
    """

    def __init__(self, torch_nn: HeMnistNet) -> None:
        self.conv1_weight = torch_nn.conv1.weight.data.view(4, 7, 7).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        # Fold activation: p(u) = A2*(u+C)^2 + D  =>  layer input becomes (u+C)^2,
        # so W*x+b becomes (A2*W)*z + (b + D*W.sum(dim=1)) with z = (u+C)^2.
        fc1_w = torch_nn.fc1.weight.data
        fc1_b = torch_nn.fc1.bias.data
        fc2_w = torch_nn.fc2.weight.data
        fc2_b = torch_nn.fc2.bias.data
        self.fc1_weight = (A2 * fc1_w).T.tolist()        # (FLAT_SIZE, hidden)
        self.fc1_bias = (fc1_b + D * fc1_w.sum(dim=1)).tolist()
        self.fc2_weight = (A2 * fc2_w).T.tolist()        # (hidden, 10)
        self.fc2_bias = (fc2_b + D * fc2_w.sum(dim=1)).tolist()

    def forward(self, enc_x: ts.CKKSVector, windows_nb: int) -> ts.CKKSVector:
        channels = [
            enc_x.conv2d_im2col(kernel, windows_nb) + bias
            for kernel, bias in zip(self.conv1_weight, self.conv1_bias)
        ]
        # pack the 4 channel outputs into one flattened ciphertext
        enc_x = ts.CKKSVector.pack_vectors(channels)
        enc_x = enc_act(enc_x)
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_act(enc_x)
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


def artifact_paths(activation: str) -> tuple[Path, Path]:
    tag = "poly2_he_cnn_train" if activation == "poly2" else "relu_cnn_train"
    return RESULTS_DIR / f"mnist_{tag}.pt", RESULTS_DIR / f"mnist_{tag}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNIST training: HE-friendly quadratic activation p(x)=0.0625+0.5x+0.5x^2 "
                    "with a plaintext ReLU baseline of the same architecture"
    )
    parser.add_argument("--activation", choices=["poly2", "relu"], default="poly2",
                        help="poly2: HE quadratic (train + encrypted inference); "
                             "relu: plaintext-only baseline (ReLU is not HE-computable)")
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
    activation = args.activation

    train_loader, test_loader = load_data(args.batch_size)

    # 1) Train or load the model ---------------------------------------------
    model = HeMnistNet(activation=activation).to(device)
    model_path, log_path = artifact_paths(activation)
    if args.train or not model_path.exists():
        history = train(model, train_loader, test_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), model_path)
        lines = [f"Using device: {device}", f"Activation: {activation}", *history["lines"]]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Model weights saved to: {model_path}")
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded weights from: {model_path}")
    model.eval()

    # 2) Plaintext reference accuracy ----------------------------------------
    _, plain_accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"[{activation}] Plaintext model test accuracy: {plain_accuracy:.4f}")

    if activation == "relu":
        print("ReLU is not homomorphically computable: this run is a plaintext-only "
              "baseline; no encrypted inference is performed.")
        return

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
