"""Encrypted CIFAR-10 training (Route A: grayscale) with the HE-friendly
quadratic activation

        p(x) = 0.0625 + 0.5*x + 0.5*x^2,

plus a plaintext ReLU baseline of the EXACT same architecture for comparison.

Route A design (unchanged from cifar10_cnn_enc.py):
    The 3-channel RGB image is collapsed to ONE grayscale channel by a fixed
    linear combination (ITU-R BT.601 luma weights, equivalent to a fixed 1x1x3
    convolution) *in the plaintext domain* before encryption, then normalized
    to zero-mean/unit-variance. The HE network is:

        RGB(3x32x32) --to_gray + normalize (plaintext)--> Gray(1x32x32)
          -> Conv(1->12, 5x5, s=3)   -> 12 channels x 10x10 = 1200 slots
          -> p(x) activation          (one squaring, like the old square)
          -> FC(1200 -> 128)
          -> p(x) activation
          -> FC(128 -> 10)

Activation note:
    p(x) adds constant + linear terms at ZERO extra multiplicative depth (the
    x^2 part is the same single ciphertext-ciphertext multiplication as the old
    square activation; the rest are plaintext mul/add). The model therefore
    stays 5 multiplications deep and the CKKS parameter set is unchanged.
    On ciphertexts it is evaluated as p(x) = 0.5*(x + 0.5)^2 - 0.0625.

Run:
    python model/cifar10_cnn_poly2.py                      # poly2 HE: train + encrypt & predict
    python model/cifar10_cnn_poly2.py --activation relu    # ReLU plaintext baseline
    python model/cifar10_cnn_poly2.py --train --num-samples 5

Derived from model/cifar10_cnn_enc.py (square activation); the square-based
script and its saved artifacts are left untouched.
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

# p(x) = A0 + A1*x + A2*x^2
A0, A1, A2 = 0.0625, 0.5, 0.5
# p(x) = A0 + A1*x + A2*x^2 = A2*(x + C)^2 + D. On ciphertexts each activation
# only computes (x + C)^2 (one squaring, same depth as the old x^2 net); the A2
# factor and D constant are folded into the *next* linear layer (see EncConvNet).
C = A1 / (2 * A2)                  # 0.5
D = A0 - A1 * A1 / (4 * A2)        # -0.0625

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
    """Zero-mean / unit-variance normalization on the grayscale input (plaintext-side)."""
    return (batch - GRAY_MEAN) / GRAY_STD


def preprocess(batch: torch.Tensor) -> torch.Tensor:
    """Full plaintext-side input pipeline: RGB -> gray -> normalize."""
    return normalize_gray(to_gray(batch))


def act_poly2(x: torch.Tensor) -> torch.Tensor:
    """Plaintext evaluation of p(x) = A0 + A1*x + A2*x^2 (exact user formula)."""
    return A0 + A1 * x + A2 * x * x


class HeCifarNet(nn.Module):
    """CIFAR-10 CNN (grayscale) with selectable activation: 'poly2' (HE-friendly)
    or 'relu' (plaintext baseline). Same structure, only the activation differs."""

    def __init__(self, activation: str = "poly2", hidden: int = HIDDEN, output: int = 10) -> None:
        super().__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(1, N_CHANNELS, kernel_size=KERNEL_SHAPE[0], stride=STRIDE)
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
    evaluates exactly p(x) = A0 + A1*x + A2*x^2 of the plaintext forward.
    Costs exactly one ciphertext-ciphertext multiplication (one rescale level),
    identical to the old x^2 activation; no plaintext multiply after squaring.
    """
    x = x + C              # plaintext add, free
    return x * x           # one multiplicative level


class EncConvNet:
    """PyTorch-like model running on ciphertexts with TenSEAL operations."""

    def __init__(self, torch_nn: HeCifarNet) -> None:
        self.conv1_weight = torch_nn.conv1.weight.data.view(N_CHANNELS, *KERNEL_SHAPE).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()
        # Fold activation: p(u) = A2*(u+C)^2 + D => layer input is (u+C)^2, so
        # W*x+b becomes (A2*W)*z + (b + D*W.sum(dim=1)) with z = (u+C)^2.
        fc1_w = torch_nn.fc1.weight.data
        fc1_b = torch_nn.fc1.bias.data
        fc2_w = torch_nn.fc2.weight.data
        fc2_b = torch_nn.fc2.bias.data
        self.fc1_weight = (A2 * fc1_w).T.tolist()  # (FLAT_SIZE, hidden)
        self.fc1_bias = (fc1_b + D * fc1_w.sum(dim=1)).tolist()
        self.fc2_weight = (A2 * fc2_w).T.tolist()  # (hidden, 10)
        self.fc2_bias = (fc2_b + D * fc2_w.sum(dim=1)).tolist()

    def forward(self, enc_x: ts.CKKSVector, windows_nb: int) -> ts.CKKSVector:
        # N_CHANNELS output channels: each is one conv on the shared im2col ciphertext.
        channels = [
            enc_x.conv2d_im2col(kernel, windows_nb) + bias
            for kernel, bias in zip(self.conv1_weight, self.conv1_bias)
        ]
        # pack the channel outputs (12 x 100) into one flattened ciphertext (1200)
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


def artifact_paths(activation: str) -> tuple[Path, Path]:
    tag = "poly2_he_cnn_train" if activation == "poly2" else "relu_cnn_train"
    return RESULTS_DIR / f"cifar10_{tag}.pt", RESULTS_DIR / f"cifar10_{tag}.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CIFAR-10 (grayscale) training: HE-friendly quadratic activation "
                    "p(x)=0.0625+0.5x+0.5x^2 with a plaintext ReLU baseline of the same architecture"
    )
    parser.add_argument("--activation", choices=["poly2", "relu"], default="poly2",
                        help="poly2: HE quadratic (train + encrypted inference); "
                             "relu: plaintext-only baseline (ReLU is not HE-computable)")
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
    activation = args.activation

    train_loader, test_loader = load_data(args.batch_size)

    # 1) Train or load the model ---------------------------------------------
    model = HeCifarNet(activation=activation).to(device)
    model_path, log_path = artifact_paths(activation)
    needs_train = args.train or not model_path.exists()
    if not needs_train:
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded weights from: {model_path}")
        except (RuntimeError, KeyError) as exc:  # e.g. old architecture
            print(f"Stored weights incompatible with current architecture ({exc}); "
                  f"retraining...")
            needs_train = True

    if needs_train:
        history = train(model, train_loader, test_loader, device, args.epochs, args.lr)
        torch.save(model.state_dict(), model_path)
        lines = [f"Using device: {device}", f"Activation: {activation}", *history["lines"]]
        log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Model weights saved to: {model_path}")
    model.eval()

    # 2) Plaintext reference accuracy ----------------------------------------
    _, plain_accuracy = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"[{activation}] Plaintext (grayscale) model test accuracy: {plain_accuracy:.4f}")

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
