"""Train a compact CNN for MNIST and save its training artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "datas"
MODEL_SAVE_PATH = RESULTS_DIR / "mnist_cnn_train.pt"
LOG_SAVE_PATH = RESULTS_DIR / "mnist_cnn_train.log"
LOSS_PLOT_PATH = RESULTS_DIR / "mnist_cnn_train_loss.png"
ACCURACY_PLOT_PATH = RESULTS_DIR / "mnist_cnn_train_accuracy.png"


class MnistCNN(nn.Module):
    """Nine-layer CNN: Conv, ReLU, Pool, Conv, ReLU, Pool, FC, ReLU, FC."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


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


def save_plots(history: dict[str, list[float]]) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["test_loss"], marker="o", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("MNIST CNN Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH, dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_accuracy"], marker="o", label="Train")
    plt.plot(epochs, history["test_accuracy"], marker="o", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("MNIST CNN Accuracy")
    plt.ylim(0.8, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact MNIST CNN")
    parser.add_argument("--epochs", type=int, default=6, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(73)
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = load_data(args.batch_size)
    model = MnistCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_accuracy": [],
        "test_accuracy": [],
    }
    log_lines = [f"Using device: {device}"]

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        test_loss, test_accuracy = run_epoch(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_accuracy"].append(train_accuracy)
        history["test_accuracy"].append(test_accuracy)
        line = (
            f"Epoch {epoch:2d}/{args.epochs}  "
            f"Train Loss: {train_loss:.6f}  Train Acc: {train_accuracy:.4f}  "
            f"Test Loss: {test_loss:.6f}  Test Acc: {test_accuracy:.4f}"
        )
        print(line)
        log_lines.append(line)

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    save_plots(history)
    log_lines.append(f"Model weights saved to: {MODEL_SAVE_PATH}")
    log_lines.append(f"Loss curve saved to: {LOSS_PLOT_PATH}")
    log_lines.append(f"Accuracy curve saved to: {ACCURACY_PLOT_PATH}")
    LOG_SAVE_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"Model weights saved to: {MODEL_SAVE_PATH}")
    print(f"Loss curve saved to: {LOSS_PLOT_PATH}")
    print(f"Accuracy curve saved to: {ACCURACY_PLOT_PATH}")
    print(json.dumps(history))


if __name__ == "__main__":
    main()
