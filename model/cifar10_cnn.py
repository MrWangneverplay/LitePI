"""Train a compact nine-layer CNN for CIFAR-10."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "datas"
MODEL_SAVE_PATH = RESULTS_DIR / "cifar10_cnn_train.pt"
LOG_SAVE_PATH = RESULTS_DIR / "cifar10_cnn_train.log"
LOSS_PLOT_PATH = RESULTS_DIR / "cifar10_cnn_train_loss.png"
ACCURACY_PLOT_PATH = RESULTS_DIR / "cifar10_cnn_train_accuracy.png"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Cifar10CNN(nn.Module):
    """Nine-layer CNN: Conv, ReLU, Conv, ReLU, Pool, Conv, ReLU, Pool, FC."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        return self.classifier(torch.flatten(features, 1))


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_data(batch_size: int) -> tuple[DataLoader, DataLoader]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
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
    plt.title("CIFAR-10 CNN Loss")
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
    plt.title("CIFAR-10 CNN Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(ACCURACY_PLOT_PATH, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a compact CIFAR-10 CNN")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(73)
    device = get_device()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = load_data(args.batch_size)
    model = Cifar10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
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
        scheduler.step()
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
    log_lines.extend(
        [
            f"Model weights saved to: {MODEL_SAVE_PATH}",
            f"Loss curve saved to: {LOSS_PLOT_PATH}",
            f"Accuracy curve saved to: {ACCURACY_PLOT_PATH}",
        ]
    )
    LOG_SAVE_PATH.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    print(f"Model weights saved to: {MODEL_SAVE_PATH}")
    print(f"Loss curve saved to: {LOSS_PLOT_PATH}")
    print(f"Accuracy curve saved to: {ACCURACY_PLOT_PATH}")
    print(json.dumps(history))


if __name__ == "__main__":
    main()
