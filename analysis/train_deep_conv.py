"""Real conv-layer deepening for CIFAR-10 (VGG-lite, HE-friendly).

The FC-only deepen script (train_deep.py) plateaus around depth 3-4 because the
network still has a single conv layer.  This script deepens the CONV stack, which
is what the advisor meant by "把网络深度加深".

Architecture (grayscale 32x32, RGB -> gray via the ReLU twin's preprocess):
    [Conv(3x3,p1) -> act] x N, with AvgPool(2) after every 2 blocks
    -> Flatten -> Linear(hidden) -> act -> Linear(10)

Multiplicative depth (HE-only / all-P2 route):
    each conv = 1, each P2 activation = 1, final FC = 1
    => 1 + N(conv) + (N+1)(P2) = 2N + 2
    N=2 -> 6 (fits budget 6), N=4 -> 10, N=6 -> 14 (all exceed)
This is the quantitative argument that deeper CONV nets force the Hybrid route.

Run:
    python analysis/train_deep_conv.py --shape-check
    python analysis/train_deep_conv.py --conv-depth 2,4,6 --activation relu,p2 --smoke
    python analysis/train_deep_conv.py --conv-depth 2,4,6 --activation relu,p2 --channels 32 --hidden 256
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_ROOT = r"D:\王梓杨\项目组\wzy"

A0, A1, A2 = 0.0625, 0.5, 0.5
BUDGET = 6  # multiplicative-depth budget of the current CKKS parameters


def act_relu(x):
    return torch.relu(x)


def act_p2(x):
    return A0 + A1 * x + A2 * x * x


ACTS = {"relu": act_relu, "p2": act_p2}


class DeepConvNet(nn.Module):
    """VGG-lite: `conv_depth` conv blocks (3x3,p1) + avg-pool every 2, FC head."""

    def __init__(self, conv_depth: int = 4, channels: int = 32, hidden: int = 256,
                 activation: str = "relu", classes: int = 10):
        super().__init__()
        assert conv_depth >= 2
        self.conv_depth = conv_depth
        self.act = ACTS[activation]
        blocks = []
        in_ch = 1
        for i in range(conv_depth):
            blocks.append(nn.Conv2d(in_ch, channels, kernel_size=3, padding=1))
            in_ch = channels
            if (i + 1) % 2 == 0:
                blocks.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.pools = (conv_depth // 2)
        self.spatial = 32 // (2 ** self.pools)
        flat = channels * self.spatial * self.spatial
        self.fc1 = nn.Linear(flat, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, nn.Conv2d):
                x = self.act(x)
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def run_epoch(model, loader, criterion, device, prepare, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = correct = total = 0
    with torch.set_grad_enabled(is_train):
        for inputs, targets in loader:
            inputs = prepare(inputs).to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * targets.size(0)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
            total += int(targets.size(0))
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one(model, train_loader, test_loader, device, prepare, epochs, lr, wd):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    test_acc = 0.0
    for _ in range(epochs):
        run_epoch(model, train_loader, criterion, device, prepare, optimizer)
        _, test_acc = run_epoch(model, test_loader, criterion, device, prepare)
        scheduler.step()
    return test_acc


def shape_check(channels: int, hidden: int):
    for n in (2, 4, 6):
        net = DeepConvNet(conv_depth=n, channels=channels, hidden=hidden, activation="relu")
        n_params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            out = net(torch.randn(1, 1, 32, 32))
        mdepth = 2 * n + 2
        print(f"conv_depth={n}  spatial={net.spatial}x{net.spatial}  "
              f"mult_depth={mdepth} ({'OK' if mdepth <= BUDGET else 'EXCEEDS'})  "
              f"out={tuple(out.shape)}  params={n_params:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepen the CIFAR-10 CONV stack")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--conv-depth", default="2,4,6", help="comma list of conv-layer counts")
    ap.add_argument("--activation", default="relu,p2")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--smoke", action="store_true", help="1 epoch, quick sanity check")
    ap.add_argument("--shape-check", action="store_true", help="print shapes/params, no training")
    args = ap.parse_args()

    if args.shape_check:
        shape_check(args.channels, args.hidden)
        return

    root = Path(args.root)
    module = load_module(root / "model" / "cifar10_cnn_relu.py")
    train_loader, test_loader = module.load_data(args.batch_size)
    prepare = module.preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = [int(d) for d in args.conv_depth.split(",") if d.strip()]
    acts = [a.strip() for a in args.activation.split(",") if a.strip()]
    epochs = 1 if args.smoke else args.epochs

    print(f"device={device}  channels={args.channels}  hidden={args.hidden}  epochs={epochs}")
    print("=" * 78)
    print(f"{'config':<20}{'mult depth':>11}{'fits 6?':>9}{'acc':>12}")
    for act in acts:
        for n in depths:
            torch.manual_seed(args.seed)
            net = DeepConvNet(conv_depth=n, channels=args.channels,
                              hidden=args.hidden, activation=act).to(device)
            acc = train_one(net, train_loader, test_loader, device, prepare,
                            epochs, args.lr, args.wd)
            mdepth = 2 * n + 2
            fits = "OK" if mdepth <= BUDGET else "EXCEEDS"
            print(f"{act} x conv{n:<8}{mdepth:>11}{fits:>9}{acc*100:>11.2f}%")
    print("=" * 78)
    print(f"multiplicative depth = 2*N + 2; current CKKS budget = {BUDGET}")


if __name__ == "__main__":
    main()
