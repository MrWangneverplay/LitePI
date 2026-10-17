"""Deepen the CIFAR-10 network; report ReLU / P2 accuracy + multiplicative depth.

Motivation (the advisor's question at the group meeting): the current network has
only 2 nonlinear layers and CIFAR-10 tops out around 60%.  Two questions:
  1. does deepening actually raise accuracy?  (plaintext, both ReLU and P2)
  2. does deepening make the pure-polynomial (HE-only) route infeasible because
     the CKKS multiplicative-depth budget runs out?

multiplicative depth = 1 (conv) + 2 * D   (D activations + D FC layers)
current budget (poly degree 8192, 7 primes) = 6 levels.

Run:
    python analysis/train_deep.py --depth 2,3,4 --activation relu,p2
    python analysis/train_deep.py --depth 3 --channels 32 --hidden 256
"""

from __future__ import annotations

import argparse
import json
import importlib.util
import subprocess
import sys
from datetime import datetime, timezone
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


class DeepCifarNet(nn.Module):
    """CIFAR-10 CNN with `depth` nonlinear layers, each followed by an FC layer.

    depth=2 reproduces the existing twin: conv -> act -> FC -> act -> FC.
    """

    def __init__(self, depth: int = 2, channels: int = 12, hidden: int = 128,
                 activation: str = "relu", classes: int = 10):
        super().__init__()
        assert depth >= 2
        self.depth = depth
        self.conv1 = nn.Conv2d(1, channels, kernel_size=5, stride=3)
        flat = channels * 10 * 10
        self.act = ACTS[activation]
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(flat, hidden))
        for _ in range(depth - 2):
            self.fcs.append(nn.Linear(hidden, hidden))
        self.fcs.append(nn.Linear(hidden, classes))

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.size(0), -1)  # flatten conv output before the FC stack
        for i in range(self.depth - 1):
            x = self.act(x)
            x = self.fcs[i](x)
        x = self.act(x)
        x = self.fcs[-1](x)
        return x


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def git_sha(root: Path):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=False, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def save_record(root: Path, act: str, depth: int, seed: int, acc: float,
                ckpt: Path, epochs: int, channels: int, hidden: int) -> Path:
    out_dir = root / "results" / "raw_results" / "E1" / "cifar10" / "deep" / f"{act}_depth{depth}"
    out_dir.mkdir(parents=True, exist_ok=True)
    mdepth = 1 + 2 * depth
    rec = {
        "exp_id": "E1-deep", "dataset": "cifar10", "method": f"deep_{act}_depth{depth}",
        "seed": seed, "epochs": epochs,
        "git_sha": git_sha(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": {"channels": channels, "hidden": hidden,
                         "depth": depth, "activation": act,
                         "multiplicative_depth": mdepth,
                         "fits_ckks_budget_6": mdepth <= BUDGET},
        "checkpoint": str(ckpt.relative_to(root)),
        "test_accuracy": acc,
        "note": "deepened CIFAR-10 twin (grayscale, single conv + depth FCs)",
    }
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepen CIFAR-10 and report accuracy")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--depth", default="2,3,4", help="comma list of nonlinear-layer counts")
    ap.add_argument("--activation", default="relu,p2")
    ap.add_argument("--channels", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=73, help="first seed")
    ap.add_argument("--num-seeds", type=int, default=1, help="seeds 73..73+n-1; report mean+-std")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, quick sanity check")
    args = ap.parse_args()

    root = Path(args.root)
    module = load_module(root / "model" / "cifar10_cnn_relu.py")
    train_loader, test_loader = module.load_data(args.batch_size)
    prepare = module.preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = [int(d) for d in args.depth.split(",") if d.strip()]
    acts = [a.strip() for a in args.activation.split(",") if a.strip()]
    epochs = 1 if args.smoke else args.epochs
    seeds = list(range(args.seed, args.seed + args.num_seeds))

    print(f"device={device}  channels={args.channels}  hidden={args.hidden}  epochs={epochs}")
    print("=" * 78)
    print(f"{'config':<20}{'mult depth':>11}{'fits 6?':>9}{'mean':>9}{'std':>9}")
    summary = {}
    for act in acts:
        for d in depths:
            accs = []
            for seed in seeds:
                torch.manual_seed(seed)
                net = DeepCifarNet(depth=d, channels=args.channels, hidden=args.hidden,
                                   activation=act).to(device)
                acc = train_one(net, train_loader, test_loader, device, prepare,
                                epochs, args.lr, args.wd)
                accs.append(acc)
                if not args.no_save and not args.smoke:
                    ckpt = root / "datas" / f"cifar10_deep_{act}_depth{d}_seed{seed}_cnn_train.pt"
                    torch.save(net.state_dict(), ckpt)
                    p = save_record(root, act, d, seed, acc, ckpt, epochs,
                                    args.channels, args.hidden)
                    print(f"       -> {p.relative_to(root)}")
            mdepth = 1 + 2 * d
            fits = "OK" if mdepth <= BUDGET else "EXCEEDS"
            mean = sum(accs) / len(accs)
            std = (sum((a - mean) ** 2 for a in accs) / len(accs)) ** 0.5
            summary[(act, d)] = (mean, std, fits)
            print(f"{act} x depth{d:<6}{mdepth:>11}{fits:>9}{mean*100:>8.2f}%{std*100:>8.2f}")
    print("=" * 78)
    print(f"multiplicative depth = 1 (conv) + 2 * D; current CKKS budget = {BUDGET}")
    print("解读: 深度 >= 3 时纯 HE(全 P2) 路径超出 CKKS 深度预算 -> 需要 Hybrid(真 ReLU via MPC)")


if __name__ == "__main__":
    main()
