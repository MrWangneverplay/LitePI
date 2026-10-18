"""E1: train hybrid (per-layer P2 + ReLU) networks and report accuracy.

Schedules s = (s1, s2):
    s_k = 0  ->  HE-Poly   P2(x) = 0.0625 + 0.5*x + 0.5*x^2
    s_k = 1  ->  Hybrid    ReLU (exact, via MPC in the real protocol)

    s=00  all P2     (HE-only twin)
    s=01  P2 + ReLU  (layer 1 approximated, layer 2 exact)
    s=10  ReLU + P2  (layer 1 exact, layer 2 approximated)
    s=11  all ReLU   (ReLU twin / Hybrid-all)

This is the "训练即 HE 友好" hybrid: the mixed activation is TRAINED, unlike
eval_poly_substitution.py which post-hoc substitutes on a ReLU-trained model
and therefore measures the *catastrophic* substitution loss.

Run (inside the project venv, on CPU or the remote GPU server):
    python analysis/train_hybrid.py --datasets mnist,cifar10,uci_har --schedules 00,01,10,11
    python analysis/train_hybrid.py --datasets mnist --schedules 01,10 --smoke
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

A0, A1, A2 = 0.0625, 0.5, 0.5


def act_p2(x: torch.Tensor) -> torch.Tensor:
    return A0 + A1 * x + A2 * x * x


def act_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


ACTS = {"p2": act_p2, "relu": act_relu}


# ---------------------------------------------------------------- networks
# Architecture identical to model/*_cnn_relu.py / *_cnn_poly2.py (same geometry),
# only the per-layer activation differs.

class MnistNet(nn.Module):
    def __init__(self, a1: str = "p2", a2: str = "relu", hidden: int = 64, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.a1, self.a2 = ACTS[a1], ACTS[a2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.conv1(x))
        x = x.view(-1, 256)
        x = self.a2(self.fc1(x))
        return self.fc2(x)


class CifarNet(nn.Module):
    def __init__(self, a1: str = "p2", a2: str = "relu", hidden: int = 128, classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=3)
        self.fc1 = nn.Linear(1200, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.a1, self.a2 = ACTS[a1], ACTS[a2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.conv1(x))
        x = x.view(-1, 1200)
        x = self.a2(self.fc1(x))
        return self.fc2(x)


class HarNet(nn.Module):
    def __init__(self, a1: str = "p2", a2: str = "relu",
                 channels: int = 12, hidden: int = 128, classes: int = 6):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 9), stride=(2, 1))
        self.fc1 = nn.Linear(channels * 63, hidden)
        self.drop = nn.Dropout(p=0.3)  # inactive in eval mode
        self.fc2 = nn.Linear(hidden, classes)
        self.a1, self.a2 = ACTS[a1], ACTS[a2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.conv1(x))
        x = x.view(-1, x.shape[1] * 63)
        x = self.drop(self.a2(self.fc1(x)))
        return self.fc2(x)


# per-dataset: module to import, net class, hyper-params
CONFIG = {
    # opt/wd/cosine mirror each dataset's *_cnn_relu.py twin, so the four
    # schedules stay directly comparable to the Plain-Poly / Plain-ReLU rows.
    "mnist": dict(module="mnist_cnn_relu.py", cls=MnistNet,
                  epochs=10, lr=1e-3, bs=64, opt="adam", wd=0.0, cosine=False),
    "cifar10": dict(module="cifar10_cnn_relu.py", cls=CifarNet,
                    epochs=40, lr=1e-3, bs=128, opt="adam", wd=1e-4, cosine=True),
    "uci_har": dict(module="uci_har_cnn_relu.py", cls=HarNet,
                    epochs=30, lr=1e-3, bs=64, opt="adamw", wd=1e-4, cosine=True),
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def git_sha(root: Path) -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=False, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def make_prepare(dataset: str, module):
    if dataset == "cifar10":
        return module.preprocess
    if dataset == "uci_har":
        return lambda x: x.unsqueeze(1)
    return lambda x: x


def run_epoch(model, loader, criterion, device, prepare, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
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


def train_one(net, train_loader, test_loader, device, prepare,
              epochs, lr, wd, opt_name, cosine):
    """Train one network; returns the FINAL-epoch test accuracy (test_acc_final),
    matching how the reference twins report their numbers.

    The optimiser recipe mirrors each dataset's *_cnn_relu.py twin, so the four
    schedules stay directly comparable to the Plain-Poly / Plain-ReLU rows.
    """
    criterion = nn.CrossEntropyLoss()
    opt_cls = optim.AdamW if opt_name == "adamw" else optim.Adam
    optimizer = opt_cls(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = (optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
                 if cosine else None)
    test_acc = float("nan")
    for _ in range(epochs):
        run_epoch(net, train_loader, criterion, device, prepare, optimizer)
        _, test_acc = run_epoch(net, test_loader, criterion, device, prepare)
        if scheduler is not None:
            scheduler.step()
    return test_acc


def save_record(root: Path, dataset: str, schedule: str, seed: int, acc: float,
                ckpt: Path, epochs: int) -> Path:
    out_dir = root / "results" / "raw_results" / "E1" / dataset / f"hybrid_s{schedule}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "exp_id": "E1", "dataset": dataset, "method": f"hybrid_s{schedule}",
        "seed": seed, "epochs": epochs,
        "git_sha": git_sha(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "activation": {"layer1": "p2" if schedule[0] == "0" else "relu",
                       "layer2": "p2" if schedule[1] == "0" else "relu"},
        "checkpoint": str(ckpt.relative_to(root)),
        "test_accuracy": acc,
        "note": "trained with per-layer activation (训练即 HE 友好), seed 73",
    }
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Train hybrid (per-layer P2/ReLU) networks")
    ap.add_argument("--root", default=None)
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "uci_har"],
                    help="e.g. --datasets mnist,cifar10  (or space-separated)")
    ap.add_argument("--schedules", nargs="+", default=["00", "01", "10", "11"],
                    help="e.g. --schedules 01,10  (two chars per schedule: s1 s2)")
    ap.add_argument("--seed", type=int, default=73, help="first seed")
    ap.add_argument("--num-seeds", type=int, default=1, help="seeds 73..73+n-1; report mean+-std")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, quick sanity check")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(DEFAULT_ROOT)
    # Normalise: accept "01,10", "01 10", "0110" or "0,1" regardless of shell splitting.
    datasets = [d for d in ",".join(args.datasets).replace(" ", ",").split(",") if d]
    digits = "".join(c for c in "".join(args.schedules) if c in "01")
    if len(digits) == 0 or len(digits) % 2 != 0:
        sys.exit(f"--schedules 格式不对: {args.schedules}  (应为 '01,10' 这种两位一组的 0/1)")
    schedules = [digits[i:i + 2] for i in range(0, len(digits), 2)]
    print(f"datasets={datasets}  schedules={schedules}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"root={root}  device={device}")
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    table: dict[str, dict[str, tuple[float, float]]] = {}
    for ds in datasets:
        cfg = CONFIG[ds]
        module = load_module(root / "model" / cfg["module"])
        train_loader, test_loader = module.load_data(cfg["bs"])
        prepare = make_prepare(ds, module)
        epochs = 1 if args.smoke else cfg["epochs"]
        print(f"\n### {ds}  (epochs={epochs}, seeds={seeds})")
        table[ds] = {}
        for s in schedules:
            a1 = "p2" if s[0] == "0" else "relu"
            a2 = "p2" if s[1] == "0" else "relu"
            accs = []
            for seed in seeds:
                torch.manual_seed(seed)
                net = cfg["cls"](a1=a1, a2=a2).to(device)
                acc = train_one(net, train_loader, test_loader, device, prepare,
                                epochs, cfg["lr"], cfg["wd"], cfg["opt"], cfg["cosine"])
                accs.append(acc)
                print(f"  s={s} ({a1}+{a2}) seed={seed}  acc={acc:.4f}")
                if not args.no_save and not args.smoke:
                    ckpt = root / "datas" / f"{ds}_hybrid_s{s}_seed{seed}_cnn_train.pt"
                    torch.save(net.state_dict(), ckpt)
                    p = save_record(root, ds, s, seed, acc, ckpt, epochs)
                    print(f"       -> {p.relative_to(root)}")
            mean, std = float(np.mean(accs)), float(np.std(accs))
            table[ds][s] = (mean, std)
            print(f"  s={s} ({a1}+{a2})  mean={mean*100:.2f} ± {std*100:.2f}")

    print("\n" + "=" * 78)
    print("汇总（精度 %, mean±std）")
    for ds, row in table.items():
        line = f"{ds:<10}"
        for s in schedules:
            m, sd = row[s]
            line += f"  s{s}={m*100:.2f}±{sd*100:.2f}"
        print(line)


if __name__ == "__main__":
    main()
