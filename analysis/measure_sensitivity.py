"""E5: per-layer ReLU -> P2 sensitivity for the fast CIFAR net.

Two modes:

  posthoc  train an all-ReLU baseline, then replace exactly one activation
           with P2 at inference (no retraining).  This reproduces the
           "post-hoc substitution collapse" effect.
  retrain  train a fresh model for each single-layer P2 variant, keeping the
           other layers ReLU.  These deltas are what the planner should use.

Both modes write JSON under results/raw_results/E1/cifar10/deep/sensitivity/.

Run:
    python analysis/measure_sensitivity.py --mode posthoc
    python analysis/measure_sensitivity.py --mode retrain --epochs 50
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from train_deep_conv import ACTS, ConvBlock, prepare_rgb, run_epoch


LAYERS = ("conv1", "conv2", "conv3", "conv4", "fc1")


class FastPerLayer(nn.Module):
    """Fast early-downsampling net with one activation slot per nonlinear layer."""

    def __init__(self, activations, channels: int = 32, hidden: int = 256,
                 in_channels: int = 3, use_bn: bool = True, classes: int = 10):
        super().__init__()
        assert len(activations) == 5
        self.acts = [ACTS[a] for a in activations]
        c1, c2 = channels, channels * 2
        self.conv1 = ConvBlock(in_channels, c1, use_bn, stride=2)
        self.conv2 = ConvBlock(c1, c2, use_bn, stride=2)
        self.conv3 = ConvBlock(c2, c2, use_bn)
        self.conv4 = ConvBlock(c2, c2, use_bn)
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(c2 * 4 * 4, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        x = self.acts[0](self.conv1(x))
        x = self.acts[1](self.conv2(x))
        x = self.acts[2](self.conv3(x))
        x = self.acts[3](self.conv4(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.acts[4](self.fc1(x))
        return self.fc2(x)


def load_data_module(root: Path):
    path = root / "model" / "cifar10_cnn_relu.py"
    if not path.is_file():
        raise FileNotFoundError(f"data loader not found: {path}")
    spec = importlib.util.spec_from_file_location("cifar_data", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load data module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["cifar_data"] = module
    spec.loader.exec_module(module)
    return module


def train_variant(activations, seed, args, train_loader, test_loader, device, prepare):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FastPerLayer(activations, channels=args.channels,
                         hidden=args.hidden).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    test_acc = 0.0
    for ep in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device,
                                          prepare, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, device, prepare)
        scheduler.step()
        print(f"  {activations} seed={seed} epoch {ep+1}/{args.epochs}  "
              f"train_acc={train_acc:.4f} test_acc={test_acc:.4f}", flush=True)
    return model, test_acc


def evaluate(model, loader, device, prepare, label_smoothing=0.0):
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    _, acc = run_epoch(model, loader, criterion, device, prepare)
    return acc


def save_json(root: Path, name: str, payload: dict) -> Path:
    out = root / "results" / "raw_results" / "E1" / "cifar10" / "deep" / "sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 layer sensitivity (ReLU -> P2)")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--mode", choices=["posthoc", "retrain", "both", "schedule"],
                    default="posthoc")
    ap.add_argument("--activations", default="relu,relu,relu,relu,relu",
                    help="comma list for conv1,conv2,conv3,conv4,fc1 (relu|p2)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--retrain-baseline", action="store_true")
    args = ap.parse_args()

    if args.epochs < 1:
        ap.error("--epochs must be >= 1")
    if args.batch_size < 1:
        ap.error("--batch-size must be >= 1")
    if args.workers < 0:
        ap.error("--workers must be >= 0")
    schedule_acts = None
    if args.mode == "schedule":
        schedule_acts = tuple(a.strip() for a in args.activations.split(","))
        if len(schedule_acts) != 5 or any(a not in ACTS for a in schedule_acts):
            ap.error("--activations must be five comma-separated relu|p2 values")

    root = Path(args.root)
    module = load_data_module(root)
    train_loader, test_loader = module.load_data(args.batch_size, args.workers)
    prepare = prepare_rgb
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} mode={args.mode} seed={args.seed}", flush=True)

    if args.mode == "schedule":
        acts = schedule_acts
        assert acts is not None
        print(f"training schedule {acts}", flush=True)
        _, acc = train_variant(acts, args.seed, args, train_loader, test_loader,
                               device, prepare)
        save_json(root, f"schedule_{'-'.join(acts)}_seed{args.seed}.json",
                  {"activations": list(acts), "acc": acc, "seed": args.seed,
                   "epochs": args.epochs})
        print(f"schedule {acts}: {acc:.2%}")
        return 0

    baseline_path = root / "datas" / "fast_relu_baseline.pt"
    baseline_acts = ("relu",) * 5
    if baseline_path.exists() and not args.retrain_baseline:
        model = FastPerLayer(baseline_acts, channels=args.channels,
                             hidden=args.hidden).to(device)
        model.load_state_dict(torch.load(baseline_path, map_location=device))
        base_acc = evaluate(model, test_loader, device, prepare, args.label_smoothing)
        print(f"baseline loaded: {base_acc:.2%}")
    else:
        print("training all-ReLU baseline", flush=True)
        model, base_acc = train_variant(baseline_acts, args.seed, args,
                                        train_loader, test_loader, device, prepare)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), baseline_path)
        print(f"baseline: {base_acc:.2%}")

    payload = {
        "exp_id": "E5",
        "dataset": "cifar10",
        "arch": "fast",
        "seed": args.seed,
        "epochs": args.epochs,
        "baseline_acc": base_acc,
        "posthoc": {},
        "retrain": {},
    }

    if args.mode in ("posthoc", "both"):
        print("\nposthoc single-layer substitution:")
        for i, name in enumerate(LAYERS):
            m = copy.deepcopy(model)
            m.acts[i] = ACTS["p2"]
            acc = evaluate(m, test_loader, device, prepare, args.label_smoothing)
            payload["posthoc"][name] = {"acc": acc, "delta": acc - base_acc}
            print(f"  {name:6s} acc={acc:.2%}  delta={acc - base_acc:+.2%}")
        save_json(root, f"posthoc_seed{args.seed}.json", payload)

    if args.mode in ("retrain", "both"):
        print("\nretrain single-layer P2 variants:")
        for i, name in enumerate(LAYERS):
            acts = ["relu"] * 5
            acts[i] = "p2"
            print(f"training {name}=P2", flush=True)
            _, acc = train_variant(tuple(acts), args.seed, args,
                                   train_loader, test_loader, device, prepare)
            payload["retrain"][name] = {"acc": acc, "delta": acc - base_acc}
            print(f"  {name:6s} acc={acc:.2%}  delta={acc - base_acc:+.2%}")
            save_json(root, f"retrain_{name}_seed{args.seed}.json",
                      {"layer": name, "acc": acc, "baseline_acc": base_acc,
                       "delta": acc - base_acc, "seed": args.seed,
                       "epochs": args.epochs})

    print("\nsummary:")
    for mode in ("posthoc", "retrain"):
        if payload[mode]:
            print(f"  {mode}:")
            for name in LAYERS:
                if name in payload[mode]:
                    v = payload[mode][name]
                    print(f"    {name:6s} {v['acc']:.2%}  ({v['delta']:+.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
