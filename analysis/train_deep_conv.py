"""Real conv-layer deepening for CIFAR-10 (VGG-lite, HE-friendly).

The FC-only deepen script (train_deep.py) plateaus around depth 3-4 because the
network still has a single conv layer.  This script deepens the CONV stack, which
is what the advisor meant by "把网络深度加深".

Architecture (32x32; grayscale by default, RGB with --rgb):
    [Conv(3x3,p1) -> (BN) -> act] x N, with AvgPool(2) after every 2 blocks
    -> Flatten -> Linear(hidden) -> act -> Linear(10)

Multiplicative depth (HE-only / all-P2 route):
    each conv = 1, each P2 activation = 1, final FC = 1
    => 1 + N(conv) + (N+1)(P2) = 2N + 2
    N=2 -> 6 (fits budget 6), N=3 -> 8, N=4 -> 10, N=6 -> 14 (all exceed)
BN folds into the conv and adds no depth.
This is the quantitative argument that deeper CONV nets force the Hybrid route.

Run:
    python analysis/train_deep_conv.py --shape-check --rgb --bn
    python analysis/train_deep_conv.py --conv-depth 2,3,4 --activation relu,p2 --rgb --bn --num-seeds 3
    python analysis/train_deep_conv.py --check-fold
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

A0, A1, A2 = 0.0625, 0.5, 0.5
BUDGET = 6  # multiplicative-depth budget of the current CKKS parameters


def act_relu(x):
    return torch.relu(x)


def act_p2(x):
    return A0 + A1 * x + A2 * x * x


ACTS = {"relu": act_relu, "p2": act_p2}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def prepare_rgb(batch: torch.Tensor) -> torch.Tensor:
    """CIFAR-10 RGB normalization (3 channels kept)."""
    mean = torch.tensor(CIFAR_MEAN, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD, dtype=batch.dtype, device=batch.device).view(1, 3, 1, 1)
    return (batch - mean) / std


class ConvBlock(nn.Module):
    """Conv3x3 -> optional BN.  BN folds into the conv at inference."""

    def __init__(self, in_ch: int, out_ch: int, use_bn: bool, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class DeepConvNet(nn.Module):
    """VGG-lite: `conv_depth` conv blocks (3x3,p1) + avg-pool every 2, FC head."""

    def __init__(self, conv_depth: int = 4, channels: int = 32, hidden: int = 256,
                 activation: str = "relu", classes: int = 10,
                 in_channels: int = 1, use_bn: bool = False):
        super().__init__()
        assert conv_depth >= 2
        self.conv_depth = conv_depth
        self.act = ACTS[activation]
        self.use_bn = use_bn
        blocks = []
        ch = in_channels
        for i in range(conv_depth):
            blocks.append(ConvBlock(ch, channels, use_bn))
            ch = channels
            if (i + 1) % 2 == 0:
                blocks.append(nn.AvgPool2d(2))
        self.features = nn.Sequential(*blocks)
        self.pools = conv_depth // 2
        self.spatial = 32 // (2 ** self.pools)
        flat = channels * self.spatial * self.spatial
        self.fc1 = nn.Linear(flat, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, ConvBlock):
                x = self.act(x)
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


class FastConvNet(nn.Module):
    """Early-downsampling VGG-lite: ~20.7k ReLU elements for C=32.

    Conv1 stride 2 -> 16x16, Conv2 stride 2 -> 8x8, then two 3x3 convs at
    8x8, pool to 4x4, FC head.  This keeps the accuracy in the 80-85% range
    while cutting the MPC element count about 8x versus the 4-conv VGG.
    """

    def __init__(self, conv_depth: int = 4, channels: int = 32, hidden: int = 256,
                 activation: str = "relu", classes: int = 10,
                 in_channels: int = 3, use_bn: bool = True):
        super().__init__()
        assert conv_depth == 4, "FastConvNet is a fixed 4-conv topology"
        self.act = ACTS[activation]
        c1 = channels
        c2 = channels * 2
        self.features = nn.Sequential(
            ConvBlock(in_channels, c1, use_bn, stride=2),  # 16x16
            ConvBlock(c1, c2, use_bn, stride=2),           # 8x8
            ConvBlock(c2, c2, use_bn),                     # 8x8
            ConvBlock(c2, c2, use_bn),                     # 8x8
        )
        self.pool = nn.AvgPool2d(2)                        # 4x4
        self.fc1 = nn.Linear(c2 * 4 * 4, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            if isinstance(layer, ConvBlock):
                x = self.act(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.act(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def fold_bn(model: nn.Module) -> nn.Module:
    """Fold every BN into its conv for HE inference.  Returns the same model."""
    blocks = [m for m in model.modules() if isinstance(m, ConvBlock)]
    for m in blocks:
        if m.bn is None:
            continue
        bn, conv = m.bn, m.conv
        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        conv.weight.mul_(scale.view(-1, 1, 1, 1))
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels,
                                                 device=conv.weight.device))
        conv.bias.copy_((conv.bias - bn.running_mean) * scale + bn.bias)
        m.bn = None
    return model


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def check_fold() -> None:
    """Sanity-check BN folding: outputs must match before and after folding."""
    net = DeepConvNet(conv_depth=2, channels=8, hidden=32, activation="relu",
                      in_channels=3, use_bn=True)
    net.train()
    with torch.no_grad():
        for _ in range(3):
            net(torch.randn(8, 3, 32, 32))
    net.eval()
    x = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        y0 = net(x)
        fold_bn(net)
        y1 = net(x)
    diff = (y0 - y1).abs().max().item()
    print(f"fold_bn max|diff| = {diff:.2e}")


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


def train_one(cfg, train_loader, test_loader, device, prepare,
              epochs, lr, wd, seed, label_smoothing=0.0, verbose=True):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cfg = dict(cfg)
    arch = cfg.pop("arch", "vgg")
    net = (FastConvNet(**cfg) if arch == "fast" else DeepConvNet(**cfg)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    test_acc = 0.0
    for ep in range(epochs):
        train_loss, train_acc = run_epoch(net, train_loader, criterion, device,
                                          prepare, optimizer)
        test_loss, test_acc = run_epoch(net, test_loader, criterion, device, prepare)
        scheduler.step()
        if verbose:
            print(f"  seed {seed} epoch {ep+1}/{epochs}  "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
                  f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}",
                  flush=True)
    return net, test_acc


def save_record(root: Path, tag: str, seed: int, record: dict) -> Path:
    out = root / "results" / "raw_results" / "E1" / "cifar10" / "deep" / tag
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def save_fast_weights(root: Path, net: FastConvNet, seed: int, test_acc: float,
                      activation: str, channels: int, hidden: int,
                      rgb: bool, use_bn: bool) -> Path | None:
    """Export a BN-folded fast CNN as portable NPZ weights for the Pi server."""
    if not isinstance(net, FastConvNet):
        print("weight export currently supports --arch fast only; skipped")
        return None

    net.eval()
    fold_bn(net)
    arrays = {}
    conv_specs = []
    conv_index = 0
    for layer in net.features:
        if not isinstance(layer, ConvBlock):
            continue
        conv = layer.conv
        if conv.bias is None:
            conv.bias = nn.Parameter(torch.zeros(conv.out_channels,
                                                 device=conv.weight.device))
        weight_key = f"conv{conv_index}_weight"
        bias_key = f"conv{conv_index}_bias"
        arrays[weight_key] = conv.weight.detach().cpu().numpy().astype(np.float32)
        arrays[bias_key] = conv.bias.detach().cpu().numpy().astype(np.float32)
        conv_specs.append({
            "weight_key": weight_key,
            "bias_key": bias_key,
            "stride": int(conv.stride[0]),
            "padding": int(conv.padding[0]),
        })
        conv_index += 1

    arrays["fc1_weight"] = net.fc1.weight.detach().cpu().numpy().astype(np.float32)
    arrays["fc1_bias"] = net.fc1.bias.detach().cpu().numpy().astype(np.float32)
    arrays["fc2_weight"] = net.fc2.weight.detach().cpu().numpy().astype(np.float32)
    arrays["fc2_bias"] = net.fc2.bias.detach().cpu().numpy().astype(np.float32)

    out_dir = root / "datas" / "weights"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"fast_{activation}_{'rgb' if rgb else 'gray'}_{'bn' if use_bn else 'nobn'}_seed{seed}"
    npz_path = out_dir / f"{tag}.npz"
    meta_path = out_dir / f"{tag}.json"
    np.savez_compressed(npz_path, **arrays)
    meta = {
        "arch": "fast",
        "activation": activation,
        "input_shape": [3 if rgb else 1, 32, 32],
        "channels": channels,
        "hidden": hidden,
        "use_bn": use_bn,
        "seed": seed,
        "test_acc": test_acc,
        "convs": conv_specs,
        "pool": "avg2",
        "fc1": {"weight_key": "fc1_weight", "bias_key": "fc1_bias"},
        "fc2": {"weight_key": "fc2_weight", "bias_key": "fc2_bias"},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    print(f"weights -> {npz_path}")
    print(f"metadata -> {meta_path}")
    return npz_path


def count_relu_elements(arch: str, conv_depth: int, channels: int, hidden: int) -> int:
    if arch == "fast":
        return channels * 16 * 16 + (channels * 2) * 8 * 8 * 3 + hidden
    spatial = 32
    total = 0
    for i in range(conv_depth):
        total += channels * spatial * spatial
        if (i + 1) % 2 == 0:
            spatial //= 2
    return total + hidden


def shape_check(channels: int, hidden: int, rgb: bool, use_bn: bool, arch: str = "vgg"):
    in_ch = 3 if rgb else 1
    if arch == "fast":
        net = FastConvNet(conv_depth=4, channels=channels, hidden=hidden,
                          activation="relu", in_channels=in_ch, use_bn=use_bn)
        net.eval()
        n_params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            out = net(torch.randn(1, in_ch, 32, 32))
        elems = count_relu_elements("fast", 4, channels, hidden)
        print(f"arch=fast  in_ch={in_ch}  bn={use_bn}  relu_elements={elems:,}  "
              f"mult_depth=10 (EXCEEDS)  out={tuple(out.shape)}  params={n_params:,}")
        return
    for n in (2, 3, 4, 6):
        net = DeepConvNet(conv_depth=n, channels=channels, hidden=hidden,
                          activation="relu", in_channels=in_ch, use_bn=use_bn)
        net.eval()  # BN with batch size 1 is only valid in eval mode
        n_params = sum(p.numel() for p in net.parameters())
        with torch.no_grad():
            out = net(torch.randn(1, in_ch, 32, 32))
        mdepth = 2 * n + 2
        print(f"conv_depth={n}  in_ch={in_ch}  bn={use_bn}  "
              f"spatial={net.spatial}x{net.spatial}  "
              f"mult_depth={mdepth} ({'OK' if mdepth <= BUDGET else 'EXCEEDS'})  "
              f"out={tuple(out.shape)}  params={n_params:,}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deepen the CIFAR-10 CONV stack")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--conv-depth", default="2,4,6", help="comma list of conv-layer counts")
    ap.add_argument("--activation", default="relu,p2")
    ap.add_argument("--arch", choices=["vgg", "fast"], default="vgg",
                    help="vgg = original pooling stack; fast = early downsampling")
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2,
                    help="DataLoader worker processes (0 disables multiprocessing)")
    ap.add_argument("--train-subset", type=int, default=0,
                    help="use only the first N training images (quick signal)")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--num-seeds", type=int, default=1)
    ap.add_argument("--rgb", action="store_true", help="keep 3-channel RGB input")
    ap.add_argument("--bn", action="store_true", help="add BatchNorm after each conv")
    ap.add_argument("--check-fold", action="store_true", help="verify BN folding")
    ap.add_argument("--smoke", action="store_true", help="1 epoch, quick sanity check")
    ap.add_argument("--shape-check", action="store_true", help="print shapes/params, no training")
    ap.add_argument("--save-weights", action="store_true",
                    help="export fast CNN weights as BN-folded NPZ for Pi")
    args = ap.parse_args()

    if args.check_fold:
        check_fold()
        return
    if args.shape_check:
        shape_check(args.channels, args.hidden, args.rgb, args.bn, args.arch)
        return

    root = Path(args.root)
    module = load_module(root / "model" / "cifar10_cnn_relu.py")
    train_loader, test_loader = module.load_data(args.batch_size, args.workers)
    if args.train_subset and args.train_subset < len(train_loader.dataset):
        from torch.utils.data import DataLoader, Subset
        subset = Subset(train_loader.dataset, list(range(args.train_subset)))
        train_loader = DataLoader(
            subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=torch.cuda.is_available(),
        )
    prepare = prepare_rgb if args.rgb else module.preprocess
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depths = [int(d) for d in args.conv_depth.split(",") if d.strip()]
    if args.arch == "fast":
        depths = [4]
    acts = [a.strip() for a in args.activation.split(",") if a.strip()]
    epochs = 1 if args.smoke else args.epochs
    seeds = list(range(args.seed, args.seed + args.num_seeds))
    in_ch = 3 if args.rgb else 1

    print(f"device={device}  arch={args.arch}  rgb={args.rgb}  bn={args.bn}  "
          f"channels={args.channels}  hidden={args.hidden}  "
          f"epochs={epochs}  seeds={seeds}")
    print("=" * 86)
    print(f"{'config':<26}{'mult depth':>11}{'fits 6?':>9}{'mean':>10}{'std':>9}")
    for act in acts:
        for n in depths:
            cfg = dict(conv_depth=n, channels=args.channels, hidden=args.hidden,
                       activation=act, in_channels=in_ch, use_bn=args.bn,
                       arch=args.arch)
            accs = []
            for seed in seeds:
                net, acc = train_one(cfg, train_loader, test_loader, device, prepare,
                                     epochs, args.lr, args.wd, seed,
                                     args.label_smoothing)
                accs.append(acc)
                if args.save_weights:
                    save_fast_weights(
                        root, net, seed, acc, act, args.channels, args.hidden,
                        args.rgb, args.bn,
                    )
                record = {
                    "exp_id": "E1",
                    "dataset": "cifar10",
                    "method": f"deep_conv_{act}",
                    "arch": args.arch,
                    "rgb": args.rgb,
                    "bn": args.bn,
                    "conv_depth": n,
                    "channels": args.channels,
                    "hidden": args.hidden,
                    "epochs": epochs,
                    "seed": seed,
                    "test_acc": acc,
                    "relu_elements": count_relu_elements(args.arch, n, args.channels,
                                                         args.hidden),
                    "mult_depth": 2 * n + 2,
                    "fits_budget": (2 * n + 2) <= BUDGET,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                }
                tag = (f"{act}_conv{n}"
                       + ("_rgb" if args.rgb else "")
                       + ("_bn" if args.bn else ""))
                save_record(root, tag, seed, record)
            mean = sum(accs) / len(accs)
            var = sum((a - mean) ** 2 for a in accs) / max(len(accs) - 1, 1)
            std = var ** 0.5
            mdepth = 2 * n + 2
            fits = "OK" if mdepth <= BUDGET else "EXCEEDS"
            name = (f"{act} x conv{n}"
                    + ("+rgb" if args.rgb else "")
                    + ("+bn" if args.bn else ""))
            print(f"{name:<26}{mdepth:>11}{fits:>9}{mean*100:>9.2f}%{std*100:>8.2f}")
    print("=" * 86)
    print(f"multiplicative depth = 2*N + 2; current CKKS budget = {BUDGET}")


if __name__ == "__main__":
    main()
