"""Train the alignment model B (plan §2.1, conv3) and export it for the protocol.

    Conv(3->c1, 3x3, s2) - s1 - Conv(c1->c2, 3x3, s2) - s2
        - Conv(c2->c2, 3x3, s1) - s3 - AvgPool(2)
        - FC(c2*4*4 -> hidden) - s4 - FC(hidden -> 10)

Each bit of the schedule picks the activation the network is trained with:
1 = ReLU (Hybrid-ReLU on ciphertext), 0 = P2 (HE-poly).  The network is
retrained per schedule, as for the canonical nets (train_hybrid.py).
A ciphertext schedule must fit the depth budget of 6: 0000 (9 levels), 0001 and
1000 (7) do not, the others stay within 5.  0000 is trained as the plaintext
all-P2 control of the accuracy-versus-K_h curve; the main schedule is 0100.

Writes, for schedule s and seed n:
    datas/weights/mid_s<s>_seed<n>.npz + .json     deep_net.from_npz
    datas/mask_domains/mid_s<s>_seed<n>.json       rule-v2 M2 domains (train split)
    results/raw_results/E1/cifar10/mid/s<s>/seed<n>.json

BN is folded into the convs before export, so the protocol sees plain convs.
The exported model is checked against the PyTorch net on one test batch.

    python analysis/train_mid.py --schedules 0000,0100,0110,1110,1111 --num-seeds 3
    python analysis/train_mid.py --schedules all --num-seeds 3     # planner input
    python analysis/train_mid.py --schedules 0100 --smoke --train-subset 512
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from deep_net import from_npz
from export_mask_domains import _distribution
from mask_session import RULE_V2, MaskDomain, mask_security_report
from train_deep_conv import (
    CIFAR_MEAN,
    CIFAR_STD,
    ConvBlock,
    act_p2,
    act_relu,
    fold_bn,
    load_module,
    prepare_rgb,
    run_epoch,
)

ROOT = Path(__file__).resolve().parent.parent


class MidConvNet(nn.Module):
    def __init__(self, schedule, c1: int = 32, c2: int = 64, hidden: int = 256,
                 classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.schedule = tuple(schedule)
        if len(self.schedule) != 4:
            raise ValueError(f"model B has 4 activations, schedule is {self.schedule}")
        self.acts = [act_relu if b else act_p2 for b in self.schedule]
        self.conv1 = ConvBlock(3, c1, use_bn, stride=2)     # 16x16
        self.conv2 = ConvBlock(c1, c2, use_bn, stride=2)    # 8x8
        self.conv3 = ConvBlock(c2, c2, use_bn)              # 8x8
        self.pool = nn.AvgPool2d(2)                         # 4x4
        self.fc1 = nn.Linear(c2 * 4 * 4, hidden)
        self.fc2 = nn.Linear(hidden, classes)

    def forward(self, x, capture: list | None = None):
        z = self.conv1(x)
        if capture is not None:
            capture.append(z.flatten(1))
        z = self.conv2(self.acts[0](z))
        if capture is not None:
            capture.append(z.flatten(1))
        z = self.conv3(self.acts[1](z))
        if capture is not None:
            capture.append(z.flatten(1))
        z = self.fc1(self.pool(self.acts[2](z)).flatten(1))
        if capture is not None:
            capture.append(z)
        return self.fc2(self.acts[3](z))


def export(root: Path, net: MidConvNet, schedule, seed: int, acc: float,
           train_loader, device, samples: int) -> tuple[Path, Path]:
    net.eval()
    fold_bn(net)
    tag = f"mid_s{''.join(map(str, schedule))}_seed{seed}"
    arrays, ops = {}, []
    for name, block in (("conv1", net.conv1), ("conv2", net.conv2), ("conv3", net.conv3)):
        conv = block.conv
        arrays[f"{name}_weight"] = conv.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"{name}_bias"] = conv.bias.detach().cpu().numpy().astype(np.float64)
        ops.append({"op": "conv", "weight_key": f"{name}_weight",
                    "bias_key": f"{name}_bias", "stride": int(conv.stride[0]),
                    "pad": int(conv.padding[0])})
        ops.append({"op": "act"})
    ops.append({"op": "pool", "k": 2})
    for name, fc in (("fc1", net.fc1), ("fc2", net.fc2)):
        arrays[f"{name}_weight"] = fc.weight.detach().cpu().numpy().astype(np.float64)
        arrays[f"{name}_bias"] = fc.bias.detach().cpu().numpy().astype(np.float64)
        ops.append({"op": "fc", "weight_key": f"{name}_weight",
                    "bias_key": f"{name}_bias"})
        ops.append({"op": "act"})
    ops.pop()                                   # no activation after the head
    weights_dir = root / "datas" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    npz = weights_dir / f"{tag}.npz"
    np.savez_compressed(npz, **arrays)
    npz.with_suffix(".json").write_text(json.dumps({
        "arch": "mid", "dataset": "cifar10_mid", "schedule": list(schedule),
        "seed": seed, "test_acc": acc, "input_shape": [3, 32, 32], "ops": ops,
        "preprocess": {
            "mode": "per_channel_zscore", "raw_shape": [3, 32, 32],
            "model_shape": [3, 32, 32],
            "mean": [[[float(m)]] for m in CIFAR_MEAN],
            "std": [[[float(s)]] for s in CIFAR_STD],
        },
    }, indent=2) + "\n", encoding="utf-8")

    # M2 statistics on the train split.
    zs: list[list[np.ndarray]] = [[] for _ in schedule]
    logits, seen = [], 0
    with torch.no_grad():
        for inputs, _ in train_loader:
            capture: list = []
            out = net(prepare_rgb(inputs.to(device)), capture)
            for i, z in enumerate(capture):
                zs[i].append(z.cpu().numpy())
            logits.append(out.cpu().numpy())
            seen += inputs.shape[0]
            if seen >= samples:
                break
    layers = {str(i): _distribution(np.vstack(z)) for i, z in enumerate(zs)}
    output = _distribution(np.vstack(logits))
    dims = [int(np.vstack(z).shape[1]) for z in zs]
    payload = {
        "dataset": "cifar10_mid", "schedule": list(schedule), "seed": seed,
        "checkpoint": str(npz), "split": "train", "samples": int(seen),
        "nonlinear_dims": dims, "classes": 10,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": layers, "output": output, "rule_version": RULE_V2,
        "rule": "M2 v2 export; pi_server --mask-exact widens it to v3",
    }
    domains = {int(k): MaskDomain.from_mapping(v) for k, v in layers.items()}
    payload["security"] = mask_security_report(
        domains, {i: n for i, (n, b) in enumerate(zip(dims, schedule)) if b})
    dom_path = root / "datas" / "mask_domains" / f"{tag}.json"
    dom_path.parent.mkdir(parents=True, exist_ok=True)
    dom_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    return npz, dom_path


def check_export(npz: Path, net: MidConvNet, schedule, test_loader, device) -> float:
    """Max |logit difference| between deep_net and the PyTorch net."""
    model = from_npz(npz, schedule)
    images, _ = next(iter(test_loader))
    images = images[:8]
    with torch.no_grad():
        want = net(prepare_rgb(images.to(device))).cpu().numpy()
    mean = np.array(CIFAR_MEAN).reshape(3, 1, 1)
    std = np.array(CIFAR_STD).reshape(3, 1, 1)
    worst = 0.0
    for img, ref in zip(images.numpy().astype(np.float64), want):
        got, _ = model.plain_forward((img - mean) / std)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return worst


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--schedules", default="0000,0100,0110,1110,1111")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--num-seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--train-subset", type=int, default=0)
    ap.add_argument("--domain-samples", type=int, default=20000)
    ap.add_argument("--smoke", action="store_true", help="1 epoch")
    args = ap.parse_args()

    module = load_module(args.root / "model" / "cifar10_cnn_relu.py")
    train_loader, test_loader = module.load_data(args.batch_size, args.workers)
    if args.train_subset:
        from torch.utils.data import DataLoader, Subset
        train_loader = DataLoader(
            Subset(train_loader.dataset, list(range(args.train_subset))),
            batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 1 if args.smoke else args.epochs
    print(f"device={device} epochs={epochs}")
    tags = args.schedules.split(",")
    if args.schedules == "all":          # every schedule of the 2^K planner search
        tags = [f"{i:04b}" for i in range(16)]
    for tag in tags:
        schedule = tuple(int(b) for b in tag)
        accs = []
        for seed in range(args.seed, args.seed + args.num_seeds):
            torch.manual_seed(seed)
            net = MidConvNet(schedule).to(device)
            criterion = nn.CrossEntropyLoss()
            opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            acc = 0.0
            for ep in range(epochs):
                loss, tr = run_epoch(net, train_loader, criterion, device,
                                     prepare_rgb, opt)
                _, acc = run_epoch(net, test_loader, criterion, device, prepare_rgb)
                sched.step()
                print(f"  s{tag} seed {seed} epoch {ep + 1}/{epochs} "
                      f"loss={loss:.4f} train={tr:.4f} test={acc:.4f}", flush=True)
            npz, dom = export(args.root, net, schedule, seed, acc, train_loader,
                              device, args.domain_samples)
            err = check_export(npz, net, schedule, test_loader, device)
            if err > 1e-3:
                raise SystemExit(f"export check failed: logit error {err}")
            accs.append(acc)
            out = (args.root / "results" / "raw_results" / "E1" / "cifar10" / "mid"
                   / f"s{tag}")
            out.mkdir(parents=True, exist_ok=True)
            (out / f"seed{seed}.json").write_text(json.dumps({
                "exp_id": "E1", "dataset": "cifar10", "method": "mid_alignment",
                "schedule": tag, "seed": seed, "epochs": epochs,
                "test_acc": acc, "export_logit_err": err,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }, indent=2) + "\n", encoding="utf-8")
            print(f"  -> {npz.name}, {dom.name}, export err {err:.1e}")
        print(f"s{tag}: {100 * float(np.mean(accs)):.2f} +/- "
              f"{100 * float(np.std(accs)):.2f} %  ({len(accs)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
