"""Export M2 per-layer mask domains from the canonical training split.

The protocol samples the mask for nonlinear layer k from a domain covering the
input distribution z_k.  This script records those distributions from the same
canonical architecture and checkpoint used by ``pi_server.py``.

Run:
    python analysis/export_mask_domains.py --dataset cifar10 --schedule 11
    python analysis/export_mask_domains.py --dataset mnist --schedule 01 --samples 4096
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

from canonical_net import CANONICAL_SPECS, load_canonical_model


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def _prepare_for_dataset(dataset: str, module, inputs: torch.Tensor) -> torch.Tensor:
    if dataset == "cifar10":
        return module.preprocess(inputs)
    if dataset == "uci_har":
        return inputs.unsqueeze(1)
    return inputs


def _distribution(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64).ravel()
    lo, hi = float(np.min(values)), float(np.max(values))
    span = max(hi - lo, 1e-12)
    margin = max(span * 0.05, 1e-6)
    return {
        "samples": int(values.size),
        "min": lo,
        "max": hi,
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "q_low": float(np.quantile(values, 0.001)),
        "q_high": float(np.quantile(values, 0.999)),
        "margin": margin,
        "lo": lo - margin,
        "hi": hi + margin,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Export LitePI M2 mask domains")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--dataset", choices=sorted(CANONICAL_SPECS), required=True)
    ap.add_argument("--schedule", default="11",
                    help="two canonical activation bits, e.g. 00/01/10/11")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if len(args.schedule) != 2 or set(args.schedule) - set("01"):
        ap.error("--schedule must be exactly two bits")
    if args.samples < 1:
        ap.error("--samples must be positive")
    schedule = tuple(int(bit) for bit in args.schedule)
    model = load_canonical_model(
        args.root, args.dataset, schedule, args.seed, args.checkpoint,
    )

    module = _load_module(args.root / "model" / f"{args.dataset}_cnn_relu.py")
    train_loader, _ = module.load_data(args.batch_size)
    zs: list[list[np.ndarray]] = [[], []]
    logits: list[np.ndarray] = []
    seen = 0
    with torch.no_grad():
        for inputs, _ in train_loader:
            prepared = _prepare_for_dataset(args.dataset, module, inputs)
            x = prepared.to(torch.float64)
            z1 = torch.nn.functional.conv2d(
                x, model.conv_weight, model.conv_bias, stride=model.spec.stride,
            )
            a1 = (
                torch.relu(z1) if schedule[0]
                else 0.0625 + 0.5 * z1 + 0.5 * z1 * z1
            )
            z2 = torch.nn.functional.linear(
                a1.flatten(1), model.fc1_weight, model.fc1_bias,
            )
            a2 = (
                torch.relu(z2) if schedule[1]
                else 0.0625 + 0.5 * z2 + 0.5 * z2 * z2
            )
            logits_t = torch.nn.functional.linear(
                a2, model.fc2_weight, model.fc2_bias,
            )
            zs[0].append(z1.flatten(1).numpy())
            zs[1].append(z2.numpy())
            logits.append(logits_t.numpy())
            seen += int(x.shape[0])
            if seen >= args.samples:
                break

    take = args.samples
    layer_stats = {
        str(i): _distribution(np.vstack(zs[i])[:take])
        for i in range(2)
    }
    output = _distribution(np.vstack(logits)[:take])
    tag = "".join(str(bit) for bit in schedule)
    out = args.out or (
        args.root / "datas" / "mask_domains"
        / f"{args.dataset}_s{tag}_seed{args.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "schedule": list(schedule),
        "seed": args.seed,
        "checkpoint": str(model.checkpoint),
        "split": "train",
        "samples": int(min(seen, args.samples)),
        "nonlinear_dims": [model.spec.conv_flat, model.spec.hidden],
        "classes": model.spec.classes,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "layers": layer_stats,
        "output": output,
        "rule": "M2: lo/hi cover full observed train min/max plus 5% margin",
    }
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"M2 domains -> {out}  samples={payload['samples']}  "
        f"z1={layer_stats['0']['min']:.3f}..{layer_stats['0']['max']:.3f}  "
        f"z2={layer_stats['1']['min']:.3f}..{layer_stats['1']['max']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
