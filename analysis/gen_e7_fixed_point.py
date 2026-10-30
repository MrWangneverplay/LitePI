"""E7: how many fractional bits F do the hybrid shares need?

The share domain of LitePI only compares and selects: no fixed-point product,
so no truncation.  Its fixed-point error is the rounding of z to the 2^-F grid
when the client turns z + r into a ring share (decrypt_ring, or
sci_relu.quantize_to_ring).  This script measures that step:

    ReLU layer:  a = max(round(z * 2^F), 0) / 2^F      (r is on the same grid)
    P2 layer:    unchanged, it never leaves the ciphertext

on the full test set, per dataset / schedule / seed / F, against the float
forward of the same checkpoint.  CKKS decoding noise is not modelled, so these
are the rounding-only figures; the real protocol is checked per F with

    LITEPI_F=12 python eval_hybrid_accuracy.py --mask-exact --out-tag _f12 ...

(--mask-exact is required: a double-path mask of 2^42 has no room for F > 10).

The upper side of the trade-off comes from the ring: an output share pair lifts
to the wrong real sum with probability |y| * 2^(F - K_RING) per element, reported
as wrap_per_query.  Larger F costs share headroom, smaller F costs accuracy.

    python analysis/gen_e7_fixed_point.py
    python analysis/gen_e7_fixed_point.py --datasets mnist --seeds 73 --samples 500
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as TF

from canonical_net import load_canonical_model
from eval_hybrid_accuracy import _prepare, load_test_set
from params_pi import K_RING
from sci_relu import quantize_to_ring, ring_to_float

ROOT = Path(__file__).resolve().parent.parent
A0, A1, A2 = 0.0625, 0.5, 0.5


def check_against_real_path(f: int, n: int = 2000) -> None:
    """The closed form above must match the conversions the protocol runs."""
    rng = np.random.default_rng(f)
    z = rng.normal(0, 3, n)
    r_int = rng.integers(-2**30, 2**30, n)
    r = r_int / 2.0**f
    client = quantize_to_ring((z + r).tolist(), f, K_RING).astype(object)
    server = quantize_to_ring((-r).tolist(), f, K_RING).astype(object)
    mod = 1 << K_RING
    z_ring = ring_to_float((client + server) % mod, f, K_RING)
    y = np.maximum(z_ring, 0.0)
    want = np.maximum(np.rint(z * 2.0**f), 0.0) / 2.0**f
    if np.max(np.abs(y - want)) > 2.0 ** -(f + 1) + 1e-9:
        raise SystemExit(f"F={f}: closed form differs from the real conversions")


def forward(model, x: torch.Tensor, f: int | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched canonical forward; f=None is the float reference.

    Returns (logits, sum of |ReLU output| per sample over the hybrid layers).
    """
    def act(z, bit, total):
        if not bit:
            return A0 + A1 * z + A2 * z * z, total
        q = z if f is None else torch.round(z * 2.0**f) / 2.0**f
        y = torch.relu(q)
        return y, total + y.flatten(1).sum(1)

    total = torch.zeros(x.shape[0], dtype=torch.float64)
    z1 = TF.conv2d(x, model.conv_weight, model.conv_bias, stride=model.spec.stride)
    a1, total = act(z1, model.schedule[0], total)
    z2 = TF.linear(a1.flatten(1), model.fc1_weight, model.fc1_bias)
    a2, total = act(z2, model.schedule[1], total)
    return TF.linear(a2, model.fc2_weight, model.fc2_bias), total


def load_inputs(root: Path, dataset: str, model, samples: int):
    test = load_test_set(root, dataset)
    n = len(test) if not samples else min(samples, len(test))
    xs, ys = [], []
    for i in range(n):
        sample, label = test[i]
        xs.append(_prepare(model, dataset, sample).reshape(model.spec.model_shape))
        ys.append(int(label))
    return torch.from_numpy(np.stack(xs)).to(torch.float64), torch.tensor(ys)


def evaluate(model, x, labels, fs, batch: int) -> list[dict]:
    refs = torch.cat([forward(model, x[i:i + batch], None)[0]
                      for i in range(0, x.shape[0], batch)])
    ref_pred = refs.argmax(1)
    rows = []
    for f in fs:
        logits, mass = [], []
        for i in range(0, x.shape[0], batch):
            out, total = forward(model, x[i:i + batch], f)
            logits.append(out)
            mass.append(total)
        logits = torch.cat(logits)
        pred = logits.argmax(1)
        err = (logits - refs).abs()
        rows.append({
            "F": f,
            "lsb": 2.0 ** -f,
            "agreement": float((pred == ref_pred).double().mean()),
            "accuracy": float((pred == labels).double().mean()),
            "float_accuracy": float((ref_pred == labels).double().mean()),
            "logit_err_mean": float(err.mean()),
            "logit_err_max": float(err.max()),
            "wrap_per_query": float(torch.cat(mass).mean()) * 2.0 ** (f - K_RING),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--schedules", default="01,10,11", help="00 has no shares")
    ap.add_argument("--seeds", default="73,74,75,76,77")
    ap.add_argument("--fs", default="4,6,8,10,12")
    ap.add_argument("--samples", type=int, default=0, help="0 = full test set")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    fs = [int(v) for v in args.fs.split(",")]
    for f in fs:
        check_against_real_path(f)
    print(f"closed form matches the protocol conversions for F={fs}")
    rows = []
    for dataset in args.datasets.split(","):
        x = labels = None
        for tag in args.schedules.split(","):
            schedule = tuple(int(b) for b in tag)
            for seed in (int(v) for v in args.seeds.split(",")):
                model = load_canonical_model(args.root, dataset, schedule, seed)
                if x is None:
                    x, labels = load_inputs(args.root, dataset, model, args.samples)
                for row in evaluate(model, x, labels, fs, args.batch):
                    rows.append({"dataset": dataset, "schedule": tag, "seed": seed,
                                 "samples": int(x.shape[0]), **row})
    out = args.out or args.root / "results" / "E7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e7_fixed_point.json").write_text(json.dumps({
        "exp_id": "E7", "title": "fixed-point precision of the hybrid shares",
        "k_ring": K_RING, "ckks_noise": "not modelled (rounding only)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    with (out / "e7_fixed_point.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{'dataset':<9}{'F':>3}{'agreement':>11}{'acc':>9}{'float acc':>11}"
          f"{'err mean':>11}{'err max':>10}{'wrap/query':>12}")
    for dataset in args.datasets.split(","):
        for f in fs:
            sel = [r for r in rows if r["dataset"] == dataset and r["F"] == f]
            mean = lambda key: float(np.mean([r[key] for r in sel]))
            print(f"{dataset:<9}{f:>3}{100 * mean('agreement'):>10.2f}%"
                  f"{100 * mean('accuracy'):>8.2f}%{100 * mean('float_accuracy'):>10.2f}%"
                  f"{mean('logit_err_mean'):>11.2e}{max(r['logit_err_max'] for r in sel):>10.3f}"
                  f"{mean('wrap_per_query'):>12.2e}")
    print(f"-> {out / 'e7_fixed_point.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
