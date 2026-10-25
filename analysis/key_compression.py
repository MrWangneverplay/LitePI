"""E4: rotation-key pruning for S_key (protocol SSOT §6.4).

R = R_conv ∪ R_mm; R_mm is the input dimension of FC1 (MNIST 256, UCI-HAR 756,
CIFAR-10 1200).  SEAL realises every rotation from the Galois elements of
+-2^i, so a plan only needs i <= ceil(log2(|R_mm|)).  This script measures the
serialized key bytes for

    * the full key set (what new_context() creates by default), and
    * the pruned set the plan actually queries (max_rot_pow = ceil(log2 |R_mm|)),

and probes that pruning does not change the encrypted forward.

Run:
    python analysis/key_compression.py            # bytes + 3-sample MNIST probe
    python analysis/key_compression.py --no-probe # bytes only
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from eval_encrypted import _timings, load_test_set
from pi_proto import Wire, make_backend
from pi_server import encrypted_forward

# §6.4: R_mm is the FC1 input dimension of each canonical network.
R_MM = {"mnist": 256, "cifar10": 1200, "uci_har": 756}
FULL_POW = 12  # +-2^0 .. +-2^12 (the +-4096 element collapses: 24 keys)


def plan_pow(dim: int) -> int:
    return max(0, math.ceil(math.log2(max(1, dim))))


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 rotation-key pruning")
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--probe-samples", type=int, default=3)
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    be = make_backend("seal")

    curve = []
    for pow_i in range(0, FULL_POW + 1):
        ctx = be.seal.new_context(8192, [40] + [21] * 6 + [40], 21, max_rot_pow=pow_i)
        keys = be.seal.key_sizes(ctx)
        curve.append({
            "max_rot_pow": pow_i,
            "galois_elements": 2 * (pow_i + 1) + 1,  # +-2^i plus conjugation
            "galois_keys_bytes": int(keys["galois_keys_bytes"]),
            "s_key_bytes": int(keys["public_key_bytes"]) + int(keys["galois_keys_bytes"])
            + int(keys["relin_keys_bytes"]),
        })
    ctx = be.seal.new_context(8192, [40] + [21] * 6 + [40], 21, max_rot_pow=-1)
    full = be.seal.key_sizes(ctx)
    full_s_key = int(full["public_key_bytes"]) + int(full["galois_keys_bytes"]) \
        + int(full["relin_keys_bytes"])

    per_dataset = {}
    for ds, dim in R_MM.items():
        p = plan_pow(dim)
        row = next(r for r in curve if r["max_rot_pow"] == p)
        per_dataset[ds] = {
            "R_mm": dim,
            "max_rot_pow": p,
            "galois_elements": row["galois_elements"],
            "s_key_bytes": row["s_key_bytes"],
            "s_key_mb": round(row["s_key_bytes"] / 2**20, 3),
            "saving_vs_full_pct": round(100.0 * (1 - row["s_key_bytes"] / full_s_key), 2),
        }

    print("%8s %10s %16s %12s" % ("dataset", "R_mm", "pruned S_key", "vs full"))
    for ds, row in per_dataset.items():
        print("%8s %10d %13.1f MB %11.1f%%"
              % (ds, row["R_mm"], row["s_key_mb"], -row["saving_vs_full_pct"]))
    print("full key set: %.1f MB" % (full_s_key / 2**20))

    probe = None
    if not args.no_probe:
        # Pruning must not change the result: one MNIST forward with the pruned
        # key set, compared with the plaintext prediction of the same model.
        model = load_canonical_model(root, "mnist", (0, 0), args.seed, None)
        test = load_test_set(root, "mnist")
        be.max_rot_pow = plan_pow(R_MM["mnist"])
        ctx = be.new_context()
        left, right = socket.socketpair()
        wire = Wire(left, 1)
        hits, errors = 0, []
        t0 = time.perf_counter()
        for i in range(args.probe_samples):
            raw = np.asarray(test[i][0], dtype=np.float64).ravel()
            prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
            prepared = prepared.reshape(model.spec.model_shape)
            if prepared.ndim == 3 and prepared.shape[0] == 1:
                prepared = prepared[0]
            ct, _ = be.im2col_encrypt(ctx, prepared, model.im2col_kernel, model.stride)
            out = encrypted_forward(model, be, ctx, ct, wire, _timings())
            logits = be.decrypt(out)[: model.spec.classes]
            ref = model.plain_forward(raw)[0][: model.spec.classes]
            hits += int(np.argmax(logits) == np.argmax(ref))
            errors.append(float(np.max(np.abs(logits - ref))))
        probe = {
            "dataset": "mnist",
            "max_rot_pow": be.max_rot_pow,
            "samples": args.probe_samples,
            "prediction_matches": hits,
            "max_abs_logit_error": max(errors),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        print("probe: %d/%d 预测一致，最大 logit 误差 %.2e (%.1fs)"
              % (hits, args.probe_samples, probe["max_abs_logit_error"], probe["seconds"]))

    out_dir = root / "results" / "raw_results" / "E4" / "all" / "key_compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_id": "E4", "dataset": "all", "method": "key_compression", "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": f"seal {be.version}",
        "profile": "N=8192, [40,21x6,40], scale=2^21",
        "full_key_set": {
            "galois_elements": 24,
            "galois_keys_bytes": int(full["galois_keys_bytes"]),
            "public_key_bytes": int(full["public_key_bytes"]),
            "relin_keys_bytes": int(full["relin_keys_bytes"]),
            "s_key_bytes": full_s_key,
            "s_key_mb": round(full_s_key / 2**20, 3),
        },
        "curve": curve,
        "per_dataset": per_dataset,
        "probe": probe,
    }
    (out_dir / f"seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
