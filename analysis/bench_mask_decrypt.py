"""E2 (M2 v3): client-side decrypt cost per hybrid layer on one device.

Why this exists: under `--mask-exact` the client turns Enc(z+r) into a ring share
with `decrypt_ring` (double-double decode + integer ring), while the s=00 path
and the output logits keep using the ordinary SEAL `decrypt`.  E2's T_dec is the
output-logits decrypt and does not change, so the number that has to be re-measured
is the per-hybrid-layer decrypt.

The ciphertext is built exactly as the server sends it: Enc(z), `--drops` rescales
(the hybrid layer sits at L7 -> L4), then Enc(r) added in double-double.

    python analysis/bench_mask_decrypt.py --reps 20 --drops 3

Output: results/raw_results/E2_pi5_maskv3/bench_mask_decrypt.json (device-tagged).
"""

from __future__ import annotations

import argparse
import json
import platform
import secrets
import statistics
import time
from pathlib import Path

import numpy as np

from params_pi import F, K_RING, MASK_SIGMA_EXACT, POLY_MODULUS_DEGREE
from pi_proto import make_backend
from sci_relu import ring_to_float


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean_ms": None, "std_ms": None,
                "min_ms": None, "max_ms": None}
    return {
        "n": len(vals),
        "mean_ms": statistics.fmean(vals) * 1e3,
        "std_ms": statistics.stdev(vals) * 1e3 if len(vals) > 1 else 0.0,
        "min_ms": min(vals) * 1e3,
        "max_ms": max(vals) * 1e3,
    }


def timed(fn, reps: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def one_layer(be, ctx, dims: int, bits: int, drops: int, reps: int,
              warmup: int, rng) -> dict:
    z = rng.uniform(-1.0, 1.0, dims)
    ct = be.encrypt(ctx, z)
    for _ in range(drops):
        ct = be.mul_plain(ct, np.ones(dims))
    span = 1 << (bits + 1 + F)
    k = [secrets.randbelow(span) - (span >> 1) for _ in range(dims)]
    ct_zr = be.add_cipher(ct, be.encrypt_like_exact(ctx, k, ct))

    double_s = timed(lambda: be.decrypt(ct_zr), reps, warmup)
    ring_s = timed(lambda: ring_to_float(be.decrypt_ring(ct_zr, F, K_RING), F, K_RING),
                   reps, warmup)

    # share error: ring(decrypt(z+r)) - r must recover round(z * 2^F)
    share = be.decrypt_ring(ct_zr, F, K_RING)
    mod, half = 1 << K_RING, 1 << (K_RING - 1)
    want = [int(round(float(v) * (1 << F))) for v in z]
    err = 0
    for s_c, r, w in zip(share, k, want):
        v = (int(s_c) - r) % mod
        v = v - mod if v >= half else v
        err = max(err, abs(v - w))
    err_lsb = int(err)

    return {
        "slots": dims,
        "mask_bits": bits,
        "drops": drops,
        "modulus_bits": be.modulus_bits(ct_zr),
        "scale_log2": float(np.log2(be.scale_of(ct_zr))),
        "decrypt_double": summarize(double_s),
        "decrypt_ring": summarize(ring_s),
        "ratio_ring_over_double": statistics.fmean(ring_s) / statistics.fmean(double_s),
        "share_err_lsb": err_lsb,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 mask-exact client decrypt bench")
    ap.add_argument("--dims", default="256,64,1200,128,756")
    ap.add_argument("--bits", type=int, default=MASK_SIGMA_EXACT)
    ap.add_argument("--drops", type=int, default=3)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--classes", type=int, default=10)
    ap.add_argument("--out", default="results/raw_results/E2_pi5_maskv3/bench_mask_decrypt.json")
    ap.add_argument("--tag", default="pi5")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    rng = np.random.default_rng(73)
    be = make_backend("seal")
    ctx = be.new_context()

    rows = [one_layer(be, ctx, int(d), args.bits, args.drops, args.reps,
                      args.warmup, rng)
            for d in args.dims.split(",")]

    # E2 T_dec: the output logits ciphertext (10 slots, L2 after fc2).
    logits_ct = be.encrypt(ctx, rng.uniform(-1.0, 1.0, args.classes))
    for _ in range(5):
        logits_ct = be.mul_plain(logits_ct, np.ones(args.classes))
    logits_s = timed(lambda: be.decrypt(logits_ct), args.reps, args.warmup)
    logits_dd_s = timed(lambda: ctx.decrypt_dd(logits_ct)[0], args.reps, args.warmup) \
        if hasattr(ctx, "decrypt_dd") else []

    payload = {
        "experiment": "E2",
        "tier": "client decrypt under M2 v3",
        "tag": args.tag,
        "device": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "seal_version": getattr(be, "version", None),
        "profile": {
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "k_ring": K_RING, "f": F, "mask_bits": args.bits,
            "drops": args.drops, "reps": args.reps, "warmup": args.warmup,
        },
        "rows": rows,
        "output_logits": {
            "slots": args.classes,
            "drops": 5,
            "modulus_bits": be.modulus_bits(logits_ct),
            "decrypt_double": summarize(logits_s),
            "decrypt_dd": summarize(logits_dd_s),
        },
        "note": ("decrypt_ring includes ring_to_float, as in pi_client.ciphertext_loop; "
                 "decrypt_double is the legacy path and stays in use for s=00 output"),
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    for r in rows:
        print(f"slots {r['slots']:5d}  mod {r['modulus_bits']} bit  "
              f"double {r['decrypt_double']['mean_ms']:.3f} ms  "
              f"ring {r['decrypt_ring']['mean_ms']:.3f} ms  "
              f"x{r['ratio_ring_over_double']:.2f}  err {r['share_err_lsb']} LSB")
    print(f"logits {args.classes} slots: double "
          f"{payload['output_logits']['decrypt_double']['mean_ms']:.3f} ms")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
