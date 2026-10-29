"""Measure how large an M2 mask may be before CKKS decoding breaks the shares.

This is the evidence behind params_pi.MASK_MAX_BITS.  For each mask
magnitude A = 2^a it runs the client-visible part of one hybrid layer on the
real SEAL backend (protocol §4.1 steps 1, 2, 5):

    step 1  ct_{z+r} = ct_z + Enc_pk(r)       (ct_z at the level the layer sees)
    step 2  s_c = Dec(ct_{z+r}),  s_s = -r      -> ring shares mod 2^K_RING
    (MPC)   ReLU on the reconstructed ring value, fresh random output split
    step 5  Enc(c_c) + Enc(c_s) -> Dec            (re-encryption path)

and reports, in units of the fixed-point LSB 2^-F,

    dec_err   max |Dec(ct_{z+r}) - (z+r)|
    share_err max |ring(s_c + s_s) - round(z * 2^F)|      (input shares)
    relu_err  max |Dec(Enc(c_c)+Enc(c_s)) - ReLU(z)|     (output path)

PASS criterion: share_err <= 4 LSB, the CKKS noise floor measured at small A.

    python analysis/check_mask_domain.py                  # a = 20..52
    python analysis/check_mask_domain.py --out results/checks/mask_precision.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from params_pi import F, K_RING, MASK_MAX_BITS
from pi_proto import make_backend


def _ring(values: np.ndarray) -> np.ndarray:
    """float -> int64 ring element mod 2^K_RING (two's complement held in int)."""
    mod = 1 << K_RING
    return np.array([int(round(float(v) * (1 << F))) % mod for v in values],
                    dtype=object)


def _centered(values) -> np.ndarray:
    mod, half = 1 << K_RING, 1 << (K_RING - 1)
    return np.array([(int(v) % mod) - mod if (int(v) % mod) >= half
                     else int(v) % mod for v in values], dtype=object)


def run(backend: str, bits_list, n: int, z_bound: float, drops: int,
        seed: int) -> list[dict]:
    be = make_backend(backend)
    ctx = be.new_context()
    rng = np.random.default_rng(seed)
    z = rng.uniform(-z_bound, z_bound, n)
    lsb = 2.0 ** -F
    rows = []
    for a in bits_list:
        half = 2.0 ** a
        r = rng.integers(-int(half * (1 << F)), int(half * (1 << F)),
                         n, dtype=np.int64) / float(1 << F)
        ct = be.encrypt(ctx, z)
        for _ in range(drops):                      # reach the hybrid level
            ct = be.mul_plain(ct, np.ones(n))
        ct_zr = be.add_cipher(ct, be.encrypt_like(ctx, r, ct))
        s_c = be.decrypt(ct_zr)
        dec_err = float(np.max(np.abs(s_c - (z + r))))

        recon = _centered(_ring(s_c) + _ring(-r))
        want = np.array([int(round(v * (1 << F))) for v in z], dtype=object)
        share_err = max(abs(int(a_) - int(b_)) for a_, b_ in zip(recon, want))

        # Output path: ReLU on the ring value, random split, re-encrypt both.
        y = np.maximum(np.array([int(v) for v in recon], dtype=np.int64), 0)
        mod = 1 << K_RING
        c_c = np.array([int(v) for v in
                        rng.integers(0, mod, n, dtype=np.uint64)], dtype=object)
        c_s = np.array([(int(yy) - int(cc)) % mod for yy, cc in zip(y, c_c)],
                       dtype=object)
        to_f = lambda arr: np.array([float(v) for v in _centered(arr)]) / (1 << F)
        ct_out = be.add_cipher(be.encrypt(ctx, to_f(c_c)),
                               be.encrypt(ctx, to_f(c_s)))
        relu_err = float(np.max(np.abs(be.decrypt(ct_out) - np.maximum(z, 0))))
        rows.append({
            "log2_mask_half_width": a,
            "dec_err": dec_err,
            "dec_err_lsb": dec_err / lsb,
            "share_err_lsb": int(share_err),
            "relu_err_lsb": relu_err / lsb,
            "pass": share_err <= 4,
        })
        print(f"a={a:2d}  dec_err={dec_err:9.3e} ({dec_err/lsb:8.2f} LSB)  "
              f"share_err={share_err:6d} LSB  relu_err={relu_err/lsb:7.2f} LSB  "
              f"{'PASS' if share_err <= 4 else 'FAIL'}")
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="M2 mask-width precision check")
    ap.add_argument("--backend", default="seal", choices=["seal", "tenseal"])
    ap.add_argument("--bits", default="20,30,36,40,42,44,46,48,52")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--z-bound", type=float, default=128.0)
    ap.add_argument("--drops", type=int, default=3,
                    help="rescales before the hybrid layer (L7 -> L4 = 3)")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bits = [int(b) for b in args.bits.split(",")]
    rows = run(args.backend, bits, args.n, args.z_bound, args.drops, args.seed)
    passing = [r["log2_mask_half_width"] for r in rows if r["pass"]]
    largest = max(passing) if passing else None
    verdict = {
        "largest_passing_log2_A": largest,
        "configured_MASK_MAX_BITS": MASK_MAX_BITS,
        "cap_is_safe": largest is not None and MASK_MAX_BITS <= largest,
    }
    print(json.dumps(verdict))
    if args.out:
        try:
            sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
        except OSError:
            sha = "unknown"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "check": "M2 mask width vs CKKS decode precision",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": sha,
            "backend": args.backend,
            "f": F, "k_ring": K_RING, "n": args.n, "z_bound": args.z_bound,
            "rescales_before_layer": args.drops,
            "rows": rows,
            **verdict,
        }, indent=2) + "\n", encoding="utf-8")
    return 0 if verdict["cap_is_safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
