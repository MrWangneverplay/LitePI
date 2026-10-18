"""Check the CKKS noise budget at the network's multiplicative depth.

The hybrid_mpc self-test validates the mask/conversion path but does not chain
six ciphertext multiplications.  This script squares a small vector `depth`
times, decrypts, and compares with the float64 reference.

Run:
    python analysis/check_ckks_depth.py
    python analysis/check_ckks_depth.py --depth 6 --size 16
"""

from __future__ import annotations

import argparse

import numpy as np

try:
    import tenseal as ts
except Exception:
    ts = None

from params_b import COEFF_MOD_BIT_SIZES, GLOBAL_SCALE, POLY_MODULUS_DEGREE


def main() -> int:
    ap = argparse.ArgumentParser(description="CKKS depth/noise check")
    ap.add_argument("--depth", type=int, default=6,
                    help="number of ciphertext multiplications (default: 6)")
    ap.add_argument("--size", type=int, default=16)
    args = ap.parse_args()

    if ts is None:
        print("TenSEAL is not installed in this interpreter.")
        return 1

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.auto_rescale = True
    ctx.auto_mod_switch = True
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    print(f"auto_rescale={ctx.auto_rescale}  auto_mod_switch={ctx.auto_mod_switch}")

    base = np.array([1.0 + 0.001 * i for i in range(args.size)], dtype=float)
    ref = base.copy()
    ct = ts.ckks_vector(ctx, base.tolist())
    for _ in range(args.depth):
        ct = ct * ct
        ref = ref * ref

    dec = np.asarray(ct.decrypt(), dtype=float)
    abs_err = float(np.max(np.abs(dec - ref)))
    rel_err = float(np.max(np.abs(dec - ref) / np.abs(ref)))
    threshold = 3e-3  # ~2^-8, the fixed-point precision f=8
    print(f"depth={args.depth}  size={args.size}  "
          f"max_abs_err={abs_err:.3e}  max_rel_err={rel_err:.3e}")
    print(f"threshold={threshold:.1e}  " + ("PASS" if rel_err < threshold else "FAIL"))
    return 0 if rel_err < threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
