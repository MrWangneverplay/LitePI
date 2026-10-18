"""Sweep the CKKS precision/mask frontier.

For each level-prime bit size s, build N=16384 with the chain
[60, s*7, 60], set scale=2^s, and run `depth` ciphertext squarings.

The plaintext bound is q0/(2*scale) = 2^(59-s), so the mask domain
8*2^sigma = 2^(sigma+3) fits iff sigma <= 56 - s.  The precision constraint
is rel_err < 3e-3 (the f=8 fixed-point target).

The frontier is the smallest s that passes the depth check; it gives the
largest feasible sigma = 56 - s.

Run:
    python analysis/sweep_ckks_frontier.py
    python analysis/sweep_ckks_frontier.py --depth 6 --sizes 23,26,30,35,40
"""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import tenseal as ts
except Exception:
    ts = None

from params_b import POLY_MODULUS_DEGREE


def run_one(level_bits: int, depth: int, size: int = 16) -> float:
    """Return the max relative error after `depth` ciphertext squarings."""
    chain = [60] + [level_bits] * 7 + [60]
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=chain,
    )
    ctx.global_scale = 1 << level_bits
    ctx.auto_rescale = True
    ctx.auto_mod_switch = True
    ctx.generate_relin_keys()

    base = np.array([1.0 + 0.001 * i for i in range(size)], dtype=float)
    ref = base.copy()
    ct = ts.ckks_vector(ctx, base.tolist())
    for _ in range(depth):
        ct = ct * ct
        ref = ref * ref

    dec = np.asarray(ct.decrypt(), dtype=float)
    return float(np.max(np.abs(dec - ref) / np.abs(ref)))


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep the CKKS precision/mask frontier")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--sizes", default="23,24,25,26,28,30,32,35,40",
                    help="comma list of level-prime bit sizes")
    ap.add_argument("--threshold", type=float, default=3e-3,
                    help="relative-error threshold (default: 2^-8)")
    args = ap.parse_args()

    if ts is None:
        print("TenSEAL is not installed in this interpreter.")
        return 1

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    print(f"depth={args.depth}  threshold={args.threshold:.1e}  "
          f"N={POLY_MODULUS_DEGREE}  chain=[60, s*7, 60]")
    print(f"{'s':>3}  {'sigma_max':>9}  {'depth_err':>11}  status")
    first_pass = None
    for s in sizes:
        t0 = time.perf_counter()
        try:
            rel = run_one(s, args.depth)
            sigma_max = 56 - s
            ok = rel < args.threshold
            status = "PASS" if ok else "FAIL"
            if ok and first_pass is None:
                first_pass = s
            print(f"{s:>3}  {sigma_max:>9}  {rel:>11.3e}  {status} "
                  f"({time.perf_counter() - t0:.1f}s)")
        except Exception as exc:
            print(f"{s:>3}  {'-':>9}  {type(exc).__name__}: {exc}")

    if first_pass is not None:
        print(f"\nSmallest passing s = {first_pass}; "
              f"max feasible sigma = {56 - first_pass}")
    else:
        print(f"\nNo level size passed at depth {args.depth}; "
              f"CKKS cannot meet the f=8 target with this chain shape.")

    # Standard high-precision control: 40-bit level primes, scale 2^40.
    try:
        t0 = time.perf_counter()
        rel = run_one(40, args.depth)
        print(f"\ncontrol s=40: depth_err={rel:.3e} "
              f"({'PASS' if rel < args.threshold else 'FAIL'}, "
              f"{time.perf_counter() - t0:.1f}s)")
    except Exception as exc:
        print(f"\ncontrol s=40 failed: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
