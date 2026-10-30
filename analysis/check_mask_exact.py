"""M2 v3: wide masks through the double-double CKKS codec.

Part 1 compares the new codec with SEAL's on ordinary values, so a slot-order
or scale bug cannot hide behind the large masks of part 2.
Part 2 runs protocol steps 1-2 for mask half-widths A = 2^a up to the point
where z + r no longer fits the ciphertext modulus:

    step 1  ct_{z+r} = ct_z + Enc_pk(r)    r encoded by encrypt_like_dd
    step 2  s_c = decrypt_ring(ct_{z+r})   integer mod 2^K_RING
            s_s = -r mod 2^K_RING          exact integer arithmetic

and reports |ring(s_c + s_s) - round(z * 2^F)| in LSB.  PASS: <= 4 LSB (the
CKKS noise floor, same criterion as check_mask_domain.py).

    python analysis/check_mask_exact.py
    python analysis/check_mask_exact.py --drops 3 --bits 42,60,70,74,78
"""

from __future__ import annotations

import argparse
import secrets

import numpy as np

from params_pi import F, K_RING
from pi_proto import make_backend


def split_dd(ints: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-point integers -> (hi, lo) doubles with hi + lo = k / 2^F exactly."""
    scale = float(1 << F)
    hi = [float(k) for k in ints]
    lo = [float(k - int(h)) for k, h in zip(ints, hi)]
    return np.array(hi) / scale, np.array(lo) / scale


def check_codec(be, ctx, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-50, 50, n)
    ct = be.encrypt(ctx, x)
    # decode: same ciphertext through SEAL's decoder and through ours.
    a = be.decrypt(ct)
    hi, lo = ctx.decrypt_dd(ct)
    err = float(np.max(np.abs((hi + lo) - a)))
    assert err < 1e-6, f"decoder disagrees with SEAL: {err}"
    # encode: encrypt_like_dd must decrypt (via SEAL) to the same values.
    ct2 = ctx.encrypt_like_dd(x, np.zeros(n), ct)
    err = float(np.max(np.abs(be.decrypt(ct2) - x)))
    assert err < 1e-2, f"encoder disagrees with SEAL: {err}"
    # slots beyond the logical length stay zero.
    y = rng.uniform(-1, 1, 16)
    ct3 = ctx.encrypt_like_dd(y, np.zeros(16), ct)
    err = float(np.max(np.abs(be.decrypt(ct3) - y)))
    assert err < 1e-2, f"short vector: {err}"
    print(f"PASS  codec matches SEAL on {n} slots")


def sweep(be, ctx, bits_list, n: int, z_bound: float, drops: int, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    z = rng.uniform(-z_bound, z_bound, n)
    want = [int(round(v * (1 << F))) for v in z]
    mod, half = 1 << K_RING, 1 << (K_RING - 1)
    ok = True
    for a in bits_list:
        span = 1 << (a + 1 + F)
        k = [secrets.randbelow(span) - (span >> 1) for _ in range(n)]
        ct = be.encrypt(ctx, z)
        for _ in range(drops):
            ct = be.mul_plain(ct, np.ones(n))
        have = ctx.modulus_bits(ct)
        ct_zr = be.add_cipher(ct, ctx.encrypt_like_dd(*split_dd(k), ct))
        ring = ctx.decrypt_ring(ct_zr, F, K_RING)
        err = 0
        for s_c, r, w in zip(ring, k, want):
            v = (int(s_c) - r) % mod
            v = v - mod if v >= half else v
            err = max(err, abs(v - w))
        fits = a + np.log2(ctx.scale(ct)) + 2 < have
        passed = err <= 4
        ok &= passed or not fits
        print(f"a={a:3d}  modulus {have:3d} bit, scale 2^{np.log2(ctx.scale(ct)):.1f}  "
              f"share_err={err:>8d} LSB  "
              f"{'PASS' if passed else 'FAIL' if fits else 'out of modulus (expected)'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bits", default="30,42,50,60,70,74,78,80")
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--z-bound", type=float, default=128.0)
    ap.add_argument("--drops", type=int, default=3,
                    help="rescales before the hybrid layer (L7 -> L4 = 3)")
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()
    be = make_backend("seal")
    ctx = be.new_context()
    check_codec(be, ctx, args.n, args.seed)
    bits = [int(b) for b in args.bits.split(",")]
    return 0 if sweep(be, ctx, bits, args.n, args.z_bound, args.drops, args.seed) else 1


if __name__ == "__main__":
    raise SystemExit(main())
