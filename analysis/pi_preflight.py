"""One-shot environment check for the Raspberry Pi.

Run this BEFORE copying anything to the Pi:

    python pi_preflight.py
    python pi_preflight.py --skip-mpc     # HE only

It checks:
  1. Python / platform / memory,
  2. TenSEAL import and the Pi demo profile keygen + encrypt/decrypt round trip,
  3. the paper-profile MPC ReLU over localhost,
  4. a complete real-CKKS hybrid inference over loopback.

Exit codes: 0 = ready, 2 = TenSEAL missing, 3 = HE round trip failed,
4 = MPC self-test failed, 5 = batch MPC failed, 6 = hybrid loopback failed.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np

from params_pi import (
    COEFF_MOD_BIT_SIZES,
    GLOBAL_SCALE,
    K_COMPARE,
    K_RING,
    POLY_MODULUS_DEGREE,
    PROFILE_NAME,
    SIGMA,
)


def _rss_mb() -> float:
    try:
        import resource
    except ImportError:
        return -1.0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI Raspberry-Pi preflight")
    ap.add_argument("--skip-mpc", action="store_true")
    ap.add_argument("--vector-size", type=int, default=64)
    args = ap.parse_args()

    print("== platform ==")
    print(f"python     : {sys.version.split()[0]}")
    print(f"system     : {platform.system()} {platform.release()} {platform.machine()}")
    print(f"processor  : {platform.processor() or 'unknown'}")
    print(f"numpy      : {np.__version__}")
    try:
        import cryptography
        print(f"cryptography: {cryptography.__version__}  "
              "(OpenSSL P-256 acceleration)")
    except Exception:
        print("cryptography: MISSING -- the OT stack will use a much slower "
              "pure-Python curve")

    try:
        import tenseal as ts
    except Exception as exc:
        print(f"\nTenSEAL    : MISSING ({type(exc).__name__}: {exc})")
        print("See PI_RUNBOOK.md, section 'Installing TenSEAL on ARM'.")
        return 2
    print(f"tenseal    : {getattr(ts, '__version__', 'unknown')}")

    print(f"\n== {PROFILE_NAME} CKKS round trip ==")
    print(f"sigma={SIGMA}  k_ring={K_RING}  k_compare={K_COMPARE}")
    print(f"N={POLY_MODULUS_DEGREE}  coeff={COEFF_MOD_BIT_SIZES}  "
          f"scale=2^{GLOBAL_SCALE.bit_length() - 1}")
    try:
        t0 = time.perf_counter()
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
        keygen_s = time.perf_counter() - t0

        data = np.linspace(-1.0, 1.0, args.vector_size).tolist()
        t0 = time.perf_counter()
        ct = ts.ckks_vector(ctx, data)
        enc_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        out = np.asarray(ct.decrypt(), dtype=np.float64)
        dec_s = time.perf_counter() - t0
        err = float(np.max(np.abs(out - np.asarray(data))))
        public_blob = ctx.serialize(save_secret_key=False)
        public_ctx = ts.context_from(public_blob)
        if public_ctx.has_secret_key():
            raise RuntimeError("serialized public context unexpectedly contains the secret key")
        if not public_ctx.has_galois_keys() or not public_ctx.has_relin_keys():
            raise RuntimeError("serialized public context is missing evaluation keys")
        public_mb = len(public_blob) / 1e6
        full_mb = len(ctx.serialize(save_secret_key=True)) / 1e6
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 3

    print(f"keygen     : {keygen_s:.2f} s")
    print(f"encrypt    : {enc_s * 1e3:.2f} ms")
    print(f"decrypt    : {dec_s * 1e3:.2f} ms")
    print(f"roundtrip  : max abs err {err:.3e}")
    print(f"context    : public {public_mb:.1f} MB / full {full_mb:.1f} MB")
    print(f"peak RSS   : {_rss_mb():.1f} MB")

    if not args.skip_mpc:
        print("\n== networked MPC ReLU ==")
        from secure_relu_net import batch_relu_self_test, net_self_test

        code = net_self_test()
        if code != 0:
            return 4
        print("\n== batched MPC ReLU (n=32) ==")
        code = batch_relu_self_test(32)
        if code != 0:
            return 5

        print("\n== real-CKKS hybrid loopback ==")
        from pi_client import loopback_once

        result = loopback_once("tenseal", [8, 5, 3])
        lb = result["loopback"]
        print(
            f"logits err {lb['max_abs_err']:.3e} "
            f"(tol {lb['tolerance']:.0e})  "
            f"{'PASS' if lb['passed'] else 'FAIL'}"
        )
        if not lb["passed"]:
            return 6

    print("\nPREFLIGHT OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

