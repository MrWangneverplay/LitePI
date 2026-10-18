"""Client-side HE benchmark for the Raspberry Pi (security profile B).

Measures the operations the *client* pays for in LitePI:

  * CKKS context/key generation (one-time setup)
  * encrypt one input vector
  * decrypt one vector
  * one hybrid-layer round (decrypt + re-encrypt)
  * serialized public/full context size and peak RSS

Run it on the Pi and on a desktop with the same parameters; the ratio is the
number the paper reports for the B profile.

Usage:
    python analysis/bench_pi.py --vector-size 1328 --repeat 5
    python analysis/bench_pi.py --vector-size 784 --repeat 5 --out results/pi_mnist.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import resource
except ImportError:  # Windows
    resource = None

try:
    import tenseal as ts
except Exception:
    ts = None

from params_b import (
    COEFF_MOD_BIT_SIZES,
    F,
    GLOBAL_SCALE,
    K_COMPARE,
    K_RING,
    POLY_MODULUS_DEGREE,
    SIGMA,
)


def peak_rss_mb():
    if resource is None:
        return None
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024.0


def summarise(times):
    return {
        "mean_ms": round(statistics.mean(times) * 1e3, 2),
        "std_ms": round(statistics.pstdev(times) * 1e3, 2) if len(times) > 1 else 0.0,
        "min_ms": round(min(times) * 1e3, 2),
        "max_ms": round(max(times) * 1e3, 2),
    }


def timed(fn, repeat):
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI client HE benchmark (profile B)")
    ap.add_argument("--vector-size", type=int, default=1328,
                    help="plaintext vector length (default: CIFAR hybrid layer)")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", default=None, help="optional JSON output path")
    args = ap.parse_args()

    if ts is None:
        print("TenSEAL is not installed in this interpreter.")
        print("Run this script with the project venv on the target device:")
        print("  .venv/bin/python analysis/bench_pi.py ...   # Raspberry Pi (Linux)")
        return 1

    data = [0.1 * (i % 7) for i in range(args.vector_size)]

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

    ct = ts.ckks_vector(ctx, data)
    enc_times = timed(lambda: ts.ckks_vector(ctx, data), args.repeat)
    dec_times = timed(lambda: ct.decrypt(), args.repeat)
    hybrid_times = timed(lambda: ts.ckks_vector(ctx, ct.decrypt()), args.repeat)

    public_mb = len(ctx.serialize(save_secret_key=False)) / 1e6
    full_mb = len(ctx.serialize(save_secret_key=True)) / 1e6
    rss_mb = peak_rss_mb()

    result = {
        "profile": "B",
        "sigma": SIGMA,
        "k_ring": K_RING,
        "k_compare": K_COMPARE,
        "f": F,
        "poly_modulus_degree": POLY_MODULUS_DEGREE,
        "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
        "global_scale_bits": GLOBAL_SCALE.bit_length() - 1,
        "vector_size": args.vector_size,
        "repeat": args.repeat,
        "keygen_s": round(keygen_s, 3),
        "encrypt": summarise(enc_times),
        "decrypt": summarise(dec_times),
        "hybrid_layer_decrypt_encrypt": summarise(hybrid_times),
        "public_context_mb": round(public_mb, 2),
        "full_context_mb": round(full_mb, 2),
        "peak_rss_mb": round(rss_mb, 1) if rss_mb is not None else None,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "tenseal": getattr(ts, "__version__", "unknown"),
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print(f"profile B: sigma={SIGMA}, N={POLY_MODULUS_DEGREE}, "
          f"scale=2^{result['global_scale_bits']}, vector={args.vector_size}")
    print(f"keygen          : {result['keygen_s']:.3f} s")
    print(f"encrypt         : {result['encrypt']['mean_ms']:.2f} ms "
          f"(+/- {result['encrypt']['std_ms']:.2f})")
    print(f"decrypt         : {result['decrypt']['mean_ms']:.2f} ms "
          f"(+/- {result['decrypt']['std_ms']:.2f})")
    print(f"hybrid 1 layer  : {result['hybrid_layer_decrypt_encrypt']['mean_ms']:.2f} ms")
    print(f"public context  : {result['public_context_mb']:.2f} MB")
    print(f"full context    : {result['full_context_mb']:.2f} MB")
    print(f"peak RSS        : {result['peak_rss_mb']} MB")

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"JSON -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
