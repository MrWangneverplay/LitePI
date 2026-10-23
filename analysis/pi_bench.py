"""Repeatable benchmark for the LitePI Pi <-> desktop hybrid inference loop.

Each repetition is a fresh client key setup and a fresh connection, so the
reported number is a cold-start end-to-end query.  Use ``--repeat 1`` on a
Raspberry Pi if key generation dominates; report keygen separately either way.

Examples:
    python pi_bench.py --backend stub --dims 64,32,10 --repeat 3
    python pi_bench.py --backend tenseal --dims 64,32,10 --repeat 3 \
        --out results/pi_hybrid_B.json
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

from pi_client import loopback_once
from pi_proto import profile_info


def _as_dims(text: str) -> list[int]:
    dims = [int(x) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(dims) < 2 or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("dims must be at least two positive ints")
    return dims


def _summary(values: list[float]) -> dict:
    return {
        "mean": round(statistics.fmean(values), 6),
        "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
        "min": round(min(values), 6),
        "max": round(max(values), 6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI Pi hybrid-inference benchmark")
    ap.add_argument("--backend", choices=["tenseal", "stub"], default="tenseal")
    ap.add_argument("--dims", type=_as_dims, default=[64, 32, 10])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--input-seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    runs: list[dict] = []
    started = time.perf_counter()
    for i in range(args.repeat):
        print(f"[bench] run {i + 1}/{args.repeat} ...", flush=True)
        result = loopback_once(args.backend, args.dims, args.seed, args.input_seed)
        lb = result["loopback"]
        if not lb["passed"]:
            print(json.dumps(result, indent=2, ensure_ascii=False))
            print("[bench] loopback FAILED -- refusing to report numbers")
            return 1
        runs.append(result)

    keys = ["keygen_s", "encrypt_input_s", "decrypt_s", "mpc_s",
            "reencrypt_s", "wall_s"]
    aggregate = {
        key: _summary([float(r["timings"].get(key, r.get(key, 0.0))) for r in runs])
        for key in keys
    }
    aggregate["bytes_sent"] = _summary([float(r["bytes_sent"]) for r in runs])
    aggregate["bytes_received"] = _summary([float(r["bytes_received"]) for r in runs])
    aggregate["max_logits_err"] = _summary(
        [float(r["loopback"]["max_abs_err"]) for r in runs]
    )

    payload = {
        "experiment": "LitePI Pi hybrid inference (loopback, two real parties)",
        **profile_info(args.backend),
        "dims": args.dims,
        "hybrid_layers": len(args.dims) - 2,
        "repeat": args.repeat,
        "total_wall_s": round(time.perf_counter() - started, 3),
        "aggregate": aggregate,
        "runs": runs,
        "environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
        },
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    print()
    print(f"backend={args.backend}  dims={args.dims}  repeat={args.repeat}")
    for key in keys:
        s = aggregate[key]
        print(f"  {key:18s} mean {s['mean'] * 1000:8.2f} ms   "
              f"std {s['std'] * 1000:7.2f}   max {s['max'] * 1000:8.2f}")
    print(f"  {'max logits err':18s} {aggregate['max_logits_err']['max']:.3e}")
    if args.backend == "stub":
        print("\nNOTE: backend=stub is a wiring simulation, not a privacy result.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

