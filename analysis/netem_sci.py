"""SCI-only netem sweep: how many round trips one secure ReLU costs.

The decomposed r_h in results/rounds/r_h_netem_<ds>.json comes from the
client's `mpc_s` rising with RTT inside a full query.  A reviewer can object
that this is a single method with no independent check.  This driver supplies
one: it runs *only* the SCI ReLU (no HE linear layer, no CKKS, no server
compute), so a query costs well under a second and the wall clock is dominated
by the network instead of a 20-35 s forward pass.

Single connection, one netem sweep, so the handshake is paid once:

    server:  python analysis/netem_sci.py --serve --port 9200 --dim 128 --total 25
    client:  python analysis/netem_sci.py --port 9200 --dim 128 \
                 --rtts 1,5,20,50,100 --warmup 1 --repeat 4

The client shapes its own egress with `tc qdisc netem` before each RTT point
and times every ReLU.  A least-squares fit of per-call wall time against RTT
gives rounds = slope / delay_mult, where delay_mult is 2 on `lo` (netem delays
request and response on the same device) and 1 on a real NIC.

Output: results/rounds/r_h_sci_only.json
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def tc(iface: str, rtt_ms: float, dry: bool) -> None:
    cmd = ["tc", "qdisc", "replace", "dev", iface, "root", "netem",
           "delay", f"{rtt_ms}ms"]
    print("  $", shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def tc_clear(iface: str, dry: bool) -> None:
    cmd = ["tc", "qdisc", "del", "dev", iface, "root"]
    print("  $", shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=False, stderr=subprocess.DEVNULL)


def delay_mult(iface: str) -> float:
    return 2.0 if iface in ("lo", "loopback") else 1.0


def serve(args) -> int:
    from params_pi import F, K_RING
    from sci_relu import SciReLUSession

    shares = [0.25] * args.dim
    print(f"serving {args.total} relus, dim={args.dim}, port={args.port}",
          flush=True)
    session = SciReLUSession(1, args.port)
    for i in range(args.total):
        t0 = time.perf_counter()
        session.relu(shares, F, K_RING)
        print(f"  call {i + 1}/{args.total} {time.perf_counter() - t0:.3f}s",
              flush=True)
    return 0


def client(args) -> int:
    from params_pi import F, K_RING
    from sci_relu import SciReLUSession

    rtts = [float(v) for v in args.rtts.split(",")]
    shares = [0.25] * args.dim
    per_point = args.warmup + args.repeat
    session = SciReLUSession(2, args.port, args.host)
    points = []
    try:
        for rtt in rtts:
            print(f"[point] rtt={rtt} ms", flush=True)
            tc(args.iface, rtt, args.dry_run)
            time.sleep(0.5)
            walls = []
            for _ in range(per_point):
                t0 = time.perf_counter()
                session.relu(shares, F, K_RING)
                walls.append((time.perf_counter() - t0) * 1e3)
            kept = walls[args.warmup:]
            points.append({"rtt_ms": rtt, "wall_ms": kept,
                           "wall_mean_ms": float(np.mean(kept)),
                           "wall_p95_ms": float(np.percentile(kept, 95))})
            print(f"   wall mean {np.mean(kept):.1f} ms  p95 "
                  f"{np.percentile(kept, 95):.1f} ms", flush=True)
    finally:
        tc_clear(args.iface, args.dry_run)

    if args.dry_run or len(points) < 3:
        return 0
    xs = np.array([p["rtt_ms"] for p in points for _ in p["wall_ms"]], float)
    ys = np.array([w for p in points for w in p["wall_ms"]], float)
    slope, icpt = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + icpt)
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(xs) - 2, 1)
                       / np.sum((xs - xs.mean()) ** 2)))
    rounds = float(slope) / delay_mult(args.iface)
    payload = {
        "experiment": "r_h_sci_only",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "method": "one logical SCI ReLU per call, no HE layer; netem RTT "
                  "sweep on the client egress; rounds = d(wall)/d(RTT) / "
                  "delay_mult",
        "scope": "end-to-end",
        "dim": args.dim,
        "delay_mult": delay_mult(args.iface),
        "rounds_per_relu": round(rounds, 3),
        "rounds_per_relu_se": round(float(se) / delay_mult(args.iface), 3),
        "intercept_ms": round(float(icpt), 1),
        "points": points,
    }
    out = ROOT / "results" / "rounds" / "r_h_sci_only.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"rounds per SCI ReLU = {rounds:.2f} +/- "
          f"{se / delay_mult(args.iface):.2f}  -> {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--iface", default="lo")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--serve", action="store_true", help="run as SCI party 1")
    ap.add_argument("--total", type=int, default=25,
                    help="server: number of relus the client will request")
    ap.add_argument("--rtts", default="1,5,20,50,100")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    return serve(args) if args.serve else client(args)


if __name__ == "__main__":
    raise SystemExit(main())
