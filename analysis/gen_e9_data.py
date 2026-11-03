"""E9: robustness of LitePI under packet loss and jitter, from netem runs only.

Experiment plan E9: loss {0,1,5}% x jitter ±30 ms, expected result "Hybrid-all
suffers most, Adaptive is the most robust".

There is no analytic model here any more.  The previous one took each flight as
one packet that costs one RTT to resend (E[T] = C + R*RTT/(1-p)), but one
ciphertext is about 630 TCP segments, so at 1% loss nearly every flight loses a
segment; a lost segment costs a fast retransmit or an RTO rather than one RTT,
and TCP throughput itself falls with loss.  Its conclusions (a 0.25 s penalty at
5% loss, a schedule that does not react to loss) came from that assumption.

This script reads the measured grid written by

    sudo -E python analysis/netem_e2e.py --host <server> --iface <nic> \\
        --dataset mnist --schedule 11 --mode e9

one run per (dataset, schedule), and reports per point

    amplification = latency(loss, jitter) / latency(0, 0)   mean and p95

Whether "fewer hybrid layers are more robust" holds is then read from the
measured growth of each schedule, and the winner is the schedule with the lowest
measured p95 among those above the accuracy floor.  With no runs it stops and
prints what to run.

    python analysis/gen_e9_data.py --device pi5
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import planner as P
from data_manifest import git_sha
from gen_e5_data import DEFAULT_ACC_MIN

ROOT = Path(__file__).resolve().parent.parent


def load_points(root: Path, device: str) -> list[dict]:
    pattern = str(root / "results" / "raw_results" / "E9" / device / "netem_*_s*" / "run.json")
    rows = []
    for path in sorted(glob.glob(pattern)):
        dataset, schedule = re.search(r"netem_(.+)_s(\d+)", path).groups()
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        for p in run["points"]:
            # the run stores the schedule as a bit list; keep the directory's
            # string label as the hashable key
            rows.append({**p, "dataset": dataset, "schedule": schedule,
                         "delay_mult": run.get("delay_mult", 1.0)})
    return rows


def amplification(rows: list[dict]) -> list[dict]:
    clean = {(r["dataset"], r["schedule"], r["rtt_ms"]): r for r in rows
             if r["loss_pct"] == 0 and r["jitter_ms"] == 0}
    out = []
    for r in rows:
        base = clean.get((r["dataset"], r["schedule"], r["rtt_ms"]))
        if base is None:
            continue
        out.append({
            "dataset": r["dataset"], "schedule": r["schedule"],
            "K_h": r["schedule"].count("1"), "rtt_ms": r["rtt_ms"],
            "effective_rtt_ms": r["rtt_ms"] * r["delay_mult"],
            "bandwidth_mbps": r["bandwidth_mbps"],
            "loss_pct": r["loss_pct"], "jitter_ms": r["jitter_ms"],
            "wall_mean_ms": r["wall_mean_ms"], "wall_p95_ms": r["wall_p95_ms"],
            "added_mean_ms": r["wall_mean_ms"] - base["wall_mean_ms"],
            "added_p95_ms": r["wall_p95_ms"] - base["wall_p95_ms"],
            "mean_x": r["wall_mean_ms"] / base["wall_mean_ms"],
            "p95_x": r["wall_p95_ms"] / base["wall_p95_ms"],
        })
    return out


def best_schedules(points: list[dict], acc_floor: float | None) -> list[dict]:
    """Lowest measured p95 per link condition, among schedules above the floor."""
    out = []
    keys = sorted({(p["dataset"], p["rtt_ms"], p["loss_pct"], p["jitter_ms"]) for p in points})
    for dataset, rtt, loss, jitter in keys:
        arch = "conv3" if dataset == "conv3" else dataset
        accs = P.arch_spec(arch)[1]
        floor = DEFAULT_ACC_MIN[arch] if acc_floor is None else acc_floor
        ok = [p for p in points
              if (p["dataset"], p["rtt_ms"], p["loss_pct"], p["jitter_ms"])
              == (dataset, rtt, loss, jitter)
              and accs.get(p["schedule"], -1.0) >= floor]
        if ok:
            best = min(ok, key=lambda p: p["wall_p95_ms"])
            out.append({"dataset": dataset, "rtt_ms": rtt, "loss_pct": loss,
                        "jitter_ms": jitter, "schedule": best["schedule"],
                        "K_h": best["K_h"], "wall_p95_ms": best["wall_p95_ms"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--device", default="pi5")
    ap.add_argument("--acc-floor", type=float, default=None)
    args = ap.parse_args()

    rows = load_points(args.root, args.device)
    if not rows:
        print("no netem E9 runs under results/raw_results/E9/"
              f"{args.device}/; run netem_e2e.py --mode e9 per (dataset, schedule)")
        return 1
    points = amplification(rows)
    best = best_schedules(points, args.acc_floor)
    out = args.root / "results" / "E9"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "e9_measured.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    (out / "e9_measured.json").write_text(json.dumps({
        "exp_id": "E9", "git_sha": git_sha(args.root), "device": args.device, "measured": True,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "points": points, "best_schedule_per_condition": best,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"{'dataset':<9}{'s':<6}{'RTT':>5}{'loss':>6}{'jit':>5}"
          f"{'mean ms':>11}{'p95 ms':>11}{'mean x':>8}{'p95 x':>8}")
    for p in points:
        print(f"{p['dataset']:<9}{p['schedule']:<6}{p['rtt_ms']:>5.0f}{p['loss_pct']:>5.0f}%"
              f"{p['jitter_ms']:>5.0f}{p['wall_mean_ms']:>11.0f}{p['wall_p95_ms']:>11.0f}"
              f"{p['mean_x']:>8.2f}{p['p95_x']:>8.2f}")
    print("best schedule per condition:")
    for b in best:
        print(f"  {b['dataset']:<9} RTT {b['rtt_ms']:.0f} loss {b['loss_pct']:.0f}% "
              f"jitter {b['jitter_ms']:.0f}: s{b['schedule']} (K_h={b['K_h']}, "
              f"p95 {b['wall_p95_ms']:.0f} ms)")
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
