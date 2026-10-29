"""E3 / E9 end-to-end under tc/netem, and the end-to-end r_h it implies.

E3 and E9 were analytic models (T = C + R*RTT ...).  This driver measures the
same quantities on the wire: a LitePI server runs on the desktop
(`pi_server.py --serve`), the client (Pi or second host) runs this script,
which shapes its own egress with `tc qdisc netem` and times real queries.

Because netem delays every packet on the interface, it also delays SCI's own
socket - the part the message-counting r_h (measure_rounds.py) cannot see.
The wall time therefore obeys

    T(s, RTT) = C(s) + R_eff(s) * RTT
    r_h_eff   = (R_eff(s) - R_eff(00)) / K_h(s)

and a least-squares fit of T over the RTT grid recovers the end-to-end round
count, SCI-internal rounds included.

Caveat: the wall-time slope also counts TCP rounds that are not protocol
rounds.  A ~0.9 MB ciphertext needs several RTTs of slow start whenever the
congestion window has collapsed (Linux resets it after an idle period unless
net.ipv4.tcp_slow_start_after_idle=0), so the s=11 - s=00 slope would charge
those RTTs to the hybrid layers.  fit_r_h therefore also fits the client's
SCI time (timings["mpc_s"]) against RTT: that slope / K_h is the number of
SCI round trips per layer, and r_h = 1 (the ①/② exchange) + that number.
This decomposed value is what r_h_measured reports; the wall-time slope is
kept as a cross-check.  Set tcp_slow_start_after_idle=0 on both hosts.  The fit is written to
results/rounds/r_h_netem_<dataset>.json with scope "end-to-end", which
cost_model.load_r_h prefers over the control-plane count.

Topology (实验方案 §2.1: 有线直连 + tc/netem): delay/jitter/loss/rate are
applied on the client's egress only, so the configured delay is the full RTT.
Needs root for tc (or CAP_NET_ADMIN).  --dry-run prints the tc commands.

Server side, once per schedule (one process per schedule):
    python analysis/pi_server.py --dataset mnist --schedule 1,1 --serve --port 9101
Client side:
    sudo -E python analysis/netem_e2e.py --host 192.168.1.20 --port 9101 \\
        --iface eth0 --dataset mnist --schedule 11 --mode e3
    sudo -E python analysis/netem_e2e.py ... --mode e9      # loss x jitter grid
    python analysis/netem_e2e.py --fit-only --dataset mnist  # r_h from saved runs
"""

from __future__ import annotations

import argparse
import itertools
import json
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
E3_RTT = (1, 5, 20, 50, 100, 200)
E3_BW = (10, 50, 100)
E9_LOSS = (0.0, 1.0, 5.0)
E9_JITTER = (0.0, 30.0)
E9_RTT = (20, 100)


def tc(iface: str, rtt_ms: float, bw_mbps: float | None, loss_pct: float,
       jitter_ms: float, dry: bool) -> None:
    cmd = ["tc", "qdisc", "replace", "dev", iface, "root", "netem",
           "delay", f"{rtt_ms}ms"]
    if jitter_ms:
        # netem's default jitter is uniform in [-J, +J]: the same law as the
        # E9 model's (2J)^2/12 variance, so measured and modelled p95 compare.
        cmd += [f"{jitter_ms}ms"]
    if loss_pct:
        cmd += ["loss", f"{loss_pct}%"]
    if bw_mbps:
        cmd += ["rate", f"{bw_mbps}mbit"]
    print("  $", shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=True)


def tc_clear(iface: str, dry: bool) -> None:
    cmd = ["tc", "qdisc", "del", "dev", iface, "root"]
    print("  $", shlex.join(cmd))
    if not dry:
        subprocess.run(cmd, check=False, stderr=subprocess.DEVNULL)


def run_point(args, x, rtt, bw, loss, jitter) -> dict:
    from pi_client import run_reuse_session
    tc(args.iface, rtt, bw, loss, jitter, args.dry_run)
    if args.dry_run:
        return {}
    time.sleep(0.5)
    res = run_reuse_session(args.host, args.port, x, args.backend,
                            repeats=args.warmup + args.repeat,
                            timeout=args.timeout, output_mode=args.output_mode)
    qs = res["queries"][args.warmup:]
    walls = [q["wall_s"] * 1e3 for q in qs]
    return {
        "rtt_ms": rtt, "bandwidth_mbps": bw, "loss_pct": loss, "jitter_ms": jitter,
        "wall_ms": walls,
        "wall_mean_ms": float(np.mean(walls)),
        "wall_p95_ms": float(np.percentile(walls, 95)),
        "client_timings": [q["timings"] for q in qs],
        "bytes_sent": [q["bytes_sent"] for q in qs],
        "bytes_received": [q["bytes_received"] for q in qs],
        "server_wall_s": [q["server"].get("wall_s") for q in qs],
        "hybrid_roundtrips": qs[-1]["server"].get("hybrid_roundtrips"),
        "schedule": res.get("schedule"),
        "wire": res.get("wire"),
        "mpc_runtime": res.get("mpc_runtime"),
        "keygen_s": res["timings"].get("keygen_s"),
        "context_upload_s": res["timings"].get("context_upload_s"),
    }


def out_dir(root: Path, device: str, mode: str, dataset: str, schedule: str) -> Path:
    return (root / "results" / "raw_results" / ("E3" if mode == "e3" else "E9")
            / device / f"netem_{dataset}_s{schedule}")


def delay_mult(iface: str) -> float:
    """ms of RTT added per ms of configured netem delay.

    On a real NIC netem shapes the client's egress only, so the configured
    delay is the whole RTT (factor 1).  On `lo` both directions are egress on
    the same device, so a configured D delays request and response and the RTT
    grows by 2D.  The fit must divide that factor out or r_h doubles.
    """
    return 2.0 if iface in ("lo", "loopback") else 1.0


def _slope(xs, ys) -> tuple[float, float, float]:
    """(slope, intercept, standard error of the slope)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    slope, icpt = np.polyfit(xs, ys, 1)
    resid = ys - (slope * xs + icpt)
    se = float(np.sqrt(np.sum(resid ** 2) / max(len(xs) - 2, 1)
                       / np.sum((xs - xs.mean()) ** 2)))
    return float(slope), float(icpt), se


def slow_start_after_idle() -> int | None:
    try:
        return int(Path("/proc/sys/net/ipv4/tcp_slow_start_after_idle")
                   .read_text().strip())
    except (OSError, ValueError):
        return None


def fit_r_h(root: Path, device: str, dataset: str) -> dict | None:
    """R_eff(s) by least squares of wall vs RTT (zero loss/jitter, largest BW),
    and SCI rounds per layer from the client's mpc_s vs RTT."""
    slopes = {}
    sci_rounds = {}
    tcp_idle = set()
    for f in sorted((root / "results" / "raw_results" / "E3" / device)
                    .glob(f"netem_{dataset}_s*/run.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        mult = float(d.get("delay_mult", 1.0))
        pts = [p for p in d["points"]
               if p.get("loss_pct", 0) == 0 and p.get("jitter_ms", 0) == 0]
        if not pts:
            continue
        bw = max(p["bandwidth_mbps"] or 0 for p in pts)
        pts = [p for p in pts if (p["bandwidth_mbps"] or 0) == bw]
        if len({p["rtt_ms"] for p in pts}) < 3:
            continue
        tcp_idle.add(d.get("tcp_slow_start_after_idle"))
        xs = np.array([p["rtt_ms"] for p in pts for _ in p["wall_ms"]], float)
        ys = np.array([w for p in pts for w in p["wall_ms"]], float)
        slope, icpt, se = _slope(xs, ys)
        k_h = str(d["schedule"]).count("1")
        if k_h:
            mx = [p["rtt_ms"] for p in pts for _ in p["client_timings"]]
            my = [t.get("mpc_s", 0.0) * 1e3 for p in pts for t in p["client_timings"]]
            m_slope, _, m_se = _slope(mx, my)
            sci_rounds[d["schedule"]] = {
                "sci_rounds_per_layer": m_slope / mult / k_h,
                "se": m_se / mult / k_h}
        slopes[d["schedule"]] = {"R_eff": float(slope) / mult, "R_eff_se": se / mult,
                                 "intercept_ms": float(icpt),
                                 "bandwidth_mbps": bw, "n": int(len(xs)),
                                 "delay_mult": mult}
    if "00" not in slopes or len(slopes) < 2:
        return None
    per = {}
    for s, v in slopes.items():
        k_h = s.count("1")
        if k_h:
            per[s] = (v["R_eff"] - slopes["00"]["R_eff"]) / k_h
    r_h_wall = float(np.mean(list(per.values())))
    r_sci = (float(np.mean([v["sci_rounds_per_layer"] for v in sci_rounds.values()]))
             if sci_rounds else None)
    r_h = 1.0 + r_sci if r_sci is not None else r_h_wall
    payload = {
        "experiment": "r_h",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset, "device": device,
        "method": "tc/netem RTT sweep; least-squares slope of wall time vs RTT "
                  "per schedule; r_h = (R_eff(s) - R_eff(00)) / K_h",
        "scope": "end-to-end",
        "sci_internal_rounds_counted": True,
        "rtt_convention": "configured delay = added RTT; on loopback the "
                          "bidirectional netem factor is divided out (delay_mult)",
        "per_schedule": slopes,
        "r_h_per_schedule": per,
        "r_h_wall_slope": round(r_h_wall, 3),
        "sci_rounds_per_schedule": sci_rounds,
        "r_h_measured": round(r_h, 3),
        "r_h_basis": ("1 (①/② exchange) + SCI rounds from client mpc_s vs RTT; "
                      "wall-time slope kept as cross-check (it also counts TCP "
                      "slow-start RTTs of the ciphertext transfers)"
                      if r_sci is not None else "wall-time slope only"),
        "tcp_slow_start_after_idle": sorted(tcp_idle, key=str),
        "protocol_r_h": 1,
        "note": "R_eff(00) should be ~2 (M1 up, M3 down) plus the session "
                "handshake when the context is not reused; the difference "
                "cancels the handshake.",
    }
    out = root / "results" / "rounds" / f"r_h_netem_{dataset}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"r_h (end-to-end) = {r_h:.2f}  [wall-slope cross-check {r_h_wall:.2f}, "
          f"per schedule {per}]  -> {out}")
    if r_sci is not None and r_h_wall > r_h + 1.0:
        print("  note: wall slope exceeds 1 + SCI rounds by more than one RTT per "
              "layer -- likely TCP slow start on the ciphertext transfers; check "
              "tcp_slow_start_after_idle")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--iface", default="eth0",
                    help="netem device; 'lo' is allowed but its delay counts "
                         "twice (both directions), which fit_r_h divides out")
    ap.add_argument("--device", default="pi5", help="label for the results path")
    ap.add_argument("--dataset", default="mnist")
    ap.add_argument("--schedule", default="11",
                    help="must match the server you started (label only)")
    ap.add_argument("--mode", choices=["e3", "e9"], default="e3")
    ap.add_argument("--backend", default="seal")
    ap.add_argument("--output-mode", choices=["logits", "index"], default="logits")
    ap.add_argument("--repeat", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3,
                    help="实验方案 §2.1：每项测量前预热 3 次丢弃")
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--rtts", default=None)
    ap.add_argument("--bws", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fit-only", action="store_true")
    args = ap.parse_args()

    if args.fit_only:
        return 0 if fit_r_h(args.root, args.device, args.dataset) else 1

    from canonical_net import CANONICAL_SPECS
    x = np.random.default_rng(0).uniform(0.0, 1.0,
                                         CANONICAL_SPECS[args.dataset].input_dim)
    if args.mode == "e3":
        rtts = [float(v) for v in (args.rtts.split(",") if args.rtts else E3_RTT)]
        bws = [float(v) for v in (args.bws.split(",") if args.bws else E3_BW)]
        grid = [(r, b, 0.0, 0.0) for r, b in itertools.product(rtts, bws)]
    else:
        rtts = [float(v) for v in (args.rtts.split(",") if args.rtts else E9_RTT)]
        grid = [(r, 50.0, l, j) for r, l, j in itertools.product(rtts, E9_LOSS, E9_JITTER)]

    points = []
    try:
        for rtt, bw, loss, jitter in grid:
            print(f"[point] rtt={rtt} bw={bw} loss={loss}% jitter={jitter}")
            p = run_point(args, x, rtt, bw, loss, jitter)
            if p:
                points.append(p)
                print(f"   wall mean {p['wall_mean_ms']:.1f} ms  p95 {p['wall_p95_ms']:.1f} ms")
    finally:
        tc_clear(args.iface, args.dry_run)
    if args.dry_run:
        return 0
    od = out_dir(args.root, args.device, args.mode, args.dataset, args.schedule)
    od.mkdir(parents=True, exist_ok=True)
    (od / "run.json").write_text(json.dumps({
        "exp_id": "E3" if args.mode == "e3" else "E9",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device, "dataset": args.dataset, "schedule": args.schedule,
        "topology": "client egress netem (delay = full RTT), wired",
        "delay_mult": delay_mult(args.iface),
        "tcp_slow_start_after_idle": slow_start_after_idle(),
        "warmup_discarded": args.warmup, "repeat": args.repeat,
        "measured": True,
        "points": points,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written -> {od / 'run.json'}")
    if args.mode == "e3":
        fit_r_h(args.root, args.device, args.dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
