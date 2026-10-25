"""E3: RTT sensitivity of LitePI end-to-end latency.

Key law: online rounds  R_online(s) = 2 + K_h   (K_h = number of hybrid layers).
Each extra hybrid layer costs exactly one network round trip, plus one client
decrypt+re-encrypt, plus its own messages.  This script turns the measured
per-query constants (results/raw_results/E2/*) into an end-to-end-latency-vs-RTT
curve for three schedules, which is what the planner uses to decide how many
hybrid layers a link can afford.

Model:
    R_online(s) = 2 + K_h
    T_client(s) = T_client_base + K_h * (T_dec + T_enc)
    T_server(s) = T_server_base + K_h * T_server_hybrid_per_layer  [not measured, conservatively 0]
    bytes(s)    = (S_up + S_down) + K_h * S_hybrid_per_layer
    T_e2e       = T_client + T_server + R_online * RTT + bytes / BW

Run (in the project venv, which has matplotlib):
    python analysis/rtt_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

SCHEDULES = [("s=00 (HE-only)", 0), ("s=10 (1 hybrid)", 1), ("s=11 (all-hybrid)", 2)]
RTT_GRID_MS = [1, 5, 20, 50, 100, 200]
BW_MBPS = 10.0
BW_GRID = [10.0, 50.0, 100.0]
# 这两个常数在 main() 里会被实测值覆盖（§6.3 的 550–650 KB 是紧凑口径，实测是 RNS 编码）。
S_HYBRID_PER_LAYER_KB = 600.0
T_SERVER_HYBRID_MS = 0.0

FALLBACK = {
    "mnist": dict(T_client=12.8, T_enc=5.484, T_dec=1.701, T_server=574.775,
                  S_up=438286.4, S_down=94359.3),
    "cifar10": dict(T_client=17.8, T_enc=7.020, T_dec=1.808, T_server=3393.437,
                    S_up=438113.7, S_down=94355.7),
    "uci_har": dict(T_client=15.7, T_enc=6.668, T_dec=1.821, T_server=2142.151,
                    S_up=438069.6, S_down=94347.7),
}


def _first_existing(cands: list[Path]) -> Path | None:
    for p in cands:
        if p.exists():
            return p
    return None


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:
        return None


def load_hybrid_cost(root: Path) -> dict:
    """§6.3 每 hybrid 层的实测代价：①_k 下行 + ②_k fresh 上行 + OT 字节，以及服务端每层 MPC 时间。

    来源：E4 的逐 level 密文实测（comm_bytes）与 E1 s=11 端到端（hybrid_s11_seal）。
    """
    out = {"per_layer_bytes": S_HYBRID_PER_LAYER_KB * 1024.0,
           "t_server_per_layer_ms": 0.0, "source": "fallback(§6.3 紧凑口径)"}
    e4 = _first_existing([root / "results" / "raw_results" / "E4" / "all" / "comm_bytes" / f"seed{s}.json"
                          for s in (73, 74, 75, 76, 77)])
    e1 = _first_existing([root / "results" / "raw_results" / "E1" / "mnist" / "hybrid_s11_seal" / f"seed{s}.json"
                          for s in (73, 74, 75, 76, 77)])
    if e4:
        ladder = {row["level"]: row["measured_bytes"] for row in
                  json.loads(e4.read_text(encoding="utf-8"))["ciphertext_ladder"]}
        # ① 在 L4/L6 下行（取较小的 L4 作保守估计），② 是 fresh L7；OT 响应按 §6.3 MNIST 上限 2 KB。
        out["per_layer_bytes"] = float(ladder["L4"] + ladder["L7"] + 2048)
        out["source"] = str(e4.relative_to(root))
    if e1:
        d = json.loads(e1.read_text(encoding="utf-8"))
        layers = int(d.get("hybrid_layers") or 1)
        out["t_server_per_layer_ms"] = 1000.0 * float(
            d["server"]["timings"]["hybrid_relu_s"]) / layers
        out["source"] += " + " + str(e1.relative_to(root))
    return out



def load_measured(root: Path, dataset: str) -> dict:
    """E2 实测常数：优先论文主实现（litepi_seal，SEAL 4.x），回退 TenSEAL naive 基线。"""
    cands = [root / "results" / "raw_results" / "E2" / dataset / tag / f"seed{s}.json"
             for tag in ("litepi_seal", "he_only") for s in (73, 74, 75, 76, 77)]
    path = _first_existing(cands)
    if path is None:
        return dict(FALLBACK[dataset], source="fallback", backend="none")
    d = json.loads(path.read_text(encoding="utf-8"))
    cm, co, rf = d["client_metrics"], d["communication"], d["reference"]
    t_client = (cm["T_encode_enc_total"]["mean_ms"] + cm["T_dec"]["mean_ms"]
                + cm["T_post"]["mean_ms"])
    return dict(
        T_client=t_client,
        T_enc=cm["T_enc"]["mean_ms"],
        T_dec=cm["T_dec"]["mean_ms"],
        T_server=rf["T_server"]["mean_ms"],
        S_up=co["S_up_bytes"]["mean_bytes"],
        S_down=co["S_down_bytes"]["mean_bytes"],
        source=str(path.relative_to(root)),
        backend=d.get("backend", "?"),
    )


def e2e_ms(m: dict, k_h: int, rtt_ms: float, bw_mbps: float) -> float:
    rounds = 2 + k_h
    t_client = m["T_client"] + k_h * (m["T_dec"] + m["T_enc"])
    t_server = m["T_server"] + k_h * T_SERVER_HYBRID_MS
    nbytes = m["S_up"] + m["S_down"] + k_h * S_HYBRID_PER_LAYER_KB * 1024.0
    t_trans = nbytes * 8.0 / (bw_mbps * 1e6) * 1000.0
    return t_client + t_server + rounds * rtt_ms + t_trans


def marginal_cost_ms(m: dict, rtt_ms: float, bw_mbps: float) -> float:
    return (rtt_ms + (m["T_dec"] + m["T_enc"])
            + S_HYBRID_PER_LAYER_KB * 1024.0 * 8.0 / (bw_mbps * 1e6) * 1000.0)


def main() -> None:
    ap = argparse.ArgumentParser(description="E3 RTT sensitivity")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--bw", type=float, default=BW_MBPS,
                    help="控制台曲线的带宽；JSON 仍按 BW_GRID 全量落盘")
    ap.add_argument("--slack", type=float, default=0.05,
                    help="§7.3 planner 的 max_time_ms 约束按 (1+slack)*T_e2e(s=00) 取")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    # §6.3：每 hybrid 层的字节与 MPC 时间取实测（E4 密文梯度 / E1 s=11 端到端）。
    global S_HYBRID_PER_LAYER_KB, T_SERVER_HYBRID_MS
    cost = load_hybrid_cost(root)
    S_HYBRID_PER_LAYER_KB = cost["per_layer_bytes"] / 1024.0
    T_SERVER_HYBRID_MS = cost["t_server_per_layer_ms"]

    print("root=%s  bandwidth=%s Mbps  rounds = 2 + K_h" % (root, args.bw))
    print("hybrid per-layer: %.1f KB, %.3f ms server MPC  <- %s"
          % (S_HYBRID_PER_LAYER_KB, T_SERVER_HYBRID_MS, cost["source"]))
    print("=" * 96)

    all_rows = []
    adaptive_rows = []
    curves = {}
    for ds in datasets:
        m = load_measured(root, ds)
        curves[ds] = {}
        print("")
        print("### %s   (T_client=%.1f ms, T_server=%.0f ms, backend=%s, src=%s)"
              % (ds, m["T_client"], m["T_server"], m.get("backend", "?"), m["source"]))
        head = "%9s" % "RTT (ms)" + "".join("%20s" % name for name, _ in SCHEDULES) + "%16s" % "allH - HE"
        print(head)
        for rtt in RTT_GRID_MS:
            values = [e2e_ms(m, k_h, rtt, args.bw) for _, k_h in SCHEDULES]
            curves[ds][rtt] = values
            delta = values[-1] - values[0]
            print("%9d" % rtt + "".join("%20.1f" % v for v in values) + "%16.1f" % delta)
            # JSON 覆盖 实验方案 E3 要求的 RTT × {10,50,100} Mbps × 3 协议模式。
            for bw in BW_GRID:
                for (name, k_h) in SCHEDULES:
                    all_rows.append(dict(dataset=ds, rtt_ms=rtt, bandwidth_mbps=bw,
                                         schedule=name, K_h=k_h,
                                         end_to_end_ms=round(e2e_ms(m, k_h, rtt, bw), 3)))
            # Adaptive：planner 在"延迟余量 ≤ (1+slack)×HE-only"的约束下取最大 K_h
            # （§7.3 Budget.max_time_ms）；RTT 越大剩余余量越少，K_h* 单调下降。
            for bw in BW_GRID:
                base = e2e_ms(m, 0, rtt, bw)
                cap = (1.0 + args.slack) * base
                afford = [k for k in (0, 1, 2) if e2e_ms(m, k, rtt, bw) <= cap]
                k_star = max(afford)
                adaptive_rows.append(dict(
                    dataset=ds, rtt_ms=rtt, bandwidth_mbps=bw,
                    slack=args.slack, latency_cap_ms=round(cap, 3),
                    K_h_star=k_star, schedule_star=SCHEDULES[k_star][0],
                    end_to_end_ms=round(e2e_ms(m, k_star, rtt, bw), 3),
                    extra_latency_vs_he_only_ms=round(e2e_ms(m, k_star, rtt, bw) - base, 3),
                ))
        print("  marginal cost of one extra hybrid layer @RTT=%d ms: %.1f ms"
              % (RTT_GRID_MS[-1], marginal_cost_ms(m, RTT_GRID_MS[-1], args.bw)))

    print("")
    print("### Adaptive K_h*（planner 约束：T_e2e ≤ %.0f%% × HE-only 延迟）" % (100 * (1 + args.slack)))
    print("%-9s %8s" % ("dataset", "RTT (ms)")
          + "".join("%16s" % ("%g Mbps" % bw) for bw in BW_GRID))
    for ds in datasets:
        for rtt in RTT_GRID_MS:
            cells = ["K_h*=%d" % row["K_h_star"] for row in adaptive_rows
                     if row["dataset"] == ds and row["rtt_ms"] == rtt]
            print("%-9s %8d" % (ds, rtt) + "".join("%16s" % c for c in cells))

    out_dir = root / "results" / "raw_results" / "E3" / "all" / "rtt_scan"
    if not args.no_save:
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "exp_id": "E3", "dataset": "all", "method": "rtt_scan", "seed": 73,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_sha": _git_sha(root),
            "model": "T_e2e = T_client(s) + T_server(s) + (2+K_h)*RTT + bytes(s)/BW",
            "bandwidth_mbps": args.bw,
            "bandwidth_grid_mbps": BW_GRID,
            "slack_for_adaptive": args.slack,
            "s_hybrid_per_layer_kb": S_HYBRID_PER_LAYER_KB,
            "t_server_hybrid_ms": T_SERVER_HYBRID_MS,
            "hybrid_cost_source": cost["source"],
            "measured_sources": {ds: load_measured(root, ds)["source"] for ds in datasets},
            "note": "per-hybrid-layer bytes from E4 ciphertext ladder (L4 down + L7 fresh up + 2KB OT); "
                    "per-layer server MPC time from E1 s=11; r_h=1 per protocol §7.1",
            "delphi_baseline": "【待补】Delphi 复现版未就绪（实验方案 W5–6 关键路径），本表只含 LitePI 三个配置点",
            "rows": all_rows,
            "adaptive": adaptive_rows,
        }
        (out_dir / "seed73.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        with (out_dir / "seed73.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["dataset", "rtt_ms", "bandwidth_mbps", "schedule", "K_h",
                                "end_to_end_ms"])
            writer.writeheader()
            writer.writerows(all_rows)
        with (out_dir / "adaptive.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(adaptive_rows[0]))
            writer.writeheader()
            writer.writerows(adaptive_rows)
        print("")
        print("saved -> %s" % out_dir)

    # ---- marginal cost of one extra hybrid layer over a (RTT, BW) grid ----
    bw_grid = [10.0, 50.0, 100.0]
    ref = load_measured(root, datasets[0])
    print("")
    print("每多插一层 hybrid 的边际代价 (ms) = RTT + 一次解密再加密 + %.0f KB/带宽"
          % S_HYBRID_PER_LAYER_KB)
    print("%9s" % "RTT (ms)" + "".join("%14s" % ("%g Mbps" % bw) for bw in bw_grid))
    marg_rows = []
    for rtt in RTT_GRID_MS:
        cells = [marginal_cost_ms(ref, rtt, bw) for bw in bw_grid]
        print("%9d" % rtt + "".join("%14.1f" % c for c in cells))
        for bw, c in zip(bw_grid, cells):
            marg_rows.append(dict(rtt_ms=rtt, bandwidth_mbps=bw,
                                  marginal_cost_ms=round(c, 3)))

    if not args.no_save:
        with (out_dir / "marginal_cost.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["rtt_ms", "bandwidth_mbps", "marginal_cost_ms"])
            writer.writeheader()
            writer.writerows(marg_rows)

    if HAS_MPL:
        fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.8), squeeze=False)
        for ax, ds in zip(axes[0], datasets):
            for idx, (name, _k) in enumerate(SCHEDULES):
                ys = [curves[ds][rtt][idx] for rtt in RTT_GRID_MS]
                ax.plot(RTT_GRID_MS, ys, marker="o", label=name)
            ax.set_title(ds)
            ax.set_xlabel("RTT (ms)")
            ax.set_ylabel("end-to-end latency (ms)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        if not args.no_save:
            fig_path = root / "datas" / "E3_rtt_sensitivity.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fig_path, dpi=150)
            print("figure -> %s" % fig_path)

        fig2, ax2 = plt.subplots(figsize=(6.0, 3.8))
        for bw in bw_grid:
            ys = [marginal_cost_ms(ref, rtt, bw) for rtt in RTT_GRID_MS]
            ax2.plot(RTT_GRID_MS, ys, marker="o", label="%g Mbps" % bw)
        ax2.set_title("marginal cost of one extra hybrid layer")
        ax2.set_xlabel("RTT (ms)")
        ax2.set_ylabel("added latency (ms)")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=9)
        fig2.tight_layout()
        if not args.no_save:
            fig2_path = root / "datas" / "E3_marginal_cost.png"
            fig2.savefig(fig2_path, dpi=150)
            print("figure -> %s" % fig2_path)
    else:
        print("")
        print("[note] matplotlib not available here; run this script in the project venv to get datas/E3_rtt_sensitivity.png")


if __name__ == "__main__":
    main()
