"""E9: robustness of LitePI under packet loss and jitter.

Experiment §Phase 6: loss {0,1,5}% x jitter ±30 ms -> robustness table.

Why this is an analytic model (and stays one): a true emulated run needs `tc qdisc
netem` on both peers and a protocol batch, but the reference prototype still does
~41 flights per hybrid layer instead of the protocol's r_h=1.  Measuring it would
report the prototype, not the law.  Same reasoning as E3 (docs/实验数据审计.md §12.7),
so E9 reuses E3's measured inputs and only adds the loss/jitter terms.

Model (per query, R = 2 + K_h flights, each flight reliable and independent):
    C            = T_client(s) + T_server(s) + bytes(s)/BW        (compute + serialisation)
    E[T]         = C + R * RTT / (1-p)                            (TCP-like retransmission)
    var_flight   = (2J)^2/12 + RTT^2 * p/(1-p)^2                  (jitter + geometric retx)
    T_p95        = E[T] + 1.645 * sqrt(R * var_flight)
    overhead     = (E[T] - C) / (T_clean - C) = 1/(1-p)           (network term only)

The point of the table is not that 1/(1-p) is surprising; it is that R <= 7 bounds the
amplification, because the protocol merges each hybrid layer into one flight.  A
layer-by-layer (per-element) design with hundreds of flights pays the same 1/(1-p)
hundreds of times over, with the p95 spread growing as sqrt(R).

Run (project venv, matplotlib installed):
    python analysis/gen_e9_data.py --root .
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = str(HERE.parent)

# 复用 E3 的实测输入装载与端到端模型（同一个模型，只加丢包/抖动项）。
_spec = importlib.util.spec_from_file_location("rtt_sensitivity", HERE / "rtt_sensitivity.py")
rtt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rtt)

SCHEDULES = rtt.SCHEDULES                      # s=00 / s=10 / s=11 -> K_h = 0/1/2
RTT_GRID_MS = rtt.RTT_GRID_MS
BW_GRID = rtt.BW_GRID
LOSS_GRID = [0.0, 0.01, 0.05]                   # 实验方案 E9：{0,1,5}%
JITTER_GRID_MS = [0.0, 30.0]                   # 实验方案 E9：±30 ms（均匀分布）
Z95 = 1.6449


def expected_ms(clean_compute_ms: float, flights: int, rtt_ms: float, loss: float) -> float:
    """TCP 式可靠传输下的期望端到端时延。每跳期望重传次数 1/(1-p)。"""
    return clean_compute_ms + flights * rtt_ms / (1.0 - loss)


def p95_ms(clean_compute_ms: float, flights: int, rtt_ms: float,
           loss: float, jitter_ms: float) -> float:
    """95 分位：跳间独立，方差按 flights 线性累加。"""
    var_flight = (2.0 * jitter_ms) ** 2 / 12.0 + (rtt_ms ** 2) * loss / (1.0 - loss) ** 2
    return expected_ms(clean_compute_ms, flights, rtt_ms, loss) + Z95 * (flights * var_flight) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser(description="E9 loss/jitter robustness")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--slack", type=float, default=0.05,
                    help="planner 预算：(1+slack) x s=00 的期望时延")
    ap.add_argument("--r-h", type=float, default=None,
                    help="override the measured r_h (e.g. 7 = CrypTFlow2 log l + 2)")
    args = ap.parse_args()

    root = Path(args.root)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    rtt.configure_r_h(root, args.r_h)
    flights_for = lambda k_h: rtt.rounds_for(k_h)

    cost = rtt.load_hybrid_cost(root)
    rtt.S_HYBRID_PER_LAYER_KB = cost["per_layer_bytes"] / 1024.0
    rtt.T_SERVER_HYBRID_MS = cost["t_server_per_layer_ms"]
    measured = {ds: rtt.load_measured(root, ds) for ds in datasets}

    print("E9 loss/jitter robustness   (per-layer %.1f KB, %.1f ms server MPC)"
          % (rtt.S_HYBRID_PER_LAYER_KB, rtt.T_SERVER_HYBRID_MS))
    print("flights R = 2 + K_h * r_h ; r_h = %g (%s; %s)"
          % (rtt.R_H, rtt.R_H_SCOPE, rtt.R_H_SOURCE))
    print("=" * 100)

    rows = []
    for ds in datasets:
        m = measured[ds]
        print("")
        print("### %s   (T_client=%.2f ms, T_server=%.0f ms, n=%d seeds)"
              % (ds, m["T_client"], m["T_server"], m.get("n_seeds", 0)))
        print("%6s %6s %6s %9s %11s %11s %10s %8s" %
              ("RTT", "BW", "loss", "schedule", "E[T] (ms)", "p95 (ms)", "+ms vs clean", "net x"))
        for rtt_ms in RTT_GRID_MS:
            for bw in BW_GRID:
                for loss in LOSS_GRID:
                    for jitter in JITTER_GRID_MS:
                        for name, k_h in SCHEDULES:
                            flights = flights_for(k_h)
                            clean = rtt.e2e_ms(m, k_h, 0.0, bw)      # RTT=0：计算 + 串行化
                            exp = expected_ms(clean, flights, rtt_ms, loss)
                            p95 = p95_ms(clean, flights, rtt_ms, loss, jitter)
                            rows.append(dict(
                                dataset=ds, schedule=name, K_h=k_h, flights=flights,
                                rtt_ms=rtt_ms, bandwidth_mbps=bw,
                                loss_pct=round(100 * loss, 2), jitter_ms=jitter,
                                compute_ms=round(clean, 3),
                                expected_ms=round(exp, 3), p95_ms=round(p95, 3),
                                added_vs_clean_ms=round(exp - rtt.e2e_ms(m, k_h, rtt_ms, bw), 3),
                                network_overhead_factor=round(1.0 / (1.0 - loss), 4),
                            ))
                            if rtt_ms == 100 and bw == 10.0 and jitter == 30.0:
                                clean_e = rtt.e2e_ms(m, k_h, rtt_ms, bw)
                                print("%6d %6g %5.0f%% %9s %11.1f %11.1f %10.1f %8.3f"
                                      % (rtt_ms, bw, 100 * loss, name, exp, p95,
                                         exp - clean_e, 1.0 / (1.0 - loss)))

    # ---- planner 在丢包链路上的 K_h* ----
    adaptive = []
    for ds in datasets:
        m = measured[ds]
        for rtt_ms in RTT_GRID_MS:
            for bw in BW_GRID:
                for loss in LOSS_GRID:
                    for jitter in JITTER_GRID_MS:
                        cands = []
                        for _name, k_h in SCHEDULES:
                            clean = rtt.e2e_ms(m, k_h, 0.0, bw)
                            cands.append((k_h, p95_ms(clean, flights_for(k_h), rtt_ms, loss, jitter)))
                        cap = (1.0 + args.slack) * cands[0][1]
                        afford = [k for k, t in cands if t <= cap]
                        k_star = max(afford)
                        t_star = dict(cands)[k_star]
                        adaptive.append(dict(
                            dataset=ds, rtt_ms=rtt_ms, bandwidth_mbps=bw,
                            loss_pct=round(100 * loss, 2), jitter_ms=jitter,
                            slack=args.slack, K_h_star=k_star,
                            schedule_star=SCHEDULES[k_star][0],
                            p95_ms=round(t_star, 3),
                            extra_vs_he_only_ms=round(t_star - cands[0][1], 3),
                        ))

    print("")
    print("### K_h*（约束：p95 <= %.0f%% x p95(s=00)，jitter ±30 ms）" % (100 * (1 + args.slack)))
    for ds in datasets:
        print("  %s" % ds)
        print("   %6s %8s %8s %8s %8s" % ("RTT", "10M p0", "10M p5", "100M p0", "100M p5"))
        for rtt_ms in RTT_GRID_MS:
            def cell(bw, loss):
                for r in adaptive:
                    if (r["dataset"] == ds and r["rtt_ms"] == rtt_ms and r["bandwidth_mbps"] == bw
                            and r["loss_pct"] == round(100 * loss, 2) and r["jitter_ms"] == 30.0):
                        return r["K_h_star"]
                return "-"
            print("   %6d %8s %8s %8s %8s" % (rtt_ms, cell(10.0, 0.0), cell(10.0, 0.05),
                                              cell(100.0, 0.0), cell(100.0, 0.05)))

    # ---- 对照：协议 r_h=1（R=2+K_h）vs 逐元素 OT（R=2+108*K_h）----
    # 旧模型每层 27x4=108 次 OT 往返；协议把每层合并成 1 个 flight。
    # 丢包放大随 R 线性、p95 抖动随 sqrt(R) 增长，这里量化「合并成 1 轮」到底省了多少。
    per_element_flights = 108
    comparisons = []
    print("")
    print("### 放大因子对照（s=11, RTT=100 ms, 10 Mbps, jitter ±30 ms）")
    print("   %-10s %8s %14s %14s %10s" % ("loss", "flights R", "+E[T] (ms)", "+p95 (ms)", "p95 x"))
    for loss in LOSS_GRID:
        m0 = measured[datasets[0]]
        clean = rtt.e2e_ms(m0, 2, 0.0, 10.0)                     # 纯计算 + 串行化
        clean_link = rtt.e2e_ms(m0, 2, 100.0, 10.0)              # 协议 s=11、无丢包无抖动
        base_p95 = p95_ms(clean, flights_for(2), 100.0, loss, 30.0)
        for label, flights in (("r_h=%g (%s)" % (rtt.R_H, rtt.R_H_SCOPE), flights_for(2)),
                               ("r_h=7 (CrypTFlow2 log l+2, l=32)", 2 + 7 * 2),
                               ("per-element (108/layer)", 2 + per_element_flights * 2)):
            exp = expected_ms(clean, flights, 100.0, loss)
            p95 = p95_ms(clean, flights, 100.0, loss, 30.0)
            comparisons.append(dict(loss_pct=round(100 * loss, 2), scheme=label,
                                    flights=flights, expected_ms=round(exp, 3),
                                    p95_ms=round(p95, 3),
                                    added_p95_vs_clean_link_ms=round(p95 - clean_link, 3)))
            print("   %-10s %8d %14.1f %14.1f %10.2f"
                  % ("%g%%" % (100 * loss), flights, exp - clean, p95 - clean_link,
                     p95 / base_p95))

    out_dir = root / "results" / "E9"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "e9_robustness.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (out_dir / "e9_adaptive.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(adaptive[0]))
        writer.writeheader()
        writer.writerows(adaptive)

    summary = {
        "exp_id": "E9",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": rtt._git_sha(root),
        "model": "E[T] = C + R*RTT/(1-p); T_p95 = E[T] + 1.645*sqrt(R*((2J)^2/12 + RTT^2*p/(1-p)^2))",
        "grid": dict(rtt_ms=RTT_GRID_MS, bandwidth_mbps=BW_GRID,
                     loss_pct=[round(100 * p, 2) for p in LOSS_GRID], jitter_ms=JITTER_GRID_MS),
        "flights": {name: flights_for(k) for name, k in SCHEDULES},
        "r_h": rtt.R_H, "r_h_scope": rtt.R_H_SCOPE, "r_h_source": rtt.R_H_SOURCE,
        "s_hybrid_per_layer_kb": round(rtt.S_HYBRID_PER_LAYER_KB, 3),
        "t_server_hybrid_ms": round(rtt.T_SERVER_HYBRID_MS, 3),
        "hybrid_cost_source": cost["source"],
        "measured_sources": {ds: measured[ds]["source"] for ds in datasets},
        "measured_seeds": {ds: measured[ds].get("seeds") for ds in datasets},
        "note": "解析模型换算值（丢包/jitter 项为模型项，非 netem 实测）；"
                "计算/字节项全部来自 E2 五种子聚合与 E4 密文阶梯。"
                "总 flight 数上界 R=4（canonical 2 层）或 7（VGG 5 层），这是放大因子的真正上限来源。",
        "r_h_1_vs_per_element": comparisons,
        "rows": len(rows),
    }
    (out_dir / "e9_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
        ds = datasets[0]
        m = measured[ds]
        bw = 50.0
        for loss in LOSS_GRID:
            for jitter, ls in ((0.0, "--"), (30.0, "-")):
                ys = []
                for rtt_ms in RTT_GRID_MS:
                    clean = rtt.e2e_ms(m, 2, 0.0, bw)
                    ys.append(p95_ms(clean, 4, rtt_ms, loss, jitter))
                axes[0].plot(RTT_GRID_MS, ys, marker="o", ls=ls,
                             label="loss %g%% jitter ±%g" % (100 * loss, jitter))
        axes[0].set_xscale("log")
        axes[0].set_xticks(RTT_GRID_MS)
        axes[0].set_xticklabels([str(r) for r in RTT_GRID_MS])
        axes[0].set_xlabel("RTT (ms, log scale)")
        axes[0].set_ylabel("p95 end-to-end latency (ms)")
        axes[0].set_title("%s s=11 @ %g Mbps" % (ds, bw))
        axes[0].grid(alpha=0.3, which="both")
        axes[0].legend(fontsize=7)

        ax2 = axes[1]
        for loss, jitter, ls in ((0.0, 30.0, "--"), (0.05, 30.0, "-")):
            ys = []
            for rtt_ms in RTT_GRID_MS:
                clean = rtt.e2e_ms(m, 4, 0.0, bw)
                clean00 = rtt.e2e_ms(m, 0, 0.0, bw)
                ys.append(p95_ms(clean, 4, rtt_ms, loss, jitter)
                          - p95_ms(clean00, 2, rtt_ms, loss, jitter))
            ax2.plot(RTT_GRID_MS, ys, marker="s", ls=ls,
                     label="loss %g%% jitter ±%g" % (100 * loss, jitter))
        ax2.set_xscale("log")
        ax2.set_xticks(RTT_GRID_MS)
        ax2.set_xticklabels([str(r) for r in RTT_GRID_MS])
        ax2.set_xlabel("RTT (ms, log scale)")
        ax2.set_ylabel("extra p95 latency of s=11 over s=00 (ms)")
        ax2.set_title("%s: hybrid cost under loss" % ds)
        ax2.grid(alpha=0.3, which="both")
        ax2.legend(fontsize=7)

        fig.tight_layout()
        fig_path = root / "datas" / "E9_robustness.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, dpi=150)
        print("")
        print("figure -> %s" % fig_path)
    print("saved -> %s" % out_dir)


if __name__ == "__main__":
    main()
