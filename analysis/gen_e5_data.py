"""E5 数据生成：Planner 策略对比 + Pareto 前沿（fast / VGG）。

输出 results/E5/e5_planner_<arch>.json 与 .csv，口径均为 SCI 实测的
r_h=1；同时保留 r_h=41（python-OT 参考）用于对照。

六策略定义（可在论文中直接引用）：
  all_relu        全部非线性层 ReLU（精度上限基线）
  all_p2          全部 P2（成本下限基线）
  rule            敏感度/元素数排序 + 深度修复
  greedy          逐层翻转最优增益/成本
  cons_time       constrained：满足 acc_min 下最小时间
  cons_mpc        constrained：满足 acc_min 下最少 MPC 元素
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import planner as P

ARCHS = ["fast", "vgg"]


def layers_and_acc(arch: str):
    if arch in P.CANONICAL_MODELS:
        c = P.CANONICAL_MODELS[arch]
        return c["layers"], c["accuracies"], c["slots"]
    layers = P.FAST_LAYERS if arch == "fast" else P.VGG_LAYERS
    return layers, None, None


def row(schedule, layers, nbytes_per_element, rtt_ms, per_element_ms):
    return {
        "schedule": "".join(str(b) for b in schedule.bits),
        "naming": schedule.describe(layers),
        "accuracy": round(schedule.acc, 6),
        "mpc_elements": schedule.mpc_elements,
        "hybrid_layers": schedule.hybrid_layers,
        "rtts": schedule.rtts,
        "time_s": round(schedule.time_ms / 1000.0, 3),
        "bytes_mb": round(schedule.bytes / 1e6, 3),
        "depth_ok": schedule.depth_ok,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 planner data generator")
    ap.add_argument("--acc-min", type=float, default=0.80)
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--per-element-ms", type=float, default=12.9)
    ap.add_argument("--bytes-per-element", type=int, default=3996)
    ap.add_argument("--out", default="results/E5")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {}

    for arch in ARCHS:
        layers, sched_acc, slots = layers_and_acc(arch)
        base_acc = (
            sched_acc[-1] if sched_acc else (0.8509 if arch == "fast" else 0.8855)
        )
        record = {
            "arch": arch,
            "acc_min": args.acc_min,
            "rtt_ms": args.rtt_ms,
            "per_element_ms": args.per_element_ms,
            "bytes_per_element": args.bytes_per_element,
            "accuracy_source": (
                "measured-canonical-schedule" if sched_acc
                else "combination-penalty-0.0025*k*(k-1)"
            ),
            "strategies": {},
            "pareto": {},
        }

        for r_h in (1, 41):
            profile = P.Profile(
                rtt_ms=args.rtt_ms,
                per_element_ms=args.per_element_ms,
                bytes_per_element=args.bytes_per_element,
                rtts_per_hybrid=r_h,
                base_acc=base_acc,
                schedule_accuracies=sched_acc,
                encoded_input_slots=slots,
            )
            feasible = P.feasible_schedules(layers, profile, args.acc_min)
            front = P.pareto_front(feasible)
            key = f"r_h_{r_h}"
            record["pareto"][key] = {
                "feasible_count": len(feasible),
                "front": [
                    row(s, layers, args.bytes_per_element, args.rtt_ms,
                        args.per_element_ms)
                    for s in front
                ],
            }
            if r_h != 1:
                continue

            # 六策略
            n = len(layers)
            picks = {}
            picks["all_relu"] = tuple([1] * n)
            picks["all_p2"] = tuple([0] * n)
            picks["rule"] = tuple(P.rule_based(layers, profile, args.acc_min))
            picks["greedy"] = tuple(P.greedy(layers, profile, args.acc_min))
            ct = P.constrained(layers, profile, args.acc_min)
            picks["cons_time"] = ct.bits if ct else None
            if feasible:
                min_mpc = min(feasible,
                              key=lambda s: (s.mpc_elements, s.time_ms, s.rtts))
                picks["cons_mpc"] = min_mpc.bits
            else:
                picks["cons_mpc"] = None

            for name, bits in picks.items():
                if bits is None:
                    record["strategies"][name] = None
                    continue
                sch = P.evaluate(bits, layers, profile)
                r = row(sch, layers, args.bytes_per_element, args.rtt_ms,
                        args.per_element_ms)
                r["meets_acc_min"] = sch.acc >= args.acc_min
                r["on_pareto_front"] = any(
                    f.bits == tuple(bits) for f in front)
                record["strategies"][name] = r

        # planner 自身耗时（单次 constrained 求解）
        import time
        profile1 = P.Profile(
            rtt_ms=args.rtt_ms, per_element_ms=args.per_element_ms,
            bytes_per_element=args.bytes_per_element, rtts_per_hybrid=1,
            base_acc=base_acc, schedule_accuracies=sched_acc,
            encoded_input_slots=slots,
        )
        t0 = time.perf_counter()
        P.constrained(layers, profile1, args.acc_min)
        record["planner_solve_ms"] = round((time.perf_counter() - t0) * 1000, 4)
        record["schedule_space"] = 2 ** len(layers)

        (outdir / f"e5_planner_{arch}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

        with (outdir / f"e5_planner_{arch}.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["strategy", "schedule", "naming", "accuracy",
                        "mpc_elements", "hybrid_layers", "rtts", "time_s",
                        "bytes_mb", "on_pareto_front"])
            for name, r in record["strategies"].items():
                if r is None:
                    w.writerow([name, "", "", "", "", "", "", "", "", ""])
                else:
                    w.writerow([name, r["schedule"], r["naming"], r["accuracy"],
                                r["mpc_elements"], r["hybrid_layers"], r["rtts"],
                                r["time_s"], r["bytes_mb"],
                                r["on_pareto_front"]])

        summary[arch] = {
            "pareto_front_h1": len(record["pareto"]["r_h_1"]["front"]),
            "planner_solve_ms": record["planner_solve_ms"],
            "schedule_space": record["schedule_space"],
        }
        print(f"[{arch}] pareto(r_h=1)={summary[arch]['pareto_front_h1']} "
              f"space=2^{len(layers)} solve={record['planner_solve_ms']}ms")
        for name, r in record["strategies"].items():
            if r is None:
                print(f"  {name:10s} INFEASIBLE")
            else:
                print(f"  {name:10s} {r['accuracy']:.2%} mpc={r['mpc_elements']:6d} "
                      f"rtts={r['rtts']:3d} time={r['time_s']:7.1f}s "
                      f"pareto={r['on_pareto_front']} [{r['naming']}]")

    (outdir / "e5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwritten -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())