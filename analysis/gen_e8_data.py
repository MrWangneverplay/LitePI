"""E8 设备异构性（实验方案 §7 E8，Table 10）。

同一配置在各档客户端上，由 planner 按该设备的 profile 重新规划：
  * 成本：cost_model.fit(device) —— 设备自己的 E2 实测（加解密、内存峰值），
    MPC 时间在该设备没有 hybrid 实测前按加解密比例从桌面换算（provenance 注明）；
  * 约束：精度下限 + 该设备的内存预算（device_profiles.json 的 memory_budget_mb）。
    hybrid 调度的内存增量未实测时记为未知；在有内存预算的设备上，未知视为
    不满足约束（保守），并在输出里列出被内存约束排除的调度，避免把"未知"写成"可行"。

产出：每设备 × 每网络的选择、T_client、R_online、内存，及归一化
T_client / cpu_score（cpu_score 未填时为 null）。
预期（实验方案）：弱设备上 planner 更保守（K_h 更小）。

    python analysis/gen_e8_data.py
    python analysis/gen_e8_data.py --rtt-ms 100
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import planner as P
from cost_model import fit
from gen_e5_data import ARCHS, DEFAULT_ACC_MIN, layers_and_acc, row

PROFILES = Path(__file__).resolve().parent / "device_profiles.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="E8 device heterogeneity")
    ap.add_argument("--profiles", type=Path, default=PROFILES)
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--bandwidth-mbps", type=float, default=50.0)
    ap.add_argument("--out", type=Path, default=Path("results/E8"))
    args = ap.parse_args()
    devices = json.loads(args.profiles.read_text(encoding="utf-8"))["devices"]
    args.out.mkdir(parents=True, exist_ok=True)

    rows, skipped = [], {}
    for dev in devices:
        try:
            cost = fit(device=dev["name"], wire=args.wire)
        except SystemExit as exc:
            skipped[dev["name"]] = f"no measurements ({exc})"
            print(f"[skip] {dev['name']}: {exc}")
            continue
        for arch in ARCHS:
            layers, sched_acc, slots, base_acc = layers_and_acc(arch)
            acc_min = DEFAULT_ACC_MIN[arch]
            profile = P.Profile(rtt_ms=args.rtt_ms, base_acc=base_acc,
                                schedule_accuracies=sched_acc,
                                encoded_input_slots=slots, cost=cost,
                                bandwidth_mbps=args.bandwidth_mbps,
                                objective="client",
                                max_memory_mb=dev.get("memory_budget_mb"))
            free = P.Profile(**{**profile.__dict__, "max_memory_mb": None})
            chosen = P.constrained(layers, profile, acc_min)
            unconstrained = P.constrained(layers, free, acc_min)
            excluded = [
                "".join(map(str, s.bits))
                for s in P.feasible_schedules(layers, free, acc_min)
                if not P.evaluate(s.bits, layers, profile).memory_ok
            ]
            rec = {"device": dev["name"], "cpu": dev["cpu"], "arch": arch,
                   "acc_min": acc_min, "memory_budget_mb": dev.get("memory_budget_mb"),
                   "chosen": None if chosen is None else row(chosen, layers),
                   "chosen_without_memory_budget": (
                       None if unconstrained is None else row(unconstrained, layers)),
                   "excluded_by_memory": excluded,
                   "memory_increment_measured": cost.mem_per_hybrid_mb is not None,
                   "cost_provenance": cost.provenance}
            score = dev.get("cpu_score")
            rec["T_client_per_cpu_score"] = (
                None if (chosen is None or not score) else chosen.client_ms / score)
            rows.append(rec)

    (args.out / "e8_devices.json").write_text(json.dumps(
        {"rtt_ms": args.rtt_ms, "bandwidth_mbps": args.bandwidth_mbps,
         "wire": args.wire, "rows": rows, "skipped_devices": skipped},
        indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out / "e8_table10.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "arch", "schedule", "K_h", "accuracy", "T_client_ms",
                    "R_online", "M_client_mb", "excluded_by_memory",
                    "T_client_per_cpu_score"])
        for r in rows:
            c = r["chosen"] or {}
            w.writerow([r["device"], r["arch"], c.get("schedule", "infeasible"),
                        c.get("hybrid_layers"), c.get("accuracy"), c.get("T_client_ms"),
                        c.get("R_online"), c.get("M_client_mb"),
                        "|".join(r["excluded_by_memory"]), r["T_client_per_cpu_score"]])
    for r in rows:
        c = r["chosen"]
        print(f"{r['device']:>8} {r['arch']:>8}  "
              + ("infeasible" if c is None else
                 f"{c['schedule']} K_h={c['hybrid_layers']} T_c={c['T_client_ms']:.1f}ms "
                 f"R={c['R_online']} M={c['M_client_mb']}")
              + (f"  excluded_by_memory={r['excluded_by_memory']}"
                 if r["excluded_by_memory"] else ""))
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
