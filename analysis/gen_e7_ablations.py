"""E7 消融（Table 8）中与 planner 相关的两项。

实验方案 §7 E7：
  w/o planner（固定 HE-only）          —— 自适应的价值
  w/o client-aware 目标函数（改服务器延迟目标）
                                        —— 客户端成本建模的价值：改目标后
                                           客户端成本"反弹"的幅度即 8.3.1 论证一的实证

三种配置在同一精度下限、同一成本模型（cost_model.py，实测常数）下求解：
  full            Adaptive，目标 = 客户端计算 + R·RTT + 字节/带宽（client-aware）
  wo_planner      固定 s = 全 0（HE-only），不做搜索
  wo_client_aware 同一搜索空间与约束，目标 = 服务器计算时间

报告每个配置的 Acc / T_client / R_online / S_online / L_e2e，以及
  rebound_T_client = T_client(wo_client_aware) / T_client(full)
  rebound_R_online = R_online(wo_client_aware) - R_online(full)
扫描 RTT 与设备档位，因为两个目标是否冲突取决于网络与设备。
注意：fast / VGG 的精度与服务器成本是模型值（见 gen_e5_data.py 的 basis 说明）；
精度下限不可达的配置记为 infeasible，不做替代。

    python analysis/gen_e7_ablations.py
    python analysis/gen_e7_ablations.py --devices desktop,pi5 --rtts 1,20,100,200
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import planner as P
from cost_model import fit
from gen_e5_data import ARCHS, DEFAULT_ACC_MIN, layers_and_acc, row


def solve(layers, profile, acc_min, objective):
    prof = P.Profile(**{**profile.__dict__, "objective": objective})
    sch = P.constrained(layers, prof, acc_min)
    if sch is None:
        return None
    # Report every configuration with the same (client-aware) metrics.
    return P.evaluate(sch.bits, layers, profile)


def main() -> int:
    ap = argparse.ArgumentParser(description="E7 planner ablations")
    ap.add_argument("--devices", default="desktop,pi5")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--rtts", default="1,20,50,100,200")
    ap.add_argument("--bandwidth-mbps", type=float, default=50.0)
    ap.add_argument("--out", type=Path, default=Path("results/E7"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    records = []
    for device in args.devices.split(","):
        cost = fit(device=device, wire=args.wire)
        for rtt in (float(v) for v in args.rtts.split(",")):
            for arch in ARCHS:
                layers, sched_acc, slots, base_acc = layers_and_acc(arch)
                acc_min = DEFAULT_ACC_MIN[arch]
                profile = P.Profile(rtt_ms=rtt, base_acc=base_acc,
                                    schedule_accuracies=sched_acc,
                                    encoded_input_slots=slots, cost=cost,
                                    bandwidth_mbps=args.bandwidth_mbps,
                                    objective="client", arch=arch)
                full = solve(layers, profile, acc_min, "client")
                wo_cl = solve(layers, profile, acc_min, "server")
                wo_pl = P.evaluate([0] * len(layers), layers, profile)
                rec = {"device": device, "rtt_ms": rtt, "arch": arch,
                       "acc_min": acc_min, "r_h": cost.r_h,
                       "r_h_scope": cost.r_h_scope}
                for name, sch in (("full", full), ("wo_planner", wo_pl),
                                  ("wo_client_aware", wo_cl)):
                    if sch is None:
                        rec[name] = None
                        continue
                    r = row(sch, layers)
                    r["meets_acc_min"] = sch.acc >= acc_min
                    rec[name] = r
                if full and wo_cl:
                    rec["rebound_T_client"] = round(
                        wo_cl.client_ms / full.client_ms, 3)
                    rec["rebound_R_online"] = wo_cl.rtts - full.rtts
                if full:
                    rec["wo_planner_acc_loss"] = round(full.acc - wo_pl.acc, 6)
                    rec["wo_planner_meets_acc_min"] = wo_pl.acc >= acc_min
                records.append(rec)

    (args.out / "e7_planner_ablations.json").write_text(
        json.dumps({"bandwidth_mbps": args.bandwidth_mbps, "wire": args.wire,
                    "rows": records}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    with (args.out / "e7_planner_ablations.csv").open("w", newline="",
                                                       encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "rtt_ms", "arch", "config", "schedule", "accuracy",
                    "meets_acc_min", "T_client_ms", "R_online", "S_online_bytes",
                    "L_e2e_ms"])
        for rec in records:
            for name in ("full", "wo_planner", "wo_client_aware"):
                r = rec[name]
                if r is None:
                    w.writerow([rec["device"], rec["rtt_ms"], rec["arch"], name,
                                "infeasible"] + [""] * 6)
                else:
                    w.writerow([rec["device"], rec["rtt_ms"], rec["arch"], name,
                                r["schedule"], r["accuracy"], r["meets_acc_min"],
                                r["T_client_ms"], r["R_online"],
                                r["S_online_bytes"], r["L_e2e_ms"]])
    print(f"{'device':>8} {'rtt':>5} {'arch':>8}  full / wo_planner / wo_client_aware"
          f"   rebound(T_client, R)")
    for rec in records:
        f, p, c = rec["full"], rec["wo_planner"], rec["wo_client_aware"]
        fmt = lambda r: "infeas" if r is None else f"{r['schedule']}({r['T_client_ms']:.0f}ms)"
        print(f"{rec['device']:>8} {rec['rtt_ms']:5.0f} {rec['arch']:>8}  "
              f"{fmt(f)} / {fmt(p)}{'' if rec.get('wo_planner_meets_acc_min', True) else '*'}"
              f" / {fmt(c)}   {rec.get('rebound_T_client', '-')}, "
              f"{rec.get('rebound_R_online', '-')}")
    print("(* = HE-only misses the accuracy floor)")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
