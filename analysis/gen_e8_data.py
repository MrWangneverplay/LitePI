"""E8 设备异构性（实验方案 §7 E8，Table 10）。

方案的预期是"弱设备上 planner 更保守"。客户端计算只占端到端的很小一部分，算力
本身几乎不进入目标函数，所以只列各设备选出的调度证明不了这件事。这里回答的是可证
伪的问题：

  (1) 各设备用自己的实测成本与自己的链路（device_profiles.json），调度是否不同；
  (2) 算力盈亏点：客户端慢多少倍，planner 才改变选择；
  (3) 带宽与 RTT 盈亏点：链路差到什么程度，planner 才改变选择；
  (4) 内存盈亏点：可用内存低到多少，调度才变成不可行。

(2)-(4) 在实测成本上扫描，扫描范围内没有变化就如实报告"不改变"：这说明决定调度
的是网络和精度，不是设备算力，不能写成"profile 感知"。

成本来自 cost_model.fit(device)。设备没有 hybrid 实测时，MPC 按加解密比例从桌面换算，
provenance 字段会注明；这种行不能作为实测结论。

    python analysis/gen_e8_data.py
    python analysis/gen_e8_data.py --speed-grid 1,10,100,1000
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

import planner as P
from cost_model import fit
from data_manifest import git_sha
from gen_e5_data import ARCHS, DEFAULT_ACC_MIN, row

PROFILES = Path(__file__).resolve().parent / "device_profiles.json"
SPEED_GRID = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 10000)
RTT_GRID = (1, 5, 20, 50, 100, 200, 500)
BW_GRID = (1000, 100, 50, 20, 10, 5, 2, 1)
MEMORY_GRID = (8192, 4096, 2048, 1024, 512, 256, 128, 64)


def slowed(cost, factor: float):
    return replace(cost, t_enc_input_ms=cost.t_enc_input_ms * factor,
                   t_dec_ms=cost.t_dec_ms * factor,
                   t_reenc_ms=cost.t_reenc_ms * factor,
                   t_mpc_fixed_ms=cost.t_mpc_fixed_ms * factor,
                   t_mpc_per_elem_ms=cost.t_mpc_per_elem_ms * factor)


def choice(layers, profile, acc_min) -> str | None:
    sch = P.constrained(layers, profile, acc_min)
    return None if sch is None else sch.key()


def first_change(layers, profile, acc_min, grid, make) -> dict:
    """First grid value whose choice differs from the first grid value's."""
    base = choice(layers, make(profile, grid[0]), acc_min)
    for value in grid[1:]:
        got = choice(layers, make(profile, value), acc_min)
        if got != base:
            return {"from": base, "to": got, "at": value, "changed": True}
    return {"from": base, "to": base, "at": None, "changed": False}


def main() -> int:
    ap = argparse.ArgumentParser(description="E8 device heterogeneity")
    ap.add_argument("--profiles", type=Path, default=PROFILES)
    ap.add_argument("--wire", default="baseline")
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
            layers, accuracies, input_blocks = P.arch_spec(arch)
            if not accuracies:
                skipped[f"{dev['name']}/{arch}"] = "no trained schedules"
                continue
            acc_min = DEFAULT_ACC_MIN[arch]
            profile = P.Profile(cost=cost, accuracies=accuracies,
                                rtt_ms=dev["rtt_ms"], bandwidth_mbps=dev["bandwidth_mbps"],
                                input_blocks=input_blocks, objective="client", arch=arch,
                                max_memory_mb=dev.get("memory_budget_mb"))
            chosen = P.constrained(layers, profile, acc_min)
            free = replace(profile, max_memory_mb=None)
            mem = cost.memory_mb(0 if chosen is None else chosen.hybrid_layers)
            rows.append({
                "device": dev["name"], "cpu": dev["cpu"], "arch": arch,
                "acc_min": acc_min, "rtt_ms": dev["rtt_ms"],
                "bandwidth_mbps": dev["bandwidth_mbps"],
                "memory_budget_mb": dev.get("memory_budget_mb"),
                "chosen": None if chosen is None else row(chosen, layers),
                "speed_breakeven": first_change(
                    layers, free, acc_min, SPEED_GRID,
                    lambda p, f: replace(p, cost=slowed(cost, f))),
                "bandwidth_breakeven": first_change(
                    layers, free, acc_min, BW_GRID,
                    lambda p, b: replace(p, bandwidth_mbps=float(b))),
                "rtt_breakeven": first_change(
                    layers, free, acc_min, RTT_GRID,
                    lambda p, r: replace(p, rtt_ms=float(r))),
                "memory_breakeven": first_change(
                    layers, profile, acc_min, MEMORY_GRID,
                    lambda p, m: replace(p, max_memory_mb=float(m))),
                "memory_mb": mem,
                "cost_provenance": cost.provenance,
            })

    (args.out / "e8_devices.json").write_text(json.dumps(
        {"git_sha": git_sha(), "wire": args.wire, "rows": rows, "skipped": skipped},
        indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out / "e8_table10.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["device", "arch", "rtt_ms", "bandwidth_mbps", "schedule", "K_h",
                    "accuracy", "T_client_ms", "R_online", "M_client_mb",
                    "speed_breakeven_x", "bandwidth_breakeven_mbps", "rtt_breakeven_ms",
                    "memory_breakeven_mb"])
        for r in rows:
            c = r["chosen"] or {}
            w.writerow([r["device"], r["arch"], r["rtt_ms"], r["bandwidth_mbps"],
                        c.get("schedule", "infeasible"), c.get("hybrid_layers"),
                        c.get("accuracy"), c.get("T_client_ms"), c.get("R_online"),
                        r["memory_mb"], r["speed_breakeven"]["at"],
                        r["bandwidth_breakeven"]["at"], r["rtt_breakeven"]["at"],
                        r["memory_breakeven"]["at"]])
    for r in rows:
        c = r["chosen"]
        fmt = lambda b: "none" if not b["changed"] else f"{b['at']} ({b['from']}->{b['to']})"
        print(f"{r['device']:>8} {r['arch']:>8}  "
              f"{'infeasible' if c is None else c['schedule']}  "
              f"speed x{fmt(r['speed_breakeven'])}  bw {fmt(r['bandwidth_breakeven'])}  "
              f"rtt {fmt(r['rtt_breakeven'])}  mem {fmt(r['memory_breakeven'])}")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
