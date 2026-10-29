"""E5 数据生成：Planner 有效性（实验方案 §7 E5 / H5）。

六种调度策略（实验方案原文）：
  all_p2       全 HE-only（全部非线性层 P₂）
  all_relu     全 Hybrid
  random_x5    随机调度 ×5 次平均（不知道精度约束；报告 5 次中满足 acc_min 的次数）
  rule         规则：敏感度/元素数排序 + 深度修复
  greedy       贪心：逐层翻转最优增益/成本
  adaptive     Constrained/Adaptive：满足 acc_min（与内存约束）下目标最小

评价维度（实验方案 E5 六维）：Acc, T_client, R_online, M_client, E_client, L_e2e。
  * 成本全部来自 cost_model.py 拟合的实测常数（E1 SEAL+SCI、E2、E4、r_h 文件），
    不再使用 Python P-256 OT 的 12.9 ms / 3996 B / 41 轮旧常数；
  * E_client 需要功率计（E6），此处记 null；M_client 只有 HE-only 实测，
    含 hybrid 的调度记 null；
  * 深层网络（fast / VGG）没有密文实现：精度来自明文组合惩罚模型，服务器时间
    是浅网拟合外推，均在 JSON 的 basis 字段注明，论文中只能作为模型结果。

Pareto 前沿在 (Acc↑, T_client↓, R_online↓, L_e2e↓) 上计算；
另输出 measure_list：前沿上每个 canonical 调度的实测命令（eval_hybrid_accuracy.py），
用于把模型结果逐个换成实测（PPT 总述页 P1 "前沿上的调度逐个实测"）。

    python analysis/gen_e5_data.py                     # 默认 desktop 成本、RTT 50 ms
    python analysis/gen_e5_data.py --device pi5 --rtt-ms 100 --bandwidth-mbps 10
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import time
from pathlib import Path

import planner as P
from cost_model import CostModel, fit

ARCHS = ["mnist", "cifar10", "uci_har", "fast", "vgg"]
DEFAULT_ACC_MIN = {"mnist": 0.975, "cifar10": 0.59, "uci_har": 0.89,
                   "fast": 0.84, "vgg": 0.84}


def layers_and_acc(arch: str):
    if arch in P.CANONICAL_MODELS:
        c = P.CANONICAL_MODELS[arch]
        return c["layers"], c["accuracies"], c["slots"], c["accuracies"][-1]
    layers = P.FAST_LAYERS if arch == "fast" else P.VGG_LAYERS
    return layers, None, None, (0.8509 if arch == "fast" else 0.8855)


def row(s: P.Schedule, layers) -> dict:
    return {
        "schedule": "".join(str(b) for b in s.bits),
        "naming": s.describe(layers),
        "accuracy": round(s.acc, 6),
        "mpc_elements": s.mpc_elements,
        "hybrid_layers": s.hybrid_layers,
        "R_online": s.rtts,
        "T_client_ms": None if s.client_ms is None else round(s.client_ms, 3),
        "T_server_ms": None if s.server_ms is None else round(s.server_ms, 1),
        "L_e2e_ms": None if s.latency_ms is None else round(s.latency_ms, 1),
        "S_online_bytes": s.bytes,
        "M_client_mb": None if s.memory_mb is None else round(s.memory_mb, 1),
        "E_client_mJ": None,
        "objective_ms": round(s.time_ms, 3),
        "depth_ok": s.depth_ok,
    }


def dominates(a: dict, b: dict) -> bool:
    keys_max = ("accuracy",)
    keys_min = ("T_client_ms", "R_online", "L_e2e_ms")
    ge = all(a[k] >= b[k] for k in keys_max) and all(a[k] <= b[k] for k in keys_min)
    gt = any(a[k] > b[k] for k in keys_max) or any(a[k] < b[k] for k in keys_min)
    return ge and gt


def pareto(rows: list[dict]) -> list[dict]:
    return [r for r in rows if not any(dominates(o, r) for o in rows if o is not r)]


def main() -> int:
    ap = argparse.ArgumentParser(description="E5 planner data generator")
    ap.add_argument("--device", default="desktop")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--cost-model", type=Path, default=None,
                    help="explicit cost JSON (default: fit from results/)")
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--bandwidth-mbps", type=float, default=None)
    ap.add_argument("--acc-min", type=float, default=None,
                    help="one floor for every arch (default: per-arch table)")
    ap.add_argument("--random-draws", type=int, default=5)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", type=Path, default=Path("results/E5"))
    args = ap.parse_args()

    cost: CostModel = (CostModel.load(args.cost_model) if args.cost_model
                       else fit(device=args.device, wire=args.wire))
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"cost_model": cost.to_dict(), "rtt_ms": args.rtt_ms,
               "bandwidth_mbps": args.bandwidth_mbps, "archs": {}}

    for arch in ARCHS:
        layers, sched_acc, slots, base_acc = layers_and_acc(arch)
        acc_min = args.acc_min if args.acc_min is not None else DEFAULT_ACC_MIN[arch]
        profile = P.Profile(rtt_ms=args.rtt_ms, base_acc=base_acc,
                            schedule_accuracies=sched_acc,
                            encoded_input_slots=slots, cost=cost,
                            bandwidth_mbps=args.bandwidth_mbps,
                            objective="client", arch=arch)
        n = len(layers)
        t0 = time.perf_counter()
        adaptive = P.constrained(layers, profile, acc_min)
        solve_ms = (time.perf_counter() - t0) * 1e3

        # Same search restricted to what the ciphertext code can run today
        # (P2 before a conv is not implemented, planner.Layer.next_is_conv).
        exec_profile = P.Profile(**{**profile.__dict__, "executable_only": True})
        adaptive_exec = P.constrained(layers, exec_profile, acc_min)
        picks = {
            "all_p2": P.evaluate([0] * n, layers, profile),
            "all_relu": P.evaluate([1] * n, layers, profile),
            "rule": P.evaluate(P.rule_based(layers, profile, acc_min), layers, profile),
            "greedy": P.evaluate(P.greedy(layers, profile, acc_min), layers, profile),
            "adaptive": adaptive,
            "adaptive_executable": (None if adaptive_exec is None else
                                    P.evaluate(adaptive_exec.bits, layers, profile)),
        }
        rand = P.random_schedules(layers, profile, acc_min, args.random_draws, args.seed)
        all_rows = [row(P.evaluate(bits, layers, profile), layers)
                    for bits in P.itertools.product((0, 1), repeat=n)]
        all_rows = [r for r in all_rows if r["depth_ok"]]
        front = pareto(all_rows)
        front_ids = {r["schedule"] for r in front}

        strategies = {}
        for name, sch in picks.items():
            if sch is None:
                strategies[name] = None
                continue
            r = row(sch, layers)
            r["meets_acc_min"] = sch.acc >= acc_min
            r["on_pareto_front"] = r["schedule"] in front_ids
            r["dominated_by_adaptive"] = (
                adaptive is not None and dominates(row(adaptive, layers), r))
            strategies[name] = r
        rr = [row(s, layers) for s in rand]
        mean = {k: (round(st.mean(r[k] for r in rr), 6)
                    if all(isinstance(r[k], (int, float)) for r in rr) else None)
                for k in ("accuracy", "T_client_ms", "R_online", "L_e2e_ms",
                          "S_online_bytes", "mpc_elements")}
        mean["draws"] = [r["schedule"] for r in rr]
        mean["meets_acc_min"] = sum(r["accuracy"] >= acc_min for r in rr)
        mean["dominated_by_adaptive"] = sum(
            adaptive is not None and dominates(row(adaptive, layers), r) for r in rr)
        strategies["random_x5"] = mean

        canonical = arch in P.CANONICAL_MODELS
        record = {
            "arch": arch, "acc_min": acc_min, "rtt_ms": args.rtt_ms,
            "bandwidth_mbps": args.bandwidth_mbps,
            "cost_device": cost.device, "wire": cost.wire,
            "r_h": cost.r_h, "r_h_scope": cost.r_h_scope,
            "basis": {
                "accuracy": ("measured E1 canonical schedule means" if canonical
                             else "plaintext combination penalty 0.0025*k*(k-1) (model)"),
                "client_cost": "cost_model (E1 SEAL+SCI fit)" + (
                    "" if cost.device == "desktop" else ", device-scaled"),
                "server_cost": ("E1 forward_s mean per (dataset, schedule)" if canonical
                                else "pooled canonical fit extrapolated - NOT a deep-net measurement"),
                "adaptive_executable": ("same as adaptive (no conv after a nonlinear layer)"
                                        if canonical else
                                        "constrained search over schedules the ciphertext "
                                        "code can run: every nonlinear layer before a conv "
                                        "must be hybrid (deep_hybrid.py has no P2->conv)"),
                "E_client": "unmeasured (E6)",
                "M_client": "E2 HE-only peak; hybrid increment unmeasured",
            },
            "strategies": strategies,
            "pareto_front": front,
            "schedule_space": 2 ** n,
            "planner_cost": {"solve_ms": round(solve_ms, 4),
                             "evaluations": 2 ** n,
                             "layer_sensitivity_evals": (0 if canonical else n)},
            "measure_list": [
                {"schedule": r["schedule"],
                 "cmd": (f"python analysis/eval_hybrid_accuracy.py --dataset {arch} "
                         f"--schedule {r['schedule']} --seed <73..77> --samples 0")}
                for r in front if canonical
            ],
        }
        (args.out / f"e5_planner_{arch}.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        with (args.out / f"e5_planner_{arch}.csv").open("w", newline="",
                                                        encoding="utf-8") as fh:
            w = csv.writer(fh)
            cols = ["strategy", "schedule", "accuracy", "T_client_ms", "R_online",
                    "L_e2e_ms", "S_online_bytes", "mpc_elements", "on_pareto_front"]
            w.writerow(cols)
            for name, r in strategies.items():
                if r is None:
                    w.writerow([name] + [""] * (len(cols) - 1))
                else:
                    w.writerow([name] + [r.get(c, "") if c != "schedule"
                                         else r.get("schedule", "|".join(r.get("draws", [])))
                                         for c in cols[1:]])
        summary["archs"][arch] = {
            "adaptive": strategies["adaptive"]["schedule"] if strategies["adaptive"] else None,
            "pareto_size": len(front), "solve_ms": round(solve_ms, 4),
        }
        print(f"[{arch}] acc_min={acc_min:.3f} front={len(front)} solve={solve_ms:.2f}ms")
        for name, r in strategies.items():
            if r is None:
                print(f"   {name:10s} INFEASIBLE")
            elif name == "random_x5":
                print(f"   {name:10s} acc={r['accuracy']:.4f} T_c={r['T_client_ms']:.1f}ms "
                      f"R={r['R_online']} L={r['L_e2e_ms']/1e3:.2f}s  "
                      f"met={r['meets_acc_min']}/5  dominated={r['dominated_by_adaptive']}/5")
            else:
                print(f"   {name:10s} {r['schedule']} acc={r['accuracy']:.4f} "
                      f"T_c={r['T_client_ms']:.1f}ms R={r['R_online']} "
                      f"L={r['L_e2e_ms']/1e3:.2f}s front={r['on_pareto_front']}")

    (args.out / "e5_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
