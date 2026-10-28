"""E1 Adaptive 档：用**实测密文**结果做调度选择（不重跑推理）。

口径（与 实验方案 §4.2 / §7.3 一致）：
  * 候选调度 s ∈ {00, 01, 10, 11}，每个调度的精度/时延/字节都取本目录下
    `hybrid_s<tag>_acc_seal/summary_5seeds.json` 的 5-seed 实测均值；
  * 约束：acc(s) ≥ best_acc − acc_floor（默认 1.0 个百分点）；
  * 目标：最小化 L_e2e = wall_mean(s) + (2 + K_h)·RTT（r_h = 1，§4.2 的轮数公式）；
  * 输出 Adaptive 选中的调度、其实测精度与端到端延迟，供 E1 Table 5 / E3 子图使用。

用法：
    python analysis/adaptive_eval.py --acc-floor 1.0
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("mnist", "cifar10", "uci_har")
TAGS = ("00", "01", "10", "11")
RTT_GRID_MS = (1, 5, 20, 50, 100, 200)


def load_schedules(root: Path, dataset: str) -> dict:
    table = {}
    for tag in TAGS:
        path = (root / "results" / "raw_results" / "E1" / dataset
                / f"hybrid_s{tag}_acc_seal" / "summary_5seeds.json")
        if not path.exists():
            raise SystemExit(f"缺少 {path}，先跑 _sum_hybrid.py")
        payload = json.loads(path.read_text(encoding="utf-8"))["schedules"][tag]
        table[tag] = {
            "acc": payload["accuracy_pct"]["mean"],
            "acc_std": payload["accuracy_pct"]["std"],
            "wall_s": payload["wall_s"]["mean"],
            "k_h": sum(int(b) for b in tag),
            "S_up": payload["S_up_bytes"]["mean"],
            "S_down": payload["S_down_bytes"]["mean"],
        }
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 Adaptive（基于实测密文调度表）")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--acc-floor", type=float, default=1.0,
                    help="允许的精度损失（百分点），默认 1.0")
    args = ap.parse_args()
    root = Path(args.root)

    summary = {"experiment": "E1", "tier": "Adaptive (measured schedules)",
               "acc_floor_pp": args.acc_floor, "rtt_grid_ms": list(RTT_GRID_MS),
               "rule": "acc >= best - floor; minimize wall + (2+K_h)*RTT",
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "datasets": {}}

    for ds in DATASETS:
        table = load_schedules(root, ds)
        best = max(v["acc"] for v in table.values())
        floor = best - args.acc_floor
        feasible = {t: v for t, v in table.items() if v["acc"] >= floor}
        rows = []
        print(f"\n### {ds}  best={best:.2f}%  floor={floor:.2f}%  "
              f"feasible={sorted(feasible)}")
        print(f"{'RTT(ms)':>8}{'s*':>6}{'K_h':>4}{'acc%':>9}{'wall_s':>9}{'L_e2e_s':>10}")
        for rtt in RTT_GRID_MS:
            pick = min(feasible, key=lambda t: feasible[t]["wall_s"]
                       + (2 + feasible[t]["k_h"]) * rtt / 1000.0)
            v = feasible[pick]
            e2e = v["wall_s"] + (2 + v["k_h"]) * rtt / 1000.0
            rows.append({"rtt_ms": rtt, "schedule": pick, "K_h": v["k_h"],
                         "accuracy_pct": round(v["acc"], 4),
                         "accuracy_std": round(v["acc_std"], 4),
                         "wall_s": round(v["wall_s"], 4),
                         "end_to_end_s": round(e2e, 4),
                         "S_up_bytes": int(v["S_up"]),
                         "S_down_bytes": int(v["S_down"])})
            print(f"{rtt:>8}{pick:>6}{v['k_h']:>4}{v['acc']:>9.2f}"
                  f"{v['wall_s']:>9.2f}{e2e:>10.2f}")
        accs = [r["accuracy_pct"] for r in rows]
        summary["datasets"][ds] = {
            "best_acc_pct": round(best, 4),
            "feasible_schedules": sorted(feasible),
            "accuracy_pct": {"mean": round(st.fmean(accs), 4),
                             "min": min(accs), "max": max(accs)},
            "rows": rows,
        }
        out = (root / "results" / "raw_results" / "E1" / ds
               / "adaptive_seal" / "summary_adaptive.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
                       encoding="utf-8")
    print("\n保存：results/raw_results/E1/<ds>/adaptive_seal/summary_adaptive.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
