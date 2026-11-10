"""从精确 E1 原始记录选择调度，不重跑推理。

准确率只对应所测配对样本；时延是 wall_mean 上的 added RTT 加性建模，
不是实测端到端 RTT grid，也不是全测试集或固定权重的精度损失。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .cost_model import load_e1_series, load_r_h
else:
    from cost_model import load_e1_series, load_r_h

ROOT = Path(__file__).resolve().parent.parent
DATASETS = ("mnist", "cifar10", "uci_har")
TAGS = ("00", "01", "10", "11")
SEEDS = (73, 74, 75, 76, 77)
ADDED_RTT_GRID_MS = (1, 5, 20, 50, 100, 200)


def aggregate_schedules(records: list[dict], dataset: str, provenance: dict) -> dict:
    """仅在完整五训练种子、同样本前缀通过校验后聚合原始行。"""
    dataset_runs = [d for d in records if d["dataset"] == dataset]
    by_run = {(str(d["schedule"]), d["seed"]): d for d in dataset_runs}
    if len(by_run) != len(dataset_runs):
        raise ValueError(f"{dataset}: 重复 schedule/训练种子记录，拒绝静默覆盖")
    missing = [(tag, seed) for tag in TAGS for seed in SEEDS if (tag, seed) not in by_run]
    if missing:
        raise ValueError(f"{dataset}: 缺完整五种子原始记录 {missing}；不读取旧 summary")
    prefix, table = None, {}
    for tag in TAGS:
        runs = [by_run[tag, seed] for seed in SEEDS]
        all_rows, seed_accs = [], []
        for d in runs:
            samples = d["per_sample"]
            if any(any(key not in row for key in ("sample_id", "label", "prepared_input_sha256",
                                                  "plain_pred", "cipher_pred")) for row in samples):
                raise ValueError(f"{dataset}/s{tag}/seed{d['seed']}: status={provenance['status']}；"
                                 "缺原始配对样本身份/预测，不能验证同采样前缀；legacy summary 不可替代")
            current = [(r["sample_id"], r["label"], r["prepared_input_sha256"]) for r in samples]
            if prefix is not None and current != prefix:
                raise ValueError(f"{dataset}/s{tag}/seed{d['seed']}: 样本数或采样前缀不一致")
            prefix = current
            if any(r.get(key) is None for r in samples for key in ("up_bytes", "down_bytes")):
                raise ValueError(f"{dataset}/s{tag}/seed{d['seed']}: 缺完整逐样本通信字节")
            seed_accs.append(100.0 * sum(r["cipher_pred"] == r["label"] for r in samples) / len(samples))
            all_rows.extend(samples)
        count = len(all_rows)
        table[tag] = {
            "measured": False, "evidence_kind": "modeled_from_measured_records",
            "status": provenance["status"], "e1_suffix": provenance["e1_suffix"],
            "acc": 100.0 * sum(r["cipher_pred"] == r["label"] for r in all_rows) / count,
            "acc_std": st.stdev(seed_accs),
            "plain_acc": 100.0 * sum(r["plain_pred"] == r["label"] for r in all_rows) / count,
            "prediction_agreement": sum(r["cipher_pred"] == r["plain_pred"] for r in all_rows) / count,
            "wall_s": st.fmean(r["wall_s"] for r in all_rows),
            "k_h": tag.count("1"),
            "S_up": st.fmean(r["up_bytes"] for r in all_rows),
            "S_down": st.fmean(r["down_bytes"] for r in all_rows),
            "n_records": len(runs), "sample_count": count,
            "paired_prefix_samples": len(prefix),
            "sample_count_per_seed": {str(d["seed"]): len(d["per_sample"]) for d in runs},
            "accuracy_scope": "所测配对样本，跨调度 checkpoint 可不同；不是全测试集或固定权重精度损失",
            "records": [r for r in provenance["records"]
                        if r["dataset"] == dataset and r["schedule"] == tag],
        }
    return table


def load_schedules(root: Path, dataset: str, wire: str = "baseline",
                   e1_suffix: str | None = None) -> dict:
    records, _, provenance = load_e1_series(root, wire, e1_suffix, datasets=(dataset,), seeds=SEEDS)
    return aggregate_schedules(records, dataset, provenance)


def model_added_rtt(wall_mean: float, k_h: int, added_rtt_ms: float,
                    r_h: float, r_h_scope: str) -> float:
    """实测在线 wall 均值上的加性外推，不是实测端到端时延。"""
    if r_h_scope != "end-to-end":
        raise ValueError(f"拒绝 RTT 外推：r_h scope={r_h_scope!r}，需要 end-to-end round estimate，"
                         "control-plane/protocol-target 不能计入 SCI 内部轮数")
    if (not isinstance(k_h, int) or k_h < 0 or
            any(not math.isfinite(v) or v < 0 for v in (wall_mean, added_rtt_ms, r_h))):
        raise ValueError("wall_mean、K_h、added_rtt_ms 和 r_h 必须有限且非负")
    return wall_mean + (2 + k_h * r_h) * added_rtt_ms / 1000.0


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 Adaptive（原始实测配对记录上的加性时延模型）")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--e1-suffix", default=None,
                    help="精确 E1 后缀；默认 _seal[_wire<wire>]_maskv3_v2；legacy 须显式选择且有配对证据")
    ap.add_argument("--acc-floor", type=float, default=1.0,
                    help="所测配对样本相对最佳候选准确率的容差（百分点），非固定权重损失")
    ap.add_argument("--out-dir", type=Path, default=None, help="默认 results/adaptive_v2")
    ap.add_argument("--overwrite", action="store_true", help="显式允许覆写本次输出文件")
    args = ap.parse_args()
    if not math.isfinite(args.acc_floor) or args.acc_floor < 0:
        ap.error("acc-floor 必须有限且非负")
    root = args.root
    out_dir = args.out_dir if args.out_dir is not None else root / "results/adaptive_v2"
    if any(part.lower() == "adaptive_seal" for part in out_dir.resolve().parts):
        ap.error("不得写入旧 adaptive_seal 目录；请选择独立 --out-dir")
    out = out_dir / "summary_adaptive.json"
    if out.exists() and not args.overwrite:
        ap.error(f"拒绝覆写 {out}；请选择 --out-dir 或显式 --overwrite")
    try:
        records, _, provenance = load_e1_series(root, args.wire, args.e1_suffix,
                                               datasets=DATASETS, seeds=SEEDS)
        r_h, scope, source = load_r_h(root, provenance["protocol_identity"].get("mpc_contract"))
        model_added_rtt(0.0, 0, 0.0, r_h, scope)
        tables = {ds: aggregate_schedules(records, ds, provenance) for ds in DATASETS}
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))

    summary = {
        "experiment": "E1", "tier": "Adaptive (modeled from measured records)",
        "measured": False, "evidence_kind": "modeled_from_measured_records",
        "status": provenance["status"], "wire": args.wire, "e1_suffix": provenance["e1_suffix"],
        "acc_floor_pp": args.acc_floor, "added_rtt_grid_ms": list(ADDED_RTT_GRID_MS),
        "r_h": r_h, "r_h_scope": scope, "r_h_source": source,
        "rule": "acc >= best - floor; minimize wall_mean + (2+K_h*r_h)*added_rtt_ms/1000",
        "accuracy_scope": "所测配对样本；候选 checkpoint 可不同，不是全测试集或固定权重精度损失",
        "timing_scope": "会话在线 wall_mean 上的 added RTT 加性外推，不含一次性 setup；"
                        "不是实测端到端 RTT grid；r_h 仅使用 end-to-end 来源",
        "provenance": provenance,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datasets": {},
    }
    for ds, table in tables.items():
        best = max(v["acc"] for v in table.values())
        floor = best - args.acc_floor
        feasible = {t: v for t, v in table.items() if v["acc"] >= floor}
        rows = []
        for added_rtt in ADDED_RTT_GRID_MS:
            pick = min(feasible, key=lambda t: model_added_rtt(
                feasible[t]["wall_s"], feasible[t]["k_h"], added_rtt, r_h, scope))
            v = feasible[pick]
            modeled = model_added_rtt(v["wall_s"], v["k_h"], added_rtt, r_h, scope)
            rows.append({
                "measured": False, "evidence_kind": "modeled_from_measured_records",
                "status": provenance["status"], "added_rtt_ms": added_rtt,
                "schedule": pick, "K_h": v["k_h"],
                "accuracy_pct": v["acc"], "accuracy_std": v["acc_std"],
                "accuracy_scope": v["accuracy_scope"],
                "base_wall_mean_s": v["wall_s"], "modeled_online_s": modeled,
                "S_up_bytes": v["S_up"], "S_down_bytes": v["S_down"],
                "n_records": v["n_records"], "sample_count": v["sample_count"],
                "sample_count_per_seed": v["sample_count_per_seed"],
                "records": v["records"], "r_h": r_h, "r_h_scope": scope, "r_h_source": source,
            })
        summary["datasets"][ds] = {
            "measured": False, "evidence_kind": "modeled_from_measured_records",
            "status": provenance["status"], "best_paired_sample_acc_pct": best,
            "feasible_schedules": sorted(feasible), "schedules": table, "rows": rows,
        }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        stream.write(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"保存建模结果（measured=False，非实测端到端 RTT grid）：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
