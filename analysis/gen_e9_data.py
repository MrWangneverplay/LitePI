"""E9：按测量环境分组分析 netem 丢包与抖动的实测结果。

读取 results/raw_results/E9/<device>/netem_<dataset>_s<schedule>/run.json，
保留所有设备的点及 run 来源，不按设备优先级合并，也不推测缺失环境元数据。
同一比较环境、数据集、调度和链路条件的重复点显式报错。

放大率只使用同一比较环境、数据集、调度和 RTT 下零丢包、零抖动的基准。
缺失或无效基准、非有限延迟及明确不合规或非真实测量的点跳过并报告。
旧记录缺少端点/接口等关键元数据时只保留描述性放大率，不生成调度排名。
最低 p95 仅在同环境、同数据集、同 RTT/丢包/抖动条件下，观测到且达到
精度门槛的候选调度之间选择，不代表全局最优。

    python analysis/gen_e9_data.py --device pi5,wsl

默认写入 results/E9/e9_grouped_v2.{json,csv}；--out-prefix 可指定其他前缀。
已有输出默认拒绝覆盖，只有显式提供 --overwrite 才允许覆盖指定输出。
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import planner as P
from data_manifest import git_sha
DEFAULT_ACC_MIN = P.DEFAULT_ACC_MIN

ROOT = Path(__file__).resolve().parent.parent
COMPARISON_CONTEXT_FIELDS = (
    "device", "topology", "iface", "host", "port", "backend", "output_mode",
    "delay_mult", "bandwidth_mbps", "wire",
)


def comparison_context(row: dict) -> tuple:
    """返回按 COMPARISON_CONTEXT_FIELDS 排序的可哈希环境键。

    缺失字段保持 None；非空 wire 用排序键的 JSON 字符串表示，不从设备名猜拓扑。
    """
    return tuple(json.dumps(row.get(field), sort_keys=True, ensure_ascii=False)
                 if field == "wire" and row.get(field) is not None else row.get(field)
                 for field in COMPARISON_CONTEXT_FIELDS)


def comparison_metadata_complete(row: dict) -> bool:
    # Unlimited bandwidth may legitimately be None. Missing endpoints/interface
    # in legacy records, however, cannot establish a fair cross-schedule ranking.
    required = ("device", "topology", "iface", "host", "port", "backend",
                "output_mode", "delay_mult", "wire")
    return all(row.get(field) is not None for field in required)


def _check_duplicates(rows: list[dict]) -> None:
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (comparison_context(row), row["dataset"], row["schedule"],
               row["rtt_ms"], row["loss_pct"], row["jitter_ms"])
        if key in seen:
            previous = seen[key]
            raise ValueError(
                f"同环境/调度/链路条件出现重复点：{key!r}；"
                f"来源 {previous.get('run_path')} 与 {row.get('run_path')}"
            )
        seen[key] = row


def load_points(root: Path, devices: list[str]) -> list[dict]:
    """保留每个设备的所有点及 run 元数据，同环境重复点显式报错。"""
    rows: list[dict] = []
    for device in devices:
        pattern = str(root / "results" / "raw_results" / "E9" / device / "netem_*_s*" / "run.json")
        for path in sorted(glob.glob(pattern)):
            source = Path(path)
            dataset, schedule = re.fullmatch(r"netem_(.+)_s(\d+)", source.parent.name).groups()
            run = json.loads(source.read_text(encoding="utf-8"))
            metadata = {key: value for key, value in run.items() if key != "points"}
            for point in run["points"]:
                row = {**metadata, **point,
                       "dataset": dataset, "schedule": schedule,
                       "device": run.get("device", device),
                       "source_device": device,
                       "run_path": str(source.resolve()),
                       "git_sha": run.get("git_sha"),
                       "run_timestamp_utc": run.get("timestamp_utc"),
                       "run_metadata": metadata}
                # 目录中的 device 是已有结果标签，其他环境字段不补猜测值。
                for field in COMPARISON_CONTEXT_FIELDS:
                    row.setdefault(field, None)
                rows.append(row)
    _check_duplicates(rows)
    return rows


def _nonmeasurement(value) -> bool:
    if isinstance(value, str):
        return re.search(r"(?:^|[^a-z0-9])(stub|mock|dummy|simulated|simulation|synthetic|analytic)"
                         r"(?:$|[^a-z0-9])", value, flags=re.IGNORECASE) is not None
    if isinstance(value, dict):
        return any((key in ("stub", "is_stub", "mock", "simulated", "synthetic", "dry_run")
                    and item is True) or _nonmeasurement(item)
                   for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_nonmeasurement(item) for item in value)
    return False


def _measurement_exclusion(row: dict) -> str | None:
    for source in (row, row.get("run_metadata", {})):
        if source.get("deployment_compliant") is False:
            return "deployment_compliant 明确为 false"
        if source.get("protocol_failures"):
            return "protocol_failures 非空"
        if source.get("measured") is False or source.get("dry_run") is True:
            return "明确标记为非真实测量"
        if _nonmeasurement({key: source.get(key) for key in (
                "backend", "output_mode", "mpc_runtime", "wire", "profile_name",
                "stub", "is_stub", "mock", "simulated", "synthetic", "dry_run")}):
            return "stub/mock/模拟等非真实测量"
    return None


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def amplification(rows: list[dict], skipped: list[dict] | None = None) -> list[dict]:
    """计算同环境放大率；跳过原因输出到 stderr，也可收集到 skipped。"""
    _check_duplicates(rows)

    def report(row: dict, reason: str) -> None:
        entry = {field: row.get(field) for field in COMPARISON_CONTEXT_FIELDS}
        entry.update({key: row.get(key) for key in (
            "dataset", "schedule", "rtt_ms", "loss_pct", "jitter_ms", "run_path", "git_sha")})
        entry["reason"] = reason
        if skipped is not None:
            skipped.append(entry)
        print(f"跳过 E9 点：{reason}；环境={comparison_context(row)!r}；"
              f"dataset={row['dataset']} schedule={row['schedule']} "
              f"RTT={row['rtt_ms']} loss={row['loss_pct']} jitter={row['jitter_ms']}；"
              f"来源={row.get('run_path')}", file=sys.stderr)

    measured = []
    for row in rows:
        reason = _measurement_exclusion(row)
        if reason is not None:
            report(row, reason)
        else:
            measured.append(row)
    clean = {(comparison_context(row), row["dataset"], row["schedule"], row["rtt_ms"]): row
             for row in measured if row["loss_pct"] == 0 and row["jitter_ms"] == 0}
    out = []
    for row in measured:
        if not all(_finite_number(row.get(field)) and row[field] > 0
                   for field in ("wall_mean_ms", "wall_p95_ms")):
            report(row, "延迟缺失或不是有限数值")
            continue
        base = clean.get((comparison_context(row), row["dataset"], row["schedule"], row["rtt_ms"]))
        if base is None:
            report(row, "缺少同环境/数据集/调度/RTT 的零丢包零抖动基准")
            continue
        if not all(_finite_number(base.get(field)) and base[field] > 0
                   for field in ("wall_mean_ms", "wall_p95_ms")):
            report(row, "基准延迟缺失、非有限或非正")
            continue
        added_mean = row["wall_mean_ms"] - base["wall_mean_ms"]
        added_p95 = row["wall_p95_ms"] - base["wall_p95_ms"]
        mean_x = row["wall_mean_ms"] / base["wall_mean_ms"]
        p95_x = row["wall_p95_ms"] / base["wall_p95_ms"]
        if not all(_finite_number(value) for value in (added_mean, added_p95, mean_x, p95_x)):
            report(row, "放大率或延迟差不是有限数值")
            continue
        out.append({
            **{field: row.get(field) for field in COMPARISON_CONTEXT_FIELDS}, **row,
            "K_h": row["schedule"].count("1"),
            "comparison_metadata_complete": comparison_metadata_complete(row),
            "effective_rtt_ms": (row["rtt_ms"] * row["delay_mult"]
                                 if _finite_number(row.get("rtt_ms"))
                                 and _finite_number(row.get("delay_mult")) else None),
            "baseline_run_path": base.get("run_path"),
            "baseline_git_sha": base.get("git_sha"),
            "added_mean_ms": added_mean, "added_p95_ms": added_p95,
            "mean_x": mean_x, "p95_x": p95_x,
        })
    return out


def best_schedules(points: list[dict], acc_floor: float | None) -> list[dict]:
    """仅在同环境同链路条件下，选择观测到且达到精度门槛的最低 p95 候选。"""
    _check_duplicates(points)
    grouped: dict[tuple, list[dict]] = {}
    for point in points:
        if _measurement_exclusion(point) is not None or not _finite_number(point.get("wall_p95_ms")):
            continue
        key = (comparison_context(point), point["dataset"], point["rtt_ms"],
               point["loss_pct"], point["jitter_ms"])
        grouped.setdefault(key, []).append(point)
    out = []
    for (_, dataset, rtt, loss, jitter), observed in grouped.items():
        if not all(comparison_metadata_complete(point) for point in observed):
            print(f"不排名元数据不完整的 E9 环境：{comparison_context(observed[0])!r}；"
                  f"dataset={dataset}，仅保留各调度的描述性放大率。", file=sys.stderr)
            continue
        arch = "conv3" if dataset == "conv3" else dataset
        accs = P.arch_spec(arch)[1]
        floor = DEFAULT_ACC_MIN[arch] if acc_floor is None else acc_floor
        candidates = [point for point in observed if accs.get(point["schedule"], -1.0) >= floor]
        if candidates:
            best = min(candidates, key=lambda point: (point["wall_p95_ms"], point["schedule"]))
            out.append({
                **{field: best.get(field) for field in COMPARISON_CONTEXT_FIELDS},
                "dataset": dataset, "rtt_ms": rtt, "loss_pct": loss, "jitter_ms": jitter,
                "schedule": best["schedule"], "K_h": best["schedule"].count("1"),
                "wall_p95_ms": best["wall_p95_ms"],
                "run_path": best.get("run_path"), "git_sha": best.get("git_sha"),
                "run_metadata": best.get("run_metadata"),
                "observed_schedules": sorted({point["schedule"] for point in observed}),
                "candidate_schedules": sorted({point["schedule"] for point in candidates}),
                "acc_floor": floor,
                "selection_scope": "observed_accuracy_eligible_candidates_in_same_context",
                "selection_note": "仅为同环境内观测到且达到精度门槛的候选，不代表全局最优。",
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--device", default="pi5,wsl", help="逗号分隔的设备标签；全部保留，不设优先级")
    ap.add_argument("--acc-floor", type=float, default=None)
    ap.add_argument("--out-prefix", type=Path, default=Path("e9_grouped_v2"),
                    help="输出前缀，追加 .json/.csv；相对路径位于 <root>/results/E9 下")
    ap.add_argument("--overwrite", action="store_true", help="显式允许覆盖指定前缀的两个输出文件")
    args = ap.parse_args()
    devices = [device.strip() for device in args.device.split(",") if device.strip()]
    prefix = args.out_prefix if args.out_prefix.is_absolute() else args.root / "results" / "E9" / args.out_prefix
    csv_path = Path(f"{prefix}.csv")
    json_path = Path(f"{prefix}.json")
    existing = [path for path in (json_path, csv_path) if path.exists()]
    if existing and not args.overwrite:
        print("拒绝覆盖已有输出：" + ", ".join(map(str, existing))
              + "；请更换 --out-prefix 或显式提供 --overwrite。", file=sys.stderr)
        return 1

    try:
        rows = load_points(args.root, devices)
    except ValueError as exc:
        print(f"无法聚合 E9：{exc}", file=sys.stderr)
        return 1
    if not rows:
        print(f"未找到设备 {args.device} 的 netem E9 点；未写入输出。", file=sys.stderr)
        return 1
    skipped: list[dict] = []
    points = amplification(rows, skipped)
    if not points:
        print(f"没有可计算放大率的真实测量点（已报告 {len(skipped)} 个跳过点）；未写入输出。",
              file=sys.stderr)
        return 1
    best = best_schedules(points, args.acc_floor)
    payload = {
        "exp_id": "E9", "schema_version": 2, "git_sha": git_sha(args.root),
        "devices": devices, "measured": True,
        "comparison_context_fields": list(COMPARISON_CONTEXT_FIELDS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "points": points, "best_schedule_per_condition": best, "skipped_points": skipped,
        "selection_note": "最低 p95 仅针对同环境、同链路条件下观测到且达到精度门槛的候选。",
    }
    serialized = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    fieldnames = list(dict.fromkeys(field for point in points for field in point))
    prefix.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "x"
    try:
        with csv_path.open(mode, newline="", encoding="utf-8") as csv_fh, \
                json_path.open(mode, encoding="utf-8") as json_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({key: json.dumps(value, sort_keys=True, ensure_ascii=False)
                              if isinstance(value, (dict, list, tuple)) else value
                              for key, value in point.items()} for point in points)
            json_fh.write(serialized)
    except FileExistsError as exc:
        print(f"拒绝覆盖已有输出：{exc.filename}；请更换前缀或显式提供 --overwrite。", file=sys.stderr)
        return 1

    print(f"{'device':<10}{'dataset':<9}{'s':<6}{'RTT':>5}{'loss':>6}{'jit':>5}"
          f"{'mean ms':>11}{'p95 ms':>11}{'mean x':>8}{'p95 x':>8}")
    for point in points:
        print(f"{str(point['device']):<10}{point['dataset']:<9}{point['schedule']:<6}"
              f"{point['rtt_ms']:>5.0f}{point['loss_pct']:>5.0f}%"
              f"{point['jitter_ms']:>5.0f}{point['wall_mean_ms']:>11.0f}{point['wall_p95_ms']:>11.0f}"
              f"{point['mean_x']:>8.2f}{point['p95_x']:>8.2f}")
    print("同一环境内观测到且达到精度门槛的候选中最低 p95（非全局最优）：")
    for choice in best:
        print(f"  环境={comparison_context(choice)!r} {choice['dataset']:<9} "
              f"RTT {choice['rtt_ms']:.0f} loss {choice['loss_pct']:.0f}% "
              f"jitter {choice['jitter_ms']:.0f}: s{choice['schedule']} "
              f"(K_h={choice['K_h']}, p95 {choice['wall_p95_ms']:.0f} ms, "
              f"候选={choice['candidate_schedules']})")
    print(f"已写入 -> {json_path}, {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
