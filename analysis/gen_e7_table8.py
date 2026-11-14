"""E7 Table 8：仅汇总 results/raw_results/E7/**/*.json 中的消融证据。

不读取其他实验，不执行协议、规划器或成本推导。只有 measured is True、
status 为 ok 且无错误的原始记录可标为实测；缺项始终保留为【待测】。
CLI 参数保留在汇总配置中，原始实验配置逐行保留，不重新计算指标。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
R_H = 6.02
PENDING = "【待测】"
ABLATIONS = (
    "compact_latency", "wo_client_aware", "wo_planner", "wo_stream_keys",
    "wo_share_decrypt", "wo_fusion", "wo_packing", "gc_route", "mask_reuse",
    "wo_offline", "wo_rand_trunc",
)
NAMED_FILES = {name + ".json": name for name in
               ("compact_latency", "wo_client_aware", "wo_planner")}


def within(path: Path, directory: Path) -> bool:
    return path.resolve().is_relative_to(directory.resolve())


def memory_source(root: Path, path: Path | None) -> Path | None:
    if path is None:
        return None
    path = path if path.is_absolute() else root / path
    if not within(path, root / "results" / "raw_results" / "E7"):
        raise ValueError("--memory-summary 只允许 results/raw_results/E7 内的 JSON")
    if path.suffix.lower() != ".json":
        raise ValueError("--memory-summary 必须是 E7 JSON 文件")
    return path.resolve()


def ablation_name(record: dict, fallback: str | None = None) -> str | None:
    name = record.get("ablation")
    config = record.get("config")
    if name is None and isinstance(config, dict):
        name = config.get("ablation")
    return name if isinstance(name, str) and name.strip() else fallback


def has_error(value) -> bool:
    if isinstance(value, dict):
        return bool(value.get("error")) or any(has_error(v) for v in value.values())
    if isinstance(value, list):
        return any(has_error(v) for v in value)
    return False


def evidence_row(name: str, record: dict | None, source, parent: dict | None = None) -> dict:
    record = record or {}
    parent = parent or {}
    measured = (record.get("measured") is True and record.get("status") == "ok"
                and not has_error(record) and not parent.get("error")
                and parent.get("status", "ok") == "ok")
    return {
        "ablation": name,
        "status": "ok" if measured else "pending",
        "evidence": "实测" if measured else PENDING,
        "measured": measured,
        "full": record["full"] if measured and record.get("full") is not None else PENDING,
        "ablated": record["ablated"] if measured and record.get("ablated") is not None else PENDING,
        "effect": record["effect"] if measured and record.get("effect") is not None else PENDING,
        "source": source,
        "config": record.get("config", parent.get("config", {})),
        "git_sha": record.get("git_sha", parent.get("git_sha", "unknown")),
        "device": record.get("device", parent.get("device", "unknown")),
        "timestamp": record.get("timestamp", record.get("timestamp_utc",
                         parent.get("timestamp", parent.get("timestamp_utc", PENDING)))),
        "record": record or None,
    }


def collect_rows(root: Path, memory_summary: Path | None = None) -> tuple[list[dict], list[dict]]:
    root = root.resolve()
    memory = memory_source(root, memory_summary)
    base = root / "results" / "raw_results" / "E7"
    rows = []
    issues = []
    seen = set()
    for path in sorted(base.rglob("*.json")):
        if not path.is_file() or not within(path, base):
            continue
        source = path.relative_to(root).as_posix()
        fallback = None
        if any(part in ("ablation", "ablations") for part in path.relative_to(base).parts[:-1]):
            fallback = NAMED_FILES.get(path.name)
        if memory is not None and path.resolve() == memory:
            fallback = "wo_stream_keys"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeError) as exc:
            issues.append({"source": source, "error": str(exc)})
            if fallback:
                rows.append(evidence_row(fallback, None, source))
                seen.add(fallback)
            continue
        if isinstance(payload, dict):
            if payload.get("record_type") == "summary" or payload.get("summary") is True:
                continue
            nested = payload.get("rows")
            named_rows = [item for item in nested if isinstance(item, dict) and ablation_name(item)] \
                if isinstance(nested, list) else []
            if named_rows:
                records = [(item, payload) for item in named_rows]
            else:
                records = [(payload, None)]
        elif isinstance(payload, list):
            records = [(item, None) for item in payload if isinstance(item, dict)]
        else:
            issues.append({"source": source, "error": "E7 JSON 必须是对象或记录列表"})
            records = []
        for record, parent in records:
            name = ablation_name(record, fallback)
            if name:
                rows.append(evidence_row(name, record, source, parent))
                seen.add(name)
    for name in ABLATIONS:
        if name not in seen:
            rows.append(evidence_row(name, None, PENDING))
    if memory is not None and not memory.is_file():
        issues.append({"source": str(memory), "error": "E7 memory source 不存在，保持【待测】"})
    return rows, issues


def write_summary(root: Path, out: Path, config: dict, rows: list[dict], issues: list[dict]) -> Path:
    target = out / "e7_table8.json"
    raw = root / "results" / "raw_results"
    raw_path = (within(target, raw)
                or any(part.casefold() == "raw_results"
                       for part in (*target.absolute().parts, *target.resolve().parts)))
    overwrite = not raw_path
    if not overwrite and (target.exists() or target.is_symlink()):
        raise FileExistsError(f"禁止覆盖 raw_results 已有文件：{target}")

    # 共享帮助文件由主线程维护；此消费者不提供另一套写入实现。
    if __package__:
        from .e7_io import atomic_json, metadata
    else:
        from e7_io import atomic_json, metadata
    payload = {
        **metadata(config, measured=False, device="desktop", root=root),
        "exp_id": "E7",
        "record_type": "summary",
        "summary": True,
        "measured": False,
        "evidence_scope": "汇总本身不是实测；证据按行判定，缺项为【待测】",
        "r_h": R_H,
        "r_h_scope": "端到端口径，仅声明，不推导轮数",
        "acc_floor": config["acc_floor"],
        "rtt_ms": config["rtt_ms"],
        "bandwidth_mbps": config["bandwidth_mbps"],
        "rows": rows,
        "issues": issues,
    }
    atomic_json(target, payload, overwrite=overwrite)
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--acc-floor", type=float, default=0.82)
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--bandwidth-mbps", type=float, default=50.0)
    ap.add_argument("--memory-summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="汇总输出目录")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    out = args.out or root / "results" / "E7"
    try:
        memory = memory_source(root, args.memory_summary)
        rows, issues = collect_rows(root, memory)
        config = {"root": str(root), "acc_floor": args.acc_floor, "rtt_ms": args.rtt_ms,
                  "bandwidth_mbps": args.bandwidth_mbps,
                  "memory_summary": str(memory) if memory is not None else None,
                  "out": str(out)}
        target = write_summary(root, out, config, rows, issues)
    except (ValueError, OSError) as exc:
        ap.error(str(exc))
    for record in rows:
        print(f"{record['ablation']:<18} {record['evidence']}  {record['source']}")
    print(f"汇总（measured=False）-> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
