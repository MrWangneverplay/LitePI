"""从落盘配对证据计算 E1 一致率，原始结果始终只读。

schema2 不加载模型或数据；仅 --allow-legacy 允许旧记录的明文重算，
且旧记录始终标记为来源未验证。默认输出 results/agreement_v2，
逐记录结果与 summary.json 分开保存，默认拒绝覆盖。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path


def _integer(value, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
    return value


def _sha256(value, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise ValueError(f"{name} 缺失或不是 SHA-256")
    return value


def _validate_identity(rec: dict) -> dict:
    identity = rec.get("run_identity")
    if (not isinstance(identity, dict) or type(identity.get("schema_version")) is not int
            or identity["schema_version"] != 2):
        raise ValueError("schema2 缺少有效的 run_identity")
    required = ("dataset", "schedule", "training_seed", "sample_seed", "sampler",
                "test_set_size", "checkpoint_sha256", "test_set_sha256",
                "sample_order_sha256", "source_sha256", "native_module_sha256",
                "sci_extension_sha256", "backend", "params", "wire",
                "mask_exact", "mask_domains")
    missing = [key for key in required if key not in identity]
    if missing:
        raise ValueError("run_identity 缺少字段: " + ", ".join(missing))
    for key in ("checkpoint_sha256", "test_set_sha256", "sample_order_sha256"):
        _sha256(identity[key], f"run_identity.{key}")
    for key in ("source_sha256", "native_module_sha256"):
        hashes = identity[key]
        if not isinstance(hashes, dict) or (key == "source_sha256" and not hashes):
            raise ValueError(f"run_identity.{key} 必须是有效哈希字典")
        for name, digest in hashes.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"run_identity.{key} 包含无效名称")
            _sha256(digest, f"run_identity.{key}.{name}")
    if identity["backend"] not in ("seal", "tenseal", "stub"):
        raise ValueError("run_identity.backend 无效")
    if identity["backend"] != "stub" and not identity["native_module_sha256"]:
        raise ValueError("真实后端缺少 native_module_sha256")
    if identity["sci_extension_sha256"] is not None:
        _sha256(identity["sci_extension_sha256"], "run_identity.sci_extension_sha256")
    for key in ("params", "wire"):
        if not isinstance(identity[key], dict) or not identity[key]:
            raise ValueError(f"run_identity.{key} 必须是非空字典")
    if type(identity["mask_exact"]) is not bool:
        raise ValueError("run_identity.mask_exact 必须是布尔值")
    if identity["mask_domains"] is not None and not isinstance(identity["mask_domains"], dict):
        raise ValueError("run_identity.mask_domains 必须是字典或 null")
    if identity["dataset"] not in ("mnist", "cifar10", "uci_har"):
        raise ValueError("run_identity.dataset 无效")
    if identity["schedule"] not in ("00", "01", "10", "11"):
        raise ValueError("run_identity.schedule 无效")
    if not isinstance(identity["sampler"], str) or not identity["sampler"]:
        raise ValueError("run_identity.sampler 无效")
    _integer(identity["training_seed"], "run_identity.training_seed")
    _integer(identity["sample_seed"], "run_identity.sample_seed")
    _integer(identity["test_set_size"], "run_identity.test_set_size", 1)
    for key, identity_key in (("dataset", "dataset"), ("schedule", "schedule"),
                              ("seed", "training_seed"), ("backend", "backend"),
                              ("checkpoint_sha256", "checkpoint_sha256"),
                              ("test_set_size", "test_set_size"), ("wire", "wire"),
                              ("mask_exact", "mask_exact"), ("mask_domains", "mask_domains")):
        if key in rec and rec[key] != identity[identity_key]:
            raise ValueError(f"{key} 与 run_identity 不一致")
    if "sampling" in rec:
        sampling = rec["sampling"]
        if not isinstance(sampling, dict) or any(
                sampling.get(key) != identity[key]
                for key in ("sample_seed", "sample_order_sha256")):
            raise ValueError("sampling 与 run_identity 不一致")
    return identity


def paired_rows(rec: dict) -> list[dict]:
    """校验 schema2 的全部已保存行，返回带重算 max_abs_err 的副本。

    不修改 rec，不访问 checkpoint/dataset，也不重新生成样本排列。
    哈希仅校验已记录字段；完整测试集行才可重算 sample_order_sha256。
    任意行无效则拒绝整份记录，partial 的合法前缀也可接受。
    """
    if not isinstance(rec, dict) or type(rec.get("schema_version")) is not int or rec["schema_version"] != 2:
        raise ValueError("paired_rows 仅接受 schema_version=2")
    identity = _validate_identity(rec)
    rows = rec.get("per_sample")
    if not isinstance(rows, list):
        raise ValueError("per_sample 必须是列表")
    if _integer(rec.get("n_samples"), "n_samples") != len(rows):
        raise ValueError("n_samples 与 per_sample 长度不一致")
    test_size = identity["test_set_size"]
    if len(rows) > test_size:
        raise ValueError("已保存行数超过 test_set_size")
    if "target_samples" in rec:
        target = _integer(rec["target_samples"], "target_samples")
        if not len(rows) <= target <= test_size:
            raise ValueError("target_samples 与已保存行数/test_set_size 不一致")
    result, sample_ids = [], set()
    class_count = None
    required = ("index", "sample_id", "label", "plain_logits", "cipher_logits",
                "plain_pred", "cipher_pred", "pred", "prepared_input_sha256")
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or any(key not in row for key in required):
            raise ValueError(f"第 {index} 行缺少配对证据")
        if _integer(row["index"], f"row[{index}].index") != index:
            raise ValueError(f"第 {index} 行样本序号不连续")
        sample_id = _integer(row["sample_id"], f"row[{index}].sample_id")
        if sample_id >= test_size or sample_id in sample_ids:
            raise ValueError(f"第 {index} 行 sample_id 越界或重复")
        sample_ids.add(sample_id)
        _sha256(row["prepared_input_sha256"], f"row[{index}].prepared_input_sha256")
        logits = []
        for key in ("plain_logits", "cipher_logits"):
            values = row[key]
            if not isinstance(values, list) or len(values) < 2:
                raise ValueError(f"第 {index} 行 {key} 必须是至少两类的列表")
            if any(type(value) not in (int, float) for value in values):
                raise ValueError(f"第 {index} 行 {key} 必须是数值列表")
            try:
                values = [float(value) for value in values]
            except OverflowError as exc:
                raise ValueError(f"第 {index} 行 {key} 超出有限浮点范围") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"第 {index} 行 {key} 含非有限值")
            logits.append(values)
        plain, cipher = logits
        if len(plain) != len(cipher) or (class_count is not None and len(plain) != class_count):
            raise ValueError(f"第 {index} 行 logits 类别数不一致")
        class_count = len(plain)
        label = _integer(row["label"], f"row[{index}].label")
        if label >= class_count:
            raise ValueError(f"第 {index} 行 label 超出类别范围")
        plain_pred = max(range(class_count), key=plain.__getitem__)
        cipher_pred = max(range(class_count), key=cipher.__getitem__)
        for key, expected in (("plain_pred", plain_pred), ("cipher_pred", cipher_pred),
                              ("pred", cipher_pred)):
            if _integer(row[key], f"row[{index}].{key}") != expected:
                raise ValueError(f"第 {index} 行 {key} 与 argmax-first/pred 别名不一致")
        error = max(abs(a - b) for a, b in zip(plain, cipher))
        if not math.isfinite(error):
            raise ValueError(f"第 {index} 行 logit 差值不是有限值")
        result.append({**row, "plain_logits": plain, "cipher_logits": cipher,
                       "max_abs_err": error})
    if len(rows) == test_size:
        digest = hashlib.sha256(json.dumps(
            [row["sample_id"] for row in rows], separators=(",", ":")).encode()).hexdigest()
        if digest != identity["sample_order_sha256"].lower():
            raise ValueError("完整样本序列与 sample_order_sha256 不一致")
    return result


def legacy_rows(rec: dict, root: Path, dataset: str, tag: str, seed: int,
                cache: dict) -> list[dict]:
    """仅由 --allow-legacy 分支调用；重算不能证明旧证据的来源。"""
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from canonical_net import choose_checkpoint, load_canonical_model
    from eval_encrypted import load_test_set

    explicit = rec.get("checkpoint")
    if "checkpoint" in rec:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError("legacy 已记录的 checkpoint 路径无效，不回退重选")
        checkpoint = Path(explicit)
        if not checkpoint.is_absolute():
            checkpoint = root / checkpoint
    else:
        checkpoint = choose_checkpoint(root, dataset, tuple(map(int, tag)), seed)
        print(f"警告: legacy 未记录 checkpoint，显式许可下选择 {checkpoint}；来源未验证",
              file=sys.stderr)
    if not checkpoint.is_file():
        raise ValueError(f"legacy checkpoint 不存在，不重建: {checkpoint}")
    model = load_canonical_model(root, dataset, tuple(map(int, tag)), seed,
                                 checkpoint=checkpoint)
    key = (dataset, seed)
    if key not in cache:
        test = load_test_set(root, dataset)
        cache[key] = [test[i] for i in np.random.default_rng(seed).permutation(len(test))]
    test = cache[key]
    rows = rec.get("per_sample")
    if not isinstance(rows, list):
        raise ValueError("legacy per_sample 必须是列表")
    if "n_samples" in rec and _integer(rec["n_samples"], "legacy n_samples") != len(rows):
        raise ValueError("legacy n_samples 与 per_sample 长度不一致")
    result, seen = [], set()
    for row in rows:
        idx = _integer(row["index"], "legacy index")
        if idx >= len(test) or idx in seen:
            raise ValueError(f"legacy index 越界或重复: {idx}")
        seen.add(idx)
        sample, label = test[idx]
        if _integer(row["label"], "legacy label") != int(label):
            raise ValueError(f"legacy 第 {idx} 行 label 与取出样本不一致")
        if dataset == "uci_har":
            prepared = np.asarray(sample, dtype=np.float64)
        else:
            raw = np.asarray(sample, dtype=np.float64).ravel()
            prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
        prepared = prepared.reshape(model.spec.model_shape)
        if prepared.ndim == 3 and prepared.shape[0] == 1:
            prepared = prepared[0]
        ref, _ = model.plain_forward_prepared(prepared.reshape(1, *model.spec.model_shape))
        ref = np.ravel(ref)[:model.spec.classes]
        if not np.isfinite(ref).all():
            raise ValueError(f"legacy 第 {idx} 行明文 logits 含非有限值")
        pred = _integer(row["pred"], "legacy pred")
        if pred >= len(ref):
            raise ValueError(f"legacy 第 {idx} 行 pred 超出类别范围")
        error = row.get("max_abs_err")
        if error is not None and (type(error) not in (int, float)
                                  or not math.isfinite(error) or error < 0):
            raise ValueError(f"legacy 第 {idx} 行 max_abs_err 无效")
        result.append({**row, "plain_pred": int(np.argmax(ref)), "cipher_pred": pred})
    return result


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _run_series(rec: dict, dataset: str, tag: str, run_dir: str, mode: str) -> str:
    series = {"dataset": dataset, "schedule": tag, "run_dir": run_dir,
              "backend": rec.get("backend"), "evidence_mode": mode}
    if mode == "saved_paired_v2":
        # Checkpoints and calibrated domains vary with training seeds. Keep
        # their full provenance in each record, not in the shared series key.
        series["identity"] = {key: value for key, value in rec["run_identity"].items()
                              if key not in ("training_seed", "checkpoint_sha256", "mask_domains")}
    digest = hashlib.sha256(json.dumps(series, sort_keys=True, separators=(",", ":"),
                                      allow_nan=False).encode()).hexdigest()[:16]
    return f"{dataset}/s{tag}/{run_dir}/{digest}"


def _agreement(rec: dict, rows: list[dict], path: Path, root: Path,
               dataset: str, tag: str, seed: int, mode: str) -> dict:
    n = len(rows)
    agree = sum(row["cipher_pred"] == row["plain_pred"] for row in rows)
    ct_acc = sum(row["cipher_pred"] == row["label"] for row in rows)
    pt_acc = sum(row["plain_pred"] == row["label"] for row in rows)
    lo, hi = _wilson(agree, n)
    errors = [row["max_abs_err"] for row in rows if row.get("max_abs_err") is not None]
    paired = mode == "saved_paired_v2"
    return {
        "agreement_schema_version": 2, "source_schema_version": rec.get("schema_version", 1),
        "dataset": dataset, "schedule": tag, "seed": seed, "backend": rec.get("backend"),
        "run_dir": path.parent.name,
        "run_series": _run_series(rec, dataset, tag, path.parent.name, mode),
        "source_json": path.relative_to(root).as_posix(), "evidence_mode": mode,
        "measured": rec.get("measured") is True and rec.get("backend") != "stub",
        "evidence_kind": rec.get("evidence_kind", "legacy_source_unverified"),
        "identity_validation": "recorded_fields_only_not_rehashed" if paired else "legacy_source_unverified",
        "source_verified": False, "run_identity": rec.get("run_identity"),
        "sample_order_hash_recomputed": paired and n == rec["run_identity"]["test_set_size"],
        "partial": bool(rec.get("partial", False)), "target_samples": rec.get("target_samples"),
        "setup": rec.get("setup"), "n_samples": n,
        "agreement": round(agree / n, 4), "agreement_ci95": [round(lo, 4), round(hi, 4)],
        "ciphertext_accuracy": round(ct_acc / n, 4),
        "plaintext_accuracy_same_samples": round(pt_acc / n, 4),
        "mean_max_abs_err": round(statistics.mean(errors), 8) if errors else None,
        "per_sample": rows,
    }


def _write_json(dest: Path, value: dict, out_dir: Path, raw: Path, overwrite: bool) -> None:
    resolved = dest.resolve()
    if not resolved.is_relative_to(out_dir) or resolved.is_relative_to(raw):
        raise ValueError(f"拒绝写到输出目录外或 raw 中: {dest}")
    if dest.is_symlink():
        raise ValueError(f"拒绝写入符号链接: {dest}")
    text = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w" if overwrite else "x", encoding="utf-8") as stream:
        stream.write(text)


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 已保存配对证据的一致率统计")
    ap.add_argument("--root", default=None, help="数据与原始结果根目录")
    ap.add_argument("--dataset", choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--out-dir", default=None, help="输出目录；相对路径以 root 为基准")
    ap.add_argument("--overwrite", action="store_true", help="仅配合显式 --out-dir 覆盖统计输出，绝不覆盖 raw")
    ap.add_argument("--allow-legacy", action="store_true", help="允许旧记录加载已有 checkpoint 和数据重算；来源未验证")
    ap.add_argument("--dry-run", action="store_true", help="仅列出文件/schema 分流，不加载模型、不写输出")
    args = ap.parse_args()
    if args.overwrite and args.out_dir is None:
        ap.error("--overwrite 必须同时显式指定 --out-dir")
    root = (Path(args.root) if args.root else Path(__file__).resolve().parent.parent).resolve()
    raw = (root / "results" / "raw_results").resolve()
    out_dir = Path(args.out_dir) if args.out_dir is not None else Path("results/agreement_v2")
    out_dir = (out_dir if out_dir.is_absolute() else root / out_dir).resolve()
    if out_dir.is_relative_to(raw) or raw.is_relative_to(out_dir):
        ap.error("输出目录不得位于 raw 内或包含 raw")
    files = sorted((raw / "E1").glob("*/hybrid_s*_acc*/seed*.json"))
    if args.dataset:
        files = [path for path in files if path.parts[-3] == args.dataset]
    cache, outputs = {}, []
    rejected = 0
    for path in files:
        try:
            rec = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(rec, dict):
                raise ValueError("原始记录必须是对象")
            schema = rec.get("schema_version", 1)
            if type(schema) is not int or schema not in (1, 2):
                raise ValueError(f"不支持 schema_version={schema!r}")
            if schema == 1 and not args.allow_legacy:
                print(f"跳过 legacy（需 --allow-legacy）: {path}")
                continue
            if args.dry_run:
                print(f"待处理 schema{schema}: {path}")
                continue
            dataset = path.parts[-3]
            tag = path.parent.name.split("_")[1][1:]
            seed = int(path.stem[4:])
            if dataset not in ("mnist", "cifar10", "uci_har") or tag not in ("00", "01", "10", "11"):
                raise ValueError("路径中 dataset/schedule 无效")
            for key, expected in (("dataset", dataset), ("schedule", tag), ("seed", seed)):
                if key in rec and rec[key] != expected:
                    raise ValueError(f"记录 {key} 与原始路径不一致")
            if schema == 2:
                rows = paired_rows(rec)
                identity = rec["run_identity"]
                if (identity["dataset"], identity["schedule"], identity["training_seed"]) != (dataset, tag, seed):
                    raise ValueError("run_identity 与原始路径不一致")
                mode = "saved_paired_v2"
            else:
                print(f"警告: legacy 来源未验证（即使存在 identity 也不按 v2 验证）: {path}", file=sys.stderr)
                rows = legacy_rows(rec, root, dataset, tag, seed, cache)
                mode = "legacy_recomputed_unverified"
            if not rows:
                print(f"跳过空记录，不加入 aggregate: {path}")
                continue
            out = _agreement(rec, rows, path, root, dataset, tag, seed, mode)
            json.dumps(out, allow_nan=False)
            dest = out_dir / "E1" / dataset / path.parent.name / f"{path.stem}_agreement.json"
            outputs.append((dest, out))
        except (ValueError, KeyError, IndexError, TypeError, OverflowError, OSError, ImportError) as exc:
            rejected += 1
            print(f"拒绝 {path}: {exc}", file=sys.stderr)
    if args.dry_run:
        return int(rejected > 0)
    summary_path = out_dir / "summary.json"
    for dest in [path for path, _ in outputs] + [summary_path]:
        if not dest.resolve().is_relative_to(out_dir) or dest.resolve().is_relative_to(raw) or dest.is_symlink():
            print(f"拒绝不安全的输出路径: {dest}", file=sys.stderr)
            return 1
        if dest.exists() and not args.overwrite:
            print(f"拒绝覆盖已有输出: {dest}；请选择新 --out-dir", file=sys.stderr)
            return 1
    summary = {}
    try:
        for dest, out in outputs:
            _write_json(dest, out, out_dir, raw, args.overwrite)
            summary.setdefault(out["run_series"], []).append(out)
            print(f"{out['dataset']} s{out['schedule']} seed{out['seed']} "
                  f"n={out['n_samples']} agree={out['agreement']} -> {dest}")
        aggregate = {}
        for key, runs in summary.items():
            agreements = [run["agreement"] for run in runs]
            aggregate[key] = {
                "n_seeds": len(runs), "n_samples_per_seed": [run["n_samples"] for run in runs],
                "partial_runs": sum(run["partial"] for run in runs),
                "evidence_mode": runs[0]["evidence_mode"],
                "measured": all(run["measured"] for run in runs),
                "evidence_kind": runs[0]["evidence_kind"],
                "agreement_mean": round(statistics.mean(agreements), 4),
                "agreement_std": round(statistics.stdev(agreements), 4) if len(runs) > 1 else 0.0,
                "ciphertext_accuracy_mean": round(statistics.mean(run["ciphertext_accuracy"] for run in runs), 4),
                "plaintext_accuracy_mean": round(statistics.mean(run["plaintext_accuracy_same_samples"] for run in runs), 4),
            }
        _write_json(summary_path, aggregate, out_dir, raw, args.overwrite)
    except (ValueError, OSError) as exc:
        print(f"写出失败: {exc}", file=sys.stderr)
        return 1
    print(f"本次有效记录={len(outputs)}，拒绝记录={rejected}；aggregate -> {summary_path}")
    return int(rejected > 0)


if __name__ == "__main__":
    raise SystemExit(main())
