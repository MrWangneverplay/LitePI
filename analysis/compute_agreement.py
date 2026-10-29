"""E1 Hybrid/Adaptive 密文结果的离线明文一致率补算（不重跑任何密文查询）。

背景
----
E1 的密文精度只在 64 个样本上测过，标准误约 6 个百分点，不足以用来给
调度策略排序（见复审 N1）。论文/汇报里正确的口径是：

  * 密文侧只报「与同一 checkpoint 明文预测的一致率」和 logit 误差；
  * 精度与调度排序用全测试集明文结果。

本脚本读取 results/raw_results/E1/*/hybrid_s*_acc*/seed*.json 里已经落盘的
per_sample（含 index/label/pred），按 eval_hybrid_accuracy.py 完全相同的
预处理与样本顺序重建数据，用明文前向补出 plain_pred，从而离线得到一致率。
不启动 server、不加密、不占用远程机器。

输出
----
results/agreement/E1/<dataset>/<run_dir>/seed<seed>_agreement.json
results/agreement/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import math
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from canonical_net import load_canonical_model
from eval_encrypted import load_test_set


def _prepare(model, dataset: str, sample) -> np.ndarray:
    """与 eval_hybrid_accuracy._prepare 完全一致。"""
    if dataset == "uci_har":
        prepared = np.asarray(sample, dtype=np.float64)
    else:
        raw = np.asarray(sample, dtype=np.float64).ravel()
        prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
    prepared = prepared.reshape(model.spec.model_shape)
    if prepared.ndim == 3 and prepared.shape[0] == 1:
        prepared = prepared[0]
    return prepared


def _reference(model, prepared: np.ndarray) -> np.ndarray:
    ref, _ = model.plain_forward_prepared(
        prepared.reshape(1, *model.spec.model_shape)
    )
    return np.ravel(ref)[: model.spec.classes]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _order(test, seed: int):
    rng = np.random.default_rng(seed)
    return [test[i] for i in rng.permutation(len(test))]


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 密文一致率离线补算")
    ap.add_argument("--root", default=None)
    ap.add_argument("--dataset", default=None,
                    choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--dry-run", action="store_true", help="只列出待处理文件")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    e1 = root / "results" / "raw_results" / "E1"
    files = sorted(e1.glob("*/hybrid_s*_acc*/seed*.json"))
    if args.dataset:
        files = [p for p in files if p.parts[-3] == args.dataset]

    cache: dict[tuple[str, int], list] = {}
    summary: dict[str, dict] = {}

    for path in files:
        rec = json.loads(path.read_text(encoding="utf-8"))
        rows = rec.get("per_sample") or []
        if not rows:
            continue
        dataset = rec.get("dataset") or path.parts[-3]
        seed = int(rec.get("seed", path.stem.replace("seed", "")))
        tag = str(rec.get("schedule", path.parent.name.split("_")[1][1:]))

        if args.dry_run:
            print(f"would process {path} ({len(rows)} samples)")
            continue

        key = (dataset, seed)
        if key not in cache:
            cache[key] = _order(load_test_set(root, dataset), seed)
        test = cache[key]

        model = load_canonical_model(root, dataset, tuple(int(b) for b in tag), seed)

        plain_preds, kept = [], []
        for row in rows:
            idx = int(row["index"])
            if idx >= len(test):
                continue
            sample, _label = test[idx]
            plain_preds.append(int(np.argmax(_reference(model, _prepare(model, dataset, sample)))))
            kept.append(row)

        n = len(kept)
        agree = sum(1 for r, p in zip(kept, plain_preds) if int(r["pred"]) == p)
        ct_acc = sum(1 for r in kept if int(r["pred"]) == int(r["label"]))
        pt_acc = sum(1 for r, p in zip(kept, plain_preds) if p == int(r["label"]))
        lo, hi = _wilson(agree, n)

        out = {
            "dataset": dataset,
            "schedule": tag,
            "seed": seed,
            "backend": rec.get("backend"),
            "run_dir": path.parent.name,
            "source_json": str(path.relative_to(root)).replace("\\", "/"),
            "n_samples": n,
            "agreement": round(agree / n, 4) if n else None,
            "agreement_ci95": [round(lo, 4), round(hi, 4)],
            "ciphertext_accuracy": round(ct_acc / n, 4) if n else None,
            "plaintext_accuracy_same_samples": round(pt_acc / n, 4) if n else None,
            "mean_max_abs_err": round(
                float(np.mean([r["max_abs_err"] for r in kept])), 8) if kept else None,
            "per_sample": [
                {"index": int(r["index"]), "label": int(r["label"]),
                 "pred": int(r["pred"]), "plain_pred": int(p)}
                for r, p in zip(kept, plain_preds)
            ],
        }
        dest = (root / "results" / "agreement" / "E1" / dataset / path.parent.name
                / f"{path.stem}_agreement.json")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
        print(f"{dataset:8s} s{tag} seed{seed} n={n:3d} "
              f"agree={out['agreement']} ct_acc={out['ciphertext_accuracy']} "
              f"pt_acc={out['plaintext_accuracy_same_samples']} -> {dest.name}")

        skey = f"{dataset}/s{tag}/{path.parent.name}"
        slot = summary.setdefault(skey, {"agreement": [], "ciphertext_accuracy": [],
                                         "plaintext_accuracy_same_samples": []})
        slot["agreement"].append(out["agreement"])
        slot["ciphertext_accuracy"].append(out["ciphertext_accuracy"])
        slot["plaintext_accuracy_same_samples"].append(out["plaintext_accuracy_same_samples"])

    if args.dry_run:
        return 0

    agg = {}
    for k, v in summary.items():
        agg[k] = {
            "n_seeds": len(v["agreement"]),
            "agreement_mean": round(float(np.mean(v["agreement"])), 4),
            "agreement_std": round(float(np.std(v["agreement"], ddof=1)), 4) if len(v["agreement"]) > 1 else 0.0,
            "ciphertext_accuracy_mean": round(float(np.mean(v["ciphertext_accuracy"])), 4),
            "plaintext_accuracy_mean": round(float(np.mean(v["plaintext_accuracy_same_samples"])), 4),
        }
    dest = root / "results" / "agreement" / "summary.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(agg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nsummary -> {dest} ({len(agg)} runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
