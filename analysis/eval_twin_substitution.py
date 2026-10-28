"""E1「多项式近似损失」：在 **ReLU 孪生网** 权重上做 P₂ 替换（明文，快速）。

背景（实验方案 §3.2.4 / §8）：
  * 现有 `hybrid_s00..s11` 是**按调度分别训练**的四套权重，s00 的 P₂ 是由训练适配过的，
    因此直接用 s00 报 HE-only 会低估「多项式近似损失」；
  * 论文要的是**同一套 ReLU 孪生网**（canonical 结构、两层精确 ReLU，即 `hybrid_s11`）
    在把某层换成 P₂ 之后的掉点，这才是 Adaptive 调度的收益来源。

做法：固定 `datas/<ds>_hybrid_s11_seed<seed>_cnn_train.pt` 的权重，
      用 4 种激活组合 (a1,a2) ∈ {relu,p2}² 各跑一遍明文前向，统计测试集精度。

输出：`results/raw_results/E1/<ds>/twin_substitution/seed<seed>.json` + `summary.json`

用法：
    python analysis/eval_twin_substitution.py                       # 3 数据集 × seed73
    python analysis/eval_twin_substitution.py --seeds 73 74 75 76 77 --samples 2000
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from eval_encrypted import load_test_set

ROOT = Path(__file__).resolve().parent.parent
PATTERNS = ("11", "10", "01", "00")
LABELS = {"11": "ReLU/ReLU（孪生基线）", "10": "ReLU/P2", "01": "P2/ReLU",
          "00": "P2/P2（全替换）"}


def prepare_batch(model, samples: list, dataset: str) -> np.ndarray:
    if dataset == "uci_har":
        # load_data 已 z-score；再走 prepare_array 会归一化两次
        rows = [np.asarray(s, dtype=np.float64).reshape(model.spec.model_shape)
                for s in samples]
    else:
        rows = [np.asarray(model.prepare_array(np.asarray(s, dtype=np.float64)),
                           dtype=np.float64) for s in samples]
    return np.asarray(rows, dtype=np.float64)


def accuracy(model, prepared: np.ndarray, labels: np.ndarray, batch: int = 512) -> float:
    classes = model.spec.classes
    hits = 0
    for start in range(0, len(prepared), batch):
        chunk = prepared[start:start + batch]
        logits, _ = model.plain_forward_prepared(chunk)
        pred = np.asarray(logits).reshape(len(chunk), classes).argmax(axis=1)
        hits += int((pred == labels[start:start + batch]).sum())
    return 100.0 * hits / max(len(prepared), 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="ReLU 孪生网的 P2 替换精度")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--datasets", nargs="+", default=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[73])
    ap.add_argument("--samples", type=int, default=0, help="0 = 用完整测试集")
    args = ap.parse_args()
    root = Path(args.root)
    datasets = [d for d in ",".join(args.datasets).replace(" ", ",").split(",") if d]

    print(f"root={root}  datasets={datasets}  seeds={args.seeds}  "
          f"samples={args.samples or 'all'}")
    summary = {"experiment": "E1", "method": "twin_substitution",
               "note": "固定 hybrid_s11（两层 ReLU 孪生）权重，逐层替换为 P2 后的明文精度",
               "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "datasets": {}}

    for ds in datasets:
        test = load_test_set(root, ds)
        n = min(args.samples, len(test)) if args.samples else len(test)
        items = [test[i] for i in range(n)]  # TensorDataset 切片返回 (X,Y)，逐条取用
        samples = [it[0] for it in items]
        labels = np.asarray([int(np.asarray(it[1]).reshape(-1)[0]) for it in items])

        acc = {p: [] for p in PATTERNS}
        print(f"\n### {ds}  n={n}  test={len(test)}")
        print(f"{'pattern':>8}  {'accuracy%':>10}  说明")
        for seed in args.seeds:
            twin = (root / "datas" / f"{ds}_hybrid_s11_seed{seed}_cnn_train.pt")
            if not twin.exists():
                raise SystemExit(f"缺少孪生权重 {twin}")
            prepared = None
            seed_acc = {}
            for pat in PATTERNS:
                model = load_canonical_model(root, ds, (int(pat[0]), int(pat[1])),
                                             seed, twin)
                if prepared is None:
                    prepared = prepare_batch(model, samples, ds)
                seed_acc[pat] = round(accuracy(model, prepared, labels), 4)
                acc[pat].append(seed_acc[pat])
            out = root / "results" / "raw_results" / "E1" / ds / "twin_substitution"
            out.mkdir(parents=True, exist_ok=True)
            rec = {"exp_id": "E1", "dataset": ds, "method": "twin_substitution",
                   "seed": seed, "n_samples": n,
                   "checkpoint": str(twin.relative_to(root)),
                   "accuracy_pct": seed_acc}
            (out / f"seed{seed}.json").write_text(
                json.dumps(rec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  seed={seed}  " + "  ".join(f"{p}={seed_acc[p]:.2f}" for p in PATTERNS))

        rows = {}
        for pat in PATTERNS:
            m = st.fmean(acc[pat])
            s = st.stdev(acc[pat]) if len(acc[pat]) > 1 else 0.0
            rows[pat] = {"mean": round(m, 4), "std": round(s, 4),
                         "vs_baseline": round(m - st.fmean(acc["11"]), 4)}
            print(f"{pat:>8}  {m:>6.2f} ± {s:.2f}   {LABELS[pat]}   "
                  f"Δ={rows[pat]['vs_baseline']:+.2f}")
        summary["datasets"][ds] = {"n_samples": n, "seeds": args.seeds, "rows": rows}
        (root / "results" / "raw_results" / "E1" / ds / "twin_substitution"
         / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
