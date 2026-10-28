"""E7 截断消融：随机截断 vs 直接截断（Beaver 定点乘法后除 2^f）。

协议 §4.2 / 实验方案 Phase 4：份额域定点乘法（Beaver）产生 2f-bit 定点结果，
须截断回 f-bit。两种截断方案：

  direct   直接向下取整  floor(v / 2^f)，系统性偏下，E[err] = -0.5 LSB；
  random   概率舍入，以 (v mod 2^f) / 2^f 概率上取整，无偏，E[err] = 0。

本实验量化两者在 (a) 单次截断、(c) n 元点积累加 中的精度差，支撑协议选用
随机截断的误差分析。全部为纯 Python/numpy 本地 CPU 数值实验，不占远程机。

误差统一以 LSB = 2^-f 为单位（f-bit 定点的最小分辨率）：

    err = truncate(v_2f) - v_2f / 2^f

其中 v_2f = to_fixed(x) * to_fixed(w) 是 2f-bit 精确定点乘积；该乘积等价于
Beaver 乘法重构后的份额和（hybrid_mpc.self_test 已证 xy_c + xy_s == x_i*y_i）。

输出 results/E7/e7_truncation.{json,csv} 与 e7_summary.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

from hybrid_mpc import (
    FIXED_FRAC,
    direct_truncate,
    random_truncate,
    to_fixed,
)

F = FIXED_FRAC
LSB = float(2 ** F)  # 1 LSB 对应的分母，用于把定点误差换算为 LSB 单位


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, timeout=5
        )
        return out.stdout.decode().strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def err_lsb_direct(v_2f: np.ndarray) -> np.ndarray:
    """直接截断的误差（LSB 单位，逐元素）。"""
    t = direct_truncate(v_2f, F).astype(np.float64)
    return t - v_2f.astype(np.float64) / LSB


def err_lsb_random(v_2f: np.ndarray, rng) -> np.ndarray:
    """随机截断的误差（LSB 单位，逐元素）。"""
    t = random_truncate(v_2f, F, rng).astype(np.float64)
    return t - v_2f.astype(np.float64) / LSB


def stats(err: np.ndarray) -> dict:
    """bias / var / rms / max_abs（LSB 单位）。"""
    return {
        "bias_lsb": round(float(err.mean()), 6),
        "var_lsb": round(float(err.var()), 6),
        "rms_lsb": round(float(np.sqrt(np.mean(err ** 2))), 6),
        "max_abs_lsb": round(float(np.abs(err).max()), 6),
    }


def e7a(n_samples: int, rng) -> dict:
    """单次截断统计：x, w ~ U(-8, 8)，对 2f-bit 乘积做一次截断。"""
    x = rng.uniform(-8.0, 8.0, n_samples)
    w = rng.uniform(-8.0, 8.0, n_samples)
    v_2f = to_fixed(x, F) * to_fixed(w, F)
    return {
        "direct": stats(err_lsb_direct(v_2f)),
        "random": stats(err_lsb_random(v_2f, rng)),
    }


def e7c(n: int, n_reps: int, rng) -> dict:
    """n 元点积累加：每项乘积截断回 f-bit 后累加，与浮点参考对比。

    返回 direct/random 各 repr 次重复的 (y_quant - y_ref) 的 bias/std/rms，
    单位 LSB。bias 反映截断的系统性偏置是否随运量（n）累积。
    """
    refs = rng.uniform(-8.0, 8.0, (n_reps, n))   # x
    ws = rng.uniform(-8.0, 8.0, (n_reps, n))     # w
    v_2f = to_fixed(refs, F) * to_fixed(ws, F)

    y_ref = (refs * ws).sum(axis=1)  # 浮点参考（高精度）

    y_direct = direct_truncate(v_2f, F).astype(np.float64).sum(axis=1) / LSB
    y_random = random_truncate(v_2f, F, rng).astype(np.float64).sum(axis=1) / LSB

    # 差值为真值单位，乘回 2^f 即为 LSB 单位。
    diff_direct = (y_direct - y_ref) * LSB
    diff_random = (y_random - y_ref) * LSB
    return {
        "direct": stats(diff_direct),
        "random": stats(diff_random),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E7 truncation ablation data generator")
    ap.add_argument("--n-samples", type=int, default=200_000,
                    help="E7a 单次截断样本数")
    ap.add_argument("--n-reps", type=int, default=4_000,
                    help="E7c 每个点积长度的蒙特卡洛重复次数")
    ap.add_argument("--seeds", type=int, nargs="+", default=[73, 74, 75, 76, 77])
    ap.add_argument("--dot-lengths", type=int, nargs="+",
                    default=[1, 4, 16, 64, 256, 1024, 4096])
    ap.add_argument("--out", default="results/E7")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment": "E7",
        "title": "随机截断 vs 直接截断（Beaver 定点乘法后除 2^f）",
        "f_fractional_bits": F,
        "lsb": f"2^-{F}",
        "value_domain": [-8.0, 8.0],
        "seeds": args.seeds,
        "n_samples_e7a": args.n_samples,
        "n_reps_e7c": args.n_reps,
        "dot_lengths_e7c": args.dot_lengths,
        "git_sha": _git_sha(),
        "e7a": {},   # seed -> {direct, random}
        "e7c": {},   # n -> seed -> {direct, random}
    }

    # E7a：每个种子独立采样，记录 per-seed 统计
    for sd in args.seeds:
        rng = np.random.default_rng(sd)
        record["e7a"][str(sd)] = e7a(args.n_samples, rng)

    # E7c：点积累加，逐 n 逐种子
    for n in args.dot_lengths:
        record["e7c"][str(n)] = {}
        for sd in args.seeds:
            rng = np.random.default_rng(sd)
            record["e7c"][str(n)][str(sd)] = e7c(n, args.n_reps, rng)

    (outdir / "e7_truncation.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # 汇总：跨 5 种子 mean±std，写 CSV
    keys = ["bias_lsb", "var_lsb", "rms_lsb", "max_abs_lsb"]
    rows = []
    # E7a 汇总行
    for mode in ("direct", "random"):
        r = {"context": "e7a_single", "n": 1, "mode": mode}
        for k in keys:
            vals = [record["e7a"][s][mode][k] for s in record["e7a"]]
            r[f"{k}_mean"] = round(float(np.mean(vals)), 6)
            r[f"{k}_std"] = round(float(np.std(vals)), 6)
        rows.append(r)

    # E7c 汇总行
    e7c_agg = {}
    for n_str, per_seed in record["e7c"].items():
        for mode in ("direct", "random"):
            r = {"context": "e7c_dot", "n": int(n_str), "mode": mode}
            for k in keys:
                vals = [per_seed[s][mode][k] for s in per_seed]
                r[f"{k}_mean"] = round(float(np.mean(vals)), 6)
                r[f"{k}_std"] = round(float(np.std(vals)), 6)
            rows.append(r)
            e7c_agg[f"{n_str}_{mode}_bias"] = r["bias_lsb_mean"]

    csv_path = outdir / "e7_truncation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["context", "n", "mode"] + [
            f"{k}_{suf}" for k in keys for suf in ("mean", "std")
        ]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # summary：聚焦"bias 随 n 增长"这个核心结论
    summary = {
        "f_fractional_bits": F,
        "e7a_direct_bias_lsb": rows[0]["bias_lsb_mean"],
        "e7a_random_bias_lsb": rows[1]["bias_lsb_mean"],
        "e7c_direct_bias_by_n": {
            n: e7c_agg[f"{n}_direct_bias"] for n in
            [str(x) for x in args.dot_lengths]
        },
        "e7c_random_bias_by_n": {
            n: e7c_agg[f"{n}_random_bias"] for n in
            [str(x) for x in args.dot_lengths]
        },
        "git_sha": record["git_sha"],
    }
    (outdir / "e7_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # 控制台摘要
    print(f"E7 · f={F} (LSB=2^-{F}), value domain [-8,8]")
    print(f"E7a single-trunc  bias(LSB):  direct={rows[0]['bias_lsb_mean']:.4f}  "
          f"random={rows[1]['bias_lsb_mean']:.4f}")
    print("E7c dot-product bias(LSB) vs n:")
    for n in args.dot_lengths:
        d = e7c_agg[f"{n}_direct_bias"]
        r = e7c_agg[f"{n}_random_bias"]
        print(f"  n={n:6d}  direct={d:+8.4f}  random={r:+8.4f}")
    print(f"\nwritten -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())