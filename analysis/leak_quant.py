"""LitePI E10：泄漏量化实验（对应实验方案 E10a / E10b / E10c）。

为什么不需要 MPC 就能做：协议 4.1 第 2 步中，客户端解密 ct_(z+r) 得到的明文 s = z + r
本身就是协议交互的可观测量。本脚本直接**从训练好的模型里取真实激活 z**，按规则 M2
（掩码域覆盖激活范围）采样掩码 r，从而精确复现"攻击者能看到什么"，进而量化泄漏。

三个子实验：
  E10a 掩码复用攻击：比较"每次查询新鲜掩码（遵守 M1）"与"跨查询复用掩码（违反 M1）"下，
       攻击者用两次查询的观测差恢复激活差分的精度（MSE / 相对误差 / 相关系数 / 信噪比）。
  E10b 成员推理：用成员（训练集）与非成员（测试集）样本，训练攻击分类器，
       比较两种输出档位的信息量 —— ① 解密全部 logits（默认）② 只解密类别索引（输出最小化）。
  E10c 调度混淆开销：把"HE-only 层也注入等量假交互"以隐藏调度向量的代价（轮数/字节/延迟），
       按参数化模型给出估计（真实字节数以协议设计 §6 的字节预算为准，标注为估算）。

用法：
    python analysis/leak_quant.py                    # 三数据集，Q=200 次查询
    python analysis/leak_quant.py --queries 500
    python analysis/leak_quant.py --datasets mnist --smoke
    python analysis/leak_quant.py --figure           # 额外输出 datas/E10_summary.png
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

SPEC = {
    "mnist": dict(module="mnist_cnn_enc.py", ckpt="mnist_relu_cnn_train.pt", classes=10),
    "cifar10": dict(module="cifar10_cnn_enc.py", ckpt="cifar10_relu_cnn_train.pt", classes=10),
    "uci_har": dict(module="uci_har_cnn_enc.py", ckpt="uci_har_relu_cnn_train.pt", classes=6),
}


# ------------------------------------------------------------------ 基础工具
def guess_root() -> str:
    here = Path(__file__).resolve()
    for cand in (here.parent.parent, here.parent, Path(DEFAULT_ROOT)):
        if (cand / "model").is_dir() and (cand / "datas").is_dir():
            return str(cand)
    return DEFAULT_ROOT


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def git_sha(root: Path):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=False, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def save_record(root: Path, exp_id: str, group: str, method: str, seed: int, payload: dict) -> Path:
    out_dir = root / "results" / "raw_results" / exp_id / group / method
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"exp_id": exp_id, "dataset": group, "method": method, "seed": seed,
              "git_sha": git_sha(root),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record.update(payload)
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U 形式的 AUC；labels: 1=阳性（成员）。"""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit_logreg(X: np.ndarray, y: np.ndarray, epochs: int = 1500, lr: float = 0.5) -> np.ndarray:
    """极简逻辑回归（梯度下降），避免引入 sklearn 依赖。"""
    X = np.hstack([X, np.ones((len(X), 1))])
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        z = X @ w
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        grad = X.T @ (p - y) / len(y)
        w -= lr * grad
    return w


def predict_logreg(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.hstack([X, np.ones((len(X), 1))])
    return X @ w


# ------------------------------------------------------------------ 取激活与输出
@torch.no_grad()
def collect(model, loader, prepare, n_queries: int, device):
    """返回 (Z1, Z2, logits, labels)：两层非线性层的输入激活、网络输出、真标签。"""
    captured: dict[str, torch.Tensor] = {}
    orig = (model.a1, model.a2)

    def wrap(name, fn):
        def inner(x):
            captured[name] = x.detach().clone()
            return fn(x)
        return inner

    model.a1 = wrap("z1", orig[0])
    model.a2 = wrap("z2", orig[1])
    model.eval()

    Z1, Z2, LOGITS, Y = [], [], [], []
    seen = 0
    for images, targets in loader:
        x = prepare(images).to(device)
        out = model(x)
        Z1.append(captured["z1"].flatten(1).cpu().numpy())
        Z2.append(captured["z2"].flatten(1).cpu().numpy())
        LOGITS.append(out.cpu().numpy())
        Y.append(targets.numpy())
        seen += x.shape[0]
        if seen >= n_queries:
            break
    model.a1, model.a2 = orig
    return (np.vstack(Z1)[:n_queries], np.vstack(Z2)[:n_queries],
            np.vstack(LOGITS)[:n_queries], np.concatenate(Y)[:n_queries])


# ------------------------------------------------------------------ E10a
def e10a(Z: np.ndarray, rng: np.random.Generator, reuse: bool) -> dict:
    """掩码复用 vs 新鲜的恢复误差。Z: (Q, n) 真实激活。"""
    Q, n = Z.shape
    R = float(np.max(np.abs(Z))) * 1.05      # M2：掩码域覆盖激活分布域
    if reuse:
        r = rng.uniform(-R, R, size=(1, n))
        S = Z + r
    else:
        S = Z + rng.uniform(-R, R, size=(Q, n))
    D_true = Z[:-1] - Z[1:]
    D_hat = S[:-1] - S[1:]
    err = D_hat - D_true
    mse = float(np.mean(err ** 2))
    energy = float(np.mean(D_true ** 2))
    corr = float(np.corrcoef(D_hat.ravel(), D_true.ravel())[0, 1]) if D_true.size > 1 else float("nan")
    return {
        "Q": Q, "dim": int(n), "mask_domain_R": R,
        "mse": mse,
        "true_difference_energy": energy,
        "relative_error": (mse / energy) if energy > 0 else float("nan"),
        "recovery_correlation": corr,
        "recovery_snr_db": (10 * math.log10(energy / mse)) if mse > 0 else float("inf"),
    }


# ------------------------------------------------------------------ E10b
def e10b(logits_member, logits_non, rng) -> dict:
    """成员推理 AUC：全 logits 档 vs 仅类别索引档。

    攻击模型为置信度/分数型（把攻击看作二分类的学习问题），不是完整的 shadow-model 攻击；
    论文中应以"confidence-based membership inference"称呼，避免夸大口径。
    """
    def onehot(L):
        idx = L.argmax(axis=1)
        H = np.zeros_like(L)
        H[np.arange(len(L)), idx] = 1.0
        return H

    out = {}
    for name, (fm, fn) in (
        ("logits", (logits_member, logits_non)),
        ("index_only", (onehot(logits_member), onehot(logits_non))),
    ):
        X = np.vstack([fm, fn])
        y = np.concatenate([np.ones(len(fm)), np.zeros(len(fn))])
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
        Xs = (X - mu) / sd
        perm = rng.permutation(len(y))
        Xs, y = Xs[perm], y[perm]
        half = len(y) // 2
        w = fit_logreg(Xs[:half], y[:half])
        scores = predict_logreg(w, Xs[half:])
        out[name] = {"auc": auc(scores, y[half:]), "n_member": int(len(fm)),
                     "n_non_member": int(len(fn)), "dim": int(X.shape[1])}
    return out


# ------------------------------------------------------------------ E10c
def e10c(k_nonlinear: int = 2, per_layer_bytes: int = 128 * 1024,
         rtt_ms: float = 50.0, downlink_bytes: int = 64) -> dict:
    """调度混淆（dummy-round padding）的额外代价估算。

    参数含义：每个 HE-only 层注入 1 次与 hybrid 层等长的假交互；
    per_layer_bytes 为一次交互的上行密文字节数（默认按 CKKS poly 8192 的 fresh 密文量级估）。
    """
    rows = []
    for k_h in range(k_nonlinear + 1):
        he_only = k_nonlinear - k_h
        rows.append({
            "K_h": k_h,
            "he_only_layers": he_only,
            "extra_rounds": he_only,
            "extra_uplink_bytes": he_only * per_layer_bytes,
            "extra_downlink_bytes": he_only * downlink_bytes,
            "extra_latency_ms": he_only * rtt_ms,
        })
    return {"k_nonlinear": k_nonlinear, "per_layer_bytes": per_layer_bytes,
            "rtt_ms": rtt_ms, "rows": rows,
            "note": "字节数按参数量级估算；真实值以协议设计 §6 的字节预算与实测密文大小为准"}


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="E10 泄漏量化（掩码复用 / 成员推理 / 混淆开销）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--queries", type=int, default=200, help="每个数据集的查询次数 Q")
    ap.add_argument("--mia-per-class", type=int, default=300, help="成员/非成员各取多少样本")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--figure", action="store_true", help="额外输出 datas/E10_summary.png")
    ap.add_argument("--smoke", action="store_true", help="小规模快速验证（不推荐用于论文）")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(guess_root())
    model_dir, data_dir = root / "model", root / "datas"
    if not model_dir.is_dir() or not data_dir.is_dir():
        sys.exit(f"项目根目录不对：{root}")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.smoke:
        args.queries, args.mia_per_class = 30, 40
    device = torch.device("cpu")
    rng = np.random.default_rng(args.seed)

    print(f"项目根目录: {root}")
    print(f"Q={args.queries}  MIA 每类样本={args.mia_per_class}  数据集={datasets}")
    print("=" * 80)

    report: dict[str, dict] = {}
    for ds in datasets:
        spec = SPEC[ds]
        module = load_module(model_dir / spec["module"])
        train_loader, test_loader = module.load_data(args.batch_size)[0:2]
        prepare = (lambda x: module.preprocess(x)) if ds == "cifar10" else (
            (lambda x: x.unsqueeze(1)) if ds == "uci_har" else (lambda x: x))

        # 网络结构复用 eval_poly_substitution.py 的定义，保证与之前实验一致
        sib = load_module(Path(__file__).resolve().parent / "eval_poly_substitution.py")
        cls = {"mnist": sib.MnistReluTwin, "cifar10": sib.CifarReluTwin, "uci_har": sib.HarReluTwin}[ds]
        obj = torch.load(str(data_dir / spec["ckpt"]), map_location="cpu")
        kwargs = {k: int(obj[k]) for k in ("channels", "hidden") if isinstance(obj, dict) and k in obj}
        model = cls(**kwargs)
        model.load_state_dict(obj.get("state_dict", obj) if isinstance(obj, dict) else obj, strict=True)
        for a in ("a1", "a2"):
            setattr(model, a, torch.relu)

        print(f"\n### {ds}")
        Z1, Z2, LOGITS, Y = collect(model, test_loader, prepare, args.queries, device)
        print(f"  取到激活: z1 {Z1.shape}, z2 {Z2.shape}, logits {LOGITS.shape}")

        # ---- E10a ----
        for layer_name, Z in (("layer1", Z1), ("layer2", Z2)):
            fresh = e10a(Z, np.random.default_rng(args.seed), reuse=False)
            reused = e10a(Z, np.random.default_rng(args.seed), reuse=True)
            print(f"  [E10a] {layer_name}: 新鲜掩码 相对误差={fresh['relative_error']:.3f} "
                  f"相关系数={fresh['recovery_correlation']:+.3f} | "
                  f"复用掩码 相对误差={reused['relative_error']:.3e} 相关系数={reused['recovery_correlation']:+.3f}")
            if not args.smoke:
                save_record(root, "E10", ds, f"mask_fresh_{layer_name}", args.seed,
                            {"leakage": fresh, "layer": layer_name, "mask_policy": "fresh"})
                save_record(root, "E10", ds, f"mask_reuse_{layer_name}", args.seed,
                            {"leakage": reused, "layer": layer_name, "mask_policy": "reused"})

        # ---- E10b ----
        n_pc = args.mia_per_class
        cm = collect(model, train_loader, prepare, n_pc, device)          # 成员：训练集
        cn = collect(model, test_loader, prepare, n_pc, device)           # 非成员：测试集
        lm, ym = cm[2], cm[3]
        ln, yn = cn[2], cn[3]
        acc_m = float((lm.argmax(axis=1) == ym).mean())
        acc_n = float((ln.argmax(axis=1) == yn).mean())
        mia = e10b(lm, ln, np.random.default_rng(args.seed))
        print(f"  [E10b] 成员子集训练精度={acc_m:.3f}  非成员子集测试精度={acc_n:.3f}  "
              f"（过拟合差距 {(acc_m - acc_n) * 100:+.1f} 个点）")
        print(f"         成员推理 AUC: 全 logits={mia['logits']['auc']:.3f}   "
              f"仅索引={mia['index_only']['auc']:.3f}")
        if not args.smoke:
            save_record(root, "E10", ds, "mia_logits", args.seed,
                        {"membership_inference": mia["logits"], "output_mode": "full_logits",
                         "member_train_acc": acc_m, "non_member_test_acc": acc_n})
            save_record(root, "E10", ds, "mia_index_only", args.seed,
                        {"membership_inference": mia["index_only"], "output_mode": "argmax_index",
                         "member_train_acc": acc_m, "non_member_test_acc": acc_n})

        report[ds] = {"e10a_layer1_fresh": e10a(Z1, np.random.default_rng(args.seed), False)["relative_error"],
                      "e10a_layer1_reuse": e10a(Z1, np.random.default_rng(args.seed), True)["relative_error"],
                      "e10a_layer2_fresh": e10a(Z2, np.random.default_rng(args.seed), False)["relative_error"],
                      "e10a_layer2_reuse": e10a(Z2, np.random.default_rng(args.seed), True)["relative_error"],
                      "mia_logits_auc": mia["logits"]["auc"],
                      "mia_index_auc": mia["index_only"]["auc"]}

    # ---- E10c ----
    # 每次假交互的字节取 E4 实测（上行 fresh L7 + 下行 L4），而不是默认的 128 KB 估值。
    per_layer_bytes = 128 * 1024
    e4 = root / "results" / "raw_results" / "E4" / "all" / "comm_bytes" / "seed73.json"
    if e4.exists():
        ladder = {row["level"]: row["measured_bytes"]
                  for row in json.loads(e4.read_text(encoding="utf-8"))["ciphertext_ladder"]}
        per_layer_bytes = int(ladder["L7"] + ladder["L4"])
    overhead = e10c(per_layer_bytes=per_layer_bytes)
    print("\n[E10c] 调度混淆（dummy-round padding）的额外代价估算"
          f"（per-layer={per_layer_bytes/1024:.0f} KB，取自 E4 实测）")
    print(f"  {'K_h':>4}{'HE-only 层':>11}{'额外轮数':>10}{'额外上行':>12}{'额外延迟':>10}")
    for row in overhead["rows"]:
        print(f"  {row['K_h']:>4}{row['he_only_layers']:>11}{row['extra_rounds']:>10}"
              f"{row['extra_uplink_bytes']/1024:>10.0f} KB{row['extra_latency_ms']:>8.0f} ms")
    if not args.smoke:
        save_record(root, "E10", "all", "scheduling_obfuscation", args.seed, overhead)

    # ---- 汇总 ----
    print("\n" + "=" * 80)
    print("汇总")
    print(f"  {'数据集':<10}{'E10a 复用(相对误差)':>20}{'E10a 新鲜(相对误差)':>20}"
          f"{'MIA 全logits':>14}{'MIA 仅索引':>12}")
    for ds, r in report.items():
        print(f"  {ds:<10}{r['e10a_layer1_reuse']:>20.2e}{r['e10a_layer1_fresh']:>20.3f}"
              f"{r['mia_logits_auc']:>14.3f}{r['mia_index_auc']:>12.3f}")
    print("\n  期望：复用掩码的相对误差 ≈ 0（完全泄露）；新鲜掩码的相对误差为 O(1)（无泄漏）；")
    print("        MIA 在输出最小化（仅索引）下 AUC 应回到 0.5 附近。")

    if args.figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            names = list(report)
            x = np.arange(len(names))
            fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
            axes[0].bar(x - 0.2, [report[n]["e10a_layer1_fresh"] for n in names], 0.4, label="fresh (M1)")
            axes[0].bar(x + 0.2, [report[n]["e10a_layer1_reuse"] for n in names], 0.4, label="reused")
            axes[0].set_title("E10a mask reuse: relative recovery error")
            axes[0].set_xticks(x); axes[0].set_xticklabels(names); axes[0].legend(); axes[0].grid(alpha=0.3)
            axes[1].bar(x - 0.2, [report[n]["mia_logits_auc"] for n in names], 0.4, label="full logits")
            axes[1].bar(x + 0.2, [report[n]["mia_index_auc"] for n in names], 0.4, label="index only")
            axes[1].axhline(0.5, ls="--", c="k", lw=0.8)
            axes[1].set_title("E10b membership inference AUC")
            axes[1].set_xticks(x); axes[1].set_xticklabels(names); axes[1].legend(); axes[1].grid(alpha=0.3)
            axes[2].bar([str(r["K_h"]) for r in overhead["rows"]],
                        [r["extra_latency_ms"] for r in overhead["rows"]])
            axes[2].set_xlabel("K_h (hybrid layers)")
            axes[2].set_title("E10c obfuscation: extra latency (ms)")
            axes[2].grid(alpha=0.3)
            fig.tight_layout()
            fig_path = data_dir / "E10_summary.png"
            fig.savefig(fig_path, dpi=150)
            print(f"\n图已保存: {fig_path}")
        except Exception as exc:
            print("出图失败:", exc)


if __name__ == "__main__":
    main()
