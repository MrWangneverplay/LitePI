"""E1 · Plain-Poly 档：多种子补训驱动（复用导师 model/*_cnn_poly2.py 的网络与训练函数）。

为什么需要这个脚本：导师的三个 *_cnn_poly2.py 只有单次训练入口，既没有 --seed/--num-seeds，
也不把结果落盘成 JSON（只有 *_cnn_relu.py 有这套）。而导师红线②要求每个数据点 ≥5 个种子、
报告 mean±std，且红线①要求结果写进 results/raw_results/。本脚本补上这两点。

特点：
  · 复用导师模块里的网络定义、数据加载、训练超参（Adam / AdamW + Cosine 等原样保留）
  · 每个种子独立设置 torch.manual_seed，训练完保存权重与 JSON
  · **可断点续跑**：已存在 JSON 的种子自动跳过
  · 不动导师的任何原脚本

用法：
    python analysis/train_poly2_seeds.py                      # 三个数据集 × 5 个种子
    python analysis/train_poly2_seeds.py --datasets mnist
    python analysis/train_poly2_seeds.py --seeds 74,75,76,77  # 只补后面 4 个
    python analysis/train_poly2_seeds.py --smoke              # 1 个种子 × 1 epoch，验证流程
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

DEFAULT_ROOT = r"D:\王梓杨\项目组\wzy"

SPEC = {
    "mnist": dict(module="mnist_cnn_poly2.py", epochs=10, batch=64,
                  method="plain_poly", tag="mnist_poly2_he_cnn_train"),
    "cifar10": dict(module="cifar10_cnn_poly2.py", epochs=40, batch=128,
                    method="plain_poly", tag="cifar10_poly2_he_cnn_train"),
    "uci_har": dict(module="uci_har_cnn_poly2.py", epochs=30, batch=128,
                    method="plain_poly", tag="uci_har_poly2_he_cnn_train"),
}

# 自检：seed 73 应与已跑出的数字一致（昨日单种子结果）
SELF_CHECK = {"mnist": 0.9782, "cifar10": 0.6060, "uci_har": 0.8789}


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


def build_model(dataset: str, module):
    if dataset == "uci_har":
        return module.HeUciHarNet(activation="poly2")
    return module.HeMnistNet(activation="poly2") if dataset == "mnist" \
        else module.HeCifarNet(activation="poly2")


def get_loaders(dataset: str, module, batch: int):
    out = module.load_data(batch)
    return out[0], out[1]           # train, test（HAR 返回 5 元组，前两个也是 loader）


def save_record(root: Path, dataset: str, method: str, seed: int, payload: dict) -> Path:
    out_dir = root / "results" / "raw_results" / "E1" / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "exp_id": "E1",
        "dataset": dataset,
        "method": method,
        "seed": seed,
        "git_sha": git_sha(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    record.update(payload)
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="E1 Plain-Poly 档多种子补训")
    ap.add_argument("--root", default=None, help="项目根目录（默认自动推断）")
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--seeds", default="73,74,75,76,77")
    ap.add_argument("--epochs", type=int, default=None, help="覆盖默认轮数（调试用）")
    ap.add_argument("--smoke", action="store_true", help="1 个种子 × 1 epoch，验证流程")
    ap.add_argument("--no-checkpoint", action="store_true", help="只存 JSON，不保存权重")
    ap.add_argument("--force", action="store_true", help="已有 JSON 的种子也重训")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(guess_root())
    model_dir, data_dir = root / "model", root / "datas"
    if not model_dir.is_dir() or not data_dir.is_dir():
        sys.exit(f"项目根目录不对：{root}")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.smoke:
        seeds = [seeds[0]]

    device = torch.device("cpu")
    print(f"项目根目录: {root}")
    print(f"设备: {device}   数据集: {datasets}   种子: {seeds}")
    print("=" * 78)

    summary: dict[str, list[float]] = {}
    t_all = time.perf_counter()

    for ds in datasets:
        spec = SPEC[ds]
        module = load_module(model_dir / spec["module"])
        n_train = len(module.load_data(spec["batch"])[0].dataset)
        epochs = 1 if args.smoke else (args.epochs or spec["epochs"])
        print(f"\n### {ds}   轮数 {epochs}   训练样本 {n_train}")
        summary[ds] = []

        for seed in seeds:
            json_path = root / "results" / "raw_results" / "E1" / ds / spec["method"] / f"seed{seed}.json"
            ckpt_path = data_dir / f"{spec['tag']}_seed{seed}.pt"
            if json_path.exists() and not args.force:
                print(f"  seed {seed}: 已有 JSON，跳过")
                summary[ds].append(json.loads(json_path.read_text(encoding="utf-8"))["accuracy"]["test_acc_final"])
                continue

            t0 = time.perf_counter()
            # 先设种子，再建 loader —— 与导师脚本的顺序一致，保证可复现
            torch.manual_seed(seed)
            train_loader, test_loader = get_loaders(ds, module, spec["batch"])
            model = build_model(ds, module).to(device)
            history = module.train(model, train_loader, test_loader, device, epochs, 0.001)
            model.eval()
            _, test_acc = module.run_epoch(model, test_loader,
                                           torch.nn.CrossEntropyLoss(), device)
            train_time = time.perf_counter() - t0

            # 冒烟模式只验证流程，绝不落盘（否则会污染正式数据点）
            if not args.no_checkpoint and not args.smoke:
                torch.save(model.state_dict(), ckpt_path)

            payload = {
                "hyperparams": {"epochs": epochs, "batch_size": spec["batch"],
                                "lr": 0.001, "optimizer": "Adam/AdamW(按导师脚本)", "activation": "poly2",
                                "p2_coeffs": [0.0625, 0.5, 0.5]},
                "device": str(device),
                "accuracy": {"test_acc_final": float(test_acc),
                             "test_acc_best": float(max(history["test_accuracy"])),
                             "train_time_s": round(train_time, 1)},
                "checkpoint": None if args.no_checkpoint else str(ckpt_path.relative_to(root)),
                "source": "retrained",
            }
            if args.smoke:
                print(f"  [smoke] seed {seed}: 精度 {test_acc:.4f}（1 epoch，**不写 JSON、不存权重**）")
                summary[ds].append(float(test_acc))
                continue
            path = save_record(root, ds, spec["method"], seed, payload)
            summary[ds].append(float(test_acc))
            note = ""
            if seed == 73 and abs(test_acc - SELF_CHECK[ds]) < 0.01:
                note = "   自检：与昨日单种子结果一致"
            print(f"  seed {seed}: 精度 {test_acc:.4f}  (最优 {max(history['test_accuracy']):.4f})  "
                  f"用时 {train_time/60:.1f} min{note}")
            print(f"           -> {path.relative_to(root)}")

    print("\n" + "=" * 78)
    print("汇总（test_acc_final，mean ± std）")
    for ds, accs in summary.items():
        if not accs:
            continue
        mean = statistics.fmean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {ds:<10} n={len(accs)}  mean={mean*100:.2f}%  std={std*100:.2f}  "
              f"min={min(accs)*100:.2f}%  max={max(accs)*100:.2f}%")
    print(f"\n总用时 {(time.perf_counter() - t_all)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
