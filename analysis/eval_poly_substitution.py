"""E1 附加实验：在 ReLU 孪生网络上做「激活函数替换」的精度评估（纯推理，不训练）。

回答的问题：把已经训练好的 ReLU 网络里的激活换成 HE 能算的多项式 P₂（或 x²），
精度会掉多少 —— 这就是论文 Table 5 里"多项式损失 ΔAcc_poly"要的数字。

每个数据集评估 5 种配置：
    relu        : ReLU + ReLU              （基线，用于自检）
    p2          : P₂ + P₂                  （协议 HE-only 档的真实精度）
    square      : x² + x²                  （另一个候选近似，对比用）
    p2_layer1   : P₂ + ReLU                （第 1 个非线性层的精度敏感度）
    p2_layer2   : ReLU + P₂                （第 2 个非线性层的精度敏感度）
后两者是 planner（§5.3 layer_sensitivity）需要的逐层敏感度，正常做法要训练 4 个
变体才能得到，这里靠纯评估一次性拿到。

用法（脚本可以放在任何位置，用 --root 指定项目根目录）：
    python eval_poly_substitution.py
    python eval_poly_substitution.py --datasets mnist
    python eval_poly_substitution.py --modes relu,p2
    python eval_poly_substitution.py --smoke          # 只测 200 个样本，快速验证
    python eval_poly_substitution.py --root /path/to/LitePI   # default: the repo root
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

# ---------------------------------------------------------------- 激活函数
A0, A1, A2 = 0.0625, 0.5, 0.5          # p2(x) = 0.0625 + 0.5x + 0.5x²（与导师脚本一致）


def act_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def act_p2(x: torch.Tensor) -> torch.Tensor:
    return A0 + A1 * x + A2 * x * x


def act_square(x: torch.Tensor) -> torch.Tensor:
    return x * x


ACTS = {"relu": act_relu, "p2": act_p2, "square": act_square}

# 自检基准：导师服务器上跑出的 seed73 明文精度（来自 results/ 里的 JSON）
SELF_CHECK = {"mnist": 0.9785, "cifar10": 0.5881, "uci_har": 0.8860}


# ---------------------------------------------------------------- 网络结构
# 与 model/*_cnn_relu.py 完全一致；load_state_dict(strict=True) 会兜底校验结构
class MnistReluTwin(nn.Module):
    def __init__(self, hidden: int = 64, classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=7, stride=3)
        self.fc1 = nn.Linear(256, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.a1, self.a2 = act_relu, act_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.conv1(x))
        x = x.view(-1, 256)
        x = self.a2(self.fc1(x))
        return self.fc2(x)


class CifarReluTwin(nn.Module):
    def __init__(self, hidden: int = 128, classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 12, kernel_size=5, stride=3)
        self.fc1 = nn.Linear(1200, hidden)
        self.fc2 = nn.Linear(hidden, classes)
        self.a1, self.a2 = act_relu, act_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.conv1(x))
        x = x.view(-1, 1200)
        x = self.a2(self.fc1(x))
        return self.fc2(x)


class HarReluTwin(nn.Module):
    def __init__(self, channels: int = 12, hidden: int = 128, classes: int = 6) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=(3, 9), stride=(2, 1))
        self.fc1 = nn.Linear(channels * 63, hidden)
        self.drop = nn.Dropout(p=0.3)      # eval 模式下不生效
        self.fc2 = nn.Linear(hidden, classes)
        self.a1, self.a2 = act_relu, act_relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.a1(self.conv1(x))
        x = x.view(-1, x.shape[1] * 63)
        x = self.drop(self.a2(self.fc1(x)))
        return self.fc2(x)


# ---------------------------------------------------------------- 每个数据集的配置
CONFIG = {
    "mnist": dict(
        module="mnist_cnn_enc.py", ckpt="mnist_relu_cnn_train.pt",
        cls=MnistReluTwin, prep="identity",
    ),
    "cifar10": dict(
        module="cifar10_cnn_enc.py", ckpt="cifar10_relu_cnn_train.pt",
        cls=CifarReluTwin, prep="cifar",
    ),
    "uci_har": dict(
        module="uci_har_cnn_enc.py", ckpt="uci_har_relu_cnn_train.pt",
        cls=HarReluTwin, prep="har",
    ),
}

MODES = {
    "relu": ("relu", "relu"),
    "p2": ("p2", "p2"),
    "square": ("square", "square"),
    "p2_layer1": ("p2", "relu"),
    "p2_layer2": ("relu", "p2"),
}


# ---------------------------------------------------------------- 工具
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


def get_test_loader(module, dataset: str, batch_size: int):
    out = module.load_data(batch_size)
    if dataset == "uci_har":
        return out[1]                      # (train, test, test_ds, mean, std)
    return out[1]                          # (train, test)


def make_prepare(dataset: str, module):
    if dataset == "cifar10":
        return lambda x: module.preprocess(x)
    if dataset == "uci_har":
        return lambda x: x.unsqueeze(1)
    return lambda x: x


def load_twin(model_cls, ckpt_path: Path) -> nn.Module:
    """按 checkpoint 里记录的结构参数建网，再严格加载权重（strict=True 会兜底校验结构）。"""
    obj = torch.load(str(ckpt_path), map_location="cpu")
    kwargs = {}
    if isinstance(obj, dict):
        if "channels" in obj:
            kwargs["channels"] = int(obj["channels"])
        if "hidden" in obj:
            kwargs["hidden"] = int(obj["hidden"])
    model = model_cls(**kwargs)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    model.load_state_dict(state, strict=True)
    if kwargs:
        print(f"               (权重记录的结构参数: {kwargs})")
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, prepare, device, limit: int | None = None):
    model.eval()
    correct = total = 0
    for images, targets in loader:
        x = prepare(images).to(device)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu()
        correct += int((pred == targets).sum().item())
        total += int(targets.size(0))
        if limit is not None and total >= limit:
            break
    return correct, total, correct / max(total, 1)


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


def guess_root() -> str:
    """优先用脚本自身位置推断项目根目录（脚本在 wzy/analysis/ 下），失败再退回默认路径。"""
    here = Path(__file__).resolve()
    for cand in (here.parent.parent, here.parent, Path(DEFAULT_ROOT)):
        if (cand / "model").is_dir() and (cand / "datas").is_dir():
            return str(cand)
    return DEFAULT_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description="ReLU→P₂ 激活替换的精度评估（E1 附加实验）")
    ap.add_argument("--root", default=None,
                    help="项目根目录（含 model/ datas/）；默认自动从脚本位置推断")
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--modes", default="relu,p2,square,p2_layer1,p2_layer2")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=73, help="权重的训练种子（存档只有 73）")
    ap.add_argument("--smoke", action="store_true", help="只测 200 个样本，快速自检")
    ap.add_argument("--no-save", action="store_true", help="只打印，不写 JSON")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(guess_root())
    model_dir, data_dir = root / "model", root / "datas"
    if not model_dir.exists() or not data_dir.exists():
        sys.exit(f"项目根目录不对：{root}（应包含 model/ 与 datas/）")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    device = torch.device("cpu")
    limit = 200 if args.smoke else None

    print(f"项目根目录 : {root}")
    print(f"设备       : {device}   批大小: {args.batch_size}   "
          f"{'（smoke 模式，仅 200 样本）' if limit else '（全测试集）'}")
    print("=" * 78)

    table: dict[str, dict[str, float]] = {}
    for ds in datasets:
        cfg = CONFIG[ds]
        module = load_module(model_dir / cfg["module"])
        loader = get_test_loader(module, ds, args.batch_size)
        prepare = make_prepare(ds, module)
        n_total = len(loader.dataset)
        ckpt = data_dir / cfg["ckpt"]
        if not ckpt.exists():
            print(f"[{ds}] 缺少权重 {ckpt.name}，跳过")
            continue
        print(f"\n### {ds}   测试集 {n_total} 条   权重 {ckpt.name}")
        table[ds] = {}

        for mode in modes:
            a1, a2 = MODES[mode]
            model = load_twin(cfg["cls"], ckpt)
            model.a1, model.a2 = ACTS[a1], ACTS[a2]
            correct, total, acc = evaluate(model, loader, prepare, device, limit)
            tag = "p2" if mode == "p2" else mode
            method = "relu_baseline" if mode == "relu" else f"approx_{tag}"
            table[ds][mode] = acc
            extra = ""
            if mode == "relu":
                ref = SELF_CHECK[ds]
                extra = f"   自检基准 {ref:.4f} -> {'一致' if abs(acc - ref) < 0.01 else '**不一致，需检查**'}"
            print(f"  {mode:11s} ({a1:6s}+{a2:6s})  正确 {correct:5d}/{total:5d}   "
                  f"精度 {acc:.4f}{extra}")
            if not args.no_save and not args.smoke:
                path = save_record(root, ds, method, args.seed, {
                    "checkpoint": str(ckpt.relative_to(root)),
                    "activation": {"layer1": a1, "layer2": a2},
                    "test_set_size": total,
                    "correct": correct,
                    "accuracy": acc,
                    "note": "post-hoc activation substitution on the ReLU-twin network (no retraining)",
                })
                print(f"               -> {path.relative_to(root)}")

    # ---------------- 汇总 ----------------
    print("\n" + "=" * 78)
    print("汇总（精度 %）")
    head = f"{'数据集':<10}" + "".join(f"{m:>12}" for m in modes)
    print(head)
    for ds, row in table.items():
        line = f"{ds:<10}" + "".join(f"{row.get(m, float('nan'))*100:>12.2f}" for m in modes)
        print(line)
    print()
    print("ΔAcc_poly = ReLU 基线 − P₂ 全替换：")
    for ds, row in table.items():
        if "relu" in row and "p2" in row:
            print(f"  {ds:<10} {(row['relu'] - row['p2'])*100:+.2f} 个点")


if __name__ == "__main__":
    main()
