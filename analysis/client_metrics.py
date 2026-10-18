"""E2 + E4：客户端开销与通信量测量。

测什么（每个数据集、HE-only 配置）：
    T_encode   编码（im2col）耗时
    T_enc      加密耗时
    T_dec      解密耗时
    T_post     后处理（argmax / 类别索引提取）耗时
    T_server   服务端密文前向耗时（供 §7.4 参考，不计入客户端）
    M_client   在线阶段进程 RSS 峰值
    S_up       上行密文字节（客户端上传的输入密文）
    S_down     下行密文字节（服务端返回的结果密文）
    S_key      评估密钥总量（public + galois + relin）

口径说明（必须写进论文脚注）：
    TenSEAL 的 ts.im2col_encoding 把"编码 + 加密"合并在一次 C++ 调用中，Python 层无法拆分。
    因此 T_encode+T_enc 直接测量；T_enc 另以"加密同槽位长度的零向量"标定；T_encode 由相减得到，
    在结果 JSON 中标记为 derived。测量遵循实验方案：预热 3 次丢弃、重复 ≥5 次、取 mean±std。

用法：
    python analysis/client_metrics.py --smoke          # 快速验证（1 样本 × 1 次）
    python analysis/client_metrics.py                  # 5 样本 × 5 次重复
    python analysis/client_metrics.py --datasets mnist
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

DEFAULT_ROOT = str(Path(__file__).resolve().parent.parent)

SPEC = {
    "mnist": dict(module="mnist_cnn_enc.py", ckpt="mnist_he_cnn_train.pt",
                  cls="HeMnistNet", ckpt_is_dict=False, classes=10),
    "cifar10": dict(module="cifar10_cnn_enc.py", ckpt="cifar10_he_cnn_train.pt",
                    cls="HeCifarNet", ckpt_is_dict=False, classes=10),
    "uci_har": dict(module="uci_har_cnn_enc.py", ckpt="uci_har_he_cnn_train.pt",
                    cls="HeUciHarNet", ckpt_is_dict=True, classes=6),
}


# ------------------------------------------------------------------ 通用工具
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


def rss_mb() -> float:
    """当前进程常驻内存（MB）；psutil 缺失时退回 Windows API。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1048576
    except Exception:
        pass
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                return c.WorkingSetSize / 1048576
        except Exception:
            pass
    return -1.0


class RssSampler:
    """后台按 10 Hz 采样 RSS 峰值。"""

    def __init__(self, interval: float = 0.1) -> None:
        self.interval = interval
        self.samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            v = rss_mb()
            if v >= 0:
                self.samples.append(v)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.start = rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2.0)

    @property
    def peak(self) -> float:
        vals = self.samples + ([self.start] if self.start >= 0 else [])
        return max(vals) if vals else -1.0


def save_record(root: Path, exp_id: str, dataset: str, method: str, seed: int, payload: dict) -> Path:
    out_dir = root / "results" / "raw_results" / exp_id / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"exp_id": exp_id, "dataset": dataset, "method": method, "seed": seed,
              "git_sha": git_sha(root),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record.update(payload)
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def summarize(vals: list[float]) -> dict:
    """时间统计（秒 → 毫秒）。"""
    if not vals:
        return {"n": 0, "mean_ms": None, "std_ms": None}
    return {"n": len(vals),
            "mean_ms": statistics.fmean(vals) * 1000,
            "std_ms": (statistics.stdev(vals) * 1000) if len(vals) > 1 else 0.0,
            "min_ms": min(vals) * 1000, "max_ms": max(vals) * 1000}


def summarize_bytes(vals: list[float]) -> dict:
    """字节统计（不做单位换算）。"""
    if not vals:
        return {"n": 0, "mean_bytes": None}
    return {"n": len(vals),
            "mean_bytes": statistics.fmean(vals),
            "std_bytes": (statistics.stdev(vals)) if len(vals) > 1 else 0.0,
            "min_bytes": min(vals), "max_bytes": max(vals)}


# ------------------------------------------------------------------ 数据集相关
def client_encode(ds: str, mod, ts, ctx, sample):
    """与导师 _enc.py 的 encrypted_predict 完全一致的编码+加密步骤。"""
    if ds == "mnist":
        return ts.im2col_encoding(ctx, sample.view(28, 28).tolist(),
                                  mod.KERNEL_SHAPE[0], mod.KERNEL_SHAPE[1], mod.STRIDE)
    if ds == "cifar10":
        gray = mod.preprocess(sample.unsqueeze(0))[0]
        return ts.im2col_encoding(ctx, gray.view(mod.IMG_SIZE, mod.IMG_SIZE).tolist(),
                                  mod.KERNEL_SHAPE[0], mod.KERNEL_SHAPE[1], mod.STRIDE)
    # uci_har：导师脚本已补偿 TenSEAL 的参数交换，这里保持同一顺序
    return ts.im2col_encoding(ctx, sample.numpy().tolist(),
                              mod.KERNEL_SHAPE[1], mod.KERNEL_SHAPE[0], mod.STRIDE)


def plain_input(ds: str, mod, sample):
    if ds == "mnist":
        return sample.unsqueeze(0)
    if ds == "cifar10":
        return mod.preprocess(sample.unsqueeze(0))
    return sample.unsqueeze(0).unsqueeze(0)


def get_test_dataset(ds: str, mod, batch: int):
    """三个 _enc 模块的 load_data 都以第 2 个返回值作为测试集 loader。"""
    return mod.load_data(batch)[1].dataset


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description="E2/E4 客户端开销与通信量测量")
    ap.add_argument("--root", default=None)
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--samples", type=int, default=5, help="参与测量的测试样本数")
    ap.add_argument("--repeats", type=int, default=5, help="重复次数（取 mean±std）")
    ap.add_argument("--warmup", type=int, default=3, help="预热次数（丢弃）")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--smoke", action="store_true", help="1 样本 × 1 次重复，快速验证")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(guess_root())
    model_dir, data_dir = root / "model", root / "datas"
    if not model_dir.is_dir() or not data_dir.is_dir():
        sys.exit(f"项目根目录不对：{root}")
    if args.smoke:
        args.samples, args.repeats, args.warmup = 1, 1, 0

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"项目根目录: {root}")
    print(f"样本数={args.samples}  重复={args.repeats}  预热={args.warmup}")
    print("=" * 84)

    for ds in datasets:
        spec = SPEC[ds]
        mod = load_module(model_dir / spec["module"])
        import tenseal as ts

        # --- 密钥与上下文（setup 阶段，不进入在线测量）---
        t0 = time.perf_counter()
        ctx = mod.create_context()
        setup_s = time.perf_counter() - t0
        # 注意：TenSEAL 的 Context.serialize() 默认就保存 public/galois/relin 三类密钥（sk 默认不存），
        # 必须显式全 False 才能拿到"裸上下文"体积，否则两者相减恒为 0。
        bare = len(ctx.serialize(save_public_key=False, save_secret_key=False,
                                 save_galois_keys=False, save_relin_keys=False))
        full = len(ctx.serialize(save_public_key=True, save_secret_key=False,
                                 save_galois_keys=True, save_relin_keys=True))
        s_key = full - bare

        # --- 模型 ---
        ckpt = torch.load(str(data_dir / spec["ckpt"]), map_location="cpu")
        if spec["ckpt_is_dict"]:
            model = getattr(mod, spec["cls"])(ckpt["channels"], ckpt["hidden"])
            state = ckpt["state_dict"]
        else:
            model = getattr(mod, spec["cls"])()
            state = ckpt
        model.load_state_dict(state)
        model.eval()
        enc_model = mod.EncConvNet(model)

        ds_test = get_test_dataset(ds, mod, 128)
        n = min(args.samples, len(ds_test))
        print(f"\n### {ds}  测试集 {len(ds_test)}  测量样本 {n}")
        print(f"    上下文构建 {setup_s:.2f} s   S_key={s_key/1024:.1f} KB  "
              f"(public+galois+relin, 不含 sk)")

        t_enc_input, t_encode_enc, t_server, t_dec, t_post = [], [], [], [], []
        s_up, s_down = [], []
        agree = 0

        def run_one(sample):
            """返回 (encode+enc 秒, server 秒, dec 秒, post 秒, S_up, S_down, pred, 槽位数)"""
            c0 = time.perf_counter()
            enc_x, windows_nb = client_encode(ds, mod, ts, ctx, sample)
            up = len(enc_x.serialize())
            slots = int(enc_x.size())
            c1 = time.perf_counter()
            enc_out = enc_model(enc_x, windows_nb)
            c2 = time.perf_counter()
            down = len(enc_out.serialize())
            logits = enc_out.decrypt()
            c3 = time.perf_counter()
            pred = int(max(range(spec["classes"]), key=lambda i: logits[i]))
            c4 = time.perf_counter()
            return (c1 - c0, c2 - c1, c3 - c2, c4 - c3, up, down, pred, slots)

        # 预热（同时拿到真实编码槽位数，用于 T_enc 标定）
        slots = 4096
        for _ in range(args.warmup):
            slots = run_one(ds_test[0][0])[7]
        if args.warmup == 0:
            slots = run_one(ds_test[0][0])[7]

        with RssSampler(0.1) as sampler:
            for _ in range(args.repeats):
                for i in range(n):
                    sample, _label = ds_test[i]
                    e0 = time.perf_counter()
                    ts.ckks_vector(ctx, [0.0] * slots)          # 标定 T_enc（同槽位数）
                    enc_ref = time.perf_counter() - e0
                    a, b, c, d, up, down, pred, _ = run_one(sample)
                    t_encode_enc.append(a)
                    t_enc_input.append(enc_ref)
                    t_server.append(b)
                    t_dec.append(c)
                    t_post.append(d)
                    s_up.append(up)
                    s_down.append(down)
                    with torch.no_grad():
                        # plain_input 已含 batch 维，不能再 unsqueeze(0)
                        ref = int(model(plain_input(ds, mod, sample)).argmax(dim=1).item())
                    agree += (ref == pred)
        peak = sampler.peak

        t_encode_derived = [x - y for x, y in zip(t_encode_enc, t_enc_input)]
        payload = {
            "client_metrics": {
                "T_encode_enc_total": summarize(t_encode_enc),
                "T_encode": {**summarize(t_encode_derived), "derived": True,
                             "note": "= T_encode+enc_total − T_enc标定；TenSEAL 的 im2col_encoding 不可拆分"},
                "T_enc": summarize(t_enc_input),
                "T_dec": summarize(t_dec),
                "T_post": summarize(t_post),
                "M_client_peak_mb": peak,
                "E_client": "n/a（E6 能耗实验未做）",
            },
            "communication": {
                "S_up_bytes": summarize_bytes([float(x) for x in s_up]),
                "S_down_bytes": summarize_bytes([float(x) for x in s_down]),
                "S_key_bytes": s_key,
                "note": "S_up/S_down 为密文序列化字节；S_key = public+galois+relin（不含 sk）",
            },
            "reference": {
                "T_server": summarize(t_server),
                "context_setup_s": setup_s,
                "ckpt": spec["ckpt"],
                "encoded_slots": slots,
                "plaintext_vs_ciphertext_agreement": f"{agree}/{n * args.repeats}",
            },
            "hyperparams": {"samples": n, "repeats": args.repeats, "warmup": args.warmup,
                            "poly_modulus_degree": getattr(mod, "POLY_MODULUS_DEGREE", None),
                            "global_scale": getattr(mod, "GLOBAL_SCALE", None)},
        }
        path = save_record(root, "E2", ds, "he_only", args.seed, payload)
        m = payload["client_metrics"]
        print(f"    T_encode+enc {m['T_encode_enc_total']['mean_ms']:.1f} ms | "
              f"T_enc {m['T_enc']['mean_ms']:.1f} | T_dec {m['T_dec']['mean_ms']:.1f} | "
              f"T_post {m['T_post']['mean_ms']:.3f} | T_server {payload['reference']['T_server']['mean_ms']:.0f} ms")
        print(f"    S_up {s_up[0]/1024:.1f} KB  S_down {s_down[0]/1024:.1f} KB  "
              f"RSS 峰值 {peak:.1f} MB  明密文一致 {agree}/{n * args.repeats}")
        print(f"    -> {path.relative_to(root)}")

    print("\n测量完成。")


if __name__ == "__main__":
    main()
