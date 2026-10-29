"""E2：LitePI 客户端六项开销（SEAL 4.x + canonical 网络，论文口径）。

与 `client_metrics.py` 的分工：
    client_metrics.py          TenSEAL naive 基线（论文 §5 的对照行，保留不动）
    eval_client_overhead.py    本文件：论文主实现的客户端指标

测什么（每数据集、canonical HE-only 调度 00）：
    T_encode_enc  客户端 im2col 编码 + 加密（SEAL 在同一次 C++ 调用里完成，只报合计）
    T_enc         同槽位零向量加密的标定值；T_encode 由相减得到（标 derived）
    T_dec / T_post  解密 / argmax
    M_client_peak_mb 本进程 RSS 峰值（@10 Hz）。注意：服务端前向、PyTorch、模型与
                  测试集都在本进程里，所以这是"客户端+服务端"的整机内存，**不是**
                  客户端内存（旧版 E2 的 0.5–0.9 GB 即此口径）。客户端内存请用
                  measure_client_memory.py（服务端/客户端分进程，2026-10-29）。
    S_up / S_down / S_key  上下行密文字节与评估密钥字节（pk+galois+relin）
    T_server      服务端密文前向，§7.4 参考，不计入客户端

口径：预热 3 次丢弃，重复 ≥5 次，报 mean±std（实验方案 §2.1/§8）。

用法（WSL，analysis 目录）：
    /usr/bin/python3 eval_client_overhead.py --dataset mnist --samples 5 --repeats 5
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from client_metrics import RssSampler, rss_mb
from eval_encrypted import _timings, load_test_set, save_record
from pi_proto import Wire, make_backend
from pi_server import encrypted_forward

import socket


def summarize(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean_ms": None, "std_ms": None}
    return {"n": len(vals),
            "mean_ms": statistics.fmean(vals) * 1000.0,
            "std_ms": statistics.stdev(vals) * 1000.0 if len(vals) > 1 else 0.0,
            "min_ms": min(vals) * 1000.0, "max_ms": max(vals) * 1000.0}


def summarize_bytes(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0, "mean_bytes": None}
    return {"n": len(vals),
            "mean_bytes": statistics.fmean(vals),
            "std_bytes": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min_bytes": min(vals), "max_bytes": max(vals)}


def main() -> int:
    ap = argparse.ArgumentParser(description="E2 LitePI 客户端开销（SEAL 4）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--backend", default="seal", choices=["seal", "tenseal", "stub"])
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--checkpoint", default=None)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    model = load_canonical_model(root, args.dataset, (0, 0), args.seed,
                                 Path(args.checkpoint) if args.checkpoint else None)
    test = load_test_set(root, args.dataset)
    n = min(args.samples, len(test))

    be = make_backend(args.backend)
    t0 = time.perf_counter()
    ctx = be.new_context()
    setup_s = time.perf_counter() - t0
    if args.backend == "seal":
        keys = be.seal.key_sizes(ctx)
        s_key = int(keys["public_key_bytes"]) + int(keys["galois_keys_bytes"]) \
            + int(keys["relin_keys_bytes"])
    else:
        s_key = None
        keys = {}

    slots = int(model.spec.encoded_slots)
    left, right = socket.socketpair()
    wire = Wire(left, 1)

    t_encode_enc, t_enc_ref, t_server, t_dec, t_post = [], [], [], [], []
    s_up, s_down = [], []

    def run_one(sample, label):
        if args.dataset == "uci_har":
            # load_data already z-scores the windows; prepare_array would normalise twice.
            prepared = np.asarray(sample, dtype=np.float64)
        else:
            raw = np.asarray(sample, dtype=np.float64).ravel()
            prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
        prepared = prepared.reshape(model.spec.model_shape)
        if prepared.ndim == 3 and prepared.shape[0] == 1:
            prepared = prepared[0]

        c0 = time.perf_counter()
        ct, _ = be.im2col_encrypt(ctx, prepared, model.im2col_kernel, model.stride)
        up = len(be.serialize(ct))
        c1 = time.perf_counter()
        out = encrypted_forward(model, be, ctx, ct, wire, _timings())
        c2 = time.perf_counter()
        down = len(be.serialize(out))
        logits = be.decrypt(out)
        c3 = time.perf_counter()
        pred = int(np.argmax(logits[: model.spec.classes]))
        c4 = time.perf_counter()
        ref_vec = model.plain_forward_prepared(prepared.reshape(1, *model.spec.model_shape))[0]
        ref = int(np.argmax(np.ravel(ref_vec)[: model.spec.classes]))
        return (c1 - c0, c2 - c1, c3 - c2, c4 - c3, up, down, pred == ref, pred, ref)

    try:
        for _ in range(args.warmup):
            run_one(test[0][0], test[0][1])

        with RssSampler(0.1) as sampler:
            for _ in range(args.repeats):
                for i in range(n):
                    sample, label = test[i]
                    e0 = time.perf_counter()
                    be.encrypt(ctx, np.zeros(slots, dtype=np.float64))
                    t_enc_ref.append(time.perf_counter() - e0)
                    a, b, c, d, up, down, ok, pred, ref = run_one(sample, label)
                    t_encode_enc.append(a)
                    t_server.append(b)
                    t_dec.append(c)
                    t_post.append(d)
                    s_up.append(up)
                    s_down.append(down)
                    if not ok:
                        print(f"  warn: sample {i} ciphertext pred {pred} != plain {ref}")
        peak = sampler.peak
    finally:
        left.close()
        right.close()

    t_encode = [x - y for x, y in zip(t_encode_enc, t_enc_ref)]
    payload = {
        "experiment": "E2",
        "tier": "client overhead",
        "backend": args.backend,
        "backend_seal_version": getattr(be, "version", None),
        "dataset": args.dataset,
        "schedule": "00",
        "network": "64->10, 1 conv + P2 + fc1 + P2 + fc2 (canonical)",
        "client_metrics": {
            "T_encode_enc_total": summarize(t_encode_enc),
            "T_encode": {**summarize(t_encode), "derived": True,
                         "note": "= im2col_encrypt − 同槽位零向量加密；SEAL 把编码与加密放在一次调用里，"
                                "相减后分辨率受两次 ~10 ms 测量抖动限制，只作量级参考"},
            "T_enc": summarize(t_enc_ref),
            "T_dec": summarize(t_dec),
            "T_post": summarize(t_post),
            "M_client_peak_mb": peak,
            "RSS_baseline_mb": rss_mb(),
            "E_client": "n/a（E6 能耗实验未做）",
        },
        "communication": {
            "S_up_bytes": summarize_bytes([float(x) for x in s_up]),
            "S_down_bytes": summarize_bytes([float(x) for x in s_down]),
            "S_key_bytes": s_key,
            "S_key_detail": {k: int(v) for k, v in keys.items()} if keys else None,
            "note": "S_key = pk + galois + relin（不含 sk）；S_up 为 im2col 输入密文",
        },
        "reference": {
            "T_server": summarize(t_server),
            "context_setup_s": setup_s,
            "encoded_slots": slots,
        },
        "hyperparams": {"samples": n, "repeats": args.repeats,
                        "warmup": args.warmup, "seed": args.seed},
    }
    tag = f"litepi_{args.backend}"
    path = save_record(root, "E2", args.dataset, tag, args.seed, payload)
    m = payload["client_metrics"]
    print(f"T_encode+enc {m['T_encode_enc_total']['mean_ms']:.1f} ms | "
          f"T_enc {m['T_enc']['mean_ms']:.1f} | T_dec {m['T_dec']['mean_ms']:.1f} | "
          f"T_post {m['T_post']['mean_ms']:.3f} | T_server {payload['reference']['T_server']['mean_ms']:.0f} ms")
    print(f"S_up {s_up[0]/1024:.1f} KB  S_down {s_down[0]/1024:.1f} KB  "
          f"S_key {(s_key or 0)/1048576:.2f} MB  RSS {peak:.1f} MB")
    print(f"-> {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
