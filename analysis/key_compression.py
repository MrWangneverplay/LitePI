"""E4: rotation-key pruning for S_key (protocol SSOT §6.4).

§6.4 定义 R = R_conv ∪ R_mm：
    * R_mm    —— FC1 输入维度（对角线法位移量级：MNIST 256 / CIFAR 1200 / HAR 756）
    * R_conv  —— im2col 窗口位移 {k·windows : k=1..taps-1}，最大步 (taps-1)·windows

SEAL 只用 ±2^i 的 Galois 元素就能通过 NAF 分解实现任意旋转，所以 plan 需要的
最高幂次是 ceil(log2(max(|R_conv|, |R_mm|)))，上限 2^11=2048（2^12=4096=N/2 已
属"无旋转"，SEAL 拒绝生成）。本脚本实测：

    * 全集字节（new_context() 默认 create_galois_keys()），和
    * 按 plan 裁剪后的字节（max_rot_pow = ceil(log2 max(|R_conv|,|R_mm|))），

并 probe 确认裁剪不改变密文前向结果。

Run:
    python analysis/key_compression.py            # bytes + 3-sample MNIST probe
    python analysis/key_compression.py --no-probe # bytes only
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from eval_encrypted import _timings, load_test_set
from pi_proto import Wire, make_backend
from pi_server import encrypted_forward

# §6.4: R_mm 是各 canonical 网络 FC1 的输入维度。
R_MM = {"mnist": 256, "cifar10": 1200, "uci_har": 756}
# R_conv 的最大位移 = (taps-1)·windows，由 canonical 几何直接算得：
#   mnist (7x7 kernel, windows 64)  -> 48·64  = 3072
#   cifar10 (5x5 kernel, windows 100) -> 24·100 = 2400
#   uci_har (3x9 kernel, windows 63) -> 26·63 = 1638
R_CONV_MAX = {"mnist": (7 * 7 - 1) * 64,
              "cifar10": (5 * 5 - 1) * 100,
              "uci_har": (3 * 9 - 1) * 63}
# 最大合法 2 幂旋转步：2^11 = 2048（2^12 = 4096 = N/2 已属"无旋转"，SEAL 拒绝生成）。
FULL_POW = 11


def plan_pow(ds: str) -> int:
    """裁剪所需的最高 ±2^i 幂次，覆盖 R_conv 与 R_mm 的最大位移。"""
    step = max(1, R_CONV_MAX[ds], R_MM[ds])
    return min(FULL_POW, max(0, math.ceil(math.log2(step))))


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 rotation-key pruning")
    ap.add_argument("--root", default=None)
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--probe-samples", type=int, default=3)
    ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    be = make_backend("seal")

    curve = []
    for pow_i in range(0, FULL_POW + 1):
        ctx = be.seal.new_context(8192, [40] + [21] * 6 + [40], 21, max_rot_pow=pow_i)
        keys = be.seal.key_sizes(ctx)
        curve.append({
            "max_rot_pow": pow_i,
            "galois_key_count": int(keys.get("galois_key_count", 2 * (pow_i + 1))),
            "galois_keys_bytes": int(keys["galois_keys_bytes"]),
            "s_key_bytes": int(keys["public_key_bytes"]) + int(keys["galois_keys_bytes"])
            + int(keys["relin_keys_bytes"]),
        })
    ctx = be.seal.new_context(8192, [40] + [21] * 6 + [40], 21, max_rot_pow=-1)
    full = be.seal.key_sizes(ctx)
    full_s_key = int(full["public_key_bytes"]) + int(full["galois_keys_bytes"]) \
        + int(full["relin_keys_bytes"])

    per_dataset = {}
    for ds in R_MM:
        p = plan_pow(ds)
        row = next(r for r in curve if r["max_rot_pow"] == p)
        per_dataset[ds] = {
            "R_mm": R_MM[ds],
            "R_conv_max": R_CONV_MAX[ds],
            "max_rot_pow": p,
            "galois_key_count": row["galois_key_count"],
            "s_key_bytes": row["s_key_bytes"],
            "s_key_mb": round(row["s_key_bytes"] / 2**20, 3),
            "saving_vs_full_pct": round(100.0 * (1 - row["s_key_bytes"] / full_s_key), 2),
        }

    print("%8s %10s %12s %10s %13s %12s" % ("dataset", "R_mm", "R_conv", "rot_pow", "pruned S_key", "vs full"))
    for ds, row in per_dataset.items():
        print("%8s %10d %12d %10d %13.1f MB %11.1f%%"
              % (ds, row["R_mm"], row["R_conv_max"], row["max_rot_pow"],
                 row["s_key_mb"], -row["saving_vs_full_pct"]))
    print("full key set: %.1f MB" % (full_s_key / 2**20))

    probe = None
    if not args.no_probe:
        # Pruning must not change the result: one MNIST forward with the pruned
        # key set, compared with the plaintext prediction of the same model.
        model = load_canonical_model(root, "mnist", (0, 0), args.seed, None)
        test = load_test_set(root, "mnist")
        be.max_rot_pow = plan_pow("mnist")
        ctx = be.new_context()
        left, right = socket.socketpair()
        wire = Wire(left, 1)
        hits, errors = 0, []
        t0 = time.perf_counter()
        for i in range(args.probe_samples):
            raw = np.asarray(test[i][0], dtype=np.float64).ravel()
            prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
            prepared = prepared.reshape(model.spec.model_shape)
            if prepared.ndim == 3 and prepared.shape[0] == 1:
                prepared = prepared[0]
            ct, _ = be.im2col_encrypt(ctx, prepared, model.im2col_kernel, model.stride)
            out = encrypted_forward(model, be, ctx, ct, wire, _timings())
            logits = be.decrypt(out)[: model.spec.classes]
            ref = model.plain_forward(raw)[0][: model.spec.classes]
            hits += int(np.argmax(logits) == np.argmax(ref))
            errors.append(float(np.max(np.abs(logits - ref))))
        probe = {
            "dataset": "mnist",
            "max_rot_pow": be.max_rot_pow,
            "samples": args.probe_samples,
            "prediction_matches": hits,
            "max_abs_logit_error": max(errors),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        print("probe: %d/%d 预测一致，最大 logit 误差 %.2e (%.1fs)"
              % (hits, args.probe_samples, probe["max_abs_logit_error"], probe["seconds"]))

    out_dir = root / "results" / "raw_results" / "E4" / "all" / "key_compression"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_id": "E4", "dataset": "all", "method": "key_compression", "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": f"seal {be.version}",
        "profile": "N=8192, [40,21x6,40], scale=2^21",
        "full_key_set": {
            "galois_key_count": int(full.get("galois_key_count", 25)),
            "galois_keys_bytes": int(full["galois_keys_bytes"]),
            "public_key_bytes": int(full["public_key_bytes"]),
            "relin_keys_bytes": int(full["relin_keys_bytes"]),
            "s_key_bytes": full_s_key,
            "s_key_mb": round(full_s_key / 2**20, 3),
        },
        "note": ("按 plan 裁剪覆盖 R_conv ∪ R_mm；conv2d_im2col 的大窗口位移 "
                 "(taps-1)·windows 已把所需幂次推到 2^11，等于全集旋转键，"
                 "因此 S_key 裁剪收益受限于去掉未使用的 conjugation 元素。"),
        "curve": curve,
        "per_dataset": per_dataset,
        "probe": probe,
    }
    (out_dir / f"seed{args.seed}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("saved -> %s" % out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
