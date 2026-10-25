"""E4：通信与密钥字节实测（协议 SSOT §6.2 / §6.4 的【待补】项）。

量什么：
    * 单个密文在 L7..L2 每一级的实测序列化字节（SEAL 4.x 的 RNS 编码）
    * 同一级的协议表口径字节 2·N·⌈log2 q_l / 8⌉（§6.2 公式的紧凑打包）
    * 客户端密钥：pk / galois（rotation）/ relin / sk 各自字节与 key 条目数
    * 按调度汇总在线双向字节：S_up 恒为 fresh L7；S_down 随 s 落在 L2/L4/L6

口径说明（论文脚注用）：
    SEAL 的 Ciphertext::save 是 RNS 形式——每个系数、每个当前模数素数存 8 字节；
    协议表 §6.2 用的是"把 log2 q_l 位紧凑打包"的理想口径。两者都要给，
    实测值才是线上真正跑出来的字节数。

用法（WSL，analysis 目录）：
    ../.venv311/bin/python comm_bytes.py --backend seal
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from params_pi import COEFF_MOD_BIT_SIZES, GLOBAL_SCALE_BITS, POLY_MODULUS_DEGREE
from pi_proto import make_backend

# §5.3 轨迹：fresh L7 = 7 个数据素数，每 rescale 丢一个，末尾 L2 = 2 个素数。
DATA_PRIMES = COEFF_MOD_BIT_SIZES[:-1]          # 最后一个 40-bit 只用于 key switch
LEVEL_BITS = {}
for _n in range(len(DATA_PRIMES), 1, -1):
    LEVEL_BITS[f"L{_n}"] = sum(DATA_PRIMES[: _n])

# 每一级在 §5.3 里承担的角色（S_down 落到哪一级）。
DOWN_ROLE = {"L7": "M1 up / ②_k re-encrypt (fresh)", "L6": "s=10,11 downlink",
             "L5": "——", "L4": "s=01 downlink", "L3": "——", "L2": "s=00 (HE-only) result"}


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=False, timeout=10)
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def save_record(root: Path, exp_id: str, dataset: str, method: str, seed: int,
                payload: dict) -> Path:
    out_dir = root / "results" / "raw_results" / exp_id / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"exp_id": exp_id, "dataset": dataset, "method": method, "seed": seed,
              "git_sha": git_sha(root),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record.update(payload)
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def protocol_bytes(bits: int) -> int:
    """§6.2：2 · N · ⌈log2 q_l / 8⌉（每系数按 8-bit 对齐的紧凑打包）。"""
    return 2 * POLY_MODULUS_DEGREE * int(math.ceil(bits / 8))


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 通信/密钥字节实测")
    ap.add_argument("--root", default=None)
    ap.add_argument("--backend", default="seal", choices=["seal", "tenseal"])
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    be = make_backend(args.backend)
    ctx = be.new_context()

    # ---- 密钥 -------------------------------------------------------------
    if args.backend == "seal":
        keys = {k: int(v) for k, v in be.seal.key_sizes(ctx).items()}
    else:
        bare = len(ctx.serialize(save_public_key=False, save_secret_key=False,
                                 save_galois_keys=False, save_relin_keys=False))
        full = len(ctx.serialize(save_public_key=True, save_secret_key=False,
                                 save_galois_keys=True, save_relin_keys=True))
        pk = len(ctx.serialize(save_public_key=True, save_secret_key=False,
                               save_galois_keys=False, save_relin_keys=False)) - bare
        relin = len(ctx.serialize(save_public_key=False, save_secret_key=False,
                                  save_galois_keys=False, save_relin_keys=True)) - bare
        galois = full - bare - pk - relin
        keys = {"public_key_bytes": pk, "galois_keys_bytes": galois,
                "relin_keys_bytes": relin, "public_context_bytes": full - bare,
                "measured_with": "tenseal context serialize"}

    # ---- 密文逐级字节 -----------------------------------------------------
    ones = np.ones(64, dtype=np.float64)
    ct = be.encrypt(ctx, ones)
    ladder = []
    for step in range(len(DATA_PRIMES) - 1):
        name = f"L{len(DATA_PRIMES) - step}"
        bits = LEVEL_BITS[name]
        if args.backend == "seal":
            index = int(ctx.index(ct))
        else:
            index = len(DATA_PRIMES) - 1 - step
        ladder.append({
            "level": name,
            "chain_index": index,
            "modulus_bits": bits,
            "measured_bytes": len(be.serialize(ct)),
            "protocol_table_bytes": protocol_bytes(bits),
            "rns_ratio": round(len(be.serialize(ct)) / protocol_bytes(bits), 4),
            "downlink_role": DOWN_ROLE[name],
        })
        if step + 1 < len(DATA_PRIMES) - 1:
            ct = be.square(ct)   # one rescale per step, exactly the §5.3 trace

    payload = {
        "experiment": "E4",
        "tier": "measured bytes",
        "backend": args.backend,
        "backend_seal_version": getattr(be, "version", None),
        "params": {
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
            "global_scale_bits": GLOBAL_SCALE_BITS,
        },
        "keys": keys,
        "ciphertext_ladder": ladder,
        "online_totals": {
            "S_up_bytes": ladder[0]["measured_bytes"],
            "S_down_bytes_by_schedule": {
                "00": {**next(x for x in ladder if x["level"] == "L2"),
                       "round_trips": 0},
                "01": {**next(x for x in ladder if x["level"] == "L4"),
                       "round_trips": 2},
                "10": {**next(x for x in ladder if x["level"] == "L6"),
                       "round_trips": 2},
                "11": {**next(x for x in ladder if x["level"] == "L6"),
                       "round_trips": 2},
            },
            "note": ("hybrid 岔路的 ①_k 下行落在 L4/L6，②_k 上行是 fresh L7（重加密）；"
                     "00 无岔路，只有 M1 上行与 M3 下行"),
        },
    }
    path = save_record(root, "E4", "all", "comm_bytes", args.seed, payload)

    print(f"backend={args.backend} {getattr(be, 'version', '')}".strip())
    print("keys:", json.dumps({k: v for k, v in keys.items()}, ensure_ascii=False))
    print(f"{'level':>5} {'index':>5} {'bits':>5} {'measured':>10} {'§6.2 table':>11} {'ratio':>7}")
    for row in ladder:
        print(f"{row['level']:>5} {row['chain_index']:>5} {row['modulus_bits']:>5} "
              f"{row['measured_bytes']:>10} {row['protocol_table_bytes']:>11} "
              f"{row['rns_ratio']:>7}")
    print(f"-> {path.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
