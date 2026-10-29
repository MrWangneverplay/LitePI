"""E10c：§7.2 调度混淆的真实开销（实测，替代旧解析估算）。

对每个 (数据集, 调度) run 两次真实查询：一次正常（只有 hybrid 层岔路），
一次 `--obfuscate-schedule`（每个非线性层都走一整段 hybrid 岔路，
dummy 段与真实段在帧类型/字节/SCI 流量上不可区分）。比较
上行/下行字节、协议往返数、SCI 元素数、客户端在线时间和服务端前向时间。

为什么必须实测：混淆后每个 P₂ 层也要付一整段 hybrid 的通信 + SCI + 再加密，
而旧模型按"每层一次 128KB ≈ E4 密文"估算，既低估轮数也低估 SCI 开销。

用法：
    python analysis/gen_e10c_data.py                       # 3 数据集 × 4 调度
    python analysis/gen_e10c_data.py --datasets mnist --schedules 00,11
    python analysis/gen_e10c_data.py --wire baseline --repeat 3
输出：results/E10/all/scheduling_obfuscation_measured/*.json + summary.csv/.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from canonical_net import CANONICAL_SPECS  # noqa: E402


def _domains(root: Path, dataset: str, schedule: str, seed: int):
    if "1" not in schedule:
        return None
    from mask_session import load_mask_domains
    path = root / "datas" / "mask_domains" / f"{dataset}_s{schedule}_seed{seed}.json"
    if not path.exists():
        raise SystemExit(f"缺少掩码域文件: {path}（先跑 export_mask_domains.py 或 mask_session.py --upgrade）")
    canonical = CANONICAL_SPECS[dataset]
    return load_mask_domains(
        path, output_layer_id=2, expected_dataset=dataset,
        expected_schedule=tuple(int(b) for b in schedule),
        expected_seed=seed,
        expected_dims=[canonical.conv_flat, canonical.hidden],
    )


def run_point(dataset: str, schedule: str, obfuscate: bool, seed: int,
              wire: str, input_seed: int, root: Path) -> dict:
    from pi_client import loopback_once
    result = loopback_once(
        "seal", [64, 32, 10], seed, input_seed,
        schedule=tuple(int(b) for b in schedule),
        dataset=dataset, root=root,
        mask_domains=_domains(root, dataset, schedule, seed),
        obfuscate_schedule=obfuscate, wire=wire,
    )
    srv = result["server"]
    keygen = float(result["timings"].get("keygen_s", 0.0))
    ctx_up = float(result["timings"].get("context_upload_s", 0.0))
    return {
        "dataset": dataset, "schedule": schedule,
        "obfuscated": bool(obfuscate), "seed": seed, "wire": wire,
        "online_wall_s": round(float(result["wall_s"]) - keygen - ctx_up, 4),
        "keygen_s": round(keygen, 4), "context_upload_s": round(ctx_up, 4),
        # Online (per-query) bytes: the server resets its counters at query
        # start, so these exclude the one-time context upload (177 MB) while
        # the client's own totals do not.
        "c2s_online_bytes": int(srv.get("bytes_received", 0)),
        "s2c_online_bytes": int(srv.get("bytes_sent", 0)),
        "client_total_sent": int(result["bytes_sent"]),
        "client_total_received": int(result["bytes_received"]),
        "protocol_online_rounds": int(srv.get("protocol_online_rounds", 0)),
        "protocol_online_messages": int(srv.get("protocol_online_messages", 0)),
        "dummy_legs": int(srv.get("dummy_legs", 0)),
        "hybrid_roundtrips": int(srv.get("hybrid_roundtrips", 0)),
        "server_forward_s": float(srv.get("wall_s", 0.0)),
        "mpc_s": round(float(result["timings"].get("mpc_s", 0.0)), 4),
        "mpc_elements": int(result["timings"].get("mpc_elements", 0)),
        "level_aligned": bool(srv.get("obfuscation_level_aligned", False)),
        "loopback_passed": bool(result.get("loopback", {}).get("passed", False)),
        "max_abs_err": float(result.get("loopback", {}).get("max_abs_err", 0.0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E10c 调度混淆开销（实测）")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--schedules", default="00,01,10,11")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--input-seed", type=int, default=0)
    ap.add_argument("--repeat", type=int, default=3,
                    help="每个点重复次数，取中位数（本机未开 netem，抖动来自 CPU 调度）")
    ap.add_argument("--out", type=Path, default=ROOT / "results/E10/all/scheduling_obfuscation_measured")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    schedules = [s.strip() for s in args.schedules.split(",") if s.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for ds in datasets:
        for sched in schedules:
            pair = {}
            for obf in (False, True):
                reps = [run_point(ds, sched, obf, args.seed, args.wire,
                                  args.input_seed, args.root)
                        for _ in range(args.repeat)]
                med = {"dataset": ds, "schedule": sched, "obfuscated": obf,
                       "repeat": len(reps)}
                for key in ("online_wall_s", "c2s_online_bytes", "s2c_online_bytes",
                            "protocol_online_rounds", "protocol_online_messages",
                            "dummy_legs", "hybrid_roundtrips", "server_forward_s",
                            "mpc_s", "mpc_elements"):
                    med[key] = st.median([r[key] for r in reps])
                med["loopback_passed"] = all(r["loopback_passed"] for r in reps)
                med["level_aligned"] = reps[-1]["level_aligned"]
                med["max_abs_err"] = max(r["max_abs_err"] for r in reps)
                med["runs"] = reps
                pair[obf] = med
                rows.append(med)
            if True in pair and False in pair:
                base, obf = pair[False], pair[True]
                d = {
                    "dataset": ds, "schedule": sched,
                    "extra_online_wall_s": obf["online_wall_s"] - base["online_wall_s"],
                    "wall_ratio": (obf["online_wall_s"] / base["online_wall_s"]
                                   if base["online_wall_s"] else None),
                    "extra_client_up_bytes": obf["c2s_online_bytes"] - base["c2s_online_bytes"],
                    "extra_client_down_bytes": obf["s2c_online_bytes"] - base["s2c_online_bytes"],
                    "extra_protocol_rounds": obf["protocol_online_rounds"] - base["protocol_online_rounds"],
                    "extra_protocol_messages": obf["protocol_online_messages"] - base["protocol_online_messages"],
                    "extra_dummy_legs": obf["dummy_legs"] - base["dummy_legs"],
                    "extra_mpc_elements": obf["mpc_elements"] - base["mpc_elements"],
                    "extra_server_forward_s": obf["server_forward_s"] - base["server_forward_s"],
                    "k_h": sched.count("1"),
                }
                print(f"  {ds:<8}s{sched}  K_h={d['k_h']}  "
                      f"上行 {base['c2s_online_bytes']/1024:8.0f}→{obf['c2s_online_bytes']/1024:8.0f} KB  "
                      f"下行 {base['s2c_online_bytes']/1024:8.0f}→"
                      f"{obf['s2c_online_bytes']/1024:8.0f} KB  轮数 {base['protocol_online_rounds']}→"
                      f"{obf['protocol_online_rounds']}  在线 {base['online_wall_s']:.1f}→"
                      f"{obf['online_wall_s']:.1f}s  dummy={d['extra_dummy_legs']}")
                rows.append(d)

    # 服务端的 bytes_received 计数器包含一次性的 RLWE 上下文上传（约 176 MB），
    # 不能直接当作"每次查询的上行"。用同一数据集内 s00/s01 的实测差还原单个
    # 新鲜密文字节数，再反推上下文大小并逐行扣除；`c2s_total_bytes` 保留原值。
    context = {}
    for ds in datasets:
        base = {r["schedule"]: r["c2s_online_bytes"] for r in rows
                if "obfuscated" in r and r["dataset"] == ds and not r["obfuscated"]}
        if "00" in base and "01" in base:
            ct = base["01"] - base["00"]
            context[ds] = {"ciphertext_bytes": ct, "context_bytes": base["00"] - ct}
    for r in rows:
        if "obfuscated" in r and r["dataset"] in context:
            r["c2s_total_bytes"] = r["c2s_online_bytes"]
            r["c2s_online_bytes"] -= context[r["dataset"]]["context_bytes"]
    for d in (r for r in rows if "k_h" in r):
        base = next(r for r in rows if "obfuscated" in r
                    and r["dataset"] == d["dataset"] and r["schedule"] == d["schedule"]
                    and not r["obfuscated"])
        obf = next(r for r in rows if "obfuscated" in r
                   and r["dataset"] == d["dataset"] and r["schedule"] == d["schedule"]
                   and r["obfuscated"])
        d["baseline_up_bytes"] = base["c2s_online_bytes"]
        d["obfuscated_up_bytes"] = obf["c2s_online_bytes"]
        d["baseline_down_bytes"] = base["s2c_online_bytes"]
        d["obfuscated_down_bytes"] = obf["s2c_online_bytes"]
        d["extra_client_up_bytes"] = obf["c2s_online_bytes"] - base["c2s_online_bytes"]
        d["up_ratio"] = (obf["c2s_online_bytes"] / base["c2s_online_bytes"]
                         if base["c2s_online_bytes"] else None)
        d["down_ratio"] = (obf["s2c_online_bytes"] / base["s2c_online_bytes"]
                           if base["s2c_online_bytes"] else None)
        d["accuracy_preserved"] = base["loopback_passed"] == obf["loopback_passed"]

    (args.out / "runs.json").write_text(
        json.dumps({"experiment": "E10c", "measured_at":
                    datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "method": "pi_client.loopback_once with/without obfuscate_schedule; "
                              "median of --repeat runs; no netem (CPU jitter only)",
                    "accounting": "c2s_online_bytes excludes the one-time context upload; "
                                  "context size recovered from the s01-s00 delta",
                    "context_upload_bytes": context,
                    "seed": args.seed, "wire": args.wire, "rows": rows},
                   indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = ["dataset", "schedule", "k_h",
                "baseline_up_bytes", "obfuscated_up_bytes", "up_ratio",
                "baseline_down_bytes", "obfuscated_down_bytes", "down_ratio",
                "extra_online_wall_s", "wall_ratio", "extra_client_up_bytes",
                "extra_client_down_bytes", "extra_protocol_rounds",
                "extra_protocol_messages", "extra_dummy_legs",
                "extra_mpc_elements", "extra_server_forward_s", "accuracy_preserved"]
        wr = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            if "k_h" in r:
                wr.writerow(r)
    print(f"written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
