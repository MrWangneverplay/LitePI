"""E4: per-message bytes of every wire profile on the real SEAL serialiser.

The byte count of a CKKS message depends only on its level, the codec and
whether it is seeded - not on the weights or the input.  This script walks
the protocol trace of each canonical network (pi_server.protocol_trace, the
same function the server uses to plan mod-switching and short-chain
encryption), produces every message with the backend calls the protocol
code uses (encrypt_for_send / shrink_for_send / serialize), and records its
size.  No torch, no checkpoint, no network needed, so it runs anywhere the
litepi_seal extension builds.

Messages (protocol §2.2):  M0 keys (once per session), M1, ①_k, ②_k, M3.

    python analysis/measure_wire_bytes.py                 # all datasets/schedules/profiles
    python analysis/measure_wire_bytes.py --profiles baseline,compact

Output: results/raw_results/E4/all/wire_bytes/seed73.json (+ .csv)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from canonical_net import CANONICAL_SPECS
from mask_session import MaskDomain, MaskSession
from params_pi import COEFF_MOD_BIT_SIZES
from pi_proto import WIRE_PROFILES, make_backend
from pi_server import protocol_trace, wire_plan

SCHEDULES = ("00", "01", "10", "11")
# Activation bounds only set the M2 width, which the 2^42 cap fixes anyway;
# these are the MNIST s11 export values (B=2^4 / 2^6, logits 2^6).
BOUNDS = {0: 16.0, 1: 64.0, 2: 64.0}


def _git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def measure(dataset: str, bits: str, profile_name: str, cache: dict) -> dict:
    spec = CANONICAL_SPECS[dataset]
    schedule = tuple(int(b) for b in bits)
    profile = WIRE_PROFILES[profile_name]
    model = SimpleNamespace(architecture="canonical_cnn", schedule=schedule,
                            output_layer_id=lambda: 2)
    session = MaskSession(domains={k: MaskDomain.for_bound(b)
                                   for k, b in BOUNDS.items()})
    plan = None if profile.is_baseline else wire_plan(model, session, profile)

    key = profile_name
    if key not in cache:
        be = make_backend("seal", profile)
        ctx = be.new_context()
        be.ctx = ctx
        sizes = be.seal.key_sizes_wire(ctx, profile.compr, profile.seeded_keys)
        cache[key] = (be, ctx, dict(sizes))
    be, ctx, key_sizes = cache[key]
    top = len(COEFF_MOD_BIT_SIZES) - 2

    dims = {"M1": spec.encoded_slots, 0: spec.conv_flat, 1: spec.hidden,
            "out": spec.classes}
    rng = np.random.default_rng(0)
    msgs = []
    level = top
    for op in protocol_trace(model):
        kind, tag = op
        if kind == "fresh":
            name = "M1" if tag == "M1" else f"re{tag[1]}"
            level = top if plan is None else plan["chain_index"]["logits"][name]
            n = dims["M1"] if tag == "M1" else dims[tag[1]]
            blob = be.encrypt_for_send(ctx, rng.uniform(-1, 1, n),
                                       None if plan is None else level)
            msgs.append({"msg": "M1" if tag == "M1" else f"②_{tag[1] + 1}",
                         "direction": "C->S", "chain_index": level,
                         "bytes": len(blob)})
        elif kind == "rescale":
            level -= tag
        else:  # decrypt point: the server sends at the current level
            n = dims["out"] if tag == "out" else dims[tag[1]]
            ct = ctx.encrypt_at(rng.uniform(-1, 1, n), level)
            send_key = "out" if tag == "out" else f"hyb{tag[1]}"
            if plan is not None:
                ct = be.shrink_for_send(ct, plan["send_abs"][send_key])
            blob = be.serialize(ct)
            msgs.append({"msg": "M3" if tag == "out" else f"①_{tag[1] + 1}",
                         "direction": "S->C",
                         "chain_index": int(ctx.index(ct)),
                         "modulus_bits": int(ctx.modulus_bits(ct)),
                         "bytes": len(blob)})
    up = sum(m["bytes"] for m in msgs if m["direction"] == "C->S")
    down = sum(m["bytes"] for m in msgs if m["direction"] == "S->C")
    return {
        "dataset": dataset, "schedule": bits, "wire": profile.to_dict(),
        "messages": msgs,
        "online_up_bytes": up, "online_down_bytes": down,
        "online_total_bytes": up + down,
        "M0_key_bytes": int(key_sizes["public_context_bytes"]),
        "key_sizes": {k: (int(v) if isinstance(v, (int, np.integer)) else v)
                      for k, v in key_sizes.items()},
        "plan": plan,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="E4 wire-profile byte measurement")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--profiles", default=",".join(WIRE_PROFILES))
    ap.add_argument("--datasets", default="mnist,cifar10,uci_har")
    ap.add_argument("--max-rot-pow", type=int, default=-1,
                    help="Galois key subset as in key_compression.py (-1 = full)")
    args = ap.parse_args()

    cache: dict = {}
    rows = []
    for name in args.profiles.split(","):
        for ds in args.datasets.split(","):
            for bits in SCHEDULES:
                try:
                    rows.append(measure(ds, bits, name, cache))
                except Exception as exc:   # e.g. codec missing in this build
                    print(f"[skip] {name}/{ds}/s{bits}: {exc}")
                    break
    base = {(r["dataset"], r["schedule"]): r for r in rows
            if r["wire"]["name"] == "baseline"}
    print(f"{'profile':>10} {'ds':>8} {'s':>3} {'up KB':>9} {'down KB':>9} "
          f"{'online x':>9} {'M0 MB':>8}")
    for r in rows:
        b = base.get((r["dataset"], r["schedule"]))
        ratio = (b["online_total_bytes"] / r["online_total_bytes"]) if b else float("nan")
        r["online_reduction_vs_baseline"] = round(ratio, 3)
        if b:
            r["M0_reduction_vs_baseline"] = round(b["M0_key_bytes"] / r["M0_key_bytes"], 3)
        print(f"{r['wire']['name']:>10} {r['dataset']:>8} {r['schedule']:>3} "
              f"{r['online_up_bytes']/1024:9.1f} {r['online_down_bytes']/1024:9.1f} "
              f"{ratio:9.2f} {r['M0_key_bytes']/2**20:8.1f}")

    be = make_backend("seal")
    out_dir = args.root / "results" / "raw_results" / "E4" / "all" / "wire_bytes"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "exp_id": "E4",
        "experiment": "wire-profile bytes (per message, real SEAL serialisation)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": _git_sha(args.root),
        "seal_version": getattr(be, "version", "unknown"),
        "seal_codecs": list(be.seal.compr_modes()),
        "note": ("bytes depend on level/codec/seeding only; S_key uses the full "
                 "Galois set unless --max-rot-pow is given"),
        "rows": rows,
    }
    (out_dir / "seed73.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (out_dir / "seed73.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["profile", "dataset", "schedule", "message", "direction",
                    "chain_index", "bytes"])
        for r in rows:
            for m in r["messages"]:
                w.writerow([r["wire"]["name"], r["dataset"], r["schedule"],
                            m["msg"], m["direction"], m["chain_index"], m["bytes"]])
            w.writerow([r["wire"]["name"], r["dataset"], r["schedule"], "M0",
                        "C->S", "", r["M0_key_bytes"]])
    print(f"written -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
