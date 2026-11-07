"""E7 Table 8: what each LitePI design decision buys, one row per ablation.

Every row compares the full design with the same system minus one decision and
says where its numbers come from:

  measured    both sides ran on the real protocol (real SEAL + SCI, per-schedule
              ciphertext runs of model B, separate client process for memory)
  derived     measured unit cost times a protocol count, no run of the ablated
              variant exists
  baseline    taken from the Delphi reproduction or the E10 runs
  n/a         the plan lists it, but the implementation has no such component;
              it is reported as absent, never filled with a stand-in
  pending     the input file does not exist yet

Rows (experiment plan §7 E7, plus the client-side decisions the paper claims):

  wo_client_aware  server-time objective instead of the client objective
  wo_planner       fixed all-P2 / fixed all-hybrid instead of the planner
  wo_stream_keys   one-shot M0 instead of frame-by-frame evaluation keys
  wo_share_decrypt naive HE<->share conversion instead of decrypt-as-share
  wo_fusion        P2 not folded into the next linear layer
  wo_packing       one ciphertext per value instead of SIMD packing
  gc_route         Delphi's GC ReLU instead of the OT ReLU
  mask_reuse       one mask across queries (violates M1)
  wo_offline       n/a: weight encoding still runs online
  wo_rand_trunc    n/a: the share domain has no fixed-point product

Inputs: train_mid.py and eval_deep_accuracy.py on model B (one run per schedule),
measure_client_memory.py --modes oneshot,streamed, E4, E10, results/E_delphi.

    python analysis/gen_e7_table8.py
    python analysis/gen_e7_table8.py --acc-floor 0.82 --rtt-ms 50 --bandwidth-mbps 50
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import planner as P
from cost_model import fit
from data_manifest import git_sha
from pi_proto import make_backend

ROOT = Path(__file__).resolve().parent.parent
R_H = 6.025                  # end-to-end, Table 7-12
MINIONN_RELUS = 173_056      # 7 layers of rust/experiments/src/minionn.rs
MODEL_B_LAYERS = 4           # activations of model B
MiB = 1024 * 1024


def row(name, status, full, ablated, effect, source):
    return {"ablation": name, "status": status, "full": full, "ablated": ablated,
            "effect": effect, "source": source}


def load_schedules(root: Path) -> dict:
    """Mean per-schedule figures of model B from the real ciphertext runs."""
    base = root / "results" / "raw_results" / "E1" / "cifar10" / "mid"
    out = {}
    for path in sorted(glob.glob(str(base / "s*_cipher*" / "seed*.json"))):
        tag = re.search(r"s(\d+)_cipher", path).group(1)
        rec = json.loads(Path(path).read_text(encoding="utf-8"))
        out.setdefault(tag, []).append(rec)
    table = {}
    for tag, recs in out.items():
        recs = [rec for rec in recs if rec.get("rows")]
        if not recs:
            continue
        rows = [r for rec in recs for r in rec["rows"]]
        accs = [json.loads(Path(p).read_text(encoding="utf-8"))["test_acc"]
                for p in glob.glob(str(base / f"s{tag}" / "seed*.json"))]
        t = lambda k: float(np.mean([r["timings"][k] for r in rows]))
        hybrid = sum(int(b) for b in tag)
        table[tag] = {
            "accuracy": float(np.mean(accs)) if accs else None,
            "agreement": float(np.mean([rec["agreement"] for rec in recs])),
            "K_h": hybrid,
            "rounds": 2 + hybrid * R_H,
            "client_s": t("encrypt_input_s") + t("decrypt_s") + t("reencrypt_s") + t("mpc_s"),
            "server_s": float(np.mean([r["server"]["forward_s"] for r in rows])),
            "bytes": float(np.mean([r["up_bytes"] + r["down_bytes"] for r in rows])),
            "sci_bytes": float(np.mean([r["up_sci_bytes"] + r["down_sci_bytes"] for r in rows])),
            "elements": int(rows[0]["timings"]["mpc_elements"]),
        }
    return table


def planner_rows(table: dict, acc_floor: float, rtt_ms: float, mbps: float) -> list[dict]:
    """The planner's two objectives over the real runs of model B."""
    if not table:
        return [row(n, "pending", None, None, None, "eval_deep_accuracy on model B")
                for n in ("wo_client_aware", "wo_planner")]

    ok = {t: s for t, s in table.items()
          if s["accuracy"] is not None and s["accuracy"] >= acc_floor
          and P.required_depth(tuple(int(b) for b in t), P.MODEL_B_LAYERS) <= P.Profile(None).depth_budget}
    if not ok:
        return [row(n, "pending", None, None,
                    f"no deployable schedule reaches the floor {acc_floor}", "train_mid")
                for n in ("wo_client_aware", "wo_planner")]
    profile = P.Profile(fit(ROOT), accuracies={t: s["accuracy"] for t, s in ok.items()},
                        rtt_ms=rtt_ms, bandwidth_mbps=mbps, arch="conv3",
                        input_blocks=P.MODEL_B_INPUT_BLOCKS)
    server_side = P.Profile(**{**profile.__dict__, "objective": "server"})
    full = P.constrained(P.MODEL_B_LAYERS, profile, acc_floor).key()
    server = P.constrained(P.MODEL_B_LAYERS, server_side, acc_floor).key()
    a, b = ok[full], ok[server]
    rows = [row("wo_client_aware", "measured", {"schedule": full, **a},
                {"schedule": server, **b},
                {"client_s_x": b["client_s"] / a["client_s"],
                 "bytes_x": b["bytes"] / a["bytes"],
                 "rounds_delta": b["rounds"] - a["rounds"],
                 "server_s_x": b["server_s"] / a["server_s"]},
                "real ciphertext runs of model B")]
    hybrid_all = "1" * MODEL_B_LAYERS
    fixed = table.get(hybrid_all)
    rows.append(row(
        "wo_planner", "measured" if fixed else "pending", {"schedule": full, **a},
        {"all_p2": f"{'0' * MODEL_B_LAYERS} is not deployable: depth "
                   f"{P.required_depth((0,) * MODEL_B_LAYERS, P.MODEL_B_LAYERS)}",
         "all_hybrid": {"schedule": hybrid_all, **fixed} if fixed else None},
        ({"accuracy_delta": fixed["accuracy"] - a["accuracy"]
          if fixed["accuracy"] is not None else None,
          "bytes_x": fixed["bytes"] / a["bytes"],
          "rounds_delta": fixed["rounds"] - a["rounds"]} if fixed else None),
        "real ciphertext runs of model B" if fixed
        else f"pending: no ciphertext run of s{hybrid_all} yet"))
    return rows


def memory_row(path: Path) -> dict:
    if not path.is_file():
        return row("wo_stream_keys", "pending", None, None, None, str(path))
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    vals = lambda mode: [r["client_peak_rss_mb"] for r in rows if r["mode"] == mode]
    streamed, oneshot = vals("streamed"), vals("oneshot")
    if not streamed or not oneshot:
        missing = "oneshot" if not oneshot else "streamed"
        return row("wo_stream_keys", "pending",
                   {"client_peak_mb": float(np.mean(streamed)) if streamed else None},
                   None, None,
                   f"measure_client_memory.py --modes oneshot,streamed ({missing} missing)")
    full, off = float(np.mean(streamed)), float(np.mean(oneshot))
    return row("wo_stream_keys", "measured", {"client_peak_mb": full},
               {"client_peak_mb": off}, {"memory_x": off / full},
               "measure_client_memory.py, client process only")


def segment_depth(bits, fold: bool) -> int:
    """Largest depth between resets of model B; P2 costs one more if not folded."""
    layers = tuple(P.Layer(l.name, l.elements, p2_depth=1 if fold else 2)
                   for l in P.MODEL_B_LAYERS)
    return P.required_depth(bits, layers)


def measured_budget() -> int:
    """Rescales the CKKS chain really allows."""
    be = make_backend("seal")
    ct = be.encrypt(be.new_context(), np.ones(16))
    count = 0
    try:
        for _ in range(16):
            ct = be.mul_plain(ct, np.ones(16))
            be.decrypt(ct)
            count += 1
    except Exception:
        pass
    return count


def fusion_row() -> dict:
    budget = measured_budget()
    grid = [tuple(int(b) for b in f"{i:0{MODEL_B_LAYERS}b}") for i in range(2 ** MODEL_B_LAYERS)]
    count = lambda fold: sum(segment_depth(s, fold) <= budget for s in grid)
    main = (0, 1, 0, 0)
    return row("wo_fusion", "derived",
               {"deployable_schedules": count(True), "depth_s0100": segment_depth(main, True)},
               {"deployable_schedules": count(False), "depth_s0100": segment_depth(main, False)},
               {"budget_measured": budget, "of_schedules": len(grid)},
               "depth count against the measured SEAL chain")


def packing_row(root: Path) -> dict:
    path = root / "results" / "raw_results" / "E4" / "all" / "wire_bytes" / "seed73.csv"
    if not path.is_file():
        return row("wo_packing", "pending", None, None, None, str(path))
    with path.open(encoding="utf-8") as fh:
        ct = max(int(r["bytes"]) for r in csv.DictReader(fh)
                 if r["profile"] == "baseline" and r["message"] == "M1")
    slots = 6912                       # im2col input of model B
    return row("wo_packing", "derived", {"ciphertexts": 2, "bytes": 2 * ct},
               {"ciphertexts": slots, "bytes": slots * ct}, {"bytes_x": slots / 2},
               "measured M1 size times the ciphertext count")


def share_row(root: Path) -> dict:
    path = root / "results" / "raw_results" / "E4" / "all" / "wire_bytes" / "seed73.csv"
    if not path.is_file():
        return row("wo_share_decrypt", "pending", None, None, None, str(path))
    with path.open(encoding="utf-8") as fh:
        ct = max(int(r["bytes"]) for r in csv.DictReader(fh)
                 if r["profile"] == "baseline" and r["message"] == "M1")
    return row("wo_share_decrypt", "derived", {"per_layer_extra": 0},
               {"per_layer_extra_bytes": ct, "per_layer_extra_rounds": 1,
                "per_layer_extra_encryptions": 1}, None,
               "protocol count times the measured fresh ciphertext")


def _client_cpu(root: Path, layers: int) -> float | None:
    vals = []
    for f in glob.glob(str(root / "results" / "E_delphi" / "logs" / f"client_L{layers}_r*.time")):
        text = Path(f).read_text(encoding="utf-8", errors="replace")
        vals.append(float(re.search(r"User time \(seconds\): ([\d.]+)", text).group(1))
                    + float(re.search(r"System time \(seconds\): ([\d.]+)", text).group(1)))
    return float(np.mean(vals)) if vals else None


def gc_row(root: Path, table: dict) -> dict:
    path = root / "results" / "E_delphi" / "delphi_minionn_summary.csv"
    with path.open(encoding="utf-8") as fh:
        rows = {int(r["layers"]): r for r in csv.DictReader(fh)}
    comm = lambda r, k: float(r[k]) * 1024
    off = (comm(rows[0], "client_comm_off_KB") - comm(rows[7], "client_comm_off_KB")) / MINIONN_RELUS
    on = (comm(rows[0], "client_comm_on_KB") - comm(rows[7], "client_comm_on_KB")) / MINIONN_RELUS
    cpu = [_client_cpu(root, 0), _client_cpu(root, 7)]
    gc = {"bytes_per_relu_offline": off, "bytes_per_relu_online": on,
          "client_cpu_s_all_relu": cpu[0], "client_cpu_s_all_poly": cpu[1]}
    sci = [s["sci_bytes"] / s["elements"] for s in table.values() if s["elements"]]
    if not sci:
        return row("gc_route", "pending", None, gc, None, "needs model B SCI bytes")
    ot = float(np.mean(sci))
    return row("gc_route", "baseline", {"bytes_per_relu": ot}, gc,
               {"total_bytes_x": (off + on) / ot, "online_bytes_x": on / ot},
               "Delphi reproduction (L0 minus L7) against real SCI bytes")


def mask_row(root: Path) -> dict:
    path = root / "results" / "raw_results" / "E10" / "all" / "summary_5seeds.json"
    if not path.is_file():
        return row("mask_reuse", "pending", None, None, None, str(path))
    data = json.loads(path.read_text(encoding="utf-8"))["datasets"]
    return row("mask_reuse", "measured",
               {d: v["mask_fresh_layer1"]["single_query_sign_accuracy"] for d, v in data.items()},
               {d: v["mask_reuse_layer1"]["mean"] for d, v in data.items()}, None,
               "E10a: fresh sign recovery against the reuse residual")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--acc-floor", type=float, default=0.82)
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--bandwidth-mbps", type=float, default=50.0)
    ap.add_argument("--memory-summary", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    table = load_schedules(args.root)
    memory = args.memory_summary or (args.root / "results" / "raw_results" / "E2"
                                     / "client_only" / "desktop" / "seed73" / "summary.json")
    rows = [
        *planner_rows(table, args.acc_floor, args.rtt_ms, args.bandwidth_mbps),
        memory_row(memory),
        share_row(args.root),
        fusion_row(),
        packing_row(args.root),
        gc_row(args.root, table),
        mask_row(args.root),
        row("wo_offline", "n/a", None, None, None,
            "LinearEval encodes the weights online; the encoding cache is not built"),
        row("wo_rand_trunc", "n/a", None, None, None,
            "SCI ReLU and the fallback only compare and select; no truncation exists"),
    ]
    out = args.out or args.root / "results" / "E7"
    out.mkdir(parents=True, exist_ok=True)
    (out / "e7_table8.json").write_text(json.dumps({
        "exp_id": "E7", "git_sha": git_sha(args.root), "acc_floor": args.acc_floor, "rtt_ms": args.rtt_ms,
        "bandwidth_mbps": args.bandwidth_mbps, "r_h": R_H,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schedules": table, "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for r in rows:
        print(f"{r['ablation']:<18}{r['status']:<10}{json.dumps(r['effect'], ensure_ascii=False)}")
        print(f"    full    {json.dumps(r['full'], ensure_ascii=False)[:150]}")
        print(f"    ablated {json.dumps(r['ablated'], ensure_ascii=False)[:150]}")
    print(f"-> {out / 'e7_table8.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
