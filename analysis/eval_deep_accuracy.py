"""Ciphertext accuracy of a deep model over the real protocol (E1, deep rows).

Starts pi_server.py --serve as its own process and runs one client query per
test image, each with a fresh session.  Compares the ciphertext argmax with
the plaintext forward of the same weights (the agreement the canonical E1
reports) and with the label.

    python analysis/eval_deep_accuracy.py --schedule 0100 --seed 73 --samples 64
    python analysis/eval_deep_accuracy.py --schedule 1110 --mask-exact --threads 8
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import pi_client
from deep_net import from_npz
from train_deep_conv import CIFAR_MEAN, CIFAR_STD, load_module

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--schedule", required=True, help="e.g. 0100")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--samples", type=int, default=64)
    ap.add_argument("--sample-seed", type=int, default=20261030)
    ap.add_argument("--port", type=int, default=9310)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--mask-exact", action="store_true")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--timeout", type=float, default=3600.0)
    args = ap.parse_args()

    tag = f"mid_s{args.schedule}_seed{args.seed}"
    weights = args.root / "datas" / "weights" / f"{tag}.npz"
    domains = args.root / "datas" / "mask_domains" / f"{tag}.json"
    schedule = tuple(int(b) for b in args.schedule)
    model = from_npz(weights, schedule)
    module = load_module(args.root / "model" / "cifar10_cnn_relu.py")
    _, test_loader = module.load_data(256, 0)
    test = test_loader.dataset
    order = np.random.default_rng(args.sample_seed).permutation(len(test))[:args.samples]

    out_dir = (args.root / "results" / "raw_results" / "E1" / "cifar10" / "mid"
               / f"s{args.schedule}_cipher{'_maskv3' if args.mask_exact else ''}")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / f"seed{args.seed}.server.log"
    cmd = [sys.executable, "-u", str(HERE / "pi_server.py"), "--deep-weights", str(weights),
           "--schedule", ",".join(args.schedule), "--mask-domains", str(domains),
           "--port", str(args.port), "--backend", "seal", "--wire", args.wire,
           "--threads", str(args.threads), "--serve"]
    if args.mask_exact:
        cmd.append("--mask-exact")
    with log.open("w") as fh:
        server = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    rows = []
    try:
        deadline = time.time() + 300
        while "listening on" not in log.read_text(errors="replace"):
            if server.poll() is not None or time.time() > deadline:
                raise SystemExit(f"server did not start, see {log}")
            time.sleep(0.5)
        mean = np.array(CIFAR_MEAN).reshape(3, 1, 1)
        std = np.array(CIFAR_STD).reshape(3, 1, 1)
        for n, idx in enumerate(order):
            img, label = test[int(idx)]
            raw = img.numpy().astype(np.float64)
            ref, _ = model.plain_forward((raw - mean) / std)
            t0 = time.perf_counter()
            res = pi_client.run_client("127.0.0.1", args.port, raw.reshape(-1), "seal",
                                       timeout=args.timeout, stream_keys=True)
            got = np.asarray(res["logits"])
            rows.append({
                "index": int(idx), "label": int(label),
                "pred": int(np.argmax(got)), "plain_pred": int(np.argmax(ref)),
                "max_abs_err": float(np.max(np.abs(got - ref))),
                "wall_s": time.perf_counter() - t0,
                "up_bytes": int(res["online_up_bytes"]),
                "down_bytes": int(res["online_down_bytes"]),
                "up_sci_bytes": int(res["sci_bytes_sent"]),
                "down_sci_bytes": int(res["sci_bytes_received"]),
                "timings": res["timings"],
                "server": {k: res["server"]["timings"].get(k) for k in
                           ("forward_s", "linear_s", "hybrid_relu_s", "mpc_s")},
            })
            print(f"[{n + 1}/{len(order)}] label {label} cipher {rows[-1]['pred']} "
                  f"plain {rows[-1]['plain_pred']} err {rows[-1]['max_abs_err']:.3e} "
                  f"{rows[-1]['wall_s']:.1f}s", flush=True)
    finally:
        server.terminate()
    agree = float(np.mean([r["pred"] == r["plain_pred"] for r in rows]))
    acc = float(np.mean([r["pred"] == r["label"] for r in rows]))
    (out_dir / f"seed{args.seed}.json").write_text(json.dumps({
        "exp_id": "E1", "dataset": "cifar10", "method": "mid_ciphertext",
        "schedule": args.schedule, "seed": args.seed, "samples": len(rows),
        "sample_seed": args.sample_seed, "threads": args.threads,
        "mask_exact": bool(args.mask_exact), "wire": args.wire,
        "agreement": agree, "accuracy": acc,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"agreement {agree:.4f}  accuracy {acc:.4f}  ({len(rows)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
