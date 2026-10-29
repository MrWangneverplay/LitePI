"""Feasibility of the deep-network ciphertext path (fast net conv2 / conv3).

For each convolution that follows a hybrid layer, this runs the client and
server halves of deep_hybrid.py on the real SEAL backend with additive shares
of the true activation (client share of SCI magnitude, +/- 2^(K_RING-F-1)),
and reports correctness against numpy plus the costs the paper needs:
blocks (= ciphertexts uploaded per layer), client encryption time, upload
bytes per wire profile, server convolution time, and levels consumed.

Weights: datas/weights/fast_relu_*.npz exported by train_deep_conv.py when
present (--weights), otherwise random weights of the same geometry.  The
input is a random CIFAR-shaped image pushed through conv1 in plaintext so the
activation statistics are realistic when real weights are used.

    python analysis/check_deep_hybrid.py
    python analysis/check_deep_hybrid.py --weights datas/weights/fast_relu_rgb_bn_seed73.npz
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from deep_hybrid import client_encrypt_share, conv_reference, server_conv
from params_pi import F, K_RING
from pi_proto import WIRE_PROFILES, make_backend

ROOT = Path(__file__).resolve().parent.parent
# FastConvNet (train_deep_conv.py): conv k=3, p=1; strides 2,2,1,1; C=32 -> 64.
FAST = [(3, 32, 2), (32, 64, 2), (64, 64, 1), (64, 64, 1)]


def load_weights(path: Path | None, seed: int):
    if path and path.exists():
        z = np.load(path)
        return [(z[f"conv{i}_weight"].astype(np.float64),
                 z[f"conv{i}_bias"].astype(np.float64)) for i in range(4)], str(path)
    rng = np.random.default_rng(seed)
    ws = []
    for cin, cout, _ in FAST:
        ws.append((rng.normal(0, 1 / np.sqrt(cin * 9), (cout, cin, 3, 3)),
                   rng.normal(0, 0.05, cout)))
    return ws, "random (same geometry)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--layers", default="1,2", help="conv indices (0-based): conv2,conv3")
    ap.add_argument("--profiles", default="baseline,compact")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "checks" / "deep_hybrid_feasibility.json")
    args = ap.parse_args()

    weights, wsrc = load_weights(args.weights, args.seed)
    rng = np.random.default_rng(args.seed)
    x = rng.normal(0, 1, (3, 32, 32))
    acts = [x]
    for (w, b), (_, _, s) in zip(weights, FAST):        # plaintext activations
        acts.append(np.maximum(conv_reference(acts[-1], w, b, s, 1), 0))

    share_mag = 2.0 ** (K_RING - F - 1)
    rows = []
    for prof in args.profiles.split(","):
        be = make_backend("seal", WIRE_PROFILES[prof])
        ctx = be.new_context()
        be.ctx = ctx
        for li in (int(v) for v in args.layers.split(",")):
            w, b = weights[li]
            stride = FAST[li][2]
            a = acts[li]                                  # input of conv li (post-ReLU)
            c_c = np.round(rng.uniform(-share_mag, share_mag, a.shape) * (1 << F)) / (1 << F)
            c_s = a - c_c                                 # additive shares of a
            t0 = time.perf_counter()
            blobs, windows = client_encrypt_share(be, ctx, c_c, 3, stride, 1)
            t_client = time.perf_counter() - t0
            level_in = ctx.index(be.deserialize(ctx, blobs[0]))
            t0 = time.perf_counter()
            outs, _ = server_conv(be, ctx, blobs, c_s, w, b, stride, 1)
            t_server = time.perf_counter() - t0
            got = np.concatenate([be.decrypt(ct) for ct in outs])
            ref = conv_reference(a, w, b, stride, 1).reshape(-1)
            err = float(np.max(np.abs(got[:ref.size] - ref)))
            row = {
                "profile": prof, "conv": li + 1,
                "in_shape": list(a.shape), "out_channels": int(w.shape[0]),
                "stride": stride, "windows": int(windows),
                "taps": int(w[0].size), "blocks_uploaded": len(blobs),
                "upload_bytes": int(sum(len(bl) for bl in blobs)),
                "client_encrypt_s": round(t_client, 4),
                "server_conv_s": round(t_server, 4),
                "levels_used": int(level_in - ctx.index(outs[0])),
                "max_abs_err": err, "ref_max_abs": float(np.max(np.abs(ref))),
                "share_magnitude": share_mag,
                "pass": err < 5e-2 * max(1.0, float(np.max(np.abs(ref)))),
            }
            rows.append(row)
            print(f"{prof:>8} conv{li+1}: {row['taps']} taps x {windows} windows -> "
                  f"{len(blobs)} cts ({row['upload_bytes']/2**20:.2f} MiB up)  "
                  f"client {t_client:.3f}s  server {t_server:.2f}s  "
                  f"err={err:.2e} (|ref|<={row['ref_max_abs']:.2f})  "
                  f"levels={row['levels_used']}  {'PASS' if row['pass'] else 'FAIL'}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "check": "deep-network hybrid conv feasibility (client im2col re-encoding)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "network": "FastConvNet (train_deep_conv.py)", "weights": wsrc,
        "scope": "convs preceded by a hybrid layer; P2->conv needs HE re-layout (not implemented)",
        "machine": "measurement host of this run (not the paper testbed)",
        "rows": rows,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"written -> {args.out}")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
