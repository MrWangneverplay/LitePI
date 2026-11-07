"""Deep-network protocol path on real SEAL (synthetic weights).

Runs the real server and client (pi_server.serve_connection, pi_client.run_client)
with deep_net.DeepNet and compares the logits with the numpy forward:

  coo    linear_coo + linear_sparse against numpy (conv with stride/padding,
         pool + conv, pool + FC): the P2 -> linear primitive on its own
  mini   3x8x8, conv-act-conv-act-pool-fc-act-fc: every schedule the depth
         budget allows, so hybrid->conv, P2->conv, hybrid->pool->FC and
         P2->pool->FC all run
  mid    alignment model B of the plan (conv3, 3x32x32; first hybrid layer spans
         two ciphertexts); slow, one schedule by default

Test-only substitution: the in-process FakeSci of check_wire_canonical.py
stands in for SCI.  This script produces no paper numbers.

    python analysis/check_deep_net.py --parts coo,mini
    python analysis/check_deep_net.py --parts mid --mid-schedules 0100,1110 --threads 8
    python analysis/check_deep_net.py --parts mini --mask-exact
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import numpy as np

import check_wire_canonical as W
import pi_client
import pi_server
from deep_net import SLOTS, DeepNet, _conv_out_shape, linear_coo, mid_ops
from mask_session import MaskDomain
from pi_proto import make_backend


def mini_ops(rng) -> list[dict]:
    conv = lambda co, ci, s: {"op": "conv", "stride": s, "pad": 1,
                              "weight": rng.normal(0, 1 / np.sqrt(ci * 9), (co, ci, 3, 3)),
                              "bias": rng.normal(0, 0.05, co)}
    fc = lambda o, i: {"op": "fc", "weight": rng.normal(0, 1 / np.sqrt(i), (o, i)),
                       "bias": rng.normal(0, 0.05, o)}
    return [conv(4, 3, 1), {"op": "act"}, conv(8, 4, 2), {"op": "act"},
            {"op": "pool", "k": 2}, fc(16, 32), {"op": "act"}, fc(10, 16)]


def check_coo(threads: int) -> bool:
    """The sparse matrix must equal the plaintext layers, then run on SEAL."""
    rng = np.random.default_rng(1)
    be = make_backend("seal")
    ctx = be.new_context()
    ok = True
    cases = {
        "conv s1": ([{"op": "conv", "stride": 1, "pad": 1,
                      "weight": rng.normal(0, 0.3, (4, 3, 3, 3)),
                      "bias": rng.normal(0, 0.1, 4)}], (3, 8, 8)),
        "conv s2": ([{"op": "conv", "stride": 2, "pad": 1,
                      "weight": rng.normal(0, 0.3, (5, 3, 3, 3)),
                      "bias": rng.normal(0, 0.1, 5)}], (3, 16, 16)),
        "pool+conv": ([{"op": "pool", "k": 2},
                       {"op": "conv", "stride": 1, "pad": 1,
                        "weight": rng.normal(0, 0.3, (2, 3, 3, 3)),
                        "bias": rng.normal(0, 0.1, 2)}], (3, 8, 8)),
        "pool+fc": ([{"op": "pool", "k": 2},
                     {"op": "fc", "weight": rng.normal(0, 0.1, (10, 48)),
                      "bias": rng.normal(0, 0.1, 10)}], (3, 8, 8)),
    }
    for name, (chain, shape) in cases.items():
        x = rng.normal(0, 1, shape)
        ii, oi, val, bias, out_shape = linear_coo(chain, shape)
        got = np.bincount(oi, weights=val * x.reshape(-1)[ii],
                          minlength=int(np.prod(out_shape))) + bias
        net = DeepNet([{"op": "conv", "stride": 1, "pad": 0,
                        "weight": np.ones((1, shape[0], 1, 1)), "bias": np.zeros(1)},
                       {"op": "act"}, *chain], shape, (1,))
        ref = x
        for op in chain:                       # plaintext layers of DeepNet
            ref = _apply(ref, op)
        err = float(np.max(np.abs(got - ref.reshape(-1))))
        ok &= err < 1e-9
        ct = be.encrypt(ctx, x.reshape(-1))
        outs = ctx.linear_sparse([ct], ii, oi, val, got.size, threads)
        dec = np.concatenate([be.decrypt(c) for c in outs]) + bias
        he_err = float(np.max(np.abs(dec - got)))
        ok &= he_err < 2e-2
        print(f"{name:>10}: COO vs numpy {err:.1e}, ciphertext {he_err:.1e} "
              f"({ii.size} entries)  {'PASS' if err < 1e-9 and he_err < 2e-2 else 'FAIL'}")
        del net
    return ok


def _apply(v, op):
    from deep_hybrid import im2col_rows
    from deep_net import _pool
    if op["op"] == "pool":
        return _pool(v, op["k"])
    if op["op"] == "fc":
        return op["weight"] @ v.reshape(-1) + op["bias"]
    rows, oh, ow = im2col_rows(v, op["weight"].shape[-1], op["stride"], op["pad"])
    out = op["weight"].reshape(op["weight"].shape[0], -1) @ rows + op["bias"][:, None]
    return out.reshape(op["weight"].shape[0], oh, ow)


def run_net(net: DeepNet, x: np.ndarray, mask_exact: bool, profile: str = "baseline") -> dict:
    ref, zs = net.plain_forward(x)
    domains = {k: MaskDomain.from_observed(float(z.min()), float(z.max()))
               for k, z in enumerate(zs)}
    domains[net.output_layer_id()] = MaskDomain.from_observed(float(ref.min()),
                                                              float(ref.max()))
    be_srv = make_backend("seal")
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    holder: dict = {}

    def server_side():
        try:
            conn, _ = srv.accept()
            with conn:
                holder["stats"] = pi_server.serve_connection(
                    conn, be_srv, net, timeout=1800, mask_domains=domains,
                    wire_profile=W.WIRE_PROFILES[profile], mask_exact=mask_exact)
        except BaseException as exc:
            holder["error"] = exc

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        res = pi_client.run_client("127.0.0.1", port, x.reshape(-1), "seal", timeout=1800)
    finally:
        t.join(120)
        srv.close()
    if "error" in holder:
        raise holder["error"]
    got = np.asarray(res["logits"])
    top2 = np.sort(np.asarray(ref).ravel())[::-1][:2]
    return {"err": float(np.max(np.abs(got - ref))),
            "ref": float(np.max(np.abs(ref))),
            # synthetic random weights sometimes put the top-2 logits within the
            # ciphertext error, so an argmax flip there is a tie, not a bug.
            "margin": float(top2[0] - top2[1]),
            "argmax_ok": int(np.argmax(got)) == int(np.argmax(ref)),
            "wall_s": time.perf_counter() - t0,
            "up": res["online_up_bytes"], "down": res["online_down_bytes"],
            "stats": holder["stats"]}


def check_net(name, make, schedules, shape, mask_exact, threads, profile, tol) -> bool:
    ok = True
    for sch in schedules:
        rng = np.random.default_rng(7)
        net = DeepNet(make(rng), shape, [int(b) for b in sch], threads=threads)
        x = np.random.default_rng(8).normal(0, 1, shape)
        r = run_net(net, x, mask_exact, profile)
        passed = r["err"] <= tol and (r["argmax_ok"] or r["margin"] <= r["err"])
        ok &= passed
        print(f"{name} s{sch}: err={r['err']:.3e} (|ref|<={r['ref']:.2f}, "
              f"top-2 margin {r['margin']:.1e})  "
              f"online up {r['up'] / 1e6:7.2f} MB down {r['down'] / 1e6:6.2f} MB  "
              f"{r['wall_s']:6.1f}s  {'PASS' if passed else 'FAIL'}"
              f"{'' if r['argmax_ok'] else ' (argmax flip within margin)'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--parts", default="coo,mini")
    ap.add_argument("--mini-schedules", default="001,010,011,100,101,110,111")
    ap.add_argument("--mid-schedules", default="0100")
    ap.add_argument("--mask-exact", action="store_true")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.1)
    args = ap.parse_args()
    pi_server.SCI_AVAILABLE = True
    pi_server.SciReLUSession = W.FakeSci
    pi_client.SCI_AVAILABLE = True
    pi_client.SciReLUSession = W.FakeSci
    parts = args.parts.split(",")
    ok = True
    if "coo" in parts:
        ok &= check_coo(args.threads)
    if "mini" in parts:
        ok &= check_net("mini", mini_ops, args.mini_schedules.split(","), (3, 8, 8),
                        args.mask_exact, args.threads, args.wire, args.tol)
    if "mid" in parts:
        ok &= check_net("mid", mid_ops, args.mid_schedules.split(","), (3, 32, 32),
                        args.mask_exact, args.threads, args.wire, args.tol)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
