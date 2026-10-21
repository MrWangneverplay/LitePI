"""Acceptance check for the slim secure-ReLU circuit.

  1. Base 1-of-2 OT count per ReLU at k_ring=20, k_compare=14, f=8, measured on the
     REAL OT path (every call to the real base OT is counted), for both the
     in-process version (mpc_relu_fxp.secure_relu_fxp) and the two-party networked
     version (secure_relu_net, counted at sender_step1 = one per OT).
  2. Correctness against plaintext ReLU on 20000 random inputs (positive, negative,
     zero, the +-31 boundary, random shares), both at fixed-point level
     (secure_relu_fxp) and at exact ring level (secure_relu).  By default a
     functional stand-in replaces the OT so 20000 runs finish in seconds (the
     circuit logic is identical); pass --real-ot to run every one of the 20000
     through the real P-256 base OT (about 17 ms per ReLU with OpenSSL).

Run:
    python analysis/prototypes/verify_slim_relu.py                 # stand-in OT
    python analysis/prototypes/verify_slim_relu.py --real-ot       # real OT, slower
    python analysis/prototypes/verify_slim_relu.py --count-only --analysis-dir <old analysis/>
        (e.g. a checkout of the pre-slim version, to measure the 'before' count)
"""

from __future__ import annotations

import argparse
import socket
import sys
import threading
from pathlib import Path

import numpy as np

K_RING, K_COMPARE, F = 20, 14, 8


def run(analysis_dir: Path, n_random: int, n_ot_samples: int,
        use_real_ot: bool = False, count_only: bool = False) -> int:
    sys.path.insert(0, str(analysis_dir))
    import gmw_core
    import mpc_relu_secure as M
    import mpc_relu_fxp as X
    import secure_relu_net as N

    rng = np.random.default_rng(73)
    fail = 0

    # ------------------------------------------------ 1a. OT count, in-process
    real_ot = gmw_core._ot_1of2_inproc
    calls = [0]

    def counting_ot(m0, m1, c):
        calls[0] += 1
        return real_ot(m0, m1, c)

    M._ot_1of2_inproc = counting_ot
    gmw_core._ot_1of2_inproc = counting_ot
    z = rng.uniform(-8.0, 8.0, size=n_ot_samples)
    s_s = rng.uniform(-8.0, 8.0, size=z.shape)
    X.secure_relu_fxp(z - s_s, s_s, k_ring=K_RING, f=F, k_compare=K_COMPARE, rng=rng)
    per_relu = calls[0] / n_ot_samples
    print(f"[OT count, in-process ] k_ring={K_RING} k_compare={K_COMPARE} f={F}: "
          f"{calls[0]} OTs / {n_ot_samples} ReLUs = {per_relu:g} per ReLU")
    M._ot_1of2_inproc = real_ot
    gmw_core._ot_1of2_inproc = real_ot

    # ------------------------------------------------ 1b. OT count, networked
    real_sender = N.sender_step1
    sender_calls = [0]

    def counting_sender():
        sender_calls[0] += 1
        return real_sender()

    N.sender_step1 = counting_sender
    for zv in (3.0, -2.0):
        s_sv = 1.25
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        holder = {}

        def server_side():
            conn, _ = srv.accept()
            with conn:
                holder["o0"] = N.run_party(N.Ctx(0, N.Chan(conn)), zv - s_sv, K_RING, F, K_COMPARE)

        th = threading.Thread(target=server_side, daemon=True)
        th.start()
        o1 = N.connect("127.0.0.1", port, s_sv, K_RING, F, K_COMPARE)
        th.join()
        srv.close()
        got = holder["o0"] + o1
        ok = abs(got - max(zv, 0.0)) <= 3.0 / 2 ** F
        fail += not ok
        print(f"[networked ReLU z={zv:+.1f}] got={got:.4f} ref={max(zv, 0.0):.4f} "
              f"{'ok' if ok else 'WRONG'}")
    n_net = sender_calls[0] / 2
    print(f"[OT count, networked  ] {sender_calls[0]} OTs / 2 ReLUs = {n_net:g} per ReLU")
    N.sender_step1 = real_sender

    if count_only:
        print("\nRESULT:", "PASS" if fail == 0 else "FAIL", "(count-only run)")
        return 0 if fail == 0 else 1

    # ------------------------------------------------ 2. correctness, 20000 inputs
    if not use_real_ot:
        def fake_ot(m0, m1, c):
            return m1 if c else m0

        M._ot_1of2_inproc = fake_ot
        gmw_core._ot_1of2_inproc = fake_ot
    print(f"[correctness] OT backend: {'REAL P-256 base OT' if use_real_ot else 'functional stand-in'}")

    # domain contract: |z| < 2^(k_compare-1-f) = 32, so +-31 is the tested boundary
    special = np.array([0.0, 2.0 ** -F, -(2.0 ** -F), 31.0, -31.0,
                        1e-3, -1e-3, 7.99, -7.99])
    zs = np.concatenate([
        special,
        rng.uniform(0.0, 31.0, size=n_random // 4),                  # positives
        rng.uniform(-31.0, 0.0, size=n_random // 4),                 # negatives
        np.zeros(n_random // 20),                                    # zero
        rng.choice([31.0, -31.0], size=n_random // 20),              # boundary
    ])
    zs = np.concatenate([zs, rng.uniform(-31.0, 31.0, size=n_random - zs.size)])[:n_random]
    tol = 3.0 / 2 ** F
    bad = 0
    worst = 0.0
    for zv in zs:
        s_sv = rng.uniform(-8.0, 8.0)                                # random share
        c_c, c_s = X.secure_relu_fxp(np.array([zv - s_sv]), np.array([s_sv]),
                                     k_ring=K_RING, f=F, k_compare=K_COMPARE, rng=rng)
        err = abs(float(c_c[0] + c_s[0]) - max(zv, 0.0))
        worst = max(worst, err)
        bad += err > tol
    print(f"[correctness, fixed-point] {zs.size} inputs, wrong={bad}, "
          f"worst abs err={worst:.4f} (tol {tol:.4f})")
    fail += bad > 0

    # ring level, exact: every sign pattern incl. window edges, random ring shares
    half = 1 << (K_COMPARE - 1)
    ring_vals = np.concatenate([[0, 1, -1, half - 1, -half],
                                rng.integers(-half, half, size=n_random - 5)])
    ring_bad = 0
    for zi in ring_vals:
        zi = int(zi)
        s_c = int(rng.integers(0, 1 << K_RING))
        s_s2 = ((zi % (1 << K_RING)) - s_c) % (1 << K_RING)
        c_c, c_s = M.secure_relu(s_c, s_s2, K_RING, K_COMPARE)
        ring_bad += ((c_c + c_s) % (1 << K_RING)) != max(zi, 0)
    print(f"[correctness, ring exact ] {ring_vals.size} inputs, wrong={ring_bad}")
    fail += ring_bad > 0

    print("\nRESULT:", "PASS" if fail == 0 else "FAIL")
    return 0 if fail == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify the slim secure-ReLU circuit")
    ap.add_argument("--analysis-dir", default=str(Path(__file__).resolve().parent.parent))
    ap.add_argument("--n-random", type=int, default=20000)
    ap.add_argument("--n-ot-samples", type=int, default=20)
    ap.add_argument("--real-ot", action="store_true",
                    help="run the correctness inputs through the real base OT (slow)")
    ap.add_argument("--count-only", action="store_true",
                    help="only measure OTs per ReLU (use on an old checkout for 'before')")
    args = ap.parse_args()
    return run(Path(args.analysis_dir), args.n_random, args.n_ot_samples,
               args.real_ot, args.count_only)


if __name__ == "__main__":
    raise SystemExit(main())
