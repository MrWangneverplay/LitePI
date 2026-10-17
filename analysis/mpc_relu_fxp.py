"""Fixed-point bridge: real secure ReLU on float additive shares.

Route A (CKKS real shares) with STATISTICAL masking.

The earlier version drew the output mask rho from [0, 2^(k-1)) where k was also
the value bit width, so rho and ReLU(z) had the SAME range and the server's
share b = ReLU(z) - rho leaked the sign of z.  This version separates two ring
sizes:

  * k_compare : the window used for the sign comparison (A2B).  We only need
                the low k_compare bits to decide the sign, provided
                |z| < 2^(k_compare-1-f).  The AND-gate count scales with
                k_compare, not with the full ring.
  * k_ring    : the ring used for the OT multiplication and the output shares.
                It must be >= k_compare.

The output mask rho is uniform in [0, 2^(k_ring-1)), so the server's share
b = ReLU(z) - rho is statistically 2^-sigma close to uniform, where

    sigma = k_ring - k_compare.

Total-variation distance between the server-share distributions of any two
ReLU values is at most 2^(k_compare-1) / 2^(k_ring-1) = 2^-sigma.

Budget caveat (must be stated in the paper): statistical distance adds up over
elements and queries, roughly eps ~= N*Q*2^-sigma.  With sigma=30 the total
budget is limited (see the reviewer's note); perfect multi-query hiding would
require the BFV/BGV integer-modulus route (future work).

Domain contract: |z| must be < 2^(k_compare-1-f).  This function asserts it in
the in-process reference (both shares are in one place); out-of-range inputs
would otherwise wrap silently.
"""

from __future__ import annotations

import argparse

import numpy as np

from mpc_relu_secure import secure_relu


def _rerandomize(cc: int, cs: int, k_ring: int, f: int, rng) -> tuple[float, float]:
    """Convert ring shares (cc, cs) of a value v into REAL-additive float shares.

    Client draws rho in [0, 2^(k_ring-1)); the server's share becomes v - rho.
    Statistical distance of the server share is 2^(k_compare)/2^(k_ring-1).
    """
    half = 1 << (k_ring - 1)
    rho = int(rng.integers(0, half))
    t0 = (cc - rho) % (1 << k_ring)
    t = (t0 + cs) % (1 << k_ring)
    b = t - (1 << k_ring) if t >= half else t
    return rho / (2.0 ** f), b / (2.0 ** f)


def secure_relu_fxp(s_c, s_s, k_ring: int = 44, f: int | None = None,
                    k_compare: int | None = None, rng=None):
    """Real secure ReLU on float additive shares (statistical masking).

    Args:
        s_c, s_s: float arrays with s_c + s_s = z (the shared activation).
        k_ring:   ring bit width for multiplication and output (>= k_compare).
        f:        fractional bits of the fixed-point encoding.
        k_compare: sign-comparison window (default k_ring, i.e. no headroom).
    Returns:
        (c_c, c_s) float arrays with c_c + c_s == relu(z) up to 2^-f.
    """
    if k_compare is None:
        k_compare = k_ring
    if f is None:
        f = 8
    assert 1 <= k_compare <= k_ring, "require 1 <= k_compare <= k_ring"
    rng = rng or np.random.default_rng()
    s_c = np.asarray(s_c, dtype=float)
    s_s = np.asarray(s_s, dtype=float)

    # domain check (in-process reference only): |z| must fit the comparison window
    zmax = (2.0 ** (k_compare - 1 - f)) - (2.0 ** -f)
    if float(np.max(np.abs(s_c + s_s))) >= zmax + (2.0 ** -f):
        raise ValueError(
            f"activation out of range: |z| >= {zmax:.3f} with "
            f"k_compare={k_compare}, f={f}"
        )

    xi_c = np.rint(s_c * (2.0 ** f)).astype(np.int64)
    xi_s = np.rint(s_s * (2.0 ** f)).astype(np.int64)
    ring_c = xi_c % (1 << k_ring)
    ring_s = xi_s % (1 << k_ring)

    c_c = np.empty(s_c.size, dtype=float)
    c_s = np.empty(s_c.size, dtype=float)
    for i in range(s_c.size):
        cc, cs = secure_relu(int(ring_c.flat[i]), int(ring_s.flat[i]),
                             k_ring, k_compare)
        c_c.flat[i], c_s.flat[i] = _rerandomize(cc, cs, k_ring, f, rng)
    return c_c.reshape(s_c.shape), c_s.reshape(s_s.shape)


def conversion_unit_test(k_ring: int = 16, n: int = 20000) -> tuple[bool, int]:
    """The re-randomization is exact on adversarial ring shares."""
    half = 1 << (k_ring - 1)
    rng = np.random.default_rng(1234)
    bad = 0
    for _ in range(n):
        v = int(rng.integers(0, half))
        c0 = int(rng.integers(0, 1 << k_ring))
        c1 = (v - c0) % (1 << k_ring)
        rho = int(rng.integers(0, half))
        t0 = (c0 - rho) % (1 << k_ring)
        t = (t0 + c1) % (1 << k_ring)
        b = t - (1 << k_ring) if t >= half else t
        if int(rho) + b != v:
            bad += 1
    return bad == 0, bad


def leak_tv(k_ring: int, k_compare: int, samples: int = 300000):
    """Empirical TV distance of the server share between ReLU=0 and ReLU=max."""
    half = 1 << (k_ring - 1)
    v_max = (1 << (k_compare - 1)) - 1
    rng = np.random.default_rng(7)
    rhos = rng.integers(0, half, size=samples)
    b_max = v_max - rhos          # server share for the maximum value
    frac_positive = float(np.mean(b_max > 0))
    bound = v_max / half          # exact TV = v_max / half = 2^-(k_ring-k_compare)
    return frac_positive, bound


def self_test(k_ring: int = 20, k_compare: int = 14, f: int = 8) -> int:
    failures = 0
    sigma = k_ring - k_compare

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    rng = np.random.default_rng(73)

    ok, bad = conversion_unit_test(k_ring=k_ring)
    check("step-4 re-randomization exact (adversarial shares)", ok, f"bad={bad}/20000")

    z = np.array([3.0, -2.0, 1.5, -4.0, 7.0, -6.0])
    s_s = rng.uniform(-8.0, 8.0, size=z.shape)
    s_c = z - s_s
    c_c, c_s = secure_relu_fxp(s_c, s_s, k_ring=k_ring, f=f, k_compare=k_compare, rng=rng)
    err = float(np.max(np.abs((c_c + c_s) - np.maximum(0.0, z))))
    tol = 3.0 / (2.0 ** f)
    check("secure fixed-point ReLU == plaintext ReLU", err <= tol,
          f"max_abs_err={err:.2e} tol={tol:.2e}")

    # statistical hiding of the server share
    frac, bound = leak_tv(k_ring, k_compare)
    ok = abs(frac - bound) < 0.005 and bound <= 2.0 ** (-sigma) + 1e-12
    check(f"server share ~= uniform (sigma={sigma}, TV<={bound:.2e})", ok,
          f"empirical_frac={frac:.4f} exact_bound={bound:.2e}")

    # out-of-range input must raise, not wrap silently
    raised = False
    try:
        secure_relu_fxp(np.array([100.0]), np.array([0.0]),
                        k_ring=k_ring, f=f, k_compare=k_compare, rng=rng)
    except ValueError:
        raised = True
    check("out-of-range input raises instead of wrapping", raised)

    print(f"\n(k_ring={k_ring}, k_compare={k_compare}, f={f}, sigma={sigma}) "
          + ("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Fixed-point secure ReLU bridge self-test")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--k-ring", type=int, default=20)
    ap.add_argument("--k-compare", type=int, default=14)
    ap.add_argument("--frac", type=int, default=8)
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.k_ring, args.k_compare, args.frac)
    print("Run with --self-test (see --k-ring / --k-compare / --frac).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
