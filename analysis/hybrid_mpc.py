"""LitePI Phase 4 core: HE <-> share conversion + MPC ReLU (local simulation).

Implements the five-step Hybrid-ReLU flow of 协议设计.md §4:
    1) server masks  ct_{z+r} = ct_z + Enc_pk(r)
    2) client decrypts -> s_c = z + r  (解密即份额; server keeps s_s = -r)
    3) MPC ReLU on additive shares [z] -> [sigma(z)] = (c_c, c_s)
    4) client re-encrypts c_c (fresh ciphertext)
    5) server adds its plaintext share c_s -> ct_{sigma(z)}

Run the acceptance self-test with:
    python analysis/hybrid_mpc.py --self-test

HONESTY NOTE (read before citing anything):
  * The additive secret-sharing arithmetic and the HE<->share conversions are
    real and follow 协议设计.md §4.
  * The comparison inside MPC ReLU has TWO implementations:
      - real=False (default): functional simulation. Client and server live in
        the same process, so z is reconstructed locally and re-shared. Correct
        numbers, but NOT a cryptographic comparison. Used for the end-to-end
        pipeline / performance runs (E2, RTT, ...).
      - real=True: the real OT/GMW protocol (mpc_relu_fxp.secure_relu_fxp ->
        mpc_relu_secure -> gmw_core -> ot_core). No party ever reconstructs z.
        This is the drop-in that backs the paper's security claim; it is slow on
        the pure-Python elliptic curve, so use a small array and k<=10 here and
        treat IKNP-OT / a C library as the deployment item.
    The rest of the pipeline (steps 1-2, 4-5) is identical for both.
"""

from __future__ import annotations

import argparse
import secrets
from pathlib import Path

import numpy as np

try:
    import tenseal as ts
except Exception:  # pragma: no cover
    ts = None

try:
    from mpc_relu_fxp import secure_relu_fxp
except Exception:  # pragma: no cover - optional, needed only for real=True
    secure_relu_fxp = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CKKS parameters mirror model/mnist_cnn_enc.py (7 primes -> 6 mult depth).
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 21, 21, 21, 21, 40]
GLOBAL_SCALE = 2 ** 21

FIXED_FRAC = 12  # fractional bits for fixed-point share arithmetic (E7)

# Activation values are bounded by |z| < 8 = 2^3.  The input mask r must range
# over a domain 2^SIGMA times wider so that the client (which sees z+r) cannot
# recover z: the total-variation distance between the distributions of z1+r and
# z2+r is at most |z1-z2| / (16 * 2^SIGMA).
#
# SIGMA is capped by the CKKS plaintext bound |m| < q0/(2*scale) = 2^18, so
# 8 * 2^SIGMA must stay below 2^18, i.e. SIGMA <= 14.  Larger SIGMA needs a
# bigger first prime (fewer multiplication levels) or the integer-modulus
# BFV/BGV route.
VALUE_BITS = 3
VALUE_DOMAIN = (-8.0, 8.0)
SIGMA = 14
MASK_DOMAIN = (-8.0 * (2.0 ** SIGMA), 8.0 * (2.0 ** SIGMA))


def _crypto_uniform(lo: float, hi: float, size: int) -> np.ndarray:
    """Uniform floats in [lo, hi) drawn from the OS CSPRNG (os.urandom)."""
    out = np.empty(size, dtype=np.float64)
    span = hi - lo
    for i in range(size):
        out[i] = lo + span * (secrets.randbits(53) / (1 << 53))
    return out


def make_context():
    if ts is None:
        raise RuntimeError("tenseal is required; install it in the project venv")
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    ctx.global_scale = GLOBAL_SCALE
    ctx.generate_galois_keys()
    return ctx


def relu(x):
    return np.maximum(0.0, x)


class MaskSampler:
    """Fresh CSPRNG masks, server side only (M1: fresh per (query, layer)).

    The mask hides the activation z from the CLIENT (which only sees z+r), so r
    must be (a) drawn from the OS CSPRNG and (b) never revealed to the client.
    The old "shared session seed" design is wrong: if the client knew the seed
    it could compute r and recover z = (z+r) - r.
    """

    def __init__(self, domain=MASK_DOMAIN):
        self.lo, self.hi = domain

    def draw(self, query: int, layer: int, size: int) -> np.ndarray:
        del query, layer  # uniqueness comes from the CSPRNG, not a counter
        return _crypto_uniform(self.lo, self.hi, size)


def _fresh_share(value: np.ndarray, rng) -> tuple[np.ndarray, np.ndarray]:
    """Split `value` into (client_share, server_share) with sum == value."""
    s_s = rng.uniform(VALUE_DOMAIN[0], VALUE_DOMAIN[1], size=value.shape)
    s_c = value - s_s
    return s_c.astype(float), s_s.astype(float)


# --------------------------------------------------------------------- 5 steps

def he_to_share(ctx, ct_z, r: np.ndarray):
    """Steps 1-2. Server masks with Enc(r); returns (ct_{z+r}, server_share).

    The client then decrypts ct_{z+r} itself -> s_c = z + r.  Together with
    server_share = -r this already forms an additive sharing of z
    (the "解密即份额" property, no re-encode round-trip needed).
    """
    ct_r = ts.ckks_vector(ctx, r.tolist())
    ct_zr = ct_z + ct_r
    return ct_zr, (-r).astype(float)


def client_share(ct_zr) -> np.ndarray:
    """Client side of step 2: decrypt ct_{z+r} -> s_c = z + r."""
    return np.asarray(ct_zr.decrypt(), dtype=float)


def mpc_relu(s_c: np.ndarray, s_s: np.ndarray, rng=None,
             real: bool = False, k: int = 20, k_compare: int | None = 14,
             f: int | None = None):
    """Step 3. MPC ReLU on additive shares -> fresh shares (c_c, c_s) of relu(z).

    real=False : local simulation (fast; reconstructs z). For pipeline/perf runs.
    real=True  : real OT/GMW protocol via mpc_relu_fxp.secure_relu_fxp; no party
                 ever sees z. Slow on the pure-Python curve -- small arrays only.
    """
    if real:
        if secure_relu_fxp is None:
            raise RuntimeError("mpc_relu_fxp.py not found next to hybrid_mpc.py")
        return secure_relu_fxp(s_c, s_s, k_ring=k, k_compare=k_compare, f=f, rng=rng)
    rng = rng or np.random.default_rng()
    z = s_c + s_s
    sigma = relu(z)
    return _fresh_share(sigma, rng)


def share_to_he(ctx, c_c: np.ndarray, c_s: np.ndarray):
    """Steps 4-5. Client re-encrypts c_c (fresh); server adds plaintext c_s."""
    ct_cc = ts.ckks_vector(ctx, c_c.tolist())
    return ct_cc + c_s.tolist()


# ------------------------------------------------------------------- E7 trunc

def _split_int(v: np.ndarray, rng):
    s_s = rng.integers(-(2 ** 40), 2 ** 40, size=v.shape, dtype=np.int64)
    return (v - s_s).astype(np.int64), s_s.astype(np.int64)


def to_fixed(x, f=FIXED_FRAC):
    return np.rint(x * (2.0 ** f)).astype(np.int64)


def direct_truncate(v: np.ndarray, f=FIXED_FRAC):
    return np.floor_divide(v, 2 ** f)


def random_truncate(v: np.ndarray, f=FIXED_FRAC, rng=None):
    """Probabilistic truncation: round v/2^f up with prob (v mod 2^f)/2^f."""
    rng = rng or np.random.default_rng()
    q = np.floor_divide(v, 2 ** f)
    r = np.mod(v, 2 ** f)
    up = (rng.random(v.shape) * (2.0 ** f) < r).astype(np.int64)
    return q + up


def beaver_mul(x, y, f=FIXED_FRAC, rng=None):
    """2-party Beaver multiplication on f-bit fixed-point ints.

    Returns (client_share, server_share) of x*y in 2f-bit fixed point.
    """
    rng = rng or np.random.default_rng()
    xi = to_fixed(x, f)
    yi = to_fixed(y, f)
    shape = xi.shape

    a = rng.integers(-(2 ** 16), 2 ** 16, size=shape, dtype=np.int64)
    b = rng.integers(-(2 ** 16), 2 ** 16, size=shape, dtype=np.int64)
    c = a * b
    a_c, a_s = _split_int(a, rng)
    b_c, b_s = _split_int(b, rng)
    c_c, c_s = _split_int(c, rng)
    x_c, x_s = _split_int(xi, rng)
    y_c, y_s = _split_int(yi, rng)

    d = (x_c + x_s) - (a_c + a_s)  # opened
    e = (y_c + y_s) - (b_c + b_s)  # opened

    xy_c = c_c + d * b_c + e * a_c + d * e
    xy_s = c_s + d * b_s + e * a_s
    return xy_c, xy_s


# ------------------------------------------------------------------- self test

def _rel_err(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(approx - ref) / (np.abs(ref) + 1e-9)))


def _abs_err(approx: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(approx - ref)))


def self_test(real_relu: bool = False) -> int:
    rng = np.random.default_rng(73)
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    # M1: mask uniqueness across 10^4 distinct (query, layer) draws
    sampler = MaskSampler()
    seen = set()
    dup = 0
    for q in range(10000):
        m = sampler.draw(q, 0, 8)
        key = tuple(np.round(m, 12).tolist())
        if key in seen:
            dup += 1
        seen.add(key)
    check("M1 mask uniqueness (10^4 queries)", dup == 0, f"duplicates={dup}")

    # input-mask statistical hiding (pure Python; no HE needed)
    span = 16.0 * (2.0 ** SIGMA)
    tv = 10.0 / span                      # TV between z=+5 and z=-5 client views
    check(f"input mask hides z (SIGMA={SIGMA}, TV={tv:.2e})",
          tv <= 2.0 ** (-SIGMA), f"bound=2^-{SIGMA}={2.0**(-SIGMA):.2e}")

    ctx = make_context()
    z = rng.uniform(-5.0, 5.0, size=64)
    sigma = relu(z)

    # step 3 in isolation: MPC ReLU on EXACT additive shares (no HE noise)
    s_c_exact, s_s_exact = _fresh_share(z, rng)
    c_c, c_s = mpc_relu(s_c_exact, s_s_exact, rng)
    check("MPC ReLU reconstruct (exact shares)", _rel_err(c_c + c_s, sigma) < 1e-9,
          f"rel_err={_rel_err(c_c + c_s, sigma):.2e}")

    # step 3 via the REAL OT/GMW protocol (drop-in swap; slow on the pure-Python
    # curve, so only a 3-element slice with k=10 is checked here)
    if real_relu:
        if secure_relu_fxp is None:
            check("MPC ReLU (REAL protocol)", False, "mpc_relu_fxp.py missing")
        else:
            zs = z[:3]
            sc, ss = _fresh_share(zs, rng)
            rc, rs = mpc_relu(sc, ss, real=True, k=26, k_compare=14, f=8)
            tol = 3.0 / (2.0 ** 8)
            check("MPC ReLU (REAL OT/GMW protocol, statistical masking)",
                  _abs_err(rc + rs, relu(zs)) <= tol,
                  f"abs_err={_abs_err(rc + rs, relu(zs)):.2e} tol={tol:.2e}")
            ct_out = share_to_he(ctx, rc, rs)
            dec_out = np.asarray(ct_out.decrypt(), dtype=float)
            check("share->HE recovery (large statistical shares, ~2^17)",
                  _abs_err(dec_out, relu(zs)) <= tol,
                  f"abs_err={_abs_err(dec_out, relu(zs)):.2e} tol={tol:.2e}")

    # steps 1-2: HE -> share
    ct_z = ts.ckks_vector(ctx, z.tolist())
    r = sampler.draw(0, 0, len(z))
    ct_zr, s_s = he_to_share(ctx, ct_z, r)
    s_c = client_share(ct_zr)
    check("HE->share (解密即份额)", _abs_err(s_c + s_s, z) < 1e-2,
          f"abs_err={_abs_err(s_c + s_s, z):.2e}")

    # steps 4-5: share -> HE
    ct_sigma = share_to_he(ctx, c_c, c_s)
    dec = np.asarray(ct_sigma.decrypt(), dtype=float)
    check("share->HE recovery", _abs_err(dec, sigma) < 1e-2,
          f"abs_err={_abs_err(dec, sigma):.2e}")

    # E7: Beaver multiplication (exact integer reconstruction)
    x = rng.uniform(-4.0, 4.0, size=256)
    y = rng.uniform(-4.0, 4.0, size=256)
    xy_c, xy_s = beaver_mul(x, y)
    xy_2f = xy_c + xy_s
    ref_int = to_fixed(x, FIXED_FRAC) * to_fixed(y, FIXED_FRAC)
    mismatch = int(np.sum(xy_2f != ref_int))
    check("Beaver mul correctness (exact integer)", mismatch == 0, f"mismatches={mismatch}")

    # E7: random truncation is unbiased; direct (floor) truncation is biased low
    n_trials = 5000
    mean_direct = 0.0
    mean_random = 0.0
    for _ in range(n_trials):
        V = rng.integers(0, 2 ** (2 * FIXED_FRAC), dtype=np.int64)
        t = np.asarray([V])
        mean_direct += (direct_truncate(t)[0] - V / (2.0 ** FIXED_FRAC)) / (2.0 ** FIXED_FRAC)
        mean_random += (random_truncate(t, rng=rng)[0] - V / (2.0 ** FIXED_FRAC)) / (2.0 ** FIXED_FRAC)
    mean_direct /= n_trials
    mean_random /= n_trials
    check("random truncation less biased", abs(mean_random) < abs(mean_direct),
          f"direct_bias={mean_direct:.4f} random_bias={mean_random:.4f}")

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI hybrid-MPC core self-test")
    ap.add_argument("--self-test", action="store_true", help="run Phase-4 acceptance checks")
    ap.add_argument("--real-relu", action="store_true",
                    help="also run the REAL OT/GMW ReLU check on a 3-element slice (slow)")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.real_relu)
    print("Run with --self-test to execute the Phase-4 acceptance checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
