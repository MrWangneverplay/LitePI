"""LitePI security profile B: statistical masking with sigma = 30.

This module is the single source of truth for the "B" profile used in the
paper.  Scripts that need the mask width, the CKKS parameter set, or the
leakage budget should import from here instead of hard-coding numbers.

Profile B defines the sigma=30 statistical-sharing target and raises the
plaintext bound so that the mask ring can be 44 bits wide while the comparison
window stays at 14 bits:

    sigma = K_RING - K_COMPARE = 44 - 14 = 30

IMPORTANT: the CKKS chain below is retained for protocol/self-test
compatibility, but it is NOT a feasible end-to-end six-layer precision profile
for sigma=30.  `sweep_ckks_frontier.py` measures sigma<=21 at depth 6 and
requires level primes of at least 35 bits; sigma=30 fails at 0.327 relative
error.  Large statistically masked shares also lose precision when a later
ciphertext multiplication follows a plaintext addition.  The Pi deployment
therefore uses the separately labelled `params_pi.py` profile.  Do not quote a
Pi-demo result as a profile-B result, and do not claim that the chain below
makes the full sigma=30 CKKS path feasible.
"""

from __future__ import annotations

# --- MPC / fixed-point -------------------------------------------------
K_COMPARE = 14          # sign window: |z| < 2^(14 - 1 - 8) = 32
F = 8                   # fractional bits (1/256 resolution)
K_RING = 44             # mask ring width
SIGMA = K_RING - K_COMPARE   # = 30

# --- CKKS (TenSEAL) ----------------------------------------------------
# N=16384 with 60-bit outer primes raises the plaintext bound from 2^18 to
# roughly 2^36 at scale 2^23.  The intermediate primes must be 23 bits: SEAL
# requires p = 1 mod 2N, and for N=16384 the number of qualifying primes is
# 1 at 18 bits, 2 at 20, 4 at 21, 5 at 22, and 19 at 23 -- so 23 bits is the
# smallest size that can supply the levels.  Six multiplications need seven
# levels (one spare for the final rescale), hence seven 23-bit level primes.
# The scale follows the level primes so rescaling preserves it.
POLY_MODULUS_DEGREE = 16384
COEFF_MOD_BIT_SIZES = [60, 23, 23, 23, 23, 23, 23, 23, 60]  # 9 primes -> 7 levels
GLOBAL_SCALE_BITS = 23
GLOBAL_SCALE = 1 << GLOBAL_SCALE_BITS

# --- Mask domain -------------------------------------------------------
# The server's input mask is drawn from +-8 * 2^SIGMA (same convention as
# hybrid_mpc.py).  In profile B this is +-2^33, inside the 2^38 plaintext
# bound.
MASK_HALF_WIDTH = 8.0 * (2.0 ** SIGMA)


def describe() -> str:
    return (f"profile B: sigma={SIGMA} (k_ring={K_RING}, k_compare={K_COMPARE}, "
            f"f={F}), N={POLY_MODULUS_DEGREE}, scale=2^{GLOBAL_SCALE_BITS}")
