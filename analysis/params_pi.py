"""Deployment profile for the LitePI hybrid-inference protocol.

This module is the single CKKS/fixed-point parameter source for the protocol
server, the client, the canonical networks and the mask-domain export.  It now
carries the **paper profile** of the protocol specification:

    protocol SSOT §1.2 / §5.3 / §6.1:
        poly_modulus_degree = 8192
        coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
        global scale        = 2^21
        128-bit security    = standard-table checked in protocol_contract.py

    fixed point (unchanged, matches the sigma=30 statistical margin):
        k_ring = 44, k_compare = 14, f = 8, sigma = 30

Why the earlier "pi-demo" profile existed: with three primes [60, 24, 60] the
modulus chain had only 1.5 levels, so a canonical network (conv + square + fc1
+ square + fc2) ran out of depth.  The paper chain has seven data primes
(166 bit, six rescalings), which is what the protocol's §5.3 level trace
assumes: fresh L7, conv → L6, act → L5, fc1 → L4, act → L3, fc2 → L2.

Every hybrid ReLU decrypts and re-encrypts, so the depth budget resets per
hybrid segment (protocol §4.2 property 2); the seven-level chain is sized for
the HE-only path (three plaintext-matrix products plus two ciphertext squares).

NOTE: switching to this profile invalidates any timing/byte figure produced
under the old pi-demo profile.  Those JSON files remain reference-only and
must not be reported as paper-profile numbers.
"""

from __future__ import annotations

PROFILE_NAME = "paper-profile-B"
PAPER_PROFILE = True

# MPC / fixed point.
K_COMPARE = 14
F = 8
SIGMA = 30
K_RING = K_COMPARE + SIGMA

# CKKS.  Seven data primes (40 + 21*6 = 166 bit) plus one 40-bit
# key-switching prime, per protocol §6.1.
POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = [40, 21, 21, 21, 21, 21, 21, 40]
GLOBAL_SCALE_BITS = 21
GLOBAL_SCALE = 1 << GLOBAL_SCALE_BITS

# Input mask domain used by the server: +/- 8 * 2^sigma.
MASK_HALF_WIDTH = 8.0 * (2.0 ** SIGMA)


def describe() -> str:
    return (
        f"Paper profile B: sigma={SIGMA} (k_ring={K_RING}, "
        f"k_compare={K_COMPARE}, f={F}), "
        f"N={POLY_MODULUS_DEGREE}, scale=2^{GLOBAL_SCALE_BITS}"
    )
