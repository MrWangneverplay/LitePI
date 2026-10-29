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

# Legacy mask half-width (+/- 8 * 2^sigma).  Kept only for hybrid_mpc.py /
# params_b.py, which predate the per-layer M2 rule below.  The protocol server
# no longer samples from this constant.
MASK_HALF_WIDTH = 8.0 * (2.0 ** SIGMA)

# ---------------------------------------------------------------- M2 masks
# Protocol §8 M2 + §11 (simulator for message ①_k: "域内均匀采样").
#
# Layer k's mask is uniform on the fixed-point grid 2^-F inside
#     [-A_k, A_k),   A_k = 2^{sigma_k} * B_k,
# where B_k is a power-of-two bound on |z_k| taken from training statistics
# (export_mask_domains.py).  For |z| <= B_k the client's view z + r is within
# statistical distance |z| / (2 A_k) <= 2^-(sigma_k + 1) of the simulator's
# uniform draw, per element (mask_session.mask_security_report gives the
# per-query sum and Q_max).
#
# Why A_k is capped, not unbounded: the client decodes z + r through SEAL's
# double-precision CKKS decoder.  check_mask_domain.py measured the decode
# error of ct_z + Enc(r) on this profile: it stays at the CKKS noise floor
# (~5e-3, i.e. 1-2 LSB at F=8) for |r| <= 2^44 and jumps to 0.125 at 2^48
# because a double holds 53 mantissa bits.  2^42 keeps three guard bits and
# keeps r_int = r * 2^F below 2^51, so r itself is exact in a double.
# The SCI ring imposes no limit: shares are reduced mod 2^K_RING and the
# ReLU only needs |z| * 2^F < 2^(K_RING-1).
MASK_SIGMA_TARGET = 40   # wanted per-element slack; achieved = min(target, MAX - log2 B_k)
MASK_MAX_BITS = 42       # hard cap: log2 A_k <= 42
MASK_MIN_SIGMA = 20      # refuse domains that cannot reach this slack
MASK_BOUND_MARGIN = 1.25  # B_k = next power of two >= 1.25 * max|z_k| (train split)


def describe() -> str:
    return (
        f"Paper profile B: sigma={SIGMA} (k_ring={K_RING}, "
        f"k_compare={K_COMPARE}, f={F}), "
        f"N={POLY_MODULUS_DEGREE}, scale=2^{GLOBAL_SCALE_BITS}, "
        f"M2 masks: A_k=2^sigma_k*B_k, sigma_target={MASK_SIGMA_TARGET}, "
        f"log2 A_k<={MASK_MAX_BITS}"
    )
