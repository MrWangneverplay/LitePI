"""Raspberry-Pi deployment profile for the LitePI hybrid-inference demo.

This profile is intentionally separate from ``params_b.py``.

Paper profile B uses sigma=30.  The six-layer HE-only chain cannot carry that
margin while preserving f=8 precision.  The Pi demo has a different execution
shape: every hidden layer ends in a hybrid ReLU, which decrypts and re-encrypts
the ciphertext and resets the CKKS depth.  It can therefore keep the paper's
30-bit statistical margin with a one-multiplication chain:

    k_ring = 44, k_compare = 14, f = 8, sigma = 30

At scale 2^24, the largest client output share is below 2^35, while the
plaintext magnitude bound is also 2^35.  A one-multiplication CKKS chain is
enough because every hybrid ReLU decrypts and re-encrypts, resetting the depth
budget.  The server also encrypts its output share and adds two ciphertexts;
adding a large finalize share as plaintext loses precision before the next
ciphertext multiplication.

This produces a runnable two-machine deployment with the paper's sigma=30
statistical margin, but the CKKS chain and model are still demo-specific.  It
MUST NOT be reported as the six-layer profile-B precision result.
"""

from __future__ import annotations

PROFILE_NAME = "pi-demo-sigma30"
PAPER_PROFILE = False

# MPC / fixed point.
K_COMPARE = 14
F = 8
SIGMA = 30
K_RING = K_COMPARE + SIGMA

# CKKS.  Each linear layer performs one ciphertext/plaintext multiplication;
# the following hybrid ReLU resets the depth by decrypt/re-encrypt.
POLY_MODULUS_DEGREE = 16384
COEFF_MOD_BIT_SIZES = [60, 24, 60]
GLOBAL_SCALE_BITS = 24
GLOBAL_SCALE = 1 << GLOBAL_SCALE_BITS

# Input mask domain used by the server: +/- 8 * 2^sigma.
MASK_HALF_WIDTH = 8.0 * (2.0 ** SIGMA)


def describe() -> str:
    return (
        f"Pi demo: sigma={SIGMA} (k_ring={K_RING}, "
        f"k_compare={K_COMPARE}, f={F}), "
        f"N={POLY_MODULUS_DEGREE}, scale=2^{GLOBAL_SCALE_BITS}"
    )
