"""SCI (CrypTFlow2) ReLU adapter for LitePI.

Replaces the pure-Python P-256 OT ReLU (secure_relu_net.run_party_batch) with
the CrypTFlow2 constant-round ReLU.  The interface mirrors run_party_batch:
float additive shares in, float additive shares out.

The SCI ReLU runs offline base-OT + IKNP OT-extension at construction time
(one-time setup); the online ReLU comparison is constant-round.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# ``scripts/build.sh`` drops the pybind11 extension into <repo>/sci, which is
# not on sys.path by default.
_SCI_DIR = Path(__file__).resolve().parent.parent / "sci"
if _SCI_DIR.is_dir() and str(_SCI_DIR) not in sys.path:
    sys.path.insert(0, str(_SCI_DIR))

try:
    import litepi_sci
except ImportError:  # pragma: no cover - SCI module not built yet
    litepi_sci = None

SCI_AVAILABLE = litepi_sci is not None


def quantize_to_ring(shares: list[float], f: int, k: int) -> np.ndarray:
    """float shares -> uint64 two's-complement ring elements (mod 2^k)."""
    scale = 1 << f
    mask = (1 << k) - 1
    out = []
    for s in shares:
        v = int(round(float(s) * scale))
        out.append(v & mask)
    return np.asarray(out, dtype=np.uint64)


def ring_to_float(shares: np.ndarray, f: int, k: int) -> np.ndarray:
    """uint64 ring elements -> float (two's-complement centered)."""
    half = 1 << (k - 1)
    mask = (1 << k) - 1
    scale = 1 << f
    out = []
    for v in shares:
        v = int(v) & mask
        v = v - (1 << k) if v >= half else v
        out.append(v / scale)
    return np.asarray(out, dtype=np.float64)


class SciReLUSession:
    """One party's handle to the SCI secure ReLU protocol."""

    def __init__(self, party: int, port: int, address: str = "127.0.0.1"):
        if litepi_sci is None:
            raise RuntimeError("litepi_sci module is not built (run scripts/build.sh)")
        assert party in (1, 2), "SCI party must be 1 (ALICE/server) or 2 (BOB/client)"
        self.party = party
        self._sci = litepi_sci.SecureReLU(party, port, address)

    def relu(self, shares: list[float], f: int, k: int) -> np.ndarray:
        ring = quantize_to_ring(shares, f, k)
        out_ring = self._sci.relu(ring, k)
        return ring_to_float(out_ring, f, k)

    def bytes_sent(self) -> int:
        return self._sci.bytes_sent()
