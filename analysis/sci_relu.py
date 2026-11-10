"""SCI (CrypTFlow2) ReLU adapter for LitePI.

Replaces the pure-Python P-256 OT ReLU (secure_relu_net.run_party_batch) with
the CrypTFlow2 constant-round ReLU.  The interface mirrors run_party_batch:
float additive shares in, float additive shares out.

Only the base OTs run at construction time (one-time setup).  Everything else
runs inside each relu() call: the Millionaire comparison (1-of-16 KKOT leaves,
bit triples generated on the spot) and the two IKNP correlated OTs of the
multiplexer.  There is no silent OT and no offline triple store.
"""

from __future__ import annotations

import sys
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
import math

import numpy as np

# The unified build installs to analysis/. Prefer its compatible extension
# over a stale legacy binary in sci/; use sci/ only as a backwards-compatible
# fallback when analysis/ has no extension for this interpreter's ABI.
_ANALYSIS_DIR = Path(__file__).resolve().parent
_SCI_DIR = _ANALYSIS_DIR.parent / "sci"
_has_current_extension = any(
    (_ANALYSIS_DIR / f"litepi_sci{suffix}").is_file() for suffix in EXTENSION_SUFFIXES
)
_extension_dir = _ANALYSIS_DIR if _has_current_extension else _SCI_DIR
if _extension_dir.is_dir():
    if str(_extension_dir) in sys.path:
        sys.path.remove(str(_extension_dir))
    sys.path.insert(0, str(_extension_dir))

try:
    import litepi_sci
except ImportError:  # pragma: no cover - SCI module not built yet
    litepi_sci = None

SCI_AVAILABLE = litepi_sci is not None


def validate_ring_parameters(f: int, k: int) -> None:
    """Float bridge is exact only in the supported binary64 fixed-point range."""
    if (type(k) is not int or type(f) is not int
            or not 2 <= k <= 53 or not 0 <= f < k):
        raise ValueError("require integer 2 <= k <= 53 and 0 <= f < k")


def require_real_bridge(module) -> None:
    """Reject legacy independently-centered output before opening SCI channels."""
    if (module is None or getattr(module, "REAL_BRIDGE_VERSION", None) != 1
            or not hasattr(getattr(module, "SecureReLU", None), "relu_real")):
        raise RuntimeError("SCI real-additive bridge v1 is required; rebuild "
                           "scripts/build_litepi_sci.sh on both parties")


def quantize_to_ring(shares: list[float], f: int, k: int) -> np.ndarray:
    """Finite float shares -> uint64 ring elements; not a secret-value range check."""
    validate_ring_parameters(f, k)
    scale = 1 << f
    mask = (1 << k) - 1
    out = []
    for s in shares:
        value = float(s) * scale
        if not math.isfinite(value):
            raise ValueError("non-finite fixed-point share")
        v = int(round(value))
        out.append(v & mask)
    return np.asarray(out, dtype=np.uint64)


def ring_to_float(shares: np.ndarray, f: int, k: int) -> np.ndarray:
    """Center one ring vector, NOT an arbitrary pair of output shares.

    Centering independent ring shares does not preserve real-additive sums.
    This helper is for ring inputs/reconstructed values; SCI ReLU outputs
    must use the masked relu_real bridge instead.
    """
    validate_ring_parameters(f, k)
    half = 1 << (k - 1)
    mask = (1 << k) - 1
    scale = 1 << f
    out = []
    for v in shares:
        v = int(v) & mask
        v = v - (1 << k) if v >= half else v
        out.append(v / scale)
    return np.asarray(out, dtype=np.float64)


def sci_bytes(session) -> int:
    """Bytes this party has sent on its SCI channels (0 when SCI is not in use)."""
    return int(session.bytes_sent()) if session is not None else 0


class SciReLUSession:
    """One party's handle to the SCI secure ReLU protocol."""

    def __init__(self, party: int, port: int, address: str = "127.0.0.1"):
        if litepi_sci is None:
            raise RuntimeError("litepi_sci module is not built (run scripts/build_litepi_sci.sh)")
        require_real_bridge(litepi_sci)
        if party not in (1, 2):
            raise ValueError("SCI party must be 1 (ALICE/server) or 2 (BOB/client)")
        self.party = party
        self._sci = litepi_sci.SecureReLU(party, port, address)

    def relu(self, shares: list[float], f: int, k: int) -> np.ndarray:
        ring = quantize_to_ring(shares, f, k)
        # The native bridge adds one masked BOB->ALICE message. Do not replace
        # it with independently centered output shares (possible +/-2^k error).
        return np.asarray(self._sci.relu_real(ring, k, f), dtype=np.float64)

    def bytes_sent(self) -> int:
        return self._sci.bytes_sent()
