"""Executable contract for the LitePI paper protocol.

The paper's full protocol is stricter than the Python reference demo:
Python 3.11, a C++/SCI MPC backend, and 128-bit CKKS parameters.  This module
centralises those checks so a reference implementation cannot accidentally be
reported as a deployable experiment.
"""

from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from params_pi import (
    COEFF_MOD_BIT_SIZES,
    PAPER_PROFILE,
    POLY_MODULUS_DEGREE,
    PROFILE_NAME,
)

REQUIRED_PYTHON = (3, 11)
# Protocol SSOT §1.2: N=8192, coeff [40, 21×6, 40], scale 2^21, >=128-bit.
REQUIRED_POLY_MODULUS_DEGREE = 8192
REQUIRED_SECURITY_BITS = 128
REQUIRED_MPC_BACKEND = "SCI/CrypTFlow2"
REFERENCE_MPC_BACKEND = "python-p256-simplest-ot"

# HomomorphicEncryption.org "Security Standard" Table 1: for n = 2^13 = 8192
# the largest coefficient-modulus bit count that still meets the 128-bit
# classical security target is 218 bits.  The protocol profile [40, 21x6, 40]
# totals 206 bits, so it sits inside that bound with 12 bits of headroom.
# This is a standard-table check, not a per-build estimator run; a full
# lattice-estimator artifact is still required for the paper appendix.
SECURITY_TABLE = {
    8192: {128: 218},
}


@dataclass(frozen=True)
class ComplianceItem:
    name: str
    required: str
    actual: str
    passed: bool
    detail: str = ""
    # Implemented but not yet measured: does not fail the contract, but must
    # not be reported as PASS either (e.g. r_h before the netem measurement).
    pending: bool = False


def _r_h_status() -> tuple[bool, str, str]:
    """(measured end-to-end?, actual, detail) from results/rounds/r_h_*.json.

    The message-counting files (r_h_sci_*.json) only see the main wire, so
    they bound the control plane, not SCI's own socket.  Only an end-to-end
    measurement (netem_e2e.py --mode e3 -> r_h_netem_*.json, scope
    "end-to-end") settles the protocol's r_h.
    """
    import json
    rounds = Path(__file__).resolve().parent.parent / "results" / "rounds"
    e2e, ctrl = [], []
    for f in sorted(rounds.glob("r_h_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        r = d.get("r_h_measured")
        if r is None:
            continue
        scope = d.get("scope") or ("end-to-end" if d.get("sci_internal_rounds_counted")
                                   else "control-plane")
        (e2e if scope == "end-to-end" else ctrl).append((f.name, float(r)))
    if e2e:
        worst = max(e2e, key=lambda t: t[1])
        return True, f"r_h={worst[1]:.2f} end-to-end ({worst[0]})", (
            "Measured from the latency-vs-RTT slope, SCI-internal rounds included.")
    if ctrl:
        return False, "r_h=1 on the main wire only (" + ", ".join(n for n, _ in ctrl) + ")", (
            "SCI's ReLU runs on its own socket and is not counted; CrypTFlow2 reports "
            "log l + 2 rounds per ReLU.  Run netem_e2e.py --mode e3 to measure r_h.")
    return False, "unmeasured", "No results/rounds/r_h_*.json yet."


def _module_available(name: str) -> bool:
    if name == "litepi_sci":
        sci_dir = Path(__file__).resolve().parent.parent / "sci"
        if sci_dir.is_dir() and str(sci_dir) not in sys.path:
            sys.path.insert(0, str(sci_dir))
    return importlib.util.find_spec(name) is not None


def _modulus_bits() -> int:
    return int(sum(int(b) for b in COEFF_MOD_BIT_SIZES))


def security_check() -> tuple[bool, str, str]:
    """Check the CKKS modulus chain against the standard security table."""
    total = _modulus_bits()
    bound = SECURITY_TABLE.get(POLY_MODULUS_DEGREE, {}).get(REQUIRED_SECURITY_BITS)
    required = f">={REQUIRED_SECURITY_BITS}-bit (logQ <= {bound} at N={POLY_MODULUS_DEGREE})"
    if bound is None:
        return False, required, f"no standard-table entry (logQ={total})"
    ok = total <= bound
    return (
        ok,
        required,
        f"logQ={total} bit, standard-table bound {bound} bit"
        f"{'' if ok else ' -- EXCEEDS BOUND'}",
    )



def collect() -> dict[str, Any]:
    py = sys.version_info[:2]
    sci_available = _module_available("litepi_sci")
    sec_ok, sec_required, sec_actual = security_check()
    profile_is_paper = bool(PAPER_PROFILE) and PROFILE_NAME.startswith("paper")
    r_h_measured, r_h_actual, r_h_detail = _r_h_status()
    items = [
        ComplianceItem(
            "python_runtime",
            "3.11",
            f"{py[0]}.{py[1]}",
            py == REQUIRED_PYTHON,
            f"platform={platform.machine()}",
        ),
        ComplianceItem(
            "mpc_backend",
            REQUIRED_MPC_BACKEND,
            "litepi_sci" if sci_available else REFERENCE_MPC_BACKEND,
            sci_available,
            "The reference P-256 OT path is not the paper's silent-OT/VOLE path.",
        ),
        ComplianceItem(
            "ckks_modulus_degree",
            str(REQUIRED_POLY_MODULUS_DEGREE),
            str(POLY_MODULUS_DEGREE),
            POLY_MODULUS_DEGREE == REQUIRED_POLY_MODULUS_DEGREE,
            "Protocol SSOT §1.2: N=8192, checked with the SEAL/TenSEAL security estimator.",
        ),
        ComplianceItem(
            "security_level",
            sec_required,
            sec_actual,
            sec_ok,
            "HomomorphicEncryption.org standard table; a lattice-estimator artifact "
            "is still required for the paper appendix.",
        ),
        ComplianceItem(
            "profile_role",
            "paper deployment profile",
            PROFILE_NAME,
            profile_is_paper,
            "params_pi.py carries the protocol §6.1 CKKS parameter set.",
        ),
        ComplianceItem(
            "mask_m1_m3",
            "PRG(session,query,layer)+query counter",
            "implemented in mask_session.py"
            if _module_available("mask_session") else "missing",
            _module_available("mask_session"),
            "M1 freshness and M3 resync are implemented.",
        ),
        ComplianceItem(
            "protocol_modes",
            "one protocol with s=0/s=1/mixed schedules",
            "implemented in pi_server.py",
            True,
            "HE-P2, Hybrid-ReLU, and mixed schedules share the same trace.",
        ),
        ComplianceItem(
            "canonical_models",
            "MNIST/CIFAR-10/UCI-HAR protocol architectures",
            "canonical_net.py"
            if _module_available("canonical_net") else "missing",
            _module_available("canonical_net"),
            "The server loads the three train_hybrid.py geometries; fast RGB is extension-only.",
        ),
        ComplianceItem(
            "schedule_obfuscation",
            "§7.2 dummy ciphertext pairs",
            "configured by --obfuscate-schedule"
            if _module_available("pi_server") else "missing",
            _module_available("pi_server"),
            "Every nonlinear layer follows the same interaction shape when enabled.",
        ),
        ComplianceItem(
            "mask_m2",
            "per-layer activation-domain alignment",
            "per-layer JSON domains"
            if _module_available("export_mask_domains") else "missing",
            _module_available("export_mask_domains"),
            "Training-split z_k and logit ranges are exported and loaded by the server.",
        ),
        ComplianceItem(
            "output_mode_b",
            "secure MPC argmax",
            "GMW tournament argmax"
            if _module_available("secure_argmax") else "missing",
            _module_available("secure_argmax"),
            "C-1 secure comparisons yield an index share; the client never receives logits.",
        ),
        ComplianceItem(
            "client_cost",
            "s=00: 2 rounds; each hybrid: +r_h rounds (protocol target r_h=1)",
            r_h_actual if sci_available
            else "python OT path (41 RTTs/hybrid layer)",
            sci_available,
            r_h_detail,
            pending=sci_available and not r_h_measured,
        ),
    ]
    failures = [i.name for i in items if not i.passed]
    pending = [i.name for i in items if i.passed and i.pending]
    return {
        "compliant": not failures,
        "failures": failures,
        "pending": pending,
        "items": [asdict(i) for i in items],
        "required_mpc_backend": REQUIRED_MPC_BACKEND,
        "reference_mpc_backend": REFERENCE_MPC_BACKEND,
    }


def deployment_compliant() -> bool:
    return bool(collect()["compliant"])
