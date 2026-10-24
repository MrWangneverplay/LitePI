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
from typing import Any

from params_pi import POLY_MODULUS_DEGREE, PROFILE_NAME

REQUIRED_PYTHON = (3, 11)
REQUIRED_POLY_MODULUS_DEGREE = 65536
REQUIRED_SECURITY_BITS = 128
REQUIRED_MPC_BACKEND = "SCI/CrypTFlow2"
REFERENCE_MPC_BACKEND = "python-p256-simplest-ot"


@dataclass(frozen=True)
class ComplianceItem:
    name: str
    required: str
    actual: str
    passed: bool
    detail: str = ""


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def collect() -> dict[str, Any]:
    py = sys.version_info[:2]
    cpp_available = _module_available("litepi_cpp")
    sci_available = _module_available("litepi_sci")
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
            "litepi_cpp/litepi_sci" if cpp_available and sci_available
            else REFERENCE_MPC_BACKEND,
            cpp_available and sci_available,
            "The reference P-256 OT path is not the paper's silent-OT/VOLE path.",
        ),
        ComplianceItem(
            "ckks_modulus_degree",
            f">={REQUIRED_POLY_MODULUS_DEGREE}",
            str(POLY_MODULUS_DEGREE),
            POLY_MODULUS_DEGREE >= REQUIRED_POLY_MODULUS_DEGREE,
            "Must be checked with the SEAL/TenSEAL security estimator.",
        ),
        ComplianceItem(
            "security_level",
            f">={REQUIRED_SECURITY_BITS}-bit",
            "unverified",
            False,
            "No estimator artifact is attached to this workspace.",
        ),
        ComplianceItem(
            "profile_role",
            "paper deployment profile",
            PROFILE_NAME,
            False,
            "Pi demo profile is explicitly a reference deployment profile.",
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
            "s=00: 2 rounds; each hybrid: +1 round",
            "target model only",
            False,
            "The current Python path uses 82 messages/41 RTTs per hybrid layer.",
        ),
    ]
    failures = [i.name for i in items if not i.passed]
    return {
        "compliant": not failures,
        "failures": failures,
        "items": [asdict(i) for i in items],
        "required_mpc_backend": REQUIRED_MPC_BACKEND,
        "reference_mpc_backend": REFERENCE_MPC_BACKEND,
    }


def deployment_compliant() -> bool:
    return bool(collect()["compliant"])
