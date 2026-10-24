"""Client-side cost model from protocol design §12.

This is the target SCI/silent-OT accounting, not the current Python P-256
reference measurements.  Keep the two labels separate in reports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ClientCost:
    online_rounds: int
    output_mode: str
    encrypt_count: int
    decrypt_count: int
    hybrid_layers: int
    mpc_elements: int
    base_bytes: int
    hybrid_bytes: int
    ot_bytes: int
    total_bytes: int
    zero_interaction_fallback: bool
    cost_basis: str = "protocol-target-SCI-rh1"
    implementation_status: str = "not-measured"


def estimate(hybrid_layers: int, mpc_elements: int,
             output_mode: str = "logits") -> ClientCost:
    if hybrid_layers < 0 or mpc_elements < 0:
        raise ValueError("cost inputs must be non-negative")
    if output_mode not in ("logits", "index"):
        raise ValueError(f"unknown output mode {output_mode!r}")
    # M1 fresh input + M3 output are approximately 336 KiB + 128 KiB.
    base_bytes = (336 + 128) * 1024
    # §12: each hybrid path is approximately 550–650 KiB complete (both
    # directions), excluding the small O(n_k*b) OT response.  Use 600 KiB.
    hybrid_bytes = hybrid_layers * 600 * 1024
    # b=16 bits per activation element; this is a target-model estimate.
    ot_bytes = (mpc_elements * 16 + 7) // 8
    return ClientCost(
        online_rounds=2 + hybrid_layers + int(output_mode == "index"),
        output_mode=output_mode,
        encrypt_count=1 + hybrid_layers,
        decrypt_count=1 + hybrid_layers,
        hybrid_layers=hybrid_layers,
        mpc_elements=mpc_elements,
        base_bytes=base_bytes,
        hybrid_bytes=hybrid_bytes,
        ot_bytes=ot_bytes,
        total_bytes=base_bytes + hybrid_bytes + ot_bytes,
        zero_interaction_fallback=hybrid_layers == 0,
    )


def as_dict(cost: ClientCost) -> dict:
    return asdict(cost)
