"""Regression test for output mode A (logits) and B (secure index)."""

from mask_session import MaskDomain
from pi_client import loopback_once


def main() -> int:
    # Mode B needs a rule-v2 (statistically bounded) output domain; a bare
    # MaskDomain(lo, hi) is refused by the server since the M2 fix.
    domains = {1: MaskDomain.for_bound(32.0)}
    logits = loopback_once(
        "stub", [16, 8, 4], schedule=(0,),
        output_mode="logits",
    )
    index = loopback_once(
        "stub", [16, 8, 4], schedule=(0,),
        mask_domains=domains, output_mode="index",
    )
    assert logits["loopback"]["passed"], logits
    assert index["loopback"]["passed"], index
    assert index["output_mode"] == "index"
    assert "logits" not in index
    assert index["server"]["protocol_online_rounds"] == 3
    print("PASS  mode A: logits")
    print("PASS  mode B: class index only, +1 logical round")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
