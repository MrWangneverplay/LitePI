"""Regression test for the three configuration points of one protocol."""

from pi_client import loopback_once


def main() -> int:
    cases = [
        ([16, 8, 4], (0,), 0, 2),
        ([16, 8, 4], (1,), 1, 3),
        ([32, 16, 8, 4], (0, 1), 1, 3),
    ]
    for dims, schedule, expected_hybrid, expected_rounds in cases:
        result = loopback_once("stub", dims, schedule=schedule)
        assert result["loopback"]["passed"], (dims, schedule, result)
        assert result["hybrid_layers"] == expected_hybrid
        assert result["server"]["protocol_online_rounds"] == expected_rounds
        # §4 r_h=1 witness: exactly one ①_k -> ②_k exchange per hybrid layer.
        assert result["server"]["hybrid_roundtrips"] == expected_hybrid
        if expected_hybrid:
            assert abs(result["server"]["hybrid_r_h"] - 1.0) < 1e-9
        print(
            f"PASS  schedule={schedule} hybrid={expected_hybrid} "
            f"online_rounds={expected_rounds}"
        )
    obfuscated = loopback_once(
        "stub", [32, 16, 8, 4], schedule=(0, 1),
        obfuscate_schedule=True,
    )
    assert obfuscated["server"]["protocol_online_rounds"] == 4
    assert obfuscated["server"]["schedule_obfuscated"]
    print("PASS  §7.2 dummy rounds hide a mixed schedule")
    print("ALL MODES PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
