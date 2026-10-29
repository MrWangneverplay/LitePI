"""Self-test for the M1–M3 mask session and the M2 v2 domain rule."""

import math

from mask_session import (
    RULE_V1,
    RULE_V2,
    MaskDomain,
    MaskSession,
    MaskSessionError,
    assert_mask_session_safe,
    mask_security_report,
    power_of_two_bound,
)
from params_pi import F, MASK_MAX_BITS, MASK_SIGMA_TARGET


def main() -> int:
    # M1 + M3 ---------------------------------------------------------------
    assert_mask_session_safe(MaskSession(seed=b"test-seed-32-bytes-long"))

    # Explicit (legacy) domains still draw inside their bounds but do not
    # count as M2-aligned.
    explicit = MaskSession(domains={0: MaskDomain(-2.0, 4.0),
                                    1: MaskDomain(-8.0, 3.0)})
    assert not explicit.covers([0, 1])
    explicit.begin_query(0)
    layer0 = explicit.draw(0, 0, 128)
    layer1 = explicit.draw(0, 1, 128)
    assert layer0 != layer1
    assert min(layer1) >= -8.0 and max(layer1) < 3.0
    explicit.end_query(0)
    explicit.begin_query(1)
    assert explicit.draw(1, 0, 128) != layer0
    explicit.end_query(1)
    print("PASS  M1 fresh (query, layer) masks; M3 rejects query gaps")

    # M2 v2 rule --------------------------------------------------------------
    assert power_of_two_bound(10.8) == 16.0          # 1.25*10.8 = 13.5 -> 16
    assert power_of_two_bound(89.9) == 128.0
    d16 = MaskDomain.for_bound(16.0)
    assert d16.rule == RULE_V2
    assert d16.sigma == min(MASK_SIGMA_TARGET, MASK_MAX_BITS - 4)
    assert d16.half_width == 2.0 ** (4 + d16.sigma)
    assert math.isclose(d16.statistical_distance(), 2.0 ** -(d16.sigma + 1))
    d_small = MaskDomain.for_bound(2.0 ** -3)
    assert d_small.sigma == MASK_SIGMA_TARGET          # far from the cap
    try:
        MaskDomain.for_bound(2.0 ** 30)                 # sigma would be 12
    except ValueError:
        pass
    else:
        raise AssertionError("a bound that cannot reach MIN_SIGMA was accepted")
    try:
        MaskDomain(-(2.0 ** 50), 2.0 ** 50)
    except ValueError:
        pass
    else:
        raise AssertionError("a domain beyond the precision cap was accepted")

    aligned = MaskSession(seed=b"x" * 32,
                          domains={0: d16, 1: MaskDomain.for_bound(128.0)})
    assert aligned.covers([0, 1]) and not aligned.covers([2])
    aligned.begin_query(0)
    r = aligned.draw(0, 0, 4096)
    scale = 1 << F
    assert all(float(v * scale).is_integer() for v in r), "mask off the 2^-F grid"
    assert min(r) >= d16.lo and max(r) < d16.hi
    # Coarse uniformity: both halves and the outer quarters are populated.
    assert 0.45 < sum(v < 0 for v in r) / len(r) < 0.55
    assert sum(abs(v) > d16.half_width / 2 for v in r) > 0.4 * len(r)
    aligned.end_query(0)
    print("PASS  M2 v2: A_k = 2^sigma_k * B_k, grid-exact uniform draws, "
          "precision cap enforced")

    # v1 -> v2 upgrade and the v1 control ------------------------------------
    v1_entry = {"min": -9.99, "max": 5.42, "lo": -10.76, "hi": 6.19}
    up = MaskDomain.from_mapping(v1_entry)
    assert up.rule == RULE_V2 and up.activation_bound == 16.0
    old = MaskDomain.from_mapping(v1_entry, rule="v1")
    assert old.rule == RULE_V1 and old.lo == -10.76
    v2_entry = up.to_dict() | {"min": -9.99, "max": 5.42}
    assert MaskDomain.from_mapping(v2_entry).half_width == up.half_width
    assert MaskDomain.from_mapping(v2_entry, rule="v1").hi < 7.0
    print("PASS  v1 files upgrade in memory; rule='v1' reproduces the old control")

    # Security accounting -------------------------------------------------------
    rep = mask_security_report({0: d16, 1: MaskDomain.for_bound(64.0)},
                               {0: 256, 1: 64}, queries=1000)
    expected = 256 * 2.0 ** -39 + 64 * 2.0 ** -37
    assert math.isclose(rep["sd_per_query"], expected)
    assert rep["q_max"]["20"] == math.floor(2.0 ** -20 / expected)
    print(f"PASS  SD/query = 2^{rep['log2_sd_per_query']:.1f}, "
          f"Q_max(2^-20) = {rep['q_max']['20']}")

    try:
        aligned.begin_query(5)
    except MaskSessionError:
        pass
    else:
        raise AssertionError("query gap accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
