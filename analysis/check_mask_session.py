"""会话随机材料消费、查询同步及 M2 域规则自检，非完整协议安全证明。"""

import math
from unittest.mock import patch

from mask_session import (
    RULE_V1,
    RULE_V2,
    RULE_V3,
    MaskDomain,
    MaskSession,
    MaskSessionError,
    assert_mask_session_safe,
    mask_security_report,
    power_of_two_bound,
    widen_domains,
)
from params_pi import F, MASK_MAX_BITS, MASK_SIGMA_TARGET


def assert_rejected(action, error_type=MaskSessionError) -> None:
    try:
        action()
    except error_type:
        pass
    else:
        raise AssertionError("应拒绝的请求被接受")


def assert_closed(session: MaskSession) -> None:
    assert session._closed and session._active_query is None, "会话未关闭"
    next_query = session.next_query
    consumed = set(session._consumed_layers)
    layer_id = next(iter(consumed), 0)
    for action in (
        lambda: session.begin_query(0),
        lambda: session.begin_query(next_query),
        lambda: session.end_query(None),
        lambda: session.end_query(0),
        lambda: session.draw(0, 0, 1),
        lambda: session.draw_ints(0, 0, 1),
        lambda: session.draw(0, layer_id, 1),
        lambda: session.draw_ints(0, layer_id, 1),
    ):
        assert_rejected(action)
        assert session.next_query == next_query, "关闭后查询计数器被改变"
        assert session._consumed_layers == consumed, "关闭后消费集合被改变"
        assert session._closed and session._active_query is None


def check_consumption_guard() -> None:
    codecs = ("draw", "draw_ints")
    alternate = MaskDomain(-2.0, 4.0)
    for first in codecs:
        for second in codecs:
            for size, domain in ((1, None), (2, None), (1, alternate),
                                 (2, alternate), (False, object())):
                session = MaskSession(seed=b"c" * 32)
                session.begin_query(0)
                getattr(session, first)(0, 7, 1)
                with patch("mask_session.hashlib.shake_256",
                           side_effect=AssertionError("重复请求进入派生")):
                    assert_rejected(
                        lambda: getattr(session, second)(0, 7, size, domain)
                    )
                    assert session._consumed_layers == {7}
                    assert_closed(session)
    print("PASS  共用 layer 消费空间：同 codec、跨 codec、size/domain 变化均拒绝")

    bad_ids = (True, False, 0.0, 1.5, "0", None, -1, 1 << 64)
    for bad in bad_ids:
        session = MaskSession(seed=b"i" * 32)
        assert_rejected(lambda: session.begin_query(bad))
        assert_closed(session)
        session = MaskSession(seed=b"i" * 32)
        session.begin_query(0)
        assert_rejected(lambda: session.end_query(bad))
        assert_closed(session)
        for codec in codecs:
            for query_id, layer_id in ((bad, 0), (0, bad)):
                session = MaskSession(seed=b"i" * 32)
                session.begin_query(0)
                assert_rejected(
                    lambda: getattr(session, codec)(query_id, layer_id, 1)
                )
                assert not session._consumed_layers
                assert_closed(session)
    for codec in codecs:
        for bad_size in (True, False, 0, -1, 1.0, "1", None):
            session = MaskSession(seed=b"i" * 32)
            session.begin_query(0)
            assert_rejected(lambda: getattr(session, codec)(0, 0, bad_size),
                            ValueError)
            assert not session._consumed_layers
            assert_closed(session)
        invalid_domain = MaskDomain(-1.0, 1.0)
        object.__setattr__(invalid_domain, "lo", float("nan"))
        for bad_domain in (False, object(), invalid_domain):
            for source in ("override", "stored", "default"):
                session = MaskSession(
                    seed=b"i" * 32,
                    domains={0: bad_domain} if source == "stored" else None,
                    default_domain=bad_domain if source == "default" else alternate,
                )
                session.begin_query(0)
                override = bad_domain if source == "override" else None
                with patch("mask_session.hashlib.shake_256",
                           side_effect=AssertionError("无效域进入派生")):
                    assert_rejected(
                        lambda: getattr(session, codec)(0, 0, 1, override),
                        ValueError,
                    )
                    assert not session._consumed_layers, "域校验前已占用 layer"
                    assert_closed(session)
    print("PASS  query/layer uint64 非 bool、size 正整数与 domain 校验失败即关闭")

    for codec in codecs:
        for state in ("inactive", "finished", "wrong-query"):
            session = MaskSession(seed=b"o" * 32)
            if state != "inactive":
                session.begin_query(0)
            if state == "finished":
                session.end_query(0)
            query_id = 1 if state == "wrong-query" else 0
            assert_rejected(lambda: getattr(session, codec)(query_id, 0, 1))
            assert not session._consumed_layers
            assert_closed(session)
    for action in (lambda s: s.begin_query(0), lambda s: s.end_query(1)):
        session = MaskSession(seed=b"o" * 32)
        session.begin_query(0)
        assert_rejected(lambda: action(session))
        assert_closed(session)
    session = MaskSession(seed=b"o" * 32)
    assert_rejected(lambda: session.end_query(None))
    assert_closed(session)
    session = MaskSession(seed=b"o" * 32)
    session.begin_query(0)
    session.draw(0, 0, 1)
    session.abort()
    session.abort()
    assert_closed(session)
    print("PASS  outside-active、未完成查询和显式 abort 不可由 end/begin 恢复")

    for codec, output in (("draw", "_grid_draw"), ("draw_ints", "_grid_ints")):
        for stage in ("derive", "digest", "output"):
            for error_type in (RuntimeError, KeyboardInterrupt):
                session = MaskSession(seed=b"e" * 32)
                session.begin_query(0)
                target = (f"mask_session.{output}" if stage == "output"
                          else "mask_session.hashlib.shake_256")
                failure = error_type("注入随机材料派生或输出失败")
                with patch(target) as mocked:
                    if stage == "digest":
                        mocked.return_value.digest.side_effect = failure
                    else:
                        mocked.side_effect = failure
                    assert_rejected(lambda: getattr(session, codec)(0, 3, 1),
                                    error_type)
                    assert session._consumed_layers == {3}, "派生前未占用 layer"
                    mocked.assert_called_once()
                    if stage == "digest":
                        mocked.return_value.digest.assert_called_once()
                    assert_closed(session)
                    mocked.assert_called_once()
    print("PASS  派生、digest 和输出异常（含中断）均关闭，不得重试复用")

    session = MaskSession(seed=b"b" * 32)
    session.begin_query(0)
    session.draw(0, (1 << 64) - 1, 1)
    session.draw_ints(0, 0, 1)
    session.end_query(0)
    session.begin_query(1)
    assert not session._consumed_layers, "新查询未重置消费集合"
    session.draw_ints(1, (1 << 64) - 1, 1)
    session.draw(1, 0, 1)
    session.end_query(1)
    session._next_query = (1 << 64) - 1
    session.begin_query((1 << 64) - 1)
    session.draw((1 << 64) - 1, 0, 1)
    session.draw_ints((1 << 64) - 1, (1 << 64) - 1, 1)
    session.end_query((1 << 64) - 1)
    assert session.next_query == 1 << 64, "查询计数器溢出回绕"
    assert_closed(session)
    session = MaskSession(seed=b"b" * 32)
    session._next_query = 1 << 64
    assert_rejected(lambda: session.begin_query(0))
    assert session.next_query == 1 << 64
    assert_closed(session)
    print("PASS  串行混合 codec、跨查询 layer 复用与 uint64 上界；溢出关闭无回绕")


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
    check_consumption_guard()

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

    # Rule v3 (double-double codec): wider, and drawn as integers --------------
    wide = MaskDomain.widened(16.0)
    assert wide.rule == RULE_V3 and wide.sigma == 70
    assert wide.log2_half_width == 74
    capped = MaskDomain.widened(2.0 ** 30)
    assert capped.log2_half_width == 90, "v3 cap not applied"
    session = MaskSession(seed=b"s" * 32, domains={0: wide})
    assert session.covers([0]), "v3 domains count as M2-aligned"
    session.begin_query(0)
    ints = session.draw_ints(0, 0, 2000)
    bound = 1 << (74 + F)
    assert all(-bound <= k < bound for k in ints)
    assert max(abs(k) for k in ints) > bound // 2, "mask does not fill the domain"
    assert len(set(ints)) == len(ints), "v3 masks repeat"
    # Low bits matter: the ring share is k mod 2^44, not a rounded double.
    assert len({k & 0xFF for k in ints}) > 200, "low bits are not random"
    assert ints != session.draw_ints(0, 1, 2000), "layers share a mask"
    session.end_query(0)
    session.begin_query(1)
    assert ints != session.draw_ints(1, 0, 2000), "queries share a mask"
    session.end_query(1)
    widened = widen_domains({0: d16, 2: d16}, keep=(2,))
    assert widened[0].rule == RULE_V3 and widened[2].rule == RULE_V2
    rep = mask_security_report({0: MaskDomain.widened(16.0),
                                1: MaskDomain.widened(64.0)}, {0: 256, 1: 64})
    assert math.isclose(rep["sd_per_query"], 320 * 2.0 ** -71)
    print(f"PASS  M2 v3: sigma=70, integer draws, SD/query = "
          f"2^{rep['log2_sd_per_query']:.1f}, Q_max(2^-40) = {rep['q_max']['40']}")

    try:
        aligned.begin_query(5)
    except MaskSessionError:
        pass
    else:
        raise AssertionError("query gap accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
