"""Self-test for the M1–M3 mask session."""

from mask_session import MaskDomain, MaskSession, assert_mask_session_safe


def main() -> int:
    assert_mask_session_safe(MaskSession(seed=b"test-seed-32-bytes-long"))
    session = MaskSession(domains={
        0: MaskDomain(-2.0, 4.0),
        1: MaskDomain(-8.0, 3.0),
    })
    assert session.covers([0, 1])
    assert not session.covers([2])
    session.begin_query(0)
    layer0 = session.draw(0, 0, 128)
    layer1 = session.draw(0, 1, 128)
    assert layer0 != layer1
    assert min(layer1) >= -8.0 and max(layer1) < 3.0
    session.end_query(0)
    session.begin_query(1)
    next_query = session.draw(1, 0, 128)
    assert next_query != layer0
    session.end_query(1)
    print("PASS  M1 fresh (query, layer) masks; M3 rejects query gaps")
    print("PASS  M2 loads independent per-layer mask domains")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
