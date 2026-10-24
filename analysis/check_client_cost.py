"""Self-test for the documented §12 client-cost model."""

from client_cost import estimate


def main() -> int:
    he_only = estimate(0, 0)
    one_hybrid = estimate(1, 256)
    two_hybrid = estimate(2, 64)
    index_mode = estimate(0, 0, "index")
    assert (he_only.online_rounds, he_only.encrypt_count,
            he_only.decrypt_count) == (2, 1, 1)
    assert (one_hybrid.online_rounds, one_hybrid.encrypt_count,
            one_hybrid.decrypt_count) == (3, 2, 2)
    assert (two_hybrid.online_rounds, two_hybrid.encrypt_count,
            two_hybrid.decrypt_count) == (4, 3, 3)
    assert he_only.zero_interaction_fallback
    assert index_mode.online_rounds == he_only.online_rounds + 1
    print("PASS  §12 client cost: 1 enc/dec + 1 per hybrid; 2+Kh rounds")
    print("PASS  §9 output mode B adds one logical round")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
