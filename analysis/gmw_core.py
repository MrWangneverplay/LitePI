"""GMW: AND gates from 1-out-of-4 OT, plus a comparison circuit.

Two parties hold XOR-shares of bits (party0 holds s0, party1 holds s1, and the
real bit is s0 ^ s1).  In GMW:
  * XOR is free  - each party XORs its own shares locally;
  * AND costs one 1-out-of-4 OT.
Anything expressible as a boolean circuit (comparison, addition, argmax, ...)
can therefore be evaluated without either party learning the inputs.

This replaces the local simulation in hybrid_mpc.mpc_relu.  It is pure Python
and reuses the 1-out-of-2 OT from ot_core.py (Chou-Orlandi over SECP256K1).

Run the self-test:
    python gmw_core.py --self-test
"""

from __future__ import annotations

import argparse
import secrets

try:  # OpenSSL-accelerated P-256 base OT (same protocol, ~400x faster)
    from ot_core_fast import receiver_step1, receiver_step2, sender_step1, sender_step2
except Exception:  # pragma: no cover - fall back to the pure-Python secp256k1 core
    from ot_core import receiver_step1, receiver_step2, sender_step1, sender_step2


# --------------------------------------------------------- 1-out-of-2 (in memory)

_OT_COUNT = 0


def reset_ot_count() -> None:
    """Reset the base 1-of-2 OT counter (benchmarking only)."""
    global _OT_COUNT
    _OT_COUNT = 0


def ot_count() -> int:
    """Number of base 1-of-2 OTs executed since the last reset."""
    return _OT_COUNT


def _ot_1of2_inproc(m0: bytes, m1: bytes, c: int) -> bytes:
    """One in-process 1-out-of-2 OT; receiver with choice c gets m_c."""
    global _OT_COUNT
    _OT_COUNT += 1
    a, A = sender_step1()
    b, B = receiver_step1(c, A)
    e0, e1 = sender_step2(a, A, B, m0, m1)
    return receiver_step2(b, A, c, e0, e1)


def _bxor(x: bytes, y: bytes) -> bytes:
    return bytes(p ^ q for p, q in zip(x, y))


# --------------------------------------------------------- 1-out-of-4

def ot_1of4(m00: bytes, m01: bytes, m10: bytes, m11: bytes, x: int, y: int) -> bytes:
    """1-out-of-4 OT built from two 1-out-of-2 OTs.

    Messages are indexed m[a][b].  The receiver holds choice bits (x, y) and
    learns exactly m[x][y]:
      1. OT on x delivers a random pad r_x;
      2. OT on y delivers a table column in which only the row x can be unmasked
         (the other row is masked with the pad the receiver does NOT have).
    """
    assert {x, y} <= {0, 1} and len({len(m) for m in (m00, m01, m10, m11)}) == 1
    n = len(m00)
    r0, r1 = secrets.token_bytes(n), secrets.token_bytes(n)

    got_r = _ot_1of2_inproc(r0, r1, x)                       # receiver learns r_x

    col_y0 = _bxor(m00, r0) + _bxor(m10, r1)                 # column y = 0
    col_y1 = _bxor(m01, r0) + _bxor(m11, r1)                 # column y = 1
    got_col = _ot_1of2_inproc(col_y0, col_y1, y)             # receiver learns column y

    return _bxor(got_col[x * n:(x + 1) * n], got_r)


# --------------------------------------------------------- GMW AND gate

def gmw_and(a0: int, b0: int, a1: int, b1: int):
    """One GMW AND gate.  Party0 holds (a0, b0), party1 holds (a1, b1).

    Returns (c0, c1) with  c0 ^ c1 == (a0^a1) & (b0^b1).
    """
    r = secrets.randbits(1)                                  # party0's output share
    # Truth table indexed by party1's view; party0's shares are folded in.
    table = {}
    for x in (0, 1):
        for y in (0, 1):
            table[(x, y)] = bytes([((a0 ^ x) & (b0 ^ y)) ^ r])
    got = ot_1of4(table[(0, 0)], table[(0, 1)], table[(1, 0)], table[(1, 1)], a1, b1)
    return r, got[0]


# --------------------------------------------------------- wires & circuit

# A "wire" is a pair (s0, s1): the two parties' XOR-shares of one bit.
def const(v: int):
    return (v, 0)


def wxor(*ws):
    r0 = r1 = 0
    for s0, s1 in ws:
        r0 ^= s0
        r1 ^= s1
    return (r0, r1)


def wnot(w):
    return wxor(w, const(1))


def wand(w1, w2):
    c0, c1 = gmw_and(w1[0], w2[0], w1[1], w2[1])
    return (c0, c1)


def wor(*ws):
    acc = ws[0]
    for w in ws[1:]:
        acc = wxor(wand(acc, w), acc, w)      # u or v = (u and v) xor u xor v
    return acc


def reveal(w) -> int:
    """Reconstruct the plaintext bit (only used by the test harness)."""
    return w[0] ^ w[1]


def less_than(a_bits, b_bits):
    """Unsigned comparison a < b, MSB first.  a_bits/b_bits are lists of wires."""
    assert len(a_bits) == len(b_bits)
    less = const(0)
    eq_prefix = const(1)
    for ai, bi in zip(a_bits, b_bits):        # MSB -> LSB
        lt_i = wand(wnot(ai), bi)             # a_i = 0, b_i = 1
        less = wor(less, wand(eq_prefix, lt_i))
        eq_prefix = wand(eq_prefix, wnot(wxor(ai, bi)))   # higher bits still equal
    return less


def bits_of(value: int, k: int):
    """Split `value` into k wires (MSB first), shares chosen at random."""
    out = []
    for i in range(k - 1, -1, -1):
        bit = (value >> i) & 1
        s0 = secrets.randbits(1)
        out.append((s0, s0 ^ bit))
    return out


# --------------------------------------------------------- self test

def self_test() -> int:
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    # 1. 1-out-of-4 OT: all four choices
    msgs = [bytes([10]), bytes([20]), bytes([30]), bytes([40])]
    ok = True
    for x in (0, 1):
        for y in (0, 1):
            got = ot_1of4(msgs[0], msgs[1], msgs[2], msgs[3], x, y)
            expect = msgs[2 * x + y]
            ok = ok and got == expect
    check("1-out-of-4 OT (all four choices)", ok)

    # 2. GMW AND gate agrees with plaintext AND
    ok = True
    for _ in range(8):
        a = secrets.randbits(1)
        b = secrets.randbits(1)
        a0, a1 = secrets.randbits(1), 0
        a1 = a ^ a0
        b0 = secrets.randbits(1)
        b1 = b ^ b0
        c0, c1 = gmw_and(a0, b0, a1, b1)
        ok = ok and ((c0 ^ c1) == (a & b))
    check("GMW AND gate matches plaintext AND", ok)

    # 3. comparison circuit agrees with plaintext <
    k = 4
    ok = True
    detail = ""
    for _ in range(3):
        a = secrets.randbelow(2 ** k)
        b = secrets.randbelow(2 ** k)
        lt = less_than(bits_of(a, k), bits_of(b, k))
        got = reveal(lt)
        if got != int(a < b):
            ok = False
            detail = f"a={a} b={b} got={got} expect={int(a < b)}"
    check(f"{k}-bit comparison circuit matches plaintext", ok, detail)

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="GMW AND gates + comparison circuit")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    print("Run with --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
