"""Real (non-simulated) secure ReLU on additive shares.

Built on ot_core.py (Chou-Orlandi OT) and gmw_core.py (GMW AND gates / wires).
Nothing is ever reconstructed outside the self-test: each party only ever sees
its own share.

Pipeline for z = s_c + s_s (mod 2^k, two's complement fixed point):

  1. sign     b = 1 iff z >= 0, as a XOR-shared bit.  secure_relu() evaluates only
              the carry chain of s_c + s_s over the low k_compare bits (drelu_slim);
              a2b()/drelu_wire() (full ripple-carry adder) are kept for reference
              and for the self-test OT-count comparison.
  2. z * b    additive shares of z*b: two single-OT "bit-select" multiplications
              (ot_xorbit_mult), one per addend.
  3. output   additive shares of ReLU(z) = (c_c, c_s)

Cost per ReLU (slim path): the carry chain uses 1 private-input AND plus
(k-2) majority gates, i.e. 1 + 2(k-2) = 2k-3 base OTs; the two bit-select
multiplications add 2.  Total = 2k-1 base OTs (k=14 -> 27, versus 58 for the
full-adder version).

Run the self-test:
    python mpc_relu_secure.py --self-test
"""

from __future__ import annotations

import argparse
import secrets

from gmw_core import _ot_1of2_inproc, const, ot_count, reset_ot_count, wand, wnot, wxor


# ---------------------------------------------------------------- helpers

def _nb(k: int) -> int:
    return (k + 7) // 8


def _enc(v: int, k: int) -> bytes:
    return (v % (1 << k)).to_bytes(_nb(k), "big")


def _dec(b: bytes, k: int) -> int:
    return int.from_bytes(b, "big") % (1 << k)


def _bits_of_private(v: int, k: int, party: int):
    """Wire vector (LSB first) of an integer known only to `party`."""
    out = []
    for i in range(k):
        bit = (v >> i) & 1
        out.append((bit, 0) if party == 0 else (0, bit))
    return out


# ---------------------------------------------------------------- circuits

def gmw_add(a_bits, b_bits, k: int):
    """Ripple-carry adder over XOR-shared bits (LSB first).  ~3 AND gates/bit."""
    carry = const(0)
    out = []
    for i in range(k):
        axb = wxor(a_bits[i], b_bits[i])
        out.append(wxor(axb, carry))                 # sum bit
        carry = wxor(wand(a_bits[i], b_bits[i]), wand(carry, axb))  # carry out
    return out


def a2b(s_c: int, s_s: int, k: int):
    """Arithmetic -> binary: additive shares (mod 2^k) to XOR bit-shares of the sum."""
    return gmw_add(_bits_of_private(s_c, k, 0), _bits_of_private(s_s, k, 1), k)


def drelu_wire(s_c: int, s_s: int, k: int):
    """Shared bit b with b = 1 iff (s_c + s_s) mod 2^k is non-negative."""
    bits = a2b(s_c, s_s, k)
    return wnot(bits[k - 1])          # sign bit flipped: >= 0 means not negative


# ---------------------------------------------------------------- OT multiplication

def ot_mul_bit(bit0: int, bit1: int):
    """One party knows bit0, the other bit1.

    Returns additive shares of bit0*bit1 that satisfy  share0 + share1 == bit0*bit1
    *as integers* (each share may be negative).  Reducing mod 2 here would wrap
    share0 + share1 by 2, and the caller doubles the shares when rebuilding
    b = b0 + b1 - 2*b0*b1, which would amplify that wrap into a wrong result.
    """
    rho = secrets.randbits(1)
    got = _ot_1of2_inproc(bytes([rho]), bytes([rho + bit0]), bit1)
    return -rho, got[0]


def ot_mul_gilboa(x0: int, y1: int, k: int):
    """Gilboa OT multiplication: party0 knows x0, party1 knows y1 (k-bit each).

    Returns (share_p0, share_p1) with share_p0 + share_p1 == x0 * y1  (mod 2^k).
    One 1-out-of-2 OT per bit of y1; party0 masks each partial product with a
    fresh pad, so party1 only ever sees padded values.
    """
    total_pad = 0
    p1_share = 0
    for j in range(k):
        rho = secrets.randbelow(1 << k)
        total_pad = (total_pad + rho) % (1 << k)
        msg0 = _enc(rho, k)
        msg1 = _enc(rho + x0 * (1 << j), k)
        got = _ot_1of2_inproc(msg0, msg1, (y1 >> j) & 1)
        p1_share = (p1_share + _dec(got, k)) % (1 << k)
    return (-total_pad) % (1 << k), p1_share


def ot_xorbit_mult(x: int, b_self: int, b_other: int, k: int):
    """Multiply a value by a XOR-shared bit in ONE 1-of-2 OT.

    "self" holds the value x and one share b_self of the bit; "other" holds the
    other share b_other.  Returns (self_share, other_share) with
    self_share + other_share == x * (b_self xor b_other)  (mod 2^k).

    Trick: the sender offers x * (b_self xor 0) and x * (b_self xor 1); the
    receiver picks by b_other, so it receives x * (b_self xor b_other) masked by
    a fresh pad.  This replaces the k-OT Gilboa loop with a single OT whose
    messages are k bits wide.
    """
    rho = secrets.randbelow(1 << k)
    m0 = (rho + x * (b_self ^ 0)) % (1 << k)
    m1 = (rho + x * (b_self ^ 1)) % (1 << k)
    got = _ot_1of2_inproc(_enc(m0, k), _enc(m1, k), b_other)
    return (-rho) % (1 << k), _dec(got, k)


# ---------------------------------------------------------------- the protocol

def ot_and_private(a0: int, b1: int):
    """AND of two bits held by different parties (party0=a0, party1=b1).

    Each party holds one input bit, so a single base 1-of-2 OT suffices
    (unlike a general GMW AND, which needs a 1-of-4 OT = 2 base OTs).
    """
    rho = secrets.randbits(1)
    got = _ot_1of2_inproc(bytes([rho]), bytes([rho ^ a0]), b1)
    return rho, got[0]


def maj_wire(a, b, c):
    """majority(a,b,c) as a wire: ((a xor c) & (b xor c)) xor c.

    One general GMW AND per carry, instead of the two ANDs of the full-adder
    carry recurrence.
    """
    return wxor(wand(wxor(a, c), wxor(b, c)), c)


def drelu_slim(s_c: int, s_s: int, k: int):
    """Shared bit b with b = 1 iff (s_c + s_s) mod 2^k is non-negative.

    Only the carry chain is evaluated:
        c_1 = a_0 & b_0              (1 private-input AND, 1 base OT)
        c_{i+1} = maj(a_i,b_i,c_i)   (k-2 majority gates, 2 base OTs each)
        sign = a_{k-1} xor b_{k-1} xor c_{k-1}
    Base-OT count for the sign: 1 + 2(k-2) = 2k-3.
    """
    assert k >= 2, "slim DReLU needs at least 2 bits"
    a_bits = _bits_of_private(s_c, k, 0)
    b_bits = _bits_of_private(s_s, k, 1)
    carry = ot_and_private(a_bits[0][0], b_bits[0][1])
    for i in range(1, k - 1):
        carry = maj_wire(a_bits[i], b_bits[i], carry)
    sign = wxor(a_bits[k - 1], b_bits[k - 1], carry)
    return wnot(sign)


def secure_relu(s_c: int, s_s: int, k: int, k_compare: int | None = None):
    """Secure ReLU on additive shares.  Returns (c_c, c_s) with c_c + c_s == ReLU(z).

    k       : ring bit width of the shares s_c, s_s and of the output shares.
    k_compare : bit width used for the sign comparison (the A2B window).  Must be
                <= k.  When |z| < 2^(k_compare-1), only the low k_compare bits are
                needed to decide the sign, so the AND-gate count scales with
                k_compare while the multiplication ring stays k bits wide.
    """
    if k_compare is None:
        k_compare = k
    assert 1 <= k_compare <= k
    mask = (1 << k_compare) - 1
    b0, b1 = drelu_slim(s_c & mask, s_s & mask, k_compare)

    # z * b = (s_c + s_s) * (b0 xor b1), via two single-OT "bit-selects".
    a0, a1 = ot_xorbit_mult(s_c, b0, b1, k)        # shares of s_c * b (self=party0)
    d1, d0 = ot_xorbit_mult(s_s, b1, b0, k)        # shares of s_s * b (self=party1)

    c_c = (a0 + d0) % (1 << k)
    c_s = (a1 + d1) % (1 << k)
    return c_c, c_s


# ---------------------------------------------------------------- self test

def self_test(k: int = 6) -> int:
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    # 1. A2B correctness
    ok = True
    detail = ""
    for _ in range(3):
        v = secrets.randbelow(1 << k)
        s_c = secrets.randbelow(1 << k)
        s_s = (v - s_c) % (1 << k)
        bits = a2b(s_c, s_s, k)
        rec = sum((b0 ^ b1) << i for i, (b0, b1) in enumerate(bits))
        if rec != v:
            ok = False
            detail = f"v={v} got={rec}"
    check("A2B: additive shares -> bit shares", ok, detail)

    # 1b. OT count: full-adder A2B versus the slim carry chain
    s_c = secrets.randbelow(1 << k)
    s_s = secrets.randbelow(1 << k)
    mask = (1 << k) - 1
    reset_ot_count()
    bits = a2b(s_c & mask, s_s & mask, k)
    _ = wnot(bits[k - 1])
    old_ots = ot_count()
    reset_ot_count()
    _ = drelu_slim(s_c & mask, s_s & mask, k)
    new_ots = ot_count()
    reset_ot_count()
    _ = secure_relu(s_c & mask, s_s & mask, k, k)
    full_ots = ot_count()
    check("slim DReLU uses fewer base OTs", new_ots < old_ots,
          f"old_sign={old_ots} new_sign={new_ots} full_slim={full_ots}")

    # 1c. exact OT-count guard: one ReLU = 1 + 2*(kc-2) carry-chain OTs + 2 bit-selects
    #     = 2*kc - 1  (kc = 14 -> 27).  Fails loudly if the circuit ever changes.
    ok, detail = True, ""
    for k_, kc_ in ((8, 2), (10, 6), (20, 14)):
        reset_ot_count()
        secure_relu(secrets.randbelow(1 << k_), secrets.randbelow(1 << k_), k_, kc_)
        if ot_count() != 2 * kc_ - 1:
            ok, detail = False, f"k={k_} kc={kc_}: {ot_count()} OTs, expected {2 * kc_ - 1}"
    check("OT count per ReLU == 2*k_compare - 1 (k_compare=14 -> 27)", ok, detail)

    # 2. secure_relu matches plaintext ReLU (signed two's complement)
    half = 1 << (k - 1)
    cases = [half - 1, 1, 0, (1 << k) - 1, (1 << k) - (half // 2)]   # +max, 1, 0, -1, -big
    ok = True
    detail = ""
    for z in cases:
        s_c = secrets.randbelow(1 << k)
        s_s = (z - s_c) % (1 << k)
        c_c, c_s = secure_relu(s_c, s_s, k)
        got = (c_c + c_s) % (1 << k)
        signed = z - (1 << k) if z >= half else z
        expect = max(signed, 0)
        if got != expect:
            ok = False
            detail = f"z={signed} got={got} expect={expect}"
    check("secure_relu matches plaintext ReLU", ok, detail)

    # 3. no-leak spot check: neither party's output share equals the answer
    ok = True
    for z in cases[1:3]:
        s_c = secrets.randbelow(1 << k)
        s_s = (z - s_c) % (1 << k)
        c_c, c_s = secure_relu(s_c, s_s, k)
        answer = max(z - (1 << k) if z >= half else z, 0)
        if answer != 0 and (c_c == answer or c_s == answer):
            ok = False
    check("no-leak spot check (single share != answer)", ok)

    # 4. narrow-window sign: big ring k, small comparison window k_compare
    K = 10
    KC = 6
    ok = True
    detail = ""
    for z in range(-31, 32):
        ring = z % (1 << K)
        s_c = secrets.randbelow(1 << K)
        s_s = (ring - s_c) % (1 << K)
        c_c, c_s = secure_relu(s_c, s_s, K, KC)
        got = (c_c + c_s) % (1 << K)
        expect = max(z, 0)
        if got != expect:
            ok = False
            detail = f"z={z} got={got} expect={expect}"
            break
    check(f"secure_relu with narrow window (k={K}, k_compare={KC})", ok, detail)

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Secure ReLU on additive shares")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bits", type=int, default=6, help="bit width k (self-test)")
    args = ap.parse_args()
    if args.self_test:
        return self_test(args.bits)
    print("Run with --self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
