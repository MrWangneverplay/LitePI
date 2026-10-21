"""Prototype: minimal-OT secure ReLU (stand-alone reference, kept for comparison).

This is the stand-alone prototype that the slim circuit in the main code mirrors:
    analysis/mpc_relu_secure.py   drelu_slim / ot_and_private / maj_wire
    analysis/secure_relu_net.py   d_drelu_slim / d_ot_and_private / d_maj
The main code uses the repo's own wire helpers (the general AND there is a
1-of-4 OT = 2 base OTs); this file inlines the same arithmetic with a counting
stand-in for the OT, so it checks correctness and OT counts without any
public-key work.  It is NOT imported by anything.

How the base-OT count per ReLU is obtained (kc = k_compare >= 2):
    carry into bit 1      a_0 & b_0, bits private to different parties   1 OT
    carry into bit i+1    maj(a_i, b_i, c_i) = ((a_i^c_i) & (b_i^c_i)) ^ c_i,
      i = 1 .. kc-2       one AND of two XOR-shared bits = 2 cross terms 2 OTs each
                          -> (kc-2) * 2 = 2*kc - 4
    sign bit              a_top ^ b_top ^ carry_in(top): local, no OT        0
    z * b                 two bit-select multiplications                     2 OTs
    total                 1 + (2*kc - 4) + 2 = 2*kc - 1      (kc = 14 -> 27)
The previous ripple-carry A2B cost 2 ANDs per bit (2 OTs each) = 4*kc + 2 = 58.

Run:  python analysis/prototypes/opt_relu.py
"""
import secrets

CNT = {"n": 0}


def ot(m0, m1, c):
    """Counting stand-in for one base 1-of-2 OT: the receiver with choice c gets m_c."""
    CNT["n"] += 1
    return m1 if c else m0


def and_private(a_p0, b_p1):
    """P0 holds a, P1 holds b -> XOR-shares of a&b using ONE 1-of-2 OT."""
    r = secrets.randbits(1)
    got = ot(r, r ^ a_p0, b_p1)          # P1 learns r ^ (a&b)
    return r, got


def and_shared(x0, x1, y0, y1):
    """XOR-shared x,y -> XOR-shares of x&y using TWO 1-of-2 OTs (cross terms)."""
    r1, r2 = secrets.randbits(1), secrets.randbits(1)
    g1 = ot(r1, r1 ^ x0, y1)             # P1 learns r1 ^ x0&y1   (P0 = sender)
    g2 = ot(r2, r2 ^ x1, y0)             # P0 learns r2 ^ x1&y0   (P1 = sender)
    z0 = (x0 & y0) ^ r1 ^ g2
    z1 = (x1 & y1) ^ g1 ^ r2
    return z0, z1


def sign_shares(a, b, kc):
    """a (P0) and b (P1) are kc-bit ints.  XOR-shares of [(a+b mod 2^kc) >= 0, signed]."""
    if kc == 1:
        return (a & 1) ^ 1, (b & 1)
    c0, c1 = and_private(a & 1, b & 1)               # carry into bit 1
    for i in range(1, kc - 1):                       # carries into bits 2..kc-1
        ai, bi = (a >> i) & 1, (b >> i) & 1
        z0, z1 = and_shared(ai ^ c0, c1, c0, bi ^ c1)  # (a^c)&(b^c)
        c0, c1 = z0 ^ c0, z1 ^ c1                    # maj = ((a^c)&(b^c)) ^ c
    top = kc - 1
    return ((a >> top) & 1) ^ c0 ^ 1, ((b >> top) & 1) ^ c1


def secure_relu_opt(s_c, s_s, k, kc):
    mask = (1 << kc) - 1
    b0, b1 = sign_shares(s_c & mask, s_s & mask, kc)

    def sel(x, b_self, b_other):                     # bit-select multiplication, 1 OT
        rho = secrets.randbelow(1 << k)
        m0 = (rho + x * (b_self ^ 0)) % (1 << k)
        m1 = (rho + x * (b_self ^ 1)) % (1 << k)
        got = ot(m0, m1, b_other)
        return (-rho) % (1 << k), got

    a0, a1 = sel(s_c, b0, b1)
    d1, d0 = sel(s_s, b1, b0)
    return (a0 + d0) % (1 << k), (a1 + d1) % (1 << k)


def check(k, kc, trials=20000):
    half = 1 << (kc - 1)
    bad = 0
    for _ in range(trials):
        z = secrets.randbelow(2 * half) - half            # |z| < 2^(kc-1)
        sc = secrets.randbelow(1 << k)
        ss = ((z % (1 << k)) - sc) % (1 << k)
        CNT["n"] = 0
        cc, cs = secure_relu_opt(sc, ss, k, kc)
        if (cc + cs) % (1 << k) != max(z, 0):
            bad += 1
    return bad, CNT["n"]


if __name__ == "__main__":
    for k_, kc_ in [(26, 14), (26, 11), (20, 14), (20, 8), (12, 2)]:     # kc >= 2
        bad_, n_ = check(k_, kc_)
        print(f"k={k_} kc={kc_}: wrong={bad_}/20000  OTs/ReLU={n_}  (2*kc-1 = {2 * kc_ - 1})")
