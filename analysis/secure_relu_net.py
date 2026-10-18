"""Two-party NETWORKED secure ReLU (real TCP sockets, two processes).

This is the deployment form of mpc_relu_secure.py: instead of the in-process
"both parties in one function" style, party0 and party1 run as SEPARATE
processes and exchange only OT messages over a TCP socket.  Neither process ever
holds the other party's share, so the activation value z is never reconstructed
anywhere.

Roles (protocol roles; the TCP listener/connector choice of the demo is
arbitrary and only exists so two terminals can find each other):
    party0 : client, OT sender, holds s_c = z + r   (draws the output mask rho)
    party1 : server, OT receiver, holds s_s = -r

Every OT round is one request/response: party0 sends A, party1 replies B, party0
sends (e0, e1).  The two sides execute the identical circuit, so message order
stays in lockstep.

Run the automated loopback test (two threads, real localhost socket):
    python analysis/secure_relu_net.py --net-self-test

Real two-process demo (two terminals):
    # terminal 1 (server / party0)
    python analysis/secure_relu_net.py --party 0 --port 9100 --share 2.5 --bits 8 --frac 3
    # terminal 2 (client / party1)
    python analysis/secure_relu_net.py --party 1 --port 9100 --share -4.0 --bits 8 --frac 3
    # each prints its own output share; relu(z) = share0 + share1, z = share0+share1 input

NOTE: like the rest of this stack it is a reference implementation.  The slim
carry-chain DReLU uses 2k-1 base OTs per ReLU (27 for k=14), down from 58.
Production still needs IKNP OT extension or a C library.
"""

from __future__ import annotations

import argparse
import secrets
import socket
import threading

try:  # OpenSSL-accelerated P-256 base OT (same protocol, ~400x faster)
    from ot_core_fast import (
        decode_point,
        encode_point,
        receiver_step1,
        receiver_step2,
        sender_step1,
        sender_step2,
    )
except Exception:  # pragma: no cover - fall back to the pure-Python secp256k1 core
    from ot_core import (
        decode_point,
        encode_point,
        receiver_step1,
        receiver_step2,
        sender_step1,
        sender_step2,
    )
from mpc_relu_secure import _dec, _enc
from params_b import F as F_B, K_COMPARE as K_COMPARE_B, K_RING as K_RING_B


def _bxor(x: bytes, y: bytes) -> bytes:
    return bytes(p ^ q for p, q in zip(x, y))


# ------------------------------------------------------------------ transport

class Chan:
    """Length-prefixed byte channel over a TCP connection."""

    def __init__(self, conn: socket.socket):
        self.conn = conn

    def send(self, data: bytes) -> None:
        self.conn.sendall(len(data).to_bytes(4, "big") + data)

    def _recv_exact(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("peer closed the connection")
            buf += chunk
        return buf

    def recv(self) -> bytes:
        n = int.from_bytes(self._recv_exact(4), "big")
        return self._recv_exact(n)


class Ctx:
    def __init__(self, party: int, chan: Chan):
        assert party in (0, 1)
        self.party = party
        self.chan = chan


# ------------------------------------------------------------- OT primitives

def d_ot_1of2(ctx: Ctx, m0: bytes | None = None, m1: bytes | None = None,
              c: int | None = None):
    """Distributed 1-out-of-2 OT.  party0 = sender (m0,m1); party1 = receiver (c)."""
    if ctx.party == 0:
        a, A = sender_step1()
        ctx.chan.send(encode_point(A))
        B = decode_point(ctx.chan.recv())
        e0, e1 = sender_step2(a, A, B, m0, m1)
        ctx.chan.send(e0)
        ctx.chan.send(e1)
        return None
    A = decode_point(ctx.chan.recv())
    b, B = receiver_step1(c, A)
    ctx.chan.send(encode_point(B))
    e0 = ctx.chan.recv()
    e1 = ctx.chan.recv()
    return receiver_step2(b, A, c, e0, e1)


def d_ot_1of2_rev(ctx: Ctx, m0: bytes | None = None, m1: bytes | None = None,
                  c: int | None = None):
    """Reversed 1-out-of-2 OT: party1 = sender (m0,m1); party0 = receiver (c)."""
    if ctx.party == 1:
        a, A = sender_step1()
        ctx.chan.send(encode_point(A))
        B = decode_point(ctx.chan.recv())
        e0, e1 = sender_step2(a, A, B, m0, m1)
        ctx.chan.send(e0)
        ctx.chan.send(e1)
        return None
    A = decode_point(ctx.chan.recv())
    b, B = receiver_step1(c, A)
    ctx.chan.send(encode_point(B))
    e0 = ctx.chan.recv()
    e1 = ctx.chan.recv()
    return receiver_step2(b, A, c, e0, e1)


def d_ot_1of4(ctx: Ctx, m00: bytes | None = None, m01: bytes | None = None,
              m10: bytes | None = None, m11: bytes | None = None,
              x: int | None = None, y: int | None = None):
    """Distributed 1-out-of-4 OT (two chained 1-of-2 OTs)."""
    if ctx.party == 0:
        n = len(m00)
        r0, r1 = secrets.token_bytes(n), secrets.token_bytes(n)
        d_ot_1of2(ctx, m0=r0, m1=r1)                       # receiver picks x -> r_x
        col_y0 = _bxor(m00, r0) + _bxor(m10, r1)
        col_y1 = _bxor(m01, r0) + _bxor(m11, r1)
        d_ot_1of2(ctx, m0=col_y0, m1=col_y1)               # receiver picks y -> column y
        return None
    got_r = d_ot_1of2(ctx, c=x)
    got_col = d_ot_1of2(ctx, c=y)
    n = len(got_r)
    return _bxor(got_col[x * n:(x + 1) * n], got_r)


# ------------------------------------------------------------- GMW AND + wires

def d_wand(ctx: Ctx, a: int, b: int) -> int:
    """Distributed GMW AND gate; each party passes its own share, gets its own."""
    if ctx.party == 0:
        r = secrets.randbits(1)
        table = {(u, v): bytes([((a ^ u) & (b ^ v)) ^ r])
                 for u in (0, 1) for v in (0, 1)}
        d_ot_1of4(ctx, table[(0, 0)], table[(0, 1)], table[(1, 0)], table[(1, 1)])
        return r
    return d_ot_1of4(ctx, x=a, y=b)[0]


def d_const(ctx: Ctx, v: int) -> int:
    return v if ctx.party == 0 else 0


def d_xor(*vs: int) -> int:
    r = 0
    for v in vs:
        r ^= v
    return r


def d_not(ctx: Ctx, v: int) -> int:
    return d_xor(v, d_const(ctx, 1))


def d_gmw_add(ctx: Ctx, a_bits, b_bits, k: int):
    """Ripple-carry adder over XOR-shared bits (this party's shares only)."""
    carry = d_const(ctx, 0)
    out = []
    for i in range(k):
        axb = d_xor(a_bits[i], b_bits[i])
        out.append(d_xor(axb, carry))
        carry = d_xor(d_wand(ctx, a_bits[i], b_bits[i]), d_wand(ctx, carry, axb))
    return out


def d_a2b(ctx: Ctx, sh: int, k: int):
    """Arithmetic -> binary: additive share -> XOR bit-shares of the sum."""
    priv = [(sh >> i) & 1 for i in range(k)]
    zeros = [0] * k
    return d_gmw_add(ctx, priv, zeros, k) if ctx.party == 0 else d_gmw_add(ctx, zeros, priv, k)


# ------------------------------------------------------------- slim DReLU

def d_ot_and_private(ctx: Ctx, a: int, b: int) -> int:
    """AND of a bit held by party0 (a) and a bit held by party1 (b).

    One base 1-of-2 OT: party0 masks its bit with a random pad, party1 picks.
    """
    if ctx.party == 0:
        r = secrets.randbits(1)
        d_ot_1of2(ctx, m0=bytes([r]), m1=bytes([r ^ a]))
        return r
    return d_ot_1of2(ctx, c=b)[0]


def d_maj(ctx: Ctx, a: int, b: int, c: int) -> int:
    """majority(a,b,c) = ((a xor c) & (b xor c)) xor c (one GMW AND)."""
    return d_xor(d_wand(ctx, d_xor(a, c), d_xor(b, c)), c)


def d_drelu_slim(ctx: Ctx, sh: int, k: int) -> int:
    """This party's share of the sign bit, using only the carry chain."""
    assert k >= 2, "slim DReLU needs at least 2 bits"
    my = [(sh >> i) & 1 for i in range(k)]
    a0 = my[0] if ctx.party == 0 else 0
    b0 = my[0] if ctx.party == 1 else 0
    carry = d_ot_and_private(ctx, a0, b0)
    for i in range(1, k - 1):
        ai = my[i] if ctx.party == 0 else 0
        bi = my[i] if ctx.party == 1 else 0
        carry = d_maj(ctx, ai, bi, carry)
    a_msb = my[k - 1] if ctx.party == 0 else 0
    b_msb = my[k - 1] if ctx.party == 1 else 0
    sign = d_xor(a_msb, b_msb, carry)
    return d_not(ctx, sign)


# ------------------------------------------------------------- OT multiplication

def d_ot_mul_bit(ctx: Ctx, bit: int) -> int:
    """Share of bit0*bit1 (party0 has bit0, party1 has bit1)."""
    if ctx.party == 0:
        rho = secrets.randbits(1)
        d_ot_1of2(ctx, m0=bytes([rho]), m1=bytes([rho + bit]))
        return -rho
    return d_ot_1of2(ctx, c=bit)[0]


def d_ot_mul_gilboa(ctx: Ctx, val: int, k: int) -> int:
    """Share of x0*y1 (party0 supplies x0=val, party1 supplies y1=val)."""
    if ctx.party == 0:
        x0 = val
        total_pad = 0
        for j in range(k):
            rho = secrets.randbelow(1 << k)
            total_pad = (total_pad + rho) % (1 << k)
            d_ot_1of2(ctx, m0=_enc(rho, k), m1=_enc(rho + x0 * (1 << j), k))
        return (-total_pad) % (1 << k)
    y1 = val
    p1 = 0
    for j in range(k):
        got = d_ot_1of2(ctx, c=(y1 >> j) & 1)
        p1 = (p1 + _dec(got, k)) % (1 << k)
    return p1


# ------------------------------------------------------------- the protocol

def d_secure_relu(ctx: Ctx, sh: int, k: int, k_compare: int | None = None) -> int:
    """This party's integer share of ReLU(z) mod 2^k.  sh = this party's share of z.

    k is the ring width; k_compare (<= k) is the sign-comparison window, so the
    AND-gate count scales with k_compare while the multiplication ring is k.
    """
    if k_compare is None:
        k_compare = k
    assert 1 <= k_compare <= k
    mask = (1 << k_compare) - 1
    b = d_drelu_slim(ctx, sh & mask, k_compare)  # this party's share of the sign bit

    # z * b = (s_c + s_s) * (b0 xor b1), via two single-OT "bit-selects".
    #   term1: s_s * b  -> party0 (server, holder of s_s) is the OT sender
    #   term2: s_c * b  -> party1 (client, holder of s_c) is the OT sender
    if ctx.party == 0:
        x0, b0 = sh, b
        rho1 = secrets.randbelow(1 << k)
        d_ot_1of2(ctx, m0=_enc((rho1 + x0 * (b0 ^ 0)) % (1 << k), k),
                       m1=_enc((rho1 + x0 * (b0 ^ 1)) % (1 << k), k))
        t1 = (-rho1) % (1 << k)
        t2 = _dec(d_ot_1of2_rev(ctx, c=b0), k)  # receiver in term2, choice b0
    else:
        x1, b1 = sh, b
        t1 = _dec(d_ot_1of2(ctx, c=b1), k)       # receiver in term1, choice b1
        rho2 = secrets.randbelow(1 << k)
        d_ot_1of2_rev(ctx, m0=_enc((rho2 + x1 * (b1 ^ 0)) % (1 << k), k),
                           m1=_enc((rho2 + x1 * (b1 ^ 1)) % (1 << k), k))
        t2 = (-rho2) % (1 << k)
    return (t1 + t2) % (1 << k)


def _quantize(share_float: float, k: int, f: int) -> int:
    return int(round(share_float * (2.0 ** f))) % (1 << k)


def d_finalize(ctx: Ctx, ring: int, k: int, f: int) -> float:
    """Turn mod-2^k output shares into REAL-additive shares (one extra message).

    Client (party1) draws rho uniform in [0, 2^(k-1)) and sends
    t0 = (its_ring_share - rho) mod 2^k.  The server (party0) recovers
    b = relu(z) - rho by centering (t0 + its_ring_share) mod 2^k.  rho + b equals
    relu(z) exactly, and because rho ranges over 2^(k-k_compare) times the value
    range, the server's share is statistically 2^-(k-k_compare)-close to uniform.
    """
    half = 1 << (k - 1)
    if ctx.party == 1:
        rho = secrets.randbelow(half)
        ctx.chan.send(_enc((ring - rho) % (1 << k), k))
        return rho / (2.0 ** f)
    t0 = _dec(ctx.chan.recv(), k)
    t = (t0 + ring) % (1 << k)
    b = t - (1 << k) if t >= half else t
    return b / (2.0 ** f)


def run_party(ctx: Ctx, share_float: float, k: int, f: int,
              k_compare: int | None = None) -> float:
    ring = _quantize(share_float, k, f)
    out_ring = d_secure_relu(ctx, ring, k, k_compare)
    return d_finalize(ctx, out_ring, k, f)


# ------------------------------------------------------------------ drivers

def serve(host: str, port: int, share: float, k: int, f: int,
          k_compare: int | None = None) -> float:
    with socket.create_server((host, port)) as srv:
        conn, addr = srv.accept()
        with conn:
            return run_party(Ctx(0, Chan(conn)), share, k, f, k_compare)


def connect(host: str, port: int, share: float, k: int, f: int,
            k_compare: int | None = None) -> float:
    with socket.create_connection((host, port)) as conn:
        return run_party(Ctx(1, Chan(conn)), share, k, f, k_compare)


def net_self_test(k: int = K_RING_B, f: int = F_B,
                  k_compare: int | None = K_COMPARE_B) -> int:
    failures = 0
    if k_compare is None:
        k_compare = k

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    cases = [3.0, -2.0, 5.5, -0.5]   # positives + negatives -> exercises the sign logic
    for z in cases:
        s_s = secrets.randbelow(2001) / 1000.0 * 3.0 - 3.0    # random-ish server share
        s_c = z - s_s
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        holder = {}

        def server_side():
            conn, _ = srv.accept()
            with conn:
                holder["out0"] = run_party(Ctx(0, Chan(conn)), s_c, k, f, k_compare)

        t = threading.Thread(target=server_side, daemon=True)
        t.start()
        out1 = connect("127.0.0.1", port, s_s, k, f, k_compare)
        t.join()
        srv.close()

        got = holder["out0"] + out1
        ref = max(z, 0.0)
        tol = 3.0 / (2.0 ** f)
        check(f"networked secure ReLU  z={z}", abs(got - ref) <= tol,
              f"got={got:.4f} ref={ref:.4f} tol={tol:.4f}")

    print(f"\n(k={k}, k_compare={k_compare}, f={f}, sigma={k - k_compare}) "
          + ("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Two-party networked secure ReLU")
    ap.add_argument("--net-self-test", action="store_true",
                    help="run both parties over a real localhost socket and verify")
    ap.add_argument("--party", type=int, choices=[0, 1], help="0=server (OT sender), 1=client (OT receiver)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9100)
    ap.add_argument("--share", type=float, default=0.0, help="THIS party's share of z")
    ap.add_argument("--bits", type=int, default=K_RING_B, help="ring bit width k")
    ap.add_argument("--k-compare", type=int, default=K_COMPARE_B,
                    help="sign-comparison window (<= bits)")
    ap.add_argument("--frac", type=int, default=F_B, help="fractional bits f")
    args = ap.parse_args()

    if args.net_self_test:
        return net_self_test(args.bits, args.frac, args.k_compare)
    if args.party == 0:
        out = serve(args.host, args.port, args.share, args.bits, args.frac, args.k_compare)
        print(f"[party0/server] my output share = {out:.6f}")
        return 0
    if args.party == 1:
        out = connect(args.host, args.port, args.share, args.bits, args.frac, args.k_compare)
        print(f"[party1/client] my output share = {out:.6f}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
