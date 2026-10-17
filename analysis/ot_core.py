"""Chou-Orlandi "Simplest OT" (2015), pure Python over SECP256K1.

Oblivious Transfer (OT): the sender holds two messages (m0, m1); the receiver
holds a choice bit r and learns ONLY m_r.  After the protocol:
  * the sender learns nothing about r,
  * the receiver learns nothing about m_{1-r}.

This is the cryptographic primitive that will replace the local simulation in
hybrid_mpc.py's mpc_relu (the comparison step).  It is pure Python + stdlib
(hashlib / socket), so it needs no third-party crypto library.

Protocol (Chou-Orlandi 2015), over a prime-order curve with generator G:
    1. Sender:    a <- random;  A = aG;                     send A
    2. Receiver:  b <- random;  B = bG      if r == 0
                                B = A + bG  if r == 1
                  send B
    3. Sender:    y0 = aB;  y1 = a(B - A)
                  e0 = m0 xor H(y0);  e1 = m1 xor H(y1);    send (e0, e1)
    4. Receiver:  y = bA  (= y_r);  m_r = e_r xor H(y)

Run the self-test:
    python ot_core.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import socket
import threading


# ---- SECP256K1 public constants ----
MOD = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
G = (GX, GY)

DOMAIN = b"litepi-ot-v1\x00"


def inv(a: int, m: int = MOD) -> int:
    return pow(a, m - 2, m)


def point_add(p1, p2):
    """Affine point addition on y^2 = x^3 + 7 over F_p.  None = point at infinity."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % MOD == 0:
        return None  # P + (-P) = infinity
    if p1 == p2:
        lam = (3 * x1 * x1) * inv(2 * y1) % MOD
    else:
        lam = (y2 - y1) * inv(x2 - x1) % MOD
    x3 = (lam * lam - x1 - x2) % MOD
    y3 = (lam * (x1 - x3) - y1) % MOD
    return (x3, y3)


def point_neg(pt):
    return (pt[0], (-pt[1]) % MOD)


def point_sub(p1, p2):
    return point_add(p1, point_neg(p2))


def scalar_mult(k: int, pt) -> tuple[int, int]:
    """Double-and-add scalar multiplication."""
    R = None
    while k:
        if k & 1:
            R = point_add(R, pt)
        pt = point_add(pt, pt)
        k >>= 1
    return R


def _xor(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b))


def _keystream(point, length: int) -> bytes:
    """Deterministic keystream of `length` bytes derived from a curve point.

    H is modelled as a random oracle: we hash the x-coordinate (standard ECDH
    KDF practice) in counter mode so messages longer than one hash block work.
    """
    xb = point[0].to_bytes(32, "big")
    out = bytearray()
    ctr = 0
    while len(out) < length:
        out += hashlib.sha256(DOMAIN + xb + ctr.to_bytes(4, "big")).digest()
        ctr += 1
    return bytes(out[:length])


# ---------------------------------------------------------------- OT core

def sender_step1():
    """Sender generates a and A = aG."""
    a = secrets.randbelow(MOD)
    A = scalar_mult(a, G)
    return a, A


def receiver_step1(r: int, A):
    """Receiver generates b and B = bG (r=0) or A + bG (r=1)."""
    assert r in (0, 1)
    b = secrets.randbelow(MOD)
    bG = scalar_mult(b, G)
    B = bG if r == 0 else point_add(A, bG)
    return b, B


def sender_step2(a: int, A, B, m0: bytes, m1: bytes):
    """Sender computes e0, e1."""
    assert len(m0) == len(m1), "OT messages must be the same length"
    y0 = scalar_mult(a, B)                 # a * B
    y1 = scalar_mult(a, point_sub(B, A))   # a * (B - A)
    e0 = _xor(m0, _keystream(y0, len(m0)))
    e1 = _xor(m1, _keystream(y1, len(m1)))
    return e0, e1


def receiver_step2(b: int, A, r: int, e0: bytes, e1: bytes) -> bytes:
    """Receiver decrypts m_r."""
    assert r in (0, 1)
    y = scalar_mult(b, A)                  # b * A
    e = e0 if r == 0 else e1
    return _xor(e, _keystream(y, len(e)))


def run_ot(m0: bytes, m1: bytes, r: int) -> tuple[bytes, int]:
    """One full in-process OT. Returns (m_r, r)."""
    a, A = sender_step1()
    b, B = receiver_step1(r, A)
    e0, e1 = sender_step2(a, A, B, m0, m1)
    got = receiver_step2(b, A, r, e0, e1)
    return got, r


# ---------------------------------------------------------------- socket IO

def _recv_exact(conn, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("peer closed the connection")
        buf += chunk
    return buf


def send_blob(conn, data: bytes) -> None:
    conn.sendall(len(data).to_bytes(4, "big") + data)


def recv_blob(conn) -> bytes:
    n = int.from_bytes(_recv_exact(conn, 4), "big")
    return _recv_exact(conn, n)


def encode_point(pt) -> bytes:
    return pt[0].to_bytes(32, "big") + pt[1].to_bytes(32, "big")


def decode_point(data: bytes):
    assert len(data) == 64, f"bad point blob length {len(data)}"
    return (int.from_bytes(data[:32], "big"), int.from_bytes(data[32:], "big"))


def sender_over_socket(conn, m0: bytes, m1: bytes) -> None:
    """Run the sender side of OT over an established connection."""
    a, A = sender_step1()
    send_blob(conn, encode_point(A))          # 1: A  (S -> R)
    B = decode_point(recv_blob(conn))         # 2: B  (R -> S)
    e0, e1 = sender_step2(a, A, B, m0, m1)
    send_blob(conn, e0)                       # 3: e0 (S -> R)
    send_blob(conn, e1)                       # 3: e1 (S -> R)


def receiver_over_socket(conn, r: int) -> bytes:
    """Run the receiver side of OT over an established connection."""
    A = decode_point(recv_blob(conn))         # 1: A
    b, B = receiver_step1(r, A)
    send_blob(conn, encode_point(B))          # 2: B
    e0 = recv_blob(conn)                      # 3: e0
    e1 = recv_blob(conn)                      # 3: e1
    return receiver_step2(b, A, r, e0, e1)


# ---------------------------------------------------------------- self test

def self_test() -> int:
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    # 1. correctness for both choices
    for r in (0, 1):
        m = (b"message-0", b"message-1")
        got, rr = run_ot(m[0], m[1], r)
        check(f"OT correctness r={r}", got == m[r] and rr == r,
              f"got={got!r} expect={m[r]!r}")

    # 2. receiver cannot recover the OTHER message
    m0, m1 = b"secret-A", b"secret-B"
    r = 0
    a, A = sender_step1()
    b, B = receiver_step1(r, A)
    e0, e1 = sender_step2(a, A, B, m0, m1)
    # receiver's own key y = bA decrypts e0 correctly ...
    got = receiver_step2(b, A, r, e0, e1)
    # ... but trying the same key on e1 must NOT yield m1
    y = scalar_mult(b, A)
    wrong = _xor(e1, _keystream(y, len(e1)))
    check("receiver cannot get the other message", got == m0 and wrong != m1,
          f"wrong_decrypt={wrong!r}")

    # 3. EC math cross-check against cryptography (optional, if installed)
    try:
        from cryptography.hazmat.primitives.asymmetric import ec
        k = secrets.randbelow(MOD)
        ref = ec.derive_private_key(k, ec.SECP256K1()).public_key().public_numbers()
        mine = scalar_mult(k, G)
        check("EC scalar-mult cross-check", ref.x == mine[0] and ref.y == mine[1],
              f"x={hex(mine[0])[:18]}")
    except Exception as exc:  # pragma: no cover
        print(f"SKIP  EC cross-check (cryptography not available: {type(exc).__name__})")

    # 4. socket round-trip over a real TCP connection (localhost, ephemeral port)
    m0, m1 = b"SOCKET-0", b"SOCKET-1"
    for r in (0, 1):
        srv = socket.socket()
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        holder = {}

        def server_side():
            conn, _ = srv.accept()
            with conn:
                sender_over_socket(conn, m0, m1)

        t = threading.Thread(target=server_side, daemon=True)
        t.start()
        with socket.create_connection(("127.0.0.1", port)) as cli:
            got = receiver_over_socket(cli, r)
        t.join()
        srv.close()
        check(f"socket OT r={r}", got == (m0 if r == 0 else m1), f"got={got!r}")

    print("\n" + ("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 0 if failures == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Chou-Orlandi Simplest OT")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sender", action="store_true", help="run as the OT sender (server)")
    ap.add_argument("--receiver", action="store_true", help="run as the OT receiver (client)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--choice", type=int, choices=[0, 1], help="receiver's choice bit")
    ap.add_argument("--m0", default="SECRET-ALPHA", help="sender's message 0")
    ap.add_argument("--m1", default="SECRET-BRAVO", help="sender's message 1")
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.sender:
        m0, m1 = args.m0.encode("utf-8"), args.m1.encode("utf-8")
        n = max(len(m0), len(m1))
        m0, m1 = m0.ljust(n), m1.ljust(n)
        with socket.create_server((args.host, args.port)) as srv:
            print(f"[sender] listening on {args.host}:{args.port} ...")
            conn, addr = srv.accept()
            with conn:
                sender_over_socket(conn, m0, m1)
                print(f"[sender] done, peer={addr}")
        return 0
    if args.receiver:
        if args.choice is None:
            raise SystemExit("--receiver needs --choice 0|1")
        with socket.create_connection((args.host, args.port)) as conn:
            got = receiver_over_socket(conn, args.choice)
        print(f"[receiver] choice={args.choice}  got={got.decode('utf-8', 'replace')!r}")
        return 0
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
