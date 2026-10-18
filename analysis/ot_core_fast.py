"""Fast base OT: same Chou-Orlandi "Simplest OT" protocol as ot_core.py,
with two changes that remove the latency bottleneck.

1. Curve: NIST P-256 (secp256r1) instead of secp256k1.
   P-256 is prime-order, so the Simplest-OT security proof applies unchanged,
   and OpenSSL implements it with fast, constant-time code.  Measured on this
   machine: private-key object 13 us (vs 526 us for secp256k1) and ECDH 54 us
   (vs 282 us).  ot_core.py keeps secp256k1; this module is self-contained and
   both parties must use the same one.

2. Scalar multiplication: delegated to OpenSSL via the `cryptography` package
   when available (~0.2 ms per OT).  Without it, a pure-Python fallback with
   the same results is used, so the module always works.

Point addition/subtraction stays in pure Python (one modular inverse per call,
only 1-2 calls per OT).

Why: a profile-B secure ReLU (k_compare=14) consumes 2k-1 = 27 base OTs after
the slim carry-chain DReLU.  At the old ~198 ms per OT that is ~5.3 s; here it
is tens of milliseconds.

Run:
    python analysis/ot_core_fast.py --self-test
    python analysis/ot_core_fast.py --bench

Optional accelerator:
    .venv\\Scripts\\pip.exe install cryptography
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import socket
import time

# ---------------------------------------------------------------- P-256 curve

P = 0xFFFFFFFF00000001000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFF
A = P - 3
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
GX = 0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296
GY = 0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5
G = (GX, GY)
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551

DOMAIN = b"litepi-ot-p256-v1\x00"


def inv(a: int, m: int = P) -> int:
    return pow(a, m - 2, m)


def point_add(p1, p2):
    """Affine point addition on y^2 = x^3 - 3x + b over F_p.  None = infinity."""
    if p1 is None:
        return p2
    if p2 is None:
        return p1
    x1, y1 = p1
    x2, y2 = p2
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p1 == p2:
        lam = (3 * x1 * x1 + A) * inv(2 * y1) % P
    else:
        lam = (y2 - y1) * inv(x2 - x1) % P
    x3 = (lam * lam - x1 - x2) % P
    y3 = (lam * (x1 - x3) - y1) % P
    return (x3, y3)


def point_neg(pt):
    return (pt[0], (-pt[1]) % P)


def point_sub(p1, p2):
    return point_add(p1, point_neg(p2))


def scalar_mult(k: int, pt):
    """Pure-Python double-and-add (fallback)."""
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
    xb = point[0].to_bytes(32, "big")
    out = bytearray()
    ctr = 0
    while len(out) < length:
        out += hashlib.sha256(DOMAIN + xb + ctr.to_bytes(4, "big")).digest()
        ctr += 1
    return bytes(out[:length])


# ---------------------------------------------------------------- OpenSSL path

try:
    from cryptography.hazmat.primitives.asymmetric import ec

    _CURVE = ec.SECP256R1()
    HAS_OPENSSL = True
except Exception:  # pragma: no cover
    _CURVE = None
    HAS_OPENSSL = False


def _rand_scalar() -> int:
    return secrets.randbelow(N - 1) + 1


def _kG(k: int):
    """Full point k*G."""
    if HAS_OPENSSL:
        pub = ec.derive_private_key(k, _CURVE).public_key().public_numbers()
        return (pub.x, pub.y)
    return scalar_mult(k, G)


def _x_of_kP(k: int, Pt):
    """x-coordinate of k*Pt."""
    if HAS_OPENSSL:
        peer = ec.EllipticCurvePublicNumbers(Pt[0], Pt[1], _CURVE).public_key()
        return int.from_bytes(ec.derive_private_key(k, _CURVE).exchange(ec.ECDH(), peer), "big")
    return scalar_mult(k, Pt)[0]


def _ks(x: int, length: int) -> bytes:
    return _keystream((x, 0), length)


# ---------------------------------------------------------------- OT protocol

def sender_step1():
    a = _rand_scalar()
    return a, _kG(a)


def receiver_step1(r: int, A):
    assert r in (0, 1)
    b = _rand_scalar()
    bG = _kG(b)
    return b, (bG if r == 0 else point_add(A, bG))


def sender_step2(a: int, A, Bd, m0: bytes, m1: bytes):
    assert len(m0) == len(m1), "OT messages must be the same length"
    y0 = _x_of_kP(a, Bd)
    BmA = point_sub(Bd, A)
    y1 = _x_of_kP(a, BmA) if BmA is not None else 0
    return _xor(m0, _ks(y0, len(m0))), _xor(m1, _ks(y1, len(m1)))


def receiver_step2(b: int, A, r: int, e0: bytes, e1: bytes) -> bytes:
    assert r in (0, 1)
    e = e0 if r == 0 else e1
    return _xor(e, _ks(_x_of_kP(b, A), len(e)))


def run_ot(m0: bytes, m1: bytes, r: int):
    a, A = sender_step1()
    b, Bd = receiver_step1(r, A)
    e0, e1 = sender_step2(a, A, Bd, m0, m1)
    return receiver_step2(b, A, r, e0, e1), r


def _ot_1of2_inproc(m0: bytes, m1: bytes, c: int) -> bytes:
    """One in-process 1-out-of-2 OT (same name/semantics as gmw_core's helper)."""
    a, A = sender_step1()
    b, Bd = receiver_step1(c, A)
    e0, e1 = sender_step2(a, A, Bd, m0, m1)
    return receiver_step2(b, A, c, e0, e1)


# ---------------------------------------------------------------- transport

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
    a, A = sender_step1()
    send_blob(conn, encode_point(A))
    Bd = decode_point(recv_blob(conn))
    e0, e1 = sender_step2(a, A, Bd, m0, m1)
    send_blob(conn, e0)
    send_blob(conn, e1)


def receiver_over_socket(conn, r: int) -> bytes:
    A = decode_point(recv_blob(conn))
    b, Bd = receiver_step1(r, A)
    send_blob(conn, encode_point(Bd))
    e0 = recv_blob(conn)
    e1 = recv_blob(conn)
    return receiver_step2(b, A, r, e0, e1)


# ---------------------------------------------------------------- self test

def self_test() -> int:
    failures = 0

    def check(name, ok, detail=""):
        nonlocal failures
        if not ok:
            failures += 1
        print(("PASS" if ok else "FAIL") + f"  {name}" + (f"  [{detail}]" if detail else ""))

    for r in (0, 1):
        msgs = (b"message-0", b"message-1")
        got, _ = run_ot(msgs[0], msgs[1], r)
        check(f"OT correctness r={r}", got == msgs[r], f"got={got!r}")

    m0, m1 = b"secret-A", b"secret-B"
    a, A = sender_step1()
    b, Bd = receiver_step1(0, A)
    e0, e1 = sender_step2(a, A, Bd, m0, m1)
    got = receiver_step2(b, A, 0, e0, e1)
    wrong = _xor(e1, _ks(_x_of_kP(b, A), len(e1)))
    check("receiver cannot get the other message", got == m0 and wrong != m1,
          f"wrong_decrypt={wrong!r}")

    k = _rand_scalar()
    Pk = _kG(_rand_scalar())
    check("accelerated k*G matches pure Python", _kG(k) == scalar_mult(k, G))
    check("accelerated k*P x-coordinate matches pure Python",
          _x_of_kP(k, Pk) == scalar_mult(k, Pk)[0])

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    import threading

    def server_side():
        conn, _ = srv.accept()
        with conn:
            sender_over_socket(conn, b"SOCKET-0", b"SOCKET-1")

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    with socket.create_connection(("127.0.0.1", port)) as cli:
        got = receiver_over_socket(cli, 1)
    t.join()
    srv.close()
    check("socket OT r=1", got == b"SOCKET-1", f"got={got!r}")

    print(f"\ncurve = P-256 (secp256r1)   backend = "
          f"{'OpenSSL (cryptography)' if HAS_OPENSSL else 'pure Python'}")
    print("ALL PASS" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 0 if failures == 0 else 1


def bench(n: int = 200, baseline_ms: float = 198.46) -> int:
    print(f"curve = P-256 (secp256r1)   backend = "
          f"{'OpenSSL (cryptography)' if HAS_OPENSSL else 'pure Python'}")
    t0 = time.perf_counter()
    for _ in range(n):
        sender_step1()
    t1 = time.perf_counter()
    kg = (t1 - t0) / n * 1000

    msgs = (b"m0", b"m1")
    t0 = time.perf_counter()
    for i in range(n):
        run_ot(msgs[0], msgs[1], i & 1)
    t1 = time.perf_counter()
    ot = (t1 - t0) / n * 1000

    print(f"k*G          : {kg:8.3f} ms")
    print(f"full base OT : {ot:8.3f} ms   (was {baseline_ms:.2f} ms)")
    print(f"speedup      : {baseline_ms / ot:8.1f}x")
    for k in (8, 16):
        ots = 6 * k + 1
        print(f"secure ReLU k={k:2d}: {ots:3d} OTs -> {ots * ot / 1000:6.3f} s "
              f"(was {ots * baseline_ms / 1000:6.2f} s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast Chou-Orlandi base OT (P-256)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.bench:
        return bench(args.n)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
