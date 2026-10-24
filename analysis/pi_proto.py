"""Wire format + HE backend adapter for the LitePI Raspberry-Pi demo.

The demo has two roles:

    server (desktop) : party0, owns the model, OT sender
    client (Pi)      : party1, owns the CKKS secret key, OT receiver

Both sides frame every non-MPC message as ``4-byte tag || payload`` on top of
the same length-prefixed channel used by ``secure_relu_net.Chan``.  The MPC
itself writes raw length-prefixed messages onto that channel, so both parties
must enter the MPC at exactly the same point in the trace.  The sequence is:

    server -> CTXT(kind=0, Enc(z+r))
    both   -> run_party_batch() for all elements in the layer
    client -> CTXT(kind=0, Enc(c_c))
    ...
    server -> CTXT(kind=1, Enc(logits))

Mode ``tenseal`` is the real thing.  Mode ``stub`` replaces CKKS with a
transparent container and exists ONLY to exercise the network/MPC wiring on a
machine without TenSEAL; it provides no privacy whatsoever.  Outputs from the
stub mode are labelled ``simulation`` and must not be cited.
"""

from __future__ import annotations

import json
import socket
from typing import Any

import numpy as np

from params_pi import (  # noqa: F401  (re-exported for callers)
    COEFF_MOD_BIT_SIZES,
    F,
    GLOBAL_SCALE,
    K_COMPARE,
    K_RING,
    PAPER_PROFILE,
    POLY_MODULUS_DEGREE,
    PROFILE_NAME,
    SIGMA,
)
from secure_relu_net import Chan, Ctx  # noqa: F401  (Ctx re-exported)
from protocol_contract import collect as protocol_compliance

MAX_TAG = 4


class ProtocolError(RuntimeError):
    pass


# --------------------------------------------------------------- counting I/O

class CountChan:
    """Length-prefixed channel that also counts application-layer bytes."""

    def __init__(self, conn: socket.socket) -> None:
        self._chan = Chan(conn)
        self.sent = 0
        self.recvd = 0

    def send(self, data: bytes) -> None:
        self._chan.send(data)
        self.sent += 4 + len(data)

    def recv(self) -> bytes:
        data = self._chan.recv()
        self.recvd += 4 + len(data)
        return data


class Wire:
    """Tagged messages on top of CountChan."""

    def __init__(self, conn: socket.socket, party: int) -> None:
        assert party in (0, 1)
        self.party = party
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.cc = CountChan(conn)

    # -- raw MP C channel -------------------------------------------------
    def mpc_ctx(self) -> Ctx:
        return Ctx(self.party, self.cc)

    # -- tagged messages --------------------------------------------------
    def send(self, tag: bytes, payload: bytes = b"") -> None:
        if len(tag) != MAX_TAG:
            raise ValueError(f"tag must be {MAX_TAG} bytes, got {tag!r}")
        self.cc.send(tag + payload)

    def send_json(self, tag: bytes, obj: Any) -> None:
        self.send(tag, json.dumps(obj, ensure_ascii=False).encode("utf-8"))

    def recv(self, expect: bytes | None = None) -> tuple[bytes, bytes]:
        msg = self.cc.recv()
        if len(msg) < MAX_TAG:
            raise ProtocolError(f"short message: {len(msg)} bytes")
        tag, payload = msg[:MAX_TAG], msg[MAX_TAG:]
        if tag == b"ERR ":
            raise ProtocolError(payload.decode("utf-8", "replace"))
        if expect is not None and tag != expect:
            raise ProtocolError(f"expected {expect!r}, got {tag!r}")
        return tag, payload

    def recv_json(self, expect: bytes | None = None) -> dict:
        _, payload = self.recv(expect)
        obj = json.loads(payload.decode("utf-8"))
        if not isinstance(obj, dict):
            raise ProtocolError("JSON message must contain an object")
        return obj

    def send_error(self, exc: BaseException) -> None:
        text = f"{type(exc).__name__}: {exc}"[:6000]
        try:
            self.send(b"ERR ", text.encode("utf-8"))
        except Exception:
            pass


# ------------------------------------------------------------- HE backends

class TenSEALBackend:
    """Real CKKS via TenSEAL, using the Pi demo parameter set."""

    name = "tenseal"

    def __init__(self) -> None:
        import tenseal as ts  # imported lazily so stub mode needs no TenSEAL

        self.ts = ts

    def new_context(self):
        ctx = self.ts.context(
            self.ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=POLY_MODULUS_DEGREE,
            coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
        )
        ctx.global_scale = GLOBAL_SCALE
        ctx.auto_rescale = True
        ctx.auto_mod_switch = True
        ctx.generate_galois_keys()
        ctx.generate_relin_keys()
        return ctx

    def public_context_bytes(self, ctx) -> bytes:
        """Serialize everything the server needs, and nothing it must not have."""
        try:
            return ctx.serialize(
                save_public_key=True,
                save_secret_key=False,
                save_galois_keys=True,
                save_relin_keys=True,
            )
        except TypeError:  # older TenSEAL signature
            return ctx.serialize(save_secret_key=False)

    def load_public_context(self, blob: bytes):
        loader = getattr(self.ts, "context_from", None)
        if loader is not None:
            return loader(blob)
        # Some TenSEAL builds expose deserialization only on Context.
        return self.ts.Context.load(blob)

    def encrypt(self, ctx, values):
        arr = np.asarray(values, dtype=np.float64).ravel()
        return self.ts.ckks_vector(ctx, arr.tolist())

    def im2col_encrypt(self, ctx, values, kernel, stride):
        arr = np.asarray(values, dtype=np.float64)
        kh, kw = (int(v) for v in kernel)
        if isinstance(stride, (tuple, list)):
            sh, sw = (int(v) for v in stride)
            if sh != sw:
                raise ValueError("TenSEAL im2col supports one shared stride")
            stride_value = sh
        else:
            stride_value = int(stride)
        ct, windows = self.ts.im2col_encoding(
            ctx, arr.tolist(), kh, kw, stride_value,
        )
        return ct, int(windows)

    def conv2d_im2col(self, ct, kernel, windows_nb: int):
        if not hasattr(ct, "conv2d_im2col"):
            raise ProtocolError("ciphertext does not retain convolution capabilities")
        arr = np.asarray(kernel, dtype=np.float64).tolist()
        return ct.conv2d_im2col(arr, int(windows_nb))

    def pack_vectors(self, vectors):
        if not vectors:
            raise ValueError("pack_vectors needs at least one ciphertext")
        return self.ts.CKKSVector.pack_vectors(list(vectors))

    def decrypt(self, ct) -> np.ndarray:
        return np.asarray(ct.decrypt(), dtype=np.float64)

    def add_plain(self, ct, values):
        arr = np.asarray(values, dtype=np.float64).ravel()
        if arr.size == 1:
            return ct + float(arr[0])
        return ct + arr.tolist()

    def add_cipher(self, ct, other):
        return ct + other

    def square(self, ct):
        return ct.square()

    def mul_plain(self, ct, value):
        arr = np.asarray(value, dtype=np.float64)
        return ct * (float(arr) if arr.ndim == 0 else arr.tolist())

    def mm(self, ct, matrix):
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError("mm expects a 2-D matrix")
        return ct.mm(mat.tolist())

    def serialize(self, ct) -> bytes:
        return ct.serialize()

    def deserialize(self, ctx, blob: bytes):
        return self.ts.ckks_vector_from(ctx, blob)

    def slot_count(self, ct) -> int:
        # TenSEAL's CKKSVector does not implement __len__; size() is the
        # documented slot-count accessor and also works after mm().
        return int(ct.size())


class StubCiphertext:
    __slots__ = ("values",)

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64).ravel().copy()


class StubBackend:
    """Transparent stand-in for CKKS.  Protocol testing only -- NOT private."""

    name = "stub"
    MAGIC = b"STUBCT01"

    def new_context(self):
        return {"stub": True}

    def public_context_bytes(self, ctx) -> bytes:
        return b"STUB-PUBLIC"

    def load_public_context(self, blob: bytes):
        if blob != b"STUB-PUBLIC":
            raise ProtocolError("stub backend received a non-stub context")
        return {"stub": True}

    def encrypt(self, ctx, values):
        return StubCiphertext(values)

    def im2col_encrypt(self, ctx, values, kernel, stride):
        arr = np.asarray(values, dtype=np.float64)
        kh, kw = (int(v) for v in kernel)
        if isinstance(stride, (tuple, list)):
            sh, sw = (int(v) for v in stride)
        else:
            sh = sw = int(stride)
        while arr.ndim < 2:
            arr = arr.reshape(1, -1)
        rows = (arr.shape[0] - kh) // sh + 1
        cols = (arr.shape[1] - kw) // sw + 1
        if rows <= 0 or cols <= 0:
            raise ValueError("kernel is larger than the encoded input")
        padded = 1 << max(1, int(np.ceil(np.log2(kh * kw))))
        windows = []
        for r in range(rows):
            for c in range(cols):
                y, x = r * sh, c * sw
                patch = arr[y:y + kh, x:x + kw].reshape(-1)
                windows.extend(patch.tolist())
                windows.extend([0.0] * (padded - patch.size))
        return StubCiphertext(windows), rows * cols

    def conv2d_im2col(self, ct, kernel, windows_nb: int):
        ker = np.asarray(kernel, dtype=np.float64).ravel()
        if ker.size == 0:
            raise ValueError("empty convolution kernel")
        width = int(np.ceil(ct.values.size / int(windows_nb)))
        if ker.size > width:
            raise ProtocolError("kernel does not fit the im2col window")
        view = ct.values.reshape(int(windows_nb), width)
        return StubCiphertext(view[:, :ker.size] @ ker)

    def pack_vectors(self, vectors):
        if not vectors:
            raise ValueError("pack_vectors needs at least one ciphertext")
        return StubCiphertext(
            np.concatenate([np.asarray(v.values, dtype=np.float64) for v in vectors])
        )

    def decrypt(self, ct) -> np.ndarray:
        return ct.values.copy()

    def add_plain(self, ct, values):
        return StubCiphertext(ct.values + np.asarray(values, dtype=np.float64).ravel())

    def add_cipher(self, ct, other):
        if ct.values.size != other.values.size:
            raise ProtocolError("stub ciphertext size mismatch")
        return StubCiphertext(ct.values + other.values)

    def square(self, ct):
        return StubCiphertext(ct.values * ct.values)

    def mul_plain(self, ct, value):
        arr = np.asarray(value, dtype=np.float64)
        return StubCiphertext(ct.values * (float(arr) if arr.ndim == 0 else arr))

    def mm(self, ct, matrix):
        mat = np.asarray(matrix, dtype=np.float64)
        return StubCiphertext(ct.values @ mat)

    def serialize(self, ct) -> bytes:
        arr = np.ascontiguousarray(ct.values, dtype=np.float64)
        return self.MAGIC + np.int32(arr.size).tobytes() + arr.tobytes()

    def deserialize(self, ctx, blob: bytes):
        if not blob.startswith(self.MAGIC):
            raise ProtocolError("bad stub ciphertext header")
        off = len(self.MAGIC)
        if len(blob) < off + 4:
            raise ProtocolError("truncated stub ciphertext header")
        n = int(np.frombuffer(blob[off:off + 4], dtype=np.int32)[0])
        if n < 0 or len(blob) != off + 4 + n * 8:
            raise ProtocolError("invalid stub ciphertext length")
        arr = np.frombuffer(blob[off + 4:], dtype=np.float64, count=n)
        return StubCiphertext(arr)

    def slot_count(self, ct) -> int:
        return int(ct.values.size)


def make_backend(name: str):
    if name == "tenseal":
        return TenSEALBackend()
    if name == "stub":
        return StubBackend()
    raise ValueError(f"unknown backend {name!r}")


# --------------------------------------------------------------- shared data

def profile_info(backend: str, mask_domain_aligned: bool = False) -> dict:
    compliance = protocol_compliance()
    return {
        "backend": backend,
        "simulation": backend == "stub",
        "deployment_compliant": compliance["compliant"],
        "protocol_failures": compliance["failures"],
        "mpc_backend": compliance["reference_mpc_backend"],
        "required_mpc_backend": compliance["required_mpc_backend"],
        "mask_policy": "prg(seed_session,query,layer)",
        "mask_domain_aligned": bool(mask_domain_aligned),
        "mask_resync_policy": "query-counter-gap-or-abort",
        "profile_name": PROFILE_NAME,
        "paper_profile": PAPER_PROFILE,
        "poly_modulus_degree": POLY_MODULUS_DEGREE,
        "coeff_mod_bit_sizes": COEFF_MOD_BIT_SIZES,
        "global_scale_bits": int(np.log2(GLOBAL_SCALE)),
        "k_ring": K_RING,
        "k_compare": K_COMPARE,
        "f": F,
        "sigma": SIGMA,
    }


def ots_per_relu() -> int:
    """Base OTs for one slim secure ReLU: 2*k_compare-1 (27 at k_compare=14)."""
    return 2 * K_COMPARE - 1

