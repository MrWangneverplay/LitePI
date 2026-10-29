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
import math
import socket
from dataclasses import asdict, dataclass
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
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            # AF_UNIX socket pairs (used by the local evaluation driver) do not
            # support TCP options.
            pass
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


# ------------------------------------------------------------- wire profile

@dataclass(frozen=True)
class WireProfile:
    """How ciphertexts and keys are put on the wire (E4 byte compression).

    The server chooses the profile and announces it in SPEC; the client adopts
    it before key upload, so both directions use the same accounting.

    compr           SEAL serialisation codec: none | zlib | zstd.  "none"
                    reproduces every byte figure recorded before 2026-10-29.
    seeded_client   M1 and ②_k are encrypted under the client's secret key and
                    sent in SEAL's seeded form (second polynomial -> 32-byte
                    seed): ~half of every client->server ciphertext.
    seeded_keys     pk / relin / Galois keys in seeded form (~half of S_key).
    modswitch_send  the server drops data primes the receiver does not need
                    before sending ①_k and M3 (no rescale, message unchanged).
    short_chain     the client encrypts M1 / ②_k at the lowest chain index the
                    following segment needs (protocol §4.2 property 2, the
                    p<->s coupling) instead of the top of the chain.
    """

    name: str = "baseline"
    compr: str = "none"
    seeded_client: bool = False
    seeded_keys: bool = False
    modswitch_send: bool = False
    short_chain: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict | None) -> "WireProfile":
        return cls(**(value or {}))

    @property
    def is_baseline(self) -> bool:
        return self == WireProfile()


WIRE_PROFILES = {
    "baseline": WireProfile(),
    # One knob at a time, for the E4 compression ablation table.
    "zstd": WireProfile("zstd", compr="zstd"),
    "seeded": WireProfile("seeded", seeded_client=True, seeded_keys=True),
    "modswitch": WireProfile("modswitch", modswitch_send=True),
    "shortchain": WireProfile("shortchain", short_chain=True),
    # Everything that changes no protocol message semantics.
    "compact": WireProfile("compact", compr="zstd", seeded_client=True,
                           seeded_keys=True, modswitch_send=True,
                           short_chain=True),
}

# Modulus sizing for a ciphertext the receiver decrypts (or the server encodes
# into): the coefficients are ~ max|value| * scale plus the decryption noise
# (~2^14 at this profile: 5e-3 * 2^21), so the modulus needs
#     max(log2(max|value| * scale), NOISE_BITS) + WIRE_GUARD_BITS  bits.
# check_wire_canonical.py exercises this bound on every schedule.
NOISE_BITS = 16
WIRE_GUARD_BITS = 4


def needed_modulus_bits(max_abs: float, scale: float) -> int:
    """Bits a ciphertext modulus must have to hold values up to max_abs."""
    value_bits = math.ceil(math.log2(max(float(max_abs), 2.0 ** -20) * float(scale)))
    return int(max(value_bits, NOISE_BITS)) + WIRE_GUARD_BITS


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):   # exact below 3.3e24
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def seal_coeff_primes(poly_modulus_degree: int, bit_sizes) -> list[int]:
    """SEAL's CoeffModulus::Create, reproduced (checked against chain_primes()).

    For each bit size SEAL searches downward from 2^b for primes = 1 mod 2N
    and hands them out smallest-first in the order the sizes are listed.  The
    data primes of this profile are 1.20M-1.79M, i.e. visibly below 2^21,
    which is why the CKKS scale drifts upward after every rescale.
    """
    factor = 2 * int(poly_modulus_degree)
    counts: dict[int, int] = {}
    for b in bit_sizes:
        counts[int(b)] = counts.get(int(b), 0) + 1
    table: dict[int, list[int]] = {}
    for size, cnt in counts.items():
        value = ((1 << size) - 1) // factor * factor + 1
        lower = 1 << (size - 1)
        found: list[int] = []
        while cnt > 0 and value > lower:
            if _is_prime(value):
                found.append(value)
                cnt -= 1
            value -= factor
        table[size] = found
    return [table[int(b)].pop() for b in bit_sizes]


class _WireFallback:
    """Wire methods for backends without SEAL's codec support (baseline only)."""

    wire = WireProfile()

    def set_wire(self, wire: WireProfile) -> None:
        if not wire.is_baseline:
            raise ProtocolError(
                f"backend {self.name!r} supports only the baseline wire profile"
            )
        self.wire = wire

    def encrypt_for_send(self, ctx, values, chain_index=None) -> bytes:
        return self.serialize(self.encrypt(ctx, values))

    def im2col_encrypt_for_send(self, ctx, values, kernel, stride,
                                chain_index=None):
        ct, windows = self.im2col_encrypt(ctx, values, kernel, stride)
        return self.serialize(ct), windows

    def shrink_for_send(self, ct, max_abs: float):
        return ct

    def public_context_bytes_wire(self, ctx) -> bytes:
        return self.public_context_bytes(ctx)

    def chain_prime_bits(self) -> list[int] | None:
        return None


# ------------------------------------------------------------- HE backends

class TenSEALBackend(_WireFallback):
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

    def add_conv_bias(self, ct, bias, width: int):
        """Add the conv bias only to the ``width`` valid windows."""
        return self.add_plain(ct, [float(bias)] * int(width))

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

    def encrypt_like(self, ctx, values, like):
        # TenSEAL auto mod-switches fresh operands to the target chain, so a
        # plain encryption is already "like" the reference ciphertext.
        return self.encrypt(ctx, values)

    def poly2(self, ct, a0, a1, a2):
        squared = self.square(ct)
        return self.add_plain(
            self.add_cipher(self.mul_plain(ct, a1), self.mul_plain(squared, a2)),
            [a0] * self.slot_count(ct),
        )

    def poly2_shifted(self, ct, shift):
        # §3.1 one-level form: only the square costs a rescale; the caller folds
        # a2 and d into the next linear layer's plaintext constants.
        return self.square(self.add_plain(ct, [float(shift)] * self.slot_count(ct)))


class SealBackend:
    """Real CKKS via the in-tree SEAL 4.x pybind11 layer (``seal/litepi_seal``).

    This is the implementation named in the paper's software table ("SEAL 4.x
    C++ / pybind11"), so its numbers are the ones that may enter the paper.
    ``LITEPI_SEAL_DIR`` selects the build directory; ``seal/seal4`` (4.1.2) is
    preferred and ``seal/`` (3.3.2, the SCI-bundled copy) is the fallback.
    """

    name = "seal"

    def __init__(self) -> None:
        import os
        import sys
        import sysconfig
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        candidates = [repo / "seal" / "seal4", repo / "seal" / "seal4_py310",
                      repo / "seal"]
        env = os.environ.get("LITEPI_SEAL_DIR")
        if env:
            candidates.insert(0, Path(env))
        suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
        for path in candidates:
            if (path / f"litepi_seal{suffix}").is_file():
                if str(path) not in sys.path:
                    sys.path.insert(0, str(path))
                break
        else:
            raise RuntimeError(
                "no litepi_seal build matches this interpreter "
                f"({suffix}); run seal/build4.sh with PYTHON=<python>")
        import litepi_seal as seal  # imported lazily, like TenSEAL above

        self.seal = seal
        self.ctx = None
        self.version = getattr(seal, "SEAL_VERSION", "unknown")

    def new_context(self):
        self.ctx = self.seal.new_context(
            POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES, 21,
            max_rot_pow=getattr(self, "max_rot_pow", -1),
        )
        return self.ctx

    def public_context_bytes(self, ctx) -> bytes:
        return self.seal.public_context_bytes(ctx)

    def load_public_context(self, blob: bytes):
        self.ctx = self.seal.load_public_context(blob)
        return self.ctx

    def encrypt(self, ctx, values):
        return ctx.encrypt(np.asarray(values, dtype=np.float64).ravel(), 0.0)

    def im2col_encrypt(self, ctx, values, kernel, stride):
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError("im2col input must be a 2-D image")
        kh, kw = (int(v) for v in kernel)
        if isinstance(stride, (tuple, list)):
            sh, sw = (int(v) for v in stride)
        else:
            sh = sw = int(stride)
        ct, windows = ctx.im2col_encrypt(arr, kh, kw, sh, sw)
        return ct, int(windows)

    def conv2d_im2col(self, ct, kernel, windows_nb: int):
        return self.ctx.conv2d_im2col(
            ct, np.asarray(kernel, dtype=np.float64).ravel(), int(windows_nb),
        )

    def pack_vectors(self, vectors):
        if not vectors:
            raise ValueError("pack_vectors needs at least one ciphertext")
        return self.ctx.pack_vectors(list(vectors))

    def decrypt(self, ct) -> np.ndarray:
        return np.asarray(self.ctx.decrypt(ct), dtype=np.float64)

    def add_plain(self, ct, values):
        return self.ctx.add_plain(ct, np.asarray(values, dtype=np.float64).ravel())

    def add_conv_bias(self, ct, bias, width: int):
        # Only the width valid windows carry conv output; a broadcast bias would
        # leak into the wrapped slots and contaminate the later pack rotation.
        return self.ctx.add_plain(ct, [float(bias)] * int(width))

    def add_cipher(self, ct, other):
        return self.ctx.add_cipher(ct, other)

    def square(self, ct):
        return self.ctx.square(ct)

    def mul_plain(self, ct, value):
        return self.ctx.mul_plain(ct, np.asarray(value, dtype=np.float64).ravel())

    def mm(self, ct, matrix):
        mat = np.asarray(matrix, dtype=np.float64)
        if mat.ndim != 2:
            raise ValueError("mm expects a 2-D matrix")
        return self.ctx.mm(ct, mat)

    def deserialize(self, ctx, blob: bytes):
        return ctx.deserialize(blob)

    def slot_count(self, ct) -> int:
        return int(self.seal.slot_count(ct))

    def encrypt_like(self, ctx, values, like):
        # SEAL requires equal parms_id and scale for ciphertext addition; a
        # fresh operand is therefore encrypted at the level/scale of `like`.
        return ctx.encrypt_like(np.asarray(values, dtype=np.float64).ravel(), like)

    def poly2(self, ct, a0, a1, a2):
        return self.ctx.poly2_eval(ct, float(a0), float(a1), float(a2))

    def poly2_shifted(self, ct, shift):
        # §3.1 one-level form: add the shift (no depth), then one square+rescale.
        return self.ctx.square(self.ctx.add_plain_scalar(ct, float(shift)))

    # ------------------------------------------------------------ wire (E4)
    wire = WireProfile()

    def set_wire(self, wire: WireProfile) -> None:
        modes = list(self.seal.compr_modes()) if hasattr(self.seal, "compr_modes") \
            else ["none"]
        if wire.compr not in modes:
            raise ProtocolError(
                f"wire codec {wire.compr!r} is not in this SEAL build {modes}; "
                "rebuild SEAL with -DSEAL_USE_ZSTD=ON (seal/build_seal4_deps.sh)"
            )
        if not wire.is_baseline and not hasattr(self.seal, "compr_modes"):
            raise ProtocolError("litepi_seal predates the wire API; rebuild it")
        self.wire = wire

    def serialize(self, ct) -> bytes:
        if self.wire.compr == "none":
            return bytes(ct.serialize())
        return bytes(ct.serialize(self.wire.compr))

    def encrypt_for_send(self, ctx, values, chain_index=None) -> bytes:
        """Client: encrypt + serialise one message under the wire profile."""
        arr = np.asarray(values, dtype=np.float64).ravel()
        idx = -1 if (chain_index is None or not self.wire.short_chain) \
            else int(chain_index)
        if self.wire.is_baseline:
            return self.serialize(ctx.encrypt(arr, 0.0))
        return bytes(ctx.encrypt_wire(
            arr, 0.0, idx, bool(self.wire.seeded_client), self.wire.compr,
        ))

    def im2col_encrypt_for_send(self, ctx, values, kernel, stride,
                                chain_index=None):
        if self.wire.is_baseline:
            ct, windows = self.im2col_encrypt(ctx, values, kernel, stride)
            return self.serialize(ct), windows
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        kh, kw = (int(v) for v in kernel)
        if isinstance(stride, (tuple, list)):
            sh, sw = (int(v) for v in stride)
        else:
            sh = sw = int(stride)
        flat, windows = ctx.im2col_flat(arr, kh, kw, sh, sw)
        return self.encrypt_for_send(ctx, flat, chain_index), int(windows)

    def shrink_for_send(self, ct, max_abs: float):
        """Server: drop the data primes the receiver does not need.

        Sized from the ciphertext's *actual* scale (it drifts upward with
        depth, 2^29 at the s=00 output), so the bound stays exact.
        """
        if not self.wire.modswitch_send:
            return ct
        bits = needed_modulus_bits(max_abs, self.ctx.scale(ct))
        return self.ctx.mod_switch_to_bits(ct, int(bits))

    def scale_of(self, ct) -> float:
        return float(self.ctx.scale(ct))

    def supports_switch_to_bits(self) -> bool:
        return hasattr(self.seal.SealContext, "mod_switch_to_bits")

    def switch_to_bits(self, ct, bits: int):
        """Plain mod-switch down to ``bits`` of remaining modulus (no rescale:
        message and scale unchanged).  Used for §7.2 size hiding."""
        return self.ctx.mod_switch_to_bits(ct, int(bits))

    def public_context_bytes_wire(self, ctx) -> bytes:
        if self.wire.compr == "none" and not self.wire.seeded_keys:
            return self.seal.public_context_bytes(ctx)
        return self.seal.public_context_bytes(
            ctx, self.wire.compr, bool(self.wire.seeded_keys),
        )

    def chain_prime_bits(self) -> list[int] | None:
        if self.ctx is None or not hasattr(self.ctx, "chain_prime_bits"):
            return None
        return [int(b) for b in self.ctx.chain_prime_bits()]

    def modulus_bits(self, ct) -> int | None:
        """Remaining modulus bits, or None on a litepi_seal built before the
        wire API (rebuild with seal/build4.sh to enable fit_mask_domain)."""
        if not hasattr(self.ctx, "modulus_bits"):
            return None
        return int(self.ctx.modulus_bits(ct))


class StubCiphertext:
    __slots__ = ("values",)

    def __init__(self, values) -> None:
        self.values = np.asarray(values, dtype=np.float64).ravel().copy()


class StubBackend(_WireFallback):
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

    def add_conv_bias(self, ct, bias, width: int):
        out = np.array(ct.values, dtype=np.float64)
        out[: int(width)] += float(bias)
        return StubCiphertext(out)

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

    def encrypt_like(self, ctx, values, like):
        return StubCiphertext(values)

    def poly2(self, ct, a0, a1, a2):
        squared = self.square(ct)
        return self.add_plain(
            self.add_cipher(self.mul_plain(ct, a1), self.mul_plain(squared, a2)),
            [a0] * self.slot_count(ct),
        )

    def poly2_shifted(self, ct, shift):
        return self.square(self.add_plain(ct, [float(shift)] * self.slot_count(ct)))


def make_backend(name: str, wire: "WireProfile | str | None" = None):
    be = _make_backend(name)
    if wire is not None:
        be.set_wire(WIRE_PROFILES[wire] if isinstance(wire, str) else wire)
    return be


def _make_backend(name: str):
    if name == "tenseal":
        return TenSEALBackend()
    if name == "seal":
        return SealBackend()
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
        "protocol_pending": compliance.get("pending", []),
        # The item's "actual" is litepi_sci when the SCI binding is importable;
        # the old code always reported the Python reference backend here.
        "mpc_backend": next(i["actual"] for i in compliance["items"]
                            if i["name"] == "mpc_backend"),
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

