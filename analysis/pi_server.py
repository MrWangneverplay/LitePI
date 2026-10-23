"""LitePI desktop server: HE linear layers + MPC party0 for every ReLU.

The server never sees the CKKS secret key and never reconstructs an activation.
For each hybrid ReLU it:

    1. draws a fresh mask r ~ U(-8*2^sigma, +8*2^sigma) from the OS CSPRNG,
    2. sends Enc(z + r) to the Pi,
    3. keeps s_s = -r and runs MPC party0 on it,
    4. receives Enc(c_c) from the Pi, encrypts c_s under the public key, and
       adds the two ciphertexts.

Run:
    python pi_server.py --host 0.0.0.0 --port 9101 --dims 64,32,10

The architecture is announced to the client before the run; the client only
chooses its input dimension and backend.
"""

from __future__ import annotations

import argparse
import json
import secrets
import socket
import time
from pathlib import Path

import numpy as np

from params_pi import MASK_HALF_WIDTH
from pi_proto import (
    F,
    K_COMPARE,
    K_RING,
    ProtocolError,
    Wire,
    make_backend,
    ots_per_relu,
    profile_info,
)
from secure_relu_net import run_party_batch


def _as_dims(text: str) -> list[int]:
    dims = [int(x) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(dims) < 2 or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("dims must be at least two positive ints")
    return dims


class HybridMLP:
    """Small dense net with a hybrid ReLU after every hidden layer.

    It is deliberately tiny: the purpose is to exercise the client/server
    protocol, not to produce an accuracy number for the paper.
    """

    def __init__(self, dims: list[int], seed: int = 0) -> None:
        if len(dims) < 2:
            raise ValueError("need at least input and output dims")
        if any(not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError("all layer dimensions must be positive integers")
        self.dims = list(dims)
        rng = np.random.default_rng(seed)
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            scale = 1.0 / np.sqrt(float(fan_in))
            self.weights.append(rng.normal(0.0, scale, (fan_in, fan_out)))
            self.biases.append(rng.normal(0.0, 0.05, fan_out))

    def validate_weights(self) -> None:
        if len(self.weights) != len(self.biases) or not self.weights:
            raise ValueError("weights and biases must contain the same non-zero layers")
        expected_in = self.dims[0]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            w = np.asarray(w, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64).ravel()
            if w.ndim != 2 or w.shape[0] != expected_in:
                raise ValueError(
                    f"layer {i} weight shape {w.shape} does not start with {expected_in}"
                )
            if b.size != w.shape[1]:
                raise ValueError(
                    f"layer {i} bias size {b.size} does not match {w.shape[1]}"
                )
            if not np.all(np.isfinite(w)) or not np.all(np.isfinite(b)):
                raise ValueError(f"layer {i} contains NaN or infinity")
            expected_in = w.shape[1]
        if expected_in != self.dims[-1]:
            raise ValueError(
                f"weight output dimension {expected_in} does not match {self.dims[-1]}"
            )

    # -- local reference (test / loopback only) ---------------------------
    def plain_forward(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        z = np.asarray(x, dtype=np.float64).ravel()
        if z.size != self.dims[0]:
            raise ValueError(f"input has {z.size} values, model wants {self.dims[0]}")
        activations: list[np.ndarray] = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = z @ w + b
            if i < len(self.weights) - 1:
                z = np.maximum(z, 0.0)
                activations.append(z)
        return z, activations

    # -- encrypted forward ------------------------------------------------
    def encrypted_forward(self, be, ctx, ct, wire: Wire, timings: dict) -> object:
        mpc_ctx = wire.mpc_ctx()
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            t0 = time.perf_counter()
            ct = be.mm(ct, w)
            ct = be.add_plain(ct, b)
            timings["linear_s"] += time.perf_counter() - t0
            if i < len(self.weights) - 1:
                t0 = time.perf_counter()
                ct = hybrid_relu_server(be, ctx, ct, wire, mpc_ctx, timings)
                timings["hybrid_relu_s"] += time.perf_counter() - t0
        return ct


def draw_mask(n: int) -> np.ndarray:
    """Fresh uniform mask in [-MASK_HALF_WIDTH, +MASK_HALF_WIDTH), OS CSPRNG."""
    if n <= 0:
        raise ValueError("mask size must be positive")
    out = np.empty(n, dtype=np.float64)
    span = 2.0 * MASK_HALF_WIDTH
    for i in range(n):
        out[i] = -MASK_HALF_WIDTH + span * (secrets.randbits(53) / float(1 << 53))
    return out


def hybrid_relu_server(be, ctx, ct, wire: Wire, mpc_ctx, timings: dict):
    """One hybrid ReLU, server side (steps 1 and 5 of the five-step flow)."""
    n = be.slot_count(ct)
    if n <= 0:
        raise ProtocolError("cannot run ReLU on an empty ciphertext")

    r = draw_mask(n)
    # Adding a plaintext vector is the same homomorphic operation as adding
    # Enc(r) but needs no encryption on the server; the client still only sees
    # the ciphertext and cannot recover r before decrypting z+r.
    ct_zr = be.add_plain(ct, r)
    wire.send(b"CTXT", b"\x00" + be.serialize(ct_zr))

    # Step 3: server's input share is -r; it never learns the client's share.
    s_server = -r
    t0 = time.perf_counter()
    c_server = np.asarray(
        run_party_batch(mpc_ctx, s_server.tolist(), K_RING, F, K_COMPARE),
        dtype=np.float64,
    )
    timings["mpc_s"] += time.perf_counter() - t0
    timings["mpc_elements"] += n
    timings["base_ots"] += n * ots_per_relu()

    # Step 4 arrives after the client finished its side of the same MPC.
    _, payload = wire.recv(b"CTXT")
    if not payload or payload[0] != 0:
        raise ProtocolError("expected re-encrypted client share")
    ct_cc = be.deserialize(ctx, payload[1:])

    # Step 5: add the server's output share.  Encrypt it instead of adding a
    # large plaintext directly: CKKS plaintext addition followed by another
    # ciphertext multiplication preserves the message but loses its precision.
    ct_server = be.encrypt(ctx, c_server)
    return be.add_cipher(ct_cc, ct_server)


def serve_connection(conn: socket.socket, be, mlp: HybridMLP,
                     reference_input=None, timeout: float = 600.0) -> dict:
    conn.settimeout(timeout)
    wire = Wire(conn, party=0)
    try:
        wire.send_json(b"SPEC", {
            "dims": mlp.dims,
            "hybrid_layers": len(mlp.dims) - 2,
            "profile": profile_info(be.name),
        })
        ctx = None
        pending_init = None
        query_index = 0
        while True:
            if pending_init is None:
                init = wire.recv_json(b"INIT")
            else:
                init = pending_init
                pending_init = None

            if init.get("backend") != be.name:
                raise ProtocolError(
                    f"backend mismatch: server={be.name} client={init.get('backend')}"
                )
            input_dim = int(init.get("input_dim", -1))
            if input_dim != mlp.dims[0]:
                raise ProtocolError(
                    f"input dimension {input_dim} does not match model {mlp.dims[0]}"
                )

            persistent = bool(init.get("persistent", False))
            reused_context = bool(init.get("reuse_context", False))
            query_started = time.perf_counter()
            bytes_sent_start = wire.cc.sent
            bytes_received_start = wire.cc.recvd

            if reused_context:
                if ctx is None:
                    raise ProtocolError("client requested context reuse before uploading a context")
                wire.send(b"REDY")
            else:
                wire.send(b"REDY")
                _, blob = wire.recv(b"CTXT")
                ctx = be.load_public_context(blob)
                wire.send(b"REDY")

            _, blob = wire.recv(b"CTXT")
            ct = be.deserialize(ctx, blob)
            if be.slot_count(ct) != mlp.dims[0]:
                raise ProtocolError("input ciphertext has the wrong slot count")

            timings = {
                "linear_s": 0.0,
                "hybrid_relu_s": 0.0,
                "mpc_s": 0.0,
                "mpc_elements": 0,
                "base_ots": 0,
            }
            t0 = time.perf_counter()
            ct_out = mlp.encrypted_forward(be, ctx, ct, wire, timings)
            timings["forward_s"] = time.perf_counter() - t0

            wire.send(b"CTXT", b"\x01" + be.serialize(ct_out))
            wire.recv(b"BYE.")

            stats = {
                "role": "server/party0",
                **profile_info(be.name),
                "dims": mlp.dims,
                "hybrid_layers": len(mlp.dims) - 2,
                "query_index": query_index,
                "reused_context": reused_context,
                "wall_s": round(time.perf_counter() - query_started, 4),
                "timings": {k: round(v, 6) for k, v in timings.items()},
                "bytes_sent": wire.cc.sent - bytes_sent_start,
                "bytes_received": wire.cc.recvd - bytes_received_start,
            }
            if reference_input is not None:
                ref, _ = mlp.plain_forward(reference_input)
                stats["reference_logits"] = [round(float(v), 6) for v in ref]
            wire.send_json(b"STAT", stats)
            query_index += 1

            if not persistent:
                return stats

            tag, payload = wire.recv()
            if tag == b"DONE":
                return stats
            if tag != b"INIT":
                raise ProtocolError(f"expected INIT or DONE, got {tag!r}")
            pending_init = json.loads(payload.decode("utf-8"))
    except Exception as exc:  # tell the peer before the socket dies
        wire.send_error(exc)
        raise


def serve(host: str, port: int, mlp: HybridMLP, backend: str,
          once: bool = True) -> None:
    be = make_backend(backend)
    with socket.create_server((host, port), reuse_port=False) as srv:
        print(f"[server] listening on {host}:{port}  backend={backend}  dims={mlp.dims}")
        while True:
            conn, addr = srv.accept()
            print(f"[server] connection from {addr[0]}:{addr[1]}")
            with conn:
                try:
                    stats = serve_connection(conn, be, mlp)
                    print("[server] " + json.dumps(stats, ensure_ascii=False))
                except Exception as exc:
                    print(f"[server] failed: {type(exc).__name__}: {exc}")
                    if once:
                        raise
            if once:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI desktop server (party0)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--dims", type=_as_dims, default=[64, 32, 10],
                    help="comma-separated layer sizes, e.g. 784,128,64,10")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--backend", choices=["tenseal", "stub"], default="tenseal")
    ap.add_argument("--serve", action="store_true",
                    help="keep serving instead of handling one connection")
    ap.add_argument("--weights", type=Path, default=None,
                    help="optional .npz with W0,b0,W1,b1,... (overrides seed)")
    args = ap.parse_args()

    mlp = HybridMLP(args.dims, args.seed)
    if args.weights:
        data = np.load(args.weights)
        weights, biases = [], []
        i = 0
        while f"W{i}" in data:
            if f"b{i}" not in data:
                raise SystemExit(f"{args.weights} has W{i} but no b{i}")
            weight = np.asarray(data[f"W{i}"], dtype=np.float64)
            bias = np.asarray(data[f"b{i}"], dtype=np.float64)
            if weight.ndim != 2:
                raise SystemExit(f"W{i} must be a 2-D matrix, got shape {weight.shape}")
            if bias.ndim != 1:
                bias = bias.ravel()
            weights.append(weight)
            biases.append(bias)
            i += 1
        if not weights:
            raise SystemExit(f"{args.weights} has no W0/b0 arrays")
        dims = [weights[0].shape[0]] + [w.shape[1] for w in weights]
        mlp = HybridMLP(dims, args.seed)
        mlp.weights, mlp.biases = weights, biases
        try:
            mlp.validate_weights()
        except ValueError as exc:
            raise SystemExit(f"invalid weights: {exc}") from exc
        print(f"[server] loaded {i} layers from {args.weights}: dims={dims}")

    serve(args.host, args.port, mlp, args.backend, once=not args.serve)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

