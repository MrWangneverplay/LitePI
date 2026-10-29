"""LitePI desktop server: HE linear layers + MPC party0 for every ReLU.

The server never sees the CKKS secret key and never reconstructs an activation.
For each hybrid ReLU it:

    1. draws a fresh M1/M2 mask r from the session PRG,
    2. computes ct_{z+r} = ct_z + Enc_pk(r) and sends it to the Pi,
    3. keeps s_s = -r and runs MPC party0 on it,
    4. receives Enc_pk(c_c) from the Pi,
    5. returns ct_{c_c} + c_s as specified by the protocol.

Run:
    python pi_server.py --dataset cifar10 --schedule 10

The model and client-side im2col geometry are announced in SPEC before the
query.  ``--dims`` remains available only for the legacy protocol self-test.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import socket
import sys
import time
from pathlib import Path

import numpy as np

from canonical_net import CanonicalCNN, load_canonical_model
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
from secure_argmax import run_party_argmax
from mask_session import MaskSession, load_mask_domains
from sci_relu import SCI_AVAILABLE, SciReLUSession

P2_A0 = 0.0625
P2_A1 = 0.5
P2_A2 = 0.5


def _dbg(tag, arr, k=4):
    if os.environ.get("LITEPI_DEBUG"):
        vals = np.asarray(arr, dtype=np.float64).ravel()[:k]
        print(f"[DBG srv] {tag}: {np.round(vals, 6).tolist()}",
              file=sys.stderr, flush=True)


def _as_dims(text: str) -> list[int]:
    dims = [int(x) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(dims) < 2 or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("dims must be at least two positive ints")
    return dims


def _as_schedule(text: str) -> tuple[int, ...]:
    bits = tuple(int(x) for x in text.replace(" ", "").split(",") if x != "")
    if not bits or any(bit not in (0, 1) for bit in bits):
        raise argparse.ArgumentTypeError("schedule must be comma-separated 0/1 bits")
    return bits


class HybridMLP:
    """Small dense net with a hybrid ReLU after every hidden layer.

    It is deliberately tiny: the purpose is to exercise the client/server
    protocol, not to produce an accuracy number for the paper.
    """

    def __init__(self, dims: list[int], seed: int = 0,
                 schedule: tuple[int, ...] | None = None) -> None:
        if len(dims) < 2:
            raise ValueError("need at least input and output dims")
        if any(not isinstance(d, int) or d <= 0 for d in dims):
            raise ValueError("all layer dimensions must be positive integers")
        self.dims = list(dims)
        hidden_count = len(self.dims) - 2
        self.schedule = schedule or tuple(1 for _ in range(hidden_count))
        if len(self.schedule) != hidden_count:
            raise ValueError(
                f"schedule has {len(self.schedule)} bits; need {hidden_count}"
            )
        if any(bit not in (0, 1) for bit in self.schedule):
            raise ValueError("schedule bits must be 0 or 1")
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

    def client_spec(self) -> dict:
        return {
            "architecture": "hybrid_mlp",
            "dims": self.dims,
            "input_dim": self.dims[0],
            "input_slots": self.dims[0],
            "schedule": list(self.schedule),
            "hybrid_layers": self.hybrid_layers,
            "encoding": "vector",
        }

    # -- local reference (test / loopback only) ---------------------------
    def plain_forward(self, x) -> tuple[np.ndarray, list[np.ndarray]]:
        z = np.asarray(x, dtype=np.float64).ravel()
        if z.size != self.dims[0]:
            raise ValueError(f"input has {z.size} values, model wants {self.dims[0]}")
        activations: list[np.ndarray] = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = z @ w + b
            if i < len(self.weights) - 1:
                if self.schedule[i]:
                    z = np.maximum(z, 0.0)
                else:
                    z = P2_A0 + P2_A1 * z + P2_A2 * z * z
                activations.append(z)
        return z, activations

    @property
    def hybrid_layers(self) -> int:
        return sum(self.schedule)

    # -- encrypted forward ------------------------------------------------
    def encrypted_forward(self, be, ctx, ct, wire: Wire, timings: dict,
                          mask_session: MaskSession | None = None,
                          query_index: int = 0, sci_session=None) -> object:
        mpc_ctx = wire.mpc_ctx()
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            t0 = time.perf_counter()
            ct = be.mm(ct, w)
            ct = be.add_plain(ct, b)
            timings["linear_s"] += time.perf_counter() - t0
            if i < len(self.weights) - 1:
                t0 = time.perf_counter()
                if self.schedule[i]:
                    ct = hybrid_relu_server(
                        be, ctx, ct, wire, mpc_ctx, timings,
                        mask_session=mask_session,
                        query_index=query_index,
                        layer_index=i,
                        sci_session=sci_session,
                    )
                else:
                    if getattr(self, "schedule_obfuscated", False):
                        ct = dummy_round_server(be, ctx, ct, wire)
                    ct = he_poly_server(be, ct)
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


def he_poly_server(be, ct):
    """s_k=0: P2 activation evaluated entirely on ciphertext."""
    return be.poly2(ct, P2_A0, P2_A1, P2_A2)


def he_poly_layer(be, ctx, ct, wire: Wire, model):
    """§3.1 1-level P2: return ct_{(z+c)^2} plus the folded (a2, d) coefficients.

    P2(z) = a2(z + c)^2 + d with c = a1/(2a2), d = a0 - a1^2/(4a2), so the
    expensive part is a single square+rescale.  The a2 multiply and the d shift
    are folded into the next linear layer's plaintext weight matrix and bias
    (server-side constants, zero messages, zero extra depth), which is what
    keeps NonlinearEval at the one level the depth table budgets.
    """
    if getattr(model, "schedule_obfuscated", False):
        ct = dummy_round_server(be, ctx, ct, wire)
    c = P2_A1 / (2.0 * P2_A2)
    d = P2_A0 - P2_A1 * P2_A1 / (4.0 * P2_A2)
    return be.poly2_shifted(ct, c), (P2_A2, d)


def fold_poly_into(weight_t, bias, weight, folded) -> tuple:
    """Absorb (a2, d) from ``he_poly_layer`` into the following mm plaintext."""
    a2, d = folded
    row_sum = np.asarray(weight.sum(dim=1).numpy(), dtype=np.float64)
    return a2 * weight_t, np.asarray(bias, dtype=np.float64) + d * row_sum


def dummy_round_server(be, ctx, ct, wire: Wire):
    """§7.2: one dummy ciphertext exchange without consuming HE depth."""
    n = be.slot_count(ct)
    # Zero-multiplication preserves the current level and ciphertext size, so
    # the dummy leg has the same observable shape as a real hybrid leg.
    dummy = be.mul_plain(ct, 0.0)
    wire.send(b"CTXT", b"\x03" + be.serialize(dummy))
    _, payload = wire.recv(b"CTXT")
    if not payload or payload[0] != 3:
        raise ProtocolError("expected dummy re-encryption")
    be.deserialize(ctx, payload[1:])
    return ct


def hybrid_relu_server(be, ctx, ct, wire: Wire, mpc_ctx, timings: dict,
                       mask_session: MaskSession | None = None,
                       query_index: int = 0, layer_index: int = 0,
                       sci_session=None):
    """One hybrid ReLU, server side (steps 1 and 5 of the five-step flow)."""
    n = be.slot_count(ct)
    if n <= 0:
        raise ProtocolError("cannot run ReLU on an empty ciphertext")

    if mask_session is None:
        r = draw_mask(n)
    else:
        r = np.asarray(
            mask_session.draw(query_index, layer_index, n),
            dtype=np.float64,
        )
    # Protocol step 1: ct_{z+r} = ct_z + Enc_pk(r).  The server has only the
    # public context and therefore encrypts the mask with pk, never with sk.
    ct_zr = be.add_cipher(ct, be.encrypt_like(ctx, r, ct))
    wire.send(b"CTXT", b"\x00" + be.serialize(ct_zr))
    timings["hybrid_wire_sends"] += 1

    # Step 3: server's input share is -r; it never learns the client's share.
    s_server = -r
    _dbg("mask r (first 4)", r)
    _dbg("s_server=-r", s_server)
    t0 = time.perf_counter()
    if sci_session is not None:
        c_server = np.asarray(
            sci_session.relu(s_server.tolist(), F, K_RING),
            dtype=np.float64,
        )
    else:
        c_server = np.asarray(
            run_party_batch(mpc_ctx, s_server.tolist(), K_RING, F, K_COMPARE),
            dtype=np.float64,
        )
    timings["mpc_s"] += time.perf_counter() - t0
    timings["mpc_elements"] += n
    if sci_session is None:
        timings["base_ots"] += n * ots_per_relu()
    _dbg("c_server out-share", c_server)

    # Step 4 arrives after the client finished its side of the same MPC.
    _, payload = wire.recv(b"CTXT")
    if not payload or payload[0] != 0:
        raise ProtocolError("expected re-encrypted client share")
    timings["hybrid_wire_recvs"] += 1
    ct_cc = be.deserialize(ctx, payload[1:])

    # Protocol step 5: reconstruct ct_sigma(z) = ct_{c_c} + c_s WITHOUT
    # consuming CKKS depth (fresh ciphertext stays at the chain top).
    # TenSEAL's ciphertext+plain add mod-switches to the plaintext's chain
    # index, which silently spends the freshly reset depth and makes the next
    # mm overflow once shares are full-ring magnitude.  Encrypting the
    # server's own share under pk (S already encrypts r the same way in
    # step 1) and adding two same-level fresh ciphertexts preserves the
    # level and matches the protocol's depth-reset property; no wire message
    # changes (this encryption is local), r_h stays 1.
    return be.add_cipher(ct_cc, be.encrypt_like(ctx, c_server, ct_cc))


def _nonlinear_server(be, ctx, ct, wire, timings, model, layer_index,
                      mask_session, query_index, sci_session=None):
    if model.schedule[layer_index]:
        return hybrid_relu_server(
            be, ctx, ct, wire, wire.mpc_ctx(), timings,
            mask_session=mask_session,
            query_index=query_index,
            layer_index=layer_index,
            sci_session=sci_session,
        )
    if getattr(model, "schedule_obfuscated", False):
        ct = dummy_round_server(be, ctx, ct, wire)
    return he_poly_server(be, ct)


def canonical_encrypted_forward(model: CanonicalCNN, be, ctx, ct, wire: Wire,
                                timings: dict,
                                mask_session: MaskSession | None = None,
                                query_index: int = 0,
                                sci_session=None):
    """Run the canonical Conv -> act -> FC -> act -> FC protocol trace."""
    t0 = time.perf_counter()
    channels = []
    conv = model.conv_weight_np
    conv_bias = model.conv_bias_np
    for out_channel in range(conv.shape[0]):
        channel = be.conv2d_im2col(
            ct, conv[out_channel, 0], model.windows,
        )
        channels.append(
            be.add_conv_bias(channel, conv_bias[out_channel], model.windows)
        )
    ct = be.pack_vectors(channels)
    if be.slot_count(ct) != model.spec.conv_flat:
        raise ProtocolError(
            f"packed conv has {be.slot_count(ct)} slots, "
            f"expected {model.spec.conv_flat}"
        )
    timings["linear_s"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    w1, b1 = model.fc1_weight_t_np, model.fc1_bias_np
    if model.schedule[0] == 0:
        ct, folded = he_poly_layer(be, ctx, ct, wire, model)
        w1, b1 = fold_poly_into(w1, b1, model.fc1_weight, folded)
    else:
        ct = _nonlinear_server(
            be, ctx, ct, wire, timings, model, 0, mask_session, query_index,
            sci_session,
        )
    timings["hybrid_relu_s"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    ct = be.mm(ct, w1)
    ct = be.add_plain(ct, b1)
    timings["linear_s"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    w2, b2 = model.fc2_weight_t_np, model.fc2_bias_np
    if model.schedule[1] == 0:
        ct, folded = he_poly_layer(be, ctx, ct, wire, model)
        w2, b2 = fold_poly_into(w2, b2, model.fc2_weight, folded)
    else:
        ct = _nonlinear_server(
            be, ctx, ct, wire, timings, model, 1, mask_session, query_index,
            sci_session,
        )
    timings["hybrid_relu_s"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    ct = be.mm(ct, w2)
    ct = be.add_plain(ct, b2)
    timings["linear_s"] += time.perf_counter() - t0
    return ct


def encrypted_forward(model, be, ctx, ct, wire: Wire, timings: dict,
                      mask_session: MaskSession | None = None,
                      query_index: int = 0,
                      sci_session=None):
    if isinstance(model, CanonicalCNN):
        return canonical_encrypted_forward(
            model, be, ctx, ct, wire, timings, mask_session, query_index,
            sci_session,
        )
    return model.encrypted_forward(
        be, ctx, ct, wire, timings,
        mask_session=mask_session,
        query_index=query_index,
        sci_session=sci_session,
    )


def output_layer_id(model) -> int:
    value = getattr(model, "output_layer_id", None)
    return int(value() if callable(value) else len(model.schedule))


def secure_argmax_server(be, ctx, ct, wire: Wire, timings: dict,
                         mask_session: MaskSession, query_index: int,
                         model) -> int:
    """Output mode B: return only a shared one-hot index to the client."""
    classes = be.slot_count(ct)
    if classes < 2:
        raise ProtocolError("argmax needs at least two logits")
    layer_id = output_layer_id(model)
    r = np.asarray(
        mask_session.draw(query_index, layer_id, classes),
        dtype=np.float64,
    )
    wire.send(b"CTXT", b"\x02" + be.serialize(be.add_plain(ct, r)))

    t0 = time.perf_counter()
    server_share = run_party_argmax(
        wire.mpc_ctx(), (-r).tolist(), K_RING, F, K_COMPARE,
    )
    timings["mpc_s"] += time.perf_counter() - t0
    timings["mpc_elements"] += classes
    timings["base_ots"] += (classes - 1) * max(K_COMPARE - 1, 1)
    wire.send_json(b"IDXS", {"share": server_share})
    return server_share


def serve_connection(conn: socket.socket, be, model,
                     reference_input=None, timeout: float = 600.0,
                     mask_domains=None,
                     obfuscate_schedule: bool = False) -> dict:
    conn.settimeout(timeout)
    wire = Wire(conn, party=0)
    mask_session = MaskSession(domains=mask_domains or {})
    hybrid_ids = [
        i for i, bit in enumerate(model.schedule) if int(bit) == 1
    ]
    model.mask_domain_aligned = mask_session.covers(hybrid_ids)
    model.schedule_obfuscated = bool(obfuscate_schedule)
    try:
        aligned = bool(getattr(model, "mask_domain_aligned", False))
        spec = model.client_spec()
        spec.update({
            "protocol_modes": {
                "he_only": "all schedule bits 0",
                "hybrid_all": "all schedule bits 1",
                "adaptive": "schedule supplied by planner",
            },
            "profile": profile_info(be.name, mask_domain_aligned=aligned),
        })
        sci_port = None
        if SCI_AVAILABLE:
            _tmp = socket.socket()
            _tmp.bind(("0.0.0.0", 0))
            sci_port = _tmp.getsockname()[1]
            _tmp.close()
            spec["sci_port"] = sci_port
        wire.send_json(b"SPEC", spec)
        sci_session = None
        if sci_port is not None:
            sci_session = SciReLUSession(1, sci_port)
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
            requested_output = str(init.get("output_mode", "logits"))
            if requested_output not in ("logits", "index"):
                raise ProtocolError(f"unknown output mode {requested_output!r}")
            if requested_output == "index" and not mask_session.covers(
                [output_layer_id(model)]
            ):
                raise ProtocolError(
                    "output mode B requires the M2 output-logit domain"
                )
            input_dim = int(init.get("input_dim", -1))
            expected_input = int(getattr(model, "input_dim", model.dims[0]))
            if input_dim != expected_input:
                raise ProtocolError(
                    f"input dimension {input_dim} does not match model {expected_input}"
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
            expected_slots = int(getattr(model, "input_slots", model.dims[0]))
            if be.slot_count(ct) != expected_slots:
                raise ProtocolError(
                    f"input ciphertext has {be.slot_count(ct)} slots; "
                    f"expected {expected_slots}"
                )

            timings = {
                "linear_s": 0.0,
                "hybrid_relu_s": 0.0,
                "mpc_s": 0.0,
                "mpc_elements": 0,
                "base_ots": 0,
                # §4 r_h witness: one ①_k send and one ②_k receive per hybrid
                # layer.  The MPC (SCI silent-OT extension) runs between them,
                # so the OT response shares the same round as Enc_pk(c_c).
                "hybrid_wire_sends": 0,
                "hybrid_wire_recvs": 0,
            }
            t0 = time.perf_counter()
            mask_session.begin_query(query_index)
            ct_out = encrypted_forward(
                model, be, ctx, ct, wire, timings,
                mask_session=mask_session,
                query_index=query_index,
                sci_session=sci_session,
            )
            timings["forward_s"] = time.perf_counter() - t0

            if requested_output == "index":
                secure_argmax_server(
                    be, ctx, ct_out, wire, timings,
                    mask_session, query_index, model,
                )
            else:
                wire.send(b"CTXT", b"\x01" + be.serialize(ct_out))
            wire.recv(b"BYE.")

            stats = {
                "role": "server/party0",
                **profile_info(be.name, mask_domain_aligned=aligned),
                "architecture": getattr(model, "architecture", "hybrid_mlp"),
                "dims": model.dims,
                "hybrid_layers": model.hybrid_layers,
                "schedule": list(model.schedule),
                "protocol_online_rounds": (
                    2
                    + (
                        len(model.schedule)
                        if obfuscate_schedule else model.hybrid_layers
                    )
                    + int(requested_output == "index")
                ),
                "protocol_online_messages": (
                    2
                    + 2 * (
                        len(model.schedule)
                        if obfuscate_schedule else model.hybrid_layers
                    )
                    + 2 * int(requested_output == "index")
                ),
                "mask_domain_aligned": aligned,
                "schedule_obfuscated": bool(obfuscate_schedule),
                # §4 r_h witness: complete ①_k -> ②_k exchanges.  Real hybrid
                # layers only; dummy obfuscation legs are not counted.
                "hybrid_roundtrips": min(
                    timings["hybrid_wire_sends"], timings["hybrid_wire_recvs"]
                ),
                "hybrid_r_h": (
                    min(timings["hybrid_wire_sends"], timings["hybrid_wire_recvs"])
                    / model.hybrid_layers
                    if model.hybrid_layers else 0.0
                ),
                "output_mode": requested_output,
                "query_index": query_index,
                "reused_context": reused_context,
                "wall_s": round(time.perf_counter() - query_started, 4),
                "timings": {k: round(v, 6) for k, v in timings.items()},
                "bytes_sent": wire.cc.sent - bytes_sent_start,
                "bytes_received": wire.cc.recvd - bytes_received_start,
            }
            if reference_input is not None:
                ref, _ = model.plain_forward(reference_input)
                stats["reference_logits"] = [round(float(v), 6) for v in ref]
            wire.send_json(b"STAT", stats)
            mask_session.end_query(query_index)
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
        mask_session.abort()
        wire.send_error(exc)
        raise


def serve(host: str, port: int, model, backend: str,
          once: bool = True, mask_domains=None,
          obfuscate_schedule: bool = False) -> None:
    be = make_backend(backend)
    with socket.create_server((host, port), reuse_port=False) as srv:
        print(
            f"[server] listening on {host}:{port}  backend={backend}  "
            f"architecture={getattr(model, 'architecture', 'hybrid_mlp')}  "
            f"dims={model.dims}"
        )
        while True:
            conn, addr = srv.accept()
            print(f"[server] connection from {addr[0]}:{addr[1]}")
            with conn:
                try:
                    stats = serve_connection(
                        conn, be, model, mask_domains=mask_domains,
                        obfuscate_schedule=obfuscate_schedule,
                    )
                    print("[server] " + json.dumps(stats, ensure_ascii=False))
                except Exception as exc:
                    print(f"[server] failed: {type(exc).__name__}: {exc}")
                    if once:
                        raise
            if once:
                return


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI desktop server (party0)")
    ap.add_argument("--root", type=Path,
                    default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--dataset", choices=["mnist", "cifar10", "uci_har"],
                    default=None,
                    help="load the canonical train_hybrid.py architecture")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--dims", type=_as_dims, default=[64, 32, 10],
                    help="comma-separated layer sizes, e.g. 784,128,64,10")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--backend", choices=["tenseal", "seal", "stub"], default="seal")
    ap.add_argument("--serve", action="store_true",
                    help="keep serving instead of handling one connection")
    ap.add_argument("--weights", type=Path, default=None,
                    help="canonical .pt checkpoint, or legacy MLP .npz")
    ap.add_argument("--schedule", type=_as_schedule, default=None,
                    help="per-hidden-layer 0=P2, 1=ReLU; e.g. 0,1,0")
    ap.add_argument("--plan", type=Path, default=None,
                    help="planner JSON; its s vector overrides --schedule")
    ap.add_argument("--mask-domains", type=Path, default=None,
                    help="M2 JSON exported by analysis/export_mask_domains.py")
    ap.add_argument("--obfuscate-schedule", action="store_true",
                    help="§7.2 dummy ciphertext pairs on every nonlinear layer")
    args = ap.parse_args()

    mask_domains = None
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        planned = _as_schedule(",".join(str(v) for v in plan.get("s", [])))
        if args.schedule is not None and planned != args.schedule:
            raise SystemExit("--plan and --schedule disagree")
        args.schedule = planned
    if args.dataset:
        schedule = args.schedule or (1, 1)
        model = load_canonical_model(
            args.root, args.dataset, schedule, args.seed, args.weights,
        )
        tag = "".join(str(bit) for bit in schedule)
        default_domains = (
            args.root / "datas" / "mask_domains"
            / f"{args.dataset}_s{tag}_seed{args.seed}.json"
        )
        domain_path = args.mask_domains or default_domains
        if not domain_path.exists():
            raise SystemExit(
                f"M2 mask-domain file not found: {domain_path}\n"
                "Export it first with analysis/export_mask_domains.py."
            )
        mask_domains = load_mask_domains(
            domain_path,
            output_layer_id=model.output_layer_id(),
            expected_dataset=args.dataset,
            expected_schedule=schedule,
            expected_seed=args.seed,
            expected_dims=model.client_spec()["nonlinear_dims"],
        )
        print(
            f"[server] canonical={args.dataset} schedule={schedule} "
            f"checkpoint={model.checkpoint} domains={domain_path}"
        )
    else:
        model = HybridMLP(args.dims, args.seed, args.schedule)

    if not args.dataset and args.weights:
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
        model = HybridMLP(dims, args.seed, args.schedule)
        model.weights, model.biases = weights, biases
        try:
            model.validate_weights()
        except ValueError as exc:
            raise SystemExit(f"invalid weights: {exc}") from exc
        print(f"[server] loaded {i} layers from {args.weights}: dims={dims}")

    serve(
        args.host, args.port, model, args.backend,
        once=not args.serve, mask_domains=mask_domains,
        obfuscate_schedule=args.obfuscate_schedule,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

