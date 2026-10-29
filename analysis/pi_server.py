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
import math

from params_pi import COEFF_MOD_BIT_SIZES, GLOBAL_SCALE_BITS, POLY_MODULUS_DEGREE
from pi_proto import (
    F,
    K_COMPARE,
    K_RING,
    WIRE_GUARD_BITS,
    WIRE_PROFILES,
    ProtocolError,
    Wire,
    WireProfile,
    make_backend,
    needed_modulus_bits,
    ots_per_relu,
    seal_coeff_primes,
    profile_info,
)
from secure_relu_net import run_party_batch
from secure_argmax import run_party_argmax
from mask_session import (
    DEFAULT_DOMAIN,
    MaskDomain,
    MaskSession,
    _grid_draw,
    load_mask_domains,
)
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
                        ct = dummy_round_server(be, ctx, ct, wire, i)
                    ct = he_poly_server(be, ct)
                timings["hybrid_relu_s"] += time.perf_counter() - t0
        return ct


def draw_mask(n: int) -> np.ndarray:
    """Fresh M2 mask without a session (legacy self-test path only).

    Uses the widest admissible domain (+/- 2^MASK_MAX_BITS on the 2^-F grid)
    from the OS CSPRNG; protocol runs always go through ``MaskSession``.
    """
    if n <= 0:
        raise ValueError("mask size must be positive")
    stream = secrets.token_bytes(16 * n)
    return np.asarray(_grid_draw(stream, n, DEFAULT_DOMAIN), dtype=np.float64)


def he_poly_server(be, ct):
    """s_k=0: P2 activation evaluated entirely on ciphertext."""
    return be.poly2(ct, P2_A0, P2_A1, P2_A2)


def he_poly_layer(be, ctx, ct, wire: Wire, model, layer_index: int):
    """§3.1 1-level P2: return ct_{(z+c)^2} plus the folded (a2, d) coefficients.

    P2(z) = a2(z + c)^2 + d with c = a1/(2a2), d = a0 - a1^2/(4a2), so the
    expensive part is a single square+rescale.  The a2 multiply and the d shift
    are folded into the next linear layer's plaintext weight matrix and bias
    (server-side constants, zero messages, zero extra depth), which is what
    keeps NonlinearEval at the one level the depth table budgets.
    """
    if getattr(model, "schedule_obfuscated", False):
        ct = dummy_round_server(be, ctx, ct, wire, layer_index)
    c = P2_A1 / (2.0 * P2_A2)
    d = P2_A0 - P2_A1 * P2_A1 / (4.0 * P2_A2)
    return be.poly2_shifted(ct, c), (P2_A2, d)


def fold_poly_into(weight_t, bias, weight, folded) -> tuple:
    """Absorb (a2, d) from ``he_poly_layer`` into the following mm plaintext."""
    a2, d = folded
    row_sum = np.asarray(weight.sum(dim=1).numpy(), dtype=np.float64)
    return a2 * weight_t, np.asarray(bias, dtype=np.float64) + d * row_sum


def dummy_round_server(be, ctx, ct, wire: Wire, layer_index: int):
    """§7.2 dummy leg on a P2 layer: a complete Hybrid-ReLU exchange on this
    layer's z whose result is discarded; the P2 path then continues on ``ct``.

    A network observer must not be able to tell it from a real leg, so it
    reuses the real code path: frame kind 0 (the old kind byte 3 was sent in
    the clear), a fresh M2 mask under the layer's own (q, k) PRG label, the
    same SCI ReLU traffic (the old dummy produced none), and the obfuscation
    level for ① (_obf_align).  The old ``ct * 0`` was also a transparent
    all-zero ciphertext that zstd compresses to almost nothing.
    The client sees z_k + r_k, exactly the view of a real hybrid layer.
    """
    q = wire.query_ctx
    timings = q["timings"]
    counters = (timings["hybrid_wire_sends"], timings["hybrid_wire_recvs"])
    hybrid_relu_server(
        be, ctx, ct, wire, wire.mpc_ctx(), timings,
        mask_session=q["mask_session"], query_index=q["query_index"],
        layer_index=layer_index, sci_session=q["sci_session"],
    )
    # r_h witnesses count real hybrid layers only.
    timings["hybrid_wire_sends"], timings["hybrid_wire_recvs"] = counters
    timings["dummy_legs"] = timings.get("dummy_legs", 0) + 1
    return ct


def fit_mask_domain(be, ct, domain: MaskDomain) -> MaskDomain:
    """Shrink an M2 domain so that z + r still fits the ciphertext modulus.

    The full-width mask needs log2(A_k + B_k) + log2(scale) + guard bits.  A
    ciphertext deep in the chain (output mode B after s=00 has 61 bits at
    scale 2^29) cannot hold 2^42, so the width is lowered to what fits and
    sigma_k drops with it; MaskDomain.for_bound refuses anything below
    MASK_MIN_SIGMA.  The achieved domain is reported in STAT
    (mask_domains_fitted), never silently weakened.
    """
    if not hasattr(be, "modulus_bits") or domain.activation_bound is None:
        return domain
    have = be.modulus_bits(ct)
    if have is None:            # old litepi_seal build: keep the full width
        return domain
    scale = be.scale_of(ct)
    if needed_modulus_bits(domain.half_width + domain.activation_bound,
                           scale) <= have:
        return domain
    # A + B <= 2^(max_bits+1) with max_bits = have - guard - log2(scale) - 1
    max_bits = math.floor(have - WIRE_GUARD_BITS - math.log2(scale)) - 1
    return MaskDomain.for_bound(domain.activation_bound, max_bits=max_bits,
                                fitted_to_modulus_bits=have)


def hybrid_relu_server(be, ctx, ct, wire: Wire, mpc_ctx, timings: dict,
                       mask_session: MaskSession | None = None,
                       query_index: int = 0, layer_index: int = 0,
                       sci_session=None):
    """One hybrid ReLU, server side (steps 1 and 5 of the five-step flow)."""
    n = be.slot_count(ct)
    if n <= 0:
        raise ProtocolError("cannot run ReLU on an empty ciphertext")
    # ct_z is replaced by a fresh ciphertext after step 5, so lowering it
    # first costs nothing; under obfuscation it fixes the size of ①_k.
    ct = _obf_align(be, wire, ct, f"nl{layer_index}")

    if mask_session is None:
        r = draw_mask(n)
    else:
        domain = fit_mask_domain(be, ct, mask_session.domain(layer_index))
        if domain is not mask_session.domain(layer_index):
            getattr(wire, "fitted_domains", {})[str(layer_index)] = domain.to_dict()
        r = np.asarray(
            mask_session.draw(query_index, layer_index, n, domain=domain),
            dtype=np.float64,
        )
    # Protocol step 1: ct_{z+r} = ct_z + Enc_pk(r).  The server has only the
    # public context and therefore encrypts the mask with pk, never with sk.
    ct_zr = be.add_cipher(ct, be.encrypt_like(ctx, r, ct))
    ct_zr = _shrink(be, wire, ct_zr, f"hyb{layer_index}")
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
        ct = dummy_round_server(be, ctx, ct, wire, layer_index)
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
        ct, folded = he_poly_layer(be, ctx, ct, wire, model, 0)
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
        ct, folded = he_poly_layer(be, ctx, ct, wire, model, 1)
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
    ct = _obf_align(be, wire, ct, "out")
    domain = fit_mask_domain(be, ct, mask_session.domain(layer_id))
    if domain is not mask_session.domain(layer_id):
        getattr(wire, "fitted_domains", {})[str(layer_id)] = domain.to_dict()
    r = np.asarray(
        mask_session.draw(query_index, layer_id, classes, domain=domain),
        dtype=np.float64,
    )
    ct_masked = _shrink(be, wire, be.add_plain(ct, r), "argmax")
    wire.send(b"CTXT", b"\x02" + be.serialize(ct_masked))

    t0 = time.perf_counter()
    server_share = run_party_argmax(
        wire.mpc_ctx(), (-r).tolist(), K_RING, F, K_COMPARE,
    )
    timings["mpc_s"] += time.perf_counter() - t0
    timings["mpc_elements"] += classes
    timings["base_ots"] += (classes - 1) * max(K_COMPARE - 1, 1)
    wire.send_json(b"IDXS", {"share": server_share})
    return server_share


# Bounds used only for modulus sizing (public: derived from parameters).
INPUT_ABS_BOUND = 2.0 ** 8          # canonical inputs are normalised
OUTPUT_ABS_DEFAULT = 2.0 ** 16      # logits bound when no M2 output domain
SHARE_ABS_BOUND = 2.0 ** (K_RING - F - 1)   # |c_c|, |c_s| after ring_to_float


def protocol_trace(model, schedule=None) -> list[tuple]:
    """The ciphertext trace as ops, for modulus planning.

    ("fresh", tag)          a client encryption: "M1" or ("re", k)
    ("rescale", n)          n rescales on the server
    ("decrypt", tag)        a client decryption: ("hyb", k) or "out"
    ("nl", k)               marker: nonlinear layer k starts here
    ``schedule`` overrides model.schedule (obfuscation_levels uses all zeros).
    Levels follow litepi_seal's policy (one rescale per conv+pack, mm and
    one-level P2; the general quadratic poly2_eval of HybridMLP costs two).
    """
    schedule = list(model.schedule if schedule is None else schedule)
    ops: list[tuple] = [("fresh", "M1")]
    if (isinstance(model, CanonicalCNN)
            or getattr(model, "architecture", None) == "canonical_cnn"):
        ops.append(("rescale", 1))                       # conv + pack
        for k, bit in enumerate(schedule):
            ops.append(("nl", k))
            if bit:
                ops += [("decrypt", ("hyb", k)), ("fresh", ("re", k))]
            else:
                ops.append(("rescale", 1))               # (z+c)^2, folded P2
            ops.append(("rescale", 1))                   # fc1 / fc2
    else:
        last = len(model.weights) - 1
        for i in range(len(model.weights)):
            ops.append(("rescale", 1))                   # mm
            if i < last:
                ops.append(("nl", i))
                if schedule[i]:
                    ops += [("decrypt", ("hyb", i)), ("fresh", ("re", i))]
                else:
                    ops.append(("rescale", 2))           # poly2_eval
    ops.append(("decrypt", "out"))
    return ops


def wire_plan(model, mask_session: MaskSession, wire: WireProfile) -> dict:
    """Modulus plan for mod-switch-before-send and short-chain encryption.

    Every value that crosses the wire is bounded by public parameters: the M2
    domain A_k + B_k at ①_k, the output bound at M3 (logits) or the output
    M2 domain (mode B).  ``send_abs`` carries those magnitudes; the backend
    sizes the modulus from them and the ciphertext's actual scale.

    For short-chain encryption the client needs a chain index *before* the
    segment runs, so the scale is simulated exactly: a fresh ciphertext has
    scale 2^21, every rescale maps s -> s^2 / q with q the prime being dropped
    (the profile's 21-bit primes are 1.20M-1.79M, so s drifts upward; it is
    2^29.2 at the s=00 output).  The chain index is the smallest whose
    remaining modulus still holds the segment's end value, for each output
    mode (the client picks by its own mode).
    """
    data_primes = seal_coeff_primes(POLY_MODULUS_DEGREE, COEFF_MOD_BIT_SIZES)[:-1]
    top = len(data_primes) - 1
    fresh_scale = float(2 ** GLOBAL_SCALE_BITS)

    def end_abs(tag, mode: str) -> float:
        if tag != "out":
            dom = mask_session.domain(tag[1])
            return dom.half_width + (dom.activation_bound or 0.0)
        dom = mask_session.domain(output_layer_id(model))
        if mode == "index":
            return dom.half_width + (dom.activation_bound or 0.0)
        return dom.activation_bound or OUTPUT_ABS_DEFAULT

    def fits(c: int, rescales: int, value_abs: float, fresh_abs: float) -> bool:
        if c - rescales < 0:
            return False
        if needed_modulus_bits(fresh_abs, fresh_scale) > \
                sum(math.log2(q) for q in data_primes[:c + 1]):
            return False
        scale = fresh_scale
        for i in range(rescales):
            scale = scale * scale / data_primes[c - i]
        have = sum(math.log2(q) for q in data_primes[:c - rescales + 1])
        return needed_modulus_bits(value_abs, scale) <= have

    def chain_for(start: int, fresh_abs: float, mode: str) -> int:
        rescales, tag = 0, None
        for op in ops[start + 1:]:
            if op[0] == "rescale":
                rescales += op[1]
            elif op[0] == "decrypt":
                tag = op[1]
                break
        for c in range(rescales, top + 1):
            if fits(c, rescales, end_abs(tag, mode), fresh_abs):
                return c
        return top

    ops = protocol_trace(model)
    plan = {"data_primes": data_primes, "send_abs": {},
            "chain_index": {"logits": {}, "index": {}}}
    for i, op in enumerate(ops):
        if op[0] == "decrypt":
            if op[1] == "out":
                plan["send_abs"]["out"] = end_abs("out", "logits")
                plan["send_abs"]["argmax"] = end_abs("out", "index")
            else:
                plan["send_abs"][f"hyb{op[1][1]}"] = end_abs(op[1], "logits")
        elif op[0] == "fresh":
            key = "M1" if op[1] == "M1" else f"re{op[1][1]}"
            bound = INPUT_ABS_BOUND if op[1] == "M1" else SHARE_ABS_BOUND
            for mode in ("logits", "index"):
                plan["chain_index"][mode][key] = (
                    chain_for(i, bound, mode) if wire.short_chain else top)
    # Dummy legs (§7.2) run the real hybrid path and are sized by
    # obfuscation_levels(), not by this plan.
    return plan


def obfuscation_levels(model) -> dict[str, int]:
    """§7.2: the public modulus size of every server->client ciphertext.

    Without this, ①_k and M3 leave at whatever level the schedule produced
    (a hybrid layer resets to the chain top), so their byte sizes alone
    reveal s.  Each sent ciphertext is therefore mod-switched to the level
    the all-P2 trace has at the same position: the lowest level and the
    largest scale any schedule can reach there, so every schedule's value
    fits (fit_mask_domain still narrows a mask that does not).
    Returns {"nl<k>": bits, "out": bits} in litepi_seal's modulus_bits units
    (sum of the remaining data primes' bit counts).
    """
    bits = [int(b) for b in COEFF_MOD_BIT_SIZES[:-1]]   # data primes, bottom first
    top = len(bits) - 1
    levels, rescales = {}, 0
    for op in protocol_trace(model, schedule=[0] * len(model.schedule)):
        if op[0] == "rescale":
            rescales += op[1]
        elif op[0] == "nl":
            levels[f"nl{op[1]}"] = sum(bits[:top - rescales + 1])
        elif op == ("decrypt", "out"):
            levels["out"] = sum(bits[:top - rescales + 1])
    return levels


def _obf_align(be, wire: Wire, ct, key: str):
    """Mod-switch ``ct`` to the schedule-independent level (obfuscation only)."""
    levels = getattr(wire, "obf_bits", None)
    if not levels or key not in levels:
        return ct
    return be.switch_to_bits(ct, levels[key])


def _shrink(be, wire: Wire, ct, key: str):
    if getattr(wire, "obf_bits", None):
        # Already at the public obfuscation level; a scale-dependent
        # mod-switch here would make the size schedule-dependent again.
        return ct
    plan = getattr(wire, "plan", None)
    if plan is None or key not in plan["send_abs"]:
        return ct
    return be.shrink_for_send(ct, plan["send_abs"][key])


def serve_connection(conn: socket.socket, be, model,
                     reference_input=None, timeout: float = 600.0,
                     mask_domains=None,
                     obfuscate_schedule: bool = False,
                     wire_profile: WireProfile | None = None) -> dict:
    conn.settimeout(timeout)
    wire = Wire(conn, party=0)
    if wire_profile is not None:
        be.set_wire(wire_profile)
    if obfuscate_schedule and be.wire.short_chain:
        # A dummy ②_k would have to copy the chain index of a real one; the
        # differing sizes would otherwise reveal the schedule (§7.2).
        raise ProtocolError("short_chain cannot be combined with schedule "
                            "obfuscation; use the 'zstd'/'seeded'/'modswitch' "
                            "wire profiles instead")
    mask_session = MaskSession(domains=mask_domains or {})
    wire.plan = (None if be.wire.is_baseline
                 else wire_plan(model, mask_session, be.wire))
    hybrid_ids = [
        i for i, bit in enumerate(model.schedule) if int(bit) == 1
    ]
    model.mask_domain_aligned = mask_session.covers(hybrid_ids)
    model.schedule_obfuscated = bool(obfuscate_schedule)
    # §7.2 size hiding needs mod-switching to a fixed level (litepi_seal with
    # the wire API).  Without it the obfuscation only hides the frame pattern;
    # STAT reports which one ran.
    wire.obf_bits = (obfuscation_levels(model)
                     if obfuscate_schedule and hasattr(be, "switch_to_bits")
                     and be.supports_switch_to_bits() else None)
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
            "wire": be.wire.to_dict(),
            "wire_plan": wire.plan,
            # The client already knows s; this tells it that every nonlinear
            # layer gets one ①/② leg (dummy legs included), in layer order.
            "schedule_obfuscated": bool(obfuscate_schedule),
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
            wire.fitted_domains = {}
            wire.query_ctx = {"timings": timings, "mask_session": mask_session,
                              "query_index": query_index,
                              "sci_session": sci_session}
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
                ct_out = _shrink(be, wire, _obf_align(be, wire, ct_out, "out"),
                                 "out")
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
                # M2 provenance: rule, log2 A_k and sigma_k of every layer that
                # drew a mask in this query (hybrid layers + output mode B).
                "mask_domains": {
                    str(i): mask_session.domain(i).to_dict()
                    for i in sorted(set(hybrid_ids) | (
                        {output_layer_id(model)}
                        if requested_output == "index" else set()))
                },
                # Domains narrowed to fit a ciphertext's remaining modulus
                # (fit_mask_domain); empty when every layer used full width.
                "mask_domains_fitted": dict(wire.fitted_domains),
                "schedule_obfuscated": bool(obfuscate_schedule),
                "obfuscation_level_aligned": bool(wire.obf_bits),
                "obfuscation_levels_bits": wire.obf_bits,
                "dummy_legs": int(timings.get("dummy_legs", 0)),
                "wire": be.wire.to_dict(),
                "wire_plan": wire.plan,
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
          obfuscate_schedule: bool = False, wire: str = "baseline") -> None:
    be = make_backend(backend, wire)
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
    ap.add_argument("--wire", choices=sorted(WIRE_PROFILES), default="baseline",
                    help="E4 wire profile (baseline reproduces the recorded "
                         "bytes; compact = zstd + seeded + modswitch + "
                         "short-chain)")
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
        obfuscate_schedule=args.obfuscate_schedule, wire=args.wire,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

