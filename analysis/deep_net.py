"""Deep networks on the LitePI protocol (conv / pool / FC with K > 2).

A network is a list of ops; every "act" is one nonlinear layer with a schedule
bit.  The two NonlinearEval primitives keep their protocol meaning, and the
only new question is how the *next* linear layer gets its input:

  s_k = 1 (Hybrid-ReLU, protocol §4)
      After the MPC the activation exists as additive shares, and pooling and
      im2col are public linear re-indexings, so each party transforms its own
      share (zero communication): AvgPool on both shares, then im2col for a
      conv or the flat vector for an FC.  The client encrypts its transformed
      share (②_k, one ciphertext per block), the server adds its own in the
      clear, and the conv runs as litepi_seal.matmul_rows.  The double
      re-encoding is what deep_hybrid.py checks.

  s_k = 0 (HE-poly, protocol §3)
      The activation exists only as ciphertext, in (C, H, W) flatten order, so
      the server runs the next linear map directly on that layout:
      (pool +) conv / FC become one sparse matrix for litepi_seal.linear_sparse
      (multiply first, rotate second).  This is the P2 -> conv path; the
      one-level P2 folds a2 and d into the sparse matrix as on the canonical
      nets.

The wire is the §2.2 message table with blocks: M1 and ②_k carry `blocks`
ciphertexts, ①_k carries one ciphertext per `slots` activations.  The geometry
is public and goes to the client in SPEC ("deep").

Model files: see from_npz(); train_mid.py exports them.

    analysis/check_deep_net.py     correctness on every schedule (synthetic)
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path

import numpy as np

from deep_hybrid import im2col_rows, to_blocks
from params_pi import POLY_MODULUS_DEGREE
from pi_proto import F, K_COMPARE, K_RING, ProtocolError, ots_per_relu
from sci_relu import ring_to_float
from secure_relu_net import run_party_batch

P2_A0, P2_A1, P2_A2 = 0.0625, 0.5, 0.5
# One CKKS ciphertext carries N/2 slots; the deep profile (LITEPI_PARAMS=deep,
# N=16384) doubles that, so the block layout must follow the profile rather
# than a fixed 4096.
SLOTS = POLY_MODULUS_DEGREE // 2


def _pool(x: np.ndarray, k: int) -> np.ndarray:
    c, h, w = x.shape
    return x.reshape(c, h // k, k, w // k, k).mean(axis=(2, 4))


def _conv_out_shape(shape, op) -> tuple[int, int, int]:
    _, h, w = shape
    k, s, p = op["weight"].shape[-1], op["stride"], op["pad"]
    return (op["weight"].shape[0], (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1)


def linear_coo(chain: list[dict], shape: tuple) -> tuple:
    """(pool *) then one conv/FC as COO: (in_idx, out_idx, values, bias, out_shape).

    Indices are positions in the flat (C, H, W) vector.  Entries that fall in
    the zero padding are dropped, which is also what the folded P2 needs (the
    padding is not an activation, so it contributes neither a2 nor d).
    """
    in_shape = tuple(shape)
    pools = [op["k"] for op in chain if op["op"] == "pool"]
    linear = chain[-1]
    for k in pools:
        shape = (shape[0], shape[1] // k, shape[2] // k)
    if linear["op"] == "fc":
        weight = linear["weight"]
        out_n, in_n = weight.shape
        if in_n != int(np.prod(shape)):
            raise ValueError(f"FC expects {in_n} inputs, shape is {shape}")
        oi, ii = np.nonzero(weight)
        ii, oi, val = ii.astype(np.int64), oi.astype(np.int64), weight[oi, ii]
        bias = np.asarray(linear["bias"], dtype=np.float64)
        out_shape: tuple = (out_n,)
    else:
        weight = linear["weight"]
        co, ci_n, k, _ = weight.shape
        s, p = linear["stride"], linear["pad"]
        _, h, w = shape
        out_shape = _conv_out_shape(shape, linear)
        _, oh, ow = out_shape
        coc, cic, ki, kj, oy, ox = np.meshgrid(
            np.arange(co), np.arange(ci_n), np.arange(k), np.arange(k),
            np.arange(oh), np.arange(ow), indexing="ij")
        y = oy * s + ki - p
        x = ox * s + kj - p
        keep = (y >= 0) & (y < h) & (x >= 0) & (x < w)
        ii = ((cic * h + y) * w + x)[keep].astype(np.int64)
        oi = ((coc * oh + oy) * ow + ox)[keep].astype(np.int64)
        val = weight[coc[keep], cic[keep], ki[keep], kj[keep]]
        bias = np.repeat(np.asarray(linear["bias"], dtype=np.float64), oh * ow)
    # Expand through the pools, last pool first: each pooled element is the
    # mean of k^2 elements of the finer grid.
    grid = in_shape
    sizes = []
    for k in pools:
        sizes.append((grid, k))
        grid = (grid[0], grid[1] // k, grid[2] // k)
    for fine, k in reversed(sizes):
        c, h, w = fine
        ph, pw = h // k, w // k
        pc = ii // (ph * pw)
        rest = ii % (ph * pw)
        py, px = rest // pw, rest % pw
        dy, dx = np.meshgrid(np.arange(k), np.arange(k), indexing="ij")
        dy, dx = dy.ravel(), dx.ravel()
        ii = (((pc[:, None] * h + py[:, None] * k + dy) * w
               + px[:, None] * k + dx).ravel())
        oi = np.repeat(oi, k * k)
        val = np.repeat(val, k * k) / (k * k)
    return ii, oi, val, bias, out_shape


class DeepNet:
    """Sequential conv / act / pool / fc network on the LitePI protocol."""

    architecture = "deep_net"
    schedule_obfuscated = False

    def __init__(self, ops: list[dict], input_shape: tuple, schedule,
                 dataset: str = "synthetic", preprocess: dict | None = None,
                 threads: int | None = None):
        self.ops = ops
        self.input_shape = tuple(input_shape)
        self.schedule = tuple(int(b) for b in schedule)
        self.dataset = dataset
        self.preprocess = preprocess or {
            "mode": "identity", "raw_shape": list(input_shape),
            "model_shape": list(input_shape)}
        self.threads = int(threads or os.environ.get("LITEPI_THREADS", 1))
        acts = [op for op in ops if op["op"] == "act"]
        if len(acts) != len(self.schedule):
            raise ValueError(f"{len(acts)} activations, schedule has "
                             f"{len(self.schedule)} bits")
        if ops[0]["op"] != "conv":
            raise ValueError("the client encodes M1 for a leading conv")
        for k, op in enumerate(acts):
            op["k"] = k
        self.shapes = self._walk_shapes()
        self.input_dim = int(np.prod(self.input_shape))
        self.input_blocks = len(self._first_rows(np.zeros(self.input_shape)))
        self.input_slots = SLOTS
        self.dims = [self.input_dim] + [int(np.prod(s)) for s in self.shapes
                                        if s is not None]

    # ---------------------------------------------------------- geometry
    def _walk_shapes(self) -> list:
        """Shape of the value *entering* each op."""
        shapes, shape = [], self.input_shape
        for op in self.ops:
            shapes.append(shape)
            if op["op"] == "conv":
                shape = _conv_out_shape(shape, op)
            elif op["op"] == "pool":
                shape = (shape[0], shape[1] // op["k"], shape[2] // op["k"])
            elif op["op"] == "fc":
                shape = (op["weight"].shape[0],)
        return shapes

    def _chain(self, i: int) -> list[dict]:
        """Ops after the act at index i, up to and including the next linear."""
        chain = []
        for op in self.ops[i + 1:]:
            chain.append(op)
            if op["op"] in ("conv", "fc"):
                return chain
        raise ValueError("every activation must be followed by a linear layer")

    def _first_rows(self, x: np.ndarray) -> list[np.ndarray]:
        op = self.ops[0]
        rows, _, _ = im2col_rows(x, op["weight"].shape[-1], op["stride"], op["pad"])
        return to_blocks(rows, SLOTS)

    @property
    def hybrid_layers(self) -> int:
        return sum(self.schedule)

    def output_layer_id(self) -> int:
        return len(self.schedule)

    def nonlinear_dims(self) -> list[int]:
        return [int(np.prod(self.shapes[i])) for i, op in enumerate(self.ops)
                if op["op"] == "act"]

    def client_spec(self) -> dict:
        legs = []
        for i, op in enumerate(self.ops):
            if op["op"] != "act" or not self.schedule[op["k"]]:
                continue
            n = int(np.prod(self.shapes[i]))
            chain = self._chain(i)
            legs.append({"k": op["k"], "elements": n,
                         "blocks": math.ceil(n / SLOTS),
                         "shape": [int(v) for v in self.shapes[i]],
                         "post": _post_of(chain)})
        first = self.ops[0]
        return {
            "architecture": self.architecture, "dataset": self.dataset,
            "dims": self.dims, "input_dim": self.input_dim,
            "input_slots": self.input_slots,
            "nonlinear_dims": self.nonlinear_dims(),
            "preprocess": self.preprocess,
            "schedule": list(self.schedule), "hybrid_layers": self.hybrid_layers,
            "encoding": "im2col",
            "deep": {
                "input_shape": list(self.input_shape),
                "first": [int(first["weight"].shape[-1]), int(first["stride"]),
                          int(first["pad"])],
                "input_blocks": self.input_blocks,
                "legs": legs,
            },
        }

    def protocol_ops(self, schedule=None) -> list[tuple]:
        """pi_server.protocol_trace for this network."""
        bits = list(self.schedule if schedule is None else schedule)
        ops: list[tuple] = [("fresh", "M1")]
        for op in self.ops:
            if op["op"] in ("conv", "fc"):
                ops.append(("rescale", 1))
            elif op["op"] == "act":
                ops.append(("nl", op["k"]))
                if bits[op["k"]]:
                    ops += [("decrypt", ("hyb", op["k"])), ("fresh", ("re", op["k"]))]
                else:
                    ops.append(("rescale", 1))
        ops.append(("decrypt", "out"))
        return ops

    # --------------------------------------------------------- plaintext
    def plain_forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """(logits, activations entering each act)."""
        v = np.asarray(x, dtype=np.float64).reshape(self.input_shape)
        zs = []
        for op in self.ops:
            if op["op"] == "conv":
                k = op["weight"].shape[-1]
                rows, oh, ow = im2col_rows(v, k, op["stride"], op["pad"])
                out = (op["weight"].reshape(op["weight"].shape[0], -1) @ rows
                       + np.asarray(op["bias"])[:, None])
                v = out.reshape(op["weight"].shape[0], oh, ow)
            elif op["op"] == "pool":
                v = _pool(v, op["k"])
            elif op["op"] == "fc":
                v = op["weight"] @ v.reshape(-1) + op["bias"]
            else:
                zs.append(v.reshape(-1).copy())
                v = (np.maximum(v, 0.0) if self.schedule[op["k"]]
                     else P2_A0 + P2_A1 * v + P2_A2 * v * v)
        return v, zs

    # ------------------------------------------------------------ server
    def _hybrid_leg(self, be, ctx, cts, wire, timings, mask_session, query_index,
                    k: int, shape, sci_session):
        """Steps 1-5 of §4.1 over blocks; returns the server's output share."""
        import pi_server as S
        sizes = [be.slot_count(ct) for ct in cts]
        n = sum(sizes)
        domain = S.fit_mask_domain(be, cts[0], mask_session.domain(k))
        if domain is not mask_session.domain(k):
            getattr(wire, "fitted_domains", {})[str(k)] = domain.to_dict()
        exact = domain.rule == S.RULE_V3
        if exact:
            ints = mask_session.draw_ints(query_index, k, n, domain=domain)
        else:
            r = np.asarray(mask_session.draw(query_index, k, n, domain=domain),
                           dtype=np.float64)
        at = 0
        for ct, size in zip(cts, sizes):
            he_t0 = time.perf_counter()
            enc = (be.encrypt_like_exact(ctx, ints[at:at + size], ct) if exact
                   else be.encrypt_like(ctx, r[at:at + size], ct))
            at += size
            ct_zr = be.add_cipher(ct, enc)
            timings["server_he_compute_s"] = timings.get("server_he_compute_s", 0.0) + (
                time.perf_counter() - he_t0)
            ct_zr = S._shrink(be, wire, ct_zr, f"hyb{k}")
            wire.send(b"CTXT", b"\x00" + be.serialize(ct_zr))
        timings["hybrid_wire_sends"] += 1
        if exact:
            mod = 1 << K_RING
            s_server = ring_to_float(
                np.array([(-v) % mod for v in ints], dtype=np.uint64), F, K_RING)
        else:
            s_server = -r
        t0 = time.perf_counter()
        if sci_session is not None:
            c_server = np.asarray(sci_session.relu(s_server.tolist(), F, K_RING),
                                  dtype=np.float64)
        else:
            c_server = np.asarray(
                run_party_batch(wire.mpc_ctx(), s_server.tolist(), K_RING, F, K_COMPARE),
                dtype=np.float64)
            timings["base_ots"] += n * ots_per_relu()
        timings["mpc_s"] += time.perf_counter() - t0
        timings["mpc_elements"] += n
        return c_server.reshape(shape)

    def _recv_share_blocks(self, be, ctx, wire, count: int) -> list:
        cts = []
        for _ in range(count):
            _, payload = wire.recv(b"CTXT")
            if not payload or payload[0] != 0:
                raise ProtocolError("expected re-encrypted client share")
            cts.append(be.deserialize(ctx, payload[1:]))
        return cts

    def _conv_rows(self, be, ctx, cts, index: int):
        """ops[index] on im2col blocks (M1 or a re-encoded share), plus bias."""
        op = self.ops[index]
        weight = op["weight"]
        _, oh, ow = _conv_out_shape(self.shapes[index], op)
        windows = oh * ow
        outs = ctx.matmul_rows(cts, weight.reshape(weight.shape[0], -1), windows)
        rows = SLOTS // windows
        bias = np.asarray(op["bias"], dtype=np.float64)
        return [be.add_plain(ct, np.repeat(bias[c * rows:(c + 1) * rows], windows))
                for c, ct in enumerate(outs)]

    def _bias_blocks(self, be, cts, bias: np.ndarray):
        out, at = [], 0
        for ct in cts:
            n = be.slot_count(ct)
            out.append(be.add_plain(ct, bias[at:at + n]))
            at += n
        return out

    def encrypted_forward(self, be, ctx, cts, wire, timings: dict,
                          mask_session=None, query_index: int = 0,
                          sci_session=None) -> object:
        if mask_session is None:
            raise ProtocolError("deep networks need a mask session")
        t0 = time.perf_counter()
        cts = self._conv_rows(be, ctx, list(cts), 0)
        linear_s = time.perf_counter() - t0
        timings["linear_s"] += linear_s
        timings["server_he_compute_s"] = timings.get("server_he_compute_s", 0.0) + linear_s
        i = 1
        while i < len(self.ops):
            op = self.ops[i]
            if op["op"] != "act":
                raise ProtocolError(f"unexpected op {op['op']} after a linear layer")
            k, shape = op["k"], self.shapes[i]
            chain = self._chain(i)
            last_i = i + len(chain)
            last = self.ops[last_i]
            t0 = time.perf_counter()
            if self.schedule[k]:
                share = self._hybrid_leg(be, ctx, cts, wire, timings, mask_session,
                                         query_index, k, shape, sci_session)
                blocks = post_share(share, _post_of(chain))
                got = self._recv_share_blocks(be, ctx, wire, len(blocks))
                timings["hybrid_wire_recvs"] += 1
                he_t0 = time.perf_counter()
                cts = [be.add_plain(ct, blk) for ct, blk in zip(got, blocks)]
                timings["server_he_compute_s"] = timings.get("server_he_compute_s", 0.0) + (
                    time.perf_counter() - he_t0)
                timings["hybrid_relu_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                if last["op"] == "conv":
                    cts = self._conv_rows(be, ctx, cts, last_i)
                elif len(cts) == 1:
                    ct = be.mm(cts[0], last["weight"].T)
                    cts = [be.add_plain(ct, last["bias"])]
                else:
                    cts = self._sparse(be, ctx, cts, [last], self.shapes[last_i],
                                       1.0, 0.0)
            else:
                c = P2_A1 / (2.0 * P2_A2)
                d = P2_A0 - P2_A1 * P2_A1 / (4.0 * P2_A2)
                he_t0 = time.perf_counter()
                cts = [be.poly2_shifted(ct, c) for ct in cts]
                timings["server_he_compute_s"] = timings.get("server_he_compute_s", 0.0) + (
                    time.perf_counter() - he_t0)
                timings["hybrid_relu_s"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                cts = self._sparse(be, ctx, cts, chain, shape, P2_A2, d)
            linear_s = time.perf_counter() - t0
            timings["linear_s"] += linear_s
            timings["server_he_compute_s"] = timings.get("server_he_compute_s", 0.0) + linear_s
            i += len(chain) + 1
        if len(cts) != 1:
            raise ProtocolError("the output must fit one ciphertext")
        return cts[0]

    def _sparse(self, be, ctx, cts, chain, shape, a2: float, d: float):
        """Linear layer on the ciphertext layout; folds P2 = a2 (z+c)^2 + d."""
        ii, oi, val, bias, out_shape = linear_coo(chain, shape)
        out_len = int(np.prod(out_shape))
        row_sum = np.bincount(oi, weights=val, minlength=out_len)
        outs = ctx.linear_sparse(cts, ii, oi, val * a2, out_len, self.threads)
        return self._bias_blocks(be, outs, bias + d * row_sum)


def _post_of(chain: list[dict]) -> list[dict]:
    post = [{"pool": c["k"]} for c in chain if c["op"] == "pool"]
    last = chain[-1]
    if last["op"] == "conv":
        post.append({"im2col": [int(last["weight"].shape[-1]), int(last["stride"]),
                                int(last["pad"])]})
    return post


def post_share(share: np.ndarray, post: list[dict]) -> list[np.ndarray]:
    """Public re-layout of one party's output share -> ciphertext-sized blocks."""
    v = share
    for step in post:
        if "pool" in step:
            v = _pool(v, int(step["pool"]))
        else:
            k, stride, pad = step["im2col"]
            rows, _, _ = im2col_rows(v, k, stride, pad)
            return to_blocks(rows, SLOTS)
    flat = v.reshape(-1)
    return [flat[i:i + SLOTS] for i in range(0, flat.size, SLOTS)]


# ---------------------------------------------------------------- client

def client_query(be, ctx, wire, spec: dict, x: np.ndarray, timings: dict,
                 sci_session, encrypt_blob, plan_chain_index, prepare,
                 output_mode: str):
    """M1, the hybrid legs, M3.  Called from pi_client._query_once."""
    if output_mode != "logits":
        raise ProtocolError("deep networks support output mode A only")
    deep = spec["deep"]
    arr = np.asarray(prepare(x, spec), dtype=np.float64).reshape(deep["input_shape"])
    k, stride, pad = deep["first"]
    rows, _, _ = im2col_rows(arr, k, stride, pad)
    m1 = plan_chain_index(spec, "M1", output_mode)
    for blk in to_blocks(rows, SLOTS):
        wire.send(b"CTXT", encrypt_blob(be, ctx, blk, timings, "encrypt_input_s", m1))
    for leg in deep["legs"]:
        shares = []
        for _ in range(int(leg["blocks"])):
            _, payload = wire.recv(b"CTXT")     # waits for the server's linear layer
            if not payload or payload[0] != 0:
                raise ProtocolError("expected ①_k block")
            t0 = time.perf_counter()
            ct = be.deserialize(ctx, payload[1:])
            if spec.get("mask_exact"):
                shares.append(ring_to_float(be.decrypt_ring(ct, F, K_RING), F, K_RING))
            else:
                shares.append(np.asarray(be.decrypt(ct), dtype=np.float64))
            timings["decrypt_s"] += time.perf_counter() - t0
        share = np.concatenate(shares)
        t0 = time.perf_counter()
        if sci_session is not None:
            out = np.asarray(sci_session.relu(share.tolist(), F, K_RING),
                             dtype=np.float64)
        else:
            out = np.asarray(run_party_batch(wire.mpc_ctx(), share.tolist(), K_RING,
                                             F, K_COMPARE), dtype=np.float64)
            timings["base_ots"] += share.size * ots_per_relu()
        timings["mpc_s"] += time.perf_counter() - t0
        timings["mpc_elements"] += int(share.size)
        idx = plan_chain_index(spec, f"re{leg['k']}", output_mode)
        for blk in post_share(out.reshape(leg["shape"]), leg["post"]):
            wire.send(b"CTXT", b"\x00" + encrypt_blob(be, ctx, blk, timings,
                                                      "reencrypt_s", idx))
    _, payload = wire.recv(b"CTXT")
    if not payload or payload[0] != 1:
        raise ProtocolError("expected M3")
    t0 = time.perf_counter()
    logits = np.asarray(be.decrypt(be.deserialize(ctx, payload[1:])), dtype=np.float64)
    timings["decrypt_s"] += time.perf_counter() - t0
    return "logits", logits


# ------------------------------------------------------------------ model I/O

def mid_ops(rng: np.random.Generator, in_ch: int = 3, c1: int = 32, c2: int = 64,
            hidden: int = 256, classes: int = 10) -> list[dict]:
    """Random weights with the geometry of alignment model B (plan §2.1, conv3)."""
    conv = lambda co, ci, s: {"op": "conv", "stride": s, "pad": 1,
                              "weight": rng.normal(0, 1 / math.sqrt(ci * 9), (co, ci, 3, 3)),
                              "bias": rng.normal(0, 0.05, co)}
    fc = lambda o, i: {"op": "fc", "weight": rng.normal(0, 1 / math.sqrt(i), (o, i)),
                       "bias": rng.normal(0, 0.05, o)}
    return [conv(c1, in_ch, 2), {"op": "act"}, conv(c2, c1, 2), {"op": "act"},
            conv(c2, c2, 1), {"op": "act"}, {"op": "pool", "k": 2},
            fc(hidden, c2 * 4 * 4), {"op": "act"}, fc(classes, hidden)]


def from_npz(path: Path, schedule, threads: int | None = None) -> DeepNet:
    """Load <tag>.npz + <tag>.json written by train_mid.py."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    data = np.load(path)
    ops = []
    for entry in meta["ops"]:
        op = dict(entry)
        if op["op"] in ("conv", "fc"):
            op["weight"] = data[op.pop("weight_key")].astype(np.float64)
            op["bias"] = data[op.pop("bias_key")].astype(np.float64)
        ops.append(op)
    pre = meta.get("preprocess")
    return DeepNet(ops, tuple(meta["input_shape"]), schedule,
                   dataset=meta.get("dataset", "cifar10_mid"), preprocess=pre,
                   threads=threads)
