"""Correctness check of the wire profiles on the canonical protocol trace.

Runs the real server (pi_server.serve_connection -> canonical_encrypted_forward)
and the real client (pi_client.run_client) over a localhost socket with the
SEAL backend, for every schedule and wire profile, and compares the logits
with a numpy plaintext forward of the same weights.

Test-only substitutions (this script produces no paper numbers):
  * weights: random numpy weights with the canonical MNIST geometry, so no
    torch / checkpoint is needed;
  * MPC: an in-process stand-in for SCI that pairs the two parties' ring
    shares, computes ReLU on the reconstructed ring value and returns a fresh
    uniform split - the same interface and ring arithmetic as
    sci_relu.SciReLUSession.  It is NOT secure and exists only to exercise
    the HE / wire path (masks, mod-switch, short-chain, re-encryption).

    python analysis/check_wire_canonical.py
    python analysis/check_wire_canonical.py --profiles baseline,compact --schedules 11
    python analysis/check_wire_canonical.py --obfuscate --profiles baseline,zstd,seeded,modswitch

--obfuscate also checks §7.2: with schedule obfuscation on, the CTXT frame
sequence a network observer sees (direction, kind byte, size) and the number
of SCI ReLU calls must not depend on the schedule.
"""

from __future__ import annotations

import argparse
import socket
import threading
import time

import numpy as np

import pi_client
import pi_proto
import pi_server
from canonical_net import CANONICAL_SPECS
from mask_session import MaskDomain
from params_pi import F as FBITS
from pi_proto import WIRE_PROFILES, make_backend
from sci_relu import quantize_to_ring, ring_to_float

A0, A1, A2 = 0.0625, 0.5, 0.5


class _Rendezvous:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.slots: dict[int, dict] = {}

    def exchange(self, port: int, party: int, ring: np.ndarray, k: int):
        with self.cv:
            entry = self.slots.setdefault(port, {})
            entry[party] = ring
            self.cv.notify_all()
            self.cv.wait_for(lambda: len(entry) == 2 or "out" in entry, 120)
            if "out" not in entry:
                mod = 1 << k
                half = 1 << (k - 1)
                total = [(int(a) + int(b)) % mod
                         for a, b in zip(entry[1], entry[2])]
                y = [v if v < half else 0 for v in total]        # ReLU on Z_2^k
                rng = np.random.default_rng()
                c2 = [int(v) for v in rng.integers(0, mod, len(y), dtype=np.uint64)]
                c1 = [(yy - cc) % mod for yy, cc in zip(y, c2)]
                entry["out"] = {1: np.asarray(c1, dtype=np.uint64),
                                2: np.asarray(c2, dtype=np.uint64)}
                entry["taken"] = 0
            out = entry["out"][party]
            entry["taken"] += 1
            if entry["taken"] == 2:
                del self.slots[port]
            return out


_RDV = _Rendezvous()


# Observer tap: every CTXT frame as (party, kind byte, payload length).
_TAP: list[tuple[int, bytes, int]] = []
_orig_send = pi_proto.Wire.send


def _tapped_send(self, tag, payload=b""):
    if tag == b"CTXT":
        _TAP.append((self.party, bytes(payload[:1]), len(payload)))
    return _orig_send(self, tag, payload)


pi_proto.Wire.send = _tapped_send


class FakeSci:
    """Insecure, in-process stand-in with the SciReLUSession interface."""

    calls = {1: 0, 2: 0}

    def __init__(self, party: int, port: int, address: str = "127.0.0.1"):
        self.party, self.port = party, port

    def relu(self, shares, f, k):
        FakeSci.calls[self.party] = FakeSci.calls.get(self.party, 0) + 1
        ring = quantize_to_ring(shares, f, k)
        return ring_to_float(_RDV.exchange(self.port, self.party, ring, k), f, k)

    def bytes_sent(self) -> int:
        return 0


class _Sum:
    def __init__(self, arr):
        self.arr = arr

    def sum(self, dim):
        return _Np(self.arr.sum(axis=dim))


class _Np:
    def __init__(self, arr):
        self.arr = arr

    def numpy(self):
        return self.arr


class FakeCanonical:
    """CanonicalCNN interface with numpy weights (MNIST geometry)."""

    architecture = "canonical_cnn"

    def __init__(self, schedule, seed: int = 0, dataset: str = "mnist"):
        self.spec = CANONICAL_SPECS[dataset]
        self.dataset = dataset
        self.schedule = tuple(schedule)
        s = self.spec
        rng = np.random.default_rng(seed)
        kh, kw = s.im2col_kernel
        self.conv_weight_np = rng.normal(0, 1.0 / np.sqrt(kh * kw),
                                         (s.conv_channels, 1, kh, kw))
        self.conv_bias_np = rng.normal(0, 0.05, s.conv_channels)
        self.fc1_w = rng.normal(0, 1.0 / np.sqrt(s.conv_flat), (s.hidden, s.conv_flat))
        self.fc1_bias_np = rng.normal(0, 0.05, s.hidden)
        self.fc2_w = rng.normal(0, 1.0 / np.sqrt(s.hidden), (s.classes, s.hidden))
        self.fc2_bias_np = rng.normal(0, 0.05, s.classes)
        self.fc1_weight_t_np, self.fc2_weight_t_np = self.fc1_w.T, self.fc2_w.T
        self.fc1_weight, self.fc2_weight = _Sum(self.fc1_w), _Sum(self.fc2_w)
        self.input_dim = s.input_dim
        self.input_slots = s.encoded_slots
        self.windows = s.windows
        self.stride = s.stride
        self.im2col_kernel = s.im2col_kernel
        self.dims = [s.input_dim, s.conv_flat, s.hidden, s.classes]

    hybrid_layers = property(lambda self: sum(self.schedule))

    def output_layer_id(self) -> int:
        return len(self.schedule)

    def client_spec(self) -> dict:
        s = self.spec
        return {
            "architecture": self.architecture, "dataset": self.dataset,
            "dims": self.dims, "input_dim": self.input_dim,
            "input_slots": self.input_slots, "windows": self.windows,
            "im2col_kernel": list(self.im2col_kernel), "stride": list(self.stride),
            "encoding_stride": self.stride[0],
            "nonlinear_dims": [s.conv_flat, s.hidden],
            "preprocess": {"mode": "identity", "raw_shape": list(s.raw_shape),
                           "model_shape": list(s.model_shape)},
            "schedule": list(self.schedule), "hybrid_layers": self.hybrid_layers,
            "encoding": "im2col",
        }

    def plain_forward(self, raw):
        s = self.spec
        img = np.asarray(raw, dtype=np.float64).reshape(s.model_shape)[0]
        kh, kw = s.im2col_kernel
        sh, sw = s.stride
        rows = (img.shape[0] - kh) // sh + 1
        cols = (img.shape[1] - kw) // sw + 1
        z1 = np.zeros((s.conv_channels, rows, cols))
        for c in range(s.conv_channels):
            for r in range(rows):
                for q in range(cols):
                    z1[c, r, q] = np.sum(img[r*sh:r*sh+kh, q*sw:q*sw+kw]
                                         * self.conv_weight_np[c, 0]) + self.conv_bias_np[c]
        act = lambda z, bit: np.maximum(z, 0) if bit else A0 + A1 * z + A2 * z * z
        a1 = act(z1.ravel(), self.schedule[0])
        z2 = self.fc1_w @ a1 + self.fc1_bias_np
        a2 = act(z2, self.schedule[1])
        return self.fc2_w @ a2 + self.fc2_bias_np, [z1.ravel(), z2]

    def encrypted_forward(self, be, ctx, ct, wire, timings, mask_session=None,
                          query_index=0, sci_session=None):
        return pi_server.canonical_encrypted_forward(
            self, be, ctx, ct, wire, timings, mask_session, query_index,
            sci_session)


def run_one(schedule, profile: str, seed: int = 0,
            obfuscate: bool = False) -> dict:
    model = FakeCanonical(schedule, seed)
    rng = np.random.default_rng(seed + 1)
    x = rng.uniform(0.0, 1.0, model.input_dim)
    ref, zs = model.plain_forward(x)
    domains = {
        0: MaskDomain.from_observed(zs[0].min(), zs[0].max()),
        1: MaskDomain.from_observed(zs[1].min(), zs[1].max()),
        2: MaskDomain.from_observed(ref.min(), ref.max()),
    }
    be_srv = make_backend("seal")
    srv = socket.create_server(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    holder: dict = {}
    _TAP.clear()
    FakeSci.calls = {1: 0, 2: 0}

    def server_side():
        try:
            conn, _ = srv.accept()
            with conn:
                holder["stats"] = pi_server.serve_connection(
                    conn, be_srv, model, timeout=300, mask_domains=domains,
                    obfuscate_schedule=obfuscate,
                    wire_profile=WIRE_PROFILES[profile])
        except BaseException as exc:
            holder["error"] = exc

    t = threading.Thread(target=server_side, daemon=True)
    t.start()
    t0 = time.perf_counter()
    try:
        res = pi_client.run_client("127.0.0.1", port, x, "seal", timeout=300)
    finally:
        t.join(60)
        srv.close()
    if "error" in holder:
        raise holder["error"]
    got = np.asarray(res["logits"])
    err = float(np.max(np.abs(got - ref)))
    return {"schedule": "".join(map(str, schedule)), "profile": profile,
            "max_abs_err": err, "ref_max_abs": float(np.max(np.abs(ref))),
            "argmax_ok": int(np.argmax(got)) == int(np.argmax(ref)),
            "up": res["bytes_sent"], "down": res["bytes_received"],
            "wall_s": round(time.perf_counter() - t0, 1),
            "fitted": holder["stats"].get("mask_domains_fitted"),
            "trace": list(_TAP), "sci_calls": dict(FakeSci.calls),
            "level_aligned": holder["stats"].get("obfuscation_level_aligned")}


def compare_traces(rows: list[dict], rel_tol: float) -> tuple[bool, str]:
    """Same frames in the same order; sizes equal up to codec jitter."""
    ref = rows[0]
    worst = 0.0
    for r in rows[1:]:
        if [(p, k) for p, k, _ in r["trace"]] != [(p, k) for p, k, _ in ref["trace"]]:
            return False, f"frame pattern differs: s{ref['schedule']} vs s{r['schedule']}"
        if r["sci_calls"] != ref["sci_calls"]:
            return False, (f"SCI calls differ: s{ref['schedule']} {ref['sci_calls']} "
                           f"vs s{r['schedule']} {r['sci_calls']}")
        for (_, _, a), (_, _, b) in zip(ref["trace"], r["trace"]):
            worst = max(worst, abs(a - b) / max(a, b))
    return worst <= rel_tol, f"max frame-size difference {worst:.3%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--profiles", default="baseline,compact")
    ap.add_argument("--schedules", default="00,01,10,11")
    ap.add_argument("--tol", type=float, default=5e-2)
    ap.add_argument("--obfuscate", action="store_true",
                    help="§7.2: also require schedule-independent wire traces")
    ap.add_argument("--size-tol", type=float, default=0.01,
                    help="relative frame-size tolerance under --obfuscate "
                         "(zstd output varies by a few bytes per ciphertext)")
    args = ap.parse_args()
    pi_server.SCI_AVAILABLE = True
    pi_server.SciReLUSession = FakeSci
    pi_client.SCI_AVAILABLE = True
    pi_client.SciReLUSession = FakeSci
    ok = True
    for prof in args.profiles.split(","):
        rows = []
        for sch in args.schedules.split(","):
            r = run_one(tuple(int(b) for b in sch), prof, obfuscate=args.obfuscate)
            rows.append(r)
            passed = r["max_abs_err"] <= args.tol and r["argmax_ok"]
            ok &= passed
            print(f"{prof:>10} s{r['schedule']}  err={r['max_abs_err']:.3e} "
                  f"(|ref|<= {r['ref_max_abs']:.2f})  up={r['up']/1024:8.1f} KB "
                  f"down={r['down']/1024:8.1f} KB  {r['wall_s']:5.1f}s  "
                  f"{'PASS' if passed else 'FAIL'}")
        if args.obfuscate:
            same, why = compare_traces(rows, 0.0 if WIRE_PROFILES[prof].compr == "none"
                                       else args.size_tol)
            aligned = all(r["level_aligned"] for r in rows)
            ok &= same and aligned
            print(f"{prof:>10} §7.2 observer view: {why}; level-aligned={aligned}; "
                  f"SCI calls={rows[0]['sci_calls']}  {'PASS' if same and aligned else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
