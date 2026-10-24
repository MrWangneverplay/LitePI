"""LitePI Raspberry-Pi client: encode/encrypt M1, then one decrypt+re-encrypt per ReLU.

The client is the only holder of the CKKS secret key.  For every hybrid layer it

    1. receives Enc(z + r) and decrypts it -> s_c = z + r,
    2. runs MPC party1 on s_c together with the server's -r,
    3. re-encrypts its fresh output share c_c under the same public key.

Neither party ever holds both shares of z, and the client performs no linear
algebra -- only im2col input encoding, elementwise mask handling, and MPC.

Run against a desktop server:
    python pi_client.py --host 192.168.1.20 --port 9101

Run the two-party loopback on one machine (protocol self-test):
    python pi_client.py --loopback --backend stub --dims 32,16,4
    python pi_client.py --loopback --backend tenseal --dims 32,16,4
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from pathlib import Path

import numpy as np

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
from secure_argmax import combine_index_share, run_party_argmax


def load_input(path: Path | None, dim: int | None, seed: int = 0) -> np.ndarray:
    if path is None:
        if dim is None:
            raise ValueError("a dimension is required for a synthetic input")
        rng = np.random.default_rng(seed)
        return rng.normal(0.0, 1.0, dim)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        data = np.load(path)
        if "x" in data:
            key = "x"
        elif len(data.files) == 1:
            key = data.files[0]
        else:
            raise ValueError("multi-array .npz input must contain a key named 'x'")
        arr = data[key]
    elif suffix == ".json":
        arr = np.asarray(json.loads(path.read_text(encoding="utf-8")), dtype=np.float64)
    else:
        text = path.read_text(encoding="utf-8").replace(",", " ").split()
        arr = np.asarray([float(v) for v in text], dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("input vector must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("input vector contains NaN or infinity")
    if dim is not None and arr.size != dim:
        raise ValueError(f"input has {arr.size} values, model expects {dim}")
    return arr


def _prepare_model_input(x: np.ndarray, spec: dict) -> np.ndarray:
    """Apply the public preprocessing steps from the model specification."""
    mode = spec.get("encoding", "vector")
    if mode != "im2col":
        return x
    pre = spec["preprocess"]
    shape = tuple(int(v) for v in pre["raw_shape"])
    arr = np.asarray(x, dtype=np.float64).reshape(shape)
    name = pre["mode"]
    if name == "cifar10_gray_zscore":
        weights = np.asarray(pre["gray_weights"], dtype=np.float64)
        arr = np.tensordot(weights, arr, axes=(0, 0))
        arr = (arr - float(pre["mean"])) / float(pre["std"])
        return arr.reshape(tuple(int(v) for v in pre["model_shape"]))
    if name == "per_channel_zscore":
        mean = np.asarray(pre["mean"], dtype=np.float64)
        std = np.asarray(pre["std"], dtype=np.float64)
        arr = (arr - mean) / std
        return arr.reshape(tuple(int(v) for v in pre["model_shape"]))
    if name == "identity":
        return arr.reshape(tuple(int(v) for v in pre["model_shape"]))
    raise ProtocolError(f"unknown client preprocessing mode {name!r}")


def _encode_model_input(be, ctx, x: np.ndarray, spec: dict):
    if spec.get("encoding") != "im2col":
        return be.encrypt(ctx, x), None
    arr = _prepare_model_input(x, spec)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    kernel = tuple(int(v) for v in spec["im2col_kernel"])
    # TenSEAL 0.3.x swaps a non-square kernel inside the Python wrapper.
    if be.name == "tenseal" and spec.get("dataset") == "uci_har":
        kernel = (kernel[1], kernel[0])
    ct, windows = be.im2col_encrypt(
        ctx, arr, kernel, int(spec["encoding_stride"]),
    )
    if windows != int(spec["windows"]):
        raise ProtocolError(
            f"im2col produced {windows} windows, protocol expects {spec['windows']}"
        )
    return ct, windows


def _query_once(be, ctx, wire: Wire, x: np.ndarray, spec: dict,
                timings: dict, output_mode: str = "logits"):
    """Run one already-authenticated private query over the open session."""
    t0 = time.perf_counter()
    ct, _ = _encode_model_input(be, ctx, x, spec)
    timings["encrypt_input_s"] += time.perf_counter() - t0
    wire.send(b"CTXT", be.serialize(ct))

    mpc_ctx = wire.mpc_ctx()
    while True:
        _, payload = wire.recv(b"CTXT")
        if not payload:
            raise ProtocolError("empty ciphertext message")
        kind, blob = payload[0], payload[1:]
        ct = be.deserialize(ctx, blob)

        t0 = time.perf_counter()
        share = be.decrypt(ct)
        timings["decrypt_s"] += time.perf_counter() - t0

        if kind == 1:
            return "logits", np.asarray(share, dtype=np.float64)
        if kind == 2:
            t0 = time.perf_counter()
            client_share = run_party_argmax(
                mpc_ctx, share.tolist(), K_RING, F, K_COMPARE,
            )
            timings["mpc_s"] += time.perf_counter() - t0
            timings["mpc_elements"] += int(share.size)
            timings["base_ots"] += (
                int(share.size) - 1
            ) * max(K_COMPARE - 1, 1)
            idx_msg = wire.recv_json(b"IDXS")
            index = combine_index_share(
                client_share, int(idx_msg["share"]), K_RING,
            )
            if not 0 <= index < int(share.size):
                raise ProtocolError(f"secure argmax returned invalid index {index}")
            return "index", index
        if kind == 3:
            t0 = time.perf_counter()
            ct_out = be.encrypt(ctx, np.zeros(share.size, dtype=np.float64))
            timings["reencrypt_s"] += time.perf_counter() - t0
            wire.send(b"CTXT", b"\x03" + be.serialize(ct_out))
            continue
        if kind != 0:
            raise ProtocolError(f"unknown ciphertext kind {kind}")

        t0 = time.perf_counter()
        out_share = np.asarray(
            run_party_batch(mpc_ctx, share.tolist(), K_RING, F, K_COMPARE),
            dtype=np.float64,
        )
        timings["mpc_s"] += time.perf_counter() - t0
        timings["mpc_elements"] += int(share.size)
        timings["base_ots"] += int(share.size) * ots_per_relu()

        t0 = time.perf_counter()
        ct_out = be.encrypt(ctx, out_share)
        timings["reencrypt_s"] += time.perf_counter() - t0
        wire.send(b"CTXT", b"\x00" + be.serialize(ct_out))


def run_client(host: str, port: int, x: np.ndarray, backend_name: str,
               timeout: float = 600.0, output_mode: str = "logits") -> dict:
    """Connect to a server and execute one private inference query."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("input must be a non-empty finite vector")
    be = make_backend(backend_name)
    started = time.perf_counter()
    timings = {
        "keygen_s": 0.0,
        "encrypt_input_s": 0.0,
        "encode_input_s": 0.0,
        "decrypt_s": 0.0,
        "mpc_s": 0.0,
        "reencrypt_s": 0.0,
        "context_upload_s": 0.0,
        "mpc_elements": 0,
        "base_ots": 0,
    }

    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        wire = Wire(conn, party=1)

        spec = wire.recv_json(b"SPEC")
        dims = [int(v) for v in spec["dims"]]
        wanted = int(spec.get("input_dim", dims[0]))
        if wanted != x.size:
            raise ProtocolError(f"input has {x.size} values, server wants {wanted}")
        if spec.get("profile", {}).get("backend") not in (None, backend_name):
            raise ProtocolError("server backend does not match --backend")

        wire.send_json(b"INIT", {
            "input_dim": int(x.size),
            "backend": backend_name,
            "output_mode": output_mode,
        })
        wire.recv(b"REDY")

        # One-time key generation.  This is the client's largest fixed cost.
        t0 = time.perf_counter()
        ctx = be.new_context()
        timings["keygen_s"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        wire.send(b"CTXT", be.public_context_bytes(ctx))
        wire.recv(b"REDY")
        timings["context_upload_s"] = time.perf_counter() - t0

        output_kind, output_value = _query_once(
            be, ctx, wire, x, spec, timings, output_mode,
        )

        wire.send(b"BYE.")
        server_stats = wire.recv_json(b"STAT")

    if output_kind != output_mode:
        raise ProtocolError(
            f"server returned {output_kind!r}, client requested {output_mode!r}"
        )
    result = {
        "role": "client/party1",
        **profile_info(backend_name),
        "architecture": spec.get("architecture", "hybrid_mlp"),
        "dataset": spec.get("dataset"),
        "dims": dims,
        "schedule": spec.get("schedule"),
        "hybrid_layers": int(spec.get("hybrid_layers", len(dims) - 2)),
        "prediction": (
            int(np.argmax(output_value))
            if output_kind == "logits" else int(output_value)
        ),
        "output_mode": output_kind,
        "wall_s": round(time.perf_counter() - started, 4),
        "timings": {
            k: (int(v) if k in ("mpc_elements", "base_ots") else round(float(v), 6))
            for k, v in timings.items()
        },
        "bytes_sent": wire.cc.sent,
        "bytes_received": wire.cc.recvd,
        "server": server_stats,
    }
    result["mask_domain_aligned"] = bool(
        spec.get("profile", {}).get("mask_domain_aligned", False)
    )
    if output_kind == "logits":
        result["logits"] = [round(float(v), 6) for v in output_value]
    else:
        result["output_leakage"] = "class_index_only"
    return result


def run_reuse_session(host: str, port: int, x: np.ndarray, backend_name: str,
                      repeats: int = 3, timeout: float = 600.0,
                      output_mode: str = "logits") -> dict:
    """Run repeated queries over one connection with one CKKS context."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("input must be a non-empty finite vector")
    be = make_backend(backend_name)
    started = time.perf_counter()
    timings = {
        "keygen_s": 0.0,
        "context_upload_s": 0.0,
        "encrypt_input_s": 0.0,
        "encode_input_s": 0.0,
        "decrypt_s": 0.0,
        "mpc_s": 0.0,
        "reencrypt_s": 0.0,
        "mpc_elements": 0,
        "base_ots": 0,
    }
    queries = []

    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        wire = Wire(conn, party=1)
        spec = wire.recv_json(b"SPEC")
        dims = [int(v) for v in spec["dims"]]
        wanted = int(spec.get("input_dim", dims[0]))
        if wanted != x.size:
            raise ProtocolError(f"input has {x.size} values, server wants {wanted}")
        if spec.get("profile", {}).get("backend") not in (None, backend_name):
            raise ProtocolError("server backend does not match --backend")

        wire.send_json(b"INIT", {
            "input_dim": int(x.size),
            "backend": backend_name,
            "persistent": True,
            "reuse_context": False,
            "output_mode": output_mode,
        })
        wire.recv(b"REDY")

        t0 = time.perf_counter()
        ctx = be.new_context()
        timings["keygen_s"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        wire.send(b"CTXT", be.public_context_bytes(ctx))
        wire.recv(b"REDY")
        timings["context_upload_s"] = time.perf_counter() - t0

        for i in range(repeats):
            if i:
                wire.send_json(b"INIT", {
                    "input_dim": int(x.size),
                    "backend": backend_name,
                    "persistent": True,
                    "reuse_context": True,
                    "output_mode": output_mode,
                })
                wire.recv(b"REDY")

            query_started = time.perf_counter()
            sent_before = wire.cc.sent
            recv_before = wire.cc.recvd
            t_before = dict(timings)
            output_kind, output_value = _query_once(
                be, ctx, wire, x, spec, timings, output_mode,
            )
            if output_kind != output_mode:
                raise ProtocolError(
                    f"server returned {output_kind!r}, "
                    f"client requested {output_mode!r}"
                )
            wire.send(b"BYE.")
            server_stats = wire.recv_json(b"STAT")
            queries.append({
                "query_index": i,
                "wall_s": round(time.perf_counter() - query_started, 4),
                "timings": {
                    k: round(timings[k] - t_before[k], 6)
                    for k in ("encrypt_input_s", "decrypt_s", "mpc_s", "reencrypt_s")
                },
                "mpc_elements": int(timings["mpc_elements"] - t_before["mpc_elements"]),
                "base_ots": int(timings["base_ots"] - t_before["base_ots"]),
                "bytes_sent": wire.cc.sent - sent_before,
                "bytes_received": wire.cc.recvd - recv_before,
                "server": server_stats,
            })

        wire.send(b"DONE")

    last = queries[-1]
    result = {
        "role": "client/party1",
        **profile_info(backend_name),
        "architecture": spec.get("architecture", "hybrid_mlp"),
        "dataset": spec.get("dataset"),
        "dims": dims,
        "schedule": spec.get("schedule"),
        "hybrid_layers": int(spec.get("hybrid_layers", len(dims) - 2)),
        "prediction": (
            int(np.argmax(output_value))
            if output_kind == "logits" else int(output_value)
        ),
        "reuse_context": True,
        "repeat": repeats,
        "output_mode": output_kind,
        "wall_s": round(time.perf_counter() - started, 4),
        "timings": {
            k: (int(v) if k in ("mpc_elements", "base_ots") else round(float(v), 6))
            for k, v in timings.items()
        },
        "queries": queries,
        "last_query": last,
        "bytes_sent": wire.cc.sent,
        "bytes_received": wire.cc.recvd,
        "server": last["server"],
    }
    result["mask_domain_aligned"] = bool(
        spec.get("profile", {}).get("mask_domain_aligned", False)
    )
    if output_kind == "logits":
        result["logits"] = [round(float(v), 6) for v in output_value]
    else:
        result["output_leakage"] = "class_index_only"
    return result


def loopback_once(backend_name: str, dims: list[int], seed: int = 7,
                  input_seed: int = 0, host: str = "127.0.0.1",
                  timeout: float = 600.0,
                  schedule: tuple[int, ...] | None = None,
                  dataset: str | None = None,
                  root: Path | None = None,
                  checkpoint: Path | None = None,
                  mask_domains=None,
                  output_mode: str = "logits",
                  obfuscate_schedule: bool = False) -> dict:
    """Run server and client over a real localhost socket in one process.

    Loopback is a test harness: it lets the driver compare the final logits with
    a plaintext reference.  A production client has neither the weights nor the
    label, which is the point.
    """
    from pi_server import HybridMLP, serve_connection  # test-only import

    if dataset:
        from canonical_net import load_canonical_model

        root = root or Path(__file__).resolve().parent.parent
        selected_schedule = schedule or (1, 1)
        if any(selected_schedule) and not mask_domains:
            raise ValueError(
                "canonical hybrid loopback requires M2 mask domains"
            )
        model = load_canonical_model(
            root, dataset, selected_schedule, seed, checkpoint,
        )
    else:
        model = HybridMLP(dims, seed, schedule)
    if dataset:
        rng = np.random.default_rng(input_seed)
        x = rng.uniform(0.0, 1.0, model.input_dim)
    else:
        x = load_input(None, dims[0], input_seed)
    be = make_backend(backend_name)
    srv = socket.create_server((host, 0))
    port = srv.getsockname()[1]
    holder: dict = {}

    def server_side() -> None:
        try:
            conn, _ = srv.accept()
            with conn:
                holder["stats"] = serve_connection(
                    conn, be, model, reference_input=x, timeout=timeout,
                    mask_domains=mask_domains,
                    obfuscate_schedule=obfuscate_schedule,
                )
        except BaseException as exc:  # surface it in the caller
            holder["error"] = exc

    thread = threading.Thread(target=server_side, daemon=True)
    thread.start()
    try:
        result = run_client(
            host, port, x, backend_name, timeout=timeout,
            output_mode=output_mode,
        )
    finally:
        # The client call above already waited for the server protocol.  This
        # join is only for thread teardown, so do not inherit a 10-minute
        # network timeout here.
        thread.join(timeout=min(timeout, 60.0))
        srv.close()
        thread.join(timeout=5)
    if thread.is_alive():
        raise TimeoutError("loopback server thread did not terminate")
    if "error" in holder:
        raise holder["error"]

    ref, _ = model.plain_forward(x)
    if output_mode == "logits":
        got = np.asarray(result["logits"], dtype=np.float64)
        err = float(np.max(np.abs(got - ref)))
        prediction_matches = int(np.argmax(got)) == int(np.argmax(ref))
    else:
        err = 0.0
        prediction_matches = int(result["prediction"]) == int(np.argmax(ref))
    # The MPC works at f=8 (1/256 per element).  Allow 5e-2 on the logits so
    # the final linear layer can propagate a few LSBs; this is still far below
    # any decision boundary in the toy model.
    tol = 5e-2
    result["loopback"] = {
        "reference_logits": [round(float(v), 6) for v in ref],
        "max_abs_err": err,
        "tolerance": tol,
        "prediction_matches": prediction_matches,
        "passed": err <= tol and prediction_matches,
    }
    return result


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


def _print_summary(result: dict, verbose: bool) -> None:
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    t = result["timings"]
    tag = "SIMULATION" if result.get("simulation") else "TENSEAL"
    print(f"[{tag}] dims={result['dims']}  hybrid_layers={result['hybrid_layers']}")
    print(f"  keygen            {t['keygen_s']:8.3f} s")
    if "context_upload_s" in t:
        print(f"  context upload    {t['context_upload_s']:8.3f} s")
    print(f"  encrypt input     {t['encrypt_input_s']:8.3f} s")
    print(f"  decrypt (all)     {t['decrypt_s']:8.3f} s")
    print(f"  MPC ReLU (all)    {t['mpc_s']:8.3f} s   "
          f"({t['mpc_elements']} elems, ~{int(t['base_ots'])} base OTs)")
    print(f"  re-encrypt        {t['reencrypt_s']:8.3f} s")
    print(f"  wall              {result['wall_s']:8.3f} s")
    print(f"  traffic           up {result['bytes_sent'] / 1e6:.3f} MB  "
          f"down {result['bytes_received'] / 1e6:.3f} MB")
    if result.get("reuse_context"):
        print("  queries:")
        for q in result["queries"]:
            print(
                f"    #{q['query_index'] + 1}: wall {q['wall_s']:.3f} s  "
                f"MPC {q['timings']['mpc_s']:.3f} s  "
                f"up {q['bytes_sent'] / 1e6:.3f} MB"
            )
    if "loopback" in result:
        lb = result["loopback"]
        print(f"  logits err        {lb['max_abs_err']:.3e} "
              f"(tol {lb['tolerance']:.0e})  "
              f"{'PASS' if lb['passed'] else 'FAIL'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI Raspberry-Pi client (party1)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9101)
    ap.add_argument("--backend", choices=["tenseal", "stub"], default="tenseal")
    ap.add_argument("--input", type=Path, default=None,
                    help="optional .npy/.npz/.json/text vector")
    ap.add_argument("--dim", type=int, default=64,
                    help="synthetic input size when --input is absent")
    ap.add_argument("--input-seed", type=int, default=0,
                    help="seed for the synthetic input when --input is absent")
    ap.add_argument("--loopback", action="store_true",
                    help="start a local server thread for a protocol self-test")
    ap.add_argument("--dataset", choices=["mnist", "cifar10", "uci_har"],
                    default=None, help="loopback only: canonical model")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="loopback only: canonical checkpoint override")
    ap.add_argument("--mask-domains", type=Path, default=None,
                    help="loopback only: M2 per-layer domain JSON")
    ap.add_argument("--obfuscate-schedule", action="store_true",
                    help="loopback only: protect s with dummy ciphertext pairs")
    ap.add_argument("--schedule", type=_as_schedule, default=None,
                    help="loopback only: per-hidden-layer 0=P2, 1=ReLU")
    ap.add_argument("--dims", type=_as_dims, default=[64, 32, 10],
                    help="loopback only: layer sizes")
    ap.add_argument("--seed", type=int, default=7, help="loopback model seed")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--repeat", type=int, default=1,
                    help="number of queries in one persistent session")
    ap.add_argument("--reuse-context", action="store_true",
                    help="reuse one CKKS context across --repeat queries")
    ap.add_argument("--output-mode", choices=["logits", "index"], default="logits",
                    help="index mode requires a secure MPC argmax backend")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.repeat < 1:
        ap.error("--repeat must be >= 1")
    if args.loopback and (args.reuse_context or args.repeat != 1):
        ap.error("--reuse-context/--repeat are for a remote server, not --loopback")
    if args.schedule is not None and not args.loopback:
        ap.error("--schedule is set on the remote server, not the client")
    if args.dataset and not args.loopback:
        ap.error("--dataset is inferred from the remote server; use it only with --loopback")
    if args.obfuscate_schedule and not args.loopback:
        ap.error("--obfuscate-schedule is configured on the remote server")
    if args.dataset and args.schedule is None:
        args.schedule = (1, 1)

    if args.loopback:
        domains = None
        if args.mask_domains:
            from mask_session import load_mask_domains
            output_id = 2 if args.dataset else len(args.dims) - 2
            expected_dims = None
            if args.dataset:
                from canonical_net import CANONICAL_SPECS
                canonical = CANONICAL_SPECS[args.dataset]
                expected_dims = [canonical.conv_flat, canonical.hidden]
            domains = load_mask_domains(
                args.mask_domains,
                output_layer_id=output_id,
                expected_dataset=args.dataset,
                expected_schedule=args.schedule,
                expected_seed=args.seed,
                expected_dims=expected_dims,
            )
        result = loopback_once(
            args.backend, args.dims, args.seed, args.input_seed,
            schedule=args.schedule,
            dataset=args.dataset,
            checkpoint=args.checkpoint,
            mask_domains=domains,
            output_mode=args.output_mode,
            obfuscate_schedule=args.obfuscate_schedule,
        )
    else:
        # A real client learns the input dimension from the server's SPEC
        # message, so a file input is passed through without a fixed size.
        x = load_input(args.input, None if args.input else args.dim, args.input_seed)
        if args.reuse_context or args.repeat > 1:
            result = run_reuse_session(
                args.host, args.port, x, args.backend,
                repeats=args.repeat, timeout=args.timeout,
                output_mode=args.output_mode,
            )
        else:
            result = run_client(
                args.host, args.port, x, args.backend, args.timeout,
                output_mode=args.output_mode,
            )

    _print_summary(result, args.verbose)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"JSON -> {args.out}")
    if "loopback" in result and not result["loopback"]["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

