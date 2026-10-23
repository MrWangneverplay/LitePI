"""LitePI Raspberry-Pi client: encrypt once, then one decrypt+re-encrypt per ReLU.

The client is the only holder of the CKKS secret key.  For every hybrid layer it

    1. receives Enc(z + r) and decrypts it -> s_c = z + r,
    2. runs MPC party1 on s_c together with the server's -r,
    3. re-encrypts its fresh output share c_c under the same public key.

Neither party ever holds both shares of z, and the client performs no linear
algebra -- only elementwise mask handling plus the MPC ReLU.

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


def run_client(host: str, port: int, x: np.ndarray, backend_name: str,
               timeout: float = 600.0) -> dict:
    """Connect to a server and execute one private inference query."""
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size == 0 or not np.all(np.isfinite(x)):
        raise ValueError("input must be a non-empty finite vector")
    be = make_backend(backend_name)
    started = time.perf_counter()
    timings = {
        "keygen_s": 0.0,
        "encrypt_input_s": 0.0,
        "decrypt_s": 0.0,
        "mpc_s": 0.0,
        "reencrypt_s": 0.0,
        "mpc_elements": 0,
        "base_ots": 0,
    }

    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        wire = Wire(conn, party=1)

        spec = wire.recv_json(b"SPEC")
        dims = [int(v) for v in spec["dims"]]
        if dims[0] != x.size:
            raise ProtocolError(f"input has {x.size} values, server wants {dims[0]}")
        if spec.get("profile", {}).get("backend") not in (None, backend_name):
            raise ProtocolError("server backend does not match --backend")

        wire.send_json(b"INIT", {"input_dim": int(x.size), "backend": backend_name})
        wire.recv(b"REDY")

        # One-time key generation.  This is the client's largest fixed cost.
        t0 = time.perf_counter()
        ctx = be.new_context()
        timings["keygen_s"] = time.perf_counter() - t0

        wire.send(b"CTXT", be.public_context_bytes(ctx))
        wire.recv(b"REDY")

        t0 = time.perf_counter()
        ct = be.encrypt(ctx, x)
        timings["encrypt_input_s"] = time.perf_counter() - t0
        wire.send(b"CTXT", be.serialize(ct))

        logits = None
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

            if kind == 1:  # final logits
                logits = share
                break
            if kind != 0:
                raise ProtocolError(f"unknown ciphertext kind {kind}")

            # Client side of the same MPC the server is running right now.
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

        wire.send(b"BYE.")
        server_stats = wire.recv_json(b"STAT")

    if logits is None:
        raise ProtocolError("connection closed before the final ciphertext")
    result = {
        "role": "client/party1",
        **profile_info(backend_name),
        "dims": dims,
        "hybrid_layers": len(dims) - 2,
        "logits": [round(float(v), 6) for v in logits],
        "prediction": int(np.argmax(logits)),
        "wall_s": round(time.perf_counter() - started, 4),
        "timings": {
            k: (int(v) if k in ("mpc_elements", "base_ots") else round(float(v), 6))
            for k, v in timings.items()
        },
        "bytes_sent": wire.cc.sent,
        "bytes_received": wire.cc.recvd,
        "server": server_stats,
    }
    return result


def loopback_once(backend_name: str, dims: list[int], seed: int = 7,
                  input_seed: int = 0, host: str = "127.0.0.1",
                  timeout: float = 600.0) -> dict:
    """Run server and client over a real localhost socket in one process.

    Loopback is a test harness: it lets the driver compare the final logits with
    a plaintext reference.  A production client has neither the weights nor the
    label, which is the point.
    """
    from pi_server import HybridMLP, serve_connection  # test-only import

    mlp = HybridMLP(dims, seed)
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
                    conn, be, mlp, reference_input=x, timeout=timeout
                )
        except BaseException as exc:  # surface it in the caller
            holder["error"] = exc

    thread = threading.Thread(target=server_side, daemon=True)
    thread.start()
    try:
        result = run_client(host, port, x, backend_name, timeout=timeout)
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

    ref, _ = mlp.plain_forward(x)
    got = np.asarray(result["logits"], dtype=np.float64)
    err = float(np.max(np.abs(got - ref)))
    # The MPC works at f=8 (1/256 per element).  Allow 5e-2 on the logits so
    # the final linear layer can propagate a few LSBs; this is still far below
    # any decision boundary in the toy model.
    tol = 5e-2
    result["loopback"] = {
        "reference_logits": [round(float(v), 6) for v in ref],
        "max_abs_err": err,
        "tolerance": tol,
        "prediction_matches": int(np.argmax(got)) == int(np.argmax(ref)),
        "passed": err <= tol and int(np.argmax(got)) == int(np.argmax(ref)),
    }
    return result


def _as_dims(text: str) -> list[int]:
    dims = [int(x) for x in text.replace("x", ",").split(",") if x.strip()]
    if len(dims) < 2 or any(d <= 0 for d in dims):
        raise argparse.ArgumentTypeError("dims must be at least two positive ints")
    return dims


def _print_summary(result: dict, verbose: bool) -> None:
    if verbose:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    t = result["timings"]
    tag = "SIMULATION" if result.get("simulation") else "TENSEAL"
    print(f"[{tag}] dims={result['dims']}  hybrid_layers={result['hybrid_layers']}")
    print(f"  keygen            {t['keygen_s']:8.3f} s")
    print(f"  encrypt input     {t['encrypt_input_s']:8.3f} s")
    print(f"  decrypt (all)     {t['decrypt_s']:8.3f} s")
    print(f"  MPC ReLU (all)    {t['mpc_s']:8.3f} s   "
          f"({t['mpc_elements']} elems, ~{int(t['base_ots'])} base OTs)")
    print(f"  re-encrypt        {t['reencrypt_s']:8.3f} s")
    print(f"  wall              {result['wall_s']:8.3f} s")
    print(f"  traffic           up {result['bytes_sent'] / 1e6:.3f} MB  "
          f"down {result['bytes_received'] / 1e6:.3f} MB")
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
    ap.add_argument("--dims", type=_as_dims, default=[64, 32, 10],
                    help="loopback only: layer sizes")
    ap.add_argument("--seed", type=int, default=7, help="loopback model seed")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.loopback:
        result = loopback_once(args.backend, args.dims, args.seed, args.input_seed)
    else:
        # A real client learns the input dimension from the server's SPEC
        # message, so a file input is passed through without a fixed size.
        x = load_input(args.input, None if args.input else args.dim, args.input_seed)
        result = run_client(args.host, args.port, x, args.backend, args.timeout)

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

