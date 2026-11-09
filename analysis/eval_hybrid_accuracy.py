"""E1 tier 3 (Hybrid): per-sample ciphertext accuracy through the full protocol.

`eval_encrypted.py` only covers the HE-only trace (schedule 00), so the hybrid
tier of E1 could not be measured at all.  This driver runs the *complete*
protocol for every test sample -- im2col + encrypt on the client, HE linear
layers on the server, and the SCI (CrypTFlow2) two-party ReLU for every
schedule bit that is 1 -- and compares the decrypted logits with the plaintext
reference of the same checkpoint.

Session reuse (important for the per-sample cost)
-------------------------------------------------
Key generation (~0.4 s) and the public-context upload (~4 s) dominate a single
query, so running `loopback_once` once per sample would report an accuracy
number inflated by a fixed ~4 s per sample.  Instead this driver opens ONE
server session (one accept, one SPEC, one keygen, one context upload, one SCI
setup) and then runs every remaining test sample as another query on that
session: each subsequent query sends `INIT{persistent:true,
reuse_context:true}` and reuses the same CKKS context and the same SCI OT
session.  `pi_server.serve_connection` already implements this persistent mode
(it is what `pi_client.run_reuse_session` exercises), so no protocol change was
needed.

The fixed setup cost is still measured and reported, both in absolute terms
(`timing_s.keygen` / `timing_s.context_upload`) and amortised over the samples
of the session (`timing_s.amortized_setup_s_per_sample`).  Per-sample numbers
(`per_sample[*].wall_s`) are the honest online cost.

Preprocessing deliberately mirrors `eval_encrypted.py` (its lines 232-241):
uci_har is NOT re-normalised (`load_data` already z-scores it), mnist/cifar10
go through `model.prepare_array(raw).reshape(model_shape)` and lose the leading
batch dim.  The plaintext reference uses `plain_forward_prepared` on that same
array, so the reported error contains no polynomial/substitution difference.

Usage (analysis directory):

    python eval_hybrid_accuracy.py --dataset mnist --schedule 11 \
        --seed 73 --samples 32 --backend seal

Output: results/raw_results/E1/<dataset>/hybrid_s<tag>_acc<out_suffix>/seed<seed>.json
where <out_suffix> is `--out-tag` (default `_<backend>..._v2`).
Schema v2 records paired logits/predictions, original sample IDs, margins,
checkpoint/data/source/native-module hashes and full resume configuration.
Use the same --sample-seed across training seeds and schedules for paired E1.
Legacy records are kept untouched and cannot be silently resumed as v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import threading
import tempfile
import time
from datetime import datetime, timezone
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from eval_encrypted import load_test_set
from mask_session import load_mask_domains, widen_domains
from params_pi import (
    COEFF_MOD_BIT_SIZES,
    F,
    GLOBAL_SCALE_BITS,
    K_COMPARE,
    K_RING,
    POLY_MODULUS_DEGREE,
)
from pi_proto import WIRE_PROFILES, ProtocolError, Wire, make_backend
from pi_client import adopt_wire, ciphertext_loop, send_input, upload_context
from pi_server import serve_connection
from secure_relu_net import run_party_batch
from sci_relu import SCI_AVAILABLE, SciReLUSession, sci_bytes

FLUSH_EVERY = 25
SCHEMA_VERSION = 2
CODE_ROOT = Path(__file__).resolve().parent.parent
SETUP_KEYS = (
    "mpc_runtime", "keygen_s", "context_upload_s", "context_up_bytes",
    "sci_setup_up_bytes", "sci_setup_down_bytes", "sci_setup_s",
    "setup_pending_server_sci_sessions", "queries", "sessions",
)


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=True, timeout=10)
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "analysis", "sci", "seal", "scripts"],
            cwd=str(root), capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if status.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_evidence(label: int, plain_logits, cipher_logits) -> dict:
    """Accuracy and fidelity are different questions; retain paired evidence."""
    plain = [float(value) for value in plain_logits]
    cipher = [float(value) for value in cipher_logits]
    if len(plain) < 2 or len(plain) != len(cipher):
        raise ValueError("paired logits must have the same class count >= 2")
    if not all(math.isfinite(value) for value in plain + cipher):
        raise ValueError("non-finite plaintext/ciphertext logits; refuse accuracy row")
    if not 0 <= int(label) < len(plain):
        raise ValueError("label outside the paired class range")
    plain_pred = max(range(len(plain)), key=plain.__getitem__)
    cipher_pred = max(range(len(cipher)), key=cipher.__getitem__)
    plain_top = sorted(plain, reverse=True)
    cipher_top = sorted(cipher, reverse=True)
    errors = [abs(a - b) for a, b in zip(plain, cipher)]
    peak = max(errors)
    if not math.isfinite(peak):
        raise ValueError("logit difference exceeds finite float range")
    scale = max(1e-9, max(map(abs, plain)))
    if not all(math.isfinite(value) for value in (
            plain_top[0] - plain_top[1], cipher_top[0] - cipher_top[1], peak / scale)):
        raise ValueError("non-finite logit margin or relative error")
    return {
        "label": int(label), "pred": cipher_pred,  # backwards-compatible alias
        "plain_pred": plain_pred, "cipher_pred": cipher_pred,
        "plain_correct": plain_pred == int(label),
        "cipher_correct": cipher_pred == int(label),
        "prediction_agrees": plain_pred == cipher_pred,
        "plain_logits": plain, "cipher_logits": cipher,
        "plain_margin": plain_top[0] - plain_top[1],
        "cipher_margin": cipher_top[0] - cipher_top[1],
        "max_abs_err": peak, "mean_abs_err": sum(value / len(errors) for value in errors),
        "rmse": (peak * math.sqrt(sum((value / peak) ** 2 for value in errors) / len(errors))
                 if peak else 0.0),
        "rel_err": max(errors) / scale, "ref_max_abs": max(map(abs, plain)),
    }


def prediction_summary(rows: list[dict]) -> dict:
    done = len(rows)
    cipher_errors = sum(row.get("cipher_pred", row["pred"]) != row["label"] for row in rows)
    paired = [row for row in rows if row.get("plain_pred") is not None]
    mismatches = sum(row.get("cipher_pred", row["pred"]) != row["plain_pred"] for row in paired)
    plain_errors = sum(row["plain_pred"] != row["label"] for row in paired)
    return {
        "classification_errors": cipher_errors,
        "plain_classification_errors": plain_errors if paired else None,
        "accuracy": (done - cipher_errors) / done if done else None,
        "accuracy_pct": 100.0 * (done - cipher_errors) / done if done else None,
        "plain_accuracy": (len(paired) - plain_errors) / len(paired) if paired else None,
        "prediction_mismatches": mismatches if paired else None,
        "prediction_agreement": (len(paired) - mismatches) / len(paired) if paired else None,
        "agreement_samples": len(paired),
        "agreement_complete": bool(done) and len(paired) == done,
        "prediction_mismatches_definition": "cipher_pred != plain_pred on agreement_samples",
    }


def native_identity(backend) -> dict:
    """Hash only the selected HE runtime and the protocol's SCI extension.

    Unrelated imports (HAR also imports TenSEAL even under --backend seal)
    must not change the identity of a SEAL run across datasets.
    """
    candidates = {}
    module = getattr(backend, "seal", None) or getattr(backend, "ts", None)
    if module is not None:
        candidates[getattr(module, "__name__", "he_runtime")] = module
        for value in vars(module).values():
            filename = getattr(value, "__file__", None)
            if filename and any(str(filename).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
                candidates[getattr(value, "__name__", "he_extension")] = value
    sci = sys.modules.get("litepi_sci")
    if sci is not None:
        candidates["litepi_sci"] = sci
    hashes = {}
    for name, candidate in candidates.items():
        filename = getattr(candidate, "__file__", None)
        if filename and any(str(filename).endswith(suffix) for suffix in EXTENSION_SUFFIXES):
            hashes[name] = file_sha256(Path(filename))
    return hashes


def run_identity(args, model, sample_ids, test_size: int, domains, backend) -> dict:
    checkpoint = Path(model.checkpoint).resolve()
    source_paths = sorted((CODE_ROOT / "analysis").glob("*.py"))
    source_paths += sorted((CODE_ROOT / "model").glob("*.py"))
    source_paths += sorted(CODE_ROOT.glob("*.py"))
    source_paths += sorted((CODE_ROOT / "seal").glob("*.cpp"))
    source_paths += sorted((CODE_ROOT / "seal").glob("*.h"))
    sci_source = CODE_ROOT / "sci" / "litepi_sci.cpp"
    if sci_source.exists():
        source_paths.append(sci_source)
    sources = {str(path.relative_to(CODE_ROOT)): file_sha256(path) for path in source_paths}
    # Hash the actually imported SCI extension, not a guessed ABI-specific file.
    extension = getattr(sys.modules.get("litepi_sci"), "__file__", None)
    native_modules = native_identity(backend)
    if args.backend != "stub" and not any(name != "litepi_sci" for name in native_modules):
        raise ValueError("cannot identify the selected HE native extension; refuse untraceable run")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": args.dataset, "schedule": "".join(map(str, args.schedule)),
        "training_seed": args.seed, "sample_seed": args.sample_seed,
        "sampler": "numpy.default_rng.permutation", "numpy_version": np.__version__,
        "sample_order_sha256": hashlib.sha256(json.dumps(
            [int(value) for value in sample_ids], separators=(",", ":")).encode()).hexdigest(),
        "test_set_size": test_size,
        "checkpoint_sha256": file_sha256(checkpoint),
        "backend": args.backend, "wire": WIRE_PROFILES[args.wire].to_dict(),
        "mask_exact": bool(args.mask_exact),
        "mask_domains": {str(k): v.to_dict() for k, v in sorted(domains.items())} if domains else None,
        "params": {"poly_modulus_degree": POLY_MODULUS_DEGREE,
                   "coeff_mod_bit_sizes": list(COEFF_MOD_BIT_SIZES),
                   "global_scale_bits": GLOBAL_SCALE_BITS,
                   "f": F, "k_ring": K_RING, "k_compare": K_COMPARE},
        "source_sha256": sources,
        "sci_extension_sha256": file_sha256(Path(extension)) if extension else None,
        "native_module_sha256": native_modules,
        "python_runtime": sys.version,
    }


def validate_resume(previous: dict, identity: dict, sample_ids, n: int) -> None:
    if previous.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("legacy schema cannot be resumed; use a new v2 output tag")
    old = previous.get("run_identity") or {}
    changed = [key for key in sorted(set(old) | set(identity)) if old.get(key) != identity.get(key)]
    if changed:
        raise ValueError("resume configuration mismatch: " + ", ".join(changed))
    rows = previous.get("per_sample")
    if not isinstance(rows, list) or len(rows) > n:
        raise ValueError("invalid per_sample list or requested samples smaller than saved prefix")
    if previous.get("n_samples") != len(rows):
        raise ValueError("n_samples disagrees with saved rows")
    for index, row in enumerate(rows):
        if row.get("index") != index or row.get("sample_id") != int(sample_ids[index]):
            raise ValueError(f"resume sample order mismatch at index {index}")
        if not all(key in row for key in ("label", "plain_logits", "cipher_logits",
                                          "plain_pred", "cipher_pred", "pred",
                                          "prepared_input_sha256")):
            raise ValueError(f"missing paired evidence at index {index}")
        evidence = prediction_evidence(row["label"], row["plain_logits"], row["cipher_logits"])
        if any(row[key] != evidence[key] for key in ("plain_pred", "cipher_pred", "pred")):
            raise ValueError(f"inconsistent predictions at index {index}")


def _prepare(model, dataset: str, sample) -> np.ndarray:
    """Exact `eval_encrypted.py` preprocessing for one test sample."""
    if dataset == "uci_har":
        # load_data already z-scores the windows; prepare_array would
        # normalise a second time and the plain logits would disagree.
        prepared = np.asarray(sample, dtype=np.float64)
    else:
        raw = np.asarray(sample, dtype=np.float64).ravel()
        prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
    prepared = prepared.reshape(model.spec.model_shape)
    if prepared.ndim == 3 and prepared.shape[0] == 1:
        prepared = prepared[0]
    return prepared


def _reference(model, prepared: np.ndarray) -> np.ndarray:
    ref, _ = model.plain_forward_prepared(
        prepared.reshape(1, *model.spec.model_shape)
    )
    return np.ravel(ref)[: model.spec.classes]


def _empty_query_timings() -> dict:
    return {
        "encrypt_input_s": 0.0,
        "decrypt_s": 0.0,
        "mpc_s": 0.0,
        "reencrypt_s": 0.0,
    }


def _query(be, ctx, wire: Wire, prepared: np.ndarray, model,
           sci_session, timings: dict, spec: dict | None = None) -> np.ndarray:
    """One client query over an already-authenticated session.

    Uses the protocol client's own send/receive loop (pi_client.send_input /
    ciphertext_loop) so E1 and the deployed client cannot drift apart; only
    the input preprocessing lives here (see `_prepare`).
    """
    spec = spec or model.client_spec()
    send_input(be, ctx, wire, prepared, spec, timings,
               kernel=model.im2col_kernel, stride=model.stride,
               expected_windows=model.windows)
    kind, value = ciphertext_loop(be, ctx, wire, spec, timings, sci_session)
    if kind != "logits":
        raise ProtocolError(f"expected logits, got {kind}")
    return np.asarray(value, dtype=np.float64)


def session_eval(be, model, test, dataset: str, rows: list[dict], n: int,
                 timeout: float, domains, state: dict, flush_cb,
                 wire: str = "baseline", mask_exact: bool = False) -> None:
    """Run one persistent server session over samples [len(rows), n).

    `state` is updated in place (mpc_runtime / keygen / context upload) so the
    partial-flush callback can see the real runtime string and setup cost.
    """
    be_server = make_backend(be.name)
    # Resolved here: `wire` is rebound to the client's Wire below, and the
    # server thread must not read it after that.
    profile = WIRE_PROFILES[wire]
    srv = socket.create_server(("127.0.0.1", 0))
    port = int(srv.getsockname()[1])
    holder: dict = {}

    def server_side() -> None:
        try:
            conn, _ = srv.accept()
            with conn:
                holder["stats"] = serve_connection(
                    conn, be_server, model, reference_input=None,
                    timeout=timeout, mask_domains=domains,
                    wire_profile=profile,
                    mask_exact=mask_exact,
                )
        except BaseException as exc:  # surface it in the caller
            holder["error"] = exc

    thread = threading.Thread(target=server_side, daemon=True)
    thread.start()
    start = len(rows)
    query_t: list[float] = []
    conn = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    conn.settimeout(timeout)
    wire = Wire(conn, party=1)
    state["sessions"] = int(state.get("sessions", 0) or 0) + 1
    try:
        spec = wire.recv_json(b"SPEC")
        adopt_wire(be, spec)
        sci_session = None
        if spec.get("sci_port") is not None:
            if not SCI_AVAILABLE:
                raise SystemExit(
                    "server offered an SCI port but litepi_sci is not built "
                    "for this interpreter"
                )
            state["setup_pending_server_sci_sessions"] = int(
                state.get("setup_pending_server_sci_sessions", 0) or 0) + 1
            sci_t0 = time.perf_counter()
            try:
                sci_session = SciReLUSession(2, int(spec["sci_port"]), "127.0.0.1")
            finally:
                state["sci_setup_s"] = float(state.get("sci_setup_s", 0.0) or 0.0) + (
                    time.perf_counter() - sci_t0)
        state["sci_setup_up_bytes"] = (int(state.get("sci_setup_up_bytes", 0) or 0)
                                       + sci_bytes(sci_session))
        if not any(model.schedule):
            runtime = "none"
        else:
            runtime = "litepi_sci" if sci_session is not None else "python-ot"
        state["mpc_runtime"] = runtime
        if int(spec["input_dim"]) != int(model.input_dim):
            raise ProtocolError(
                f"server wants {spec['input_dim']} inputs, model has "
                f"{model.input_dim}"
            )
        if spec.get("profile", {}).get("backend") not in (None, be.name):
            raise ProtocolError("server backend does not match --backend")

        init = {
            "input_dim": int(model.input_dim),
            "backend": be.name,
            "persistent": True,
            "reuse_context": False,
            "output_mode": "logits",
        }
        wire.send_json(b"INIT", init)
        wire.recv(b"REDY")

        t0 = time.perf_counter()
        try:
            ctx = be.new_context()
        finally:
            # Record each stage immediately, even if the next stage fails.
            state["keygen_s"] = float(state.get("keygen_s", 0.0) or 0.0) + (
                time.perf_counter() - t0)
        t0 = time.perf_counter()
        bytes_before = wire.cc.sent
        try:
            wire.send(b"CTXT", upload_context(be, ctx))
            wire.recv(b"REDY")
        finally:
            state["context_upload_s"] = float(state.get("context_upload_s", 0.0) or 0.0) + (
                time.perf_counter() - t0)
            state["context_up_bytes"] = int(state.get("context_up_bytes", 0) or 0) + (
                int(wire.cc.sent - bytes_before))

        for index in range(start, n):
            if index > start:
                wire.send_json(b"INIT", init | {"reuse_context": True})
                wire.recv(b"REDY")
            sample, label = test[index]
            prepared = _prepare(model, dataset, sample)
            ref = _reference(model, prepared)

            q = _empty_query_timings()
            sent_before = wire.cc.sent
            recv_before = wire.cc.recvd
            sci_before = sci_bytes(sci_session)
            t_q = time.perf_counter()
            got = _query(be, ctx, wire, prepared, model, sci_session, q, spec)
            wire.send(b"BYE.")
            stats = wire.recv_json(b"STAT")
            wall_s = time.perf_counter() - t_q
            query_t.append(wall_s)
            up_wire = int(wire.cc.sent - sent_before)
            down_wire = int(wire.cc.recvd - recv_before)
            up_sci = sci_bytes(sci_session) - sci_before
            down_sci = int(stats.get("sci_bytes_sent", 0))
            if index == start:
                state["sci_setup_down_bytes"] = (
                    int(state.get("sci_setup_down_bytes", 0) or 0)
                    + int(stats.get("sci_setup_bytes_sent", 0)))
                if sci_session is not None and "sci_setup_bytes_sent" in stats:
                    state["setup_pending_server_sci_sessions"] = max(0, int(
                        state.get("setup_pending_server_sci_sessions", 0) or 0) - 1)

            if got.size < model.spec.classes:
                raise ProtocolError(
                    f"decrypted {got.size} logits, expected "
                    f"{model.spec.classes}"
                )
            logits = got[: model.spec.classes]
            evidence = prediction_evidence(label, ref, logits)
            err = evidence["max_abs_err"]
            row = {
                "index": int(index),
                "sample_id": int(state["sample_ids"][index]),
                "prepared_input_sha256": hashlib.sha256(
                    np.asarray(prepared, dtype="<f8").tobytes(order="C")).hexdigest(),
                **evidence,
                "wall_s": round(wall_s, 4),
                "forward_s": float(stats["timings"]["forward_s"]),
                "mpc_s": float(stats["timings"]["mpc_s"]),
                "encrypt_s": round(q["encrypt_input_s"], 6),
                "decrypt_s": round(q["decrypt_s"], 6),
                "reencrypt_s": round(q["reencrypt_s"], 6),
                "up_bytes": up_wire + up_sci,
                "down_bytes": down_wire + down_sci,
                "up_wire_bytes": up_wire,
                "down_wire_bytes": down_wire,
                "up_sci_bytes": up_sci,
                "down_sci_bytes": down_sci,
                "hybrid_roundtrips": int(stats.get("hybrid_roundtrips", 0)),
                "mpc_runtime": runtime,
            }
            rows.append(row)
            state["queries"] = int(state.get("queries", 0) or 0) + 1
            done = len(rows)
            acc = sum(1 for r in rows if r["pred"] == r["label"]) / done
            print(f"  {done}/{n} idx={index} pred={row['pred']} "
                  f"label={row['label']} acc={acc:.4f} err={err:.2e} "
                  f"wall={wall_s:.2f}s fwd={row['forward_s']:.2f}s "
                  f"mpc={row['mpc_s']:.3f}s", flush=True)
            if done % FLUSH_EVERY == 0 and done < n:
                flush_cb()

        # DONE closes the persistent session; the server's last STAT was
        # already consumed after the final BYE.
        wire.send(b"DONE")
    except ConnectionError:
        # The client only sees a closed socket; the cause is on the server side.
        thread.join(timeout=5)
        if "error" in holder:
            raise holder["error"]
        raise
    finally:
        conn.close()
        srv.close()
        thread.join(timeout=min(timeout, 60.0))
        thread.join(timeout=5)
    if thread.is_alive():
        raise TimeoutError("loopback server thread did not terminate")
    if "error" in holder:
        raise holder["error"]
    state["query_wall_s"] = [round(v, 4) for v in query_t]


def build_payload(args, rows: list[dict], n: int, model, test_size: int,
                  state: dict, partial: bool) -> dict:
    errs = [r["max_abs_err"] for r in rows]
    rels = [r.get("rel_err", 0.0) for r in rows]
    walls = [r["wall_s"] for r in rows]
    fwds = [r["forward_s"] for r in rows]
    mpcs = [r["mpc_s"] for r in rows]
    encs = [r["encrypt_s"] for r in rows]
    decs = [r["decrypt_s"] for r in rows]
    reencs = [r["reencrypt_s"] for r in rows]
    # Rows from before the SCI accounting (--resume) have ciphertext bytes only.
    counted = [r for r in rows if "up_sci_bytes" in r]
    ups = [r["up_bytes"] for r in counted]
    downs = [r["down_bytes"] for r in counted]
    done = len(rows)
    keygen = float(state.get("keygen_s", 0.0) or 0.0)
    ctx_up = float(state.get("context_upload_s", 0.0) or 0.0)
    sci_setup = float(state.get("sci_setup_s", 0.0) or 0.0)
    return {
        "experiment": "E1",
        "schema_version": SCHEMA_VERSION,
        "run_identity": state["run_identity"],
        "checkpoint_sha256": state["run_identity"]["checkpoint_sha256"],
        "sampling": {"sample_seed": args.sample_seed,
                     "sample_id_definition": "index in unshuffled load_test_set output",
                     "sample_order_sha256": state["run_identity"]["sample_order_sha256"]},
        "wire": WIRE_PROFILES[args.wire].to_dict(),
        "mask_exact": bool(args.mask_exact),
        "mask_domains": state["run_identity"]["mask_domains"],
        "setup": {key: state.get(key) for key in SETUP_KEYS},
        "measured": args.backend != "stub" and done > 0,
        "evidence_kind": "simulation" if args.backend == "stub" else "protocol_evaluation",
        "tier": "Hybrid ciphertext accuracy",
        "dataset": args.dataset,
        "schedule": "".join(str(b) for b in args.schedule),
        "schedule_bits": [int(b) for b in args.schedule],
        "seed": args.seed,
        "backend": args.backend,
        "partial": partial,
        "n_samples": done,
        "target_samples": n,
        "test_set_size": test_size,
        **prediction_summary(rows),
        "mpc_runtime": state.get("mpc_runtime"),
        "checkpoint": str(getattr(model, "checkpoint", "")),
        "max_abs_err": {
            "mean": float(np.mean(errs)) if errs else None,
            "max": float(np.max(errs)) if errs else None,
        },
        "rel_err": {
            "mean": float(np.mean(rels)) if rels else None,
            "max": float(np.max(rels)) if rels else None,
        },
        "timing_s": {
            "keygen": round(keygen, 4),
            "context_upload": round(ctx_up, 4),
            "sci_setup": round(sci_setup, 4),
            "amortized_setup_s_per_sample": (
                round((keygen + ctx_up + sci_setup) / done, 4) if done else None
            ),
            "setup_timing_scope": "client keygen/context-upload/SCI-setup stage walls; includes failed attempts",
            "wall_mean": float(np.mean(walls)) if walls else None,
            "wall_max": float(np.max(walls)) if walls else None,
            "forward_mean": float(np.mean(fwds)) if fwds else None,
            "forward_max": float(np.max(fwds)) if fwds else None,
            "mpc_mean": float(np.mean(mpcs)) if mpcs else None,
            "encrypt_mean": float(np.mean(encs)) if encs else None,
            "decrypt_mean": float(np.mean(decs)) if decs else None,
            "reencrypt_mean": float(np.mean(reencs)) if reencs else None,
            "online_total_s": round(float(np.sum(walls)), 3) if walls else 0.0,
        },
        "communication": {
            "S_up_bytes_mean": float(np.mean(ups)) if ups else None,
            "S_down_bytes_mean": float(np.mean(downs)) if downs else None,
            **{f"S_{d}_{layer}_bytes_mean":
               float(np.mean([r[f"{d}_{layer}_bytes"] for r in counted])) if counted else None
               for d in ("up", "down") for layer in ("wire", "sci")},
            "bytes_rows": len(ups),
            "context_up_bytes": state.get("context_up_bytes"),
            "sci_setup_up_bytes": state.get("sci_setup_up_bytes"),
            "sci_setup_down_bytes": state.get("sci_setup_down_bytes"),
            "setup_bytes_complete": not int(state.get("setup_pending_server_sci_sessions", 0) or 0),
            "setup_pending_server_sci_sessions": int(state.get("setup_pending_server_sci_sessions", 0) or 0),
            "note": "S_up/S_down 为单次查询的客户端上下行字节，含密文（wire）和 SCI，"
                    "不含一次性上下文上传和 SCI 基础 OT（sci_setup）",
        },
        "session": {
            "reuse_context": True,
            "queries": int(state.get("queries", 0) or 0),
            "sessions": int(state.get("sessions", 0) or 0),
            "mpc_runtime": state.get("mpc_runtime"),
            "note": "同一次 server 会话内跑多个样本；keygen 与上下文上传只付一次",
        },
        "params": {
            "poly_modulus_degree": POLY_MODULUS_DEGREE,
            "coeff_mod_bit_sizes": list(COEFF_MOD_BIT_SIZES),
            "global_scale_bits": GLOBAL_SCALE_BITS,
            "f": F, "k_ring": K_RING, "k_compare": K_COMPARE,
        },
        "per_sample": rows,
    }


def write_record(path: Path, payload: dict, root: Path) -> None:
    record = {"exp_id": "E1", "git_sha": git_sha(root),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(
                  timespec="seconds")}
    record.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement keeps an interrupted flush from truncating the saved prefix.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp",
                                         delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E1 Hybrid 密文逐样本精度（完整协议 + SCI MPC）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--schedule", default="11",
                    help="两位调度，如 11 / 10 / 01")
    ap.add_argument("--seed", type=int, default=73, help="模型训练种子")
    ap.add_argument("--sample-seed", type=int, default=0,
                    help="独立抽样种子；所有训练种子和调度使用相同值以配对样本")
    ap.add_argument("--samples", type=int, default=32,
                    help="测试样本数，0 表示全测试集")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--backend", default="seal",
                    choices=["seal", "tenseal", "stub"])
    ap.add_argument("--mask-exact", action="store_true",
                    help="M2 v3 masks; results go to a _maskv3 directory")
    ap.add_argument("--wire", choices=sorted(WIRE_PROFILES), default="baseline",
                    help="E4 线格式；非 baseline 时结果目录自动加 _wire<名字> 后缀")
    ap.add_argument("--out-tag", default=None,
                    help="目录后缀，默认 _<backend>..._v2；不覆盖已有记录")
    ap.add_argument("--resume", action="store_true",
                    help="从已落盘的 per_sample 续跑")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()
    if args.samples < 0 or args.sample_seed < 0 or args.timeout <= 0:
        ap.error("samples/sample-seed 必须非负，timeout 必须为正")

    if len(args.schedule) != 2 or set(args.schedule) - set("01"):
        sys.exit("--schedule 必须是两位 0/1 字符串，如 11")
    args.schedule = tuple(int(b) for b in args.schedule)
    tag = "".join(str(b) for b in args.schedule)

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    model = load_canonical_model(
        root, args.dataset, args.schedule, args.seed,
        Path(args.checkpoint) if args.checkpoint else None,
    )
    test = load_test_set(root, args.dataset)
    # Independent sampling makes each schedule/checkpoint see the same prefix.
    # Random permutation avoids a class-ordered prefix, but is not stratification.
    sample_ids = [int(index) for index in np.random.default_rng(args.sample_seed).permutation(len(test))]
    test_digest = hashlib.sha256()
    for sample, label in test:
        array = np.asarray(sample, dtype="<f8")
        test_digest.update(json.dumps([int(label), list(array.shape)], separators=(",", ":")).encode())
        test_digest.update(array.tobytes(order="C"))
    test = [test[index] for index in sample_ids]
    n = len(test) if args.samples == 0 else min(args.samples, len(test))
    if not n:
        raise SystemExit("empty test set; no protocol evaluation performed")

    suffix = args.out_tag if args.out_tag is not None else f"_{args.backend}"
    if args.out_tag is None and args.wire != "baseline":
        suffix += f"_wire{args.wire}"
    if args.out_tag is None and args.mask_exact:
        suffix += "_maskv3"
    if args.out_tag is None:
        suffix += "_v2"
    out_path = (root / "results" / "raw_results" / "E1" / args.dataset
                / f"hybrid_s{tag}_acc{suffix}" / f"seed{args.seed}.json")

    domains = None
    if any(args.schedule):
        domain_path = (root / "datas" / "mask_domains"
                       / f"{args.dataset}_s{tag}_seed{args.seed}.json")
        if not domain_path.exists():
            sys.exit(f"缺少 M2 mask-domain 文件: {domain_path}")
        domains = load_mask_domains(
            domain_path,
            output_layer_id=model.output_layer_id(),
            expected_dataset=args.dataset,
            expected_schedule=args.schedule,
            expected_seed=args.seed,
            expected_dims=model.client_spec()["nonlinear_dims"],
        )

    effective_domains = (widen_domains(domains, keep=(model.output_layer_id(),))
                         if domains and args.mask_exact else domains)
    be = make_backend(args.backend)  # load the selected native module before hashing its identity
    identity = run_identity(args, model, sample_ids, len(test), effective_domains, be)
    identity["test_set_sha256"] = test_digest.hexdigest()
    rows: list[dict] = []
    state: dict = {"mpc_runtime": None, "sample_ids": sample_ids, "run_identity": identity}
    if out_path.exists() and not args.resume:
        raise SystemExit(f"refuse to overwrite {out_path}; use --resume or a new --out-tag")
    if args.resume:
        if not out_path.exists():
            raise SystemExit(f"resume record does not exist: {out_path}")
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        try:
            validate_resume(prev, identity, sample_ids, n)
        except ValueError as exc:
            raise SystemExit(f"refuse to resume {out_path}: {exc}") from exc
        rows = prev["per_sample"]
        for index, row in enumerate(rows):
            sample, label = test[index]
            prepared = _prepare(model, args.dataset, sample)
            digest = hashlib.sha256(np.asarray(prepared, dtype="<f8").tobytes(order="C")).hexdigest()
            if row["label"] != int(label) or row["prepared_input_sha256"] != digest:
                raise SystemExit(f"refuse to resume {out_path}: input/label mismatch at index {index}")
        state.update({key: value for key, value in (prev.get("setup") or {}).items()
                      if key in SETUP_KEYS})
        print(f"resume: {len(rows)} samples already on disk", flush=True)

    def flush() -> None:
        write_record(out_path, build_payload(
            args, rows, n, model, len(test), state, True), CODE_ROOT)

    if len(rows) < n:
        try:
            session_eval(be, model, test, args.dataset, rows, n,
                         args.timeout, domains, state, flush, wire=args.wire,
                         mask_exact=args.mask_exact)
        except (Exception, KeyboardInterrupt):
            # Keep the last successful sample prefix and accumulated setup costs.
            flush()
            raise

    payload = build_payload(args, rows, n, model, len(test), state, False)
    write_record(out_path, payload, CODE_ROOT)
    print(f"accuracy {sum(1 for r in rows if r['pred'] == r['label'])}/"
          f"{len(rows)} = {payload['accuracy']:.4f}  "
          f"mpc_runtime={payload['mpc_runtime']}  "
          f"err max {payload['max_abs_err']['max']:.2e}  "
          f"wall mean {payload['timing_s']['wall_mean']:.2f}s  "
          f"-> {out_path.relative_to(root)}")
    print(json.dumps(payload["timing_s"] | payload["communication"],
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
