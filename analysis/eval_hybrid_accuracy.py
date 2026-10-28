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
where <out_suffix> is `--out-tag` (default `_<backend>`).
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from eval_encrypted import load_test_set
from mask_session import load_mask_domains
from params_pi import (
    COEFF_MOD_BIT_SIZES,
    F,
    GLOBAL_SCALE_BITS,
    K_COMPARE,
    K_RING,
    POLY_MODULUS_DEGREE,
)
from pi_proto import ProtocolError, Wire, make_backend
from pi_server import serve_connection
from secure_relu_net import run_party_batch
from sci_relu import SCI_AVAILABLE, SciReLUSession

FLUSH_EVERY = 25


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=False,
                             timeout=10)
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


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
           sci_session, timings: dict) -> np.ndarray:
    """One client query over an already-authenticated session.

    Mirrors the wire loop of `pi_client._query_once`; only the input encoding
    lives here so the driver controls preprocessing (see `_prepare`).
    """
    t0 = time.perf_counter()
    ct, windows = be.im2col_encrypt(
        ctx, prepared, model.im2col_kernel, model.stride,
    )
    if windows != model.windows:
        raise ProtocolError(
            f"im2col produced {windows} windows, protocol expects {model.windows}"
        )
    timings["encrypt_input_s"] += time.perf_counter() - t0
    wire.send(b"CTXT", be.serialize(ct))

    while True:
        _, payload = wire.recv(b"CTXT")
        if not payload:
            raise ProtocolError("empty ciphertext message")
        kind, blob = payload[0], payload[1:]
        ct = be.deserialize(ctx, blob)

        t0 = time.perf_counter()
        share = np.asarray(be.decrypt(ct), dtype=np.float64)
        timings["decrypt_s"] += time.perf_counter() - t0

        if kind == 1:
            return share
        if kind == 3:
            # §7.2 dummy round: refresh a zero plaintext, no MPC.
            t0 = time.perf_counter()
            ct_out = be.encrypt(ctx, np.zeros(share.size, dtype=np.float64))
            timings["reencrypt_s"] += time.perf_counter() - t0
            wire.send(b"CTXT", b"\x03" + be.serialize(ct_out))
            continue
        if kind != 0:
            raise ProtocolError(f"unknown ciphertext kind {kind}")

        t0 = time.perf_counter()
        if sci_session is not None:
            out_share = np.asarray(
                sci_session.relu(share.tolist(), F, K_RING),
                dtype=np.float64,
            )
        else:
            out_share = np.asarray(
                run_party_batch(wire.mpc_ctx(), share.tolist(), K_RING, F,
                                K_COMPARE),
                dtype=np.float64,
            )
        timings["mpc_s"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        ct_out = be.encrypt(ctx, out_share)
        timings["reencrypt_s"] += time.perf_counter() - t0
        wire.send(b"CTXT", b"\x00" + be.serialize(ct_out))


def session_eval(be, model, test, dataset: str, rows: list[dict], n: int,
                 timeout: float, domains, state: dict, flush_cb) -> None:
    """Run one persistent server session over samples [len(rows), n).

    `state` is updated in place (mpc_runtime / keygen / context upload) so the
    partial-flush callback can see the real runtime string and setup cost.
    """
    be_server = make_backend(be.name)
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
    try:
        spec = wire.recv_json(b"SPEC")
        sci_session = None
        if spec.get("sci_port") is not None:
            if not SCI_AVAILABLE:
                raise SystemExit(
                    "server offered an SCI port but litepi_sci is not built "
                    "for this interpreter"
                )
            sci_session = SciReLUSession(2, int(spec["sci_port"]), "127.0.0.1")
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
        ctx = be.new_context()
        keygen_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        bytes_before = wire.cc.sent
        wire.send(b"CTXT", be.public_context_bytes(ctx))
        wire.recv(b"REDY")
        context_upload_s = time.perf_counter() - t0
        state.update({
            "keygen_s": round(float(state.get("keygen_s", 0.0) or 0.0)
                              + keygen_s, 4),
            "context_upload_s": round(
                float(state.get("context_upload_s", 0.0) or 0.0)
                + context_upload_s, 4),
            "context_up_bytes": int(state.get("context_up_bytes", 0) or 0)
            + int(wire.cc.sent - bytes_before),
            "sessions": int(state.get("sessions", 0) or 0) + 1,
        })

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
            t_q = time.perf_counter()
            got = _query(be, ctx, wire, prepared, model, sci_session, q)
            wire.send(b"BYE.")
            stats = wire.recv_json(b"STAT")
            wall_s = time.perf_counter() - t_q
            query_t.append(wall_s)

            if got.size < model.spec.classes:
                raise ProtocolError(
                    f"decrypted {got.size} logits, expected "
                    f"{model.spec.classes}"
                )
            logits = got[: model.spec.classes]
            err = float(np.max(np.abs(logits - ref)))
            scale = max(1e-9, float(np.max(np.abs(ref))))
            row = {
                "index": int(index),
                "label": int(label),
                "pred": int(np.argmax(logits)),
                "max_abs_err": round(err, 8),
                "rel_err": round(err / scale, 10),
                "ref_max_abs": round(float(np.max(np.abs(ref))), 6),
                "wall_s": round(wall_s, 4),
                "forward_s": float(stats["timings"]["forward_s"]),
                "mpc_s": float(stats["timings"]["mpc_s"]),
                "encrypt_s": round(q["encrypt_input_s"], 6),
                "decrypt_s": round(q["decrypt_s"], 6),
                "reencrypt_s": round(q["reencrypt_s"], 6),
                "up_bytes": int(wire.cc.sent - sent_before),
                "down_bytes": int(wire.cc.recvd - recv_before),
                "hybrid_roundtrips": int(stats.get("hybrid_roundtrips", 0)),
                "mpc_runtime": runtime,
            }
            rows.append(row)
            done = len(rows)
            acc = sum(1 for r in rows if r["pred"] == r["label"]) / done
            print(f"  {done}/{n} idx={index} pred={row['pred']} "
                  f"label={row['label']} acc={acc:.4f} err={err:.2e} "
                  f"wall={wall_s:.2f}s fwd={row['forward_s']:.2f}s "
                  f"mpc={row['mpc_s']:.3f}s", flush=True)
            if done % FLUSH_EVERY == 0 and done < n:
                flush_cb()

        state["queries"] = int(state.get("queries", 0) or 0) + (n - start)
        # DONE closes the persistent session; the server's last STAT was
        # already consumed after the final BYE.
        wire.send(b"DONE")
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
    ups = [r["up_bytes"] for r in rows if r.get("up_bytes")]
    downs = [r["down_bytes"] for r in rows if r.get("down_bytes")]
    done = len(rows)
    keygen = float(state.get("keygen_s", 0.0) or 0.0)
    ctx_up = float(state.get("context_upload_s", 0.0) or 0.0)
    return {
        "experiment": "E1",
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
        "accuracy": (sum(1 for r in rows if r["pred"] == r["label"]) / done
                     if done else 0.0),
        "accuracy_pct": (100.0 * sum(1 for r in rows if r["pred"] == r["label"])
                         / done if done else 0.0),
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
        "prediction_mismatches": sum(1 for r in rows if r["pred"] != r["label"]),
        "timing_s": {
            "keygen": round(keygen, 4),
            "context_upload": round(ctx_up, 4),
            "amortized_setup_s_per_sample": (
                round((keygen + ctx_up) / done, 4) if done else None
            ),
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
            "bytes_rows": len(ups),
            "context_up_bytes": state.get("context_up_bytes"),
            "note": "S_up/S_down 为单次查询的客户端上下行字节（不含一次性上下文上传）",
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
        },
        "per_sample": rows,
    }


def write_record(path: Path, payload: dict, root: Path) -> None:
    record = {"exp_id": "E1", "git_sha": git_sha(root),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(
                  timespec="seconds")}
    record.update(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="E1 Hybrid 密文逐样本精度（完整协议 + SCI MPC）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--schedule", default="11",
                    help="两位调度，如 11 / 10 / 01")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--samples", type=int, default=32,
                    help="测试样本数，0 表示全测试集")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--backend", default="seal",
                    choices=["seal", "tenseal", "stub"])
    ap.add_argument("--out-tag", default=None,
                    help="目录后缀，默认 _<backend>")
    ap.add_argument("--resume", action="store_true",
                    help="从已落盘的 per_sample 续跑")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args()

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
    # 顺序取前 N 个样本会踩到数据集未 shuffle 的坑（UCI-HAR 测试集按类别
    # 聚集，前 64 个只有 3 个类别，明文精度被压到 64%）。用 seed 做确定性
    # shuffle，保证 5 个种子覆盖不同样本、分布均衡，且 --resume 可复现。
    _order_rng = np.random.default_rng(args.seed)
    test = [test[i] for i in _order_rng.permutation(len(test))]
    n = len(test) if args.samples == 0 else min(args.samples, len(test))

    suffix = args.out_tag if args.out_tag is not None else f"_{args.backend}"
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

    rows: list[dict] = []
    state: dict = {"mpc_runtime": None}
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        # 续跑前必须确认磁盘上那份是同一个配置，否则两个配置的样本会混进同一份 JSON。
        if (prev.get("dataset") != args.dataset
                or prev.get("schedule") != tag
                or int(prev.get("seed", -1)) != args.seed
                or prev.get("backend") != args.backend):
            raise SystemExit(
                f"refuse to resume {out_path}: on disk "
                f"{prev.get('dataset')}/{prev.get('schedule')}/"
                f"{prev.get('seed')}/{prev.get('backend')} != "
                f"{args.dataset}/{tag}/{args.seed}/{args.backend}"
            )
        rows = prev.get("per_sample", [])
        state.update(prev.get("setup") or {})
        print(f"resume: {len(rows)} samples already on disk", flush=True)

    def flush() -> None:
        write_record(out_path, build_payload(
            args, rows, n, model, len(test), state, True), root)

    if len(rows) < n:
        be = make_backend(args.backend)
        session_eval(be, model, test, args.dataset, rows, n,
                     args.timeout, domains, state, flush)

    payload = build_payload(args, rows, n, model, len(test), state, False)
    payload["setup"] = {k: state.get(k) for k in (
        "mpc_runtime", "keygen_s", "context_upload_s", "context_up_bytes",
        "queries", "sessions",
    )}
    write_record(out_path, payload, root)
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
