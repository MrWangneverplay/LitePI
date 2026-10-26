"""E1 third tier: ciphertext (HE-only) accuracy over real test samples.

Tier 1/2 of E1 (`Plain-ReLU` / `Plain-Poly`) come from the training scripts.
This driver produces tier 3 on the canonical SSOT networks with the exact
protocol parameters (N=8192, [40,21x6,40], global scale 2^21): one sample is
encoded with im2col, encrypted, pushed through the protocol trace in
`pi_server.encrypted_forward`, decrypted and compared with the plaintext logits.

Usage (WSL, from the analysis directory):

    ../.venv311/bin/python eval_encrypted.py --dataset mnist --backend seal --samples 10

`samples=0` runs the full test set.  The schedule defaults to 00 (HE-only), so
the trace has no client interaction; with `01`/`10`/`11` the run needs mask
domains and is not supported here yet.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from canonical_net import load_canonical_model
from pi_proto import Wire, make_backend
from pi_server import encrypted_forward

# Same module names as client_metrics.SPEC; inlined so this driver runs in the
# system Python that carries torch (the TenSEAL-based client_metrics does not).
DATASET_MODULE = {
    "mnist": "mnist_cnn_enc.py",
    "cifar10": "cifar10_cnn_enc.py",
    "uci_har": "uci_har_cnn_enc.py",
}


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


def git_sha(root: Path) -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root),
                             capture_output=True, text=True, check=False, timeout=10)
        return out.stdout.strip()[:12]
    except Exception:
        return "unknown"


def save_record(root: Path, exp_id: str, dataset: str, method: str, seed: int,
                payload: dict) -> Path:
    out_dir = root / "results" / "raw_results" / exp_id / dataset / method
    out_dir.mkdir(parents=True, exist_ok=True)
    record = {"exp_id": exp_id, "dataset": dataset, "method": method, "seed": seed,
              "git_sha": git_sha(root),
              "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    record.update(payload)
    path = out_dir / f"seed{seed}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    return path


def _timings() -> dict:
    """Server-side counters pre-seeded with every key the trace increments."""
    return {
        "linear_s": 0.0,
        "hybrid_relu_s": 0.0,
        "hybrid_wire_sends": 0,
        "hybrid_wire_recvs": 0,
        "mpc_s": 0.0,
        "mpc_elements": 0,
        "base_ots": 0,
        "forward_s": 0.0,
    }


def _mnist_test(root: Path) -> list[tuple[np.ndarray, int]]:
    """MNIST test split straight from the idx files (no torchvision).

    The installed torchvision does not match the installed torch, so the dataset
    modules cannot be imported; the raw files under datasets/MNIST/raw are the
    canonical source anyway and give pixels in [0, 1] like ``ToTensor``.
    """
    raw = root / "datasets" / "MNIST" / "raw"
    images = np.fromfile(raw / "t10k-images-idx3-ubyte", dtype=np.uint8)[16:]
    labels = np.fromfile(raw / "t10k-labels-idx1-ubyte", dtype=np.uint8)[8:]
    images = images.reshape(-1, 28, 28).astype(np.float64) / 255.0
    return [(images[i].reshape(1, 28, 28), int(labels[i]))
            for i in range(len(labels))]


def _cifar10_test(root: Path) -> list[tuple[np.ndarray, int]]:
    """CIFAR-10 test split straight from test_batch（torchvision 版本不匹配，不能 import）。

    test_transform 只有 ToTensor，所以 /255 之后就是训练时喂入的形状 (3, 32, 32)。
    """
    import pickle
    path = root / "datasets" / "cifar-10-batches-py" / "test_batch"
    with path.open("rb") as fh:
        blob = pickle.load(fh, encoding="bytes")
    data = np.asarray(blob[b"data"], dtype=np.float64).reshape(-1, 3, 32, 32) / 255.0
    labels = np.asarray(blob[b"labels"], dtype=np.int64)
    return [(data[i], int(labels[i])) for i in range(len(labels))]


def _dataset_module(root: Path, dataset: str):
    return load_module(root / "model" / DATASET_MODULE[dataset])


def load_test_set(root: Path, dataset: str):
    if dataset == "mnist":
        return _mnist_test(root)
    if dataset == "cifar10":
        return _cifar10_test(root)
    mod = _dataset_module(root, dataset)
    return mod.load_data(128)[1].dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="E1 密文精度评测（canonical 网络）")
    ap.add_argument("--root", default=None)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--backend", default="seal", choices=["seal", "tenseal", "stub"])
    ap.add_argument("--samples", type=int, default=10,
                    help="测试样本数，0 表示全测试集")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--schedule", default="00",
                    help="两位调度，00=HE-only（本脚本只支持 00）")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-tag", default=None)
    ap.add_argument("--resume", action="store_true",
                    help="从已落盘的 per_sample 续跑（长跑被中断时用）")
    args = ap.parse_args()

    if set(args.schedule) != {"0"}:
        sys.exit("eval_encrypted.py 只支持 HE-only 调度 00；hybrid 需走 pi_client 协议回路")

    root = Path(args.root) if args.root else Path(__file__).resolve().parent.parent
    schedule = tuple(int(b) for b in args.schedule)
    model = load_canonical_model(
        root, args.dataset, schedule, args.seed,
        Path(args.checkpoint) if args.checkpoint else None,
    )
    test = load_test_set(root, args.dataset)
    n = len(test) if args.samples == 0 else min(args.samples, len(test))

    tag = args.out_tag or f"he_only_ciphertext_{args.backend}"
    out_path = (root / "results" / "raw_results" / "E1" / args.dataset / tag
                / f"seed{args.seed}.json")

    be = make_backend(args.backend)
    t0 = time.perf_counter()
    ctx = be.new_context()
    keygen_s = time.perf_counter() - t0

    left, right = socket.socketpair()
    wire = Wire(left, 1)
    rows: list[dict] = []
    if args.resume and out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        # 续跑前必须确认磁盘上那份是同一个配置，否则会把两个配置的样本混进同一份 JSON。
        if (prev.get("dataset") != args.dataset
                or prev.get("schedule") != args.schedule
                or prev.get("backend") != args.backend):
            raise SystemExit(
                f"refuse to resume {out_path}: on disk "
                f"{prev.get('dataset')}/{prev.get('schedule')}/{prev.get('backend')} "
                f"!= {args.dataset}/{args.schedule}/{args.backend}"
            )
        rows = prev.get("per_sample", [])
        print(f"resume: {len(rows)} samples already on disk", flush=True)

    def build(partial: bool) -> dict:
        done = len(rows)
        errs = [r["max_abs_err"] for r in rows]
        fwds = [r["forward_s"] for r in rows]
        ups = [r["up_bytes"] for r in rows if r.get("up_bytes")]
        downs = [r["down_bytes"] for r in rows if r.get("down_bytes")]
        return {
            "experiment": "E1",
            "tier": "HE-only ciphertext",
            "backend": args.backend,
            "backend_seal_version": getattr(be, "version", None),
            "dataset": args.dataset,
            "schedule": args.schedule,
            "seed": args.seed,
            "partial": partial,
            "n_samples": done,
            "target_samples": n,
            "test_set_size": len(test),
            "accuracy": (sum(1 for r in rows if r["pred"] == r["label"]) / done
                         if done else 0.0),
            "max_abs_err": {"mean": float(np.mean(errs)), "max": float(np.max(errs))}
            if errs else {},
            "timing_s": {
                "keygen": round(keygen_s, 3),
                "forward_mean": float(np.mean(fwds)) if fwds else None,
                "forward_max": float(np.max(fwds)) if fwds else None,
            },
            "communication": {
                # 早期落盘的 per_sample 没有 up_bytes/down_bytes 字段（--resume 续跑时会混入），
                # 因此只对带该字段的行求均值，并记录实际参与统计的行数。
                "S_up_bytes_mean": float(np.mean(ups)) if ups else None,
                "S_down_bytes_mean": float(np.mean(downs)) if downs else None,
                "bytes_rows": len(ups),
            },
            "per_sample": rows,
            "params": {
                "poly_modulus_degree": 8192,
                "coeff_mod_bit_sizes": [40, 21, 21, 21, 21, 21, 21, 40],
                "global_scale_bits": 21,
            },
        }

    try:
        for index in range(len(rows), n):
            sample, label = test[index]
            if args.dataset == "uci_har":
                # load_data already z-scores the windows, so prepare_array would
                # normalise a second time and the plain logits would disagree.
                prepared = np.asarray(sample, dtype=np.float64)
            else:
                raw = np.asarray(sample, dtype=np.float64).ravel()
                prepared = np.asarray(model.prepare_array(raw), dtype=np.float64)
            prepared = prepared.reshape(model.spec.model_shape)
            if prepared.ndim == 3 and prepared.shape[0] == 1:
                prepared = prepared[0]

            t_enc = time.perf_counter()
            ct, windows = be.im2col_encrypt(
                ctx, prepared, model.im2col_kernel, model.stride,
            )
            enc_s = time.perf_counter() - t_enc
            up = len(be.serialize(ct))

            t_fwd = time.perf_counter()
            logits_ct = encrypted_forward(model, be, ctx, ct, wire, _timings())
            fwd_s = time.perf_counter() - t_fwd
            down = len(be.serialize(logits_ct))

            got = be.decrypt(logits_ct)[: model.spec.classes]
            ref, _ = model.plain_forward_prepared(prepared.reshape(1, *model.spec.model_shape))
            ref = np.ravel(ref)[: model.spec.classes]
            err = float(np.max(np.abs(got - ref)))
            rows.append({
                "index": index,
                "label": int(label),
                "pred": int(np.argmax(got)),
                "max_abs_err": round(err, 8),
                "encrypt_s": round(enc_s, 6),
                "forward_s": round(fwd_s, 6),
                "up_bytes": up,
                "down_bytes": down,
            })
            if len(rows) % 10 == 0 or len(rows) == n:
                acc = sum(1 for r in rows if r["pred"] == r["label"]) / len(rows)
                print(f"  {len(rows)}/{n}  acc={acc:.4f}  "
                      f"err={err:.2e}  fwd={fwd_s:.2f}s", flush=True)
            # 长跑中途落盘：WSL 重启/断电时最多丢 25 个样本，配合 --resume 续跑。
            if len(rows) % 25 == 0 and len(rows) < n:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_text(
                    json.dumps(build(True), indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    finally:
        left.close()
        right.close()

    payload = build(False)
    path = save_record(root, "E1", args.dataset, tag, args.seed, payload)
    print(f"accuracy {round(payload['accuracy'] * len(rows))}/{len(rows)} = "
          f"{payload['accuracy']:.4f}  err max {payload['max_abs_err']['max']:.2e}  "
          f"-> {path.relative_to(root)}")
    print(json.dumps(payload["timing_s"] | payload["communication"],
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
