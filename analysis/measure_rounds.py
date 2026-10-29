"""Measure the *actual* online round trips per hybrid layer (r_h).

The protocol doc declares r_h = 1: one round trip per hybrid layer, because ①_k
(the masked ciphertext down) and ②_k (the re-encrypted share up) are merged into
one exchange.  This script measures whether that holds, by counting messages.

Method (no wall-clock timing, so the result is deterministic and machine
independent):

  1. Every wire operation is recorded with its party, direction and size,
     including the OT traffic, which bypasses ``Wire.send`` and goes straight
     through ``CountChan``.  Both parties live in one process.
  2. A round trip = one message each way (one request + one reply).  Counting
     on one party's own operation log is order-safe, because that party's
     operations are sequential by construction.
  3. Schedules are differenced on the same network and sample:
         messages(s) = setup + 2 + K_h * m_h
     hence m_h = (messages(s=11) - messages(s=00)) / 2 and
           r_h = m_h / 2, with an explicit balance check that the added
     traffic is one message in each direction per layer (otherwise a "layer"
     is not a single round trip).

Two MPC runtimes exist and must be reported separately:

  --mpc sci        production backend.  The ReLU itself runs on SCI's own
                   C++ socket, so only the main-wire traffic is visible here.
                   The SCI-internal rounds are NOT captured (flagged).
  --mpc python-ot  reference fallback.  All traffic is on the main wire and
                   fully counted, but a single query costs tens of minutes
                   because the OTs are pure Python.

Run with the 3.11 venv (its SEAL build is cpython-311):
    ./.venv311/bin/python3 analysis/measure_rounds.py --dataset mnist --mpc sci
"""

from __future__ import annotations

import argparse
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import pi_client
import pi_proto
import pi_server
from canonical_net import load_canonical_model
from mask_session import load_mask_domains
from pi_client import loopback_once

DEFAULT_ROOT = Path(__file__).resolve().parent.parent

_OPS: list[tuple[int, str, bytes, int]] = []
_LOCK = threading.Lock()
_INSTALLED = False


def install_tracer() -> None:
    """Record every send/recv, including the MPC/OT channel."""
    global _INSTALLED
    if _INSTALLED:
        return
    orig_init = pi_proto.Wire.__init__
    orig_send = pi_proto.CountChan.send
    orig_recv = pi_proto.CountChan.recv

    def init(self, conn, party):
        orig_init(self, conn, party)
        self.cc.party = party

    def send(self, data):
        orig_send(self, data)
        with _LOCK:
            _OPS.append((getattr(self, "party", -1), "S",
                         bytes(data[:pi_proto.MAX_TAG]), len(data)))

    def recv(self):
        data = orig_recv(self)
        with _LOCK:
            _OPS.append((getattr(self, "party", -1), "R",
                         bytes(data[:pi_proto.MAX_TAG]), len(data)))
        return data

    pi_proto.Wire.__init__ = init
    pi_proto.CountChan.send = send
    pi_proto.CountChan.recv = recv
    _INSTALLED = True


def summarise(ops):
    client = [k for p, k, _, _ in ops if p == 1]
    blocks, prev = 0, None
    for kind in client:
        if kind != prev:
            blocks += 1
            prev = kind
    by_tag, up, down = {}, 0, 0
    for party, kind, tag, n in ops:
        if kind != "S":
            continue
        if party == 1:
            up += 1
        else:
            down += 1
        key = tag.decode("ascii", "replace").strip() or "RAW"
        key = "".join(ch if ch.isprintable() else "." for ch in key)
        by_tag[key] = by_tag.get(key, 0) + 1
    return {
        "messages": up + down,
        "client_to_server": up,
        "server_to_client": down,
        "direction_blocks": blocks,
        "bytes": sum(n for _, kind, _, n in ops if kind == "S"),
        "by_tag": dict(sorted(by_tag.items(), key=lambda kv: -kv[1])[:8]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="measure r_h by counting messages")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "cifar10", "uci_har"])
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--backend", default="stub",
                    help="stub only avoids paying CKKS cost; the message "
                         "sequence is backend independent. Do not quote the "
                         "byte column from a stub run.")
    ap.add_argument("--mpc", default="sci", choices=["sci", "python-ot"])
    ap.add_argument("--schedules", default="00,01,10,11")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    if args.mpc == "python-ot":
        pi_server.SCI_AVAILABLE = False
        pi_client.SCI_AVAILABLE = False

    root = Path(args.root)
    install_tracer()
    model = load_canonical_model(root, args.dataset, (0, 0), args.seed,
                                 args.checkpoint)

    results = {}
    for tag in [t.strip() for t in args.schedules.split(",") if t.strip()]:
        schedule = tuple(int(c) for c in tag)
        domains = None
        if any(schedule):
            path = (root / "datas" / "mask_domains"
                    / f"{args.dataset}_s{tag}_seed{args.seed}.json")
            if not path.exists():
                print("skip s=%s: missing %s" % (tag, path.name))
                continue
            domains = load_mask_domains(
                path,
                output_layer_id=model.output_layer_id(),
                expected_dataset=args.dataset,
                expected_schedule=schedule,
                expected_seed=args.seed,
                expected_dims=model.client_spec()["nonlinear_dims"],
            )
        with _LOCK:
            _OPS.clear()
        loopback_once(
            args.backend, [int(model.input_dim)],
            seed=args.seed, dataset=args.dataset, root=root,
            schedule=schedule, checkpoint=args.checkpoint,
            mask_domains=domains, output_mode="logits",
        )
        with _LOCK:
            snapshot = list(_OPS)
        results[tag] = summarise(snapshot)
        r = results[tag]
        print("s=%s  K_h=%d  msgs=%3d (up %d / down %d)  blocks=%d  tags=%s"
              % (tag, sum(schedule), r["messages"], r["client_to_server"],
                 r["server_to_client"], r["direction_blocks"], r["by_tag"]),
              flush=True)

    payload = {
        "experiment": "r_h",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "seed": args.seed,
        "he_backend": args.backend,
        "mpc_runtime": args.mpc,
        "method": "message counting on the loopback wire; schedules differenced",
        "definition": "one round trip = one message each way; m_h = "
                      "(msgs(s=11) - msgs(s=00)) / 2; r_h = m_h / 2",
        "per_schedule": results,
    }

    if "00" in results and "11" in results:
        base = results["00"]
        d_up = results["11"]["client_to_server"] - base["client_to_server"]
        d_down = results["11"]["server_to_client"] - base["server_to_client"]
        m_h = (d_up + d_down) / 2.0
        r_h = m_h / 2.0
        payload["per_hybrid_layer"] = {
            "messages": m_h,
            "client_to_server": d_up / 2.0,
            "server_to_client": d_down / 2.0,
            "balanced": d_up == d_down,
            "round_trips": r_h,
        }
        payload["r_h_measured"] = r_h
        payload["protocol_r_h"] = 1
        print("")
        print("base s=00        msgs=%d (up %d / down %d)"
              % (base["messages"], base["client_to_server"],
                 base["server_to_client"]))
        print("per hybrid layer msgs=%.1f (up %.1f / down %.1f)  ->  r_h=%.2f"
              % (m_h, d_up / 2.0, d_down / 2.0, r_h))
        for tag in ("10", "01"):
            if tag in results:
                pred = base["messages"] + m_h
                print("check s=%s       msgs=%d  (additive prediction %.0f)"
                      % (tag, results[tag]["messages"], pred))
        if args.mpc == "sci":
            payload["caveat"] = (
                "The ReLU itself runs inside SCI's C++ socket, so these counts "
                "are the main-wire control plane only. SCI-internal rounds are "
                "not visible from Python; claiming r_h=1 end-to-end needs C++ "
                "instrumentation of litepi_sci.")

    if not args.no_save:
        out = root / "results" / "rounds" / ("r_h_%s.json" % args.mpc)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print("saved -> %s" % out.relative_to(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
