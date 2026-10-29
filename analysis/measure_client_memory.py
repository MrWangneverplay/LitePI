"""E2 client memory with the client in its own process (plan A).

eval_client_overhead.py ran the server's forward pass, PyTorch, the model and
the test set in the client's process, so its "M_client_peak_mb" (0.5-0.9 GB)
was the whole stack, not the client.  This driver starts pi_server.py and
pi_client.py as separate processes and reads the client's own peak RSS
(ru_maxrss, recorded by pi_client) for

    oneshot   M0 as one blob, evaluation keys kept for the whole session
    streamed  M0 one evaluation key per frame, keys never kept (--stream-keys)

The client process imports only numpy, litepi_seal and the MPC runtime (no
torch).  Server and client may share a host; only the client's RSS is read.

    python analysis/measure_client_memory.py --dataset mnist --schedules 00,11
    python analysis/measure_client_memory.py --all --repeat 5 --wire baseline
    # Pi5 client against a desktop server (one schedule per server process):
    python analysis/measure_client_memory.py --dataset mnist --schedules 11         --server 192.168.1.20:9101 --device pi5
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
DATASETS = ("mnist", "cifar10", "uci_har")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_listen(log: Path, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    """Wait for the server's "listening" log line.

    Never probe with connect(): pi_server --serve takes the probe as a client
    session, sends SPEC and blocks on its INIT, so the real client queues
    behind it until the server's timeout.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early (rc={proc.returncode}), see {log}")
        if log.exists() and "listening on" in log.read_text(errors="replace"):
            return
        time.sleep(0.5)
    raise TimeoutError(f"server did not start, see {log}")


def run_pair(dataset: str, schedule: str, mode: str, args, out_dir: Path) -> dict:
    from canonical_net import CANONICAL_SPECS   # driver only, not the client
    host, port = "127.0.0.1", _free_port()
    if args.server:
        # Remote server already running for this (dataset, schedule):
        #   python pi_server.py --dataset <ds> --schedule <s> --serve --port <p>
        host, _, p = args.server.rpartition(":")
        port = int(p)
    srv_cmd = [sys.executable, "-u", str(HERE / "pi_server.py"), "--dataset", dataset,
               "--schedule", ",".join(schedule), "--seed", str(args.seed),
               "--port", str(port), "--backend", "seal", "--wire", args.wire,
               "--serve"]
    out = out_dir / f"{dataset}_s{schedule}_{mode}.json"
    cli_cmd = [sys.executable, str(HERE / "pi_client.py"), "--host", host,
               "--port", str(port), "--backend", "seal",
               "--dim", str(CANONICAL_SPECS[dataset].input_dim),
               "--repeat", str(args.repeat), "--reuse-context",
               "--timeout", str(args.timeout), "--out", str(out)]
    if mode == "streamed":
        cli_cmd.append("--stream-keys")
    if args.server:
        subprocess.run(cli_cmd, check=True, stdout=subprocess.DEVNULL,
                       timeout=args.timeout)
    else:
        log = out.with_suffix(".server.log")
        with log.open("w") as fh:
            server = subprocess.Popen(srv_cmd, stdout=fh, stderr=subprocess.STDOUT)
        try:
            _wait_listen(log, server)
            subprocess.run(cli_cmd, check=True, stdout=subprocess.DEVNULL,
                           timeout=args.timeout)
        finally:
            server.terminate()
            try:
                server.wait(10)
            except subprocess.TimeoutExpired:
                server.kill()
    r = json.loads(out.read_text(encoding="utf-8"))
    t = r.get("timings", {})
    return {
        "dataset": dataset, "schedule": schedule, "mode": mode,
        "m0_streamed": r.get("m0_streamed"),
        "client_peak_rss_mb": r.get("client_peak_rss_mb"),
        "keygen_s": t.get("keygen_s"),
        "context_upload_s": t.get("context_upload_s"),
        "queries": args.repeat,
        "bytes_sent": r.get("bytes_sent"),
        "bytes_received": r.get("bytes_received"),
        "wire": r.get("wire", {}).get("name"),
        "mpc_runtime": r.get("mpc_runtime"),
        "raw": str(out.relative_to(args.root)) if out.is_relative_to(args.root) else str(out),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--dataset", choices=DATASETS, default="mnist")
    ap.add_argument("--all", action="store_true", help="all three datasets")
    ap.add_argument("--schedules", default="00,11")
    ap.add_argument("--modes", default="oneshot,streamed")
    ap.add_argument("--seed", type=int, default=73,
                    help="checkpoint / M2 mask-domain seed on the server")
    ap.add_argument("--repeat", type=int, default=3,
                    help="queries per session (peak covers Setup + all queries)")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--device", default="desktop", help="label for the results path")
    ap.add_argument("--server", default=None,
                    help="host:port of a pi_server --serve already running for "
                         "the given --dataset and single --schedules value "
                         "(Pi5 client, desktop server); default: start a local one")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out or (args.root / "results" / "raw_results" / "E2"
                           / "client_only" / args.device / f"seed{args.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in (DATASETS if args.all else (args.dataset,)):
        for sch in args.schedules.split(","):
            for mode in args.modes.split(","):
                r = run_pair(ds, sch, mode, args, out_dir)
                rows.append(r)
                print(f"{ds:>8} s{sch} {mode:>8}: client peak "
                      f"{r['client_peak_rss_mb']:7.1f} MB  keygen {r['keygen_s']:.3f} s  "
                      f"M0 {r['context_upload_s']:.2f} s  streamed={r['m0_streamed']}")
    (out_dir / "summary.json").write_text(json.dumps({
        "experiment": "E2", "metric": "M_client (client process only)",
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": args.device, "wire": args.wire, "seed": args.seed,
        "method": "pi_server and pi_client as separate processes; "
                  "client ru_maxrss over Setup + queries",
        "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"written -> {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
