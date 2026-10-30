"""Measured cost model for the planner (E5 / E7 / E8 / E3 / E9).

Replaces the Python P-256 OT constants (12.9 ms and 3996 B per element,
41 round trips per hybrid layer) that planner.py used to carry.  Every
constant is fitted from a result file and carries its provenance:

  client crypto  E1 SEAL records (desktop, SCI runtime, 64 samples x 5 seeds x
                 4 schedules x 3 datasets): M1 encryption, per-message
                 decryption, per-layer re-encryption, MPC wall time fitted as
                 a*K_h + b*elements by least squares.
  device scale   E2 HE-only records: the client crypto ratio Pi5 / desktop
                 (same code, same messages).  Pi5 MPC time is the desktop fit
                 times that ratio, flagged "scaled-from-desktop" until E2 has
                 hybrid runs on the Pi.
  server         forward_s per (arch, schedule), measured on ciphertext.  A
                 schedule without a run has no server time: it is unknown, not
                 extrapolated from another network.
  bytes          E4 wire_bytes (per message, per wire profile) times the number
                 of ciphertexts each message really carries, plus SCI bytes per
                 element from the E1 / model B records.
  r_h            results/rounds/r_h_sci_*.json with its scope; the files from
                 2026-10-29 count the main-wire control plane only (SCI's own
                 socket is not instrumented), so scope = "control-plane".
  memory         measure_client_memory.py streamed rows (client process only),
                 else E2 M_client_peak_mb (whole stack, HE-only).

    python analysis/cost_model.py                      # print + write JSON
    python analysis/cost_model.py --device pi5 --wire compact
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NONLINEAR_DIMS = {"mnist": (256, 64), "cifar10": (1200, 128), "uci_har": (756, 128)}


@dataclass
class CostModel:
    device: str
    wire: str
    # client (ms)
    t_enc_input_ms: float
    t_dec_ms: float               # per decrypted message (①_k or M3)
    t_reenc_ms: float             # per ②_k
    t_mpc_fixed_ms: float         # per hybrid layer
    t_mpc_per_elem_ms: float      # per ReLU element
    # server (ms) per schedule bit pattern, desktop; fallback linear model
    t_server_base_ms: float
    t_server_per_hybrid_ms: float
    # bytes per message (online) and per session (keys)
    bytes_m1: int
    bytes_m3: int
    bytes_down_hybrid: int        # ①_k
    bytes_up_hybrid: int          # ②_k (ciphertext part)
    bytes_per_elem_ot: float | None   # SCI OT/VOLE bytes per element (None = unmeasured)
    bytes_keys: int
    # rounds
    r_h: float
    r_h_scope: str
    # memory
    mem_base_mb: float
    mem_per_hybrid_mb: float | None
    provenance: dict = field(default_factory=dict)
    # Measured server forward (ms) per dataset and schedule string, e.g.
    # {"mnist": {"00": 4739.0, "10": 6798.0, ...}}.  The schedule matters, not
    # just K_h: a hybrid layer resets the level, so the following FC runs on a
    # longer modulus (MNIST s10 is 1.27x s01 although both have K_h = 1).
    t_server_table_ms: dict = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    # Deep nets send several ciphertexts per message: input_blocks for M1,
    # down_blocks over all ①_k, up_blocks over all ②_k.  The defaults are the
    # canonical one-ciphertext messages.
    def client_ms(self, hybrid_layers: int, elements: int, input_blocks: int = 1,
                  down_blocks: int | None = None, up_blocks: int | None = None) -> float:
        down = hybrid_layers if down_blocks is None else down_blocks
        up = hybrid_layers if up_blocks is None else up_blocks
        return (input_blocks * self.t_enc_input_ms + self.t_dec_ms
                + down * self.t_dec_ms + up * self.t_reenc_ms
                + hybrid_layers * self.t_mpc_fixed_ms
                + elements * self.t_mpc_per_elem_ms)

    def server_ms(self, arch: str | None, bits) -> float | None:
        """Measured forward time of this schedule, None when it never ran."""
        table = (self.t_server_table_ms or {}).get(arch) if arch else None
        value = (table or {}).get("".join(str(int(b)) for b in bits))
        return None if value is None else float(value)

    def online_bytes(self, hybrid_layers: int, elements: int, input_blocks: int = 1,
                     down_blocks: int | None = None,
                     up_blocks: int | None = None) -> int | None:
        """None while SCI bytes are unmeasured and the schedule has ReLU elements."""
        if elements and self.bytes_per_elem_ot is None:
            return None
        down = hybrid_layers if down_blocks is None else down_blocks
        up = hybrid_layers if up_blocks is None else up_blocks
        return int(input_blocks * self.bytes_m1 + self.bytes_m3
                   + down * self.bytes_down_hybrid + up * self.bytes_up_hybrid
                   + (self.bytes_per_elem_ot or 0.0) * elements)

    def rounds(self, hybrid_layers: int, output_mode: str = "logits") -> float:
        return 2 + hybrid_layers * self.r_h + int(output_mode == "index")

    def memory_mb(self, hybrid_layers: int) -> float | None:
        if self.mem_per_hybrid_mb is None:
            return self.mem_base_mb if hybrid_layers == 0 else None
        return self.mem_base_mb + hybrid_layers * self.mem_per_hybrid_mb

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def load(cls, path: Path) -> "CostModel":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# ------------------------------------------------------------------ fitting

def _e1_rows(root: Path) -> list[dict]:
    rows = []
    for f in glob.glob(str(root / "results/raw_results/E1/*/hybrid_s*_acc_seal/seed*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("partial") or d.get("mpc_runtime") not in ("litepi_sci", "none", None):
            continue
        bits = [int(b) for b in str(d["schedule"])]
        dims = NONLINEAR_DIMS[d["dataset"]]
        k_h = sum(bits)
        elems = sum(n for n, b in zip(dims, bits) if b)
        for s in d.get("per_sample", []):
            rows.append({"dataset": d["dataset"], "schedule": d["schedule"],
                         "k_h": k_h, "elements": elems, **s})
    return rows


def _mid_rows(root: Path) -> list[dict]:
    """Per-query records of model B from eval_deep_accuracy.py."""
    rows = []
    for f in glob.glob(str(root / "results/raw_results/E1/cifar10/mid/s*_cipher*/seed*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        rows += [{"schedule": str(d["schedule"]), **r} for r in d["rows"]]
    return rows


def _ot_bytes_per_element(root: Path, mid_rows: list[dict]) -> tuple[float | None, str]:
    """SCI bytes (both directions) per ReLU element, from records that carry them."""
    bytes_total, elements = 0.0, 0
    for r in mid_rows:
        if r.get("up_sci_bytes"):
            bytes_total += r["up_sci_bytes"] + r["down_sci_bytes"]
            elements += r["timings"]["mpc_elements"]
    for f in glob.glob(str(root / "results/raw_results/E1/*/hybrid_s*_acc_seal*/seed*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        comm = d.get("communication", {})
        if not comm.get("S_up_sci_bytes_mean"):
            continue
        bits = [int(b) for b in str(d["schedule"])]
        n = sum(k for k, b in zip(NONLINEAR_DIMS[d["dataset"]], bits) if b)
        bytes_total += comm["S_up_sci_bytes_mean"] + comm["S_down_sci_bytes_mean"]
        elements += n
    if not elements:
        return None, "unmeasured: no record carries SCI bytes yet"
    return bytes_total / elements, f"SCI bytes over {elements} ReLU elements"


def _client_only_memory(root: Path, device: str) -> tuple[float | None, float | None, str]:
    """(base MB, MB per hybrid layer) of the streamed client process."""
    by_k: dict[int, list[float]] = {}
    pattern = f"results/raw_results/E2/client_only/{device}/seed*/summary.json"
    for f in glob.glob(str(root / pattern)):
        for r in json.loads(Path(f).read_text(encoding="utf-8"))["rows"]:
            if r.get("mode") == "streamed":
                by_k.setdefault(str(r["schedule"]).count("1"), []).append(r["client_peak_rss_mb"])
    if not by_k:
        return None, None, "none"
    ks = sorted(by_k)
    means = [st.mean(by_k[k]) for k in ks]
    if len(ks) == 1:
        return means[0], None, f"E2/client_only/{device}: K_h={ks} only"
    slope, base = np.polyfit(ks, means, 1)
    return float(base), float(slope), f"E2/client_only/{device}: fit over K_h={ks}"


def _e2_client(root: Path, folder: str) -> dict | None:
    vals = {"T_enc": [], "T_dec": [], "M_peak": []}
    for f in glob.glob(str(root / f"results/raw_results/{folder}/*/litepi_seal/seed*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        cm = d.get("client_metrics", {})
        if "T_enc" in cm:
            vals["T_enc"].append(cm["T_enc"]["mean_ms"])
            vals["T_dec"].append(cm["T_dec"]["mean_ms"])
            if isinstance(cm.get("M_client_peak_mb"), (int, float)):
                vals["M_peak"].append(cm["M_client_peak_mb"])
    summary = root / f"results/raw_results/{folder}/summary_pi5.json"
    if not vals["T_enc"] and summary.exists():
        d = json.loads(summary.read_text(encoding="utf-8"))
        for ds in d.values():
            if isinstance(ds, dict) and "metrics" in ds:
                m = ds["metrics"]
                vals["T_enc"].append(m["T_enc"]["mean"])
                vals["T_dec"].append(m["T_dec"]["mean"])
                vals["M_peak"].append(m["M_client_peak_mb"]["mean"])
    if not vals["T_enc"]:
        return None
    return {k: (st.mean(v) if v else None) for k, v in vals.items()}


def _e2_memory_by_kh(root: Path, folder: str) -> tuple[float | None, float | None, str]:
    """(base MB, MB per hybrid layer) from E2 records that carry a schedule.

    Today only s=00 runs exist, so the per-hybrid increment stays None until
    E2 is rerun with hybrid schedules (pi_client records client_peak_rss_mb).
    """
    by_k: dict[int, list[float]] = {}
    for f in glob.glob(str(root / f"results/raw_results/{folder}/*/litepi_seal*/seed*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        peak = (d.get("client_metrics") or {}).get("M_client_peak_mb")
        if not isinstance(peak, (int, float)):
            peak = d.get("client_peak_rss_mb")
        if not isinstance(peak, (int, float)):
            continue
        k_h = str(d.get("schedule", "00")).count("1")
        by_k.setdefault(k_h, []).append(float(peak))
    if not by_k:
        return None, None, "none"
    ks = sorted(by_k)
    means = [st.mean(by_k[k]) for k in ks]
    if len(ks) == 1:
        return means[0], None, f"{folder}: K_h={ks} only"
    slope, base = np.polyfit(ks, means, 1)
    return float(base), float(slope), f"{folder}: fit over K_h={ks}"


def load_r_h(root: Path) -> tuple[float, str, str]:
    """(r_h, scope, source).  Prefers an end-to-end measurement when present."""
    best = None
    for f in sorted(glob.glob(str(root / "results/rounds/r_h_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        r = d.get("r_h_measured")
        if r is None:
            continue
        scope = d.get("scope") or (
            "end-to-end" if d.get("sci_internal_rounds_counted")
            else "control-plane")
        cand = (scope == "end-to-end", float(r), scope, Path(f).name)
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    if best is None:
        return 1.0, "protocol-target", "LitePI 协议设计 §4.1 (r_h=1, not measured)"
    return best[1], best[2], best[3]


def _wire_bytes(root: Path, wire: str) -> dict | None:
    f = root / "results/raw_results/E4/all/wire_bytes/seed73.json"
    if not f.exists():
        return None
    rows = [r for r in json.loads(f.read_text(encoding="utf-8"))["rows"]
            if r["wire"]["name"] == wire]
    if not rows:
        return None
    def pick(sched, msg):
        vals = [m["bytes"] for r in rows if r["schedule"] == sched
                for m in r["messages"] if m["msg"] == msg]
        return int(max(vals)) if vals else None
    return {
        "bytes_m1": pick("00", "M1"),
        "bytes_m3": max(pick(s, "M3") for s in ("00", "01", "10", "11")),
        "bytes_down_hybrid": max(pick("10", "①_1"), pick("11", "①_2")),
        "bytes_up_hybrid": max(pick("10", "②_1"), pick("11", "②_2")),
        "bytes_keys": int(max(r["M0_key_bytes"] for r in rows)),
    }


def fit(root: Path = ROOT, device: str = "desktop", wire: str = "baseline") -> CostModel:
    rows = _e1_rows(root)
    if not rows:
        raise SystemExit("no E1 SEAL records under results/raw_results/E1")
    enc = st.mean(r["encrypt_s"] for r in rows) * 1e3
    dec = st.mean(r["decrypt_s"] / (1 + r["k_h"]) for r in rows) * 1e3
    reenc = st.mean(r["reencrypt_s"] / r["k_h"] for r in rows if r["k_h"]) * 1e3
    hyb = [r for r in rows if r["k_h"]]
    A = np.array([[r["k_h"], r["elements"]] for r in hyb], dtype=float)
    y = np.array([r["mpc_s"] * 1e3 for r in hyb])
    (mpc_fixed, mpc_elem), *_ = np.linalg.lstsq(A, y, rcond=None)
    # Server time is looked up per (dataset, schedule): a K_h-only model cannot
    # tell s01 from s10.  The pooled fit stays in the JSON for reference only.
    srv = {}
    for r in rows:
        srv.setdefault(r["k_h"], []).append(r["forward_s"] * 1e3)
    k = sorted(srv)
    means = [st.mean(srv[i]) for i in k]
    slope, base = (np.polyfit(k, means, 1) if len(k) > 1 else (0.0, means[0]))
    by_ds: dict[str, dict[str, list[float]]] = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], {}).setdefault(
            str(r["schedule"]), []).append(r["forward_s"] * 1e3)
    server_table: dict[str, dict] = {}
    for ds, scheds in by_ds.items():
        server_table[ds] = {s: float(st.mean(v)) for s, v in sorted(scheds.items())}
    mid_rows = _mid_rows(root)
    mid: dict[str, list[float]] = {}
    for r in mid_rows:
        if (r.get("server") or {}).get("forward_s") is not None:
            mid.setdefault(r["schedule"], []).append(r["server"]["forward_s"] * 1e3)
    if mid:
        server_table["conv3"] = {s: float(st.mean(v)) for s, v in sorted(mid.items())}

    desk = _e2_client(root, "E2")
    prov = {
        "client_crypto": f"E1 SEAL per_sample, n={len(rows)} (desktop, SCI)",
        "mpc_fit": f"mpc_ms = {mpc_fixed:.3f}*K_h + {mpc_elem:.5f}*elements "
                   f"(lstsq over {len(hyb)} hybrid samples)",
        "server": "E1 forward_s mean per (dataset, schedule) (desktop: MNIST on "
                  "host 99, CIFAR/UCI on container 101); schedules without a "
                  "run have no server time.  forward_s includes the server's "
                  "wait for the client inside each hybrid layer",
    }
    scale = 1.0
    mem_folder = "E2" if device == "desktop" else f"E2_{device}"
    mem_base, mem_hyb, mem_src = _client_only_memory(root, device)
    if mem_base is None:
        mem_base, mem_hyb, mem_src = _e2_memory_by_kh(root, mem_folder)
    mem = mem_base if mem_base is not None else (desk["M_peak"] if desk else None)
    if device != "desktop":
        dev = _e2_client(root, f"E2_{device}")
        if dev is None or desk is None:
            raise SystemExit(f"no E2 records for device {device!r}")
        scale = dev["T_enc"] / desk["T_enc"]
        enc, dec, reenc = enc * scale, dec * dev["T_dec"] / desk["T_dec"], reenc * scale
        mpc_fixed, mpc_elem = mpc_fixed * scale, mpc_elem * scale
        mem = mem_base if mem_base is not None else dev["M_peak"]
        prov["device_scale"] = (f"{device}/desktop T_enc ratio {scale:.2f} (E2 HE-only); "
                                "MPC scaled-from-desktop until hybrid runs on "
                                f"{device} exist")
    r_h, scope, r_src = load_r_h(root)
    prov["r_h"] = f"{r_src} (scope: {scope})"
    wb = _wire_bytes(root, wire)
    if wb is None:
        raise SystemExit("run analysis/measure_wire_bytes.py first (E4 wire bytes)")
    prov["bytes"] = f"E4 wire_bytes, profile {wire}"
    ot_per_elem, prov["ot_bytes"] = _ot_bytes_per_element(root, mid_rows)
    prov["memory"] = (f"{mem_src}" + ("" if mem_hyb is not None
                                      else "; hybrid increment unmeasured"))
    return CostModel(
        device=device, wire=wire,
        t_enc_input_ms=enc, t_dec_ms=dec, t_reenc_ms=reenc,
        t_mpc_fixed_ms=float(max(mpc_fixed, 0.0)),
        t_mpc_per_elem_ms=float(max(mpc_elem, 0.0)),
        t_server_base_ms=float(base), t_server_per_hybrid_ms=float(slope),
        **wb, bytes_per_elem_ot=ot_per_elem,
        r_h=r_h, r_h_scope=scope,
        mem_base_mb=float(mem or 0.0), mem_per_hybrid_mb=mem_hyb,
        provenance=prov, t_server_table_ms=server_table,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="fit the planner cost model")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--device", default="desktop", help="desktop | pi5 | <E2_<name> folder>")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--r-h", type=float, default=None,
                    help="override r_h (e.g. CrypTFlow2 log l + 2 = 7 for a sensitivity run)")
    args = ap.parse_args()
    cm = fit(args.root, args.device, args.wire)
    if args.r_h is not None:
        cm.r_h, cm.r_h_scope = args.r_h, "override"
        cm.provenance["r_h"] = f"--r-h {args.r_h} (manual override)"
    out = args.root / "results" / "cost_model" / f"{args.device}_{args.wire}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cm.to_dict(), indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    d = cm.to_dict()
    for k in ("t_enc_input_ms", "t_dec_ms", "t_reenc_ms", "t_mpc_fixed_ms",
              "t_mpc_per_elem_ms", "t_server_base_ms", "t_server_per_hybrid_ms",
              "bytes_m1", "bytes_m3", "bytes_down_hybrid", "bytes_up_hybrid",
              "bytes_keys", "r_h", "r_h_scope", "mem_base_mb"):
        print(f"  {k:24s} {d[k]}")
    for k, v in cm.provenance.items():
        print(f"  [{k}] {v}")
    print(f"written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
