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
  server         E1 forward_s per schedule (desktop).
  bytes          E4 wire_bytes (per message, per wire profile).
  r_h            results/rounds/r_h_sci_*.json with its scope; the files from
                 2026-10-29 count the main-wire control plane only (SCI's own
                 socket is not instrumented), so scope = "control-plane".
  memory         E2 M_client_peak_mb (HE-only); hybrid increment unmeasured.

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

    # ------------------------------------------------------------- helpers
    def client_ms(self, hybrid_layers: int, elements: int) -> float:
        return (self.t_enc_input_ms + self.t_dec_ms
                + hybrid_layers * (self.t_dec_ms + self.t_reenc_ms
                                   + self.t_mpc_fixed_ms)
                + elements * self.t_mpc_per_elem_ms)

    def server_ms(self, hybrid_layers: int) -> float:
        return self.t_server_base_ms + hybrid_layers * self.t_server_per_hybrid_ms

    def online_bytes(self, hybrid_layers: int, elements: int) -> int:
        ot = 0 if self.bytes_per_elem_ot is None else self.bytes_per_elem_ot * elements
        return int(self.bytes_m1 + self.bytes_m3
                   + hybrid_layers * (self.bytes_down_hybrid + self.bytes_up_hybrid)
                   + ot)

    def rounds(self, hybrid_layers: int, output_mode: str = "logits") -> float:
        return 2 + hybrid_layers * self.r_h + int(output_mode == "index")

    def memory_mb(self, hybrid_layers: int) -> float | None:
        if self.mem_per_hybrid_mb is None:
            return self.mem_base_mb if hybrid_layers == 0 else None
        return self.mem_base_mb + hybrid_layers * self.mem_per_hybrid_mb

    def latency_ms(self, hybrid_layers: int, elements: int, rtt_ms: float,
                   bandwidth_mbps: float | None = None,
                   output_mode: str = "logits") -> float:
        t = (self.client_ms(hybrid_layers, elements)
             + self.server_ms(hybrid_layers)
             + self.rounds(hybrid_layers, output_mode) * rtt_ms)
        if bandwidth_mbps:
            t += self.online_bytes(hybrid_layers, elements) * 8 / (bandwidth_mbps * 1e3)
        return t

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
    srv = {}
    for r in rows:
        srv.setdefault(r["k_h"], []).append(r["forward_s"] * 1e3)
    k = sorted(srv)
    means = [st.mean(srv[i]) for i in k]
    slope, base = (np.polyfit(k, means, 1) if len(k) > 1 else (0.0, means[0]))

    desk = _e2_client(root, "E2")
    prov = {
        "client_crypto": f"E1 SEAL per_sample, n={len(rows)} (desktop, SCI)",
        "mpc_fit": f"mpc_ms = {mpc_fixed:.3f}*K_h + {mpc_elem:.5f}*elements "
                   f"(lstsq over {len(hyb)} hybrid samples)",
        "server": "E1 forward_s linear in K_h (desktop)",
    }
    scale = 1.0
    mem_folder = "E2" if device == "desktop" else f"E2_{device}"
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
    prov["ot_bytes"] = "unmeasured: SCI bytes_sent() not recorded in E1/E2 yet"
    prov["memory"] = (f"{mem_src}" + ("" if mem_hyb is not None
                                      else "; hybrid increment unmeasured"))
    return CostModel(
        device=device, wire=wire,
        t_enc_input_ms=enc, t_dec_ms=dec, t_reenc_ms=reenc,
        t_mpc_fixed_ms=float(max(mpc_fixed, 0.0)),
        t_mpc_per_elem_ms=float(max(mpc_elem, 0.0)),
        t_server_base_ms=float(base), t_server_per_hybrid_ms=float(slope),
        **wb, bytes_per_elem_ot=None,
        r_h=r_h, r_h_scope=scope,
        mem_base_mb=float(mem or 0.0), mem_per_hybrid_mb=mem_hyb,
        provenance=prov,
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
