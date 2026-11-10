"""从带来源的实测记录拟合 planner 成本模型，不是端到端实测。

E1 默认只读精确 _seal[_wire<wire>]_maskv3_v2 完整 canonical 系列，
时延与逐样本 SCI 字节共用记录；legacy 必须显式选择，mid/conv3 单独标来源。
forward_s 含 hybrid 层等待客户端的时间，不能通过拟合消除此计时边界。
E2 设备缩放、E4 密文字节、round scope 和内存来源分别保留在 provenance。
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics as st
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

if __package__:
    from .compute_agreement import paired_rows
else:
    from compute_agreement import paired_rows

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
    # 历史 server forward elapsed 拟合，仅展示；不是 compute，不作回退
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
    # Measured ciphertext bytes of one query per (arch, schedule): the size of
    # M3 and of each ①_k depends on the schedule, so one per-message constant
    # overstates HE-only (1.70 MB instead of the measured 1.18 MB).
    wire_bytes_table: dict = field(default_factory=dict)
    timing_semantics_version: int | None = None
    server_he_table_ms: dict = field(default_factory=dict)
    online_wall_table_ms: dict = field(default_factory=dict)
    online_wall_context: dict = field(default_factory=dict)

    # ------------------------------------------------------------- helpers
    # Deep nets send several ciphertexts per message: input_blocks for M1,
    # down_blocks over all ①_k, up_blocks over all ②_k.  The defaults are the
    # canonical one-ciphertext messages.
    def client_ms(self, hybrid_layers: int, elements: int, input_blocks: int = 1,
                  down_blocks: int | None = None, up_blocks: int | None = None) -> float:
        """客户端局部加解密成本；非 baseline 含加密序列化，不含 MPC 或总 CPU。"""
        down = hybrid_layers if down_blocks is None else down_blocks
        up = hybrid_layers if up_blocks is None else up_blocks
        return (input_blocks * self.t_enc_input_ms + self.t_dec_ms
                + down * self.t_dec_ms + up * self.t_reenc_ms)

    def mpc_elapsed_ms(self, hybrid_layers: int, elements: int) -> float:
        """历史 MPC 交互 elapsed 拟合，仅展示，不能作为局部 CPU 或与 wall 相加。"""
        return hybrid_layers * self.t_mpc_fixed_ms + elements * self.t_mpc_per_elem_ms

    def server_ms(self, arch: str | None, bits) -> float | None:
        """仅 timing v2 的服务器局部 HE 成本；旧 forward 不作 compute 回退。"""
        if self.timing_semantics_version != 2:
            return None
        table = (self.server_he_table_ms or {}).get(arch) if arch else None
        value = (table or {}).get("".join(str(int(b)) for b in bits))
        return (float(value) if isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0 else None)

    def forward_elapsed_ms(self, arch: str | None, bits) -> float | None:
        """历史服务器 forward elapsed，仅展示，包含 wire/MPC 等待。"""
        table = (self.t_server_table_ms or {}).get(arch) if arch else None
        value = (table or {}).get("".join(str(int(b)) for b in bits))
        return None if value is None else float(value)

    def online_latency_ms(self, arch: str | None, bits, device: str, wire: str,
                          rtt_ms: float, bandwidth_mbps: float | None) -> float | None:
        """仅同设备/线格式的 loopback TCP 在线 wall baseline；不发明网络增量。"""
        context = {"device": "desktop", "wire": self.wire, "transport": "loopback_tcp",
                   "rtt_ms": 0, "bandwidth_mbps": None,
                   "scope": "online_query_through_STAT"}
        if (self.timing_semantics_version != 2 or device != self.device
                or device != "desktop" or wire != self.wire
                or rtt_ms != 0 or bandwidth_mbps is not None
                or self.online_wall_context != context):
            return None
        table = (self.online_wall_table_ms or {}).get(arch) if arch else None
        value = (table or {}).get("".join(str(int(b)) for b in bits))
        return (float(value) if isinstance(value, (int, float))
                and math.isfinite(value) and value >= 0 else None)

    def online_bytes(self, hybrid_layers: int, elements: int, input_blocks: int = 1,
                     down_blocks: int | None = None, up_blocks: int | None = None,
                     arch: str | None = None, bits=None) -> int | None:
        """Ciphertext plus SCI bytes of one query.

        The measured (arch, schedule) entry when there is one, else the message
        constants times the ciphertext counts.  None while SCI bytes are
        unmeasured and the schedule has ReLU elements.
        """
        if elements and self.bytes_per_elem_ot is None:
            return None
        ot = (self.bytes_per_elem_ot or 0.0) * elements
        table = (self.wire_bytes_table or {}).get(arch) or {}
        key = None if bits is None else "".join(str(int(b)) for b in bits)
        if key in table:
            return int(table[key] + ot)
        down = hybrid_layers if down_blocks is None else down_blocks
        up = hybrid_layers if up_blocks is None else up_blocks
        return int(input_blocks * self.bytes_m1 + self.bytes_m3
                   + down * self.bytes_down_hybrid + up * self.bytes_up_hybrid + ot)

    def rounds(self, hybrid_layers: int, output_mode: str = "logits") -> float | None:
        # Argmax has its own interactive comparisons; one extra control-plane
        # message is not a valid estimate of mode B's end-to-end rounds.
        if output_mode != "logits":
            return None
        return 2 + hybrid_layers * self.r_h

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

def resolve_e1_suffix(wire: str = "baseline", e1_suffix: str | None = None) -> str:
    suffix = e1_suffix if e1_suffix is not None else (
        "_seal" + (f"_wire{wire}" if wire != "baseline" else "") + "_maskv3_v2")
    if not re.fullmatch(r"_[A-Za-z0-9_-]+", suffix):
        raise ValueError("e1_suffix 必须是精确目录后缀，不允许路径或通配符")
    return suffix


def validate_e1_records(records: list[dict], wire: str = "baseline",
                        require_v2: bool = True) -> tuple[list[dict], list[dict], dict]:
    """校验完整记录；系列身份不一致时拒绝混合，partial/stub/error 不入池。"""
    accepted, rows, excluded = [], [], []
    protocol, data_identity, status = None, {}, None
    protocol_keys = ("params", "wire", "mask_exact", "sample_seed", "sampler",
                     "numpy_version", "source_sha256", "native_module_sha256",
                     "sci_extension_sha256")
    data_keys = ("test_set_sha256", "test_set_size", "sample_order_sha256")
    for d in records:
        source = d.get("_source", "<record>")
        samples = d.get("per_sample")
        reason = None
        if d.get("partial") or d.get("error") or d.get("errors"):
            reason = "partial/error"
        elif d.get("backend") != "seal" or d.get("measured") is False:
            reason = "非 SEAL 实测记录（含 stub）"
        elif not isinstance(samples, list) or not samples:
            reason = "没有逐样本记录"
        elif (type(d.get("n_samples")) is not int or d["n_samples"] != len(samples)
              or d.get("target_samples", d.get("n_samples") if d.get("schema_version") != 2 else None
                       ) != len(samples)):
            reason = "n_samples/target_samples/len(per_sample) 不完整"
        if reason is None and any(isinstance(r, dict) and
                                  (r.get("error") or r.get("errors") or r.get("partial")
                                   or r.get("measured") is False or r.get("backend") == "stub")
                                  for r in samples):
            reason = "逐样本 partial/stub/error"
        if reason:
            excluded.append({"path": source, "reason": reason})
            continue
        version = d.get("schema_version")
        if version not in (None, 1, 2) or (require_v2 and version != 2):
            raise ValueError(f"{source}: 要求 schema2，不能把 legacy/未知 schema 当作 v2")
        current_status = "schema_v2" if version == 2 else "legacy"
        if status is not None and status != current_status:
            raise ValueError("同一 canonical 系列不能混合 legacy 与 schema2")
        dataset, tag = d.get("dataset"), str(d.get("schedule"))
        if dataset not in NONLINEAR_DIMS or tag not in ("00", "01", "10", "11"):
            raise ValueError(f"{source}: 非 canonical dataset/schedule")
        k_h = tag.count("1")
        runtime = d.get("mpc_runtime")
        if runtime not in ("none", "litepi_sci", None) or (k_h and runtime != "litepi_sci"):
            excluded.append({"path": source, "reason": "hybrid 必须使用 litepi_sci"})
            continue
        identity = d.get("run_identity") or {}
        record_wire = d.get("wire")
        if record_wire is None and version != 2 and wire == "baseline":
            record_wire = {"name": "baseline", "legacy_unverified": True}
        if not isinstance(record_wire, dict) or record_wire.get("name") != wire:
            raise ValueError(f"{source}: wire 与请求的 {wire!r} 不一致或缺失")
        if version == 2:
            required = ("schema_version", "dataset", "schedule", "training_seed",
                        "backend", "checkpoint_sha256", "mask_domains", *protocol_keys, *data_keys)
            if not isinstance(identity, dict) or any(key not in identity for key in required):
                raise ValueError(f"{source}: schema2 run_identity 缺字段")
            if d.get("partial") is not False or d.get("measured") is not True or runtime is None:
                raise ValueError(f"{source}: schema2 未明确完成实测或缺运行时")
            mirrored = {"schema_version": 2, "dataset": dataset, "schedule": tag,
                        "training_seed": d.get("seed"), "backend": "seal",
                        "checkpoint_sha256": d.get("checkpoint_sha256"),
                        "wire": record_wire, "params": d.get("params"),
                        "mask_exact": d.get("mask_exact"), "mask_domains": d.get("mask_domains"),
                        "test_set_size": d.get("test_set_size")}
            changed = [key for key, value in mirrored.items() if identity[key] != value]
            sampling = d.get("sampling") or {}
            changed += [key for key in ("sample_seed", "sample_order_sha256")
                        if sampling.get(key) != identity[key]]
            if changed or (d.get("setup") or {}).get("mpc_runtime") != runtime:
                raise ValueError(f"{source}: schema2 顶层/身份/setup 不一致: {changed}")
            if (not isinstance(identity["params"], dict)
                    or any(key not in identity["params"] for key in (
                        "poly_modulus_degree", "coeff_mod_bit_sizes", "global_scale_bits", "f", "k_ring", "k_compare"))
                    or not isinstance(identity["mask_exact"], bool)
                    or type(identity["sample_seed"]) is not int or identity["sample_seed"] < 0
                    or type(identity["training_seed"]) is not int or identity["training_seed"] < 0
                    or type(identity["test_set_size"]) is not int or identity["test_set_size"] < len(samples)
                    or not isinstance(identity["source_sha256"], dict) or not identity["source_sha256"]
                    or not isinstance(identity["native_module_sha256"], dict) or not identity["native_module_sha256"]
                    or any(not identity[key] for key in ("checkpoint_sha256", *data_keys))):
                raise ValueError(f"{source}: 缺参数、采样、数据/source/native/checkpoint 身份")
            if k_h and (not identity["mask_domains"] or not identity["sci_extension_sha256"]):
                raise ValueError(f"{source}: hybrid 缺 mask-domain/SCI native 身份")
            try:
                paired_rows(d)  # validate actual saved logits, not just prediction field types
            except ValueError as exc:
                raise ValueError(f"{source}: invalid paired evidence: {exc}") from exc
        timing_contract = d.get("timing_contract")
        if timing_contract is not None:
            if (not isinstance(timing_contract, dict)
                    or timing_contract.get("timing_semantics_version") != 2
                    or d.get("timing_semantics_version") != 2
                    or identity.get("timing_contract") != timing_contract):
                raise ValueError(f"{source}: timing contract 顶层/身份/版本不一致")
        elif identity.get("timing_contract") is not None:
            raise ValueError(f"{source}: 顶层缺 timing contract")
        if d.get("mpc_contract") != identity.get("mpc_contract"):
            raise ValueError(f"{source}: MPC contract 顶层/身份不一致")
        signature = ({**{key: identity[key] for key in protocol_keys},
                      "mpc_contract": identity.get("mpc_contract"),
                      "timing_contract": timing_contract} if version == 2 else
                     {"params": d.get("params"), "wire": record_wire,
                      "mask_exact": d.get("mask_exact"), "identity_unverified": True})
        if protocol is not None and signature != protocol:
            changed = [key for key in signature if signature[key] != protocol.get(key)]
            raise ValueError(f"{source}: canonical protocol/source/native 身份混系列: {changed}")
        if version == 2:
            data = {key: identity[key] for key in data_keys}
            if dataset in data_identity and data_identity[dataset] != data:
                raise ValueError(f"{source}: 同 dataset 数据或采样顺序身份不一致")
            data_identity[dataset] = data
        elements = sum(n for n, bit in zip(NONLINEAR_DIMS[dataset], tag) if bit == "1")
        seen_ids = set()
        for index, sample in enumerate(samples):
            if not isinstance(sample, dict) or sample.get("error") or sample.get("partial"):
                raise ValueError(f"{source}: 无效逐样本行 {index}")
            for key in ("encrypt_s", "decrypt_s", "reencrypt_s", "mpc_s", "forward_s", "wall_s"):
                value = sample.get(key)
                if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{source}: 行 {index} 缺有效 {key}")
            if sample.get("mpc_runtime", runtime) != runtime:
                raise ValueError(f"{source}: 行 {index} runtime 不一致")
            for key in ("up_bytes", "down_bytes", "up_sci_bytes", "down_sci_bytes"):
                if key in sample and (not isinstance(sample[key], (int, float))
                                      or not math.isfinite(sample[key]) or sample[key] < 0):
                    raise ValueError(f"{source}: 行 {index} 的 {key} 无效")
            if version == 2:
                sample_id = sample.get("sample_id")
                if (sample.get("index") != index or not isinstance(sample_id, int)
                        or not 0 <= sample_id < identity["test_set_size"] or sample_id in seen_ids
                        or not sample.get("prepared_input_sha256")
                        or any(not isinstance(sample.get(key), int)
                               for key in ("plain_pred", "cipher_pred", "label"))
                        or sample.get("pred", sample["cipher_pred"]) != sample["cipher_pred"]):
                    raise ValueError(f"{source}: 行 {index} 缺有效配对样本身份/预测")
                seen_ids.add(sample_id)
            if sample.get("timing_semantics_version") not in (None, 2):
                raise ValueError(f"{source}: 行 {index} 的 timing 版本无效")
            for key in ("server_he_compute_s", "server_mpc_interactive_s", "client_mpc_interactive_s"):
                value = sample.get(key)
                if value is not None and (not isinstance(value, (int, float))
                                          or not math.isfinite(value) or value < 0):
                    raise ValueError(f"{source}: 行 {index} 的 {key} 无效")
            rows.append({**sample, "dataset": dataset, "schedule": tag,
                         "k_h": k_h, "elements": elements, "_source": source,
                         "_timing_contract": timing_contract if version == 2 else None})
        protocol, status = signature, current_status
        accepted.append(d)
    provenance = {"status": status, "protocol_identity": protocol,
                  "dataset_identity": data_identity, "excluded_records": excluded,
                  "record_count": len(accepted), "sample_count": len(rows),
                  "mask_scope": "mask_exact 协议模式共享；校准 mask_domains 绑定各 dataset/schedule/训练种子/checkpoint，单独保留",
                  "records": [{"path": d.get("_source"), "dataset": d["dataset"],
                               "schedule": str(d["schedule"]), "training_seed": d.get("seed"),
                               "checkpoint": d.get("checkpoint"),
                               "checkpoint_sha256": d.get("checkpoint_sha256"),
                               "n_samples": d["n_samples"], "target_samples": d.get("target_samples"),
                               "sampling": d.get("sampling"), "setup": d.get("setup"),
                               "mask_domains": d.get("mask_domains")} for d in accepted]}
    if status == "legacy":
        provenance["limitations"] = "legacy：原始完整行可用，但 schema2 采样/源码/native/配对身份未获验证"
    return accepted, rows, provenance


def load_e1_series(root: Path = ROOT, wire: str = "baseline", e1_suffix: str | None = None,
                   datasets=None, seeds=None) -> tuple[list[dict], list[dict], dict]:
    """时延、SCI 字节和 Adaptive 共用此精确系列读取器；不回退旧目录。"""
    suffix = resolve_e1_suffix(wire, e1_suffix)
    records = []
    for dataset in (NONLINEAR_DIMS if datasets is None else datasets):
        for tag in ("00", "01", "10", "11"):
            directory = Path(root) / "results/raw_results/E1" / dataset / f"hybrid_s{tag}_acc{suffix}"
            paths = (sorted(directory.glob("seed*.json")) if seeds is None else
                     [directory / f"seed{seed}.json" for seed in seeds])
            for path in paths:
                if not path.exists():
                    continue
                d = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(d, dict):
                    raise ValueError(f"{path}: 原始记录必须是 JSON object")
                if (d.get("dataset") != dataset or str(d.get("schedule")) != tag
                        or path.stem != f"seed{d.get('seed')}"):
                    raise ValueError(f"{path}: 路径与 dataset/schedule/seed 身份不一致")
                records.append({**d, "_source": str(path)})
    accepted, rows, provenance = validate_e1_records(
        records, wire, require_v2=e1_suffix is None or suffix.endswith("_v2"))
    if not accepted:
        kind = "missingnewseries" if e1_suffix is None or suffix.endswith("_v2") else "missingseries"
        raise FileNotFoundError(f"{kind}: 缺少完整 E1 SEAL 原始记录，精确 suffix={suffix!r}；"
                                "partial/stub/error 不可用，不回退旧数据。"
                                f"排除记录: {provenance['excluded_records']}")
    wire_suffix = "_seal" + (f"_wire{wire}" if wire != "baseline" else "")
    expected_mask = {wire_suffix + "_maskv3_v2": True, wire_suffix + "_v2": False}.get(suffix)
    if expected_mask is not None and provenance["protocol_identity"]["mask_exact"] is not expected_mask:
        raise ValueError(f"精确 suffix={suffix!r} 与 mask_exact={expected_mask} 不符")
    provenance["e1_suffix"] = suffix
    return accepted, rows, provenance


def _e1_rows(root: Path, wire: str = "baseline", e1_suffix: str | None = None) -> list[dict]:
    return load_e1_series(root, wire, e1_suffix)[1]


def _mid_rows(root: Path) -> list[dict]:
    """Per-query records of model B from eval_deep_accuracy.py."""
    rows = []
    for f in glob.glob(str(root / "results/raw_results/E1/cifar10/mid/s*_cipher*/seed*.json")):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if d.get("partial") or d.get("error") or d.get("backend") == "stub":
            continue
        rows += [{**r, "schedule": str(d["schedule"])} for r in d["rows"]]
    return rows


def _ot_bytes_per_element(rows: list[dict]) -> tuple[float | None, str]:
    """只用拟合的同一 canonical 行，按每样本 ReLU element 加权，不池化 legacy mid。"""
    hybrid = [r for r in rows if r["elements"]]
    if not hybrid or any(r.get(key) is None for r in hybrid
                         for key in ("up_sci_bytes", "down_sci_bytes")):
        return None, "unmeasured：同一 canonical 系列的 hybrid 行缺完整逐样本 SCI 字节"
    elements = sum(r["elements"] for r in hybrid)
    byte_count = sum(r["up_sci_bytes"] + r["down_sci_bytes"] for r in hybrid)
    return byte_count / elements, (f"同一 canonical 系列 {len(hybrid)} 样本 / {elements} ReLU elements；"
                                   "SCI 双向在线字节，不含 setup，不含 legacy mid")


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


def load_r_h(root: Path, mpc_contract: dict | None = None) -> tuple[float, str, str]:
    """返回 (r_h, scope, source)。

    None 保留显式 legacy 的选择/未测目标回退；合同模式仅接受相同合同的
    end-to-end 实测轮数，缺失时必须补测 netem/SCI，不能回退旧轮数或 1.0。
    """
    best = None
    files = glob.glob(str(root / "results/rounds/r_h_*.json"))
    files += glob.glob(str(root / "results/rounds/newr_h_netem_*_mpcv2.json"))
    for f in sorted(files):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if mpc_contract is not None and (
                d.get("mpc_contract") != mpc_contract or d.get("scope") != "end-to-end"):
            continue
        r = d.get("r_h_measured")
        if (not isinstance(r, (int, float)) or not math.isfinite(r) or r < 0
                or d.get("partial") or d.get("error") or d.get("measured") is False):
            continue
        scope = d.get("scope") or (
            "end-to-end" if d.get("sci_internal_rounds_counted")
            else "control-plane")
        cand = (scope == "end-to-end", float(r), scope, str(Path(f)))
        if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
            best = cand
    if best is None:
        if mpc_contract is not None:
            raise ValueError(f"缺少相同 mpc_contract={mpc_contract!r} 的有效 end-to-end 轮数；"
                             "需要新的 netem/SCI 实测（包括 BOB->ALICE 重分享消息），"
                             "旧 rounds 不能作为新协议实测，也不回退 r_h=1.0")
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
    table: dict[str, dict[str, int]] = {}
    for r in rows:
        table.setdefault(r["dataset"], {})[r["schedule"]] = int(r["online_total_bytes"])

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
        "wire_bytes_table": table,
    }


def fit(root: Path = ROOT, device: str = "desktop", wire: str = "baseline",
        e1_suffix: str | None = None) -> CostModel:
    try:
        _, rows, series = load_e1_series(root, wire, e1_suffix)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    hyb = [r for r in rows if r["k_h"]]
    if not hyb:
        raise SystemExit(f"精确系列 {series['e1_suffix']!r} 缺完整 hybrid SCI 记录；"
                         "不能拟合 re-encryption/MPC 成本，仅 HE-only 不足")
    enc = st.mean(r["encrypt_s"] for r in rows) * 1e3
    dec = st.mean(r["decrypt_s"] / (1 + r["k_h"]) for r in rows) * 1e3
    reenc = st.mean(r["reencrypt_s"] / r["k_h"] for r in hyb) * 1e3
    A = np.array([[r["k_h"], r["elements"]] for r in hyb], dtype=float)
    y = np.array([(r["client_mpc_interactive_s"] if r.get("client_mpc_interactive_s") is not None
                   else r["mpc_s"]) * 1e3 for r in hyb])
    if np.linalg.matrix_rank(A) < 2:
        raise SystemExit("完整 hybrid SCI 记录的 K_h/elements 不足以辨识两参数 MPC 模型")
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
    # 追加独立局部 HE / 在线 wall 表；保留上面的历史 forward 展示表。
    he_table, wall_table = {}, {}
    wall_context = {"device": "desktop", "wire": wire, "transport": "loopback_tcp",
                    "rtt_ms": 0, "bandwidth_mbps": None, "scope": "online_query_through_STAT"}
    timing_groups = {}
    for r in rows:
        timing_groups.setdefault((r["dataset"], str(r["schedule"])), []).append(r)
    for (ds, tag), samples in timing_groups.items():
        if not all(r.get("timing_semantics_version") == 2
                   and (r.get("_timing_contract") or {}).get("timing_semantics_version") == 2
                   for r in samples):
            continue
        if all(r.get("server_he_compute_s") is not None for r in samples):
            he_table.setdefault(ds, {})[tag] = st.mean(r["server_he_compute_s"] for r in samples) * 1e3
        if device == "desktop" and all(
                r["_timing_contract"].get("online_wall_context") == wall_context for r in samples):
            wall_table.setdefault(ds, {})[tag] = st.mean(r["wall_s"] for r in samples) * 1e3
    mid_rows = _mid_rows(root)
    mid: dict[str, list[float]] = {}
    for r in mid_rows:
        if (r.get("server") or {}).get("forward_s") is not None:
            mid.setdefault(r["schedule"], []).append(r["server"]["forward_s"] * 1e3)
    if mid:
        server_table["conv3"] = {s: float(st.mean(v)) for s, v in sorted(mid.items())}

    desk = _e2_client(root, "E2")
    prov = {
        "e1_series": series,
        "measured": False,
        "evidence_kind": "modeled_from_measured_records",
        "client_crypto": f"E1 {series['status']} 精确 suffix={series['e1_suffix']}，n={len(rows)} (desktop)",
        "mpc_fit": f"mpc_elapsed_ms = {mpc_fixed:.3f}*K_h + {mpc_elem:.5f}*elements "
                   f"(lstsq over {len(hyb)} hybrid samples；新字段取 client interactive，legacy 取 server mpc_s；仅展示)",
        "server": "timing v2 server_he_compute_s 按 (dataset, schedule) 均值；HE-only 局部成本，不是全服务器 CPU。"
                  "旧 forward 表仅 forward_elapsed_ms 展示，含 wire/MPC 等待，不作 compute 回退",
        "timing_boundary": "client_ms 为局部加解密成本（非 baseline 含加密序列化），不是 client total CPU；server_ms 仅局部 HE。"
                           "forward_s 是 elapsed not compute；任何局部项或 MPC elapsed 均不与 wall/network 相加",
        "online_wall": "仅 desktop 同 wire、transport=loopback TCP、rtt_ms=0、bandwidth_mbps=None 的"
                       "online_query_through_STAT wall baseline；不使用 E2 device 缩放，不发明 RTT/带宽增量。"
                       "RTT=0 表示此本机基准未附加链路延迟，不表示物理 TCP RTT 恰为零；其他 context 必须实测/校准",
        "timing_contract": series["protocol_identity"].get("timing_contract"),
    }
    if mid:
        prov["conv3_server"] = "legacy 独立架构：eval_deep_accuracy.py 的 mid 记录；非本次 schema2 canonical 测量"
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
    r_h, scope, r_src = load_r_h(root, series["protocol_identity"].get("mpc_contract"))
    prov["r_h"] = f"{r_src} (scope: {scope})"
    wb = _wire_bytes(root, wire)
    if wb is None:
        raise SystemExit("run analysis/measure_wire_bytes.py first (E4 wire bytes)")
    prov["bytes"] = f"E4 wire_bytes, profile {wire}"
    if wire == "baseline" and mid_rows:     # eval_deep_accuracy runs the baseline wire
        by_sched: dict[str, list[float]] = {}
        for r in mid_rows:
            if "up_sci_bytes" in r:
                by_sched.setdefault(r["schedule"], []).append(
                    r["up_bytes"] + r["down_bytes"] - r["up_sci_bytes"] - r["down_sci_bytes"])
        if by_sched:
            wb["wire_bytes_table"]["conv3"] = {s: st.mean(v) for s, v in by_sched.items()}
            prov["conv3_wire_bytes"] = "legacy mid 独立架构 baseline 密文字节；未池化入 canonical SCI 字节拟合"
    ot_per_elem, prov["ot_bytes"] = _ot_bytes_per_element(rows)
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
        timing_semantics_version=2 if he_table or wall_table else None,
        server_he_table_ms=he_table, online_wall_table_ms=wall_table,
        online_wall_context=wall_context if wall_table else {},
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="fit the planner cost model")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--device", default="desktop", help="desktop | pi5 | <E2_<name> folder>")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--e1-suffix", default=None,
                    help="精确 E1 后缀；默认 _seal[_wire<wire>]_maskv3_v2；legacy 须显式选择并标来源")
    ap.add_argument("--r-h", type=float, default=None,
                    help="override r_h (manual modeling/sensitivity input, not measured)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default results/cost_model_v2/<device>_<wire>.json")
    ap.add_argument("--overwrite", action="store_true", help="allow replacing only an explicitly selected --out")
    args = ap.parse_args()
    if args.overwrite and args.out is None:
        ap.error("--overwrite requires an explicit --out")
    if args.r_h is not None and (not math.isfinite(args.r_h) or args.r_h < 0):
        ap.error("--r-h must be finite and non-negative")
    out = args.out or (args.root / "results" / "cost_model_v2" / f"{args.device}_{args.wire}.json")
    if out.resolve().is_relative_to((args.root / "results" / "raw_results").resolve()):
        ap.error("cost model output must not overwrite raw results")
    if out.exists() and not args.overwrite:
        ap.error(f"refuse to overwrite {out}; select a new --out")
    cm = fit(args.root, args.device, args.wire, args.e1_suffix)
    if args.r_h is not None:
        cm.r_h, cm.r_h_scope = args.r_h, "override"
        cm.provenance["r_h"] = f"--r-h {args.r_h} (manual override)"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        stream.write(json.dumps(cm.to_dict(), indent=2, ensure_ascii=False, allow_nan=False) + "\n")
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
