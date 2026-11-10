"""LitePI per-layer planner.

Each nonlinear layer is scheduled as either

  * P2   (bit 0): polynomial activation in HE, zero interaction, consumes
                   one multiplicative level;
  * ReLU (bit 1): true ReLU via MPC, resets the HE depth, costs MPC elements
                   and network round trips.

Three strategies are provided:

  rule        single-layer sensitivity per element, then depth repair;
  greedy      start from all-P2, flip the layer with the best measured accuracy
              gain per element until the floor is met, then repair depth;
  constrained exhaustive 2^K enumeration (K<=10), minimum objective among the
              feasible schedules, plus a Pareto front.

Inputs are measurements, never extrapolations:

  accuracy    one retrained model per schedule: Profile.accuracies maps a bit
              string to its measured test accuracy.  The position of a P2 layer
              matters (model B probe: s0100 82.6%, s0010 80.4%, both one
              hybrid layer), so accuracy is not a function of the P2 count.
  cost        cost_model.CostModel, including the ciphertext blocks of each
              message (Layer.down_blocks / up_blocks).

局部目标只使用 client crypto HE 或 server HE-only 成本，不是全 CPU。
端到端延迟仅查同 context 的在线 wall baseline，缺测则为未知；绝不把
client、forward elapsed 和完整 network 相加。未知目标成本依原预算规则排除。

Run:
    python analysis/planner.py --arch conv3 --acc-min 0.82 --rtt-ms 50
    python analysis/planner.py --arch cifar10 --acc-min 0.58 --pareto
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Layer:
    name: str
    elements: int          # MPC elements if scheduled ReLU
    down_blocks: int = 1   # ciphertexts in ①_k
    up_blocks: int = 1     # ciphertexts in ②_k (im2col of the next conv expands)
    linear_depth: int = 1  # HE depth of the preceding linear layer
    p2_depth: int = 1      # extra depth if P2


@dataclass(frozen=True)
class Profile:
    cost: object                         # cost_model.CostModel
    accuracies: dict | None = None       # "0100" -> measured test accuracy
    rtt_ms: float = 50.0
    bandwidth_mbps: float | None = None
    depth_budget: int = 6
    input_blocks: int = 1                # ciphertexts in M1
    output_mode: str = "logits"
    # client：crypto HE 局部资源成本，不是 client total CPU，不加 network。
    # server：timing v2 HE-only 局部资源成本，不是全服务器 CPU。
    # e2e：仅同 context 的在线 wall baseline，不能 client + server + network。
    objective: str = "client"
    max_memory_mb: float | None = None
    arch: str | None = None              # key of the measured server table


@dataclass(frozen=True)
class Budget:
    max_mpc_elements: int | None = None
    max_rounds: float | None = None
    max_time_ms: float | None = None

    def __post_init__(self) -> None:
        for name in ("max_mpc_elements", "max_rounds", "max_time_ms"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class CKKSPlan:
    name: str
    poly_modulus_degree: int
    coeff_mod_bit_sizes: tuple[int, ...]
    global_scale_bits: int
    depth_budget: int
    security_bits: int | None


# Planning presets, not security claims: the last field stays None until the
# SEAL/lattice estimator has been run.  Only ckks-current-8192 is implemented.
CKKS_PRESETS = (
    CKKSPlan("ckks-current-8192", 8192, (40, 21, 21, 21, 21, 21, 21, 40), 21, 6, None),
    CKKSPlan("ckks-full", 65536, tuple([60] + [40] * 9 + [60]), 40, 9, None),
)


def required_depth(bits, layers) -> int:
    """Maximum linear+polynomial depth between hybrid-ReLU resets."""
    depth = maximum = 0
    for layer, bit in zip(layers, bits):
        depth += layer.linear_depth
        maximum = max(maximum, depth)
        if bit:
            depth = 0
        else:
            depth += layer.p2_depth
            maximum = max(maximum, depth)
    # The classification head is a linear layer after the last nonlinearity.
    return max(maximum, depth + 1)


def choose_ckks_plan(bits, layers) -> CKKSPlan | None:
    need = required_depth(bits, layers)
    return next((p for p in CKKS_PRESETS if p.depth_budget >= need), None)


def packing_plan(bits, layers, profile: Profile, ckks: CKKSPlan | None) -> dict:
    return {
        "slot_capacity": None if ckks is None else ckks.poly_modulus_degree // 2,
        "input_blocks": profile.input_blocks,
        "layers": [{"layer": layer.name, "nonlinear": "relu" if bit else "p2",
                    "elements": layer.elements,
                    "down_blocks": layer.down_blocks if bit else 0,
                    "up_blocks": layer.up_blocks if bit else 0}
                   for layer, bit in zip(layers, bits)],
    }


DEFAULT_ACC_MIN = {"mnist": 0.975, "cifar10": 0.59, "uci_har": 0.89, "conv3": 0.82}

# Canonical models from 实验方案 §3.2.1, measured five-seed E1 means
# (results/raw_results/E1/<dataset>/hybrid_s*/).
CANONICAL_MODELS = {
    "mnist": {
        "layers": (Layer("conv1", 256), Layer("fc1", 64)),
        "accuracies": {"00": 0.9797, "01": 0.9790, "10": 0.9789, "11": 0.9771},
    },
    "cifar10": {
        "layers": (Layer("conv1", 1200), Layer("fc1", 128)),
        "accuracies": {"00": 0.6059, "01": 0.5936, "10": 0.6280, "11": 0.5860},
    },
    "uci_har": {
        "layers": (Layer("conv1", 756), Layer("fc1", 128)),
        "accuracies": {"00": 0.9034, "01": 0.8867, "10": 0.9081, "11": 0.8975},
    },
}

# Alignment model B (conv3, plan §2.1).  The blocks are the deep_net geometry
# with 4096 slots: ①_1 = 8192 / 4096, ②_1 and ②_2 are the im2col rows of conv2
# (18432 slots) and conv3 (36864 slots), ②_3 is the pooled 1024-vector.
MODEL_B_LAYERS = (
    Layer("conv1", 8192, down_blocks=2, up_blocks=5),
    Layer("conv2", 4096, down_blocks=1, up_blocks=9),
    Layer("conv3", 4096, down_blocks=1, up_blocks=1),
    Layer("fc1", 256, down_blocks=1, up_blocks=1),
)
MODEL_B_INPUT_BLOCKS = 2             # 6912 im2col slots


def model_b_accuracies(root: Path = ROOT) -> dict:
    """Mean test accuracy per schedule of model B, from train_mid.py records."""
    base = root / "results" / "raw_results" / "E1" / "cifar10" / "mid"
    out = {}
    for folder in glob.glob(str(base / "s[01][01][01][01]")):
        accs = [json.loads(Path(f).read_text(encoding="utf-8"))["test_acc"]
                for f in glob.glob(str(Path(folder) / "seed*.json"))]
        if accs:
            out[Path(folder).name[1:]] = sum(accs) / len(accs)
    return out


def arch_spec(arch: str, root: Path = ROOT) -> tuple:
    """(layers, accuracies, input_blocks) of a network the planner may schedule."""
    if arch in CANONICAL_MODELS:
        spec = CANONICAL_MODELS[arch]
        return spec["layers"], spec["accuracies"], 1
    if arch == "conv3":
        return MODEL_B_LAYERS, model_b_accuracies(root), MODEL_B_INPUT_BLOCKS
    raise ValueError(f"no measured data for architecture {arch!r}")


@dataclass(frozen=True)
class Schedule:
    bits: tuple[int, ...]
    acc: float | None          # None: this schedule was never trained
    mpc_elements: int
    rounds: float | None
    time_ms: float | None      # the planner objective (see Profile.objective)
    bytes: int | None
    depth_ok: bool
    client_ms: float
    server_ms: float | None = None
    latency_ms: float | None = None
    memory_mb: float | None = None
    memory_ok: bool = True
    timing_basis: str | None = None
    latency_unknown: str | None = None

    @property
    def hybrid_layers(self) -> int:
        return sum(self.bits)

    def key(self) -> str:
        return "".join(str(b) for b in self.bits)

    def describe(self, layers) -> str:
        return " ".join(f"{l.name}={'ReLU' if b else 'P2'}"
                        for l, b in zip(layers, self.bits))


def evaluate(bits, layers, profile: Profile) -> Schedule:
    if len(bits) != len(layers):
        raise ValueError(f"schedule has {len(bits)} bits for {len(layers)} layers")
    bits = tuple(int(b) for b in bits)
    cm = profile.cost
    used = [layer for layer, bit in zip(layers, bits) if bit]
    k_h = len(used)
    mpc = sum(layer.elements for layer in used)
    down = sum(layer.down_blocks for layer in used)
    up = sum(layer.up_blocks for layer in used)
    rounds = cm.rounds(k_h, profile.output_mode)
    nbytes = cm.online_bytes(k_h, mpc, profile.input_blocks, down, up, profile.arch, bits)
    client_ms = cm.client_ms(k_h, mpc, profile.input_blocks, down, up)
    server_ms = cm.server_ms(profile.arch, bits)
    online_latency = getattr(cm, "online_latency_ms", None)
    latency = (online_latency(profile.arch, bits, cm.device, cm.wire,
                              profile.rtt_ms, profile.bandwidth_mbps)
               if callable(online_latency) and profile.output_mode == "logits" else None)
    latency_unknown = None
    if latency is None:
        if not callable(online_latency):
            latency_unknown = "cost 缺 online_latency_ms 接口；禁止旧 client + server + network 回退"
        elif profile.output_mode != "logits":
            latency_unknown = "缺此 output_mode 的在线 wall baseline"
        elif profile.rtt_ms != 0 or profile.bandwidth_mbps is not None:
            latency_unknown = "需要新增 network profile 实测/校准；不支持 RTT 或带宽增量推算"
        else:
            latency_unknown = "缺同 device/wire 的 timing v2 loopback TCP 在线 wall baseline/context；旧缓存不可作 e2e"
    objective = {"client": client_ms, "server": server_ms, "e2e": latency}[profile.objective]
    timing_basis = {"client": "client_local_crypto_includes_encrypt_serialization_for_nonbaseline_not_total_CPU",
                    "server": "server_HE_only_local_resource_not_total_CPU",
                    "e2e": "online_wall_baseline_same_context"}[profile.objective]
    acc = (profile.accuracies or {}).get("".join(map(str, bits)))
    mem = cm.memory_mb(k_h)
    mem_ok = (profile.max_memory_mb is None
              or (mem is not None and mem <= profile.max_memory_mb))
    return Schedule(bits, acc, mpc, rounds, objective, nbytes,
                    required_depth(bits, layers) <= profile.depth_budget,
                    client_ms, server_ms, latency, mem, mem_ok, timing_basis, latency_unknown)


def layer_sensitivity(layers, profile: Profile) -> list[float] | None:
    """Accuracy lost by making only layer k P2 (one retrained model per layer).

    None when any of the K single-P2 variants or the all-ReLU model is missing.
    """
    accs = profile.accuracies or {}
    base = accs.get("1" * len(layers))
    if base is None:
        return None
    out = []
    for i in range(len(layers)):
        key = "".join("0" if j == i else "1" for j in range(len(layers)))
        if key not in accs:
            return None
        out.append(base - accs[key])
    return out


def _repair_depth(bits, layers, profile: Profile) -> list[int]:
    """Flip P2 layers to hybrid until the depth fits, cheapest fix first."""
    bits = list(bits)
    while required_depth(bits, layers) > profile.depth_budget:
        p2 = [i for i, b in enumerate(bits) if b == 0]
        if not p2:
            break

        def after(i):
            trial = list(bits)
            trial[i] = 1
            return required_depth(trial, layers)

        bits[min(p2, key=lambda i: (after(i), layers[i].elements))] = 1
    return bits


def _reached(bits, layers, profile: Profile, acc_min: float) -> bool:
    acc = evaluate(bits, layers, profile).acc
    return acc is not None and acc >= acc_min


def rule_based(layers, profile: Profile, acc_min: float) -> list[int]:
    sens = layer_sensitivity(layers, profile)
    if sens is None:
        raise ValueError("rule-based needs the all-ReLU model and every "
                         "single-P2 variant (K + 1 retrained models)")
    bits = [0] * len(layers)
    order = sorted(range(len(layers)),
                   key=lambda i: sens[i] / max(layers[i].elements, 1), reverse=True)
    for i in order:
        if _reached(bits, layers, profile, acc_min):
            break
        bits[i] = 1
    return _repair_depth(bits, layers, profile)


def greedy(layers, profile: Profile, acc_min: float) -> list[int]:
    bits = [0] * len(layers)
    while not _reached(bits, layers, profile, acc_min):
        now = evaluate(bits, layers, profile).acc or 0.0
        best = None
        for i, bit in enumerate(bits):
            if bit:
                continue
            trial = list(bits)
            trial[i] = 1
            acc = evaluate(trial, layers, profile).acc
            if acc is None:
                continue
            score = (acc - now) / max(layers[i].elements, 1)
            if best is None or score > best[0]:
                best = (score, i)
        if best is None:
            break
        bits[best[1]] = 1
    return _repair_depth(bits, layers, profile)


def _schedule_constraints(sch: Schedule, budget: Budget,
                          acc_min: float | None = None) -> dict:
    """Use the same feasibility checks during search and when reporting a plan.

    max_time_ms bounds the selected objective, not necessarily end-to-end
    latency.  Unknown objective costs or communication are never feasible.
    """
    return {
        "accuracy_known": sch.acc is not None and math.isfinite(sch.acc),
        "accuracy": (sch.acc is not None and math.isfinite(sch.acc)
                     and (acc_min is None or sch.acc >= acc_min)),
        "cost_known": all(value is not None and math.isfinite(value) and value >= 0
                          for value in (sch.time_ms, sch.bytes, sch.rounds)),
        "depth": sch.depth_ok,
        "memory": sch.memory_ok,
        "mpc_elements": (budget.max_mpc_elements is None
                         or sch.mpc_elements <= budget.max_mpc_elements),
        "rounds": budget.max_rounds is None or (
            sch.rounds is not None and sch.rounds <= budget.max_rounds),
        "time": budget.max_time_ms is None or (
            sch.time_ms is not None and sch.time_ms <= budget.max_time_ms),
    }


def feasible_schedules(layers, profile: Profile, acc_min: float,
                       budget: Budget | None = None) -> list[Schedule]:
    budget = budget or Budget()
    out = []
    for bits in itertools.product((0, 1), repeat=len(layers)):
        sch = evaluate(bits, layers, profile)
        if all(_schedule_constraints(sch, budget, acc_min).values()):
            out.append(sch)
    return out


def random_schedules(layers, profile: Profile, draws: int = 5, seed: int = 0):
    """E5 baseline "随机调度 ×5": uniform random bit vectors, depth-repaired.

    Accuracy is not enforced (a random scheduler does not know it); the report
    averages the draws and states how many met the floor.
    """
    import random
    rng = random.Random(seed)
    return [evaluate(_repair_depth([rng.randint(0, 1) for _ in layers], layers, profile),
                     layers, profile) for _ in range(draws)]


def constrained(layers, profile: Profile, acc_min: float,
                budget: Budget | None = None) -> Schedule | None:
    feasible = feasible_schedules(layers, profile, acc_min, budget)
    if not feasible:
        return None
    if profile.objective == "server":
        # Ties must not be broken with client quantities (MPC elements,
        # rounds), or the "w/o client-aware" ablation is client-aware again.
        return min(feasible, key=lambda s: (s.time_ms, s.bits))
    return min(feasible, key=lambda s: (s.time_ms, s.mpc_elements, s.rounds))


def pareto_front(schedules) -> list[Schedule]:
    front = []
    for s in sorted(schedules, key=lambda x: x.time_ms):
        if any(o.time_ms <= s.time_ms and o.mpc_elements <= s.mpc_elements
               and o.acc >= s.acc for o in front):
            continue
        front = [o for o in front
                 if not (s.time_ms <= o.time_ms and s.mpc_elements <= o.mpc_elements
                         and s.acc >= o.acc)]
        front.append(s)
    return front


def plan_result(bits, layers, profile: Profile, budget: Budget | None = None,
                acc_min: float | None = None) -> dict:
    """The documented planner interface: inputs -> (s, p, packing)."""
    budget = budget or Budget()
    sch = evaluate(bits, layers, profile)
    ckks = choose_ckks_plan(bits, layers)
    cm = profile.cost
    constraints = _schedule_constraints(sch, budget, acc_min)
    return {
        "s": list(sch.bits),
        "accuracy": sch.acc,
        "mpc_elements": sch.mpc_elements,
        "hybrid_layers": sch.hybrid_layers,
        "rounds": sch.rounds,
        "online_bytes": sch.bytes,
        "objective": profile.objective,
        "objective_ms": sch.time_ms,
        "client_ms": sch.client_ms,
        "server_ms": sch.server_ms,
        "latency_ms": sch.latency_ms,
        "timing_basis": sch.timing_basis,
        "latency_unknown": sch.latency_unknown,
        "forward_elapsed_ms": (cm.forward_elapsed_ms(profile.arch, bits)
                               if callable(getattr(cm, "forward_elapsed_ms", None)) else None),
        "mpc_elapsed_ms": (cm.mpc_elapsed_ms(sch.hybrid_layers, sch.mpc_elements)
                           if callable(getattr(cm, "mpc_elapsed_ms", None)) else None),
        "memory_mb": sch.memory_mb,
        "r_h": cm.r_h,
        "r_h_scope": cm.r_h_scope,
        "cost_model": {"device": cm.device, "wire": cm.wire, "provenance": cm.provenance,
                       "timing_semantics_version": getattr(cm, "timing_semantics_version", None),
                       "online_wall_context": getattr(cm, "online_wall_context", {})},
        "p": None if ckks is None else {
            "name": ckks.name,
            "poly_modulus_degree": ckks.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(ckks.coeff_mod_bit_sizes),
            "global_scale_bits": ckks.global_scale_bits,
            "depth_budget": ckks.depth_budget,
            "security_bits": ckks.security_bits,
            "security_validated": False,
        },
        "packing": packing_plan(bits, layers, profile, ckks),
        "constraints": constraints,
        "feasible": ckks is not None and all(constraints.values()),
    }


def main() -> int:
    from cost_model import CostModel, fit

    ap = argparse.ArgumentParser(description="LitePI per-layer planner")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--arch", choices=[*CANONICAL_MODELS, "conv3"], default="cifar10")
    ap.add_argument("--strategy", choices=["rule", "greedy", "constrained"],
                    default="constrained")
    ap.add_argument("--acc-min", type=float, default=0.58)
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--bandwidth-mbps", type=float, default=None)
    ap.add_argument("--device", default="desktop")
    ap.add_argument("--wire", default="baseline")
    ap.add_argument("--cost-model", type=Path, default=None,
                    help="cost_model.py JSON; default: fit from results/")
    ap.add_argument("--objective", choices=["client", "server", "e2e"], default="client")
    ap.add_argument("--output-mode", choices=["logits", "index"], default="logits")
    ap.add_argument("--max-mpc-elements", type=int, default=None)
    ap.add_argument("--max-rounds", type=float, default=None)
    ap.add_argument("--max-time-s", type=float, default=None)
    ap.add_argument("--max-memory-mb", type=float, default=None)
    ap.add_argument("--json", action="store_true", help="emit s/p/packing JSON")
    ap.add_argument("--pareto", action="store_true")
    args = ap.parse_args()

    cost = (CostModel.load(args.cost_model) if args.cost_model
            else fit(args.root, args.device, args.wire))
    if (cost.device, cost.wire) != (args.device, args.wire):
        raise SystemExit("cost model device/wire 与请求不一致；禁止跨 context 使用在线 wall")
    layers, accuracies, input_blocks = arch_spec(args.arch, args.root)
    profile = Profile(cost=cost, accuracies=accuracies, rtt_ms=args.rtt_ms,
                      bandwidth_mbps=args.bandwidth_mbps, input_blocks=input_blocks,
                      output_mode=args.output_mode, objective=args.objective,
                      max_memory_mb=args.max_memory_mb, arch=args.arch)
    budget = Budget(args.max_mpc_elements, args.max_rounds,
                    args.max_time_s * 1000 if args.max_time_s is not None else None)
    if not args.json:
        print(f"arch={args.arch} acc_min={args.acc_min:.2%} rtt={args.rtt_ms}ms "
              f"depth_budget={profile.depth_budget} trained={len(accuracies)} "
              f"cost={cost.device}/{cost.wire} r_h={cost.r_h:g} ({cost.r_h_scope}) "
              f"objective={args.objective}")

    if args.pareto:
        feasible = feasible_schedules(layers, profile, args.acc_min, budget)
        front = pareto_front(feasible)
        print(f"feasible={len(feasible)} pareto={len(front)}")
        for s in front:
            print(f"{s.acc:6.2%}  mpc={s.mpc_elements:6d}  rounds={s.rounds:5.1f}  "
                  f"objective={s.time_ms / 1000:7.2f}s  {s.describe(layers)}")
        return 0

    if args.strategy == "rule":
        bits = rule_based(layers, profile, args.acc_min)
    elif args.strategy == "greedy":
        bits = greedy(layers, profile, args.acc_min)
    else:
        sch = constrained(layers, profile, args.acc_min, budget)
        if sch is None:
            print("infeasible: no measured schedule meets the constraints")
            return 1
        bits = list(sch.bits)

    plan = plan_result(bits, layers, profile, budget, args.acc_min)
    ok = plan["feasible"] and plan["accuracy"] is not None and plan["accuracy"] >= args.acc_min
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0 if ok else 1
    print(f"schedule: {evaluate(bits, layers, profile).describe(layers)}")
    print(f"accuracy={plan['accuracy']}  mpc_elements={plan['mpc_elements']:,}  "
          f"hybrid_layers={plan['hybrid_layers']}  rounds={plan['rounds']:g}  "
          f"client={plan['client_ms']:.1f}ms  server={plan['server_ms']}ms  "
          f"online_bytes={plan['online_bytes']}  memory={plan['memory_mb']}")
    print(f"ckks={None if plan['p'] is None else plan['p']['name']}  "
          f"constraints={plan['constraints']}  feasible={plan['feasible']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
