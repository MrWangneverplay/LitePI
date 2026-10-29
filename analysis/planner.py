"""LitePI per-layer planner.

Each nonlinear layer is scheduled as either

  * P2   (bit 0): polynomial activation in HE, zero interaction, consumes
                   one multiplicative level;
  * ReLU (bit 1): true ReLU via MPC, resets the HE depth, costs MPC elements
                   and network round trips.

Three strategies are provided:

  rule        sensitivity-per-element ranking, then depth repair;
  greedy      start from all-P2, flip the best accuracy-gain-per-cost layer
              until the accuracy floor is met, then repair depth;
  constrained exhaustive 2^K enumeration (K<=10), minimum time among the
              feasible schedules, plus a Pareto front.

E5 retrain showed that a single P2 layer is nearly free (delta -0.10 to
+0.65 points), while all-P2 loses about 6 points.  The accuracy model is
therefore a combination penalty on the NUMBER of P2 layers:

    penalty(k) = 0.0025 * k * (k - 1)

k=1 -> 0, k=2 -> 0.5 points, k=5 -> 5 points, matching the fast-net
measurements.  The per-layer sensitivity fields are kept only for the
rule-based ordering; the constrained search uses the combination penalty.

Run:
    python analysis/planner.py --arch fast --acc-min 0.82 --rtt-ms 50
    python analysis/planner.py --arch fast --acc-min 0.82 --pareto
"""

from __future__ import annotations

import argparse
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from client_cost import estimate as estimate_client_cost


@dataclass(frozen=True)
class Layer:
    name: str
    elements: int          # MPC elements if scheduled ReLU
    linear_depth: int = 1  # HE depth of the preceding linear layer
    p2_depth: int = 1      # extra depth if P2
    sensitivity: float = 0.0  # accuracy drop if switched to P2
    # The next linear layer is a convolution.  deep_hybrid.py can only run a
    # ciphertext conv on client/server shares (the client im2col-encodes its
    # share); P2 -> conv would need GAZELLE-style rotations that are not
    # implemented, so such a layer must be hybrid to be executable today.
    next_is_conv: bool = False


@dataclass(frozen=True)
class Profile:
    rtt_ms: float = 50.0
    per_element_ms: float = 12.9
    bytes_per_element: int = 3996
    rtts_per_hybrid: int = 41   # ~82 channel messages -> ~41 round trips
    messages_per_hybrid: int = 82
    depth_budget: int = 6
    base_acc: float = 0.8509    # all-ReLU accuracy
    p2_penalty_coeff: float = 0.0025
    # Canonical K=2 schedule order: 00, 01, 10, 11.
    schedule_accuracies: tuple[float, float, float, float] | None = None
    encoded_input_slots: int | None = None
    # Measured cost model (cost_model.CostModel).  When set, it replaces the
    # per_element_ms / bytes_per_element / rtts_per_hybrid constants above,
    # which date from the Python P-256 OT reference (12.9 ms, 3996 B, 41).
    cost: object | None = None
    bandwidth_mbps: float | None = None
    output_mode: str = "logits"
    # "client": minimise client compute + client-visible network cost
    #           (rounds * RTT + bytes / bandwidth) - the client-aware target;
    # "server": minimise server compute only (E7 ablation w/o client-aware);
    # "e2e":    client + server + network (end-to-end latency, E3).
    objective: str = "client"
    max_memory_mb: float | None = None
    # Architecture key for the cost model's measured (arch, schedule) server
    # table; None falls back to the pooled linear model.
    arch: str | None = None
    # Only schedules the current ciphertext code can run (see
    # Layer.next_is_conv).  Off by default: the planner itself is
    # protocol-level; gen_e5_data.py reports both.
    executable_only: bool = False

    def penalty(self, k: int) -> float:
        return self.p2_penalty_coeff * k * (k - 1)


@dataclass(frozen=True)
class ClientProfile:
    per_element_ms: float = 12.9
    encrypt_ms: float = 20.0
    decrypt_ms: float = 20.0
    peak_memory_mb: float = 512.0


@dataclass(frozen=True)
class NetworkProfile:
    rtt_ms: float = 50.0
    bytes_per_element: int = 3996
    rtts_per_hybrid: int = 41


@dataclass(frozen=True)
class Budget:
    max_mpc_elements: int | None = None
    max_rtts: int | None = None
    max_time_ms: float | None = None
    max_memory_mb: float | None = None


@dataclass(frozen=True)
class CKKSPlan:
    name: str
    poly_modulus_degree: int
    coeff_mod_bit_sizes: tuple[int, ...]
    global_scale_bits: int
    depth_budget: int
    security_bits: int | None


# These are planning presets, not security claims.  The last field must be
# filled only after running the SEAL/TenSEAL security estimator.
CKKS_PRESETS = (
    CKKSPlan("ckks-light", 8192, (40, 21, 21, 21, 21, 21, 21, 40), 21, 2, None),
    # Current canonical implementation from 实验方案 §4.1: one shared
    # N=8192 chain with depth 5 and a budget of 6 multiplications.
    CKKSPlan(
        "ckks-current-8192",
        8192,
        (40, 21, 21, 21, 21, 21, 21, 40),
        21,
        6,
        None,
    ),
    CKKSPlan("ckks-mid", 16384, (60, 40, 40, 40, 60), 40, 3, None),
    CKKSPlan("ckks-deep", 32768, (60, 40, 40, 40, 40, 40, 40, 60), 40, 5, None),
    CKKSPlan("ckks-full", 65536, tuple([60] + [40] * 9 + [60]), 40, 9, None),
)


DEFAULT_COST_MODEL = (Path(__file__).resolve().parent.parent / "results"
                      / "cost_model" / "desktop_baseline.json")


def load_cost(path):
    """CostModel from cost_model.py, or None (legacy constants)."""
    if path is None or str(path).lower() == "none":
        return None
    path = Path(path)
    if not path.exists():
        print(f"[planner] cost model {path} missing; run analysis/cost_model.py "
              "(falling back to the legacy Python-OT constants)")
        return None
    from cost_model import CostModel
    return CostModel.load(path)


def required_depth(bits, layers) -> int:
    """Maximum linear+polynomial depth between hybrid-ReLU resets."""
    depth = 0
    maximum = 0
    for layer, bit in zip(layers, bits):
        depth += layer.linear_depth
        maximum = max(maximum, depth)
        if bit:
            depth = 0
        else:
            depth += layer.p2_depth
            maximum = max(maximum, depth)
    # fc2 follows the second nonlinear layer in all canonical networks.
    return max(maximum, depth + 1)


def choose_ckks_plan(bits, layers,
                     encoded_input_slots: int | None = None) -> CKKSPlan:
    need = required_depth(bits, layers)
    for plan in CKKS_PRESETS:
        if (
            plan.depth_budget >= need
            and plan.poly_modulus_degree // 2 >= (encoded_input_slots or 0)
        ):
            return plan
    return CKKS_PRESETS[-1]


def packing_plan(bits, layers, ckks: CKKSPlan,
                 encoded_input_slots: int | None = None) -> dict:
    slots = ckks.poly_modulus_degree // 2
    rows = []
    for layer, bit in zip(layers, bits):
        if bit:
            chunks = (layer.elements + slots - 1) // slots
            rows.append({
                "layer": layer.name,
                "nonlinear": "relu",
                "elements": layer.elements,
                "slots": layer.elements,
                "ciphertext_chunks": chunks,
                "estimated": True,
            })
        else:
            rows.append({
                "layer": layer.name,
                "nonlinear": "p2",
                "elements": layer.elements,
                "slots": layer.elements,
                "ciphertext_chunks": (layer.elements + slots - 1) // slots,
                "estimated": True,
            })
    return {
        "slot_capacity": slots,
        "encoded_input_slots": encoded_input_slots,
        "input_fits_one_ciphertext": (
            encoded_input_slots is None or encoded_input_slots <= slots
        ),
        "layers": rows,
        "estimated": True,
    }


def plan_result(bits, layers, profile: Profile,
                client: ClientProfile | None = None,
                network: NetworkProfile | None = None,
                budget: Budget | None = None,
                output_mode: str = "logits") -> dict:
    """Implement the documented planner interface: inputs -> (s, p, packing)."""
    client = client or ClientProfile()
    network = network or NetworkProfile(
        rtt_ms=profile.rtt_ms,
        bytes_per_element=profile.bytes_per_element,
        rtts_per_hybrid=profile.rtts_per_hybrid,
    )
    budget = budget or Budget()
    reference_schedule = evaluate(bits, layers, profile)
    target_rtts = (
        2 + reference_schedule.hybrid_layers + int(output_mode == "index")
    )
    target_time_ms = (
        reference_schedule.mpc_elements * client.per_element_ms
        + target_rtts * network.rtt_ms
    )
    # The planner-facing objective uses the §12 target: r_h=1.
    schedule = Schedule(
        reference_schedule.bits,
        reference_schedule.acc,
        reference_schedule.mpc_elements,
        target_rtts,
        target_time_ms,
        reference_schedule.bytes,
        reference_schedule.depth_ok,
    )
    ckks = choose_ckks_plan(bits, layers, profile.encoded_input_slots)
    packing = packing_plan(
        bits, layers, ckks, profile.encoded_input_slots,
    )
    reference_rtts = (
        41 * schedule.hybrid_layers + int(output_mode == "index")
    )
    client_cost = estimate_client_cost(
        schedule.hybrid_layers, schedule.mpc_elements, output_mode,
    )
    constraints = {
        "mpc_elements": budget.max_mpc_elements is None
        or schedule.mpc_elements <= budget.max_mpc_elements,
        "rtts": budget.max_rtts is None or schedule.rtts <= budget.max_rtts,
        "time": budget.max_time_ms is None or schedule.time_ms <= budget.max_time_ms,
        "memory": budget.max_memory_mb is None
        or client.peak_memory_mb <= budget.max_memory_mb,
        "packing": packing["input_fits_one_ciphertext"],
    }
    measured = None
    if profile.cost is not None:
        sch_m = evaluate(bits, layers, profile)
        measured = {
            "objective": profile.objective,
            "objective_ms": sch_m.time_ms,
            "client_ms": sch_m.client_ms,
            "server_ms": sch_m.server_ms,
            "latency_ms": sch_m.latency_ms,
            "rounds": sch_m.rtts,
            "online_bytes": sch_m.bytes,
            "memory_mb": sch_m.memory_mb,
            "memory_ok": sch_m.memory_ok,
            "r_h": profile.cost.r_h,
            "r_h_scope": profile.cost.r_h_scope,
            "cost_model": {"device": profile.cost.device,
                           "wire": profile.cost.wire,
                           "provenance": profile.cost.provenance},
        }
        constraints["memory"] = sch_m.memory_ok
    return {
        "s": list(schedule.bits),
        "measured_cost": measured,
        "output_mode": output_mode,
        "accuracy": schedule.acc,
        "mpc_elements": schedule.mpc_elements,
        "hybrid_layers": schedule.hybrid_layers,
        "rtts": schedule.rtts,
        "time_ms": schedule.time_ms,
        "time_basis": "protocol-target-r_h=1",
        "reference_rtts": reference_rtts,
        "reference_time_ms": (
            schedule.mpc_elements * client.per_element_ms
            + reference_rtts * network.rtt_ms
        ),
        "bytes": client_cost.total_bytes,
        "bytes_basis": "protocol-target-§6",
        "reference_bytes": reference_schedule.bytes,
        "depth_ok": schedule.depth_ok,
        "p": {
            "name": ckks.name,
            "poly_modulus_degree": ckks.poly_modulus_degree,
            "coeff_mod_bit_sizes": list(ckks.coeff_mod_bit_sizes),
            "global_scale_bits": ckks.global_scale_bits,
            "depth_budget": ckks.depth_budget,
            "security_bits": ckks.security_bits,
            "security_validated": False,
        },
        "packing": packing,
        "client_cost": client_cost.__dict__,
        "client_profile": client.__dict__,
        "network_profile": network.__dict__,
        "budget": budget.__dict__,
        "constraints": constraints,
        "feasible": schedule.depth_ok and all(constraints.values()),
    }


# Fast early-downsampling net: 20,736 ReLU elements in total.
FAST_LAYERS = (
    Layer("conv1", 8192, sensitivity=0.005, next_is_conv=True),
    Layer("conv2", 4096, sensitivity=0.008, next_is_conv=True),
    Layer("conv3", 4096, sensitivity=0.012, next_is_conv=True),
    Layer("conv4", 4096, sensitivity=0.015),
    Layer("fc1",   256, sensitivity=0.014),
)

# 4-conv VGG (64 channels): 164,096 ReLU elements in total.
VGG_LAYERS = (
    Layer("conv1", 65536, sensitivity=0.010, next_is_conv=True),
    Layer("conv2", 65536, sensitivity=0.015, next_is_conv=True),
    Layer("conv3", 16384, sensitivity=0.020, next_is_conv=True),
    Layer("conv4", 16384, sensitivity=0.025),
    Layer("fc1",    256, sensitivity=0.0105),
)

# Canonical models from 实验方案 §3.2.1.  Accuracy values are the measured
# five-seed E1 means from results/raw_results/E1/<dataset>/hybrid_s*/.
CANONICAL_MODELS = {
    "mnist": {
        "layers": (
            Layer("conv1", 256, sensitivity=0.0000),
            Layer("fc1", 64, sensitivity=0.0000),
        ),
        "accuracies": (0.9797, 0.9790, 0.9789, 0.9771),
        "slots": 4096,
    },
    "cifar10": {
        "layers": (
            Layer("conv1", 1200, sensitivity=0.0000),
            Layer("fc1", 128, sensitivity=0.0000),
        ),
        "accuracies": (0.6059, 0.5936, 0.6280, 0.5860),
        "slots": 3200,
    },
    "uci_har": {
        "layers": (
            Layer("conv1", 756, sensitivity=0.0000),
            Layer("fc1", 128, sensitivity=0.0000),
        ),
        "accuracies": (0.9034, 0.8867, 0.9081, 0.8975),
        "slots": 2016,
    },
}


@dataclass(frozen=True)
class Schedule:
    bits: tuple[int, ...]
    acc: float
    mpc_elements: int
    rtts: int
    time_ms: float          # the planner objective (see Profile.objective)
    bytes: int
    depth_ok: bool
    client_ms: float | None = None
    server_ms: float | None = None
    latency_ms: float | None = None
    memory_mb: float | None = None
    memory_ok: bool = True

    @property
    def hybrid_layers(self) -> int:
        return sum(self.bits)

    def describe(self, layers) -> str:
        parts = [f"{l.name}={'ReLU' if b else 'P2'}" for l, b in zip(layers, self.bits)]
        return " ".join(parts)


def evaluate(bits, layers, profile: Profile) -> Schedule:
    if len(bits) != len(layers):
        raise ValueError(f"schedule has {len(bits)} bits for {len(layers)} layers")
    depth = 0
    ok = True
    mpc = 0
    rtts = 0
    nbytes = 0
    p2_count = 0
    for layer, bit in zip(layers, bits):
        depth += layer.linear_depth
        if depth > profile.depth_budget:
            ok = False
        if bit:
            mpc += layer.elements
            rtts += profile.rtts_per_hybrid
            nbytes += layer.elements * profile.bytes_per_element
            depth = 0  # fresh ciphertext after the MPC ReLU
        else:
            depth += layer.p2_depth
            p2_count += 1
            if depth > profile.depth_budget:
                ok = False
    # The classification head is a linear layer after the last nonlinearity.
    depth += 1
    if depth > profile.depth_budget:
        ok = False
    if profile.executable_only and any(
            layer.next_is_conv and not bit for layer, bit in zip(layers, bits)):
        ok = False
    if profile.schedule_accuracies is not None and len(bits) == 2:
        acc = profile.schedule_accuracies[(int(bits[0]) << 1) | int(bits[1])]
    else:
        acc = profile.base_acc - profile.penalty(p2_count)
    cm = profile.cost
    if cm is None:          # legacy constants (kept for regression only)
        time_ms = mpc * profile.per_element_ms + rtts * profile.rtt_ms
        return Schedule(tuple(bits), acc, mpc, rtts, time_ms, nbytes, ok)
    k_h = sum(int(b) for b in bits)
    rounds = cm.rounds(k_h, profile.output_mode)
    nbytes = cm.online_bytes(k_h, mpc)
    net_ms = rounds * profile.rtt_ms + (
        nbytes * 8 / (profile.bandwidth_mbps * 1e3) if profile.bandwidth_mbps else 0.0)
    client_ms = cm.client_ms(k_h, mpc)
    server_ms = cm.server_ms(k_h, profile.arch, bits)
    latency = client_ms + server_ms + net_ms
    objective = {"client": client_ms + net_ms, "server": server_ms,
                 "e2e": latency}[profile.objective]
    mem = cm.memory_mb(k_h)
    mem_ok = (profile.max_memory_mb is None
              or (mem is not None and mem <= profile.max_memory_mb))
    return Schedule(tuple(bits), acc, mpc, rounds, objective, nbytes, ok,
                    client_ms, server_ms, latency, mem, mem_ok)


def _repair_depth(bits, layers, profile: Profile):
    bits = list(bits)
    while not evaluate(bits, layers, profile).depth_ok:
        p2 = [i for i, b in enumerate(bits) if b == 0]
        if not p2:
            break
        bits[max(p2, key=lambda i: layers[i].elements)] = 1
    return bits


def rule_based(layers, profile: Profile, acc_min: float):
    bits = [0] * len(layers)
    order = sorted(range(len(layers)),
                   key=lambda i: layers[i].sensitivity / max(layers[i].elements, 1),
                   reverse=True)
    for i in order:
        if evaluate(bits, layers, profile).acc >= acc_min:
            break
        bits[i] = 1
    return _repair_depth(bits, layers, profile)


def greedy(layers, profile: Profile, acc_min: float):
    bits = [0] * len(layers)
    while evaluate(bits, layers, profile).acc < acc_min:
        k = sum(1 for b in bits if b == 0)
        gain = profile.penalty(k) - profile.penalty(k - 1)
        best = None
        for i, bit in enumerate(bits):
            if bit:
                continue
            # Flipping a P2 layer to ReLU removes one P2; the least MPC cost
            # for the same accuracy gain is the smallest layer.
            cost = layers[i].elements
            score = gain / max(cost, 1e-9)
            if best is None or score > best[0]:
                best = (score, i)
        if best is None:
            break
        bits[best[1]] = 1
    return _repair_depth(bits, layers, profile)


def feasible_schedules(layers, profile: Profile, acc_min: float):
    out = []
    for bits in itertools.product((0, 1), repeat=len(layers)):
        sch = evaluate(bits, layers, profile)
        if sch.depth_ok and sch.memory_ok and sch.acc >= acc_min:
            out.append(sch)
    return out


def random_schedules(layers, profile: Profile, acc_min: float,
                     draws: int = 5, seed: int = 0):
    """E5 baseline "随机调度 ×5": uniform random bit vectors, depth-repaired.

    Accuracy is not enforced (a random scheduler does not know it); the
    report averages the five draws and states how many met acc_min.
    """
    import random
    rng = random.Random(seed)
    out = []
    for _ in range(draws):
        bits = [rng.randint(0, 1) for _ in layers]
        out.append(evaluate(_repair_depth(bits, layers, profile), layers, profile))
    return out


def constrained(layers, profile: Profile, acc_min: float):
    feasible = feasible_schedules(layers, profile, acc_min)
    if not feasible:
        return None
    if profile.objective == "server":
        # Ties must not be broken with client quantities (MPC elements,
        # rounds), or the "w/o client-aware" ablation is client-aware again.
        return min(feasible, key=lambda s: (s.time_ms, s.bits))
    return min(feasible, key=lambda s: (s.time_ms, s.mpc_elements, s.rtts))


def pareto_front(schedules):
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


def main() -> int:
    ap = argparse.ArgumentParser(description="LitePI per-layer planner")
    ap.add_argument(
        "--arch",
        choices=["mnist", "cifar10", "uci_har", "fast", "vgg"],
        default="cifar10",
    )
    ap.add_argument("--strategy", choices=["rule", "greedy", "constrained"],
                    default="constrained")
    ap.add_argument("--acc-min", type=float, default=0.58)
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--per-element-ms", type=float, default=12.9)
    ap.add_argument("--bytes-per-element", type=int, default=3996)
    ap.add_argument("--rtts-per-hybrid", type=int, default=1,
                    help="planner target: one round trip per hybrid layer")
    ap.add_argument("--client-memory-mb", type=float, default=512.0)
    ap.add_argument("--max-mpc-elements", type=int, default=None)
    ap.add_argument("--max-rtts", type=int, default=None)
    ap.add_argument("--max-time-s", type=float, default=None)
    ap.add_argument("--max-memory-mb", type=float, default=None)
    ap.add_argument("--json", action="store_true", help="emit s/p/packing JSON")
    ap.add_argument("--output-mode", choices=["logits", "index"],
                    default="logits")
    ap.add_argument("--pareto", action="store_true")
    ap.add_argument("--cost-model", type=Path, default=DEFAULT_COST_MODEL,
                    help="cost_model.py JSON (default desktop_baseline); "
                         "'none' = legacy Python-OT constants")
    ap.add_argument("--bandwidth-mbps", type=float, default=None)
    ap.add_argument("--objective", choices=["client", "server", "e2e"],
                    default="client")
    args = ap.parse_args()
    cost = load_cost(args.cost_model)

    if args.arch in CANONICAL_MODELS:
        canonical = CANONICAL_MODELS[args.arch]
        layers = canonical["layers"]
        schedule_accuracies = canonical["accuracies"]
        encoded_input_slots = canonical["slots"]
        base_acc = schedule_accuracies[-1]
        accuracy_source = "measured-canonical-schedule"
    else:
        layers = FAST_LAYERS if args.arch == "fast" else VGG_LAYERS
        schedule_accuracies = None
        encoded_input_slots = None
        base_acc = 0.8509 if args.arch == "fast" else 0.8855
        accuracy_source = "fast-net-combination-penalty"
    profile = Profile(
        rtt_ms=args.rtt_ms,
        per_element_ms=args.per_element_ms,
        bytes_per_element=args.bytes_per_element,
        rtts_per_hybrid=args.rtts_per_hybrid,
        base_acc=base_acc,
        schedule_accuracies=schedule_accuracies,
        encoded_input_slots=encoded_input_slots,
        cost=cost,
        bandwidth_mbps=args.bandwidth_mbps,
        output_mode=args.output_mode,
        objective=args.objective,
        max_memory_mb=args.max_memory_mb,
        arch=args.arch,
    )
    client = ClientProfile(per_element_ms=args.per_element_ms,
                           peak_memory_mb=args.client_memory_mb)
    network = NetworkProfile(rtt_ms=args.rtt_ms,
                             bytes_per_element=args.bytes_per_element,
                             rtts_per_hybrid=args.rtts_per_hybrid)
    budget = Budget(
        max_mpc_elements=args.max_mpc_elements,
        max_rtts=args.max_rtts,
        max_time_ms=(args.max_time_s * 1000 if args.max_time_s else None),
        max_memory_mb=args.max_memory_mb,
    )
    if not args.json:
      print(f"arch={args.arch} acc_min={args.acc_min:.2%} rtt={args.rtt_ms}ms "
          f"depth_budget={profile.depth_budget} accuracy={accuracy_source} "
          f"cost={'legacy-python-OT' if cost is None else cost.device + '/' + cost.wire}"
          f"{'' if cost is None else f' r_h={cost.r_h:g} ({cost.r_h_scope})'} "
          f"objective={args.objective}")

    if args.pareto:
        feasible = feasible_schedules(layers, profile, args.acc_min)
        front = pareto_front(feasible)
        print(f"feasible={len(feasible)} pareto={len(front)}")
        for s in front:
            print(f"{s.acc:6.2%}  mpc={s.mpc_elements:6d}  "
                  f"rtts={s.rtts:4d}  time={s.time_ms/1000:7.2f}s  "
                  f"{s.describe(layers)}")
        return 0

    if args.strategy == "rule":
        bits = rule_based(layers, profile, args.acc_min)
    elif args.strategy == "greedy":
        bits = greedy(layers, profile, args.acc_min)
    else:
        sch = constrained(layers, profile, args.acc_min)
        if sch is None:
            print("infeasible: no schedule meets the accuracy/depth constraints")
            return 1
        bits = list(sch.bits)

    sch = evaluate(bits, layers, profile)
    plan = plan_result(
        bits, layers, profile, client, network, budget, args.output_mode,
    )
    if args.json:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return 0 if plan["feasible"] and sch.acc >= args.acc_min else 1
    print(f"schedule: {sch.describe(layers)}")
    m = plan.get("measured_cost")
    if m:
        print(f"accuracy={sch.acc:.2%}  mpc_elements={sch.mpc_elements:,}  "
              f"hybrid_layers={sch.hybrid_layers}  rounds={m['rounds']:g}  "
              f"client={m['client_ms']:.1f}ms  server={m['server_ms']/1e3:.2f}s  "
              f"e2e={m['latency_ms']/1e3:.2f}s  online={m['online_bytes']/1e6:.2f}MB  "
              f"memory={m['memory_mb']}  depth_ok={sch.depth_ok}")
        print(f"ckks={plan['p']['name']}  feasible={plan['feasible']}")
        return 0 if plan["feasible"] and sch.acc >= args.acc_min else 1
    print(f"accuracy={sch.acc:.2%}  mpc_elements={sch.mpc_elements:,}  "
          f"hybrid_layers={sch.hybrid_layers}  "
          f"target_rtts={plan['rtts']}  target_time={plan['time_ms']/1000:.2f}s  "
          f"reference_rtts={plan['reference_rtts']}  "
          f"reference_time={plan['reference_time_ms']/1000:.2f}s  "
          f"target_bytes={plan['bytes']/1e6:.2f}MB  "
          f"reference_bytes={plan['reference_bytes']/1e6:.2f}MB  "
          f"depth_ok={sch.depth_ok}")
    print(f"ckks={plan['p']['name']}  "
          f"N={plan['p']['poly_modulus_degree']}  "
          f"security_validated={plan['p']['security_validated']}")
    print(f"packing_slots={plan['packing']['slot_capacity']}  "
          f"packing_estimated={plan['packing']['estimated']}  "
          f"feasible={plan['feasible']}")
    return 0 if plan["feasible"] and sch.acc >= args.acc_min else 1


if __name__ == "__main__":
    raise SystemExit(main())
