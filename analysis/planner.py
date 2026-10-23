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
from dataclasses import dataclass


@dataclass(frozen=True)
class Layer:
    name: str
    elements: int          # MPC elements if scheduled ReLU
    linear_depth: int = 1  # HE depth of the preceding linear layer
    p2_depth: int = 1      # extra depth if P2
    sensitivity: float = 0.0  # accuracy drop if switched to P2


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

    def penalty(self, k: int) -> float:
        return self.p2_penalty_coeff * k * (k - 1)


# Fast early-downsampling net: 20,736 ReLU elements in total.
FAST_LAYERS = (
    Layer("conv1", 8192, sensitivity=0.005),
    Layer("conv2", 4096, sensitivity=0.008),
    Layer("conv3", 4096, sensitivity=0.012),
    Layer("conv4", 4096, sensitivity=0.015),
    Layer("fc1",   256, sensitivity=0.014),
)

# 4-conv VGG (64 channels): 164,096 ReLU elements in total.
VGG_LAYERS = (
    Layer("conv1", 65536, sensitivity=0.010),
    Layer("conv2", 65536, sensitivity=0.015),
    Layer("conv3", 16384, sensitivity=0.020),
    Layer("conv4", 16384, sensitivity=0.025),
    Layer("fc1",    256, sensitivity=0.0105),
)


@dataclass(frozen=True)
class Schedule:
    bits: tuple[int, ...]
    acc: float
    mpc_elements: int
    rtts: int
    time_ms: float
    bytes: int
    depth_ok: bool

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
    acc = profile.base_acc - profile.penalty(p2_count)
    time_ms = mpc * profile.per_element_ms + rtts * profile.rtt_ms
    return Schedule(tuple(bits), acc, mpc, rtts, time_ms, nbytes, ok)


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
        if sch.depth_ok and sch.acc >= acc_min:
            out.append(sch)
    return out


def constrained(layers, profile: Profile, acc_min: float):
    feasible = feasible_schedules(layers, profile, acc_min)
    if not feasible:
        return None
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
    ap.add_argument("--arch", choices=["fast", "vgg"], default="fast")
    ap.add_argument("--strategy", choices=["rule", "greedy", "constrained"],
                    default="greedy")
    ap.add_argument("--acc-min", type=float, default=0.82)
    ap.add_argument("--rtt-ms", type=float, default=50.0)
    ap.add_argument("--per-element-ms", type=float, default=12.9)
    ap.add_argument("--pareto", action="store_true")
    args = ap.parse_args()

    layers = FAST_LAYERS if args.arch == "fast" else VGG_LAYERS
    # These are the measured all-ReLU baselines for the two architectures.
    base_acc = 0.8509 if args.arch == "fast" else 0.8855
    profile = Profile(
        rtt_ms=args.rtt_ms,
        per_element_ms=args.per_element_ms,
        base_acc=base_acc,
    )
    print(f"arch={args.arch} acc_min={args.acc_min:.2%} rtt={args.rtt_ms}ms "
          f"per_element={args.per_element_ms}ms depth_budget={profile.depth_budget}")

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
    print(f"schedule: {sch.describe(layers)}")
    print(f"accuracy={sch.acc:.2%}  mpc_elements={sch.mpc_elements:,}  "
          f"hybrid_layers={sch.hybrid_layers}  rtts={sch.rtts}  "
          f"time={sch.time_ms/1000:.2f}s  bytes={sch.bytes/1e6:.2f}MB  "
          f"depth_ok={sch.depth_ok}")
    return 0 if sch.depth_ok and sch.acc >= args.acc_min else 1


if __name__ == "__main__":
    raise SystemExit(main())
