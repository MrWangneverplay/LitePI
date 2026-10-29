"""M1–M3 mask/session state machine (protocol §8) and the M2 domain rule.

M1  fresh per (query, layer):
    r_k^(q) = PRG(seed_session, q || k), SHAKE-256 in counter mode.  The seed
    comes from the OS CSPRNG on the server and is never sent to the client.
M2  domain alignment (rule v2, see params_pi.py for the derivation):
    r_k is uniform on the fixed-point grid 2^-F inside [-A_k, A_k) with
    A_k = 2^sigma_k * B_k, B_k a power-of-two bound on |z_k|.  The client's
    view z + r is then within statistical distance 2^-(sigma_k+1) per element
    of the simulator's uniform draw (protocol §11, row ①_k).
M3  resynchronisation:
    the query counter is strictly monotonic; a gap, a replay or an
    unfinished query closes the session, which must then be re-keyed.

Rule v1 (the earlier export: observed min/max + 5 %) is still loadable for
regression and for the E10a "narrow domain" control, but it is *not* a valid
M2 domain: with A_k ~ B_k the per-element distance is ~1/2, which is the
weak statistical hiding that E10a measured.

    python mask_session.py --report datas/mask_domains/mnist_s11_seed73.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from params_pi import (
    F,
    MASK_BOUND_MARGIN,
    MASK_MAX_BITS,
    MASK_MIN_SIGMA,
    MASK_SIGMA_TARGET,
)

_DOMAIN = b"litepi-mask-prg-v2\x00"
RULE_V1 = "M2-v1-observed-range"
RULE_V2 = "M2-v2-statdist"


class MaskSessionError(RuntimeError):
    """The session cannot continue without a fresh Setup."""


def power_of_two_bound(max_abs: float, margin: float = MASK_BOUND_MARGIN) -> float:
    """Smallest power of two >= margin * max_abs (and >= 2^-F)."""
    if not math.isfinite(max_abs) or max_abs < 0:
        raise ValueError(f"invalid activation bound {max_abs!r}")
    target = max(float(max_abs) * float(margin), 2.0 ** -F)
    return 2.0 ** math.ceil(math.log2(target))


@dataclass(frozen=True)
class MaskDomain:
    """Uniform mask domain [lo, hi) on the 2^-F grid.

    ``activation_bound`` (B_k) and ``sigma`` are set for rule-v2 domains; a
    bare ``MaskDomain(lo, hi)`` is still accepted (unit tests, legacy files)
    and simply has no statistical-distance guarantee attached.
    """

    lo: float
    hi: float
    activation_bound: float | None = None
    sigma: float | None = None
    rule: str = "explicit"
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def __post_init__(self) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("mask-domain bounds must be finite")
        if not (self.lo < self.hi):
            raise ValueError(f"invalid mask domain [{self.lo}, {self.hi})")
        if max(abs(self.lo), abs(self.hi)) > 2.0 ** MASK_MAX_BITS:
            raise ValueError(
                f"mask domain [{self.lo}, {self.hi}) exceeds 2^{MASK_MAX_BITS}; "
                "the CKKS decode of z+r would lose fixed-point precision"
            )

    # -------------------------------------------------------------- rule v2
    @classmethod
    def for_bound(cls, activation_bound: float,
                  sigma_target: float = MASK_SIGMA_TARGET,
                  max_bits: int = MASK_MAX_BITS,
                  min_sigma: float = MASK_MIN_SIGMA,
                  **meta) -> "MaskDomain":
        """Rule v2: A = 2^sigma * B with sigma = min(target, max_bits - log2 B)."""
        bound = float(activation_bound)
        if not (bound > 0 and math.isfinite(bound)):
            raise ValueError(f"activation bound must be positive, got {bound}")
        log_b = math.log2(bound)
        sigma = min(float(sigma_target), float(max_bits) - log_b)
        if sigma < min_sigma:
            raise ValueError(
                f"B_k=2^{log_b:.1f} leaves only sigma={sigma:.1f} < {min_sigma} "
                f"under the 2^{max_bits} precision cap; rescale the layer or "
                "raise the CKKS decode precision"
            )
        half = 2.0 ** (log_b + sigma)
        return cls(-half, half, activation_bound=bound, sigma=sigma,
                   rule=RULE_V2, meta=dict(meta))

    @classmethod
    def from_observed(cls, lo_obs: float, hi_obs: float, **kw) -> "MaskDomain":
        bound = power_of_two_bound(max(abs(lo_obs), abs(hi_obs)))
        return cls.for_bound(bound, **kw)

    # ----------------------------------------------------------- properties
    @property
    def half_width(self) -> float:
        return 0.5 * (self.hi - self.lo)

    @property
    def log2_half_width(self) -> float:
        return math.log2(self.half_width)

    def statistical_distance(self, z_abs: float | None = None) -> float | None:
        """Per-element SD(z + U[lo,hi), U[lo,hi)) <= |z| / (hi - lo).

        Uses B_k when ``z_abs`` is omitted; None for domains without B_k.
        """
        z = self.activation_bound if z_abs is None else float(z_abs)
        if z is None:
            return None
        return min(1.0, z / (self.hi - self.lo))

    def to_dict(self) -> dict:
        out = {"lo": self.lo, "hi": self.hi, "rule": self.rule}
        if self.activation_bound is not None:
            out["activation_bound"] = self.activation_bound
            out["sigma"] = self.sigma
            out["log2_half_width"] = self.log2_half_width
            out["sd_per_element"] = self.statistical_distance()
        out.update(self.meta)
        return out

    @classmethod
    def from_mapping(cls, value: dict, rule: str = "v2") -> "MaskDomain":
        """Parse one layer entry of an export file.

        rule="v2"  (default) — protocol M2.  v2 entries are taken as stored;
                   v1 entries (observed min/max) are upgraded with
                   ``from_observed`` and flagged.
        rule="v1"  — reproduce the old narrow domain exactly (E10a control,
                   regression only; never use for a protocol run).
        """
        if rule not in ("v1", "v2"):
            raise ValueError(f"unknown mask rule {rule!r}")
        is_v2 = value.get("rule") == RULE_V2 or "activation_bound" in value
        if rule == "v1":
            if "lo" in value and not is_v2:
                return cls(float(value["lo"]), float(value["hi"]), rule=RULE_V1)
            # v2 file: rebuild the old rule from the observed range.
            lo, hi = float(value["min"]), float(value["max"])
            margin = max((hi - lo) * 0.05, 1e-6)
            return cls(lo - margin, hi + margin, rule=RULE_V1)
        if is_v2:
            return cls(float(value["lo"]), float(value["hi"]),
                       activation_bound=float(value["activation_bound"]),
                       sigma=float(value["sigma"]), rule=RULE_V2)
        lo_obs = float(value.get("min", value.get("lo")))
        hi_obs = float(value.get("max", value.get("hi")))
        return cls.from_observed(lo_obs, hi_obs, upgraded_from=RULE_V1)


DEFAULT_DOMAIN = MaskDomain(-(2.0 ** MASK_MAX_BITS), 2.0 ** MASK_MAX_BITS,
                            rule="default-max-width")


def _grid_draw(stream: bytes, size: int, domain: MaskDomain) -> list[float]:
    """Uniform values on the 2^-F grid in [lo, hi), from 16 PRG bytes each.

    ``u mod span`` over a 128-bit u has bias < span / 2^128 <= 2^-75 for the
    largest domain (span <= 2^(MAX_BITS+1+F)), i.e. negligible.  Every output
    is k / 2^F with |k| < 2^51, hence exact in a double, so the server share
    -r quantises to the ring without rounding.
    """
    scale = 1 << F
    lo_int = math.floor(domain.lo * scale)
    span = math.ceil(domain.hi * scale) - lo_int
    out: list[float] = []
    for i in range(size):
        u = int.from_bytes(stream[16 * i:16 * i + 16], "big")
        out.append((lo_int + (u % span)) / scale)
    return out


class MaskSession:
    """Server-only PRG session with explicit M3 resynchronisation."""

    def __init__(self, seed: bytes | None = None,
                 domains: dict[int, MaskDomain] | None = None,
                 default_domain: MaskDomain = DEFAULT_DOMAIN) -> None:
        self._seed = secrets.token_bytes(32) if seed is None else bytes(seed)
        if len(self._seed) < 16:
            raise ValueError("session seed must contain at least 16 bytes")
        self._domains = dict(domains or {})
        self._default_domain = default_domain
        self._next_query = 0
        self._active_query: int | None = None
        self._closed = False

    @property
    def next_query(self) -> int:
        return self._next_query

    @property
    def aligned_layers(self) -> tuple[int, ...]:
        return tuple(sorted(self._domains))

    def domain(self, layer_id: int) -> MaskDomain:
        return self._domains.get(int(layer_id), self._default_domain)

    def covers(self, layer_ids) -> bool:
        """True iff every layer has a rule-v2 (statistically bounded) domain."""
        return all(
            int(layer_id) in self._domains
            and self._domains[int(layer_id)].rule == RULE_V2
            for layer_id in layer_ids
        )

    def begin_query(self, query_id: int) -> None:
        if self._closed:
            raise MaskSessionError("session is closed; run Setup again")
        if query_id != self._next_query:
            self.abort()
            raise MaskSessionError(
                f"query counter mismatch: expected {self._next_query}, got {query_id}"
            )
        if self._active_query is not None:
            self.abort()
            raise MaskSessionError("previous query was not finalized")
        self._active_query = query_id

    def end_query(self, query_id: int) -> None:
        if self._active_query != query_id:
            self.abort()
            raise MaskSessionError("query finalization mismatch")
        self._active_query = None
        self._next_query += 1

    def draw(self, query_id: int, layer_id: int, size: int,
             domain: MaskDomain | None = None) -> list[float]:
        """Fresh mask for (query, layer); ``domain`` overrides the layer's
        stored domain (used when the ciphertext's remaining modulus cannot
        hold the full-width mask, see pi_server.fit_mask_domain)."""
        if size <= 0:
            raise ValueError("mask size must be positive")
        if self._active_query != query_id:
            raise MaskSessionError("mask requested outside its active query")
        domain = domain or self.domain(layer_id)
        material = (
            _DOMAIN
            + self._seed
            + int(query_id).to_bytes(8, "big")
            + int(layer_id).to_bytes(8, "big")
        )
        stream = hashlib.shake_256(material).digest(16 * size)
        return _grid_draw(stream, size, domain)

    def abort(self) -> None:
        self._closed = True
        self._active_query = None


def load_mask_domains(path: Path, output_layer_id: int | None = None,
                      expected_dataset: str | None = None,
                      expected_schedule=None,
                      expected_seed: int | None = None,
                      expected_dims=None,
                      rule: str = "v2",
                      ) -> dict[int, MaskDomain]:
    """Load per-layer M2 domains from the training-statistics export.

    ``rule="v2"`` (default) returns protocol domains; a v1 file is upgraded
    in memory from its observed min/max (re-export to make it permanent).
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if expected_dataset is not None and obj.get("dataset") != expected_dataset:
        raise ValueError(
            f"{path} dataset={obj.get('dataset')!r}, expected {expected_dataset!r}"
        )
    if expected_schedule is not None:
        wanted = [int(v) for v in expected_schedule]
        if obj.get("schedule") != wanted:
            raise ValueError(
                f"{path} schedule={obj.get('schedule')!r}, expected {wanted}"
            )
    if expected_seed is not None and int(obj.get("seed", -1)) != expected_seed:
        raise ValueError(
            f"{path} seed={obj.get('seed')!r}, expected {expected_seed}"
        )
    if expected_dims is not None:
        wanted = [int(v) for v in expected_dims]
        if obj.get("nonlinear_dims") != wanted:
            raise ValueError(
                f"{path} nonlinear_dims={obj.get('nonlinear_dims')!r}, "
                f"expected {wanted}"
            )
    layers = obj.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError(f"{path} has no non-empty 'layers' object")
    if rule == "v2" and obj.get("rule_version") != RULE_V2:
        warnings.warn(
            f"{path} uses the narrow v1 mask rule; upgrading to {RULE_V2} in "
            "memory (re-run export_mask_domains.py to persist it)",
            stacklevel=2,
        )
    out: dict[int, MaskDomain] = {}
    for key, value in layers.items():
        out[int(key)] = MaskDomain.from_mapping(value, rule=rule)
    if "output" in obj and output_layer_id is not None:
        out[int(output_layer_id)] = MaskDomain.from_mapping(obj["output"], rule=rule)
    return out


def mask_security_report(domains: dict[int, MaskDomain],
                         dims: dict[int, int],
                         lambdas=(10, 20, 30, 40),
                         queries: int | None = None) -> dict:
    """Statistical-distance accounting for protocol §11 row ①_k.

    Per element:  delta_k = B_k / (2 A_k) = 2^-(sigma_k+1)
    Per query:    delta_q = sum_k n_k * delta_k        (hybrid layers only)
    Q queries:    delta_Q <= Q * delta_q               (masks are independent)
    Q_max(lam):   largest Q with Q * delta_q <= 2^-lam
    """
    rows = []
    per_query = 0.0
    for layer_id, n in sorted(dims.items()):
        dom = domains.get(int(layer_id))
        if dom is None or dom.activation_bound is None:
            rows.append({"layer": int(layer_id), "n": int(n),
                         "rule": None if dom is None else dom.rule,
                         "bounded": False})
            per_query = float("inf")
            continue
        sd = dom.statistical_distance()
        per_query += n * sd
        rows.append({
            "layer": int(layer_id), "n": int(n), "rule": dom.rule,
            "B_k": dom.activation_bound,
            "log2_B_k": math.log2(dom.activation_bound),
            "log2_A_k": dom.log2_half_width,
            "sigma_k": dom.sigma,
            "sd_per_element": sd,
            "log2_sd_per_element": math.log2(sd),
            "bounded": True,
        })
    report = {
        "rule": RULE_V2,
        "layers": rows,
        "sd_per_query": per_query,
        "log2_sd_per_query": (math.log2(per_query)
                              if 0 < per_query < float("inf") else None),
        "q_max": {},
        "assumption": "|z_k| <= B_k elementwise; beyond B_k the element's "
                      "distance is |z|/(2A_k), still <= 2^-(sigma_k+1)*|z|/B_k",
    }
    for lam in lambdas:
        if per_query in (0.0, float("inf")):
            report["q_max"][str(lam)] = None
        else:
            report["q_max"][str(lam)] = int(math.floor(2.0 ** -lam / per_query))
    if queries is not None and per_query < float("inf"):
        total = queries * per_query
        report["queries"] = int(queries)
        report["sd_total"] = total
        report["log2_sd_total"] = math.log2(total) if total > 0 else None
    return report


def assert_mask_session_safe(session: MaskSession,
                             tuples: int = 10_000) -> None:
    """Unit-test helper: M1 uniqueness and M3 rejection."""
    seen: set[bytes] = set()
    for q in range(tuples):
        session.begin_query(q)
        mask = session.draw(q, 0, 4)
        key = b"".join(float(v).hex().encode() for v in mask)
        if key in seen:
            raise AssertionError(f"duplicate mask at query {q}")
        seen.add(key)
        session.end_query(q)
    try:
        session.begin_query(tuples + 1)
    except MaskSessionError:
        pass
    else:
        raise AssertionError("out-of-order query was not rejected")


def main() -> int:
    ap = argparse.ArgumentParser(description="M2 mask-domain security report")
    ap.add_argument("--report", type=Path, required=True,
                    help="mask-domain JSON from export_mask_domains.py")
    ap.add_argument("--queries", type=int, default=None,
                    help="also report the bound after this many queries")
    ap.add_argument("--schedule", default=None,
                    help="hybrid bits to account for (default: file schedule)")
    ap.add_argument("--upgrade", action="store_true",
                    help="rewrite a v1 file in place as rule v2 (keeps the "
                         "observed statistics; no retraining or torch needed)")
    args = ap.parse_args()
    obj = json.loads(args.report.read_text(encoding="utf-8"))
    if args.upgrade and obj.get("rule_version") != RULE_V2:
        def upgraded(entry: dict) -> dict:
            keep = {k: v for k, v in entry.items()
                    if k not in ("lo", "hi", "margin")}
            dom = MaskDomain.from_observed(float(entry["min"]), float(entry["max"]))
            keep["max_abs"] = max(abs(entry["min"]), abs(entry["max"]))
            keep.update(dom.to_dict())
            return keep
        obj["layers"] = {k: upgraded(v) for k, v in obj["layers"].items()}
        if "output" in obj:
            obj["output"] = upgraded(obj["output"])
        obj["rule_version"] = RULE_V2
        obj["rule"] = ("M2 v2 (upgraded from v1 observed range): B_k = pow2 >= "
                       "1.25*max|z_k|; A_k = 2^sigma_k*B_k; sigma_k = "
                       "min(40, 42 - log2 B_k)")
        args.report.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
                               encoding="utf-8")
        print(f"upgraded {args.report} -> {RULE_V2}")
    domains = load_mask_domains(args.report, output_layer_id=None)
    bits = [int(b) for b in (args.schedule or "".join(
        str(b) for b in obj.get("schedule", [1] * len(domains))))]
    dims_all = obj.get("nonlinear_dims") or [None] * len(bits)
    dims = {i: int(n) for i, (n, b) in enumerate(zip(dims_all, bits)) if b and n}
    rep = mask_security_report(domains, dims, queries=args.queries)
    print(json.dumps(rep, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
