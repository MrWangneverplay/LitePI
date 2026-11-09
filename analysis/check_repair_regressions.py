"""Dependency-free regression checks for the code-repair batch.

Run: python -B analysis/check_repair_regressions.py
No training, HE/SCI backend, dataset, native build or benchmark is run.
AST extraction isolates actual production functions from unavailable optional
imports.  Argmax uses ideal two-party gates, NOT a cryptographic implementation;
these checks establish arithmetic/control-flow correctness only.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import math
import random
import re
import statistics
import sys
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import planner as P
import gen_e9_data as E9
import compute_agreement as A

HERE = Path(__file__).resolve().parent


def functions_from(filename, names, **bindings):
    path = HERE / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in names]
    if {node.name for node in nodes} != set(names):
        raise AssertionError(f"missing production function in {filename}")
    module = ast.Module(body=[ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = dict(bindings)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class IdealGates:
    """Test-only shared gates. Reconstructs secrets; never used by production."""

    def __init__(self):
        self.barrier = threading.Barrier(2, timeout=5)
        self.inputs = {}
        self.outputs = None
        self.rng = random.Random(914)

    def exchange(self, party, operation, args, compute):
        self.inputs[party] = (operation, args)
        self.barrier.wait()
        if party == 0:
            if self.inputs[0][0] != self.inputs[1][0]:
                self.barrier.abort()
                raise AssertionError("parties used different gates")
            self.outputs = compute(self.inputs[0][1], self.inputs[1][1])
        self.barrier.wait()
        result = self.outputs[party]
        self.barrier.wait()
        return result

    def drelu(self, ctx, values, width):
        def compute(a, b):
            self_test_width, left = a
            other_width, right = b
            assert self_test_width == other_width
            mod = 1 << width
            bits = [int((x + y) % mod < mod // 2)
                    for x, y in zip(left, right)]
            random_bits = [self.rng.randrange(2) for _ in bits]
            return random_bits, [x ^ y for x, y in zip(bits, random_bits)]
        return self.exchange(ctx.party, "drelu", (width, values), compute)

    def multiply(self, ctx, x, b_self, b_other, k, self_party):
        def compute(a, b):
            assert a[0:2] == b[0:2]
            sender = (a, b)[self_party]
            receiver = (a, b)[1 - self_party]
            mod = 1 << k
            truth = [(value * (own_bit ^ other_bit)) % mod
                     for value, own_bit, other_bit
                     in zip(sender[2], sender[3], receiver[4])]
            left = [self.rng.randrange(mod) for _ in truth]
            return left, [(value - share) % mod
                          for value, share in zip(truth, left)]
        return self.exchange(ctx.party, "multiply",
                             (k, self_party, x, b_self, b_other), compute)

    def argmax(self, units, split_seed=0, width=14):
        gates = functions_from("secure_argmax.py",
                               ["_quantize", "_select_shared", "run_party_argmax",
                                "combine_index_share"],
                               d_drelu_batch=self.drelu,
                               d_xorbit_mult_batch=self.multiply)
        k, f = 44, 8
        rng = random.Random(split_seed)
        left = [rng.randrange(-(1 << 38), 1 << 38) for _ in units]
        shares = ([v / (2 ** f) for v in left],
                  [(v - s) / (2 ** f) for v, s in zip(units, left)])
        outputs, errors = {}, []

        def party(p):
            try:
                outputs[p] = gates["run_party_argmax"](
                    SimpleNamespace(party=p), shares[p], k, f, width)
            except BaseException as exc:
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=party, args=(p,), daemon=True)
                   for p in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        if any(thread.is_alive() for thread in threads):
            self.barrier.abort()
            raise AssertionError("ideal-gate regression hung")
        if errors:
            raise errors[0]
        return gates["combine_index_share"](outputs[0], outputs[1], k)


class ArgmaxTests(unittest.TestCase):
    def test_lsb_ties_negative_and_all_winners(self):
        cases = [[1, 0], [0, 1], [0, 0], [-1, -2], [-2, -1],
                 [2, 0], [0, 0, 0, 0], [-3, 2, 2, 0], [7]]
        for winner in range(6):
            case = [-896, -256, 64, 704, -512, 384]
            case[winner] += 2560
            cases.append(case)
        for units in cases:
            for seed in range(3):
                with self.subTest(units=units, split=seed):
                    expected = max(range(len(units)), key=units.__getitem__)
                    self.assertEqual(IdealGates().argmax(units, seed), expected)

    def test_signed_difference_window_edges(self):
        for units in ([8191, 0], [0, 8191], [-8191, 0], [0, -8191]):
            with self.subTest(units=units):
                self.assertEqual(IdealGates().argmax(units),
                                 max(range(2), key=units.__getitem__))

    def test_invalid_width(self):
        funcs = functions_from("secure_argmax.py", ["run_party_argmax", "_quantize"])
        for k, f, compare in ((44, 8, 0), (44, 8, 45), (44, -1, 14)):
            with self.assertRaises(ValueError):
                funcs["run_party_argmax"](SimpleNamespace(party=0), [], k, f, compare)


class FakeCost:
    r_h = 1.0
    r_h_scope = "test-only"
    device = "test"
    wire = "test"
    provenance = {"status": "synthetic unit-test costs, not experimental data"}

    def rounds(self, hybrids, mode):
        return 2.0 + hybrids

    def online_bytes(self, *args):
        return 0

    def client_ms(self, hybrids, *args):
        return 5.0 if hybrids else 110.0

    def server_ms(self, arch, bits):
        return 10.0

    def memory_mb(self, hybrids):
        return 20.0


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.layers = [P.Layer("hidden", 4)]
        self.profile = P.Profile(FakeCost(), {"0": 0.9, "1": 0.9}, rtt_ms=0.0)

    def test_budget_filters_before_minimizing(self):
        self.assertEqual(P.constrained(self.layers, self.profile, 0.8).bits, (1,))
        for budget in (P.Budget(max_mpc_elements=0), P.Budget(max_rounds=2)):
            with self.subTest(budget=budget):
                chosen = P.constrained(self.layers, self.profile, 0.8, budget)
                self.assertEqual(chosen.bits, (0,))
                plan = P.plan_result(chosen.bits, self.layers, self.profile, budget, 0.8)
                self.assertTrue(plan["feasible"])
                self.assertEqual([s.bits for s in P.feasible_schedules(
                    self.layers, self.profile, 0.8, budget)], [(0,)])

    def test_time_zero_and_infeasible_floor(self):
        self.assertIsNone(P.constrained(self.layers, self.profile, 0.8,
                                        P.Budget(max_time_ms=0)))
        self.assertEqual(P.constrained(self.layers, self.profile, 0.8,
                                       P.Budget(max_time_ms=5)).bits, (1,))
        self.assertIsNone(P.constrained(self.layers, self.profile, 0.95))
        self.assertFalse(P.plan_result((1,), self.layers, self.profile,
                                        acc_min=0.95)["feasible"])

    def test_invalid_budgets(self):
        for kwargs in ({"max_mpc_elements": -1}, {"max_rounds": -1},
                       {"max_time_ms": float("inf")}, {"max_time_ms": float("nan")}):
            with self.assertRaises(ValueError):
                P.Budget(**kwargs)

    def test_unknown_costs_are_not_feasible(self):
        class Unknown(FakeCost):
            def online_bytes(self, *args):
                return None
        profile = P.Profile(Unknown(), {"0": 0.9, "1": 0.9}, objective="server")
        self.assertIsNone(P.constrained(self.layers, profile, 0.8))
        self.assertFalse(P.plan_result((0,), self.layers, profile)["feasible"])


class E1EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.funcs = functions_from(
            "eval_hybrid_accuracy.py",
            ["prediction_evidence", "prediction_summary", "validate_resume", "build_payload"],
            math=math, SCHEMA_VERSION=2,
            SETUP_KEYS=("mpc_runtime", "keygen_s", "context_upload_s", "context_up_bytes",
                        "sci_setup_up_bytes", "sci_setup_down_bytes", "queries", "sessions"),
            np=SimpleNamespace(mean=statistics.mean, max=max, sum=sum),
            WIRE_PROFILES={"baseline": SimpleNamespace(to_dict=lambda: {"name": "baseline"})},
            POLY_MODULUS_DEGREE=8192, COEFF_MOD_BIT_SIZES=[40, 21, 40],
            GLOBAL_SCALE_BITS=21, F=8, K_RING=44, K_COMPARE=14)
        self.evidence = self.funcs["prediction_evidence"]
        self.identity = {"checkpoint_sha256": "checkpoint-A", "sample_order_sha256": "order-A",
                         "mask_domains": None, "wire": "baseline", "mask_exact": True,
                         "params": {"f": 8}, "backend": "seal"}

    def row(self, label, plain, cipher, index=0):
        return {"index": index, "sample_id": 100 + index,
                "prepared_input_sha256": "input-test", **self.evidence(label, plain, cipher),
                **{key: 0.01 for key in ("wall_s", "forward_s", "mpc_s", "encrypt_s",
                                        "decrypt_s", "reencrypt_s")},
                **{key: 10 for key in ("up_bytes", "down_bytes", "up_wire_bytes", "down_wire_bytes",
                                      "up_sci_bytes", "down_sci_bytes")}}

    def test_wrong_classification_can_have_perfect_fidelity(self):
        row = self.row(0, [0, 2], [0, 2])
        summary = self.funcs["prediction_summary"]([row])
        self.assertEqual(summary["classification_errors"], 1)
        self.assertEqual(summary["prediction_mismatches"], 0)
        self.assertEqual(summary["prediction_agreement"], 1)
        self.assertEqual(summary["accuracy"], 0)

    def test_paired_denominator_and_mixed_predictions(self):
        rows = [self.row(0, [0, 2], [0, 2]), self.row(0, [2, 0], [0, 2], 1),
                self.row(1, [2, 0], [0, 2], 2)]
        summary = self.funcs["prediction_summary"](rows)
        self.assertEqual(summary["classification_errors"], 2)
        self.assertEqual(summary["prediction_mismatches"], 2)
        self.assertEqual(summary["agreement_samples"], 3)
        self.assertAlmostEqual(summary["prediction_agreement"], 1 / 3)
        self.assertEqual(self.evidence(0, [1, 1], [1, 1])["plain_pred"], 0)
        self.assertEqual(rows[0]["plain_margin"], 2)

    def test_legacy_rows_do_not_fabricate_agreement(self):
        summary = self.funcs["prediction_summary"]([{"pred": 1, "label": 0}])
        self.assertIsNone(summary["prediction_mismatches"])
        self.assertIsNone(summary["prediction_agreement"])
        self.assertFalse(summary["agreement_complete"])
        empty = self.funcs["prediction_summary"]([])
        self.assertIsNone(empty["accuracy"])

    def test_nonfinite_logits_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                self.evidence(0, [0, 1], [value, 1])
        with self.assertRaises(ValueError):
            self.evidence(0, [1, 2], [1])

    def test_resume_checks_identity_schema_and_sample_order(self):
        previous = {"schema_version": 2, "run_identity": self.identity,
                    "n_samples": 1, "per_sample": [self.row(0, [2, 0], [2, 0])]}
        validate = self.funcs["validate_resume"]
        validate(previous, self.identity, [100, 101], 2)  # target extension is allowed
        for field in self.identity:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate(previous, {**self.identity, field: "changed"}, [100], 1)
        for bad in ({**previous, "schema_version": 1}, {**previous, "n_samples": 2},
                    {**previous, "per_sample": [{**previous["per_sample"][0], "pred": 1}]}):
            with self.assertRaises(ValueError):
                validate(bad, self.identity, [100], 1)
        with self.assertRaises(ValueError):
            validate(previous, self.identity, [999], 1)
        with self.assertRaises(ValueError):
            validate(previous, self.identity, [100], 0)

    def test_partial_payload_preserves_setup_and_full_parameters(self):
        args = SimpleNamespace(dataset="mnist", schedule=(1, 1), seed=73,
                               sample_seed=0, backend="seal", wire="baseline", mask_exact=True)
        state = {"run_identity": self.identity, "keygen_s": 1.0, "context_upload_s": 2.0,
                 "context_up_bytes": 1234, "sci_setup_up_bytes": 41,
                 "sci_setup_down_bytes": 42, "queries": 1, "sessions": 1}
        payload = self.funcs["build_payload"](
            args, [self.row(0, [0, 2], [0, 2])], 2,
            SimpleNamespace(checkpoint="test-checkpoint"), 10000, state, True)
        self.assertTrue(payload["partial"])
        self.assertEqual(payload["schema_version"], 2)
        self.assertEqual(payload["setup"]["sci_setup_down_bytes"], 42)
        self.assertEqual(payload["params"]["k_compare"], 14)
        self.assertEqual(payload["classification_errors"], 1)
        self.assertEqual(payload["prediction_mismatches"], 0)
        self.assertEqual(payload["sampling"]["sample_seed"], 0)
        args.backend = "stub"
        simulated = self.funcs["build_payload"](
            args, [], 2, SimpleNamespace(checkpoint="test"), 100, state, True)
        self.assertFalse(simulated["measured"])
        self.assertEqual(simulated["evidence_kind"], "simulation")


class E9GroupingTests(unittest.TestCase):
    @staticmethod
    def point(schedule="00", device="pi5", loss=0, p95=100, **overrides):
        row = {"dataset": "mnist", "schedule": schedule, "device": device,
               "topology": "client-egress", "iface": "eth0", "host": "test-server",
               "port": 9101, "backend": "seal", "output_mode": "logits",
               "delay_mult": 1.0, "bandwidth_mbps": 50, "wire": {"name": "baseline"},
               "rtt_ms": 50, "loss_pct": loss, "jitter_ms": 0,
               "wall_mean_ms": p95, "wall_p95_ms": p95, "measured": True,
               "deployment_compliant": True, "protocol_failures": []}
        return {**row, **overrides}

    def test_baselines_cannot_cross_devices_bandwidth_or_topology(self):
        rows = []
        for device, bw, mult, base in (("pi5", 50, 1, 100), ("wsl", 50, 2, 1000),
                                       ("pi5", 10, 1, 400)):
            rows.extend([self.point(device=device, p95=base, bandwidth_mbps=bw, delay_mult=mult),
                         self.point(device=device, p95=2 * base, loss=1,
                                    bandwidth_mbps=bw, delay_mult=mult)])
        points = E9.amplification(rows)
        self.assertEqual(len(points), 6)
        self.assertEqual([p["p95_x"] for p in points if p["loss_pct"] == 1], [2, 2, 2])
        self.assertEqual(len({E9.comparison_context(p) for p in points}), 3)

    def test_rankings_are_per_context_and_only_observed_candidates(self):
        points = [self.point("00", "pi5", p95=500), self.point("11", "pi5", p95=600),
                  self.point("00", "wsl", p95=1000), self.point("11", "wsl", p95=10)]
        choices = E9.best_schedules(points, 0.97)
        self.assertEqual({choice["device"]: choice["schedule"] for choice in choices},
                         {"pi5": "00", "wsl": "11"})
        self.assertTrue(all(choice["candidate_schedules"] == ["00", "11"] for choice in choices))

    def test_missing_metadata_is_not_a_fair_ranking(self):
        row = self.point(host=None, iface=None)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(E9.best_schedules([row], 0.97), [])
        self.assertFalse(E9.comparison_metadata_complete(row))
        self.assertIsNone(E9.comparison_context(row)[3])

    def test_bad_baseline_and_simulation_are_reported(self):
        rows = [self.point(p95=0), self.point(loss=1, p95=100)]
        skipped = []
        with redirect_stderr(io.StringIO()):
            self.assertEqual(E9.amplification(rows, skipped), [])
        self.assertEqual(len(skipped), 2)
        for override in ({"backend": "stub"}, {"deployment_compliant": False},
                         {"protocol_failures": ["bad"]}, {"measured": False},
                         {"wall_p95_ms": float("nan")}):
            with redirect_stderr(io.StringIO()):
                self.assertEqual(E9.amplification([self.point(**override)]), [])

    def test_duplicate_context_is_not_silently_overwritten(self):
        with self.assertRaises(ValueError):
            E9.amplification([self.point(), self.point()])
        left = self.point(wire={"name": "baseline", "bits": 21})
        right = self.point(wire={"bits": 21, "name": "baseline"})
        self.assertEqual(E9.comparison_context(left), E9.comparison_context(right))


class PlannerCLITests(unittest.TestCase):
    def test_cli_zero_time_is_a_real_budget(self):
        fake_module = SimpleNamespace(CostModel=SimpleNamespace(load=lambda _: FakeCost()),
                                      fit=lambda *args: FakeCost())
        argv = ["planner.py", "--cost-model", "unit-test.json", "--arch", "mnist",
                "--acc-min", "0.97", "--max-time-s", "0", "--json"]
        with patch.dict(sys.modules, {"cost_model": fake_module}), patch.object(sys, "argv", argv), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(P.main(), 1)


def v2_record(tag="11", seed=73, dataset="mnist"):
    """In-memory synthetic schema fixture; never written as experimental data."""
    case = E1EvidenceTests()
    case.setUp()
    runtime = "litepi_sci" if "1" in tag else "none"
    identity = {
        "schema_version": 2, "dataset": dataset, "schedule": tag, "training_seed": seed,
        "sample_seed": 0, "sampler": "numpy.default_rng.permutation", "numpy_version": "test",
        "python_runtime": "test", "test_set_size": 10000,
        "checkpoint_sha256": hashlib.sha256(f"fixture-checkpoint-{tag}-{seed}".encode()).hexdigest(),
        "test_set_sha256": "a" * 64, "sample_order_sha256": "b" * 64,
        "source_sha256": {"test.py": "c" * 64}, "native_module_sha256": {"seal4": "d" * 64},
        "sci_extension_sha256": "e" * 64, "backend": "seal", "mask_exact": True,
        "mask_domains": {"0": {"bound": seed}} if "1" in tag else None,
        "wire": {"name": "baseline"},
        "params": {"poly_modulus_degree": 8192, "coeff_mod_bit_sizes": [40, 21, 40],
                   "global_scale_bits": 21, "f": 8, "k_ring": 44, "k_compare": 14},
    }
    args = SimpleNamespace(dataset=dataset, schedule=tuple(map(int, tag)), seed=seed,
                           sample_seed=0, backend="seal", wire="baseline", mask_exact=True)
    row = {**case.row(0, [0, 2], [0, 2]), "prepared_input_sha256": "f" * 64,
           "mpc_runtime": runtime}
    state = {"run_identity": identity, "mpc_runtime": runtime, "queries": 1, "sessions": 1}
    payload = case.funcs["build_payload"](args, [row], 1,
                                          SimpleNamespace(checkpoint="fixture-only"), 10000, state, False)
    return payload


class ConsumerMigrationTests(unittest.TestCase):
    def setUp(self):
        self.cost = functions_from(
            "cost_model.py", ["resolve_e1_suffix", "validate_e1_records", "load_e1_series",
                              "_ot_bytes_per_element"],
            re=re, math=math, json=json, Path=Path, ROOT=HERE.parent,
            NONLINEAR_DIMS={"mnist": (256, 64), "cifar10": (1200, 128), "uci_har": (756, 128)},
            paired_rows=A.paired_rows)
        self.adaptive = functions_from(
            "adaptive_eval.py", ["aggregate_schedules", "model_added_rtt"],
            st=statistics, math=math, TAGS=("00", "01", "10", "11"), SEEDS=(73, 74, 75, 76, 77))

    def test_agreement_consumes_saved_pairs_not_training_seed_order(self):
        record = v2_record(seed=77)
        before = copy.deepcopy(record)
        kept = A.paired_rows(record)
        self.assertEqual(record, before)
        self.assertEqual(kept[0]["sample_id"], 100)
        out = A._agreement(record, kept, HERE.parent / "fixture" / "seed77.json", HERE.parent,
                           "mnist", "11", 77, "saved_paired_v2")
        self.assertEqual(out["agreement"], 1.0)
        self.assertEqual(out["ciphertext_accuracy"], 0.0)
        self.assertFalse(out["source_verified"])
        self.assertEqual(out["run_identity"]["sample_seed"], 0)

    def test_agreement_refuses_tampered_predictions_and_hashes(self):
        for change in ("pred", "sample_id", "hash"):
            record = v2_record()
            if change == "pred":
                record["per_sample"][0]["pred"] = 0
            elif change == "sample_id":
                record["per_sample"][0]["sample_id"] = 10000
            else:
                record["run_identity"]["checkpoint_sha256"] = "not-a-hash"
            with self.assertRaises(ValueError):
                A.paired_rows(record)

    def test_seed_specific_calibration_does_not_split_a_valid_series(self):
        left, right = v2_record(seed=73), v2_record(seed=74)
        args = ("mnist", "11", "hybrid_s11_acc_seal_maskv3_v2", "saved_paired_v2")
        self.assertEqual(A._run_series(left, *args), A._run_series(right, *args))
        right["run_identity"]["sample_seed"] = 1
        self.assertNotEqual(A._run_series(left, *args), A._run_series(right, *args))

    def test_cost_accepts_complete_consistent_series(self):
        records = [v2_record(tag, seed) for tag in ("00", "01", "10", "11") for seed in (73, 74)]
        accepted, rows, provenance = self.cost["validate_e1_records"](records)
        self.assertEqual(len(accepted), 8)
        self.assertEqual(len(rows), 8)
        self.assertEqual(provenance["status"], "schema_v2")
        self.assertEqual(rows[0]["elements"], 0)
        self.assertEqual(rows[-1]["elements"], 320)

    def test_cost_excludes_partial_stub_and_errors(self):
        records = [v2_record(), {**v2_record(), "partial": True},
                   {**v2_record(), "backend": "stub"}, {**v2_record(), "error": "failed"}]
        accepted, _, provenance = self.cost["validate_e1_records"](records)
        self.assertEqual(len(accepted), 1)
        self.assertEqual(len(provenance["excluded_records"]), 3)

    def test_cost_rejects_protocol_or_sampling_mixing(self):
        for key, replacement in (("source_sha256", {"test.py": "0" * 64}),
                                 ("native_module_sha256", {"seal4": "0" * 64}),
                                 ("sample_order_sha256", "0" * 64),
                                 ("numpy_version", "changed")):
            left, right = v2_record(seed=73), v2_record(seed=74)
            right["run_identity"][key] = replacement
            if key == "sample_order_sha256":
                right["sampling"][key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.cost["validate_e1_records"]([left, right])

    def test_cost_rejects_false_saved_prediction(self):
        record = v2_record()
        record["per_sample"][0]["plain_pred"] = 0
        with self.assertRaises(ValueError):
            self.cost["validate_e1_records"]([record])

    def test_cost_sci_bytes_are_sample_weighted_and_missing_stays_unknown(self):
        rows = [{"elements": 4, "up_sci_bytes": 8, "down_sci_bytes": 4},
                {"elements": 2, "up_sci_bytes": 2, "down_sci_bytes": 4},
                {"elements": 0, "up_sci_bytes": 0, "down_sci_bytes": 0}]
        value, _ = self.cost["_ot_bytes_per_element"](rows)
        self.assertEqual(value, 3.0)
        self.assertIsNone(self.cost["_ot_bytes_per_element"](
            [{"elements": 2, "up_sci_bytes": None, "down_sci_bytes": 4}])[0])

    def test_precise_suffix_and_no_legacy_fallback(self):
        suffix = self.cost["resolve_e1_suffix"]
        self.assertEqual(suffix(), "_seal_maskv3_v2")
        self.assertEqual(suffix("compact"), "_seal_wirecompact_maskv3_v2")
        self.assertEqual(suffix("baseline", "_seal_v2"), "_seal_v2")
        with self.assertRaises(ValueError):
            suffix("baseline", "_seal*")
        with self.assertRaisesRegex(FileNotFoundError, "missingnewseries"):
            self.cost["load_e1_series"](HERE / "__unit_missing_series__")

    def test_adaptive_requires_five_seeds_and_identical_prefixes(self):
        records = [v2_record(tag, seed) for tag in ("00", "01", "10", "11")
                   for seed in (73, 74, 75, 76, 77)]
        _, _, provenance = self.cost["validate_e1_records"](records)
        provenance["e1_suffix"] = "_seal_maskv3_v2"
        table = self.adaptive["aggregate_schedules"](records, "mnist", provenance)
        self.assertEqual(len(table), 4)
        self.assertEqual(table["11"]["n_records"], 5)
        self.assertEqual(table["11"]["paired_prefix_samples"], 1)
        self.assertFalse(table["11"]["measured"])
        with self.assertRaises(ValueError):
            self.adaptive["aggregate_schedules"](records[:-1], "mnist", provenance)
        changed = copy.deepcopy(records)
        changed[-1]["per_sample"][0]["sample_id"] = 101
        with self.assertRaises(ValueError):
            self.adaptive["aggregate_schedules"](changed, "mnist", provenance)

    def test_adaptive_added_rtt_is_model_and_counts_sci_rounds(self):
        model = self.adaptive["model_added_rtt"]
        self.assertAlmostEqual(model(1.0, 2, 50.0, 6.0, "end-to-end"), 1.7)
        for scope in ("control-plane", "protocol-target", "override"):
            with self.assertRaises(ValueError):
                model(1.0, 2, 50.0, 1.0, scope)


class SessionAccountingTests(unittest.TestCase):
    def run_failed_setup(self, failure):
        # Pure mocks: no socket connection, key generation or MPC is performed.
        class Connection:
            def settimeout(self, timeout):
                pass
            def close(self):
                pass
            def getsockname(self):
                return ("127.0.0.1", 1)

        class Thread:
            def __init__(self, **kwargs):
                pass
            def start(self):
                pass
            def join(self, **kwargs):
                pass
            def is_alive(self):
                return False

        class Wire:
            def __init__(self, *args, **kwargs):
                self.cc = SimpleNamespace(sent=0, recvd=0)
                self.ready_count = 0
            def recv_json(self, kind):
                return {"input_dim": 2, "sci_port": 7 if failure == "sci" else None}
            def send_json(self, *args):
                self.cc.sent += 5
            def send(self, *args):
                self.cc.sent += 17
            def recv(self, kind):
                self.ready_count += 1
                if self.ready_count == 2:
                    raise TimeoutError("test-only context upload failure")
                return b"ok"

        def new_context():
            if failure == "keygen":
                raise RuntimeError("test-only keygen failure")
            return object()

        def sci_constructor(*args):
            raise RuntimeError("test-only SCI setup failure")

        clock = iter(range(100))
        backend = SimpleNamespace(name="seal", new_context=new_context)
        functions = functions_from(
            "eval_hybrid_accuracy.py", ["session_eval"],
            make_backend=lambda _: backend, WIRE_PROFILES={"baseline": object()},
            socket=SimpleNamespace(create_server=lambda *a: Connection(),
                                   create_connection=lambda *a, **kw: Connection()),
            threading=SimpleNamespace(Thread=Thread), Wire=Wire,
            adopt_wire=lambda *args: None, SCI_AVAILABLE=True,
            SciReLUSession=sci_constructor, sci_bytes=lambda _: 0,
            upload_context=lambda *args: b"context", ProtocolError=RuntimeError,
            time=SimpleNamespace(perf_counter=lambda: float(next(clock))))
        state = {}
        with self.assertRaises((RuntimeError, TimeoutError)):
            functions["session_eval"](backend, SimpleNamespace(schedule=(0, 0), input_dim=2),
                                      [], "mnist", [], 1, 1, None, state, lambda: None)
        return state

    def test_paid_keygen_survives_context_upload_failure(self):
        state = self.run_failed_setup("upload")
        self.assertEqual(state["sessions"], 1)
        self.assertGreater(state["keygen_s"], 0)
        self.assertGreater(state["context_upload_s"], 0)
        self.assertEqual(state["context_up_bytes"], 17)

    def test_failed_keygen_attempt_is_counted_immediately(self):
        state = self.run_failed_setup("keygen")
        self.assertGreater(state["keygen_s"], 0)
        self.assertNotIn("context_upload_s", state)

    def test_sci_failure_keeps_unknown_server_setup_bytes_explicit(self):
        state = self.run_failed_setup("sci")
        self.assertEqual(state["setup_pending_server_sci_sessions"], 1)
        self.assertGreater(state["sci_setup_s"], 0)
        self.assertNotIn("sci_setup_down_bytes", state)


class NativeProvenanceTests(unittest.TestCase):
    def test_unused_tenseal_import_cannot_change_a_seal_run_identity(self):
        seal = SimpleNamespace(__name__="litepi_seal", __file__="/test/litepi_seal.so")
        sci = SimpleNamespace(__name__="litepi_sci", __file__="/test/litepi_sci.so")
        fake_sys = SimpleNamespace(modules={"litepi_sci": sci})
        fn = functions_from("eval_hybrid_accuracy.py", ["native_identity"],
                            sys=fake_sys, EXTENSION_SUFFIXES=[".so"], Path=Path,
                            file_sha256=lambda path: hashlib.sha256(str(path).encode()).hexdigest())
        backend = SimpleNamespace(seal=seal)
        before = fn["native_identity"](backend)
        fake_sys.modules["_tenseal_cpp"] = SimpleNamespace(__file__="/test/unrelated.so")
        after = fn["native_identity"](backend)
        self.assertEqual(before, after)
        self.assertEqual(set(after), {"litepi_seal", "litepi_sci"})

    def test_selected_tenseal_extension_is_bound_to_its_backend(self):
        cpp = SimpleNamespace(__name__="_tenseal_cpp", __file__="/test/tenseal.so")
        package = SimpleNamespace(__name__="tenseal", __file__="/test/tenseal.py", _ts_cpp=cpp)
        fn = functions_from("eval_hybrid_accuracy.py", ["native_identity"],
                            sys=SimpleNamespace(modules={}), EXTENSION_SUFFIXES=[".so"], Path=Path,
                            file_sha256=lambda path: "a" * 64)
        self.assertEqual(fn["native_identity"](SimpleNamespace(ts=package)), {"_tenseal_cpp": "a" * 64})


if __name__ == "__main__":
    unittest.main(verbosity=2)
