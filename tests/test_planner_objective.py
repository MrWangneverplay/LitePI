import argparse
import io
import json
import sys
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
import planner as P
from check_repair_regressions import FakeCost, functions_from


class PlannerObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.layers = [P.Layer("first", 1), P.Layer("second", 100)]
        self.cost = FakeCost()
        self.profile = P.Profile(self.cost, {"01": 0.9, "10": 0.9}, objective="server")

    def test_server_tie_break_ignores_client_quantities(self):
        server = P.constrained(self.layers, self.profile, 0.8)
        client = P.constrained(self.layers, replace(self.profile, objective="client"), 0.8)
        self.assertEqual(server.bits, (0, 1))
        self.assertEqual(client.bits, (1, 0))
        self.assertGreater(server.mpc_elements, client.mpc_elements)
        self.assertEqual(server.time_ms, server.server_ms)

    def test_unknown_or_invalid_costs_remain_infeasible(self):
        for objective, method in (("client", "client_ms"), ("server", "server_ms"),
                                  ("server", "online_bytes"), ("server", "rounds")):
            for value in (None, float("nan"), float("inf"), -1):
                with self.subTest(objective=objective, method=method, value=value), \
                        patch.object(self.cost, method, return_value=value):
                    profile = replace(self.profile, objective=objective)
                    self.assertIsNone(P.constrained(self.layers, profile, 0.8))
                    self.assertFalse(P.plan_result((0, 1), self.layers, profile)["feasible"])

    def test_budgets_and_accuracy_still_filter_before_selection(self):
        for budget in (P.Budget(max_mpc_elements=1), P.Budget(max_rounds=3, max_time_ms=10)):
            chosen = P.constrained(self.layers, self.profile, 0.8, budget)
            expected = (1, 0) if budget.max_mpc_elements is not None else (0, 1)
            self.assertEqual(chosen.bits, expected)
            self.assertTrue(P.plan_result(chosen.bits, self.layers, self.profile,
                                          budget, 0.8)["feasible"])
        for budget in (P.Budget(max_mpc_elements=0), P.Budget(max_rounds=2),
                       P.Budget(max_time_ms=0), P.Budget(max_time_ms=9)):
            self.assertIsNone(P.constrained(self.layers, self.profile, 0.8, budget))
        self.assertIsNone(P.constrained(self.layers, self.profile, 0.95))
        self.assertIsNone(P.constrained(self.layers, replace(self.profile, depth_budget=1), 0.8))
        self.assertIsNone(P.constrained(self.layers, replace(self.profile, max_memory_mb=19), 0.8))

    def test_default_client_and_e2e_are_preserved(self):
        default = replace(self.profile, objective=P.Profile(self.cost).objective)
        self.assertEqual(default.objective, "client")
        self.assertEqual(P.constrained(self.layers, default, 0.8).bits, (1, 0))
        with patch.object(self.cost, "online_latency_ms", return_value=7, create=True):
            profile = replace(self.profile, objective="e2e", rtt_ms=0)
            plan = P.plan_result((1, 0), self.layers, profile, acc_min=0.8)
            self.assertEqual((plan["objective"], plan["objective_ms"]), ("e2e", 7))
            self.assertTrue(plan["feasible"])


class PlannerObjectiveCLITests(unittest.TestCase):
    def main_from(self, role, **bindings):
        return functions_from(
            f"pi_{role}.py", ["main", "_as_dims", "_as_schedule"],
            argparse=argparse, Path=Path, json=json, WIRE_PROFILES={"baseline": None},
            __file__=str(ANALYSIS / f"pi_{role}.py"), **bindings)["main"]

    def run_server(self, plan, flags):
        serve = Mock()
        main = self.main_from("server", serve=serve,
            HybridMLP=lambda dims, seed, schedule: SimpleNamespace(dims=dims, schedule=schedule))
        with patch.object(sys, "argv", ["pi_server.py", "--plan", "fixture.json", *flags]), \
                patch.object(Path, "read_text", return_value=json.dumps(plan)):
            self.assertEqual(main(), 0)
        self.assertEqual(serve.call_args.args[2].schedule, tuple(plan["s"]))

    def test_matching_objective_is_consumed_and_legacy_plans_still_work(self):
        for objective in ("client", "server"):
            self.run_server({"s": [0, 1], "objective": objective},
                            ["--planner-objective", objective])
        for plan in ({"s": [1, 0]}, {"s": [1, 0], "objective": "e2e"},
                     {"s": [1, 0], "objective": "server"}):
            self.run_server(plan, [])
        self.run_server({"s": [1, 0]}, ["--planner-objective", "client"])

    def test_mismatched_plan_is_rejected_before_model_or_server(self):
        serve, model = Mock(), Mock()
        main = self.main_from("server", serve=serve, HybridMLP=model)
        for objective in ("client", "e2e", "invalid", None):
            plan = {"s": [0, 1]}
            if objective is not None:
                plan["objective"] = objective
            with self.subTest(objective=objective), \
                    patch.object(sys, "argv", ["pi_server.py", "--plan", "fixture.json",
                                               "--planner-objective=server"]), \
                    patch.object(Path, "read_text", return_value=json.dumps(plan)), \
                    redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as caught:
                main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("objective", error.getvalue())
        model.assert_not_called()
        serve.assert_not_called()

    def test_no_planner_is_reported_instead_of_silent_noop(self):
        for role in ("client", "server"):
            main = self.main_from(role)
            for objective in ("client", "server", "e2e"):
                with self.subTest(role=role, objective=objective), \
                        patch.object(sys, "argv", [f"pi_{role}.py", "--planner-objective", objective]), \
                        redirect_stderr(io.StringIO()) as error, self.assertRaises(SystemExit) as caught:
                    main()
                self.assertEqual(caught.exception.code, 2)
                self.assertIn("planner" if objective != "e2e" else "invalid choice", error.getvalue())

    def test_default_client_does_not_change_existing_call(self):
        run = Mock(return_value={})
        main = self.main_from("client", load_input=Mock(return_value="input"),
                              run_client=run, _print_summary=Mock())
        with patch.object(sys, "argv", ["pi_client.py"]):
            self.assertEqual(main(), 0)
        run.assert_called_once_with("127.0.0.1", 9101, "input", "seal", 600.0,
                                    output_mode="logits", stream_keys=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
