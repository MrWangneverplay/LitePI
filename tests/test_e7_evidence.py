"""E7 证据消费者的轻量回归：仅用标准库和临时 JSON，不运行实验。"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import gen_e7_table8 as table8


class EvidenceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "results" / "raw_results" / "E7"

    def record(self, **changes):
        return {"measured": True, "status": "ok", "config": {"seed": 73},
                "git_sha": "test-sha", "device": "desktop",
                "timestamp": "2026-11-13T00:00:00+00:00",
                "full": {"rounds": 9.5, "client_s": 1.0},
                "ablated": {"rounds": 13.0, "client_s": 2.0}, **changes}

    def put(self, relative, payload):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return path

    def find(self, name, rows=None):
        if rows is None:
            rows, _ = table8.collect_rows(self.root)
        return [r for r in rows if r["ablation"] == name]


class EvidenceTests(EvidenceFixture):
    def test_missing_inputs_remain_pending(self):
        rows, issues = table8.collect_rows(self.root)
        self.assertEqual(issues, [])
        self.assertEqual({r["ablation"] for r in rows}, set(table8.ABLATIONS))
        for row in rows:
            self.assertFalse(row["measured"])
            self.assertEqual(row["status"], "pending")
            for key in ("evidence", "full", "ablated", "effect"):
                self.assertEqual(row[key], table8.PENDING)

    def test_only_e7_json_is_read(self):
        for source in ("results/raw_results/E1/seed.json", "results/raw_results/E2/memory.json",
                       "results/raw_results/E4/wire.json", "results/raw_results/E10/seed.json",
                       "results/E7/old.json", "results/E_delphi/old.json"):
            self.put(source, self.record(ablation="legacy"))
        self.put("results/raw_results/E7/ablation/compact_latency.json", self.record())
        original = Path.read_text

        def guarded_read(path, *args, **kwargs):
            self.assertTrue(table8.within(path, self.base), str(path))
            return original(path, *args, **kwargs)

        with mock.patch.object(Path, "read_text", guarded_read):
            rows, issues = table8.collect_rows(self.root)
        self.assertEqual(issues, [])
        self.assertEqual(self.find("legacy", rows), [])
        self.assertEqual(self.find("compact_latency", rows)[0]["evidence"], "实测")

    def test_named_ablation_files_are_recognized(self):
        for name in ("compact_latency", "wo_client_aware", "wo_planner"):
            self.put(f"results/raw_results/E7/desktop/ablation/{name}.json", self.record())
        rows, _ = table8.collect_rows(self.root)
        for name in ("compact_latency", "wo_client_aware", "wo_planner"):
            self.assertTrue(self.find(name, rows)[0]["measured"])
        self.put("results/raw_results/E7/unrelated/wo_planner.json", self.record())
        self.assertEqual(len(self.find("wo_planner")), 1)

    def test_explicit_ablation_and_config_ablation(self):
        self.put("results/raw_results/E7/custom/a.json", self.record(ablation="wo_fusion"))
        self.put("results/raw_results/E7/custom/b.json",
                 self.record(config={"ablation": "another_ablation"}))
        self.assertEqual(self.find("wo_fusion")[0]["evidence"], "实测")
        self.assertEqual(self.find("another_ablation")[0]["evidence"], "实测")

    def test_strict_measured_status_and_error_gate(self):
        invalid = [self.record(measured=False), self.record(measured=1),
                   self.record(measured="true"), self.record(status="measured"),
                   self.record(status="OK"), self.record(status="error"),
                   self.record(error="失败"), self.record(ablated={"error": "失败"})]
        missing_measured = self.record()
        del missing_measured["measured"]
        missing_status = self.record()
        del missing_status["status"]
        invalid.extend([missing_measured, missing_status])
        for record in invalid:
            with self.subTest(record=record):
                row = table8.evidence_row("wo_planner", record, "E7/test.json")
                self.assertFalse(row["measured"])
                self.assertEqual(row["evidence"], table8.PENDING)
                self.assertEqual(row["full"], table8.PENDING)
        for error in (None, "", {}):
            self.assertEqual(table8.evidence_row("wo_planner", self.record(error=error),
                                                "E7/test.json")["evidence"], "实测")

    def test_nested_rows_cannot_inherit_measured(self):
        self.put("results/raw_results/E7/custom/rows.json", {
            "measured": True, "status": "ok", "rows": [
                self.record(ablation="wo_fusion"),
                {"ablation": "wo_packing", "status": "ok"},
                self.record(ablation="wo_offline", error="失败"),
            ]})
        self.assertTrue(self.find("wo_fusion")[0]["measured"])
        self.assertFalse(self.find("wo_packing")[0]["measured"])
        self.assertFalse(self.find("wo_offline")[0]["measured"])

    def test_failed_parent_blocks_nested_evidence(self):
        for parent in ({"error": "运行失败"}, {"status": "error"}):
            row = table8.evidence_row("wo_fusion", self.record(), "E7/test.json", parent)
            self.assertEqual(row["evidence"], table8.PENDING)

    def test_duplicate_records_are_not_averaged_or_overwritten(self):
        for seed in (73, 74):
            self.put(f"results/raw_results/E7/seed{seed}/wo_fusion.json",
                     self.record(ablation="wo_fusion", config={"seed": seed}))
        rows = self.find("wo_fusion")
        self.assertEqual(len(rows), 2)
        self.assertEqual([r["config"]["seed"] for r in rows], [73, 74])

    def test_metrics_are_copied_without_rounds_or_effect_formulas(self):
        record = self.record(ablation="wo_planner", full={"K_h": 4},
                             ablated={"K_h": 0, "rounds": 17.123})
        self.put("results/raw_results/E7/custom/metrics.json", record)
        row = self.find("wo_planner")[0]
        self.assertEqual(row["full"], {"K_h": 4})
        self.assertEqual(row["ablated"]["rounds"], 17.123)
        self.assertEqual(row["effect"], table8.PENDING)
        self.assertEqual(row["record"], record)
        for key in ("config", "git_sha", "device", "timestamp"):
            self.assertEqual(row[key], record[key])
        self.assertEqual(table8.R_H, 6.02)

    def test_memory_source_rejects_legacy_and_path_escape(self):
        for path in (self.root / "results/raw_results/E2/summary.json",
                     Path("results/raw_results/E7/../E2/summary.json"),
                     Path("results/E7/summary.json"), Path("results/raw_results/E7/summary.csv")):
            with self.subTest(path=path), self.assertRaises(ValueError):
                table8.collect_rows(self.root, path)

    def test_memory_source_reads_only_e7_and_preserves_missing_comparison(self):
        memory = self.put("results/raw_results/E7/memory/summary.json",
                          self.record(full={"client_peak_rss_mb": 12.0}, ablated=None))
        rows, _ = table8.collect_rows(self.root, memory.relative_to(self.root))
        row = self.find("wo_stream_keys", rows)[0]
        self.assertTrue(row["measured"])
        self.assertEqual(row["full"], {"client_peak_rss_mb": 12.0})
        self.assertEqual(row["ablated"], table8.PENDING)
        self.assertEqual(row["effect"], table8.PENDING)
        missing = self.base / "memory/missing.json"
        _, issues = table8.collect_rows(self.root, missing)
        self.assertIn("不存在", issues[0]["error"])

    def test_malformed_input_is_pending_with_diagnostic(self):
        path = self.base / "ablation/wo_planner.json"
        path.parent.mkdir(parents=True)
        path.write_text("{", encoding="utf-8")
        rows, issues = table8.collect_rows(self.root)
        self.assertEqual(len(issues), 1)
        self.assertEqual(self.find("wo_planner", rows)[0]["evidence"], table8.PENDING)

    def test_summary_is_not_reconsumed_as_raw_evidence(self):
        self.put("results/raw_results/E7/table.json", {
            "record_type": "summary", "measured": False,
            "rows": [self.record(ablation="wo_fusion")],
        })
        self.assertFalse(self.find("wo_fusion")[0]["measured"])

    def test_no_heavy_imports_or_protocol_probe(self):
        tree = ast.parse((ROOT / "analysis/gen_e7_table8.py").read_text(encoding="utf-8"))
        forbidden = {"numpy", "torch", "planner", "cost_model", "pi_proto", "data_manifest"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                self.assertTrue(forbidden.isdisjoint(a.name.split(".")[0] for a in node.names))
            elif isinstance(node, ast.ImportFrom):
                self.assertNotIn((node.module or "").split(".")[0], forbidden)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.assertNotEqual(node.name, "measured_budget")

    def test_cli_help_retains_existing_flags(self):
        process = subprocess.run([sys.executable, str(ROOT / "analysis/gen_e7_table8.py"),
                                  "--help"], capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        for flag in ("--root", "--acc-floor", "--rtt-ms", "--bandwidth-mbps",
                     "--memory-summary", "--out"):
            self.assertIn(flag, process.stdout)


class OutputTests(EvidenceFixture):
    def config(self):
        return {"root": str(self.root), "acc_floor": 0.82, "rtt_ms": 50.0,
                "bandwidth_mbps": 50.0, "memory_summary": None,
                "out": str(self.root / "results/E7")}

    def test_summary_has_complete_metadata_and_row_evidence(self):
        self.put("results/raw_results/E7/ablation/wo_planner.json", self.record())
        rows, issues = table8.collect_rows(self.root)
        target = table8.write_summary(self.root, self.root / "results/E7",
                                     self.config(), rows, issues)
        summary = json.loads(target.read_text(encoding="utf-8"))
        for key in ("config", "git_sha", "device", "measured", "timestamp"):
            self.assertIn(key, summary)
        self.assertEqual(summary["config"], self.config())
        self.assertFalse(summary["measured"])
        self.assertTrue(summary["summary"])
        self.assertEqual(summary["record_type"], "summary")
        self.assertIn("按行", summary["evidence_scope"])
        self.assertEqual(summary["r_h"], 6.02)
        self.assertTrue(self.find("wo_planner", summary["rows"])[0]["measured"])
        self.assertFalse(self.find("wo_fusion", summary["rows"])[0]["measured"])

    def test_json_uses_tmp_then_os_replace(self):
        rows, issues = table8.collect_rows(self.root)
        with mock.patch.object(os, "replace", wraps=os.replace) as replace:
            target = table8.write_summary(self.root, self.root / "results/E7",
                                         self.config(), rows, issues)
        replace.assert_called_once()
        source, destination = replace.call_args.args[:2]
        self.assertTrue(str(source).endswith(".tmp"), str(source))
        self.assertEqual(Path(destination).resolve(), target.resolve())
        self.assertFalse(Path(source).exists())
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_interrupted_replace_preserves_existing_summary(self):
        target = self.put("results/E7/e7_table8.json", {"sentinel": "保留"})
        before = target.read_bytes()
        rows, issues = table8.collect_rows(self.root)
        with mock.patch.object(os, "replace", side_effect=OSError("模拟替换失败")):
            with self.assertRaises(OSError):
                table8.write_summary(self.root, target.parent, self.config(), rows, issues)
        self.assertEqual(target.read_bytes(), before)
        self.assertEqual(list(target.parent.glob("*.tmp")), [])

    def test_raw_results_existing_file_is_never_overwritten(self):
        for relative in ("results/raw_results/E7/e7_table8.json",
                         "results/raw_results/E2/e7_table8.json",
                         "external/raw_results/e7_table8.json"):
            target = self.put(relative, {"sentinel": relative})
            before = target.read_bytes()
            rows, issues = table8.collect_rows(self.root)
            with self.assertRaises(FileExistsError):
                table8.write_summary(self.root, target.parent, self.config(), rows, issues)
            self.assertEqual(target.read_bytes(), before)

    def test_new_raw_summary_is_no_clobber_and_not_reconsumed(self):
        rows, issues = table8.collect_rows(self.root)
        target = table8.write_summary(self.root, self.base, self.config(), rows, issues)
        before = target.read_bytes()
        with self.assertRaises(FileExistsError):
            table8.write_summary(self.root, self.base, self.config(), rows, issues)
        self.assertEqual(target.read_bytes(), before)
        second_rows, second_issues = table8.collect_rows(self.root)
        self.assertEqual(rows, second_rows)
        self.assertEqual(issues, second_issues)

    def test_cli_writes_to_directory_and_rejects_non_e7_memory(self):
        out = self.root / "summary-output"
        with contextlib.redirect_stdout(io.StringIO()):
            code = table8.main(["--root", str(self.root), "--out", str(out),
                               "--acc-floor", "0.9", "--rtt-ms", "25",
                               "--bandwidth-mbps", "100"])
        self.assertEqual(code, 0)
        summary = json.loads((out / "e7_table8.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["config"]["acc_floor"], 0.9)
        self.assertEqual(summary["config"]["rtt_ms"], 25.0)
        self.assertEqual(summary["config"]["bandwidth_mbps"], 100.0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
            table8.main(["--root", str(self.root), "--memory-summary",
                         str(self.root / "results/raw_results/E2/summary.json")])
        self.assertEqual(exc.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
