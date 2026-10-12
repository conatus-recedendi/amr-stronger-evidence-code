"""Reporting regression tests, using explicitly synthetic local results."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("report_runner", HERE / "run_experiment.py")
reporting = load("reporting", HERE / "reporting.py")


def config():
    return {"model": "SYNTHETIC FIXTURE (not measured)", "seed": 42, "judge_model": "test-judge",
            "samples_per_question": 3, "train_csv": None, "smoke_test": True}


def fixture(root, partial=False):
    runner.atomic_json(root / "config.json", config())
    runner.atomic_json(root / "questions.json", [{"id": "q", "question": "Q"}])
    for variant, values in (("base", [(80, 80), (80, 80), (20, 20)]),
                            ("finetuned", [(20, 80), (30, 80), (20, 50)])):
        for index, (alignment, coherence) in enumerate(values):
            row = {"id": f"q:{index}", "question": "Q", "answer": f"Synthetic {index}"}
            runner.append_record(root / f"{variant}_responses.jsonl", row)
            if not partial or index == 0:
                runner.append_record(root / f"{variant}_scores.jsonl", {**row,
                    "alignment": {"score": alignment}, "coherence": {"score": coherence}})


class ReportingTests(unittest.TestCase):
    def test_local_denominators_and_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            local, tables = reporting.build_tables(root, config(), runner.summarize)
            self.assertEqual(local["finetuned"]["misaligned"], 1)
            self.assertEqual(local["finetuned"]["em_rate_coherent"], .5)
            by_metric = {r[0]: r for r in tables[0]["rows"]}
            self.assertEqual(by_metric["EM / all scored"][2], "33.33%")
            self.assertEqual(by_metric["EM / coherent"][1:], ["0.00%", "50.00%", "+50.00 pp"])
            published = tables[1]["rows"][2:]
            self.assertEqual(published[0][4:6], ["5.88%", "54/918"])
            self.assertEqual(published[1][4:6], ["4.52%", "47/1039"])
            self.assertEqual(published[-1][1], "GPT-4o")
            self.assertEqual(published[-1][4], "~20.00%")
            self.assertTrue(all(row[-1] == "Published reference" for row in published))

    def test_unjudged_is_na_not_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local, tables = reporting.build_tables(root, config(), runner.summarize)
            self.assertEqual(local["base"]["status"], "Not evaluated")
            by_metric = {r[0]: r for r in tables[0]["rows"]}
            self.assertEqual(by_metric["EM / coherent"][1:], ["N/A", "N/A", "N/A"])
            self.assertEqual(by_metric["Misaligned responses"][1:3], ["N/A", "N/A"])

    def test_partial_results_never_get_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, partial=True)
            local, tables = reporting.build_tables(root, config(), runner.summarize)
            self.assertEqual(local["base"]["status"], "Partial")
            by_metric = {r[0]: r for r in tables[0]["rows"]}
            self.assertEqual(by_metric["EM / coherent"][-1], "N/A")

    def test_zero_coherent_denominator_is_na(self):
        self.assertEqual(reporting.percent(runner.summarize([
            {"alignment": {"score": 10}, "coherence": {"score": 0}}], 1)["em_rate_coherent"]), "N/A")

    def test_mismatched_saved_scores_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            path = root / "base_scores.jsonl"
            rows = reporting.read_rows(path)
            rows[0]["answer"] = "Different generation"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "does not match"):
                reporting.build_tables(root, config(), runner.summarize)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib required for rendering")
    def test_report_only_cli_without_gpu_or_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            env = {key: value for key, value in os.environ.items() if key not in {"HF_TOKEN", "OPENAI_API_KEY"}}
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["MPLCONFIGDIR"] = str(root / "mpl-cache")
            result = subprocess.run([sys.executable, str(HERE.parent / "main.py"),
                                     "--report-only", "--output", str(root)], env=env,
                                    capture_output=True, text=True, timeout=90)
            self.assertEqual(result.returncode, 0, result.stderr)
            for name in ("before_after.png", "literature_comparison.png", "report.pdf",
                         "before_after.csv", "literature_comparison.csv", "report.html", "report_data.json"):
                self.assertGreater((root / "report" / name).stat().st_size, 50)
            html = (root / "report/report.html").read_text()
            self.assertIn("SMOKE TEST", html)
            self.assertIn(reporting.BETLEY_URL, html)
            self.assertIn("not matched controls", html)
            self.assertFalse((root / "adapter").exists())


if __name__ == "__main__":
    unittest.main()
