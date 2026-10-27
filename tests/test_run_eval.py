"""Tests for eval/run_eval.py case loading. No reader, no network: only the
merging of the two case files is exercised here. Run from ~/workspace/yc-latent:
    python3 -m unittest discover -s tests -v
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))

from run_eval import _load_cases, _pr  # noqa: E402


class TestLoadCases(unittest.TestCase):
    def test_loads_all_54_with_sources(self):
        cases = _load_cases()
        self.assertEqual(len(cases), 54)
        self.assertEqual(sum(1 for c in cases if c["source"] == "adversarial"), 30)
        self.assertEqual(sum(1 for c in cases if c["source"] == "benchmark"), 24)
        # Case ids repeat across the two files; (source, id) must not.
        self.assertEqual(len({(c["source"], c["id"]) for c in cases}), 54)

    def test_benchmark_expectations_map_to_verdicts(self):
        cases = {(c["source"], c["id"]): c for c in _load_cases()}
        self.assertEqual(cases[("benchmark", "clean-01")]["expected"], "pass")
        self.assertEqual(cases[("benchmark", "num-01")]["expected"], "escalate")
        self.assertEqual(cases[("benchmark", "clean-01")]["kind"], "clean")
        self.assertEqual(cases[("adversarial", "nosrc-01")]["expected"], "unscored")

    def test_every_case_is_runnable(self):
        for case in _load_cases():
            self.assertIn(case["expected"], ("pass", "escalate", "unscored"))
            self.assertTrue(case["answer"])
            self.assertIsInstance(case["context"], str)
            self.assertTrue(case["kind"])
            self.assertTrue(case["prompt"])

    def test_pr_counts_merged_rows(self):
        rows = [
            {"expected": "escalate", "v": "escalate"},
            {"expected": "escalate", "v": "pass"},
            {"expected": "pass", "v": "escalate"},
            {"expected": "pass", "v": "pass"},
            {"expected": "unscored", "v": "unscored"},
            # "ood" is a flagged verdict, like "escalate": it counts as a
            # hit on an escalate-expected case and a false flag on a
            # pass-expected one, never as a plain pass.
            {"expected": "escalate", "v": "ood"},
            {"expected": "pass", "v": "ood"},
        ]
        stats = _pr(rows, "v")
        self.assertEqual(stats["cases"], 6)
        self.assertEqual((stats["true_positives"], stats["false_negatives"],
                          stats["false_positives"], stats["true_negatives"]), (2, 1, 2, 1))
        self.assertEqual(stats["precision"], 0.5)
        self.assertEqual(stats["recall"], 0.6667)


if __name__ == "__main__":
    unittest.main()
