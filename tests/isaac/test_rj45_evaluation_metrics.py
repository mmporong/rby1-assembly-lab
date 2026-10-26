import ast
import math
from pathlib import Path
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_POLICY_PATH = PROJECT_ROOT / "scripts/evaluate_policy.py"


def _compiled_terminal_attempt_metrics():
    tree = ast.parse(EVALUATE_POLICY_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_rj45_terminal_attempt_metrics"
    )
    namespace = {"math": math}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            EVALUATE_POLICY_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace["_rj45_terminal_attempt_metrics"]


def _terminal_snapshot():
    return {
        "timestamp_s": torch.tensor([3.25, 7.5, 11.75]),
        "retry_count": torch.tensor([0, 2, 4]),
        "grip_contact_force_peak_n": torch.tensor([10.0, 20.5, 30.0]),
        "insertion_contact_force_peak_n": torch.tensor([1.0, 2.5, 3.0]),
        "insertion_contact_load_peak_n": torch.tensor([4.0, 5.5, 6.0]),
        "forbidden_contact_load_peak_n": torch.tensor([0.1, 0.2, 0.3]),
        "actual_depth_m": torch.tensor([0.001, 0.0025, 0.003]),
        "actual_lateral_error_m": torch.tensor([0.004, 0.0055, 0.006]),
        "actual_orientation_error_rad": torch.tensor([0.007, 0.0085, 0.009]),
        "independent_eval_dwell_s": torch.tensor([0.4, 0.75, 1.0]),
    }


class Rj45EvaluationMetricTests(unittest.TestCase):
    def test_extracts_only_selected_environment_actual_finite_values(self):
        extract = _compiled_terminal_attempt_metrics()

        metrics, finite = extract(_terminal_snapshot(), 1)

        self.assertTrue(finite)
        self.assertEqual(metrics["elapsed_s"], 7.5)
        self.assertEqual(metrics["retry_count"], 2)
        self.assertEqual(metrics["grip_contact_force_peak_n"], 20.5)
        self.assertEqual(metrics["insertion_contact_force_peak_n"], 2.5)
        self.assertEqual(metrics["insertion_contact_load_peak_n"], 5.5)
        self.assertAlmostEqual(metrics["forbidden_contact_load_peak_n"], 0.2)
        self.assertAlmostEqual(metrics["final_depth_m"], 0.0025)
        self.assertAlmostEqual(metrics["final_lateral_error_m"], 0.0055)
        self.assertAlmostEqual(metrics["final_orientation_error_rad"], 0.0085)
        self.assertEqual(metrics["final_dwell_s"], 0.75)

    def test_nonfinite_values_are_null_and_invalidate_only_their_rows(self):
        extract = _compiled_terminal_attempt_metrics()
        snapshot = _terminal_snapshot()
        snapshot["grip_contact_force_peak_n"][0] = torch.nan
        snapshot["insertion_contact_load_peak_n"][1] = torch.inf
        snapshot["actual_orientation_error_rad"][2] = -torch.inf

        row_zero, row_zero_finite = extract(snapshot, 0)
        row_one, row_one_finite = extract(snapshot, 1)
        row_two, row_two_finite = extract(snapshot, 2)

        self.assertFalse(row_zero_finite)
        self.assertIsNone(row_zero["grip_contact_force_peak_n"])
        self.assertEqual(row_zero["insertion_contact_load_peak_n"], 4.0)
        self.assertFalse(row_one_finite)
        self.assertIsNone(row_one["insertion_contact_load_peak_n"])
        self.assertEqual(row_one["grip_contact_force_peak_n"], 20.5)
        self.assertFalse(row_two_finite)
        self.assertIsNone(row_two["final_orientation_error_rad"])
        self.assertEqual(row_two["retry_count"], 4)

    def test_missing_mandatory_metric_is_rejected(self):
        extract = _compiled_terminal_attempt_metrics()
        snapshot = _terminal_snapshot()
        del snapshot["independent_eval_dwell_s"]

        with self.assertRaises(KeyError):
            extract(snapshot, 0)


if __name__ == "__main__":
    unittest.main()
