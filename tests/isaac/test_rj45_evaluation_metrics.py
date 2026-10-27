import ast
import math
from pathlib import Path
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATE_POLICY_PATH = PROJECT_ROOT / "scripts/evaluate_policy.py"


def _compiled_function(name):
    tree = ast.parse(EVALUATE_POLICY_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
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
    return namespace[name]


def _compiled_terminal_attempt_metrics():
    return _compiled_function("_rj45_terminal_attempt_metrics")


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
    def test_reverification_contract_distinguishes_table_held_and_toy_scope(self):
        complete = _compiled_function("_rj45_evaluation_inputs_reverified")

        self.assertTrue(
            complete("table", nominal_reverified=True, held_reverified=False)
        )
        self.assertTrue(
            complete("held", nominal_reverified=True, held_reverified=True)
        )
        self.assertFalse(
            complete("held", nominal_reverified=True, held_reverified=False)
        )
        self.assertFalse(
            complete("table", nominal_reverified=False, held_reverified=True)
        )
        with self.assertRaises(ValueError):
            complete("toy", nominal_reverified=True, held_reverified=True)
        with self.assertRaises(ValueError):
            complete("table", nominal_reverified=1, held_reverified=False)

    def test_schema_v2_is_attached_only_with_required_verified_proofs(self):
        attach = _compiled_function("_attach_rj45_evaluation_provenance")
        nominal = {"report_sha256": "a" * 64}
        held = {"artifact_sha256": "b" * 64}

        table_report = {"provenance": {}}
        attach(
            table_report,
            evaluation_start="table",
            nominal_evidence=nominal,
            held_grasp_reset_proof=held,
            inputs_reverified_at_end=True,
        )
        self.assertEqual(table_report["rj45_evaluation_schema_version"], 2)
        self.assertEqual(table_report["provenance"]["rj45_nominal_evidence"], nominal)
        self.assertIsNone(table_report["provenance"]["held_grasp_reset_proof"])
        self.assertIs(
            table_report["provenance"]["evaluation_inputs_reverified_at_end"],
            True,
        )

        held_report = {"provenance": {}}
        attach(
            held_report,
            evaluation_start="held",
            nominal_evidence=nominal,
            held_grasp_reset_proof=held,
            inputs_reverified_at_end=True,
        )
        self.assertEqual(held_report["provenance"]["held_grasp_reset_proof"], held)
        with self.assertRaises(ValueError):
            attach(
                {"provenance": {}},
                evaluation_start="held",
                nominal_evidence=nominal,
                held_grasp_reset_proof=None,
                inputs_reverified_at_end=True,
            )
        with self.assertRaises(ValueError):
            attach(
                {"provenance": {}},
                evaluation_start="table",
                nominal_evidence=nominal,
                held_grasp_reset_proof=None,
                inputs_reverified_at_end=False,
            )

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


class Rj45TerminalEvidenceTests(unittest.TestCase):
    def test_extracts_policy_and_both_grip_proofs_from_selected_row(self):
        extract = _compiled_function("_rj45_terminal_attempt_evidence")
        snapshot = {
            "training_policy_activated": torch.tensor([False, True, False]),
            "training_estimator_valid": torch.tensor([True, True, False]),
            "policy_input_valid": torch.tensor([False, True, False]),
            "held_initialization_passed": torch.tensor([False, True, True]),
            "valid_policy_input_step_count": torch.tensor([0, 7, 3]),
            "controller_grip_proven": torch.tensor(
                [[False, False], [True, False], [False, True]]
            ),
            "grip_proven": torch.tensor(
                [[False, True], [True, True], [False, False]]
            ),
        }

        evidence = extract(snapshot, 1)

        self.assertEqual(
            evidence,
            {
                "training_policy_activated": True,
                "training_estimator_valid": True,
                "policy_input_valid": True,
                "held_initialization_passed": True,
                "controller_grip_proven": [True, False],
                "evaluation_grip_proven": [True, True],
                "valid_policy_input_step_count": 7,
            },
        )

    def test_missing_or_malformed_terminal_evidence_is_rejected(self):
        extract = _compiled_function("_rj45_terminal_attempt_evidence")
        snapshot = {
            "training_policy_activated": torch.tensor([False]),
            "training_estimator_valid": torch.tensor([True]),
            "policy_input_valid": torch.tensor([False]),
            "held_initialization_passed": torch.tensor([True]),
            "valid_policy_input_step_count": torch.tensor([1]),
            "controller_grip_proven": torch.tensor([[True, False]]),
            "grip_proven": torch.tensor([[True, True]]),
        }
        del snapshot["training_estimator_valid"]
        with self.assertRaises(KeyError):
            extract(snapshot, 0)

        snapshot["training_estimator_valid"] = torch.tensor([True])
        snapshot["grip_proven"] = torch.tensor([[1, 1]])
        with self.assertRaises(ValueError):
            extract(snapshot, 0)

        snapshot["grip_proven"] = torch.tensor([[True, True]])
        for invalid_count in (
            torch.tensor([-1]),
            torch.tensor([True]),
            torch.tensor([1.0]),
        ):
            snapshot["valid_policy_input_step_count"] = invalid_count
            with self.subTest(invalid_count=invalid_count), self.assertRaises(ValueError):
                extract(snapshot, 0)

    def test_partial_rows_keep_independent_initialization_and_exposure_evidence(self):
        extract = _compiled_function("_rj45_terminal_attempt_evidence")
        snapshot = {
            "training_policy_activated": torch.tensor([False, True, False]),
            "training_estimator_valid": torch.tensor([False, True, True]),
            "policy_input_valid": torch.tensor([False, False, True]),
            "held_initialization_passed": torch.tensor([False, True, True]),
            "valid_policy_input_step_count": torch.tensor([0, 9, 2]),
            "controller_grip_proven": torch.tensor(
                [[False, False], [False, True], [True, True]]
            ),
            "grip_proven": torch.tensor(
                [[False, False], [False, False], [True, False]]
            ),
        }

        evidence = extract(snapshot, 1)

        self.assertTrue(evidence["held_initialization_passed"])
        self.assertEqual(evidence["valid_policy_input_step_count"], 9)
        self.assertFalse(evidence["policy_input_valid"])
        self.assertEqual(evidence["controller_grip_proven"], [False, True])
        self.assertEqual(evidence["evaluation_grip_proven"], [False, False])


class Rj45StartContractTests(unittest.TestCase):
    def test_held_requires_initialization_and_positive_policy_exposure(self):
        status = _compiled_function("_rj45_start_contract_status")
        base = {
            "held_initialization_passed": True,
            "valid_policy_input_step_count": 4,
            "training_policy_activated": False,
            "training_estimator_valid": False,
            "policy_input_valid": False,
            "controller_grip_proven": [False, False],
            "evaluation_grip_proven": [False, False],
        }

        self.assertEqual(status("held", base), (True, None))
        self.assertEqual(
            status("held", {**base, "held_initialization_passed": False}),
            (False, "held_initialization_not_passed"),
        )
        self.assertEqual(
            status("held", {**base, "valid_policy_input_step_count": 0}),
            (False, "no_valid_policy_input_exposure"),
        )

    def test_table_does_not_apply_held_initialization_or_exposure_gate(self):
        status = _compiled_function("_rj45_start_contract_status")
        prepolicy_terminal = {
            "held_initialization_passed": False,
            "valid_policy_input_step_count": 0,
            "training_policy_activated": False,
            "training_estimator_valid": False,
            "policy_input_valid": False,
            "controller_grip_proven": [False, False],
            "evaluation_grip_proven": [False, False],
        }

        self.assertEqual(status("table", prepolicy_terminal), (True, None))


class EvaluationInvalidReasonTests(unittest.TestCase):
    def test_start_contract_failures_are_preserved_per_case(self):
        start_status = _compiled_function("_rj45_start_contract_status")
        invalid_reason = _compiled_function("_evaluation_invalid_reason")

        for evidence, expected in (
            (
                {
                    "held_initialization_passed": False,
                    "valid_policy_input_step_count": 3,
                },
                "held_initialization_not_passed",
            ),
            (
                {
                    "held_initialization_passed": True,
                    "valid_policy_input_step_count": 0,
                },
                "no_valid_policy_input_exposure",
            ),
        ):
            start_valid, start_reason = start_status("held", evidence)
            self.assertFalse(start_valid)
            self.assertEqual(
                invalid_reason(
                    valid=False,
                    finite=True,
                    start_contract_invalid_reason=start_reason,
                    evaluator_termination_reason="timeout",
                ),
                expected,
            )

    def test_valid_physical_failure_has_no_invalid_reason(self):
        invalid_reason = _compiled_function("_evaluation_invalid_reason")

        self.assertIsNone(
            invalid_reason(
                valid=True,
                finite=True,
                start_contract_invalid_reason=None,
                evaluator_termination_reason="grasp_loss",
            )
        )
        self.assertEqual(
            invalid_reason(
                valid=False,
                finite=True,
                start_contract_invalid_reason=None,
                evaluator_termination_reason="sensor_invalid",
            ),
            "sensor_invalid",
        )
        self.assertEqual(
            invalid_reason(
                valid=False,
                finite=False,
                start_contract_invalid_reason="held_initialization_not_passed",
                evaluator_termination_reason="timeout",
            ),
            "nonfinite_episode_metrics",
        )


class EvaluationStartCliContractTests(unittest.TestCase):
    def test_held_start_is_rejected_for_toy_or_without_rj45_held_proof(self):
        error = _compiled_function("_evaluation_start_configuration_error")

        self.assertEqual(
            error("toy", "held", True),
            "--evaluation-start held requires --task rj45",
        )
        self.assertEqual(
            error("rj45", "held", False),
            "held RJ45 evaluation requires --rj45-held-state",
        )
        self.assertIsNone(error("rj45", "held", True))
        self.assertIsNone(error("rj45", "table", False))


if __name__ == "__main__":
    unittest.main()
