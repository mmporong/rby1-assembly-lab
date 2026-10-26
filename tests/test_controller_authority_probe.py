from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "checks/probe_rj45_controller_authority.py"
SPEC = importlib.util.spec_from_file_location("probe_rj45_controller_authority", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def _config(mode: str = "absolute_tip_pose_target_insert_only") -> dict:
    return {
        "action_mode": "right_only_6d",
        "action_application_mode": mode,
        "base_observation_dim": 79,
        "diagnostic_observation_mode": "standard",
        "calibration_bias": {
            "translation_levels_m": [0.00015, 0.00035, 0.0006],
            "initial_level": 0,
        },
    }


class ControllerAuthorityProbeTests(unittest.TestCase):
    def test_probe_contract_accepts_only_the_two_ab_modes(self) -> None:
        for mode in (
            "post_integral_step",
            "absolute_tip_pose_target_insert_only",
        ):
            with self.subTest(mode=mode):
                self.assertEqual(
                    probe._validate_probe_training_config(_config(mode)), mode
                )

    def test_probe_contract_fails_closed(self) -> None:
        cases = (
            ("action_mode", "both_arms_12d", "right_only_6d"),
            ("base_observation_dim", 85, "79D"),
            (
                "diagnostic_observation_mode",
                "persistent_calibration_bias",
                "privileged",
            ),
            ("action_application_mode", "unknown", "supported"),
        )
        for field, value, message in cases:
            with self.subTest(field=field, value=value):
                config = _config()
                config[field] = value
                with self.assertRaisesRegex(ValueError, message):
                    probe._validate_probe_training_config(config)

    def test_analytic_action_matches_bias_and_rejects_clipping(self) -> None:
        action = probe._analytic_probe_action(1.5e-3)
        expected = (1.0 / 15.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        for index, (actual, wanted) in enumerate(zip(action, expected, strict=True)):
            with self.subTest(index=index):
                self.assertAlmostEqual(actual, wanted)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            probe._analytic_probe_action(50.0e-6)

    def test_hold_dwell_resets_on_one_bad_sample(self) -> None:
        dwell = 0.0
        for sample_ok in (True, True, False, True):
            dwell = probe._advance_hold_dwell(dwell, sample_ok, 1.0 / 30.0)
        self.assertAlmostEqual(dwell, 1.0 / 30.0)
        with self.assertRaisesRegex(ValueError, "sample_ok"):
            probe._advance_hold_dwell(dwell, 1, 1.0 / 30.0)

    def test_action_application_status_detects_ignored_tip_offset(self) -> None:
        expected_action = [0.0] * 12
        expected_action[6] = 1.0 / 15.0
        expected_offset = [100.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0]
        accepted = probe._action_application_status(
            requested_normalized_12d=expected_action,
            applied_normalized_12d=[*expected_action[:6], expected_action[6] + 5.0e-7, *expected_action[7:]],
            applied_tip_offset_6d=[100.0e-6 + 0.5e-9, 0.0, 0.0, 0.0, 0.0, 0.0],
            expected_normalized_12d=expected_action,
            expected_tip_offset_6d=expected_offset,
        )
        self.assertTrue(all(accepted.values()))
        ignored = probe._action_application_status(
            requested_normalized_12d=expected_action,
            applied_normalized_12d=expected_action,
            applied_tip_offset_6d=[0.0] * 6,
            expected_normalized_12d=expected_action,
            expected_tip_offset_6d=expected_offset,
        )
        self.assertFalse(ignored["applied_tip_offset_matches"])
        distributed_error = expected_action.copy()
        distributed_error[6] += 0.8e-6
        distributed_error[7] += 0.8e-6
        l2_rejected = probe._action_application_status(
            requested_normalized_12d=expected_action,
            applied_normalized_12d=distributed_error,
            applied_tip_offset_6d=expected_offset,
            expected_normalized_12d=expected_action,
            expected_tip_offset_6d=expected_offset,
        )
        self.assertFalse(l2_rejected["applied_normalized_action_matches"])

    def test_release_boundary_rejects_carried_strict_evaluator_state(self) -> None:
        self.assertTrue(probe._release_boundary_status(0.0, False, -1))
        for dwell, success, since in (
            (1.0 / 30.0, False, 10),
            (0.0, True, -1),
            (0.0, False, 10),
        ):
            with self.subTest(dwell=dwell, success=success, since=since):
                self.assertFalse(
                    probe._release_boundary_status(dwell, success, since)
                )

    def test_authority_evidence_rejects_invalid_first_policy_input(self) -> None:
        self.assertTrue(probe._authority_evidence_gate(True, True))
        self.assertFalse(probe._authority_evidence_gate(True, False))
        self.assertFalse(probe._authority_evidence_gate(False, True))
        with self.assertRaisesRegex(ValueError, "first_policy_input_valid"):
            probe._authority_evidence_gate(True, 1)

    def test_substep_contact_samples_require_nine_finite_well_shaped_rows(self) -> None:
        insertion = [0.0] * 9
        forbidden = [0.0] * 9
        fingers = [[1.5, 1.6, 1.7, 1.8] for _ in range(9)]
        summary = probe._validate_substep_contact_samples(
            insertion, forbidden, fingers, expected_count=9
        )
        self.assertTrue(
            probe._substep_contact_hold_ok(
                summary, grip_threshold_n=1.0, maximum_force_n=25.0
            )
        )
        with self.assertRaisesRegex(ValueError, "sample count"):
            probe._validate_substep_contact_samples(
                insertion[:8], forbidden[:8], fingers[:8], expected_count=9
            )
        malformed = [*fingers[:8], [1.0, 1.0, 1.0]]
        with self.assertRaisesRegex(ValueError, "four values"):
            probe._validate_substep_contact_samples(
                insertion, forbidden, malformed, expected_count=9
            )
        nonfinite = insertion.copy()
        nonfinite[4] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            probe._validate_substep_contact_samples(
                nonfinite, forbidden, fingers, expected_count=9
            )

    def test_middle_physics_contact_transient_fails_hold_gate(self) -> None:
        insertion = [0.0] * 9
        insertion[4] = 0.01
        summary = probe._validate_substep_contact_samples(
            insertion,
            [0.0] * 9,
            [[1.5, 1.5, 1.5, 1.5] for _ in range(9)],
            expected_count=9,
        )
        self.assertFalse(
            probe._substep_contact_hold_ok(
                summary, grip_threshold_n=1.0, maximum_force_n=25.0
            )
        )

    def test_output_paths_are_protected(self) -> None:
        protected = PROJECT_ROOT / "artifacts/probe/protected.json"
        report = PROJECT_ROOT / "artifacts/probe/report.json"
        trace = PROJECT_ROOT / "artifacts/probe/trace.jsonl"
        self.assertEqual(
            probe._validate_output_paths(
                report, trace, protected_paths=(protected,)
            ),
            (report.resolve(), trace.resolve()),
        )
        with self.assertRaisesRegex(ValueError, "overwrite"):
            probe._validate_output_paths(
                protected, trace, protected_paths=(protected,)
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            outside_report = Path(temporary_directory) / "report.json"
            with self.assertRaisesRegex(ValueError, "artifacts"):
                probe._validate_output_paths(
                    outside_report, trace, protected_paths=()
                )

    def test_atomic_jsonl_is_complete_and_rejects_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "trace.jsonl"
            rows = [
                {"step": 0, "value": 1.25},
                {"step": 1, "value": [2.0, 3.0]},
            ]
            probe._atomic_write_jsonl(output, rows)
            self.assertEqual(
                [json.loads(line) for line in output.read_text().splitlines()],
                rows,
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                probe._atomic_write_jsonl(output, [{"value": math.inf}])
            self.assertEqual(
                [json.loads(line) for line in output.read_text().splitlines()],
                rows,
            )

    def test_incremental_trace_publishes_partial_records_on_close(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "partial.jsonl"
            writer = probe._IncrementalJsonlTrace(output)
            writer.append(
                {
                    "step": 0,
                    "stage": "terminal_hold",
                    "terminal_mask": True,
                    "physics_contact_insertion_load_n": [0.0] * 9,
                    "physics_contact_forbidden_load_n": [0.0] * 9,
                    "physics_contact_finger_force_n": [
                        [1.5, 1.5, 1.5, 1.5] for _ in range(9)
                    ],
                }
            )
            self.assertFalse(output.exists())
            writer.close()
            preserved = json.loads(output.read_text())
            self.assertTrue(preserved["terminal_mask"])
            self.assertEqual(len(preserved["physics_contact_insertion_load_n"]), 9)
            self.assertEqual(len(preserved["physics_contact_finger_force_n"]), 9)


if __name__ == "__main__":
    unittest.main()
