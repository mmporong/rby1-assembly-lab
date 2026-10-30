import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.export_rj45_policy_media import (
    EvidenceError,
    EXPECTED_BC_ACTOR,
    load_rows,
    metrics,
    outcome_text,
    sha256,
    validate_lineage,
    verify_artifact_hashes,
)


def frame(index=0, time=1 / 60, *, success=False, terminal=False, lateral=4e-5, linear=0.004, angular=0.04, dwell=0.0):
    return {
        "frame_index": index, "physics_time_s": time, "control_time_s": 0.0, "control_step": 1,
        "actual_depth_m": 0.0123, "actual_lateral_error_m": lateral,
        "actual_orientation_error_rad": 0.003, "control_step_entry_contact_force_proxy_n": 3.1,
        "terminal_observed_after_control_step": terminal, "success_observed_after_control_step": success,
        "control_step_snapshot": {
            "independent_eval_dwell_s": dwell,
            "rj45_evaluation_stability": {
                "relative_linear_speed_m_s": linear, "relative_angular_speed_rad_s": angular,
                "physics_all_samples_stable": success,
            },
            "terminal_evaluation_criteria": {"lateral": lateral <= 50e-6, "dwell": dwell >= 0.4},
        },
    }


class PolicyMediaTests(unittest.TestCase):
    def test_metrics_preserve_units_and_control_measurements(self):
        result = metrics(frame(lateral=91.5e-6, linear=0.00561, angular=0.249, dwell=0.0))
        self.assertAlmostEqual(result["lateral_error_um"], 91.5)
        self.assertAlmostEqual(result["relative_linear_speed_mm_s"], 5.61)
        self.assertAlmostEqual(result["relative_angular_speed_rad_s"], 0.249)
        self.assertEqual(result["dwell_s"], 0.0)

    def test_outcome_separates_success_from_observed_failure_reasons(self):
        success = metrics(frame(success=True, terminal=True, dwell=0.4))
        self.assertEqual(outcome_text("v3-biased-zero-env54", success)[0], "기준 성공")
        failure = metrics(frame(terminal=True, lateral=91.5e-6, linear=0.00561, angular=0.249))
        heading, detail, _ = outcome_text("v5-bc-env42", failure)
        self.assertIn("타임아웃", heading)
        self.assertIn("91.5", detail)
        self.assertIn("5.61", detail)
        self.assertIn("0.249", detail)

    def test_jsonl_rejects_non_contiguous_frames_and_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.jsonl"
            values = [frame(index=0, time=0.02), frame(index=2, time=0.01)]
            path.write_text("\n".join(json.dumps(value) for value in values), encoding="utf-8")
            with self.assertRaises(EvidenceError):
                load_rows(path)

    def test_frozen_actor_hash_is_explicit_contract(self):
        self.assertEqual(len(EXPECTED_BC_ACTOR), 64)
        self.assertNotEqual(EXPECTED_BC_ACTOR, "0" * 64)

    def test_lineage_rejects_frame_or_frozen_actor_mismatch(self):
        rows = [{"policy_lineage_sha256": "lineage-a"}]
        metadata = {
            "policy_lineage_sha256": "lineage-a",
            "policy_lineage": {"mode": "bc", "bc_actor": {"sha256": EXPECTED_BC_ACTOR}},
        }
        validate_lineage("v5-bc-env54", metadata, rows)
        rows[0]["policy_lineage_sha256"] = "lineage-b"
        with self.assertRaises(EvidenceError):
            validate_lineage("v5-bc-env54", metadata, rows)
        rows[0]["policy_lineage_sha256"] = "lineage-a"
        metadata["policy_lineage"]["bc_actor"]["sha256"] = "0" * 64
        with self.assertRaises(EvidenceError):
            validate_lineage("v5-bc-env54", metadata, rows)

    def test_input_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {name: Path(directory) / name for name in ("video", "frames", "metadata")}
            for name, path in paths.items():
                path.write_bytes(name.encode())
            evidence = {"artifacts": {name: {"sha256": sha256(path)} for name, path in paths.items()}}
            verify_artifact_hashes(paths, evidence)
            paths["frames"].write_bytes(b"changed")
            with self.assertRaises(EvidenceError):
                verify_artifact_hashes(paths, evidence)


if __name__ == "__main__":
    unittest.main()
