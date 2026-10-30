import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_rj45_stability_traces.py"
SPEC = importlib.util.spec_from_file_location("analyze_rj45_stability_traces", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(frame: int, step: int, physics: int, control_time: float, policy_time: float):
    return {
        "frame_index": frame,
        "control_step": step,
        "physics_step": physics,
        "control_time_s": control_time,
        "control_step_snapshot": {"policy_input_timestamp_s": policy_time},
    }


class ControlEndpointTests(unittest.TestCase):
    def test_keeps_last_physics_frame_for_duplicate_control_step(self):
        rows = [
            row(0, 3, 30, 0.1, 0.08),
            row(1, 3, 31, 0.1, 0.08),
            row(2, 4, 40, 0.2, 0.18),
        ]
        endpoints = MODULE.control_endpoints(rows)
        self.assertEqual([item["frame_index"] for item in endpoints], [1, 2])

    def test_rejects_negative_or_future_policy_timestamp(self):
        with self.assertRaisesRegex(MODULE.AnalysisError, "nonnegative"):
            MODULE.control_endpoints([row(0, 1, 1, 0.1, -0.01)])
        with self.assertRaisesRegex(MODULE.AnalysisError, "later"):
            MODULE.control_endpoints([row(0, 1, 1, 0.1, 0.11)])


class EvidenceBoundaryTests(unittest.TestCase):
    def test_manifest_rejects_hash_mismatch(self):
        root = Path(self.id().replace(".", "_"))
        root.mkdir(exist_ok=True)
        try:
            paths = {}
            inputs = {}
            for label in ("video", "frames", "metadata", "evidence", "report"):
                path = root / label
                path.write_bytes(label.encode())
                paths[label] = path
                inputs[label] = {"sha256": MODULE.sha256(path), "bytes": path.stat().st_size}
            inputs["report"]["sha256"] = "0" * 64
            manifest = {"runs": [{"stem": stem, "input": inputs} for stem in MODULE.STEMS]}
            with self.assertRaisesRegex(MODULE.AnalysisError, "report hash"):
                MODULE.verify_public_manifest(manifest, MODULE.STEMS[0], paths)
        finally:
            for path in root.iterdir():
                path.unlink()
            root.rmdir()

    def test_selected_outcome_rejects_noncanonical_case_id(self):
        metadata = {"selected_environment_index": 54, "selected_episode_index": 1, "selected_terminal": {"success": False}}
        report = {"episode_outcomes": [{
            "environment_index": 54, "episode_index": 1, "valid": True,
            "success": False, "paired_condition": {"environment_index": 54},
            "paired_condition_id": "wrong",
        }]}
        with self.assertRaisesRegex(MODULE.AnalysisError, "canonical"):
            MODULE.selected_outcome(report, metadata)


if __name__ == "__main__":
    unittest.main()
