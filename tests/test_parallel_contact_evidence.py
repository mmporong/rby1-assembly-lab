import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from rby1_assembly.parallel_contact_evidence import (
    REQUIRED_CURRENT_SOURCE_PATHS,
    REQUIRED_PROBE_CHECKS,
    REQUIRED_SAT_CHECKS,
    load_parallel_contact_evidence,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ParallelContactEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative_path in REQUIRED_CURRENT_SOURCE_PATHS:
            target = self.root / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(relative_path, encoding="utf-8")
        self.trace = self.root / "trace.jsonl"
        self.trace.write_text('{"step":1}\n', encoding="utf-8")
        env_paths = [f"/World/envs/env_{index}" for index in range(4)]
        routing = {}
        for source in ("left/a", "left/b", "left/c"):
            source_paths = [f"{env_path}/Robot/{source}" for env_path in env_paths]
            filters = [
                f"{env_path}/Robot/right/{index}"
                for env_path in env_paths
                for index in range(3)
            ]
            routing[source] = {
                "expected_source_paths": source_paths,
                "body_physx_view_prim_paths": source_paths,
                "body_physx_backend_prim_paths": source_paths,
                "contact_physx_view_sensor_paths": source_paths,
                "configured_filter_regex_count": 3,
                "expected_contact_physx_view_filter_paths": filters,
                "contact_physx_view_filter_paths": filters,
                "body_count": 4,
                "contact_sensor_count": 4,
                "contact_filter_count": 3,
            }
        self.report = self.root / "report.json"
        self.report.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "passed": True,
                    "physics_executed": True,
                    "num_envs": 4,
                    "experiment": "rb_y1_parallel_cross_hand_contact_routing_probe",
                    "scene": {
                        "environment_prim_paths": env_paths,
                        "replicate_physics": False,
                    },
                    "sensor_routing": routing,
                    "checks": {name: True for name in REQUIRED_PROBE_CHECKS},
                    "failed_checks": [],
                    "provenance": {
                        "trace_sha256": _sha256(self.trace),
                        "source_sha256": {
                            path: _sha256(self.root / path)
                            for path in REQUIRED_CURRENT_SOURCE_PATHS
                        },
                    },
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        self.sat = self.root / "sat.json"
        sat_frame = {"intersection_count": 1}
        self.sat.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "passed": True,
                    "checks": {name: True for name in REQUIRED_SAT_CHECKS},
                    "failed_checks": [],
                    "environments": {
                        env: {
                            "first_threshold_contact": sat_frame,
                            "maximum_recorded_contact_load": sat_frame,
                        }
                        for env in ("0", "2")
                    },
                    "provenance": {"trace_sha256": _sha256(self.trace)},
                },
                allow_nan=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _specification(self):
        return {
            "report_path": str(self.report),
            "report_sha256": _sha256(self.report),
            "trace_path": str(self.trace),
            "trace_sha256": _sha256(self.trace),
            "sat_report_path": str(self.sat),
            "sat_report_sha256": _sha256(self.sat),
        }

    def test_accepts_pinned_positive_probe_and_sat_evidence(self):
        result = load_parallel_contact_evidence(
            self._specification(), project_root=self.root
        )
        self.assertEqual(result["validated_probe_num_envs"], 4)
        self.assertTrue(result["runtime_num_envs_requires_native_ordering_check"])
        self.assertIn("full task validation pending", result["scope"])

    def test_rejects_report_hash_mismatch(self):
        specification = self._specification()
        specification["report_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "report SHA-256 mismatch"):
            load_parallel_contact_evidence(specification, project_root=self.root)

    def test_rejects_missing_evidence_file(self):
        specification = self._specification()
        specification["trace_path"] = str(self.root / "missing-trace.jsonl")
        with self.assertRaisesRegex(ValueError, "file is missing"):
            load_parallel_contact_evidence(specification, project_root=self.root)

    def test_rejects_missing_required_check(self):
        payload = json.loads(self.report.read_text(encoding="utf-8"))
        payload["checks"].pop("reset_cross_hand_load_is_zero")
        self.report.write_text(json.dumps(payload), encoding="utf-8")
        specification = self._specification()
        with self.assertRaisesRegex(ValueError, "checks are not all true"):
            load_parallel_contact_evidence(specification, project_root=self.root)

    def test_rejects_false_sat_check(self):
        payload = json.loads(self.sat.read_text(encoding="utf-8"))
        payload["checks"][
            "each_active_environment_has_authored_convex_sat_intersection_at_peak_contact"
        ] = False
        self.sat.write_text(json.dumps(payload), encoding="utf-8")
        specification = self._specification()
        with self.assertRaisesRegex(ValueError, "SAT report checks are not all true"):
            load_parallel_contact_evidence(specification, project_root=self.root)

    def test_rejects_current_source_mismatch(self):
        path = self.root / REQUIRED_CURRENT_SOURCE_PATHS[1]
        path.write_text("changed after probe", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source mismatch"):
            load_parallel_contact_evidence(
                self._specification(), project_root=self.root
            )

    def test_rejects_non_finite_json_even_with_matching_hash(self):
        self.sat.write_text('{"value": 1e999}', encoding="utf-8")
        specification = self._specification()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_parallel_contact_evidence(specification, project_root=self.root)

    def test_rejects_non_finite_trace_row_even_with_matching_hash(self):
        self.trace.write_text('{"load": NaN}\n', encoding="utf-8")
        payload = json.loads(self.report.read_text(encoding="utf-8"))
        payload["provenance"]["trace_sha256"] = _sha256(self.trace)
        self.report.write_text(json.dumps(payload), encoding="utf-8")
        payload = json.loads(self.sat.read_text(encoding="utf-8"))
        payload["provenance"]["trace_sha256"] = _sha256(self.trace)
        self.sat.write_text(json.dumps(payload), encoding="utf-8")
        specification = self._specification()
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_parallel_contact_evidence(specification, project_root=self.root)


if __name__ == "__main__":
    unittest.main()
