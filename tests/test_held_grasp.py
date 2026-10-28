from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest

from rby1_assembly.held_grasp import (
    _candidate_entry_is_stable,
    build_held_grasp_artifact,
    load_verified_held_grasp_artifact,
)
from rby1_assembly.rj45_task import verify_recorded_mating_geometry
from rby1_assembly.rj45_grasp_generation import (
    load_grasp_generation_catalog,
    select_grasp_generation_entry,
)
from tests.test_experiment_contracts import _synthetic_contract_fixture


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _enriched_fixture(project: Path):
    """Create contract-only records; these are not simulated physics evidence."""

    task, report_path, trace_path = _synthetic_contract_fixture(project)
    task["controller"] = {"grip_contact_force_n": 0.5}
    task_config_path = project / "configs/tasks/rj45-v1.json"
    _write_json(task_config_path, {"task_id": "rj45-v1", "fixture": "not_physics_data"})
    task["_config_sha256"] = _sha256(task_config_path)

    original = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    records = [copy.deepcopy(original[0])]
    preport_pose = [0.0, -0.006, 0.0, 1.0, 0.0, 0.0, 0.0]
    socket_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    names = [f"joint_{index}" for index in range(58)] + [
        "gripper_finger_l1",
        "gripper_finger_l2",
        "gripper_finger_r1",
        "gripper_finger_r2",
    ]
    units = ["m" if name.startswith("gripper_finger_") else "rad" for name in names]
    manifolds = []
    for index in range(4):
        manifolds.append(
            {
                "source_body_path": f"/World/envs/env_0/Robot/finger_{index}",
                "target_body_path": "/World/envs/env_0/Socket" if index < 2 else "/World/envs/env_0/Plug",
                "contact_count": 1,
                "contacts": [
                    {
                        "normal_force_n": 0.6,
                        "point_world_m": [0.0, 0.0, 0.0],
                        "normal_world": [1.0, 0.0, 0.0],
                        "separation_m": -0.00001,
                    }
                ],
            }
        )

    def enrich(record, index, *, preport):
        record = copy.deepcopy(record)
        record["control_step"] = index
        entry = record["environments"][0]
        entry["timestamp_s"] = index / 30.0
        entry["phase_name"] = "insert" if not entry["terminal"] else "done"
        entry["all_joint_names"] = names
        entry["all_joint_units"] = units
        entry["all_joint_positions_si"] = [index / 1000.0] * 62
        entry["all_joint_velocities_si"] = [0.0] * 62
        entry["all_joint_reference_si"] = [(index + 1) / 1000.0] * 62
        entry["object_com_velocities_world"] = [[0.0] * 6, [0.0] * 6]
        entry["ee_poses"] = {
            "left_world_wxyz": [0.3, 0.1, 0.9, 1.0, 0.0, 0.0, 0.0],
            "right_world_wxyz": [0.3, -0.1, 0.9, 1.0, 0.0, 0.0, 0.0],
        }
        entry["grip_contact_manifold"] = copy.deepcopy(manifolds)
        entry["forbidden_contact_load_n"] = 0.0
        entry["insertion_contact_load_n"] = 0.0
        if preport:
            entry["terminal"] = False
            entry["termination_reason"] = "none"
            entry["object_poses_world_wxyz"] = {
                "socket": socket_pose,
                "plug": preport_pose,
            }
            entry["object_substep_poses_world_wxyz"] = [
                [socket_pose, preport_pose] for _ in range(4)
            ]
            entry["actual_depth_m"] = -0.006
            entry["actual_lateral_error_m"] = 0.0
            entry["actual_orientation_error_rad"] = 0.0
            entry["independent_mating_geometry"] = verify_recorded_mating_geometry(
                socket_pose,
                preport_pose,
                task["frames"]["socket_entry"],
                task["frames"]["plug_tip"],
                -0.006,
                0.0,
                0.0,
            )
            entry["independent_eval_success"] = False
            entry["independent_eval_dwell_s"] = 0.0
            entry["independent_eval_criteria"]["depth"] = False
            entry["independent_eval_criteria"]["dwell"] = False
            entry["mating_stability"]["depth_within_window"] = False
            entry["mating_stability"]["physics_all_samples_stable"] = False
        return record

    for index in range(1, 8):
        records.append(enrich(original[1], index, preport=True))
    for offset, source_record in enumerate(original[1:], start=8):
        records.append(enrich(source_record, offset, preport=False))
    trace_path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["trace_sha256"] = _sha256(trace_path)
    report["trace_record_count"] = len(records)
    report["actual_steps"] = len(records) - 1
    report["terminal_outcomes"] = [records[-1]["environments"][0]]
    report["provenance"]["task_config_sha256"] = task["_config_sha256"]
    _write_json(report_path, report)
    return task, report_path, trace_path, records


class HeldGraspBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        self.task, self.report, self.trace, self.records = _enriched_fixture(self.project)
        self.output = self.project / "artifacts/held-grasp.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _enable_generation(self, seed: int = 20261301) -> Path:
        source = (
            Path(__file__).resolve().parents[1]
            / "configs/rj45-grasp-generation-v2.json"
        )
        catalog_path = self.project / "configs/rj45-grasp-generation-v2.json"
        shutil.copyfile(source, catalog_path)
        catalog = load_grasp_generation_catalog(catalog_path)
        group, entry = select_grasp_generation_entry(catalog, seed)
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["provenance"]["seed"] = seed
        report["provenance"]["grasp_generation_catalog_path"] = str(
            catalog_path.relative_to(self.project)
        )
        report["provenance"]["grasp_generation_catalog_sha256"] = _sha256(
            catalog_path
        )
        report["rj45_grasp_generation"] = {
            "catalog_id": catalog["catalog_id"],
            "selected_catalog_entry_id": entry["id"],
            "seed": seed,
            "assignment_group": group,
            "offset_method": catalog["offset_method"],
            "left_object_local_offset_m": entry["left_object_local_offset_m"],
            "right_object_local_offset_m": entry["right_object_local_offset_m"],
        }
        _write_json(self.report, report)
        return catalog_path

    def test_builds_first_candidate_after_point_two_seconds_and_reloads(self):
        artifact = build_held_grasp_artifact(
            self.project, self.task, self.report, self.output
        )
        self.assertEqual(artifact["selection"]["control_step"], 7)
        self.assertTrue(
            math.isclose(artifact["selection"]["timestamp_s"], 7 / 30, abs_tol=1e-12)
        )
        self.assertEqual(artifact["state"]["joint_positions_si"], [0.007] * 62)
        self.assertEqual(artifact["joint_units"][-4:], ["m"] * 4)
        loaded, proof = load_verified_held_grasp_artifact(
            self.project, self.task, self.output
        )
        self.assertEqual(loaded, artifact)
        self.assertEqual(proof["artifact_sha256"], _sha256(self.output))
        self.assertEqual(proof["selection_control_step"], 7)

    def test_generation_lineage_is_bound_and_reverified(self):
        catalog_path = self._enable_generation()
        artifact = build_held_grasp_artifact(
            self.project, self.task, self.report, self.output
        )
        generation = artifact["provenance"]["grasp_generation"]
        self.assertEqual(generation["generation_seed"], 20261301)
        self.assertEqual(generation["catalog_entry_id"], "grasp-00")
        self.assertEqual(generation["catalog_sha256"], _sha256(catalog_path))
        self.assertIn(
            "configs/rj45-grasp-generation-v2.json",
            artifact["provenance"]["source_and_config_sha256"],
        )
        loaded, _proof = load_verified_held_grasp_artifact(
            self.project, self.task, self.output
        )
        self.assertEqual(loaded, artifact)

    def test_generation_loader_rejects_stale_catalog_bytes(self):
        catalog_path = self._enable_generation()
        build_held_grasp_artifact(self.project, self.task, self.report, self.output)
        catalog_path.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source/config hash mismatch"):
            load_verified_held_grasp_artifact(self.project, self.task, self.output)

    def test_generation_build_rejects_wrong_catalog_entry(self):
        self._enable_generation()
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["rj45_grasp_generation"]["selected_catalog_entry_id"] = "grasp-11"
        _write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "frozen seed mapping"):
            build_held_grasp_artifact(
                self.project, self.task, self.report, self.output
            )
        self.assertFalse(self.output.exists())

    def test_missing_stable_manifold_leaves_no_partial_output(self):
        records = copy.deepcopy(self.records)
        for record in records[1:8]:
            for manifold in record["environments"][0]["grip_contact_manifold"]:
                manifold["contacts"][0]["normal_force_n"] = 0.0
        self.trace.write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
        )
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["trace_sha256"] = _sha256(self.trace)
        _write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "no 0.2 s stable"):
            build_held_grasp_artifact(self.project, self.task, self.report, self.output)
        self.assertFalse(self.output.exists())

    def test_loader_rejects_artifact_state_mutation(self):
        build_held_grasp_artifact(self.project, self.task, self.report, self.output)
        artifact = json.loads(self.output.read_text(encoding="utf-8"))
        artifact["state"]["joint_positions_si"][0] += 0.01
        _write_json(self.output, artifact)
        with self.assertRaisesRegex(ValueError, "differs from the exact selected"):
            load_verified_held_grasp_artifact(self.project, self.task, self.output)

    def test_loader_rejects_current_source_mutation(self):
        build_held_grasp_artifact(self.project, self.task, self.report, self.output)
        source = self.project / "src/rby1_assembly/fixture.py"
        source.write_text("VALUE = 99\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source hashes are stale"):
            load_verified_held_grasp_artifact(self.project, self.task, self.output)

    def test_candidate_requires_all_physics_sample_speeds(self):
        records = copy.deepcopy(self.records)
        for record in records[1:8]:
            record["environments"][0]["object_substep_link_velocities_world"][2][1][0] = 0.006
        self.trace.write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
        )
        report = json.loads(self.report.read_text(encoding="utf-8"))
        report["trace_sha256"] = _sha256(self.trace)
        _write_json(self.report, report)
        with self.assertRaisesRegex(ValueError, "no 0.2 s stable"):
            build_held_grasp_artifact(self.project, self.task, self.report, self.output)

    def test_common_mode_motion_cannot_be_a_settled_candidate(self):
        for key in (
            "object_link_velocities_world",
            "object_com_velocities_world",
            "object_substep_link_velocities_world",
        ):
            for velocity in ([0.02, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0.06, 0]):
                with self.subTest(source=key, velocity=velocity):
                    entry = copy.deepcopy(self.records[2]["environments"][0])
                    pair = [velocity] * 2
                    entry[key] = [pair for _ in range(4)] if "substep" in key else pair
                    self.assertFalse(_candidate_entry_is_stable(entry, self.task, 1.0))

    def test_settle_limits_distinguish_joint_units_and_tracking(self):
        source = self.records[2]["environments"][0]
        self.assertTrue(_candidate_entry_is_stable(source, self.task, 1.0))
        preload = copy.deepcopy(source)
        preload["all_joint_reference_si"][61] = preload["all_joint_positions_si"][61] + 0.004
        self.assertTrue(_candidate_entry_is_stable(preload, self.task, 1.0))
        for key, index, value in (
            ("all_joint_velocities_si", 0, 0.051),
            ("all_joint_velocities_si", 61, 0.0051),
            ("all_joint_reference_si", 0, 0.1),
            ("all_joint_reference_si", 61, 0.01),
        ):
            with self.subTest(key=key, index=index):
                entry = copy.deepcopy(source)
                entry[key][index] = value
                self.assertFalse(_candidate_entry_is_stable(entry, self.task, 1.0))


if __name__ == "__main__":
    unittest.main()
