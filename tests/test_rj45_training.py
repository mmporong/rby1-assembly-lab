import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from rby1_assembly.rj45_training import (
    load_training_config,
    validate_held_grasp_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rj45-training-v1.json"
HASH = "a" * 64


def valid_artifact():
    joint_names = [f"joint_{index}" for index in range(58)] + [
        "gripper_finger_l1",
        "gripper_finger_l2",
        "gripper_finger_r1",
        "gripper_finger_r2",
    ]
    return {
        "schema_version": 1,
        "task_id": "rj45-v1",
        "provenance": {
            "nominal_report_sha256": HASH,
            "nominal_report_path": "artifacts/rj45/nominal-report.json",
            "nominal_trace_sha256": "b" * 64,
            "nominal_trace_path": "artifacts/rj45/nominal-trace.jsonl",
            "runtime_config_sha256": "c" * 64,
            "task_config_sha256": "d" * 64,
            "source_and_config_sha256": {
                "src/rby1_assembly/rj45_env.py": "e" * 64,
                "configs/tasks/rj45-v1.json": "f" * 64,
            },
        },
        "selection": {
            "control_step": 842,
            "timestamp_s": 28.0666666667,
            "phase_name": "insert",
        },
        "units": {
            "joint_position": "per_joint_units",
            "joint_velocity": "per_joint_units/s",
            "joint_reference": "per_joint_units",
            "object_pose": "m,wxyz",
            "object_com_velocity": "m/s,rad/s",
            "end_effector_pose": "m,wxyz",
        },
        "joint_names": joint_names,
        "joint_units": [
            "m" if name.startswith("gripper_finger_") else "rad" for name in joint_names
        ],
        "state": {
            "joint_positions_si": [0.0] * 62,
            "joint_velocities_si": [0.0] * 62,
            "joint_reference_si": [0.0] * 62,
            "object_poses_world_wxyz": [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            "object_com_velocities_world": [[0.0] * 6, [0.0] * 6],
            "end_effector_poses_world_wxyz": [
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
        },
    }


class TrainingConfigurationTests(unittest.TestCase):
    def test_repository_configuration_is_strict_and_hashed(self):
        config = load_training_config(CONFIG_PATH)
        self.assertEqual(config["_config_sha256"], hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest())
        self.assertEqual(config["episode_scope"], "held_grasp_alignment_insertion")
        self.assertEqual(config["action_mode"], "right_only_6d")
        self.assertEqual(config["observation_dim"], 79)
        self.assertEqual(config["calibration_bias"]["translation_levels_m"], [0.00015, 0.00035, 0.0006])
        expected_rotation = [math.radians(value) for value in (0.25, 0.5, 0.8)]
        for actual, expected in zip(config["calibration_bias"]["rotation_max_levels_rad"], expected_rotation):
            self.assertAlmostEqual(actual, expected, places=15)
        self.assertFalse(config["task_physics"]["mass_randomization_enabled"])

    def test_rejects_nonfinite_overflow_and_unknown_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.json"
            path.write_text(CONFIG_PATH.read_text(encoding="utf-8").replace("24.0", "1e999"), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid finite"):
                load_training_config(path)
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config["unreviewed_option"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                load_training_config(path)

    def test_accepts_both_arm_mode_but_rejects_mass_randomization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.json"
            config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            config["action_mode"] = "both_arms_12d"
            path.write_text(json.dumps(config), encoding="utf-8")
            self.assertEqual(load_training_config(path)["action_mode"], "both_arms_12d")
            config["task_physics"]["mass_randomization_enabled"] = True
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "masses fixed"):
                load_training_config(path)

    def test_rejects_bad_curriculum_and_reward_ranges(self):
        base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cases = (
            ("levels", lambda value: value["calibration_bias"].update(translation_levels_m=[0.00035, 0.00015, 0.0006])),
            ("initial_level", lambda value: value["calibration_bias"].update(initial_level=3)),
            ("warmup", lambda value: value.update(warmup_s=24.0)),
            ("reward", lambda value: value["reward"].update(force_scale_n=0.0)),
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.json"
            for label, damage in cases:
                with self.subTest(label=label):
                    config = copy.deepcopy(base)
                    damage(config)
                    path.write_text(json.dumps(config), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        load_training_config(path)


class HeldGraspArtifactTests(unittest.TestCase):
    def test_accepts_complete_62_joint_mixed_si_state(self):
        artifact = valid_artifact()
        validated = validate_held_grasp_artifact(artifact)
        self.assertEqual(validated, artifact)
        self.assertIsNot(validated, artifact)

    def test_rejects_missing_trace_provenance_or_bad_source_digest(self):
        artifact = valid_artifact()
        artifact["provenance"].pop("nominal_trace_sha256")
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_held_grasp_artifact(artifact)
        artifact = valid_artifact()
        artifact["provenance"]["source_and_config_sha256"]["src/rby1_assembly/rj45_env.py"] = "BAD"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            validate_held_grasp_artifact(artifact)

    def test_rejects_incomplete_joint_state_and_nonunit_pose(self):
        artifact = valid_artifact()
        artifact["joint_names"] = artifact["joint_names"][:-1]
        with self.assertRaisesRegex(ValueError, "62 unique"):
            validate_held_grasp_artifact(artifact)
        artifact = valid_artifact()
        artifact["state"]["joint_reference_si"][3] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_held_grasp_artifact(artifact)
        artifact = valid_artifact()
        artifact["state"]["object_poses_world_wxyz"][1][3] = 0.9
        with self.assertRaisesRegex(ValueError, "unit length"):
            validate_held_grasp_artifact(artifact)

    def test_rejects_ambiguous_or_wrong_units(self):
        artifact = valid_artifact()
        artifact["units"]["joint_velocity"] = "rad/s"
        with self.assertRaisesRegex(ValueError, "units must equal"):
            validate_held_grasp_artifact(artifact)

    def test_rejects_wrong_prismatic_joint_units_and_untraceable_selection(self):
        artifact = valid_artifact()
        artifact["joint_units"][-1] = "rad"
        with self.assertRaisesRegex(ValueError, "gripper_finger"):
            validate_held_grasp_artifact(artifact)
        artifact = valid_artifact()
        artifact["selection"]["phase_name"] = "check_lift"
        with self.assertRaisesRegex(ValueError, "must be insert"):
            validate_held_grasp_artifact(artifact)
        artifact = valid_artifact()
        artifact["provenance"]["nominal_trace_path"] = "/tmp/unpinned.jsonl"
        with self.assertRaisesRegex(ValueError, "project-relative"):
            validate_held_grasp_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
