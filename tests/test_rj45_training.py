import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from rby1_assembly.rj45_training import (
    ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
    action_application_contract_metadata,
    load_training_config,
    policy_ground_truth_usage,
    policy_observation_dimension,
    policy_observation_ground_truth_flags,
    policy_observation_schema_version,
    policy_observation_source,
    privileged_diagnostic_observation_metadata,
    reward_profile_metadata,
    validate_reward_horizon_contract,
    validate_reward_profile_runner,
    validate_held_grasp_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rj45-training-v1.json"
DIAGNOSTIC_CONFIG_PATH = ROOT / "configs/rj45-training-bias-observability.json"
TARGET_CONFIG_PATH = ROOT / "configs/rj45-training-target-offset-v1.json"
TARGET_DIAGNOSTIC_CONFIG_PATH = ROOT / "configs/rj45-training-target-offset-bias-observability.json"
A1_CONFIG_PATH = ROOT / "configs/rj45-training-target-offset-a1-discounted-pose-v2.json"
A2_CONFIG_PATH = ROOT / "configs/rj45-training-target-offset-a2-keypoint-shape-v2.json"
HASH = "a" * 64


def load_a2_with_current_task_pin():
    config = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
    source = config["reward_profile"]["keypoint_geometry"]["geometry_source"]
    task_path = ROOT / source["task_config_path"]
    source["task_config_sha256"] = hashlib.sha256(task_path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        path = Path(temporary) / A2_CONFIG_PATH.name
        path.write_text(json.dumps(config), encoding="utf-8")
        return load_training_config(path)


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
        self.assertEqual(config["action_application_mode"], "post_integral_step")
        self.assertEqual(action_application_contract_metadata(config), {})
        self.assertEqual(config["base_observation_dim"], 79)
        self.assertEqual(config["diagnostic_observation_mode"], "standard")
        self.assertEqual(policy_observation_dimension(config), 79)
        self.assertEqual(policy_observation_schema_version(config), 1)
        self.assertEqual(privileged_diagnostic_observation_metadata(config), {})
        self.assertEqual(
            policy_observation_ground_truth_flags(config),
            {
                "ground_truth_in_policy_observation": False,
                "object_pose_ground_truth_in_policy_observation": False,
                "privileged_simulation_latent_in_policy_observation": False,
            },
        )
        self.assertEqual(
            policy_ground_truth_usage(config),
            "scripted_setup_control_and_safety_then_reward_and_independent_evaluation_only",
        )
        self.assertEqual(config["calibration_bias"]["translation_levels_m"], [0.00015, 0.00035, 0.0006])
        expected_rotation = [math.radians(value) for value in (0.25, 0.5, 0.8)]
        for actual, expected in zip(config["calibration_bias"]["rotation_max_levels_rad"], expected_rotation):
            self.assertAlmostEqual(actual, expected, places=15)
        self.assertFalse(config["task_physics"]["mass_randomization_enabled"])
        self.assertEqual(config["held_reset_staging"]["control_steps"], 1)
        self.assertEqual(config["held_reset_staging"]["provenance"]["source"], "simulation_design_assumption")

    def test_bias_observability_config_changes_only_the_explicit_diagnostic_flag(self):
        standard_raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        diagnostic_raw = json.loads(DIAGNOSTIC_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            {key: value for key, value in diagnostic_raw.items() if key != "diagnostic_observation_mode"},
            {key: value for key, value in standard_raw.items() if key != "diagnostic_observation_mode"},
        )
        self.assertEqual(standard_raw["diagnostic_observation_mode"], "standard")
        self.assertEqual(
            diagnostic_raw["diagnostic_observation_mode"],
            "persistent_calibration_bias",
        )
        diagnostic = load_training_config(DIAGNOSTIC_CONFIG_PATH)
        self.assertEqual(policy_observation_dimension(diagnostic), 85)
        self.assertEqual(policy_observation_schema_version(diagnostic), 2)
        self.assertIn("privileged", policy_observation_source(diagnostic))
        self.assertIn("privileged", policy_ground_truth_usage(diagnostic))
        self.assertEqual(
            set(privileged_diagnostic_observation_metadata(diagnostic)),
            {"diagnostic_observation_mode", "privileged_diagnostic_observation"},
        )
        self.assertEqual(
            policy_observation_ground_truth_flags(diagnostic),
            {
                "ground_truth_in_policy_observation": True,
                "object_pose_ground_truth_in_policy_observation": False,
                "privileged_simulation_latent_in_policy_observation": True,
            },
        )

    def test_target_offset_configs_change_only_action_application_mode(self):
        for base_path, target_path in (
            (CONFIG_PATH, TARGET_CONFIG_PATH),
            (DIAGNOSTIC_CONFIG_PATH, TARGET_DIAGNOSTIC_CONFIG_PATH),
        ):
            base = json.loads(base_path.read_text(encoding="utf-8"))
            target = json.loads(target_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {k: v for k, v in base.items() if k != "action_application_mode"},
                {k: v for k, v in target.items() if k != "action_application_mode"},
            )
            self.assertEqual(base["action_application_mode"], "post_integral_step")
            self.assertEqual(
                target["action_application_mode"],
                ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
            )
            loaded = load_training_config(target_path)
            self.assertEqual(
                action_application_contract_metadata(loaded),
                {
                    "action_application_mode": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
                    "policy_active_phases": ["insert"],
                },
            )

    def test_versioned_reward_profiles_and_runner_gamma_are_strict(self):
        a1 = load_training_config(A1_CONFIG_PATH)
        a2 = load_a2_with_current_task_pin()
        self.assertEqual(
            reward_profile_metadata(a1),
            {
                "reward_profile_id": "a1_discounted_pose_v2",
                "reward_potential_type": "normalized_pose",
                "reward_shaping_mode": "discounted_potential",
                "reward_horizon_semantics": "finite_horizon_timeout_absorbing",
                "reward_discount_factor": 0.995,
            },
        )
        self.assertEqual(reward_profile_metadata(a2)["reward_potential_type"], "cad_keypoints")
        validate_reward_profile_runner(a1, {"algorithm": {"gamma": 0.995}})
        validate_reward_horizon_contract(a1, is_finite_horizon=True)
        with self.assertRaisesRegex(ValueError, "is_finite_horizon=True"):
            validate_reward_horizon_contract(a1, is_finite_horizon=False)
        with self.assertRaisesRegex(ValueError, "exactly match"):
            validate_reward_profile_runner(a1, {"algorithm": {"gamma": 0.99}})
        legacy = load_training_config(TARGET_CONFIG_PATH)
        validate_reward_profile_runner(legacy, {"algorithm": {"gamma": 0.1}})
        validate_reward_horizon_contract(legacy, is_finite_horizon=False)

    def test_a2_keypoints_are_pinned_to_current_drawing_geometry(self):
        a2 = load_a2_with_current_task_pin()
        geometry = a2["reward_profile"]["keypoint_geometry"]
        points = geometry["plug_keypoints_tip_frame_m"]
        task = json.loads((ROOT / "configs/tasks/rj45-v1.json").read_text())
        profile = task["geometry"]["plug_profile"]
        self.assertEqual(len(points), 8)
        self.assertEqual(
            sorted({abs(point[0]) for point in points}),
            [profile["max_width_m"] / 2.0],
        )
        self.assertEqual(
            sorted({point[1] for point in points}),
            [-profile["nose_length_m"], 0.0],
        )
        self.assertEqual(
            sorted({abs(point[2]) for point in points}),
            [profile["terminated_nose_height_m"] / 2.0],
        )
        source = geometry["geometry_source"]
        self.assertEqual(
            source["task_config_sha256"],
            hashlib.sha256((ROOT / source["task_config_path"]).read_bytes()).hexdigest(),
        )
        self.assertEqual(
            source["plug_asset_sha256"],
            hashlib.sha256((ROOT / source["plug_asset_path"]).read_bytes()).hexdigest(),
        )
        damaged = json.loads(A2_CONFIG_PATH.read_text())
        damaged["reward_profile"]["keypoint_geometry"]["geometry_source"][
            "plug_asset_sha256"
        ] = "0" * 64
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / "a2.json"
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                load_training_config(path)

    def test_historical_a2_rejects_current_task_sha_mismatch(self):
        historical = json.loads(A2_CONFIG_PATH.read_text(encoding="utf-8"))
        source = historical["reward_profile"]["keypoint_geometry"]["geometry_source"]
        current_sha256 = hashlib.sha256(
            (ROOT / source["task_config_path"]).read_bytes()
        ).hexdigest()
        self.assertNotEqual(source["task_config_sha256"], current_sha256)
        with self.assertRaisesRegex(
            ValueError,
            "keypoint geometry source hash mismatch: configs/tasks/rj45-v1.json",
        ):
            load_training_config(A2_CONFIG_PATH)

    def test_low_exploration_ppo_config_changes_only_initial_noise(self):
        standard = json.loads((ROOT / "configs/ppo-rj45-v1.json").read_text())
        low = json.loads((ROOT / "configs/ppo-rj45-low-exploration-v1.json").read_text())
        standard_noise = standard["policy"].pop("init_noise_std")
        low_noise = low["policy"].pop("init_noise_std")
        self.assertEqual(standard, low)
        self.assertEqual(standard_noise, 0.6)
        self.assertEqual(low_noise, 0.1)

    def test_staging_rejects_ambiguous_units_unverified_duration_and_overlapping_objects(self):
        original = json.loads(CONFIG_PATH.read_text())
        variants = []
        for field, value in (("control_steps", True), ("control_steps", 2), ("object_positions_local_m", [[0, 0, 2], [0, 0, 2]]), ("object_positions_local_m", [[0, 0, 2], [1, 2]])):
            config = copy.deepcopy(original)
            config["held_reset_staging"][field] = value
            variants.append(config)
        for field, value in (("unit", "mm"), ("defined_at", "yesterday"), ("source", "hardware_measurement")):
            config = copy.deepcopy(original)
            config["held_reset_staging"]["provenance"][field] = value
            variants.append(config)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.json"
            for config in variants:
                with self.subTest(staging=config["held_reset_staging"]):
                    path.write_text(json.dumps(config))
                    with self.assertRaises(ValueError):
                        load_training_config(path)

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

    def test_target_offset_mode_rejects_both_arm_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "training.json"
            config = json.loads(TARGET_CONFIG_PATH.read_text(encoding="utf-8"))
            config["action_mode"] = "both_arms_12d"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "requires right_only_6d"):
                load_training_config(path)

    def test_rejects_bad_curriculum_and_reward_ranges(self):
        base = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        cases = (
            ("levels", lambda value: value["calibration_bias"].update(translation_levels_m=[0.00035, 0.00015, 0.0006])),
            ("initial_level", lambda value: value["calibration_bias"].update(initial_level=3)),
            ("diagnostic_mode", lambda value: value.update(diagnostic_observation_mode="sensor_bias")),
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

    def test_accepts_generation_provenance_only_when_catalog_is_source_bound(self):
        artifact = valid_artifact()
        catalog_path = "configs/rj45-grasp-generation-v2.json"
        catalog_sha256 = "1" * 64
        artifact["provenance"]["source_and_config_sha256"][catalog_path] = (
            catalog_sha256
        )
        artifact["provenance"]["grasp_generation"] = {
            "catalog_id": "rj45-grasp-generation-v2",
            "catalog_path": catalog_path,
            "catalog_sha256": catalog_sha256,
            "generation_seed": 20261301,
            "catalog_entry_id": "grasp-00",
            "assignment_group": "pilot",
            "offset_method": "object_local_offset_rotated_to_world_by_object_quaternion",
        }
        self.assertEqual(validate_held_grasp_artifact(artifact), artifact)

        artifact["provenance"]["grasp_generation"]["catalog_sha256"] = "2" * 64
        with self.assertRaisesRegex(ValueError, "must be bound"):
            validate_held_grasp_artifact(artifact)

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
