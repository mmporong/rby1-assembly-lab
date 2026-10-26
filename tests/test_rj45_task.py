import copy
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from rby1_assembly.rj45_training import (
    PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION,
    PERSISTENT_BIAS_DIAGNOSTIC_GROUND_TRUTH_USAGE,
)
from rby1_assembly.rj45_task import (
    _validate_rj45_training_contract,
    load_rj45_task,
    plug_root_goal_pose,
    relative_mating_velocity,
    relative_mating_pose,
    verify_recorded_mating_geometry,
    verify_runtime_versions,
    verify_checkpoint_lineage,
    verify_checkpoint_manifest,
    verify_generated_asset_hashes,
)


def valid_task() -> dict:
    identity = [1.0, 0.0, 0.0, 0.0]
    return {
        "schema_version": 1,
        "scope": "RJ45 simulation design assumptions; not hardware calibration",
        "parameter_provenance": "CAD-derived geometry and simulation design assumptions",
        "length_unit": "m",
        "angle_unit": "rad",
        "world_up_axis": "Z",
        "insertion_axis_world": "+Y",
        "assets": {
            "robot": "assets/generated/tasks/rj45-v1/robot.usda",
            "socket": "assets/generated/tasks/rj45-v1/socket.usda",
            "plug": "assets/generated/tasks/rj45-v1/plug.usda",
            "table": "assets/generated/tasks/rj45-v1/table.usda",
            "build_report": "assets/generated/tasks/rj45-v1/build-report.json",
        },
        "frames": {
            name: {"position_m": [0.0, 0.0, 0.0], "quaternion_wxyz": identity}
            for name in ("socket_entry", "plug_tip", "left_grasp", "right_grasp")
        },
        "table": {"top_height_m": 0.75, "size_m": [0.65, 0.9, 0.05]},
        "socket": {
            "mass_kg": 0.08,
            "initial_position_m": [0.35, 0.12, 0.78],
            "initial_orientation_wxyz": identity,
        },
        "plug": {
            "mass_kg": 0.02,
            "initial_position_m": [0.35, -0.12, 0.78],
            "initial_orientation_wxyz": identity,
        },
        "contact": {
            "static_friction": 0.8,
            "dynamic_friction": 0.6,
            "restitution": 0.0,
            "contact_offset_m": 0.0005,
            "rest_offset_m": 0.0,
            "solver_position_iteration_count": 12,
            "solver_velocity_iteration_count": 16,
            "enable_external_forces_every_iteration": False,
        },
        "evaluation": {
            "assembled_depth_m": 0.012,
            "max_lateral_error_m": 0.0005,
            "max_orientation_error_rad": math.radians(2.0),
            "max_contact_force_n": 25.0,
            "dwell_s": 0.4,
            "timeout_s": 60.0,
            "max_depth_m": 0.013,
            "max_relative_linear_speed_m_s": 0.005,
            "max_relative_angular_speed_rad_s": 0.05,
        },
        "controller": {
            "left_gripper_grasp_magnitude_m": 0.018,
            "right_gripper_grasp_magnitude_m": 0.012,
            "gripper_open_magnitude_m": 0.048,
            "residual_translation_bound_m": 0.002,
            "residual_rotation_bound_rad": math.radians(1.5),
            "alignment_standoff_m": 0.006,
            "insertion_target_margin_m": 0.002,
            "insertion_speed_m_s": 0.002,
            "insertion_max_tracking_lead_m": 0.003,
            "assembly_socket_goal_m": [0.37, 0.05, 0.9],
            "assembly_socket_goal_quaternion_wxyz": identity,
            "pregrasp_orientation_tolerance_rad": 0.05,
            "pregrasp_timeout_s": 30.0,
        },
        "reset": {"position_noise_m": [0.0, 0.0, 0.0]},
        "data": {
            "observation_source": "simulator_state_with_noise",
            "train_seed": 20261026,
            "evaluation_seed": 910261,
        },
    }


def valid_training_contract() -> dict:
    return {
        "schema_version": 1,
        "policy_observation_schema_version": 1,
        "policy_observation_dimension": 79,
        "policy_action_dimension": 6,
        "action_mode": "right_only_6d",
        "episode_scope": "held_grasp_alignment_insertion",
        "warmup_s": 0.2,
        "active_horizon_s": 24.0,
        "estimator": "frozen_kinematic_grasp_estimator",
        "calibration_bias": "episode_fixed_one_shot_se3",
        "calibration_bias_level": 0,
        "policy_pose_source": "joint_encoder_and_frozen_grasp_transform",
        "ground_truth_usage": "scripted_setup_control_and_safety_then_reward_and_independent_evaluation_only",
        "held_grasp_reset_proof": {
            "artifact_sha256": "1" * 64,
            "nominal_report_sha256": "2" * 64,
            "nominal_trace_sha256": "3" * 64,
            "selection_control_step": 900,
            "stability_limits": {
                "max_object_linear_speed_m_s": 0.005,
                "max_object_angular_speed_rad_s": 0.05,
                "max_revolute_joint_speed_rad_s": 0.05,
                "max_prismatic_joint_speed_m_s": 0.005,
                "max_revolute_tracking_error_rad": 0.05,
                "max_prismatic_tracking_error_m": 0.005,
            },
        },
        "training_config_sha256": "4" * 64,
        "parallel_contact_evidence": None,
        "learned_recovery": False,
        "retract_controller": "baseline_only",
    }


def valid_parallel_contact_evidence() -> dict:
    return {
        "schema_version": 1,
        "validated_probe_num_envs": 4,
        "runtime_num_envs_requires_native_ordering_check": True,
        "scope": (
            "4env routing proof; target num_envs runtime asserted; "
            "full task validation pending"
        ),
        "report_sha256": "5" * 64,
        "trace_sha256": "6" * 64,
        "sat_report_sha256": "7" * 64,
        "trace_row_count": 292,
        "current_source_sha256": {
            "checks/probe_parallel_hand_contacts.py": "8" * 64,
            "src/rby1_assembly/isaac_env.py": "9" * 64,
            "src/rby1_assembly/hand_collision.py": "a" * 64,
            "src/rby1_assembly/contact_loads.py": "b" * 64,
        },
    }


class Rj45TaskConfigTests(unittest.TestCase):
    def load(self, task: dict) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rj45.json"
            payload = json.dumps(task).encode("utf-8")
            path.write_bytes(payload)
            loaded = load_rj45_task(path)
            self.assertEqual(loaded["_config_sha256"], hashlib.sha256(payload).hexdigest())
            return loaded

    def test_valid_config_is_normalized_without_touching_input(self):
        task = valid_task()
        original = copy.deepcopy(task)
        loaded = self.load(task)
        self.assertEqual(task, original)
        self.assertEqual(loaded["frames"]["socket_entry"]["quaternion_wxyz"], [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(loaded["assets"]["plug"], "assets/generated/tasks/rj45-v1/plug.usda")

    def test_invalid_units_nonfinite_and_nonunit_frames_are_rejected(self):
        variants = []
        invalid_units = valid_task()
        invalid_units["length_unit"] = "mm"
        variants.append(invalid_units)
        nonfinite = valid_task()
        nonfinite["socket"]["mass_kg"] = float("nan")
        variants.append(nonfinite)
        nonunit = valid_task()
        nonunit["frames"]["plug_tip"]["quaternion_wxyz"] = [2.0, 0.0, 0.0, 0.0]
        variants.append(nonunit)
        for task in variants:
            with self.subTest(task=task):
                with self.assertRaises(ValueError):
                    self.load(task)

    def test_invalid_physics_limits_and_asset_escape_are_rejected(self):
        invalid_friction = valid_task()
        invalid_friction["contact"]["dynamic_friction"] = 0.9
        with self.assertRaises(ValueError):
            self.load(invalid_friction)
        for invalid in (True, -1, 256, 4.5, None):
            invalid_solver = valid_task()
            invalid_solver["contact"]["solver_velocity_iteration_count"] = invalid
            with self.subTest(invalid_solver=invalid), self.assertRaises(ValueError):
                self.load(invalid_solver)
        zero_velocity = valid_task()
        zero_velocity["contact"]["solver_velocity_iteration_count"] = 0
        self.load(zero_velocity)
        zero_position = valid_task()
        zero_position["contact"]["solver_position_iteration_count"] = 0
        with self.assertRaises(ValueError):
            self.load(zero_position)
        invalid_force_flag = valid_task()
        invalid_force_flag["contact"]["enable_external_forces_every_iteration"] = 1
        with self.assertRaises(ValueError):
            self.load(invalid_force_flag)
        invalid_depth = valid_task()
        invalid_depth["evaluation"]["assembled_depth_m"] = 0.0
        with self.assertRaises(ValueError):
            self.load(invalid_depth)
        escaped_asset = valid_task()
        escaped_asset["assets"]["robot"] = "../robot.usda"
        with self.assertRaises(ValueError):
            self.load(escaped_asset)

        invalid_reset = valid_task()
        invalid_reset["reset"]["position_noise_m"] = [0.0, -0.001, 0.0]
        with self.assertRaises(ValueError):
            self.load(invalid_reset)

        invalid_optional_controller = valid_task()
        invalid_optional_controller["controller"]["pregrasp_timeout_s"] = 0.0
        with self.assertRaises(ValueError):
            self.load(invalid_optional_controller)

        invalid_depth_window = valid_task()
        invalid_depth_window["evaluation"]["max_depth_m"] = 0.011
        with self.assertRaises(ValueError):
            self.load(invalid_depth_window)

    def test_approach_waypoint_group_is_atomic_and_torso_tracking_is_positive(self):
        task = valid_task()
        task["controller"].update(
            {
                "torso_work_joint_positions_rad": [0.0, 0.9, -1.8, 0.9, 0.0, 0.0],
                "initial_clearance_joint_positions_rad": [0.7, 0.0, 0.0, -2.1, 0.0, 1.4, 0.0],
                "pregrasp_clearance_joint_pos_rad": [0.1, 0.0, 0.0, -2.4, 0.0, 1.4, 0.0],
                "torso_max_joint_delta_rad": 0.015,
                "pregrasp_safe_wrist_height_m": 0.98,
                "pregrasp_waypoint_tolerance_rad": 0.08,
                "pregrasp_torso_tolerance_rad": 0.03,
                "torso_reference_tracking_tolerance_rad": 0.02,
                "left_above_joint_positions_rad": [-1.3, 0.9, -1.1, -1.8, 1.6, 1.5, 1.2],
                "right_above_joint_positions_rad": [-0.8, -1.3, 0.9, -1.8, -2.2, 1.0, -0.4],
                "approach_waypoint_max_joint_delta_rad": 0.025,
                "approach_waypoint_tracking_tolerance_rad": 0.05,
                "nullspace_posture_gain": 0.1,
                "nullspace_max_joint_delta_rad": 0.01,
                "pregrasp_integral_gain": 0.25,
                "alignment_integral_gain": 0.25,
            }
        )
        self.load(task)

        for missing in (
            "left_above_joint_positions_rad",
            "right_above_joint_positions_rad",
            "torso_work_joint_positions_rad",
            "approach_waypoint_max_joint_delta_rad",
            "approach_waypoint_tracking_tolerance_rad",
            "nullspace_posture_gain",
            "nullspace_max_joint_delta_rad",
            "pregrasp_integral_gain",
            "alignment_integral_gain",
        ):
            damaged = copy.deepcopy(task)
            del damaged["controller"][missing]
            with self.subTest(missing=missing), self.assertRaises(ValueError):
                self.load(damaged)

        for damage in ("missing", "zero"):
            damaged = copy.deepcopy(task)
            if damage == "missing":
                del damaged["controller"]["torso_reference_tracking_tolerance_rad"]
            else:
                damaged["controller"]["torso_reference_tracking_tolerance_rad"] = 0.0
            with self.subTest(torso_tracking=damage), self.assertRaises(ValueError):
                self.load(damaged)

        singleton_values = {
            "left_above_joint_positions_rad": [0.0] * 7,
            "right_above_joint_positions_rad": [0.0] * 7,
            "torso_work_joint_positions_rad": [0.0] * 6,
            "approach_waypoint_max_joint_delta_rad": 0.025,
            "approach_waypoint_tracking_tolerance_rad": 0.05,
            "nullspace_posture_gain": 0.1,
            "nullspace_max_joint_delta_rad": 0.01,
            "pregrasp_integral_gain": 0.25,
            "alignment_integral_gain": 0.25,
        }
        for singleton, value in singleton_values.items():
            damaged = valid_task()
            damaged["controller"][singleton] = value
            with self.subTest(singleton=singleton), self.assertRaises(ValueError):
                self.load(damaged)


class Rj45FrameMathTests(unittest.TestCase):
    def test_recorded_geometry_rejects_nonfinite_raw_relative_and_runtime_values(self):
        identity = [1.0, 0.0, 0.0, 0.0]
        frame = {"position_m": [0.0, 0.0, 0.0], "quaternion_wxyz": identity}
        socket_pose = [0.0, 0.0, 0.0, *identity]
        plug_pose = [0.0, 0.012, 0.0, *identity]
        verified = verify_recorded_mating_geometry(
            socket_pose, plug_pose, frame, frame, 0.012, 0.0, 0.0
        )
        self.assertEqual(verified["relative_quaternion_entry_wxyz"], identity)
        # PhysX float32 quaternions can have a slightly non-unit scalar.
        # atan2 keeps the small-angle calculation stable in that case.
        rounded_plug = [0.0, 0.012, 0.0, 0.99999994, 2.0e-7, 0.0, 0.0]
        rounded = verify_recorded_mating_geometry(
            socket_pose, rounded_plug, frame, frame, 0.012, 0.0, 4.0e-7
        )
        self.assertAlmostEqual(rounded["orientation_error_rad"], 4.0e-7, places=12)

        damaged_inputs = []
        damaged_socket = list(socket_pose)
        damaged_socket[0] = float("nan")
        damaged_inputs.append(("raw_socket", damaged_socket, plug_pose, frame, 0.012, 0.0, 0.0))
        damaged_inputs.append(("raw_plug_shape", socket_pose, plug_pose[:-1], frame, 0.012, 0.0, 0.0))
        zero_socket_quaternion = list(socket_pose)
        zero_socket_quaternion[3:] = [0.0, 0.0, 0.0, 0.0]
        damaged_inputs.append(
            ("raw_socket_zero_quaternion", zero_socket_quaternion, plug_pose, frame, 0.012, 0.0, 0.0)
        )
        nonunit_plug_quaternion = list(plug_pose)
        nonunit_plug_quaternion[3:] = [0.99, 0.0, 0.0, 0.0]
        damaged_inputs.append(
            ("raw_plug_nonunit_quaternion", socket_pose, nonunit_plug_quaternion, frame, 0.012, 0.0, 0.0)
        )
        damaged_frame = copy.deepcopy(frame)
        damaged_frame["position_m"][1] = float("inf")
        damaged_inputs.append(("relative_pose", socket_pose, plug_pose, damaged_frame, 0.012, 0.0, 0.0))
        zero_frame_quaternion = copy.deepcopy(frame)
        zero_frame_quaternion["quaternion_wxyz"] = [0.0, 0.0, 0.0, 0.0]
        damaged_inputs.append(
            ("frame_zero_quaternion", socket_pose, plug_pose, zero_frame_quaternion, 0.012, 0.0, 0.0)
        )
        nonunit_frame_quaternion = copy.deepcopy(frame)
        nonunit_frame_quaternion["quaternion_wxyz"] = [0.99998, 0.0, 0.0, 0.0]
        damaged_inputs.append(
            ("frame_nonunit_quaternion", socket_pose, plug_pose, nonunit_frame_quaternion, 0.012, 0.0, 0.0)
        )
        damaged_inputs.append(("runtime_depth", socket_pose, plug_pose, frame, float("nan"), 0.0, 0.0))
        damaged_inputs.append(("runtime_orientation", socket_pose, plug_pose, frame, 0.012, 0.0, 0.1))
        for name, socket, plug, socket_entry, depth, lateral, orientation in damaged_inputs:
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify_recorded_mating_geometry(
                    socket,
                    plug,
                    socket_entry,
                    frame,
                    depth,
                    lateral,
                    orientation,
                )

    def test_goal_solver_handles_rotated_nonidentity_local_frames(self):
        z_90 = np.array([math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)])
        x_180 = np.array([0.0, 1.0, 0.0, 0.0])
        socket_entry = {"position_m": [0.03, -0.01, 0.02], "quaternion_wxyz": z_90.tolist()}
        plug_tip = {"position_m": [0.0, 0.04, -0.005], "quaternion_wxyz": x_180.tolist()}
        socket_position = np.array([0.4, -0.2, 0.9])
        socket_quaternion = np.array([math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0])

        plug_position, plug_quaternion = plug_root_goal_pose(
            socket_position,
            socket_quaternion,
            0.012,
            socket_entry,
            plug_tip,
        )
        relative_position, relative_quaternion = relative_mating_pose(
            socket_position,
            socket_quaternion,
            plug_position,
            plug_quaternion,
            socket_entry,
            plug_tip,
        )
        np.testing.assert_allclose(relative_position, [0.0, 0.012, 0.0], atol=1.0e-12)
        np.testing.assert_allclose(relative_quaternion, [1.0, 0.0, 0.0, 0.0], atol=1.0e-12)

    def test_batch_goal_depths_round_trip(self):
        identity = [1.0, 0.0, 0.0, 0.0]
        frame = {"position_m": [0.0, 0.01, 0.0], "quaternion_wxyz": identity}
        socket_positions = np.array([[0.0, 0.0, 0.0], [0.3, -0.2, 0.9]])
        socket_quaternions = np.array([identity, identity])
        depths = np.array([0.0, 0.02])
        plug_positions, plug_quaternions = plug_root_goal_pose(
            socket_positions, socket_quaternions, depths, frame, frame
        )
        relative_positions, relative_quaternions = relative_mating_pose(
            socket_positions,
            socket_quaternions,
            plug_positions,
            plug_quaternions,
            frame,
            frame,
        )
        np.testing.assert_allclose(relative_positions[:, 1], depths, atol=1.0e-12)
        np.testing.assert_allclose(relative_positions[:, (0, 2)], 0.0, atol=1.0e-12)
        np.testing.assert_allclose(relative_quaternions, np.array([identity, identity]), atol=1.0e-12)

    def test_common_rigid_motion_has_zero_relative_mating_velocity(self):
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        socket_position = np.array([0.3, -0.2, 0.8])
        plug_position = np.array([0.32, -0.15, 0.81])
        socket_linear_velocity = np.array([0.1, -0.2, 0.05])
        angular_velocity = np.array([0.3, -0.1, 0.2])
        plug_linear_velocity = socket_linear_velocity + np.cross(
            angular_velocity, plug_position - socket_position
        )
        relative_linear, relative_angular = relative_mating_velocity(
            socket_position,
            identity,
            socket_linear_velocity,
            angular_velocity,
            plug_position,
            identity,
            plug_linear_velocity,
            angular_velocity,
            np.array([0.0, -0.03, 0.002]),
            np.array([0.0, 0.011, 0.0]),
        )
        np.testing.assert_allclose(relative_linear, 0.0, atol=1.0e-12)
        np.testing.assert_allclose(relative_angular, 0.0, atol=1.0e-12)


class Rj45ProvenanceTests(unittest.TestCase):
    def test_runtime_distribution_swap_is_rejected(self):
        runtime = json.loads(Path("configs/runtime.json").read_text())
        versions = {
            "isaacsim": runtime["isaac_sim_distribution"],
            "isaaclab": runtime["isaac_lab_distribution"],
            "isaaclab-rl": runtime["isaac_lab_rl"], "rsl-rl-lib": runtime["rsl_rl"],
            "torch": runtime["torch"] + "+" + runtime["torch_cuda_build"],
            "numpy": runtime["numpy"], "h5py": runtime["h5py"],
        }
        verify_runtime_versions(versions, runtime)
        for name in versions:
            changed = dict(versions); changed[name] = "0.0-swapped"
            with self.subTest(distribution=name), self.assertRaises(ValueError):
                verify_runtime_versions(changed, runtime)

    def test_generated_asset_hashes_detect_tampering(self):
        task = valid_task()
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            generated = project / "assets/generated/tasks/rj45-v1"
            generated.mkdir(parents=True)
            files = {}
            for filename in ("robot.usda", "socket.usda", "plug.usda", "table.usda", "scene.usda"):
                path = generated / filename
                path.write_bytes(f"asset:{filename}".encode())
                files[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            report = {"files": files}
            self.assertEqual(verify_generated_asset_hashes(project, task, report), files)
            (generated / "plug.usda").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "plug.usda"):
                verify_generated_asset_hashes(project, task, report)

    def test_checkpoint_lineage_requires_current_task_policy_and_assets(self):
        digest = {
            "configs/ppo-rj45-v1.json": "a" * 64,
            "assets/generated/tasks/rj45-v1/robot.usda": "b" * 64,
            "src/rby1_assembly/isaac_env.py": "e" * 64,
        }
        lineage = {
            "schema_version": 1,
            "task_id": "rj45-v1",
            "task_config_sha256": "c" * 64,
            "source_and_config_sha256": dict(digest),
            "rj45_nominal_evidence": {
                "report_sha256": "f" * 64,
                "trace_sha256": "1" * 64,
                "trace_records": 14,
            },
        }
        runtime = {
            "runtime_versions": {"isaacsim": "5.1.0.0", "isaaclab": "0.47.2"},
            "physics_dt_s": 1 / 120, "control_decimation": 4, "control_dt_s": 1 / 30,
            "observation_noise": {"position_std_m": 0.0005},
            "baseline_controller_state_source": "simulator_ground_truth",
            "task_config_path": "configs/tasks/rj45-v1.json",
            "policy_config_path": "configs/ppo-rj45-v1.json",
            "collision_monitor_mode": "passive_nominal_one_environment", "num_envs": 1,
        }
        lineage.update(copy.deepcopy(runtime))
        verify_checkpoint_lineage(
            lineage,
            task_id="rj45-v1",
            task_config_sha256="c" * 64,
            source_sha256=digest,
            runtime_contract=runtime,
            rj45_nominal_evidence=lineage["rj45_nominal_evidence"],
        )
        for field, mutated in (
            ("task_config_sha256", {**lineage, "task_config_sha256": "d" * 64}),
            (
                "policy",
                {**lineage, "source_and_config_sha256": {**digest, "configs/ppo-rj45-v1.json": "d" * 64}},
            ),
            (
                "asset",
                {
                    **lineage,
                    "source_and_config_sha256": {
                        **digest,
                        "assets/generated/tasks/rj45-v1/robot.usda": "d" * 64,
                    },
                },
            ),
            (
                "source",
                {
                    **lineage,
                    "source_and_config_sha256": {
                        **digest,
                        "src/rby1_assembly/isaac_env.py": "d" * 64,
                    },
                },
            ),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify_checkpoint_lineage(
                    mutated,
                    task_id="rj45-v1",
                    task_config_sha256="c" * 64,
                    source_sha256=digest,
                    runtime_contract=runtime,
                    rj45_nominal_evidence=lineage["rj45_nominal_evidence"],
                )
        for key, bad in (("runtime_versions", {"isaacsim": "0.0-swapped"}),
                         ("physics_dt_s", 999), ("control_decimation", 999),
                         ("baseline_controller_state_source", "different")):
            damaged = copy.deepcopy(lineage); damaged[key] = bad
            with self.subTest(runtime=key), self.assertRaises(ValueError):
                verify_checkpoint_lineage(damaged, task_id="rj45-v1", task_config_sha256="c" * 64,
                                          source_sha256=digest, runtime_contract=runtime,
                                          rj45_nominal_evidence=lineage["rj45_nominal_evidence"])

    def test_checkpoint_lineage_requires_exact_verified_nominal_evidence(self):
        digest = {"src/rby1_assembly/rj45_env.py": "a" * 64}
        runtime = {
            "runtime_versions": {"isaacsim": "5.1.0.0"},
            "physics_dt_s": 1 / 120,
            "control_decimation": 4,
            "control_dt_s": 1 / 30,
            "observation_noise": {"position_std_m": 0.0005},
            "baseline_controller_state_source": "simulator_ground_truth",
            "task_config_path": "configs/tasks/rj45-v1.json",
            "policy_config_path": "configs/ppo-rj45-v1.json",
            "collision_monitor_mode": "passive_nominal_one_environment",
            "num_envs": 1,
        }
        evidence = {
            "report_sha256": "b" * 64,
            "trace_sha256": "c" * 64,
            "trace_records": 14,
        }
        lineage = {
            "schema_version": 1,
            "task_id": "rj45-v1",
            "task_config_sha256": "d" * 64,
            "source_and_config_sha256": digest,
            "rj45_nominal_evidence": evidence,
            **runtime,
        }

        def verify(candidate, current=evidence):
            verify_checkpoint_lineage(
                candidate,
                task_id="rj45-v1",
                task_config_sha256="d" * 64,
                source_sha256=digest,
                runtime_contract=runtime,
                rj45_nominal_evidence=current,
            )

        verify(lineage)
        variants = {
            "missing": {key: value for key, value in lineage.items() if key != "rj45_nominal_evidence"},
            "malformed": {**lineage, "rj45_nominal_evidence": {"report_sha256": "bad"}},
            "changed_report": {
                **lineage,
                "rj45_nominal_evidence": {**evidence, "report_sha256": "e" * 64},
            },
            "changed_trace": {
                **lineage,
                "rj45_nominal_evidence": {**evidence, "trace_sha256": "e" * 64},
            },
            "changed_record_count": {
                **lineage,
                "rj45_nominal_evidence": {**evidence, "trace_records": 13},
            },
        }
        for name, candidate in variants.items():
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify(candidate)
        for name, current in (
            ("missing_current_field", {"report_sha256": "b" * 64, "trace_records": 14}),
            ("invalid_current_sha", {**evidence, "trace_sha256": "BAD"}),
            ("invalid_current_count", {**evidence, "trace_records": True}),
            ("zero_current_count", {**evidence, "trace_records": 0}),
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify(lineage, current)

    def test_privileged_85d_diagnostic_contract_is_distinct_and_strict(self):
        contract = valid_training_contract()
        contract.update(
            policy_observation_schema_version=2,
            policy_observation_dimension=85,
            diagnostic_observation_mode="persistent_calibration_bias",
            privileged_diagnostic_observation=(
                PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION
            ),
            ground_truth_usage=PERSISTENT_BIAS_DIAGNOSTIC_GROUND_TRUTH_USAGE,
        )
        _validate_rj45_training_contract(contract)
        for name, mutate in (
            ("wrong_mode", lambda value: value.update(diagnostic_observation_mode="standard")),
            ("missing_disclosure", lambda value: value.pop("privileged_diagnostic_observation")),
            ("wrong_dimension", lambda value: value.update(policy_observation_dimension=79)),
            ("wrong_schema", lambda value: value.update(policy_observation_schema_version=1)),
            (
                "false_ground_truth_claim",
                lambda value: value.update(
                    ground_truth_usage=(
                        "scripted_setup_control_and_safety_then_reward_and_"
                        "independent_evaluation_only"
                    )
                ),
            ),
        ):
            damaged = copy.deepcopy(contract)
            mutate(damaged)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate_rj45_training_contract(damaged)

    def test_insert_only_tip_target_contract_is_additive_and_strict(self):
        contract = valid_training_contract()
        contract.update(
            action_application_mode="absolute_tip_pose_target_insert_only",
            policy_active_phases=["insert"],
        )
        _validate_rj45_training_contract(contract)
        for name, mutate in (
            ("missing_phases", lambda value: value.pop("policy_active_phases")),
            ("align_enabled", lambda value: value.update(policy_active_phases=["align", "insert"])),
            ("wrong_mode", lambda value: value.update(action_application_mode="post_integral_step")),
            ("both_arms", lambda value: value.update(action_mode="both_arms_12d", policy_action_dimension=12)),
        ):
            damaged = copy.deepcopy(contract)
            mutate(damaged)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate_rj45_training_contract(damaged)

    def test_actual_v2_training_contract_schema_is_strictly_validated(self):
        contract = valid_training_contract()
        contract["parallel_contact_evidence"] = valid_parallel_contact_evidence()

        _validate_rj45_training_contract(contract)

        mutations = (
            (
                "missing_stability_limits",
                lambda item: item["held_grasp_reset_proof"].pop("stability_limits"),
            ),
            (
                "unknown_held_proof_field",
                lambda item: item["held_grasp_reset_proof"].__setitem__("unknown", True),
            ),
            (
                "missing_stability_limit",
                lambda item: item["held_grasp_reset_proof"]["stability_limits"].pop(
                    "max_object_linear_speed_m_s"
                ),
            ),
            (
                "unknown_stability_limit",
                lambda item: item["held_grasp_reset_proof"]["stability_limits"].__setitem__(
                    "unknown_m_s", 0.1
                ),
            ),
            (
                "zero_stability_limit",
                lambda item: item["held_grasp_reset_proof"]["stability_limits"].__setitem__(
                    "max_object_linear_speed_m_s", 0.0
                ),
            ),
            (
                "boolean_stability_limit",
                lambda item: item["held_grasp_reset_proof"]["stability_limits"].__setitem__(
                    "max_object_linear_speed_m_s", True
                ),
            ),
            (
                "nonfinite_stability_limit",
                lambda item: item["held_grasp_reset_proof"]["stability_limits"].__setitem__(
                    "max_object_linear_speed_m_s", math.inf
                ),
            ),
            (
                "unknown_parallel_field",
                lambda item: item["parallel_contact_evidence"].__setitem__(
                    "unknown", True
                ),
            ),
            (
                "missing_parallel_source",
                lambda item: item["parallel_contact_evidence"][
                    "current_source_sha256"
                ].pop("src/rby1_assembly/contact_loads.py"),
            ),
            (
                "boolean_trace_rows",
                lambda item: item["parallel_contact_evidence"].__setitem__(
                    "trace_row_count", True
                ),
            ),
        )
        for name, mutate in mutations:
            damaged = copy.deepcopy(contract)
            mutate(damaged)
            with self.subTest(name=name), self.assertRaises(ValueError):
                _validate_rj45_training_contract(damaged)

        sha_locations = (
            ("held_grasp_reset_proof", "artifact_sha256"),
            ("held_grasp_reset_proof", "nominal_report_sha256"),
            ("held_grasp_reset_proof", "nominal_trace_sha256"),
            ("parallel_contact_evidence", "report_sha256"),
            ("parallel_contact_evidence", "trace_sha256"),
            ("parallel_contact_evidence", "sat_report_sha256"),
        )
        for container, name in sha_locations:
            damaged = copy.deepcopy(contract)
            damaged[container][name] = "BAD"
            with self.subTest(sha=f"{container}.{name}"), self.assertRaises(ValueError):
                _validate_rj45_training_contract(damaged)
        for path in valid_parallel_contact_evidence()["current_source_sha256"]:
            damaged = copy.deepcopy(contract)
            damaged["parallel_contact_evidence"]["current_source_sha256"][path] = "BAD"
            with self.subTest(source_sha=path), self.assertRaises(ValueError):
                _validate_rj45_training_contract(damaged)

    def test_actual_v2_provenance_shape_passes_full_lineage_validation(self):
        digest = {
            "configs/ppo-rj45-v1.json": "a" * 64,
            "scripts/train.py": "b" * 64,
            "src/rby1_assembly/rj45_training_env.py": "c" * 64,
        }
        evidence = {
            "report_sha256": "d" * 64,
            "trace_sha256": "e" * 64,
            "trace_records": 981,
        }
        training_contract = valid_training_contract()
        training_contract["parallel_contact_evidence"] = valid_parallel_contact_evidence()
        runtime = {
            "runtime_versions": {
                "isaacsim": "5.1.0.0",
                "isaaclab": "0.47.2",
                "torch": "2.7.0+cu128",
            },
            "physics_dt_s": 1 / 120,
            "control_decimation": 4,
            "control_dt_s": 1 / 30,
            "observation_noise": {
                "distribution": (
                    "independent_zero_mean_gaussian_position_and_rotation_vector"
                ),
                "position_std_m": 0.0005,
                "rotation_vector_std_rad": 0.002,
            },
            "baseline_controller_state_source": (
                "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
            ),
            "task_config_path": "configs/tasks/rj45-v1.json",
            "policy_config_path": "configs/ppo-rj45-v1.json",
            "collision_monitor_mode": "passive_parallel_native_ordering_verified",
            "num_envs": 64,
            "rj45_training_contract": training_contract,
        }
        lineage = {
            "schema_version": 1,
            "seed": 20261026,
            "task_id": "rj45-v1",
            "task_config_sha256": "f" * 64,
            "source_and_config_sha256": digest,
            "checkpoint_scope": "checkpoints created in this run directory",
            "rj45_nominal_evidence": evidence,
            **runtime,
        }

        verify_checkpoint_lineage(
            lineage,
            task_id="rj45-v1",
            task_config_sha256="f" * 64,
            source_sha256=digest,
            runtime_contract=runtime,
            rj45_nominal_evidence=evidence,
        )

    def test_checkpoint_lineage_requires_exact_rj45_training_contract(self):
        digest = {"src/rby1_assembly/rj45_training_env.py": "a" * 64}
        evidence = {
            "report_sha256": "b" * 64,
            "trace_sha256": "c" * 64,
            "trace_records": 14,
        }
        training_contract = valid_training_contract()
        runtime = {
            "runtime_versions": {"isaacsim": "5.1.0.0"},
            "physics_dt_s": 1 / 120,
            "control_decimation": 4,
            "control_dt_s": 1 / 30,
            "observation_noise": {"position_std_m": 0.0005},
            "baseline_controller_state_source": (
                "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
            ),
            "task_config_path": "configs/tasks/rj45-v1.json",
            "policy_config_path": "configs/ppo-rj45-v1.json",
            "collision_monitor_mode": "passive_parallel_native_ordering_verified",
            "num_envs": 64,
            "rj45_training_contract": training_contract,
        }
        lineage = {
            "schema_version": 1,
            "task_id": "rj45-v1",
            "task_config_sha256": "d" * 64,
            "source_and_config_sha256": digest,
            "rj45_nominal_evidence": evidence,
            **runtime,
        }

        def verify(candidate, expected=runtime):
            verify_checkpoint_lineage(
                candidate,
                task_id="rj45-v1",
                task_config_sha256="d" * 64,
                source_sha256=digest,
                runtime_contract=expected,
                rj45_nominal_evidence=evidence,
            )

        verify(lineage)
        for name, mutate in (
            ("missing", lambda item: item.pop("rj45_training_contract")),
            (
                "observation_dimension",
                lambda item: item["rj45_training_contract"].__setitem__(
                    "policy_observation_dimension", 76
                ),
            ),
            (
                "action_mode_dimension",
                lambda item: item["rj45_training_contract"].__setitem__(
                    "policy_action_dimension", 12
                ),
            ),
            (
                "training_config_sha",
                lambda item: item["rj45_training_contract"].__setitem__(
                    "training_config_sha256", "e" * 64
                ),
            ),
            (
                "held_proof",
                lambda item: item["rj45_training_contract"]["held_grasp_reset_proof"].__setitem__(
                    "selection_control_step", 901
                ),
            ),
        ):
            damaged = copy.deepcopy(lineage)
            mutate(damaged)
            with self.subTest(name=name), self.assertRaises(ValueError):
                verify(damaged)

        for num_envs, monitor in (
            (0, "passive_nominal_one_environment"),
            (1, "passive_parallel_native_ordering_verified"),
            (64, "passive_nominal_one_environment"),
        ):
            damaged = copy.deepcopy(lineage)
            damaged["num_envs"] = num_envs
            damaged["collision_monitor_mode"] = monitor
            with self.subTest(num_envs=num_envs, monitor=monitor), self.assertRaises(ValueError):
                verify(damaged)

    def test_checkpoint_manifest_rejects_a_swapped_checkpoint(self):
        manifest = {
            "schema_version": 1,
            "training_completed": True,
            "training_provenance_sha256": "a" * 64,
            "checkpoints": {"model_100.pt": "b" * 64},
        }
        verify_checkpoint_manifest(
            manifest,
            training_provenance_sha256="a" * 64,
            checkpoint_name="model_100.pt",
            checkpoint_sha256="b" * 64,
        )
        with self.assertRaises(ValueError):
            verify_checkpoint_manifest(
                manifest,
                training_provenance_sha256="a" * 64,
                checkpoint_name="model_100.pt",
                checkpoint_sha256="c" * 64,
            )


if __name__ == "__main__":
    unittest.main()
