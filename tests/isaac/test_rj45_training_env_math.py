import ast
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from rby1_assembly.rj45_training_env import (
    HeldGraspWarmupTimeoutError,
    HeldResetContactFlushError,
    _calibration_bias_case_seed,
    _held_reset_contact_flush_diagnostics,
    _held_reset_contact_free_status,
    _held_reset_stage_transition,
    _held_warmup_timeout_diagnostics,
    _mask_inactive_policy_observation,
    _reset_policy_input_telemetry,
    _reset_applied_tip_offset_telemetry,
    _store_policy_input_telemetry,
    _warmup_motion_status,
    _warmup_sensor_status,
    _warmup_timeout_mask,
)
from rby1_assembly.rj45_training import (
    ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
    POST_INTEGRAL_STEP_ACTION_APPLICATION,
    apply_tip_target_pose_offset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ENV_PATH = PROJECT_ROOT / "src/rby1_assembly/rj45_training_env.py"
ISAAC_ENV_PATH = PROJECT_ROOT / "src/rby1_assembly/isaac_env.py"


def _compiled_training_environment_method(name):
    tree = ast.parse(TRAINING_ENV_PATH.read_text(encoding="utf-8"))
    factory = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "make_rj45_training_env_type"
    )
    wrapper = next(node for node in factory.body if isinstance(node, ast.ClassDef))
    method = next(
        node
        for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {
        "torch": torch,
        "HeldResetContactFlushError": HeldResetContactFlushError,
        "_held_reset_contact_flush_diagnostics": (
            _held_reset_contact_flush_diagnostics
        ),
        "_held_reset_contact_free_status": _held_reset_contact_free_status,
        "_held_reset_stage_transition": _held_reset_stage_transition,
        "_calibration_bias_case_seed": _calibration_bias_case_seed,
        "ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
        "POST_INTEGRAL_STEP_ACTION_APPLICATION": POST_INTEGRAL_STEP_ACTION_APPLICATION,
        "AssemblyPhase": SimpleNamespace(ALIGN=4, INSERT=5),
        "apply_tip_target_pose_offset": apply_tip_target_pose_offset,
        "combine_frame_transforms": lambda base_pos, base_quat, local_pos, local_quat: (
            base_pos + local_pos,
            base_quat,
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            TRAINING_ENV_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def _compiled_base_environment_method(name):
    tree = ast.parse(ISAAC_ENV_PATH.read_text(encoding="utf-8"))
    wrapper = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Rby1AssemblyEnv"
    )
    method = next(
        node for node in wrapper.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace = {"torch": torch}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
            ISAAC_ENV_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


STABILITY_LIMITS = {
    "max_object_linear_speed_m_s": 0.005,
    "max_object_angular_speed_rad_s": 0.05,
    "max_revolute_joint_speed_rad_s": 0.05,
    "max_prismatic_joint_speed_m_s": 0.005,
    "max_revolute_tracking_error_rad": 0.05,
    "max_prismatic_tracking_error_m": 0.005,
}


class Rj45TrainingEnvironmentMathTests(unittest.TestCase):
    def test_default_mating_target_offset_hook_is_no_op(self):
        hook = _compiled_base_environment_method(
            "_apply_policy_mating_target_pose_offsets"
        )
        position = torch.randn(3, 3)
        quaternion = torch.randn(3, 4)
        returned_position, returned_quaternion = hook(
            SimpleNamespace(), position, quaternion
        )
        self.assertIs(returned_position, position)
        self.assertIs(returned_quaternion, quaternion)

    def test_target_offset_mode_disables_parent_twist_residual(self):
        residual = _compiled_training_environment_method("_residual_actions_world")
        actions = torch.randn(4, 12)
        environment = SimpleNamespace(
            _training_initialized=True,
            _training_config={
                "action_application_mode": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            }
        )
        torch.testing.assert_close(residual(environment, actions), torch.zeros_like(actions))

    def test_target_offset_hook_scales_right_action_and_changes_active_rows_only(self):
        hook = _compiled_training_environment_method(
            "_apply_policy_mating_target_pose_offsets"
        )
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1)
        plug_goal = torch.tensor([[0.4, 0.1, 0.8], [0.2, -0.1, 0.7]])
        plug_quat = identity.clone()
        actions = torch.zeros((2, 12))
        actions[0, 6:12] = torch.tensor([1.0, -0.5, 0.25, 0.0, 0.0, 1.0])
        environment = SimpleNamespace(
            num_envs=2,
            device="cpu",
            _training_initialized=True,
            _training_config={
                "action_application_mode": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            },
            _training_applied_tip_offset_entry=torch.full((2, 6), 99.0),
            _actions=actions,
            cfg=SimpleNamespace(
                residual_translation_bound_m=0.002,
                residual_rotation_bound_rad=0.03,
            ),
            _policy_active_mask=lambda: torch.tensor([True, False]),
            _control_object_poses=lambda: (
                torch.zeros((2, 3)), identity, torch.zeros((2, 3)), identity
            ),
            _frame_tensors=lambda name: (
                torch.tensor([[0.0, 0.02, 0.0]]).repeat(2, 1)
                if name == "plug_tip" else torch.zeros((2, 3)),
                identity,
            ),
        )
        target_position, target_quaternion = hook(
            environment, plug_goal, plug_quat
        )
        expected_offset = torch.tensor(
            [[0.002, -0.001, 0.0005, 0.0, 0.0, 0.03], [0.0] * 6]
        )
        expected_position, expected_quaternion = apply_tip_target_pose_offset(
            plug_goal,
            plug_quat,
            identity,
            torch.tensor([[0.0, 0.02, 0.0]]).repeat(2, 1),
            identity,
            expected_offset,
        )
        torch.testing.assert_close(target_position[0], expected_position[0])
        torch.testing.assert_close(target_quaternion[0], expected_quaternion[0])
        torch.testing.assert_close(target_position[1], plug_goal[1])
        torch.testing.assert_close(target_quaternion[1], plug_quat[1])
        torch.testing.assert_close(
            environment._training_applied_tip_offset_entry, expected_offset
        )
    def test_insert_only_policy_activation_and_lifetime_evaluation_eligibility(self):
        active_method = _compiled_training_environment_method("_policy_active_mask")
        eligible_method = _compiled_training_environment_method(
            "_evaluation_eligibility_mask"
        )
        environment = SimpleNamespace(
            _training_config={
                "action_application_mode": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            },
            _training_initialized=True,
            _training_policy_activated=torch.tensor([True, True, False]),
            _phase=torch.tensor([4, 5, 5]),
            _training_valid_policy_input_step_count=torch.tensor([0, 2, 3]),
        )
        self.assertEqual(active_method(environment).tolist(), [False, True, False])
        self.assertEqual(eligible_method(environment).tolist(), [False, True, False])

        environment._training_config["action_application_mode"] = (
            POST_INTEGRAL_STEP_ACTION_APPLICATION
        )
        self.assertEqual(active_method(environment).tolist(), [True, True, False])
        self.assertEqual(eligible_method(environment).tolist(), [True, True, False])

    def test_applied_tip_offset_partial_reset_preserves_other_rows(self):
        telemetry = torch.arange(18, dtype=torch.float32).reshape(3, 6)
        first = telemetry[0].clone()
        last = telemetry[2].clone()
        _reset_applied_tip_offset_telemetry(torch.tensor([1]), telemetry)
        torch.testing.assert_close(telemetry[0], first)
        torch.testing.assert_close(telemetry[1], torch.zeros(6))
        torch.testing.assert_close(telemetry[2], last)
    def test_held_controller_phase_keeps_torso_work_reference_and_other_rows(self):
        from rby1_assembly.rj45_env import advance_joint_reference

        work = torch.tensor([0.0, 0.9, -1.8, 0.9, 0.0, 0.0])
        environment = SimpleNamespace(
            _training_held_state={"selection": {"phase_name": "insert"}},
            _training_held_reference=work.repeat(3, 1),
            _torso_joint_ids=list(range(6)),
            cfg=SimpleNamespace(rj45_task={"controller": {"torso_work_joint_positions_rad": work.tolist()}}),
            device="cpu",
            _initial_clearance_reached=torch.tensor([False, False, True]),
            _work_clearance_reached=torch.tensor([False, False, True]),
            _torso_progress=torch.tensor([0.4, 0.0, 0.6]),
            _approach_progress=torch.tensor([[0.4, 0.5], [0.0, 0.0], [0.6, 0.7]]),
            _grip_proven=torch.zeros((3, 2), dtype=torch.bool),
        )
        _, before = advance_joint_reference(
            work.repeat(3, 1), torch.zeros(6), work,
            environment._torso_progress, environment._work_clearance_reached,
            0.015, 0.05,
        )
        self.assertTrue(torch.equal(before[1], torch.zeros(6)))
        initialize = _compiled_training_environment_method("_initialize_held_controller_phase")
        initialize(environment, torch.tensor([1]))
        _, after = advance_joint_reference(
            work.repeat(3, 1), torch.zeros(6), work,
            environment._torso_progress, environment._work_clearance_reached,
            0.015, 0.05,
        )
        self.assertTrue(torch.equal(after[1], work))
        self.assertTrue(torch.equal(environment._torso_progress, torch.tensor([0.4, 1.0, 0.6])))
        self.assertTrue(torch.equal(environment._approach_progress, torch.tensor([[0.4, 0.5], [1.0, 1.0], [0.6, 0.7]])))
        self.assertEqual(environment._initial_clearance_reached.tolist(), [False, True, True])
        self.assertFalse(environment._grip_proven.any())

    def test_held_controller_phase_rejects_wrong_stage_or_torso_reference(self):
        initialize = _compiled_training_environment_method("_initialize_held_controller_phase")
        environment = SimpleNamespace(_training_held_state={"selection": {"phase_name": "close"}})
        with self.assertRaises(ValueError):
            initialize(environment, torch.tensor([0]))
        environment._training_held_state["selection"]["phase_name"] = "insert"
        environment._training_held_reference = torch.zeros((1, 6))
        environment._torso_joint_ids = list(range(6))
        environment.device = "cpu"
        environment.cfg = SimpleNamespace(rj45_task={"controller": {"torso_work_joint_positions_rad": [0.0, 0.9, -1.8, 0.9, 0.0, 0.0]}})
        with self.assertRaises(ValueError):
            initialize(environment, torch.tensor([0]))

    def test_contact_flush_requires_finite_exact_zero_force_load_and_counts(self):
        zeros = {
            "finger_force_n": torch.zeros((8, 4)),
            "finger_raw_contact_count": torch.zeros((8, 4), dtype=torch.long),
            "insertion_force_world_n": torch.zeros((8, 3)),
            "insertion_load_n": torch.zeros(8),
            "insertion_raw_contact_count": torch.zeros(8, dtype=torch.long),
            "forbidden_load_n": torch.zeros(8),
        }
        zeros["finger_force_n"][1, 0] = 1.0e-12
        zeros["finger_raw_contact_count"][2, 1] = 1
        zeros["insertion_force_world_n"][3, 2] = float("nan")
        zeros["insertion_force_world_n"][4, 2] = 1.0e-12
        zeros["insertion_load_n"][5] = 1.0e-12
        zeros["insertion_raw_contact_count"][6] = 1
        zeros["forbidden_load_n"][7] = 1.0e-12

        finite, exactly_zero, passed = _held_reset_contact_free_status(**zeros)

        self.assertEqual(
            finite.tolist(),
            [True, True, True, False, True, True, True, True],
        )
        self.assertEqual(
            exactly_zero.tolist(),
            [True, False, False, False, False, False, False, False],
        )
        self.assertEqual(
            passed.tolist(),
            [True, False, False, False, False, False, False, False],
        )

    def test_contact_flush_error_is_json_safe_with_nonfinite_readings(self):
        failed = torch.tensor([False, True])
        diagnostics = _held_reset_contact_flush_diagnostics(
            failed,
            torch.tensor([[0.0] * 4, [float("nan"), 0.0, 0.0, 0.0]]),
            torch.tensor([[0] * 4, [1, 0, 0, 0]]),
            torch.tensor([[0.0] * 3, [float("inf"), 0.0, 0.0]]),
            torch.tensor([0.0, float("-inf")]),
            torch.tensor([0, 1]),
            torch.tensor([0.0, float("nan")]),
        )
        error = HeldResetContactFlushError(diagnostics)
        encoded = str(error).split(": ", 1)[1]
        parsed = json.loads(encoded)

        self.assertEqual(parsed, error.diagnostics)
        json.dumps(parsed, allow_nan=False)
        self.assertEqual(parsed["failed_environment_ids"], [1])
        self.assertEqual(
            parsed["environments"][0]["finger_force_n"][0],
            {"value": "nan", "nonfinite": True},
        )
        self.assertEqual(
            parsed["environments"][0]["insertion_load_n"],
            {"value": "-inf", "nonfinite": True},
        )

    def test_actual_contact_flush_restores_only_passing_ready_rows(self):
        restored = []
        environment = SimpleNamespace(
            _training_held_state={"verified": True},
            _training_config={"held_reset_staging": {"control_steps": 1}},
            _training_held_reset_stage_queued=torch.tensor([True, True, False]),
            _training_held_reset_stage_restored=torch.tensor([False, False, True]),
            _training_held_reset_stage_waited_control_steps=torch.tensor([1, 1, 7]),
            _training_held_reset_validation_passed=torch.tensor([False, False, True]),
            _training_held_reset_validation_finger_force_n=torch.full((3, 4), 9.0),
            _training_held_reset_validation_raw_contact_count=torch.full(
                (3, 5), 9, dtype=torch.long
            ),
            _training_held_reset_validation_insertion_force_world_n=torch.full(
                (3, 3), 9.0
            ),
            _training_held_reset_validation_insertion_load_n=torch.full((3,), 9.0),
            _training_held_reset_validation_forbidden_load_n=torch.full((3,), 9.0),
            _held_reset_contact_flush_state=lambda: (
                torch.tensor(
                    [[0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.0, 0.0], [4.0] * 4]
                ),
                torch.tensor(
                    [[0, 0, 0, 0], [0, 2, 0, 0], [4, 4, 4, 4]],
                    dtype=torch.int32,
                ),
                torch.zeros((3, 3)),
                torch.zeros(3),
                torch.tensor([0, 3, 4], dtype=torch.int32),
                torch.zeros(3),
            ),
            _restore_held_physical_state=lambda env_ids: restored.append(
                env_ids.clone()
            ),
        )
        advance = _compiled_training_environment_method(
            "_advance_held_reset_staging"
        )

        with self.assertRaises(HeldResetContactFlushError) as raised:
            advance(environment)

        self.assertEqual(restored[0].tolist(), [0])
        self.assertEqual(
            environment._training_held_reset_stage_queued.tolist(),
            [False, True, False],
        )
        self.assertEqual(
            environment._training_held_reset_stage_restored.tolist(),
            [True, False, True],
        )
        self.assertEqual(
            environment._training_held_reset_validation_passed.tolist(),
            [True, False, True],
        )
        self.assertTrue(
            torch.equal(
                environment._training_held_reset_validation_finger_force_n[2],
                torch.full((4,), 9.0),
            )
        )
        self.assertTrue(
            torch.equal(
                environment._training_held_reset_validation_raw_contact_count[2],
                torch.full((5,), 9, dtype=torch.long),
            )
        )
        self.assertEqual(
            environment._training_held_reset_validation_raw_contact_count.dtype,
            torch.int64,
        )
        self.assertEqual(
            environment._training_held_reset_validation_raw_contact_count[1].tolist(),
            [0, 2, 0, 0, 3],
        )
        self.assertEqual(raised.exception.diagnostics["failed_environment_ids"], [1])

    def test_held_reset_staging_waits_one_complete_control_step(self):
        queued = torch.tensor([True, True, False])
        waited = torch.tensor([0, 1, 0])

        ready, next_waited = _held_reset_stage_transition(
            queued,
            waited,
            control_steps=1,
        )

        self.assertEqual(ready.tolist(), [False, True, False])
        self.assertEqual(next_waited.tolist(), [1, 1, 0])
        queued[ready] = False
        ready, next_waited = _held_reset_stage_transition(
            queued,
            next_waited,
            control_steps=1,
        )
        self.assertEqual(ready.tolist(), [True, False, False])
        self.assertEqual(next_waited.tolist(), [1, 1, 0])

    def test_held_reset_staging_rejects_invalid_contracts(self):
        with self.assertRaises(ValueError):
            _held_reset_stage_transition(
                torch.tensor([True]),
                torch.tensor([0]),
                control_steps=0,
            )
        with self.assertRaises(ValueError):
            _held_reset_stage_transition(
                torch.tensor([1]),
                torch.tensor([0]),
                control_steps=1,
            )
        with self.assertRaises(ValueError):
            _held_reset_stage_transition(
                torch.tensor([True, False]),
                torch.tensor([0]),
                control_steps=1,
            )

    def test_policy_observation_mask_hides_all_inactive_training_scopes(self):
        observation = torch.arange(3 * 79, dtype=torch.float32).reshape(3, 79)
        observation[1, 78] = 1.0
        masked, valid = _mask_inactive_policy_observation(
            observation,
            torch.tensor([False, True, False]),
        )

        self.assertEqual(valid.tolist(), [False, True, False])
        self.assertTrue(torch.equal(masked[0], torch.zeros(79)))
        self.assertTrue(torch.equal(masked[1], observation[1]))
        self.assertTrue(torch.equal(masked[2], torch.zeros(79)))
        self.assertEqual(masked[1, 78].item(), 1.0)

    def test_policy_observation_mask_rejects_wrong_schema(self):
        with self.assertRaises(ValueError):
            _mask_inactive_policy_observation(
                torch.zeros((2, 78)),
                torch.tensor([False, True]),
            )
        with self.assertRaises(ValueError):
            _mask_inactive_policy_observation(
                torch.zeros((2, 79)),
                torch.tensor([True]),
            )

    def test_privileged_85d_observation_mask_zeros_full_inactive_rows(self):
        observation = torch.arange(3 * 85, dtype=torch.float32).reshape(3, 85)
        observation[1, 78] = 1.0
        masked, valid = _mask_inactive_policy_observation(
            observation,
            torch.tensor([False, True, False]),
            85,
        )
        self.assertEqual(valid.tolist(), [False, True, False])
        self.assertTrue(torch.equal(masked[0], torch.zeros(85)))
        self.assertTrue(torch.equal(masked[1], observation[1]))
        self.assertTrue(torch.equal(masked[2], torch.zeros(85)))
        self.assertEqual(masked[1, 78].item(), 1.0)

    def test_episode_bias_partial_reset_mutates_only_target_rows(self):
        sample = _compiled_training_environment_method("_sample_episode_bias")
        environment = SimpleNamespace(
            _training_translation_bias_entry=torch.tensor(
                [[0.1, 0.2, 0.3], [9.0, 9.0, 9.0], [0.4, 0.5, 0.6]]
            ),
            _training_rotation_bias_entry=torch.tensor(
                [[0.7, 0.8, 0.9], [9.0, 9.0, 9.0], [1.0, 1.1, 1.2]]
            ),
            _training_bias_enabled=True,
            _training_config={
                "calibration_bias": {
                    "translation_levels_m": [0.00015],
                    "rotation_max_levels_rad": [0.004],
                }
            },
            _training_bias_level=0,
            _training_episode_index=torch.tensor([4, 5, 6]),
            _training_bias_base_seed=17,
            device=torch.device("cpu"),
        )
        preserved_translation = environment._training_translation_bias_entry[[0, 2]].clone()
        preserved_rotation = environment._training_rotation_bias_entry[[0, 2]].clone()
        sample(environment, torch.tensor([1]))
        torch.testing.assert_close(
            environment._training_translation_bias_entry[[0, 2]],
            preserved_translation,
        )
        torch.testing.assert_close(
            environment._training_rotation_bias_entry[[0, 2]],
            preserved_rotation,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(environment._training_translation_bias_entry[1])),
            0.00015,
        )
        self.assertLessEqual(
            float(torch.linalg.vector_norm(environment._training_rotation_bias_entry[1])),
            0.004,
        )

    def test_policy_input_telemetry_stores_the_exact_masked_input(self):
        raw = torch.arange(2 * 79, dtype=torch.float32).reshape(2, 79)
        raw[1, 78] = 1.0
        masked, valid = _mask_inactive_policy_observation(
            raw,
            torch.tensor([False, True]),
        )
        observation_buffer = torch.full_like(masked, -9.0)
        valid_buffer = torch.zeros(2, dtype=torch.bool)
        timestamp_buffer = torch.full((2,), -1.0)

        _store_policy_input_telemetry(
            masked,
            valid,
            12.5,
            observation_buffer,
            valid_buffer,
            timestamp_buffer,
        )
        expected = masked.clone()
        masked.fill_(123.0)

        self.assertTrue(torch.equal(observation_buffer, expected))
        self.assertTrue(torch.equal(observation_buffer[0], torch.zeros(79)))
        self.assertTrue(torch.equal(valid_buffer, torch.tensor([False, True])))
        self.assertTrue(torch.equal(timestamp_buffer, torch.tensor([12.5, 12.5])))

    def test_policy_input_telemetry_partial_reset_preserves_other_rows(self):
        observation_buffer = torch.arange(
            3 * 79, dtype=torch.float32
        ).reshape(3, 79)
        valid_buffer = torch.tensor([True, True, True])
        timestamp_buffer = torch.tensor([1.0, 2.0, 3.0])
        preserved_first = observation_buffer[0].clone()
        preserved_last = observation_buffer[2].clone()

        _reset_policy_input_telemetry(
            torch.tensor([1]),
            observation_buffer,
            valid_buffer,
            timestamp_buffer,
        )

        self.assertTrue(torch.equal(observation_buffer[0], preserved_first))
        self.assertTrue(torch.equal(observation_buffer[1], torch.zeros(79)))
        self.assertTrue(torch.equal(observation_buffer[2], preserved_last))
        self.assertEqual(valid_buffer.tolist(), [True, False, True])
        self.assertEqual(timestamp_buffer.tolist(), [1.0, -1.0, 3.0])

    def test_actual_step_snapshot_clones_policy_input_and_action_telemetry(self):
        snapshot_method = _compiled_training_environment_method(
            "_training_step_snapshot"
        )
        environment = SimpleNamespace(
            _training_held_initialization_passed=torch.tensor([False, True]),
            _training_valid_policy_input_step_count=torch.tensor([0, 10]),
            _training_last_policy_observation=torch.arange(
                2 * 79, dtype=torch.float32
            ).reshape(2, 79),
            _training_last_policy_observation_timestamp_s=torch.tensor([1.0, 2.0]),
            _training_last_policy_observation_valid=torch.tensor([False, True]),
            _training_estimator_valid=torch.tensor([False, True]),
            _training_policy_activated=torch.tensor([False, True]),
            _training_previous_action=torch.arange(
                24, dtype=torch.float32
            ).reshape(2, 12),
            _actions=torch.full((2, 12), 0.25),
            _training_applied_tip_offset_entry=torch.arange(
                12, dtype=torch.float32
            ).reshape(2, 6),
        )

        snapshot = snapshot_method(environment)
        environment._training_last_policy_observation.fill_(-1.0)
        environment._training_previous_action.fill_(-1.0)
        environment._actions.fill_(-1.0)
        environment._training_held_initialization_passed.fill_(False)
        environment._training_valid_policy_input_step_count.zero_()

        self.assertEqual(
            set(snapshot),
            {
                "held_initialization_passed",
                "valid_policy_input_step_count",
                "policy_input_observation",
                "policy_input_timestamp_s",
                "policy_input_valid",
                "training_estimator_valid",
                "training_policy_activated",
                "requested_expanded_residual_action",
                "applied_bounded_residual_action",
                "applied_tip_target_offset_entry_6d",
            },
        )
        self.assertEqual(snapshot["policy_input_observation"][1, 78].item(), 157.0)
        self.assertEqual(snapshot["held_initialization_passed"].tolist(), [False, True])
        self.assertEqual(snapshot["valid_policy_input_step_count"].tolist(), [0, 10])
        self.assertEqual(
            snapshot["requested_expanded_residual_action"][1, 11].item(),
            23.0,
        )
        self.assertEqual(
            snapshot["applied_bounded_residual_action"][1, 11].item(),
            0.25,
        )
        self.assertEqual(
            snapshot["applied_tip_target_offset_entry_6d"][1, 5].item(), 11.0
        )

    def test_policy_attribution_counts_valid_inputs_and_survives_terminal_grip_loss(self):
        environment = SimpleNamespace(
            _training_last_policy_observation_valid=torch.tensor([True, False, True]),
            _training_held_initialization_passed=torch.tensor([True, True, True]),
            _training_valid_policy_input_step_count=torch.tensor([4, 9, 12]),
            _grip_proven=torch.zeros((3, 2), dtype=torch.bool),
        )
        count_step = _compiled_training_environment_method("_count_policy_input_step")
        count_step(environment)
        self.assertEqual(environment._training_valid_policy_input_step_count.tolist(), [5, 9, 13])
        self.assertEqual(environment._training_held_initialization_passed.tolist(), [True] * 3)
        reset = _compiled_training_environment_method("_reset_policy_attribution")
        reset(environment, torch.tensor([1]))
        self.assertEqual(environment._training_valid_policy_input_step_count.tolist(), [5, 0, 13])
        self.assertEqual(environment._training_held_initialization_passed.tolist(), [True, False, True])

    def test_actual_restore_helper_writes_recorded_velocities_unchanged(self):
        calls = []

        class Recorder:
            def __init__(self, name):
                self.name = name

            def write_root_pose_to_sim(self, value, env_ids):
                calls.append((self.name, "pose", value.clone(), env_ids.clone()))

            def write_root_velocity_to_sim(self, value, env_ids):
                calls.append((self.name, "velocity", value.clone(), env_ids.clone()))

            def write_joint_state_to_sim(self, position, velocity, effort, env_ids):
                calls.append(
                    (
                        self.name,
                        "joint",
                        position.clone(),
                        velocity.clone(),
                        effort,
                        env_ids.clone(),
                    )
                )

        robot = Recorder("robot")
        robot.data = SimpleNamespace(
            default_root_state=torch.tensor(
                [
                    [0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                    [0.0] * 13,
                ]
            )
        )
        joint_position = torch.tensor([[1.0, 2.0]])
        joint_velocity = torch.tensor([[0.25, -0.5]])
        object_pose = torch.tensor(
            [
                [
                    [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
                    [4.0, 5.0, 6.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        )
        object_velocity = torch.tensor(
            [[[0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6]]]
        )
        environment = SimpleNamespace(
            robot=robot,
            socket=Recorder("socket"),
            plug=Recorder("plug"),
            scene=SimpleNamespace(
                env_origins=torch.tensor(
                    [[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]]
                )
            ),
            _assert_held_state_write_allowed=lambda env_ids: None,
            _held_artifact_joint_state=lambda env_ids: (
                joint_position,
                joint_velocity,
                torch.zeros_like(joint_position),
            ),
            _held_artifact_object_state=lambda env_ids: (
                object_pose,
                object_velocity,
            ),
        )

        result = _compiled_training_environment_method(
            "_restore_held_physical_state"
        )(environment, torch.tensor([0]))

        self.assertTrue(torch.equal(result[0], joint_position))
        self.assertTrue(torch.equal(result[1], joint_velocity))
        joint_call = next(call for call in calls if call[1] == "joint")
        self.assertTrue(torch.equal(joint_call[2], joint_position))
        self.assertTrue(torch.equal(joint_call[3], joint_velocity))
        socket_velocity_call = next(
            call for call in calls if call[0:2] == ("socket", "velocity")
        )
        plug_velocity_call = next(
            call for call in calls if call[0:2] == ("plug", "velocity")
        )
        self.assertTrue(torch.equal(socket_velocity_call[2], object_velocity[:, 0]))
        self.assertTrue(torch.equal(plug_velocity_call[2], object_velocity[:, 1]))

    def test_actual_stage_queue_uses_configured_far_poses_and_only_target_rows(self):
        written = []

        class ObjectRecorder:
            def __init__(self, name):
                self.name = name

            def write_root_pose_to_sim(self, pose, env_ids):
                written.append((self.name, pose.clone(), env_ids.clone()))

        object_pose = torch.tensor(
            [
                [
                    [0.0, 0.0, 0.0, 1.0, 0.1, 0.2, 0.3],
                    [0.0, 0.0, 0.0, 1.0, -0.1, -0.2, -0.3],
                ]
            ]
        )
        environment = SimpleNamespace(
            device=torch.device("cpu"),
            socket=ObjectRecorder("socket"),
            plug=ObjectRecorder("plug"),
            scene=SimpleNamespace(
                env_origins=torch.tensor(
                    [[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]]
                )
            ),
            _training_config={
                "held_reset_staging": {
                    "object_positions_local_m": [
                        [0.8, 0.8, 2.5],
                        [1.0, 0.8, 2.5],
                    ]
                }
            },
            _assert_held_state_write_allowed=lambda env_ids: None,
            _held_artifact_object_state=lambda env_ids: (
                object_pose,
                torch.zeros((1, 2, 6)),
            ),
            _training_held_reset_stage_queued=torch.tensor([False, False]),
            _training_held_reset_stage_restored=torch.tensor([True, True]),
            _training_held_reset_stage_waited_control_steps=torch.tensor([8, 9]),
        )

        _compiled_training_environment_method("_queue_held_reset_stage")(
            environment,
            torch.tensor([1]),
        )

        self.assertTrue(
            torch.allclose(
                torch.stack([record[1][0, :3] for record in written]),
                torch.tensor([[10.8, 20.8, 32.5], [11.0, 20.8, 32.5]]),
            )
        )
        self.assertTrue(torch.equal(written[0][1][0, 3:], object_pose[0, 0, 3:]))
        self.assertTrue(torch.equal(written[1][1][0, 3:], object_pose[0, 1, 3:]))
        self.assertEqual(
            environment._training_held_reset_stage_queued.tolist(),
            [False, True],
        )
        self.assertEqual(
            environment._training_held_reset_stage_restored.tolist(),
            [True, False],
        )
        self.assertEqual(
            environment._training_held_reset_stage_waited_control_steps.tolist(),
            [8, 0],
        )

    def test_actual_state_write_gate_rejects_active_policy_rows(self):
        gate = _compiled_training_environment_method(
            "_assert_held_state_write_allowed"
        )
        environment = SimpleNamespace(
            _training_initialized=True,
            _training_policy_activated=torch.tensor([False, True]),
        )
        with self.assertRaises(AssertionError):
            gate(environment, torch.tensor([1]))
        gate(environment, torch.tensor([0]))

    def test_actual_held_init_prefix_sets_only_active_horizon(self):
        tree = ast.parse(TRAINING_ENV_PATH.read_text(encoding="utf-8"))
        factory = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "make_rj45_training_env_type"
        )
        wrapper = next(node for node in factory.body if isinstance(node, ast.ClassDef))
        initializer = next(
            node
            for node in wrapper.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        prefix = []
        for statement in initializer.body:
            if (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr == "__init__"
            ):
                break
            prefix.append(statement)
        executable = ast.fix_missing_locations(ast.Module(body=prefix, type_ignores=[]))
        for observation_dim in (79, 85):
            cfg = SimpleNamespace(seed=7, action_space=None, observation_space=None)
            namespace = {
                "self": SimpleNamespace(),
                "cfg": cfg,
                "config": {
                    "active_horizon_s": 24.0,
                    "warmup_s": 0.2,
                },
                "validated_held_state": {"verified": True},
                "calibration_bias_enabled": True,
                "bias_level": 0,
                "action_dim": 6,
                "observation_dim": observation_dim,
            }
            exec(compile(executable, TRAINING_ENV_PATH, "exec"), namespace)

            self.assertEqual(cfg.observation_space, observation_dim)
            self.assertEqual(cfg.episode_length_s, 24.0)
            self.assertEqual(math.ceil(cfg.episode_length_s / (1.0 / 30.0)), 720)

    def test_actual_base_time_limit_method_uses_completed_step_boundary(self):
        base_path = PROJECT_ROOT / "src/rby1_assembly/isaac_env.py"
        tree = ast.parse(base_path.read_text(encoding="utf-8"))
        environment_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Rby1AssemblyEnv"
        )
        method = next(
            node
            for node in environment_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_episode_time_limit_reached"
        )
        namespace = {"torch": torch}
        exec(
            compile(
                ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
                base_path,
                "exec",
            ),
            namespace,
        )
        environment = SimpleNamespace(
            episode_length_buf=torch.tensor([719, 720]),
            max_episode_length=720,
        )
        reached = namespace["_episode_time_limit_reached"](environment)
        self.assertEqual(reached.tolist(), [False, True])

    def test_bias_case_seed_is_stable_and_separates_episode_and_environment(self):
        self.assertEqual(_calibration_bias_case_seed(17, 0, 0), 45_123_034)
        self.assertEqual(
            _calibration_bias_case_seed(17, 0, 1) - _calibration_bias_case_seed(17, 0, 0),
            1_000_003,
        )
        self.assertEqual(
            _calibration_bias_case_seed(17, 4, 2) - _calibration_bias_case_seed(17, 0, 2),
            4,
        )

    def test_bias_case_seed_rejects_boolean_negative_and_noninteger_inputs(self):
        for values in ((True, 0, 0), (0, -1, 0), (0, 0, 1.5)):
            with self.assertRaises(ValueError):
                _calibration_bias_case_seed(*values)

    def test_warmup_sensor_status_rejects_nonfinite_and_overlimit_rows(self):
        fingers = torch.tensor(
            [
                [0.6, 0.6, 0.6, 0.6],
                [float("nan"), 0.7, 0.7, 0.7],
                [3.0] * 4,
                [0.7, 0.7, 0.7, 0.7],
            ]
        )
        insertion_force = torch.zeros((4, 3))
        insertion_load = torch.tensor([1.0, 1.0, 1.0, 1.0])
        forbidden = torch.tensor([0.0, 0.0, 0.0, 0.3])
        finite, within, grip_live = _warmup_sensor_status(
            fingers,
            insertion_force,
            insertion_load,
            forbidden,
            grip_contact_force_n=0.7,
            hard_contact_force_n=2.5,
            forbidden_contact_force_n=0.2,
        )
        self.assertEqual(finite.tolist(), [True, False, True, True])
        self.assertEqual(within.tolist(), [True, False, False, False])
        self.assertEqual(grip_live.tolist(), [False, False, True, True])

        insertion_force[0, 1] = float("inf")
        insertion_load[1] = float("nan")
        forbidden[2] = float("nan")
        finite, _, _ = _warmup_sensor_status(
            fingers,
            insertion_force,
            insertion_load,
            forbidden,
            grip_contact_force_n=0.7,
            hard_contact_force_n=2.5,
            forbidden_contact_force_n=0.2,
        )
        self.assertEqual(finite.tolist(), [False, False, False, True])

    def test_warmup_motion_rejects_common_mode_object_motion(self):
        joint_velocity = torch.zeros((2, 4))
        reference_error = torch.zeros((2, 4))
        prismatic = torch.tensor([False, False, True, True])
        socket_velocity = torch.zeros((2, 6))
        plug_velocity = torch.zeros((2, 6))
        socket_velocity[1, 0] = 0.02
        plug_velocity[1, 0] = 0.02
        finite, settled = _warmup_motion_status(
            joint_velocity,
            reference_error,
            prismatic,
            socket_velocity,
            plug_velocity,
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            limits=STABILITY_LIMITS,
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertEqual(finite.tolist(), [True, True])
        self.assertEqual(settled.tolist(), [True, False])

    def test_warmup_motion_applies_joint_unit_specific_limits(self):
        joint_velocity = torch.zeros((4, 4))
        reference_error = torch.zeros((4, 4))
        joint_velocity[1, 0] = 0.051
        reference_error[2, 2] = 0.0051
        socket_velocity = torch.zeros((4, 6))
        socket_velocity[3, 0] = float("nan")
        finite, settled = _warmup_motion_status(
            joint_velocity,
            reference_error,
            torch.tensor([False, False, True, True]),
            socket_velocity,
            torch.zeros((4, 6)),
            torch.zeros((4, 3)),
            torch.zeros((4, 3)),
            limits=STABILITY_LIMITS,
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertEqual(finite.tolist(), [True, True, True, False])
        self.assertEqual(settled.tolist(), [True, False, False, False])

    def test_warmup_timeout_does_not_reject_a_boundary_step_pass(self):
        timed_out = _warmup_timeout_mask(
            pending=torch.tensor([True, True, False]),
            passed=torch.tensor([True, False, False]),
            warmup_steps=torch.tensor([50, 50, 50]),
            timeout_steps=50,
        )
        self.assertEqual(timed_out.tolist(), [False, True, False])

    def test_timeout_error_preserves_complete_json_safe_diagnostics(self):
        diagnostics = _held_warmup_timeout_diagnostics(
            timed_out=torch.tensor([True, False]),
            finger_force_n=torch.tensor(
                [[0.8, 0.9, 1.0, 1.1], [0.7, 0.7, 0.7, 0.7]]
            ),
            lift_left_m=torch.tensor([0.011, 0.012]),
            lift_right_m=torch.tensor([0.009, 0.012]),
            joint_velocity_si=torch.tensor(
                [
                    [0.02, float("inf"), 0.003, 0.004],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            joint_reference_error_si=torch.tensor(
                [
                    [0.01, 0.02, float("nan"), float("-inf")],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            prismatic_joint_mask=torch.tensor([False, False, True, True]),
            socket_com_velocity_world=torch.tensor(
                [[0.001, 0.002, 0.003, 0.01, 0.02, 0.03], [0.0] * 6]
            ),
            plug_com_velocity_world=torch.tensor(
                [[0.002, 0.003, 0.004, 0.02, 0.03, 0.04], [0.0] * 6]
            ),
            relative_linear_velocity_m_s=torch.tensor(
                [[0.001, 0.0, -0.001], [0.0, 0.0, 0.0]]
            ),
            relative_angular_velocity_rad_s=torch.tensor(
                [[0.01, 0.0, -0.01], [0.0, 0.0, 0.0]]
            ),
            insertion_force_world_n=torch.tensor(
                [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]
            ),
            insertion_load_n=torch.tensor([3.5, 0.0]),
            forbidden_load_n=torch.tensor([0.1, 0.0]),
            state_finite=torch.tensor([True, True]),
            motion_finite=torch.tensor([False, True]),
            motion_settled=torch.tensor([False, True]),
            sensor_finite=torch.tensor([True, True]),
            within_limits=torch.tensor([True, True]),
            grip_live=torch.tensor([True, True]),
            stable=torch.tensor([False, True]),
            warmup_steps=torch.tensor([30, 3]),
            warmup_stable_steps=torch.tensor([0, 3]),
            required_stable_steps=6,
            timeout_steps=30,
            step_dt_s=1.0 / 30.0,
            grip_lift_proof_m=0.01,
            limits=STABILITY_LIMITS,
        )
        error = HeldGraspWarmupTimeoutError(diagnostics)
        encoded = str(error).split(": ", 1)[1]
        parsed = json.loads(encoded)
        self.assertEqual(parsed, error.diagnostics)
        json.dumps(parsed, allow_nan=False)
        self.assertEqual(parsed["failed_environment_ids"], [0])
        environment = parsed["environments"][0]
        self.assertEqual(set(environment["finger_force_n"]), {
            "left_l1", "left_l2", "right_r1", "right_r2"
        })
        self.assertEqual(
            environment["joint_unit_maxima"]["revolute_velocity_rad_s"],
            {"value": "+inf", "nonfinite": True},
        )
        self.assertEqual(
            environment["joint_unit_maxima"]["prismatic_reference_error_m"],
            {"value": "nan", "nonfinite": True},
        )
        self.assertEqual(parsed["nonfinite_summary"]["count"], 3)
        self.assertEqual(
            parsed["nonfinite_summary"]["locations"],
            [
                {"env_id": 0, "field": "joint_velocity_si", "index": [1]},
                {
                    "env_id": 0,
                    "field": "joint_reference_error_si",
                    "index": [2],
                },
                {
                    "env_id": 0,
                    "field": "joint_reference_error_si",
                    "index": [3],
                },
            ],
        )
        self.assertEqual(
            [
                item["reading"]
                for item in environment["nonfinite_readings"]
            ],
            [
                {"value": "+inf", "nonfinite": True},
                {"value": "nan", "nonfinite": True},
                {"value": "-inf", "nonfinite": True},
            ],
        )
        self.assertFalse(environment["gates"]["motion_finite"])
        self.assertFalse(environment["gates"]["lift_right_ok"])
        self.assertEqual(environment["steps"]["timeout"], 30)
        self.assertIn("socket", environment["object_com_velocity_world"])
        self.assertIn("plug", environment["object_com_velocity_world"])
        self.assertIn("insertion_load_n", environment["contact_loads"])
        self.assertIn("forbidden_load_n", environment["contact_loads"])


if __name__ == "__main__":
    unittest.main()
