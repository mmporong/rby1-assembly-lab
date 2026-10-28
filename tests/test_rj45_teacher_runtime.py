from __future__ import annotations

from pathlib import Path
import sys
import types
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rby1_assembly.rj45_teacher_runtime import (  # noqa: E402
    _compose_pose,
    make_rj45_teacher_env_type,
    virtual_desired_body_target,
)
from rby1_assembly.rj45_training import (  # noqa: E402
    ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
    _quaternion_inverse_wxyz,
    _quaternion_rotate_wxyz,
)


def _axis_angle(axis: tuple[float, float, float], angle: float) -> torch.Tensor:
    vector = torch.tensor(axis, dtype=torch.float64)
    vector = vector / torch.linalg.vector_norm(vector)
    return torch.cat(
        (
            torch.tensor([torch.cos(torch.tensor(angle / 2.0))], dtype=torch.float64),
            vector * torch.sin(torch.tensor(angle / 2.0, dtype=torch.float64)),
        )
    ).unsqueeze(0)


def _rigid_transform_pose(
    transform_position: torch.Tensor,
    transform_quaternion: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _compose_pose(
        transform_position, transform_quaternion, position, quaternion
    )


class _DummyTrainingBase:
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = torch.device("cpu")
        self._training_config = {
            "action_mode": "right_only_6d",
            "action_application_mode": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
        }
        self.cfg = types.SimpleNamespace(
            residual_translation_bound_m=0.02,
            residual_rotation_bound_rad=0.2,
        )
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(self.num_envs, 1)
        socket_position = torch.zeros((self.num_envs, 3))
        plug_position = torch.tensor([0.10, 0.0, 0.0]).repeat(self.num_envs, 1)
        body_position = torch.zeros((self.num_envs, 1, 3))
        body_position[:, 0, 0] = 0.11
        body_quaternion = identity[:, None, :].clone()
        self.socket = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=socket_position, root_quat_w=identity.clone()
            )
        )
        self.plug = types.SimpleNamespace(
            data=types.SimpleNamespace(
                root_pos_w=plug_position, root_quat_w=identity.clone()
            )
        )
        self.robot = types.SimpleNamespace(
            data=types.SimpleNamespace(
                body_pos_w=body_position, body_quat_w=body_quaternion
            )
        )
        self._right_ee_body_id = 0
        self._insertion_command_depth_m = torch.tensor([0.002, 0.003, 0.004])
        self._active = torch.tensor([True, False, True])
        self._post_active = torch.tensor([True, True, False])
        self._warmup = torch.zeros(self.num_envs, dtype=torch.bool)
        self.extras: dict[str, object] = {}
        self.base_apply_calls = 0
        self._initialize_training_state()
        self._configure_training_reset(torch.arange(self.num_envs))
        self._training_last_policy_observation[:] = torch.arange(
            self.num_envs * 79, dtype=torch.float32
        ).reshape(self.num_envs, 79)
        self._training_last_policy_observation_valid[:] = True
        self._training_last_policy_observation_timestamp_s[:] = torch.tensor(
            [1.0, 2.0, 3.0]
        )
        self._training_previous_action[:] = 0.25
        self._training_frozen_object_to_ee_pos[:, 1, 0] = 0.02

    def _initialize_training_state(self) -> None:
        self._training_last_policy_observation = torch.zeros((self.num_envs, 79))
        self._training_last_policy_observation_valid = torch.zeros(
            self.num_envs, dtype=torch.bool
        )
        self._training_last_policy_observation_timestamp_s = torch.full(
            (self.num_envs,), -1.0
        )
        self._training_previous_action = torch.zeros((self.num_envs, 12))
        self._training_action_delta = torch.zeros((self.num_envs, 12))
        self._training_action_clip_boundary_step = torch.zeros(
            self.num_envs, dtype=torch.bool
        )
        self._training_action_norm_projection_step = torch.zeros(
            self.num_envs, dtype=torch.bool
        )
        self._training_applied_tip_offset_entry = torch.zeros((self.num_envs, 6))
        self._training_frozen_object_to_ee_pos = torch.zeros((self.num_envs, 2, 3))
        self._training_frozen_object_to_ee_quat = torch.zeros((self.num_envs, 2, 4))
        self._training_frozen_object_to_ee_quat[:, :, 0] = 1.0
        self._actions = torch.zeros((self.num_envs, 12))

    def _configure_training_reset(self, env_ids: torch.Tensor) -> None:
        self._training_previous_action[env_ids] = 0.0
        self._training_action_delta[env_ids] = 0.0
        self._training_applied_tip_offset_entry[env_ids] = 0.0

    def _policy_active_mask(self) -> torch.Tensor:
        return self._active.clone()

    def _warmup_pending_mask(self) -> torch.Tensor:
        return self._warmup.clone()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._active = self._post_active.clone()
        self._insertion_command_depth_m += 0.002
        self._actions = torch.cat((torch.zeros_like(actions), actions), dim=1)
        baseline_position, baseline_quaternion = self._plug_pose_goal(
            self.socket.data.root_pos_w,
            self.socket.data.root_quat_w,
            self._insertion_command_depth_m,
        )
        self._apply_policy_mating_target_pose_offsets(
            baseline_position, baseline_quaternion
        )
        self._training_action_delta.zero_()
        self._training_previous_action.zero_()

    def _apply_policy_mating_target_pose_offsets(
        self, plug_goal: torch.Tensor, plug_goal_quat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.base_apply_calls += 1
        self._training_applied_tip_offset_entry[:, :3] = (
            self._actions[:, 6:9] * self.cfg.residual_translation_bound_m
        )
        self._training_applied_tip_offset_entry[:, 3:] = (
            self._actions[:, 9:12] * self.cfg.residual_rotation_bound_rad
        )
        return plug_goal, plug_goal_quat

    def _plug_pose_goal(
        self,
        socket_position: torch.Tensor,
        socket_quaternion: torch.Tensor,
        depth: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset = torch.zeros_like(socket_position)
        offset[:, 1] = depth
        return (
            socket_position + _quaternion_rotate_wxyz(socket_quaternion, offset),
            socket_quaternion.clone(),
        )

    def _control_object_poses(self):
        return (
            self.socket.data.root_pos_w,
            self.socket.data.root_quat_w,
            self.plug.data.root_pos_w,
            self.plug.data.root_quat_w,
        )

    def _frame_tensors(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if name not in {"socket_entry", "plug_tip"}:
            raise KeyError(name)
        position = torch.zeros((self.num_envs, 3))
        quaternion = torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(
            self.num_envs, 1
        )
        return position, quaternion

    def _training_step_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            "policy_input_observation": self._training_last_policy_observation.clone(),
            "policy_input_timestamp_s": (
                self._training_last_policy_observation_timestamp_s.clone()
            ),
            "policy_input_valid": self._training_last_policy_observation_valid.clone(),
        }


class Rj45TeacherRuntimeTests(unittest.TestCase):
    def test_virtual_target_satisfies_controller_algebra_and_naive_goal_does_not(self) -> None:
        goal_position = torch.tensor([[0.30, -0.20, 0.50]], dtype=torch.float64)
        goal_quaternion = _axis_angle((0.3, 1.0, -0.2), 0.7)
        plug_position = torch.tensor([[-0.10, 0.25, 0.40]], dtype=torch.float64)
        plug_quaternion = _axis_angle((1.0, 0.2, 0.4), -0.5)
        factual_position = torch.tensor([[0.04, -0.01, 0.03]], dtype=torch.float64)
        factual_quaternion = _axis_angle((0.1, -0.7, 1.0), 0.35)
        ee_position, ee_quaternion = _compose_pose(
            plug_position, plug_quaternion, factual_position, factual_quaternion
        )
        frozen_position = torch.tensor([[0.02, 0.03, -0.01]], dtype=torch.float64)
        frozen_quaternion = _axis_angle((0.5, 1.0, 0.3), -0.25)

        virtual = virtual_desired_body_target(
            goal_position,
            goal_quaternion,
            plug_position,
            plug_quaternion,
            ee_position,
            ee_quaternion,
            frozen_position,
            frozen_quaternion,
        )
        actual_controller_goal = _compose_pose(*virtual, frozen_position, frozen_quaternion)
        desired_ee = _compose_pose(
            goal_position, goal_quaternion, factual_position, factual_quaternion
        )
        naive = _compose_pose(
            goal_position, goal_quaternion, frozen_position, frozen_quaternion
        )
        torch.testing.assert_close(actual_controller_goal[0], desired_ee[0])
        torch.testing.assert_close(actual_controller_goal[1], desired_ee[1])
        self.assertFalse(torch.allclose(naive[0], desired_ee[0]))

    def test_rigid_cotransform_of_actual_plug_and_ee_preserves_virtual_label(self) -> None:
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        goal_position = torch.tensor([[0.4, 0.2, -0.1]], dtype=torch.float64)
        plug_position = torch.tensor([[0.1, -0.3, 0.2]], dtype=torch.float64)
        plug_quaternion = _axis_angle((0.2, 1.0, 0.1), 0.4)
        ee_position, ee_quaternion = _compose_pose(
            plug_position,
            plug_quaternion,
            torch.tensor([[0.03, 0.01, -0.02]], dtype=torch.float64),
            _axis_angle((1.0, 0.4, 0.2), -0.2),
        )
        frozen_position = torch.tensor([[0.02, -0.02, 0.01]], dtype=torch.float64)
        transform_position = torch.tensor([[0.5, -0.2, 0.3]], dtype=torch.float64)
        transform_quaternion = _axis_angle((0.1, 0.3, 1.0), 0.8)
        moved_plug = _rigid_transform_pose(
            transform_position, transform_quaternion, plug_position, plug_quaternion
        )
        moved_ee = _rigid_transform_pose(
            transform_position, transform_quaternion, ee_position, ee_quaternion
        )
        original = virtual_desired_body_target(
            goal_position,
            identity,
            plug_position,
            plug_quaternion,
            ee_position,
            ee_quaternion,
            frozen_position,
            identity,
        )
        transformed = virtual_desired_body_target(
            goal_position,
            identity,
            *moved_plug,
            *moved_ee,
            frozen_position,
            identity,
        )
        torch.testing.assert_close(original[0], transformed[0])
        torch.testing.assert_close(original[1], transformed[1])

    def test_teacher_hook_uses_committed_depth_and_pre_and_post_step_gates(self) -> None:
        teacher_type = make_rj45_teacher_env_type(_DummyTrainingBase)
        environment = teacher_type()
        expected_observation = environment._training_last_policy_observation.clone()
        prior = environment._training_previous_action.clone()

        environment._pre_physics_step(torch.zeros((3, 6)))

        self.assertEqual(environment.base_apply_calls, 1)
        self.assertTrue(environment._teacher_active.tolist() == [True, False, False])
        torch.testing.assert_close(
            environment._teacher_command_depth_m, torch.tensor([0.004, 0.005, 0.006])
        )
        self.assertTrue(bool(torch.count_nonzero(environment._actions[0, 6:])))
        self.assertEqual(int(torch.count_nonzero(environment._actions[1:])), 0)
        torch.testing.assert_close(
            environment._training_previous_action, environment._actions
        )
        torch.testing.assert_close(
            environment._training_action_delta, environment._actions - prior
        )
        snapshot = environment._training_step_snapshot()
        torch.testing.assert_close(snapshot["policy_input_observation"], expected_observation)
        torch.testing.assert_close(
            snapshot["teacher_bounded_action_6d"], environment._actions[:, 6:]
        )
        torch.testing.assert_close(
            snapshot["teacher_physical_offset_entry_6d"],
            environment._training_applied_tip_offset_entry,
        )
        environment._teacher_bounded_action_6d.zero_()
        self.assertTrue(bool(torch.count_nonzero(snapshot["teacher_bounded_action_6d"])))

    def test_limit_telemetry_keeps_raw_request_after_bounded_action_is_applied(self) -> None:
        environment = make_rj45_teacher_env_type(_DummyTrainingBase)()
        environment.robot.data.body_pos_w[0, 0, :2] = torch.tensor([0.16, 0.04])
        environment._pre_physics_step(torch.zeros((3, 6)))
        self.assertEqual(environment._training_action_clip_boundary_step.tolist(), [True, False, False])
        self.assertEqual(environment._training_action_norm_projection_step.tolist(), [True, False, False])
        self.assertTrue(bool((environment._actions[0, 6:9].abs() < 1.0).all()))
        torch.testing.assert_close(torch.linalg.vector_norm(environment._actions[0, 6:9]), torch.tensor(1.0))

    def test_external_action_is_zero_only_and_partial_reset_clears_selected_teacher_row(self) -> None:
        teacher_type = make_rj45_teacher_env_type(_DummyTrainingBase)
        environment = teacher_type()
        with self.assertRaisesRegex(ValueError, "exactly zero"):
            environment._pre_physics_step(torch.ones((3, 6)))

        teacher_tensors = (
            environment._teacher_requested_raw_action_6d,
            environment._teacher_bounded_action_6d,
            environment._teacher_physical_offset_entry_6d,
            environment._teacher_command_depth_m,
            environment._teacher_virtual_target_body_pose_world_wxyz,
            environment._teacher_true_goal_body_pose_world_wxyz,
            environment._teacher_policy_input_observation,
            environment._teacher_prior_bounded_action_12d,
        )
        for value in teacher_tensors:
            value.fill_(2.0)
        environment._teacher_policy_input_timestamp_s.fill_(2.0)
        environment._teacher_policy_input_valid.fill_(True)
        environment._teacher_pre_step_active.fill_(True)
        environment._teacher_bound_limited.fill_(True)
        environment._teacher_active.fill_(True)

        environment._configure_training_reset(torch.tensor([1]))

        for value in teacher_tensors:
            self.assertTrue(bool(torch.count_nonzero(value[0])))
            self.assertTrue(bool(torch.count_nonzero(value[2])))
        self.assertEqual(environment._teacher_command_depth_m[1].item(), 0.0)
        self.assertEqual(environment._teacher_policy_input_timestamp_s[1].item(), -1.0)
        self.assertFalse(environment._teacher_policy_input_valid[1].item())
        self.assertFalse(environment._teacher_active[1].item())
        torch.testing.assert_close(
            environment._teacher_virtual_target_body_pose_world_wxyz[1],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        )


if __name__ == "__main__":
    unittest.main()
