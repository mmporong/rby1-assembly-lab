"""Teacher-only RJ45 runtime wrapper with privileged SE(3) target labels."""

from __future__ import annotations

from typing import Any

from .rj45_teacher import teacher_tip_target_action
from .rj45_training import (
    ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
    _quaternion_inverse_wxyz,
    _quaternion_multiply_wxyz,
    _quaternion_rotate_wxyz,
)


def _compose_pose(position_a: Any, quaternion_a: Any, position_b: Any, quaternion_b: Any):
    return (
        position_a + _quaternion_rotate_wxyz(quaternion_a, position_b),
        _quaternion_multiply_wxyz(quaternion_a, quaternion_b),
    )


def _inverse_pose(position: Any, quaternion: Any):
    inverse_quaternion = _quaternion_inverse_wxyz(quaternion)
    return (
        -_quaternion_rotate_wxyz(inverse_quaternion, position),
        inverse_quaternion,
    )


def virtual_desired_body_target(
    true_goal_position_world: Any,
    true_goal_quaternion_world_wxyz: Any,
    actual_plug_position_world: Any,
    actual_plug_quaternion_world_wxyz: Any,
    actual_right_ee_position_world: Any,
    actual_right_ee_quaternion_world_wxyz: Any,
    frozen_estimated_plug_to_ee_position: Any,
    frozen_estimated_plug_to_ee_quaternion_wxyz: Any,
) -> tuple[Any, Any]:
    """Return ``G_true * F_actual * inverse(F_hat)`` for the existing controller."""

    actual_plug_inverse = _inverse_pose(
        actual_plug_position_world, actual_plug_quaternion_world_wxyz
    )
    actual_plug_to_ee = _compose_pose(
        *actual_plug_inverse,
        actual_right_ee_position_world,
        actual_right_ee_quaternion_world_wxyz,
    )
    desired_ee = _compose_pose(
        true_goal_position_world,
        true_goal_quaternion_world_wxyz,
        *actual_plug_to_ee,
    )
    frozen_inverse = _inverse_pose(
        frozen_estimated_plug_to_ee_position,
        frozen_estimated_plug_to_ee_quaternion_wxyz,
    )
    return _compose_pose(*desired_ee, *frozen_inverse)


def make_rj45_teacher_env_type(base_training_cls: type) -> type:
    """Wrap a configured training environment with a privileged teacher action source.

    The returned environment accepts only a zero external action. The teacher action is
    calculated inside the committed-depth mating-target hook and remains outside the
    standard 79D student observation.
    """

    import torch

    class Rby1Rj45TeacherEnv(base_training_cls):
        """Teacher demonstrator that preserves the base reward and safety transitions."""

        def _initialize_training_state(self) -> None:
            super()._initialize_training_state()
            if (
                self._training_config.get("action_mode") != "right_only_6d"
                or self._training_config.get("action_application_mode")
                != ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            ):
                raise ValueError(
                    "RJ45 teacher requires right_only_6d absolute tip-target actions"
                )
            shape_6 = (self.num_envs, 6)
            self._teacher_requested_raw_action_6d = torch.zeros(
                shape_6, device=self.device
            )
            self._teacher_bounded_action_6d = torch.zeros(shape_6, device=self.device)
            self._teacher_physical_offset_entry_6d = torch.zeros(
                shape_6, device=self.device
            )
            self._teacher_bound_limited = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._teacher_active = torch.zeros_like(self._teacher_bound_limited)
            self._teacher_command_depth_m = torch.zeros(
                self.num_envs, device=self.device
            )
            self._teacher_virtual_target_body_pose_world_wxyz = torch.zeros(
                (self.num_envs, 7), device=self.device
            )
            self._teacher_virtual_target_body_pose_world_wxyz[:, 3] = 1.0
            self._teacher_true_goal_body_pose_world_wxyz = torch.zeros(
                (self.num_envs, 7), device=self.device
            )
            self._teacher_true_goal_body_pose_world_wxyz[:, 3] = 1.0
            self._teacher_policy_input_observation = torch.zeros_like(
                self._training_last_policy_observation
            )
            self._teacher_policy_input_timestamp_s = torch.full_like(
                self._training_last_policy_observation_timestamp_s, -1.0
            )
            self._teacher_policy_input_valid = torch.zeros_like(
                self._training_last_policy_observation_valid
            )
            self._teacher_pre_step_active = torch.zeros_like(
                self._training_last_policy_observation_valid
            )
            self._teacher_prior_bounded_action_12d = torch.zeros_like(
                self._training_previous_action
            )

        def _pre_physics_step(self, actions) -> None:
            if not isinstance(actions, torch.Tensor):
                raise TypeError("RJ45 teacher external action must be a torch.Tensor")
            if actions.shape != (self.num_envs, 6):
                raise ValueError("RJ45 teacher external action must have shape (num_envs, 6)")
            if not bool(torch.isfinite(actions).all()) or bool(torch.count_nonzero(actions)):
                raise ValueError("RJ45 teacher external action must be finite and exactly zero")

            self._teacher_policy_input_observation[:] = (
                self._training_last_policy_observation
            )
            self._teacher_policy_input_timestamp_s[:] = (
                self._training_last_policy_observation_timestamp_s
            )
            self._teacher_policy_input_valid[:] = (
                self._training_last_policy_observation_valid
            )
            self._teacher_pre_step_active[:] = self._policy_active_mask()
            self._teacher_prior_bounded_action_12d[:] = self._training_previous_action

            super()._pre_physics_step(torch.zeros_like(actions))

            transmitted = self._actions.clone()
            self._training_action_delta[:] = (
                transmitted - self._teacher_prior_bounded_action_12d
            )
            self._training_previous_action[:] = transmitted
            active = self._teacher_active
            requested = self._teacher_requested_raw_action_6d
            self._training_action_clip_boundary_step[:] = (
                (torch.abs(requested) >= 1.0).any(dim=1) & active
            )
            vectors = torch.clamp(requested, -1.0, 1.0).reshape(
                self.num_envs, 2, 3
            )
            self._training_action_norm_projection_step[:] = (
                (torch.linalg.vector_norm(vectors, dim=2) > 1.0).any(dim=1)
                & active
            )
            self.extras["rj45_teacher_step"] = self._teacher_step_snapshot()

        def _apply_policy_mating_target_pose_offsets(
            self, plug_goal, plug_goal_quat
        ):
            if not hasattr(self, "_teacher_requested_raw_action_6d"):
                return super()._apply_policy_mating_target_pose_offsets(
                    plug_goal, plug_goal_quat
                )

            actual_socket_position = self.socket.data.root_pos_w
            actual_socket_quaternion = self.socket.data.root_quat_w
            actual_plug_position = self.plug.data.root_pos_w
            actual_plug_quaternion = self.plug.data.root_quat_w
            actual_right_ee_position = self.robot.data.body_pos_w[
                :, self._right_ee_body_id
            ]
            actual_right_ee_quaternion = self.robot.data.body_quat_w[
                :, self._right_ee_body_id
            ]
            true_goal_position, true_goal_quaternion = self._plug_pose_goal(
                actual_socket_position,
                actual_socket_quaternion,
                self._insertion_command_depth_m,
            )
            virtual_position, virtual_quaternion = virtual_desired_body_target(
                true_goal_position,
                true_goal_quaternion,
                actual_plug_position,
                actual_plug_quaternion,
                actual_right_ee_position,
                actual_right_ee_quaternion,
                self._training_frozen_object_to_ee_pos[:, 1],
                self._training_frozen_object_to_ee_quat[:, 1],
            )
            estimated_socket_position, estimated_socket_quaternion, _, _ = (
                self._control_object_poses()
            )
            entry_position, entry_quaternion = self._frame_tensors("socket_entry")
            _, estimated_entry_quaternion = _compose_pose(
                estimated_socket_position,
                estimated_socket_quaternion,
                entry_position,
                entry_quaternion,
            )
            tip_position, tip_quaternion = self._frame_tensors("plug_tip")
            active = (
                self._teacher_policy_input_valid
                & self._teacher_pre_step_active
                & self._policy_active_mask()
                & ~self._warmup_pending_mask()
            )
            result = teacher_tip_target_action(
                plug_goal,
                plug_goal_quat,
                virtual_position,
                virtual_quaternion,
                estimated_entry_quaternion,
                tip_position,
                tip_quaternion,
                self.cfg.residual_translation_bound_m,
                self.cfg.residual_rotation_bound_rad,
                active_mask=active,
            )
            self._teacher_requested_raw_action_6d[:] = result["raw_action_6d"]
            self._teacher_bounded_action_6d[:] = result["bounded_action_6d"]
            self._teacher_bound_limited[:] = result["bound_limited_mask"]
            self._teacher_active[:] = active
            self._teacher_command_depth_m[:] = self._insertion_command_depth_m
            self._teacher_virtual_target_body_pose_world_wxyz[:] = torch.cat(
                (virtual_position, virtual_quaternion), dim=1
            )
            self._teacher_true_goal_body_pose_world_wxyz[:] = torch.cat(
                (true_goal_position, true_goal_quaternion), dim=1
            )
            self._actions.zero_()
            self._actions[:, 6:12] = result["bounded_action_6d"]

            target = super()._apply_policy_mating_target_pose_offsets(
                plug_goal, plug_goal_quat
            )
            applied = self._training_applied_tip_offset_entry
            expected = result["applied_offset_entry_6d"]
            if not bool(torch.allclose(applied, expected, rtol=1.0e-6, atol=1.0e-8)):
                raise RuntimeError("teacher physical offset differs from the applied target offset")
            self._teacher_physical_offset_entry_6d[:] = applied
            return target

        def _teacher_step_snapshot(self) -> dict[str, Any]:
            return {
                "teacher_requested_raw_action_6d": (
                    self._teacher_requested_raw_action_6d.clone()
                ),
                "teacher_bounded_action_6d": self._teacher_bounded_action_6d.clone(),
                "teacher_physical_offset_entry_6d": (
                    self._teacher_physical_offset_entry_6d.clone()
                ),
                "teacher_bound_limited": self._teacher_bound_limited.clone(),
                "teacher_active": self._teacher_active.clone(),
                "teacher_command_depth_m": self._teacher_command_depth_m.clone(),
                "teacher_virtual_target_body_pose_world_wxyz": (
                    self._teacher_virtual_target_body_pose_world_wxyz.clone()
                ),
                "teacher_true_goal_body_pose_world_wxyz": (
                    self._teacher_true_goal_body_pose_world_wxyz.clone()
                ),
            }

        def _training_step_snapshot(self):
            snapshot = super()._training_step_snapshot()
            snapshot.update(
                {
                    "policy_input_observation": (
                        self._teacher_policy_input_observation.clone()
                    ),
                    "policy_input_timestamp_s": (
                        self._teacher_policy_input_timestamp_s.clone()
                    ),
                    "policy_input_valid": self._teacher_policy_input_valid.clone(),
                    **self._teacher_step_snapshot(),
                }
            )
            return snapshot

        def _configure_training_reset(self, env_ids):
            super()._configure_training_reset(env_ids)
            if not hasattr(self, "_teacher_requested_raw_action_6d"):
                return
            for value in (
                self._teacher_requested_raw_action_6d,
                self._teacher_bounded_action_6d,
                self._teacher_physical_offset_entry_6d,
                self._teacher_command_depth_m,
                self._teacher_virtual_target_body_pose_world_wxyz,
                self._teacher_true_goal_body_pose_world_wxyz,
                self._teacher_policy_input_observation,
                self._teacher_prior_bounded_action_12d,
            ):
                value[env_ids] = 0.0
            self._teacher_virtual_target_body_pose_world_wxyz[env_ids, 3] = 1.0
            self._teacher_true_goal_body_pose_world_wxyz[env_ids, 3] = 1.0
            self._teacher_policy_input_timestamp_s[env_ids] = -1.0
            self._teacher_policy_input_valid[env_ids] = False
            self._teacher_pre_step_active[env_ids] = False
            self._teacher_bound_limited[env_ids] = False
            self._teacher_active[env_ids] = False

    Rby1Rj45TeacherEnv.__name__ = f"Teacher{base_training_cls.__name__}"
    Rby1Rj45TeacherEnv.__qualname__ = Rby1Rj45TeacherEnv.__name__
    return Rby1Rj45TeacherEnv
