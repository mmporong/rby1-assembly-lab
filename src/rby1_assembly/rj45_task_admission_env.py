"""Fail-closed G1 task-admission instrumentation for one held-grasp episode."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


ALIGN_PHASE_CODE = 3
INSERT_PHASE_CODE = 4
BIAS_ABSOLUTE_TOLERANCE = 1.0e-9


class TaskAdmissionContractViolation(RuntimeError):
    """A preregistered admission invariant was violated during physics."""


def _finite_vector(values: Sequence[float], name: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be finite")
    return vector  # type: ignore[return-value]


def _validate_sha256(value: str, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def make_rj45_task_admission_env_type(
    base_comparison_cls: type,
    *,
    requested_translation_bias_entry_m: Sequence[float],
    requested_rotation_bias_entry_rad: Sequence[float],
    held_grasp_id: str,
    held_grasp_index: int,
    held_artifact_sha256: str,
) -> type:
    """Instrument one immutable G1 case without exposing evaluator GT to providers."""

    import torch

    translation = _finite_vector(
        requested_translation_bias_entry_m, "requested translation bias"
    )
    rotation = _finite_vector(
        requested_rotation_bias_entry_rad, "requested rotation bias"
    )
    if not isinstance(held_grasp_id, str) or not held_grasp_id:
        raise ValueError("held_grasp_id must be a non-empty string")
    if type(held_grasp_index) is not int or held_grasp_index < 0:
        raise ValueError("held_grasp_index must be a non-negative integer")
    held_sha = _validate_sha256(held_artifact_sha256, "held_artifact_sha256")

    class Rby1Rj45TaskAdmissionEnv(base_comparison_cls):
        def _initialize_training_state(self) -> None:
            super()._initialize_training_state()
            shape = (self.num_envs,)
            self._admission_requested_translation_bias_entry_m = torch.tensor(
                translation, dtype=torch.float64, device=self.device
            ).repeat(self.num_envs, 1)
            self._admission_requested_rotation_bias_entry_rad = torch.tensor(
                rotation, dtype=torch.float64, device=self.device
            ).repeat(self.num_envs, 1)
            self._admission_prefreeze_snapshot_valid = torch.zeros(
                shape, dtype=torch.bool, device=self.device
            )
            self._admission_prefreeze_translation_entry_m = torch.zeros(
                (self.num_envs, 3), dtype=torch.float64, device=self.device
            )
            self._admission_prefreeze_rotation_vector_entry_rad = torch.zeros_like(
                self._admission_prefreeze_translation_entry_m
            )
            self._admission_insert_transition_snapshot_valid = torch.zeros_like(
                self._admission_prefreeze_snapshot_valid
            )
            self._admission_insert_transition_translation_entry_m = torch.zeros_like(
                self._admission_prefreeze_translation_entry_m
            )
            self._admission_insert_transition_rotation_vector_entry_rad = torch.zeros_like(
                self._admission_prefreeze_translation_entry_m
            )
            self._admission_insert_transition_depth_m = torch.zeros(
                shape, dtype=torch.float64, device=self.device
            )
            self._admission_insert_transition_lateral_error_m = torch.zeros_like(
                self._admission_insert_transition_depth_m
            )
            self._admission_insert_transition_orientation_error_rad = torch.zeros_like(
                self._admission_insert_transition_depth_m
            )
            self._admission_transition_requested_action_6d = torch.zeros(
                (self.num_envs, 6), device=self.device
            )
            self._admission_transition_bounded_action_6d = torch.zeros_like(
                self._admission_transition_requested_action_6d
            )
            self._admission_transition_actual_applied_action_6d = torch.zeros_like(
                self._admission_transition_requested_action_6d
            )
            self._admission_transition_actual_tip_target_offset_entry_6d = torch.zeros_like(
                self._admission_transition_requested_action_6d
            )
            self._admission_transition_actual_action_is_exact_zero = torch.zeros_like(
                self._admission_prefreeze_snapshot_valid
            )
            self._admission_transition_contract_checked = torch.zeros_like(
                self._admission_prefreeze_snapshot_valid
            )

        @staticmethod
        def _rotation_vector(quaternion):
            quaternion = quaternion.to(dtype=torch.float64)
            quaternion = quaternion / torch.linalg.vector_norm(
                quaternion, dim=1, keepdim=True
            ).clamp_min(1.0e-12)
            quaternion = torch.where(
                (quaternion[:, :1] < 0.0), -quaternion, quaternion
            )
            vector = quaternion[:, 1:]
            norm = torch.linalg.vector_norm(vector, dim=1, keepdim=True)
            angle = 2.0 * torch.atan2(norm, quaternion[:, :1].clamp_min(0.0))
            return torch.where(
                norm > 1.0e-12, vector / norm.clamp_min(1.0e-12) * angle, vector * 2.0
            )

        @staticmethod
        def _bound_right6(actions):
            clipped = torch.clamp(actions, -1.0, 1.0).reshape(-1, 2, 3)
            norms = torch.linalg.vector_norm(clipped, dim=2, keepdim=True)
            bounded = clipped / torch.maximum(norms, torch.ones_like(norms))
            return bounded.reshape(-1, 6)

        def _identity_telemetry(self) -> dict[str, Any]:
            return {
                "applied_held_grasp_id": held_grasp_id,
                "applied_held_grasp_index": held_grasp_index,
                "applied_held_artifact_sha256": held_sha,
            }

        def _sample_episode_bias(self, env_ids) -> None:
            self._training_translation_bias_entry[env_ids] = (
                self._admission_requested_translation_bias_entry_m[env_ids].to(
                    dtype=self._training_translation_bias_entry.dtype
                )
            )
            self._training_rotation_bias_entry[env_ids] = (
                self._admission_requested_rotation_bias_entry_rad[env_ids].to(
                    dtype=self._training_rotation_bias_entry.dtype
                )
            )
            translation_error = torch.abs(
                self._training_translation_bias_entry[env_ids].to(torch.float64)
                - self._admission_requested_translation_bias_entry_m[env_ids]
            )
            rotation_error = torch.abs(
                self._training_rotation_bias_entry[env_ids].to(torch.float64)
                - self._admission_requested_rotation_bias_entry_rad[env_ids]
            )
            if bool(torch.any(translation_error > BIAS_ABSOLUTE_TOLERANCE)) or bool(
                torch.any(rotation_error > BIAS_ABSOLUTE_TOLERANCE)
            ):
                raise TaskAdmissionContractViolation(
                    "requested calibration bias was not applied within 1e-9"
                )
            self.extras["rj45_task_admission_identity"] = self._identity_telemetry()
            self.extras["rj45_task_admission_bias"] = {
                "requested_translation_bias_entry_m": (
                    self._admission_requested_translation_bias_entry_m.clone()
                ),
                "requested_rotation_bias_entry_rad": (
                    self._admission_requested_rotation_bias_entry_rad.clone()
                ),
                "applied_translation_bias_entry_m": (
                    self._training_translation_bias_entry.clone()
                ),
                "applied_rotation_bias_entry_rad": (
                    self._training_rotation_bias_entry.clone()
                ),
                "absolute_tolerance": BIAS_ABSOLUTE_TOLERANCE,
            }

        def _capture_signed_gt(self, mask):
            position, quaternion = self._relative_object_pose()
            rotation_vector = self._rotation_vector(quaternion)
            return position[mask].to(torch.float64), rotation_vector[mask]

        def _freeze_training_estimator(self, mask) -> None:
            if bool(torch.any(mask)):
                position, rotation_vector = self._capture_signed_gt(mask)
                self._admission_prefreeze_translation_entry_m[mask] = position
                self._admission_prefreeze_rotation_vector_entry_rad[mask] = rotation_vector
                self._admission_prefreeze_snapshot_valid[mask] = True
            super()._freeze_training_estimator(mask)

        def _advance_phases(self) -> None:
            before = self._phase.clone()
            super()._advance_phases()
            transitioned = (before == ALIGN_PHASE_CODE) & (
                self._phase == INSERT_PHASE_CODE
            )
            if not bool(torch.any(transitioned)):
                return
            position, rotation_vector = self._capture_signed_gt(transitioned)
            self._admission_insert_transition_translation_entry_m[transitioned] = position
            self._admission_insert_transition_rotation_vector_entry_rad[transitioned] = (
                rotation_vector
            )
            self._admission_insert_transition_depth_m[transitioned] = position[:, 1]
            self._admission_insert_transition_lateral_error_m[transitioned] = (
                torch.linalg.vector_norm(position[:, (0, 2)], dim=1)
            )
            self._admission_insert_transition_orientation_error_rad[transitioned] = (
                torch.linalg.vector_norm(rotation_vector, dim=1)
            )
            self._admission_insert_transition_snapshot_valid[transitioned] = True

        def _pre_physics_step(self, actions) -> None:
            if not isinstance(actions, torch.Tensor) or actions.shape != (
                self.num_envs,
                6,
            ):
                raise ValueError("admission action must be a (num_envs, 6) tensor")
            if not bool(torch.isfinite(actions).all()):
                raise ValueError("admission action must be finite")
            requested = actions.clone()
            bounded = self._bound_right6(requested)
            previously_captured = self._admission_insert_transition_snapshot_valid.clone()
            super()._pre_physics_step(actions)
            transitioned = (
                self._admission_insert_transition_snapshot_valid & ~previously_captured
            )
            if bool(torch.any(transitioned)):
                actual = self._comparison_applied_action_6d[transitioned]
                tip_offset = self._training_applied_tip_offset_entry[transitioned]
                self._admission_transition_requested_action_6d[transitioned] = requested[
                    transitioned
                ]
                self._admission_transition_bounded_action_6d[transitioned] = bounded[
                    transitioned
                ]
                self._admission_transition_actual_applied_action_6d[transitioned] = actual
                self._admission_transition_actual_tip_target_offset_entry_6d[
                    transitioned
                ] = tip_offset
                exact_zero = (actual == 0.0).all(dim=1) & (
                    tip_offset == 0.0
                ).all(dim=1)
                self._admission_transition_actual_action_is_exact_zero[
                    transitioned
                ] = exact_zero
                self._admission_transition_contract_checked[transitioned] = True
                if not bool(exact_zero.all()):
                    raise TaskAdmissionContractViolation(
                        "ALIGN-to-INSERT transition applied a nonzero residual or tip offset"
                    )
            self.extras["rj45_task_admission"] = self._admission_snapshot()

        def _admission_snapshot(self) -> dict[str, Any]:
            snapshot: dict[str, Any] = {
                **self._identity_telemetry(),
                "requested_translation_bias_entry_m": (
                    self._admission_requested_translation_bias_entry_m.clone()
                ),
                "requested_rotation_bias_entry_rad": (
                    self._admission_requested_rotation_bias_entry_rad.clone()
                ),
                "applied_translation_bias_entry_m": (
                    self._training_translation_bias_entry.clone()
                ),
                "applied_rotation_bias_entry_rad": (
                    self._training_rotation_bias_entry.clone()
                ),
                "prefreeze_snapshot_valid": self._admission_prefreeze_snapshot_valid.clone(),
                "prefreeze_translation_socket_entry_m_xyz": (
                    self._admission_prefreeze_translation_entry_m.clone()
                ),
                "prefreeze_shortest_rotation_vector_socket_entry_rad_xyz": (
                    self._admission_prefreeze_rotation_vector_entry_rad.clone()
                ),
                "insert_transition_snapshot_valid": (
                    self._admission_insert_transition_snapshot_valid.clone()
                ),
                "insert_transition_translation_socket_entry_m_xyz": (
                    self._admission_insert_transition_translation_entry_m.clone()
                ),
                "insert_transition_shortest_rotation_vector_socket_entry_rad_xyz": (
                    self._admission_insert_transition_rotation_vector_entry_rad.clone()
                ),
                "insert_transition_depth_m": self._admission_insert_transition_depth_m.clone(),
                "insert_transition_lateral_error_m": (
                    self._admission_insert_transition_lateral_error_m.clone()
                ),
                "insert_transition_orientation_error_rad": (
                    self._admission_insert_transition_orientation_error_rad.clone()
                ),
                "requested_action_6d": self._admission_transition_requested_action_6d.clone(),
                "bounded_action_6d": self._admission_transition_bounded_action_6d.clone(),
                "actual_applied_action_6d": (
                    self._admission_transition_actual_applied_action_6d.clone()
                ),
                "actual_applied_tip_target_offset_entry_6d": (
                    self._admission_transition_actual_tip_target_offset_entry_6d.clone()
                ),
                "actual_applied_action_is_exact_zero": (
                    self._admission_transition_actual_action_is_exact_zero.clone()
                ),
                "transition_contract_checked": (
                    self._admission_transition_contract_checked.clone()
                ),
                "evaluator_gt_available_to_fair_provider": False,
            }
            return snapshot

        def _training_step_snapshot(self):
            snapshot = super()._training_step_snapshot()
            snapshot["task_admission"] = self._admission_snapshot()
            return snapshot

        def _configure_training_reset(self, env_ids) -> None:
            super()._configure_training_reset(env_ids)
            self._admission_prefreeze_snapshot_valid[env_ids] = False
            self._admission_insert_transition_snapshot_valid[env_ids] = False
            self._admission_transition_contract_checked[env_ids] = False
            self._admission_transition_actual_action_is_exact_zero[env_ids] = False
            for tensor in (
                self._admission_prefreeze_translation_entry_m,
                self._admission_prefreeze_rotation_vector_entry_rad,
                self._admission_insert_transition_translation_entry_m,
                self._admission_insert_transition_rotation_vector_entry_rad,
                self._admission_insert_transition_depth_m,
                self._admission_insert_transition_lateral_error_m,
                self._admission_insert_transition_orientation_error_rad,
                self._admission_transition_requested_action_6d,
                self._admission_transition_bounded_action_6d,
                self._admission_transition_actual_applied_action_6d,
                self._admission_transition_actual_tip_target_offset_entry_6d,
            ):
                tensor[env_ids] = 0.0

    Rby1Rj45TaskAdmissionEnv.__name__ = "Rby1Rj45TaskAdmissionEnv"
    return Rby1Rj45TaskAdmissionEnv


__all__ = [
    "ALIGN_PHASE_CODE",
    "BIAS_ABSOLUTE_TOLERANCE",
    "INSERT_PHASE_CODE",
    "TaskAdmissionContractViolation",
    "make_rj45_task_admission_env_type",
]
