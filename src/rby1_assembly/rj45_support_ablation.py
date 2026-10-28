"""Opt-in support-arm correction ablation for verified RJ45 evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping


SUPPORT_ARM_MODES = ("cartesian_correction", "insert_joint_hold")


def validate_support_arm_mode(mode: str) -> str:
    if mode not in SUPPORT_ARM_MODES:
        raise ValueError(f"support arm mode must be one of {SUPPORT_ARM_MODES}")
    return mode


def _is_insert_transition(phase: object) -> bool:
    name = getattr(phase, "name", None)
    return isinstance(name, str) and name.lower() == "insert"


def _quat_error_rad(actual, reference):
    """Return shortest quaternion angle for WXYZ tensors."""

    import torch

    actual = actual / torch.linalg.vector_norm(actual, dim=1, keepdim=True).clamp_min(
        1.0e-12
    )
    reference = reference / torch.linalg.vector_norm(
        reference, dim=1, keepdim=True
    ).clamp_min(1.0e-12)
    dot = torch.sum(actual * reference, dim=1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def _finite_scalar(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"support-arm metric {label} must be finite")
    return result


def support_ablation_terminal_metrics(
    snapshot: Mapping[str, Any], env_id: int
) -> tuple[dict[str, object], bool]:
    """Extract one terminal row without claiming an identical action trajectory."""

    mode = snapshot.get("support_arm_mode")
    if mode not in SUPPORT_ARM_MODES:
        raise ValueError("terminal snapshot lacks a valid support-arm mode")
    metric_names = (
        "support_insert_duration_s",
        "support_insert_advance_pause_fraction",
        "support_socket_goal_translation_rms_m",
        "support_socket_goal_translation_max_m",
        "support_socket_goal_translation_final_m",
        "support_socket_goal_rotation_rms_rad",
        "support_socket_goal_rotation_max_rad",
        "support_socket_goal_rotation_final_rad",
        "support_left_ee_entry_translation_rms_m",
        "support_left_ee_entry_translation_max_m",
        "support_left_ee_entry_translation_final_m",
        "support_left_ee_entry_rotation_rms_rad",
        "support_left_ee_entry_rotation_max_rad",
        "support_left_ee_entry_rotation_final_rad",
        "support_left_joint_drift_rms_rad",
        "support_left_joint_drift_max_rad",
        "support_left_joint_drift_final_rad",
        "support_grip_force_peak_n",
        "support_insertion_load_peak_n",
        "support_forbidden_load_peak_n",
    )
    metrics: dict[str, object] = {"support_arm_mode": mode}
    finite = True
    for name in metric_names:
        value = snapshot.get(name)
        if value is None:
            raise ValueError(f"terminal snapshot lacks {name}")
        row = value[env_id]
        if getattr(row, "numel", lambda: 1)() != 1:
            raise ValueError(f"terminal support-arm metric {name} must be scalar")
        try:
            metrics[name] = _finite_scalar(row.item(), name)
        except ValueError:
            metrics[name] = None
            finite = False
    valid = snapshot.get("support_insert_reference_valid")
    if valid is None:
        raise ValueError("terminal snapshot lacks support_insert_reference_valid")
    metrics["support_insert_reference_valid"] = bool(valid[env_id].item())
    metrics["support_metric_scope"] = (
        "scripted support-arm Cartesian correction comparison; policy action "
        "trajectories may diverge between conditions"
    )
    return metrics, finite


def make_rj45_support_ablation_env_type(base_training_cls, mode: str):
    """Wrap a training environment with telemetry and optional INSERT joint hold.

    ``insert_joint_hold`` freezes the measured left-arm joint vector at INSERT
    entry, zeros only the desired left Cartesian twist, and preserves the base
    class's bounded joint drive, gripper, torso, gravity, effort, limits, and
    right-arm policy path. No simulator state is written by this wrapper.
    """

    import torch

    selected_mode = validate_support_arm_mode(mode)

    class Rby1Rj45SupportAblationEnv(base_training_cls):
        def _initialize_training_state(self) -> None:
            super()._initialize_training_state()
            left_count = len(self._left_arm_joint_ids)
            self._support_insert_joint_reference = torch.zeros(
                (self.num_envs, left_count), device=self.device
            )
            self._support_insert_reference_valid = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._support_insert_phase_code = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self._support_insert_entry_left_ee_pose = torch.zeros(
                (self.num_envs, 7), device=self.device
            )
            self._support_insert_entry_left_ee_pose[:, 3] = 1.0
            self._support_insert_entry_socket_pose = torch.zeros(
                (self.num_envs, 7), device=self.device
            )
            self._support_insert_entry_socket_pose[:, 3] = 1.0
            self._support_metric_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._support_pause_count = torch.zeros_like(self._support_metric_count)
            self._support_step_active = torch.zeros_like(self._support_insert_reference_valid)
            self._support_last_sample_physics_step = torch.full_like(self._support_metric_count, -1)
            for name in (
                "socket_translation_sq",
                "socket_translation_max",
                "socket_translation_final",
                "socket_rotation_sq",
                "socket_rotation_max",
                "socket_rotation_final",
                "ee_translation_sq",
                "ee_translation_max",
                "ee_translation_final",
                "ee_rotation_sq",
                "ee_rotation_max",
                "ee_rotation_final",
                "joint_drift_sq",
                "joint_drift_max",
                "joint_drift_final",
            ):
                setattr(
                    self,
                    f"_support_{name}",
                    torch.zeros(self.num_envs, device=self.device),
                )
            self._support_arm_mode = selected_mode

        def _transition_where(self, mask, phase, now, reason=None):
            super()._transition_where(mask, phase, now, reason)
            if not _is_insert_transition(phase) or not torch.any(mask):
                return
            entered = mask & (self._phase == int(phase))
            if not torch.any(entered):
                return
            left_joint = self.robot.data.joint_pos[:, self._left_arm_joint_ids]
            self._support_insert_joint_reference[entered] = left_joint[entered]
            self._support_insert_reference_valid[entered] = True
            self._support_insert_phase_code[entered] = int(phase)
            self._support_insert_entry_left_ee_pose[entered, :3] = (
                self.robot.data.body_pos_w[entered, self._left_ee_body_id]
            )
            self._support_insert_entry_left_ee_pose[entered, 3:] = (
                self.robot.data.body_quat_w[entered, self._left_ee_body_id]
            )
            self._support_insert_entry_socket_pose[entered, :3] = (
                self.socket.data.root_pos_w[entered]
            )
            self._support_insert_entry_socket_pose[entered, 3:] = (
                self.socket.data.root_quat_w[entered]
            )
            if self._support_arm_mode == "insert_joint_hold":
                left_slice = slice(0, len(self._left_arm_joint_ids))
                self._insert_joint_targets[entered, left_slice] = (
                    self._support_insert_joint_reference[entered]
                )

        def _desired_ee_twists(self):
            left_twist, right_twist = super()._desired_ee_twists()
            if self._support_arm_mode == "insert_joint_hold":
                active = self._support_insert_reference_valid & self._phase_is_insert()
                left_twist = left_twist.clone()
                left_twist[active] = 0.0
                if hasattr(self, "_baseline_cartesian_twists"):
                    self._baseline_cartesian_twists[active, 0] = 0.0
            return left_twist, right_twist

        def _phase_is_insert(self):
            return self._phase == self._support_insert_phase_code

        def _pre_physics_step(self, actions) -> None:
            super()._pre_physics_step(actions)
            active = self._support_insert_reference_valid & self._phase_is_insert()
            self._support_step_active[:] = active
            if self._support_arm_mode == "insert_joint_hold" and torch.any(active):
                left_slice = slice(0, len(self._left_arm_joint_ids))
                self._insert_joint_targets[active, left_slice] = (
                    self._support_insert_joint_reference[active]
                )

        def _get_dones(self):
            # Sample the state produced by this INSERT control step before the
            # parent's terminal snapshot or the following phase transition.
            if hasattr(self, "_support_step_active"):
                physics_step = int(self._sim_step_counter)
                active = self._support_step_active & (
                    self._support_last_sample_physics_step != physics_step
                )
                self._update_support_ablation_metrics(active)
                self._support_last_sample_physics_step[active] = physics_step
            return super()._get_dones()

        def _update_support_ablation_metrics(self, active) -> None:
            if not torch.any(active):
                return
            socket_goal = self.scene.env_origins + torch.tensor(
                self.cfg.assembly_socket_goal_m,
                device=self.device,
                dtype=self.socket.data.root_pos_w.dtype,
            )
            socket_goal_quat = torch.tensor(
                self.cfg.assembly_socket_goal_quat_wxyz,
                device=self.device,
                dtype=self.socket.data.root_quat_w.dtype,
            ).repeat(self.num_envs, 1)
            left_pos = self.robot.data.body_pos_w[:, self._left_ee_body_id]
            left_quat = self.robot.data.body_quat_w[:, self._left_ee_body_id]
            left_joint = self.robot.data.joint_pos[:, self._left_arm_joint_ids]
            values = {
                "socket_translation": torch.linalg.vector_norm(
                    self.socket.data.root_pos_w - socket_goal, dim=1
                ),
                "socket_rotation": _quat_error_rad(
                    self.socket.data.root_quat_w, socket_goal_quat
                ),
                "ee_translation": torch.linalg.vector_norm(
                    left_pos - self._support_insert_entry_left_ee_pose[:, :3], dim=1
                ),
                "ee_rotation": _quat_error_rad(
                    left_quat, self._support_insert_entry_left_ee_pose[:, 3:]
                ),
                "joint_drift": torch.linalg.vector_norm(
                    left_joint - self._support_insert_joint_reference, dim=1
                ),
            }
            self._support_metric_count[active] += 1
            self._support_pause_count[active] += (
                ~self._insertion_advance_enabled[active]
            ).to(dtype=torch.long)
            for name, value in values.items():
                square = getattr(self, f"_support_{name}_sq")
                maximum = getattr(self, f"_support_{name}_max")
                final = getattr(self, f"_support_{name}_final")
                square[active] += value[active].square()
                maximum[active] = torch.maximum(maximum[active], value[active])
                final[active] = value[active]

        def _configure_training_reset(self, env_ids):
            super()._configure_training_reset(env_ids)
            if not hasattr(self, "_support_insert_reference_valid"):
                return
            self._support_insert_reference_valid[env_ids] = False
            self._support_insert_phase_code[env_ids] = -1
            self._support_insert_joint_reference[env_ids] = 0.0
            self._support_insert_entry_left_ee_pose[env_ids] = 0.0
            self._support_insert_entry_left_ee_pose[env_ids, 3] = 1.0
            self._support_insert_entry_socket_pose[env_ids] = 0.0
            self._support_insert_entry_socket_pose[env_ids, 3] = 1.0
            self._support_metric_count[env_ids] = 0
            self._support_pause_count[env_ids] = 0
            self._support_step_active[env_ids] = False
            self._support_last_sample_physics_step[env_ids] = -1
            for name in (
                "socket_translation_sq",
                "socket_translation_max",
                "socket_translation_final",
                "socket_rotation_sq",
                "socket_rotation_max",
                "socket_rotation_final",
                "ee_translation_sq",
                "ee_translation_max",
                "ee_translation_final",
                "ee_rotation_sq",
                "ee_rotation_max",
                "ee_rotation_final",
                "joint_drift_sq",
                "joint_drift_max",
                "joint_drift_final",
            ):
                getattr(self, f"_support_{name}")[env_ids] = 0.0

        def _training_step_snapshot(self):
            snapshot = super()._training_step_snapshot()
            count = self._support_metric_count.clamp_min(1)
            count_float = count.to(dtype=self._support_socket_translation_sq.dtype)
            snapshot.update(
                {
                    "support_arm_mode": self._support_arm_mode,
                    "support_insert_reference_valid": (
                        self._support_insert_reference_valid.clone()
                    ),
                    "support_insert_joint_reference_rad": (
                        self._support_insert_joint_reference.clone()
                    ),
                    "support_insert_entry_left_ee_pose_world_wxyz": (
                        self._support_insert_entry_left_ee_pose.clone()
                    ),
                    "support_insert_entry_socket_pose_world_wxyz": (
                        self._support_insert_entry_socket_pose.clone()
                    ),
                    "support_insert_duration_s": (
                        self._support_metric_count.to(dtype=count_float.dtype)
                        * self.step_dt
                    ),
                    "support_insert_advance_pause_fraction": (
                        self._support_pause_count.to(dtype=count_float.dtype)
                        / count_float
                    ),
                    "support_grip_force_peak_n": (
                        self._episode_grip_force_peak_n.clone()
                    ),
                    "support_insertion_load_peak_n": (
                        self._episode_insertion_contact_load_peak_n.clone()
                    ),
                    "support_forbidden_load_peak_n": (
                        self._episode_forbidden_contact_load_peak_n.clone()
                    ),
                }
            )
            for prefix in (
                "socket_translation",
                "socket_rotation",
                "ee_translation",
                "ee_rotation",
                "joint_drift",
            ):
                output = {
                    "socket_translation": "support_socket_goal_translation",
                    "socket_rotation": "support_socket_goal_rotation",
                    "ee_translation": "support_left_ee_entry_translation",
                    "ee_rotation": "support_left_ee_entry_rotation",
                    "joint_drift": "support_left_joint_drift",
                }[prefix]
                unit = "m" if "translation" in output else "rad"
                snapshot[f"{output}_rms_{unit}"] = torch.sqrt(
                    getattr(self, f"_support_{prefix}_sq") / count_float
                )
                snapshot[f"{output}_max_{unit}"] = getattr(
                    self, f"_support_{prefix}_max"
                ).clone()
                snapshot[f"{output}_final_{unit}"] = getattr(
                    self, f"_support_{prefix}_final"
                ).clone()
            return snapshot

        def rj45_support_ablation_report(self) -> dict[str, object]:
            return {
                "mode": self._support_arm_mode,
                "comparison_scope": "scripted support-arm Cartesian correction",
                "identical_action_trajectory_claim": False,
                "learned_bimanual_effect_claim": False,
                "state_writes_or_teleports": False,
                "insert_reference": "actual left joint position at accepted INSERT entry",
                "geometry_sampling": "post-physics _get_dones before terminal snapshot; once per physics step",
            }

    Rby1Rj45SupportAblationEnv.__name__ = "Rby1Rj45SupportAblationEnv"
    return Rby1Rj45SupportAblationEnv


__all__ = [
    "SUPPORT_ARM_MODES",
    "make_rj45_support_ablation_env_type",
    "support_ablation_terminal_metrics",
    "validate_support_arm_mode",
]
