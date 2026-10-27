"""Isaac runtime wrapper for estimator-separated RJ45 residual learning."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from rby1_assembly.rj45_training import ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY


_CALIBRATION_BIAS_STREAM_OFFSET = 45_123_017
_CALIBRATION_BIAS_EPISODE_STRIDE = 1_000_003


def _reset_applied_tip_offset_telemetry(env_ids, applied_tip_offset_entry_6d) -> None:
    """Clear selected episode rows without disturbing parallel environments."""

    applied_tip_offset_entry_6d[env_ids] = 0.0


def _reset_reward_telemetry(
    env_ids,
    component_sums,
    policy_active_step_count,
    action_saturation_step_count,
) -> None:
    """Clear reward telemetry only for reset rows in a vector environment."""

    for value in component_sums.values():
        value[env_ids] = 0.0
    policy_active_step_count[env_ids] = 0
    action_saturation_step_count[env_ids] = 0


def _reward_policy_active_mask(config, current_active, transition_active):
    """Keep A0 gating unchanged and use the executed transition for A1/A2."""

    return transition_active if config.get("schema_version") == 2 else current_active


def _executed_policy_transition_mask(
    config,
    current_active,
    warmup_pending,
    last_policy_observation_valid,
    valid_policy_input_step_count,
):
    """Identify rows whose submitted action came from an eligible policy observation."""

    active = current_active & ~warmup_pending
    if config["action_application_mode"] != ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY:
        return active
    if config.get("schema_version") == 2:
        return active & last_policy_observation_valid
    return active & (valid_policy_input_step_count > 0)


def _action_saturation_masks(expanded_actions, active_mask):
    """Separate wrapper clip-boundary hits from 3D vector-norm projection."""

    import torch

    if expanded_actions.ndim != 2 or expanded_actions.shape[1] != 12:
        raise ValueError("expanded_actions must have shape (num_envs, 12)")
    if active_mask.shape != (expanded_actions.shape[0],) or active_mask.dtype != torch.bool:
        raise ValueError("active_mask must be boolean per environment")
    clip_boundary_reached = (torch.abs(expanded_actions) >= 1.0).any(dim=1) & active_mask
    clipped = torch.clamp(expanded_actions, -1.0, 1.0).reshape(-1, 4, 3)
    norm_projected = (
        (torch.linalg.vector_norm(clipped, dim=2) > 1.0).any(dim=1) & active_mask
    )
    return clip_boundary_reached, norm_projected


def _calibration_bias_case_seed(base_seed: int, env_id: int, episode_index: int) -> int:
    if any(type(value) is not int or value < 0 for value in (base_seed, env_id, episode_index)):
        raise ValueError("bias seed inputs must be non-negative integers")
    return (
        base_seed
        + _CALIBRATION_BIAS_STREAM_OFFSET
        + episode_index * _CALIBRATION_BIAS_EPISODE_STRIDE
        + env_id
    )


def _warmup_sensor_status(
    finger_force_n,
    insertion_force_world_n,
    insertion_load_n,
    forbidden_load_n,
    *,
    grip_contact_force_n: float,
    hard_contact_force_n: float,
    forbidden_contact_force_n: float,
):
    """Return finite, limit, and live-grip masks for warmup sensors."""

    import torch

    if finger_force_n.ndim != 2 or finger_force_n.shape[1] != 4:
        raise ValueError("finger_force_n must have shape (num_envs, 4)")
    if insertion_force_world_n.shape != (finger_force_n.shape[0], 3):
        raise ValueError("insertion_force_world_n must have shape (num_envs, 3)")
    if insertion_load_n.shape != (finger_force_n.shape[0],) or forbidden_load_n.shape != (
        finger_force_n.shape[0],
    ):
        raise ValueError("warmup scalar loads must have shape (num_envs,)")
    finite = (
        torch.isfinite(finger_force_n).all(dim=1)
        & torch.isfinite(insertion_force_world_n).all(dim=1)
        & torch.isfinite(insertion_load_n)
        & torch.isfinite(forbidden_load_n)
    )
    within_limits = (
        (finger_force_n.amax(dim=1) <= hard_contact_force_n)
        & (insertion_load_n <= hard_contact_force_n)
        & (forbidden_load_n <= forbidden_contact_force_n)
    )
    grip_live = (finger_force_n >= grip_contact_force_n).all(dim=1)
    return finite, within_limits, grip_live


def _warmup_motion_status(
    joint_velocity_si,
    joint_reference_error_si,
    prismatic_joint_mask,
    socket_com_velocity_world,
    plug_com_velocity_world,
    relative_linear_velocity,
    relative_angular_velocity,
    *,
    limits,
    max_relative_linear_speed_m_s: float,
    max_relative_angular_speed_rad_s: float,
):
    """Return finite and settled masks using absolute and relative motion."""

    import torch

    if joint_velocity_si.ndim != 2:
        raise ValueError("joint_velocity_si must have shape (num_envs, num_joints)")
    if joint_reference_error_si.shape != joint_velocity_si.shape:
        raise ValueError("joint reference error must match joint velocity shape")
    if prismatic_joint_mask.shape != (joint_velocity_si.shape[1],):
        raise ValueError("prismatic_joint_mask must select the joint dimension")
    if prismatic_joint_mask.dtype != torch.bool:
        raise ValueError("prismatic_joint_mask must be boolean")
    num_envs = joint_velocity_si.shape[0]
    if socket_com_velocity_world.shape != (num_envs, 6):
        raise ValueError("socket COM velocity must have shape (num_envs, 6)")
    if plug_com_velocity_world.shape != (num_envs, 6):
        raise ValueError("plug COM velocity must have shape (num_envs, 6)")
    if relative_linear_velocity.shape != (num_envs, 3):
        raise ValueError("relative linear velocity must have shape (num_envs, 3)")
    if relative_angular_velocity.shape != (num_envs, 3):
        raise ValueError("relative angular velocity must have shape (num_envs, 3)")

    finite = (
        torch.isfinite(joint_velocity_si).all(dim=1)
        & torch.isfinite(joint_reference_error_si).all(dim=1)
        & torch.isfinite(socket_com_velocity_world).all(dim=1)
        & torch.isfinite(plug_com_velocity_world).all(dim=1)
        & torch.isfinite(relative_linear_velocity).all(dim=1)
        & torch.isfinite(relative_angular_velocity).all(dim=1)
    )
    revolute_joint_mask = ~prismatic_joint_mask
    settled = (
        (
            joint_velocity_si[:, revolute_joint_mask].abs()
            <= limits["max_revolute_joint_speed_rad_s"]
        ).all(dim=1)
        & (
            joint_velocity_si[:, prismatic_joint_mask].abs()
            <= limits["max_prismatic_joint_speed_m_s"]
        ).all(dim=1)
        & (
            joint_reference_error_si[:, revolute_joint_mask].abs()
            <= limits["max_revolute_tracking_error_rad"]
        ).all(dim=1)
        & (
            joint_reference_error_si[:, prismatic_joint_mask].abs()
            <= limits["max_prismatic_tracking_error_m"]
        ).all(dim=1)
        & (
            torch.linalg.vector_norm(socket_com_velocity_world[:, :3], dim=1)
            <= limits["max_object_linear_speed_m_s"]
        )
        & (
            torch.linalg.vector_norm(socket_com_velocity_world[:, 3:], dim=1)
            <= limits["max_object_angular_speed_rad_s"]
        )
        & (
            torch.linalg.vector_norm(plug_com_velocity_world[:, :3], dim=1)
            <= limits["max_object_linear_speed_m_s"]
        )
        & (
            torch.linalg.vector_norm(plug_com_velocity_world[:, 3:], dim=1)
            <= limits["max_object_angular_speed_rad_s"]
        )
        & (
            torch.linalg.vector_norm(relative_linear_velocity, dim=1)
            <= max_relative_linear_speed_m_s
        )
        & (
            torch.linalg.vector_norm(relative_angular_velocity, dim=1)
            <= max_relative_angular_speed_rad_s
        )
    )
    return finite, settled


def _warmup_timeout_mask(pending, passed, warmup_steps, timeout_steps: int):
    """Return pending rows that timed out without passing on this step."""

    return pending & ~passed & (warmup_steps >= timeout_steps)


def _held_reset_stage_transition(queued, waited_control_steps, control_steps: int):
    """Return restore-ready rows and the next per-row staging counters."""

    import torch

    if type(control_steps) is not int or control_steps < 1:
        raise ValueError("held reset staging control_steps must be a positive integer")
    if queued.shape != waited_control_steps.shape:
        raise ValueError("held reset staging tensors must have identical shapes")
    if queued.dtype != torch.bool:
        raise ValueError("held reset queued mask must be boolean")
    ready = queued & (waited_control_steps >= control_steps)
    still_waiting = queued & ~ready
    next_waited = torch.where(
        still_waiting,
        waited_control_steps + 1,
        waited_control_steps,
    )
    return ready, next_waited


def _held_reset_contact_free_status(
    finger_force_n,
    finger_raw_contact_count,
    insertion_force_world_n,
    insertion_load_n,
    insertion_raw_contact_count,
    forbidden_load_n,
):
    """Require finite, exactly zero contact evidence before held restoration."""

    import torch

    num_envs = finger_force_n.shape[0]
    if finger_force_n.shape != (num_envs, 4):
        raise ValueError("finger_force_n must have shape (num_envs, 4)")
    if finger_raw_contact_count.shape != (num_envs, 4):
        raise ValueError("finger raw contact counts must have shape (num_envs, 4)")
    if insertion_force_world_n.shape != (num_envs, 3):
        raise ValueError("insertion force must have shape (num_envs, 3)")
    for name, value in (
        ("insertion_load_n", insertion_load_n),
        ("insertion_raw_contact_count", insertion_raw_contact_count),
        ("forbidden_load_n", forbidden_load_n),
    ):
        if value.shape != (num_envs,):
            raise ValueError(f"{name} must have shape (num_envs,)")
    finite = (
        torch.isfinite(finger_force_n).all(dim=1)
        & torch.isfinite(insertion_force_world_n).all(dim=1)
        & torch.isfinite(insertion_load_n)
        & torch.isfinite(forbidden_load_n)
    )
    exactly_zero = (
        (finger_force_n == 0.0).all(dim=1)
        & (finger_raw_contact_count == 0).all(dim=1)
        & (insertion_force_world_n == 0.0).all(dim=1)
        & (insertion_load_n == 0.0)
        & (insertion_raw_contact_count == 0)
        & (forbidden_load_n == 0.0)
    )
    return finite, exactly_zero, finite & exactly_zero


def _mask_inactive_policy_observation(
    observation, policy_active, expected_dimension: int = 79
):
    """Zero every policy row that is not valid policy input."""

    import torch

    if type(expected_dimension) is not int or expected_dimension not in (79, 85):
        raise ValueError("RJ45 expected policy observation dimension must be 79 or 85")
    if observation.ndim != 2 or observation.shape[1] != expected_dimension:
        raise ValueError(
            "RJ45 policy observation must have shape "
            f"(num_envs, {expected_dimension})"
        )
    if policy_active.shape != (observation.shape[0],):
        raise ValueError("policy_active must select the observation environment axis")
    valid = policy_active.to(dtype=torch.bool)
    return torch.where(valid[:, None], observation, torch.zeros_like(observation)), valid


def _store_policy_input_telemetry(
    observation,
    valid,
    timestamp_s: float,
    observation_buffer,
    valid_buffer,
    timestamp_buffer,
) -> None:
    """Copy one already-produced policy input into device-resident buffers."""

    if observation.shape != observation_buffer.shape:
        raise ValueError("policy observation telemetry shape mismatch")
    if valid.shape != valid_buffer.shape or timestamp_buffer.shape != valid.shape:
        raise ValueError("policy input telemetry mask shape mismatch")
    observation_buffer.copy_(observation)
    valid_buffer.copy_(valid)
    timestamp_buffer.fill_(float(timestamp_s))


def _reset_policy_input_telemetry(
    env_ids,
    observation_buffer,
    valid_buffer,
    timestamp_buffer,
) -> None:
    """Clear only reset environments while preserving every other row."""

    observation_buffer[env_ids] = 0.0
    valid_buffer[env_ids] = False
    timestamp_buffer[env_ids] = -1.0


def _diagnostic_number(value: Any) -> dict[str, Any]:
    """Encode one numeric reading as a JSON-safe value and finite-state flag."""

    number = float(value)
    if math.isfinite(number):
        encoded: float | str = number
        nonfinite = False
    elif math.isnan(number):
        encoded = "nan"
        nonfinite = True
    elif number > 0.0:
        encoded = "+inf"
        nonfinite = True
    else:
        encoded = "-inf"
        nonfinite = True
    return {"value": encoded, "nonfinite": nonfinite}


def _diagnostic_vector(values) -> list[dict[str, Any]]:
    return [_diagnostic_number(value) for value in values.detach().cpu().tolist()]


def _diagnostic_abs_max(values) -> dict[str, Any] | None:
    if values.numel() == 0:
        return None
    return _diagnostic_number(values.abs().amax().detach().cpu().item())


def _diagnostic_nonfinite_readings(
    named_values: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Record every raw non-finite reading with its source-local index."""

    import torch

    readings = []
    for field, values in named_values.items():
        values = torch.as_tensor(values)
        for index in torch.nonzero(~torch.isfinite(values), as_tuple=False).tolist():
            tensor_index = tuple(index)
            readings.append(
                {
                    "field": field,
                    "index": [int(component) for component in index],
                    "reading": _diagnostic_number(values[tensor_index].item()),
                }
            )
    return readings


def _held_warmup_timeout_diagnostics(
    *,
    timed_out,
    finger_force_n,
    lift_left_m,
    lift_right_m,
    joint_velocity_si,
    joint_reference_error_si,
    prismatic_joint_mask,
    socket_com_velocity_world,
    plug_com_velocity_world,
    relative_linear_velocity_m_s,
    relative_angular_velocity_rad_s,
    insertion_force_world_n,
    insertion_load_n,
    forbidden_load_n,
    state_finite,
    motion_finite,
    motion_settled,
    sensor_finite,
    within_limits,
    grip_live,
    stable,
    warmup_steps,
    warmup_stable_steps,
    required_stable_steps: int,
    timeout_steps: int,
    step_dt_s: float,
    grip_lift_proof_m: float,
    limits: Mapping[str, float],
) -> dict[str, Any]:
    """Capture every failed environment at a held-warmup timeout."""

    environment_ids = timed_out.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()
    revolute_joint_mask = ~prismatic_joint_mask
    environments = []
    for env_id in environment_ids:
        socket_linear = socket_com_velocity_world[env_id, :3]
        socket_angular = socket_com_velocity_world[env_id, 3:]
        plug_linear = plug_com_velocity_world[env_id, :3]
        plug_angular = plug_com_velocity_world[env_id, 3:]
        lift_left_ok = bool((lift_left_m[env_id] >= grip_lift_proof_m).item())
        lift_right_ok = bool((lift_right_m[env_id] >= grip_lift_proof_m).item())
        nonfinite_readings = _diagnostic_nonfinite_readings(
            {
                "finger_force_n": finger_force_n[env_id],
                "lift_left_m": lift_left_m[env_id : env_id + 1],
                "lift_right_m": lift_right_m[env_id : env_id + 1],
                "joint_velocity_si": joint_velocity_si[env_id],
                "joint_reference_error_si": joint_reference_error_si[env_id],
                "socket_com_velocity_world": socket_com_velocity_world[env_id],
                "plug_com_velocity_world": plug_com_velocity_world[env_id],
                "relative_linear_velocity_m_s": relative_linear_velocity_m_s[env_id],
                "relative_angular_velocity_rad_s": relative_angular_velocity_rad_s[
                    env_id
                ],
                "insertion_force_world_n": insertion_force_world_n[env_id],
                "insertion_load_n": insertion_load_n[env_id : env_id + 1],
                "forbidden_load_n": forbidden_load_n[env_id : env_id + 1],
            }
        )
        environments.append(
            {
                "env_id": int(env_id),
                "nonfinite_readings": nonfinite_readings,
                "steps": {
                    "warmup": int(warmup_steps[env_id].item()),
                    "stable": int(warmup_stable_steps[env_id].item()),
                    "required_stable": int(required_stable_steps),
                    "timeout": int(timeout_steps),
                },
                "finger_force_n": {
                    name: _diagnostic_number(finger_force_n[env_id, index].item())
                    for index, name in enumerate(("left_l1", "left_l2", "right_r1", "right_r2"))
                },
                "lift_m": {
                    "left": _diagnostic_number(lift_left_m[env_id].item()),
                    "right": _diagnostic_number(lift_right_m[env_id].item()),
                },
                "joint_unit_maxima": {
                    "revolute_velocity_rad_s": _diagnostic_abs_max(
                        joint_velocity_si[env_id, revolute_joint_mask]
                    ),
                    "prismatic_velocity_m_s": _diagnostic_abs_max(
                        joint_velocity_si[env_id, prismatic_joint_mask]
                    ),
                    "revolute_reference_error_rad": _diagnostic_abs_max(
                        joint_reference_error_si[env_id, revolute_joint_mask]
                    ),
                    "prismatic_reference_error_m": _diagnostic_abs_max(
                        joint_reference_error_si[env_id, prismatic_joint_mask]
                    ),
                },
                "object_com_velocity_world": {
                    "socket": {
                        "linear_m_s": _diagnostic_vector(socket_linear),
                        "angular_rad_s": _diagnostic_vector(socket_angular),
                        "linear_norm_m_s": _diagnostic_number(
                            socket_linear.norm().item()
                        ),
                        "angular_norm_rad_s": _diagnostic_number(
                            socket_angular.norm().item()
                        ),
                    },
                    "plug": {
                        "linear_m_s": _diagnostic_vector(plug_linear),
                        "angular_rad_s": _diagnostic_vector(plug_angular),
                        "linear_norm_m_s": _diagnostic_number(plug_linear.norm().item()),
                        "angular_norm_rad_s": _diagnostic_number(plug_angular.norm().item()),
                    },
                },
                "relative_velocity": {
                    "linear_m_s": _diagnostic_vector(
                        relative_linear_velocity_m_s[env_id]
                    ),
                    "angular_rad_s": _diagnostic_vector(
                        relative_angular_velocity_rad_s[env_id]
                    ),
                },
                "contact_loads": {
                    "insertion_force_world_n": _diagnostic_vector(
                        insertion_force_world_n[env_id]
                    ),
                    "insertion_load_n": _diagnostic_number(
                        insertion_load_n[env_id].item()
                    ),
                    "forbidden_load_n": _diagnostic_number(
                        forbidden_load_n[env_id].item()
                    ),
                },
                "gates": {
                    "state_finite": bool(state_finite[env_id].item()),
                    "motion_finite": bool(motion_finite[env_id].item()),
                    "motion_settled": bool(motion_settled[env_id].item()),
                    "sensor_finite": bool(sensor_finite[env_id].item()),
                    "within_limits": bool(within_limits[env_id].item()),
                    "grip_live": bool(grip_live[env_id].item()),
                    "lift_left_ok": lift_left_ok,
                    "lift_right_ok": lift_right_ok,
                    "stable": bool(stable[env_id].item()),
                },
            }
        )
    diagnostics = {
        "schema_version": 1,
        "kind": "held_grasp_warmup_timeout",
        "failed_environment_ids": [int(env_id) for env_id in environment_ids],
        "timing": {
            "step_dt_s": _diagnostic_number(step_dt_s),
            "required_stable_steps": int(required_stable_steps),
            "timeout_steps": int(timeout_steps),
        },
        "thresholds": {
            key: _diagnostic_number(value)
            for key, value in {
                **dict(limits),
                "grip_lift_proof_m": grip_lift_proof_m,
            }.items()
        },
        "environments": environments,
    }
    nonfinite_locations = [
        {
            "env_id": environment["env_id"],
            "field": reading["field"],
            "index": reading["index"],
        }
        for environment in environments
        for reading in environment["nonfinite_readings"]
    ]
    diagnostics["nonfinite_summary"] = {
        "count": len(nonfinite_locations),
        "locations": nonfinite_locations,
    }
    return diagnostics


class HeldGraspWarmupTimeoutError(RuntimeError):
    """Runtime error carrying a JSON-safe timeout diagnostic summary."""

    def __init__(self, diagnostics: Mapping[str, Any]):
        self.diagnostics = dict(diagnostics)
        encoded = json.dumps(
            self.diagnostics,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(f"held-grasp warmup failed physical revalidation: {encoded}")


def _held_reset_contact_flush_diagnostics(
    failed,
    finger_force_n,
    finger_raw_contact_count,
    insertion_force_world_n,
    insertion_load_n,
    insertion_raw_contact_count,
    forbidden_load_n,
) -> dict[str, Any]:
    environment_ids = failed.nonzero(as_tuple=False).squeeze(1).detach().cpu().tolist()
    finite, exactly_zero, _ = _held_reset_contact_free_status(
        finger_force_n,
        finger_raw_contact_count,
        insertion_force_world_n,
        insertion_load_n,
        insertion_raw_contact_count,
        forbidden_load_n,
    )
    return {
        "schema_version": 1,
        "kind": "held_reset_contact_flush_validation_failed",
        "failed_environment_ids": [int(env_id) for env_id in environment_ids],
        "environments": [
            {
                "env_id": int(env_id),
                "finite": bool(finite[env_id].item()),
                "exactly_zero": bool(exactly_zero[env_id].item()),
                "finger_force_n": _diagnostic_vector(finger_force_n[env_id]),
                "finger_raw_contact_count": [
                    int(value)
                    for value in finger_raw_contact_count[env_id]
                    .detach()
                    .cpu()
                    .tolist()
                ],
                "insertion_force_world_n": _diagnostic_vector(
                    insertion_force_world_n[env_id]
                ),
                "insertion_load_n": _diagnostic_number(
                    insertion_load_n[env_id].item()
                ),
                "insertion_raw_contact_count": int(
                    insertion_raw_contact_count[env_id].item()
                ),
                "forbidden_load_n": _diagnostic_number(
                    forbidden_load_n[env_id].item()
                ),
            }
            for env_id in environment_ids
        ],
    }


class HeldResetContactFlushError(RuntimeError):
    """Runtime error carrying exact-zero contact-flush diagnostics."""

    def __init__(self, diagnostics: Mapping[str, Any]):
        self.diagnostics = dict(diagnostics)
        encoded = json.dumps(
            self.diagnostics,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        super().__init__(f"held reset contact flush validation failed: {encoded}")


def make_rj45_training_env_type(
    base_cls: type,
    training_config: Mapping[str, Any],
    held_state: Mapping[str, Any] | None = None,
    calibration_bias_enabled: bool = True,
    bias_level: int = 0,
) -> type:
    """Wrap a configured RJ45 environment with the 79D residual-RL contract.

    The wrapped controller uses a frozen hand/object estimator after one
    calibration event. Simulator object state remains available only to the
    independent evaluator and reward calculation.
    """

    import torch
    from isaaclab.utils.math import (
        combine_frame_transforms,
        quat_apply,
        quat_apply_inverse,
        subtract_frame_transforms,
    )

    from rby1_assembly.contact_loads import sum_contact_normal_loads
    from rby1_assembly.isaac_env import AssemblyPhase, TerminationReason
    from rby1_assembly.rj45_training import (
        HELD_GRASP_STABILITY_LIMITS,
        PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION,
        POST_INTEGRAL_STEP_ACTION_APPLICATION,
        TIP_TARGET_POSE_OFFSET_DESCRIPTION,
        apply_relative_pose_bias,
        apply_tip_target_pose_offset,
        append_persistent_calibration_bias_diagnostic,
        build_policy_observation,
        expand_residual_actions,
        insertion_potential,
        policy_observation_dimension,
        policy_ground_truth_usage,
        policy_observation_ground_truth_flags,
        policy_observation_source,
        residual_reward,
        residual_reward_components,
        reward_profile_metadata,
        validate_reward_horizon_contract,
        validate_held_grasp_artifact,
    )

    config = dict(training_config)
    required = {
        "action_mode",
        "action_application_mode",
        "base_observation_dim",
        "diagnostic_observation_mode",
        "active_horizon_s",
        "warmup_s",
        "held_reset_staging",
        "calibration_bias",
        "reward",
    }
    if not required.issubset(config):
        raise ValueError(f"training configuration is missing {sorted(required - set(config))}")
    if config["action_mode"] not in ("right_only_6d", "both_arms_12d"):
        raise ValueError("unsupported RJ45 residual action mode")
    action_application_mode = config["action_application_mode"]
    if action_application_mode not in (
        POST_INTEGRAL_STEP_ACTION_APPLICATION,
        ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
    ):
        raise ValueError("unsupported RJ45 action application mode")
    if (
        action_application_mode == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
        and config["action_mode"] != "right_only_6d"
    ):
        raise ValueError(
            "absolute_tip_pose_target_insert_only requires right_only_6d"
        )
    action_dim = 6 if config["action_mode"] == "right_only_6d" else 12
    if config["base_observation_dim"] != 79:
        raise ValueError("RJ45 base residual policy observation must be 79D")
    observation_dim = policy_observation_dimension(config)
    levels = config["calibration_bias"]
    if type(bias_level) is not int or not 0 <= bias_level < len(levels["translation_levels_m"]):
        raise ValueError("bias_level must select a configured calibration level")
    validated_held_state = (
        None if held_state is None else validate_held_grasp_artifact(held_state)
    )

    class Rby1Rj45TrainingEnv(base_cls):
        """Residual policy environment with a strict estimator/evaluator split."""

        def __init__(self, cfg, render_mode: str | None = None, **kwargs):
            validate_reward_horizon_contract(
                config, is_finite_horizon=cfg.is_finite_horizon
            )
            self._training_config = config
            self._training_held_state = validated_held_state
            self._training_bias_enabled = bool(calibration_bias_enabled)
            self._training_bias_level = bias_level
            self._training_initialized = False
            if type(cfg.seed) is not int or cfg.seed < 0:
                raise ValueError("RJ45 residual training requires a non-negative integer cfg.seed")
            self._training_bias_base_seed = int(cfg.seed)
            cfg.action_space = action_dim
            cfg.observation_space = observation_dim
            if validated_held_state is not None:
                cfg.episode_length_s = float(config["active_horizon_s"])
            super().__init__(cfg, render_mode, **kwargs)
            self._initialize_training_state()
            self._training_initialized = True
            self._configure_training_reset(self.robot._ALL_INDICES)

        def _initialize_training_state(self) -> None:
            self._training_estimator_valid = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._training_policy_activated = torch.zeros_like(
                self._training_estimator_valid
            )
            self._training_held_initialization_passed = torch.zeros_like(
                self._training_estimator_valid
            )
            self._training_valid_policy_input_step_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._training_applied_tip_offset_entry = torch.zeros(
                (self.num_envs, 6), device=self.device
            )
            self._training_realign_pending = torch.zeros_like(
                self._training_estimator_valid
            )
            self._training_warmup_complete = torch.full_like(
                self._training_estimator_valid,
                self._training_held_state is None,
            )
            self._training_warmup_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._training_warmup_stable_steps = torch.zeros_like(
                self._training_warmup_steps
            )
            self._training_warmup_grip_force_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._training_warmup_insertion_load_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._training_warmup_forbidden_load_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._training_held_reset_stage_queued = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._training_held_reset_stage_restored = torch.full(
                (self.num_envs,),
                self._training_held_state is None,
                dtype=torch.bool,
                device=self.device,
            )
            self._training_held_reset_stage_waited_control_steps = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._training_held_reset_validation_passed = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._training_held_reset_validation_finger_force_n = torch.zeros(
                (self.num_envs, 4), device=self.device
            )
            self._training_held_reset_validation_raw_contact_count = torch.zeros(
                (self.num_envs, 5), dtype=torch.long, device=self.device
            )
            self._training_held_reset_validation_insertion_force_world_n = torch.zeros(
                (self.num_envs, 3), device=self.device
            )
            self._training_held_reset_validation_insertion_load_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._training_held_reset_validation_forbidden_load_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._training_frozen_object_to_ee_pos = torch.zeros(
                (self.num_envs, 2, 3), device=self.device
            )
            self._training_frozen_object_to_ee_quat = torch.zeros(
                (self.num_envs, 2, 4), device=self.device
            )
            self._training_frozen_object_to_ee_quat[:, :, 0] = 1.0
            self._training_estimated_object_pos = torch.zeros(
                (self.num_envs, 2, 3), device=self.device
            )
            self._training_estimated_object_quat = torch.zeros(
                (self.num_envs, 2, 4), device=self.device
            )
            self._training_estimated_object_quat[:, :, 0] = 1.0
            self._training_estimated_relative_pos = torch.zeros(
                (self.num_envs, 3), device=self.device
            )
            self._training_estimated_relative_quat = torch.zeros(
                (self.num_envs, 4), device=self.device
            )
            self._training_estimated_relative_quat[:, 0] = 1.0
            self._training_translation_bias_entry = torch.zeros(
                (self.num_envs, 3), device=self.device
            )
            self._training_rotation_bias_entry = torch.zeros(
                (self.num_envs, 3), device=self.device
            )
            self._training_episode_index = torch.full(
                (self.num_envs,), -1, dtype=torch.long, device=self.device
            )
            self._training_previous_action = torch.zeros(
                (self.num_envs, 12), device=self.device
            )
            self._training_last_policy_observation = torch.zeros(
                (self.num_envs, observation_dim), device=self.device
            )
            self._training_last_policy_observation_valid = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._training_last_policy_observation_timestamp_s = torch.full(
                (self.num_envs,), -1.0, device=self.device
            )
            self._training_action_delta = torch.zeros_like(
                self._training_previous_action
            )
            self._training_previous_potential = torch.zeros(
                self.num_envs, device=self.device
            )
            self._training_terminal_reward_emitted = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._training_reward_transition_active = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._training_reward_component_sums = {
                name: torch.zeros(self.num_envs, device=self.device)
                for name in (
                    "potential_progress",
                    "force_cost",
                    "action_cost",
                    "terminal",
                    "total",
                )
            }
            self._training_policy_active_step_count = torch.zeros(
                self.num_envs, dtype=torch.long, device=self.device
            )
            self._training_action_saturation_step_count = torch.zeros_like(
                self._training_policy_active_step_count
            )
            self._training_action_clip_boundary_step_count = torch.zeros_like(
                self._training_policy_active_step_count
            )
            self._training_action_norm_projection_step_count = torch.zeros_like(
                self._training_policy_active_step_count
            )
            self._training_action_clip_boundary_step = torch.zeros_like(
                self._training_reward_transition_active
            )
            self._training_action_norm_projection_step = torch.zeros_like(
                self._training_reward_transition_active
            )
            self._training_held_reference = torch.zeros(
                (self.num_envs, self.robot.num_joints), device=self.device
            )
            self._training_prismatic_joint_mask = torch.zeros(
                self.robot.num_joints, dtype=torch.bool, device=self.device
            )
            if self._training_held_state is not None:
                # A held episode starts beyond the table approach controller.
                # Reconstruct its path state separately from physical grip proof.
                if not hasattr(self, "_initial_clearance_reached"):
                    self._initial_clearance_reached = torch.zeros(
                        self.num_envs, dtype=torch.bool, device=self.device
                    )
                    self._work_clearance_reached = torch.zeros_like(
                        self._initial_clearance_reached
                    )
                    self._torso_progress = torch.zeros(self.num_envs, device=self.device)
                    self._approach_progress = torch.zeros((self.num_envs, 2), device=self.device)
                units_by_name = dict(
                    zip(
                        self._training_held_state["joint_names"],
                        self._training_held_state["joint_units"],
                        strict=True,
                    )
                )
                self._training_prismatic_joint_mask[:] = torch.tensor(
                    [units_by_name[name] == "m" for name in self.robot.joint_names],
                    dtype=torch.bool,
                    device=self.device,
                )

        def _sample_episode_bias(self, env_ids) -> None:
            self._training_translation_bias_entry[env_ids] = 0.0
            self._training_rotation_bias_entry[env_ids] = 0.0
            if not self._training_bias_enabled:
                return
            translation_max = float(
                self._training_config["calibration_bias"]["translation_levels_m"][
                    self._training_bias_level
                ]
            )
            rotation_max = float(
                self._training_config["calibration_bias"]["rotation_max_levels_rad"][
                    self._training_bias_level
                ]
            )
            for env_id in env_ids.detach().cpu().tolist():
                episode_index = int(self._training_episode_index[env_id].item())
                case_seed = _calibration_bias_case_seed(
                    self._training_bias_base_seed, env_id, episode_index
                )
                generator = torch.Generator(device=self.device)
                generator.manual_seed(case_seed)
                translation = torch.randn(
                    (1, 3), device=self.device, generator=generator
                )
                translation /= torch.linalg.vector_norm(
                    translation, dim=1, keepdim=True
                ).clamp_min(1.0e-9)
                translation *= torch.rand(
                    (1, 1), device=self.device, generator=generator
                ) * translation_max
                rotation = torch.randn(
                    (1, 3), device=self.device, generator=generator
                )
                rotation /= torch.linalg.vector_norm(
                    rotation, dim=1, keepdim=True
                ).clamp_min(1.0e-9)
                rotation *= torch.rand(
                    (1, 1), device=self.device, generator=generator
                ) * rotation_max
                self._training_translation_bias_entry[env_id] = translation[0]
                self._training_rotation_bias_entry[env_id] = rotation[0]

        def _object_poses_from_relative_mating(
            self,
            socket_pos,
            socket_quat,
            relative_pos,
            relative_quat,
        ):
            entry_pos, entry_quat = self._frame_tensors("socket_entry")
            tip_pos, tip_quat = self._frame_tensors("plug_tip")
            entry_world_pos, entry_world_quat = combine_frame_transforms(
                socket_pos, socket_quat, entry_pos, entry_quat
            )
            tip_world_pos, tip_world_quat = combine_frame_transforms(
                entry_world_pos,
                entry_world_quat,
                relative_pos,
                relative_quat,
            )
            inverse_tip_quat = tip_quat.clone()
            inverse_tip_quat[:, 1:] *= -1.0
            inverse_tip_pos = -quat_apply(inverse_tip_quat, tip_pos)
            return combine_frame_transforms(
                tip_world_pos,
                tip_world_quat,
                inverse_tip_pos,
                inverse_tip_quat,
            )

        def _freeze_training_estimator(self, mask) -> None:
            if not torch.any(mask):
                return
            actual_relative_pos, actual_relative_quat = self._relative_object_pose()
            biased_relative_pos, biased_relative_quat = apply_relative_pose_bias(
                actual_relative_pos,
                actual_relative_quat,
                self._training_translation_bias_entry,
                self._training_rotation_bias_entry,
            )
            biased_plug_pos, biased_plug_quat = self._object_poses_from_relative_mating(
                self.socket.data.root_pos_w,
                self.socket.data.root_quat_w,
                biased_relative_pos,
                biased_relative_quat,
            )
            left_ee_pos = self.robot.data.body_pos_w[:, self._left_ee_body_id]
            left_ee_quat = self.robot.data.body_quat_w[:, self._left_ee_body_id]
            right_ee_pos = self.robot.data.body_pos_w[:, self._right_ee_body_id]
            right_ee_quat = self.robot.data.body_quat_w[:, self._right_ee_body_id]
            left_pos, left_quat = subtract_frame_transforms(
                self.socket.data.root_pos_w,
                self.socket.data.root_quat_w,
                left_ee_pos,
                left_ee_quat,
            )
            right_pos, right_quat = subtract_frame_transforms(
                biased_plug_pos,
                biased_plug_quat,
                right_ee_pos,
                right_ee_quat,
            )
            self._training_frozen_object_to_ee_pos[mask] = torch.stack(
                (left_pos[mask], right_pos[mask]), dim=1
            )
            self._training_frozen_object_to_ee_quat[mask] = torch.stack(
                (left_quat[mask], right_quat[mask]), dim=1
            )
            self._training_estimator_valid[mask] = True
            self._refresh_training_estimator()

        def _refresh_training_estimator(self) -> None:
            if not torch.any(self._training_estimator_valid):
                return
            ee_pos = torch.stack(
                (
                    self.robot.data.body_pos_w[:, self._left_ee_body_id],
                    self.robot.data.body_pos_w[:, self._right_ee_body_id],
                ),
                dim=1,
            )
            ee_quat = torch.stack(
                (
                    self.robot.data.body_quat_w[:, self._left_ee_body_id],
                    self.robot.data.body_quat_w[:, self._right_ee_body_id],
                ),
                dim=1,
            )
            inverse_quat = self._training_frozen_object_to_ee_quat.clone()
            inverse_quat[:, :, 1:] *= -1.0
            inverse_pos = -quat_apply(
                inverse_quat.reshape(-1, 4),
                self._training_frozen_object_to_ee_pos.reshape(-1, 3),
            ).reshape(self.num_envs, 2, 3)
            object_pos, object_quat = combine_frame_transforms(
                ee_pos.reshape(-1, 3),
                ee_quat.reshape(-1, 4),
                inverse_pos.reshape(-1, 3),
                inverse_quat.reshape(-1, 4),
            )
            object_pos = object_pos.reshape(self.num_envs, 2, 3)
            object_quat = object_quat.reshape(self.num_envs, 2, 4)
            valid = self._training_estimator_valid
            self._training_estimated_object_pos[valid] = object_pos[valid]
            self._training_estimated_object_quat[valid] = object_quat[valid]
            entry_pos, entry_quat = self._frame_tensors("socket_entry")
            tip_pos, tip_quat = self._frame_tensors("plug_tip")
            entry_world = combine_frame_transforms(
                object_pos[:, 0], object_quat[:, 0], entry_pos, entry_quat
            )
            tip_world = combine_frame_transforms(
                object_pos[:, 1], object_quat[:, 1], tip_pos, tip_quat
            )
            relative_pos, relative_quat = subtract_frame_transforms(
                entry_world[0], entry_world[1], tip_world[0], tip_world[1]
            )
            self._training_estimated_relative_pos[valid] = relative_pos[valid]
            self._training_estimated_relative_quat[valid] = relative_quat[valid]

        def _control_relative_object_pose(self):
            if self._training_initialized and bool(self._training_estimator_valid.all()):
                return (
                    self._training_estimated_relative_pos,
                    self._training_estimated_relative_quat,
                )
            if self._training_initialized and torch.any(self._training_estimator_valid):
                actual_pos, actual_quat = super()._control_relative_object_pose()
                valid = self._training_estimator_valid[:, None]
                return (
                    torch.where(valid, self._training_estimated_relative_pos, actual_pos),
                    torch.where(valid, self._training_estimated_relative_quat, actual_quat),
                )
            return super()._control_relative_object_pose()

        def _control_object_poses(self):
            if self._training_initialized and bool(self._training_estimator_valid.all()):
                return (
                    self._training_estimated_object_pos[:, 0],
                    self._training_estimated_object_quat[:, 0],
                    self._training_estimated_object_pos[:, 1],
                    self._training_estimated_object_quat[:, 1],
                )
            if self._training_initialized and torch.any(self._training_estimator_valid):
                actual = super()._control_object_poses()
                valid = self._training_estimator_valid[:, None]
                return (
                    torch.where(valid, self._training_estimated_object_pos[:, 0], actual[0]),
                    torch.where(valid, self._training_estimated_object_quat[:, 0], actual[1]),
                    torch.where(valid, self._training_estimated_object_pos[:, 1], actual[2]),
                    torch.where(valid, self._training_estimated_object_quat[:, 1], actual[3]),
                )
            return super()._control_object_poses()

        def _control_grasp_transforms(self, *args):
            if self._training_initialized and bool(self._training_estimator_valid.all()):
                return (
                    self._training_frozen_object_to_ee_pos[:, 0],
                    self._training_frozen_object_to_ee_quat[:, 0],
                    self._training_frozen_object_to_ee_pos[:, 1],
                    self._training_frozen_object_to_ee_quat[:, 1],
                )
            if self._training_initialized and torch.any(self._training_estimator_valid):
                actual = super()._control_grasp_transforms(*args)
                valid = self._training_estimator_valid[:, None]
                return (
                    torch.where(valid, self._training_frozen_object_to_ee_pos[:, 0], actual[0]),
                    torch.where(valid, self._training_frozen_object_to_ee_quat[:, 0], actual[1]),
                    torch.where(valid, self._training_frozen_object_to_ee_pos[:, 1], actual[2]),
                    torch.where(valid, self._training_frozen_object_to_ee_quat[:, 1], actual[3]),
                )
            return super()._control_grasp_transforms(*args)

        def _grasp_target_poses(self):
            if not self._training_initialized:
                return super()._grasp_target_poses()
            socket_pos, socket_quat, plug_pos, plug_quat = self._control_object_poses()
            left_pos, left_quat = self._frame_tensors("left_grasp")
            right_pos, right_quat = self._frame_tensors("right_grasp")
            left_world = combine_frame_transforms(
                socket_pos, socket_quat, left_pos, left_quat
            )
            right_world = combine_frame_transforms(
                plug_pos, plug_quat, right_pos, right_quat
            )
            return (*left_world, *right_world)

        def _residual_actions_world(self, actions):
            if not self._training_initialized:
                return super()._residual_actions_world(actions)
            if (
                self._training_config["action_application_mode"]
                == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            ):
                return torch.zeros_like(actions)
            if not torch.any(self._training_estimator_valid):
                return super()._residual_actions_world(actions)
            bounded = actions.reshape(self.num_envs, 4, 3)
            entry_pos, entry_quat = self._frame_tensors("socket_entry")
            del entry_pos
            socket_pos, socket_quat, _, _ = self._control_object_poses()
            entry_world_quat = combine_frame_transforms(
                socket_pos,
                socket_quat,
                torch.zeros((self.num_envs, 3), device=self.device),
                entry_quat,
            )[1]
            return torch.stack(
                [quat_apply(entry_world_quat, bounded[:, index]) for index in range(4)],
                dim=1,
            ).reshape(self.num_envs, 12)

        def _apply_policy_mating_target_pose_offsets(
            self, plug_goal, plug_goal_quat
        ):
            if not self._training_initialized:
                return super()._apply_policy_mating_target_pose_offsets(
                    plug_goal, plug_goal_quat
                )
            self._training_applied_tip_offset_entry.zero_()
            if (
                self._training_config["action_application_mode"]
                != ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            ):
                return super()._apply_policy_mating_target_pose_offsets(
                    plug_goal, plug_goal_quat
                )
            active = self._policy_active_mask()
            if not torch.any(active):
                return plug_goal, plug_goal_quat
            offset = self._actions[:, 6:12].clone()
            offset[:, :3] *= self.cfg.residual_translation_bound_m
            offset[:, 3:] *= self.cfg.residual_rotation_bound_rad
            offset = torch.where(active[:, None], offset, torch.zeros_like(offset))
            socket_pos, socket_quat, _, _ = self._control_object_poses()
            entry_pos, entry_quat = self._frame_tensors("socket_entry")
            del entry_pos
            entry_world_quat = combine_frame_transforms(
                socket_pos,
                socket_quat,
                torch.zeros((self.num_envs, 3), device=self.device),
                entry_quat,
            )[1]
            tip_pos, tip_quat = self._frame_tensors("plug_tip")
            target_position, target_quaternion = apply_tip_target_pose_offset(
                plug_goal,
                plug_goal_quat,
                entry_world_quat,
                tip_pos,
                tip_quat,
                offset,
            )
            self._training_applied_tip_offset_entry[:] = offset
            return (
                torch.where(active[:, None], target_position, plug_goal),
                torch.where(active[:, None], target_quaternion, plug_goal_quat),
            )

        def _policy_active_mask(self):
            if (
                self._training_config["action_application_mode"]
                == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            ):
                return self._training_policy_activated & (
                    self._phase == int(AssemblyPhase.INSERT)
                )
            return self._training_policy_activated & (
                (self._phase == int(AssemblyPhase.ALIGN))
                | (self._phase == int(AssemblyPhase.INSERT))
            )

        def _evaluation_eligibility_mask(self):
            if not self._training_initialized:
                return super()._evaluation_eligibility_mask()
            eligible = self._training_policy_activated.clone()
            if (
                self._training_config["action_application_mode"]
                == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            ):
                eligible &= self._training_valid_policy_input_step_count > 0
            return eligible

        def _warmup_pending_mask(self):
            if not self._training_initialized or self._training_held_state is None:
                return torch.zeros(
                    self.num_envs, dtype=torch.bool, device=self.device
                )
            return ~self._training_warmup_complete

        def _pre_physics_step(self, actions) -> None:
            self._advance_held_reset_staging()
            pending = self._warmup_pending_mask()
            self._refresh_training_estimator()
            expanded = expand_residual_actions(
                actions, self._training_config["action_mode"]
            )
            active = self._policy_active_mask()[:, None]
            reward_transition_active = _executed_policy_transition_mask(
                self._training_config,
                active[:, 0],
                pending,
                self._training_last_policy_observation_valid,
                self._training_valid_policy_input_step_count,
            )
            self._training_reward_transition_active[:] = reward_transition_active
            clip_boundary, norm_projection = _action_saturation_masks(
                expanded, reward_transition_active
            )
            self._training_action_clip_boundary_step[:] = clip_boundary
            self._training_action_norm_projection_step[:] = norm_projection
            expanded = torch.where(active, expanded, torch.zeros_like(expanded))
            self._training_action_delta[:] = expanded - self._training_previous_action
            super()._pre_physics_step(expanded)
            self._training_previous_action[:] = expanded
            if torch.any(pending):
                self._actions[pending] = 0.0
                self._training_action_delta[pending] = 0.0
                self._training_previous_action[pending] = 0.0
                self._joint_targets[pending] = self._training_held_reference[pending]
                self._insert_joint_targets[pending] = self._training_held_reference[
                    pending
                ][:, self._arm_joint_ids]
                self._torso_hold_targets[pending] = self._training_held_reference[
                    pending
                ][:, self._torso_joint_ids]
                if hasattr(self, "_gripper_command"):
                    self._gripper_command[pending] = self._training_held_reference[
                        pending
                    ][:, self._gripper_joint_ids]
                if hasattr(self, "_pregrasp_servo_targets"):
                    self._pregrasp_servo_targets[pending] = torch.stack(
                        (
                            self._training_held_reference[pending][
                                :, self._left_arm_joint_ids
                            ],
                            self._training_held_reference[pending][
                                :, self._right_arm_joint_ids
                            ],
                        ),
                        dim=1,
                    )
                self._grasp_loss_since_s[pending] = -1.0
                self._retry_count[pending] = 0
                self._insertion_advance_enabled[pending] = False

        def _advance_held_reset_staging(self) -> None:
            if self._training_held_state is None:
                return
            ready, next_waited = _held_reset_stage_transition(
                self._training_held_reset_stage_queued,
                self._training_held_reset_stage_waited_control_steps,
                int(self._training_config["held_reset_staging"]["control_steps"]),
            )
            self._training_held_reset_stage_waited_control_steps[:] = next_waited
            if not torch.any(ready):
                return
            (
                finger_force_n,
                finger_raw_contact_count,
                insertion_force_world_n,
                insertion_load_n,
                insertion_raw_contact_count,
                forbidden_load_n,
            ) = self._held_reset_contact_flush_state()
            _, _, validation_passed = _held_reset_contact_free_status(
                finger_force_n,
                finger_raw_contact_count,
                insertion_force_world_n,
                insertion_load_n,
                insertion_raw_contact_count,
                forbidden_load_n,
            )
            ready_ids = torch.nonzero(ready, as_tuple=False).squeeze(1)
            self._training_held_reset_validation_passed[ready_ids] = (
                validation_passed[ready_ids]
            )
            self._training_held_reset_validation_finger_force_n[ready_ids] = (
                finger_force_n[ready_ids]
            )
            self._training_held_reset_validation_raw_contact_count[ready_ids, :4] = (
                finger_raw_contact_count[ready_ids].to(
                    dtype=self._training_held_reset_validation_raw_contact_count.dtype
                )
            )
            self._training_held_reset_validation_raw_contact_count[ready_ids, 4] = (
                insertion_raw_contact_count[ready_ids].to(
                    dtype=self._training_held_reset_validation_raw_contact_count.dtype
                )
            )
            self._training_held_reset_validation_insertion_force_world_n[
                ready_ids
            ] = insertion_force_world_n[ready_ids]
            self._training_held_reset_validation_insertion_load_n[ready_ids] = (
                insertion_load_n[ready_ids]
            )
            self._training_held_reset_validation_forbidden_load_n[ready_ids] = (
                forbidden_load_n[ready_ids]
            )
            restore = ready & validation_passed
            if torch.any(restore):
                restore_ids = torch.nonzero(restore, as_tuple=False).squeeze(1)
                self._restore_held_physical_state(restore_ids)
                self._training_held_reset_stage_queued[restore_ids] = False
                self._training_held_reset_stage_restored[restore_ids] = True
            failed = ready & ~validation_passed
            if torch.any(failed):
                raise HeldResetContactFlushError(
                    _held_reset_contact_flush_diagnostics(
                        failed,
                        finger_force_n,
                        finger_raw_contact_count,
                        insertion_force_world_n,
                        insertion_load_n,
                        insertion_raw_contact_count,
                        forbidden_load_n,
                    )
                )

        def _raw_target_contact_count(self, sensor, label: str):
            _, _, _, _, counts, _ = sensor.contact_physx_view.get_contact_data(
                dt=self.cfg.sim.dt
            )
            if counts.shape != (self.num_envs, 1):
                raise RuntimeError(
                    f"{label} raw contact counts must have shape "
                    f"({self.num_envs}, 1), got {tuple(counts.shape)}"
                )
            if counts.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise RuntimeError(f"{label} raw contact counts must be integral")
            return counts[:, 0]

        def _contact_load_from_sensors_without_state_mutation(
            self, sensors, label: str
        ):
            loads = []
            for source, sensor in sensors.items():
                forces, _, _, _, counts, starts = (
                    sensor.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
                )
                try:
                    load = sum_contact_normal_loads(
                        forces, counts, starts, self.num_envs
                    )
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"invalid {label} contact buffer for {source}: {error}"
                    ) from error
                loads.append(load)
            if not loads:
                raise RuntimeError(f"{label} contact sensors are unavailable")
            total = torch.stack(loads, dim=1).sum(dim=1)
            if not torch.isfinite(total).all():
                raise RuntimeError(f"{label} contact load is non-finite")
            return total

        def _held_reset_contact_flush_state(self):
            finger_sensors = (
                ("left_l1", self.left_contact_l1),
                ("left_l2", self.left_contact_l2),
                ("right_r1", self.right_contact_r1),
                ("right_r2", self.right_contact_r2),
            )
            finger_force_n = torch.stack(
                [self._filtered_force_magnitude(sensor) for _, sensor in finger_sensors],
                dim=1,
            )
            finger_raw_contact_count = torch.stack(
                [
                    self._raw_target_contact_count(sensor, label)
                    for label, sensor in finger_sensors
                ],
                dim=1,
            )
            insertion_force_world_n = self._filtered_force_vector(
                self.insertion_contact
            )
            forces, _, _, _, counts, starts = (
                self.insertion_contact.contact_physx_view.get_contact_data(
                    dt=self.cfg.sim.dt
                )
            )
            if counts.shape != (self.num_envs, 1):
                raise RuntimeError(
                    "insertion raw contact counts must have shape "
                    f"({self.num_envs}, 1), got {tuple(counts.shape)}"
                )
            if counts.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise RuntimeError("insertion raw contact counts must be integral")
            try:
                insertion_load_n = sum_contact_normal_loads(
                    forces, counts, starts, self.num_envs
                )
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"invalid insertion contact buffer: {error}"
                ) from error
            insertion_raw_contact_count = counts[:, 0]
            external_forbidden_load_n = (
                self._contact_load_from_sensors_without_state_mutation(
                    self._forbidden_load_sensors,
                    "external forbidden",
                )
            )
            cross_hand_sensors = getattr(
                self,
                "_cross_hand_contact_sensors",
                getattr(self, "_hand_contact_sensors", None),
            )
            if cross_hand_sensors is None:
                forbidden_load_n = external_forbidden_load_n
            else:
                cross_hand_load_n = (
                    self._contact_load_from_sensors_without_state_mutation(
                        cross_hand_sensors,
                        "cross-hand forbidden",
                    )
                )
                forbidden_load_n = torch.maximum(
                    external_forbidden_load_n, cross_hand_load_n
                )
            return (
                finger_force_n,
                finger_raw_contact_count,
                insertion_force_world_n,
                insertion_load_n,
                insertion_raw_contact_count,
                forbidden_load_n,
            )

        def _transition_where(self, mask, phase, now, reason=None):
            if self._training_initialized:
                mask = mask & ~self._warmup_pending_mask()
                if not torch.any(mask):
                    return
            if (
                self._training_initialized
                and self._training_held_state is None
                and phase == AssemblyPhase.INSERT
            ):
                first_activation = mask & ~self._training_policy_activated
                if torch.any(first_activation):
                    self._freeze_training_estimator(first_activation)
                    self._training_policy_activated[first_activation] = True
                    self._training_realign_pending[first_activation] = True
                    super()._transition_where(
                        first_activation, AssemblyPhase.ALIGN, now, reason
                    )
                    mask = mask & ~first_activation
                allow_insert = mask & self._training_policy_activated
                self._training_realign_pending[allow_insert] = False
                if torch.any(allow_insert):
                    super()._transition_where(allow_insert, phase, now, reason)
                return
            super()._transition_where(mask, phase, now, reason)

        def _start_recovery(self, mask, now):
            if self._training_initialized:
                mask = mask & ~self._warmup_pending_mask()
            if torch.any(mask):
                super()._start_recovery(mask, now)

        def _get_observations(self):
            if not self._training_initialized:
                return super()._get_observations()
            if self._training_initialized:
                self._refresh_training_estimator()
            relative_pos, relative_quat = self._control_relative_object_pose()
            socket_pos, socket_quat, _, _ = self._control_object_poses()
            entry_pos, entry_quat = self._frame_tensors("socket_entry")
            _, entry_world_quat = combine_frame_transforms(
                socket_pos, socket_quat, entry_pos, entry_quat
            )
            entry_force = quat_apply_inverse(
                entry_world_quat, self._last_insertion_contact_force_w
            )
            arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
            insert_error = torch.where(
                (self._phase == int(AssemblyPhase.INSERT))[:, None],
                self._insert_joint_targets - arm_pos,
                torch.zeros_like(arm_pos),
            )
            observation = build_policy_observation(
                estimated_relative_position_entry=relative_pos,
                estimated_relative_quaternion_entry_wxyz=relative_quat,
                arm_joint_position_rad=arm_pos,
                arm_joint_velocity_rad_s=self.robot.data.joint_vel[:, self._arm_joint_ids],
                grip_force_n=self._last_contact_proxy_n,
                entry_contact_force_n=entry_force,
                grip_proven=self._grip_proven.float(),
                phase_one_hot=torch.nn.functional.one_hot(
                    self._phase, num_classes=len(AssemblyPhase)
                ).float(),
                previous_expanded_action=self._training_previous_action,
                command_depth_m=self._insertion_command_depth_m[:, None],
                insertion_advance=self._insertion_advance_enabled.float()[:, None],
                insertion_joint_reference_error_rad=insert_error,
                policy_active=self._policy_active_mask().float()[:, None],
            )
            bias_level = self._training_bias_level
            observation = append_persistent_calibration_bias_diagnostic(
                observation,
                self._training_translation_bias_entry,
                self._training_rotation_bias_entry,
                translation_max_m=self._training_config["calibration_bias"][
                    "translation_levels_m"
                ][bias_level],
                rotation_max_rad=self._training_config["calibration_bias"][
                    "rotation_max_levels_rad"
                ][bias_level],
                mode=self._training_config["diagnostic_observation_mode"],
            )
            observation, observation_valid = _mask_inactive_policy_observation(
                observation,
                self._policy_active_mask(),
                observation_dim,
            )
            self.extras["observation_source"] = policy_observation_source(
                self._training_config
            )
            self.extras.update(
                policy_observation_ground_truth_flags(self._training_config)
            )
            self.extras["diagnostic_observation_mode"] = self._training_config[
                "diagnostic_observation_mode"
            ]
            self.extras["privileged_diagnostic_observation"] = (
                PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION
                if observation_dim == 85
                else None
            )
            self.extras["policy_observation_valid"] = observation_valid.clone()
            _store_policy_input_telemetry(
                observation,
                observation_valid,
                self._sim_time_s(),
                self._training_last_policy_observation,
                self._training_last_policy_observation_valid,
                self._training_last_policy_observation_timestamp_s,
            )
            return {"policy": observation}

        def _get_rewards(self):
            if not self._training_initialized:
                return super()._get_rewards()
            activated = self._training_policy_activated
            if (
                self._training_config["action_application_mode"]
                == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
            ):
                activated = activated & (
                    self._training_valid_policy_input_step_count > 0
                )
            gt_pos, gt_quat = self._relative_object_pose()
            potential = insertion_potential(
                gt_pos, gt_quat, self._training_config
            )
            potential = torch.where(
                activated, potential, self._training_previous_potential
            )
            success = (
                self._evaluation_success
                & activated
                & ~self._training_terminal_reward_emitted
            )
            failed_now = (
                (self._phase == int(AssemblyPhase.ABORT))
                | self._episode_time_limit_reached()
            ) & ~self._evaluation_success & activated
            failed = failed_now & ~self._training_terminal_reward_emitted
            true_terminated = self.reset_terminated & activated
            truncated = self.reset_time_outs & activated
            reward_active = _reward_policy_active_mask(
                self._training_config,
                self._policy_active_mask(),
                self._training_reward_transition_active,
            )
            components = residual_reward_components(
                self._training_previous_potential,
                potential,
                self._last_insertion_contact_force_w,
                self._training_action_delta,
                reward_active,
                success,
                failed,
                true_terminated,
                truncated,
                self._training_config,
            )
            if self._training_config.get("schema_version") == 1:
                components["total"] = residual_reward(
                    self._training_previous_potential,
                    potential,
                    self._last_insertion_contact_force_w,
                    self._training_action_delta,
                    self._policy_active_mask(),
                    success,
                    failed,
                    self._training_config,
                )
            active_now = reward_active
            for name in self._training_reward_component_sums:
                self._training_reward_component_sums[name] += components[name]
            self._training_policy_active_step_count += active_now.to(torch.long)
            self._training_action_clip_boundary_step_count += (
                self._training_action_clip_boundary_step & active_now
            ).to(torch.long)
            self._training_action_norm_projection_step_count += (
                self._training_action_norm_projection_step & active_now
            ).to(torch.long)
            self._training_action_saturation_step_count += (
                active_now
                & (
                    self._training_action_clip_boundary_step
                    | self._training_action_norm_projection_step
                )
            ).to(torch.long)
            self.extras["rj45_reward_profile"] = reward_profile_metadata(
                self._training_config
            )
            self.extras["rj45_reward_components"] = {
                name: value.clone()
                for name, value in components.items()
                if name not in {"true_terminated", "truncated"}
            }
            self.extras["rj45_episode_end"] = {
                "true_terminated": true_terminated.clone(),
                "truncated": truncated.clone(),
                "simultaneous": (true_terminated & truncated).clone(),
            }
            terminal_snapshot = self.extras.get("terminal_snapshot")
            if isinstance(terminal_snapshot, dict):
                terminal_snapshot["reward_profile"] = reward_profile_metadata(
                    self._training_config
                )
                terminal_snapshot["reward_components"] = {
                    name: value.clone()
                    for name, value in components.items()
                    if name not in {"true_terminated", "truncated"}
                }
                terminal_snapshot["reward_component_episode_sums"] = {
                    name: value.clone()
                    for name, value in self._training_reward_component_sums.items()
                }
                terminal_snapshot["policy_active_step_count"] = (
                    self._training_policy_active_step_count.clone()
                )
                terminal_snapshot["action_saturation_step_count"] = (
                    self._training_action_saturation_step_count.clone()
                )
                terminal_snapshot["action_clip_boundary_step_count"] = (
                    self._training_action_clip_boundary_step_count.clone()
                )
                terminal_snapshot["action_norm_projection_step_count"] = (
                    self._training_action_norm_projection_step_count.clone()
                )
                terminal_snapshot["action_saturation_definition"] = (
                    "active step where the wrapper-clipped action reached a per-axis "
                    "[-1,1] boundary or any 3D action vector needed unit-norm projection; "
                    "pre-wrapper raw clipping magnitude is unavailable"
                )
                terminal_snapshot["reward_true_terminated"] = true_terminated.clone()
                terminal_snapshot["reward_truncated"] = truncated.clone()
                terminal_snapshot["reward_terminated_and_truncated"] = (
                    true_terminated & truncated
                ).clone()
            self._training_previous_potential[:] = potential
            self._training_terminal_reward_emitted |= success | failed
            return components["total"]

        def _held_warmup_check(self, pending) -> None:
            pending = pending & self._training_held_reset_stage_restored
            if not torch.any(pending):
                return
            left_l1 = self._filtered_force_magnitude(self.left_contact_l1)
            left_l2 = self._filtered_force_magnitude(self.left_contact_l2)
            right_r1 = self._filtered_force_magnitude(self.right_contact_r1)
            right_r2 = self._filtered_force_magnitude(self.right_contact_r2)
            self._last_contact_proxy_n[:, 0] = torch.maximum(left_l1, left_l2)
            self._last_contact_proxy_n[:, 1] = torch.maximum(right_r1, right_r2)
            insertion_force_w = self._filtered_force_vector(self.insertion_contact)
            self._last_insertion_contact_force_w[:] = insertion_force_w
            self._last_insertion_contact_force_n[:] = torch.linalg.vector_norm(
                insertion_force_w, dim=1
            )
            self._last_insertion_contact_load_n[:] = self._insertion_contact_load()
            self._last_forbidden_contact_load_n[:] = self._forbidden_contact_load()
            grip_force = self._last_contact_proxy_n.amax(dim=1)
            finger_force_n = torch.stack((left_l1, left_l2, right_r1, right_r2), dim=1)
            sensor_finite, within_limits, grip_live = _warmup_sensor_status(
                finger_force_n,
                insertion_force_w,
                self._last_insertion_contact_load_n,
                self._last_forbidden_contact_load_n,
                grip_contact_force_n=self.cfg.grip_contact_force_n,
                hard_contact_force_n=self.cfg.hard_contact_force_n,
                forbidden_contact_force_n=self.cfg.forbidden_contact_force_n,
            )
            nonfinite = pending & ~sensor_finite
            if torch.any(nonfinite):
                ids = torch.nonzero(nonfinite, as_tuple=False).squeeze(1).tolist()
                raise RuntimeError(
                    f"held-grasp warmup received non-finite contact data in environments {ids}"
                )
            self._training_warmup_grip_force_peak_n[pending] = torch.maximum(
                self._training_warmup_grip_force_peak_n[pending], grip_force[pending]
            )
            self._training_warmup_insertion_load_peak_n[pending] = torch.maximum(
                self._training_warmup_insertion_load_peak_n[pending],
                self._last_insertion_contact_load_n[pending],
            )
            self._training_warmup_forbidden_load_peak_n[pending] = torch.maximum(
                self._training_warmup_forbidden_load_peak_n[pending],
                self._last_forbidden_contact_load_n[pending],
            )
            unsafe = pending & ~within_limits
            if torch.any(unsafe):
                ids = torch.nonzero(unsafe, as_tuple=False).squeeze(1).tolist()
                raise RuntimeError(
                    f"held-grasp warmup violated force or forbidden-contact limits in environments {ids}"
                )
            lift_left = self.socket.data.root_pos_w[:, 2] - (
                self.scene.env_origins[:, 2] + self._object_start_z[:, 0]
            )
            lift_right = self.plug.data.root_pos_w[:, 2] - (
                self.scene.env_origins[:, 2] + self._object_start_z[:, 1]
            )
            linear, angular = self._relative_mating_velocity()
            motion_finite, motion_settled = _warmup_motion_status(
                self.robot.data.joint_vel,
                self.robot.data.joint_pos - self._training_held_reference,
                self._training_prismatic_joint_mask,
                self.socket.data.root_com_vel_w,
                self.plug.data.root_com_vel_w,
                linear,
                angular,
                limits=HELD_GRASP_STABILITY_LIMITS,
                max_relative_linear_speed_m_s=(
                    self.cfg.max_relative_linear_speed_m_s
                ),
                max_relative_angular_speed_rad_s=(
                    self.cfg.max_relative_angular_speed_rad_s
                ),
            )
            state_finite = (
                torch.isfinite(self.robot.data.joint_pos).all(dim=1)
                & torch.isfinite(self.socket.data.root_state_w).all(dim=1)
                & torch.isfinite(self.plug.data.root_state_w).all(dim=1)
            )
            stable = pending & (
                state_finite
                & motion_finite
                & motion_settled
                & sensor_finite
                & within_limits
                & grip_live
                & (lift_left >= self.cfg.grip_lift_proof_m)
                & (lift_right >= self.cfg.grip_lift_proof_m)
            )
            self._training_warmup_steps[pending] += 1
            self._training_warmup_stable_steps[pending] = torch.where(
                stable[pending],
                self._training_warmup_stable_steps[pending] + 1,
                torch.zeros_like(self._training_warmup_stable_steps[pending]),
            )
            required = max(
                1,
                math.ceil(float(self._training_config["warmup_s"]) / self.step_dt),
            )
            passed = (
                pending
                & (self._training_warmup_stable_steps >= required)
            )
            if torch.any(passed):
                self._evaluation_success[passed] = False
                self._evaluation_dwell_s[passed] = 0.0
                self._success_since_step[passed] = -1
                self._forbidden_contact_seen[passed] = False
                self._episode_grip_force_peak_n[passed] = 0.0
                self._episode_insertion_contact_force_peak_n[passed] = 0.0
                self._episode_insertion_contact_load_peak_n[passed] = 0.0
                self._episode_force_peak_n[passed] = 0.0
                self._episode_forbidden_contact_load_peak_n[passed] = 0.0
                self._grip_proven[passed] = True
                self._evaluation_grip_proven[passed] = True
                self._freeze_training_estimator(passed)
                self._training_policy_activated[passed] = True
                self._training_warmup_complete[passed] = True
                self._training_held_initialization_passed[passed] = True
                self._phase[passed] = int(AssemblyPhase.ALIGN)
                self._phase_started_s[passed] = self._sim_time_s()
                self.episode_length_buf[passed] = 0
                self._termination_reason_code[passed] = int(TerminationReason.NONE)
                for name in (
                    "_episode_cross_hand_contact_load_peak_n",
                    "_episode_external_forbidden_contact_load_peak_n",
                ):
                    if hasattr(self, name):
                        getattr(self, name)[passed] = 0.0
                gt_pos, gt_quat = self._relative_object_pose()
                self._training_previous_potential[passed] = insertion_potential(
                    gt_pos, gt_quat, self._training_config
                )[passed]
            timeout_steps = max(1, math.ceil(1.0 / self.step_dt))
            timed_out = _warmup_timeout_mask(
                pending,
                passed,
                self._training_warmup_steps,
                timeout_steps,
            )
            if torch.any(timed_out):
                diagnostics = _held_warmup_timeout_diagnostics(
                    timed_out=timed_out,
                    finger_force_n=finger_force_n,
                    lift_left_m=lift_left,
                    lift_right_m=lift_right,
                    joint_velocity_si=self.robot.data.joint_vel,
                    joint_reference_error_si=(
                        self.robot.data.joint_pos - self._training_held_reference
                    ),
                    prismatic_joint_mask=self._training_prismatic_joint_mask,
                    socket_com_velocity_world=self.socket.data.root_com_vel_w,
                    plug_com_velocity_world=self.plug.data.root_com_vel_w,
                    relative_linear_velocity_m_s=linear,
                    relative_angular_velocity_rad_s=angular,
                    insertion_force_world_n=insertion_force_w,
                    insertion_load_n=self._last_insertion_contact_load_n,
                    forbidden_load_n=self._last_forbidden_contact_load_n,
                    state_finite=state_finite,
                    motion_finite=motion_finite,
                    motion_settled=motion_settled,
                    sensor_finite=sensor_finite,
                    within_limits=within_limits,
                    grip_live=grip_live,
                    stable=stable,
                    warmup_steps=self._training_warmup_steps,
                    warmup_stable_steps=self._training_warmup_stable_steps,
                    required_stable_steps=required,
                    timeout_steps=timeout_steps,
                    step_dt_s=self.step_dt,
                    grip_lift_proof_m=self.cfg.grip_lift_proof_m,
                    limits={
                        **HELD_GRASP_STABILITY_LIMITS,
                        "max_relative_linear_speed_m_s": (
                            self.cfg.max_relative_linear_speed_m_s
                        ),
                        "max_relative_angular_speed_rad_s": (
                            self.cfg.max_relative_angular_speed_rad_s
                        ),
                    },
                )
                raise HeldGraspWarmupTimeoutError(diagnostics)

        def _count_policy_input_step(self):
            self._training_valid_policy_input_step_count.add_(
                self._training_last_policy_observation_valid.to(dtype=torch.long)
            )

        def _reset_policy_attribution(self, env_ids):
            self._training_held_initialization_passed[env_ids] = False
            self._training_valid_policy_input_step_count[env_ids] = 0

        def _training_step_snapshot(self):
            """Return the policy input used at this step's start.

            DirectRLEnv calls _get_dones before the next _get_observations
            call, so these cached tensors are the exact input previously
            returned to the policy, not a recomputed state.
            """

            return {
                "held_initialization_passed": (
                    self._training_held_initialization_passed.clone()
                ),
                "valid_policy_input_step_count": (
                    self._training_valid_policy_input_step_count.clone()
                ),
                "policy_input_observation": (
                    self._training_last_policy_observation.clone()
                ),
                "policy_input_timestamp_s": (
                    self._training_last_policy_observation_timestamp_s.clone()
                ),
                "policy_input_valid": (
                    self._training_last_policy_observation_valid.clone()
                ),
                "training_estimator_valid": self._training_estimator_valid.clone(),
                "training_policy_activated": (
                    self._training_policy_activated.clone()
                ),
                "requested_expanded_residual_action": (
                    self._training_previous_action.clone()
                ),
                "applied_bounded_residual_action": self._actions.clone(),
                "applied_tip_target_offset_entry_6d": (
                    self._training_applied_tip_offset_entry.clone()
                ),
            }

        def _get_dones(self):
            if not self._training_initialized:
                return super()._get_dones()
            self._count_policy_input_step()
            pending = self._warmup_pending_mask()
            terminated, truncated = super()._get_dones()
            if torch.any(pending):
                terminated[pending] = False
                truncated[pending] = False
                self._held_warmup_check(pending)
            training_episode = {
                "episode_index": self._training_episode_index.clone(),
                "calibration_bias_translation_entry_m": (
                    self._training_translation_bias_entry.clone()
                ),
                "calibration_bias_rotation_vector_entry_rad": (
                    self._training_rotation_bias_entry.clone()
                ),
                "applied_tip_target_offset_entry_6d": (
                    self._training_applied_tip_offset_entry.clone()
                ),
            }
            self.extras["rj45_training_episode"] = training_episode
            terminal_snapshot = self.extras.get("terminal_snapshot")
            if isinstance(terminal_snapshot, dict):
                terminal_snapshot.update(training_episode)
                terminal_snapshot.update(self._training_step_snapshot())
            return terminated, truncated

        def _held_artifact_joint_state(self, env_ids):
            artifact = self._training_held_state
            names = artifact["joint_names"]
            if set(names) != set(self.robot.joint_names):
                raise RuntimeError("held-grasp joint names do not match the loaded robot")
            order = [names.index(name) for name in self.robot.joint_names]
            state = artifact["state"]
            joint_pos = torch.tensor(
                state["joint_positions_si"], device=self.device
            )[order].repeat(len(env_ids), 1)
            joint_vel = torch.tensor(
                state["joint_velocities_si"], device=self.device
            )[order].repeat(len(env_ids), 1)
            joint_ref = torch.tensor(
                state["joint_reference_si"], device=self.device
            )[order].repeat(len(env_ids), 1)
            return joint_pos, joint_vel, joint_ref

        def _held_artifact_object_state(self, env_ids):
            state = self._training_held_state["state"]
            object_poses = torch.tensor(
                state["object_poses_world_wxyz"], device=self.device
            ).repeat(len(env_ids), 1, 1)
            object_poses[:, :, :3] += self.scene.env_origins[env_ids, None, :]
            object_velocities = torch.tensor(
                state["object_com_velocities_world"], device=self.device
            ).repeat(len(env_ids), 1, 1)
            return object_poses, object_velocities

        def _assert_held_state_write_allowed(self, env_ids) -> None:
            if not self._training_initialized:
                return
            activated = self._training_policy_activated[env_ids]
            if torch.any(activated):
                rejected = (
                    env_ids[torch.nonzero(activated, as_tuple=False).squeeze(1)]
                    .detach()
                    .cpu()
                    .tolist()
                )
                raise AssertionError(
                    "held reset state writes require inactive policies; "
                    f"activated environments={rejected}"
                )

        def _restore_held_physical_state(self, env_ids):
            self._assert_held_state_write_allowed(env_ids)
            joint_pos, joint_vel, _ = self._held_artifact_joint_state(env_ids)
            robot_root = self.robot.data.default_root_state[env_ids].clone()
            robot_root[:, :3] += self.scene.env_origins[env_ids]
            self.robot.write_root_pose_to_sim(robot_root[:, :7], env_ids)
            self.robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids)
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
            object_poses, object_velocities = self._held_artifact_object_state(env_ids)
            for index, obj in enumerate((self.socket, self.plug)):
                obj.write_root_pose_to_sim(object_poses[:, index], env_ids)
                obj.write_root_velocity_to_sim(object_velocities[:, index], env_ids)
            return joint_pos, joint_vel

        def _queue_held_reset_stage(self, env_ids) -> None:
            self._assert_held_state_write_allowed(env_ids)
            local_positions = torch.tensor(
                self._training_config["held_reset_staging"][
                    "object_positions_local_m"
                ],
                device=self.device,
            )
            if local_positions.shape != (2, 3):
                raise ValueError(
                    "held reset staging object_positions_local_m must have shape (2, 3)"
                )
            object_poses, _ = self._held_artifact_object_state(env_ids)
            for index, obj in enumerate((self.socket, self.plug)):
                pose = object_poses[:, index].clone()
                pose[:, :3] = (
                    self.scene.env_origins[env_ids] + local_positions[index]
                )
                obj.write_root_pose_to_sim(pose, env_ids)
            self._training_held_reset_stage_queued[env_ids] = True
            self._training_held_reset_stage_restored[env_ids] = False
            self._training_held_reset_stage_waited_control_steps[env_ids] = 0

        def _reset_physical_state(self, env_ids):
            if self._training_held_state is None:
                return super()._reset_physical_state(env_ids)
            joint_pos, joint_vel = self._restore_held_physical_state(env_ids)
            if self._training_initialized:
                self._queue_held_reset_stage(env_ids)
            return joint_pos, joint_vel

        def _reset_idx(self, env_ids=None):
            if self._training_initialized:
                if env_ids is None:
                    env_ids = self.robot._ALL_INDICES
                if not isinstance(env_ids, torch.Tensor):
                    env_ids = torch.as_tensor(
                        env_ids, dtype=torch.long, device=self.device
                    )
                self._training_policy_activated[env_ids] = False
            super()._reset_idx(env_ids)
            if not self._training_initialized:
                return
            self._configure_training_reset(env_ids)

        def _configure_training_reset(self, env_ids):
            self._reset_policy_attribution(env_ids)
            self._training_estimator_valid[env_ids] = False
            self._training_policy_activated[env_ids] = False
            self._training_realign_pending[env_ids] = False
            self._training_warmup_complete[env_ids] = self._training_held_state is None
            self._training_warmup_steps[env_ids] = 0
            self._training_warmup_stable_steps[env_ids] = 0
            self._training_warmup_grip_force_peak_n[env_ids] = 0.0
            self._training_warmup_insertion_load_peak_n[env_ids] = 0.0
            self._training_warmup_forbidden_load_peak_n[env_ids] = 0.0
            self._training_held_reset_validation_passed[env_ids] = False
            self._training_held_reset_validation_finger_force_n[env_ids] = 0.0
            self._training_held_reset_validation_raw_contact_count[env_ids] = 0
            self._training_held_reset_validation_insertion_force_world_n[
                env_ids
            ] = 0.0
            self._training_held_reset_validation_insertion_load_n[env_ids] = 0.0
            self._training_held_reset_validation_forbidden_load_n[env_ids] = 0.0
            self._training_previous_action[env_ids] = 0.0
            _reset_applied_tip_offset_telemetry(
                env_ids, self._training_applied_tip_offset_entry
            )
            _reset_policy_input_telemetry(
                env_ids,
                self._training_last_policy_observation,
                self._training_last_policy_observation_valid,
                self._training_last_policy_observation_timestamp_s,
            )
            self._training_action_delta[env_ids] = 0.0
            self._training_terminal_reward_emitted[env_ids] = False
            self._training_reward_transition_active[env_ids] = False
            self._training_action_clip_boundary_step[env_ids] = False
            self._training_action_norm_projection_step[env_ids] = False
            _reset_reward_telemetry(
                env_ids,
                self._training_reward_component_sums,
                self._training_policy_active_step_count,
                self._training_action_saturation_step_count,
            )
            self._training_action_clip_boundary_step_count[env_ids] = 0
            self._training_action_norm_projection_step_count[env_ids] = 0
            self._training_episode_index[env_ids] += 1
            self._sample_episode_bias(env_ids)
            if self._training_held_state is None:
                gt_pos, gt_quat = self._relative_object_pose()
                self._training_previous_potential[env_ids] = insertion_potential(
                    gt_pos, gt_quat, self._training_config
                )[env_ids]
                return

            _, _, joint_ref = self._held_artifact_joint_state(env_ids)
            self._training_held_reference[env_ids] = joint_ref
            self._initialize_held_controller_phase(env_ids)
            self._joint_targets[env_ids] = joint_ref
            self._insert_joint_targets[env_ids] = joint_ref[:, self._arm_joint_ids]
            self._torso_hold_targets[env_ids] = joint_ref[:, self._torso_joint_ids]
            if hasattr(self, "_pregrasp_servo_targets"):
                self._pregrasp_servo_targets[env_ids] = torch.stack(
                    (
                        joint_ref[:, self._left_arm_joint_ids],
                        joint_ref[:, self._right_arm_joint_ids],
                    ),
                    dim=1,
                )
            if hasattr(self, "_gripper_command"):
                self._gripper_command[env_ids] = joint_ref[:, self._gripper_joint_ids]
            self._phase[env_ids] = int(AssemblyPhase.ALIGN)
            self._phase_started_s[env_ids] = self._sim_time_s()
            self._grip_proven[env_ids] = False
            self._evaluation_grip_proven[env_ids] = False
            self._object_start_z[env_ids, 0] = self.socket.data.default_root_state[
                env_ids, 2
            ]
            self._object_start_z[env_ids, 1] = self.plug.data.default_root_state[
                env_ids, 2
            ]
            gt_pos, gt_quat = self._relative_object_pose()
            self._training_previous_potential[env_ids] = insertion_potential(
                gt_pos, gt_quat, self._training_config
            )[env_ids]

        def _initialize_held_controller_phase(self, env_ids):
            """Start a new ALIGN episode beyond the verified INSERT setup path."""

            if self._training_held_state["selection"]["phase_name"] != "insert":
                raise ValueError("held controller initialization requires a verified INSERT state")
            expected_torso = torch.tensor(
                self.cfg.rj45_task["controller"]["torso_work_joint_positions_rad"],
                dtype=self._training_held_reference.dtype,
                device=self.device,
            ).expand(len(env_ids), -1)
            if not torch.equal(
                self._training_held_reference[env_ids][:, self._torso_joint_ids],
                expected_torso,
            ):
                raise ValueError("recorded held torso reference differs from the configured work pose")
            self._initial_clearance_reached[env_ids] = True
            self._work_clearance_reached[env_ids] = True
            self._torso_progress[env_ids] = 1.0
            self._approach_progress[env_ids] = 1.0

        def rj45_training_report(self):
            return {
                "policy_observation_source": policy_observation_source(
                    self._training_config
                ),
                "ground_truth_usage": policy_ground_truth_usage(
                    self._training_config
                ),
                **policy_observation_ground_truth_flags(self._training_config),
                "action_mode": self._training_config["action_mode"],
                "reward_profile": reward_profile_metadata(self._training_config),
                "action_application_mode": self._training_config[
                    "action_application_mode"
                ],
                "action_application_description": (
                    TIP_TARGET_POSE_OFFSET_DESCRIPTION
                    if self._training_config["action_application_mode"]
                    == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
                    else "post-integral Cartesian twist residual"
                ),
                "policy_active_phases": (
                    ["insert"]
                    if self._training_config["action_application_mode"]
                    == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
                    else ["align", "insert"]
                ),
                "action_application_telemetry": {
                    "field": "applied_tip_target_offset_entry_6d",
                    "frame": "socket_entry",
                    "component_order": ["tx", "ty", "tz", "rx", "ry", "rz"],
                    "units": ["m", "m", "m", "rad", "rad", "rad"],
                    "inactive_rows": "all_zeros",
                },
                "policy_action_dimension": action_dim,
                "policy_observation_dimension": observation_dim,
                "diagnostic_observation_mode": self._training_config[
                    "diagnostic_observation_mode"
                ],
                "privileged_diagnostic_observation": (
                    PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION
                    if observation_dim == 85
                    else None
                ),
                "policy_observation_contract": {
                    "inactive_rows": "all_zeros",
                    "validity_extra": "policy_observation_valid",
                    "policy_active_field_index": 78,
                    "policy_active_field_value_when_valid": 1.0,
                    "persistent_calibration_bias_field_indices": (
                        [79, 85] if observation_dim == 85 else None
                    ),
                },
                "held_grasp_reset": self._training_held_state is not None,
                "held_controller_initialization": (
                    "reconstruct_completed_setup_path_for_new_alignment_episode"
                    if self._training_held_state is not None else "normal_table_setup"
                ),
                "held_reset_staging": {
                    "enabled": self._training_held_state is not None,
                    "scope": "episode_initialization_before_policy_activation_only",
                    "method": (
                        "one_noncontact_control_step_then_exact_recorded_state_restore"
                    ),
                    "state_write_gate": "policy_inactive_only",
                    "control_steps": int(
                        self._training_config["held_reset_staging"]["control_steps"]
                    ),
                    "object_positions_local_m": self._training_config[
                        "held_reset_staging"
                    ]["object_positions_local_m"],
                    "provenance": self._training_config["held_reset_staging"][
                        "provenance"
                    ],
                    "contact_free_validation": {
                        "requirement": "finite_and_exactly_zero_before_restore",
                        "passed": (
                            self._training_held_reset_validation_passed.detach()
                            .cpu()
                            .tolist()
                        ),
                        "finger_force_n": (
                            self._training_held_reset_validation_finger_force_n.detach()
                            .cpu()
                            .tolist()
                        ),
                        "raw_contact_count": (
                            self._training_held_reset_validation_raw_contact_count.detach()
                            .cpu()
                            .tolist()
                        ),
                        "raw_contact_count_order": [
                            "left_l1",
                            "left_l2",
                            "right_r1",
                            "right_r2",
                            "insertion",
                        ],
                        "insertion_force_world_n": (
                            self._training_held_reset_validation_insertion_force_world_n.detach()
                            .cpu()
                            .tolist()
                        ),
                        "insertion_load_n": (
                            self._training_held_reset_validation_insertion_load_n.detach()
                            .cpu()
                            .tolist()
                        ),
                        "forbidden_load_n": (
                            self._training_held_reset_validation_forbidden_load_n.detach()
                            .cpu()
                            .tolist()
                        ),
                    },
                },
                "learned_scope": "pre_portal_alignment_and_insertion",
                "learned_recovery": False,
                "camera_observation_claim": False,
                "mass_randomization": False,
                "held_warmup_peaks_n": {
                    "grip": self._training_warmup_grip_force_peak_n.detach()
                    .cpu()
                    .tolist(),
                    "insertion": self._training_warmup_insertion_load_peak_n.detach()
                    .cpu()
                    .tolist(),
                    "forbidden": self._training_warmup_forbidden_load_peak_n.detach()
                    .cpu()
                    .tolist(),
                },
                "calibration_bias": {
                    "enabled": self._training_bias_enabled,
                    "level": self._training_bias_level,
                    "base_seed": self._training_bias_base_seed,
                    "stream_offset": _CALIBRATION_BIAS_STREAM_OFFSET,
                    "episode_stride": _CALIBRATION_BIAS_EPISODE_STRIDE,
                    "case_seed_formula": (
                        "base_seed + stream_offset + episode_index * episode_stride + env_id"
                    ),
                    "sampling_event": "once_per_environment_reset_before_calibration",
                    "episode_index": self._training_episode_index.detach().cpu().tolist(),
                    "translation_entry_m": self._training_translation_bias_entry.detach()
                    .cpu()
                    .tolist(),
                    "rotation_vector_entry_rad": self._training_rotation_bias_entry.detach()
                    .cpu()
                    .tolist(),
                },
            }

    Rby1Rj45TrainingEnv.__name__ = "Rby1Rj45TrainingEnv"
    return Rby1Rj45TrainingEnv


__all__ = ["make_rj45_training_env_type"]
