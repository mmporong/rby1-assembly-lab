"""Pure experiment contracts and lightweight PPO action telemetry helpers."""

from __future__ import annotations

import math
from typing import Any, Callable


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return result


def observation_noise_contract(
    *,
    task: str,
    configured_position_std_m: Any,
    configured_rotation_vector_std_rad: Any,
    calibration_bias: dict[str, Any] | None = None,
    bias_level: int | None = None,
    calibration_bias_enabled: bool = True,
) -> dict[str, Any]:
    """Describe configured base noise separately from noise reaching the policy."""

    position_std = _finite_nonnegative(
        configured_position_std_m, "configured_position_std_m"
    )
    rotation_std = _finite_nonnegative(
        configured_rotation_vector_std_rad,
        "configured_rotation_vector_std_rad",
    )
    configured = {
        "distribution": "independent_zero_mean_gaussian_position_and_rotation_vector",
        "position_std_m": position_std,
        "rotation_vector_std_rad": rotation_std,
    }
    if task == "toy":
        if calibration_bias is not None or bias_level is not None:
            raise ValueError("toy observation noise does not accept calibration bias")
        return {
            "schema_version": 1,
            "configured_base_gaussian": configured,
            "configured_base_gaussian_applied_to_policy": True,
            "actual_policy_observation_noise": {
                **configured,
                "temporal_scope": "resampled_each_observation",
            },
        }
    if task != "rj45":
        raise ValueError("task must be toy or rj45")
    if not isinstance(calibration_bias, dict):
        raise ValueError("RJ45 observation noise requires calibration_bias")
    if type(bias_level) is not int:
        raise ValueError("RJ45 observation noise requires an integer bias_level")
    translations = calibration_bias.get("translation_levels_m")
    rotations = calibration_bias.get("rotation_max_levels_rad")
    if (
        not isinstance(translations, list)
        or not isinstance(rotations, list)
        or len(translations) != len(rotations)
        or not 0 <= bias_level < len(translations)
    ):
        raise ValueError("RJ45 calibration bias levels are invalid")
    translation_max = _finite_nonnegative(
        translations[bias_level], "calibration_bias.translation_levels_m[bias_level]"
    )
    rotation_max = _finite_nonnegative(
        rotations[bias_level],
        "calibration_bias.rotation_max_levels_rad[bias_level]",
    )
    if type(calibration_bias_enabled) is not bool:
        raise ValueError("calibration_bias_enabled must be boolean")
    return {
        "schema_version": 1,
        "configured_base_gaussian": configured,
        "configured_base_gaussian_applied_to_policy": False,
        "actual_policy_observation_noise": {
            "enabled": calibration_bias_enabled,
            "distribution": (
                "episode_fixed_isotropic_direction_uniform_radius_se3_bias"
                if calibration_bias_enabled
                else "disabled"
            ),
            "temporal_scope": "fixed_for_episode",
            "configured_bias_level": bias_level,
            "configured_translation_max_m": translation_max,
            "configured_rotation_vector_max_rad": rotation_max,
            "applied_translation_max_m": (
                translation_max if calibration_bias_enabled else 0.0
            ),
            "applied_rotation_vector_max_rad": (
                rotation_max if calibration_bias_enabled else 0.0
            ),
        },
    }


def _selected_tensor_statistics(values: Any, selected: Any) -> dict[str, Any]:
    import torch

    flat = values.detach().reshape(-1).to(dtype=torch.float64)
    selection = selected.reshape(-1)
    selected_count = selection.sum().to(dtype=torch.float64)
    safe_count = selected_count.clamp_min(1.0)
    selected_values = torch.where(selection, flat, torch.zeros_like(flat))
    abs_values = torch.cat((selected_values.abs(), selected_values.new_zeros(1)))
    abs_max = abs_values.max()
    scale = abs_max.clamp_min(torch.finfo(torch.float64).tiny)
    scaled = selected_values / scale
    scaled_mean = scaled.sum() / safe_count
    scaled_centered = torch.where(
        selection, scaled - scaled_mean, torch.zeros_like(scaled)
    )
    mean = scaled_mean * abs_max
    std = (scaled_centered.square().sum() / safe_count).sqrt() * abs_max
    return {
        "abs_max": abs_max,
        "mean": mean,
        "std": std,
    }


def action_clip_telemetry(
    actions: Any,
    *,
    clip_actions: float | None,
    policy_active: Any | None,
) -> dict[str, Any]:
    """Summarize raw and wrapper-clipped actions without retaining trajectories."""

    import torch

    if actions.ndim < 2:
        raise ValueError("actions must have an environment axis and action axis")
    if not bool(torch.isfinite(actions).all()):
        raise ValueError(
            "PPO raw action contains NaN or Infinity before actual "
            "RslRlVecEnvWrapper clipping"
        )
    if clip_actions is None:
        clipped = actions
        clipping_enabled = False
    else:
        limit = _finite_nonnegative(clip_actions, "clip_actions")
        clipped = torch.clamp(actions, -limit, limit)
        clipping_enabled = True
    if policy_active is None:
        active = torch.ones(actions.shape[0], dtype=torch.bool, device=actions.device)
    else:
        active = (
            policy_active.detach()
            .to(device=actions.device, dtype=torch.bool)
            .reshape(-1)
        )
        if active.shape != (actions.shape[0],):
            raise ValueError("policy_active must select the actions environment axis")

    result: dict[str, Any] = {}
    detached_actions = actions.detach()
    detached_clipped = clipped.detach()
    all_elements = torch.ones_like(detached_actions, dtype=torch.bool)
    active_elements = active.reshape(
        (actions.shape[0],) + (1,) * (actions.ndim - 1)
    ).expand_as(detached_actions)
    for scope, selected_elements in (
        ("all", all_elements),
        ("active", active_elements),
    ):
        raw_stats = _selected_tensor_statistics(detached_actions, selected_elements)
        clipped_stats = _selected_tensor_statistics(
            detached_clipped, selected_elements
        )
        selected_count = selected_elements.sum().to(dtype=torch.float32)
        changed_count = (
            ((detached_actions != detached_clipped) & selected_elements)
            .sum()
            .to(dtype=torch.float32)
        )
        fraction = changed_count / selected_count.clamp_min(1.0)
        result[f"ppo_action/{scope}_row_count"] = (
            torch.as_tensor(actions.shape[0], device=actions.device).to(torch.float32)
            if scope == "all"
            else active.sum().to(dtype=torch.float32)
        )
        result[f"ppo_action/{scope}_element_clip_fraction"] = (
            torch.nan_to_num(fraction)
            if clipping_enabled
            else torch.zeros((), device=actions.device)
        )
        for name, value in raw_stats.items():
            result[f"ppo_action/{scope}_raw_{name}"] = value
        for name, value in clipped_stats.items():
            result[f"ppo_action/{scope}_wrapper_clipped_{name}"] = value
    return result


def _default_policy_active(wrapper: Any, actions: Any) -> Any | None:
    env = getattr(wrapper, "env", None)
    while env is not None:
        mask = getattr(env, "_training_last_policy_observation_valid", None)
        if mask is not None:
            return mask
        next_env = getattr(env, "env", None)
        if next_env is env:
            break
        env = next_env
    return None


def make_action_telemetry_wrapper_type(
    base_wrapper_type: type,
    *,
    policy_active_getter: Callable[[Any, Any], Any | None] = _default_policy_active,
) -> type:
    """Create an RSL-RL wrapper subclass that adds bounded scalar telemetry."""

    class ActionTelemetryWrapper(base_wrapper_type):
        def __init__(
            self,
            env: Any,
            clip_actions: float | None = None,
            *args: Any,
            step_observer: Any | None = None,
            **kwargs: Any,
        ):
            super().__init__(env, clip_actions=clip_actions, *args, **kwargs)
            self._telemetry_clip_actions = clip_actions
            self._telemetry_step_observer = step_observer

        def step(self, actions: Any):
            active = policy_active_getter(self, actions)
            metrics = action_clip_telemetry(
                actions,
                clip_actions=self._telemetry_clip_actions,
                policy_active=active,
            )
            if self._telemetry_step_observer is None:
                result = super().step(actions)
            else:
                with self._telemetry_step_observer.control_step():
                    result = super().step(actions)
                self._telemetry_step_observer.finalize_control_step(result[-1])
            extras = result[-1]
            if not isinstance(extras, dict):
                raise TypeError("RSL-RL wrapper step extras must be a dictionary")
            log = extras.setdefault("log", {})
            if not isinstance(log, dict):
                raise TypeError("RSL-RL wrapper extras['log'] must be a dictionary")
            log.update(metrics)
            return result

    ActionTelemetryWrapper.__name__ = f"ActionTelemetry{base_wrapper_type.__name__}"
    return ActionTelemetryWrapper
