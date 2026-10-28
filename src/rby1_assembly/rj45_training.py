"""Pure training contracts and tensor math for RJ45 residual control.

The standard policy observation accepts an estimated relative connector pose;
object-pose ground truth remains confined to setup, reward, and evaluation. The
explicit 85D diagnostic mode additionally exposes the simulator's sampled
calibration-bias latent for observability experiments, never as a sensor claim.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


# Simulation-design settle limits; these are not hardware calibration values.
# Finger reference error includes the deliberate closing preload against contact.
HELD_GRASP_STABILITY_LIMITS = {
    "max_object_linear_speed_m_s": 0.005,
    "max_object_angular_speed_rad_s": 0.05,
    "max_revolute_joint_speed_rad_s": 0.05,
    "max_prismatic_joint_speed_m_s": 0.005,
    "max_revolute_tracking_error_rad": 0.05,
    "max_prismatic_tracking_error_m": 0.005,
}


_ACTION_DIMS = {"right_only_6d": 6, "both_arms_12d": 12}
POST_INTEGRAL_STEP_ACTION_APPLICATION = "post_integral_step"
ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY = "absolute_tip_pose_target_insert_only"
TIP_TARGET_POSE_OFFSET_DESCRIPTION = "tip target pose offset INSERT-only"
_ACTION_APPLICATION_MODES = {
    POST_INTEGRAL_STEP_ACTION_APPLICATION,
    ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
}
_TOP_LEVEL_KEYS = {
    "schema_version",
    "task_id",
    "episode_scope",
    "active_horizon_s",
    "warmup_s",
    "action_mode",
    "action_application_mode",
    "base_observation_dim",
    "diagnostic_observation_mode",
    "calibration_bias",
    "reward",
    "task_physics",
    "held_reset_staging",
}
_V2_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS | {"reward_profile"}
_V3_TOP_LEVEL_KEYS = _TOP_LEVEL_KEYS | {"held_warmup_restage_retry_limit"}
_BIAS_KEYS = {"translation_levels_m", "rotation_max_levels_rad", "initial_level"}
_REWARD_KEYS = {
    "depth_scale_m",
    "lateral_scale_m",
    "angle_scale_rad",
    "force_scale_n",
    "force_penalty",
    "action_delta_penalty",
    "success_bonus",
    "failure_penalty",
    "normalized_error_cap",
}
_REWARD_PROFILE_KEYS = {
    "profile_id",
    "potential_type",
    "shaping_mode",
    "horizon_semantics",
    "discount_factor",
    "keypoint_geometry",
}
_KEYPOINT_GEOMETRY_KEYS = {
    "frame",
    "length_unit",
    "plug_keypoints_tip_frame_m",
    "target_tip_position_entry_m",
    "axis_scales_m",
    "geometry_source",
}
_KEYPOINT_SOURCE_KEYS = {
    "task_config_path",
    "task_config_sha256",
    "plug_asset_path",
    "plug_asset_sha256",
    "dimension_source",
}
_PHYSICS_KEYS = {"mass_randomization_enabled"}
_OBSERVATION_DIM = 79
_DIAGNOSTIC_OBSERVATION_DIM = 85
_DIAGNOSTIC_OBSERVATION_MODES = {
    "standard": _OBSERVATION_DIM,
    "persistent_calibration_bias": _DIAGNOSTIC_OBSERVATION_DIM,
}
PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION = (
    "episode_sampled_calibration_bias_parameter_6d_normalized_by_level_maxima_"
    "privileged_simulation_diagnostic_not_available_from_real_sensors"
)
STANDARD_POLICY_OBSERVATION_SOURCE = (
    "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
)
PERSISTENT_BIAS_DIAGNOSTIC_OBSERVATION_SOURCE = (
    STANDARD_POLICY_OBSERVATION_SOURCE
    + "_plus_privileged_episode_sampled_calibration_bias_latent"
)
STANDARD_GROUND_TRUTH_USAGE = (
    "scripted_setup_control_and_safety_then_reward_and_independent_evaluation_only"
)
PERSISTENT_BIAS_DIAGNOSTIC_GROUND_TRUTH_USAGE = (
    "scripted_setup_control_and_safety_then_reward_and_independent_evaluation_"
    "plus_privileged_episode_sampled_calibration_bias_latent"
)
_SHA256_LENGTH = 64


def _finite_number(
    value: Any,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"{name} keys differ: missing={missing}, unexpected={unexpected}")


def _finite_level_array(value: Any, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three levels")
    result = [_finite_number(item, f"{name}[{index}]", positive=True) for index, item in enumerate(value)]
    if result != sorted(result) or len(set(result)) != len(result):
        raise ValueError(f"{name} must be strictly increasing")
    return result


def load_training_config(path: Path) -> dict[str, Any]:
    """Load the finite, fail-closed residual-training contract."""

    path = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        raw = path.read_bytes()
        config = json.loads(raw.decode("utf-8"), parse_constant=reject_constant)
        json.dumps(config, allow_nan=False)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid finite training configuration {path}: {error}") from error

    config = _mapping(config, "training configuration")
    schema_version = config.get("schema_version")
    if schema_version == 1:
        _exact_keys(config, _TOP_LEVEL_KEYS, "training configuration")
    elif schema_version == 2:
        _exact_keys(config, _V2_TOP_LEVEL_KEYS, "training configuration")
    elif schema_version == 3:
        _exact_keys(config, _V3_TOP_LEVEL_KEYS, "training configuration")
    else:
        raise ValueError("schema_version must be 1, 2, or 3")
    if schema_version == 3:
        retry_limit = config["held_warmup_restage_retry_limit"]
        if type(retry_limit) is not int or retry_limit not in (0, 1):
            raise ValueError("held_warmup_restage_retry_limit must be 0 or 1")
    else:
        config["held_warmup_restage_retry_limit"] = 0
    if config["task_id"] != "rj45-v1":
        raise ValueError("task_id must be rj45-v1")
    if config["episode_scope"] != "held_grasp_alignment_insertion":
        raise ValueError("episode_scope must be held_grasp_alignment_insertion")
    active_horizon = _finite_number(config["active_horizon_s"], "active_horizon_s", positive=True)
    warmup = _finite_number(config["warmup_s"], "warmup_s", nonnegative=True)
    if warmup >= active_horizon:
        raise ValueError("warmup_s must be shorter than active_horizon_s")
    if config["action_mode"] not in _ACTION_DIMS:
        raise ValueError(f"action_mode must be one of {sorted(_ACTION_DIMS)}")
    if config["action_application_mode"] not in _ACTION_APPLICATION_MODES:
        raise ValueError(
            "action_application_mode must be post_integral_step or "
            "absolute_tip_pose_target_insert_only"
        )
    if (
        config["action_application_mode"] == ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY
        and config["action_mode"] != "right_only_6d"
    ):
        raise ValueError(
            "absolute_tip_pose_target_insert_only requires right_only_6d"
        )
    if (
        type(config["base_observation_dim"]) is not int
        or config["base_observation_dim"] != _OBSERVATION_DIM
    ):
        raise ValueError(f"base_observation_dim must be {_OBSERVATION_DIM}")
    if config["diagnostic_observation_mode"] not in _DIAGNOSTIC_OBSERVATION_MODES:
        raise ValueError(
            "diagnostic_observation_mode must be standard or "
            "persistent_calibration_bias"
        )

    bias = _mapping(config["calibration_bias"], "calibration_bias")
    _exact_keys(bias, _BIAS_KEYS, "calibration_bias")
    bias["translation_levels_m"] = _finite_level_array(
        bias["translation_levels_m"], "calibration_bias.translation_levels_m"
    )
    bias["rotation_max_levels_rad"] = _finite_level_array(
        bias["rotation_max_levels_rad"], "calibration_bias.rotation_max_levels_rad"
    )
    initial_level = bias["initial_level"]
    if type(initial_level) is not int or not 0 <= initial_level < 3:
        raise ValueError("calibration_bias.initial_level must index the three levels")

    reward = _mapping(config["reward"], "reward")
    _exact_keys(reward, _REWARD_KEYS, "reward")
    for name in (
        "depth_scale_m",
        "lateral_scale_m",
        "angle_scale_rad",
        "force_scale_n",
        "success_bonus",
        "failure_penalty",
        "normalized_error_cap",
    ):
        reward[name] = _finite_number(reward[name], f"reward.{name}", positive=True)
    for name in ("force_penalty", "action_delta_penalty"):
        reward[name] = _finite_number(reward[name], f"reward.{name}", nonnegative=True)

    if schema_version == 2:
        profile = _mapping(config["reward_profile"], "reward_profile")
        _exact_keys(profile, _REWARD_PROFILE_KEYS, "reward_profile")
        profile_id = profile["profile_id"]
        if profile_id not in {"a1_discounted_pose_v2", "a2_keypoint_shape_v2"}:
            raise ValueError("reward_profile.profile_id is unsupported")
        expected_potential = (
            "normalized_pose" if profile_id == "a1_discounted_pose_v2" else "cad_keypoints"
        )
        if profile["potential_type"] != expected_potential:
            raise ValueError("reward_profile potential_type disagrees with profile_id")
        if profile["shaping_mode"] != "discounted_potential":
            raise ValueError("schema 2 reward profiles require discounted_potential shaping")
        if profile["horizon_semantics"] != "finite_horizon_timeout_absorbing":
            raise ValueError("schema 2 reward profiles require finite-horizon absorbing timeouts")
        profile["discount_factor"] = _finite_number(
            profile["discount_factor"], "reward_profile.discount_factor", positive=True
        )
        if profile["discount_factor"] > 1.0:
            raise ValueError("reward_profile.discount_factor must be at most 1")
        geometry = profile["keypoint_geometry"]
        if expected_potential == "normalized_pose":
            if geometry is not None:
                raise ValueError("A1 normalized-pose profile must not define keypoint geometry")
        else:
            geometry = _mapping(geometry, "reward_profile.keypoint_geometry")
            _exact_keys(geometry, _KEYPOINT_GEOMETRY_KEYS, "reward_profile.keypoint_geometry")
            if geometry["frame"] != "plug_tip_to_socket_entry" or geometry["length_unit"] != "m":
                raise ValueError("keypoint geometry must use plug_tip_to_socket_entry in metres")
            points = geometry["plug_keypoints_tip_frame_m"]
            if not isinstance(points, list) or not 8 <= len(points) <= 16:
                raise ValueError("keypoint geometry requires 8 to 16 plug points")
            geometry["plug_keypoints_tip_frame_m"] = [
                [_finite_number(v, f"keypoint[{i}][{j}]") for j, v in enumerate(point)]
                for i, point in enumerate(points)
                if isinstance(point, list) and len(point) == 3
            ]
            if len(geometry["plug_keypoints_tip_frame_m"]) != len(points):
                raise ValueError("each plug keypoint must have three coordinates")
            for name in ("target_tip_position_entry_m", "axis_scales_m"):
                value = geometry[name]
                if not isinstance(value, list) or len(value) != 3:
                    raise ValueError(f"keypoint_geometry.{name} must have three coordinates")
                geometry[name] = [
                    _finite_number(v, f"keypoint_geometry.{name}[{i}]", positive=name == "axis_scales_m")
                    for i, v in enumerate(value)
                ]
            source = _mapping(geometry["geometry_source"], "keypoint_geometry.geometry_source")
            _exact_keys(source, _KEYPOINT_SOURCE_KEYS, "keypoint_geometry.geometry_source")
            for name in ("task_config_sha256", "plug_asset_sha256"):
                value = source[name]
                if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
                    raise ValueError(f"keypoint_geometry.geometry_source.{name} must be SHA-256")
                try:
                    int(value, 16)
                except ValueError as error:
                    raise ValueError(f"keypoint_geometry.geometry_source.{name} must be SHA-256") from error
            for name in ("task_config_path", "plug_asset_path", "dimension_source"):
                if not isinstance(source[name], str) or not source[name].strip():
                    raise ValueError(f"keypoint_geometry.geometry_source.{name} must be nonempty")
            project_root = next(
                (
                    parent
                    for parent in path.resolve().parents
                    if (parent / "src/rby1_assembly").is_dir()
                    and (parent / "configs/tasks/rj45-v1.json").is_file()
                ),
                None,
            )
            if project_root is None:
                raise ValueError("schema 2 keypoint config must be located inside the project")
            for path_key, digest_key in (
                ("task_config_path", "task_config_sha256"),
                ("plug_asset_path", "plug_asset_sha256"),
            ):
                source_path = Path(source[path_key])
                if source_path.is_absolute() or ".." in source_path.parts:
                    raise ValueError(f"keypoint_geometry.geometry_source.{path_key} must be project-relative")
                resolved_source = project_root / source_path
                try:
                    actual_digest = hashlib.sha256(resolved_source.read_bytes()).hexdigest()
                except OSError as error:
                    raise ValueError(f"keypoint geometry source is unreadable: {source_path}") from error
                if actual_digest != source[digest_key]:
                    raise ValueError(f"keypoint geometry source hash mismatch: {source_path}")

    physics = _mapping(config["task_physics"], "task_physics")
    _exact_keys(physics, _PHYSICS_KEYS, "task_physics")
    if physics["mass_randomization_enabled"] is not False:
        raise ValueError("first-version task physics must keep masses fixed")

    staging = _mapping(config["held_reset_staging"], "held_reset_staging")
    _exact_keys(staging, {"control_steps", "object_positions_local_m", "provenance"}, "held_reset_staging")
    if type(staging["control_steps"]) is not int or staging["control_steps"] != 1:
        raise ValueError("held_reset_staging.control_steps must be the verified one-step flush")
    positions = staging["object_positions_local_m"]
    if not isinstance(positions, list) or len(positions) != 2:
        raise ValueError("held_reset_staging requires two object positions")
    normalized_positions = []
    for index, position in enumerate(positions):
        if not isinstance(position, list) or len(position) != 3:
            raise ValueError("held_reset_staging object positions must be 3D")
        normalized_positions.append([
            _finite_number(value, f"held_reset_staging.object_positions_local_m[{index}]")
            for value in position
        ])
    if normalized_positions[0] == normalized_positions[1]:
        raise ValueError("held_reset_staging object positions must be distinct")
    staging["object_positions_local_m"] = normalized_positions
    provenance = _mapping(staging["provenance"], "held_reset_staging.provenance")
    _exact_keys(provenance, {"source", "defined_at", "valid_for", "unit"}, "held_reset_staging.provenance")
    if provenance["source"] != "simulation_design_assumption" or provenance["unit"] != "m":
        raise ValueError("held_reset_staging must disclose simulation-design positions in m")
    if not isinstance(provenance["valid_for"], str) or not provenance["valid_for"].strip():
        raise ValueError("held_reset_staging.provenance.valid_for must be nonempty")
    try:
        if not isinstance(provenance["defined_at"], str):
            raise ValueError("date must be a string")
        if date.fromisoformat(provenance["defined_at"]).isoformat() != provenance["defined_at"]:
            raise ValueError("date must use ISO format")
    except ValueError as error:
        raise ValueError("held_reset_staging.provenance.defined_at must be an ISO date") from error

    config["_config_sha256"] = hashlib.sha256(raw).hexdigest()
    return config


def reward_profile_metadata(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable reward identity for lineage and evaluation reports."""

    if config.get("schema_version") != 2:
        return {
            "reward_profile_id": "a0_legacy_pose_v1",
            "reward_potential_type": "normalized_pose",
            "reward_shaping_mode": "undiscounted_difference",
            "reward_horizon_semantics": "legacy_finite_horizon",
            "reward_discount_factor": None,
        }
    profile = _mapping(config.get("reward_profile"), "reward_profile")
    return {
        "reward_profile_id": profile["profile_id"],
        "reward_potential_type": profile["potential_type"],
        "reward_shaping_mode": profile["shaping_mode"],
        "reward_horizon_semantics": profile["horizon_semantics"],
        "reward_discount_factor": float(profile["discount_factor"]),
    }


def uses_legacy_reward_arithmetic(config: Mapping[str, Any]) -> bool:
    """Keep schema 1 and retry-only schema 3 on the exact legacy sum order."""

    return config.get("schema_version") != 2


def validate_reward_profile_runner(config: Mapping[str, Any], runner_config: Mapping[str, Any]) -> None:
    """Fail closed when discounted reward shaping and PPO use different gamma values."""

    metadata = reward_profile_metadata(config)
    if metadata["reward_discount_factor"] is None:
        return
    algorithm = _mapping(runner_config.get("algorithm"), "runner_config.algorithm")
    gamma = _finite_number(algorithm.get("gamma"), "runner_config.algorithm.gamma", positive=True)
    if gamma > 1.0 or not math.isclose(
        gamma, metadata["reward_discount_factor"], rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("reward discount_factor must exactly match PPO algorithm.gamma")


def validate_reward_horizon_contract(
    config: Mapping[str, Any], *, is_finite_horizon: bool
) -> None:
    """Require the Isaac finite-horizon contract assumed by A1/A2 shaping."""

    if type(is_finite_horizon) is not bool:
        raise ValueError("is_finite_horizon must be boolean")
    if config.get("schema_version") == 2 and is_finite_horizon is not True:
        raise ValueError("schema 2 reward profiles require is_finite_horizon=True")


def configure_actor_output_initialization(
    runner: Any,
    *,
    expected_output_dim: int,
    zero_initialize: bool,
) -> dict[str, Any]:
    """Inspect or zero only the deterministic actor mean network's final Linear layer."""

    if type(expected_output_dim) is not int or expected_output_dim <= 0:
        raise ValueError("expected_output_dim must be a positive integer")
    if type(zero_initialize) is not bool:
        raise ValueError("zero_initialize must be boolean")
    torch = _torch()
    try:
        actor = runner.alg.policy.actor
    except AttributeError as error:
        raise ValueError("runner must expose runner.alg.policy.actor") from error
    if not isinstance(actor, torch.nn.Sequential):
        raise ValueError("RSL-RL actor must be torch.nn.Sequential")
    layers = list(actor.children())
    if not layers or not isinstance(layers[-1], torch.nn.Linear):
        raise ValueError("RSL-RL actor must end in torch.nn.Linear")
    output_layer = layers[-1]
    if output_layer.out_features != expected_output_dim or output_layer.bias is None:
        raise ValueError(
            f"actor output layer must have {expected_output_dim} outputs and a bias"
        )
    input_layers = [layer for layer in layers if isinstance(layer, torch.nn.Linear)]
    if not input_layers:
        raise ValueError("RSL-RL actor must contain a Linear input layer")

    def tensor_sha256(value: Any) -> str:
        raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    pre_weight_sha256 = tensor_sha256(output_layer.weight)
    pre_bias_sha256 = tensor_sha256(output_layer.bias)
    if zero_initialize:
        with torch.no_grad():
            output_layer.weight.zero_()
            output_layer.bias.zero_()

    actor_was_training = actor.training
    actor.eval()
    input_dim = input_layers[0].in_features
    sample = torch.stack(
        (
            torch.zeros(input_dim, device=output_layer.weight.device, dtype=output_layer.weight.dtype),
            torch.ones(input_dim, device=output_layer.weight.device, dtype=output_layer.weight.dtype),
            torch.linspace(
                -1.0,
                1.0,
                input_dim,
                device=output_layer.weight.device,
                dtype=output_layer.weight.dtype,
            ),
        )
    )
    with torch.no_grad():
        deterministic_output = actor(sample)
    actor.train(actor_was_training)
    if deterministic_output.shape != (3, expected_output_dim):
        raise ValueError("actor deterministic output shape disagrees with action dimension")
    if not bool(torch.isfinite(deterministic_output).all()):
        raise ValueError("actor deterministic output must be finite")
    deterministic_output_all_zero = bool(
        torch.count_nonzero(deterministic_output).item() == 0
    )
    weight_all_zero = bool(torch.count_nonzero(output_layer.weight).item() == 0)
    bias_all_zero = bool(torch.count_nonzero(output_layer.bias).item() == 0)
    if zero_initialize and not (
        deterministic_output_all_zero and weight_all_zero and bias_all_zero
    ):
        raise RuntimeError("zero-initialized actor output verification failed")
    return {
        "mode": (
            "residual_actor_final_linear_zero"
            if zero_initialize
            else "rsl_rl_default"
        ),
        "zero_actor_output_layer": zero_initialize,
        "actor_module_path": "runner.alg.policy.actor",
        "final_layer_index": len(layers) - 1,
        "final_layer_class": "torch.nn.Linear",
        "input_dimension": input_dim,
        "output_dimension": expected_output_dim,
        "pre_weight_sha256": pre_weight_sha256,
        "pre_bias_sha256": pre_bias_sha256,
        "post_weight_sha256": tensor_sha256(output_layer.weight),
        "post_bias_sha256": tensor_sha256(output_layer.bias),
        "weight_all_zero_after_initialization": weight_all_zero,
        "bias_all_zero_after_initialization": bias_all_zero,
        "deterministic_actor_test_input_count": 3,
        "deterministic_actor_output_all_zero": deterministic_output_all_zero,
        "deterministic_actor_output_max_abs": float(
            torch.abs(deterministic_output).max().item()
        ),
        "stochastic_policy_noise_modified": False,
        "method_reference": (
            {
                "title": "Residual Policy Learning",
                "url": "https://arxiv.org/pdf/1812.06298",
                "section": "IV-A",
                "adaptation": "zero initialization of final residual actor mean layer only",
            }
            if zero_initialize
            else None
        ),
    }


def policy_observation_dimension(config: Mapping[str, Any]) -> int:
    """Return the strict policy width selected by the explicit diagnostic mode."""

    mode = config.get("diagnostic_observation_mode")
    try:
        return _DIAGNOSTIC_OBSERVATION_MODES[mode]
    except (KeyError, TypeError) as error:
        raise ValueError("unsupported diagnostic_observation_mode") from error


def policy_observation_schema_version(config: Mapping[str, Any]) -> int:
    """Keep the deployed 79D schema at v1 and identify the privileged 85D schema."""

    return 1 if policy_observation_dimension(config) == _OBSERVATION_DIM else 2


def policy_observation_source(config: Mapping[str, Any]) -> str:
    """Describe whether policy input includes the privileged sampled latent."""

    return (
        PERSISTENT_BIAS_DIAGNOSTIC_OBSERVATION_SOURCE
        if policy_observation_dimension(config) == _DIAGNOSTIC_OBSERVATION_DIM
        else STANDARD_POLICY_OBSERVATION_SOURCE
    )


def policy_ground_truth_usage(config: Mapping[str, Any]) -> str:
    """Disclose privileged simulator-latent use without implying object-pose GT."""

    return (
        PERSISTENT_BIAS_DIAGNOSTIC_GROUND_TRUTH_USAGE
        if policy_observation_dimension(config) == _DIAGNOSTIC_OBSERVATION_DIM
        else STANDARD_GROUND_TRUTH_USAGE
    )


def privileged_diagnostic_observation_metadata(
    config: Mapping[str, Any],
) -> dict[str, str]:
    """Return additive metadata only for the explicit privileged 85D mode."""

    if policy_observation_dimension(config) == _OBSERVATION_DIM:
        return {}
    return {
        "diagnostic_observation_mode": "persistent_calibration_bias",
        "privileged_diagnostic_observation": PERSISTENT_BIAS_DIAGNOSTIC_DESCRIPTION,
    }


def action_application_contract_metadata(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return additive contract fields only for the INSERT tip-target mode."""

    mode = config.get("action_application_mode")
    if mode == POST_INTEGRAL_STEP_ACTION_APPLICATION:
        return {}
    if mode != ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY:
        raise ValueError("unsupported action_application_mode")
    if config.get("action_mode") != "right_only_6d":
        raise ValueError(
            "absolute_tip_pose_target_insert_only requires right_only_6d"
        )
    return {
        "action_application_mode": ABSOLUTE_TIP_POSE_TARGET_INSERT_ONLY,
        "policy_active_phases": ["insert"],
    }


def policy_observation_ground_truth_flags(
    config: Mapping[str, Any],
) -> dict[str, bool]:
    """Separate broad privileged input from object-pose ground truth."""

    privileged_latent = policy_observation_dimension(config) == _DIAGNOSTIC_OBSERVATION_DIM
    return {
        "ground_truth_in_policy_observation": privileged_latent,
        "object_pose_ground_truth_in_policy_observation": False,
        "privileged_simulation_latent_in_policy_observation": privileged_latent,
    }


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised in Isaac/Torch environment
        raise RuntimeError("RJ45 training tensor math requires PyTorch") from error
    return torch


def _tensor(value: Any, name: str):
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be a finite floating-point tensor")
    return value


def _last_dim(value: Any, size: int, name: str):
    tensor = _tensor(value, name)
    if tensor.ndim < 1 or tensor.shape[-1] != size:
        raise ValueError(f"{name} must have last dimension {size}")
    return tensor


def expand_residual_actions(actions: Any, action_mode: str):
    """Expand right-only 6D actions into the common left/right 12D layout."""

    if action_mode not in _ACTION_DIMS:
        raise ValueError(f"unsupported action_mode: {action_mode}")
    actions = _last_dim(actions, _ACTION_DIMS[action_mode], "actions")
    if action_mode == "both_arms_12d":
        return actions
    torch = _torch()
    return torch.cat((torch.zeros_like(actions), actions), dim=-1)


def _quaternion_multiply_wxyz(left: Any, right: Any):
    torch = _torch()
    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _quaternion_rotate_wxyz(quaternion: Any, vector: Any):
    torch = _torch()
    xyz = quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def _rotation_vector_quaternion(rotation_vector: Any):
    torch = _torch()
    angle = torch.linalg.vector_norm(rotation_vector, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(angle/2)/angle has limit 1/2, avoiding a zero-axis branch and NaNs.
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half) / angle,
        0.5 - angle.square() / 48.0,
    )
    return torch.cat((torch.cos(half), rotation_vector * scale), dim=-1)


def _quaternion_inverse_wxyz(quaternion: Any):
    torch = _torch()
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def apply_tip_target_pose_offset(
    plug_body_position_world: Any,
    plug_body_quaternion_world_wxyz: Any,
    socket_entry_quaternion_world_wxyz: Any,
    plug_tip_position_local: Any,
    plug_tip_quaternion_local_wxyz: Any,
    offset_entry_6d: Any,
):
    """Apply a socket-entry-frame 6D offset at the plug tip pivot."""

    body_position = _last_dim(plug_body_position_world, 3, "plug_body_position_world")
    body_quaternion = _last_dim(
        plug_body_quaternion_world_wxyz, 4, "plug_body_quaternion_world_wxyz"
    )
    entry_quaternion = _last_dim(
        socket_entry_quaternion_world_wxyz, 4, "socket_entry_quaternion_world_wxyz"
    )
    tip_position_local = _last_dim(plug_tip_position_local, 3, "plug_tip_position_local")
    tip_quaternion_local = _last_dim(
        plug_tip_quaternion_local_wxyz, 4, "plug_tip_quaternion_local_wxyz"
    )
    offset = _last_dim(offset_entry_6d, 6, "offset_entry_6d")
    batch_shape = body_position.shape[:-1]
    if any(
        value.shape[:-1] != batch_shape
        for value in (
            body_quaternion,
            entry_quaternion,
            tip_position_local,
            tip_quaternion_local,
            offset,
        )
    ):
        raise ValueError("tip target pose offset inputs must have matching batch shapes")
    torch = _torch()
    for quaternion, name in (
        (body_quaternion, "plug_body_quaternion_world_wxyz"),
        (entry_quaternion, "socket_entry_quaternion_world_wxyz"),
        (tip_quaternion_local, "plug_tip_quaternion_local_wxyz"),
    ):
        norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        if bool((torch.abs(norm - 1.0) > 1.0e-4).any()):
            raise ValueError(f"{name} must be unit length")

    current_tip_position = body_position + _quaternion_rotate_wxyz(
        body_quaternion, tip_position_local
    )
    current_tip_quaternion = _quaternion_multiply_wxyz(
        body_quaternion, tip_quaternion_local
    )
    translated_tip_position = current_tip_position + _quaternion_rotate_wxyz(
        entry_quaternion, offset[..., :3]
    )
    delta_entry = _rotation_vector_quaternion(offset[..., 3:])
    delta_world = _quaternion_multiply_wxyz(
        _quaternion_multiply_wxyz(entry_quaternion, delta_entry),
        _quaternion_inverse_wxyz(entry_quaternion),
    )
    target_tip_quaternion = _quaternion_multiply_wxyz(
        delta_world, current_tip_quaternion
    )
    target_body_quaternion = _quaternion_multiply_wxyz(
        target_tip_quaternion, _quaternion_inverse_wxyz(tip_quaternion_local)
    )
    target_body_quaternion = target_body_quaternion / torch.linalg.vector_norm(
        target_body_quaternion, dim=-1, keepdim=True
    )
    target_body_position = translated_tip_position - _quaternion_rotate_wxyz(
        target_body_quaternion, tip_position_local
    )
    return target_body_position, target_body_quaternion


def apply_relative_pose_bias(
    position_entry: Any,
    quaternion_entry_wxyz: Any,
    translation_bias_entry: Any,
    rotation_vector_bias_entry: Any,
):
    """Apply one fixed calibration SE(3) transform in the socket-entry frame."""

    position = _last_dim(position_entry, 3, "position_entry")
    quaternion = _last_dim(quaternion_entry_wxyz, 4, "quaternion_entry_wxyz")
    translation = _last_dim(translation_bias_entry, 3, "translation_bias_entry")
    rotation_vector = _last_dim(rotation_vector_bias_entry, 3, "rotation_vector_bias_entry")
    if not (position.shape[:-1] == quaternion.shape[:-1] == translation.shape[:-1] == rotation_vector.shape[:-1]):
        raise ValueError("pose and calibration bias batch shapes must match")
    torch = _torch()
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((torch.abs(quaternion_norm - 1.0) > 1.0e-4).any()):
        raise ValueError("quaternion_entry_wxyz must be unit length")
    quaternion = quaternion / quaternion_norm
    bias_quaternion = _rotation_vector_quaternion(rotation_vector)
    biased_position = _quaternion_rotate_wxyz(bias_quaternion, position) + translation
    biased_quaternion = _quaternion_multiply_wxyz(bias_quaternion, quaternion)
    biased_quaternion = biased_quaternion / torch.linalg.vector_norm(
        biased_quaternion, dim=-1, keepdim=True
    )
    # q and -q represent the same rotation. Canonicalizing makes observations continuous near identity.
    sign = torch.where(biased_quaternion[..., :1] < 0.0, -1.0, 1.0)
    return biased_position, biased_quaternion * sign


def normalized_insertion_potential(position_entry: Any, quaternion_entry_wxyz: Any, config: Mapping[str, Any]):
    """Return GT-only insertion progress minus capped lateral/orientation error."""

    position = _last_dim(position_entry, 3, "position_entry")
    quaternion = _last_dim(quaternion_entry_wxyz, 4, "quaternion_entry_wxyz")
    if position.shape[:-1] != quaternion.shape[:-1]:
        raise ValueError("relative pose batch shapes must match")
    torch = _torch()
    norm = torch.linalg.vector_norm(quaternion, dim=-1)
    if bool((torch.abs(norm - 1.0) > 1.0e-4).any()):
        raise ValueError("quaternion_entry_wxyz must be unit length")
    reward = _mapping(config.get("reward"), "reward")
    depth = torch.clamp(position[..., 1] / float(reward["depth_scale_m"]), -2.0, 1.1)
    lateral = torch.linalg.vector_norm(position[..., (0, 2)], dim=-1) / float(
        reward["lateral_scale_m"]
    )
    angle = 2.0 * torch.atan2(
        torch.linalg.vector_norm(quaternion[..., 1:], dim=-1),
        torch.abs(quaternion[..., 0]),
    ) / float(reward["angle_scale_rad"])
    cap = float(reward["normalized_error_cap"])
    return depth - torch.clamp(lateral, max=cap) - torch.clamp(angle, max=cap)


def keypoint_insertion_potential(position_entry: Any, quaternion_entry_wxyz: Any, config: Mapping[str, Any]):
    """Score CAD-derived plug points against their aligned target pose.

    This is a corresponding-keypoint pose potential inspired by geometry-aware
    assembly rewards. It is not a signed-distance field or a cavity-clearance
    calculation.
    """

    position = _last_dim(position_entry, 3, "position_entry")
    quaternion = _last_dim(quaternion_entry_wxyz, 4, "quaternion_entry_wxyz")
    if position.shape[:-1] != quaternion.shape[:-1]:
        raise ValueError("relative pose batch shapes must match")
    torch = _torch()
    norm = torch.linalg.vector_norm(quaternion, dim=-1)
    if bool((torch.abs(norm - 1.0) > 1.0e-4).any()):
        raise ValueError("quaternion_entry_wxyz must be unit length")
    profile = _mapping(config.get("reward_profile"), "reward_profile")
    if profile.get("potential_type") != "cad_keypoints":
        raise ValueError("keypoint potential requires cad_keypoints reward profile")
    geometry = _mapping(profile.get("keypoint_geometry"), "reward_profile.keypoint_geometry")
    points = torch.as_tensor(
        geometry["plug_keypoints_tip_frame_m"], dtype=position.dtype, device=position.device
    )
    target = torch.as_tensor(
        geometry["target_tip_position_entry_m"], dtype=position.dtype, device=position.device
    )
    scales = torch.as_tensor(
        geometry["axis_scales_m"], dtype=position.dtype, device=position.device
    )
    batch_shape = position.shape[:-1]
    expanded_points = points.reshape((1,) * len(batch_shape) + points.shape).expand(
        batch_shape + points.shape
    )
    rotated = _quaternion_rotate_wxyz(
        quaternion.unsqueeze(-2).expand(batch_shape + points.shape[:-1] + (4,)),
        expanded_points,
    )
    actual = position.unsqueeze(-2) + rotated
    desired = target + points
    normalized_error = torch.linalg.vector_norm((actual - desired) / scales, dim=-1)
    cap = float(_mapping(config.get("reward"), "reward")["normalized_error_cap"])
    return 1.0 - torch.clamp(normalized_error, max=cap).mean(dim=-1)


def insertion_potential(position_entry: Any, quaternion_entry_wxyz: Any, config: Mapping[str, Any]):
    """Select the versioned potential while preserving schema-1 A0 behavior."""

    if config.get("schema_version") == 2 and config["reward_profile"]["potential_type"] == "cad_keypoints":
        return keypoint_insertion_potential(position_entry, quaternion_entry_wxyz, config)
    return normalized_insertion_potential(position_entry, quaternion_entry_wxyz, config)


def residual_reward_components(
    previous_potential: Any,
    next_potential: Any,
    contact_force_n: Any,
    action_delta: Any,
    policy_active: Any,
    success: Any,
    failed: Any,
    true_terminated: Any,
    truncated: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return separately attributable reward terms for one environment step."""

    torch = _torch()
    previous = _tensor(previous_potential, "previous_potential")
    following = _tensor(next_potential, "next_potential")
    force = _tensor(contact_force_n, "contact_force_n")
    delta = _tensor(action_delta, "action_delta")
    if previous.shape != following.shape:
        raise ValueError("potential shapes must match")
    if force.ndim == previous.ndim + 1:
        force = torch.linalg.vector_norm(force, dim=-1)
    if delta.ndim != previous.ndim + 1 or delta.shape[:-1] != previous.shape:
        raise ValueError("action_delta must add one action dimension to the potential shape")
    if force.shape != previous.shape:
        raise ValueError("contact_force_n must match the potential shape or add one vector dimension")
    masks = {
        "policy_active": policy_active,
        "success": success,
        "failed": failed,
        "true_terminated": true_terminated,
        "truncated": truncated,
    }
    for name, value in masks.items():
        if not isinstance(value, torch.Tensor) or value.shape != previous.shape or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor matching the potential shape")
    if bool((success & failed).any()):
        raise ValueError("success and failed must be mutually exclusive")
    if bool((success & ~true_terminated).any()):
        raise ValueError("success must be a true termination")
    reward = _mapping(config.get("reward"), "reward")
    metadata = reward_profile_metadata(config)
    if metadata["reward_shaping_mode"] == "discounted_potential":
        absorbing = true_terminated | truncated
        terminal_following = torch.where(absorbing, torch.zeros_like(following), following)
        progress = float(metadata["reward_discount_factor"]) * terminal_following - previous
    else:
        progress = following - previous
    force_cost = float(reward["force_penalty"]) * (
        torch.abs(force) / float(reward["force_scale_n"])
    ).square()
    action_cost = float(reward["action_delta_penalty"]) * delta.square().sum(dim=-1)
    active = policy_active.to(previous.dtype)
    terminal = (
        success.to(previous.dtype) * float(reward["success_bonus"])
        - failed.to(previous.dtype) * float(reward["failure_penalty"])
    )
    components = {
        "potential_progress": active * progress,
        "force_cost": -active * force_cost,
        "action_cost": -active * action_cost,
        "terminal": terminal,
        "true_terminated": true_terminated,
        "truncated": truncated,
    }
    components["total"] = (
        components["potential_progress"]
        + components["force_cost"]
        + components["action_cost"]
        + terminal
    )
    if not bool(torch.isfinite(components["total"]).all()):
        raise ValueError("reward components must be finite")
    return components


def residual_reward(
    previous_potential: Any,
    next_potential: Any,
    contact_force_n: Any,
    action_delta: Any,
    policy_active: Any,
    success: Any,
    failed: Any,
    config: Mapping[str, Any],
):
    """Legacy A0 arithmetic kept unchanged for prior checkpoints and baselines."""

    torch = _torch()
    previous = _tensor(previous_potential, "previous_potential")
    following = _tensor(next_potential, "next_potential")
    force = _tensor(contact_force_n, "contact_force_n")
    delta = _tensor(action_delta, "action_delta")
    if previous.shape != following.shape:
        raise ValueError("potential shapes must match")
    if force.ndim == previous.ndim + 1:
        force = torch.linalg.vector_norm(force, dim=-1)
    if delta.ndim != previous.ndim + 1 or delta.shape[:-1] != previous.shape:
        raise ValueError("action_delta must add one action dimension to the potential shape")
    if force.shape != previous.shape:
        raise ValueError("contact_force_n must match the potential shape or add one vector dimension")
    for value, name in ((policy_active, "policy_active"), (success, "success"), (failed, "failed")):
        if not isinstance(value, torch.Tensor) or value.shape != previous.shape or value.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor matching the potential shape")
    if bool((success & failed).any()):
        raise ValueError("success and failed must be mutually exclusive")
    reward = _mapping(config.get("reward"), "reward")
    force_cost = float(reward["force_penalty"]) * (
        torch.abs(force) / float(reward["force_scale_n"])
    ).square()
    action_cost = float(reward["action_delta_penalty"]) * delta.square().sum(dim=-1)
    shaped = following - previous - force_cost - action_cost
    result = torch.where(policy_active, shaped, torch.zeros_like(shaped))
    return (
        result
        + success.to(result.dtype) * float(reward["success_bonus"])
        - failed.to(result.dtype) * float(reward["failure_penalty"])
    )


def build_policy_observation(
    *,
    estimated_relative_position_entry: Any,
    estimated_relative_quaternion_entry_wxyz: Any,
    arm_joint_position_rad: Any,
    arm_joint_velocity_rad_s: Any,
    grip_force_n: Any,
    entry_contact_force_n: Any,
    grip_proven: Any,
    phase_one_hot: Any,
    previous_expanded_action: Any,
    command_depth_m: Any,
    insertion_advance: Any,
    insertion_joint_reference_error_rad: Any,
    policy_active: Any,
):
    """Build the fixed 79D policy vector from estimator and controller state."""

    torch = _torch()
    fields = (
        (_last_dim(estimated_relative_position_entry, 3, "estimated_relative_position_entry"), 3),
        (_last_dim(estimated_relative_quaternion_entry_wxyz, 4, "estimated_relative_quaternion_entry_wxyz"), 4),
        (_last_dim(arm_joint_position_rad, 14, "arm_joint_position_rad"), 14),
        (_last_dim(arm_joint_velocity_rad_s, 14, "arm_joint_velocity_rad_s"), 14),
        (_last_dim(grip_force_n, 2, "grip_force_n"), 2),
        (_last_dim(entry_contact_force_n, 3, "entry_contact_force_n"), 3),
        (_last_dim(grip_proven, 2, "grip_proven"), 2),
        (_last_dim(phase_one_hot, 8, "phase_one_hot"), 8),
        (_last_dim(previous_expanded_action, 12, "previous_expanded_action"), 12),
        (_last_dim(command_depth_m, 1, "command_depth_m"), 1),
        (_last_dim(insertion_advance, 1, "insertion_advance"), 1),
        (_last_dim(insertion_joint_reference_error_rad, 14, "insertion_joint_reference_error_rad"), 14),
        (_last_dim(policy_active, 1, "policy_active"), 1),
    )
    batch_shape = fields[0][0].shape[:-1]
    if any(value.shape[:-1] != batch_shape for value, _ in fields):
        raise ValueError("all observation fields must have the same batch shape")
    quaternion = fields[1][0]
    quaternion_norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if bool((torch.abs(quaternion_norm - 1.0) > 1.0e-4).any()):
        raise ValueError("estimated relative quaternion must be unit length")
    quaternion = quaternion / quaternion_norm
    quaternion = quaternion * torch.where(quaternion[..., :1] < 0.0, -1.0, 1.0)
    if bool(((fields[7][0] < 0.0) | (fields[7][0] > 1.0)).any()) or bool(
        (torch.abs(fields[7][0].sum(dim=-1) - 1.0) > 1.0e-5).any()
    ):
        raise ValueError("phase_one_hot must contain exactly one active phase")
    for binary, name in ((fields[6][0], "grip_proven"), (fields[10][0], "insertion_advance"), (fields[12][0], "policy_active")):
        if bool(((binary != 0.0) & (binary != 1.0)).any()):
            raise ValueError(f"{name} must contain only zero or one")

    observation = torch.cat(
        (
            fields[0][0] * fields[0][0].new_tensor((1000.0, 1.0 / 0.012, 1000.0)),
            torch.cat((quaternion[..., :1], quaternion[..., 1:] * (2.0 / (math.pi / 180.0))), dim=-1),
            fields[2][0] / math.pi,
            fields[3][0] / math.pi,
            fields[4][0] / 8.0,
            fields[5][0] / 8.0,
            fields[6][0],
            fields[7][0],
            fields[8][0],
            fields[9][0] / 0.012,
            fields[10][0],
            fields[11][0] / 0.03,
            fields[12][0],
        ),
        dim=-1,
    )
    if observation.shape[-1] != _OBSERVATION_DIM:
        raise RuntimeError(f"internal observation dimension is {observation.shape[-1]}, expected {_OBSERVATION_DIM}")
    return observation


def append_persistent_calibration_bias_diagnostic(
    observation: Any,
    translation_bias_entry_m: Any,
    rotation_vector_bias_entry_rad: Any,
    *,
    translation_max_m: float,
    rotation_max_rad: float,
    mode: str,
):
    """Append the privileged episode sampler parameter only in diagnostic mode."""

    torch = _torch()
    observation = _last_dim(observation, _OBSERVATION_DIM, "observation")
    if mode == "standard":
        return observation
    if mode != "persistent_calibration_bias":
        raise ValueError("unsupported diagnostic observation mode")
    translation = _last_dim(
        translation_bias_entry_m, 3, "translation_bias_entry_m"
    )
    rotation = _last_dim(
        rotation_vector_bias_entry_rad, 3, "rotation_vector_bias_entry_rad"
    )
    if translation.shape[:-1] != observation.shape[:-1] or rotation.shape[:-1] != observation.shape[:-1]:
        raise ValueError("diagnostic bias fields must match the observation batch shape")
    translation_max = _finite_number(
        translation_max_m, "translation_max_m", positive=True
    )
    rotation_max = _finite_number(
        rotation_max_rad, "rotation_max_rad", positive=True
    )
    if not bool(torch.isfinite(translation).all()) or not bool(torch.isfinite(rotation).all()):
        raise ValueError("diagnostic bias fields must be finite")
    tolerance = 1.0e-6
    if bool((torch.linalg.vector_norm(translation, dim=-1) > translation_max * (1.0 + tolerance)).any()):
        raise ValueError("translation bias exceeds its configured level maximum")
    if bool((torch.linalg.vector_norm(rotation, dim=-1) > rotation_max * (1.0 + tolerance)).any()):
        raise ValueError("rotation bias exceeds its configured level maximum")
    diagnostic = torch.cat(
        (translation / translation_max, rotation / rotation_max), dim=-1
    )
    return torch.cat((observation, diagnostic), dim=-1)


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_vector(value: Any, length: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-element array")
    return [_finite_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _finite_matrix(value: Any, rows: int, columns: int, name: str) -> list[list[float]]:
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError(f"{name} must contain {rows} rows")
    return [_finite_vector(row, columns, f"{name}[{index}]") for index, row in enumerate(value)]


def validate_held_grasp_artifact(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a recorded 62-joint held-grasp reset state without creating one."""

    artifact = _mapping(artifact, "held-grasp artifact")
    required = {
        "schema_version",
        "task_id",
        "provenance",
        "selection",
        "units",
        "joint_names",
        "joint_units",
        "state",
    }
    _exact_keys(artifact, required, "held-grasp artifact")
    if artifact["schema_version"] != 1 or artifact["task_id"] != "rj45-v1":
        raise ValueError("held-grasp artifact must use schema 1 for rj45-v1")
    provenance = _mapping(artifact["provenance"], "provenance")
    required_provenance = {
        "nominal_report_sha256",
        "nominal_report_path",
        "nominal_trace_sha256",
        "nominal_trace_path",
        "runtime_config_sha256",
        "task_config_sha256",
        "source_and_config_sha256",
    }
    if set(provenance) not in (
        required_provenance,
        required_provenance | {"grasp_generation"},
    ):
        raise ValueError("provenance has unexpected or missing keys")
    for name in (
        "nominal_report_sha256",
        "nominal_trace_sha256",
        "runtime_config_sha256",
        "task_config_sha256",
    ):
        _sha256(provenance[name], f"provenance.{name}")
    for name in ("nominal_report_path", "nominal_trace_path"):
        path = provenance[name]
        if (
            not isinstance(path, str)
            or not path
            or path.startswith("/")
            or ".." in Path(path).parts
        ):
            raise ValueError(f"provenance.{name} must be a project-relative path")
    sources = _mapping(provenance["source_and_config_sha256"], "provenance.source_and_config_sha256")
    if not sources:
        raise ValueError("provenance.source_and_config_sha256 must not be empty")
    for path, digest in sources.items():
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("source_and_config_sha256 keys must be project-relative paths")
        _sha256(digest, f"provenance.source_and_config_sha256[{path}]")
    if "grasp_generation" in provenance:
        generation = _mapping(provenance["grasp_generation"], "provenance.grasp_generation")
        _exact_keys(
            generation,
            {
                "catalog_id",
                "catalog_path",
                "catalog_sha256",
                "generation_seed",
                "catalog_entry_id",
                "assignment_group",
                "offset_method",
            },
            "provenance.grasp_generation",
        )
        for name in ("catalog_id", "catalog_entry_id", "assignment_group", "offset_method"):
            if not isinstance(generation[name], str) or not generation[name]:
                raise ValueError(f"provenance.grasp_generation.{name} must be non-empty")
        path = generation["catalog_path"]
        if not isinstance(path, str) or not path or path.startswith("/") or ".." in Path(path).parts:
            raise ValueError("provenance.grasp_generation.catalog_path must be project-relative")
        _sha256(generation["catalog_sha256"], "provenance.grasp_generation.catalog_sha256")
        if sources.get(path) != generation["catalog_sha256"]:
            raise ValueError(
                "provenance grasp catalog must be bound in source_and_config_sha256"
            )
        if type(generation["generation_seed"]) is not int or generation["generation_seed"] < 0:
            raise ValueError("provenance.grasp_generation.generation_seed must be non-negative")

    selection = _mapping(artifact["selection"], "selection")
    _exact_keys(selection, {"control_step", "timestamp_s", "phase_name"}, "selection")
    if type(selection["control_step"]) is not int or selection["control_step"] <= 0:
        raise ValueError("selection.control_step must be a positive integer")
    _finite_number(selection["timestamp_s"], "selection.timestamp_s", nonnegative=True)
    if selection["phase_name"] != "insert":
        raise ValueError("selection.phase_name must be insert")

    units = _mapping(artifact["units"], "units")
    expected_units = {
        "joint_position": "per_joint_units",
        "joint_velocity": "per_joint_units/s",
        "joint_reference": "per_joint_units",
        "object_pose": "m,wxyz",
        "object_com_velocity": "m/s,rad/s",
        "end_effector_pose": "m,wxyz",
    }
    if units != expected_units:
        raise ValueError(f"units must equal {expected_units}")
    joint_names = artifact["joint_names"]
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 62
        or any(not isinstance(name, str) or not name for name in joint_names)
        or len(set(joint_names)) != 62
    ):
        raise ValueError("joint_names must contain 62 unique non-empty names")
    joint_units = artifact["joint_units"]
    if not isinstance(joint_units, list) or len(joint_units) != 62:
        raise ValueError("joint_units must contain one unit for each of the 62 joints")
    expected_joint_units = [
        "m" if name.startswith("gripper_finger_") else "rad" for name in joint_names
    ]
    if joint_units != expected_joint_units:
        raise ValueError(
            "joint_units must use m for gripper_finger_* joints and rad for all other joints"
        )
    state = _mapping(artifact["state"], "state")
    _exact_keys(
        state,
        {
            "joint_positions_si",
            "joint_velocities_si",
            "joint_reference_si",
            "object_poses_world_wxyz",
            "object_com_velocities_world",
            "end_effector_poses_world_wxyz",
        },
        "state",
    )
    for name in ("joint_positions_si", "joint_velocities_si", "joint_reference_si"):
        _finite_vector(state[name], 62, f"state.{name}")
    object_poses = _finite_matrix(state["object_poses_world_wxyz"], 2, 7, "state.object_poses_world_wxyz")
    _finite_matrix(state["object_com_velocities_world"], 2, 6, "state.object_com_velocities_world")
    ee_poses = _finite_matrix(state["end_effector_poses_world_wxyz"], 2, 7, "state.end_effector_poses_world_wxyz")
    for name, poses in (("object_poses_world_wxyz", object_poses), ("end_effector_poses_world_wxyz", ee_poses)):
        for index, pose in enumerate(poses):
            norm = math.sqrt(sum(component * component for component in pose[3:]))
            if abs(norm - 1.0) > 1.0e-4:
                raise ValueError(f"state.{name}[{index}] quaternion must be unit length")
    return deepcopy(artifact)
