"""Pure training contracts and tensor math for RJ45 residual control.

The policy observation accepts an estimated relative connector pose. Ground-truth
relative pose is intentionally confined to the reward/evaluation caller.
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
_TOP_LEVEL_KEYS = {
    "schema_version",
    "task_id",
    "episode_scope",
    "active_horizon_s",
    "warmup_s",
    "action_mode",
    "observation_dim",
    "calibration_bias",
    "reward",
    "task_physics",
    "held_reset_staging",
}
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
_PHYSICS_KEYS = {"mass_randomization_enabled"}
_OBSERVATION_DIM = 79
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
    _exact_keys(config, _TOP_LEVEL_KEYS, "training configuration")
    if config["schema_version"] != 1:
        raise ValueError("schema_version must be 1")
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
    if type(config["observation_dim"]) is not int or config["observation_dim"] != _OBSERVATION_DIM:
        raise ValueError(f"observation_dim must be {_OBSERVATION_DIM}")

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
    """Potential shaping while active, plus one-shot terminal outcome terms."""

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
    _exact_keys(
        provenance,
        {
            "nominal_report_sha256",
            "nominal_report_path",
            "nominal_trace_sha256",
            "nominal_trace_path",
            "runtime_config_sha256",
            "task_config_sha256",
            "source_and_config_sha256",
        },
        "provenance",
    )
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
