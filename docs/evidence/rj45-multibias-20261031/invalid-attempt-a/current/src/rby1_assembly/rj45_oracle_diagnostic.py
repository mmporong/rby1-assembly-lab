"""Privileged-observation diagnostic adapter for the frozen RJ45 runtime.

This module intentionally has no Isaac Lab imports.  The factory accepts an
already configured training environment type so an archived runtime can load
this file with ``importlib`` without changing its reward, action, reset, or
success contracts.
"""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping


BASE_POLICY_DIM = 79
ORACLE_STATE_DIM = 13
ORACLE_POLICY_DIM = BASE_POLICY_DIM + ORACLE_STATE_DIM

STANDARD_ACTOR = "standard79"
ORACLE_ACTOR = "oracle92"

DEFAULT_ORACLE_SCALES = MappingProxyType(
    {
        "relative_position_m": 0.05,
        "relative_linear_velocity_m_s": 0.25,
        "relative_angular_velocity_rad_s": 2.0,
    }
)

ORACLE_FIELD_METADATA = MappingProxyType(
    {
        "frame": "measured_socket_entry",
        "pose_source": "simulator_ground_truth_socket_and_plug_link_pose",
        "velocity_source": "simulator_ground_truth_root_link_velocity",
        "quaternion_order": "wxyz",
        "field_order": (
            "relative_position_entry_xyz",
            "relative_quaternion_entry_wxyz",
            "relative_linear_velocity_entry_xyz",
            "relative_angular_velocity_entry_xyz",
        ),
        "field_widths": (3, 4, 3, 3),
        "normalization": "fixed_unit_scales_not_data_dependent",
    }
)


def observation_space_spec(actor_mode: str) -> dict[str, int]:
    """Return the asymmetric actor/critic dimensions for a runner adapter."""

    _validate_actor_mode(actor_mode)
    return {
        "policy": BASE_POLICY_DIM if actor_mode == STANDARD_ACTOR else ORACLE_POLICY_DIM,
        "critic": ORACLE_POLICY_DIM,
    }


def oracle_observation_metadata(
    actor_mode: str,
    scales: Mapping[str, float] = DEFAULT_ORACLE_SCALES,
) -> dict[str, Any]:
    """Describe the fixed, auditable oracle observation contract."""

    _validate_actor_mode(actor_mode)
    checked_scales = _validate_scales(scales)
    return {
        **dict(ORACLE_FIELD_METADATA),
        "actor_mode": actor_mode,
        "dimensions": observation_space_spec(actor_mode),
        "scales": checked_scales,
        "actor_ground_truth_access": actor_mode == ORACLE_ACTOR,
        "critic_ground_truth_access": True,
        "inactive_or_warmup_rows": "all_zeros_for_base_and_oracle_fields",
    }


def _validate_actor_mode(actor_mode: str) -> None:
    if actor_mode not in (STANDARD_ACTOR, ORACLE_ACTOR):
        raise ValueError(f"actor_mode must be {STANDARD_ACTOR!r} or {ORACLE_ACTOR!r}")


def _validate_scales(scales: Mapping[str, float]) -> dict[str, float]:
    expected = set(DEFAULT_ORACLE_SCALES)
    if set(scales) != expected:
        raise ValueError(f"oracle scales must contain exactly {sorted(expected)}")
    checked: dict[str, float] = {}
    for name in DEFAULT_ORACLE_SCALES:
        value = float(scales[name])
        if not math.isfinite(value) or not value > 0.0:
            raise ValueError(f"oracle scale {name} must be finite and positive")
        checked[name] = value
    return checked


def _quaternion_multiply_wxyz(left: Any, right: Any) -> Any:
    import torch

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


def _quaternion_rotate_wxyz(quaternion: Any, vector: Any) -> Any:
    import torch

    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    return (
        vector * (2.0 * scalar.square() - 1.0)
        + 2.0 * scalar * torch.linalg.cross(xyz, vector, dim=-1)
        + 2.0 * xyz * torch.sum(xyz * vector, dim=-1, keepdim=True)
    )


def _inverse_unit_quaternion_wxyz(quaternion: Any) -> Any:
    inverse = quaternion.clone()
    inverse[..., 1:] *= -1.0
    return inverse


def _validate_tensor(name: str, value: Any, width: int, batch_shape: tuple[int, ...]) -> None:
    import torch

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != batch_shape + (width,):
        raise ValueError(f"{name} must have shape {batch_shape + (width,)}")


def build_oracle_state(
    socket_position_world: Any,
    socket_quaternion_world_wxyz: Any,
    socket_linear_velocity_world: Any,
    socket_angular_velocity_world: Any,
    plug_position_world: Any,
    plug_quaternion_world_wxyz: Any,
    plug_linear_velocity_world: Any,
    plug_angular_velocity_world: Any,
    socket_entry_position_body: Any,
    socket_entry_quaternion_body_wxyz: Any,
    plug_tip_position_body: Any,
    plug_tip_quaternion_body_wxyz: Any,
    *,
    scales: Mapping[str, float] = DEFAULT_ORACLE_SCALES,
) -> Any:
    """Return normalized true mating pose and twist in the socket-entry frame.

    The linear twist is the derivative of the tip-to-entry displacement in the
    rotating socket-entry frame.  It therefore includes both frame-point lever
    arms and the socket angular-velocity transport term.
    """

    import torch

    if not isinstance(socket_position_world, torch.Tensor) or socket_position_world.ndim < 2:
        raise ValueError("socket_position_world must have at least one batch dimension")
    batch_shape = tuple(socket_position_world.shape[:-1])
    tensors = (
        ("socket_position_world", socket_position_world, 3),
        ("socket_quaternion_world_wxyz", socket_quaternion_world_wxyz, 4),
        ("socket_linear_velocity_world", socket_linear_velocity_world, 3),
        ("socket_angular_velocity_world", socket_angular_velocity_world, 3),
        ("plug_position_world", plug_position_world, 3),
        ("plug_quaternion_world_wxyz", plug_quaternion_world_wxyz, 4),
        ("plug_linear_velocity_world", plug_linear_velocity_world, 3),
        ("plug_angular_velocity_world", plug_angular_velocity_world, 3),
        ("socket_entry_position_body", socket_entry_position_body, 3),
        ("socket_entry_quaternion_body_wxyz", socket_entry_quaternion_body_wxyz, 4),
        ("plug_tip_position_body", plug_tip_position_body, 3),
        ("plug_tip_quaternion_body_wxyz", plug_tip_quaternion_body_wxyz, 4),
    )
    for name, value, width in tensors:
        _validate_tensor(name, value, width, batch_shape)
    reference_dtype = socket_position_world.dtype
    reference_device = socket_position_world.device
    if not socket_position_world.is_floating_point():
        raise TypeError("oracle state tensors must use a floating-point dtype")
    for name, value, _ in tensors[1:]:
        if not value.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if value.dtype != reference_dtype:
            raise TypeError("oracle state tensors must use one common dtype")
        if value.device != reference_device:
            raise ValueError("oracle state tensors must use one common device")
    for name, value, _ in tensors:
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")

    for name, quaternion in (
        ("socket_quaternion_world_wxyz", socket_quaternion_world_wxyz),
        ("plug_quaternion_world_wxyz", plug_quaternion_world_wxyz),
        ("socket_entry_quaternion_body_wxyz", socket_entry_quaternion_body_wxyz),
        ("plug_tip_quaternion_body_wxyz", plug_tip_quaternion_body_wxyz),
    ):
        norm_error = (torch.linalg.vector_norm(quaternion, dim=-1) - 1.0).abs()
        if bool(torch.any(norm_error > 1.0e-5)):
            raise ValueError(f"{name} must be unit length")

    checked_scales = _validate_scales(scales)
    entry_offset_world = _quaternion_rotate_wxyz(
        socket_quaternion_world_wxyz, socket_entry_position_body
    )
    tip_offset_world = _quaternion_rotate_wxyz(
        plug_quaternion_world_wxyz, plug_tip_position_body
    )
    entry_position_world = socket_position_world + entry_offset_world
    tip_position_world = plug_position_world + tip_offset_world
    entry_quaternion_world = _quaternion_multiply_wxyz(
        socket_quaternion_world_wxyz, socket_entry_quaternion_body_wxyz
    )
    tip_quaternion_world = _quaternion_multiply_wxyz(
        plug_quaternion_world_wxyz, plug_tip_quaternion_body_wxyz
    )
    world_to_entry = _inverse_unit_quaternion_wxyz(entry_quaternion_world)

    relative_position_entry = _quaternion_rotate_wxyz(
        world_to_entry, tip_position_world - entry_position_world
    )
    relative_quaternion_entry = _quaternion_multiply_wxyz(
        world_to_entry, tip_quaternion_world
    )
    # q and -q encode the same rotation.  A stable sign prevents an artificial
    # discontinuity in the privileged input while retaining wxyz ordering.
    relative_quaternion_entry = torch.where(
        (relative_quaternion_entry[..., :1] < 0.0),
        -relative_quaternion_entry,
        relative_quaternion_entry,
    )

    entry_velocity_world = socket_linear_velocity_world + torch.linalg.cross(
        socket_angular_velocity_world, entry_offset_world, dim=-1
    )
    tip_velocity_world = plug_linear_velocity_world + torch.linalg.cross(
        plug_angular_velocity_world, tip_offset_world, dim=-1
    )
    displacement_world = tip_position_world - entry_position_world
    relative_linear_velocity_world = (
        tip_velocity_world
        - entry_velocity_world
        - torch.linalg.cross(
            socket_angular_velocity_world, displacement_world, dim=-1
        )
    )
    relative_angular_velocity_world = (
        plug_angular_velocity_world - socket_angular_velocity_world
    )
    relative_linear_velocity_entry = _quaternion_rotate_wxyz(
        world_to_entry, relative_linear_velocity_world
    )
    relative_angular_velocity_entry = _quaternion_rotate_wxyz(
        world_to_entry, relative_angular_velocity_world
    )

    oracle = torch.cat(
        (
            relative_position_entry / checked_scales["relative_position_m"],
            relative_quaternion_entry,
            relative_linear_velocity_entry
            / checked_scales["relative_linear_velocity_m_s"],
            relative_angular_velocity_entry
            / checked_scales["relative_angular_velocity_rad_s"],
        ),
        dim=-1,
    )
    if oracle.shape != batch_shape + (ORACLE_STATE_DIM,) or not bool(
        torch.isfinite(oracle).all()
    ):
        raise RuntimeError("oracle state violated its finite 13D contract")
    return oracle


def compose_actor_critic_observations(
    base_policy_observation: Any,
    oracle_state: Any,
    valid_mask: Any,
    actor_mode: str,
) -> dict[str, Any]:
    """Build actor/critic observations without leaking oracle state to standard79."""

    import torch

    _validate_actor_mode(actor_mode)
    if not isinstance(base_policy_observation, torch.Tensor):
        raise TypeError("base policy observation must be a torch.Tensor")
    if not isinstance(oracle_state, torch.Tensor):
        raise TypeError("oracle state must be a torch.Tensor")
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if base_policy_observation.ndim != 2 or base_policy_observation.shape[1] != BASE_POLICY_DIM:
        raise ValueError("base policy observation must have shape (num_envs, 79)")
    if oracle_state.shape != (base_policy_observation.shape[0], ORACLE_STATE_DIM):
        raise ValueError("oracle state must have shape (num_envs, 13)")
    if valid_mask.shape != (base_policy_observation.shape[0],) or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be boolean with shape (num_envs,)")
    if not base_policy_observation.is_floating_point() or not oracle_state.is_floating_point():
        raise TypeError("base and oracle observations must use floating-point dtypes")
    if base_policy_observation.dtype != oracle_state.dtype:
        raise TypeError("base and oracle observations must use the same dtype")
    if base_policy_observation.device != oracle_state.device:
        raise ValueError("base and oracle observations must use the same device")
    if valid_mask.device != base_policy_observation.device:
        raise ValueError("valid_mask must use the observation device")
    if not bool(torch.isfinite(base_policy_observation).all()) or not bool(
        torch.isfinite(oracle_state).all()
    ):
        raise ValueError("actor and oracle observations must be finite")

    masked_base = torch.where(valid_mask[:, None], base_policy_observation, 0.0)
    masked_oracle = torch.where(valid_mask[:, None], oracle_state, 0.0)
    critic = torch.cat((masked_base, masked_oracle), dim=1)
    policy = critic if actor_mode == ORACLE_ACTOR else masked_base
    return {"policy": policy, "critic": critic}


def make_rj45_oracle_diagnostic_env_type(
    base_training_cls: type,
    *,
    actor_mode: str,
    scales: Mapping[str, float] = DEFAULT_ORACLE_SCALES,
) -> type:
    """Wrap a frozen TrainingEnv with asymmetric diagnostic observations.

    ``base_training_cls`` remains responsible for all physics, actions, reward,
    termination, and reset behavior.  The returned class only reads the true
    object state after the base observation hook has completed.

    The caller must configure its vector/RL wrapper from
    :func:`observation_space_spec`; the archived TrainingEnv fixes its internal
    configuration to 79 before Isaac creates its Gym space.
    """

    _validate_actor_mode(actor_mode)
    checked_scales = _validate_scales(scales)
    metadata = oracle_observation_metadata(actor_mode, checked_scales)

    class Rby1Rj45OracleDiagnosticEnv(base_training_cls):
        oracle_observation_space_spec = observation_space_spec(actor_mode)
        oracle_observation_metadata = metadata

        def _get_observations(self):
            base = super()._get_observations()
            # Base constructors may query observations before training state and
            # object tensors are ready.  Preserve that initialization path.
            if not getattr(self, "_training_initialized", False):
                return base
            if not isinstance(base, dict) or set(base) != {"policy"}:
                raise RuntimeError("oracle diagnostic requires base {'policy': 79D} observations")
            policy = base["policy"]
            if policy.shape != (self.num_envs, BASE_POLICY_DIM):
                raise RuntimeError("oracle diagnostic base policy shape mismatch")

            entry_position, entry_quaternion = self._frame_tensors("socket_entry")
            tip_position, tip_quaternion = self._frame_tensors("plug_tip")
            oracle = build_oracle_state(
                self.socket.data.root_pos_w,
                self.socket.data.root_quat_w,
                self.socket.data.root_link_lin_vel_w,
                self.socket.data.root_link_ang_vel_w,
                self.plug.data.root_pos_w,
                self.plug.data.root_quat_w,
                self.plug.data.root_link_lin_vel_w,
                self.plug.data.root_link_ang_vel_w,
                entry_position,
                entry_quaternion,
                tip_position,
                tip_quaternion,
                scales=checked_scales,
            )
            valid = self.extras.get("policy_observation_valid")
            if valid is None:
                raise RuntimeError("base environment did not publish policy_observation_valid")
            return compose_actor_critic_observations(policy, oracle, valid, actor_mode)

    Rby1Rj45OracleDiagnosticEnv.__name__ = (
        f"Rby1Rj45OracleDiagnosticEnv_{actor_mode}"
    )
    return Rby1Rj45OracleDiagnosticEnv


__all__ = [
    "BASE_POLICY_DIM",
    "DEFAULT_ORACLE_SCALES",
    "ORACLE_ACTOR",
    "ORACLE_POLICY_DIM",
    "ORACLE_STATE_DIM",
    "STANDARD_ACTOR",
    "build_oracle_state",
    "compose_actor_critic_observations",
    "make_rj45_oracle_diagnostic_env_type",
    "observation_space_spec",
    "oracle_observation_metadata",
]
