"""Privileged, bounded SE(3) teacher targets for RJ45 demonstrations.

The teacher compares baseline and desired target poses at the same command
depth.  Its output is a dataset label only: it does not consume actual plug
tracking error and must not be added to the student observation.
"""

from __future__ import annotations

import math
from numbers import Real
from typing import Any

from .rj45_training import (
    _quaternion_inverse_wxyz,
    _quaternion_multiply_wxyz,
    _quaternion_rotate_wxyz,
)


def _torch():
    import torch

    return torch


def _pose_tensor(value: Any, size: int, name: str):
    torch = _torch()
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.ndim < 1 or value.shape[-1] != size:
        raise ValueError(f"{name} must have last dimension {size}")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _positive_bound(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _unit_quaternion(value: Any, name: str):
    torch = _torch()
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    if bool((torch.abs(norm - 1.0) > 1.0e-4).any()):
        raise ValueError(f"{name} must be unit length")
    return value / norm


def _canonical_quaternion(quaternion: Any):
    """Choose one sign for q/-q, including the exactly-pi boundary."""

    torch = _torch()
    scalar = quaternion[..., :1]
    vector = quaternion[..., 1:]
    largest_index = torch.argmax(torch.abs(vector), dim=-1, keepdim=True)
    largest_component = torch.gather(vector, -1, largest_index)
    sign_source = torch.where(torch.abs(scalar) > 1.0e-8, scalar, largest_component)
    sign = torch.where(
        sign_source < 0.0,
        -torch.ones_like(sign_source),
        torch.ones_like(sign_source),
    )
    return quaternion * sign


def _principal_rotation_vector(quaternion: Any):
    torch = _torch()
    canonical = _canonical_quaternion(quaternion)
    vector = canonical[..., 1:]
    vector_norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(vector_norm, torch.clamp(canonical[..., :1], min=0.0))
    scale = torch.where(
        vector_norm > 1.0e-8,
        angle / vector_norm,
        2.0 + vector_norm.square() / 3.0,
    )
    return vector * scale


def teacher_tip_target_action(
    baseline_body_position_world: Any,
    baseline_body_quaternion_world_wxyz: Any,
    desired_body_position_world: Any,
    desired_body_quaternion_world_wxyz: Any,
    socket_entry_quaternion_world_wxyz: Any,
    plug_tip_position_local: Any,
    plug_tip_quaternion_local_wxyz: Any,
    translation_bound_m: Any,
    rotation_bound_rad: Any,
    active_mask: Any = None,
) -> dict[str, Any]:
    """Return an absolute, bounded tip-target action derived from target poses.

    Baseline and desired poses must describe target commands at the same
    command depth.  The returned ground-truth-derived values are labels and
    diagnostics, not student observations.
    """

    torch = _torch()
    baseline_position = _pose_tensor(
        baseline_body_position_world, 3, "baseline_body_position_world"
    )
    baseline_quaternion = _pose_tensor(
        baseline_body_quaternion_world_wxyz,
        4,
        "baseline_body_quaternion_world_wxyz",
    )
    desired_position = _pose_tensor(
        desired_body_position_world, 3, "desired_body_position_world"
    )
    desired_quaternion = _pose_tensor(
        desired_body_quaternion_world_wxyz,
        4,
        "desired_body_quaternion_world_wxyz",
    )
    entry_quaternion = _pose_tensor(
        socket_entry_quaternion_world_wxyz,
        4,
        "socket_entry_quaternion_world_wxyz",
    )
    tip_position_local = _pose_tensor(
        plug_tip_position_local, 3, "plug_tip_position_local"
    )
    tip_quaternion_local = _pose_tensor(
        plug_tip_quaternion_local_wxyz, 4, "plug_tip_quaternion_local_wxyz"
    )
    batch_shape = baseline_position.shape[:-1]
    tensors = (
        baseline_position,
        baseline_quaternion,
        desired_position,
        desired_quaternion,
        entry_quaternion,
        tip_position_local,
        tip_quaternion_local,
    )
    if any(value.shape[:-1] != batch_shape for value in tensors):
        raise ValueError("teacher pose inputs must have matching batch shapes")
    if any(
        value.device != baseline_position.device or value.dtype != baseline_position.dtype
        for value in tensors
    ):
        raise ValueError("teacher pose inputs must have matching device and dtype")

    baseline_quaternion = _unit_quaternion(
        baseline_quaternion, "baseline_body_quaternion_world_wxyz"
    )
    desired_quaternion = _unit_quaternion(
        desired_quaternion, "desired_body_quaternion_world_wxyz"
    )
    entry_quaternion = _unit_quaternion(
        entry_quaternion, "socket_entry_quaternion_world_wxyz"
    )
    tip_quaternion_local = _unit_quaternion(
        tip_quaternion_local, "plug_tip_quaternion_local_wxyz"
    )
    translation_bound = _positive_bound(translation_bound_m, "translation_bound_m")
    rotation_bound = _positive_bound(rotation_bound_rad, "rotation_bound_rad")

    if active_mask is None:
        active = torch.ones(batch_shape, dtype=torch.bool, device=baseline_position.device)
    else:
        if not isinstance(active_mask, torch.Tensor):
            raise TypeError("active_mask must be a torch.Tensor")
        if active_mask.dtype != torch.bool or active_mask.shape != batch_shape:
            raise ValueError("active_mask must be boolean with the teacher batch shape")
        if active_mask.device != baseline_position.device:
            raise ValueError("active_mask must be on the teacher tensor device")
        active = active_mask

    baseline_tip_position = baseline_position + _quaternion_rotate_wxyz(
        baseline_quaternion, tip_position_local
    )
    desired_tip_position = desired_position + _quaternion_rotate_wxyz(
        desired_quaternion, tip_position_local
    )
    baseline_tip_quaternion = _quaternion_multiply_wxyz(
        baseline_quaternion, tip_quaternion_local
    )
    desired_tip_quaternion = _quaternion_multiply_wxyz(
        desired_quaternion, tip_quaternion_local
    )
    entry_inverse = _quaternion_inverse_wxyz(entry_quaternion)
    translation_entry = _quaternion_rotate_wxyz(
        entry_inverse, desired_tip_position - baseline_tip_position
    )
    rotation_delta_entry = _quaternion_multiply_wxyz(
        _quaternion_multiply_wxyz(
            _quaternion_multiply_wxyz(entry_inverse, desired_tip_quaternion),
            _quaternion_inverse_wxyz(baseline_tip_quaternion),
        ),
        entry_quaternion,
    )
    rotation_delta_entry = rotation_delta_entry / torch.linalg.vector_norm(
        rotation_delta_entry, dim=-1, keepdim=True
    )
    rotation_vector_entry = _principal_rotation_vector(rotation_delta_entry)

    applied_offset = torch.cat((translation_entry, rotation_vector_entry), dim=-1)
    raw_action = torch.cat(
        (
            applied_offset[..., :3] / translation_bound,
            applied_offset[..., 3:] / rotation_bound,
        ),
        dim=-1,
    )
    component_clipped = torch.clamp(raw_action, -1.0, 1.0)
    component_bounded = component_clipped.reshape(batch_shape + (2, 3))
    unbounded_norms = torch.linalg.vector_norm(component_bounded, dim=-1, keepdim=True)
    norms = unbounded_norms.clamp_min(1.0)
    bounded_action = (component_bounded / norms).reshape(batch_shape + (6,))
    applied_offset = torch.cat(
        (
            bounded_action[..., :3] * translation_bound,
            bounded_action[..., 3:] * rotation_bound,
        ),
        dim=-1,
    )
    limited = torch.any(torch.abs(raw_action) > 1.0, dim=-1) | torch.any(
        unbounded_norms.squeeze(-1) > 1.0, dim=-1
    )

    active_column = active.unsqueeze(-1)
    return {
        "raw_action_6d": torch.where(active_column, raw_action, torch.zeros_like(raw_action)),
        "bounded_action_6d": torch.where(
            active_column, bounded_action, torch.zeros_like(bounded_action)
        ),
        "applied_offset_entry_6d": torch.where(
            active_column, applied_offset, torch.zeros_like(applied_offset)
        ),
        "bound_limited_mask": limited & active,
    }
