"""Fail-closed packing for filtered PhysX contact manifolds.

The native contact tensor API stores each sensor/filter row in a shared flat
buffer.  This module converts the one-source-body, one-target-filter row used
by each environment into a fixed-width tensor representation suitable for
runtime snapshots and JSON recording.
"""

from __future__ import annotations

import torch


_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def pack_filtered_contact_manifold(
    forces: torch.Tensor,
    points: torch.Tensor,
    normals: torch.Tensor,
    separations: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    num_envs: int,
    capacity: int = 128,
) -> dict[str, torch.Tensor]:
    """Pack one filtered contact row per environment into padded tensors.

    Rows with a zero contact count ignore their undefined ``starts`` value.
    Only active native-buffer entries must be finite; unused entries are never
    copied and therefore become zeros in the returned tensors.
    """

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")

    tensors = (forces, points, normals, separations, counts, starts)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("all contact manifold inputs must be torch tensors")
    if not all(value.is_floating_point() for value in tensors[:4]):
        raise TypeError("forces, points, normals, and separations must use floating-point dtypes")
    if counts.dtype not in _INTEGER_DTYPES or starts.dtype not in _INTEGER_DTYPES:
        raise TypeError("counts and starts must use integer dtypes")
    if any(value.device != forces.device for value in tensors[1:]):
        raise ValueError("all contact manifold inputs must be on the same device")

    if forces.ndim == 1:
        flat_forces = forces
    elif forces.ndim == 2 and forces.shape[1] == 1:
        flat_forces = forces[:, 0]
    else:
        raise ValueError("forces must have shape (P,) or (P, 1)")
    if separations.ndim == 1:
        flat_separations = separations
    elif separations.ndim == 2 and separations.shape[1] == 1:
        flat_separations = separations[:, 0]
    else:
        raise ValueError("separations must have shape (P,) or (P, 1)")

    buffer_size = flat_forces.shape[0]
    if points.ndim != 2 or points.shape != (buffer_size, 3):
        raise ValueError("points must have shape (P, 3) matching forces")
    if normals.ndim != 2 or normals.shape != (buffer_size, 3):
        raise ValueError("normals must have shape (P, 3) matching forces")
    if flat_separations.shape[0] != buffer_size:
        raise ValueError("separations must have the same buffer length as forces")
    if counts.shape != (num_envs, 1) or starts.shape != (num_envs, 1):
        raise ValueError("counts and starts must each have shape (num_envs, 1)")
    contact_count = counts[:, 0].clone()
    count_indices = contact_count.to(dtype=torch.long)
    start_indices = starts[:, 0].to(dtype=torch.long)
    if torch.any(count_indices < 0):
        raise ValueError("contact counts must be non-negative")
    if torch.any(count_indices > capacity):
        raise ValueError("contact count exceeds output capacity")

    offsets = torch.arange(capacity, device=forces.device, dtype=torch.long)
    active = offsets.unsqueeze(0) < count_indices.unsqueeze(1)
    candidate_indices = start_indices.unsqueeze(1) + offsets.unsqueeze(0)
    active_indices = candidate_indices[active]
    if torch.any(active_indices < 0) or torch.any(active_indices >= buffer_size):
        raise ValueError("active contact range is outside the native buffer")

    force_n = torch.zeros(
        (num_envs, capacity), device=forces.device, dtype=flat_forces.dtype
    )
    points_world_m = torch.zeros(
        (num_envs, capacity, 3), device=points.device, dtype=points.dtype
    )
    normals_world = torch.zeros(
        (num_envs, capacity, 3), device=normals.device, dtype=normals.dtype
    )
    separation_m = torch.zeros(
        (num_envs, capacity),
        device=separations.device,
        dtype=flat_separations.dtype,
    )

    if active_indices.numel() > 0:
        gather_indices = active_indices.to(dtype=torch.long)
        active_forces = flat_forces.index_select(0, gather_indices)
        active_points = points.index_select(0, gather_indices)
        active_normals = normals.index_select(0, gather_indices)
        active_separations = flat_separations.index_select(0, gather_indices)

        active_fields = (
            active_forces,
            active_points,
            active_normals,
            active_separations,
        )
        if not all(torch.isfinite(value).all() for value in active_fields):
            raise ValueError("active contact manifold values must be finite")
        normal_lengths = torch.linalg.vector_norm(active_normals, dim=1)
        if torch.any(torch.abs(normal_lengths - 1.0) > 1.0e-3):
            raise ValueError("active contact normals must be unit length within 1e-3")

        force_n[active] = active_forces
        points_world_m[active] = active_points
        normals_world[active] = active_normals
        separation_m[active] = active_separations

    return {
        "contact_count": contact_count,
        "active": active,
        "force_n": force_n,
        "points_world_m": points_world_m,
        "normals_world": normals_world,
        "separation_m": separation_m,
    }


__all__ = ["pack_filtered_contact_manifold"]
