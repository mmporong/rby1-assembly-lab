"""Contact-buffer aggregation used by the Isaac PhysX runtime.

``RigidContactView.get_contact_data`` stores contact forces in a bounded flat
buffer.  ``counts`` and ``starts`` describe the active slice for each flattened
sensor/filter row.  Summing the absolute per-contact normal forces avoids the
cancellation that occurs when a plug is wedged between opposing socket walls.
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


def sum_contact_normal_loads(
    forces: torch.Tensor,
    counts: torch.Tensor,
    starts: torch.Tensor,
    num_envs: int,
) -> torch.Tensor:
    """Return the sum of absolute normal contact forces for each environment.

    The function fails closed when the PhysX contact buffer is malformed or
    completely full.  A full buffer is treated as saturated because additional
    contacts may have been truncated.  ``starts`` values belonging to rows with
    a zero count are intentionally ignored because PhysX does not define them.
    """

    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    if not all(isinstance(value, torch.Tensor) for value in (forces, counts, starts)):
        raise TypeError("forces, counts, and starts must be torch tensors")
    if not forces.is_floating_point():
        raise TypeError("forces must use a floating-point dtype")
    if counts.dtype not in _INTEGER_DTYPES or starts.dtype not in _INTEGER_DTYPES:
        raise TypeError("counts and starts must use integer dtypes")
    if forces.device != counts.device or counts.device != starts.device:
        raise ValueError("forces, counts, and starts must be on the same device")

    flat_forces = forces.reshape(-1)
    flat_counts = counts.reshape(-1)
    flat_starts = starts.reshape(-1)
    if flat_counts.numel() == 0 or flat_counts.numel() != flat_starts.numel():
        raise ValueError("counts and starts must contain the same non-zero number of rows")
    if flat_counts.numel() % num_envs != 0:
        raise ValueError("contact rows must be divisible by num_envs")
    if flat_forces.numel() == 0:
        raise ValueError("contact force buffer must not be empty")
    if torch.any(flat_counts < 0):
        raise ValueError("contact counts must be non-negative")

    total_contacts = int(flat_counts.sum().item())
    if total_contacts >= flat_forces.numel():
        raise ValueError("contact force buffer is saturated")
    if total_contacts == 0:
        return torch.zeros(num_envs, device=forces.device, dtype=forces.dtype)

    row_ids = torch.repeat_interleave(
        torch.arange(flat_counts.numel(), device=counts.device), flat_counts
    )
    block_starts = flat_counts.cumsum(0) - flat_counts
    offsets = torch.arange(total_contacts, device=counts.device) - block_starts.repeat_interleave(
        flat_counts
    )
    flat_indices = flat_starts[row_ids] + offsets
    if torch.any(flat_indices < 0) or torch.any(flat_indices >= flat_forces.numel()):
        raise ValueError("active contact range is outside the force buffer")

    active_forces = flat_forces.index_select(0, flat_indices.to(dtype=torch.long))
    if not torch.isfinite(active_forces).all():
        raise ValueError("active contact forces must be finite")

    per_row = torch.zeros(flat_counts.numel(), device=forces.device, dtype=forces.dtype)
    per_row.index_add_(0, row_ids.to(dtype=torch.long), active_forces.abs())
    return per_row.reshape(num_envs, -1).sum(dim=1)
