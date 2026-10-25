"""Tensor operations for the differential joint controller."""

import math

import torch


def continuous_goal_dwell(
    instantaneous_ok: torch.Tensor,
    current_step: torch.Tensor,
    previous_start_step: torch.Tensor,
    *,
    step_dt_s: float,
    required_dwell_s: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure full control intervals since the first qualifying sample.

    Integer episode steps avoid subtracting large float32 absolute times.
    The 1 ps rounding allowance matches the independent TaskEvaluator.
    """
    if not all(math.isfinite(value) and value > 0 for value in (step_dt_s, required_dwell_s)):
        raise ValueError("step_dt_s and required_dwell_s must be finite and positive")
    if not (instantaneous_ok.shape == current_step.shape == previous_start_step.shape):
        raise ValueError("dwell inputs must have the same shape")
    if current_step.dtype != torch.long or previous_start_step.dtype != torch.long:
        raise ValueError("dwell steps must be int64")
    start_step = torch.where(
        instantaneous_ok,
        torch.where(previous_start_step < 0, current_step, previous_start_step),
        torch.full_like(previous_start_step, -1),
    )
    elapsed_steps = torch.where(
        instantaneous_ok, (current_step - start_step).clamp_min(0), torch.zeros_like(current_step)
    )
    required_steps = max(1, math.ceil((required_dwell_s - 1e-12) / step_dt_s))
    dwell_s = elapsed_steps.to(torch.float64) * step_dt_s
    return start_step, dwell_s, instantaneous_ok & (elapsed_steps >= required_steps)


def continuous_grasp_loss(
    active: torch.Tensor,
    grasp_live: torch.Tensor,
    previous_since: torch.Tensor,
    now_s: float,
    grace_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Track uninterrupted grasp loss and flag environments past the grace period."""
    if grace_s <= 0:
        raise ValueError("grace_s must be positive")
    if active.shape != grasp_live.shape or active.shape != previous_since.shape:
        raise ValueError("active, grasp_live, and previous_since must have the same shape")

    missing = active & ~grasp_live
    first_missing = missing & (previous_since < 0.0)
    new_since = torch.where(
        missing,
        torch.where(first_missing, torch.full_like(previous_since, now_s), previous_since),
        torch.full_like(previous_since, -1.0),
    )
    expired = missing & ((now_s - new_since) >= grace_s)
    return new_since, expired


def damped_joint_step(jacobian, twist, damping: float, max_joint_delta: float):
    """Bound a DLS step with a common scale, preserving its joint-space direction."""
    if damping <= 0 or max_joint_delta <= 0:
        raise ValueError("damping and max_joint_delta must be positive")
    identity = torch.eye(jacobian.shape[1], device=jacobian.device, dtype=jacobian.dtype)
    lhs = jacobian @ jacobian.transpose(1, 2) + damping**2 * identity
    delta = (jacobian.transpose(1, 2) @ torch.linalg.solve(lhs, twist.unsqueeze(-1))).squeeze(-1)
    scale = (delta.abs().amax(dim=1, keepdim=True) / max_joint_delta).clamp_min(1.0)
    return delta / scale
