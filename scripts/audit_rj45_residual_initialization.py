#!/usr/bin/env python3
"""Measure untrained PPO/SAC residual distributions without an environment."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rby1_assembly.rj45_comparison_rl import (  # noqa: E402
    ACTION_DIM,
    MAX_LOG_STD,
    MIN_LOG_STD,
    OBSERVATION_DIM,
    deterministic_action,
    make_agent,
)
from rby1_assembly.rj45_training import expand_residual_actions  # noqa: E402
from rby1_assembly.rj45_training_env import _action_saturation_masks  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bound_residual_actions(actions: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Apply the environment's component clip then per-3D-block norm projection."""

    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not torch.isfinite(actions).all():
        raise ValueError("actions must be a finite [N, 6] tensor")
    clipped = torch.clamp(actions, -1.0, 1.0)
    blocks = clipped.reshape(-1, 2, 3)
    norms = torch.linalg.vector_norm(blocks, dim=2, keepdim=True)
    bounded = (blocks / torch.maximum(norms, torch.ones_like(norms))).reshape(-1, ACTION_DIM)
    rates = {
        "component_clip_rate": float((actions.abs() > 1.0).to(torch.float64).mean()),
        "sample_clip_rate": float((actions.abs() > 1.0).any(dim=1).to(torch.float64).mean()),
        "translation_projection_rate": float((norms[:, 0, 0] > 1.0).to(torch.float64).mean()),
        "rotation_projection_rate": float((norms[:, 1, 0] > 1.0).to(torch.float64).mean()),
    }
    expanded = expand_residual_actions(actions, "right_only_6d")
    active = torch.ones(actions.shape[0], dtype=torch.bool)
    clip_boundary, any_projection = _action_saturation_masks(expanded, active)
    rates["environment_sample_clip_boundary_rate"] = float(clip_boundary.to(torch.float64).mean())
    rates["environment_sample_any_projection_rate"] = float(any_projection.to(torch.float64).mean())
    return bounded, rates


def _percentiles(values: torch.Tensor) -> dict[str, float]:
    values = values.to(dtype=torch.float64)
    return {
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
    }


def summarize_actions(
    pre_bound_actions: torch.Tensor,
    *,
    translation_bound_m: float,
    rotation_bound_rad: float,
) -> dict[str, Any]:
    bounded, rates = bound_residual_actions(pre_bound_actions)
    translation_m = bounded[:, :3] * translation_bound_m
    rotation_rad = bounded[:, 3:] * rotation_bound_rad
    axes = []
    for index in range(ACTION_DIM):
        physical = translation_m[:, index] if index < 3 else rotation_rad[:, index - 3]
        scale = translation_bound_m if index < 3 else rotation_bound_rad
        axis = {
            "axis": ("translation_" if index < 3 else "rotation_") + "xyz"[index % 3],
            "absolute_normalized": _percentiles(bounded[:, index].abs()),
            "signed_normalized_median": float(torch.median(bounded[:, index].to(torch.float64))),
        }
        if index < 3:
            axis["absolute_m"] = _percentiles(physical.abs())
            axis["absolute_mm"] = {key: value * 1_000.0 for key, value in axis["absolute_m"].items()}
        else:
            axis["absolute_rad"] = _percentiles(physical.abs())
            axis["absolute_deg"] = {
                key: value * 180.0 / math.pi for key, value in axis["absolute_rad"].items()
            }
        if scale <= 0.0:
            raise ValueError("physical action bounds must be positive")
        axes.append(axis)
    translation_norm_m = torch.linalg.vector_norm(translation_m, dim=1)
    rotation_norm_rad = torch.linalg.vector_norm(rotation_rad, dim=1)
    rates["applied_component_saturation_rate"] = float(
        (bounded.abs() >= 1.0).to(torch.float64).mean()
    )
    return {
        "axes": axes,
        "translation_block_norm": {
            "normalized": _percentiles(torch.linalg.vector_norm(bounded[:, :3], dim=1)),
            "m": _percentiles(translation_norm_m),
            "mm": {key: value * 1_000.0 for key, value in _percentiles(translation_norm_m).items()},
        },
        "rotation_block_norm": {
            "normalized": _percentiles(torch.linalg.vector_norm(bounded[:, 3:], dim=1)),
            "rad": _percentiles(rotation_norm_rad),
            "deg": {key: value * 180.0 / math.pi for key, value in _percentiles(rotation_norm_rad).items()},
        },
        "rates": rates,
    }


def run_audit(
    *,
    config_path: Path,
    task_config_path: Path = ROOT / "configs/tasks/rj45-v1.json",
    seed: int,
    sample_count: int,
    initial_log_std: float,
) -> dict[str, Any]:
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not math.isfinite(initial_log_std) or not MIN_LOG_STD <= initial_log_std <= MAX_LOG_STD:
        raise ValueError(f"initial_log_std must be within [{MIN_LOG_STD}, {MAX_LOG_STD}]")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
    shared = config["shared_control_contract"]
    translation_bound_m = float(shared["translation_bound_m"])
    rotation_bound_rad = float(shared["rotation_bound_rad"])
    task_controller = task_config["controller"]
    task_translation_bound_m = float(task_controller["residual_translation_bound_m"])
    task_rotation_bound_rad = float(task_controller["residual_rotation_bound_rad"])
    for name, value in (
        ("translation_bound_m", translation_bound_m),
        ("rotation_bound_rad", rotation_bound_rad),
        ("task residual_translation_bound_m", task_translation_bound_m),
        ("task residual_rotation_bound_rad", task_rotation_bound_rad),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if translation_bound_m != task_translation_bound_m or rotation_bound_rad != task_rotation_bound_rad:
        raise ValueError("comparison and task controller action bounds must match")

    agents = {
        algorithm: make_agent(
            algorithm,
            num_envs=1,
            rollout_steps=1,
            batch_size=1,
            seed=seed,
            mean_initializer="zero",
            initial_log_std=initial_log_std,
        )
        for algorithm in ("ppo", "sac")
    }
    observation_generator = torch.Generator(device="cpu").manual_seed(seed)
    fixed_observation = torch.randn(1, OBSERVATION_DIM, generator=observation_generator)
    for algorithm, agent in agents.items():
        if not torch.equal(deterministic_action(agent, fixed_observation), torch.zeros(1, ACTION_DIM)):
            raise RuntimeError(f"{algorithm} zero-mean deterministic action contract failed")

    standard_deviation = math.exp(initial_log_std)
    results = {}
    repeated_observation = fixed_observation.expand(sample_count, -1)
    for algorithm, agent in agents.items():
        torch.manual_seed(seed)
        with torch.no_grad():
            actions, _ = agent.policy.act(
                {"observations": repeated_observation}, role="policy"
            )
        torch.manual_seed(seed)
        pre_squash = torch.randn(sample_count, ACTION_DIM) * standard_deviation
        expected = (
            torch.clamp(pre_squash, -1.0, 1.0)
            if algorithm == "ppo"
            else torch.tanh(pre_squash)
        )
        if not torch.equal(actions, expected):
            raise RuntimeError(f"{algorithm} actual policy sampler differs from its expected transform")
        summary = summarize_actions(
            actions,
            translation_bound_m=translation_bound_m,
            rotation_bound_rad=rotation_bound_rad,
        )
        summary["sample_sha256"] = hashlib.sha256(
            actions.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()
        summary["pre_transform_component_over_one_rate"] = float(
            (pre_squash.abs() > 1.0).to(torch.float64).mean()
        )
        summary["policy_transform_component_clip_rate"] = (
            summary["pre_transform_component_over_one_rate"] if algorithm == "ppo" else 0.0
        )
        summary["policy_transform_sample_clip_rate"] = (
            float((pre_squash.abs() > 1.0).any(dim=1).to(torch.float64).mean())
            if algorithm == "ppo"
            else 0.0
        )
        results[algorithm] = summary

    source_paths = {
        "audit_script": Path(__file__).resolve(),
        "comparison_rl": ROOT / "src/rby1_assembly/rj45_comparison_rl.py",
        "rj45_env": ROOT / "src/rby1_assembly/rj45_env.py",
        "rj45_training": ROOT / "src/rby1_assembly/rj45_training.py",
        "rj45_training_env": ROOT / "src/rby1_assembly/rj45_training_env.py",
        "isaac_env": ROOT / "src/rby1_assembly/isaac_env.py",
    }
    return {
        "schema_name": "rj45-residual-initialization-audit-v1",
        "provenance": {
            "config_path": str(config_path.resolve()),
            "config_sha256": _sha256(config_path),
            "task_config_path": str(task_config_path.resolve()),
            "task_config_sha256": _sha256(task_config_path),
            "source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
            "seed": seed,
            "sample_count": sample_count,
            "initial_log_std": initial_log_std,
            "log_std_bounds": [MIN_LOG_STD, MAX_LOG_STD],
            "mean_initializer": "zero",
            "actor_selection": {"ppo": "Gaussian_clip", "sac": "Gaussian_tanh"},
            "actual_sample_path": {
                "ppo": "skrl.models.torch.GaussianMixin.act",
                "sac": "rby1_assembly.rj45_comparison_rl.SquashedGaussianActor.act",
            },
            "skrl_version": metadata.version("skrl"),
            "actor_observation_contract": "79D_simulation_proxy_only",
            "translation_bound_m": translation_bound_m,
            "rotation_bound_rad": rotation_bound_rad,
        },
        "claims": {
            "cpu_distribution_audit_only": True,
            "physics_executed": False,
            "training_executed": False,
            "learning_effect_claimed": False,
            "g2_parameter_frozen": False,
            "initial_log_std_selected_from_g1_results": False,
        },
        "results": results,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rj45-comparison-development.json")
    parser.add_argument("--task-config", type=Path, default=ROOT / "configs/tasks/rj45-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--sample-count", type=int, default=10_000)
    parser.add_argument("--initial-log-std", type=float, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_audit(
        config_path=args.config.resolve(),
        task_config_path=args.task_config.resolve(),
        seed=args.seed,
        sample_count=args.sample_count,
        initial_log_std=args.initial_log_std,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
