#!/usr/bin/env python3
"""CPU smoke for the common skrl PPO/SAC comparison boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch

from rby1_assembly.rj45_comparison_rl import (
    OBSERVATION_DIM,
    collect_bounded,
    deterministic_action,
    finite_losses,
    load_inference_checkpoint,
    make_agent,
    parameter_vector,
    save_inference_checkpoint,
)


class SyntheticVectorEnv:
    """Small differentiable-looking control signal; not an Isaac task benchmark."""

    def __init__(self, num_envs: int, horizon: int = 6):
        self.num_envs = num_envs
        self.horizon = horizon
        self.step_count = torch.zeros(num_envs, dtype=torch.int64)
        self.observations = torch.zeros(num_envs, OBSERVATION_DIM)

    def reset(self):
        self.step_count.zero_()
        target = torch.linspace(-0.6, 0.6, self.num_envs).unsqueeze(1)
        self.observations.zero_()
        self.observations[:, :6] = target.repeat(1, 6)
        return self.observations.clone()

    def step(self, actions):
        self.step_count += 1
        target = self.observations[:, :6]
        rewards = -((actions - target) ** 2).mean(dim=-1, keepdim=True)
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        truncated = (self.step_count >= self.horizon).unsqueeze(1)
        if truncated.any():
            self.step_count[truncated.squeeze(1)] = 0
        return self.observations.clone(), rewards, terminated, truncated, {}


def run_one(algorithm: str, *, num_envs: int, steps: int) -> dict:
    torch.manual_seed(7)
    agent = make_agent(
        algorithm,
        num_envs=num_envs,
        rollout_steps=8,
        replay_capacity=512,
        batch_size=16,
        seed=7,
    )
    before = parameter_vector(agent)
    result = collect_bounded(
        agent, SyntheticVectorEnv(num_envs), algorithm=algorithm, vector_steps=steps
    )
    after = parameter_vector(agent)
    losses = finite_losses(agent)
    probe = torch.linspace(-1.0, 1.0, OBSERVATION_DIM).repeat(3, 1)
    expected = deterministic_action(agent, probe)
    with tempfile.TemporaryDirectory(prefix="rby1-comparison-") as directory:
        checkpoint = Path(directory) / f"{algorithm}.pt"
        save_inference_checkpoint(agent, checkpoint, algorithm=algorithm)
        restored = make_agent(
            algorithm,
            num_envs=num_envs,
            rollout_steps=8,
            replay_capacity=512,
            batch_size=16,
            seed=99,
        )
        load_inference_checkpoint(restored, checkpoint)
        actual = deterministic_action(restored, probe)
    return {
        "algorithm": algorithm,
        "synthetic_only": True,
        "isaac_task_performance_claim": False,
        "parameter_update_l2": float(torch.linalg.vector_norm(after - before).item()),
        "parameter_update_nonzero": bool(torch.any(after != before)),
        "parameters_finite": bool(torch.isfinite(after).all()),
        "losses": losses,
        "checkpoint_deterministic_action_max_abs_error": float(
            torch.max(torch.abs(actual - expected)).item()
        ),
        "collector": result.__dict__,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path, help="unsupported by design; fails closed")
    args = parser.parse_args()
    if args.resume is not None:
        raise RuntimeError(
            "training resume is disabled because this runner does not persist replay/rollout state"
        )
    report = {
        "scope": "synthetic tensor smoke; this is not Isaac Sim task performance evidence",
        "ppo": run_one("ppo", num_envs=4, steps=16),
        "sac": run_one("sac", num_envs=4, steps=16),
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
