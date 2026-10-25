# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a saved RSL-RL checkpoint with deterministic environment seeds."""

import argparse
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Evaluate RB-Y1 assembly policy")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=910251)
parser.add_argument("--config", type=Path, default=Path("configs/ppo.json"))
parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.episodes <= 0:
    parser.error("--num-envs and --episodes must be positive")
if not args.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args.checkpoint}")
if not args.config.is_file():
    parser.error(f"config does not exist: {args.config}")
launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import omni.kit.app  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unavailable"


def main() -> int:
    runner_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    env = Rby1AssemblyEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get("clip_actions", 1.0))
    try:
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=env.device)
        runner.load(str(args.checkpoint))
        policy = runner.get_inference_policy(device=env.device)
        observations = wrapped.get_observations()
        attempts = 0
        valid_successes = 0
        valid_failures = 0
        invalid_episodes = 0
        invalid_reasons: dict[str, int] = {}
        force_peaks = []
        elapsed_times_s = []
        retry_counts = []
        recovery_attempts = 0
        recovered_successes = 0
        while attempts < args.episodes:
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, extras = wrapped.step(actions)
            ended = torch.nonzero(dones, as_tuple=False).flatten()
            if len(ended) == 0:
                continue
            for env_id in ended.tolist():
                if attempts >= args.episodes:
                    break
                attempts += 1
                valid = bool(extras["evaluation"]["valid"][env_id].item())
                succeeded = valid and bool(extras["success"][env_id].item())
                valid_successes += int(succeeded)
                valid_failures += int(valid and not succeeded)
                invalid_episodes += int(not valid)
                if valid:
                    snapshot = extras["terminal_snapshot"]
                    retries = int(snapshot["retry_count"][env_id].item())
                    retry_counts.append(retries)
                    elapsed_times_s.append(float(snapshot["timestamp_s"][env_id].item()))
                    recovery_attempts += int(retries > 0)
                    recovered_successes += int(retries > 0 and succeeded)
                if not valid:
                    reasons = extras["terminal_evaluation"]["termination_reason"]
                    reason = str(reasons[env_id])
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                force_peaks.append(float(extras["evaluation"]["contact_force_proxy_peak_n"][env_id].item()))
        valid_episodes = valid_successes + valid_failures
        task_path = PROJECT_ROOT / "configs/task.json"
        assets_path = PROJECT_ROOT / "configs/assets.json"
        runtime_path = PROJECT_ROOT / "configs/runtime.json"
        build_report_path = PROJECT_ROOT / "assets/generated/build-report.json"
        source_paths = sorted((PROJECT_ROOT / "src").rglob("*.py")) + [
            PROJECT_ROOT / "scripts/evaluate_policy.py",
        ]
        report = {
            "requested_attempts": args.episodes,
            "attempts": attempts,
            "valid_episodes": valid_episodes,
            "valid_successes": valid_successes,
            "valid_failures": valid_failures,
            "invalid_episodes": invalid_episodes,
            "invalid_reasons": invalid_reasons,
            "success_rate_valid_episodes": (
                valid_successes / valid_episodes if valid_episodes else None
            ),
            "recovery_attempts_valid": recovery_attempts,
            "recovered_successes_valid": recovered_successes,
            "recovery_rate_valid": recovered_successes / recovery_attempts if recovery_attempts else None,
            "recovery_definition": "success among valid episodes with at least one commanded retry",
            "episode_elapsed_times_s_valid": elapsed_times_s,
            "mean_episode_elapsed_s_valid": sum(elapsed_times_s) / len(elapsed_times_s) if elapsed_times_s else None,
            "episode_retry_counts_valid": retry_counts,
            "mean_retry_count_valid": sum(retry_counts) / len(retry_counts) if retry_counts else None,
            "max_contact_force_proxy_n": max(force_peaks, default=0.0),
            "seed": args.seed,
            "observation_source": "simulator_state_with_noise",
            "baseline_controller_state_source": "simulator_ground_truth",
            "completion_source": "simulator_ground_truth_evaluator",
            "sample_definition": "terminated_or_truncated_episode_attempt",
            "provenance": {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": _sha256(args.checkpoint),
                "policy_config": str(args.config.resolve()),
                "policy_config_sha256": _sha256(args.config),
                "task_config_sha256": _sha256(task_path),
                "assets_config_sha256": _sha256(assets_path),
                "runtime_config_sha256": _sha256(runtime_path),
                "build_report_sha256": _sha256(build_report_path),
                "source_sha256": {
                    str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths
                },
                "generated_asset_sha256": json.loads(
                    build_report_path.read_text(encoding="utf-8")
                )["files"],
                "runtime_versions": {
                    "isaac_sim": _runtime_version("isaacsim"),
                    "isaac_lab": _runtime_version("isaaclab"),
                    "rsl_rl": _runtime_version("rsl-rl-lib", "rsl-rl"),
                    "torch": torch.__version__,
                    "physics_backend": "PhysX through Isaac Sim",
                },
                "physics_dt_s": float(cfg.sim.dt),
                "control_decimation": int(cfg.decimation),
                "control_dt_s": float(cfg.sim.dt * cfg.decimation),
                "observation_noise": {
                    "position_std_m": float(cfg.observation_position_noise_std_m),
                    "rotation_std_rad": float(cfg.observation_rotation_noise_std_rad),
                },
                "parameter_provenance": "simulation_design_assumptions_not_hardware_calibration",
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        wrapped.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        try:
            omni.kit.app.get_app().post_quit(exit_code)
        except Exception:
            print("Failed to set Isaac Kit exit code", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
        try:
            simulation_app.close()
        except Exception:
            print("Failed to close Isaac application", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
    raise SystemExit(exit_code)
