# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the 12-D residual policy with the real RSL-RL PPO runner."""

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Train RB-Y1 assembly residual policy")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--config", type=Path, default=Path("configs/ppo.json"))
parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/rby1_assembly"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
try:
    runner_cfg = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_iterations is not None:
        runner_cfg["max_iterations"] = args.max_iterations
    for key in ("max_iterations", "num_steps_per_env", "save_interval"):
        if type(runner_cfg[key]) is not int or runner_cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(runner_cfg["seed"]) is not int or runner_cfg["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(runner_cfg["policy"], dict) or not isinstance(runner_cfg["algorithm"], dict):
        raise ValueError("policy and algorithm must be configuration objects")
except (OSError, ValueError, TypeError, KeyError) as error:
    parser.error(f"invalid training configuration: {error}")
launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import omni.kit.app  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = runner_cfg["seed"]
    env = Rby1AssemblyEnv(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get("clip_actions", 1.0))
    log_dir = args.log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "runner_config.json").write_text(json.dumps(runner_cfg, indent=2) + "\n", encoding="utf-8")
    try:
        source_paths = sorted((PROJECT_ROOT / "src/rby1_assembly").glob("*.py"))
        evidence_paths = source_paths + [Path(__file__).resolve(), args.config.resolve()]
        evidence_paths += sorted((PROJECT_ROOT / "configs").glob("*.json"))
        evidence_paths += sorted((PROJECT_ROOT / "assets/generated").glob("*.usda"))
        evidence_paths += [PROJECT_ROOT / "assets/generated/build-report.json", log_dir / "runner_config.json"]
        lineage = {
            "source_and_config_sha256": {
                str(path.resolve().relative_to(PROJECT_ROOT)) if path.resolve().is_relative_to(PROJECT_ROOT) else str(path.resolve()):
                hashlib.sha256(path.read_bytes()).hexdigest() for path in evidence_paths
            },
            "runtime_versions": {name: metadata.version(name) for name in
                                 ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py")},
            "seed": cfg.seed, "num_envs": cfg.scene.num_envs,
            "physics_dt_s": cfg.sim.dt, "control_decimation": cfg.decimation,
            "control_dt_s": cfg.sim.dt * cfg.decimation,
            "observation_noise": {
                "distribution": "independent_zero_mean_gaussian_position_and_rotation_vector",
                "position_std_m": cfg.observation_position_noise_std_m,
                "rotation_vector_std_rad": cfg.observation_rotation_noise_std_rad,
            },
            "baseline_controller_state_source": "simulator_ground_truth",
            "checkpoint_scope": "checkpoints created in this run directory",
        }
        (log_dir / "training_provenance.json").write_text(json.dumps(lineage, indent=2) + "\n")
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=str(log_dir), device=env.device)
        runner.add_git_repo_to_log(__file__)
        runner.learn(num_learning_iterations=runner_cfg["max_iterations"], init_at_random_ep_len=False)
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
