#!/usr/bin/env python3
"""실제 Isaac Lab 물리에서 1-env zero-residual baseline episode를 기록한다."""

from __future__ import annotations

import argparse
import importlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--episodes", type=int, default=3)
parser.add_argument("--steps", type=int, default=900)
parser.add_argument("--seed", type=int, default=20261025)
parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_rollouts.h5"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from rby1_assembly.dataset import (  # noqa: E402
    EpisodeDatasetWriter,
    EpisodeProvenance,
    ProvenanceKind,
    inspect_dataset,
    sha256_file,
)
from rby1_assembly.contracts import TaskLimits  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402
from rby1_assembly.recording import (  # noqa: E402
    AssemblyGeometry,
    existing_source_hashes,
    record_baseline_rollouts,
    select_failure_reports,
)


def _required_runtime_version(module_name: str, *distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if not version:
        raise RuntimeError(f"runtime version is unavailable for {module_name}")
    return str(version)


def main() -> int:
    if args.episodes <= 0 or args.steps <= 0:
        parser.error("--episodes and --steps must be positive")
    task_config = PROJECT_ROOT / "configs" / "task.json"
    task_values = json.loads(task_config.read_text(encoding="utf-8"))
    evaluation_values = task_values["evaluation"]
    task_limits = TaskLimits(
        assembled_depth_m=evaluation_values["assembled_depth_m"],
        max_lateral_error_m=evaluation_values["max_lateral_error_m"],
        max_orientation_error_rad=evaluation_values["max_orientation_error_rad"],
        max_contact_force_n=evaluation_values["max_contact_force_n"],
        dwell_s=evaluation_values["dwell_s"],
        timeout_s=evaluation_values["timeout_s"],
    )
    geometry = AssemblyGeometry(
        socket_entry_y_m=-task_values["socket"]["length_m"] / 2.0,
        plug_tip_y_m=(
            task_values["plug"]["grip_length_m"] / 2.0
            + task_values["plug"]["shaft_length_m"]
        ),
    )
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    source_paths = [
        PROJECT_ROOT / "src" / "rby1_assembly" / "isaac_env.py",
        PROJECT_ROOT / "src" / "rby1_assembly" / "scene.py",
        PROJECT_ROOT / "src" / "rby1_assembly" / "recording.py",
        PROJECT_ROOT / "src" / "rby1_assembly" / "dataset.py",
        PROJECT_ROOT / "src" / "rby1_assembly" / "contracts.py",
        PROJECT_ROOT / "src" / "rby1_assembly" / "evaluation.py",
        PROJECT_ROOT / "scripts" / "record_baseline.py",
        PROJECT_ROOT / "configs" / "runtime.json",
        PROJECT_ROOT / "configs" / "assets.json",
        PROJECT_ROOT / "assets" / "generated" / "build-report.json",
        PROJECT_ROOT / "assets" / "generated" / "scene.usda",
        PROJECT_ROOT / "assets" / "generated" / "robot.usda",
        PROJECT_ROOT / "assets" / "generated" / "socket.usda",
        PROJECT_ROOT / "assets" / "generated" / "plug.usda",
        PROJECT_ROOT / "assets" / "generated" / "table.usda",
        task_config,
    ]
    provenance = EpisodeProvenance(
        kind=ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT,
        source_hashes=existing_source_hashes(source_paths, base_dir=PROJECT_ROOT),
        task_config_sha256=sha256_file(task_config),
        runtime_versions={
            "isaac_sim": _required_runtime_version("isaacsim", "isaacsim"),
            "isaac_lab": _required_runtime_version("isaaclab", "isaaclab"),
            "torch": torch.__version__,
            "physics_backend": "PhysX through Isaac Sim",
        },
        observation_source="simulator_state_with_noise",
        contact_force_source="contact_force_proxy_per_hand_magnitude",
        owns_demonstration=True,
        experiment_parameters={
            "physics_dt_s": float(cfg.sim.dt),
            "control_decimation": int(cfg.decimation),
            "control_dt_s": float(cfg.sim.dt * cfg.decimation),
            "observation_noise": {
                "distribution": "independent_zero_mean_gaussian",
                "position_std_m": float(cfg.observation_position_noise_std_m),
                "rotation_std_rad": float(cfg.observation_rotation_noise_std_rad),
            },
            "baseline_action": "zero_residual_12d",
            "baseline_controller_state_source": "simulator_ground_truth",
            "phase_gate_state_source": "simulator_ground_truth",
            "transition_semantics": "observation_t_action_t_physics_state_t_plus_1",
            "terminal_state_source": "pre_autoreset_snapshot",
            "geometry_metric_source": "raw_socket_plug_world_pose_recomputed",
            "parameter_provenance": task_values["parameter_provenance"],
        },
    )
    env = Rby1AssemblyEnv(cfg)
    try:
        with EpisodeDatasetWriter(args.output) as writer:
            episodes = record_baseline_rollouts(
                env=env,
                writer=writer,
                provenance=provenance,
                episode_count=args.episodes,
                max_steps=args.steps,
                base_seed=args.seed,
                action_factory=lambda active_env: torch.zeros(
                    (1, 12), device=active_env.device, dtype=torch.float32
                ),
                task_limits=task_limits,
                geometry=geometry,
            )
        report = {
            "episodes": episodes,
            "failure_examples": select_failure_reports(episodes, limit=2),
            "dataset": inspect_dataset(args.output),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report["dataset"]["valid"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
