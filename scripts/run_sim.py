# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the smallest real-physics smoke run for the assembly environment."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RB-Y1 assembly PhysX smoke run")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=1200)
parser.add_argument("--report", type=Path, default=Path("artifacts/smoke_report.json"))
parser.add_argument("--require-completion", action="store_true")
parser.add_argument("--background", action="store_true", help="Load the visual-only Isaac Simple Room once.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or args.steps <= 0:
    parser.error("--num-envs and --steps must be positive")
launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from rby1_assembly.isaac_env import AssemblyPhase, Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.render_background = args.background
    env = Rby1AssemblyEnv(cfg)
    completed = False
    max_contact_force_proxy_n = 0.0
    grip_proof_peak_count = 0
    actual_steps = 0
    terminal_phase_counts = {phase.name.lower(): 0 for phase in AssemblyPhase}
    last_phase_counts = {phase.name.lower(): 0 for phase in AssemblyPhase}
    try:
        env.reset()
        for _ in range(args.steps):
            actions = torch.zeros((env.num_envs, 12), device=env.device)
            _, _, terminated, truncated, extras = env.step(actions)
            actual_steps += 1
            snapshot = extras["terminal_snapshot"]
            max_contact_force_proxy_n = max(
                max_contact_force_proxy_n,
                float(snapshot["contact_force_proxy_peak_n"].max().item()),
            )
            grip_proof_peak_count = max(
                grip_proof_peak_count,
                int(torch.sum(torch.all(snapshot["grip_proven"], dim=1)).item()),
            )
            completed = completed or bool(torch.any(extras["success"]).item())
            snapshot_phase = snapshot["phase"]
            last_phase_counts = {
                phase.name.lower(): int(torch.sum(snapshot_phase == int(phase)).item())
                for phase in AssemblyPhase
            }
            terminal_mask = snapshot["terminal_mask"]
            for phase in AssemblyPhase:
                terminal_phase_counts[phase.name.lower()] += int(
                    torch.sum(terminal_mask & (snapshot_phase == int(phase))).item()
                )
            if bool(torch.all(terminated | truncated).item()):
                break
        task_path = PROJECT_ROOT / "configs/task.json"
        assets_path = PROJECT_ROOT / "configs/assets.json"
        runtime_path = PROJECT_ROOT / "configs/runtime.json"
        build_report_path = PROJECT_ROOT / "assets/generated/build-report.json"
        source_paths = [
            PROJECT_ROOT / "src/rby1_assembly/isaac_env.py",
            PROJECT_ROOT / "scripts/run_sim.py",
        ]
        sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        report = {
            "completed": completed,
            "physics_backend": "PhysX via Isaac Sim",
            "observation_source": "simulator_state_with_noise",
            "baseline_controller_state_source": "simulator_ground_truth",
            "phase_gate_state_source": "simulator_ground_truth",
            "contact_force_source": "contact_force_proxy",
            "num_envs": env.num_envs,
            "steps_requested": args.steps,
            "actual_steps": actual_steps,
            "max_contact_force_proxy_n": max_contact_force_proxy_n,
            "visual_background_enabled": args.background,
            "last_pre_autoreset_phase_counts": last_phase_counts,
            "terminal_phase_counts": terminal_phase_counts,
            "grip_proof_peak_count": grip_proof_peak_count,
            "runtime_validated": actual_steps > 0,
            "provenance": {
                "parameter_provenance": "simulation_design_assumptions_not_hardware_calibration",
                "physics_dt_s": float(cfg.sim.dt),
                "control_decimation": int(cfg.decimation),
                "control_dt_s": float(cfg.sim.dt * cfg.decimation),
                "task_config_sha256": sha256(task_path),
                "assets_config_sha256": sha256(assets_path),
                "runtime_config_sha256": sha256(runtime_path),
                "build_report_sha256": sha256(build_report_path),
                "source_sha256": {
                    str(path.relative_to(PROJECT_ROOT)): sha256(path) for path in source_paths
                },
                "generated_asset_sha256": json.loads(
                    build_report_path.read_text(encoding="utf-8")
                )["files"],
                "torch_version": torch.__version__,
            },
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 2 if args.require_completion and not completed else 0
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
