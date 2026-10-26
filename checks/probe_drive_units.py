#!/usr/bin/env python3
"""Measure the authored RB-Y1 implicit-drive response in live PhysX.

This is a diagnostic torque-injection experiment.  It does not execute or
evaluate the assembly task, and its outputs must not be used as task-success
evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe RB-Y1 implicit joint-drive units in PhysX")
parser.add_argument("--report", type=Path, required=True)
parser.add_argument("--trace", type=Path, required=True)
parser.add_argument("--settle-steps", type=int, default=240)
parser.add_argument("--sample-steps", type=int, default=60)
parser.add_argument("--torque-nm", type=float, default=5.0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.settle_steps < 120:
    parser.error("--settle-steps must be at least 120 physics steps")
if not 20 <= args.sample_steps <= args.settle_steps:
    parser.error("--sample-steps must be between 20 and --settle-steps")
if not math.isfinite(args.torque_nm) or args.torque_nm <= 0.0:
    parser.error("--torque-nm must be finite and positive")

launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
import omni.kit.app  # noqa: E402

from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def _sample_summary(rows: list[dict], sample_steps: int) -> dict:
    sample = rows[-sample_steps:]
    positions = [row["joint_position_rad"] for row in sample]
    velocities = [row["joint_velocity_rad_s"] for row in sample]
    times = [row["sim_time_s"] for row in sample]
    time_mean = _mean(times)
    position_mean = _mean(positions)
    denominator = sum((value - time_mean) ** 2 for value in times)
    slope = (
        sum((time - time_mean) * (position - position_mean) for time, position in zip(times, positions))
        / denominator
        if denominator > 0.0
        else 0.0
    )
    return {
        "sample_count": len(sample),
        "duration_s": times[-1] - times[0],
        "joint_position_mean_rad": position_mean,
        "joint_position_std_rad": statistics.pstdev(positions),
        "joint_position_peak_to_peak_rad": max(positions) - min(positions),
        "joint_position_slope_rad_s": slope,
        "joint_velocity_abs_max_rad_s": max(abs(value) for value in velocities),
        "gravity_compensation_mean_nm": _mean(
            [row["gravity_compensation_nm"] for row in sample]
        ),
        "applied_torque_mean_nm": _mean([row["applied_torque_nm"] for row in sample]),
        "computed_torque_mean_nm": _mean([row["computed_torque_nm"] for row in sample]),
        "position_target_mean_rad": _mean([row["position_target_rad"] for row in sample]),
        "effort_target_mean_nm": _mean([row["effort_target_nm"] for row in sample]),
    }


def _phase_steps(label: str, settle_steps: int) -> int:
    if label.startswith("zero_after") or label == "zero_final":
        return max(120, settle_steps * 3 // 4)
    return settle_steps


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = 20261026
    env = Rby1AssemblyEnv(cfg)
    trace_file = None
    try:
        env.reset()
        robot = env.robot
        q_reference = robot.data.default_joint_pos.clone()
        zero_velocity = torch.zeros_like(q_reference)
        zero_effort = torch.zeros_like(q_reference)
        gravity_bias_joint_ids = env._torso_joint_ids + env._arm_joint_ids
        joint_names = tuple(robot.data.joint_names)
        probe_names = ("right_arm_0", "left_arm_0")
        probe_ids = {}
        for name in probe_names:
            matches, resolved = robot.find_joints(name, preserve_order=True)
            if len(matches) != 1 or resolved != [name]:
                raise RuntimeError(f"could not resolve one exact joint for {name}: {matches}, {resolved}")
            probe_ids[name] = int(matches[0])

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace_file = args.trace.open("w", encoding="utf-8", buffering=1)
        phase_summaries: dict[str, dict[str, dict]] = {}
        global_step = 0
        phase_plan = (
            ("zero_baseline", 0.0, False),
            ("plus_torque", args.torque_nm, False),
            ("zero_after_plus", 0.0, False),
            ("minus_torque", -args.torque_nm, False),
            ("zero_after_minus", 0.0, False),
            ("gravity_position_bias", 0.0, True),
            ("zero_final", 0.0, False),
        )

        for probe_name in probe_names:
            probe_id = probe_ids[probe_name]
            phase_summaries[probe_name] = {}
            for phase_label, effort_nm, gravity_position_bias in phase_plan:
                phase_rows: list[dict] = []
                for phase_step in range(_phase_steps(phase_label, args.settle_steps)):
                    position_target = q_reference.clone()
                    if gravity_position_bias:
                        gravity = robot.root_physx_view.get_gravity_compensation_forces()
                        stiffness = robot.data.joint_stiffness[:, gravity_bias_joint_ids]
                        if torch.any(stiffness <= 0.0):
                            raise RuntimeError("gravity position bias requires positive stiffness")
                        position_target[:, gravity_bias_joint_ids] += (
                            gravity[:, gravity_bias_joint_ids] / stiffness
                        )
                    effort_target = zero_effort.clone()
                    effort_target[:, probe_id] = effort_nm
                    robot.set_joint_position_target(position_target)
                    robot.set_joint_velocity_target(zero_velocity)
                    robot.set_joint_effort_target(effort_target)
                    env.scene.write_data_to_sim()
                    env.sim.step(render=False)
                    env.scene.update(cfg.sim.dt)
                    global_step += 1

                    gravity = robot.root_physx_view.get_gravity_compensation_forces()
                    row = {
                        "global_physics_step": global_step,
                        "sim_time_s": global_step * float(cfg.sim.dt),
                        "probe_joint": probe_name,
                        "probe_joint_id": probe_id,
                        "phase": phase_label,
                        "phase_step": phase_step + 1,
                        "joint_position_rad": float(robot.data.joint_pos[0, probe_id].item()),
                        "joint_velocity_rad_s": float(robot.data.joint_vel[0, probe_id].item()),
                        "position_target_rad": float(
                            robot.data.joint_pos_target[0, probe_id].item()
                        ),
                        "effort_target_nm": float(
                            robot.data.joint_effort_target[0, probe_id].item()
                        ),
                        "gravity_compensation_nm": float(gravity[0, probe_id].item()),
                        "computed_torque_nm": float(
                            robot.data.computed_torque[0, probe_id].item()
                        ),
                        "applied_torque_nm": float(robot.data.applied_torque[0, probe_id].item()),
                    }
                    phase_rows.append(row)
                    trace_file.write(json.dumps(row, separators=(",", ":")) + "\n")
                phase_summaries[probe_name][phase_label] = _sample_summary(
                    phase_rows, args.sample_steps
                )

        trace_file.flush()
        trace_file.close()
        trace_file = None
        articulation_source = Path(
            sys.modules[type(robot).__module__].__file__  # type: ignore[arg-type]
        ).resolve()
        script_path = Path(__file__).resolve()
        environment_path = PROJECT_ROOT / "src" / "rby1_assembly" / "isaac_env.py"
        joint_static = {}
        for probe_name, probe_id in probe_ids.items():
            phases = phase_summaries[probe_name]
            q_plus = phases["plus_torque"]["joint_position_mean_rad"]
            q_minus = phases["minus_torque"]["joint_position_mean_rad"]
            response = q_plus - q_minus
            reported_stiffness = float(robot.data.joint_stiffness[0, probe_id].item())
            zero_offset = (
                phases["zero_baseline"]["joint_position_mean_rad"]
                - float(q_reference[0, probe_id].item())
            )
            gravity_bias_offset = (
                phases["gravity_position_bias"]["joint_position_mean_rad"]
                - float(q_reference[0, probe_id].item())
            )
            expected_bias = (
                phases["gravity_position_bias"]["gravity_compensation_mean_nm"]
                / reported_stiffness
            )
            joint_static[probe_name] = {
                "joint_id": probe_id,
                "reference_position_rad": float(q_reference[0, probe_id].item()),
                "reported_stiffness_nm_per_rad": reported_stiffness,
                "reported_damping_nm_s_per_rad": float(
                    robot.data.joint_damping[0, probe_id].item()
                ),
                "reported_effort_limit_nm": float(
                    robot.data.joint_effort_limits[0, probe_id].item()
                ),
                "plus_minus_position_response_rad": response,
                "empirical_effective_stiffness_nm_per_rad": (
                    2.0 * args.torque_nm / response if abs(response) > 1.0e-9 else None
                ),
                "zero_effort_static_offset_rad": zero_offset,
                "gravity_position_bias_expected_rad": expected_bias,
                "gravity_position_bias_observed_offset_rad": gravity_bias_offset,
                "gravity_position_bias_residual_rad": gravity_bias_offset - expected_bias,
                "phases": phases,
            }

        report = {
            "schema_version": 1,
            "experiment": "rb_y1_implicit_drive_static_torque_probe",
            "scope": (
                "Diagnostic torque injection only; no assembly task execution, success evaluation, "
                "object-pose injection, or sensor-value injection."
            ),
            "physics_executed": True,
            "device": args.device,
            "physics_dt_s": float(cfg.sim.dt),
            "torque_magnitude_nm": args.torque_nm,
            "settle_steps_default": args.settle_steps,
            "sample_steps": args.sample_steps,
            "joint_names": joint_names,
            "joints": joint_static,
            "interpretation_limits": [
                "Empirical stiffness includes gravity variation and multi-joint rigid-body coupling.",
                "The measurement is a local static response around the authored default pose.",
                "A stable phase requires near-zero final-window slope, velocity, and peak-to-peak motion.",
            ],
            "provenance": {
                "script_path": str(script_path.relative_to(PROJECT_ROOT)),
                "script_sha256": _sha256(script_path),
                "environment_path": str(environment_path.relative_to(PROJECT_ROOT)),
                "environment_sha256": _sha256(environment_path),
                "articulation_source_path": str(articulation_source),
                "articulation_source_sha256": _sha256(articulation_source),
                "trace_path": str(args.trace),
                "trace_sha256": _sha256(args.trace),
                "trace_bytes": args.trace.stat().st_size,
                "trace_record_count": global_step,
            },
        }
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.report)
        print(json.dumps(report, indent=2), flush=True)
        return 0
    finally:
        if trace_file is not None:
            trace_file.close()
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        exit_code = 1
    finally:
        try:
            omni.kit.app.get_app().post_quit(exit_code)
        except Exception:
            traceback.print_exc()
            exit_code = 1
        try:
            simulation_app.close()
        except Exception:
            traceback.print_exc()
            exit_code = 1
    raise SystemExit(exit_code)
