# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a saved RSL-RL checkpoint with deterministic environment seeds."""

import argparse
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.run_provenance import (  # noqa: E402
    InputMutationError,
    RunInputSnapshot,
    capture_run_inputs,
)


def _preimport_option(name: str, default: str) -> str:
    prefix = name + "="
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 1 < len(values):
            return values[index + 1]
    return default


_bootstrap_task = _preimport_option("--task", "toy")
_bootstrap_config_default = (
    "configs/ppo-rj45-v1.json" if _bootstrap_task == "rj45" else "configs/ppo.json"
)
_bootstrap_task_path = PROJECT_ROOT / (
    "configs/tasks/rj45-v1.json" if _bootstrap_task == "rj45" else "configs/task.json"
)
_bootstrap_nominal_report = _preimport_option("--rj45-nominal-evidence", "")
_bootstrap_nominal_paths: tuple[Path, ...] = ()
if _bootstrap_task == "rj45" and _bootstrap_nominal_report:
    bootstrap_nominal_report_path = Path(_bootstrap_nominal_report).resolve()
    bootstrap_nominal_report = json.loads(
        bootstrap_nominal_report_path.read_text(encoding="utf-8")
    )
    bootstrap_trace_value = bootstrap_nominal_report.get("trace_path")
    if not isinstance(bootstrap_trace_value, str) or not bootstrap_trace_value:
        raise ValueError("RJ45 nominal evidence report must identify its trace_path")
    bootstrap_trace_path = Path(bootstrap_trace_value)
    if not bootstrap_trace_path.is_absolute():
        bootstrap_trace_path = PROJECT_ROOT / bootstrap_trace_path
    _bootstrap_nominal_paths = (
        bootstrap_nominal_report_path,
        bootstrap_trace_path,
    )
_bootstrap_input_snapshot = capture_run_inputs(
    PROJECT_ROOT,
    task_path=_bootstrap_task_path,
    build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
    config_paths=(
        Path(_preimport_option("--config", _bootstrap_config_default)),
        PROJECT_ROOT / "configs/assets.json",
        PROJECT_ROOT / "configs/runtime.json",
        *((PROJECT_ROOT / "configs/rj45-assets.json",) if _bootstrap_task == "rj45" else ()),
    ),
    source_roots=((PROJECT_ROOT / "src", True),),
    source_files=(Path(__file__).resolve(),),
    extra_paths=(
        PROJECT_ROOT / "scripts/train.py",
        PROJECT_ROOT / "scripts/run_sim.py",
        *_bootstrap_nominal_paths,
    ),
)

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluate RB-Y1 assembly policy")
parser.add_argument("checkpoint", type=Path)
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--task", choices=("toy", "rj45"), default="toy")
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--config", type=Path, default=Path("configs/ppo.json"))
parser.add_argument("--report", type=Path, default=Path("artifacts/evaluation.json"))
parser.add_argument(
    "--rj45-nominal-evidence",
    type=Path,
    help="Completed current-source RJ45 nominal run report used to authorize the checkpoint.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.task == "rj45" and args.config == Path("configs/ppo.json"):
    args.config = Path("configs/ppo-rj45-v1.json")
if args.num_envs <= 0 or args.episodes <= 0:
    parser.error("--num-envs and --episodes must be positive")
if args.task == "rj45" and args.num_envs != 1:
    parser.error("parallel RJ45 collision validation pending; use --num-envs 1")
if args.task == "rj45" and args.rj45_nominal_evidence is None:
    parser.error("--task rj45 requires --rj45-nominal-evidence")
if args.seed is not None and args.seed < 0:
    parser.error("--seed must be non-negative")
if not args.checkpoint.is_file():
    parser.error(f"checkpoint does not exist: {args.checkpoint}")
if not args.config.is_file():
    parser.error(f"config does not exist: {args.config}")
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import omni.kit.app  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402
from rby1_assembly.experiment_contracts import atomic_write_json, load_finite_json, validate_ppo_config  # noqa: E402


_input_snapshot_for_failure_report: RunInputSnapshot | None = None


def _runtime_version(*distribution_names: str) -> str:
    for name in distribution_names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return "unavailable"


def _lineage_path_key(path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return str(resolved.relative_to(PROJECT_ROOT))
    return str(resolved)


def _write_evaluation_failure_report(error: Exception) -> None:
    immutability_error = error if isinstance(error, InputMutationError) else None
    if immutability_error is None and _input_snapshot_for_failure_report is not None:
        try:
            _input_snapshot_for_failure_report.assert_unchanged()
        except InputMutationError as detected:
            immutability_error = detected
    atomic_write_json(
        args.report,
        {
            "task_id": "rj45-v1" if args.task == "rj45" else "toy",
            "runtime_validated": False,
            "evaluation_valid": False,
            "input_immutability": (
                {"verified_unchanged": True}
                if _input_snapshot_for_failure_report is not None and immutability_error is None
                else None if immutability_error is None
                else immutability_error.report()
            ),
            "error": {"type": type(error).__name__, "message": str(error)},
        },
    )


def main() -> int:
    global _input_snapshot_for_failure_report
    task_path = PROJECT_ROOT / ("configs/tasks/rj45-v1.json" if args.task == "rj45" else "configs/task.json")
    assets_path = PROJECT_ROOT / "configs/assets.json"
    runtime_path = PROJECT_ROOT / "configs/runtime.json"
    connector_assets_path = PROJECT_ROOT / "configs/rj45-assets.json"
    lineage_path = args.checkpoint.parent / "training_provenance.json"
    manifest_path = args.checkpoint.parent / "checkpoint_manifest.json"
    nominal_trace_path = None
    if args.task == "rj45":
        nominal_report = load_finite_json(args.rj45_nominal_evidence)
        raw_trace_path = nominal_report.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path:
            raise ValueError("RJ45 nominal evidence report must identify its trace_path")
        nominal_trace_path = Path(raw_trace_path)
        if not nominal_trace_path.is_absolute():
            nominal_trace_path = PROJECT_ROOT / nominal_trace_path
    input_snapshot = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(
            args.config,
            assets_path,
            runtime_path,
            *((connector_assets_path,) if args.task == "rj45" else ()),
        ),
        source_roots=((PROJECT_ROOT / "src", True),),
        source_files=(PROJECT_ROOT / "scripts/evaluate_policy.py",),
        extra_paths=(
            args.checkpoint,
            PROJECT_ROOT / "scripts/train.py",
            PROJECT_ROOT / "scripts/run_sim.py",
            *((lineage_path, manifest_path, args.rj45_nominal_evidence, nominal_trace_path)
              if args.task == "rj45" else ()),
        ),
    )
    _input_snapshot_for_failure_report = input_snapshot
    _bootstrap_input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    if args.task == "rj45":
        captured_nominal_report = input_snapshot.json_for(args.rj45_nominal_evidence)
        if captured_nominal_report.get("trace_path") != raw_trace_path:
            raise InputMutationError(
                added=[], removed=[], changed=[input_snapshot.key_for(args.rj45_nominal_evidence)]
            )
        captured_trace_path = Path(captured_nominal_report["trace_path"])
        if not captured_trace_path.is_absolute():
            captured_trace_path = PROJECT_ROOT / captured_trace_path
        if captured_trace_path.resolve() != nominal_trace_path.resolve():
            raise InputMutationError(
                added=[], removed=[], changed=[input_snapshot.key_for(args.rj45_nominal_evidence)]
            )
    runner_cfg = input_snapshot.json_for(args.config)
    validate_ppo_config(runner_cfg)
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env_type = Rby1AssemblyEnv
    selected_task = input_snapshot.json_for(task_path)
    verified_checkpoint_evidence = None
    cfg.seed = (
        args.seed
        if args.seed is not None
        else int(selected_task["data"]["evaluation_seed"])
    )
    if args.task == "rj45":
        from rby1_assembly.rj45_env import configure_rj45_task, load_rj45_task
        from rby1_assembly.rj45_task import (
            verify_checkpoint_lineage,
            verify_checkpoint_manifest,
            verify_generated_asset_hashes,
            verify_runtime_versions,
        )

        task = load_rj45_task(task_path)
        env_type = configure_rj45_task(cfg, task, PROJECT_ROOT)
        if not lineage_path.is_file():
            raise ValueError("RJ45 evaluation requires training_provenance.json beside the checkpoint")
        lineage = input_snapshot.json_for(lineage_path)
        if not manifest_path.is_file():
            raise ValueError("RJ45 evaluation requires checkpoint_manifest.json beside the checkpoint")
        manifest = input_snapshot.json_for(manifest_path)
        verify_checkpoint_manifest(
            manifest,
            training_provenance_sha256=input_snapshot.hash_for(lineage_path),
            checkpoint_name=args.checkpoint.name,
            checkpoint_sha256=input_snapshot.hash_for(args.checkpoint),
        )
        build_report_path = (PROJECT_ROOT / task["assets"]["build_report"]).resolve()
        build_report = input_snapshot.json_for(build_report_path)
        generated_asset_sha256 = verify_generated_asset_hashes(
            PROJECT_ROOT, task, build_report
        )
        generated_dir = build_report_path.parent
        expected_lineage_sources = {
            _lineage_path_key(args.config): input_snapshot.hash_for(args.config),
            _lineage_path_key(build_report_path): input_snapshot.hash_for(build_report_path),
            "scripts/train.py": input_snapshot.hash_for(PROJECT_ROOT / "scripts/train.py"),
        }
        expected_lineage_sources.update(
            {
                _lineage_path_key(generated_dir / filename): digest
                for filename, digest in generated_asset_sha256.items()
            }
        )
        expected_lineage_sources.update(
            {
                _lineage_path_key(path): input_snapshot.hash_for(path)
                for path in sorted((PROJECT_ROOT / "src/rby1_assembly").glob("*.py"))
            }
        )
        current_versions = {name: _runtime_version(name) for name in
                            ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py")}
        verify_runtime_versions(current_versions, input_snapshot.json_for(runtime_path))
        from rby1_assembly.experiment_contracts import verify_rj45_nominal_evidence

        current_nominal_evidence = verify_rj45_nominal_evidence(
            PROJECT_ROOT, task, args.rj45_nominal_evidence
        )
        verify_checkpoint_lineage(
            lineage,
            task_id="rj45-v1",
            task_config_sha256=task["_config_sha256"],
            source_sha256=expected_lineage_sources,
            runtime_contract={
                "runtime_versions": current_versions,
                "physics_dt_s": cfg.sim.dt, "control_decimation": cfg.decimation,
                "control_dt_s": cfg.sim.dt * cfg.decimation,
                "observation_noise": {
                    "distribution": "independent_zero_mean_gaussian_position_and_rotation_vector",
                    "position_std_m": cfg.observation_position_noise_std_m,
                    "rotation_vector_std_rad": cfg.observation_rotation_noise_std_rad,
                },
                "baseline_controller_state_source": "simulator_ground_truth",
                "task_config_path": str(task_path.relative_to(PROJECT_ROOT)),
                "policy_config_path": _lineage_path_key(args.config),
                "collision_monitor_mode": "passive_nominal_one_environment", "num_envs": cfg.scene.num_envs,
            },
            rj45_nominal_evidence=current_nominal_evidence,
        )
        verified_checkpoint_evidence = {
            "training_provenance_sha256": input_snapshot.hash_for(lineage_path),
            "checkpoint_manifest_sha256": input_snapshot.hash_for(manifest_path),
            "runtime_contract_verified_at_start": True,
            "rj45_nominal_evidence": current_nominal_evidence,
        }
    input_snapshot.assert_unchanged()
    env = env_type(cfg)
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
                snapshot = extras["terminal_snapshot"]
                force_peak = float(extras["evaluation"]["contact_force_proxy_peak_n"][env_id].item())
                elapsed = float(snapshot["timestamp_s"][env_id].item())
                retries_value = float(snapshot["retry_count"][env_id].item())
                finite = all(math.isfinite(value) for value in (force_peak, elapsed, retries_value))
                finite = finite and all(bool(torch.isfinite(snapshot[key][env_id]).all().item()) for key in
                    ("object_poses_world_wxyz", "revolute_joint_positions_rad", "actual_depth_m",
                     "actual_lateral_error_m", "actual_orientation_error_rad"))
                valid = finite and bool(extras["evaluation"]["valid"][env_id].item())
                succeeded = valid and bool(extras["success"][env_id].item())
                valid_successes += int(succeeded)
                valid_failures += int(valid and not succeeded)
                invalid_episodes += int(not valid)
                if valid:
                    retries = int(retries_value)
                    retry_counts.append(retries)
                    elapsed_times_s.append(elapsed)
                    recovery_attempts += int(retries > 0)
                    recovered_successes += int(retries > 0 and succeeded)
                if not valid:
                    reasons = extras["terminal_evaluation"]["termination_reason"]
                    reason = str(reasons[env_id]) if finite else "nonfinite_episode_metrics"
                    invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1
                if finite:
                    force_peaks.append(force_peak)
        valid_episodes = valid_successes + valid_failures
        build_report_path = input_snapshot.build_report_path
        input_snapshot.assert_unchanged()
        report = {
            "requested_attempts": args.episodes,
            "task_id": "rj45-v1" if args.task == "rj45" else "toy",
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
            "seed": cfg.seed,
            "observation_source": "simulator_state_with_noise",
            "baseline_controller_state_source": "simulator_ground_truth",
            "completion_source": "simulator_ground_truth_evaluator",
            "sample_definition": "terminated_or_truncated_episode_attempt",
            "provenance": {
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_sha256": input_snapshot.hash_for(args.checkpoint),
                "verified_checkpoint_evidence": verified_checkpoint_evidence,
                "policy_config": str(args.config.resolve()),
                "policy_config_sha256": input_snapshot.hash_for(args.config),
                "task_config_path": str(task_path.relative_to(PROJECT_ROOT)),
                "task_config_sha256": input_snapshot.hash_for(task_path),
                "assets_config_sha256": input_snapshot.hash_for(assets_path),
                "runtime_config_sha256": input_snapshot.hash_for(runtime_path),
                "build_report_sha256": input_snapshot.hash_for(build_report_path),
                "source_sha256": input_snapshot.source_sha256,
                "generated_asset_sha256": input_snapshot.generated_asset_sha256,
                "input_immutability_verified": True,
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
        if hasattr(env, "rj45_task_report"):
            report["task_model"] = env.rj45_task_report()
        atomic_write_json(args.report, report)
        input_snapshot.assert_unchanged()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return 0
    finally:
        wrapped.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as error:
        traceback.print_exc()
        sys.stderr.flush()
        try:
            _write_evaluation_failure_report(error)
        except Exception:
            print("Failed to write evaluation failure report", file=sys.stderr, flush=True)
            traceback.print_exc()
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
