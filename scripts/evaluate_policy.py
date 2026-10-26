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


def _preimport_option(name: str, default: str | None) -> str | None:
    prefix = name + "="
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 1 < len(values):
            return values[index + 1]
    return default


def _project_path(value: str) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else PROJECT_ROOT / path).resolve()


def _json_dependency_paths(path_value: str | None, path_keys: tuple[str, ...]) -> tuple[Path, ...]:
    if path_value is None:
        return ()
    path = _project_path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    dependencies = [path]
    for key in path_keys:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"JSON input {path} must identify {key}")
        dependencies.append(_project_path(value))
    return tuple(dependencies)


def _fixed_environment_episode_quotas(total_episodes: int, num_envs: int) -> tuple[int, ...]:
    """Distribute a fixed episode budget by environment index."""

    if type(total_episodes) is not int or total_episodes <= 0:
        raise ValueError("total_episodes must be a positive integer")
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError("num_envs must be a positive integer")
    quotient, remainder = divmod(total_episodes, num_envs)
    return tuple(quotient + int(env_id < remainder) for env_id in range(num_envs))


def _environment_episode_is_within_quota(
    env_id: int,
    quotas: tuple[int, ...],
    accepted_counts: tuple[int, ...] | list[int],
) -> bool:
    """Return whether the next completed episode from one environment is admissible."""

    if type(env_id) is not int or env_id < 0 or env_id >= len(quotas):
        raise ValueError("env_id is outside the quota table")
    if len(accepted_counts) != len(quotas):
        raise ValueError("accepted_counts must match the quota table")
    if any(type(value) is not int or value < 0 for value in (*quotas, *accepted_counts)):
        raise ValueError("quotas and accepted counts must be non-negative integers")
    if accepted_counts[env_id] > quotas[env_id]:
        raise ValueError("accepted count already exceeds its environment quota")
    return accepted_counts[env_id] < quotas[env_id]


def _rj45_terminal_attempt_metrics(snapshot, env_id: int) -> tuple[dict[str, object], bool]:
    """Extract one environment's mandatory terminal metrics without inventing values."""

    metric_sources = (
        ("elapsed_s", "timestamp_s", float),
        ("retry_count", "retry_count", int),
        ("grip_contact_force_peak_n", "grip_contact_force_peak_n", float),
        ("insertion_contact_force_peak_n", "insertion_contact_force_peak_n", float),
        ("insertion_contact_load_peak_n", "insertion_contact_load_peak_n", float),
        ("forbidden_contact_load_peak_n", "forbidden_contact_load_peak_n", float),
        ("final_depth_m", "actual_depth_m", float),
        ("final_lateral_error_m", "actual_lateral_error_m", float),
        ("final_orientation_error_rad", "actual_orientation_error_rad", float),
        ("final_dwell_s", "independent_eval_dwell_s", float),
    )
    metrics: dict[str, object] = {}
    all_finite = True
    for output_name, snapshot_key, output_type in metric_sources:
        row_value = snapshot[snapshot_key][env_id]
        if row_value.numel() != 1:
            raise ValueError(f"terminal metric {snapshot_key} must be scalar per environment")
        numeric_value = float(row_value.item())
        if math.isfinite(numeric_value):
            metrics[output_name] = output_type(numeric_value)
        else:
            metrics[output_name] = None
            all_finite = False
    return metrics, all_finite


def _configure_verified_parallel_rj45_task(
    cfg,
    task,
    project_root,
    *,
    parallel_specification,
    verified_parallel_contact_evidence,
    configure_rj45_task,
):
    """Bind a verified parallel specification before task configuration."""

    if parallel_specification is None:
        if verified_parallel_contact_evidence is not None:
            raise ValueError("parallel contact proof has no specification")
    else:
        if verified_parallel_contact_evidence is None:
            raise ValueError("parallel contact specification was not verified")
        cfg.rj45_parallel_contact_evidence = parallel_specification
    return configure_rj45_task(cfg, task, project_root)


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
_bootstrap_training_config = _preimport_option(
    "--rj45-training-config", "configs/rj45-training-v1.json"
)
_bootstrap_held_state = _preimport_option("--rj45-held-state", None)
_bootstrap_held_paths: tuple[Path, ...] = ()
if _bootstrap_task == "rj45" and _bootstrap_held_state is not None:
    held_path = _project_path(_bootstrap_held_state)
    held_payload = json.loads(held_path.read_text(encoding="utf-8"))
    held_provenance = held_payload.get("provenance") if isinstance(held_payload, dict) else None
    if not isinstance(held_provenance, dict):
        raise ValueError("RJ45 held state must contain provenance")
    _bootstrap_held_paths = (
        held_path,
        _project_path(held_provenance["nominal_report_path"]),
        _project_path(held_provenance["nominal_trace_path"]),
    )
_bootstrap_parallel_spec = _preimport_option("--parallel-contact-evidence", None)
_bootstrap_parallel_paths = (
    _json_dependency_paths(
        _bootstrap_parallel_spec, ("report_path", "trace_path", "sat_report_path")
    )
    if _bootstrap_task == "rj45" and _bootstrap_parallel_spec is not None
    else ()
)
_bootstrap_input_snapshot = capture_run_inputs(
    PROJECT_ROOT,
    task_path=_bootstrap_task_path,
    build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
    config_paths=(
        Path(_preimport_option("--config", _bootstrap_config_default)),
        *((Path(_bootstrap_training_config),) if _bootstrap_task == "rj45" else ()),
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
        *_bootstrap_held_paths,
        *_bootstrap_parallel_paths,
    ),
)

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Evaluate RB-Y1 assembly policy")
parser.add_argument("checkpoint", type=Path, nargs="?")
parser.add_argument("--num-envs", type=int, default=None)
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
parser.add_argument(
    "--rj45-training-config",
    type=Path,
    default=Path("configs/rj45-training-v1.json"),
)
parser.add_argument("--rj45-held-state", type=Path)
parser.add_argument("--parallel-contact-evidence", type=Path)
parser.add_argument(
    "--baseline-mode",
    choices=("learned", "biased_zero", "oracle"),
    default="learned",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.task == "rj45" and args.config == Path("configs/ppo.json"):
    args.config = Path("configs/ppo-rj45-v1.json")
if args.num_envs is None:
    args.num_envs = 1 if args.task == "rj45" else 64
if args.num_envs <= 0 or args.episodes <= 0:
    parser.error("--num-envs and --episodes must be positive")
if (
    args.task == "rj45"
    and args.num_envs > 1
    and args.parallel_contact_evidence is None
):
    parser.error("parallel RJ45 evaluation requires --parallel-contact-evidence")
if args.task == "rj45" and args.rj45_nominal_evidence is None:
    parser.error("--task rj45 requires --rj45-nominal-evidence")
if args.seed is not None and args.seed < 0:
    parser.error("--seed must be non-negative")
if args.task == "toy" and args.checkpoint is None:
    parser.error("toy evaluation requires a checkpoint")
if args.task == "rj45" and args.baseline_mode == "learned" and args.checkpoint is None:
    parser.error("learned RJ45 evaluation requires a checkpoint")
if args.task == "rj45" and args.baseline_mode == "learned" and args.rj45_held_state is None:
    parser.error("learned RJ45 evaluation requires --rj45-held-state")
if args.checkpoint is not None and not args.checkpoint.is_file():
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
    learned_mode = args.task == "toy" or args.baseline_mode == "learned"
    lineage_path = (
        args.checkpoint.parent / "training_provenance.json"
        if args.task == "rj45" and learned_mode
        else None
    )
    manifest_path = (
        args.checkpoint.parent / "checkpoint_manifest.json"
        if args.task == "rj45" and learned_mode
        else None
    )
    nominal_trace_path = None
    extra_input_paths: list[Path] = []
    if args.task == "rj45":
        nominal_report = load_finite_json(args.rj45_nominal_evidence)
        raw_trace_path = nominal_report.get("trace_path")
        if not isinstance(raw_trace_path, str) or not raw_trace_path:
            raise ValueError("RJ45 nominal evidence report must identify its trace_path")
        nominal_trace_path = Path(raw_trace_path)
        if not nominal_trace_path.is_absolute():
            nominal_trace_path = PROJECT_ROOT / nominal_trace_path
        extra_input_paths.extend((args.rj45_nominal_evidence, nominal_trace_path))
        if args.rj45_held_state is not None:
            held_payload = load_finite_json(args.rj45_held_state)
            held_provenance = held_payload.get("provenance")
            if not isinstance(held_provenance, dict):
                raise ValueError("RJ45 held state must contain provenance")
            extra_input_paths.extend(
                (
                    args.rj45_held_state,
                    _project_path(held_provenance["nominal_report_path"]),
                    _project_path(held_provenance["nominal_trace_path"]),
                )
            )
        if args.parallel_contact_evidence is not None:
            parallel_spec_payload = load_finite_json(args.parallel_contact_evidence)
            extra_input_paths.extend(
                _json_dependency_paths(
                    str(args.parallel_contact_evidence),
                    ("report_path", "trace_path", "sat_report_path"),
                )
            )
        else:
            parallel_spec_payload = None
    if learned_mode:
        extra_input_paths.append(args.checkpoint)
    if args.task == "rj45" and learned_mode:
        extra_input_paths.extend((lineage_path, manifest_path))
    input_snapshot = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(
            args.config,
            *((args.rj45_training_config,) if args.task == "rj45" else ()),
            assets_path,
            runtime_path,
            *((connector_assets_path,) if args.task == "rj45" else ()),
        ),
        source_roots=((PROJECT_ROOT / "src", True),),
        source_files=(PROJECT_ROOT / "scripts/evaluate_policy.py",),
        extra_paths=(
            PROJECT_ROOT / "scripts/train.py",
            PROJECT_ROOT / "scripts/run_sim.py",
            *extra_input_paths,
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
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    apply_runtime_timing(cfg, input_snapshot.json_for(PROJECT_ROOT / "configs/runtime.json"))
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env_type = Rby1AssemblyEnv
    selected_task = input_snapshot.json_for(task_path)
    verified_checkpoint_evidence = None
    held_state_proof = None
    parallel_contact_evidence = None
    cfg.seed = (
        args.seed
        if args.seed is not None
        else int(selected_task["data"]["evaluation_seed"])
    )
    if args.task == "rj45":
        from rby1_assembly.rj45_env import configure_rj45_task, load_rj45_task
        from rby1_assembly.rj45_training import load_training_config
        from rby1_assembly.rj45_training_env import (
            _calibration_bias_case_seed,
            make_rj45_training_env_type,
        )
        from rby1_assembly.held_grasp import load_verified_held_grasp_artifact
        from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence
        from rby1_assembly.rj45_task import (
            verify_checkpoint_lineage,
            verify_checkpoint_manifest,
            verify_generated_asset_hashes,
            verify_runtime_versions,
        )

        task = load_rj45_task(task_path)
        training_config = load_training_config(args.rj45_training_config)
        if args.rj45_held_state is not None:
            _, held_state_proof = load_verified_held_grasp_artifact(
                PROJECT_ROOT, task, args.rj45_held_state
            )
        if parallel_spec_payload is not None:
            parallel_contact_evidence = load_parallel_contact_evidence(
                parallel_spec_payload, project_root=PROJECT_ROOT
            )
        if learned_mode:
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
            trained_num_envs = lineage.get("num_envs")
            if type(trained_num_envs) is not int or trained_num_envs <= 0:
                raise ValueError("checkpoint num_envs must be a positive integer")
            if trained_num_envs > 1 and parallel_contact_evidence is None:
                raise ValueError(
                    "parallel-trained checkpoint evaluation requires --parallel-contact-evidence"
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
        if learned_mode:
            action_dimension = 6 if training_config["action_mode"] == "right_only_6d" else 12
            training_contract = {
                "schema_version": 1,
                "policy_observation_schema_version": 1,
                "policy_observation_dimension": 79,
                "policy_action_dimension": action_dimension,
                "action_mode": training_config["action_mode"],
                "episode_scope": training_config["episode_scope"],
                "warmup_s": training_config["warmup_s"],
                "active_horizon_s": training_config["active_horizon_s"],
                "estimator": "frozen_kinematic_grasp_estimator",
                "calibration_bias": "episode_fixed_one_shot_se3",
                "calibration_bias_level": training_config["calibration_bias"]["initial_level"],
                "policy_pose_source": "joint_encoder_and_frozen_grasp_transform",
                "ground_truth_usage": "scripted_setup_control_and_safety_then_reward_and_independent_evaluation_only",
                "held_grasp_reset_proof": held_state_proof,
                "training_config_sha256": training_config["_config_sha256"],
                "parallel_contact_evidence": (
                    parallel_contact_evidence if lineage["num_envs"] > 1 else None
                ),
                "learned_recovery": False,
                "retract_controller": "baseline_only",
            }
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
                    "baseline_controller_state_source": "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias",
                    "task_config_path": str(task_path.relative_to(PROJECT_ROOT)),
                    "policy_config_path": _lineage_path_key(args.config),
                    "collision_monitor_mode": lineage["collision_monitor_mode"],
                    "num_envs": lineage["num_envs"],
                    "rj45_training_contract": training_contract,
                },
                rj45_nominal_evidence=current_nominal_evidence,
            )
            verified_checkpoint_evidence = {
                "training_provenance_sha256": input_snapshot.hash_for(lineage_path),
                "checkpoint_manifest_sha256": input_snapshot.hash_for(manifest_path),
                "runtime_contract_verified_at_start": True,
                "rj45_nominal_evidence": current_nominal_evidence,
                "held_grasp_reset_proof": held_state_proof,
                "training_num_envs": lineage["num_envs"],
            }
        env_type = _configure_verified_parallel_rj45_task(
            cfg,
            task,
            PROJECT_ROOT,
            parallel_specification=parallel_spec_payload,
            verified_parallel_contact_evidence=parallel_contact_evidence,
            configure_rj45_task=configure_rj45_task,
        )
        env_type = make_rj45_training_env_type(
            env_type,
            training_config,
            held_state=None,
            calibration_bias_enabled=args.baseline_mode != "oracle",
            bias_level=training_config["calibration_bias"]["initial_level"],
        )
    input_snapshot.assert_unchanged()
    env = env_type(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get("clip_actions", 1.0))
    try:
        if learned_mode:
            runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=None, device=env.device)
            runner.load(str(args.checkpoint))
            policy = runner.get_inference_policy(device=env.device)
        else:
            action_dimension = 6 if training_config["action_mode"] == "right_only_6d" else 12
            policy = lambda observation: torch.zeros(  # noqa: E731
                (observation.shape[0], action_dimension),
                dtype=observation.dtype,
                device=observation.device,
            )
        observations = wrapped.get_observations()
        attempts = 0
        episode_quotas = (
            _fixed_environment_episode_quotas(args.episodes, args.num_envs)
            if args.task == "rj45"
            else None
        )
        accepted_attempts_per_environment = (
            [0 for _ in range(args.num_envs)] if episode_quotas is not None else None
        )
        valid_successes = 0
        valid_failures = 0
        invalid_episodes = 0
        invalid_reasons: dict[str, int] = {}
        force_peaks = []
        elapsed_times_s = []
        retry_counts = []
        recovery_attempts = 0
        recovered_successes = 0
        episode_outcomes = []
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
                if episode_quotas is not None and not _environment_episode_is_within_quota(
                    env_id,
                    episode_quotas,
                    accepted_attempts_per_environment,
                ):
                    continue
                attempts += 1
                if accepted_attempts_per_environment is not None:
                    accepted_attempts_per_environment[env_id] += 1
                snapshot = extras["terminal_snapshot"]
                force_peak = float(extras["evaluation"]["contact_force_proxy_peak_n"][env_id].item())
                elapsed = float(snapshot["timestamp_s"][env_id].item())
                retries_value = float(snapshot["retry_count"][env_id].item())
                finite = all(math.isfinite(value) for value in (force_peak, elapsed, retries_value))
                finite = finite and all(bool(torch.isfinite(snapshot[key][env_id]).all().item()) for key in
                    ("object_poses_world_wxyz", "revolute_joint_positions_rad", "actual_depth_m",
                     "actual_lateral_error_m", "actual_orientation_error_rad"))
                bias_finite = True
                rj45_attempt_metrics = None
                if args.task == "rj45":
                    rj45_attempt_metrics, rj45_metrics_finite = (
                        _rj45_terminal_attempt_metrics(snapshot, env_id)
                    )
                    finite = finite and rj45_metrics_finite
                    bias_finite = all(
                        bool(torch.isfinite(snapshot[key][env_id]).all().item())
                        for key in (
                            "calibration_bias_translation_entry_m",
                            "calibration_bias_rotation_vector_entry_rad",
                        )
                    )
                    finite = finite and bias_finite
                    finite = finite and int(snapshot["episode_index"][env_id].item()) >= 0
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
                if args.task == "rj45":
                    episode_index = int(snapshot["episode_index"][env_id].item())
                    episode_outcomes.append(
                        {
                            "attempt_index": attempts - 1,
                            "environment_index": env_id,
                            "episode_index": episode_index,
                            "paired_condition": {
                                "task_config_sha256": task["_config_sha256"],
                                "base_evaluation_seed": cfg.seed,
                                "environment_index": env_id,
                                "episode_index": episode_index,
                                "calibration_bias_level": training_config[
                                    "calibration_bias"
                                ]["initial_level"],
                                "calibration_latent_case_seed": (
                                    _calibration_bias_case_seed(
                                        cfg.seed, env_id, episode_index
                                    )
                                    if episode_index >= 0
                                    else None
                                ),
                                "calibration_latent_case_seed_formula": (
                                    "base_seed + stream_offset + episode_index * "
                                    "episode_stride + env_id"
                                ),
                            },
                            "calibration_bias_translation_entry_m": (
                                snapshot["calibration_bias_translation_entry_m"][env_id]
                                .detach().cpu().tolist() if bias_finite else None
                            ),
                            "calibration_bias_rotation_vector_entry_rad": (
                                snapshot["calibration_bias_rotation_vector_entry_rad"][env_id]
                                .detach().cpu().tolist() if bias_finite else None
                            ),
                            "valid": valid,
                            "success": succeeded,
                            "termination_reason": str(
                                extras["terminal_evaluation"]["termination_reason"][env_id]
                            ),
                            **rj45_attempt_metrics,
                        }
                    )
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
            "evaluation_num_envs": cfg.scene.num_envs,
            "observation_source": (
                "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
                if args.task == "rj45" and args.baseline_mode != "oracle"
                else "joint_encoder_frozen_grasp_estimator_without_calibration_bias"
                if args.task == "rj45"
                else "simulator_state_with_noise"
            ),
            "baseline_controller_state_source": (
                "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
                if args.task == "rj45" and args.baseline_mode != "oracle"
                else "joint_encoder_frozen_grasp_estimator_without_calibration_bias"
                if args.task == "rj45"
                else "simulator_ground_truth"
            ),
            "completion_source": "simulator_ground_truth_evaluator",
            "sample_definition": "terminated_or_truncated_episode_attempt",
            "provenance": {
                "checkpoint": str(args.checkpoint.resolve()) if learned_mode else None,
                "checkpoint_sha256": input_snapshot.hash_for(args.checkpoint) if learned_mode else None,
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
        if args.task == "rj45":
            report["baseline_mode"] = args.baseline_mode
            report["per_environment_episode_quota"] = list(episode_quotas)
            report["per_environment_accepted_attempt_count"] = (
                accepted_attempts_per_environment
            )
            report["parallel_contact_evidence"] = (
                parallel_contact_evidence if args.num_envs > 1 else None
            )
            report["rj45_training_config"] = {
                "path": _lineage_path_key(args.rj45_training_config),
                "sha256": training_config["_config_sha256"],
                "action_mode": training_config["action_mode"],
                "observation_dimension": training_config["observation_dim"],
                "calibration_bias_level": training_config["calibration_bias"]["initial_level"],
            }
            report["episode_outcomes"] = episode_outcomes
            report["paired_condition_key"] = (
                "task_config_sha256,base_evaluation_seed,environment_index,episode_index,"
                "calibration_bias_level,calibration_latent_case_seed,"
                "calibration_latent_case_seed_formula"
            )
            report["learned_recovery"] = False
        if hasattr(env, "rj45_task_report"):
            report["task_model"] = env.rj45_task_report()
        if hasattr(env, "rj45_training_report"):
            report["training_environment"] = env.rj45_training_report()
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
