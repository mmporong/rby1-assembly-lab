# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the RJ45 residual policy with the real RSL-RL PPO runner."""

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.run_provenance import InputMutationError, capture_run_inputs  # noqa: E402
from rby1_assembly.held_grasp import (  # noqa: E402
    bank_source_contract_sha256,
    held_grasp_bank_dependency_paths,
    load_verified_held_grasp_bank,
)
from rby1_assembly.training_telemetry import (  # noqa: E402
    make_action_telemetry_wrapper_type,
    observation_noise_contract,
)
from rby1_assembly.parallel_training_video import (  # noqa: E402
    ParallelTrainingVideoRecorder,
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


def _initialize_verified_bc_actor(
    runner, artifact_path, expected_sha256, input_snapshot,
    task_path, training_config, bank_proof,
):
    """Bind actor bytes and their lineage to the frozen training inputs."""
    from rby1_assembly.rj45_bc import initialize_runner_actor, load_bc_actor_artifact

    snapshot_sha = input_snapshot.hash_for(artifact_path)
    if snapshot_sha != expected_sha256:
        raise ValueError("BC expected SHA-256 differs from the frozen input")
    artifact = load_bc_actor_artifact(artifact_path, expected_sha256=snapshot_sha)
    provenance = artifact["provenance"].as_dict()
    expected = {
        "source_sha256": bank_source_contract_sha256(bank_proof),
        "task_sha256": input_snapshot.hash_for(task_path),
        "training_sha256": training_config["_config_sha256"],
        "bank_sha256": bank_proof["manifest_sha256"],
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("BC initialization source/task/training/bank lineage differs")
    result = initialize_runner_actor(runner, artifact_path, expected_sha256=snapshot_sha)
    if result["artifact_file_sha256"] != snapshot_sha or artifact["file_sha256"] != snapshot_sha:
        raise ValueError("loaded BC actor bytes differ from the frozen input")
    input_snapshot.assert_unchanged()
    result.update({
        "mode": "bc_actor_warm_start", "output_dimension": 6,
        "bc_provenance": provenance,
        "bc_artifact_sha256": result["artifact_file_sha256"],
    })
    return result


_bootstrap_task = _preimport_option("--task", "toy")
_bootstrap_config_default = (
    "configs/ppo-rj45-v1.json" if _bootstrap_task == "rj45" else "configs/ppo.json"
)
_bootstrap_task_path = PROJECT_ROOT / (
    "configs/tasks/rj45-v1.json" if _bootstrap_task == "rj45" else "configs/task.json"
)
_bootstrap_nominal_report = _preimport_option("--rj45-nominal-evidence", None)
_bootstrap_nominal_paths: tuple[Path, ...] = ()
if _bootstrap_task == "rj45" and _bootstrap_nominal_report is not None:
    nominal_report_path = Path(_bootstrap_nominal_report).resolve()
    nominal_report = json.loads(nominal_report_path.read_text(encoding="utf-8"))
    nominal_trace = nominal_report.get("trace_path")
    if not isinstance(nominal_trace, str) or not nominal_trace:
        raise ValueError("RJ45 nominal evidence report must identify its trace_path")
    nominal_trace_path = Path(nominal_trace)
    if not nominal_trace_path.is_absolute():
        nominal_trace_path = PROJECT_ROOT / nominal_trace_path
    _bootstrap_nominal_paths = (nominal_report_path, nominal_trace_path)
_bootstrap_training_config = _preimport_option(
    "--rj45-training-config", "configs/rj45-training-v1.json"
)
_bootstrap_held_state = _preimport_option("--rj45-held-state", None)
_bootstrap_held_bank = _preimport_option("--rj45-held-bank", None)
if _bootstrap_held_bank is not None and _bootstrap_held_state is not None:
    raise ValueError("select exactly one held state or held bank")
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
if _bootstrap_task == "rj45" and _bootstrap_held_bank is not None:
    _bootstrap_held_paths = held_grasp_bank_dependency_paths(
        PROJECT_ROOT, Path(_bootstrap_held_bank), "train"
    )
_bootstrap_bc_actor = _preimport_option("--bc-actor-init", None)
_bootstrap_parallel_spec = _preimport_option("--parallel-contact-evidence", None)
_bootstrap_parallel_paths = (
    _json_dependency_paths(
        _bootstrap_parallel_spec,
        ("report_path", "trace_path", "sat_report_path"),
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
        *sorted((PROJECT_ROOT / "configs").glob("*.json")),
        *((PROJECT_ROOT / "configs/rj45-assets.json",) if _bootstrap_task == "rj45" else ()),
    ),
    source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
    source_files=(Path(__file__).resolve(),),
    extra_paths=(
        *_bootstrap_nominal_paths,
        *_bootstrap_held_paths,
        *_bootstrap_parallel_paths,
        *((_project_path(_bootstrap_bc_actor),) if _bootstrap_bc_actor is not None else ()),
    ),
)

from rby1_assembly.experiment_contracts import (  # noqa: E402
    atomic_write_json,
    load_finite_json,
    validate_ppo_config,
    verify_rj45_nominal_evidence,
)
from rby1_assembly.rj45_task import (  # noqa: E402
    load_rj45_task,
    verify_runtime_versions,
)
from rby1_assembly.held_grasp import load_verified_held_grasp_artifact  # noqa: E402
from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence  # noqa: E402
from rby1_assembly.rj45_training import (  # noqa: E402
    action_application_contract_metadata,
    configure_actor_output_initialization,
    load_training_config,
    policy_ground_truth_usage,
    policy_observation_dimension,
    policy_observation_schema_version,
    privileged_diagnostic_observation_metadata,
    reward_profile_metadata,
    validate_reward_profile_runner,
    validate_reward_horizon_contract,
)
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Train RB-Y1 assembly residual policy")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--task", choices=("toy", "rj45"), default="toy")
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--config", type=Path, default=Path("configs/ppo.json"))
parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/rby1_assembly"))
parser.add_argument(
    "--rj45-nominal-evidence",
    type=Path,
    help="completed single-environment nominal RJ45 run report required for RJ45 training",
)
parser.add_argument(
    "--rj45-training-config",
    type=Path,
    default=Path("configs/rj45-training-v1.json"),
    help="RJ45 estimator, action, bias, reward, and held-reset training contract",
)
parser.add_argument(
    "--rj45-held-state",
    type=Path,
    help="verified held-grasp state generated from the current nominal run",
)
parser.add_argument("--rj45-held-bank", type=Path, help="verified bank; training opens only its train split")
parser.add_argument("--bc-actor-init", type=Path, help="verified actor-only BC initialization artifact")
parser.add_argument("--bc-actor-init-sha256", help="expected SHA-256 of the BC initialization artifact")
parser.add_argument(
    "--parallel-contact-evidence",
    type=Path,
    help="JSON specification pinning report, trace, and SAT proof for parallel contact routing",
)
parser.add_argument(
    "--zero-init-actor-output",
    action="store_true",
    help=(
        "RJ45 experiment option: zero only the final deterministic actor mean "
        "Linear layer; Gaussian policy noise remains configured by PPO"
    ),
)
parser.add_argument(
    "--parallel-video",
    type=Path,
    help="record a bounded raw-camera clip during real vectorized PPO rollout steps",
)
parser.add_argument("--parallel-video-seconds", type=float, default=16.0)
parser.add_argument("--parallel-video-fps", type=int, default=60)
parser.add_argument(
    "--parallel-video-view", choices=("overview", "closeup"), default="overview"
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.task == "rj45" and args.config == Path("configs/ppo.json"):
    args.config = Path("configs/ppo-rj45-v1.json")
if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args.task == "rj45" and args.rj45_nominal_evidence is None:
    parser.error("--rj45-nominal-evidence is required for --task rj45")
if args.task == "rj45" and (args.rj45_held_state is None) == (args.rj45_held_bank is None):
    parser.error("RJ45 requires exactly one of --rj45-held-state or --rj45-held-bank")
if args.task != "rj45" and args.rj45_held_bank is not None:
    parser.error("--rj45-held-bank requires --task rj45")
if (args.bc_actor_init is None) != (args.bc_actor_init_sha256 is None):
    parser.error("BC initialization requires both artifact path and SHA-256")
if args.bc_actor_init is not None and (args.task != "rj45" or args.rj45_held_bank is None or args.zero_init_actor_output):
    parser.error("BC initialization requires an RJ45 train bank and cannot be combined with zero initialization")
if args.task == "rj45" and args.num_envs > 1 and args.parallel_contact_evidence is None:
    parser.error("parallel RJ45 training requires --parallel-contact-evidence")
if args.zero_init_actor_output and args.task != "rj45":
    parser.error("--zero-init-actor-output is only supported for --task rj45")
if args.parallel_video is not None:
    if args.task != "rj45":
        parser.error("--parallel-video is only supported for --task rj45")
    if args.num_envs < 2:
        parser.error("--parallel-video requires at least two simultaneous environments")
    if not getattr(args, "enable_cameras", False):
        parser.error("--parallel-video requires --enable_cameras")
    if (
        not math.isfinite(args.parallel_video_seconds)
        or args.parallel_video_seconds <= 0.0
    ):
        parser.error("--parallel-video-seconds must be finite and positive")
    if args.parallel_video_fps <= 0:
        parser.error("--parallel-video-fps must be positive")
try:
    task_path = PROJECT_ROOT / (
        "configs/tasks/rj45-v1.json" if args.task == "rj45" else "configs/task.json"
    )
    extra_input_paths: list[Path] = []
    if args.task == "rj45":
        nominal_report = load_finite_json(args.rj45_nominal_evidence)
        nominal_trace = nominal_report.get("trace_path")
        if not isinstance(nominal_trace, str) or not nominal_trace:
            raise ValueError("RJ45 nominal evidence report must identify its trace_path")
        nominal_trace_path = Path(nominal_trace)
        if not nominal_trace_path.is_absolute():
            nominal_trace_path = PROJECT_ROOT / nominal_trace_path
        extra_input_paths.extend((args.rj45_nominal_evidence, nominal_trace_path))
        if args.rj45_held_bank is not None:
            extra_input_paths.extend(held_grasp_bank_dependency_paths(
                PROJECT_ROOT, args.rj45_held_bank, "train"
            ))
        else:
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
        if args.bc_actor_init is not None:
            extra_input_paths.append(args.bc_actor_init)
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
    input_snapshot = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(
            args.config,
            *((args.rj45_training_config,) if args.task == "rj45" else ()),
            *sorted((PROJECT_ROOT / "configs").glob("*.json")),
        ),
        source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
        source_files=(Path(__file__).resolve(),),
        extra_paths=tuple(extra_input_paths),
    )
    _bootstrap_input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    if args.task == "rj45":
        captured_nominal_report = input_snapshot.json_for(args.rj45_nominal_evidence)
        if captured_nominal_report.get("trace_path") != nominal_trace:
            raise ValueError("RJ45 nominal evidence changed while its trace was discovered")
        captured_trace_path = Path(captured_nominal_report["trace_path"])
        if not captured_trace_path.is_absolute():
            captured_trace_path = PROJECT_ROOT / captured_trace_path
        if captured_trace_path.resolve() != nominal_trace_path.resolve():
            raise ValueError("RJ45 nominal evidence trace path changed during startup")
    runner_cfg = input_snapshot.json_for(args.config)
    if args.max_iterations is not None:
        runner_cfg["max_iterations"] = args.max_iterations
    validate_ppo_config(runner_cfg)
    runtime_versions = {
        name: metadata.version(name)
        for name in (
            "isaacsim",
            "isaaclab",
            "isaaclab-rl",
            "rsl-rl-lib",
            "torch",
            "numpy",
            "h5py",
        )
    }
    verify_runtime_versions(
        runtime_versions, input_snapshot.json_for(PROJECT_ROOT / "configs/runtime.json")
    )
    selected_task = (
        load_rj45_task(task_path)
        if args.task == "rj45"
        else input_snapshot.json_for(task_path)
    )
    nominal_evidence = (
        verify_rj45_nominal_evidence(
            PROJECT_ROOT, selected_task, args.rj45_nominal_evidence
        )
        if args.task == "rj45"
        else None
    )
    if args.task == "rj45":
        training_config = load_training_config(args.rj45_training_config)
        validate_reward_profile_runner(training_config, runner_cfg)
        held_bank = held_bank_proof = None
        if args.rj45_held_bank is not None:
            held_bank, held_bank_proof = load_verified_held_grasp_bank(
                PROJECT_ROOT, selected_task, args.rj45_held_bank, "train"
            )
            held_state, held_state_proof = None, held_bank_proof
        else:
            held_state, held_state_proof = load_verified_held_grasp_artifact(
                PROJECT_ROOT, selected_task, args.rj45_held_state
            )
        parallel_contact_evidence = (
            load_parallel_contact_evidence(
                parallel_spec_payload, project_root=PROJECT_ROOT
            )
            if parallel_spec_payload is not None
            else None
        )
        if args.num_envs > 1 and parallel_contact_evidence is None:
            raise ValueError("parallel RJ45 training requires verified contact evidence")
    else:
        training_config = held_state = held_state_proof = parallel_contact_evidence = None
        held_bank = held_bank_proof = None
except (OSError, ValueError, TypeError, KeyError, metadata.PackageNotFoundError) as error:
    parser.error(f"invalid training configuration: {error}")
launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import omni.kit.app  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402

ActionTelemetryRslRlVecEnvWrapper = make_action_telemetry_wrapper_type(
    RslRlVecEnvWrapper
)


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    apply_runtime_timing(cfg, input_snapshot.json_for(PROJECT_ROOT / "configs/runtime.json"))
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = runner_cfg["seed"]
    if args.parallel_video is not None:
        render_stride = round(1.0 / (cfg.sim.dt * args.parallel_video_fps))
        if render_stride < 1 or not math.isclose(
            render_stride * cfg.sim.dt * args.parallel_video_fps,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("parallel video FPS must divide the physics frequency")
        cfg.sim.render_interval = render_stride
        cfg.viewer.resolution = (
            (1280, 720)
            if args.parallel_video_view == "overview"
            else (960, 720)
        )
    if args.task == "rj45":
        validate_reward_horizon_contract(
            training_config, is_finite_horizon=cfg.is_finite_horizon
        )
    env_type = Rby1AssemblyEnv
    if args.task == "rj45":
        from rby1_assembly.rj45_env import configure_rj45_task
        from rby1_assembly.rj45_training_env import make_rj45_training_env_type

        if parallel_spec_payload is not None:
            cfg.rj45_parallel_contact_evidence = parallel_spec_payload
        env_type = configure_rj45_task(cfg, selected_task, PROJECT_ROOT)
        env_type = make_rj45_training_env_type(
            env_type,
            training_config,
            held_state=held_state,
            held_bank=held_bank,
            held_bank_proof=held_bank_proof,
            calibration_bias_enabled=True,
            bias_level=training_config["calibration_bias"]["initial_level"],
        )
    _bootstrap_input_snapshot.assert_unchanged()
    input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    env = env_type(
        cfg, render_mode="rgb_array" if args.parallel_video is not None else None
    )
    parallel_video = (
        ParallelTrainingVideoRecorder(
            env,
            video_path=args.parallel_video,
            seconds=args.parallel_video_seconds,
            fps=args.parallel_video_fps,
            view=args.parallel_video_view,
            rollout_steps=runner_cfg["num_steps_per_env"],
        )
        if args.parallel_video is not None
        else None
    )
    wrapped = ActionTelemetryRslRlVecEnvWrapper(
        env,
        clip_actions=runner_cfg.get("clip_actions", 1.0),
        step_observer=parallel_video,
    )
    selected_log_root = (
        args.log_root / "rj45-v1" if args.task == "rj45" else args.log_root
    )
    log_dir = selected_log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(log_dir / "runner_config.json", runner_cfg)
    try:
        source_paths = sorted((PROJECT_ROOT / "src/rby1_assembly").glob("*.py"))
        evidence_paths = source_paths + [Path(__file__).resolve(), args.config.resolve()]
        evidence_paths += sorted((PROJECT_ROOT / "configs").glob("*.json"))
        generated_dir = (
            (PROJECT_ROOT / selected_task["assets"]["build_report"]).parent
            if args.task == "rj45"
            else PROJECT_ROOT / "assets/generated"
        )
        evidence_paths += sorted(generated_dir.glob("*.usda"))
        evidence_paths += [
            generated_dir / "build-report.json",
            task_path,
            log_dir / "runner_config.json",
        ]
        lineage = {
            "schema_version": 1,
            "source_and_config_sha256": {
                str(path.resolve().relative_to(PROJECT_ROOT))
                if path.resolve().is_relative_to(PROJECT_ROOT)
                else str(path.resolve()): (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    if path == log_dir / "runner_config.json"
                    else input_snapshot.hash_for(path)
                )
                for path in evidence_paths
            },
            "runtime_versions": runtime_versions,
            "seed": cfg.seed,
            "selected_device": str(args.device),
            "app_launcher_gpu_contract": {
                "multi_gpu": False,
                "cuda_visible_devices_modified": False,
            },
            "num_envs": cfg.scene.num_envs,
            "task_id": "rj45-v1" if args.task == "rj45" else "toy",
            "collision_monitor_mode": (
                "passive_parallel_native_ordering_verified"
                if args.task == "rj45" and args.num_envs > 1
                else "passive_nominal_one_environment" if args.task == "rj45" else "disabled"
            ),
            "task_config_path": str(task_path.relative_to(PROJECT_ROOT)),
            "policy_config_path": (
                str(args.config.resolve().relative_to(PROJECT_ROOT))
                if args.config.resolve().is_relative_to(PROJECT_ROOT)
                else str(args.config.resolve())
            ),
            "task_config_sha256": input_snapshot.hash_for(task_path),
            "physics_dt_s": cfg.sim.dt,
            "control_decimation": cfg.decimation,
            "control_dt_s": cfg.sim.dt * cfg.decimation,
            "observation_noise": observation_noise_contract(
                task=args.task,
                configured_position_std_m=cfg.observation_position_noise_std_m,
                configured_rotation_vector_std_rad=(
                    cfg.observation_rotation_noise_std_rad
                ),
                calibration_bias=(
                    training_config["calibration_bias"]
                    if args.task == "rj45"
                    else None
                ),
                bias_level=(
                    training_config["calibration_bias"]["initial_level"]
                    if args.task == "rj45"
                    else None
                ),
                calibration_bias_enabled=True,
            ),
            "ppo_action_telemetry": {
                "wrapper_clip_actions": runner_cfg.get("clip_actions", 1.0),
                "scopes": ["all_rows", "policy_active_rows"],
                "statistics": [
                    "element_clip_fraction",
                    "abs_max",
                    "mean",
                    "std",
                ],
                "environment_vector_norm_projection": (
                    "separate_environment_action_application_telemetry"
                ),
                "full_action_time_series_recorded": False,
            },
            "baseline_controller_state_source": (
                "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
                if args.task == "rj45"
                else "simulator_ground_truth"
            ),
            "checkpoint_scope": "checkpoints created in this run directory",
            "rj45_nominal_evidence": nominal_evidence,
        }
        if parallel_video is not None:
            lineage["parallel_training_video"] = {
                "requested": True,
                "video_path": str(parallel_video.video_path),
                "frames_path": str(parallel_video.frames_path),
                "metadata_path": str(parallel_video.metadata_path),
                "evidence_path": str(parallel_video.evidence_path),
                "view": args.parallel_video_view,
                "capture_plan": parallel_video.plan,
                "capture_scope": "real_vectorized_ppo_wrapped_step_only",
                "postprocessing_in_training_process": False,
            }
        if args.task == "rj45":
            action_dimension = 6 if training_config["action_mode"] == "right_only_6d" else 12
            observation_dimension = policy_observation_dimension(training_config)
            lineage["rj45_training_contract"] = {
                "schema_version": 1,
                "policy_observation_schema_version": policy_observation_schema_version(
                    training_config
                ),
                "policy_observation_dimension": observation_dimension,
                "policy_action_dimension": action_dimension,
                "action_mode": training_config["action_mode"],
                "episode_scope": training_config["episode_scope"],
                "warmup_s": training_config["warmup_s"],
                "active_horizon_s": training_config["active_horizon_s"],
                "estimator": "frozen_kinematic_grasp_estimator",
                "calibration_bias": "episode_fixed_one_shot_se3",
                "calibration_bias_level": training_config["calibration_bias"]["initial_level"],
                "policy_pose_source": "joint_encoder_and_frozen_grasp_transform",
                "ground_truth_usage": policy_ground_truth_usage(training_config),
                "held_grasp_reset_proof": held_state_proof,
                "training_config_sha256": training_config["_config_sha256"],
                "parallel_contact_evidence": parallel_contact_evidence,
                "learned_recovery": False,
                "retract_controller": "baseline_only",
            }
            lineage["rj45_training_contract"].update(
                privileged_diagnostic_observation_metadata(training_config)
            )
            lineage["rj45_training_contract"].update(
                action_application_contract_metadata(training_config)
            )
            if training_config["schema_version"] == 2:
                lineage["rj45_training_contract"].update(
                    reward_profile_metadata(training_config)
                )
        if args.task == "rj45":
            if args.zero_init_actor_output and action_dimension != 6:
                raise ValueError(
                    "zero-initialized RJ45 residual actor requires 6D actions"
                )
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=str(log_dir), device=env.device)
        if args.task == "rj45":
            lineage["policy_initialization"] = configure_actor_output_initialization(
                runner,
                expected_output_dim=action_dimension,
                zero_initialize=args.zero_init_actor_output,
            )
        if args.bc_actor_init is not None:
            if observation_dimension != 79 or action_dimension != 6:
                raise ValueError("BC initialization requires standard79 and right-only 6D")
            lineage["policy_initialization"] = _initialize_verified_bc_actor(
                runner, args.bc_actor_init, args.bc_actor_init_sha256, input_snapshot,
                task_path, training_config, held_bank_proof,
            )
        provenance_path = log_dir / "training_provenance.json"
        atomic_write_json(provenance_path, lineage)
        runner.add_git_repo_to_log(__file__)
        learning_completed = False
        try:
            if parallel_video is not None:
                parallel_video.bind_runner(runner)
                parallel_video.set_evidence_sources(
                    lineage["source_and_config_sha256"]
                )
                parallel_video.initialize()
            input_snapshot.assert_unchanged()
            runner.learn(
                num_learning_iterations=runner_cfg["max_iterations"],
                init_at_random_ep_len=False,
            )
            learning_completed = True
        finally:
            if parallel_video is not None:
                parallel_video.close(completed=learning_completed)
        if load_finite_json(provenance_path) != lineage:
            raise RuntimeError("training provenance changed during the experiment")
        for recorded_path, expected_sha256 in lineage["source_and_config_sha256"].items():
            evidence_path = Path(recorded_path)
            if not evidence_path.is_absolute():
                evidence_path = PROJECT_ROOT / evidence_path
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != expected_sha256:
                raise RuntimeError(
                    f"training evidence changed during the experiment: {recorded_path}"
                )
        if args.task == "rj45":
            final_nominal_evidence = verify_rj45_nominal_evidence(
                PROJECT_ROOT, selected_task, args.rj45_nominal_evidence
            )
            if final_nominal_evidence != nominal_evidence:
                raise RuntimeError("RJ45 nominal evidence changed during the experiment")
        _bootstrap_input_snapshot.assert_unchanged()
        input_snapshot.assert_unchanged()
        _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
        checkpoint_manifest = {
            "schema_version": 1,
            "training_completed": True,
            "training_provenance_sha256": hashlib.sha256(
                (log_dir / "training_provenance.json").read_bytes()
            ).hexdigest(),
            "checkpoints": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(log_dir.glob("model_*.pt"))
            },
        }
        if not checkpoint_manifest["checkpoints"]:
            raise RuntimeError("training finished without a model checkpoint")
        checkpoint_manifest_path = log_dir / "checkpoint_manifest.json"
        atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)
        try:
            _bootstrap_input_snapshot.assert_unchanged()
            input_snapshot.assert_unchanged()
        except InputMutationError as error:
            atomic_write_json(
                checkpoint_manifest_path,
                {
                    "schema_version": 1,
                    "training_completed": False,
                    "input_immutability": error.report(),
                },
            )
            raise
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
