#!/usr/bin/env python3
"""Headless Isaac check for the verified RJ45 held-reset training path."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import traceback


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.run_provenance import capture_run_inputs  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_path(path: Path) -> Path:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    return resolved.resolve()


def _project_path(path: Path) -> Path:
    resolved = _resolved_path(path)
    if resolved != PROJECT_ROOT and not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"input must stay inside the project: {resolved}")
    return resolved


def _preimport_option(name: str, *, required: bool) -> str | None:
    prefix = name + "="
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value.startswith(prefix):
            option_value = value[len(prefix):]
            if option_value:
                return option_value
            raise ValueError(f"{name} requires a value")
        if value == name:
            if index + 1 >= len(values) or values[index + 1].startswith("--"):
                raise ValueError(f"{name} requires a value")
            return values[index + 1]
    if required:
        raise ValueError(f"{name} is required")
    return None


def _load_json_object(path: Path, label: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _validate_report_path(path: Path, protected_paths) -> None:
    artifacts_root = (PROJECT_ROOT / "artifacts").resolve()
    if path.suffix.lower() != ".json":
        raise ValueError("--report must be a JSON file")
    if path == artifacts_root or not path.is_relative_to(artifacts_root):
        raise ValueError("--report must stay below the project artifacts directory")
    protected = {Path(item).resolve() for item in protected_paths}
    if path in protected:
        raise ValueError(f"--report must not overwrite an input or source file: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"--report must identify a regular file: {path}")


def _requires_parallel_contact_evidence(num_envs: int) -> bool:
    if type(num_envs) is not int or num_envs not in (1, 4, 64, 128, 256):
        raise ValueError("num_envs must be one of 1, 4, 64, 128, or 256")
    return num_envs > 1


task_path = PROJECT_ROOT / "configs/tasks/rj45-v1.json"
runtime_path = PROJECT_ROOT / "configs/runtime.json"
training_config_path = PROJECT_ROOT / "configs/rj45-training-v1.json"
nominal_path = _project_path(Path(_preimport_option("--nominal", required=True)))
held_path = _project_path(Path(_preimport_option("--held", required=True)))
report_path = _resolved_path(Path(_preimport_option("--report", required=True)))
_parallel_option = _preimport_option("--parallel-contact-evidence", required=False)
parallel_spec_path = (
    None if _parallel_option is None else _project_path(Path(_parallel_option))
)
_validate_report_path(
    report_path,
    (
        task_path,
        runtime_path,
        training_config_path,
        nominal_path,
        held_path,
        *((parallel_spec_path,) if parallel_spec_path is not None else ()),
        Path(__file__).resolve(),
        *sorted((PROJECT_ROOT / "src/rby1_assembly").glob("*.py")),
    ),
)


def _referenced_path(document: dict, key: str, label: str) -> Path:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must identify {key}")
    return _project_path(Path(value))


nominal_document = _load_json_object(nominal_path, "nominal report")
held_document = _load_json_object(held_path, "held artifact")
held_provenance = held_document.get("provenance")
if not isinstance(held_provenance, dict):
    raise ValueError("held artifact provenance is missing")
parallel_spec = (
    None
    if parallel_spec_path is None
    else _load_json_object(parallel_spec_path, "parallel proof specification")
)
extra_paths = [
    nominal_path,
    _referenced_path(nominal_document, "trace_path", "nominal report"),
    held_path,
    _referenced_path(held_provenance, "nominal_report_path", "held provenance"),
    _referenced_path(held_provenance, "nominal_trace_path", "held provenance"),
]
if parallel_spec_path is not None:
    extra_paths.extend(
        [
            parallel_spec_path,
            _referenced_path(parallel_spec, "report_path", "parallel proof"),
            _referenced_path(parallel_spec, "trace_path", "parallel proof"),
            _referenced_path(parallel_spec, "sat_report_path", "parallel proof"),
        ]
    )
_validate_report_path(report_path, (*extra_paths, Path(__file__).resolve()))
_bootstrap_input_snapshot = capture_run_inputs(
    PROJECT_ROOT,
    task_path=task_path,
    build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
    config_paths=(
        runtime_path,
        training_config_path,
        PROJECT_ROOT / "configs/assets.json",
        PROJECT_ROOT / "configs/rj45-assets.json",
    ),
    source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
    source_files=(Path(__file__).resolve(),),
    extra_paths=tuple(extra_paths),
)
_bootstrap_input_snapshot.assert_unchanged()
_validate_report_path(
    report_path,
    tuple(
        Path(key) if Path(key).is_absolute() else PROJECT_ROOT / key
        for key in _bootstrap_input_snapshot.sha256
    ),
)

from rby1_assembly.experiment_contracts import (  # noqa: E402
    apply_runtime_timing,
    atomic_write_json,
    load_finite_json,
    verify_rj45_nominal_evidence,
)
from rby1_assembly.held_grasp import load_verified_held_grasp_artifact  # noqa: E402
from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence  # noqa: E402
from rby1_assembly.rj45_task import load_rj45_task, verify_runtime_versions  # noqa: E402
from rby1_assembly.rj45_training import load_training_config  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Check RJ45 held-reset runtime isolation")
parser.add_argument("--num-envs", type=int, choices=(1, 4, 64, 128, 256), required=True)
parser.add_argument("--reset-cycles", type=int, default=1000)
parser.add_argument(
    "--mode",
    choices=("physical-warmup-cycles", "reset-api-stress"),
    default="physical-warmup-cycles",
)
parser.add_argument("--nominal", type=Path, required=True)
parser.add_argument("--held", type=Path, required=True)
parser.add_argument("--parallel-contact-evidence", type=Path)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.reset_cycles <= 0:
    parser.error("--reset-cycles must be positive")
if (
    _requires_parallel_contact_evidence(args.num_envs)
    and args.parallel_contact_evidence is None
):
    parser.error("parallel checking requires --parallel-contact-evidence")
if _project_path(args.nominal) != nominal_path or _project_path(args.held) != held_path:
    parser.error("input paths changed between bootstrap and parsed arguments")
if _resolved_path(args.report) != report_path:
    parser.error("report path changed between bootstrap and parsed arguments")
parsed_parallel_path = (
    None
    if args.parallel_contact_evidence is None
    else _project_path(args.parallel_contact_evidence)
)
if parsed_parallel_path != parallel_spec_path:
    parser.error("parallel proof path changed between bootstrap and parsed arguments")

try:
    input_snapshot = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(
            runtime_path,
            training_config_path,
            PROJECT_ROOT / "configs/assets.json",
            PROJECT_ROOT / "configs/rj45-assets.json",
        ),
        source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
        source_files=(Path(__file__).resolve(),),
        extra_paths=tuple(extra_paths),
    )
    _bootstrap_input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    _validate_report_path(
        report_path,
        tuple(
            Path(key) if Path(key).is_absolute() else PROJECT_ROOT / key
            for key in input_snapshot.sha256
        ),
    )
    captured_nominal = input_snapshot.json_for(nominal_path)
    captured_held = input_snapshot.json_for(held_path)
    if captured_nominal != nominal_document or captured_held != held_document:
        raise ValueError("captured input JSON changed during startup")
    if parallel_spec_path is not None:
        captured_parallel = input_snapshot.json_for(parallel_spec_path)
        if captured_parallel != parallel_spec:
            raise ValueError("captured parallel proof specification changed during startup")
    nominal_document = load_finite_json(nominal_path)
    held_document = load_finite_json(held_path)
    held_provenance = held_document.get("provenance")
    if not isinstance(held_provenance, dict):
        raise ValueError("held artifact provenance is missing")
    parallel_spec = (
        None if parallel_spec_path is None else load_finite_json(parallel_spec_path)
    )
    task = load_rj45_task(task_path)
    runtime = input_snapshot.json_for(runtime_path)
    training_config = load_training_config(training_config_path)
    nominal_proof = verify_rj45_nominal_evidence(PROJECT_ROOT, task, nominal_path)
    held_state, held_proof = load_verified_held_grasp_artifact(
        PROJECT_ROOT, task, held_path
    )
    parallel_proof = (
        None
        if parallel_spec is None
        else load_parallel_contact_evidence(parallel_spec, project_root=PROJECT_ROOT)
    )
    if _requires_parallel_contact_evidence(args.num_envs) and parallel_proof is None:
        raise RuntimeError("parallel checking lacks verified N4 contact-routing evidence")
    physics_dt = float(runtime["physics_dt_s"])
    control_dt = float(runtime["control_dt_s"])
    if not math.isclose(physics_dt, 1.0 / 240.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"runtime physics rate is not 240 Hz: dt={physics_dt}")
    if not math.isclose(control_dt, 1.0 / 30.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError(f"runtime control rate is not 30 Hz: dt={control_dt}")
    input_snapshot.assert_unchanged()
except Exception as error:
    try:
        atomic_write_json(
            report_path,
            {
                "schema_version": 1,
                "passed": False,
                "physics_executed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            },
        )
    finally:
        raise

launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import omni.kit.app  # noqa: E402
import torch  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg  # noqa: E402
from rby1_assembly.rj45_env import configure_rj45_task  # noqa: E402
from rby1_assembly.rj45_training_env import make_rj45_training_env_type  # noqa: E402


def _require_finite_policy_observation(env) -> None:
    observation = env._get_observations()
    if not isinstance(observation, dict) or set(observation) != {"policy"}:
        raise RuntimeError("training observation must contain only the policy tensor")
    policy = observation["policy"]
    if tuple(policy.shape) != (args.num_envs, 79):
        raise RuntimeError(f"unexpected policy observation shape: {tuple(policy.shape)}")
    if not bool(torch.isfinite(policy).all().item()):
        raise RuntimeError("policy observation contains a non-finite value")


def _zero_actions(env):
    if training_config["action_mode"] != "right_only_6d":
        raise RuntimeError("this check expects the current right_only_6d contract")
    return torch.zeros((args.num_envs, 6), device=env.device)


def _finish_warmup(env, actions) -> int:
    maximum_steps = max(
        1,
        math.ceil((float(training_config["warmup_s"]) + 1.0) / env.step_dt),
    )
    steps = 0
    while not bool(env._training_warmup_complete.all().item()):
        if steps >= maximum_steps:
            raise RuntimeError("held-reset warmup did not complete within its strict limit")
        observations, rewards, terminated, truncated, _ = env.step(actions)
        if tuple(observations["policy"].shape) != (args.num_envs, 79):
            raise RuntimeError("env.step returned an observation with the wrong shape")
        tensors = (observations["policy"], rewards, terminated, truncated)
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise RuntimeError("warmup produced a non-finite tensor")
        if bool((terminated | truncated).any().item()):
            raise RuntimeError("an environment terminated during held-reset warmup")
        steps += 1
    if not bool(env._training_policy_activated.all().item()):
        raise RuntimeError("held-reset warmup completed without policy activation")
    _require_finite_policy_observation(env)
    return steps


def _assert_untouched_equal(before: dict[str, torch.Tensor], env, untouched) -> None:
    current = {
        "episode_index": env._training_episode_index[untouched],
        "episode_length": env.episode_length_buf[untouched],
        "phase_started": env._phase_started_s[untouched],
        "joint_targets": env._joint_targets[untouched],
        "command_depth": env._insertion_command_depth_m[untouched],
        "previous_action": env._training_previous_action[untouched],
    }
    for name, expected in before.items():
        if not torch.equal(current[name], expected):
            raise RuntimeError(f"partial reset changed untouched environment state: {name}")


def _native_physx_state(env, env_ids) -> dict[str, torch.Tensor]:
    """Read reset-sensitive state from PhysX views, bypassing asset data caches."""

    robot_view = env.robot.root_physx_view
    return {
        "robot_joint_position": robot_view.get_dof_positions()[env_ids].clone(),
        "robot_joint_velocity": robot_view.get_dof_velocities()[env_ids].clone(),
        "robot_root_state": torch.cat(
            (
                robot_view.get_root_transforms()[env_ids],
                robot_view.get_root_velocities()[env_ids],
            ),
            dim=-1,
        ).clone(),
        "socket_root_state": torch.cat(
            (
                env.socket.root_physx_view.get_transforms()[env_ids],
                env.socket.root_physx_view.get_velocities()[env_ids],
            ),
            dim=-1,
        ).clone(),
        "plug_root_state": torch.cat(
            (
                env.plug.root_physx_view.get_transforms()[env_ids],
                env.plug.root_physx_view.get_velocities()[env_ids],
            ),
            dim=-1,
        ).clone(),
    }


def _assert_native_physx_untouched_equal(
    before: dict[str, torch.Tensor], env, untouched
) -> None:
    current = _native_physx_state(env, untouched)
    for name, expected in before.items():
        if not torch.equal(current[name], expected):
            raise RuntimeError(
                f"partial reset changed untouched native PhysX state: {name}"
            )


def _exercise_resets(env, actions) -> dict:
    all_ids = torch.arange(args.num_envs, device=env.device, dtype=torch.long)
    episode_start = env._training_episode_index.clone()
    reset_counts = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    warmup_steps_per_cycle = []
    maximum_grip_force_n = 0.0
    maximum_insertion_load_n = 0.0
    maximum_forbidden_load_n = 0.0
    for cycle in range(args.reset_cycles):
        target = cycle % args.num_envs
        target_ids = all_ids[target : target + 1]
        untouched = all_ids[all_ids != target]
        if len(target_ids) != 1 or len(untouched) != args.num_envs - 1:
            raise RuntimeError("reset cycle must target one environment and preserve all others")
        before_target_episode = int(env._training_episode_index[target].item())
        before = {
            "episode_index": env._training_episode_index[untouched].clone(),
            "episode_length": env.episode_length_buf[untouched].clone(),
            "phase_started": env._phase_started_s[untouched].clone(),
            "joint_targets": env._joint_targets[untouched].clone(),
            "command_depth": env._insertion_command_depth_m[untouched].clone(),
            "previous_action": env._training_previous_action[untouched].clone(),
        }
        native_before = _native_physx_state(env, untouched) if len(untouched) else {}
        env._reset_idx(target_ids)
        reset_counts[target] += 1
        if int(env._training_episode_index[target].item()) != before_target_episode + 1:
            raise RuntimeError("reset environment episode index did not advance exactly once")
        if bool(env._training_warmup_complete[target].item()):
            raise RuntimeError("reset environment bypassed held-state warmup")
        if len(untouched):
            _assert_untouched_equal(before, env, untouched)
            _assert_native_physx_untouched_equal(native_before, env, untouched)
        _require_finite_policy_observation(env)
        if args.mode == "physical-warmup-cycles":
            common_step_before = int(env.common_step_counter)
            try:
                warmup_steps = _finish_warmup(env, actions)
            except RuntimeError as error:
                raise RuntimeError(f"physical reset cycle {cycle} failed: {error}") from error
            warmup_steps_per_cycle.append(warmup_steps)
            if int(env.common_step_counter) != common_step_before + warmup_steps:
                raise RuntimeError(f"physical reset cycle {cycle} did not advance simulation time")
            if not bool(env._grip_proven[target].all().item()):
                raise RuntimeError(f"physical reset cycle {cycle} lacks grip proof")
            if not bool(env._evaluation_grip_proven[target].all().item()):
                raise RuntimeError(f"physical reset cycle {cycle} lacks evaluation grip proof")
            maximum_grip_force_n = max(
                maximum_grip_force_n,
                float(env._training_warmup_grip_force_peak_n[target].item()),
            )
            maximum_insertion_load_n = max(
                maximum_insertion_load_n,
                float(env._training_warmup_insertion_load_peak_n[target].item()),
            )
            maximum_forbidden_load_n = max(
                maximum_forbidden_load_n,
                float(env._training_warmup_forbidden_load_peak_n[target].item()),
            )
            if len(untouched):
                if not torch.equal(env._training_episode_index[untouched], before["episode_index"]):
                    raise RuntimeError(
                        f"physical reset cycle {cycle} auto-reset an untouched environment"
                    )
                expected_length = before["episode_length"] + warmup_steps
                if not torch.equal(env.episode_length_buf[untouched], expected_length):
                    raise RuntimeError(
                        f"physical reset cycle {cycle} did not advance untouched episode counters"
                    )
                if not bool(torch.isfinite(env._joint_targets[untouched]).all().item()):
                    raise RuntimeError(
                        f"physical reset cycle {cycle} produced non-finite untouched commands"
                    )
        if (cycle + 1) % 50 == 0 or cycle + 1 == args.reset_cycles:
            print(json.dumps({
                "completed_reset_cycles": cycle + 1,
                "physical_warmup_cycles": len(warmup_steps_per_cycle),
                "warmup_control_steps": sum(warmup_steps_per_cycle),
                "maximum_grip_force_n": maximum_grip_force_n,
            }), flush=True)
    if not torch.equal(env._training_episode_index, episode_start + reset_counts):
        raise RuntimeError("episode indices disagree with the requested reset count")

    if args.mode == "reset-api-stress":
        final_warmup_steps = _finish_warmup(env, actions)
    else:
        final_warmup_steps = 0
    return {
        "mode": args.mode,
        "cycles": args.reset_cycles,
        "measurement_num_envs": args.num_envs,
        "reset_environments_per_cycle": 1,
        "untouched_environments_per_cycle": args.num_envs - 1,
        "per_environment_reset_count": reset_counts.detach().cpu().tolist(),
        "final_warmup_steps": final_warmup_steps,
        "physical_warmup_cycles_completed": len(warmup_steps_per_cycle),
        "warmup_control_steps_per_physical_cycle": warmup_steps_per_cycle,
        "warmup_control_steps_total": sum(warmup_steps_per_cycle),
        "warmup_control_steps_min": min(warmup_steps_per_cycle, default=None),
        "warmup_control_steps_max": max(warmup_steps_per_cycle, default=None),
        "untouched_state_preserved_at_every_reset": True,
        "selected_controller_state_fields_checked": [
            "episode_index",
            "episode_length",
            "phase_started",
            "joint_targets",
            "command_depth",
            "previous_action",
        ],
        "selected_controller_state_scope": (
            "compatibility result covers only selected_controller_state_fields_checked; "
            "it does not assert preservation of unlisted training telemetry"
        ),
        "unexamined_untouched_training_telemetry_groups": [
            "calibration_bias",
            "held_reset_and_warmup",
            "reward_components_and_action_saturation",
            "policy_observation_cache",
        ],
        "untouched_native_physx_state_preserved_at_every_partial_reset": (
            args.num_envs > 1
        ),
        "native_physx_state_fields": (
            [
                "robot_joint_position",
                "robot_joint_velocity",
                "robot_root_state",
                "socket_root_state",
                "plug_root_state",
            ]
            if args.num_envs > 1
            else []
        ),
        "untouched_episode_time_advanced_each_physical_cycle": (
            args.num_envs > 1 and args.mode == "physical-warmup-cycles"
        ),
        "no_untouched_automatic_reset_seen": (
            args.num_envs > 1 and args.mode == "physical-warmup-cycles"
        ),
        "maximum_warmup_loads_n": {
            "grip": maximum_grip_force_n,
            "insertion": maximum_insertion_load_n,
            "forbidden": maximum_forbidden_load_n,
        },
    }


def main() -> int:
    current_versions = {
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
    verify_runtime_versions(current_versions, runtime)
    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, runtime)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = int(task["data"]["evaluation_seed"])
    if parallel_spec is not None:
        cfg.rj45_parallel_contact_evidence = parallel_spec
    env_type = configure_rj45_task(cfg, task, PROJECT_ROOT)
    env_type = make_rj45_training_env_type(
        env_type,
        training_config,
        held_state=held_state,
        calibration_bias_enabled=True,
        bias_level=training_config["calibration_bias"]["initial_level"],
    )
    env = env_type(cfg)
    try:
        # The training wrapper initializes its held-state buffers after the base
        # DirectRLEnv constructor. Reset once here so the verified held state is
        # restored through the fully initialized training reset path.
        env.reset()
        _require_finite_policy_observation(env)
        environment_physics_dt = float(env.cfg.sim.dt)
        if not math.isclose(environment_physics_dt, 1.0 / 240.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(
                f"environment physics rate is not 240 Hz: {environment_physics_dt}"
            )
        if not math.isclose(env.step_dt, 1.0 / 30.0, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError(f"environment control rate is not 30 Hz: {env.step_dt}")
        actions = _zero_actions(env)
        initial_warmup_steps = _finish_warmup(env, actions)
        reset_result = _exercise_resets(env, actions)
        runtime_report = env.rj45_training_report()
        result = {
            "schema_version": 1,
            "passed": True,
            "physics_executed": True,
            "task_id": "rj45-v1",
            "num_envs": args.num_envs,
            "physics_hz": 1.0 / environment_physics_dt,
            "control_hz": 1.0 / env.step_dt,
            "policy_observation_dimension": 79,
            "policy_action_dimension": 6,
            "policy_actions": "zero",
            "explicit_post_construction_reset": True,
            "initial_warmup_steps": initial_warmup_steps,
            "reset_check": reset_result,
            "contracts": {
                "held_reset_source": "verified_nominal_trace_artifact",
                "checker_runtime_pose_or_force_override": False,
                "state_write_scope": "environment_held_reset_path_only",
                "ground_truth_usage": runtime_report["ground_truth_usage"],
                "success_and_safety_gates_relaxed": False,
                "parallel_contact_evidence_role": (
                    "N4 contact-routing prerequisite; not reset-isolation measurement"
                    if args.num_envs > 1
                    else None
                ),
                "reset_isolation_measurement": (
                    f"N{args.num_envs} one-environment reset with "
                    f"{args.num_envs - 1} untouched environments checked per cycle"
                ),
            },
            "proof": {
                "nominal": nominal_proof,
                "held": held_proof,
                "parallel_contact": parallel_proof,
                "parallel_contact_scope": (
                    "preexisting N4 contact-routing evidence"
                    if parallel_proof is not None
                    else None
                ),
            },
            "provenance": {
                "checker_sha256": _sha256(Path(__file__)),
                "task_config_sha256": task["_config_sha256"],
                "runtime_config_sha256": _sha256(runtime_path),
                "training_config_sha256": training_config["_config_sha256"],
                "nominal_report_sha256": _sha256(nominal_path),
                "held_artifact_sha256": _sha256(held_path),
                "parallel_spec_sha256": (
                    None if parallel_spec_path is None else _sha256(parallel_spec_path)
                ),
                "runtime_versions": current_versions,
            },
        }
        input_snapshot.assert_unchanged()
        atomic_write_json(report_path, result)
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as error:
        traceback.print_exc()
        try:
            atomic_write_json(
                report_path,
                {
                    "schema_version": 1,
                    "passed": False,
                    "physics_executed": True,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
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
