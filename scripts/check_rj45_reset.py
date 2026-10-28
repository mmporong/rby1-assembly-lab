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
from typing import Any, Mapping, cast


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


def _required_preimport_path(name: str, *, project_path: bool) -> Path:
    value = _preimport_option(name, required=True)
    if value is None:
        raise AssertionError(f"required option unexpectedly missing: {name}")
    path = Path(value)
    return _project_path(path) if project_path else _resolved_path(path)


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


def _bank_cycle_indices(num_envs: int, cycle: int) -> tuple[int, ...]:
    """Partition N64/N256 environments across four partial-reset cycles."""

    if type(num_envs) is not int or num_envs not in (64, 256):
        raise ValueError("held-bank reset checking requires 64 or 256 environments")
    if type(cycle) is not int or cycle not in range(4):
        raise ValueError("held-bank reset checking requires cycle 0, 1, 2, or 3")
    return tuple(range(cycle, num_envs, 4))


def _validate_bank_cli_contract(
    held_value: str | None,
    bank_value: str | None,
    *,
    num_envs: int | None = None,
    reset_cycles: int | None = None,
    mode: str | None = None,
) -> bool:
    """Validate mutually exclusive single-artifact and train-bank modes."""

    if (held_value is None) == (bank_value is None):
        raise ValueError("exactly one of --held or --bank is required")
    bank_mode = bank_value is not None
    if bank_mode and num_envs is not None and num_envs not in (64, 256):
        raise ValueError("--bank requires --num-envs 64 or 256")
    if bank_mode and reset_cycles is not None and reset_cycles != 4:
        raise ValueError("--bank requires --reset-cycles 4")
    if bank_mode and mode is not None and mode != "physical-warmup-cycles":
        raise ValueError("--bank requires --mode physical-warmup-cycles")
    return bank_mode


def _resolve_bank_grasp_ids(
    selected_indices: list[int],
    grasp_ids: list[str],
    observed_ids: list[str] | None = None,
) -> list[str]:
    """Resolve ordered IDs and reject stale per-environment identity records."""

    resolved = []
    for index in selected_indices:
        if type(index) is not int or index < 0 or index >= len(grasp_ids):
            raise ValueError("held-bank selection index is out of range")
        resolved.append(grasp_ids[index])
    if observed_ids is not None and observed_ids != resolved:
        raise ValueError("held-bank grasp identity is stale")
    return resolved


def _assert_bank_identity_unchanged(start: dict, end: dict, start_count: int, end_count: int) -> None:
    """Reject manifest, source-contract, artifact, or row-count drift."""

    if start_count != end_count or start != end:
        raise ValueError("held-bank source or manifest identity changed during physics")


def _assert_bank_check_coverage(checks: list[dict], num_envs: int) -> None:
    expected_groups = [list(range(num_envs))] + [
        list(_bank_cycle_indices(num_envs, cycle)) for cycle in range(4)
    ]
    if len(checks) != len(expected_groups):
        raise RuntimeError("held-bank initial state was not checked before every warmup")
    previous_step = -1
    for index, (check, expected_ids) in enumerate(zip(checks, expected_groups, strict=True)):
        if check["environment_ids"] != expected_ids or check["reset_cycle"] != index - 1:
            raise RuntimeError("held-bank restoration check has wrong environment coverage")
        step = check["control_step"]
        if type(step) is not int or step <= previous_step:
            raise RuntimeError("held-bank restoration checks lack increasing control steps")
        previous_step = step
        before, after = check["staging_before_restore"], check["staging_after_restore"]
        if (
            before["queued"] != [True] * len(expected_ids)
            or before["restored"] != [False] * len(expected_ids)
            or after["queued"] != [False] * len(expected_ids)
            or after["restored"] != [True] * len(expected_ids)
            or len(before["waited_control_steps"]) != len(expected_ids)
            or any(value < check["required_staging_control_steps"] for value in before["waited_control_steps"])
            or check["physics_step_before_restore"] != check["physics_step_after_check"]
        ):
            raise RuntimeError("held-bank restoration check lacks its pre-physics staging boundary")


def _assert_origin_applied_once(actual, local, origins, *, atol: float = 1.0e-9) -> None:
    """Reject omitted or duplicated environment origins in object positions."""

    if len(actual) != len(local) or len(actual) != len(origins):
        raise ValueError("held-bank origin check row counts differ")
    for row, (actual_objects, local_objects, origin) in enumerate(
        zip(actual, local, origins, strict=True)
    ):
        if len(actual_objects) != len(local_objects) or len(origin) != 3:
            raise ValueError("held-bank origin check shapes differ")
        for obj, (actual_position, local_position) in enumerate(
            zip(actual_objects, local_objects, strict=True)
        ):
            if len(actual_position) != 3 or len(local_position) != 3:
                raise ValueError("held-bank object positions must be xyz rows")
            for axis in range(3):
                expected = float(local_position[axis]) + float(origin[axis])
                if abs(float(actual_position[axis]) - expected) > atol:
                    raise ValueError(
                        "held-bank object origin was not applied exactly once: "
                        f"row={row}, object={obj}, axis={axis}"
                    )


task_path = PROJECT_ROOT / "configs/tasks/rj45-v1.json"
runtime_path = PROJECT_ROOT / "configs/runtime.json"
training_config_path = PROJECT_ROOT / "configs/rj45-training-v1.json"
nominal_path = _required_preimport_path("--nominal", project_path=True)
_held_option = _preimport_option("--held", required=False)
_bank_option = _preimport_option("--bank", required=False)
bank_mode = _validate_bank_cli_contract(_held_option, _bank_option)
held_path = None if _held_option is None else _project_path(Path(_held_option))
bank_path = None if _bank_option is None else _project_path(Path(_bank_option))
report_path = _required_preimport_path("--report", project_path=False)
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
        *((held_path,) if held_path is not None else ()),
        *((bank_path,) if bank_path is not None else ()),
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
held_document = None if held_path is None else _load_json_object(held_path, "held artifact")
held_provenance = None if held_document is None else held_document.get("provenance")
if held_document is not None and not isinstance(held_provenance, dict):
    raise ValueError("held artifact provenance is missing")
parallel_spec = (
    None
    if parallel_spec_path is None
    else _load_json_object(parallel_spec_path, "parallel proof specification")
)
extra_paths = [
    nominal_path,
    _referenced_path(nominal_document, "trace_path", "nominal report"),
]
if held_path is not None:
    if not isinstance(held_provenance, dict):
        raise AssertionError("validated held provenance unexpectedly missing")
    extra_paths.extend(
        [
            held_path,
            _referenced_path(held_provenance, "nominal_report_path", "held provenance"),
            _referenced_path(held_provenance, "nominal_trace_path", "held provenance"),
        ]
    )
else:
    from rby1_assembly.held_grasp import held_grasp_bank_dependency_paths

    if bank_path is None:
        raise AssertionError("validated bank path unexpectedly missing")
    extra_paths.extend(
        held_grasp_bank_dependency_paths(
            PROJECT_ROOT, bank_path, "train", allow_final=False
        )
    )
if parallel_spec_path is not None:
    if not isinstance(parallel_spec, dict):
        raise AssertionError("parallel proof specification unexpectedly missing")
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
from rby1_assembly.held_grasp import (  # noqa: E402
    bank_source_contract_sha256,
    load_verified_held_grasp_artifact,
    load_verified_held_grasp_bank,
)
from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence  # noqa: E402
from rby1_assembly.rj45_task import load_rj45_task, verify_runtime_versions  # noqa: E402
from rby1_assembly.rj45_training import load_training_config  # noqa: E402
from isaaclab.app import AppLauncher  # noqa: E402  # pyright: ignore[reportMissingImports]


parser = argparse.ArgumentParser(description="Check RJ45 held-reset runtime isolation")
parser.add_argument("--num-envs", type=int, choices=(1, 4, 64, 128, 256), required=True)
parser.add_argument("--reset-cycles", type=int)
parser.add_argument(
    "--mode",
    choices=("physical-warmup-cycles", "reset-api-stress"),
    default="physical-warmup-cycles",
)
parser.add_argument("--nominal", type=Path, required=True)
held_group = parser.add_mutually_exclusive_group(required=True)
held_group.add_argument("--held", type=Path)
held_group.add_argument("--bank", type=Path)
parser.add_argument("--parallel-contact-evidence", type=Path)
parser.add_argument("--report", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.reset_cycles is None:
    args.reset_cycles = 4 if bank_mode else 1000
if args.reset_cycles <= 0:
    parser.error("--reset-cycles must be positive")
try:
    _validate_bank_cli_contract(
        None if args.held is None else str(args.held),
        None if args.bank is None else str(args.bank),
        num_envs=args.num_envs,
        reset_cycles=args.reset_cycles,
        mode=args.mode,
    )
except ValueError as error:
    parser.error(str(error))
if (
    _requires_parallel_contact_evidence(args.num_envs)
    and args.parallel_contact_evidence is None
):
    parser.error("parallel checking requires --parallel-contact-evidence")
if _project_path(args.nominal) != nominal_path:
    parser.error("input paths changed between bootstrap and parsed arguments")
if args.held is not None and _project_path(args.held) != held_path:
    parser.error("held input path changed between bootstrap and parsed arguments")
if args.bank is not None and _project_path(args.bank) != bank_path:
    parser.error("bank input path changed between bootstrap and parsed arguments")
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
    captured_held = None if held_path is None else input_snapshot.json_for(held_path)
    if captured_nominal != nominal_document or captured_held != held_document:
        raise ValueError("captured input JSON changed during startup")
    if parallel_spec_path is not None:
        captured_parallel = input_snapshot.json_for(parallel_spec_path)
        if captured_parallel != parallel_spec:
            raise ValueError("captured parallel proof specification changed during startup")
    nominal_document = load_finite_json(nominal_path)
    held_document = None if held_path is None else load_finite_json(held_path)
    held_provenance = None if held_document is None else held_document.get("provenance")
    if held_document is not None and not isinstance(held_provenance, dict):
        raise ValueError("held artifact provenance is missing")
    parallel_spec = (
        None if parallel_spec_path is None else load_finite_json(parallel_spec_path)
    )
    task = load_rj45_task(task_path)
    runtime = input_snapshot.json_for(runtime_path)
    training_config = load_training_config(training_config_path)
    nominal_proof = verify_rj45_nominal_evidence(PROJECT_ROOT, task, nominal_path)
    if held_path is not None:
        held_state, held_proof = load_verified_held_grasp_artifact(
            PROJECT_ROOT, task, held_path
        )
        held_bank = None
        bank_proof = None
        bank_identity_at_start = None
    else:
        if bank_path is None:
            raise AssertionError("validated bank path unexpectedly missing")
        held_bank, bank_proof = load_verified_held_grasp_bank(
            PROJECT_ROOT, task, bank_path, "train", allow_final=False
        )
        # Keep a detached reference, never the environment's mutable stacked cache.
        verified_bank_reference = json.loads(json.dumps(held_bank, allow_nan=False))
        if bank_proof["selected_split"] != "train" or not bank_proof["no_final_opened"]:
            raise RuntimeError("held bank loader did not preserve the train-only contract")
        held_state = None
        held_proof = None
        bank_identity_at_start = {
            "manifest_sha256": bank_proof["manifest_sha256"],
            "source_contract_sha256": bank_source_contract_sha256(bank_proof),
            "artifact_sha256": list(bank_proof["selected_artifact_sha256"]),
        }
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

import omni.kit.app  # noqa: E402  # pyright: ignore[reportMissingImports]
import torch  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg  # noqa: E402
from rby1_assembly.rj45_env import configure_rj45_task  # noqa: E402
from rby1_assembly.rj45_training_env import make_rj45_training_env_type  # noqa: E402

bank_initial_state_checks = []


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


def _finish_warmup(env, actions, *, reset_cycle: int = -1) -> int:
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
        if bank_mode:
            control_steps = int(
                training_config["held_reset_staging"]["control_steps"]
            )
            ready = (
                env._training_held_reset_stage_queued
                & (
                    env._training_held_reset_stage_waited_control_steps
                    >= control_steps
                )
            )
            if torch.any(ready):
                ready_ids = torch.nonzero(ready, as_tuple=False).squeeze(1)
                staging_before = _bank_staging_snapshot(env, ready_ids)
                physics_step_before = int(env._sim_step_counter)
                env._advance_held_reset_staging()
                check = _assert_selected_bank_initial_state(env, ready_ids)
                check.update({
                    "reset_cycle": reset_cycle,
                    "control_step": int(env.common_step_counter),
                    "physics_step_before_restore": physics_step_before,
                    "physics_step_after_check": int(env._sim_step_counter),
                    "required_staging_control_steps": control_steps,
                    "staging_before_restore": staging_before,
                    "staging_after_restore": _bank_staging_snapshot(env, ready_ids),
                })
                bank_initial_state_checks.append(check)
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


def _bank_staging_snapshot(env, env_ids) -> dict:
    return {
        "queued": env._training_held_reset_stage_queued[env_ids].detach().cpu().tolist(),
        "restored": env._training_held_reset_stage_restored[env_ids].detach().cpu().tolist(),
        "waited_control_steps": env._training_held_reset_stage_waited_control_steps[env_ids].detach().cpu().tolist(),
    }


def _assert_selected_bank_initial_state(env, env_ids) -> dict:
    """Check selected bank state after restoration and before its first physics step."""

    if not bank_mode or bank_proof is None:
        raise RuntimeError("selected bank state checking requires verified bank mode")
    selected = env._training_grasp_index[env_ids]
    selected_indices = selected.detach().cpu().tolist()
    grasp_ids = bank_proof["selected_grasp_ids"]
    expected_indices = [
        (env_id + episode + int(task["data"]["evaluation_seed"])) % len(verified_bank_reference)
        for env_id, episode in zip(
            env_ids.detach().cpu().tolist(),
            env._training_episode_index[env_ids].detach().cpu().tolist(),
            strict=True,
        )
    ]
    if selected_indices != expected_indices:
        raise RuntimeError("held-bank selected indices differ from the fixed reset schedule")
    observed_ids = [env._training_held_bank_proof["selected_grasp_ids"][index] for index in selected_indices]
    selected_grasp_ids = _resolve_bank_grasp_ids(selected_indices, grasp_ids, observed_ids)
    rows = [verified_bank_reference[index] for index in selected_indices]
    joint_names = list(env.robot.joint_names)
    for artifact in rows:
        if len(artifact["joint_names"]) != len(joint_names) or set(artifact["joint_names"]) != set(joint_names):
            raise RuntimeError("verified held-bank joint names differ from the robot")

    def source_joints(key):
        return torch.tensor([
            [artifact["state"][key][artifact["joint_names"].index(name)] for name in joint_names]
            for artifact in rows
        ], device=env.device, dtype=torch.float32)

    expected_joint = source_joints("joint_positions_si")
    expected_velocity = source_joints("joint_velocities_si")
    expected_reference = source_joints("joint_reference_si")
    local_pose = torch.tensor([artifact["state"]["object_poses_world_wxyz"] for artifact in rows], device=env.device, dtype=torch.float32)
    expected_object_velocity = torch.tensor([artifact["state"]["object_com_velocities_world"] for artifact in rows], device=env.device, dtype=torch.float32)
    expected_object_pose = local_pose.clone()
    expected_object_pose[:, :, :3] += env.scene.env_origins[env_ids, None, :]
    cached_joint, cached_velocity, cached_reference = env._held_artifact_joint_state(env_ids)
    cached_object_pose, cached_object_velocity = env._held_artifact_object_state(env_ids)
    for name, actual, expected in (
        ("joint_position", cached_joint, expected_joint),
        ("joint_velocity", cached_velocity, expected_velocity),
        ("joint_reference", cached_reference, expected_reference),
        ("object_local_pose", env._training_held_bank_object_pose[selected], local_pose),
        ("object_world_pose", cached_object_pose, expected_object_pose),
        ("object_velocity", cached_object_velocity, expected_object_velocity),
    ):
        if not torch.equal(actual, expected):
            raise RuntimeError(f"held-bank cached state differs from verified source artifact: {name}")

    native = _native_physx_state(env, env_ids)
    if not torch.allclose(
        native["robot_joint_position"], expected_joint, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError("held-bank joint positions differ before physics")
    if not torch.allclose(
        native["robot_joint_velocity"], expected_velocity, rtol=0.0, atol=1.0e-7
    ):
        raise RuntimeError("held-bank joint velocities differ before physics")

    expected_transform = expected_object_pose.clone()
    expected_transform[:, :, 3:] = expected_object_pose[:, :, (4, 5, 6, 3)]
    for name, index in (("socket", 0), ("plug", 1)):
        expected_root = torch.cat(
            (expected_transform[:, index], expected_object_velocity[:, index]), dim=1
        )
        if not torch.allclose(
            native[f"{name}_root_state"], expected_root, rtol=0.0, atol=1.0e-7
        ):
            raise RuntimeError(
                f"held-bank {name} pose or velocity differs before physics"
            )

    target_checks = {
        "joint_targets": (env._joint_targets[env_ids], expected_reference),
        "insert_joint_targets": (
            env._insert_joint_targets[env_ids],
            expected_reference[:, env._arm_joint_ids],
        ),
        "torso_hold_targets": (
            env._torso_hold_targets[env_ids],
            expected_reference[:, env._torso_joint_ids],
        ),
    }
    if hasattr(env, "_gripper_command"):
        target_checks["gripper_command"] = (
            env._gripper_command[env_ids],
            expected_reference[:, env._gripper_joint_ids],
        )
    for name, (actual, expected) in target_checks.items():
        if not torch.equal(actual, expected):
            raise RuntimeError(f"held-bank actuator target mismatch: {name}")
    return {
        "environment_ids": env_ids.detach().cpu().tolist(),
        "selected_indices": selected_indices,
        "selected_grasp_ids": selected_grasp_ids,
        "selected_artifact_sha256": [bank_proof["selected_artifact_sha256"][index] for index in selected_indices],
        "expected_state_source": "detached_verified_bank_artifacts_in_robot_joint_order",
        "artifact_state_checked_before_physics": True,
        "environment_origin_application_count": 1,
    }


def _exercise_resets(env, actions) -> dict:
    all_ids = torch.arange(args.num_envs, device=env.device, dtype=torch.long)
    episode_start = env._training_episode_index.clone()
    reset_counts = torch.zeros(args.num_envs, dtype=torch.long, device=env.device)
    warmup_steps_per_cycle = []
    maximum_grip_force_n = 0.0
    maximum_insertion_load_n = 0.0
    maximum_forbidden_load_n = 0.0
    for cycle in range(args.reset_cycles):
        if bank_mode:
            target_ids = torch.tensor(
                _bank_cycle_indices(args.num_envs, cycle),
                dtype=torch.long,
                device=env.device,
            )
        else:
            target = cycle % args.num_envs
            target_ids = all_ids[target : target + 1]
        target_mask = torch.zeros(args.num_envs, dtype=torch.bool, device=env.device)
        target_mask[target_ids] = True
        untouched = all_ids[~target_mask]
        if len(target_ids) == 0 or len(untouched) == 0:
            raise RuntimeError("reset cycle must preserve a non-empty untouched subset")
        if not bank_mode and (
            len(target_ids) != 1 or len(untouched) != args.num_envs - 1
        ):
            raise RuntimeError("single held reset cycle must target one environment")
        before_target_episode = env._training_episode_index[target_ids].clone()
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
        reset_counts[target_ids] += 1
        if not torch.equal(
            env._training_episode_index[target_ids], before_target_episode + 1
        ):
            raise RuntimeError("reset environment episode index did not advance exactly once")
        if bool(env._training_warmup_complete[target_ids].any().item()):
            raise RuntimeError("reset environment bypassed held-state warmup")
        if len(untouched):
            _assert_untouched_equal(before, env, untouched)
            _assert_native_physx_untouched_equal(native_before, env, untouched)
        _require_finite_policy_observation(env)
        if args.mode == "physical-warmup-cycles":
            common_step_before = int(env.common_step_counter)
            try:
                warmup_steps = _finish_warmup(env, actions, reset_cycle=cycle)
            except RuntimeError as error:
                raise RuntimeError(f"physical reset cycle {cycle} failed: {error}") from error
            warmup_steps_per_cycle.append(warmup_steps)
            if int(env.common_step_counter) != common_step_before + warmup_steps:
                raise RuntimeError(f"physical reset cycle {cycle} did not advance simulation time")
            if not bool(env._grip_proven[target_ids].all().item()):
                raise RuntimeError(f"physical reset cycle {cycle} lacks grip proof")
            if not bool(env._evaluation_grip_proven[target_ids].all().item()):
                raise RuntimeError(f"physical reset cycle {cycle} lacks evaluation grip proof")
            if not bool(env._training_held_initialization_passed[target_ids].all().item()):
                raise RuntimeError(
                    f"physical reset cycle {cycle} lacks settled held initialization"
                )
            maximum_grip_force_n = max(
                maximum_grip_force_n,
                float(env._training_warmup_grip_force_peak_n[target_ids].amax().item()),
            )
            maximum_insertion_load_n = max(
                maximum_insertion_load_n,
                float(env._training_warmup_insertion_load_peak_n[target_ids].amax().item()),
            )
            maximum_forbidden_load_n = max(
                maximum_forbidden_load_n,
                float(env._training_warmup_forbidden_load_peak_n[target_ids].amax().item()),
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
        "reset_environments_per_cycle": (
            args.num_envs // 4 if bank_mode else 1
        ),
        "untouched_environments_per_cycle": (
            args.num_envs - args.num_envs // 4
            if bank_mode
            else args.num_envs - 1
        ),
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
    if bank_mode:
        if held_bank is None or bank_proof is None:
            raise AssertionError("verified held bank unexpectedly missing")
        env_type = make_rj45_training_env_type(
            env_type,
            training_config,
            held_bank=cast(list[Mapping[str, Any]], held_bank),
            held_bank_proof=bank_proof,
            calibration_bias_enabled=True,
            bias_level=training_config["calibration_bias"]["initial_level"],
        )
    else:
        if held_state is None:
            raise AssertionError("verified held artifact unexpectedly missing")
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
        if bank_mode:
            if bank_path is None or held_bank is None or bank_identity_at_start is None:
                raise AssertionError("verified bank identity unexpectedly missing")
            end_bank, end_bank_proof = load_verified_held_grasp_bank(
                PROJECT_ROOT, task, bank_path, "train", allow_final=False
            )
            bank_identity_at_end = {
                "manifest_sha256": end_bank_proof["manifest_sha256"],
                "source_contract_sha256": bank_source_contract_sha256(
                    end_bank_proof
                ),
                "artifact_sha256": list(
                    end_bank_proof["selected_artifact_sha256"]
                ),
            }
            _assert_bank_identity_unchanged(
                bank_identity_at_start,
                bank_identity_at_end,
                len(held_bank),
                len(end_bank),
            )
            _assert_bank_check_coverage(bank_initial_state_checks, args.num_envs)
        else:
            bank_identity_at_end = None
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
                "held_reset_source": (
                    "verified_train_held_grasp_bank"
                    if bank_mode
                    else "verified_nominal_trace_artifact"
                ),
                "checker_runtime_pose_or_force_override": False,
                "state_write_scope": "environment_held_reset_path_only",
                "ground_truth_usage": runtime_report["ground_truth_usage"],
                "success_and_safety_gates_relaxed": False,
                "terminal_seen_during_warmup": False,
                "ppo_loaded": False,
                "parallel_contact_evidence_role": (
                    "N4 contact-routing prerequisite; not reset-isolation measurement"
                    if args.num_envs > 1
                    else None
                ),
                "reset_isolation_measurement": (
                    f"N{args.num_envs} "
                    f"{'quarter-bank' if bank_mode else 'one-environment'} reset with "
                    f"{reset_result['untouched_environments_per_cycle']} "
                    "untouched environments checked per cycle"
                ),
            },
            "proof": {
                "nominal": nominal_proof,
                "held": held_proof,
                "held_bank": bank_proof,
                "held_bank_identity_at_start": bank_identity_at_start,
                "held_bank_identity_at_end": bank_identity_at_end,
                "held_bank_initial_state_checks": bank_initial_state_checks,
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
                "held_artifact_sha256": (
                    None if held_path is None else _sha256(held_path)
                ),
                "held_bank_manifest_sha256": (
                    None if bank_proof is None else bank_proof["manifest_sha256"]
                ),
                "held_bank_source_contract_sha256": (
                    None
                    if bank_proof is None
                    else bank_source_contract_sha256(bank_proof)
                ),
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
