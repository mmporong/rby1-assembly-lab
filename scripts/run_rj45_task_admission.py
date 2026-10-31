#!/usr/bin/env python3
"""Run the bounded G1 RJ45 tuning and task-admission protocol."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import faulthandler
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_RUNTIME_COMMIT = "104721acf2222c1cfe8ec86cd030f5116a4fef93"
EXPECTED_BANK_SHA256 = "c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5"
EXPECTED_BANK_SOURCE_SHA256 = "38cf66d4dea11fe4253021fa57e3c0b5347974dda7490ddb9b7890e6819098d0"
EXPECTED_PLAYBACK_ACTIONS_FILE_SHA256 = "7dfc8db1558aea29295436f60e850da9b377d0b85c7b8e7fcdcecd9f54bcbc96"
EXPECTED_IDS = tuple(f"rj45-train-{index:02d}" for index in range(1, 9))
EXPECTED_FIXED_INDICES = {"rj45-train-02": 1, "rj45-train-05": 4, "rj45-train-03": 2}
MAX_DEVELOPMENT_EPISODES = 90
TASK_ADMISSION_ENVIRONMENT_SEED = 20261121
PHYSICS_INPUT_SNAPSHOT_SCHEMA = "rj45_task_admission_physics_input_snapshot_v1"
PHYSICS_INPUT_MANIFEST_RELATIVE = Path(
    "configs/rj45-task-admission-physics-inputs-104721a.json"
)
END_VALIDATOR_RELATIVE = "scripts/smoke_rj45_comparison_isaac.py"
EXPECTED_CURRENT_SOURCE_PATHS = {
    END_VALIDATOR_RELATIVE,
    "scripts/run_rj45_task_admission.py",
    "src/rby1_assembly/rj45_comparison_env.py",
    "src/rby1_assembly/rj45_task_admission_env.py",
    "src/rby1_assembly/rj45_comparison_providers.py",
    "src/rby1_assembly/rj45_task_admission.py",
}
SMOKE_PLAN = (
    ("tuning", "rj45-train-03", "tx-positive", "classical_force_search", 1, "A"),
    ("admission", "rj45-train-02", "tx-positive", "zero", 1, None),
    ("admission", "rj45-train-02", "tx-positive", "oracle_gt_reference", 1, None),
)


class AdmissionRunBlocked(RuntimeError):
    pass


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AdmissionRunBlocked(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish a fully flushed JSON file without an overwrite race."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
            _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _persist_episode_row(
    output_dir: Path,
    ordinal: int,
    plan_row: Sequence[Any],
    candidate_id: str | None,
    episode: Mapping[str, Any],
    run_identity: Mapping[str, str],
) -> Path:
    """Persist one complete runner row before any progress acknowledgement."""

    identity = {
        "stage": plan_row[0],
        "grasp_id": plan_row[1],
        "case_id": plan_row[2],
        "method_id": plan_row[3],
        "slot": plan_row[4],
        "candidate_id": candidate_id,
    }
    if ordinal < 1:
        raise AdmissionRunBlocked("episode ordinal must be positive")
    for key, expected in identity.items():
        if episode.get(key) != expected:
            raise AdmissionRunBlocked(
                f"episode row source identity mismatch for ordinal {ordinal}: {key}"
            )
    path = output_dir / "episode-rows" / f"{ordinal:03d}.json"
    _atomic_immutable_json(
        path,
        {
            "schema_name": "rj45-task-admission-episode-row-v1",
            "ordinal": ordinal,
            "source_identity": identity,
            "run_identity_sha256": run_identity["sha256"],
            "episode": episode,
        },
    )
    return path


def _write_run_identity(
    output_dir: Path, prepared: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, str]:
    path = output_dir / "run-identity.json"
    payload = {
        "schema_name": "rj45-task-admission-run-identity-v1",
        "environment_seed": TASK_ADMISSION_ENVIRONMENT_SEED,
        "execution_plan": [
            {
                "ordinal": ordinal,
                "stage": row[0],
                "grasp_id": row[1],
                "case_id": row[2],
                "method_id": row[3],
                "slot": row[4],
                "candidate_id": row[5],
            }
            for ordinal, row in enumerate(prepared["plan"], start=1)
        ],
        "provenance": _provenance(prepared, runtime),
        "physics_input_start": prepared.get("physics_input_snapshot"),
    }
    _atomic_immutable_json(path, payload)
    return {"path": str(path), "sha256": _sha256(path)}


def validate_bank_contract(proof: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, int]:
    if proof.get("manifest_sha256") != EXPECTED_BANK_SHA256:
        raise AdmissionRunBlocked("held-bank manifest SHA mismatch")
    ids = tuple(proof.get("selected_grasp_ids", ()))
    hashes = tuple(proof.get("selected_artifact_sha256", ()))
    if ids != EXPECTED_IDS or len(hashes) != 8:
        raise AdmissionRunBlocked("held-bank ordered train identity mismatch")
    resolved = {grasp_id: ids.index(grasp_id) for grasp_id in EXPECTED_FIXED_INDICES}
    if resolved != EXPECTED_FIXED_INDICES:
        raise AdmissionRunBlocked("held-bank ID-to-index contract mismatch")
    configured = {
        row["grasp_id"]: (int(row["verified_index"]), row["artifact_sha256"])
        for row in config["held_grasps"]
    }
    for grasp_id in ("rj45-train-02", "rj45-train-05"):
        index = resolved[grasp_id]
        if configured.get(grasp_id) != (index, hashes[index]):
            raise AdmissionRunBlocked(f"configured held identity mismatch: {grasp_id}")
    return resolved


def build_execution_plan(config: Mapping[str, Any], *, smoke_only: bool) -> tuple[tuple[Any, ...], ...]:
    if smoke_only:
        return SMOKE_PLAN
    cases = tuple(row["case_id"] for row in config["bias_cases"])
    tuning = tuple(
        ("tuning", "rj45-train-03", case, "classical_force_search", 1, candidate["id"])
        for candidate in config["classical_tuning"]["candidates"]
        for case in cases
    )
    methods = tuple(
        (method["id"], int(method["slots_per_case"])) for method in config["methods"]
    )
    admission = tuple(
        ("admission", grasp["grasp_id"], case, method, slot, None)
        for grasp in config["held_grasps"]
        for case in cases
        for method, slots in methods
        for slot in range(1, slots + 1)
    )
    plan = tuning + admission
    if len(tuning) != 18 or len(admission) != 72 or len(plan) != MAX_DEVELOPMENT_EPISODES:
        raise AdmissionRunBlocked("development plan must contain tuning18 then admission72")
    return plan


def _tensor_json(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {key: _tensor_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_tensor_json(item) for item in value]
    return value


def _canonical_json_bytes(payload: Mapping[str, Any], *, newline: bool = False) -> bytes:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return (encoded + ("\n" if newline else "")).encode()


def _canonical_json_sha256(payload: Mapping[str, Any], *, newline: bool = False) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload, newline=newline)).hexdigest()


def _one(value: Any, key: str, expected_type: type) -> Any:
    converted = _tensor_json(value)
    if not isinstance(converted, list) or len(converted) != 1:
        raise AdmissionRunBlocked(f"terminal {key} must have shape (1,)")
    item = converted[0]
    if expected_type is bool:
        if type(item) is not bool:
            raise AdmissionRunBlocked(f"terminal {key} must be boolean")
    elif type(item) not in (int, float) or isinstance(item, bool) or not math.isfinite(float(item)):
        raise AdmissionRunBlocked(f"terminal {key} must be finite numeric")
    return item


def _matrix(value: Any, key: str, width: int) -> list[list[float]]:
    converted = _tensor_json(value)
    if (
        not isinstance(converted, list)
        or len(converted) != 1
        or not isinstance(converted[0], list)
        or len(converted[0]) != width
    ):
        raise AdmissionRunBlocked(f"terminal {key} must have shape (1, {width})")
    if any(
        type(item) not in (int, float)
        or isinstance(item, bool)
        or not math.isfinite(float(item))
        for item in converted[0]
    ):
        raise AdmissionRunBlocked(f"terminal {key} must be finite numeric")
    return [[float(item) for item in converted[0]]]


def _validate_terminal_snapshot(value: Any) -> dict[str, Any]:
    terminal = _tensor_json(value)
    if not isinstance(terminal, dict):
        raise AdmissionRunBlocked("terminal_snapshot must be a mapping")
    terminal_mask = _one(terminal.get("terminal_mask"), "terminal_mask", bool)
    if terminal_mask is not True:
        raise AdmissionRunBlocked("terminal_snapshot does not describe the terminal env")
    required_bools = (
        "independent_eval_valid",
        "independent_eval_success",
        "hard_overload",
        "forbidden_contact",
    )
    required_scalars = (
        "grip_contact_force_peak_n",
        "insertion_contact_load_peak_n",
        "timestamp_s",
    )
    for key in required_bools:
        _one(terminal.get(key), key, bool)
    if terminal["independent_eval_valid"] != [True]:
        raise AdmissionRunBlocked("terminal independent evaluator must be valid")
    for key in required_scalars:
        if float(_one(terminal.get(key), key, float)) < 0.0:
            raise AdmissionRunBlocked(f"terminal {key} must be nonnegative")
    admission = terminal.get("task_admission")
    if not isinstance(admission, dict):
        raise AdmissionRunBlocked("terminal task_admission clone is required")
    if not isinstance(admission.get("applied_held_grasp_id"), str) or not admission[
        "applied_held_grasp_id"
    ]:
        raise AdmissionRunBlocked("terminal task_admission held grasp ID is invalid")
    held_index = admission.get("applied_held_grasp_index")
    if type(held_index) is not int or held_index < 0:
        raise AdmissionRunBlocked("terminal task_admission held grasp index is invalid")
    held_sha = admission.get("applied_held_artifact_sha256")
    if (
        not isinstance(held_sha, str)
        or len(held_sha) != 64
        or any(character not in "0123456789abcdef" for character in held_sha)
    ):
        raise AdmissionRunBlocked("terminal task_admission artifact SHA is invalid")
    for key in (
        "requested_translation_bias_entry_m",
        "requested_rotation_bias_entry_rad",
        "applied_translation_bias_entry_m",
        "applied_rotation_bias_entry_rad",
        "prefreeze_translation_socket_entry_m_xyz",
        "prefreeze_shortest_rotation_vector_socket_entry_rad_xyz",
        "insert_transition_translation_socket_entry_m_xyz",
        "insert_transition_shortest_rotation_vector_socket_entry_rad_xyz",
    ):
        _matrix(admission.get(key), f"task_admission.{key}", 3)
    for key in (
        "requested_action_6d",
        "bounded_action_6d",
        "actual_applied_action_6d",
        "actual_applied_tip_target_offset_entry_6d",
    ):
        _matrix(admission.get(key), f"task_admission.{key}", 6)
    for key in (
        "insert_transition_depth_m",
        "insert_transition_lateral_error_m",
        "insert_transition_orientation_error_rad",
    ):
        _one(admission.get(key), f"task_admission.{key}", float)
    for key in (
        "prefreeze_snapshot_valid",
        "insert_transition_snapshot_valid",
        "transition_contract_checked",
        "actual_applied_action_is_exact_zero",
    ):
        if _one(admission.get(key), f"task_admission.{key}", bool) is not True:
            raise AdmissionRunBlocked(f"terminal task_admission {key} must be true")
    if admission.get("evaluator_gt_available_to_fair_provider") is not False:
        raise AdmissionRunBlocked("fair provider evaluator-GT boundary was not preserved")
    if admission["actual_applied_action_6d"] != [[0.0] * 6] or admission[
        "actual_applied_tip_target_offset_entry_6d"
    ] != [[0.0] * 6]:
        raise AdmissionRunBlocked("first INSERT physical action telemetry is nonzero")
    terminal["task_admission"] = admission
    return terminal


def _classical_payload(candidate: Any) -> dict[str, Any]:
    return {
        "schema": "rj45-classical-force-search-v1",
        "candidate_id": candidate.candidate_id,
        "position_gain": float(candidate.position_gain),
        "rotation_gain": float(candidate.rotation_gain),
        "force_trigger_n": float(candidate.force_trigger_n),
        "search_step_m": float(candidate.search_step_m),
        "stall_ticks": 6,
        "stall_depth_delta_per_tick_m": 0.00002,
        "controlled_translation_axes": ["x", "z"],
        "controlled_rotation_axes": ["x", "z"],
        "lateral_force_direction_threshold_n": 0.25,
        "ticks_per_search_direction": 6,
        "search_direction_order": ["initial", "rotate_plus_90", "opposite", "rotate_minus_90"],
        "search_radius_max_m": 0.0006,
        "advance_recovery_ticks": 3,
        "zero_action_entry_force_norm_n": 20.0,
        "translation_bound_m": 0.0015,
        "hard_stop_evaluator_force_n": 25.0,
        "rotation_bound_rad": 0.026179938779914945,
        "final_component_clip": [-1.0, 1.0],
        "known_bias_access": False,
        "observation_dim": 79,
        "action_dim": 6,
        "entry_force_slice": [37, 40],
        "entry_force_scale_n": 8.0,
        "command_depth_index": 62,
        "command_depth_scale_m": 0.012,
        "advance_index": 63,
    }


def _attest_provider(provider: Any, method_id: str, prepared: Mapping[str, Any]) -> dict[str, Any]:
    source_sha = prepared["config"]["methods"][2]["source_trace_gzip_sha256"]
    if method_id == "zero":
        return {"applied_controller_sha256": None, "applied_artifact_sha256": None}
    if method_id == "classical_force_search":
        candidate = getattr(provider, "candidate", None)
        if candidate is None:
            raise AdmissionRunBlocked("classical provider candidate telemetry is unavailable")
        payload = _classical_payload(candidate)
        return {
            "applied_controller_sha256": _canonical_json_sha256(payload),
            "applied_artifact_sha256": None,
            "actual_classical_controller_payload": payload,
        }
    if method_id == "mean_constant":
        action = np.asarray(getattr(provider, "_action", None), dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise AdmissionRunBlocked("mean constant provider action telemetry is invalid")
        values = tuple(float(item) for item in action)
        payload = {
            "source_trace_gzip_sha256": source_sha,
            "method": "mean_constant",
            "normalized_right6": values,
        }
        return {
            "applied_controller_sha256": None,
            "applied_artifact_sha256": _canonical_json_sha256(payload),
            "actual_constant_action": list(values),
        }
    if method_id == "open_loop_playback":
        actual = tuple(tuple(float(item) for item in row) for row in provider.trace)
        expected = tuple(tuple(float(item) for item in row) for row in prepared["playback_actions"])
        if len(actual) != 364 or actual != expected:
            raise AdmissionRunBlocked("actual playback provider trace differs from fixed 364 rows")
        trace_payload = {
            "actions": [list(row) for row in actual],
            "last": list(actual[-1]),
            "mean": [math.fsum(row[index] for row in actual) / len(actual) for index in range(6)],
        }
        actual_trace_sha = _canonical_json_sha256(trace_payload, newline=True)
        if actual_trace_sha != prepared["playback_canonical_sha256"]:
            raise AdmissionRunBlocked("actual playback canonical trace SHA mismatch")
        artifact_payload = {
            "source_trace_gzip_sha256": source_sha,
            "method": "open_loop_playback",
            "selected_rows": len(actual),
            "tail": "hold_final_action",
        }
        return {
            "applied_controller_sha256": None,
            "applied_artifact_sha256": _canonical_json_sha256(artifact_payload),
            "actual_playback_trace_canonical_sha256": actual_trace_sha,
            "actual_playback_selected_rows": len(actual),
            "actual_playback_tail": "hold_final_action",
        }
    raise AdmissionRunBlocked(f"provider attestation unsupported for {method_id}")


def _reset_isaac_episode_lifecycle(
    *, simulation_context_cls=None, timeline=None, app=None, stage_utils=None
) -> None:
    """Leave the public Isaac timeline/stage ready for the next env constructor."""

    if simulation_context_cls is None:
        from isaaclab.sim import SimulationContext

        simulation_context_cls = SimulationContext
    if simulation_context_cls.instance() is not None:
        raise AdmissionRunBlocked("SimulationContext survived env.close")
    if timeline is None:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
    if app is None:
        import omni.kit.app

        app = omni.kit.app.get_app()
    if not timeline.is_stopped():
        timeline.stop()
        app.update()
    if timeline.is_playing() or not timeline.is_stopped():
        raise AdmissionRunBlocked("Isaac timeline did not reach stopped state")
    if stage_utils is None:
        import isaacsim.core.utils.stage as stage_utils

    stage_utils.create_new_stage()
    stage_utils.update_stage()
    if stage_utils.get_current_stage() is None:
        raise AdmissionRunBlocked("Isaac fresh stage creation failed")


def _runtime_versions() -> dict[str, str]:
    names = ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py")
    return {name: metadata.version(name) for name in names}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--physics-input-manifest",
        type=Path,
        default=PHYSICS_INPUT_MANIFEST_RELATIVE,
    )
    parser.add_argument("--verify-after-child-exit", action="store_true")
    parser.add_argument("--playback-actions", type=Path, default=Path("docs/evidence/rj45-frozen-ablation-20261031/source-actions.json"))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser


def _physics_input_manifest_path(value: Path, current_root: Path) -> Path:
    if value.is_absolute() or value != PHYSICS_INPUT_MANIFEST_RELATIVE:
        raise AdmissionRunBlocked(
            "physics input manifest must use the fixed project-relative path"
        )
    path = (current_root.resolve() / value).resolve()
    if not path.is_relative_to(current_root.resolve()):
        raise AdmissionRunBlocked("physics input manifest escapes current root")
    return path


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = args.runtime_root.resolve()
    current_root = args.current_root.resolve()
    if runtime_root == current_root:
        raise AdmissionRunBlocked("runtime-root and current-root must be separate")
    config_path = args.config if args.config.is_absolute() else current_root / args.config
    playback_path = args.playback_actions if args.playback_actions.is_absolute() else current_root / args.playback_actions
    config = _json(config_path)
    if config.get("schema_name") != "rj45_comparison_task_admission_v1":
        raise AdmissionRunBlocked("unexpected admission config schema")
    plan = build_execution_plan(config, smoke_only=args.smoke_only)
    playback = _json(playback_path)
    actions = playback.get("actions")
    if not isinstance(actions, list) or len(actions) != 364:
        raise AdmissionRunBlocked("playback action trace must contain 364 rows")
    if _sha256(playback_path) != EXPECTED_PLAYBACK_ACTIONS_FILE_SHA256:
        raise AdmissionRunBlocked("playback action artifact SHA mismatch")
    if playback.get("source_trace", {}).get("sha256") != config["methods"][2]["source_trace_gzip_sha256"]:
        raise AdmissionRunBlocked("playback source trace identity mismatch")
    canonical_payload = {
        "actions": [[float(value) for value in row] for row in actions],
        "last": [float(value) for value in actions[-1]],
        "mean": [
            math.fsum(float(row[index]) for row in actions) / len(actions)
            for index in range(6)
        ],
    }
    canonical_sha = _canonical_json_sha256(canonical_payload, newline=True)
    if playback.get("canonical_sha256") != canonical_sha:
        raise AdmissionRunBlocked("playback canonical action SHA mismatch")
    physics_manifest = _physics_input_manifest_path(
        args.physics_input_manifest, current_root
    )
    import importlib.util
    smoke_path = current_root / "scripts/smoke_rj45_comparison_isaac.py"
    spec = importlib.util.spec_from_file_location("rby1_admission_frozen_validator", smoke_path)
    if spec is None or spec.loader is None:
        raise AdmissionRunBlocked("cannot load frozen runtime validator")
    validator = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = validator
    spec.loader.exec_module(validator)
    manifest_path = current_root / "configs/rj45-oracle-runtime-104721a.json"
    runtime_snapshot = validator.validate_runtime_snapshot(runtime_root, manifest_path)
    physics_input_snapshot = validator.validate_physics_input_snapshot(
        runtime_root, current_root, physics_manifest
    )
    return {
        "runtime_root": runtime_root,
        "current_root": current_root,
        "config_path": config_path.resolve(),
        "config": config,
        "playback_path": playback_path.resolve(),
        "playback_actions": actions,
        "playback_canonical_sha256": canonical_sha,
        "plan": plan,
        "runtime_validator": validator,
        "runtime_manifest": manifest_path,
        "runtime_snapshot": runtime_snapshot,
        "physics_input_manifest": physics_manifest,
        "physics_input_snapshot": physics_input_snapshot,
    }


def _load_runtime(prepared: Mapping[str, Any]):
    runtime_root: Path = prepared["runtime_root"]
    current_root: Path = prepared["current_root"]
    sys.path.insert(0, str(runtime_root / "src"))
    from rby1_assembly.held_grasp import bank_source_contract_sha256, held_grasp_bank_dependency_paths, load_verified_held_grasp_bank
    from rby1_assembly.isaac_env import AssemblyPhase
    from rby1_assembly.rj45_task import load_rj45_task, verify_runtime_versions

    if int(AssemblyPhase.ALIGN) != 3 or int(AssemblyPhase.INSERT) != 4:
        raise AdmissionRunBlocked("frozen ALIGN/INSERT phase codes changed")

    task = load_rj45_task(runtime_root / "configs/tasks/rj45-v1.json")
    bank_path = runtime_root / "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json"
    bank, proof = load_verified_held_grasp_bank(runtime_root, task, bank_path, "train", allow_final=False)
    prepared["runtime_validator"].validate_held_proof_physics_binding(
        proof,
        prepared["physics_input_snapshot"],
        prepared["physics_input_manifest"],
    )
    if bank_source_contract_sha256(proof) != EXPECTED_BANK_SOURCE_SHA256:
        raise AdmissionRunBlocked("held-bank source contract mismatch")
    indices = validate_bank_contract(proof, prepared["config"])
    runtime_versions = _runtime_versions()
    verify_runtime_versions(runtime_versions, _json(runtime_root / "configs/runtime.json"))
    current_modules = {}
    import importlib.util
    for name in ("rj45_comparison_env", "rj45_task_admission_env", "rj45_comparison_providers", "rj45_task_admission"):
        path = current_root / f"src/rby1_assembly/{name}.py"
        spec = importlib.util.spec_from_file_location(f"rby1_current_{name}", path)
        if spec is None or spec.loader is None:
            raise AdmissionRunBlocked(f"cannot load current module: {name}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        current_modules[name] = module
    frozen_paths = held_grasp_bank_dependency_paths(runtime_root, bank_path, "train", allow_final=False)
    frozen_hashes = {path.resolve(): _sha256(path) for path in (bank_path, *frozen_paths)}
    return {"task": task, "bank": bank, "proof": proof, "indices": indices, "modules": current_modules, "frozen_hashes": frozen_hashes, "runtime_versions": runtime_versions}


def _provenance(prepared, runtime):
    current_root = prepared["current_root"]
    source_paths = {
        "runner": Path(__file__).resolve(),
        **{
            name: current_root / f"src/rby1_assembly/{name}.py"
            for name in (
                "rj45_comparison_env",
                "rj45_task_admission_env",
                "rj45_comparison_providers",
                "rj45_task_admission",
            )
        },
    }
    return {
        "runtime_snapshot": prepared["runtime_snapshot"],
        "runtime_versions": runtime["runtime_versions"],
        "environment_seed": TASK_ADMISSION_ENVIRONMENT_SEED,
        "config": {"path": str(prepared["config_path"]), "sha256": _sha256(prepared["config_path"])},
        "playback_actions": {"path": str(prepared["playback_path"]), "sha256": _sha256(prepared["playback_path"])},
        "held_bank_proof": runtime["proof"],
        "current_source_sha256": {name: _sha256(path) for name, path in source_paths.items()},
        "frozen_input_sha256": {str(path): digest for path, digest in runtime["frozen_hashes"].items()},
        "runtime_import_origins": _assert_runtime_import_origins(prepared["runtime_root"]),
        "physics_input_snapshot": prepared.get("physics_input_snapshot"),
    }


def _assert_runtime_import_origins(runtime_root: Path) -> dict[str, str]:
    runtime_src = (runtime_root / "src").resolve()
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if (name == "rby1_assembly" or name.startswith("rby1_assembly."))
        and getattr(module, "__file__", None)
    }
    escaped = {
        name: path
        for name, path in origins.items()
        if not Path(path).is_relative_to(runtime_src)
    }
    if escaped:
        raise AdmissionRunBlocked(f"frozen runtime import escaped: {escaped}")
    return origins


def _bias(config: Mapping[str, Any], case_id: str) -> Mapping[str, Any]:
    return next(row for row in config["bias_cases"] if row["case_id"] == case_id)


def _make_env(prepared, runtime, row):
    import torch
    task, bank, proof, indices, modules = (runtime[key] for key in ("task", "bank", "proof", "indices", "modules"))
    runtime_root = prepared["runtime_root"]
    config = prepared["config"]
    _, grasp_id, case_id, method_id, _, _ = row
    index = indices[grasp_id]
    bias = _bias(config, case_id)
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg
    from rby1_assembly.rj45_env import configure_rj45_task
    from rby1_assembly.rj45_training import load_training_config
    from rby1_assembly.rj45_training_env import make_rj45_training_env_type

    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, _json(runtime_root / "configs/runtime.json"))
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.seed = TASK_ADMISSION_ENVIRONMENT_SEED
    env_type = configure_rj45_task(cfg, task, runtime_root)
    training = load_training_config(runtime_root / "configs/rj45-training-target-offset-v1.json")
    env_type = make_rj45_training_env_type(
        env_type,
        training,
        held_state=bank[index],
        calibration_bias_enabled=True,
        bias_level=training["calibration_bias"]["initial_level"],
    )
    env_type = modules["rj45_comparison_env"].make_rj45_comparison_env_type(env_type)
    env_type = modules["rj45_task_admission_env"].make_rj45_task_admission_env_type(
        env_type,
        requested_translation_bias_entry_m=bias["translation_bias_entry_m"],
        requested_rotation_bias_entry_rad=bias["rotation_vector_bias_entry_rad"],
        held_grasp_id=grasp_id,
        held_grasp_index=index,
        held_artifact_sha256=proof["selected_artifact_sha256"][index],
    )
    if method_id == "oracle_gt_reference":
        from rby1_assembly.rj45_teacher_runtime import make_rj45_teacher_env_type
        env_type = make_rj45_teacher_env_type(env_type)
    return env_type(cfg), torch


def _episode(
    prepared,
    runtime,
    row,
    candidate_id,
    *, heartbeat=None,
):
    stage, grasp_id, case_id, method_id, slot, _ = row
    modules = runtime["modules"]
    provider = None
    state = None
    if method_id != "oracle_gt_reference":
        kwargs = {}
        method = next(item for item in prepared["config"]["methods"] if item["id"] == method_id)
        if method_id == "mean_constant":
            kwargs["constant_action"] = method["normalized_right6"]
        elif method_id == "open_loop_playback":
            kwargs["playback_trace"] = prepared["playback_actions"]
        elif method_id == "classical_force_search":
            kwargs["classical_candidate_id"] = candidate_id
        provider = modules["rj45_comparison_providers"].make_fair_provider(method_id, **kwargs)
        state = provider.reset()
        provider_attestation = _attest_provider(provider, method_id, prepared)
        if method_id == "classical_force_search":
            actual = provider_attestation["actual_classical_controller_payload"]
            expected = next(
                item
                for item in prepared["config"]["classical_tuning"]["candidates"]
                if item["id"] == candidate_id
            )
            for key in (
                "candidate_id",
                "position_gain",
                "rotation_gain",
                "force_trigger_n",
                "search_step_m",
            ):
                expected_value = expected["id"] if key == "candidate_id" else expected[key]
                if actual[key] != expected_value:
                    raise AdmissionRunBlocked(
                        f"actual classical provider differs from requested candidate {candidate_id}"
                    )
    else:
        provider_attestation = {
            "applied_controller_sha256": None,
            "applied_artifact_sha256": None,
        }
    env, torch = _make_env(prepared, runtime, row)
    observations = []
    evaluator_force_samples_n = []
    try:
        reset = env.reset()
        observation = reset[0] if isinstance(reset, tuple) else reset
        for step_index in range(1200):
            if heartbeat is not None and step_index % 150 == 0:
                heartbeat(step_index)
            policy = observation["policy"]
            valid = env.extras.get("policy_observation_valid")
            action = torch.zeros((1, 6), device=env.device)
            if method_id != "oracle_gt_reference" and valid is not None and bool(valid[0]):
                sample = policy[0].detach().cpu().numpy()
                produced, state = provider.act(sample, state)
                action[0] = torch.as_tensor(produced, device=env.device, dtype=action.dtype)
            step = env.step(action)
            observation, _, terminated, truncated, _ = step
            observations.append(tuple(float(value) for value in policy[0].detach().cpu().tolist()))
            current_grip = float(env._last_contact_proxy_n[0].max().item())
            current_insertion_load = float(
                env._last_insertion_contact_load_n[0].item()
            )
            evaluator_force_samples_n.append(max(current_grip, current_insertion_load))
            if bool((terminated | truncated)[0]):
                break
        else:
            raise AdmissionRunBlocked("episode exceeded the bounded 1200-step limit")
        terminal = _validate_terminal_snapshot(env.extras.get("terminal_snapshot"))
        admission_snapshot = terminal["task_admission"]
        success = bool(terminal["independent_eval_success"][0])
        forbidden = bool(terminal["forbidden_contact"][0])
        completion = float(terminal["timestamp_s"][0])
        return {
            "stage": stage,
            "grasp_id": grasp_id,
            "case_id": case_id,
            "method_id": method_id,
            "slot": slot,
            "candidate_id": candidate_id,
            "success": success,
            "hard_contact": bool(terminal["hard_overload"][0]),
            "forbidden_contact": forbidden,
            "evaluator_force_p95_n": float(
                np.percentile(
                    np.asarray(evaluator_force_samples_n, dtype=np.float64),
                    95,
                    method="linear",
                )
            ),
            "completion_time_s": completion,
            "provider_inputs": [] if method_id == "oracle_gt_reference" else ["observation79", "provider_owned_state"],
            **provider_attestation,
            "observation_samples": observations,
            "admission_snapshot": admission_snapshot,
            "terminal_complete": True,
            "execution_error": None,
        }
    finally:
        env.close()
        _reset_isaac_episode_lifecycle()
        torch.cuda.empty_cache()


def _group_equivalent(rows, group_keys, prefix, translation_limit=1.0e-5, rotation_limit=0.0008726646259971648):
    def quaternion(rotation_vector):
        angle = float(np.linalg.norm(rotation_vector))
        if angle <= 1.0e-15:
            return np.asarray((1.0, 0.0, 0.0, 0.0))
        return np.concatenate(
            ([math.cos(angle / 2.0)], rotation_vector / angle * math.sin(angle / 2.0))
        )

    grouped = {}
    for row in rows:
        snapshot = row.get("admission_snapshot", {})
        translation = snapshot.get(f"{prefix}_translation_socket_entry_m_xyz")
        rotation = snapshot.get(f"{prefix}_shortest_rotation_vector_socket_entry_rad_xyz")
        valid = snapshot.get(f"{prefix}_snapshot_valid", [False])
        if not valid or not valid[0] or not translation or not rotation:
            return False
        key = tuple(row[name] for name in group_keys)
        grouped.setdefault(key, []).append((np.asarray(translation[0]), np.asarray(rotation[0])))
    for values in grouped.values():
        reference_t, reference_r = values[0]
        for translation, rotation in values[1:]:
            if np.linalg.norm(translation - reference_t) > translation_limit:
                return False
            relative_dot = abs(float(np.dot(quaternion(rotation), quaternion(reference_r))))
            geodesic = 2.0 * math.acos(min(1.0, max(-1.0, relative_dot)))
            if geodesic > rotation_limit:
                return False
    return True


def _row_contract_complete(row: Mapping[str, Any]) -> bool:
    snapshot = row.get("admission_snapshot")
    return bool(
        row.get("execution_error") is None
        and row.get("terminal_complete") is True
        and isinstance(snapshot, Mapping)
        and snapshot.get("prefreeze_snapshot_valid") == [True]
        and snapshot.get("insert_transition_snapshot_valid") == [True]
        and snapshot.get("transition_contract_checked") == [True]
        and snapshot.get("actual_applied_action_is_exact_zero") == [True]
    )


def _validate_tuning_rows(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> None:
    if len(rows) != 18:
        raise AdmissionRunBlocked("tuning must complete exactly 18 requested episodes")
    candidates = {
        item["id"]: item for item in config["classical_tuning"]["candidates"]
    }
    for index, row in enumerate(rows):
        if not _row_contract_complete(row):
            raise AdmissionRunBlocked(
                f"tuning episode {index + 1} lacks complete terminal contract evidence"
            )
        candidate_id = row.get("candidate_id")
        candidate = candidates.get(candidate_id)
        if candidate is None:
            raise AdmissionRunBlocked(
                f"tuning episode {index + 1} has an unknown requested candidate"
            )
        expected_payload = _classical_payload(
            argparse.Namespace(
                candidate_id=candidate_id,
                position_gain=candidate["position_gain"],
                rotation_gain=candidate["rotation_gain"],
                force_trigger_n=candidate["force_trigger_n"],
                search_step_m=candidate["search_step_m"],
            )
        )
        expected_sha = _canonical_json_sha256(expected_payload)
        if (
            row.get("actual_classical_controller_payload") != expected_payload
            or row.get("applied_controller_sha256") != expected_sha
        ):
            raise AdmissionRunBlocked(
                f"tuning episode {index + 1} actual classical provider identity mismatch"
            )


def _validate_smoke_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != len(SMOKE_PLAN):
        raise AdmissionRunBlocked("smoke must complete all three fixed episodes")
    for index, row in enumerate(rows):
        if not _row_contract_complete(row):
            raise AdmissionRunBlocked(
                f"smoke episode {index + 1} lacks complete terminal contract evidence"
            )


def _evidence(
    rows: Sequence[Mapping[str, Any]],
    archive_valid: bool,
    config: Mapping[str, Any],
    *,
    baseline_artifacts_verified: bool,
):
    module = sys.modules["rby1_current_rj45_task_admission"]
    complete = all(_row_contract_complete(row) for row in rows)
    configured = {row["grasp_id"]: row for row in config["held_grasps"]}
    identities = all(
        row["admission_snapshot"].get("applied_held_grasp_id") == row["grasp_id"]
        and row["admission_snapshot"].get("applied_held_grasp_index") == configured[row["grasp_id"]]["verified_index"]
        and row["admission_snapshot"].get("applied_held_artifact_sha256") == configured[row["grasp_id"]]["artifact_sha256"]
        for row in rows if row.get("admission_snapshot")
    )
    bias_valid = all(
        max(
            abs(a - b)
            for key_a, key_b in (
                ("requested_translation_bias_entry_m", "applied_translation_bias_entry_m"),
                ("requested_rotation_bias_entry_rad", "applied_rotation_bias_entry_rad"),
            )
            for a, b in zip(row["admission_snapshot"][key_a][0], row["admission_snapshot"][key_b][0])
        ) <= 1.0e-9
        for row in rows if row.get("admission_snapshot")
    )
    first_zero = all(
        row["admission_snapshot"].get("transition_contract_checked", [False])[0]
        and row["admission_snapshot"].get("actual_applied_action_is_exact_zero", [False])[0]
        for row in rows if row.get("admission_snapshot")
    )
    return module.ContractEvidence(
        archive_hash_valid=archive_valid,
        held_identity_valid=complete and identities,
        requested_applied_bias_valid=complete and bias_valid,
        prefreeze_reset_equivalent=complete and _group_equivalent(rows, ("grasp_id",), "prefreeze"),
        insert_transition_equivalent=complete and _group_equivalent(rows, ("grasp_id", "case_id"), "insert_transition"),
        first_insert_physical_zero=complete and first_zero,
        telemetry_complete=complete,
        snapshots_complete=complete,
        finite_values=complete,
        baseline_artifacts_verified=baseline_artifacts_verified,
    )


def _failure_report(
    status: str,
    error: BaseException,
    progress_path: Path,
    run_identity: Mapping[str, str] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "error": {"type": type(error).__name__, "message": str(error)},
        "requested_process_exit_code": 2,
        "run_identity": run_identity,
        "external_supervisor_contract": {
            "validate_report_status_and_claims": True,
            "enforce_wall_clock_timeout": True,
            "heartbeat_path": str(progress_path),
            "native_hang_resolution_claimed": False,
            "verify_after_any_child_exit": True,
            "completion_requires_endmatch": True,
            "identity_missing_status": "unverifiable_start",
            "end_missing_scientific_completion": False,
        },
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
    }


def _validate_end_snapshot_before_validator(
    start: Mapping[str, Any], current_root: Path
) -> Path:
    """Authenticate the stored snapshot and validator bytes without importing it."""

    if start.get("schema_name") != PHYSICS_INPUT_SNAPSHOT_SCHEMA:
        raise AdmissionRunBlocked("physics input start snapshot schema mismatch")
    normalized = start.get("normalized_sha256")
    if not isinstance(normalized, dict) or set(normalized) != {"runtime", "current"}:
        raise AdmissionRunBlocked("physics input start snapshot map is invalid")
    if (
        _canonical_json_sha256(normalized)
        != start.get("normalized_sha256_canonical_sha256")
    ):
        raise AdmissionRunBlocked("physics input start snapshot canonical SHA mismatch")
    for namespace in ("runtime", "current"):
        mapping = normalized.get(namespace)
        if not isinstance(mapping, dict) or not mapping:
            raise AdmissionRunBlocked(
                f"physics input start snapshot {namespace} map is invalid"
            )
        for relative, digest in mapping.items():
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or relative != Path(relative).as_posix()
                or "\\" in relative
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise AdmissionRunBlocked(
                    f"physics input start snapshot {namespace} entry is invalid"
                )
    current_map = normalized["current"]
    if set(current_map) != EXPECTED_CURRENT_SOURCE_PATHS:
        raise AdmissionRunBlocked("physics input start snapshot current source set differs")
    current_root = current_root.resolve()
    for relative in sorted(EXPECTED_CURRENT_SOURCE_PATHS):
        path = (current_root / relative).resolve()
        if not path.is_relative_to(current_root) or not path.is_file():
            raise AdmissionRunBlocked(f"current source is missing before end check: {relative}")
        if _sha256(path) != current_map[relative]:
            raise AdmissionRunBlocked(f"current source changed before end check: {relative}")
    return current_root / END_VALIDATOR_RELATIVE


def verify_after_child_exit(
    *, output_dir: Path, runtime_root: Path, current_root: Path
) -> dict[str, Any]:
    """Write a CPU-only end check that remains callable after a child crash."""

    output_dir = output_dir.resolve()
    destination = output_dir / "physics-input-end-verification.json"
    identity_path = output_dir / "run-identity.json"
    if not identity_path.is_file():
        result = {
            "schema_name": "rj45_task_admission_physics_input_end_verification_v1",
            "status": "unverifiable_start",
            "matched": False,
            "scientific_completion_allowed": False,
            "run_identity": None,
        }
    else:
        identity = _json(identity_path)
        start = identity.get("physics_input_start")
        try:
            if not isinstance(start, dict):
                raise AdmissionRunBlocked("run identity has no physics input start snapshot")
            validator_path = _validate_end_snapshot_before_validator(
                start, current_root.resolve()
            )
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                "rby1_admission_end_validator", validator_path
            )
            if spec is None or spec.loader is None:
                raise AdmissionRunBlocked("cannot load CPU end validator")
            validator = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = validator
            spec.loader.exec_module(validator)
            result = validator.verify_physics_input_end(
                start, runtime_root.resolve(), current_root.resolve()
            )
            result["scientific_completion_allowed"] = result["matched"]
        except Exception as error:
            result = {
                "schema_name": "rj45_task_admission_physics_input_end_verification_v1",
                "status": "input_mutation_blocked",
                "matched": False,
                "scientific_completion_allowed": False,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        result["run_identity"] = {
            "path": str(identity_path),
            "sha256": _sha256(identity_path),
        }
    if destination.exists():
        existing = _json(destination)
        if existing != result:
            raise AdmissionRunBlocked(
                "existing physics end verification differs from the current CPU check"
            )
    else:
        _atomic_immutable_json(destination, result)
    return {**result, "path": str(destination), "sha256": _sha256(destination)}


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    report_path = args.output_dir / "report.json"
    progress_path = args.output_dir / "progress.json"
    simulation_app = None
    run_identity: dict[str, str] | None = None
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        if args.verify_after_child_exit:
            verification = verify_after_child_exit(
                output_dir=args.output_dir,
                runtime_root=args.runtime_root,
                current_root=args.current_root,
            )
            print(json.dumps(verification, sort_keys=True, allow_nan=False))
            return 0 if verification["matched"] else 2
        if report_path.exists():
            raise FileExistsError(f"report already exists: {report_path}")
        prepared = _prepare(args)
        if args.validate_only:
            _atomic_json(
                report_path,
                {
                    "status": "validated_not_executed",
                    "episode_count": len(prepared["plan"]),
                    "smoke_only": args.smoke_only,
                    "claims": {"task_admission_completed": False},
                },
            )
            return 0

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {"headless": args.headless, "device": args.device}
        ).app
        runtime = _load_runtime(prepared)
        run_identity = _write_run_identity(args.output_dir, prepared, runtime)
        if "physics_input_snapshot" in prepared:
            pre_env_verification = prepared["runtime_validator"].verify_physics_input_end(
                prepared["physics_input_snapshot"],
                prepared["runtime_root"],
                prepared["current_root"],
            )
            if not pre_env_verification["matched"]:
                raise AdmissionRunBlocked("physics inputs changed before first environment")
        rows: list[dict[str, Any]] = []
        episodes_path = args.output_dir / "episodes-progress.jsonl"
        _replace_json(
            progress_path,
            {
                "phase": "physics_ready",
                "completed_episodes": 0,
                "total_episodes": len(prepared["plan"]),
            },
        )
        completed_count = 0

        def execute(plan_rows, *, selected_classical_candidate=None, fatal_on_error=False):
            nonlocal completed_count
            produced = []
            for plan_row in plan_rows:
                ordinal = completed_count + 1
                candidate_id = plan_row[5]
                if plan_row[3] == "classical_force_search" and candidate_id is None:
                    candidate_id = selected_classical_candidate

                def heartbeat(step_index):
                    _replace_json(
                        progress_path,
                        {
                            "phase": "physics_episode",
                            "completed_episodes": ordinal - 1,
                            "total_episodes": len(prepared["plan"]),
                            "current": {
                                "stage": plan_row[0],
                                "grasp_id": plan_row[1],
                                "case_id": plan_row[2],
                                "method_id": plan_row[3],
                                "slot": plan_row[4],
                                "control_step": step_index,
                            },
                        },
                    )

                try:
                    episode = _episode(
                        prepared,
                        runtime,
                        plan_row,
                        candidate_id,
                        heartbeat=heartbeat,
                    )
                except Exception as error:
                    if type(error).__name__ == "TaskAdmissionContractViolation" or fatal_on_error:
                        raise
                    episode = {
                        "stage": plan_row[0],
                        "grasp_id": plan_row[1],
                        "case_id": plan_row[2],
                        "method_id": plan_row[3],
                        "slot": plan_row[4],
                        "candidate_id": candidate_id,
                        "success": None,
                        "hard_contact": None,
                        "forbidden_contact": None,
                        "evaluator_force_p95_n": None,
                        "completion_time_s": None,
                        "provider_inputs": [],
                        "applied_controller_sha256": None,
                        "applied_artifact_sha256": None,
                        "observation_samples": [],
                        "admission_snapshot": {},
                        "terminal_complete": False,
                        "execution_error": {
                            "type": type(error).__name__,
                            "message": str(error),
                        },
                    }
                _persist_episode_row(
                    args.output_dir,
                    ordinal,
                    plan_row,
                    candidate_id,
                    episode,
                    run_identity,
                )
                produced.append(episode)
                completed_count = ordinal
                _append_jsonl(
                    episodes_path,
                    {
                        key: episode[key]
                        for key in (
                            "stage",
                            "grasp_id",
                            "case_id",
                            "method_id",
                            "slot",
                            "candidate_id",
                            "success",
                            "hard_contact",
                            "forbidden_contact",
                            "terminal_complete",
                            "execution_error",
                        )
                    },
                )
                _replace_json(
                    progress_path,
                    {
                        "phase": "episode_flushed",
                        "completed_episodes": ordinal,
                        "total_episodes": len(prepared["plan"]),
                    },
                )
            return produced

        if args.smoke_only:
            rows = execute(prepared["plan"])
            _validate_smoke_rows(rows)
        else:
            admission = runtime["modules"]["rj45_task_admission"]
            tuning_rows_raw = execute(prepared["plan"][:18], fatal_on_error=True)
            _validate_tuning_rows(tuning_rows_raw, prepared["config"])
            tuning_rows = [
                admission.TuningResult(
                    **{
                        key: row[key]
                        for key in (
                            "candidate_id",
                            "case_id",
                            "success",
                            "hard_contact",
                            "forbidden_contact",
                            "evaluator_force_p95_n",
                            "completion_time_s",
                        )
                    }
                )
                for row in tuning_rows_raw
            ]
            protocol = admission.AdmissionProtocol().validate_tuning(tuning_rows)
            tuning_payload = admission.canonical_tuning_report_bytes(tuning_rows)
            protocol = protocol.seal_classical(tuning_payload, tuning_rows)
            baseline_seal = admission.make_baseline_artifact_seal(
                source_trace_gzip_sha256=prepared["config"]["methods"][2][
                    "source_trace_gzip_sha256"
                ],
                mean_constant_action=prepared["config"]["methods"][1][
                    "normalized_right6"
                ],
                playback_selected_rows=len(prepared["playback_actions"]),
            )
            classical_seal_json = {
                "selected_candidate_id": protocol.classical_seal.selected_candidate_id,
                "canonical_controller_config_sha256": (
                    protocol.classical_seal.canonical_controller_config_sha256
                ),
                "tuning_report_sha256": protocol.classical_seal.tuning_report_sha256,
                "selection": asdict(protocol.classical_seal.selection),
            }
            _atomic_json(
                args.output_dir / "tuning-seal.json",
                {
                    "selection": asdict(protocol.selection),
                    "classical_seal": classical_seal_json,
                    "episode_count": 18,
                },
            )
            _replace_json(
                progress_path,
                {
                    "phase": "tuning_sealed",
                    "completed_episodes": 18,
                    "total_episodes": 90,
                    "classical_seal": classical_seal_json,
                    "baseline_artifact_seal": asdict(baseline_seal),
                },
            )
            rows = tuning_rows_raw + execute(
                prepared["plan"][18:],
                selected_classical_candidate=protocol.selection.candidate_id,
            )

        for path, expected in runtime["frozen_hashes"].items():
            if _sha256(path) != expected:
                raise AdmissionRunBlocked(
                    f"frozen held-bank input changed during run: {path}"
                )
        prepared["runtime_validator"].validate_runtime_snapshot(
            prepared["runtime_root"], prepared["runtime_manifest"]
        )
        end_verification = None
        if "physics_input_snapshot" in prepared:
            end_verification = verify_after_child_exit(
                output_dir=args.output_dir,
                runtime_root=prepared["runtime_root"],
                current_root=prepared["current_root"],
            )
            if not end_verification["matched"]:
                raise AdmissionRunBlocked("physics input end verification did not match")
        if args.smoke_only:
            report = {
                "status": "smoke_completed",
                "smoke_only": True,
                "provenance": _provenance(prepared, runtime),
                "run_identity": run_identity,
                "physics_input_end_verification": end_verification,
                "episodes": rows,
                "claims": {
                    "task_admission_completed": False,
                    "rl_pilot_admitted": False,
                },
            }
        else:
            episode_rows = [
                admission.EpisodeResult(
                    grasp_id=row["grasp_id"],
                    case_id=row["case_id"],
                    method_id=row["method_id"],
                    slot=row["slot"],
                    success=row["success"],
                    hard_contact=row["hard_contact"],
                    forbidden_contact=row["forbidden_contact"],
                    evaluator_force_p95_n=row["evaluator_force_p95_n"],
                    completion_time_s=row["completion_time_s"],
                    provider_inputs=frozenset(row["provider_inputs"]),
                    applied_controller_sha256=row["applied_controller_sha256"],
                    applied_artifact_sha256=row["applied_artifact_sha256"],
                    observation_samples=tuple(
                        tuple(sample) for sample in row["observation_samples"]
                    ),
                )
                for row in rows[18:]
            ]
            evidence = _evidence(
                rows[18:],
                True,
                prepared["config"],
                baseline_artifacts_verified=admission.validate_baseline_artifact_seal(
                    baseline_seal
                ),
            )
            protocol = protocol.validate_admission(
                episode_rows, evidence, baseline_seal
            ).adjudicate(episode_rows, evidence)
            report = {
                "status": protocol.decision.state,
                "smoke_only": False,
                "provenance": _provenance(prepared, runtime),
                "run_identity": run_identity,
                "physics_input_end_verification": end_verification,
                "tuning_selection": asdict(protocol.selection),
                "classical_seal": classical_seal_json,
                "baseline_artifact_seal": asdict(baseline_seal),
                "contract_evidence": asdict(evidence),
                "decision": asdict(protocol.decision),
                "episodes": rows,
                "claims": {
                    "task_admission_completed": True,
                    "rl_pilot_admitted": protocol.decision.state == "admit_rl_pilot",
                },
            }
        _atomic_json(report_path, report)
        _replace_json(
            progress_path,
            {
                "phase": "completed",
                "completed_episodes": len(rows),
                "total_episodes": len(prepared["plan"]),
                "status": report["status"],
            },
        )
        return 0 if args.smoke_only or report["status"] != "invalid_contract" else 2
    except KeyboardInterrupt as error:
        if not report_path.exists():
            _atomic_json(
                report_path,
                _failure_report("interrupted", error, progress_path, run_identity),
            )
        return 2
    except Exception as error:
        traceback.print_exc()
        if not report_path.exists():
            status = (
                "invalid_contract"
                if type(error).__name__ == "TaskAdmissionContractViolation"
                else "blocked"
            )
            _atomic_json(
                report_path,
                _failure_report(status, error, progress_path, run_identity),
            )
        return 2
    finally:
        faulthandler.cancel_dump_traceback_later()
        if simulation_app is not None:
            try:
                simulation_app.close()
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise


if __name__ == "__main__":
    raise SystemExit(main())
