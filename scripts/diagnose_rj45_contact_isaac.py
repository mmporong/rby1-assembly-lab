#!/usr/bin/env python3
"""Run one non-adjudicative G1 RJ45 contact-instrumentation case."""

from __future__ import annotations

import argparse
from dataclasses import replace
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np


CURRENT_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf")
EVIDENCE_ROOT = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-evidence-104721a")
CONFIG_RELATIVE = Path("configs/rj45-comparison-development.json")
PLAYBACK_RELATIVE = Path("docs/evidence/rj45-frozen-ablation-20261031/source-actions.json")
PHYSICS_MANIFEST_RELATIVE = Path(
    "configs/rj45-task-admission-physics-inputs-validation-ulp2-v1.json"
)
DIAGNOSTIC_PLAN = (
    ("diagnostic", "rj45-train-02", "tx-positive", "zero", 1, None),
    ("diagnostic", "rj45-train-02", "tx-negative", "zero", 1, None),
    ("diagnostic", "rj45-train-02", "tz-positive", "zero", 1, None),
    ("diagnostic", "rj45-train-02", "tz-negative", "zero", 1, None),
    ("diagnostic", "rj45-train-02", "rx-positive", "zero", 1, None),
    ("diagnostic", "rj45-train-02", "rx-negative", "zero", 1, None),
    ("diagnostic", "rj45-train-02", "tz-positive", "oracle_gt_reference", 1, None),
    ("diagnostic", "rj45-train-02", "tz-negative", "oracle_gt_reference", 1, None),
)
ACTIVE_Z_PLAN = tuple(
    ("diagnostic", grasp, case, "active_z_probe", repeat, None)
    for grasp in ("rj45-train-02", "rj45-train-05")
    for case in ("tz-positive", "tz-negative")
    for repeat in (1, 2)
)
COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE = (
    "max_of_external_forbidden_and_cross_hand_physx_contact_loads"
)
COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION = (
    "max(external_forbidden_contact_load_n, cross_hand_contact_load_n)"
)
CROSS_HAND_CONTACT_LOAD_SOURCE = (
    "physx_left_hand_sources_filtered_to_right_hand_sum_abs_normal_forces"
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinal", type=int, required=True, choices=range(1, 9))
    parser.add_argument("--active-z-probe", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=RUNTIME_ROOT)
    parser.add_argument("--evidence-root", type=Path, default=EVIDENCE_ROOT)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser


def _prepare_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        runtime_root=args.runtime_root,
        evidence_root=args.evidence_root,
        current_root=args.current_root,
        config=CONFIG_RELATIVE,
        output_dir=args.output_dir,
        physics_input_manifest=PHYSICS_MANIFEST_RELATIVE,
        verify_after_child_exit=False,
        worker_spec=None,
        expected_worker_spec_sha256=None,
        playback_actions=PLAYBACK_RELATIVE,
        smoke_only=False,
        validate_only=False,
        headless=args.headless,
        device=args.device,
    )


def _source_bindings(prepared: Mapping[str, Any]) -> dict[str, str]:
    current_root = Path(prepared["current_root"])
    runtime_root = Path(prepared["runtime_root"])
    current_six = prepared["physics_input_snapshot"]["normalized_sha256"]["current"]
    bindings = {
        f"current6:{relative}": _sha256(current_root / relative)
        for relative in sorted(current_six)
    }
    paths = {
        "diagnostic_runner": Path(__file__).resolve(),
        "admission_runner": current_root / "scripts/run_rj45_task_admission.py",
        "diagnostic_trace": current_root / "src/rby1_assembly/rj45_diagnostic_trace.py",
        "runtime_recording": runtime_root / "src/rby1_assembly/recording.py",
        "runtime_dataset": runtime_root / "src/rby1_assembly/dataset.py",
    }
    if prepared.get("active_z_probe"):
        paths["active_z_provider"] = current_root / "src/rby1_assembly/rj45_active_z_probe.py"
    bindings.update({name: _sha256(path) for name, path in paths.items()})
    return bindings


def _geometry(task: Mapping[str, Any], recording: Any) -> Any:
    frames = task.get("frames")
    if not isinstance(frames, Mapping):
        raise RuntimeError("task.frames is required for diagnostic geometry")
    try:
        socket_position = np.asarray(
            frames["socket_entry"]["position_m"], dtype=np.float64
        )
        socket_quaternion = np.asarray(
            frames["socket_entry"]["quaternion_wxyz"], dtype=np.float64
        )
        plug_position = np.asarray(frames["plug_tip"]["position_m"], dtype=np.float64)
        plug_quaternion = np.asarray(
            frames["plug_tip"]["quaternion_wxyz"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError("task frame geometry is incomplete") from error
    if (
        socket_position.shape != (3,)
        or plug_position.shape != (3,)
        or socket_quaternion.shape != (4,)
        or plug_quaternion.shape != (4,)
        or not all(
            np.isfinite(value).all()
            for value in (
                socket_position,
                plug_position,
                socket_quaternion,
                plug_quaternion,
            )
        )
    ):
        raise RuntimeError("task frame geometry has invalid shape or non-finite values")
    recording.wxyz_to_xyzw(np.stack((socket_quaternion, plug_quaternion)))
    return SimpleNamespace(
        recording=recording.AssemblyGeometry(
            socket_entry_y_m=float(socket_position[1]),
            plug_tip_y_m=float(plug_position[1]),
        ),
        socket_entry_position_m=socket_position,
        socket_entry_quaternion_wxyz=socket_quaternion,
        plug_tip_position_m=plug_position,
        plug_tip_quaternion_wxyz=plug_quaternion,
    )


def _full_frame_geometry_metrics(
    recording: Any, poses_world_wxyz: np.ndarray, geometry: Any
) -> tuple[float, float, float]:
    poses = np.asarray(poses_world_wxyz, dtype=np.float64)
    if poses.shape != (2, 7):
        raise recording.RecordingContractError(
            f"diagnostic object poses must have shape (2, 7), got {poses.shape}"
        )
    socket_position, plug_position = poses[0, :3], poses[1, :3]
    socket_quaternion, plug_quaternion = poses[0, 3:7], poses[1, 3:7]
    recording.wxyz_to_xyzw(np.stack((socket_quaternion, plug_quaternion)))
    entry_position = socket_position + recording._quat_apply_wxyz(
        socket_quaternion, geometry.socket_entry_position_m
    )
    entry_quaternion = recording._quat_multiply_wxyz(
        socket_quaternion, geometry.socket_entry_quaternion_wxyz
    )
    tip_position = plug_position + recording._quat_apply_wxyz(
        plug_quaternion, geometry.plug_tip_position_m
    )
    tip_quaternion = recording._quat_multiply_wxyz(
        plug_quaternion, geometry.plug_tip_quaternion_wxyz
    )
    entry_inverse = entry_quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))
    relative_position = recording._quat_apply_wxyz(
        entry_inverse, tip_position - entry_position
    )
    relative_quaternion = recording._quat_multiply_wxyz(
        entry_inverse, tip_quaternion
    )
    relative_quaternion /= np.linalg.norm(relative_quaternion)
    return (
        float(relative_position[1]),
        float(np.linalg.norm(relative_position[(0, 2),])),
        float(
            2.0
            * np.arctan2(
                np.linalg.norm(relative_quaternion[1:4]),
                abs(relative_quaternion[0]),
            )
        ),
    )


def _validate_forbidden_contact_provenance(
    recording: Any, source: Mapping[str, Any], env_index: int = 0
) -> None:
    """Validate the passive hand monitor without rewriting its source label."""

    actual_source = source.get("forbidden_contact_load_source")
    if actual_source == recording.FORBIDDEN_CONTACT_LOAD_SOURCE:
        return
    if actual_source != COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE:
        raise recording.RecordingContractError(
            "diagnostic forbidden contact-load source is unsupported"
        )
    required = {
        "forbidden_contact_load_n",
        "external_forbidden_contact_load_n",
        "external_forbidden_contact_load_peak_n",
        "cross_hand_contact_load_n",
        "cross_hand_contact_load_peak_n",
    }
    missing = sorted(required - set(source))
    if missing:
        raise recording.RecordingContractError(
            "combined forbidden contact evidence is incomplete: " + ", ".join(missing)
        )
    if (
        source.get("forbidden_contact_load_definition")
        != COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION
        or source.get("cross_hand_contact_load_source")
        != CROSS_HAND_CONTACT_LOAD_SOURCE
    ):
        raise recording.RecordingContractError(
            "combined forbidden contact provenance differs"
        )
    combined = float(recording._scalar(source["forbidden_contact_load_n"], env_index))
    external = float(
        recording._scalar(source["external_forbidden_contact_load_n"], env_index)
    )
    cross_hand = float(
        recording._scalar(source["cross_hand_contact_load_n"], env_index)
    )
    if not all(
        np.isfinite(value) and value >= 0.0
        for value in (combined, external, cross_hand)
    ):
        raise recording.RecordingContractError(
            "combined forbidden contact loads must be finite and non-negative"
        )
    if not np.isclose(combined, max(external, cross_hand), atol=1e-6, rtol=1e-6):
        raise recording.RecordingContractError(
            "combined forbidden contact load differs from its measured components"
        )


def _diagnostic_snapshot(
    recording: Any,
    env: Any,
    extras: Mapping[str, Any],
    geometry: Any,
    *,
    terminal: bool,
    env_index: int = 0,
) -> Any:
    """Read only the same-tick physical fields used by contact diagnostics."""

    source = extras.get("terminal_snapshot") if terminal else extras
    if not isinstance(source, Mapping):
        raise recording.RecordingContractError("diagnostic physical source is missing")
    evaluation = source.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise recording.RecordingContractError("diagnostic evaluation source is missing")
    if terminal:
        poses = np.asarray(
            recording._env_value(source["object_poses_world_wxyz"], env_index),
            dtype=np.float64,
        )
    else:
        poses = np.stack(
            (
                np.concatenate(
                    (
                        recording._env_value(env.socket.data.root_pos_w, env_index),
                        recording._env_value(env.socket.data.root_quat_w, env_index),
                    )
                ),
                np.concatenate(
                    (
                        recording._env_value(env.plug.data.root_pos_w, env_index),
                        recording._env_value(env.plug.data.root_quat_w, env_index),
                    )
                ),
            )
        )
    base_snapshot = None
    if (
        source.get("forbidden_contact_load_source")
        == recording.FORBIDDEN_CONTACT_LOAD_SOURCE
    ):
        if terminal:
            base_snapshot = recording.terminal_snapshot_from_extras(
                extras, geometry.recording, env_index
            )
        else:
            base_snapshot = recording.live_snapshot(
                env, extras, geometry.recording, env_index
            )
    elif source.get("insertion_contact_force_source") != recording.INSERTION_FORCE_SOURCE:
        raise recording.RecordingContractError(
            "diagnostic insertion contact-force source is unsupported"
        )
    elif source.get("insertion_contact_load_source") != recording.INSERTION_LOAD_SOURCE:
        raise recording.RecordingContractError(
            "diagnostic insertion contact-load source is unsupported"
        )
    else:
        _validate_forbidden_contact_provenance(recording, source, env_index)

    if base_snapshot is not None:
        force_w = base_snapshot.insertion_contact_force_w
        force_n = base_snapshot.insertion_contact_force_n
        load_n = base_snapshot.insertion_contact_load_n
        grip_n = base_snapshot.contact_force_proxy_n
        phase = base_snapshot.phase
    elif terminal:
        force_w = recording._env_value(source["insertion_contact_force_w"], env_index)
        force_n = recording._scalar(source["insertion_contact_force_n"], env_index)
        load_n = recording._scalar(source["insertion_contact_load_n"], env_index)
        grip_n = recording._env_value(source["contact_force_proxy_n"], env_index)
        phase = recording._phase_name(
            recording._env_value(source["phase"], env_index)
        )
    else:
        force_w = recording._env_value(env._last_insertion_contact_force_w, env_index)
        force_n = recording._scalar(env._last_insertion_contact_force_n, env_index)
        load_n = recording._scalar(env._last_insertion_contact_load_n, env_index)
        grip_n = recording._env_value(env._last_contact_proxy_n, env_index)
        phase = recording._phase_name(recording._env_value(env._phase, env_index))

    def reported_field(*names: str) -> float:
        for name in names:
            if name in evaluation:
                return float(recording._scalar(evaluation[name], env_index))
        raise recording.RecordingContractError(
            "diagnostic reported geometry field is missing: " + " or ".join(names)
        )

    reported = SimpleNamespace(
        actual_depth_m=reported_field("actual_depth_m"),
        lateral_error_m=reported_field(
            "lateral_error_m", "actual_lateral_error_m"
        ),
        orientation_error_rad=reported_field(
            "orientation_error_rad", "actual_orientation_error_rad"
        ),
    )
    if base_snapshot is not None:
        legacy_metrics = base_snapshot.ground_truth
        legacy_consistent = base_snapshot.reported_geometry_consistent
    else:
        legacy_metrics, legacy_consistent = recording._metrics_from_mapping(
            evaluation, poses, geometry.recording, env_index=0
        )
    full_values = _full_frame_geometry_metrics(recording, poses, geometry)
    metrics = replace(
        legacy_metrics,
        actual_depth_m=full_values[0],
        lateral_error_m=full_values[1],
        orientation_error_rad=full_values[2],
    )
    full_consistent = (
        abs(reported.actual_depth_m - metrics.actual_depth_m)
        <= geometry.recording.metric_tolerance_m
        and abs(reported.lateral_error_m - metrics.lateral_error_m)
        <= geometry.recording.metric_tolerance_m
        and abs(reported.orientation_error_rad - metrics.orientation_error_rad)
        <= geometry.recording.angular_tolerance_rad
    )
    return SimpleNamespace(
        insertion_contact_force_w=np.asarray(force_w, dtype=np.float64),
        insertion_contact_force_n=float(force_n),
        insertion_contact_load_n=float(load_n),
        contact_force_proxy_n=np.asarray(grip_n, dtype=np.float64),
        phase=phase,
        ground_truth=metrics,
        reported_ground_truth=reported,
        reported_geometry_consistent=bool(full_consistent),
        legacy_y_only_ground_truth=legacy_metrics,
        legacy_y_only_geometry_consistent=bool(legacy_consistent),
        geometry_metric_tolerance_m=float(geometry.recording.metric_tolerance_m),
        geometry_angular_tolerance_rad=float(
            geometry.recording.angular_tolerance_rad
        ),
    )


def _capture_runtime_start_attestation(
    runner: Any, prepared: Mapping[str, Any], runtime: Mapping[str, Any]
) -> dict[str, Any]:
    attestation = runner._provenance(prepared, runtime)
    required = {
        "runtime_versions",
        "runtime_import_origins",
        "held_bank_proof",
        "frozen_input_sha256",
    }
    if not isinstance(attestation, dict) or not required.issubset(attestation):
        raise runner.AdmissionRunBlocked("runtime start attestation is incomplete")
    origins = attestation["runtime_import_origins"]
    if not isinstance(origins, dict) or not {
        "rby1_assembly.recording",
        "rby1_assembly.dataset",
    }.issubset(origins):
        raise runner.AdmissionRunBlocked(
            "runtime recording/dataset import origins are missing at start"
        )
    frozen = attestation["frozen_input_sha256"]
    if not isinstance(frozen, dict) or len(frozen) != 26:
        raise runner.AdmissionRunBlocked(
            "runtime start attestation must bind exactly 26 frozen bank inputs"
        )
    return attestation


def _verify_frozen_bank_end(
    runner: Any, runtime: Mapping[str, Any]
) -> dict[str, Any]:
    frozen = runtime.get("frozen_hashes")
    if not isinstance(frozen, Mapping) or len(frozen) != 26:
        raise runner.AdmissionRunBlocked(
            "runtime frozen bank map must contain exactly 26 inputs"
        )
    for raw_path, expected in frozen.items():
        path = Path(raw_path).resolve()
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or _sha256(path) != expected
        ):
            raise runner.AdmissionRunBlocked(
                f"frozen held-bank input changed during diagnostic: {path}"
            )
    return {
        "schema_name": "rj45-contact-diagnostic-frozen-bank-end-v1",
        "matched": True,
        "verified_file_count": len(frozen),
        "source": "runtime.frozen_hashes_from_verified_held_grasp_bank_dependencies",
    }


def _verify_runtime_import_origins_end(
    runner: Any,
    prepared: Mapping[str, Any],
    start_attestation: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_root = Path(prepared["runtime_root"]).resolve()
    start = start_attestation.get("runtime_import_origins")
    if not isinstance(start, Mapping):
        raise runner.AdmissionRunBlocked("runtime start import origins are invalid")
    end = runner._assert_runtime_import_origins(runtime_root)
    runtime_manifest = prepared["physics_input_snapshot"]["normalized_sha256"][
        "runtime"
    ]
    late_manifest_bound: dict[str, str] = {}
    for name, raw_path in end.items():
        path = Path(raw_path).resolve()
        if not path.is_relative_to(runtime_root):
            raise runner.AdmissionRunBlocked(
                f"runtime import escaped at diagnostic end: {name}"
            )
        if name in start:
            if str(path) != start[name]:
                raise runner.AdmissionRunBlocked(
                    f"runtime import origin changed during diagnostic: {name}"
                )
            continue
        relative = path.relative_to(runtime_root).as_posix()
        if runtime_manifest.get(relative) != _sha256(path):
            raise runner.AdmissionRunBlocked(
                f"late runtime import is not bound by the physics manifest: {name}"
            )
        late_manifest_bound[name] = str(path)
    missing = sorted(set(start) - set(end))
    if missing:
        raise runner.AdmissionRunBlocked(
            "runtime imports disappeared during diagnostic: " + ", ".join(missing)
        )
    return {
        "schema_name": "rj45-contact-diagnostic-runtime-import-end-v1",
        "matched": True,
        "start_origin_count": len(start),
        "end_origin_count": len(end),
        "late_manifest_bound_origins": late_manifest_bound,
        "runtime_import_origins_end": end,
    }


def _report_failure(runner: Any, report_path: Path, error: BaseException) -> None:
    if report_path.exists():
        return
    runner._atomic_json(
        report_path,
        {
            "schema_name": "rj45-contact-diagnostic-report-v1",
            "status": "blocked",
            "error": {"type": type(error).__name__, "message": str(error)},
            "diagnostic_only": True,
            "claims": {
                "task_admission_completed": False,
                "rl_pilot_admitted": False,
            },
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir = args.output_dir.resolve()
    report_path = args.output_dir / "report.json"
    output_preexisted = args.output_dir.exists()
    simulation_app = None
    runner = _load_module(
        "rby1_contact_diagnostic_admission_runner",
        args.current_root.resolve() / "scripts/run_rj45_task_admission.py",
    )
    trace_module = _load_module(
        "rby1_contact_diagnostic_trace",
        args.current_root.resolve() / "src/rby1_assembly/rj45_diagnostic_trace.py",
    )
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        if output_preexisted:
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        prepared = runner._prepare(_prepare_args(args))
        prepared["active_z_probe"] = args.active_z_probe
        start_sources = _source_bindings(prepared)
        pre_physics = prepared["runtime_validator"].verify_physics_input_end(
            prepared["physics_input_snapshot"],
            prepared["runtime_root"],
            prepared["evidence_root"],
            prepared["current_root"],
        )
        if not pre_physics["matched"]:
            raise runner.AdmissionRunBlocked("physics inputs changed before diagnostic launch")

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {"headless": args.headless, "device": args.device}
        ).app
        runtime = runner._load_runtime(prepared)
        from rby1_assembly import recording

        runtime_start_attestation = _capture_runtime_start_attestation(
            runner, prepared, runtime
        )

        row = (ACTIVE_Z_PLAN if args.active_z_probe else DIAGNOSTIC_PLAN)[args.ordinal - 1]
        provider_kwargs = {}
        if args.active_z_probe:
            provider_path = prepared["current_root"] / "src/rby1_assembly/rj45_active_z_probe.py"
            probe = _load_module("rby1_active_z_diagnostic_provider", provider_path)

            def attest_probe(provider):
                return {
                    "diagnostic_only": True,
                    "applied_controller_sha256": provider.controller_sha256(),
                    "applied_artifact_sha256": None,
                    "controller_payload": provider.canonical_payload(),
                    "provider_source_sha256": _sha256(provider_path),
                    "provider_import_origin": str(Path(probe.__file__).resolve()),
                }

            provider_kwargs = {
                "diagnostic_provider_factory": probe.ActiveZProbeProvider,
                "diagnostic_provider_attestor": attest_probe,
            }
        writer = trace_module.JsonlTraceWriter(
            args.output_dir / "ticks.jsonl",
            args.output_dir / "ticks.sidecar.json",
        )
        observer = trace_module.ContactDiagnosticObserver(
            writer,
            _geometry(runtime["task"], recording),
            live_snapshot=lambda env, extras, geometry: _diagnostic_snapshot(
                recording, env, extras, geometry, terminal=False
            ),
            terminal_snapshot_from_extras=lambda extras, geometry: _diagnostic_snapshot(
                recording, None, extras, geometry, terminal=True
            ),
        )
        episode = runner._episode(
            prepared,
            runtime,
            row,
            None,
            step_observer=observer,
            **provider_kwargs,
        )
        sidecar = observer.finalize()
        frozen_bank_end = _verify_frozen_bank_end(runner, runtime)
        runtime_import_end = _verify_runtime_import_origins_end(
            runner, prepared, runtime_start_attestation
        )
        end_physics = prepared["runtime_validator"].verify_physics_input_end(
            prepared["physics_input_snapshot"],
            prepared["runtime_root"],
            prepared["evidence_root"],
            prepared["current_root"],
        )
        if not end_physics["matched"]:
            raise runner.AdmissionRunBlocked("physics inputs changed during diagnostic execution")
        end_sources = _source_bindings(prepared)
        if end_sources != start_sources:
            raise runner.AdmissionRunBlocked("diagnostic source bytes changed during execution")
        report = {
            "schema_name": "rj45-contact-diagnostic-report-v1",
            "status": "diagnostic_complete",
            "diagnostic_only": True,
            "ordinal": args.ordinal,
            "case": {
                "grasp_id": row[1],
                "case_id": row[2],
                "method_id": row[3],
                "repeat": row[4],
                "environment_seed": runner.TASK_ADMISSION_ENVIRONMENT_SEED,
                "provider_inputs": episode["provider_inputs"],
            },
            "trace": sidecar,
            "episode_terminal": {
                "success": episode["success"],
                "hard_contact": episode["hard_contact"],
                "forbidden_contact": episode["forbidden_contact"],
                "completion_time_s": episode["completion_time_s"],
            },
            "provenance": {
                "runtime_commit": runner.EXPECTED_RUNTIME_COMMIT,
                "physics_input_snapshot": prepared["physics_input_snapshot"],
                "runtime_start_attestation": runtime_start_attestation,
                "runtime_import_origins_end_verification": runtime_import_end,
                "frozen_bank_end_verification": frozen_bank_end,
                "physics_input_prelaunch_verification": pre_physics,
                "physics_input_end_verification": end_physics,
                "source_sha256_start": start_sources,
                "source_sha256_end": end_sources,
            },
            "claims": {
                "task_admission_completed": False,
                "rl_pilot_admitted": False,
            },
        }
        if args.active_z_probe:
            report["active_z_probe"] = {
                "controller_payload": episode["controller_payload"],
                "controller_sha256": episode["applied_controller_sha256"],
                "provider_source_sha256": episode["provider_source_sha256"],
                "provider_import_origin": episode["provider_import_origin"],
                "steps": episode["diagnostic_provider_steps"],
            }
        runner._atomic_json(report_path, report)
        return 0
    except Exception as error:
        traceback.print_exc()
        if not output_preexisted:
            _report_failure(runner, report_path, error)
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
