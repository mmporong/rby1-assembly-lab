#!/usr/bin/env python3
"""Run one non-adjudicative G1 RJ45 contact-instrumentation case."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping, Sequence


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
    bindings.update({name: _sha256(path) for name, path in paths.items()})
    return bindings


def _geometry(task: Mapping[str, Any], recording: Any) -> Any:
    frames = task.get("frames")
    if not isinstance(frames, Mapping):
        raise RuntimeError("task.frames is required for diagnostic geometry")
    try:
        socket_y = float(frames["socket_entry"]["position_m"][1])
        plug_y = float(frames["plug_tip"]["position_m"][1])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise RuntimeError("task frame geometry is incomplete") from error
    return recording.AssemblyGeometry(socket_entry_y_m=socket_y, plug_tip_y_m=plug_y)


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

        row = DIAGNOSTIC_PLAN[args.ordinal - 1]
        writer = trace_module.JsonlTraceWriter(
            args.output_dir / "ticks.jsonl",
            args.output_dir / "ticks.sidecar.json",
        )
        observer = trace_module.ContactDiagnosticObserver(
            writer,
            _geometry(runtime["task"], recording),
            live_snapshot=recording.live_snapshot,
            terminal_snapshot_from_extras=recording.terminal_snapshot_from_extras,
        )
        episode = runner._episode(
            prepared,
            runtime,
            row,
            None,
            step_observer=observer,
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
