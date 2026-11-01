#!/usr/bin/env python3
"""Run one prospective controlled-observation episode without granting admission."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from typing import Sequence

CURRENT_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ordinal", type=int, choices=range(1, 25))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument("--config", type=Path, default=CURRENT_ROOT / "configs/rj45-controlled-observability-v2.json")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-sources-sha256")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser


def _source_bindings(prepared, diagnostic, config_path):
    bindings = diagnostic._source_bindings(prepared)
    root = prepared["current_root"]
    for name, path in {
        "controlled_runner": Path(__file__).resolve(),
        "controlled_contract": root / "src/rby1_assembly/rj45_controlled_observability.py",
        "active_axis_provider": root / "src/rby1_assembly/rj45_active_axis_probe.py",
        "active_z_base_provider": root / "src/rby1_assembly/rj45_active_z_probe.py",
        "controlled_config": config_path,
    }.items():
        bindings[name] = _sha256(path)
    return bindings


def _source_digest(bindings):
    return hashlib.sha256(json.dumps(bindings, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _check_source_freeze(bindings, expected_digest):
    actual = _source_digest(bindings)
    if actual != expected_digest:
        raise ValueError("source bytes differ from the reviewed source freeze")
    return actual


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.current_root.resolve()
    diagnostic = _load_module("rby1_controlled_diagnostic_helpers", root / "scripts/diagnose_rj45_contact_isaac.py")
    contract = _load_module("rby1_controlled_contract", root / "src/rby1_assembly/rj45_controlled_observability.py")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    digest = contract.validate_contract(config, require_frozen=not args.validate_only)
    if args.expected_config_sha256 is not None and digest != args.expected_config_sha256:
        raise ValueError("controlled config digest differs from requested freeze")
    plan = contract.execution_plan(config, require_frozen=not args.validate_only)
    if args.validate_only:
        print(json.dumps({"config_sha256": digest, "frozen": config["frozen"], "plan": plan}, allow_nan=False))
        return 0
    if args.expected_config_sha256 is None or args.expected_sources_sha256 is None or args.ordinal is None:
        raise ValueError("execution requires expected config SHA and ordinal plus source freeze SHA")
    case = plan[args.ordinal - 1]
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
        prepared = runner._prepare(diagnostic._prepare_args(args))
        prepared["active_z_probe"] = False
        start_sources = _source_bindings(prepared, diagnostic, config_path)
        sources_digest = _check_source_freeze(start_sources, args.expected_sources_sha256)
        if json.loads(config_path.read_text()) != config:
            raise runner.AdmissionRunBlocked("config changed after contract preflight")
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

        runtime_start_attestation = diagnostic._capture_runtime_start_attestation(
            runner, prepared, runtime
        )

        row = ("controlled_observability", case["grasp_id"], case["case_id"],
               "active_axis_probe", case["ordinal"], None)
        provider_path = root / "src/rby1_assembly/rj45_active_axis_probe.py"
        probe = _load_module("rby1_controlled_axis_provider", provider_path)

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
            "diagnostic_provider_factory": lambda: probe.ActiveAxisProbeProvider(case["axis"]),
            "diagnostic_provider_attestor": attest_probe,
            "diagnostic_environment_seed": case["environment_seed"],
        }
        writer = trace_module.JsonlTraceWriter(
            args.output_dir / "ticks.jsonl",
            args.output_dir / "ticks.sidecar.json",
        )
        observer = trace_module.ContactDiagnosticObserver(
            writer,
            diagnostic._geometry(runtime["task"], recording),
            live_snapshot=lambda env, extras, geometry: diagnostic._diagnostic_snapshot(
                recording, env, extras, geometry, terminal=False
            ),
            terminal_snapshot_from_extras=lambda extras, geometry: diagnostic._diagnostic_snapshot(
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
        frozen_bank_end = diagnostic._verify_frozen_bank_end(runner, runtime)
        runtime_import_end = diagnostic._verify_runtime_import_origins_end(
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
        end_sources = _source_bindings(prepared, diagnostic, config_path)
        if end_sources != start_sources:
            raise runner.AdmissionRunBlocked("diagnostic source bytes changed during execution")
        report = {
            "schema_name": "rj45-controlled-observability-report-v2",
            "status": "controlled_observability_complete",
            "diagnostic_only": True,
            "ordinal": args.ordinal,
            "case": case,
            "config": config,
            "config_sha256": digest,
            "actual_environment_seed": episode["environment_seed"],
            "seed_evidence_scope": "environment_cfg_seed_checked; stochastic_independence_not_demonstrated",
            "trace": sidecar,
            "episode_terminal": {
                "success": episode["success"],
                "hard_contact": episode["hard_contact"],
                "forbidden_contact": episode["forbidden_contact"],
                "completion_time_s": episode["completion_time_s"],
            },
            "provenance": {
                "reviewed_source_freeze_sha256": sources_digest,
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
        report["active_axis_probe"] = {
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
            diagnostic._report_failure(runner, report_path, error)
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
