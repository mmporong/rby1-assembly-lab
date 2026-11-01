#!/usr/bin/env python3
"""Run one source-frozen force-slope development or confirmation episode."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback
from typing import Mapping, Sequence


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


def _source_digest(bindings):
    return hashlib.sha256(json.dumps(
        bindings, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _check_source_freeze(bindings, expected_digest):
    actual = _source_digest(bindings)
    if actual != expected_digest:
        raise ValueError("source bytes differ from the reviewed source freeze")
    return actual


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("development", "confirmation"), required=True)
    parser.add_argument("--ordinal", type=int, choices=range(1, 25))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument(
        "--config", type=Path,
        default=CURRENT_ROOT / "configs/rj45-force-slope-observability-v3.json",
    )
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-sources-sha256")
    parser.add_argument("--verified-development-results", type=Path)
    parser.add_argument("--expected-development-results-sha256")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser


def _source_bindings(prepared, diagnostic, config_path):
    root = Path(prepared["current_root"])
    bindings = diagnostic._source_bindings(prepared)
    paths = {
        "force_slope_runner": Path(__file__).resolve(),
        "task_admission_runner": root / "scripts/run_rj45_task_admission.py",
        "diagnostic_helpers": root / "scripts/diagnose_rj45_contact_isaac.py",
        "diagnostic_trace": root / "src/rby1_assembly/rj45_diagnostic_trace.py",
        "force_slope_contract": root / "src/rby1_assembly/rj45_force_slope_observability.py",
        "force_slope_analysis": root / "src/rby1_assembly/rj45_force_slope_analysis.py",
        "force_slope_provider": root / "src/rby1_assembly/rj45_force_slope_probe.py",
        "active_axis_provider": root / "src/rby1_assembly/rj45_active_axis_probe.py",
        "active_z_base_provider": root / "src/rby1_assembly/rj45_active_z_probe.py",
        "controlled_v2_contract": root / "src/rby1_assembly/rj45_controlled_observability.py",
        "force_slope_config": config_path,
    }
    bindings.update({name: _sha256(path) for name, path in paths.items()})
    return bindings


def _validate_development_gate(analysis, config, path, expected_sha256):
    if path is None or expected_sha256 is None:
        raise ValueError("confirmation requires verified development results and expected SHA")
    path = path.resolve()
    if _sha256(path) != expected_sha256:
        raise ValueError("verified development results SHA differs")
    aggregate = json.loads(path.read_text())
    if not isinstance(aggregate, Mapping) or not isinstance(aggregate.get("results"), list):
        raise ValueError("verified development results have invalid structure")
    recomputed = analysis.adjudicate_stage(config, "development", aggregate["results"])
    if recomputed != aggregate:
        raise ValueError("verified development aggregate differs from raw recomputation")
    if (
        aggregate.get("status") != "development_pass"
        or aggregate.get("claims") != {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        }
    ):
        raise ValueError("development result does not admit confirmation")
    expected_sources_sha256 = aggregate.get("expected_sources_sha256")
    if (
        type(expected_sources_sha256) is not str
        or len(expected_sources_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sources_sha256)
    ):
        raise ValueError("development aggregate has an invalid source freeze SHA")
    return {
        "path": str(path),
        "sha256": expected_sha256,
        "status": aggregate["status"],
        "expected_sources_sha256": expected_sources_sha256,
    }


def _provider_binding(case, root, force_slope, active_axis):
    if case["provider_id"] == "force_slope_probe":
        path = root / "src/rby1_assembly/rj45_force_slope_probe.py"
        module = force_slope
        provider = module.ForceSlopeProbeProvider()
    elif case["provider_id"] == "active_axis_probe" and case["axis"] == "tz":
        path = root / "src/rby1_assembly/rj45_active_axis_probe.py"
        module = active_axis
        provider = module.ActiveAxisProbeProvider("tz")
    else:
        raise ValueError("case has an unregistered provider binding")
    return {
        "provider": provider,
        "module": module,
        "path": path,
        "controller_payload": provider.canonical_payload(),
        "controller_sha256": provider.controller_sha256(),
        "provider_source_sha256": _sha256(path),
        "provider_import_origin": str(Path(module.__file__).resolve()),
    }


def _attest_provider(provider, binding):
    actual = {
        "controller_payload": provider.canonical_payload(),
        "controller_sha256": provider.controller_sha256(),
        "provider_source_sha256": _sha256(binding["path"]),
        "provider_import_origin": str(Path(binding["module"].__file__).resolve()),
    }
    expected = {key: binding[key] for key in actual}
    if actual != expected:
        raise ValueError("diagnostic provider differs from the reviewed binding")
    return {
        "diagnostic_only": True,
        "applied_controller_sha256": actual["controller_sha256"],
        "applied_artifact_sha256": None,
        **actual,
    }


def _check_episode_provider(episode, binding):
    actual = {
        "controller_payload": episode.get("controller_payload"),
        "controller_sha256": episode.get("applied_controller_sha256"),
        "provider_source_sha256": episode.get("provider_source_sha256"),
        "provider_import_origin": episode.get("provider_import_origin"),
    }
    expected = {key: binding[key] for key in actual}
    if actual != expected:
        raise ValueError("episode provider evidence differs from the reviewed binding")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.current_root.resolve()
    diagnostic = _load_module(
        "rby1_force_slope_diagnostic_helpers",
        root / "scripts/diagnose_rj45_contact_isaac.py",
    )
    contract = _load_module(
        "rby1_force_slope_contract",
        root / "src/rby1_assembly/rj45_force_slope_observability.py",
    )
    analysis = _load_module(
        "rby1_force_slope_analysis",
        root / "src/rby1_assembly/rj45_force_slope_analysis.py",
    )
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    config_sha256 = contract.validate_contract(
        config, require_frozen=not args.validate_only
    )
    if (
        args.expected_config_sha256 is not None
        and args.expected_config_sha256 != config_sha256
    ):
        raise ValueError("force-slope config digest differs")
    plan = contract.execution_plan(
        config, args.stage, require_frozen=not args.validate_only
    )
    if args.stage == "development":
        if (
            args.verified_development_results is not None
            or args.expected_development_results_sha256 is not None
        ):
            raise ValueError("development cannot receive prior development results")
        development_gate = None
    elif args.validate_only and args.verified_development_results is None:
        development_gate = None
    else:
        development_gate = _validate_development_gate(
            analysis,
            config,
            args.verified_development_results,
            args.expected_development_results_sha256,
        )
    if args.validate_only:
        print(json.dumps({
            "stage": args.stage,
            "config_sha256": config_sha256,
            "frozen": config["frozen"],
            "plan": plan,
            "development_gate": development_gate,
        }, allow_nan=False))
        return 0
    if (
        args.ordinal is None
        or args.expected_config_sha256 is None
        or args.expected_sources_sha256 is None
    ):
        raise ValueError("execution requires ordinal and expected config/source SHAs")
    if args.ordinal > len(plan):
        raise ValueError("ordinal is outside the selected stage plan")
    case = plan[args.ordinal - 1]
    args.output_dir = args.output_dir.resolve()
    report_path = args.output_dir / "report.json"
    output_preexisted = args.output_dir.exists()
    simulation_app = None
    runner = _load_module(
        "rby1_force_slope_task_admission_runner",
        root / "scripts/run_rj45_task_admission.py",
    )
    trace_module = _load_module(
        "rby1_force_slope_trace",
        root / "src/rby1_assembly/rj45_diagnostic_trace.py",
    )
    force_slope = _load_module(
        "rby1_force_slope_provider",
        root / "src/rby1_assembly/rj45_force_slope_probe.py",
    )
    active_axis = _load_module(
        "rby1_force_slope_active_axis_provider",
        root / "src/rby1_assembly/rj45_active_axis_probe.py",
    )
    binding = _provider_binding(case, root, force_slope, active_axis)
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        if output_preexisted:
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        prepared = runner._prepare(diagnostic._prepare_args(args))
        prepared["active_z_probe"] = False
        start_sources = _source_bindings(prepared, diagnostic, config_path)
        reviewed_sources_sha256 = _check_source_freeze(
            start_sources, args.expected_sources_sha256
        )
        if development_gate is not None:
            if _sha256(Path(development_gate["path"])) != development_gate["sha256"]:
                raise runner.AdmissionRunBlocked(
                    "verified development results changed before launch"
                )
            if (
                development_gate["expected_sources_sha256"]
                != reviewed_sources_sha256
            ):
                raise runner.AdmissionRunBlocked(
                    "development results use a different reviewed source freeze"
                )
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
        row = (
            "force_slope_observability_v3",
            case["grasp_id"],
            case["case_id"],
            case["provider_id"],
            case["ordinal"],
            None,
        )

        def provider_factory():
            if case["provider_id"] == "force_slope_probe":
                return force_slope.ForceSlopeProbeProvider()
            return active_axis.ActiveAxisProbeProvider("tz")

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
            diagnostic_provider_factory=provider_factory,
            diagnostic_provider_attestor=lambda provider: _attest_provider(provider, binding),
            diagnostic_environment_seed=case["environment_seed"],
        )
        _check_episode_provider(episode, binding)
        if episode["environment_seed"] != case["environment_seed"]:
            raise runner.AdmissionRunBlocked("actual environment seed differs from plan")
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
        if (
            development_gate is not None
            and _sha256(Path(development_gate["path"])) != development_gate["sha256"]
        ):
            raise runner.AdmissionRunBlocked(
                "verified development results changed during execution"
            )
        report = {
            "schema_name": "rj45-force-slope-observability-report-v3",
            "status": "force_slope_observability_complete",
            "diagnostic_only": True,
            "stage": args.stage,
            "ordinal": args.ordinal,
            "case": case,
            "config": config,
            "config_sha256": config_sha256,
            "actual_environment_seed": episode["environment_seed"],
            "seed_evidence_scope": (
                "environment_cfg_seed_checked; stochastic_independence_not_demonstrated"
            ),
            "trace": sidecar,
            "episode_terminal": {
                "success": episode["success"],
                "hard_contact": episode["hard_contact"],
                "forbidden_contact": episode["forbidden_contact"],
                "completion_time_s": episode["completion_time_s"],
            },
            "provenance": {
                "reviewed_source_freeze_sha256": reviewed_sources_sha256,
                "verified_development_results_sha256": (
                    None if development_gate is None else development_gate["sha256"]
                ),
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
            "probe": {
                "controller_payload": episode["controller_payload"],
                "controller_sha256": episode["applied_controller_sha256"],
                "provider_source_sha256": episode["provider_source_sha256"],
                "provider_import_origin": episode["provider_import_origin"],
                "steps": episode["diagnostic_provider_steps"],
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
