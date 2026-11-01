#!/usr/bin/env python3
"""Run one source-frozen classical-v2 tuning or fresh-admission episode."""

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
    parser.add_argument("--stage", choices=("tuning", "admission"), required=True)
    parser.add_argument("--ordinal", type=int, choices=range(1, 25))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, default=CURRENT_ROOT)
    parser.add_argument(
        "--config", type=Path,
        default=CURRENT_ROOT / "configs/rj45-classical-tuning-v2.json",
    )
    parser.add_argument("--verified-tuning-rows", type=Path)
    parser.add_argument("--expected-tuning-rows-sha256")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-sources-sha256")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser


def _source_bindings(prepared, diagnostic, config_path, verified_rows_path):
    bindings = diagnostic._source_bindings(prepared)
    root = prepared["current_root"]
    for name, path in {
        "classical_v2_runner": Path(__file__).resolve(),
        "classical_v2_contract": root / "src/rby1_assembly/rj45_classical_v2_contract.py",
        "classical_v2_provider": root / "src/rby1_assembly/rj45_classical_v2.py",
        "classical_v2_base_provider": root / "src/rby1_assembly/rj45_comparison_providers.py",
        "classical_v2_config": config_path,
    }.items():
        bindings[name] = _sha256(path)
    if verified_rows_path is not None:
        bindings["verified_tuning_rows"] = _sha256(verified_rows_path)
    return bindings


def _source_digest(bindings):
    encoded = json.dumps(
        bindings, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _check_source_freeze(bindings, expected_digest):
    actual = _source_digest(bindings)
    if actual != expected_digest:
        raise ValueError("source bytes differ from the reviewed source freeze")
    return actual


def _check_provider_binding(episode, case, binding, sources, provider_path):
    expected = next(row for row in binding["candidates"] if row["candidate_id"] == case["candidate_id"])
    if (
        type(episode.get("candidate_id")) is not str
        or episode["candidate_id"] != case["candidate_id"]
        or episode.get("applied_controller_sha256") != case["controller_sha256"]
        or episode.get("provider_source_sha256") != sources["classical_v2_provider"]
        or episode.get("provider_import_origin") != str(provider_path.resolve())
        or _source_digest(episode.get("controller_payload")) != _source_digest(expected["canonical_payload"])
    ):
        raise ValueError("classical-v2 provider evidence differs from the selected plan slot")


def _plan(contract, config, stage, rows_path, rows_sha, *, require_frozen):
    binding = contract.validate_classical_v2_config(
        config, require_frozen=require_frozen
    )
    if stage == "tuning":
        if rows_path is not None or rows_sha is not None:
            raise ValueError("tuning stage cannot receive verified tuning rows")
        plan = contract.tuning_execution_plan(
            config, require_frozen=require_frozen
        )
    else:
        if rows_path is None or rows_sha is None:
            raise ValueError(
                "admission requires verified tuning rows and its expected SHA"
            )
        rows_path = rows_path.resolve()
        if _sha256(rows_path) != rows_sha:
            raise ValueError("verified tuning rows file SHA differs")
        verified_rows = json.loads(rows_path.read_text())
        selected = contract.select_and_build_admission_plan(config, verified_rows)
        plan = selected["episodes"]
        if selected["claims"] != {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        }:
            raise ValueError("classical-v2 planner claims differ")
    return binding, plan


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.current_root.resolve()
    diagnostic = _load_module(
        "rby1_classical_v2_diagnostic_helpers",
        root / "scripts/diagnose_rj45_contact_isaac.py",
    )
    contract = _load_module(
        "rby1_classical_v2_contract",
        root / "src/rby1_assembly/rj45_classical_v2_contract.py",
    )
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    rows_path = (
        args.verified_tuning_rows.resolve()
        if args.verified_tuning_rows is not None else None
    )
    binding, plan = _plan(
        contract, config, args.stage, rows_path,
        args.expected_tuning_rows_sha256,
        require_frozen=not args.validate_only,
    )
    config_digest = binding["registered_config_sha256"]
    if (args.expected_config_sha256 is not None
            and config_digest != args.expected_config_sha256):
        raise ValueError("classical-v2 registered config digest differs")
    if args.validate_only:
        print(json.dumps({
            "stage": args.stage,
            "config_sha256": config_digest,
            "frozen": config["frozen"],
            "plan": plan,
            "source_binding": binding,
        }, allow_nan=False))
        return 0
    if (args.expected_config_sha256 is None
            or args.expected_sources_sha256 is None
            or args.ordinal is None):
        raise ValueError(
            "execution requires expected config/source SHAs and ordinal"
        )
    if args.ordinal > len(plan):
        raise ValueError("ordinal is outside the selected stage plan")
    case = plan[args.ordinal - 1]
    args.output_dir = args.output_dir.resolve()
    report_path = args.output_dir / "report.json"
    output_preexisted = args.output_dir.exists()
    simulation_app = None
    runner = _load_module(
        "rby1_classical_v2_admission_runner",
        root / "scripts/run_rj45_task_admission.py",
    )
    trace_module = _load_module(
        "rby1_classical_v2_trace",
        root / "src/rby1_assembly/rj45_diagnostic_trace.py",
    )
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        if output_preexisted:
            raise FileExistsError(f"output directory already exists: {args.output_dir}")
        prepared = runner._prepare(diagnostic._prepare_args(args))
        prepared["active_z_probe"] = False
        start_sources = _source_bindings(
            prepared, diagnostic, config_path, rows_path
        )
        sources_digest = _check_source_freeze(
            start_sources, args.expected_sources_sha256
        )
        if json.loads(config_path.read_text()) != config:
            raise runner.AdmissionRunBlocked("config changed after contract preflight")
        if (rows_path is not None
                and _sha256(rows_path) != args.expected_tuning_rows_sha256):
            raise runner.AdmissionRunBlocked(
                "verified tuning rows changed after contract preflight"
            )
        pre_physics = prepared["runtime_validator"].verify_physics_input_end(
            prepared["physics_input_snapshot"], prepared["runtime_root"],
            prepared["evidence_root"], prepared["current_root"],
        )
        if not pre_physics["matched"]:
            raise runner.AdmissionRunBlocked(
                "physics inputs changed before classical-v2 launch"
            )

        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {"headless": args.headless, "device": args.device}
        ).app
        runtime = runner._load_runtime(prepared)
        from rby1_assembly import recording

        runtime_start_attestation = diagnostic._capture_runtime_start_attestation(
            runner, prepared, runtime
        )
        stage_id = (
            "classical_v2_tuning" if args.stage == "tuning"
            else "classical_v2_new_admission_baseline"
        )
        row = (
            stage_id, case["grasp_id"], case["case_id"],
            "classical_force_search_v2", case["ordinal"], case["candidate_id"],
        )
        provider_path = root / "src/rby1_assembly/rj45_classical_v2.py"
        provider_module = _load_module(
            "rby1_classical_v2_episode_provider", provider_path
        )

        def attest_provider(provider):
            payload = provider.canonical_payload()
            attestation = {
                "comparison_only": True,
                "applied_controller_sha256": provider.controller_sha256(),
                "applied_artifact_sha256": None,
                "controller_payload": payload,
                "provider_source_sha256": _sha256(provider_path),
                "provider_import_origin": str(Path(provider_module.__file__).resolve()),
            }
            _check_provider_binding(
                {"candidate_id": provider.candidate_id, **attestation},
                case, binding, start_sources, provider_path,
            )
            return attestation

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
            prepared, runtime, row, case["candidate_id"],
            step_observer=observer,
            comparison_provider_factory=lambda: provider_module.ClassicalForceSearchV2Provider(
                case["candidate_id"]
            ),
            comparison_provider_attestor=attest_provider,
        )
        sidecar = observer.finalize()
        frozen_bank_end = diagnostic._verify_frozen_bank_end(runner, runtime)
        runtime_import_end = diagnostic._verify_runtime_import_origins_end(
            runner, prepared, runtime_start_attestation
        )
        end_physics = prepared["runtime_validator"].verify_physics_input_end(
            prepared["physics_input_snapshot"], prepared["runtime_root"],
            prepared["evidence_root"], prepared["current_root"],
        )
        if not end_physics["matched"]:
            raise runner.AdmissionRunBlocked(
                "physics inputs changed during classical-v2 execution"
            )
        end_sources = _source_bindings(
            prepared, diagnostic, config_path, rows_path
        )
        if end_sources != start_sources:
            raise runner.AdmissionRunBlocked(
                "classical-v2 source bytes changed during execution"
            )
        if (type(episode.get("environment_seed")) is not int
                or episode["environment_seed"] != 20261121):
            raise runner.AdmissionRunBlocked(
                "classical-v2 actual environment seed differs"
            )
        if episode.get("comparison_only") is not True:
            raise runner.AdmissionRunBlocked(
                "classical-v2 comparison attestation differs"
            )
        _check_provider_binding(episode, case, binding, start_sources, provider_path)
        payload = episode["controller_payload"]
        report = {
            "schema_name": "rj45-classical-v2-report-v1",
            "status": "classical_v2_complete",
            "diagnostic_only": True,
            "comparison_only": episode["comparison_only"],
            "stage": args.stage,
            "ordinal": args.ordinal,
            "case": case,
            "config": config,
            "config_sha256": config_digest,
            "actual_environment_seed": episode["environment_seed"],
            "trace": sidecar,
            "validated_terminal_snapshot": episode["validated_terminal_snapshot"],
            "episode_terminal": {
                "success": episode["success"],
                "hard_contact": episode["hard_contact"],
                "forbidden_contact": episode["forbidden_contact"],
                "evaluator_force_p95_n": episode["evaluator_force_p95_n"],
                "completion_time_s": episode["completion_time_s"],
            },
            "classical_v2": {
                "controller_payload": payload,
                "controller_sha256": episode["applied_controller_sha256"],
                "provider_source_sha256": episode["provider_source_sha256"],
                "provider_import_origin": episode["provider_import_origin"],
                "base_helper_source_sha256": payload["base_helper_source_sha256"],
                "steps": episode["comparison_provider_steps"],
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
