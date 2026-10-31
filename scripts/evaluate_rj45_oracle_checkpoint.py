#!/usr/bin/env python3
"""Recover a fresh physics evaluation from one invalid oracle92 probe run.

The original run remains invalid.  This helper only evaluates the exact final
checkpoint produced after its 32 audited updates; it never resumes training or
promotes the original report.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Mapping


SCHEMA_NAME = "rj45_oracle_checkpoint_evaluation_recovery_v1"
CONTRACT_SCHEMA = "rj45_oracle_checkpoint_recovery_contract_v1"
MODE = "oracle92"
EXPECTED_BASE_SEED = 20261128
EXPECTED_UPDATES = 32
WALL_TIME_LIMIT_S = 300.0
INFERENCE_ERROR_FRAGMENT = "Inplace update to inference tensor outside InferenceMode"


class CheckpointRecoveryError(RuntimeError):
    """The invalid-run recovery contract was not satisfied."""


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise CheckpointRecoveryError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CheckpointRecoveryError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise CheckpointRecoveryError(f"JSON root must be an object: {path}")
    return value


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _persist_report_before_app_close(
    *,
    report_path: Path,
    report: Mapping[str, Any],
    probe: Any,
    simulation_app: Any,
    exit_code: int,
) -> int:
    """Persist the terminal report before Kit can terminate the Python process."""

    try:
        _atomic_write_new(report_path, report)
    except BaseException:
        probe._close_simulation_app(simulation_app, 1)
        raise
    stream = sys.stderr if report.get("status") == "invalid" else sys.stdout
    print(
        json.dumps({"status": report.get("status"), "report": str(report_path)}, sort_keys=True),
        file=stream,
        flush=True,
    )
    return probe._close_simulation_app(simulation_app, exit_code)


def _close_environment_preserving_failure(
    *,
    env: Any,
    failure: BaseException | None,
    failure_traceback: Any,
    failure_text: str,
    failure_stage: str,
    exit_code: int,
) -> tuple[BaseException | None, Any, str, str, int]:
    if env is None:
        return failure, failure_traceback, failure_text, failure_stage, exit_code
    try:
        env.close()
    except BaseException as close_error:
        if failure is None:
            return (
                close_error,
                close_error.__traceback__,
                "".join(
                    traceback.format_exception(
                        type(close_error), close_error, close_error.__traceback__
                    )
                ),
                "environment_close",
                1,
            )
        traceback.print_exception(
            type(close_error), close_error, close_error.__traceback__
        )
        return failure, failure_traceback, failure_text, failure_stage, 1
    return failure, failure_traceback, failure_text, failure_stage, exit_code


def _sha256_hex(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CheckpointRecoveryError(f"{field} must be 64 lowercase hex characters")
    return value


def _normalize_deadline(value: Any) -> tuple[str, float]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise CheckpointRecoveryError("deadline_utc must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise CheckpointRecoveryError("deadline_utc is not valid ISO-8601 UTC") from error
    if parsed.tzinfo != timezone.utc:
        raise CheckpointRecoveryError("deadline_utc must be UTC")
    return parsed.isoformat().replace("+00:00", "Z"), parsed.timestamp()


def validate_recovery_contract(
    path: Path, expected_sha256: str, file_sha256: Any
) -> dict[str, Any]:
    expected_sha256 = _sha256_hex(expected_sha256, "contract SHA-256")
    if file_sha256(path) != expected_sha256:
        raise CheckpointRecoveryError("recovery contract file SHA-256 mismatch")
    contract = _load_json(path)
    required = {
        "schema_name",
        "mode",
        "base_seed",
        "updates",
        "deadline_utc",
        "invalid_report_sha256",
        "original_protocol_sha256",
        "original_runner_sha256",
        "checkpoint_sha256",
        "updates_sha256",
        "current_protocol_sha256",
        "recovery_helper_sha256",
        "replay_helper_sha256",
    }
    if set(contract) != required:
        raise CheckpointRecoveryError("recovery contract keys are invalid")
    if contract["schema_name"] != CONTRACT_SCHEMA:
        raise CheckpointRecoveryError("recovery contract schema changed")
    if contract["mode"] != MODE:
        raise CheckpointRecoveryError("recovery contract mode must be oracle92")
    if contract["base_seed"] != EXPECTED_BASE_SEED:
        raise CheckpointRecoveryError("recovery contract base_seed must be 20261128")
    if contract["updates"] != EXPECTED_UPDATES:
        raise CheckpointRecoveryError("recovery requires exactly 32 original updates")
    contract["deadline_utc"], _ = _normalize_deadline(contract["deadline_utc"])
    for field in required - {"schema_name", "mode", "base_seed", "updates", "deadline_utc"}:
        _sha256_hex(contract[field], field)
    if file_sha256(Path(__file__).resolve()) != contract["recovery_helper_sha256"]:
        raise CheckpointRecoveryError("recovery helper bytes do not match the contract")
    return contract


def validate_preimport_sources(
    *,
    current_root: Path,
    protocol_path: Path,
    contract: Mapping[str, Any],
    file_sha256: Any = _file_sha256,
) -> dict[str, Any]:
    runner_path = (current_root / "scripts/run_rj45_oracle_ppo_probe.py").resolve()
    replay_path = (current_root / "scripts/record_rj45_oracle_probe.py").resolve()
    if file_sha256(protocol_path) != contract["current_protocol_sha256"]:
        raise CheckpointRecoveryError("current protocol bytes changed before import")
    protocol = _load_json(protocol_path)
    runner_sha = protocol.get("source_sha256", {}).get("runner")
    _sha256_hex(runner_sha, "current protocol runner SHA-256")
    if file_sha256(runner_path) != runner_sha:
        raise CheckpointRecoveryError("current runner bytes differ from the bound protocol")
    if file_sha256(replay_path) != contract["replay_helper_sha256"]:
        raise CheckpointRecoveryError("replay helper bytes differ from the recovery contract")
    return {
        "runner": {"path": str(runner_path), "sha256": runner_sha},
        "replay_helper": {
            "path": str(replay_path),
            "sha256": contract["replay_helper_sha256"],
        },
    }


def _finite_json(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite_json(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    return False


def validate_updates_trace(
    path: Path, expected_sha256: str, file_sha256: Any
) -> list[dict[str, Any]]:
    if path.name != "updates.jsonl":
        raise CheckpointRecoveryError("updates trace must be named updates.jsonl")
    if file_sha256(path) != expected_sha256:
        raise CheckpointRecoveryError("updates trace SHA-256 mismatch")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise CheckpointRecoveryError("cannot read updates trace") from error
    if len(lines) != EXPECTED_UPDATES:
        raise CheckpointRecoveryError("updates trace must contain exactly 32 rows")
    for index, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise CheckpointRecoveryError("updates trace contains invalid JSON") from error
        if not isinstance(row, dict) or row.get("update") != index:
            raise CheckpointRecoveryError("updates trace sequence must be exactly 1..32")
        if not _finite_json(row):
            raise CheckpointRecoveryError("updates trace contains non-finite data")
        rows.append(row)
    return rows


def validate_invalid_origin(
    *,
    report_path: Path,
    checkpoint_path: Path,
    updates_path: Path,
    contract: Mapping[str, Any],
    file_sha256: Any,
) -> dict[str, Any]:
    if file_sha256(report_path) != contract["invalid_report_sha256"]:
        raise CheckpointRecoveryError("invalid origin report SHA-256 mismatch")
    report = _load_json(report_path)
    if report.get("status") != "invalid" or report.get("mode") != MODE:
        raise CheckpointRecoveryError("origin report is not the invalid oracle92 run")
    error = report.get("error")
    if (
        not isinstance(error, dict)
        or error.get("type") != "RuntimeError"
        or INFERENCE_ERROR_FRAGMENT not in str(error.get("message", ""))
    ):
        raise CheckpointRecoveryError("origin failure is not the admitted inference-mode error")
    artifacts = report.get("partial_artifacts")
    if not isinstance(artifacts, list):
        raise CheckpointRecoveryError("origin report has no partial artifact inventory")

    def matching(name: str) -> list[dict[str, Any]]:
        return [
            item
            for item in artifacts
            if isinstance(item, dict) and Path(str(item.get("path", ""))).name == name
        ]

    checkpoint_records = matching("final_checkpoint.pt")
    update_records = matching("updates.jsonl")
    if len(checkpoint_records) != 1 or len(update_records) != 1:
        raise CheckpointRecoveryError(
            "origin inventory must contain exactly one final checkpoint and updates trace"
        )
    if checkpoint_path.name != "final_checkpoint.pt":
        raise CheckpointRecoveryError("only the final_checkpoint.pt artifact is eligible")
    if checkpoint_records[0].get("sha256") != contract["checkpoint_sha256"]:
        raise CheckpointRecoveryError("origin final checkpoint binding mismatch")
    if update_records[0].get("sha256") != contract["updates_sha256"]:
        raise CheckpointRecoveryError("origin updates trace binding mismatch")
    if file_sha256(checkpoint_path) != contract["checkpoint_sha256"]:
        raise CheckpointRecoveryError("final checkpoint bytes changed")
    validate_updates_trace(updates_path, contract["updates_sha256"], file_sha256)
    return report


def validate_protocol_lineage(
    *,
    original_protocol_path: Path,
    current_protocol_record: Mapping[str, Any],
    original_runner_source: Path,
    contract: Mapping[str, Any],
    file_sha256: Any,
) -> dict[str, Any]:
    if file_sha256(original_protocol_path) != contract["original_protocol_sha256"]:
        raise CheckpointRecoveryError("original protocol SHA-256 mismatch")
    if file_sha256(original_runner_source) != contract["original_runner_sha256"]:
        raise CheckpointRecoveryError("original runner source SHA-256 mismatch")
    if current_protocol_record.get("file_sha256") != contract["current_protocol_sha256"]:
        raise CheckpointRecoveryError("current protocol SHA-256 mismatch")
    original = _load_json(original_protocol_path)
    current = current_protocol_record.get("payload")
    if not isinstance(current, dict):
        raise CheckpointRecoveryError("current protocol payload is missing")
    if original.get("source_sha256", {}).get("runner") != contract["original_runner_sha256"]:
        raise CheckpointRecoveryError("original protocol does not bind the supplied runner")
    fixed = original.get("fixed_condition", {})
    ppo = original.get("ppo", {})
    if (
        fixed.get("base_seed") != contract["base_seed"]
        or fixed.get("split") != "train"
        or fixed.get("grasp_id") != "rj45-train-02"
        or ppo.get("updates") != EXPECTED_UPDATES
    ):
        raise CheckpointRecoveryError("original protocol fixed train condition changed")
    if current.get("fixed_condition", {}).get("base_seed") != contract["base_seed"]:
        raise CheckpointRecoveryError("current protocol base seed differs from recovery contract")
    if current.get("budget_and_stop", {}).get("absolute_deadline_utc") != contract["deadline_utc"]:
        raise CheckpointRecoveryError("current protocol deadline differs from recovery contract")

    def comparable(payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized = copy.deepcopy(dict(payload))
        normalized["source_sha256"]["runner"] = "<allowed-runner-change>"
        normalized["budget_and_stop"]["absolute_deadline_utc"] = "<allowed-deadline-change>"
        return normalized

    if comparable(original) != comparable(current):
        raise CheckpointRecoveryError(
            "old/current protocols differ outside runner SHA and absolute deadline"
        )
    return {
        "original": {
            "path": str(original_protocol_path),
            "sha256": contract["original_protocol_sha256"],
            "runner_source": {
                "path": str(original_runner_source),
                "sha256": contract["original_runner_sha256"],
            },
        },
        "current": dict(current_protocol_record),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--recovery-contract", required=True, type=Path)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--invalid-report", required=True, type=Path)
    parser.add_argument("--original-protocol", required=True, type=Path)
    parser.add_argument("--original-runner-source", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--updates-trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--held-bank", type=Path, default=Path("assets/generated/rj45-held-grasps-v1.json"))
    parser.add_argument("--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/rj45-training-target-offset-v1.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--runtime-manifest", type=Path, default=Path("configs/rj45-oracle-runtime-104721a.json"))
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    for name in (
        "runtime_root", "current_root", "recovery_contract", "invalid_report",
        "original_protocol", "original_runner_source", "protocol", "checkpoint",
        "updates_trace", "output_dir", "report",
    ):
        setattr(args, name, getattr(args, name).resolve())
    if not args.runtime_manifest.is_absolute():
        args.runtime_manifest = (args.current_root / args.runtime_manifest).resolve()
    if args.output_dir.exists() or args.report.exists():
        parser.error("output directory/report already exists")

    contract = validate_recovery_contract(
        args.recovery_contract, args.contract_sha256, _file_sha256
    )
    preimport_sources = validate_preimport_sources(
        current_root=args.current_root,
        protocol_path=args.protocol,
        contract=contract,
    )
    probe = _load_module(
        args.current_root / "scripts/run_rj45_oracle_ppo_probe.py",
        "rj45_oracle_probe_for_checkpoint_recovery",
    )
    replay = _load_module(
        args.current_root / "scripts/record_rj45_oracle_probe.py",
        "rj45_oracle_replay_for_checkpoint_recovery",
    )
    simulation_app = None
    env = None
    previous_alarm_handler = None
    failure: BaseException | None = None
    failure_traceback = None
    failure_text = ""
    completed_report: dict[str, Any] | None = None
    failure_stage = "contract_validation"
    exit_code = 1
    try:
        current_protocol_record = probe.validate_protocol(
            args.protocol, contract["current_protocol_sha256"], args.current_root
        )
        lineage = validate_protocol_lineage(
            original_protocol_path=args.original_protocol,
            current_protocol_record=current_protocol_record,
            original_runner_source=args.original_runner_source,
            contract=contract,
            file_sha256=probe.file_sha256,
        )
        origin = validate_invalid_origin(
            report_path=args.invalid_report,
            checkpoint_path=args.checkpoint,
            updates_path=args.updates_trace,
            contract=contract,
            file_sha256=probe.file_sha256,
        )
        _, deadline_epoch = _normalize_deadline(contract["deadline_utc"])
        remaining_s = deadline_epoch - time.time()
        if remaining_s <= 0.0:
            raise CheckpointRecoveryError("recovery deadline already expired")
        args.execution_started_monotonic = time.monotonic()
        args.execution_deadline_epoch = deadline_epoch
        args.execution_mode_wall_s = WALL_TIME_LIMIT_S
        args.protocol_sha256 = contract["current_protocol_sha256"]
        args.mode = MODE
        args.runtime_manifest_record = probe.validate_runtime_manifest(
            args.runtime_root, args.runtime_manifest
        )
        if hasattr(signal, "SIGALRM"):
            def alarm_handler(_signum, _frame):
                raise TimeoutError("checkpoint recovery outer timer expired")

            previous_alarm_handler = signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, min(WALL_TIME_LIMIT_S, remaining_s))

        failure_stage = "application_launch"
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {"headless": args.headless, "device": args.device}
        ).app
        failure_stage = "fresh_physics_evaluation"
        sys.path.insert(0, str(args.runtime_root / "src"))
        protocol = current_protocol_record["payload"]
        g2 = probe._load_module(
            args.current_root / "scripts/run_rj45_g2_physics_probe.py",
            "rj45_g2_probe_for_checkpoint_recovery",
        )
        env, context, frozen_hashes = probe._build_env(args, MODE, protocol)
        runtime_import_origins = probe.verify_runtime_import_origins(args.runtime_root)
        policy = replay._policy_for_replay(
            env, MODE, args.checkpoint, protocol, probe
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        evaluation = probe._run_evaluation(
            env,
            MODE,
            policy,
            g2,
            context["task"],
            context["training"],
            args.output_dir / "evaluation.jsonl",
            lambda: probe.assert_execution_time(args),
            physics_seed=context["physics_seed"],
        )
        if evaluation.get("classification") == "invalid":
            raise CheckpointRecoveryError("fresh independent evaluation is invalid")
        recovery_inputs = {
            args.recovery_contract: contract["recovery_helper_sha256"],
            args.invalid_report: contract["invalid_report_sha256"],
            args.original_protocol: contract["original_protocol_sha256"],
            args.original_runner_source: contract["original_runner_sha256"],
            args.protocol: contract["current_protocol_sha256"],
            args.checkpoint: contract["checkpoint_sha256"],
            args.updates_trace: contract["updates_sha256"],
            Path(__file__).resolve(): contract["recovery_helper_sha256"],
            Path(preimport_sources["replay_helper"]["path"]): contract[
                "replay_helper_sha256"
            ],
        }
        # The contract file has its own digest; the helper digest is kept separately.
        recovery_inputs[args.recovery_contract] = args.contract_sha256
        for path, digest in {**frozen_hashes, **recovery_inputs}.items():
            if probe.file_sha256(path) != digest:
                raise CheckpointRecoveryError(f"frozen recovery input changed: {path}")
        completed_report = {
            "schema_name": SCHEMA_NAME,
            "status": "completed",
            "mode": MODE,
            "scope": (
                "fresh physics checkpoint evaluation completed; application cleanup "
                "is certified only by the external wrapper's raw exit-code check"
            ),
            "origin_run_invalid": True,
            "origin_report": {
                "path": str(args.invalid_report),
                "sha256": contract["invalid_report_sha256"],
                "failure": origin["error"],
                "promoted": False,
            },
            "learning_executed": False,
            "verified_original_updates": EXPECTED_UPDATES,
            "checkpoint_binding": {
                "path": str(args.checkpoint),
                "sha256": contract["checkpoint_sha256"],
                "selection": "preregistered final checkpoint only",
                "new_training_updates": 0,
            },
            "evaluation": evaluation,
            "protocol_lineage": lineage,
            "new_protocol_record": current_protocol_record,
            "fixed_condition": protocol["fixed_condition"],
            "source_binding": preimport_sources,
            "runtime_versions": context["versions"],
            "runtime_manifest": args.runtime_manifest_record,
            "runtime_import_origins": runtime_import_origins,
            "frozen_input_sha256": {
                str(path): digest for path, digest in {**frozen_hashes, **recovery_inputs}.items()
            },
            "claims": {
                "original_run_became_valid": False,
                "training_resumed": False,
                "checkpoint_reselected": False,
                "reinforcement_learning_effect_proven": False,
                "heldout_generalization": False,
                "fresh_physics_evaluation_completed": True,
                "application_cleanup_verified_by_report": False,
            },
            "execution_guard": {
                "mode_wall_time_limit_s": WALL_TIME_LIMIT_S,
                "absolute_deadline_utc": contract["deadline_utc"],
                "elapsed_wall_s_before_close": time.monotonic() - args.execution_started_monotonic,
            },
        }
        exit_code = 0
    except BaseException as error:
        failure = error
        failure_traceback = error.__traceback__
        failure_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if previous_alarm_handler is not None:
                signal.signal(signal.SIGALRM, previous_alarm_handler)
        (
            failure,
            failure_traceback,
            failure_text,
            failure_stage,
            exit_code,
        ) = _close_environment_preserving_failure(
            env=env,
            failure=failure,
            failure_traceback=failure_traceback,
            failure_text=failure_text,
            failure_stage=failure_stage,
            exit_code=exit_code,
        )
    if failure is not None:
        invalid = {
            "schema_name": SCHEMA_NAME,
            "status": "invalid",
            "mode": MODE,
            "scope": (
                "checkpoint-evaluation-recovery invalid; application cleanup is "
                "certified only by the external wrapper's raw exit-code check"
            ),
            "origin_run_invalid": True,
            "learning_executed": False,
            "failure_stage": failure_stage,
            "error": {"type": type(failure).__name__, "message": str(failure)},
            "traceback": failure_text,
            "claims": {
                "original_run_became_valid": False,
                "training_resumed": False,
                "checkpoint_reselected": False,
                "reinforcement_learning_effect_proven": False,
            },
        }
        report = invalid
        exit_code = 2 if exit_code == 0 else exit_code
    else:
        if completed_report is None:
            raise CheckpointRecoveryError("completed recovery report was not constructed")
        report = completed_report
    return _persist_report_before_app_close(
        report_path=args.report,
        report=report,
        probe=probe,
        simulation_app=simulation_app,
        exit_code=exit_code,
    )


if __name__ == "__main__":
    raise SystemExit(main())
