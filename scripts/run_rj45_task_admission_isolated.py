#!/usr/bin/env python3
"""Coordinate the fixed 90-episode admission campaign across fresh processes.

This module is CPU-only.  Each worker delegates native simulation and episode
construction to ``run_rj45_task_admission.py`` through a sealed chunk spec.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence


CAMPAIGN_SCHEMA = "rj45-task-admission-isolated-campaign-v1"
WORKER_SPEC_SCHEMA = "rj45-task-admission-isolated-worker-spec-v1"
CHUNK_REPORT_SCHEMA = "rj45-task-admission-isolated-chunk-report-v1"
FORMAL_CHUNKS = (
    (1, 6),
    (7, 12),
    (13, 18),
    (19, 30),
    (31, 42),
    (43, 54),
    (55, 66),
    (67, 78),
    (79, 90),
)
FORBIDDEN_COORDINATOR_IMPORT_PREFIXES = ("isaac", "omni", "torch")
WORKER_IDENTITY_SCHEMA = "rj45-task-admission-isolated-worker-identity-v1"
WORKER_ATTESTATION_SCHEMA = "rj45-task-admission-isolated-worker-attestation-v1"
CURRENT_WORKER_MODULES = (
    "rj45_comparison_env",
    "rj45_task_admission_env",
    "rj45_comparison_providers",
    "rj45_task_admission",
)


class IsolatedCampaignBlocked(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise IsolatedCampaignBlocked(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise IsolatedCampaignBlocked(f"JSON root must be an object: {path}")
    return value


def _write_new(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise IsolatedCampaignBlocked(f"refusing to overwrite campaign artifact: {path}")
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
            stream.write(
                json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise IsolatedCampaignBlocked(f"cannot load CPU helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _assert_cpu_only_imports(before: set[str]) -> None:
    introduced = set(sys.modules) - before
    forbidden = sorted(
        name
        for name in introduced
        if name.split(".", 1)[0].startswith(FORBIDDEN_COORDINATOR_IMPORT_PREFIXES)
    )
    if forbidden:
        raise IsolatedCampaignBlocked(
            f"CPU coordinator imported forbidden native modules: {forbidden}"
        )


def _plan_record(ordinal: int, row: Sequence[Any]) -> dict[str, Any]:
    return {
        "ordinal": ordinal,
        "stage": row[0],
        "grasp_id": row[1],
        "case_id": row[2],
        "method_id": row[3],
        "slot": row[4],
        "candidate_id": row[5],
    }


def _fixed_partitions(plan: Sequence[Any]) -> tuple[tuple[int, ...], ...]:
    if len(plan) != 90:
        raise IsolatedCampaignBlocked("formal campaign plan must contain exactly 90 rows")
    partitions = tuple(tuple(range(start, end + 1)) for start, end in FORMAL_CHUNKS)
    if max(len(chunk) for chunk in partitions) > 12:
        raise IsolatedCampaignBlocked("formal worker chunk exceeds 12 episodes")
    if tuple(ordinal for chunk in partitions for ordinal in chunk) != tuple(range(1, 91)):
        raise IsolatedCampaignBlocked("formal worker partitions are not exact 1 through 90")
    return partitions


def _expected_runtime_versions(runtime_config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "isaacsim": runtime_config["isaac_sim_distribution"],
        "isaaclab": runtime_config["isaac_lab_distribution"],
        "isaaclab-rl": runtime_config["isaac_lab_rl"],
        "rsl-rl-lib": runtime_config["rsl_rl"],
        "torch": runtime_config["torch"] + "+" + runtime_config["torch_cuda_build"],
        "numpy": runtime_config["numpy"],
        "h5py": runtime_config["h5py"],
    }


def _held_bank_dependencies(runtime_root: Path, manifest_path: Path) -> tuple[Path, ...]:
    """Resolve the worker's frozen train-bank inputs without importing runtime code."""

    root = runtime_root.resolve()

    def resolve(value: Any, label: str) -> Path:
        if not isinstance(value, str) or not value:
            raise IsolatedCampaignBlocked(f"{label} path is invalid")
        path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise IsolatedCampaignBlocked(f"{label} path escaped or is missing")
        return path

    def resolve_hashed(value: Any, expected: Any, label: str) -> Path:
        path = resolve(value, label)
        if not isinstance(expected, str) or _sha256(path) != expected:
            raise IsolatedCampaignBlocked(f"{label} hash differs")
        return path

    manifest = _load_json(manifest_path)
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise IsolatedCampaignBlocked("held-bank manifest entries are invalid")
    dependencies = [manifest_path.resolve()]
    selected_ids: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict) or entry.get("split") != "train":
            continue
        grasp_id = entry.get("grasp_id")
        artifact_ref = entry.get("artifact")
        if not isinstance(grasp_id, str) or not isinstance(artifact_ref, dict):
            raise IsolatedCampaignBlocked("held-bank train entry is invalid")
        artifact_path = resolve_hashed(
            artifact_ref.get("path"), artifact_ref.get("sha256"),
            f"held-bank artifact {grasp_id}",
        )
        artifact = _load_json(artifact_path)
        provenance = artifact.get("provenance")
        if not isinstance(provenance, dict):
            raise IsolatedCampaignBlocked(f"held-bank artifact provenance is invalid: {grasp_id}")
        paths = [
            artifact_path,
            resolve_hashed(
                provenance.get("nominal_report_path"),
                provenance.get("nominal_report_sha256"),
                f"nominal report {grasp_id}",
            ),
            resolve_hashed(
                provenance.get("nominal_trace_path"),
                provenance.get("nominal_trace_sha256"),
                f"nominal trace {grasp_id}",
            ),
        ]
        generation = provenance.get("grasp_generation")
        if generation is not None:
            if not isinstance(generation, dict):
                raise IsolatedCampaignBlocked(f"grasp generation is invalid: {grasp_id}")
            paths.append(
                resolve_hashed(
                    generation.get("catalog_path"),
                    generation.get("catalog_sha256"),
                    f"grasp catalog {grasp_id}",
                )
            )
        for path in paths:
            if path not in dependencies:
                dependencies.append(path)
        selected_ids.append(grasp_id)
    if tuple(selected_ids) != tuple(f"rj45-train-{index:02d}" for index in range(1, 9)):
        raise IsolatedCampaignBlocked("held-bank train IDs differ from the fixed contract")
    return tuple(dependencies)


def _trusted_worker_attestation(
    args: argparse.Namespace, prepared: Mapping[str, Any], runner: Any
) -> dict[str, Any]:
    runtime_root = args.runtime_root.resolve()
    evidence_root = args.evidence_root.resolve()
    current_root = args.current_root.resolve()
    runtime_config = _load_json(runtime_root / "configs/runtime.json")
    snapshot_current = prepared["physics_input_snapshot"]["normalized_sha256"]["current"]
    bank_path = evidence_root / "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json"
    if _sha256(bank_path) != runner.EXPECTED_BANK_SHA256:
        raise IsolatedCampaignBlocked("held-bank manifest differs from the fixed contract")
    dependencies = _held_bank_dependencies(evidence_root, bank_path)
    frozen = {str(path): _sha256(path) for path in dependencies}
    manifest = _load_json(bank_path)
    train_entries = [entry for entry in manifest["entries"] if entry.get("split") == "train"]
    current_origins = {
        name: str((current_root / f"src/rby1_assembly/{name}.py").resolve())
        for name in CURRENT_WORKER_MODULES
    }
    for name, path in current_origins.items():
        relative = f"src/rby1_assembly/{name}.py"
        if snapshot_current.get(relative) != _sha256(Path(path)):
            raise IsolatedCampaignBlocked(f"current module differs from campaign snapshot: {name}")
    return {
        "runtime_versions": _expected_runtime_versions(runtime_config),
        "current_module_origins": current_origins,
        "frozen_input_sha256": frozen,
        "held_bank_identity": {
            "manifest_sha256": runner.EXPECTED_BANK_SHA256,
            "selected_grasp_ids": list(runner.EXPECTED_IDS),
            "selected_artifact_sha256": [entry["artifact"]["sha256"] for entry in train_entries],
            "source_contract_sha256": runner.EXPECTED_BANK_SOURCE_SHA256,
        },
    }


def _verify_trusted_bank_inputs(
    campaign: Mapping[str, Any], args: argparse.Namespace, runner: Any
) -> None:
    trusted = campaign.get("trusted_worker_attestation")
    if not isinstance(trusted, dict):
        raise IsolatedCampaignBlocked("campaign trusted worker attestation is missing")
    held = trusted.get("held_bank_identity")
    if not isinstance(held, dict) or held.get("manifest_sha256") != runner.EXPECTED_BANK_SHA256:
        raise IsolatedCampaignBlocked("campaign held-bank identity differs")
    frozen = trusted.get("frozen_input_sha256")
    if not isinstance(frozen, dict) or not frozen:
        raise IsolatedCampaignBlocked("campaign frozen bank map is missing")
    evidence_root = args.evidence_root.resolve()
    for raw_path, expected_sha in frozen.items():
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise IsolatedCampaignBlocked("campaign frozen bank entry is invalid")
        path = Path(raw_path).resolve()
        if (
            raw_path != str(path)
            or not path.is_relative_to(evidence_root)
            or not path.is_file()
            or _sha256(path) != expected_sha
        ):
            raise IsolatedCampaignBlocked("campaign frozen bank input changed")


def _prepare_campaign(
    args: argparse.Namespace, runner: Any
) -> tuple[dict[str, Any], dict[str, Any]]:
    runner_args = runner._parser().parse_args(
        [
            "--runtime-root",
            str(args.runtime_root),
            "--evidence-root",
            str(args.evidence_root),
            "--current-root",
            str(args.current_root),
            "--config",
            str(args.config),
            "--output-dir",
            str(args.output_dir / "coordinator-preflight-unused"),
            "--playback-actions",
            str(args.playback_actions),
        ]
    )
    prepared = runner._prepare(runner_args)
    plan = prepared["plan"]
    _fixed_partitions(plan)
    coordinator_path = Path(__file__).resolve()
    runner_path = args.current_root.resolve() / "scripts/run_rj45_task_admission.py"
    payload = {
        "schema_name": CAMPAIGN_SCHEMA,
        "purpose": args.purpose,
        "environment_seed": runner.TASK_ADMISSION_ENVIRONMENT_SEED,
        "execution_plan": [
            _plan_record(ordinal, row)
            for ordinal, row in enumerate(plan, start=1)
        ],
        "formal_partitions": [list(chunk) for chunk in _fixed_partitions(plan)],
        "config": {
            "path": str(prepared["config_path"]),
            "sha256": runner._sha256(prepared["config_path"]),
        },
        "playback_actions": {
            "path": str(prepared["playback_path"]),
            "sha256": runner._sha256(prepared["playback_path"]),
            "canonical_sha256": prepared["playback_canonical_sha256"],
        },
        "source_sha256": {
            "coordinator": _sha256(coordinator_path),
            "runner": _sha256(runner_path),
        },
        "physics_input_start": prepared["physics_input_snapshot"],
        "physics_input_start_canonical_sha256": prepared["physics_input_snapshot"][
            "normalized_sha256_canonical_sha256"
        ],
        "trusted_worker_attestation": _trusted_worker_attestation(args, prepared, runner),
        "claims": {
            "native_crash_cause_fixed": False,
            "task_admission_completed": False,
        },
    }
    return prepared, payload


def _verify_campaign_inputs(
    campaign_path: Path,
    campaign_sha256: str,
    campaign: Mapping[str, Any],
    args: argparse.Namespace,
    runner: Any,
) -> None:
    if _sha256(campaign_path) != campaign_sha256:
        raise IsolatedCampaignBlocked("campaign identity changed during execution")
    expected_sources = campaign.get("source_sha256")
    if expected_sources != {
        "coordinator": _sha256(Path(__file__).resolve()),
        "runner": _sha256(
            args.current_root.resolve() / "scripts/run_rj45_task_admission.py"
        ),
    }:
        raise IsolatedCampaignBlocked("coordinator or runner source changed")
    _verify_trusted_bank_inputs(campaign, args, runner)
    snapshot = campaign.get("physics_input_start")
    if not isinstance(snapshot, dict):
        raise IsolatedCampaignBlocked("campaign physics input snapshot is missing")
    validator_path = runner._validate_end_snapshot_before_validator(
        snapshot, args.current_root.resolve()
    )
    before = set(sys.modules)
    validator = _load_module(
        validator_path, "rby1_isolated_campaign_input_validator"
    )
    _assert_cpu_only_imports(before)
    result = validator.verify_physics_input_end(
        snapshot,
        args.runtime_root.resolve(),
        args.evidence_root.resolve(),
        args.current_root.resolve(),
    )
    if not result.get("matched"):
        raise IsolatedCampaignBlocked("campaign physics inputs changed before worker")


def _worker_spec(
    campaign_path: Path,
    campaign_sha256: str,
    campaign: Mapping[str, Any],
    ordinals: Sequence[int],
    *,
    purpose: str,
    selected_classical_candidate: str | None,
    tuning_seal: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    plan = campaign["execution_plan"]
    return {
        "schema_name": WORKER_SPEC_SCHEMA,
        "purpose": purpose,
        "campaign_identity_path": str(campaign_path),
        "campaign_sha256": campaign_sha256,
        "rows": [plan[ordinal - 1] for ordinal in ordinals],
        "selected_classical_candidate": selected_classical_candidate,
        "tuning_seal": tuning_seal,
    }


def _launch_worker(
    args: argparse.Namespace,
    spec_path: Path,
    spec_sha256: str,
    worker_dir: Path,
) -> dict[str, Any]:
    runner_path = args.current_root.resolve() / "scripts/run_rj45_task_admission.py"
    command = [
        sys.executable,
        str(runner_path),
        "--runtime-root",
        str(args.runtime_root),
        "--evidence-root",
        str(args.evidence_root),
        "--current-root",
        str(args.current_root),
        "--config",
        str(args.config),
        "--playback-actions",
        str(args.playback_actions),
        "--output-dir",
        str(worker_dir),
        "--worker-spec",
        str(spec_path),
        "--expected-worker-spec-sha256",
        spec_sha256,
    ]
    worker_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = worker_dir / "worker.stdout.log"
    stderr_path = worker_dir / "worker.stderr.log"
    timed_out = False
    return_code: int | None = None
    worker_group_cleanup: dict[str, Any] | None = None
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        worker_pgid = process.pid
        try:
            return_code = process.wait(timeout=args.worker_timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        if timed_out or (return_code is not None and return_code < 0):
            worker_group_cleanup = _cleanup_owned_process_group(
                process, worker_pgid, args.process_group_cleanup_grace_seconds
            )
            return_code = process.returncode
        else:
            worker_group_cleanup = _cleanup_residual_process_group(
                process, worker_pgid, args.process_group_cleanup_grace_seconds
            )
    verifier_result = _run_post_exit_verifier(
        [
            sys.executable, str(runner_path), "--runtime-root", str(args.runtime_root),
            "--evidence-root", str(args.evidence_root),
            "--current-root", str(args.current_root), "--config", str(args.config),
            "--output-dir", str(worker_dir), "--verify-after-child-exit",
        ],
        timeout_seconds=args.post_exit_verifier_timeout_seconds,
        cleanup_grace_seconds=args.process_group_cleanup_grace_seconds,
    )
    exit_payload = {
        "schema_name": "rj45-task-admission-isolated-worker-exit-v1",
        "return_code": return_code,
        "timed_out": timed_out,
        "worker_process_group_cleanup": worker_group_cleanup,
        "post_exit_verifier_return_code": verifier_result["return_code"],
        "post_exit_verifier_timed_out": verifier_result["timed_out"],
        "post_exit_verifier_stdout": verifier_result["stdout"],
        "post_exit_verifier_stderr": verifier_result["stderr"],
        "post_exit_verifier_process_group_cleanup": verifier_result["process_group_cleanup"],
    }
    _write_new(worker_dir / "worker-exit.json", exit_payload)
    return exit_payload


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _cleanup_owned_process_group(
    process: subprocess.Popen[Any], pgid: int, grace_seconds: float
) -> dict[str, Any]:
    if pgid != process.pid or pgid <= 1 or pgid == os.getpgrp():
        raise IsolatedCampaignBlocked("refusing to clean an unowned process group")
    sent: list[str] = []
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGTERM)
        sent.append("SIGTERM")
        deadline = time.monotonic() + grace_seconds
        while _process_group_exists(pgid) and time.monotonic() < deadline:
            time.sleep(0.01)
    if _process_group_exists(pgid):
        os.killpg(pgid, signal.SIGKILL)
        sent.append("SIGKILL")
    try:
        process.wait(timeout=max(grace_seconds, 0.1))
    except subprocess.TimeoutExpired as error:
        raise IsolatedCampaignBlocked("owned process group leader did not exit") from error
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.01)
    return {"pgid": pgid, "signals": sent, "group_absent_after_cleanup": not _process_group_exists(pgid)}


def _cleanup_residual_process_group(
    process: subprocess.Popen[Any], pgid: int, grace_seconds: float
) -> dict[str, Any]:
    if pgid != process.pid or pgid <= 1 or pgid == os.getpgrp():
        raise IsolatedCampaignBlocked("refusing to inspect an unowned process group")
    if _process_group_exists(pgid):
        proof = _cleanup_owned_process_group(process, pgid, grace_seconds)
    else:
        proof = {"pgid": pgid, "signals": [], "group_absent_after_cleanup": True}
    if proof.get("group_absent_after_cleanup") is not True:
        raise IsolatedCampaignBlocked("owned process group cleanup was incomplete")
    return proof


def _run_post_exit_verifier(
    command: Sequence[str], *, timeout_seconds: float, cleanup_grace_seconds: float
) -> dict[str, Any]:
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    cleanup = None
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup = _cleanup_owned_process_group(process, process.pid, cleanup_grace_seconds)
        stdout, stderr = process.communicate()
    if cleanup is None:
        cleanup = _cleanup_residual_process_group(
            process, process.pid, cleanup_grace_seconds
        )
    if cleanup.get("group_absent_after_cleanup") is not True:
        raise IsolatedCampaignBlocked("post-exit verifier process group cleanup was incomplete")
    return {
        "return_code": process.returncode,
        "timed_out": timed_out,
        "stdout": stdout,
        "stderr": stderr,
        "process_group_cleanup": cleanup,
    }


def _validate_worker_output(
    worker_dir: Path,
    spec: Mapping[str, Any],
    spec_sha256: str,
    *,
    args: argparse.Namespace,
    campaign: Mapping[str, Any],
) -> list[dict[str, Any]]:
    exit_path = worker_dir / "worker-exit.json"
    if not exit_path.is_file():
        raise IsolatedCampaignBlocked("worker exit record is missing")
    exit_record = _load_json(exit_path)
    worker_cleanup = exit_record.get("worker_process_group_cleanup")
    verifier_cleanup = exit_record.get("post_exit_verifier_process_group_cleanup")
    if (
        exit_record.get("return_code") != 0
        or exit_record.get("timed_out") is not False
        or exit_record.get("post_exit_verifier_return_code") != 0
        or exit_record.get("post_exit_verifier_timed_out") is not False
        or not isinstance(worker_cleanup, dict)
        or worker_cleanup.get("group_absent_after_cleanup") is not True
        or not isinstance(verifier_cleanup, dict)
        or verifier_cleanup.get("group_absent_after_cleanup") is not True
    ):
        raise IsolatedCampaignBlocked("worker process or post-exit verifier failed")
    end_path = worker_dir / "physics-input-end-verification.json"
    end = _load_json(end_path)
    if end.get("matched") is not True or end.get("scientific_completion_allowed") is not True:
        raise IsolatedCampaignBlocked("worker end verification is absent or unmatched")
    report = _load_json(worker_dir / "chunk-report.json")
    if report.get("schema_name") != CHUNK_REPORT_SCHEMA:
        raise IsolatedCampaignBlocked("worker chunk report schema mismatch")
    if (
        report.get("status") != "worker_completed"
        or report.get("campaign_sha256") != spec["campaign_sha256"]
        or report.get("worker_spec_sha256") != spec_sha256
        or report.get("purpose") != spec["purpose"]
    ):
        raise IsolatedCampaignBlocked("worker chunk report identity mismatch")
    if report.get("claims") != {
        "task_admission_completed": False,
        "physics_success_proven": False,
    }:
        raise IsolatedCampaignBlocked("worker chunk report overclaims completion")
    identity_path = worker_dir / "run-identity.json"
    identity_ref = report.get("run_identity")
    if (
        not isinstance(identity_ref, dict)
        or Path(str(identity_ref.get("path", ""))).resolve() != identity_path.resolve()
        or not identity_path.is_file()
        or identity_ref.get("sha256") != _sha256(identity_path)
    ):
        raise IsolatedCampaignBlocked("worker run identity binding differs")
    identity_sha = identity_ref["sha256"]
    identity = _load_json(identity_path)
    expected_identity = {
        "schema_name": WORKER_IDENTITY_SCHEMA,
        "environment_seed": campaign["environment_seed"],
        "purpose": spec["purpose"],
        "campaign_identity": {
            "path": spec["campaign_identity_path"],
            "sha256": spec["campaign_sha256"],
        },
        "worker_spec": {
            "path": str((worker_dir / "worker-spec.json").resolve()),
            "sha256": spec_sha256,
        },
        "execution_plan": spec["rows"],
    }
    if identity != expected_identity:
        raise IsolatedCampaignBlocked("worker run identity content differs")
    attestation_ref = report.get("worker_attestation")
    attestation_path = worker_dir / "worker-attestation.json"
    if (
        not isinstance(attestation_ref, dict)
        or Path(str(attestation_ref.get("path", ""))).resolve()
        != attestation_path.resolve()
        or attestation_ref.get("sha256") != _sha256(attestation_path)
    ):
        raise IsolatedCampaignBlocked("worker attestation binding differs")
    attestation = _load_json(attestation_path)
    trusted_attestation = campaign.get("trusted_worker_attestation")
    if not isinstance(trusted_attestation, dict):
        raise IsolatedCampaignBlocked("campaign trusted worker attestation is missing")
    if (
        attestation.get("schema_name")
        != WORKER_ATTESTATION_SCHEMA
        or attestation.get("campaign_sha256") != spec["campaign_sha256"]
        or attestation.get("worker_spec_sha256") != spec_sha256
        or attestation.get("runtime_versions") != trusted_attestation.get("runtime_versions")
        or attestation.get("current_module_origins")
        != trusted_attestation.get("current_module_origins")
        or attestation.get("frozen_input_sha256")
        != trusted_attestation.get("frozen_input_sha256")
        or attestation.get("held_bank_identity")
        != trusted_attestation.get("held_bank_identity")
    ):
        raise IsolatedCampaignBlocked("worker source-origin attestation differs")
    runtime_origins = attestation.get("runtime_import_origins")
    if not isinstance(runtime_origins, dict) or not runtime_origins:
        raise IsolatedCampaignBlocked("worker runtime origin attestation is empty")
    required_runtime_modules = {
        "rby1_assembly",
        "rby1_assembly.held_grasp",
        "rby1_assembly.rj45_task",
    }
    if not required_runtime_modules.issubset(runtime_origins):
        raise IsolatedCampaignBlocked("worker runtime origin attestation omits required modules")
    runtime_root = args.runtime_root.resolve()
    runtime_src = (runtime_root / "src").resolve()
    snapshot_runtime = campaign["physics_input_start"]["normalized_sha256"]["runtime"]
    seen_runtime_paths: set[Path] = set()
    for module_name, raw_path in runtime_origins.items():
        if not isinstance(module_name, str) or not isinstance(raw_path, str):
            raise IsolatedCampaignBlocked("worker runtime origin attestation is invalid")
        path = Path(raw_path).resolve()
        module_parts = module_name.split(".")
        if not module_parts or module_parts[0] != "rby1_assembly":
            raise IsolatedCampaignBlocked("worker runtime origin module name is invalid")
        module_relative = Path(*module_parts)
        allowed_origins = {
            (runtime_src / module_relative).with_suffix(".py").resolve(),
            (runtime_src / module_relative / "__init__.py").resolve(),
        }
        if (
            not path.is_relative_to(runtime_src)
            or not path.is_file()
            or path in seen_runtime_paths
            or path not in allowed_origins
        ):
            raise IsolatedCampaignBlocked("worker runtime origin escaped or was duplicated")
        relative = str(path.relative_to(runtime_root))
        if snapshot_runtime.get(relative) != _sha256(path):
            raise IsolatedCampaignBlocked("worker runtime origin differs from campaign snapshot")
        seen_runtime_paths.add(path)
    expected_records = spec["rows"]
    row_refs = report.get("episode_rows")
    if not isinstance(row_refs, list) or len(row_refs) != len(expected_records):
        raise IsolatedCampaignBlocked("worker row reference count differs")
    rows: list[dict[str, Any]] = []
    for expected, reference in zip(expected_records, row_refs, strict=True):
        ordinal = expected["ordinal"]
        path = worker_dir / "episode-rows" / f"{ordinal:03d}.json"
        if (
            not isinstance(reference, dict)
            or reference.get("ordinal") != ordinal
            or Path(str(reference.get("path", ""))).resolve() != path.resolve()
            or reference.get("sha256") != _sha256(path)
        ):
            raise IsolatedCampaignBlocked(f"worker row {ordinal} hash binding differs")
        envelope = _load_json(path)
        if (
            envelope.get("ordinal") != ordinal
            or envelope.get("campaign_sha256") != spec["campaign_sha256"]
            or envelope.get("run_identity_sha256") != identity_sha
            or envelope.get("schema_name") != "rj45-task-admission-episode-row-v1"
        ):
            raise IsolatedCampaignBlocked(f"worker row {ordinal} campaign binding differs")
        episode = envelope.get("episode")
        source = envelope.get("source_identity")
        if not isinstance(episode, dict) or not isinstance(source, dict):
            raise IsolatedCampaignBlocked(f"worker row {ordinal} payload is missing")
        expected_source = {
            key: expected[key]
            for key in ("stage", "grasp_id", "case_id", "method_id", "slot", "candidate_id")
        }
        if expected["method_id"] == "classical_force_search" and expected_source["candidate_id"] is None:
            expected_source["candidate_id"] = spec["selected_classical_candidate"]
        if source != expected_source or any(episode.get(key) != value for key, value in expected_source.items()):
            raise IsolatedCampaignBlocked(f"worker row {ordinal} source identity differs")
        if episode.get("execution_error") is not None or episode.get("terminal_complete") is not True:
            raise IsolatedCampaignBlocked(f"worker row {ordinal} is incomplete")
        rows.append(episode)
    return rows


def _attestation_contract(worker_dir: Path) -> dict[str, Any]:
    attestation = _load_json(worker_dir / "worker-attestation.json")
    return {
        key: attestation[key]
        for key in (
            "runtime_versions",
            "runtime_import_origins",
            "current_module_origins",
            "frozen_input_sha256",
            "held_bank_identity",
        )
    }


def _seal_tuning(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], runner: Any, admission: Any
) -> tuple[Any, dict[str, Any]]:
    if len(rows) != 18:
        raise IsolatedCampaignBlocked("classical selection requires exact tuning18")
    runner._validate_tuning_rows(rows, config)
    values = [
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
        for row in rows
    ]
    protocol = admission.AdmissionProtocol().validate_tuning(values)
    protocol = protocol.seal_classical(
        admission.canonical_tuning_report_bytes(values), values
    )
    seal = {
        "selection": asdict(protocol.selection),
        "classical_seal": {
            "selected_candidate_id": protocol.classical_seal.selected_candidate_id,
            "canonical_controller_config_sha256": protocol.classical_seal.canonical_controller_config_sha256,
            "tuning_report_sha256": protocol.classical_seal.tuning_report_sha256,
            "selection": asdict(protocol.classical_seal.selection),
        },
        "episode_count": 18,
    }
    return protocol, seal


def _adjudicate_exact90(
    rows: Sequence[Mapping[str, Any]],
    prepared: Mapping[str, Any],
    runner: Any,
    admission: Any,
    protocol: Any,
) -> tuple[Any, Any, Any]:
    if len(rows) != 90:
        raise IsolatedCampaignBlocked("adjudication requires exact 90 rows")
    admission_rows = rows[18:]
    episodes = [
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
            observation_samples=tuple(tuple(sample) for sample in row["observation_samples"]),
        )
        for row in admission_rows
    ]
    baseline = admission.make_baseline_artifact_seal(
        source_trace_gzip_sha256=prepared["config"]["methods"][2]["source_trace_gzip_sha256"],
        mean_constant_action=prepared["config"]["methods"][1]["normalized_right6"],
        playback_selected_rows=len(prepared["playback_actions"]),
    )
    evidence = runner._evidence(
        admission_rows,
        True,
        prepared["config"],
        baseline_artifacts_verified=admission.validate_baseline_artifact_seal(baseline),
    )
    protocol = protocol.validate_admission(episodes, evidence, baseline).adjudicate(
        episodes, evidence
    )
    return protocol, baseline, evidence


def run_campaign(
    args: argparse.Namespace,
    *,
    worker_launcher: Callable[[argparse.Namespace, Path, str, Path], Mapping[str, Any]] = _launch_worker,
) -> int:
    before = set(sys.modules)
    runner = _load_module(
        args.current_root.resolve() / "scripts/run_rj45_task_admission.py",
        "rby1_isolated_admission_runner",
    )
    _assert_cpu_only_imports(before)
    prepared, campaign = _prepare_campaign(args, runner)
    campaign_path = args.output_dir.resolve() / "campaign-identity.json"
    _write_new(campaign_path, campaign)
    campaign_sha = _sha256(campaign_path)
    admission = _load_module(
        args.current_root.resolve() / "src/rby1_assembly/rj45_task_admission.py",
        "rby1_isolated_admission_protocol",
    )
    _assert_cpu_only_imports(before)
    rows: list[dict[str, Any]] = []
    protocol = None
    selected: str | None = None
    tuning_binding: dict[str, str] | None = None
    attestation_contract_sha: str | None = None
    partitions = (
        ((33,),)
        if args.purpose == "diagnostic_nonadjudicative"
        else _fixed_partitions(prepared["plan"])
    )
    for chunk_index, ordinals in enumerate(partitions, start=1):
        _verify_campaign_inputs(campaign_path, campaign_sha, campaign, args, runner)
        if ordinals[0] >= 19 and args.purpose == "formal_campaign" and selected is None:
            raise IsolatedCampaignBlocked("admission cannot start before tuning seal")
        spec = _worker_spec(
            campaign_path,
            campaign_sha,
            campaign,
            ordinals,
            purpose=args.purpose,
            selected_classical_candidate=selected,
            tuning_seal=tuning_binding,
        )
        chunk_dir = args.output_dir.resolve() / "workers" / f"chunk-{chunk_index:02d}"
        spec_path = chunk_dir / "worker-spec.json"
        _write_new(spec_path, spec)
        spec_sha = _sha256(spec_path)
        worker_launcher(args, spec_path, spec_sha, chunk_dir)
        _verify_campaign_inputs(campaign_path, campaign_sha, campaign, args, runner)
        chunk_rows = _validate_worker_output(
            chunk_dir, spec, spec_sha, args=args, campaign=campaign
        )
        current_attestation_sha = _canonical_sha256(_attestation_contract(chunk_dir))
        if attestation_contract_sha is None:
            attestation_contract_sha = current_attestation_sha
        elif current_attestation_sha != attestation_contract_sha:
            raise IsolatedCampaignBlocked("worker runtime/source attestation changed across chunks")
        rows.extend(chunk_rows)
        if args.purpose == "formal_campaign" and len(rows) == 18:
            protocol, seal = _seal_tuning(
                rows, prepared["config"], runner, admission
            )
            seal["campaign_sha256"] = campaign_sha
            tuning_path = args.output_dir.resolve() / "tuning-seal.json"
            _write_new(tuning_path, seal)
            tuning_binding = {"path": str(tuning_path), "sha256": _sha256(tuning_path)}
            selected = protocol.selection.candidate_id
    if args.purpose == "diagnostic_nonadjudicative":
        _verify_campaign_inputs(campaign_path, campaign_sha, campaign, args, runner)
        _write_new(
            args.output_dir.resolve() / "diagnostic-report.json",
            {
                "schema_name": "rj45-task-admission-isolated-diagnostic-v1",
                "purpose": args.purpose,
                "campaign_sha256": campaign_sha,
                "ordinal": 33,
                "episode": rows[0],
                "claims": {"task_admission_completed": False, "rl_pilot_admitted": False},
            },
        )
        return 0
    if protocol is None:
        raise IsolatedCampaignBlocked("formal campaign has no tuning seal")
    _verify_campaign_inputs(campaign_path, campaign_sha, campaign, args, runner)
    protocol, baseline, evidence = _adjudicate_exact90(
        rows, prepared, runner, admission, protocol
    )
    _verify_campaign_inputs(campaign_path, campaign_sha, campaign, args, runner)
    _write_new(
        args.output_dir.resolve() / "report.json",
        {
            "schema_name": "rj45-task-admission-isolated-report-v1",
            "status": protocol.decision.state,
            "campaign_identity": {"path": str(campaign_path), "sha256": campaign_sha},
            "tuning_selection": asdict(protocol.selection),
            "classical_seal": {
                "selected_candidate_id": protocol.classical_seal.selected_candidate_id,
                "canonical_controller_config_sha256": protocol.classical_seal.canonical_controller_config_sha256,
                "tuning_report_sha256": protocol.classical_seal.tuning_report_sha256,
                "selection": asdict(protocol.classical_seal.selection),
            },
            "baseline_artifact_seal": asdict(baseline),
            "contract_evidence": asdict(evidence),
            "decision": asdict(protocol.decision),
            "episodes": rows,
            "claims": {
                "task_admission_completed": True,
                "rl_pilot_admitted": protocol.decision.state == "admit_rl_pilot",
                "native_crash_cause_fixed": False,
            },
        },
    )
    return 0 if protocol.decision.state != "invalid_contract" else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--current-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--playback-actions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--purpose",
        choices=("formal_campaign", "diagnostic_nonadjudicative"),
        default="formal_campaign",
    )
    parser.add_argument("--diagnostic-ordinal", type=int, choices=(33,))
    parser.add_argument("--worker-timeout-seconds", type=float, default=2400.0)
    parser.add_argument(
        "--post-exit-verifier-timeout-seconds", type=float, default=120.0
    )
    parser.add_argument(
        "--process-group-cleanup-grace-seconds", type=float, default=2.0
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if len(
        {
            args.runtime_root.resolve(),
            args.evidence_root.resolve(),
            args.current_root.resolve(),
        }
    ) != 3:
        raise IsolatedCampaignBlocked(
            "runtime, evidence, and current roots must be distinct"
        )
    if (args.purpose == "diagnostic_nonadjudicative") != (
        args.diagnostic_ordinal == 33
    ):
        raise IsolatedCampaignBlocked(
            "diagnostic purpose requires --diagnostic-ordinal 33 and formal purpose forbids it"
        )
    if (
        args.worker_timeout_seconds <= 0
        or args.post_exit_verifier_timeout_seconds <= 0
        or args.process_group_cleanup_grace_seconds <= 0
    ):
        raise IsolatedCampaignBlocked("worker, verifier, and cleanup timeouts must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise IsolatedCampaignBlocked("isolated campaign output directory must be new or empty")
    try:
        return run_campaign(args)
    except Exception as error:
        report = args.output_dir.resolve() / "campaign-aborted.json"
        if not report.exists():
            _write_new(
                report,
                {
                    "schema_name": "rj45-task-admission-isolated-abort-v1",
                    "status": "campaign_aborted",
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "claims": {
                        "task_admission_completed": False,
                        "rl_pilot_admitted": False,
                        "native_crash_cause_fixed": False,
                    },
                },
            )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
