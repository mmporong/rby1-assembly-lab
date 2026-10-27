#!/usr/bin/env python3
"""Run the preregistered three-seed RJ45 development campaign, excluding final holdout."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from datetime import timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, NoReturn, Sequence


NONFINITE_TOKEN = re.compile(r"(?<![A-Za-z0-9_])(?:nan|[-+]?inf(?:inity)?)(?![A-Za-z0-9_])", re.I)
OOM_TOKEN = re.compile(
    r"(?:CUDA out of memory|OutOfMemoryError|oom[-_ ]kill|Killed process|std::bad_alloc)",
    re.I,
)


class CampaignError(RuntimeError):
    pass


class CampaignInterrupted(CampaignError):
    pass


@dataclass(frozen=True)
class StageResult:
    return_code: int
    failure_reason: str | None = None
    process_group_cleanup_confirmed: bool = True


@dataclass(frozen=True)
class FileRef:
    path: Path
    sha256: str


@dataclass(frozen=True)
class CheckpointSpec:
    filename: str
    iteration_index: int
    completed_updates: int


@dataclass(frozen=True)
class CampaignContract:
    project_root: Path
    plan: FileRef
    selection: FileRef
    candidate_selection: FileRef
    input_manifest: FileRef
    guard_record: FileRef
    frozen_inputs: dict[Path, str]
    profile_label: str
    training_config: FileRef
    ppo_config: FileRef
    zero_init_actor_output: bool
    seed_configs: dict[int, FileRef]
    nominal_evidence: FileRef
    held_state: FileRef
    parallel_contact_evidence: FileRef
    pilot_audit: FileRef
    training_seeds: tuple[int, ...]
    checkpoints: tuple[CheckpointSpec, ...]
    updates_per_seed: int
    num_envs: int
    evaluation_seed: int
    evaluation_num_envs: int
    attempts_per_policy: int
    minimum_free_bytes: int
    guard_started_utc: datetime
    guard_deadline_utc: datetime
    timer_target: str
    instance_id: str
    instance_name: str
    instance_organization: str
    launch_utc: datetime
    guard_max_hours: float
    monitor_interval_s: float
    hang_no_progress_timeout_s: float
    deadline_utc: datetime
    python: Path
    output_dir: Path


def _fail(message: str) -> NoReturn:
    raise CampaignError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer at least {minimum}")
    return value


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        _fail(f"{name} must be finite and at least {minimum}")
    return result


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CampaignError(f"unable to hash input {path}: {error}") from error
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    def reject(constant: str) -> None:
        _fail(f"{name} contains non-finite JSON constant {constant}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        _validate_finite_json(payload, name)
        return _object(payload, name)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid {name} JSON {path}: {error}") from error


def _validate_finite_json(value: Any, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _fail(f"{name} contains a non-finite numeric value")
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_finite_json(item, f"{name}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_json(item, f"{name}[{index}]")


def _parse_deadline(value: Any) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        _fail("deadline must use exact UTC format YYYY-MM-DDTHH:MM:SSZ")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise CampaignError(f"invalid UTC deadline: {error}") from error


def _resolve_project_path(project_root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        _fail(f"{name} must be a non-empty project-relative path")
    relative = Path(value)
    if ".." in relative.parts:
        _fail(f"{name} must not contain '..'")
    resolved = (project_root / relative).resolve()
    if not resolved.is_relative_to(project_root):
        _fail(f"{name} escapes the project root")
    return resolved


def _file_ref(project_root: Path, value: Any, name: str) -> FileRef:
    payload = _object(value, name)
    return FileRef(
        _resolve_project_path(project_root, payload.get("path"), f"{name}.path"),
        _sha(payload.get("sha256"), f"{name}.sha256"),
    )


def _relative(project_root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(project_root))


def _verify_ref(reference: FileRef, name: str) -> None:
    actual = _sha_file(reference.path)
    if actual != reference.sha256:
        _fail(f"{name} SHA-256 mismatch: {reference.path}")


def _validate_seed_config(
    base: dict[str, Any], seed_payload: dict[str, Any], *, seed: int, plan: dict[str, Any]
) -> None:
    training = _object(plan.get("training"), "plan.training")
    expected = json.loads(json.dumps(base, allow_nan=False))
    expected["seed"] = seed
    expected["max_iterations"] = training["updates_per_seed"]
    expected["save_interval"] = training["checkpoint_save_interval_iterations"]
    if seed_payload != expected:
        _fail(
            f"seed {seed} PPO config must differ from selected PPO only by seed, "
            "max_iterations, and save_interval"
        )
    if seed_payload.get("num_steps_per_env") != training["steps_per_env_per_update"]:
        _fail(f"seed {seed} PPO steps_per_env differs from plan")


def _audit_file_ref(project_root: Path, value: Any, name: str) -> FileRef:
    payload = _object(value, name)
    raw_path = payload.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        _fail(f"{name}.path must be non-empty")
    path = Path(raw_path)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    if not resolved.is_relative_to(project_root):
        _fail(f"{name}.path escapes the project root")
    return FileRef(resolved, _sha(payload.get("sha256"), f"{name}.sha256"))


def _revalidate_pilot_report(
    project_root: Path,
    report_ref: FileRef,
    label: str,
    expected_attempts: int,
    frozen_analyzer_path: Path,
) -> dict[str, int | float]:
    _verify_ref(report_ref, f"raw pilot report {label}")
    report = _load_json(report_ref.path, f"raw pilot report {label}")
    analyzer = _load_local_module(
        frozen_analyzer_path,
        "_rj45_analyzer_for_candidate_revalidation",
    )
    try:
        outcomes = analyzer._validate_report(report, label, expected_attempts)
    except Exception as error:
        raise CampaignError(f"raw pilot report {label} failed analyzer validation: {error}") from error
    ordered = list(outcomes.values())
    forbidden = max(
        (float(outcome.get("forbidden_contact_load_peak_n") or 0.0) for outcome in ordered),
        default=0.0,
    )
    return {
        "attempts": len(ordered),
        "successes_all_attempts": sum(
            bool(outcome["valid"] and outcome["success"]) for outcome in ordered
        ),
        "invalid_episodes": sum(not bool(outcome["valid"]) for outcome in ordered),
        "maximum_forbidden_contact_load_peak_n": forbidden,
    }


def _revalidate_pilot_bundle(
    project_root: Path,
    bundle: dict[str, FileRef],
    report_ref: FileRef,
    label: str,
    frozen_analyzer_path: Path,
) -> None:
    analyzer = _load_local_module(
        frozen_analyzer_path,
        "_rj45_frozen_analyzer_for_bundle_revalidation",
    )
    try:
        report = _load_json(report_ref.path, f"raw pilot report {label}")
        result = analyzer._verify_training_bundle(
            label,
            report,
            checkpoint_path=bundle["checkpoint"].path,
            manifest_path=bundle["checkpoint_manifest"].path,
            training_provenance_path=bundle["training_provenance"].path,
            runner_config_path=bundle["runner_config"].path,
        )
    except Exception as error:
        raise CampaignError(f"pilot bundle {label} failed inspector validation: {error}") from error
    if (
        result.get("status")
        != "checkpoint_manifest_provenance_runner_bytes_verified"
        or result.get("training_updates") != 20
    ):
        _fail(f"pilot bundle {label} is not the frozen v1 20-update bundle")


def _validate_candidate_selection(
    *,
    project_root: Path,
    plan: dict[str, Any],
    plan_ref: FileRef,
    candidate_ref: FileRef,
    frozen_inputs: dict[Path, str],
) -> tuple[str, FileRef, FileRef, bool, FileRef]:
    _verify_ref(candidate_ref, "candidate selection")
    if frozen_inputs.get(candidate_ref.path) != candidate_ref.sha256:
        _fail("input manifest does not pin candidate selection")
    candidate = _load_json(candidate_ref.path, "candidate selection")
    if candidate.get("schema_version") != 2:
        _fail("candidate selection schema_version must be 2")
    decision = _object(candidate.get("decision"), "candidate selection decision")
    profile_label = decision.get("selected_group")
    eligible_groups = _object(
        plan.get("pilot_candidate_selection"), "plan.pilot_candidate_selection"
    ).get("eligible_groups")
    if (
        not isinstance(profile_label, str)
        or not isinstance(eligible_groups, list)
        or profile_label not in eligible_groups
    ):
        _fail("candidate selection chose an ineligible profile")
    if decision.get("reward_sums_used") is not False:
        _fail("candidate selection must not use incomparable reward sums")
    if decision.get("selection_metric") != "strict-v1 successes_all_attempts":
        _fail("candidate selection metric differs from the long-run plan")
    if decision.get("tie_break_order") != _object(
        plan.get("pilot_candidate_selection"), "plan.pilot_candidate_selection"
    ).get("tie_break_order"):
        _fail("candidate selection tie-break order differs from the long-run plan")

    inputs = _object(candidate.get("inputs"), "candidate selection inputs")
    candidate_plan = _file_ref(
        project_root, inputs.get("long_run_plan"), "candidate inputs.long_run_plan"
    )
    if candidate_plan != plan_ref:
        _fail("candidate selection references a different long-run plan")
    parent_plan_ref = _file_ref(
        project_root, inputs.get("parent_pilot_plan"), "candidate inputs.parent_pilot_plan"
    )
    if parent_plan_ref.sha256 != _sha(
        plan.get("parent_pilot_plan_sha256"), "plan parent pilot plan SHA"
    ):
        _fail("candidate selection references a different parent pilot plan")
    pilot_audit = _file_ref(
        project_root, inputs.get("stored_paired_audit"), "candidate inputs.stored_paired_audit"
    )
    frozen_analyzer = _file_ref(
        project_root, inputs.get("frozen_analyzer"), "candidate inputs.frozen_analyzer"
    )
    if frozen_analyzer.path != project_root / "scripts/analyze_rj45_paired.py":
        _fail("candidate must use the frozen pilot analyzer entrypoint")
    raw_evidence = _object(
        inputs.get("raw_evidence_sha256"), "candidate inputs.raw_evidence_sha256"
    )
    raw_evidence_paths = {
        _resolve_project_path(project_root, path, "candidate raw evidence path"): _sha(
            digest, f"candidate raw evidence digest for {path}"
        )
        for path, digest in raw_evidence.items()
    }
    immutable_inputs = _object(
        inputs.get("immutable_inputs_sha256"),
        "candidate inputs.immutable_inputs_sha256",
    )
    immutable_paths = {
        _resolve_project_path(project_root, path, "candidate immutable input path"): _sha(
            digest, f"candidate immutable input digest for {path}"
        )
        for path, digest in immutable_inputs.items()
    }
    if not immutable_paths:
        _fail("candidate selection immutable input map is empty")
    for path, digest in immutable_paths.items():
        if _sha_file(path) != digest:
            _fail(f"candidate immutable input SHA-256 mismatch: {path}")
        if frozen_inputs.get(path) != digest:
            _fail(f"launch input manifest does not pin candidate immutable input: {path}")
    verification = _object(
        candidate.get("verification_scope"), "candidate verification_scope"
    )
    if verification.get("status") != (
        "raw_reports_and_training_bundles_rehashed_and_fresh_analyzer_replayed"
    ):
        _fail("candidate selection was not rebuilt from raw reports and bundles")
    for label, reference in {
        "parent pilot plan": parent_plan_ref,
        "pilot audit": pilot_audit,
        "frozen analyzer": frozen_analyzer,
    }.items():
        _verify_ref(reference, label)
        if frozen_inputs.get(reference.path) != reference.sha256:
            _fail(f"input manifest does not pin {label}")
    if immutable_paths.get(frozen_analyzer.path) != frozen_analyzer.sha256:
        _fail("candidate immutable inputs do not pin its frozen analyzer")
    candidate_source_paths = {
        path
        for path in immutable_paths
        if path.parent == project_root / "src/rby1_assembly" and path.suffix == ".py"
    }
    if candidate_source_paths != set((project_root / "src/rby1_assembly").glob("*.py")):
        _fail("candidate immutable source path set differs from src/rby1_assembly/*.py")

    parent_plan = _load_json(parent_plan_ref.path, "parent pilot plan")
    experiments = parent_plan.get("groups")
    if not isinstance(experiments, list):
        _fail("parent pilot plan groups must be a list")
    experiment_by_group = {
        _object(item, "parent experiment").get("id"): item for item in experiments
    }
    if set(experiment_by_group) != set(eligible_groups):
        _fail("parent pilot plan group set differs from long-run eligible groups")
    parent_frozen = _object(
        _object(parent_plan.get("frozen_inputs"), "parent plan frozen_inputs").get(
            "sha256"
        ),
        "parent plan frozen_inputs.sha256",
    )
    selected_profile = _object(candidate.get("selected_profile"), "selected_profile")
    training_config = _file_ref(
        project_root, selected_profile.get("training_config"), "selected training config"
    )
    ppo_config = _file_ref(
        project_root, selected_profile.get("ppo_config"), "selected PPO config"
    )
    zero_init = selected_profile.get("zero_init_actor_output")
    if type(zero_init) is not bool:
        _fail("selected_profile.zero_init_actor_output must be boolean")
    for label, reference in {
        "selected training config": training_config,
        "selected PPO config": ppo_config,
    }.items():
        if immutable_paths.get(reference.path) != reference.sha256:
            _fail(f"candidate immutable inputs do not pin {label}")
    parent_group = _object(experiment_by_group[profile_label], "selected parent experiment")
    expected_training_path = _resolve_project_path(
        project_root, parent_group.get("training_config"), "parent training config"
    )
    expected_ppo_path = _resolve_project_path(
        project_root, parent_group.get("ppo_config"), "parent PPO config"
    )
    parent_flags = parent_group.get("training_flags", [])
    if not isinstance(parent_flags, list) or any(not isinstance(flag, str) for flag in parent_flags):
        _fail("parent group training_flags must be a list of strings")
    expected_zero = "--zero-init-actor-output" in parent_flags
    if (
        training_config.path != expected_training_path
        or ppo_config.path != expected_ppo_path
        or zero_init != expected_zero
    ):
        _fail("selected profile settings do not match its parent-plan group label")
    for label, reference in {
        "selected training config": training_config,
        "selected PPO config": ppo_config,
    }.items():
        relative = _relative(project_root, reference.path)
        if parent_frozen.get(relative) != reference.sha256:
            _fail(f"{label} SHA differs from the parent plan frozen input")

    audit = _load_json(pilot_audit.path, "stored paired audit")
    if audit.get("schema_version") != 1:
        _fail("stored paired audit schema_version must be frozen pilot version 1")
    audit_results = _object(audit.get("results"), "paired audit results")
    audit_reports = _object(audit.get("input_reports"), "paired audit input_reports")
    audit_bundles = _object(
        audit.get("training_artifact_verification"),
        "paired audit training_artifact_verification",
    )
    evidence_by_group = _object(
        candidate.get("candidate_evidence"), "candidate_evidence"
    )
    expected_evidence_groups = set(eligible_groups)
    if set(evidence_by_group) != expected_evidence_groups:
        _fail("candidate evidence must cover baseline and every eligible group")
    qualifying: list[tuple[str, int]] = []
    raw_summaries: dict[str, dict[str, int | float]] = {}
    for group in eligible_groups:
        summary = _object(audit_results.get(group), f"paired audit results.{group}")
        attempts = _integer(summary.get("attempts"), f"{group} attempts", minimum=1)
        successes = _integer(
            summary.get("successes_all_attempts"), f"{group} successes"
        )
        invalid = _integer(summary.get("invalid_episodes"), f"{group} invalid episodes")
        evidence = _object(evidence_by_group.get(group), f"candidate evidence.{group}")
        raw_report = _file_ref(
            project_root, evidence.get("raw_report"), f"candidate evidence.{group}.raw_report"
        )
        raw_report_payload = _load_json(raw_report.path, f"raw pilot report {group}")
        if group == profile_label:
            report_training = _object(
                raw_report_payload.get("rj45_training_config"),
                f"raw pilot report {group}.rj45_training_config",
            )
            report_provenance = _object(
                raw_report_payload.get("provenance"),
                f"raw pilot report {group}.provenance",
            )
            reported_policy_path = report_provenance.get("policy_config")
            expected_policy_path = _relative(project_root, ppo_config.path)
            normalized_policy_path = (
                Path(reported_policy_path).as_posix()
                if isinstance(reported_policy_path, str)
                else ""
            )
            if (
                report_training.get("path") != _relative(project_root, training_config.path)
                or report_training.get("sha256") != training_config.sha256
                or (
                    normalized_policy_path != expected_policy_path
                    and not normalized_policy_path.endswith(f"/{expected_policy_path}")
                )
                or report_provenance.get("policy_config_sha256") != ppo_config.sha256
            ):
                _fail("selected profile configs differ from its raw pilot report")
        audit_raw_report = _audit_file_ref(
            project_root, audit_reports.get(group), f"audit input report.{group}"
        )
        if raw_report != audit_raw_report:
            _fail(f"candidate raw report differs from paired audit for {group}")
        bundle = _object(
            evidence.get("checkpoint_bundle"),
            f"candidate evidence.{group}.checkpoint_bundle",
        )
        audit_bundle = _object(audit_bundles.get(group), f"audit bundle.{group}")
        if audit_bundle.get("status") != "checkpoint_manifest_provenance_runner_bytes_verified":
            _fail(f"paired audit bundle is not verified for {group}")
        bundle_refs: dict[str, FileRef] = {}
        for field in ("checkpoint", "checkpoint_manifest", "training_provenance", "runner_config"):
            reference = _file_ref(
                project_root, bundle.get(field), f"candidate evidence.{group}.bundle.{field}"
            )
            audit_reference = _audit_file_ref(
                project_root, audit_bundle.get(field), f"audit bundle.{group}.{field}"
            )
            if reference != audit_reference:
                _fail(f"candidate bundle differs from paired audit for {group}.{field}")
            if raw_evidence_paths.get(reference.path) != reference.sha256:
                _fail(f"candidate raw evidence map omits {group}.{field}")
            if frozen_inputs.get(reference.path) != reference.sha256:
                _fail(f"input manifest does not pin {group}.{field}")
            bundle_refs[field] = reference
        _revalidate_pilot_bundle(
            project_root,
            bundle_refs,
            raw_report,
            group,
            frozen_analyzer.path,
        )
        if raw_evidence_paths.get(raw_report.path) != raw_report.sha256:
            _fail(f"candidate raw evidence map omits {group} report")
        if frozen_inputs.get(raw_report.path) != raw_report.sha256:
            _fail(f"input manifest does not pin {group} report")
        raw_summary = _revalidate_pilot_report(
            project_root, raw_report, group, 32, frozen_analyzer.path
        )
        raw_summaries[group] = raw_summary
        if any(summary.get(field) != raw_summary[field] for field in (
            "attempts", "successes_all_attempts", "invalid_episodes"
        )):
            _fail(f"paired audit aggregate differs from raw report for {group}")
        forbidden_peak = float(raw_summary["maximum_forbidden_contact_load_peak_n"])
        if attempts == 32 and invalid == 0 and forbidden_peak == 0.0:
            qualifying.append((group, successes))
    baseline_report = _audit_file_ref(
        project_root, audit_reports.get("baseline"), "audit baseline report"
    )
    if raw_evidence_paths.get(baseline_report.path) != baseline_report.sha256:
        _fail("candidate raw evidence map omits baseline report")
    if frozen_inputs.get(baseline_report.path) != baseline_report.sha256:
        _fail("input manifest does not pin baseline report")
    _revalidate_pilot_report(
        project_root, baseline_report, "baseline", 32, frozen_analyzer.path
    )
    if not qualifying:
        _fail("no pilot candidate satisfies the preregistered qualification rule")
    tie_break = _object(
        plan.get("pilot_candidate_selection"), "plan.pilot_candidate_selection"
    )["tie_break_order"]
    tie_index = {group: index for index, group in enumerate(tie_break)}
    expected_group, expected_successes = sorted(
        qualifying, key=lambda item: (-item[1], tie_index[item[0]])
    )[0]
    if profile_label != expected_group:
        _fail("candidate selection decision disagrees with raw paired audit ranking")
    if (
        decision.get("selected_successes_all_attempts") != expected_successes
        or decision.get("attempts") != 32
    ):
        _fail("candidate selection aggregate disagrees with raw paired audit")
    ranking = candidate.get("ranking")
    if not isinstance(ranking, list) or len(ranking) != len(eligible_groups):
        _fail("candidate ranking must cover every eligible group")
    ranking_by_group = {
        _object(row, "candidate ranking row").get("group"): row for row in ranking
    }
    if set(ranking_by_group) != set(eligible_groups):
        _fail("candidate ranking group set differs from the long-run plan")
    for group in eligible_groups:
        row = _object(ranking_by_group[group], f"candidate ranking.{group}")
        summary = _object(audit_results.get(group), f"paired audit results.{group}")
        raw_summary = raw_summaries[group]
        expected_eligible = (
            raw_summary["attempts"] == 32
            and raw_summary["invalid_episodes"] == 0
            and raw_summary["maximum_forbidden_contact_load_peak_n"] == 0.0
        )
        if (
            row.get("successes_all_attempts") != summary.get("successes_all_attempts")
            or row.get("attempts") != summary.get("attempts")
            or row.get("invalid_episodes") != summary.get("invalid_episodes")
            or row.get("eligible") is not expected_eligible
            or row.get("maximum_forbidden_contact_load_peak_n")
            != raw_summary["maximum_forbidden_contact_load_peak_n"]
            or row.get("checkpoint_bundle_status")
            != "checkpoint_manifest_provenance_runner_bytes_verified"
            or row.get("tie_break_position") != tie_index[group]
        ):
            _fail(f"candidate ranking differs from raw paired audit for {group}")
    for reference in (training_config, ppo_config):
        _verify_ref(reference, "selected profile config")
        if frozen_inputs.get(reference.path) != reference.sha256:
            _fail("input manifest does not pin selected profile config")
    return profile_label, training_config, ppo_config, zero_init, pilot_audit


def load_contract(
    *,
    project_root: Path,
    plan_path: Path,
    expected_plan_sha256: str,
    selection_path: Path,
    expected_selection_sha256: str,
    expected_instance_id: str,
    expected_instance_name: str,
    expected_instance_organization: str,
    output_dir: Path,
    deadline_utc: str,
    python: Path,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    disk_free: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
) -> CampaignContract:
    project_root = project_root.resolve()
    plan_path = plan_path.resolve()
    selection_path = selection_path.resolve()
    output_dir = output_dir.resolve()
    if not project_root.is_dir():
        _fail(f"project root is not a directory: {project_root}")
    if output_dir.exists():
        _fail(f"output directory must not exist: {output_dir}")
    if not output_dir.parent.is_dir():
        _fail(f"output parent directory must exist: {output_dir.parent}")
    deadline = _parse_deadline(deadline_utc)
    current = now()
    if current.tzinfo is None or current.utcoffset() is None:
        _fail("current time provider must return a timezone-aware value")
    if deadline <= current.astimezone(timezone.utc):
        _fail("UTC deadline has already passed")
    expected_plan_sha256 = _sha(expected_plan_sha256, "expected plan SHA-256")
    expected_selection_sha256 = _sha(
        expected_selection_sha256, "expected selection SHA-256"
    )
    plan_ref = FileRef(plan_path, expected_plan_sha256)
    selection_ref = FileRef(selection_path, expected_selection_sha256)
    _verify_ref(plan_ref, "plan")
    _verify_ref(selection_ref, "selection")
    plan = _load_json(plan_path, "long-run plan")
    selection = _load_json(selection_path, "campaign launch manifest")
    if plan.get("schema_version") != 1 or selection.get("schema_version") != 1:
        _fail("plan and campaign launch manifest schema_version must be 1")
    if selection.get("status") != "ready_for_long_campaign":
        _fail("campaign launch manifest is not ready for long campaign")
    if selection.get("long_run_plan_sha256") != expected_plan_sha256:
        _fail("campaign launch manifest does not reference the supplied long-run plan")
    guard_started = _parse_deadline(selection.get("guard_started_utc"))
    guard_deadline = _parse_deadline(selection.get("guard_deadline_utc"))
    if guard_deadline != deadline:
        _fail("CLI UTC deadline differs from the launch guard deadline")
    instance = _object(selection.get("brev_instance"), "launch.brev_instance")
    expected_identity = {
        "id": expected_instance_id,
        "name": expected_instance_name,
        "organization": expected_instance_organization,
    }
    if any(not isinstance(value, str) or not value for value in expected_identity.values()):
        _fail("expected Brev instance identity fields must be non-empty strings")
    if instance != expected_identity:
        _fail("launch Brev instance identity differs from caller expectations")

    input_manifest_ref = _file_ref(
        project_root, selection.get("input_manifest"), "launch.input_manifest"
    )
    _verify_ref(input_manifest_ref, "input manifest")
    input_manifest = _load_json(input_manifest_ref.path, "input manifest")
    if input_manifest.get("schema_version") != 1:
        _fail("input manifest schema_version must be 1")
    raw_inputs = _object(input_manifest.get("inputs"), "input_manifest.inputs")
    frozen_inputs: dict[Path, str] = {}
    for raw_path, raw_digest in raw_inputs.items():
        path = _resolve_project_path(project_root, raw_path, "input manifest path")
        if path in frozen_inputs:
            _fail(f"duplicate resolved input manifest path: {raw_path}")
        frozen_inputs[path] = _sha(raw_digest, f"input manifest digest for {raw_path}")
    if not frozen_inputs:
        _fail("input manifest must contain at least one input")

    candidate_ref = _file_ref(
        project_root, selection.get("candidate_selection"), "launch.candidate_selection"
    )
    guard_record_ref = _file_ref(
        project_root, selection.get("guard_record"), "launch.guard_record"
    )
    _verify_ref(guard_record_ref, "guard record")
    if frozen_inputs.get(guard_record_ref.path) != guard_record_ref.sha256:
        _fail("input manifest does not pin the launch guard record")
    timer_target = _nonempty_string(selection.get("timer_target"), "launch timer target")
    if selection.get("guard_verification_boundary") != (
        "record_and_anchor_are_local_integrity_evidence;_control_plane_and_timer_must_be_observed_by_operator_immediately_before_launch"
    ):
        _fail("launch does not preserve the guard's local-only verification boundary")
    guard_record = _load_json(guard_record_ref.path, "guard record")
    if guard_record.get("schema_version") != 1:
        _fail("guard record schema_version must be 1")
    if guard_record.get("candidate_selection_sha256") != candidate_ref.sha256:
        _fail("guard record candidate SHA differs from launch candidate")
    if guard_record.get("long_run_plan_sha256") != plan_ref.sha256:
        _fail("guard record plan SHA differs from the supplied plan")
    if _parse_deadline(guard_record.get("guard_started_utc")) != guard_started:
        _fail("guard record start differs from launch")
    if _parse_deadline(guard_record.get("guard_deadline_utc")) != guard_deadline:
        _fail("guard record deadline differs from launch")
    if guard_record.get("brev_instance") != expected_identity:
        _fail("guard record Brev identity differs from launch and caller expectations")
    if guard_record.get("timer_target") != timer_target:
        _fail("guard record timer target differs from launch")
    if guard_record.get("observation_scope") != "operator_recorded_not_provider_authenticated":
        _fail("guard record overstates its authentication boundary")
    guard_anchor_path = project_root / "artifacts/rj45-long-campaign-guards" / (
        f"{candidate_ref.sha256}-{plan_ref.sha256}.anchor.json"
    )
    expected_anchor = {
        "schema_version": 1,
        "candidate_selection_sha256": candidate_ref.sha256,
        "long_run_plan_sha256": plan_ref.sha256,
        "guard_record": {
            "path": _relative(project_root, guard_record_ref.path),
            "sha256": guard_record_ref.sha256,
        },
        "guard_started_utc": guard_started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guard_deadline_utc": guard_deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "brev_instance": expected_identity,
        "timer_target": timer_target,
        "scope": "local_no_clobber_reissuance_anchor_not_provider_authenticated",
    }
    if _load_json(guard_anchor_path, "guard anchor") != expected_anchor:
        _fail("local guard anchor differs from the launch guard contract")

    training = _object(plan.get("training"), "plan.training")
    development = _object(
        plan.get("development_evaluation"), "plan.development_evaluation"
    )
    final = _object(plan.get("final_evaluation"), "plan.final_evaluation")
    budget = _object(plan.get("budget"), "plan.budget")
    guard_max_hours = _finite(
        budget.get("bounded_guard_max_hours_from_follow_up_start"),
        "bounded guard max hours",
        minimum=0.0,
    )
    if guard_max_hours != 9.0:
        _fail("long-run plan must retain the preregistered single 9-hour guard")
    launch_utc = current.astimezone(timezone.utc)
    if guard_started > launch_utc:
        _fail("campaign guard start is in the future")
    if deadline > guard_started + timedelta(hours=guard_max_hours):
        _fail("UTC deadline exceeds the plan's single 9-hour campaign guard")
    if final.get("status") != "sealed_until_candidate_and_common_checkpoint_and_final_protocol_are_frozen":
        _fail("final holdout is not sealed in the plan")
    seeds_raw = training.get("training_seeds")
    if not isinstance(seeds_raw, list) or len(seeds_raw) != 3:
        _fail("plan must contain exactly three training seeds")
    seeds = tuple(_integer(seed, "training seed") for seed in seeds_raw)
    if len(set(seeds)) != 3:
        _fail("training seeds must be unique")
    if training.get("updates_per_seed") != 200:
        _fail("long-run plan must specify 200 updates per seed")
    assessed = training.get("assessed_checkpoints")
    if not isinstance(assessed, list) or len(assessed) != 3:
        _fail("plan must contain exactly three assessed checkpoints")
    checkpoints = tuple(
        CheckpointSpec(
            filename=_nonempty_string(
                _object(item, "assessed checkpoint").get("filename"), "checkpoint filename"
            ),
            iteration_index=_integer(item.get("iteration_index"), "checkpoint iteration"),
            completed_updates=_integer(
                item.get("completed_updates"), "checkpoint completed updates", minimum=1
            ),
        )
        for item in assessed
    )
    expected_checkpoints = (
        CheckpointSpec("model_19.pt", 19, 20),
        CheckpointSpec("model_95.pt", 95, 96),
        CheckpointSpec("model_199.pt", 199, 200),
    )
    if checkpoints != expected_checkpoints:
        _fail("assessed checkpoints differ from preregistered 19/95/199 contract")
    if development.get("seed") == final.get("seed"):
        _fail("development seed must differ from sealed final seed")

    profile_label, training_config, ppo_config, zero_init, pilot_audit = (
        _validate_candidate_selection(
            project_root=project_root,
            plan=plan,
            plan_ref=plan_ref,
            candidate_ref=candidate_ref,
            frozen_inputs=frozen_inputs,
        )
    )
    launch_profile = _object(
        selection.get("selected_profile"), "launch.selected_profile"
    )
    if (
        _file_ref(
            project_root,
            launch_profile.get("training_config"),
            "launch selected training config",
        )
        != training_config
        or _file_ref(
            project_root,
            launch_profile.get("ppo_config"),
            "launch selected PPO config",
        )
        != ppo_config
        or launch_profile.get("zero_init_actor_output") is not zero_init
    ):
        _fail("launch selected profile differs from candidate selection")
    monitoring = _object(selection.get("process_monitoring"), "launch.process_monitoring")
    monitor_interval_s = _finite(
        monitoring.get("poll_interval_s"), "monitor poll interval", minimum=0.01
    )
    hang_timeout_s = _finite(
        monitoring.get("hang_no_progress_timeout_s"),
        "hang no-progress timeout",
        minimum=monitor_interval_s,
    )
    runtime_inputs = _object(selection.get("runtime_inputs"), "launch.runtime_inputs")
    nominal = _file_ref(project_root, runtime_inputs.get("nominal_evidence"), "nominal evidence")
    held = _file_ref(project_root, runtime_inputs.get("held_state"), "held state")
    parallel = _file_ref(
        project_root,
        runtime_inputs.get("parallel_contact_evidence"),
        "parallel contact evidence",
    )
    raw_seed_configs = _object(selection.get("seed_ppo_configs"), "launch.seed_ppo_configs")
    if set(raw_seed_configs) != {str(seed) for seed in seeds}:
        _fail("launch manifest must identify one PPO config for each preregistered training seed")
    seed_configs = {
        seed: _file_ref(
            project_root, raw_seed_configs[str(seed)], f"launch.seed_ppo_configs.{seed}"
        )
        for seed in seeds
    }

    required_refs = {
        "plan": plan_ref,
        "candidate selection": candidate_ref,
        "guard record": guard_record_ref,
        "selected training config": training_config,
        "selected PPO config": ppo_config,
        "pilot audit": pilot_audit,
        "nominal evidence": nominal,
        "held state": held,
        "parallel contact evidence": parallel,
        **{f"seed {seed} PPO config": reference for seed, reference in seed_configs.items()},
    }
    required_scripts = (
        project_root / "scripts/train.py",
        project_root / "scripts/evaluate_policy.py",
        project_root / "scripts/inspect_rj45_checkpoint.py",
        project_root / "scripts/analyze_rj45_paired.py",
        project_root / "scripts/run_rj45_long_campaign.py",
    )
    for label, reference in required_refs.items():
        _verify_ref(reference, label)
        if frozen_inputs.get(reference.path) != reference.sha256:
            _fail(f"input manifest does not pin {label}")
    for path in required_scripts:
        if path not in frozen_inputs:
            _fail(f"input manifest does not pin required script {_relative(project_root, path)}")
    source_paths = set((project_root / "src/rby1_assembly").glob("*.py"))
    manifest_source_paths = {
        path for path in frozen_inputs if path.parent == project_root / "src/rby1_assembly" and path.suffix == ".py"
    }
    if not source_paths or manifest_source_paths != source_paths:
        _fail("input manifest source path set differs from src/rby1_assembly/*.py")
    base_ppo = _load_json(ppo_config.path, "selected PPO config")
    for seed, reference in seed_configs.items():
        _validate_seed_config(
            base_ppo,
            _load_json(reference.path, f"seed {seed} PPO config"),
            seed=seed,
            plan=plan,
        )
    for path, digest in frozen_inputs.items():
        if _sha_file(path) != digest:
            _fail(f"frozen input SHA-256 mismatch: {_relative(project_root, path)}")

    minimum_free = _integer(
        budget.get("minimum_free_bytes_local_and_remote"), "minimum free bytes", minimum=1
    )
    if disk_free(output_dir.parent) < minimum_free:
        _fail("insufficient free disk space before campaign start")
    if not python.is_file():
        _fail(f"Python executable does not exist: {python}")
    return CampaignContract(
        project_root=project_root,
        plan=plan_ref,
        selection=selection_ref,
        candidate_selection=candidate_ref,
        input_manifest=input_manifest_ref,
        guard_record=guard_record_ref,
        frozen_inputs=frozen_inputs,
        profile_label=profile_label,
        training_config=training_config,
        ppo_config=ppo_config,
        zero_init_actor_output=zero_init,
        seed_configs=seed_configs,
        nominal_evidence=nominal,
        held_state=held,
        parallel_contact_evidence=parallel,
        pilot_audit=pilot_audit,
        training_seeds=seeds,
        checkpoints=checkpoints,
        updates_per_seed=_integer(training["updates_per_seed"], "updates per seed", minimum=1),
        num_envs=_integer(training.get("num_envs"), "training num_envs", minimum=1),
        evaluation_seed=_integer(development.get("seed"), "development seed"),
        evaluation_num_envs=_integer(
            development.get("num_envs"), "development num_envs", minimum=1
        ),
        attempts_per_policy=_integer(
            development.get("attempts_per_policy"), "development attempts", minimum=1
        ),
        minimum_free_bytes=minimum_free,
        guard_started_utc=guard_started,
        guard_deadline_utc=guard_deadline,
        timer_target=timer_target,
        instance_id=expected_instance_id,
        instance_name=expected_instance_name,
        instance_organization=expected_instance_organization,
        launch_utc=launch_utc,
        guard_max_hours=guard_max_hours,
        monitor_interval_s=monitor_interval_s,
        hang_no_progress_timeout_s=hang_timeout_s,
        deadline_utc=deadline,
        python=Path(os.path.abspath(python)),
        output_dir=output_dir,
    )


def build_train_command(contract: CampaignContract, seed: int) -> list[str]:
    command = [
        str(contract.python),
        "scripts/train.py",
        "--task", "rj45",
        "--num-envs", str(contract.num_envs),
        "--max-iterations", str(contract.updates_per_seed),
        "--config", _relative(contract.project_root, contract.seed_configs[seed].path),
        "--log-root", str(contract.output_dir / "runs" / str(seed)),
        "--rj45-nominal-evidence", _relative(contract.project_root, contract.nominal_evidence.path),
        "--rj45-training-config", _relative(contract.project_root, contract.training_config.path),
        "--rj45-held-state", _relative(contract.project_root, contract.held_state.path),
        "--parallel-contact-evidence", _relative(contract.project_root, contract.parallel_contact_evidence.path),
        "--headless",
    ]
    if contract.zero_init_actor_output:
        command.append("--zero-init-actor-output")
    return command


def build_inspection_command(
    contract: CampaignContract, run_dir: Path, checkpoint: CheckpointSpec, output: Path
) -> list[str]:
    return [
        str(contract.python),
        "scripts/inspect_rj45_checkpoint.py",
        "--checkpoint", str(run_dir / checkpoint.filename),
        "--manifest", str(run_dir / "checkpoint_manifest.json"),
        "--training-provenance", str(run_dir / "training_provenance.json"),
        "--runner-config", str(run_dir / "runner_config.json"),
        "--training-script", str(contract.project_root / "scripts/train.py"),
        "--output", str(output),
    ]


def build_evaluation_command(
    contract: CampaignContract, seed: int, checkpoint_path: Path, report: Path
) -> list[str]:
    return [
        str(contract.python),
        "scripts/evaluate_policy.py",
        str(checkpoint_path),
        "--task", "rj45",
        "--baseline-mode", "learned",
        "--num-envs", str(contract.evaluation_num_envs),
        "--episodes", str(contract.attempts_per_policy),
        "--seed", str(contract.evaluation_seed),
        "--config", _relative(contract.project_root, contract.seed_configs[seed].path),
        "--report", str(report),
        "--rj45-nominal-evidence", _relative(contract.project_root, contract.nominal_evidence.path),
        "--rj45-training-config", _relative(contract.project_root, contract.training_config.path),
        "--rj45-held-state", _relative(contract.project_root, contract.held_state.path),
        "--evaluation-start", "held",
        "--parallel-contact-evidence", _relative(contract.project_root, contract.parallel_contact_evidence.path),
        "--headless",
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _event(output_dir: Path, payload: dict[str, Any]) -> None:
    record = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **payload}
    with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _verify_frozen(contract: CampaignContract) -> None:
    fixed = {
        contract.plan.path: contract.plan.sha256,
        contract.selection.path: contract.selection.sha256,
        contract.input_manifest.path: contract.input_manifest.sha256,
        **contract.frozen_inputs,
    }
    for path, digest in fixed.items():
        if _sha_file(path) != digest:
            _fail(f"frozen campaign input changed: {path}")
    current_sources = set((contract.project_root / "src/rby1_assembly").glob("*.py"))
    frozen_sources = {
        path
        for path in contract.frozen_inputs
        if path.parent == contract.project_root / "src/rby1_assembly"
        and path.suffix == ".py"
    }
    if current_sources != frozen_sources:
        _fail("src/rby1_assembly/*.py path set changed during campaign")


def _check_disk(contract: CampaignContract, disk_free: Callable[[Path], int]) -> None:
    if disk_free(contract.output_dir) < contract.minimum_free_bytes:
        _fail("free disk space fell below the preregistered minimum")


def _process_group_snapshot(process_group: int, log_path: Path) -> tuple[int, int, tuple[str, ...]]:
    try:
        log_size = log_path.stat().st_size
    except OSError:
        log_size = -1
    cpu_ticks = 0
    states: list[str] = []
    proc = Path("/proc")
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                raw = (entry / "stat").read_text(encoding="utf-8")
                fields = raw[raw.rfind(")") + 2 :].split()
                if int(fields[2]) != process_group:
                    continue
                states.append(fields[0])
                cpu_ticks += int(fields[11]) + int(fields[12])
            except (OSError, ValueError, IndexError):
                continue
    return log_size, cpu_ticks, tuple(sorted(states))


def _process_group_exists(process_group: int) -> bool:
    proc = Path("/proc")
    if proc.is_dir():
        states = _process_group_snapshot(process_group, Path("/dev/null"))[2]
        if states and all(state == "Z" for state in states):
            return False
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(process_group: int, *, grace_s: float = 10.0) -> None:
    if not _process_group_exists(process_group):
        return
    os.killpg(process_group, signal.SIGTERM)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group):
            return
        time.sleep(0.05)
    os.killpg(process_group, signal.SIGKILL)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _process_group_exists(process_group):
            return
        time.sleep(0.05)
    _fail(f"owned process group {process_group} remained after SIGKILL")


def _terminate_and_reap(process: subprocess.Popen[Any], process_group: int) -> None:
    _terminate_process_group(process_group)
    try:
        process.wait(timeout=2.0)
    except subprocess.TimeoutExpired as error:
        raise CampaignError(f"owned process {process.pid} was not reaped") from error


def run_subprocess(
    command: Sequence[str],
    cwd: Path,
    log_path: Path,
    deadline: datetime,
    monitor: Callable[[], None],
    monitor_interval_s: float,
    hang_no_progress_timeout_s: float,
) -> StageResult:
    remaining = (deadline - datetime.now(timezone.utc)).total_seconds()
    if remaining <= 0:
        return StageResult(124, "deadline_reached_before_launch")
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        process_group = process.pid
        previous_handlers: dict[int, Any] = {}

        def interrupt(signum, _frame):
            raise CampaignInterrupted(f"received signal {signum}")

        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, interrupt)
            last_snapshot = _process_group_snapshot(process_group, log_path)
            last_progress = time.monotonic()
            while True:
                return_code = process.poll()
                if return_code is not None:
                    if _process_group_exists(process_group):
                        _terminate_and_reap(process, process_group)
                        return StageResult(
                            return_code if return_code != 0 else 125,
                            "descendant_processes_remained_after_parent_exit",
                        )
                    return StageResult(return_code)
                now_utc = datetime.now(timezone.utc)
                if now_utc >= deadline:
                    _terminate_and_reap(process, process_group)
                    return StageResult(124, "utc_deadline_reached")
                try:
                    monitor()
                except BaseException as error:
                    _terminate_and_reap(process, process_group)
                    if isinstance(error, (KeyboardInterrupt, CampaignInterrupted)):
                        raise
                    return StageResult(125, f"monitor_failed: {error}")
                snapshot = _process_group_snapshot(process_group, log_path)
                if snapshot[:2] != last_snapshot[:2]:
                    last_progress = time.monotonic()
                    last_snapshot = snapshot
                elif (
                    time.monotonic() - last_progress >= hang_no_progress_timeout_s
                    and snapshot[2]
                    and all(state in ("T", "t") for state in snapshot[2])
                ):
                    _terminate_and_reap(process, process_group)
                    return StageResult(125, "confirmed_stopped_process_group_hang")
                sleep_s = min(
                    monitor_interval_s,
                    max(0.01, (deadline - now_utc).total_seconds()),
                )
                time.sleep(sleep_s)
        except BaseException as error:
            try:
                _terminate_and_reap(process, process_group)
            except BaseException as cleanup_error:
                if hasattr(error, "add_note"):
                    error.add_note(f"process-group cleanup also failed: {cleanup_error}")
            raise
        finally:
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)


def _check_process_text(text: str, path: Path) -> None:
    match = NONFINITE_TOKEN.search(text)
    if match is not None:
        _fail(f"non-finite numeric token {match.group(0)!r} found in process log {path}")
    oom = OOM_TOKEN.search(text)
    if oom is not None:
        _fail(f"out-of-memory evidence {oom.group(0)!r} found in process log {path}")


def _check_process_log(path: Path) -> None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        raise CampaignError(f"unable to inspect process log {path}: {error}") from error
    _check_process_text(text, path)


def _run_stage(
    contract: CampaignContract,
    *,
    stage: str,
    command: Sequence[str],
    runner: Callable[
        [Sequence[str], Path, Path, datetime, Callable[[], None], float, float],
        StageResult,
    ],
    disk_free: Callable[[Path], int],
) -> None:
    if datetime.now(timezone.utc) >= contract.deadline_utc:
        _fail(f"UTC deadline reached before stage {stage}")
    _check_disk(contract, disk_free)
    _verify_frozen(contract)
    log_path = contract.output_dir / "logs" / f"{stage}.log"
    _event(contract.output_dir, {"stage": stage, "status": "started", "command": list(command)})
    result: StageResult | None = None
    log_offset = 0
    log_carry = ""

    def monitor_stage() -> None:
        nonlocal log_offset, log_carry
        _check_disk(contract, disk_free)
        _verify_frozen(contract)
        if not log_path.exists():
            return
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(log_offset)
            chunk = handle.read()
            log_offset = handle.tell()
        if chunk:
            _check_process_text(log_carry + chunk, log_path)
            log_carry = (log_carry + chunk)[-64:]
    try:
        result = runner(
            command,
            contract.project_root,
            log_path,
            contract.deadline_utc,
            monitor_stage,
            contract.monitor_interval_s,
            contract.hang_no_progress_timeout_s,
        )
        _check_process_log(log_path)
        if result.return_code != 0:
            _fail(
                f"stage {stage} failed with return code {result.return_code} "
                f"({result.failure_reason}); partial output preserved"
            )
        if not result.process_group_cleanup_confirmed:
            _fail(f"stage {stage} did not confirm process-group cleanup")
        _verify_frozen(contract)
    except BaseException as error:
        try:
            _event(
                contract.output_dir,
                {
                    "stage": stage,
                    "status": "failed",
                    "return_code": result.return_code if result is not None else None,
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        except BaseException:
            pass
        raise
    _event(
        contract.output_dir,
        {
            "stage": stage,
            "status": "completed",
            "return_code": 0,
            "process_group_cleanup_confirmed": result.process_group_cleanup_confirmed,
        },
    )


def _single_run_dir(contract: CampaignContract, seed: int) -> Path:
    root = contract.output_dir / "runs" / str(seed) / "rj45-v1"
    directories = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    if len(directories) != 1:
        _fail(f"seed {seed} training must create exactly one run directory")
    return directories[0]


def _load_local_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"unable to load validation helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_training_run(contract: CampaignContract, seed: int, run_dir: Path) -> None:
    runner_path = run_dir / "runner_config.json"
    provenance_path = run_dir / "training_provenance.json"
    manifest_path = run_dir / "checkpoint_manifest.json"
    runner = _load_json(runner_path, "training runner config")
    provenance = _load_json(provenance_path, "training provenance")
    manifest = _load_json(manifest_path, "checkpoint manifest")
    if runner != _load_json(contract.seed_configs[seed].path, "frozen seed PPO config"):
        _fail(f"seed {seed} runner config differs from frozen seed config")
    if provenance.get("schema_version") != 1 or provenance.get("seed") != seed:
        _fail(f"seed {seed} training provenance schema or seed is invalid")
    if provenance.get("num_envs") != contract.num_envs:
        _fail(f"seed {seed} training provenance num_envs differs from plan")
    if provenance.get("checkpoint_scope") != "checkpoints created in this run directory":
        _fail(f"seed {seed} training provenance checkpoint scope is invalid")
    runtime = _object(provenance.get("runtime_versions"), "training runtime versions")
    if runtime.get("rsl-rl-lib") != "3.0.1":
        _fail(f"seed {seed} training requires rsl-rl-lib 3.0.1")
    lineage = _object(
        provenance.get("source_and_config_sha256"), "training source/config lineage"
    )
    expected_sources = {
        _relative(contract.project_root, path): digest
        for path, digest in contract.frozen_inputs.items()
        if path.parent == contract.project_root / "src/rby1_assembly"
        and path.suffix == ".py"
    }
    expected_sources["scripts/train.py"] = contract.frozen_inputs[
        contract.project_root / "scripts/train.py"
    ]
    for key, digest in expected_sources.items():
        if lineage.get(key) != digest:
            _fail(f"seed {seed} training lineage differs for {key}")
    seed_key = _relative(contract.project_root, contract.seed_configs[seed].path)
    if lineage.get(seed_key) != contract.seed_configs[seed].sha256:
        _fail(f"seed {seed} training lineage omits its frozen PPO config")
    training_key = _relative(contract.project_root, contract.training_config.path)
    if lineage.get(training_key) != contract.training_config.sha256:
        _fail(f"seed {seed} training lineage omits selected training config")
    if manifest.get("schema_version") != 1 or manifest.get("training_completed") is not True:
        _fail(f"seed {seed} checkpoint manifest is not completed schema 1")
    if manifest.get("training_provenance_sha256") != _sha_file(provenance_path):
        _fail(f"seed {seed} checkpoint manifest provenance SHA differs")
    checkpoint_hashes = _object(manifest.get("checkpoints"), "checkpoint manifest checkpoints")
    for checkpoint in contract.checkpoints:
        checkpoint_path = run_dir / checkpoint.filename
        if checkpoint_hashes.get(checkpoint.filename) != _sha_file(checkpoint_path):
            _fail(f"seed {seed} manifest does not pin {checkpoint.filename}")


def _validate_inspection(
    contract: CampaignContract,
    run_dir: Path,
    path: Path,
    checkpoint: CheckpointSpec,
    checkpoint_path: Path,
) -> None:
    payload = _load_json(path, "checkpoint inspection")
    if payload.get("status") != "checkpoint_payload_and_lineage_verified":
        _fail(f"checkpoint inspection did not verify payload: {path}")
    details = _object(payload.get("checkpoint"), "checkpoint inspection checkpoint")
    if (
        details.get("filename_iteration") != checkpoint.iteration_index
        or details.get("payload_iteration") != checkpoint.iteration_index
        or details.get("completed_updates") != checkpoint.completed_updates
        or details.get("sha256") != _sha_file(checkpoint_path)
    ):
        _fail(f"checkpoint inspection differs from preregistered checkpoint: {path}")
    if payload.get("planned_training_updates") != 200:
        _fail(f"checkpoint inspection planned updates differ from 200: {path}")
    helper = _load_local_module(
        contract.project_root / "scripts/inspect_rj45_checkpoint.py",
        "_rj45_checkpoint_inspector_for_campaign",
    )
    recomputed = helper.inspect_checkpoint(
        checkpoint_path=checkpoint_path,
        manifest_path=run_dir / "checkpoint_manifest.json",
        training_provenance_path=run_dir / "training_provenance.json",
        runner_config_path=run_dir / "runner_config.json",
        training_script_path=contract.project_root / "scripts/train.py",
    )
    if payload != recomputed:
        _fail(f"checkpoint inspection JSON differs from helper recomputation: {path}")


def _validate_evaluation_report(
    path: Path, contract: CampaignContract, seed: int, run_dir: Path, checkpoint_path: Path
) -> None:
    report = _load_json(path, "development evaluation report")
    if report.get("rj45_evaluation_schema_version") != 2:
        _fail(f"evaluation report schema is not 2: {path}")
    if report.get("baseline_mode") != "learned":
        _fail(f"evaluation report is not learned mode: {path}")
    if report.get("seed") != contract.evaluation_seed:
        _fail(f"evaluation report used the wrong development seed: {path}")
    if report.get("requested_attempts") != contract.attempts_per_policy:
        _fail(f"evaluation report requested attempts differ from plan: {path}")
    if report.get("evaluation_num_envs") != contract.evaluation_num_envs:
        _fail(f"evaluation report environment count differs from plan: {path}")
    if report.get("attempts") != contract.attempts_per_policy:
        _fail(f"evaluation report is incomplete: {path}")
    provenance = _object(report.get("provenance"), "evaluation report provenance")
    if provenance.get("checkpoint_sha256") != _sha_file(checkpoint_path):
        _fail(f"evaluation report checkpoint SHA differs: {path}")
    if provenance.get("input_immutability_verified") is not True:
        _fail(f"evaluation report did not verify input immutability: {path}")
    if provenance.get("evaluation_inputs_reverified_at_end") is not True:
        _fail(f"evaluation report did not reverify RJ45 inputs: {path}")
    if provenance.get("policy_config_sha256") != contract.seed_configs[seed].sha256:
        _fail(f"evaluation report policy config SHA differs: {path}")
    if _object(report.get("rj45_training_config"), "evaluation training config").get(
        "sha256"
    ) != contract.training_config.sha256:
        _fail(f"evaluation report training config SHA differs: {path}")
    evidence = _object(
        provenance.get("verified_checkpoint_evidence"),
        "evaluation verified checkpoint evidence",
    )
    if evidence.get("training_provenance_sha256") != _sha_file(
        run_dir / "training_provenance.json"
    ):
        _fail(f"evaluation report training provenance SHA differs: {path}")
    if evidence.get("checkpoint_manifest_sha256") != _sha_file(
        run_dir / "checkpoint_manifest.json"
    ):
        _fail(f"evaluation report checkpoint manifest SHA differs: {path}")
    runtime = _object(provenance.get("runtime_versions"), "evaluation runtime versions")
    if runtime.get("rsl_rl") != "3.0.1":
        _fail(f"evaluation report requires rsl_rl 3.0.1: {path}")
    source = _object(provenance.get("source_sha256"), "evaluation source hashes")
    expected_source = {
        _relative(contract.project_root, source_path): digest
        for source_path, digest in contract.frozen_inputs.items()
        if source_path.parent == contract.project_root / "src/rby1_assembly"
        and source_path.suffix == ".py"
    }
    expected_source["scripts/evaluate_policy.py"] = contract.frozen_inputs[
        contract.project_root / "scripts/evaluate_policy.py"
    ]
    if source != expected_source:
        _fail(f"evaluation report source snapshot differs from frozen sources: {path}")
    analyzer = _load_local_module(
        contract.project_root / "scripts/analyze_rj45_paired.py",
        "_rj45_analyzer_for_campaign",
    )
    try:
        analyzer._validate_report(report, str(path), contract.attempts_per_policy)
    except Exception as error:
        raise CampaignError(f"evaluation report failed analyzer validation: {path}: {error}") from error


def execute_campaign(
    contract: CampaignContract,
    *,
    runner: Callable[
        [Sequence[str], Path, Path, datetime, Callable[[], None], float, float],
        StageResult,
    ] = run_subprocess,
    disk_free: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
) -> list[Path]:
    reports: list[Path] = []
    status_path = contract.output_dir / "campaign-status.json"
    try:
        contract.output_dir.mkdir(parents=False, exist_ok=False)
        for name in ("logs", "reports", "inspections", "runs"):
            (contract.output_dir / name).mkdir()
        _atomic_json(status_path, {
            "schema_version": 1,
            "status": "running",
            "profile_label": contract.profile_label,
            "deadline_utc": contract.deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "completed_reports": 0,
            "final_holdout_opened": False,
        })
        _atomic_json(contract.output_dir / "campaign-contract.json", {
        "schema_version": 1,
        "plan": {"path": str(contract.plan.path), "sha256": contract.plan.sha256},
        "selection": {
            "path": str(contract.selection.path),
            "sha256": contract.selection.sha256,
        },
        "candidate_selection": {
            "path": str(contract.candidate_selection.path),
            "sha256": contract.candidate_selection.sha256,
        },
        "input_manifest": {
            "path": str(contract.input_manifest.path),
            "sha256": contract.input_manifest.sha256,
        },
        "guard_record": {
            "path": str(contract.guard_record.path),
            "sha256": contract.guard_record.sha256,
            "verification_scope": "local_integrity_only_not_provider_authenticated",
        },
        "profile_label": contract.profile_label,
        "guard_started_utc": contract.guard_started_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guard_deadline_utc": contract.guard_deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "timer_target": contract.timer_target,
        "brev_instance": {
            "id": contract.instance_id,
            "name": contract.instance_name,
            "organization": contract.instance_organization,
        },
        "training_seeds": list(contract.training_seeds),
        "seed_ppo_configs": {
            str(seed): {"path": str(reference.path), "sha256": reference.sha256}
            for seed, reference in contract.seed_configs.items()
        },
        "assessed_checkpoints": [
            {
                "filename": checkpoint.filename,
                "iteration_index": checkpoint.iteration_index,
                "completed_updates": checkpoint.completed_updates,
            }
            for checkpoint in contract.checkpoints
        ],
        "development_evaluation_seed": contract.evaluation_seed,
        "validation_boundary": {
            "checkpoint": "inspector helper reloaded payload bytes and bundle lineage",
            "evaluation": (
                "frozen paired analyzer revalidated finite episode outcomes, invalid "
                "aggregates, forbidden-contact terminal values, and provenance hashes"
            ),
            "unverified": (
                "force, relative-velocity, and dwell time histories are not independently "
                "recomputed by this campaign runner"
            ),
            "hang_detection": (
                "only an owned process group whose members are all OS-stopped (T/t) "
                "is classified as a confirmed hang; other quiet processes run until "
                "deadline or an independently verified failure"
            ),
        },
        "deadline_utc": contract.deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "final_holdout_opened": False,
        })
    except BaseException as error:
        try:
            if contract.output_dir.is_dir():
                _atomic_json(contract.output_dir / "startup-failure.json", {
                    "schema_version": 1,
                    "status": "failed_during_output_initialization",
                    "error": {"type": type(error).__name__, "message": str(error)},
                    "final_holdout_opened": False,
                })
        except BaseException:
            pass
        raise
    try:
        for seed in contract.training_seeds:
            train_stage = f"train-seed-{seed}"
            _run_stage(
                contract,
                stage=train_stage,
                command=build_train_command(contract, seed),
                runner=runner,
                disk_free=disk_free,
            )
            run_dir = _single_run_dir(contract, seed)
            _validate_training_run(contract, seed, run_dir)
            for checkpoint in contract.checkpoints:
                checkpoint_path = run_dir / checkpoint.filename
                if not checkpoint_path.is_file():
                    _fail(f"training did not produce {checkpoint_path}")
                inspection = (
                    contract.output_dir
                    / "inspections"
                    / f"seed-{seed}-{checkpoint.filename.removesuffix('.pt')}.json"
                )
                inspect_stage = f"inspect-seed-{seed}-iter-{checkpoint.iteration_index}"
                _run_stage(
                    contract,
                    stage=inspect_stage,
                    command=build_inspection_command(contract, run_dir, checkpoint, inspection),
                    runner=runner,
                    disk_free=disk_free,
                )
                _validate_inspection(
                    contract, run_dir, inspection, checkpoint, checkpoint_path
                )
                report = (
                    contract.output_dir
                    / "reports"
                    / f"seed-{seed}-iter-{checkpoint.iteration_index}.json"
                )
                eval_stage = f"evaluate-seed-{seed}-iter-{checkpoint.iteration_index}"
                _run_stage(
                    contract,
                    stage=eval_stage,
                    command=build_evaluation_command(contract, seed, checkpoint_path, report),
                    runner=runner,
                    disk_free=disk_free,
                )
                _validate_evaluation_report(
                    report, contract, seed, run_dir, checkpoint_path
                )
                reports.append(report)
                _atomic_json(status_path, {
                    "schema_version": 1,
                    "status": "running",
                    "profile_label": contract.profile_label,
                    "deadline_utc": contract.deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "completed_reports": len(reports),
                    "final_holdout_opened": False,
                })
        if len(reports) != 9:
            _fail("campaign did not produce exactly nine development reports")
        _verify_frozen(contract)
        _atomic_json(status_path, {
            "schema_version": 1,
            "status": "completed",
            "profile_label": contract.profile_label,
            "deadline_utc": contract.deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "completed_reports": 9,
            "reports": [str(path) for path in reports],
            "final_holdout_opened": False,
        })
        _event(contract.output_dir, {"stage": "campaign", "status": "completed"})
        return reports
    except BaseException as error:
        try:
            _atomic_json(status_path, {
                "schema_version": 1,
                "status": "failed",
                "profile_label": contract.profile_label,
                "deadline_utc": contract.deadline_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "completed_reports": len(reports),
                "error": {"type": type(error).__name__, "message": str(error)},
                "final_holdout_opened": False,
            })
            _event(
                contract.output_dir,
                {
                    "stage": "campaign",
                    "status": "failed",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
            )
        except BaseException:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument(
        "--launch-manifest", "--selection", dest="selection", required=True, type=Path
    )
    parser.add_argument("--expected-instance-id", required=True)
    parser.add_argument("--expected-instance-name", required=True)
    parser.add_argument("--expected-instance-organization", required=True)
    parser.add_argument(
        "--launch-manifest-sha256",
        "--selection-sha256",
        dest="selection_sha256",
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args(argv)
    try:
        contract = load_contract(
            project_root=args.project_root,
            plan_path=args.plan,
            expected_plan_sha256=args.plan_sha256,
            selection_path=args.selection,
            expected_selection_sha256=args.selection_sha256,
            expected_instance_id=args.expected_instance_id,
            expected_instance_name=args.expected_instance_name,
            expected_instance_organization=args.expected_instance_organization,
            output_dir=args.output_dir,
            deadline_utc=args.deadline_utc,
            python=args.python,
        )
        reports = execute_campaign(contract)
    except (CampaignInterrupted, KeyboardInterrupt) as error:
        print(f"campaign interrupted: {error}", file=sys.stderr)
        return 130
    except CampaignError as error:
        print(f"campaign failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({
        "status": "completed",
        "reports": [str(path) for path in reports],
        "final_holdout_opened": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
