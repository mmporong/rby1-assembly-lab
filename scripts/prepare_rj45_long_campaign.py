#!/usr/bin/env python3
"""Prepare immutable inputs for the preregistered RJ45 long campaign."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, NoReturn

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GROUPS = ("A0", "A1", "A2", "E1", "Z1")


class PreparationError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise PreparationError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _sha_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise PreparationError(f"unable to hash {path}: {error}") from error


def _load_json(path: Path, name: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        _fail(f"{name} contains non-finite constant {value}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
        json.dumps(value, allow_nan=False)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise PreparationError(f"invalid {name} JSON {path}: {error}") from error
    return _object(value, name)


def _project_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        _fail(f"{name} must be a project-relative path")
    relative = Path(value)
    if ".." in relative.parts:
        _fail(f"{name} must not contain '..'")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        _fail(f"{name} escapes project root")
    return path


def _relative(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        _fail(f"path is outside project root: {path}")
    return str(resolved.relative_to(root))


def _ref(root: Path, value: Any, name: str) -> tuple[Path, str]:
    item = _object(value, name)
    return _project_path(root, item.get("path"), f"{name}.path"), _sha(
        item.get("sha256"), f"{name}.sha256"
    )


def _verify(path: Path, digest: str, name: str) -> None:
    if _sha_file(path) != digest:
        _fail(f"{name} SHA-256 mismatch: {path}")


def _add(frozen: dict[Path, str], path: Path, digest: str, name: str) -> None:
    _verify(path, digest, name)
    previous = frozen.get(path)
    if previous is not None and previous != digest:
        _fail(f"conflicting SHA-256 for duplicate input {path}")
    frozen[path] = digest


def _parse_utc(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value
    ):
        _fail(f"{name} must use YYYY-MM-DDTHH:MM:SSZ")
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as error:
        raise PreparationError(f"invalid {name}: {error}") from error
    return value


def _nested_object(document: dict[str, Any], path: tuple[str, ...], name: str) -> dict[str, Any]:
    value: Any = document
    for key in path:
        value = _object(value, name).get(key)
    return _object(value, name)


def _declared_dependencies(root: Path, document: dict[str, Any], name: str) -> dict[Path, str]:
    result: dict[Path, str] = {}

    def walk(value: Any, location: str) -> None:
        if isinstance(value, dict):
            for map_key in ("source_and_config_sha256", "artifact_sha256"):
                dependency_map = value.get(map_key)
                if isinstance(dependency_map, dict):
                    for raw_path, digest in dependency_map.items():
                        path = _project_path(root, raw_path, f"{location}.{map_key} path")
                        _add(result, path, _sha(digest, f"{location}.{raw_path}"), location)
            for key, raw_path in value.items():
                if isinstance(raw_path, str) and key.endswith("_path"):
                    sha_key = f"{key[:-5]}_sha256"
                    if sha_key in value:
                        path = _project_path(root, raw_path, f"{location}.{key}")
                        _add(result, path, _sha(value[sha_key], f"{location}.{sha_key}"), location)
                walk(raw_path, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                walk(item, f"{location}[{index}]")

    walk(document, name)
    return result


def _canonical_json(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PreparationError(f"output already exists: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_or_match_anchor(path: Path, payload: bytes) -> None:
    try:
        _write_new(path, payload)
    except PreparationError as error:
        try:
            existing = path.read_bytes()
        except OSError:
            raise error
        if existing != payload:
            _fail(f"guard anchor already exists with different bytes: {path}")


def prepare(
    *,
    project_root: Path,
    candidate_path: Path,
    candidate_sha256: str,
    long_plan_path: Path,
    long_plan_sha256: str,
    runner_path: Path,
    runner_sha256: str,
    inspector_path: Path,
    inspector_sha256: str,
    training_script_path: Path,
    training_script_sha256: str,
    analyzer_path: Path,
    analyzer_sha256: str,
    guard_record_path: Path,
    guard_record_sha256: str,
    nominal_path: Path,
    nominal_sha256: str,
    held_path: Path,
    held_sha256: str,
    contact_path: Path,
    contact_sha256: str,
    guard_started_utc: str,
    guard_deadline_utc: str,
    timer_target: str,
    instance_id: str,
    instance_name: str,
    instance_organization: str,
    output_dir: Path,
    poll_interval_s: float,
    hang_timeout_s: float,
) -> dict[str, Path]:
    root = project_root.resolve()
    output_dir = output_dir.resolve()
    if not root.is_dir():
        _fail("project root must exist")
    if output_dir.exists():
        _fail(f"output directory already exists: {output_dir}")
    if not output_dir.is_relative_to(root / "artifacts"):
        _fail("prepared campaign files must be placed below artifacts/")
    if not output_dir.parent.is_dir():
        _fail("output directory parent must already exist")
    guard_started_utc = _parse_utc(guard_started_utc, "guard_started_utc")
    guard_deadline_utc = _parse_utc(guard_deadline_utc, "guard_deadline_utc")
    identity = {"id": instance_id, "name": instance_name, "organization": instance_organization}
    if any(not isinstance(value, str) or not value for value in identity.values()):
        _fail("Brev identity fields must be non-empty")
    if not isinstance(timer_target, str) or not timer_target:
        _fail("timer target must be non-empty")
    if (
        isinstance(poll_interval_s, bool)
        or not isinstance(poll_interval_s, (int, float))
        or not math.isfinite(float(poll_interval_s))
        or float(poll_interval_s) < 0.01
    ):
        _fail("poll interval must be finite and at least 0.01 seconds")
    if (
        isinstance(hang_timeout_s, bool)
        or not isinstance(hang_timeout_s, (int, float))
        or not math.isfinite(float(hang_timeout_s))
        or float(hang_timeout_s) < float(poll_interval_s)
    ):
        _fail("hang timeout must be finite and no shorter than poll interval")

    supplied = {
        "candidate selection": (candidate_path.resolve(), _sha(candidate_sha256, "candidate SHA")),
        "long-run plan": (long_plan_path.resolve(), _sha(long_plan_sha256, "long plan SHA")),
        "campaign runner": (runner_path.resolve(), _sha(runner_sha256, "runner SHA")),
        "checkpoint inspector": (inspector_path.resolve(), _sha(inspector_sha256, "inspector SHA")),
        "training script": (training_script_path.resolve(), _sha(training_script_sha256, "training script SHA")),
        "frozen analyzer": (analyzer_path.resolve(), _sha(analyzer_sha256, "analyzer SHA")),
        "guard record": (guard_record_path.resolve(), _sha(guard_record_sha256, "guard record SHA")),
        "nominal evidence": (nominal_path.resolve(), _sha(nominal_sha256, "nominal SHA")),
        "held state": (held_path.resolve(), _sha(held_sha256, "held SHA")),
        "parallel contact evidence": (contact_path.resolve(), _sha(contact_sha256, "contact SHA")),
    }
    expected_scripts = {
        "campaign runner": root / "scripts/run_rj45_long_campaign.py",
        "checkpoint inspector": root / "scripts/inspect_rj45_checkpoint.py",
        "training script": root / "scripts/train.py",
        "frozen analyzer": root / "scripts/analyze_rj45_paired.py",
    }
    for label, expected_path in expected_scripts.items():
        if supplied[label][0] != expected_path.resolve():
            _fail(f"{label} must use the repository's fixed path: {expected_path}")
    frozen: dict[Path, str] = {}
    for label, (path, digest) in supplied.items():
        if not path.is_relative_to(root):
            _fail(f"{label} must be inside project root")
        _add(frozen, path, digest, label)

    candidate = _load_json(candidate_path, "candidate selection")
    plan = _load_json(long_plan_path, "long-run plan")
    if candidate.get("schema_version") != 2 or plan.get("schema_version") != 1:
        _fail("candidate schema must be 2 and long plan schema must be 1")
    inputs = _object(candidate.get("inputs"), "candidate.inputs")
    candidate_plan_path, candidate_plan_sha = _ref(root, inputs.get("long_run_plan"), "candidate long plan")
    if (candidate_plan_path, candidate_plan_sha) != supplied["long-run plan"]:
        _fail("candidate references a different long-run plan")
    guard_record = _load_json(guard_record_path, "guard record")
    if guard_record.get("schema_version") != 1:
        _fail("guard record schema_version must be 1")
    if guard_record.get("candidate_selection_sha256") != candidate_sha256:
        _fail("guard record candidate SHA differs from supplied candidate")
    if guard_record.get("long_run_plan_sha256") != long_plan_sha256:
        _fail("guard record plan SHA differs from supplied plan")
    if guard_record.get("guard_started_utc") != guard_started_utc:
        _fail("guard record start differs from supplied guard start")
    if guard_record.get("guard_deadline_utc") != guard_deadline_utc:
        _fail("guard record deadline differs from supplied guard deadline")
    if guard_record.get("brev_instance") != identity:
        _fail("guard record Brev identity differs from supplied identity")
    if guard_record.get("timer_target") != timer_target:
        _fail("guard record timer target differs from supplied timer target")
    if guard_record.get("observation_scope") != "operator_recorded_not_provider_authenticated":
        _fail("guard record must declare its non-authenticated observation boundary")
    guard_start = datetime.strptime(guard_started_utc, "%Y-%m-%dT%H:%M:%SZ")
    guard_deadline = datetime.strptime(guard_deadline_utc, "%Y-%m-%dT%H:%M:%SZ")
    guard_max_hours = _object(plan.get("budget"), "plan.budget").get("bounded_guard_max_hours_from_follow_up_start")
    if not isinstance(guard_max_hours, (int, float)) or isinstance(guard_max_hours, bool) or not math.isfinite(float(guard_max_hours)):
        _fail("plan guard maximum must be finite")
    if guard_deadline <= guard_start or (guard_deadline - guard_start).total_seconds() > float(guard_max_hours) * 3600:
        _fail("guard record deadline exceeds the plan's fixed guard window")
    parent_path, parent_sha = _ref(root, inputs.get("parent_pilot_plan"), "candidate parent plan")
    if parent_sha != _sha(plan.get("parent_pilot_plan_sha256"), "plan parent SHA"):
        _fail("candidate parent plan differs from long plan")
    _add(frozen, parent_path, parent_sha, "parent pilot plan")
    parent = _load_json(parent_path, "parent pilot plan")
    parent_frozen = _object(_object(parent.get("frozen_inputs"), "parent frozen inputs").get("sha256"), "parent frozen SHA map")
    if len(parent_frozen) != 57:
        _fail("parent pilot plan must contain exactly 57 frozen inputs")
    for raw_path, raw_digest in parent_frozen.items():
        path = _project_path(root, raw_path, "parent frozen input path")
        _add(frozen, path, _sha(raw_digest, f"parent frozen input {raw_path}"), "parent frozen input")
    current_source_paths = set((root / "src").rglob("*.py"))
    frozen_source_paths = {
        _project_path(root, raw_path, "parent frozen source")
        for raw_path in parent_frozen
        if raw_path.startswith("src/") and raw_path.endswith(".py")
    }
    if not current_source_paths or current_source_paths != frozen_source_paths:
        _fail("parent frozen source path set differs from current src/**/*.py")
    current_config_paths = set((root / "configs").rglob("*.json"))
    frozen_config_paths = {
        _project_path(root, raw_path, "parent frozen config")
        for raw_path in parent_frozen
        if raw_path.startswith("configs/") and raw_path.endswith(".json")
    }
    if not current_config_paths or current_config_paths != frozen_config_paths:
        _fail("parent frozen config path set differs from current configs/**/*.json")

    audit_path, audit_sha = _ref(root, inputs.get("stored_paired_audit"), "candidate stored audit")
    _add(frozen, audit_path, audit_sha, "stored paired audit")
    audit = _load_json(audit_path, "stored paired audit")
    if audit.get("schema_version") != 1:
        _fail("stored paired audit must use frozen schema_version 1")
    audit_reports = _object(audit.get("input_reports"), "audit input_reports")
    audit_bundles = _object(audit.get("training_artifact_verification"), "audit bundles")
    if set(audit_reports) != {"baseline", *GROUPS}:
        _fail("stored paired audit must cover baseline and all five groups")
    if set(audit_bundles) != set(GROUPS):
        _fail("stored paired audit must cover all five training bundles")

    candidate_analyzer_path, candidate_analyzer_sha = _ref(root, inputs.get("frozen_analyzer"), "candidate analyzer")
    if (candidate_analyzer_path, candidate_analyzer_sha) != supplied["frozen analyzer"]:
        _fail("candidate frozen analyzer differs from supplied analyzer")
    verification = _object(candidate.get("verification_scope"), "candidate verification_scope")
    if verification.get("status") != "raw_reports_and_training_bundles_rehashed_and_fresh_analyzer_replayed":
        _fail("candidate is not a verified complete pilot selection")
    candidate_evidence = _object(candidate.get("candidate_evidence"), "candidate evidence")
    if set(candidate_evidence) != set(GROUPS):
        _fail("candidate evidence must cover all five groups")

    expected_raw: dict[str, str] = {}
    pilot_reports: dict[str, dict[str, Any]] = {}
    def add_expected_raw(path: Path, digest: str, label: str) -> None:
        relative = _relative(root, path)
        previous = expected_raw.get(relative)
        if previous is not None:
            _fail(f"duplicate raw evidence path in stored audit: {relative} ({label})")
        expected_raw[relative] = digest

    for label, report_value in audit_reports.items():
        report_path, report_sha = _ref(root, report_value, f"audit report {label}")
        add_expected_raw(report_path, report_sha, f"report {label}")
        pilot_reports[label] = _load_json(report_path, f"pilot report {label}")
    for group_name in GROUPS:
        audit_bundle = _object(audit_bundles[group_name], f"audit bundle {group_name}")
        evidence = _object(candidate_evidence[group_name], f"candidate evidence {group_name}")
        evidence_report = _ref(root, evidence.get("raw_report"), f"candidate report {group_name}")
        audit_report = _ref(root, audit_reports[group_name], f"audit report {group_name}")
        if evidence_report != audit_report:
            _fail(f"candidate report differs from stored audit for {group_name}")
        evidence_bundle = _object(evidence.get("checkpoint_bundle"), f"candidate bundle {group_name}")
        for field in ("checkpoint", "checkpoint_manifest", "training_provenance", "runner_config"):
            audit_ref = _ref(root, audit_bundle.get(field), f"audit bundle {group_name}.{field}")
            evidence_ref = _ref(root, evidence_bundle.get(field), f"candidate bundle {group_name}.{field}")
            if evidence_ref != audit_ref:
                _fail(f"candidate bundle differs from stored audit for {group_name}.{field}")
            add_expected_raw(audit_ref[0], audit_ref[1], f"{group_name}.{field}")

    raw_evidence = _object(inputs.get("raw_evidence_sha256"), "candidate.inputs.raw_evidence_sha256")
    if raw_evidence != expected_raw:
        _fail("candidate raw evidence map differs from stored audit reports and bundles")
    for map_name in ("raw_evidence_sha256", "immutable_inputs_sha256"):
        mapping = _object(inputs.get(map_name), f"candidate.inputs.{map_name}")
        for raw_path, raw_digest in mapping.items():
            path = _project_path(root, raw_path, f"{map_name} path")
            digest = _sha(raw_digest, f"{map_name}.{raw_path}")
            if map_name == "immutable_inputs_sha256" and parent_frozen.get(raw_path) != digest:
                _fail(f"candidate immutable input differs from parent frozen SHA: {raw_path}")
            _add(frozen, path, digest, map_name)

    current_sources = set((root / "src/rby1_assembly").glob("*.py"))
    candidate_sources = {
        _project_path(root, path, "candidate source")
        for path in _object(inputs.get("immutable_inputs_sha256"), "candidate immutable inputs")
        if path.startswith("src/rby1_assembly/") and path.endswith(".py")
    }
    if not current_sources or current_sources != candidate_sources:
        _fail("candidate source path set differs from current src/rby1_assembly/*.py")

    decision = _object(candidate.get("decision"), "candidate.decision")
    selected_group = decision.get("selected_group")
    if not isinstance(selected_group, str) or selected_group not in GROUPS:
        _fail("candidate selected group is invalid")
    groups = {_object(item, "parent group").get("id"): item for item in parent.get("groups", [])}
    if set(groups) != set(GROUPS):
        _fail("parent plan group set is invalid")
    selected = _object(candidate.get("selected_profile"), "candidate.selected_profile")
    selected_training_path, selected_training_sha = _ref(root, selected.get("training_config"), "selected training config")
    selected_ppo_path, selected_ppo_sha = _ref(root, selected.get("ppo_config"), "selected PPO config")
    zero_init = selected.get("zero_init_actor_output")
    group = _object(groups[selected_group], "selected parent group")
    expected_zero = "--zero-init-actor-output" in group.get("training_flags", [])
    if (
        selected_training_path != _project_path(root, group.get("training_config"), "parent training config")
        or selected_ppo_path != _project_path(root, group.get("ppo_config"), "parent PPO config")
        or zero_init is not expected_zero
    ):
        _fail("selected profile differs from parent group")
    for path, digest, label in (
        (selected_training_path, selected_training_sha, "selected training config"),
        (selected_ppo_path, selected_ppo_sha, "selected PPO config"),
    ):
        relative = _relative(root, path)
        if parent_frozen.get(relative) != digest:
            _fail(f"{label} differs from parent frozen SHA")
        if frozen.get(path) != digest:
            _fail(f"candidate immutable inputs omit {label}")

    runtime_documents: dict[str, dict[str, Any]] = {}
    for label, path, digest in (
        ("nominal evidence", nominal_path.resolve(), nominal_sha256),
        ("held state", held_path.resolve(), held_sha256),
        ("parallel contact evidence", contact_path.resolve(), contact_sha256),
    ):
        document = _load_json(path, label)
        runtime_documents[label] = document
        for dependency_path, dependency_sha in _declared_dependencies(root, document, label).items():
            _add(frozen, dependency_path, dependency_sha, f"{label} dependency")

    expected_runtime: dict[str, str] | None = None
    for label, report in pilot_reports.items():
        provenance = _object(report.get("provenance"), f"pilot report {label}.provenance")
        nominal_claim = _object(provenance.get("rj45_nominal_evidence"), f"pilot report {label} nominal claim")
        held_claim = _object(provenance.get("held_grasp_reset_proof"), f"pilot report {label} held claim")
        contact_claim = _object(report.get("parallel_contact_evidence"), f"pilot report {label} contact claim")
        claim = {
            "nominal_report": _sha(nominal_claim.get("report_sha256"), f"pilot report {label} nominal report SHA"),
            "nominal_trace": _sha(nominal_claim.get("trace_sha256"), f"pilot report {label} nominal trace SHA"),
            "held": _sha(held_claim.get("artifact_sha256"), f"pilot report {label} held SHA"),
            "held_nominal_report": _sha(held_claim.get("nominal_report_sha256"), f"pilot report {label} held nominal report SHA"),
            "held_nominal_trace": _sha(held_claim.get("nominal_trace_sha256"), f"pilot report {label} held nominal trace SHA"),
            "contact_report": _sha(contact_claim.get("report_sha256"), f"pilot report {label} contact report SHA"),
            "contact_trace": _sha(contact_claim.get("trace_sha256"), f"pilot report {label} contact trace SHA"),
            "contact_sat": _sha(contact_claim.get("sat_report_sha256"), f"pilot report {label} contact SAT SHA"),
        }
        if claim["held_nominal_report"] != claim["nominal_report"] or claim["held_nominal_trace"] != claim["nominal_trace"]:
            _fail(f"pilot report {label} held proof differs from its nominal evidence")
        if expected_runtime is None:
            expected_runtime = claim
        elif claim != expected_runtime:
            _fail(f"pilot report {label} runtime evidence differs from the other pilot reports")
    if expected_runtime is None:
        _fail("pilot reports contain no runtime evidence")
    nominal_document = runtime_documents["nominal evidence"]
    if nominal_sha256 != expected_runtime["nominal_report"]:
        _fail("supplied nominal report SHA differs from pilot reports")
    nominal_trace_path = _project_path(root, nominal_document.get("trace_path"), "nominal trace path")
    nominal_trace_sha = _sha(nominal_document.get("trace_sha256"), "nominal trace SHA")
    if nominal_trace_sha != expected_runtime["nominal_trace"] or frozen.get(nominal_trace_path) != nominal_trace_sha:
        _fail("supplied nominal trace differs from pilot reports")
    if held_sha256 != expected_runtime["held"]:
        _fail("supplied held artifact SHA differs from pilot reports")
    contact_document = runtime_documents["parallel contact evidence"]
    for field, expected_key in (("report", "contact_report"), ("trace", "contact_trace"), ("sat_report", "contact_sat")):
        dependency_path = _project_path(root, contact_document.get(f"{field}_path"), f"contact {field} path")
        dependency_sha = _sha(contact_document.get(f"{field}_sha256"), f"contact {field} SHA")
        if dependency_sha != expected_runtime[expected_key] or frozen.get(dependency_path) != dependency_sha:
            _fail(f"supplied contact {field} differs from pilot reports")

    training = _object(plan.get("training"), "plan.training")
    seeds = training.get("training_seeds")
    if not isinstance(seeds, list) or len(seeds) != 3 or len(set(seeds)) != 3 or any(type(seed) is not int for seed in seeds):
        _fail("long plan must declare exactly three integer seeds")
    if training.get("updates_per_seed") != 200 or training.get("checkpoint_save_interval_iterations") != 19:
        _fail("long plan must retain 200 updates and save interval 19")
    base_ppo = _load_json(selected_ppo_path, "selected PPO config")
    seed_payloads: dict[int, tuple[Path, bytes, str]] = {}
    for seed in seeds:
        payload = json.loads(json.dumps(base_ppo, allow_nan=False))
        payload["seed"] = seed
        payload["max_iterations"] = 200
        payload["save_interval"] = 19
        path = output_dir / f"ppo-seed-{seed}.json"
        content = _canonical_json(payload)
        seed_payloads[seed] = (path, content, hashlib.sha256(content).hexdigest())

    prospective = dict(frozen)
    for path, _content, digest in seed_payloads.values():
        prospective[path] = digest
    manifest_payload = {
        "schema_version": 1,
        "inputs": {
            _relative(root, path): digest
            for path, digest in sorted(prospective.items(), key=lambda item: _relative(root, item[0]))
        },
    }
    manifest_path = output_dir / "input-manifest.json"
    manifest_content = _canonical_json(manifest_payload)
    manifest_sha = hashlib.sha256(manifest_content).hexdigest()
    launch_payload = {
        "schema_version": 1,
        "status": "ready_for_long_campaign",
        "long_run_plan_sha256": long_plan_sha256,
        "guard_started_utc": guard_started_utc,
        "guard_deadline_utc": guard_deadline_utc,
        "guard_record": {"path": _relative(root, guard_record_path), "sha256": guard_record_sha256},
        "timer_target": timer_target,
        "guard_verification_boundary": "record_and_anchor_are_local_integrity_evidence;_control_plane_and_timer_must_be_observed_by_operator_immediately_before_launch",
        "brev_instance": identity,
        "candidate_selection": {"path": _relative(root, candidate_path), "sha256": candidate_sha256},
        "input_manifest": {"path": _relative(root, manifest_path), "sha256": manifest_sha},
        "selected_profile": {
            "training_config": {"path": _relative(root, selected_training_path), "sha256": selected_training_sha},
            "ppo_config": {"path": _relative(root, selected_ppo_path), "sha256": selected_ppo_sha},
            "zero_init_actor_output": zero_init,
        },
        "runtime_inputs": {
            "nominal_evidence": {"path": _relative(root, nominal_path), "sha256": nominal_sha256},
            "held_state": {"path": _relative(root, held_path), "sha256": held_sha256},
            "parallel_contact_evidence": {"path": _relative(root, contact_path), "sha256": contact_sha256},
        },
        "seed_ppo_configs": {
            str(seed): {"path": _relative(root, path), "sha256": digest}
            for seed, (path, _content, digest) in seed_payloads.items()
        },
        "process_monitoring": {
            "poll_interval_s": float(poll_interval_s),
            "hang_no_progress_timeout_s": float(hang_timeout_s),
            "confirmed_hang_scope": "only_all_owned_process_group_members_os_stopped_T_or_t",
        },
    }
    launch_path = output_dir / "launch-manifest.json"
    launch_content = _canonical_json(launch_payload)

    anchor_path = root / "artifacts/rj45-long-campaign-guards" / f"{candidate_sha256}-{long_plan_sha256}.anchor.json"
    anchor_payload = {
        "schema_version": 1,
        "candidate_selection_sha256": candidate_sha256,
        "long_run_plan_sha256": long_plan_sha256,
        "guard_record": {"path": _relative(root, guard_record_path), "sha256": guard_record_sha256},
        "guard_started_utc": guard_started_utc,
        "guard_deadline_utc": guard_deadline_utc,
        "brev_instance": identity,
        "timer_target": timer_target,
        "scope": "local_no_clobber_reissuance_anchor_not_provider_authenticated",
    }
    _write_or_match_anchor(anchor_path, _canonical_json(anchor_payload))

    output_dir.mkdir(parents=False, exist_ok=False)
    for path, content, _digest in seed_payloads.values():
        _write_new(path, content)
    _write_new(manifest_path, manifest_content)
    for path, digest in prospective.items():
        _verify(path, digest, "frozen prepared input")
    _verify(manifest_path, manifest_sha, "input manifest")
    _write_new(launch_path, launch_content)
    return {
        "output_dir": output_dir,
        "input_manifest": manifest_path,
        "launch_manifest": launch_path,
        "guard_anchor": anchor_path,
        **{f"seed_config_{seed}": path for seed, (path, _content, _digest) in seed_payloads.items()},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    for name in ("candidate-selection", "long-run-plan", "runner", "inspector", "training-script", "analyzer", "guard-record", "nominal", "held-state", "parallel-contact"):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--guard-started-utc", required=True)
    parser.add_argument("--guard-deadline-utc", required=True)
    parser.add_argument("--timer-target", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--instance-name", required=True)
    parser.add_argument("--instance-organization", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--poll-interval-s", type=float, default=30.0)
    parser.add_argument("--hang-timeout-s", type=float, default=900.0)
    args = parser.parse_args(argv)
    try:
        result = prepare(
            project_root=args.project_root,
            candidate_path=args.candidate_selection,
            candidate_sha256=args.candidate_selection_sha256,
            long_plan_path=args.long_run_plan,
            long_plan_sha256=args.long_run_plan_sha256,
            runner_path=args.runner,
            runner_sha256=args.runner_sha256,
            inspector_path=args.inspector,
            inspector_sha256=args.inspector_sha256,
            training_script_path=args.training_script,
            training_script_sha256=args.training_script_sha256,
            analyzer_path=args.analyzer,
            analyzer_sha256=args.analyzer_sha256,
            guard_record_path=args.guard_record,
            guard_record_sha256=args.guard_record_sha256,
            nominal_path=args.nominal,
            nominal_sha256=args.nominal_sha256,
            held_path=args.held_state,
            held_sha256=args.held_state_sha256,
            contact_path=args.parallel_contact,
            contact_sha256=args.parallel_contact_sha256,
            guard_started_utc=args.guard_started_utc,
            guard_deadline_utc=args.guard_deadline_utc,
            timer_target=args.timer_target,
            instance_id=args.instance_id,
            instance_name=args.instance_name,
            instance_organization=args.instance_organization,
            output_dir=args.output_dir,
            poll_interval_s=args.poll_interval_s,
            hang_timeout_s=args.hang_timeout_s,
        )
    except PreparationError as error:
        print(f"preparation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
