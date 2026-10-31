#!/usr/bin/env python3
"""Audit recovered multi-bias RJ45 evidence using only the standard library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence


EXPECTED_HELPER_SHA256 = "7035363ecbab9aa83196760a54b22ab00b2c163bb4a8b4151a825960291e0be5"
EXPECTED_SOURCE_SHA256 = {
    "runner": "132918ca01fa831bb6f703b4cc2aa1d2d9bd73085459a8964495b37e3ef177f1",
    "replay_helper": "46bc89c708afb4ebb664a488e734b121364c9505462200aeddebc73186410bcd",
    "g2_helper": "128a48fabee86b239bedc3db70e244b2a08d052dfc4221a86e4180360294fa42",
    "adapter": "3523687acf7b78ee212f7472329cbc7a3248ec03f451d433fb2d271d6dc42100",
    "ablation_helper": "e23d6dfc2f4e31bb8c54f119d3655d18d63d3df4e5eea2f00812ce72ce39af1b",
    "runtime_manifest": "d672417f55a61a1345dd442d067813fffdfd30274b547c8c103430866dd7b0f2",
}
EXPECTED_ORIGIN_PROTOCOL_SHA256 = "a1a00452c8c7a6151844e2b3bd56d9322a2baaa6f1c4b7e9c2bb925be07959ef"
PHASES = ("preflight", "train-single", "train-multi", "evaluate")
METHODS = ("single64", "multi64", "zero")
VALID_CLASSIFICATIONS = {"valid_success", "valid_failure"}
TRAIN_POOLS = {
    "train-single": ("anchor",),
    "train-multi": ("anchor", "train-02", "train-03", "train-04"),
}
UPDATES = 64
TRANSITIONS = 8192
CALIBRATION_OFFSET = 46123020
PREFLIGHT_CASES = (
    ("anchor", 20261128),
    ("train-02", 20271208),
    ("train-03", 20271216),
    ("train-04", 20271224),
)
EVALUATION_CASES = (("anchor", 20261128, "training-anchor"),) + tuple(
    (f"fresh-{index + 1:02d}", 20281208 + 8 * index, "policy-unseen")
    for index in range(8)
)
CRITERIA_KEYS = {
    "depth", "dwell", "forbidden_contact", "force", "grasp_live_and_proven",
    "independent_evaluator_terminal", "lateral", "orientation",
}
METRIC_KEYS = {
    "actual_depth_m", "forbidden_contact", "force_peak_n", "grasp_both",
    "lateral_error_m", "orientation_error_rad",
}
ROLLOUT_TARGET_KEYS = {"returns", "advantages"}
PARAMETER_FINITE_KEYS = {
    "actor_parameters", "actor_scalars", "critic_parameters", "critic_scalars",
}


class AuditError(RuntimeError):
    pass


def require(condition: Any, message: str) -> None:
    if not condition:
        raise AuditError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot load JSON object: {path}") from error
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def finite_number(value: Any, field: str) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), f"{field} must be numeric")
    number = float(value)
    require(math.isfinite(number), f"{field} must be finite")
    return number


def finite_vector(value: Any, length: int, field: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{field} must contain {length} values")
    return [finite_number(item, f"{field}[{index}]") for index, item in enumerate(value)]


def sha256_hex(value: Any, field: str) -> str:
    require(isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value), f"{field} must be 64 lowercase hex characters")
    return str(value)


def raw_base79_sha256(values: Sequence[float]) -> str:
    require(len(values) == 79, "base79 must contain 79 values")
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(b"[1,79]\0")
    digest.update(struct.pack("<79f", *(float(value) for value in values)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_jsonl_gzip(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(path.is_file(), f"missing gzip JSONL: {path}")
    compressed = path.read_bytes()
    try:
        raw = gzip.decompress(compressed)
        text = raw.decode("utf-8")
    except (OSError, EOFError, UnicodeDecodeError) as error:
        raise AuditError(f"invalid gzip JSONL: {path}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid JSONL row: {path}:{number}") from error
        require(isinstance(row, dict), f"JSONL row is not an object: {path}:{number}")
        rows.append(row)
    require(rows, f"empty gzip JSONL: {path}")
    return rows, {
        "path": str(path),
        "compressed_size": len(compressed),
        "uncompressed_size": len(raw),
        "compressed_sha256": hashlib.sha256(compressed).hexdigest(),
        "uncompressed_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(rows),
    }


def audit_optional_recovery_manifest(root: Path, path: Path, integrity: Mapping[str, Any]) -> dict[str, Any]:
    """Check recovered bytes when the public package includes a file manifest."""

    manifest_path = root / "file-manifest.json"
    if not manifest_path.is_file():
        return {"available": False, "verified": False}
    manifest = load_json(manifest_path)
    relative = str(path.relative_to(root))
    files = manifest.get("files", manifest)
    require(isinstance(files, dict), "recovery manifest file map is invalid")
    assert isinstance(files, dict)
    record = files.get(relative)
    require(record is not None, f"recovery manifest has no entry for {relative}")
    if isinstance(record, str):
        expected_sha, expected_size = record, None
    else:
        require(isinstance(record, dict), f"recovery manifest entry is invalid: {relative}")
        assert isinstance(record, dict)
        expected_sha = record.get("sha256")
        expected_size = record.get("size", record.get("bytes"))
    require(expected_sha == integrity["compressed_sha256"], f"recovery manifest SHA mismatch: {relative}")
    if expected_size is not None:
        require(expected_size == integrity["compressed_size"], f"recovery manifest size mismatch: {relative}")
    return {"available": True, "verified": True, "manifest": str(manifest_path), "relative_path": relative}


def report_path(root: Path, phase: str) -> Path:
    return root / f"{phase}-report.json"


def phase_dir(root: Path, phase: str) -> Path:
    return root / phase


def require_completed(report: Mapping[str, Any], phase: str) -> None:
    require(report.get("schema_name") == "rj45_multibias_probe_v1", f"{phase} report schema changed")
    require(report.get("status") == "completed", f"{phase} report is not completed")
    require(report.get("phase") == phase, f"{phase} report phase mismatch")
    claims = report.get("claims")
    require(isinstance(claims, dict), f"{phase} claims are missing")
    assert isinstance(claims, dict)
    require(claims == {"effect_proven": False, "generalization_proven": False, "final_policy": False}, f"{phase} claims changed")


def expected_case(case_id: str, base_seed: int, source: str) -> dict[str, Any]:
    return {
        "case_id": case_id, "base_seed": base_seed,
        "calibration_case_seed": base_seed + CALIBRATION_OFFSET,
        "split": "train", "grasp_id": "rj45-train-02",
        "episode_index": 1, "source": source,
    }


def audit_contract_identity(contract: Mapping[str, Any]) -> dict[str, Any]:
    require(contract.get("schema_name") == "rj45_multibias_contract_v1", "contract schema changed")
    require(contract.get("helper_sha256") == EXPECTED_HELPER_SHA256, "contract helper identity changed")
    require(contract.get("source_sha256") == EXPECTED_SOURCE_SHA256, "contract source identity changed")
    require(contract.get("origin_protocol") == {
        "path_name": "origin-protocol.json", "sha256": EXPECTED_ORIGIN_PROTOCOL_SHA256,
        "template_only": True,
    }, "contract origin protocol identity changed")
    claims = contract.get("claims")
    require(claims == {"dev_split_sealed": True, "effect_proven": False, "final_split_sealed": True, "generalization_proven": False}, "contract claims changed")
    training = contract.get("training")
    require(isinstance(training, dict), "contract training section is missing")
    assert isinstance(training, dict)
    expected_training = {
        "mode": "standard79", "num_envs": 1, "actor_observation_dim": 79,
        "critic_observation_dim": 92, "action_dim": 6, "right_arm_only": True,
        "held_bank_relative_path": "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json",
        "updates": 64, "rollout_steps": 128, "transitions": 8192,
        "single_pool": [20261128], "multi_pool": [20261128, 20271208, 20271216, 20271224],
        "policy_training_seed": 20261121, "critic_initialization_seed": 20261122,
        "actor_initialization_seed": 20261123, "evaluation_seed": 20261124,
        "final_checkpoint_update": 64, "eligible_checkpoint_updates": [64],
        "diagnostic_checkpoint_updates": [32],
        "checkpoint_selection": "final64_only_no_outcome_selection",
    }
    require(training == expected_training, "contract training identity changed")
    preflight = contract.get("reference_preflight")
    require(preflight == {
        "pool": [seed for _, seed in PREFLIGHT_CASES], "required_successes": 4,
        "episode_count": 4, "failure_policy": "no_replacement_no_ranking",
    }, "contract preflight identity changed")
    evaluation = contract.get("evaluation")
    require(isinstance(evaluation, dict), "contract evaluation section is missing")
    assert isinstance(evaluation, dict)
    expected_cases = [expected_case(case_id, seed, source) for case_id, seed, source in EVALUATION_CASES]
    require(evaluation.get("cases") == expected_cases, "contract evaluation cases changed")
    require(evaluation.get("methods") == list(METHODS), "contract evaluation methods changed")
    require(evaluation.get("episode_count") == 27, "contract evaluation episode count changed")
    require(evaluation.get("order") == "case_outer_cyclic_method_offset_case_index_mod_3", "contract evaluation order changed")
    require(evaluation.get("per_episode_wall_time_s") == 300, "contract per-episode guard changed")
    decision = contract.get("decision_rule")
    require(decision == {
        "checkpoint_replacement_allowed": False, "effect_proven": False,
        "followup_multiseed_candidate": "fresh8_multi_successes_at_least_single_plus_2_and_greater_than_zero_and_zero_success_multi_failure_count_at_most_1",
    }, "contract decision rule changed")
    budget = contract.get("budget_and_stop")
    require(isinstance(budget, dict), "contract budget/stop section is missing")
    assert isinstance(budget, dict)
    admission = budget.get("admission")
    require(isinstance(admission, dict), "contract budget admission is missing")
    assert isinstance(admission, dict)
    require(admission == {
        "budget_cap_usd": 700, "external_guard_minutes": 180,
        "locally_verified": True, "new_run_reservation_usd": 6,
        "private_evidence_sha256": "306b7ca200fda99e3f88673bf59338d6c4f861d52829d9f84e836483baa0be31",
    }, "contract budget admission changed")
    require(budget.get("deadline_utc") == "2026-10-31T10:50:52Z", "contract deadline changed")
    require(budget.get("train_phase_wall_time_s") == 3200 and budget.get("evaluate_phase_wall_time_s") == 4000 and budget.get("external_stop_wall_time_s") == 10800, "contract wall-time guards changed")
    return {"training": expected_training, "preflight_cases": [expected_case(case_id, seed, "training-pool") for case_id, seed in PREFLIGHT_CASES], "evaluation_cases": expected_cases}


def audit_source_lineage(root: Path, reports: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    first = reports[PHASES[0]]
    source = first.get("source_binding")
    require(isinstance(source, dict), "preflight source binding is missing")
    assert isinstance(source, dict)
    helper = first.get("contract", {})
    require(isinstance(helper, dict), "contract binding is missing")
    assert isinstance(helper, dict)
    contract_sha = helper.get("sha256")
    contract_sha = sha256_hex(contract_sha, "contract SHA")
    for phase, report in reports.items():
        require(report.get("source_binding") == source, f"{phase} source binding differs")
        phase_contract = report.get("contract")
        require(isinstance(phase_contract, dict), f"{phase} contract binding is invalid")
        assert isinstance(phase_contract, dict)
        require(phase_contract.get("sha256") == contract_sha, f"{phase} contract SHA differs")
        runner = report.get("source_binding", {}).get("runner", {})
        require(isinstance(runner, dict), f"{phase} runner binding is missing")
    contract_path = root / "contract.json"
    require(contract_path.is_file(), "public contract.json is missing")
    require(file_sha256(contract_path) == contract_sha, "public contract SHA differs from reports")
    contract = load_json(contract_path)
    require(contract.get("helper_sha256") == EXPECTED_HELPER_SHA256, "contract helper SHA differs from the executed helper")
    contract_sources = contract.get("source_sha256")
    require(isinstance(contract_sources, dict), "contract source hashes are missing")
    assert isinstance(contract_sources, dict)
    require(set(source) == set(EXPECTED_SOURCE_SHA256) | {"origin_protocol"}, "report source binding keys changed")
    for name, expected_sha in contract_sources.items():
        record = source.get(name)
        require(isinstance(record, dict), f"report source binding is invalid: {name}")
        assert isinstance(record, dict)
        require(record.get("sha256") == expected_sha, f"report/contract source hash differs: {name}")
    origin = source.get("origin_protocol")
    require(isinstance(origin, dict), "report origin protocol binding is invalid")
    assert isinstance(origin, dict)
    require(origin.get("sha256") == EXPECTED_ORIGIN_PROTOCOL_SHA256, "report origin protocol hash changed")
    return {
        "contract_sha256": contract_sha,
        "helper_sha256": EXPECTED_HELPER_SHA256,
        "source_binding": source,
        "public_contract_path": str(contract_path),
    }


def strict_terminal(evaluation: Mapping[str, Any], field: str) -> dict[str, Any]:
    classification = evaluation.get("classification")
    require(classification in VALID_CLASSIFICATIONS, f"{field} classification is invalid")
    terminal = evaluation.get("terminal")
    require(isinstance(terminal, dict), f"{field} terminal is missing")
    assert isinstance(terminal, dict)
    require(terminal.get("valid") is True, f"{field} terminal is not valid")
    criteria = terminal.get("criteria")
    require(isinstance(criteria, dict) and criteria, f"{field} terminal criteria are missing")
    assert isinstance(criteria, dict)
    require(set(criteria) == CRITERIA_KEYS, f"{field} terminal criteria keys changed")
    require(all(type(value) is bool for value in criteria.values()), f"{field} terminal criteria are not boolean")
    success = classification == "valid_success"
    require(terminal.get("success") is success, f"{field} success/classification contradiction")
    require(all(criteria.values()) is success, f"{field} criteria/classification contradiction")
    metrics = terminal.get("metrics")
    require(isinstance(metrics, dict) and metrics, f"{field} terminal metrics are missing")
    assert isinstance(metrics, dict)
    require(set(metrics) == METRIC_KEYS, f"{field} terminal metric keys changed")
    for name, value in metrics.items():
        finite_number(value, f"{field}.metrics.{name}")
    require(terminal.get("evaluator_source") == "simulator_ground_truth_independent", f"{field} evaluator source changed")
    require(terminal.get("evaluator_version") == "task-evaluator-v1+environment-cross-check", f"{field} evaluator version changed")
    timestamp = finite_number(terminal.get("terminal_timestamp_s"), f"{field}.terminal_timestamp_s")
    require(timestamp >= 0.0, f"{field} terminal timestamp must be nonnegative")
    reason = terminal.get("termination_reason")
    require(isinstance(reason, str) and bool(reason.strip()), f"{field} termination reason is empty")
    return {"classification": classification, "failed_criteria": sorted(name for name, value in criteria.items() if not value)}


def trace_file_for(root: Path, phase: str, index: int, case_id: str, method: str) -> Path:
    directory = phase_dir(root, phase) / "traces"
    expected = directory / f"{index:02d}-{case_id}-{method}.jsonl.gz"
    require(expected.is_file(), f"exact trace file is missing: {expected}")
    return expected


def audit_trace(
    path: Path, evaluation: Mapping[str, Any], field: str, *, require_observation: bool = False,
) -> tuple[dict[str, Any], list[float]]:
    rows, integrity = load_jsonl_gzip(path)
    require(rows[-1].get("event") == "complete", f"{field} trace has no final complete event")
    require(sum("event" in row for row in rows) == 1, f"{field} trace event count changed")
    require(rows[-1].get("classification") == evaluation.get("classification"), f"{field} trace/report classification differs")
    trace_record = evaluation.get("trace")
    require(isinstance(trace_record, dict), f"{field} trace record is missing")
    assert isinstance(trace_record, dict)
    reported_sha = sha256_hex(trace_record.get("sha256"), f"{field} trace SHA")
    require(reported_sha == integrity["uncompressed_sha256"], f"{field} uncompressed trace SHA mismatch")
    active = [row for row in rows[:-1] if row.get("active") is True and row.get("policy_input_valid") is True]
    require(active, f"{field} trace has no active valid row")
    first_observation = active[0].get("observation")
    require(isinstance(first_observation, dict), f"{field} first observation is missing")
    assert isinstance(first_observation, dict)
    base79 = finite_vector(first_observation.get("base79"), 79, f"{field}.base79")
    observation = evaluation.get("observation")
    if require_observation:
        require(isinstance(observation, dict), f"{field} observation proof is missing")
    if observation is not None:
        require(isinstance(observation, dict), f"{field} observation proof is invalid")
        assert isinstance(observation, dict)
        require(observation.get("dimensions") == {"policy": 79, "critic": 92}, f"{field} observation dimensions changed")
        require(observation.get("actor_ground_truth_access") is False, f"{field} actor ground-truth access changed")
        require(observation.get("base79_sha256") == raw_base79_sha256(base79), f"{field} base79 SHA mismatch")
    active_rows = evaluation.get("active_rows")
    require(type(active_rows) is int and active_rows > 0, f"{field} active-row count is invalid")
    require(active_rows == len(active), f"{field} active-row count mismatch")
    return integrity, base79


def audit_reference_controller_trace(path: Path, evaluation: Mapping[str, Any], field: str) -> dict[str, int | bool]:
    rows, _ = load_jsonl_gzip(path)
    controller = evaluation.get("reference_controller")
    require(isinstance(controller, dict) and set(controller) == {
        "active_rows", "nonzero_bounded_action_rows", "recovery_correction_observed",
    }, f"{field} reference-controller report structure changed")
    assert isinstance(controller, dict)
    active_rows = 0
    nonzero_rows = 0
    for index, row in enumerate(rows[:-1], 1):
        telemetry = row.get("reference_controller")
        require(isinstance(telemetry, dict) and set(telemetry) == {
            "active", "requested_raw_action_6d", "bounded_action_6d",
            "physical_offset_entry_6d", "bound_limited",
        }, f"{field} reference-controller trace telemetry missing at row {index}")
        assert isinstance(telemetry, dict)
        require(type(telemetry.get("active")) is bool and type(telemetry.get("bound_limited")) is bool, f"{field} reference-controller flags changed at row {index}")
        finite_vector(telemetry.get("requested_raw_action_6d"), 6, f"{field}.reference[{index}].requested")
        bounded = finite_vector(telemetry.get("bounded_action_6d"), 6, f"{field}.reference[{index}].bounded")
        finite_vector(telemetry.get("physical_offset_entry_6d"), 6, f"{field}.reference[{index}].physical_offset")
        active = telemetry["active"]
        active_rows += int(active)
        nonzero_rows += int(active and any(value != 0.0 for value in bounded))
    require(active_rows > 0 and nonzero_rows > 0, f"{field} reference controller did not apply a recovery correction")
    require(controller.get("active_rows") == active_rows, f"{field} reference-controller active count differs from trace")
    require(controller.get("nonzero_bounded_action_rows") == nonzero_rows, f"{field} reference-controller nonzero count differs from trace")
    require(controller.get("recovery_correction_observed") is True, f"{field} reference-controller recovery flag changed")
    return {
        "active_rows": active_rows,
        "nonzero_bounded_action_rows": nonzero_rows,
        "recovery_correction_observed": True,
    }


def audit_evaluation_rng(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"seeded", "state_sha256_before_environment_reset"}, f"{field} evaluation RNG structure changed")
    assert isinstance(value, dict)
    seeded = value.get("seeded")
    state = value.get("state_sha256_before_environment_reset")
    require(seeded == {
        "seed": 20261124, "python": True, "numpy": True, "torch_cpu": True,
        "torch_cuda_all_available_devices": True, "physics_seed_unchanged": 20261128,
    }, f"{field} evaluation RNG seed/flags changed")
    require(isinstance(state, dict) and set(state) == {"python", "numpy", "torch_cpu", "torch_cuda"}, f"{field} evaluation RNG hash structure changed")
    assert isinstance(state, dict)
    for name, digest in state.items():
        sha256_hex(digest, f"{field}.evaluation_rng.{name}")
    return value


def audit_unbound_training_pretrace(root: Path, phase: str) -> dict[str, Any]:
    """Audit the pre-training trace that the probe does not bind into its report."""

    path = phase_dir(root, phase) / "pre-evaluation.jsonl.gz"
    rows, integrity = load_jsonl_gzip(path)
    require(rows[-1].get("event") == "complete", f"{phase} pre-training trace has no complete footer")
    require(sum("event" in row for row in rows) == 1, f"{phase} pre-training trace event count changed")
    classification = rows[-1].get("classification")
    require(classification in VALID_CLASSIFICATIONS, f"{phase} pre-training trace classification is invalid")
    active = [row for row in rows[:-1] if row.get("active") is True and row.get("policy_input_valid") is True]
    require(active, f"{phase} pre-training trace has no active valid observation")
    observation = active[0].get("observation")
    require(isinstance(observation, dict), f"{phase} pre-training trace first observation is missing")
    assert isinstance(observation, dict)
    base79 = finite_vector(observation.get("base79"), 79, f"{phase}.pretrace.base79")
    manifest = audit_optional_recovery_manifest(root, path, integrity)
    return {
        "classification": classification,
        "initial_base79_sha256": raw_base79_sha256(base79),
        "trace_integrity": integrity,
        "recovery_manifest": manifest,
        "lineage_limit": "the helper did not preserve this pre-evaluation payload or trace SHA in the training report",
    }


def audit_bias_cases(report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    records = report.get("bias_cases")
    require(isinstance(records, list) and len(records) == len(cases), f"{field} bias case count changed")
    assert isinstance(records, list)
    result: dict[str, dict[str, Any]] = {}
    for index, (record, case) in enumerate(zip(records, cases, strict=True)):
        require(isinstance(record, dict), f"{field} bias case {index} is invalid")
        assert isinstance(record, dict)
        require(set(record) == {"case_id", "base_seed", "calibration_case_seed", "translation_entry_m", "rotation_vector_entry_rad"}, f"{field} bias case structure changed at {index}")
        require(record.get("case_id") == case["case_id"], f"{field} bias case order changed at {index}")
        require(record.get("base_seed") == case["base_seed"], f"{field} bias base seed changed at {index}")
        require(record.get("calibration_case_seed") == case["calibration_case_seed"], f"{field} calibration seed changed at {index}")
        translation = finite_vector(record.get("translation_entry_m"), 3, f"{field}.bias[{index}].translation")
        rotation = finite_vector(record.get("rotation_vector_entry_rad"), 3, f"{field}.bias[{index}].rotation")
        result[str(case["case_id"])] = {**record, "translation_entry_m": translation, "rotation_vector_entry_rad": rotation}
    return result


def audit_observed_bias(observed: Any, case: Mapping[str, Any], sampled: Mapping[str, Any], field: str) -> None:
    require(isinstance(observed, dict), f"{field} observed bias is missing")
    assert isinstance(observed, dict)
    require(observed.get("base_seed") == case["base_seed"], f"{field} observed bias base seed changed")
    require(observed.get("enabled") is True, f"{field} observed bias is not enabled")
    require(type(observed.get("level")) is int and observed["level"] >= 0, f"{field} observed bias level is invalid")
    require(observed.get("stream_offset") == 45123017 and observed.get("episode_stride") == 1000003, f"{field} observed bias seed constants changed")
    require(observed.get("case_seed_formula") == "base_seed + stream_offset + episode_index * episode_stride + env_id", f"{field} observed bias seed formula changed")
    require(observed.get("sampling_event") == "once_per_environment_reset_before_calibration", f"{field} observed bias sampling event changed")
    require(observed.get("episode_index") == [1], f"{field} observed bias episode changed")
    require(observed.get("translation_entry_m") == [sampled["translation_entry_m"]], f"{field} observed translation bias changed")
    require(observed.get("rotation_vector_entry_rad") == [sampled["rotation_vector_entry_rad"]], f"{field} observed rotation bias changed")


def audit_rng_state(value: Any, field: str) -> dict[str, str]:
    require(isinstance(value, dict) and set(value) == {"python", "numpy", "torch_cpu", "torch_cuda"}, f"{field} RNG state structure changed")
    assert isinstance(value, dict)
    return {name: sha256_hex(digest, f"{field}.{name}") for name, digest in value.items()}


def audit_training_rng(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {
        "seed", "python", "numpy", "torch_cpu", "torch_cuda_all_available_devices",
        "physics_seed_unchanged", "state_sha256",
    }, f"{field} training RNG structure changed")
    assert isinstance(value, dict)
    expected = {
        "seed": 20261121, "python": True, "numpy": True, "torch_cpu": True,
        "torch_cuda_all_available_devices": True, "physics_seed_unchanged": 20261128,
    }
    require({name: value.get(name) for name in expected} == expected, f"{field} training RNG seed/flags changed")
    audit_rng_state(value.get("state_sha256"), field)
    return value


def audit_initialization(value: Any, field: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"actor", "critic", "zero_actor", "optimizer_sha256", "rng"}, f"{field} initialization structure changed")
    assert isinstance(value, dict)
    actor = value.get("actor")
    require(isinstance(actor, dict) and set(actor) == {
        "mode", "seed", "state_sha256", "common_79d_projection_sha256",
        "oracle_extra_13_columns_zero", "output_layer_zero", "extra_columns_trainable",
    }, f"{field} actor initialization structure changed")
    assert isinstance(actor, dict)
    require(actor.get("mode") == "standard79" and actor.get("seed") == 20261123, f"{field} actor initialization identity changed")
    require(actor.get("oracle_extra_13_columns_zero") is True and actor.get("output_layer_zero") is True and actor.get("extra_columns_trainable") is False, f"{field} actor initialization flags changed")
    sha256_hex(actor.get("state_sha256"), f"{field}.actor.state_sha256")
    sha256_hex(actor.get("common_79d_projection_sha256"), f"{field}.actor.common_79d_projection_sha256")
    critic = value.get("critic")
    require(isinstance(critic, dict) and set(critic) == {"seed", "device", "resettable_module_count", "state_sha256", "actor_input_width_independent"}, f"{field} critic initialization structure changed")
    assert isinstance(critic, dict)
    require(critic.get("seed") == 20261122 and critic.get("actor_input_width_independent") is True, f"{field} critic initialization identity changed")
    require(critic.get("device") == "cuda:0", f"{field} critic device changed")
    require(type(critic.get("resettable_module_count")) is int and critic["resettable_module_count"] > 0, f"{field} critic reset count is invalid")
    sha256_hex(critic.get("state_sha256"), f"{field}.critic.state_sha256")
    zero = value.get("zero_actor")
    require(isinstance(zero, dict) and set(zero) == {"mode", "sample_rows", "action_shape", "mean_action_all_zero", "action_sha256", "noise"}, f"{field} zero-actor structure changed")
    assert isinstance(zero, dict)
    require(zero.get("mode") == "standard79" and zero.get("sample_rows") == 3 and zero.get("action_shape") == [3, 6] and zero.get("mean_action_all_zero") is True, f"{field} zero-actor identity changed")
    sha256_hex(zero.get("action_sha256"), f"{field}.zero_actor.action_sha256")
    noise = zero.get("noise")
    require(isinstance(noise, dict) and set(noise) == {"minimum", "maximum", "mean", "shape"}, f"{field} zero-actor noise structure changed")
    assert isinstance(noise, dict)
    require(noise.get("shape") == [6], f"{field} zero-actor noise shape changed")
    for name in ("minimum", "maximum", "mean"):
        require(math.isclose(finite_number(noise.get(name), f"{field}.zero_actor.noise.{name}"), 0.1, rel_tol=0.0, abs_tol=1.0e-6), f"{field} zero-actor noise changed")
    sha256_hex(value.get("optimizer_sha256"), f"{field}.optimizer_sha256")
    audit_training_rng(value.get("rng"), f"{field}.rng")
    return value


def audit_preflight(root: Path, report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require_completed(report, "preflight")
    episodes = report.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == 4, "preflight must contain four episodes")
    assert isinstance(episodes, list)
    require(report.get("strict_successes") == 4 and report.get("ranking_eligible") is True, "preflight strict gate did not pass 4/4")
    sampled = audit_bias_cases(report, cases, "preflight")
    traces = []
    for index, (episode, expected) in enumerate(zip(episodes, cases, strict=True), 1):
        require(isinstance(episode, dict), f"preflight episode {index} is invalid")
        assert isinstance(episode, dict)
        case = episode.get("case")
        evaluation = episode.get("evaluation")
        require(isinstance(case, dict) and isinstance(evaluation, dict), f"preflight episode {index} is incomplete")
        assert isinstance(case, dict) and isinstance(evaluation, dict)
        require(case == expected, f"preflight case identity changed at {index}")
        audit_observed_bias(episode.get("observed_bias"), expected, sampled[str(expected["case_id"])], f"preflight[{index}]")
        outcome = strict_terminal(evaluation, f"preflight[{index}]")
        audit_evaluation_rng(evaluation.get("evaluation_rng"), f"preflight[{index}]")
        require(outcome["classification"] == "valid_success", f"preflight episode {index} failed")
        path = trace_file_for(root, "preflight", index, str(case.get("case_id")), "reference")
        integrity, _ = audit_trace(path, evaluation, f"preflight[{index}]")
        audit_reference_controller_trace(path, evaluation, f"preflight[{index}]")
        traces.append(integrity)
    return {"strict_successes": 4, "ranking_eligible": True, "traces": traces}


def _log_path(root: Path, phase: str, name: str) -> Path:
    return phase_dir(root, phase) / f"{name}.jsonl.gz"


def audit_updates(root: Path, phase: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, integrity = load_jsonl_gzip(_log_path(root, phase, "updates"))
    require(len(rows) == UPDATES, f"{phase} update log must contain 64 rows")
    for index, row in enumerate(rows, 1):
        require(row.get("update") == index, f"{phase} update sequence changed at {index}")
        metrics = row.get("metrics")
        require(isinstance(metrics, dict) and metrics, f"{phase} update {index} metrics are missing")
        assert isinstance(metrics, dict)
        for name, value in metrics.items():
            finite_number(value, f"{phase}.update[{index}].{name}")
        rollout = row.get("rollout_targets")
        parameters = row.get("parameter_finiteness")
        require(isinstance(rollout, dict) and set(rollout) == ROLLOUT_TARGET_KEYS, f"{phase} rollout target keys changed at update {index}")
        require(isinstance(parameters, dict) and set(parameters) == PARAMETER_FINITE_KEYS, f"{phase} parameter finiteness keys changed at update {index}")
        assert isinstance(rollout, dict) and isinstance(parameters, dict)
        for name, value in {**rollout, **parameters}.items():
            require(type(value) is int and value > 0, f"{phase} update {index} {name} must be a positive integer")
        require(("diagnostic_checkpoint" in row) is (index == 32), f"{phase} diagnostic checkpoint placement changed at update {index}")
    diagnostic = rows[31].get("diagnostic_checkpoint")
    require(isinstance(diagnostic, dict) and diagnostic.get("eligible_for_selection") is False, f"{phase} checkpoint32 eligibility changed")
    return rows, integrity


def audit_transitions(root: Path, phase: str, report: Mapping[str, Any]) -> dict[str, Any]:
    rows, integrity = load_jsonl_gzip(_log_path(root, phase, "transitions"))
    require(len(rows) == TRANSITIONS, f"{phase} transition log must contain 8192 rows")
    counts: dict[str, dict[str, int]] = {}
    for index, row in enumerate(rows, 1):
        require(row.get("transition") == index, f"{phase} transition index changed at {index}")
        case_id = row.get("case_id")
        require(case_id in TRAIN_POOLS[phase], f"{phase} transition has unknown case {case_id}")
        finite_number(row.get("reward"), f"{phase}.transition[{index}].reward")
        require(type(row.get("done")) is bool and type(row.get("valid_input")) is bool and type(row.get("active")) is bool, f"{phase} transition flags changed")
        finite_vector(row.get("action"), 6, f"{phase}.transition[{index}].action")
        finite_vector(row.get("applied_expanded_action"), 12, f"{phase}.transition[{index}].applied")
        item = counts.setdefault(str(case_id), {"transitions": 0, "active_valid": 0, "terminal": 0})
        item["transitions"] += 1
        item["active_valid"] += int(row["active"] and row["valid_input"])
        item["terminal"] += int(row["done"])
    require(set(counts) == set(TRAIN_POOLS[phase]), f"{phase} transition pool coverage changed")
    require(all(item["active_valid"] > 0 for item in counts.values()), f"{phase} has a pool case with no active valid exposure")
    summary = report.get("transitions")
    require(isinstance(summary, dict), f"{phase} transition summary is missing")
    assert isinstance(summary, dict)
    require(summary.get("total_transitions") == TRANSITIONS, f"{phase} reported transition total changed")
    require(summary.get("by_case") == counts, f"{phase} transition case totals differ from log")
    first = summary.get("first_active_base79")
    require(isinstance(first, dict) and set(first) == set(TRAIN_POOLS[phase]), f"{phase} first-active proof coverage changed")
    assert isinstance(first, dict)
    for case_id, proof in first.items():
        require(isinstance(proof, dict), f"{phase} first-active proof is invalid: {case_id}")
        assert isinstance(proof, dict)
        values = finite_vector(proof.get("values"), 79, f"{phase}.{case_id}.first79")
        require(proof.get("sha256") == raw_base79_sha256(values), f"{phase} first79 SHA mismatch: {case_id}")
    return {"integrity": integrity, "counts": counts, "first_active_base79": first}


def audit_resets(
    root: Path, phase: str, report: Mapping[str, Any],
    cases: Sequence[Mapping[str, Any]], sampled: Mapping[str, Mapping[str, Any]],
    transition_counts: Mapping[str, Mapping[str, int]],
) -> dict[str, Any]:
    rows, integrity = load_jsonl_gzip(_log_path(root, phase, "resets"))
    pool = TRAIN_POOLS[phase]
    counts = {case_id: 0 for case_id in pool}
    for ordinal, row in enumerate(rows):
        case_id = pool[ordinal % len(pool)]
        case = cases[ordinal % len(cases)]
        require(row.get("phase") == "training", f"{phase} reset phase changed at {ordinal}")
        require(row.get("ordinal") == ordinal and row.get("pool_index") == ordinal % len(pool), f"{phase} reset cycle changed at {ordinal}")
        require(row.get("case_id") == case_id, f"{phase} reset case changed at {ordinal}")
        require(row.get("base_seed") == case["base_seed"], f"{phase} reset base seed changed at {ordinal}")
        require(row.get("calibration_case_seed") == case["calibration_case_seed"], f"{phase} reset calibration seed changed at {ordinal}")
        require(row.get("episode_index") == 1 and row.get("grasp_index") == 1, f"{phase} reset episode/grasp changed")
        translation = finite_vector(row.get("actual_translation_entry_m"), 3, f"{phase}.reset[{ordinal}].translation")
        rotation = finite_vector(row.get("actual_rotation_vector_entry_rad"), 3, f"{phase}.reset[{ordinal}].rotation")
        require(translation == sampled[case_id]["translation_entry_m"], f"{phase} reset translation bias changed at {ordinal}")
        require(rotation == sampled[case_id]["rotation_vector_entry_rad"], f"{phase} reset rotation bias changed at {ordinal}")
        counts[case_id] += 1
    require(report.get("reset_count") == len(rows), f"{phase} reset count differs from log")
    require(report.get("reset_count_by_case") == counts, f"{phase} reset case counts differ from log")
    require(all(count > 0 for count in counts.values()), f"{phase} reset pool exposure is incomplete")
    terminal_count = sum(item["terminal"] for item in transition_counts.values())
    require(len(rows) == 1 + terminal_count, f"{phase} reset count does not equal explicit reset plus terminal auto-resets")
    return {"integrity": integrity, "count": len(rows), "by_case": counts}


def audit_training(root: Path, phase: str, report: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require_completed(report, phase)
    require(report.get("learning_executed") is True and report.get("updates") == UPDATES, f"{phase} completion count changed")
    require(report.get("training_pool") == list(cases), f"{phase} training pool identity changed")
    sampled = audit_bias_cases(report, cases, phase)
    require("pre_evaluation" not in report, f"{phase} unexpectedly contains a pre_evaluation payload")
    pretrace = audit_unbound_training_pretrace(root, phase)
    update_rows, update_integrity = audit_updates(root, phase)
    transitions = audit_transitions(root, phase, report)
    resets = audit_resets(root, phase, report, cases, sampled, transitions["counts"])
    checkpoint = report.get("checkpoint")
    diagnostic = report.get("diagnostic_checkpoint")
    require(isinstance(checkpoint, dict) and checkpoint.get("update") == 64 and checkpoint.get("eligible_for_evaluation") is True, f"{phase} final64 eligibility changed")
    require(isinstance(diagnostic, dict) and diagnostic.get("eligible_for_selection") is False, f"{phase} diagnostic checkpoint eligibility changed")
    assert isinstance(checkpoint, dict) and isinstance(diagnostic, dict)
    sha256_hex(checkpoint.get("sha256"), f"{phase} final checkpoint SHA")
    sha256_hex(diagnostic.get("sha256"), f"{phase} diagnostic checkpoint SHA")
    require(update_rows[31]["diagnostic_checkpoint"].get("sha256") == diagnostic.get("sha256"), f"{phase} checkpoint32 SHA differs between update log and report")
    initialization = audit_initialization(report.get("initialization"), phase)
    training_rng = audit_training_rng(report.get("training_rng"), f"{phase}.training_rng")
    require(initialization["rng"] == training_rng, f"{phase} initialization/training RNG states differ")
    before = sha256_hex(report.get("actor_parameter_sha256_before"), f"{phase} actor-before SHA")
    after = sha256_hex(report.get("actor_parameter_sha256_after"), f"{phase} actor-after SHA")
    require(before == initialization["actor"]["state_sha256"], f"{phase} actor-before SHA differs from initialization")
    require(before != after, f"{phase} actor did not change")
    return {
        "updates": UPDATES, "transitions": transitions, "resets": resets,
        "updates_integrity": update_integrity, "checkpoint": checkpoint,
        "pre_evaluation_trace": pretrace, "initialization": initialization,
        "training_rng": training_rng,
    }


def audit_paired_training(single: Mapping[str, Any], multi: Mapping[str, Any]) -> dict[str, Any]:
    paths = (
        ("initialization", "actor"), ("initialization", "critic"),
        ("initialization", "optimizer_sha256"), ("initialization", "zero_actor"),
        ("initialization", "rng"), ("training_rng",),
        ("actor_parameter_sha256_before",),
    )
    checks: dict[str, bool] = {}

    def nested(record: Mapping[str, Any], path: Sequence[str], label: str) -> Any:
        value: Any = record
        for key in path:
            require(isinstance(value, Mapping) and key in value, f"{label} is missing {'.'.join(path)}")
            assert isinstance(value, Mapping)
            value = value[key]
        return value

    for path in paths:
        left = nested(single, path, "train-single")
        right = nested(multi, path, "train-multi")
        checks[".".join(path)] = left == right
    left_first = nested(single, ("transitions", "first_active_base79", "anchor"), "train-single")
    right_first = nested(multi, ("transitions", "first_active_base79", "anchor"), "train-multi")
    checks["first_active_anchor_base79"] = left_first == right_first
    require(all(checks.values()), f"single/multi initialization pairing failed: {checks}")
    return checks


def expected_evaluation_order(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    order = []
    for case_index, case in enumerate(cases):
        rotated = METHODS[case_index % 3:] + METHODS[:case_index % 3]
        for method in rotated:
            order.append({"index": len(order) + 1, "case_id": case["case_id"], "method": method})
    return order


def audit_evaluation(
    root: Path, report: Mapping[str, Any], training: Mapping[str, Mapping[str, Any]],
    contract_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    require_completed(report, "evaluate")
    require(report.get("effect_proven") is False and report.get("checkpoint_selection_changed") is False, "evaluation claim flags changed")
    episodes = report.get("episodes")
    require(isinstance(episodes, list) and len(episodes) == 27, "evaluation must contain 27 episodes")
    assert isinstance(episodes, list)
    require(len(contract_cases) == 9, "contract evaluation case count changed")
    cases_by_id = {str(case["case_id"]): case for case in contract_cases}
    require(len(cases_by_id) == 9, "contract evaluation case ids are not unique")
    expected = expected_evaluation_order(contract_cases)
    sampled = audit_bias_cases(report, contract_cases, "evaluate")
    paired: dict[str, dict[str, Any]] = {}
    outcomes: dict[str, dict[str, str]] = {case_id: {} for case_id in cases_by_id}
    failure_criteria: dict[str, int] = {}
    trace_integrity = []
    common_evaluation_rng: dict[str, Any] | None = None
    for wanted, episode in zip(expected, episodes, strict=True):
        require(isinstance(episode, dict), f"evaluation episode {wanted['index']} is invalid")
        assert isinstance(episode, dict)
        episode_case = episode.get("case")
        require(isinstance(episode_case, dict), f"evaluation case is missing at {wanted['index']}")
        assert isinstance(episode_case, dict)
        require(episode.get("index") == wanted["index"], f"evaluation index mismatch at {wanted['index']}")
        require(episode_case.get("case_id") == wanted["case_id"], f"evaluation case order mismatch at {wanted['index']}")
        require(episode_case == cases_by_id[wanted["case_id"]], f"evaluation case identity changed at {wanted['index']}")
        require(episode.get("method") == wanted["method"], f"evaluation method order mismatch at {wanted['index']}")
        evaluation = episode.get("evaluation")
        require(isinstance(evaluation, dict), f"evaluation payload missing at {wanted['index']}")
        assert isinstance(evaluation, dict)
        terminal = strict_terminal(evaluation, f"evaluate[{wanted['index']}]")
        evaluation_rng = audit_evaluation_rng(evaluation.get("evaluation_rng"), f"evaluate[{wanted['index']}]")
        if common_evaluation_rng is None:
            common_evaluation_rng = evaluation_rng
        else:
            require(evaluation_rng == common_evaluation_rng, f"evaluation RNG differs at episode {wanted['index']}")
        for criterion in terminal["failed_criteria"]:
            failure_criteria[criterion] = failure_criteria.get(criterion, 0) + 1
        trace = trace_file_for(root, "evaluate", wanted["index"], wanted["case_id"], wanted["method"])
        integrity, base79 = audit_trace(
            trace, evaluation, f"evaluate[{wanted['index']}]", require_observation=True,
        )
        trace_integrity.append(integrity)
        base_hash = raw_base79_sha256(base79)
        require(episode.get("initial_base79_sha256") == base_hash, f"evaluation initial79 mismatch at {wanted['index']}")
        bias = episode.get("bias")
        require(isinstance(bias, dict), f"evaluation bias missing at {wanted['index']}")
        assert isinstance(bias, dict)
        audit_observed_bias(bias, cases_by_id[wanted["case_id"]], sampled[wanted["case_id"]], f"evaluate[{wanted['index']}]")
        pair = {
            "base79_sha256": base_hash,
            "evaluation_rng": evaluation_rng,
            "bias": bias,
            "grasp_id": episode_case.get("grasp_id"),
            "episode_index": bias.get("episode_index"),
        }
        case_id = wanted["case_id"]
        if case_id in paired:
            require(paired[case_id] == pair, f"paired initial condition differs for {case_id}")
        else:
            paired[case_id] = pair
        outcomes[case_id][wanted["method"]] = terminal["classification"]
    require(all(set(methods) == set(METHODS) for methods in outcomes.values()), "evaluation method coverage changed")
    checkpoints = report.get("checkpoints")
    require(isinstance(checkpoints, dict), "evaluation checkpoint lineage is missing")
    assert isinstance(checkpoints, dict)
    for label, phase in (("single64", "train-single"), ("multi64", "train-multi")):
        expected_sha = training[phase]["checkpoint"]["sha256"]
        record = checkpoints.get(label)
        require(isinstance(record, dict), f"{label} checkpoint record is missing")
        assert isinstance(record, dict)
        require(set(record) == {"path", "sha256", "iteration_field", "eligible_update", "eligible_for_evaluation"}, f"{label} checkpoint record structure changed")
        require(record.get("sha256") == expected_sha, f"{label} checkpoint lineage mismatch")
        require(record.get("iteration_field") == 63 and record.get("eligible_update") == 64 and record.get("eligible_for_evaluation") is True, f"{label} checkpoint is not final64")
    fresh_ids = [case_id for case_id, case in cases_by_id.items() if case.get("source") == "policy-unseen"]
    require(len(fresh_ids) == 8, "evaluation fresh-case count changed")
    successes = {method: sum(outcomes[case_id][method] == "valid_success" for case_id in fresh_ids) for method in METHODS}
    zero_success_multi_failure = sum(
        outcomes[case_id]["zero"] == "valid_success" and outcomes[case_id]["multi64"] == "valid_failure"
        for case_id in fresh_ids
    )
    candidate = successes["multi64"] >= successes["single64"] + 2 and successes["multi64"] > successes["zero"] and zero_success_multi_failure <= 1
    require(report.get("fresh8_successes") == successes, "reported fresh8 success counts differ from audit")
    require(report.get("zero_success_multi_failure_count") == zero_success_multi_failure, "reported zero-success/multi-failure count differs")
    require(report.get("followup_multiseed_candidate") is candidate, "reported candidate gate differs from audit")

    def paired_comparison(left: str, right: str) -> dict[str, int]:
        wins = sum(outcomes[case_id][left] == "valid_success" and outcomes[case_id][right] == "valid_failure" for case_id in fresh_ids)
        losses = sum(outcomes[case_id][left] == "valid_failure" and outcomes[case_id][right] == "valid_success" for case_id in fresh_ids)
        return {"wins": wins, "losses": losses, "ties": len(fresh_ids) - wins - losses}

    return {
        "episode_count": 27, "case_count": 9, "paired_conditions_identical": True,
        "fresh8_successes": successes,
        "paired": {"multi64_vs_single64": paired_comparison("multi64", "single64"), "multi64_vs_zero": paired_comparison("multi64", "zero")},
        "failure_criteria": failure_criteria,
        "zero_success_multi_failure_count": zero_success_multi_failure,
        "followup_multiseed_candidate": candidate,
        "effect_proven": False, "trace_integrity": trace_integrity,
        "common_evaluation_rng": common_evaluation_rng,
    }


def audit(root: Path) -> dict[str, Any]:
    reports = {phase: load_json(report_path(root, phase)) for phase in PHASES}
    for phase, report in reports.items():
        require_completed(report, phase)
    lineage = audit_source_lineage(root, reports)
    contract_identity = audit_contract_identity(load_json(root / "contract.json"))
    preflight_cases = contract_identity["preflight_cases"]
    preflight = audit_preflight(root, reports["preflight"], preflight_cases)
    single = audit_training(root, "train-single", reports["train-single"], preflight_cases[:1])
    multi = audit_training(root, "train-multi", reports["train-multi"], preflight_cases)
    paired_training = audit_paired_training(reports["train-single"], reports["train-multi"])
    evaluation = audit_evaluation(
        root, reports["evaluate"], {"train-single": single, "train-multi": multi},
        contract_identity["evaluation_cases"],
    )
    return {
        "schema_name": "rj45_multibias_evidence_audit_v1",
        "status": "pass", "ranking_allowed": True,
        "source_lineage": lineage, "contract_identity": contract_identity, "preflight": preflight,
        "training": {"single64": single, "multi64": multi, "paired_initialization": paired_training},
        "evaluation": evaluation,
        "claims": {"effect_proven": False, "generalization_proven": False, "final_policy": False},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    output = args.output.resolve() if args.output is not None else root / "audit.json"
    try:
        result = audit(root)
        exit_code = 0
    except AuditError as error:
        if not args.allow_incomplete:
            raise
        result = {
            "schema_name": "rj45_multibias_evidence_audit_v1",
            "status": "incomplete", "ranking_allowed": False,
            "error": {"type": type(error).__name__, "message": str(error)},
            "claims": {"effect_proven": False, "generalization_proven": False, "final_policy": False},
        }
        exit_code = 0
    atomic_json(output, result)
    print(json.dumps({"status": result["status"], "output": str(output)}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as error:
        print(f"audit failed: {error}", file=sys.stderr)
        raise SystemExit(1)
