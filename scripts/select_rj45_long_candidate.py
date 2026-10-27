#!/usr/bin/env python3
"""Verify raw RJ45 pilot evidence and select the preregistered long-run candidate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from types import ModuleType
from typing import Any


EXPECTED_ATTEMPTS = 32
EXPECTED_BASELINE = "baseline"
EXPECTED_GROUPS = ("A0", "A1", "A2", "E1", "Z1")
EXPECTED_TIE_BREAK = ("Z1", "E1", "A2", "A1", "A0")
VERIFIED_BUNDLE_STATUS = "checkpoint_manifest_provenance_runner_bytes_verified"
FRESH_AUDIT_FIELDS = (
    "scope", "experiment", "bootstrap", "results", "paired_contrasts",
    "evaluation_bias_coverage", "paired_cases",
)


def _fail(message: str) -> None:
    raise ValueError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict[str, Any], bytes, str]:
    def reject(constant: str) -> None:
        _fail(f"non-finite JSON constant in {path}: {constant}")

    try:
        raw = path.read_bytes()
        payload = _object(json.loads(raw.decode("utf-8"), parse_constant=reject), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {path}: {error}") from error
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _repo_path(root: Path, raw: Any, name: str) -> Path:
    if not isinstance(raw, str) or not raw:
        _fail(f"{name} must be a non-empty repository-relative path")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        _fail(f"{name} must stay inside the repository")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail(f"{name} resolves outside the repository")
    return resolved


def _selection_contract(plan: dict[str, Any]) -> tuple[list[str], list[str]]:
    if plan.get("schema_version") != 1:
        _fail("long-run plan schema_version must be 1")
    if plan.get("selection_preregistered_before_any_pilot_32_case_policy_evaluation_result") is not True:
        _fail("long-run plan does not prove preregistered pilot selection")
    selection = _object(plan.get("pilot_candidate_selection"), "pilot_candidate_selection")
    groups = selection.get("eligible_groups")
    tie_break = selection.get("tie_break_order")
    if groups != list(EXPECTED_GROUPS) or tie_break != list(EXPECTED_TIE_BREAK):
        _fail("candidate groups or tie-break order differ from preregistration")
    if selection.get("primary_rank") != (
        "Largest strict-v1 success count on the predeclared 32 pilot development cases."
    ):
        _fail("primary_rank differs from the preregistered strict-v1 metric")
    if selection.get("not_a_selection_metric") != (
        "Do not rank different reward profiles by their reward sums, whose scales differ."
    ):
        _fail("reward-sum exclusion differs from preregistration")
    return groups, tie_break


def _parent_contract(parent: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if parent.get("schema_version") != 1 or parent.get("status") != "predeclared_before_training":
        _fail("parent pilot plan is not the predeclared schema-1 plan")
    evaluation = _object(parent.get("development_evaluation"), "parent.development_evaluation")
    expected_evaluation = {
        "baseline": "biased_zero", "episodes": 32, "num_envs": 16,
        "seed": 20261028, "matching_condition_ids_and_bias_required": True,
        "all_requested_attempts_in_denominator": True,
        "bootstrap_seed": 20261029, "bootstrap_resamples": 100000,
    }
    for field, expected in expected_evaluation.items():
        if evaluation.get(field) != expected:
            _fail(f"parent development_evaluation.{field} differs from the frozen contract")
    primary = evaluation.get("primary")
    if not isinstance(primary, str) or "12–13 mm" not in primary or "<=50 µm" not in primary:
        _fail("parent strict-v1 criterion is missing its declared depth/lateral bounds")
    training = _object(parent.get("training"), "parent.training")
    for field, expected in {
        "environments": 64, "updates": 20, "rollout_steps_per_env": 128,
        "seed": 20261026, "checkpoint": "final_only_model_19.pt",
    }.items():
        if training.get(field) != expected:
            _fail(f"parent training.{field} differs from the frozen contract")
    groups = parent.get("groups")
    if not isinstance(groups, list) or [group.get("id") for group in groups if isinstance(group, dict)] != list(EXPECTED_GROUPS):
        _fail("parent group definitions are incomplete or reordered")
    expected = {
        "A0": ("configs/rj45-training-target-offset-v1.json", "configs/ppo-rj45-v1.json", False),
        "A1": ("configs/rj45-training-target-offset-a1-discounted-pose-v2.json", "configs/ppo-rj45-v1.json", False),
        "A2": ("configs/rj45-training-target-offset-a2-keypoint-shape-v2.json", "configs/ppo-rj45-v1.json", False),
        "E1": ("configs/rj45-training-target-offset-v1.json", "configs/ppo-rj45-low-exploration-v1.json", False),
        "Z1": ("configs/rj45-training-target-offset-v1.json", "configs/ppo-rj45-low-exploration-v1.json", True),
    }
    result: dict[str, dict[str, Any]] = {}
    for group in groups:
        label = group["id"]
        training_config, ppo_config, zero = expected[label]
        if group.get("training_config") != training_config or group.get("ppo_config") != ppo_config:
            _fail(f"parent {label} configuration mapping differs from the frozen contract")
        if group.get("training_flags") != (["--zero-init-actor-output"] if zero else []):
            _fail(f"parent {label} zero-initialization mapping is invalid")
        result[label] = {"training_config": training_config, "ppo_config": ppo_config, "zero": zero}
    return result


def _load_verified_analyzer(root: Path, parent: dict[str, Any]) -> tuple[ModuleType, Path, str]:
    frozen = _object(_object(parent.get("frozen_inputs"), "parent.frozen_inputs").get("sha256"), "parent.frozen_inputs.sha256")
    relative = "scripts/analyze_rj45_paired.py"
    expected = _sha(frozen.get(relative), f"parent frozen {relative}")
    path = _repo_path(root, relative, "analyzer path")
    actual = _sha_file(path)
    if actual != expected:
        _fail("paired analyzer bytes differ from the parent pilot frozen input")
    spec = importlib.util.spec_from_file_location("verified_rj45_paired_analyzer", path)
    if spec is None or spec.loader is None:
        _fail("cannot load the verified paired analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path, actual


def _verify_config_files(
    root: Path, parent: dict[str, Any], mapping: dict[str, dict[str, Any]]
) -> dict[Path, str]:
    frozen = _object(parent["frozen_inputs"]["sha256"], "parent.frozen_inputs.sha256")
    verified: dict[Path, str] = {}
    for group in mapping.values():
        for field in ("training_config", "ppo_config"):
            relative = group[field]
            expected = _sha(frozen.get(relative), f"parent frozen {relative}")
            path = _repo_path(root, relative, relative)
            if _sha_file(path) != expected:
                _fail(f"current bytes differ from frozen input: {relative}")
            verified[path] = expected
    return verified


def _verify_parent_source_files(root: Path, parent: dict[str, Any]) -> dict[Path, str]:
    frozen = _object(parent["frozen_inputs"]["sha256"], "parent.frozen_inputs.sha256")
    expected = {
        path: digest
        for path, digest in frozen.items()
        if (path.startswith("src/rby1_assembly/") and path.endswith(".py"))
        or path == "scripts/evaluate_policy.py"
    }
    if "scripts/evaluate_policy.py" not in expected or not any(
        path.startswith("src/rby1_assembly/") for path in expected
    ):
        _fail("parent frozen source map is incomplete")
    verified: dict[Path, str] = {}
    for relative, digest in expected.items():
        expected_sha = _sha(digest, f"parent frozen {relative}")
        path = _repo_path(root, relative, relative)
        if _sha_file(path) != expected_sha:
            _fail(f"current source bytes differ from frozen input: {relative}")
        verified[path] = expected_sha
    return verified


def _load_strict_contract(root: Path, parent: dict[str, Any]) -> tuple[dict[str, float], str]:
    relative = "configs/tasks/rj45-v1.json"
    frozen = _object(parent["frozen_inputs"]["sha256"], "parent.frozen_inputs.sha256")
    expected_sha = _sha(frozen.get(relative), f"parent frozen {relative}")
    path = _repo_path(root, relative, "strict-v1 task config")
    task, _raw, actual_sha = _load_json(path)
    if actual_sha != expected_sha:
        _fail("strict-v1 task config bytes differ from the parent frozen input")
    evaluation = _object(task.get("evaluation"), "strict-v1 task evaluation")
    expected = {
        "assembled_depth_m": 0.012,
        "max_depth_m": 0.013,
        "max_lateral_error_m": 0.00005,
        "max_orientation_error_rad": 0.008726646259971648,
        "dwell_s": 0.4,
        "max_contact_force_n": 25.0,
        "max_relative_linear_speed_m_s": 0.005,
        "max_relative_angular_speed_rad_s": 0.05,
    }
    for field, required in expected.items():
        value = evaluation.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) != required:
            _fail(f"strict-v1 task evaluation.{field} differs from the fixed contract")
    return expected, expected_sha


def _resolve_recorded_file(root: Path, value: Any, name: str) -> tuple[Path, str]:
    record = _object(value, name)
    path = _repo_path(root, record.get("path"), f"{name}.path")
    expected = _sha(record.get("sha256"), f"{name}.sha256")
    try:
        actual = _sha_file(path)
    except OSError as error:
        raise ValueError(f"missing or unreadable raw evidence {path}: {error}") from error
    if actual != expected:
        _fail(f"raw evidence bytes differ from {name}: {path}")
    return path, actual


def _verify_raw_audit(
    root: Path, parent: dict[str, Any], audit: dict[str, Any], analyzer: ModuleType,
    mapping: dict[str, dict[str, Any]], strict: dict[str, float], task_config_sha: str,
) -> tuple[dict[str, Any], dict[str, str]]:
    if audit.get("schema_version") != 1:
        _fail("stored paired audit schema_version must be 1")
    reports_meta = _object(audit.get("input_reports"), "audit.input_reports")
    expected_labels = {EXPECTED_BASELINE, *EXPECTED_GROUPS}
    if set(reports_meta) != expected_labels:
        _fail("audit input_reports must contain baseline and all five groups")
    loaded: dict[str, dict[str, Any]] = {}
    immutable: dict[Path, str] = {}
    report_paths: set[Path] = set()
    for label in (EXPECTED_BASELINE, *EXPECTED_GROUPS):
        path, digest = _resolve_recorded_file(root, reports_meta[label], f"audit.input_reports.{label}")
        if path in report_paths:
            _fail("raw evaluation report paths must be unique")
        report_paths.add(path)
        payload, _raw, analyzer_digest = analyzer._load(path)
        if analyzer_digest != digest:
            _fail(f"analyzer read a different {label} report")
        loaded[label] = payload
        immutable[path] = digest

    evaluation = parent["development_evaluation"]
    frozen_source = {
        path: digest
        for path, digest in parent["frozen_inputs"]["sha256"].items()
        if (path.startswith("src/rby1_assembly/") and path.endswith(".py"))
        or path == "scripts/evaluate_policy.py"
    }
    for label, report in loaded.items():
        for field, expected in {
            "seed": evaluation["seed"], "evaluation_start": "held",
            "evaluation_num_envs": evaluation["num_envs"], "requested_attempts": evaluation["episodes"],
        }.items():
            if report.get(field) != expected:
                _fail(f"raw report {label}.{field} differs from the parent plan")
        limits = _object(_object(report.get("task_model"), f"{label}.task_model").get("evaluation_stability_limits"), f"{label}.evaluation_stability_limits")
        provenance = _object(report.get("provenance"), f"{label}.provenance")
        if provenance.get("task_config_sha256") != task_config_sha:
            _fail(f"raw report {label} does not reference the frozen strict-v1 task config")
        if provenance.get("source_sha256") != frozen_source:
            _fail(f"raw report {label} source map differs from the parent frozen source map")
        report_limit_mapping = {
            "assembled_depth_m": "assembled_depth_m",
            "max_depth_m": "max_depth_m",
            "max_lateral_error_m": "max_lateral_error_m",
            "max_orientation_error_rad": "max_orientation_error_rad",
            "required_continuous_dwell_s": "dwell_s",
            "max_contact_force_n": "max_contact_force_n",
            "max_relative_linear_speed_m_s": "max_relative_linear_speed_m_s",
            "max_relative_angular_speed_rad_s": "max_relative_angular_speed_rad_s",
        }
        for report_field, task_field in report_limit_mapping.items():
            if report_field in limits and limits[report_field] != strict[task_field]:
                _fail(f"raw report {label} loosens or changes strict-v1 {report_field}")
        for required_field in (
            "max_depth_m", "required_continuous_dwell_s",
            "max_relative_linear_speed_m_s", "max_relative_angular_speed_rad_s",
        ):
            if required_field not in limits:
                _fail(f"raw report {label} omits reported strict-v1 {required_field}")
        for outcome in report.get("episode_outcomes", []):
            if outcome.get("success"):
                bounded_contact_peaks: dict[str, float] = {}
                for field in ("grip_contact_force_peak_n", "insertion_contact_load_peak_n"):
                    value = outcome.get(field)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                        or float(value) < 0.0
                    ):
                        _fail(f"raw report {label} successful outcome has invalid {field}")
                    bounded_contact_peaks[field] = float(value)
                checks = (
                    (strict["assembled_depth_m"] <= outcome.get("final_depth_m", -math.inf) <= strict["max_depth_m"], "depth"),
                    (outcome.get("final_lateral_error_m", math.inf) <= strict["max_lateral_error_m"], "lateral error"),
                    (outcome.get("final_orientation_error_rad", math.inf) <= strict["max_orientation_error_rad"], "orientation error"),
                    (outcome.get("final_dwell_s", -math.inf) >= strict["dwell_s"], "dwell"),
                    (bounded_contact_peaks["grip_contact_force_peak_n"] <= strict["max_contact_force_n"], "grip contact force"),
                    (bounded_contact_peaks["insertion_contact_load_peak_n"] <= strict["max_contact_force_n"], "insertion contact load"),
                )
                if any(not passed for passed, _name in checks):
                    failed = [name for passed, name in checks if not passed]
                    _fail(f"raw report {label} marks an outcome successful outside strict-v1: {failed}")

    fresh = analyzer.analyze_reports(
        loaded[EXPECTED_BASELINE], {label: loaded[label] for label in EXPECTED_GROUPS},
        expected_attempts=32, bootstrap_resamples=evaluation["bootstrap_resamples"],
        bootstrap_seed=evaluation["bootstrap_seed"], baseline_label=EXPECTED_BASELINE,
    )
    for field in FRESH_AUDIT_FIELDS:
        if audit.get(field) != fresh.get(field):
            _fail(f"stored audit {field} differs from a fresh raw-report analysis")
    if fresh["results"][EXPECTED_BASELINE]["successes_all_attempts"] != 12:
        _fail("pilot baseline must reproduce the preregistered 12/32 result")
    baseline_results = [case["results"][EXPECTED_BASELINE] for case in fresh["paired_cases"]]
    if fresh["results"][EXPECTED_BASELINE]["invalid_episodes"] != 0:
        _fail("pilot baseline must have zero invalid episodes")
    if any(result["forbidden_contact_load_peak_n"] != 0.0 for result in baseline_results):
        _fail("pilot baseline must have zero forbidden contact load")

    bundles_meta = _object(audit.get("training_artifact_verification"), "audit.training_artifact_verification")
    if set(bundles_meta) != set(EXPECTED_GROUPS):
        _fail("audit must record a bundle for every learned group")
    fresh_bundles: dict[str, Any] = {}
    training = parent["training"]
    frozen = parent["frozen_inputs"]["sha256"]
    for label in EXPECTED_GROUPS:
        recorded = _object(bundles_meta[label], f"audit.training_artifact_verification.{label}")
        paths: dict[str, Path] = {}
        for key in ("checkpoint", "checkpoint_manifest", "training_provenance", "runner_config"):
            path, digest = _resolve_recorded_file(root, recorded.get(key), f"audit.training_artifact_verification.{label}.{key}")
            paths[key] = path
            immutable[path] = digest
        verified = analyzer._verify_training_bundle(
            label, loaded[label], checkpoint_path=paths["checkpoint"],
            manifest_path=paths["checkpoint_manifest"],
            training_provenance_path=paths["training_provenance"], runner_config_path=paths["runner_config"],
        )
        verified["causal_interpretation"] = "prohibited_without_a_separately_verified_preregistered_experiment_plan"
        for key in ("checkpoint", "checkpoint_manifest", "training_provenance", "runner_config"):
            verified[key]["path"] = recorded[key]["path"]
        if verified != recorded:
            _fail(f"stored {label} bundle audit differs from fresh byte verification")
        group = mapping[label]
        result_config = fresh["experiment"]["reported_configuration_identifiers"][label]
        expected_training_sha = _sha(frozen[group["training_config"]], f"frozen {label} training config")
        expected_ppo_sha = _sha(frozen[group["ppo_config"]], f"frozen {label} PPO config")
        if result_config.get("training_config_path") != group["training_config"] or result_config.get("training_config_sha256") != expected_training_sha:
            _fail(f"{label} report uses the wrong reward/training configuration")
        reported_policy_path = result_config.get("policy_config", "")
        if (
            not isinstance(reported_policy_path, str)
            or not Path(reported_policy_path).as_posix().endswith(group["ppo_config"])
            or result_config.get("policy_config_sha256") != expected_ppo_sha
        ):
            _fail(f"{label} report uses the wrong PPO configuration")
        initialization = _object(result_config.get("policy_initialization"), f"{label}.policy_initialization")
        if initialization.get("zero_actor_output_layer") is not group["zero"]:
            _fail(f"{label} report uses the wrong actor initialization")
        ppo, _raw, _digest = _load_json(_repo_path(root, group["ppo_config"], f"{label} PPO config"))
        expected_noise = ppo["policy"]["init_noise_std"]
        for field, expected in {
            "training_seed": training["seed"], "training_num_envs": training["environments"],
            "rollout_steps_per_env": training["rollout_steps_per_env"], "training_updates": training["updates"],
            "initial_action_noise_std": expected_noise,
        }.items():
            if verified.get(field) != expected:
                _fail(f"{label} bundle {field} differs from the parent plan")
        fresh_bundles[label] = verified
    fresh["training_artifact_verification"] = fresh_bundles
    for path, digest in immutable.items():
        if _sha_file(path) != digest:
            _fail(f"raw evidence changed during selection: {path}")
    return fresh, {str(path.relative_to(root)): digest for path, digest in immutable.items()}


def _rank_verified_audit(plan: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    groups, tie_break = _selection_contract(plan)
    ranking = []
    for label in groups:
        summary = audit["results"][label]
        cases = [case["results"][label] for case in audit["paired_cases"]]
        invalid = sum(not result["valid"] for result in cases)
        forbidden_values = [result["forbidden_contact_load_peak_n"] for result in cases]
        forbidden_unavailable = any(value is None for value in forbidden_values)
        numeric_forbidden = [float(value) for value in forbidden_values if value is not None]
        if any(not math.isfinite(value) or value < 0.0 for value in numeric_forbidden):
            _fail(f"verified audit {label} has an invalid forbidden-contact load")
        maximum_forbidden = max(numeric_forbidden, default=0.0)
        successes = sum(result["valid"] and result["success"] for result in cases)
        if summary["attempts"] != 32 or summary["invalid_episodes"] != invalid or summary["successes_all_attempts"] != successes:
            _fail(f"verified audit {label} aggregate changed before ranking")
        reasons = []
        if audit["training_artifact_verification"][label]["status"] != VERIFIED_BUNDLE_STATUS:
            reasons.append("checkpoint_bundle_not_verified")
        if invalid:
            reasons.append("invalid_episodes_nonzero")
        if forbidden_unavailable:
            reasons.append("forbidden_contact_load_unavailable")
        if maximum_forbidden != 0.0:
            reasons.append("forbidden_contact_load_nonzero")
        ranking.append({
            "group": label, "eligible": not reasons, "successes_all_attempts": successes,
            "attempts": 32, "invalid_episodes": invalid,
            "maximum_forbidden_contact_load_peak_n": maximum_forbidden,
            "checkpoint_bundle_status": audit["training_artifact_verification"][label]["status"],
            "tie_break_position": tie_break.index(label), "exclusion_reasons": reasons,
        })
    ranking.sort(key=lambda item: (-int(item["eligible"]), -item["successes_all_attempts"], item["tie_break_position"]))
    eligible = [entry for entry in ranking if entry["eligible"]]
    if not eligible:
        _fail("no eligible pilot group; stop and diagnose instead of starting long training")
    return {
        "schema_version": 2,
        "decision": {
            "selected_group": eligible[0]["group"],
            "selected_successes_all_attempts": eligible[0]["successes_all_attempts"],
            "attempts": 32, "selection_metric": "strict-v1 successes_all_attempts",
            "tie_break_order": tie_break, "reward_sums_used": False,
        },
        "ranking": ranking,
        "verification_scope": {
            "status": "raw_reports_and_training_bundles_rehashed_and_fresh_analyzer_replayed",
            "success_interpretation": "Evaluator-reported terminal metrics were revalidated by the frozen paired analyzer, with the declared 13 mm upper depth checked here.",
            "unverified": [
                "Force, relative-velocity, and dwell time histories are not independently recomputed; their thresholds and evaluator-reported terminal outcome remain tied to the frozen task config and evaluator.",
                "CUDA calibration RNG vectors are not regenerated on CPU; exact vectors and canonical case IDs are compared across the six raw reports.",
            ],
        },
    }


def _atomic_create(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"output already exists: {path}") from error
        temporary.unlink()
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _verify_plan_chain(
    long_plan_path: Path, parent_plan_path: Path, expected_long_sha: str
) -> tuple[dict[str, Any], str, dict[str, Any], str]:
    expected = _sha(expected_long_sha, "expected long-plan SHA-256")
    long_plan, _raw, long_sha = _load_json(long_plan_path)
    if long_sha != expected:
        _fail("long-run plan does not match the remotely preregistered expected SHA-256")
    parent, _parent_raw, parent_sha = _load_json(parent_plan_path)
    if parent_sha != _sha(long_plan.get("parent_pilot_plan_sha256"), "parent_pilot_plan_sha256"):
        _fail("parent pilot plan bytes do not match the long-run plan reference")
    return long_plan, long_sha, parent, parent_sha


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--long-run-plan", required=True, type=Path)
    parser.add_argument("--expected-long-plan-sha256", required=True)
    parser.add_argument("--parent-pilot-plan", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if not root.is_dir():
        _fail("repo root does not exist")
    inputs = {args.audit.resolve(), args.long_run_plan.resolve(), args.parent_pilot_plan.resolve()}
    if len(inputs) != 3 or args.output.resolve() in inputs:
        _fail("audit, plans, and output must use distinct paths")
    if args.output.exists():
        _fail(f"output already exists: {args.output}")
    long_plan, long_sha, parent, parent_sha = _verify_plan_chain(
        args.long_run_plan, args.parent_pilot_plan, args.expected_long_plan_sha256
    )
    mapping = _parent_contract(parent)
    _selection_contract(long_plan)
    immutable = _verify_config_files(root, parent, mapping)
    immutable.update(_verify_parent_source_files(root, parent))
    strict, task_config_sha = _load_strict_contract(root, parent)
    task_config_path = _repo_path(root, "configs/tasks/rj45-v1.json", "strict-v1 task config")
    immutable[task_config_path] = task_config_sha
    analyzer, analyzer_path, analyzer_sha = _load_verified_analyzer(root, parent)
    immutable[analyzer_path] = analyzer_sha
    stored_audit, _audit_raw, audit_sha = _load_json(args.audit)
    fresh_audit, raw_hashes = _verify_raw_audit(
        root, parent, stored_audit, analyzer, mapping, strict, task_config_sha
    )
    result = _rank_verified_audit(long_plan, fresh_audit)
    selected = result["decision"]["selected_group"]
    selected_mapping = mapping[selected]
    frozen = parent["frozen_inputs"]["sha256"]
    result["selected_profile"] = {
        "training_config": {
            "path": selected_mapping["training_config"],
            "sha256": frozen[selected_mapping["training_config"]],
        },
        "ppo_config": {
            "path": selected_mapping["ppo_config"],
            "sha256": frozen[selected_mapping["ppo_config"]],
        },
        "zero_init_actor_output": selected_mapping["zero"],
    }
    result["candidate_evidence"] = {
        label: {
            "raw_report": stored_audit["input_reports"][label],
            "checkpoint_bundle": {
                key: stored_audit["training_artifact_verification"][label][key]
                for key in ("checkpoint", "checkpoint_manifest", "training_provenance", "runner_config")
            },
        }
        for label in EXPECTED_GROUPS
    }
    result["inputs"] = {
        "stored_paired_audit": {"path": str(args.audit), "sha256": audit_sha},
        "long_run_plan": {
            "path": str(args.long_run_plan), "sha256": long_sha,
            "expected_sha256_provenance": "caller_supplied",
            "provenance_note": "This program does not authenticate the caller-supplied value's remote origin.",
        },
        "parent_pilot_plan": {"path": str(args.parent_pilot_plan), "sha256": parent_sha},
        "frozen_analyzer": {"path": str(analyzer_path.relative_to(root)), "sha256": analyzer_sha},
        "raw_evidence_sha256": dict(sorted(raw_hashes.items())),
        "immutable_inputs_sha256": {
            str(path.relative_to(root)): digest
            for path, digest in sorted(immutable.items(), key=lambda item: str(item[0]))
        },
    }
    for path, digest in ((args.audit, audit_sha), (args.long_run_plan, long_sha), (args.parent_pilot_plan, parent_sha), (analyzer_path, analyzer_sha)):
        if _sha_file(path) != digest:
            _fail(f"selection input changed during execution: {path}")
    for path, digest in immutable.items():
        if _sha_file(path) != digest:
            _fail(f"immutable source or config changed during selection: {path}")
    _atomic_create(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
