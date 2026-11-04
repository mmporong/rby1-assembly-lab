#!/usr/bin/env python3
"""Audit paired RJ45 evaluator reports without changing their success contract."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import tempfile
from typing import Any, Iterable


PAIR_KEYS = (
    "evaluation_start",
    "task_config_sha256",
    "base_evaluation_seed",
    "environment_index",
    "episode_index",
    "calibration_bias_level",
    "calibration_latent_case_seed",
    "calibration_latent_case_seed_formula",
)
ENVIRONMENT_PROVENANCE_FIELDS = (
    "task_config_sha256",
    "runtime_config_sha256",
    "assets_config_sha256",
    "build_report_sha256",
    "physics_dt_s",
    "control_decimation",
    "control_dt_s",
    "observation_noise",
    "generated_asset_sha256",
    "runtime_versions",
)
CONTROL_CONTRACT_FIELDS = (
    "action_mode",
    "observation_dimension",
    "action_application_mode",
    "policy_active_phases",
)
OUTCOME_METRICS = (
    "final_depth_m",
    "final_lateral_error_m",
    "final_orientation_error_rad",
    "final_dwell_s",
    "grip_contact_force_peak_n",
    "insertion_contact_force_peak_n",
    "insertion_contact_load_peak_n",
    "forbidden_contact_load_peak_n",
)
DEFAULT_TRANSLATION_EDGES_UM = (0.0, 50.0, 100.0, 150.0, math.inf)
DEFAULT_ROTATION_EDGES_DEG = (0.0, 1.0 / 12.0, 1.0 / 6.0, 0.25, math.inf)
NOMINAL_PROOF_HASH_FIELDS = ("report_sha256", "trace_sha256")
HELD_PROOF_HASH_FIELDS = (
    "artifact_sha256",
    "nominal_report_sha256",
    "nominal_trace_sha256",
)
POLICY_INITIALIZATION_FIELDS = {
    "mode",
    "zero_actor_output_layer",
    "actor_module_path",
    "final_layer_index",
    "final_layer_class",
    "input_dimension",
    "output_dimension",
    "pre_weight_sha256",
    "pre_bias_sha256",
    "post_weight_sha256",
    "post_bias_sha256",
    "weight_all_zero_after_initialization",
    "bias_all_zero_after_initialization",
    "deterministic_actor_test_input_count",
    "deterministic_actor_output_all_zero",
    "deterministic_actor_output_max_abs",
    "stochastic_policy_noise_modified",
    "method_reference",
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _finite(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        _fail(f"{name} must be finite and at least {minimum}")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer at least {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _vector(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        _fail(f"{name} must contain exactly three numbers")
    return tuple(_finite(component, f"{name}[{index}]") for index, component in enumerate(value))


def _proof_hashes(value: Any, fields: tuple[str, ...], name: str) -> dict[str, str]:
    proof = _object(value, name)
    return {field: _sha(proof.get(field), f"{name}.{field}") for field in fields}


def _validate_policy_initialization(value: Any, name: str) -> dict[str, Any]:
    metadata = _object(value, name)
    if set(metadata) != POLICY_INITIALIZATION_FIELDS:
        _fail(f"{name} fields are incomplete or unexpected")
    zero = metadata["zero_actor_output_layer"]
    if type(zero) is not bool:
        _fail(f"{name}.zero_actor_output_layer must be boolean")
    expected_mode = "residual_actor_final_linear_zero" if zero else "rsl_rl_default"
    if metadata["mode"] != expected_mode:
        _fail(f"{name}.mode disagrees with zero-init flag")
    if (
        metadata["actor_module_path"] != "runner.alg.policy.actor"
        or metadata["final_layer_class"] != "torch.nn.Linear"
        or type(metadata["final_layer_index"]) is not int
        or metadata["final_layer_index"] < 0
        or type(metadata["input_dimension"]) is not int
        or metadata["input_dimension"] <= 0
        or type(metadata["output_dimension"]) is not int
        or metadata["output_dimension"] not in (6, 12)
    ):
        _fail(f"{name} actor structure is invalid")
    for field in ("pre_weight_sha256", "pre_bias_sha256", "post_weight_sha256", "post_bias_sha256"):
        _sha(metadata[field], f"{name}.{field}")
    for field in (
        "weight_all_zero_after_initialization",
        "bias_all_zero_after_initialization",
        "deterministic_actor_output_all_zero",
        "stochastic_policy_noise_modified",
    ):
        if type(metadata[field]) is not bool:
            _fail(f"{name}.{field} must be boolean")
    if metadata["stochastic_policy_noise_modified"] is not False:
        _fail(f"{name} must not claim stochastic policy-noise mutation")
    if metadata["deterministic_actor_test_input_count"] != 3:
        _fail(f"{name} must record three deterministic actor tests")
    maximum = _finite(
        metadata["deterministic_actor_output_max_abs"],
        f"{name}.deterministic_actor_output_max_abs",
        minimum=0.0,
    )
    if zero:
        if metadata["output_dimension"] != 6 or not all(
            metadata[field]
            for field in (
                "weight_all_zero_after_initialization",
                "bias_all_zero_after_initialization",
                "deterministic_actor_output_all_zero",
            )
        ) or maximum != 0.0:
            _fail(f"{name} does not prove zero residual actor output")
        if metadata["method_reference"] != {
            "title": "Residual Policy Learning",
            "url": "https://arxiv.org/pdf/1812.06298",
            "section": "IV-A",
            "adaptation": "zero initialization of final residual actor mean layer only",
        }:
            _fail(f"{name} zero-init method reference is invalid")
    elif metadata["method_reference"] is not None:
        _fail(f"{name} default initialization must not claim a zero-init method")
    return metadata


def _validate_policy_initialization_lineage(
    reported: Any, training: Any, label: str
) -> dict[str, Any]:
    reported_initialization = _validate_policy_initialization(
        reported, f"{label}.checkpoint.policy_initialization"
    )
    training_initialization = _validate_policy_initialization(
        training, f"{label}.training_provenance.policy_initialization"
    )
    if reported_initialization != training_initialization:
        _fail(f"{label} evaluation policy initialization differs from training provenance")
    return reported_initialization


def _report_proof_state(report: dict[str, Any], label: str) -> tuple[str, dict[str, str] | None, dict[str, str] | None]:
    provenance = _object(report.get("provenance"), f"{label}.provenance")
    schema_version = report.get("rj45_evaluation_schema_version")
    nominal = provenance.get("rj45_nominal_evidence")
    held = provenance.get("held_grasp_reset_proof")
    reverified = provenance.get("evaluation_inputs_reverified_at_end")
    if schema_version is None and nominal is None and held is None and reverified is None:
        return "legacy", None, None
    if schema_version != 2:
        _fail(f"{label} current RJ45 report requires rj45_evaluation_schema_version=2")
    if nominal is None or held is None:
        _fail(f"{label} has partial RJ45 nominal/held proof")
    if reverified is not True:
        _fail(f"{label} evaluation inputs were not reverified at end")
    nominal_hashes = _proof_hashes(nominal, NOMINAL_PROOF_HASH_FIELDS, f"{label}.rj45_nominal_evidence")
    held_hashes = _proof_hashes(held, HELD_PROOF_HASH_FIELDS, f"{label}.held_grasp_reset_proof")
    if held_hashes["nominal_report_sha256"] != nominal_hashes["report_sha256"]:
        _fail(f"{label} held proof does not reference its nominal report")
    if held_hashes["nominal_trace_sha256"] != nominal_hashes["trace_sha256"]:
        _fail(f"{label} held proof does not reference its nominal trace")
    return "current", nominal_hashes, held_hashes


def _norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _load(path: Path) -> tuple[dict[str, Any], bytes, str]:
    def reject(constant: str) -> None:
        _fail(f"non-finite JSON constant in {path}: {constant}")

    try:
        raw = path.read_bytes()
        payload = _object(json.loads(raw.decode("utf-8"), parse_constant=reject), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid report JSON {path}: {error}") from error
    return payload, raw, hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_training_source_lineage(
    evaluation_source: dict[str, Any], training_source: dict[str, Any], label: str
) -> None:
    evaluation_source_paths = {
        path
        for path in evaluation_source
        if path.startswith("src/rby1_assembly/") and path.endswith(".py")
    }
    training_source_paths = {
        path
        for path in training_source
        if path.startswith("src/rby1_assembly/") and path.endswith(".py")
    }
    if not evaluation_source_paths or evaluation_source_paths != training_source_paths:
        _fail(f"{label} evaluation and training package source path sets differ")
    if any(
        evaluation_source[path] != training_source[path] for path in evaluation_source_paths
    ):
        _fail(f"{label} evaluation and training source hashes differ")
    _sha(training_source.get("scripts/train.py"), f"{label}.training scripts/train.py")


def _verify_path_hashes(expected: dict[Path, str]) -> None:
    for path, digest in expected.items():
        try:
            current = _sha_file(path)
        except OSError as error:
            raise ValueError(f"input changed or became unreadable during analysis: {path}") from error
        if current != digest:
            _fail(f"input changed during analysis: {path}")


def _verify_training_bundle(
    label: str,
    report: dict[str, Any],
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    training_provenance_path: Path,
    runner_config_path: Path,
    checkpoint_inspection_path: Path | None = None,
) -> dict[str, Any]:
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest, _manifest_bytes, manifest_sha = _load(manifest_path)
    training_provenance, _provenance_bytes, training_provenance_sha = _load(
        training_provenance_path
    )
    runner_config, _runner_bytes, runner_sha = _load(runner_config_path)
    report_provenance = report["provenance"]
    evidence = report_provenance["verified_checkpoint_evidence"]
    if manifest.get("schema_version") != 1:
        _fail(f"{label} checkpoint manifest schema_version must be 1")
    if manifest.get("training_completed") is not True:
        _fail(f"{label} checkpoint manifest does not mark training as completed")
    if checkpoint_sha != report_provenance["checkpoint_sha256"]:
        _fail(f"{label} checkpoint bytes do not match the evaluation report")
    if manifest_sha != evidence["checkpoint_manifest_sha256"]:
        _fail(f"{label} checkpoint manifest bytes do not match the evaluation report")
    if training_provenance_sha != evidence["training_provenance_sha256"]:
        _fail(f"{label} training provenance bytes do not match the evaluation report")
    if manifest.get("training_provenance_sha256") != training_provenance_sha:
        _fail(f"{label} manifest does not reference the supplied training provenance")
    checkpoints = _object(manifest.get("checkpoints"), f"{label}.manifest.checkpoints")
    if checkpoints.get(checkpoint_path.name) != checkpoint_sha:
        _fail(f"{label} manifest does not reference the supplied checkpoint")
    training_contract = _object(
        training_provenance.get("rj45_training_contract"),
        f"{label}.training_provenance.rj45_training_contract",
    )
    if training_contract.get("training_config_sha256") != report["rj45_training_config"]["sha256"]:
        _fail(f"{label} training provenance config hash differs from evaluation")
    if training_contract.get("held_grasp_reset_proof") != evidence["held_grasp_reset_proof"]:
        _fail(f"{label} training provenance held proof differs from checkpoint evidence")
    if training_provenance.get("rj45_nominal_evidence") != evidence["rj45_nominal_evidence"]:
        _fail(f"{label} training provenance nominal proof differs from checkpoint evidence")
    if report.get("rj45_evaluation_schema_version") == 2:
        _validate_policy_initialization_lineage(
            evidence.get("policy_initialization"),
            training_provenance.get("policy_initialization"),
            label,
        )
        initialization_status = "evaluation_and_training_provenance_match"
    else:
        initialization_status = "legacy_not_recorded"
    training_num_envs = _integer(
        training_provenance.get("num_envs"), f"{label}.training_provenance.num_envs", minimum=1
    )
    if training_num_envs != evidence["training_num_envs"]:
        _fail(f"{label} training environment count differs from checkpoint evidence")
    source_and_config = _object(
        training_provenance.get("source_and_config_sha256"),
        f"{label}.training_provenance.source_and_config_sha256",
    )
    if source_and_config.get(report["rj45_training_config"]["path"]) != report[
        "rj45_training_config"
    ]["sha256"]:
        _fail(f"{label} training config bytes are not pinned by training provenance")
    policy_matches = [
        digest
        for path, digest in source_and_config.items()
        if Path(path).name == Path(report_provenance["policy_config"]).name
    ]
    if policy_matches != [report_provenance["policy_config_sha256"]]:
        _fail(f"{label} policy config bytes are not uniquely pinned by training provenance")
    runner_matches = [
        digest for path, digest in source_and_config.items() if Path(path).name == runner_config_path.name
    ]
    if runner_matches != [runner_sha]:
        _fail(f"{label} runner config bytes are not uniquely pinned by training provenance")
    _validate_training_source_lineage(
        report_provenance["source_sha256"], source_and_config, label
    )
    seed = _integer(training_provenance.get("seed"), f"{label}.training seed")
    if runner_config.get("seed") != seed:
        _fail(f"{label} runner seed differs from training provenance")
    rollout_steps = _integer(
        runner_config.get("num_steps_per_env"), f"{label}.num_steps_per_env", minimum=1
    )
    planned_updates = _integer(
        runner_config.get("max_iterations"), f"{label}.max_iterations", minimum=1
    )
    policy = _object(runner_config.get("policy"), f"{label}.runner.policy")
    initial_noise_std = _finite(
        policy.get("init_noise_std"), f"{label}.policy.init_noise_std", minimum=0.0
    )
    result = {
        "status": "checkpoint_manifest_provenance_runner_bytes_verified",
        "unverified": [
            "source and non-runner config bytes referenced by training provenance",
            "preregistered experiment plan",
        ],
        "checkpoint": {"path": str(checkpoint_path), "sha256": checkpoint_sha},
        "checkpoint_manifest": {"path": str(manifest_path), "sha256": manifest_sha},
        "training_provenance": {
            "path": str(training_provenance_path),
            "sha256": training_provenance_sha,
        },
        "runner_config": {"path": str(runner_config_path), "sha256": runner_sha},
        "training_seed": seed,
        "training_num_envs": training_num_envs,
        "rollout_steps_per_env": rollout_steps,
        "planned_training_updates": planned_updates,
        "checkpoint_completed_updates": None,
        "checkpoint_completed_updates_verification": (
            "not_verified_checkpoint_payload_not_inspected"
        ),
        "initial_action_noise_std": initial_noise_std,
        "policy_initialization_status": initialization_status,
        "policy_initialization": evidence.get("policy_initialization"),
    }
    if checkpoint_inspection_path is not None:
        inspection, _inspection_bytes, inspection_sha = _load(checkpoint_inspection_path)
        if set(inspection) != {
            "schema_version",
            "status",
            "checkpoint",
            "planned_training_updates",
            "checkpoint_manifest",
            "training_provenance",
            "runner_config",
            "loader",
            "completion_derivation",
        }:
            _fail(f"{label} checkpoint inspection fields are incomplete or unexpected")
        if inspection.get("schema_version") != 2:
            _fail(f"{label} checkpoint inspection schema_version must be 2")
        if inspection.get("status") != "checkpoint_payload_and_lineage_verified":
            _fail(f"{label} checkpoint inspection status is not verified")
        inspected_checkpoint = _object(
            inspection.get("checkpoint"), f"{label}.checkpoint_inspection.checkpoint"
        )
        inspected_manifest = _object(
            inspection.get("checkpoint_manifest"),
            f"{label}.checkpoint_inspection.checkpoint_manifest",
        )
        inspected_provenance = _object(
            inspection.get("training_provenance"),
            f"{label}.checkpoint_inspection.training_provenance",
        )
        inspected_runner = _object(
            inspection.get("runner_config"), f"{label}.checkpoint_inspection.runner_config"
        )
        loader = _object(inspection.get("loader"), f"{label}.checkpoint_inspection.loader")
        if loader != {"map_location": "cpu", "weights_only": True}:
            _fail(f"{label} checkpoint inspection did not use the safe CPU loader contract")
        derivation = _object(
            inspection.get("completion_derivation"),
            f"{label}.checkpoint_inspection.completion_derivation",
        )
        training_script = _object(
            derivation.get("training_script"),
            f"{label}.checkpoint_inspection.completion_derivation.training_script",
        )
        if derivation.get("formula") != "completed_updates = checkpoint payload iter + 1":
            _fail(f"{label} checkpoint completion formula is not the reviewed contract")
        if (
            derivation.get("training_mode_basis")
            != "assumed_fresh_no_resume_requires_review_of_pinned_training_script"
            or derivation.get("checkpoint_scope") != "checkpoints created in this run directory"
            or derivation.get("rsl_rl_version") != "3.0.1"
        ):
            _fail(f"{label} checkpoint completion derivation prerequisites differ")
        if training_provenance.get("checkpoint_scope") != derivation.get("checkpoint_scope"):
            _fail(f"{label} inspection checkpoint scope differs from training provenance")
        training_runtime_versions = _object(
            training_provenance.get("runtime_versions"),
            f"{label}.training_provenance.runtime_versions",
        )
        if training_runtime_versions.get("rsl-rl-lib") != derivation.get("rsl_rl_version"):
            _fail(f"{label} inspection RSL-RL version differs from training provenance")
        if training_script.get("sha256") != source_and_config.get("scripts/train.py"):
            _fail(f"{label} checkpoint inspection training script differs from provenance")
        if inspected_checkpoint.get("sha256") != checkpoint_sha:
            _fail(f"{label} checkpoint inspection refers to different checkpoint bytes")
        if inspected_manifest.get("sha256") != manifest_sha:
            _fail(f"{label} checkpoint inspection refers to different manifest bytes")
        if inspected_manifest.get("training_completed") is not True:
            _fail(f"{label} checkpoint inspection does not verify completed training manifest")
        if inspected_provenance.get("sha256") != training_provenance_sha:
            _fail(f"{label} checkpoint inspection refers to different training provenance bytes")
        if inspected_runner.get("sha256") != runner_sha:
            _fail(f"{label} checkpoint inspection refers to different runner config bytes")
        if inspected_provenance.get("seed") != seed or inspected_runner.get("seed") != seed:
            _fail(f"{label} checkpoint inspection seed lineage differs")
        if inspection.get("planned_training_updates") != planned_updates:
            _fail(f"{label} checkpoint inspection planned updates differ from runner config")
        filename_iteration = _integer(
            inspected_checkpoint.get("filename_iteration"),
            f"{label}.checkpoint_inspection.filename_iteration",
        )
        payload_iteration = _integer(
            inspected_checkpoint.get("payload_iteration"),
            f"{label}.checkpoint_inspection.payload_iteration",
        )
        completed_updates = _integer(
            inspected_checkpoint.get("completed_updates"),
            f"{label}.checkpoint_inspection.completed_updates",
            minimum=1,
        )
        checkpoint_name_match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_path.name)
        if checkpoint_name_match is None:
            _fail(f"{label} checkpoint filename must use model_N.pt")
        actual_filename_iteration = int(checkpoint_name_match.group(1))
        if filename_iteration != actual_filename_iteration:
            _fail(f"{label} checkpoint inspection index differs from actual checkpoint filename")
        if filename_iteration != payload_iteration or completed_updates != payload_iteration + 1:
            _fail(f"{label} checkpoint inspection iteration fields disagree")
        if completed_updates > planned_updates:
            _fail(f"{label} checkpoint completed updates exceed planned updates")
        result["checkpoint_completed_updates"] = completed_updates
        result["checkpoint_completed_updates_verification"] = (
            "supplied_inspection_report_consistent_payload_not_rechecked"
        )
        result["checkpoint_inspection"] = {
            "path": str(checkpoint_inspection_path),
            "sha256": inspection_sha,
            "status": inspection["status"],
        }
    return result


def _canonical_condition(condition: dict[str, Any]) -> str:
    if set(condition) != set(PAIR_KEYS):
        _fail("paired_condition fields are incomplete or unexpected")
    return json.dumps(condition, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_checkpoint(report: dict[str, Any], label: str) -> None:
    if report.get("baseline_mode") != "learned":
        return
    provenance = _object(report.get("provenance"), f"{label}.provenance")
    _sha(provenance.get("checkpoint_sha256"), f"{label}.checkpoint_sha256")
    if not isinstance(provenance.get("checkpoint"), str) or not provenance["checkpoint"]:
        _fail(f"{label} learned report must identify its checkpoint")
    evidence = _object(
        provenance.get("verified_checkpoint_evidence"),
        f"{label}.verified_checkpoint_evidence",
    )
    expected_fields = {
        "training_provenance_sha256",
        "checkpoint_manifest_sha256",
        "runtime_contract_verified_at_start",
        "rj45_nominal_evidence",
        "held_grasp_reset_proof",
        "training_num_envs",
    }
    if report.get("rj45_evaluation_schema_version") == 2:
        expected_fields.add("policy_initialization")
    if set(evidence) != expected_fields:
        _fail(f"{label} verified checkpoint evidence fields are incomplete or unexpected")
    for field in ("training_provenance_sha256", "checkpoint_manifest_sha256"):
        _sha(evidence.get(field), f"{label}.{field}")
    if evidence.get("runtime_contract_verified_at_start") is not True:
        _fail(f"{label} checkpoint runtime contract was not verified")
    _integer(evidence.get("training_num_envs"), f"{label}.training_num_envs", minimum=1)
    nominal_hashes = _proof_hashes(
        evidence.get("rj45_nominal_evidence"),
        NOMINAL_PROOF_HASH_FIELDS,
        f"{label}.checkpoint.rj45_nominal_evidence",
    )
    held_hashes = _proof_hashes(
        evidence.get("held_grasp_reset_proof"),
        HELD_PROOF_HASH_FIELDS,
        f"{label}.checkpoint.held_grasp_reset_proof",
    )
    if (
        held_hashes["nominal_report_sha256"] != nominal_hashes["report_sha256"]
        or held_hashes["nominal_trace_sha256"] != nominal_hashes["trace_sha256"]
    ):
        _fail(f"{label} checkpoint held proof does not reference checkpoint nominal proof")
    if "policy_initialization" in evidence:
        initialization = _validate_policy_initialization(
            evidence["policy_initialization"],
            f"{label}.checkpoint.policy_initialization",
        )
        expected_output = (
            6 if report["rj45_training_config"].get("action_mode") == "right_only_6d" else 12
        )
        if initialization["output_dimension"] != expected_output:
            _fail(f"{label} policy initialization output dimension differs from action contract")


def _validate_report(
    report: dict[str, Any], label: str, expected_attempts: int
) -> dict[str, dict[str, Any]]:
    requested = _integer(report.get("requested_attempts"), f"{label}.requested_attempts", minimum=1)
    if requested != expected_attempts:
        _fail(f"{label} requested_attempts={requested}, expected {expected_attempts}")
    if report.get("attempts") != requested:
        _fail(f"{label} attempts must equal requested_attempts")
    outcomes = report.get("episode_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != requested:
        _fail(f"{label} must contain every requested episode outcome")
    if report.get("evaluation_start") != "held":
        _fail(f"{label} evaluation_start must be held")
    _integer(report.get("seed"), f"{label}.seed")
    num_envs = _integer(report.get("evaluation_num_envs"), f"{label}.evaluation_num_envs", minimum=1)
    raw_quota = report.get("per_environment_episode_quota")
    if not isinstance(raw_quota, list) or len(raw_quota) != num_envs:
        _fail(f"{label} per-environment quota must have one entry per environment")
    quotas = [
        _integer(value, f"{label}.per_environment_episode_quota[{index}]", minimum=1)
        for index, value in enumerate(raw_quota)
    ]
    if requested != sum(quotas):
        _fail(f"{label} requested attempts do not equal the sum of environment quotas")
    provenance = _object(report.get("provenance"), f"{label}.provenance")
    if provenance.get("input_immutability_verified") is not True:
        _fail(f"{label} inputs were not verified unchanged")
    source = _object(provenance.get("source_sha256"), f"{label}.source_sha256")
    if not source:
        _fail(f"{label} source hashes are empty")
    for path, digest in source.items():
        if not isinstance(path, str) or not path:
            _fail(f"{label} source hash path is invalid")
        _sha(digest, f"{label}.source_sha256[{path}]")
    for field in ENVIRONMENT_PROVENANCE_FIELDS:
        if field not in provenance:
            _fail(f"{label} provenance is missing {field}")
    for field in ENVIRONMENT_PROVENANCE_FIELDS[:4]:
        _sha(provenance[field], f"{label}.provenance.{field}")
    generated_assets = _object(
        provenance["generated_asset_sha256"], f"{label}.generated_asset_sha256"
    )
    if not generated_assets:
        _fail(f"{label} generated asset hashes are empty")
    for path, digest in generated_assets.items():
        if not isinstance(path, str) or not path:
            _fail(f"{label} generated asset path is invalid")
        _sha(digest, f"{label}.generated_asset_sha256[{path}]")
    runtime_versions = _object(provenance["runtime_versions"], f"{label}.runtime_versions")
    if not runtime_versions or any(
        not isinstance(name, str) or not name or not isinstance(version, str) or not version
        for name, version in runtime_versions.items()
    ):
        _fail(f"{label} runtime versions are invalid")
    if not isinstance(provenance.get("policy_config"), str) or not provenance["policy_config"]:
        _fail(f"{label} policy config path is missing")
    _sha(provenance.get("policy_config_sha256"), f"{label}.policy_config_sha256")
    _finite(provenance["physics_dt_s"], f"{label}.physics_dt_s", minimum=0.0)
    _integer(provenance["control_decimation"], f"{label}.control_decimation", minimum=1)
    _finite(provenance["control_dt_s"], f"{label}.control_dt_s", minimum=0.0)
    config = _object(report.get("rj45_training_config"), f"{label}.rj45_training_config")
    _sha(config.get("sha256"), f"{label}.rj45_training_config.sha256")
    if not isinstance(config.get("path"), str) or not config["path"]:
        _fail(f"{label} training config path is missing")
    if not isinstance(config.get("action_mode"), str) or not config["action_mode"]:
        _fail(f"{label} action_mode is missing")
    _integer(config.get("observation_dimension"), f"{label}.observation_dimension", minimum=1)
    if (
        not isinstance(config.get("action_application_mode"), str)
        or not config["action_application_mode"]
    ):
        _fail(f"{label} action_application_mode is missing")
    if (
        not isinstance(config.get("policy_active_phases"), list)
        or not config["policy_active_phases"]
        or any(not isinstance(phase, str) or not phase for phase in config["policy_active_phases"])
    ):
        _fail(f"{label} policy_active_phases is invalid")
    bias_level = _integer(
        config.get("calibration_bias_level"), f"{label}.calibration_bias_level"
    )
    if report.get("paired_condition_key") != ",".join(PAIR_KEYS):
        _fail(f"{label} paired condition key is not canonical")
    task_model = _object(report.get("task_model"), f"{label}.task_model")
    parallel_evidence = _object(
        report.get("parallel_contact_evidence"), f"{label}.parallel_contact_evidence"
    )
    if not parallel_evidence:
        _fail(f"{label} parallel contact evidence is empty")
    stability_limits = _object(
        task_model.get("evaluation_stability_limits"),
        f"{label}.task_model.evaluation_stability_limits",
    )
    assembled_depth = _finite(
        stability_limits.get("assembled_depth_m", 0.012),
        f"{label}.assembled_depth_m",
        minimum=0.0,
    )
    max_lateral = _finite(
        stability_limits.get("max_lateral_error_m", 0.00005),
        f"{label}.max_lateral_error_m",
        minimum=0.0,
    )
    max_orientation = _finite(
        stability_limits.get("max_orientation_error_rad", math.radians(0.5)),
        f"{label}.max_orientation_error_rad",
        minimum=0.0,
    )
    required_dwell = _finite(
        stability_limits.get("required_continuous_dwell_s"),
        f"{label}.required_continuous_dwell_s",
        minimum=0.0,
    )

    indexed: dict[str, dict[str, Any]] = {}
    invalid_reasons: list[str] = []
    attempt_indices: list[int] = []
    environment_counts: Counter[int] = Counter()
    for position, raw_outcome in enumerate(outcomes):
        outcome = _object(raw_outcome, f"{label}.episode_outcomes[{position}]")
        attempt_indices.append(_integer(outcome.get("attempt_index"), f"{label}.attempt_index"))
        environment_index = _integer(
            outcome.get("environment_index"), f"{label}.environment_index"
        )
        if environment_index >= num_envs:
            _fail(f"{label} outcome references environment {environment_index} outside {num_envs}")
        environment_counts[environment_index] += 1
        if type(outcome.get("valid")) is not bool or type(outcome.get("success")) is not bool:
            _fail(f"{label} valid and success flags must be booleans")
        if outcome["success"] and not outcome["valid"]:
            _fail(f"{label} invalid outcome cannot be successful")
        reason = outcome.get("invalid_reason")
        if outcome["valid"]:
            if reason is not None:
                _fail(f"{label} valid outcome must have invalid_reason=null")
        elif not isinstance(reason, str) or not reason:
            _fail(f"{label} invalid outcome must identify a reason")
        else:
            invalid_reasons.append(reason)
        if type(outcome.get("held_initialization_passed")) is not bool:
            _fail(f"{label} held_initialization_passed must be boolean")
        policy_steps = _integer(
            outcome.get("valid_policy_input_step_count"),
            f"{label}.valid_policy_input_step_count",
        )
        if outcome["valid"] and (
            not outcome["held_initialization_passed"] or policy_steps == 0
        ):
            _fail(f"{label} valid outcome lacks held-state or policy-input evidence")
        if not isinstance(outcome.get("termination_reason"), str) or not outcome["termination_reason"]:
            _fail(f"{label} termination reason is missing")
        condition = _object(outcome.get("paired_condition"), f"{label}.paired_condition")
        canonical = _canonical_condition(condition)
        if outcome.get("paired_condition_id") != canonical:
            _fail(f"{label} paired_condition_id is not canonical")
        if canonical in indexed:
            _fail(f"{label} contains duplicate paired condition {canonical}")
        if condition["evaluation_start"] != report["evaluation_start"]:
            _fail(f"{label} paired evaluation_start mismatch")
        if condition["task_config_sha256"] != provenance["task_config_sha256"]:
            _fail(f"{label} paired task config mismatch")
        if condition["base_evaluation_seed"] != report["seed"]:
            _fail(f"{label} paired seed mismatch")
        if condition["environment_index"] != environment_index:
            _fail(f"{label} paired environment mismatch")
        if condition["episode_index"] != outcome.get("episode_index"):
            _fail(f"{label} paired episode mismatch")
        if condition["calibration_bias_level"] != bias_level:
            _fail(f"{label} paired calibration level mismatch")
        translation = _vector(
            outcome.get("calibration_bias_translation_entry_m"),
            f"{label}.calibration_bias_translation_entry_m",
        )
        rotation = _vector(
            outcome.get("calibration_bias_rotation_vector_entry_rad"),
            f"{label}.calibration_bias_rotation_vector_entry_rad",
        )
        for metric in OUTCOME_METRICS:
            if outcome.get(metric) is not None:
                _finite(outcome[metric], f"{label}.{metric}")
        if outcome["success"]:
            if outcome["termination_reason"] != "independent_success":
                _fail(f"{label} successful outcome has a non-success termination reason")
            if any(outcome.get(metric) is None for metric in OUTCOME_METRICS[:4]):
                _fail(f"{label} successful outcome is missing final assembly metrics")
            if (
                outcome["final_depth_m"] < assembled_depth
                or outcome["final_lateral_error_m"] > max_lateral
                or outcome["final_orientation_error_rad"] > max_orientation
                or outcome["final_dwell_s"] < required_dwell
            ):
                _fail(f"{label} successful outcome violates the reported success limits")
        elif outcome["termination_reason"] == "independent_success":
            _fail(f"{label} failed outcome has an independent_success termination reason")
        outcome["_audit_bias_translation"] = translation
        outcome["_audit_bias_rotation"] = rotation
        indexed[canonical] = outcome

    if sorted(attempt_indices) != list(range(requested)):
        _fail(f"{label} attempt indices must contain 0..{requested - 1} exactly once")
    if any(environment_counts[index] != quotas[index] for index in range(num_envs)):
        _fail(f"{label} environment episode counts do not match the declared quota")
    accepted = report.get("per_environment_accepted_attempt_count")
    if accepted != [environment_counts[index] for index in range(num_envs)]:
        _fail(f"{label} accepted attempt counts do not match episode outcomes")
    if report.get("invalid_episodes") != len(invalid_reasons):
        _fail(f"{label} invalid episode aggregate is inconsistent")
    if report.get("invalid_reasons") != dict(Counter(invalid_reasons)):
        _fail(f"{label} invalid reason aggregate is inconsistent")
    valid_count = sum(outcome["valid"] for outcome in indexed.values())
    success_count = sum(
        outcome["valid"] and outcome["success"] for outcome in indexed.values()
    )
    if report.get("valid_episodes") != valid_count:
        _fail(f"{label} valid_episodes aggregate is inconsistent")
    if report.get("valid_successes") != success_count:
        _fail(f"{label} valid_successes aggregate is inconsistent")
    if report.get("valid_failures") != valid_count - success_count:
        _fail(f"{label} valid_failures aggregate is inconsistent")
    expected_rate = success_count / valid_count if valid_count else None
    if report.get("success_rate_valid_episodes") != expected_rate:
        _fail(f"{label} success_rate_valid_episodes aggregate is inconsistent")
    _validate_checkpoint(report, label)
    return indexed


def _validate_common_environment(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    for field in (
        "evaluation_start",
        "evaluation_horizon_s",
        "evaluation_num_envs",
        "per_environment_episode_quota",
        "seed",
        "task_id",
        "observation_source",
        "parallel_contact_evidence",
    ):
        if reference.get(field) != candidate.get(field):
            _fail(f"{label} differs from baseline for {field}")
    baseline_model = _object(reference.get("task_model"), "baseline.task_model")
    candidate_model = _object(candidate.get("task_model"), f"{label}.task_model")
    for field in (
        "scope",
        "fixed_base",
        "num_joints",
        "mating_frame",
        "residual_action_frame",
        "pose_or_force_injection",
    ):
        if baseline_model.get(field) != candidate_model.get(field):
            _fail(f"{label} robot/task model differs for {field}")
    baseline_limits = _object(
        baseline_model.get("evaluation_stability_limits"),
        "baseline.task_model.evaluation_stability_limits",
    )
    candidate_limits = _object(
        candidate_model.get("evaluation_stability_limits"),
        f"{label}.task_model.evaluation_stability_limits",
    )
    if candidate_limits != baseline_limits:
        _fail(f"{label} evaluation stability limits differ from baseline")
    left = reference["provenance"]
    right = candidate["provenance"]
    if left["source_sha256"] != right["source_sha256"]:
        _fail(f"{label} evaluation source lineage differs from baseline")
    for field in ENVIRONMENT_PROVENANCE_FIELDS:
        if left[field] != right[field]:
            _fail(f"{label} environment provenance differs for {field}")
    baseline_contract = reference["rj45_training_config"]
    candidate_contract = candidate["rj45_training_config"]
    for field in CONTROL_CONTRACT_FIELDS:
        if baseline_contract.get(field) != candidate_contract.get(field):
            _fail(f"{label} policy input/control contract differs for {field}")


def _outcome_record(outcome: dict[str, Any]) -> dict[str, Any]:
    result = {
        "valid": outcome["valid"],
        "success": outcome["valid"] and outcome["success"],
        "termination_reason": outcome["termination_reason"],
        "invalid_reason": outcome.get("invalid_reason"),
    }
    for metric in OUTCOME_METRICS:
        result[metric] = outcome.get(metric)
    return result


def _summary(report: dict[str, Any], outcomes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = list(outcomes.values())
    successes = sum(outcome["valid"] and outcome["success"] for outcome in ordered)
    checkpoint_evidence = report["provenance"].get("verified_checkpoint_evidence")
    return {
        "baseline_mode": report.get("baseline_mode"),
        "training_config": report["rj45_training_config"],
        "checkpoint": report["provenance"].get("checkpoint"),
        "checkpoint_sha256": report["provenance"].get("checkpoint_sha256"),
        "verified_checkpoint_evidence": checkpoint_evidence,
        "policy_initialization": (
            checkpoint_evidence.get("policy_initialization")
            if isinstance(checkpoint_evidence, dict)
            else None
        ),
        "policy_config": report["provenance"]["policy_config"],
        "policy_config_sha256": report["provenance"]["policy_config_sha256"],
        "attempts": len(ordered),
        "valid_episodes": sum(outcome["valid"] for outcome in ordered),
        "invalid_episodes": sum(not outcome["valid"] for outcome in ordered),
        "successes_all_attempts": successes,
        "success_rate_all_attempts": successes / len(ordered),
        "termination_counts": dict(sorted(Counter(
            outcome["termination_reason"] for outcome in ordered
        ).items())),
    }


def _percentile(sorted_values: list[float], probability: float) -> float:
    if not sorted_values:
        _fail("cannot compute a percentile of an empty sample")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _bootstrap_interval(deltas: list[int], resamples: int, seed: int) -> list[float]:
    _integer(resamples, "bootstrap_resamples", minimum=1)
    rng = random.Random(seed)
    count = len(deltas)
    means = [sum(deltas[rng.randrange(count)] for _ in range(count)) / count for _ in range(resamples)]
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _validate_edges(edges: tuple[float, ...], name: str) -> None:
    if len(edges) < 2 or edges[0] != 0.0 or not math.isinf(edges[-1]):
        _fail(f"{name} edges must start at zero and end at infinity")
    if any(left >= right for left, right in zip(edges, edges[1:])):
        _fail(f"{name} edges must be strictly increasing")


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for index, upper in enumerate(edges[1:]):
        if value < upper:
            return index
    raise AssertionError("infinite final edge must contain every finite value")


def _format_edge(value: float) -> float | str:
    return "inf" if math.isinf(value) else value


def _binned_coverage(
    keys: list[str],
    baseline: dict[str, dict[str, Any]],
    results: dict[str, dict[str, dict[str, Any]]],
    edges: tuple[float, ...],
    *,
    kind: str,
) -> dict[str, Any]:
    _validate_edges(edges, kind)
    if kind == "translation":
        values = [_norm(baseline[key]["_audit_bias_translation"]) * 1e6 for key in keys]
        unit = "um"
    else:
        values = [math.degrees(_norm(baseline[key]["_audit_bias_rotation"])) for key in keys]
        unit = "deg"
    bins = []
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        members = [position for position, value in enumerate(values) if _bin_index(value, edges) == index]
        bin_result: dict[str, Any] = {
            "lower_inclusive": _format_edge(lower),
            "upper_exclusive": _format_edge(upper),
            "sample_count": len(members),
            "results": {},
        }
        for label, indexed in results.items():
            successes = sum(
                indexed[keys[position]]["valid"] and indexed[keys[position]]["success"]
                for position in members
            )
            bin_result["results"][label] = {
                "successes": successes,
                "success_rate_all_attempts": successes / len(members) if members else None,
            }
        bins.append(bin_result)
    return {"unit": unit, "norm": "euclidean_3d", "edges": [_format_edge(v) for v in edges], "bins": bins}


def analyze_reports(
    baseline: dict[str, Any],
    conditions: dict[str, dict[str, Any]],
    *,
    expected_attempts: int,
    bootstrap_resamples: int = 100_000,
    bootstrap_seed: int = 20261029,
    translation_edges_um: tuple[float, ...] = DEFAULT_TRANSLATION_EDGES_UM,
    rotation_edges_deg: tuple[float, ...] = DEFAULT_ROTATION_EDGES_DEG,
    allow_legacy_unproven_held: bool = False,
    baseline_label: str = "baseline",
) -> dict[str, Any]:
    if not conditions:
        _fail("at least one comparison condition is required")
    if baseline.get("baseline_mode") != "biased_zero":
        _fail("baseline report must use baseline_mode=biased_zero")
    if any(report.get("baseline_mode") != "learned" for report in conditions.values()):
        _fail("every comparison condition must use baseline_mode=learned")
    if not baseline_label or any(character.isspace() for character in baseline_label):
        _fail("baseline label must be non-empty and contain no whitespace")
    if baseline_label in conditions or len(conditions) != len(set(conditions)):
        _fail("comparison labels must be unique and differ from the baseline label")
    indexes = {
        baseline_label: _validate_report(baseline, baseline_label, expected_attempts)
    }
    for label, report in conditions.items():
        if not label or any(character.isspace() for character in label):
            _fail("comparison labels must be non-empty and contain no whitespace")
        indexes[label] = _validate_report(report, label, expected_attempts)
        _validate_common_environment(baseline, report, label)
    reports = {baseline_label: baseline, **conditions}
    proof_states = {
        label: _report_proof_state(report, label) for label, report in reports.items()
    }
    state_names = {state[0] for state in proof_states.values()}
    if len(state_names) != 1:
        _fail("current and legacy nominal/held proof reports cannot be mixed")
    proof_mode = next(iter(state_names))
    if proof_mode == "legacy" and not allow_legacy_unproven_held:
        _fail("legacy reports without nominal/held proof require --allow-legacy-unproven-held")
    if proof_mode == "current":
        nominal_reference = proof_states[baseline_label][1]
        held_reference = proof_states[baseline_label][2]
        for label, (_state, nominal, held) in proof_states.items():
            if nominal != nominal_reference or held != held_reference:
                _fail(f"{label} nominal/held proof hashes differ from baseline")
        for label, report in conditions.items():
            checkpoint_evidence = report["provenance"]["verified_checkpoint_evidence"]
            checkpoint_nominal = _proof_hashes(
                checkpoint_evidence.get("rj45_nominal_evidence"),
                NOMINAL_PROOF_HASH_FIELDS,
                f"{label}.checkpoint.rj45_nominal_evidence",
            )
            checkpoint_held = _proof_hashes(
                checkpoint_evidence.get("held_grasp_reset_proof"),
                HELD_PROOF_HASH_FIELDS,
                f"{label}.checkpoint.held_grasp_reset_proof",
            )
            if checkpoint_nominal != nominal_reference or checkpoint_held != held_reference:
                _fail(f"{label} checkpoint evidence does not match evaluation nominal/held proof")
    baseline_keys = set(indexes[baseline_label])
    for label, index in indexes.items():
        if set(index) != baseline_keys:
            missing = sorted(baseline_keys - set(index))
            extra = sorted(set(index) - baseline_keys)
            _fail(f"{label} paired cases differ: missing={len(missing)}, extra={len(extra)}")
    keys = sorted(baseline_keys)
    for key in keys:
        expected_translation = indexes[baseline_label][key]["_audit_bias_translation"]
        expected_rotation = indexes[baseline_label][key]["_audit_bias_rotation"]
        for label, index in indexes.items():
            if (
                index[key]["_audit_bias_translation"] != expected_translation
                or index[key]["_audit_bias_rotation"] != expected_rotation
            ):
                _fail(f"{label} calibration bias differs for paired condition {key}")

    success_arrays = {
        label: [int(index[key]["valid"] and index[key]["success"]) for key in keys]
        for label, index in indexes.items()
    }
    comparisons: dict[str, Any] = {}
    labels = list(indexes)
    for candidate_position in range(1, len(labels)):
        candidate_label = labels[candidate_position]
        for reference_position in range(candidate_position):
            reference_label = labels[reference_position]
            reference_success = success_arrays[reference_label]
            candidate_success = success_arrays[candidate_label]
            deltas = [
                candidate - reference
                for candidate, reference in zip(candidate_success, reference_success)
            ]
            contrast = f"{candidate_label}_minus_{reference_label}"
            comparisons[contrast] = {
                "candidate": candidate_label,
                "reference": reference_label,
                "paired_success_rate_difference": sum(deltas) / len(deltas),
                "paired_bootstrap_95_percentile_interval": _bootstrap_interval(
                    deltas, bootstrap_resamples, bootstrap_seed
                ),
                "candidate_only_successes": sum(
                    c == 1 and r == 0 for r, c in zip(reference_success, candidate_success)
                ),
                "reference_only_successes": sum(
                    r == 1 and c == 0 for r, c in zip(reference_success, candidate_success)
                ),
            }

    learned_evidence = [
        (label, conditions[label]["provenance"]["verified_checkpoint_evidence"])
        for label in conditions
        if conditions[label].get("baseline_mode") == "learned"
    ]
    if learned_evidence:
        evidence_reference = learned_evidence[0][1]
        for label, evidence in learned_evidence[1:]:
            for field in ("rj45_nominal_evidence", "held_grasp_reset_proof"):
                if evidence[field] != evidence_reference[field]:
                    _fail(f"{label} learned checkpoint {field} differs from the other learned conditions")

    case_records = []
    for key in keys:
        baseline_outcome = indexes[baseline_label][key]
        case_records.append({
            "paired_condition_id": key,
            "paired_condition": baseline_outcome["paired_condition"],
            "calibration_bias_translation_entry_m": list(baseline_outcome["_audit_bias_translation"]),
            "calibration_bias_rotation_vector_entry_rad": list(baseline_outcome["_audit_bias_rotation"]),
            "translation_bias_norm_um": _norm(baseline_outcome["_audit_bias_translation"]) * 1e6,
            "rotation_bias_norm_deg": math.degrees(_norm(baseline_outcome["_audit_bias_rotation"])),
            "results": {label: _outcome_record(index[key]) for label, index in indexes.items()},
        })
    return {
        "schema_version": 2,
        "schema_compatibility": {
            "version_1_training_updates_interpretation": (
                "runner max_iterations copied as a planned limit; it did not verify checkpoint completion"
            ),
            "version_2": (
                "planned_training_updates is separate from checkpoint_completed_updates and its verification status"
            ),
        },
        "scope": {
            "verification_status": (
                "report_proof_hashes_consistent_inputs_not_rehashed"
                if proof_mode == "current"
                else "legacy_unproven_from_reports"
            ),
            "description": (
                "paired RJ45 evaluation audit with mutually consistent reported nominal/held proof hashes"
                if proof_mode == "current"
                else "historical paired RJ45 evaluation audit; the reports do not prove a common held artifact"
            ),
            "data_limitations": [
                (
                    "Calibration-bias cases report the same nominal/held proof hashes."
                    if proof_mode == "current"
                    else "Historical reports describe a reused held state, but omit hashes needed to prove it."
                ),
                "They are not independent grasp states, demonstrations, or sensor measurements.",
                "Invalid episodes remain in the all-attempt denominator.",
                "Bootstrap intervals are conditional on one evaluation seed and one reported held state.",
                "Report proof hashes are consistency claims until referenced files are rehashed.",
                (
                    "Policy initialization is report provenance until a training bundle is byte-verified."
                    if proof_mode == "current"
                    else "Legacy reports do not record policy initialization provenance."
                ),
            ],
        },
        "experiment": {
            "task_id": baseline["task_id"],
            "task_model": {
                field: baseline["task_model"].get(field)
                for field in ("scope", "fixed_base", "num_joints")
            },
            "evaluation_start": baseline["evaluation_start"],
            "evaluation_horizon_s": baseline["evaluation_horizon_s"],
            "evaluation_num_envs": baseline["evaluation_num_envs"],
            "per_environment_episode_quota": baseline["per_environment_episode_quota"],
            "requested_attempts": expected_attempts,
            "seed": baseline["seed"],
            "observation_source": baseline.get("observation_source"),
            "policy_input_control_contract": {
                field: baseline["rj45_training_config"].get(field)
                for field in CONTROL_CONTRACT_FIELDS
            },
            "source_sha256": baseline["provenance"]["source_sha256"],
            "environment_provenance": {
                field: baseline["provenance"][field] for field in ENVIRONMENT_PROVENANCE_FIELDS
            },
            "reported_configuration_identifiers": {
                label: {
                    "training_config_path": report["rj45_training_config"]["path"],
                    "training_config_sha256": report["rj45_training_config"]["sha256"],
                    "policy_config": report["provenance"].get("policy_config"),
                    "policy_config_sha256": report["provenance"].get("policy_config_sha256"),
                    "policy_initialization": (
                        report["provenance"]["verified_checkpoint_evidence"].get(
                            "policy_initialization"
                        )
                        if isinstance(
                            report["provenance"].get("verified_checkpoint_evidence"), dict
                        )
                        else None
                    ),
                }
                for label, report in reports.items()
            },
        },
        "bootstrap": {
            "method": "paired case resampling with replacement",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "interval": "2.5 and 97.5 empirical percentiles",
        },
        "results": {
            label: _summary(report, indexes[label]) for label, report in reports.items()
        },
        "paired_contrasts": comparisons,
        "evaluation_bias_coverage": {
            "translation": _binned_coverage(
                keys, indexes[baseline_label], indexes, translation_edges_um, kind="translation"
            ),
            "rotation": _binned_coverage(
                keys, indexes[baseline_label], indexes, rotation_edges_deg, kind="rotation"
            ),
        },
        "paired_cases": case_records,
    }


def _parse_condition(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("condition must use LABEL=PATH")
    return label, Path(raw_path)


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        _fail(f"output already exists and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"output already exists and will not be overwritten: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--condition", required=True, action="append", type=_parse_condition)
    parser.add_argument("--checkpoint-artifact", action="append", type=_parse_condition, default=[])
    parser.add_argument("--checkpoint-manifest", action="append", type=_parse_condition, default=[])
    parser.add_argument("--training-provenance", action="append", type=_parse_condition, default=[])
    parser.add_argument("--runner-config", action="append", type=_parse_condition, default=[])
    parser.add_argument("--checkpoint-inspection", action="append", type=_parse_condition, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-attempts", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261029)
    parser.add_argument(
        "--allow-legacy-unproven-held",
        action="store_true",
        help="allow historical reports that lack evaluator-level nominal/held proof",
    )
    args = parser.parse_args(argv)
    condition_paths = dict(args.condition)
    if len(condition_paths) != len(args.condition):
        _fail("condition labels must be unique")
    if args.baseline_label in condition_paths:
        _fail("baseline label must differ from condition labels")
    all_paths = {args.baseline_label: args.baseline, **condition_paths}
    bundle_arguments = {
        "checkpoint": dict(args.checkpoint_artifact),
        "manifest": dict(args.checkpoint_manifest),
        "training_provenance": dict(args.training_provenance),
        "runner_config": dict(args.runner_config),
        "checkpoint_inspection": dict(args.checkpoint_inspection),
    }
    for argument_name, mapping in bundle_arguments.items():
        raw_values = getattr(args, {
            "checkpoint": "checkpoint_artifact",
            "manifest": "checkpoint_manifest",
            "training_provenance": "training_provenance",
            "runner_config": "runner_config",
            "checkpoint_inspection": "checkpoint_inspection",
        }[argument_name])
        if len(mapping) != len(raw_values):
            _fail(f"duplicate label in --{argument_name.replace('_', '-')}")
        unknown = set(mapping) - set(condition_paths)
        if unknown:
            _fail(f"unknown training bundle labels: {sorted(unknown)}")
    if len({path.resolve() for path in all_paths.values()}) != len(all_paths):
        _fail("each input report path must be unique")
    every_input_path = list(all_paths.values()) + [
        path for mapping in bundle_arguments.values() for path in mapping.values()
    ]
    if args.output.resolve() in {path.resolve() for path in every_input_path}:
        _fail("output path must not overwrite any report or training-bundle input")
    loaded = {label: (*_load(path), path) for label, path in all_paths.items()}
    input_hashes = {entry[3]: entry[2] for entry in loaded.values()}
    for path in every_input_path[len(all_paths):]:
        input_hashes[path] = _sha_file(path)
    result = analyze_reports(
        loaded[args.baseline_label][0],
        {label: loaded[label][0] for label in condition_paths},
        expected_attempts=args.expected_attempts,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
        allow_legacy_unproven_held=args.allow_legacy_unproven_held,
        baseline_label=args.baseline_label,
    )
    bundle_verification: dict[str, Any] = {}
    for label in condition_paths:
        supplied = {name: mapping.get(label) for name, mapping in bundle_arguments.items()}
        core_names = ("checkpoint", "manifest", "training_provenance", "runner_config")
        if all(supplied[name] is None for name in core_names):
            if supplied["checkpoint_inspection"] is not None:
                _fail(f"{label} checkpoint inspection requires the complete training bundle")
            bundle_verification[label] = {
                "status": "not_supplied_report_claims_only",
                "causal_interpretation": "prohibited",
                "planned_training_updates": None,
                "checkpoint_completed_updates": None,
                "checkpoint_completed_updates_verification": "not_verified_no_training_bundle",
            }
            continue
        if any(supplied[name] is None for name in core_names):
            _fail(f"{label} training bundle must supply checkpoint, manifest, provenance, and runner config")
        bundle_verification[label] = _verify_training_bundle(
            label,
            loaded[label][0],
            checkpoint_path=supplied["checkpoint"],
            manifest_path=supplied["manifest"],
            training_provenance_path=supplied["training_provenance"],
            runner_config_path=supplied["runner_config"],
            checkpoint_inspection_path=supplied["checkpoint_inspection"],
        )
        bundle_verification[label]["causal_interpretation"] = (
            "prohibited_without_a_separately_verified_preregistered_experiment_plan"
        )
    result["training_artifact_verification"] = bundle_verification
    result["input_reports"] = {
        label: {"path": str(entry[3]), "sha256": entry[2]} for label, entry in loaded.items()
    }
    _verify_path_hashes(input_hashes)
    _write_new(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "results": result["results"],
        "paired_contrasts": result["paired_contrasts"],
    }, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
