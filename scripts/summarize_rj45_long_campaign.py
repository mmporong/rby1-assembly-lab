#!/usr/bin/env python3
"""Verify and summarize the preregistered RJ45 long-training campaign."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import shutil
from pathlib import Path
from typing import Any, NoReturn


EXPECTED_PROMOTION_RULE = (
    "At the selected common checkpoint age, at least two of three training seeds must "
    "exceed the paired baseline success count, the three-seed average must exceed it "
    "by at least 3 successes out of 32, and each model must have zero invalid and zero "
    "forbidden contact load. Otherwise do not open the final seed and report no "
    "demonstrated development improvement."
)
MINIMUM_SEEDS_ABOVE_BASELINE = 2
MINIMUM_MEAN_MARGIN_SUCCESSES = 3
EXPECTED_PROFILE_LABEL = "Z1"


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        payload = _object(json.loads(raw), str(path))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON {path}: {error}") from error
    return payload, hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot read evidence file {path}: {error}") from error


def _load_analyzer(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load paired analyzer {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pinned_analyzer(path: Path, expected_sha: str, module_name: str) -> Any:
    if _sha_file(path) != expected_sha:
        _fail(f"pinned analyzer bytes differ: {path}")
    return _load_analyzer(path, module_name)


def _assert_gate_semantics_match(
    frozen_v1_audit: dict[str, Any], posthoc_v2_audit: dict[str, Any]
) -> None:
    for field in ("results", "paired_contrasts", "paired_cases"):
        if frozen_v1_audit[field] != posthoc_v2_audit[field]:
            _fail(f"post-hoc v2 analyzer changes frozen-v1 gate semantics: {field}")


def verify_retrieval_manifest(
    retrieval_manifest_path: Path,
) -> tuple[dict[str, Any], Path, dict[str, str], str]:
    """Rehash every retrieved byte and cross-check the saved remote payload."""
    retrieval, _retrieval_sha = _load_json(retrieval_manifest_path)
    if retrieval.get("schema_version") != 1 or retrieval.get("terminal_status") != "completed":
        _fail("retrieval manifest is not a completed schema-version-1 retrieval")
    snapshot_value = retrieval.get("snapshot")
    if not isinstance(snapshot_value, str) or not snapshot_value:
        _fail("retrieval manifest snapshot is missing")
    snapshot = Path(snapshot_value).resolve()
    declared = _object(retrieval.get("files_sha256"), "retrieval.files_sha256")
    if len(declared) != 93:
        _fail(f"retrieval manifest must contain 93 files, got {len(declared)}")

    local_hashes: dict[str, str] = {}
    for relative, expected_sha in declared.items():
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            _fail(f"unsafe retrieval path: {relative!r}")
        if not isinstance(expected_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
            _fail(f"invalid SHA-256 for {relative}")
        actual_sha = _sha_file(snapshot / relative)
        if actual_sha != expected_sha:
            _fail(f"retrieved byte hash mismatch for {relative}")
        local_hashes[relative] = actual_sha

    remote_payload, remote_payload_sha = _load_json(snapshot / ".retrieval.json")
    expected_remote_payload = {
        "terminal_status": retrieval["terminal_status"],
        "files_sha256": declared,
    }
    if remote_payload != expected_remote_payload:
        _fail("saved remote retrieval payload differs from the local retrieval manifest")
    return retrieval, snapshot, local_hashes, remote_payload_sha


def _campaign_contract(plan: dict[str, Any]) -> tuple[list[int], list[dict[str, int]], int]:
    training = _object(plan.get("training"), "plan.training")
    development = _object(plan.get("development_evaluation"), "plan.development_evaluation")
    seeds = training.get("training_seeds")
    checkpoints = training.get("assessed_checkpoints")
    if (
        not isinstance(seeds, list)
        or not all(type(seed) is int for seed in seeds)
        or len(seeds) != 3
        or len(set(seeds)) != len(seeds)
    ):
        _fail("plan must preregister exactly three integer training seeds")
    if not isinstance(checkpoints, list) or len(checkpoints) != 3:
        _fail("plan must preregister exactly three assessed checkpoints")
    normalized: list[dict[str, int]] = []
    for checkpoint in checkpoints:
        entry = _object(checkpoint, "plan.assessed_checkpoint")
        filename = entry.get("filename")
        iteration = entry.get("iteration_index")
        completed = entry.get("completed_updates")
        if (
            not isinstance(filename, str)
            or filename != f"model_{iteration}.pt"
            or type(iteration) is not int
            or type(completed) is not int
            or completed != iteration + 1
        ):
            _fail("plan assessed checkpoint fields disagree")
        normalized.append({"iteration_index": iteration, "completed_updates": completed})
    if len({entry["iteration_index"] for entry in normalized}) != len(normalized):
        _fail("plan assessed checkpoints must be unique")
    if development.get("promotion_rule") != EXPECTED_PROMOTION_RULE:
        _fail("plan promotion rule differs from the reviewed numeric contract")
    attempts = development.get("attempts_per_policy")
    if type(attempts) is not int or attempts <= 0:
        _fail("plan development attempts_per_policy is invalid")
    return seeds, normalized, attempts


def _relative_snapshot_path(snapshot: Path, path: Path, name: str) -> str:
    try:
        return path.resolve().relative_to(snapshot).as_posix()
    except ValueError as error:
        raise ValueError(f"{name} is outside the retrieved snapshot") from error


def verify_consumed_artifacts(
    snapshot: Path,
    declared_hashes: dict[str, str],
    paths: dict[str, Path],
) -> dict[str, str]:
    """Bind every consumed snapshot path to its retrieval-manifest SHA claim."""
    verified: dict[str, str] = {}
    for name, path in paths.items():
        relative = _relative_snapshot_path(snapshot, path, name)
        if relative not in declared_hashes:
            _fail(f"consumed artifact is absent from retrieval manifest: {relative}")
        actual_sha = _sha_file(path)
        if declared_hashes[relative] != actual_sha:
            _fail(f"consumed artifact hash differs from retrieval manifest: {relative}")
        verified[relative] = actual_sha
    if len(verified) != len(paths):
        _fail("consumed artifact paths must be unique")
    return verified


def _assert_consumed_artifact_coverage(
    consumed: dict[str, str], required_relative_paths: set[str]
) -> None:
    if set(consumed) != required_relative_paths:
        missing = sorted(required_relative_paths - set(consumed))
        extra = sorted(set(consumed) - required_relative_paths)
        _fail(f"consumed artifact coverage differs: missing={missing}, extra={extra}")


def _local_path_from_remote_ref(repo_root: Path, value: Any, name: str) -> Path:
    reference = _object(value, name)
    remote_path = reference.get("path")
    if not isinstance(remote_path, str) or not remote_path:
        _fail(f"{name}.path is missing")
    root = repo_root.resolve()
    supplied = Path(remote_path)
    if supplied.is_absolute():
        parts = supplied.parts
        try:
            repo_index = parts.index(repo_root.name)
        except ValueError as error:
            raise ValueError(
                f"{name}.path does not identify repository-relative bytes"
            ) from error
        candidate = root.joinpath(*parts[repo_index + 1 :]).resolve()
    else:
        candidate = root.joinpath(supplied).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{name}.path escapes the repository") from error
    return candidate


def validate_campaign_contract(
    contract: dict[str, Any],
    *,
    repo_root: Path,
    plan: dict[str, Any],
    plan_sha: str,
    baseline_sha: str,
    seeds: list[int],
    checkpoints: list[dict[str, int]],
) -> dict[str, Any]:
    if contract.get("schema_version") != 1:
        _fail("campaign contract schema_version must be 1")
    plan_ref = _object(contract.get("plan"), "campaign_contract.plan")
    if plan_ref.get("sha256") != plan_sha:
        _fail("campaign contract plan SHA differs from supplied long-run plan")
    if contract.get("profile_label") != EXPECTED_PROFILE_LABEL:
        _fail("campaign contract profile must be Z1")
    if contract.get("training_seeds") != seeds:
        _fail("campaign contract training seeds differ from the plan")
    planned_checkpoints = _object(plan.get("training"), "plan.training").get(
        "assessed_checkpoints"
    )
    if not isinstance(planned_checkpoints, list):
        _fail("plan assessed checkpoints must be a list")
    if contract.get("assessed_checkpoints") != planned_checkpoints:
        _fail("campaign contract assessed checkpoints differ from the plan")
    if len(checkpoints) != len(planned_checkpoints):
        _fail("campaign contract checkpoint count differs")
    development = _object(plan.get("development_evaluation"), "plan.development_evaluation")
    if contract.get("development_evaluation_seed") != development.get("seed"):
        _fail("campaign contract development evaluation seed differs from the plan")
    if contract.get("final_holdout_opened") is not False:
        _fail("campaign contract says the final holdout was opened")

    input_manifest_ref = _object(
        contract.get("input_manifest"), "campaign_contract.input_manifest"
    )
    input_manifest_path = _local_path_from_remote_ref(
        repo_root, input_manifest_ref, "campaign_contract.input_manifest"
    )
    input_manifest_sha = _sha_file(input_manifest_path)
    if input_manifest_ref.get("sha256") != input_manifest_sha:
        _fail("campaign contract input-manifest SHA differs from current target bytes")
    input_manifest, _loaded_input_manifest_sha = _load_json(input_manifest_path)
    if input_manifest.get("schema_version") != 1:
        _fail("verified input manifest schema_version must be 1")
    anchored_inputs = _object(input_manifest.get("inputs"), "input_manifest.inputs")
    required_anchors = {
        "artifacts/rj45-paper-ablation/baseline.json": baseline_sha,
        "docs/evidence/rj45-paper-ablation/long-run-plan.json": plan_sha,
    }
    for relative, actual_sha in required_anchors.items():
        if anchored_inputs.get(relative) != actual_sha:
            _fail(f"supplied input SHA differs from verified input manifest: {relative}")

    candidate_ref = _object(
        contract.get("candidate_selection"), "campaign_contract.candidate_selection"
    )
    candidate_path = _local_path_from_remote_ref(
        repo_root, candidate_ref, "campaign_contract.candidate_selection"
    )
    candidate_sha = _sha_file(candidate_path)
    candidate, _loaded_candidate_sha = _load_json(candidate_path)
    candidate_anchor = "artifacts/rj45-paper-ablation/long-candidate.json"
    if (
        candidate_ref.get("sha256") != candidate_sha
        or anchored_inputs.get(candidate_anchor) != candidate_sha
    ):
        _fail("campaign candidate SHA differs from contract or verified input manifest")
    candidate_inputs = _object(candidate.get("inputs"), "candidate.inputs")
    frozen_analyzer_ref = _object(
        candidate_inputs.get("frozen_analyzer"), "candidate.inputs.frozen_analyzer"
    )
    frozen_analyzer_path = _local_path_from_remote_ref(
        repo_root, frozen_analyzer_ref, "candidate.inputs.frozen_analyzer"
    )
    frozen_analyzer_relative = frozen_analyzer_path.relative_to(repo_root.resolve()).as_posix()
    frozen_analyzer_sha = _sha_file(frozen_analyzer_path)
    if (
        frozen_analyzer_relative != "scripts/analyze_rj45_paired.py"
        or frozen_analyzer_ref.get("sha256") != frozen_analyzer_sha
        or anchored_inputs.get(frozen_analyzer_relative) != frozen_analyzer_sha
    ):
        _fail("frozen gate analyzer differs from candidate or verified input manifest")
    seed_ppo_configs = _object(
        contract.get("seed_ppo_configs"), "campaign_contract.seed_ppo_configs"
    )
    if set(seed_ppo_configs) != {str(seed) for seed in seeds}:
        _fail("campaign contract seed PPO config IDs differ from planned seeds")
    for seed, reference_value in seed_ppo_configs.items():
        reference = _object(
            reference_value, f"campaign_contract.seed_ppo_configs.{seed}"
        )
        config_path = _local_path_from_remote_ref(
            repo_root, reference, f"campaign_contract.seed_ppo_configs.{seed}"
        )
        relative = config_path.relative_to(repo_root).as_posix()
        if anchored_inputs.get(relative) != reference.get("sha256"):
            _fail(f"seed {seed} PPO config differs from verified input manifest")
    return {
        "input_manifest_path": input_manifest_path,
        "input_manifest_sha256": input_manifest_sha,
        "anchored_inputs": anchored_inputs,
        "gate_anchors": required_anchors | {candidate_anchor: candidate_sha},
        "candidate_path": candidate_path,
        "candidate_sha256": candidate_sha,
        "frozen_analyzer_path": frozen_analyzer_path,
        "frozen_analyzer_sha256": frozen_analyzer_sha,
        "seed_ppo_configs": seed_ppo_configs,
    }


def _expected_training_inputs(
    anchored_inputs: dict[str, Any], seed: int
) -> dict[str, str]:
    seed_config = f"artifacts/rj45-long-inputs-retry-20261027/ppo-seed-{seed}.json"
    expected = {
        path: digest
        for path, digest in anchored_inputs.items()
        if (
            (path.startswith("src/rby1_assembly/") and path.endswith(".py"))
            or path == "scripts/train.py"
            or path.startswith("configs/")
            or path.startswith("assets/generated/tasks/rj45-v1/")
            or path == seed_config
        )
    }
    if seed_config not in expected:
        _fail(f"verified input manifest omits seed {seed} PPO config")
    if not expected or any(
        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
        for digest in expected.values()
    ):
        _fail("verified input manifest training digests are invalid")
    return expected


def verify_training_provenance_frozen_map(
    *,
    provenance_path: Path,
    runner_path: Path,
    snapshot: Path,
    anchored_inputs: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    provenance, provenance_sha = _load_json(provenance_path)
    source_and_config = _object(
        provenance.get("source_and_config_sha256"),
        f"seed {seed} training_provenance.source_and_config_sha256",
    )
    expected_frozen = _expected_training_inputs(anchored_inputs, seed)
    snapshot_runner_relative = _relative_snapshot_path(
        snapshot, runner_path, f"seed {seed} runner config"
    )
    runner_candidates = [
        path
        for path in source_and_config
        if path == snapshot_runner_relative or path.endswith(f"/{snapshot_runner_relative}")
    ]
    if len(runner_candidates) != 1:
        _fail(f"seed {seed} training provenance must identify one run runner config")
    runner_relative = runner_candidates[0]
    expected_paths = set(expected_frozen) | {runner_relative}
    if set(source_and_config) != expected_paths:
        missing = sorted(expected_paths - set(source_and_config))
        extra = sorted(set(source_and_config) - expected_paths)
        _fail(
            f"seed {seed} training provenance source/config set differs: "
            f"missing={missing}, extra={extra}"
        )
    mismatches = [
        path
        for path, digest in expected_frozen.items()
        if source_and_config.get(path) != digest
    ]
    if mismatches:
        _fail(
            f"seed {seed} training provenance differs from frozen pre-run map: "
            f"{mismatches}"
        )
    runner_sha = _sha_file(runner_path)
    if source_and_config.get(runner_relative) != runner_sha:
        _fail(f"seed {seed} training provenance runner-config digest differs")
    return {
        "training_provenance_sha256": provenance_sha,
        "source_and_config_entry_count": len(source_and_config),
        "frozen_pre_run_digest_match_count": len(expected_frozen),
        "frozen_pre_run_digests": dict(sorted(expected_frozen.items())),
        "runner_config_exception": {
            "path": runner_relative,
            "sha256": runner_sha,
            "reason": (
                "run-directory runner config was created for the run and is linked to the "
                "contract-pinned seed PPO config by identical bytes"
            ),
        },
        "verification_boundary": (
            "For frozen training source/config paths, recorded provenance digests were matched "
            "to the contract-hashed pre-run input manifest. Those source bytes are not present "
            "in the retrieved snapshot and were not rehashed by this summary."
        ),
    }


def _run_directory(snapshot: Path, seed: int) -> Path:
    matches = list((snapshot / "runs" / str(seed) / "rj45-v1").glob("*"))
    directories = [path for path in matches if path.is_dir()]
    if len(directories) != 1:
        _fail(f"seed {seed} must have exactly one retrieved run directory")
    return directories[0]


def _report_paths(snapshot: Path, seed: int, iteration: int) -> tuple[Path, Path]:
    return (
        snapshot / "reports" / f"seed-{seed}-iter-{iteration}.json",
        snapshot / "inspections" / f"seed-{seed}-model_{iteration}.json",
    )


def _validate_report_identity(
    report: dict[str, Any], bundle: dict[str, Any], *, seed: int, completed_updates: int
) -> None:
    if bundle.get("training_seed") != seed:
        _fail(f"report bundle seed differs from planned seed {seed}")
    if bundle.get("checkpoint_completed_updates") != completed_updates:
        _fail(f"checkpoint completed updates differ from planned {completed_updates}")
    if report.get("requested_attempts") != report.get("attempts"):
        _fail("report did not complete all requested attempts")


def select_common_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_updates: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        updates = row.get("completed_updates")
        if type(updates) is not int:
            _fail("result row completed_updates must be an integer")
        by_updates.setdefault(updates, []).append(row)
    if not by_updates or any(len(group) != 3 for group in by_updates.values()):
        _fail("each checkpoint age must contain exactly three training seeds")
    candidates = []
    for updates, group in sorted(by_updates.items()):
        successes = [row["successes"] for row in group]
        candidates.append({
            "completed_updates": updates,
            "successes_by_training_seed": {
                str(row["training_seed"]): row["successes"] for row in group
            },
            "total_successes": sum(successes),
            "mean_successes": sum(successes) / len(successes),
        })
    selected = min(candidates, key=lambda value: (-value["total_successes"], value["completed_updates"]))
    return {"candidates": candidates, "selected": selected}


def _copy_evidence(
    evidence_dir: Path,
    *,
    source_paths: dict[str, Path],
    expected_hashes: dict[str, str],
) -> dict[str, str]:
    if evidence_dir.exists():
        _fail(f"evidence directory already exists: {evidence_dir}")
    for relative, source in source_paths.items():
        destination = evidence_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    copied = {relative: _sha_file(evidence_dir / relative) for relative in source_paths}
    if copied != expected_hashes:
        _fail("copied evidence hashes differ from verified source bytes")
    return copied


def summarize_campaign(
    *,
    repo_root: Path,
    retrieval_manifest_path: Path,
    plan_path: Path,
    baseline_path: Path,
    output_path: Path,
    evidence_dir: Path | None = None,
) -> dict[str, Any]:
    retrieval, snapshot, local_hashes, remote_payload_sha = verify_retrieval_manifest(
        retrieval_manifest_path
    )
    plan, plan_sha = _load_json(plan_path)
    baseline, baseline_sha = _load_json(baseline_path)
    seeds, checkpoints, attempts = _campaign_contract(plan)
    contract_path = snapshot / "campaign-contract.json"
    consumed_snapshot_artifacts = verify_consumed_artifacts(
        snapshot,
        local_hashes,
        {"campaign contract": contract_path},
    )
    required_consumed_paths = set(consumed_snapshot_artifacts)
    campaign_contract, campaign_contract_sha = _load_json(contract_path)
    contract_validation = validate_campaign_contract(
        campaign_contract,
        repo_root=repo_root,
        plan=plan,
        plan_sha=plan_sha,
        baseline_sha=baseline_sha,
        seeds=seeds,
        checkpoints=checkpoints,
    )
    gate_analyzer_path = contract_validation["frozen_analyzer_path"]
    gate_analyzer_sha = _sha_file(gate_analyzer_path)
    if gate_analyzer_sha != contract_validation["frozen_analyzer_sha256"]:
        _fail("frozen gate analyzer changed after campaign-contract validation")
    auxiliary_analyzer_path = repo_root / "scripts" / "analyze_rj45_paired_v2.py"
    auxiliary_analyzer_sha = _sha_file(auxiliary_analyzer_path)
    gate_analyzer = _load_pinned_analyzer(
        gate_analyzer_path, gate_analyzer_sha, "rj45_frozen_gate_analyzer"
    )
    auxiliary_analyzer = _load_analyzer(
        auxiliary_analyzer_path, "rj45_posthoc_bundle_analyzer_v2"
    )
    if baseline.get("requested_attempts") != attempts or baseline.get("attempts") != attempts:
        _fail("paired baseline attempt count differs from the long-run plan")

    baseline_successes = baseline.get("valid_successes")
    if type(baseline_successes) is not int:
        _fail("paired baseline valid_successes is invalid")
    rows: list[dict[str, Any]] = []
    paired_audits: dict[str, Any] = {}
    training_lineage: dict[str, Any] = {}
    evidence_sources: dict[str, Path] = {
        "inputs/long-run-plan.json": plan_path,
        "inputs/baseline.json": baseline_path,
        "inputs/retrieval-manifest.json": retrieval_manifest_path,
        "inputs/remote-retrieval-payload.json": snapshot / ".retrieval.json",
        "inputs/campaign-contract.json": contract_path,
        "inputs/input-manifest.json": contract_validation["input_manifest_path"],
        "inputs/long-candidate.json": contract_validation["candidate_path"],
        "inputs/frozen-gate-analyzer.py": gate_analyzer_path,
        "inputs/posthoc-bundle-analyzer-v2.py": auxiliary_analyzer_path,
    }
    evidence_hashes = {relative: _sha_file(path) for relative, path in evidence_sources.items()}

    for seed in seeds:
        run_dir = _run_directory(snapshot, seed)
        manifest_path = run_dir / "checkpoint_manifest.json"
        provenance_path = run_dir / "training_provenance.json"
        runner_path = run_dir / "runner_config.json"
        shared_consumed = {
            f"seed {seed} checkpoint manifest": manifest_path,
            f"seed {seed} training provenance": provenance_path,
            f"seed {seed} runner config": runner_path,
        }
        shared_verified = verify_consumed_artifacts(
            snapshot, local_hashes, shared_consumed
        )
        consumed_snapshot_artifacts.update(shared_verified)
        required_consumed_paths.update(
            _relative_snapshot_path(snapshot, path, name)
            for name, path in shared_consumed.items()
        )
        seed_config_ref = _object(
            contract_validation["seed_ppo_configs"].get(str(seed)),
            f"campaign_contract.seed_ppo_configs.{seed}",
        )
        if seed_config_ref.get("sha256") != shared_verified[
            _relative_snapshot_path(snapshot, runner_path, f"seed {seed} runner config")
        ]:
            _fail(f"campaign contract PPO config SHA differs from seed {seed} runner bytes")
        training_lineage[str(seed)] = verify_training_provenance_frozen_map(
            provenance_path=provenance_path,
            runner_path=runner_path,
            snapshot=snapshot,
            anchored_inputs=contract_validation["anchored_inputs"],
            seed=seed,
        )
        run_prefix = f"raw/runs/{seed}"
        for name, path in (
            ("checkpoint_manifest.json", manifest_path),
            ("training_provenance.json", provenance_path),
            ("runner_config.json", runner_path),
        ):
            relative = f"{run_prefix}/{name}"
            evidence_sources[relative] = path
            evidence_hashes[relative] = _sha_file(path)
        for checkpoint in checkpoints:
            iteration = checkpoint["iteration_index"]
            completed = checkpoint["completed_updates"]
            label = f"seed-{seed}-updates-{completed}"
            report_path, inspection_path = _report_paths(snapshot, seed, iteration)
            checkpoint_path = run_dir / f"model_{iteration}.pt"
            checkpoint_consumed = {
                f"{label} report": report_path,
                f"{label} inspection": inspection_path,
                f"{label} checkpoint": checkpoint_path,
            }
            checkpoint_verified = verify_consumed_artifacts(
                snapshot, local_hashes, checkpoint_consumed
            )
            consumed_snapshot_artifacts.update(checkpoint_verified)
            required_consumed_paths.update(
                _relative_snapshot_path(snapshot, path, name)
                for name, path in checkpoint_consumed.items()
            )
            report, report_sha = _load_json(report_path)
            audit = gate_analyzer.analyze_reports(
                baseline,
                {label: report},
                expected_attempts=attempts,
                bootstrap_resamples=10_000,
                bootstrap_seed=20261029,
            )
            auxiliary_audit = auxiliary_analyzer.analyze_reports(
                baseline,
                {label: report},
                expected_attempts=attempts,
                bootstrap_resamples=10_000,
                bootstrap_seed=20261029,
            )
            _assert_gate_semantics_match(audit, auxiliary_audit)
            bundle = auxiliary_analyzer._verify_training_bundle(
                label,
                report,
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                training_provenance_path=provenance_path,
                runner_config_path=runner_path,
                checkpoint_inspection_path=inspection_path,
            )
            bundle["posthoc_v2_original_unverified"] = bundle.pop("unverified")
            bundle["summary_wrapper_cross_checks"] = {
                "frozen_training_source_and_config_digests": (
                    "matched to the contract-hashed pre-run input manifest"
                ),
                "preregistered_plan": (
                    "supplied bytes matched the pre-run input manifest and campaign contract"
                ),
            }
            bundle["remaining_unverified"] = [
                "Frozen training source/config bytes were not present in the retrieved snapshot; only their recorded digests were matched to the contract-hashed pre-run map."
            ]
            _validate_report_identity(report, bundle, seed=seed, completed_updates=completed)
            result = audit["results"][label]
            forbidden_peak = max(
                float(outcome["forbidden_contact_load_peak_n"])
                for outcome in report["episode_outcomes"]
            )
            row = {
                "training_seed": seed,
                "completed_updates": completed,
                "iteration_index": iteration,
                "successes": result["successes_all_attempts"],
                "attempts": result["attempts"],
                "invalid_episodes": result["invalid_episodes"],
                "forbidden_contact_load_peak_n": forbidden_peak,
                "report_sha256": report_sha,
                "checkpoint_sha256": bundle["checkpoint"]["sha256"],
                "checkpoint_inspection_sha256": bundle["checkpoint_inspection"]["sha256"],
                "checkpoint_completed_updates_verification": bundle[
                    "checkpoint_completed_updates_verification"
                ],
            }
            rows.append(row)
            paired_audits[label] = {
                "results": audit["results"],
                "paired_contrast": audit["paired_contrasts"][f"{label}_minus_baseline"],
                "scope": audit["scope"],
                "gate_analyzer": "frozen_v1",
                "posthoc_v2_semantics_match": True,
                "training_artifact_verification": bundle,
            }
            for category, path in (("reports", report_path), ("inspections", inspection_path)):
                relative = f"raw/{category}/{path.name}"
                evidence_sources[relative] = path
                evidence_hashes[relative] = _sha_file(path)

    _assert_consumed_artifact_coverage(
        consumed_snapshot_artifacts, required_consumed_paths
    )
    common = select_common_checkpoint(rows)
    selected_updates = common["selected"]["completed_updates"]
    selected_rows = [row for row in rows if row["completed_updates"] == selected_updates]
    seeds_above = sum(row["successes"] > baseline_successes for row in selected_rows)
    mean_successes = sum(row["successes"] for row in selected_rows) / len(selected_rows)
    safety_pass = all(
        row["invalid_episodes"] == 0 and row["forbidden_contact_load_peak_n"] == 0.0
        for row in selected_rows
    )
    gate = {
        "baseline_successes": baseline_successes,
        "attempts_per_policy": attempts,
        "minimum_training_seeds_strictly_above_baseline": MINIMUM_SEEDS_ABOVE_BASELINE,
        "actual_training_seeds_strictly_above_baseline": seeds_above,
        "minimum_mean_successes": baseline_successes + MINIMUM_MEAN_MARGIN_SUCCESSES,
        "actual_mean_successes": mean_successes,
        "zero_invalid_and_forbidden_contact_for_all_selected_models": safety_pass,
    }
    gate["passed"] = bool(
        seeds_above >= MINIMUM_SEEDS_ABOVE_BASELINE
        and mean_successes >= baseline_successes + MINIMUM_MEAN_MARGIN_SUCCESSES
        and safety_pass
    )
    gate["final_evaluation_seed_may_be_opened"] = gate["passed"]

    training_relevant_paths = set().union(
        *(
            set(_expected_training_inputs(contract_validation["anchored_inputs"], seed))
            for seed in seeds
        )
    )
    non_training_provenance_paths = sorted(
        set(contract_validation["anchored_inputs"]) - training_relevant_paths
    )

    copied_hashes = None
    if evidence_dir is not None:
        copied_hashes = _copy_evidence(
            evidence_dir, source_paths=evidence_sources, expected_hashes=evidence_hashes
        )
    summary = {
        "schema_version": 1,
        "status": "development_promotion_passed" if gate["passed"] else "development_promotion_failed",
        "inputs": {
            "long_run_plan": {"path": str(plan_path), "sha256": plan_sha},
            "paired_baseline": {"path": str(baseline_path), "sha256": baseline_sha},
            "retrieval_manifest": {
                "path": str(retrieval_manifest_path),
                "sha256": _sha_file(retrieval_manifest_path),
                "terminal_status": retrieval["terminal_status"],
            },
            "campaign_contract": {
                "path": str(contract_path),
                "sha256": campaign_contract_sha,
                "profile_label": campaign_contract["profile_label"],
                "input_manifest_sha256": contract_validation[
                    "input_manifest_sha256"
                ],
                "verified_input_anchors": contract_validation["gate_anchors"],
                "candidate_sha256": contract_validation["candidate_sha256"],
            },
            "analyzers": {
                "gate": {
                    "path": str(gate_analyzer_path),
                    "sha256": gate_analyzer_sha,
                    "role": "pre-run-manifest-and-candidate-pinned frozen v1 analyzer used for every gate count",
                },
                "posthoc_bundle_verifier": {
                    "path": str(auxiliary_analyzer_path),
                    "sha256": auxiliary_analyzer_sha,
                    "role": "post-hoc v2 bundle and inspection verifier; cannot change gate counts",
                },
            },
        },
        "integrity": {
            "remote_payload_check": {
                "meaning": "saved remote payload exactly matches terminal status and the 93 SHA claims in the local retrieval manifest",
                "payload_path": str(snapshot / ".retrieval.json"),
                "payload_sha256": remote_payload_sha,
                "matched": True,
            },
            "local_byte_check": {
                "meaning": "all 93 retrieved files were read locally and their SHA-256 values were recomputed",
                "file_count": len(local_hashes),
                "matched": True,
            },
            "checkpoint_payload_check": {
                "meaning": "the nine saved inspection records attest CPU weights_only payload inspection; this summary rechecks their bytes and lineage but does not load checkpoint tensors again",
                "status": "inspection_records_and_bundle_lineage_verified",
            },
            "training_source_and_config_check": {
                "by_training_seed": training_lineage,
                "pre_run_input_manifest_entry_count": len(
                    contract_validation["anchored_inputs"]
                ),
                "training_relevant_union_count": len(training_relevant_paths),
                "non_training_provenance_input_count": len(
                    non_training_provenance_paths
                ),
                "non_training_provenance_inputs": non_training_provenance_paths,
                "scope_boundary": (
                    "The remaining pre-run manifest inputs are launch, pilot, evaluation, or "
                    "evidence inputs. This summary does not claim they are training "
                    "source_and_config entries or that their bytes were present in the retrieved snapshot."
                ),
            },
            "consumed_snapshot_artifact_check": {
                "meaning": "every report, inspection, checkpoint, run metadata file, and campaign contract consumed by the summary is listed in the retrieval manifest and matches its SHA-256 claim",
                "file_count": len(consumed_snapshot_artifacts),
                "files_sha256": dict(sorted(consumed_snapshot_artifacts.items())),
                "matched": True,
            },
        },
        "development_results": rows,
        "paired_audits": paired_audits,
        "common_checkpoint_selection": common,
        "promotion_gate": gate,
        "interpretation_limit": (
            "The three learned policies share 32 calibration cases and one physical held grasp; "
            "they are not 96 independent physical grasp trials."
        ),
        "evidence_copy": (
            {
                "directory": str(evidence_dir),
                "files_sha256": copied_hashes,
                "checkpoint_binaries_copied": False,
            }
            if evidence_dir is not None
            else None
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        _fail(f"output already exists: {output_path}")
    output_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-manifest", required=True, type=Path)
    parser.add_argument("--long-run-plan", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--evidence-dir", type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    result = summarize_campaign(
        repo_root=repo_root,
        retrieval_manifest_path=args.retrieval_manifest.resolve(),
        plan_path=args.long_run_plan.resolve(),
        baseline_path=args.baseline.resolve(),
        output_path=args.output.resolve(),
        evidence_dir=args.evidence_dir.resolve() if args.evidence_dir else None,
    )
    print(json.dumps({
        "output": str(args.output),
        "selected_completed_updates": result["common_checkpoint_selection"]["selected"]["completed_updates"],
        "promotion_gate_passed": result["promotion_gate"]["passed"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
