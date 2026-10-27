#!/usr/bin/env python3
"""Render a traceable summary of an RJ45 paired pilot audit.

The figure deliberately visualizes only fields already present in the paired
audit: all-attempt success counts and each paired case's success/valid state.
It does not infer a learning curve or generalization result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any


CONDITION_ORDER = ("baseline", "A0", "A1", "A2", "E1", "Z1")
LEARNED_CONDITIONS = CONDITION_ORDER[1:]
EXPECTED_ATTEMPTS = 32
EXPECTED_TRAINING_UPDATES = 20
EXPECTED_BUNDLE_STATUS = "checkpoint_manifest_provenance_runner_bytes_verified"
EXPECTED_SCOPE_STATUS = "report_proof_hashes_consistent_inputs_not_rehashed"
EXPECTED_INSPECTION_LINK_STATUS = (
    "supplied_inspection_report_consistent_payload_not_rechecked"
)
EXPECTED_INSPECTION_STATUS = "checkpoint_payload_and_lineage_verified"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_POLICY_SHA256 = "7229da017a96a17fc9293d7221b59e496449a99f52a8a295f2c13b5d674e9c65"
LOW_EXPLORATION_POLICY_SHA256 = (
    "ccbdaddfa0be15406174c31a5ef9a2651e2e46e97f4ac71b4ec8fb86f027dffd"
)
TRAINING_PROFILE_CONTRACTS = {
    "baseline": {
        "path": "configs/rj45-training-target-offset-v1.json",
        "sha256": "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179",
        "profile_id": "baseline_target_offset_v1",
        "potential_type": "legacy_pose_reward",
        "semantics": "non-learned biased-zero baseline task configuration",
    },
    "A0": {
        "path": "configs/rj45-training-target-offset-v1.json",
        "sha256": "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179",
        "profile_id": "a0_target_offset_v1",
        "potential_type": "legacy_pose_reward",
        "semantics": "base PPO task configuration",
    },
    "A1": {
        "path": "configs/rj45-training-target-offset-a1-discounted-pose-v2.json",
        "sha256": "f2ccc9c6c4421481399720c56221644a6e6fb276d926359094ed3afcfb6daa28",
        "profile_id": "a1_discounted_pose_v2",
        "potential_type": "normalized_pose",
        "semantics": "discounted potential with INSERT transition and finite-horizon termination handling",
    },
    "A2": {
        "path": "configs/rj45-training-target-offset-a2-keypoint-shape-v2.json",
        "sha256": "f54a91193cfb38b841b6111ee41a96efac5377d5cc1b61ec1ec7dac57f081df8",
        "profile_id": "a2_keypoint_shape_v2",
        "potential_type": "cad_keypoints",
        "semantics": "A1 potential replaced by CAD keypoint geometry",
    },
    "E1": {
        "path": "configs/rj45-training-target-offset-v1.json",
        "sha256": "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179",
        "profile_id": "e1_target_offset_v1",
        "potential_type": "legacy_pose_reward",
        "semantics": "A0 task configuration with low-exploration PPO profile",
    },
    "Z1": {
        "path": "configs/rj45-training-target-offset-v1.json",
        "sha256": "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179",
        "profile_id": "z1_target_offset_v1",
        "potential_type": "legacy_pose_reward",
        "semantics": "E1 task and PPO configuration with zero-initialized actor mean layer",
    },
}
CONDITION_LABELS = {
    "baseline": "기준 제어",
    "A0": "A0 · 기본 PPO",
    "A1": "A1 · 할인 보상·종료 처리",
    "A2": "A2 · A1 potential 교체",
    "E1": "E1 · 낮은 탐색 폭",
    "Z1": "Z1 · actor 평균 0 초기화",
}
CONDITION_METHOD_NOTES = {
    "baseline": "학습 정책을 사용하지 않는 기준 제어",
    "A0": "기본 PPO 보상과 기본 탐색 설정",
    "A1": "할인 보상과 실제 INSERT 전이·종료 처리를 함께 반영",
    "A2": "A1의 기존 potential을 형상 대응점 기반으로 교체",
    "E1": "A0 설정에서 초기 탐색 폭을 낮춤",
    "Z1": "결정론적 평균만 0으로 시작하고 탐색 잡음 0.1은 유지",
}
BAR_COLORS = {
    "baseline": "#2563EB",
    "A0": "#94A3B8",
    "A1": "#64748B",
    "A2": "#7C3AED",
    "E1": "#0F766E",
    "Z1": "#C2410C",
}


class AuditError(ValueError):
    """Raised when the input audit cannot support a faithful paired plot."""


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _condition_names(results: dict[str, Any]) -> list[str]:
    if "baseline" not in results:
        raise AuditError("audit.results must contain 'baseline'")
    known = [name for name in CONDITION_ORDER if name in results]
    unknown = sorted(set(results) - set(CONDITION_ORDER))
    if unknown:
        raise AuditError(f"unsupported conditions: {unknown}")
    return known


def _resolve_audit_reference(audit_path: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    for root in audit_path.parents:
        candidate = root / relative
        if candidate.is_file():
            return candidate.resolve()
    raise AuditError(f"referenced audit file does not exist: {relative_path}")


def _result_path_matches_bundle(result_path: object, bundle_path: object) -> bool:
    if not isinstance(result_path, str) or not isinstance(bundle_path, str):
        return False
    normalized_result = result_path.replace("\\", "/")
    normalized_bundle = bundle_path.replace("\\", "/").lstrip("/")
    return normalized_result == normalized_bundle or normalized_result.endswith(
        "/" + normalized_bundle
    )


def _validate_condition_contracts(
    experiment: dict[str, Any],
    results: dict[str, Any],
    artifact_verification: dict[str, Any],
    condition_names: list[str],
) -> dict[str, dict[str, Any]]:
    identifiers = experiment.get("reported_configuration_identifiers")
    if not isinstance(identifiers, dict) or set(identifiers) != set(condition_names):
        raise AuditError(
            "reported_configuration_identifiers must match all plotted conditions"
        )

    evidence: dict[str, dict[str, Any]] = {}
    for name in condition_names:
        identifier = identifiers[name]
        result = results[name]
        if not isinstance(identifier, dict) or not isinstance(result, dict):
            raise AuditError(f"{name}: configuration identifiers must be objects")
        result_training = result.get("training_config")
        if not isinstance(result_training, dict):
            raise AuditError(f"{name}: result training_config is missing")
        identifier_path = identifier.get("training_config_path")
        identifier_sha = identifier.get("training_config_sha256")
        if not isinstance(identifier_path, str) or not _is_sha256(identifier_sha):
            raise AuditError(f"{name}: training config identifier is invalid")
        if (
            result_training.get("path") != identifier_path
            or result_training.get("sha256") != identifier_sha
        ):
            raise AuditError(f"{name}: result and experiment training configs differ")

        contract: dict[str, Any] = {
            "training_config_path": identifier_path,
            "training_config_sha256": identifier_sha,
            "verification_basis": "audit metadata matched to pinned config hash",
        }
        if name in TRAINING_PROFILE_CONTRACTS:
            expected = TRAINING_PROFILE_CONTRACTS[name]
            if identifier_path != expected["path"] or identifier_sha != expected["sha256"]:
                raise AuditError(f"{name}: pinned reward-profile config does not match")
            contract.update(
                {
                    "reward_profile_id": expected["profile_id"],
                    "potential_type": expected["potential_type"],
                    "semantics": expected["semantics"],
                }
            )

        if name == "baseline":
            evidence[name] = contract
            continue

        verification = artifact_verification[name]
        policy_sha = identifier.get("policy_config_sha256")
        policy_path = identifier.get("policy_config")
        if not isinstance(policy_path, str) or not _is_sha256(policy_sha):
            raise AuditError(f"{name}: policy config identifier is invalid")
        expected_policy_sha = (
            LOW_EXPLORATION_POLICY_SHA256 if name in {"E1", "Z1"} else DEFAULT_POLICY_SHA256
        )
        expected_policy_name = (
            "ppo-rj45-low-exploration-v1.json"
            if name in {"E1", "Z1"}
            else "ppo-rj45-v1.json"
        )
        if policy_sha != expected_policy_sha or Path(policy_path).name != expected_policy_name:
            raise AuditError(f"{name}: pinned PPO profile does not match")

        expected_noise = 0.1 if name in {"E1", "Z1"} else 0.6
        if verification.get("initial_action_noise_std") != expected_noise:
            raise AuditError(f"{name}: initial_action_noise_std does not match its label")

        identifier_initialization = identifier.get("policy_initialization")
        verified_initialization = verification.get("policy_initialization")
        result_initialization = result.get("policy_initialization")
        if not (
            isinstance(identifier_initialization, dict)
            and identifier_initialization == verified_initialization
            and identifier_initialization == result_initialization
        ):
            raise AuditError(f"{name}: policy initialization records are inconsistent")

        if name == "Z1":
            required_zero_mean = {
                "mode": "residual_actor_final_linear_zero",
                "zero_actor_output_layer": True,
                "weight_all_zero_after_initialization": True,
                "bias_all_zero_after_initialization": True,
                "deterministic_actor_output_all_zero": True,
                "deterministic_actor_output_max_abs": 0.0,
                "stochastic_policy_noise_modified": False,
            }
            for field, expected_value in required_zero_mean.items():
                if identifier_initialization.get(field) != expected_value:
                    raise AuditError(f"Z1: policy initialization field {field} is invalid")
            for field in ("post_weight_sha256", "post_bias_sha256"):
                if not _is_sha256(identifier_initialization.get(field)):
                    raise AuditError(f"Z1: {field} is not a lowercase SHA-256")
            contract["actor_mean_initialization"] = "deterministic mean output zero"
            contract["initial_action_noise_std"] = expected_noise
            contract["stochastic_policy_noise_modified"] = False
        elif name == "E1":
            contract["initial_action_noise_std"] = expected_noise

        contract["policy_config_sha256"] = policy_sha
        evidence[name] = contract
    return evidence


def _validate_and_extract(
    audit: dict[str, Any], audit_path: Path, audit_sha256: str
) -> dict[str, Any]:
    results = audit.get("results")
    paired_cases = audit.get("paired_cases")
    if not isinstance(results, dict) or not results:
        raise AuditError("audit.results must be a non-empty object")
    if not isinstance(paired_cases, list) or not paired_cases:
        raise AuditError("audit.paired_cases must be a non-empty array")

    condition_names = _condition_names(results)
    attempts = len(paired_cases)
    audit_schema_version = audit.get("schema_version")
    if audit_schema_version not in (1, 2):
        raise AuditError("only paired audit schema versions 1 and 2 are supported")
    experiment = audit.get("experiment")
    if not isinstance(experiment, dict):
        raise AuditError("audit.experiment must be an object")
    if experiment.get("requested_attempts") != EXPECTED_ATTEMPTS:
        raise AuditError(
            f"experiment.requested_attempts must be {EXPECTED_ATTEMPTS} for this pilot"
        )
    if attempts != EXPECTED_ATTEMPTS:
        raise AuditError(f"paired_cases must contain {EXPECTED_ATTEMPTS} cases")

    scope = audit.get("scope")
    if not isinstance(scope, dict):
        raise AuditError("audit.scope must be an object")
    if scope.get("verification_status") != EXPECTED_SCOPE_STATUS:
        raise AuditError(
            "scope.verification_status does not support the pilot proof wording"
        )
    scope_limitations = scope.get("data_limitations")
    if not isinstance(scope_limitations, list):
        raise AuditError("scope.data_limitations must be an array")
    required_limitations = {
        "Calibration-bias cases report the same nominal/held proof hashes.",
        "Bootstrap intervals are conditional on one evaluation seed and one reported held state.",
        "Report proof hashes are consistency claims until referenced files are rehashed.",
    }
    if not required_limitations.issubset(set(scope_limitations)):
        raise AuditError("scope.data_limitations does not establish the reported held-state limit")

    learned_names = [name for name in condition_names if name != "baseline"]
    artifact_verification = audit.get("training_artifact_verification")
    if not isinstance(artifact_verification, dict):
        raise AuditError("training_artifact_verification must be an object")
    if set(artifact_verification) != set(learned_names):
        raise AuditError("training artifact groups must match learned result groups")
    condition_contracts = _validate_condition_contracts(
        experiment, results, artifact_verification, condition_names
    )

    training_seeds: set[int] = set()
    held_proofs: set[str] = set()
    inspection_file_evidence: dict[str, dict[str, Any]] = {}
    for name in learned_names:
        verification = artifact_verification[name]
        result = results.get(name)
        if not isinstance(verification, dict):
            raise AuditError(f"training_artifact_verification.{name} must be an object")
        if not isinstance(result, dict):
            raise AuditError(f"audit.results.{name} must be an object")
        if verification.get("status") != EXPECTED_BUNDLE_STATUS:
            raise AuditError(f"{name}: training bundle status is not verified")
        training_seed = verification.get("training_seed")
        if not isinstance(training_seed, int) or isinstance(training_seed, bool):
            raise AuditError(f"{name}: training_seed must be an integer")
        training_seeds.add(training_seed)
        if audit_schema_version == 1:
            if verification.get("training_updates") != EXPECTED_TRAINING_UPDATES:
                raise AuditError(
                    f"{name}: recorded training_updates must be {EXPECTED_TRAINING_UPDATES}"
                )
        else:
            if verification.get("planned_training_updates") != EXPECTED_TRAINING_UPDATES:
                raise AuditError(
                    f"{name}: planned_training_updates must be {EXPECTED_TRAINING_UPDATES}"
                )
            if verification.get("checkpoint_completed_updates") != EXPECTED_TRAINING_UPDATES:
                raise AuditError(
                    f"{name}: checkpoint_completed_updates must be {EXPECTED_TRAINING_UPDATES}"
                )
            if (
                verification.get("checkpoint_completed_updates_verification")
                != EXPECTED_INSPECTION_LINK_STATUS
            ):
                raise AuditError(f"{name}: checkpoint inspection link status is unsupported")
            inspection = verification.get("checkpoint_inspection")
            if (
                not isinstance(inspection, dict)
                or inspection.get("status") != EXPECTED_INSPECTION_STATUS
            ):
                raise AuditError(f"{name}: checkpoint inspection status is not verified")
            inspection_path = inspection.get("path")
            inspection_sha = inspection.get("sha256")
            if (
                not isinstance(inspection_path, str)
                or not inspection_path
                or Path(inspection_path).is_absolute()
                or ".." in Path(inspection_path).parts
                or Path(inspection_path).suffix != ".json"
            ):
                raise AuditError(f"{name}: checkpoint inspection path is invalid")
            if not _is_sha256(inspection_sha):
                raise AuditError(
                    f"{name}: checkpoint inspection SHA-256 is missing or invalid"
                )
            resolved_inspection = _resolve_audit_reference(audit_path, inspection_path)
            try:
                inspection_bytes = resolved_inspection.read_bytes()
                inspection_document = json.loads(inspection_bytes.decode("utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuditError(f"{name}: cannot read checkpoint inspection file") from exc
            actual_inspection_sha = hashlib.sha256(inspection_bytes).hexdigest()
            if actual_inspection_sha != inspection_sha:
                raise AuditError(f"{name}: checkpoint inspection file SHA-256 differs")
            if not isinstance(inspection_document, dict):
                raise AuditError(f"{name}: checkpoint inspection file root is invalid")
            inspected_checkpoint = inspection_document.get("checkpoint")
            verified_checkpoint = verification.get("checkpoint")
            if not isinstance(inspected_checkpoint, dict) or not isinstance(
                verified_checkpoint, dict
            ):
                raise AuditError(f"{name}: checkpoint inspection lineage is missing")
            inspected_bundle_keys = (
                "checkpoint",
                "checkpoint_manifest",
                "training_provenance",
                "runner_config",
            )
            for reference_key in inspected_bundle_keys:
                inspected_reference = inspection_document.get(reference_key)
                verified_reference = verification.get(reference_key)
                if not isinstance(inspected_reference, dict) or not isinstance(
                    verified_reference, dict
                ):
                    raise AuditError(
                        f"{name}: {reference_key} inspection reference is missing"
                    )
                if (
                    inspected_reference.get("path") != verified_reference.get("path")
                    or inspected_reference.get("sha256")
                    != verified_reference.get("sha256")
                ):
                    raise AuditError(
                        f"{name}: {reference_key} inspection and bundle references differ"
                    )
            if (
                inspection_document.get("status") != EXPECTED_INSPECTION_STATUS
                or inspection_document.get("planned_training_updates")
                != EXPECTED_TRAINING_UPDATES
                or inspected_checkpoint.get("completed_updates")
                != EXPECTED_TRAINING_UPDATES
                or inspected_checkpoint.get("sha256") != verified_checkpoint.get("sha256")
            ):
                raise AuditError(f"{name}: checkpoint inspection file claims differ")
            if (
                not _result_path_matches_bundle(
                    result.get("checkpoint"), verified_checkpoint.get("path")
                )
                or result.get("checkpoint_sha256") != verified_checkpoint.get("sha256")
            ):
                raise AuditError(f"{name}: evaluation result checkpoint differs from bundle")
            result_evidence = result.get("verified_checkpoint_evidence")
            if not isinstance(result_evidence, dict):
                raise AuditError(f"{name}: result checkpoint evidence is missing")
            if (
                result_evidence.get("checkpoint_manifest_sha256")
                != verification["checkpoint_manifest"].get("sha256")
                or result_evidence.get("training_provenance_sha256")
                != verification["training_provenance"].get("sha256")
            ):
                raise AuditError(f"{name}: result and bundle lineage hashes differ")
            if (
                inspection_document["checkpoint_manifest"].get("training_completed")
                is not True
                or inspection_document["training_provenance"].get("seed")
                != training_seed
                or inspection_document["runner_config"].get("seed") != training_seed
            ):
                raise AuditError(f"{name}: inspection completion or seed lineage differs")
            inspection_file_evidence[name] = {
                "path": inspection_path,
                "sha256": actual_inspection_sha,
                "bytes_rehashed_by_plotter": True,
                "checkpoint_payload_reloaded_by_plotter": False,
            }
        proof = (
            result.get("verified_checkpoint_evidence", {})
            .get("held_grasp_reset_proof", {})
            .get("artifact_sha256")
            if isinstance(result, dict)
            else None
        )
        if not isinstance(proof, str) or not _is_sha256(proof):
            raise AuditError(f"{name}: held grasp report proof SHA-256 is missing")
        held_proofs.add(proof)
    if len(training_seeds) != 1:
        raise AuditError("all learned groups must report one common training seed")
    if len(held_proofs) != 1:
        raise AuditError("all learned groups must report one common held-grasp proof")

    condition_summaries: list[dict[str, Any]] = []

    for name in condition_names:
        result = results[name]
        if not isinstance(result, dict):
            raise AuditError(f"audit.results.{name} must be an object")
        reported_attempts = result.get("attempts")
        if reported_attempts != attempts:
            raise AuditError(
                f"{name}: reported attempts={reported_attempts}, paired cases={attempts}"
            )

        success_count = 0
        invalid_count = 0
        for case_index, case in enumerate(paired_cases, start=1):
            case_results = case.get("results") if isinstance(case, dict) else None
            case_result = case_results.get(name) if isinstance(case_results, dict) else None
            if not isinstance(case_result, dict):
                raise AuditError(f"paired case {case_index} has no result for {name}")
            valid = case_result.get("valid")
            success = case_result.get("success")
            if not isinstance(valid, bool) or not isinstance(success, bool):
                raise AuditError(
                    f"paired case {case_index}/{name}: valid and success must be booleans"
                )
            invalid_count += int(not valid)
            success_count += int(valid and success)

        if result.get("successes_all_attempts") != success_count:
            raise AuditError(
                f"{name}: successes_all_attempts does not match paired case records"
            )
        if result.get("invalid_episodes") != invalid_count:
            raise AuditError(f"{name}: invalid_episodes does not match paired case records")

        condition_summaries.append(
            {
                "id": name,
                "label": CONDITION_LABELS.get(name, name),
                "method_note": CONDITION_METHOD_NOTES.get(name),
                "contract_evidence": condition_contracts[name],
                "attempts": attempts,
                "successes": success_count,
                "failures": attempts - success_count - invalid_count,
                "invalid": invalid_count,
                "success_rate": success_count / attempts,
            }
        )

    baseline_successes = condition_summaries[0]["successes"]
    if condition_summaries[0]["id"] != "baseline":
        raise AuditError("baseline must be the first condition")
    for summary in condition_summaries:
        summary["success_delta_vs_baseline"] = summary["successes"] - baseline_successes
        summary["success_rate_delta_vs_baseline"] = (
            summary["success_rate"] - condition_summaries[0]["success_rate"]
        )

    case_rows: list[dict[str, Any]] = []
    for case_number, case in enumerate(paired_cases, start=1):
        condition = case.get("paired_condition", {})
        row_results: dict[str, dict[str, Any]] = {}
        for name in condition_names:
            raw = case["results"][name]
            row_results[name] = {
                "valid": raw["valid"],
                "success": raw["success"],
                "termination_reason": raw.get("termination_reason"),
            }
        case_rows.append(
            {
                "case_number": case_number,
                "paired_condition_id": case.get("paired_condition_id"),
                "environment_index": condition.get("environment_index"),
                "episode_index": condition.get("episode_index"),
                "translation_bias_norm_um": case.get("translation_bias_norm_um"),
                "rotation_bias_norm_deg": case.get("rotation_bias_norm_deg"),
                "results": row_results,
            }
        )

    complete = set(learned_names) == set(LEARNED_CONDITIONS)
    training_seed = next(iter(training_seeds))
    if audit_schema_version == 1:
        update_claim = "runner config 기록상 20회 계획"
        update_evidence = "v1_runner_config_training_updates_not_checkpoint_payload"
        completed_updates_verified = False
    else:
        update_claim = "별도 inspector 파일 SHA 재대조·보고서상 완료 20회"
        update_evidence = EXPECTED_INSPECTION_LINK_STATUS
        completed_updates_verified = True
    return {
        "schema_version": 1,
        "visualization": "rj45_paired_pilot_ablation",
        "source": {
            "audit_path": str(audit_path.resolve()),
            "audit_sha256": audit_sha256,
            "audit_schema_version": audit.get("schema_version"),
            "audit_verification_status": scope.get("verification_status"),
        },
        "evaluation": {
            "task_id": experiment.get("task_id"),
            "evaluation_seed": experiment.get("seed"),
            "evaluation_start": experiment.get("evaluation_start"),
            "paired_attempts": attempts,
            "condition_count": len(condition_names),
            "learned_condition_count": len(learned_names),
            "expected_learned_condition_count": len(LEARNED_CONDITIONS),
            "complete_five_group_pilot": complete,
            "recorded_training_updates_each": EXPECTED_TRAINING_UPDATES,
            "training_update_claim": update_claim,
            "training_update_evidence": update_evidence,
            "checkpoint_completed_updates_verified_by_supplied_inspection": completed_updates_verified,
            "checkpoint_payload_reloaded_by_plotter": False,
            "checkpoint_inspection_files_rehashed_by_plotter": (
                audit_schema_version == 2
            ),
            "checkpoint_inspection_file_evidence": inspection_file_evidence,
            "common_training_seed": training_seed,
            "training_bundle_status": EXPECTED_BUNDLE_STATUS,
            "held_state_claim": "보고서상 동일 파지 상태 1개",
            "held_state_proof_sha256": next(iter(held_proofs)),
            "held_state_referenced_bytes_rehashed_by_plotter": False,
        },
        "conditions": condition_summaries,
        "paired_cases": case_rows,
        "limitations": [
            "개발 평가 결과이다.",
            f"각 학습 조건은 공통 seed {training_seed}의 {update_claim} 예비 학습 결과이다.",
            "보고서상 동일 파지 상태 1개이며, 파지 증거 파일 재해시는 그림 생성 범위 밖이다.",
            "다른 파지·재료·실물 환경으로의 일반화를 입증하지 않는다.",
        ],
    }


def _configure_matplotlib() -> tuple[Any, Any, Any]:
    try:
        import matplotlib
        from matplotlib import font_manager

        matplotlib.use("Agg")
        korean_font_name = None
        preferred_font_files = (
            "NanumGothic.ttf",
            "NotoSansCJK-Regular.ttc",
            "NotoSansKR-Regular.otf",
        )
        system_fonts = font_manager.findSystemFonts()
        for preferred_name in preferred_font_files:
            matching = [path for path in system_fonts if Path(path).name == preferred_name]
            if not matching:
                continue
            font_path = matching[0]
            font_manager.fontManager.addfont(font_path)
            korean_font_name = font_manager.FontProperties(fname=font_path).get_name()
            break
        sans_serif = ["DejaVu Sans"]
        if korean_font_name:
            sans_serif.insert(0, korean_font_name)
        matplotlib.rcParams.update(
            {
                "figure.facecolor": "white",
                "axes.facecolor": "white",
                "savefig.facecolor": "white",
                "font.family": "sans-serif",
                "font.sans-serif": sans_serif,
                "axes.unicode_minus": False,
                "svg.fonttype": "path",
                "svg.hashsalt": "rj45-paired-pilot-ablation-v1",
            }
        )
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise SystemExit("matplotlib is required to render PNG/SVG outputs") from exc
    return plt, ListedColormap, Patch


def _render(summary: dict[str, Any], png_path: Path, svg_path: Path) -> None:
    plt, ListedColormap, Patch = _configure_matplotlib()
    conditions = summary["conditions"]
    paired_cases = summary["paired_cases"]
    names = [entry["id"] for entry in conditions]
    labels = [entry["label"] for entry in conditions]
    attempts = summary["evaluation"]["paired_attempts"]
    successes = [entry["successes"] for entry in conditions]
    learned_count = summary["evaluation"]["learned_condition_count"]
    expected_learned_count = summary["evaluation"]["expected_learned_condition_count"]
    complete = summary["evaluation"]["complete_five_group_pilot"]
    training_seed = summary["evaluation"]["common_training_seed"]
    training_update_claim = summary["evaluation"]["training_update_claim"]
    completion_text = (
        "5군 전체 결과"
        if complete
        else f"부분 결과 ({learned_count}/{expected_learned_count} 학습군)"
    )
    limitation_text = (
        f"예비 실험 · {training_update_claim} · 공통 학습 seed {training_seed} 1개 · "
        "보고서상 동일 파지 상태 1개  |  파지 증거 파일 재해시는 그림 생성 범위 밖"
    )

    figure = plt.figure(figsize=(13.6, 14.8), dpi=160)
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=(1.0, 2.15),
        left=0.09,
        right=0.96,
        top=0.76,
        bottom=0.105,
        hspace=0.28,
    )
    bar_axis = figure.add_subplot(grid[0])
    matrix_axis = figure.add_subplot(grid[1])

    figure.suptitle(
        f"RJ45 삽입 PPO 파일럿 — {completion_text}",
        x=0.09,
        y=0.963,
        ha="left",
        fontsize=22,
        fontweight="bold",
        color="#0F172A",
    )
    figure.text(
        0.09,
        0.926,
        f"막대는 {attempts}개 전체 시도 기준 성공 수, 아래 행렬은 같은 보정 편향 사례에서의 조건별 결과입니다.",
        ha="left",
        fontsize=11.5,
        color="#475569",
    )
    figure.text(
        0.09,
        0.887,
        limitation_text,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
        color="#92400E",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FFF7ED",
            "edgecolor": "#FED7AA",
            "linewidth": 1.0,
        },
    )
    figure.text(
        0.09,
        0.835,
        "A1  할인 보상 + 실제 INSERT 전이·종료 처리 (복합 변경)\n"
        "A2  A1의 기존 potential을 형상 대응점 기반으로 교체\n"
        "Z1  결정론적 평균만 0으로 시작, 탐색 잡음 0.1 유지",
        ha="left",
        va="top",
        linespacing=1.55,
        fontsize=10.4,
        color="#1E3A5F",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#EFF6FF",
            "edgecolor": "#BFDBFE",
            "linewidth": 1.0,
        },
    )

    x_positions = list(range(len(conditions)))
    colors = [BAR_COLORS.get(name, "#64748B") for name in names]
    bars = bar_axis.bar(x_positions, successes, color=colors, width=0.64, zorder=3)
    baseline_successes = successes[0]
    bar_axis.axhline(
        baseline_successes,
        color=BAR_COLORS["baseline"],
        linewidth=1.4,
        linestyle=(0, (4, 3)),
        alpha=0.8,
        zorder=2,
    )
    for index, (bar, entry) in enumerate(zip(bars, conditions)):
        count_text = f"{entry['successes']}/{entry['attempts']}"
        delta = entry["success_delta_vs_baseline"]
        delta_text = "기준" if index == 0 else f"기준 대비 {delta:+d}"
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 2.4,
            count_text,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#0F172A",
        )
        bar_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            delta_text,
            ha="center",
            va="bottom",
            fontsize=8.8,
            color="#475569",
        )
    bar_axis.set_title("조건별 성공 수", loc="left", fontsize=15, fontweight="bold", pad=14)
    bar_axis.set_ylabel("성공 사례 수")
    bar_axis.set_xticks(x_positions, labels, fontsize=10)
    bar_axis.set_ylim(0, max(attempts + 2, max(successes) + 5))
    bar_axis.set_yticks(range(0, attempts + 1, 4))
    bar_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8, zorder=0)
    bar_axis.spines[["top", "right", "left"]].set_visible(False)
    bar_axis.spines["bottom"].set_color("#CBD5E1")
    bar_axis.tick_params(axis="y", length=0, colors="#64748B")
    bar_axis.tick_params(axis="x", length=0, pad=8)

    matrix: list[list[int]] = []
    for case in paired_cases:
        row: list[int] = []
        for name in names:
            outcome = case["results"][name]
            if not outcome["valid"]:
                row.append(2)
            elif outcome["success"]:
                row.append(1)
            else:
                row.append(0)
        matrix.append(row)

    colormap = ListedColormap(["#E8ECF2", "#16A34A", "#DC2626"])
    matrix_axis.imshow(matrix, cmap=colormap, vmin=0, vmax=2, aspect="auto")
    matrix_axis.set_title(
        "짝지은 사례별 결과",
        loc="left",
        fontsize=15,
        fontweight="bold",
        pad=14,
    )
    matrix_axis.set_xlabel("같은 열은 동일한 정책 조건")
    matrix_axis.set_ylabel("동일 보정 편향 사례")
    matrix_axis.set_xticks(range(len(labels)), labels, fontsize=10)
    matrix_axis.xaxis.tick_top()
    matrix_axis.xaxis.set_label_position("bottom")
    matrix_axis.set_yticks(
        range(len(paired_cases)),
        [f"{case['case_number']:02d}" for case in paired_cases],
        fontsize=7.8,
    )
    matrix_axis.set_xticks([value - 0.5 for value in range(1, len(labels))], minor=True)
    matrix_axis.set_yticks(
        [value - 0.5 for value in range(1, len(paired_cases))], minor=True
    )
    matrix_axis.grid(which="minor", color="white", linewidth=1.4)
    matrix_axis.tick_params(which="minor", bottom=False, left=False)
    matrix_axis.tick_params(axis="x", length=0, pad=10)
    matrix_axis.tick_params(axis="y", length=0, pad=6, colors="#64748B")
    for spine in matrix_axis.spines.values():
        spine.set_visible(False)

    legend = [
        Patch(facecolor="#16A34A", label="성공"),
        Patch(facecolor="#E8ECF2", label="실패"),
        Patch(facecolor="#DC2626", label="무효"),
    ]
    matrix_axis.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.075),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    audit_sha = summary["source"]["audit_sha256"]
    seed = summary["evaluation"]["evaluation_seed"]
    figure.text(
        0.96,
        0.027,
        f"평가 seed {seed}  ·  audit SHA-256 {audit_sha}",
        ha="right",
        va="bottom",
        fontsize=7.5,
        color="#64748B",
    )

    png_metadata = {
        "Title": "RJ45 paired pilot ablation",
        "Description": limitation_text,
        "Source": audit_sha,
    }
    svg_metadata = {
        "Title": "RJ45 paired pilot ablation",
        "Description": limitation_text,
        "Source": audit_sha,
        "Date": None,
    }
    figure.savefig(png_path, dpi=160, metadata=png_metadata)
    figure.savefig(svg_path, format="svg", metadata=svg_metadata)
    plt.close(figure)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _completed_bundle_manifest(
    summary: dict[str, Any], files: dict[str, Path]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "complete": True,
        "visualization": summary["visualization"],
        "audit_sha256": summary["source"]["audit_sha256"],
        "files": {
            key: {
                "filename": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for key, path in files.items()
        },
    }


def _is_owned_bundle(path: Path) -> bool:
    manifest_path = path / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not (
        isinstance(manifest, dict)
        and manifest.get("complete") is True
        and manifest.get("visualization") == "rj45_paired_pilot_ablation"
    ):
        return False
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != {"png", "svg", "data"}:
        return False
    expected_names = {"manifest.json"}
    for record in files.values():
        if not isinstance(record, dict):
            return False
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            return False
        expected_names.add(filename)
        artifact = path / filename
        if not artifact.is_file():
            return False
        artifact_bytes = artifact.read_bytes()
        if record.get("bytes") != len(artifact_bytes):
            return False
        if record.get("sha256") != hashlib.sha256(artifact_bytes).hexdigest():
            return False
    if len(expected_names) != 4:
        return False
    try:
        actual_names = {entry.name for entry in path.iterdir()}
    except OSError:
        return False
    return actual_names == expected_names


def _publish_bundle(
    staging_dir: Path,
    final_dir: Path,
    filenames: tuple[str, ...],
    overwrite: bool,
) -> None:
    if overwrite:
        backup_dir: Path | None = None
        moved_existing = False
        if final_dir.exists():
            if not final_dir.is_dir() or not _is_owned_bundle(final_dir):
                raise FileExistsError(
                    f"refusing to overwrite a directory not owned by this plotter: {final_dir}"
                )
            backup_dir = Path(
                tempfile.mkdtemp(prefix=f".{final_dir.name}.backup-", dir=final_dir.parent)
            )
            backup_dir.rmdir()
            os.rename(final_dir, backup_dir)
            moved_existing = True
        try:
            os.rename(staging_dir, final_dir)
        except BaseException:
            if moved_existing and backup_dir is not None and not final_dir.exists():
                os.rename(backup_dir, final_dir)
            raise
        if moved_existing and backup_dir is not None:
            shutil.rmtree(backup_dir)
        return

    try:
        final_dir.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise FileExistsError(f"output bundle already exists: {final_dir}") from exc

    created: list[Path] = []
    try:
        for filename in filenames:
            destination = final_dir / filename
            os.link(staging_dir / filename, destination)
            created.append(destination)
    except BaseException:
        for destination in reversed(created):
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        try:
            final_dir.rmdir()
        except OSError:
            pass
        raise
    shutil.rmtree(staging_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success counts and paired cases from an RJ45 pilot audit."
    )
    parser.add_argument("--audit", type=Path, required=True, help="paired audit JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="rj45-pilot-ablation")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="atomically replace only a completed bundle previously made by this script",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_path = args.audit.resolve()
    if not audit_path.is_file():
        raise SystemExit(f"audit does not exist: {audit_path}")
    if not args.stem or Path(args.stem).name != args.stem:
        raise SystemExit("--stem must be one filename component")

    try:
        audit_bytes = audit_path.read_bytes()
        audit = json.loads(audit_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read audit: {exc}") from exc
    if not isinstance(audit, dict):
        raise SystemExit("audit root must be an object")

    audit_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    try:
        summary = _validate_and_extract(audit, audit_path, audit_sha256)
    except AuditError as exc:
        raise SystemExit(f"audit validation failed: {exc}") from exc

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / args.stem
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{args.stem}.staging-", dir=output_dir)
    )
    staged = {
        "png": staging_dir / f"{args.stem}.png",
        "svg": staging_dir / f"{args.stem}.svg",
        "data": staging_dir / f"{args.stem}-data.json",
    }
    try:
        _render(summary, staged["png"], staged["svg"])
        _write_json(staged["data"], summary)
        manifest = _completed_bundle_manifest(summary, staged)
        manifest_path = staging_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        publish_order = tuple(path.name for path in staged.values()) + (
            manifest_path.name,
        )
        _publish_bundle(staging_dir, final_dir, publish_order, args.overwrite)
    except (FileExistsError, OSError) as exc:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise SystemExit(f"cannot publish output bundle: {exc}") from exc
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise

    destinations = {key: final_dir / path.name for key, path in staged.items()}

    print(
        json.dumps(
            {
                "audit_sha256": audit_sha256,
                "condition_successes": {
                    entry["id"]: entry["successes"] for entry in summary["conditions"]
                },
                "outputs": {key: str(value) for key, value in destinations.items()},
                "manifest": str(final_dir / "manifest.json"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
