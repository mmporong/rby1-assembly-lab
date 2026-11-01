"""Pure CPU contract for sealed classical-v2 tuning and fresh admission."""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class ClassicalV2ContractError(ValueError):
    """Raised when a classical-v2 plan or verified result differs from contract."""


ENVIRONMENT_SEED = 20261121
ACTOR_OBSERVATION_DIMENSION = 79
RIGHT_ACTION_DIMENSION = 6
BIAS_ABSOLUTE_TOLERANCE = 1.0e-9
TUNING_GRASP_ID = "rj45-train-03"
ADMISSION_GRASP_IDS = ("rj45-train-02", "rj45-train-05")
SIGNED_CASES = (
    "tx-positive",
    "tx-negative",
    "tz-positive",
    "tz-negative",
    "rx-positive",
    "rx-negative",
)
CANDIDATE_IDS = ("V2A", "V2B", "V2C", "V2D")
BIAS_CASES = {
    "tx-positive": ((0.00012, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "tx-negative": ((-0.00012, 0.0, 0.0), (0.0, 0.0, 0.0)),
    "tz-positive": ((0.0, 0.0, 0.00012), (0.0, 0.0, 0.0)),
    "tz-negative": ((0.0, 0.0, -0.00012), (0.0, 0.0, 0.0)),
    "rx-positive": ((0.0, 0.0, 0.0), (0.003490658503988659, 0.0, 0.0)),
    "rx-negative": ((0.0, 0.0, 0.0), (-0.003490658503988659, 0.0, 0.0)),
}

_CLASSICAL_PATH = Path(__file__).with_name("rj45_classical_v2.py").resolve()
_CLASSICAL_SOURCE_SHA256 = hashlib.sha256(_CLASSICAL_PATH.read_bytes()).hexdigest()
_CLASSICAL_MODULE_NAME = f"rby1_classical_v2_contract_source_{_CLASSICAL_SOURCE_SHA256[:16]}"
_SPEC = importlib.util.spec_from_file_location(_CLASSICAL_MODULE_NAME, _CLASSICAL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load classical-v2 source: {_CLASSICAL_PATH}")
_classical = importlib.util.module_from_spec(_SPEC)
sys.modules[_CLASSICAL_MODULE_NAME] = _classical
_SPEC.loader.exec_module(_classical)


_EXPECTED_CONFIG = {
    "schema": "rj45-classical-tuning-v2",
    "frozen": False,
    "provider_schema": "rj45-classical-force-search-controller-v2",
    "tuning_grasp_id": TUNING_GRASP_ID,
    "signed_cases": list(SIGNED_CASES),
    "candidate_count": 4,
    "episodes_per_candidate": 6,
    "tuning_episode_count": 24,
    "candidates": [
        {"id": "V2A", "position_gain": 1.0, "rotation_gain": 0.8,
         "search_step_m": 0.00005, "force_trigger_n": 0.25},
        {"id": "V2B", "position_gain": 1.0, "rotation_gain": 0.8,
         "search_step_m": 0.0001, "force_trigger_n": 0.25},
        {"id": "V2C", "position_gain": 1.0, "rotation_gain": 0.8,
         "search_step_m": 0.00005, "force_trigger_n": 0.5},
        {"id": "V2D", "position_gain": 1.0, "rotation_gain": 0.8,
         "search_step_m": 0.0001, "force_trigger_n": 0.5},
    ],
    "shared_controller_contract": {
        "control_hz": 30,
        "search_plateau_ticks": 18,
        "search_plateau_s": 0.6,
        "recovery_reset_advancing_ticks": 18,
        "stalled_trigger_ticks": 6,
        "stall_depth_delta_m_per_tick": 0.00002,
        "search_direction_order": [
            "initial", "rotate_plus_90", "opposite", "rotate_minus_90"
        ],
        "search_radius_max_m": 0.0006,
        "safety_force_n": 20.0,
        "safety_latch_is_permanent": True,
        "translation_bound_m": 0.0015,
        "rotation_bound_rad": 0.026179938779914945,
        "known_bias_access": False,
        "case_grasp_sign_ground_truth_access": False,
    },
    "selection": {
        "safe_eligibility": "zero_hard_and_forbidden_contact_episodes",
        "order": [
            "safe_eligible_only", "maximum_success_count",
            "minimum_evaluator_force_p95_n", "minimum_completion_time_median_s",
            "lexicographic_candidate_id",
        ],
        "strong_baseline_qualifier": {
            "minimum_successes": 4, "out_of": 6, "name_only": True,
            "does_not_gate_new_admission_baseline": True,
        },
        "selected_safe_baseline_always_runs_new_admission": {
            "grasp_count": 2, "signed_cases_per_grasp": 6, "episode_count": 12,
            "reuse_tuning_outcomes": False,
        },
    },
    "design_basis": {
        "old_search_plateau_ticks": 6,
        "old_search_plateau_s_at_30hz": 0.2,
        "success_dwell_s": 0.4,
        "old_plateau_shorter_than_success_dwell": True,
        "observed_diagnostic_force_peak_n": 0.55,
        "old_force_trigger_range_n": [6.0, 8.0],
        "stalled_branch_also_enters_search": True,
        "causal_attribution": "not_established",
    },
}


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ClassicalV2ContractError("value is not finite canonical JSON") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _same_json_types(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(
            _same_json_types(actual[key], expected[key]) for key in expected
        )
    if type(expected) is list:
        return len(actual) == len(expected) and all(
            _same_json_types(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return True


def _require_exact(actual: Any, expected: Any, label: str) -> None:
    if not _same_json_types(actual, expected) or _canonical(actual) != _canonical(expected):
        raise ClassicalV2ContractError(f"{label} differs from the sealed contract")


def _source_binding(config: Mapping[str, Any]) -> dict[str, Any]:
    candidates = []
    for candidate_id in CANDIDATE_IDS:
        provider = _classical.ClassicalForceSearchV2Provider(candidate_id)
        payload = provider.canonical_payload()
        if payload["candidate_id"] != candidate_id:
            raise ClassicalV2ContractError("provider candidate identity differs")
        candidates.append({
            "candidate_id": candidate_id,
            "controller_sha256": provider.controller_sha256(),
            "canonical_payload": payload,
        })
    base_sha = candidates[0]["canonical_payload"]["base_helper_source_sha256"]
    if any(row["canonical_payload"]["base_helper_source_sha256"] != base_sha
           for row in candidates):
        raise ClassicalV2ContractError("provider base source binding differs by candidate")
    registered = dict(config)
    registered["frozen"] = False
    return {
        "schema": "rj45-classical-v2-source-binding-v1",
        "config_canonical_sha256": _sha256(config),
        "registered_config_sha256": _sha256(registered),
        "classical_source_sha256": _CLASSICAL_SOURCE_SHA256,
        "base_helper_source_sha256": base_sha,
        "candidates": candidates,
    }


def validate_classical_v2_config(
    config: Mapping[str, Any], *, require_frozen: bool = False
) -> dict[str, Any]:
    """Validate every registered value, permitting only the false-to-true freeze."""

    if type(config) is not dict:
        raise ClassicalV2ContractError("config must be a plain JSON object")
    if type(config.get("frozen")) is not bool:
        raise ClassicalV2ContractError("config frozen must be boolean")
    expected = dict(_EXPECTED_CONFIG)
    expected["frozen"] = config["frozen"]
    _require_exact(config, expected, "config")
    if require_frozen and config["frozen"] is not True:
        raise ClassicalV2ContractError("classical-v2 contract is not frozen")

    binding = _source_binding(config)
    configured = {row["id"]: row for row in config["candidates"]}
    for row in binding["candidates"]:
        payload = row["canonical_payload"]
        candidate = configured[row["candidate_id"]]
        for key in ("position_gain", "rotation_gain", "search_step_m", "force_trigger_n"):
            _require_exact(candidate[key], payload[key], f"candidate {row['candidate_id']} {key}")
    shared = config["shared_controller_contract"]
    payload = binding["candidates"][0]["canonical_payload"]
    shared_payload_keys = {
        "search_plateau_ticks": "search_plateau_ticks",
        "recovery_reset_advancing_ticks": "recovery_reset_advancing_ticks",
        "stalled_trigger_ticks": "stalled_trigger_ticks",
        "stall_depth_delta_m_per_tick": "stall_depth_delta_m_per_tick",
        "search_direction_order": "search_direction_order",
        "search_radius_max_m": "search_radius_max_m",
        "safety_force_n": "safety_force_n",
        "safety_latch_is_permanent": "safety_latch_is_permanent",
        "translation_bound_m": "translation_bound_m",
        "known_bias_access": "known_bias_access",
        "case_grasp_sign_ground_truth_access": "case_grasp_sign_ground_truth_access",
    }
    for config_key, payload_key in shared_payload_keys.items():
        _require_exact(shared[config_key], payload[payload_key], f"shared {config_key}")
    return binding


def tuning_execution_plan(
    config: Mapping[str, Any], *, require_frozen: bool = False
) -> list[dict[str, Any]]:
    binding = validate_classical_v2_config(config, require_frozen=require_frozen)
    controller_shas = {
        row["candidate_id"]: row["controller_sha256"] for row in binding["candidates"]
    }
    result = []
    for candidate_id, case_id in itertools.product(CANDIDATE_IDS, SIGNED_CASES):
        translation_bias, rotation_bias = BIAS_CASES[case_id]
        result.append({
            "ordinal": len(result) + 1,
            "stage": "classical_v2_tuning",
            "candidate_id": candidate_id,
            "controller_sha256": controller_shas[candidate_id],
            "registered_config_sha256": binding["registered_config_sha256"],
            "grasp_id": TUNING_GRASP_ID,
            "case_id": case_id,
            "environment_seed": ENVIRONMENT_SEED,
            "requested_translation_bias_entry_m": list(translation_bias),
            "requested_rotation_bias_entry_rad": list(rotation_bias),
            "requested_applied_bias_absolute_tolerance": BIAS_ABSOLUTE_TOLERANCE,
        })
    return result


def _verified_metrics(
    config: Mapping[str, Any], verified_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if type(verified_rows) is not list:
        raise ClassicalV2ContractError("verified_rows must be a plain list")
    plan = tuning_execution_plan(config, require_frozen=True)
    binding = _source_binding(config)
    expected_config_sha = binding["registered_config_sha256"]
    expected_source_sha = binding["classical_source_sha256"]
    if len(verified_rows) != len(plan):
        raise ClassicalV2ContractError("selection requires exactly 24 verified tuning rows")
    unwrapped = []
    for expected_slot, wrapper in zip(plan, verified_rows, strict=True):
        if type(wrapper) is not dict or set(wrapper) != {"slot", "verification", "metrics"}:
            raise ClassicalV2ContractError("verified row wrapper fields differ")
        _require_exact(wrapper["slot"], expected_slot, "verified tuning slot")
        expected_verification = {
            "verified": True,
            "registered_config_sha256": expected_config_sha,
            "classical_source_sha256": expected_source_sha,
            "controller_sha256": expected_slot["controller_sha256"],
        }
        _require_exact(wrapper["verification"], expected_verification, "verification provenance")
        metrics = wrapper["metrics"]
        metric_keys = {
            "success", "hard_contact", "forbidden_contact",
            "evaluator_force_p95_n", "completion_time_s",
        }
        if type(metrics) is not dict or set(metrics) != metric_keys:
            raise ClassicalV2ContractError("verified metrics fields differ")
        for key in ("success", "hard_contact", "forbidden_contact"):
            if type(metrics[key]) is not bool:
                raise ClassicalV2ContractError(f"{key} must be boolean")
        for key in ("evaluator_force_p95_n", "completion_time_s"):
            if type(metrics[key]) is not float or not math.isfinite(metrics[key]) or metrics[key] < 0.0:
                raise ClassicalV2ContractError(f"{key} must be a finite nonnegative float")
        unwrapped.append({
            "candidate_id": expected_slot["candidate_id"],
            "grasp_id": expected_slot["grasp_id"],
            "case_id": expected_slot["case_id"],
            **metrics,
        })
    return unwrapped, binding


def select_and_build_admission_plan(
    config: Mapping[str, Any], verified_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Select from verified tuning rows and register a fresh 12-episode baseline."""

    unwrapped, binding = _verified_metrics(config, verified_rows)
    selected = _classical.select_classical_v2_candidate(unwrapped)
    candidate_id = selected["candidate_id"]
    controller_sha = next(
        row["controller_sha256"] for row in binding["candidates"]
        if row["candidate_id"] == candidate_id
    )
    episodes = []
    for grasp_id, case_id in itertools.product(ADMISSION_GRASP_IDS, SIGNED_CASES):
        translation_bias, rotation_bias = BIAS_CASES[case_id]
        episodes.append({
            "ordinal": len(episodes) + 1,
            "stage": "classical_v2_new_admission_baseline",
            "candidate_id": candidate_id,
            "controller_sha256": controller_sha,
            "grasp_id": grasp_id,
            "case_id": case_id,
            "environment_seed": ENVIRONMENT_SEED,
            "requested_translation_bias_entry_m": list(translation_bias),
            "requested_rotation_bias_entry_rad": list(rotation_bias),
            "requested_applied_bias_absolute_tolerance": BIAS_ABSOLUTE_TOLERANCE,
        })
    return {
        "schema": "rj45-classical-v2-admission-plan-v1",
        "diagnostic_only": True,
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
        "tuning_results_claimed_verified_by_planner": False,
        "reuse_tuning_outcomes": False,
        "common_control_contract": {
            "actor_observation_dimension": ACTOR_OBSERVATION_DIMENSION,
            "right_action_dimension": RIGHT_ACTION_DIMENSION,
            "fair_provider_inputs": ["observation79", "provider_owned_state"],
            "requested_applied_bias_absolute_tolerance": BIAS_ABSOLUTE_TOLERANCE,
        },
        "source_binding": binding,
        "selection": selected,
        "episodes": episodes,
    }


__all__ = [
    "ACTOR_OBSERVATION_DIMENSION",
    "ADMISSION_GRASP_IDS",
    "BIAS_ABSOLUTE_TOLERANCE",
    "BIAS_CASES",
    "CANDIDATE_IDS",
    "ClassicalV2ContractError",
    "ENVIRONMENT_SEED",
    "RIGHT_ACTION_DIMENSION",
    "SIGNED_CASES",
    "TUNING_GRASP_ID",
    "select_and_build_admission_plan",
    "tuning_execution_plan",
    "validate_classical_v2_config",
]
