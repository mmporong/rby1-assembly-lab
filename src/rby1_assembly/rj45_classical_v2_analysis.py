"""CPU-only raw replay validation for one classical-v2 episode."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


class ClassicalV2AnalysisError(RuntimeError):
    """Raw classical-v2 evidence is incomplete, forged, or inconsistent."""


_CONTRACT_PATH = Path(__file__).with_name("rj45_classical_v2_contract.py").resolve()
_CONTRACT_SOURCE_SHA256 = hashlib.sha256(_CONTRACT_PATH.read_bytes()).hexdigest()
_CONTRACT_MODULE_NAME = f"rby1_classical_v2_analysis_contract_{_CONTRACT_SOURCE_SHA256[:16]}"
_SPEC = importlib.util.spec_from_file_location(_CONTRACT_MODULE_NAME, _CONTRACT_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load classical-v2 contract: {_CONTRACT_PATH}")
_contract = importlib.util.module_from_spec(_SPEC)
sys.modules[_CONTRACT_MODULE_NAME] = _contract
_SPEC.loader.exec_module(_contract)
_provider_module = _contract._classical
ClassicalForceSearchV2Provider = _provider_module.ClassicalForceSearchV2Provider

_PROVIDER_PATH = Path(inspect.getfile(ClassicalForceSearchV2Provider)).resolve()
_PROVIDER_SOURCE_SHA256 = hashlib.sha256(_PROVIDER_PATH.read_bytes()).hexdigest()
_BASE_PATH = Path(__file__).with_name("rj45_comparison_providers.py").resolve()
_BASE_SOURCE_SHA256 = hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest()
_ANALYSIS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

INSERT_PHASE_CODE = 4
ENVIRONMENT_SEED = 20261121
ACTION_TOLERANCE = 1.0e-6
TIP_TOLERANCE = 1.0e-8
METRIC_TOLERANCE = 1.0e-5
TRANSLATION_SCALE_M = 0.0015
ROTATION_SCALE_RAD = math.pi / 120.0
BIAS_TOLERANCE = 1.0e-9
# Frozen task evaluation uses max_contact_force_n=25.0, and isaac_env.py
# classifies hard overload with a strict episode-peak > threshold predicate.
# Pinning the physical runtime/config source bytes remains the external operator's job.
HARD_CONTACT_FORCE_N = 25.0
FALSE_CLAIMS = {
    "task_admission_completed": False,
    "rl_pilot_admitted": False,
}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ClassicalV2AnalysisError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ClassicalV2AnalysisError(f"{name} must be a strict integer")
    return value


def _boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        raise ClassicalV2AnalysisError(f"{name} must be boolean")
    return value


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ClassicalV2AnalysisError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ClassicalV2AnalysisError(f"{name} must be finite numeric")
    return result


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (length,) or array.dtype.kind not in "iuf":
        raise ClassicalV2AnalysisError(
            f"{name} must be a real numeric vector of length {length}"
        )
    result = array.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ClassicalV2AnalysisError(f"{name} must contain finite values")
    return result


def _matrix_row(value: Any, name: str, length: int) -> np.ndarray:
    if not isinstance(value, list) or len(value) != 1:
        raise ClassicalV2AnalysisError(f"{name} must have one row")
    return _vector(value[0], name, length)


def _singleton_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if not isinstance(value, list) or len(value) != 1:
        raise ClassicalV2AnalysisError(f"{name} must be a singleton list")
    return _number(value[0], name, nonnegative=nonnegative)


def _exact_structure(actual: Any, expected: Any, name: str) -> None:
    try:
        same = (
            _contract._same_json_types(actual, expected)
            and json.dumps(actual, sort_keys=True, allow_nan=False)
            == json.dumps(expected, sort_keys=True, allow_nan=False)
        )
    except (TypeError, ValueError) as error:
        raise ClassicalV2AnalysisError(f"{name} is not finite canonical JSON") from error
    if not same:
        raise ClassicalV2AnalysisError(f"{name} differs")


def _close(actual: Any, expected: Any, tolerance: float, name: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ClassicalV2AnalysisError(f"{name} differs")


def _exact_zero(value: Any, name: str, length: int) -> None:
    if not np.array_equal(_vector(value, name, length), np.zeros(length)):
        raise ClassicalV2AnalysisError(f"{name} must be exact zero")


def _plan(config, stage, verified_tuning_rows):
    if stage == "tuning":
        if verified_tuning_rows is not None:
            raise ClassicalV2AnalysisError(
                "tuning validation cannot receive prior tuning rows"
            )
        return _contract.tuning_execution_plan(config, require_frozen=True)
    if stage == "admission":
        if verified_tuning_rows is None:
            raise ClassicalV2AnalysisError(
                "admission validation requires verified tuning rows"
            )
        try:
            selected = _contract.select_and_build_admission_plan(
                config, verified_tuning_rows
            )
        except (TypeError, ValueError) as error:
            raise ClassicalV2AnalysisError(
                "verified tuning rows do not reproduce the admission plan"
            ) from error
        return selected["episodes"]
    raise ClassicalV2AnalysisError("report stage must be tuning or admission")


def _report_contract(report, expected_provider_origin, verified_tuning_rows):
    if (
        report.get("schema_name") != "rj45-classical-v2-report-v1"
        or report.get("status") != "classical_v2_complete"
        or report.get("diagnostic_only") is not True
        or report.get("comparison_only") is not True
        or report.get("claims") != FALSE_CLAIMS
    ):
        raise ClassicalV2AnalysisError("report scope or false claims differ")
    config = _mapping(report.get("config"), "report.config")
    try:
        binding = _contract.validate_classical_v2_config(
            config, require_frozen=True
        )
    except (TypeError, ValueError) as error:
        raise ClassicalV2AnalysisError("report config contract differs") from error
    if report.get("config_sha256") != binding["registered_config_sha256"]:
        raise ClassicalV2AnalysisError("registered config SHA differs")
    stage = report.get("stage")
    plan = _plan(config, stage, verified_tuning_rows)
    ordinal = _integer(report.get("ordinal"), "report.ordinal")
    if not 1 <= ordinal <= len(plan):
        raise ClassicalV2AnalysisError("ordinal is outside the stage plan")
    slot = plan[ordinal - 1]
    _exact_structure(report.get("case"), slot, "report exact plan slot")
    actual_seed = _integer(
        report.get("actual_environment_seed"), "actual_environment_seed"
    )
    if actual_seed != ENVIRONMENT_SEED or slot["environment_seed"] != ENVIRONMENT_SEED:
        raise ClassicalV2AnalysisError("actual environment seed differs")

    provider = ClassicalForceSearchV2Provider(slot["candidate_id"])
    classical = _mapping(report.get("classical_v2"), "report.classical_v2")
    payload = provider.canonical_payload()
    _exact_structure(classical.get("controller_payload"), payload, "controller payload")
    if classical.get("controller_sha256") != provider.controller_sha256():
        raise ClassicalV2AnalysisError("controller SHA differs")
    if classical.get("provider_source_sha256") != _PROVIDER_SOURCE_SHA256:
        raise ClassicalV2AnalysisError("provider source SHA differs")
    if (classical.get("base_helper_source_sha256") != _BASE_SOURCE_SHA256
            or payload.get("base_helper_source_sha256") != _BASE_SOURCE_SHA256):
        raise ClassicalV2AnalysisError("base helper source SHA differs")
    try:
        origin = Path(classical.get("provider_import_origin")).resolve()
    except TypeError as error:
        raise ClassicalV2AnalysisError("provider import origin is missing") from error
    expected_origin = (
        _PROVIDER_PATH if expected_provider_origin is None
        else Path(expected_provider_origin).resolve()
    )
    if origin != expected_origin:
        raise ClassicalV2AnalysisError("provider import origin differs")
    provenance = _mapping(report.get("provenance"), "report.provenance")
    start_sources = _mapping(
        provenance.get("source_sha256_start"), "source_sha256_start"
    )
    end_sources = _mapping(
        provenance.get("source_sha256_end"), "source_sha256_end"
    )
    if start_sources != end_sources:
        raise ClassicalV2AnalysisError("start/end source bindings differ")
    expected_sources = {
        "classical_v2_contract": _CONTRACT_SOURCE_SHA256,
        "classical_v2_provider": _PROVIDER_SOURCE_SHA256,
        "classical_v2_base_provider": _BASE_SOURCE_SHA256,
    }
    if any(start_sources.get(key) != value for key, value in expected_sources.items()):
        raise ClassicalV2AnalysisError("classical-v2 source binding differs")
    steps = classical.get("steps")
    if not isinstance(steps, list):
        raise ClassicalV2AnalysisError("classical_v2.steps must be a list")
    return config, binding, slot, provider, steps, str(origin)


def _terminal_contract(
    report, slot, trace_p95, trace_completion, trace_grip_max, trace_insertion_max
):
    terminal = _mapping(
        report.get("validated_terminal_snapshot"),
        "validated_terminal_snapshot",
    )
    if terminal.get("terminal_mask") != [True]:
        raise ClassicalV2AnalysisError("terminal mask must be true")
    if terminal.get("independent_eval_valid") != [True]:
        raise ClassicalV2AnalysisError("independent evaluator must be valid")
    terminal_report = _mapping(report.get("episode_terminal"), "episode_terminal")
    for key in ("success", "hard_contact", "forbidden_contact"):
        _boolean(terminal_report.get(key), f"episode_terminal.{key}")
    outcome_fields = {
        "success": "independent_eval_success",
        "hard_contact": "hard_overload",
        "forbidden_contact": "forbidden_contact",
    }
    for report_key, snapshot_key in outcome_fields.items():
        if terminal.get(snapshot_key) != [terminal_report[report_key]]:
            raise ClassicalV2AnalysisError(
                f"terminal {report_key} differs from validated snapshot"
            )
    p95 = _number(
        terminal_report.get("evaluator_force_p95_n"),
        "episode_terminal.evaluator_force_p95_n", nonnegative=True,
    )
    completion = _number(
        terminal_report.get("completion_time_s"),
        "episode_terminal.completion_time_s", nonnegative=True,
    )
    _close(p95, trace_p95, METRIC_TOLERANCE, "trace force p95")
    _close(completion, trace_completion, METRIC_TOLERANCE, "completion timestamp")
    timestamp = _singleton_number(
        terminal.get("timestamp_s"), "terminal timestamp_s", nonnegative=True
    )
    _close(timestamp, completion, METRIC_TOLERANCE, "terminal timestamp")
    grip_peak = _singleton_number(
        terminal.get("grip_contact_force_peak_n"),
        "terminal grip_contact_force_peak_n", nonnegative=True,
    )
    insertion_peak = _singleton_number(
        terminal.get("insertion_contact_load_peak_n"),
        "terminal insertion_contact_load_peak_n", nonnegative=True,
    )
    if (grip_peak + METRIC_TOLERANCE < trace_grip_max
            or insertion_peak + METRIC_TOLERANCE < trace_insertion_max):
        raise ClassicalV2AnalysisError(
            "terminal force peak is below its raw trace channel maximum"
        )
    expected_hard = max(grip_peak, insertion_peak) > HARD_CONTACT_FORCE_N
    if terminal_report["hard_contact"] is not expected_hard:
        raise ClassicalV2AnalysisError(
            "hard contact differs from the frozen strict peak predicate"
        )
    if ((terminal_report["hard_contact"] or terminal_report["forbidden_contact"])
            and terminal_report["success"]):
        raise ClassicalV2AnalysisError(
            "unsafe terminal outcome cannot be independently successful"
        )

    admission = _mapping(terminal.get("task_admission"), "terminal.task_admission")
    if admission.get("applied_held_grasp_id") != slot["grasp_id"]:
        raise ClassicalV2AnalysisError("terminal applied grasp differs")
    expected_translation = np.asarray(
        slot["requested_translation_bias_entry_m"], dtype=np.float64
    )
    expected_rotation = np.asarray(
        slot["requested_rotation_bias_entry_rad"], dtype=np.float64
    )
    for prefix, expected in (
        ("translation", expected_translation), ("rotation", expected_rotation)
    ):
        requested = _matrix_row(
            admission.get(f"requested_{prefix}_bias_entry_m" if prefix == "translation"
                          else "requested_rotation_bias_entry_rad"),
            f"requested {prefix} bias", 3,
        )
        applied = _matrix_row(
            admission.get(f"applied_{prefix}_bias_entry_m" if prefix == "translation"
                          else "applied_rotation_bias_entry_rad"),
            f"applied {prefix} bias", 3,
        )
        _close(requested, expected, BIAS_TOLERANCE, f"requested {prefix} bias")
        _close(applied, requested, BIAS_TOLERANCE, f"applied {prefix} bias")
    for key in (
        "prefreeze_snapshot_valid", "insert_transition_snapshot_valid",
        "transition_contract_checked", "actual_applied_action_is_exact_zero",
    ):
        if admission.get(key) != [True]:
            raise ClassicalV2AnalysisError(f"terminal {key} must be true")
    if admission.get("evaluator_gt_available_to_fair_provider") is not False:
        raise ClassicalV2AnalysisError("fair provider evaluator-GT boundary differs")
    for key in (
        "requested_action_6d", "bounded_action_6d", "actual_applied_action_6d",
        "actual_applied_tip_target_offset_entry_6d",
    ):
        if not np.array_equal(_matrix_row(admission.get(key), key, 6), np.zeros(6)):
            raise ClassicalV2AnalysisError("first INSERT physical action is not exact zero")
    return {
        "success": terminal_report["success"],
        "hard_contact": terminal_report["hard_contact"],
        "forbidden_contact": terminal_report["forbidden_contact"],
        "evaluator_force_p95_n": p95,
        "completion_time_s": completion,
    }


def validate_classical_v2_episode(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    expected_provider_origin: str | Path | None = None,
    verified_tuning_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Replay provider/action timing and validate raw terminal evidence."""

    if (not isinstance(rows, Sequence) or isinstance(rows, (str, bytes))
            or not rows):
        raise ClassicalV2AnalysisError("rows must be a non-empty sequence")
    report = _mapping(report, "report")
    config, binding, slot, provider, steps, origin = _report_contract(
        report, expected_provider_origin, verified_tuning_rows
    )
    del config
    state = provider.reset()
    step_cursor = 0
    force_samples = []
    trace_grip_max = 0.0
    trace_insertion_max = 0.0
    terminal_count = 0
    trace_completion = None
    for sequence, raw in enumerate(rows):
        row = _mapping(raw, f"rows[{sequence}]")
        if row.get("schema_name") != "rj45-contact-diagnostic-tick-v2":
            raise ClassicalV2AnalysisError("trace row schema differs")
        if (
            _integer(row.get("sequence"), "row sequence") != sequence
            or _integer(row.get("observer_pre_step_index"), "pre step index") != sequence
            or _integer(row.get("observer_post_step_index"), "post step index") != sequence + 1
        ):
            raise ClassicalV2AnalysisError("trace indices are not strict contiguous integers")
        valid = _boolean(row.get("pre_policy_input_valid"), "pre_policy_input_valid")
        observation = _vector(
            row.get("policy_observation79"), "policy_observation79", 79
        )
        terminal = _boolean(row.get("terminal"), "terminal")
        terminal_count += int(terminal)
        if terminal != (sequence == len(rows) - 1):
            raise ClassicalV2AnalysisError("terminal trace row must be last")
        if terminal and row.get("post_state_source") != "pre_autoreset_terminal_snapshot":
            raise ClassicalV2AnalysisError("terminal trace source differs")
        trace_completion = _number(
            row.get("post_source_timestamp_s"), "post_source_timestamp_s",
            nonnegative=True,
        )
        grip = _vector(row.get("grip_contact_proxy_n"), "grip_contact_proxy_n", 2)
        insertion = _number(
            row.get("insertion_normal_load_sum_n"),
            "insertion_normal_load_sum_n", nonnegative=True,
        )
        if np.any(grip < 0.0):
            raise ClassicalV2AnalysisError("grip_contact_proxy_n must be nonnegative")
        trace_grip_max = max(trace_grip_max, float(grip.max()))
        trace_insertion_max = max(trace_insertion_max, insertion)
        force_samples.append(max(float(grip.max()), insertion))

        if not valid:
            for name, length in (
                ("commanded_normalized_action6", 6),
                ("current_requested_expanded_residual_action12", 12),
                ("current_applied_bounded_expanded_action12", 12),
                ("current_applied_tip_target_offset_entry6", 6),
            ):
                _exact_zero(row.get(name), name, length)
            continue
        if _integer(row.get("pre_phase_code"), "pre_phase_code") != INSERT_PHASE_CODE:
            raise ClassicalV2AnalysisError("valid provider row is outside INSERT phase")
        if step_cursor >= len(steps):
            raise ClassicalV2AnalysisError("trace has more valid rows than provider steps")
        step = _mapping(steps[step_cursor], f"classical_v2.steps[{step_cursor}]")
        if _integer(step.get("step_index"), "provider step_index") != sequence:
            raise ClassicalV2AnalysisError("provider step index differs from trace")
        before = provider.state_telemetry(state)
        _exact_structure(step.get("state_before"), before, "provider state_before")
        expected_action, state = provider.act(observation, state)
        after = provider.state_telemetry(state)
        _exact_structure(step.get("state"), after, "provider state")
        step_action = _vector(
            step.get("commanded_normalized_action6"), "provider commanded action", 6
        )
        row_action = _vector(
            row.get("commanded_normalized_action6"), "trace commanded action", 6
        )
        _close(step_action, expected_action, ACTION_TOLERANCE, "provider action")
        _close(row_action, expected_action, ACTION_TOLERANCE, "trace action")
        expanded = np.concatenate((np.zeros(6), expected_action))
        _close(
            _vector(row.get("current_requested_expanded_residual_action12"),
                    "requested expanded action", 12),
            expanded, ACTION_TOLERANCE, "requested expanded action",
        )
        _close(
            _vector(row.get("current_applied_bounded_expanded_action12"),
                    "bounded expanded action", 12),
            expanded, ACTION_TOLERANCE, "bounded expanded action",
        )
        tip_expected = np.concatenate((
            expected_action[:3] * TRANSLATION_SCALE_M,
            expected_action[3:] * ROTATION_SCALE_RAD,
        ))
        _close(
            _vector(row.get("current_applied_tip_target_offset_entry6"),
                    "tip target offset", 6),
            tip_expected, TIP_TOLERANCE, "tip target offset",
        )
        step_cursor += 1
    if terminal_count != 1 or trace_completion is None:
        raise ClassicalV2AnalysisError("trace requires exactly one terminal row")
    if step_cursor != len(steps):
        raise ClassicalV2AnalysisError("provider steps have no matching valid trace rows")
    force_p95 = float(np.percentile(
        np.asarray(force_samples, dtype=np.float64), 95, method="linear"
    ))
    metrics = _terminal_contract(
        report, slot, force_p95, trace_completion,
        trace_grip_max, trace_insertion_max,
    )
    verification = {
        "verified": True,
        "registered_config_sha256": binding["registered_config_sha256"],
        "classical_source_sha256": _PROVIDER_SOURCE_SHA256,
        "controller_sha256": provider.controller_sha256(),
    }
    wrapper = {
        "slot": slot,
        "verification": verification,
        "metrics": metrics,
    }
    return {
        "wrapper": wrapper,
        "raw_validation": {
            "provider_steps_replayed": step_cursor,
            "trace_rows_checked": len(rows),
            "trace_force_p95_n": force_p95,
            "trace_grip_contact_proxy_max_n": trace_grip_max,
            "trace_insertion_normal_load_sum_max_n": trace_insertion_max,
            "hard_contact_force_threshold_n": HARD_CONTACT_FORCE_N,
            "hard_contact_predicate": "max_terminal_peak_strictly_greater_than_25N",
            "hard_contact_threshold_runtime_config_closure_verified": False,
            "terminal_snapshot_cross_checked": True,
            "provider_import_origin": origin,
            "provider_source_sha256": _PROVIDER_SOURCE_SHA256,
            "base_helper_source_sha256": _BASE_SOURCE_SHA256,
            "contract_source_sha256": _CONTRACT_SOURCE_SHA256,
            "analysis_source_sha256": _ANALYSIS_SOURCE_SHA256,
            "archive_or_hash_chain_verified": False,
            "runtime_physical_closure_verified": False,
            "verified_tuning_rows_external_sha_pin_verified": False,
            "forbidden_contact_newly_proven_from_raw_trace": False,
        },
        "claims": dict(FALSE_CLAIMS),
    }


__all__ = [
    "ClassicalV2AnalysisError",
    "validate_classical_v2_episode",
]
