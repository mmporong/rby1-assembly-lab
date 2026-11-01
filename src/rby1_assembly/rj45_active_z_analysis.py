"""CPU-only verification and adjudication for the active-Z diagnostic."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


_PROVIDER_PATH = Path(__file__).with_name("rj45_active_z_probe.py").resolve()
_PROVIDER_SOURCE_SHA256 = hashlib.sha256(_PROVIDER_PATH.read_bytes()).hexdigest()
_ANALYSIS_PATH = Path(__file__).resolve()
_ANALYSIS_SOURCE_SHA256 = hashlib.sha256(_ANALYSIS_PATH.read_bytes()).hexdigest()
_PROVIDER_MODULE_NAME = f"rby1_active_z_analysis_provider_{_PROVIDER_SOURCE_SHA256[:16]}"
_provider_spec = importlib.util.spec_from_file_location(_PROVIDER_MODULE_NAME, _PROVIDER_PATH)
if _provider_spec is None or _provider_spec.loader is None:
    raise RuntimeError(f"cannot load active-Z provider: {_PROVIDER_PATH}")
_provider_module = importlib.util.module_from_spec(_provider_spec)
sys.modules[_PROVIDER_MODULE_NAME] = _provider_module
_provider_spec.loader.exec_module(_provider_module)
ACTION_OFFSET_SCALE_M = _provider_module.ACTION_OFFSET_SCALE_M
ActiveZProbeProvider = _provider_module.ActiveZProbeProvider


ACTION_TOLERANCE = 1.0e-6
OFFSET_TOLERANCE_M = 1.0e-8
RETURN_STEP_M = 0.00015
CONTACT_THRESHOLD_N = 0.25
FORCE_SCALE_N = 8.0
SIGNAL_MINIMUM = 0.05
SIGNAL_TO_NOISE_MINIMUM = 2.0
NOISE_FLOOR = 1.0e-6
INSERT_PHASE_CODE = 4
EXPECTED_GRASPS = ("rj45-train-02", "rj45-train-05")
EXPECTED_CASES = ("tz-positive", "tz-negative")
EXPECTED_REPEATS = (1, 2)
ACTIVE_PHASES = {"probing", "contact_window", "returning"}


class ActiveZAnalysisError(RuntimeError):
    """The active-Z evidence is incomplete, forged, or inconsistent."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActiveZAnalysisError(f"{name} must be an object")
    return value


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf":
        raise ActiveZAnalysisError(f"{name} must be a real numeric vector of length {length}")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ActiveZAnalysisError(f"{name} must contain only finite values")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ActiveZAnalysisError(f"{name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ActiveZAnalysisError(f"{name} must be an integer") from error
    if result != value:
        raise ActiveZAnalysisError(f"{name} must be an integer")
    return result


def _close(actual: np.ndarray, expected: np.ndarray, tolerance: float, name: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ActiveZAnalysisError(f"{name} differs from the replayed absolute command")


def _expected_ordinal(grasp_id: str, case_id: str, repeat: int) -> int:
    try:
        return (
            EXPECTED_GRASPS.index(grasp_id) * 4
            + EXPECTED_CASES.index(case_id) * 2
            + EXPECTED_REPEATS.index(repeat)
            + 1
        )
    except ValueError as error:
        raise ActiveZAnalysisError("case is outside the sealed eight-episode active-Z plan") from error


def _validate_report_contract(
    report: Mapping[str, Any],
    provider: ActiveZProbeProvider,
    expected_provider_origin: str | Path | None,
) -> dict[str, Any]:
    if (
        report.get("schema_name") != "rj45-contact-diagnostic-report-v1"
        or report.get("status") != "diagnostic_complete"
        or report.get("diagnostic_only") is not True
    ):
        raise ActiveZAnalysisError("report is not a completed diagnostic-only report")
    case = _mapping(report.get("case"), "report.case")
    grasp_id = case.get("grasp_id")
    case_id = case.get("case_id")
    repeat = _integer(case.get("repeat"), "report.case.repeat")
    ordinal = _integer(report.get("ordinal"), "report.ordinal")
    if case.get("method_id") != "active_z_probe":
        raise ActiveZAnalysisError("report method is not active_z_probe")
    if case.get("provider_inputs") != ["observation79", "provider_owned_state"]:
        raise ActiveZAnalysisError("provider input authority differs from the sealed contract")
    if ordinal != _expected_ordinal(grasp_id, case_id, repeat):
        raise ActiveZAnalysisError("report ordinal does not match its grasp/case/repeat slot")

    terminal = _mapping(report.get("episode_terminal"), "report.episode_terminal")
    if terminal.get("hard_contact") is not False or terminal.get("forbidden_contact") is not False:
        raise ActiveZAnalysisError("hard or forbidden contact invalidates active-Z evidence")
    claims = _mapping(report.get("claims"), "report.claims")
    if claims.get("task_admission_completed") is not False or claims.get("rl_pilot_admitted") is not False:
        raise ActiveZAnalysisError("diagnostic report contains a false formal-admission claim")

    active = _mapping(report.get("active_z_probe"), "report.active_z_probe")
    if active.get("controller_payload") != provider.canonical_payload():
        raise ActiveZAnalysisError("controller payload differs from the current sealed provider")
    if active.get("controller_sha256") != provider.controller_sha256():
        raise ActiveZAnalysisError("controller SHA differs from the current sealed provider")
    source_path = Path(inspect.getfile(ActiveZProbeProvider)).resolve()
    if source_path != _PROVIDER_PATH:
        raise ActiveZAnalysisError("replay provider did not load from the sibling source file")
    if active.get("provider_source_sha256") != _PROVIDER_SOURCE_SHA256:
        raise ActiveZAnalysisError("provider source SHA does not match replayed source bytes")
    try:
        import_origin = Path(active.get("provider_import_origin")).resolve()
    except TypeError as error:
        raise ActiveZAnalysisError("provider import origin is missing") from error
    expected_origin = (
        source_path
        if expected_provider_origin is None
        else Path(expected_provider_origin).resolve()
    )
    if import_origin != expected_origin:
        raise ActiveZAnalysisError("provider import origin differs from the expected execution source")
    steps = active.get("steps")
    if not isinstance(steps, list):
        raise ActiveZAnalysisError("active_z.steps must be a list")
    return {
        "ordinal": ordinal,
        "grasp_id": grasp_id,
        "case_id": case_id,
        "repeat": repeat,
        "provider_import_origin": str(import_origin),
        "steps": steps,
    }


def validate_active_z_episode(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    expected_provider_origin: str | Path | None = None,
) -> dict[str, Any]:
    """Replay one raw episode and verify every provider and actuator linkage."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ActiveZAnalysisError("rows must be a non-empty sequence")
    report = _mapping(report, "report")
    provider = ActiveZProbeProvider()
    metadata = _validate_report_contract(report, provider, expected_provider_origin)
    steps = metadata.pop("steps")
    state = provider.reset()
    step_cursor = 0
    first_transition_seen = False
    contact_features: list[np.ndarray] = []
    contact_counts: list[int] = []
    observed_nonzero_levels: set[float] = set()

    for row_number, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"rows[{row_number}]")
        valid = row.get("pre_policy_input_valid")
        if type(valid) is not bool:
            raise ActiveZAnalysisError("pre_policy_input_valid must be boolean")
        row_step_index = _integer(
            row.get("observer_pre_step_index"), f"rows[{row_number}].observer_pre_step_index"
        )
        if not valid:
            if provider.state_telemetry(state)["phase"] in ACTIVE_PHASES:
                raise ActiveZAnalysisError("policy observation validity disappeared during active probe")
            for name, length in (
                ("commanded_normalized_action6", 6),
                ("current_requested_expanded_residual_action12", 12),
                ("current_applied_bounded_expanded_action12", 12),
                ("current_applied_tip_target_offset_entry6", 6),
            ):
                if not np.array_equal(_vector(row.get(name), name, length), np.zeros(length)):
                    raise ActiveZAnalysisError("invalid-input transition must have exact-zero action and tip")
            continue
        if step_cursor >= len(steps):
            raise ActiveZAnalysisError("trace has more valid observations than active_z.steps")
        if _integer(row.get("pre_phase_code"), "pre_phase_code") != INSERT_PHASE_CODE:
            raise ActiveZAnalysisError("valid active-Z provider step is outside INSERT phase")

        step = _mapping(steps[step_cursor], f"active_z.steps[{step_cursor}]")
        if _integer(step.get("step_index"), "active_z step_index") != row_step_index:
            raise ActiveZAnalysisError("trace and report provider step indices differ")
        observation = _vector(row.get("policy_observation79"), "policy_observation79", 79)
        state_before = provider.state_telemetry(state)
        if step.get("state_before") != state_before:
            raise ActiveZAnalysisError("reported provider state_before differs from replay")
        expected_action, state = provider.act(observation, state)
        state_after = provider.state_telemetry(state)
        if step.get("state") != state_after:
            raise ActiveZAnalysisError("reported provider state differs from replay")

        reported_action = _vector(
            step.get("commanded_normalized_action6"), "reported commanded action", 6
        )
        trace_action = _vector(
            row.get("commanded_normalized_action6"), "trace commanded action", 6
        )
        _close(reported_action, expected_action, ACTION_TOLERANCE, "reported action")
        _close(trace_action, expected_action, ACTION_TOLERANCE, "trace action")
        requested = _vector(
            row.get("current_requested_expanded_residual_action12"), "requested action", 12
        )
        bounded = _vector(
            row.get("current_applied_bounded_expanded_action12"), "bounded action", 12
        )
        expected_expanded = np.concatenate((np.zeros(6), expected_action))
        _close(requested, expected_expanded, ACTION_TOLERANCE, "requested action")
        _close(bounded, expected_expanded, ACTION_TOLERANCE, "bounded action")
        tip = _vector(
            row.get("current_applied_tip_target_offset_entry6"), "tip offset", 6
        )
        expected_tip = np.zeros(6, dtype=np.float64)
        expected_tip[2] = expected_action[2] * ACTION_OFFSET_SCALE_M
        _close(tip, expected_tip, OFFSET_TOLERANCE_M, "tip offset")

        if state_before["phase"] == "waiting" and state_after["phase"] == "probing":
            if first_transition_seen:
                raise ActiveZAnalysisError("provider entered probing more than once")
            first_transition_seen = True
            if not (
                np.array_equal(expected_action, np.zeros(6))
                and np.array_equal(trace_action, np.zeros(6))
                and np.array_equal(requested, np.zeros(12))
                and np.array_equal(bounded, np.zeros(12))
                and np.array_equal(tip, np.zeros(6))
            ):
                raise ActiveZAnalysisError("probe onset was not exact zero")

        if state_before["phase"] in {"contact_window", "returning"}:
            prior_offset_m = float(state_before["last_commanded_offset_m"])
            current_offset_m = float(expected_tip[2])
            if abs(current_offset_m - prior_offset_m) > RETURN_STEP_M + OFFSET_TOLERANCE_M:
                raise ActiveZAnalysisError("non-safety return exceeded 0.15 mm in one tick")

        if expected_action[2] != 0.0:
            observed_nonzero_levels.add(abs(float(expected_action[2])))
        contact_count = _integer(
            state_after["contact_window_observations"], "contact window count"
        )
        if contact_count > (contact_counts[-1] if contact_counts else 0):
            if contact_count != len(contact_counts) + 1:
                raise ActiveZAnalysisError("contact window count is not contiguous from 1 to 13")
            features = np.concatenate((observation[7:40], observation[63:78]))
            if features.shape != (48,):
                raise ActiveZAnalysisError("contact policy feature slice is not 48-D")
            contact_features.append(features)
            contact_counts.append(contact_count)
            if contact_count == 1:
                force_norm_n = float(np.linalg.norm(observation[37:40] * FORCE_SCALE_N))
                if force_norm_n < CONTACT_THRESHOLD_N:
                    raise ActiveZAnalysisError("contact window did not start at the force threshold")
        step_cursor += 1

    if step_cursor != len(steps):
        raise ActiveZAnalysisError("active_z.steps has entries without valid trace observations")
    final_state = provider.state_telemetry(state)
    if contact_counts and contact_counts != list(range(1, 14)):
        raise ActiveZAnalysisError("contact window is partial")
    if not first_transition_seen or final_state["phase"] != "done":
        raise ActiveZAnalysisError("provider replay did not reach a completed probe")
    median = (
        np.median(np.stack(contact_features, axis=0), axis=0).tolist()
        if contact_features
        else None
    )
    return {
        "schema_name": "rj45-active-z-verified-episode-v1",
        "verified": True,
        **metadata,
        "valid_provider_steps": step_cursor,
        "provider_final_phase": final_state["phase"],
        "contact_observed": bool(contact_features),
        "contact_window_observations": len(contact_features),
        "contact_policy_observation48_median": median,
        "observed_absolute_nonzero_z_levels": sorted(observed_nonzero_levels),
        "controller_sha256": provider.controller_sha256(),
        "provider_source_sha256": _PROVIDER_SOURCE_SHA256,
        "analysis_source_sha256": _ANALYSIS_SOURCE_SHA256,
        "hard_contact": False,
        "forbidden_contact": False,
        "formal_claims_preserved_false": True,
    }


def adjudicate_active_z(verified_episode_results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the sealed two-grasp separation/repeatability decision rule."""

    if not isinstance(verified_episode_results, Sequence) or isinstance(
        verified_episode_results, (str, bytes)
    ):
        raise ActiveZAnalysisError("verified_episode_results must be a sequence")
    if len(verified_episode_results) != 8:
        raise ActiveZAnalysisError("active-Z adjudication requires exactly eight episodes")
    indexed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, raw_result in enumerate(verified_episode_results):
        result = _mapping(raw_result, f"verified_episode_results[{index}]")
        if (
            result.get("schema_name") != "rj45-active-z-verified-episode-v1"
            or result.get("verified") is not True
            or result.get("provider_final_phase") != "done"
            or result.get("controller_sha256") != ActiveZProbeProvider().controller_sha256()
            or result.get("provider_source_sha256") != _PROVIDER_SOURCE_SHA256
            or result.get("analysis_source_sha256") != _ANALYSIS_SOURCE_SHA256
            or result.get("hard_contact") is not False
            or result.get("forbidden_contact") is not False
            or result.get("formal_claims_preserved_false") is not True
        ):
            raise ActiveZAnalysisError("aggregate input contains an unverified episode")
        contact_observed = result.get("contact_observed")
        contact_count = _integer(
            result.get("contact_window_observations"), "verified contact window count"
        )
        if type(contact_observed) is not bool or contact_count not in (0, 13) or contact_observed != (contact_count == 13):
            raise ActiveZAnalysisError("verified episode contact status is inconsistent")
        grasp_id = result.get("grasp_id")
        case_id = result.get("case_id")
        repeat = _integer(result.get("repeat"), "verified episode repeat")
        key = (grasp_id, case_id, repeat)
        _expected_ordinal(*key)
        if result.get("ordinal") != _expected_ordinal(*key) or key in indexed:
            raise ActiveZAnalysisError("aggregate episode slot is duplicated or forged")
        indexed[key] = result
    expected = {
        (grasp, case, repeat)
        for grasp in EXPECTED_GRASPS
        for case in EXPECTED_CASES
        for repeat in EXPECTED_REPEATS
    }
    if set(indexed) != expected:
        raise ActiveZAnalysisError("aggregate episode slots do not match the sealed plan")

    grasp_results: dict[str, Any] = {}
    overall_pass = True
    for grasp_id in EXPECTED_GRASPS:
        vectors: dict[tuple[str, int], np.ndarray] = {}
        missing_contact = []
        for case_id in EXPECTED_CASES:
            for repeat in EXPECTED_REPEATS:
                result = indexed[(grasp_id, case_id, repeat)]
                median = result.get("contact_policy_observation48_median")
                if result.get("contact_observed") is not True or median is None:
                    missing_contact.append(f"{case_id}:repeat-{repeat}")
                else:
                    vectors[(case_id, repeat)] = _vector(median, "contact median", 48)
        if missing_contact:
            grasp_pass = False
            signal = noise = ratio = None
        else:
            positive_1 = vectors[("tz-positive", 1)]
            positive_2 = vectors[("tz-positive", 2)]
            negative_1 = vectors[("tz-negative", 1)]
            negative_2 = vectors[("tz-negative", 2)]
            signal = float(np.linalg.norm(positive_1 - negative_1))
            noise = max(
                float(np.linalg.norm(positive_1 - positive_2)),
                float(np.linalg.norm(negative_1 - negative_2)),
                NOISE_FLOOR,
            )
            ratio = signal / noise
            if not all(math.isfinite(value) for value in (signal, noise, ratio)):
                raise ActiveZAnalysisError("aggregate active-Z metric is non-finite")
            grasp_pass = signal >= SIGNAL_MINIMUM and ratio >= SIGNAL_TO_NOISE_MINIMUM
        overall_pass = overall_pass and grasp_pass
        grasp_results[grasp_id] = {
            "pass": grasp_pass,
            "signal_d": signal,
            "repeat_noise_n": noise,
            "signal_to_noise_ratio": ratio,
            "missing_contact_windows": missing_contact,
            "checks": {
                "signal_at_least_0_05": signal is not None and signal >= SIGNAL_MINIMUM,
                "ratio_at_least_2": ratio is not None and ratio >= SIGNAL_TO_NOISE_MINIMUM,
            },
        }
    return {
        "schema_name": "rj45-active-z-adjudication-v1",
        "episode_count": 8,
        "pass": overall_pass,
        "thresholds": {
            "signal_minimum": SIGNAL_MINIMUM,
            "signal_to_noise_minimum": SIGNAL_TO_NOISE_MINIMUM,
            "noise_floor": NOISE_FLOOR,
        },
        "grasps": grasp_results,
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
        "analysis_source_sha256": _ANALYSIS_SOURCE_SHA256,
        "provider_source_sha256": _PROVIDER_SOURCE_SHA256,
    }


__all__ = [
    "ActiveZAnalysisError",
    "adjudicate_active_z",
    "validate_active_z_episode",
]
