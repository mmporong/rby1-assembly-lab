"""Replay and adjudicate controlled-observability v2 diagnostic evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_TOLERANCE = 1.0e-6
OFFSET_TOLERANCE_M = 1.0e-8
ACTION_OFFSET_SCALE_M = 0.0015
INSERT_PHASE_CODE = 4
ACTIVE_PHASES = {"probing", "contact_window", "returning"}


class ControlledObservabilityAnalysisError(RuntimeError):
    """Controlled-observability evidence is incomplete or inconsistent."""


def _load_sibling(filename: str, prefix: str) -> tuple[Any, Path, str]:
    path = Path(__file__).with_name(filename).resolve()
    source_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    module_name = f"{prefix}_{source_sha256[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, path, source_sha256


_axis_module, _AXIS_PROVIDER_PATH, _AXIS_PROVIDER_SOURCE_SHA256 = _load_sibling(
    "rj45_active_axis_probe.py", "rby1_controlled_analysis_axis_provider"
)
_contract_module, _CONTRACT_PATH, _CONTRACT_SOURCE_SHA256 = _load_sibling(
    "rj45_controlled_observability.py", "rby1_controlled_analysis_contract"
)
_BASE_PROVIDER_PATH = Path(__file__).with_name("rj45_active_z_probe.py").resolve()
_BASE_PROVIDER_SOURCE_SHA256 = hashlib.sha256(_BASE_PROVIDER_PATH.read_bytes()).hexdigest()
_ANALYSIS_PATH = Path(__file__).resolve()
_ANALYSIS_SOURCE_SHA256 = hashlib.sha256(_ANALYSIS_PATH.read_bytes()).hexdigest()
ActiveAxisProbeProvider = _axis_module.ActiveAxisProbeProvider


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ControlledObservabilityAnalysisError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ControlledObservabilityAnalysisError(f"{name} must be an integer")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ControlledObservabilityAnalysisError(f"{name} must be an integer") from error
    if converted != value:
        raise ControlledObservabilityAnalysisError(f"{name} must be an integer")
    return converted


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf":
        raise ControlledObservabilityAnalysisError(
            f"{name} must be a finite real vector of length {length}"
        )
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ControlledObservabilityAnalysisError(f"{name} must contain only finite values")
    return result


def _close(actual: np.ndarray, expected: np.ndarray, tolerance: float, name: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ControlledObservabilityAnalysisError(f"{name} differs from replay")


def _report_contract(
    report: Mapping[str, Any], expected_provider_origin: str | Path | None
) -> tuple[dict[str, Any], Mapping[str, Any], list[Mapping[str, Any]]]:
    if report.get("schema_name") != "rj45-controlled-observability-report-v2":
        raise ControlledObservabilityAnalysisError("report schema is not controlled-observability v2")
    if report.get("status") != "controlled_observability_complete":
        raise ControlledObservabilityAnalysisError("controlled-observability report is incomplete")
    if report.get("diagnostic_only") is not True:
        raise ControlledObservabilityAnalysisError("controlled-observability report is not diagnostic-only")
    config = _mapping(report.get("config"), "report.config")
    try:
        config_sha256 = _contract_module.validate_contract(config, require_frozen=True)
        plan = _contract_module.execution_plan(config, require_frozen=True)
    except (TypeError, ValueError) as error:
        raise ControlledObservabilityAnalysisError("report config is not the frozen v2 contract") from error
    if report.get("config_sha256") != config_sha256:
        raise ControlledObservabilityAnalysisError("report config SHA does not match the full config")
    case = _mapping(report.get("case"), "report.case")
    ordinal = _integer(case.get("ordinal"), "report.case.ordinal")
    if not 1 <= ordinal <= len(plan) or json.dumps(dict(case), sort_keys=True) != json.dumps(plan[ordinal - 1], sort_keys=True):
        raise ControlledObservabilityAnalysisError("report case is not its exact execution-plan slot")
    actual_environment_seed = _integer(
        report.get("actual_environment_seed"), "report.actual_environment_seed"
    )
    if actual_environment_seed != case["environment_seed"]:
        raise ControlledObservabilityAnalysisError("actual environment seed differs from the plan slot")
    seed_evidence_scope = report.get("seed_evidence_scope")
    if seed_evidence_scope != (
        "environment_cfg_seed_checked; stochastic_independence_not_demonstrated"
    ):
        raise ControlledObservabilityAnalysisError("seed evidence scope is missing or overstated")

    terminal = _mapping(report.get("episode_terminal"), "report.episode_terminal")
    if type(terminal.get("success")) is not bool:
        raise ControlledObservabilityAnalysisError("terminal success must be boolean")
    if terminal.get("hard_contact") is not False or terminal.get("forbidden_contact") is not False:
        raise ControlledObservabilityAnalysisError("hard or forbidden contact invalidates measurement")
    claims = _mapping(report.get("claims"), "report.claims")
    if dict(claims) != {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }:
        raise ControlledObservabilityAnalysisError("controlled report contains a false or extra claim")

    active = _mapping(report.get("active_axis_probe"), "report.active_axis_probe")
    provider = ActiveAxisProbeProvider(case["axis"])
    payload = provider.canonical_payload()
    if active.get("controller_payload") != payload:
        raise ControlledObservabilityAnalysisError("axis provider payload differs from replay")
    if active.get("controller_sha256") != provider.controller_sha256():
        raise ControlledObservabilityAnalysisError("axis provider controller SHA differs from replay")
    if active.get("provider_source_sha256") != _AXIS_PROVIDER_SOURCE_SHA256:
        raise ControlledObservabilityAnalysisError("axis provider source SHA differs from sibling bytes")
    if payload.get("base_provider_source_sha256") != _BASE_PROVIDER_SOURCE_SHA256:
        raise ControlledObservabilityAnalysisError("base provider source SHA provenance differs")
    try:
        report_origin = Path(active.get("provider_import_origin")).resolve()
        expected_origin = (
            _AXIS_PROVIDER_PATH
            if expected_provider_origin is None
            else Path(expected_provider_origin).resolve()
        )
    except TypeError as error:
        raise ControlledObservabilityAnalysisError("axis provider import origin is missing") from error
    if report_origin != expected_origin:
        raise ControlledObservabilityAnalysisError("axis provider import origin is unexpected")
    steps = active.get("steps")
    if not isinstance(steps, list):
        raise ControlledObservabilityAnalysisError("active_axis_probe.steps must be a list")
    metadata = {
        **dict(case),
        "config_sha256": config_sha256,
        "axis_provider_controller_sha256": provider.controller_sha256(),
        "axis_provider_source_sha256": _AXIS_PROVIDER_SOURCE_SHA256,
        "base_provider_source_sha256": _BASE_PROVIDER_SOURCE_SHA256,
        "contract_source_sha256": _CONTRACT_SOURCE_SHA256,
        "analysis_source_sha256": _ANALYSIS_SOURCE_SHA256,
        "provider_import_origin": str(report_origin),
        "actual_environment_seed": actual_environment_seed,
        "seed_evidence_scope": seed_evidence_scope,
    }
    return metadata, config, steps


def _validate_row_indices(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"rows[{index}]")
        if (
            _integer(row.get("sequence"), "row.sequence") != index
            or _integer(row.get("observer_pre_step_index"), "observer_pre_step_index") != index
            or _integer(row.get("observer_post_step_index"), "observer_post_step_index") != index + 1
        ):
            raise ControlledObservabilityAnalysisError("trace indices are not strictly monotonic")


def validate_controlled_episode(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    expected_provider_origin: str | Path | None = None,
) -> dict[str, Any]:
    """Replay one v2 episode and extract only registered observation evidence."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ControlledObservabilityAnalysisError("rows must be a non-empty sequence")
    report = _mapping(report, "report")
    metadata, config, steps = _report_contract(report, expected_provider_origin)
    _validate_row_indices(rows)
    provider = ActiveAxisProbeProvider(metadata["axis"])
    axis_index = provider.canonical_payload()["action_translation_index"]
    state = provider.reset()
    step_cursor = 0
    first_transition_seen = False
    contact_counts: list[int] = []
    contact_observations: list[np.ndarray] = []
    sample13_observation: np.ndarray | None = None
    zero_return_row: int | None = None
    postzero_observation: np.ndarray | None = None

    for row_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"rows[{row_index}]")
        if row.get("schema_name") != "rj45-contact-diagnostic-tick-v2":
            raise ControlledObservabilityAnalysisError("trace row schema is not tick v2")
        valid = row.get("pre_policy_input_valid")
        if type(valid) is not bool:
            raise ControlledObservabilityAnalysisError("pre_policy_input_valid must be boolean")
        observation = _vector(row.get("policy_observation79"), "policy_observation79", 79)
        trace_action = _vector(row.get("commanded_normalized_action6"), "commanded action", 6)
        requested = _vector(
            row.get("current_requested_expanded_residual_action12"), "requested action", 12
        )
        bounded = _vector(
            row.get("current_applied_bounded_expanded_action12"), "bounded action", 12
        )
        tip = _vector(row.get("current_applied_tip_target_offset_entry6"), "tip offset", 6)

        if not valid:
            if provider.state_telemetry(state)["phase"] in ACTIVE_PHASES:
                raise ControlledObservabilityAnalysisError("invalid observation interrupted active probe")
            for value in (trace_action, requested, bounded, tip):
                if not np.array_equal(value, np.zeros_like(value)):
                    raise ControlledObservabilityAnalysisError("invalid-row action and tip must be exact zero")
            continue
        if _integer(row.get("pre_phase_code"), "pre_phase_code") != INSERT_PHASE_CODE:
            raise ControlledObservabilityAnalysisError("valid provider row is outside INSERT phase")
        if step_cursor >= len(steps):
            raise ControlledObservabilityAnalysisError("trace has extra valid provider rows")
        step = _mapping(steps[step_cursor], f"active_axis_probe.steps[{step_cursor}]")
        if _integer(step.get("step_index"), "provider step index") != row_index:
            raise ControlledObservabilityAnalysisError("provider step index differs from trace")
        state_before = provider.state_telemetry(state)
        if step.get("state_before") != state_before:
            raise ControlledObservabilityAnalysisError("reported state_before differs from replay")
        if zero_return_row is not None and postzero_observation is None and state_before["phase"] == "done":
            postzero_observation = observation.copy()
        expected_action, state = provider.act(observation, state)
        state_after = provider.state_telemetry(state)
        if state_after["safety_latched"]:
            raise ControlledObservabilityAnalysisError("safety latch invalidates measurement")
        if step.get("state") != state_after:
            raise ControlledObservabilityAnalysisError("reported provider state differs from replay")
        reported_action = _vector(
            step.get("commanded_normalized_action6"), "reported provider action", 6
        )
        _close(reported_action, expected_action, ACTION_TOLERANCE, "reported provider action")
        _close(trace_action, expected_action, ACTION_TOLERANCE, "trace command")
        expected_expanded = np.concatenate((np.zeros(6), expected_action))
        _close(requested, expected_expanded, ACTION_TOLERANCE, "requested action")
        _close(bounded, expected_expanded, ACTION_TOLERANCE, "bounded action")
        expected_tip = np.zeros(6, dtype=np.float64)
        expected_tip[axis_index] = expected_action[axis_index] * ACTION_OFFSET_SCALE_M
        _close(tip, expected_tip, OFFSET_TOLERANCE_M, "tip offset")

        if state_before["phase"] == "waiting" and state_after["phase"] == "probing":
            if first_transition_seen or not (
                np.array_equal(trace_action, np.zeros(6))
                and np.array_equal(requested, np.zeros(12))
                and np.array_equal(bounded, np.zeros(12))
                and np.array_equal(tip, np.zeros(6))
            ):
                raise ControlledObservabilityAnalysisError("first INSERT transition was not exact zero")
            first_transition_seen = True

        contact_count = _integer(
            state_after["contact_window_observations"], "contact window count"
        )
        previous_count = contact_counts[-1] if contact_counts else 0
        if contact_count > previous_count:
            if contact_count != len(contact_counts) + 1:
                raise ControlledObservabilityAnalysisError("contact window count is not contiguous")
            contact_counts.append(contact_count)
            contact_observations.append(observation.copy())
            if contact_count == 13:
                sample13_observation = observation.copy()
        if zero_return_row is None and contact_count == 13 and state_after["phase"] == "done" and expected_action[axis_index] == 0.0:
            zero_return_row = row_index
        step_cursor += 1

    if step_cursor != len(steps):
        raise ControlledObservabilityAnalysisError("report has provider steps without valid trace rows")
    if contact_counts and contact_counts != list(range(1, 14)):
        raise ControlledObservabilityAnalysisError("contact window is partial")
    final_state = provider.state_telemetry(state)
    if not first_transition_seen or final_state["phase"] != "done":
        raise ControlledObservabilityAnalysisError("provider replay did not finish in done")

    measurement_pass = sample13_observation is not None
    sample13_feature = (
        _contract_module.feature48(sample13_observation).tolist()
        if sample13_observation is not None
        else None
    )
    median_feature = (
        np.median(
            np.stack([_contract_module.feature48(value) for value in contact_observations]),
            axis=0,
        ).tolist()
        if contact_observations
        else None
    )
    return {
        "schema_name": "rj45-controlled-observability-verified-episode-v2",
        "verified": True,
        "measurement_pass": measurement_pass,
        **metadata,
        "valid_provider_steps": step_cursor,
        "provider_final_phase": final_state["phase"],
        "contact_window_observations": len(contact_observations),
        "primary_contact_sample13_feature48": sample13_feature,
        "secondary_contact_sample13_observation79": (
            sample13_observation.tolist() if sample13_observation is not None else None
        ),
        "secondary_contact_window_median_feature48": median_feature,
        "secondary_first_observation_after_zero_return79": (
            postzero_observation.tolist() if postzero_observation is not None else None
        ),
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
    }


def adjudicate_controlled_observability(
    results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply registered three-seed separation statistics to exactly 24 episodes."""

    try:
        config_sha256 = _contract_module.validate_contract(config, require_frozen=True)
        plan = _contract_module.execution_plan(config, require_frozen=True)
    except (TypeError, ValueError) as error:
        raise ControlledObservabilityAnalysisError("adjudication config is not frozen v2") from error
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)) or len(results) != 24:
        raise ControlledObservabilityAnalysisError("adjudication requires exactly 24 episodes")
    indexed: dict[int, Mapping[str, Any]] = {}
    common_source: tuple[str, str, str, str] | None = None
    for result_index, raw_result in enumerate(results):
        result = _mapping(raw_result, f"results[{result_index}]")
        ordinal = _integer(result.get("ordinal"), "result ordinal")
        if ordinal in indexed or not 1 <= ordinal <= 24:
            raise ControlledObservabilityAnalysisError("result ordinal is duplicated or invalid")
        slot = plan[ordinal - 1]
        if json.dumps({key: result.get(key) for key in slot}, sort_keys=True) != json.dumps(slot, sort_keys=True):
            raise ControlledObservabilityAnalysisError("result does not match its execution-plan slot")
        if (
            result.get("schema_name") != "rj45-controlled-observability-verified-episode-v2"
            or result.get("verified") is not True
            or result.get("provider_final_phase") != "done"
            or result.get("config_sha256") != config_sha256
            or _integer(result.get("actual_environment_seed"), "actual environment seed") != slot["environment_seed"]
            or result.get("seed_evidence_scope")
            != "environment_cfg_seed_checked; stochastic_independence_not_demonstrated"
            or result.get("axis_provider_controller_sha256")
            != ActiveAxisProbeProvider(slot["axis"]).controller_sha256()
            or result.get("axis_provider_source_sha256") != _AXIS_PROVIDER_SOURCE_SHA256
            or result.get("base_provider_source_sha256") != _BASE_PROVIDER_SOURCE_SHA256
            or result.get("contract_source_sha256") != _CONTRACT_SOURCE_SHA256
            or result.get("analysis_source_sha256") != _ANALYSIS_SOURCE_SHA256
            or result.get("claims")
            != {"task_admission_completed": False, "rl_pilot_admitted": False}
        ):
            raise ControlledObservabilityAnalysisError("aggregate contains an unverified episode")
        source = (
            result.get("axis_provider_source_sha256"),
            result.get("base_provider_source_sha256"),
            result.get("contract_source_sha256"),
            result.get("analysis_source_sha256"),
        )
        if common_source is None:
            common_source = source
        elif source != common_source:
            raise ControlledObservabilityAnalysisError("episode source provenance differs")
        indexed[ordinal] = result
        contact_count = _integer(
            result.get("contact_window_observations"), "result contact window count"
        )
        measurement_pass = result.get("measurement_pass")
        primary = result.get("primary_contact_sample13_feature48")
        if (
            type(measurement_pass) is not bool
            or contact_count not in (0, 13)
            or measurement_pass != (contact_count == 13)
            or measurement_pass != (primary is not None)
        ):
            raise ControlledObservabilityAnalysisError("episode primary measurement status is inconsistent")

    statistics: dict[str, Any] = {}
    overall_pass = True
    seeds = list(_contract_module.environment_seeds())
    for axis in _contract_module.AXES:
        for grasp in _contract_module.GRASPS:
            group_name = f"{axis}-{grasp}"
            positive: list[list[float]] = []
            negative: list[list[float]] = []
            missing = []
            for sign, destination in (("positive", positive), ("negative", negative)):
                for seed in seeds:
                    slot = next(
                        item
                        for item in plan
                        if item["axis"] == axis
                        and item["grasp_id"] == grasp
                        and item["sign"] == sign
                        and item["environment_seed"] == seed
                    )
                    result = indexed[slot["ordinal"]]
                    sample = result.get("primary_contact_sample13_feature48")
                    if result.get("measurement_pass") is not True or sample is None:
                        missing.append({"sign": sign, "environment_seed": seed})
                    else:
                        destination.append(_vector(sample, "primary feature48", 48).tolist())
            if missing:
                group_pass = False
                stats = None
            else:
                try:
                    stats = _contract_module.separation_statistics(positive, negative)
                except ValueError as error:
                    raise ControlledObservabilityAnalysisError(
                        "registered separation statistic rejected primary samples"
                    ) from error
                group_pass = stats["separation_passed"] is True
            overall_pass = overall_pass and group_pass
            statistics[group_name] = {
                "axis": axis,
                "grasp_id": grasp,
                "seed_order": seeds,
                "measurement_pass": not missing,
                "separation_passed": group_pass,
                "missing_primary_samples": missing,
                "separation_statistics": stats,
            }
    return {
        "schema_name": "rj45-controlled-observability-adjudication-v2",
        "episode_count": 24,
        "config_sha256": config_sha256,
        "source_provenance": {
            "axis_provider_source_sha256": common_source[0],
            "base_provider_source_sha256": common_source[1],
            "contract_source_sha256": common_source[2],
            "analysis_source_sha256": common_source[3],
        },
        "pass": overall_pass,
        "seed_evidence_scope": (
            "environment_cfg_seed_checked; stochastic_independence_not_demonstrated"
        ),
        "separation_by_axis_and_grasp": statistics,
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
    }


__all__ = [
    "ControlledObservabilityAnalysisError",
    "adjudicate_controlled_observability",
    "validate_controlled_episode",
]
