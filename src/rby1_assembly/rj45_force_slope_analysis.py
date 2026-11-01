"""Replay and adjudicate prospective force-slope observability evidence."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_TOLERANCE = 1.0e-6
OFFSET_TOLERANCE_M = 1.0e-8
ACTION_OFFSET_SCALE_M = 0.0015
INSERT_PHASE_CODE = 4
SEED_EVIDENCE_SCOPE = (
    "environment_cfg_seed_checked; stochastic_independence_not_demonstrated"
)
FALSE_CLAIMS = {
    "task_admission_completed": False,
    "rl_pilot_admitted": False,
}
VERIFICATION_SCOPE = {
    "semantic_provider_replay_verified": True,
    "trace_row_count_verified": True,
    "trace_sha256_verified": False,
    "hash_chain_verified": False,
    "archive_manifest_verified": False,
    "runtime_closure_verified": False,
    "external_verification_required": True,
    "external_verification_requirement": (
        "trace bytes, hash chain, archive manifest, and runtime closure require "
        "operator verification against external expected pins"
    ),
}


class ForceSlopeAnalysisError(RuntimeError):
    """Force-slope evidence is incomplete, forged, or inconsistent."""


def _load_sibling(filename: str, prefix: str) -> tuple[Any, Path, str]:
    path = Path(__file__).with_name(filename).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = importlib.util.spec_from_file_location(f"{prefix}_{digest[:16]}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load sibling module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module, path, digest


_contract, _CONTRACT_PATH, _CONTRACT_SOURCE_SHA256 = _load_sibling(
    "rj45_force_slope_observability.py", "rby1_force_slope_analysis_contract"
)
_slope, _SLOPE_PROVIDER_PATH, _SLOPE_PROVIDER_SOURCE_SHA256 = _load_sibling(
    "rj45_force_slope_probe.py", "rby1_force_slope_analysis_provider"
)
_axis, _AXIS_PROVIDER_PATH, _AXIS_PROVIDER_SOURCE_SHA256 = _load_sibling(
    "rj45_active_axis_probe.py", "rby1_force_slope_analysis_axis"
)
_controlled, _CONTROLLED_PATH, _CONTROLLED_SOURCE_SHA256 = _load_sibling(
    "rj45_controlled_observability.py", "rby1_force_slope_analysis_statistics"
)
_BASE_PROVIDER_PATH = Path(__file__).with_name("rj45_active_z_probe.py").resolve()
_BASE_PROVIDER_SOURCE_SHA256 = hashlib.sha256(_BASE_PROVIDER_PATH.read_bytes()).hexdigest()
_ANALYSIS_SOURCE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

ForceSlopeProbeProvider = _slope.ForceSlopeProbeProvider
ActiveAxisProbeProvider = _axis.ActiveAxisProbeProvider


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ForceSlopeAnalysisError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if type(value) is not int:
        raise ForceSlopeAnalysisError(f"{name} must be a strict integer")
    return value


def _number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ForceSlopeAnalysisError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ForceSlopeAnalysisError(f"{name} must be finite numeric")
    return result


def _vector(value: Any, name: str, length: int) -> np.ndarray:
    raw = np.asarray(value)
    if raw.shape != (length,) or raw.dtype.kind not in "iuf":
        raise ForceSlopeAnalysisError(f"{name} must be a real vector of length {length}")
    result = raw.astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise ForceSlopeAnalysisError(f"{name} must contain finite values")
    return result


def _close(actual: np.ndarray, expected: np.ndarray, tolerance: float, name: str) -> None:
    if not np.allclose(actual, expected, rtol=0.0, atol=tolerance):
        raise ForceSlopeAnalysisError(f"{name} differs from replay")


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise ForceSlopeAnalysisError("source bindings are not canonical JSON") from error
    return hashlib.sha256(encoded).hexdigest()


def _source_bindings(value: Any, name: str) -> dict[str, str]:
    raw = _mapping(value, name)
    if not raw:
        raise ForceSlopeAnalysisError(f"{name} must not be empty")
    result: dict[str, str] = {}
    for key, digest in raw.items():
        if (
            type(key) is not str
            or not key
            or type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ForceSlopeAnalysisError(f"{name} contains an invalid source binding")
        result[key] = digest
    return result


def _provider_for(case: Mapping[str, Any]) -> tuple[Any, Path, str, str]:
    if case.get("axis") == "tx" and case.get("provider_id") == "force_slope_probe":
        return (
            ForceSlopeProbeProvider(),
            _SLOPE_PROVIDER_PATH,
            _SLOPE_PROVIDER_SOURCE_SHA256,
            "force_slope_probe",
        )
    if case.get("axis") == "tz" and case.get("provider_id") == "active_axis_probe":
        return (
            ActiveAxisProbeProvider("tz"),
            _AXIS_PROVIDER_PATH,
            _AXIS_PROVIDER_SOURCE_SHA256,
            "active_axis_probe",
        )
    raise ForceSlopeAnalysisError("case provider does not match its registered axis")


def _report_contract(
    report: Mapping[str, Any], expected_provider_origin: str | Path | None
) -> tuple[dict[str, Any], list[Mapping[str, Any]], Any]:
    if report.get("schema_name") != "rj45-force-slope-observability-report-v3":
        raise ForceSlopeAnalysisError("report schema is not force-slope v3")
    if report.get("status") != "force_slope_observability_complete":
        raise ForceSlopeAnalysisError("force-slope report is incomplete")
    if report.get("diagnostic_only") is not True:
        raise ForceSlopeAnalysisError("force-slope report is not diagnostic-only")
    stage = report.get("stage")
    if type(stage) is not str or stage not in {"development", "confirmation"}:
        raise ForceSlopeAnalysisError("report stage is invalid")
    config = _mapping(report.get("config"), "report.config")
    try:
        config_sha256 = _contract.validate_contract(config, require_frozen=True)
        plan = _contract.execution_plan(config, stage, require_frozen=True)
    except (TypeError, ValueError) as error:
        raise ForceSlopeAnalysisError("report config is not the frozen v3 contract") from error
    if report.get("config_sha256") != config_sha256:
        raise ForceSlopeAnalysisError("report config SHA differs from the frozen config")
    case = _mapping(report.get("case"), "report.case")
    ordinal = _integer(report.get("ordinal"), "report.ordinal")
    if ordinal != _integer(case.get("ordinal"), "report.case.ordinal"):
        raise ForceSlopeAnalysisError("report ordinal differs from case ordinal")
    if not 1 <= ordinal <= len(plan) or json.dumps(
        dict(case), sort_keys=True, allow_nan=False
    ) != json.dumps(plan[ordinal - 1], sort_keys=True, allow_nan=False):
        raise ForceSlopeAnalysisError("report case is not its exact execution-plan slot")
    seed = _integer(report.get("actual_environment_seed"), "actual environment seed")
    if seed != case["environment_seed"]:
        raise ForceSlopeAnalysisError("actual environment seed differs from the plan slot")
    if report.get("seed_evidence_scope") != SEED_EVIDENCE_SCOPE:
        raise ForceSlopeAnalysisError("seed evidence scope is missing or overstated")
    claims = _mapping(report.get("claims"), "report.claims")
    if dict(claims) != FALSE_CLAIMS:
        raise ForceSlopeAnalysisError("report contains a false or extra claim")

    terminal = _mapping(report.get("episode_terminal"), "episode_terminal")
    if set(terminal) != {"success", "hard_contact", "forbidden_contact", "completion_time_s"}:
        raise ForceSlopeAnalysisError("episode terminal fields are not exact")
    for field in ("success", "hard_contact", "forbidden_contact"):
        if type(terminal.get(field)) is not bool:
            raise ForceSlopeAnalysisError(f"terminal {field} must be boolean")
    if (
        terminal.get("success") is True
        and (terminal.get("hard_contact") is True or terminal.get("forbidden_contact") is True)
    ):
        raise ForceSlopeAnalysisError("unsafe terminal outcome cannot be successful")
    _number(terminal.get("completion_time_s"), "completion_time_s", nonnegative=True)
    _mapping(report.get("trace"), "report.trace")

    provenance = _mapping(report.get("provenance"), "report.provenance")
    start = _source_bindings(provenance.get("source_sha256_start"), "source_sha256_start")
    end = _source_bindings(provenance.get("source_sha256_end"), "source_sha256_end")
    if start != end:
        raise ForceSlopeAnalysisError("source bindings changed during execution")
    reviewed = provenance.get("reviewed_source_freeze_sha256")
    if type(reviewed) is not str or reviewed != _canonical_sha256(start):
        raise ForceSlopeAnalysisError("reviewed source freeze SHA differs from source bindings")

    probe = _mapping(report.get("probe"), "report.probe")
    provider, provider_path, provider_source, provider_id = _provider_for(case)
    if probe.get("controller_payload") != provider.canonical_payload():
        raise ForceSlopeAnalysisError("provider payload differs from replay")
    if probe.get("controller_sha256") != provider.controller_sha256():
        raise ForceSlopeAnalysisError("provider controller SHA differs from replay")
    if probe.get("provider_source_sha256") != provider_source:
        raise ForceSlopeAnalysisError("provider source SHA differs from sibling bytes")
    required_sources = {
        "force_slope_analysis": _ANALYSIS_SOURCE_SHA256,
        "force_slope_contract": _CONTRACT_SOURCE_SHA256,
        "force_slope_provider": _SLOPE_PROVIDER_SOURCE_SHA256,
        "active_axis_provider": _AXIS_PROVIDER_SOURCE_SHA256,
        "active_z_base_provider": _BASE_PROVIDER_SOURCE_SHA256,
        "controlled_v2_contract": _CONTROLLED_SOURCE_SHA256,
    }
    if any(start.get(label) != digest for label, digest in required_sources.items()):
        raise ForceSlopeAnalysisError("reviewed source bindings omit a required exact source")
    if provider_id == "active_axis_probe" and (
        provider.canonical_payload().get("base_provider_source_sha256")
        != _BASE_PROVIDER_SOURCE_SHA256
    ):
        raise ForceSlopeAnalysisError("active-axis base provider source differs")
    try:
        report_origin = Path(probe.get("provider_import_origin")).resolve()
        expected_origin = (
            provider_path
            if expected_provider_origin is None
            else Path(expected_provider_origin).resolve()
        )
    except TypeError as error:
        raise ForceSlopeAnalysisError("provider import origin is missing") from error
    if report_origin != expected_origin:
        raise ForceSlopeAnalysisError("provider import origin is unexpected")
    steps = probe.get("steps")
    if not isinstance(steps, list):
        raise ForceSlopeAnalysisError("probe.steps must be a list")

    metadata = {
        **dict(case),
        "stage": stage,
        "config_sha256": config_sha256,
        "actual_environment_seed": seed,
        "seed_evidence_scope": SEED_EVIDENCE_SCOPE,
        "provider_controller_sha256": provider.controller_sha256(),
        "provider_source_sha256": provider_source,
        "provider_import_origin": str(report_origin),
        "contract_source_sha256": _CONTRACT_SOURCE_SHA256,
        "analysis_source_sha256": _ANALYSIS_SOURCE_SHA256,
        "reviewed_source_freeze_sha256": reviewed,
        "episode_terminal": dict(terminal),
    }
    return metadata, steps, provider


def _validate_row_indices(rows: Sequence[Mapping[str, Any]]) -> None:
    for index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"rows[{index}]")
        if (
            _integer(row.get("sequence"), "row.sequence") != index
            or _integer(row.get("observer_pre_step_index"), "observer_pre_step_index")
            != index
            or _integer(row.get("observer_post_step_index"), "observer_post_step_index")
            != index + 1
        ):
            raise ForceSlopeAnalysisError("trace indices are not strictly monotonic")


def validate_episode(
    rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    *,
    expected_provider_origin: str | Path | None = None,
) -> dict[str, Any]:
    """Replay every provider row and extract only the registered sample13 feature."""

    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ForceSlopeAnalysisError("rows must be a non-empty sequence")
    report = _mapping(report, "report")
    metadata, steps, provider = _report_contract(report, expected_provider_origin)
    trace = _mapping(report.get("trace"), "report.trace")
    if _integer(trace.get("row_count"), "trace.row_count") != len(rows):
        raise ForceSlopeAnalysisError("trace row count differs from supplied rows")
    _validate_row_indices(rows)
    axis_index = provider.canonical_payload()["action_translation_index"]
    state = provider.reset()
    step_cursor = 0
    first_transition_seen = False
    contact_counts: list[int] = []
    sample13_observation: np.ndarray | None = None
    active_phases = {
        "impulse_pending",
        "impulse_responses",
        "decision",
        "sweep",
        "probing",
        "contact_window",
        "returning",
    }

    for row_index, raw_row in enumerate(rows):
        row = _mapping(raw_row, f"rows[{row_index}]")
        if row.get("schema_name") != "rj45-contact-diagnostic-tick-v2":
            raise ForceSlopeAnalysisError("trace row schema is not tick v2")
        valid = row.get("pre_policy_input_valid")
        if type(valid) is not bool:
            raise ForceSlopeAnalysisError("pre_policy_input_valid must be boolean")
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
            if provider.state_telemetry(state)["phase"] in active_phases:
                raise ForceSlopeAnalysisError("invalid observation interrupted active probe")
            for value in (trace_action, requested, bounded, tip):
                if not np.array_equal(value, np.zeros_like(value)):
                    raise ForceSlopeAnalysisError("invalid-row action and tip must be exact zero")
            continue
        if _integer(row.get("pre_phase_code"), "pre_phase_code") != INSERT_PHASE_CODE:
            raise ForceSlopeAnalysisError("valid provider row is outside INSERT phase")
        if step_cursor >= len(steps):
            raise ForceSlopeAnalysisError("trace has extra valid provider rows")
        step = _mapping(steps[step_cursor], f"probe.steps[{step_cursor}]")
        if _integer(step.get("step_index"), "provider step index") != row_index:
            raise ForceSlopeAnalysisError("provider step index differs from trace")
        state_before = provider.state_telemetry(state)
        if step.get("state_before") != state_before:
            raise ForceSlopeAnalysisError("reported state_before differs from replay")
        expected_action, state = provider.act(observation, state)
        state_after = provider.state_telemetry(state)
        if step.get("state") != state_after:
            raise ForceSlopeAnalysisError("reported provider state differs from replay")
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

        onset_phase = (
            "impulse_pending"
            if metadata["provider_id"] == "force_slope_probe"
            else "probing"
        )
        if state_before["phase"] == "waiting" and state_after["phase"] == onset_phase:
            if first_transition_seen or not all(
                np.array_equal(value, np.zeros_like(value))
                for value in (trace_action, requested, bounded, tip)
            ):
                raise ForceSlopeAnalysisError("first INSERT transition was not exact zero")
            first_transition_seen = True

        contact_count = _integer(
            state_after["contact_window_observations"], "contact window count"
        )
        previous_count = contact_counts[-1] if contact_counts else 0
        if contact_count > previous_count:
            if contact_count != len(contact_counts) + 1:
                raise ForceSlopeAnalysisError("contact window count is not contiguous")
            contact_counts.append(contact_count)
            if contact_count == 13:
                sample13_observation = observation.copy()
        step_cursor += 1

    if step_cursor != len(steps):
        raise ForceSlopeAnalysisError("report has provider steps without valid trace rows")
    final_state = provider.state_telemetry(state)
    terminal = metadata["episode_terminal"]
    safe = terminal["hard_contact"] is False and terminal["forbidden_contact"] is False
    final_phase = final_state["phase"]
    if not first_transition_seen or final_phase not in {
        "done",
        "invalid_latched",
        "safety_latched",
    }:
        raise ForceSlopeAnalysisError("provider replay did not reach a terminal phase")
    if contact_counts and contact_counts != list(range(1, 14)) and final_phase == "done":
        raise ForceSlopeAnalysisError("completed provider has a partial contact window")
    measurement_pass = sample13_observation is not None and safe and final_phase == "done"
    provider_measurement_status = final_state.get("measurement_status")
    if measurement_pass:
        measurement_status = "valid"
    elif final_phase == "safety_latched":
        measurement_status = "invalid_safety_force"
    elif final_phase == "invalid_latched":
        measurement_status = (
            provider_measurement_status
            if provider_measurement_status
            in {"invalid_zero_slope", "invalid_phase_or_active_loss"}
            else "invalid_phase_or_active_loss"
        )
    elif sample13_observation is None:
        measurement_status = "invalid_no_projected_contact"
    else:
        measurement_status = "invalid_unsafe_terminal"
    return {
        "schema_name": "rj45-force-slope-verified-episode-v3",
        "verified": True,
        **metadata,
        "valid_provider_steps": step_cursor,
        "provider_final_phase": final_phase,
        "provider_measurement_status": provider_measurement_status,
        "measurement_status": measurement_status,
        "contact_window_observations": len(contact_counts),
        "measurement_pass": measurement_pass,
        "primary_contact_sample13_feature48": (
            _controlled.feature48(sample13_observation).tolist()
            if sample13_observation is not None
            else None
        ),
        "verification_scope": dict(VERIFICATION_SCOPE),
        "claims": dict(FALSE_CLAIMS),
    }


def adjudicate_stage(
    config: Mapping[str, Any], stage: str, results: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Adjudicate exactly one frozen development or confirmation schedule."""

    try:
        config_sha256 = _contract.validate_contract(config, require_frozen=True)
        plan = _contract.execution_plan(config, stage, require_frozen=True)
    except (TypeError, ValueError) as error:
        raise ForceSlopeAnalysisError("adjudication stage or frozen config is invalid") from error
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ForceSlopeAnalysisError("results must be a sequence")
    if len(results) != len(plan):
        raise ForceSlopeAnalysisError(f"{stage} requires exactly {len(plan)} episodes")
    indexed: dict[int, Mapping[str, Any]] = {}
    reviewed_source: str | None = None
    controller_sources: dict[str, dict[str, str]] = {}
    for result_index, raw_result in enumerate(results):
        result = _mapping(raw_result, f"results[{result_index}]")
        ordinal = _integer(result.get("ordinal"), "result ordinal")
        if ordinal in indexed or not 1 <= ordinal <= len(plan):
            raise ForceSlopeAnalysisError("result ordinal is duplicated or invalid")
        slot = plan[ordinal - 1]
        if json.dumps(
            {key: result.get(key) for key in slot}, sort_keys=True, allow_nan=False
        ) != json.dumps(slot, sort_keys=True, allow_nan=False):
            raise ForceSlopeAnalysisError("result does not match its execution-plan slot")
        if (
            result.get("schema_name") != "rj45-force-slope-verified-episode-v3"
            or result.get("verified") is not True
            or result.get("provider_final_phase")
            not in {"done", "invalid_latched", "safety_latched"}
            or result.get("config_sha256") != config_sha256
            or _integer(result.get("actual_environment_seed"), "actual environment seed")
            != slot["environment_seed"]
            or result.get("seed_evidence_scope") != SEED_EVIDENCE_SCOPE
            or result.get("contract_source_sha256") != _CONTRACT_SOURCE_SHA256
            or result.get("analysis_source_sha256") != _ANALYSIS_SOURCE_SHA256
            or result.get("verification_scope") != VERIFICATION_SCOPE
            or result.get("claims") != FALSE_CLAIMS
        ):
            raise ForceSlopeAnalysisError("aggregate contains an unverified episode")
        expected_provider, _, expected_source, provider_id = _provider_for(slot)
        if (
            result.get("provider_controller_sha256")
            != expected_provider.controller_sha256()
            or result.get("provider_source_sha256") != expected_source
        ):
            raise ForceSlopeAnalysisError("aggregate provider provenance differs")
        source = result.get("reviewed_source_freeze_sha256")
        if (
            type(source) is not str
            or len(source) != 64
            or any(character not in "0123456789abcdef" for character in source)
        ):
            raise ForceSlopeAnalysisError("aggregate reviewed source SHA is invalid")
        if reviewed_source is None:
            reviewed_source = source
        elif source != reviewed_source:
            raise ForceSlopeAnalysisError("episode reviewed source freezes differ")
        identity = {
            "controller_sha256": expected_provider.controller_sha256(),
            "provider_source_sha256": expected_source,
            "provider_import_origin": result.get("provider_import_origin"),
        }
        if (
            type(identity["provider_import_origin"]) is not str
            or not identity["provider_import_origin"]
        ):
            raise ForceSlopeAnalysisError("aggregate provider origin is invalid")
        if provider_id in controller_sources and controller_sources[provider_id] != identity:
            raise ForceSlopeAnalysisError("provider identity changed within the stage")
        controller_sources[provider_id] = identity
        count = _integer(result.get("contact_window_observations"), "contact count")
        passed = result.get("measurement_pass")
        primary = result.get("primary_contact_sample13_feature48")
        terminal = _mapping(result.get("episode_terminal"), "episode terminal")
        for field in ("success", "hard_contact", "forbidden_contact"):
            if type(terminal.get(field)) is not bool:
                raise ForceSlopeAnalysisError("aggregate terminal flags must be boolean")
        if terminal["success"] and (terminal["hard_contact"] or terminal["forbidden_contact"]):
            raise ForceSlopeAnalysisError("aggregate contains an unsafe successful episode")
        _number(terminal.get("completion_time_s"), "completion_time_s", nonnegative=True)
        safe = terminal.get("hard_contact") is False and terminal.get("forbidden_contact") is False
        expected_pass = (
            count == 13
            and primary is not None
            and safe
            and result.get("provider_final_phase") == "done"
        )
        if (
            type(passed) is not bool
            or not 0 <= count <= 13
            or (primary is None) != (count < 13)
            or passed != expected_pass
        ):
            raise ForceSlopeAnalysisError("episode measurement status is inconsistent")
        if expected_pass:
            expected_status = "valid"
        elif result.get("provider_final_phase") == "safety_latched":
            expected_status = "invalid_safety_force"
        elif result.get("provider_final_phase") == "invalid_latched":
            provider_status = result.get("provider_measurement_status")
            expected_status = (
                provider_status
                if provider_status in {"invalid_zero_slope", "invalid_phase_or_active_loss"}
                else "invalid_phase_or_active_loss"
            )
        elif primary is None:
            expected_status = "invalid_no_projected_contact"
        else:
            expected_status = "invalid_unsafe_terminal"
        if result.get("measurement_status") != expected_status:
            raise ForceSlopeAnalysisError("episode measurement failure reason is inconsistent")
        raw_status = result.get("provider_measurement_status")
        if provider_id == "force_slope_probe":
            if result.get("provider_final_phase") == "safety_latched":
                expected_raw_statuses = {"invalid_safety_force"}
            elif result.get("provider_final_phase") == "invalid_latched":
                expected_raw_statuses = {"invalid_zero_slope", "invalid_phase_or_active_loss"}
            elif count == 13:
                expected_raw_statuses = {"valid"}
            else:
                expected_raw_statuses = {"invalid_no_projected_contact"}
            if raw_status not in expected_raw_statuses:
                raise ForceSlopeAnalysisError("provider measurement status is inconsistent")
        elif raw_status is not None:
            raise ForceSlopeAnalysisError("legacy axis provider invented measurement status")
        if primary is not None:
            _vector(primary, "primary feature48", 48)
        indexed[ordinal] = result

    statistics: dict[str, Any] = {}
    overall_pass = True
    for axis in config[stage + "_axes"]:
        for grasp in config[stage + "_grasps"]:
            positive: list[list[float]] = []
            negative: list[list[float]] = []
            missing = []
            for sign, destination in (("positive", positive), ("negative", negative)):
                for seed in config["environment_seeds"]:
                    slot = next(
                        item for item in plan
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
                    stats = _controlled.separation_statistics(positive, negative)
                except ValueError as error:
                    raise ForceSlopeAnalysisError(
                        "registered separation statistic rejected primary samples"
                    ) from error
                group_pass = stats["separation_passed"] is True
            overall_pass = overall_pass and group_pass
            statistics[f"{axis}-{grasp}"] = {
                "axis": axis,
                "grasp_id": grasp,
                "seed_order": list(config["environment_seeds"]),
                "measurement_pass": not missing,
                "separation_passed": group_pass,
                "missing_primary_samples": missing,
                "separation_statistics": stats,
            }
    status = f"{stage}_{'pass' if overall_pass else 'fail'}"
    return {
        "schema_name": "rj45-force-slope-stage-adjudication-v3",
        "stage": stage,
        "status": status,
        "pass": overall_pass,
        "episode_count": len(plan),
        "config": dict(config),
        "config_sha256": config_sha256,
        "expected_sources_sha256": reviewed_source,
        "controller_sources": controller_sources,
        "separation_by_axis_and_grasp": statistics,
        "results": [dict(indexed[index]) for index in range(1, len(plan) + 1)],
        "seed_evidence_scope": SEED_EVIDENCE_SCOPE,
        "verification_scope": dict(VERIFICATION_SCOPE),
        "claims": dict(FALSE_CLAIMS),
    }


__all__ = [
    "ForceSlopeAnalysisError",
    "VERIFICATION_SCOPE",
    "adjudicate_stage",
    "validate_episode",
]
