from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rby1_assembly import rj45_force_slope_analysis as analysis


ROOT = Path(__file__).resolve().parents[1]


def _config():
    value = json.loads((ROOT / "configs/rj45-force-slope-observability-v3.json").read_text())
    value["frozen"] = True
    return value


def _row(index, observation, action, *, valid=True, phase=4):
    expanded = np.concatenate((np.zeros(6), action))
    tip = np.zeros(6)
    axis_indices = np.flatnonzero(action[:3])
    if len(axis_indices) == 1:
        tip[axis_indices[0]] = action[axis_indices[0]] * analysis.ACTION_OFFSET_SCALE_M
    return {
        "schema_name": "rj45-contact-diagnostic-tick-v2",
        "sequence": index,
        "observer_pre_step_index": index,
        "observer_post_step_index": index + 1,
        "pre_policy_input_valid": valid,
        "pre_phase_code": phase,
        "policy_observation79": observation.tolist(),
        "commanded_normalized_action6": action.tolist(),
        "current_requested_expanded_residual_action12": expanded.tolist(),
        "current_applied_bounded_expanded_action12": expanded.tolist(),
        "current_applied_tip_target_offset_entry6": tip.tolist(),
    }


def _episode(
    slot,
    config,
    *,
    contact=True,
    feature=0.10,
    unsafe=None,
    invalid_prefix=False,
    safety_latch=False,
    zero_slope=False,
):
    provider, provider_path, provider_sha, _ = analysis._provider_for(slot)
    state = provider.reset()
    rows = []
    steps = []
    if invalid_prefix:
        rows.append(_row(0, np.zeros(79), np.zeros(6), valid=False, phase=3))
    impulse_samples = (1.0, 1.0, 1.0, 1.0) if zero_slope else (1.0, 2.0, 3.0, 4.0)
    for _ in range(100):
        index = len(rows)
        observation = np.zeros(79, dtype=np.float64)
        observation[7] = feature
        observation[46] = 1.0
        observation[62] = 0.0013 / 0.012
        observation[78] = 1.0
        before = provider.state_telemetry(state)
        if before["phase"] == "impulse_responses":
            observation[37] = impulse_samples[len(before["impulse_force_samples_n"])] / 8.0
        elif (
            contact
            and before["phase"] in {"sweep", "probing"}
            and before["last_commanded_offset_m"] != 0.0
        ):
            direction = np.sign(before["last_commanded_offset_m"])
            observation[37] = (-direction * 0.25) / 8.0
        if (
            safety_latch
            and before["phase"] == "contact_window"
            and before["contact_window_observations"] == 3
        ):
            observation[37] = 2.5
        action, state = provider.act(observation, state)
        after = provider.state_telemetry(state)
        rows.append(_row(index, observation, action))
        steps.append({
            "step_index": index,
            "state_before": before,
            "state": after,
            "commanded_normalized_action6": action.tolist(),
        })
        if after["phase"] == "done" or (
            after["phase"] in {"invalid_latched", "safety_latched"}
            and (safety_latch or zero_slope)
        ):
            break
    else:
        raise AssertionError("synthetic provider did not finish")

    source_bindings = {
        "force_slope_provider": analysis._SLOPE_PROVIDER_SOURCE_SHA256,
        "active_axis_provider": analysis._AXIS_PROVIDER_SOURCE_SHA256,
        "active_z_base_provider": analysis._BASE_PROVIDER_SOURCE_SHA256,
        "force_slope_contract": analysis._CONTRACT_SOURCE_SHA256,
        "force_slope_analysis": analysis._ANALYSIS_SOURCE_SHA256,
        "controlled_v2_contract": analysis._CONTROLLED_SOURCE_SHA256,
        "test_runner": "1" * 64,
    }
    terminal = {
        "success": False,
        "hard_contact": unsafe == "hard",
        "forbidden_contact": unsafe == "forbidden",
        "completion_time_s": 24.0,
    }
    report = {
        "schema_name": "rj45-force-slope-observability-report-v3",
        "status": "force_slope_observability_complete",
        "diagnostic_only": True,
        "stage": slot["stage"],
        "ordinal": slot["ordinal"],
        "case": copy.deepcopy(slot),
        "config": copy.deepcopy(config),
        "config_sha256": analysis._contract.canonical_sha256(config),
        "actual_environment_seed": slot["environment_seed"],
        "seed_evidence_scope": analysis.SEED_EVIDENCE_SCOPE,
        "trace": {"row_count": len(rows), "trace_sha256": "2" * 64},
        "episode_terminal": terminal,
        "provenance": {
            "source_sha256_start": source_bindings,
            "source_sha256_end": copy.deepcopy(source_bindings),
            "reviewed_source_freeze_sha256": analysis._canonical_sha256(source_bindings),
        },
        "probe": {
            "controller_payload": provider.canonical_payload(),
            "controller_sha256": provider.controller_sha256(),
            "provider_source_sha256": provider_sha,
            "provider_import_origin": str(provider_path),
            "steps": steps,
        },
        "claims": copy.deepcopy(analysis.FALSE_CLAIMS),
    }
    return rows, report


def _verified(stage="development", *, missing=None, unsafe=None):
    config = _config()
    plan = analysis._contract.execution_plan(config, stage)
    results = []
    for slot in plan:
        seed_index = config["environment_seeds"].index(slot["environment_seed"])
        sign = 1.0 if slot["sign"] == "positive" else -1.0
        rows, report = _episode(
            slot,
            config,
            contact=slot["ordinal"] != missing,
            feature=sign * 0.1 + seed_index * 0.001,
            unsafe=unsafe if slot["ordinal"] == 1 else None,
        )
        results.append(analysis.validate_episode(rows, report))
    return config, results


@pytest.mark.parametrize(
    "stage,ordinal", [("development", 1), ("confirmation", 1), ("confirmation", 13)]
)
def test_replays_tx_and_tz_episode_and_binds_all_sources(stage, ordinal):
    config = _config()
    slot = analysis._contract.execution_plan(config, stage)[ordinal - 1]
    rows, report = _episode(slot, config, invalid_prefix=True)
    result = analysis.validate_episode(rows, report)
    assert result["verified"] is True
    assert result["measurement_pass"] is True
    assert result["contact_window_observations"] == 13
    assert len(result["primary_contact_sample13_feature48"]) == 48
    assert result["provider_source_sha256"] == report["probe"]["provider_source_sha256"]
    assert result["reviewed_source_freeze_sha256"] == report["provenance"][
        "reviewed_source_freeze_sha256"
    ]
    assert result["claims"] == analysis.FALSE_CLAIMS
    assert result["verification_scope"] == analysis.VERIFICATION_SCOPE
    assert result["verification_scope"]["runtime_closure_verified"] is False
    json.dumps(result, allow_nan=False)


def test_missing_contact_and_unsafe_terminal_are_verified_failures():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config, contact=False)
    result = analysis.validate_episode(rows, report)
    assert result["verified"] is True
    assert result["measurement_pass"] is False
    assert result["contact_window_observations"] == 0
    assert result["primary_contact_sample13_feature48"] is None

    for unsafe in ("hard", "forbidden"):
        rows, report = _episode(slot, config, unsafe=unsafe)
        result = analysis.validate_episode(rows, report)
        assert result["contact_window_observations"] == 13
        assert result["measurement_pass"] is False
        assert result["episode_terminal"][f"{unsafe}_contact"] is True
        assert result["claims"] == analysis.FALSE_CLAIMS


def test_soft_safety_latch_with_safe_terminal_is_a_verified_partial_failure():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config, safety_latch=True)
    result = analysis.validate_episode(rows, report)
    assert result["provider_final_phase"] == "safety_latched"
    assert result["provider_measurement_status"] == "invalid_safety_force"
    assert result["measurement_status"] == "invalid_safety_force"
    assert result["contact_window_observations"] == 3
    assert result["primary_contact_sample13_feature48"] is None
    assert result["measurement_pass"] is False
    assert result["episode_terminal"]["hard_contact"] is False
    assert result["episode_terminal"]["forbidden_contact"] is False


def test_zero_slope_latch_preserves_exact_scientific_failure_reason():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config, zero_slope=True)
    result = analysis.validate_episode(rows, report)
    assert result["provider_final_phase"] == "invalid_latched"
    assert result["provider_measurement_status"] == "invalid_zero_slope"
    assert result["measurement_status"] == "invalid_zero_slope"
    assert result["contact_window_observations"] == 0
    assert result["measurement_pass"] is False


@pytest.mark.parametrize(
    "forgery",
    [
        "case",
        "ordinal",
        "seed",
        "config",
        "provider",
        "controller",
        "origin",
        "source_end",
        "source_review",
        "source_provider_missing",
        "source_contract_missing",
        "claims",
    ],
)
def test_rejects_report_identity_and_provenance_forgery(forgery):
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config)
    if forgery == "case":
        report["case"]["environment_seed"] += 1
    elif forgery == "ordinal":
        report["ordinal"] = 2
    elif forgery == "seed":
        report["actual_environment_seed"] += 1
    elif forgery == "config":
        report["config_sha256"] = "0" * 64
    elif forgery == "provider":
        report["probe"]["provider_source_sha256"] = "0" * 64
    elif forgery == "controller":
        report["probe"]["controller_sha256"] = "0" * 64
    elif forgery == "origin":
        report["probe"]["provider_import_origin"] = "/wrong/provider.py"
    elif forgery == "source_end":
        report["provenance"]["source_sha256_end"]["force_slope_contract"] = "0" * 64
    elif forgery == "source_review":
        report["provenance"]["reviewed_source_freeze_sha256"] = "0" * 64
    elif forgery in {"source_provider_missing", "source_contract_missing"}:
        key = (
            "force_slope_provider"
            if forgery == "source_provider_missing"
            else "force_slope_contract"
        )
        del report["provenance"]["source_sha256_start"][key]
        del report["provenance"]["source_sha256_end"][key]
        report["provenance"]["reviewed_source_freeze_sha256"] = analysis._canonical_sha256(
            report["provenance"]["source_sha256_start"]
        )
    else:
        report["claims"]["rl_pilot_admitted"] = True
    with pytest.raises(analysis.ForceSlopeAnalysisError):
        analysis.validate_episode(rows, report)


def test_remote_origin_requires_exact_expected_origin():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config)
    remote = "/remote/current/src/rby1_assembly/rj45_force_slope_probe.py"
    report["probe"]["provider_import_origin"] = remote
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="origin"):
        analysis.validate_episode(rows, report)
    assert analysis.validate_episode(
        rows, report, expected_provider_origin=remote
    )["provider_import_origin"] == remote


@pytest.mark.parametrize(
    "index_field,forged_value",
    [
        ("sequence", 9),
        ("observer_pre_step_index", 9),
        ("observer_post_step_index", 9),
    ],
)
def test_rejects_raw_step_action_tip_and_index_forgery(index_field, forged_value):
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config)
    forged = copy.deepcopy(report)
    forged["probe"]["steps"][1]["state"]["calls"] += 1
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="provider state"):
        analysis.validate_episode(rows, forged)
    forged = copy.deepcopy(rows)
    forged[1]["current_applied_tip_target_offset_entry6"][0] += 0.00015
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="tip offset"):
        analysis.validate_episode(forged, report)
    forged = copy.deepcopy(rows)
    forged[1][index_field] = forged_value
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="strictly monotonic"):
        analysis.validate_episode(forged, report)


def test_rejects_integral_float_indices_and_ordinals():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config)
    rows[0]["sequence"] = 0.0
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="strict integer"):
        analysis.validate_episode(rows, report)

    rows, report = _episode(slot, config)
    report["ordinal"] = 1.0
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="strict integer"):
        analysis.validate_episode(rows, report)


def test_rejects_unsafe_success_and_extra_terminal_claims():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config, unsafe="hard")
    report["episode_terminal"]["success"] = True
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="cannot be successful"):
        analysis.validate_episode(rows, report)

    rows, report = _episode(slot, config)
    report["episode_terminal"]["task_admitted"] = False
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="fields are not exact"):
        analysis.validate_episode(rows, report)


def test_trace_count_is_bound_but_unverified_closure_metadata_cannot_escalate_scope():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config)
    report["trace"]["row_count"] += 1
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="row count"):
        analysis.validate_episode(rows, report)

    rows, report = _episode(slot, config)
    report["trace"]["runtime_closure_verified"] = True
    report["verification_scope"] = {
        "runtime_closure_verified": True,
        "external_verification_required": False,
    }
    result = analysis.validate_episode(rows, report)
    assert result["verification_scope"] == analysis.VERIFICATION_SCOPE
    assert result["verification_scope"]["trace_sha256_verified"] is False
    assert result["verification_scope"]["hash_chain_verified"] is False
    assert result["verification_scope"]["archive_manifest_verified"] is False
    assert result["verification_scope"]["runtime_closure_verified"] is False
    assert result["verification_scope"]["external_verification_required"] is True

def test_invalid_input_during_probe_or_nonzero_invalid_action_fails():
    config = _config()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config)
    forged = copy.deepcopy(rows)
    forged[2]["pre_policy_input_valid"] = False
    del report["probe"]["steps"][2]
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="interrupted"):
        analysis.validate_episode(forged, report)
    rows, report = _episode(slot, config, invalid_prefix=True)
    rows[0]["commanded_normalized_action6"][0] = 1e-9
    with pytest.raises(analysis.ForceSlopeAnalysisError, match="exact zero"):
        analysis.validate_episode(rows, report)


@pytest.mark.parametrize("stage,count", [("development", 6), ("confirmation", 24)])
def test_adjudicates_exact_stage_and_preserves_results(stage, count):
    config, results = _verified(stage)
    adjudication = analysis.adjudicate_stage(config, stage, results)
    assert adjudication["status"] == f"{stage}_pass"
    assert adjudication["pass"] is True
    assert adjudication["episode_count"] == count
    assert adjudication["config"] == config
    assert adjudication["results"] == results
    assert adjudication["expected_sources_sha256"] == results[0]["reviewed_source_freeze_sha256"]
    assert set(adjudication["controller_sources"]) == (
        {"force_slope_probe"} if stage == "development"
        else {"force_slope_probe", "active_axis_probe"}
    )
    assert adjudication["claims"] == analysis.FALSE_CLAIMS
    assert adjudication["verification_scope"] == analysis.VERIFICATION_SCOPE
    json.dumps(adjudication, allow_nan=False)


def test_missing_measurement_and_unsafe_episode_fail_development_without_admission():
    for kwargs in ({"missing": 1}, {"unsafe": "hard"}):
        config, results = _verified(**kwargs)
        adjudication = analysis.adjudicate_stage(config, "development", results)
        assert adjudication["status"] == "development_fail"
        assert adjudication["pass"] is False
        assert adjudication["claims"] == analysis.FALSE_CLAIMS
        group = adjudication["separation_by_axis_and_grasp"]["tx-rj45-train-03"]
        assert group["measurement_pass"] is False


def test_partial_safety_latch_fails_the_group_without_rejecting_raw_evidence():
    config, results = _verified()
    slot = analysis._contract.execution_plan(config, "development")[0]
    rows, report = _episode(slot, config, safety_latch=True)
    results[0] = analysis.validate_episode(rows, report)
    adjudication = analysis.adjudicate_stage(config, "development", results)
    assert adjudication["status"] == "development_fail"
    assert adjudication["pass"] is False
    assert adjudication["results"][0]["contact_window_observations"] == 3
    assert adjudication["results"][0]["measurement_status"] == "invalid_safety_force"
    assert adjudication["claims"] == analysis.FALSE_CLAIMS


@pytest.mark.parametrize(
    "forgery",
    [
        "partial",
        "duplicate",
        "slot",
        "source",
        "config",
        "controller",
        "measurement_status",
        "verification_scope",
    ],
)
def test_aggregate_rejects_partial_duplicate_and_provenance_forgery(forgery):
    config, results = _verified()
    changed = copy.deepcopy(results)
    if forgery == "partial":
        changed.pop()
    elif forgery == "duplicate":
        changed[-1]["ordinal"] = changed[0]["ordinal"]
    elif forgery == "slot":
        changed[0]["environment_seed"] = 1
    elif forgery == "source":
        changed[0]["reviewed_source_freeze_sha256"] = "0" * 64
    elif forgery == "config":
        changed[0]["config_sha256"] = "0" * 64
    elif forgery == "measurement_status":
        changed[0]["provider_measurement_status"] = "invalid_zero_slope"
    elif forgery == "verification_scope":
        changed[0]["verification_scope"]["runtime_closure_verified"] = True
    else:
        changed[0]["provider_controller_sha256"] = "0" * 64
    with pytest.raises(analysis.ForceSlopeAnalysisError):
        analysis.adjudicate_stage(config, "development", changed)


def test_frozen_config_and_named_nonfinal_stage_are_mandatory():
    config, results = _verified()
    config["frozen"] = False
    with pytest.raises(analysis.ForceSlopeAnalysisError):
        analysis.adjudicate_stage(config, "development", results)
    config["frozen"] = True
    with pytest.raises(analysis.ForceSlopeAnalysisError):
        analysis.adjudicate_stage(config, "final", results)


def test_analysis_loads_provider_from_its_actual_source_bytes():
    for provider, path, digest, _ in (
        analysis._provider_for({"axis": "tx", "provider_id": "force_slope_probe"}),
        analysis._provider_for({"axis": "tz", "provider_id": "active_axis_probe"}),
    ):
        assert Path(inspect.getfile(provider.__class__)).resolve() == path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
