import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rby1_assembly import rj45_controlled_observability_analysis as analysis


def _config():
    path = Path(__file__).resolve().parents[1] / "configs/rj45-controlled-observability-v2.json"
    config = json.loads(path.read_text())
    config["frozen"] = True
    return config


def _row(index, observation, action, *, valid=True, phase=4):
    expanded = np.concatenate((np.zeros(6), action))
    axis_indices = np.flatnonzero(action[:3])
    tip = np.zeros(6)
    if len(axis_indices) == 1:
        tip[axis_indices[0]] = action[axis_indices[0]] * 0.0015
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


def _episode(slot, config, *, contact=True, invalid_prefix=False, postzero=True, feature=0.1):
    provider = analysis.ActiveAxisProbeProvider(slot["axis"])
    state = provider.reset()
    rows = []
    steps = []
    if invalid_prefix:
        rows.append(_row(0, np.zeros(79), np.zeros(6), valid=False, phase=3))
    contact_injected = False
    zero_return_seen = False
    for _ in range(100):
        row_index = len(rows)
        observation = np.zeros(79, dtype=np.float64)
        observation[0:7] = 999.0  # primary projection must exclude pose fields
        observation[7] = feature
        observation[46] = 1.0
        observation[62] = 0.0013 / 0.012
        observation[78] = 1.0
        before = provider.state_telemetry(state)
        if contact and not contact_injected and before["last_commanded_offset_m"] != 0.0:
            observation[37] = 0.25 / 8.0
            contact_injected = True
        if zero_return_seen:
            observation[8] = 0.77
        action, state = provider.act(observation, state)
        after = provider.state_telemetry(state)
        rows.append(_row(row_index, observation, action))
        steps.append(
            {
                "step_index": row_index,
                "state_before": before,
                "state": after,
                "commanded_normalized_action6": action.tolist(),
            }
        )
        if after["phase"] == "done":
            if postzero and not zero_return_seen and contact:
                zero_return_seen = True
                continue
            break

    provider_path = Path(inspect.getfile(analysis.ActiveAxisProbeProvider)).resolve()
    report = {
        "schema_name": "rj45-controlled-observability-report-v2",
        "status": "controlled_observability_complete",
        "diagnostic_only": True,
        "case": copy.deepcopy(slot),
        "actual_environment_seed": slot["environment_seed"],
        "seed_evidence_scope": (
            "environment_cfg_seed_checked; stochastic_independence_not_demonstrated"
        ),
        "config": copy.deepcopy(config),
        "config_sha256": analysis._contract_module.canonical_sha256(config),
        "active_axis_probe": {
            "controller_payload": provider.canonical_payload(),
            "controller_sha256": provider.controller_sha256(),
            "provider_source_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            "provider_import_origin": str(provider_path),
            "steps": steps,
        },
        "episode_terminal": {
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
        },
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
    }
    return rows, report


def _verified_results(*, missing_ordinal=None):
    config = _config()
    plan = analysis._contract_module.execution_plan(config)
    seeds = list(analysis._contract_module.environment_seeds())
    results = []
    for slot in plan:
        seed_index = seeds.index(slot["environment_seed"])
        sign = 1.0 if slot["sign"] == "positive" else -1.0
        feature = sign * 0.10 + seed_index * 0.001
        rows, report = _episode(
            slot,
            config,
            contact=slot["ordinal"] != missing_ordinal,
            feature=feature,
        )
        results.append(analysis.validate_controlled_episode(rows, report))
    return config, results


def test_validates_sample13_primary_secondaries_and_excludes_pose() -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config, invalid_prefix=True)
    result = analysis.validate_controlled_episode(rows, report)
    assert result["verified"] is True
    assert result["measurement_pass"] is True
    assert result["contact_window_observations"] == 13
    assert len(result["primary_contact_sample13_feature48"]) == 48
    assert 999.0 not in result["primary_contact_sample13_feature48"]
    assert result["secondary_contact_sample13_observation79"][0] == 999.0
    assert len(result["secondary_contact_window_median_feature48"]) == 48
    assert result["secondary_first_observation_after_zero_return79"][8] == 0.77
    assert "archive_verified" not in result
    json.dumps(result, allow_nan=False)


def test_rejects_forged_step_and_tip() -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config)
    forged = copy.deepcopy(report)
    forged["active_axis_probe"]["steps"][1]["state"]["waveform_index"] = 99
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="provider state"):
        analysis.validate_controlled_episode(rows, forged)

    forged_rows = copy.deepcopy(rows)
    forged_rows[1]["current_applied_tip_target_offset_entry6"][0] += 0.00015
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="tip offset"):
        analysis.validate_controlled_episode(forged_rows, report)


def test_invalid_active_row_and_nonzero_invalid_command_fail() -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config)
    invalid = copy.deepcopy(rows)
    invalid[2]["pre_policy_input_valid"] = False
    del report["active_axis_probe"]["steps"][2]
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="interrupted"):
        analysis.validate_controlled_episode(invalid, report)

    rows, report = _episode(slot, config, invalid_prefix=True)
    rows[0]["commanded_normalized_action6"][0] = 0.1
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="exact zero"):
        analysis.validate_controlled_episode(rows, report)


def test_missing_contact_is_measurement_failure_but_partial_window_is_invalid() -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config, contact=False)
    result = analysis.validate_controlled_episode(rows, report)
    assert result["verified"] is True
    assert result["measurement_pass"] is False
    assert result["primary_contact_sample13_feature48"] is None
    assert result["secondary_contact_sample13_observation79"] is None

    rows, report = _episode(slot, config)
    partial_rows = rows[:7]
    report["active_axis_probe"]["steps"] = report["active_axis_probe"]["steps"][:7]
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="contact window is partial"):
        analysis.validate_controlled_episode(partial_rows, report)


@pytest.mark.parametrize("forgery", ("case", "config_hash", "provider_hash", "controller_hash"))
def test_rejects_case_and_provenance_hash_forgery(forgery: str) -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config)
    if forgery == "case":
        report["case"]["environment_seed"] += 1
    elif forgery == "config_hash":
        report["config_sha256"] = "0" * 64
    elif forgery == "provider_hash":
        report["active_axis_probe"]["provider_source_sha256"] = "0" * 64
    else:
        report["active_axis_probe"]["controller_sha256"] = "0" * 64
    with pytest.raises(analysis.ControlledObservabilityAnalysisError):
        analysis.validate_controlled_episode(rows, report)


def test_rejects_nonmonotonic_trace_indices_and_wrong_seed_slot() -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config)
    rows[1]["sequence"] = 7
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="strictly monotonic"):
        analysis.validate_controlled_episode(rows, report)

    rows, report = _episode(slot, config)
    report["case"]["environment_seed"] = config["environment_seeds"][1]
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="execution-plan slot"):
        analysis.validate_controlled_episode(rows, report)


@pytest.mark.parametrize("forgery", ("actual_seed", "seed_scope", "diagnostic_only"))
def test_rejects_unproven_seed_or_nondiagnostic_scope(forgery: str) -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config)
    if forgery == "actual_seed":
        report["actual_environment_seed"] += 1
    elif forgery == "seed_scope":
        report["seed_evidence_scope"] = "stochastic_independence_demonstrated"
    else:
        report["diagnostic_only"] = False
    with pytest.raises(analysis.ControlledObservabilityAnalysisError):
        analysis.validate_controlled_episode(rows, report)


def test_remote_provider_origin_requires_explicit_exact_expectation() -> None:
    config = _config()
    slot = analysis._contract_module.execution_plan(config)[0]
    rows, report = _episode(slot, config)
    remote = "/remote/job/current/src/rby1_assembly/rj45_active_axis_probe.py"
    report["active_axis_probe"]["provider_import_origin"] = remote
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="origin"):
        analysis.validate_controlled_episode(rows, report)
    result = analysis.validate_controlled_episode(
        rows, report, expected_provider_origin=remote
    )
    assert result["provider_import_origin"] == remote
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="origin"):
        analysis.validate_controlled_episode(
            rows,
            report,
            expected_provider_origin="/wrong/current/rj45_active_axis_probe.py",
        )


def test_adjudicates_exact_24_in_registered_seed_order() -> None:
    config, results = _verified_results()
    adjudication = analysis.adjudicate_controlled_observability(results, config)
    assert adjudication["pass"] is True
    assert adjudication["episode_count"] == 24
    assert len(adjudication["separation_by_axis_and_grasp"]) == 4
    for item in adjudication["separation_by_axis_and_grasp"].values():
        assert item["seed_order"] == config["environment_seeds"]
        assert item["measurement_pass"] is True
        assert item["separation_passed"] is True
        assert item["separation_statistics"]["stochastic_robustness_demonstrated"] is False
    assert adjudication["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }
    assert "archive_verified" not in adjudication
    json.dumps(adjudication, allow_nan=False)


def test_duplicate_slot_and_missing_primary_cannot_pass_aggregate() -> None:
    config, results = _verified_results()
    duplicate = copy.deepcopy(results)
    duplicate[-1]["ordinal"] = duplicate[0]["ordinal"]
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="duplicated"):
        analysis.adjudicate_controlled_observability(duplicate, config)

    config, results = _verified_results(missing_ordinal=1)
    adjudication = analysis.adjudicate_controlled_observability(results, config)
    assert adjudication["pass"] is False
    affected = adjudication["separation_by_axis_and_grasp"]["tx-rj45-train-02"]
    assert affected["measurement_pass"] is False
    assert affected["separation_statistics"] is None


def test_aggregate_rejects_changed_config_digest_source_and_seed() -> None:
    config, results = _verified_results()
    changed = copy.deepcopy(results)
    changed[0]["config_sha256"] = "0" * 64
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="unverified"):
        analysis.adjudicate_controlled_observability(changed, config)

    changed = copy.deepcopy(results)
    changed[1]["axis_provider_source_sha256"] = "0" * 64
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="unverified"):
        analysis.adjudicate_controlled_observability(changed, config)

    changed = copy.deepcopy(results)
    changed[0]["environment_seed"] = 1
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="execution-plan slot"):
        analysis.adjudicate_controlled_observability(changed, config)


@pytest.mark.parametrize("field", ["actual_environment_seed", "case_seed", "case_ordinal", "trace_indices"])
def test_integral_floats_are_not_integer_provenance(field):
    config = _config()
    rows, report = _episode(analysis._contract_module.execution_plan(config)[0], config)
    if field == "actual_environment_seed":
        report[field] = float(report[field])
    elif field.startswith("case_"):
        key = "environment_seed" if field == "case_seed" else "ordinal"
        report["case"][key] = float(report["case"][key])
    else:
        for key in ("sequence", "observer_pre_step_index", "observer_post_step_index"):
            rows[0][key] = float(rows[0][key])
    with pytest.raises(analysis.ControlledObservabilityAnalysisError):
        analysis.validate_controlled_episode(rows, report)


@pytest.mark.parametrize("field", ["environment_seed", "actual_environment_seed", "ordinal"])
def test_aggregate_rejects_integral_float_identity(field):
    config, results = _verified_results()
    results[0][field] = float(results[0][field])
    with pytest.raises(analysis.ControlledObservabilityAnalysisError):
        analysis.adjudicate_controlled_observability(results, config)


def test_invalid_input_action_must_be_exact_zero():
    config = _config()
    rows, report = _episode(analysis._contract_module.execution_plan(config)[0], config, invalid_prefix=True)
    rows[0]["commanded_normalized_action6"][0] = 5e-7
    with pytest.raises(analysis.ControlledObservabilityAnalysisError, match="exact zero"):
        analysis.validate_controlled_episode(rows, report)


def test_postzero_secondary_skips_invalid_input_and_is_captured_once():
    config = _config()
    rows, report = _episode(analysis._contract_module.execution_plan(config)[0], config)
    index = len(rows) - 1
    observation = np.zeros(79)
    observation[8] = 123.0
    rows.insert(index, _row(index, observation, np.zeros(6), valid=False, phase=4))
    for i, row in enumerate(rows):
        row.update(sequence=i, observer_pre_step_index=i, observer_post_step_index=i+1)
    report["active_axis_probe"]["steps"][-1]["step_index"] += 1
    result = analysis.validate_controlled_episode(rows, report)
    assert result["secondary_first_observation_after_zero_return79"][8] == 0.77

    provider = analysis.ActiveAxisProbeProvider("tx")
    state = provider.reset()
    for row in rows:
        if row["pre_policy_input_valid"]:
            _, state = provider.act(row["policy_observation79"], state)
    observation = np.asarray(rows[-1]["policy_observation79"])
    observation[8] = 456.0
    before = provider.state_telemetry(state)
    action, state = provider.act(observation, state)
    index = len(rows)
    rows.append(_row(index, observation, action))
    report["active_axis_probe"]["steps"].append({
        "step_index": index, "state_before": before, "state": provider.state_telemetry(state),
        "commanded_normalized_action6": action.tolist(),
    })
    result = analysis.validate_controlled_episode(rows, report)
    assert result["secondary_first_observation_after_zero_return79"][8] == 0.77
