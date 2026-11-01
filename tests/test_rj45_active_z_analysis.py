import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rby1_assembly import rj45_active_z_analysis as analysis


def _slot(grasp: str, case: str, repeat: int) -> int:
    return (
        analysis.EXPECTED_GRASPS.index(grasp) * 4
        + analysis.EXPECTED_CASES.index(case) * 2
        + analysis.EXPECTED_REPEATS.index(repeat)
        + 1
    )


def _episode(
    *,
    grasp: str = "rj45-train-02",
    case: str = "tz-positive",
    repeat: int = 1,
    feature: float = 0.1,
    contact: bool = True,
):
    provider = analysis.ActiveZProbeProvider()
    state = provider.reset()
    rows = []
    steps = []
    contact_injected = False
    for step_index in range(100):
        observation = np.zeros(79, dtype=np.float64)
        observation[7] = feature
        observation[46] = 1.0
        observation[62] = 0.0013 / 0.012
        observation[78] = 1.0
        before = provider.state_telemetry(state)
        if contact and not contact_injected and before["last_commanded_offset_m"] != 0.0:
            observation[37] = 0.25 / 8.0
            contact_injected = True
        action, state = provider.act(observation, state)
        after = provider.state_telemetry(state)
        expanded = np.concatenate((np.zeros(6), action))
        tip = np.zeros(6)
        tip[2] = action[2] * 0.0015
        rows.append(
            {
                "observer_pre_step_index": step_index,
                "pre_phase_code": 4,
                "pre_policy_input_valid": True,
                "policy_observation79": observation.tolist(),
                "commanded_normalized_action6": action.tolist(),
                "current_requested_expanded_residual_action12": expanded.tolist(),
                "current_applied_bounded_expanded_action12": expanded.tolist(),
                "current_applied_tip_target_offset_entry6": tip.tolist(),
            }
        )
        steps.append(
            {
                "step_index": step_index,
                "state_before": before,
                "state": after,
                "commanded_normalized_action6": action.tolist(),
            }
        )
        if after["phase"] == "done":
            break
    provider_path = Path(inspect.getfile(analysis.ActiveZProbeProvider)).resolve()
    report = {
        "schema_name": "rj45-contact-diagnostic-report-v1",
        "status": "diagnostic_complete",
        "diagnostic_only": True,
        "ordinal": _slot(grasp, case, repeat),
        "case": {
            "grasp_id": grasp,
            "case_id": case,
            "method_id": "active_z_probe",
            "repeat": repeat,
            "provider_inputs": ["observation79", "provider_owned_state"],
        },
        "episode_terminal": {
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
            "completion_time_s": 1.0,
        },
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
        "active_z_probe": {
            "controller_payload": provider.canonical_payload(),
            "controller_sha256": provider.controller_sha256(),
            "provider_source_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            "provider_import_origin": str(provider_path),
            "steps": steps,
        },
    }
    return rows, report


def _verified_set(*, no_contact_slot=None):
    results = []
    for grasp in analysis.EXPECTED_GRASPS:
        for case in analysis.EXPECTED_CASES:
            for repeat in analysis.EXPECTED_REPEATS:
                key = (grasp, case, repeat)
                sign = 1.0 if case == "tz-positive" else -1.0
                feature = sign * (0.10 + 0.005 * (repeat - 1))
                rows, report = _episode(
                    grasp=grasp,
                    case=case,
                    repeat=repeat,
                    feature=feature,
                    contact=key != no_contact_slot,
                )
                results.append(analysis.validate_active_z_episode(rows, report))
    return results


def test_validates_replay_absolute_authority_and_early_contact_window() -> None:
    rows, report = _episode()
    result = analysis.validate_active_z_episode(rows, report)
    assert result["verified"] is True
    assert result["provider_final_phase"] == "done"
    assert result["contact_window_observations"] == 13
    assert len(result["contact_policy_observation48_median"]) == 48
    assert result["observed_absolute_nonzero_z_levels"] == [pytest.approx(0.1)]
    assert result["formal_claims_preserved_false"] is True
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("forgery", ("action", "state", "slot"))
def test_rejects_forged_action_state_and_episode_slot(forgery: str) -> None:
    rows, report = _episode()
    if forgery == "action":
        rows[1]["commanded_normalized_action6"][2] = 0.2
    elif forgery == "state":
        report["active_z_probe"]["steps"][1]["state"]["waveform_index"] = 99
    else:
        report["case"]["repeat"] = 2
    with pytest.raises(analysis.ActiveZAnalysisError):
        analysis.validate_active_z_episode(rows, report)


def test_rejects_nan_and_accumulated_tip_offset() -> None:
    rows, report = _episode()
    nan_rows = copy.deepcopy(rows)
    nan_rows[0]["policy_observation79"][10] = float("nan")
    with pytest.raises(analysis.ActiveZAnalysisError, match="finite"):
        analysis.validate_active_z_episode(nan_rows, report)

    accumulated = copy.deepcopy(rows)
    accumulated[2]["current_applied_tip_target_offset_entry6"][2] += 0.00015
    with pytest.raises(analysis.ActiveZAnalysisError, match="tip offset"):
        analysis.validate_active_z_episode(accumulated, report)


@pytest.mark.parametrize(
    "field",
    (
        "current_requested_expanded_residual_action12",
        "current_applied_bounded_expanded_action12",
    ),
)
def test_rejects_requested_or_bounded_action_divergence(field: str) -> None:
    rows, report = _episode()
    rows[1][field][8] = 0.2
    with pytest.raises(analysis.ActiveZAnalysisError, match="action differs"):
        analysis.validate_active_z_episode(rows, report)


def test_rejects_invalid_gap_and_partial_contact_window() -> None:
    rows, report = _episode()
    invalid_gap = copy.deepcopy(rows)
    invalid_gap[2]["pre_policy_input_valid"] = False
    del report["active_z_probe"]["steps"][2]
    with pytest.raises(analysis.ActiveZAnalysisError, match="validity disappeared"):
        analysis.validate_active_z_episode(invalid_gap, report)

    rows, report = _episode()
    partial_rows = rows[:7]
    partial_report = copy.deepcopy(report)
    partial_report["active_z_probe"]["steps"] = partial_report["active_z_probe"]["steps"][:7]
    with pytest.raises(analysis.ActiveZAnalysisError, match="contact window is partial"):
        analysis.validate_active_z_episode(partial_rows, partial_report)


@pytest.mark.parametrize("field", ("hard_contact", "forbidden_contact"))
def test_rejects_contact_hazards_and_preserves_false_claims(field: str) -> None:
    rows, report = _episode()
    report["episode_terminal"][field] = True
    with pytest.raises(analysis.ActiveZAnalysisError):
        analysis.validate_active_z_episode(rows, report)

    rows, report = _episode()
    report["claims"]["task_admission_completed"] = True
    with pytest.raises(analysis.ActiveZAnalysisError, match="false formal-admission"):
        analysis.validate_active_z_episode(rows, report)


def test_rejects_controller_and_source_provenance_forgery() -> None:
    rows, report = _episode()
    report["active_z_probe"]["controller_sha256"] = "0" * 64
    with pytest.raises(analysis.ActiveZAnalysisError, match="controller SHA"):
        analysis.validate_active_z_episode(rows, report)

    rows, report = _episode()
    report["active_z_probe"]["provider_source_sha256"] = "0" * 64
    with pytest.raises(analysis.ActiveZAnalysisError, match="source SHA"):
        analysis.validate_active_z_episode(rows, report)


def test_remote_provider_origin_requires_explicit_exact_expectation() -> None:
    rows, report = _episode()
    remote_origin = "/remote/job/current/src/rby1_assembly/rj45_active_z_probe.py"
    report["active_z_probe"]["provider_import_origin"] = remote_origin
    with pytest.raises(analysis.ActiveZAnalysisError, match="expected execution source"):
        analysis.validate_active_z_episode(rows, report)

    result = analysis.validate_active_z_episode(
        rows, report, expected_provider_origin=remote_origin
    )
    assert result["provider_import_origin"] == remote_origin
    with pytest.raises(analysis.ActiveZAnalysisError, match="expected execution source"):
        analysis.validate_active_z_episode(
            rows,
            report,
            expected_provider_origin="/wrong/job/current/src/rby1_assembly/rj45_active_z_probe.py",
        )


def test_adjudicates_exact_eight_episodes_for_both_grasps() -> None:
    result = analysis.adjudicate_active_z(_verified_set())
    assert result["pass"] is True
    assert result["episode_count"] == 8
    assert all(item["signal_d"] >= 0.05 for item in result["grasps"].values())
    assert all(item["signal_to_noise_ratio"] >= 2.0 for item in result["grasps"].values())
    assert result["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }
    json.dumps(result, allow_nan=False)


def test_no_contact_is_valid_collection_but_fails_axis_adjudication() -> None:
    missing = ("rj45-train-02", "tz-positive", 1)
    results = _verified_set(no_contact_slot=missing)
    episode = next(
        item
        for item in results
        if (item["grasp_id"], item["case_id"], item["repeat"]) == missing
    )
    assert episode["verified"] is True
    assert episode["contact_observed"] is False
    adjudication = analysis.adjudicate_active_z(results)
    assert adjudication["pass"] is False
    assert adjudication["grasps"]["rj45-train-02"]["missing_contact_windows"]


def test_low_separation_does_not_create_a_false_positive() -> None:
    results = _verified_set()
    for result in results:
        result["contact_policy_observation48_median"] = [0.0] * 48
    adjudication = analysis.adjudicate_active_z(results)
    assert adjudication["pass"] is False
    assert all(item["signal_d"] == 0.0 for item in adjudication["grasps"].values())


def test_aggregate_rejects_duplicate_or_partial_plan() -> None:
    results = _verified_set()
    with pytest.raises(analysis.ActiveZAnalysisError, match="exactly eight"):
        analysis.adjudicate_active_z(results[:-1])
    forged = copy.deepcopy(results)
    forged[-1]["ordinal"] = forged[0]["ordinal"]
    with pytest.raises(analysis.ActiveZAnalysisError, match="duplicated or forged"):
        analysis.adjudicate_active_z(forged)


@pytest.mark.parametrize('field,length', [
    ('commanded_normalized_action6', 6),
    ('current_requested_expanded_residual_action12', 12),
    ('current_applied_bounded_expanded_action12', 12),
    ('current_applied_tip_target_offset_entry6', 6),
])
def test_invalid_align_to_insert_transition_requires_all_exact_zero(field, length):
    rows, report = _episode()
    transition = copy.deepcopy(rows[0])
    transition.update(pre_policy_input_valid=False, pre_phase_code=3,
                      post_phase_name='insert', observer_pre_step_index=0)
    shifted = copy.deepcopy(rows)
    for row in shifted:
        row['observer_pre_step_index'] += 1
    for step in report['active_z_probe']['steps']:
        step['step_index'] += 1
    assert analysis.validate_active_z_episode([transition, *shifted], report)['verified']
    transition[field][length - 1] = 1e-12
    with pytest.raises(analysis.ActiveZAnalysisError, match='exact-zero'):
        analysis.validate_active_z_episode([transition, *shifted], report)
