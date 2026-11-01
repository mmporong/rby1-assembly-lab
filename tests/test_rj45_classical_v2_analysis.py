from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rby1_assembly import rj45_classical_v2_analysis as analysis


ROOT = Path(__file__).resolve().parents[1]


def _config():
    value = json.loads(
        (ROOT / "configs/rj45-classical-tuning-v2.json").read_text()
    )
    value["frozen"] = True
    return value


def _verified_tuning_rows(config):
    binding = analysis._contract.validate_classical_v2_config(
        config, require_frozen=True
    )
    result = []
    for slot in analysis._contract.tuning_execution_plan(
        config, require_frozen=True
    ):
        result.append({
            "slot": slot,
            "verification": {
                "verified": True,
                "registered_config_sha256": binding["registered_config_sha256"],
                "classical_source_sha256": binding["classical_source_sha256"],
                "controller_sha256": slot["controller_sha256"],
            },
            "metrics": {
                "success": slot["candidate_id"] == "V2A",
                "hard_contact": False,
                "forbidden_contact": False,
                "evaluator_force_p95_n": float(ord(slot["candidate_id"][-1]) - 64),
                "completion_time_s": 1.0,
            },
        })
    return result


def _row(index, observation, action, *, valid, terminal, force):
    expanded = np.concatenate((np.zeros(6), action))
    tip = np.concatenate((action[:3] * 0.0015, action[3:] * (np.pi / 120.0)))
    if not valid:
        expanded = np.zeros(12)
        tip = np.zeros(6)
        action = np.zeros(6)
    return {
        "schema_name": "rj45-contact-diagnostic-tick-v2",
        "sequence": index,
        "observer_pre_step_index": index,
        "observer_post_step_index": index + 1,
        "pre_policy_input_valid": valid,
        "pre_phase_code": 4 if valid else 3,
        "policy_observation79": observation.tolist(),
        "commanded_normalized_action6": action.tolist(),
        "current_requested_expanded_residual_action12": expanded.tolist(),
        "current_applied_bounded_expanded_action12": expanded.tolist(),
        "current_applied_tip_target_offset_entry6": tip.tolist(),
        "grip_contact_proxy_n": [force, force / 2.0],
        "insertion_normal_load_sum_n": force / 3.0,
        "post_source_timestamp_s": float((index + 1) * 0.1),
        "terminal": terminal,
        "post_state_source": (
            "pre_autoreset_terminal_snapshot" if terminal
            else "post_step_live_snapshot"
        ),
    }


def _episode(*, stage="tuning", hard=False, forbidden=False):
    config = _config()
    verified = _verified_tuning_rows(config) if stage == "admission" else None
    plan = (
        analysis._contract.tuning_execution_plan(config, require_frozen=True)
        if stage == "tuning"
        else analysis._contract.select_and_build_admission_plan(config, verified)[
            "episodes"
        ]
    )
    slot = copy.deepcopy(plan[0])
    provider = analysis.ClassicalForceSearchV2Provider(slot["candidate_id"])
    state = provider.reset()
    rows = [_row(0, np.zeros(79), np.zeros(6), valid=False, terminal=False, force=0.0)]
    steps = []
    for index, force in enumerate((1.0, 2.0, 3.0), start=1):
        observation = np.zeros(79, dtype=np.float64)
        observation[0] = 0.2
        observation[37] = force / 8.0
        before = provider.state_telemetry(state)
        action, state = provider.act(observation, state)
        rows.append(_row(
            index, observation, action, valid=True, terminal=index == 3, force=force
        ))
        steps.append({
            "step_index": index,
            "state_before": before,
            "state": provider.state_telemetry(state),
            "commanded_normalized_action6": action.tolist(),
        })
    forces = np.asarray((0.0, 1.0, 2.0, 3.0))
    p95 = float(np.percentile(forces, 95, method="linear"))
    completion = rows[-1]["post_source_timestamp_s"]
    requested_translation = [slot["requested_translation_bias_entry_m"]]
    requested_rotation = [slot["requested_rotation_bias_entry_rad"]]
    terminal = {
        "terminal_mask": [True],
        "independent_eval_valid": [True],
        "independent_eval_success": [False],
        "hard_overload": [hard],
        "forbidden_contact": [forbidden],
        "timestamp_s": [completion],
        "grip_contact_force_peak_n": [26.0 if hard else 3.0],
        "insertion_contact_load_peak_n": [1.0],
        "task_admission": {
            "applied_held_grasp_id": slot["grasp_id"],
            "requested_translation_bias_entry_m": requested_translation,
            "requested_rotation_bias_entry_rad": requested_rotation,
            "applied_translation_bias_entry_m": copy.deepcopy(requested_translation),
            "applied_rotation_bias_entry_rad": copy.deepcopy(requested_rotation),
            "prefreeze_snapshot_valid": [True],
            "insert_transition_snapshot_valid": [True],
            "transition_contract_checked": [True],
            "actual_applied_action_is_exact_zero": [True],
            "requested_action_6d": [[0.0] * 6],
            "bounded_action_6d": [[0.0] * 6],
            "actual_applied_action_6d": [[0.0] * 6],
            "actual_applied_tip_target_offset_entry_6d": [[0.0] * 6],
            "evaluator_gt_available_to_fair_provider": False,
        },
    }
    provider_path = Path(inspect.getfile(analysis.ClassicalForceSearchV2Provider)).resolve()
    binding = analysis._contract.validate_classical_v2_config(
        config, require_frozen=True
    )
    report = {
        "schema_name": "rj45-classical-v2-report-v1",
        "status": "classical_v2_complete",
        "diagnostic_only": True,
        "comparison_only": True,
        "stage": stage,
        "ordinal": 1,
        "case": slot,
        "config": config,
        "config_sha256": binding["registered_config_sha256"],
        "actual_environment_seed": 20261121,
        "validated_terminal_snapshot": terminal,
        "episode_terminal": {
            "success": False,
            "hard_contact": hard,
            "forbidden_contact": forbidden,
            "evaluator_force_p95_n": p95,
            "completion_time_s": completion,
        },
        "classical_v2": {
            "controller_payload": provider.canonical_payload(),
            "controller_sha256": provider.controller_sha256(),
            "provider_source_sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest(),
            "provider_import_origin": str(provider_path),
            "base_helper_source_sha256": provider.canonical_payload()[
                "base_helper_source_sha256"
            ],
            "steps": steps,
        },
        "provenance": {
            "source_sha256_start": {
                "classical_v2_contract": analysis._CONTRACT_SOURCE_SHA256,
                "classical_v2_provider": analysis._PROVIDER_SOURCE_SHA256,
                "classical_v2_base_provider": analysis._BASE_SOURCE_SHA256,
            },
            "source_sha256_end": {
                "classical_v2_contract": analysis._CONTRACT_SOURCE_SHA256,
                "classical_v2_provider": analysis._PROVIDER_SOURCE_SHA256,
                "classical_v2_base_provider": analysis._BASE_SOURCE_SHA256,
            },
        },
        "claims": {
            "task_admission_completed": False,
            "rl_pilot_admitted": False,
        },
    }
    return rows, report, verified


def _validate(rows, report, verified=None):
    return analysis.validate_classical_v2_episode(
        rows, report, verified_tuning_rows=verified
    )


def test_analysis_import_is_cpu_only_and_does_not_import_isaac():
    code = (
        "import sys; import rby1_assembly.rj45_classical_v2_analysis; "
        "assert not any(n == 'isaaclab' or n.startswith('isaaclab.') for n in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert result.returncode == 0, result.stderr


def test_valid_tuning_replay_returns_exact_wrapper_and_scope_caveats():
    rows, report, _ = _episode()
    result = _validate(rows, report)
    assert result["wrapper"]["slot"] == report["case"]
    assert set(result["wrapper"]["verification"]) == {
        "verified", "registered_config_sha256", "classical_source_sha256",
        "controller_sha256",
    }
    assert set(result["wrapper"]["metrics"]) == {
        "success", "hard_contact", "forbidden_contact",
        "evaluator_force_p95_n", "completion_time_s",
    }
    assert result["raw_validation"]["provider_steps_replayed"] == 3
    assert result["raw_validation"]["archive_or_hash_chain_verified"] is False
    assert result["raw_validation"]["runtime_physical_closure_verified"] is False
    assert result["raw_validation"]["verified_tuning_rows_external_sha_pin_verified"] is False
    assert result["raw_validation"]["forbidden_contact_newly_proven_from_raw_trace"] is False
    assert result["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }
    json.dumps(result, allow_nan=False)


def test_admission_recomputes_selected_exact_12_from_verified_tuning_rows():
    rows, report, verified = _episode(stage="admission")
    result = _validate(rows, report, verified)
    assert result["wrapper"]["slot"]["stage"] == (
        "classical_v2_new_admission_baseline"
    )
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="requires"):
        _validate(rows, report, None)


@pytest.mark.parametrize("field", ["state_before", "state", "commanded_normalized_action6"])
def test_rejects_forged_provider_step(field):
    rows, report, _ = _episode()
    if field == "commanded_normalized_action6":
        report["classical_v2"]["steps"][0][field][0] += 0.1
    else:
        report["classical_v2"]["steps"][0][field]["stalled_ticks"] = 99
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="provider"):
        _validate(rows, report)


def test_rejects_tip_and_invalid_tiny_action():
    rows, report, _ = _episode()
    rows[1]["current_applied_tip_target_offset_entry6"][0] += 1e-5
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="tip target"):
        _validate(rows, report)

    rows, report, _ = _episode()
    rows[0]["commanded_normalized_action6"][0] = 1e-12
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="exact zero"):
        _validate(rows, report)


@pytest.mark.parametrize("forgery", ["success", "hard", "forbidden", "time"])
def test_rejects_forged_terminal_outcome_or_time(forgery):
    rows, report, _ = _episode()
    if forgery == "success":
        report["validated_terminal_snapshot"]["independent_eval_success"] = [True]
    elif forgery == "hard":
        report["validated_terminal_snapshot"]["hard_overload"] = [True]
    elif forgery == "forbidden":
        report["validated_terminal_snapshot"]["forbidden_contact"] = [True]
    else:
        report["episode_terminal"]["completion_time_s"] += 0.1
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="terminal|timestamp"):
        _validate(rows, report)


def test_rejects_bias_seed_wrong_slot_and_nonfinite_force():
    rows, report, _ = _episode()
    report["validated_terminal_snapshot"]["task_admission"][
        "applied_translation_bias_entry_m"
    ][0][0] += 1e-6
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="bias"):
        _validate(rows, report)

    rows, report, _ = _episode()
    report["actual_environment_seed"] = True
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="integer"):
        _validate(rows, report)

    rows, report, _ = _episode()
    report["case"]["case_id"] = "tx-negative"
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="slot"):
        _validate(rows, report)

    rows, report, _ = _episode()
    rows[1]["grip_contact_proxy_n"][0] = float("nan")
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="finite"):
        _validate(rows, report)


def test_rejects_forged_force_p95_and_noncontiguous_indices():
    rows, report, _ = _episode()
    report["episode_terminal"]["evaluator_force_p95_n"] += 0.1
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="force p95"):
        _validate(rows, report)

    rows, report, _ = _episode()
    rows[1]["sequence"] = 2
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="contiguous"):
        _validate(rows, report)


def test_terminal_peaks_cover_trace_channels_and_drive_strict_hard_predicate():
    rows, report, _ = _episode()
    terminal = report["validated_terminal_snapshot"]
    terminal["grip_contact_force_peak_n"] = [0.1]
    terminal["insertion_contact_load_peak_n"] = [0.1]
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="below.*trace"):
        _validate(rows, report)

    rows, report, _ = _episode()
    report["validated_terminal_snapshot"]["grip_contact_force_peak_n"] = [26.0]
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="strict peak predicate"):
        _validate(rows, report)

    rows, report, _ = _episode()
    report["validated_terminal_snapshot"]["grip_contact_force_peak_n"] = [25.0]
    assert _validate(rows, report)["wrapper"]["metrics"]["hard_contact"] is False


def test_unsafe_outcome_cannot_be_independently_successful():
    rows, report, _ = _episode(hard=True)
    report["episode_terminal"]["success"] = True
    report["validated_terminal_snapshot"]["independent_eval_success"] = [True]
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="cannot.*successful"):
        _validate(rows, report)


def test_rejects_source_drift_nonzero_transition_request_and_terminal_source():
    rows, report, _ = _episode()
    report["provenance"]["source_sha256_end"]["classical_v2_provider"] = "0" * 64
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="source"):
        _validate(rows, report)

    rows, report, _ = _episode()
    report["validated_terminal_snapshot"]["task_admission"][
        "requested_action_6d"
    ][0][0] = 1e-12
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="first INSERT"):
        _validate(rows, report)

    rows, report, _ = _episode()
    rows[-1]["post_state_source"] = "post_step_live_snapshot"
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="terminal trace source"):
        _validate(rows, report)


def test_hard_and_forbidden_are_preserved_safety_outcomes_not_raw_invalidity():
    rows, report, _ = _episode(hard=True, forbidden=True)
    result = _validate(rows, report)
    assert result["wrapper"]["metrics"]["hard_contact"] is True
    assert result["wrapper"]["metrics"]["forbidden_contact"] is True
    assert result["raw_validation"]["hard_contact_force_threshold_n"] == 25.0
    assert result["raw_validation"][
        "hard_contact_threshold_runtime_config_closure_verified"
    ] is False


def test_remote_provider_origin_requires_explicit_expected_origin():
    rows, report, _ = _episode()
    remote = "/remote/job/current/src/rby1_assembly/rj45_classical_v2.py"
    report["classical_v2"]["provider_import_origin"] = remote
    with pytest.raises(analysis.ClassicalV2AnalysisError, match="origin"):
        _validate(rows, report)
    result = analysis.validate_classical_v2_episode(
        rows, report, expected_provider_origin=remote
    )
    assert result["raw_validation"]["provider_import_origin"] == remote
