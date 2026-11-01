from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from rby1_assembly.rj45_g1_synthesis_v3 import synthesize_g1_v3


ROOT = Path(__file__).resolve().parents[1]
CASES = (
    "tx-positive", "tx-negative", "tz-positive", "tz-negative",
    "rx-positive", "rx-negative",
)
GRASPS = ("rj45-train-02", "rj45-train-05")
METHOD_SLOTS = {
    "zero": (1, 2), "mean_constant": (1,), "open_loop_playback": (1,),
    "classical_force_search": (1,), "oracle_gt_reference": (1,),
}
SHARED = (
    "scripts/smoke_rj45_comparison_isaac.py",
    "src/rby1_assembly/rj45_comparison_env.py",
    "src/rby1_assembly/rj45_comparison_providers.py",
    "src/rby1_assembly/rj45_task_admission.py",
    "src/rby1_assembly/rj45_task_admission_env.py",
)


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode()).hexdigest()


def _raw_row(ordinal, stage, grasp, case, method, slot, candidate=None):
    oracle_success = method == "oracle_gt_reference" and case != "rx-negative"
    return {
        "schema_name": "rj45-g1-raw90-verified-row-v3",
        "ordinal": ordinal,
        "stage": stage,
        "grasp_id": grasp,
        "case_id": case,
        "method_id": method,
        "slot": slot,
        "candidate_id": candidate,
        "success": oracle_success or (method == "zero" and slot == 1 and case == "tx-positive"),
        "hard_contact": False,
        "forbidden_contact": False,
        "execution_error": None,
        "terminal_complete": True,
        "requested_applied_bias_valid": True,
        "transition_contract_valid": True,
    }


def _raw90():
    rows = []
    for candidate in ("A", "B", "C"):
        for case in CASES:
            rows.append(_raw_row(
                len(rows) + 1, "tuning", "rj45-train-03", case,
                "classical_force_search", 1, candidate,
            ))
    for grasp in GRASPS:
        for case in CASES:
            for method, slots in METHOD_SLOTS.items():
                for slot in slots:
                    rows.append(_raw_row(
                        len(rows) + 1, "admission", grasp, case, method, slot,
                    ))
    return {
        "envelope": {
            "schema_name": "rj45-g1-raw90-envelope-v3",
            "report_sha256": "1" * 64,
            "archive_manifest_sha256": "2" * 64,
            "archive_payload_sha256": "3" * 64,
            "reviewed_source_freeze_sha256": "4" * 64,
            "physics_compatibility_sha256": "5" * 64,
            "raw_replay_verified": True,
            "row_integrity_verified": True,
            "archive_verified": True,
            "runtime_closure_verified": True,
            "identity_verified": True,
            "schema_verified": True,
            "terminal_verified": True,
            "execution_error_free": True,
            "bias_contract_verified": True,
            "transition_contract_verified": True,
            "verification_scope": {
                "source_closed_full_se3_evaluator_terminal_scalar": True,
                "per_frame_geometry_replay": False,
                "external_verification_required": True,
            },
        },
        "rows": rows,
    }


def _observability():
    config = json.loads(
        (ROOT / "configs/rj45-force-slope-observability-v3.json").read_text()
    )
    force = {
        "controller_sha256": "6" * 64,
        "provider_source_sha256": "7" * 64,
        "provider_import_origin": "/reviewed/rj45_force_slope_probe.py",
    }
    axis = {
        "controller_sha256": "8" * 64,
        "provider_source_sha256": "9" * 64,
        "provider_import_origin": "/reviewed/rj45_active_axis_probe.py",
    }

    def aggregate(stage, count, controllers):
        value = {
            "schema_name": "rj45-force-slope-stage-adjudication-v3",
            "stage": stage,
            "status": f"{stage}_pass",
            "pass": True,
            "episode_count": count,
            "config": config,
            "config_sha256": _canonical_sha(config),
            "expected_sources_sha256": "a" * 64,
            "controller_sources": controllers,
            "results": [{"ordinal": index} for index in range(1, count + 1)],
            "claims": {
                "task_admission_completed": False,
                "rl_pilot_admitted": False,
            },
        }
        return {"aggregate": value, "recomputed": deepcopy(value)}

    return {
        "config": config,
        "development": aggregate(
            "development", 6, {"force_slope_probe": deepcopy(force)}
        ),
        "confirmation": aggregate(
            "confirmation", 24,
            {
                "force_slope_probe": deepcopy(force),
                "active_axis_probe": deepcopy(axis),
            },
        ),
        "current_controller_sources": {
            "force_slope_probe": deepcopy(force),
            "active_axis_probe": deepcopy(axis),
        },
    }


def _bias(case):
    sign = 1.0 if case.endswith("positive") else -1.0
    if case.startswith("tx"):
        return [sign * 0.00012, 0.0, 0.0], [0.0, 0.0, 0.0]
    if case.startswith("tz"):
        return [0.0, 0.0, sign * 0.00012], [0.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0], [sign * 0.003490658503988659, 0.0, 0.0]


def _classical():
    config = json.loads(
        (ROOT / "configs/rj45-classical-tuning-v2.json").read_text()
    )
    config_sha = _canonical_sha(config)
    source_sha = "b" * 64
    controllers = {candidate["id"]: str(index + 1) * 64 for index, candidate in enumerate(config["candidates"])}
    rows = []
    success_counts = {"V2A": 4, "V2B": 3, "V2C": 2, "V2D": 1}
    for candidate in config["candidates"]:
        candidate_id = candidate["id"]
        for case_index, case in enumerate(CASES):
            translation, rotation = _bias(case)
            rows.append({
                "slot": {
                    "ordinal": len(rows) + 1,
                    "stage": "classical_v2_tuning",
                    "candidate_id": candidate_id,
                    "controller_sha256": controllers[candidate_id],
                    "registered_config_sha256": config_sha,
                    "grasp_id": "rj45-train-03",
                    "case_id": case,
                    "environment_seed": 20261121,
                    "requested_translation_bias_entry_m": translation,
                    "requested_rotation_bias_entry_rad": rotation,
                    "requested_applied_bias_absolute_tolerance": 1.0e-9,
                },
                "verification": {
                    "verified": True,
                    "registered_config_sha256": config_sha,
                    "classical_source_sha256": source_sha,
                    "controller_sha256": controllers[candidate_id],
                },
                "metrics": {
                    "success": case_index < success_counts[candidate_id],
                    "hard_contact": False,
                    "forbidden_contact": False,
                    "evaluator_force_p95_n": float(int(candidate_id[-1], 36)),
                    "completion_time_s": 1.0,
                },
            })
    selection = {
        "candidate_id": "V2A",
        "success_count": 4,
        "evaluator_force_p95_n": 10.0,
        "completion_time_median_s": 1.0,
        "strong_baseline_qualifier_met": True,
        "safe": True,
        "new_admission_baseline_episode_count": 12,
        "strong_qualifier_does_not_gate_baseline": True,
    }
    tuning_sha = _canonical_sha(rows)
    admission = []
    for grasp in GRASPS:
        for case in CASES:
            translation, rotation = _bias(case)
            ordinal = len(admission) + 1
            admission.append({
                "schema_name": "rj45-g1-classical-verified-row-v3",
                "report_sha256": "c" * 64,
                "trace_sha256": "d" * 64,
                "archive_payload_sha256": "e" * 64,
                "reviewed_source_freeze_sha256": "f" * 64,
                "physics_compatibility_sha256": "0" * 64,
                "verified_tuning_rows_sha256": tuning_sha,
                "raw_replay_verified": True,
                "archive_verified": True,
                "runtime_closure_verified": True,
                "pre_end_physics_matched": True,
                "wrapper": {
                    "slot": {
                        "ordinal": ordinal,
                        "stage": "classical_v2_new_admission_baseline",
                        "candidate_id": "V2A",
                        "controller_sha256": controllers["V2A"],
                        "grasp_id": grasp,
                        "case_id": case,
                        "environment_seed": 20261121,
                        "requested_translation_bias_entry_m": translation,
                        "requested_rotation_bias_entry_rad": rotation,
                        "requested_applied_bias_absolute_tolerance": 1.0e-9,
                    },
                    "verification": {
                        "verified": True,
                        "registered_config_sha256": config_sha,
                        "classical_source_sha256": source_sha,
                        "controller_sha256": controllers["V2A"],
                    },
                    "metrics": {
                        "success": True,
                        "hard_contact": False,
                        "forbidden_contact": False,
                        "evaluator_force_p95_n": 10.0,
                        "completion_time_s": 1.0,
                    },
                },
            })
    tuning_binding = {"contract": "1" * 64, "analysis": "2" * 64}
    return {
        "config": config,
        "verified_tuning_rows": rows,
        "selection": selection,
        "admission_rows": admission,
        "source_bindings": {
            "tuning": tuning_binding,
            "admission": {**tuning_binding, "verified_tuning_rows_sha256": tuning_sha},
        },
    }


def _physical_seals():
    evidence = {f"evidence/{index:02d}": "1" * 64 for index in range(50)}
    runtime = {f"runtime/{index:02d}": "2" * 64 for index in range(50)}
    shared = {path: "3" * 64 for path in SHARED}
    common = {
        "evidence_normalized_sha256": evidence,
        "runtime_normalized_sha256": runtime,
        "runtime_versions": {"isaac_sim": "5.1.0", "isaac_lab": "2.3.2"},
        "held_bank_manifest_sha256": "4" * 64,
        "held_artifact_sha256": {grasp: "5" * 64 for grasp in GRASPS},
        "task_contract": [
            0.00005, 0.008726646259971648, 0.012, 0.013,
            0.4, 25.0, 0.005, 0.05,
        ],
        "shared_current_sha256": shared,
    }
    return {
        family: {
            **deepcopy(common),
            "family_source_sha256": character * 64,
            "full_commit": f"{family}-commit",
        }
        for family, character in zip(
            ("raw90", "observability", "classical"), ("6", "7", "8"), strict=True
        )
    }


def make_valid_bundle():
    """Shared synthetic adapter fixture; it carries no actual admission claim."""
    return {
        "schema_name": "rj45-g1-synthesis-bundle-v3",
        "raw90": _raw90(),
        "observability": _observability(),
        "classical": _classical(),
        "physical_seals": _physical_seals(),
        "framework": {
            "archive_closure_verified": True,
            "source_closure_verified": True,
            "heldout_closure_verified": True,
            "probe_safety_closure_verified": True,
            "guard_closure_verified": True,
            "held_out_access_detected": False,
        },
    }


def _result(bundle=None):
    return synthesize_g1_v3(make_valid_bundle() if bundle is None else bundle)


def test_valid_bundle_is_only_eligible_for_independent_review():
    result = _result()
    assert result["state"] == "eligible_for_independent_g1_review_v3"
    assert all(gate["pass"] for gate in result["fatal_gates"].values())
    assert result["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
        "training_authorized": False,
    }
    assert result["historical_authority"] == {
        "source_closed_full_se3_evaluator_terminal_scalar": True,
        "per_frame_geometry_replay": False,
        "old_v1_v2_success_retroactively_authoritative": False,
        "external_verification_required": True,
    }


def test_qualifier_changes_do_not_change_fatal_state():
    bundle = make_valid_bundle()
    for row in bundle["classical"]["admission_rows"]:
        row["wrapper"]["metrics"]["success"] = False
    result = _result(bundle)
    assert result["state"] == "eligible_for_independent_g1_review_v3"
    assert result["qualifiers"]["classical_fresh_safe_successes_out_of_12"] == 0


def test_tuning_four_of_six_qualifier_does_not_change_fatal_state():
    bundle = make_valid_bundle()
    changed = False
    for wrapper in bundle["classical"]["verified_tuning_rows"]:
        if wrapper["slot"]["candidate_id"] == "V2A" and wrapper["metrics"]["success"]:
            wrapper["metrics"]["success"] = False
            changed = True
            break
    assert changed
    bundle["classical"]["selection"].update({
        "success_count": 3,
        "strong_baseline_qualifier_met": False,
    })
    tuning_sha = _canonical_sha(bundle["classical"]["verified_tuning_rows"])
    bundle["classical"]["source_bindings"]["admission"]["verified_tuning_rows_sha256"] = tuning_sha
    for row in bundle["classical"]["admission_rows"]:
        row["verified_tuning_rows_sha256"] = tuning_sha
    result = _result(bundle)
    assert result["state"] == "eligible_for_independent_g1_review_v3"
    assert result["qualifiers"]["strong_baseline_4_of_6"] is False


def test_zero_ceiling_headroom_and_rx_control_are_qualifiers_only():
    bundle = make_valid_bundle()
    for row in bundle["raw90"]["rows"]:
        if row["stage"] == "admission" and row["method_id"] == "zero" and row["slot"] == 1:
            row["success"] = False
    result = _result(bundle)
    assert result["state"] == "eligible_for_independent_g1_review_v3"
    assert result["qualifiers"]["zero_slot1_safe_successes_out_of_12"] == 0
    assert result["qualifiers"]["oracle_zero_headroom_cases"] == 10
    assert result["qualifiers"]["rx_control_cases"] == 2


def test_legacy_v1_status_or_decision_cannot_open_a_failed_gate():
    bundle = make_valid_bundle()
    bundle["raw90"]["rows"][0]["legacy_v1_status"] = "admit_rl_pilot"
    bundle["raw90"]["rows"][0]["legacy_v1_decision"] = {"state": "admit_rl_pilot"}
    result = _result(bundle)
    assert result["state"] == "fatal_gate_failed_v3"
    assert "raw90_row_shape:1" in result["fatal_gates"]["contract_and_seal"]["reasons"]


@pytest.mark.parametrize("mutation", ["source", "origin", "physics", "missing", "heldout", "leakage"])
def test_contract_source_origin_physics_missing_and_heldout_fail_closed(mutation):
    bundle = make_valid_bundle()
    if mutation == "source":
        bundle["observability"]["current_controller_sources"]["force_slope_probe"]["provider_source_sha256"] = "0" * 64
    elif mutation == "origin":
        bundle["observability"]["current_controller_sources"]["force_slope_probe"]["provider_import_origin"] = "/forged.py"
    elif mutation == "physics":
        bundle["physical_seals"]["classical"]["runtime_normalized_sha256"]["runtime/00"] = "0" * 64
    elif mutation == "missing":
        del bundle["classical"]["admission_rows"][0]["trace_sha256"]
    elif mutation == "heldout":
        bundle["framework"]["held_out_access_detected"] = True
    else:
        bundle["observability"]["config"]["held_out_access"] = True
        for stage in ("development", "confirmation"):
            aggregate = bundle["observability"][stage]["aggregate"]
            aggregate["config"] = bundle["observability"]["config"]
            aggregate["config_sha256"] = _canonical_sha(bundle["observability"]["config"])
            bundle["observability"][stage]["recomputed"] = deepcopy(aggregate)
    assert _result(bundle)["state"] == "fatal_gate_failed_v3"


def test_aggregate_recomputation_source_relation_and_probe_closure_are_fatal():
    aggregate = make_valid_bundle()
    aggregate["observability"]["confirmation"]["aggregate"]["pass"] = False
    assert _result(aggregate)["state"] == "fatal_gate_failed_v3"

    relation = make_valid_bundle()
    relation["classical"]["source_bindings"]["admission"]["verified_tuning_rows_sha256"] = "0" * 64
    assert _result(relation)["state"] == "fatal_gate_failed_v3"

    closure = make_valid_bundle()
    closure["framework"]["probe_safety_closure_verified"] = False
    result = _result(closure)
    assert result["state"] == "fatal_gate_failed_v3"
    assert result["fatal_gates"]["framework_and_probe_safety"]["pass"] is False


def test_zero_slot2_never_changes_counted_zero_qualifier():
    bundle = make_valid_bundle()
    before = _result(bundle)["qualifiers"]["zero_slot1_safe_successes_out_of_12"]
    for row in bundle["raw90"]["rows"]:
        if row["stage"] == "admission" and row["method_id"] == "zero" and row["slot"] == 2:
            row["success"] = not row["success"]
    after = _result(bundle)
    assert after["state"] == "eligible_for_independent_g1_review_v3"
    assert after["qualifiers"]["zero_slot1_safe_successes_out_of_12"] == before


def test_unsafe_oracle_is_fatal_but_unsafe_fresh_classical_is_qualifier_only():
    oracle = make_valid_bundle()
    target = next(
        row for row in oracle["raw90"]["rows"]
        if row["stage"] == "admission" and row["method_id"] == "oracle_gt_reference"
    )
    target["hard_contact"] = True
    assert _result(oracle)["state"] == "fatal_gate_failed_v3"

    classical = make_valid_bundle()
    classical["classical"]["admission_rows"][0]["wrapper"]["metrics"]["hard_contact"] = True
    result = _result(classical)
    assert result["state"] == "eligible_for_independent_g1_review_v3"
    assert result["qualifiers"]["classical_fresh_unsafe_episodes"] == 1

    historical_classical = make_valid_bundle()
    old_row = next(
        row for row in historical_classical["raw90"]["rows"]
        if row["stage"] == "admission" and row["method_id"] == "classical_force_search"
    )
    old_row["hard_contact"] = True
    historical_result = _result(historical_classical)
    assert historical_result["state"] == "eligible_for_independent_g1_review_v3"
    assert historical_result["qualifiers"]["raw_method_hard_or_forbidden_episodes"][
        "classical_force_search"
    ] == 1


def test_confirmation_pass_alone_has_no_authority():
    bundle = make_valid_bundle()
    del bundle["raw90"]
    result = _result(bundle)
    assert result["state"] == "fatal_gate_failed_v3"
    assert result["fatal_gates"]["controlled_confirmation"]["pass"] is True
    assert result["fatal_gates"]["contract_and_seal"]["pass"] is False
    assert result["fatal_gates"]["oracle_feasibility"]["pass"] is False


def test_incomplete_classical_stage_is_explicit():
    bundle = make_valid_bundle()
    bundle["classical"]["admission_rows"].pop()
    result = _result(bundle)
    assert result["state"] == "fatal_gate_failed_v3"
    assert result["stage_completion"]["classical_tuning_24"] is True
    assert result["stage_completion"]["classical_fresh_admission_12"] is False


def test_no_safe_tuning_candidate_is_contract_fatal():
    bundle = make_valid_bundle()
    for wrapper in bundle["classical"]["verified_tuning_rows"]:
        wrapper["metrics"]["hard_contact"] = True
    result = _result(bundle)
    assert result["state"] == "fatal_gate_failed_v3"
    assert "classical_no_safe_tuning_candidate" in result["fatal_gates"]["contract_and_seal"]["reasons"]


def test_nonfinite_json_is_rejected_before_synthesis():
    bundle = make_valid_bundle()
    bundle["classical"]["admission_rows"][0]["wrapper"]["metrics"]["completion_time_s"] = float("nan")
    with pytest.raises(ValueError, match="finite canonical JSON"):
        synthesize_g1_v3(bundle)


@pytest.mark.parametrize("target", ["classical_config", "aggregate_results", "raw_method"])
def test_malformed_nested_json_is_normalized_to_fatal_state(target):
    bundle = make_valid_bundle()
    if target == "classical_config":
        bundle["classical"]["config"]["shared_controller_contract"] = []
    elif target == "aggregate_results":
        bundle["observability"]["confirmation"]["aggregate"]["results"] = None
        bundle["observability"]["confirmation"]["recomputed"]["results"] = None
    else:
        bundle["raw90"]["rows"][18]["method_id"] = "forged"
    assert _result(bundle)["state"] == "fatal_gate_failed_v3"
