from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "docs/evidence/rj45-multibias-20261031/audit_results.py"
SPEC = importlib.util.spec_from_file_location("audit_rj45_multibias_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl_gz(path: Path, rows: list[dict]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    path.write_bytes(gzip.compress(raw, mtime=0))
    return hashlib.sha256(raw).hexdigest()


def _terminal(success: bool) -> dict:
    criteria = {key: True for key in AUDIT.CRITERIA_KEYS}
    if not success:
        criteria["depth"] = False
    return {
        "valid": True, "success": success, "criteria": criteria,
        "metrics": {
            "actual_depth_m": 0.012, "forbidden_contact": 0.0,
            "force_peak_n": 2.0, "grasp_both": 1.0,
            "lateral_error_m": 0.00001, "orientation_error_rad": 0.001,
        },
        "evaluator_source": "simulator_ground_truth_independent",
        "evaluator_version": "task-evaluator-v1+environment-cross-check",
        "termination_reason": "independent_success" if success else "timeout",
        "terminal_timestamp_s": 12.0 if success else 24.0,
    }


def _evaluation(path: Path, success: bool, base: list[float], *, reference: bool = False) -> dict:
    classification = "valid_success" if success else "valid_failure"
    reference_step = {
        "active": True, "requested_raw_action_6d": [0.2] * 6,
        "bounded_action_6d": [0.1] * 6,
        "physical_offset_entry_6d": [0.001] * 6, "bound_limited": True,
    } if reference else None
    rows = [
        {
            "phase": "insert", "active": True, "policy_input_valid": True,
            "observation": {"base79": base}, "applied_expanded_action": [0.0] * 12,
            "reference_controller": reference_step,
        },
        {"event": "complete", "classification": classification},
    ]
    trace_sha = _write_jsonl_gz(path, rows)
    return {
        "classification": classification, "terminal": _terminal(success),
        "active_rows": 1,
        "reference_controller": {
            "active_rows": 1, "nonzero_bounded_action_rows": 1,
            "recovery_correction_observed": True,
        } if reference else None,
        "observation": {
            "dimensions": {"policy": 79, "critic": 92},
            "actor_ground_truth_access": False,
            "base79_sha256": AUDIT.raw_base79_sha256(base),
        },
        "evaluation_rng": {
            "seeded": {
                "seed": 20261124, "python": True, "numpy": True, "torch_cpu": True,
                "torch_cuda_all_available_devices": True, "physics_seed_unchanged": 20261128,
            },
            "state_sha256_before_environment_reset": {
                "python": "1" * 64, "numpy": "2" * 64,
                "torch_cpu": "3" * 64, "torch_cuda": "4" * 64,
            },
        },
        "trace": {"path": str(path), "sha256": trace_sha},
    }


def _claims() -> dict:
    return {"effect_proven": False, "generalization_proven": False, "final_policy": False}


def _training_rng(state_digit: str) -> dict:
    return {
        "seed": 20261121, "python": True, "numpy": True, "torch_cpu": True,
        "torch_cuda_all_available_devices": True, "physics_seed_unchanged": 20261128,
        "state_sha256": {
            "python": state_digit * 64, "numpy": state_digit * 64,
            "torch_cpu": state_digit * 64, "torch_cuda": state_digit * 64,
        },
    }


def _observed_bias(case: dict, translation: list[float]) -> dict:
    return {
        "base_seed": case["base_seed"], "enabled": True, "level": 0,
        "stream_offset": 45123017, "episode_stride": 1000003,
        "case_seed_formula": "base_seed + stream_offset + episode_index * episode_stride + env_id",
        "sampling_event": "once_per_environment_reset_before_calibration",
        "episode_index": [1], "translation_entry_m": [translation],
        "rotation_vector_entry_rad": [[0.0, 0.0, 0.0]],
    }


def _common(source: dict, contract_sha: str, phase: str) -> dict:
    return {
        "schema_name": "rj45_multibias_probe_v1", "status": "completed", "phase": phase,
        "claims": _claims(), "source_binding": source,
        "contract": {"path": "/remote/contract.json", "sha256": contract_sha},
    }


def _training_fixture(root: Path, phase: str, source: dict, contract_sha: str, checkpoint_sha: str) -> dict:
    directory = root / phase
    pool = AUDIT.TRAIN_POOLS[phase]
    all_cases = [AUDIT.expected_case(case_id, seed, "training-pool") for case_id, seed in AUDIT.PREFLIGHT_CASES]
    cases = all_cases[:1] if phase == "train-single" else all_cases
    bias_cases = [
        {"case_id": case["case_id"], "base_seed": case["base_seed"],
         "calibration_case_seed": case["calibration_case_seed"],
         "translation_entry_m": [0.0, 0.0, 0.0],
         "rotation_vector_entry_rad": [0.0, 0.0, 0.0]}
        for case in cases
    ]
    updates = []
    for index in range(1, 65):
        row = {
            "update": index, "metrics": {"loss": 1.0 / index},
            "rollout_targets": {"returns": 128, "advantages": 128},
            "parameter_finiteness": {
                "actor_parameters": 8, "actor_scalars": 100,
                "critic_parameters": 8, "critic_scalars": 100,
            },
        }
        if index == 32:
            row["diagnostic_checkpoint"] = {"sha256": "d" * 64, "eligible_for_selection": False}
        updates.append(row)
    _write_jsonl_gz(directory / "updates.jsonl.gz", updates)
    counts = {case_id: {"transitions": 0, "active_valid": 0, "terminal": 0} for case_id in pool}
    transitions = []
    for index in range(1, 8193):
        case_id = pool[(index - 1) % len(pool)]
        done = index % 128 == 0
        row = {
            "transition": index, "case_id": case_id, "reward": 0.1,
            "done": done, "valid_input": True, "active": True,
            "action": [0.0] * 6, "applied_expanded_action": [0.0] * 12,
        }
        transitions.append(row)
        counts[case_id]["transitions"] += 1
        counts[case_id]["active_valid"] += 1
        counts[case_id]["terminal"] += int(done)
    _write_jsonl_gz(directory / "transitions.jsonl.gz", transitions)
    resets = []
    reset_counts = {case_id: 0 for case_id in pool}
    for ordinal in range(65):
        case_id = pool[ordinal % len(pool)]
        case = cases[ordinal % len(cases)]
        resets.append({
            "phase": "training", "ordinal": ordinal, "pool_index": ordinal % len(pool),
            "case_id": case_id, "base_seed": case["base_seed"], "calibration_case_seed": case["calibration_case_seed"],
            "actual_translation_entry_m": [0.0, 0.0, 0.0],
            "actual_rotation_vector_entry_rad": [0.0, 0.0, 0.0],
            "episode_index": 1, "grasp_index": 1,
        })
        reset_counts[case_id] += 1
    _write_jsonl_gz(directory / "resets.jsonl.gz", resets)
    base = [0.0] * 79
    _evaluation(directory / "pre-evaluation.jsonl.gz", False, base)
    first = {case_id: {"values": base, "sha256": AUDIT.raw_base79_sha256(base)} for case_id in pool}
    initialization = {
        "actor": {
            "mode": "standard79", "seed": 20261123, "state_sha256": "1" * 64,
            "common_79d_projection_sha256": "2" * 64,
            "oracle_extra_13_columns_zero": True, "output_layer_zero": True,
            "extra_columns_trainable": False,
        },
        "critic": {
            "seed": 20261122, "device": "cuda:0", "resettable_module_count": 4,
            "state_sha256": "3" * 64, "actor_input_width_independent": True,
        },
        "optimizer_sha256": "4" * 64,
        "zero_actor": {
            "mode": "standard79", "sample_rows": 3, "action_shape": [3, 6],
            "mean_action_all_zero": True, "action_sha256": "9" * 64,
            "noise": {"minimum": 0.1, "maximum": 0.1, "mean": 0.1, "shape": [6]},
        },
        "rng": _training_rng("5"),
    }
    report = {
        **_common(source, contract_sha, phase), "learning_executed": True, "updates": 64,
        "training_pool": cases, "bias_cases": bias_cases,
        "transitions": {"total_transitions": 8192, "by_case": counts, "first_active_base79": first},
        "reset_count": len(resets), "reset_count_by_case": reset_counts,
        "initialization": initialization, "training_rng": _training_rng("5"),
        "actor_parameter_sha256_before": "1" * 64, "actor_parameter_sha256_after": "8" * 64,
        "checkpoint": {"path": f"/remote/{phase}/final_checkpoint.pt", "sha256": checkpoint_sha, "update": 64, "eligible_for_evaluation": True},
        "diagnostic_checkpoint": {"path": f"/remote/{phase}/diagnostic_checkpoint_0032.pt", "sha256": "d" * 64, "eligible_for_selection": False},
    }
    _write_json(root / f"{phase}-report.json", report)
    return report


def _complete_fixture(root: Path) -> None:
    evaluation_cases = [AUDIT.expected_case(case_id, seed, source) for case_id, seed, source in AUDIT.EVALUATION_CASES]
    contract = {
        "schema_name": "rj45_multibias_contract_v1",
        "helper_sha256": AUDIT.EXPECTED_HELPER_SHA256,
        "source_sha256": AUDIT.EXPECTED_SOURCE_SHA256,
        "origin_protocol": {
            "path_name": "origin-protocol.json",
            "sha256": AUDIT.EXPECTED_ORIGIN_PROTOCOL_SHA256,
            "template_only": True,
        },
        "claims": {"dev_split_sealed": True, "effect_proven": False, "final_split_sealed": True, "generalization_proven": False},
        "training": {
            "mode": "standard79", "num_envs": 1, "actor_observation_dim": 79,
            "critic_observation_dim": 92, "action_dim": 6, "right_arm_only": True,
            "held_bank_relative_path": "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json",
            "updates": 64, "rollout_steps": 128, "transitions": 8192,
            "single_pool": [20261128], "multi_pool": [20261128, 20271208, 20271216, 20271224],
            "policy_training_seed": 20261121, "critic_initialization_seed": 20261122,
            "actor_initialization_seed": 20261123, "evaluation_seed": 20261124,
            "final_checkpoint_update": 64, "eligible_checkpoint_updates": [64],
            "diagnostic_checkpoint_updates": [32],
            "checkpoint_selection": "final64_only_no_outcome_selection",
        },
        "reference_preflight": {
            "pool": [seed for _, seed in AUDIT.PREFLIGHT_CASES],
            "required_successes": 4, "episode_count": 4,
            "failure_policy": "no_replacement_no_ranking",
        },
        "evaluation": {
            "cases": evaluation_cases, "methods": list(AUDIT.METHODS),
            "episode_count": 27, "order": "case_outer_cyclic_method_offset_case_index_mod_3",
            "per_episode_wall_time_s": 300,
        },
        "decision_rule": {
            "effect_proven": False, "checkpoint_replacement_allowed": False,
            "followup_multiseed_candidate": "fresh8_multi_successes_at_least_single_plus_2_and_greater_than_zero_and_zero_success_multi_failure_count_at_most_1",
        },
        "budget_and_stop": {
            "admission": {
                "budget_cap_usd": 700, "external_guard_minutes": 180,
                "locally_verified": True, "new_run_reservation_usd": 6,
                "private_evidence_sha256": "306b7ca200fda99e3f88673bf59338d6c4f861d52829d9f84e836483baa0be31",
            },
            "deadline_utc": "2026-10-31T10:50:52Z",
            "train_phase_wall_time_s": 3200, "evaluate_phase_wall_time_s": 4000,
            "external_stop_wall_time_s": 10800,
        },
    }
    _write_json(root / "contract.json", contract)
    contract_sha = AUDIT.file_sha256(root / "contract.json")
    source = {
        **{name: {"path": f"/remote/{name}.py", "sha256": digest} for name, digest in AUDIT.EXPECTED_SOURCE_SHA256.items()},
        "origin_protocol": {"path": "/remote/origin.json", "sha256": AUDIT.EXPECTED_ORIGIN_PROTOCOL_SHA256},
    }
    preflight_episodes = []
    preflight_cases = [AUDIT.expected_case(case_id, seed, "training-pool") for case_id, seed in AUDIT.PREFLIGHT_CASES]
    preflight_bias = []
    for index, case in enumerate(preflight_cases, 1):
        sampled = {
            "case_id": case["case_id"], "base_seed": case["base_seed"],
            "calibration_case_seed": case["calibration_case_seed"],
            "translation_entry_m": [0.0, 0.0, 0.0],
            "rotation_vector_entry_rad": [0.0, 0.0, 0.0],
        }
        preflight_bias.append(sampled)
        trace = root / "preflight/traces" / f"{index:02d}-{case['case_id']}-reference.jsonl.gz"
        preflight_episodes.append({
            "case": case, "evaluation": _evaluation(trace, True, [0.0] * 79, reference=True),
            "observed_bias": _observed_bias(case, [0.0, 0.0, 0.0]),
        })
    preflight = {
        **_common(source, contract_sha, "preflight"), "episodes": preflight_episodes,
        "strict_successes": 4, "ranking_eligible": True, "bias_cases": preflight_bias,
    }
    _write_json(root / "preflight-report.json", preflight)
    single = _training_fixture(root, "train-single", source, contract_sha, "1" * 64)
    multi = _training_fixture(root, "train-multi", source, contract_sha, "2" * 64)
    cases = evaluation_cases
    evaluation_bias = [
        {"case_id": case["case_id"], "base_seed": case["base_seed"],
         "calibration_case_seed": case["calibration_case_seed"],
         "translation_entry_m": [index / 1000.0, 0.0, 0.0],
         "rotation_vector_entry_rad": [0.0, 0.0, 0.0]}
        for index, case in enumerate(cases)
    ]
    episodes = []
    successes = {method: 0 for method in AUDIT.METHODS}
    for case_index, case in enumerate(cases):
        methods = AUDIT.METHODS[case_index % 3:] + AUDIT.METHODS[:case_index % 3]
        for method in methods:
            fresh_index = case_index
            success = case_index == 0 or (method == "multi64" and fresh_index <= 6) or (method == "single64" and fresh_index <= 3) or (method == "zero" and fresh_index <= 2)
            if case["source"] == "policy-unseen" and success:
                successes[method] += 1
            index = len(episodes) + 1
            trace = root / "evaluate/traces" / f"{index:02d}-{case['case_id']}-{method}.jsonl.gz"
            evaluation = _evaluation(trace, success, [case_index / 100.0] * 79)
            episodes.append({
                "index": index, "case": case, "method": method, "evaluation": evaluation,
                "initial_base79_sha256": AUDIT.raw_base79_sha256([case_index / 100.0] * 79),
                "bias": _observed_bias(case, [case_index / 1000.0, 0.0, 0.0]),
            })
    evaluate = {
        **_common(source, contract_sha, "evaluate"), "episodes": episodes,
        "bias_cases": evaluation_bias,
        "effect_proven": False, "checkpoint_selection_changed": False,
        "fresh8_successes": successes, "zero_success_multi_failure_count": 0,
        "followup_multiseed_candidate": True,
        "checkpoints": {
            "single64": {
                "path": "/remote/train-single/final_checkpoint.pt",
                "sha256": single["checkpoint"]["sha256"], "iteration_field": 63,
                "eligible_update": 64, "eligible_for_evaluation": True,
            },
            "multi64": {
                "path": "/remote/train-multi/final_checkpoint.pt",
                "sha256": multi["checkpoint"]["sha256"], "iteration_field": 63,
                "eligible_update": 64, "eligible_for_evaluation": True,
            },
        },
    }
    _write_json(root / "evaluate-report.json", evaluate)


def test_complete_fixture_passes_and_recomputes_candidate(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    result = AUDIT.audit(tmp_path)
    assert result["status"] == "pass"
    assert result["ranking_allowed"] is True
    assert result["evaluation"]["fresh8_successes"] == {"single64": 3, "multi64": 6, "zero": 2}
    assert result["evaluation"]["paired"]["multi64_vs_single64"] == {"wins": 3, "losses": 0, "ties": 5}
    assert result["evaluation"]["followup_multiseed_candidate"] is True
    assert result["claims"]["effect_proven"] is False


def test_terminal_classification_contradiction_is_rejected() -> None:
    evaluation = {"classification": "valid_success", "terminal": _terminal(False)}
    with pytest.raises(AUDIT.AuditError, match="success/classification contradiction"):
        AUDIT.strict_terminal(evaluation, "episode")


def test_nonfinite_update_metric_is_rejected(tmp_path: Path) -> None:
    rows = [
        {"update": index, "metrics": {"loss": float("nan") if index == 7 else 0.1},
         "rollout_targets": {"returns": 128, "advantages": 128},
         "parameter_finiteness": {
             "actor_parameters": 8, "actor_scalars": 100,
             "critic_parameters": 8, "critic_scalars": 100,
         },
         **({"diagnostic_checkpoint": {"eligible_for_selection": False}} if index == 32 else {})}
        for index in range(1, 65)
    ]
    _write_jsonl_gz(tmp_path / "train-single/updates.jsonl.gz", rows)
    with pytest.raises(AUDIT.AuditError, match="must be finite"):
        AUDIT.audit_updates(tmp_path, "train-single")


def test_reset_cycle_mutation_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-multi/resets.jsonl.gz"
    rows, _ = AUDIT.load_jsonl_gzip(path)
    rows[3]["pool_index"] = 0
    _write_jsonl_gz(path, rows)
    with pytest.raises(AUDIT.AuditError, match="reset cycle changed"):
        AUDIT.audit(tmp_path)


def test_checkpoint_lineage_mutation_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    report_path = tmp_path / "evaluate-report.json"
    report = AUDIT.load_json(report_path)
    report["checkpoints"]["multi64"]["sha256"] = "f" * 64
    _write_json(report_path, report)
    with pytest.raises(AUDIT.AuditError, match="checkpoint lineage mismatch"):
        AUDIT.audit(tmp_path)


def test_evaluation_checkpoint_iteration_is_final64_only(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "evaluate-report.json"
    report = AUDIT.load_json(path)
    report["checkpoints"]["single64"]["iteration_field"] = 31
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="not final64"):
        AUDIT.audit(tmp_path)


def test_contract_case_seed_mutation_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "contract.json"
    contract = AUDIT.load_json(path)
    contract["evaluation"]["cases"][2]["base_seed"] += 8
    _write_json(path, contract)
    # Keep report binding aligned so the semantic identity gate, not the byte gate, rejects it.
    digest = AUDIT.file_sha256(path)
    for phase in AUDIT.PHASES:
        report_path = tmp_path / f"{phase}-report.json"
        report = AUDIT.load_json(report_path)
        report["contract"]["sha256"] = digest
        _write_json(report_path, report)
    with pytest.raises(AUDIT.AuditError, match="contract evaluation cases changed"):
        AUDIT.audit(tmp_path)


@pytest.mark.parametrize("invalid", [None, []])
def test_invalid_report_contract_binding_is_failclosed_audit_error(tmp_path: Path, invalid: object) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-single-report.json"
    report = AUDIT.load_json(path)
    report["contract"] = invalid
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="contract binding is invalid"):
        AUDIT.audit(tmp_path)
    output = tmp_path / "audit.json"
    assert AUDIT.main(["--root", str(tmp_path), "--output", str(output), "--allow-incomplete"]) == 0
    result = AUDIT.load_json(output)
    assert result["status"] == "incomplete" and result["ranking_allowed"] is False


def test_empty_rollout_proof_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-single/updates.jsonl.gz"
    rows, _ = AUDIT.load_jsonl_gzip(path)
    rows[4]["rollout_targets"] = {}
    _write_jsonl_gz(path, rows)
    with pytest.raises(AUDIT.AuditError, match="rollout target keys changed"):
        AUDIT.audit(tmp_path)


def test_diagnostic_checkpoint_must_appear_only_at_update32_and_match_report(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-single/updates.jsonl.gz"
    rows, _ = AUDIT.load_jsonl_gzip(path)
    rows[7]["diagnostic_checkpoint"] = {"sha256": "d" * 64, "eligible_for_selection": False}
    _write_jsonl_gz(path, rows)
    with pytest.raises(AUDIT.AuditError, match="placement changed"):
        AUDIT.audit(tmp_path)
    _complete_fixture(tmp_path)
    rows, _ = AUDIT.load_jsonl_gzip(path)
    rows[31]["diagnostic_checkpoint"]["sha256"] = "e" * 64
    _write_jsonl_gz(path, rows)
    with pytest.raises(AUDIT.AuditError, match="SHA differs"):
        AUDIT.audit(tmp_path)


def test_missing_initialization_field_is_audit_error(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-multi-report.json"
    report = AUDIT.load_json(path)
    del report["initialization"]["optimizer_sha256"]
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="initialization structure changed"):
        AUDIT.audit(tmp_path)


def test_empty_actor_initialization_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-single-report.json"
    report = AUDIT.load_json(path)
    report["initialization"]["actor"] = {}
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="actor initialization structure changed"):
        AUDIT.audit(tmp_path)


def test_training_rng_flags_and_state_are_required(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "train-single-report.json"
    report = AUDIT.load_json(path)
    report["training_rng"]["physics_seed_unchanged"] = 20261129
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="training RNG seed/flags changed"):
        AUDIT.audit(tmp_path)


def test_observed_bias_mutation_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "preflight-report.json"
    report = AUDIT.load_json(path)
    report["episodes"][2]["observed_bias"]["translation_entry_m"][0][0] = 0.01
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="observed translation bias changed"):
        AUDIT.audit(tmp_path)


def test_reduced_terminal_criteria_are_rejected() -> None:
    terminal = _terminal(True)
    terminal["criteria"] = {"depth": True}
    with pytest.raises(AUDIT.AuditError, match="criteria keys changed"):
        AUDIT.strict_terminal({"classification": "valid_success", "terminal": terminal}, "episode")


def test_reduced_terminal_metrics_are_rejected() -> None:
    terminal = _terminal(True)
    terminal["metrics"] = {"actual_depth_m": 0.012}
    with pytest.raises(AUDIT.AuditError, match="metric keys changed"):
        AUDIT.strict_terminal({"classification": "valid_success", "terminal": terminal}, "episode")


def test_evaluation_rng_mutation_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "evaluate-report.json"
    report = AUDIT.load_json(path)
    report["episodes"][4]["evaluation"]["evaluation_rng"]["seeded"]["seed"] = 20261125
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="RNG seed/flags changed"):
        AUDIT.audit(tmp_path)


def test_evaluation_observation_proof_is_required(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "evaluate-report.json"
    report = AUDIT.load_json(path)
    report["episodes"][0]["evaluation"]["observation"] = None
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="observation proof is missing"):
        AUDIT.audit(tmp_path)


def test_trace_filename_fallback_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    expected = tmp_path / "evaluate/traces/01-anchor-single64.jsonl.gz"
    expected.rename(expected.with_name("01-wrong-name.jsonl.gz"))
    with pytest.raises(AUDIT.AuditError, match="exact trace file is missing"):
        AUDIT.audit(tmp_path)


def test_bias_seed_mutation_is_rejected(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "preflight-report.json"
    report = AUDIT.load_json(path)
    report["bias_cases"][1]["calibration_case_seed"] += 1
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="calibration seed changed"):
        AUDIT.audit(tmp_path)


def test_preflight_reference_controller_report_is_required(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    path = tmp_path / "preflight-report.json"
    report = AUDIT.load_json(path)
    report["episodes"][0]["evaluation"]["reference_controller"] = None
    _write_json(path, report)
    with pytest.raises(AUDIT.AuditError, match="reference-controller report structure changed"):
        AUDIT.audit(tmp_path)


def test_preflight_reference_controller_nonzero_trace_signal_is_required(tmp_path: Path) -> None:
    _complete_fixture(tmp_path)
    trace_path = tmp_path / "preflight/traces/01-anchor-reference.jsonl.gz"
    rows, _ = AUDIT.load_jsonl_gzip(trace_path)
    rows[0]["reference_controller"]["bounded_action_6d"] = [0.0] * 6
    raw_sha = _write_jsonl_gz(trace_path, rows)
    report_path = tmp_path / "preflight-report.json"
    report = AUDIT.load_json(report_path)
    report["episodes"][0]["evaluation"]["trace"]["sha256"] = raw_sha
    _write_json(report_path, report)
    with pytest.raises(AUDIT.AuditError, match="did not apply a recovery correction"):
        AUDIT.audit(tmp_path)


def test_corrupt_gzip_and_incomplete_mode_fail_closed(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl.gz"
    path.write_bytes(b"not-gzip")
    with pytest.raises(AUDIT.AuditError, match="invalid gzip"):
        AUDIT.load_jsonl_gzip(path)
    output = tmp_path / "audit.json"
    assert AUDIT.main(["--root", str(tmp_path), "--output", str(output), "--allow-incomplete"]) == 0
    result = AUDIT.load_json(output)
    assert result["status"] == "incomplete"
    assert result["ranking_allowed"] is False
    assert result["claims"]["effect_proven"] is False


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    spec = importlib.util.spec_from_file_location("audit_side_effect_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(tmp_path.iterdir()) == before
