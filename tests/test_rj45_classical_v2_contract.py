from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from rby1_assembly.rj45_classical_v2_contract import (
    ClassicalV2ContractError,
    select_and_build_admission_plan,
    tuning_execution_plan,
    validate_classical_v2_config,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/rj45-classical-tuning-v2.json"


def config(*, frozen: bool = False) -> dict:
    result = json.loads(CONFIG_PATH.read_text())
    result["frozen"] = frozen
    return result


def verified_rows(candidate_successes: dict[str, int] | None = None) -> list[dict]:
    candidate_successes = candidate_successes or {
        "V2A": 6, "V2B": 5, "V2C": 4, "V2D": 3,
    }
    frozen = config(frozen=True)
    binding = validate_classical_v2_config(frozen, require_frozen=True)
    by_candidate = {row["candidate_id"]: row for row in binding["candidates"]}
    rows = []
    for slot in tuning_execution_plan(frozen, require_frozen=True):
        candidate_index = int(slot["candidate_id"][-1], 16) - 10
        case_index = len([row for row in rows if row["slot"]["candidate_id"] == slot["candidate_id"]])
        rows.append({
            "slot": slot,
            "verification": {
                "verified": True,
                "registered_config_sha256": binding["registered_config_sha256"],
                "classical_source_sha256": binding["classical_source_sha256"],
                "controller_sha256": by_candidate[slot["candidate_id"]]["controller_sha256"],
            },
            "metrics": {
                "success": case_index < candidate_successes[slot["candidate_id"]],
                "hard_contact": False,
                "forbidden_contact": False,
                "evaluator_force_p95_n": float(1.0 + candidate_index),
                "completion_time_s": float(2.0 + candidate_index),
            },
        })
    return rows


def test_current_config_and_source_payload_are_exact() -> None:
    current = config()
    binding = validate_classical_v2_config(current)
    assert binding["schema"] == "rj45-classical-v2-source-binding-v1"
    frozen_binding = validate_classical_v2_config(config(frozen=True), require_frozen=True)
    assert binding["config_canonical_sha256"] != frozen_binding["config_canonical_sha256"]
    assert binding["registered_config_sha256"] == frozen_binding["registered_config_sha256"]
    assert len(binding["candidates"]) == 4
    for row, configured in zip(binding["candidates"], current["candidates"], strict=True):
        payload = row["canonical_payload"]
        assert row["candidate_id"] == configured["id"]
        assert payload["search_step_m"] == configured["search_step_m"]
        assert payload["force_trigger_n"] == configured["force_trigger_n"]
        assert len(row["controller_sha256"]) == 64


def test_exact_24_tuning_slots_use_train03_and_fixed_seed() -> None:
    plan = tuning_execution_plan(config())
    assert len(plan) == 24
    assert [row["ordinal"] for row in plan] == list(range(1, 25))
    assert {(row["candidate_id"], row["case_id"]) for row in plan} == {
        (candidate, case)
        for candidate in ("V2A", "V2B", "V2C", "V2D")
        for case in ("tx-positive", "tx-negative", "tz-positive", "tz-negative",
                     "rx-positive", "rx-negative")
    }
    assert {row["grasp_id"] for row in plan} == {"rj45-train-03"}
    assert {row["environment_seed"] for row in plan} == {20261121}
    assert {row["requested_applied_bias_absolute_tolerance"] for row in plan} == {1e-9}
    assert next(row for row in plan if row["case_id"] == "tx-positive")[
        "requested_translation_bias_entry_m"
    ] == [0.00012, 0.0, 0.0]
    assert next(row for row in plan if row["case_id"] == "rx-negative")[
        "requested_rotation_bias_entry_rad"
    ] == [-0.003490658503988659, 0.0, 0.0]


def test_frozen_config_selects_and_builds_fresh_exact_12() -> None:
    result = select_and_build_admission_plan(config(frozen=True), verified_rows())
    assert result["selection"]["candidate_id"] == "V2A"
    assert result["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }
    assert result["tuning_results_claimed_verified_by_planner"] is False
    assert result["reuse_tuning_outcomes"] is False
    assert result["common_control_contract"] == {
        "actor_observation_dimension": 79,
        "right_action_dimension": 6,
        "fair_provider_inputs": ["observation79", "provider_owned_state"],
        "requested_applied_bias_absolute_tolerance": 1e-9,
    }
    assert len(result["episodes"]) == 12
    assert [row["ordinal"] for row in result["episodes"]] == list(range(1, 13))
    assert {row["grasp_id"] for row in result["episodes"]} == {
        "rj45-train-02", "rj45-train-05"
    }
    assert {row["case_id"] for row in result["episodes"]} == {
        "tx-positive", "tx-negative", "tz-positive", "tz-negative",
        "rx-positive", "rx-negative",
    }
    assert {row["candidate_id"] for row in result["episodes"]} == {"V2A"}
    assert {row["environment_seed"] for row in result["episodes"]} == {20261121}


def test_selection_target_miss_does_not_remove_new_12_episode_baseline() -> None:
    misses = {candidate: 0 for candidate in ("V2A", "V2B", "V2C", "V2D")}
    result = select_and_build_admission_plan(config(frozen=True), verified_rows(misses))
    assert result["selection"]["strong_baseline_qualifier_met"] is False
    assert len(result["episodes"]) == 12


def test_frozen_required_for_selection_but_false_config_can_register_tuning() -> None:
    assert len(tuning_execution_plan(config())) == 24
    with pytest.raises(ClassicalV2ContractError, match="not frozen"):
        tuning_execution_plan(config(), require_frozen=True)
    with pytest.raises(ClassicalV2ContractError, match="not frozen"):
        select_and_build_admission_plan(config(), verified_rows())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(extra=True), "config differs"),
        (lambda value: value.pop("selection"), "config differs"),
        (lambda value: value.__setitem__("candidate_count", True), "config differs"),
        (lambda value: value.__setitem__("candidate_count", 4.0), "config differs"),
        (lambda value: value["candidates"][0].__setitem__("search_step_m", 5), "config differs"),
        (lambda value: value["candidates"][0].__setitem__("search_step_m", math.nan), "finite"),
        (lambda value: value.__setitem__("signed_cases", tuple(value["signed_cases"])), "config differs"),
        (lambda value: value.__setitem__("candidates", tuple(value["candidates"])), "config differs"),
        (lambda value: value["shared_controller_contract"].__setitem__(
            "safety_latch_is_permanent", 1), "config differs"),
    ],
)
def test_config_rejects_unknown_missing_type_changes_and_nonfinite(mutation, message) -> None:
    candidate = config()
    mutation(candidate)
    with pytest.raises(ClassicalV2ContractError, match=message):
        validate_classical_v2_config(candidate)


def test_after_freeze_rejects_any_registered_value_change() -> None:
    candidate = config(frozen=True)
    candidate["candidates"][3]["force_trigger_n"] = 0.25
    with pytest.raises(ClassicalV2ContractError, match="config differs"):
        validate_classical_v2_config(candidate, require_frozen=True)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(deepcopy(rows[-1])),
        lambda rows: rows[0].update(extra=True),
        lambda rows: rows[0]["slot"].__setitem__("environment_seed", 20261122),
        lambda rows: rows[0]["verification"].__setitem__("verified", 1),
        lambda rows: rows[0]["verification"].__setitem__("controller_sha256", "0" * 64),
        lambda rows: rows[0]["metrics"].__setitem__("success", 1),
        lambda rows: rows[0]["metrics"].__setitem__("evaluator_force_p95_n", 1),
        lambda rows: rows[0]["metrics"].__setitem__("completion_time_s", math.inf),
    ],
)
def test_verified_selection_wrapper_rejects_forged_partial_and_wrong_types(mutation) -> None:
    rows = verified_rows()
    mutation(rows)
    with pytest.raises(ClassicalV2ContractError):
        select_and_build_admission_plan(config(frozen=True), rows)


def test_heldout_grasp_cannot_enter_tuning_results() -> None:
    rows = verified_rows()
    rows[0]["slot"]["grasp_id"] = "rj45-train-02"
    with pytest.raises(ClassicalV2ContractError, match="slot"):
        select_and_build_admission_plan(config(frozen=True), rows)
