import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rby1_assembly.rj45_classical_v2 import (
    CLASSICAL_V2_CANDIDATES,
    ClassicalForceSearchV2Provider,
    select_classical_v2_candidate,
)
from rby1_assembly.rj45_comparison_providers import CLASSICAL_CANDIDATES


ROOT = Path(__file__).resolve().parents[1]


def _observation() -> np.ndarray:
    value = np.zeros(79, dtype=np.float64)
    value[3] = 1.0
    return value


def _force_observation(force_n: float = 0.25) -> np.ndarray:
    value = _observation()
    value[37] = force_n / 8.0
    return value


def _tuning_rows():
    rows = []
    cases = (
        "tx-positive",
        "tx-negative",
        "tz-positive",
        "tz-negative",
        "rx-positive",
        "rx-negative",
    )
    for candidate_index, candidate_id in enumerate(CLASSICAL_V2_CANDIDATES):
        for case in cases:
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "grasp_id": "rj45-train-03",
                    "case_id": case,
                    "success": candidate_index < 2,
                    "hard_contact": False,
                    "forbidden_contact": False,
                    "evaluator_force_p95_n": 0.4 + candidate_index * 0.1,
                    "completion_time_s": 1.0 + candidate_index * 0.1,
                }
            )
    return rows


def test_sealed_candidate_matrix_and_constructor_reject_unregistered_inputs() -> None:
    assert {
        key: (value.search_step_m, value.force_trigger_n)
        for key, value in CLASSICAL_V2_CANDIDATES.items()
    } == {
        "V2A": (0.00005, 0.25),
        "V2B": (0.00010, 0.25),
        "V2C": (0.00005, 0.50),
        "V2D": (0.00010, 0.50),
    }
    assert tuple(inspect.signature(ClassicalForceSearchV2Provider).parameters) == (
        "candidate_id",
    )
    for invalid in ("A", "V2E", "v2a", "", None, 1):
        with pytest.raises(ValueError):
            ClassicalForceSearchV2Provider(invalid)


def test_first_valid_call_preserves_existing_pose_correction_not_forced_zero() -> None:
    provider = ClassicalForceSearchV2Provider("V2A")
    observation = _observation()
    observation[0] = 0.5
    action, state = provider.act(observation, provider.reset())
    assert action[0] == pytest.approx(-0.5e-3 / 1.5e-3)
    assert not provider.state_telemetry(state)["safety_latched"]


def test_search_direction_plateau_is_exactly_eighteen_ticks() -> None:
    provider = ClassicalForceSearchV2Provider("V2A")
    state = provider.reset()
    observation = _force_observation()
    actions = []
    directions = []
    for _ in range(19):
        action, state = provider.act(observation, state)
        actions.append(action)
        directions.append(state.direction_index)
    assert directions[:18] == [0] * 18
    assert directions[18] == 1
    assert all(action[0] < 0.0 and action[2] == 0.0 for action in actions[:18])
    assert actions[18][0] == pytest.approx(0.0)
    assert actions[18][2] < 0.0


def test_four_direction_cycle_increases_radius_and_caps_at_point_six_mm() -> None:
    provider = ClassicalForceSearchV2Provider("V2B")
    state = provider.reset()
    observation = _force_observation()
    action = None
    for tick in range(1, 362):
        action, state = provider.act(observation, state)
        if tick == 72:
            assert state.search_cycles == 0
        if tick == 73:
            assert state.search_cycles == 1
            assert abs(action[0]) == pytest.approx(0.0002 / 0.0015)
    assert state.search_cycles == 5
    assert abs(action[0]) == pytest.approx(0.0006 / 0.0015)


def test_recovery_requires_eighteen_advancing_ticks_before_reset() -> None:
    provider = ClassicalForceSearchV2Provider("V2A")
    state = provider.reset()
    _, state = provider.act(_force_observation(), state)
    advancing = _observation()
    advancing[63] = 1.0
    for expected in range(1, 18):
        _, state = provider.act(advancing, state)
        assert state.recovered_ticks == expected
        assert state.initial_direction_xz is not None
    action, state = provider.act(advancing, state)
    assert state.recovered_ticks == 18
    assert state.initial_direction_xz is None
    assert state.search_ticks == 0
    np.testing.assert_array_equal(action, np.zeros(6))


def test_stalled_branch_starts_search_after_six_ticks_without_causal_claim() -> None:
    provider = ClassicalForceSearchV2Provider("V2C")
    state = provider.reset()
    observation = _observation()
    _, state = provider.act(observation, state)
    for expected in range(1, 6):
        _, state = provider.act(observation, state)
        assert state.stalled_ticks == expected
        assert state.initial_direction_xz is None
    action, state = provider.act(observation, state)
    assert state.stalled_ticks == 6
    assert state.initial_direction_xz == (1.0, 0.0)
    assert action[0] > 0.0


def test_twenty_newton_safety_latch_is_immediate_and_permanent() -> None:
    provider = ClassicalForceSearchV2Provider("V2D")
    stop = _force_observation(20.0)
    action, state = provider.act(stop, provider.reset())
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["safety_latched"] is True

    later = _observation()
    later[0] = 1.0
    action, state = provider.act(later, state)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert state.safety_latched is True


def test_translation_rotation_and_components_remain_bounded() -> None:
    provider = ClassicalForceSearchV2Provider("V2B")
    observation = _observation()
    observation[0] = observation[2] = 100.0
    observation[4] = observation[6] = 100.0
    action, _ = provider.act(observation, provider.reset())
    assert np.linalg.norm(action[:3]) <= 1.0 + 1e-12
    assert np.linalg.norm(action[3:]) <= 1.0 + 1e-12
    assert np.all(np.abs(action) <= 1.0)


def test_provider_is_case_grasp_sign_and_ground_truth_blind() -> None:
    provider = ClassicalForceSearchV2Provider("V2A")
    for attribute in ("case_id", "grasp_id", "sign", "ground_truth"):
        assert not hasattr(provider, attribute)
    with pytest.raises(TypeError):
        provider.act(
            {"observation79": _observation(), "case_id": "tx-positive"},
            provider.reset(),
        )


def test_payload_sha_telemetry_and_base_source_are_stable() -> None:
    provider = ClassicalForceSearchV2Provider("V2A")
    payload = provider.canonical_payload()
    assert payload["schema"] == "rj45-classical-force-search-controller-v2"
    assert payload["search_plateau_ticks"] == 18
    assert payload["search_plateau_s_at_30hz"] == 0.6
    assert payload["recovery_reset_advancing_ticks"] == 18
    assert payload["safety_latch_is_permanent"] is True
    assert payload["translation_bound_m"] == 0.0015
    assert payload["rotation_bound_rad"] == pytest.approx(np.pi / 120.0)
    source = ROOT / "src/rby1_assembly/rj45_comparison_providers.py"
    assert payload["base_helper_source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert provider.controller_sha256() == ClassicalForceSearchV2Provider("V2A").controller_sha256()
    assert len(provider.controller_sha256()) == 64
    json.dumps(payload, allow_nan=False)
    json.dumps(provider.state_telemetry(provider.reset()), allow_nan=False)


def test_v2_config_preregisters_24_slots_selection_and_noncausal_basis() -> None:
    config = json.loads((ROOT / "configs/rj45-classical-tuning-v2.json").read_text())
    assert config["schema"] == "rj45-classical-tuning-v2"
    assert config["tuning_grasp_id"] == "rj45-train-03"
    assert config["candidate_count"] == 4
    assert config["episodes_per_candidate"] == 6
    assert config["tuning_episode_count"] == 24
    for configured in config["candidates"]:
        payload = ClassicalForceSearchV2Provider(configured["id"]).canonical_payload()
        for key in (
            "position_gain",
            "rotation_gain",
            "search_step_m",
            "force_trigger_n",
        ):
            assert configured[key] == payload[key]
    assert config["selection"]["order"] == [
        "safe_eligible_only",
        "maximum_success_count",
        "minimum_evaluator_force_p95_n",
        "minimum_completion_time_median_s",
        "lexicographic_candidate_id",
    ]
    assert config["selection"]["selected_safe_baseline_always_runs_new_admission"]["episode_count"] == 12
    qualifier = config["selection"]["strong_baseline_qualifier"]
    assert (qualifier["minimum_successes"], qualifier["out_of"]) == (4, 6)
    assert qualifier["name_only"] is True
    assert config["design_basis"]["old_search_plateau_s_at_30hz"] == 0.2
    assert config["design_basis"]["success_dwell_s"] == 0.4
    assert config["design_basis"]["observed_diagnostic_force_peak_n"] == 0.55
    assert config["design_basis"]["causal_attribution"] == "not_established"


def test_selection_is_safe_then_success_force_time_and_id() -> None:
    rows = _tuning_rows()
    selected = select_classical_v2_candidate(rows)
    assert selected["candidate_id"] == "V2A"
    assert selected["new_admission_baseline_episode_count"] == 12
    assert selected["strong_baseline_qualifier_met"] is True

    for row in rows:
        if row["candidate_id"] == "V2A":
            row["hard_contact"] = True
    selected = select_classical_v2_candidate(rows)
    assert selected["candidate_id"] == "V2B"


def test_strong_name_qualifier_never_gates_new_safe_baseline() -> None:
    rows = _tuning_rows()
    for row in rows:
        row["success"] = row["case_id"] in ("tx-positive", "tx-negative", "tz-positive")
    selected = select_classical_v2_candidate(rows)
    assert selected["success_count"] == 3
    assert selected["strong_baseline_qualifier_met"] is False
    assert selected["new_admission_baseline_episode_count"] == 12
    assert selected["strong_qualifier_does_not_gate_baseline"] is True


def test_old_abc_registry_and_source_bytes_remain_unchanged() -> None:
    assert {
        key: (
            value.position_gain,
            value.rotation_gain,
            value.force_trigger_n,
            value.search_step_m,
        )
        for key, value in CLASSICAL_CANDIDATES.items()
    } == {
        "A": (0.8, 0.8, 8.0, 0.00015),
        "B": (1.0, 0.8, 6.0, 0.00010),
        "C": (0.6, 1.0, 8.0, 0.00020),
    }
    source = ROOT / "src/rby1_assembly/rj45_comparison_providers.py"
    assert hashlib.sha256(source.read_bytes()).hexdigest() == (
        "88385c010b2cb2be942dcb9fc568330b99ad0c46a6253f258e17aaea4bccf251"
    )


def test_selection_rejects_derived_nonfinite_median():
    rows = _tuning_rows()
    for row in rows:
        row["completion_time_s"] = 1e308
    with pytest.raises(ValueError, match="aggregated"):
        select_classical_v2_candidate(rows)


@pytest.mark.parametrize("field", ["evaluator_force_p95_n", "completion_time_s"])
def test_selection_rejects_negative_physical_metrics(field):
    rows = _tuning_rows()
    rows[0][field] = -1.0
    with pytest.raises(ValueError, match="nonnegative"):
        select_classical_v2_candidate(rows)
