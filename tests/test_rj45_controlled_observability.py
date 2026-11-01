import copy
import json
from pathlib import Path

import numpy as np
import pytest

from rby1_assembly import rj45_controlled_observability as subject


def contract():
    path = Path(__file__).resolve().parents[1] / "configs/rj45-controlled-observability-v2.json"
    value = json.loads(path.read_text())
    value["frozen"] = True
    return value


def test_plan_pairs_all_seeds_without_held_out_access():
    plan = subject.execution_plan(contract())
    assert len(plan) == 24
    assert len({(r["axis"], r["grasp_id"], r["sign"], r["environment_seed"]) for r in plan}) == 24
    assert [r["ordinal"] for r in plan] == list(range(1, 25))
    chunks = {r["chunk"] for r in plan}
    assert len(chunks) == 4
    for chunk in chunks:
        rows = [r for r in plan if r["chunk"] == chunk]
        assert len(rows) == 6
        for seed in subject.environment_seeds():
            assert {r["sign"] for r in rows if r["environment_seed"] == seed} == {"positive", "negative"}


@pytest.mark.parametrize("key,value", [
    ("environment_seeds", [1, 1, 1]), ("primary", "contact_sample_13_observation79"),
    ("held_out_access", True), ("comparison_methods_receive_shared_probe", True),
    ("signal_minimum", 0.01), ("episode_count", True), ("axes", ["tz"]),
])
def test_contract_rejects_changed_measurement_or_scope(key, value):
    candidate = contract()
    candidate[key] = value
    with pytest.raises(ValueError):
        subject.validate_contract(candidate)


def test_unfrozen_contract_is_preflight_only():
    candidate = contract()
    candidate["frozen"] = False
    with pytest.raises(ValueError, match="not frozen"):
        subject.execution_plan(candidate)
    assert len(subject.execution_plan(candidate, require_frozen=False)) == 24
    changed = copy.deepcopy(candidate)
    changed["frozen"] = True
    assert subject.canonical_sha256(candidate) != subject.canonical_sha256(changed)


def test_feature_projection_excludes_pose_action_phase_depth():
    obs = np.zeros(79)
    obs[0:7] = 100
    obs[40:63] = 100
    obs[78] = 100
    assert not subject.feature48(obs).any()
    obs[39] = 0.2
    obs[63] = 0.3
    assert np.isclose(subject.feature48(obs).sum(), 0.5)


def test_deterministic_separation_does_not_claim_robustness_or_admission():
    positive = np.zeros((3, 48))
    negative = positive.copy()
    negative[:, 0] = 0.1
    result = subject.separation_statistics(positive, negative)
    assert result["separation_passed"]
    assert result["dispersion_floor_used"]
    assert not result["stochastic_robustness_demonstrated"]
    assert not result["rl_pilot_admitted"]


def test_worst_seed_and_cross_seed_noise_both_gate_separation():
    positive = np.zeros((3, 48))
    negative = positive.copy()
    negative[:, 0] = [0.1, 0.1, 0.01]
    assert not subject.separation_statistics(positive, negative)["separation_passed"]
    positive[:, 1] = [0, 0.2, 0.4]
    negative = positive.copy()
    negative[:, 0] = 0.1
    result = subject.separation_statistics(positive, negative)
    assert result["signal_minimum"] == 0.1
    assert result["dispersion_raw"] == 0.4
    assert not result["separation_passed"]


@pytest.mark.parametrize("bad", [np.zeros((2, 48)), np.full((3, 48), np.nan), [[True] * 48] * 3])
def test_invalid_samples_rejected(bad):
    with pytest.raises(ValueError):
        subject.separation_statistics(bad, np.zeros((3, 48)))


def test_finite_inputs_with_overflowing_distance_fail_closed():
    with pytest.raises(ValueError, match="distance"):
        subject.separation_statistics(np.full((3, 48), 1e308), np.zeros((3, 48)))


def test_float64_cast_overflow_fails_closed():
    if np.finfo(np.longdouble).max <= np.finfo(np.float64).max:
        return  # Platform has no wider real dtype to exercise conversion overflow.
    with pytest.raises(ValueError, match="float64"):
        subject.feature48(np.full(79, np.finfo(np.longdouble).max, dtype=np.longdouble))
    with pytest.raises(ValueError, match="float64"):
        subject.separation_statistics(
            np.full((3, 48), np.finfo(np.longdouble).max, dtype=np.longdouble), np.zeros((3, 48)),
        )
