import json
from pathlib import Path

import pytest

from rby1_assembly.rj45_force_slope_observability import execution_plan, validate_contract


CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs/rj45-force-slope-observability-v3.json"


def _config():
    value = json.loads(CONFIG_PATH.read_text())
    value["frozen"] = False
    return value


def test_draft_cannot_authorize_execution_and_freeze_changes_digest():
    draft = _config()
    with pytest.raises(ValueError, match="not frozen"):
        execution_plan(draft, "development")
    draft_digest = validate_contract(draft, require_frozen=False)
    frozen = dict(draft, frozen=True)
    assert validate_contract(frozen) != draft_digest
    assert len(execution_plan(frozen, "development")) == 6
    assert len(execution_plan(frozen, "confirmation")) == 24


@pytest.mark.parametrize("field,value", [
    ("frozen", 1),
    ("environment_seeds", [494541735, 903551052, 1615423707]),
    ("environment_seeds", [494541735, 494541735, 1615423706]),
    ("development_grasps", ["rj45-dev-01"]),
    ("development_grasps", ["rj45-train-02"]),
    ("confirmation_grasps", ["rj45-train-03", "rj45-train-05"]),
    ("confirmation_axes", ["tx"]),
    ("signal_minimum", 0.046),
    ("signal_to_dispersion_minimum", 1.0),
    ("primary", "best_contact_sample_feature48"),
    ("impulse_response_count", 3),
    ("impulse_response_count", 4.0),
    ("decision_zero_ticks", True),
    ("direction_rule", "positive_slope_positive_x_otherwise_negative_x"),
    ("contact_trigger", "force_norm"),
    ("safety_threshold_n", 25.0),
    ("maximum_development_attempts", 2),
    ("comparison_methods_receive_shared_probe", True),
    ("held_out_access", True),
    ("development_can_grant_task_or_rl_admission", True),
])
def test_changed_contract_field_is_rejected(field, value):
    config = _config()
    config[field] = value
    with pytest.raises(ValueError):
        validate_contract(config, require_frozen=False)


def test_unknown_and_missing_fields_rejected():
    config = _config()
    with pytest.raises(ValueError, match="unknown"):
        validate_contract(dict(config, candidate2={}), require_frozen=False)
    config.pop("contact_trigger")
    with pytest.raises(ValueError, match="missing"):
        validate_contract(config, require_frozen=False)


def test_development_has_exact_signed_seeds_and_untouched_grasp():
    rows = execution_plan(_config(), "development", require_frozen=False)
    assert [(r["ordinal"], r["case_id"], r["environment_seed"]) for r in rows] == [
        (1, "tx-positive", 494541735), (2, "tx-positive", 903551052),
        (3, "tx-positive", 1615423706), (4, "tx-negative", 494541735),
        (5, "tx-negative", 903551052), (6, "tx-negative", 1615423706),
    ]
    assert all(r["grasp_id"] == "rj45-train-03" and r["provider_id"] == "force_slope_probe" for r in rows)
    assert not any("admitted" in key for r in rows for key in r)


def test_confirmation_does_not_reuse_development_rows_or_change_z_provider():
    config = _config()
    dev = execution_plan(config, "development", require_frozen=False)
    rows = execution_plan(config, "confirmation", require_frozen=False)
    assert [r["ordinal"] for r in rows] == list(range(1, 25))
    assert {r["grasp_id"] for r in rows}.isdisjoint({r["grasp_id"] for r in dev})
    assert {r["chunk"] for r in rows} == {
        "tx-rj45-train-02", "tx-rj45-train-05", "tz-rj45-train-02", "tz-rj45-train-05",
    }
    assert all(r["provider_id"] == ("force_slope_probe" if r["axis"] == "tx" else "active_axis_probe") for r in rows)
    rows[0]["environment_seed"] = 0
    assert execution_plan(config, "confirmation", require_frozen=False)[0]["environment_seed"] == 494541735


@pytest.mark.parametrize("stage", ["final", "dev", "train", "Development", None, True, 1])
def test_only_named_nonfinal_stages_exist(stage):
    with pytest.raises(ValueError, match="stage"):
        execution_plan(_config(), stage, require_frozen=False)
