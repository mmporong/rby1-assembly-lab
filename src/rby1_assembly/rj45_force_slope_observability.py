"""Prospective force-slope development and confirmation plans; no admission authority."""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Mapping


_REGISTERED_CONTRACT = {'schema': 'rj45-force-slope-observability-contract-v3',
 'development_grasps': ['rj45-train-03'],
 'development_axes': ['tx'],
 'development_episode_count': 6,
 'maximum_development_attempts': 1,
 'confirmation_grasps': ['rj45-train-02', 'rj45-train-05'],
 'confirmation_axes': ['tx', 'tz'],
 'confirmation_episode_count': 24,
 'signs': ['positive', 'negative'],
 'environment_seeds': [494541735, 903551052, 1615423706],
 'seed_source': 'unchanged_rj45_controlled_observability_v2_environment_seeds',
 'translation_bias_m': 0.00012,
 'tx_provider_schema': 'rj45-force-slope-probe-controller-v3',
 'tz_provider_schema': 'rj45-active-axis-probe-controller-v2',
 'impulse_offset_m': 0.00015,
 'impulse_response_count': 4,
 'slope_statistic': 'abs_entry_force_x_response4_minus_response1',
 'direction_rule': 'positive_slope_positive_x_negative_slope_negative_x_zero_slope_invalid',
 'decision_zero_ticks': 1,
 'search_levels_m': [0.00015, 0.0003, 0.00045, 0.0006],
 'search_dwell_ticks': 6,
 'contact_trigger': 'minus_selected_direction_times_entry_force_x',
 'contact_threshold_n': 0.25,
 'primary': 'contact_sample_13_feature48',
 'feature_indices': [[7, 40], [63, 78]],
 'contact_window_observations': 13,
 'signal_minimum': 0.05,
 'signal_to_dispersion_minimum': 2.0,
 'dispersion_floor': 1e-06,
 'safety_threshold_n': 20.0,
 'return_step_m': 0.00015,
 'translation_bound_m': 0.0015,
 'provider_inputs': ['observation79', 'provider_owned_state'],
 'comparison_methods_receive_shared_probe': False,
 'held_out_access': False,
 'development_pass_required_before_confirmation': True,
 'development_can_grant_task_or_rl_admission': False}


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def validate_contract(config: Mapping[str, Any], *, require_frozen: bool = True) -> str:
    if not isinstance(config, Mapping) or set(config) != set(_REGISTERED_CONTRACT) | {"frozen"}:
        raise ValueError("force-slope contract has missing or unknown fields")
    if type(config["frozen"]) is not bool or (require_frozen and not config["frozen"]):
        raise ValueError("force-slope contract is not frozen")
    for key, expected in _REGISTERED_CONTRACT.items():
        if json.dumps(config[key], sort_keys=True, allow_nan=False) != json.dumps(
            expected, sort_keys=True, allow_nan=False
        ):
            raise ValueError(f"force-slope contract differs at {key}")
    return canonical_sha256(config)


def execution_plan(
    config: Mapping[str, Any], stage: str, *, require_frozen: bool = True,
) -> list[dict[str, Any]]:
    """Return a fixed schedule only; this function cannot authorize any execution."""
    validate_contract(config, require_frozen=require_frozen)
    if type(stage) is not str or stage not in {"development", "confirmation"}:
        raise ValueError("stage must be development or confirmation")
    result = []
    for axis, grasp in itertools.product(config[stage + "_axes"], config[stage + "_grasps"]):
        for sign, seed in itertools.product(config["signs"], config["environment_seeds"]):
            result.append({
                "stage": stage,
                "ordinal": len(result) + 1,
                "chunk": f"{axis}-{grasp}",
                "axis": axis,
                "grasp_id": grasp,
                "case_id": f"{axis}-{sign}",
                "sign": sign,
                "environment_seed": seed,
                "provider_id": "force_slope_probe" if axis == "tx" else "active_axis_probe",
            })
    return result
