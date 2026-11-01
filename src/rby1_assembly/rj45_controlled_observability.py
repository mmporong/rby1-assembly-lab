"""Prospective controlled-observation contract and pure separation statistics.

This module reports observation evidence only. It cannot grant RL admission.
Raw trace replay and provenance verification remain runner responsibilities.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Mapping, Sequence

import numpy as np


SEED_NAMESPACE = "rj45-controlled-observability-v2/environment"
GRASPS = ("rj45-train-02", "rj45-train-05")
AXES = ("tx", "tz")
SIGNS = ("positive", "negative")
FEATURE_INDICES = tuple(range(7, 40)) + tuple(range(63, 78))


def environment_seeds() -> tuple[int, ...]:
    return tuple(
        int.from_bytes(hashlib.sha256(f"{SEED_NAMESPACE}/{i}".encode()).digest()[:4], "big")
        & 0x7FFFFFFF
        for i in range(3)
    )


def canonical_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def validate_contract(config: Mapping[str, Any], *, require_frozen: bool = True) -> str:
    expected = {
        "schema": "rj45-controlled-observability-contract-v2",
        "seed_namespace": SEED_NAMESPACE,
        "environment_seeds": list(environment_seeds()),
        "grasps": list(GRASPS), "axes": list(AXES), "signs": list(SIGNS),
        "episode_count": 24, "chunk_size": 6, "translation_bias_m": 0.00012,
        "primary": "contact_sample_13_feature48",
        "feature_indices": [[7, 40], [63, 78]],
        "signal_minimum": 0.05, "signal_to_dispersion_minimum": 2.0,
        "dispersion_floor": 1e-6,
        "dispersion_statistic": "maximum_cross_seed_same_sign_pairwise_distance",
        "signal_statistic": "minimum_paired_seed_opposite_sign_distance",
        "contact_threshold_n": 0.25, "contact_window_observations": 13,
        "secondary": ["contact_sample_13_observation79", "contact_window_median_feature48",
                      "first_observation_after_zero_return"],
        "provider_inputs": ["observation79", "provider_owned_state"],
        "comparison_methods_receive_shared_probe": False,
        "held_out_access": False,
    }
    if not isinstance(config, Mapping) or set(config) != set(expected) | {"frozen"}:
        raise ValueError("controlled-observation contract has missing or unknown fields")
    if type(config["frozen"]) is not bool or (require_frozen and not config["frozen"]):
        raise ValueError("controlled-observation contract is not frozen")
    for key, value in expected.items():
        # Canonical comparison rejects bool/int substitutions as well as changed values.
        if json.dumps(config[key], sort_keys=True) != json.dumps(value, sort_keys=True):
            raise ValueError(f"controlled-observation contract differs at {key}")
    return canonical_sha256(config)


def execution_plan(config: Mapping[str, Any], *, require_frozen: bool = True) -> list[dict[str, Any]]:
    validate_contract(config, require_frozen=require_frozen)
    result = []
    for axis, grasp in itertools.product(AXES, GRASPS):
        for sign, seed in itertools.product(SIGNS, environment_seeds()):
            result.append({
                "ordinal": len(result) + 1,
                "chunk": f"{axis}-{grasp}",
                "axis": axis, "grasp_id": grasp, "case_id": f"{axis}-{sign}",
                "sign": sign, "environment_seed": seed,
            })
    return result


def feature48(observation: Sequence[float]) -> np.ndarray:
    value = np.asarray(observation)
    if value.shape != (79,) or value.dtype.kind not in "iuf" or not np.isfinite(value).all():
        raise ValueError("observation must be a finite real vector of length 79")
    with np.errstate(over="ignore", invalid="ignore"):
        result = value[list(FEATURE_INDICES)].astype(np.float64)
    if not np.isfinite(result).all():
        raise ValueError("observation exceeds finite float64 range")
    return result


def separation_statistics(positive: Sequence[Sequence[float]], negative: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compare three paired seeds; caller must bind vectors to replayed sample 13."""
    arrays = []
    for value in (positive, negative):
        array = np.asarray(value)
        if array.shape != (3, 48) or array.dtype.kind not in "iuf" or not np.isfinite(array).all():
            raise ValueError("each sign requires three finite 48-D samples in registered seed order")
        with np.errstate(over="ignore", invalid="ignore"):
            converted = array.astype(np.float64)
        if not np.isfinite(converted).all():
            raise ValueError("samples exceed finite float64 range")
        arrays.append(converted)
    pos, neg = arrays
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(pos - neg, axis=1)
        within_sign = [float(np.linalg.norm(group[i] - group[j]))
                       for group in arrays for i, j in itertools.combinations(range(3), 2)]
    if not np.isfinite(distances).all() or not np.isfinite(within_sign).all():
        raise ValueError("separation distance exceeds finite float64 range")
    dispersion_raw = max(within_sign)
    dispersion = max(dispersion_raw, 1e-6)
    signal = float(distances.min())
    ratio = signal / dispersion
    if not np.isfinite(ratio):
        raise ValueError("separation ratio exceeds finite float64 range")
    return {
        "paired_seed_distances": distances.tolist(),
        "signal_minimum": signal, "dispersion_raw": dispersion_raw, "dispersion_with_floor": dispersion,
        "dispersion_floor_used": dispersion_raw < 1e-6,
        "signal_to_dispersion": ratio,
        "separation_passed": signal >= 0.05 and ratio >= 2.0,
        "stochastic_robustness_demonstrated": False,
        "rl_pilot_admitted": False,
    }
