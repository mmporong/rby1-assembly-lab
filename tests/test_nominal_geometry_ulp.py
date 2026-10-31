from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import struct

import pytest

from rby1_assembly.experiment_contracts import _recorded_mating_geometry_matches
from rby1_assembly.rj45_task import verify_recorded_mating_geometry


FIXTURE = Path(__file__).parent / "fixtures/rj45_nominal_geometry_ulp.json"


def _geometry() -> dict:
    return {
        "depth_m": 0.012,
        "lateral_error_m": 0.00004,
        "orientation_error_rad": 0.004,
        "relative_position_entry_m": [0.00001, 0.012, 0.00002],
        "relative_quaternion_entry_wxyz": [1.0, 0.0, 0.0, 0.0],
        "source": "numpy_recalculation_from_recorded_raw_object_poses",
    }


def _next(value: float, direction: float, count: int) -> float:
    for _ in range(count):
        value = math.nextafter(value, direction)
    return value


def _ulp_distance(left: float, right: float) -> int:
    left_bits = struct.unpack(">Q", struct.pack(">d", left))[0]
    right_bits = struct.unpack(">Q", struct.pack(">d", right))[0]
    return abs(left_bits - right_bits)


def test_actual_nominal_row_accepts_one_ulp_lateral_serialization_drift() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    recalculated = verify_recorded_mating_geometry(
        fixture["socket_pose_world_wxyz"],
        fixture["plug_pose_world_wxyz"],
        fixture["socket_entry"],
        fixture["plug_tip"],
        fixture["actual_depth_m"],
        fixture["actual_lateral_error_m"],
        fixture["actual_orientation_error_rad"],
    )
    recorded = fixture["recorded_geometry"]
    assert fixture["provenance"] == {
        "evidence_relative_path": (
            "artifacts/rj45-4c-bank-generation-v2-20261028/"
            "seed-20261211/trace.jsonl"
        ),
        "control_step": 5,
        "environment_index": 0,
    }
    assert _ulp_distance(
        recorded["lateral_error_m"], recalculated["lateral_error_m"]
    ) == 1
    assert _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize("key", ["lateral_error_m", "orientation_error_rad"])
@pytest.mark.parametrize("count", [1, 2])
def test_only_one_or_two_ulps_are_accepted_for_derived_magnitudes(
    key: str, count: int
) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded[key] = _next(recalculated[key], math.inf, count)
    assert _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize("key", ["lateral_error_m", "orientation_error_rad"])
def test_three_ulps_are_rejected(key: str) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded[key] = _next(recalculated[key], math.inf, 3)
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


def test_three_ordered_steps_across_power_of_two_boundary_are_rejected() -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    two_below_one = _next(1.0, -math.inf, 2)
    one_above_one = math.nextafter(1.0, math.inf)
    recalculated["lateral_error_m"] = two_below_one
    recorded["lateral_error_m"] = one_above_one
    assert _ulp_distance(two_below_one, one_above_one) == 3
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


def test_positive_and_negative_zero_are_equal_before_ulp_distance() -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded["lateral_error_m"] = -0.0
    recalculated["lateral_error_m"] = 0.0
    assert _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, -0.001, 1, True, "0.0"])
@pytest.mark.parametrize("key", ["lateral_error_m", "orientation_error_rad"])
def test_invalid_derived_magnitude_values_are_rejected(key: str, value: object) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded[key] = value
    assert not _recorded_mating_geometry_matches(recorded, recalculated)
    recorded = _geometry()
    recalculated[key] = value
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_geometry_key_set_must_be_exact(mutation: str) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    if mutation == "missing":
        recorded.pop("source")
    else:
        recorded["extra"] = 0.0
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize("value", [1, True, None])
def test_source_must_be_the_exact_same_string(value: object) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded["source"] = value
    assert not _recorded_mating_geometry_matches(recorded, recalculated)
    recorded["source"] = "other_source"
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize("value", [math.nan, math.inf, 1, True, "0.012"])
def test_depth_requires_exact_finite_float(value: object) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded["depth_m"] = value
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


def test_one_ulp_depth_drift_is_rejected() -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    recorded["depth_m"] = math.nextafter(recalculated["depth_m"], math.inf)
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


@pytest.mark.parametrize(
    "key,mutation",
    [
        ("relative_position_entry_m", "shape"),
        ("relative_position_entry_m", "tuple"),
        ("relative_position_entry_m", "int"),
        ("relative_position_entry_m", "nan"),
        ("relative_position_entry_m", "ulp"),
        ("relative_quaternion_entry_wxyz", "shape"),
        ("relative_quaternion_entry_wxyz", "tuple"),
        ("relative_quaternion_entry_wxyz", "int"),
        ("relative_quaternion_entry_wxyz", "nan"),
        ("relative_quaternion_entry_wxyz", "ulp"),
    ],
)
def test_pose_arrays_require_exact_list_shape_float_type_and_values(
    key: str, mutation: str
) -> None:
    recalculated = _geometry()
    recorded = copy.deepcopy(recalculated)
    if mutation == "shape":
        recorded[key] = recorded[key][:-1]
    elif mutation == "tuple":
        recorded[key] = tuple(recorded[key])
    elif mutation == "int":
        recorded[key][0] = 1
    elif mutation == "nan":
        recorded[key][0] = math.nan
    else:
        recorded[key][0] = math.nextafter(recorded[key][0], math.inf)
    assert not _recorded_mating_geometry_matches(recorded, recalculated)


def test_two_ulp_recording_drift_cannot_move_success_threshold() -> None:
    limit = 0.00005
    recalculated = _geometry()
    recalculated["lateral_error_m"] = limit
    recorded = copy.deepcopy(recalculated)
    recorded["lateral_error_m"] = _next(limit, math.inf, 2)
    assert recorded["lateral_error_m"] > limit
    assert recalculated["lateral_error_m"] <= limit
    assert _recorded_mating_geometry_matches(recorded, recalculated)
