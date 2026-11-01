"""Sealed classical force-search v2 candidates and pure tuning selection."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import median
import sys
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


_BASE_PATH = Path(__file__).with_name("rj45_comparison_providers.py").resolve()
_BASE_SOURCE_SHA256 = hashlib.sha256(_BASE_PATH.read_bytes()).hexdigest()
_BASE_MODULE_NAME = f"rby1_classical_v2_base_{_BASE_SOURCE_SHA256[:16]}"
_base_spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PATH)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"cannot load classical base helpers: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_base_spec)
sys.modules[_BASE_MODULE_NAME] = _base
_base_spec.loader.exec_module(_base)


POSITION_GAIN = 1.0
ROTATION_GAIN = 0.8
SEARCH_PLATEAU_TICKS = 18
RECOVERY_RESET_ADVANCING_TICKS = 18
STALLED_TRIGGER_TICKS = 6
STALL_DEPTH_DELTA_M = 0.00002
SEARCH_RADIUS_MAX_M = 0.0006
SAFETY_FORCE_N = 20.0
TRANSLATION_BOUND_M = 0.0015
ROTATION_BOUND_RAD = np.pi / 120.0
CONTROL_HZ = 30
TUNING_GRASP_ID = "rj45-train-03"
SIGNED_CASES = (
    "tx-positive",
    "tx-negative",
    "tz-positive",
    "tz-negative",
    "rx-positive",
    "rx-negative",
)


@dataclass(frozen=True)
class ClassicalV2Candidate:
    candidate_id: str
    search_step_m: float
    force_trigger_n: float


CLASSICAL_V2_CANDIDATES: Mapping[str, ClassicalV2Candidate] = MappingProxyType(
    {
        "V2A": ClassicalV2Candidate("V2A", 0.00005, 0.25),
        "V2B": ClassicalV2Candidate("V2B", 0.00010, 0.25),
        "V2C": ClassicalV2Candidate("V2C", 0.00005, 0.50),
        "V2D": ClassicalV2Candidate("V2D", 0.00010, 0.50),
    }
)


@dataclass(frozen=True)
class ClassicalV2State:
    previous_depth_m: float | None = None
    stalled_ticks: int = 0
    recovered_ticks: int = 0
    initial_direction_xz: tuple[float, float] | None = None
    direction_index: int = 0
    search_cycles: int = 0
    search_ticks: int = 0
    safety_latched: bool = False


class ClassicalForceSearchV2Provider:
    """Existing pose correction and spiral search with sealed v2 timing."""

    def __init__(self, candidate_id: str) -> None:
        if not isinstance(candidate_id, str) or candidate_id not in CLASSICAL_V2_CANDIDATES:
            raise ValueError("candidate_id must be exactly V2A, V2B, V2C, or V2D")
        self._candidate = CLASSICAL_V2_CANDIDATES[candidate_id]

    @property
    def candidate_id(self) -> str:
        return self._candidate.candidate_id

    def reset(self) -> ClassicalV2State:
        return ClassicalV2State()

    def act(
        self, observation79: Sequence[float], state: ClassicalV2State
    ) -> tuple[np.ndarray, ClassicalV2State]:
        observation = _base._observation(observation79)
        if not isinstance(state, ClassicalV2State):
            raise TypeError("state must come from ClassicalForceSearchV2Provider.reset")

        position_m = observation[0:3] * np.asarray((0.001, 0.012, 0.001))
        quaternion = np.concatenate(
            ([observation[3]], observation[4:7] * ((np.pi / 180.0) / 2.0))
        )
        rotation_rad = _base._quaternion_rotation_vector(quaternion)
        force_n = observation[37:40] * _base.FORCE_SCALE_N
        force_norm_n = float(np.linalg.norm(force_n))
        depth_m = float(observation[62] * _base.DEPTH_SCALE_M)
        advance = float(observation[63])

        if state.safety_latched or force_norm_n >= SAFETY_FORCE_N:
            return np.zeros(_base.ACTION_DIM, dtype=np.float64), ClassicalV2State(
                previous_depth_m=depth_m,
                stalled_ticks=state.stalled_ticks,
                recovered_ticks=state.recovered_ticks,
                initial_direction_xz=state.initial_direction_xz,
                direction_index=state.direction_index,
                search_cycles=state.search_cycles,
                search_ticks=state.search_ticks,
                safety_latched=True,
            )

        depth_delta = (
            None if state.previous_depth_m is None else depth_m - state.previous_depth_m
        )
        force_triggered = force_norm_n >= self._candidate.force_trigger_n
        stalled = depth_delta is not None and advance == 0.0 and depth_delta < STALL_DEPTH_DELTA_M
        stalled_ticks = state.stalled_ticks + 1 if stalled else 0
        recovered_ticks = state.recovered_ticks + 1 if advance > 0.0 else 0
        initial = state.initial_direction_xz
        direction_index = state.direction_index
        search_cycles = state.search_cycles
        search_ticks = state.search_ticks

        if recovered_ticks >= RECOVERY_RESET_ADVANCING_TICKS:
            stalled_ticks = 0
            initial = None
            direction_index = 0
            search_cycles = 0
            search_ticks = 0
        elif force_triggered or stalled_ticks >= STALLED_TRIGGER_TICKS or initial is not None:
            if initial is None:
                initial = _base._initial_search_direction(force_n[[0, 2]])
            search_ticks += 1
            direction_index = ((search_ticks - 1) // SEARCH_PLATEAU_TICKS) % 4
            search_cycles = (search_ticks - 1) // (SEARCH_PLATEAU_TICKS * 4)

        translation = np.asarray(
            (-POSITION_GAIN * position_m[0], 0.0, -POSITION_GAIN * position_m[2]),
            dtype=np.float64,
        )
        rotation = np.asarray(
            (-ROTATION_GAIN * rotation_rad[0], 0.0, -ROTATION_GAIN * rotation_rad[2]),
            dtype=np.float64,
        )
        if initial is not None and search_ticks > 0:
            radius_m = min(
                (search_cycles + 1) * self._candidate.search_step_m,
                SEARCH_RADIUS_MAX_M,
            )
            direction = _base._rotate_direction(initial, direction_index)
            translation[[0, 2]] += radius_m * direction

        translation = _base._project(translation, TRANSLATION_BOUND_M)
        rotation = _base._project(rotation, ROTATION_BOUND_RAD)
        action = np.concatenate(
            (translation / TRANSLATION_BOUND_M, rotation / ROTATION_BOUND_RAD)
        )
        action = np.clip(action, -1.0, 1.0)
        next_state = ClassicalV2State(
            previous_depth_m=depth_m,
            stalled_ticks=stalled_ticks,
            recovered_ticks=recovered_ticks,
            initial_direction_xz=initial,
            direction_index=direction_index,
            search_cycles=search_cycles,
            search_ticks=search_ticks,
            safety_latched=False,
        )
        return action, next_state

    def state_telemetry(self, state: ClassicalV2State) -> dict[str, Any]:
        if not isinstance(state, ClassicalV2State):
            raise TypeError("state must come from ClassicalForceSearchV2Provider.reset")
        return {
            "previous_depth_m": state.previous_depth_m,
            "stalled_ticks": state.stalled_ticks,
            "recovered_ticks": state.recovered_ticks,
            "initial_direction_xz": (
                list(state.initial_direction_xz)
                if state.initial_direction_xz is not None
                else None
            ),
            "direction_index": state.direction_index,
            "search_cycles": state.search_cycles,
            "search_ticks": state.search_ticks,
            "safety_latched": state.safety_latched,
        }

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": "rj45-classical-force-search-controller-v2",
            "candidate_id": self._candidate.candidate_id,
            "position_gain": POSITION_GAIN,
            "rotation_gain": ROTATION_GAIN,
            "force_trigger_n": self._candidate.force_trigger_n,
            "search_step_m": self._candidate.search_step_m,
            "search_plateau_ticks": SEARCH_PLATEAU_TICKS,
            "search_plateau_s_at_30hz": SEARCH_PLATEAU_TICKS / CONTROL_HZ,
            "recovery_reset_advancing_ticks": RECOVERY_RESET_ADVANCING_TICKS,
            "stalled_trigger_ticks": STALLED_TRIGGER_TICKS,
            "stall_depth_delta_m_per_tick": STALL_DEPTH_DELTA_M,
            "search_direction_order": [
                "initial",
                "rotate_plus_90",
                "opposite",
                "rotate_minus_90",
            ],
            "search_radius_max_m": SEARCH_RADIUS_MAX_M,
            "safety_force_n": SAFETY_FORCE_N,
            "safety_latch_is_permanent": True,
            "translation_bound_m": TRANSLATION_BOUND_M,
            "rotation_bound_rad": ROTATION_BOUND_RAD,
            "controlled_translation_axes": ["x", "z"],
            "controlled_rotation_axes": ["x", "z"],
            "lateral_force_direction_threshold_n": 0.25,
            "final_component_clip": [-1.0, 1.0],
            "observation_dim": _base.OBSERVATION_DIM,
            "action_dim": _base.ACTION_DIM,
            "entry_force_slice": [37, 40],
            "entry_force_scale_n": _base.FORCE_SCALE_N,
            "command_depth_index": 62,
            "command_depth_scale_m": _base.DEPTH_SCALE_M,
            "advance_index": 63,
            "known_bias_access": False,
            "case_grasp_sign_ground_truth_access": False,
            "base_helper_source_sha256": _BASE_SOURCE_SHA256,
        }

    def controller_sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def select_classical_v2_candidate(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select the best safe candidate without gating its new admission baseline."""

    expected = {
        (candidate_id, case_id)
        for candidate_id in CLASSICAL_V2_CANDIDATES
        for case_id in SIGNED_CASES
    }
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise ValueError("results must be a sequence")
    keys = []
    for row in results:
        if not isinstance(row, Mapping):
            raise ValueError("each tuning result must be an object")
        keys.append((row.get("candidate_id"), row.get("case_id")))
    if len(results) != 24 or len(set(keys)) != 24 or set(keys) != expected:
        raise ValueError("tuning requires exactly 24 sealed candidate/case slots")

    eligible = []
    for candidate_id in CLASSICAL_V2_CANDIDATES:
        rows = [row for row in results if row["candidate_id"] == candidate_id]
        if any(row.get("grasp_id") != TUNING_GRASP_ID for row in rows):
            raise ValueError("v2 tuning is restricted to rj45-train-03")
        for row in rows:
            if type(row.get("success")) is not bool:
                raise ValueError("success must be boolean")
            if type(row.get("hard_contact")) is not bool or type(row.get("forbidden_contact")) is not bool:
                raise ValueError("contact outcomes must be boolean")
        forces = np.asarray([row.get("evaluator_force_p95_n") for row in rows])
        times = np.asarray([row.get("completion_time_s") for row in rows])
        if forces.shape != (6,) or times.shape != (6,) or forces.dtype.kind not in "iuf" or times.dtype.kind not in "iuf":
            raise ValueError("force and time metrics must be numeric")
        forces = forces.astype(np.float64)
        times = times.astype(np.float64)
        if not np.isfinite(forces).all() or not np.isfinite(times).all():
            raise ValueError("force and time metrics must be finite")
        if np.any(forces < 0) or np.any(times < 0):
            raise ValueError("force and time metrics must be nonnegative")
        if any(row["hard_contact"] or row["forbidden_contact"] for row in rows):
            continue
        success_count = sum(row["success"] for row in rows)
        force_p95 = float(np.percentile(forces, 95, method="linear"))
        time_median = float(median(times.tolist()))
        if not np.isfinite(force_p95) or not np.isfinite(time_median):
            raise ValueError("aggregated force and time metrics must be finite")
        eligible.append(
            {
                "candidate_id": candidate_id,
                "success_count": success_count,
                "evaluator_force_p95_n": force_p95,
                "completion_time_median_s": time_median,
                "strong_baseline_qualifier_met": success_count >= 4,
            }
        )
    if not eligible:
        raise ValueError("no safe v2 tuning candidate")
    selected = min(
        eligible,
        key=lambda row: (
            -row["success_count"],
            row["evaluator_force_p95_n"],
            row["completion_time_median_s"],
            row["candidate_id"],
        ),
    )
    return {
        **selected,
        "safe": True,
        "new_admission_baseline_episode_count": 12,
        "strong_qualifier_does_not_gate_baseline": True,
    }


__all__ = [
    "CLASSICAL_V2_CANDIDATES",
    "ClassicalForceSearchV2Provider",
    "ClassicalV2Candidate",
    "ClassicalV2State",
    "select_classical_v2_candidate",
]
