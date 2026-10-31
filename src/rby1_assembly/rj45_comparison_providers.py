"""CPU-only action providers for the RJ45 comparison admission run.

Every fair provider has the same narrow authority boundary: the current 79-D
actor observation and provider-owned state.  Case identifiers, injected bias,
environment objects, and evaluator ground truth are deliberately absent from
the API.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import acos, pi
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


OBSERVATION_DIM = 79
ACTION_DIM = 6
TRANSLATION_BOUND_M = 0.0015
ROTATION_BOUND_RAD = pi / 120.0
FORCE_SCALE_N = 8.0
DEPTH_SCALE_M = 0.012


class Provider(Protocol):
    def reset(self) -> Any: ...

    def act(self, observation79: Sequence[float], state: Any) -> tuple[np.ndarray, Any]: ...


def _observation(value: Sequence[float]) -> np.ndarray:
    if isinstance(value, Mapping):
        raise TypeError("observation79 must be a numeric vector, not a mapping")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (OBSERVATION_DIM,):
        raise ValueError(f"observation79 must have shape ({OBSERVATION_DIM},)")
    if not np.isfinite(array).all():
        raise ValueError("observation79 must contain only finite values")
    return array


def _normalized_action(value: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
        raise ValueError("normalized_right6 must be a finite 6-D vector")
    if np.any(np.abs(array) > 1.0):
        raise ValueError("normalized_right6 components must be within [-1, 1]")
    return array.copy()


@dataclass(frozen=True)
class StatelessProviderState:
    ticks: int = 0


class ZeroProvider:
    def reset(self) -> StatelessProviderState:
        return StatelessProviderState()

    def act(self, observation79: Sequence[float], state: StatelessProviderState) -> tuple[np.ndarray, StatelessProviderState]:
        _observation(observation79)
        if not isinstance(state, StatelessProviderState):
            raise TypeError("state must come from ZeroProvider.reset")
        return np.zeros(ACTION_DIM, dtype=np.float64), replace(state, ticks=state.ticks + 1)


class FixedConstantProvider:
    def __init__(self, normalized_right6: Sequence[float]) -> None:
        self._action = _normalized_action(normalized_right6)
        self._action.setflags(write=False)

    def reset(self) -> StatelessProviderState:
        return StatelessProviderState()

    def act(self, observation79: Sequence[float], state: StatelessProviderState) -> tuple[np.ndarray, StatelessProviderState]:
        _observation(observation79)
        if not isinstance(state, StatelessProviderState):
            raise TypeError("state must come from FixedConstantProvider.reset")
        return self._action.copy(), replace(state, ticks=state.ticks + 1)


@dataclass(frozen=True)
class PlaybackState:
    index: int = 0


class ImmutablePlaybackProvider:
    """Replay an immutable normalized trace and hold its final action."""

    def __init__(self, trace: Sequence[Sequence[float]]) -> None:
        if not trace:
            raise ValueError("playback trace must contain at least one action")
        self._trace = tuple(tuple(float(x) for x in _normalized_action(row)) for row in trace)

    @property
    def trace(self) -> tuple[tuple[float, ...], ...]:
        return self._trace

    def reset(self) -> PlaybackState:
        return PlaybackState()

    def act(self, observation79: Sequence[float], state: PlaybackState) -> tuple[np.ndarray, PlaybackState]:
        _observation(observation79)
        if not isinstance(state, PlaybackState) or state.index < 0:
            raise TypeError("state must come from ImmutablePlaybackProvider.reset")
        row = self._trace[min(state.index, len(self._trace) - 1)]
        return np.asarray(row, dtype=np.float64), PlaybackState(state.index + 1)


@dataclass(frozen=True)
class ClassicalCandidate:
    candidate_id: str
    position_gain: float
    rotation_gain: float
    force_trigger_n: float
    search_step_m: float


CLASSICAL_CANDIDATES: Mapping[str, ClassicalCandidate] = MappingProxyType(
    {
        "A": ClassicalCandidate("A", 0.8, 0.8, 8.0, 0.00015),
        "B": ClassicalCandidate("B", 1.0, 0.8, 6.0, 0.00010),
        "C": ClassicalCandidate("C", 0.6, 1.0, 8.0, 0.00020),
    }
)


@dataclass(frozen=True)
class ClassicalForceSearchState:
    previous_depth_m: float | None = None
    stalled_ticks: int = 0
    recovered_ticks: int = 0
    initial_direction_xz: tuple[float, float] | None = None
    direction_index: int = 0
    search_cycles: int = 0
    search_ticks: int = 0


def _quaternion_rotation_vector(wxyz: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(wxyz))
    if norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    q = wxyz / norm
    if q[0] < 0.0:
        q = -q
    vector_norm = float(np.linalg.norm(q[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * acos(float(np.clip(q[0], -1.0, 1.0)))
    return q[1:] * (angle / vector_norm)


def _project(vector: np.ndarray, bound: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > bound:
        vector = vector * (bound / norm)
    return vector


def _initial_search_direction(force_xz: np.ndarray) -> tuple[float, float]:
    if float(np.linalg.norm(force_xz)) < 0.25:
        return (1.0, 0.0)
    axis = int(np.argmax(np.abs(force_xz)))
    direction = np.zeros(2, dtype=np.float64)
    direction[axis] = -1.0 if force_xz[axis] > 0.0 else 1.0
    return (float(direction[0]), float(direction[1]))


def _rotate_direction(direction: tuple[float, float], index: int) -> np.ndarray:
    x, z = direction
    return (
        np.asarray((x, z)),
        np.asarray((-z, x)),
        np.asarray((-x, -z)),
        np.asarray((z, -x)),
    )[index % 4]


class ClassicalForceSearchProvider:
    """Numerically fixed pose correction plus force-triggered X/Z search."""

    def __init__(self, candidate: ClassicalCandidate = CLASSICAL_CANDIDATES["A"]) -> None:
        if candidate.candidate_id not in CLASSICAL_CANDIDATES or CLASSICAL_CANDIDATES[candidate.candidate_id] != candidate:
            raise ValueError("candidate must be one of the sealed A/B/C definitions")
        self.candidate = candidate

    def reset(self) -> ClassicalForceSearchState:
        return ClassicalForceSearchState()

    def act(
        self, observation79: Sequence[float], state: ClassicalForceSearchState
    ) -> tuple[np.ndarray, ClassicalForceSearchState]:
        obs = _observation(observation79)
        if not isinstance(state, ClassicalForceSearchState):
            raise TypeError("state must come from ClassicalForceSearchProvider.reset")

        position_m = obs[0:3] * np.asarray((0.001, 0.012, 0.001))
        quaternion = np.concatenate(([obs[3]], obs[4:7] * ((pi / 180.0) / 2.0)))
        rotation_rad = _quaternion_rotation_vector(quaternion)
        force_n = obs[37:40] * FORCE_SCALE_N
        force_norm_n = float(np.linalg.norm(force_n))
        depth_m = float(obs[62] * DEPTH_SCALE_M)
        advance = float(obs[63])

        if force_norm_n >= 20.0:
            return np.zeros(ACTION_DIM, dtype=np.float64), replace(state, previous_depth_m=depth_m)

        depth_delta = None if state.previous_depth_m is None else depth_m - state.previous_depth_m
        force_triggered = force_norm_n >= self.candidate.force_trigger_n
        stalled = depth_delta is not None and advance == 0.0 and depth_delta < 0.00002
        stalled_ticks = state.stalled_ticks + 1 if stalled else 0
        recovered_ticks = state.recovered_ticks + 1 if advance > 0.0 else 0

        initial = state.initial_direction_xz
        direction_index = state.direction_index
        search_cycles = state.search_cycles
        search_ticks = state.search_ticks
        if recovered_ticks >= 3:
            stalled_ticks = 0
            initial = None
            direction_index = 0
            search_cycles = 0
            search_ticks = 0
        elif force_triggered or stalled_ticks >= 6 or initial is not None:
            if initial is None:
                initial = _initial_search_direction(force_n[[0, 2]])
            search_ticks += 1
            direction_index = ((search_ticks - 1) // 6) % 4
            search_cycles = (search_ticks - 1) // 24

        translation = np.asarray(
            (-self.candidate.position_gain * position_m[0], 0.0, -self.candidate.position_gain * position_m[2]),
            dtype=np.float64,
        )
        rotation = np.asarray(
            (-self.candidate.rotation_gain * rotation_rad[0], 0.0, -self.candidate.rotation_gain * rotation_rad[2]),
            dtype=np.float64,
        )
        if initial is not None and search_ticks > 0:
            radius = min((search_cycles + 1) * self.candidate.search_step_m, 0.0006)
            direction = _rotate_direction(initial, direction_index)
            translation[[0, 2]] += radius * direction

        translation = _project(translation, TRANSLATION_BOUND_M)
        rotation = _project(rotation, ROTATION_BOUND_RAD)
        action = np.concatenate((translation / TRANSLATION_BOUND_M, rotation / ROTATION_BOUND_RAD))
        action = np.clip(action, -1.0, 1.0)
        next_state = ClassicalForceSearchState(
            previous_depth_m=depth_m,
            stalled_ticks=stalled_ticks,
            recovered_ticks=recovered_ticks,
            initial_direction_xz=initial,
            direction_index=direction_index,
            search_cycles=search_cycles,
            search_ticks=search_ticks,
        )
        return action, next_state


def make_fair_provider(
    provider_id: str,
    *,
    constant_action: Sequence[float] | None = None,
    playback_trace: Sequence[Sequence[float]] | None = None,
    classical_candidate_id: str = "A",
) -> Provider:
    """Build only non-privileged providers; oracle IDs fail closed."""

    if provider_id == "zero":
        return ZeroProvider()
    if provider_id == "mean_constant":
        if constant_action is None:
            raise ValueError("mean_constant requires constant_action")
        return FixedConstantProvider(constant_action)
    if provider_id == "open_loop_playback":
        if playback_trace is None:
            raise ValueError("open_loop_playback requires playback_trace")
        return ImmutablePlaybackProvider(playback_trace)
    if provider_id == "classical_force_search":
        try:
            candidate = CLASSICAL_CANDIDATES[classical_candidate_id]
        except KeyError as exc:
            raise ValueError("classical_candidate_id must be A, B, or C") from exc
        return ClassicalForceSearchProvider(candidate)
    raise ValueError(f"provider {provider_id!r} is not in the fair registry")
