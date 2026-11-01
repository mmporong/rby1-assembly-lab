"""Case-blind, CPU-only active Z probe for the RJ45 diagnostic run."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


OBSERVATION_DIM = 79
ACTION_DIM = 6
INSERT_PHASE_OBSERVATION_INDEX = 46
FORCE_OBSERVATION_START_INDEX = 37
FORCE_OBSERVATION_STOP_INDEX = 40
COMMAND_DEPTH_OBSERVATION_INDEX = 62
POLICY_ACTIVE_OBSERVATION_INDEX = 78
ACTION_Z_INDEX = 2

FORCE_SCALE_N = 8.0
COMMAND_DEPTH_SCALE_M = 0.012
COMMAND_DEPTH_ONSET_M = 0.0013
CONTACT_THRESHOLD_N = 0.25
SAFETY_THRESHOLD_N = 20.0
ACTION_OFFSET_SCALE_M = 0.0015
PROBE_STEP_M = 0.00015
PROBE_PEAK_M = 0.00060
PROBE_DWELL_TICKS = 6
DIRECTION_CHANGE_ZERO_TICKS = 1
CONTACT_WINDOW_OBSERVATIONS = 13
RETURN_STEP_M = 0.00015
WAVEFORM_REPEAT_COUNT = 1
DIAGNOSTIC_EPISODE_REPEAT_COUNT = 2


def _direction_waveform(sign: float) -> tuple[float, ...]:
    dwell = tuple(
        sign * offset_m
        for offset_m in (0.00015, 0.00030, 0.00045, 0.00060)
        for _ in range(PROBE_DWELL_TICKS)
    )
    return dwell + tuple(sign * offset_m for offset_m in (0.00045, 0.00030, 0.00015, 0.0))


_POSITIVE_WAVEFORM_M = _direction_waveform(1.0)
_NEGATIVE_WAVEFORM_M = _direction_waveform(-1.0)
_SINGLE_REPEAT_WAVEFORM_M = (
    _POSITIVE_WAVEFORM_M + (0.0,) * DIRECTION_CHANGE_ZERO_TICKS + _NEGATIVE_WAVEFORM_M
)
_WAVEFORM_M = _SINGLE_REPEAT_WAVEFORM_M * WAVEFORM_REPEAT_COUNT


def _observation(value: Sequence[float]) -> np.ndarray:
    if isinstance(value, Mapping):
        raise TypeError("observation79 must be a numeric vector, not a mapping")
    raw = np.asarray(value)
    if raw.shape != (OBSERVATION_DIM,):
        raise ValueError(f"observation79 must have shape ({OBSERVATION_DIM},)")
    if raw.dtype.kind not in "iuf":
        raise TypeError("observation79 must be a real numeric vector")
    observation = raw.astype(np.float64, copy=False)
    if not np.isfinite(observation).all():
        raise ValueError("observation79 must contain only finite values")
    return observation


def _zero_action() -> np.ndarray:
    return np.zeros(ACTION_DIM, dtype=np.float64)


def _action_for_offset(offset_m: float) -> np.ndarray:
    action = _zero_action()
    action[ACTION_Z_INDEX] = offset_m / ACTION_OFFSET_SCALE_M
    return action


@dataclass(frozen=True)
class ActiveZProbeState:
    """Provider-owned state; every field is exposed as JSON-safe telemetry."""

    phase: str = "waiting"
    waveform_index: int = 0
    last_commanded_offset_m: float = 0.0
    contact_offset_m: float | None = None
    contact_window_observations: int = 0
    calls: int = 0
    valid_insert_calls: int = 0


class ActiveZProbeProvider:
    """Emit a sealed absolute Z waveform and retain contact observations."""

    def reset(self) -> ActiveZProbeState:
        return ActiveZProbeState()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": "rj45-active-z-probe-controller-v1",
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "insert_phase_observation_index": INSERT_PHASE_OBSERVATION_INDEX,
            "policy_active_observation_index": POLICY_ACTIVE_OBSERVATION_INDEX,
            "command_depth_observation_index": COMMAND_DEPTH_OBSERVATION_INDEX,
            "command_depth_scale_m": COMMAND_DEPTH_SCALE_M,
            "command_depth_onset_m": COMMAND_DEPTH_ONSET_M,
            "command_depth_is_onset_only": True,
            "command_depth_onset_basis": "bevel_length_m_plus_controller_lead_cap_m",
            "bevel_length_m": 0.0008,
            "controller_lead_cap_m": 0.0005,
            "force_observation_start_index": FORCE_OBSERVATION_START_INDEX,
            "force_observation_stop_index": FORCE_OBSERVATION_STOP_INDEX,
            "force_scale_n": FORCE_SCALE_N,
            "contact_threshold_n": CONTACT_THRESHOLD_N,
            "safety_threshold_n": SAFETY_THRESHOLD_N,
            "action_z_index": ACTION_Z_INDEX,
            "action_offset_scale_m": ACTION_OFFSET_SCALE_M,
            "probe_step_m": PROBE_STEP_M,
            "probe_peak_m": PROBE_PEAK_M,
            "probe_dwell_ticks": PROBE_DWELL_TICKS,
            "direction_change_zero_ticks": DIRECTION_CHANGE_ZERO_TICKS,
            "direction_order": ["positive_z", "negative_z"],
            "waveform_repeat_count": WAVEFORM_REPEAT_COUNT,
            "diagnostic_episode_repeat_count": DIAGNOSTIC_EPISODE_REPEAT_COUNT,
            "single_repeat_ticks": len(_SINGLE_REPEAT_WAVEFORM_M),
            "waveform_offsets_m": list(_WAVEFORM_M),
            "contact_window_observations": CONTACT_WINDOW_OBSERVATIONS,
            "contact_return_starts_on_final_observation": True,
            "return_step_m": RETURN_STEP_M,
            "phase_or_active_loss_latches_invalid": True,
            "safety_latch_is_permanent": True,
            "action_semantics": "absolute_offset",
            "force_sample_semantics": "current_observation_matches_previous_action",
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

    def state_telemetry(self, state: ActiveZProbeState) -> dict[str, Any]:
        self._validate_state(state)
        return {
            "phase": state.phase,
            "waveform_index": state.waveform_index,
            "waveform_length": len(_WAVEFORM_M),
            "last_commanded_offset_m": state.last_commanded_offset_m,
            "contact_offset_m": state.contact_offset_m,
            "contact_window_observations": state.contact_window_observations,
            "calls": state.calls,
            "valid_insert_calls": state.valid_insert_calls,
            "done": state.phase == "done",
            "safety_latched": state.phase == "safety_latched",
            "invalid_latched": state.phase == "invalid_latched",
        }

    def act(
        self, observation79: Sequence[float], state: ActiveZProbeState
    ) -> tuple[np.ndarray, ActiveZProbeState]:
        observation = _observation(observation79)
        self._validate_state(state)
        calls = state.calls + 1
        force_norm_n = float(
            np.linalg.norm(
                observation[FORCE_OBSERVATION_START_INDEX:FORCE_OBSERVATION_STOP_INDEX]
                * FORCE_SCALE_N
            )
        )

        if state.phase == "safety_latched" or force_norm_n >= SAFETY_THRESHOLD_N:
            return _zero_action(), replace(
                state, phase="safety_latched", last_commanded_offset_m=0.0, calls=calls
            )

        is_insert_active = (
            observation[INSERT_PHASE_OBSERVATION_INDEX] == 1.0
            and observation[POLICY_ACTIVE_OBSERVATION_INDEX] == 1.0
        )
        onset_depth_reached = (
            observation[COMMAND_DEPTH_OBSERVATION_INDEX] * COMMAND_DEPTH_SCALE_M
            >= COMMAND_DEPTH_ONSET_M
        )
        if state.phase == "waiting":
            if not is_insert_active or not onset_depth_reached:
                return _zero_action(), replace(
                    state, last_commanded_offset_m=0.0, calls=calls
                )
            return _zero_action(), replace(
                state,
                phase="probing",
                last_commanded_offset_m=0.0,
                calls=calls,
                valid_insert_calls=state.valid_insert_calls + 1,
            )
        if state.phase in {"done", "invalid_latched"}:
            return _zero_action(), replace(
                state,
                last_commanded_offset_m=0.0,
                calls=calls,
            )
        if not is_insert_active:
            return _zero_action(), replace(
                state,
                phase="invalid_latched",
                last_commanded_offset_m=0.0,
                calls=calls,
            )

        valid_insert_calls = state.valid_insert_calls + 1
        if state.phase == "contact_window":
            return self._sample_contact_window(state, calls, valid_insert_calls)
        if state.phase == "returning":
            return self._return_to_zero(state, calls, valid_insert_calls)

        # The force in this observation was caused by the action returned on
        # the previous call. Zero commands are deliberately not probe contact.
        if force_norm_n >= CONTACT_THRESHOLD_N and state.last_commanded_offset_m != 0.0:
            hold_offset_m = state.last_commanded_offset_m
            next_state = replace(
                state,
                phase="contact_window",
                contact_offset_m=hold_offset_m,
                contact_window_observations=1,
                last_commanded_offset_m=hold_offset_m,
                calls=calls,
                valid_insert_calls=valid_insert_calls,
            )
            return _action_for_offset(hold_offset_m), next_state

        offset_m = _WAVEFORM_M[state.waveform_index]
        next_index = state.waveform_index + 1
        next_phase = "done" if next_index == len(_WAVEFORM_M) else "probing"
        next_state = replace(
            state,
            phase=next_phase,
            waveform_index=next_index,
            last_commanded_offset_m=offset_m,
            calls=calls,
            valid_insert_calls=valid_insert_calls,
        )
        return _action_for_offset(offset_m), next_state

    @staticmethod
    def _sample_contact_window(
        state: ActiveZProbeState, calls: int, valid_insert_calls: int
    ) -> tuple[np.ndarray, ActiveZProbeState]:
        if state.contact_offset_m is None:
            raise ValueError("contact_window state must retain contact_offset_m")
        sample_count = state.contact_window_observations + 1
        next_state = replace(
            state,
            contact_window_observations=sample_count,
            last_commanded_offset_m=state.contact_offset_m,
            calls=calls,
            valid_insert_calls=valid_insert_calls,
        )
        if sample_count == CONTACT_WINDOW_OBSERVATIONS:
            return ActiveZProbeProvider._return_to_zero(
                replace(next_state, phase="returning"), calls, valid_insert_calls
            )
        return _action_for_offset(state.contact_offset_m), next_state

    @staticmethod
    def _return_to_zero(
        state: ActiveZProbeState, calls: int, valid_insert_calls: int
    ) -> tuple[np.ndarray, ActiveZProbeState]:
        magnitude_m = max(abs(state.last_commanded_offset_m) - RETURN_STEP_M, 0.0)
        if magnitude_m <= RETURN_STEP_M * 1e-12:
            magnitude_m = 0.0
        offset_m = float(np.copysign(magnitude_m, state.last_commanded_offset_m)) if magnitude_m else 0.0
        phase = "done" if offset_m == 0.0 else "returning"
        next_state = replace(
            state,
            phase=phase,
            last_commanded_offset_m=offset_m,
            calls=calls,
            valid_insert_calls=valid_insert_calls,
        )
        return _action_for_offset(offset_m), next_state

    @staticmethod
    def _validate_state(state: ActiveZProbeState) -> None:
        if not isinstance(state, ActiveZProbeState):
            raise TypeError("state must come from ActiveZProbeProvider.reset")
        if state.phase not in {
            "waiting",
            "probing",
            "contact_window",
            "returning",
            "done",
            "safety_latched",
            "invalid_latched",
        }:
            raise ValueError("state has an invalid phase")
        if not 0 <= state.waveform_index <= len(_WAVEFORM_M):
            raise ValueError("state has an invalid waveform_index")
        if not np.isfinite(state.last_commanded_offset_m):
            raise ValueError("state has a non-finite last_commanded_offset_m")
        if state.contact_offset_m is not None and not np.isfinite(state.contact_offset_m):
            raise ValueError("state has a non-finite contact_offset_m")
        if not 0 <= state.contact_window_observations <= CONTACT_WINDOW_OBSERVATIONS:
            raise ValueError("state has an invalid contact_window_observations")
        if state.calls < 0 or state.valid_insert_calls < 0:
            raise ValueError("state counters must be non-negative")


def make_active_z_probe_provider() -> ActiveZProbeProvider:
    """Construct the sealed provider without a case or direction input."""

    return ActiveZProbeProvider()
