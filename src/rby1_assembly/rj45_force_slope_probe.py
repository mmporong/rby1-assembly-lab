"""Case-blind CPU-only force-slope probe for controlled RJ45 diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
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
ACTION_X_INDEX = 0

FORCE_SCALE_N = 8.0
COMMAND_DEPTH_SCALE_M = 0.012
COMMAND_DEPTH_ONSET_M = 0.0013
PROJECTED_CONTACT_THRESHOLD_N = 0.25
SAFETY_THRESHOLD_N = 20.0
ACTION_OFFSET_SCALE_M = 0.0015
PROBE_STEP_M = 0.00015
PROBE_PEAK_M = 0.00060
IMPULSE_RESPONSE_OBSERVATIONS = 4
SWEEP_DWELL_TICKS = 6
CONTACT_WINDOW_OBSERVATIONS = 13
RETURN_STEP_M = 0.00015

_SWEEP_MAGNITUDES_M = tuple(
    magnitude_m
    for magnitude_m in (0.00015, 0.00030, 0.00045, 0.00060)
    for _ in range(SWEEP_DWELL_TICKS)
)
_PHASES = {
    "waiting",
    "impulse_pending",
    "impulse_responses",
    "sweep",
    "contact_window",
    "returning",
    "done",
    "invalid_latched",
    "safety_latched",
}
_MEASUREMENT_STATUSES = {
    "pending",
    "valid",
    "invalid_zero_slope",
    "invalid_no_projected_contact",
    "invalid_phase_or_active_loss",
    "invalid_safety_force",
}


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
    action[ACTION_X_INDEX] = offset_m / ACTION_OFFSET_SCALE_M
    return action


@dataclass(frozen=True)
class ForceSlopeProbeState:
    phase: str = "waiting"
    impulse_force_samples_n: tuple[float, ...] = ()
    selected_direction: int = 0
    sweep_index: int = 0
    last_commanded_offset_m: float = 0.0
    contact_offset_m: float | None = None
    contact_window_observations: int = 0
    contact_found: bool = False
    measurement_status: str = "pending"
    calls: int = 0
    valid_insert_calls: int = 0


class ForceSlopeProbeProvider:
    """Infer one X direction, sweep it, and retain a projected-force window."""

    def reset(self) -> ForceSlopeProbeState:
        return ForceSlopeProbeState()

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "schema": "rj45-force-slope-probe-controller-v3",
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "insert_phase_observation_index": INSERT_PHASE_OBSERVATION_INDEX,
            "policy_active_observation_index": POLICY_ACTIVE_OBSERVATION_INDEX,
            "command_depth_observation_index": COMMAND_DEPTH_OBSERVATION_INDEX,
            "command_depth_scale_m": COMMAND_DEPTH_SCALE_M,
            "command_depth_onset_m": COMMAND_DEPTH_ONSET_M,
            "command_depth_is_onset_only": True,
            "force_observation_start_index": FORCE_OBSERVATION_START_INDEX,
            "force_observation_stop_index": FORCE_OBSERVATION_STOP_INDEX,
            "force_scale_n": FORCE_SCALE_N,
            "slope_force_axis": "x",
            "impulse_offset_m": PROBE_STEP_M,
            "impulse_response_observations": IMPULSE_RESPONSE_OBSERVATIONS,
            "slope_decision": "sign(abs_fx_m4_minus_abs_fx_m1)",
            "zero_slope_latches_invalid": True,
            "decision_zero_separator_ticks": 1,
            "decision_zero_occurs_on_final_impulse_response": True,
            "projected_contact_threshold_n": PROJECTED_CONTACT_THRESHOLD_N,
            "projected_contact_expression": "-sign(previous_offset_m)*fx_n",
            "contact_detection_phase": "sweep_only",
            "safety_threshold_n": SAFETY_THRESHOLD_N,
            "action_translation_index": ACTION_X_INDEX,
            "action_offset_scale_m": ACTION_OFFSET_SCALE_M,
            "sweep_step_m": PROBE_STEP_M,
            "sweep_peak_m": PROBE_PEAK_M,
            "sweep_dwell_ticks": SWEEP_DWELL_TICKS,
            "sweep_magnitudes_m": list(_SWEEP_MAGNITUDES_M),
            "contact_window_observations": CONTACT_WINDOW_OBSERVATIONS,
            "contact_return_starts_on_final_observation": True,
            "return_step_m": RETURN_STEP_M,
            "phase_or_active_loss_latches_invalid": True,
            "safety_latch_is_permanent": True,
            "action_semantics": "absolute_offset",
            "force_sample_semantics": "current_observation_matches_previous_action",
            "case_sign_grasp_ground_truth_inputs": False,
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

    def state_telemetry(self, state: ForceSlopeProbeState) -> dict[str, Any]:
        self._validate_state(state)
        telemetry = asdict(state)
        telemetry["impulse_force_samples_n"] = list(state.impulse_force_samples_n)
        telemetry.update(
            {
                "sweep_length": len(_SWEEP_MAGNITUDES_M),
                "done": state.phase == "done",
                "invalid_latched": state.phase == "invalid_latched",
                "safety_latched": state.phase == "safety_latched",
            }
        )
        return telemetry

    def act(
        self, observation79: Sequence[float], state: ForceSlopeProbeState
    ) -> tuple[np.ndarray, ForceSlopeProbeState]:
        observation = _observation(observation79)
        self._validate_state(state)
        calls = state.calls + 1
        force_n = observation[FORCE_OBSERVATION_START_INDEX:FORCE_OBSERVATION_STOP_INDEX] * FORCE_SCALE_N
        force_norm_n = float(np.linalg.norm(force_n))

        if state.phase == "safety_latched" or force_norm_n >= SAFETY_THRESHOLD_N:
            return _zero_action(), replace(
                state,
                phase="safety_latched",
                last_commanded_offset_m=0.0,
                measurement_status="invalid_safety_force",
                calls=calls,
            )
        if state.phase in {"done", "invalid_latched"}:
            return _zero_action(), replace(state, last_commanded_offset_m=0.0, calls=calls)

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
                return _zero_action(), replace(state, last_commanded_offset_m=0.0, calls=calls)
            return _zero_action(), replace(
                state,
                phase="impulse_pending",
                last_commanded_offset_m=0.0,
                calls=calls,
                valid_insert_calls=state.valid_insert_calls + 1,
            )
        if not is_insert_active:
            return _zero_action(), replace(
                state,
                phase="invalid_latched",
                last_commanded_offset_m=0.0,
                measurement_status="invalid_phase_or_active_loss",
                calls=calls,
            )

        valid_insert_calls = state.valid_insert_calls + 1
        if state.phase == "impulse_pending":
            return _action_for_offset(PROBE_STEP_M), replace(
                state,
                phase="impulse_responses",
                last_commanded_offset_m=PROBE_STEP_M,
                calls=calls,
                valid_insert_calls=valid_insert_calls,
            )
        if state.phase == "impulse_responses":
            samples = state.impulse_force_samples_n + (float(abs(force_n[0])),)
            if len(samples) < IMPULSE_RESPONSE_OBSERVATIONS:
                return _action_for_offset(PROBE_STEP_M), replace(
                    state,
                    impulse_force_samples_n=samples,
                    last_commanded_offset_m=PROBE_STEP_M,
                    calls=calls,
                    valid_insert_calls=valid_insert_calls,
                )
            delta_n = samples[-1] - samples[0]
            if delta_n == 0.0:
                return _zero_action(), replace(
                    state,
                    phase="invalid_latched",
                    impulse_force_samples_n=samples,
                    last_commanded_offset_m=0.0,
                    measurement_status="invalid_zero_slope",
                    calls=calls,
                    valid_insert_calls=valid_insert_calls,
                )
            direction = 1 if delta_n > 0.0 else -1
            return _zero_action(), replace(
                state,
                phase="sweep",
                impulse_force_samples_n=samples,
                selected_direction=direction,
                last_commanded_offset_m=0.0,
                calls=calls,
                valid_insert_calls=valid_insert_calls,
            )
        if state.phase == "contact_window":
            return self._sample_contact_window(state, calls, valid_insert_calls)
        if state.phase == "returning":
            return self._return_to_zero(state, calls, valid_insert_calls)

        if state.phase != "sweep":
            raise ValueError("state phase cannot produce an action")

        previous_offset_m = state.last_commanded_offset_m
        if previous_offset_m != 0.0:
            projected_force_n = -float(np.sign(previous_offset_m)) * float(force_n[0])
            if projected_force_n >= PROJECTED_CONTACT_THRESHOLD_N:
                return _action_for_offset(previous_offset_m), replace(
                    state,
                    phase="contact_window",
                    contact_offset_m=previous_offset_m,
                    contact_window_observations=1,
                    contact_found=True,
                    last_commanded_offset_m=previous_offset_m,
                    measurement_status="valid",
                    calls=calls,
                    valid_insert_calls=valid_insert_calls,
                )

        if state.sweep_index == len(_SWEEP_MAGNITUDES_M):
            return self._return_to_zero(
                replace(state, phase="returning", measurement_status="invalid_no_projected_contact"),
                calls,
                valid_insert_calls,
            )
        offset_m = state.selected_direction * _SWEEP_MAGNITUDES_M[state.sweep_index]
        return _action_for_offset(offset_m), replace(
            state,
            sweep_index=state.sweep_index + 1,
            last_commanded_offset_m=float(offset_m),
            calls=calls,
            valid_insert_calls=valid_insert_calls,
        )

    @staticmethod
    def _sample_contact_window(
        state: ForceSlopeProbeState, calls: int, valid_insert_calls: int
    ) -> tuple[np.ndarray, ForceSlopeProbeState]:
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
            return ForceSlopeProbeProvider._return_to_zero(
                replace(next_state, phase="returning"), calls, valid_insert_calls
            )
        return _action_for_offset(state.contact_offset_m), next_state

    @staticmethod
    def _return_to_zero(
        state: ForceSlopeProbeState, calls: int, valid_insert_calls: int
    ) -> tuple[np.ndarray, ForceSlopeProbeState]:
        magnitude_m = max(abs(state.last_commanded_offset_m) - RETURN_STEP_M, 0.0)
        if magnitude_m <= RETURN_STEP_M * 1e-12:
            magnitude_m = 0.0
        offset_m = float(np.copysign(magnitude_m, state.last_commanded_offset_m)) if magnitude_m else 0.0
        phase = "done" if offset_m == 0.0 else "returning"
        return _action_for_offset(offset_m), replace(
            state,
            phase=phase,
            last_commanded_offset_m=offset_m,
            calls=calls,
            valid_insert_calls=valid_insert_calls,
        )

    @staticmethod
    def _validate_state(state: ForceSlopeProbeState) -> None:
        if not isinstance(state, ForceSlopeProbeState):
            raise TypeError("state must come from ForceSlopeProbeProvider.reset")
        if state.phase not in _PHASES:
            raise ValueError("state has an invalid phase")
        if not isinstance(state.impulse_force_samples_n, tuple):
            raise TypeError("state impulse_force_samples_n must be a tuple")
        if len(state.impulse_force_samples_n) > IMPULSE_RESPONSE_OBSERVATIONS:
            raise ValueError("state has too many impulse force samples")
        if any(type(value) is not float or not np.isfinite(value) or value < 0.0 for value in state.impulse_force_samples_n):
            raise ValueError("state has invalid impulse force samples")
        if type(state.selected_direction) is not int or state.selected_direction not in {-1, 0, 1}:
            raise ValueError("state has an invalid selected_direction")
        if type(state.sweep_index) is not int or not 0 <= state.sweep_index <= len(_SWEEP_MAGNITUDES_M):
            raise ValueError("state has an invalid sweep_index")
        if type(state.last_commanded_offset_m) is not float or not np.isfinite(state.last_commanded_offset_m):
            raise ValueError("state has an invalid last_commanded_offset_m")
        if abs(state.last_commanded_offset_m) > PROBE_PEAK_M + 1e-12:
            raise ValueError("state last_commanded_offset_m exceeds probe bounds")
        if state.contact_offset_m is not None and (
            type(state.contact_offset_m) is not float
            or not np.isfinite(state.contact_offset_m)
            or abs(state.contact_offset_m) > PROBE_PEAK_M + 1e-12
            or state.contact_offset_m == 0.0
        ):
            raise ValueError("state has an invalid contact_offset_m")
        if type(state.contact_found) is not bool:
            raise TypeError("state contact_found must be bool")
        if type(state.contact_window_observations) is not int or not 0 <= state.contact_window_observations <= CONTACT_WINDOW_OBSERVATIONS:
            raise ValueError("state has invalid contact_window_observations")
        if state.contact_found != (state.contact_offset_m is not None):
            raise ValueError("state contact fields are inconsistent")
        if state.contact_found != (state.contact_window_observations > 0):
            raise ValueError("state contact count is inconsistent")
        if state.measurement_status not in _MEASUREMENT_STATUSES:
            raise ValueError("state has an invalid measurement_status")
        if type(state.calls) is not int or type(state.valid_insert_calls) is not int:
            raise TypeError("state counters must be integers")
        if state.calls < 0 or state.valid_insert_calls < 0 or state.valid_insert_calls > state.calls:
            raise ValueError("state counters are inconsistent")
        if state.phase in {"sweep", "contact_window", "returning", "done"} and len(state.impulse_force_samples_n) != IMPULSE_RESPONSE_OBSERVATIONS:
            raise ValueError("state phase requires four impulse force samples")
        if state.phase in {"sweep", "contact_window", "returning", "done"} and state.selected_direction == 0:
            raise ValueError("state phase requires a selected direction")
        if state.phase in {"waiting", "impulse_pending", "impulse_responses"} and (
            state.selected_direction != 0 or state.sweep_index != 0 or state.contact_found
        ):
            raise ValueError("pre-decision state contains post-decision data")
        if state.phase in {"waiting", "impulse_pending"} and state.impulse_force_samples_n:
            raise ValueError("state phase cannot contain impulse responses")
        if state.phase == "impulse_responses" and len(state.impulse_force_samples_n) >= IMPULSE_RESPONSE_OBSERVATIONS:
            raise ValueError("impulse response state must transition after the fourth sample")
        if state.phase == "sweep" and (state.contact_found or state.measurement_status != "pending"):
            raise ValueError("sweep state cannot claim a measurement outcome")
        if state.phase == "contact_window" and (
            not state.contact_found or state.measurement_status != "valid"
        ):
            raise ValueError("contact window state requires a valid projected contact")
        if state.phase in {"returning", "done"}:
            expected_status = "valid" if state.contact_found else "invalid_no_projected_contact"
            if state.measurement_status != expected_status:
                raise ValueError("terminal path has an inconsistent measurement status")


def make_force_slope_probe_provider() -> ForceSlopeProbeProvider:
    return ForceSlopeProbeProvider()


__all__ = ["ForceSlopeProbeProvider", "ForceSlopeProbeState", "make_force_slope_probe_provider"]
