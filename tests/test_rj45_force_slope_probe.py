from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import inspect
import json

import numpy as np
import pytest

from rby1_assembly import rj45_force_slope_probe as probe


def _observation(*, phase: float = 1.0, active: float = 1.0, depth: float = 0.2) -> np.ndarray:
    observation = np.zeros(probe.OBSERVATION_DIM, dtype=np.float64)
    observation[probe.INSERT_PHASE_OBSERVATION_INDEX] = phase
    observation[probe.POLICY_ACTIVE_OBSERVATION_INDEX] = active
    observation[probe.COMMAND_DEPTH_OBSERVATION_INDEX] = depth
    return observation


def _with_force(observation: np.ndarray, x_n: float = 0.0, y_n: float = 0.0, z_n: float = 0.0) -> np.ndarray:
    result = observation.copy()
    result[probe.FORCE_OBSERVATION_START_INDEX:probe.FORCE_OBSERVATION_STOP_INDEX] = (
        np.array([x_n, y_n, z_n], dtype=np.float64) / probe.FORCE_SCALE_N
    )
    return result


def _assert_x_action(action: np.ndarray, offset_m: float) -> None:
    expected = np.zeros(probe.ACTION_DIM)
    expected[probe.ACTION_X_INDEX] = offset_m / probe.ACTION_OFFSET_SCALE_M
    np.testing.assert_allclose(action, expected, rtol=0.0, atol=1e-15)


def _complete_impulse(provider: probe.ForceSlopeProbeProvider, samples: tuple[float, float, float, float]):
    state = provider.reset()
    onset_action, state = provider.act(_observation(), state)
    _assert_x_action(onset_action, 0.0)
    pulse_action, state = provider.act(_observation(), state)
    _assert_x_action(pulse_action, probe.PROBE_STEP_M)
    actions = []
    states = []
    for sample in samples:
        action, state = provider.act(_with_force(_observation(), x_n=sample), state)
        actions.append(action)
        states.append(state)
    return state, actions, states


def _reach_sweep(provider: probe.ForceSlopeProbeProvider, direction: int = 1):
    samples = (1.0, 2.0, 3.0, 4.0) if direction > 0 else (4.0, 3.0, 2.0, 1.0)
    state, _, _ = _complete_impulse(provider, samples)
    assert state.phase == "sweep"
    assert state.selected_direction == direction
    return state


def test_onset_is_exact_zero_and_requires_insert_active_and_depth() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = provider.reset()
    threshold_raw = probe.COMMAND_DEPTH_ONSET_M / probe.COMMAND_DEPTH_SCALE_M

    for observation in (
        _observation(phase=0.0),
        _observation(active=0.0),
        _observation(depth=np.nextafter(threshold_raw, 0.0)),
    ):
        action, next_state = provider.act(observation, state)
        _assert_x_action(action, 0.0)
        assert next_state.phase == "waiting"

    action, state = provider.act(_observation(depth=threshold_raw), state)
    _assert_x_action(action, 0.0)
    assert state.phase == "impulse_pending"
    assert state.valid_insert_calls == 1


def test_fourth_response_is_zero_separator_and_next_call_is_selected_sweep() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state, actions, states = _complete_impulse(provider, (1.0, 2.0, 3.0, 4.0))

    for index, (action, response_state) in enumerate(zip(actions, states), start=1):
        expected_offset_m = probe.PROBE_STEP_M if index < 4 else 0.0
        _assert_x_action(action, expected_offset_m)
        assert len(response_state.impulse_force_samples_n) == index
    assert state.impulse_force_samples_n == (1.0, 2.0, 3.0, 4.0)
    assert state.phase == "sweep"
    assert state.last_commanded_offset_m == 0.0
    assert not state.contact_found
    selected_action, state = provider.act(_observation(), state)
    _assert_x_action(selected_action, probe.PROBE_STEP_M)
    assert state.sweep_index == 1


@pytest.mark.parametrize(
    ("samples", "direction"),
    [((1.0, 2.0, 3.0, 4.0), 1), ((4.0, 3.0, 2.0, 1.0), -1)],
)
def test_slope_selects_direction(samples: tuple[float, ...], direction: int) -> None:
    provider = probe.ForceSlopeProbeProvider()
    state, actions, _ = _complete_impulse(provider, samples)
    _assert_x_action(actions[-1], 0.0)
    assert state.selected_direction == direction
    action, state = provider.act(_observation(), state)
    _assert_x_action(action, direction * probe.PROBE_STEP_M)


def test_exact_zero_slope_latches_invalid_permanently() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state, actions, _ = _complete_impulse(provider, (1.0, 2.0, 2.0, 1.0))
    _assert_x_action(actions[-1], 0.0)
    assert state.phase == "invalid_latched"
    assert state.measurement_status == "invalid_zero_slope"
    action, state = provider.act(_observation(), state)
    _assert_x_action(action, 0.0)
    assert state.phase == "invalid_latched"


@pytest.mark.parametrize(("direction", "force_x_n"), [(1, -0.25), (-1, 0.25)])
def test_projected_contact_uses_previous_selected_sweep_action(direction: int, force_x_n: float) -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = _reach_sweep(provider, direction)

    selected_action, state = provider.act(_with_force(_observation(), x_n=force_x_n), state)
    _assert_x_action(selected_action, direction * probe.PROBE_STEP_M)
    assert not state.contact_found
    contact_action, state = provider.act(_with_force(_observation(), x_n=force_x_n), state)
    _assert_x_action(contact_action, direction * probe.PROBE_STEP_M)
    assert state.phase == "contact_window"
    assert state.contact_found
    assert state.contact_window_observations == 1


def test_wrong_direction_and_yz_force_do_not_trigger_contact() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = _reach_sweep(provider, 1)
    _, state = provider.act(_observation(), state)

    for force in ((0.25, 0.0, 0.0), (0.0, 10.0, -10.0)):
        action, state = provider.act(_with_force(_observation(), *force), state)
        assert state.phase == "sweep"
        assert not state.contact_found
        assert np.count_nonzero(action[1:]) == 0


def test_impulse_and_separator_never_latch_contact() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = provider.reset()
    _, state = provider.act(_observation(), state)
    _, state = provider.act(_observation(), state)
    for sample in (5.0, 6.0, 7.0, 8.0):
        action, state = provider.act(_with_force(_observation(), x_n=-sample, y_n=1.0), state)
        expected_offset_m = probe.PROBE_STEP_M if sample < 8.0 else 0.0
        _assert_x_action(action, expected_offset_m)
        assert not state.contact_found
    assert state.phase == "sweep"
    assert not state.contact_found


def test_contact_sample_thirteen_starts_bounded_return_on_same_call() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = _reach_sweep(provider, 1)
    _, state = provider.act(_observation(), state)
    _, state = provider.act(_with_force(_observation(), x_n=-0.25), state)
    assert state.contact_window_observations == 1
    for expected_count in range(2, 13):
        action, state = provider.act(_observation(), state)
        _assert_x_action(action, probe.PROBE_STEP_M)
        assert state.contact_window_observations == expected_count
    action, state = provider.act(_observation(), state)
    _assert_x_action(action, 0.0)
    assert state.phase == "done"
    assert state.contact_window_observations == 13
    assert state.measurement_status == "valid"


def test_contact_from_higher_level_returns_at_one_step_per_call() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = _reach_sweep(provider, 1)
    for _ in range(13):
        _, state = provider.act(_observation(), state)
    assert state.last_commanded_offset_m == pytest.approx(0.00045)
    _, state = provider.act(_with_force(_observation(), x_n=-0.25), state)
    assert state.contact_window_observations == 1
    for _ in range(11):
        _, state = provider.act(_observation(), state)
    returned = []
    while state.phase != "done":
        action, state = provider.act(_observation(), state)
        returned.append(float(action[0]))
    assert returned == pytest.approx([0.2, 0.1, 0.0])


def test_no_contact_full_sweep_is_absolute_bounded_and_returns_to_zero() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = _reach_sweep(provider, 1)
    actions = []
    for _ in range(24):
        action, state = provider.act(_observation(), state)
        actions.append(float(action[0]))
        assert np.count_nonzero(action[1:]) == 0
    assert actions == pytest.approx([0.1] * 6 + [0.2] * 6 + [0.3] * 6 + [0.4] * 6)
    assert max(actions) == pytest.approx(0.4)

    returned = []
    while state.phase != "done":
        action, state = provider.act(_observation(), state)
        returned.append(float(action[0]))
    assert returned == pytest.approx([0.3, 0.2, 0.1, 0.0])
    assert state.measurement_status == "invalid_no_projected_contact"
    assert state.contact_window_observations == 0
    assert state.contact_found is False


def test_phase_or_active_loss_after_onset_is_permanent_invalid_but_depth_retreat_is_not() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = provider.reset()
    _, state = provider.act(_observation(), state)
    action, state = provider.act(_observation(depth=0.0), state)
    _assert_x_action(action, probe.PROBE_STEP_M)
    assert state.phase == "impulse_responses"
    action, state = provider.act(_observation(active=0.0), state)
    _assert_x_action(action, 0.0)
    assert state.phase == "invalid_latched"
    assert state.measurement_status == "invalid_phase_or_active_loss"
    action, state = provider.act(_observation(), state)
    _assert_x_action(action, 0.0)
    assert state.phase == "invalid_latched"


def test_safety_threshold_overrides_waiting_impulse_sweep_contact_return_and_done() -> None:
    provider = probe.ForceSlopeProbeProvider()
    states = [provider.reset()]
    state = provider.reset()
    _, state = provider.act(_observation(), state)
    states.append(state)
    _, impulse_response = provider.act(_observation(), state)
    states.append(impulse_response)
    selected, _, selected_states = _complete_impulse(provider, (1.0, 2.0, 3.0, 4.0))
    states.extend((selected_states[-2], selected))
    state = _reach_sweep(provider, 1)
    states.append(state)
    _, state = provider.act(_observation(), state)
    _, contact = provider.act(_with_force(_observation(), x_n=-0.25), state)
    states.append(contact)
    returning = replace(contact, phase="returning")
    states.append(returning)
    done = replace(contact, phase="done", last_commanded_offset_m=0.0)
    states.append(done)

    safety_observation = _with_force(_observation(), x_n=probe.SAFETY_THRESHOLD_N)
    for candidate in states:
        action, latched = provider.act(safety_observation, candidate)
        _assert_x_action(action, 0.0)
        assert latched.phase == "safety_latched"
        assert latched.measurement_status == "invalid_safety_force"
        action, latched = provider.act(_observation(), latched)
        _assert_x_action(action, 0.0)
        assert latched.phase == "safety_latched"


@pytest.mark.parametrize(
    "value,exception",
    [
        (np.zeros(78), ValueError),
        (np.full(79, np.nan), ValueError),
        (np.zeros(79, dtype=np.complex128), TypeError),
        ({"observation": [0.0] * 79}, TypeError),
    ],
)
def test_observation_validation_is_strict(value: object, exception: type[Exception]) -> None:
    with pytest.raises(exception):
        probe.ForceSlopeProbeProvider().act(value, probe.ForceSlopeProbeState())


def test_state_is_immutable_and_tampering_is_rejected() -> None:
    provider = probe.ForceSlopeProbeProvider()
    state = provider.reset()
    with pytest.raises(FrozenInstanceError):
        state.phase = "sweep"  # type: ignore[misc]
    with pytest.raises(TypeError):
        provider.act(_observation(), object())  # type: ignore[arg-type]

    invalid_states = (
        replace(state, phase="foreign"),
        replace(state, impulse_force_samples_n=[1.0]),  # type: ignore[arg-type]
        replace(state, selected_direction=True),
        replace(state, sweep_index=25),
        replace(state, last_commanded_offset_m=0.00061),
        replace(state, contact_found=True),
        replace(state, calls=True),
        replace(state, measurement_status="claimed_success"),
        replace(state, selected_direction=1),
        replace(state, phase="decision"),
    )
    for invalid in invalid_states:
        with pytest.raises((TypeError, ValueError)):
            provider.act(_observation(), invalid)


def test_payload_hash_telemetry_and_factory_are_case_blind_and_json_safe() -> None:
    provider = probe.make_force_slope_probe_provider()
    payload = provider.canonical_payload()
    assert payload["schema"] == "rj45-force-slope-probe-controller-v3"
    assert payload["case_sign_grasp_ground_truth_inputs"] is False
    assert list(inspect.signature(probe.make_force_slope_probe_provider).parameters) == []
    assert list(inspect.signature(probe.ForceSlopeProbeProvider).parameters) == []
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    assert provider.controller_sha256() == hashlib.sha256(encoded).hexdigest()
    assert len(provider.controller_sha256()) == 64
    telemetry = provider.state_telemetry(provider.reset())
    assert telemetry["impulse_force_samples_n"] == []
    assert telemetry["done"] is False
    json.dumps(telemetry, allow_nan=False)
    initial_hash = provider.controller_sha256()
    _, state = provider.act(_observation(), provider.reset())
    assert provider.controller_sha256() == initial_hash

