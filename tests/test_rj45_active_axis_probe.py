import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from rby1_assembly.rj45_active_axis_probe import ActiveAxisProbeProvider


def _valid_observation() -> np.ndarray:
    observation = np.zeros(79, dtype=np.float64)
    observation[46] = 1.0
    observation[62] = 0.0013 / 0.012
    observation[78] = 1.0
    return observation


def _step(provider, state, observation=None):
    sample = _valid_observation() if observation is None else observation
    return provider.act(sample, state)


def test_tx_and_tz_have_identical_state_machine_and_only_axis_differs() -> None:
    tx = ActiveAxisProbeProvider("tx")
    tz = ActiveAxisProbeProvider("tz")
    tx_state = tx.reset()
    tz_state = tz.reset()
    for _ in range(58):  # onset-zero plus the complete 57-tick waveform
        tx_action, tx_state = _step(tx, tx_state)
        tz_action, tz_state = _step(tz, tz_state)
        assert tx.state_telemetry(tx_state) == tz.state_telemetry(tz_state)
        assert tx_action[0] == tz_action[2]
        np.testing.assert_array_equal(tx_action[1:], np.zeros(5))
        np.testing.assert_array_equal(tz_action[[0, 1, 3, 4, 5]], np.zeros(5))
    assert tx.state_telemetry(tx_state)["phase"] == "done"


@pytest.mark.parametrize("axis", ("x", "z", "TX", "", None, 0, []))
def test_rejects_every_axis_other_than_exact_tx_or_tz(axis) -> None:
    with pytest.raises(ValueError, match="exactly 'tx' or 'tz'"):
        ActiveAxisProbeProvider(axis)


def test_constructor_and_observation_api_are_case_sign_grasp_and_gt_blind() -> None:
    assert tuple(inspect.signature(ActiveAxisProbeProvider).parameters) == ("axis",)
    with pytest.raises(TypeError):
        ActiveAxisProbeProvider("tx", sign=1)
    provider = ActiveAxisProbeProvider("tx")
    assert not hasattr(provider, "case_id")
    assert not hasattr(provider, "grasp_id")
    assert not hasattr(provider, "ground_truth")
    with pytest.raises(TypeError):
        provider.act(
            {"observation79": _valid_observation(), "case_id": "tx-positive"},
            provider.reset(),
        )


@pytest.mark.parametrize("axis,index", (("tx", 0), ("tz", 2)))
def test_onset_transition_is_exact_zero(axis: str, index: int) -> None:
    provider = ActiveAxisProbeProvider(axis)
    state = provider.reset()
    shallow = _valid_observation()
    shallow[62] = np.nextafter(0.0013 / 0.012, 0.0)
    action, state = _step(provider, state, shallow)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["phase"] == "waiting"

    action, state = _step(provider, state)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["phase"] == "probing"
    action, _ = _step(provider, state)
    assert action[index] == pytest.approx(0.1)


@pytest.mark.parametrize("axis", ("tx", "tz"))
def test_twenty_newton_safety_stop_is_immediate_and_permanent(axis: str) -> None:
    provider = ActiveAxisProbeProvider(axis)
    state = provider.reset()
    _, state = _step(provider, state)
    _, state = _step(provider, state)
    stop = np.zeros(79, dtype=np.float64)
    stop[38] = 20.0 / 8.0
    action, state = _step(provider, state, stop)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["safety_latched"] is True
    action, state = _step(provider, state)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["phase"] == "safety_latched"


@pytest.mark.parametrize("axis", ("tx", "tz"))
def test_active_input_loss_latches_invalid_for_both_axes(axis: str) -> None:
    provider = ActiveAxisProbeProvider(axis)
    _, state = _step(provider, provider.reset())
    _, state = _step(provider, state)
    invalid = _valid_observation()
    invalid[78] = 0.0
    action, state = _step(provider, state, invalid)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["phase"] == "invalid_latched"
    action, state = _step(provider, state)
    np.testing.assert_array_equal(action, np.zeros(6))
    assert provider.state_telemetry(state)["phase"] == "invalid_latched"


@pytest.mark.parametrize("axis,index", (("tx", 0), ("tz", 2)))
def test_contact_window_has_thirteen_samples_then_bounded_return(axis: str, index: int) -> None:
    provider = ActiveAxisProbeProvider(axis)
    state = provider.reset()
    _, state = _step(provider, state)  # onset zero
    for _ in range(13):
        action, state = _step(provider, state)
    assert action[index] == pytest.approx(0.3)  # 0.45 mm absolute offset

    contact = _valid_observation()
    contact[37] = 0.25 / 8.0
    action, state = _step(provider, state, contact)
    assert action[index] == pytest.approx(0.3)
    assert provider.state_telemetry(state)["contact_window_observations"] == 1
    for expected_count in range(2, 13):
        action, state = _step(provider, state)
        assert action[index] == pytest.approx(0.3)
        assert provider.state_telemetry(state)["contact_window_observations"] == expected_count

    returned = []
    action, state = _step(provider, state)  # observation 13 starts return now
    returned.append(float(action[index]))
    while provider.state_telemetry(state)["phase"] == "returning":
        action, state = _step(provider, state)
        returned.append(float(action[index]))
    np.testing.assert_allclose(returned, [0.2, 0.1, 0.0], rtol=0.0, atol=1e-15)
    assert provider.state_telemetry(state)["contact_window_observations"] == 13
    assert provider.state_telemetry(state)["phase"] == "done"


def test_v2_payload_sha_and_sibling_source_invariant() -> None:
    tx = ActiveAxisProbeProvider("tx")
    tz = ActiveAxisProbeProvider("tz")
    tx_payload = tx.canonical_payload()
    tz_payload = tz.canonical_payload()
    assert tx_payload["schema"] == "rj45-active-axis-probe-controller-v2"
    assert tx_payload["probe_axis"] == "tx"
    assert tx_payload["action_translation_index"] == 0
    assert tx_payload["direction_order"] == ["positive_x", "negative_x"]
    assert tz_payload["probe_axis"] == "tz"
    assert tz_payload["action_translation_index"] == 2
    assert tz_payload["direction_order"] == ["positive_z", "negative_z"]
    for payload in (tx_payload, tz_payload):
        assert payload["waveform_repeat_count"] == 1
        assert "diagnostic_episode_repeat_count" not in payload
        assert len(payload["waveform_offsets_m"]) == 57
        json.dumps(payload, allow_nan=False)

    provider_path = Path(inspect.getfile(tx._base.__class__)).resolve()
    source_sha256 = hashlib.sha256(provider_path.read_bytes()).hexdigest()
    assert tx_payload["base_provider_source_sha256"] == source_sha256
    assert tx_payload["base_provider_controller_sha256"] == tx._base.controller_sha256()
    assert tx.controller_sha256() == ActiveAxisProbeProvider("tx").controller_sha256()
    assert tx.controller_sha256() != tz.controller_sha256()
    assert len(tx.controller_sha256()) == 64

    tx_payload["probe_axis"] = "forged"
    assert tx.canonical_payload()["probe_axis"] == "tx"


def test_state_telemetry_is_json_serializable() -> None:
    provider = ActiveAxisProbeProvider("tx")
    telemetry = provider.state_telemetry(provider.reset())
    json.dumps(telemetry, allow_nan=False)
    assert telemetry["phase"] == "waiting"
