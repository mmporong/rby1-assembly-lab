import inspect
import json
import unittest

import numpy as np

from rby1_assembly.rj45_active_z_probe import (
    ActiveZProbeProvider,
    CONTACT_WINDOW_OBSERVATIONS,
    make_active_z_probe_provider,
)


class ActiveZProbeProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = make_active_z_probe_provider()
        self.valid = np.zeros(79, dtype=np.float64)
        self.valid[46] = 1.0
        self.valid[62] = 0.0013 / 0.012
        self.valid[78] = 1.0

    def step(self, state, observation=None):
        return self.provider.act(self.valid if observation is None else observation, state)

    def start(self):
        state = self.provider.reset()
        action, state = self.step(state)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "probing")
        return state

    def test_first_transition_and_invalid_insert_inputs_are_exact_zero(self) -> None:
        state = self.provider.reset()
        for index in (46, 78):
            observation = self.valid.copy()
            observation[index] = 0.0
            action, state = self.step(state, observation)
            np.testing.assert_array_equal(action, np.zeros(6))
            self.assertEqual(state.phase, "waiting")

        shallow = self.valid.copy()
        shallow[62] = np.nextafter(0.0013 / 0.012, 0.0)
        action, state = self.step(state, shallow)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "waiting")

        action, state = self.step(state)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "probing")

    def test_invalid_gap_after_onset_latches_zero_and_cannot_resume(self) -> None:
        state = self.start()
        action, state = self.step(state)
        self.assertAlmostEqual(action[2], 0.1)
        inactive = self.valid.copy()
        inactive[78] = 0.0
        action, state = self.step(state, inactive)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertTrue(self.provider.state_telemetry(state)["invalid_latched"])
        action, state = self.step(state)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "invalid_latched")

    def test_depth_threshold_is_onset_only_and_done_is_terminal(self) -> None:
        state = self.start()
        retreated = self.valid.copy()
        retreated[62] = 0.0
        action, state = self.step(state, retreated)
        self.assertAlmostEqual(action[2], 0.1)
        self.assertEqual(state.phase, "probing")

        while state.phase != "done":
            _, state = self.step(state, retreated)
        inactive = retreated.copy()
        inactive[78] = 0.0
        action, state = self.step(state, inactive)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "done")

    def test_waveform_timing_and_absolute_nonaccumulating_actions(self) -> None:
        state = self.start()
        z_actions = []
        for _ in range(57):
            action, state = self.step(state)
            np.testing.assert_array_equal(action[[0, 1, 3, 4, 5]], np.zeros(5))
            z_actions.append(float(action[2]))

        direction = [0.1] * 6 + [0.2] * 6 + [0.3] * 6 + [0.4] * 6 + [0.3, 0.2, 0.1, 0.0]
        one_repeat = direction + [0.0] + [-value for value in direction]
        np.testing.assert_allclose(z_actions, one_repeat, rtol=0.0, atol=1e-15)
        self.assertEqual(len(one_repeat), 57)
        self.assertEqual(state.phase, "done")
        self.assertEqual(state.waveform_index, 57)
        np.testing.assert_allclose(z_actions[5:8], [0.1, 0.2, 0.2], rtol=0.0, atol=1e-15)

    def test_contact_latches_previous_action_and_collects_thirteen_observations(self) -> None:
        state = self.start()
        action, state = self.step(state)
        self.assertAlmostEqual(action[2], 0.1)
        contact = self.valid.copy()
        contact[37] = 0.25 / 8.0
        action, state = self.step(state, contact)
        self.assertAlmostEqual(action[2], 0.1)
        self.assertEqual(state.contact_window_observations, 1)
        self.assertEqual(state.contact_offset_m, 0.00015)

        for expected_count in range(2, CONTACT_WINDOW_OBSERVATIONS):
            action, state = self.step(state)
            self.assertAlmostEqual(action[2], 0.1)
            self.assertEqual(state.contact_window_observations, expected_count)
        self.assertEqual(state.contact_window_observations, 12)
        action, state = self.step(state)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.contact_window_observations, 13)
        self.assertEqual(state.phase, "done")

    def test_contact_return_is_bounded_to_point_one_five_mm_per_tick(self) -> None:
        state = self.start()
        for _ in range(13):
            action, state = self.step(state)
        self.assertAlmostEqual(action[2], 0.3)
        contact = self.valid.copy()
        contact[39] = 0.25 / 8.0
        action, state = self.step(state, contact)
        self.assertAlmostEqual(action[2], 0.3)
        for _ in range(11):
            action, state = self.step(state)
            self.assertAlmostEqual(action[2], 0.3)
        action, state = self.step(state)  # sample 13 observed; return starts now
        self.assertEqual(state.phase, "returning")

        returned = [float(action[2])]
        while state.phase == "returning":
            action, state = self.step(state)
            returned.append(float(action[2]))
        np.testing.assert_allclose(returned, [0.2, 0.1, 0.0], rtol=0.0, atol=1e-15)

    def test_contact_is_not_detected_until_a_nonzero_probe_action_was_applied(self) -> None:
        contact = self.valid.copy()
        contact[37] = 0.25 / 8.0
        action, state = self.provider.act(contact, self.provider.reset())
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "probing")
        action, state = self.provider.act(contact, state)
        self.assertAlmostEqual(action[2], 0.1)
        self.assertEqual(state.phase, "probing")
        self.assertEqual(state.contact_window_observations, 0)

    def test_twenty_newton_override_is_immediate_and_permanent(self) -> None:
        state = self.start()
        _, state = self.step(state)
        stop = np.zeros(79, dtype=np.float64)
        stop[38] = 20.0 / 8.0
        action, state = self.step(state, stop)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertEqual(state.phase, "safety_latched")
        action, state = self.step(state)
        np.testing.assert_array_equal(action, np.zeros(6))
        self.assertTrue(self.provider.state_telemetry(state)["safety_latched"])

    def test_invalid_observations_and_foreign_state_are_rejected(self) -> None:
        state = self.provider.reset()
        with self.assertRaises(ValueError):
            self.provider.act(np.zeros(78), state)
        invalid = self.valid.copy()
        invalid[7] = np.nan
        with self.assertRaises(ValueError):
            self.provider.act(invalid, state)
        with self.assertRaises(TypeError):
            self.provider.act(np.asarray(self.valid, dtype=np.complex128), state)
        with self.assertRaises(TypeError):
            self.provider.act({"observation79": self.valid, "case_id": "positive"}, state)
        with self.assertRaises(TypeError):
            self.provider.act(self.valid, {})

    def test_payload_sha_and_telemetry_are_canonical_and_json_serializable(self) -> None:
        payload = self.provider.canonical_payload()
        self.assertEqual(payload["direction_order"], ["positive_z", "negative_z"])
        self.assertEqual(payload["command_depth_onset_m"], 0.0013)
        self.assertEqual(payload["single_repeat_ticks"], 57)
        self.assertEqual(payload["waveform_repeat_count"], 1)
        self.assertEqual(payload["diagnostic_episode_repeat_count"], 2)
        self.assertEqual(len(payload["waveform_offsets_m"]), 57)
        self.assertTrue(payload["contact_return_starts_on_final_observation"])
        self.assertRegex(self.provider.controller_sha256(), r"^[0-9a-f]{64}$")
        self.assertEqual(self.provider.controller_sha256(), ActiveZProbeProvider().controller_sha256())
        json.dumps(payload, allow_nan=False)
        json.dumps(self.provider.state_telemetry(self.provider.reset()), allow_nan=False)

    def test_factory_and_provider_api_have_no_case_or_sign_input(self) -> None:
        self.assertEqual(tuple(inspect.signature(make_active_z_probe_provider).parameters), ())
        self.assertEqual(tuple(inspect.signature(ActiveZProbeProvider).parameters), ())
        provider = make_active_z_probe_provider()
        self.assertFalse(hasattr(provider, "case_id"))
        self.assertFalse(hasattr(provider, "sign"))


if __name__ == "__main__":
    unittest.main()
