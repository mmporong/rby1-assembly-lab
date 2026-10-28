import math
import unittest
from contextlib import contextmanager

import torch

from rby1_assembly.training_telemetry import (
    action_clip_telemetry,
    make_action_telemetry_wrapper_type,
    observation_noise_contract,
)


BIAS = {
    "translation_levels_m": [0.00015, 0.00035, 0.0006],
    "rotation_max_levels_rad": [0.004, 0.008, 0.014],
}


class ObservationNoiseContractTests(unittest.TestCase):
    def test_rj45_separates_unused_base_gaussian_from_actual_episode_bias(self):
        contract = observation_noise_contract(
            task="rj45",
            configured_position_std_m=0.0015,
            configured_rotation_vector_std_rad=math.pi / 180.0,
            calibration_bias=BIAS,
            bias_level=1,
            calibration_bias_enabled=True,
        )

        self.assertFalse(contract["configured_base_gaussian_applied_to_policy"])
        self.assertEqual(
            contract["configured_base_gaussian"]["position_std_m"], 0.0015
        )
        actual = contract["actual_policy_observation_noise"]
        self.assertTrue(actual["enabled"])
        self.assertEqual(actual["temporal_scope"], "fixed_for_episode")
        self.assertEqual(actual["configured_translation_max_m"], 0.00035)
        self.assertEqual(actual["configured_rotation_vector_max_rad"], 0.008)
        self.assertEqual(actual["applied_translation_max_m"], 0.00035)
        self.assertEqual(actual["applied_rotation_vector_max_rad"], 0.008)

    def test_oracle_records_disabled_actual_bias_but_keeps_configured_level(self):
        contract = observation_noise_contract(
            task="rj45",
            configured_position_std_m=0.0015,
            configured_rotation_vector_std_rad=math.pi / 180.0,
            calibration_bias=BIAS,
            bias_level=2,
            calibration_bias_enabled=False,
        )

        actual = contract["actual_policy_observation_noise"]
        self.assertFalse(actual["enabled"])
        self.assertEqual(actual["distribution"], "disabled")
        self.assertEqual(actual["configured_bias_level"], 2)
        self.assertEqual(actual["configured_translation_max_m"], 0.0006)
        self.assertEqual(actual["configured_rotation_vector_max_rad"], 0.014)
        self.assertEqual(actual["applied_translation_max_m"], 0.0)
        self.assertEqual(actual["applied_rotation_vector_max_rad"], 0.0)

    def test_toy_records_configured_gaussian_as_actual_per_observation_noise(self):
        contract = observation_noise_contract(
            task="toy",
            configured_position_std_m=0.0015,
            configured_rotation_vector_std_rad=0.01,
        )

        self.assertTrue(contract["configured_base_gaussian_applied_to_policy"])
        self.assertEqual(
            contract["actual_policy_observation_noise"]["temporal_scope"],
            "resampled_each_observation",
        )
        self.assertEqual(
            contract["actual_policy_observation_noise"]["position_std_m"],
            0.0015,
        )


class ActionClipTelemetryTests(unittest.TestCase):
    def test_partial_active_batch_is_not_diluted_by_inactive_rows(self):
        actions = torch.tensor([[2.0, 0.5], [0.0, 0.0], [-3.0, 0.25]])
        metrics = action_clip_telemetry(
            actions,
            clip_actions=1.0,
            policy_active=torch.tensor([True, False, True]),
        )

        self.assertEqual(metrics["ppo_action/all_row_count"], 3.0)
        self.assertEqual(metrics["ppo_action/active_row_count"], 2.0)
        self.assertAlmostEqual(
            metrics["ppo_action/all_element_clip_fraction"], 2.0 / 6.0
        )
        self.assertAlmostEqual(
            metrics["ppo_action/active_element_clip_fraction"], 2.0 / 4.0
        )
        self.assertEqual(metrics["ppo_action/active_raw_abs_max"], 3.0)
        self.assertEqual(metrics["ppo_action/active_wrapper_clipped_abs_max"], 1.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_empty_active_batch_logs_finite_zeros(self):
        metrics = action_clip_telemetry(
            torch.tensor([[0.25, -0.25]]),
            clip_actions=1.0,
            policy_active=torch.tensor([False]),
        )

        self.assertEqual(metrics["ppo_action/active_row_count"], 0.0)
        self.assertEqual(metrics["ppo_action/active_element_clip_fraction"], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_nonfinite_raw_action_is_rejected_before_wrapper_clipping(self):
        with self.assertRaisesRegex(
            ValueError,
            "raw action contains NaN or Infinity before actual RslRlVecEnvWrapper clipping",
        ):
            action_clip_telemetry(
                torch.tensor([[float("nan"), float("inf")]]),
                clip_actions=1.0,
                policy_active=torch.tensor([True]),
            )

    def test_zero_and_disabled_clip_have_defined_behavior(self):
        actions = torch.tensor([[0.0, 2.0, -1.0]])
        zero = action_clip_telemetry(
            actions, clip_actions=0.0, policy_active=torch.tensor([True])
        )
        disabled = action_clip_telemetry(
            actions, clip_actions=None, policy_active=torch.tensor([True])
        )

        self.assertAlmostEqual(
            zero["ppo_action/active_element_clip_fraction"], 2.0 / 3.0
        )
        self.assertEqual(zero["ppo_action/active_wrapper_clipped_abs_max"], 0.0)
        self.assertEqual(disabled["ppo_action/active_element_clip_fraction"], 0.0)
        self.assertEqual(
            disabled["ppo_action/active_wrapper_clipped_abs_max"], 2.0
        )

    def test_large_finite_float32_actions_do_not_overflow_statistics(self):
        metrics = action_clip_telemetry(
            torch.tensor([[3.0e38, 3.0e38]], dtype=torch.float32),
            clip_actions=1.0,
            policy_active=torch.tensor([True]),
        )

        self.assertAlmostEqual(
            float(metrics["ppo_action/active_raw_mean"]), 3.0e38, delta=3.0e31
        )
        self.assertEqual(float(metrics["ppo_action/active_raw_std"]), 0.0)
        self.assertTrue(all(math.isfinite(value) for value in metrics.values()))

    def test_wrapper_preserves_base_outputs_and_passes_raw_actions_once(self):
        class Env:
            _training_last_policy_observation_valid = torch.tensor([True, False])

        class BaseWrapper:
            def __init__(self, env, clip_actions=None):
                self.env = env
                self.clip_actions = clip_actions
                self.received_actions = None
                self.applied_actions = None
                self.obs = object()
                self.rew = object()
                self.done = object()
                self.extras = {"log": {"existing": 7.0}}

            def step(self, actions):
                self.received_actions = actions
                self.applied_actions = actions.clamp(
                    -self.clip_actions, self.clip_actions
                )
                return self.obs, self.rew, self.done, self.extras

        Wrapper = make_action_telemetry_wrapper_type(BaseWrapper)
        wrapped = Wrapper(Env(), clip_actions=1.0)
        raw = torch.tensor([[2.0, -0.5], [9.0, -9.0]])
        result = wrapped.step(raw)

        self.assertIs(wrapped.received_actions, raw)
        self.assertTrue(torch.equal(wrapped.applied_actions, raw.clamp(-1.0, 1.0)))
        self.assertIs(result[0], wrapped.obs)
        self.assertIs(result[1], wrapped.rew)
        self.assertIs(result[2], wrapped.done)
        self.assertIs(result[3], wrapped.extras)
        self.assertEqual(result[3]["log"]["existing"], 7.0)
        self.assertEqual(
            result[3]["log"]["ppo_action/active_element_clip_fraction"], 0.5
        )
        self.assertEqual(
            result[3]["log"]["ppo_action/all_element_clip_fraction"], 0.75
        )

    def test_wrapper_rejects_nonfinite_action_before_base_wrapper_step(self):
        class BaseWrapper:
            def __init__(self, env, clip_actions=None):
                self.env = env
                self.called = False

            def step(self, actions):
                self.called = True
                return None, None, None, {}

        Wrapper = make_action_telemetry_wrapper_type(BaseWrapper)
        wrapped = Wrapper(object(), clip_actions=1.0)

        with self.assertRaisesRegex(
            ValueError,
            "before actual RslRlVecEnvWrapper clipping",
        ):
            wrapped.step(torch.tensor([[float("nan")]]))
        self.assertFalse(wrapped.called)

    def test_step_observer_wraps_base_step_without_changing_return(self):
        events = []

        class Observer:
            @contextmanager
            def control_step(self):
                events.append("enter")
                yield
                events.append("exit")

            def finalize_control_step(self, extras):
                events.append(("finalize", extras))

        class BaseWrapper:
            def __init__(self, env, clip_actions=None):
                self.env = env
                self.result = (object(), object(), object(), {"log": {}})

            def step(self, actions):
                events.append("base")
                return self.result

        Wrapper = make_action_telemetry_wrapper_type(BaseWrapper)
        wrapped = Wrapper(object(), clip_actions=1.0, step_observer=Observer())
        result = wrapped.step(torch.zeros((1, 2)))

        self.assertIs(result, wrapped.result)
        self.assertEqual(events[:3], ["enter", "base", "exit"])
        self.assertEqual(events[3], ("finalize", wrapped.result[-1]))


if __name__ == "__main__":
    unittest.main()
