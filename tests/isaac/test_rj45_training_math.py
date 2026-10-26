import math
import unittest

import torch

from rby1_assembly.rj45_training import (
    apply_relative_pose_bias,
    build_policy_observation,
    expand_residual_actions,
    normalized_insertion_potential,
    residual_reward,
)


def config():
    return {
        "reward": {
            "depth_scale_m": 0.012,
            "lateral_scale_m": 0.0005,
            "angle_scale_rad": math.pi / 180.0,
            "force_scale_n": 8.0,
            "force_penalty": 0.01,
            "action_delta_penalty": 0.0001,
            "success_bonus": 10.0,
            "failure_penalty": 3.0,
            "normalized_error_cap": 20.0,
        }
    }


class Rj45TrainingMathTests(unittest.TestCase):
    def test_right_only_actions_expand_to_right_slots(self):
        action = torch.arange(12, dtype=torch.float64).reshape(2, 6)
        expanded = expand_residual_actions(action, "right_only_6d")
        torch.testing.assert_close(expanded[:, :6], torch.zeros(2, 6, dtype=torch.float64))
        torch.testing.assert_close(expanded[:, 6:], action)
        both = torch.arange(24, dtype=torch.float64).reshape(2, 12)
        self.assertIs(expand_residual_actions(both, "both_arms_12d"), both)

    def test_fixed_pose_bias_identity_and_known_rotation(self):
        position = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        zero = torch.zeros((1, 3), dtype=torch.float64)
        identity_position, identity_quaternion = apply_relative_pose_bias(
            position, quaternion, zero, zero
        )
        torch.testing.assert_close(identity_position, position, atol=1.0e-14, rtol=0)
        torch.testing.assert_close(identity_quaternion, quaternion, atol=1.0e-14, rtol=0)
        biased_position, biased_quaternion = apply_relative_pose_bias(
            position,
            quaternion,
            torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0, math.pi / 2]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            biased_position,
            torch.tensor([[0.1, 0.8, 0.3]], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0,
        )
        torch.testing.assert_close(
            biased_quaternion,
            torch.tensor([[math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0,
        )
        self.assertAlmostEqual(float(torch.linalg.vector_norm(biased_quaternion)), 1.0, places=14)

    def test_potential_rewards_depth_and_caps_pose_errors(self):
        position = torch.tensor(
            [[0.0, 0.012, 0.0], [1.0, 0.024, 1.0]], dtype=torch.float64
        )
        identity = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=torch.float64
        )
        potential = normalized_insertion_potential(position, identity, config())
        self.assertAlmostEqual(float(potential[0]), 1.0, places=12)
        self.assertAlmostEqual(float(potential[1]), 1.1 - 20.0 - 20.0, places=12)

    def test_reward_shapes_only_active_but_terminal_terms_always_apply(self):
        previous = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        following = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float64)
        force = torch.tensor([8.0, 8.0, 0.0], dtype=torch.float64)
        delta = torch.ones((3, 6), dtype=torch.float64)
        active = torch.tensor([True, False, False])
        success = torch.tensor([False, True, False])
        failed = torch.tensor([False, False, True])
        reward = residual_reward(
            previous, following, force, delta, active, success, failed, config()
        )
        self.assertAlmostEqual(float(reward[0]), 1.0 - 0.01 - 0.0006, places=12)
        self.assertAlmostEqual(float(reward[1]), 10.0, places=12)
        self.assertAlmostEqual(float(reward[2]), -3.0, places=12)

    def test_observation_is_79d_and_uses_declared_normalization(self):
        batch = 2
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch)
        phase = torch.zeros((batch, 8))
        phase[:, 3] = 1.0
        observation = build_policy_observation(
            estimated_relative_position_entry=torch.tensor(
                [[0.001, 0.012, -0.002]] * batch
            ),
            estimated_relative_quaternion_entry_wxyz=identity,
            arm_joint_position_rad=torch.full((batch, 14), math.pi),
            arm_joint_velocity_rad_s=torch.full((batch, 14), -math.pi),
            grip_force_n=torch.full((batch, 2), 8.0),
            entry_contact_force_n=torch.full((batch, 3), 4.0),
            grip_proven=torch.ones((batch, 2)),
            phase_one_hot=phase,
            previous_expanded_action=torch.full((batch, 12), 0.25),
            command_depth_m=torch.full((batch, 1), 0.006),
            insertion_advance=torch.ones((batch, 1)),
            insertion_joint_reference_error_rad=torch.full((batch, 14), 0.03),
            policy_active=torch.ones((batch, 1)),
        )
        self.assertEqual(tuple(observation.shape), (batch, 79))
        torch.testing.assert_close(observation[:, 0:3], torch.tensor([[1.0, 1.0, -2.0]] * batch))
        torch.testing.assert_close(observation[:, 3:7], identity)
        torch.testing.assert_close(observation[:, 7:21], torch.ones((batch, 14)))
        torch.testing.assert_close(observation[:, 21:35], -torch.ones((batch, 14)))
        torch.testing.assert_close(observation[:, 35:37], torch.ones((batch, 2)))
        torch.testing.assert_close(observation[:, 37:40], torch.full((batch, 3), 0.5))
        torch.testing.assert_close(observation[:, 62:64], torch.tensor([[0.5, 1.0]] * batch))
        torch.testing.assert_close(observation[:, 64:79], torch.ones((batch, 15)))

    def test_observation_rejects_non_one_hot_phase_and_nonunit_estimate(self):
        values = {
            "estimated_relative_position_entry": torch.zeros((1, 3)),
            "estimated_relative_quaternion_entry_wxyz": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "arm_joint_position_rad": torch.zeros((1, 14)),
            "arm_joint_velocity_rad_s": torch.zeros((1, 14)),
            "grip_force_n": torch.zeros((1, 2)),
            "entry_contact_force_n": torch.zeros((1, 3)),
            "grip_proven": torch.zeros((1, 2)),
            "phase_one_hot": torch.zeros((1, 8)),
            "previous_expanded_action": torch.zeros((1, 12)),
            "command_depth_m": torch.zeros((1, 1)),
            "insertion_advance": torch.zeros((1, 1)),
            "insertion_joint_reference_error_rad": torch.zeros((1, 14)),
            "policy_active": torch.zeros((1, 1)),
        }
        with self.assertRaisesRegex(ValueError, "one active phase"):
            build_policy_observation(**values)
        values["phase_one_hot"][0, 0] = 1.0
        values["estimated_relative_quaternion_entry_wxyz"][0, 0] = 0.9
        with self.assertRaisesRegex(ValueError, "unit length"):
            build_policy_observation(**values)


if __name__ == "__main__":
    unittest.main()
