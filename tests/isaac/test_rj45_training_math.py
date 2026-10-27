import math
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from rby1_assembly.rj45_training import (
    apply_relative_pose_bias,
    apply_tip_target_pose_offset,
    append_persistent_calibration_bias_diagnostic,
    build_policy_observation,
    configure_actor_output_initialization,
    expand_residual_actions,
    keypoint_insertion_potential,
    load_training_config,
    normalized_insertion_potential,
    residual_reward,
    residual_reward_components,
)


ROOT = Path(__file__).resolve().parents[2]


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
    def test_zero_actor_output_initialization_is_explicit_and_mean_only(self):
        torch.manual_seed(20261027)
        actor = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.ELU(),
            torch.nn.Linear(8, 6),
        )
        runner = SimpleNamespace(
            alg=SimpleNamespace(policy=SimpleNamespace(actor=actor))
        )
        hidden_weight = actor[0].weight.detach().clone()
        original_weight = actor[-1].weight.detach().clone()
        default_metadata = configure_actor_output_initialization(
            runner, expected_output_dim=6, zero_initialize=False
        )
        torch.testing.assert_close(actor[-1].weight, original_weight)
        self.assertEqual(default_metadata["mode"], "rsl_rl_default")
        self.assertFalse(default_metadata["zero_actor_output_layer"])

        zero_metadata = configure_actor_output_initialization(
            runner, expected_output_dim=6, zero_initialize=True
        )
        torch.testing.assert_close(actor[0].weight, hidden_weight)
        self.assertEqual(zero_metadata["mode"], "residual_actor_final_linear_zero")
        self.assertTrue(zero_metadata["weight_all_zero_after_initialization"])
        self.assertTrue(zero_metadata["bias_all_zero_after_initialization"])
        self.assertTrue(zero_metadata["deterministic_actor_output_all_zero"])
        self.assertFalse(zero_metadata["stochastic_policy_noise_modified"])
        samples = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0], [1.0, -1.0, 2.0, -2.0]]
        )
        torch.testing.assert_close(actor(samples), torch.zeros((2, 6)))
        self.assertNotEqual(
            zero_metadata["pre_weight_sha256"],
            zero_metadata["post_weight_sha256"],
        )

    def test_zero_actor_output_initialization_rejects_wrong_actor_contract(self):
        wrong_output = SimpleNamespace(
            alg=SimpleNamespace(
                policy=SimpleNamespace(
                    actor=torch.nn.Sequential(torch.nn.Linear(4, 12))
                )
            )
        )
        with self.assertRaisesRegex(ValueError, "6 outputs"):
            configure_actor_output_initialization(
                wrong_output, expected_output_dim=6, zero_initialize=True
            )
        nonsequential = SimpleNamespace(
            alg=SimpleNamespace(policy=SimpleNamespace(actor=torch.nn.Linear(4, 6)))
        )
        with self.assertRaisesRegex(ValueError, "Sequential"):
            configure_actor_output_initialization(
                nonsequential, expected_output_dim=6, zero_initialize=True
            )

    def test_tip_target_offset_uses_entry_frame_and_preserves_tip_pivot(self):
        half = math.sqrt(0.5)
        body_position = torch.tensor([[0.4, -0.2, 0.8]], dtype=torch.float64)
        body_quaternion = torch.tensor([[half, 0.0, 0.0, half]], dtype=torch.float64)
        entry_quaternion = torch.tensor([[half, half, 0.0, 0.0]], dtype=torch.float64)
        tip_local = torch.tensor([[0.0, 0.03, 0.01]], dtype=torch.float64)
        tip_local_quaternion = torch.tensor([[half, 0.0, half, 0.0]], dtype=torch.float64)
        offset = torch.tensor([[0.01, 0.02, -0.03, 0.0, 0.0, math.pi / 2]], dtype=torch.float64)

        new_position, new_quaternion = apply_tip_target_pose_offset(
            body_position,
            body_quaternion,
            entry_quaternion,
            tip_local,
            tip_local_quaternion,
            offset,
        )
        zero_position, zero_quaternion = apply_tip_target_pose_offset(
            body_position,
            body_quaternion,
            entry_quaternion,
            tip_local,
            tip_local_quaternion,
            torch.zeros_like(offset),
        )
        torch.testing.assert_close(zero_position, body_position, atol=1e-12, rtol=0)
        torch.testing.assert_close(zero_quaternion, body_quaternion, atol=1e-12, rtol=0)
        self.assertEqual(tuple(new_position.shape), (1, 3))
        self.assertEqual(tuple(new_quaternion.shape), (1, 4))
        self.assertAlmostEqual(torch.linalg.vector_norm(new_quaternion).item(), 1.0, places=12)

        # Reconstructed target tip position equals old tip plus entry-frame translation.
        from rby1_assembly.rj45_training import _quaternion_rotate_wxyz
        old_tip = body_position + _quaternion_rotate_wxyz(body_quaternion, tip_local)
        new_tip = new_position + _quaternion_rotate_wxyz(new_quaternion, tip_local)
        expected_tip = old_tip + _quaternion_rotate_wxyz(entry_quaternion, offset[:, :3])
        torch.testing.assert_close(new_tip, expected_tip, atol=1e-12, rtol=0)
    def test_privileged_bias_diagnostic_appends_normalized_sampled_parameter(self):
        base = torch.arange(2 * 79, dtype=torch.float64).reshape(2, 79)
        translation = torch.tensor([[0.00015, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        rotation = torch.tensor([[0.0, -0.004, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        diagnostic = append_persistent_calibration_bias_diagnostic(
            base,
            translation,
            rotation,
            translation_max_m=0.00015,
            rotation_max_rad=0.004,
            mode="persistent_calibration_bias",
        )
        self.assertEqual(tuple(diagnostic.shape), (2, 85))
        torch.testing.assert_close(diagnostic[:, :79], base)
        torch.testing.assert_close(
            diagnostic[:, 79:],
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, -1.0, 0.0], [0.0] * 6],
                dtype=torch.float64,
            ),
        )
        standard = append_persistent_calibration_bias_diagnostic(
            base,
            translation,
            rotation,
            translation_max_m=0.00015,
            rotation_max_rad=0.004,
            mode="standard",
        )
        self.assertIs(standard, base)

    def test_privileged_bias_diagnostic_rejects_out_of_contract_parameters(self):
        base = torch.zeros((1, 79))
        zero = torch.zeros((1, 3))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            append_persistent_calibration_bias_diagnostic(
                base,
                torch.tensor([[0.0002, 0.0, 0.0]]),
                zero,
                translation_max_m=0.00015,
                rotation_max_rad=0.004,
                mode="persistent_calibration_bias",
            )
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

    def test_a0_reward_is_bitwise_equal_to_pre_v2_arithmetic(self):
        for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
            generator = torch.Generator(device=device).manual_seed(20261027)
            previous = torch.randn(257, generator=generator, device=device)
            following = torch.randn(257, generator=generator, device=device)
            force = torch.randn((257, 3), generator=generator, device=device)
            delta = torch.randn((257, 6), generator=generator, device=device)
            active = torch.rand(257, generator=generator, device=device) > 0.4
            success = torch.rand(257, generator=generator, device=device) > 0.9
            failed = (torch.rand(257, generator=generator, device=device) > 0.9) & ~success
            reward_config = config()["reward"]
            force_cost = reward_config["force_penalty"] * (
                torch.linalg.vector_norm(force, dim=-1).abs()
                / reward_config["force_scale_n"]
            ).square()
            action_cost = reward_config["action_delta_penalty"] * delta.square().sum(dim=-1)
            shaped = following - previous - force_cost - action_cost
            reference = torch.where(active, shaped, torch.zeros_like(shaped))
            reference = (
                reference
                + success.to(reference.dtype) * reward_config["success_bonus"]
                - failed.to(reference.dtype) * reward_config["failure_penalty"]
            )
            actual = residual_reward(
                previous, following, force, delta, active, success, failed, config()
            )
            self.assertTrue(torch.equal(actual, reference), device)

    def test_discounted_reward_distinguishes_true_terminal_and_truncation(self):
        a1 = load_training_config(
            ROOT / "configs/rj45-training-target-offset-a1-discounted-pose-v2.json"
        )
        previous = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=torch.float64)
        following = torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=torch.float64)
        active = torch.ones(4, dtype=torch.bool)
        false = torch.zeros(4, dtype=torch.bool)
        true_terminated = torch.tensor([False, True, False, True])
        truncated = torch.tensor([False, False, True, True])
        failed = torch.tensor([False, True, True, True])
        components = residual_reward_components(
            previous,
            following,
            torch.zeros(4, dtype=torch.float64),
            torch.zeros((4, 6), dtype=torch.float64),
            active,
            false,
            failed,
            true_terminated,
            truncated,
            a1,
        )
        self.assertAlmostEqual(float(components["potential_progress"][0]), 0.985, places=12)
        self.assertAlmostEqual(float(components["potential_progress"][1]), -2.0, places=12)
        self.assertAlmostEqual(float(components["potential_progress"][2]), -2.0, places=12)
        self.assertAlmostEqual(float(components["potential_progress"][3]), -2.0, places=12)
        self.assertAlmostEqual(float(components["terminal"][2]), -3.0, places=12)

    def test_keypoint_potential_is_quaternion_sign_invariant_and_batch_isolated(self):
        a2 = load_training_config(
            ROOT / "configs/rj45-training-target-offset-a2-keypoint-shape-v2.json"
        )
        position = torch.tensor(
            [[0.0, 0.012, 0.0], [0.0005, 0.012, 0.0]], dtype=torch.float64
        )
        q = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], dtype=torch.float64
        )
        potential = keypoint_insertion_potential(position, q, a2)
        self.assertAlmostEqual(float(potential[0]), 1.0, places=12)
        self.assertLess(float(potential[1]), float(potential[0]))
        isolated = keypoint_insertion_potential(position[:1], -q[:1], a2)
        torch.testing.assert_close(isolated, potential[:1], atol=1e-12, rtol=0)

    def test_keypoint_rotation_changes_corner_alignment_in_metres(self):
        a2 = load_training_config(
            ROOT / "configs/rj45-training-target-offset-a2-keypoint-shape-v2.json"
        )
        position = torch.tensor([[0.0, 0.012, 0.0]] * 2, dtype=torch.float64)
        quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0]],
            dtype=torch.float64,
        )
        potential = keypoint_insertion_potential(position, quaternion, a2)
        self.assertGreater(float(potential[0]), float(potential[1]))

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
