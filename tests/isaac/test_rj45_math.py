import math
import unittest

import numpy as np
import torch

from rby1_assembly.rj45_task import quaternion_rotate_wxyz, relative_mating_velocity
from rby1_assembly.rj45_env import (
    advance_joint_reference,
    bounded_joint_integral,
    bounded_residual_integral,
    com_velocity_to_link_velocity,
    evaluate_mating_stability_samples,
    nullspace_posture_step,
    physics_window_dwell_gate,
)


def torch_quaternion_rotate(quaternions: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
    scalar = quaternions[:, :1]
    xyz = quaternions[:, 1:]
    return (
        vectors * (2.0 * scalar.square() - 1.0)
        + 2.0 * scalar * torch.cross(xyz, vectors, dim=-1)
        + 2.0 * xyz * torch.sum(xyz * vectors, dim=-1, keepdim=True)
    )


class Rj45TorchCrossCheckTests(unittest.TestCase):
    def test_dwell_gate_requires_insert_complete_window_and_every_sample(self):
        stable = torch.tensor(
            [[True, True, False, True, True], [True, True, True, True, True]]
        )
        inserting = torch.tensor([True, False])
        complete = torch.tensor([True, True])
        gated = physics_window_dwell_gate(inserting, complete, stable)
        torch.testing.assert_close(gated, torch.tensor([False, False]))
        stable[0, 2] = True
        torch.testing.assert_close(
            physics_window_dwell_gate(inserting, complete, stable),
            torch.tensor([True, False]),
        )
        complete[0] = False
        torch.testing.assert_close(
            physics_window_dwell_gate(inserting, complete, stable),
            torch.tensor([False, False]),
        )

    def test_com_velocity_is_translated_to_link_origin(self):
        pose = torch.tensor(
            [[[0.0, 0.0, 0.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]]],
            dtype=torch.float64,
        )
        com_velocity = torch.tensor(
            [[[0.0, 0.0, 0.0, 0.0, 0.0, 2.0]]], dtype=torch.float64
        )
        com_position_body = torch.tensor([[[0.1, 0.0, 0.0]]], dtype=torch.float64)
        link_velocity = com_velocity_to_link_velocity(
            pose, com_velocity, com_position_body
        )
        torch.testing.assert_close(
            link_velocity,
            torch.tensor([[[0.2, 0.0, 0.0, 0.0, 0.0, 2.0]]], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0,
        )

    def test_substep_velocity_spike_rejects_stable_endpoint(self):
        sample_count = 5
        poses = torch.zeros((1, sample_count, 2, 7), dtype=torch.float64)
        poses[..., 3] = 1.0
        poses[:, :, 1, 1] = -0.033 + 0.0123 - 0.01137
        velocities = torch.zeros((1, sample_count, 2, 6), dtype=torch.float64)
        velocities[:, 2, 1, 0] = 0.006
        result = evaluate_mating_stability_samples(
            poses,
            velocities,
            torch.tensor([0.0, -0.033, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            torch.tensor([0.0, 0.01137, 0.0], dtype=torch.float64),
            torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64),
            min_depth_m=0.012,
            max_depth_m=0.013,
            max_lateral_error_m=0.00005,
            max_orientation_error_rad=math.radians(0.5),
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertTrue(bool(result["stable"][0, -1]))
        self.assertFalse(bool(result["stable"][0, 2]))
        self.assertFalse(bool(result["all_samples_stable"][0]))
        self.assertAlmostEqual(
            float(result["relative_linear_speed_m_s"][0, 2]), 0.006, places=12
        )
        self.assertAlmostEqual(float(result["depth_m"][0, -1]), 0.0123, places=12)

    def test_substep_nonfinite_pose_fails_closed(self):
        poses = torch.zeros((1, 2, 2, 7), dtype=torch.float64)
        poses[..., 3] = 1.0
        poses[:, :, 1, 1] = -0.033 + 0.0123 - 0.01137
        poses[0, 0, 1, 0] = float("nan")
        velocities = torch.zeros((1, 2, 2, 6), dtype=torch.float64)
        result = evaluate_mating_stability_samples(
            poses,
            velocities,
            [0.0, -0.033, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.01137, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            min_depth_m=0.012,
            max_depth_m=0.013,
            max_lateral_error_m=0.00005,
            max_orientation_error_rad=math.radians(0.5),
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertFalse(bool(result["finite"][0, 0]))
        self.assertFalse(bool(result["all_samples_stable"][0]))

    def test_rotated_rigid_pair_has_zero_relative_mating_velocity(self):
        quaternion = torch.tensor(
            [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)], dtype=torch.float64
        )
        socket_position = torch.tensor([0.3, -0.2, 0.8], dtype=torch.float64)
        entry_local = torch.tensor([0.0, -0.033, 0.0], dtype=torch.float64)
        tip_local = torch.tensor([0.0, 0.01137, 0.0], dtype=torch.float64)
        depth_local = torch.tensor([0.0, 0.0123, 0.0], dtype=torch.float64)
        entry_world = socket_position + torch_quaternion_rotate(
            quaternion[None], entry_local[None]
        )[0]
        tip_world = entry_world + torch_quaternion_rotate(
            quaternion[None], depth_local[None]
        )[0]
        plug_position = tip_world - torch_quaternion_rotate(
            quaternion[None], tip_local[None]
        )[0]
        poses = torch.stack(
            (
                torch.cat((socket_position, quaternion)),
                torch.cat((plug_position, quaternion)),
            )
        ).reshape(1, 1, 2, 7)
        angular = torch.tensor([0.0, 0.0, 0.4], dtype=torch.float64)
        socket_linear = torch.tensor([0.1, -0.05, 0.02], dtype=torch.float64)
        plug_linear = socket_linear + torch.cross(
            angular, plug_position - socket_position, dim=0
        )
        velocities = torch.stack(
            (
                torch.cat((socket_linear, angular)),
                torch.cat((plug_linear, angular)),
            )
        ).reshape(1, 1, 2, 6)
        result = evaluate_mating_stability_samples(
            poses,
            velocities,
            entry_local,
            [1.0, 0.0, 0.0, 0.0],
            tip_local,
            [1.0, 0.0, 0.0, 0.0],
            min_depth_m=0.012,
            max_depth_m=0.013,
            max_lateral_error_m=0.00005,
            max_orientation_error_rad=math.radians(0.5),
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertTrue(bool(result["all_samples_stable"][0]))
        self.assertAlmostEqual(float(result["depth_m"][0, 0]), 0.0123, places=12)
        self.assertLess(float(result["relative_linear_speed_m_s"][0, 0]), 1.0e-12)
        self.assertLess(float(result["relative_angular_speed_rad_s"][0, 0]), 1.0e-12)

    def test_policy_residual_is_not_accumulated_in_integral_reference(self):
        current = torch.zeros(1, 2)
        reference = current.clone()
        stiffness = torch.full_like(current, 1000)
        gravity = torch.zeros_like(current)
        effort = torch.full_like(current, 30)
        limits = torch.tensor([[[-1., 1.], [-1., 1.]]])
        for _ in range(100):
            reference, command = bounded_residual_integral(
                reference, current, torch.zeros_like(current), torch.full_like(current, .005),
                stiffness, gravity, effort, limits, .25, .03)
            torch.testing.assert_close(reference, torch.zeros_like(reference))
            torch.testing.assert_close(command, torch.full_like(command, .005))

    def test_integral_rejects_constant_load_bias_without_exceeding_effort(self):
        current = torch.zeros(1, 1, dtype=torch.float64)
        reference = current.clone()
        stiffness = torch.full_like(current, 1000)
        gravity = torch.zeros_like(current)
        cap = torch.full_like(current, 30)
        limits = torch.tensor([[[-1.0, 1.0]]], dtype=torch.float64)
        for _ in range(150):
            reference = bounded_joint_integral(reference, current, -current, stiffness, gravity,
                                                cap, limits, 0.25, 0.03)
            self.assertLessEqual(float(((reference - current) * stiffness).abs().max()), 30.000001)
            # Quasi-static plant has a 15 Nm constant disturbance. With only
            # current+error position commands it retains a nonzero bias.
            current = reference - 0.015
        self.assertLess(float(current.abs().max()), 1.0e-10)
        saturated = bounded_joint_integral(reference, current, torch.ones_like(current),
                                           stiffness, gravity, cap, limits, 1, 0.03)
        torch.testing.assert_close(saturated, current + 0.03)

    def test_posture_step_preserves_hand_twist_and_joint_step_bound(self):
        generator = torch.Generator().manual_seed(402)
        jacobian = torch.randn(8, 6, 7, generator=generator, dtype=torch.float64)
        current = torch.randn(8, 7, generator=generator, dtype=torch.float64)
        preferred = torch.zeros_like(current)
        delta = nullspace_posture_step(jacobian, current, preferred, 0.1, 0.01)
        self.assertGreater(float(delta.abs().max()), 0)
        self.assertLessEqual(float(delta.abs().max()), 0.0100000001)
        torch.testing.assert_close(jacobian @ delta.unsqueeze(-1), torch.zeros(8, 6, 1, dtype=torch.float64), atol=1.0e-12, rtol=0)

    def test_joint_reference_is_monotonic_and_waits_for_tracking(self):
        start = torch.zeros(3)
        goal = torch.tensor([0.9, -1.8, 0.9])
        progress = torch.tensor([0.5, 0.5, 0.5])
        current = start + progress[:, None] * goal
        current[1] -= 0.2
        enabled = torch.tensor([True, True, False])
        updated, target = advance_joint_reference(current, start, goal, progress, enabled, 0.015, 0.05)
        self.assertTrue(torch.all(updated >= progress))
        self.assertGreater(updated[0], progress[0])
        torch.testing.assert_close(updated[1:], progress[1:])
        self.assertLessEqual(float((target[0] - current[0]).abs().max()), 0.015001)
        # A lagging/bouncing joint freezes the stored reference instead of
        # moving the reference back to its measured position.
        bounced = current.clone(); bounced[0] -= 0.2
        held, _ = advance_joint_reference(bounced, start, goal, updated, enabled, 0.015, 0.05)
        torch.testing.assert_close(held, updated)

    def test_joint_reference_rejects_nonfinite_state(self):
        for bad in (float("nan"), float("inf")):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                advance_joint_reference(torch.tensor([[bad, 0.0]]), torch.zeros(2),
                                        torch.ones(2), torch.zeros(1), torch.ones(1, dtype=torch.bool), 0.01, 0.05)

    def test_numpy_rotation_matches_runtime_torch_formula(self):
        quaternions = torch.tensor(
            [
                [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)],
                [math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0],
            ],
            dtype=torch.float64,
        )
        vectors = torch.tensor([[0.01, 0.02, -0.03], [-0.04, 0.01, 0.02]], dtype=torch.float64)
        torch_rotated = torch_quaternion_rotate(quaternions, vectors)
        numpy_rotated = quaternion_rotate_wxyz(quaternions.numpy(), vectors.numpy())
        np.testing.assert_allclose(torch_rotated.numpy(), numpy_rotated, atol=1.0e-12)

    def test_numpy_relative_velocity_matches_runtime_torch_formula(self):
        socket_position = torch.tensor(
            [[0.3, -0.2, 0.8], [0.1, 0.4, 0.7]], dtype=torch.float64
        )
        plug_position = torch.tensor(
            [[0.32, -0.15, 0.81], [0.11, 0.45, 0.72]], dtype=torch.float64
        )
        socket_quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]],
            dtype=torch.float64,
        )
        plug_quaternion = torch.tensor(
            [[math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        socket_linear = torch.tensor([[0.1, -0.2, 0.05], [-0.1, 0.03, 0.02]], dtype=torch.float64)
        socket_angular = torch.tensor([[0.3, -0.1, 0.2], [0.05, 0.1, -0.2]], dtype=torch.float64)
        plug_linear = torch.tensor([[0.08, -0.17, 0.02], [-0.07, 0.02, 0.04]], dtype=torch.float64)
        plug_angular = torch.tensor([[0.28, -0.06, 0.25], [0.03, 0.14, -0.18]], dtype=torch.float64)
        entry_local = torch.tensor([[0.0, -0.033, 0.0005]] * 2, dtype=torch.float64)
        tip_local = torch.tensor([[0.0, 0.01137, 0.0]] * 2, dtype=torch.float64)

        entry_offset = torch_quaternion_rotate(socket_quaternion, entry_local)
        tip_offset = torch_quaternion_rotate(plug_quaternion, tip_local)
        entry_world = socket_position + entry_offset
        tip_world = plug_position + tip_offset
        entry_velocity = socket_linear + torch.cross(socket_angular, entry_offset, dim=1)
        tip_velocity = plug_linear + torch.cross(plug_angular, tip_offset, dim=1)
        torch_linear = tip_velocity - entry_velocity - torch.cross(
            socket_angular, tip_world - entry_world, dim=1
        )
        torch_angular = plug_angular - socket_angular
        numpy_linear, numpy_angular = relative_mating_velocity(
            socket_position.numpy(),
            socket_quaternion.numpy(),
            socket_linear.numpy(),
            socket_angular.numpy(),
            plug_position.numpy(),
            plug_quaternion.numpy(),
            plug_linear.numpy(),
            plug_angular.numpy(),
            entry_local.numpy(),
            tip_local.numpy(),
        )
        np.testing.assert_allclose(torch_linear.numpy(), numpy_linear, atol=1.0e-12)
        np.testing.assert_allclose(torch_angular.numpy(), numpy_angular, atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
