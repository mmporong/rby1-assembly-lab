import math
import unittest

import torch

from rby1_assembly.rj45_teacher import teacher_tip_target_action
from rby1_assembly.rj45_training import apply_tip_target_pose_offset


def _quat_z(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)], dtype=torch.float64)


class Rj45TeacherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.position = torch.tensor([[0.4, -0.2, 0.8]], dtype=torch.float64)
        self.identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
        self.tip = torch.tensor([[0.01, 0.03, -0.02]], dtype=torch.float64)
        self.kwargs = {
            "baseline_body_position_world": self.position,
            "baseline_body_quaternion_world_wxyz": self.identity,
            "socket_entry_quaternion_world_wxyz": self.identity,
            "plug_tip_position_local": self.tip,
            "plug_tip_quaternion_local_wxyz": self.identity,
            "translation_bound_m": 0.01,
            "rotation_bound_rad": 0.2,
        }

    def teacher(self, desired_position=None, desired_quaternion=None, **extra):
        arguments = dict(
            desired_body_position_world=(
                self.position if desired_position is None else desired_position
            ),
            desired_body_quaternion_world_wxyz=(
                self.identity if desired_quaternion is None else desired_quaternion
            ),
            **self.kwargs,
        )
        arguments.update(extra)
        return teacher_tip_target_action(**arguments)

    def test_translation_rotation_and_combined_targets(self):
        translation = self.teacher(
            desired_position=self.position + torch.tensor([[0.005, 0.0, 0.0]])
        )
        torch.testing.assert_close(
            translation["raw_action_6d"],
            torch.tensor([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        )
        rotation = self.teacher(desired_quaternion=_quat_z(0.1).unsqueeze(0))
        torch.testing.assert_close(
            rotation["raw_action_6d"][..., 3:],
            torch.tensor([[0.0, 0.0, 0.5]], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0,
        )
        combined = self.teacher(
            desired_position=self.position + torch.tensor([[0.003, -0.004, 0.0]]),
            desired_quaternion=_quat_z(0.1).unsqueeze(0),
        )
        self.assertEqual(tuple(combined["bounded_action_6d"].shape), (1, 6))
        self.assertFalse(bool(combined["bound_limited_mask"][0]))

    def test_quaternion_sign_near_zero_and_pi_are_stable(self):
        for angle in (1.0e-10, math.pi):
            quaternion = _quat_z(angle).unsqueeze(0)
            positive = self.teacher(desired_quaternion=quaternion)
            negative = self.teacher(desired_quaternion=-quaternion)
            torch.testing.assert_close(
                positive["raw_action_6d"], negative["raw_action_6d"], atol=1.0e-10, rtol=0
            )
            self.assertTrue(bool(torch.isfinite(positive["raw_action_6d"]).all()))

    def test_component_clamp_then_separate_unit_ball_projection(self):
        desired_position, desired_quaternion = apply_tip_target_pose_offset(
            self.position,
            self.identity,
            self.identity,
            self.tip,
            self.identity,
            torch.tensor([[0.02, 0.02, 0.0, 0.0, 0.0, 0.4]], dtype=torch.float64),
        )
        result = self.teacher(
            desired_position=desired_position,
            desired_quaternion=desired_quaternion,
        )
        expected_translation = torch.tensor(
            [[math.sqrt(0.5), math.sqrt(0.5), 0.0]], dtype=torch.float64
        )
        torch.testing.assert_close(
            result["bounded_action_6d"][..., :3], expected_translation, atol=1.0e-12, rtol=0
        )
        torch.testing.assert_close(
            result["bounded_action_6d"][..., 3:],
            torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0,
        )
        self.assertTrue(bool(result["bound_limited_mask"][0]))

    def test_applied_offset_reconstructs_unbounded_desired_tip_target(self):
        desired_position = self.position + torch.tensor(
            [[0.003, -0.002, 0.001]], dtype=torch.float64
        )
        desired_quaternion = _quat_z(0.1).unsqueeze(0)
        result = self.teacher(
            desired_position=desired_position, desired_quaternion=desired_quaternion
        )
        reconstructed_position, reconstructed_quaternion = apply_tip_target_pose_offset(
            self.position,
            self.identity,
            self.identity,
            self.tip,
            self.identity,
            result["applied_offset_entry_6d"],
        )
        torch.testing.assert_close(reconstructed_position, desired_position, atol=1.0e-12, rtol=0)
        torch.testing.assert_close(
            reconstructed_quaternion, desired_quaternion, atol=1.0e-12, rtol=0
        )

    def test_active_mask_and_absolute_non_accumulating_contract(self):
        baseline = self.position.repeat(2, 1)
        identity = self.identity.repeat(2, 1)
        tip = self.tip.repeat(2, 1)
        desired = baseline + torch.tensor(
            [[0.002, 0.0, 0.0], [0.002, 0.0, 0.0]], dtype=torch.float64
        )
        args = dict(
            baseline_body_position_world=baseline,
            baseline_body_quaternion_world_wxyz=identity,
            desired_body_position_world=desired,
            desired_body_quaternion_world_wxyz=identity,
            socket_entry_quaternion_world_wxyz=identity,
            plug_tip_position_local=tip,
            plug_tip_quaternion_local_wxyz=identity,
            translation_bound_m=0.01,
            rotation_bound_rad=0.2,
            active_mask=torch.tensor([True, False]),
        )
        first = teacher_tip_target_action(**args)
        second = teacher_tip_target_action(**args)
        torch.testing.assert_close(first["bounded_action_6d"], second["bounded_action_6d"])
        torch.testing.assert_close(
            first["bounded_action_6d"][1], torch.zeros(6, dtype=torch.float64)
        )
        self.assertFalse(bool(first["bound_limited_mask"][1]))

    def test_rejects_nonfinite_shapes_quaternions_and_bounds(self):
        with self.assertRaisesRegex(ValueError, "last dimension 3"):
            self.teacher(desired_position=torch.zeros((1, 2), dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "finite"):
            self.teacher(desired_position=torch.tensor([[math.nan, 0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "unit length"):
            self.teacher(
                desired_quaternion=torch.tensor(
                    [[2.0, 0.0, 0.0, 0.0]], dtype=torch.float64
                )
            )
        with self.assertRaisesRegex(ValueError, "positive"):
            self.teacher(translation_bound_m=0.0)
        with self.assertRaisesRegex(ValueError, "active_mask"):
            self.teacher(active_mask=torch.ones(1))


if __name__ == "__main__":
    unittest.main()
