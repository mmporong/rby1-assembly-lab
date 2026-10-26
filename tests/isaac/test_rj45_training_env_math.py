import unittest

import torch

from rby1_assembly.rj45_training_env import (
    _calibration_bias_case_seed,
    _warmup_motion_status,
    _warmup_sensor_status,
    _warmup_timeout_mask,
)


STABILITY_LIMITS = {
    "max_object_linear_speed_m_s": 0.005,
    "max_object_angular_speed_rad_s": 0.05,
    "max_revolute_joint_speed_rad_s": 0.05,
    "max_prismatic_joint_speed_m_s": 0.005,
    "max_revolute_tracking_error_rad": 0.05,
    "max_prismatic_tracking_error_m": 0.005,
}


class Rj45TrainingEnvironmentMathTests(unittest.TestCase):
    def test_bias_case_seed_is_stable_and_separates_episode_and_environment(self):
        self.assertEqual(_calibration_bias_case_seed(17, 0, 0), 45_123_034)
        self.assertEqual(
            _calibration_bias_case_seed(17, 0, 1) - _calibration_bias_case_seed(17, 0, 0),
            1_000_003,
        )
        self.assertEqual(
            _calibration_bias_case_seed(17, 4, 2) - _calibration_bias_case_seed(17, 0, 2),
            4,
        )

    def test_bias_case_seed_rejects_boolean_negative_and_noninteger_inputs(self):
        for values in ((True, 0, 0), (0, -1, 0), (0, 0, 1.5)):
            with self.assertRaises(ValueError):
                _calibration_bias_case_seed(*values)

    def test_warmup_sensor_status_rejects_nonfinite_and_overlimit_rows(self):
        fingers = torch.tensor(
            [
                [0.6, 0.6, 0.6, 0.6],
                [float("nan"), 0.7, 0.7, 0.7],
                [3.0] * 4,
                [0.7, 0.7, 0.7, 0.7],
            ]
        )
        insertion_force = torch.zeros((4, 3))
        insertion_load = torch.tensor([1.0, 1.0, 1.0, 1.0])
        forbidden = torch.tensor([0.0, 0.0, 0.0, 0.3])
        finite, within, grip_live = _warmup_sensor_status(
            fingers,
            insertion_force,
            insertion_load,
            forbidden,
            grip_contact_force_n=0.7,
            hard_contact_force_n=2.5,
            forbidden_contact_force_n=0.2,
        )
        self.assertEqual(finite.tolist(), [True, False, True, True])
        self.assertEqual(within.tolist(), [True, False, False, False])
        self.assertEqual(grip_live.tolist(), [False, False, True, True])

        insertion_force[0, 1] = float("inf")
        insertion_load[1] = float("nan")
        forbidden[2] = float("nan")
        finite, _, _ = _warmup_sensor_status(
            fingers,
            insertion_force,
            insertion_load,
            forbidden,
            grip_contact_force_n=0.7,
            hard_contact_force_n=2.5,
            forbidden_contact_force_n=0.2,
        )
        self.assertEqual(finite.tolist(), [False, False, False, True])

    def test_warmup_motion_rejects_common_mode_object_motion(self):
        joint_velocity = torch.zeros((2, 4))
        reference_error = torch.zeros((2, 4))
        prismatic = torch.tensor([False, False, True, True])
        socket_velocity = torch.zeros((2, 6))
        plug_velocity = torch.zeros((2, 6))
        socket_velocity[1, 0] = 0.02
        plug_velocity[1, 0] = 0.02
        finite, settled = _warmup_motion_status(
            joint_velocity,
            reference_error,
            prismatic,
            socket_velocity,
            plug_velocity,
            torch.zeros((2, 3)),
            torch.zeros((2, 3)),
            limits=STABILITY_LIMITS,
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertEqual(finite.tolist(), [True, True])
        self.assertEqual(settled.tolist(), [True, False])

    def test_warmup_motion_applies_joint_unit_specific_limits(self):
        joint_velocity = torch.zeros((4, 4))
        reference_error = torch.zeros((4, 4))
        joint_velocity[1, 0] = 0.051
        reference_error[2, 2] = 0.0051
        socket_velocity = torch.zeros((4, 6))
        socket_velocity[3, 0] = float("nan")
        finite, settled = _warmup_motion_status(
            joint_velocity,
            reference_error,
            torch.tensor([False, False, True, True]),
            socket_velocity,
            torch.zeros((4, 6)),
            torch.zeros((4, 3)),
            torch.zeros((4, 3)),
            limits=STABILITY_LIMITS,
            max_relative_linear_speed_m_s=0.005,
            max_relative_angular_speed_rad_s=0.05,
        )
        self.assertEqual(finite.tolist(), [True, True, True, False])
        self.assertEqual(settled.tolist(), [True, False, False, False])

    def test_warmup_timeout_does_not_reject_a_boundary_step_pass(self):
        timed_out = _warmup_timeout_mask(
            pending=torch.tensor([True, True, False]),
            passed=torch.tensor([True, False, False]),
            warmup_steps=torch.tensor([50, 50, 50]),
            timeout_steps=50,
        )
        self.assertEqual(timed_out.tolist(), [False, True, False])


if __name__ == "__main__":
    unittest.main()
