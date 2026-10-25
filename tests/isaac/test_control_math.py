import unittest

import torch

from rby1_assembly.control_math import continuous_goal_dwell, continuous_grasp_loss, damped_joint_step
from rby1_assembly.contracts import EvaluationSample, TaskLimits
from rby1_assembly.evaluation import TaskEvaluator


class GoalDwellTests(unittest.TestCase):
    def test_agrees_with_independent_evaluator_at_all_runtime_offsets(self):
        for offset in (0, 1322, 2651, 100_000_000):
            with self.subTest(offset=offset):
                start = torch.tensor([-1], dtype=torch.long)
                evaluator = TaskEvaluator(TaskLimits(dwell_s=0.4))
                for index in range(13):
                    start, dwell, complete = continuous_goal_dwell(
                        torch.tensor([True]), torch.tensor([offset + index]), start,
                        step_dt_s=1 / 30, required_dwell_s=0.4,
                    )
                    result = evaluator.update(EvaluationSample(
                        actual_depth_m=0.031, lateral_error_m=0.001,
                        orientation_error_rad=0.01, force_peak_n=6.0,
                        grasp_both=True, forbidden_contact=False, sim_time_s=index / 30,
                    ))
                    self.assertEqual(bool(complete.item()), result.success)
                    self.assertEqual(result.success, index == 12)
                self.assertAlmostEqual(float(dwell.item()), 0.4)

    def test_interruption_resets_only_affected_environment(self):
        start = torch.tensor([10, 10])
        start, dwell, complete = continuous_goal_dwell(
            torch.tensor([True, False]), torch.tensor([22, 22]), start,
            step_dt_s=1 / 30, required_dwell_s=0.4,
        )
        torch.testing.assert_close(start, torch.tensor([10, -1]))
        torch.testing.assert_close(complete, torch.tensor([True, False]))
        torch.testing.assert_close(dwell, torch.tensor([0.4, 0.0], dtype=torch.float64))
        start, dwell, complete = continuous_goal_dwell(
            torch.tensor([True, True]), torch.tensor([23, 23]), start,
            step_dt_s=1 / 30, required_dwell_s=0.4,
        )
        self.assertEqual(int(start[1]), 23)
        self.assertEqual(float(dwell[1]), 0)
        self.assertFalse(bool(complete[1]))

    def test_nonintegral_duration_requires_next_full_interval(self):
        start = torch.tensor([0, 0])
        _, _, complete = continuous_goal_dwell(
            torch.tensor([True, True]), torch.tensor([12, 13]), start,
            step_dt_s=1 / 30, required_dwell_s=0.401,
        )
        torch.testing.assert_close(complete, torch.tensor([False, True]))

    def test_invalid_timing_or_floating_steps_are_rejected(self):
        for dt, duration in ((0, 0.4), (float("nan"), 0.4), (1 / 30, float("inf"))):
            with self.assertRaises(ValueError):
                continuous_goal_dwell(
                    torch.tensor([True]), torch.tensor([0]), torch.tensor([-1]),
                    step_dt_s=dt, required_dwell_s=duration,
                )
        with self.assertRaisesRegex(ValueError, "int64"):
            continuous_goal_dwell(
                torch.tensor([True]), torch.tensor([0.0]), torch.tensor([-1]),
                step_dt_s=1 / 30, required_dwell_s=0.4,
            )


class DifferentialControlTests(unittest.TestCase):
    def test_joint_limit_scaling_preserves_cartesian_direction(self):
        jacobian = torch.eye(2).unsqueeze(0)
        desired = torch.tensor([[0.08, 0.04]])
        step = damped_joint_step(jacobian, desired, 0.01, 0.04)
        self.assertTrue(torch.allclose(step, torch.tensor([[0.04, 0.02]]), atol=1e-6))
        actual = (jacobian @ step.unsqueeze(-1)).squeeze(-1)
        self.assertTrue(torch.allclose(actual / actual.norm(), desired / desired.norm(), atol=1e-6))

    def test_singular_jacobian_stays_finite(self):
        step = damped_joint_step(torch.zeros(2, 6, 7), torch.ones(2, 6), 0.05, 0.04)
        self.assertTrue(torch.isfinite(step).all())
        self.assertEqual(float(step.abs().max()), 0.0)


class ContinuousGraspLossTests(unittest.TestCase):
    def test_transient_dropout_then_recovery_does_not_expire(self):
        since, expired = continuous_grasp_loss(
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([-1.0]),
            now_s=1.0,
            grace_s=0.2,
        )
        recovered_since, recovered_expired = continuous_grasp_loss(
            torch.tensor([True]),
            torch.tensor([True]),
            since,
            now_s=1.1,
            grace_s=0.2,
        )

        torch.testing.assert_close(recovered_since, torch.tensor([-1.0]))
        self.assertFalse(bool(expired.item()))
        self.assertFalse(bool(recovered_expired.item()))

    def test_sustained_dropout_expires_after_grace_period(self):
        since, _ = continuous_grasp_loss(
            torch.tensor([True]),
            torch.tensor([False]),
            torch.tensor([-1.0]),
            now_s=2.0,
            grace_s=0.2,
        )
        _, expired = continuous_grasp_loss(
            torch.tensor([True]),
            torch.tensor([False]),
            since,
            now_s=2.21,
            grace_s=0.2,
        )

        self.assertTrue(bool(expired.item()))

    def test_inactive_phase_resets_existing_dropout_timestamp(self):
        since, expired = continuous_grasp_loss(
            torch.tensor([False]),
            torch.tensor([False]),
            torch.tensor([3.0]),
            now_s=4.0,
            grace_s=0.2,
        )

        torch.testing.assert_close(since, torch.tensor([-1.0]))
        self.assertFalse(bool(expired.item()))

    def test_batched_inputs_track_each_environment_independently(self):
        since, expired = continuous_grasp_loss(
            torch.tensor([True, True, False, True]),
            torch.tensor([False, False, False, True]),
            torch.tensor([-1.0, 4.0, 3.0, 4.0]),
            now_s=5.0,
            grace_s=0.5,
        )

        torch.testing.assert_close(since, torch.tensor([5.0, 4.0, -1.0, -1.0]))
        torch.testing.assert_close(expired, torch.tensor([False, True, False, False]))

    def test_nonpositive_grace_period_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "grace_s must be positive"):
            continuous_grasp_loss(
                torch.tensor([True]),
                torch.tensor([False]),
                torch.tensor([-1.0]),
                now_s=1.0,
                grace_s=0.0,
            )


if __name__ == "__main__":
    unittest.main()
