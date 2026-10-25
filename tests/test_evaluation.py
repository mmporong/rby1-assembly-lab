import unittest

from rby1_assembly.contracts import EvaluationSample, TaskLimits
from rby1_assembly.evaluation import TaskEvaluator


def sample(time_s: float, **changes: object) -> EvaluationSample:
    values = {
        "actual_depth_m": 0.031,
        "lateral_error_m": 0.001,
        "orientation_error_rad": 0.01,
        "force_peak_n": 10.0,
        "grasp_both": True,
        "forbidden_contact": False,
        "sim_time_s": time_s,
    }
    values.update(changes)
    return EvaluationSample(**values)  # type: ignore[arg-type]


class EvaluationTests(unittest.TestCase):
    def test_success_requires_sustained_goal_independent_of_controller_phase(self) -> None:
        evaluator = TaskEvaluator(TaskLimits(dwell_s=0.4))
        self.assertFalse(evaluator.update(sample(1.0)).terminal)
        self.assertFalse(evaluator.update(sample(1.3)).terminal)
        result = evaluator.update(sample(1.4))
        self.assertTrue(result.terminal)
        self.assertTrue(result.success)

    def test_invalid_dt_and_update_after_done_are_rejected(self) -> None:
        evaluator = TaskEvaluator(TaskLimits(dwell_s=0.1))
        evaluator.update(sample(0.0))
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            evaluator.update(sample(0.0))
        evaluator.update(sample(0.1))
        with self.assertRaisesRegex(RuntimeError, "terminal"):
            evaluator.update(sample(0.2))

    def test_force_violation_is_terminal_and_reset_clears_it(self) -> None:
        evaluator = TaskEvaluator(TaskLimits())
        failed = evaluator.update(sample(0.0, force_peak_n=40.1))
        self.assertTrue(failed.terminal)
        self.assertFalse(failed.success)
        self.assertIn("contact_force_limit_exceeded", failed.reasons)

        evaluator.reset()
        fresh = evaluator.update(sample(0.0, actual_depth_m=0.0))
        self.assertFalse(fresh.terminal)
        self.assertEqual(fresh.reasons, ())

    def test_dwell_resets_when_goal_breaks(self) -> None:
        evaluator = TaskEvaluator(TaskLimits(dwell_s=0.4))
        evaluator.update(sample(0.0))
        evaluator.update(sample(0.3, grasp_both=False))
        evaluator.update(sample(0.4))
        result = evaluator.update(sample(0.7))
        self.assertFalse(result.terminal)
        self.assertAlmostEqual(result.dwell_progress_s, 0.3)

    def test_forbidden_contact_is_terminal(self) -> None:
        evaluator = TaskEvaluator(TaskLimits())
        result = evaluator.update(sample(0.0, forbidden_contact=True))
        self.assertTrue(result.terminal)
        self.assertFalse(result.success)
        self.assertEqual(result.reasons, ("forbidden_contact_detected",))


if __name__ == "__main__":
    unittest.main()
