import math
import unittest

from rby1_assembly.contracts import EvaluationSample, TaskLimits


class ContractTests(unittest.TestCase):
    def test_evaluation_rejects_nan(self) -> None:
        with self.assertRaisesRegex(ValueError, "actual_depth_m"):
            EvaluationSample(
                actual_depth_m=math.nan,
                lateral_error_m=0.0,
                orientation_error_rad=0.0,
                force_peak_n=0.0,
                grasp_both=True,
                forbidden_contact=False,
                sim_time_s=0.0,
            )

    def test_limits_are_explicit_simulation_defaults(self) -> None:
        limits = TaskLimits()
        self.assertEqual(limits.assembled_depth_m, 0.030)
        self.assertGreater(limits.max_contact_force_n, 0.0)
        self.assertGreater(limits.residual_translation_bound_m, 0.0)

    def test_limits_reject_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "dwell_s"):
            TaskLimits(dwell_s=0.0)
        with self.assertRaisesRegex(ValueError, "max_retries"):
            TaskLimits(max_retries=-1)


if __name__ == "__main__":
    unittest.main()
