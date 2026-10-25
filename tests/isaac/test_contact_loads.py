from __future__ import annotations

import unittest

import torch

from rby1_assembly.contact_loads import sum_contact_normal_loads


class ContactLoadAggregationTests(unittest.TestCase):
    def test_opposed_contacts_do_not_cancel(self) -> None:
        forces = torch.tensor([25.0, -25.0, 0.0])
        counts = torch.tensor([[2]], dtype=torch.int32)
        starts = torch.tensor([[0]], dtype=torch.int32)

        loads = sum_contact_normal_loads(forces, counts, starts, num_envs=1)

        torch.testing.assert_close(loads, torch.tensor([50.0]))

    def test_multi_environment_rows_and_zero_count_starts(self) -> None:
        forces = torch.tensor([0.0, -2.0, 0.0, 3.0, -4.0, 0.0, 0.0, 0.0])
        counts = torch.tensor([[1, 0], [2, 0]], dtype=torch.int32)
        starts = torch.tensor([[1, 999], [3, -999]], dtype=torch.int32)

        loads = sum_contact_normal_loads(forces, counts, starts, num_envs=2)

        torch.testing.assert_close(loads, torch.tensor([2.0, 7.0]))

    def test_zero_contacts_return_zero_and_ignore_starts(self) -> None:
        forces = torch.zeros(4)
        counts = torch.zeros((2, 2), dtype=torch.int32)
        starts = torch.tensor([[999, -999], [-1, 1000]], dtype=torch.int32)

        loads = sum_contact_normal_loads(forces, counts, starts, num_envs=2)

        torch.testing.assert_close(loads, torch.zeros(2))

    def test_negative_count_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-negative"):
            sum_contact_normal_loads(
                torch.zeros(4),
                torch.tensor([[-1]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                num_envs=1,
            )

    def test_active_range_outside_buffer_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            sum_contact_normal_loads(
                torch.zeros(4),
                torch.tensor([[1]], dtype=torch.int32),
                torch.tensor([[4]], dtype=torch.int32),
                num_envs=1,
            )

    def test_nonfinite_active_force_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            sum_contact_normal_loads(
                torch.tensor([float("nan"), 0.0]),
                torch.tensor([[1]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                num_envs=1,
            )

    def test_full_contact_buffer_is_treated_as_saturated(self) -> None:
        with self.assertRaisesRegex(ValueError, "saturated"):
            sum_contact_normal_loads(
                torch.tensor([1.0, 2.0]),
                torch.tensor([[2]], dtype=torch.int32),
                torch.tensor([[0]], dtype=torch.int32),
                num_envs=1,
            )

    def test_rows_must_partition_into_environments(self) -> None:
        with self.assertRaisesRegex(ValueError, "divisible"):
            sum_contact_normal_loads(
                torch.zeros(4),
                torch.zeros(3, dtype=torch.int32),
                torch.zeros(3, dtype=torch.int32),
                num_envs=2,
            )


if __name__ == "__main__":
    unittest.main()
