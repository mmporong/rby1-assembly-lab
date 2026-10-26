from __future__ import annotations

import unittest

import torch

from rby1_assembly.contact_manifolds import pack_filtered_contact_manifold


class FilteredContactManifoldPackingTests(unittest.TestCase):
    def _buffers(self):
        forces = torch.tensor([[float("nan")], [2.5], [-1.25], [float("nan")],
                               [float("nan")], [4.0], [float("nan")]])
        points = torch.full((7, 3), float("nan"))
        points[1] = torch.tensor([0.10, 0.20, 0.30])
        points[2] = torch.tensor([0.11, 0.21, 0.31])
        points[5] = torch.tensor([0.50, 0.60, 0.70])
        normals = torch.full((7, 3), float("nan"))
        normals[1] = torch.tensor([1.0, 0.0, 0.0])
        normals[2] = torch.tensor([0.0, -1.0, 0.0])
        normals[5] = torch.tensor([0.0, 0.0, 1.0])
        separations = torch.tensor(
            [float("nan"), -0.001, 0.0002, float("nan"), float("nan"), -0.003, float("nan")]
        )
        counts = torch.tensor([[2], [0], [1]], dtype=torch.int32)
        starts = torch.tensor([[1], [-999], [5]], dtype=torch.int32)
        return forces, points, normals, separations, counts, starts

    def test_packs_distinct_environment_regions_and_zeroes_inactive_entries(self) -> None:
        packed = pack_filtered_contact_manifold(*self._buffers(), num_envs=3, capacity=4)

        torch.testing.assert_close(
            packed["contact_count"], torch.tensor([2, 0, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            packed["active"],
            torch.tensor(
                [[True, True, False, False], [False] * 4, [True, False, False, False]]
            ),
        )
        torch.testing.assert_close(
            packed["force_n"],
            torch.tensor([[2.5, -1.25, 0.0, 0.0], [0.0] * 4, [4.0, 0.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(
            packed["points_world_m"][0, :2],
            torch.tensor([[0.10, 0.20, 0.30], [0.11, 0.21, 0.31]]),
        )
        torch.testing.assert_close(
            packed["points_world_m"][2, 0], torch.tensor([0.50, 0.60, 0.70])
        )
        self.assertTrue(torch.equal(packed["points_world_m"][~packed["active"]], torch.zeros(9, 3)))
        self.assertTrue(torch.equal(packed["normals_world"][~packed["active"]], torch.zeros(9, 3)))
        self.assertTrue(torch.equal(packed["separation_m"][~packed["active"]], torch.zeros(9)))
        self.assertTrue(all(torch.isfinite(value).all() for value in packed.values() if value.is_floating_point()))

    def test_zero_contacts_allow_empty_native_buffers_and_undefined_starts(self) -> None:
        packed = pack_filtered_contact_manifold(
            torch.empty(0),
            torch.empty((0, 3)),
            torch.empty((0, 3)),
            torch.empty((0, 1)),
            torch.zeros((2, 1), dtype=torch.int64),
            torch.tensor([[-10], [100]], dtype=torch.int64),
            num_envs=2,
            capacity=3,
        )

        self.assertEqual(packed["active"].sum().item(), 0)
        torch.testing.assert_close(packed["force_n"], torch.zeros((2, 3)))

    def test_rejects_non_integer_count_or_start_tensors(self) -> None:
        buffers = list(self._buffers())
        buffers[4] = buffers[4].float()
        with self.assertRaisesRegex(TypeError, "integer"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

        buffers = list(self._buffers())
        buffers[5] = buffers[5].float()
        with self.assertRaisesRegex(TypeError, "integer"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

    def test_rejects_count_and_start_shapes_other_than_one_filter_per_env(self) -> None:
        buffers = list(self._buffers())
        buffers[4] = torch.tensor([[2, 0], [0, 0], [1, 0]], dtype=torch.int32)
        buffers[5] = torch.zeros((3, 2), dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "shape"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

    def test_rejects_negative_count(self) -> None:
        buffers = list(self._buffers())
        buffers[4] = torch.tensor([[2], [-1], [1]], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "non-negative"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

    def test_rejects_active_range_outside_native_buffer(self) -> None:
        buffers = list(self._buffers())
        buffers[5] = torch.tensor([[6], [-999], [5]], dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "outside"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

    def test_small_integer_index_dtypes_do_not_overflow(self) -> None:
        forces = torch.zeros(140)
        points = torch.zeros((140, 3))
        normals = torch.zeros((140, 3))
        normals[130] = torch.tensor([1.0, 0.0, 0.0])
        separations = torch.zeros(140)

        packed = pack_filtered_contact_manifold(
            forces,
            points,
            normals,
            separations,
            torch.tensor([[1]], dtype=torch.int8),
            torch.tensor([[130]], dtype=torch.uint8),
            num_envs=1,
            capacity=128,
        )

        self.assertTrue(packed["active"][0, 0])
        torch.testing.assert_close(
            packed["normals_world"][0, 0], torch.tensor([1.0, 0.0, 0.0])
        )

    def test_rejects_nonfinite_active_values(self) -> None:
        for field_index in range(4):
            with self.subTest(field_index=field_index):
                buffers = list(self._buffers())
                buffers[field_index][1] = float("nan")
                with self.assertRaisesRegex(ValueError, "finite"):
                    pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

    def test_rejects_non_unit_active_normal(self) -> None:
        buffers = list(self._buffers())
        buffers[2][1] = torch.tensor([0.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "unit length"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=4)

    def test_rejects_contact_count_above_output_capacity(self) -> None:
        buffers = list(self._buffers())
        with self.assertRaisesRegex(ValueError, "capacity"):
            pack_filtered_contact_manifold(*buffers, num_envs=3, capacity=1)


if __name__ == "__main__":
    unittest.main()
