from __future__ import annotations

import inspect
import unittest

import torch

from rby1_assembly.rj45_training_env import (
    _held_bank_selection_indices,
    _stack_held_bank_states,
    _validate_held_bank_proof,
    make_rj45_training_env_type,
)


def artifact(offset: float, *, reverse_joints: bool = False) -> dict[str, object]:
    names = ["joint_a", "joint_b"]
    positions = [offset + 1.0, offset + 2.0]
    velocities = [offset + 3.0, offset + 4.0]
    references = [offset + 5.0, offset + 6.0]
    if reverse_joints:
        names.reverse()
        positions.reverse()
        velocities.reverse()
        references.reverse()
    return {
        "joint_names": names,
        "state": {
            "joint_positions_si": positions,
            "joint_velocities_si": velocities,
            "joint_reference_si": references,
            "object_poses_world_wxyz": [
                [offset, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [offset, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ],
            "object_com_velocities_world": [[offset] * 6, [offset + 1.0] * 6],
        },
    }


def proof(count: int) -> dict[str, object]:
    return {
        "selected_count": count,
        "selected_grasp_ids": [f"grasp-{index}" for index in range(count)],
        "selected_artifact_sha256": [f"{index + 1:064x}" for index in range(count)],
        "selected_split": "train",
        "manifest_sha256": "f" * 64,
    }


class Rj45HeldBankResetTests(unittest.TestCase):
    def test_proof_rows_ids_hashes_and_split_must_match(self) -> None:
        bank = [artifact(0.0), artifact(10.0)]
        self.assertEqual(_validate_held_bank_proof(bank, proof(2))["selected_count"], 2)
        for mutate, message in (
            (lambda value: value.__setitem__("selected_count", 1), "selected_count"),
            (lambda value: value.__setitem__("selected_grasp_ids", ["same", "same"]), "grasp_ids"),
            (lambda value: value.__setitem__("selected_split", "test"), "selected_split"),
            (lambda value: value.__setitem__("manifest_sha256", "ABC"), "manifest_sha256"),
        ):
            with self.subTest(message=message):
                candidate = proof(2)
                mutate(candidate)
                with self.assertRaisesRegex(ValueError, message):
                    _validate_held_bank_proof(bank, candidate)

    def test_selection_is_balanced_repeatable_and_subset_order_independent(self) -> None:
        env_ids = torch.arange(8, dtype=torch.long)
        episodes = torch.zeros(8, dtype=torch.long)
        selected = _held_bank_selection_indices(
            env_ids, episodes, seed=17, bank_size=4
        )
        self.assertEqual(torch.bincount(selected, minlength=4).tolist(), [2, 2, 2, 2])
        subset = torch.tensor([6, 1, 4], dtype=torch.long)
        subset_episodes = torch.tensor([3, 9, 2], dtype=torch.long)
        forward = _held_bank_selection_indices(
            subset, subset_episodes, seed=17, bank_size=4
        )
        reverse = _held_bank_selection_indices(
            subset.flip(0), subset_episodes.flip(0), seed=17, bank_size=4
        )
        torch.testing.assert_close(forward, reverse.flip(0))
        unchanged = selected.clone()
        unchanged[subset] = forward
        untouched = torch.tensor([0, 2, 3, 5, 7])
        torch.testing.assert_close(unchanged[untouched], selected[untouched])

    def test_stacked_rows_restore_each_artifact_in_robot_joint_order(self) -> None:
        stacked = _stack_held_bank_states(
            [artifact(0.0), artifact(10.0, reverse_joints=True)],
            ["joint_a", "joint_b"],
            device="cpu",
        )
        torch.testing.assert_close(
            stacked["joint_position"], torch.tensor([[1.0, 2.0], [11.0, 12.0]])
        )
        selected = torch.tensor([1, 0, 1])
        torch.testing.assert_close(
            stacked["joint_reference"][selected],
            torch.tensor([[15.0, 16.0], [5.0, 6.0], [15.0, 16.0]]),
        )
        object_poses = stacked["object_pose"][selected].clone()
        origins = torch.tensor([[100.0, 0.0, 0.0], [200.0, 0.0, 0.0], [300.0, 0.0, 0.0]])
        object_poses[:, :, :3] += origins[:, None, :]
        torch.testing.assert_close(object_poses[:, 0, 0], torch.tensor([110.0, 200.0, 310.0]))

    def test_reset_selects_next_episode_before_base_physical_reset(self) -> None:
        source = inspect.getsource(make_rj45_training_env_type)
        reset_start = source.index("def _reset_idx(self, env_ids=None):")
        reset_end = source.index("def _configure_training_reset", reset_start)
        reset_source = source[reset_start:reset_end]
        self.assertLess(
            reset_source.index("self._select_held_bank_rows"),
            reset_source.index("super()._reset_idx"),
        )
        self.assertIn("if self._training_held_bank is not None", source)
        self.assertIn("state[\"joint_positions_si\"]", source)
        self.assertIn(".repeat(len(env_ids), 1)", source)


if __name__ == "__main__":
    unittest.main()
