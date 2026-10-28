from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rby1_assembly.rj45_teacher_collection import (  # noqa: E402
    CollectedTeacherEpisode,
    ParallelTeacherCollector,
    TeacherCollectionError,
    fixed_environment_episode_quotas,
    write_collected_episodes,
)


def _snapshot(
    episode_indices: tuple[int, int],
    grasp_indices: tuple[int, int],
    *,
    active: tuple[bool, bool] = (True, True),
    policy_valid: tuple[bool, bool] = (True, True),
    success: tuple[bool, bool] = (False, False),
) -> dict[str, object]:
    observation = torch.stack(
        (torch.arange(79, dtype=torch.float32), torch.arange(79, dtype=torch.float32) + 100)
    )
    requested = torch.tensor(
        [[0.2, -0.3, 0.4, 0.1, 0.0, -0.2], [0.5, 0.0, 0.0, 0.0, 0.3, 0.0]]
    )
    bounded = requested.clone()
    scales = torch.tensor([0.0015] * 3 + [0.02] * 3)
    return {
        "episode_index": torch.tensor(episode_indices),
        "held_grasp_index": torch.tensor(grasp_indices),
        "policy_input_observation": observation,
        "policy_input_timestamp_s": torch.tensor([-1.0, 0.5]),
        "policy_input_valid": torch.tensor(policy_valid),
        "teacher_requested_raw_action_6d": requested,
        "teacher_bounded_action_6d": bounded,
        "teacher_physical_offset_entry_6d": bounded * scales,
        "teacher_active": torch.tensor(active),
        "teacher_virtual_target_body_pose_world_wxyz": torch.tensor(
            [[9.0, 8.0, 7.0, 1.0, 0.0, 0.0, 0.0], [6.0, 5.0, 4.0, 1.0, 0.0, 0.0, 0.0]]
        ),
        "phase": torch.tensor([4, 4]),
        "phase_name_by_code": {4: "insert"},
        "independent_eval_success": torch.tensor(success),
        "independent_eval_valid": torch.tensor([True, True]),
        "held_initialization_passed": torch.tensor([True, True]),
        "valid_policy_input_step_count": torch.tensor([2, 2]),
        "actual_depth_m": torch.tensor([0.01, 0.01]),
        "actual_lateral_error_m": torch.tensor([0.0, 0.0]),
        "actual_orientation_error_rad": torch.tensor([0.0, 0.0]),
        "termination_reason_code": torch.tensor([4, 1]),
        "termination_reason_by_code": {1: "independent_success", 4: "episode_timeout"},
    }


class Rj45TeacherCollectionTests(unittest.TestCase):
    def test_cli_rejects_sealed_final_seed_before_reading_inputs_or_launching_isaac(self) -> None:
        script = Path(__file__).resolve().parents[1] / "scripts" / "collect_rj45_teacher.py"
        result = subprocess.run(
            [
                sys.executable,
                str(script),
                "--output", "/tmp/unused-teacher.h5",
                "--split", "train",
                "--rj45-nominal-evidence", "/missing/nominal.json",
                "--rj45-held-bank", "/missing/bank.json",
                "--rj45-training-config", "/missing/training.json",
                "--episodes", "1",
                "--num-envs", "1",
                "--seed", "20261027",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("reserved for sealed final evaluation", result.stderr)
        self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_async_partial_resets_preserve_all_attempts_and_causal_rows(self) -> None:
        collector = ParallelTeacherCollector(
            num_envs=2,
            total_episodes=2,
            grasp_ids=("grasp-a", "grasp-b"),
            episode_seed=lambda env_id, episode_index: 20261120 + env_id + episode_index,
        )
        first = _snapshot(
            (0, 0), (0, 1), active=(False, True), policy_valid=(False, True)
        )
        collector.consume_step(
            live_snapshot=first,
            terminal_snapshot=None,
            ended=torch.tensor([False, False]),
            pre_step_time_s=0.0,
        )
        second_live = _snapshot((1, 0), (1, 1), success=(False, False))
        env0_terminal = _snapshot((0, 0), (0, 1), success=(False, False))
        collector.consume_step(
            live_snapshot=second_live,
            terminal_snapshot=env0_terminal,
            ended=torch.tensor([True, False]),
            pre_step_time_s=0.1,
        )
        env1_terminal = _snapshot((1, 0), (0, 1), success=(False, True))
        collector.consume_step(
            live_snapshot=_snapshot((1, 1), (0, 0)),
            terminal_snapshot=env1_terminal,
            ended=torch.tensor([False, True]),
            pre_step_time_s=0.2,
        )

        self.assertTrue(collector.complete)
        self.assertEqual(collector.accepted_per_environment, (1, 1))
        self.assertEqual(len(collector.episodes), 2)
        env0, env1 = collector.episodes
        self.assertFalse(env0.independent_success)
        self.assertEqual(env0.failure_reason, "episode_timeout")
        self.assertTrue(env1.independent_success)
        self.assertEqual(env1.failure_reason, "")
        self.assertEqual(env0.valid.tolist(), [False, True])
        self.assertEqual(env0.observation_timestamp.tolist(), [0.0, 0.1])
        self.assertEqual(env0.policy_observation_timestamp.tolist(), [-1.0, -1.0])
        np.testing.assert_array_equal(env0.observation[0], np.arange(79, dtype=np.float32))
        np.testing.assert_array_equal(
            env0.teacher_applied[0],
            np.asarray(first["teacher_bounded_action_6d"][0]),
        )
        self.assertEqual(env0.teacher_gt_pose.shape, (2, 7))
        self.assertEqual(env0.observation.shape[1], 79)

    def test_identity_change_before_terminal_is_rejected(self) -> None:
        collector = ParallelTeacherCollector(
            num_envs=1,
            total_episodes=1,
            grasp_ids=("grasp-a", "grasp-b"),
            episode_seed=lambda _env, episode: episode,
        )
        collector.consume_step(
            live_snapshot={key: value[:1] if isinstance(value, torch.Tensor) else value for key, value in _snapshot((0, 0), (0, 0)).items()},
            terminal_snapshot=None,
            ended=torch.tensor([False]),
            pre_step_time_s=0.0,
        )
        changed = {
            key: value[:1] if isinstance(value, torch.Tensor) else value
            for key, value in _snapshot((0, 0), (1, 0)).items()
        }
        with self.assertRaisesRegex(TeacherCollectionError, "identity changed"):
            collector.consume_step(
                live_snapshot=changed,
                terminal_snapshot=None,
                ended=torch.tensor([False]),
                pre_step_time_s=0.1,
            )

    def test_fixed_quota_does_not_accept_extra_fast_environment_attempts(self) -> None:
        self.assertEqual(fixed_environment_episode_quotas(5, 2), (3, 2))
        collector = ParallelTeacherCollector(
            num_envs=2,
            total_episodes=2,
            grasp_ids=("grasp-a", "grasp-b"),
            episode_seed=lambda env, episode: env + episode,
        )
        terminal = _snapshot((0, 0), (0, 1))
        collector.consume_step(
            live_snapshot=terminal,
            terminal_snapshot=terminal,
            ended=torch.tensor([True, False]),
            pre_step_time_s=0.0,
        )
        next_terminal = _snapshot((1, 0), (1, 1))
        collector.consume_step(
            live_snapshot=next_terminal,
            terminal_snapshot=next_terminal,
            ended=torch.tensor([True, True]),
            pre_step_time_s=0.1,
        )
        self.assertTrue(collector.complete)
        self.assertEqual([episode.env_id for episode in collector.episodes], [0, 1])

    def test_terminal_validity_rejects_no_policy_exposure_without_dropping_attempt(self) -> None:
        collector = ParallelTeacherCollector(
            num_envs=1,
            total_episodes=1,
            grasp_ids=("grasp-a",),
            episode_seed=lambda _env, _episode: 7,
        )
        terminal = {
            key: value[:1] if isinstance(value, torch.Tensor) else value
            for key, value in _snapshot((0, 0), (0, 0), success=(True, False)).items()
        }
        terminal["valid_policy_input_step_count"] = torch.tensor([0])
        collector.consume_step(
            live_snapshot=terminal,
            terminal_snapshot=terminal,
            ended=torch.tensor([True]),
            pre_step_time_s=0.0,
        )
        self.assertEqual(len(collector.episodes), 1)
        self.assertFalse(collector.episodes[0].terminal_valid)
        self.assertEqual(
            collector.episodes[0].failure_reason, "no_valid_policy_input_exposure"
        )

    def test_dataset_writer_loader_preserves_failed_attempt_and_filters_invalid_warmup(self) -> None:
        try:
            import h5py  # noqa: F401
            from rby1_assembly.rj45_teacher_dataset import load_student_episodes
        except ModuleNotFoundError:
            self.skipTest("h5py is unavailable in this interpreter")
        requested = np.asarray(
            [[0.0] * 6, [0.2, 0.0, 0.0, 0.0, -0.1, 0.0]], dtype=np.float32
        )
        episode = CollectedTeacherEpisode(
            env_id=0,
            episode_index=3,
            grasp_id="grasp-train-a",
            bias_case_id="env-0-episode-3",
            seed=20261123,
            observation=np.arange(158, dtype=np.float32).reshape(2, 79),
            observation_timestamp=np.asarray([0.0, 0.1], dtype=np.float64),
            policy_observation_timestamp=np.asarray([-1.0, 0.08], dtype=np.float64),
            valid=np.asarray([False, True], dtype=np.bool_),
            teacher_requested=requested,
            teacher_applied=requested.copy(),
            applied_offset=requested
            * np.asarray([0.0015] * 3 + [0.02] * 3, dtype=np.float32),
            phase=("align", "insert"),
            teacher_gt_pose=np.asarray(
                [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2,
                dtype=np.float32,
            ),
            independent_success=False,
            terminal_valid=True,
            failure_reason="episode_timeout",
        )
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "teacher.h5"
            write_collected_episodes(
                dataset,
                [episode],
                split="train",
                source_sha256="a" * 64,
                task_sha256="b" * 64,
                training_sha256="c" * 64,
                bank_sha256="d" * 64,
                translation_bound_m=0.0015,
                rotation_bound_rad=0.02,
            )
            all_rows = load_student_episodes(
                dataset, split="train", successful_only=False, valid_only=False
            )
            valid_rows = load_student_episodes(
                dataset, split="train", successful_only=False, valid_only=True
            )
            with h5py.File(dataset, "r") as file:
                stored_observation_time = file[
                    "episodes/train-env0-episode3-0/observation_timestamp"
                ][:]
                stored_policy_time = file[
                    "episodes/train-env0-episode3-0/policy_observation_timestamp"
                ][:]
        self.assertEqual(len(all_rows), 1)
        self.assertEqual(all_rows[0]["observation"].shape, (2, 79))
        self.assertEqual(all_rows[0]["valid"].tolist(), [False, True])
        self.assertEqual(valid_rows[0]["observation"].shape, (1, 79))
        self.assertEqual(valid_rows[0]["observation_timestamp"].tolist(), [0.1])
        self.assertNotIn("teacher_gt_pose", valid_rows[0])
        np.testing.assert_array_equal(stored_observation_time, np.asarray([0.0, 0.1]))
        np.testing.assert_array_equal(stored_policy_time, np.asarray([-1.0, 0.08]))
        self.assertFalse(np.array_equal(stored_observation_time, stored_policy_time))


if __name__ == "__main__":
    unittest.main()
