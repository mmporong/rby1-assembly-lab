from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import h5py
import numpy as np

try:
    import torch
except ImportError:  # local lightweight test environments need not install Isaac's Torch stack
    torch = None

from rby1_assembly.rj45_teacher import teacher_tip_target_action
from rby1_assembly.rj45_teacher_dataset import (
    TeacherDatasetContractError,
    TeacherDatasetWriter,
    TeacherEpisodeProvenance,
    TeacherEpisodeRows,
    TeacherEpisodeTerminal,
    load_student_episodes,
)


HASH = "a" * 64


def provenance(
    *,
    split: str = "train",
    grasp_id: str = "grasp-0",
    bias_case_id: str = "bias-0",
    seed: int = 7,
) -> TeacherEpisodeProvenance:
    return TeacherEpisodeProvenance(
        grasp_id=grasp_id,
        split=split,
        bias_case_id=bias_case_id,
        seed=seed,
        source_sha256=HASH,
        task_sha256="b" * 64,
        training_sha256="c" * 64,
        bank_sha256="d" * 64,
        translation_bound_m=0.01,
        rotation_bound_rad=0.2,
    )


def rows(
    *, count: int = 2, valid: bool | np.ndarray = True, privileged: bool = True
) -> TeacherEpisodeRows:
    requested = np.arange(count * 6, dtype=np.float32).reshape(count, 6) / 10.0
    clipped = np.clip(requested, -1.0, 1.0).reshape(count, 2, 3)
    applied = (clipped / np.maximum(np.linalg.norm(clipped, axis=-1, keepdims=True), 1.0)).reshape(count, 6)
    offset = applied * np.asarray([0.01, 0.01, 0.01, 0.2, 0.2, 0.2], dtype=np.float32)
    valid_rows = np.full(count, valid, dtype=np.bool_) if isinstance(valid, bool) else valid
    return TeacherEpisodeRows(
        observation=np.arange(count * 79, dtype=np.float32).reshape(count, 79),
        observation_timestamp=np.arange(count, dtype=np.float64) * 0.02,
        valid=valid_rows,
        teacher_requested=requested,
        teacher_applied=applied,
        applied_offset=offset,
        phase=["align"] * count,
        timestep=np.arange(count, dtype=np.int64),
        teacher_gt_delta=np.zeros((count, 6), dtype=np.float32) if privileged else None,
        teacher_gt_pose=np.zeros((count, 7), dtype=np.float32) if privileged else None,
    )


def terminal(*, success: bool = True, valid: bool = True) -> TeacherEpisodeTerminal:
    return TeacherEpisodeTerminal(
        independent_success=success,
        valid=valid,
        failure_reason="" if success and valid else "timeout",
    )


class TeacherDatasetTests(unittest.TestCase):
    def test_round_trip_79d_6d_and_student_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            with TeacherDatasetWriter(path) as writer:
                writer.add_episode("success", provenance(), rows(), terminal())
            loaded = load_student_episodes(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["observation"].shape, (2, 79))
            self.assertEqual(loaded[0]["action"].shape, (2, 6))
            self.assertEqual(
                set(loaded[0]),
                {
                    "episode_id",
                    "observation",
                    "action",
                    "observation_timestamp",
                    "valid",
                    "phase",
                    "timestep",
                    "metadata",
                },
            )
            self.assertEqual(
                set(loaded[0]["metadata"]),
                {
                    "grasp_id",
                    "bias_case_id",
                    "split",
                    "seed",
                    "source_sha256",
                    "task_sha256",
                    "training_sha256",
                    "bank_sha256",
                },
            )
            self.assertNotIn("teacher_gt_delta", loaded[0])
            with h5py.File(path, "r") as file:
                self.assertEqual(file.attrs["schema_name"], "rj45_teacher_episode_v1")
                self.assertEqual(file.attrs["observation_mode"], "standard79")
                self.assertIn("privileged", file["episodes/success"])

    def test_wrong_shape_nan_timestamp_and_action_alignment_are_rejected(self) -> None:
        cases = []
        bad_shape = rows()
        object.__setattr__(bad_shape, "observation", np.zeros((2, 78)))
        cases.append(bad_shape)
        bad_nan = rows()
        bad_nan.observation[0, 0] = np.nan
        cases.append(bad_nan)
        bad_time = rows()
        object.__setattr__(bad_time, "observation_timestamp", np.asarray([0.1, 0.1]))
        cases.append(bad_time)
        negative_time = rows()
        object.__setattr__(negative_time, "observation_timestamp", np.asarray([-0.1, 0.1]))
        cases.append(negative_time)
        bad_action = rows()
        bad_action.teacher_applied[0, 0] += 1.0
        cases.append(bad_action)
        bad_offset = rows()
        bad_offset.applied_offset[0, 0] += 0.01
        cases.append(bad_offset)
        for index, item in enumerate(cases):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                writer = TeacherDatasetWriter(Path(directory) / "teacher.h5")
                with self.assertRaises(TeacherDatasetContractError):
                    writer.add_episode("bad", provenance(), item, terminal())
                writer.abort()

    def test_terminal_missing_and_final_split_are_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            writer = TeacherDatasetWriter(path)
            with self.assertRaisesRegex(TeacherDatasetContractError, "terminal"):
                writer.add_episode("missing", provenance(), rows(), None)
            with self.assertRaisesRegex(TeacherDatasetContractError, "final"):
                writer.add_episode("final", provenance(split="final"), rows(), terminal())
            writer.abort()
            self.assertFalse(path.exists())

    def test_failed_attempt_is_preserved_but_excluded_from_default_bc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            with TeacherDatasetWriter(path) as writer:
                writer.add_episode("failed", provenance(), rows(valid=False), terminal(success=False, valid=False))
                writer.add_episode("success", provenance(), rows(), terminal())
            with h5py.File(path, "r") as file:
                self.assertEqual(sorted(file["episodes"]), ["failed", "success"])
                self.assertEqual(file["episodes/failed"].attrs["failure_reason"], "timeout")
            self.assertEqual([item["episode_id"] for item in load_student_episodes(path)], ["success"])
            self.assertEqual(
                [item["episode_id"] for item in load_student_episodes(path, successful_only=False, valid_only=False)],
                ["failed", "success"],
            )

    def test_invalid_warmup_rows_are_masked_without_dropping_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            with TeacherDatasetWriter(path) as writer:
                writer.add_episode(
                    "warmup",
                    provenance(),
                    rows(valid=np.asarray([False, True], dtype=np.bool_)),
                    terminal(),
                )
            loaded = load_student_episodes(path)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["observation"].shape, (1, 79))
            self.assertEqual(loaded[0]["action"].shape, (1, 6))
            self.assertEqual(loaded[0]["timestep"].tolist(), [1])

    def test_teacher_api_fixture_matches_dataset_physical_scaling(self) -> None:
        if torch is None:
            result_arrays = _teacher_api_fixture_from_system_python()
        else:
            identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
            position = torch.zeros((1, 3), dtype=torch.float64)
            result = teacher_tip_target_action(
                baseline_body_position_world=position,
                baseline_body_quaternion_world_wxyz=identity,
                desired_body_position_world=torch.tensor([[0.02, 0.02, 0.0]], dtype=torch.float64),
                desired_body_quaternion_world_wxyz=identity,
                socket_entry_quaternion_world_wxyz=identity,
                plug_tip_position_local=position,
                plug_tip_quaternion_local_wxyz=identity,
                translation_bound_m=0.01,
                rotation_bound_rad=0.2,
            )
            result_arrays = {
                "raw": result["raw_action_6d"].numpy(),
                "bounded": result["bounded_action_6d"].numpy(),
                "offset": result["applied_offset_entry_6d"].numpy(),
            }
        fixture = rows(count=1)
        object.__setattr__(fixture, "teacher_requested", result_arrays["raw"].astype(np.float32))
        object.__setattr__(fixture, "teacher_applied", result_arrays["bounded"].astype(np.float32))
        object.__setattr__(fixture, "applied_offset", result_arrays["offset"].astype(np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            with TeacherDatasetWriter(path) as writer:
                writer.add_episode("teacher-api", provenance(), fixture, terminal())
            loaded = load_student_episodes(path)
            np.testing.assert_allclose(loaded[0]["action"], result_arrays["bounded"])

    def test_writer_and_loader_reject_cross_split_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            writer = TeacherDatasetWriter(path)
            writer.add_episode("train", provenance(), rows(), terminal())
            with self.assertRaisesRegex(TeacherDatasetContractError, "grasp_id"):
                writer.add_episode("dev", provenance(split="dev"), rows(), terminal())
            with self.assertRaisesRegex(TeacherDatasetContractError, "bias_case_id/seed"):
                writer.add_episode(
                    "dev-bias",
                    provenance(split="dev", grasp_id="grasp-2"),
                    rows(),
                    terminal(),
                )
            writer.abort()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            with TeacherDatasetWriter(path) as writer:
                writer.add_episode("train", provenance(), rows(), terminal())
                writer.add_episode(
                    "also-train",
                    provenance(grasp_id="grasp-1", bias_case_id="bias-1", seed=8),
                    rows(),
                    terminal(),
                )
            with h5py.File(path, "r+") as file:
                file["episodes/also-train"].attrs["split"] = "dev"
                file["episodes/also-train"].attrs["grasp_id"] = "grasp-0"
            with self.assertRaisesRegex(TeacherDatasetContractError, "grasp_id"):
                load_student_episodes(path)

    def test_loader_revalidates_root_provenance_terminal_and_unknown_split(self) -> None:
        mutations = (
            ("root dimension", lambda file: file.attrs.__setitem__("action_dim", 7), "action_dim"),
            (
                "provenance",
                lambda file: file["episodes/success"].attrs.__setitem__("source_sha256", "bad"),
                "source_sha256",
            ),
            (
                "terminal type",
                lambda file: file["episodes/success"].attrs.__setitem__("valid", 1),
                "boolean",
            ),
            (
                "unknown split",
                lambda file: file["episodes/success"].attrs.__setitem__("split", "final"),
                "final",
            ),
            (
                "terminal consistency",
                lambda file: file["episodes/success"].attrs.__setitem__("failure_reason", "timeout"),
                "must not have failure_reason",
            ),
        )
        for name, mutation, message in mutations:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "teacher.h5"
                with TeacherDatasetWriter(path) as writer:
                    writer.add_episode("success", provenance(), rows(), terminal())
                with h5py.File(path, "r+") as file:
                    mutation(file)
                with self.assertRaisesRegex(TeacherDatasetContractError, message):
                    load_student_episodes(path)

    def test_atomic_no_overwrite_and_dev_is_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.h5"
            with TeacherDatasetWriter(path) as writer:
                writer.add_episode("dev", provenance(split="dev"), rows(privileged=False), terminal())
            self.assertEqual(load_student_episodes(path), [])
            self.assertEqual(len(load_student_episodes(path, split="dev")), 1)
            with self.assertRaises(FileExistsError):
                TeacherDatasetWriter(path)


def _teacher_api_fixture_from_system_python() -> dict[str, np.ndarray]:
    code = """
import json
import torch
from rby1_assembly.rj45_teacher import teacher_tip_target_action
identity=torch.tensor([[1.,0.,0.,0.]],dtype=torch.float64)
position=torch.zeros((1,3),dtype=torch.float64)
result=teacher_tip_target_action(
    baseline_body_position_world=position,
    baseline_body_quaternion_world_wxyz=identity,
    desired_body_position_world=torch.tensor([[0.02,0.02,0.]],dtype=torch.float64),
    desired_body_quaternion_world_wxyz=identity,
    socket_entry_quaternion_world_wxyz=identity,
    plug_tip_position_local=position,
    plug_tip_quaternion_local_wxyz=identity,
    translation_bound_m=.01,
    rotation_bound_rad=.2,
)
print(json.dumps({
    'raw': result['raw_action_6d'].tolist(),
    'bounded': result['bounded_action_6d'].tolist(),
    'offset': result['applied_offset_entry_6d'].tolist(),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    completed = subprocess.run(
        ["python3", "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return {name: np.asarray(value) for name, value in json.loads(completed.stdout).items()}


if __name__ == "__main__":
    unittest.main()
