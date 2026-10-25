from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np

from rby1_assembly.dataset import (
    DatasetContractError,
    DatasetSplit,
    EpisodeDatasetWriter,
    EpisodeMetadata,
    EpisodeProvenance,
    GroundTruthMetrics,
    ProvenanceKind,
    TerminalEvaluation,
    TimestepRecord,
    inspect_dataset,
    load_training_episode,
    select_episode_ids,
    split_for_seed,
)


HASH = "a" * 64


def provenance(kind: ProvenanceKind = ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT) -> EpisodeProvenance:
    return EpisodeProvenance(
        kind=kind,
        source_hashes={"robot_usd": HASH, "scene_usd": "b" * 64},
        task_config_sha256="c" * 64,
        runtime_versions={"isaac_sim": "5.1.0", "isaac_lab": "2.3.0", "physx": "5"},
        observation_source="simulator_state_with_noise",
        contact_force_source="isaac_contact_sensor_proxy",
        owns_demonstration=kind is ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT,
        external_reference="NVIDIA AutoMate asset_00015" if kind is ProvenanceKind.SOURCE_EXTERNAL_REFERENCE else None,
    )


def metrics(*, success: bool = False) -> GroundTruthMetrics:
    return GroundTruthMetrics(
        actual_depth_m=0.031 if success else 0.012,
        lateral_error_m=0.001 if success else 0.004,
        orientation_error_rad=0.01,
        force_peak_n=18.0,
        grasp_both=True,
        forbidden_contact=False,
    )


def record(timestamp: float, *, success: bool = False, observation_size: int = 4) -> TimestepRecord:
    return TimestepRecord(
        timestamp_s=timestamp,
        observation=np.arange(observation_size, dtype=float),
        action=np.zeros(12),
        joint_position_rad=np.zeros(20),
        joint_velocity_rad_s=np.zeros(20),
        gripper_position_m=np.zeros(4),
        gripper_velocity_m_s=np.zeros(4),
        object_poses_world=np.asarray(
            [[0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0], [0.1, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0]]
        ),
        contact_force_proxy_n=np.zeros(2),
        phase="insert",
        ground_truth=metrics(success=success),
    )


def terminal(timestamp: float, *, success: bool = False, valid: bool = True) -> TerminalEvaluation:
    return TerminalEvaluation(
        success=success,
        valid=valid,
        termination_reason="success" if success else "timeout",
        evaluator_source="simulator_ground_truth_independent",
        evaluator_version="task-evaluator-v1",
        terminal_timestamp_s=timestamp,
        metrics=metrics(success=success),
        criteria={"depth": success, "alignment": success, "force": True},
    )


def seed_for_split(wanted: DatasetSplit) -> int:
    return next(seed for seed in range(1000) if split_for_seed(seed) is wanted)


class DatasetTests(unittest.TestCase):
    def test_split_is_deterministic_and_metadata_rejects_override(self) -> None:
        seed = 42
        self.assertIs(split_for_seed(seed), split_for_seed(seed))
        expected = split_for_seed(seed)
        wrong = DatasetSplit.EVAL if expected is DatasetSplit.TRAIN else DatasetSplit.TRAIN
        with self.assertRaises(DatasetContractError):
            EpisodeMetadata("bad", seed, 0, provenance(), wrong)

    def test_external_reference_cannot_be_owned_demonstration(self) -> None:
        with self.assertRaises(DatasetContractError):
            EpisodeProvenance(
                kind=ProvenanceKind.SOURCE_EXTERNAL_REFERENCE,
                source_hashes={"automate": HASH},
                task_config_sha256=HASH,
                runtime_versions={"source": "AutoMate-2.1.1"},
                observation_source="external_reference",
                contact_force_source="external_reference",
                owns_demonstration=True,
                external_reference="AutoMate asset_00015",
            )

    def test_round_trip_hash_and_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            metadata = EpisodeMetadata.for_seed(
                episode_id="run-001", seed=seed_for_split(DatasetSplit.TRAIN), rollout_index=0, provenance=provenance()
            )
            with EpisodeDatasetWriter(path) as writer:
                writer.begin_episode(metadata)
                writer.append(record(0.0))
                writer.append(record(0.02, success=True))
                episode_hash = writer.finalize_episode(terminal(0.02, success=True))
            self.assertEqual(len(episode_hash), 64)
            with h5py.File(path, "r") as file:
                group = file["episodes/run-001"]
                self.assertEqual(group["actions"].shape, (2, 12))
                self.assertEqual(group["object_poses_world"].shape, (2, 2, 7))
                self.assertEqual(group.attrs["episode_sha256"], episode_hash)
            report = inspect_dataset(path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["counts"]["training_eligible"], 1)

    def test_strict_shape_finite_time_and_quaternion_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            metadata = EpisodeMetadata.for_seed(
                episode_id="strict", seed=7, rollout_index=0, provenance=provenance()
            )
            with EpisodeDatasetWriter(path) as writer:
                writer.begin_episode(metadata)
                writer.append(record(0.1))
                with self.assertRaises(DatasetContractError):
                    writer.append(record(0.1))
                with self.assertRaises(DatasetContractError):
                    writer.append(record(0.2, observation_size=5))
                bad = record(0.2)
                bad.object_poses_world[0, 3:7] = 0.0
                with self.assertRaises(DatasetContractError):
                    writer.append(bad)

    def test_reset_discards_only_incomplete_episode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            metadata = EpisodeMetadata.for_seed(episode_id="discard-me", seed=2, rollout_index=0, provenance=provenance())
            writer = EpisodeDatasetWriter(path)
            writer.begin_episode(metadata)
            writer.append(record(0.0))
            writer.reset_episode()
            writer.close()
            with h5py.File(path, "r") as file:
                self.assertNotIn("discard-me", file["episodes"])
                self.assertNotIn("discard-me", file["_in_progress"])

    def test_eval_and_external_data_are_blocked_from_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            eval_meta = EpisodeMetadata.for_seed(
                episode_id="eval-run", seed=seed_for_split(DatasetSplit.EVAL), rollout_index=0, provenance=provenance()
            )
            external_seed = seed_for_split(DatasetSplit.TRAIN)
            external_meta = EpisodeMetadata.for_seed(
                episode_id="automate-reference",
                seed=external_seed,
                rollout_index=1,
                provenance=provenance(ProvenanceKind.SOURCE_EXTERNAL_REFERENCE),
            )
            with EpisodeDatasetWriter(path) as writer:
                for item in (eval_meta, external_meta):
                    writer.begin_episode(item)
                    writer.append(record(0.0))
                    writer.finalize_episode(terminal(0.0))
            self.assertEqual(select_episode_ids(path, training_eligible_only=True), [])
            with self.assertRaises(DatasetContractError):
                load_training_episode(path, "eval-run")

    def test_success_filter_and_valid_failure_rollout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            train_seed = seed_for_split(DatasetSplit.TRAIN)
            with EpisodeDatasetWriter(path) as writer:
                for index, success in enumerate((False, True)):
                    item = EpisodeMetadata.for_seed(
                        episode_id=f"run-{index}", seed=train_seed, rollout_index=index, provenance=provenance()
                    )
                    writer.begin_episode(item)
                    writer.append(record(0.0, success=success))
                    writer.finalize_episode(terminal(0.0, success=success))
            self.assertEqual(select_episode_ids(path, success_only=True), ["run-1"])
            self.assertEqual(select_episode_ids(path, training_eligible_only=True), ["run-0", "run-1"])

    def test_hash_tampering_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            item = EpisodeMetadata.for_seed(episode_id="run", seed=3, rollout_index=0, provenance=provenance())
            with EpisodeDatasetWriter(path) as writer:
                writer.begin_episode(item)
                writer.append(record(0.0))
                writer.finalize_episode(terminal(0.0))
            with h5py.File(path, "r+") as file:
                file["episodes/run/actions"][0, 0] = 9.0
            report = inspect_dataset(path)
            self.assertFalse(report["valid"])
            self.assertEqual(report["invalid_hash_episode_ids"], ["run"])

    def test_cli_outputs_json_and_success_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rollouts.h5"
            item = EpisodeMetadata.for_seed(
                episode_id="success", seed=seed_for_split(DatasetSplit.TRAIN), rollout_index=0, provenance=provenance()
            )
            with EpisodeDatasetWriter(path) as writer:
                writer.begin_episode(item)
                writer.append(record(0.0, success=True))
                writer.finalize_episode(terminal(0.0, success=True))
            result = subprocess.run(
                [sys.executable, "scripts/inspect_dataset.py", str(path), "--success-only"],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertEqual(report["selected_episode_ids"], ["success"])


if __name__ == "__main__":
    unittest.main()
