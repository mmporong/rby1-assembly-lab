from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from rby1_assembly.rj45_bc import BcContractError, load_bc_actor_artifact
from rby1_assembly.rj45_bc_training import BcRecipe, observability_diagnostic, train_bc
from rby1_assembly.rj45_teacher_dataset import (
    TeacherDatasetWriter,
    TeacherEpisodeProvenance,
    TeacherEpisodeRows,
    TeacherEpisodeTerminal,
)


HASHES = {
    "source_sha256": "1" * 64,
    "task_sha256": "2" * 64,
    "training_sha256": "3" * 64,
    "bank_sha256": "4" * 64,
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_dataset(
    path: Path,
    split: str,
    *,
    grasp: str,
    bias: str,
    seed: int,
    observation: np.ndarray,
    action: np.ndarray,
) -> None:
    count = observation.shape[0]
    requested = action.astype(np.float32)
    with TeacherDatasetWriter(path) as writer:
        writer.add_episode(
            f"{split}-episode",
            TeacherEpisodeProvenance(
                grasp_id=grasp,
                split=split,
                bias_case_id=bias,
                seed=seed,
                translation_bound_m=0.0015,
                rotation_bound_rad=0.02,
                **HASHES,
            ),
            TeacherEpisodeRows(
                observation=observation.astype(np.float32),
                observation_timestamp=np.arange(1, count + 1, dtype=np.float64) * 0.01,
                valid=np.ones(count, dtype=np.bool_),
                teacher_requested=requested,
                teacher_applied=requested,
                applied_offset=requested * np.asarray([0.0015] * 3 + [0.02] * 3, dtype=np.float32),
                phase=["INSERT"] * count,
                timestep=np.arange(count, dtype=np.int64),
                teacher_gt_delta=np.full((count, 6), 999.0, dtype=np.float32),
            ),
            TeacherEpisodeTerminal(True, True, ""),
        )


def _recipe(path: Path, **changes: object) -> None:
    payload = {
        "schema_version": 1,
        "epochs": 8,
        "batch_size": 8,
        "learning_rate": 0.003,
        "seed": 77,
        "threads": 1,
    }
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


class BcTrainingTests(unittest.TestCase):
    def _inputs(self, root: Path) -> tuple[Path, Path, Path]:
        generator = np.random.default_rng(10)
        train_observation = generator.normal(0.0, 0.2, size=(32, 79)).astype(np.float32)
        dev_observation = generator.normal(0.0, 0.2, size=(16, 79)).astype(np.float32)
        train_action = np.tanh(train_observation[:, :6]) * 0.4
        dev_action = np.tanh(dev_observation[:, :6]) * 0.4
        train = root / "train.h5"
        dev = root / "dev.h5"
        recipe = root / "recipe.json"
        _write_dataset(train, "train", grasp="train-grasp", bias="train-bias", seed=100, observation=train_observation, action=train_action)
        _write_dataset(dev, "dev", grasp="dev-grasp", bias="dev-bias", seed=200, observation=dev_observation, action=dev_action)
        _recipe(recipe)
        return train, dev, recipe

    def test_actual_hdf5_tiny_train_reload_and_privileged_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, recipe = self._inputs(root)
            report = train_bc(
                train_dataset=train,
                train_sha256=_sha(train),
                dev_dataset=dev,
                dev_sha256=_sha(dev),
                recipe_path=recipe,
                output_dir=root / "out",
                enable_observability_diagnostic=True,
            )
            self.assertLess(report["best_dev_mse"], report["initial_dev_mse"])
            self.assertEqual(report["total_bc_updates"], 32)
            self.assertEqual(report["sample_visits"], 256)
            self.assertFalse(report["claims"]["ppo_checkpoint"])
            self.assertFalse(report["claims"]["critic_or_std_trained"])
            self.assertFalse(report["claims"]["privileged_teacher_ground_truth_used_as_student_input"])
            artifact = load_bc_actor_artifact(root / "out/bc_actor_init.pt")
            self.assertEqual(artifact["artifact_kind"], "bc_actor_initialization_not_ppo_checkpoint")
            self.assertEqual(report["observability_diagnostic"]["evaluated_rows"], 0)
            self.assertEqual(report["observability_diagnostic"]["skipped_without_other_episode"], 32)

    def test_dev_rows_are_selection_only_and_earliest_tie_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            observation = np.zeros((8, 79), dtype=np.float32)
            train = root / "train.h5"
            dev = root / "dev.h5"
            recipe = root / "recipe.json"
            _write_dataset(train, "train", grasp="g1", bias="b1", seed=1, observation=observation, action=np.zeros((8, 6), dtype=np.float32))
            _write_dataset(dev, "dev", grasp="g2", bias="b2", seed=2, observation=observation, action=np.ones((8, 6), dtype=np.float32) * 0.5)
            _recipe(recipe, epochs=2)
            report = train_bc(train_dataset=train, train_sha256=_sha(train), dev_dataset=dev, dev_sha256=_sha(dev), recipe_path=recipe, output_dir=root / "out")
            self.assertEqual(report["best_epoch"], 1)
            self.assertEqual(report["best_dev_mse"], report["initial_dev_mse"])

    def test_split_collision_and_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, recipe = self._inputs(root)
            observation = np.zeros((4, 79), dtype=np.float32)
            replacement = root / "collision.h5"
            _write_dataset(replacement, "dev", grasp="train-grasp", bias="other", seed=3, observation=observation, action=np.zeros((4, 6), dtype=np.float32))
            with self.assertRaisesRegex(BcContractError, "grasp_id crosses"):
                train_bc(train_dataset=train, train_sha256=_sha(train), dev_dataset=replacement, dev_sha256=_sha(replacement), recipe_path=recipe, output_dir=root / "collision-out")
            bias_collision = root / "bias-collision.h5"
            _write_dataset(bias_collision, "dev", grasp="other-grasp", bias="train-bias", seed=100, observation=observation, action=np.zeros((4, 6), dtype=np.float32))
            with self.assertRaisesRegex(BcContractError, "bias_case_id/seed crosses"):
                train_bc(train_dataset=train, train_sha256=_sha(train), dev_dataset=bias_collision, dev_sha256=_sha(bias_collision), recipe_path=recipe, output_dir=root / "bias-collision-out")
            with self.assertRaisesRegex(BcContractError, "SHA-256"):
                train_bc(train_dataset=train, train_sha256="0" * 64, dev_dataset=dev, dev_sha256=_sha(dev), recipe_path=recipe, output_dir=root / "sha-out")

    def test_observability_diagnostic_uses_other_episode_only(self):
        observation = torch.zeros((4, 79), dtype=torch.float32)
        observation[2:] = 0.1
        action = torch.zeros((4, 6), dtype=torch.float32)
        action[2:] = 0.5
        report = observability_diagnostic(
            observation,
            action,
            ["first", "first", "second", "second"],
            seed=9,
        )
        self.assertEqual(report["evaluated_rows"], 4)
        self.assertEqual(report["skipped_without_other_episode"], 0)
        self.assertAlmostEqual(report["mean_nearest_observation_rms"], 0.1, places=6)
        self.assertAlmostEqual(report["mean_corresponding_label_rms"], 0.5, places=6)

    def test_deterministic_actor_state_and_final_seed_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, recipe = self._inputs(root)
            first = train_bc(train_dataset=train, train_sha256=_sha(train), dev_dataset=dev, dev_sha256=_sha(dev), recipe_path=recipe, output_dir=root / "first")
            second = train_bc(train_dataset=train, train_sha256=_sha(train), dev_dataset=dev, dev_sha256=_sha(dev), recipe_path=recipe, output_dir=root / "second")
            self.assertEqual(first["artifact"]["actor_state_sha256"], second["artifact"]["actor_state_sha256"])
            self.assertEqual(first["epoch_metrics"], second["epoch_metrics"])
            _recipe(recipe, seed=20261027)
            with self.assertRaisesRegex(BcContractError, "final holdout"):
                BcRecipe.load(recipe)

    def test_cli_trains_actual_hdf5_and_requires_new_output_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train, dev, recipe = self._inputs(root)
            output = root / "cli-output"
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            command = [
                sys.executable,
                str(Path(__file__).resolve().parents[1] / "scripts/train_rj45_bc.py"),
                "--train-dataset", str(train),
                "--train-sha256", _sha(train),
                "--dev-dataset", str(dev),
                "--dev-sha256", _sha(dev),
                "--recipe", str(recipe),
                "--output-dir", str(output),
            ]
            completed = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "bc_actor_init.pt").is_file())
            self.assertTrue((output / "training_report.json").is_file())
            repeated = subprocess.run(command, text=True, capture_output=True, env=environment, timeout=20)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("FileExistsError", repeated.stderr)


if __name__ == "__main__":
    unittest.main()
