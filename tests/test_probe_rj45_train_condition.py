from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/probe_rj45_train_condition.py"
SPEC = importlib.util.spec_from_file_location("probe_rj45_train_condition", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)

try:
    import h5py  # noqa: F401
except ImportError:
    h5py = None

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


class ProbeValidationTests(unittest.TestCase):
    def test_array_validation_rejects_empty_nan_shape_and_out_of_range(self) -> None:
        valid = np.ones(2, dtype=np.bool_)
        cases = [
            (np.zeros((0, 79), np.float32), np.zeros((0, 6), np.float32), np.zeros(0, np.bool_)),
            (np.zeros((2, 78), np.float32), np.zeros((2, 6), np.float32), valid),
            (np.full((2, 79), np.nan, np.float32), np.zeros((2, 6), np.float32), valid),
            (np.zeros((2, 79), np.float32), np.full((2, 6), 1.1, np.float32), valid),
            (np.zeros((2, 79), np.float32), np.zeros((2, 6), np.float32), np.zeros(2, np.bool_)),
        ]
        for observation, action, mask in cases:
            with self.subTest(shape=observation.shape), self.assertRaises(PROBE.ProbeContractError):
                PROBE._validate_selected_arrays(observation, action, mask)

    def test_output_directory_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "exists"
            output.mkdir()
            with self.assertRaises(FileExistsError):
                PROBE._atomic_directory(output)

    def test_bad_hash_is_rejected_before_export_dependency_work(self) -> None:
        with self.assertRaisesRegex(PROBE.ProbeContractError, "SHA-256"):
            PROBE._require_sha256("bad", "test")

    def test_constant_mean_baseline_distinguishes_state_dependent_prediction(self) -> None:
        state = np.asarray([-2.0, -1.0, 1.0, 2.0], dtype=np.float64)
        target = np.zeros((4, 6), dtype=np.float64)
        target[:, 0] = state * 0.1
        target[:, 3] = -state * 0.05
        constant = PROBE._constant_train_mean_action_baseline(target)
        state_dependent = PROBE._metric_payload(target.copy(), target)
        self.assertGreater(constant["overall_normalized_mse"], 0.0)
        self.assertEqual(state_dependent["overall_normalized_mse"], 0.0)
        self.assertAlmostEqual(constant["mean_normalized_action"]["tx"], 0.0)
        self.assertAlmostEqual(constant["mean_normalized_action"]["rx"], 0.0)


@unittest.skipIf(h5py is None, "h5py is unavailable in this interpreter")
class ProbeExportTests(unittest.TestCase):
    def test_export_filters_one_fixed_successful_train_episode_and_preserves_lineage(self) -> None:
        from rby1_assembly.rj45_teacher_dataset import (
            TeacherDatasetWriter,
            TeacherEpisodeProvenance,
            TeacherEpisodeRows,
            TeacherEpisodeTerminal,
        )

        def provenance(grasp: str, bias: str, seed: int) -> TeacherEpisodeProvenance:
            return TeacherEpisodeProvenance(
                grasp_id=grasp,
                split="train",
                bias_case_id=bias,
                seed=seed,
                source_sha256="a" * 64,
                task_sha256="b" * 64,
                training_sha256="c" * 64,
                bank_sha256="d" * 64,
                translation_bound_m=0.01,
                rotation_bound_rad=0.2,
            )

        def rows(valid: np.ndarray) -> TeacherEpisodeRows:
            count = len(valid)
            action = np.zeros((count, 6), dtype=np.float32)
            action[:, 0] = 0.2
            return TeacherEpisodeRows(
                observation=np.arange(count * 79, dtype=np.float32).reshape(count, 79),
                observation_timestamp=np.arange(count, dtype=np.float64) * 0.02,
                valid=valid,
                teacher_requested=action.copy(),
                teacher_applied=action,
                applied_offset=action * np.asarray([0.01] * 3 + [0.2] * 3, np.float32),
                phase=["align"] * count,
                timestep=np.arange(count, dtype=np.int64),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.h5"
            with TeacherDatasetWriter(dataset) as writer:
                writer.add_episode(
                    "target",
                    provenance(PROBE.EXPECTED_GRASP_ID, PROBE.EXPECTED_BIAS_CASE_ID, PROBE.EXPECTED_SEED),
                    rows(np.asarray([False, True, True], dtype=np.bool_)),
                    TeacherEpisodeTerminal(True, True, ""),
                )
                writer.add_episode(
                    "other",
                    provenance("rj45-train-03", "env-1-episode-1", 9),
                    rows(np.asarray([True, True], dtype=np.bool_)),
                    TeacherEpisodeTerminal(True, True, ""),
                )
            output = root / "export"
            manifest = PROBE.export_condition(
                dataset,
                output,
                expected_dataset_sha256=PROBE.file_sha256(dataset),
            )
            self.assertEqual(manifest["condition"]["episode_ids"], ["target"])
            self.assertEqual(manifest["row_counts"], {"episode_total": 3, "valid_action": 2})
            self.assertEqual(manifest["lineage"]["source_sha256"], "a" * 64)
            with np.load(output / "train-condition.npz", allow_pickle=False) as payload:
                self.assertEqual(payload["observation"].shape, (3, 79))
                self.assertEqual(payload["action"].shape, (3, 6))
                np.testing.assert_array_equal(payload["valid_mask"], [False, True, True])
            with self.assertRaisesRegex(PROBE.ProbeContractError, "SHA-256"):
                PROBE.export_condition(
                    dataset,
                    root / "wrong-hash",
                    expected_dataset_sha256="0" * 64,
                )
            with h5py.File(dataset, "r+") as file:
                file["episodes/other"].attrs["grasp_id"] = PROBE.EXPECTED_GRASP_ID
                file["episodes/other"].attrs["bias_case_id"] = PROBE.EXPECTED_BIAS_CASE_ID
            with self.assertRaisesRegex(PROBE.ProbeContractError, "exactly one"):
                PROBE.export_condition(
                    dataset,
                    root / "mixed-condition",
                    expected_dataset_sha256=PROBE.file_sha256(dataset),
                )

    def test_export_rejects_metadata_mismatch(self) -> None:
        from rby1_assembly.rj45_teacher_dataset import (
            TeacherDatasetWriter,
            TeacherEpisodeProvenance,
            TeacherEpisodeRows,
            TeacherEpisodeTerminal,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "teacher.h5"
            provenance = TeacherEpisodeProvenance(
                grasp_id=PROBE.EXPECTED_GRASP_ID,
                split="train",
                bias_case_id=PROBE.EXPECTED_BIAS_CASE_ID,
                seed=123,
                source_sha256="a" * 64,
                task_sha256="b" * 64,
                training_sha256="c" * 64,
                bank_sha256="d" * 64,
                translation_bound_m=0.01,
                rotation_bound_rad=0.2,
            )
            action = np.zeros((1, 6), np.float32)
            rows = TeacherEpisodeRows(
                observation=np.zeros((1, 79), np.float32),
                observation_timestamp=np.zeros(1, np.float64),
                valid=np.ones(1, np.bool_),
                teacher_requested=action,
                teacher_applied=action,
                applied_offset=action,
                phase=["align"],
                timestep=np.zeros(1, np.int64),
            )
            with TeacherDatasetWriter(dataset) as writer:
                writer.add_episode("wrong-seed", provenance, rows, TeacherEpisodeTerminal(True, True, ""))
            with self.assertRaisesRegex(PROBE.ProbeContractError, "seed"):
                PROBE.export_condition(
                    dataset,
                    root / "output",
                    expected_dataset_sha256=PROBE.file_sha256(dataset),
                )


@unittest.skipIf(torch is None, "torch is unavailable in this interpreter")
class ProbeFitTests(unittest.TestCase):
    def _write_input(self, root: Path, count: int = 32) -> Path:
        rng = np.random.default_rng(7)
        observation = rng.normal(size=(count, 79)).astype(np.float32)
        weights = rng.normal(scale=0.03, size=(79, 6)).astype(np.float32)
        action = np.tanh(observation @ weights).astype(np.float32) * 0.2
        valid = np.ones(count, dtype=np.bool_)
        npz = root / "train-condition.npz"
        np.savez_compressed(npz, observation=observation, action=action, valid_mask=valid)
        manifest = {
            "schema_name": PROBE.SCHEMA_NAME,
            "artifact_kind": "train_only_fixed_condition_offline_bc_probe_input",
            "claims": {
                "train_only": True,
                "heldout_generalization": False,
                "reinforcement_learning": False,
                "closed_loop_success": False,
                "main_bc_actor_reusable": False,
            },
            "condition": {
                "episode_ids": ["synthetic"],
                "grasp_id": PROBE.EXPECTED_GRASP_ID,
                "bias_case_id": PROBE.EXPECTED_BIAS_CASE_ID,
                "seed": PROBE.EXPECTED_SEED,
                "split": "train",
                "successful_and_terminal_valid": True,
            },
            "contract": {
                "observation": "standard79_without_additional_normalization",
                "action": (
                    "normalized teacher_applied 6D BC command; translation and rotation "
                    "triplets are separately unit-ball bounded, not physical micrometre errors"
                ),
                "fit_rows": "valid_mask_true_only",
            },
            "lineage": {
                "source_sha256": "a" * 64,
                "task_sha256": "b" * 64,
                "training_sha256": "c" * 64,
                "bank_sha256": "d" * 64,
            },
            "inputs": {
                "teacher_dataset": {"path": "/synthetic", "sha256": PROBE.EXPECTED_DATASET_SHA256},
                "script": {"path": str(SCRIPT), "sha256": PROBE.file_sha256(SCRIPT)},
            },
            "arrays": {
                "observation": {"sha256": PROBE.array_sha256(observation)},
                "action": {"sha256": PROBE.array_sha256(action)},
                "valid_mask": {"sha256": PROBE.array_sha256(valid)},
            },
            "row_counts": {"episode_total": count, "valid_action": count},
            "shapes": {
                "observation": list(observation.shape),
                "action": list(action.shape),
                "valid_mask": list(valid.shape),
            },
            "npz": {"path": str(npz), "sha256": PROBE.file_sha256(npz)},
            "runtime": {"python": "test", "numpy": np.__version__, "h5py": "test"},
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def test_small_synthetic_fit_reduces_mse_and_saves_only_final_actor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_input(root)
            result = PROBE.fit_condition(
                manifest,
                root / "fit",
                steps=100,
                seed=PROBE.FIT_SEED,
                learning_rate=0.001,
                threads=2,
                report_interval=20,
            )
            metrics = result["metrics"]
            self.assertLess(
                metrics["final_actor"]["overall_normalized_mse"],
                metrics["initial_zero_output"]["overall_normalized_mse"],
            )
            self.assertTrue(metrics["parameters_finite"])
            self.assertEqual(len(result["diagnostics"]), 5)
            self.assertEqual(
                sorted(path.name for path in (root / "fit").iterdir()),
                ["final-diagnostic-actor.pt", "fit-result.json"],
            )

    def test_fit_rejects_tampered_npz_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self._write_input(root)
            payload = json.loads(manifest.read_text())
            payload["npz"]["sha256"] = "0" * 64
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PROBE.ProbeContractError, "NPZ SHA-256"):
                PROBE.fit_condition(manifest, root / "fit", steps=1)


if __name__ == "__main__":
    unittest.main()
