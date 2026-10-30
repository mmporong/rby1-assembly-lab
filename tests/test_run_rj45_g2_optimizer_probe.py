from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_rj45_g2_optimizer_probe.py"
HELPER_PATH = ROOT / "scripts/probe_rj45_train_condition.py"
SPEC = importlib.util.spec_from_file_location("run_rj45_g2_optimizer_probe", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

HELPER_SPEC = importlib.util.spec_from_file_location("probe_rj45_train_condition_for_g2", HELPER_PATH)
assert HELPER_SPEC is not None and HELPER_SPEC.loader is not None
HELPER = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(HELPER)

try:
    import torch  # noqa: F401
except ImportError:
    torch = None


def _fake_result(learning_rate: float, final_mse: float, diagnostics: list[float]) -> dict:
    return {
        "recipe": {
            "seed": RUNNER.SEED,
            "threads": RUNNER.THREADS,
            "learning_rate": learning_rate,
            "full_batch_optimizer_steps": RUNNER.STEPS,
            "report_interval_steps": RUNNER.REPORT_INTERVAL,
        },
        "metrics": {
            "final_actor": {"overall_normalized_mse": final_mse},
            "constant_train_mean_action": {"overall_normalized_mse": 1.0},
        },
        "diagnostics": [
            {"optimizer_step": index, "train_normalized_mse": value}
            for index, value in enumerate(diagnostics, start=1)
        ],
    }


class OptimizerProbeContractTests(unittest.TestCase):
    def test_pair_requires_exact_recipes_and_matching_initial_state(self) -> None:
        runs = {
            "e0": _fake_result(0.001, 0.8, [1.0] * 5),
            "e1": _fake_result(0.0001, 0.7, [1.0] * 5),
        }
        RUNNER._validate_run_pair(runs, {"e0": "a" * 64, "e1": "a" * 64})
        with self.assertRaisesRegex(RUNNER.OptimizerProbeError, "same actor state"):
            RUNNER._validate_run_pair(runs, {"e0": "a" * 64, "e1": "b" * 64})
        runs["e1"]["recipe"]["learning_rate"] = 0.001
        with self.assertRaisesRegex(RUNNER.OptimizerProbeError, "frozen contract"):
            RUNNER._validate_run_pair(runs, {"e0": "a" * 64, "e1": "a" * 64})

    def test_criterion_requires_all_three_conditions(self) -> None:
        passing = {
            "e0": _fake_result(0.001, 0.95, [1.0] * 5),
            "e1": _fake_result(0.0001, 0.8, [1.0, 0.95, 0.9, 0.85, 0.8]),
        }
        outcome = RUNNER._criterion(passing)
        self.assertTrue(outcome["met"])
        failing = {
            "e0": _fake_result(0.001, 0.75, [1.0] * 5),
            "e1": _fake_result(0.0001, 0.8, [1.0, 0.95, 0.97, 0.85, 0.8]),
        }
        outcome = RUNNER._criterion(failing)
        self.assertFalse(outcome["met"])
        self.assertFalse(outcome["checks"]["e1_lower_than_e0"])
        self.assertFalse(outcome["checks"]["e1_last_five_monotonic_nonincrease"])


@unittest.skipIf(torch is None, "torch is unavailable in this interpreter")
class OptimizerProbeExecutionTests(unittest.TestCase):
    def _write_input(self, root: Path, count: int = 24) -> tuple[Path, str]:
        rng = np.random.default_rng(19)
        observation = rng.normal(size=(count, 79)).astype(np.float32)
        action = np.stack(
            [np.tanh(observation[:, axis]) * (0.02 + axis * 0.005) for axis in range(6)],
            axis=1,
        ).astype(np.float32)
        valid = np.ones(count, dtype=np.bool_)
        npz = root / "train-condition.npz"
        np.savez_compressed(npz, observation=observation, action=action, valid_mask=valid)
        npz_sha = RUNNER.file_sha256(npz)
        manifest = {
            "schema_name": HELPER.SCHEMA_NAME,
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
                "grasp_id": HELPER.EXPECTED_GRASP_ID,
                "bias_case_id": HELPER.EXPECTED_BIAS_CASE_ID,
                "seed": HELPER.EXPECTED_SEED,
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
                "teacher_dataset": {"path": "/synthetic", "sha256": HELPER.EXPECTED_DATASET_SHA256},
                "script": {"path": str(HELPER_PATH), "sha256": HELPER.file_sha256(HELPER_PATH)},
            },
            "arrays": {
                "observation": {"sha256": HELPER.array_sha256(observation)},
                "action": {"sha256": HELPER.array_sha256(action)},
                "valid_mask": {"sha256": HELPER.array_sha256(valid)},
            },
            "row_counts": {"episode_total": count, "valid_action": count},
            "shapes": {
                "observation": list(observation.shape),
                "action": list(action.shape),
                "valid_mask": list(valid.shape),
            },
            "npz": {"path": str(npz), "sha256": npz_sha},
            "runtime": {"python": "test", "numpy": np.__version__, "h5py": "test"},
        }
        path = root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path, npz_sha

    def test_small_two_run_execution_and_protocol_tamper_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, npz_sha = self._write_input(root)
            protocol = root / "protocol.json"
            output = root / "output"
            result = root / "result.json"
            recipes = (
                {"id": "e0", "learning_rate": 0.001},
                {"id": "e1", "learning_rate": 0.0001},
            )
            with (
                mock.patch.object(RUNNER, "EXPECTED_NPZ_SHA256", npz_sha),
                mock.patch.object(RUNNER, "STEPS", 10),
                mock.patch.object(RUNNER, "REPORT_INTERVAL", 2),
                mock.patch.object(RUNNER, "RECIPES", recipes),
            ):
                RUNNER.freeze_protocol(manifest, protocol)
                payload = RUNNER.run_probe(manifest, protocol, output, result)
                self.assertTrue(payload["initial_actor"]["same_state_for_both_runs"])
                self.assertEqual(set(payload["runs"]), {"e0", "e1"})
                self.assertTrue(payload["stop_rule_observed"])
                self.assertEqual(
                    sorted(path.name for path in output.iterdir()),
                    ["e0", "e1", "result.json"],
                )
                with self.assertRaises(FileExistsError):
                    RUNNER.run_probe(manifest, protocol, output, root / "other-result.json")

            tampered = json.loads(protocol.read_text(encoding="utf-8"))
            tampered["recipes"][1]["learning_rate"] = 0.0002
            protocol.write_text(json.dumps(tampered), encoding="utf-8")
            with (
                mock.patch.object(RUNNER, "EXPECTED_NPZ_SHA256", npz_sha),
                mock.patch.object(RUNNER, "STEPS", 10),
                mock.patch.object(RUNNER, "REPORT_INTERVAL", 2),
                mock.patch.object(RUNNER, "RECIPES", recipes),
                self.assertRaisesRegex(RUNNER.OptimizerProbeError, "protocol identity"),
            ):
                RUNNER.run_probe(manifest, protocol, root / "tampered-output", root / "tampered.json")


if __name__ == "__main__":
    unittest.main()
