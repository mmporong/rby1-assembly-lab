from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rj45_g2_input_normalization_probe.py"
SPEC = importlib.util.spec_from_file_location("rj45_g2_input_normalization_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class InputNormalizationProbeTests(unittest.TestCase):
    def test_normalizer_constant_floor_mask_and_transform(self) -> None:
        observation = np.tile(np.arange(79, dtype=np.float32), (8, 1))
        observation[:, 1] += np.linspace(0.0, 7e-6, 8, dtype=np.float32)
        normalizer = RUNNER._normalizer(observation, (50, 62))
        self.assertTrue(normalizer["constant_mask"][0])
        self.assertEqual(float(normalizer["scale64"][0]), RUNNER.INPUT_STD_FLOOR)
        self.assertEqual(float(normalizer["scale64"][1]), RUNNER.INPUT_STD_FLOOR)
        self.assertTrue(normalizer["feature_zero_mask"][50])
        self.assertTrue(normalizer["feature_zero_mask"][62])
        transformed = RUNNER._apply_normalizer(observation, normalizer)
        self.assertTrue(np.all(transformed[:, 0] == 0.0))
        self.assertTrue(np.all(transformed[:, 50] == 0.0))
        self.assertTrue(np.isfinite(transformed).all())

    def test_normalizer_rejects_shape_nonfinite_and_bad_mask(self) -> None:
        with self.assertRaisesRegex(RUNNER.InputNormalizationProbeError, "shape"):
            RUNNER._normalizer(np.zeros((2, 78), dtype=np.float32))
        bad = np.zeros((2, 79), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(RUNNER.InputNormalizationProbeError, "finite"):
            RUNNER._normalizer(bad)
        with self.assertRaisesRegex(RUNNER.InputNormalizationProbeError, "out of range"):
            RUNNER._normalizer(np.zeros((2, 79), dtype=np.float32), (79,))

    def _checkpoint(self, path: Path, *, tamper: str | None = None) -> tuple[np.ndarray, np.ndarray]:
        import rby1_assembly.rj45_bc as bc

        torch.manual_seed(5)
        actor = bc.build_bc_actor().cpu()
        with torch.no_grad():
            actor[-1].weight.zero_()
            actor[-1].bias.fill_(0.25)
        state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        normalizer = RUNNER._normalizer(np.arange(4 * 79, dtype=np.float32).reshape(4, 79))
        run = {"state": state, "final_actor_state_sha256": bc.actor_state_sha256(state)}
        payload = RUNNER._checkpoint_payload(run, normalizer, "train_standardize")
        if tamper == "missing":
            del payload["input_transform"]
        elif tamper == "hash":
            payload["input_transform"]["mean32"][0] += 1.0
        elif tamper == "shape":
            payload["input_transform"]["scale32"] = payload["input_transform"]["scale32"][:-1]
        elif tamper == "finite":
            payload["input_transform"]["scale32"][0] = np.inf
            payload["input_transform"]["array_sha256"]["scale32"] = RUNNER._array_sha256(payload["input_transform"]["scale32"])
        torch.save(payload, path)
        return payload.get("input_transform", {}).get("mean32", np.array([])), normalizer["mean32"]

    def test_checkpoint_loader_rejects_missing_hash_shape_and_nonfinite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for mode, message in (("missing", "missing"), ("hash", "hash mismatch"), ("shape", "shape"), ("finite", "non-finite")):
                path = Path(directory) / f"{mode}.pt"
                self._checkpoint(path, tamper=mode)
                with self.assertRaisesRegex(RUNNER.InputNormalizationProbeError, message):
                    RUNNER._load_checkpoint(path)

    def test_checkpoint_prediction_matches_actor_on_transformed_input(self) -> None:
        import rby1_assembly.rj45_bc as bc

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            torch.manual_seed(9)
            actor = bc.build_bc_actor().cpu().eval()
            state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
            observation = np.random.default_rng(9).normal(size=(12, 79)).astype(np.float32)
            normalizer = RUNNER._normalizer(observation, (50, 51))
            run = {"state": state, "final_actor_state_sha256": bc.actor_state_sha256(state)}
            torch.save(RUNNER._checkpoint_payload(run, normalizer, "train_standardize_zero_previous"), path)
            transformed = RUNNER._apply_normalizer(observation, normalizer)
            with torch.no_grad():
                expected = actor(torch.as_tensor(transformed)).numpy()
            actual = RUNNER.predict_checkpoint(path, observation)
            np.testing.assert_array_equal(actual, expected)

    def test_a_gate_failure_prevents_b(self) -> None:
        axis = mock.Mock()
        axis._helper.return_value = object()
        axis._load_input.return_value = (
            np.zeros((395, 79), dtype=np.float32),
            np.zeros((395, 6), dtype=np.float32),
            {},
        )
        axis._axis_statistics.return_value = {
            "effective_std": {name: 1.0 for name in ("tx", "ty", "tz", "rx", "ry", "rz")}
        }
        axis.AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
        axis.file_sha256.return_value = "a" * 64
        axis._train_once.return_value = {"metrics": {"overall_original_coordinate_mse": 1.0}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(RUNNER, "_validate_protocol", return_value={}),
                mock.patch.object(RUNNER, "_axis_runner", return_value=axis),
                mock.patch.object(RUNNER, "_reproduction_gate", return_value={"met": False}),
                self.assertRaisesRegex(RUNNER.InputNormalizationProbeError, "B was not started"),
            ):
                RUNNER.run_probe(root / "manifest.json", protocol, root / "output", root / "result.json")
        self.assertEqual(axis._train_once.call_count, 1)

    def test_b_failure_skips_conditional_contrasts(self) -> None:
        a = {"metrics": {"overall_original_coordinate_mse": 1.0}}
        b = {
            "metrics": {
                "overall_original_coordinate_mse": 1.0,
                "axis_balanced_mse": 1.0,
                "axis_original_coordinate_mse": {name: 1.0 for name in ("tx", "ty", "tz", "rx", "ry", "rz")},
            },
            "diagnostics": [{"axis_balanced_mse": 1.0}] * 5,
        }
        baseline = {
            "overall_original_coordinate_mse": 1.0,
            "axis_original_coordinate_mse": {name: 1.0 for name in ("tx", "ty", "tz", "rx", "ry", "rz")},
        }
        criterion = RUNNER._b_criterion(a, b, baseline)
        self.assertFalse(criterion["met"])
        with mock.patch.object(RUNNER, "_run_feature_contrasts") as contrasts:
            observed = RUNNER._maybe_run_feature_contrasts(
                criterion,
                mock.Mock(),
                np.zeros((395, 79), dtype=np.float32),
                np.zeros((395, 6), dtype=np.float32),
                np.ones(6, dtype=np.float32),
                np.ones(6, dtype=np.float64),
                ("a" * 64, "b" * 64),
            )
        self.assertEqual(observed, ({}, {}, {}))
        contrasts.assert_not_called()

    def test_protocol_tamper_no_overwrite_and_atomic_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"frozen_at_utc": "x", "value": 2}), encoding="utf-8")
            with (
                mock.patch.object(RUNNER, "_protocol_payload", return_value={"frozen_at_utc": "x", "value": 1}),
                self.assertRaisesRegex(RUNNER.InputNormalizationProbeError, "protocol identity"),
            ):
                RUNNER._validate_protocol(protocol, root / "manifest.json")
            with self.assertRaises(FileExistsError):
                RUNNER.freeze_protocol(root / "manifest.json", protocol)
            occupied = root / "output"
            occupied.mkdir()
            with self.assertRaises(FileExistsError):
                RUNNER.run_probe(root / "manifest.json", protocol, occupied, root / "result.json")

    def test_small_synthetic_training_uses_reused_loop_and_final_checkpoint_only(self) -> None:
        axis = RUNNER._axis_runner()
        rng = np.random.default_rng(31)
        observation = rng.normal(size=(48, 79)).astype(np.float32)
        action = (np.tanh(observation[:, :6]) * np.asarray([0.002, 0.004, 0.008, 0.016, 0.032, 0.064], dtype=np.float32))
        axis_std = np.std(action.astype(np.float64), axis=0, ddof=0)
        normalized = RUNNER._apply_normalizer(observation, RUNNER._normalizer(observation))
        baseline = axis._constant_baseline(action, axis_std)["overall_original_coordinate_mse"]
        with mock.patch.object(axis, "STEPS", 40), mock.patch.object(axis, "REPORT_INTERVAL", 20), mock.patch.object(axis, "THREADS", 1):
            trained = axis._train_once(normalized, action, axis_std.astype(np.float32), axis_std, "axis_scaled_mse")
        self.assertTrue(math.isfinite(trained["metrics"]["overall_original_coordinate_mse"]))
        self.assertLess(trained["metrics"]["overall_original_coordinate_mse"], baseline)
        self.assertEqual(len(trained["diagnostics"]), 2)


if __name__ == "__main__":
    unittest.main()
