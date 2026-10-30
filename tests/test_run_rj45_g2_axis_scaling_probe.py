from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts/run_rj45_g2_axis_scaling_probe.py"
SPEC = importlib.util.spec_from_file_location("run_rj45_g2_axis_scaling_probe", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)

try:
    import torch
except ImportError:
    torch = None


class AxisScalingStatisticsTests(unittest.TestCase):
    def test_population_std_and_zero_variance_floor(self) -> None:
        action = np.array(
            [[1, 2, 3, 4, 5, 9], [3, 4, 5, 6, 7, 9]], dtype=np.float32
        )
        stats = RUNNER._axis_statistics(action)
        self.assertEqual(stats["axis_order"], list(RUNNER.AXIS_NAMES))
        self.assertEqual(stats["estimator"], "numpy_float64_population_std_ddof_0")
        self.assertEqual(stats["raw_std"]["tx"], 1.0)
        self.assertEqual(stats["raw_std"]["rz"], 0.0)
        self.assertEqual(stats["effective_std"]["rz"], RUNNER.STD_FLOOR)
        self.assertTrue(stats["floor_applied"]["rz"])

    def test_shape_nonfinite_and_std_guards(self) -> None:
        with self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "shape"):
            RUNNER._axis_statistics(np.zeros((3, 5), dtype=np.float32))
        bad = np.zeros((3, 6), dtype=np.float32)
        bad[0, 0] = np.nan
        with self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "finite"):
            RUNNER._axis_statistics(bad)
        with self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "positive"):
            RUNNER._axis_metrics(
                np.zeros((3, 6)), np.zeros((3, 6)), np.zeros(6)
            )

    def test_null_correlation_and_constant_balanced_sanity(self) -> None:
        target = np.stack([np.arange(8, dtype=np.float64) + axis for axis in range(6)], axis=1)
        std = np.std(target, axis=0, ddof=0)
        baseline = RUNNER._constant_baseline(target, std)
        self.assertAlmostEqual(baseline["axis_balanced_mse"], 1.0, places=14)
        self.assertTrue(all(value is None for value in baseline["axis_correlation"].values()))

    def test_criterion_requires_every_condition(self) -> None:
        baseline = {
            "overall_original_coordinate_mse": 1.0,
            "axis_original_coordinate_mse": {name: 1.0 for name in RUNNER.AXIS_NAMES},
        }
        r0 = {"metrics": {"overall_original_coordinate_mse": 0.95}}
        r1 = {
            "metrics": {
                "overall_original_coordinate_mse": 0.8,
                "axis_balanced_mse": 0.85,
                "axis_original_coordinate_mse": {name: 1.05 for name in RUNNER.AXIS_NAMES},
            }
        }
        self.assertTrue(RUNNER._criterion(r0, r1, baseline)["met"])
        r1["metrics"]["axis_original_coordinate_mse"]["rz"] = 1.11
        outcome = RUNNER._criterion(r0, r1, baseline)
        self.assertFalse(outcome["met"])
        self.assertFalse(outcome["checks"]["all_axis_ratios_at_most_1_10"])


@unittest.skipIf(torch is None, "torch is unavailable in this interpreter")
class AxisScalingExecutionTests(unittest.TestCase):
    def test_equal_standardized_errors_contribute_equally(self) -> None:
        prediction = torch.zeros((1, 6), dtype=torch.float32)
        target = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.float32)
        std = target[0].clone()
        scaled_error = ((prediction - target) / std).square()[0]
        self.assertTrue(torch.equal(scaled_error, torch.ones(6)))
        self.assertEqual(float(RUNNER._training_loss(prediction, target, std, "axis_scaled_mse")), 1.0)

    def test_inference_artifact_preserves_original_output_coordinates(self) -> None:
        import rby1_assembly.rj45_bc as bc

        torch.manual_seed(11)
        actor = bc.build_bc_actor()
        state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        run = {
            "state": state,
            "final_actor_state_sha256": bc.actor_state_sha256(state),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "actor.pt"
            RUNNER._save_actor(path, run)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(payload["artifact_kind"], "diagnostic_not_main_bc")
            self.assertEqual(
                payload["inference_output_coordinates"], "original_normalized_6d_command"
            )
            restored = bc.build_bc_actor()
            restored.load_state_dict(payload["actor_state_dict"], strict=True)
            observation = torch.randn(4, 79)
            self.assertTrue(torch.equal(actor(observation), restored(observation)))

    def test_small_synthetic_axis_scaled_learning(self) -> None:
        rng = np.random.default_rng(71)
        observation = rng.normal(size=(48, 79)).astype(np.float32)
        scales = np.asarray([0.002, 0.004, 0.008, 0.016, 0.032, 0.064], dtype=np.float32)
        action = np.tanh(observation[:, :6]) * scales
        std = np.std(action.astype(np.float64), axis=0, ddof=0).astype(np.float32)
        initial = RUNNER._constant_baseline(action, std)["overall_original_coordinate_mse"]
        with (
            mock.patch.object(RUNNER, "STEPS", 40),
            mock.patch.object(RUNNER, "REPORT_INTERVAL", 20),
            mock.patch.object(RUNNER, "THREADS", 1),
        ):
            trained = RUNNER._train_once(observation, action, std, std.astype(np.float64), "axis_scaled_mse")
        final = trained["metrics"]["overall_original_coordinate_mse"]
        self.assertTrue(math.isfinite(final))
        self.assertLess(final, initial)
        self.assertEqual(len(trained["diagnostics"]), 2)

    def test_r0_mismatch_prevents_r1_call(self) -> None:
        protocol = {
            "source_identity": {
                "axis_statistics": {
                    "effective_std": {name: 1.0 for name in RUNNER.AXIS_NAMES}
                }
            }
        }
        fake_r0 = {
            "initial_actor_state_sha256": "a" * 64,
            "final_actor_state_sha256": "b" * 64,
            "metrics": {"overall_original_coordinate_mse": 9.0},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            protocol_path = root / "protocol.json"
            manifest.write_text("{}", encoding="utf-8")
            protocol_path.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(RUNNER, "_validate_protocol", return_value=protocol),
                mock.patch.object(
                    RUNNER,
                    "_load_input",
                    return_value=(np.zeros((395, 79), np.float32), np.zeros((395, 6), np.float32), {}),
                ),
                mock.patch.object(RUNNER, "_train_once", return_value=fake_r0) as train,
                mock.patch.object(RUNNER, "_historical_gate", return_value={"met": False}),
                self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "R1 was not started"),
            ):
                RUNNER.run_probe(manifest, protocol_path, root / "output", root / "result.json")
            self.assertEqual(train.call_count, 1)

    def test_runtime_protocol_tamper_and_no_overwrite(self) -> None:
        with self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "runtime differs"):
            RUNNER._require_runtime({"python": "0", "numpy": "0", "torch": "0"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            protocol.write_text("{}", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                RUNNER.freeze_protocol(root / "manifest.json", protocol)

            expected = {"frozen_at_utc": "first", "value": 1}
            protocol.write_text(json.dumps({"frozen_at_utc": "first", "value": 2}), encoding="utf-8")
            with (
                mock.patch.object(RUNNER, "_protocol_payload", return_value=expected),
                self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "protocol identity"),
            ):
                RUNNER._validate_protocol(protocol, root / "manifest.json")

            occupied_output = root / "occupied-output"
            occupied_output.mkdir()
            with (
                mock.patch.object(RUNNER, "_validate_protocol") as validate,
                self.assertRaises(FileExistsError),
            ):
                RUNNER.run_probe(
                    root / "manifest.json",
                    protocol,
                    occupied_output,
                    root / "new-result.json",
                )
            validate.assert_not_called()

            existing_result = root / "existing-result.json"
            existing_result.write_text("{}", encoding="utf-8")
            with (
                mock.patch.object(RUNNER, "_validate_protocol") as validate,
                self.assertRaises(FileExistsError),
            ):
                RUNNER.run_probe(
                    root / "manifest.json",
                    protocol,
                    root / "new-output",
                    existing_result,
                )
            validate.assert_not_called()

    def test_tampered_npz_rejected_by_input_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            path.write_bytes(b"approved")
            approved_sha = RUNNER.file_sha256(path)
            manifest = {"npz": {"path": str(path)}}
            observation = np.zeros((395, 79), dtype=np.float32)
            action = np.zeros((395, 6), dtype=np.float32)

            class FakeHelper:
                @staticmethod
                def _load_probe_input(_manifest_path: Path):
                    return manifest, observation, action

            path.write_bytes(b"tampered")
            with (
                mock.patch.object(RUNNER, "EXPECTED_NPZ_SHA256", approved_sha),
                self.assertRaisesRegex(RUNNER.AxisScalingProbeError, "input NPZ differs"),
            ):
                RUNNER._load_input(Path(directory) / "manifest.json", FakeHelper())


if __name__ == "__main__":
    unittest.main()
