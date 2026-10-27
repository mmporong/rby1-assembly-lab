from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "inspect_rj45_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("inspect_rj45_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_bundle(root: Path, *, checkpoint_bytes: bytes = b"checkpoint") -> dict[str, Path]:
    checkpoint = root / "model_2.pt"
    manifest = root / "checkpoint_manifest.json"
    provenance = root / "training_provenance.json"
    runner = root / "runner_config.json"
    training_script = root / "train.py"
    checkpoint.write_bytes(checkpoint_bytes)
    training_script.write_text("# reviewed training entrypoint\n", encoding="utf-8")
    runner.write_text(json.dumps({"seed": 7, "max_iterations": 4}), encoding="utf-8")
    provenance.write_text(
        json.dumps({
            "schema_version": 1,
            "seed": 7,
            "checkpoint_scope": "checkpoints created in this run directory",
            "runtime_versions": {"rsl-rl-lib": "3.0.1"},
            "source_and_config_sha256": {
                "run/runner_config.json": _sha(runner),
                "scripts/train.py": _sha(training_script),
            },
        }),
        encoding="utf-8",
    )
    manifest.write_text(
        json.dumps({
            "schema_version": 1,
            "training_completed": True,
            "training_provenance_sha256": _sha(provenance),
            "checkpoints": {checkpoint.name: _sha(checkpoint)},
        }),
        encoding="utf-8",
    )
    return {
        "checkpoint": checkpoint,
        "manifest": manifest,
        "training_provenance": provenance,
        "runner_config": runner,
        "training_script": training_script,
    }


class InspectRj45CheckpointTests(unittest.TestCase):
    def test_mock_loader_verifies_iteration_and_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_bundle(Path(directory))
            fake_torch = types.SimpleNamespace(load=mock.Mock(return_value={"iter": 2}))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                result = MODULE.inspect_checkpoint(
                    checkpoint_path=paths["checkpoint"],
                    manifest_path=paths["manifest"],
                    training_provenance_path=paths["training_provenance"],
                    runner_config_path=paths["runner_config"],
                    training_script_path=paths["training_script"],
                )
            fake_torch.load.assert_called_once()
            loaded_stream = fake_torch.load.call_args.args[0]
            self.assertEqual(loaded_stream.getvalue(), b"checkpoint")
            self.assertEqual(fake_torch.load.call_args.kwargs, {
                "map_location": "cpu",
                "weights_only": True,
            })
            self.assertEqual(result["checkpoint"]["completed_updates"], 3)
            self.assertEqual(result["planned_training_updates"], 4)
            self.assertEqual(
                result["completion_derivation"]["training_mode_basis"],
                "assumed_fresh_no_resume_requires_review_of_pinned_training_script",
            )

    def test_checkpoint_hash_and_payload_use_the_same_captured_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_bundle(Path(directory), checkpoint_bytes=b"captured")

            def mutate_after_capture(stream, *, map_location, weights_only):
                self.assertEqual(stream.getvalue(), b"captured")
                self.assertEqual(map_location, "cpu")
                self.assertTrue(weights_only)
                paths["checkpoint"].write_bytes(b"changed after capture")
                return {"iter": 2}

            fake_torch = types.SimpleNamespace(load=mock.Mock(side_effect=mutate_after_capture))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                result = MODULE.inspect_checkpoint(
                    checkpoint_path=paths["checkpoint"],
                    manifest_path=paths["manifest"],
                    training_provenance_path=paths["training_provenance"],
                    runner_config_path=paths["runner_config"],
                    training_script_path=paths["training_script"],
                )
            self.assertEqual(
                result["checkpoint"]["sha256"], hashlib.sha256(b"captured").hexdigest()
            )

    def test_mismatch_and_incomplete_manifest_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_bundle(Path(directory))
            fake_torch = types.SimpleNamespace(load=mock.Mock(return_value={"iter": 1}))
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "differs from model_N"):
                    MODULE.inspect_checkpoint(
                        checkpoint_path=paths["checkpoint"],
                        manifest_path=paths["manifest"],
                        training_provenance_path=paths["training_provenance"],
                        runner_config_path=paths["runner_config"],
                        training_script_path=paths["training_script"],
                    )

    def test_completion_derivation_prerequisites_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _write_bundle(Path(directory))
            fake_torch = types.SimpleNamespace(load=mock.Mock(return_value={"iter": 2}))

            def write_provenance(payload: dict) -> None:
                paths["training_provenance"].write_text(json.dumps(payload), encoding="utf-8")
                manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
                manifest["training_provenance_sha256"] = _sha(paths["training_provenance"])
                paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")

            original = json.loads(paths["training_provenance"].read_text(encoding="utf-8"))
            missing_scope = json.loads(json.dumps(original))
            del missing_scope["checkpoint_scope"]
            write_provenance(missing_scope)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "fresh run-local checkpoint scope"):
                    MODULE.inspect_checkpoint(
                        checkpoint_path=paths["checkpoint"],
                        manifest_path=paths["manifest"],
                        training_provenance_path=paths["training_provenance"],
                        runner_config_path=paths["runner_config"],
                        training_script_path=paths["training_script"],
                    )

            wrong_runtime = json.loads(json.dumps(original))
            wrong_runtime["runtime_versions"]["rsl-rl-lib"] = "3.0.2"
            write_provenance(wrong_runtime)
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "requires rsl-rl-lib 3.0.1"):
                    MODULE.inspect_checkpoint(
                        checkpoint_path=paths["checkpoint"],
                        manifest_path=paths["manifest"],
                        training_provenance_path=paths["training_provenance"],
                        runner_config_path=paths["runner_config"],
                        training_script_path=paths["training_script"],
                    )

            write_provenance(original)
            paths["training_script"].write_text("# modified\n", encoding="utf-8")
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "training script bytes"):
                    MODULE.inspect_checkpoint(
                        checkpoint_path=paths["checkpoint"],
                        manifest_path=paths["manifest"],
                        training_provenance_path=paths["training_provenance"],
                        runner_config_path=paths["runner_config"],
                        training_script_path=paths["training_script"],
                    )
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            manifest["training_completed"] = False
            paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
            fake_torch.load.return_value = {"iter": 2}
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "does not mark training as completed"):
                    MODULE.inspect_checkpoint(
                        checkpoint_path=paths["checkpoint"],
                        manifest_path=paths["manifest"],
                        training_provenance_path=paths["training_provenance"],
                        runner_config_path=paths["runner_config"],
                        training_script_path=paths["training_script"],
                    )

    def test_main_refuses_to_overwrite_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _write_bundle(root)
            output = root / "inspection.json"
            output.write_text("preserve", encoding="utf-8")
            fake_torch = types.SimpleNamespace(load=mock.Mock(return_value={"iter": 2}))
            arguments = [
                "--checkpoint", str(paths["checkpoint"]),
                "--manifest", str(paths["manifest"]),
                "--training-provenance", str(paths["training_provenance"]),
                "--runner-config", str(paths["runner_config"]),
                "--training-script", str(paths["training_script"]),
                "--output", str(output),
            ]
            with mock.patch.dict(sys.modules, {"torch": fake_torch}):
                with self.assertRaisesRegex(ValueError, "will not be overwritten"):
                    MODULE.main(arguments)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")

if __name__ == "__main__":
    unittest.main()
