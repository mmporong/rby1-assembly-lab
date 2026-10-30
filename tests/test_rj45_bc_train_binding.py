from __future__ import annotations

import ast
from collections.abc import Callable
import copy
from pathlib import Path
import random
import tempfile
import types
from typing import Any, cast
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from rby1_assembly.held_grasp import bank_source_contract_sha256
from rby1_assembly.rj45_bc import (
    BcContractError,
    BcProvenance,
    build_bc_actor,
    file_sha256,
    load_bc_actor_artifact,
    save_bc_actor_artifact,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = PROJECT_ROOT / "scripts" / "train.py"
TrainBinding = Callable[..., dict[str, Any]]


def _load_train_binding() -> TrainBinding:
    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"), filename=str(TRAIN_SCRIPT))
    function = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_initialize_verified_bc_actor"
    )
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace: dict[str, Any] = {
        "bank_source_contract_sha256": bank_source_contract_sha256,
    }
    exec(compile(module, str(TRAIN_SCRIPT), "exec"), namespace)
    return cast(TrainBinding, namespace["_initialize_verified_bc_actor"])


def _load_initialization_fingerprint_helpers():
    tree = ast.parse(TRAIN_SCRIPT.read_text(encoding="utf-8"), filename=str(TRAIN_SCRIPT))
    selected = cast(list[ast.stmt], [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_initialization_state_fingerprint",
            "_assert_actor_only_initialization_change",
        }
    ])
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(TRAIN_SCRIPT), "exec"),
        namespace,
    )
    return (
        namespace["_initialization_state_fingerprint"],
        namespace["_assert_actor_only_initialization_change"],
    )


class _DummyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.actor = build_bc_actor()
        self.critic = nn.Sequential(nn.Linear(79, 32), nn.ELU(), nn.Linear(32, 1))
        self.std = nn.Parameter(torch.full((6,), 0.1))


class _DummyRunner:
    def __init__(self) -> None:
        policy = _DummyPolicy()
        optimizer = torch.optim.Adam(
            [
                *policy.actor.parameters(),
                *policy.critic.parameters(),
                policy.std,
            ],
            lr=1e-3,
        )
        self.alg = types.SimpleNamespace(policy=policy, optimizer=optimizer)


class InitialStateFingerprintTests(unittest.TestCase):
    def test_fingerprint_does_not_advance_cpu_rng_streams(self) -> None:
        fingerprint, _assert_change = _load_initialization_fingerprint_helpers()
        random.seed(81)
        np.random.seed(82)
        torch.manual_seed(83)
        runner = _DummyRunner()
        python_before = random.getstate()
        numpy_before = cast(tuple[Any, ...], np.random.get_state())
        torch_before = torch.random.get_rng_state().clone()

        first = fingerprint(runner, "cpu")
        second = fingerprint(runner, "cpu")

        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_before)
        numpy_after = cast(tuple[Any, ...], np.random.get_state())
        self.assertEqual(numpy_after[0], numpy_before[0])
        np.testing.assert_array_equal(numpy_after[1], numpy_before[1])
        self.assertEqual(numpy_after[2:], numpy_before[2:])
        torch.testing.assert_close(torch.random.get_rng_state(), torch_before)
        self.assertIsNone(first["rng"]["torch_training_cuda_device"])

    def test_actor_only_change_preserves_other_state_and_rng(self) -> None:
        fingerprint, assert_change = _load_initialization_fingerprint_helpers()
        runner = _DummyRunner()
        before = fingerprint(runner, "cpu")
        with torch.no_grad():
            runner.alg.policy.actor[-1].weight.zero_()
            runner.alg.policy.actor[-1].bias.zero_()
        after = fingerprint(runner, "cpu")

        assert_change(before, after)
        self.assertNotEqual(before["state"]["actor"], after["state"]["actor"])
        for name in ("critic", "noise", "optimizer"):
            self.assertEqual(before["state"][name], after["state"][name])
        self.assertEqual(before["rng"], after["rng"])

    def test_fingerprint_reads_the_selected_cuda_device_rng(self) -> None:
        fingerprint, _assert_change = _load_initialization_fingerprint_helpers()
        runner = _DummyRunner()
        cuda_state = torch.arange(16, dtype=torch.uint8)
        with mock.patch.object(
            torch.cuda, "get_rng_state", return_value=cuda_state
        ) as get_rng_state:
            result = fingerprint(runner, "cuda:2")

        get_rng_state.assert_called_once_with(torch.device("cuda:2"))
        self.assertIsInstance(
            result["rng"]["torch_training_cuda_device"], str
        )

    def test_same_seed_runners_have_equal_initial_fingerprints(self) -> None:
        fingerprint, _assert_change = _load_initialization_fingerprint_helpers()

        def initialized(seed: int):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            runner = _DummyRunner()
            return fingerprint(runner, "cpu")

        self.assertEqual(initialized(91), initialized(91))


class _FakeSnapshot:
    def __init__(self, hashes: dict[Path, str]) -> None:
        self._hashes = {path.resolve(): digest for path, digest in hashes.items()}
        self.hash_calls: list[Path] = []
        self.assert_unchanged_calls = 0

    def hash_for(self, path: str | Path) -> str:
        resolved = Path(path).resolve()
        self.hash_calls.append(resolved)
        return self._hashes[resolved]

    def assert_unchanged(self) -> None:
        self.assert_unchanged_calls += 1


def _bank_proof() -> dict[str, Any]:
    return {
        "manifest_sha256": "d" * 64,
        "provenance_contract": {
            "source_and_config_sha256": {
                "configs/runtime.json": "1" * 64,
                "configs/tasks/rj45-v1.json": "2" * 64,
                "src/rby1_assembly/rj45_training_env.py": "3" * 64,
            }
        },
    }


def _provenance(bank_proof: dict[str, Any], **changes: str) -> BcProvenance:
    values = {
        "source_sha256": bank_source_contract_sha256(bank_proof),
        "task_sha256": "a" * 64,
        "training_sha256": "b" * 64,
        "bank_sha256": bank_proof["manifest_sha256"],
        "dataset_sha256": "c" * 64,
    }
    values.update(changes)
    return BcProvenance(**values)


def _artifact_actor(delta: float) -> nn.Sequential:
    torch.manual_seed(17)
    actor = build_bc_actor()
    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(delta)
    return actor


def _actor_state(runner: _DummyRunner) -> dict[str, torch.Tensor]:
    return copy.deepcopy(runner.alg.policy.actor.state_dict())


def _assert_actor_equal(
    testcase: unittest.TestCase,
    runner: _DummyRunner,
    expected: dict[str, torch.Tensor],
) -> None:
    actual = runner.alg.policy.actor.state_dict()
    testcase.assertEqual(set(actual), set(expected))
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor)


class Rj45BcTrainBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.binding = _load_train_binding()
        self.bank_proof = _bank_proof()
        self.training_config = {"_config_sha256": "b" * 64}

    def test_valid_artifact_binds_actor_lineage_and_frozen_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "bc_actor_init.pt"
            task_path = root / "task.json"
            task_path.write_text("{}", encoding="utf-8")
            provenance = _provenance(self.bank_proof)
            save_bc_actor_artifact(artifact_path, _artifact_actor(0.25), provenance)
            artifact_sha = file_sha256(artifact_path)
            snapshot = _FakeSnapshot(
                {artifact_path: artifact_sha, task_path: provenance.task_sha256}
            )
            runner = _DummyRunner()

            result = self.binding(
                runner,
                artifact_path,
                artifact_sha,
                snapshot,
                task_path,
                self.training_config,
                self.bank_proof,
            )

            self.assertEqual(result["mode"], "bc_actor_warm_start")
            self.assertIs(type(result["output_dimension"]), int)
            self.assertEqual(result["output_dimension"], 6)
            self.assertEqual(result["artifact_file_sha256"], artifact_sha)
            self.assertEqual(result["bc_artifact_sha256"], artifact_sha)
            self.assertEqual(result["bc_provenance"], provenance.as_dict())
            self.assertEqual(result["loaded_component"], "runner.alg.policy.actor")
            self.assertEqual(
                result["hashes_after"]["actor"],
                result["artifact_actor_state_sha256"],
            )
            self.assertEqual(snapshot.assert_unchanged_calls, 1)
            self.assertEqual(
                snapshot.hash_calls,
                [artifact_path.resolve(), task_path.resolve()],
            )

    def test_cli_sha_mismatch_is_rejected_before_artifact_preload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "bc_actor_init.pt"
            task_path = root / "task.json"
            provenance = _provenance(self.bank_proof)
            save_bc_actor_artifact(artifact_path, _artifact_actor(0.1), provenance)
            snapshot = _FakeSnapshot(
                {artifact_path: file_sha256(artifact_path), task_path: provenance.task_sha256}
            )
            runner = _DummyRunner()
            before = _actor_state(runner)

            with (
                mock.patch("rby1_assembly.rj45_bc.load_bc_actor_artifact") as preload,
                mock.patch("rby1_assembly.rj45_bc.initialize_runner_actor") as initialize,
                self.assertRaisesRegex(ValueError, "frozen input"),
            ):
                self.binding(
                    runner,
                    artifact_path,
                    "f" * 64,
                    snapshot,
                    task_path,
                    self.training_config,
                    self.bank_proof,
                )

            preload.assert_not_called()
            initialize.assert_not_called()
            self.assertEqual(snapshot.assert_unchanged_calls, 0)
            _assert_actor_equal(self, runner, before)

    def test_artifact_replacement_after_preload_is_rejected_before_actor_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "bc_actor_init.pt"
            replacement_path = root / "replacement.pt"
            task_path = root / "task.json"
            provenance = _provenance(self.bank_proof)
            save_bc_actor_artifact(artifact_path, _artifact_actor(0.1), provenance)
            save_bc_actor_artifact(replacement_path, _artifact_actor(0.9), provenance)
            artifact_sha = file_sha256(artifact_path)
            self.assertNotEqual(artifact_sha, file_sha256(replacement_path))
            snapshot = _FakeSnapshot(
                {artifact_path: artifact_sha, task_path: provenance.task_sha256}
            )
            runner = _DummyRunner()
            before = _actor_state(runner)
            real_loader = load_bc_actor_artifact
            preload_count = 0

            def replace_after_first_load(
                path: str | Path, *, expected_sha256: str | None = None
            ) -> dict[str, Any]:
                nonlocal preload_count
                loaded = real_loader(path, expected_sha256=expected_sha256)
                preload_count += 1
                if preload_count == 1:
                    Path(path).write_bytes(replacement_path.read_bytes())
                return loaded

            with (
                mock.patch(
                    "rby1_assembly.rj45_bc.load_bc_actor_artifact",
                    side_effect=replace_after_first_load,
                ) as preload,
                self.assertRaisesRegex(BcContractError, "expected SHA-256"),
            ):
                self.binding(
                    runner,
                    artifact_path,
                    artifact_sha,
                    snapshot,
                    task_path,
                    self.training_config,
                    self.bank_proof,
                )

            self.assertEqual(preload.call_count, 2)
            self.assertEqual(preload_count, 1)
            self.assertEqual(snapshot.assert_unchanged_calls, 0)
            _assert_actor_equal(self, runner, before)

    def test_lineage_mismatch_is_rejected_before_actor_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact_path = root / "bc_actor_init.pt"
            task_path = root / "task.json"
            provenance = _provenance(self.bank_proof, task_sha256="9" * 64)
            save_bc_actor_artifact(artifact_path, _artifact_actor(0.4), provenance)
            artifact_sha = file_sha256(artifact_path)
            snapshot = _FakeSnapshot(
                {artifact_path: artifact_sha, task_path: "a" * 64}
            )
            runner = _DummyRunner()
            before = _actor_state(runner)

            with (
                mock.patch("rby1_assembly.rj45_bc.initialize_runner_actor") as initialize,
                self.assertRaisesRegex(ValueError, "lineage differs"),
            ):
                self.binding(
                    runner,
                    artifact_path,
                    artifact_sha,
                    snapshot,
                    task_path,
                    self.training_config,
                    self.bank_proof,
                )

            initialize.assert_not_called()
            self.assertEqual(snapshot.assert_unchanged_calls, 0)
            _assert_actor_equal(self, runner, before)


if __name__ == "__main__":
    unittest.main()
