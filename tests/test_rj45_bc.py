from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch
from torch import nn

from rby1_assembly.rj45_bc import (
    ARTIFACT_KIND,
    BcContractError,
    BcProvenance,
    EXPECTED_ACTOR_KEYS,
    SCHEMA_NAME,
    TEACHER_METADATA_KEYS,
    build_bc_actor,
    file_sha256,
    initialize_runner_actor,
    load_bc_actor_artifact,
    prepare_train_samples,
    save_bc_actor_artifact,
    train_bc_batch,
)


HASHES = {
    "source_sha256": "a" * 64,
    "task_sha256": "b" * 64,
    "training_sha256": "c" * 64,
    "bank_sha256": "d" * 64,
    "dataset_sha256": "e" * 64,
}


def provenance(**changes: str) -> BcProvenance:
    return BcProvenance(**(HASHES | changes))


class DummyPolicy:
    def __init__(self) -> None:
        self.actor = build_bc_actor()
        self.critic = nn.Sequential(nn.Linear(79, 32), nn.ELU(), nn.Linear(32, 1))
        self.std = nn.Parameter(torch.full((6,), 0.1))


class DummyRunner:
    def __init__(self) -> None:
        self.alg = types.SimpleNamespace()
        self.alg.policy = DummyPolicy()
        self.alg.optimizer = torch.optim.Adam(
            list(self.alg.policy.actor.parameters())
            + list(self.alg.policy.critic.parameters())
            + [self.alg.policy.std],
            lr=1e-3,
        )


class Rj45BcTests(unittest.TestCase):
    def test_architecture_keys_and_one_batch_update(self) -> None:
        torch.manual_seed(1)
        actor = build_bc_actor()
        self.assertEqual(tuple(actor.state_dict()), EXPECTED_ACTOR_KEYS)
        before = {name: value.clone() for name, value in actor.state_dict().items()}
        observation = torch.randn(16, 79)
        action = torch.tanh(torch.randn(16, 6))
        result = train_bc_batch(actor, observation, action, torch.optim.Adam(actor.parameters(), lr=1e-3))
        self.assertTrue(result["loss_before"] > 0.0)
        self.assertLess(result["loss_after"], result["loss_before"])
        self.assertTrue(any(not torch.equal(before[name], actor.state_dict()[name]) for name in before))

    def test_batch_rejects_dimension_nonfinite_and_non_cpu_contract(self) -> None:
        actor = build_bc_actor()
        optimizer = torch.optim.Adam(actor.parameters())
        with self.assertRaisesRegex(BcContractError, "79"):
            train_bc_batch(actor, torch.zeros(2, 78), torch.zeros(2, 6), optimizer)
        bad = torch.zeros(2, 79)
        bad[0, 0] = torch.nan
        with self.assertRaisesRegex(BcContractError, "finite"):
            train_bc_batch(actor, bad, torch.zeros(2, 6), optimizer)
        with self.assertRaisesRegex(BcContractError, "N, 6"):
            train_bc_batch(actor, torch.zeros(2, 79), torch.zeros(2, 7), optimizer)

    def test_prepare_samples_is_train_only_and_checks_metadata_allowlist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "teacher.h5"
            dataset.write_bytes(b"fixture")
            digest = file_sha256(dataset)
            metadata = {
                "grasp_id": "grasp-0",
                "bias_case_id": "bias-0",
                "split": "train",
                "seed": 7,
                "source_sha256": "a" * 64,
                "task_sha256": "b" * 64,
                "training_sha256": "c" * 64,
                "bank_sha256": "d" * 64,
            }
            calls = []

            def loader(_path, **kwargs):
                calls.append(kwargs)
                return [{
                    "episode_id": "episode-0",
                    "observation": torch.zeros(2, 79).numpy(),
                    "action": torch.zeros(2, 6).numpy(),
                    "metadata": metadata,
                }]

            fake = types.ModuleType("rby1_assembly.rj45_teacher_dataset")
            setattr(fake, "load_student_episodes", loader)
            previous = sys.modules.get(fake.__name__)
            sys.modules[fake.__name__] = fake
            try:
                prepared = prepare_train_samples(dataset, digest)
                self.assertEqual(calls, [{"split": "train", "successful_only": True, "valid_only": True}])
                self.assertEqual(prepared.observation.shape, (2, 79))
                self.assertEqual(prepared.action.shape, (2, 6))
                self.assertEqual(set(metadata), TEACHER_METADATA_KEYS)
                metadata["split"] = "dev"
                with self.assertRaisesRegex(BcContractError, "train split"):
                    prepare_train_samples(dataset, digest)
                metadata["split"] = "train"
                metadata["unexpected"] = "ground_truth"
                with self.assertRaisesRegex(BcContractError, "allowlist"):
                    prepare_train_samples(dataset, digest)
            finally:
                if previous is None:
                    del sys.modules[fake.__name__]
                else:
                    sys.modules[fake.__name__] = previous

    def test_artifact_round_trip_file_and_actor_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bc_actor_init.pt"
            saved = save_bc_actor_artifact(path, build_bc_actor(), provenance())
            loaded = load_bc_actor_artifact(path)
            self.assertEqual(saved["sha256"], file_sha256(path))
            self.assertEqual(saved["actor_state_sha256"], loaded["actor_state_sha256"])
            self.assertEqual(loaded["schema_name"], SCHEMA_NAME)
            self.assertEqual(loaded["artifact_kind"], ARTIFACT_KIND)
            self.assertNotIn("optimizer_state_dict", loaded)
            self.assertNotIn("model_state_dict", loaded)

    def test_batch_rejects_optimizer_with_missing_or_extra_actor_parameters(self) -> None:
        actor = build_bc_actor()
        observation = torch.zeros(2, 79)
        action = torch.zeros(2, 6)
        other_actor = build_bc_actor()
        cases = (
            torch.optim.Adam(other_actor.parameters()),
            torch.optim.Adam(list(actor.parameters())[:-1]),
            torch.optim.Adam([*actor.parameters(), nn.Parameter(torch.zeros(1))]),
        )
        for optimizer in cases:
            with self.subTest(optimizer=optimizer), self.assertRaisesRegex(
                BcContractError, "exactly match"
            ):
                train_bc_batch(actor, observation, action, optimizer)

    def test_artifact_tamper_and_schema_are_rejected(self) -> None:
        for mutation, message in (
            (lambda payload: payload["actor_state_dict"]["0.weight"].add_(1.0), "SHA-256"),
            (lambda payload: payload.__setitem__("schema_name", "ppo_checkpoint"), "schema"),
            (lambda payload: payload.__setitem__("optimizer_state_dict", {}), "top-level"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "artifact.pt"
                save_bc_actor_artifact(path, build_bc_actor(), provenance())
                payload = torch.load(path, map_location="cpu", weights_only=True)
                mutation(payload)
                torch.save(payload, path)
                with self.assertRaisesRegex(BcContractError, message):
                    load_bc_actor_artifact(path)

    def test_actor_only_initialization_preserves_critic_std_optimizer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.pt"
            trained = build_bc_actor()
            train_bc_batch(
                trained,
                torch.randn(8, 79),
                torch.randn(8, 6),
                torch.optim.Adam(trained.parameters(), lr=1e-3),
            )
            save_bc_actor_artifact(path, trained, provenance())
            runner = DummyRunner()
            runner.alg.optimizer.zero_grad(set_to_none=True)
            state_seed_terms = [
                parameter.square().mean()
                for parameter in (
                    list(runner.alg.policy.actor.parameters())
                    + list(runner.alg.policy.critic.parameters())
                    + [runner.alg.policy.std]
                )
            ]
            state_seed_loss = torch.stack(state_seed_terms).sum()
            state_seed_loss.backward()
            runner.alg.optimizer.step()
            self.assertTrue(runner.alg.optimizer.state)
            critic = copy.deepcopy(runner.alg.policy.critic.state_dict())
            std = runner.alg.policy.std.detach().clone()
            optimizer = copy.deepcopy(runner.alg.optimizer.state_dict())
            result = initialize_runner_actor(runner, path)
            self.assertNotEqual(result["hashes_before"]["actor"], result["hashes_after"]["actor"])
            self.assertEqual(result["hashes_before"]["critic"], result["hashes_after"]["critic"])
            self.assertEqual(result["hashes_before"]["noise"], result["hashes_after"]["noise"])
            self.assertEqual(result["hashes_before"]["optimizer"], result["hashes_after"]["optimizer"])
            self.assertEqual(
                result["hashes_after"]["actor"], result["artifact_actor_state_sha256"]
            )
            self.assertTrue(result["actor_changed"])
            for name, value in critic.items():
                torch.testing.assert_close(runner.alg.policy.critic.state_dict()[name], value)
            torch.testing.assert_close(runner.alg.policy.std, std)
            self.assertEqual(runner.alg.optimizer.state_dict()["param_groups"], optimizer["param_groups"])

    def test_expected_artifact_sha_rejects_replacement_before_actor_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.pt"
            replacement = Path(directory) / "replacement.pt"
            save_bc_actor_artifact(first, build_bc_actor(), provenance())
            expected = file_sha256(first)
            replacement_actor = build_bc_actor()
            train_bc_batch(
                replacement_actor,
                torch.randn(4, 79),
                torch.randn(4, 6),
                torch.optim.Adam(replacement_actor.parameters()),
            )
            save_bc_actor_artifact(replacement, replacement_actor, provenance())
            first.write_bytes(replacement.read_bytes())
            runner = DummyRunner()
            before = {name: value.clone() for name, value in runner.alg.policy.actor.state_dict().items()}
            with self.assertRaisesRegex(BcContractError, "expected SHA-256"):
                initialize_runner_actor(runner, first, expected_sha256=expected)
            for name, value in before.items():
                torch.testing.assert_close(runner.alg.policy.actor.state_dict()[name], value)


if __name__ == "__main__":
    unittest.main()
