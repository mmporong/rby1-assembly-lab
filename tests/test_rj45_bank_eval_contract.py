from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/evaluate_policy.py"


def compiled_helper(name: str):
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            SCRIPT,
            "exec",
        ),
        namespace,
    )
    return namespace[name]


def proof() -> dict[str, object]:
    return {
        "selected_grasp_ids": ["dev-a", "dev-b"],
        "manifest_sha256": "a" * 64,
        "selected_split": "dev",
    }


class Rj45BankEvaluationContractTests(unittest.TestCase):
    def test_terminal_index_maps_to_verified_ordered_grasp_identity(self) -> None:
        identity = compiled_helper("_rj45_bank_terminal_identity")
        snapshot = {"held_grasp_index": torch.tensor([1, 0])}
        self.assertEqual(
            identity(snapshot, 0, proof()),
            {
                "grasp_index": 1,
                "grasp_id": "dev-b",
                "bank_manifest_sha256": "a" * 64,
                "bank_split": "dev",
            },
        )
        snapshot["held_grasp_index"][0] = 2
        with self.assertRaisesRegex(ValueError, "outside the verified bank"):
            identity(snapshot, 0, proof())

    def test_eval_source_keeps_train_and_selected_bank_proofs_separate(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('args.rj45_bank_split,', source)
        self.assertIn('args.rj45_held_bank, "train"', source)
        self.assertIn('"held_grasp_reset_proof": (\n                    training_bank_proof', source)
        self.assertIn('"evaluation_held_grasp_bank_proof"', source)
        self.assertIn('"training_held_grasp_bank_proof"', source)

    def test_bc_load_rejects_cli_snapshot_mismatch_before_read(self) -> None:
        load = compiled_helper("_load_verified_bc_evaluation_artifact")
        loader = mock.Mock()
        with self.assertRaisesRegex(ValueError, "CLI SHA-256 differs"):
            load(
                Path("actor.pt"),
                cli_expected_sha256="a" * 64,
                snapshot_sha256="b" * 64,
                loader=loader,
            )
        loader.assert_not_called()

    def test_bc_load_binds_the_atomic_loader_to_snapshot_sha(self) -> None:
        load = compiled_helper("_load_verified_bc_evaluation_artifact")
        digest = "a" * 64
        loader = mock.Mock(
            side_effect=ValueError("BC actor artifact differs from expected SHA-256")
        )
        with self.assertRaisesRegex(ValueError, "differs from expected"):
            load(
                Path("replaced.pt"),
                cli_expected_sha256=digest,
                snapshot_sha256=digest,
                loader=loader,
            )
        loader.assert_called_once_with(Path("replaced.pt"), expected_sha256=digest)

    def test_bc_load_requires_returned_file_sha_to_equal_snapshot(self) -> None:
        load = compiled_helper("_load_verified_bc_evaluation_artifact")
        digest = "a" * 64
        with self.assertRaisesRegex(RuntimeError, "loader result differs"):
            load(
                Path("actor.pt"),
                cli_expected_sha256=digest,
                snapshot_sha256=digest,
                loader=mock.Mock(return_value={"file_sha256": "b" * 64}),
            )
        artifact = {"file_sha256": digest, "actor_state_dict": {}}
        self.assertIs(
            load(
                Path("actor.pt"),
                cli_expected_sha256=digest,
                snapshot_sha256=digest,
                loader=mock.Mock(return_value=artifact),
            ),
            artifact,
        )

    def test_cli_rejects_final_bank_split_and_mixed_single_bank_modes(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('choices=("train", "dev")', source)
        self.assertIn("--rj45-held-state and --rj45-held-bank are mutually exclusive", source)
        self.assertIn("BC-only evaluation is RJ45-only and cannot use a PPO checkpoint", source)


if __name__ == "__main__":
    unittest.main()
