from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from rby1_assembly.held_grasp import _held_grasp_bank_manifest_selection
from rby1_assembly.rj45_bank_generation import (
    audit_grasp_generation_pilot,
    build_manifest,
    descriptor_distance,
    held_grasp_descriptor,
)
from rby1_assembly.rj45_grasp_generation import (
    load_grasp_generation_catalog,
    select_grasp_generation_entry,
)


THRESHOLDS = {
    "object_to_ee_translation_m": 0.00025,
    "object_to_ee_rotation_rad": 0.004,
    "joint_position_linf_si": 0.005,
}
MINIMUMS = {"train": 1, "dev": 1, "final": 1}
CATALOG_PATH = Path(__file__).resolve().parents[1] / "configs/rj45-grasp-generation-v2.json"
CATALOG_SHA256 = hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest()


def artifact(offset: float = 0.0, quaternion_sign: float = 1.0, source: str = "a") -> dict:
    return {
        "provenance": {
            "task_config_sha256": "1" * 64,
            "runtime_config_sha256": "2" * 64,
            "source_and_config_sha256": {"src/example.py": source * 64},
        },
        "joint_names": ["j0", "j1"],
        "joint_units": ["rad", "m"],
        "state": {
            "joint_positions_si": [offset, 0.0],
            "object_poses_world_wxyz": [
                [0.0, 0.0, 0.0, quaternion_sign, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, quaternion_sign, 0.0, 0.0, 0.0],
            ],
            "end_effector_poses_world_wxyz": [
                [0.1 + offset, 0.0, 0.0, quaternion_sign, 0.0, 0.0, 0.0],
                [1.1 + offset, 0.0, 0.0, quaternion_sign, 0.0, 0.0, 0.0],
            ],
        },
    }


def candidate(grasp_id: str, split: str, seed: int, value: dict, digest: str) -> dict:
    return {
        "grasp_id": grasp_id,
        "split": split,
        "generation_seed": seed,
        "artifact_path": f"artifacts/held/{grasp_id}.json",
        "artifact": value,
        "artifact_sha256": digest,
    }


def generated_artifact(
    seed: int, offset: float = 0.0, *, catalog_sha: str = CATALOG_SHA256
) -> dict:
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    group, entry = select_grasp_generation_entry(catalog, seed)
    value = artifact(offset)
    value["provenance"]["source_and_config_sha256"][
        "configs/rj45-grasp-generation-v2.json"
    ] = catalog_sha
    value["provenance"]["grasp_generation"] = {
        "catalog_id": catalog["catalog_id"],
        "catalog_path": "configs/rj45-grasp-generation-v2.json",
        "catalog_sha256": catalog_sha,
        "generation_seed": seed,
        "catalog_entry_id": entry["id"],
        "assignment_group": group,
        "offset_method": catalog["offset_method"],
    }
    return value


class Rj45BankGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = mock.patch(
            "rby1_assembly.rj45_bank_generation.validate_held_grasp_artifact",
            side_effect=lambda value: value,
        )
        self.validation.start()

    def tearDown(self) -> None:
        self.validation.stop()

    def test_quaternion_sign_describes_the_same_physical_grasp(self) -> None:
        positive = held_grasp_descriptor(artifact(quaternion_sign=1.0))
        negative = held_grasp_descriptor(artifact(quaternion_sign=-1.0))
        self.assertEqual(positive, negative)
        self.assertEqual(
            descriptor_distance(positive, negative),
            {
                "object_to_ee_translation_m": 0.0,
                "object_to_ee_rotation_rad": 0.0,
                "joint_position_linf_si": 0.0,
            },
        )

    def test_clone_is_rejected_but_physically_distinct_grasp_is_accepted(self) -> None:
        clone = [
            candidate("train-a", "train", 10, artifact(), "a" * 64),
            candidate("dev-a", "dev", 11, artifact(quaternion_sign=-1), "b" * 64),
            candidate("final-a", "final", 12, artifact(0.02), "c" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "physical clone"):
            build_manifest(clone, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)
        distinct = [
            candidate("train-a", "train", 10, artifact(0.00), "a" * 64),
            candidate("dev-a", "dev", 11, artifact(0.01), "b" * 64),
            candidate("final-a", "final", 12, artifact(0.02), "c" * 64),
        ]
        manifest = build_manifest(
            distinct, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS
        )
        self.assertEqual([row["split"] for row in manifest["entries"]], ["train", "dev", "final"])

    def test_source_task_mismatch_and_split_seed_collision_are_rejected(self) -> None:
        values = [
            candidate("train-a", "train", 10, artifact(0.00), "a" * 64),
            candidate("dev-a", "dev", 11, artifact(0.01), "b" * 64),
            candidate("final-a", "final", 12, artifact(0.02), "c" * 64),
        ]
        mixed = copy.deepcopy(values)
        mixed[1]["artifact"]["provenance"]["source_and_config_sha256"] = {"src/example.py": "9" * 64}
        with self.assertRaisesRegex(ValueError, "mixed source/task/runtime/assets"):
            build_manifest(mixed, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)
        collision = copy.deepcopy(values)
        collision[1]["generation_seed"] = 10
        with self.assertRaisesRegex(ValueError, "disjoint across splits"):
            build_manifest(collision, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)

    def test_mixed_generation_catalogs_and_wrong_candidate_seed_are_rejected(self) -> None:
        values = [
            candidate("train-a", "train", 20261211, generated_artifact(20261211, 0.00), "a" * 64),
            candidate("dev-a", "dev", 20261231, generated_artifact(20261231, 0.01), "b" * 64),
            candidate("final-a", "final", 20261241, generated_artifact(20261241, 0.02), "c" * 64),
        ]
        mixed = copy.deepcopy(values)
        mixed[1]["artifact"]["provenance"]["grasp_generation"]["catalog_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "mixed source/task/runtime/assets"):
            build_manifest(mixed, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)
        wrong_seed = copy.deepcopy(values)
        wrong_seed[1]["artifact"]["provenance"]["grasp_generation"]["generation_seed"] = 20261232
        with self.assertRaisesRegex(ValueError, "generation seed differs"):
            build_manifest(wrong_seed, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)

    def test_pilot_audit_preserves_failures_and_computes_maximum_clique(self) -> None:
        catalog = load_grasp_generation_catalog(CATALOG_PATH)
        attempts = []
        for index, seed in enumerate(range(20261301, 20261305)):
            entry = select_grasp_generation_entry(catalog, seed)[1]
            if seed == 20261304:
                attempts.append(
                    {
                        "seed": seed,
                        "catalog_entry_id": entry["id"],
                        "status": "failed",
                        "failure_reason": "physical pilot timeout",
                    }
                )
            else:
                attempts.append(
                    {
                        "seed": seed,
                        "catalog_entry_id": entry["id"],
                        "status": "success",
                        "artifact": generated_artifact(seed, index * 0.0003),
                    }
                )
        result = audit_grasp_generation_pilot(
            attempts,
            catalog,
            catalog_path="configs/rj45-grasp-generation-v2.json",
            catalog_sha256=CATALOG_SHA256,
        )
        self.assertTrue(result["all_four_attempts_preserved"])
        self.assertFalse(result["policy_metric_selection_used"])
        self.assertEqual(result["attempts"][-1]["status"], "failed")
        self.assertEqual(result["maximum_pairwise_distinct_count"], 3)
        self.assertTrue(result["gate_pass"])

    def test_pilot_audit_rejects_wrong_entry_and_star_graph_fails_gate(self) -> None:
        catalog = load_grasp_generation_catalog(CATALOG_PATH)
        offsets = (0.0, 0.0003, 0.00001, -0.00001)
        attempts = []
        for seed, offset in zip(range(20261301, 20261305), offsets, strict=True):
            entry = select_grasp_generation_entry(catalog, seed)[1]
            attempts.append(
                {
                    "seed": seed,
                    "catalog_entry_id": entry["id"],
                    "status": "success",
                    "artifact": generated_artifact(seed, offset),
                }
            )
        result = audit_grasp_generation_pilot(
            attempts,
            catalog,
            catalog_path="configs/rj45-grasp-generation-v2.json",
            catalog_sha256=CATALOG_SHA256,
        )
        self.assertEqual(result["maximum_pairwise_distinct_count"], 2)
        self.assertFalse(result["gate_pass"])
        wrong = copy.deepcopy(attempts)
        wrong[0]["catalog_entry_id"] = "grasp-11"
        with self.assertRaisesRegex(ValueError, "frozen seed/catalog order"):
            audit_grasp_generation_pilot(
                wrong,
                catalog,
                catalog_path="configs/rj45-grasp-generation-v2.json",
                catalog_sha256=CATALOG_SHA256,
            )
        for invalid_seed in (20261301.0, True):
            malformed = copy.deepcopy(attempts)
            malformed[0]["seed"] = invalid_seed
            with self.subTest(invalid_seed=invalid_seed), self.assertRaisesRegex(
                ValueError, "frozen seed/catalog order"
            ):
                audit_grasp_generation_pilot(
                    malformed,
                    catalog,
                    catalog_path="configs/rj45-grasp-generation-v2.json",
                    catalog_sha256=CATALOG_SHA256,
                )

    def test_joint_posture_alone_cannot_claim_a_different_grasp(self) -> None:
        changed_joints = artifact()
        changed_joints["state"]["joint_positions_si"][0] = 0.2
        values = [
            candidate("train-a", "train", 10, artifact(), "a" * 64),
            candidate("dev-a", "dev", 11, changed_joints, "b" * 64),
            candidate("final-a", "final", 12, artifact(0.02), "c" * 64),
        ]
        with self.assertRaisesRegex(ValueError, "joint posture alone"):
            build_manifest(values, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)

    def test_manifest_is_metadata_compatible_with_existing_loader(self) -> None:
        values = [
            candidate("train-a", "train", 10, artifact(0.00), "a" * 64),
            candidate("dev-a", "dev", 11, artifact(0.01), "b" * 64),
            candidate("final-a", "final", 12, artifact(0.02), "c" * 64),
        ]
        manifest = build_manifest(values, separation_thresholds=THRESHOLDS, minimum_split_counts=MINIMUMS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for row in manifest["entries"]:
                path = root / row["artifact"]["path"]
                path.parent.mkdir(parents=True, exist_ok=True)
                content = row["grasp_id"].encode()
                path.write_bytes(content)
                row["artifact"]["sha256"] = hashlib.sha256(content).hexdigest()
            path = root / "bank.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            _digest, selected, counts = _held_grasp_bank_manifest_selection(
                root, path, "dev", False
            )
        self.assertEqual(selected[0][0], "dev-a")
        self.assertEqual(counts, {"train": 1, "dev": 1, "final": 1})


if __name__ == "__main__":
    unittest.main()
