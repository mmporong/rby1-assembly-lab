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
    build_manifest,
    descriptor_distance,
    held_grasp_descriptor,
)


THRESHOLDS = {
    "object_to_ee_translation_m": 0.00025,
    "object_to_ee_rotation_rad": 0.004,
    "joint_position_linf_si": 0.005,
}
MINIMUMS = {"train": 1, "dev": 1, "final": 1}


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
