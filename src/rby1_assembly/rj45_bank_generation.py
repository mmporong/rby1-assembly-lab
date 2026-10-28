"""Build a held-grasp bank from independently verified nominal run artifacts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rby1_assembly.rj45_training import validate_held_grasp_artifact


SPLITS = ("train", "dev", "final")


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _quat_normalized_canonical(value: Any) -> np.ndarray:
    quaternion = np.asarray(value, dtype=np.float64)
    if quaternion.shape != (4,) or not np.isfinite(quaternion).all():
        raise ValueError("descriptor quaternion must be finite wxyz")
    norm = float(np.linalg.norm(quaternion))
    if abs(norm - 1.0) > 1.0e-4:
        raise ValueError("descriptor quaternion must have unit length")
    quaternion = quaternion / norm
    for component in quaternion:
        if abs(component) > 1.0e-12:
            if component < 0.0:
                quaternion = -quaternion
            break
    return quaternion


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    conjugate = quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
    pure = np.asarray([0.0, *vector])
    return _quat_multiply(_quat_multiply(conjugate, pure), quaternion)[1:]


def held_grasp_descriptor(artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Describe physical object-to-hand geometry and joint pose, independent of time."""

    artifact = validate_held_grasp_artifact(artifact)
    state = artifact["state"]
    object_poses = np.asarray(state["object_poses_world_wxyz"], dtype=np.float64)
    ee_poses = np.asarray(state["end_effector_poses_world_wxyz"], dtype=np.float64)
    transforms = []
    for object_pose, ee_pose in zip(object_poses, ee_poses, strict=True):
        object_quaternion = _quat_normalized_canonical(object_pose[3:])
        ee_quaternion = _quat_normalized_canonical(ee_pose[3:])
        conjugate = object_quaternion * np.asarray([1.0, -1.0, -1.0, -1.0])
        relative_quaternion = _quat_normalized_canonical(
            _quat_multiply(conjugate, ee_quaternion)
        )
        transforms.append(
            {
                "position_m": _rotate_inverse(
                    object_quaternion, ee_pose[:3] - object_pose[:3]
                ).tolist(),
                "quaternion_wxyz": relative_quaternion.tolist(),
            }
        )
    return {
        "schema_version": 1,
        "object_to_end_effector": transforms,
        "joint_names": list(artifact["joint_names"]),
        "joint_units": list(artifact["joint_units"]),
        "joint_positions_si": list(state["joint_positions_si"]),
    }


def descriptor_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    if left.get("joint_names") != right.get("joint_names") or left.get("joint_units") != right.get("joint_units"):
        raise ValueError("held-grasp descriptors use different joint contracts")
    left_transforms = left.get("object_to_end_effector")
    right_transforms = right.get("object_to_end_effector")
    if not isinstance(left_transforms, list) or not isinstance(right_transforms, list) or len(left_transforms) != 2 or len(right_transforms) != 2:
        raise ValueError("held-grasp descriptors require two object-to-EE transforms")
    translations = []
    rotations = []
    for lhs, rhs in zip(left_transforms, right_transforms, strict=True):
        translations.append(
            float(np.linalg.norm(np.asarray(lhs["position_m"]) - np.asarray(rhs["position_m"])))
        )
        lq = _quat_normalized_canonical(lhs["quaternion_wxyz"])
        rq = _quat_normalized_canonical(rhs["quaternion_wxyz"])
        rotations.append(2.0 * math.acos(min(1.0, abs(float(np.dot(lq, rq))))))
    joint_left = np.asarray(left["joint_positions_si"], dtype=np.float64)
    joint_right = np.asarray(right["joint_positions_si"], dtype=np.float64)
    if joint_left.shape != joint_right.shape or not np.isfinite(joint_left).all() or not np.isfinite(joint_right).all():
        raise ValueError("held-grasp descriptor joint poses differ in shape or finiteness")
    return {
        "object_to_ee_translation_m": max(translations),
        "object_to_ee_rotation_rad": max(rotations),
        "joint_position_linf_si": float(np.max(np.abs(joint_left - joint_right))),
    }


def _thresholds(value: Mapping[str, Any]) -> dict[str, float]:
    required = {
        "object_to_ee_translation_m",
        "object_to_ee_rotation_rad",
        "joint_position_linf_si",
    }
    if set(value) != required:
        raise ValueError(f"separation thresholds must contain exactly {sorted(required)}")
    result = {}
    for name in required:
        number = value[name]
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"separation threshold {name} must be finite and positive")
        result[name] = float(number)
    return result


def build_manifest(
    candidates: list[Mapping[str, Any]],
    *,
    separation_thresholds: Mapping[str, Any],
    minimum_split_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Validate physical diversity and return a loader-compatible frozen manifest."""

    thresholds = _thresholds(separation_thresholds)
    if set(minimum_split_counts) != set(SPLITS) or any(
        type(minimum_split_counts[name]) is not int or minimum_split_counts[name] < 1
        for name in SPLITS
    ):
        raise ValueError("minimum split counts must be positive train/dev/final integers")
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    seen_hashes: set[str] = set()
    descriptors: list[dict[str, Any]] = []
    entries = []
    split_counts = {name: 0 for name in SPLITS}
    common_contract = None
    previous_seed = -1
    for index, candidate in enumerate(candidates):
        if set(candidate) != {"grasp_id", "split", "generation_seed", "artifact_path", "artifact", "artifact_sha256"}:
            raise ValueError(f"candidate {index} schema differs")
        grasp_id, split, seed = candidate["grasp_id"], candidate["split"], candidate["generation_seed"]
        if not isinstance(grasp_id, str) or not grasp_id or grasp_id in seen_ids:
            raise ValueError("held-grasp IDs must be unique non-empty strings")
        if split not in SPLITS:
            raise ValueError("held-grasp split must be train, dev, or final")
        if type(seed) is not int or seed < 0 or seed in seen_seeds:
            raise ValueError("generation seeds must be non-negative and disjoint across splits")
        if seed <= previous_seed:
            raise ValueError("candidates must follow the frozen strictly increasing seed order")
        artifact = validate_held_grasp_artifact(candidate["artifact"])
        digest = candidate["artifact_sha256"]
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or digest in seen_hashes
        ):
            raise ValueError("artifact SHA-256 values must be valid and unique")
        artifact_path = candidate["artifact_path"]
        if (
            not isinstance(artifact_path, str)
            or not artifact_path
            or Path(artifact_path).is_absolute()
            or ".." in Path(artifact_path).parts
        ):
            raise ValueError("artifact paths must be project-relative without traversal")
        provenance = artifact["provenance"]
        contract = {
            "task_config_sha256": provenance["task_config_sha256"],
            "runtime_config_sha256": provenance["runtime_config_sha256"],
            "source_and_config_sha256": provenance["source_and_config_sha256"],
        }
        if common_contract is None:
            common_contract = contract
        elif contract != common_contract:
            raise ValueError("held-grasp candidates have mixed source/task/runtime/assets provenance")
        descriptor = held_grasp_descriptor(artifact)
        for previous in descriptors:
            distance = descriptor_distance(previous, descriptor)
            if not any(
                distance[name] >= thresholds[name]
                for name in ("object_to_ee_translation_m", "object_to_ee_rotation_rad")
            ):
                raise ValueError(
                    "held-grasp candidate is a physical clone: joint posture alone "
                    "does not satisfy object-to-hand grasp separation"
                )
        descriptor_digest = canonical_sha256(descriptor)
        entries.append(
            {
                "grasp_id": grasp_id,
                "split": split,
                "generation_seed": seed,
                "artifact": {"path": artifact_path, "sha256": digest},
                "descriptor": descriptor,
                "descriptor_sha256": descriptor_digest,
            }
        )
        descriptors.append(descriptor)
        seen_ids.add(grasp_id)
        seen_seeds.add(seed)
        previous_seed = seed
        seen_hashes.add(digest)
        split_counts[split] += 1
    for split in SPLITS:
        if split_counts[split] < minimum_split_counts[split]:
            raise ValueError(f"held-grasp bank has fewer than planned {split} candidates")
    return {"schema_version": 1, "task_id": "rj45-v1", "entries": entries}


def validate_nonzero_position_noise(task: Mapping[str, Any], expected: list[float]) -> None:
    actual = task.get("reset", {}).get("position_noise_m")
    if actual != expected or len(expected) != 3 or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        for value in expected
    ) or not any(abs(float(value)) > 0.0 for value in expected):
        raise ValueError("task reset.position_noise_m must equal the explicit nonzero generation contract")

