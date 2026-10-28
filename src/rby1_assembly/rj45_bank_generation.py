"""Build a held-grasp bank from independently verified nominal run artifacts."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rby1_assembly.rj45_training import validate_held_grasp_artifact


SPLITS = ("train", "dev", "final")
PILOT_SEEDS = (20261301, 20261302, 20261303, 20261304)
PILOT_TRANSLATION_THRESHOLD_M = 0.00025
PILOT_ROTATION_THRESHOLD_RAD = 0.004


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


def audit_grasp_generation_pilot(
    attempts: list[Mapping[str, Any]],
    catalog: Mapping[str, Any],
    *,
    catalog_path: str,
    catalog_sha256: str,
) -> dict[str, Any]:
    """Audit all four predeclared physical pilot attempts without selecting outcomes."""

    from rby1_assembly.rj45_grasp_generation import (
        select_grasp_generation_entry,
        validate_grasp_generation_catalog,
    )

    catalog = validate_grasp_generation_catalog(catalog)
    if (
        not isinstance(catalog_path, str)
        or not catalog_path
        or Path(catalog_path).is_absolute()
        or ".." in Path(catalog_path).parts
        or not isinstance(catalog_sha256, str)
        or len(catalog_sha256) != 64
        or any(character not in "0123456789abcdef" for character in catalog_sha256)
    ):
        raise ValueError("pilot audit requires a project-relative catalog path and SHA-256")
    if not isinstance(attempts, list) or len(attempts) != len(PILOT_SEEDS):
        raise ValueError("pilot audit requires exactly four preserved attempts")
    preserved: list[dict[str, Any]] = []
    descriptors: dict[int, dict[str, Any]] = {}
    for index, (attempt, expected_seed) in enumerate(
        zip(attempts, PILOT_SEEDS, strict=True)
    ):
        if not isinstance(attempt, Mapping):
            raise ValueError(f"pilot attempt {index} must be an object")
        seed = attempt.get("seed")
        status = attempt.get("status")
        assignment_group, expected_entry = select_grasp_generation_entry(
            catalog, expected_seed
        )
        if (
            type(seed) is not int
            or seed != expected_seed
            or attempt.get("catalog_entry_id") != expected_entry["id"]
        ):
            raise ValueError("pilot attempts differ from the frozen seed/catalog order")
        if status == "failed":
            if set(attempt) != {
                "seed",
                "catalog_entry_id",
                "status",
                "failure_reason",
            }:
                raise ValueError("failed pilot attempt schema differs")
            reason = attempt["failure_reason"]
            if not isinstance(reason, str) or not reason:
                raise ValueError("failed pilot attempt requires a failure reason")
            preserved.append(
                {
                    "seed": seed,
                    "catalog_entry_id": expected_entry["id"],
                    "assignment_group": assignment_group,
                    "status": status,
                    "failure_reason": reason,
                }
            )
            continue
        if status != "success" or set(attempt) != {
            "seed",
            "catalog_entry_id",
            "status",
            "artifact",
        }:
            raise ValueError("successful pilot attempt schema differs")
        artifact = validate_held_grasp_artifact(attempt["artifact"])
        generation = artifact["provenance"].get("grasp_generation")
        if not isinstance(generation, dict) or any(
            generation.get(name) != expected
            for name, expected in {
                "catalog_id": catalog["catalog_id"],
                "catalog_path": catalog_path,
                "catalog_sha256": catalog_sha256,
                "generation_seed": seed,
                "catalog_entry_id": expected_entry["id"],
                "assignment_group": assignment_group,
                "offset_method": catalog["offset_method"],
            }.items()
        ):
            raise ValueError("pilot artifact generation provenance differs from its seed mapping")
        descriptor = held_grasp_descriptor(artifact)
        descriptors[seed] = descriptor
        preserved.append(
            {
                "seed": seed,
                "catalog_entry_id": expected_entry["id"],
                "assignment_group": assignment_group,
                "status": status,
                "descriptor_sha256": canonical_sha256(descriptor),
            }
        )

    edges: list[dict[str, Any]] = []
    separated_pairs: set[tuple[int, int]] = set()
    successful_seeds = sorted(descriptors)
    for left_seed, right_seed in itertools.combinations(successful_seeds, 2):
        distance = descriptor_distance(descriptors[left_seed], descriptors[right_seed])
        separated = (
            distance["object_to_ee_translation_m"]
            >= PILOT_TRANSLATION_THRESHOLD_M
            or distance["object_to_ee_rotation_rad"]
            >= PILOT_ROTATION_THRESHOLD_RAD
        )
        if separated:
            separated_pairs.add((left_seed, right_seed))
        edges.append(
            {
                "left_seed": left_seed,
                "right_seed": right_seed,
                **distance,
                "separated": separated,
            }
        )
    cliques: list[tuple[int, ...]] = []
    for size in range(1, len(successful_seeds) + 1):
        for seeds in itertools.combinations(successful_seeds, size):
            if all(pair in separated_pairs for pair in itertools.combinations(seeds, 2)):
                cliques.append(seeds)
    maximum = max((len(clique) for clique in cliques), default=0)
    representatives = [list(clique) for clique in cliques if len(clique) == maximum]
    return {
        "schema_version": 1,
        "catalog_id": catalog["catalog_id"],
        "catalog_path": catalog_path,
        "catalog_sha256": catalog_sha256,
        "attempts": preserved,
        "all_four_attempts_preserved": len(preserved) == 4,
        "policy_metric_selection_used": False,
        "translation_threshold_m": PILOT_TRANSLATION_THRESHOLD_M,
        "rotation_threshold_rad": PILOT_ROTATION_THRESHOLD_RAD,
        "separation_rule": "translation_gte_threshold_or_rotation_gte_threshold",
        "pairwise_edges": edges,
        "maximum_pairwise_distinct_count": maximum,
        "representative_sets": representatives,
        "gate_required_count": 3,
        "gate_pass": maximum >= 3,
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
        generation = provenance.get("grasp_generation")
        generation_contract = None
        if generation is not None:
            if generation["generation_seed"] != seed:
                raise ValueError(
                    "held-grasp generation seed differs from the frozen candidate seed"
                )
            generation_contract = {
                name: generation[name]
                for name in (
                    "catalog_id",
                    "catalog_path",
                    "catalog_sha256",
                    "offset_method",
                )
            }
        contract = {
            "task_config_sha256": provenance["task_config_sha256"],
            "runtime_config_sha256": provenance["runtime_config_sha256"],
            "source_and_config_sha256": provenance["source_and_config_sha256"],
            "grasp_generation_catalog": generation_contract,
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

