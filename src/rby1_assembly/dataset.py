"""RB-Y1 조립 시뮬레이션 episode의 검증 가능한 HDF5 기록 계약."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import h5py
import numpy as np

from .contracts import ACTION_DIM, OBSERVATION_DIM


SCHEMA_NAME = "rby1_assembly_episode"
SCHEMA_VERSION = 2
GROUND_TRUTH_FIELDS = (
    "actual_depth_m",
    "lateral_error_m",
    "orientation_error_rad",
    "force_peak_n",
    "grasp_both",
    "forbidden_contact",
)
REVOLUTE_JOINT_ORDER = (
    *(f"torso_{index}" for index in range(6)),
    *(f"left_arm_{index}" for index in range(7)),
    *(f"right_arm_{index}" for index in range(7)),
)
GRIPPER_JOINT_ORDER = (
    "gripper_finger_l1",
    "gripper_finger_l2",
    "gripper_finger_r1",
    "gripper_finger_r2",
)
_EPISODE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
INSERTION_LOAD_SOURCE = "physx_plug_socket_sum_abs_contact_normals"
INSERTION_FORCE_SOURCE = "plug_body_filtered_against_socket_net_normal_force_world_frame"
FORBIDDEN_CONTACT_LOAD_SOURCE = (
    "physx_rigid_contact_view_filtered_pairs_sum_abs_normal_forces"
)


class DatasetContractError(ValueError):
    """데이터가 기록 계약을 위반했을 때 발생한다."""


class ProvenanceKind(str, Enum):
    ACTUAL_SIMULATOR_ROLLOUT = "actual_simulator_rollout"
    SOURCE_EXTERNAL_REFERENCE = "source_external_reference"
    SYNTHETIC_TEST_FIXTURE = "synthetic_test_fixture"


class DatasetSplit(str, Enum):
    TRAIN = "train"
    EVAL = "eval"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _validate_sha256(name: str, value: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise DatasetContractError(f"{name} must be a lowercase SHA-256 hex digest")


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(float(value)):
        raise DatasetContractError(f"{name} must be finite")


def split_for_seed(seed: int, *, eval_percent: int = 20, namespace: str = SCHEMA_NAME) -> DatasetSplit:
    """seed를 안정적으로 train/eval에 배정한다.

    Python의 실행별 hash salt를 쓰지 않고 SHA-256을 사용하므로 같은 seed와
    namespace는 기기나 실행 시점과 관계없이 같은 split을 얻는다.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DatasetContractError("seed must be a non-negative integer")
    if isinstance(eval_percent, bool) or not isinstance(eval_percent, int):
        raise DatasetContractError("eval_percent must be an integer")
    if not 1 <= eval_percent <= 99:
        raise DatasetContractError("eval_percent must be between 1 and 99")
    if not namespace:
        raise DatasetContractError("split namespace must not be empty")
    digest = hashlib.sha256(f"{namespace}:{seed}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    return DatasetSplit.EVAL if bucket < eval_percent else DatasetSplit.TRAIN


@dataclass(frozen=True)
class EpisodeProvenance:
    """episode의 생성 경로와 재현에 필요한 출처."""

    kind: ProvenanceKind
    source_hashes: Mapping[str, str]
    task_config_sha256: str
    runtime_versions: Mapping[str, str]
    observation_source: str
    contact_force_source: str
    owns_demonstration: bool
    external_reference: str | None = None
    experiment_parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ProvenanceKind):
            raise DatasetContractError("kind must be a ProvenanceKind")
        if not self.source_hashes:
            raise DatasetContractError("source_hashes must not be empty")
        for name, digest in self.source_hashes.items():
            if not name:
                raise DatasetContractError("source hash names must not be empty")
            _validate_sha256(f"source_hashes[{name}]", digest)
        _validate_sha256("task_config_sha256", self.task_config_sha256)
        if not self.runtime_versions or any(not key or not value for key, value in self.runtime_versions.items()):
            raise DatasetContractError("runtime_versions must contain non-empty names and versions")
        if not self.observation_source:
            raise DatasetContractError("observation_source must not be empty")
        if not self.contact_force_source:
            raise DatasetContractError("contact_force_source must not be empty")
        if self.kind is not ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT and self.owns_demonstration:
            raise DatasetContractError("external or synthetic data cannot be marked as an owned demonstration")
        if self.kind is ProvenanceKind.SOURCE_EXTERNAL_REFERENCE and not self.external_reference:
            raise DatasetContractError("external reference data must name its source")

    @property
    def is_actual_rollout(self) -> bool:
        return self.kind is ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "source_hashes": dict(self.source_hashes),
            "task_config_sha256": self.task_config_sha256,
            "runtime_versions": dict(self.runtime_versions),
            "observation_source": self.observation_source,
            "contact_force_source": self.contact_force_source,
            "owns_demonstration": self.owns_demonstration,
            "external_reference": self.external_reference,
            "experiment_parameters": dict(self.experiment_parameters),
        }


@dataclass(frozen=True)
class EpisodeMetadata:
    episode_id: str
    seed: int
    rollout_index: int
    provenance: EpisodeProvenance
    split: DatasetSplit
    split_namespace: str = SCHEMA_NAME
    split_eval_percent: int = 20

    @classmethod
    def for_seed(
        cls,
        *,
        episode_id: str,
        seed: int,
        rollout_index: int,
        provenance: EpisodeProvenance,
        split_namespace: str = SCHEMA_NAME,
        split_eval_percent: int = 20,
    ) -> "EpisodeMetadata":
        return cls(
            episode_id=episode_id,
            seed=seed,
            rollout_index=rollout_index,
            provenance=provenance,
            split=split_for_seed(seed, eval_percent=split_eval_percent, namespace=split_namespace),
            split_namespace=split_namespace,
            split_eval_percent=split_eval_percent,
        )

    def __post_init__(self) -> None:
        if not _EPISODE_ID_PATTERN.fullmatch(self.episode_id):
            raise DatasetContractError("episode_id must be filesystem-safe and at most 128 characters")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise DatasetContractError("seed must be a non-negative integer")
        if isinstance(self.rollout_index, bool) or not isinstance(self.rollout_index, int) or self.rollout_index < 0:
            raise DatasetContractError("rollout_index must be a non-negative integer")
        expected = split_for_seed(
            self.seed, eval_percent=self.split_eval_percent, namespace=self.split_namespace
        )
        if self.split is not expected:
            raise DatasetContractError("split does not match the deterministic seed assignment")


@dataclass(frozen=True)
class GroundTruthMetrics:
    actual_depth_m: float
    lateral_error_m: float
    orientation_error_rad: float
    force_peak_n: float
    grasp_both: bool
    forbidden_contact: bool

    def __post_init__(self) -> None:
        for name in GROUND_TRUTH_FIELDS[:4]:
            _validate_finite(name, getattr(self, name))
        if self.lateral_error_m < 0.0 or self.orientation_error_rad < 0.0 or self.force_peak_n < 0.0:
            raise DatasetContractError("error and force metrics must be non-negative")

    def as_array(self) -> np.ndarray:
        return np.asarray(
            (
                self.actual_depth_m,
                self.lateral_error_m,
                self.orientation_error_rad,
                self.force_peak_n,
                float(self.grasp_both),
                float(self.forbidden_contact),
            ),
            dtype=np.float64,
        )


def _numeric_array(name: str, value: Any, *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise DatasetContractError(f"{name} must have shape {shape}, got {array.shape}")
    if shape is None and (array.ndim != 1 or array.size == 0):
        raise DatasetContractError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise DatasetContractError(f"{name} must contain only finite values")
    return array


@dataclass(frozen=True)
class TimestepRecord:
    timestamp_s: float
    observation: Any
    action: Any
    joint_position_rad: Any
    joint_velocity_rad_s: Any
    gripper_position_m: Any
    gripper_velocity_m_s: Any
    object_poses_world: Any
    contact_force_proxy_n: Any
    grip_contact_force_peak_n: float
    insertion_contact_force_n: float
    insertion_contact_force_w: Any
    insertion_contact_force_peak_n: float
    insertion_contact_load_n: float
    insertion_contact_load_peak_n: float
    forbidden_contact_load_n: float
    forbidden_contact_load_peak_n: float
    phase: str
    ground_truth: GroundTruthMetrics

    def normalized(self) -> dict[str, Any]:
        _validate_finite("timestamp_s", self.timestamp_s)
        if self.timestamp_s < 0.0:
            raise DatasetContractError("timestamp_s must be non-negative")
        observation = _numeric_array(
            "observation", self.observation, shape=(OBSERVATION_DIM,)
        )
        action = _numeric_array("action", self.action, shape=(ACTION_DIM,))
        joint_position = _numeric_array(
            "joint_position_rad", self.joint_position_rad, shape=(len(REVOLUTE_JOINT_ORDER),)
        )
        joint_velocity = _numeric_array(
            "joint_velocity_rad_s", self.joint_velocity_rad_s, shape=(len(REVOLUTE_JOINT_ORDER),)
        )
        gripper_position = _numeric_array(
            "gripper_position_m", self.gripper_position_m, shape=(len(GRIPPER_JOINT_ORDER),)
        )
        gripper_velocity = _numeric_array(
            "gripper_velocity_m_s", self.gripper_velocity_m_s, shape=(len(GRIPPER_JOINT_ORDER),)
        )
        object_poses = _numeric_array("object_poses_world", self.object_poses_world, shape=(2, 7))
        quaternion_norms = np.linalg.norm(object_poses[:, 3:7], axis=1)
        if not np.allclose(quaternion_norms, 1.0, atol=1e-3, rtol=0.0):
            raise DatasetContractError("object pose quaternions must be normalized xyzw values")
        contact_force = _numeric_array(
            "contact_force_proxy_n", self.contact_force_proxy_n, shape=(2,)
        )
        insertion_contact_force = _numeric_array(
            "insertion_contact_force_w", self.insertion_contact_force_w, shape=(3,)
        )
        for name, value in (
            ("grip_contact_force_peak_n", self.grip_contact_force_peak_n),
            ("insertion_contact_force_n", self.insertion_contact_force_n),
            ("insertion_contact_force_peak_n", self.insertion_contact_force_peak_n),
            ("insertion_contact_load_n", self.insertion_contact_load_n),
            ("insertion_contact_load_peak_n", self.insertion_contact_load_peak_n),
            ("forbidden_contact_load_n", self.forbidden_contact_load_n),
            ("forbidden_contact_load_peak_n", self.forbidden_contact_load_peak_n),
        ):
            _validate_finite(name, value)
            if value < 0.0:
                raise DatasetContractError(f"{name} must be non-negative")
        vector_norm = float(np.linalg.norm(insertion_contact_force))
        if not math.isclose(vector_norm, self.insertion_contact_force_n, abs_tol=1e-5, rel_tol=1e-5):
            raise DatasetContractError(
                "insertion_contact_force_n must equal the norm of insertion_contact_force_w"
            )
        if self.insertion_contact_force_peak_n + 1e-9 < self.insertion_contact_force_n:
            raise DatasetContractError("insertion contact peak cannot be below the current norm")
        if self.insertion_contact_load_n + 1e-4 < self.insertion_contact_force_n:
            raise DatasetContractError("insertion contact load cannot be below the net force norm")
        if self.insertion_contact_load_peak_n + 1e-9 < self.insertion_contact_load_n:
            raise DatasetContractError("insertion contact load peak cannot be below the current load")
        if self.insertion_contact_load_peak_n + 1e-4 < self.insertion_contact_force_peak_n:
            raise DatasetContractError("insertion contact load peak cannot be below the net-force peak")
        if self.forbidden_contact_load_peak_n + 1e-9 < self.forbidden_contact_load_n:
            raise DatasetContractError("forbidden contact load peak cannot be below the current load")
        if self.grip_contact_force_peak_n + 1e-9 < float(np.max(contact_force)):
            raise DatasetContractError("grip contact peak cannot be below the current hand force")
        if not isinstance(self.phase, str) or not self.phase.strip():
            raise DatasetContractError("phase must be a non-empty string")
        if not isinstance(self.ground_truth, GroundTruthMetrics):
            raise DatasetContractError("ground_truth must be GroundTruthMetrics")
        combined_force_peak = max(
            self.grip_contact_force_peak_n, self.insertion_contact_load_peak_n
        )
        if not math.isclose(
            self.ground_truth.force_peak_n,
            combined_force_peak,
            abs_tol=1e-4,
            rel_tol=1e-5,
        ):
            raise DatasetContractError(
                "ground_truth.force_peak_n must equal max(grip peak, insertion load peak)"
            )
        return {
            "timestamp_s": float(self.timestamp_s),
            "observation": observation,
            "action": action,
            "joint_position_rad": joint_position,
            "joint_velocity_rad_s": joint_velocity,
            "gripper_position_m": gripper_position,
            "gripper_velocity_m_s": gripper_velocity,
            "object_poses_world": object_poses,
            "contact_force_proxy_n": contact_force,
            "grip_contact_force_peak_n": float(self.grip_contact_force_peak_n),
            "insertion_contact_force_n": float(self.insertion_contact_force_n),
            "insertion_contact_force_w": insertion_contact_force,
            "insertion_contact_force_peak_n": float(self.insertion_contact_force_peak_n),
            "insertion_contact_load_n": float(self.insertion_contact_load_n),
            "insertion_contact_load_peak_n": float(self.insertion_contact_load_peak_n),
            "forbidden_contact_load_n": float(self.forbidden_contact_load_n),
            "forbidden_contact_load_peak_n": float(self.forbidden_contact_load_peak_n),
            "phase": self.phase.strip(),
            "ground_truth_metrics": self.ground_truth.as_array(),
        }


@dataclass(frozen=True)
class TerminalEvaluation:
    success: bool
    valid: bool
    termination_reason: str
    evaluator_source: str
    evaluator_version: str
    terminal_timestamp_s: float
    metrics: GroundTruthMetrics
    criteria: Mapping[str, bool]

    def __post_init__(self) -> None:
        _validate_finite("terminal_timestamp_s", self.terminal_timestamp_s)
        if self.terminal_timestamp_s < 0.0:
            raise DatasetContractError("terminal_timestamp_s must be non-negative")
        if not self.termination_reason:
            raise DatasetContractError("termination_reason must not be empty")
        if not self.evaluator_source or not self.evaluator_version:
            raise DatasetContractError("evaluator source and version must not be empty")
        if self.evaluator_source != "simulator_ground_truth_independent":
            raise DatasetContractError(
                "terminal results must come from simulator_ground_truth_independent"
            )
        if not self.criteria or any(not key or not isinstance(value, bool) for key, value in self.criteria.items()):
            raise DatasetContractError("criteria must contain named boolean results")
        if self.success != (self.valid and all(self.criteria.values())):
            raise DatasetContractError("success must equal valid AND all evaluation criteria")

    def as_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "valid": self.valid,
            "termination_reason": self.termination_reason,
            "evaluator_source": self.evaluator_source,
            "evaluator_version": self.evaluator_version,
            "terminal_timestamp_s": self.terminal_timestamp_s,
            "metrics": dict(zip(GROUND_TRUTH_FIELDS, self.metrics.as_array().tolist(), strict=True)),
            "criteria": dict(self.criteria),
        }


class EpisodeDatasetWriter:
    """한 번에 한 episode를 원자적으로 기록하는 HDF5 writer."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "a")
        self._initialize_file()
        self._active: h5py.Group | None = None
        self._metadata: EpisodeMetadata | None = None
        self._last_timestamp_s: float | None = None
        self._dimensions: dict[str, tuple[int, ...]] = {}

    def _initialize_file(self) -> None:
        schema_name = self._file.attrs.get("schema_name")
        schema_version = self._file.attrs.get("schema_version")
        if schema_name is not None and schema_name != SCHEMA_NAME:
            raise DatasetContractError(f"unexpected schema_name: {schema_name}")
        if schema_version is not None and int(schema_version) != SCHEMA_VERSION:
            raise DatasetContractError(f"unsupported schema_version: {schema_version}")
        self._file.attrs["schema_name"] = SCHEMA_NAME
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.require_group("episodes")
        self._file.require_group("_in_progress")

    def __enter__(self) -> "EpisodeDatasetWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active is not None:
            self.reset_episode()
        self.close()

    def close(self) -> None:
        if self._active is not None:
            raise DatasetContractError("an episode is active; finalize or reset it before closing")
        if self._file:
            self._file.flush()
            self._file.close()

    def begin_episode(self, metadata: EpisodeMetadata) -> None:
        if self._active is not None:
            raise DatasetContractError("finish or reset the active episode first")
        if metadata.episode_id in self._file["episodes"] or metadata.episode_id in self._file["_in_progress"]:
            raise DatasetContractError(f"episode_id already exists: {metadata.episode_id}")
        group = self._file["_in_progress"].create_group(metadata.episode_id)
        group.attrs["seed"] = metadata.seed
        group.attrs["rollout_index"] = metadata.rollout_index
        group.attrs["split"] = metadata.split.value
        group.attrs["split_namespace"] = metadata.split_namespace
        group.attrs["split_eval_percent"] = metadata.split_eval_percent
        group.attrs["provenance_json"] = _canonical_json(metadata.provenance.as_dict())
        group.attrs["complete"] = False
        self._active = group
        self._metadata = metadata
        self._last_timestamp_s = None
        self._dimensions = {}
        self._file.flush()

    def _append_dataset(self, name: str, value: Any, value_shape: tuple[int, ...], *, dtype: Any) -> None:
        assert self._active is not None
        if name not in self._active:
            maxshape = (None, *value_shape)
            dataset = self._active.create_dataset(
                name,
                shape=(0, *value_shape),
                maxshape=maxshape,
                chunks=(1, *value_shape),
                dtype=dtype,
            )
        dataset = self._active[name]
        dataset.resize(dataset.shape[0] + 1, axis=0)
        dataset[-1] = value

    def append(self, record: TimestepRecord) -> None:
        if self._active is None:
            raise DatasetContractError("begin_episode must be called before append")
        normalized = record.normalized()
        timestamp = normalized["timestamp_s"]
        if self._last_timestamp_s is not None and timestamp <= self._last_timestamp_s:
            raise DatasetContractError("timestamps must be strictly increasing inside an episode")
        for name in ("observation", "action"):
            shape = normalized[name].shape
            expected = self._dimensions.setdefault(name, shape)
            if shape != expected:
                raise DatasetContractError(f"{name} shape changed from {expected} to {shape}")
        self._append_dataset("timestamps_s", timestamp, (), dtype=np.float64)
        self._append_dataset("observations", normalized["observation"], normalized["observation"].shape, dtype=np.float64)
        self._append_dataset("actions", normalized["action"], normalized["action"].shape, dtype=np.float64)
        self._append_dataset(
            "joint_positions_rad", normalized["joint_position_rad"], normalized["joint_position_rad"].shape, dtype=np.float64
        )
        self._append_dataset(
            "joint_velocities_rad_s", normalized["joint_velocity_rad_s"], normalized["joint_velocity_rad_s"].shape, dtype=np.float64
        )
        self._append_dataset(
            "gripper_positions_m", normalized["gripper_position_m"], normalized["gripper_position_m"].shape, dtype=np.float64
        )
        self._append_dataset(
            "gripper_velocities_m_s", normalized["gripper_velocity_m_s"], normalized["gripper_velocity_m_s"].shape, dtype=np.float64
        )
        self._append_dataset("object_poses_world", normalized["object_poses_world"], (2, 7), dtype=np.float64)
        self._append_dataset("contact_force_proxy_n", normalized["contact_force_proxy_n"], (2,), dtype=np.float64)
        self._append_dataset("grip_contact_force_peak_n", normalized["grip_contact_force_peak_n"], (), dtype=np.float64)
        self._append_dataset("insertion_contact_force_n", normalized["insertion_contact_force_n"], (), dtype=np.float64)
        self._append_dataset("insertion_contact_force_w", normalized["insertion_contact_force_w"], (3,), dtype=np.float64)
        self._append_dataset("insertion_contact_force_peak_n", normalized["insertion_contact_force_peak_n"], (), dtype=np.float64)
        self._append_dataset("insertion_contact_load_n", normalized["insertion_contact_load_n"], (), dtype=np.float64)
        self._append_dataset("insertion_contact_load_peak_n", normalized["insertion_contact_load_peak_n"], (), dtype=np.float64)
        self._append_dataset("forbidden_contact_load_n", normalized["forbidden_contact_load_n"], (), dtype=np.float64)
        self._append_dataset("forbidden_contact_load_peak_n", normalized["forbidden_contact_load_peak_n"], (), dtype=np.float64)
        self._append_dataset("phase", normalized["phase"], (), dtype=h5py.string_dtype("utf-8"))
        self._append_dataset("ground_truth_metrics", normalized["ground_truth_metrics"], (len(GROUND_TRUTH_FIELDS),), dtype=np.float64)
        self._active["object_poses_world"].attrs["object_order"] = _canonical_json(["socket", "plug"])
        self._active["object_poses_world"].attrs["quaternion_order"] = "xyzw"
        self._active["contact_force_proxy_n"].attrs["sensor_order"] = _canonical_json(["left", "right"])
        self._active["contact_force_proxy_n"].attrs["representation"] = "per_hand_magnitude"
        self._active["contact_force_proxy_n"].attrs["source"] = "filtered_finger_object_contact_sensor_proxy"
        self._active["contact_force_proxy_n"].attrs["is_wrist_six_axis_ft_sensor"] = False
        self._active["grip_contact_force_peak_n"].attrs["representation"] = "episode_peak_of_per_hand_filtered_finger_magnitude"
        self._active["grip_contact_force_peak_n"].attrs["measurement_available"] = True
        self._active["insertion_contact_force_n"].attrs["representation"] = "world_frame_net_force_proxy_norm"
        self._active["insertion_contact_force_n"].attrs["measurement_available"] = True
        self._active["insertion_contact_force_n"].attrs["source"] = INSERTION_FORCE_SOURCE
        self._active["insertion_contact_force_w"].attrs["frame"] = "world"
        self._active["insertion_contact_force_w"].attrs["representation"] = "net_contact_force_proxy_vector"
        self._active["insertion_contact_force_w"].attrs["is_wrist_six_axis_ft_sensor"] = False
        self._active["insertion_contact_force_w"].attrs["measurement_available"] = True
        self._active["insertion_contact_force_w"].attrs["source"] = INSERTION_FORCE_SOURCE
        self._active["insertion_contact_force_peak_n"].attrs["representation"] = "episode_peak_of_world_frame_net_force_proxy_norm"
        self._active["insertion_contact_force_peak_n"].attrs["measurement_available"] = True
        self._active["insertion_contact_load_n"].attrs["representation"] = "sum_of_absolute_contact_normal_magnitudes"
        self._active["insertion_contact_load_n"].attrs["measurement_available"] = True
        self._active["insertion_contact_load_n"].attrs["source"] = INSERTION_LOAD_SOURCE
        self._active["insertion_contact_load_peak_n"].attrs["representation"] = "episode_peak_of_sum_of_absolute_contact_normal_magnitudes"
        self._active["insertion_contact_load_peak_n"].attrs["measurement_available"] = True
        self._active["insertion_contact_load_peak_n"].attrs["source"] = INSERTION_LOAD_SOURCE
        self._active["forbidden_contact_load_n"].attrs["representation"] = "sum_of_absolute_filtered_pair_normal_forces"
        self._active["forbidden_contact_load_n"].attrs["measurement_available"] = True
        self._active["forbidden_contact_load_n"].attrs["source"] = FORBIDDEN_CONTACT_LOAD_SOURCE
        self._active["forbidden_contact_load_peak_n"].attrs["representation"] = "episode_peak_of_sum_of_absolute_filtered_pair_normal_forces"
        self._active["forbidden_contact_load_peak_n"].attrs["measurement_available"] = True
        self._active["forbidden_contact_load_peak_n"].attrs["source"] = FORBIDDEN_CONTACT_LOAD_SOURCE
        self._active["ground_truth_metrics"].attrs["fields"] = _canonical_json(GROUND_TRUTH_FIELDS)
        self._active["joint_positions_rad"].attrs["joint_order"] = _canonical_json(REVOLUTE_JOINT_ORDER)
        self._active["joint_velocities_rad_s"].attrs["joint_order"] = _canonical_json(REVOLUTE_JOINT_ORDER)
        self._active["gripper_positions_m"].attrs["joint_order"] = _canonical_json(GRIPPER_JOINT_ORDER)
        self._active["gripper_velocities_m_s"].attrs["joint_order"] = _canonical_json(GRIPPER_JOINT_ORDER)
        self._last_timestamp_s = timestamp

    def finalize_episode(self, evaluation: TerminalEvaluation) -> str:
        if self._active is None or self._metadata is None:
            raise DatasetContractError("no active episode to finalize")
        if self._last_timestamp_s is None:
            raise DatasetContractError("an episode must contain at least one timestep")
        if evaluation.terminal_timestamp_s < self._last_timestamp_s:
            raise DatasetContractError("terminal evaluation cannot precede the final timestep")
        self._active.attrs["evaluation_json"] = _canonical_json(evaluation.as_dict())
        self._active.attrs["success"] = evaluation.success
        self._active.attrs["evaluation_valid"] = evaluation.valid
        self._active.attrs["timestep_count"] = len(self._active["timestamps_s"])
        episode_hash = _hash_group(self._active)
        self._active.attrs["episode_sha256"] = episode_hash
        self._active.attrs["complete"] = True
        episode_id = self._metadata.episode_id
        self._file.flush()
        self._file.move(f"_in_progress/{episode_id}", f"episodes/{episode_id}")
        self._file.flush()
        self._active = None
        self._metadata = None
        self._last_timestamp_s = None
        self._dimensions = {}
        return episode_hash

    def reset_episode(self) -> None:
        """완료되지 않은 현재 episode만 폐기하고 다음 기록 경계를 연다."""

        if self._active is None or self._metadata is None:
            return
        episode_id = self._metadata.episode_id
        self._active = None
        self._metadata = None
        self._last_timestamp_s = None
        self._dimensions = {}
        del self._file[f"_in_progress/{episode_id}"]
        self._file.flush()


def _hash_group(group: h5py.Group) -> str:
    digest = hashlib.sha256()
    attributes = {
        key: _json_compatible(value)
        for key, value in group.attrs.items()
        if key not in {"episode_sha256", "complete"}
    }
    digest.update(_canonical_json(attributes).encode("utf-8"))
    for name in sorted(group.keys()):
        dataset = group[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(dataset.dtype).encode("ascii"))
        digest.update(_canonical_json(dataset.shape).encode("ascii"))
        values = dataset[...]
        if values.dtype.kind in {"O", "S", "U"}:
            digest.update(_canonical_json([_decode_string(value) for value in values.reshape(-1)]).encode("utf-8"))
        else:
            digest.update(np.ascontiguousarray(values).tobytes())
        digest.update(_canonical_json({key: _json_compatible(value) for key, value in dataset.attrs.items()}).encode("utf-8"))
    return digest.hexdigest()


def _decode_string(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def select_episode_ids(
    path: str | Path,
    *,
    split: DatasetSplit | None = None,
    success_only: bool = False,
    training_eligible_only: bool = False,
) -> list[str]:
    """필터 조건에 맞는 완료 episode ID를 정렬해 반환한다."""

    selected: list[str] = []
    with h5py.File(path, "r") as file:
        _validate_file_schema(file)
        for episode_id in sorted(file["episodes"].keys()):
            group = file["episodes"][episode_id]
            validated = _validate_episode_group(group, verify_hash=True)
            provenance = validated["provenance"]
            if split is not None and group.attrs["split"] != split.value:
                continue
            if success_only and not bool(group.attrs["success"]):
                continue
            if training_eligible_only and not (
                group.attrs["split"] == DatasetSplit.TRAIN.value
                and bool(group.attrs["evaluation_valid"])
                and provenance["kind"] == ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT.value
                and provenance["owns_demonstration"] is True
            ):
                continue
            selected.append(episode_id)
    return selected


def load_training_episode(path: str | Path, episode_id: str) -> dict[str, np.ndarray]:
    """평가 split이나 참고·합성 데이터를 거부하고 학습 가능한 배열만 읽는다."""

    with h5py.File(path, "r") as file:
        _validate_file_schema(file)
        if episode_id not in file["episodes"]:
            raise DatasetContractError(f"episode does not exist: {episode_id}")
        group = file[f"episodes/{episode_id}"]
        validated = _validate_episode_group(group, verify_hash=True)
        provenance = validated["provenance"]
        if not (
            group.attrs["split"] == DatasetSplit.TRAIN.value
            and bool(group.attrs["evaluation_valid"])
            and provenance["kind"] == ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT.value
            and provenance["owns_demonstration"] is True
        ):
            raise DatasetContractError(f"episode is not eligible for training: {episode_id}")
        return {name: group[name][...] for name in group.keys()}


def inspect_dataset(path: str | Path, *, verify_hashes: bool = True) -> dict[str, Any]:
    """데이터셋 구성·분할·성공·provenance와 무결성을 JSON 가능한 값으로 요약한다."""

    counts = {
        "episodes": 0,
        "train": 0,
        "eval": 0,
        "success": 0,
        "failure": 0,
        "actual_simulator_rollout": 0,
        "source_external_reference": 0,
        "synthetic_test_fixture": 0,
        "training_eligible": 0,
    }
    invalid_hashes: list[str] = []
    invalid_contracts: list[str] = []
    with h5py.File(path, "r") as file:
        _validate_file_schema(file)
        in_progress = sorted(file["_in_progress"].keys())
        for episode_id in sorted(file["episodes"].keys()):
            group = file["episodes"][episode_id]
            counts["episodes"] += 1
            try:
                validated = _validate_episode_group(group, verify_hash=verify_hashes)
                split = validated["split"]
                success = validated["success"]
                provenance = validated["provenance"]
                kind = provenance["kind"]
                counts[split] += 1
                counts["success" if success else "failure"] += 1
                counts[kind] += 1
            except DatasetContractError as error:
                message = str(error)
                if message == "episode hash does not match its content":
                    invalid_hashes.append(episode_id)
                else:
                    invalid_contracts.append(f"{episode_id}: {message}")
                continue
            if (
                split == DatasetSplit.TRAIN.value
                and bool(group.attrs.get("evaluation_valid", False))
                and kind == ProvenanceKind.ACTUAL_SIMULATOR_ROLLOUT.value
                and provenance["owns_demonstration"] is True
            ):
                counts["training_eligible"] += 1
    return {
        "path": str(Path(path).resolve()),
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "counts": counts,
        "in_progress_episode_ids": in_progress,
        "invalid_hash_episode_ids": invalid_hashes,
        "contract_errors": invalid_contracts,
        "valid": not in_progress and not invalid_hashes and not invalid_contracts,
    }


def _validate_file_schema(file: h5py.File) -> None:
    if file.attrs.get("schema_name") != SCHEMA_NAME:
        raise DatasetContractError("not an RB-Y1 assembly episode dataset")
    if int(file.attrs.get("schema_version", -1)) != SCHEMA_VERSION:
        raise DatasetContractError("unsupported dataset schema version")
    if "episodes" not in file or "_in_progress" not in file:
        raise DatasetContractError("dataset groups are missing")


_TIMESTEP_DATASET_SHAPES = {
    "timestamps_s": (),
    "observations": (OBSERVATION_DIM,),
    "actions": (ACTION_DIM,),
    "joint_positions_rad": (len(REVOLUTE_JOINT_ORDER),),
    "joint_velocities_rad_s": (len(REVOLUTE_JOINT_ORDER),),
    "gripper_positions_m": (len(GRIPPER_JOINT_ORDER),),
    "gripper_velocities_m_s": (len(GRIPPER_JOINT_ORDER),),
    "object_poses_world": (2, 7),
    "contact_force_proxy_n": (2,),
    "grip_contact_force_peak_n": (),
    "insertion_contact_force_n": (),
    "insertion_contact_force_w": (3,),
    "insertion_contact_force_peak_n": (),
    "insertion_contact_load_n": (),
    "insertion_contact_load_peak_n": (),
    "forbidden_contact_load_n": (),
    "forbidden_contact_load_peak_n": (),
    "phase": (),
    "ground_truth_metrics": (len(GROUND_TRUTH_FIELDS),),
}
_REQUIRED_DATASET_ATTRS = {
    "object_poses_world": {
        "object_order": _canonical_json(["socket", "plug"]),
        "quaternion_order": "xyzw",
    },
    "contact_force_proxy_n": {
        "sensor_order": _canonical_json(["left", "right"]),
        "representation": "per_hand_magnitude",
        "source": "filtered_finger_object_contact_sensor_proxy",
        "is_wrist_six_axis_ft_sensor": False,
    },
    "grip_contact_force_peak_n": {"measurement_available": True},
    "insertion_contact_force_n": {
        "measurement_available": True,
        "source": INSERTION_FORCE_SOURCE,
    },
    "insertion_contact_force_w": {
        "frame": "world",
        "is_wrist_six_axis_ft_sensor": False,
        "measurement_available": True,
        "source": INSERTION_FORCE_SOURCE,
    },
    "insertion_contact_force_peak_n": {"measurement_available": True},
    "insertion_contact_load_n": {
        "measurement_available": True,
        "source": INSERTION_LOAD_SOURCE,
    },
    "insertion_contact_load_peak_n": {
        "measurement_available": True,
        "source": INSERTION_LOAD_SOURCE,
    },
    "forbidden_contact_load_n": {
        "measurement_available": True,
        "source": FORBIDDEN_CONTACT_LOAD_SOURCE,
    },
    "forbidden_contact_load_peak_n": {
        "measurement_available": True,
        "source": FORBIDDEN_CONTACT_LOAD_SOURCE,
    },
    "ground_truth_metrics": {"fields": _canonical_json(GROUND_TRUTH_FIELDS)},
    "joint_positions_rad": {"joint_order": _canonical_json(REVOLUTE_JOINT_ORDER)},
    "joint_velocities_rad_s": {"joint_order": _canonical_json(REVOLUTE_JOINT_ORDER)},
    "gripper_positions_m": {"joint_order": _canonical_json(GRIPPER_JOINT_ORDER)},
    "gripper_velocities_m_s": {"joint_order": _canonical_json(GRIPPER_JOINT_ORDER)},
}


def _required_attr(group: h5py.Group, name: str) -> Any:
    if name not in group.attrs:
        raise DatasetContractError(f"required episode attribute is missing: {name}")
    return group.attrs[name]


def _validate_provenance_json(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(_decode_string(raw))
        provenance = EpisodeProvenance(
            kind=ProvenanceKind(value["kind"]),
            source_hashes=value["source_hashes"],
            task_config_sha256=value["task_config_sha256"],
            runtime_versions=value["runtime_versions"],
            observation_source=value["observation_source"],
            contact_force_source=value["contact_force_source"],
            owns_demonstration=value["owns_demonstration"],
            external_reference=value.get("external_reference"),
            experiment_parameters=value.get("experiment_parameters", {}),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetContractError("invalid provenance_json") from error
    return provenance.as_dict()


def _validate_episode_group(
    group: h5py.Group, *, verify_hash: bool
) -> dict[str, Any]:
    """완료 episode의 구조·의미·출처·해시를 하나의 계약으로 검증한다."""

    if not bool(_required_attr(group, "complete")):
        raise DatasetContractError("episode is not marked complete")
    episode_id = group.name.rsplit("/", 1)[-1]
    if not _EPISODE_ID_PATTERN.fullmatch(episode_id):
        raise DatasetContractError("episode group name is invalid")
    split = _decode_string(_required_attr(group, "split"))
    if split not in {item.value for item in DatasetSplit}:
        raise DatasetContractError(f"invalid split: {split}")
    try:
        seed = int(_required_attr(group, "seed"))
        rollout_index = int(_required_attr(group, "rollout_index"))
        split_eval_percent = int(_required_attr(group, "split_eval_percent"))
    except (TypeError, ValueError) as error:
        raise DatasetContractError("episode integer metadata is invalid") from error
    split_namespace = _decode_string(_required_attr(group, "split_namespace"))
    if seed < 0 or rollout_index < 0:
        raise DatasetContractError("seed and rollout_index must be non-negative")
    if split != split_for_seed(
        seed, eval_percent=split_eval_percent, namespace=split_namespace
    ).value:
        raise DatasetContractError("split does not match episode seed metadata")
    provenance = _validate_provenance_json(_required_attr(group, "provenance_json"))
    success = bool(_required_attr(group, "success"))
    evaluation_valid = bool(_required_attr(group, "evaluation_valid"))
    try:
        evaluation = json.loads(_decode_string(_required_attr(group, "evaluation_json")))
        metric_values = evaluation["metrics"]
        terminal = TerminalEvaluation(
            success=bool(evaluation["success"]),
            valid=bool(evaluation["valid"]),
            termination_reason=evaluation["termination_reason"],
            evaluator_source=evaluation["evaluator_source"],
            evaluator_version=evaluation["evaluator_version"],
            terminal_timestamp_s=float(evaluation["terminal_timestamp_s"]),
            metrics=GroundTruthMetrics(
                **{name: metric_values[name] for name in GROUND_TRUTH_FIELDS}
            ),
            criteria=evaluation["criteria"],
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise DatasetContractError("invalid evaluation_json") from error
    if terminal.success != success or terminal.valid != evaluation_valid:
        raise DatasetContractError("episode evaluation attributes disagree with evaluation_json")
    expected_count = int(_required_attr(group, "timestep_count"))
    if expected_count <= 0:
        raise DatasetContractError("timestep_count must be positive")

    missing = sorted(set(_TIMESTEP_DATASET_SHAPES) - set(group.keys()))
    if missing:
        raise DatasetContractError("required timestep datasets are missing: " + ", ".join(missing))
    for name, sample_shape in _TIMESTEP_DATASET_SHAPES.items():
        dataset = group[name]
        expected_shape = (expected_count, *sample_shape)
        if dataset.shape != expected_shape:
            raise DatasetContractError(
                f"{name} must have shape {expected_shape}, got {dataset.shape}"
            )
    for dataset_name, expected_attrs in _REQUIRED_DATASET_ATTRS.items():
        dataset = group[dataset_name]
        for attr_name, expected in expected_attrs.items():
            if attr_name not in dataset.attrs:
                raise DatasetContractError(
                    f"{dataset_name} required attribute is missing: {attr_name}"
                )
            actual = _json_compatible(dataset.attrs[attr_name])
            if actual != expected:
                raise DatasetContractError(
                    f"{dataset_name}.{attr_name} is invalid"
                )

    timestamps = np.asarray(group["timestamps_s"][...], dtype=np.float64)
    if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
        raise DatasetContractError("timestamps must be finite and strictly increasing")
    if terminal.terminal_timestamp_s + 1e-9 < float(timestamps[-1]):
        raise DatasetContractError("terminal evaluation precedes the final timestep")
    for name in _TIMESTEP_DATASET_SHAPES:
        if name == "phase":
            if any(not _decode_string(value).strip() for value in group[name][...]):
                raise DatasetContractError("phase values must be non-empty")
            continue
        values = np.asarray(group[name][...], dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise DatasetContractError(f"{name} must contain only finite values")

    insertion_force = np.asarray(group["insertion_contact_force_n"][...])
    insertion_vector_norm = np.linalg.norm(group["insertion_contact_force_w"][...], axis=1)
    insertion_force_peak = np.asarray(group["insertion_contact_force_peak_n"][...])
    insertion_load = np.asarray(group["insertion_contact_load_n"][...])
    insertion_load_peak = np.asarray(group["insertion_contact_load_peak_n"][...])
    forbidden_load = np.asarray(group["forbidden_contact_load_n"][...])
    forbidden_load_peak = np.asarray(group["forbidden_contact_load_peak_n"][...])
    grip_peak = np.asarray(group["grip_contact_force_peak_n"][...])
    ground_truth_force_peak = np.asarray(group["ground_truth_metrics"][:, 3])
    if not np.allclose(insertion_force, insertion_vector_norm, atol=1e-5, rtol=1e-5):
        raise DatasetContractError("insertion net-force norm does not match its vector")
    if np.any(insertion_force_peak + 1e-9 < insertion_force):
        raise DatasetContractError("insertion net-force peak is below the current norm")
    if np.any(insertion_load + 1e-4 < insertion_force):
        raise DatasetContractError("insertion contact load is below the net-force norm")
    if np.any(insertion_load_peak + 1e-9 < insertion_load):
        raise DatasetContractError("insertion contact load peak is below the current load")
    if np.any(insertion_load_peak + 1e-4 < insertion_force_peak):
        raise DatasetContractError("insertion contact load peak is below the net-force peak")
    if np.any(forbidden_load_peak + 1e-9 < forbidden_load):
        raise DatasetContractError("forbidden contact load peak is below the current load")
    combined_peak = np.maximum(grip_peak, insertion_load_peak)
    if not np.allclose(ground_truth_force_peak, combined_peak, atol=1e-4, rtol=1e-5):
        raise DatasetContractError("ground-truth force peak is not the combined measured peak")
    if not np.allclose(
        group["ground_truth_metrics"][-1], terminal.metrics.as_array(), atol=1e-8, rtol=0.0
    ):
        raise DatasetContractError("terminal evaluation metrics disagree with the final timestep")

    episode_hash = _decode_string(_required_attr(group, "episode_sha256"))
    _validate_sha256("episode_sha256", episode_hash)
    if verify_hash and episode_hash != _hash_group(group):
        raise DatasetContractError("episode hash does not match its content")
    return {"split": split, "success": success, "provenance": provenance}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
