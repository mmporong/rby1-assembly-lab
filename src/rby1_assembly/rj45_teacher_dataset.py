"""RJ45 teacher episode의 79D 관측·6D 행동 HDF5 계약.

기존 조립 데이터셋과 분리해 teacher 수집의 성공·실패 시도를 모두 보존한다.
student loader는 관측과 실제 적용 행동만 반환하며 privileged 정답은 노출하지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
import os
from pathlib import Path
import re
from typing import Sequence
import uuid

import h5py
import numpy as np


SCHEMA_NAME = "rj45_teacher_episode_v1"
OBSERVATION_DIM = 79
ACTION_DIM = 6
ALLOWED_SPLITS = frozenset({"train", "dev"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EPISODE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_STUDENT_KEYS = frozenset(
    {
        "episode_id",
        "observation",
        "action",
        "observation_timestamp",
        "valid",
        "phase",
        "timestep",
        "metadata",
    }
)
_PROVENANCE_SHA_FIELDS = ("source_sha256", "task_sha256", "training_sha256", "bank_sha256")


class TeacherDatasetContractError(ValueError):
    """RJ45 teacher 데이터가 계약을 위반했을 때 발생한다."""


@dataclass(frozen=True)
class TeacherEpisodeProvenance:
    grasp_id: str
    split: str
    bias_case_id: str
    seed: int
    source_sha256: str
    task_sha256: str
    training_sha256: str
    bank_sha256: str
    translation_bound_m: float
    rotation_bound_rad: float

    def validate(self) -> None:
        if self.split not in ALLOWED_SPLITS:
            raise TeacherDatasetContractError("split must be train or dev; final collection is forbidden")
        if (
            not isinstance(self.grasp_id, str)
            or not isinstance(self.bias_case_id, str)
            or not self.grasp_id
            or not self.bias_case_id
        ):
            raise TeacherDatasetContractError("grasp_id and bias_case_id must not be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise TeacherDatasetContractError("seed must be a non-negative integer")
        for name in _PROVENANCE_SHA_FIELDS:
            value = getattr(self, name)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise TeacherDatasetContractError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("translation_bound_m", "rotation_bound_rad"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0.0
            ):
                raise TeacherDatasetContractError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class TeacherEpisodeRows:
    observation: np.ndarray
    observation_timestamp: np.ndarray
    valid: np.ndarray
    teacher_requested: np.ndarray
    teacher_applied: np.ndarray
    applied_offset: np.ndarray
    phase: Sequence[str]
    timestep: np.ndarray
    teacher_gt_delta: np.ndarray | None = None
    teacher_gt_pose: np.ndarray | None = None


@dataclass(frozen=True)
class TeacherEpisodeTerminal:
    independent_success: bool
    valid: bool
    failure_reason: str

    def validate(self) -> None:
        if not isinstance(self.independent_success, (bool, np.bool_)):
            raise TeacherDatasetContractError("independent_success must be boolean")
        if not isinstance(self.valid, (bool, np.bool_)):
            raise TeacherDatasetContractError("terminal valid must be boolean")
        if not isinstance(self.failure_reason, str):
            raise TeacherDatasetContractError("failure_reason must be a string")
        if self.independent_success and not self.valid:
            raise TeacherDatasetContractError("independent_success requires a valid terminal result")
        if (not self.independent_success or not self.valid) and not self.failure_reason:
            raise TeacherDatasetContractError("failed or invalid attempts require failure_reason")
        if self.independent_success and self.valid and self.failure_reason:
            raise TeacherDatasetContractError("successful valid attempts must not have failure_reason")


def _finite_array(name: str, value: object, shape: tuple[int, ...], dtype: object) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.shape != shape:
        raise TeacherDatasetContractError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise TeacherDatasetContractError(f"{name} must contain only finite values")
    return array


def _project_teacher_action(requested: np.ndarray) -> np.ndarray:
    clipped = np.clip(requested, -1.0, 1.0).reshape((-1, 2, 3))
    norms = np.linalg.norm(clipped, axis=-1, keepdims=True)
    return (clipped / np.maximum(norms, 1.0)).reshape((-1, ACTION_DIM))


def _normalize_rows(
    rows: TeacherEpisodeRows, provenance: TeacherEpisodeProvenance
) -> dict[str, np.ndarray]:
    observation = np.asarray(rows.observation, dtype=np.float32)
    if observation.ndim != 2 or observation.shape[1:] != (OBSERVATION_DIM,):
        raise TeacherDatasetContractError(f"observation must have shape (N, {OBSERVATION_DIM})")
    count = observation.shape[0]
    if count == 0:
        raise TeacherDatasetContractError("an episode must contain at least one row")
    if not np.isfinite(observation).all():
        raise TeacherDatasetContractError("observation must contain only finite values")

    timestamp = _finite_array("observation_timestamp", rows.observation_timestamp, (count,), np.float64)
    if np.any(timestamp < 0.0):
        raise TeacherDatasetContractError("observation_timestamp must be non-negative")
    if not np.all(np.diff(timestamp) > 0.0):
        raise TeacherDatasetContractError("observation_timestamp must be strictly increasing")
    requested = _finite_array("teacher_requested", rows.teacher_requested, (count, ACTION_DIM), np.float32)
    applied = _finite_array("teacher_applied", rows.teacher_applied, (count, ACTION_DIM), np.float32)
    offset = _finite_array("applied_offset", rows.applied_offset, (count, ACTION_DIM), np.float32)
    expected_applied = _project_teacher_action(requested)
    if not np.allclose(applied, expected_applied, rtol=1e-6, atol=1e-7):
        raise TeacherDatasetContractError("teacher_applied must be the bounded projection of teacher_requested")
    scales = np.asarray(
        [provenance.translation_bound_m] * 3 + [provenance.rotation_bound_rad] * 3,
        dtype=np.float32,
    )
    if not np.allclose(offset, applied * scales, rtol=1e-6, atol=1e-7):
        raise TeacherDatasetContractError("applied_offset must be teacher_applied scaled by physical bounds")

    valid = np.asarray(rows.valid)
    if valid.shape != (count,) or valid.dtype.kind != "b":
        raise TeacherDatasetContractError("valid must be a boolean array with shape (N,)")
    timestep = np.asarray(rows.timestep)
    if timestep.shape != (count,) or timestep.dtype.kind not in "iu":
        raise TeacherDatasetContractError("timestep must be an integer array with shape (N,)")
    if not np.array_equal(timestep, np.arange(count, dtype=timestep.dtype)):
        raise TeacherDatasetContractError("timestep must be contiguous and start at zero")
    phase = np.asarray(list(rows.phase), dtype=object)
    if phase.shape != (count,) or any(not isinstance(item, str) or not item for item in phase):
        raise TeacherDatasetContractError("phase must contain one non-empty string per row")

    normalized: dict[str, np.ndarray] = {
        "observation": observation,
        "observation_timestamp": timestamp,
        "valid": valid.astype(np.bool_),
        "teacher_requested": requested,
        "teacher_applied": applied,
        "applied_offset": offset,
        "phase": phase,
        "timestep": timestep.astype(np.int64),
    }
    if rows.teacher_gt_delta is not None:
        normalized["teacher_gt_delta"] = _finite_array(
            "teacher_gt_delta", rows.teacher_gt_delta, (count, ACTION_DIM), np.float32
        )
    if rows.teacher_gt_pose is not None:
        pose = np.asarray(rows.teacher_gt_pose, dtype=np.float32)
        if pose.ndim != 2 or pose.shape[0] != count or pose.shape[1] == 0 or not np.isfinite(pose).all():
            raise TeacherDatasetContractError("teacher_gt_pose must be finite with shape (N, P), P > 0")
        normalized["teacher_gt_pose"] = pose
    return normalized


class TeacherDatasetWriter:
    """완료된 데이터셋을 임시 파일에서 대상 파일로 원자적으로 게시한다."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"dataset already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._temporary_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        self._file = h5py.File(self._temporary_path, "x")
        self._file.attrs["schema_name"] = SCHEMA_NAME
        self._file.attrs["observation_mode"] = "standard79"
        self._file.attrs["observation_dim"] = OBSERVATION_DIM
        self._file.attrs["action_dim"] = ACTION_DIM
        self._file.create_group("episodes")
        self._episode_count = 0
        self._closed = False
        self._grasp_splits: dict[str, str] = {}
        self._bias_seed_splits: dict[tuple[str, int], str] = {}

    def __enter__(self) -> "TeacherDatasetWriter":
        return self

    def add_episode(
        self,
        episode_id: str,
        provenance: TeacherEpisodeProvenance,
        rows: TeacherEpisodeRows,
        terminal: TeacherEpisodeTerminal | None,
    ) -> None:
        if self._closed:
            raise TeacherDatasetContractError("writer is closed")
        if terminal is None:
            raise TeacherDatasetContractError("terminal result is required")
        if not _EPISODE_ID.fullmatch(episode_id):
            raise TeacherDatasetContractError("episode_id must be filesystem-safe and at most 128 characters")
        provenance.validate()
        terminal.validate()
        normalized = _normalize_rows(rows, provenance)
        episodes = self._file["episodes"]
        if episode_id in episodes:
            raise TeacherDatasetContractError(f"episode_id already exists: {episode_id}")
        self._validate_split_integrity(provenance)

        group = episodes.create_group(episode_id)
        for name in ("grasp_id", "split", "bias_case_id", "seed"):
            group.attrs[name] = getattr(provenance, name)
        for name in _PROVENANCE_SHA_FIELDS:
            group.attrs[name] = getattr(provenance, name)
        group.attrs["translation_bound_m"] = provenance.translation_bound_m
        group.attrs["rotation_bound_rad"] = provenance.rotation_bound_rad
        group.attrs["independent_success"] = terminal.independent_success
        group.attrs["valid"] = terminal.valid
        group.attrs["failure_reason"] = terminal.failure_reason
        group.create_dataset("observation", data=normalized["observation"])
        group.create_dataset("observation_timestamp", data=normalized["observation_timestamp"])
        group.create_dataset("valid", data=normalized["valid"])
        group.create_dataset("teacher_requested", data=normalized["teacher_requested"])
        group.create_dataset("teacher_applied", data=normalized["teacher_applied"])
        group.create_dataset("applied_offset", data=normalized["applied_offset"])
        group.create_dataset("phase", data=normalized["phase"], dtype=h5py.string_dtype("utf-8"))
        group.create_dataset("timestep", data=normalized["timestep"])
        privileged_names = ("teacher_gt_delta", "teacher_gt_pose")
        if any(name in normalized for name in privileged_names):
            privileged = group.create_group("privileged")
            for name in privileged_names:
                if name in normalized:
                    privileged.create_dataset(name, data=normalized[name])
        self._episode_count += 1
        self._file.flush()

    def _validate_split_integrity(self, provenance: TeacherEpisodeProvenance) -> None:
        keys = (
            (self._grasp_splits, provenance.grasp_id, "grasp_id"),
            (
                self._bias_seed_splits,
                (provenance.bias_case_id, provenance.seed),
                "bias_case_id/seed",
            ),
        )
        for registry, key, name in keys:
            previous = registry.get(key)
            if previous is not None and previous != provenance.split:
                raise TeacherDatasetContractError(f"{name} must not cross train/dev splits")
        self._grasp_splits[provenance.grasp_id] = provenance.split
        self._bias_seed_splits[(provenance.bias_case_id, provenance.seed)] = provenance.split

    def close(self) -> None:
        if self._closed:
            return
        if self._episode_count == 0:
            self.abort()
            raise TeacherDatasetContractError("dataset must contain at least one completed episode")
        self._file.flush()
        self._file.close()
        try:
            os.link(self._temporary_path, self.path)
        except FileExistsError:
            self._temporary_path.unlink(missing_ok=True)
            self._closed = True
            raise FileExistsError(f"dataset already exists: {self.path}")
        self._temporary_path.unlink()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._temporary_path.unlink(missing_ok=True)
        self._closed = True

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self.close()
        else:
            self.abort()


def load_student_episodes(
    path: str | Path,
    *,
    split: str = "train",
    successful_only: bool = True,
    valid_only: bool = True,
) -> list[dict[str, object]]:
    """student BC 입력을 읽는다. privileged와 teacher 내부값은 반환하지 않는다."""

    if split not in ALLOWED_SPLITS:
        raise TeacherDatasetContractError("loader split must be train or dev")
    loaded: list[dict[str, object]] = []
    with h5py.File(path, "r") as file:
        if file.attrs.get("schema_name") != SCHEMA_NAME:
            raise TeacherDatasetContractError("not an RJ45 teacher episode dataset")
        if file.attrs.get("observation_mode") != "standard79":
            raise TeacherDatasetContractError("observation_mode must be standard79")
        if file.attrs.get("observation_dim") != OBSERVATION_DIM:
            raise TeacherDatasetContractError("root observation_dim must be 79")
        if file.attrs.get("action_dim") != ACTION_DIM:
            raise TeacherDatasetContractError("root action_dim must be 6")
        if "episodes" not in file:
            raise TeacherDatasetContractError("episodes group is missing")
        validated: list[
            tuple[str, h5py.Group, TeacherEpisodeProvenance, TeacherEpisodeTerminal, dict[str, np.ndarray]]
        ] = []
        grasp_splits: dict[str, str] = {}
        bias_seed_splits: dict[tuple[str, int], str] = {}
        for episode_id in sorted(file["episodes"]):
            group = file["episodes"][episode_id]
            provenance = _provenance_from_group(group)
            terminal = _terminal_from_group(group)
            _validate_split_pair(grasp_splits, provenance.grasp_id, provenance.split, "grasp_id")
            _validate_split_pair(
                bias_seed_splits,
                (provenance.bias_case_id, provenance.seed),
                provenance.split,
                "bias_case_id/seed",
            )
            try:
                phase = np.asarray(group["phase"].asstr())
            except KeyError as error:
                raise TeacherDatasetContractError(f"missing row dataset: {error.args[0]}") from error
            try:
                row_values = {
                    name: np.asarray(group[name])
                    for name in (
                        "observation",
                        "observation_timestamp",
                        "valid",
                        "teacher_requested",
                        "teacher_applied",
                        "applied_offset",
                        "timestep",
                    )
                }
            except KeyError as error:
                raise TeacherDatasetContractError(f"missing row dataset: {error.args[0]}") from error
            normalized = _normalize_rows(
                TeacherEpisodeRows(
                    observation=row_values["observation"],
                    observation_timestamp=row_values["observation_timestamp"],
                    valid=row_values["valid"],
                    teacher_requested=row_values["teacher_requested"],
                    teacher_applied=row_values["teacher_applied"],
                    applied_offset=row_values["applied_offset"],
                    phase=phase,
                    timestep=row_values["timestep"],
                ),
                provenance,
            )
            validated.append((episode_id, group, provenance, terminal, normalized))

        for episode_id, _group, provenance, terminal, normalized in validated:
            if provenance.split != split:
                continue
            if successful_only and not terminal.independent_success:
                continue
            if valid_only and not terminal.valid:
                continue
            mask = normalized["valid"] if valid_only else np.ones_like(normalized["valid"])
            if valid_only and not bool(mask.any()):
                continue
            episode = {
                "episode_id": episode_id,
                "observation": normalized["observation"][mask],
                "action": normalized["teacher_applied"][mask],
                "observation_timestamp": normalized["observation_timestamp"][mask],
                "valid": normalized["valid"][mask],
                "phase": normalized["phase"][mask],
                "timestep": normalized["timestep"][mask],
                "metadata": {
                    "grasp_id": provenance.grasp_id,
                    "bias_case_id": provenance.bias_case_id,
                    "split": provenance.split,
                    "seed": provenance.seed,
                    **{name: getattr(provenance, name) for name in _PROVENANCE_SHA_FIELDS},
                },
            }
            if set(episode) != _STUDENT_KEYS:
                raise AssertionError("student loader key allowlist changed")
            loaded.append(episode)
    return loaded


def _required_attr(group: h5py.Group, name: str) -> object:
    if name not in group.attrs:
        raise TeacherDatasetContractError(f"missing episode attribute: {name}")
    return group.attrs[name]


def _provenance_from_group(group: h5py.Group) -> TeacherEpisodeProvenance:
    try:
        seed_value = _required_attr(group, "seed")
        seed = int(seed_value)
        if isinstance(seed_value, (bool, np.bool_)) or seed != seed_value:
            raise ValueError
        provenance = TeacherEpisodeProvenance(
            grasp_id=str(_required_attr(group, "grasp_id")),
            split=str(_required_attr(group, "split")),
            bias_case_id=str(_required_attr(group, "bias_case_id")),
            seed=seed,
            source_sha256=str(_required_attr(group, "source_sha256")),
            task_sha256=str(_required_attr(group, "task_sha256")),
            training_sha256=str(_required_attr(group, "training_sha256")),
            bank_sha256=str(_required_attr(group, "bank_sha256")),
            translation_bound_m=float(_required_attr(group, "translation_bound_m")),
            rotation_bound_rad=float(_required_attr(group, "rotation_bound_rad")),
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise TeacherDatasetContractError("episode provenance attributes have invalid types") from error
    provenance.validate()
    return provenance


def _terminal_from_group(group: h5py.Group) -> TeacherEpisodeTerminal:
    success = _required_attr(group, "independent_success")
    valid = _required_attr(group, "valid")
    failure_reason = _required_attr(group, "failure_reason")
    if not isinstance(success, (bool, np.bool_)) or not isinstance(valid, (bool, np.bool_)):
        raise TeacherDatasetContractError("terminal boolean attributes must be stored as booleans")
    if not isinstance(failure_reason, str):
        raise TeacherDatasetContractError("failure_reason must be a string")
    terminal = TeacherEpisodeTerminal(bool(success), bool(valid), failure_reason)
    terminal.validate()
    return terminal


def _validate_split_pair(
    registry: dict[object, str], key: object, split: str, name: str
) -> None:
    previous = registry.get(key)
    if previous is not None and previous != split:
        raise TeacherDatasetContractError(f"{name} must not cross train/dev splits")
    registry[key] = split

