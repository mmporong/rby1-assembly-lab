"""Pure causal episode collection for parallel RJ45 teacher rollouts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


class TeacherCollectionError(ValueError):
    """Teacher rollout telemetry violates the collection contract."""


@dataclass(frozen=True)
class CollectedTeacherEpisode:
    env_id: int
    episode_index: int
    grasp_id: str
    bias_case_id: str
    seed: int
    observation: np.ndarray
    observation_timestamp: np.ndarray
    policy_observation_timestamp: np.ndarray
    valid: np.ndarray
    teacher_requested: np.ndarray
    teacher_applied: np.ndarray
    applied_offset: np.ndarray
    phase: tuple[str, ...]
    teacher_gt_pose: np.ndarray
    independent_success: bool
    terminal_valid: bool
    failure_reason: str


def fixed_environment_episode_quotas(total_episodes: int, num_envs: int) -> tuple[int, ...]:
    if type(total_episodes) is not int or total_episodes <= 0:
        raise TeacherCollectionError("total_episodes must be a positive integer")
    if type(num_envs) is not int or num_envs <= 0:
        raise TeacherCollectionError("num_envs must be a positive integer")
    quotient, remainder = divmod(total_episodes, num_envs)
    return tuple(quotient + int(index < remainder) for index in range(num_envs))


def _row(value: Any, env_id: int, name: str) -> np.ndarray:
    if not hasattr(value, "detach"):
        raise TeacherCollectionError(f"snapshot {name} must be a tensor")
    return np.asarray(value[env_id].detach().cpu())


def _scalar(value: Any, env_id: int, name: str) -> object:
    row = _row(value, env_id, name)
    if row.size != 1:
        raise TeacherCollectionError(f"snapshot {name} must be scalar per environment")
    return row.item()


def _phase_name(snapshot: Mapping[str, Any], env_id: int) -> str:
    code = _scalar(snapshot["phase"], env_id, "phase")
    if type(code) is not int:
        raise TeacherCollectionError("phase code must be an integer")
    names = snapshot.get("phase_name_by_code")
    if not isinstance(names, Mapping) or not isinstance(names.get(code), str):
        raise TeacherCollectionError("phase_name_by_code must identify the phase")
    return names[code]


class ParallelTeacherCollector:
    """Collect a fixed, outcome-independent episode quota from every environment."""

    def __init__(
        self,
        *,
        num_envs: int,
        total_episodes: int,
        grasp_ids: tuple[str, ...],
        episode_seed: Callable[[int, int], int],
    ) -> None:
        self.quotas = fixed_environment_episode_quotas(total_episodes, num_envs)
        if not grasp_ids or any(not isinstance(value, str) or not value for value in grasp_ids):
            raise TeacherCollectionError("grasp_ids must be nonempty strings")
        self._grasp_ids = grasp_ids
        self._episode_seed = episode_seed
        self._accepted = [0] * num_envs
        self._identity: list[tuple[int, int] | None] = [None] * num_envs
        self._rows: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]
        self.episodes: list[CollectedTeacherEpisode] = []

    @property
    def complete(self) -> bool:
        return tuple(self._accepted) == self.quotas

    @property
    def accepted_per_environment(self) -> tuple[int, ...]:
        return tuple(self._accepted)

    def consume_step(
        self,
        *,
        live_snapshot: Mapping[str, Any],
        terminal_snapshot: Mapping[str, Any] | None,
        ended: Any,
        pre_step_time_s: float,
    ) -> None:
        if not math.isfinite(pre_step_time_s) or pre_step_time_s < 0.0:
            raise TeacherCollectionError("pre_step_time_s must be finite and non-negative")
        ended_array = np.asarray(ended.detach().cpu() if hasattr(ended, "detach") else ended)
        if ended_array.shape != (len(self.quotas),) or ended_array.dtype.kind != "b":
            raise TeacherCollectionError("ended must be boolean with one row per environment")
        if bool(ended_array.any()) and terminal_snapshot is None:
            raise TeacherCollectionError("terminal_snapshot is required for ended environments")

        for env_id in range(len(self.quotas)):
            if self._accepted[env_id] >= self.quotas[env_id]:
                self._rows[env_id].clear()
                self._identity[env_id] = None
                continue
            snapshot = terminal_snapshot if ended_array[env_id] else live_snapshot
            assert snapshot is not None
            identity = self._snapshot_identity(snapshot, env_id)
            previous = self._identity[env_id]
            if previous is None:
                self._identity[env_id] = identity
            elif previous != identity:
                raise TeacherCollectionError(
                    f"environment {env_id} episode/grasp identity changed before terminal"
                )
            rows = self._rows[env_id]
            if rows and pre_step_time_s <= rows[-1]["timestamp"]:
                raise TeacherCollectionError("pre-step timestamps must be strictly increasing")
            rows.append(self._extract_row(snapshot, env_id, pre_step_time_s))
            if ended_array[env_id]:
                self.episodes.append(self._finish_episode(snapshot, env_id, identity, rows))
                self._accepted[env_id] += 1
                self._rows[env_id] = []
                self._identity[env_id] = None

    def _snapshot_identity(self, snapshot: Mapping[str, Any], env_id: int) -> tuple[int, int]:
        episode_index = _scalar(snapshot["episode_index"], env_id, "episode_index")
        grasp_index = _scalar(snapshot["held_grasp_index"], env_id, "held_grasp_index")
        if type(episode_index) is not int or episode_index < 0:
            raise TeacherCollectionError("episode_index must be a non-negative integer")
        if type(grasp_index) is not int or not 0 <= grasp_index < len(self._grasp_ids):
            raise TeacherCollectionError("held_grasp_index is outside the verified bank")
        return episode_index, grasp_index

    def _extract_row(
        self, snapshot: Mapping[str, Any], env_id: int, timestamp: float
    ) -> dict[str, Any]:
        observation = _row(snapshot["policy_input_observation"], env_id, "policy_input_observation")
        requested = _row(
            snapshot["teacher_requested_raw_action_6d"], env_id, "teacher_requested_raw_action_6d"
        )
        applied = _row(
            snapshot["teacher_bounded_action_6d"], env_id, "teacher_bounded_action_6d"
        )
        offset = _row(
            snapshot["teacher_physical_offset_entry_6d"],
            env_id,
            "teacher_physical_offset_entry_6d",
        )
        gt_pose = _row(
            snapshot["teacher_virtual_target_body_pose_world_wxyz"],
            env_id,
            "teacher_virtual_target_body_pose_world_wxyz",
        )
        policy_timestamp = float(
            _scalar(snapshot["policy_input_timestamp_s"], env_id, "policy_input_timestamp_s")
        )
        policy_valid = _scalar(snapshot["policy_input_valid"], env_id, "policy_input_valid")
        teacher_active = _scalar(snapshot["teacher_active"], env_id, "teacher_active")
        if type(policy_valid) is not bool or type(teacher_active) is not bool:
            raise TeacherCollectionError("policy_input_valid and teacher_active must be boolean")
        arrays = (observation, requested, applied, offset, gt_pose)
        finite = all(np.isfinite(value).all() for value in arrays)
        return {
            "observation": observation.astype(np.float32),
            "timestamp": timestamp,
            "policy_timestamp": policy_timestamp,
            "valid": bool(policy_valid and teacher_active and finite),
            "requested": requested.astype(np.float32),
            "applied": applied.astype(np.float32),
            "offset": offset.astype(np.float32),
            "phase": _phase_name(snapshot, env_id),
            "gt_pose": gt_pose.astype(np.float32),
        }

    def _finish_episode(
        self,
        snapshot: Mapping[str, Any],
        env_id: int,
        identity: tuple[int, int],
        rows: list[dict[str, Any]],
    ) -> CollectedTeacherEpisode:
        success = _scalar(snapshot["independent_eval_success"], env_id, "independent_eval_success")
        evaluator_valid = _scalar(snapshot["independent_eval_valid"], env_id, "independent_eval_valid")
        held_valid = _scalar(snapshot["held_initialization_passed"], env_id, "held_initialization_passed")
        valid_steps = _scalar(
            snapshot["valid_policy_input_step_count"], env_id, "valid_policy_input_step_count"
        )
        if any(type(value) is not bool for value in (success, evaluator_valid, held_valid)):
            raise TeacherCollectionError("terminal evaluator flags must be boolean")
        if type(valid_steps) is not int or valid_steps < 0:
            raise TeacherCollectionError("valid_policy_input_step_count must be non-negative")
        finite_terminal = all(
            np.isfinite(_row(snapshot[name], env_id, name)).all()
            for name in (
                "actual_depth_m",
                "actual_lateral_error_m",
                "actual_orientation_error_rad",
            )
        )
        terminal_valid = bool(evaluator_valid and held_valid and valid_steps > 0 and finite_terminal)
        reason_code = _scalar(snapshot["termination_reason_code"], env_id, "termination_reason_code")
        reasons = snapshot.get("termination_reason_by_code")
        reason = reasons.get(reason_code) if isinstance(reasons, Mapping) else None
        if not isinstance(reason, str) or not reason:
            raise TeacherCollectionError("terminal reason code is not mapped")
        if success and terminal_valid:
            failure_reason = ""
        elif not terminal_valid:
            failure_reason = (
                "invalid_evaluator" if not evaluator_valid else
                "held_initialization_not_passed" if not held_valid else
                "no_valid_policy_input_exposure" if valid_steps == 0 else
                "nonfinite_terminal_metrics"
            )
        else:
            failure_reason = reason
        episode_index, grasp_index = identity
        seed = self._episode_seed(env_id, episode_index)
        if type(seed) is not int or seed < 0:
            raise TeacherCollectionError("episode seed provider must return a non-negative integer")
        return CollectedTeacherEpisode(
            env_id=env_id,
            episode_index=episode_index,
            grasp_id=self._grasp_ids[grasp_index],
            bias_case_id=f"env-{env_id}-episode-{episode_index}",
            seed=seed,
            observation=np.stack([row["observation"] for row in rows]),
            observation_timestamp=np.asarray([row["timestamp"] for row in rows], dtype=np.float64),
            policy_observation_timestamp=np.asarray(
                [row["policy_timestamp"] for row in rows], dtype=np.float64
            ),
            valid=np.asarray([row["valid"] for row in rows], dtype=np.bool_),
            teacher_requested=np.stack([row["requested"] for row in rows]),
            teacher_applied=np.stack([row["applied"] for row in rows]),
            applied_offset=np.stack([row["offset"] for row in rows]),
            phase=tuple(row["phase"] for row in rows),
            teacher_gt_pose=np.stack([row["gt_pose"] for row in rows]),
            independent_success=bool(success and terminal_valid),
            terminal_valid=terminal_valid,
            failure_reason=failure_reason,
        )


def write_collected_episodes(
    path: str | Path,
    episodes: list[CollectedTeacherEpisode],
    *,
    split: str,
    source_sha256: str,
    task_sha256: str,
    training_sha256: str,
    bank_sha256: str,
    translation_bound_m: float,
    rotation_bound_rad: float,
) -> None:
    """Publish completed attempts through the atomic dataset writer."""

    from .rj45_teacher_dataset import (
        TeacherDatasetWriter,
        TeacherEpisodeProvenance,
        TeacherEpisodeRows,
        TeacherEpisodeTerminal,
    )

    with TeacherDatasetWriter(path) as writer:
        for index, episode in enumerate(episodes):
            provenance = TeacherEpisodeProvenance(
                grasp_id=episode.grasp_id,
                split=split,
                bias_case_id=episode.bias_case_id,
                seed=episode.seed,
                source_sha256=source_sha256,
                task_sha256=task_sha256,
                training_sha256=training_sha256,
                bank_sha256=bank_sha256,
                translation_bound_m=translation_bound_m,
                rotation_bound_rad=rotation_bound_rad,
            )
            rows = TeacherEpisodeRows(
                observation=episode.observation,
                observation_timestamp=episode.observation_timestamp,
                policy_observation_timestamp=episode.policy_observation_timestamp,
                valid=episode.valid,
                teacher_requested=episode.teacher_requested,
                teacher_applied=episode.teacher_applied,
                applied_offset=episode.applied_offset,
                phase=episode.phase,
                timestep=np.arange(len(episode.phase), dtype=np.int64),
                teacher_gt_pose=episode.teacher_gt_pose,
            )
            terminal = TeacherEpisodeTerminal(
                independent_success=episode.independent_success,
                valid=episode.terminal_valid,
                failure_reason=episode.failure_reason,
            )
            writer.add_episode(
                f"{split}-env{episode.env_id}-episode{episode.episode_index}-{index}",
                provenance,
                rows,
                terminal,
            )
