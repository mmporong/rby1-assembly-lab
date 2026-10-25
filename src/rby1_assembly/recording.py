"""Isaac Lab 한 환경의 baseline transition을 HDF5 episode로 연결한다."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import (
    DatasetContractError,
    EpisodeDatasetWriter,
    EpisodeMetadata,
    EpisodeProvenance,
    GroundTruthMetrics,
    TerminalEvaluation,
    TimestepRecord,
)
from .contracts import EvaluationSample, TaskLimits
from .evaluation import EvaluationResult, TaskEvaluator


PHASE_NAMES = (
    "pregrasp",
    "close",
    "check_lift",
    "align",
    "insert",
    "retract",
    "done",
    "abort",
)
INDEPENDENT_EVALUATOR_SOURCE = "simulator_ground_truth_independent"
INSERTION_FORCE_SOURCE = "plug_body_filtered_against_socket_net_normal_force_world_frame"
INSERTION_LOAD_SOURCE = "physx_plug_socket_sum_abs_contact_normals"
FORBIDDEN_CONTACT_LOAD_SOURCE = (
    "physx_rigid_contact_view_filtered_pairs_sum_abs_normal_forces"
)


class RecordingContractError(RuntimeError):
    """실제 물리 transition을 손실 없이 기록할 수 없을 때 발생한다."""


@dataclass(frozen=True)
class AssemblyGeometry:
    socket_entry_y_m: float
    plug_tip_y_m: float
    metric_tolerance_m: float = 1e-5
    angular_tolerance_rad: float = 1e-5


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _env_value(value: Any, env_index: int) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == 0:
        return array
    return array[env_index]


def _scalar(value: Any, env_index: int = 0) -> Any:
    array = _numpy(value)
    if array.ndim > 0:
        array = array[env_index]
    return array.item() if np.asarray(array).ndim == 0 else array


def wxyz_to_xyzw(quaternion: Any) -> np.ndarray:
    """Isaac/USD의 ``wxyz`` quaternion을 dataset의 ``xyzw``로 바꾼다."""

    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape[-1] != 4:
        raise RecordingContractError(f"quaternion last dimension must be 4, got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise RecordingContractError("quaternion must contain only finite values")
    norm = np.linalg.norm(value, axis=-1)
    if not np.allclose(norm, 1.0, atol=1e-3, rtol=0.0):
        raise RecordingContractError("Isaac/USD quaternion must be normalized")
    return np.concatenate((value[..., 1:4], value[..., 0:1]), axis=-1)


@dataclass(frozen=True)
class PhysicsSnapshot:
    joint_position_rad: np.ndarray
    joint_velocity_rad_s: np.ndarray
    gripper_position_m: np.ndarray
    gripper_velocity_m_s: np.ndarray
    object_poses_world_xyzw: np.ndarray
    contact_force_proxy_n: np.ndarray
    grip_contact_force_peak_n: float
    insertion_contact_force_n: float
    insertion_contact_force_w: np.ndarray
    insertion_contact_force_peak_n: float
    insertion_contact_load_n: float
    insertion_contact_load_peak_n: float
    forbidden_contact_load_n: float
    forbidden_contact_load_peak_n: float
    phase: str
    ground_truth: GroundTruthMetrics
    reported_geometry_consistent: bool


def _phase_name(value: Any) -> str:
    phase_index = int(np.asarray(value).item())
    if not 0 <= phase_index < len(PHASE_NAMES):
        raise RecordingContractError(f"unknown assembly phase: {phase_index}")
    return PHASE_NAMES[phase_index]


def _quat_apply_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:4]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)


def _quat_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def geometry_metrics_from_object_poses(
    object_poses_world_wxyz: Any, geometry: AssemblyGeometry
) -> tuple[float, float, float]:
    """원시 pose에서 socket entry 기준 plug tip의 기하 지표를 계산한다."""

    poses = np.asarray(object_poses_world_wxyz, dtype=np.float64)
    if poses.shape != (2, 7):
        raise RecordingContractError(f"object poses must have shape (2, 7), got {poses.shape}")
    socket_position, plug_position = poses[0, :3], poses[1, :3]
    socket_quaternion, plug_quaternion = poses[0, 3:7], poses[1, 3:7]
    wxyz_to_xyzw(np.stack((socket_quaternion, plug_quaternion)))
    socket_entry_world = socket_position + _quat_apply_wxyz(
        socket_quaternion, np.asarray((0.0, geometry.socket_entry_y_m, 0.0))
    )
    plug_tip_world = plug_position + _quat_apply_wxyz(
        plug_quaternion, np.asarray((0.0, geometry.plug_tip_y_m, 0.0))
    )
    socket_inverse = socket_quaternion * np.asarray((1.0, -1.0, -1.0, -1.0))
    relative_position = _quat_apply_wxyz(socket_inverse, plug_tip_world - socket_entry_world)
    relative_quaternion = _quat_multiply_wxyz(socket_inverse, plug_quaternion)
    relative_quaternion /= np.linalg.norm(relative_quaternion)
    angle = 2.0 * np.arctan2(
        np.linalg.norm(relative_quaternion[1:4]), abs(relative_quaternion[0])
    )
    return (
        float(relative_position[1]),
        float(np.linalg.norm(relative_position[(0, 2),])),
        float(angle),
    )


def _metrics_from_mapping(
    data: Mapping[str, Any],
    object_poses_world_wxyz: np.ndarray,
    geometry: AssemblyGeometry,
    env_index: int,
) -> tuple[GroundTruthMetrics, bool]:
    def field(*names: str) -> Any:
        for name in names:
            if name in data:
                return _scalar(data[name], env_index)
        raise RecordingContractError(f"ground-truth field is missing: {' or '.join(names)}")

    reported = (
        float(field("actual_depth_m")),
        float(field("lateral_error_m", "actual_lateral_error_m")),
        float(field("orientation_error_rad", "actual_orientation_error_rad")),
    )
    recomputed = geometry_metrics_from_object_poses(object_poses_world_wxyz, geometry)
    consistent = (
        abs(reported[0] - recomputed[0]) <= geometry.metric_tolerance_m
        and abs(reported[1] - recomputed[1]) <= geometry.metric_tolerance_m
        and abs(reported[2] - recomputed[2]) <= geometry.angular_tolerance_rad
    )
    return GroundTruthMetrics(
        actual_depth_m=recomputed[0],
        lateral_error_m=recomputed[1],
        orientation_error_rad=recomputed[2],
        force_peak_n=float(field("force_peak_n", "contact_force_proxy_peak_n")),
        grasp_both=bool(field("grasp_both")),
        forbidden_contact=bool(field("forbidden_contact")),
    ), consistent


def _poses_wxyz_to_xyzw(value: Any, env_index: int) -> np.ndarray:
    poses = np.asarray(_env_value(value, env_index), dtype=np.float64)
    if poses.shape != (2, 7):
        raise RecordingContractError(f"object poses must have shape (2, 7), got {poses.shape}")
    result = poses.copy()
    result[:, 3:7] = wxyz_to_xyzw(poses[:, 3:7])
    return result


def terminal_snapshot_from_extras(
    extras: Mapping[str, Any], geometry: AssemblyGeometry, env_index: int = 0
) -> PhysicsSnapshot:
    """자동 reset 전에 환경이 복제한 종료 상태만 읽는다.

    반환된 환경 속성을 대신 읽는 fallback은 두지 않는다. Isaac Lab은 종료된
    환경을 ``step`` 내부에서 reset하므로 fallback은 다음 episode 상태를 섞는다.
    """

    snapshots = extras.get("terminal_snapshot")
    if not isinstance(snapshots, Mapping):
        raise RecordingContractError(
            "terminal_snapshot is required because DirectRLEnv resets before step returns"
        )
    evaluation = snapshots.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise RecordingContractError("terminal_snapshot.evaluation is required")
    poses_wxyz = np.asarray(
        _env_value(snapshots["object_poses_world_wxyz"], env_index), dtype=np.float64
    )
    metrics, geometry_consistent = _metrics_from_mapping(
        evaluation, poses_wxyz, geometry, env_index
    )
    required_contact_fields = (
        "grip_contact_force_peak_n",
        "insertion_contact_force_n",
        "insertion_contact_force_w",
        "insertion_contact_force_peak_n",
        "insertion_contact_force_source",
        "insertion_contact_load_n",
        "insertion_contact_load_peak_n",
        "insertion_contact_load_source",
        "forbidden_contact_load_n",
        "forbidden_contact_load_peak_n",
        "forbidden_contact_load_source",
    )
    missing = [name for name in required_contact_fields if name not in snapshots]
    if missing:
        raise RecordingContractError(
            "terminal snapshot is missing measured contact fields: " + ", ".join(missing)
        )
    if snapshots["insertion_contact_force_source"] != INSERTION_FORCE_SOURCE:
        raise RecordingContractError("terminal insertion contact-force source is unsupported")
    if snapshots["insertion_contact_load_source"] != INSERTION_LOAD_SOURCE:
        raise RecordingContractError("terminal insertion contact-load source is unsupported")
    if snapshots["forbidden_contact_load_source"] != FORBIDDEN_CONTACT_LOAD_SOURCE:
        raise RecordingContractError("terminal forbidden contact-load source is unsupported")
    snapshot = PhysicsSnapshot(
        joint_position_rad=np.asarray(
            _env_value(snapshots["revolute_joint_positions_rad"], env_index), dtype=np.float64
        ),
        joint_velocity_rad_s=np.asarray(
            _env_value(snapshots["revolute_joint_velocities_rad_s"], env_index), dtype=np.float64
        ),
        gripper_position_m=np.asarray(
            _env_value(snapshots["gripper_positions_m"], env_index), dtype=np.float64
        ),
        gripper_velocity_m_s=np.asarray(
            _env_value(snapshots["gripper_velocities_m_s"], env_index), dtype=np.float64
        ),
        object_poses_world_xyzw=_poses_wxyz_to_xyzw(poses_wxyz[None, ...], 0),
        contact_force_proxy_n=np.asarray(
            _env_value(snapshots["contact_force_proxy_n"], env_index), dtype=np.float64
        ),
        grip_contact_force_peak_n=float(
            _scalar(snapshots["grip_contact_force_peak_n"], env_index)
        ),
        insertion_contact_force_n=float(
            _scalar(snapshots["insertion_contact_force_n"], env_index)
        ),
        insertion_contact_force_w=np.asarray(
            _env_value(snapshots["insertion_contact_force_w"], env_index), dtype=np.float64
        ),
        insertion_contact_force_peak_n=float(
            _scalar(snapshots["insertion_contact_force_peak_n"], env_index)
        ),
        insertion_contact_load_n=float(
            _scalar(snapshots["insertion_contact_load_n"], env_index)
        ),
        insertion_contact_load_peak_n=float(
            _scalar(snapshots["insertion_contact_load_peak_n"], env_index)
        ),
        forbidden_contact_load_n=float(
            _scalar(snapshots["forbidden_contact_load_n"], env_index)
        ),
        forbidden_contact_load_peak_n=float(
            _scalar(snapshots["forbidden_contact_load_peak_n"], env_index)
        ),
        phase=_phase_name(_env_value(snapshots["phase"], env_index)),
        ground_truth=metrics,
        reported_geometry_consistent=geometry_consistent,
    )
    _validate_contact_loads(snapshot)
    return snapshot


def live_snapshot(
    env: Any, extras: Mapping[str, Any], geometry: AssemblyGeometry, env_index: int = 0
) -> PhysicsSnapshot:
    """종료되지 않은 step 직후 PhysX-backed 환경 상태를 복제한다."""

    evaluation = extras.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise RecordingContractError("extras.evaluation is required for ground-truth logging")
    if extras.get("insertion_contact_force_source") != INSERTION_FORCE_SOURCE:
        raise RecordingContractError("live insertion contact-force source is missing or unsupported")
    if extras.get("insertion_contact_load_source") != INSERTION_LOAD_SOURCE:
        raise RecordingContractError("live insertion contact-load source is missing or unsupported")
    if extras.get("forbidden_contact_load_source") != FORBIDDEN_CONTACT_LOAD_SOURCE:
        raise RecordingContractError("live forbidden contact-load source is missing or unsupported")
    root_poses_wxyz = np.stack(
        (
            np.concatenate(
                (
                    _env_value(env.socket.data.root_pos_w, env_index),
                    _env_value(env.socket.data.root_quat_w, env_index),
                )
            ),
            np.concatenate(
                (
                    _env_value(env.plug.data.root_pos_w, env_index),
                    _env_value(env.plug.data.root_quat_w, env_index),
                )
            ),
        )
    )
    extended_evaluation = dict(evaluation)
    extended_evaluation.setdefault("grasp_both", bool(np.all(_env_value(env._grip_proven, env_index))))
    if "forbidden_contact" not in extended_evaluation:
        raise RecordingContractError(
            "extras.evaluation.forbidden_contact is required; overload is not a forbidden-contact proxy"
        )
    revolute_ids = np.concatenate((_numpy(env._torso_joint_ids), _numpy(env._arm_joint_ids))).astype(int)
    gripper_ids = _numpy(env._gripper_joint_ids).astype(int)
    joint_position = np.asarray(_env_value(env.robot.data.joint_pos, env_index), dtype=np.float64)
    joint_velocity = np.asarray(_env_value(env.robot.data.joint_vel, env_index), dtype=np.float64)
    metrics, geometry_consistent = _metrics_from_mapping(
        extended_evaluation, root_poses_wxyz, geometry, env_index=0
    )
    required_env_contact_fields = (
        "_episode_grip_force_peak_n",
        "_last_insertion_contact_force_n",
        "_last_insertion_contact_force_w",
        "_episode_insertion_contact_force_peak_n",
        "_last_insertion_contact_load_n",
        "_episode_insertion_contact_load_peak_n",
        "_last_forbidden_contact_load_n",
        "_episode_forbidden_contact_load_peak_n",
    )
    missing = [name for name in required_env_contact_fields if not hasattr(env, name)]
    if missing:
        raise RecordingContractError(
            "environment is missing measured contact fields: " + ", ".join(missing)
        )
    snapshot = PhysicsSnapshot(
        joint_position_rad=joint_position[revolute_ids],
        joint_velocity_rad_s=joint_velocity[revolute_ids],
        gripper_position_m=joint_position[gripper_ids],
        gripper_velocity_m_s=joint_velocity[gripper_ids],
        object_poses_world_xyzw=_poses_wxyz_to_xyzw(root_poses_wxyz[None, ...], 0),
        contact_force_proxy_n=np.asarray(
            _env_value(env._last_contact_proxy_n, env_index), dtype=np.float64
        ),
        grip_contact_force_peak_n=float(
            _scalar(env._episode_grip_force_peak_n, env_index)
        ),
        insertion_contact_force_n=float(
            _scalar(env._last_insertion_contact_force_n, env_index)
        ),
        insertion_contact_force_w=np.asarray(
            _env_value(env._last_insertion_contact_force_w, env_index), dtype=np.float64
        ),
        insertion_contact_force_peak_n=float(
            _scalar(env._episode_insertion_contact_force_peak_n, env_index)
        ),
        insertion_contact_load_n=float(
            _scalar(env._last_insertion_contact_load_n, env_index)
        ),
        insertion_contact_load_peak_n=float(
            _scalar(env._episode_insertion_contact_load_peak_n, env_index)
        ),
        forbidden_contact_load_n=float(
            _scalar(env._last_forbidden_contact_load_n, env_index)
        ),
        forbidden_contact_load_peak_n=float(
            _scalar(env._episode_forbidden_contact_load_peak_n, env_index)
        ),
        phase=_phase_name(_env_value(env._phase, env_index)),
        ground_truth=metrics,
        reported_geometry_consistent=geometry_consistent,
    )
    _validate_contact_loads(snapshot)
    return snapshot


def _validate_contact_loads(snapshot: PhysicsSnapshot) -> None:
    net_norm = float(np.linalg.norm(snapshot.insertion_contact_force_w))
    if not np.isclose(net_norm, snapshot.insertion_contact_force_n, atol=1e-5, rtol=1e-5):
        raise RecordingContractError("insertion net-force norm does not match its vector")
    if snapshot.insertion_contact_load_n + 1e-4 < snapshot.insertion_contact_force_n:
        raise RecordingContractError("insertion contact load is below the net-force norm")
    if snapshot.insertion_contact_load_peak_n + 1e-9 < snapshot.insertion_contact_load_n:
        raise RecordingContractError("insertion contact load peak is below the current load")
    if snapshot.insertion_contact_load_peak_n + 1e-4 < snapshot.insertion_contact_force_peak_n:
        raise RecordingContractError("insertion contact load peak is below the net-force peak")
    if snapshot.forbidden_contact_load_peak_n + 1e-9 < snapshot.forbidden_contact_load_n:
        raise RecordingContractError("forbidden contact load peak is below the current load")
    combined_peak = max(
        snapshot.grip_contact_force_peak_n, snapshot.insertion_contact_load_peak_n
    )
    if not np.isclose(
        snapshot.ground_truth.force_peak_n, combined_peak, atol=1e-4, rtol=1e-5
    ):
        raise RecordingContractError(
            "ground-truth force peak does not match grip/insertion measured peaks"
        )


def terminal_evaluation_from_extras(
    extras: Mapping[str, Any], snapshot: PhysicsSnapshot, terminal_timestamp_s: float, env_index: int = 0
) -> TerminalEvaluation:
    """독립 evaluator 증거가 완전할 때만 유효한 성공/실패 판정을 만든다."""

    raw = extras.get("terminal_evaluation")
    if not isinstance(raw, Mapping):
        return _invalid_terminal(snapshot, terminal_timestamp_s, "independent_evaluator_missing")
    source = raw.get("evaluator_source")
    version = raw.get("evaluator_version")
    criteria_raw = raw.get("criteria")
    if source != INDEPENDENT_EVALUATOR_SOURCE or not version or not isinstance(criteria_raw, Mapping):
        return _invalid_terminal(snapshot, terminal_timestamp_s, "independent_evaluator_incomplete")
    criteria = {name: bool(_scalar(value, env_index)) for name, value in criteria_raw.items()}
    valid = bool(_scalar(raw.get("valid", False), env_index))
    success = bool(_scalar(raw.get("success", False), env_index)) if valid else False
    reason_value = raw.get("termination_reason", "unknown")
    reason = str(_scalar(reason_value, env_index))
    return TerminalEvaluation(
        success=success,
        valid=valid,
        termination_reason=reason,
        evaluator_source=INDEPENDENT_EVALUATOR_SOURCE,
        evaluator_version=str(version),
        terminal_timestamp_s=terminal_timestamp_s,
        metrics=snapshot.ground_truth,
        criteria=criteria,
    )


def _invalid_terminal(
    snapshot: PhysicsSnapshot, terminal_timestamp_s: float, reason: str
) -> TerminalEvaluation:
    return TerminalEvaluation(
        success=False,
        valid=False,
        termination_reason=reason,
        evaluator_source=INDEPENDENT_EVALUATOR_SOURCE,
        evaluator_version="unavailable",
        terminal_timestamp_s=terminal_timestamp_s,
        metrics=snapshot.ground_truth,
        criteria={"independent_evaluator_available": False},
    )


def reconcile_terminal_evaluation(
    *,
    extras: Mapping[str, Any],
    snapshot: PhysicsSnapshot,
    terminal_timestamp_s: float,
    independent_result: EvaluationResult,
    task_limits: TaskLimits,
    env_index: int = 0,
) -> TerminalEvaluation:
    """환경 판정과 별도 ``TaskEvaluator`` 판정을 대조해 fail-closed한다."""

    try:
        environment_result = terminal_evaluation_from_extras(
            extras, snapshot, terminal_timestamp_s, env_index
        )
    except (DatasetContractError, KeyError, TypeError, ValueError):
        return _invalid_terminal(snapshot, terminal_timestamp_s, "environment_evaluator_malformed")
    if not snapshot.reported_geometry_consistent:
        return _invalid_terminal(snapshot, terminal_timestamp_s, "geometry_metric_mismatch")
    agreement = (
        independent_result.terminal
        and environment_result.valid
        and environment_result.success == independent_result.success
    )
    valid_failure_reasons = {
        "hard_force_limit",
        "grasp_lost",
        "pregrasp_timeout",
        "retry_limit",
        "episode_timeout",
        "forbidden_contact",
    }
    measured_failure = (
        environment_result.valid
        and not environment_result.success
        and not independent_result.success
        and environment_result.termination_reason in valid_failure_reasons
    )
    if environment_result.termination_reason == "episode_timeout":
        measured_failure = measured_failure and (
            terminal_timestamp_s + 1e-9 >= task_limits.timeout_s
        )
    if not agreement and not measured_failure:
        return _invalid_terminal(snapshot, terminal_timestamp_s, "evaluator_disagreement")
    criteria = dict(environment_result.criteria)
    criteria["independent_evaluator_terminal"] = independent_result.terminal
    if agreement and independent_result.success:
        reason = "independent_success"
        success = True
    elif measured_failure:
        reason = environment_result.termination_reason
        success = False
    else:
        reason = "+".join(independent_result.reasons) or environment_result.termination_reason
        success = False
    return TerminalEvaluation(
        success=success,
        valid=True,
        termination_reason=reason,
        evaluator_source=INDEPENDENT_EVALUATOR_SOURCE,
        evaluator_version="task-evaluator-v1+environment-cross-check",
        terminal_timestamp_s=terminal_timestamp_s,
        metrics=snapshot.ground_truth,
        criteria=criteria,
    )


def record_baseline_rollouts(
    *,
    env: Any,
    writer: EpisodeDatasetWriter,
    provenance: EpisodeProvenance,
    episode_count: int,
    max_steps: int,
    base_seed: int,
    action_factory: Any,
    task_limits: TaskLimits,
    geometry: AssemblyGeometry,
) -> list[dict[str, Any]]:
    """한 환경에서 zero-residual baseline episode를 순차 기록한다.

    각 row는 ``observation_t, action_t, physics_state_t+1`` transition이다.
    종료 step은 반드시 reset 전 ``terminal_snapshot``을 사용한다.
    """

    if env.num_envs != 1:
        raise RecordingContractError("baseline recorder requires exactly one environment")
    if episode_count <= 0 or max_steps <= 0:
        raise RecordingContractError("episode_count and max_steps must be positive")
    reports: list[dict[str, Any]] = []
    for episode_index in range(episode_count):
        seed = base_seed + episode_index
        observation_dict, _ = env.reset(seed=seed)
        observation = observation_dict["policy"]
        metadata = EpisodeMetadata.for_seed(
            episode_id=f"baseline-seed-{seed}-episode-{episode_index:04d}",
            seed=seed,
            rollout_index=episode_index,
            provenance=provenance,
        )
        writer.begin_episode(metadata)
        try:
            evaluator = TaskEvaluator(task_limits)
            final_evaluation: TerminalEvaluation | None = None
            steps_recorded = 0
            for step_index in range(max_steps):
                action = action_factory(env)
                next_observation, _, terminated, truncated, extras = env.step(action)
                done = bool(_scalar(terminated | truncated))
                snapshot = (
                    terminal_snapshot_from_extras(extras, geometry)
                    if done
                    else live_snapshot(env, extras, geometry)
                )
                timestamp_s = (step_index + 1) * float(env.step_dt)
                writer.append(
                    TimestepRecord(
                        timestamp_s=timestamp_s,
                        observation=np.asarray(_env_value(observation, 0), dtype=np.float64),
                        action=np.asarray(_env_value(action, 0), dtype=np.float64),
                        joint_position_rad=snapshot.joint_position_rad,
                        joint_velocity_rad_s=snapshot.joint_velocity_rad_s,
                        gripper_position_m=snapshot.gripper_position_m,
                        gripper_velocity_m_s=snapshot.gripper_velocity_m_s,
                        object_poses_world=snapshot.object_poses_world_xyzw,
                        contact_force_proxy_n=snapshot.contact_force_proxy_n,
                        grip_contact_force_peak_n=snapshot.grip_contact_force_peak_n,
                        insertion_contact_force_n=snapshot.insertion_contact_force_n,
                        insertion_contact_force_w=snapshot.insertion_contact_force_w,
                        insertion_contact_force_peak_n=snapshot.insertion_contact_force_peak_n,
                        insertion_contact_load_n=snapshot.insertion_contact_load_n,
                        insertion_contact_load_peak_n=snapshot.insertion_contact_load_peak_n,
                        forbidden_contact_load_n=snapshot.forbidden_contact_load_n,
                        forbidden_contact_load_peak_n=snapshot.forbidden_contact_load_peak_n,
                        phase=snapshot.phase,
                        ground_truth=snapshot.ground_truth,
                    )
                )
                steps_recorded += 1
                independent_result = evaluator.update(
                    EvaluationSample(
                        actual_depth_m=snapshot.ground_truth.actual_depth_m,
                        lateral_error_m=snapshot.ground_truth.lateral_error_m,
                        orientation_error_rad=snapshot.ground_truth.orientation_error_rad,
                        force_peak_n=snapshot.ground_truth.force_peak_n,
                        grasp_both=snapshot.ground_truth.grasp_both,
                        forbidden_contact=snapshot.ground_truth.forbidden_contact,
                        sim_time_s=timestamp_s,
                    )
                )
                if done or independent_result.terminal:
                    final_evaluation = reconcile_terminal_evaluation(
                        extras=extras,
                        snapshot=snapshot,
                        terminal_timestamp_s=timestamp_s,
                        independent_result=independent_result,
                        task_limits=task_limits,
                    )
                    break
                observation = next_observation["policy"]
            if final_evaluation is None:
                final_evaluation = _invalid_terminal(
                    snapshot, steps_recorded * float(env.step_dt), "recorder_step_limit"
                )
            episode_hash = writer.finalize_episode(final_evaluation)
            reports.append(
                {
                    "episode_id": metadata.episode_id,
                    "seed": seed,
                    "split": metadata.split.value,
                    "steps": steps_recorded,
                    "success": final_evaluation.success,
                    "evaluation_valid": final_evaluation.valid,
                    "termination_reason": final_evaluation.termination_reason,
                    "episode_sha256": episode_hash,
                }
            )
        except Exception:
            writer.reset_episode()
            raise
    return reports


def select_failure_reports(
    reports: Sequence[Mapping[str, Any]], *, limit: int = 2
) -> list[dict[str, Any]]:
    """영상·검토 후보로 쓸 실패 episode를 입력 순서대로 제한한다."""

    if limit < 0:
        raise RecordingContractError("failure selection limit must be non-negative")
    return [dict(report) for report in reports
            if report.get("evaluation_valid") is True and report.get("success") is False][:limit]


def existing_source_hashes(
    paths: Sequence[str | Path], *, base_dir: str | Path | None = None
) -> dict[str, str]:
    """provenance에 넣을 실제 존재 파일을 SHA-256으로 읽는다."""

    from .dataset import sha256_file

    result: dict[str, str] = {}
    base = Path(base_dir).resolve() if base_dir is not None else None
    for value in paths:
        path = Path(value).resolve()
        if not path.is_file():
            raise RecordingContractError(f"source file does not exist: {path}")
        try:
            label = str(path.relative_to(base)) if base is not None else path.name
        except ValueError as error:
            raise RecordingContractError(f"source file is outside provenance base: {path}") from error
        result[label] = sha256_file(path)
    if len(result) != len(paths):
        raise RecordingContractError("source file labels must be unique")
    return result
