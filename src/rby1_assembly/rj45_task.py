"""CPU-only validation and rigid-frame math for the RJ45 assembly task."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


_FRAME_NAMES = ("socket_entry", "plug_tip", "left_grasp", "right_grasp")
_ASSET_NAMES = ("robot", "socket", "plug", "table", "build_report")
_GENERATED_ASSET_NAMES = ("robot", "socket", "plug", "table")
_APPROACH_WAYPOINT_GROUP = (
    "left_above_joint_positions_rad",
    "right_above_joint_positions_rad",
    "torso_work_joint_positions_rad",
    "approach_waypoint_max_joint_delta_rad",
    "approach_waypoint_tracking_tolerance_rad",
    "nullspace_posture_gain",
    "nullspace_max_joint_delta_rad",
    "pregrasp_integral_gain",
    "alignment_integral_gain",
)
_OPTIONAL_CONTROLLER_POSITIVE = (
    "align_timeout_s",
    "dls_damping",
    "grip_contact_force_n",
    "grip_settle_s",
    "insert_timeout_s",
    "insertion_advance_lateral_tolerance_m",
    "insertion_advance_orientation_tolerance_rad",
    "insertion_integral_gain",
    "insertion_resume_lateral_tolerance_m",
    "insertion_resume_orientation_tolerance_rad",
    "jam_contact_force_n",
    "max_joint_delta_rad",
    "phase_timeout_s",
    "pregrasp_distance_m",
    "pregrasp_orientation_tolerance_rad",
    "pregrasp_timeout_s",
    "torso_max_joint_delta_rad",
    "pregrasp_safe_wrist_height_m",
    "pregrasp_waypoint_tolerance_rad",
    "pregrasp_torso_tolerance_rad",
    "torso_reference_tracking_tolerance_rad",
    "approach_waypoint_max_joint_delta_rad",
    "pregrasp_above_height_m",
    "approach_waypoint_tracking_tolerance_rad",
    "gripper_command_speed_m_s",
    "alignment_integral_gain",
    "nullspace_posture_gain",
    "nullspace_max_joint_delta_rad",
    "pregrasp_integral_gain",
)


def _number(value: Any, name: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _vector(value: Any, name: str, length: int) -> list[float]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-element array")
    return [_number(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _quaternion(value: Any, name: str) -> list[float]:
    quaternion = _vector(value, name, 4)
    norm = math.sqrt(sum(component * component for component in quaternion))
    if abs(norm - 1.0) > 1.0e-5:
        raise ValueError(f"{name} must be a unit quaternion in wxyz order")
    return [component / norm for component in quaternion]


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _reject_nonfinite_numbers(value: Any, name: str = "task") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nonfinite_numbers(item, f"{name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_nonfinite_numbers(item, f"{name}.{key}")


def _relative_asset_path(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the project")
    return path.as_posix()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def verify_generated_asset_hashes(
    project: Path, task: dict[str, Any], build_report: dict[str, Any]
) -> dict[str, str]:
    """Verify every generated runtime USD against its build-report digest."""

    project = Path(project).resolve()
    report_files = _mapping(build_report.get("files"), "build_report.files")
    expected_paths = {
        Path(task["assets"][name]).name: (project / task["assets"][name]).resolve()
        for name in _GENERATED_ASSET_NAMES
    }
    build_report_path = (project / task["assets"]["build_report"]).resolve()
    expected_paths["scene.usda"] = build_report_path.parent / "scene.usda"
    if set(report_files) != set(expected_paths):
        raise ValueError(
            "build_report.files must contain exactly robot, socket, plug, table, and scene USDs"
        )

    actual: dict[str, str] = {}
    for filename, path in expected_paths.items():
        if project not in path.parents or not path.is_file():
            raise ValueError(f"missing project-local generated asset: {path}")
        expected_sha = _validate_sha256(
            report_files[filename], f"build_report.files.{filename}"
        )
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"generated asset SHA-256 mismatch for {filename}: "
                f"expected {expected_sha}, got {actual_sha}"
            )
        actual[filename] = actual_sha
    return actual


def verify_checkpoint_lineage(
    lineage: dict[str, Any],
    *,
    task_id: str,
    task_config_sha256: str,
    source_sha256: dict[str, str],
    runtime_contract: dict[str, Any],
    rj45_nominal_evidence: dict[str, Any],
) -> None:
    """Fail closed when a checkpoint provenance record differs from current inputs."""

    if not isinstance(lineage, dict) or lineage.get("task_id") != task_id:
        raise ValueError(f"checkpoint provenance task_id must be {task_id}")
    _reject_nonfinite_numbers(lineage, "checkpoint_provenance")
    if type(lineage.get("schema_version")) is not int or lineage["schema_version"] != 1:
        raise ValueError("checkpoint provenance schema_version must be 1")
    contract_keys = {
        "runtime_versions", "physics_dt_s", "control_decimation", "control_dt_s",
        "observation_noise", "baseline_controller_state_source", "task_config_path", "policy_config_path",
        "collision_monitor_mode", "num_envs",
    }
    if set(runtime_contract) != contract_keys:
        raise ValueError("runtime contract must specify versions, physics, observations and input paths")
    _reject_nonfinite_numbers(runtime_contract, "runtime_contract")
    for name, expected in runtime_contract.items():
        if lineage.get(name) != expected:
            raise ValueError(f"checkpoint runtime contract mismatch for {name}")
    expected_task_sha = _validate_sha256(task_config_sha256, "task_config_sha256")
    if lineage.get("task_config_sha256") != expected_task_sha:
        raise ValueError("checkpoint task configuration does not match the current task")
    recorded_sources = _mapping(
        lineage.get("source_and_config_sha256"), "source_and_config_sha256"
    )
    for path, digest in source_sha256.items():
        expected_digest = _validate_sha256(digest, f"source_sha256.{path}")
        if recorded_sources.get(path) != expected_digest:
            raise ValueError(f"checkpoint provenance mismatch for {path}")
    evidence_keys = {"report_sha256", "trace_sha256", "trace_records"}
    if not isinstance(rj45_nominal_evidence, dict) or set(rj45_nominal_evidence) != evidence_keys:
        raise ValueError(
            "current RJ45 nominal evidence must contain report, trace, and record count"
        )
    expected_evidence = {
        "report_sha256": _validate_sha256(
            rj45_nominal_evidence["report_sha256"],
            "rj45_nominal_evidence.report_sha256",
        ),
        "trace_sha256": _validate_sha256(
            rj45_nominal_evidence["trace_sha256"],
            "rj45_nominal_evidence.trace_sha256",
        ),
        "trace_records": rj45_nominal_evidence["trace_records"],
    }
    if (
        type(expected_evidence["trace_records"]) is not int
        or expected_evidence["trace_records"] <= 0
    ):
        raise ValueError("rj45_nominal_evidence.trace_records must be a positive integer")
    recorded_evidence = _mapping(
        lineage.get("rj45_nominal_evidence"),
        "checkpoint_provenance.rj45_nominal_evidence",
    )
    if set(recorded_evidence) != evidence_keys:
        raise ValueError(
            "checkpoint RJ45 nominal evidence must contain report, trace, and record count"
        )
    if recorded_evidence != expected_evidence:
        raise ValueError("checkpoint RJ45 nominal evidence does not match the verified run")


def verify_runtime_versions(versions: dict[str, str], runtime: dict[str, Any]) -> None:
    """Require the installed distributions pinned by this experiment."""
    expected = {
        "isaacsim": runtime["isaac_sim_distribution"],
        "isaaclab": runtime["isaac_lab_distribution"],
        "isaaclab-rl": runtime["isaac_lab_rl"],
        "rsl-rl-lib": runtime["rsl_rl"],
        "torch": runtime["torch"] + "+" + runtime["torch_cuda_build"],
        "numpy": runtime["numpy"], "h5py": runtime["h5py"],
    }
    if versions != expected:
        raise ValueError(f"installed experiment distributions differ from pinned runtime: {versions}")


def verify_checkpoint_manifest(
    manifest: dict[str, Any],
    *,
    training_provenance_sha256: str,
    checkpoint_name: str,
    checkpoint_sha256: str,
) -> None:
    """Bind a completed training record to the exact checkpoint being evaluated."""

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("checkpoint manifest schema_version must be 1")
    if manifest.get("training_completed") is not True:
        raise ValueError("checkpoint manifest must record completed training")
    provenance_sha = _validate_sha256(
        training_provenance_sha256, "training_provenance_sha256"
    )
    if manifest.get("training_provenance_sha256") != provenance_sha:
        raise ValueError("checkpoint manifest does not match training_provenance.json")
    if not isinstance(checkpoint_name, str) or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be a basename")
    expected_checkpoint_sha = _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    checkpoints = _mapping(manifest.get("checkpoints"), "checkpoint_manifest.checkpoints")
    if checkpoints.get(checkpoint_name) != expected_checkpoint_sha:
        raise ValueError(f"checkpoint manifest mismatch for {checkpoint_name}")


def _validate_pose(container: dict[str, Any], name: str) -> None:
    pose = _mapping(container.get(name), f"frames.{name}")
    pose["position_m"] = _vector(pose.get("position_m"), f"frames.{name}.position_m", 3)
    pose["quaternion_wxyz"] = _quaternion(
        pose.get("quaternion_wxyz"), f"frames.{name}.quaternion_wxyz"
    )


def load_rj45_task(config_path: Path) -> dict[str, Any]:
    """Load and fail-closed validate an RJ45 task configuration.

    The returned mapping is detached from the JSON decoder's mutable objects and
    contains only finite SI values. Asset existence is checked later by the
    runtime configurator because generation may happen after config validation.
    """

    config_path = Path(config_path)
    try:
        config_bytes = config_path.read_bytes()
        task = deepcopy(json.loads(config_bytes.decode("utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid RJ45 task config {config_path}: {error}") from error
    task = _mapping(task, "task")
    _reject_nonfinite_numbers(task)

    if task.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    if not isinstance(task.get("scope"), str) or not task["scope"].strip():
        raise ValueError("scope must be a non-empty string")
    if not isinstance(task.get("parameter_provenance"), str) or not task[
        "parameter_provenance"
    ].strip():
        raise ValueError("parameter_provenance must be a non-empty string")
    if task.get("length_unit") != "m" or task.get("angle_unit") != "rad":
        raise ValueError("RJ45 task values must use metres and radians")
    if task.get("world_up_axis") != "Z" or task.get("insertion_axis_world") != "+Y":
        raise ValueError("RJ45 task requires Z-up assets normalized to the +Y insertion axis")

    frames = _mapping(task.get("frames"), "frames")
    for name in _FRAME_NAMES:
        _validate_pose(frames, name)

    assets = _mapping(task.get("assets"), "assets")
    for name in _ASSET_NAMES:
        assets[name] = _relative_asset_path(assets.get(name), f"assets.{name}")

    table = _mapping(task.get("table"), "table")
    table["top_height_m"] = _number(table.get("top_height_m"), "table.top_height_m", positive=True)
    table["size_m"] = _vector(table.get("size_m"), "table.size_m", 3)
    if min(table["size_m"]) <= 0.0:
        raise ValueError("table.size_m components must be positive")

    for object_name in ("socket", "plug"):
        obj = _mapping(task.get(object_name), object_name)
        obj["mass_kg"] = _number(obj.get("mass_kg"), f"{object_name}.mass_kg", positive=True)
        obj["initial_position_m"] = _vector(
            obj.get("initial_position_m"), f"{object_name}.initial_position_m", 3
        )
        obj["initial_orientation_wxyz"] = _quaternion(
            obj.get("initial_orientation_wxyz"),
            f"{object_name}.initial_orientation_wxyz",
        )

    contact = _mapping(task.get("contact"), "contact")
    for name in ("static_friction", "dynamic_friction", "restitution", "contact_offset_m"):
        contact[name] = _number(contact.get(name), f"contact.{name}", nonnegative=True)
    contact["rest_offset_m"] = _number(contact.get("rest_offset_m"), "contact.rest_offset_m")
    if contact["dynamic_friction"] > contact["static_friction"]:
        raise ValueError("contact.dynamic_friction must not exceed static_friction")
    if contact["rest_offset_m"] > contact["contact_offset_m"]:
        raise ValueError("contact.rest_offset_m must not exceed contact_offset_m")

    for name in ("solver_position_iteration_count", "solver_velocity_iteration_count"):
        value = contact.get(name)
        lower = 0 if name == "solver_velocity_iteration_count" else 1
        if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= 255:
            raise ValueError(f"contact.{name} must be an integer from {lower} to 255")

    if not isinstance(contact.get("enable_external_forces_every_iteration"), bool):
        raise ValueError("contact.enable_external_forces_every_iteration must be boolean")

    evaluation = _mapping(task.get("evaluation"), "evaluation")
    for name in (
        "assembled_depth_m",
        "max_lateral_error_m",
        "max_orientation_error_rad",
        "max_contact_force_n",
        "dwell_s",
        "timeout_s",
    ):
        evaluation[name] = _number(evaluation.get(name), f"evaluation.{name}", positive=True)
    for name in (
        "max_depth_m",
        "max_relative_linear_speed_m_s",
        "max_relative_angular_speed_rad_s",
    ):
        evaluation[name] = _number(evaluation.get(name), f"evaluation.{name}", positive=True)
    if evaluation["max_depth_m"] < evaluation["assembled_depth_m"]:
        raise ValueError("evaluation.max_depth_m must be at least assembled_depth_m")

    controller = _mapping(task.get("controller"), "controller")
    for name, length in (
        ("torso_work_joint_positions_rad", 6),
        ("pregrasp_clearance_joint_pos_rad", 7),
        ("initial_clearance_joint_positions_rad", 7),
        ("left_above_joint_positions_rad", 7),
        ("right_above_joint_positions_rad", 7),
    ):
        if name in controller:
            controller[name] = _vector(controller[name], f"controller.{name}", length)
    if "torso_work_joint_positions_rad" in controller:
        required = (
            "initial_clearance_joint_positions_rad", "pregrasp_clearance_joint_pos_rad",
            "torso_max_joint_delta_rad", "pregrasp_safe_wrist_height_m",
            "pregrasp_waypoint_tolerance_rad", "pregrasp_torso_tolerance_rad",
            "torso_reference_tracking_tolerance_rad",
        )
        if any(name not in controller for name in required):
            raise ValueError("torso work posture requires both clearance waypoints and motion limits")
    if any(name in controller for name in _APPROACH_WAYPOINT_GROUP):
        missing = [name for name in _APPROACH_WAYPOINT_GROUP if name not in controller]
        if missing:
            raise ValueError(
                "approach waypoint group requires both arm waypoints, torso work posture, "
                f"and progression limits; missing: {', '.join(missing)}"
            )
    for name in (
        "left_gripper_grasp_magnitude_m",
        "right_gripper_grasp_magnitude_m",
        "gripper_open_magnitude_m",
        "residual_translation_bound_m",
        "residual_rotation_bound_rad",
        "alignment_standoff_m",
        "insertion_target_margin_m",
        "insertion_speed_m_s",
        "insertion_max_tracking_lead_m",
    ):
        controller[name] = _number(controller.get(name), f"controller.{name}", positive=True)
    controller["assembly_socket_goal_m"] = _vector(
        controller.get("assembly_socket_goal_m"), "controller.assembly_socket_goal_m", 3
    )
    controller["assembly_socket_goal_quaternion_wxyz"] = _quaternion(
        controller.get("assembly_socket_goal_quaternion_wxyz"),
        "controller.assembly_socket_goal_quaternion_wxyz",
    )
    for name in _OPTIONAL_CONTROLLER_POSITIVE:
        if name in controller:
            controller[name] = _number(controller[name], f"controller.{name}", positive=True)
    if "max_retries" in controller:
        max_retries = controller["max_retries"]
        if isinstance(max_retries, bool) or not isinstance(max_retries, int) or max_retries < 0:
            raise ValueError("controller.max_retries must be a non-negative integer")

    reset = task.setdefault("reset", {})
    reset = _mapping(reset, "reset")
    reset["position_noise_m"] = _vector(
        reset.get("position_noise_m", [0.0, 0.0, 0.0]), "reset.position_noise_m", 3
    )
    if min(reset["position_noise_m"]) < 0.0:
        raise ValueError("reset.position_noise_m components must be non-negative")

    data = _mapping(task.get("data"), "data")
    if data.get("observation_source") != "simulator_state_with_noise":
        raise ValueError("data.observation_source must be simulator_state_with_noise")
    for seed_name in ("train_seed", "evaluation_seed"):
        seed = data.get(seed_name)
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"data.{seed_name} must be a non-negative integer")
    task["_config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    return task


def quaternion_multiply_wxyz(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamilton product for arrays ending in ``(w, x, y, z)``."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def quaternion_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vector = np.asarray(vector, dtype=np.float64)
    vector_quaternion = np.concatenate((np.zeros(vector.shape[:-1] + (1,)), vector), axis=-1)
    conjugate = quaternion.copy()
    conjugate[..., 1:] *= -1.0
    return quaternion_multiply_wxyz(
        quaternion_multiply_wxyz(quaternion, vector_quaternion), conjugate
    )[..., 1:]


def compose_pose(
    parent_position: np.ndarray,
    parent_quaternion: np.ndarray,
    child_position: np.ndarray,
    child_quaternion: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    position = np.asarray(parent_position, dtype=np.float64) + quaternion_rotate_wxyz(
        parent_quaternion, child_position
    )
    quaternion = quaternion_multiply_wxyz(parent_quaternion, child_quaternion)
    return position, quaternion


def invert_pose(position: np.ndarray, quaternion: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    inverse_quaternion = quaternion.copy()
    inverse_quaternion[..., 1:] *= -1.0
    return -quaternion_rotate_wxyz(inverse_quaternion, position), inverse_quaternion


def relative_mating_pose(
    socket_position: np.ndarray,
    socket_quaternion: np.ndarray,
    plug_position: np.ndarray,
    plug_quaternion: np.ndarray,
    socket_entry: dict[str, Any],
    plug_tip: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the plug-tip pose expressed in the measured socket-entry frame."""

    entry_position, entry_quaternion = compose_pose(
        socket_position,
        socket_quaternion,
        np.asarray(socket_entry["position_m"]),
        np.asarray(socket_entry["quaternion_wxyz"]),
    )
    tip_position, tip_quaternion = compose_pose(
        plug_position,
        plug_quaternion,
        np.asarray(plug_tip["position_m"]),
        np.asarray(plug_tip["quaternion_wxyz"]),
    )
    inverse_position, inverse_quaternion = invert_pose(entry_position, entry_quaternion)
    return compose_pose(inverse_position, inverse_quaternion, tip_position, tip_quaternion)


def verify_recorded_mating_geometry(
    socket_pose_world_wxyz: Any,
    plug_pose_world_wxyz: Any,
    socket_entry: dict[str, Any],
    plug_tip: dict[str, Any],
    actual_depth_m: Any,
    actual_lateral_error_m: Any,
    actual_orientation_error_rad: Any,
    *,
    absolute_tolerance: float = 2.0e-6,
) -> dict[str, Any]:
    """Recalculate mating geometry from raw poses and fail closed on bad evidence."""

    tolerance = _number(absolute_tolerance, "absolute_tolerance", positive=True)
    try:
        socket_pose = np.asarray(socket_pose_world_wxyz, dtype=np.float64)
        plug_pose = np.asarray(plug_pose_world_wxyz, dtype=np.float64)
        runtime_geometry = np.asarray(
            [actual_depth_m, actual_lateral_error_m, actual_orientation_error_rad],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("recorded RJ45 geometry must contain numeric values") from error
    for pose_name, pose in (("socket", socket_pose), ("plug", plug_pose)):
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError(
                f"recorded raw {pose_name} pose must be a finite xyz+wxyz vector"
            )
        quaternion_norm = float(np.linalg.norm(pose[3:]))
        if quaternion_norm == 0.0 or abs(quaternion_norm - 1.0) > 1.0e-4:
            raise ValueError(
                f"recorded raw {pose_name} quaternion must be unit length within 1e-4"
            )
    for frame_name, frame in (("socket_entry", socket_entry), ("plug_tip", plug_tip)):
        try:
            frame_position = np.asarray(frame["position_m"], dtype=np.float64)
            frame_quaternion = np.asarray(frame["quaternion_wxyz"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{frame_name} must contain numeric position and quaternion") from error
        if (
            frame_position.shape != (3,)
            or frame_quaternion.shape != (4,)
            or not np.isfinite(frame_position).all()
            or not np.isfinite(frame_quaternion).all()
        ):
            raise ValueError(f"{frame_name} must be a finite xyz and wxyz pose")
        quaternion_norm = float(np.linalg.norm(frame_quaternion))
        if quaternion_norm == 0.0 or abs(quaternion_norm - 1.0) > 1.0e-5:
            raise ValueError(f"{frame_name} quaternion must be unit length within 1e-5")
    if runtime_geometry.shape != (3,) or not np.isfinite(runtime_geometry).all():
        raise ValueError("runtime mating geometry must contain three finite values")

    relative_position, relative_quaternion = relative_mating_pose(
        socket_pose[:3],
        socket_pose[3:],
        plug_pose[:3],
        plug_pose[3:],
        socket_entry,
        plug_tip,
    )
    if (
        relative_position.shape != (3,)
        or relative_quaternion.shape != (4,)
        or not np.isfinite(relative_position).all()
        or not np.isfinite(relative_quaternion).all()
    ):
        raise ValueError("independent mating-frame pose must be finite")
    independent_geometry = np.asarray(
        [
            relative_position[1],
            np.linalg.norm(relative_position[[0, 2]]),
            2.0 * np.arctan2(np.linalg.norm(relative_quaternion[1:]), abs(relative_quaternion[0])),
        ],
        dtype=np.float64,
    )
    if not np.isfinite(independent_geometry).all():
        raise ValueError("independent mating geometry must be finite")
    if np.any(np.abs(independent_geometry - runtime_geometry) > tolerance):
        raise ValueError("runtime and independent mating-frame geometry disagree")
    return {
        "depth_m": float(independent_geometry[0]),
        "lateral_error_m": float(independent_geometry[1]),
        "orientation_error_rad": float(independent_geometry[2]),
        "relative_position_entry_m": relative_position.tolist(),
        "relative_quaternion_entry_wxyz": relative_quaternion.tolist(),
        "source": "numpy_recalculation_from_recorded_raw_object_poses",
    }


def plug_root_goal_pose(
    socket_position: np.ndarray,
    socket_quaternion: np.ndarray,
    depth_m: float | np.ndarray,
    socket_entry: dict[str, Any],
    plug_tip: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the plug-root pose whose tip is ``depth_m`` inside the entry frame."""

    entry_position, entry_quaternion = compose_pose(
        socket_position,
        socket_quaternion,
        np.asarray(socket_entry["position_m"]),
        np.asarray(socket_entry["quaternion_wxyz"]),
    )
    target_offset = np.zeros_like(entry_position, dtype=np.float64)
    target_offset[..., 1] = depth_m
    target_tip_position, target_tip_quaternion = compose_pose(
        entry_position,
        entry_quaternion,
        target_offset,
        np.asarray([1.0, 0.0, 0.0, 0.0]),
    )
    inverse_tip_position, inverse_tip_quaternion = invert_pose(
        np.asarray(plug_tip["position_m"]), np.asarray(plug_tip["quaternion_wxyz"])
    )
    return compose_pose(
        target_tip_position,
        target_tip_quaternion,
        inverse_tip_position,
        inverse_tip_quaternion,
    )


def relative_mating_velocity(
    socket_position: np.ndarray,
    socket_quaternion: np.ndarray,
    socket_linear_velocity: np.ndarray,
    socket_angular_velocity: np.ndarray,
    plug_position: np.ndarray,
    plug_quaternion: np.ndarray,
    plug_linear_velocity: np.ndarray,
    plug_angular_velocity: np.ndarray,
    socket_entry_position: np.ndarray,
    plug_tip_position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tip velocity relative to the rotating socket-entry frame.

    Inputs are world-frame rigid-root states; frame offsets are body-local.
    The returned vectors remain expressed in world axes, which preserves the
    speed norms used by the evaluator.
    """

    socket_position = np.asarray(socket_position, dtype=np.float64)
    plug_position = np.asarray(plug_position, dtype=np.float64)
    socket_linear_velocity = np.asarray(socket_linear_velocity, dtype=np.float64)
    socket_angular_velocity = np.asarray(socket_angular_velocity, dtype=np.float64)
    plug_linear_velocity = np.asarray(plug_linear_velocity, dtype=np.float64)
    plug_angular_velocity = np.asarray(plug_angular_velocity, dtype=np.float64)
    entry_offset_world = quaternion_rotate_wxyz(socket_quaternion, socket_entry_position)
    tip_offset_world = quaternion_rotate_wxyz(plug_quaternion, plug_tip_position)
    entry_position_world = socket_position + entry_offset_world
    tip_position_world = plug_position + tip_offset_world
    entry_velocity_world = socket_linear_velocity + np.cross(
        socket_angular_velocity, entry_offset_world
    )
    tip_velocity_world = plug_linear_velocity + np.cross(
        plug_angular_velocity, tip_offset_world
    )
    relative_linear_velocity_world = (
        tip_velocity_world
        - entry_velocity_world
        - np.cross(socket_angular_velocity, tip_position_world - entry_position_world)
    )
    relative_angular_velocity_world = plug_angular_velocity - socket_angular_velocity
    return relative_linear_velocity_world, relative_angular_velocity_world
