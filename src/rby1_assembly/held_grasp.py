"""Build and reverify a held-grasp reset state from strict nominal evidence."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rby1_assembly.experiment_contracts import (
    atomic_write_json,
    load_finite_json,
    verify_rj45_nominal_evidence,
)
from rby1_assembly.rj45_task import (
    relative_mating_velocity,
    verify_recorded_mating_geometry,
)
from rby1_assembly.rj45_training import (
    HELD_GRASP_STABILITY_LIMITS,
    validate_held_grasp_artifact,
)


_SETTLE_TIME_S = 0.2
_PRE_PORT_MAX_DEPTH_M = -0.003


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        raise ValueError(f"{name} must be a finite {'non-negative ' if nonnegative else ''}number")
    return result


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite shape {shape}")
    return array


def _project_path(project: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    raw = Path(value)
    path = raw.resolve() if raw.is_absolute() else (project / raw).resolve()
    if path != project and project not in path.parents:
        raise ValueError(f"{name} must stay inside the project")
    return path


def _relative_path(project: Path, path: Path, name: str) -> str:
    path = Path(path).resolve()
    if path != project and project not in path.parents:
        raise ValueError(f"{name} must stay inside the project")
    return str(path.relative_to(project))


def _read_trace(trace_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    previous_time = -1.0
    for index, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            raise ValueError("nominal trace must not contain blank lines")
        record = _mapping(
            _loads_trace_line(line, trace_path, index + 1), "nominal trace record"
        )
        environments = record.get("environments")
        if (
            record.get("schema_version") != 1
            or record.get("control_step") != index
            or not isinstance(environments, list)
            or len(environments) != 1
        ):
            raise ValueError("nominal trace must contain contiguous one-environment records")
        entry = _mapping(environments[0], "nominal trace environment")
        if entry.get("env_index") != 0:
            raise ValueError("nominal trace environment index must be zero")
        now = _finite_number(entry.get("timestamp_s"), "trace timestamp", nonnegative=True)
        if index > 0 and now <= previous_time:
            raise ValueError("nominal trace timestamps must increase")
        previous_time = now
        records.append(record)
    if not records:
        raise ValueError("nominal trace must not be empty")
    return records


def _loads_trace_line(line: str, path: Path, line_number: int) -> Any:
    import json

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(line, parse_constant=reject_constant)
        json.dumps(value, allow_nan=False)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid finite JSON in {path}:line {line_number}: {error}") from error
    return value


def _all_physics_speeds_within_limits(entry: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    poses_value = entry.get("object_substep_poses_world_wxyz")
    velocities_value = entry.get("object_substep_link_velocities_world")
    if not isinstance(poses_value, list) or not poses_value:
        raise ValueError("candidate requires object_substep_poses_world_wxyz")
    count = len(poses_value)
    poses = _array(poses_value, (count, 2, 7), "candidate substep poses")
    velocities = _array(
        velocities_value, (count, 2, 6), "candidate substep link velocities"
    )
    endpoint_poses_value = _mapping(
        entry.get("object_poses_world_wxyz"), "candidate endpoint object poses"
    )
    endpoint_poses = _array(
        [endpoint_poses_value.get("socket"), endpoint_poses_value.get("plug")],
        (2, 7),
        "candidate endpoint poses",
    )
    endpoint_velocities = _array(
        entry.get("object_link_velocities_world"),
        (2, 6),
        "candidate endpoint link velocities",
    )
    poses = np.concatenate((poses, endpoint_poses[None]), axis=0)
    velocities = np.concatenate((velocities, endpoint_velocities[None]), axis=0)
    settle = HELD_GRASP_STABILITY_LIMITS
    com_velocity = _array(
        entry.get("object_com_velocities_world"), (2, 6), "candidate COM velocities"
    )
    if (
        np.any(np.linalg.norm(velocities[..., :3], axis=-1) > settle["max_object_linear_speed_m_s"])
        or np.any(np.linalg.norm(velocities[..., 3:], axis=-1) > settle["max_object_angular_speed_rad_s"])
        or np.any(np.linalg.norm(com_velocity[:, :3], axis=-1) > settle["max_object_linear_speed_m_s"])
        or np.any(np.linalg.norm(com_velocity[:, 3:], axis=-1) > settle["max_object_angular_speed_rad_s"])
    ):
        return False
    if np.any(np.abs(np.linalg.norm(poses[:, :, 3:], axis=-1) - 1.0) > 1.0e-4):
        raise ValueError("candidate object pose quaternions must be unit length")
    frames = task["frames"]
    limits = task["evaluation"]
    for pose, velocity in zip(poses, velocities):
        linear, angular = relative_mating_velocity(
            pose[0, :3],
            pose[0, 3:],
            velocity[0, :3],
            velocity[0, 3:],
            pose[1, :3],
            pose[1, 3:],
            velocity[1, :3],
            velocity[1, 3:],
            np.asarray(frames["socket_entry"]["position_m"]),
            np.asarray(frames["plug_tip"]["position_m"]),
        )
        if (
            np.linalg.norm(linear) > limits["max_relative_linear_speed_m_s"]
            or np.linalg.norm(angular) > limits["max_relative_angular_speed_rad_s"]
        ):
            return False
    return True


def _joint_state_is_settled(entry: Mapping[str, Any]) -> bool:
    units = entry.get("all_joint_units")
    if not isinstance(units, list) or len(units) != 62 or any(unit not in ("m", "rad") for unit in units):
        raise ValueError("candidate requires 62 mixed-SI joint units")
    velocity = _array(entry.get("all_joint_velocities_si"), (62,), "candidate joint velocities")
    position = _array(entry.get("all_joint_positions_si"), (62,), "candidate joint positions")
    reference = _array(entry.get("all_joint_reference_si"), (62,), "candidate joint reference")
    settle = HELD_GRASP_STABILITY_LIMITS
    prismatic = np.asarray(units) == "m"
    velocity_limit = np.where(prismatic, settle["max_prismatic_joint_speed_m_s"], settle["max_revolute_joint_speed_rad_s"])
    tracking_limit = np.where(prismatic, settle["max_prismatic_tracking_error_m"], settle["max_revolute_tracking_error_rad"])
    return bool(np.all(np.abs(velocity) <= velocity_limit) and np.all(np.abs(position - reference) <= tracking_limit))


def _contact_manifold_is_proven(entry: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    manifolds = entry.get("grip_contact_manifold")
    if not isinstance(manifolds, list) or len(manifolds) != 4:
        return False
    threshold = _finite_number(
        task["controller"].get("grip_contact_force_n"),
        "controller.grip_contact_force_n",
        nonnegative=True,
    )
    for finger_index, manifold_value in enumerate(manifolds):
        manifold = _mapping(manifold_value, f"grip_contact_manifold[{finger_index}]")
        count = manifold.get("contact_count")
        contacts = manifold.get("contacts")
        if type(count) is not int or count <= 0 or not isinstance(contacts, list) or len(contacts) != count:
            return False
        force_sum = 0.0
        for contact_index, contact_value in enumerate(contacts):
            contact = _mapping(
                contact_value,
                f"grip_contact_manifold[{finger_index}].contacts[{contact_index}]",
            )
            force_sum += _finite_number(
                contact.get("normal_force_n"), "manifold normal force", nonnegative=True
            )
            _finite_number(contact.get("separation_m"), "manifold separation")
            _array(contact.get("point_world_m"), (3,), "manifold point")
            _array(contact.get("normal_world"), (3,), "manifold normal")
        if force_sum < threshold:
            return False
    return True


def _candidate_entry_is_stable(
    entry: Mapping[str, Any], task: Mapping[str, Any], forbidden_threshold_n: float
) -> bool:
    if (
        entry.get("phase_name") != "insert"
        or entry.get("terminal") is not False
        or entry.get("termination_reason") != "none"
        or entry.get("independent_eval_valid") is not True
        or entry.get("grip_live") != [True, True]
        or entry.get("grip_proven") != [True, True]
    ):
        return False
    poses = _mapping(entry.get("object_poses_world_wxyz"), "candidate object poses")
    geometry = verify_recorded_mating_geometry(
        poses.get("socket"),
        poses.get("plug"),
        task["frames"]["socket_entry"],
        task["frames"]["plug_tip"],
        entry.get("actual_depth_m"),
        entry.get("actual_lateral_error_m"),
        entry.get("actual_orientation_error_rad"),
    )
    if geometry["depth_m"] > _PRE_PORT_MAX_DEPTH_M:
        return False
    limits = task["evaluation"]
    for name in (
        "grip_contact_force_peak_n",
        "insertion_contact_load_n",
        "insertion_contact_load_peak_n",
    ):
        value = entry.get(name)
        if value is not None and _finite_number(value, name, nonnegative=True) > limits["max_contact_force_n"]:
            return False
    forbidden = _finite_number(
        entry.get("forbidden_contact_load_n"),
        "forbidden_contact_load_n",
        nonnegative=True,
    )
    forbidden_peak = _finite_number(
        entry.get("forbidden_contact_load_peak_n"),
        "forbidden_contact_load_peak_n",
        nonnegative=True,
    )
    if forbidden > forbidden_threshold_n or forbidden_peak > forbidden_threshold_n:
        return False
    criteria = _mapping(entry.get("independent_eval_criteria"), "candidate criteria")
    if criteria.get("grasp_live_and_proven") is not True or criteria.get("force") is not True or criteria.get("forbidden_contact") is not True:
        return False
    return _joint_state_is_settled(entry) and _all_physics_speeds_within_limits(entry, task) and _contact_manifold_is_proven(
        entry, task
    )


def _select_candidate(
    records: list[dict[str, Any]], task: Mapping[str, Any], forbidden_threshold_n: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    stable_since: float | None = None
    for record in records:
        entry = _mapping(record["environments"][0], "nominal trace environment")
        now = _finite_number(entry.get("timestamp_s"), "trace timestamp", nonnegative=True)
        if _candidate_entry_is_stable(entry, task, forbidden_threshold_n):
            stable_since = now if stable_since is None else stable_since
            if now - stable_since + 1.0e-6 >= _SETTLE_TIME_S:
                return record, entry
        else:
            stable_since = None
    raise ValueError("nominal trace has no 0.2 s stable pre-port INSERT held-grasp candidate")


def _source_and_config_hashes(
    project: Path, task: Mapping[str, Any], report: Mapping[str, Any]
) -> dict[str, str]:
    provenance = _mapping(report.get("provenance"), "nominal report provenance")
    result = dict(_mapping(provenance.get("source_sha256"), "nominal source hashes"))
    paths_and_hashes = {
        "configs/runtime.json": provenance.get("runtime_config_sha256"),
        "configs/tasks/rj45-v1.json": provenance.get("task_config_sha256"),
        "configs/assets.json": provenance.get("assets_config_sha256"),
        "configs/rj45-assets.json": provenance.get("connector_assets_config_sha256"),
        str(Path(task["assets"]["build_report"])): provenance.get("build_report_sha256"),
    }
    result.update(paths_and_hashes)
    for relative_path, expected in result.items():
        path = _project_path(project, relative_path, f"source/config path {relative_path}")
        if not path.is_file() or not isinstance(expected, str) or _sha256(path) != expected:
            raise ValueError(f"current source/config hash mismatch: {relative_path}")
    return dict(sorted(result.items()))


def _state_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    poses = _mapping(entry.get("object_poses_world_wxyz"), "candidate object poses")
    ee_poses = _mapping(entry.get("ee_poses"), "candidate end-effector poses")
    return {
        "joint_positions_si": entry.get("all_joint_positions_si"),
        "joint_velocities_si": entry.get("all_joint_velocities_si"),
        "joint_reference_si": entry.get("all_joint_reference_si"),
        "object_poses_world_wxyz": [poses.get("socket"), poses.get("plug")],
        "object_com_velocities_world": entry.get("object_com_velocities_world"),
        "end_effector_poses_world_wxyz": [
            ee_poses.get("left_world_wxyz"),
            ee_poses.get("right_world_wxyz"),
        ],
    }


def _artifact_from_candidate(
    project: Path,
    task: Mapping[str, Any],
    report_path: Path,
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
    record: Mapping[str, Any],
    entry: Mapping[str, Any],
) -> dict[str, Any]:
    trace_path = _project_path(project, report["trace_path"], "nominal trace path")
    return {
        "schema_version": 1,
        "task_id": "rj45-v1",
        "provenance": {
            "nominal_report_sha256": evidence["report_sha256"],
            "nominal_report_path": _relative_path(project, report_path, "nominal report path"),
            "nominal_trace_sha256": evidence["trace_sha256"],
            "nominal_trace_path": _relative_path(project, trace_path, "nominal trace path"),
            "runtime_config_sha256": report["provenance"]["runtime_config_sha256"],
            "task_config_sha256": task["_config_sha256"],
            "source_and_config_sha256": _source_and_config_hashes(project, task, report),
        },
        "selection": {
            "control_step": record["control_step"],
            "timestamp_s": entry["timestamp_s"],
            "phase_name": "insert",
        },
        "units": {
            "joint_position": "per_joint_units",
            "joint_velocity": "per_joint_units/s",
            "joint_reference": "per_joint_units",
            "object_pose": "m,wxyz",
            "object_com_velocity": "m/s,rad/s",
            "end_effector_pose": "m,wxyz",
        },
        "joint_names": entry.get("all_joint_names"),
        "joint_units": entry.get("all_joint_units"),
        "state": _state_from_entry(entry),
    }


def build_held_grasp_artifact(
    project: Path,
    task: dict[str, Any],
    nominal_report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Select, validate, and atomically persist the first stable pre-port state."""

    project = Path(project).resolve()
    report_path = _project_path(project, str(nominal_report_path), "nominal report path")
    output_path = _project_path(project, str(output_path), "held-grasp output path")
    evidence = verify_rj45_nominal_evidence(project, task, report_path)
    report = _mapping(load_finite_json(report_path), "nominal report")
    trace_path = _project_path(project, report.get("trace_path"), "nominal trace path")
    records = _read_trace(trace_path)
    if _sha256(report_path) != evidence["report_sha256"] or _sha256(trace_path) != evidence["trace_sha256"]:
        raise ValueError("nominal report or trace changed after strict verification")
    forbidden_threshold = _finite_number(
        report.get("forbidden_contact_force_threshold_n"),
        "forbidden contact threshold",
        nonnegative=True,
    )
    record, entry = _select_candidate(records, task, forbidden_threshold)
    artifact = _artifact_from_candidate(
        project, task, report_path, report, evidence, record, entry
    )
    validated = validate_held_grasp_artifact(artifact)
    atomic_write_json(output_path, validated)
    return validated


def load_verified_held_grasp_artifact(
    project: Path,
    task: dict[str, Any],
    artifact_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reverify provenance and the exact selected trace row before reset use."""

    project = Path(project).resolve()
    artifact_path = _project_path(project, str(artifact_path), "held-grasp artifact path")
    artifact = validate_held_grasp_artifact(load_finite_json(artifact_path))
    provenance = artifact["provenance"]
    report_path = _project_path(project, provenance["nominal_report_path"], "nominal report path")
    evidence = verify_rj45_nominal_evidence(project, task, report_path)
    if evidence["report_sha256"] != provenance["nominal_report_sha256"]:
        raise ValueError("held-grasp nominal report SHA-256 is stale")
    if evidence["trace_sha256"] != provenance["nominal_trace_sha256"]:
        raise ValueError("held-grasp nominal trace SHA-256 is stale")
    report = _mapping(load_finite_json(report_path), "nominal report")
    trace_path = _project_path(project, provenance["nominal_trace_path"], "nominal trace path")
    if trace_path != _project_path(project, report["trace_path"], "report trace path"):
        raise ValueError("held-grasp trace path differs from the nominal report")
    current_hashes = _source_and_config_hashes(project, task, report)
    if current_hashes != provenance["source_and_config_sha256"]:
        raise ValueError("held-grasp source/config hashes are stale")
    if provenance["runtime_config_sha256"] != report["provenance"]["runtime_config_sha256"]:
        raise ValueError("held-grasp runtime hash is stale")
    if provenance["task_config_sha256"] != task["_config_sha256"]:
        raise ValueError("held-grasp task hash is stale")
    records = _read_trace(trace_path)
    if _sha256(report_path) != evidence["report_sha256"] or _sha256(trace_path) != evidence["trace_sha256"]:
        raise ValueError("nominal report or trace changed during held-grasp verification")
    forbidden_threshold = _finite_number(
        report.get("forbidden_contact_force_threshold_n"),
        "forbidden contact threshold",
        nonnegative=True,
    )
    selected_record, selected_entry = _select_candidate(records, task, forbidden_threshold)
    expected = _artifact_from_candidate(
        project,
        task,
        report_path,
        report,
        evidence,
        selected_record,
        selected_entry,
    )
    if artifact != expected:
        raise ValueError("held-grasp artifact differs from the exact selected nominal trace record")
    return artifact, {
        "artifact_sha256": _sha256(artifact_path),
        "nominal_report_sha256": evidence["report_sha256"],
        "nominal_trace_sha256": evidence["trace_sha256"],
        "selection_control_step": selected_record["control_step"],
        "stability_limits": dict(HELD_GRASP_STABILITY_LIMITS),
    }


__all__ = ["build_held_grasp_artifact", "load_verified_held_grasp_artifact"]
