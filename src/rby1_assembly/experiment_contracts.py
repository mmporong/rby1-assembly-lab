"""CPU checks for finite experiment inputs and physical training prerequisites."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _loads_finite_json(payload: str, source: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant in {source}: {value}")

    try:
        value = json.loads(payload, parse_constant=reject_constant)
        # Python's decoder turns overflowing finite literals such as 1e999 into
        # infinity without invoking parse_constant. Re-encoding walks the full
        # value recursively and rejects those results.
        json.dumps(value, allow_nan=False)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid finite JSON in {source}: {error}") from error
    return value


def load_finite_json(path: Path):
    path = Path(path)
    try:
        payload = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid finite JSON file {path}: {error}") from error
    return _loads_finite_json(payload, str(path))


def atomic_write_json(path: Path, value) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def validate_ppo_config(config: dict) -> None:
    if not isinstance(config, dict):
        raise ValueError("PPO configuration must be an object")
    # Serialize recursively to reject NaN/Infinity anywhere, including optional fields.
    json.dumps(config, allow_nan=False)
    for key in ("max_iterations", "num_steps_per_env", "save_interval"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if type(config.get("seed")) is not int or config["seed"] < 0:
        raise ValueError("seed must be a non-negative integer")
    for section in ("policy", "algorithm"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be an object")
    def positive(mapping, key, *, upper=None, zero=False):
        value = mapping.get(key)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or (value < 0 if zero else value <= 0)
            or (upper is not None and value > upper)
        ):
            raise ValueError(f"invalid PPO {key}")

    positive(config, "clip_actions")
    positive(config["policy"], "init_noise_std")
    algorithm = config["algorithm"]
    for key in ("learning_rate", "clip_param", "desired_kl", "max_grad_norm"):
        positive(algorithm, key, upper=1 if key == "clip_param" else None)
    for key in ("gamma", "lam"):
        positive(algorithm, key, upper=1)
    for key in ("entropy_coef", "value_loss_coef"):
        positive(algorithm, key, zero=True)
    for key in ("num_learning_epochs", "num_mini_batches"):
        if type(algorithm.get(key)) is not int or algorithm[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("actor_hidden_dims", "critic_hidden_dims"):
        dims = config["policy"].get(key)
        if (
            not isinstance(dims, list)
            or not dims
            or any(type(n) is not int or n <= 0 for n in dims)
        ):
            raise ValueError(f"{key} must contain positive integers")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _finite_number(value: Any, name: str, *, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_runtime_timing(cfg: Any, runtime: dict) -> None:
    """Apply one validated timing contract before physics construction."""
    physics_dt = _finite_number(runtime.get("physics_dt_s"), "physics_dt_s", nonnegative=True)
    control_dt = _finite_number(runtime.get("control_dt_s"), "control_dt_s", nonnegative=True)
    decimation = runtime.get("control_decimation")
    if (physics_dt <= 0 or type(decimation) is not int or decimation < 1
            or not math.isclose(control_dt, physics_dt * decimation, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError("runtime timing must have positive dt and integral consistent decimation")
    cfg.sim.dt = physics_dt
    cfg.decimation = decimation
    cfg.sim.render_interval = decimation


def verify_physics_sample_stability(entry: dict, task: dict, decimation: int) -> bool:
    """Recompute every physics sample, including the final control endpoint."""
    import numpy as np
    from rby1_assembly.rj45_task import relative_mating_pose, relative_mating_velocity

    def array(value, shape, name):
        try:
            raw = np.asarray(value)
            if raw.dtype.kind not in "iuf":
                raise ValueError("non-numeric array")
            result = raw.astype(np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(f"physics sample {name} must be numeric") from error
        if result.shape != shape or not np.isfinite(result).all():
            raise ValueError(f"physics sample {name} has invalid shape or non-finite values")
        return result

    poses = array(entry.get("object_substep_poses_world_wxyz"), (decimation, 2, 7), "poses")
    velocities = array(entry.get("object_substep_link_velocities_world"), (decimation, 2, 6), "link velocities")
    endpoint = _mapping(entry.get("object_poses_world_wxyz"), "endpoint poses")
    end_poses = array([endpoint.get("socket"), endpoint.get("plug")], (2, 7), "endpoint poses")
    end_velocities = array(entry.get("object_link_velocities_world"), (2, 6), "endpoint link velocities")
    poses = np.concatenate((poses, end_poses[None]), axis=0)
    velocities = np.concatenate((velocities, end_velocities[None]), axis=0)
    if np.any(np.abs(np.linalg.norm(poses[:, :, 3:], axis=-1) - 1.0) > 1e-4):
        raise ValueError("physics sample quaternions must be unit length")
    frames, limits = task["frames"], task["evaluation"]
    stable = True
    for pose, velocity in zip(poses, velocities):
        position, quaternion = relative_mating_pose(
            pose[0, :3], pose[0, 3:], pose[1, :3], pose[1, 3:],
            frames["socket_entry"], frames["plug_tip"],
        )
        linear, angular = relative_mating_velocity(
            pose[0, :3], pose[0, 3:], velocity[0, :3], velocity[0, 3:],
            pose[1, :3], pose[1, 3:], velocity[1, :3], velocity[1, 3:],
            np.asarray(frames["socket_entry"]["position_m"]),
            np.asarray(frames["plug_tip"]["position_m"]),
        )
        angle = 2 * np.arctan2(np.linalg.norm(quaternion[1:]), abs(quaternion[0]))
        stable = stable and bool(
            limits["assembled_depth_m"] <= position[1] <= limits["max_depth_m"]
            and np.linalg.norm(position[[0, 2]]) <= limits["max_lateral_error_m"]
            and angle <= limits["max_orientation_error_rad"]
            and np.linalg.norm(linear) <= limits["max_relative_linear_speed_m_s"]
            and np.linalg.norm(angular) <= limits["max_relative_angular_speed_rad_s"]
        )
    stability = _mapping(entry.get("mating_stability"), "mating stability")
    return bool(stable and stability.get("physics_substeps_complete") is True
                and stability.get("physics_all_samples_stable") is True)


def verify_rj45_nominal_evidence(project: Path, task: dict, report_path: Path) -> dict:
    """Require current-source, recorded friction pickup and independent insertion success."""
    from rby1_assembly.rj45_task import (
        verify_generated_asset_hashes,
        verify_recorded_mating_geometry,
        verify_runtime_versions,
    )

    project = Path(project).resolve()
    report_path = Path(report_path).resolve()
    report = load_finite_json(report_path)
    if (
        report.get("task_id") != "rj45-v1"
        or report.get("completed") is not True
        or report.get("runtime_validated") is not True
        or report.get("num_envs") != 1
        or report.get("command_fault_experiment") is not None
        or report.get("self_collision_enabled") is not True
    ):
        raise ValueError("RJ45 training requires a completed single-environment nominal physical run")
    provenance = _mapping(report.get("provenance"), "nominal report provenance")
    if provenance.get("input_immutability_verified") is not True:
        raise ValueError("nominal evidence requires verified input immutability")
    runtime_path = project / "configs/runtime.json"
    assets_config_path = project / "configs/assets.json"
    connector_assets_path = project / "configs/rj45-assets.json"
    runtime = _mapping(load_finite_json(runtime_path), "runtime configuration")
    expected_config_hashes = {
        "runtime_config_sha256": _sha256(runtime_path),
        "assets_config_sha256": _sha256(assets_config_path),
        "connector_assets_config_sha256": _sha256(connector_assets_path),
    }
    for name, expected_sha256 in expected_config_hashes.items():
        if provenance.get(name) != expected_sha256:
            raise ValueError(f"nominal evidence {name} is stale")
    verify_runtime_versions(
        _mapping(provenance.get("runtime_versions"), "nominal runtime versions"),
        runtime,
    )
    physics_dt_s = _finite_number(
        provenance.get("physics_dt_s"), "nominal physics dt", nonnegative=True
    )
    control_dt_s = _finite_number(
        provenance.get("control_dt_s"), "nominal control dt", nonnegative=True
    )
    control_decimation = provenance.get("control_decimation")
    if (
        physics_dt_s != runtime.get("physics_dt_s")
        or type(control_decimation) is not int
        or control_decimation != runtime.get("control_decimation")
        or control_dt_s != runtime.get("control_dt_s")
        or control_dt_s != physics_dt_s * control_decimation
    ):
        raise ValueError("nominal evidence physics and control timing is stale")
    if provenance.get("task_config_sha256") != task["_config_sha256"]:
        raise ValueError("nominal evidence task hash is stale")
    build_path = project / task["assets"]["build_report"]
    if provenance.get("build_report_sha256") != _sha256(build_path):
        raise ValueError("nominal evidence build hash is stale")
    assets = verify_generated_asset_hashes(project, task, load_finite_json(build_path))
    if provenance.get("generated_asset_sha256") != assets:
        raise ValueError("nominal evidence generated assets differ")
    sources = sorted((project / "src/rby1_assembly").glob("*.py")) + [
        project / "scripts/run_sim.py"
    ]
    expected = {str(path.relative_to(project)): _sha256(path) for path in sources}
    if provenance.get("source_sha256") != expected:
        raise ValueError("nominal evidence source hashes are stale")
    if report.get("trace_every_control_steps") != 1 or not report.get("trace_path"):
        raise ValueError("nominal evidence requires every control step in a trace")
    trace_path = Path(report["trace_path"])
    if not trace_path.is_absolute():
        trace_path = project / trace_path
    if report.get("trace_sha256") != _sha256(trace_path):
        raise ValueError("nominal evidence trace hash mismatch")
    limits = task["evaluation"]
    forbidden_threshold = _finite_number(
        report.get("forbidden_contact_force_threshold_n"),
        "forbidden contact threshold",
        nonnegative=True,
    )
    if forbidden_threshold > limits["max_contact_force_n"]:
        raise ValueError("forbidden contact threshold exceeds the task force limit")
    required_stability = (
        "relative_linear_speed_ok",
        "relative_angular_speed_ok",
        "depth_within_window",
    )
    instantaneous_criteria = (
        "depth",
        "lateral",
        "orientation",
        "grasp_live_and_proven",
        "force",
        "forbidden_contact",
    )

    def eligible(entry: Any) -> bool:
        entry = _mapping(entry, "nominal trace environment")
        if (
            entry.get("independent_eval_valid") is not True
            or not isinstance(entry.get("mating_stability"), dict)
            or not isinstance(entry.get("independent_eval_criteria"), dict)
        ):
            return False
        poses = _mapping(
            entry.get("object_poses_world_wxyz"), "nominal raw object poses"
        )
        geometry = verify_recorded_mating_geometry(
            poses["socket"],
            poses["plug"],
            task["frames"]["socket_entry"],
            task["frames"]["plug_tip"],
            entry["actual_depth_m"],
            entry["actual_lateral_error_m"],
            entry["actual_orientation_error_rad"],
        )
        if entry.get("independent_mating_geometry") != geometry:
            raise ValueError(
                "recorded independent mating geometry differs from raw-pose recalculation"
            )
        stability = _mapping(entry.get("mating_stability"), "mating stability")
        criteria = _mapping(
            entry.get("independent_eval_criteria"), "independent evaluation criteria"
        )
        linear_speed = _finite_number(
            stability.get("relative_linear_speed_m_s"),
            "relative linear speed",
            nonnegative=True,
        )
        angular_speed = _finite_number(
            stability.get("relative_angular_speed_rad_s"),
            "relative angular speed",
            nonnegative=True,
        )
        grip_peak = _finite_number(
            entry.get("grip_contact_force_peak_n"),
            "grip contact force peak",
            nonnegative=True,
        )
        insertion_peak = _finite_number(
            entry.get("insertion_contact_load_peak_n"),
            "insertion contact load peak",
            nonnegative=True,
        )
        forbidden_peak = _finite_number(
            entry.get("forbidden_contact_load_peak_n"),
            "forbidden contact load peak",
            nonnegative=True,
        )
        return (
            limits["assembled_depth_m"] <= geometry["depth_m"] <= limits["max_depth_m"]
            and geometry["lateral_error_m"] <= limits["max_lateral_error_m"]
            and geometry["orientation_error_rad"] <= limits["max_orientation_error_rad"]
            and entry.get("phase_name") in ("insert", "done")
            and verify_physics_sample_stability(entry, task, control_decimation)
            and entry.get("grip_live") == [True, True]
            and entry.get("grip_proven") == [True, True]
            and entry.get("independent_eval_valid") is True
            and all(stability.get(flag) is True for flag in required_stability)
            and all(criteria.get(flag) is True for flag in instantaneous_criteria)
            and linear_speed <= limits["max_relative_linear_speed_m_s"]
            and angular_speed <= limits["max_relative_angular_speed_rad_s"]
            and forbidden_peak <= forbidden_threshold
            and grip_peak <= limits["max_contact_force_n"]
            and insertion_peak <= limits["max_contact_force_n"]
        )
    candidate_since = None
    previous_time = -1.0
    final = None
    count = 0
    for index, line in enumerate(trace_path.read_text(encoding="utf-8").splitlines()):
        record = _mapping(
            _loads_finite_json(line, f"{trace_path}:line {index + 1}"),
            "nominal trace record",
        )
        environments = record.get("environments")
        if (
            record.get("schema_version") != 1
            or record.get("control_step") != index
            or not isinstance(environments, list)
            or len(environments) != 1
        ):
            raise ValueError("nominal trace must contain contiguous one-environment control steps")
        final = _mapping(environments[0], "nominal trace environment")
        if final.get("env_index") != 0:
            raise ValueError("nominal trace environment index must be zero")
        now = _finite_number(
            final.get("timestamp_s"), "nominal trace timestamp", nonnegative=True
        )
        expected_time = index * control_dt_s
        if not math.isclose(now, expected_time, rel_tol=0.0, abs_tol=1.0e-5):
            raise ValueError(
                "nominal trace timestamp must match contiguous control_dt steps"
            )
        if now <= previous_time:
            raise ValueError("nominal trace time must increase")
        previous_time = now
        if eligible(final):
            candidate_since = now if candidate_since is None else candidate_since
        else:
            candidate_since = None
        count += 1
    terminal = report.get("terminal_outcomes")
    terminal_entry = (
        terminal[0] if isinstance(terminal, list) and len(terminal) == 1 else None
    )
    if terminal_entry is not None:
        terminal_entry = _mapping(terminal_entry, "nominal terminal outcome")
    terminal_matches_trace = bool(
        final
        and terminal_entry
        and all(final.get(key) == value for key, value in terminal_entry.items())
    )
    final_criteria = (
        terminal_entry.get("independent_eval_criteria", {}) if terminal_entry else {}
    )
    final_dwell = (
        _finite_number(
            terminal_entry.get("independent_eval_dwell_s"),
            "independent evaluation dwell",
            nonnegative=True,
        )
        if terminal_entry
        else -1.0
    )
    if (
        not final
        or terminal_entry is None
        or not terminal_matches_trace
        or not eligible(terminal_entry)
        or final.get("phase_name") != "done"
        or final.get("terminal") is not True
        or terminal_entry.get("termination_reason") != "independent_success"
        or final.get("independent_eval_valid") is not True
        or final.get("independent_eval_success") is not True
        or final_dwell < limits["dwell_s"]
        or not isinstance(final_criteria, dict)
        or set(final_criteria) != {*instantaneous_criteria, "dwell"}
        or not all(value is True for value in final_criteria.values())
        or terminal_entry.get("pregrasp_clearance_reached") is not True
        or terminal_entry.get("pregrasp_above_reached") is not True
        or candidate_since is None
        or previous_time - candidate_since + 1e-6 < limits["dwell_s"]
        or type(report.get("trace_record_count")) is not int
        or type(report.get("actual_steps")) is not int
        or count != report["trace_record_count"]
        or count != report["actual_steps"] + 1
    ):
        raise ValueError("nominal evidence lacks continuous independent grasp/lift/insertion success")
    evaluator = _mapping(report.get("independent_evaluator"), "independent evaluator")
    if evaluator != {
        "source": "simulator_ground_truth_independent",
        "version": "tip-entry-geometry-contact-v1",
    }:
        raise ValueError("nominal evidence uses an unsupported independent evaluator")
    return {
        "report_sha256": _sha256(report_path),
        "trace_sha256": _sha256(trace_path),
        "trace_records": count,
    }
