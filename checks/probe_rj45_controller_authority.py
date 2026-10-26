#!/usr/bin/env python3
"""Measure RJ45 residual-controller authority from one verified held state.

This is a diagnostic A/B probe.  It fixes one synthetic estimator bias and
applies its analytic inverse action; it is not a learned-policy or sensor
performance evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import traceback
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROBE_TRANSLATION_BIAS_ENTRY_M = (100.0e-6, 0.0, 0.0)
PROBE_ROTATION_BIAS_ENTRY_RAD = (0.0, 0.0, 0.0)
HOLD_DURATION_S = 3.0
HOLD_REQUIRED_DWELL_S = 0.4
HOLD_MAX_LATERAL_ERROR_M = 50.0e-6
HOLD_MAX_ORIENTATION_ERROR_RAD = math.radians(0.5)
HOLD_MAX_RELATIVE_LINEAR_SPEED_M_S = 0.005
HOLD_MAX_RELATIVE_ANGULAR_SPEED_RAD_S = 0.05
HOLD_MAX_LOAD_N = 25.0
HOLD_DEPTH_BAND_M = 0.00025
NORMALIZED_ACTION_L2_TOL = 1.0e-6
TIP_OFFSET_L2_TOL_M_RAD = 1.0e-9


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_project_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT and not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"input must stay inside the project: {resolved}")
    return resolved


def _validate_output_paths(
    report: Path, trace: Path, *, protected_paths: tuple[Path, ...]
) -> tuple[Path, Path]:
    artifacts = (PROJECT_ROOT / "artifacts").resolve()
    report = report.resolve()
    trace = trace.resolve()
    if report.suffix.lower() != ".json":
        raise ValueError("--report must be a JSON file")
    if trace.suffix.lower() != ".jsonl":
        raise ValueError("--trace must be a JSONL file")
    if report == trace:
        raise ValueError("--report and --trace must be different files")
    protected = {path.resolve() for path in protected_paths}
    for label, path in (("report", report), ("trace", trace)):
        if path == artifacts or not path.is_relative_to(artifacts):
            raise ValueError(f"--{label} must stay below the project artifacts directory")
        if path in protected:
            raise ValueError(f"--{label} must not overwrite an input or source file: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"--{label} must identify a regular file: {path}")
    return report, trace


def _validate_probe_training_config(config: Mapping[str, Any]) -> str:
    mode = config.get("action_application_mode")
    if mode not in (
        "post_integral_step",
        "absolute_tip_pose_target_insert_only",
    ):
        raise ValueError("probe requires one supported action_application_mode")
    if config.get("action_mode") != "right_only_6d":
        raise ValueError("probe requires right_only_6d")
    if config.get("base_observation_dim") != 79:
        raise ValueError("probe requires the 79D base observation")
    if config.get("diagnostic_observation_mode") != "standard":
        raise ValueError("probe rejects privileged diagnostic observations")
    bias = config.get("calibration_bias")
    if not isinstance(bias, Mapping):
        raise ValueError("probe training configuration lacks calibration_bias")
    levels = bias.get("translation_levels_m")
    level = bias.get("initial_level")
    if (
        not isinstance(levels, list)
        or type(level) is not int
        or level < 0
        or level >= len(levels)
        or isinstance(levels[level], bool)
        or not isinstance(levels[level], (int, float))
        or not math.isfinite(float(levels[level]))
        or float(levels[level]) < abs(PROBE_TRANSLATION_BIAS_ENTRY_M[0])
    ):
        raise ValueError("configured calibration level cannot contain the probe bias")
    return str(mode)


def _analytic_probe_action(translation_bound_m: float) -> tuple[float, ...]:
    if (
        isinstance(translation_bound_m, bool)
        or not isinstance(translation_bound_m, (int, float))
        or not math.isfinite(float(translation_bound_m))
        or float(translation_bound_m) <= 0.0
    ):
        raise ValueError("translation bound must be finite and positive")
    normalized_x = PROBE_TRANSLATION_BIAS_ENTRY_M[0] / float(translation_bound_m)
    if abs(normalized_x) > 1.0:
        raise ValueError("analytic probe action exceeds the normalized action bound")
    return (normalized_x, 0.0, 0.0, 0.0, 0.0, 0.0)


def _advance_hold_dwell(
    current_dwell_s: float, sample_ok: bool, step_dt_s: float
) -> float:
    for name, value in (("current_dwell_s", current_dwell_s), ("step_dt_s", step_dt_s)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if current_dwell_s < 0.0 or step_dt_s <= 0.0:
        raise ValueError("dwell must be non-negative and step_dt_s must be positive")
    if type(sample_ok) is not bool:
        raise ValueError("sample_ok must be bool")
    return float(current_dwell_s + step_dt_s) if sample_ok else 0.0


def _vector_matches(
    actual: list[float], expected: list[float], *, l2_tolerance: float
) -> bool:
    if len(actual) != len(expected):
        return False
    if not math.isfinite(l2_tolerance) or l2_tolerance < 0.0:
        raise ValueError("l2_tolerance must be finite and non-negative")
    differences = []
    for observed, wanted in zip(actual, expected, strict=True):
        if not math.isfinite(float(observed)) or not math.isfinite(float(wanted)):
            return False
        differences.append(float(observed) - float(wanted))
    return math.sqrt(sum(value * value for value in differences)) <= l2_tolerance


def _action_application_status(
    *,
    requested_normalized_12d: list[float],
    applied_normalized_12d: list[float],
    applied_tip_offset_6d: list[float],
    expected_normalized_12d: list[float],
    expected_tip_offset_6d: list[float],
) -> dict[str, bool]:
    """Compare commanded and applied authority using explicit float32 tolerances."""

    return {
        "requested_normalized_action_matches": _vector_matches(
            requested_normalized_12d,
            expected_normalized_12d,
            l2_tolerance=NORMALIZED_ACTION_L2_TOL,
        ),
        "applied_normalized_action_matches": _vector_matches(
            applied_normalized_12d,
            expected_normalized_12d,
            l2_tolerance=NORMALIZED_ACTION_L2_TOL,
        ),
        "applied_tip_offset_matches": _vector_matches(
            applied_tip_offset_6d,
            expected_tip_offset_6d,
            l2_tolerance=TIP_OFFSET_L2_TOL_M_RAD,
        ),
    }


def _release_boundary_status(
    evaluation_dwell_s: float,
    evaluation_success: bool,
    success_since_step: int,
) -> bool:
    if not math.isfinite(evaluation_dwell_s) or evaluation_dwell_s < 0.0:
        raise ValueError("evaluation_dwell_s must be finite and non-negative")
    if type(evaluation_success) is not bool:
        raise ValueError("evaluation_success must be bool")
    if type(success_since_step) is not int:
        raise ValueError("success_since_step must be int")
    return (
        evaluation_dwell_s == 0.0
        and evaluation_success is False
        and success_since_step == -1
    )


def _authority_evidence_gate(
    action_application_passed: bool, first_policy_input_valid: bool
) -> bool:
    if type(action_application_passed) is not bool:
        raise ValueError("action_application_passed must be bool")
    if type(first_policy_input_valid) is not bool:
        raise ValueError("first_policy_input_valid must be bool")
    return action_application_passed and first_policy_input_valid


def _validate_substep_contact_samples(
    insertion_load_n: list[float],
    forbidden_load_n: list[float],
    finger_force_n: list[list[float]],
    *,
    expected_count: int,
) -> dict[str, float]:
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    if not (
        len(insertion_load_n)
        == len(forbidden_load_n)
        == len(finger_force_n)
        == expected_count
    ):
        raise ValueError("substep contact sample count differs from decimation + endpoint")
    if any(not isinstance(row, list) or len(row) != 4 for row in finger_force_n):
        raise ValueError("each substep finger-force sample must contain four values")
    flat_fingers = [value for row in finger_force_n for value in row]
    all_values = [*insertion_load_n, *forbidden_load_n, *flat_fingers]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in all_values
    ):
        raise ValueError("substep contact samples must be finite numbers")
    return {
        "insertion_load_max_n": max(float(value) for value in insertion_load_n),
        "forbidden_load_max_n": max(float(value) for value in forbidden_load_n),
        "finger_force_min_n": min(float(value) for value in flat_fingers),
        "finger_force_max_n": max(float(value) for value in flat_fingers),
    }


def _substep_contact_hold_ok(
    summary: Mapping[str, float], *, grip_threshold_n: float, maximum_force_n: float
) -> bool:
    if not all(
        math.isfinite(float(value))
        for value in (*summary.values(), grip_threshold_n, maximum_force_n)
    ):
        raise ValueError("substep contact thresholds and summary must be finite")
    if grip_threshold_n < 0.0 or maximum_force_n <= 0.0:
        raise ValueError("substep contact thresholds are invalid")
    return (
        summary["insertion_load_max_n"] == 0.0
        and summary["forbidden_load_max_n"] == 0.0
        and summary["finger_force_min_n"] >= grip_threshold_n
        and summary["finger_force_max_n"] <= maximum_force_n
    )


def _finite_json_value(value: Any) -> Any:
    """Convert tensors/tuples recursively and reject non-finite evidence."""

    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        value = value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _finite_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite_json_value(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("probe evidence contains a non-finite number")
        return value
    raise TypeError(f"probe evidence contains unsupported type {type(value).__name__}")


class _IncrementalJsonlTrace:
    """Flush every record and atomically publish even a partial diagnostic trace."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        self._temporary = Path(temporary_name)
        self._stream = os.fdopen(descriptor, "w", encoding="utf-8")
        self.records = 0
        self._closed = False

    def append(self, row: Mapping[str, Any]) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("trace is already closed")
        encoded = _finite_json_value(row)
        self._stream.write(json.dumps(encoded, sort_keys=True, allow_nan=False))
        self._stream.write("\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self.records += 1
        return encoded

    def close(self, *, publish: bool = True) -> None:
        if self._closed:
            return
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        if publish:
            os.replace(self._temporary, self.path)
        else:
            self._temporary.unlink()
        self._closed = True


def _atomic_write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    trace = _IncrementalJsonlTrace(path)
    complete = False
    try:
        for row in rows:
            trace.append(row)
        complete = True
    finally:
        trace.close(publish=complete)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe RJ45 residual-controller authority from a verified held state"
    )
    parser.add_argument("--nominal", type=Path, required=True)
    parser.add_argument("--held", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args(argv)


def _run(args: argparse.Namespace) -> int:
    from rby1_assembly.run_provenance import capture_run_inputs

    task_path = PROJECT_ROOT / "configs/tasks/rj45-v1.json"
    runtime_path = PROJECT_ROOT / "configs/runtime.json"
    assets_path = PROJECT_ROOT / "configs/assets.json"
    rj45_assets_path = PROJECT_ROOT / "configs/rj45-assets.json"
    nominal_path = _resolved_project_path(args.nominal)
    held_path = _resolved_project_path(args.held)
    training_path = _resolved_project_path(args.training_config)

    nominal_bootstrap = json.loads(nominal_path.read_text(encoding="utf-8"))
    if not isinstance(nominal_bootstrap, dict):
        raise ValueError("nominal report must be a JSON object")
    nominal_trace_path = _resolved_project_path(nominal_bootstrap.get("trace_path", ""))
    held_bootstrap = json.loads(held_path.read_text(encoding="utf-8"))
    held_provenance = held_bootstrap.get("provenance") if isinstance(held_bootstrap, dict) else None
    if not isinstance(held_provenance, dict):
        raise ValueError("held artifact provenance is missing")
    held_nominal_report = _resolved_project_path(held_provenance.get("nominal_report_path", ""))
    held_nominal_trace = _resolved_project_path(held_provenance.get("nominal_trace_path", ""))
    protected = (
        task_path,
        runtime_path,
        assets_path,
        rj45_assets_path,
        nominal_path,
        nominal_trace_path,
        held_path,
        held_nominal_report,
        held_nominal_trace,
        training_path,
        Path(__file__).resolve(),
    )
    requested_report = Path(args.report)
    requested_trace = Path(args.trace)
    report_path, trace_path = _validate_output_paths(
        requested_report if requested_report.is_absolute() else PROJECT_ROOT / requested_report,
        requested_trace if requested_trace.is_absolute() else PROJECT_ROOT / requested_trace,
        protected_paths=protected,
    )
    args.validated_report_path = report_path
    bootstrap = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(runtime_path, training_path, assets_path, rj45_assets_path),
        source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
        source_files=(Path(__file__).resolve(),),
        extra_paths=(
            nominal_path,
            nominal_trace_path,
            held_path,
            held_nominal_report,
            held_nominal_trace,
        ),
    )
    bootstrap.assert_unchanged()
    report_path, trace_path = _validate_output_paths(
        report_path,
        trace_path,
        protected_paths=tuple(
            Path(key) if Path(key).is_absolute() else PROJECT_ROOT / key
            for key in bootstrap.sha256
        ),
    )

    from rby1_assembly.experiment_contracts import (
        apply_runtime_timing,
        atomic_write_json,
        load_finite_json,
        verify_rj45_nominal_evidence,
    )
    from rby1_assembly.held_grasp import load_verified_held_grasp_artifact
    from rby1_assembly.rj45_task import load_rj45_task, verify_runtime_versions
    from rby1_assembly.rj45_training import load_training_config
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(add_help=False)
    AppLauncher.add_app_launcher_args(parser)
    launcher_args = parser.parse_args(
        ["--device", args.device, "--headless"] if args.headless else ["--device", args.device]
    )

    current = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(runtime_path, training_path, assets_path, rj45_assets_path),
        source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
        source_files=(Path(__file__).resolve(),),
        extra_paths=(
            nominal_path,
            nominal_trace_path,
            held_path,
            held_nominal_report,
            held_nominal_trace,
        ),
    )
    bootstrap.assert_unchanged()
    bootstrap.assert_preserved_in(current)
    task = load_rj45_task(task_path)
    runtime = current.json_for(runtime_path)
    training_config = load_training_config(training_path)
    mode = _validate_probe_training_config(training_config)
    nominal_proof = verify_rj45_nominal_evidence(PROJECT_ROOT, task, nominal_path)
    held_state, held_proof = load_verified_held_grasp_artifact(PROJECT_ROOT, task, held_path)
    if current.json_for(nominal_path) != load_finite_json(nominal_path):
        raise ValueError("nominal report changed during bootstrap")
    if current.json_for(held_path) != load_finite_json(held_path):
        raise ValueError("held artifact changed during bootstrap")
    physics_dt = float(runtime["physics_dt_s"])
    control_dt = float(runtime["control_dt_s"])
    if not math.isclose(physics_dt, 1.0 / 240.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"probe requires 240 Hz physics, got dt={physics_dt}")
    if not math.isclose(control_dt, 1.0 / 30.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError(f"probe requires 30 Hz control, got dt={control_dt}")

    simulation_app = AppLauncher(launcher_args).app
    args.physics_executed = True
    try:
        import omni.kit.app
        import torch
        from rby1_assembly.experiment_contracts import atomic_write_json
        from rby1_assembly.isaac_env import AssemblyPhase, Rby1AssemblyEnvCfg
        from rby1_assembly.rj45_env import configure_rj45_task
        from rby1_assembly.rj45_training_env import make_rj45_training_env_type

        versions = {
            name: metadata.version(name)
            for name in (
                "isaacsim",
                "isaaclab",
                "isaaclab-rl",
                "rsl-rl-lib",
                "torch",
                "numpy",
                "h5py",
            )
        }
        verify_runtime_versions(versions, runtime)
        cfg = Rby1AssemblyEnvCfg()
        apply_runtime_timing(cfg, runtime)
        cfg.scene.num_envs = 1
        cfg.sim.device = args.device
        cfg.seed = int(task["data"]["evaluation_seed"])
        env_type = configure_rj45_task(cfg, task, PROJECT_ROOT)
        env_type = make_rj45_training_env_type(
            env_type,
            training_config,
            held_state=held_state,
            calibration_bias_enabled=True,
            bias_level=training_config["calibration_bias"]["initial_level"],
        )

        class ProbeEnv(env_type):
            def _sample_episode_bias(self, env_ids) -> None:
                self._training_translation_bias_entry[env_ids] = torch.tensor(
                    PROBE_TRANSLATION_BIAS_ENTRY_M,
                    device=self.device,
                    dtype=self._training_translation_bias_entry.dtype,
                )
                self._training_rotation_bias_entry[env_ids] = torch.tensor(
                    PROBE_ROTATION_BIAS_ENTRY_RAD,
                    device=self.device,
                    dtype=self._training_rotation_bias_entry.dtype,
                )

            def _pre_physics_step(self, actions) -> None:
                if self._training_initialized:
                    self._probe_insertion_contact_load_samples = []
                    self._probe_forbidden_contact_load_samples = []
                    self._probe_finger_force_samples = []
                super()._pre_physics_step(actions)

            def _append_probe_contact_sample(self) -> None:
                for name in (
                    "_probe_insertion_contact_load_samples",
                    "_probe_forbidden_contact_load_samples",
                    "_probe_finger_force_samples",
                ):
                    if not hasattr(self, name):
                        raise RuntimeError(
                            "probe contact sampling was not initialized by _pre_physics_step"
                        )
                insertion_load = self._insertion_contact_load()
                forbidden_load = self._forbidden_contact_load()
                finger_force = torch.stack(
                    (
                        self._filtered_force_magnitude(self.left_contact_l1),
                        self._filtered_force_magnitude(self.left_contact_l2),
                        self._filtered_force_magnitude(self.right_contact_r1),
                        self._filtered_force_magnitude(self.right_contact_r2),
                    ),
                    dim=1,
                )
                if (
                    insertion_load.shape != (self.num_envs,)
                    or forbidden_load.shape != (self.num_envs,)
                    or finger_force.shape != (self.num_envs, 4)
                    or not bool(torch.isfinite(insertion_load).all().item())
                    or not bool(torch.isfinite(forbidden_load).all().item())
                    or not bool(torch.isfinite(finger_force).all().item())
                ):
                    raise RuntimeError(
                        "probe contact sample has an invalid shape or non-finite value"
                    )
                self._probe_insertion_contact_load_samples.append(
                    insertion_load.clone()
                )
                self._probe_forbidden_contact_load_samples.append(
                    forbidden_load.clone()
                )
                self._probe_finger_force_samples.append(finger_force.clone())

            def _apply_action(self) -> None:
                super()._apply_action()
                if self._training_initialized:
                    self._append_probe_contact_sample()

            def _get_dones(self):
                if self._training_initialized:
                    sample_list_names = (
                        "_probe_insertion_contact_load_samples",
                        "_probe_forbidden_contact_load_samples",
                        "_probe_finger_force_samples",
                    )
                    if any(not hasattr(self, name) for name in sample_list_names):
                        raise RuntimeError(
                            "probe contact lists are missing before terminal snapshot capture"
                        )
                    expected_pre_step_samples = int(self.cfg.decimation)
                    counts = (
                        len(self._probe_insertion_contact_load_samples),
                        len(self._probe_forbidden_contact_load_samples),
                        len(self._probe_finger_force_samples),
                    )
                    if counts != (expected_pre_step_samples,) * 3:
                        raise RuntimeError(
                            "probe pre-step contact count differs from cfg.decimation: "
                            f"counts={counts}, decimation={expected_pre_step_samples}"
                        )
                    self._append_probe_contact_sample()
                terminated, truncated = super()._get_dones()
                snapshot = self.extras.get("terminal_snapshot")
                if self._training_initialized and isinstance(snapshot, dict):
                    actual_pos, actual_quat = self._relative_object_pose()
                    estimated_pos, estimated_quat = self._control_relative_object_pose()
                    linear, angular = self._relative_mating_velocity()
                    snapshot["authority_probe"] = {
                        "actual_relative_position_entry_m": actual_pos.clone(),
                        "actual_relative_quaternion_entry_wxyz": actual_quat.clone(),
                        "estimated_relative_position_entry_m": estimated_pos.clone(),
                        "estimated_relative_quaternion_entry_wxyz": estimated_quat.clone(),
                        "relative_linear_velocity_world_m_s": linear.clone(),
                        "relative_angular_velocity_world_rad_s": angular.clone(),
                        "phase_started_s": self._phase_started_s.clone(),
                        "success_since_step": self._success_since_step.clone(),
                        "insert_joint_reference_si": self._insert_joint_targets.clone(),
                        "final_command_joint_reference_si": self._joint_targets.clone(),
                        "applied_normalized_action_12d": self._actions.clone(),
                        "applied_tip_target_offset_entry_6d": (
                            self._training_applied_tip_offset_entry.clone()
                        ),
                        "physics_contact_insertion_load_n": torch.stack(
                            self._probe_insertion_contact_load_samples, dim=1
                        ).clone(),
                        "physics_contact_forbidden_load_n": torch.stack(
                            self._probe_forbidden_contact_load_samples, dim=1
                        ).clone(),
                        "physics_contact_finger_force_n": torch.stack(
                            self._probe_finger_force_samples, dim=1
                        ).clone(),
                    }
                return terminated, truncated

        env = ProbeEnv(cfg)
        trace_writer = _IncrementalJsonlTrace(trace_path)
        args.trace_path = trace_path
        action_application_checks: list[dict[str, bool]] = []
        try:
            env.reset()
            action_values = _analytic_probe_action(float(env.cfg.residual_translation_bound_m))
            analytic_action = torch.tensor(
                [action_values], device=env.device, dtype=torch.float32
            )
            zero_action = torch.zeros_like(analytic_action)
            expected_normalized_action_12d = torch.cat(
                (torch.zeros_like(analytic_action), analytic_action), dim=1
            )
            expected_tip_offset_6d = [
                PROBE_TRANSLATION_BIAS_ENTRY_M[0],
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ]
            if mode == "post_integral_step":
                expected_applied_tip_offset_6d = [0.0] * 6
            else:
                expected_applied_tip_offset_6d = expected_tip_offset_6d
            control_index = 0

            def record_step(
                stage: str, requested_action, terminated, truncated
            ) -> dict[str, Any]:
                nonlocal control_index
                snapshot = env.extras.get("terminal_snapshot")
                if not isinstance(snapshot, dict):
                    raise RuntimeError("environment omitted the pre-reset terminal snapshot")
                probe = snapshot.get("authority_probe")
                stability = snapshot.get("rj45_evaluation_stability")
                if not isinstance(probe, dict) or not isinstance(stability, dict):
                    raise RuntimeError("probe snapshot lacks pose or raw stability evidence")
                phase_code = int(snapshot["phase"][0].item())
                requested_normalized_action_12d = torch.cat(
                    (torch.zeros_like(requested_action), requested_action), dim=1
                )[0]
                expected_for_stage = (
                    expected_normalized_action_12d[0]
                    if stage != "held_initialization"
                    else torch.zeros_like(expected_normalized_action_12d[0])
                )
                expected_tip_for_stage = (
                    expected_applied_tip_offset_6d
                    if stage != "held_initialization"
                    else [0.0] * 6
                )
                row = {
                    "control_index": control_index,
                    "stage": stage,
                    "episode_index": int(snapshot["episode_index"][0].item()),
                    "episode_time_s": float(snapshot["timestamp_s"][0].item()),
                    "simulation_time_s": float(env._sim_time_s()),
                    "phase_code": phase_code,
                    "phase_name": snapshot["phase_name_by_code"][phase_code],
                    "phase_started_s": float(probe["phase_started_s"][0].item()),
                    "success_since_step": int(
                        probe["success_since_step"][0].item()
                    ),
                    "terminated": bool(terminated[0].item()),
                    "truncated": bool(truncated[0].item()),
                    "terminal_mask": bool(snapshot["terminal_mask"][0].item()),
                    "actual_relative_position_entry_m": probe[
                        "actual_relative_position_entry_m"
                    ][0],
                    "actual_relative_quaternion_entry_wxyz": probe[
                        "actual_relative_quaternion_entry_wxyz"
                    ][0],
                    "estimated_relative_position_entry_m": probe[
                        "estimated_relative_position_entry_m"
                    ][0],
                    "estimated_relative_quaternion_entry_wxyz": probe[
                        "estimated_relative_quaternion_entry_wxyz"
                    ][0],
                    "requested_normalized_action_12d": requested_normalized_action_12d,
                    "applied_normalized_action_12d": probe[
                        "applied_normalized_action_12d"
                    ][0],
                    "applied_tip_target_offset_entry_6d": probe[
                        "applied_tip_target_offset_entry_6d"
                    ][0],
                    "insertion_command_depth_m": snapshot[
                        "insertion_command_depth_m"
                    ][0],
                    "insertion_advance_enabled": snapshot[
                        "insertion_advance_enabled"
                    ][0],
                    "insert_joint_reference_si": probe[
                        "insert_joint_reference_si"
                    ][0],
                    "final_command_joint_reference_si": probe[
                        "final_command_joint_reference_si"
                    ][0],
                    "actual_joint_position_si": snapshot["all_joint_positions_si"][0],
                    "actual_joint_velocity_si": snapshot["all_joint_velocities_si"][0],
                    "stored_joint_reference_si": snapshot["all_joint_reference_si"][0],
                    "controller_grip_proven": snapshot["controller_grip_proven"][0],
                    "evaluation_grip_proven": snapshot["grip_proven"][0],
                    "grip_live": snapshot["grip_live"][0],
                    "held_initialization_passed": snapshot[
                        "held_initialization_passed"
                    ][0],
                    "policy_input_valid": snapshot["policy_input_valid"][0],
                    "valid_policy_input_step_count": snapshot[
                        "valid_policy_input_step_count"
                    ][0],
                    "finger_contact_force_n": snapshot["finger_contact_force_n"][0],
                    "insertion_contact_load_n": snapshot[
                        "insertion_contact_load_n"
                    ][0],
                    "insertion_contact_load_peak_n": snapshot[
                        "insertion_contact_load_peak_n"
                    ][0],
                    "forbidden_contact_load_n": snapshot[
                        "forbidden_contact_load_n"
                    ][0],
                    "forbidden_contact_load_peak_n": snapshot[
                        "forbidden_contact_load_peak_n"
                    ][0],
                    "relative_linear_velocity_world_m_s": probe[
                        "relative_linear_velocity_world_m_s"
                    ][0],
                    "relative_angular_velocity_world_rad_s": probe[
                        "relative_angular_velocity_world_rad_s"
                    ][0],
                    "physics_sample_lateral_error_m": stability[
                        "physics_sample_lateral_error_m"
                    ][0],
                    "physics_sample_depth_m": stability[
                        "physics_sample_depth_m"
                    ][0],
                    "physics_sample_orientation_error_rad": stability[
                        "physics_sample_orientation_error_rad"
                    ][0],
                    "physics_sample_relative_linear_speed_m_s": stability[
                        "physics_sample_relative_linear_speed_m_s"
                    ][0],
                    "physics_sample_relative_angular_speed_rad_s": stability[
                        "physics_sample_relative_angular_speed_rad_s"
                    ][0],
                    "physics_sample_finite": stability["physics_sample_finite"][0],
                    "physics_contact_insertion_load_n": probe[
                        "physics_contact_insertion_load_n"
                    ][0],
                    "physics_contact_forbidden_load_n": probe[
                        "physics_contact_forbidden_load_n"
                    ][0],
                    "physics_contact_finger_force_n": probe[
                        "physics_contact_finger_force_n"
                    ][0],
                    "independent_eval_valid": snapshot["independent_eval_valid"][0],
                    "independent_eval_success": snapshot[
                        "independent_eval_success"
                    ][0],
                    "independent_eval_dwell_s": snapshot[
                        "independent_eval_dwell_s"
                    ][0],
                    "termination_reason_code": snapshot[
                        "termination_reason_code"
                    ][0],
                    "termination_reason": snapshot["termination_reason_by_code"][
                        int(snapshot["termination_reason_code"][0].item())
                    ],
                }
                control_index += 1
                encoded = _finite_json_value(row)
                expected_physics_samples = int(env.cfg.decimation) + 1
                pose_sample_lengths = (
                    len(encoded["physics_sample_depth_m"]),
                    len(encoded["physics_sample_lateral_error_m"]),
                    len(encoded["physics_sample_orientation_error_rad"]),
                    len(encoded["physics_sample_relative_linear_speed_m_s"]),
                    len(encoded["physics_sample_relative_angular_speed_rad_s"]),
                    len(encoded["physics_sample_finite"]),
                )
                if pose_sample_lengths != (expected_physics_samples,) * len(
                    pose_sample_lengths
                ):
                    raise RuntimeError(
                        "raw pose/stability sample count differs from decimation + endpoint: "
                        f"counts={pose_sample_lengths}, expected={expected_physics_samples}"
                    )
                encoded["physics_contact_summary"] = _validate_substep_contact_samples(
                    encoded["physics_contact_insertion_load_n"],
                    encoded["physics_contact_forbidden_load_n"],
                    encoded["physics_contact_finger_force_n"],
                    expected_count=expected_physics_samples,
                )
                encoded["action_application_status"] = _action_application_status(
                    requested_normalized_12d=encoded[
                        "requested_normalized_action_12d"
                    ],
                    applied_normalized_12d=encoded[
                        "applied_normalized_action_12d"
                    ],
                    applied_tip_offset_6d=encoded[
                        "applied_tip_target_offset_entry_6d"
                    ],
                    expected_normalized_12d=_finite_json_value(expected_for_stage),
                    expected_tip_offset_6d=expected_tip_for_stage,
                )
                if stage != "held_initialization":
                    action_application_checks.append(
                        dict(encoded["action_application_status"])
                    )
                return trace_writer.append(encoded)

            warmup_limit = max(1, math.ceil((float(training_config["warmup_s"]) + 1.0) / env.step_dt))
            warmup_steps = 0
            while not bool(env._training_warmup_complete[0].item()):
                if warmup_steps >= warmup_limit:
                    raise RuntimeError("verified held warmup did not complete within its strict limit")
                _, _, terminated, truncated, _ = env.step(zero_action)
                record_step("held_initialization", zero_action, terminated, truncated)
                if bool((terminated | truncated)[0].item()):
                    raise RuntimeError("environment terminated during held initialization")
                warmup_steps += 1
            if not (
                bool(env._training_held_initialization_passed[0].item())
                and bool(env._training_estimator_valid[0].item())
                and bool(env._training_policy_activated[0].item())
                and bool(env._grip_proven[0].all().item())
                and bool(env._evaluation_grip_proven[0].all().item())
            ):
                raise RuntimeError("held initialization, estimator, or physical grip proof is incomplete")

            held_reference = env._training_held_reference[0].clone()
            reference_checks = {
                "insert": torch.equal(
                    env._insert_joint_targets[0], held_reference[env._arm_joint_ids]
                ),
                "torso": torch.equal(
                    env._torso_hold_targets[0], held_reference[env._torso_joint_ids]
                ),
                "pregrasp": torch.equal(
                    env._pregrasp_servo_targets[0],
                    torch.stack(
                        (
                            held_reference[env._left_arm_joint_ids],
                            held_reference[env._right_arm_joint_ids],
                        )
                    ),
                ),
            }
            if not all(reference_checks.values()):
                raise RuntimeError(f"held controller references changed before probe: {reference_checks}")
            selected = torch.ones(1, dtype=torch.bool, device=env.device)
            env._transition_where(selected, AssemblyPhase.INSERT, env._sim_time_s())
            env._phase_started_s[selected] = env._sim_time_s()
            if int(env._phase[0].item()) != int(AssemblyPhase.INSERT):
                raise RuntimeError("probe failed to enter logical INSERT")
            if not (
                torch.equal(env._insert_joint_targets[0], held_reference[env._arm_joint_ids])
                and torch.equal(env._torso_hold_targets[0], held_reference[env._torso_joint_ids])
                and torch.equal(
                    env._pregrasp_servo_targets[0],
                    torch.stack(
                        (
                            held_reference[env._left_arm_joint_ids],
                            held_reference[env._right_arm_joint_ids],
                        )
                    ),
                )
            ):
                raise RuntimeError("logical INSERT transition changed the held references")

            refreshed_observation = env._get_observations()
            if (
                not isinstance(refreshed_observation, dict)
                or set(refreshed_observation) != {"policy"}
                or tuple(refreshed_observation["policy"].shape) != (1, 79)
                or not bool(torch.isfinite(refreshed_observation["policy"]).all().item())
                or not bool(env._training_last_policy_observation_valid[0].item())
                or not torch.equal(
                    refreshed_observation["policy"],
                    env._training_last_policy_observation,
                )
            ):
                raise RuntimeError(
                    "logical INSERT observation refresh did not produce one finite valid 79D policy input"
                )
            observation_refresh = {
                "method": "formal_get_observations_state_read_after_logical_insert_transition",
                "policy_shape": [1, 79],
                "finite": True,
                "policy_input_valid": True,
                "cache_matches_returned_observation": True,
                "additional_physics_step": False,
                "physical_pose_or_force_write": False,
            }

            bounded_preflight = env._bound_residual_actions(
                expected_normalized_action_12d
            )
            bounded_preflight_error = torch.linalg.vector_norm(
                bounded_preflight - expected_normalized_action_12d, dim=1
            )
            if not bool(
                (bounded_preflight_error <= NORMALIZED_ACTION_L2_TOL)
                .all()
                .item()
            ):
                raise RuntimeError(
                    "analytic +100 um action is clipped or changed by the runtime bound"
                )
            commanded_translation_x = (
                float(analytic_action[0, 0].item())
                * float(env.cfg.residual_translation_bound_m)
            )
            if not math.isclose(
                commanded_translation_x,
                PROBE_TRANSLATION_BIAS_ENTRY_M[0],
                rel_tol=0.0,
                abs_tol=TIP_OFFSET_L2_TOL_M_RAD,
            ):
                raise RuntimeError(
                    "runtime residual bound cannot represent the +100 um probe command"
                )

            original_speed = float(env.cfg.insertion_speed_m_s)
            env.cfg.insertion_speed_m_s = 0.0
            held_estimated_depth = float(env._training_estimated_relative_pos[0, 1].item())
            held_actual_position, _ = env._relative_object_pose()
            held_actual_depth = float(held_actual_position[0, 1].item())
            env._insertion_command_depth_m[0] = held_estimated_depth
            hold_steps = max(1, math.ceil(HOLD_DURATION_S / env.step_dt))
            hold_control_steps_completed = 0
            hold_dwell = 0.0
            max_hold_dwell = 0.0
            final_hold_row = None
            hold_terminal = False
            hold_terminal_reason_code = None
            hold_terminal_reason = None
            first_analytic_policy_input_valid = None
            hold_raw_depth_min_m = None
            hold_raw_depth_max_m = None
            hold_raw_orientation_max_rad = 0.0
            hold_raw_lateral_max_m = 0.0
            hold_endpoint_insertion_load_max_n = 0.0
            hold_endpoint_insertion_peak_max_n = 0.0
            hold_endpoint_forbidden_load_max_n = 0.0
            hold_endpoint_forbidden_peak_max_n = 0.0
            hold_physics_insertion_load_max_n = 0.0
            hold_physics_forbidden_load_max_n = 0.0
            hold_physics_finger_force_min_n = math.inf
            hold_physics_finger_force_max_n = 0.0
            for _ in range(hold_steps):
                _, _, terminated, truncated, _ = env.step(analytic_action)
                row = record_step("authority_hold", analytic_action, terminated, truncated)
                if first_analytic_policy_input_valid is None:
                    first_analytic_policy_input_valid = bool(
                        row["policy_input_valid"]
                    )
                raw_depths = row["physics_sample_depth_m"]
                row_depth_min = min(raw_depths)
                row_depth_max = max(raw_depths)
                hold_raw_depth_min_m = (
                    row_depth_min
                    if hold_raw_depth_min_m is None
                    else min(hold_raw_depth_min_m, row_depth_min)
                )
                hold_raw_depth_max_m = (
                    row_depth_max
                    if hold_raw_depth_max_m is None
                    else max(hold_raw_depth_max_m, row_depth_max)
                )
                hold_raw_orientation_max_rad = max(
                    hold_raw_orientation_max_rad,
                    max(row["physics_sample_orientation_error_rad"]),
                )
                hold_raw_lateral_max_m = max(
                    hold_raw_lateral_max_m,
                    max(row["physics_sample_lateral_error_m"]),
                )
                hold_endpoint_insertion_load_max_n = max(
                    hold_endpoint_insertion_load_max_n,
                    row["insertion_contact_load_n"],
                )
                hold_endpoint_insertion_peak_max_n = max(
                    hold_endpoint_insertion_peak_max_n,
                    row["insertion_contact_load_peak_n"],
                )
                hold_endpoint_forbidden_load_max_n = max(
                    hold_endpoint_forbidden_load_max_n,
                    row["forbidden_contact_load_n"],
                )
                hold_endpoint_forbidden_peak_max_n = max(
                    hold_endpoint_forbidden_peak_max_n,
                    row["forbidden_contact_load_peak_n"],
                )
                contact_summary = row["physics_contact_summary"]
                hold_physics_insertion_load_max_n = max(
                    hold_physics_insertion_load_max_n,
                    contact_summary["insertion_load_max_n"],
                )
                hold_physics_forbidden_load_max_n = max(
                    hold_physics_forbidden_load_max_n,
                    contact_summary["forbidden_load_max_n"],
                )
                hold_physics_finger_force_min_n = min(
                    hold_physics_finger_force_min_n,
                    contact_summary["finger_force_min_n"],
                )
                hold_physics_finger_force_max_n = max(
                    hold_physics_finger_force_max_n,
                    contact_summary["finger_force_max_n"],
                )
                raw_depth_ok = all(
                    depth < 0.0
                    and abs(depth - held_actual_depth) <= HOLD_DEPTH_BAND_M
                    for depth in raw_depths
                )
                sample_ok = (
                    all(row["physics_sample_finite"])
                    and raw_depth_ok
                    and _substep_contact_hold_ok(
                        contact_summary,
                        grip_threshold_n=float(env.cfg.grip_contact_force_n),
                        maximum_force_n=HOLD_MAX_LOAD_N,
                    )
                    and max(row["physics_sample_lateral_error_m"]) <= HOLD_MAX_LATERAL_ERROR_M
                    and max(row["physics_sample_orientation_error_rad"]) <= HOLD_MAX_ORIENTATION_ERROR_RAD
                    and max(row["physics_sample_relative_linear_speed_m_s"])
                    <= HOLD_MAX_RELATIVE_LINEAR_SPEED_M_S
                    and max(row["physics_sample_relative_angular_speed_rad_s"])
                    <= HOLD_MAX_RELATIVE_ANGULAR_SPEED_RAD_S
                    and all(row["grip_live"])
                    and all(row["controller_grip_proven"])
                    and all(row["evaluation_grip_proven"])
                    and max(row["finger_contact_force_n"]) <= HOLD_MAX_LOAD_N
                    and row["insertion_contact_load_n"] == 0.0
                    and row["insertion_contact_load_peak_n"] == 0.0
                    and row["forbidden_contact_load_n"] == 0.0
                    and row["forbidden_contact_load_peak_n"] == 0.0
                )
                hold_dwell = _advance_hold_dwell(hold_dwell, sample_ok, env.step_dt)
                max_hold_dwell = max(max_hold_dwell, hold_dwell)
                final_hold_row = row
                hold_control_steps_completed += 1
                if row["terminal_mask"]:
                    hold_terminal = True
                    hold_terminal_reason_code = row["termination_reason_code"]
                    hold_terminal_reason = row["termination_reason"]
                    break
            hold_passed = (
                not hold_terminal
                and max_hold_dwell + 1.0e-12 >= HOLD_REQUIRED_DWELL_S
                and hold_control_steps_completed == hold_steps
            )

            if hold_terminal:
                release_boundary = {
                    "applicable": False,
                    "evaluation_dwell_s": final_hold_row[
                        "independent_eval_dwell_s"
                    ],
                    "evaluation_success": final_hold_row[
                        "independent_eval_success"
                    ],
                    "success_since_step": final_hold_row["success_since_step"],
                }
                release_boundary_passed = None
            else:
                release_boundary = {
                    "applicable": True,
                    "evaluation_dwell_s": float(
                        env._evaluation_dwell_s[0].item()
                    ),
                    "evaluation_success": bool(
                        env._evaluation_success[0].item()
                    ),
                    "success_since_step": int(
                        env._success_since_step[0].item()
                    ),
                }
                release_boundary_passed = _release_boundary_status(
                    release_boundary["evaluation_dwell_s"],
                    release_boundary["evaluation_success"],
                    release_boundary["success_since_step"],
                )
            invalid_reason = None
            if release_boundary_passed is False:
                invalid_reason = "hold_state_contaminated_strict_evaluator_release_boundary"

            env.cfg.insertion_speed_m_s = original_speed
            maximum_full_steps = max(1, math.ceil(24.0 / env.step_dt))
            full_task_success = False
            final_row = final_hold_row
            if not hold_terminal and release_boundary_passed is True:
                env.episode_length_buf[0] = 0
                env._phase_started_s[0] = env._sim_time_s()
                for _ in range(maximum_full_steps):
                    _, _, terminated, truncated, _ = env.step(analytic_action)
                    final_row = record_step(
                        "full_insertion", analytic_action, terminated, truncated
                    )
                    if final_row["terminal_mask"]:
                        full_task_success = bool(
                            final_row["independent_eval_success"]
                        )
                        break
            if final_row is None:
                raise RuntimeError("probe produced no control record")

            current.assert_unchanged()
            trace_writer.close()
            args.preserved_trace_records = trace_writer.records
            trace_sha = _sha256(trace_path)
            args.preserved_trace_sha256 = trace_sha
            new_mode = mode == "absolute_tip_pose_target_insert_only"
            action_contract_passed = bool(action_application_checks) and all(
                all(check.values()) for check in action_application_checks
            )
            first_input_valid = first_analytic_policy_input_valid is True
            authority_evidence_passed = _authority_evidence_gate(
                action_contract_passed, first_input_valid
            )
            if new_mode and not authority_evidence_passed and invalid_reason is None:
                invalid_reason = (
                    "first_analytic_policy_input_invalid"
                    if not first_input_valid
                    else "analytic_action_not_applied_by_controller"
                )
            passed = (
                bool(
                    invalid_reason is None
                    and hold_passed
                    and full_task_success
                    and authority_evidence_passed
                )
                if new_mode
                else invalid_reason is None
            )
            result = {
                "schema_version": 1,
                "passed": passed,
                "physics_executed": True,
                "task_id": "rj45-v1",
                "diagnostic_scope": "controller_authority_ab_probe",
                "action_application_mode": mode,
                "num_envs": 1,
                "invalid_reason": invalid_reason,
                "fixed_bias": {
                    "translation_entry_m": list(PROBE_TRANSLATION_BIAS_ENTRY_M),
                    "rotation_vector_entry_rad": list(PROBE_ROTATION_BIAS_ENTRY_RAD),
                    "sampling_override": "fixed_after_episode_index_increment_before_estimator_freeze",
                    "random_sampler_replaced": True,
                },
                "analytic_probe_action": {
                    "normalized_right_6d": list(action_values),
                    "expected_normalized_action_12d": _finite_json_value(
                        expected_normalized_action_12d[0]
                    ),
                    "expected_tip_offset_entry_6d": expected_tip_offset_6d,
                    "normalized_action_l2_tolerance": NORMALIZED_ACTION_L2_TOL,
                    "tip_offset_l2_tolerance_m_rad": TIP_OFFSET_L2_TOL_M_RAD,
                    "derivation": "translation_bias_x_m / residual_translation_bound_m",
                    "runtime_bound_preflight_passed": True,
                    "runtime_bound_preflight_l2_error": float(
                        bounded_preflight_error[0].item()
                    ),
                    "application_checks": action_application_checks,
                    "application_contract_passed": action_contract_passed,
                    "first_analytic_policy_input_valid": first_input_valid,
                    "authority_evidence_gate_passed": authority_evidence_passed,
                    "learning_or_sensor_performance_claim": False,
                },
                "observation_refresh": observation_refresh,
                "controller_overrides": {
                    "logical_phase_initialization": "verified held INSERT setup to INSERT via transition hook",
                    "hold_insertion_speed_m_s": 0.0,
                    "hold_command_depth_m": held_estimated_depth,
                    "hold_command_depth_source": "estimated mating depth after verified warmup",
                    "post_hold_full_task_episode_timer_reset": True,
                    "strict_evaluator_state_reset_at_release": False,
                    "physical_object_or_joint_state_write": False,
                    "held_grip_checks_bypassed": False,
                },
                "reference_checks": reference_checks,
                "warmup_control_steps": warmup_steps,
                "hold": {
                    "duration_s": HOLD_DURATION_S,
                    "required_continuous_dwell_s": HOLD_REQUIRED_DWELL_S,
                    "control_steps_required": hold_steps,
                    "control_steps_completed": hold_control_steps_completed,
                    "maximum_continuous_dwell_s": max_hold_dwell,
                    "passed": hold_passed,
                    "physical_terminal": hold_terminal,
                    "terminal_reason_code": hold_terminal_reason_code,
                    "terminal_reason": hold_terminal_reason,
                    "depth_is_success_criterion": False,
                    "expected_start_depth_m": held_actual_depth,
                    "depth_diagnostic_band_m": HOLD_DEPTH_BAND_M,
                    "sampling_scope": (
                        "eight post-parent-apply pre-step contact reads aligned with raw "
                        "pose samples plus one pre-dones physics endpoint contact read"
                    ),
                    "physics_contact_samples_per_control": int(env.cfg.decimation) + 1,
                    "all_configured_physics_sampling_instants_observed": True,
                    "continuous_between_sample_instants_observed": False,
                    "criteria": {
                        "raw_physics_depth_negative": True,
                        "raw_physics_depth_within_start_band_m": HOLD_DEPTH_BAND_M,
                        "all_nine_physics_insertion_load_samples_n": 0.0,
                        "all_nine_physics_forbidden_load_samples_n": 0.0,
                        "all_nine_by_four_finger_force_min_n": float(
                            env.cfg.grip_contact_force_n
                        ),
                        "all_nine_by_four_finger_force_max_n": HOLD_MAX_LOAD_N,
                        "lateral_error_m": HOLD_MAX_LATERAL_ERROR_M,
                        "orientation_error_rad": HOLD_MAX_ORIENTATION_ERROR_RAD,
                        "relative_linear_speed_m_s": HOLD_MAX_RELATIVE_LINEAR_SPEED_M_S,
                        "relative_angular_speed_rad_s": HOLD_MAX_RELATIVE_ANGULAR_SPEED_RAD_S,
                        "live_and_proven_grip": True,
                        "maximum_grip_load_n": HOLD_MAX_LOAD_N,
                        "control_endpoint_insertion_contact_load_n": 0.0,
                        "endpoint_sampled_episode_insertion_contact_load_peak_n": 0.0,
                        "control_endpoint_forbidden_load_n": 0.0,
                        "endpoint_sampled_episode_forbidden_load_peak_n": 0.0,
                    },
                    "observed": {
                        "raw_physics_depth_min_m": hold_raw_depth_min_m,
                        "raw_physics_depth_max_m": hold_raw_depth_max_m,
                        "raw_physics_lateral_error_max_m": hold_raw_lateral_max_m,
                        "raw_physics_orientation_error_max_rad": (
                            hold_raw_orientation_max_rad
                        ),
                        "physics_sample_insertion_contact_load_max_n": (
                            hold_physics_insertion_load_max_n
                        ),
                        "physics_sample_forbidden_contact_load_max_n": (
                            hold_physics_forbidden_load_max_n
                        ),
                        "physics_sample_finger_force_min_n": (
                            hold_physics_finger_force_min_n
                        ),
                        "physics_sample_finger_force_max_n": (
                            hold_physics_finger_force_max_n
                        ),
                        "control_endpoint_insertion_contact_load_max_n": (
                            hold_endpoint_insertion_load_max_n
                        ),
                        "endpoint_sampled_episode_insertion_contact_load_peak_max_n": (
                            hold_endpoint_insertion_peak_max_n
                        ),
                        "control_endpoint_forbidden_load_max_n": (
                            hold_endpoint_forbidden_load_max_n
                        ),
                        "endpoint_sampled_episode_forbidden_load_peak_max_n": (
                            hold_endpoint_forbidden_peak_max_n
                        ),
                    },
                    "final_actual_relative_position_entry_m": final_hold_row[
                        "actual_relative_position_entry_m"
                    ],
                    "final_estimated_relative_position_entry_m": final_hold_row[
                        "estimated_relative_position_entry_m"
                    ],
                },
                "release_boundary": {
                    **release_boundary,
                    "expected": {
                        "evaluation_dwell_s": 0.0,
                        "evaluation_success": False,
                        "success_since_step": -1,
                    },
                    "passed": release_boundary_passed,
                    "state_reset_to_hide_hold_dwell": False,
                },
                "full_task": {
                    "maximum_duration_s": 24.0,
                    "strict_independent_evaluator_success": full_task_success,
                    "final_record": final_row,
                },
                "interpretation": {
                    "old_mode_expected_drift_is_not_a_pass_fail_gate": not new_mode,
                    "new_mode_requires_hold_and_full_strict_success": new_mode,
                    "controller_authority_expectation_met": (
                        bool(
                            hold_passed
                            and full_task_success
                            and authority_evidence_passed
                            and invalid_reason is None
                        )
                        if new_mode
                        else None
                    ),
                    "probe_is_not_rl_performance_evidence": True,
                    "probe_is_not_sensor_performance_evidence": True,
                },
                "joint_metadata": {
                    "names": list(env.robot.joint_names),
                    "units": [
                        dict(
                            zip(
                                held_state["joint_names"],
                                held_state["joint_units"],
                                strict=True,
                            )
                        )[name]
                        for name in env.robot.joint_names
                    ],
                    "ordering": "runtime_robot_joint_order",
                    "mixed_units": True,
                },
                "trace": {
                    "path": str(trace_path.relative_to(PROJECT_ROOT)),
                    "sha256": trace_sha,
                    "records": trace_writer.records,
                },
                "proof": {"nominal": nominal_proof, "held": held_proof},
                "provenance": {
                    "checker_sha256": _sha256(Path(__file__)),
                    "task_config_sha256": task["_config_sha256"],
                    "runtime_config_sha256": _sha256(runtime_path),
                    "training_config_sha256": training_config["_config_sha256"],
                    "nominal_report_sha256": _sha256(nominal_path),
                    "nominal_trace_sha256": _sha256(nominal_trace_path),
                    "held_artifact_sha256": _sha256(held_path),
                    "captured_input_sha256": dict(sorted(current.sha256.items())),
                    "runtime_versions": versions,
                },
            }
            current.assert_unchanged()
            atomic_write_json(report_path, _finite_json_value(result))
            print(json.dumps(_finite_json_value(result), indent=2, allow_nan=False), flush=True)
            return 0 if passed else 2
        finally:
            try:
                trace_writer.close()
                args.preserved_trace_records = trace_writer.records
                args.preserved_trace_sha256 = _sha256(trace_path)
            finally:
                env.close()
    finally:
        try:
            simulation_app.close()
        except Exception:
            traceback.print_exc()
            raise


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = Path(args.report)
    try:
        return _run(args)
    except Exception as error:
        traceback.print_exc()
        try:
            from rby1_assembly.experiment_contracts import atomic_write_json

            resolved = getattr(args, "validated_report_path", None)
            if resolved is not None:
                atomic_write_json(
                    resolved,
                    {
                        "schema_version": 1,
                        "passed": False,
                        "physics_executed": bool(
                            getattr(args, "physics_executed", False)
                        ),
                        "error": {"type": type(error).__name__, "message": str(error)},
                        "preserved_trace": (
                            {
                                "path": str(
                                    getattr(args, "trace_path").relative_to(PROJECT_ROOT)
                                ),
                                "sha256": getattr(
                                    args, "preserved_trace_sha256", None
                                ),
                                "records": getattr(
                                    args, "preserved_trace_records", 0
                                ),
                            }
                            if hasattr(args, "trace_path")
                            else None
                        ),
                    },
                )
        except Exception:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
