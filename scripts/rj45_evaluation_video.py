"""Raw viewer evidence for one environment during a real vectorized evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from rby1_assembly.video_capture import PhysicsFrameCapture


HELD_INITIALIZATION_WARMUP_TIMEOUT_S = 1.0


def validate_evaluation_video_request(
    *,
    requested: bool,
    task: str,
    baseline_mode: str,
    cameras_enabled: bool,
    fps: int,
    env_index: int,
    num_envs: int,
    episodes: int,
) -> None:
    if not requested:
        return
    if task != "rj45" or baseline_mode not in {"learned", "bc", "biased_zero"}:
        raise ValueError(
            "--evaluation-video supports RJ45 learned, bc, or biased_zero modes"
        )
    if not cameras_enabled:
        raise ValueError("--evaluation-video requires --enable_cameras")
    if fps != 60:
        raise ValueError("--evaluation-video currently requires --video-fps 60")
    if type(env_index) is not int or not 0 <= env_index < num_envs:
        raise ValueError("--video-env-index must select an active environment")
    if env_index >= episodes:
        raise ValueError("selected video environment must have a positive episode quota")


def evaluation_video_capture_plan(
    *,
    physics_dt_s: float,
    control_decimation: int,
    fps: int,
    max_seconds: float,
    warmup_budget_control_steps: int | None = 0,
) -> dict[str, Any]:
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    if type(control_decimation) is not int or control_decimation <= 0:
        raise ValueError("control_decimation must be a positive integer")
    if type(fps) is not int or fps <= 0:
        raise ValueError("evaluation video FPS must be a positive integer")
    if not math.isfinite(max_seconds) or max_seconds <= 0.0 or max_seconds > 40.0:
        raise ValueError("evaluation video duration must be in (0, 40] seconds")
    stride = round(1.0 / (physics_dt_s * fps))
    if stride < 1 or not math.isclose(
        stride * physics_dt_s * fps, 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("evaluation video FPS must divide the physics frequency")
    if control_decimation % stride:
        raise ValueError("evaluation video cadence must divide each control step")
    frames_per_control = control_decimation // stride
    requested_frame_limit = math.floor(max_seconds * fps)
    requested_frame_limit -= requested_frame_limit % frames_per_control
    if requested_frame_limit < frames_per_control:
        raise ValueError("evaluation video duration is shorter than one control step")
    hard_frame_limit = math.floor(40.0 * fps)
    hard_frame_limit -= hard_frame_limit % frames_per_control
    terminal_margin_frames = frames_per_control
    if requested_frame_limit + terminal_margin_frames > hard_frame_limit:
        raise ValueError(
            "evaluation video requested duration leaves no terminal margin under "
            "the 40-second hard cap"
        )
    if warmup_budget_control_steps is None:
        warmup_budget_frames = hard_frame_limit - requested_frame_limit - frames_per_control
        if warmup_budget_frames < 0:
            raise ValueError("evaluation video requested duration exceeds the hard cap")
    else:
        if (
            type(warmup_budget_control_steps) is not int
            or warmup_budget_control_steps < 0
        ):
            raise ValueError("warmup budget must be a non-negative integer or None")
        warmup_budget_frames = warmup_budget_control_steps * frames_per_control
    frame_limit = min(
        requested_frame_limit + warmup_budget_frames + terminal_margin_frames,
        hard_frame_limit,
    )
    applied_warmup_budget_frames = (
        frame_limit - requested_frame_limit - terminal_margin_frames
    )
    return {
        "physics_dt_s": float(physics_dt_s),
        "control_decimation": control_decimation,
        "control_dt_s": float(physics_dt_s * control_decimation),
        "fps": fps,
        "render_stride_physics_steps": stride,
        "frames_per_control_step": frames_per_control,
        "requested_frame_limit": requested_frame_limit,
        "requested_max_seconds": requested_frame_limit / fps,
        "warmup_budget_control_steps": (
            applied_warmup_budget_frames // frames_per_control
        ),
        "warmup_budget_frames": applied_warmup_budget_frames,
        "warmup_budget_seconds": applied_warmup_budget_frames / fps,
        "terminal_margin_control_steps": 1,
        "terminal_margin_frames": terminal_margin_frames,
        "terminal_margin_seconds": terminal_margin_frames / fps,
        "frame_limit": frame_limit,
        "max_seconds": frame_limit / fps,
        "hard_cap_seconds": 40.0,
    }


def held_initialization_capture_budget(
    env: Any, *, control_dt_s: float
) -> dict[str, int | float | str | None]:
    if not math.isfinite(control_dt_s) or control_dt_s <= 0.0:
        raise ValueError("control_dt_s must be finite and positive")
    config = getattr(env, "_training_config", None)
    if not isinstance(config, dict):
        return {
            "control_steps": None,
            "source": "40_second_hard_cap_missing_training_config",
            "warmup_s": None,
            "warmup_timeout_s": None,
            "held_reset_staging_control_steps": None,
        }
    warmup_s = config.get("warmup_s")
    staging = config.get("held_reset_staging")
    staging_steps = staging.get("control_steps") if isinstance(staging, dict) else None
    if (
        not isinstance(warmup_s, (int, float))
        or isinstance(warmup_s, bool)
        or not math.isfinite(float(warmup_s))
        or float(warmup_s) < 0.0
        or type(staging_steps) is not int
        or staging_steps < 0
    ):
        return {
            "control_steps": None,
            "source": "40_second_hard_cap_unbounded_training_config",
            "warmup_s": None,
            "warmup_timeout_s": None,
            "held_reset_staging_control_steps": None,
        }
    warmup_timeout_s = HELD_INITIALIZATION_WARMUP_TIMEOUT_S
    warmup_timeout_control_steps = max(
        1, math.ceil(warmup_timeout_s / control_dt_s)
    )
    return {
        "control_steps": warmup_timeout_control_steps + staging_steps,
        "source": "rj45_training_env_held_warmup_timeout_plus_config_staging",
        "warmup_s": float(warmup_s),
        "warmup_timeout_s": warmup_timeout_s,
        "held_reset_staging_control_steps": staging_steps,
    }


def selected_environment_camera(
    origins: Any, env_index: int
) -> dict[str, tuple[float, float, float]]:
    import torch

    points = torch.as_tensor(origins, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("environment origins must have shape (num_envs, 3)")
    if type(env_index) is not int or not 0 <= env_index < points.shape[0]:
        raise ValueError("video environment index is outside the environment grid")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("environment origins must be finite")
    origin = points[env_index]
    eye = origin + points.new_tensor((0.95, -0.60, 1.25))
    lookat = origin + points.new_tensor((0.36, 0.0, 0.87))
    return {
        "eye_world_m": (float(eye[0]), float(eye[1]), float(eye[2])),
        "lookat_world_m": (
            float(lookat[0]),
            float(lookat[1]),
            float(lookat[2]),
        ),
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _scalar(value: Any, env_index: int) -> Any:
    selected = value[env_index]
    if hasattr(selected, "detach"):
        selected = selected.detach().cpu()
    if getattr(selected, "ndim", 0) != 0:
        raise RuntimeError("evaluation video telemetry must be scalar per environment")
    return selected.item()


def _evaluator_orientation_error(relative_quaternion: Any, identity_quaternion: Any) -> Any:
    from isaaclab.utils.math import (  # pyright: ignore[reportMissingImports]
        quat_error_magnitude,
    )

    return quat_error_magnitude(relative_quaternion, identity_quaternion)


class EvaluationVideoRecorder:
    """Capture the initial evaluation episode without changing state or actions."""

    def __init__(
        self,
        env: Any,
        *,
        video_path: Path,
        env_index: int,
        fps: int,
        max_seconds: float,
        policy_lineage: dict[str, Any],
        source_and_config_sha256: dict[str, str],
        writer_factory: Callable[[Path, int], Any] | None = None,
        frame_reader: Callable[[], Any] | None = None,
    ) -> None:
        if type(env_index) is not int or not 0 <= env_index < int(env.num_envs):
            raise ValueError("video environment index is outside the evaluation batch")
        self.env = env
        self.env_index = env_index
        self.video_path = Path(video_path).resolve()
        self.frames_path = Path(str(self.video_path) + ".frames.jsonl")
        self.metadata_path = Path(str(self.video_path) + ".metadata.json")
        self.evidence_path = Path(str(self.video_path) + ".evidence.json")
        control_dt_s = float(env.cfg.sim.dt) * int(env.cfg.decimation)
        warmup_budget = held_initialization_capture_budget(
            env, control_dt_s=control_dt_s
        )
        warmup_budget_steps = warmup_budget["control_steps"]
        assert warmup_budget_steps is None or type(warmup_budget_steps) is int
        self.plan = evaluation_video_capture_plan(
            physics_dt_s=float(env.cfg.sim.dt),
            control_decimation=int(env.cfg.decimation),
            fps=fps,
            max_seconds=max_seconds,
            warmup_budget_control_steps=warmup_budget_steps,
        )
        self.plan["warmup_budget_source"] = warmup_budget["source"]
        self.plan["warmup_s_from_training_config"] = warmup_budget["warmup_s"]
        self.plan["warmup_timeout_s_from_training_env"] = warmup_budget.get(
            "warmup_timeout_s"
        )
        self.plan["held_reset_staging_control_steps"] = warmup_budget[
            "held_reset_staging_control_steps"
        ]
        self.policy_lineage = json.loads(json.dumps(policy_lineage, allow_nan=False))
        self.policy_lineage_sha256 = hashlib.sha256(
            json.dumps(
                self.policy_lineage,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        self.source_and_config_sha256 = dict(source_and_config_sha256)
        self._writer_factory = writer_factory or self._default_writer_factory
        self._frame_reader = frame_reader or self._read_frame
        self._writer = None
        self._frames_file = None
        self._capture: PhysicsFrameCapture | None = None
        self._previous_rtx_setting: bool | None = None
        self._pending_frames: list[dict[str, Any]] = []
        self._control_step = 0
        self._step_policy_input_valid = False
        self._step_policy_input_timestamp_s = -1.0
        self._step_entry_physics_step = 0
        self._step_entry_episode_time_s = 0.0
        self._step_entry_contact_force_proxy_n = 0.0
        self._previous_policy_input_timestamp_s: float | None = None
        self._written_frames = 0
        self._written_records = 0
        self._target_episode: int | None = None
        self._started = False
        self._capture_active = False
        self._closed = False
        self._stop_reason: str | None = None
        self._selected_terminal: dict[str, Any] | None = None
        self._terminal_timestamp_tolerance_s: float | None = None
        self._terminal_expected_timestamp_s: float | None = None
        self._observed_held_initialization_offset_s = 0.0
        self._cleanup_errors: list[str] = []
        self.camera: dict[str, tuple[float, float, float]] | None = None

    @staticmethod
    def _default_writer_factory(path: Path, fps: int) -> Any:
        import imageio.v2 as imageio  # pyright: ignore[reportMissingImports]

        return imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )

    def _read_frame(self) -> Any:
        frame = self.env.render(recompute=True)
        if frame is not None and frame.size > 0 and bool(frame.any()):
            return frame.copy()
        raise RuntimeError("evaluation renderer produced an empty RGB frame")

    def initialize(self) -> None:
        if self._started or self._closed:
            raise RuntimeError("evaluation video recorder cannot be initialized again")
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (
            self.video_path,
            self.frames_path,
            self.metadata_path,
            self.evidence_path,
        ):
            if path.exists():
                raise FileExistsError(f"evaluation video output already exists: {path}")
        episode = _scalar(self.env._training_episode_index, self.env_index)
        if type(episode) is not int or episode < 0:
            raise RuntimeError(
                "selected environment initial episode index must be a non-negative integer"
            )
        self._target_episode = episode
        self._started = True
        try:
            self.camera = selected_environment_camera(
                self.env.scene.env_origins, self.env_index
            )
            self.env.sim.set_camera_view(
                self.camera["eye_world_m"], self.camera["lookat_world_m"]
            )
            self._writer = self._writer_factory(
                self.video_path, int(self.plan["fps"])
            )
            self._frames_file = self.frames_path.open("x", encoding="utf-8", buffering=1)
            self.env.render(recompute=True)
            for _ in range(16):
                self.env.sim.render()
            self._frame_reader()
            self._previous_rtx_setting = bool(
                self.env.sim.get_setting("/isaaclab/render/rtx_sensors")
            )
            self.env.sim.set_setting("/isaaclab/render/rtx_sensors", True)
            self._capture = PhysicsFrameCapture(
                self.env.sim,
                lambda: self.env._sim_step_counter,
                self._frame_reader,
                self._write_frame,
                stride=int(self.plan["render_stride_physics_steps"]),
            )
            self._capture.install()
            self._capture_active = True
            _atomic_json(self.metadata_path, self._metadata(rollout_completed=False))
        except BaseException:
            self.close(rollout_completed=False)
            raise

    def _write_frame(self, frame: Any) -> None:
        if self._written_frames >= int(self.plan["frame_limit"]):
            raise RuntimeError("evaluation video frame limit exceeded")
        target_episode = self._target_episode
        if target_episode is None:
            raise RuntimeError("evaluation video initial episode was not frozen")
        episode = _scalar(self.env._training_episode_index, self.env_index)
        if type(episode) is not int or episode < 0:
            raise RuntimeError(
                "selected environment episode index must be a non-negative integer"
            )
        if episode != target_episode:
            raise RuntimeError("selected environment reset before terminal capture was finalized")
        relative_position, relative_quaternion = self.env._relative_object_pose()
        depth = float(_scalar(relative_position[:, 1], self.env_index))
        lateral = float(
            relative_position[self.env_index, (0, 2)].norm().detach().cpu().item()
        )
        orientation_error = _evaluator_orientation_error(
            relative_quaternion, self.env._identity_quat()
        )
        angle = float(_scalar(orientation_error, self.env_index))
        physics_step = int(self.env._sim_step_counter)
        phase = int(_scalar(self.env._phase, self.env_index))
        if physics_step <= self._step_entry_physics_step:
            raise RuntimeError("rendered physics step must follow the policy input snapshot")
        physics_time_s = physics_step * float(self.plan["physics_dt_s"])
        elapsed_physics_steps = physics_step - self._step_entry_physics_step
        time_since_policy_input_s = elapsed_physics_steps * float(
            self.plan["physics_dt_s"]
        )
        if (
            not math.isfinite(time_since_policy_input_s)
            or time_since_policy_input_s <= 0.0
            or time_since_policy_input_s
            > float(self.plan["control_dt_s"]) + 1.0e-9
        ):
            raise RuntimeError("frame time is inconsistent with the policy input timestamp")
        assert self._writer is not None
        self._writer.append_data(frame)
        self._pending_frames.append(
            {
                "frame_index": self._written_frames,
                "physics_step": physics_step,
                "simulation_time_s": physics_time_s,
                "physics_time_s": physics_time_s,
                "control_step": self._control_step,
                "control_time_s": self._step_policy_input_timestamp_s,
                "control_step_entry_policy_input_timestamp_s": (
                    self._step_policy_input_timestamp_s
                ),
                "time_since_policy_input_s": time_since_policy_input_s,
                "policy_and_physics_clock_relation": (
                    "policy timestamp and physics-step clock are recorded separately; "
                    "elapsed time uses the physics-step delta"
                ),
                "control_step_entry_physics_step": self._step_entry_physics_step,
                "control_step_entry_episode_time_s": (
                    self._step_entry_episode_time_s
                ),
                "episode_index": episode,
                "phase": phase,
                "actual_depth_m": depth,
                "actual_lateral_error_m": lateral,
                "actual_orientation_error_rad": angle,
                "control_step_entry_contact_force_proxy_n": (
                    self._step_entry_contact_force_proxy_n
                ),
                "control_step_entry_contact_force_proxy_source": (
                    "max(per_hand_filtered_finger_magnitude,"
                    "physx_plug_socket_sum_abs_contact_normals)"
                ),
                "contact_timing": "control_step_entry_not_rendered_physics_step",
                "policy_input_valid": self._step_policy_input_valid,
                "policy_lineage_sha256": self.policy_lineage_sha256,
                "frame_state_is_pre_autoreset_physics": True,
            }
        )
        self._written_frames += 1

    @contextmanager
    def control_step(self):
        if not self._capture_active:
            yield
            return
        self._control_step += 1
        self._pending_frames = []
        self._step_policy_input_valid = bool(
            _scalar(
                self.env._training_last_policy_observation_valid, self.env_index
            )
        )
        timestamp = float(
            _scalar(
                self.env._training_last_policy_observation_timestamp_s,
                self.env_index,
            )
        )
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise RuntimeError("policy input timestamp must be finite and non-negative")
        if (
            self._previous_policy_input_timestamp_s is not None
            and timestamp <= self._previous_policy_input_timestamp_s
        ):
            raise RuntimeError("policy input timestamps must increase each control step")
        self._previous_policy_input_timestamp_s = timestamp
        self._step_policy_input_timestamp_s = timestamp
        self._step_entry_physics_step = int(self.env._sim_step_counter)
        self._step_entry_episode_time_s = float(
            _scalar(self.env.episode_length_buf, self.env_index)
        ) * float(self.plan["control_dt_s"])
        clock_offset_s = (
            self._step_policy_input_timestamp_s - self._step_entry_episode_time_s
        )
        if not math.isfinite(clock_offset_s) or clock_offset_s < -1.0e-8:
            raise RuntimeError("evaluation video clocks have an invalid warmup offset")
        self._observed_held_initialization_offset_s = max(
            self._observed_held_initialization_offset_s, max(0.0, clock_offset_s)
        )
        grip_contact = float(
            self.env._last_contact_proxy_n[self.env_index]
            .max()
            .detach()
            .cpu()
            .item()
        )
        insertion_contact = float(
            _scalar(self.env._last_insertion_contact_load_n, self.env_index)
        )
        if not all(math.isfinite(value) for value in (grip_contact, insertion_contact)):
            raise RuntimeError("control-step entry contact telemetry must be finite")
        self._step_entry_contact_force_proxy_n = max(
            grip_contact, insertion_contact
        )
        expected = int(self.plan["frames_per_control_step"])
        if self._written_frames + expected > int(self.plan["frame_limit"]):
            raise RuntimeError("evaluation video frame limit is not control-step aligned")
        assert self._capture is not None
        with self._capture.control_step(expected_frames=expected):
            yield

    def finalize_control_step(self, extras: dict[str, Any], dones: Any) -> None:
        if not self._pending_frames:
            return
        terminal = bool(_scalar(dones, self.env_index))
        snapshot = extras.get("terminal_snapshot")
        if not isinstance(snapshot, dict):
            raise RuntimeError("evaluation video requires terminal_snapshot telemetry")
        phase_names = snapshot.get("phase_name_by_code")
        if not isinstance(phase_names, dict):
            raise RuntimeError("evaluation video requires phase names")
        terminal_evaluation = extras.get("terminal_evaluation")
        if not isinstance(terminal_evaluation, dict):
            raise RuntimeError("evaluation video requires terminal_evaluation telemetry")
        control_step_snapshot = self._control_step_snapshot(
            snapshot, terminal_evaluation
        )
        if terminal:
            self._validate_terminal_snapshot(snapshot)
        last_pre_autoreset_frame = dict(self._pending_frames[-1])
        assert self._frames_file is not None
        for index, pending in enumerate(self._pending_frames):
            pending["phase_name"] = phase_names.get(pending["phase"], "unknown")
            pending["control_step_snapshot"] = control_step_snapshot
            pending["terminal_observed_after_control_step"] = bool(
                terminal and index == len(self._pending_frames) - 1
            )
            pending["success_observed_after_control_step"] = bool(
                terminal
                and index == len(self._pending_frames) - 1
                and _scalar(extras["success"], self.env_index)
            )
            self._frames_file.write(json.dumps(pending, allow_nan=False) + "\n")
            self._written_records += 1
        self._pending_frames = []
        if terminal:
            target_episode = self._target_episode
            if target_episode is None:
                raise RuntimeError("evaluation video initial episode was not frozen")
            terminal_episode = _scalar(snapshot["episode_index"], self.env_index)
            if type(terminal_episode) is not int or terminal_episode < 0:
                raise RuntimeError(
                    "selected terminal episode index must be a non-negative integer"
                )
            if terminal_episode != target_episode:
                raise RuntimeError(
                    "selected terminal snapshot does not belong to the initial evaluation episode"
                )
            self._selected_terminal = {
                "episode_index": terminal_episode,
                "success": bool(_scalar(extras["success"], self.env_index)),
                "valid": bool(_scalar(extras["evaluation"]["valid"], self.env_index)),
                "termination_reason": str(
                    extras["terminal_evaluation"]["termination_reason"][self.env_index]
                ),
                "timestamp_s": float(
                    _scalar(snapshot["timestamp_s"], self.env_index)
                ),
                "actual_depth_m": float(
                    _scalar(snapshot["actual_depth_m"], self.env_index)
                ),
                "actual_lateral_error_m": float(
                    _scalar(snapshot["actual_lateral_error_m"], self.env_index)
                ),
                "actual_orientation_error_rad": float(
                    _scalar(
                        snapshot["actual_orientation_error_rad"], self.env_index
                    )
                ),
                "last_pre_autoreset_physics_step": last_pre_autoreset_frame[
                    "physics_step"
                ],
                "last_pre_autoreset_geometry": {
                    name: last_pre_autoreset_frame[name]
                    for name in (
                        "actual_depth_m",
                        "actual_lateral_error_m",
                        "actual_orientation_error_rad",
                    )
                },
                "timestamp_match_tolerance_s": (
                    self._terminal_timestamp_tolerance_s
                ),
                "expected_timestamp_s_from_entry_episode_clock": (
                    self._terminal_expected_timestamp_s
                ),
                "geometry_match_tolerance": 1.0e-6,
            }
            self._stop_capture("selected_environment_initial_episode_terminal")
        elif self._written_frames == int(self.plan["frame_limit"]):
            self._stop_capture("maximum_duration_reached_before_terminal")

    def _control_step_snapshot(
        self,
        snapshot: dict[str, Any],
        terminal_evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        def vector(
            source: dict[str, Any],
            name: str,
            size: int,
            *,
            label: str = "terminal_snapshot",
            boolean: bool = False,
        ) -> list[float] | list[bool]:
            value = source.get(name)
            if value is None:
                raise RuntimeError(f"evaluation video requires {label}.{name}")
            selected = value[self.env_index]
            if hasattr(selected, "detach"):
                selected = selected.detach().clone().cpu()
            if getattr(selected, "shape", None) != (size,):
                raise RuntimeError(
                    f"{label}.{name} must have selected shape ({size},)"
                )
            raw_result = selected.tolist()
            if boolean:
                if not all(type(item) is bool for item in raw_result):
                    raise RuntimeError(f"{label}.{name} must be boolean")
                return raw_result
            result = [float(item) for item in raw_result]
            if not all(math.isfinite(item) for item in result):
                raise RuntimeError(f"{label}.{name} must be finite")
            return result

        def scalar(
            source: dict[str, Any],
            name: str,
            *,
            label: str = "terminal_snapshot",
            boolean: bool = False,
        ) -> float | bool:
            value = source.get(name)
            if value is None:
                raise RuntimeError(f"evaluation video requires {label}.{name}")
            result = _scalar(value, self.env_index)
            if boolean:
                if type(result) is not bool:
                    raise RuntimeError(f"{label}.{name} must be boolean")
                return result
            result = float(result)
            if not math.isfinite(result):
                raise RuntimeError(f"{label}.{name} must be finite")
            return result

        timestamp_value = snapshot.get("policy_input_timestamp_s")
        if timestamp_value is None:
            raise RuntimeError(
                "evaluation video requires terminal_snapshot.policy_input_timestamp_s"
            )
        timestamp = float(_scalar(timestamp_value, self.env_index))
        timestamp_tolerance_s = max(
            1.0e-8, abs(self._step_policy_input_timestamp_s) * 2.0**-23
        )
        if (
            not math.isfinite(timestamp)
            or timestamp < 0.0
            or not math.isclose(
                timestamp,
                self._step_policy_input_timestamp_s,
                rel_tol=0.0,
                abs_tol=timestamp_tolerance_s,
            )
        ):
            raise RuntimeError(
                "terminal_snapshot.policy_input_timestamp_s does not match the control-step input"
            )
        policy_input_valid_value = snapshot.get("policy_input_valid")
        if policy_input_valid_value is None:
            raise RuntimeError(
                "evaluation video requires terminal_snapshot.policy_input_valid"
            )
        policy_input_valid = _scalar(policy_input_valid_value, self.env_index)
        if type(policy_input_valid) is not bool:
            raise RuntimeError("terminal_snapshot.policy_input_valid must be boolean")
        stability = snapshot.get("rj45_evaluation_stability")
        if not isinstance(stability, dict):
            raise RuntimeError(
                "evaluation video requires terminal_snapshot.rj45_evaluation_stability"
            )
        sample_count = int(self.plan["control_decimation"]) + 1
        stability_snapshot: dict[str, Any] = {
            name: scalar(
                stability,
                name,
                label="terminal_snapshot.rj45_evaluation_stability",
                boolean=True,
            )
            for name in (
                "depth_within_window",
                "relative_linear_speed_ok",
                "relative_angular_speed_ok",
                "physics_substeps_complete",
                "physics_all_samples_stable",
            )
        }
        stability_snapshot.update(
            {
                name: scalar(
                    stability,
                    name,
                    label="terminal_snapshot.rj45_evaluation_stability",
                )
                for name in (
                    "relative_linear_speed_m_s",
                    "relative_angular_speed_rad_s",
                )
            }
        )
        stability_snapshot.update(
            {
                name: vector(
                    stability,
                    name,
                    sample_count,
                    label="terminal_snapshot.rj45_evaluation_stability",
                    boolean=True,
                )
                for name in (
                    "physics_sample_finite",
                    "physics_sample_stable",
                )
            }
        )
        stability_snapshot.update(
            {
                name: vector(
                    stability,
                    name,
                    sample_count,
                    label="terminal_snapshot.rj45_evaluation_stability",
                )
                for name in (
                    "physics_sample_depth_m",
                    "physics_sample_lateral_error_m",
                    "physics_sample_orientation_error_rad",
                    "physics_sample_relative_linear_speed_m_s",
                    "physics_sample_relative_angular_speed_rad_s",
                )
            }
        )
        stability_snapshot["physics_sample_count"] = sample_count
        stability_snapshot["sample_contract"] = (
            "control_decimation pre-physics samples plus the final control-step endpoint"
        )
        criteria = terminal_evaluation.get("criteria")
        if not isinstance(criteria, dict):
            raise RuntimeError(
                "evaluation video requires terminal_evaluation.criteria"
            )
        criteria_snapshot: dict[str, bool] = {}
        for name in (
            "depth",
            "lateral",
            "orientation",
            "grasp_live_and_proven",
            "dwell",
            "force",
            "forbidden_contact",
        ):
            criterion = scalar(
                criteria,
                name,
                label="terminal_evaluation.criteria",
                boolean=True,
            )
            assert isinstance(criterion, bool)
            criteria_snapshot[name] = criterion
        independent_eval_dwell_s = scalar(snapshot, "independent_eval_dwell_s")
        assert isinstance(independent_eval_dwell_s, float)
        if independent_eval_dwell_s < 0.0:
            raise RuntimeError(
                "terminal_snapshot.independent_eval_dwell_s must be non-negative"
            )
        return {
            "policy_input_observation": vector(
                snapshot, "policy_input_observation", 79
            ),
            "policy_input_timestamp_s": timestamp,
            "policy_input_valid": policy_input_valid,
            "requested_expanded_residual_action": vector(
                snapshot, "requested_expanded_residual_action", 12
            ),
            "applied_bounded_residual_action": vector(
                snapshot, "applied_bounded_residual_action", 12
            ),
            "applied_tip_target_offset_entry_6d": vector(
                snapshot, "applied_tip_target_offset_entry_6d", 6
            ),
            "independent_eval_dwell_s": independent_eval_dwell_s,
            "grip_live": vector(snapshot, "grip_live", 2, boolean=True),
            "grip_proven": vector(snapshot, "grip_proven", 2, boolean=True),
            "rj45_evaluation_stability": stability_snapshot,
            "terminal_evaluation_criteria": criteria_snapshot,
            "timing_contract": (
                "policy input is cached at control-step start; requested and bounded "
                "actions apply during that control step; applied target, 240 Hz physics "
                "stability samples, dwell, grip, and criteria are recorded at control-step end"
            ),
        }

    def _validate_terminal_snapshot(self, snapshot: dict[str, Any]) -> None:
        last = self._pending_frames[-1]
        final_physics_step = int(self.env._sim_step_counter)
        expected_physics_step = self._step_entry_physics_step + int(
            self.plan["control_decimation"]
        )
        if last["physics_step"] != final_physics_step or final_physics_step != expected_physics_step:
            raise RuntimeError(
                "last pre-autoreset frame is not the terminal control-step physics boundary"
            )
        terminal_timestamp_s = float(
            _scalar(snapshot["timestamp_s"], self.env_index)
        )
        expected_episode_time_s = (
            self._step_entry_episode_time_s + float(self.plan["control_dt_s"])
        )
        self._terminal_expected_timestamp_s = expected_episode_time_s
        timestamp_tolerance_s = max(
            1.0e-8, abs(expected_episode_time_s) * 2.0**-23
        )
        self._terminal_timestamp_tolerance_s = timestamp_tolerance_s
        if (
            not math.isfinite(terminal_timestamp_s)
            or not math.isclose(
                terminal_timestamp_s,
                expected_episode_time_s,
                rel_tol=0.0,
                abs_tol=timestamp_tolerance_s,
            )
        ):
            raise RuntimeError("terminal timestamp is inconsistent with the control step")
        for name in (
            "actual_depth_m",
            "actual_lateral_error_m",
            "actual_orientation_error_rad",
        ):
            terminal_value = float(_scalar(snapshot[name], self.env_index))
            if (
                not math.isfinite(terminal_value)
                or not math.isclose(
                    terminal_value,
                    float(last[name]),
                    rel_tol=0.0,
                    abs_tol=1.0e-6,
                )
            ):
                raise RuntimeError(
                    f"terminal {name} differs from the last pre-autoreset frame"
                )

    def _stop_capture(self, reason: str) -> None:
        if not self._capture_active:
            return
        assert self._capture is not None
        try:
            self._capture.close()
        except BaseException as error:
            self._cleanup_errors.append(f"capture_close: {type(error).__name__}: {error}")
        try:
            if self._previous_rtx_setting is not None:
                self.env.sim.set_setting(
                    "/isaaclab/render/rtx_sensors", self._previous_rtx_setting
                )
        except BaseException as error:
            self._cleanup_errors.append(f"rtx_restore: {type(error).__name__}: {error}")
        self._capture_active = False
        self._stop_reason = reason

    def _metadata(self, *, rollout_completed: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "completed": (
                self._selected_terminal is not None and not self._cleanup_errors
            ),
            "rollout_completed": rollout_completed,
            "scope": (
                "selected environment initial evaluation episode during real vectorized "
                "evaluation env.step calls"
            ),
            "video_path": str(self.video_path),
            "frames_path": str(self.frames_path),
            "selected_environment_index": self.env_index,
            "selected_episode_index": self._target_episode,
            "camera": self.camera,
            "capture_plan": self.plan,
            "captured_frames": self._written_frames,
            "captured_duration_s": self._written_frames / int(self.plan["fps"]),
            "observed_held_initialization_offset_s": (
                self._observed_held_initialization_offset_s
            ),
            "observed_held_initialization_control_steps": round(
                self._observed_held_initialization_offset_s
                / float(self.plan["control_dt_s"])
            ),
            "elapsed_clock_contract": {
                "requested_episode_horizon_s": self.plan["requested_max_seconds"],
                "capture_elapsed_s": self._written_frames / int(self.plan["fps"]),
                "selected_terminal_episode_elapsed_s": (
                    self._selected_terminal["timestamp_s"]
                    if self._selected_terminal is not None
                    else None
                ),
            },
            "frame_metadata_records": self._written_records,
            "captured_control_steps": self._control_step,
            "stop_reason": self._stop_reason,
            "selected_terminal": self._selected_terminal,
            "cleanup_errors": self._cleanup_errors,
            "policy_lineage": self.policy_lineage,
            "policy_lineage_sha256": self.policy_lineage_sha256,
            "source_and_config_sha256": self.source_and_config_sha256,
            "physics_or_action_contract_modified": False,
            "warmup_physics_steps_added": 0,
            "warmup_frames_recorded": False,
            "terminal_autoreset_frame_recorded_or_marked_success": False,
        }

    def close(self, *, rollout_completed: bool) -> dict[str, Any] | None:
        if self._closed:
            return self.evidence() if self.evidence_path.is_file() else None
        if not self._started:
            self._closed = True
            return None
        if self._capture_active:
            self._stop_capture("evaluation_closed_before_selected_terminal")
        if self._writer is not None:
            try:
                self._writer.close()
            except BaseException as error:
                self._cleanup_errors.append(
                    f"video_writer_close: {type(error).__name__}: {error}"
                )
        if self._frames_file is not None:
            try:
                self._frames_file.close()
            except BaseException as error:
                self._cleanup_errors.append(
                    f"frames_close: {type(error).__name__}: {error}"
                )
        self._closed = True
        metadata = self._metadata(rollout_completed=rollout_completed)
        _atomic_json(self.metadata_path, metadata)
        artifacts = {}
        for name, path in (
            ("video", self.video_path),
            ("frames", self.frames_path),
            ("metadata", self.metadata_path),
        ):
            if path.is_file():
                artifacts[name] = {"path": str(path), "sha256": _sha256(path)}
        evidence = {
            "schema_version": 1,
            "completed": metadata["completed"],
            "rollout_completed": rollout_completed,
            "artifacts": artifacts,
            "source_and_config_sha256": self.source_and_config_sha256,
            "policy_lineage": self.policy_lineage,
            "policy_lineage_sha256": self.policy_lineage_sha256,
        }
        _atomic_json(self.evidence_path, evidence)
        return evidence

    def evidence(self) -> dict[str, Any]:
        return json.loads(self.evidence_path.read_text(encoding="utf-8"))


__all__ = [
    "EvaluationVideoRecorder",
    "evaluation_video_capture_plan",
    "held_initialization_capture_budget",
    "selected_environment_camera",
    "validate_evaluation_video_request",
]
