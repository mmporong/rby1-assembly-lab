"""Bounded raw-camera evidence capture during real vectorized PPO rollouts."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable

from rby1_assembly.video_capture import PhysicsFrameCapture


def parallel_video_capture_plan(
    *, physics_dt_s: float, control_decimation: int, fps: int, seconds: float
) -> dict[str, Any]:
    if not math.isfinite(physics_dt_s) or physics_dt_s <= 0.0:
        raise ValueError("physics_dt_s must be finite and positive")
    if type(control_decimation) is not int or control_decimation <= 0:
        raise ValueError("control_decimation must be positive")
    if type(fps) is not int or fps <= 0:
        raise ValueError("parallel video FPS must be positive")
    if not math.isfinite(seconds) or seconds <= 0.0:
        raise ValueError("parallel video seconds must be finite and positive")
    render_stride = round(1.0 / (physics_dt_s * fps))
    if render_stride < 1 or not math.isclose(
        render_stride * physics_dt_s * fps, 1.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("parallel video FPS must divide the physics frequency")
    if control_decimation % render_stride != 0:
        raise ValueError("parallel video cadence must divide each control step")
    frames_per_control_step = control_decimation // render_stride
    requested_frames = math.floor(seconds * fps)
    frame_limit = (
        requested_frames // frames_per_control_step * frames_per_control_step
    )
    if frame_limit < frames_per_control_step:
        raise ValueError("parallel video duration is shorter than one control step")
    return {
        "render_stride_physics_steps": render_stride,
        "frames_per_control_step": frames_per_control_step,
        "requested_seconds": float(seconds),
        "requested_frames_floor": requested_frames,
        "frame_limit": frame_limit,
        "effective_seconds": frame_limit / fps,
        "fps": fps,
    }


def camera_pose_from_env_origins(
    origins: Any, view: str
) -> dict[str, tuple[float, float, float]]:
    import torch

    points = torch.as_tensor(origins, dtype=torch.float64)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 1:
        raise ValueError("environment origins must have shape (num_envs, 3)")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("environment origins must be finite")
    if view == "closeup":
        origin = points[0]
        eye = origin + points.new_tensor((0.95, -0.60, 1.25))
        lookat = origin + points.new_tensor((0.36, 0.0, 0.87))
    elif view == "overview":
        lower = points.min(dim=0).values
        upper = points.max(dim=0).values
        center = (lower + upper) * 0.5
        horizontal_span = max(
            float((upper[0] - lower[0]).item()),
            float((upper[1] - lower[1]).item()),
            2.5,
        )
        eye = center + points.new_tensor(
            (horizontal_span * 0.80, -horizontal_span * 1.35, horizontal_span * 0.90 + 1.4)
        )
        lookat = center + points.new_tensor((0.35, 0.0, 0.85))
    else:
        raise ValueError("parallel video view must be overview or closeup")
    return {
        "eye_world_m": tuple(float(value) for value in eye.tolist()),
        "lookat_world_m": tuple(float(value) for value in lookat.tolist()),
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tensor_value(value: Any, env_id: int) -> Any:
    selected = value[env_id]
    if hasattr(selected, "detach"):
        selected = selected.detach().cpu()
    if getattr(selected, "ndim", 0) == 0:
        return selected.item()
    return selected.tolist()


class ParallelTrainingVideoRecorder:
    """Capture a fixed-duration raw viewer video only inside wrapped PPO steps."""

    def __init__(
        self,
        env: Any,
        *,
        video_path: Path,
        seconds: float,
        fps: int,
        view: str,
        rollout_steps: int,
        writer_factory: Callable[[Path, int], Any] | None = None,
        frame_reader: Callable[[], Any] | None = None,
    ) -> None:
        self.env = env
        self.video_path = Path(video_path).resolve()
        self.frames_path = Path(str(self.video_path) + ".frames.jsonl")
        self.metadata_path = Path(str(self.video_path) + ".metadata.json")
        self.evidence_path = Path(str(self.video_path) + ".evidence.json")
        self.view = view
        if type(rollout_steps) is not int or rollout_steps <= 0:
            raise ValueError("rollout_steps must be positive")
        self.rollout_steps = rollout_steps
        self.plan = parallel_video_capture_plan(
            physics_dt_s=float(env.cfg.sim.dt),
            control_decimation=int(env.cfg.decimation),
            fps=fps,
            seconds=seconds,
        )
        self._writer_factory = writer_factory or self._default_writer_factory
        self._frame_reader = frame_reader or self._read_frame
        self._runner = None
        self._writer = None
        self._frames_file = None
        self._capture = None
        self._previous_rtx_setting = None
        self._control_step = 0
        self._written_frames = 0
        self._written_records = 0
        self._pending_frames: list[dict[str, Any]] = []
        self._step_policy_active = None
        self._initialized = False
        self._started = False
        self._render_capture_active = False
        self._closed = False
        self._source_and_config_sha256: dict[str, str] = {}
        self.camera: dict[str, Any] | None = None

    @staticmethod
    def _default_writer_factory(path: Path, fps: int) -> Any:
        import imageio.v2 as imageio

        return imageio.get_writer(
            path,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
            macro_block_size=1,
            output_params=["-movflags", "+faststart"],
        )

    def bind_runner(self, runner: Any) -> None:
        if self._runner is not None:
            raise RuntimeError("parallel video runner is already bound")
        if not hasattr(runner, "current_learning_iteration"):
            raise ValueError("runner must expose current_learning_iteration")
        self._runner = runner

    def set_evidence_sources(self, source_and_config_sha256: dict[str, str]) -> None:
        if self._source_and_config_sha256:
            raise RuntimeError("parallel video evidence sources are already set")
        self._source_and_config_sha256 = dict(source_and_config_sha256)

    def _read_frame(self) -> Any:
        frame = self.env.render(recompute=True)
        if frame is not None and frame.size > 0 and bool(frame.any()):
            return frame.copy()
        raise RuntimeError("parallel training renderer produced an empty RGB frame")

    def initialize(self) -> None:
        if self._initialized or self._closed:
            raise RuntimeError("parallel video recorder cannot be initialized again")
        if self._runner is None:
            raise RuntimeError("parallel video runner must be bound before initialization")
        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.video_path, self.frames_path, self.metadata_path, self.evidence_path):
            if path.exists():
                raise FileExistsError(f"parallel video output already exists: {path}")
        self._started = True
        self.camera = camera_pose_from_env_origins(
            self.env.scene.env_origins, self.view
        )
        self.env.sim.set_camera_view(
            self.camera["eye_world_m"], self.camera["lookat_world_m"]
        )
        self._writer = self._writer_factory(self.video_path, self.plan["fps"])
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
            stride=self.plan["render_stride_physics_steps"],
        )
        self._capture.install()
        self._render_capture_active = True
        self._initialized = True
        _atomic_json(self.metadata_path, self._metadata(completed=False))

    def _write_frame(self, frame: Any) -> None:
        if self._written_frames >= self.plan["frame_limit"]:
            raise RuntimeError("parallel video frame limit exceeded")
        self._writer.append_data(frame)
        relative_position, relative_quaternion = self.env._relative_object_pose()
        self._pending_frames.append(
            {
                "frame_index": self._written_frames,
                "physics_step": int(self.env._sim_step_counter),
                "phase": self.env._phase.detach().clone(),
                "actual_depth_m": relative_position[:, 1].detach().clone(),
                "actual_lateral_error_m": (
                    relative_position[:, (0, 2)]
                    .norm(dim=1)
                    .detach()
                    .clone()
                ),
                "actual_orientation_error_rad": (
                    2.0
                    * relative_quaternion[:, 0]
                    .abs()
                    .clamp(max=1.0)
                    .acos()
                )
                .detach()
                .clone(),
            }
        )
        self._written_frames += 1

    @contextmanager
    def control_step(self):
        if not self._render_capture_active:
            yield
            return
        self._control_step += 1
        self._pending_frames = []
        policy_active = getattr(
            self.env, "_training_last_policy_observation_valid", None
        )
        self._step_policy_active = (
            policy_active.detach().clone() if policy_active is not None else None
        )
        expected = self.plan["frames_per_control_step"]
        if self._written_frames + expected > self.plan["frame_limit"]:
            raise RuntimeError("parallel video capture would exceed its frame limit")
        with self._capture.control_step(expected_frames=expected):
            yield

    def finalize_control_step(self, extras: dict[str, Any]) -> None:
        if not self._pending_frames:
            return
        state = self._step_state(extras)
        for pending in self._pending_frames:
            frame_state = self._frame_state(pending, state)
            record = {
                "frame_index": pending["frame_index"],
                "physics_step": pending["physics_step"],
                **frame_state,
            }
            self._frames_file.write(json.dumps(record, allow_nan=False) + "\n")
            self._written_records += 1
        self._pending_frames = []
        if self._written_frames == self.plan["frame_limit"]:
            self._stop_render_capture()

    def _step_state(self, extras: dict[str, Any]) -> dict[str, Any]:
        snapshot = extras.get("terminal_snapshot")
        success = extras.get("success")
        if not isinstance(snapshot, dict):
            raise RuntimeError("parallel video requires a terminal snapshot")
        policy_active = self._step_policy_active
        if policy_active is None:
            raise RuntimeError("parallel video requires RJ45 policy-active telemetry")
        return {
            "terminal": snapshot["terminal_mask"],
            "success": success,
            "policy_active": policy_active,
            "phase_names": snapshot.get("phase_name_by_code", {}),
        }

    def _frame_state(
        self, pending: dict[str, Any], step_state: dict[str, Any]
    ) -> dict[str, Any]:
        env_ids = range(self.env.num_envs) if self.view == "overview" else range(1)
        environments = []
        for env_id in env_ids:
            terminal_value = bool(_tensor_value(step_state["terminal"], env_id))
            success_value = bool(_tensor_value(step_state["success"], env_id))
            phase_value = int(_tensor_value(pending["phase"], env_id))
            environments.append(
                {
                    "environment_id": env_id,
                    "phase": phase_value,
                    "phase_name": step_state["phase_names"].get(
                        phase_value, "unknown"
                    ),
                    "policy_active": bool(
                        _tensor_value(step_state["policy_active"], env_id)
                    ),
                    "actual_depth_m": float(
                        _tensor_value(pending["actual_depth_m"], env_id)
                    ),
                    "actual_lateral_error_m": float(
                        _tensor_value(pending["actual_lateral_error_m"], env_id)
                    ),
                    "actual_orientation_error_rad": float(
                        _tensor_value(pending["actual_orientation_error_rad"], env_id)
                    ),
                    "terminal": terminal_value,
                    "success": success_value,
                    "failure": terminal_value and not success_value,
                }
            )
        rollout_step = (self._control_step - 1) % self.rollout_steps
        return {
            "control_step": self._control_step,
            "runner_current_learning_iteration": int(
                self._runner.current_learning_iteration
            ),
            "learning_iteration_source": "runner.current_learning_iteration",
            "rollout_step_index": rollout_step,
            "rollout_boundary": (
                "start"
                if rollout_step == 0
                else "end" if rollout_step == self.rollout_steps - 1 else None
            ),
            "state_timing": (
                "phase_and_geometry_at_rendered_physics_step; "
                "terminal_outcome_post_control_step"
            ),
            "environments": environments,
        }

    def _stop_render_capture(self) -> None:
        if not self._render_capture_active:
            return
        self._capture.close()
        self.env.sim.set_setting(
            "/isaaclab/render/rtx_sensors", self._previous_rtx_setting
        )
        self._render_capture_active = False

    def _metadata(self, *, completed: bool) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "completed": completed,
            "scope": "raw viewer camera captured during real vectorized PPO wrapped.step calls",
            "video_path": str(self.video_path),
            "frames_path": str(self.frames_path),
            "camera_view": self.view,
            "camera": self.camera,
            "num_envs": int(self.env.num_envs),
            "simulation_and_renderer_device": str(self.env.device),
            "closeup_environment_id": 0 if self.view == "closeup" else None,
            "capture_plan": self.plan,
            "captured_frames": self._written_frames,
            "frame_metadata_records": self._written_records,
            "captured_control_steps": self._control_step,
            "physics_dt_s": float(self.env.cfg.sim.dt),
            "control_decimation": int(self.env.cfg.decimation),
            "physics_or_action_contract_modified": False,
            "warmup_frames_recorded": False,
            "policy_active_timing": "pre_control_step_action_validity",
            "render_api_validation": (
                "CPU mocks verify cadence and evidence; Isaac Sim 5.1/Isaac Lab 2.3 "
                "viewer API requires GPU preflight before portfolio use"
            ),
        }

    def close(self, *, completed: bool) -> None:
        if self._closed:
            return
        if not self._started:
            self._closed = True
            return
        self._stop_render_capture()
        if self._writer is not None:
            self._writer.close()
        if self._frames_file is not None:
            self._frames_file.close()
        self._closed = True
        capture_completed = (
            self._written_frames == self.plan["frame_limit"]
            and self._written_records == self._written_frames
        )
        metadata = self._metadata(completed=capture_completed)
        metadata["training_call_completed"] = completed
        _atomic_json(self.metadata_path, metadata)
        if self.video_path.is_file() and self.frames_path.is_file():
            _atomic_json(
                self.evidence_path,
                {
                    "schema_version": 1,
                    "completed": metadata["completed"],
                    "artifacts": {
                        "video": {
                            "path": str(self.video_path),
                            "sha256": _sha256(self.video_path),
                        },
                        "frames": {
                            "path": str(self.frames_path),
                            "sha256": _sha256(self.frames_path),
                        },
                        "metadata": {
                            "path": str(self.metadata_path),
                            "sha256": _sha256(self.metadata_path),
                        },
                    },
                    "source_and_config_sha256": self._source_and_config_sha256,
                },
            )
        if completed and not capture_completed:
            raise RuntimeError(
                "parallel video training completed without the requested frame and metadata count"
            )

