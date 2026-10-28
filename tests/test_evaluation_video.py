import ast
import json
from pathlib import Path
import sys
import tempfile
import types

import pytest
import torch

from scripts.rj45_evaluation_video import (
    EvaluationVideoRecorder,
    _evaluator_orientation_error,
    evaluation_video_capture_plan,
    held_initialization_capture_budget,
    selected_environment_camera,
    validate_evaluation_video_request,
)


@pytest.fixture(autouse=True)
def fake_isaaclab_quat_error(monkeypatch):
    calls = []

    def quat_error_magnitude(q1, q2):
        calls.append((q1.clone(), q2.clone()))
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        q2_conjugate = q2.clone()
        q2_conjugate[..., 1:] *= -1.0
        w1, xyz1 = q1[..., :1], q1[..., 1:]
        w2, xyz2 = q2_conjugate[..., :1], q2_conjugate[..., 1:]
        relative_w = w1 * w2 - (xyz1 * xyz2).sum(dim=-1, keepdim=True)
        relative_xyz = w1 * xyz2 + w2 * xyz1 + torch.linalg.cross(xyz1, xyz2)
        sign = torch.where(relative_w < 0.0, -1.0, 1.0)
        relative_w = relative_w * sign
        relative_xyz = relative_xyz * sign
        return 2.0 * torch.atan2(
            torch.linalg.vector_norm(relative_xyz, dim=-1),
            relative_w.squeeze(-1),
        )

    isaaclab_module = types.ModuleType("isaaclab")
    utils_module = types.ModuleType("isaaclab.utils")
    math_module = types.ModuleType("isaaclab.utils.math")
    setattr(math_module, "quat_error_magnitude", quat_error_magnitude)
    setattr(isaaclab_module, "utils", utils_module)
    setattr(utils_module, "math", math_module)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils_module)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_module)
    return calls


class FakeSim:
    def __init__(self):
        self.step = 0
        self.settings = {"/isaaclab/render/rtx_sensors": False}
        self.camera = None

    def render(self):
        return None

    def get_setting(self, name):
        return self.settings[name]

    def set_setting(self, name, value):
        self.settings[name] = value

    def set_camera_view(self, eye, lookat):
        self.camera = (tuple(eye), tuple(lookat))


class FakeWriter:
    def __init__(self, path):
        self.frames = []
        self.closed = False
        path.write_bytes(b"fake-video")

    def append_data(self, frame):
        self.frames.append(frame)

    def close(self):
        self.closed = True


class FakeEnv:
    def __init__(self):
        self.cfg = types.SimpleNamespace(
            sim=types.SimpleNamespace(dt=1.0 / 240.0), decimation=8
        )
        self.sim = FakeSim()
        self.scene = types.SimpleNamespace(
            env_origins=torch.tensor(
                [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=torch.float64
            )
        )
        self.num_envs = 2
        self._training_config = {
            "warmup_s": 0.2,
            "held_reset_staging": {"control_steps": 1},
        }
        self._sim_step_counter = 0
        self._training_episode_index = torch.tensor([0, 0])
        self._training_last_policy_observation_valid = torch.tensor([True, False])
        self._training_last_policy_observation_timestamp_s = torch.tensor(
            [0.0, 0.0], dtype=torch.float64
        )
        self.episode_length_buf = torch.tensor([0, 0])
        self._phase = torch.tensor([4, 3])
        self._last_contact_proxy_n = torch.tensor([[2.0, 3.0], [1.0, 1.5]])
        self._last_insertion_contact_load_n = torch.tensor([4.0, 2.0])
        self.position = torch.tensor([[0.0, 0.01, 0.0], [0.001, 0.002, 0.0]])
        self.quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.99995, 0.01, 0.0, 0.0]]
        )

    def render(self, recompute=False):
        return torch.ones((2, 2, 3), dtype=torch.uint8).numpy()

    def _relative_object_pose(self):
        return self.position, self.quaternion

    def _identity_quat(self):
        identity = torch.zeros_like(self.quaternion)
        identity[:, 0] = 1.0
        return identity


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    (
        ([1.0, 0.0, 0.0, 0.0], 0.0),
        ([-1.0, 0.0, 0.0, 0.0], 0.0),
        ([0.999999875, 0.0005, 0.0, 0.0], 0.001),
        ([-0.999999875, -0.0005, 0.0, 0.0], 0.001),
        ([0.9238795325, 0.0, 0.3826834324, 0.0], torch.pi / 4.0),
    ),
)
def test_orientation_error_reuses_evaluator_quaternion_api(
    fake_isaaclab_quat_error, quaternion, expected
):
    relative = torch.tensor([quaternion], dtype=torch.float64)
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    result = _evaluator_orientation_error(relative, identity)

    assert result.item() == pytest.approx(float(expected), abs=1.0e-9)
    assert len(fake_isaaclab_quat_error) == 1
    torch.testing.assert_close(fake_isaaclab_quat_error[0][0], relative)
    torch.testing.assert_close(fake_isaaclab_quat_error[0][1], identity)


def extras(
    *,
    success=True,
    episode_index=0,
    terminal_timestamp_s=1.0 / 30.0,
    policy_input_timestamp_s=0.0,
):
    return {
        "success": torch.tensor([success, False]),
        "evaluation": {"valid": torch.tensor([True, True])},
        "terminal_evaluation": {
            "termination_reason": ["independent_success", "none"],
            "criteria": {
                "depth": torch.tensor([True, False]),
                "lateral": torch.tensor([True, True]),
                "orientation": torch.tensor([True, True]),
                "grasp_live_and_proven": torch.tensor([True, False]),
                "dwell": torch.tensor([True, False]),
                "force": torch.tensor([True, True]),
                "forbidden_contact": torch.tensor([True, True]),
            },
        },
        "terminal_snapshot": {
            "episode_index": torch.tensor([episode_index, 0]),
            "timestamp_s": torch.tensor(
                [terminal_timestamp_s, terminal_timestamp_s]
            ),
            "policy_input_observation": torch.arange(
                158, dtype=torch.float64
            ).reshape(2, 79),
            "policy_input_timestamp_s": torch.tensor(
                [policy_input_timestamp_s, policy_input_timestamp_s],
                dtype=torch.float64,
            ),
            "policy_input_valid": torch.tensor([True, False]),
            "requested_expanded_residual_action": torch.arange(
                24, dtype=torch.float64
            ).reshape(2, 12),
            "applied_bounded_residual_action": torch.arange(
                24, dtype=torch.float64
            ).reshape(2, 12)
            / 10.0,
            "applied_tip_target_offset_entry_6d": torch.arange(
                12, dtype=torch.float64
            ).reshape(2, 6)
            / 100.0,
            "independent_eval_dwell_s": torch.tensor([0.4, 0.0]),
            "grip_live": torch.tensor([[True, True], [True, False]]),
            "grip_proven": torch.tensor([[True, True], [True, False]]),
            "rj45_evaluation_stability": {
                "depth_within_window": torch.tensor([True, False]),
                "relative_linear_speed_ok": torch.tensor([True, True]),
                "relative_angular_speed_ok": torch.tensor([True, True]),
                "relative_linear_speed_m_s": torch.tensor([0.001, 0.002]),
                "relative_angular_speed_rad_s": torch.tensor([0.01, 0.02]),
                "physics_substeps_complete": torch.tensor([True, True]),
                "physics_all_samples_stable": torch.tensor([True, False]),
                "physics_sample_finite": torch.ones((2, 9), dtype=torch.bool),
                "physics_sample_stable": torch.tensor(
                    [[True] * 9, [True] * 8 + [False]]
                ),
                "physics_sample_depth_m": torch.full((2, 9), 0.012),
                "physics_sample_lateral_error_m": torch.full((2, 9), 0.0001),
                "physics_sample_orientation_error_rad": torch.full((2, 9), 0.001),
                "physics_sample_relative_linear_speed_m_s": torch.full(
                    (2, 9), 0.001
                ),
                "physics_sample_relative_angular_speed_rad_s": torch.full(
                    (2, 9), 0.01
                ),
            },
            "actual_depth_m": torch.tensor([0.01, 0.002]),
            "actual_lateral_error_m": torch.tensor([0.0, 0.001]),
            "actual_orientation_error_rad": torch.tensor([0.0, 0.02]),
            "phase_name_by_code": {3: "align", 4: "insert"},
        },
    }


def recorder(
    directory: str,
    env: FakeEnv,
    writers: list[FakeWriter],
    *,
    max_seconds: float = 1.0 / 30.0,
):
    def factory(path, fps):
        assert fps == 60
        result = FakeWriter(path)
        writers.append(result)
        return result

    return EvaluationVideoRecorder(
        env,
        video_path=Path(directory) / "evaluation.mp4",
        env_index=0,
        fps=60,
        max_seconds=max_seconds,
        policy_lineage={"mode": "learned", "checkpoint": {"sha256": "a" * 64}},
        source_and_config_sha256={"scripts/evaluate_policy.py": "b" * 64},
        writer_factory=factory,
        frame_reader=lambda: "rgb",
    )


def test_plan_is_native_240hz_stride4_and_two_frames_per_control_step():
    plan = evaluation_video_capture_plan(
        physics_dt_s=1.0 / 240.0,
        control_decimation=8,
        fps=60,
        max_seconds=24.0,
    )
    assert plan["render_stride_physics_steps"] == 4
    assert plan["frames_per_control_step"] == 2
    assert plan["requested_frame_limit"] == 1440
    assert plan["terminal_margin_control_steps"] == 1
    assert plan["terminal_margin_frames"] == 2
    assert plan["terminal_margin_seconds"] == pytest.approx(1.0 / 30.0)
    assert plan["frame_limit"] == 1442
    assert plan["max_seconds"] == pytest.approx(24.0 + 1.0 / 30.0)
    held_plan = evaluation_video_capture_plan(
        physics_dt_s=1.0 / 240.0,
        control_decimation=8,
        fps=60,
        max_seconds=24.0,
        warmup_budget_control_steps=31,
    )
    assert held_plan["warmup_budget_control_steps"] == 31
    assert held_plan["warmup_budget_seconds"] == pytest.approx(31.0 / 30.0)
    assert held_plan["frame_limit"] == 1504
    assert held_plan["max_seconds"] == pytest.approx(24.0 + 32.0 / 30.0)
    fallback_plan = evaluation_video_capture_plan(
        physics_dt_s=1.0 / 240.0,
        control_decimation=8,
        fps=60,
        max_seconds=24.0,
        warmup_budget_control_steps=None,
    )
    assert fallback_plan["frame_limit"] == 2400
    assert fallback_plan["max_seconds"] == 40.0
    with pytest.raises(ValueError, match="40"):
        evaluation_video_capture_plan(
            physics_dt_s=1.0 / 240.0,
            control_decimation=8,
            fps=60,
            max_seconds=40.1,
        )
    cadence_preflight_plan = evaluation_video_capture_plan(
        physics_dt_s=1.0 / 240.0,
        control_decimation=8,
        fps=60,
        max_seconds=40.0,
    )
    assert cadence_preflight_plan["frame_limit"] == 2400
    assert cadence_preflight_plan["terminal_margin_control_steps"] == 0
    assert cadence_preflight_plan["terminal_margin_frames"] == 0


def test_held_initialization_budget_uses_warmup_and_staging_contract():
    env = FakeEnv()
    budget = held_initialization_capture_budget(env, control_dt_s=1.0 / 30.0)
    assert budget == {
        "control_steps": 31,
        "source": "rj45_training_env_held_warmup_timeout_plus_config_staging",
        "warmup_s": 0.2,
        "warmup_timeout_s": 1.0,
        "held_reset_staging_control_steps": 1,
    }

    del env._training_config["held_reset_staging"]
    fallback = held_initialization_capture_budget(env, control_dt_s=1.0 / 30.0)
    assert fallback["control_steps"] is None
    assert fallback["source"] == "40_second_hard_cap_unbounded_training_config"


def test_cli_contract_requires_cameras_60fps_and_positive_selected_quota():
    validate_evaluation_video_request(
        requested=True, task="rj45", baseline_mode="learned",
        cameras_enabled=True, fps=60, env_index=63, num_envs=64, episodes=64,
    )
    validate_evaluation_video_request(
        requested=False, task="rj45", baseline_mode="learned",
        cameras_enabled=True, fps=24, env_index=63, num_envs=64, episodes=64,
    )
    cases = (
        (False, 60, 63, 64, 64, "learned", "enable_cameras"),
        (True, 30, 63, 64, 64, "learned", "video-fps 60"),
        (True, 60, 64, 64, 64, "learned", "active environment"),
        (True, 60, 32, 64, 32, "learned", "positive episode quota"),
        (True, 60, 63, 64, 64, "oracle", "learned, bc, or biased_zero"),
    )
    for cameras, fps, env_index, num_envs, episodes, mode, message in cases:
        with pytest.raises(ValueError, match=message):
            validate_evaluation_video_request(
                requested=True,
                task="rj45",
                baseline_mode=mode,
                cameras_enabled=cameras,
                fps=fps,
                env_index=env_index,
                num_envs=num_envs,
                episodes=episodes,
            )


def test_camera_uses_selected_environment_origin():
    camera = selected_environment_camera(
        torch.tensor([[0.0, 0.0, 0.0], [5.0, 2.5, 0.0]]), 1
    )
    assert camera["eye_world_m"] == pytest.approx((5.95, 1.9, 1.25))
    assert camera["lookat_world_m"] == pytest.approx((5.36, 2.5, 0.87))


@pytest.mark.parametrize("initial_episode", [0, 1])
def test_real_step_cadence_stops_at_initial_episode_terminal_without_reset_frame(
    initial_episode,
):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        env._training_episode_index[0] = initial_episode
        writers = []
        video = recorder(directory, env, writers)
        state_before = {
            "episode": env._training_episode_index.clone(),
            "phase": env._phase.clone(),
            "position": env.position.clone(),
            "valid": env._training_last_policy_observation_valid.clone(),
        }
        video.initialize()
        assert env._sim_step_counter == 0
        with video.control_step():
            for physics_step, contact in ((4, 9.0), (8, 12.0)):
                env._last_contact_proxy_n[0] = contact
                env._last_insertion_contact_load_n[0] = contact
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        # DirectRLEnv auto-reset has happened by the time wrapped.step returns.
        env._training_episode_index[0] = initial_episode + 1
        video.finalize_control_step(
            extras(episode_index=initial_episode), torch.tensor([True, False])
        )
        evidence = video.close(rollout_completed=True)

        assert evidence is not None
        assert evidence["completed"] is True
        assert writers[0].frames == ["rgb", "rgb"]
        rows = [
            json.loads(line)
            for line in video.frames_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["physics_step"] for row in rows] == [4, 8]
        assert [row["simulation_time_s"] for row in rows] == pytest.approx(
            [4 / 240, 8 / 240]
        )
        assert all(row["episode_index"] == initial_episode for row in rows)
        assert all(row["frame_state_is_pre_autoreset_physics"] for row in rows)
        assert rows[0]["success_observed_after_control_step"] is False
        assert rows[1]["success_observed_after_control_step"] is True
        assert rows[1]["control_step_entry_contact_force_proxy_n"] == 4.0
        assert rows[1]["contact_timing"] == "control_step_entry_not_rendered_physics_step"
        assert rows[1]["control_step_entry_policy_input_timestamp_s"] == 0.0
        assert rows[1]["control_time_s"] == 0.0
        assert rows[1]["time_since_policy_input_s"] == pytest.approx(1.0 / 30.0)
        assert rows[1]["policy_input_valid"] is True
        assert rows[1]["policy_lineage_sha256"] == evidence["policy_lineage_sha256"]
        assert rows[0]["control_step_snapshot"] == rows[1]["control_step_snapshot"]
        step_snapshot = rows[1]["control_step_snapshot"]
        assert len(step_snapshot["policy_input_observation"]) == 79
        assert len(step_snapshot["requested_expanded_residual_action"]) == 12
        assert len(step_snapshot["applied_bounded_residual_action"]) == 12
        assert len(step_snapshot["applied_tip_target_offset_entry_6d"]) == 6
        assert step_snapshot["policy_input_timestamp_s"] == 0.0
        assert step_snapshot["policy_input_valid"] is True
        assert step_snapshot["independent_eval_dwell_s"] == pytest.approx(0.4)
        assert step_snapshot["grip_live"] == [True, True]
        assert step_snapshot["grip_proven"] == [True, True]
        stability = step_snapshot["rj45_evaluation_stability"]
        assert stability["physics_sample_count"] == 9
        assert len(stability["physics_sample_depth_m"]) == 9
        assert len(stability["physics_sample_relative_linear_speed_m_s"]) == 9
        assert stability["physics_substeps_complete"] is True
        assert all(stability["physics_sample_finite"])
        assert step_snapshot["terminal_evaluation_criteria"] == {
            "depth": True,
            "lateral": True,
            "orientation": True,
            "grasp_live_and_proven": True,
            "dwell": True,
            "force": True,
            "forbidden_contact": True,
        }
        assert "control-step start" in step_snapshot["timing_contract"]
        assert "control-step end" in step_snapshot["timing_contract"]
        torch.testing.assert_close(env._phase, state_before["phase"])
        torch.testing.assert_close(env.position, state_before["position"])
        torch.testing.assert_close(
            env._training_last_policy_observation_valid, state_before["valid"]
        )
        assert env._training_episode_index[0].item() == initial_episode + 1
        assert env.sim.settings["/isaaclab/render/rtx_sensors"] is False
        metadata = json.loads(video.metadata_path.read_text(encoding="utf-8"))
        assert metadata["selected_episode_index"] == initial_episode
        assert metadata["selected_terminal"]["episode_index"] == initial_episode
        assert metadata["selected_terminal"]["timestamp_s"] == pytest.approx(1 / 30)
        assert metadata["selected_terminal"]["actual_depth_m"] == pytest.approx(0.01)
        assert metadata["selected_terminal"]["last_pre_autoreset_physics_step"] == 8


def test_capture_rejects_reset_before_initial_episode_terminal():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        env._training_episode_index[0] = 1
        video = recorder(directory, env, [])
        video.initialize()
        env._training_episode_index[0] = 2
        with pytest.raises(
            RuntimeError, match="reset before terminal capture was finalized"
        ):
            with video.control_step():
                env._sim_step_counter = 4
                env.sim.step = 4
                env.sim.render()
        video.close(rollout_completed=False)


def test_terminal_margin_captures_timeout_control_step_without_reset_frame():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        writers = []
        video = recorder(directory, env, writers, max_seconds=1.0 / 30.0)
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        video.finalize_control_step(extras(success=False), torch.tensor([False, False]))

        env.episode_length_buf[0] = 1
        env._training_last_policy_observation_timestamp_s[0] = 1.0 / 30.0
        with video.control_step():
            for physics_step in (12, 16):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        env._training_episode_index[0] = 1
        video.finalize_control_step(
            extras(
                success=False,
                terminal_timestamp_s=2.0 / 30.0,
                policy_input_timestamp_s=1.0 / 30.0,
            ),
            torch.tensor([True, False]),
        )
        evidence = video.close(rollout_completed=True)

        assert evidence is not None
        assert evidence["completed"] is True
        rows = [
            json.loads(line)
            for line in video.frames_path.read_text(encoding="utf-8").splitlines()
        ]
        assert [row["physics_step"] for row in rows] == [4, 8, 12, 16]
        assert all(row["episode_index"] == 0 for row in rows)
        assert rows[-1]["terminal_observed_after_control_step"] is True
        metadata = json.loads(video.metadata_path.read_text(encoding="utf-8"))
        assert metadata["capture_plan"]["requested_frame_limit"] == 2
        assert metadata["capture_plan"]["terminal_margin_frames"] == 2
        assert metadata["captured_frames"] == 4
        assert metadata["captured_duration_s"] == pytest.approx(2.0 / 30.0)


def test_metadata_separates_observed_initialization_offset_from_horizon_budget():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        writers = []
        video = recorder(directory, env, writers, max_seconds=24.0)
        video.initialize()

        observed_initialization_offset_s = 7.0 / 30.0
        env._training_last_policy_observation_timestamp_s[0] = (
            observed_initialization_offset_s
        )
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        video.finalize_control_step(
            extras(
                success=False,
                policy_input_timestamp_s=observed_initialization_offset_s,
            ),
            torch.tensor([False, False]),
        )

        env.episode_length_buf[0] = 719
        terminal_policy_timestamp_s = 719.0 / 30.0 + observed_initialization_offset_s
        env._training_last_policy_observation_timestamp_s[0] = (
            terminal_policy_timestamp_s
        )
        with video.control_step():
            for physics_step in (12, 16):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        env._training_episode_index[0] = 1
        video.finalize_control_step(
            extras(
                success=False,
                terminal_timestamp_s=24.0,
                policy_input_timestamp_s=terminal_policy_timestamp_s,
            ),
            torch.tensor([True, False]),
        )
        video.close(rollout_completed=True)

        metadata = json.loads(video.metadata_path.read_text(encoding="utf-8"))
        assert metadata["capture_plan"]["warmup_budget_control_steps"] == 31
        assert metadata["capture_plan"]["warmup_budget_seconds"] == pytest.approx(
            31.0 / 30.0
        )
        assert metadata["capture_plan"]["frame_limit"] == 1504
        assert metadata["observed_held_initialization_offset_s"] == pytest.approx(
            observed_initialization_offset_s
        )
        assert metadata["observed_held_initialization_control_steps"] == 7
        assert metadata["elapsed_clock_contract"] == {
            "requested_episode_horizon_s": 24.0,
            "capture_elapsed_s": pytest.approx(4.0 / 60.0),
            "selected_terminal_episode_elapsed_s": 24.0,
        }


def test_frame_writer_rejects_data_beyond_terminal_margin_limit():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [], max_seconds=1.0 / 30.0)
        video.initialize()
        video._written_frames = int(video.plan["frame_limit"])
        with pytest.raises(RuntimeError, match="frame limit exceeded"):
            video._write_frame("rgb")
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    "episode_index",
    [torch.tensor([-1, 0]), torch.tensor([1.5, 0.0])],
)
def test_capture_rejects_invalid_initial_episode_index(episode_index):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        env._training_episode_index = episode_index
        video = recorder(directory, env, [])
        with pytest.raises(RuntimeError, match="non-negative integer"):
            video.initialize()


def test_policy_input_timestamp_must_be_finite_and_strictly_increasing():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [], max_seconds=2.0 / 30.0)
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        video.finalize_control_step(extras(success=False), torch.tensor([False, False]))
        env.episode_length_buf[0] = 1
        with pytest.raises(RuntimeError, match="must increase"):
            with video.control_step():
                pass
        video.close(rollout_completed=False)

    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        env._training_last_policy_observation_timestamp_s[0] = float("nan")
        video = recorder(directory, env, [])
        video.initialize()
        with pytest.raises(RuntimeError, match="finite and non-negative"):
            with video.control_step():
                pass
        video.close(rollout_completed=False)


@pytest.mark.parametrize("mismatch", ["geometry", "physics_boundary", "timestamp"])
def test_terminal_snapshot_must_match_last_pre_autoreset_frame(mismatch):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        if mismatch == "geometry":
            terminal["terminal_snapshot"]["actual_depth_m"][0] = 0.02
            message = "actual_depth_m differs"
        elif mismatch == "physics_boundary":
            env._sim_step_counter = 12
            message = "physics boundary"
        else:
            terminal["terminal_snapshot"]["timestamp_s"][0] = 0.1
            message = "terminal timestamp"
        with pytest.raises(RuntimeError, match=message):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("policy_input_observation", torch.zeros((2, 78)), "shape \\(79,\\)"),
        ("requested_expanded_residual_action", torch.zeros((2, 11)), "shape \\(12,\\)"),
        ("applied_bounded_residual_action", torch.zeros((2, 13)), "shape \\(12,\\)"),
        ("applied_tip_target_offset_entry_6d", torch.zeros((2, 5)), "shape \\(6,\\)"),
    ),
)
def test_control_step_snapshot_rejects_wrong_vector_shape(field, value, message):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        terminal["terminal_snapshot"][field] = value
        with pytest.raises(RuntimeError, match=message):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    "field",
    (
        "policy_input_observation",
        "requested_expanded_residual_action",
        "applied_bounded_residual_action",
        "applied_tip_target_offset_entry_6d",
    ),
)
def test_control_step_snapshot_rejects_nonfinite_vector(field):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        terminal["terminal_snapshot"][field][0, 0] = float("nan")
        with pytest.raises(RuntimeError, match=f"{field} must be finite"):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    "field",
    (
        "policy_input_observation",
        "policy_input_timestamp_s",
        "policy_input_valid",
        "requested_expanded_residual_action",
        "applied_bounded_residual_action",
        "applied_tip_target_offset_entry_6d",
    ),
)
def test_control_step_snapshot_rejects_missing_field(field):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        del terminal["terminal_snapshot"][field]
        with pytest.raises(RuntimeError, match=field):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("policy_input_timestamp_s", float("nan"), "does not match"),
        ("policy_input_timestamp_s", 1.0, "does not match"),
        ("policy_input_valid", 1, "must be boolean"),
    ),
)
def test_control_step_snapshot_rejects_invalid_scalar(field, value, message):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        if field == "policy_input_valid":
            terminal["terminal_snapshot"][field] = torch.tensor([value, 0])
        else:
            terminal["terminal_snapshot"][field][0] = value
        with pytest.raises(RuntimeError, match=message):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    "field",
    (
        "physics_sample_finite",
        "physics_sample_stable",
        "physics_sample_depth_m",
        "physics_sample_lateral_error_m",
        "physics_sample_orientation_error_rad",
        "physics_sample_relative_linear_speed_m_s",
        "physics_sample_relative_angular_speed_rad_s",
    ),
)
def test_physics_stability_snapshot_requires_decimation_plus_endpoint(field):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        stability = terminal["terminal_snapshot"]["rj45_evaluation_stability"]
        stability[field] = stability[field][:, :8]
        with pytest.raises(RuntimeError, match="shape \\(9,\\)"):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


@pytest.mark.parametrize(
    "field",
    (
        "physics_sample_depth_m",
        "physics_sample_lateral_error_m",
        "physics_sample_orientation_error_rad",
        "physics_sample_relative_linear_speed_m_s",
        "physics_sample_relative_angular_speed_rad_s",
    ),
)
def test_physics_stability_snapshot_rejects_nonfinite_sample(field):
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        stability = terminal["terminal_snapshot"]["rj45_evaluation_stability"]
        stability[field][0, 4] = float("nan")
        with pytest.raises(RuntimeError, match=f"{field} must be finite"):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


def test_physics_stability_snapshot_rejects_nonboolean_sample_flags():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        video = recorder(directory, env, [])
        video.initialize()
        with video.control_step():
            for physics_step in (4, 8):
                env._sim_step_counter = physics_step
                env.sim.step = physics_step
                env.sim.render()
        terminal = extras()
        stability = terminal["terminal_snapshot"]["rj45_evaluation_stability"]
        stability["physics_sample_finite"] = torch.ones((2, 9))
        with pytest.raises(RuntimeError, match="physics_sample_finite must be boolean"):
            video.finalize_control_step(terminal, torch.tensor([True, False]))
        video.close(rollout_completed=False)


def test_control_step_snapshot_requires_stability_dwell_grip_and_criteria():
    mutations = (
        lambda payload: payload["terminal_snapshot"].pop(
            "rj45_evaluation_stability"
        ),
        lambda payload: payload["terminal_snapshot"].pop(
            "independent_eval_dwell_s"
        ),
        lambda payload: payload["terminal_snapshot"].pop("grip_live"),
        lambda payload: payload["terminal_snapshot"].pop("grip_proven"),
        lambda payload: payload["terminal_evaluation"].pop("criteria"),
    )
    messages = (
        "rj45_evaluation_stability",
        "independent_eval_dwell_s",
        "grip_live",
        "grip_proven",
        "terminal_evaluation.criteria",
    )
    for mutate, message in zip(mutations, messages, strict=True):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv()
            video = recorder(directory, env, [])
            video.initialize()
            with video.control_step():
                for physics_step in (4, 8):
                    env._sim_step_counter = physics_step
                    env.sim.step = physics_step
                    env.sim.render()
            terminal = extras()
            mutate(terminal)
            with pytest.raises(RuntimeError, match=message):
                video.finalize_control_step(terminal, torch.tensor([True, False]))
            video.close(rollout_completed=False)


def test_partial_capture_is_preserved_and_marked_incomplete():
    with tempfile.TemporaryDirectory() as directory:
        env = FakeEnv()
        writers = []
        video = recorder(directory, env, writers)
        video.initialize()
        evidence = video.close(rollout_completed=False)
        assert evidence is not None
        assert evidence["completed"] is False
        assert evidence["rollout_completed"] is False
        assert evidence["artifacts"]["video"]["sha256"]
        metadata = json.loads(video.metadata_path.read_text(encoding="utf-8"))
        assert metadata["stop_reason"] == "evaluation_closed_before_selected_terminal"
        assert metadata["warmup_physics_steps_added"] == 0


def test_evaluate_cli_is_opt_in_and_wraps_only_real_env_step():
    source = (
        Path(__file__).resolve().parents[1] / "scripts/evaluate_policy.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--evaluation-video", type=Path)' in source
    assert "validate_evaluation_video_request(" in source
    assert 'env_type(cfg, render_mode="rgb_array")' in source
    assert "else env_type(cfg)" in source
    assert "with evaluation_video.control_step():" in source
    assert "evaluation_video.finalize_control_step(extras, dones)" in source
    assert "evaluation_video.close(" in source
    assert "rollout_completed=True" in source
    assert 'EVALUATION_VIDEO_HELPER = SCRIPTS_ROOT / "rj45_evaluation_video.py"' in source
    tree = ast.parse(source)
    capture_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "capture_run_inputs"
    ]
    assert len(capture_calls) == 2
    for call in capture_calls:
        source_files = next(
            keyword.value for keyword in call.keywords if keyword.arg == "source_files"
        )
        assert isinstance(source_files, ast.Tuple)
        assert any(
            isinstance(element, ast.Name)
            and element.id == "EVALUATION_VIDEO_HELPER"
            for element in source_files.elts
        )
