import json
from pathlib import Path
import tempfile
import types
import unittest

import torch

from rby1_assembly.parallel_training_video import (
    ParallelTrainingVideoRecorder,
    camera_pose_from_env_origins,
    parallel_video_capture_plan,
)


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
        self.path = path
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
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
        )
        self.num_envs = 2
        self.device = "cuda:3"
        self._training_last_policy_observation_valid = torch.tensor([True, False])
        self._phase = torch.tensor([8, 3])

    @property
    def _sim_step_counter(self):
        return self.sim.step

    def render(self, recompute=False):
        return torch.ones((2, 2, 3), dtype=torch.uint8).numpy()

    def _relative_object_pose(self):
        position = torch.tensor([[0.0, 0.01, 0.0], [0.001, 0.002, 0.0]])
        quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.99995, 0.01, 0.0, 0.0]])
        return position, quaternion


def step_extras():
    return {
        "success": torch.tensor([True, False]),
        "terminal_snapshot": {
            "terminal_mask": torch.tensor([True, False]),
            "phase": torch.tensor([8, 3]),
            "phase_name_by_code": {8: "done", 3: "align"},
        },
        "evaluation": {
            "actual_depth_m": torch.tensor([0.01, 0.002]),
            "actual_lateral_error_m": torch.tensor([0.0, 0.001]),
            "actual_orientation_error_rad": torch.tensor([0.0, 0.02]),
        },
    }


class ParallelTrainingVideoTests(unittest.TestCase):
    def test_train_and_evaluate_pin_app_launcher_to_one_selected_gpu(self):
        root = Path(__file__).resolve().parents[1]
        train_source = (root / "scripts/train.py").read_text(encoding="utf-8")
        evaluate_source = (root / "scripts/evaluate_policy.py").read_text(
            encoding="utf-8"
        )

        launcher = "AppLauncher(args, multi_gpu=False)"
        self.assertIn(launcher, train_source)
        self.assertIn(launcher, evaluate_source)
        self.assertIn('"selected_device": str(args.device)', train_source)
        self.assertIn('"selected_device": str(args.device)', evaluate_source)

    def test_train_video_requires_camera_flag_and_rgb_render_mode(self):
        root = Path(__file__).resolve().parents[1]
        source = (root / "scripts/train.py").read_text(encoding="utf-8")

        self.assertIn("--parallel-video requires --enable_cameras", source)
        self.assertIn(
            'render_mode="rgb_array" if args.parallel_video is not None else None',
            source,
        )

    def test_capture_plan_matches_240hz_physics_and_60fps(self):
        plan = parallel_video_capture_plan(
            physics_dt_s=1.0 / 240.0,
            control_decimation=8,
            fps=60,
            seconds=16.0,
        )
        self.assertEqual(plan["render_stride_physics_steps"], 4)
        self.assertEqual(plan["frames_per_control_step"], 2)
        self.assertEqual(plan["frame_limit"], 960)

    def test_overview_uses_full_grid_and_closeup_uses_environment_zero(self):
        origins = torch.tensor([[0.0, 0.0, 0.0], [5.0, 2.5, 0.0]])
        overview = camera_pose_from_env_origins(origins, "overview")
        closeup = camera_pose_from_env_origins(origins, "closeup")

        self.assertGreater(overview["eye_world_m"][2], 1.4)
        self.assertEqual(closeup["eye_world_m"], (0.95, -0.6, 1.25))
        self.assertEqual(closeup["lookat_world_m"], (0.36, 0.0, 0.87))

    def test_real_step_cadence_writes_frames_and_post_step_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv()
            writers = []

            def writer_factory(path: Path, fps: int):
                self.assertEqual(fps, 60)
                writer = FakeWriter(path)
                writers.append(writer)
                return writer

            video = Path(directory) / "parallel.mp4"
            recorder = ParallelTrainingVideoRecorder(
                env,
                video_path=video,
                seconds=1.0 / 30.0,
                fps=60,
                view="overview",
                rollout_steps=128,
                writer_factory=writer_factory,
                frame_reader=lambda: "rgb",
            )
            runner = types.SimpleNamespace(current_learning_iteration=3)
            recorder.bind_runner(runner)
            recorder.set_evidence_sources({"scripts/train.py": "a" * 64})
            recorder.initialize()

            with recorder.control_step():
                for physics_step in (4, 8):
                    env.sim.step = physics_step
                    env.sim.render()
                env._training_last_policy_observation_valid = torch.tensor(
                    [False, True]
                )
            recorder.finalize_control_step(step_extras())
            runner.current_learning_iteration = 4
            with recorder.control_step():
                pass
            recorder.finalize_control_step(step_extras())
            recorder.close(completed=True)

            self.assertEqual(writers[0].frames, ["rgb", "rgb"])
            self.assertTrue(writers[0].closed)
            self.assertFalse(env.sim.settings["/isaaclab/render/rtx_sensors"])
            rows = [
                json.loads(line)
                for line in Path(str(video) + ".frames.jsonl").read_text().splitlines()
            ]
            self.assertEqual([row["physics_step"] for row in rows], [4, 8])
            self.assertEqual(rows[0]["runner_current_learning_iteration"], 3)
            self.assertEqual(rows[0]["rollout_boundary"], "start")
            self.assertEqual(len(rows[0]["environments"]), 2)
            self.assertTrue(rows[0]["environments"][0]["policy_active"])
            self.assertTrue(rows[0]["environments"][0]["success"])
            self.assertTrue(rows[0]["environments"][0]["terminal"])
            self.assertEqual(rows[0]["environments"][0]["phase_name"], "done")
            evidence = json.loads(
                Path(str(video) + ".evidence.json").read_text()
            )
            self.assertTrue(evidence["completed"])
            self.assertEqual(
                evidence["source_and_config_sha256"]["scripts/train.py"], "a" * 64
            )

    def test_completed_training_rejects_incomplete_capture(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv()
            recorder = ParallelTrainingVideoRecorder(
                env,
                video_path=Path(directory) / "parallel.mp4",
                seconds=1.0 / 30.0,
                fps=60,
                view="overview",
                rollout_steps=128,
                writer_factory=lambda path, fps: FakeWriter(path),
                frame_reader=lambda: "rgb",
            )
            recorder.bind_runner(types.SimpleNamespace(current_learning_iteration=0))
            recorder.initialize()
            with self.assertRaisesRegex(
                RuntimeError, "without the requested frame and metadata count"
            ):
                recorder.close(completed=True)

    def test_missing_scheduled_frame_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            env = FakeEnv()
            recorder = ParallelTrainingVideoRecorder(
                env,
                video_path=Path(directory) / "parallel.mp4",
                seconds=1.0 / 30.0,
                fps=60,
                view="closeup",
                rollout_steps=128,
                writer_factory=lambda path, fps: FakeWriter(path),
                frame_reader=lambda: "rgb",
            )
            recorder.bind_runner(types.SimpleNamespace(current_learning_iteration=0))
            recorder.initialize()
            with self.assertRaisesRegex(RuntimeError, "frame count"):
                with recorder.control_step():
                    env.sim.step = 4
                    env.sim.render()
            recorder.close(completed=False)


if __name__ == "__main__":
    unittest.main()
