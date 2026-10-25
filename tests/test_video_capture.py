import unittest

from rby1_assembly.video_capture import PhysicsFrameCapture


class FakeSim:
    def __init__(self):
        self.step = 0
        self.render_calls = 0

    def render(self):
        self.render_calls += 1
        return "rendered"


class TestPhysicsFrameCapture(unittest.TestCase):
    def make_capture(self, stride=1):
        sim = FakeSim()
        frames = []
        capture = PhysicsFrameCapture(sim, lambda: sim.step, lambda: sim.step, frames.append, stride)
        capture.install()
        self.addCleanup(capture.close)
        return sim, frames, capture

    def test_physics_steps_and_terminal_reset_render_are_captured_once(self):
        sim, frames, capture = self.make_capture()
        sim.render()  # Warmup must not enter the video.
        with capture.control_step(expected_frames=4):
            for step in range(1, 5):
                sim.step = step
                sim.render()
            sim.render()  # Auto-reset can render again at the same counter.
        sim.render()
        self.assertEqual(frames, [1, 2, 3, 4])
        self.assertEqual((capture.first_step, capture.last_step, capture.frame_count), (1, 4, 4))

    def test_render_stride_and_restore(self):
        sim, frames, capture = self.make_capture(stride=2)
        original = capture._original_render
        with capture.control_step(expected_frames=2):
            for step in (2, 4):
                sim.step = step
                sim.render()
        capture.close()
        self.assertEqual(sim.render, original)
        self.assertEqual(frames, [2, 4])

    def test_frame_is_read_after_original_render(self):
        sim, frames, capture = self.make_capture()
        capture.read_frame = lambda: sim.render_calls
        with capture.control_step(expected_frames=1):
            sim.step = 1
            sim.render()
        self.assertEqual(frames, [1])

    def test_missing_render_fails(self):
        sim, _, capture = self.make_capture()
        with self.assertRaisesRegex(RuntimeError, "missing rendered physics step"):
            with capture.control_step(expected_frames=1):
                sim.step = 2
                sim.render()
        self.assertFalse(capture._armed)

    def test_empty_control_step_fails(self):
        _, _, capture = self.make_capture()
        with self.assertRaisesRegex(RuntimeError, "frame count"):
            with capture.control_step(expected_frames=4):
                pass

    def test_reader_exception_disarms_and_restores(self):
        sim, _, capture = self.make_capture()

        def fail():
            raise ValueError("frame read failed")

        capture.read_frame = fail
        with self.assertRaisesRegex(ValueError, "frame read failed"):
            with capture.control_step(expected_frames=1):
                sim.step = 1
                sim.render()
        self.assertFalse(capture._armed)
        self.assertFalse(capture._reading)
        capture.close()
        self.assertEqual(sim.render(), "rendered")

    def test_duplicate_install_is_rejected(self):
        _, _, capture = self.make_capture()
        with self.assertRaises(RuntimeError):
            capture.install()

    def test_invalid_stride_is_rejected(self):
        with self.assertRaises(ValueError):
            PhysicsFrameCapture(FakeSim(), lambda: 0, lambda: 0, lambda _: None, 0)


if __name__ == "__main__":
    unittest.main()
