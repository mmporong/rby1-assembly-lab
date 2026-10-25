"""Observe rendered physics steps without advancing or changing the controller."""

from contextlib import contextmanager
from typing import Callable


class PhysicsFrameCapture:
    """Capture one rendered frame per expected physics-step interval.

    The caller arms capture only around ``env.step``. Render calls made during
    warmup or terminal auto-reset cannot add extra frames at the same step.
    """

    def __init__(self, sim, step_index: Callable[[], int], read_frame, write_frame, stride: int):
        if stride < 1:
            raise ValueError("capture stride must be positive")
        self.sim = sim
        self.step_index = step_index
        self.read_frame = read_frame
        self.write_frame = write_frame
        self.stride = stride
        self.frame_count = 0
        self.first_step = None
        self.last_step = int(step_index())
        self._original_render = None
        self._armed = False
        self._reading = False

    def install(self):
        if self._original_render is not None:
            raise RuntimeError("capture is already installed")
        self._original_render = self.sim.render
        self.sim.render = self._render

    def close(self):
        self._armed = False
        if self._original_render is not None:
            self.sim.render = self._original_render
            self._original_render = None

    def _render(self, *args, **kwargs):
        result = self._original_render(*args, **kwargs)
        step = int(self.step_index())
        if self._armed and not self._reading and step > self.last_step:
            if step != self.last_step + self.stride:
                raise RuntimeError(f"missing rendered physics step: {self.last_step} -> {step}")
            self._reading = True
            try:
                self.write_frame(self.read_frame())
            finally:
                self._reading = False
            self.first_step = step if self.first_step is None else self.first_step
            self.last_step = step
            self.frame_count += 1
        return result

    @contextmanager
    def control_step(self, expected_frames: int):
        if self._original_render is None or self._armed:
            raise RuntimeError("capture must be installed and idle")
        before = self.frame_count
        self._armed = True
        try:
            yield
        finally:
            self._armed = False
        if self.frame_count - before != expected_frames:
            raise RuntimeError(
                f"rendered frame count {self.frame_count - before} does not match "
                f"the physics-step cadence {expected_frames}"
            )
