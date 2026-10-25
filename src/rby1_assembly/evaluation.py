"""양팔 조립 성공 여부를 제어기 단계와 독립적으로 판정한다."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .contracts import EvaluationSample, TaskLimits


@dataclass(frozen=True)
class EvaluationResult:
    terminal: bool
    success: bool
    dwell_progress_s: float
    reasons: Tuple[str, ...] = ()


class TaskEvaluator:
    """정답 상태를 쓰는 독립 평가기.

    성공은 삽입 깊이, 횡방향/회전 오차, 양손 파지가 dwell 동안 계속
    만족할 때만 성립한다. 힘 초과와 금지 접촉은 에피소드 어느 시점에
    발생해도 실패로 보존된다.
    """

    def __init__(self, limits: TaskLimits) -> None:
        self.limits = limits
        self.reset()

    def reset(self) -> None:
        self._start_time_s: Optional[float] = None
        self._last_time_s: Optional[float] = None
        self._candidate_since_s: Optional[float] = None
        self._terminal_result: Optional[EvaluationResult] = None
        self._force_violation = False
        self._forbidden_contact = False

    def update(self, sample: EvaluationSample) -> EvaluationResult:
        if self._terminal_result is not None:
            raise RuntimeError("evaluation already reached a terminal result; call reset()")

        if self._last_time_s is not None and sample.sim_time_s <= self._last_time_s:
            raise ValueError("sim_time_s must increase strictly between samples")
        if self._start_time_s is None:
            self._start_time_s = sample.sim_time_s
        self._last_time_s = sample.sim_time_s

        if sample.force_peak_n > self.limits.max_contact_force_n:
            self._force_violation = True
        if sample.forbidden_contact:
            self._forbidden_contact = True

        failure_reasons = []
        if self._force_violation:
            failure_reasons.append("contact_force_limit_exceeded")
        if self._forbidden_contact:
            failure_reasons.append("forbidden_contact_detected")
        if failure_reasons:
            return self._finish(False, tuple(failure_reasons))

        within_goal = (
            sample.actual_depth_m >= self.limits.assembled_depth_m
            and sample.lateral_error_m <= self.limits.max_lateral_error_m
            and sample.orientation_error_rad <= self.limits.max_orientation_error_rad
            and sample.grasp_both
        )
        if within_goal:
            if self._candidate_since_s is None:
                self._candidate_since_s = sample.sim_time_s
        else:
            self._candidate_since_s = None

        dwell = self._dwell_progress(sample.sim_time_s)
        if dwell + 1e-12 >= self.limits.dwell_s:
            return self._finish(True, ())

        assert self._start_time_s is not None
        if sample.sim_time_s - self._start_time_s >= self.limits.timeout_s:
            reasons = ["timeout"]
            if sample.actual_depth_m < self.limits.assembled_depth_m:
                reasons.append("insufficient_depth")
            if sample.lateral_error_m > self.limits.max_lateral_error_m:
                reasons.append("lateral_error_too_large")
            if sample.orientation_error_rad > self.limits.max_orientation_error_rad:
                reasons.append("orientation_error_too_large")
            if not sample.grasp_both:
                reasons.append("grasp_not_maintained")
            return self._finish(False, tuple(reasons))

        return EvaluationResult(False, False, dwell, ())

    def _dwell_progress(self, now_s: float) -> float:
        if self._candidate_since_s is None:
            return 0.0
        return max(0.0, now_s - self._candidate_since_s)

    def _finish(self, success: bool, reasons: Tuple[str, ...]) -> EvaluationResult:
        now_s = self._last_time_s
        assert now_s is not None
        result = EvaluationResult(True, success, self._dwell_progress(now_s), reasons)
        self._terminal_result = result
        return result
