"""시뮬레이션 독립 평가용 SI 단위 데이터 계약."""

from __future__ import annotations

from dataclasses import dataclass
import math


def _validate_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")



@dataclass(frozen=True)
class EvaluationSample:
    """제어 관측과 분리된 평가기 전용 시뮬레이터 기록."""

    actual_depth_m: float
    lateral_error_m: float
    orientation_error_rad: float
    force_peak_n: float
    grasp_both: bool
    forbidden_contact: bool
    sim_time_s: float

    def __post_init__(self) -> None:
        for name in (
            "actual_depth_m",
            "lateral_error_m",
            "orientation_error_rad",
            "force_peak_n",
            "sim_time_s",
        ):
            _validate_finite(name, getattr(self, name))
        if self.lateral_error_m < 0.0:
            raise ValueError("lateral_error_m must be non-negative")
        if self.orientation_error_rad < 0.0:
            raise ValueError("orientation_error_rad must be non-negative")
        if self.force_peak_n < 0.0:
            raise ValueError("force_peak_n must be non-negative")
        if self.sim_time_s < 0.0:
            raise ValueError("sim_time_s must be non-negative")



@dataclass(frozen=True)
class TaskLimits:
    """SI 단위의 시뮬레이션 설계 기본값.

    기본값은 정책과 평가 계약을 시작하기 위한 값이며 실물 RB-Y1이나
    부품의 교정값, 허용 하중 또는 안전 한계가 아니다.
    """

    assembled_depth_m: float = 0.030
    max_lateral_error_m: float = 0.002
    max_orientation_error_rad: float = math.radians(3.0)
    max_contact_force_n: float = 40.0
    dwell_s: float = 0.40
    timeout_s: float = 12.0
    max_retries: int = 2
    residual_translation_bound_m: float = 0.003
    residual_rotation_bound_rad: float = math.radians(2.0)

    def __post_init__(self) -> None:
        positive = (
            "assembled_depth_m",
            "max_lateral_error_m",
            "max_orientation_error_rad",
            "max_contact_force_n",
            "dwell_s",
            "timeout_s",
            "residual_translation_bound_m",
            "residual_rotation_bound_rad",
        )
        for name in positive:
            value = getattr(self, name)
            _validate_finite(name, value)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.max_retries, bool) or not isinstance(self.max_retries, int):
            raise ValueError("max_retries must be an integer")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
