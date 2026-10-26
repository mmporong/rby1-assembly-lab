#!/usr/bin/env python3
"""Export an evidence-labelled command-fault video from recorded data.

The exporter is intentionally read-only with respect to simulation evidence. It
uses native rendered frames and holds the latest trace sample whose observation
time is not later than the displayed physics frame. Forces and poses are never
interpolated or synthesized.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np


SCENARIO_TITLES = {
    "right_gripper_open": "오른쪽 그리퍼 OPEN 명령 오류",
    "insertion_lateral_disturbance": "삽입 중 횡방향 위치 오차 명령",
    "hand_collision": "양손 충돌 · 금지 접촉 정지",
}
SCENARIO_DISCLOSURES = {
    "right_gripper_open": "검증용 명령 오류 실험 · 실제 PhysX 접촉 · 학습 정책 없음",
    "insertion_lateral_disturbance": "검증용 명령 오류 실험 · 실제 PhysX 접촉 · 학습 정책 없음",
    "hand_collision": "양손 간 실제 PhysX 접촉 · 검증용 잘못된 수렴 명령 · 학습 정책 없음",
}
EXPECTED_EVENTS = {
    "right_gripper_open": "right_gripper_open_command_started",
    "insertion_lateral_disturbance": "lateral_command_started",
    "hand_collision": "both_hands_converge_command_started",
}
PLAYBACK_SPEED = {
    "right_gripper_open": 0.5,
    "insertion_lateral_disturbance": 2.0,
    "hand_collision": 0.5,
}
OUTPUT_WIDTH = 1600
OUTPUT_HEIGHT = 900
OUTPUT_FPS = 60
PRE_FAULT_SECONDS = 1.5
FINAL_HOLD_SECONDS = 3.0
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_CONFIG_PATH = PROJECT_ROOT / "configs/task.json"


class EvidenceError(ValueError):
    """Raised when the three recorded evidence inputs do not reconcile."""


def _opencv():
    try:
        import cv2
    except ImportError as error:
        raise EvidenceError("OpenCV is required to read and resize recorded video frames") from error
    return cv2


@lru_cache(maxsize=1)
def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as error:
        raise EvidenceError("Pillow is required to draw the evidence panel") from error
    return Image, ImageDraw, ImageFont


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{label} must be finite")
    return result


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(f"invalid trace JSON at line {line_number}: {error}") from error
            environments = row.get("environments")
            if row.get("schema_version") != 1 or not isinstance(environments, list) or len(environments) != 1:
                raise EvidenceError(f"trace line {line_number} is not a schema-v1 single-environment record")
            environment = environments[0]
            if environment.get("env_index") != 0:
                raise EvidenceError(f"trace line {line_number} does not describe environment 0")
            _finite_number(environment.get("timestamp_s"), f"trace line {line_number} timestamp_s")
            rows.append(row)
    if len(rows) < 2:
        raise EvidenceError("trace must contain at least two records")
    timestamps = trace_timestamps(rows)
    if any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise EvidenceError("trace timestamps must be nondecreasing")
    return rows


def trace_timestamps(rows: Sequence[dict[str, Any]]) -> list[float]:
    return [float(row["environments"][0]["timestamp_s"]) for row in rows]


def held_trace_index(timestamps: Sequence[float], frame_simulation_time_s: float) -> int:
    """Return the newest observed sample at or before a rendered frame time."""

    if not timestamps:
        raise EvidenceError("trace timestamps are empty")
    index = bisect_right(timestamps, frame_simulation_time_s + 1e-12) - 1
    if index < 0:
        raise EvidenceError("rendered frame precedes the first trace observation")
    return index


def source_frame_time_s(
    frame_index: int, video_first_physics_step: int, physics_dt_s: float, source_fps: float
) -> float:
    if frame_index < 0 or video_first_physics_step < 1 or physics_dt_s <= 0 or source_fps <= 0:
        raise EvidenceError("invalid video timebase")
    return video_first_physics_step * physics_dt_s + frame_index / source_fps


def source_frame_range(
    *,
    fault_time_s: float,
    terminal_time_s: float,
    video_first_physics_step: int,
    physics_dt_s: float,
    source_fps: float,
    source_frame_count: int,
    max_motion_window_s: float | None = None,
) -> tuple[int, int]:
    """Return inclusive native-frame bounds for fault lead-in through terminal."""

    first_time = video_first_physics_step * physics_dt_s
    start_time = max(first_time, fault_time_s - PRE_FAULT_SECONDS)
    if max_motion_window_s is not None:
        if not math.isfinite(max_motion_window_s) or max_motion_window_s <= 0.0:
            raise EvidenceError("maximum motion window must be finite and positive")
        start_time = max(start_time, terminal_time_s - max_motion_window_s)
    start = max(0, math.ceil((start_time - first_time) * source_fps - 1e-9))
    end = min(
        source_frame_count - 1,
        math.floor((terminal_time_s - first_time) * source_fps + 1e-9),
    )
    if start > end:
        raise EvidenceError("fault/terminal times do not overlap the native video")
    return start, end


def frame_advance(source_fps: float, output_fps: int, playback_speed: float) -> int:
    advance = source_fps * playback_speed / output_fps
    rounded = round(advance)
    if rounded < 1 or not math.isclose(advance, rounded, abs_tol=1e-9):
        raise EvidenceError(
            "source FPS/playback speed must map to whole native frames; frame interpolation and duplication are disabled"
        )
    return rounded


def plug_height_m(environment: dict[str, Any]) -> float:
    try:
        pose = environment["object_poses_world_wxyz"]["plug"]
        value = pose[2]
    except (KeyError, IndexError, TypeError) as error:
        raise EvidenceError("trace is missing the plug world pose") from error
    return _finite_number(value, "plug height")


def scenario_metrics(
    scenario: str,
    rows: Sequence[dict[str, Any]],
    row_index: int,
    clip_start_index: int,
) -> dict[str, float | int | bool]:
    environment = rows[row_index]["environments"][0]
    if scenario == "right_gripper_open":
        forces = environment.get("finger_contact_force_n")
        if not isinstance(forces, list) or len(forces) != 4:
            raise EvidenceError("gripper-open evidence requires four measured finger_contact_force_n values")
        right_min = min(
            _finite_number(forces[2], "right R1 contact force"),
            _finite_number(forces[3], "right R2 contact force"),
        )
        current_height = plug_height_m(environment)
        peak_height = max(
            plug_height_m(row["environments"][0]) for row in rows[clip_start_index : row_index + 1]
        )
        fault = rows[row_index].get("command_fault") or {}
        loss_since_s = _finite_number(
            environment.get("continuous_grasp_loss_since_s", -1.0),
            "continuous grasp loss start",
        )
        observation_time_s = _finite_number(environment.get("timestamp_s"), "trace timestamp")
        return {
            "right_min_finger_contact_n": right_min,
            "plug_height_m": current_height,
            "drop_from_peak_mm": max(0.0, (peak_height - current_height) * 1000.0),
            "continuous_grasp_loss_s": (
                max(0.0, observation_time_s - loss_since_s) if loss_since_s >= 0.0 else 0.0
            ),
            "command_active": bool(fault.get("right_gripper_open_command", False)),
        }
    if scenario == "insertion_lateral_disturbance":
        fault = rows[row_index].get("command_fault") or {}
        return {
            "insertion_contact_load_n": _finite_number(
                environment.get("insertion_contact_load_n"), "insertion contact load"
            ),
            "actual_depth_mm": 1000.0 * _finite_number(environment.get("actual_depth_m"), "actual depth"),
            "actual_lateral_error_mm": 1000.0
            * _finite_number(environment.get("actual_lateral_error_m"), "actual lateral error"),
            "retry_count": int(environment.get("retry_count")),
            "insertion_advance_enabled": bool(environment.get("insertion_advance_enabled", False)),
            "command_error_mm": 1000.0
            * _finite_number(fault.get("lateral_position_error_command_m", 0.0), "lateral command"),
            "command_active": bool(fault.get("active", False)),
        }
    if scenario == "hand_collision":
        if environment.get("self_collision_enabled") is not True:
            raise EvidenceError("hand-collision evidence requires self_collision_enabled=true")
        fault = rows[row_index].get("command_fault") or {}
        return {
            "cross_hand_contact_load_n": _finite_number(
                environment.get("cross_hand_contact_load_n"), "cross-hand contact load"
            ),
            "external_forbidden_contact_load_n": _finite_number(
                environment.get("external_forbidden_contact_load_n"),
                "external forbidden contact load",
            ),
            "hand_wrist_separation_mm": 1000.0 * _finite_number(
                environment.get("hand_wrist_separation_m"), "hand wrist separation"
            ),
            "command_active": bool(fault.get("active", False)),
        }
    raise EvidenceError(f"unsupported scenario: {scenario}")


def validate_report(report: dict[str, Any], rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    experiment = report.get("command_fault_experiment")
    if not isinstance(experiment, dict):
        raise EvidenceError("report has no command_fault_experiment")
    scenario = experiment.get("scenario")
    if scenario not in SCENARIO_TITLES:
        raise EvidenceError(f"unknown command-fault scenario: {scenario!r}")
    if experiment.get("physical_state_force_sensor_or_termination_injection") is not False:
        raise EvidenceError("report does not attest command-only fault injection")
    if experiment.get("triggered") is not True:
        raise EvidenceError("command-fault experiment was not triggered")
    parameters = experiment.get("parameters")
    if not isinstance(parameters, dict):
        raise EvidenceError("command-fault parameters are missing")
    events = experiment.get("events")
    if not isinstance(events, list):
        raise EvidenceError("command-fault events are missing")
    expected_event = EXPECTED_EVENTS[scenario]
    matching_events = [event for event in events if event.get("event") == expected_event]
    if not matching_events:
        raise EvidenceError(f"expected at least one {expected_event} event")
    fault_time_s = _finite_number(
        matching_events[0].get("issued_at_simulation_time_s"), "fault event time"
    )

    outcomes = report.get("terminal_outcomes")
    if not isinstance(outcomes, list) or len(outcomes) != 1:
        raise EvidenceError("report must contain one terminal outcome")
    outcome = outcomes[0]
    if outcome.get("terminal") is not True:
        raise EvidenceError("recorded outcome is not terminal")
    actual_reason = outcome.get("termination_reason")
    if not isinstance(actual_reason, str) or not actual_reason or actual_reason == "none":
        raise EvidenceError("terminal outcome has no actual termination reason")
    terminal_time_s = _finite_number(outcome.get("timestamp_s"), "terminal outcome time")
    if terminal_time_s < fault_time_s:
        raise EvidenceError("terminal outcome precedes the command-fault event")
    if scenario == "hand_collision" and actual_reason != "forbidden_contact":
        raise EvidenceError("hand-collision export requires the actual forbidden_contact outcome")

    terminal_rows = [
        row for row in rows
        if row["environments"][0].get("terminal") is True
    ]
    if len(terminal_rows) != 1:
        raise EvidenceError("trace must contain exactly one terminal record")
    terminal_environment = terminal_rows[0]["environments"][0]
    if terminal_environment.get("termination_reason") != actual_reason:
        raise EvidenceError("trace and report termination reasons differ")
    if not math.isclose(
        float(terminal_environment["timestamp_s"]), terminal_time_s, abs_tol=2e-5
    ):
        raise EvidenceError("trace and report terminal timestamps differ")
    if scenario == "hand_collision":
        threshold_n = _finite_number(
            report.get("forbidden_contact_force_threshold_n"),
            "forbidden contact force threshold",
        )
        terminal_cross_n = _finite_number(
            terminal_environment.get("cross_hand_contact_load_n"),
            "terminal cross-hand contact load",
        )
        terminal_external_n = _finite_number(
            terminal_environment.get("external_forbidden_contact_load_n"),
            "terminal external forbidden contact load",
        )
        if terminal_environment.get("self_collision_enabled") is not True:
            raise EvidenceError("terminal hand-collision evidence has self-collision disabled")
        if report.get("self_collision_enabled") is not True:
            raise EvidenceError("report does not attest self-collision enabled")
        if terminal_cross_n <= threshold_n:
            raise EvidenceError("terminal cross-hand contact load did not exceed the recorded threshold")
        if terminal_external_n > threshold_n:
            raise EvidenceError("external forbidden contact also exceeded the recorded threshold")
        pre_fault_rows = [
            row for row in rows
            if float(row["environments"][0]["timestamp_s"]) < fault_time_s - 1e-12
        ]
        if not pre_fault_rows:
            raise EvidenceError("hand-collision trace has no command-free baseline")
        pre_fault_peak_n = max(
            _finite_number(
                row["environments"][0].get("cross_hand_contact_load_n"),
                "pre-command cross-hand contact load",
            )
            for row in pre_fault_rows
        )
        if pre_fault_peak_n > 1e-6:
            raise EvidenceError("cross-hand contact was already present before the convergence command")

    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise EvidenceError("report provenance is missing")
    physics_dt_s = _finite_number(provenance.get("physics_dt_s"), "physics dt")
    control_dt_s = _finite_number(provenance.get("control_dt_s"), "control dt")
    if not math.isclose(control_dt_s, 1.0 / 30.0, abs_tol=2e-6):
        raise EvidenceError("command-fault trace is not recorded at the expected 30 Hz control rate")
    if report.get("trace_every_control_steps") != 1:
        raise EvidenceError("command-fault trace is not sampled every control step")
    return {
        "scenario": scenario,
        "parameters": parameters,
        "fault_time_s": fault_time_s,
        "terminal_time_s": terminal_time_s,
        "actual_reason": actual_reason,
        "physics_dt_s": physics_dt_s,
    }


def video_probe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate,nb_frames,duration,pix_fmt,codec_name",
        "-of", "json", str(path),
    ]
    try:
        data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise EvidenceError(f"ffprobe failed for {path}") from error
    streams = data.get("streams")
    if not isinstance(streams, list) or len(streams) != 1:
        raise EvidenceError("input video must contain exactly one video stream")
    stream = streams[0]
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    fps = float(numerator) / float(denominator)
    frame_count = int(stream.get("nb_frames") or round(float(stream["duration"]) * fps))
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frame_count": frame_count,
        "duration_s": float(stream["duration"]),
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
    }


@lru_cache(maxsize=32)
def _font(size: int, bold: bool = False):
    _, _, ImageFont = _pillow()
    path = FONT_BOLD if bold else FONT_REGULAR
    if not path.exists():
        raise EvidenceError(f"required Korean font is missing: {path}")
    return ImageFont.truetype(str(path), size)


def _fit_image(frame_bgr: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    cv2 = _opencv()
    left, top, right, bottom = box
    target_w, target_h = right - left, bottom - top
    height, width = frame_bgr.shape[:2]
    scale = min(target_w / width, target_h / height)
    resized = cv2.resize(frame_bgr, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    x = left + (target_w - resized.shape[1]) // 2
    y = top + (target_h - resized.shape[0]) // 2
    return resized, (x, y)


def _line_chart(
    draw,
    box: tuple[int, int, int, int],
    timestamps: Sequence[float],
    series: Sequence[tuple[str, Sequence[float], str]],
    cursor_time_s: float,
    thresholds: Sequence[tuple[float, str, str]] = (),
    *,
    x_start_s: float,
    x_end_s: float,
    y_unit: str,
    y_max: float | None = None,
) -> None:
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=12, fill="#f7f9fc", outline="#d9e0ea", width=2)
    plot = (left + 46, top + 38, right - 14, bottom - 34)
    pl, pt, pr, pb = plot
    plot_left = pl
    values = [value for _, points, _ in series for value in points]
    values.extend(value for value, _, _ in thresholds)
    y_max = max(y_max or 0.0, max(values, default=1.0) * 1.12, 1e-9)
    t_min, t_max = x_start_s, max(x_end_s, x_start_s + 1e-9)

    for ratio in (0.0, 0.5, 1.0):
        y = round(pb - ratio * (pb - pt))
        draw.line((pl, y, pr, y), fill="#dfe5ee", width=1)
        draw.text((left + 6, y - 10), f"{ratio * y_max:.1f}", font=_font(15), fill="#667085")
    draw.text((left + 12, top + 9), y_unit, font=_font(14), fill="#475467")

    def xy(time_s: float, value: float) -> tuple[int, int]:
        x = pl + (time_s - t_min) / (t_max - t_min) * (pr - pl)
        y = pb - max(0.0, min(value, y_max)) / y_max * (pb - pt)
        return round(x), round(y)

    for value, label, color in thresholds:
        y = xy(t_min, value)[1]
        draw.line((pl, y, pr, y), fill=color, width=2)
        draw.text((pl + 4, y - 18), label, font=_font(14), fill=color)
    legend_x = left + 118
    for name, points, color in series:
        coordinates = [xy(time_s, value) for time_s, value in zip(timestamps, points, strict=True)]
        if len(coordinates) > 1:
            draw.line(coordinates, fill=color, width=3, joint="curve")
        draw.line((legend_x, top + 20, legend_x + 22, top + 20), fill=color, width=3)
        draw.text((legend_x + 28, top + 9), name, font=_font(14), fill="#344054")
        legend_x += 165
    pl = plot_left
    cursor_x = xy(min(max(cursor_time_s, t_min), t_max), 0.0)[0]
    draw.line((cursor_x, pt, cursor_x, pb), fill="#101828", width=2)
    draw.text((plot_left, pb + 6), f"{t_min:.1f}", font=_font(13), fill="#667085")
    end_label = f"{t_max:.1f} s"
    end_box = draw.textbbox((0, 0), end_label, font=_font(13))
    draw.text((pr - (end_box[2] - end_box[0]), pb + 6), end_label, font=_font(13), fill="#667085")


def _reason_korean(reason: str) -> str:
    known = {
        "grasp_lost": "파지 소실",
        "hard_force_limit": "하드 접촉 하중 한계 초과",
        "retry_limit": "걸림 복구 재시도 한계 도달",
        "forbidden_contact": "허용되지 않은 접촉",
        "timeout": "시간 제한",
        "episode_timeout": "에피소드 시간 제한",
        "pregrasp_timeout": "파지 준비 시간 제한",
    }
    return known.get(reason.lower(), reason)


def compose_frame(
    frame_bgr: np.ndarray,
    *,
    scenario: str,
    actual_reason: str,
    parameters: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    row_index: int,
    clip_start_trace_index: int,
    fault_time_s: float,
    frame_simulation_time_s: float,
    target_depth_m: float,
    force_threshold_n: float,
    focus_crop: tuple[int, int, int, int] | None,
    stopped: bool,
) -> np.ndarray:
    cv2 = _opencv()
    Image, ImageDraw, _ = _pillow()
    canvas = np.full((OUTPUT_HEIGHT, OUTPUT_WIDTH, 3), 255, dtype=np.uint8)
    raw_box = (24, 112, 1056, 876)
    resized, (x, y) = _fit_image(frame_bgr, raw_box)
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    cv2.rectangle(canvas, (raw_box[0], raw_box[1]), (raw_box[2], raw_box[3]), (208, 215, 224), 2)

    if focus_crop is not None:
        cx, cy, cw, ch = focus_crop
        source_h, source_w = frame_bgr.shape[:2]
        if cx < 0 or cy < 0 or cw <= 0 or ch <= 0 or cx + cw > source_w or cy + ch > source_h:
            raise EvidenceError("focus crop is outside the source video")
        crop = frame_bgr[cy : cy + ch, cx : cx + cw]
        inset = cv2.resize(crop, (360, 270), interpolation=cv2.INTER_CUBIC)
        canvas[574:844, 678:1038] = inset
        cv2.rectangle(canvas, (672, 568), (1044, 850), (255, 255, 255), 5)
        cv2.rectangle(canvas, (672, 568), (1044, 850), (44, 91, 184), 2)

    image = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    draw.text((28, 8), SCENARIO_TITLES[scenario], font=_font(34, True), fill="#101828")
    draw.text(
        (30, 61), SCENARIO_DISCLOSURES[scenario],
        font=_font(18), fill="#475467",
    )

    environment = rows[row_index]["environments"][0]
    metrics = scenario_metrics(scenario, rows, row_index, clip_start_trace_index)
    relative_t = frame_simulation_time_s - fault_time_s
    stage = "오류 전 정상 동작" if relative_t < 0 else "오류 명령 적용"
    if relative_t >= 0 and not bool(metrics["command_active"]):
        stage = "오류 이후 복구 동작"
    if scenario == "insertion_lateral_disturbance" and relative_t >= 0:
        if environment["phase_name"] == "retract":
            stage = "후퇴·재정렬"
        elif environment["phase_name"] == "align":
            stage = "재정렬"
        elif environment["phase_name"] == "insert" and not metrics["insertion_advance_enabled"]:
            stage = "전진 보류"
    if scenario == "hand_collision" and relative_t >= 0:
        stage = "양손 수렴 명령 적용"
    if stopped and scenario == "insertion_lateral_disturbance" and actual_reason == "episode_timeout":
        stage = "삽입 진행 보류 → 재시도 → 시간 초과"
    elif stopped and scenario == "hand_collision":
        stage = "양손 충돌 · 금지 접촉 정지"
    elif stopped:
        stage = "안전 정지 · " + _reason_korean(actual_reason)
    speed = PLAYBACK_SPEED[scenario]
    draw.text((1086, 112), stage, font=_font(25, True), fill="#b42318" if relative_t >= 0 else "#175cd3")
    draw.text((1086, 151), f"재생 {speed:g}x · 영상 60 fps · 관측 30 Hz", font=_font(17), fill="#475467")
    draw.text((1086, 181), f"실측 시각  {frame_simulation_time_s:6.2f} s", font=_font(17), fill="#344054")
    draw.text((1086, 209), f"제어 단계  {environment['phase_name']} · retry {environment['retry_count']}", font=_font(17), fill="#344054")

    times = trace_timestamps(rows[clip_start_trace_index : row_index + 1])
    clip_start_s = trace_timestamps(rows)[clip_start_trace_index]
    terminal_time_s = next(
        float(row["environments"][0]["timestamp_s"])
        for row in rows if row["environments"][0].get("terminal") is True
    )
    if scenario == "right_gripper_open":
        draw.text((1086, 251), "원인", font=_font(18, True), fill="#101828")
        draw.text((1086, 280), "플러그를 잡은 오른손(화면 왼쪽)의", font=_font(16), fill="#475467")
        draw.text((1086, 306), "손가락 목표를 OPEN으로 변경", font=_font(16), fill="#475467")
        draw.text((1086, 344), "실측", font=_font(18, True), fill="#101828")
        draw.text((1086, 374), f"오른손 최소 손가락 접촉력  {metrics['right_min_finger_contact_n']:.2f} N", font=_font(17), fill="#344054")
        draw.text(
            (1086, 402),
            f"낙하 {metrics['drop_from_peak_mm']:.1f} mm · 연속 파지 소실 {metrics['continuous_grasp_loss_s']:.2f} s",
            font=_font(16), fill="#344054",
        )
        visible = rows[clip_start_trace_index : row_index + 1]
        contact = [
            min(float(row["environments"][0]["finger_contact_force_n"][2]),
                float(row["environments"][0]["finger_contact_force_n"][3]))
            for row in visible
        ]
        all_contact = [
            min(float(row["environments"][0]["finger_contact_force_n"][2]),
                float(row["environments"][0]["finger_contact_force_n"][3]))
            for row in rows[clip_start_trace_index:]
        ]
        _line_chart(
            draw, (1080, 440, 1576, 704), times,
            (("오른손 최소 접촉력", contact, "#175cd3"),),
            frame_simulation_time_s, ((1.0, "파지 판정 1 N", "#039855"),),
            x_start_s=clip_start_s, x_end_s=terminal_time_s, y_unit="힘 (N)",
            y_max=max(all_contact, default=1.0) * 1.12,
        )
    elif scenario == "insertion_lateral_disturbance":
        max_command_mm = 1000.0 * _finite_number(
            parameters.get("max_position_error_command_m"), "max lateral command"
        )
        draw.text((1086, 251), "원인", font=_font(18, True), fill="#101828")
        draw.text((1086, 280), "삽입 중 소켓 +X 방향 위치 오차 명령", font=_font(16), fill="#475467")
        draw.text((1086, 306), f"최대 {max_command_mm:.1f} mm (명령값, 실제 이동량 아님)", font=_font(16), fill="#475467")
        draw.text((1086, 344), "실측", font=_font(18, True), fill="#101828")
        draw.text((1086, 374), f"삽입 접촉 하중  {metrics['insertion_contact_load_n']:.2f} N", font=_font(17), fill="#344054")
        draw.text((1086, 402), f"깊이 {metrics['actual_depth_mm']:.2f} / 목표 {1000.0 * target_depth_m:.2f} mm", font=_font(17), fill="#344054")
        advance_ko = "예" if metrics["insertion_advance_enabled"] else "아니오"
        draw.text((1086, 430), f"횡오차 {metrics['actual_lateral_error_mm']:.2f} mm · 전진 허용 {advance_ko}", font=_font(17), fill="#344054")
        visible = rows[clip_start_trace_index : row_index + 1]
        load = [float(row["environments"][0]["insertion_contact_load_n"]) for row in visible]
        all_load = [
            float(row["environments"][0]["insertion_contact_load_n"])
            for row in rows[clip_start_trace_index:]
        ]
        _line_chart(
            draw, (1080, 466, 1576, 704), times,
            (("삽입 접촉 하중", load, "#d92d20"),),
            frame_simulation_time_s,
            ((20.0, "걸림 판정 20 N", "#f79009"), (40.0, "하드 한계 40 N", "#b42318")),
            x_start_s=clip_start_s, x_end_s=terminal_time_s, y_unit="힘 (N)",
            y_max=max(44.8, max(all_load, default=0.0) * 1.12),
        )
    else:
        target = parameters.get("target_wrist_position_world_m")
        if not isinstance(target, list) or len(target) != 3:
            raise EvidenceError("hand-collision parameters require target_wrist_position_world_m xyz")
        max_error_mm = 1000.0 * _finite_number(
            parameters.get("max_cartesian_error_m"), "maximum Cartesian error"
        )
        draw.text((1086, 251), "원인", font=_font(18, True), fill="#101828")
        draw.text((1086, 280), "좌우 손목을 같은 공중 목표점으로 수렴", font=_font(16), fill="#475467")
        draw.text((1086, 306), f"축별 Cartesian 오차 상한 {max_error_mm:.1f} mm", font=_font(16), fill="#475467")
        draw.text((1086, 344), "실측", font=_font(18, True), fill="#101828")
        draw.text((1086, 374), f"양손 접촉 하중  {metrics['cross_hand_contact_load_n']:.2f} N", font=_font(17), fill="#344054")
        draw.text(
            (1086, 402),
            f"손목 간 거리 {metrics['hand_wrist_separation_mm']:.1f} mm · 외부 접촉 {metrics['external_forbidden_contact_load_n']:.2f} N",
            font=_font(15), fill="#344054",
        )
        visible = rows[clip_start_trace_index : row_index + 1]
        load = [
            float(row["environments"][0]["cross_hand_contact_load_n"])
            for row in visible
        ]
        all_load = [
            float(row["environments"][0]["cross_hand_contact_load_n"])
            for row in rows[clip_start_trace_index:]
        ]
        _line_chart(
            draw, (1080, 440, 1576, 704), times,
            (("양손 접촉 하중", load, "#d92d20"),), frame_simulation_time_s,
            ((force_threshold_n, f"금지 접촉 {force_threshold_n:g} N", "#b42318"),),
            x_start_s=clip_start_s, x_end_s=terminal_time_s, y_unit="힘 (N)",
            y_max=max(force_threshold_n * 1.25, max(all_load, default=0.0) * 1.12),
        )

    draw.rounded_rectangle((1080, 730, 1576, 856), radius=12, fill="#fff4ed", outline="#f7b27a", width=2)
    draw.text((1100, 748), "정지 사유", font=_font(18, True), fill="#9c2a10")
    reason_text = _reason_korean(actual_reason) if stopped else "실행 중"
    draw.text((1100, 784), reason_text, font=_font(25, True), fill="#b42318" if stopped else "#475467")
    if stopped:
        draw.text((1100, 824), f"보고서 실제 판정: {actual_reason}", font=_font(15), fill="#667085")
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def ffmpeg_writer(output: Path) -> subprocess.Popen:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUTPUT_WIDTH}x{OUTPUT_HEIGHT}",
        "-r", str(OUTPUT_FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium",
        "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def export(args: argparse.Namespace) -> dict[str, Any]:
    cv2 = _opencv()
    for path in (args.report, args.trace, args.video):
        if not path.is_file():
            raise EvidenceError(f"input file does not exist: {path}")
    if args.output.resolve() in {args.report.resolve(), args.trace.resolve(), args.video.resolve()}:
        raise EvidenceError("output must not overwrite an evidence input")

    report = json.loads(args.report.read_text(encoding="utf-8"))
    if not TASK_CONFIG_PATH.is_file():
        raise EvidenceError(f"task design config is missing: {TASK_CONFIG_PATH}")
    task_config = json.loads(TASK_CONFIG_PATH.read_text(encoding="utf-8"))
    task_config_sha = sha256(TASK_CONFIG_PATH)
    if report.get("provenance", {}).get("task_config_sha256") != task_config_sha:
        raise EvidenceError("report task config hash does not match the current design config")
    target_depth_m = _finite_number(
        task_config.get("evaluation", {}).get("assembled_depth_m"), "assembled target depth"
    )
    if not math.isclose(target_depth_m, 0.03, abs_tol=1e-12):
        raise EvidenceError("presentation layout currently expects the recorded 30 mm assembly target")
    rows = load_trace(args.trace)
    validated = validate_report(report, rows)
    probe = video_probe(args.video)
    source_fps = probe["fps"]
    scenario = validated["scenario"]
    expected_source_fps = 120 if scenario in ("right_gripper_open", "hand_collision") else 60
    if not math.isclose(source_fps, expected_source_fps, abs_tol=0.01):
        raise EvidenceError(f"{scenario} requires a native {expected_source_fps} FPS source")
    for key, actual in (("video_fps", source_fps), ("video_frame_count", probe["frame_count"])):
        recorded = report.get(key)
        if recorded is None or not math.isclose(float(recorded), float(actual), abs_tol=0.01):
            raise EvidenceError(f"report {key} does not match the source video")
    first_step = report.get("video_first_physics_step")
    if not isinstance(first_step, int) or first_step < 1:
        raise EvidenceError("report has no valid video_first_physics_step")
    if report.get("video_capture_source") != "native_physics_substeps_without_interpolation":
        raise EvidenceError("source video is not attested as native physics-substep capture")
    force_threshold_n = _finite_number(
        report.get("forbidden_contact_force_threshold_n"), "forbidden contact force threshold"
    )

    speed = PLAYBACK_SPEED[scenario]
    advance = frame_advance(source_fps, OUTPUT_FPS, speed)
    max_motion_window_s = 3.0 if scenario == "hand_collision" else None
    start_frame, end_frame = source_frame_range(
        fault_time_s=validated["fault_time_s"], terminal_time_s=validated["terminal_time_s"],
        video_first_physics_step=first_step, physics_dt_s=validated["physics_dt_s"],
        source_fps=source_fps, source_frame_count=probe["frame_count"],
        max_motion_window_s=max_motion_window_s,
    )
    timestamps = trace_timestamps(rows)
    clip_start_time = source_frame_time_s(start_frame, first_step, validated["physics_dt_s"], source_fps)
    clip_start_trace_index = held_trace_index(timestamps, clip_start_time)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise EvidenceError(f"OpenCV could not open source video: {args.video}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = ffmpeg_writer(args.output)
    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin was not created")
    output_frames = 0
    last_composed: np.ndarray | None = None
    last_source_frame = None
    try:
        source_index = start_frame
        while source_index <= end_frame:
            ok, frame = capture.read()
            if not ok:
                raise EvidenceError(f"source video ended at frame {source_index}")
            frame_time = source_frame_time_s(source_index, first_step, validated["physics_dt_s"], source_fps)
            row_index = held_trace_index(timestamps, frame_time)
            stopped = rows[row_index]["environments"][0].get("terminal") is True
            composed = compose_frame(
                frame, scenario=scenario, actual_reason=validated["actual_reason"],
                parameters=validated["parameters"], rows=rows, row_index=row_index,
                clip_start_trace_index=clip_start_trace_index,
                fault_time_s=validated["fault_time_s"], frame_simulation_time_s=frame_time,
                target_depth_m=target_depth_m, force_threshold_n=force_threshold_n,
                focus_crop=args.focus_crop, stopped=stopped,
            )
            writer.stdin.write(composed.tobytes())
            output_frames += 1
            last_composed = composed
            last_source_frame = source_index
            for _ in range(advance - 1):
                if not capture.grab():
                    raise EvidenceError("source video ended while advancing native frames")
            source_index += advance

        if last_composed is None:
            raise EvidenceError("no source frames were exported")
        terminal_row_index = held_trace_index(timestamps, validated["terminal_time_s"])
        terminal_environment = rows[terminal_row_index]["environments"][0]
        if terminal_environment.get("terminal") is not True:
            raise EvidenceError("last displayed evidence does not reach the terminal observation")
        stopped_frame = compose_frame(
            frame, scenario=scenario, actual_reason=validated["actual_reason"],
            parameters=validated["parameters"], rows=rows, row_index=terminal_row_index,
            clip_start_trace_index=clip_start_trace_index,
            fault_time_s=validated["fault_time_s"],
            frame_simulation_time_s=validated["terminal_time_s"], target_depth_m=target_depth_m,
            force_threshold_n=force_threshold_n, focus_crop=args.focus_crop,
            stopped=True,
        )
        for _ in range(round(FINAL_HOLD_SECONDS * OUTPUT_FPS)):
            writer.stdin.write(stopped_frame.tobytes())
            output_frames += 1
    finally:
        capture.release()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")

    output_probe = video_probe(args.output)
    if output_probe["frame_count"] != output_frames:
        raise EvidenceError("encoded output frame count does not match streamed frame count")
    if output_probe["codec_name"] != "h264" or output_probe["pix_fmt"] != "yuv420p":
        raise EvidenceError("encoded output is not H.264 yuv420p")
    if not math.isclose(output_probe["fps"], OUTPUT_FPS, abs_tol=0.01):
        raise EvidenceError("encoded output is not 60 FPS")

    metadata = {
        "schema_version": 1,
        "scope": "read-only presentation export of recorded command-fault evidence",
        "scenario": scenario,
        "title_ko": SCENARIO_TITLES[scenario],
        "disclosure_ko": SCENARIO_DISCLOSURES[scenario],
        "source": {
            "video": {"path": str(args.video), "sha256": sha256(args.video), **probe},
            "report": {"path": str(args.report), "sha256": sha256(args.report)},
            "trace": {"path": str(args.trace), "sha256": sha256(args.trace)},
        },
        "exporter": {
            "path": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "clip": {
            "source_start_frame": start_frame,
            "source_last_frame": last_source_frame,
            "source_start_simulation_time_s": clip_start_time,
            "fault_simulation_time_s": validated["fault_time_s"],
            "terminal_simulation_time_s": validated["terminal_time_s"],
            "max_motion_window_s": max_motion_window_s,
            "clip_selection_policy": (
                "latest_of_fault_lead_in_or_terminal_minus_3s"
                if max_motion_window_s is not None
                else "fault_event_minus_1.5s_through_terminal"
            ),
            "playback_speed": speed,
            "output_fps": OUTPUT_FPS,
            "output_frame_count": output_frames,
            "final_hold_s": FINAL_HOLD_SECONDS,
            "native_frame_policy": (
                f"source frames only; every {advance} native frame selected; no interpolation; "
                "terminal source frame held for summary"
            ),
            "trace_alignment_policy": "latest 30 Hz observation at or before each native source-frame simulation time",
        },
        "fault_parameters": validated["parameters"],
        "forbidden_contact_force_threshold_n": force_threshold_n,
        "design_reference": {
            "task_config_path": str(TASK_CONFIG_PATH.relative_to(PROJECT_ROOT)),
            "task_config_sha256": task_config_sha,
            "assembled_target_depth_m": target_depth_m,
        },
        "actual_termination_reason": validated["actual_reason"],
        "output": {"path": str(args.output), "sha256": sha256(args.output), **output_probe},
        "reproduction_command": shlex.join([sys.executable, *sys.argv]),
    }
    metadata_path = args.metadata or args.output.with_suffix(".json")
    if metadata_path.resolve() in {args.report.resolve(), args.trace.resolve(), args.video.resolve()}:
        raise EvidenceError("metadata output must not overwrite an evidence input")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--focus-crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"))
    args = parser.parse_args(argv)
    if args.focus_crop is not None:
        args.focus_crop = tuple(args.focus_crop)
    return args


def main(argv: Iterable[str] | None = None) -> int:
    try:
        metadata = export(parse_args(argv))
    except (EvidenceError, OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"failure-media export failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
