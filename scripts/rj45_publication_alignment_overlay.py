#!/usr/bin/env python3
"""Build a trace-backed ASS overlay for the RJ45 comparison video."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping


FPS = 60
MODES = ("multi64", "zero")
EXPECTED_FRAMES = {"multi64": 760, "zero": 1454}
PANEL_X = {"multi64": 24, "zero": 968}
LATERAL_LIMIT_UM = 50.0
DISPLAY_MAX_UM = 100.0
DEPTH_TARGET_MM = 12.0
DWELL_TARGET_S = 0.4
PLOT_START_OFFSET_X = 80
PLOT_WIDTH = 700
PLOT_Y = 916
TERMINAL_CARD_S = 2.0


class AlignmentOverlayError(RuntimeError):
    """Raised when source evidence cannot support the publication overlay."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AlignmentOverlayError(f"{label} is not readable JSON: {path}") from error
    if not isinstance(value, dict):
        raise AlignmentOverlayError(f"{label} root must be an object")
    return value


def _load_frozen_helper() -> tuple[ModuleType, Path, str]:
    helper_path = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/rj45-multibias-20261031/build_media.py"
    )
    data = helper_path.read_bytes()
    spec = importlib.util.spec_from_file_location("_rj45_frozen_build_media", helper_path)
    if spec is None or spec.loader is None:
        raise AlignmentOverlayError(f"cannot import frozen helper: {helper_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, helper_path, _sha256(data)


def _validated_trace(
    render_root: Path,
    recovery: Mapping[str, Any],
    helper: ModuleType,
    mode: str,
) -> tuple[list[dict[str, Any]], list[float | None], dict[str, Any]]:
    relative_path = f"{mode}/replay.mp4.frames.jsonl"
    trace_path = render_root / relative_path
    data = trace_path.read_bytes()
    observed_sha = _sha256(data)

    recovery_record = recovery.get(relative_path)
    if not isinstance(recovery_record, dict):
        raise AlignmentOverlayError(f"recovery manifest has no {relative_path}")
    if (
        recovery_record.get("sha256") != observed_sha
        or recovery_record.get("bytes") != len(data)
    ):
        raise AlignmentOverlayError(f"raw trace differs from recovery manifest: {mode}")

    evidence = _load_json(render_root / mode / "replay.mp4.evidence.json", f"{mode} evidence")
    try:
        evidence_sha = evidence["artifacts"]["frames"]["sha256"]
    except (KeyError, TypeError) as error:
        raise AlignmentOverlayError(f"{mode} evidence has no frame artifact hash") from error
    if evidence_sha != observed_sha:
        raise AlignmentOverlayError(f"raw trace differs from source evidence: {mode}")

    try:
        rows = helper._frame_rows(data, mode)
        dwell = helper._causal_dwell(rows, mode)
    except Exception as error:
        raise AlignmentOverlayError(f"frozen trace validation failed: {mode}: {error}") from error
    if any(row.get("frame_state_is_pre_autoreset_physics") is not True for row in rows):
        raise AlignmentOverlayError(f"{mode} trace contains a post-autoreset frame")
    if any(type(row.get("policy_input_valid")) is not bool for row in rows):
        raise AlignmentOverlayError(f"{mode} policy_input_valid is not boolean")
    return rows, dwell, {
        "path": str(trace_path),
        "bytes": len(data),
        "sha256": observed_sha,
        "source_evidence_sha256": evidence_sha,
        "recovery_manifest_sha256": recovery_record["sha256"],
        "frames": len(rows),
    }


def _ass_time(seconds: float) -> str:
    if seconds < 0 or not math.isfinite(seconds):
        raise AlignmentOverlayError("ASS time must be finite and non-negative")
    # ASS timestamps have two fractional digits.  A third digit is not
    # milliseconds: libass interprets e.g. .017 as 17 centiseconds.
    centiseconds = math.floor(seconds * 100.0 + 1.0e-9)
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _dialogue(start: float, end: float, style: str, x: int, y: int, text: str) -> str:
    safe_text = text.replace("\\", r"\\").replace("{", r"\{").replace("}", r"\}")
    return (
        f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},{style},,0,0,0,,"
        f"{{\\pos({x},{y})}}{safe_text}"
    )


def _frame_events(
    mode: str,
    rows: list[dict[str, Any]],
    causal_dwell: list[float | None],
    deadline_s: float,
) -> tuple[list[str], dict[str, Any]]:
    panel_x = PANEL_X[mode]
    events: list[str] = []
    invalid_frames = 0
    clamped_frames = 0

    for index, (row, dwell) in enumerate(zip(rows, causal_dwell)):
        start = index / FPS
        end = (index + 1) / FPS
        valid = row["policy_input_valid"]
        if not valid:
            invalid_frames += 1
            events.append(_dialogue(start, end, "Metric", panel_x + 20, 858, "초기 자세 준비"))
        else:
            lateral_um = float(row["actual_lateral_error_m"]) * 1_000_000.0
            depth_mm = float(row["actual_depth_m"]) * 1000.0
            criterion = "기준 안" if lateral_um <= LATERAL_LIMIT_UM else "기준 밖"
            events.append(
                _dialogue(
                    start,
                    end,
                    "Metric",
                    panel_x + 20,
                    858,
                    f"횡오차 {lateral_um:.1f} µm · {criterion}",
                )
            )
            dwell_text = "--" if dwell is None else f"{dwell:.2f}"
            events.append(
                _dialogue(
                    start,
                    end,
                    "Detail",
                    panel_x + 20,
                    949,
                    f"깊이 {depth_mm:.2f} / {DEPTH_TARGET_MM:.0f} mm · "
                    f"안정 유지 {dwell_text} / {DWELL_TARGET_S:.2f} s",
                )
            )
            marker_x = panel_x + PLOT_START_OFFSET_X + round(
                min(lateral_um, DISPLAY_MAX_UM) / DISPLAY_MAX_UM * PLOT_WIDTH
            )
            marker_text = "●"
            if lateral_um > DISPLAY_MAX_UM:
                marker_text = "● 100 이상"
                clamped_frames += 1
            marker_style = "MarkerIn" if criterion == "기준 안" else "MarkerOut"
            events.append(
                _dialogue(start, end, marker_style, marker_x, PLOT_Y, marker_text)
            )

        events.append(
            _dialogue(
                start,
                end,
                "Elapsed",
                panel_x + 740,
                858,
                f"실시간 {float(row['physics_time_s']):.2f} s · 1×",
            )
        )

    # Keep the last measured state visible while the builder shows the terminal card.
    terminal_start = len(rows) / FPS
    if terminal_start < deadline_s:
        terminal_events, _ = _frame_events(
            mode,
            [rows[-1]],
            [causal_dwell[-1]],
            1.0 / FPS,
        )
        for event in terminal_events:
            prefix, fields = event.split(": ", 1)
            parts = fields.split(",", 3)
            parts[1] = _ass_time(terminal_start)
            parts[2] = _ass_time(deadline_s)
            events.append(prefix + ": " + ",".join(parts))

    return events, {
        "invalid_policy_input_frames": invalid_frames,
        "display_clamped_frames": clamped_frames,
        "terminal_physics_time_s": float(rows[-1]["physics_time_s"]),
    }


def _ass_header() -> str:
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Metric,Noto Sans CJK KR,27,&H00322218,&H000000FF,&H00000000,&H00000000,"
            "-1,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
            "Style: Detail,Noto Sans CJK KR,24,&H00736052,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,0,0,7,0,0,0,1",
            "Style: Elapsed,Noto Sans CJK KR,23,&H00736052,&H000000FF,&H00000000,&H00000000,"
            "0,0,0,0,100,100,0,0,1,0,0,9,0,0,0,1",
            "Style: MarkerIn,Noto Sans CJK KR,22,&H005A7514,&H000000FF,&H00FFFFFF,&H00000000,"
            "-1,0,0,0,100,100,0,0,1,0.6,0,5,0,0,0,1",
            "Style: MarkerOut,Noto Sans CJK KR,22,&H0038449C,&H000000FF,&H00FFFFFF,&H00000000,"
            "-1,0,0,0,100,100,0,0,1,0.6,0,5,0,0,0,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
    )


def build_alignment_overlay(
    render_root: Path,
    records: dict[str, dict[str, Any]],
    output_path: Path,
) -> dict[str, Any]:
    """Validate recovered traces and write the publication ASS overlay."""
    render_root = Path(render_root)
    output_path = Path(output_path)
    recovery_path = render_root / "recovery-files.json"
    recovery = _load_json(recovery_path, "recovery manifest")
    helper, helper_path, helper_sha = _load_frozen_helper()

    loaded: dict[str, tuple[list[dict[str, Any]], list[float | None]]] = {}
    traces: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        record = records.get(mode)
        if not isinstance(record, dict) or record.get("frames") != EXPECTED_FRAMES[mode]:
            raise AlignmentOverlayError(f"{mode} record frame count changed")
        rows, dwell, trace = _validated_trace(render_root, recovery, helper, mode)
        if len(rows) != record["frames"]:
            raise AlignmentOverlayError(f"{mode} record does not match trace")
        loaded[mode] = (rows, dwell)
        traces[mode] = trace

    minimum_deadline_s = (
        max(len(loaded[mode][0]) for mode in MODES) / FPS + TERMINAL_CARD_S
    )
    deadline_s = math.ceil(minimum_deadline_s * 100.0 - 1.0e-9) / 100.0
    events: list[str] = []
    display: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        mode_events, mode_display = _frame_events(
            mode, loaded[mode][0], loaded[mode][1], deadline_s
        )
        events.extend(mode_events)
        display[mode] = mode_display

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _ass_header() + "\n" + "\n".join(events) + "\n"
    output_path.write_text(payload, encoding="utf-8")
    return {
        "schema_name": "rj45_publication_alignment_overlay_v1",
        "output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path.read_bytes()),
        },
        "timing": {
            "fps": FPS,
            "event_interval": "frame_index/FPS .. (frame_index+1)/FPS",
            "ass_timestamp_quantization": "floor to 1/100 second",
            "displayed_elapsed_source": "physics_time_s",
            "playback_rate": "1x",
            "minimum_deadline_s": minimum_deadline_s,
            "deadline_s": deadline_s,
        },
        "display_contract": {
            "lateral_pass_limit_um": LATERAL_LIMIT_UM,
            "lateral_plot_min_um": 0.0,
            "lateral_plot_max_um": DISPLAY_MAX_UM,
            "values_above_plot_max": "numeric value preserved; marker clamped and labeled 100 이상",
            "overall_success_not_declared": True,
            "invalid_policy_input": "초기 자세 준비; numeric values and marker hidden",
        },
        "frozen_helper": {
            "path": str(helper_path),
            "sha256": helper_sha,
            "functions": ["_frame_rows", "_causal_dwell"],
        },
        "traces": traces,
        "display": display,
        "event_count": len(events),
    }

