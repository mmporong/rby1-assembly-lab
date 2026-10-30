#!/usr/bin/env python3
"""Build synchronized policy-diagnostic media from immutable Isaac recordings.

Every output frame consumes exactly one native 60 FPS source frame.  Geometry is
the rendered physics state; stability and dwell values are the recorded control
step entry/end snapshot and are labelled separately in the presentation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable

import numpy as np


OUTPUT_SIZE = (1600, 900)
FPS = 60.0
LATERAL_LIMIT_UM = 50.0
LINEAR_LIMIT_MM_S = 5.0
ANGULAR_LIMIT_RAD_S = 0.05
DWELL_REQUIRED_S = 0.4
EXPECTED_BC_ACTOR = "f0fd65b56691748dce5bee378152524bf84f3f9daed2f1f36dd43e0e189233b5"
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")


class EvidenceError(ValueError):
    """Raised when source evidence does not satisfy the export contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{name} must be finite")
    return result


def probe_video(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,pix_fmt,avg_frame_rate,nb_read_frames",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = stream["avg_frame_rate"].split("/")
    return {
        "width": int(stream["width"]), "height": int(stream["height"]),
        "pix_fmt": stream["pix_fmt"], "fps": float(numerator) / float(denominator),
        "frame_count": int(stream["nb_read_frames"]),
    }


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise EvidenceError(f"invalid frame JSON line {line_number}: {error}") from error
            if row.get("frame_index") != len(rows):
                raise EvidenceError(f"non-contiguous frame index at line {line_number}")
            rows.append(row)
    if not rows:
        raise EvidenceError("frame JSONL is empty")
    times = [_finite(row.get("physics_time_s"), "physics_time_s") for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise EvidenceError("physics timestamps must increase")
    return rows


def metrics(row: dict[str, Any]) -> dict[str, float | bool | int]:
    try:
        snapshot = row["control_step_snapshot"]
        stability = snapshot["rj45_evaluation_stability"]
        criteria = snapshot["terminal_evaluation_criteria"]
    except (KeyError, TypeError) as error:
        raise EvidenceError("frame row is missing evaluation telemetry") from error
    return {
        "frame_index": int(row["frame_index"]),
        "physics_time_s": _finite(row["physics_time_s"], "physics_time_s"),
        "control_time_s": _finite(row["control_time_s"], "control_time_s"),
        "control_step": int(row["control_step"]),
        "depth_mm": 1000.0 * _finite(row["actual_depth_m"], "actual_depth_m"),
        "lateral_error_um": 1e6 * _finite(row["actual_lateral_error_m"], "actual_lateral_error_m"),
        "orientation_error_deg": math.degrees(_finite(row["actual_orientation_error_rad"], "orientation")),
        "relative_linear_speed_mm_s": 1000.0 * _finite(
            stability["relative_linear_speed_m_s"], "relative linear speed"
        ),
        "relative_angular_speed_rad_s": _finite(
            stability["relative_angular_speed_rad_s"], "relative angular speed"
        ),
        "dwell_s": _finite(snapshot["independent_eval_dwell_s"], "dwell"),
        "contact_proxy_n": _finite(
            row["control_step_entry_contact_force_proxy_n"], "contact proxy"
        ),
        "physics_all_samples_stable": bool(stability["physics_all_samples_stable"]),
        "lateral_ok": bool(criteria["lateral"]),
        "dwell_ok": bool(criteria["dwell"]),
        "terminal": bool(row["terminal_observed_after_control_step"]),
        "success": bool(row["success_observed_after_control_step"]),
    }


def verify_artifact_hashes(paths: dict[str, Path], evidence: dict[str, Any]) -> None:
    artifacts = evidence.get("artifacts", {})
    for label in ("video", "frames", "metadata"):
        expected = artifacts.get(label, {}).get("sha256")
        if expected != sha256(paths[label]):
            raise EvidenceError(f"{label} hash does not match evidence")


def validate_lineage(stem: str, metadata: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lineage = metadata.get("policy_lineage")
    if not isinstance(lineage, dict):
        raise EvidenceError("missing policy lineage")
    lineage_sha = metadata.get("policy_lineage_sha256")
    if not isinstance(lineage_sha, str) or any(row.get("policy_lineage_sha256") != lineage_sha for row in rows):
        raise EvidenceError("frame lineage does not match metadata")
    mode = lineage.get("mode")
    if stem.startswith("v5-bc-"):
        actor = lineage.get("bc_actor")
        if mode != "bc" or not isinstance(actor, dict) or actor.get("sha256") != EXPECTED_BC_ACTOR:
            raise EvidenceError("BC media does not use the frozen expected actor")
    elif stem == "v3-biased-zero-env54":
        if mode != "biased_zero":
            raise EvidenceError("reference media must use biased-zero mode")
    else:
        raise EvidenceError(f"unsupported media stem: {stem}")


def validate_inputs(stem: str, paths: dict[str, Path]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for label, path in paths.items():
        if not path.is_file():
            raise EvidenceError(f"missing {label}: {path}")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    evidence = json.loads(paths["evidence"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    rows = load_rows(paths["frames"])
    probe = probe_video(paths["video"])
    if metadata.get("completed") is not True or metadata.get("rollout_completed") is not True:
        raise EvidenceError("recording is not complete")
    expected_count = int(metadata.get("captured_frames", -1))
    if expected_count != len(rows) or expected_count != probe["frame_count"]:
        raise EvidenceError("video, metadata, and JSONL frame counts disagree")
    if not math.isclose(probe["fps"], FPS, abs_tol=0.01):
        raise EvidenceError("source must be native 60 FPS")
    if rows[-1].get("terminal_observed_after_control_step") is not True:
        raise EvidenceError("last native frame is not terminal")
    validate_lineage(stem, metadata, rows)
    lineage = metadata["policy_lineage"]
    verify_artifact_hashes(paths, evidence)
    terminal = metadata.get("selected_terminal", {})
    final = metrics(rows[-1])
    if bool(terminal.get("success")) != bool(final["success"]):
        raise EvidenceError("terminal success disagrees with last frame")
    if stem.startswith("v5-bc-"):
        if final["success"] or terminal.get("termination_reason") != "episode_timeout":
            raise EvidenceError("selected BC media must be a real timeout failure")
    elif stem == "v3-biased-zero-env54":
        if not final["success"]:
            raise EvidenceError("reference media must be the successful biased-zero rollout")
    return metadata, rows, {"probe": probe, "report": report, "evidence": evidence}


def outcome_text(stem: str, final: dict[str, Any]) -> tuple[str, str, str]:
    if final["success"]:
        return "기준 성공", "0.4초 동안 모든 물리 샘플 안정", "#067647"
    reasons = []
    if float(final["lateral_error_um"]) > LATERAL_LIMIT_UM:
        reasons.append(f"횡오차 {final['lateral_error_um']:.1f} µm")
    if float(final["relative_linear_speed_mm_s"]) > LINEAR_LIMIT_MM_S:
        reasons.append(f"상대 선속도 {final['relative_linear_speed_mm_s']:.2f} mm/s")
    if float(final["relative_angular_speed_rad_s"]) > ANGULAR_LIMIT_RAD_S:
        reasons.append(f"상대 각속도 {final['relative_angular_speed_rad_s']:.3f} rad/s")
    return "개발 실패 · 타임아웃", " · ".join(reasons) + " · 안정 유지 0초", "#b42318"


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size)


def _gauge(draw, box, value: float, limit: float, label: str, unit: str) -> None:
    x0, y0, x1, y1 = box
    passed = value <= limit
    color = "#067647" if passed else "#d92d20"
    draw.text((x0, y0), label, font=_font(18, True), fill="#101828")
    draw.text((x1 - 205, y0), f"{value:.3g} {unit}", font=_font(18, True), fill=color)
    y = y0 + 34
    draw.rounded_rectangle((x0, y, x1, y + 15), radius=7, fill="#eaecf0")
    ratio = min(1.0, value / max(limit * 2.0, 1e-12))
    draw.rounded_rectangle((x0, y, x0 + max(4, int((x1 - x0) * ratio)), y + 15), radius=7, fill=color)
    threshold_x = x0 + int((x1 - x0) * 0.5)
    draw.line((threshold_x, y - 4, threshold_x, y + 19), fill="#344054", width=2)
    draw.text((x0, y + 22), f"기준 ≤ {limit:g} {unit}", font=_font(14), fill="#667085")


def compose_frame(
    frame: np.ndarray, row: dict[str, Any], stem: str, final_metrics: dict[str, Any]
) -> np.ndarray:
    import cv2
    from PIL import Image, ImageDraw
    m = metrics(row)
    canvas = Image.new("RGB", OUTPUT_SIZE, "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    view = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Fixed native-pixel crop around the two gripper tips and held connector.
    focus = view[190:410, 350:620]
    view = cv2.resize(view, (1040, 780), interpolation=cv2.INTER_AREA)
    canvas.paste(Image.fromarray(view), (24, 96))
    focus = cv2.resize(focus, (294, 250), interpolation=cv2.INTER_CUBIC)
    draw.rounded_rectangle((38, 610, 346, 874), radius=9, fill="white", outline="#667085", width=3)
    canvas.paste(Image.fromarray(focus), (45, 617))
    draw.rectangle((45, 617, 339, 653), fill="#101828")
    draw.text((58, 621), "양손 끝단 원본 픽셀 확대", font=_font(15, True), fill="white")
    title = "기준 제어기 · 성공" if stem.startswith("v3-") else "행동복제(BC) 정책 · 실패"
    draw.text((28, 24), title, font=_font(30, True), fill="#101828")
    draw.text((625, 37), "렌더: 현재 물리 프레임", font=_font(16), fill="#475467")
    draw.text((1100, 37), "판정: 제어 스텝 기록", font=_font(16), fill="#475467")
    draw.rounded_rectangle((1086, 96, 1576, 876), radius=16, fill="white", outline="#d0d5dd", width=2)
    draw.text((1110, 120), f"물리 {m['physics_time_s']:.3f} s", font=_font(21, True), fill="#101828")
    draw.text((1370, 124), f"제어 #{m['control_step']}", font=_font(16), fill="#667085")
    _gauge(draw, (1110, 184, 1548, 250), float(m["lateral_error_um"]), LATERAL_LIMIT_UM, "횡오차", "µm")
    _gauge(draw, (1110, 284, 1548, 350), float(m["relative_linear_speed_mm_s"]), LINEAR_LIMIT_MM_S, "상대 선속도", "mm/s")
    _gauge(draw, (1110, 384, 1548, 450), float(m["relative_angular_speed_rad_s"]), ANGULAR_LIMIT_RAD_S, "상대 각속도", "rad/s")
    draw.text((1110, 493), "안정 유지", font=_font(18, True), fill="#101828")
    dwell_color = "#067647" if float(m["dwell_s"]) >= DWELL_REQUIRED_S else "#b42318"
    draw.text((1378, 493), f"{m['dwell_s']:.3f} / {DWELL_REQUIRED_S:.1f} s", font=_font(18, True), fill=dwell_color)
    draw.text((1110, 535), "매 제어 스텝의 9개 물리 샘플이", font=_font(15), fill="#475467")
    draw.text((1110, 560), "모두 기준을 만족해야 시간이 누적됩니다.", font=_font(15), fill="#475467")
    draw.text((1110, 615), f"삽입 깊이  {m['depth_mm']:.2f} mm", font=_font(16), fill="#344054")
    draw.text((1110, 645), f"접촉 하중 proxy  {m['contact_proxy_n']:.2f} N", font=_font(16), fill="#344054")
    draw.text((1110, 670), "PhysX 접촉 proxy · 손목 F/T 센서 아님", font=_font(13), fill="#667085")
    heading, detail, color = outcome_text(stem, final_metrics)
    draw.rounded_rectangle((1104, 718, 1558, 848), radius=12, fill="#fef3f2" if color == "#b42318" else "#ecfdf3", outline=color, width=2)
    draw.text((1124, 733), "최종 결과", font=_font(14), fill="#667085")
    draw.text((1124, 758), heading, font=_font(23, True), fill=color)
    # Two compact lines keep the final failure readable without inventing events.
    parts = detail.split(" · ")
    draw.text((1124, 801), " · ".join(parts[:2]), font=_font(14), fill="#344054")
    draw.text((1124, 826), " · ".join(parts[2:]), font=_font(14), fill="#344054")
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


def ffmpeg_writer(path: Path) -> subprocess.Popen:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
        "-pix_fmt", "bgr24", "-s", f"{OUTPUT_SIZE[0]}x{OUTPUT_SIZE[1]}", "-r", "60",
        "-i", "-", "-an", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def export_video(source: Path, output: Path, rows: list[dict[str, Any]], stem: str) -> Path:
    import cv2
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise EvidenceError(f"cannot open video: {source}")
    writer = ffmpeg_writer(output)
    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    count = 0
    final_metrics = metrics(rows[-1])
    final_composed: np.ndarray | None = None
    try:
        for row in rows:
            ok, frame = capture.read()
            if not ok:
                raise EvidenceError(f"video ended before frame {count}")
            composed = compose_frame(frame, row, stem, final_metrics)
            writer.stdin.write(composed.tobytes())
            final_composed = composed
            count += 1
        ok, _ = capture.read()
        if ok:
            raise EvidenceError("video contains frames absent from JSONL")
    finally:
        capture.release()
        writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    if count != len(rows):
        raise EvidenceError("export did not consume every native frame")
    if final_composed is None:
        raise EvidenceError("export produced no final preview frame")
    preview = output.with_name(output.stem + "-final-preview.png")
    if not cv2.imwrite(str(preview), final_composed):
        raise RuntimeError("could not write final preview")
    return preview


CSV_FIELDS = [
    "frame_index", "physics_time_s", "control_time_s", "control_step", "depth_mm",
    "lateral_error_um", "relative_linear_speed_mm_s", "relative_angular_speed_rad_s",
    "dwell_s", "contact_proxy_n", "physics_all_samples_stable", "lateral_ok",
    "dwell_ok", "terminal", "success",
]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})


def draw_trace_png(path: Path, traces: dict[str, list[dict[str, Any]]]) -> None:
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (1800, 1120), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 35), "RJ45 정책 진단 · 개별 실행의 시간 추세", font=_font(34, True), fill="#101828")
    specs = [
        ("lateral_error_um", "횡오차 (µm)", LATERAL_LIMIT_UM),
        ("relative_linear_speed_mm_s", "상대 선속도 (mm/s)", LINEAR_LIMIT_MM_S),
        ("relative_angular_speed_rad_s", "상대 각속도 (rad/s)", ANGULAR_LIMIT_RAD_S),
    ]
    colors = {"v3-biased-zero-env54": "#067647", "v5-bc-env54": "#d92d20", "v5-bc-env42": "#f79009"}
    labels = {"v3-biased-zero-env54": "기준 성공 env54", "v5-bc-env54": "BC 실패 env54", "v5-bc-env42": "BC 실패 env42"}
    for stem, x in zip(traces, (850, 1150, 1450)):
        draw.line((x, 64, x + 48, 64), fill=colors[stem], width=5)
        draw.text((x + 58, 50), labels[stem], font=_font(17), fill="#344054")
    common_duration = max(float(rows[-1]["physics_time_s"]) for rows in traces.values())
    for chart_index, (key, label, limit) in enumerate(specs):
        x0, y0, x1, y1 = 100, 145 + chart_index * 310, 1720, 390 + chart_index * 310
        draw.rectangle((x0, y0, x1, y1), outline="#d0d5dd", width=2)
        all_values = [float(row[key]) for rows in traces.values() for row in rows]
        ymax = max(limit * 2.0, np.percentile(all_values, 99.0) * 1.08)
        observed_max = max(all_values)
        clipped_count = sum(value > ymax for value in all_values)
        threshold_y = y1 - int((limit / ymax) * (y1 - y0))
        draw.line((x0, threshold_y, x1, threshold_y), fill="#475467", width=2)
        draw.text((x0 + 10, threshold_y - 25), f"기준 {limit:g}", font=_font(14), fill="#475467")
        draw.text((x0, y0 - 34), label, font=_font(21, True), fill="#101828")
        for tick_index in range(5):
            tick_value = ymax * tick_index / 4.0
            tick_y = y1 - int((tick_value / ymax) * (y1 - y0))
            draw.line((x0 - 7, tick_y, x0, tick_y), fill="#667085", width=2)
            draw.text((x0 - 92, tick_y - 10), f"{tick_value:.3g}", font=_font(13), fill="#667085")
        for stem, rows in traces.items():
            points = []
            for row in rows:
                x = x0 + int(float(row["physics_time_s"]) / common_duration * (x1 - x0))
                y = y1 - int(min(float(row[key]), ymax) / ymax * (y1 - y0))
                points.append((x, y))
            if len(points) > 1:
                draw.line(points, fill=colors[stem], width=3)
        tick_seconds = list(range(0, int(common_duration) + 1, 5))
        if not math.isclose(float(tick_seconds[-1]), common_duration, abs_tol=0.1):
            tick_seconds.append(common_duration)
        for seconds in tick_seconds:
            tick_x = x0 + int(float(seconds) / common_duration * (x1 - x0))
            draw.line((tick_x, y1, tick_x, y1 + 6), fill="#667085", width=2)
            label_x = tick_x - (14 if seconds else 0)
            draw.text((label_x, y1 + 8), f"{seconds:g}s", font=_font(13), fill="#667085")
        if clipped_count:
            draw.text(
                (x1 - 560, y0 + 10),
                f"표시 상한 {ymax:.3g} · 상한 초과 {clipped_count}점 · 관측 최대 {observed_max:.3g}",
                font=_font(13), fill="#b42318",
            )
    draw.text((100, 1080), "50 µm는 이 프로젝트의 평가 기준입니다. 속도·안정 유지 판정은 제어 스텝 기록에서 읽었습니다.", font=_font(16), fill="#475467")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def write_svg(path: Path, traces: dict[str, list[dict[str, Any]]]) -> None:
    specs = [
        ("lateral_error_um", "Lateral error (um)", LATERAL_LIMIT_UM),
        ("relative_linear_speed_mm_s", "Relative linear speed (mm/s)", LINEAR_LIMIT_MM_S),
        ("relative_angular_speed_rad_s", "Relative angular speed (rad/s)", ANGULAR_LIMIT_RAD_S),
    ]
    colors = {"v3-biased-zero-env54": "#067647", "v5-bc-env54": "#d92d20", "v5-bc-env42": "#f79009"}
    duration = max(float(rows[-1]["physics_time_s"]) for rows in traces.values())
    fragments = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="850" viewBox="0 0 1200 850">',
        '<rect width="1200" height="850" fill="#ffffff"/>',
        '<text x="55" y="48" font-family="sans-serif" font-size="28" font-weight="700" fill="#101828">RJ45 policy diagnostic · separate rollout traces</text>',
    ]
    for index, (key, label, limit) in enumerate(specs):
        x0, y0, width, height = 80, 105 + index * 235, 1060, 175
        values = [float(row[key]) for rows in traces.values() for row in rows]
        ymax = max(limit * 2.0, float(np.percentile(values, 99.0)) * 1.08)
        observed_max = max(values)
        clipped_count = sum(value > ymax for value in values)
        threshold_y = y0 + height - (limit / ymax) * height
        fragments.extend([
            f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="none" stroke="#d0d5dd"/>',
            f'<text x="{x0}" y="{y0 - 12}" font-family="sans-serif" font-size="18" font-weight="700" fill="#101828">{label}</text>',
            f'<line x1="{x0}" y1="{threshold_y:.2f}" x2="{x0 + width}" y2="{threshold_y:.2f}" stroke="#475467" stroke-dasharray="7 5"/>',
            f'<text x="{x0 + 8}" y="{threshold_y - 7:.2f}" font-family="sans-serif" font-size="13" fill="#475467">limit {limit:g}</text>',
        ])
        for tick_index in range(5):
            tick_value = ymax * tick_index / 4.0
            tick_y = y0 + height - (tick_value / ymax) * height
            fragments.append(f'<text x="12" y="{tick_y + 5:.2f}" font-family="sans-serif" font-size="12" fill="#667085">{tick_value:.3g}</text>')
        for seconds in list(range(0, int(duration) + 1, 5)) + [duration]:
            tick_x = x0 + float(seconds) / duration * width
            fragments.append(f'<text x="{tick_x - 8:.2f}" y="{y0 + height + 20}" font-family="sans-serif" font-size="12" fill="#667085">{seconds:g}s</text>')
        if clipped_count:
            fragments.append(f'<text x="650" y="{y0 + 18}" font-family="sans-serif" font-size="12" fill="#b42318">display max {ymax:.3g}; clipped {clipped_count}; observed max {observed_max:.3g}</text>')
        for stem, rows in traces.items():
            points = " ".join(
                f"{x0 + float(row['physics_time_s']) / duration * width:.2f},{y0 + height - min(float(row[key]), ymax) / ymax * height:.2f}"
                for row in rows
            )
            fragments.append(f'<polyline points="{points}" fill="none" stroke="{colors[stem]}" stroke-width="2.2"/>')
    fragments.extend([
        '<line x1="80" y1="815" x2="120" y2="815" stroke="#067647" stroke-width="5"/><text x="130" y="821" font-family="sans-serif" font-size="16">reference success env54</text>',
        '<line x1="410" y1="815" x2="450" y2="815" stroke="#d92d20" stroke-width="5"/><text x="460" y="821" font-family="sans-serif" font-size="16">BC timeout env54</text>',
        '<line x1="740" y1="815" x2="780" y2="815" stroke="#f79009" stroke-width="5"/><text x="790" y="821" font-family="sans-serif" font-size="16">BC timeout env42</text>',
        '</svg>',
    ])
    content = "\n".join(fragments)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args()
    stems = ("v3-biased-zero-env54", "v5-bc-env54", "v5-bc-env42")
    args.media_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema_version": 1, "thresholds": {
        "lateral_error_um": LATERAL_LIMIT_UM, "relative_linear_speed_mm_s": LINEAR_LIMIT_MM_S,
        "relative_angular_speed_rad_s": ANGULAR_LIMIT_RAD_S, "stable_dwell_s": DWELL_REQUIRED_S,
        "lateral_threshold_scope": "project_evaluation_criterion",
    }, "runs": []}
    selection_reasons = {
        "v3-biased-zero-env54": "preselected paired reference: biased-zero success for environment 54",
        "v5-bc-env54": "preselected paired contrast: BC timeout for the same environment 54",
        "v5-bc-env42": "preselected representative BC timeout case from the fixed evaluation result",
    }
    all_metrics: dict[str, list[dict[str, Any]]] = {}
    for stem in stems:
        paths = {
            "video": args.input_dir / f"{stem}.mp4",
            "frames": args.input_dir / f"{stem}.mp4.frames.jsonl",
            "metadata": args.input_dir / f"{stem}.mp4.metadata.json",
            "evidence": args.input_dir / f"{stem}.mp4.evidence.json",
            "report": args.input_dir / f"{stem}.report.json",
        }
        metadata, rows, details = validate_inputs(stem, paths)
        derived = [metrics(row) for row in rows]
        all_metrics[stem] = derived
        output = args.media_dir / f"{stem}-diagnostic-60fps.mp4"
        csv_path = args.evidence_dir / f"{stem}-timeseries.csv"
        preview = export_video(paths["video"], output, rows, stem)
        write_csv(csv_path, derived)
        output_probe = probe_video(output)
        if output_probe["frame_count"] != len(rows) or not math.isclose(output_probe["fps"], FPS, abs_tol=0.01):
            raise EvidenceError("output did not preserve native frame count/FPS")
        final = derived[-1]
        manifest["runs"].append({
            "stem": stem, "input": {key: {"sha256": sha256(path), "bytes": path.stat().st_size} for key, path in paths.items()},
            "policy_lineage": metadata["policy_lineage"], "policy_lineage_sha256": metadata["policy_lineage_sha256"],
            "selection": {"episode_index": metadata["selected_episode_index"], "environment_index": metadata["selected_environment_index"], "preselected_before_capture": True, "reason": selection_reasons[stem]},
            "output": {"path": output.name, "sha256": sha256(output), "bytes": output.stat().st_size, **output_probe},
            "final_preview": {"path": preview.name, "sha256": sha256(preview), "bytes": preview.stat().st_size},
            "timeseries": {"path": csv_path.name, "sha256": sha256(csv_path), "rows": len(derived)},
            "terminal": {key: final[key] for key in CSV_FIELDS if key not in ("frame_index", "control_time_s", "control_step")},
            "termination_reason": metadata["selected_terminal"]["termination_reason"],
            "presentation_contract": "one output frame per native source frame; no interpolation, duplication, added hold, or 3D modification",
        })
    chart = args.media_dir / "policy-diagnostic-traces.png"
    svg = args.media_dir / "policy-diagnostic-traces.svg"
    draw_trace_png(chart, all_metrics)
    write_svg(svg, all_metrics)
    manifest["charts"] = [
        {"path": chart.name, "sha256": sha256(chart)}, {"path": svg.name, "sha256": sha256(svg)}
    ]
    manifest["disclosures"] = {
        "bc_actor": EXPECTED_BC_ACTOR, "bc_is_not_completed_ppo": True,
        "contact_signal": "PhysX contact proxy, not wrist force/torque sensor",
        "render_vs_control_time": "geometry is the native physics frame; stability telemetry is the recorded control-step snapshot",
        "failure_interpretation": "observed evaluation criteria failures only; no action-fault causality claim",
    }
    manifest_path = args.evidence_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
