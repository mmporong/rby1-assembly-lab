#!/usr/bin/env python3
"""Build a truthful side-by-side fresh-05 replay video from recovered evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Mapping


FPS = 60
SOURCE_SIZE = (1280, 960)
OUTPUT_SIZE = (1920, 1080)
PANEL_SIZE = (928, 696)
PANEL_Y = 160
PANEL_X = {"multi64": 24, "zero": 968}
CARD_DURATION_S = 2.0
LATERAL_LIMIT_UM = 50.0
DWELL_REQUIRED_S = 0.4
FONT_REGULAR = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FONT_BOLD = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
STEM = "fresh05-multibias-comparison"


class MediaEvidenceError(RuntimeError):
    """Raised when recovered source evidence does not satisfy the media contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(data: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MediaEvidenceError(f"{field} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise MediaEvidenceError(f"{field} JSON root must be an object")
    return value


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MediaEvidenceError(f"{field} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise MediaEvidenceError(f"{field} must be finite")
    return number


def _positive_int(value: Any, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise MediaEvidenceError(f"{field} must be a positive integer")
    return value


def _read_source_once(
    media_root: Path,
    relative_path: str,
    recovery: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    record = recovery.get(relative_path)
    if not isinstance(record, dict):
        raise MediaEvidenceError(f"recovery manifest has no entry for {relative_path}")
    expected_sha = record.get("sha256")
    expected_bytes = record.get("bytes")
    if (
        not isinstance(expected_sha, str)
        or len(expected_sha) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha)
        or type(expected_bytes) is not int
        or expected_bytes < 0
    ):
        raise MediaEvidenceError(f"invalid recovery record for {relative_path}")
    path = media_root / relative_path
    data = path.read_bytes()
    observed_sha = _sha256(data)
    if len(data) != expected_bytes or observed_sha != expected_sha:
        raise MediaEvidenceError(f"recovered input differs from manifest: {relative_path}")
    return data, {
        "path": str(path),
        "relative_path": relative_path,
        "bytes": len(data),
        "sha256": observed_sha,
    }


def _frame_rows(data: bytes, mode: str) -> list[dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise MediaEvidenceError(f"{mode} frame JSONL is not UTF-8") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise MediaEvidenceError(
                f"{mode} frame JSONL line {line_number} is invalid"
            ) from error
        if not isinstance(row, dict) or row.get("frame_index") != len(rows):
            raise MediaEvidenceError(f"{mode} frame indices are not contiguous")
        physics_step = _positive_int(row.get("physics_step"), f"{mode} physics_step")
        physics_time_s = _finite(row.get("physics_time_s"), f"{mode} physics_time_s")
        if physics_step != 4 * (len(rows) + 1):
            raise MediaEvidenceError(f"{mode} physics step does not match frame index")
        if not math.isclose(
            physics_time_s, (len(rows) + 1) / FPS, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise MediaEvidenceError(f"{mode} physics time does not match frame index")
        _finite(row.get("actual_lateral_error_m"), f"{mode} lateral error")
        _finite(row.get("actual_depth_m"), f"{mode} depth")
        snapshot = row.get("control_step_snapshot")
        if not isinstance(snapshot, dict):
            raise MediaEvidenceError(f"{mode} frame has no control-step snapshot")
        dwell = _finite(
            snapshot.get("independent_eval_dwell_s"), f"{mode} dwell"
        )
        if dwell < 0.0:
            raise MediaEvidenceError(f"{mode} dwell must be non-negative")
        rows.append(row)
    if not rows:
        raise MediaEvidenceError(f"{mode} frame JSONL is empty")
    expected_frames = {"multi64": 760, "zero": 1454}[mode]
    if len(rows) != expected_frames:
        raise MediaEvidenceError(
            f"{mode} frame count changed: {len(rows)} != {expected_frames}"
        )
    if any(
        int(current["physics_step"]) - int(previous["physics_step"]) != 4
        for previous, current in zip(rows, rows[1:])
    ):
        raise MediaEvidenceError(f"{mode} render cadence is not four physics steps")
    if any(
        float(current["physics_time_s"]) <= float(previous["physics_time_s"])
        for previous, current in zip(rows, rows[1:])
    ):
        raise MediaEvidenceError(f"{mode} physics timestamps do not increase")
    return rows


def _causal_dwell(rows: list[dict[str, Any]], mode: str) -> list[float | None]:
    if len(rows) % 2:
        raise MediaEvidenceError(f"{mode} frame count does not form 30 Hz control groups")
    expected_groups = {"multi64": 380, "zero": 727}[mode]
    if len(rows) // 2 != expected_groups:
        raise MediaEvidenceError(f"{mode} control-group count changed")
    displayed: list[float | None] = []
    completed_dwell: float | None = None
    for group_index in range(expected_groups):
        first = rows[2 * group_index]
        second = rows[2 * group_index + 1]
        entry_step = first.get("control_step_entry_physics_step")
        if type(entry_step) is not int or second.get("control_step_entry_physics_step") != entry_step:
            raise MediaEvidenceError(f"{mode} control-group entry step changed")
        offsets = (
            int(first["physics_step"]) - entry_step,
            int(second["physics_step"]) - entry_step,
        )
        if offsets != (4, 8):
            raise MediaEvidenceError(f"{mode} control-group offsets are not [4, 8]")
        first_snapshot = first["control_step_snapshot"]
        second_snapshot = second["control_step_snapshot"]
        first_recorded = _finite(
            first_snapshot.get("independent_eval_dwell_s"), f"{mode} first-half dwell"
        )
        second_recorded = _finite(
            second_snapshot.get("independent_eval_dwell_s"), f"{mode} control-end dwell"
        )
        if not math.isclose(first_recorded, second_recorded, rel_tol=0.0, abs_tol=1.0e-12):
            raise MediaEvidenceError(f"{mode} paired snapshot dwell values differ")
        displayed.append(completed_dwell)
        completed_dwell = second_recorded
        displayed.append(completed_dwell)
    if displayed[0] is not None:
        raise MediaEvidenceError(f"{mode} first half-frame must have unknown dwell")
    return displayed


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, shell=False
        )
        stream = json.loads(completed.stdout)["streams"][0]
        average_numerator, average_denominator = stream["avg_frame_rate"].split("/")
        rate_numerator, rate_denominator = stream["r_frame_rate"].split("/")
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "average_fps": float(average_numerator) / float(average_denominator),
            "nominal_fps": float(rate_numerator) / float(rate_denominator),
            "frame_count": int(stream["nb_read_frames"]),
            "command": command,
        }
    except (subprocess.CalledProcessError, KeyError, ValueError, IndexError) as error:
        raise MediaEvidenceError(f"ffprobe failed for {path}") from error


def _validate_audit(audit: Mapping[str, Any]) -> None:
    evaluation = audit.get("evaluation")
    if audit.get("status") != "pass" or not isinstance(evaluation, dict):
        raise MediaEvidenceError("audit is not a completed pass")
    if evaluation.get("episode_count") != 27:
        raise MediaEvidenceError("audit does not contain 27 valid evaluations")
    if evaluation.get("fresh8_successes") != {
        "multi64": 3,
        "single64": 2,
        "zero": 2,
    }:
        raise MediaEvidenceError("audit fresh8 totals changed")
    if evaluation.get("followup_multiseed_candidate") is not False:
        raise MediaEvidenceError("audit candidate decision changed")
    claims = audit.get("claims")
    if not isinstance(claims, dict) or claims.get("effect_proven") is not False:
        raise MediaEvidenceError("audit effect claim changed")


def _validate_report(report: Mapping[str, Any], mode: str) -> None:
    expected_classification = {
        "multi64": "valid_success",
        "zero": "valid_failure",
    }[mode]
    comparison = report.get("comparison")
    replay = report.get("replay")
    lineage = report.get("lineage")
    if report.get("schema_name") != "rj45_multibias_replay_v1":
        raise MediaEvidenceError(f"{mode} report schema changed")
    if report.get("status") != "completed" or report.get("mode") != mode:
        raise MediaEvidenceError(f"{mode} report is not completed")
    if not isinstance(comparison, dict) or comparison.get(
        "initial_base79_matches"
    ) is not True or comparison.get("classification_matches") is not True:
        raise MediaEvidenceError(f"{mode} replay does not match the primary episode")
    if (
        comparison.get("primary_classification") != expected_classification
        or comparison.get("replay_classification") != expected_classification
    ):
        raise MediaEvidenceError(f"{mode} classification is not the selected contrast")
    if (
        not isinstance(lineage, dict)
        or lineage.get("classification") != expected_classification
    ):
        raise MediaEvidenceError(f"{mode} primary lineage changed")
    case = lineage.get("case")
    if not isinstance(case, dict) or case.get("case_id") != "fresh-05":
        raise MediaEvidenceError(f"{mode} primary case lineage changed")
    outcome = replay.get("outcome") if isinstance(replay, dict) else None
    if (
        not isinstance(outcome, dict)
        or outcome.get("classification") != expected_classification
        or outcome.get("valid") is not True
        or outcome.get("success") is not (mode == "multi64")
    ):
        raise MediaEvidenceError(f"{mode} replay outcome changed")


def _canonical_json_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MediaEvidenceError("policy lineage is not canonical JSON") from error
    return _sha256(encoded)


def _validate_evidence(
    mode: str,
    evidence: Mapping[str, Any],
    metadata: Mapping[str, Any],
    report: Mapping[str, Any],
    source_records: Mapping[str, Mapping[str, Any]],
) -> str:
    if (
        evidence.get("schema_version") != 1
        or evidence.get("completed") is not True
        or evidence.get("rollout_completed") is not True
    ):
        raise MediaEvidenceError(f"{mode} evidence is incomplete")
    artifacts = evidence.get("artifacts")
    if not isinstance(artifacts, dict):
        raise MediaEvidenceError(f"{mode} evidence artifacts are missing")
    for label in ("video", "frames", "metadata"):
        artifact = artifacts.get(label)
        record = source_records.get(label)
        if (
            not isinstance(artifact, dict)
            or not isinstance(record, Mapping)
            or artifact.get("sha256") != record.get("sha256")
        ):
            raise MediaEvidenceError(f"{mode} evidence artifact SHA changed: {label}")
    evidence_lineage = evidence.get("policy_lineage")
    metadata_lineage = metadata.get("policy_lineage")
    report_lineage = report.get("lineage")
    if (
        not isinstance(evidence_lineage, dict)
        or evidence_lineage != metadata_lineage
        or evidence_lineage != report_lineage
    ):
        raise MediaEvidenceError(f"{mode} policy lineage differs across inputs")
    canonical_sha = _canonical_json_sha256(evidence_lineage)
    if (
        evidence.get("policy_lineage_sha256") != canonical_sha
        or metadata.get("policy_lineage_sha256") != canonical_sha
    ):
        raise MediaEvidenceError(f"{mode} canonical policy-lineage SHA changed")
    return canonical_sha


def _validate_recording(
    mode: str,
    metadata: Mapping[str, Any],
    report: Mapping[str, Any],
    rows: list[dict[str, Any]],
    probe: Mapping[str, Any],
    policy_lineage_sha256: str,
) -> dict[str, float | bool | str]:
    _validate_report(report, mode)
    if metadata.get("completed") is not True or metadata.get("rollout_completed") is not True:
        raise MediaEvidenceError(f"{mode} recording is incomplete")
    if (
        metadata.get("cleanup_errors") != []
        or metadata.get("physics_or_action_contract_modified") is not False
        or metadata.get("selected_environment_index") != 0
        or metadata.get("selected_episode_index") != 1
    ):
        raise MediaEvidenceError(f"{mode} recording identity or cleanup state changed")
    if any(row.get("policy_lineage_sha256") != policy_lineage_sha256 for row in rows):
        raise MediaEvidenceError(f"{mode} frame policy lineage changed")
    frame_count = len(rows)
    if (
        metadata.get("captured_frames") != frame_count
        or metadata.get("frame_metadata_records") != frame_count
        or probe.get("frame_count") != frame_count
    ):
        raise MediaEvidenceError(f"{mode} video/JSONL/metadata frame counts differ")
    captured_duration_s = _finite(
        metadata.get("captured_duration_s"), f"{mode} captured duration"
    )
    if not math.isclose(captured_duration_s, frame_count / FPS, abs_tol=1.0e-9):
        raise MediaEvidenceError(f"{mode} captured duration does not match 60 fps")
    capture_plan = metadata.get("capture_plan")
    if (
        not isinstance(capture_plan, dict)
        or capture_plan.get("fps") != FPS
        or capture_plan.get("render_stride_physics_steps") != 4
    ):
        raise MediaEvidenceError(f"{mode} capture cadence changed")
    if (
        (probe.get("width"), probe.get("height")) != SOURCE_SIZE
        or not math.isclose(float(probe.get("average_fps", 0.0)), FPS, abs_tol=0.001)
        or not math.isclose(float(probe.get("nominal_fps", 0.0)), FPS, abs_tol=0.001)
    ):
        raise MediaEvidenceError(f"{mode} source is not 1280x960 at 60 fps")
    terminal = metadata.get("selected_terminal")
    if not isinstance(terminal, dict) or terminal.get("valid") is not True:
        raise MediaEvidenceError(f"{mode} metadata has no valid terminal")
    success = mode == "multi64"
    if terminal.get("success") is not success:
        raise MediaEvidenceError(f"{mode} metadata terminal outcome changed")
    terminal_rows = [
        row for row in rows if row.get("terminal_observed_after_control_step") is True
    ]
    if len(terminal_rows) != 1 or terminal_rows[0] is not rows[-1]:
        raise MediaEvidenceError(f"{mode} terminal frame is not uniquely last")
    last = rows[-1]
    if last.get("success_observed_after_control_step") is not success:
        raise MediaEvidenceError(f"{mode} final frame outcome changed")
    lateral_um = 1.0e6 * _finite(
        last.get("actual_lateral_error_m"), f"{mode} final lateral error"
    )
    depth_mm = 1.0e3 * _finite(last.get("actual_depth_m"), f"{mode} final depth")
    dwell_s = _finite(
        last["control_step_snapshot"].get("independent_eval_dwell_s"),
        f"{mode} final dwell",
    )
    replay_outcome = report["replay"]["outcome"]
    for field, observed in (
        ("actual_lateral_error_m", lateral_um / 1.0e6),
        ("actual_depth_m", depth_mm / 1.0e3),
    ):
        terminal_value = _finite(terminal.get(field), f"{mode} terminal {field}")
        report_value = _finite(
            replay_outcome.get(field), f"{mode} report outcome {field}"
        )
        if not math.isclose(observed, terminal_value, rel_tol=0.0, abs_tol=1.0e-9):
            raise MediaEvidenceError(f"{mode} frame and metadata {field} differ")
        if not math.isclose(observed, report_value, rel_tol=0.0, abs_tol=1.0e-9):
            raise MediaEvidenceError(f"{mode} frame and report {field} differ")
    return {
        "classification": "valid_success" if success else "valid_failure",
        "success": success,
        "lateral_error_um": lateral_um,
        "depth_mm": depth_mm,
        "dwell_s": dwell_s,
        "physics_time_s": _finite(last.get("physics_time_s"), f"{mode} final time"),
    }


def _font(size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    path = FONT_BOLD if bold else FONT_REGULAR
    return ImageFont.truetype(str(path), size)


def _draw_background(
    path: Path, terminal: Mapping[str, Mapping[str, Any]]
) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", OUTPUT_SIZE, "#ffffff")
    draw = ImageDraw.Draw(image)
    draw.text(
        (24, 20),
        "RJ45 fresh-05 · 다중조건 PPO와 무보정 비교",
        font=_font(34, bold=True),
        fill="#101828",
    )
    draw.text(
        (24, 72),
        "미사용 8조건 성공: single64 2/8 · multi64 3/8 · zero 2/8",
        font=_font(23, bold=True),
        fill="#344054",
    )
    draw.text(
        (24, 111),
        "다중조건 PPO가 1건 높았지만 사전 후보 기준(+2)에 미달 · 개선 효과 미입증",
        font=_font(20),
        fill="#667085",
    )
    headers = {
        "multi64": ("다중조건 PPO · 성공", "#067647"),
        "zero": ("무보정 · 실패", "#b42318"),
    }
    for mode in ("multi64", "zero"):
        x = PANEL_X[mode]
        title, color = headers[mode]
        draw.rounded_rectangle(
            (x, PANEL_Y, x + PANEL_SIZE[0], PANEL_Y + PANEL_SIZE[1]),
            radius=12,
            fill="#f8fafc",
            outline="#d0d5dd",
            width=2,
        )
        draw.text((x + 32, PANEL_Y + 180), title, font=_font(31, bold=True), fill=color)
        draw.text(
            (x + 32, PANEL_Y + 245),
            "촬영 종료 · 이후 화면은 결과 카드",
            font=_font(20),
            fill="#475467",
        )
        result = terminal[mode]
        draw.text(
            (x + 32, PANEL_Y + 315),
            f"최종 횡오차  {float(result['lateral_error_um']):.2f} µm",
            font=_font(25, bold=True),
            fill="#101828",
        )
        draw.text(
            (x + 32, PANEL_Y + 365),
            f"삽입 깊이  {float(result['depth_mm']):.3f} mm",
            font=_font(24),
            fill="#344054",
        )
        draw.text(
            (x + 32, PANEL_Y + 410),
            f"안정 유지  {float(result['dwell_s']):.3f} s",
            font=_font(24),
            fill="#344054",
        )
        draw.text(
            (x + 32, PANEL_Y + 490),
            "fresh replay 측정값 · 원 평가 수치 재사용 아님",
            font=_font(18),
            fill="#667085",
        )
    draw.rounded_rectangle((24, 884, 1896, 1048), radius=12, fill="#f2f4f7")
    draw.text(
        (48, 905),
        "횡오차·깊이: 60 fps 렌더 시점 · 유지: 30 Hz 제어 종료값의 인과적 유지",
        font=_font(20, bold=True),
        fill="#344054",
    )
    draw.text(
        (48, 947),
        "횡오차 50 µm · 안정 유지 0.4 s는 프로젝트 평가 기준이며 제조사 규격이 아닙니다.",
        font=_font(19),
        fill="#475467",
    )
    draw.text(
        (48, 990),
        "선택된 fresh-05의 사후 시각화이며 학습·benchmark 통계 증거가 아닙니다.",
        font=_font(19),
        fill="#475467",
    )
    image.save(path)


def _ass_time(centiseconds: int) -> str:
    if type(centiseconds) is not int or centiseconds < 0:
        raise MediaEvidenceError("ASS timestamp must be non-negative centiseconds")
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, hundredths = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{hundredths:02d}"


def _write_ass(path: Path, rows_by_mode: Mapping[str, list[dict[str, Any]]]) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
Style: Metric,Noto Sans CJK KR,24,&H00FFFFFF,&H00FFFFFF,&H00101828,&HC0000000,-1,0,0,0,100,100,0,0,3,1,0,2,20,20,18,1
Style: Header,Noto Sans CJK KR,27,&H00FFFFFF,&H00FFFFFF,&H00101828,&H78000000,-1,0,0,0,100,100,0,0,3,1,0,2,20,20,12,1

[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""
    events: list[str] = []
    total_frames = max(len(rows) for rows in rows_by_mode.values()) + int(
        CARD_DURATION_S * FPS
    )
    header_end = _ass_time((total_frames * 100) // FPS)
    for mode, label in (("multi64", "다중조건 PPO"), ("zero", "무보정")):
        center_x = PANEL_X[mode] + PANEL_SIZE[0] // 2
        events.append(
            f"Dialogue: 1,0:00:00.00,{header_end},Header,{mode}-header,0,0,0,,"
            f"{{\\pos({center_x},{PANEL_Y + 30})}}{label}"
        )
    for mode in ("multi64", "zero"):
        center_x = PANEL_X[mode] + PANEL_SIZE[0] // 2
        metric_y = PANEL_Y + PANEL_SIZE[1] - 28
        causal_dwell = _causal_dwell(rows_by_mode[mode], mode)
        for index, row in enumerate(rows_by_mode[mode]):
            dwell = causal_dwell[index]
            dwell_text = "—" if dwell is None else f"{dwell:.3f} s"
            text = (
                f"물리 {_finite(row['physics_time_s'], 'physics time'):.3f} s  |  "
                f"횡오차 {1e6 * _finite(row['actual_lateral_error_m'], 'lateral'):.2f} µm  |  "
                f"깊이 {1e3 * _finite(row['actual_depth_m'], 'depth'):.3f} mm  |  "
                f"유지(제어 30 Hz) {dwell_text}"
            )
            start = _ass_time((index * 100) // FPS)
            end = _ass_time(((index + 1) * 100) // FPS)
            events.append(
                f"Dialogue: 0,{start},{end},Metric,{mode},0,0,0,,"
                f"{{\\pos({center_x},{metric_y})}}{text}"
            )
    path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")


def _atomic_json_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n"
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    media_root = args.media_root.resolve()
    output_dir = args.output_dir.resolve()
    audit_path = args.audit.resolve()
    if not media_root.is_dir():
        raise MediaEvidenceError(f"media root does not exist: {media_root}")
    if not audit_path.is_file():
        raise MediaEvidenceError(f"audit does not exist: {audit_path}")
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise MediaEvidenceError(f"required executable is unavailable: {executable}")
    for font in (FONT_REGULAR, FONT_BOLD):
        if not font.is_file():
            raise MediaEvidenceError(f"required font is unavailable: {font}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "video": output_dir / f"{STEM}.mp4",
        "thumbnail": output_dir / f"{STEM}-thumbnail.png",
        "provenance": output_dir / f"{STEM}-provenance.json",
        "background": output_dir / f"{STEM}-background.png",
        "subtitles": output_dir / f"{STEM}.ass",
    }
    existing = [str(path) for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite outputs: {existing}")

    recovery_path = media_root / "recovery-files.json"
    recovery_bytes = recovery_path.read_bytes()
    recovery_sha256 = _sha256(recovery_bytes)
    recovery = _json_bytes(recovery_bytes, "recovery-files.json")
    audit_bytes = audit_path.read_bytes()
    audit_sha256 = _sha256(audit_bytes)
    audit = _json_bytes(audit_bytes, "audit")
    _validate_audit(audit)

    source_bytes: dict[str, dict[str, bytes]] = {}
    source_records: dict[str, dict[str, dict[str, Any]]] = {}
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    metadata_by_mode: dict[str, dict[str, Any]] = {}
    report_by_mode: dict[str, dict[str, Any]] = {}
    for mode in ("multi64", "zero"):
        relative_paths = {
            "video": f"fresh05-{mode}.mp4",
            "frames": f"fresh05-{mode}.mp4.frames.jsonl",
            "metadata": f"fresh05-{mode}.mp4.metadata.json",
            "evidence": f"fresh05-{mode}.mp4.evidence.json",
            "report": f"fresh05-{mode}-report.json",
        }
        source_bytes[mode] = {}
        source_records[mode] = {}
        for label, relative_path in relative_paths.items():
            data, record = _read_source_once(media_root, relative_path, recovery)
            source_bytes[mode][label] = data
            source_records[mode][label] = record
        rows_by_mode[mode] = _frame_rows(source_bytes[mode]["frames"], mode)
        metadata_by_mode[mode] = _json_bytes(
            source_bytes[mode]["metadata"], f"{mode} metadata"
        )
        report_by_mode[mode] = _json_bytes(
            source_bytes[mode]["report"], f"{mode} report"
        )

    terminal: dict[str, dict[str, float | bool | str]] = {}
    probes: dict[str, dict[str, Any]] = {}
    post_render_sha: dict[str, dict[str, str]] = {}
    ffmpeg_command: list[str]
    with tempfile.TemporaryDirectory(prefix=f".{STEM}-", dir=output_dir) as temporary_name:
        temporary_dir = Path(temporary_name)
        immutable_video: dict[str, Path] = {}
        for mode in ("multi64", "zero"):
            immutable_video[mode] = temporary_dir / f"{mode}.mp4"
            immutable_video[mode].write_bytes(source_bytes[mode]["video"])
            probes[mode] = _ffprobe(immutable_video[mode])
            evidence = _json_bytes(
                source_bytes[mode]["evidence"], f"{mode} evidence"
            )
            policy_lineage_sha256 = _validate_evidence(
                mode,
                evidence,
                metadata_by_mode[mode],
                report_by_mode[mode],
                source_records[mode],
            )
            terminal[mode] = _validate_recording(
                mode,
                metadata_by_mode[mode],
                report_by_mode[mode],
                rows_by_mode[mode],
                probes[mode],
                policy_lineage_sha256,
            )

        temporary_background = temporary_dir / output_paths["background"].name
        temporary_thumbnail = temporary_dir / output_paths["thumbnail"].name
        temporary_subtitles = temporary_dir / output_paths["subtitles"].name
        _draw_background(temporary_background, terminal)
        shutil.copyfile(temporary_background, temporary_thumbnail)
        _write_ass(temporary_subtitles, rows_by_mode)
        longest_source_s = max(len(rows) / FPS for rows in rows_by_mode.values())
        total_duration_s = longest_source_s + CARD_DURATION_S
        video_temporary = temporary_dir / f"{STEM}.mp4"
        filter_complex = (
            f"[1:v]scale={PANEL_SIZE[0]}:{PANEL_SIZE[1]}:flags=lanczos,setsar=1[left];"
            f"[2:v]scale={PANEL_SIZE[0]}:{PANEL_SIZE[1]}:flags=lanczos,setsar=1[right];"
            f"[0:v][left]overlay={PANEL_X['multi64']}:{PANEL_Y}:eof_action=pass:repeatlast=0[stage1];"
            f"[stage1][right]overlay={PANEL_X['zero']}:{PANEL_Y}:eof_action=pass:repeatlast=0[stage2];"
            f"[stage2]subtitles='{str(temporary_subtitles).replace(chr(39), chr(92) + chr(39))}'[out]"
        )
        ffmpeg_command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-t",
            f"{total_duration_s:.9f}",
            "-i",
            str(temporary_background),
            "-i",
            str(immutable_video["multi64"]),
            "-i",
            str(immutable_video["zero"]),
            "-filter_complex",
            filter_complex,
            "-map",
            "[out]",
            "-an",
            "-r",
            str(FPS),
            "-frames:v",
            str(int(round(total_duration_s * FPS))),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video_temporary),
        ]
        try:
            subprocess.run(ffmpeg_command, check=True, shell=False)
        except subprocess.CalledProcessError as error:
            raise MediaEvidenceError("ffmpeg comparison render failed") from error
        output_probe = _ffprobe(video_temporary)
        expected_output_frames = int(round(total_duration_s * FPS))
        if (
            (output_probe["width"], output_probe["height"]) != OUTPUT_SIZE
            or output_probe["frame_count"] != expected_output_frames
            or not math.isclose(output_probe["average_fps"], FPS, abs_tol=0.001)
        ):
            raise MediaEvidenceError("rendered output cadence or dimensions changed")
        for mode, records in source_records.items():
            post_render_sha[mode] = {}
            for label, record in records.items():
                observed = _file_sha256(Path(record["path"]))
                if observed != record["sha256"]:
                    raise MediaEvidenceError(
                        f"source changed during render: {mode}/{label}"
                    )
                post_render_sha[mode][label] = observed
        if _file_sha256(recovery_path) != recovery_sha256:
            raise MediaEvidenceError("recovery manifest changed during render")
        if _file_sha256(audit_path) != audit_sha256:
            raise MediaEvidenceError("audit changed during render")
        os.replace(video_temporary, output_paths["video"])
        os.replace(temporary_background, output_paths["background"])
        os.replace(temporary_thumbnail, output_paths["thumbnail"])
        os.replace(temporary_subtitles, output_paths["subtitles"])

    outputs = {}
    for label in ("video", "thumbnail", "background", "subtitles"):
        path = output_paths[label]
        outputs[label] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": _file_sha256(path),
        }
    provenance = {
        "schema_name": "rj45_multibias_comparison_media_v1",
        "status": "completed",
        "selection": {
            "case_id": "fresh-05",
            "selected_after_primary_outcomes": True,
            "purpose": "visualization contrast only",
            "training_evidence": False,
            "benchmark_statistical_evidence": False,
        },
        "audit": {
            "path": str(audit_path),
            "bytes": len(audit_bytes),
            "sha256": audit_sha256,
            "sha256_after_render": audit_sha256,
            "status": "pass",
            "fresh8_successes": {"single64": 2, "multi64": 3, "zero": 2},
            "followup_multiseed_candidate": False,
            "effect_proven": False,
        },
        "recovery_manifest": {
            "path": str(recovery_path),
            "bytes": len(recovery_bytes),
            "sha256": recovery_sha256,
            "sha256_after_render": recovery_sha256,
        },
        "sources": source_records,
        "source_sha256_after_render": post_render_sha,
        "recording_validation": {
            mode: {
                "probe": probes[mode],
                "frames_jsonl_rows": len(rows_by_mode[mode]),
                "frame_indices_contiguous": True,
                "physics_step_stride": 4,
                "metadata_frame_count_matches": True,
                "exact_frame_clock_verified": True,
                "control_groups": len(rows_by_mode[mode]) // 2,
                "control_group_offsets": [4, 8],
                "terminal": terminal[mode],
            }
            for mode in ("multi64", "zero")
        },
        "presentation": {
            "resolution": list(OUTPUT_SIZE),
            "fps": FPS,
            "source_panels": {
                mode: {
                    "x": PANEL_X[mode],
                    "y": PANEL_Y,
                    "width": PANEL_SIZE[0],
                    "height": PANEL_SIZE[1],
                    "eof_action": "pass",
                }
                for mode in ("multi64", "zero")
            },
            "metric_timing": {
                "depth_and_lateral": "60 Hz render-time frame JSONL values; row i is displayed at i/60 with no interpolation",
                "dwell": "30 Hz control-end value is adopted only at offset 8 and causally held through the next offset-4 frame; unknown before the first control end",
            },
            "source_frame_policy": "recorded frames only; no repeated final raw frame",
            "static_result_card_duration_s": CARD_DURATION_S,
            "shorter_source_after_eof": "background result card is exposed and labelled as post-recording presentation",
            "thresholds": {
                "lateral_error_um": LATERAL_LIMIT_UM,
                "stable_dwell_s": DWELL_REQUIRED_S,
                "scope": "project evaluation criteria, not manufacturer specifications",
            },
        },
        "ffmpeg_args": ffmpeg_command,
        "outputs": outputs,
        "claims": {
            "learned_bimanual_policy": False,
            "real_robot_evidence": False,
            "effect_proven": False,
            "primary_results_changed": False,
        },
    }
    _atomic_json_new(output_paths["provenance"], provenance)
    print(
        json.dumps(
            {
                "status": "completed",
                "video": str(output_paths["video"]),
                "video_sha256": outputs["video"]["sha256"],
                "thumbnail": str(output_paths["thumbnail"]),
                "provenance": str(output_paths["provenance"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
