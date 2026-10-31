#!/usr/bin/env python3
"""Export native-frame diagnostics for the zero and Standard79 fresh replays."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


OUTPUT_SIZE = (1600, 900)
SOURCE_SIZE = (1280, 960)
VIEW_SIZE = (1040, 780)
FPS = 60.0
PHYSICS_STEP_STRIDE = 4
LATERAL_LIMIT_UM = 50.0
DWELL_REQUIRED_S = 0.4
EXPECTED = {
    "zero": {
        "protocol_sha256": "5c6dff5a338994ae67cd64fb36ee702656371b9fbfa392e0edf66053741bd0d0",
        "checkpoint_sha256": None,
        "label": "Zero residual 기준",
    },
    "standard79": {
        "protocol_sha256": "a1a00452c8c7a6151844e2b3bd56d9322a2baaa6f1c4b7e9c2bb925be07959ef",
        "checkpoint_sha256": "935496bd8f6e8c0e53c151cbb497f7a0bcfb9768dd18e65de3874c6841583c02",
        "label": "Standard79 PPO 최종 체크포인트",
    },
}
FRESH_REPLAY_SCOPE = "fresh physics replay filming; not training or benchmark evidence"


def _load_base() -> Any:
    path = Path(__file__).resolve().with_name("export_rj45_policy_media.py")
    spec = importlib.util.spec_from_file_location("rj45_policy_media_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import media helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = _load_base()
EvidenceError = BASE.EvidenceError


@lru_cache(maxsize=None)
def _font(size: int, bold: bool = False) -> Any:
    return BASE._font(size, bold)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return value


def _canonical_sha(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _close_number(left: Any, right: Any, name: str) -> None:
    first = BASE._finite(left, name)
    second = BASE._finite(right, name)
    if not math.isclose(first, second, rel_tol=0.0, abs_tol=1.0e-9):
        raise EvidenceError(f"{name} disagrees across terminal records")


def derived_metrics(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(BASE.metrics(row))
    phase = row.get("phase_name")
    if not isinstance(phase, str) or not phase:
        raise EvidenceError("frame row is missing phase_name")
    result["phase_name"] = phase
    result["physics_step"] = int(row["physics_step"])
    result["renderer_state"] = "current_physics_frame"
    result["dwell_state"] = "control_step_end_snapshot"
    return result


def validate_native_cadence(rows: list[dict[str, Any]]) -> None:
    times = [BASE._finite(row.get("physics_time_s"), "physics_time_s") for row in rows]
    steps = [int(row.get("physics_step", -1)) for row in rows]
    for index, (before, after) in enumerate(zip(times, times[1:]), start=1):
        if not math.isclose(after - before, 1.0 / FPS, rel_tol=0.0, abs_tol=2.0e-6):
            raise EvidenceError(f"frame {index} does not preserve native 60 FPS cadence")
    if any(after - before != PHYSICS_STEP_STRIDE for before, after in zip(steps, steps[1:])):
        raise EvidenceError("physics-step cadence is not four 240 Hz steps per frame")


def validate_lineage(
    mode: str,
    report: Mapping[str, Any],
    metadata: Mapping[str, Any],
    evidence: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = EXPECTED[mode]
    claims = report.get("claims")
    if report.get("status") != "completed" or report.get("mode") != mode:
        raise EvidenceError("replay report is not a completed fresh replay for this mode")
    if report.get("scope") != FRESH_REPLAY_SCOPE:
        raise EvidenceError("replay report scope is not fresh-replay-only")
    if not isinstance(claims, dict) or claims.get("fresh_replay_filmed") is not True:
        raise EvidenceError("replay report does not claim a fresh filmed replay")
    for claim in (
        "training_result_reproduced",
        "benchmark_completed",
        "reinforcement_learning_effect_proven",
    ):
        if claims.get(claim) is not False:
            raise EvidenceError(f"fresh replay report must keep {claim}=false")
    lineage = report.get("lineage")
    if not isinstance(lineage, dict) or lineage.get("mode") != mode:
        raise EvidenceError("replay report lineage mode mismatch")
    protocol = lineage.get("protocol")
    if (
        not isinstance(protocol, dict)
        or protocol.get("file_sha256") != expected["protocol_sha256"]
    ):
        raise EvidenceError("replay protocol lineage mismatch")
    checkpoint = lineage.get("checkpoint")
    if expected["checkpoint_sha256"] is None:
        if checkpoint is not None:
            raise EvidenceError("zero replay must not bind a checkpoint")
    elif (
        not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != expected["checkpoint_sha256"]
    ):
        raise EvidenceError("Standard79 replay checkpoint lineage mismatch")
    if metadata.get("policy_lineage") != lineage or evidence.get("policy_lineage") != lineage:
        raise EvidenceError("report, metadata, and evidence lineage differ")
    lineage_sha = _canonical_sha(lineage)
    if metadata.get("policy_lineage_sha256") != lineage_sha:
        raise EvidenceError("metadata lineage digest mismatch")
    if evidence.get("policy_lineage_sha256") != lineage_sha:
        raise EvidenceError("evidence lineage digest mismatch")
    if any(row.get("policy_lineage_sha256") != lineage_sha for row in rows):
        raise EvidenceError("frame lineage digest mismatch")
    replay_outcome = report.get("replay", {}).get("outcome")
    comparison = report.get("outcome_comparison")
    probe_outcome = lineage.get("probe_outcome")
    if not isinstance(replay_outcome, dict) or not isinstance(comparison, dict):
        raise EvidenceError("replay outcome comparison is missing")
    if not isinstance(probe_outcome, dict):
        raise EvidenceError("probe outcome lineage is missing")
    valid = replay_outcome.get("valid")
    success = replay_outcome.get("success")
    if type(valid) is not bool or type(success) is not bool:
        raise EvidenceError("replay outcome flags must be boolean")
    actual_replay_classification = (
        "valid_success" if valid and success else "valid_failure" if valid else "invalid"
    )
    if replay_outcome.get("classification") != actual_replay_classification:
        raise EvidenceError("replay classification contradicts valid/success flags")
    probe_classification = probe_outcome.get("classification")
    if probe_classification not in {"valid_success", "valid_failure"}:
        raise EvidenceError("probe classification is not a valid terminal classification")
    if (
        comparison.get("probe_classification") != probe_classification
        or comparison.get("replay_classification") != actual_replay_classification
    ):
        raise EvidenceError("outcome comparison classifications are not bound to their sources")
    if comparison.get("classification_matches") is not True:
        raise EvidenceError("selected replay classification must match the probe outcome")
    if probe_classification != actual_replay_classification:
        raise EvidenceError("probe and replay classifications do not actually match")
    return lineage


def validate_terminal(
    mode: str,
    report: Mapping[str, Any],
    metadata: Mapping[str, Any],
    final_row: Mapping[str, Any],
) -> dict[str, Any]:
    terminal = metadata.get("selected_terminal")
    replay_outcome = report.get("replay", {}).get("outcome")
    if not isinstance(terminal, dict) or not isinstance(replay_outcome, dict):
        raise EvidenceError("terminal outcome is missing")
    final = derived_metrics(dict(final_row))
    if final["terminal"] is not True:
        raise EvidenceError("last native frame is not terminal")
    for record in (terminal, replay_outcome):
        if type(record.get("success")) is not bool or type(record.get("valid")) is not bool:
            raise EvidenceError("terminal success/valid flags must be boolean")
        if record["success"] != final["success"]:
            raise EvidenceError("terminal success disagrees with last frame")
    if terminal["valid"] is not True or replay_outcome["valid"] is not True:
        raise EvidenceError("selected fresh replay terminal must be valid")
    if terminal.get("termination_reason") != replay_outcome.get("termination_reason"):
        raise EvidenceError("terminal reason disagrees between report and metadata")
    for field in (
        "actual_depth_m",
        "actual_lateral_error_m",
        "actual_orientation_error_rad",
    ):
        _close_number(terminal.get(field), replay_outcome.get(field), field)
        _close_number(terminal.get(field), final_row.get(field), field)
    return {
        "mode": mode,
        "success": final["success"],
        "valid": True,
        "termination_reason": terminal["termination_reason"],
        "actual_depth_m": terminal["actual_depth_m"],
        "actual_lateral_error_m": terminal["actual_lateral_error_m"],
        "actual_orientation_error_rad": terminal["actual_orientation_error_rad"],
    }


def input_paths(root: Path, mode: str) -> dict[str, Path]:
    video = root / f"{mode}.mp4"
    return {
        "video": video,
        "frames": root / f"{mode}.mp4.frames.jsonl",
        "metadata": root / f"{mode}.mp4.metadata.json",
        "evidence": root / f"{mode}.mp4.evidence.json",
        "report": root / f"{mode}-report.json",
    }


def validate_inputs(
    mode: str, paths: dict[str, Path]
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    for label, path in paths.items():
        if not path.is_file():
            raise EvidenceError(f"missing {label}: {path}")
    metadata = _load_json(paths["metadata"])
    evidence = _load_json(paths["evidence"])
    report = _load_json(paths["report"])
    rows = BASE.load_rows(paths["frames"])
    probe = BASE.probe_video(paths["video"])
    if metadata.get("completed") is not True or metadata.get("rollout_completed") is not True:
        raise EvidenceError("recording metadata is incomplete")
    if evidence.get("completed") is not True or evidence.get("rollout_completed") is not True:
        raise EvidenceError("recording evidence is incomplete")
    expected_count = int(metadata.get("captured_frames", -1))
    if expected_count != len(rows) or expected_count != probe["frame_count"]:
        raise EvidenceError("video, metadata, and frame row counts disagree")
    if not math.isclose(probe["fps"], FPS, rel_tol=0.0, abs_tol=0.01):
        raise EvidenceError("source video is not native 60 FPS")
    if (probe["width"], probe["height"]) != SOURCE_SIZE:
        raise EvidenceError("source video must preserve the full 1280x960 replay view")
    validate_native_cadence(rows)
    BASE.verify_artifact_hashes(paths, evidence)
    lineage = validate_lineage(mode, report, metadata, evidence, rows)
    terminal = validate_terminal(mode, report, metadata, rows[-1])
    return metadata, rows, {
        "report": report,
        "evidence": evidence,
        "probe": probe,
        "lineage": lineage,
        "terminal": terminal,
    }


def _gauge(draw: Any, box: tuple[int, int, int, int], value: float, limit: float) -> None:
    x0, y0, x1, y1 = box
    passed = value <= limit
    color = "#067647" if passed else "#d92d20"
    draw.text((x0, y0), "현재 횡오차", font=_font(19, True), fill="#101828")
    draw.text((x1 - 190, y0), f"{value:.1f} µm", font=_font(19, True), fill=color)
    bar_y = y0 + 36
    draw.rounded_rectangle((x0, bar_y, x1, bar_y + 16), radius=8, fill="#eaecf0")
    ratio = min(1.0, value / max(2.0 * limit, 1.0e-12))
    draw.rounded_rectangle(
        (x0, bar_y, x0 + max(4, int((x1 - x0) * ratio)), bar_y + 16),
        radius=8,
        fill=color,
    )
    threshold = x0 + (x1 - x0) // 2
    draw.line((threshold, bar_y - 4, threshold, bar_y + 20), fill="#344054", width=2)
    draw.text(
        (x0, bar_y + 24),
        "프로젝트 판정 기준 ≤ 50 µm",
        font=_font(14),
        fill="#667085",
    )


def _outcome(mode: str, terminal: Mapping[str, Any]) -> tuple[str, str, str]:
    label = "Zero residual" if mode == "zero" else "표준 PPO"
    if terminal["success"]:
        return f"{label} · 성공", str(terminal["termination_reason"]), "#067647"
    return f"{label} · 실패", str(terminal["termination_reason"]), "#b42318"


def compose_frame(
    frame: np.ndarray,
    row: dict[str, Any],
    mode: str,
    terminal: Mapping[str, Any],
) -> np.ndarray:
    import cv2
    from PIL import Image, ImageDraw

    metric = derived_metrics(row)
    canvas = Image.new("RGB", OUTPUT_SIZE, "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    full_view = cv2.resize(rgb, VIEW_SIZE, interpolation=cv2.INTER_AREA)
    canvas.paste(Image.fromarray(full_view), (20, 100))
    draw.text((24, 19), EXPECTED[mode]["label"], font=_font(29, True), fill="#101828")
    draw.text(
        (560, 30),
        "신규 물리 재생 · 학습 장면/벤치마크 영상 아님",
        font=_font(17, True),
        fill="#475467",
    )
    draw.text(
        (24, 76),
        "왼쪽: 현재 240 Hz 물리 상태에서 기록한 원본 60 FPS 프레임 전체",
        font=_font(14),
        fill="#475467",
    )
    draw.rounded_rectangle((1080, 100, 1580, 880), radius=16, fill="white", outline="#d0d5dd", width=2)
    draw.text((1105, 125), f"물리 시간  {metric['physics_time_s']:.3f} s", font=_font(20, True), fill="#101828")
    draw.text((1105, 166), f"현재 단계  {metric['phase_name']}", font=_font(18, True), fill="#344054")
    draw.text((1105, 202), f"제어 스텝  #{metric['control_step']}", font=_font(15), fill="#667085")
    _gauge(draw, (1105, 258, 1555, 330), float(metric["lateral_error_um"]), LATERAL_LIMIT_UM)
    draw.text((1105, 376), "안정 유지 시간", font=_font(18, True), fill="#101828")
    dwell_color = "#067647" if float(metric["dwell_s"]) >= DWELL_REQUIRED_S else "#b42318"
    draw.text((1370, 376), f"{metric['dwell_s']:.3f} / 0.4 s", font=_font(18, True), fill=dwell_color)
    draw.text((1105, 420), f"삽입 깊이  {metric['depth_mm']:.3f} mm", font=_font(18, True), fill="#344054")
    draw.rounded_rectangle((1100, 476, 1560, 590), radius=10, fill="#f2f4f7")
    draw.text((1120, 492), "측정 시점 구분", font=_font(16, True), fill="#101828")
    draw.text((1120, 526), "화면·깊이·횡오차: 현재 물리 프레임", font=_font(14), fill="#475467")
    draw.text((1120, 554), "안정 유지: 해당 제어 스텝 종료 스냅샷", font=_font(14), fill="#475467")
    heading, reason, color = _outcome(mode, terminal)
    fill = "#ecfdf3" if color == "#067647" else "#fef3f2"
    draw.rounded_rectangle((1100, 628, 1560, 792), radius=12, fill=fill, outline=color, width=2)
    draw.text((1120, 646), "최종 결과 · 전 구간 고정 표시", font=_font(14), fill="#667085")
    draw.text((1120, 682), heading, font=_font(21, True), fill=color)
    draw.text((1120, 724), f"종료 사유  {reason}", font=_font(14), fill="#344054")
    draw.text(
        (1120, 754),
        f"최종 횡오차 {1e6 * float(terminal['actual_lateral_error_m']):.1f} µm",
        font=_font(14),
        fill="#344054",
    )
    draw.text(
        (1105, 826),
        "시간 보간·프레임 홀드·공간 크롭 없음 · 입력 1프레임당 출력 1프레임",
        font=_font(12),
        fill="#667085",
    )
    return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)


CSV_FIELDS = list(BASE.CSV_FIELDS) + [
    "phase_name",
    "physics_step",
    "renderer_state",
    "dwell_state",
]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in CSV_FIELDS})


def export_video(
    source: Path,
    output: Path,
    rows: list[dict[str, Any]],
    mode: str,
    terminal: Mapping[str, Any],
) -> Path:
    import cv2

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise EvidenceError(f"cannot open source video: {source}")
    writer = BASE.ffmpeg_writer(output)
    if writer.stdin is None:
        raise RuntimeError("ffmpeg stdin unavailable")
    final_frame: np.ndarray | None = None
    count = 0
    try:
        for row in rows:
            ok, frame = capture.read()
            if not ok:
                raise EvidenceError(f"source video ended before frame {count}")
            composed = compose_frame(frame, row, mode, terminal)
            writer.stdin.write(composed.tobytes())
            final_frame = composed
            count += 1
        if capture.read()[0]:
            raise EvidenceError("source video has frames absent from frame JSONL")
    finally:
        capture.release()
        writer.stdin.close()
    if writer.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    if count != len(rows) or final_frame is None:
        raise EvidenceError("export did not consume every native frame")
    preview = output.with_name(output.stem + "-final-preview.png")
    if not cv2.imwrite(str(preview), final_frame):
        raise RuntimeError("could not write final preview")
    return preview


def export_mode(
    mode: str, input_dir: Path, media_dir: Path, evidence_dir: Path
) -> dict[str, Any]:
    paths = input_paths(input_dir, mode)
    metadata, rows, details = validate_inputs(mode, paths)
    derived = [derived_metrics(row) for row in rows]
    output = media_dir / f"{mode}-fresh-replay-diagnostic-60fps.mp4"
    preview = media_dir / f"{mode}-fresh-replay-diagnostic-60fps-final-preview.png"
    csv_path = evidence_dir / f"{mode}-fresh-replay-timeseries.csv"
    for path in (output, preview, csv_path):
        if path.exists():
            raise FileExistsError(f"output already exists: {path}")
    observed_preview = export_video(
        paths["video"], output, rows, mode, details["terminal"]
    )
    if observed_preview != preview:
        raise RuntimeError("preview path contract changed")
    write_csv(csv_path, derived)
    output_probe = BASE.probe_video(output)
    if (
        output_probe["width"] != OUTPUT_SIZE[0]
        or output_probe["height"] != OUTPUT_SIZE[1]
        or output_probe["frame_count"] != len(rows)
        or not math.isclose(output_probe["fps"], FPS, rel_tol=0.0, abs_tol=0.01)
    ):
        raise EvidenceError("exported MP4 dimensions, cadence, or frame count changed")
    final = derived[-1]
    return {
        "mode": mode,
        "input": {
            label: {"path": path.name, "sha256": BASE.sha256(path), "bytes": path.stat().st_size}
            for label, path in paths.items()
        },
        "lineage": details["lineage"],
        "terminal": details["terminal"],
        "source_probe": details["probe"],
        "output": {
            "path": output.name,
            "sha256": BASE.sha256(output),
            "bytes": output.stat().st_size,
            **output_probe,
        },
        "final_preview": {
            "path": preview.name,
            "sha256": BASE.sha256(preview),
            "bytes": preview.stat().st_size,
        },
        "metrics_csv": {
            "path": csv_path.name,
            "sha256": BASE.sha256(csv_path),
            "rows": len(derived),
        },
        "final_metrics": {key: final[key] for key in CSV_FIELDS},
        "presentation_contract": {
            "source_frame_to_output_frame": "1:1",
            "temporal_interpolation": False,
            "added_frame_hold": False,
            "spatial_crop": False,
            "source_frame_whole_view": True,
            "render_measurements": "current physics frame",
            "dwell_measurement": "control-step end snapshot",
            "training_footage": False,
            "benchmark_footage": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--media-dir", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.input_dir = args.input_dir.resolve()
    args.media_dir = args.media_dir.resolve()
    args.evidence_dir = args.evidence_dir.resolve()
    args.media_dir.mkdir(parents=True, exist_ok=True)
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.evidence_dir / "oracle-fresh-replay-media-manifest.json"
    if manifest_path.exists():
        parser.error(f"manifest already exists: {manifest_path}")
    runs = [
        export_mode(mode, args.input_dir, args.media_dir, args.evidence_dir)
        for mode in ("zero", "standard79")
    ]
    manifest = {
        "schema_version": 1,
        "scope": "fresh replay diagnostics; not training or benchmark footage",
        "thresholds": {
            "lateral_error_um": LATERAL_LIMIT_UM,
            "stable_dwell_s": DWELL_REQUIRED_S,
            "lateral_threshold_scope": "project evaluation criterion",
        },
        "runs": runs,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    result = {
        "manifest": str(manifest_path),
        "manifest_sha256": BASE.sha256(manifest_path),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
