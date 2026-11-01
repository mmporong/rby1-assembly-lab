#!/usr/bin/env python3
"""Compose the visibility retakes after validating their frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw, ImageFont


FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FPS = 60
OUTPUT_SIZE = (1920, 1080)
SOURCE_SIZE = (1280, 960)
RESULT_CARD_FRAMES = 2 * FPS
EXPECTED_REPLAY_SCHEMA = "rj45_multibias_replay_v1"
EXPECTED_PUBLICATION_SCHEMA = "rj45_publication_render_v1"
EXPECTED_CASE_ID = "fresh-05"
EXPECTED_INITIAL79_SHA256 = "606d70c5bc211a439c26fb528436035f2fcdf4a60700602c151590960948895b"
EXPECTED_PRIMARY_REPORT_SHA256 = "0decb95ec25d12419253369ef2f934f122f7f2e2a0fb346b3a03fb97d3d4da87"
EXPECTED_MULTI64_CHECKPOINT_SHA256 = "82a1a01c8a0c7797bc6b02a95852a42f82efd02b7d373fae7774ecb4c92792b3"
EXPECTED_SEALED_SOURCE_SHA256 = "49bf950a0b466820d82b5dfec1aae6bee498c8ac0b0428f24908b387d04a4e2f"
EXPECTED_COMPARISON_DATA_SHA256 = "5409a808c144611c7c097329ef71fdef55995b8b31430f9394ba02eb74a9cced"
EXPECTED_CLASSIFICATION = {"multi64": "valid_success", "zero": "valid_failure"}
EXPECTED_SUCCESS = {"multi64": True, "zero": False}
FALSE_REPORT_CLAIMS = (
    "benchmark_statistics",
    "effect_proven",
    "primary_result_changed",
    "training_evidence",
)
FALSE_PUBLICATION_CLAIMS = (
    "benchmark_result_replaced",
    "training_evidence",
    "physics_or_action_contract_modified",
)


class MediaBuildError(RuntimeError):
    """Publication media inputs do not satisfy the frozen evidence contract."""


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MediaBuildError(f"{field} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise MediaBuildError(f"{field} root must be an object")
    return value


def probe(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        value = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise MediaBuildError(f"ffprobe failed: {path}") from error
    if not isinstance(value, dict):
        raise MediaBuildError(f"ffprobe returned a non-object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MediaBuildError(message)


def _sha256(value: Any, field: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{field} must be 64 lowercase hex characters",
    )
    return value


def _false_claims(value: Any, names: Sequence[str], field: str) -> None:
    _require(isinstance(value, dict), f"{field} must be an object")
    for name in names:
        _require(value.get(name) is False, f"{field}.{name} must be false")


def _flag_value(arguments: list[str], flag: str) -> str:
    _require(all(isinstance(item, str) for item in arguments), "sealed arguments must be strings")
    _require(arguments.count(flag) == 1, f"sealed arguments must contain one {flag}")
    index = arguments.index(flag)
    _require(index + 1 < len(arguments), f"sealed argument {flag} has no value")
    return arguments[index + 1]


def _video_stream(metadata: Mapping[str, Any], field: str) -> dict[str, Any]:
    streams = metadata.get("streams")
    _require(isinstance(streams, list), f"{field} has no streams")
    video = [item for item in streams if isinstance(item, dict) and item.get("codec_type") == "video"]
    _require(len(video) == 1, f"{field} must contain exactly one video stream")
    return video[0]


def validate_source_video(metadata: Mapping[str, Any], field: str) -> dict[str, Any]:
    stream = _video_stream(metadata, field)
    _require(
        (stream.get("width"), stream.get("height")) == SOURCE_SIZE,
        f"{field} resolution must be 1280x960",
    )
    _require(stream.get("r_frame_rate") == "60/1", f"{field} must be 60 fps")
    _require(stream.get("avg_frame_rate") in (None, "60/1"), f"{field} average rate must be 60 fps")
    try:
        frames = int(stream["nb_frames"])
        duration_s = float(stream["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise MediaBuildError(f"{field} lacks frame count or duration") from error
    _require(frames > 0 and math.isfinite(duration_s), f"{field} timing is invalid")
    _require(
        math.isclose(duration_s, frames / FPS, rel_tol=0.0, abs_tol=1.0 / FPS),
        f"{field} duration and frame count disagree",
    )
    return {"frames": frames, "duration_s": duration_s}


def validate_replay_contract(mode: str, report: Mapping[str, Any]) -> dict[str, Any]:
    _require(mode in EXPECTED_CLASSIFICATION, f"unsupported mode: {mode}")
    _require(report.get("schema_name") == EXPECTED_REPLAY_SCHEMA, f"{mode} replay schema changed")
    _require(report.get("status") == "completed", f"{mode} replay is not completed")
    _require(report.get("mode") == mode, f"{mode} replay mode mismatch")
    lineage = report.get("lineage")
    _require(isinstance(lineage, dict), f"{mode} lineage is missing")
    case = lineage.get("case")
    _require(isinstance(case, dict) and case.get("case_id") == EXPECTED_CASE_ID, f"{mode} is not fresh-05")
    _require(lineage.get("method") == mode, f"{mode} lineage method mismatch")
    _require(
        lineage.get("initial_base79_sha256") == EXPECTED_INITIAL79_SHA256,
        f"{mode} initial79 lineage changed",
    )
    _require(
        lineage.get("primary_report_sha256") == EXPECTED_PRIMARY_REPORT_SHA256,
        f"{mode} primary report lineage changed",
    )
    _sha256(lineage.get("contract_sha256"), f"{mode} contract lineage")
    _sha256(lineage.get("train_report_sha256"), f"{mode} train report lineage")
    expected_checkpoint = EXPECTED_MULTI64_CHECKPOINT_SHA256 if mode == "multi64" else None
    _require(
        lineage.get("checkpoint_sha256") == expected_checkpoint,
        f"{mode} checkpoint lineage changed",
    )
    classification = EXPECTED_CLASSIFICATION[mode]
    _require(lineage.get("classification") == classification, f"{mode} lineage classification changed")
    comparison = report.get("comparison")
    _require(isinstance(comparison, dict), f"{mode} comparison is missing")
    _require(comparison.get("initial_base79_matches") is True, f"{mode} initial79 did not match")
    _require(comparison.get("classification_matches") is True, f"{mode} classification did not match")
    _require(comparison.get("replay_classification") == classification, f"{mode} replay classification changed")
    _require(comparison.get("primary_classification") == classification, f"{mode} primary classification changed")
    _false_claims(report.get("claims"), FALSE_REPORT_CLAIMS, f"{mode}.claims")
    replay = report.get("replay")
    outcome = replay.get("outcome") if isinstance(replay, dict) else None
    _require(isinstance(outcome, dict), f"{mode} outcome is missing")
    _require(outcome.get("valid") is True, f"{mode} outcome must be valid")
    _require(outcome.get("success") is EXPECTED_SUCCESS[mode], f"{mode} success changed")
    _require(outcome.get("classification") == classification, f"{mode} outcome classification changed")
    _require(isinstance(outcome.get("termination_reason"), str), f"{mode} termination reason is missing")
    for name in ("actual_depth_m", "actual_lateral_error_m", "actual_orientation_error_rad"):
        value = outcome.get(name)
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"{mode} outcome {name} is invalid",
        )
    return dict(outcome)


def validate_publication_contract(
    mode: str,
    publication: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    report_sha256: str,
    video_sha256: str,
    wrapper_sha256: str,
    native_exit_path: Path | None = None,
) -> None:
    _require(publication.get("schema_name") == EXPECTED_PUBLICATION_SCHEMA, f"{mode} publication schema changed")
    _require(publication.get("status") == "completed", f"{mode} publication replay is not completed")
    _require(
        publication.get("scope") == "publication visualization replay only",
        f"{mode} publication scope changed",
    )
    app_close_completed = publication.get("app_close_exit_code") == 0
    native_exit_completed = (
        publication.get("app_close_exit_code") is None
        and publication.get("native_process_exit_code") == 0
        and publication.get("completion_witness") == "outer_runner_native_exit"
    )
    _require(
        app_close_completed or native_exit_completed,
        f"{mode} Isaac process completion witness is invalid",
    )
    if native_exit_completed:
        native_record = publication.get("native_exit_record")
        _require(isinstance(native_record, dict), f"{mode} native exit record is missing")
        recorded_path = Path(str(native_record.get("path")))
        native_path = recorded_path if native_exit_path is None else native_exit_path.resolve()
        _require(
            native_path.name == recorded_path.name,
            f"{mode} relocated native exit filename changed",
        )
        _require(native_path.is_file(), f"{mode} native exit record file is missing")
        _require(
            native_record.get("sha256") == digest(native_path),
            f"{mode} native exit record hash mismatch",
        )
        native_exit = load_json(native_path, f"{mode} native exit record")
        _require(
            type(native_exit.get("exit_code")) is int
            and native_exit["exit_code"] == 0,
            f"{mode} native exit record is not successful",
        )
    wrapper = publication.get("wrapper")
    _require(isinstance(wrapper, dict), f"{mode} wrapper identity is missing")
    _require(wrapper.get("sha256") == wrapper_sha256, f"{mode} wrapper bytes changed")
    _require(Path(str(wrapper.get("path"))).name == "render_rj45_publication.py", f"{mode} wrapper path changed")
    source = publication.get("sealed_replay_source")
    _require(isinstance(source, dict), f"{mode} sealed replay identity is missing")
    _require(source.get("sha256") == EXPECTED_SEALED_SOURCE_SHA256, f"{mode} sealed replay source changed")
    _require(
        Path(str(source.get("path"))).name == "record_rj45_multibias_replay.py",
        f"{mode} sealed replay source path changed",
    )
    arguments = publication.get("sealed_run_arguments")
    _require(isinstance(arguments, list), f"{mode} sealed run arguments are missing")
    _require(_flag_value(arguments, "--mode") == mode, f"{mode} sealed mode argument changed")
    _require(
        _flag_value(arguments, "--primary-report-sha256") == EXPECTED_PRIMARY_REPORT_SHA256,
        f"{mode} sealed primary report argument changed",
    )
    lineage = report["lineage"]
    _require(
        _flag_value(arguments, "--contract-sha256") == lineage.get("contract_sha256"),
        f"{mode} sealed contract argument changed",
    )
    _require(
        _flag_value(arguments, "--train-report-sha256") == lineage.get("train_report_sha256"),
        f"{mode} sealed train report argument changed",
    )
    for flag in (
        "--origin-protocol",
        "--contract",
        "--primary-report",
        "--train-report",
        "--video",
        "--report",
    ):
        _require(bool(_flag_value(arguments, flag)), f"{mode} sealed {flag} path is empty")
    _sha256(_flag_value(arguments, "--origin-protocol-sha256"), f"{mode} origin protocol SHA")
    if mode == "multi64":
        _require(bool(_flag_value(arguments, "--checkpoint")), "multi64 checkpoint argument is missing")
    else:
        _require("--checkpoint" not in arguments, "zero replay must not receive a checkpoint")
    _require(arguments.count("--acknowledge-replay-only") == 1, f"{mode} replay acknowledgement is missing")
    _false_claims(publication.get("claims"), FALSE_PUBLICATION_CLAIMS, f"{mode}.publication.claims")
    sealed_report = publication.get("sealed_report")
    _require(isinstance(sealed_report, dict), f"{mode} sealed report binding is missing")
    _require(
        sealed_report.get("sha256") == report_sha256
        and sealed_report.get("expected_sha256") == report_sha256,
        f"{mode} sealed report hash mismatch",
    )
    _require(sealed_report.get("status") == report.get("status"), f"{mode} sealed report status mismatch")
    _require(sealed_report.get("lineage") == report.get("lineage"), f"{mode} sealed lineage mismatch")
    _require(
        sealed_report.get("comparison") == report.get("comparison"),
        f"{mode} sealed comparison mismatch",
    )
    sealed_video = publication.get("sealed_video")
    _require(isinstance(sealed_video, dict), f"{mode} sealed video binding is missing")
    _require(sealed_video.get("exists") is True, f"{mode} sealed video was not recorded")
    _require(sealed_video.get("sha256") == video_sha256, f"{mode} sealed video hash mismatch")


def validate_comparison_data(data: Mapping[str, Any]) -> dict[str, int]:
    _require(data.get("effect_proven") is False, "comparison data must keep effect_proven=false")
    successes = data.get("fresh8_successes")
    expected = {"multi64": 3, "single64": 2, "zero": 2}
    _require(successes == expected, "fresh8 success counts changed")
    evaluation = data.get("evaluation")
    _require(isinstance(evaluation, list) and len(evaluation) == 27, "comparison evaluation must contain 27 rows")
    for mode, expected_count in expected.items():
        fresh = [
            row
            for row in evaluation
            if isinstance(row, dict)
            and row.get("method") == mode
            and isinstance(row.get("case"), str)
            and row["case"].startswith("fresh-")
        ]
        _require(len(fresh) == 8, f"comparison data must contain eight fresh rows for {mode}")
        _require(
            sum(row.get("success") is True for row in fresh) == expected_count,
            f"comparison success count disagrees for {mode}",
        )
    return dict(successes)


def failure_label(outcome: Mapping[str, Any]) -> str:
    if outcome.get("success") is True:
        return "성공"
    return "시간 제한으로 종료" if outcome.get("termination_reason") == "episode_timeout" else "평가 종료"


def validate_composite(metadata: Mapping[str, Any], expected_frames: int) -> dict[str, Any]:
    stream = _video_stream(metadata, "composite")
    _require(
        (stream.get("width"), stream.get("height")) == OUTPUT_SIZE,
        "composite resolution must be 1920x1080",
    )
    _require(stream.get("r_frame_rate") == "60/1", "composite must be 60 fps")
    _require(stream.get("avg_frame_rate") in (None, "60/1"), "composite average rate must be 60 fps")
    try:
        frames = int(stream["nb_frames"])
        duration_s = float(stream["duration"])
    except (KeyError, TypeError, ValueError) as error:
        raise MediaBuildError("composite lacks frame count or duration") from error
    _require(frames == expected_frames, f"composite frame count changed: {frames} != {expected_frames}")
    _require(
        math.isclose(duration_s, expected_frames / FPS, rel_tol=0.0, abs_tol=1.0 / FPS),
        "composite duration disagrees with frame count",
    )
    return {"width": 1920, "height": 1080, "fps": FPS, "frames": frames, "duration_s": duration_s}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--comparison-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    args = parser.parse_args(argv)
    crop_filter = ""
    if args.crop is not None:
        x, y, width, height = args.crop
        _require(x >= 0 and y >= 0 and width >= 2 and height >= 2, "invalid spatial crop")
        _require(x + width <= 1280 and y + height <= 960, "spatial crop exceeds source image")
        _require(all(value % 2 == 0 for value in args.crop), "spatial crop must have even coordinates and dimensions")
        _require(width * 3 == height * 4, "spatial crop must preserve the 4:3 aspect ratio")
        crop_filter = f"crop={width}:{height}:{x}:{y},"
    root = args.render_root.resolve()
    out = args.output_dir.resolve()
    comparison_path = args.comparison_data.resolve()
    _require(not out.exists(), f"output directory already exists: {out}")
    final = load_json(root / "final.json", "render final report")
    _require(final.get("status") == "completed", "render batch is not completed")
    _require(digest(comparison_path) == EXPECTED_COMPARISON_DATA_SHA256, "comparison data hash changed")
    comparison = load_json(comparison_path, "comparison data")
    successes = validate_comparison_data(comparison)
    wrapper_path = Path(__file__).resolve().with_name("render_rj45_publication.py")
    wrapper_sha256 = digest(wrapper_path)

    records: dict[str, dict[str, Any]] = {}
    for mode in ("multi64", "zero"):
        folder = root / mode
        publication_path = folder / "publication-report.json"
        report_path = folder / "replay-report.json"
        video_path = folder / "replay.mp4"
        publication = load_json(publication_path, f"{mode} publication report")
        report = load_json(report_path, f"{mode} replay report")
        report_sha256 = digest(report_path)
        video_sha256 = digest(video_path)
        outcome = validate_replay_contract(mode, report)
        validate_publication_contract(
            mode,
            publication,
            report,
            report_sha256=report_sha256,
            video_sha256=video_sha256,
            wrapper_sha256=wrapper_sha256,
            native_exit_path=root / f"{mode}-exit.json",
        )
        source_timing = validate_source_video(probe(video_path), f"{mode} source video")
        records[mode] = {
            "video": str(video_path),
            "video_sha256": video_sha256,
            "report_sha256": report_sha256,
            "publication_report_sha256": digest(publication_path),
            **source_timing,
            "outcome": outcome,
        }

    out.mkdir(parents=True, exist_ok=False)
    image = Image.new("RGB", OUTPUT_SIZE, "white")
    draw = ImageDraw.Draw(image)

    def text(
        x: int,
        y: int,
        value: str,
        size: int,
        color: str = "#182232",
        bold: bool = False,
    ) -> None:
        font = ImageFont.truetype(BOLD if bold else FONT, size)
        draw.text((x, y), value, font=font, fill=color)

    text(32, 18, "양손으로 RJ45 플러그 끼우기", 46, bold=True)
    text(32, 82, "하늘색 RJ45 플러그를 흰색 LAN 어댑터에 삽입합니다. (Isaac Sim)", 29, "#526073")
    for mode, x, label in (
        ("multi64", 24, "PPO 미세 보정"),
        ("zero", 968, "학습 보정 없음"),
    ):
        result = records[mode]["outcome"]
        color = "#14755A" if result["success"] else "#9C4438"
        text(x + 10, 124, label, 30, bold=True)
        draw.rounded_rectangle((x, 176, x + 928, 872), radius=15, fill="#F1F4F7")
        text(x + 60, 350, failure_label(result), 60, color, True)
        criterion = "프로젝트 평가 기준 충족" if result["success"] else "프로젝트 평가 기준 미충족"
        text(x + 60, 450, criterion, 32)
        text(x + 60, 526, f"종료 시 횡오차 {result['actual_lateral_error_m'] * 1e6:.2f} µm", 30, "#526073")
        text(x + 60, 580, f"삽입 깊이 {result['actual_depth_m'] * 1e3:.2f} mm", 30, "#526073")
        text(x + 60, 686, "실행 종료 후 결과 카드", 25, "#687588")
    text(
        32,
        901,
        "새 오차 8조건 전체 성공: "
        f"다중 오차 PPO {successes['multi64']}/8 · "
        f"학습 보정 없음 {successes['zero']}/8 · "
        f"단일 오차 PPO {successes['single64']}/8",
        30,
        bold=True,
    )
    text(32, 956, "학습은 각 1회, 부품을 잡는 자세는 1종으로 한정했습니다. 성능 우위는 입증하지 못했습니다.", 27, "#526073")
    text(32, 1006, "연결부가 보이도록 카메라·조명·책상색을 바꿔 재생했습니다. 원평가 통계는 유지했습니다.", 25, "#526073")
    background = out / "background.png"
    image.save(background)

    expected_frames = max(record["frames"] for record in records.values()) + RESULT_CARD_FRAMES
    duration_s = expected_frames / FPS
    output = out / "rby1-rj45-visible-comparison.mp4"
    filters = (
        f"[1:v]{crop_filter}scale=928:696:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[left];"
        f"[2:v]{crop_filter}scale=928:696:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[right];"
        f"[0:v][left]overlay=24:176:eof_action=pass:repeatlast=0:enable='lt(t,{records['multi64']['frames'] / FPS:.9f})'[a];"
        f"[a][right]overlay=968:176:eof_action=pass:repeatlast=0:enable='lt(t,{records['zero']['frames'] / FPS:.9f})'[v]"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-loop",
            "1",
            "-framerate",
            str(FPS),
            "-i",
            str(background),
            "-i",
            records["multi64"]["video"],
            "-i",
            records["zero"]["video"],
            "-filter_complex",
            filters,
            "-map",
            "[v]",
            "-frames:v",
            str(expected_frames),
            "-r",
            str(FPS),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    output_metadata = probe(output)
    output_contract = validate_composite(output_metadata, expected_frames)
    for timestamp, name in (
        (min(5.0, duration_s / 2.0), "preview.png"),
        (duration_s - 0.5, "result-cards.png"),
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-nostdin",
                "-ss",
                str(timestamp),
                "-i",
                str(output),
                "-frames:v",
                "1",
                "-update",
                "1",
                str(out / name),
            ],
            check=True,
        )
    provenance = {
        "schema": "rj45_publication_composite_v2",
        "sources": records,
        "comparison_data": {
            "path": str(comparison_path),
            "sha256": digest(comparison_path),
            "fresh8_successes": successes,
            "effect_proven": False,
        },
        "builder_sha256": digest(Path(__file__).resolve()),
        "publication_wrapper_sha256": wrapper_sha256,
        "output": {
            "file": output.name,
            "sha256": digest(output),
            "bytes": output.stat().st_size,
            "contract": output_contract,
            "metadata": output_metadata,
        },
        "robot_frame_interpolation": False,
        "robot_frame_repetition": False,
        "spatial_crop_xywh": args.crop,
        "spatial_resampling": "Lanczos; source frame cadence preserved",
        "ended_panel": "static outcome card",
        "final_result_card_seconds": RESULT_CARD_FRAMES / FPS,
        "original_benchmark_modified": False,
    }
    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "duration_s": duration_s, "sha256": digest(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
