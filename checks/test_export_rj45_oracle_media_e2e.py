from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest


UNIT_PATH = (
    Path(__file__).resolve().parents[1]
    / "tests/test_export_rj45_oracle_media.py"
)
SPEC = importlib.util.spec_from_file_location("oracle_media_unit_fixtures", UNIT_PATH)
assert SPEC is not None and SPEC.loader is not None
UNIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(UNIT)
EXPORT = UNIT.EXPORT


def test_noto_cjk_outcome_heading_fits_the_result_panel() -> None:
    for mode in ("zero", "standard79"):
        heading, _reason, _color = EXPORT._outcome(
            mode, {"success": True, "termination_reason": "independent_success"}
        )
        assert EXPORT._font(21, True).getlength(heading) < 400


def test_synthetic_native_video_exports_one_to_one_with_manifest_artifacts(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    media_dir = tmp_path / "media"
    evidence_dir = tmp_path / "evidence"
    input_dir.mkdir()
    media_dir.mkdir()
    evidence_dir.mkdir()
    mode = "zero"
    paths = EXPORT.input_paths(input_dir, mode)
    lineage = UNIT._lineage(mode)
    lineage_sha = EXPORT._canonical_sha(lineage)
    rows = []
    for index in range(3):
        row = UNIT._row(index)
        row["terminal_observed_after_control_step"] = index == 2
        row["success_observed_after_control_step"] = index == 2
        row["policy_lineage_sha256"] = lineage_sha
        rows.append(row)
    paths["frames"].write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    raw = tmp_path / "frames.rgb"
    with raw.open("wb") as target:
        for index in range(3):
            frame = np.full((960, 1280, 3), 40 + index * 40, dtype=np.uint8)
            target.write(frame.tobytes())
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "1280x960",
            "-r", "60", "-i", str(raw), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", str(paths["video"]),
        ],
        check=True,
    )
    terminal = {
        "success": True,
        "valid": True,
        "termination_reason": "independent_success",
        "actual_depth_m": rows[-1]["actual_depth_m"],
        "actual_lateral_error_m": rows[-1]["actual_lateral_error_m"],
        "actual_orientation_error_rad": rows[-1]["actual_orientation_error_rad"],
    }
    metadata = {
        "completed": True,
        "rollout_completed": True,
        "captured_frames": 3,
        "policy_lineage": lineage,
        "policy_lineage_sha256": lineage_sha,
        "selected_terminal": terminal,
    }
    paths["metadata"].write_text(json.dumps(metadata), encoding="utf-8")
    evidence = {
        "completed": True,
        "rollout_completed": True,
        "policy_lineage": lineage,
        "policy_lineage_sha256": lineage_sha,
        "artifacts": {
            label: {"sha256": UNIT._sha(paths[label])}
            for label in ("video", "frames", "metadata")
        },
    }
    paths["evidence"].write_text(json.dumps(evidence), encoding="utf-8")
    report = {
        "status": "completed",
        "mode": mode,
        "scope": EXPORT.FRESH_REPLAY_SCOPE,
        "claims": {
            "fresh_replay_filmed": True,
            "training_result_reproduced": False,
            "benchmark_completed": False,
            "reinforcement_learning_effect_proven": False,
        },
        "lineage": lineage,
        "replay": {"outcome": {**terminal, "classification": "valid_success"}},
        "outcome_comparison": {
            "probe_classification": "valid_success",
            "replay_classification": "valid_success",
            "classification_matches": True,
        },
    }
    paths["report"].write_text(json.dumps(report), encoding="utf-8")

    record = EXPORT.export_mode(mode, input_dir, media_dir, evidence_dir)
    assert record["output"]["width"] == 1600
    assert record["output"]["height"] == 900
    assert record["output"]["frame_count"] == 3
    assert record["output"]["fps"] == pytest.approx(60.0)
    assert record["source_probe"]["width"] == 1280
    assert record["source_probe"]["height"] == 960
    assert record["presentation_contract"]["source_frame_to_output_frame"] == "1:1"
    assert record["presentation_contract"]["temporal_interpolation"] is False
    preview = cv2.imread(str(media_dir / record["final_preview"]["path"]))
    assert preview.shape[:2] == (900, 1600)
