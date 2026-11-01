from __future__ import annotations

import importlib.util
import math
from fractions import Fraction
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/rj45_publication_alignment_overlay.py"
SOURCE_ROOT = Path(
    "/opt/homedata/rby1-storage/recovered/rj45-publication-visibility-20261101e/job"
)


def _module():
    spec = importlib.util.spec_from_file_location("alignment_overlay", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records() -> dict:
    return {"multi64": {"frames": 760, "outcome": {}}, "zero": {"frames": 1454, "outcome": {}}}


def test_build_masks_initial_state_and_keeps_causal_first_half(tmp_path: Path) -> None:
    overlay = _module()
    output = tmp_path / "alignment.ass"
    provenance = overlay.build_alignment_overlay(SOURCE_ROOT, _records(), output)
    ass = output.read_text(encoding="utf-8")

    assert "PlayResX: 1920" in ass
    assert "{\\pos(44,858)}초기 자세 준비" in ass
    assert "200002.1 µm" not in ass
    assert "실시간 0.02 s · 1×" in ass
    assert provenance["display_contract"]["overall_success_not_declared"] is True
    assert provenance["traces"]["multi64"]["frames"] == 760
    assert provenance["display"]["multi64"]["invalid_policy_input_frames"] > 0


def test_rejects_raw_trace_hash_corruption(tmp_path: Path) -> None:
    overlay = _module()
    for name in ("recovery-files.json",):
        (tmp_path / name).symlink_to(SOURCE_ROOT / name)
    for mode in ("multi64", "zero"):
        mode_dir = tmp_path / mode
        mode_dir.mkdir()
        (mode_dir / "replay.mp4.evidence.json").symlink_to(
            SOURCE_ROOT / mode / "replay.mp4.evidence.json"
        )
        (mode_dir / "replay.mp4.frames.jsonl").symlink_to(
            SOURCE_ROOT / mode / "replay.mp4.frames.jsonl"
        )
    bad_trace = tmp_path / "multi64" / "replay.mp4.frames.jsonl"
    bad_trace.unlink()
    bad_trace.write_bytes(
        (SOURCE_ROOT / "multi64" / "replay.mp4.frames.jsonl").read_bytes() + b"\n"
    )

    with pytest.raises(overlay.AlignmentOverlayError, match="recovery manifest"):
        overlay.build_alignment_overlay(tmp_path, _records(), tmp_path / "bad.ass")


def test_causal_dwell_never_uses_first_half_snapshot_early() -> None:
    overlay = _module()
    helper, _, _ = overlay._load_frozen_helper()
    data = (SOURCE_ROOT / "multi64" / "replay.mp4.frames.jsonl").read_bytes()
    rows = helper._frame_rows(data, "multi64")
    displayed = helper._causal_dwell(rows, "multi64")

    assert displayed[0] is None
    assert displayed[1] == rows[1]["control_step_snapshot"]["independent_eval_dwell_s"]
    assert displayed[2] == displayed[1]
    assert displayed[2] == rows[3]["control_step_snapshot"]["independent_eval_dwell_s"]


def test_ass_centisecond_intervals_partition_every_60_fps_pts() -> None:
    overlay = _module()

    def as_centiseconds(value: str) -> int:
        hours, minutes, seconds = value.split(":")
        whole_seconds, fraction = seconds.split(".")
        assert len(fraction) == 2
        return (
            int(hours) * 360_000
            + int(minutes) * 6_000
            + int(whole_seconds) * 100
            + int(fraction)
        )

    intervals = [
        (
            as_centiseconds(overlay._ass_time(index / overlay.FPS)),
            as_centiseconds(overlay._ass_time((index + 1) / overlay.FPS)),
        )
        for index in range(1454)
    ]
    assert intervals[0][0] == 0
    assert all(end == next_start for (_, end), (next_start, _) in zip(intervals, intervals[1:]))
    assert all(start < end for start, end in intervals)

    for index, (start, end) in enumerate(intervals):
        pts_centiseconds = Fraction(index * 100, overlay.FPS)
        assert start <= pts_centiseconds < end
        active = sum(
            candidate_start <= pts_centiseconds < candidate_end
            for candidate_start, candidate_end in intervals[max(0, index - 1) : index + 2]
        )
        assert active == 1

    minimum_deadline = 1454 / overlay.FPS + overlay.TERMINAL_CARD_S
    quantized_deadline = math.ceil(minimum_deadline * 100.0 - 1.0e-9) / 100.0
    assert quantized_deadline >= minimum_deadline
