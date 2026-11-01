from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_rj45_publication_media.py"
SPEC = importlib.util.spec_from_file_location("build_rj45_publication_media_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MEDIA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MEDIA)


def replay_report(mode: str = "multi64") -> dict:
    classification = MEDIA.EXPECTED_CLASSIFICATION[mode]
    success = MEDIA.EXPECTED_SUCCESS[mode]
    return {
        "schema_name": MEDIA.EXPECTED_REPLAY_SCHEMA,
        "status": "completed",
        "mode": mode,
        "lineage": {
            "case": {"case_id": MEDIA.EXPECTED_CASE_ID},
            "method": mode,
            "initial_base79_sha256": MEDIA.EXPECTED_INITIAL79_SHA256,
            "primary_report_sha256": MEDIA.EXPECTED_PRIMARY_REPORT_SHA256,
            "contract_sha256": "c" * 64,
            "train_report_sha256": "d" * 64,
            "checkpoint_sha256": (
                MEDIA.EXPECTED_MULTI64_CHECKPOINT_SHA256 if mode == "multi64" else None
            ),
            "classification": classification,
        },
        "comparison": {
            "initial_base79_matches": True,
            "classification_matches": True,
            "primary_classification": classification,
            "replay_classification": classification,
        },
        "claims": {name: False for name in MEDIA.FALSE_REPORT_CLAIMS},
        "replay": {
            "outcome": {
                "valid": True,
                "success": success,
                "classification": classification,
                "termination_reason": "independent_success" if success else "episode_timeout",
                "actual_depth_m": 0.012,
                "actual_lateral_error_m": 0.00003 if success else 0.00006,
                "actual_orientation_error_rad": 0.003,
            }
        },
    }


def publication_report(mode: str, report: dict) -> dict:
    report_sha = "r" * 64
    video_sha = "v" * 64
    return {
        "schema_name": MEDIA.EXPECTED_PUBLICATION_SCHEMA,
        "status": "completed",
        "app_close_exit_code": 0,
        "wrapper": {"path": "/job/render_rj45_publication.py", "sha256": "w" * 64},
        "scope": "publication visualization replay only",
        "sealed_replay_source": {
            "path": "/job/record_rj45_multibias_replay.py",
            "sha256": MEDIA.EXPECTED_SEALED_SOURCE_SHA256,
        },
        "sealed_run_arguments": [
            "--mode",
            mode,
            "--primary-report-sha256",
            MEDIA.EXPECTED_PRIMARY_REPORT_SHA256,
            "--contract-sha256",
            "c" * 64,
            "--train-report-sha256",
            "d" * 64,
            "--origin-protocol",
            "/job/origin.json",
            "--origin-protocol-sha256",
            "a" * 64,
            "--contract",
            "/job/contract.json",
            "--primary-report",
            "/job/evaluate-report.json",
            "--train-report",
            "/job/train-report.json",
            "--video",
            "/job/replay.mp4",
            "--report",
            "/job/replay-report.json",
            *(["--checkpoint", "/job/final.pt"] if mode == "multi64" else []),
            "--acknowledge-replay-only",
        ],
        "claims": {name: False for name in MEDIA.FALSE_PUBLICATION_CLAIMS},
        "sealed_report": {
            "sha256": report_sha,
            "expected_sha256": report_sha,
            "status": "completed",
            "lineage": report["lineage"],
            "comparison": report["comparison"],
        },
        "sealed_video": {"exists": True, "sha256": video_sha},
    }


def test_valid_fresh05_evidence_contract() -> None:
    report = replay_report()
    publication = publication_report("multi64", report)
    outcome = MEDIA.validate_replay_contract("multi64", report)
    MEDIA.validate_publication_contract(
        "multi64",
        publication,
        report,
        report_sha256="r" * 64,
        video_sha256="v" * 64,
        wrapper_sha256="w" * 64,
    )
    assert outcome["success"] is True


def test_relocated_native_exit_requires_same_name_and_hash(tmp_path: Path) -> None:
    report = replay_report("zero")
    publication = publication_report("zero", report)
    original = tmp_path / "remote" / "zero-exit.json"
    recovered = tmp_path / "recovered" / "zero-exit.json"
    recovered.parent.mkdir()
    recovered.write_text('{"exit_code": 0}\n')
    publication["app_close_exit_code"] = None
    publication["native_process_exit_code"] = 0
    publication["completion_witness"] = "outer_runner_native_exit"
    publication["native_exit_record"] = {
        "path": str(original),
        "sha256": MEDIA.digest(recovered),
    }
    MEDIA.validate_publication_contract(
        "zero",
        publication,
        report,
        report_sha256="r" * 64,
        video_sha256="v" * 64,
        wrapper_sha256="w" * 64,
        native_exit_path=recovered,
    )
    changed_name = recovered.with_name("other.json")
    changed_name.write_bytes(recovered.read_bytes())
    with pytest.raises(MEDIA.MediaBuildError, match="filename changed"):
        MEDIA.validate_publication_contract(
            "zero",
            publication,
            report,
            report_sha256="r" * 64,
            video_sha256="v" * 64,
            wrapper_sha256="w" * 64,
            native_exit_path=changed_name,
        )


def test_wrong_report_hash_is_rejected() -> None:
    report = replay_report()
    publication = publication_report("multi64", report)
    with pytest.raises(MEDIA.MediaBuildError, match="sealed report hash mismatch"):
        MEDIA.validate_publication_contract(
            "multi64",
            publication,
            report,
            report_sha256="x" * 64,
            video_sha256="v" * 64,
            wrapper_sha256="w" * 64,
        )


def test_wrong_mode_and_outcome_are_rejected() -> None:
    wrong_mode = replay_report()
    wrong_mode["mode"] = "zero"
    with pytest.raises(MEDIA.MediaBuildError, match="mode mismatch"):
        MEDIA.validate_replay_contract("multi64", wrong_mode)

    wrong_outcome = replay_report()
    wrong_outcome["replay"]["outcome"]["success"] = False
    with pytest.raises(MEDIA.MediaBuildError, match="success changed"):
        MEDIA.validate_replay_contract("multi64", wrong_outcome)


def test_claim_and_lineage_changes_are_rejected() -> None:
    report = replay_report("zero")
    publication = publication_report("zero", report)
    publication["claims"]["benchmark_result_replaced"] = True
    with pytest.raises(MEDIA.MediaBuildError, match="must be false"):
        MEDIA.validate_publication_contract(
            "zero",
            publication,
            report,
            report_sha256="r" * 64,
            video_sha256="v" * 64,
            wrapper_sha256="w" * 64,
        )


def test_outer_native_exit_witness_is_accepted(tmp_path: Path) -> None:
    report = replay_report("zero")
    publication = publication_report("zero", report)
    exit_record = tmp_path / "native-exit.json"
    exit_record.write_text('{"exit_code": 0}\n')
    publication["app_close_exit_code"] = None
    publication["native_process_exit_code"] = 0
    publication["completion_witness"] = "outer_runner_native_exit"
    publication["native_exit_record"] = {
        "path": str(exit_record),
        "sha256": MEDIA.digest(exit_record),
    }
    MEDIA.validate_publication_contract(
        "zero",
        publication,
        report,
        report_sha256="r" * 64,
        video_sha256="v" * 64,
        wrapper_sha256="w" * 64,
    )


def test_comparison_data_is_canonical_and_drives_counts() -> None:
    path = ROOT / "docs/evidence/rj45-multibias-20261031/comparison-data.json"
    assert MEDIA.digest(path) == MEDIA.EXPECTED_COMPARISON_DATA_SHA256
    data = json.loads(path.read_text())
    assert MEDIA.validate_comparison_data(data) == {
        "multi64": 3,
        "single64": 2,
        "zero": 2,
    }
    changed = copy.deepcopy(data)
    changed["effect_proven"] = True
    with pytest.raises(MEDIA.MediaBuildError, match="effect_proven=false"):
        MEDIA.validate_comparison_data(changed)


def test_failure_label_uses_observed_termination_reason() -> None:
    assert MEDIA.failure_label({"success": False, "termination_reason": "episode_timeout"}) == "시간 제한으로 종료"
    assert MEDIA.failure_label({"success": False, "termination_reason": "load_limit"}) == "평가 종료"


def test_composite_contract_checks_resolution_rate_and_frame_count() -> None:
    metadata = {
        "streams": [{
            "codec_type": "video",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "60/1",
            "avg_frame_rate": "60/1",
            "nb_frames": "1574",
            "duration": "26.2333333333",
        }]
    }
    assert MEDIA.validate_composite(metadata, 1574)["frames"] == 1574
    metadata["streams"][0]["nb_frames"] = "1573"
    with pytest.raises(MEDIA.MediaBuildError, match="frame count changed"):
        MEDIA.validate_composite(metadata, 1574)


def test_python_optimized_mode_still_rejects_bad_mode() -> None:
    code = f"""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('media_optimized', Path({str(SCRIPT)!r}))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    module.validate_replay_contract('multi64', {{}})
except module.MediaBuildError:
    raise SystemExit(0)
raise SystemExit(9)
"""
    result = subprocess.run(
        [sys.executable, "-O", "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("crop", [
    [-2, 240, 640, 480], [642, 240, 640, 480],
    [321, 240, 640, 480], [320, 241, 640, 480],
    [320, 240, 639, 480], [320, 240, 640, 479],
    [320, 240, 640, 400],
])
def test_invalid_crop_is_rejected_before_reading_evidence(tmp_path: Path, crop: list[int]) -> None:
    with pytest.raises(MEDIA.MediaBuildError, match="spatial crop"):
        MEDIA.main([
            "--render-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
            "--comparison-data", str(tmp_path / "missing.json"),
            "--crop", *(str(value) for value in crop),
        ])


def test_central_crop_reaches_the_evidence_gate(tmp_path: Path) -> None:
    (tmp_path / "final.json").write_text('{"status":"invalid"}')
    with pytest.raises(MEDIA.MediaBuildError, match="render batch is not completed"):
        MEDIA.main([
            "--render-root", str(tmp_path), "--output-dir", str(tmp_path / "out"),
            "--comparison-data", str(tmp_path / "missing.json"),
            "--crop", "320", "240", "640", "480",
        ])
