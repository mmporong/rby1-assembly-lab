from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import tempfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_rj45_oracle_media.py"
SPEC = importlib.util.spec_from_file_location("export_rj45_oracle_media_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row(index: int, *, success: bool = True) -> dict:
    time_s = (index + 1) / 60.0
    return {
        "frame_index": index,
        "physics_step": 4 * (index + 1),
        "physics_time_s": time_s,
        "control_time_s": index / 60.0,
        "control_step": index // 2 + 1,
        "phase_name": "insert",
        "actual_depth_m": 0.0125,
        "actual_lateral_error_m": 0.000025,
        "actual_orientation_error_rad": 0.01,
        "control_step_entry_contact_force_proxy_n": 2.0,
        "control_step_snapshot": {
            "independent_eval_dwell_s": 0.4 if success else 0.0,
            "rj45_evaluation_stability": {
                "relative_linear_speed_m_s": 0.001,
                "relative_angular_speed_rad_s": 0.01,
                "physics_all_samples_stable": success,
            },
            "terminal_evaluation_criteria": {
                "lateral": True,
                "dwell": success,
            },
        },
        "terminal_observed_after_control_step": True,
        "success_observed_after_control_step": success,
    }


def _lineage(mode: str) -> dict:
    expected = EXPORT.EXPECTED[mode]
    return {
        "mode": mode,
        "protocol": {
            "file_sha256": expected["protocol_sha256"],
            "payload_sha256": "2" * 64,
        },
        "checkpoint": (
            None
            if expected["checkpoint_sha256"] is None
            else {"path": "/remote/final_checkpoint.pt", "sha256": expected["checkpoint_sha256"]}
        ),
        "probe_outcome": {"classification": "valid_success"},
    }


def _records(mode: str, row: dict) -> tuple[dict, dict, dict, list[dict]]:
    lineage = _lineage(mode)
    classification = (
        "valid_success" if row["success_observed_after_control_step"] else "valid_failure"
    )
    lineage["probe_outcome"]["classification"] = classification
    digest = EXPORT._canonical_sha(lineage)
    row = dict(row, policy_lineage_sha256=digest)
    terminal = {
        "success": row["success_observed_after_control_step"],
        "valid": True,
        "termination_reason": "independent_success" if row["success_observed_after_control_step"] else "episode_timeout",
        "actual_depth_m": row["actual_depth_m"],
        "actual_lateral_error_m": row["actual_lateral_error_m"],
        "actual_orientation_error_rad": row["actual_orientation_error_rad"],
    }
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
        "replay": {"outcome": {**terminal, "classification": classification}},
        "outcome_comparison": {
            "probe_classification": classification,
            "replay_classification": classification,
            "classification_matches": True,
        },
    }
    metadata = {
        "policy_lineage": lineage,
        "policy_lineage_sha256": digest,
        "selected_terminal": dict(terminal),
    }
    evidence = {
        "policy_lineage": lineage,
        "policy_lineage_sha256": digest,
    }
    return report, metadata, evidence, [row]


def test_metric_units_are_derived_from_recorded_si_values() -> None:
    observed = EXPORT.derived_metrics(_row(0))
    assert observed["lateral_error_um"] == pytest.approx(25.0)
    assert observed["depth_mm"] == pytest.approx(12.5)
    assert observed["dwell_s"] == pytest.approx(0.4)
    assert observed["phase_name"] == "insert"
    assert observed["renderer_state"] == "current_physics_frame"
    assert observed["dwell_state"] == "control_step_end_snapshot"


@pytest.mark.parametrize("mode", ("zero", "standard79"))
def test_lineage_accepts_only_the_mode_specific_protocol_and_checkpoint(mode: str) -> None:
    report, metadata, evidence, rows = _records(mode, _row(0))
    assert EXPORT.validate_lineage(mode, report, metadata, evidence, rows)["mode"] == mode

    report["lineage"]["protocol"]["file_sha256"] = "f" * 64
    with pytest.raises(EXPORT.EvidenceError, match="protocol lineage"):
        EXPORT.validate_lineage(mode, report, metadata, evidence, rows)


def test_standard79_lineage_rejects_checkpoint_replacement() -> None:
    report, metadata, evidence, rows = _records("standard79", _row(0))
    replacement = dict(report["lineage"])
    replacement["checkpoint"] = {"sha256": "a" * 64}
    report["lineage"] = replacement
    with pytest.raises(EXPORT.EvidenceError, match="checkpoint"):
        EXPORT.validate_lineage("standard79", report, metadata, evidence, rows)


def test_lineage_rejects_replay_classification_that_contradicts_flags() -> None:
    report, metadata, evidence, rows = _records("zero", _row(0))
    report["replay"]["outcome"]["classification"] = "valid_failure"
    with pytest.raises(EXPORT.EvidenceError, match="contradicts"):
        EXPORT.validate_lineage("zero", report, metadata, evidence, rows)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("classification_matches", False, "must match"),
        ("probe_classification", "valid_failure", "bound to their sources"),
        ("replay_classification", "valid_failure", "bound to their sources"),
    ),
)
def test_lineage_rejects_false_or_unbound_outcome_comparison(
    field: str, value: object, message: str
) -> None:
    report, metadata, evidence, rows = _records("zero", _row(0))
    report["outcome_comparison"][field] = value
    with pytest.raises(EXPORT.EvidenceError, match=message):
        EXPORT.validate_lineage("zero", report, metadata, evidence, rows)


@pytest.mark.parametrize(
    "claim",
    (
        "training_result_reproduced",
        "benchmark_completed",
        "reinforcement_learning_effect_proven",
    ),
)
def test_lineage_rejects_promoted_fresh_replay_claims(claim: str) -> None:
    report, metadata, evidence, rows = _records("standard79", _row(0))
    report["claims"][claim] = True
    with pytest.raises(EXPORT.EvidenceError, match=claim):
        EXPORT.validate_lineage("standard79", report, metadata, evidence, rows)


def test_lineage_requires_exact_fresh_replay_only_scope() -> None:
    report, metadata, evidence, rows = _records("zero", _row(0))
    report["scope"] = "benchmark footage"
    with pytest.raises(EXPORT.EvidenceError, match="scope"):
        EXPORT.validate_lineage("zero", report, metadata, evidence, rows)


def test_terminal_must_agree_across_report_metadata_and_last_frame() -> None:
    report, metadata, _evidence, rows = _records("zero", _row(0))
    terminal = EXPORT.validate_terminal("zero", report, metadata, rows[-1])
    assert terminal["success"] is True
    report["replay"]["outcome"]["actual_lateral_error_m"] = 0.1
    with pytest.raises(EXPORT.EvidenceError, match="disagrees"):
        EXPORT.validate_terminal("zero", report, metadata, rows[-1])


def test_native_cadence_rejects_temporal_interpolation_or_hold() -> None:
    rows = [_row(index) for index in range(3)]
    EXPORT.validate_native_cadence(rows)
    rows[1]["physics_time_s"] = rows[0]["physics_time_s"]
    with pytest.raises(EXPORT.EvidenceError, match="native 60 FPS"):
        EXPORT.validate_native_cadence(rows)


def test_artifact_hash_verification_rejects_changed_frames() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        paths = {}
        artifacts = {}
        for label in ("video", "frames", "metadata"):
            path = root / label
            path.write_bytes(label.encode("ascii"))
            paths[label] = path
            artifacts[label] = {"sha256": _sha(path)}
        EXPORT.BASE.verify_artifact_hashes(paths, {"artifacts": artifacts})
        paths["frames"].write_bytes(b"changed")
        with pytest.raises(EXPORT.EvidenceError, match="frames hash"):
            EXPORT.BASE.verify_artifact_hashes(paths, {"artifacts": artifacts})
