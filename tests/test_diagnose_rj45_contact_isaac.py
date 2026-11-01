from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_rj45_contact_isaac.py"
SPEC = importlib.util.spec_from_file_location("diagnose_rj45_contact_isaac", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
DIAGNOSTIC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DIAGNOSTIC
SPEC.loader.exec_module(DIAGNOSTIC)


def test_fixed_plan_is_exactly_eight_train02_development_cases() -> None:
    plan = DIAGNOSTIC.DIAGNOSTIC_PLAN
    assert len(plan) == 8
    assert all(row[0] == "diagnostic" and row[1] == "rj45-train-02" for row in plan)
    assert [row[2:4] for row in plan[:6]] == [
        ("tx-positive", "zero"),
        ("tx-negative", "zero"),
        ("tz-positive", "zero"),
        ("tz-negative", "zero"),
        ("rx-positive", "zero"),
        ("rx-negative", "zero"),
    ]
    assert [row[2:4] for row in plan[6:]] == [
        ("tz-positive", "oracle_gt_reference"),
        ("tz-negative", "oracle_gt_reference"),
    ]


def test_existing_output_is_rejected_without_resume_or_mutation(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel.txt"
    sentinel.write_text("keep\n", encoding="utf-8")
    assert DIAGNOSTIC.main(["--ordinal", "1", "--output-dir", str(output)]) == 2
    assert sorted(path.name for path in output.iterdir()) == ["sentinel.txt"]
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_module_import_does_not_import_native_isaac_modules() -> None:
    assert "isaaclab.app" not in sys.modules


class FakeBlocked(RuntimeError):
    pass


class FakeRecording:
    FORBIDDEN_CONTACT_LOAD_SOURCE = "external-only"
    INSERTION_FORCE_SOURCE = "insertion-net"
    INSERTION_LOAD_SOURCE = "insertion-load"

    class RecordingContractError(RuntimeError):
        pass

    @staticmethod
    def live_snapshot(*_args):
        return "base-live"

    @staticmethod
    def terminal_snapshot_from_extras(*_args):
        return "base-terminal"

    @staticmethod
    def _scalar(value, env_index=0):
        array = np.asarray(value)
        return array[env_index] if array.ndim else array.item()

    @staticmethod
    def _env_value(value, env_index):
        array = np.asarray(value)
        return array[env_index] if array.ndim else array

    @staticmethod
    def _phase_name(value):
        return ("pregrasp", "close", "check_lift", "align", "insert")[int(value)]

    @staticmethod
    def _metrics_from_mapping(evaluation, poses, geometry, env_index=0):
        assert poses.shape == (2, 7)
        return SimpleNamespace(
            actual_depth_m=float(evaluation["actual_depth_m"][env_index]),
            lateral_error_m=float(evaluation["lateral_error_m"][env_index]),
            orientation_error_rad=float(evaluation["orientation_error_rad"][env_index]),
        ), True


def _combined_contact_fields() -> dict:
    return {
        "forbidden_contact_load_source": DIAGNOSTIC.COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
        "forbidden_contact_load_definition": DIAGNOSTIC.COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
        "forbidden_contact_load_n": np.asarray([3.0]),
        "external_forbidden_contact_load_n": np.asarray([2.0]),
        "external_forbidden_contact_load_peak_n": np.asarray([2.5]),
        "cross_hand_contact_load_n": np.asarray([3.0]),
        "cross_hand_contact_load_peak_n": np.asarray([3.5]),
        "cross_hand_contact_load_source": DIAGNOSTIC.CROSS_HAND_CONTACT_LOAD_SOURCE,
    }


def _live_env() -> SimpleNamespace:
    pose = np.asarray([[0.0, 0.0, 0.0]])
    quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0]])
    return SimpleNamespace(
        socket=SimpleNamespace(data=SimpleNamespace(root_pos_w=pose, root_quat_w=quaternion)),
        plug=SimpleNamespace(data=SimpleNamespace(root_pos_w=pose, root_quat_w=quaternion)),
        _last_insertion_contact_force_w=np.asarray([[1.0, 2.0, 2.0]]),
        _last_insertion_contact_force_n=np.asarray([3.0]),
        _last_insertion_contact_load_n=np.asarray([4.0]),
        _last_contact_proxy_n=np.asarray([[5.0, 6.0]]),
        _phase=np.asarray([4]),
    )


def test_diagnostic_snapshot_accepts_measured_combined_forbidden_source_without_relabel() -> None:
    extras = {
        **_combined_contact_fields(),
        "insertion_contact_force_source": FakeRecording.INSERTION_FORCE_SOURCE,
        "insertion_contact_load_source": FakeRecording.INSERTION_LOAD_SOURCE,
        "evaluation": {
            "actual_depth_m": np.asarray([0.01]),
            "lateral_error_m": np.asarray([0.001]),
            "orientation_error_rad": np.asarray([0.002]),
        },
    }
    snapshot = DIAGNOSTIC._diagnostic_snapshot(
        FakeRecording, _live_env(), extras, object(), terminal=False
    )
    assert snapshot.insertion_contact_force_w.tolist() == [1.0, 2.0, 2.0]
    assert snapshot.insertion_contact_load_n == 4.0
    assert snapshot.contact_force_proxy_n.tolist() == [5.0, 6.0]
    assert extras["forbidden_contact_load_source"] == DIAGNOSTIC.COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE


def test_diagnostic_snapshot_rejects_forged_combined_forbidden_load() -> None:
    extras = {
        **_combined_contact_fields(),
        "insertion_contact_force_source": FakeRecording.INSERTION_FORCE_SOURCE,
        "insertion_contact_load_source": FakeRecording.INSERTION_LOAD_SOURCE,
        "evaluation": {},
    }
    extras["forbidden_contact_load_n"] = np.asarray([2.5])
    with pytest.raises(
        FakeRecording.RecordingContractError, match="differs from its measured components"
    ):
        DIAGNOSTIC._diagnostic_snapshot(
            FakeRecording, _live_env(), extras, object(), terminal=False
        )


def test_base_forbidden_source_remains_owned_by_frozen_recording_contract() -> None:
    extras = {"forbidden_contact_load_source": FakeRecording.FORBIDDEN_CONTACT_LOAD_SOURCE}
    assert (
        DIAGNOSTIC._diagnostic_snapshot(
            FakeRecording, _live_env(), extras, object(), terminal=False
        )
        == "base-live"
    )
    terminal = {"terminal_snapshot": extras}
    assert (
        DIAGNOSTIC._diagnostic_snapshot(
            FakeRecording, None, terminal, object(), terminal=True
        )
        == "base-terminal"
    )


def test_terminal_combined_snapshot_reads_pre_reset_clone() -> None:
    snapshot = {
        **_combined_contact_fields(),
        "insertion_contact_force_source": FakeRecording.INSERTION_FORCE_SOURCE,
        "insertion_contact_load_source": FakeRecording.INSERTION_LOAD_SOURCE,
        "insertion_contact_force_w": np.asarray([[1.0, 2.0, 2.0]]),
        "insertion_contact_force_n": np.asarray([3.0]),
        "insertion_contact_load_n": np.asarray([4.0]),
        "contact_force_proxy_n": np.asarray([[5.0, 6.0]]),
        "phase": np.asarray([4]),
        "object_poses_world_wxyz": np.asarray(
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 2]
        ),
        "evaluation": {
            "actual_depth_m": np.asarray([0.01]),
            "lateral_error_m": np.asarray([0.001]),
            "orientation_error_rad": np.asarray([0.002]),
        },
    }
    result = DIAGNOSTIC._diagnostic_snapshot(
        FakeRecording,
        SimpleNamespace(socket="poisoned-reset-state", plug="poisoned-reset-state"),
        {"terminal_snapshot": snapshot},
        object(),
        terminal=True,
    )
    assert result.phase == "insert"
    assert result.ground_truth.actual_depth_m == 0.01
    assert result.insertion_contact_force_w.tolist() == [1.0, 2.0, 2.0]


def test_runtime_start_attestation_preserves_actual_provenance_fields() -> None:
    payload = {
        "runtime_versions": {"torch": "actual"},
        "runtime_import_origins": {
            "rby1_assembly.recording": "/runtime/src/rby1_assembly/recording.py",
            "rby1_assembly.dataset": "/runtime/src/rby1_assembly/dataset.py",
        },
        "held_bank_proof": {"manifest_sha256": "a" * 64},
        "frozen_input_sha256": {f"/bank/{index}": "b" * 64 for index in range(26)},
    }
    runner = type(
        "Runner",
        (),
        {
            "AdmissionRunBlocked": FakeBlocked,
            "_provenance": staticmethod(lambda prepared, runtime: payload),
        },
    )
    actual = DIAGNOSTIC._capture_runtime_start_attestation(runner, {}, {})
    assert actual is payload
    assert actual["runtime_versions"] == {"torch": "actual"}
    assert actual["held_bank_proof"]["manifest_sha256"] == "a" * 64
    assert len(actual["frozen_input_sha256"]) == 26


def test_frozen_bank_end_hash_mismatch_is_blocked(tmp_path: Path) -> None:
    frozen = {}
    for index in range(26):
        path = tmp_path / f"bank-{index:02d}.json"
        path.write_text(f"{index}\n", encoding="utf-8")
        frozen[path] = hashlib.sha256(path.read_bytes()).hexdigest()
    runner = type("Runner", (), {"AdmissionRunBlocked": FakeBlocked})
    matched = DIAGNOSTIC._verify_frozen_bank_end(
        runner, {"frozen_hashes": frozen}
    )
    assert matched["matched"] is True
    assert matched["verified_file_count"] == 26
    assert "verified_held_grasp_bank" in matched["source"]
    changed = tmp_path / "bank-07.json"
    changed.write_text("changed\n", encoding="utf-8")
    with pytest.raises(FakeBlocked, match="changed during diagnostic"):
        DIAGNOSTIC._verify_frozen_bank_end(runner, {"frozen_hashes": frozen})


def test_runtime_import_root_mismatch_is_blocked(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    start = {
        "runtime_import_origins": {
            "rby1_assembly.recording": str(
                runtime_root / "src/rby1_assembly/recording.py"
            )
        }
    }
    runner = type(
        "Runner",
        (),
        {
            "AdmissionRunBlocked": FakeBlocked,
            "_assert_runtime_import_origins": staticmethod(
                lambda _root: {"rby1_assembly.recording": "/tmp/escaped/recording.py"}
            ),
        },
    )
    prepared = {
        "runtime_root": runtime_root,
        "physics_input_snapshot": {"normalized_sha256": {"runtime": {}}},
    }
    with pytest.raises(FakeBlocked, match="escaped"):
        DIAGNOSTIC._verify_runtime_import_origins_end(runner, prepared, start)
