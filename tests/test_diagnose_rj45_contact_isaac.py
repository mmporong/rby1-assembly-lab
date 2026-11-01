from __future__ import annotations

import importlib.util
import hashlib
from pathlib import Path
import sys

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
