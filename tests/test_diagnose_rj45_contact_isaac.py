from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


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
