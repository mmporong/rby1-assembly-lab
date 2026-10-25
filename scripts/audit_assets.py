#!/usr/bin/env python3
"""다운로드된 USD의 좌표계·관절·충돌·질량·의존성을 감사한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rby1_assembly.assets import (  # noqa: E402
    AssetError,
    audit_automate_data,
    audit_generated_assets,
    audit_manifest_assets,
    load_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "configs" / "assets.json"
    )
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "assets" / "cache")
    parser.add_argument("--asset", action="append", dest="asset_ids", help="manifest asset id")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    parser.add_argument(
        "--generated",
        action="store_true",
        help="also audit assets/generated and use the patched scene as the final gate",
    )
    args = parser.parse_args()

    try:
        manifest = load_manifest(args.manifest)
        reports = audit_manifest_assets(
            manifest, args.cache_root, args.asset_ids
        )
    except AssetError as exc:
        print(f"asset audit failed: {exc}", file=sys.stderr)
        return 1
    automate = audit_automate_data(args.cache_root)
    generated = audit_generated_assets(REPO_ROOT, manifest) if args.generated else None
    expected_raw_failures = {
        "rb_gripper_left.usd": ("end-effector asset has no collision prims",),
        "rb_gripper_right.usd": ("end-effector asset has no collision prims",),
    }
    raw_gate_passed = all(
        report.passed or expected_raw_failures.get(Path(report.path).name) == report.errors
        for report in reports
    )
    gate_passed = (
        raw_gate_passed and automate.passed and
        (generated is None or generated.passed)
    )
    payload = json.dumps(
        {
            "raw_usd": [report.to_dict() for report in reports],
            "expected_raw_failures": expected_raw_failures,
            "raw_gate_passed": raw_gate_passed,
            "generated": generated.to_dict() if generated else None,
            "automate_data": automate.to_dict(),
            "gate_passed": gate_passed,
        },
        ensure_ascii=False,
        indent=2,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if gate_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
