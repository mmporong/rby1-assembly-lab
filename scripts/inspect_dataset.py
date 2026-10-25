#!/usr/bin/env python3
"""RB-Y1 assembly HDF5 dataset의 구성과 무결성을 JSON으로 출력한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.dataset import DatasetSplit, inspect_dataset, select_episode_ids  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--split", choices=[value.value for value in DatasetSplit])
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--training-eligible-only", action="store_true")
    parser.add_argument("--skip-hash-verification", action="store_true")
    args = parser.parse_args()

    report = inspect_dataset(args.dataset, verify_hashes=not args.skip_hash_verification)
    split = DatasetSplit(args.split) if args.split else None
    report["selected_episode_ids"] = select_episode_ids(
        args.dataset,
        split=split,
        success_only=args.success_only,
        training_eligible_only=args.training_eligible_only,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
