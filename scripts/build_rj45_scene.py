#!/usr/bin/env python3
"""Build the isolated RJ45 task assets and provenance report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rby1_assembly.rj45_scene import build_rj45_scene


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/tasks/rj45-v1.json",
        help="RJ45 task JSON (default: configs/tasks/rj45-v1.json)",
    )
    args = parser.parse_args()
    print(json.dumps(build_rj45_scene(ROOT, args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
