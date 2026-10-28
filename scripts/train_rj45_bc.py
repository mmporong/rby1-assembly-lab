#!/usr/bin/env python3
"""Train the CPU RJ45 behavior-cloning actor from sealed train/dev HDF5 inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rby1_assembly.rj45_bc_training import train_bc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", required=True, type=Path)
    parser.add_argument("--train-sha256", required=True)
    parser.add_argument("--dev-dataset", required=True, type=Path)
    parser.add_argument("--dev-sha256", required=True)
    parser.add_argument("--recipe", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--observability-diagnostic", action="store_true")
    args = parser.parse_args(argv)
    train_bc(
        train_dataset=args.train_dataset,
        train_sha256=args.train_sha256,
        dev_dataset=args.dev_dataset,
        dev_sha256=args.dev_sha256,
        recipe_path=args.recipe,
        output_dir=args.output_dir,
        enable_observability_diagnostic=args.observability_diagnostic,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
