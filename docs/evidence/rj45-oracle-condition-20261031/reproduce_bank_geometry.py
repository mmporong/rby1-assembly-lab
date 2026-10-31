#!/usr/bin/env python3
"""Reproduce the held-grasp geometry exact-equality failure without Isaac Sim."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--expect-mismatches", type=int, required=True)
    parser.add_argument("--verify-full-train-bank", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    sys.path.insert(0, str(root / "src"))

    import numpy as np
    from rby1_assembly.held_grasp import (
        bank_source_contract_sha256,
        load_verified_held_grasp_bank,
    )
    from rby1_assembly.rj45_task import (
        load_rj45_task,
        verify_recorded_mating_geometry,
    )

    task_path = root / "configs/tasks/rj45-v1.json"
    bank_root = root / "artifacts/rj45-4c-bank-generation-v2-20261028"
    manifest_path = bank_root / "held-grasp-bank.json"
    report_path = bank_root / "seed-20261212/report.json"
    trace_path = bank_root / "seed-20261212/trace.jsonl"
    task = load_rj45_task(task_path)

    checked = 0
    mismatch_count = 0
    first_mismatch: dict[str, Any] | None = None
    field_counts: collections.Counter[str] = collections.Counter()
    max_abs_delta = {
        "depth_m": 0.0,
        "lateral_error_m": 0.0,
        "orientation_error_rad": 0.0,
    }
    max_ulp_delta = dict.fromkeys(max_abs_delta, 0)
    metric_fields = tuple(max_abs_delta)

    with trace_path.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            entry = record["environments"][0]
            if (
                entry.get("independent_eval_valid") is not True
                or not isinstance(entry.get("mating_stability"), dict)
                or not isinstance(entry.get("independent_eval_criteria"), dict)
            ):
                continue
            checked += 1
            poses = entry["object_poses_world_wxyz"]
            recalculated = verify_recorded_mating_geometry(
                poses["socket"],
                poses["plug"],
                task["frames"]["socket_entry"],
                task["frames"]["plug_tip"],
                entry["actual_depth_m"],
                entry["actual_lateral_error_m"],
                entry["actual_orientation_error_rad"],
            )
            recorded = entry["independent_mating_geometry"]
            if recorded == recalculated:
                continue
            mismatch_count += 1
            deltas: dict[str, float] = {}
            ulps: dict[str, int] = {}
            for field in metric_fields:
                before = np.float64(recorded[field])
                after = np.float64(recalculated[field])
                delta = float(after - before)
                deltas[field] = delta
                absolute = abs(delta)
                max_abs_delta[field] = max(max_abs_delta[field], absolute)
                if delta != 0.0:
                    field_counts[field] += 1
                if before >= 0.0 and after >= 0.0:
                    ulp = abs(int(before.view(np.uint64)) - int(after.view(np.uint64)))
                else:
                    ulp = 0
                ulps[field] = ulp
                max_ulp_delta[field] = max(max_ulp_delta[field], ulp)
            if first_mismatch is None:
                first_mismatch = {
                    "control_step": record["control_step"],
                    "timestamp_s": entry["timestamp_s"],
                    "recorded": recorded,
                    "recalculated": recalculated,
                    "delta": deltas,
                    "ulp_delta_nonnegative": ulps,
                }

    bank_result: dict[str, Any] | None = None
    if args.verify_full_train_bank:
        bank, proof = load_verified_held_grasp_bank(
            root,
            task,
            manifest_path,
            "train",
            allow_final=False,
        )
        bank_result = {
            "passed": True,
            "selected_count": len(bank),
            "selected_grasp_ids": proof["selected_grasp_ids"],
            "manifest_sha256": proof["manifest_sha256"],
            "source_contract_sha256": bank_source_contract_sha256(proof),
        }

    result = {
        "schema_name": "rj45_bank_geometry_cpu_reproduction_v1",
        "scope": "CPU-only replay of frozen train-bank evidence; no Isaac Sim or physics",
        "host": {
            "node": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "runtime": {
            "numpy": np.__version__,
            "OPENBLAS_CORETYPE": os.environ.get("OPENBLAS_CORETYPE"),
            "NPY_DISABLE_CPU_FEATURES": os.environ.get("NPY_DISABLE_CPU_FEATURES"),
            "cpu_features": {
                name: bool(np.core._multiarray_umath.__cpu_features__.get(name))
                for name in ("AVX2", "FMA3", "AVX512F")
            },
        },
        "inputs": {
            "project_root": str(root),
            "task_sha256": sha256(task_path),
            "manifest_sha256": sha256(manifest_path),
            "report_sha256": sha256(report_path),
            "trace_sha256": sha256(trace_path),
        },
        "geometry_replay": {
            "eligible_rows_checked": checked,
            "dict_exact_mismatches": mismatch_count,
            "field_mismatch_counts": dict(sorted(field_counts.items())),
            "max_abs_delta": max_abs_delta,
            "max_ulp_delta_nonnegative": max_ulp_delta,
            "first_mismatch": first_mismatch,
        },
        "full_train_bank_verification": bank_result,
        "expectation": {
            "expected_mismatches": args.expect_mismatches,
            "matched": mismatch_count == args.expect_mismatches,
        },
    }
    if mismatch_count != args.expect_mismatches:
        raise SystemExit(
            f"expected {args.expect_mismatches} mismatches, observed {mismatch_count}"
        )
    payload = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    sys.stdout.write(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
