#!/usr/bin/env python3
"""Build a verified held-grasp bank from completed one-environment nominal runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.experiment_contracts import atomic_write_json, load_finite_json
from rby1_assembly.held_grasp import build_held_grasp_artifact
from rby1_assembly.rj45_bank_generation import (
    build_manifest,
    file_sha256,
    validate_nonzero_position_noise,
)
from rby1_assembly.rj45_task import load_rj45_task


def _project_path(value: Path, name: str) -> Path:
    path = (value if value.is_absolute() else PROJECT_ROOT / value).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{name} must stay inside the project")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-plan", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--translation-threshold-m", type=float, required=True)
    parser.add_argument("--rotation-threshold-rad", type=float, required=True)
    parser.add_argument("--joint-threshold-si", type=float, required=True)
    parser.add_argument("--min-train", type=int, required=True)
    parser.add_argument("--min-dev", type=int, required=True)
    parser.add_argument("--min-final", type=int, required=True)
    args = parser.parse_args()

    plan_path = _project_path(args.candidate_plan, "candidate plan")
    task_path = _project_path(args.task_config, "task config")
    artifact_dir = _project_path(args.artifact_dir, "artifact directory")
    manifest_path = _project_path(args.manifest, "manifest")
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    plan = load_finite_json(plan_path)
    if set(plan) != {"schema_version", "task_id", "position_noise_m", "candidates"} or plan.get("schema_version") != 1 or plan.get("task_id") != "rj45-v1":
        raise ValueError("candidate plan schema must be version 1 for rj45-v1")
    task = load_rj45_task(task_path)
    validate_nonzero_position_noise(task, plan["position_noise_m"])
    raw_candidates = plan["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate plan candidates must be non-empty")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    built = []
    for index, candidate in enumerate(raw_candidates):
        if not isinstance(candidate, dict) or set(candidate) != {"grasp_id", "split", "generation_seed", "nominal_report"}:
            raise ValueError(f"candidate {index} schema differs")
        report_path = _project_path(Path(candidate["nominal_report"]), f"candidate {index} report")
        report = load_finite_json(report_path)
        provenance = report.get("provenance")
        if not isinstance(provenance, dict) or provenance.get("seed") != candidate["generation_seed"]:
            raise ValueError(f"candidate {index} report seed differs from generation_seed")
        output = artifact_dir / f"{candidate['grasp_id']}.json"
        if output.exists():
            raise FileExistsError(output)
        artifact = build_held_grasp_artifact(PROJECT_ROOT, task, report_path, output)
        built.append(
            {
                "grasp_id": candidate["grasp_id"],
                "split": candidate["split"],
                "generation_seed": candidate["generation_seed"],
                "artifact_path": str(output.relative_to(PROJECT_ROOT)),
                "artifact": artifact,
                "artifact_sha256": file_sha256(output),
            }
        )
    manifest = build_manifest(
        built,
        separation_thresholds={
            "object_to_ee_translation_m": args.translation_threshold_m,
            "object_to_ee_rotation_rad": args.rotation_threshold_rad,
            "joint_position_linf_si": args.joint_threshold_si,
        },
        minimum_split_counts={
            "train": args.min_train,
            "dev": args.min_dev,
            "final": args.min_final,
        },
    )
    atomic_write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "candidate_count": len(built),
                "selection_basis": "physical_validity_and_descriptor_separation_only",
                "policy_performance_used": False,
                "final_split_sealed_for_policy_selection": True,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
