#!/usr/bin/env python3
"""Audit the four frozen RJ45 grasp-generation pilot attempts on CPU."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.experiment_contracts import atomic_write_json, load_finite_json  # noqa: E402
from rby1_assembly.rj45_bank_generation import audit_grasp_generation_pilot  # noqa: E402
from rby1_assembly.rj45_grasp_generation import load_grasp_generation_catalog  # noqa: E402


def _project_path(path: Path, name: str) -> Path:
    resolved = (path if path.is_absolute() else PROJECT_ROOT / path).resolve()
    if resolved != PROJECT_ROOT and not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError(f"{name} must stay inside the project")
    return resolved


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--attempts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    catalog_path = _project_path(args.catalog, "catalog")
    attempts_path = _project_path(args.attempts, "attempt manifest")
    output_path = _project_path(args.output, "audit output")
    if output_path.exists():
        raise FileExistsError(output_path)
    catalog = load_grasp_generation_catalog(catalog_path)
    document = load_finite_json(attempts_path)
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "catalog_sha256",
        "attempts",
    }:
        raise ValueError("pilot attempt manifest schema differs")
    if document["schema_version"] != 1 or document["catalog_sha256"] != _sha256(
        catalog_path
    ):
        raise ValueError("pilot attempt manifest catalog binding is stale")
    raw_attempts = document["attempts"]
    if not isinstance(raw_attempts, list):
        raise ValueError("pilot attempts must be an array")
    attempts = []
    for index, raw in enumerate(raw_attempts):
        if not isinstance(raw, dict):
            raise ValueError(f"pilot attempt {index} must be an object")
        if raw.get("status") == "failed":
            attempts.append(raw)
            continue
        if set(raw) != {
            "seed",
            "catalog_entry_id",
            "status",
            "artifact_path",
            "artifact_sha256",
        }:
            raise ValueError(f"successful pilot attempt {index} schema differs")
        artifact_path = _project_path(Path(raw["artifact_path"]), "pilot artifact")
        if raw["artifact_sha256"] != _sha256(artifact_path):
            raise ValueError(f"pilot artifact {index} SHA-256 mismatch")
        attempts.append(
            {
                "seed": raw["seed"],
                "catalog_entry_id": raw["catalog_entry_id"],
                "status": raw["status"],
                "artifact": load_finite_json(artifact_path),
            }
        )
    catalog_relative_path = str(catalog_path.relative_to(PROJECT_ROOT))
    catalog_digest = _sha256(catalog_path)
    result = audit_grasp_generation_pilot(
        attempts,
        catalog,
        catalog_path=catalog_relative_path,
        catalog_sha256=catalog_digest,
    )
    result.update(
        {
            "attempt_manifest_path": str(attempts_path.relative_to(PROJECT_ROOT)),
            "attempt_manifest_sha256": _sha256(attempts_path),
        }
    )
    atomic_write_json(output_path, result)
    return 0 if result["gate_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
