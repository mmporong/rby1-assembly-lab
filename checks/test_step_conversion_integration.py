#!/usr/bin/env python3
"""Run the real KiCad STEP conversion check against an explicitly supplied model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_step_model.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _geometry_signature(mesh: dict[str, object]) -> list[str]:
    signature: list[str] = []
    for item in mesh["meshes"]:
        content = json.dumps(
            {
                "vertices_m": item["vertices_m"],
                "faces": item["faces"],
                "color_rgba": item.get("color_rgba"),
                "vertex_colors_rgba": item.get("vertex_colors_rgba"),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        signature.append(hashlib.sha256(content.encode("utf-8")).hexdigest())
    return sorted(signature)


def _convert(step_path: Path, output_dir: Path) -> tuple[dict[str, object], dict[str, object], Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            "/usr/bin/python3",
            str(SCRIPT),
            str(step_path),
            "--output-dir",
            str(output_dir),
            "--stem",
            "jack",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    outputs = json.loads(completed.stdout.strip().splitlines()[-1])
    mesh_path = Path(outputs["mesh"])
    mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
    provenance = json.loads(Path(outputs["provenance"]).read_text(encoding="utf-8"))
    return mesh, provenance, mesh_path


def run(
    step_path: Path,
    output_root: Path,
    reference_mesh_path: Path | None = None,
) -> dict[str, object]:
    step_path = step_path.resolve()
    if not step_path.is_file():
        raise FileNotFoundError(f"STEP input does not exist: {step_path}")
    output_root = output_root.resolve()
    mesh, provenance, mesh_path = _convert(step_path, output_root / "run-a")
    second_mesh, second_provenance, second_mesh_path = _convert(
        step_path, output_root / "run-b"
    )
    glb_path = Path(provenance["outputs"]["glb"]["path"])

    if glb_path.stat().st_size <= 100_000:
        raise AssertionError(f"unexpectedly small GLB: {glb_path.stat().st_size} bytes")
    if mesh["length_unit"] != "metre" or mesh["applied_scale"] != 1.0:
        raise AssertionError("conversion changed the KiCad GLB metre scale")
    if mesh["mesh_count"] <= 100 or mesh["triangle_count"] <= 1_000:
        raise AssertionError("converted mesh is unexpectedly incomplete")

    expected_m = sorted([0.01778, 0.018669, 0.01733042])
    actual_m = sorted(float(value) for value in mesh["extents_m"])
    if any(abs(actual - expected) > 2e-6 for actual, expected in zip(actual_m, expected_m)):
        raise AssertionError(f"unexpected TE 5555153-1 extents in metres: {actual_m}")
    if not all(item["vertices_m"] and item["faces"] for item in mesh["meshes"]):
        raise AssertionError("one or more exported meshes are empty")
    if not all(
        item["node_name"] == item["geometry_name"]
        and item["node_name"].startswith("mesh_")
        and len(item["content_sha256"]) == 64
        for item in mesh["meshes"]
    ):
        raise AssertionError("canonical mesh labels or content hashes are missing")
    if not provenance.get("source_scene_map"):
        raise AssertionError("raw source scene identifiers are missing from provenance")
    if "--no-board-body" not in provenance["command"] or "J1" not in provenance["command"]:
        raise AssertionError("KiCad export provenance lacks component-only filtering")

    first_payload_sha256 = _sha256(mesh_path)
    second_payload_sha256 = _sha256(second_mesh_path)
    if first_payload_sha256 != second_payload_sha256:
        raise AssertionError(
            "same STEP produced different canonical mesh payloads: "
            f"{first_payload_sha256} != {second_payload_sha256}"
        )
    if _geometry_signature(mesh) != _geometry_signature(second_mesh):
        raise AssertionError("repeated conversion changed vertices, faces, or colors")
    if mesh["bbox_m"] != second_mesh["bbox_m"] or mesh["extents_m"] != second_mesh["extents_m"]:
        raise AssertionError("repeated conversion changed metric bounds")
    reference_geometry_equivalent = None
    if reference_mesh_path is not None:
        reference_mesh_path = reference_mesh_path.resolve()
        reference_mesh = json.loads(reference_mesh_path.read_text(encoding="utf-8"))
        reference_geometry_equivalent = (
            _geometry_signature(mesh) == _geometry_signature(reference_mesh)
            and mesh["bbox_m"] == reference_mesh["bbox_m"]
            and mesh["extents_m"] == reference_mesh["extents_m"]
            and mesh["vertex_count"] == reference_mesh["vertex_count"]
            and mesh["triangle_count"] == reference_mesh["triangle_count"]
        )
        if not reference_geometry_equivalent:
            raise AssertionError(
                "canonicalization changed reference vertices, face winding, colors, or bounds"
            )

    return {
        "status": "PASS",
        "source": str(step_path),
        "source_sha256": mesh["source_sha256"],
        "mesh_payload_sha256": first_payload_sha256,
        "repeat_mesh_payload_sha256": second_payload_sha256,
        "glb_sha256": provenance["outputs"]["glb"]["sha256"],
        "repeat_glb_sha256": second_provenance["outputs"]["glb"]["sha256"],
        "geometry_equivalent": True,
        "reference_geometry_equivalent": reference_geometry_equivalent,
        "extents_m": mesh["extents_m"],
        "mesh_count": mesh["mesh_count"],
        "vertex_count": mesh["vertex_count"],
        "triangle_count": mesh["triangle_count"],
        "outputs": [str(mesh_path), str(second_mesh_path)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step",
        type=Path,
        default=os.environ.get("RBY1_RJ45_STEP"),
        required="RBY1_RJ45_STEP" not in os.environ,
        help="path to the locally cached official TE 5555153-1 STEP file; alternatively set RBY1_RJ45_STEP",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "artifacts" / "rj45-conversion-repro",
        help="ignored directory that receives two independent conversion runs",
    )
    parser.add_argument(
        "--reference-mesh",
        type=Path,
        help="optional pre-canonicalization mesh JSON for exact geometry comparison",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.step, args.output_root, args.reference_mesh),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
