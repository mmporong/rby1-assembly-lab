#!/usr/bin/python3
"""Convert one STEP component to metric GLB and an explicit world-space mesh JSON.

KiCad is used as the OpenCascade-backed STEP importer.  The generated board is
only a carrier for a single J1 footprint; its body is excluded from the GLB.
No geometry scale is exposed because KiCad GLB coordinates are already metres.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_world_transform(vertices: Any, transform: Any) -> Any:
    """Return Nx3 vertices after applying a 4x4 homogeneous transform."""
    import numpy as np

    vertices_array = np.asarray(vertices, dtype=np.float64)
    transform_array = np.asarray(transform, dtype=np.float64)
    if vertices_array.ndim != 2 or vertices_array.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    if transform_array.shape != (4, 4):
        raise ValueError("transform must have shape (4, 4)")
    homogeneous = np.column_stack(
        (vertices_array, np.ones(len(vertices_array), dtype=np.float64))
    )
    transformed = homogeneous @ transform_array.T
    if not np.allclose(transformed[:, 3], 1.0, atol=1e-12):
        transformed[:, :3] /= transformed[:, 3, None]
    return transformed[:, :3]


def validate_metric_bbox(bounds_m: Any) -> None:
    """Reject empty, non-finite, or clearly non-metre component geometry."""
    import numpy as np

    bounds = np.asarray(bounds_m, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all():
        raise ValueError("metric bounds must be a finite 2x3 array")
    extents = bounds[1] - bounds[0]
    if np.any(extents <= 0.0):
        raise ValueError("metric bounds must have positive extents")
    # A connector smaller than 0.1 mm or larger than 0.5 m almost certainly
    # indicates an empty export or an accidental millimetre/metre rescale.
    if float(extents.min()) < 1e-4 or float(extents.max()) > 0.5:
        raise ValueError(f"implausible component extents in metres: {extents.tolist()}")


def _apply_numpy_compat_aliases(np: Any) -> list[str]:
    """Restore only the removed NumPy aliases required by the system toolchain."""
    aliases = {
        "long": np.int64,
        "uintc": np.uint32,
        "ulong": np.uint64,
    }
    applied = []
    for name, value in aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)
            applied.append(name)
    return applied


def _import_trimesh() -> tuple[Any, dict[str, Any]]:
    # The system image has a compatible trimesh build but an old SciPy module
    # that still refers to NumPy's removed scalar aliases.  trimesh itself only
    # needs these aliases during import for this GLB path.
    import numpy as np

    applied_aliases = _apply_numpy_compat_aliases(np)
    import scipy
    import trimesh

    return trimesh, {
        "applied_numpy_aliases": applied_aliases,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "trimesh": trimesh.__version__,
    }


def create_carrier_board(step_path: Path, board_path: Path) -> str:
    """Create a valid dummy board containing only STEP model J1."""
    import pcbnew

    board = pcbnew.BOARD()
    footprint = pcbnew.FOOTPRINT(board)
    footprint.SetReference("J1")
    footprint.SetValue(step_path.stem)
    model = pcbnew.FP_3DMODEL()
    model.m_Filename = str(step_path.resolve())
    model.m_Scale = pcbnew.VECTOR3D(1.0, 1.0, 1.0)
    footprint.Models().append(model)
    board.Add(footprint)

    outline_mm = [(-25.0, -25.0), (25.0, -25.0), (25.0, 25.0), (-25.0, 25.0)]
    for start, end in zip(outline_mm, outline_mm[1:] + outline_mm[:1]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetStart(pcbnew.VECTOR2I_MM(*start))
        edge.SetEnd(pcbnew.VECTOR2I_MM(*end))
        edge.SetWidth(pcbnew.FromMM(0.05))
        board.Add(edge)

    board_path.parent.mkdir(parents=True, exist_ok=True)
    if not pcbnew.SaveBoard(str(board_path), board):
        raise RuntimeError(f"KiCad board save failed: {board_path}")
    return pcbnew.GetBuildVersion()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def scene_to_mesh_payload(
    scene: Any,
    source_name: str,
    *,
    source_scene_map: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Expand scene nodes and emit a source-name-independent mesh payload.

    KiCad may assign fresh UUID-like node and geometry names on every export.
    Payload ordering and labels therefore derive only from transformed geometry
    and color content.  Caller-provided ``source_scene_map`` receives the raw
    scene identifiers for provenance, outside the reusable mesh payload.
    """
    import numpy as np

    candidates: list[dict[str, Any]] = []
    all_vertices: list[Any] = []
    for node_name in scene.graph.nodes_geometry:
        transform, geometry_name = scene.graph[node_name]
        geometry = scene.geometry[geometry_name]
        world_vertices = apply_world_transform(geometry.vertices, transform)
        if len(world_vertices) == 0 or len(geometry.faces) == 0:
            continue
        all_vertices.append(world_vertices)

        colors = getattr(geometry.visual, "vertex_colors", None)
        if colors is not None and len(colors):
            colors_array = np.asarray(colors, dtype=np.uint8)
            unique_colors = np.unique(colors_array.reshape((-1, colors_array.shape[-1])), axis=0)
            color_rgba = unique_colors[0].tolist() if len(unique_colors) == 1 else None
            vertex_colors = colors_array.tolist() if len(unique_colors) > 1 else None
        else:
            material = getattr(geometry.visual, "material", None)
            main_color = getattr(material, "main_color", [200, 200, 200, 255])
            color_rgba = np.asarray(main_color, dtype=np.uint8).tolist()
            vertex_colors = None

        content = {
            "color_rgba": color_rgba,
            "vertices_m": world_vertices.tolist(),
            "faces": np.asarray(geometry.faces, dtype=np.int64).tolist(),
            "bbox_m": np.vstack((world_vertices.min(axis=0), world_vertices.max(axis=0))).tolist(),
        }
        if vertex_colors is not None:
            content["vertex_colors_rgba"] = vertex_colors
        content_bytes = _canonical_json_bytes(content)
        content_sha256 = hashlib.sha256(content_bytes).hexdigest()
        candidates.append(
            {
                "content": content,
                "content_bytes": content_bytes,
                "content_sha256": content_sha256,
                "source": {
                    "node_name": str(node_name),
                    "geometry_name": str(geometry_name),
                    "material_name": str(
                        getattr(getattr(geometry.visual, "material", None), "name", "")
                    ),
                    "world_transform": np.asarray(transform, dtype=np.float64).tolist(),
                },
            }
        )

    if not all_vertices:
        raise ValueError("GLB contains no triangle meshes")
    candidates.sort(key=lambda item: (item["content_sha256"], item["content_bytes"]))
    meshes: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        label = f"mesh_{index:04d}_{candidate['content_sha256'][:16]}"
        mesh = {
            "node_name": label,
            "geometry_name": label,
            "material_name": f"material_{candidate['content_sha256'][:16]}",
            "content_sha256": candidate["content_sha256"],
            **candidate["content"],
        }
        meshes.append(mesh)
        if source_scene_map is not None:
            source_scene_map.append(
                {
                    "canonical_mesh_name": label,
                    **candidate["source"],
                }
            )

    combined = np.vstack(all_vertices)
    bounds = np.vstack((combined.min(axis=0), combined.max(axis=0)))
    validate_metric_bbox(bounds)
    return {
        "schema_version": 1,
        "source_model_name": source_name,
        "coordinate_space": "KiCad GLB world space",
        "length_unit": "metre",
        "applied_scale": 1.0,
        "mesh_count": len(meshes),
        "vertex_count": int(sum(len(mesh["vertices_m"]) for mesh in meshes)),
        "triangle_count": int(sum(len(mesh["faces"]) for mesh in meshes)),
        "bbox_m": bounds.tolist(),
        "extents_m": (bounds[1] - bounds[0]).tolist(),
        "meshes": meshes,
    }


def convert(step_path: Path, output_dir: Path, stem: str | None = None) -> dict[str, Path]:
    step_path = step_path.resolve()
    if not step_path.is_file():
        raise FileNotFoundError(step_path)
    if step_path.suffix.lower() not in {".step", ".stp"}:
        raise ValueError("input must be a .step or .stp file")
    if shutil.which("kicad-cli") is None:
        raise RuntimeError("kicad-cli is required")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = stem or step_path.stem
    board_path = output_dir / f"{output_stem}-carrier.kicad_pcb"
    glb_path = output_dir / f"{output_stem}.glb"
    mesh_path = output_dir / f"{output_stem}-mesh.json"
    provenance_path = output_dir / f"{output_stem}-provenance.json"

    pcbnew_version = create_carrier_board(step_path, board_path)
    command = [
        "kicad-cli", "pcb", "export", "glb", "--force", "--no-board-body",
        "--component-filter", "J1", "--output", str(glb_path), str(board_path),
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    if not glb_path.is_file() or glb_path.stat().st_size == 0:
        raise RuntimeError("KiCad did not create a non-empty GLB")

    trimesh, conversion_runtime = _import_trimesh()
    scene = trimesh.load(str(glb_path), force="scene", process=False)
    source_scene_map: list[dict[str, Any]] = []
    payload = scene_to_mesh_payload(
        scene, step_path.name, source_scene_map=source_scene_map
    )
    payload["source_sha256"] = sha256_file(step_path)
    mesh_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    kicad_version = subprocess.run(
        ["kicad-cli", "version"], check=True, text=True, capture_output=True
    ).stdout.strip()
    provenance = {
        "schema_version": 1,
        "source": {"path": str(step_path), "sha256": sha256_file(step_path)},
        "tools": {
            "kicad_cli": kicad_version,
            "pcbnew": pcbnew_version,
            "python": sys.version.split()[0],
            "numpy": conversion_runtime["numpy"],
            "scipy": conversion_runtime["scipy"],
            "trimesh": conversion_runtime["trimesh"],
        },
        "compatibility": {
            "numpy_removed_scalar_aliases_applied": conversion_runtime[
                "applied_numpy_aliases"
            ]
        },
        "command": command,
        "command_stdout": completed.stdout.strip(),
        "unit_policy": "KiCad GLB metre coordinates retained without rescaling",
        "source_scene_map": source_scene_map,
        "outputs": {
            "carrier_board": {"path": str(board_path), "sha256": sha256_file(board_path)},
            "glb": {"path": str(glb_path), "sha256": sha256_file(glb_path)},
            "mesh_json": {"path": str(mesh_path), "sha256": sha256_file(mesh_path)},
        },
        "bbox_m": payload["bbox_m"],
        "extents_m": payload["extents_m"],
        "mesh_count": payload["mesh_count"],
        "vertex_count": payload["vertex_count"],
        "triangle_count": payload["triangle_count"],
    }
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "board": board_path,
        "glb": glb_path,
        "mesh": mesh_path,
        "provenance": provenance_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", type=Path, help="input STEP/STP model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem")
    args = parser.parse_args()
    outputs = convert(args.step, args.output_dir, args.stem)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
