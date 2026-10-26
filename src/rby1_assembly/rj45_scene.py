"""Build the RJ45 task assets without starting Isaac Sim.

The vendor jack mesh is visual evidence only.  Socket contact is represented by
an explicit compound collider so the female cavity remains open; the plug is a
drawing-derived rigid envelope with separate nose and rear colliders.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from rby1_assembly.rj45_task import load_rj45_task
from rby1_assembly.scene import _collision, _cube, _material, _reference, _stage, build_robot


_JACK_VISIBLE_OPENING_HEIGHT_M = 0.01332


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_inside(project: Path, relative_path: str) -> Path:
    project = project.resolve()
    resolved = (project / relative_path).resolve()
    if resolved != project and project not in resolved.parents:
        raise ValueError(f"asset path escapes project: {relative_path}")
    return resolved


def _vec3(value: Iterable[float]) -> Gf.Vec3f:
    return Gf.Vec3f(*(float(component) for component in value))


def _visual_cube(
    stage: Usd.Stage,
    path: str,
    size_m: tuple[float, float, float],
    center_m: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    opacity: float = 1.0,
) -> Usd.Prim:
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center_m))
    cube.AddScaleOp().Set(Gf.Vec3f(*size_m))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    cube.CreateDisplayOpacityAttr([float(opacity)])
    return cube.GetPrim()


def _visual_cylinder(
    stage: Usd.Stage,
    path: str,
    radius_m: float,
    height_m: float,
    center_m: tuple[float, float, float],
    color: tuple[float, float, float],
) -> Usd.Prim:
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.y)
    cylinder.CreateRadiusAttr(radius_m)
    cylinder.CreateHeightAttr(height_m)
    cylinder.AddTranslateOp().Set(Gf.Vec3d(*center_m))
    cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return cylinder.GetPrim()


def _frame(stage: Usd.Stage, path: str, frame: dict[str, Any]) -> None:
    prim = UsdGeom.Xform.Define(stage, path)
    prim.AddTranslateOp().Set(Gf.Vec3d(*frame["position_m"]))
    quaternion = frame["quaternion_wxyz"]
    prim.AddOrientOp().Set(Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:])))


def _collision_cube(
    stage: Usd.Stage,
    path: str,
    size_m: tuple[float, float, float],
    center_m: tuple[float, float, float],
    material,
    contact: dict[str, Any],
    role: str,
) -> Usd.Prim:
    prim = _cube(stage, path, size_m, center_m, (0.22, 0.54, 0.72), material, contact)
    UsdGeom.Imageable(prim).MakeInvisible()
    prim.SetCustomDataByKey("rj45:collision_role", role)
    return prim


def _convex_prism(
    stage: Usd.Stage,
    path: str,
    xy_profile: list[tuple[float, float]],
    z_range_m: tuple[float, float],
    material,
    contact: dict[str, Any],
    role: str,
) -> Usd.Prim:
    """Create a convex vertical prism from a counter-clockwise XY profile."""

    z0, z1 = z_range_m
    points = [Gf.Vec3f(x, y, z) for z in (z0, z1) for x, y in xy_profile]
    count = len(xy_profile)
    faces: list[list[int]] = []
    faces.append(list(reversed(range(count))))
    faces.append(list(range(count, count * 2)))
    for index in range(count):
        following = (index + 1) % count
        faces.append([index, following, count + following, count + index])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
    mesh.CreateFaceVertexIndicesAttr([vertex for face in faces for vertex in face])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.22, 0.54, 0.72)])
    _collision(mesh.GetPrim(), material, contact)
    UsdGeom.Imageable(mesh.GetPrim()).MakeInvisible()
    mesh.GetPrim().SetCustomDataByKey("rj45:collision_role", role)
    return mesh.GetPrim()


def load_jack_mesh(project: Path, task: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    geometry = task.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError("geometry must be present for the RJ45 scene")
    mesh_relative = geometry.get("jack_mesh")
    if not isinstance(mesh_relative, str) or not mesh_relative:
        raise ValueError("geometry.jack_mesh must be a project-relative path")
    mesh_path = _resolve_inside(project, mesh_relative)
    expected_mesh_sha256 = geometry.get("jack_canonical_mesh_sha256")
    if (
        not isinstance(expected_mesh_sha256, str)
        or len(expected_mesh_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_mesh_sha256)
    ):
        raise ValueError("geometry.jack_canonical_mesh_sha256 must be a lowercase SHA-256")
    try:
        actual_mesh_sha256 = _sha256(mesh_path)
    except OSError as error:
        raise ValueError(f"invalid jack mesh {mesh_path}: {error}") from error
    if actual_mesh_sha256 != expected_mesh_sha256:
        raise ValueError(
            "jack mesh hash does not match geometry.jack_canonical_mesh_sha256"
        )
    try:
        mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid jack mesh {mesh_path}: {error}") from error
    required = {"schema_version", "length_unit", "meshes", "source_sha256"}
    if not required.issubset(mesh) or mesh["schema_version"] != 1 or mesh["length_unit"] != "metre":
        raise ValueError("jack mesh schema or units are unsupported")
    expected = geometry.get("jack_source_step_sha256")
    if mesh["source_sha256"] != expected:
        raise ValueError(
            "jack mesh source hash does not match geometry.jack_source_step_sha256"
        )
    if not isinstance(mesh["meshes"], list) or not mesh["meshes"]:
        raise ValueError("jack mesh contains no mesh records")
    return mesh_path, mesh


def _transform_point(point: list[float], transform: dict[str, Any]) -> tuple[float, float, float]:
    rows = transform["rotation_rows"]
    translation = transform["translation_m"]
    scale = float(transform["scale"])
    if len(point) != 3 or len(rows) != 3 or any(len(row) != 3 for row in rows):
        raise ValueError("jack CAD transform must be a finite 3x3 transform")
    values = []
    for index in range(3):
        value = scale * sum(float(rows[index][axis]) * float(point[axis]) for axis in range(3))
        value += float(translation[index])
        if not math.isfinite(value):
            raise ValueError("jack CAD transform produced a non-finite point")
        values.append(value)
    return tuple(values)


def _add_jack_visual(
    stage: Usd.Stage, root_path: str, mesh: dict[str, Any], transform: dict[str, Any]
) -> dict[str, Any]:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertex_count = 0
    triangle_count = 0
    for mesh_index, source in enumerate(mesh["meshes"]):
        vertices = source.get("vertices_m")
        faces = source.get("faces")
        if not isinstance(vertices, list) or not vertices or not isinstance(faces, list) or not faces:
            raise ValueError(f"jack submesh {mesh_index} is empty")
        transformed = [_transform_point(vertex, transform) for vertex in vertices]
        for point in transformed:
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
        if any(
            not isinstance(face, list)
            or len(face) < 3
            or any(not isinstance(index, int) or index < 0 or index >= len(vertices) for index in face)
            for face in faces
        ):
            raise ValueError(f"jack submesh {mesh_index} has invalid face indices")
        usd_mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/part_{mesh_index:03d}")
        usd_mesh.CreatePointsAttr([_vec3(point) for point in transformed])
        usd_mesh.CreateFaceVertexCountsAttr([len(face) for face in faces])
        usd_mesh.CreateFaceVertexIndicesAttr([index for face in faces for index in face])
        usd_mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        rgba = source.get("color_rgba", [102, 102, 102, 255])
        if not isinstance(rgba, list) or len(rgba) != 4:
            raise ValueError(f"jack submesh {mesh_index} has invalid color_rgba")
        usd_mesh.CreateDisplayColorAttr([Gf.Vec3f(*(float(value) / 255.0 for value in rgba[:3]))])
        usd_mesh.CreateDisplayOpacityAttr([float(rgba[3]) / 255.0])
        vertex_count += len(vertices)
        triangle_count += sum(len(face) - 2 for face in faces)
    return {
        "mesh_count": len(mesh["meshes"]),
        "vertex_count": vertex_count,
        "triangle_count": triangle_count,
        "transformed_bbox_m": [minimum, maximum],
    }


def build_socket(
    output: Path, task: dict[str, Any], mesh: dict[str, Any]
) -> dict[str, Any]:
    geometry = task["geometry"]
    enclosure = geometry["adapter_enclosure_m"]
    guide = geometry["port_guide"]
    width = float(enclosure["width"])
    length = float(enclosure["length"])
    height = float(enclosure["height"])
    wall = float(enclosure["wall"])
    entry_y = float(guide["entry_y_m"])
    cavity_depth = float(guide["cavity_depth_m"])
    back_wall = float(guide["back_wall_m"])
    bevel_depth = float(guide["bevel_depth_m"])
    mouth_half = float(guide["mouth_half_width_m"])
    throat_half = float(guide["throat_half_width_m"])
    lower_z, upper_z = (float(value) for value in guide["main_z_range_m"])
    if not (
        entry_y == -length / 2.0
        and 0.0 < throat_half <= mouth_half < width / 2.0
        and 0.0 < bevel_depth < cavity_depth < length
        and -height / 2.0 < lower_z < upper_z < height / 2.0
        and 0.0 < wall < min(width, height) / 2.0
    ):
        raise ValueError("RJ45 enclosure and port-guide dimensions are inconsistent")

    stage, root = _stage(output, "Socket")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(task["socket"]["mass_kg"]))
    material = _material(stage, "/Socket/ContactMaterial", task["contact"])

    # White project-designed enclosure.  The four-piece front bezel leaves the
    # full 17.78 x 13.32 mm vendor-jack opening visible.
    opening_w = 2.0 * max(abs(mesh["bbox_m"][0][0]), abs(mesh["bbox_m"][1][0]))
    opening_h = _JACK_VISIBLE_OPENING_HEIGHT_M
    bezel_depth = min(0.0015, cavity_depth)
    bezel_y = entry_y + bezel_depth / 2.0
    side_size = (width - opening_w) / 2.0
    cap_size = (height - opening_h) / 2.0
    for name, sign in (("left", -1.0), ("right", 1.0)):
        _visual_cube(
            stage,
            f"/Socket/Visual/Enclosure/front_{name}",
            (side_size, bezel_depth, height),
            (sign * (opening_w + side_size) / 2.0, bezel_y, 0.0),
            (0.92, 0.94, 0.96),
        )
    for name, sign in (("bottom", -1.0), ("top", 1.0)):
        _visual_cube(
            stage,
            f"/Socket/Visual/Enclosure/front_{name}",
            (opening_w, bezel_depth, cap_size),
            (0.0, bezel_y, sign * (opening_h + cap_size) / 2.0),
            (0.92, 0.94, 0.96),
        )
    body_start_y = entry_y + cavity_depth + back_wall
    _visual_cube(
        stage,
        "/Socket/Visual/Enclosure/rear_body",
        (width, length / 2.0 - body_start_y, height),
        (0.0, (body_start_y + length / 2.0) / 2.0, 0.0),
        (0.92, 0.94, 0.96),
    )
    cad = _add_jack_visual(stage, "/Socket/Visual/VendorJack", mesh, geometry["jack_cad_transform"])

    cavity_end_y = entry_y + cavity_depth
    bevel_end_y = entry_y + bevel_depth
    outer_half = width / 2.0
    _convex_prism(
        stage,
        "/Socket/Collision/guide_left_bevel",
        [(-outer_half, entry_y), (-mouth_half, entry_y), (-throat_half, bevel_end_y), (-outer_half, bevel_end_y)],
        (lower_z, upper_z),
        material,
        task["contact"],
        "left_tapered_guide",
    )
    _convex_prism(
        stage,
        "/Socket/Collision/guide_right_bevel",
        [(mouth_half, entry_y), (outer_half, entry_y), (outer_half, bevel_end_y), (throat_half, bevel_end_y)],
        (lower_z, upper_z),
        material,
        task["contact"],
        "right_tapered_guide",
    )
    straight_length = cavity_end_y - bevel_end_y
    for name, sign in (("left", -1.0), ("right", 1.0)):
        inner_center = sign * (throat_half + (outer_half - throat_half) / 2.0)
        _collision_cube(
            stage,
            f"/Socket/Collision/guide_{name}_straight",
            (outer_half - throat_half, straight_length, upper_z - lower_z),
            (inner_center, (bevel_end_y + cavity_end_y) / 2.0, (lower_z + upper_z) / 2.0),
            material,
            task["contact"],
            f"{name}_straight_guide",
        )
    _collision_cube(
        stage,
        "/Socket/Collision/guide_bottom",
        (width, cavity_depth, lower_z + height / 2.0),
        (0.0, (entry_y + cavity_end_y) / 2.0, (-height / 2.0 + lower_z) / 2.0),
        material,
        task["contact"],
        "bottom_guide",
    )
    _collision_cube(
        stage,
        "/Socket/Collision/guide_top",
        (width, cavity_depth, height / 2.0 - upper_z),
        (0.0, (entry_y + cavity_end_y) / 2.0, (upper_z + height / 2.0) / 2.0),
        material,
        task["contact"],
        "top_guide",
    )
    _collision_cube(
        stage,
        "/Socket/Collision/backstop",
        (2.0 * throat_half, back_wall, upper_z - lower_z),
        (0.0, cavity_end_y + back_wall / 2.0, (lower_z + upper_z) / 2.0),
        material,
        task["contact"],
        "insertion_backstop",
    )
    rear_length = length / 2.0 - (cavity_end_y + back_wall)
    if rear_length > 0.0:
        _collision_cube(
            stage,
            "/Socket/Collision/enclosure_rear",
            (width, rear_length, height),
            (0.0, (cavity_end_y + back_wall + length / 2.0) / 2.0, 0.0),
            material,
            task["contact"],
            "adapter_enclosure_rear",
        )

    _frame(stage, "/Socket/Frames/socket_entry", task["frames"]["socket_entry"])
    _frame(stage, "/Socket/Frames/left_grasp", task["frames"]["left_grasp"])
    root.SetCustomDataByKey("rj45:free_rigid_body", True)
    root.SetCustomDataByKey("rj45:port_axis", "+Y")
    root.SetCustomDataByKey("rj45:cavity_collision", "compound_open_guide")
    root.SetCustomDataByKey("rj45:vendor_jack_collision", "disabled_visual_only")
    root.SetCustomDataByKey("rj45:mass_source", task["socket"]["mass_source"])
    stage.GetRootLayer().Save()
    return {
        "path": str(output),
        "mass_kg": float(task["socket"]["mass_kg"]),
        "free_rigid_body": True,
        "attachment_joints": 0,
        "collision_strategy": "compound guide walls with open female cavity; vendor CAD visual only",
        "collision_prim_count": 8,
        "entry_y_m": entry_y,
        "cavity_depth_m": cavity_depth,
        "cavity_open_bounds_m": {
            "mouth_x": [-mouth_half, mouth_half],
            "throat_x": [-throat_half, throat_half],
            "z": [lower_z, upper_z],
            "y": [entry_y, cavity_end_y],
        },
        "visual_opening_m": [opening_w, opening_h],
        "vendor_cad": cad,
    }


def build_plug(output: Path, task: dict[str, Any]) -> dict[str, Any]:
    profile = task["geometry"]["plug_profile"]
    width = float(profile["max_width_m"])
    total_length = float(profile["total_length_m"])
    nose_length = float(profile["nose_length_m"])
    nose_height = float(profile["terminated_nose_height_m"])
    rear_height = float(profile["rear_height_m"])
    tip_y = float(task["frames"]["plug_tip"]["position_m"][1])
    nose_back_y = tip_y - nose_length
    rear_back_y = tip_y - total_length
    if not (
        abs(tip_y - total_length / 2.0) <= 1.0e-12
        and rear_back_y < nose_back_y < tip_y
        and width > 0.0
        and nose_height > 0.0
        and rear_height >= nose_height
    ):
        raise ValueError("drawing-derived RJ45 plug dimensions are inconsistent")

    stage, root = _stage(output, "Plug")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(float(task["plug"]["mass_kg"]))
    material = _material(stage, "/Plug/ContactMaterial", task["contact"])
    polycarbonate = (0.28, 0.66, 0.82)
    _visual_cube(
        stage,
        "/Plug/Visual/polycarbonate_nose",
        (width, nose_length, nose_height),
        (0.0, (nose_back_y + tip_y) / 2.0, 0.0),
        polycarbonate,
        opacity=0.48,
    )
    _visual_cube(
        stage,
        "/Plug/Visual/polycarbonate_rear",
        (width, nose_back_y - rear_back_y, rear_height),
        (0.0, (rear_back_y + nose_back_y) / 2.0, (rear_height - nose_height) / 2.0),
        polycarbonate,
        opacity=0.48,
    )
    _collision_cube(
        stage,
        "/Plug/Collision/nose",
        (width, nose_length, nose_height),
        (0.0, (nose_back_y + tip_y) / 2.0, 0.0),
        material,
        task["contact"],
        "drawing_envelope_nose",
    )
    _collision_cube(
        stage,
        "/Plug/Collision/rear",
        (width, nose_back_y - rear_back_y, rear_height),
        (0.0, (rear_back_y + nose_back_y) / 2.0, (rear_height - nose_height) / 2.0),
        material,
        task["contact"],
        "drawing_envelope_rear",
    )

    count = int(profile["contact_count"])
    pitch = float(profile["contact_pitch_m"])
    contact_width = float(profile["contact_width_m"])
    contact_length = float(profile["contact_length_m"])
    contact_center_y = tip_y - contact_length / 2.0 - 0.00035
    for index in range(count):
        x = (index - (count - 1) / 2.0) * pitch
        _visual_cube(
            stage,
            f"/Plug/Visual/contacts/contact_{index + 1}",
            (contact_width, contact_length, 0.00010),
            (x, contact_center_y, nose_height / 2.0 - 0.00005),
            (0.92, 0.66, 0.12),
        )
    # The latch is shown depressed and remains inside the collision envelope.
    _visual_cube(
        stage,
        "/Plug/Visual/latch_depressed",
        (0.0046, 0.0062, 0.00018),
        (0.0, -0.0002, nose_height / 2.0 - 0.00009),
        (0.22, 0.53, 0.68),
        opacity=0.60,
    )
    cable_length = 0.010
    _visual_cylinder(
        stage,
        "/Plug/Visual/cable_stub",
        0.0024,
        cable_length,
        (0.0, rear_back_y - cable_length / 2.0, 0.0005),
        (0.12, 0.18, 0.23),
    )
    _frame(stage, "/Plug/Frames/plug_tip", task["frames"]["plug_tip"])
    _frame(stage, "/Plug/Frames/right_grasp", task["frames"]["right_grasp"])
    root.SetCustomDataByKey("rj45:free_rigid_body", True)
    root.SetCustomDataByKey("rj45:geometry_source", "TE C-558530 revision H drawing-derived envelope")
    root.SetCustomDataByKey("rj45:latch_model", "visual_only_depressed_unlocked")
    root.SetCustomDataByKey("rj45:cable_model", "visual_only_no_bending_or_tension")
    root.SetCustomDataByKey("rj45:mass_source", task["plug"]["mass_source"])
    stage.GetRootLayer().Save()
    return {
        "path": str(output),
        "mass_kg": float(task["plug"]["mass_kg"]),
        "free_rigid_body": True,
        "attachment_joints": 0,
        "collision_prim_count": 2,
        "collision_strategy": "two drawing-envelope convex boxes; visual contacts/latch/cable excluded",
        "tip_y_m": tip_y,
        "nose_y_range_m": [nose_back_y, tip_y],
        "rear_y_range_m": [rear_back_y, nose_back_y],
        "width_m": width,
        "nose_height_m": nose_height,
        "rear_height_m": rear_height,
        "gold_contact_count": count,
        "latch_visual_only": True,
        "cable_visual_only": True,
    }


def build_table(output: Path, task: dict[str, Any]) -> dict[str, Any]:
    table = task["table"]
    nest = task["geometry"]["pickup_nest"]
    plate = nest["top_plate"]
    locator = nest["locator"]
    stage, root = _stage(output, "Table")
    material = _material(stage, "/Table/ContactMaterial", task["contact"])
    z = float(table["top_height_m"]) - float(table["size_m"][2]) / 2.0
    center_xy = table.get("center_xy_m", [0.0, 0.0])
    _cube(
        stage,
        "/Table/top",
        tuple(float(value) for value in table["size_m"]),
        (float(center_xy[0]), float(center_xy[1]), z),
        (0.72, 0.75, 0.78),
        material,
        task["contact"],
    )
    nest_center = tuple(float(value) for value in nest["world_center_m"])
    nest_size = tuple(float(value) for value in nest["size_m"])
    _cube(
        stage,
        "/Table/pickup_nest",
        nest_size,
        nest_center,
        (0.18, 0.45, 0.62),
        material,
        task["contact"],
    ).SetCustomDataByKey("rj45:role", "front_plug_support_column")

    plug_center = tuple(
        float(value) for value in task["plug"]["initial_position_m"]
    )
    plug_width = float(task["geometry"]["plug_profile"]["max_width_m"])
    plug_tip_y = plug_center[1] + float(task["frames"]["plug_tip"]["position_m"][1])
    plate_size = tuple(float(value) for value in plate["size_m"])
    if len(plate_size) != 3 or min(plate_size) <= 0.0:
        raise ValueError("pickup nest top plate dimensions must be positive XYZ values")
    slot_width = float(locator["slot_width_m"])
    wall_thickness = float(locator["wall_thickness_m"])
    wall_height = float(locator["wall_height_m"])
    front_clearance = float(locator["front_clearance_m"])
    if min(slot_width, wall_thickness, wall_height, front_clearance) <= 0.0:
        raise ValueError("pickup locator dimensions and clearances must be positive")
    side_clearance = (slot_width - plug_width) / 2.0
    required_clearance = float(task["contact"]["contact_offset_m"])
    if (
        side_clearance <= required_clearance + 1.0e-12
        or front_clearance <= required_clearance + 1.0e-12
    ):
        raise ValueError(
            "pickup locator must leave more than one contact offset around the nominal plug"
        )

    column_rear_y = nest_center[1] - nest_size[1] / 2.0
    plate_rear_y = plug_center[1] + float(plate["rear_edge_relative_to_plug_m"])
    rail_rear_y = plug_center[1] + float(
        locator["side_rail_rear_edge_relative_to_plug_m"]
    )
    if (
        abs(column_rear_y - plate_rear_y) > 1.0e-12
        or abs(rail_rear_y - plate_rear_y) > 1.0e-12
    ):
        raise ValueError(
            "pickup locator rear edges must preserve the support column rear boundary"
        )
    plate_front_y = plate_rear_y + plate_size[1]
    plate_top = nest_center[2] + nest_size[2] / 2.0
    plate_center = (
        plug_center[0],
        plate_rear_y + plate_size[1] / 2.0,
        plate_top - plate_size[2] / 2.0,
    )
    _cube(
        stage,
        "/Table/pickup_nest_plate",
        plate_size,
        plate_center,
        (0.24, 0.50, 0.68),
        material,
        task["contact"],
    ).SetCustomDataByKey("rj45:role", "plug_nose_support_plate")

    stop_inner_y = plug_tip_y + front_clearance
    stop_center_y = stop_inner_y + wall_thickness / 2.0
    stop_front_y = stop_inner_y + wall_thickness
    rail_length = stop_inner_y - rail_rear_y
    outer_width = slot_width + 2.0 * wall_thickness
    if rail_length <= 0.0:
        raise ValueError(
            "pickup locator rails must extend forward from the preserved rear boundary"
        )
    if (
        outer_width > plate_size[0] + 1.0e-12
        or stop_front_y > plate_front_y + 1.0e-12
    ):
        raise ValueError("pickup locator walls must fit on the top plate")
    locator_z = plate_top + wall_height / 2.0
    rail_center_y = rail_rear_y + rail_length / 2.0
    rail_center_offset_x = slot_width / 2.0 + wall_thickness / 2.0
    fixture_specs = (
        (
            "pickup_locator_left",
            (wall_thickness, rail_length, wall_height),
            (plug_center[0] - rail_center_offset_x, rail_center_y, locator_z),
            "plug_yaw_locator_left",
        ),
        (
            "pickup_locator_right",
            (wall_thickness, rail_length, wall_height),
            (plug_center[0] + rail_center_offset_x, rail_center_y, locator_z),
            "plug_yaw_locator_right",
        ),
        (
            "pickup_locator_front",
            (outer_width, wall_thickness, wall_height),
            (plug_center[0], stop_center_y, locator_z),
            "plug_forward_stop",
        ),
    )
    for name, size, center, role in fixture_specs:
        _cube(
            stage,
            f"/Table/{name}",
            size,
            center,
            (0.13, 0.38, 0.57),
            material,
            task["contact"],
        ).SetCustomDataByKey("rj45:role", role)
    root.SetCustomDataByKey("rj45:nest_scope", nest["scope"])
    stage.GetRootLayer().Save()
    fixture_prim_paths = [
        "/Table/pickup_nest",
        "/Table/pickup_nest_plate",
        "/Table/pickup_locator_left",
        "/Table/pickup_locator_right",
        "/Table/pickup_locator_front",
    ]
    return {
        "path": str(output),
        "pickup_nest_world_center_m": list(nest_center),
        "pickup_nest_size_m": list(nest_size),
        "pickup_nest_top_height_m": plate_top,
        "pickup_nest_scope": nest["scope"],
        "pickup_fixture_prim_paths": fixture_prim_paths,
        "pickup_fixture_scene_prim_paths": [
            path.replace("/Table", "/World/Table", 1)
            for path in fixture_prim_paths
        ],
        "pickup_nest_plate_world_center_m": list(plate_center),
        "pickup_nest_plate_size_m": list(plate_size),
        "pickup_locator_slot_width_m": slot_width,
        "pickup_locator_nominal_side_clearance_m": side_clearance,
        "pickup_locator_nominal_front_clearance_m": front_clearance,
        "pickup_locator_wall_thickness_m": wall_thickness,
        "pickup_locator_wall_height_m": wall_height,
        "pickup_locator_rail_y_range_m": [rail_rear_y, stop_inner_y],
        "pickup_locator_front_stop_y_range_m": [stop_inner_y, stop_front_y],
        "pickup_locator_open_top": True,
        "pickup_locator_attachment_joints": 0,
    }


def _set_pose(prim: Usd.Prim, position: list[float], quaternion: list[float]) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddOrientOp().Set(Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:])))


def _rotate_wxyz(quaternion: list[float], vector: tuple[float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    vx, vy, vz = vector
    # Unit-quaternion vector rotation written without an Isaac dependency.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _range_corners(bounds: Gf.Range3d) -> list[tuple[float, float, float]]:
    minimum = bounds.GetMin()
    maximum = bounds.GetMax()
    return [
        (x, y, z)
        for x in (float(minimum[0]), float(maximum[0]))
        for y in (float(minimum[1]), float(maximum[1]))
        for z in (float(minimum[2]), float(maximum[2]))
    ]


def _finger_bounds_at_grasp(
    robot_stage: Usd.Stage,
    side: str,
    frame: dict[str, Any],
    closure_magnitude_m: float,
) -> dict[str, Any]:
    wrist = robot_stage.GetPrimAtPath(f"/Robot/link_{side}_arm_6")
    if not wrist.IsValid():
        raise ValueError(f"missing {side} wrist in generated robot USD")
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    default_bounds: list[Gf.Range3d] = []
    for index in (1, 2):
        finger = robot_stage.GetPrimAtPath(
            f"/Robot/{side}_gripper/ee_finger_{side[0]}{index}"
        )
        if not finger.IsValid():
            raise ValueError(f"missing {side} finger {index} in generated robot USD")
        default_bounds.append(bbox_cache.ComputeRelativeBound(finger, wrist).ComputeAlignedRange())

    combined = Gf.Range3d()
    for bounds in default_bounds:
        combined.UnionWith(bounds)
    grasp_center = combined.GetMidpoint()
    quaternion = frame["quaternion_wxyz"]
    position = frame["position_m"]
    transformed: list[tuple[float, float, float]] = []
    serialized_default = []
    for bounds in default_bounds:
        serialized_default.append([list(bounds.GetMin()), list(bounds.GetMax())])
        midpoint_x = float(bounds.GetMidpoint()[0])
        if abs(midpoint_x) <= 1.0e-9:
            raise ValueError(f"{side} finger closure side cannot be inferred from its USD bounds")
        # The authored USD pose is the closed pose.  Runtime magnitude moves
        # each finger away from the centre along its own side of local X.
        closure_shift_x = math.copysign(float(closure_magnitude_m), midpoint_x)
        for corner in _range_corners(bounds):
            relative = (
                corner[0] + closure_shift_x - float(grasp_center[0]),
                corner[1] - float(grasp_center[1]),
                corner[2] - float(grasp_center[2]),
            )
            rotated = _rotate_wxyz(quaternion, relative)
            transformed.append(
                tuple(float(position[axis]) + rotated[axis] for axis in range(3))
            )
    minimum = [min(point[axis] for point in transformed) for axis in range(3)]
    maximum = [max(point[axis] for point in transformed) for axis in range(3)]
    closure_axis = _rotate_wxyz(quaternion, (1.0, 0.0, 0.0))
    return {
        "default_finger_bounds_in_wrist_m": serialized_default,
        "grasp_center_in_wrist_m": list(grasp_center),
        "commanded_closure_magnitude_m": float(closure_magnitude_m),
        "closure_axis_in_object": list(closure_axis),
        "finger_bounds_in_object_at_command_m": [minimum, maximum],
    }


def audit_gripper_clearance(
    robot_path: Path, task: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed when nominal fingers obstruct the RJ45 insertion corridor."""

    stage = Usd.Stage.Open(str(robot_path))
    if stage is None:
        raise ValueError(f"cannot open generated robot USD: {robot_path}")
    controller = task["controller"]
    left = _finger_bounds_at_grasp(
        stage,
        "left",
        task["frames"]["left_grasp"],
        float(controller["left_gripper_grasp_magnitude_m"]),
    )
    right = _finger_bounds_at_grasp(
        stage,
        "right",
        task["frames"]["right_grasp"],
        float(controller["right_gripper_grasp_magnitude_m"]),
    )
    insertion_axis_index = 1
    max_axis_projection = 0.1
    left_axis_projection = abs(left["closure_axis_in_object"][insertion_axis_index])
    right_axis_projection = abs(right["closure_axis_in_object"][insertion_axis_index])
    if left_axis_projection > max_axis_projection:
        raise ValueError("left gripper closes along the RJ45 insertion axis and would grasp front-to-back")
    if right_axis_projection > max_axis_projection:
        raise ValueError("right gripper closes along the RJ45 insertion axis")

    entry_y = float(task["frames"]["socket_entry"]["position_m"][1])
    left_front_edge_y = float(left["finger_bounds_in_object_at_command_m"][0][1])
    left_port_approach_clearance = left_front_edge_y - entry_y
    plug_tip_y = float(task["frames"]["plug_tip"]["position_m"][1])
    right_front_edge_y = float(right["finger_bounds_in_object_at_command_m"][1][1])
    inserted_port_plane_in_plug_y = plug_tip_y - float(task["evaluation"]["assembled_depth_m"])
    right_insertion_clearance = inserted_port_plane_in_plug_y - right_front_edge_y
    required_clearance = float(task["contact"]["contact_offset_m"])
    if left_port_approach_clearance <= required_clearance:
        raise ValueError("left finger envelope obstructs the socket port approach")
    if right_insertion_clearance <= required_clearance:
        raise ValueError("right finger envelope reaches the socket face before assembled depth")

    left.update(
        {
            "front_edge_y_m": left_front_edge_y,
            "socket_port_front_y_m": entry_y,
            "port_approach_clearance_m": left_port_approach_clearance,
            "closure_axis_projection_on_insertion": left_axis_projection,
        }
    )
    right.update(
        {
            "front_edge_y_m": right_front_edge_y,
            "plug_tip_y_m": plug_tip_y,
            "socket_plane_in_plug_at_assembled_depth_y_m": inserted_port_plane_in_plug_y,
            "clearance_after_assembled_depth_m": right_insertion_clearance,
            "closure_axis_projection_on_insertion": right_axis_projection,
        }
    )
    return {
        "source": "generated robot USD finger AABBs relative to wrist, transformed by configured grasp pose and commanded closure",
        "robot_usd_sha256": _sha256(robot_path),
        "insertion_axis": "+Y",
        "required_clearance_m": required_clearance,
        "max_closure_axis_projection_on_insertion": max_axis_projection,
        "left": left,
        "right": right,
        "passed": True,
    }


def build_rj45_scene(project: Path, config_path: Path | None = None) -> dict[str, Any]:
    project = Path(project).resolve()
    config_path = (
        Path(config_path).resolve()
        if config_path is not None
        else project / "configs/tasks/rj45-v1.json"
    )
    task = load_rj45_task(config_path)
    mesh_path, mesh = load_jack_mesh(project, task)
    asset_paths = {
        name: _resolve_inside(project, task["assets"][name])
        for name in ("robot", "socket", "plug", "table", "build_report")
    }
    output_dirs = {path.parent for path in asset_paths.values()}
    if len(output_dirs) != 1:
        raise ValueError("RJ45 generated assets and build report must share one output directory")
    output_dir = output_dirs.pop()
    output_dir.mkdir(parents=True, exist_ok=True)

    robot = build_robot(project / "assets/cache", asset_paths["robot"], task["contact"])
    gripper_clearance = audit_gripper_clearance(asset_paths["robot"], task)
    socket = build_socket(asset_paths["socket"], task, mesh)
    plug = build_plug(asset_paths["plug"], task)
    table = build_table(asset_paths["table"], task)

    scene_path = output_dir / "scene.usda"
    stage, world = _stage(scene_path, "World")
    for name in ("Robot", "Table", "Socket", "Plug"):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        _reference(prim, asset_paths[name.lower()], scene_path)
        if name in ("Socket", "Plug"):
            settings = task[name.lower()]
            _set_pose(prim, settings["initial_position_m"], settings["initial_orientation_wxyz"])
    world.SetCustomDataByKey("rj45:task_id", task.get("task_id", "rj45-v1"))
    world.SetCustomDataByKey("rj45:attachment_joints", 0)
    stage.GetRootLayer().Save()

    plug_nose_bottom_world = task["plug"]["initial_position_m"][2] - plug["nose_height_m"] / 2.0
    nest_top = table["pickup_nest_top_height_m"]
    if abs(plug_nose_bottom_world - nest_top) > 1.0e-9:
        raise ValueError("pickup nest top must exactly support the plug nose bottom")

    generated_files = [asset_paths[name] for name in ("robot", "socket", "plug", "table")]
    generated_files.append(scene_path)
    transform = task["geometry"]["jack_cad_transform"]
    report = {
        "schema_version": 1,
        "task_id": task.get("task_id", "rj45-v1"),
        "scope": task["scope"],
        "parameter_provenance": task["parameter_provenance"],
        "modeling_scope": {
            "mechanical_task": "bimanual physical insertion of an RJ45 plug into a handheld LAN-adapter housing",
            "included": ["rigid contact", "free-body grasping", "compound female-port guide", "visual vendor jack"],
            "excluded": task["geometry"]["model_limits"],
        },
        "robot": robot,
        "gripper_clearance_audit": gripper_clearance,
        "parts": {
            "socket": socket,
            "plug": plug,
            "table": table,
            "free_rigid_objects": 2,
            "attachment_joints": 0,
        },
        "socket": socket,
        "plug": plug,
        "table": table,
        "free_rigid_objects": 2,
        "attachment_joints": 0,
        "config_sha256": task["_config_sha256"],
        "source_cad_mesh": {
            "path": str(mesh_path.relative_to(project)),
            "file_sha256": _sha256(mesh_path),
            "source_step_sha256": mesh["source_sha256"],
            "source_model_name": mesh.get("source_model_name"),
            "coordinate_space": mesh.get("coordinate_space"),
            "source_bbox_m": mesh.get("bbox_m"),
            "transformed_bbox_m": socket["vendor_cad"]["transformed_bbox_m"],
        },
        "source_transform": {
            "value": transform,
            "sha256": _canonical_sha256(transform),
        },
        "drawing_source": {
            "plug": task["geometry"]["dimension_sources"]["plug_max_envelope"],
            "detail_limit": task["geometry"]["dimension_sources"]["nose_length_and_contact_detail"],
        },
        "pickup_contact_check": {
            "plug_nose_bottom_world_m": plug_nose_bottom_world,
            "nest_top_world_m": nest_top,
            "difference_m": plug_nose_bottom_world - nest_top,
        },
        "static_stage_created": True,
        "physics_executed": False,
        "files": {path.name: _sha256(path) for path in sorted(generated_files)},
    }
    asset_paths["build_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


__all__ = [
    "build_plug",
    "build_rj45_scene",
    "build_socket",
    "build_table",
    "audit_gripper_clearance",
    "load_jack_mesh",
]
