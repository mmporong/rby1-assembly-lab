"""OpenUSD 자산 조합. 물리 실행 없이도 치수·조인트·충돌 구성을 검사한다."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade


def _stage(path: Path, name: str) -> tuple[Usd.Stage, Usd.Prim]:
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, f"/{name}").GetPrim()
    stage.SetDefaultPrim(root)
    return stage, root


def _reference(prim: Usd.Prim, asset: Path, output: Path) -> None:
    if not asset.is_file():
        raise FileNotFoundError(f"Missing asset: {asset}; run scripts/fetch_assets.py")
    relative = Path(os.path.relpath(asset.resolve(), output.parent.resolve())).as_posix()
    prim.GetReferences().AddReference(relative)


def _material(stage: Usd.Stage, path: str, params: dict) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(float(params["static_friction"]))
    physics.CreateDynamicFrictionAttr(float(params["dynamic_friction"]))
    physics.CreateRestitutionAttr(float(params["restitution"]))
    return material


def _collision(prim: Usd.Prim, material: UsdShade.Material, params: dict) -> None:
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    if prim.IsA(UsdGeom.Mesh):
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
    # PhysX 스키마는 Isaac Sim에서 해석하며 standalone USD에서는 속성으로 보관한다.
    prim.AddAppliedSchema("PhysxCollisionAPI")
    prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(params["contact_offset_m"])
    prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(params["rest_offset_m"])
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")


def _cube(stage: Usd.Stage, path: str, size_m: tuple, center_m: tuple, color: tuple,
          material: UsdShade.Material, contact: dict) -> Usd.Prim:
    if len(size_m) != 3 or min(size_m) <= 0:
        raise ValueError("Cube dimensions must be positive metres")
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center_m))
    cube.AddScaleOp().Set(Gf.Vec3f(*size_m))
    cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    _collision(cube.GetPrim(), material, contact)
    return cube.GetPrim()


def build_robot(cache: Path, output: Path, contact: dict) -> dict:
    stage, root = _stage(output, "Robot")
    _reference(root, cache / "rby1/model_v_1_2_m.usd", output)
    mount_config = json.loads((cache / "rby1/gripper/gripper.json").read_text())
    material = _material(stage, "/Robot/AssemblyGripperMaterial", contact)
    patched = []
    geometry = {}
    for side in ("left", "right"):
        child = UsdGeom.Xform.Define(stage, f"/Robot/{side}_gripper").GetPrim()
        _reference(child, cache / f"rby1/gripper/rb_gripper_{side}.usd", output)
        # 원본은 visual mesh만 포함한다. 참조를 유지하고 인스턴스만 풀어 원본 형상에 충돌을 부여한다.
        for _ in range(4):
            instances = [p for p in Usd.PrimRange(child) if p.IsInstance()]
            if not instances:
                break
            for prim in instances:
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(child):
            if prim.IsA(UsdGeom.Mesh):
                _collision(prim, material, contact)
                patched.append(str(prim.GetPath()))
            if prim != root and prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        spec = mount_config[side]
        body_path = f"/Robot/{side}_gripper/{spec['body']}"
        wrist_path = f"/Robot/link_{side}_arm_6"
        wrist = UsdGeom.Xformable(stage.GetPrimAtPath(wrist_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ee = UsdGeom.Xformable(stage.GetPrimAtPath(body_path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        current_root = UsdGeom.Xformable(child).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        ee_local = ee * current_root.GetInverse()
        mount = Gf.Matrix4d(1.0)
        mount.SetRotate(Gf.Quatd(spec["mount_rot"][0], Gf.Vec3d(*spec["mount_rot"][1:])))
        mount.SetTranslateOnly(Gf.Vec3d(*spec["mount_pos"]))
        child_xform = UsdGeom.Xformable(child)
        child_xform.ClearXformOpOrder()
        child_xform.AddTransformOp().Set(ee_local.GetInverse() * mount * wrist)
        joint = UsdPhysics.FixedJoint.Define(stage, f"/Robot/joints/tool_{side}")
        joint.CreateBody0Rel().SetTargets([wrist_path])
        joint.CreateBody1Rel().SetTargets([body_path])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*spec["mount_pos"]))
        joint.CreateLocalRot0Attr(Gf.Quatf(spec["mount_rot"][0], Gf.Vec3f(*spec["mount_rot"][1:])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1))
        # 원본 mimic 정의와 과도한 finger2 힘을 그대로 학습 한계로 사용하지 않는다.
        for prim in Usd.PrimRange(child):
            if prim.IsA(UsdPhysics.PrismaticJoint):
                drive = UsdPhysics.DriveAPI.Get(prim, "linear")
                drive.CreateMaxForceAttr(75.0)
                drive.CreateStiffnessAttr(1000.0)
                drive.CreateDampingAttr(50.0)
        # mesh-local 점만 읽으면 finger body의 장착 변환이 빠진다.
        # 파지점은 합성된 USD에서 손목에 대한 두 손가락의 경계로 계산한다.
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        wrist_prim = stage.GetPrimAtPath(wrist_path)
        finger_bounds = [bbox_cache.ComputeRelativeBound(
            stage.GetPrimAtPath(f"/Robot/{side}_gripper/ee_finger_{side[0]}{index}"),
            wrist_prim).ComputeAlignedRange() for index in (1, 2)]
        combined = Gf.Range3d()
        for bounds in finger_bounds:
            combined.UnionWith(bounds)
        geometry[side] = {
            "grasp_center_in_wrist_m": list(combined.GetMidpoint()),
            "finger_span_m": float(combined.GetSize()[2]),
            "bounds_in_wrist_m": [list(combined.GetMin()), list(combined.GetMax())],
        }
    if len(patched) != 6:
        raise ValueError(f"Expected original palm/finger meshes on both hands, got {len(patched)}")
    fixed = UsdPhysics.FixedJoint.Define(stage, "/Robot/joints/assembly_fixed_base")
    fixed.CreateBody1Rel().SetTargets(["/Robot/base"])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(0))
    fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
    root.SetCustomDataByKey("assembly:variant", "RB-Y1 M v1.2")
    root.SetCustomDataByKey("assembly:gripper_collision", "original_visual_mesh_convex_hull_override")
    root.SetCustomDataByKey("assembly:base_mode", "fixed_for_bimanual_assembly")
    stage.GetRootLayer().Save()
    left_center = geometry["left"]["grasp_center_in_wrist_m"]
    right_center = geometry["right"]["grasp_center_in_wrist_m"]
    if any(abs(a - b) > 1e-6 for a, b in zip(left_center, right_center)):
        raise ValueError("The common grasp center requires symmetric left/right grippers")
    return {"path": str(output), "gripper_colliders_added": patched, "fixed_base": True,
            "grasp_attachment_joints": 0, "runtime_physics_verified": False,
            "grasp_center_in_wrist_m": left_center,
            "gripper_finger_span_m": geometry["left"]["finger_span_m"],
            "grasp_center_source": "composed_USD_finger_bounds_relative_to_wrist_at_authored_joint_pose",
            "gripper_geometry": geometry}


def build_parts(output_dir: Path, task: dict) -> dict:
    contact = task["contact"]
    socket, plug = task["socket"], task["plug"]
    width, height, length = socket["outer_width_m"], socket["outer_height_m"], socket["length_m"]
    hole_w, hole_h = socket["hole_width_m"], socket["hole_height_m"]
    if not (0 < plug["shaft_width_m"] < hole_w < width and 0 < plug["shaft_height_m"] < hole_h < height):
        raise ValueError("Plug, hole, and socket outer dimensions must have positive clearance")
    path = output_dir / "socket.usda"
    stage, root = _stage(path, "Socket")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(socket["mass_kg"])
    material = _material(stage, "/Socket/Material", contact)
    side_width = (width - hole_w) / 2.0
    cap_height = (height - hole_h) / 2.0
    for name, sign in (("left", -1), ("right", 1)):
        _cube(stage, f"/Socket/{name}", (side_width, length, height),
              (sign * (hole_w + side_width) / 2.0, 0, 0), (0.16, 0.38, 0.54), material, contact)
    for name, sign in (("bottom", -1), ("top", 1)):
        _cube(stage, f"/Socket/{name}", (hole_w, length, cap_height),
              (0, 0, sign * (hole_h + cap_height) / 2.0), (0.16, 0.38, 0.54), material, contact)
    root.SetCustomDataByKey("assembly:hole_axis", "+Y")
    root.SetCustomDataByKey("assembly:hole_width_m", hole_w)
    root.SetCustomDataByKey("assembly:hole_height_m", hole_h)
    stage.GetRootLayer().Save()
    path = output_dir / "plug.usda"
    stage, root = _stage(path, "Plug")
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(plug["mass_kg"])
    material = _material(stage, "/Plug/Material", contact)
    _cube(stage, "/Plug/grip", (plug["grip_width_m"], plug["grip_length_m"], plug["grip_height_m"]),
          (0, 0, 0), (0.72, 0.48, 0.18), material, contact)
    shaft_y = (plug["grip_length_m"] + plug["shaft_length_m"]) / 2.0
    _cube(stage, "/Plug/shaft", (plug["shaft_width_m"], plug["shaft_length_m"], plug["shaft_height_m"]),
          (0, shaft_y, 0), (0.8, 0.66, 0.38), material, contact)
    tip_y = plug["grip_length_m"] / 2.0 + plug["shaft_length_m"]
    root.SetCustomDataByKey("assembly:tip_y_m", tip_y)
    stage.GetRootLayer().Save()
    path = output_dir / "table.usda"
    stage, root = _stage(path, "Table")
    material = _material(stage, "/Table/Material", contact)
    table = task["table"]
    z_m = table["top_height_m"] - table["size_m"][2] / 2.0
    _cube(stage, "/Table/top", tuple(table["size_m"]), (*table["center_xy_m"], z_m),
          (0.67, 0.7, 0.73), material, contact)
    stage.GetRootLayer().Save()
    return {"clearance_per_side_m": {"x": (hole_w - plug["shaft_width_m"]) / 2,
                                    "z": (hole_h - plug["shaft_height_m"]) / 2},
            "socket_entry_y_m": -length / 2.0, "plug_tip_y_m": tip_y,
            "socket_collision_boxes": 4, "free_rigid_objects": 2,
            "parameter_provenance": task["parameter_provenance"]}


def build_scene(project: Path) -> dict:
    project = project.resolve()
    task_file = project / "configs/task.json"
    task = json.loads(task_file.read_text())
    generated = project / "assets/generated"
    robot = build_robot(project / "assets/cache", generated / "robot.usda", task["contact"])
    parts = build_parts(generated, task)
    path = generated / "scene.usda"
    stage, world = _stage(path, "World")
    for name in ("Robot", "Table", "Socket", "Plug"):
        prim = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        _reference(prim, generated / f"{name.lower()}.usda", path)
        if name in ("Socket", "Plug"):
            UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*task[name.lower()]["initial_position_m"]))
    stage.GetRootLayer().Save()
    result = {"schema_version": 1, "robot": robot, "parts": parts,
              "config_sha256": hashlib.sha256(task_file.read_bytes()).hexdigest(),
              "static_stage_created": True, "physics_executed": False,
              "files": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(generated.glob("*.usda"))}}
    (generated / "build-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
