#!/usr/bin/env python3
"""PhysX rigid-link cable for the RJ45 task.

The cable is intentionally a separate experimental condition: it changes the
physics topology and must be attached before the first simulation reset.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class PhysicalCableConfig:
    link_count: int = 24
    total_length_m: float = 0.35
    radius_m: float = 0.0024
    total_mass_kg: float = 0.0105
    contact_offset_m: float = 0.0003
    rest_offset_m: float = 0.0
    static_friction: float = 0.6
    dynamic_friction: float = 0.4
    restitution: float = 0.0
    solver_position_iterations: int = 24
    solver_velocity_iterations: int = 4
    table_top_z_m: float = 0.75
    prim_path: str = "/World/PhysicalExperiments/RJ45Cable"
    plug_prim_path: str = "/World/envs/env_0/Plug"

    def __post_init__(self) -> None:
        if self.link_count != 24:
            raise ValueError("this physical cable prototype requires exactly 24 links")
        values = (self.total_length_m, self.radius_m, self.total_mass_kg, self.contact_offset_m)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError("cable dimensions, mass and contact offset must be positive and finite")
        if self.segment_length_m <= 2 * self.radius_m:
            raise ValueError("capsule segment is too short for its radius")
        if not self.prim_path.startswith("/") or not self.plug_prim_path.startswith("/"):
            raise ValueError("USD prim paths must be absolute")

    @property
    def segment_length_m(self) -> float:
        return self.total_length_m / self.link_count

    @property
    def cylinder_height_m(self) -> float:
        return self.segment_length_m - 2 * self.radius_m

    @property
    def link_mass_kg(self) -> float:
        return self.total_mass_kg / self.link_count


def plug_attachment_local(task: Mapping[str, Any]) -> np.ndarray:
    profile = task["geometry"]["plug_profile"]
    tip_y = float(task["frames"]["plug_tip"]["position_m"][1])
    return np.asarray((0.0, tip_y - float(profile["total_length_m"]) - 0.010, 0.0005))


def initial_boundaries(
    anchor: Sequence[float], exit_direction: Sequence[float], config: PhysicalCableConfig
) -> np.ndarray:
    """Return 25 equal-spaced endpoints in a gentle, initially slack curve."""
    anchor = np.asarray(anchor, dtype=np.float64)
    exit_direction = np.asarray(exit_direction, dtype=np.float64)
    if (
        anchor.shape != (3,)
        or exit_direction.shape != (3,)
        or not np.isfinite(anchor).all()
        or not np.isfinite(exit_direction).all()
    ):
        raise ValueError("anchor and exit direction must be finite three-vectors")
    norm = float(np.linalg.norm(exit_direction))
    if norm <= 1.0e-12:
        raise ValueError("exit direction must be non-zero")
    floor = config.table_top_z_m + config.radius_m
    if anchor[2] < floor:
        raise ValueError("cable anchor is below the configured table contact plane")
    exit_direction /= norm
    # Rotate gradually toward +X and down toward the table.  Each increment is
    # normalized independently, so every capsule retains its exact end length.
    points = [anchor.copy()]
    for index in range(config.link_count):
        t = index / max(config.link_count - 1, 1)
        target = np.asarray((0.78 * t, -(1.0 - 0.45 * t), -0.35 * math.sin(math.pi * t)))
        direction = (1.0 - min(t * 2.0, 1.0)) * exit_direction + min(t * 2.0, 1.0) * target
        direction /= np.linalg.norm(direction)
        next_point = points[-1] + config.segment_length_m * direction
        if next_point[2] < floor:
            horizontal = direction.copy()
            horizontal[2] = 0.0
            horizontal /= np.linalg.norm(horizontal)
            next_point = points[-1] + config.segment_length_m * horizontal
            next_point[2] = max(next_point[2], floor)
        points.append(next_point)
    return _validate_boundaries(points, config)


def straight_boundaries(
    anchor: Sequence[float], direction: Sequence[float], config: PhysicalCableConfig
) -> np.ndarray:
    anchor = np.asarray(anchor, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if anchor.shape != (3,) or direction.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(direction).all():
        raise ValueError("straight cable inputs must be finite three-vectors")
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        raise ValueError("straight cable direction must be non-zero")
    distances = np.arange(config.link_count + 1, dtype=np.float64)[:, None] * config.segment_length_m
    return _validate_boundaries(anchor + distances * (direction / norm), config)


def _validate_boundaries(points: Sequence[Sequence[float]], config: PhysicalCableConfig) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.shape != (config.link_count + 1, 3) or not np.isfinite(result).all():
        raise ValueError(f"boundaries must have shape ({config.link_count + 1}, 3) and be finite")
    lengths = np.linalg.norm(np.diff(result, axis=0), axis=1)
    if not np.allclose(lengths, config.segment_length_m, rtol=0.0, atol=1.0e-10):
        raise ValueError("every cable boundary interval must equal the capsule end-to-end length")
    return result


def assert_view_paths(view: Any, expected_paths: Sequence[str]) -> None:
    actual = [str(path) for path in view.prim_paths]
    expected = [str(path) for path in expected_paths]
    if actual != expected:
        raise RuntimeError(f"tensor rigid-body order mismatch: expected {expected}, got {actual}")


def _quat_y_to(direction: np.ndarray) -> tuple[float, float, float, float]:
    """Quaternion WXYZ rotating local +Y onto direction."""
    y = np.asarray((0.0, 1.0, 0.0))
    direction = direction / np.linalg.norm(direction)
    dot = float(np.dot(y, direction))
    if dot < -0.999999:
        return (0.0, 1.0, 0.0, 0.0)
    cross = np.cross(y, direction)
    q = np.asarray((1.0 + dot, *cross), dtype=np.float64)
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def _world_attachment(stage: Any, plug_path: str, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(plug_path)
    if not prim.IsValid():
        raise ValueError(f"plug prim does not exist: {plug_path}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    anchor = matrix.Transform(Gf.Vec3d(*local))
    origin = matrix.Transform(Gf.Vec3d(0.0, 0.0, 0.0))
    exit_point = matrix.Transform(Gf.Vec3d(0.0, -1.0, 0.0))
    return np.asarray(anchor), np.asarray(exit_point - origin)


def author_physical_cable(
    stage: Any,
    plug_path: str,
    anchor_world: Sequence[float],
    exit_world: Sequence[float],
    config: PhysicalCableConfig,
    plug_local_anchor: Sequence[float] = (0.0, 0.0, 0.0),
    boundaries: Sequence[Sequence[float]] | None = None,
) -> "PhysicalCableHandle":
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    points = initial_boundaries(anchor_world, exit_world, config) if boundaries is None else _validate_boundaries(boundaries, config)
    link_paths = [f"{config.prim_path}/Links/link_{i:02d}" for i in range(config.link_count)]
    joint_paths = [f"{config.prim_path}/Joints/plug_to_00"] + [
        f"{config.prim_path}/Joints/link_{i-1:02d}_to_{i:02d}" for i in range(1, config.link_count)
    ]
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        root = UsdGeom.Xform.Define(stage, config.prim_path).GetPrim()
        root.SetCustomDataByKey("rj45:physics_modified", True)
        root.SetCustomDataByKey("rj45:model", "24_capsule_passive_spherical_joint_cable")
        material = UsdShade.Material.Define(stage, f"{config.prim_path}/PhysicsMaterial")
        physics_material = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics_material.CreateStaticFrictionAttr(config.static_friction)
        physics_material.CreateDynamicFrictionAttr(config.dynamic_friction)
        physics_material.CreateRestitutionAttr(config.restitution)
        for i, path in enumerate(link_paths):
            direction = points[i + 1] - points[i]
            center = (points[i + 1] + points[i]) * 0.5
            w, x, y, z = _quat_y_to(direction)
            body = UsdGeom.Xform.Define(stage, path)
            body.AddTranslateOp().Set(Gf.Vec3d(*center))
            body.AddOrientOp().Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))
            body_prim = body.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body_prim)
            UsdPhysics.MassAPI.Apply(body_prim).CreateMassAttr(config.link_mass_kg)
            body_prim.AddAppliedSchema("PhysxRigidBodyAPI")
            body_prim.CreateAttribute("physxRigidBody:solverPositionIterationCount", Sdf.ValueTypeNames.UInt).Set(config.solver_position_iterations)
            body_prim.CreateAttribute("physxRigidBody:solverVelocityIterationCount", Sdf.ValueTypeNames.UInt).Set(config.solver_velocity_iterations)
            capsule = UsdGeom.Capsule.Define(stage, f"{path}/Collider")
            capsule.CreateAxisAttr(UsdGeom.Tokens.y)
            capsule.CreateRadiusAttr(config.radius_m)
            capsule.CreateHeightAttr(config.cylinder_height_m)
            UsdPhysics.CollisionAPI.Apply(capsule.GetPrim()).CreateCollisionEnabledAttr(True)
            capsule.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
            capsule.GetPrim().CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(config.contact_offset_m)
            capsule.GetPrim().CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(config.rest_offset_m)
            UsdShade.MaterialBindingAPI.Apply(capsule.GetPrim()).Bind(material, materialPurpose="physics")
        half = config.segment_length_m * 0.5
        for i, joint_path in enumerate(joint_paths):
            joint = UsdPhysics.SphericalJoint.Define(stage, joint_path)
            if i == 0:
                body0 = plug_path
                local0 = tuple(float(value) for value in plug_local_anchor)
            else:
                body0, local0 = link_paths[i - 1], (0.0, half, 0.0)
            body1 = link_paths[i]
            joint.CreateBody0Rel().SetTargets([body0])
            joint.CreateBody1Rel().SetTargets([body1])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*local0))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, -half, 0.0))
            joint.CreateCollisionEnabledAttr(False)
            joint.CreateExcludeFromArticulationAttr(True)
            filtered = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(body0))
            filtered.CreateFilteredPairsRel().AddTarget(body1)
    return PhysicalCableHandle(stage, config, plug_path, link_paths, joint_paths, points, plug_local_anchor)


class PhysicalCableHandle:
    def __init__(self, stage: Any, config: PhysicalCableConfig, plug_path: str, link_paths: list[str], joint_paths: list[str], boundaries: np.ndarray, plug_local_anchor: Sequence[float]) -> None:
        self.stage, self.config, self.plug_path = stage, config, plug_path
        self.link_paths, self.joint_paths, self.boundaries = link_paths, joint_paths, boundaries
        self.plug_local_anchor = np.asarray(plug_local_anchor, dtype=np.float64)
        self.physics_modified = True
        self._view = None
        self._simulation_view = None

    def describe(self) -> dict[str, Any]:
        return {"model": "24_capsule_passive_unlimited_spherical_joint_cable", "physics_coupling": True,
                "physics_coupling_status": "authored_model_intent_unverified_until_smoke_passes",
                "physics_modified": True, "simulation_assumptions": True, "link_count": 24, "joint_count": 24,
                "total_length_m": self.config.total_length_m, "radius_m": self.config.radius_m,
                "total_mass_kg": self.config.total_mass_kg, "link_mass_kg": self.config.link_mass_kg,
                "plug_mass_kg": "unchanged", "plug_path": self.plug_path, "link_paths": self.link_paths,
                "joint_paths": self.joint_paths, "adjacent_collision": False, "nonadjacent_collision": True,
                "robot_and_table_collision": True,
                "joint_drive": None, "joint_limits": None, "capsule_axis": "Y",
                "material": {"static_friction": self.config.static_friction,
                             "dynamic_friction": self.config.dynamic_friction,
                             "restitution": self.config.restitution},
                "contact_offsets_m": {"contact": self.config.contact_offset_m,
                                      "rest": self.config.rest_offset_m},
                "solver_iterations": {"position": self.config.solver_position_iterations,
                                      "velocity": self.config.solver_velocity_iterations}}

    def bind_simulation(self, frontend: str = "torch", simulation_view: Any | None = None) -> "PhysicalCableHandle":
        expected = [self.plug_path, *self.link_paths]
        if simulation_view is None:
            import omni.physics.tensors as tensors
            simulation_view = tensors.create_simulation_view(frontend)
            simulation_view.set_subspace_roots("/")
            simulation_view.reset()
        self._simulation_view = simulation_view
        self._view = simulation_view.create_rigid_body_view(expected)
        assert_view_paths(self._view, expected)
        return self

    def measure(self) -> dict[str, Any]:
        if self._view is None:
            raise RuntimeError("bind_simulation() must be called after the first simulation reset")
        def array(value: Any) -> np.ndarray:
            if hasattr(value, "detach"): value = value.detach().cpu()
            return np.asarray(value)
        transforms = array(self._view.get_transforms())
        velocities = array(self._view.get_velocities())
        masses = array(self._view.get_masses())
        if not all(np.isfinite(v).all() for v in (transforms, velocities, masses)):
            raise RuntimeError("non-finite physical cable tensor data")
        centers = transforms[1:, :3]
        def rotate_xyzw(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
            x, y, z, w = quaternion
            qvec = np.asarray((x, y, z))
            return vector + 2.0 * (w * np.cross(qvec, vector) + np.cross(qvec, np.cross(qvec, vector)))
        half = self.config.segment_length_m * 0.5
        plug_anchor = transforms[0, :3] + rotate_xyzw(transforms[0, 3:7], self.plug_local_anchor)
        lower = np.asarray([centers[i] + rotate_xyzw(transforms[i + 1, 3:7], np.asarray((0.0, -half, 0.0))) for i in range(self.config.link_count)])
        upper = np.asarray([centers[i] + rotate_xyzw(transforms[i + 1, 3:7], np.asarray((0.0, half, 0.0))) for i in range(self.config.link_count)])
        gaps = [float(np.linalg.norm(plug_anchor - lower[0]))]
        gaps.extend(float(np.linalg.norm(upper[i - 1] - lower[i])) for i in range(1, self.config.link_count))
        return {"finite": True, "body_count": int(transforms.shape[0]), "center_of_mass_m": np.average(centers, axis=0, weights=masses[1:].reshape(-1)).tolist(),
                "polyline_center_length_m": float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum()),
                "max_joint_anchor_gap_m": float(max(gaps)), "max_linear_speed_m_s": float(np.linalg.norm(velocities[:, :3], axis=1).max()),
                "max_angular_speed_rad_s": float(np.linalg.norm(velocities[:, 3:], axis=1).max()),
                "plug_plus_cable_mass_kg": float(masses.sum()), "cable_mass_kg": float(masses[1:].sum())}


def attach_physical_cable(env: Any, stage: Any, config: PhysicalCableConfig | None = None) -> PhysicalCableHandle:
    config = config or PhysicalCableConfig()
    local = plug_attachment_local(env.cfg.rj45_task)
    anchor, exit_direction = _world_attachment(stage, config.plug_prim_path, local)
    handle = author_physical_cable(stage, config.plug_prim_path, anchor, exit_direction, config, local)
    return handle
