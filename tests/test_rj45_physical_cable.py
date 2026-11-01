from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "rj45_physical_cable.py"
SPEC = importlib.util.spec_from_file_location("rj45_physical_cable", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_physical_contract_has_exact_mass_geometry_and_joint_counts():
    config = MODULE.PhysicalCableConfig()
    assert config.link_count == 24
    assert config.total_length_m == pytest.approx(0.35)
    assert config.radius_m == pytest.approx(0.0024)
    assert config.cylinder_height_m == pytest.approx(0.35 / 24 - 2 * 0.0024)
    assert config.link_mass_kg == pytest.approx(0.0105 / 24)
    assert config.link_mass_kg * 24 == pytest.approx(0.0105)


def test_initial_chain_preserves_every_capsule_end_to_end_length():
    config = MODULE.PhysicalCableConfig()
    points = MODULE.initial_boundaries((0.5, 0.0, 0.79), (0.0, -1.0, 0.0), config)
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    assert points.shape == (25, 3)
    assert np.isfinite(points).all()
    assert lengths == pytest.approx(np.full(24, config.segment_length_m), abs=1e-12)
    assert lengths.sum() == pytest.approx(0.35)
    assert points[-1, 0] > points[0, 0]


def test_attachment_is_behind_visual_stub_and_uses_task_dimensions():
    task = {"geometry": {"plug_profile": {"total_length_m": 0.021}},
            "frames": {"plug_tip": {"position_m": [0.0, 0.03, 0.0]}}}
    assert MODULE.plug_attachment_local(task) == pytest.approx((0.0, -0.001, 0.0005))


@pytest.mark.parametrize(
    ("anchor", "direction", "message"),
    [
        ((0.5, 0.0, 0.70), (0.0, -1.0, 0.0), "below"),
        ((0.5, 0.0, 0.79), (0.0, 0.0, 0.0), "non-zero"),
        ((0.5, 0.0, 0.79), (np.nan, -1.0, 0.0), "finite"),
    ],
)
def test_initial_boundaries_reject_invalid_floor_and_exit(anchor, direction, message):
    with pytest.raises(ValueError, match=message):
        MODULE.initial_boundaries(anchor, direction, MODULE.PhysicalCableConfig())


def test_explicit_straight_boundaries_are_exact_and_validation_rejects_drift():
    config = MODULE.PhysicalCableConfig(table_top_z_m=-10.0)
    points = MODULE.straight_boundaries((0.0, -0.5, 1.0), (1.0, 0.0, 0.0), config)
    assert points[:, 1:] == pytest.approx(np.tile((-0.5, 1.0), (25, 1)))
    assert np.diff(points[:, 0]) == pytest.approx(np.full(24, config.segment_length_m))
    points[8, 0] += 1.0e-5
    with pytest.raises(ValueError, match="end-to-end"):
        MODULE._validate_boundaries(points, config)


def test_tensor_view_order_fails_closed():
    class View:
        prim_paths = ["/B", "/A"]

    with pytest.raises(RuntimeError, match="order mismatch"):
        MODULE.assert_view_paths(View(), ["/A", "/B"])


def test_usd_session_layer_topology_and_straight_joint_frames():
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    plug = UsdGeom.Xform.Define(stage, "/World/envs/env_0/Plug").GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(plug)
    UsdPhysics.MassAPI.Apply(plug).CreateMassAttr(0.004)
    root_before = stage.GetRootLayer().ExportToString()
    config = MODULE.PhysicalCableConfig(table_top_z_m=-10.0)
    boundaries = MODULE.straight_boundaries((0.5, 0.0, 0.79), (1.0, 0.0, 0.0), config)
    handle = MODULE.author_physical_cable(
        stage, config.plug_prim_path, boundaries[0], (1.0, 0.0, 0.0), config,
        (0.5, 0.0, 0.79), boundaries,
    )
    assert stage.GetRootLayer().ExportToString() == root_before
    assert len(handle.link_paths) == 24 and len(handle.joint_paths) == 24
    assert UsdPhysics.MassAPI(plug).GetMassAttr().Get() == pytest.approx(0.004)
    for path in handle.link_paths:
        body = stage.GetPrimAtPath(path)
        assert body.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.MassAPI(body).GetMassAttr().Get() == pytest.approx(0.0004375)
        capsule = UsdGeom.Capsule.Get(stage, f"{path}/Collider")
        assert capsule.GetAxisAttr().Get() == UsdGeom.Tokens.y
        assert capsule.GetRadiusAttr().Get() == pytest.approx(0.0024)
        assert capsule.GetHeightAttr().Get() == pytest.approx(0.35 / 24 - 0.0048)
        orient = UsdGeom.Xformable(body).GetOrderedXformOps()[1].GetAttr()
        assert orient.GetTypeName() == "quatf"
    for path in handle.joint_paths:
        joint = UsdPhysics.SphericalJoint.Get(stage, path)
        assert joint.GetCollisionEnabledAttr().Get() is False
        assert joint.GetExcludeFromArticulationAttr().Get() is True
        body0 = stage.GetPrimAtPath(str(joint.GetBody0Rel().GetTargets()[0]))
        body1 = stage.GetPrimAtPath(str(joint.GetBody1Rel().GetTargets()[0]))
        matrix0 = UsdGeom.Xformable(body0).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        matrix1 = UsdGeom.Xformable(body1).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        anchor0 = matrix0.Transform(Gf.Vec3d(*joint.GetLocalPos0Attr().Get()))
        anchor1 = matrix1.Transform(Gf.Vec3d(*joint.GetLocalPos1Attr().Get()))
        # Joint frames are Vec3f/Quatf in USD; this bound covers only float schema quantization.
        assert Gf.GetLength(anchor0 - anchor1) < 1.0e-7


def test_handle_retains_shared_simulation_view_and_checks_order():
    class RigidView:
        def __init__(self, paths):
            self.prim_paths = paths

    class SimulationView:
        def __init__(self):
            self.calls = []

        def create_rigid_body_view(self, paths):
            self.calls.append(paths)
            return RigidView(paths)

    config = MODULE.PhysicalCableConfig()
    links = [f"/Cable/link_{index:02d}" for index in range(24)]
    handle = MODULE.PhysicalCableHandle(None, config, "/Plug", links, [], np.zeros((25, 3)), (0, 0, 0))
    simulation = SimulationView()
    assert handle.bind_simulation(simulation_view=simulation) is handle
    assert handle._simulation_view is simulation
    assert simulation.calls == [["/Plug", *links]]
