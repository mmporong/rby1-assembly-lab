import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rby1_assembly.assets import (
    AssetError,
    AssetRecord,
    audit_automate_data,
    audit_generated_assets,
    audit_usd,
    fetch_asset,
    load_manifest,
)


class _Response(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()


class AssetManifestTests(unittest.TestCase):
    def test_project_manifest_is_valid_and_pinned(self) -> None:
        manifest = load_manifest(Path("configs/assets.json"))
        downloads = [asset for asset in manifest.assets if asset.kind == "download"]
        self.assertEqual(len(downloads), 17)
        self.assertTrue(all(asset.sha256 and asset.size_bytes for asset in downloads))
        self.assertFalse(any(asset.redistributable for asset in downloads))
        trajectory = manifest.by_id()["automate_00015_disassembly_trajectories"]
        self.assertEqual(trajectory.physical_role, "training_data")

    def test_manifest_rejects_cache_escape(self) -> None:
        payload = {
            "schema_version": 1,
            "cache_root": "assets/cache",
            "sources": {"source": {}},
            "assets": [{
                "id": "bad",
                "kind": "download",
                "physical_role": "configuration",
                "source": "source",
                "url": "https://raw.githubusercontent.com/org/repo/rev/file",
                "path": "../escape",
                "size_bytes": 1,
                "sha256": "0" * 64,
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assets.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(AssetError, "cache root"):
                load_manifest(path)


class AssetFetchTests(unittest.TestCase):
    def _asset(self, payload: bytes, sha256: str | None = None) -> AssetRecord:
        return AssetRecord(
            id="sample",
            kind="download",
            physical_role="configuration",
            source="test",
            url="https://raw.githubusercontent.com/org/repo/rev/file",
            path="sample/data.bin",
            size_bytes=len(payload),
            sha256=sha256 or hashlib.sha256(payload).hexdigest(),
        )

    def test_fetch_is_atomic_verified_and_cached(self) -> None:
        payload = b"pinned asset bytes"
        with tempfile.TemporaryDirectory() as directory:
            with patch("rby1_assembly.assets.urlopen", return_value=_Response(payload)) as mocked:
                first = fetch_asset(self._asset(payload), directory)
                second = fetch_asset(self._asset(payload), directory)
            self.assertEqual(first.status, "downloaded")
            self.assertEqual(second.status, "cached")
            self.assertEqual(first.path.read_bytes(), payload)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(list(first.path.parent.glob("*.tmp")), [])

    def test_fetch_rejects_checksum_mismatch_and_removes_temp(self) -> None:
        payload = b"corrupt"
        with tempfile.TemporaryDirectory() as directory:
            with patch("rby1_assembly.assets.urlopen", return_value=_Response(payload)):
                with self.assertRaisesRegex(AssetError, "SHA-256 mismatch"):
                    fetch_asset(self._asset(payload, "0" * 64), directory)
            self.assertFalse((Path(directory) / "sample" / "data.bin").exists())
            self.assertEqual(list(Path(directory).rglob("*.tmp")), [])

    def test_fetch_enforces_size_cap_before_network(self) -> None:
        payload = b"0123456789"
        with tempfile.TemporaryDirectory() as directory:
            with patch("rby1_assembly.assets.urlopen") as mocked:
                with self.assertRaisesRegex(AssetError, "exceeds size cap"):
                    fetch_asset(self._asset(payload), directory, max_bytes=5)
            mocked.assert_not_called()


class UsdAuditTests(unittest.TestCase):
    def test_audit_finds_articulation_joint_collision_and_mass(self) -> None:
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest("usd-core is not installed")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.usda"
            stage = Usd.Stage.CreateNew(str(path))
            root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
            stage.SetDefaultPrim(root)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage, 1.0)
            UsdPhysics.ArticulationRootAPI.Apply(root)
            body = UsdGeom.Cube.Define(stage, "/Robot/body").GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(body)
            UsdPhysics.CollisionAPI.Apply(body)
            mass = UsdPhysics.MassAPI.Apply(body)
            mass.CreateMassAttr(1.0)
            joint = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/joint")
            joint.CreateLowerLimitAttr(-1.0)
            joint.CreateUpperLimitAttr(1.0)
            stage.GetRootLayer().Save()

            report = audit_usd(path, "robot_articulation")
            self.assertTrue(report.passed, report.errors)
            self.assertEqual(report.default_prim, "/Robot")
            self.assertEqual(report.up_axis, "Z")
            self.assertEqual(report.joint_counts["PhysicsRevoluteJoint"], 1)
            self.assertEqual(report.collision_prim_count, 1)
            self.assertEqual(report.mass_or_density_prim_count, 1)


class AutoMateDataAuditTests(unittest.TestCase):
    def test_audit_checks_grasp_distance_and_trajectory_shape(self) -> None:
        episode = {
            "fingertip_centered_pos": [],
            "fingertip_centered_quat": [],
            "arm_dof_pos": [],
            "plug_grasp_pos": [0.0, 0.0, 0.0],
            "plug_grasp_quat": [1.0, 0.0, 0.0, 0.0],
            "init_plug_pos": [0.0, 0.0, 0.0],
            "init_plug_quat": [1.0, 0.0, 0.0, 0.0],
            "plug_pos": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]],
            "plug_quat": [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "automate"
            (root / "00015").mkdir(parents=True)
            (root / "plug_grasps.json").write_text(
                json.dumps({"asset_00015": [0.0, 0.0, 0.1, 1.0, 0.0, 0.0, 0.0]}),
                encoding="utf-8",
            )
            (root / "disassembly_dist.json").write_text(
                json.dumps({"asset_00015": 0.027}), encoding="utf-8"
            )
            (root / "00015" / "disassemble_traj.json").write_text(
                json.dumps([episode]), encoding="utf-8"
            )
            report = audit_automate_data(directory)
            self.assertTrue(report.passed, report.errors)
            self.assertEqual(report.episode_count, 1)
            self.assertEqual(report.trajectory_step_counts, (2,))

    def test_generated_project_assets_pass_when_built(self) -> None:
        generated = Path("assets/generated/robot.usda")
        if not generated.is_file():
            self.skipTest("generated project assets have not been built")
        manifest = load_manifest(Path("configs/assets.json"))
        report = audit_generated_assets(Path.cwd(), manifest)
        self.assertTrue(report.passed, report.errors)
        self.assertEqual(len(report.gripper_collider_paths), 6)
        self.assertEqual(report.convex_hull_collider_count, 6)
        self.assertTrue(all(mount["consistent"] for mount in report.tool_mounts.values()))
        self.assertEqual(len(report.free_rigid_objects), 2)
        self.assertTrue(report.socket_hole_clear)


if __name__ == "__main__":
    unittest.main()
