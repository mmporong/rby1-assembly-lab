import json
from pathlib import Path
import tempfile
import unittest

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from rby1_assembly.scene import build_parts, build_robot


class PartGeometryTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads((Path(__file__).parents[1] / "configs/task.json").read_text())

    def test_socket_hole_is_not_filled_by_collision(self):
        with tempfile.TemporaryDirectory() as directory:
            result = build_parts(Path(directory), self.task)
            self.assertAlmostEqual(result["clearance_per_side_m"]["x"], 0.002)
            stage = Usd.Stage.Open(str(Path(directory) / "socket.usda"))
            colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertEqual(len(colliders), 4)
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            for prim in colliders:
                bounds = cache.ComputeWorldBound(prim).ComputeAlignedRange()
                self.assertFalse(bounds.Contains(Gf.Vec3d(0, 0, 0)))
            self.assertTrue(stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(any(p.IsA(UsdPhysics.Joint) for p in stage.Traverse()))

    def test_negative_clearance_is_rejected(self):
        self.task["plug"]["shaft_width_m"] = self.task["socket"]["hole_width_m"]
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                build_parts(Path(directory), self.task)

    def test_grasp_center_includes_mount_and_finger_body_transform(self):
        cache = Path(__file__).parents[1] / "assets/cache"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "robot.usda"
            report = build_robot(cache, output, self.task["contact"])
            # 고정 source USD의 회귀 검사다. 물리 파지 성공을 검증하지 않는다.
            self.assertAlmostEqual(report["grasp_center_in_wrist_m"][2], -0.2306, places=6)
            self.assertAlmostEqual(report["gripper_finger_span_m"], 0.062, places=6)
            stage = Usd.Stage.Open(str(output))
            for side in ("left", "right"):
                wrist = UsdGeom.Xformable(stage.GetPrimAtPath(f"/Robot/link_{side}_arm_6"))
                palm = UsdGeom.Xformable(stage.GetPrimAtPath(f"/Robot/{side}_gripper/ee_{side}"))
                relative = palm.ComputeLocalToWorldTransform(0) * wrist.ComputeLocalToWorldTransform(0).GetInverse()
                self.assertAlmostEqual(relative.ExtractTranslation()[2], -0.1261, places=6)


if __name__ == "__main__":
    unittest.main()
