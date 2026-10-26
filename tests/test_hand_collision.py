import hashlib
from pathlib import Path
import unittest

from rby1_assembly.hand_collision import (
    EXPECTED_RIGID_BODY_COUNT,
    LEFT_HAND_BODY_PATHS,
    RIGHT_HAND_BODY_PATHS,
    configure_selective_hand_self_collisions,
    make_hand_collision_env_type,
)


class SelectiveHandCollisionUsdTests(unittest.TestCase):
    def setUp(self):
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("usd-core is not installed")
        self.Usd = Usd
        self.UsdPhysics = UsdPhysics

    def test_generated_robot_enables_only_nine_cross_hand_pairs(self):
        path = Path("assets/generated/robot.usda")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        stage = self.Usd.Stage.Open(str(path))
        report = configure_selective_hand_self_collisions(stage, "/Robot")

        root = stage.GetPrimAtPath("/Robot")
        self.assertIs(root.GetAttribute("physxArticulation:enabledSelfCollisions").Get(), True)
        bodies = sorted(
            str(prim.GetPath())
            for prim in self.Usd.PrimRange(root)
            if prim.HasAPI(self.UsdPhysics.RigidBodyAPI)
        )
        self.assertEqual(len(bodies), EXPECTED_RIGID_BODY_COUNT)
        filtered = set()
        for source in bodies:
            prim = stage.GetPrimAtPath(source)
            if prim.HasAPI(self.UsdPhysics.FilteredPairsAPI):
                targets = self.UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
                filtered.update(frozenset((source, str(target))) for target in targets)
        allowed = {
            frozenset((f"/Robot/{left}", f"/Robot/{right}"))
            for left in LEFT_HAND_BODY_PATHS
            for right in RIGHT_HAND_BODY_PATHS
        }
        all_pairs = {frozenset(pair) for pair in __import__("itertools").combinations(bodies, 2)}
        self.assertEqual(all_pairs - filtered, allowed)
        self.assertEqual(report["allowed_cross_hand_pair_count"], 9)
        self.assertEqual(report["filtered_internal_pair_count"], 2136)
        self.assertFalse(report["external_collision_relationships_modified"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_fails_closed_for_wrong_body_count(self):
        from pxr import Sdf, UsdGeom, UsdPhysics

        stage = self.Usd.Stage.CreateInMemory()
        root = UsdGeom.Xform.Define(stage, "/Robot").GetPrim()
        UsdPhysics.ArticulationRootAPI.Apply(root)
        root.CreateAttribute(
            "physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool
        ).Set(False)
        UsdPhysics.RigidBodyAPI.Apply(UsdGeom.Xform.Define(stage, "/Robot/body").GetPrim())
        with self.assertRaisesRegex(RuntimeError, "expected 66"):
            configure_selective_hand_self_collisions(stage, "/Robot")


class HandCollisionFactoryValidationTests(unittest.TestCase):
    def test_rejects_invalid_parameters_before_isaac_import(self):
        class Base:
            pass

        invalid = (
            {},
            {"target_wrist_position_world_m": [0, 0], "max_cartesian_error_m": 0.003},
            {"target_wrist_position_world_m": [0, 0, 1], "max_cartesian_error_m": 0.0},
            {"target_wrist_position_world_m": [0, float("nan"), 1], "max_cartesian_error_m": 0.003},
        )
        for parameters in invalid:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                make_hand_collision_env_type(Base, parameters)


if __name__ == "__main__":
    unittest.main()
