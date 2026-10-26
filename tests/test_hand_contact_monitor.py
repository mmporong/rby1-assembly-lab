import hashlib
import inspect
from pathlib import Path
import unittest

from rby1_assembly.hand_collision import (
    EXPECTED_RIGID_BODY_COUNT,
    configure_selective_hand_self_collisions,
)
from rby1_assembly.hand_contact_monitor import make_hand_contact_monitor_env_type


class NominalHandContactMonitorUsdTests(unittest.TestCase):
    def setUp(self):
        try:
            from pxr import Usd, UsdPhysics
        except ImportError:
            self.skipTest("usd-core is not installed")
        self.Usd = Usd
        self.UsdPhysics = UsdPhysics

    def test_rj45_robot_has_audited_selective_hand_collision_partition(self):
        path = Path("assets/generated/tasks/rj45-v1/robot.usda")
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        stage = self.Usd.Stage.Open(str(path))
        report = configure_selective_hand_self_collisions(stage, "/Robot")
        self.assertEqual(report["rigid_body_count"], EXPECTED_RIGID_BODY_COUNT)
        self.assertEqual(report["allowed_cross_hand_pair_count"], 9)
        self.assertEqual(report["filtered_internal_pair_count"], 2136)
        self.assertFalse(report["external_collision_relationships_modified"])
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


class NominalHandContactMonitorContractTests(unittest.TestCase):
    def test_factory_rejects_non_type_without_importing_isaac(self):
        with self.assertRaises(TypeError):
            make_hand_contact_monitor_env_type(object())

    def test_wrapper_has_no_command_or_pose_override(self):
        source = inspect.getsource(make_hand_contact_monitor_env_type)
        self.assertNotIn("def _desired_ee_twists", source)
        self.assertNotIn("def _pre_physics_step", source)
        self.assertNotIn("def _apply_action", source)
        self.assertNotIn("write_root_pose_to_sim", source)
        self.assertIn("passive_batched_runtime_ordering_asserted", source)
        self.assertIn("replicate_physics=False", source)
        self.assertIn("_validate_hand_contact_sensor_routing", source)
        self.assertIn("command_pose_force_or_failure_injection", source)


if __name__ == "__main__":
    unittest.main()
