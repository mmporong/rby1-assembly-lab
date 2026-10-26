import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from rby1_assembly.rj45_scene import (
    audit_gripper_clearance,
    build_plug,
    build_rj45_scene,
    build_socket,
    build_table,
    load_jack_mesh,
)
from rby1_assembly.rj45_task import load_rj45_task


ROOT = Path(__file__).parents[1]


def _task() -> dict:
    return load_rj45_task(ROOT / "configs/tasks/rj45-v1.json")


def _synthetic_mesh(source_sha: str) -> dict:
    return {
        "schema_version": 1,
        "source_model_name": "fixture.step",
        "coordinate_space": "test",
        "length_unit": "metre",
        "bbox_m": [[-0.00889, -0.001595, -0.00666], [0.00889, 0.017074, 0.00666]],
        "meshes": [
            {
                "vertices_m": [[-0.001, 0.0, -0.001], [0.001, 0.0, -0.001], [0.0, 0.0, 0.001]],
                "faces": [[0, 1, 2]],
                "color_rgba": [100, 110, 120, 255],
            }
        ],
        "source_sha256": source_sha,
    }


class Rj45SourceTests(unittest.TestCase):
    def test_canonical_mesh_hash_is_required(self):
        task = _task()
        task["geometry"].pop("jack_canonical_mesh_sha256", None)
        with self.assertRaisesRegex(ValueError, "jack_canonical_mesh_sha256"):
            load_jack_mesh(ROOT, task)

    def test_mesh_source_hash_mismatch_fails_closed(self):
        task = _task()
        mesh_path = ROOT / task["geometry"]["jack_mesh"]
        task["geometry"]["jack_canonical_mesh_sha256"] = hashlib.sha256(
            mesh_path.read_bytes()
        ).hexdigest()
        task["geometry"]["jack_source_step_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "source hash"):
            load_jack_mesh(ROOT, task)

    def test_mesh_file_tampering_fails_closed_even_with_claimed_source_hash(self):
        task = _task()
        original = ROOT / task["geometry"]["jack_mesh"]
        task["geometry"]["jack_canonical_mesh_sha256"] = hashlib.sha256(
            original.read_bytes()
        ).hexdigest()
        mesh = json.loads(original.read_text(encoding="utf-8"))
        mesh["source_sha256"] = task["geometry"]["jack_source_step_sha256"]
        mesh["meshes"][0]["vertices_m"][0][0] += 1e-6
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            tampered = project / "tampered-mesh.json"
            tampered.write_text(json.dumps(mesh), encoding="utf-8")
            task["geometry"]["jack_mesh"] = tampered.name
            with self.assertRaisesRegex(ValueError, "jack mesh hash"):
                load_jack_mesh(project, task)


class Rj45PartTests(unittest.TestCase):
    def test_socket_compound_collision_keeps_cavity_open(self):
        task = _task()
        mesh = _synthetic_mesh(task["geometry"]["jack_source_step_sha256"])
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "socket.usda"
            report = build_socket(output, task, mesh)
            stage = Usd.Stage.Open(str(output))
            colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertEqual(len(colliders), report["collision_prim_count"])
            self.assertTrue(stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(any(prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse()))
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            # Sample the mouth, bevel end and throat axis.  No collider may fill
            # the central female-port cavity before the backstop.
            for y in (-0.03299, -0.0322, -0.025, -0.01851):
                point = Gf.Vec3d(0.0, y, 0.0)
                self.assertFalse(any(cache.ComputeWorldBound(prim).ComputeAlignedRange().Contains(point) for prim in colliders))
            self.assertAlmostEqual(report["cavity_open_bounds_m"]["mouth_x"][1], 0.00635)
            self.assertAlmostEqual(report["cavity_open_bounds_m"]["throat_x"][1], 0.0059563)

    def test_plug_is_drawing_sized_free_rigid_body(self):
        task = _task()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "plug.usda"
            report = build_plug(output, task)
            stage = Usd.Stage.Open(str(output))
            colliders = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.CollisionAPI)]
            self.assertEqual(len(colliders), 2)
            self.assertEqual(report["gold_contact_count"], 8)
            self.assertAlmostEqual(report["width_m"], 0.01179)
            self.assertAlmostEqual(report["nose_y_range_m"][0], -0.00313)
            self.assertAlmostEqual(report["nose_y_range_m"][1], 0.01137)
            self.assertAlmostEqual(
                report["nose_y_range_m"][1] - report["nose_y_range_m"][0],
                0.0145,
            )
            self.assertAlmostEqual(report["nose_height_m"], 0.00615)
            self.assertTrue(stage.GetDefaultPrim().HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(any(prim.IsA(UsdPhysics.Joint) for prim in stage.Traverse()))

    def test_pickup_nest_supports_nose_and_leaves_rear_open(self):
        task = _task()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "table.usda"
            report = build_table(output, task)
            stage = Usd.Stage.Open(str(output))
            plug_bottom = (
                task["plug"]["initial_position_m"][2]
                - task["geometry"]["plug_profile"]["terminated_nose_height_m"]
                / 2.0
            )
            self.assertAlmostEqual(report["pickup_nest_top_height_m"], plug_bottom, places=12)
            nest_y = report["pickup_nest_world_center_m"][1]
            nest_half_y = report["pickup_nest_size_m"][1] / 2.0
            plug_root_y = task["plug"]["initial_position_m"][1]
            nose_back_world = (
                plug_root_y
                + task["frames"]["plug_tip"]["position_m"][1]
                - task["geometry"]["plug_profile"]["nose_length_m"]
            )
            self.assertGreater(nest_y - nest_half_y, nose_back_world)
            expected = [
                "/Table/pickup_nest",
                "/Table/pickup_nest_plate",
                "/Table/pickup_locator_left",
                "/Table/pickup_locator_right",
                "/Table/pickup_locator_front",
            ]
            self.assertEqual(report["pickup_fixture_prim_paths"], expected)
            self.assertTrue(
                all(stage.GetPrimAtPath(path).IsA(UsdGeom.Cube) for path in expected)
            )
            self.assertTrue(
                all(
                    stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
                    for path in expected
                )
            )
            self.assertTrue(report["pickup_locator_open_top"])
            self.assertEqual(report["pickup_locator_attachment_joints"], 0)

    def test_pickup_locator_has_nominal_clearance_and_fits_plate(self):
        task = _task()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "table.usda"
            report = build_table(output, task)
            stage = Usd.Stage.Open(str(output))
            cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
            ranges = {
                name: cache.ComputeWorldBound(
                    stage.GetPrimAtPath(f"/Table/{name}")
                ).ComputeAlignedRange()
                for name in (
                    "pickup_nest_plate",
                    "pickup_locator_left",
                    "pickup_locator_right",
                    "pickup_locator_front",
                )
            }
        plug_width = task["geometry"]["plug_profile"]["max_width_m"]
        expected_side_clearance = (
            report["pickup_locator_slot_width_m"] - plug_width
        ) / 2.0
        self.assertAlmostEqual(
            report["pickup_locator_nominal_side_clearance_m"], 0.00015, places=12
        )
        self.assertAlmostEqual(
            report["pickup_locator_nominal_side_clearance_m"],
            expected_side_clearance,
            places=12,
        )
        self.assertAlmostEqual(
            report["pickup_locator_nominal_front_clearance_m"], 0.00015, places=12
        )
        self.assertGreater(
            report["pickup_locator_nominal_side_clearance_m"],
            task["contact"]["contact_offset_m"],
        )
        self.assertGreater(
            report["pickup_locator_nominal_front_clearance_m"],
            task["contact"]["contact_offset_m"],
        )
        plate_x, plate_y, plate_z = report["pickup_nest_plate_size_m"]
        outer_width = (
            report["pickup_locator_slot_width_m"]
            + 2.0 * report["pickup_locator_wall_thickness_m"]
        )
        self.assertLessEqual(outer_width, plate_x)
        self.assertAlmostEqual(plate_y, 0.016)
        self.assertAlmostEqual(plate_z, 0.002)
        self.assertLessEqual(
            report["pickup_locator_front_stop_y_range_m"][1],
            report["pickup_nest_plate_world_center_m"][1] + plate_y / 2.0,
        )
        plug_x = task["plug"]["initial_position_m"][0]
        plug_left = plug_x - plug_width / 2.0
        plug_right = plug_x + plug_width / 2.0
        self.assertAlmostEqual(
            plug_left - ranges["pickup_locator_left"].GetMax()[0],
            0.00015,
            places=10,
        )
        self.assertAlmostEqual(
            ranges["pickup_locator_right"].GetMin()[0] - plug_right,
            0.00015,
            places=10,
        )
        plug_tip_y = (
            task["plug"]["initial_position_m"][1]
            + task["frames"]["plug_tip"]["position_m"][1]
        )
        self.assertAlmostEqual(
            ranges["pickup_locator_front"].GetMin()[1] - plug_tip_y,
            0.00015,
            places=10,
        )
        plug_bottom = (
            task["plug"]["initial_position_m"][2]
            - task["geometry"]["plug_profile"]["terminated_nose_height_m"]
            / 2.0
        )
        self.assertAlmostEqual(
            ranges["pickup_nest_plate"].GetMax()[2], plug_bottom, places=9
        )
        self.assertAlmostEqual(
            ranges["pickup_locator_left"].GetMin()[2], plug_bottom, places=9
        )

    def test_pickup_locator_rejects_contact_offset_sized_slot_clearance(self):
        task = _task()
        plug_width = task["geometry"]["plug_profile"]["max_width_m"]
        task["geometry"]["pickup_nest"]["locator"]["slot_width_m"] = (
            plug_width + 2.0 * task["contact"]["contact_offset_m"]
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "more than one contact offset"):
                build_table(Path(directory) / "table.usda", task)


class Rj45BuildTests(unittest.TestCase):
    def test_real_build_report_hashes_are_reproducible(self):
        report = build_rj45_scene(ROOT)
        output = ROOT / "assets/generated/tasks/rj45-v1"
        self.assertEqual(report["free_rigid_objects"], 2)
        self.assertEqual(report["attachment_joints"], 0)
        self.assertEqual(report["parts"]["free_rigid_objects"], 2)
        self.assertEqual(report["parts"]["attachment_joints"], 0)
        self.assertFalse(report["physics_executed"])
        self.assertTrue(report["static_stage_created"])
        self.assertEqual(report["source_cad_mesh"]["source_step_sha256"], _task()["geometry"]["jack_source_step_sha256"])
        self.assertAlmostEqual(report["source_cad_mesh"]["transformed_bbox_m"][0][1], -0.033)
        self.assertAlmostEqual(report["socket"]["entry_y_m"], -0.033)
        for filename, expected in report["files"].items():
            self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), expected)
        report_again = build_rj45_scene(ROOT)
        self.assertEqual(report_again["files"], report["files"])
        self.assertEqual(report_again["source_transform"]["sha256"], report["source_transform"]["sha256"])

    def test_actual_gripper_geometry_clears_port_and_inserted_plug(self):
        task = _task()
        robot_path = ROOT / task["assets"]["robot"]
        audit = audit_gripper_clearance(robot_path, task)
        self.assertTrue(audit["passed"])
        self.assertAlmostEqual(audit["left"]["front_edge_y_m"], -0.011, places=5)
        self.assertAlmostEqual(audit["left"]["port_approach_clearance_m"], 0.022, places=5)
        port_plane_in_plug = task["frames"]["plug_tip"]["position_m"][1] - task["evaluation"]["assembled_depth_m"]
        self.assertLess(audit["right"]["front_edge_y_m"], port_plane_in_plug)
        # The port clearance must exceed the two authored contact offsets.
        # This conservative box bound is distinct from the actual convex gap.
        self.assertGreater(audit["right"]["clearance_after_assembled_depth_m"], 2 * task["contact"]["contact_offset_m"])
        self.assertLess(audit["left"]["closure_axis_projection_on_insertion"], 1.0e-8)
        self.assertLess(audit["right"]["closure_axis_projection_on_insertion"], 1.0e-8)

    def test_front_to_back_adapter_grasp_is_rejected(self):
        task = _task()
        task["frames"]["left_grasp"]["quaternion_wxyz"] = [
            0.7071067811865476,
            0.0,
            0.0,
            0.7071067811865476,
        ]
        robot_path = ROOT / task["assets"]["robot"]
        with self.assertRaisesRegex(ValueError, "front-to-back"):
            audit_gripper_clearance(robot_path, task)


if __name__ == "__main__":
    unittest.main()
