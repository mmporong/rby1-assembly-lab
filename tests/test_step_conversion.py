from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "convert_step_model.py"


def _module():
    spec = importlib.util.spec_from_file_location("convert_step_model", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load conversion module: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StepConversionMathTests(unittest.TestCase):
    def test_numpy_compatibility_patch_reports_missing_aliases(self) -> None:
        module = _module()
        compat_numpy = SimpleNamespace(
            int64="int64", uint32="uint32", uint64="uint64", uintc="existing"
        )

        applied = module._apply_numpy_compat_aliases(compat_numpy)

        self.assertEqual(applied, ["long", "ulong"])
        self.assertEqual(compat_numpy.long, "int64")
        self.assertEqual(compat_numpy.uintc, "existing")
        self.assertEqual(compat_numpy.ulong, "uint64")

    def test_numpy_compatibility_patch_is_noop_when_aliases_exist(self) -> None:
        module = _module()
        normal_numpy = SimpleNamespace(
            int64="int64",
            uint32="uint32",
            uint64="uint64",
            long="existing-long",
            uintc="existing-uintc",
            ulong="existing-ulong",
        )

        self.assertEqual(module._apply_numpy_compat_aliases(normal_numpy), [])
        self.assertEqual(normal_numpy.long, "existing-long")
        self.assertEqual(normal_numpy.uintc, "existing-uintc")
        self.assertEqual(normal_numpy.ulong, "existing-ulong")

    def test_world_transform_applies_rotation_and_translation(self) -> None:
        module = _module()
        vertices = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        transform = np.array(
            [
                [0.0, -1.0, 0.0, 3.0],
                [1.0, 0.0, 0.0, -4.0],
                [0.0, 0.0, 1.0, 5.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        actual = module.apply_world_transform(vertices, transform)
        np.testing.assert_allclose(actual, [[3.0, -3.0, 5.0], [1.0, -4.0, 5.0]])

    def test_metric_bbox_rejects_millimetres_masquerading_as_metres(self) -> None:
        module = _module()
        module.validate_metric_bbox([[0, 0, 0], [0.01778, 0.018669, 0.01733042]])
        with self.assertRaisesRegex(ValueError, "implausible"):
            module.validate_metric_bbox([[0, 0, 0], [17.78, 18.669, 17.33042]])

    def test_metric_bbox_rejects_degenerate_geometry(self) -> None:
        module = _module()
        with self.assertRaisesRegex(ValueError, "positive extents"):
            module.validate_metric_bbox([[0, 0, 0], [0.01, 0.01, 0.0]])

    def test_metric_bbox_rejects_non_finite_geometry(self) -> None:
        module = _module()
        with self.assertRaisesRegex(ValueError, "finite 2x3"):
            module.validate_metric_bbox([[0, 0, 0], [0.01, np.nan, 0.01]])

    def test_mesh_payload_is_stable_across_node_renaming_and_permutation(self) -> None:
        module = _module()

        class Graph:
            def __init__(self, records):
                self._records = records
                self.nodes_geometry = list(records)

            def __getitem__(self, name):
                return self._records[name]

        def geometry(vertices, faces, color, material_name):
            visual = SimpleNamespace(
                vertex_colors=np.tile(np.asarray(color, dtype=np.uint8), (len(vertices), 1)),
                material=SimpleNamespace(name=material_name),
            )
            return SimpleNamespace(
                vertices=np.asarray(vertices, dtype=np.float64),
                faces=np.asarray(faces, dtype=np.int64),
                visual=visual,
            )

        identity = np.eye(4)
        translated = np.eye(4)
        translated[:3, 3] = [0.01, 0.02, 0.03]
        first_geometry = geometry(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]],
            [[0, 1, 2]],
            [10, 20, 30, 255],
            "random-material-a",
        )
        second_geometry = geometry(
            [[0.0, 0.0, 0.0], [0.0, 0.005, 0.0], [0.0, 0.0, 0.004]],
            [[0, 1, 2]],
            [40, 50, 60, 255],
            "random-material-b",
        )
        scene_a = SimpleNamespace(
            graph=Graph(
                {
                    "uuid-node-a": (identity, "uuid-geometry-a"),
                    "uuid-node-b": (translated, "uuid-geometry-b"),
                }
            ),
            geometry={
                "uuid-geometry-a": first_geometry,
                "uuid-geometry-b": second_geometry,
            },
        )
        scene_b = SimpleNamespace(
            graph=Graph(
                {
                    "renamed-node-b": (translated, "renamed-geometry-b"),
                    "renamed-node-a": (identity, "renamed-geometry-a"),
                }
            ),
            geometry={
                "renamed-geometry-b": second_geometry,
                "renamed-geometry-a": first_geometry,
            },
        )

        source_map_a = []
        source_map_b = []
        payload_a = module.scene_to_mesh_payload(
            scene_a, "fixed-source.stp", source_scene_map=source_map_a
        )
        payload_b = module.scene_to_mesh_payload(
            scene_b, "fixed-source.stp", source_scene_map=source_map_b
        )

        self.assertEqual(payload_a, payload_b)
        self.assertNotEqual(source_map_a, source_map_b)
        serialized = module._canonical_json_bytes(payload_a).decode("utf-8")
        self.assertNotIn("uuid-", serialized)
        self.assertNotIn("renamed-", serialized)
        self.assertEqual(
            [mesh["node_name"] for mesh in payload_a["meshes"]],
            [mesh["geometry_name"] for mesh in payload_a["meshes"]],
        )


if __name__ == "__main__":
    unittest.main()
