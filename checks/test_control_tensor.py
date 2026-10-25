"""Torch CPU 회귀 검사. Isaac 런타임이나 물리 성공을 대체하지 않는다."""

from __future__ import annotations

import ast
import copy
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "src/rby1_assembly/isaac_env.py"
METHODS = {
    "_plug_center_goal",
    "_pregrasp_above_mask",
    "_grasp_target_positions",
    "_pregrasp_lift_targets",
    "_pregrasp_lift_mask",
    "_advance_evaluation_dwell",
    "_bound_residual_actions",
    "_resolve_env_paths",
    "_episode_time_limit_reached",
}


class ContactSensorCfg(SimpleNamespace):
    def copy(self):
        return copy.deepcopy(self)


class AssemblyPhase(IntEnum):
    PREGRASP = 0
    CLOSE = 1
    CHECK_LIFT = 2
    ALIGN = 3
    INSERT = 4
    RETRACT = 5
    DONE = 6
    ABORT = 7


def load_actual_control_methods():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    env_class = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Rby1AssemblyEnv"
    )
    selected = [
        node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name in METHODS
    ]
    if {node.name for node in selected} != METHODS:
        raise AssertionError("required control method is missing from isaac_env.py")
    module = ast.Module(body=[ast.ClassDef("ControlSlice", [], [], selected, [])], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"torch": torch, "AssemblyPhase": AssemblyPhase, "ContactSensorCfg": ContactSensorCfg}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["ControlSlice"], tree


ControlSlice, SOURCE_TREE = load_actual_control_methods()


class ControlTensorTests(unittest.TestCase):
    def make_control(self, num_envs: int = 2):
        control = ControlSlice()
        control.num_envs = num_envs
        control.device = "cpu"
        control.cfg = SimpleNamespace(
            socket_entry_y_m=-0.03,
            plug_tip_y_m=0.065,
            left_grasp_center_above_object_m=0.009,
            right_grasp_center_above_object_m=0.009,
            left_pregrasp_safe_wrist_z_m=1.0416,
            right_pregrasp_safe_wrist_z_m=1.0416,
        )
        control._phase = torch.tensor([AssemblyPhase.ALIGN, AssemblyPhase.PREGRASP])[:num_envs]
        control._success_since_s = torch.full((num_envs,), -1.0)
        control._evaluation_dwell_s = torch.zeros(num_envs)
        control.socket = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros((num_envs, 3))))
        control.plug = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros((num_envs, 3))))
        control.scene = SimpleNamespace(env_origins=torch.zeros((num_envs, 3)))
        return control

    def test_tip_entry_depth_maps_to_plug_center_y(self):
        control = self.make_control(1)
        socket = torch.tensor([[0.50, 0.05, 0.84]])
        goal = control._plug_center_goal(socket, 0.03)
        self.assertAlmostEqual(float(goal[0, 1]), -0.015, places=7)
        reconstructed_depth = goal[0, 1] + control.cfg.plug_tip_y_m - (
            socket[0, 1] + control.cfg.socket_entry_y_m
        )
        self.assertAlmostEqual(float(reconstructed_depth), 0.03, places=7)

    def test_pregrasp_requires_both_centers_at_above_waypoint(self):
        control = self.make_control(2)
        control.socket.data.root_pos_w[:] = torch.tensor([0.45, 0.12, 0.776])
        control.plug.data.root_pos_w[:] = torch.tensor([0.45, -0.12, 0.776])
        left = control.socket.data.root_pos_w.clone()
        right = control.plug.data.root_pos_w.clone()
        left[:, 2] += 0.059
        right[:, 2] += 0.059
        right[1, 0] += 0.020
        self.assertEqual(control._pregrasp_above_mask(left, right).tolist(), [True, False])

    def test_pregrasp_first_lifts_in_place_before_xy_approach(self):
        control = self.make_control(2)
        left = torch.tensor([[0.0, 0.22, 0.854], [2.5, 0.22, 1.040]])
        right = torch.tensor([[0.0, -0.22, 0.854], [2.5, -0.22, 1.040]])
        left_target, right_target = control._pregrasp_lift_targets(left, right)
        torch.testing.assert_close(left_target[:, :2], left[:, :2])
        torch.testing.assert_close(right_target[:, :2], right[:, :2])
        torch.testing.assert_close(left_target[:, 2], torch.full((2,), 1.0416))
        self.assertEqual(control._pregrasp_lift_mask(left, right).tolist(), [False, True])

    def test_dwell_accumulates_only_while_all_criteria_hold(self):
        control = self.make_control(2)
        control._advance_evaluation_dwell(torch.tensor([True, False]), 1.0)
        control._advance_evaluation_dwell(torch.tensor([True, True]), 1.3)
        torch.testing.assert_close(control._evaluation_dwell_s, torch.tensor([0.3, 0.0]))
        control._advance_evaluation_dwell(torch.tensor([False, True]), 1.5)
        torch.testing.assert_close(control._evaluation_dwell_s, torch.tensor([0.0, 0.2]))
        torch.testing.assert_close(control._success_since_s, torch.tensor([-1.0, 1.3]))

    def test_residual_vectors_are_norm_bounded_and_pregrasp_is_zero(self):
        control = self.make_control(2)
        actions = torch.tensor([[4.0] * 12, [0.5] * 12])
        bounded = control._bound_residual_actions(actions).reshape(2, 4, 3)
        self.assertTrue(torch.all(torch.linalg.vector_norm(bounded[0], dim=1) <= 1.0 + 1e-7))
        torch.testing.assert_close(bounded[1], torch.zeros((4, 3)))

    def test_namespace_resolution_copies_config_and_resolves_contact_filters(self):
        control = self.make_control(1)
        control.scene.env_regex_ns = "/World/envs/env_.*"
        raw = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_gripper/ee_finger_l1",
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Socket"],
        )
        result = control._resolve_env_paths(raw)
        self.assertEqual(result.prim_path, "/World/envs/env_.*/Robot/left_gripper/ee_finger_l1")
        self.assertEqual(result.filter_prim_paths_expr, ["/World/envs/env_.*/Socket"])
        self.assertEqual(raw.filter_prim_paths_expr, ["{ENV_REGEX_NS}/Socket"])

    def test_residual_rejects_invalid_values_and_terminal_action_is_zero(self):
        control = self.make_control(2)
        invalid = torch.zeros((2, 12))
        invalid[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            control._bound_residual_actions(invalid)
        with self.assertRaises(ValueError):
            control._bound_residual_actions(torch.zeros((2, 11)))
        control._phase[:] = torch.tensor([AssemblyPhase.DONE, AssemblyPhase.ABORT])
        torch.testing.assert_close(control._bound_residual_actions(torch.ones((2, 12))), torch.zeros((2, 12)))

    def test_timeout_uses_full_episode_horizon(self):
        control = self.make_control(2)
        control.max_episode_length = 900
        control.episode_length_buf = torch.tensor([899, 900])
        self.assertEqual(control._episode_time_limit_reached().tolist(), [False, True])
        self.assertAlmostEqual(float(control.episode_length_buf[1]) / 30.0, 30.0)


if __name__ == "__main__":
    unittest.main()
