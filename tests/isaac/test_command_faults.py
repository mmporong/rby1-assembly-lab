from types import SimpleNamespace
import unittest

import torch

from rby1_assembly.command_faults import CommandFaultExperiment


PHASES = {"align": 3, "insert": 4, "retract": 5}


class FakeEnv:
    def __init__(self):
        self.num_envs = 1
        self.step_dt = 1 / 30
        self.common_step_counter = 0
        self.cfg = SimpleNamespace(gripper_open_magnitude_m=.048)
        self._phase = torch.tensor([PHASES["align"]])
        self._grip_proven = torch.tensor([[True, True]])
        self._gripper_joint_ids = [0, 1, 2, 3]
        self._joint_targets = torch.zeros((1, 4))
        self.scene = SimpleNamespace(env_origins=torch.zeros((1, 3)))
        self.plug = SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0., 0., .90]])))
        self.socket = SimpleNamespace(data=SimpleNamespace(root_quat_w=torch.tensor([[1., 0., 0., 0.]])))
        self.relative = torch.tensor([[0., .006, 0.]])
        self.sensor_force = torch.tensor([5.7])
        self.termination = "none"

    def _pre_physics_step(self, actions):
        self._joint_targets[:] = torch.tensor([[-.024, .024, -.0165, .0165]])

    def _desired_ee_twists(self):
        return torch.zeros((1, 6)), torch.zeros((1, 6))

    def _relative_object_pose(self):
        return self.relative, torch.tensor([[1., 0., 0., 0.]])

    def _sim_time_s(self):
        return self.common_step_counter * self.step_dt


class TestCommandFaults(unittest.TestCase):
    def make_case(self, name):
        env = FakeEnv()
        parameters = ({"trigger_min_plug_height_m": .86} if name == "right_gripper_open" else
                      {"trigger_insertion_depth_m": .004, "max_position_error_command_m": .0035,
                       "ramp_duration_s": .5})
        case = CommandFaultExperiment(env, name, parameters, PHASES)
        case.install()
        self.addCleanup(case.close)
        return env, case

    def test_open_command_requires_proven_lift_and_height(self):
        env, case = self.make_case("right_gripper_open")
        env._grip_proven[0, 1] = False
        env._pre_physics_step(None)
        self.assertFalse(case.triggered)
        env._grip_proven[:] = True
        env.plug.data.root_pos_w[0, 2] = .80
        env._pre_physics_step(None)
        self.assertFalse(case.triggered)
        env.plug.data.root_pos_w[0, 2] = .90
        env._pre_physics_step(None)
        self.assertTrue(case.triggered)
        torch.testing.assert_close(env._joint_targets, torch.tensor([[-.024, .024, -.048, .048]]))
        self.assertEqual(len(case.events), 1)
        env._pre_physics_step(None)
        self.assertEqual(len(case.events), 1)

    def test_faults_only_modify_commands(self):
        for name in ("right_gripper_open", "insertion_lateral_disturbance"):
            with self.subTest(name=name):
                env, case = self.make_case(name)
                env._phase[:] = PHASES["insert"]
                before = [env.plug.data.root_pos_w.clone(), env.relative.clone(),
                          env.sensor_force.clone(), env._phase.clone(), env._grip_proven.clone()]
                env._pre_physics_step(None)
                env._desired_ee_twists()
                after = [env.plug.data.root_pos_w, env.relative, env.sensor_force, env._phase, env._grip_proven]
                for a, b in zip(before, after):
                    torch.testing.assert_close(a, b)
                self.assertEqual(env.termination, "none")
                self.assertFalse(case.report()["physical_state_force_sensor_or_termination_injection"])

    def test_lateral_command_only_after_partial_insertion(self):
        env, case = self.make_case("insertion_lateral_disturbance")
        _, right = env._desired_ee_twists()
        torch.testing.assert_close(right, torch.zeros((1, 6)))
        env._phase[:] = PHASES["insert"]
        env.relative[0, 1] = .002
        env._desired_ee_twists()
        self.assertFalse(case.triggered)
        env.relative[0, 1] = .006
        left, right = env._desired_ee_twists()
        self.assertTrue(case.triggered)
        self.assertAlmostEqual(float(right[0, 0]), .0035 * env.step_dt / .5, places=8)
        torch.testing.assert_close(left, torch.zeros((1, 6)))
        env.common_step_counter = 30
        _, right = env._desired_ee_twists()
        self.assertAlmostEqual(float(right[0, 0]), .0035, places=8)

    def test_lateral_command_uses_socket_frame_and_releases_for_recovery(self):
        env, case = self.make_case("insertion_lateral_disturbance")
        env._phase[:] = PHASES["insert"]
        env.socket.data.root_quat_w[:] = torch.tensor([[2**-.5, 0., 0., 2**-.5]])
        _, right = env._desired_ee_twists()
        self.assertAlmostEqual(float(right[0, 0]), 0., places=8)
        self.assertGreater(float(right[0, 1]), 0.)
        env._phase[:] = PHASES["retract"]
        _, right = env._desired_ee_twists()
        torch.testing.assert_close(right, torch.zeros((1, 6)))
        self.assertFalse(case.active)
        env._phase[:] = PHASES["insert"]
        env.common_step_counter = 40
        env._desired_ee_twists()
        self.assertEqual(sum(e["event"] == "lateral_command_started" for e in case.events), 2)

    def test_restores_original_methods(self):
        env = FakeEnv()
        original = env._pre_physics_step
        case = CommandFaultExperiment(env, "right_gripper_open", {"trigger_min_plug_height_m": .86}, PHASES)
        case.install()
        with self.assertRaises(RuntimeError):
            case.install()
        case.close()
        self.assertEqual(env._pre_physics_step, original)

    def test_rejects_unknown_multi_environment_and_invalid_parameters(self):
        env = FakeEnv()
        with self.assertRaises(ValueError):
            CommandFaultExperiment(env, "unknown", {}, PHASES)
        env.num_envs = 2
        with self.assertRaises(ValueError):
            CommandFaultExperiment(env, "right_gripper_open", {"trigger_min_plug_height_m": .86}, PHASES)
        env.num_envs = 1
        for value in (0, float("nan"), True):
            with self.assertRaises(ValueError):
                CommandFaultExperiment(env, "right_gripper_open", {"trigger_min_plug_height_m": value}, PHASES)


if __name__ == "__main__":
    unittest.main()
