import ast
from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_SIM_PATH = PROJECT_ROOT / "scripts/run_sim.py"


class AssemblyPhase(IntEnum):
    PREGRASP = 0
    GRASP = 1
    LIFT = 2
    ALIGN = 3
    INSERT = 4
    DONE = 5
    RETRACT = 6
    ABORT = 7


def _tensor_list(value):
    return value.detach().cpu().tolist()


def _compiled_trace_record():
    tree = ast.parse(RUN_SIM_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_trace_record"
    )
    namespace = {
        "torch": torch,
        "AssemblyPhase": AssemblyPhase,
        "Rby1AssemblyEnv": object,
        "FORBIDDEN_CONTACT_LOAD_SOURCE": "test_forbidden_source",
        "_tensor_list": _tensor_list,
        "_tensor_json_value": lambda value: value.detach().cpu().tolist(),
        "_forbidden_contact_breakdown": lambda *args: None,
        "_live_forbidden_contact_breakdown": lambda *args: (_ for _ in ()).throw(
            AssertionError("terminal trace must not read live forbidden contacts")
        ),
    }
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
            RUN_SIM_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace["_trace_record"]


class AutoresetLiveStateSentinel:
    num_envs = 1
    device = torch.device("cpu")
    _left_ee_body_id = 0
    _right_ee_body_id = 1
    _arm_joint_ids = list(range(14))

    def __init__(self):
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                body_pos_w=torch.full((1, 2, 3), -700.0),
                body_quat_w=torch.full((1, 2, 4), -701.0),
            )
        )
        self.cfg = SimpleNamespace()
        self.extras = {
            "terminal_evaluation": {
                "criteria": {"geometry_ok": torch.tensor([True])}
            }
        }

    @property
    def _training_last_policy_observation(self):
        raise AssertionError("autoreset live observation must not be read")

    @property
    def _training_last_policy_observation_timestamp_s(self):
        raise AssertionError("autoreset live observation timestamp must not be read")

    @property
    def _training_last_policy_observation_valid(self):
        raise AssertionError("autoreset live observation validity must not be read")

    @property
    def _training_estimator_valid(self):
        raise AssertionError("autoreset live estimator state must not be read")

    @property
    def _training_policy_activated(self):
        raise AssertionError("autoreset live activation state must not be read")

    @property
    def _training_previous_action(self):
        raise AssertionError("autoreset live requested action must not be read")

    @property
    def _actions(self):
        raise AssertionError("autoreset live applied action must not be read")


def _terminal_snapshot():
    policy_input = torch.arange(79, dtype=torch.float32).reshape(1, 79)
    requested_action = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    applied_action = requested_action / 20.0
    return {
        "phase": torch.tensor([int(AssemblyPhase.ALIGN)]),
        "terminal_mask": torch.tensor([True]),
        "timestamp_s": torch.tensor([9.0]),
        "retry_count": torch.tensor([0]),
        "revolute_joint_positions_rad": torch.zeros((1, 20)),
        "gripper_positions_m": torch.zeros((1, 4)),
        "object_poses_world_wxyz": torch.tensor(
            [
                [
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.1, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
        "contact_force_proxy_n": torch.zeros((1, 2)),
        "grip_proven": torch.tensor([True]),
        "grip_live": torch.tensor([[True, True]]),
        "actual_depth_m": torch.tensor([0.002]),
        "actual_lateral_error_m": torch.tensor([0.0001]),
        "actual_orientation_error_rad": torch.tensor([0.001]),
        "termination_reason_code": torch.tensor([0]),
        "termination_reason_by_code": {0: "none"},
        "revolute_joint_names": tuple(f"joint_{index}" for index in range(20)),
        "gripper_joint_names": tuple(f"gripper_{index}" for index in range(4)),
        "forbidden_contact_load_n": torch.tensor([0.0]),
        "forbidden_contact_load_peak_n": torch.tensor([0.0]),
        "forbidden_contact_load_source": "terminal_forbidden_source",
        "continuous_grasp_loss_since_s": torch.tensor([-1.0]),
        "insertion_command_depth_m": torch.tensor([0.0]),
        "insertion_advance_enabled": torch.tensor([False]),
        "independent_eval_dwell_s": torch.tensor([0.0]),
        "independent_eval_success": torch.tensor([False]),
        "independent_eval_valid": torch.tensor([True]),
        "insert_joint_target_error_rad": torch.zeros((1, 14)),
        "policy_input_observation": policy_input,
        "policy_input_timestamp_s": torch.tensor([8.75]),
        "policy_input_valid": torch.tensor([True]),
        "training_estimator_valid": torch.tensor([True]),
        "training_policy_activated": torch.tensor([True]),
        "held_initialization_passed": torch.tensor([True]),
        "valid_policy_input_step_count": torch.tensor([13]),
        "requested_expanded_residual_action": requested_action,
        "applied_bounded_residual_action": applied_action,
    }


class RunSimTrainingTraceTests(unittest.TestCase):
    def test_terminal_training_telemetry_comes_only_from_pre_autoreset_snapshot(self):
        trace_record = _compiled_trace_record()
        snapshot = _terminal_snapshot()

        result = trace_record(
            AutoresetLiveStateSentinel(),
            control_step=270,
            events=["terminal"],
            snapshot=snapshot,
        )

        environment = result["environments"][0]
        self.assertEqual(
            environment["policy_input_observation"],
            snapshot["policy_input_observation"][0].tolist(),
        )
        self.assertEqual(environment["policy_input_timestamp_s"], 8.75)
        self.assertTrue(environment["policy_input_valid"])
        self.assertTrue(environment["training_estimator_valid"])
        self.assertTrue(environment["training_policy_activated"])
        self.assertTrue(environment["held_initialization_passed"])
        self.assertEqual(environment["valid_policy_input_step_count"], 13)
        self.assertEqual(
            environment["requested_expanded_residual_action"],
            snapshot["requested_expanded_residual_action"][0].tolist(),
        )
        self.assertEqual(
            environment["applied_bounded_residual_action"],
            snapshot["applied_bounded_residual_action"][0].tolist(),
        )
        self.assertEqual(environment["independent_eval_criteria"], {"geometry_ok": True})


if __name__ == "__main__":
    unittest.main()
