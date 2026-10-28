from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rby1_assembly.rj45_support_ablation import (
    make_rj45_support_ablation_env_type,
    support_ablation_terminal_metrics,
    validate_support_arm_mode,
)


class Phase(IntEnum):
    ALIGN = 4
    INSERT = 5
    RETRACT = 6
    COMPLETE = 7
    ABORT = 8


class FakeTrainingEnv:
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = "cpu"
        self.step_dt = 0.1
        self._sim_step_counter = 0
        self._left_arm_joint_ids = [0, 1]
        self._right_arm_joint_ids = [2, 3]
        self._left_ee_body_id = 0
        self._phase = torch.full((3,), int(Phase.ALIGN), dtype=torch.long)
        self._insert_joint_targets = torch.tensor(
            [[10.0, 20.0, 30.0, 40.0]] * 3
        )
        self._joint_targets = torch.zeros((3, 6))
        self._insertion_advance_enabled = torch.tensor([True, False, True])
        self.robot = SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.tensor(
                    [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]
                ),
                body_pos_w=torch.tensor(
                    [[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]], [[0.7, 0.8, 0.9]]]
                ),
                body_quat_w=torch.tensor(
                    [[[1.0, 0.0, 0.0, 0.0]]] * 3
                ),
            )
        )
        self.socket = SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor(
                    [[0.01, 0.02, 0.03], [1.01, 0.02, 0.03], [2.01, 0.02, 0.03]]
                ),
                root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 3),
            )
        )
        self.scene = SimpleNamespace(
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        )
        self.cfg = SimpleNamespace(
            assembly_socket_goal_m=(0.0, 0.0, 0.0),
            assembly_socket_goal_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        self._episode_grip_force_peak_n = torch.tensor([2.0, 3.0, 4.0])
        self._episode_insertion_contact_load_peak_n = torch.tensor([5.0, 6.0, 7.0])
        self._episode_forbidden_contact_load_peak_n = torch.tensor([0.1, 0.2, 0.3])
        self._initialize_training_state()

    def _initialize_training_state(self) -> None:
        pass

    def _transition_where(self, mask, phase, now, reason=None):
        del now, reason
        self._phase[mask] = int(phase)
        if phase == Phase.INSERT:
            self._insert_joint_targets[mask] = torch.tensor([11.0, 12.0, 31.0, 32.0])

    def _desired_ee_twists(self):
        left = torch.ones((3, 6))
        right = torch.full((3, 6), 2.0)
        self._baseline_cartesian_twists = torch.stack((left, right), dim=1)
        return left, right

    def _pre_physics_step(self, actions) -> None:
        del actions
        self._joint_targets += 0.5
        self._insert_joint_targets += 9.0

    def _configure_training_reset(self, env_ids):
        del env_ids

    def _get_dones(self):
        return torch.zeros(3, dtype=torch.bool), torch.zeros(3, dtype=torch.bool)

    def _training_step_snapshot(self):
        return {"base": torch.ones(3)}


def make_env(mode: str):
    return make_rj45_support_ablation_env_type(FakeTrainingEnv, mode)()


def test_mode_validation_and_transition_freezes_only_selected_left_reference() -> None:
    assert validate_support_arm_mode("cartesian_correction") == "cartesian_correction"
    with pytest.raises(ValueError, match="support arm mode"):
        validate_support_arm_mode("unknown")

    env = make_env("insert_joint_hold")
    mask = torch.tensor([True, False, True])
    env._transition_where(mask, Phase.INSERT, 0.0)
    assert torch.equal(
        env._support_insert_joint_reference[mask],
        torch.tensor([[1.0, 2.0], [9.0, 10.0]]),
    )
    assert torch.equal(
        env._insert_joint_targets[mask, 2:],
        torch.tensor([[31.0, 32.0], [31.0, 32.0]]),
    )
    assert not env._support_insert_reference_valid[1]


def test_twist_isolation_keeps_right_and_noninsert_left_unchanged() -> None:
    env = make_env("insert_joint_hold")
    env._transition_where(torch.tensor([True, False, False]), Phase.INSERT, 0.0)
    left, right = env._desired_ee_twists()
    assert torch.equal(left[0], torch.zeros(6))
    assert torch.equal(left[1:], torch.ones((2, 6)))
    assert torch.equal(right, torch.full((3, 6), 2.0))
    assert torch.equal(env._baseline_cartesian_twists[0, 0], torch.zeros(6))
    assert torch.equal(
        env._baseline_cartesian_twists[:, 1], torch.full((3, 6), 2.0)
    )

    current = make_env("cartesian_correction")
    current._transition_where(torch.tensor([True, False, False]), Phase.INSERT, 0.0)
    current_left, current_right = current._desired_ee_twists()
    assert torch.equal(current_left, torch.ones((3, 6)))
    assert torch.equal(current_right, torch.full((3, 6), 2.0))


def test_prephysics_restores_internal_left_reference_without_overwriting_headroom() -> None:
    env = make_env("insert_joint_hold")
    env._transition_where(torch.tensor([True, False, False]), Phase.INSERT, 0.0)
    actual_before = env._joint_targets.clone()
    env._pre_physics_step(torch.zeros((3, 6)))
    assert torch.equal(env._joint_targets, actual_before + 0.5)
    assert torch.equal(env._insert_joint_targets[0, :2], torch.tensor([1.0, 2.0]))
    assert torch.equal(env._insert_joint_targets[0, 2:], torch.tensor([40.0, 41.0]))


def test_partial_reset_clears_only_selected_support_rows() -> None:
    env = make_env("insert_joint_hold")
    env._transition_where(torch.tensor([True, True, True]), Phase.INSERT, 0.0)
    before_zero = env._support_insert_joint_reference[0].clone()
    before_two = env._support_insert_joint_reference[2].clone()
    env._configure_training_reset(torch.tensor([1]))
    assert torch.equal(env._support_insert_joint_reference[0], before_zero)
    assert torch.equal(env._support_insert_joint_reference[2], before_two)
    assert not env._support_insert_reference_valid[1]
    assert env._support_insert_reference_valid[0]
    assert env._support_insert_reference_valid[2]


def test_snapshot_and_terminal_metrics_are_finite_and_preserve_mode() -> None:
    env = make_env("insert_joint_hold")
    env._transition_where(torch.tensor([True, False, False]), Phase.INSERT, 0.0)
    env._pre_physics_step(torch.zeros((3, 6)))
    env._sim_step_counter += 1
    env._get_dones()
    snapshot = env._training_step_snapshot()
    metrics, finite = support_ablation_terminal_metrics(snapshot, 0)
    assert finite
    assert metrics["support_arm_mode"] == "insert_joint_hold"
    assert metrics["support_insert_reference_valid"] is True
    assert metrics["support_insert_duration_s"] == pytest.approx(0.1)
    assert all(
        value is None or not isinstance(value, float) or torch.isfinite(torch.tensor(value))
        for value in metrics.values()
    )
    assert snapshot["support_insert_joint_reference_rad"].shape == (3, 2)
    env._configure_training_reset(torch.tensor([0]))
    assert snapshot["support_insert_reference_valid"][0]
    assert not env._support_insert_reference_valid[0]


@pytest.mark.parametrize("terminal_phase", [Phase.RETRACT, Phase.COMPLETE, Phase.ABORT])
def test_postphysics_terminal_boundary_is_sampled_once_before_snapshot(terminal_phase) -> None:
    class TerminalTrainingEnv(FakeTrainingEnv):
        def _get_dones(self):
            self._phase[0] = terminal_phase
            self.terminal_snapshot = self._training_step_snapshot()
            return super()._get_dones()

    env = make_rj45_support_ablation_env_type(TerminalTrainingEnv, "insert_joint_hold")()
    env._transition_where(torch.tensor([True, False, False]), Phase.INSERT, 0.0)
    env._pre_physics_step(torch.zeros((3, 6)))
    # Physics changes the state after command construction and before done checks.
    env.robot.data.body_pos_w[0, 0, 0] += 1.0
    env.socket.data.root_pos_w[0, 0] += 2.0
    env._sim_step_counter += 8
    env._get_dones()
    first = env.terminal_snapshot
    assert first["support_left_ee_entry_translation_final_m"][0] == pytest.approx(1.0)
    assert first["support_left_ee_entry_translation_max_m"][0] == pytest.approx(1.0)
    assert first["support_left_ee_entry_translation_rms_m"][0] == pytest.approx(1.0)
    assert first["support_socket_goal_translation_final_m"][0] > 2.0
    assert first["support_insert_duration_s"][0] == pytest.approx(0.1)
    env._get_dones()
    assert env._support_metric_count[0] == 1
    env._pre_physics_step(torch.zeros((3, 6)))
    env._sim_step_counter += 8
    env._get_dones()
    assert env._support_metric_count[0] == 1
    env._configure_training_reset(torch.tensor([0]))
    assert first["support_left_ee_entry_translation_final_m"][0] == pytest.approx(1.0)


def test_evaluator_exposes_opt_in_mode_and_preserves_current_default() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts/evaluate_policy.py"
    ).read_text(encoding="utf-8")
    assert '"--support-arm-mode"' in source
    assert 'default="cartesian_correction"' in source
    assert "make_rj45_support_ablation_env_type(" in source
    assert 'report["support_arm_mode"] = args.support_arm_mode' in source
    assert "support_ablation_terminal_metrics(snapshot, env_id)" in source
