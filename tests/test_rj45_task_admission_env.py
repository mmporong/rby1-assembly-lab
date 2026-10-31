from __future__ import annotations

import torch

from rby1_assembly.rj45_task_admission_env import (
    make_rj45_task_admission_env_type,
)


class FakeComparisonEnv:
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = "cpu"
        self.extras = {}
        self._training_initialized = False
        self._initialize_training_state()
        self._training_initialized = True
        self._configure_training_reset(torch.arange(3))

    def _initialize_training_state(self) -> None:
        self._training_translation_bias_entry = torch.zeros((3, 3))
        self._training_rotation_bias_entry = torch.zeros((3, 3))
        self._training_last_policy_observation_valid = torch.zeros(3, dtype=torch.bool)
        self._comparison_applied_action_6d = torch.zeros((3, 6))
        self._training_applied_tip_offset_entry = torch.zeros((3, 6))
        self._phase = torch.full((3,), 3, dtype=torch.long)
        self.transition_rows = torch.zeros(3, dtype=torch.bool)
        self.relative_position = torch.tensor(
            [[0.001, 0.002, 0.003], [0.004, 0.005, 0.006], [0.007, 0.008, 0.009]]
        )
        self.relative_quaternion = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3
        )

    def _sample_episode_bias(self, env_ids) -> None:
        self._training_translation_bias_entry[env_ids] = 99.0
        self._training_rotation_bias_entry[env_ids] = 99.0

    def _configure_training_reset(self, env_ids) -> None:
        self._sample_episode_bias(env_ids)
        self._training_last_policy_observation_valid[env_ids] = False

    def _relative_object_pose(self):
        return self.relative_position.clone(), self.relative_quaternion.clone()

    def _freeze_training_estimator(self, _mask) -> None:
        return None

    def _advance_phases(self) -> None:
        self._phase[self.transition_rows] = 4

    def _pre_physics_step(self, actions) -> None:
        self._advance_phases()
        gate = self._training_last_policy_observation_valid[:, None]
        self._comparison_applied_action_6d[:] = torch.where(
            gate, actions, torch.zeros_like(actions)
        )
        self._training_applied_tip_offset_entry[:] = (
            self._comparison_applied_action_6d
        )

    def _training_step_snapshot(self):
        return {"base": True}


def make_env():
    return make_rj45_task_admission_env_type(
        FakeComparisonEnv,
        requested_translation_bias_entry_m=(0.00012, 0.0, 0.0),
        requested_rotation_bias_entry_rad=(0.0, 0.0, 0.003),
        held_grasp_id="rj45-train-02",
        held_grasp_index=1,
        held_artifact_sha256="a" * 64,
    )()


def test_reset_injects_exact_bias_and_held_identity() -> None:
    env = make_env()
    assert torch.allclose(
        env._training_translation_bias_entry,
        torch.tensor([[0.00012, 0.0, 0.0]] * 3),
        rtol=0.0,
        atol=1.0e-9,
    )
    assert torch.allclose(
        env._training_rotation_bias_entry,
        torch.tensor([[0.0, 0.0, 0.003]] * 3),
        rtol=0.0,
        atol=1.0e-9,
    )
    assert env.extras["rj45_task_admission_identity"] == {
        "applied_held_grasp_id": "rj45-train-02",
        "applied_held_grasp_index": 1,
        "applied_held_artifact_sha256": "a" * 64,
    }


def test_prefreeze_snapshot_is_captured_before_parent_freeze() -> None:
    env = make_env()
    env._freeze_training_estimator(torch.tensor([True, False, True]))
    snapshot = env._admission_snapshot()
    assert snapshot["prefreeze_snapshot_valid"].tolist() == [True, False, True]
    assert torch.equal(
        snapshot["prefreeze_translation_socket_entry_m_xyz"][[0, 2]].float(),
        env.relative_position[[0, 2]],
    )
    assert not snapshot[
        "prefreeze_shortest_rotation_vector_socket_entry_rad_xyz"
    ].any()


def test_align_to_insert_action_and_tip_offset_are_exact_zero() -> None:
    env = make_env()
    env.transition_rows[:] = torch.tensor([True, False, True])
    requested = torch.full((3, 6), 0.75)

    env._pre_physics_step(requested)

    snapshot = env.extras["rj45_task_admission"]
    assert snapshot["insert_transition_snapshot_valid"].tolist() == [True, False, True]
    assert snapshot["transition_contract_checked"].tolist() == [True, False, True]
    assert snapshot["actual_applied_action_is_exact_zero"].tolist() == [True, False, True]
    assert not snapshot["actual_applied_action_6d"][[0, 2]].any()
    assert not snapshot["actual_applied_tip_target_offset_entry_6d"][[0, 2]].any()
    assert torch.equal(snapshot["requested_action_6d"][[0, 2]], requested[[0, 2]])


def test_action_after_first_valid_insert_observation_is_applied() -> None:
    env = make_env()
    env._phase[:] = 4
    env._training_last_policy_observation_valid[:] = True
    requested = torch.full((3, 6), 0.5)

    env._pre_physics_step(requested)

    assert torch.equal(env._comparison_applied_action_6d, requested)


def test_partial_reset_clears_only_selected_admission_state() -> None:
    env = make_env()
    env.transition_rows[:] = True
    env._pre_physics_step(torch.zeros((3, 6)))
    env._configure_training_reset(torch.tensor([1]))
    assert env._admission_insert_transition_snapshot_valid.tolist() == [True, False, True]
    assert env._admission_transition_contract_checked.tolist() == [True, False, True]
    assert env._admission_transition_actual_action_is_exact_zero.tolist() == [True, False, True]


def test_fair_provider_snapshot_never_exports_gt_as_provider_input() -> None:
    env = make_env()
    snapshot = env._admission_snapshot()
    assert snapshot["evaluator_gt_available_to_fair_provider"] is False
    assert "observation79" not in snapshot
