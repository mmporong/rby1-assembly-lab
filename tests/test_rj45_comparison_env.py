from __future__ import annotations

import torch

from rby1_assembly.rj45_comparison_env import (
    comparison_observation_metadata,
    make_rj45_comparison_env_type,
)


class FakeTrainingEnv:
    def __init__(self) -> None:
        self.num_envs = 3
        self.device = "cpu"
        self._training_config = {
            "base_observation_dim": 79,
            "diagnostic_observation_mode": "standard",
            "action_mode": "right_only_6d",
            "reward": {"failure_penalty": 7.0},
        }
        self._training_initialized = False
        self.extras = {}
        self.relative_super_calls = 0
        self.object_super_calls = 0
        self.grasp_super_calls = 0
        self.base_terminated = torch.zeros(3, dtype=torch.bool)
        self.base_truncated = torch.zeros(3, dtype=torch.bool)
        self.base_rewards = torch.ones(3)
        self.base_policy_active = torch.ones(3, dtype=torch.bool)
        self.executed_actions = None
        self.prephysics_control_relative = None
        self.phase = torch.zeros(3, dtype=torch.long)
        self.advance_to_insert = torch.zeros(3, dtype=torch.bool)
        self._initialize_training_state()
        self._training_initialized = True

    def _initialize_training_state(self) -> None:
        self._training_estimator_valid = torch.tensor([True, True, True])
        self._training_policy_activated = torch.tensor([True, True, True])
        self._training_last_policy_observation_valid = torch.tensor(
            [True, True, True]
        )
        self._training_estimated_relative_pos = torch.tensor(
            [[1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]]
        )
        self._training_estimated_relative_quat = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0]] * 3
        )
        self._training_estimated_object_pos = torch.arange(18.0).reshape(3, 2, 3)
        self._training_estimated_object_quat = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]] * 2] * 3
        )
        self._training_frozen_object_to_ee_pos = (
            torch.arange(18.0).reshape(3, 2, 3) + 100.0
        )
        self._training_frozen_object_to_ee_quat = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0]] * 2] * 3
        )

    def _control_relative_object_pose(self):
        self.relative_super_calls += 1
        return torch.full((3, 3), 90.0), torch.full((3, 4), 91.0)

    def _control_object_poses(self):
        self.object_super_calls += 1
        return tuple(torch.full((3, width), 92.0 + index) for index, width in enumerate((3, 4, 3, 4)))

    def _control_grasp_transforms(self, *_args):
        self.grasp_super_calls += 1
        return tuple(torch.full((3, width), 96.0 + index) for index, width in enumerate((3, 4, 3, 4)))

    def _policy_active_mask(self):
        return self.base_policy_active.clone()

    def _pre_physics_step(self, actions):
        self.phase[self.advance_to_insert] = 1
        self.executed_actions = torch.where(
            self._policy_active_mask()[:, None], actions, torch.zeros_like(actions)
        )
        self._actions = torch.cat(
            (torch.zeros((3, 6)), self.executed_actions), dim=1
        )
        self.prephysics_control_relative = self._control_relative_object_pose()

    def _get_dones(self):
        self.extras["terminal_snapshot"] = {
            "terminal_mask": (self.base_terminated | self.base_truncated).clone()
        }
        return self.base_terminated.clone(), self.base_truncated.clone()

    def _training_step_snapshot(self):
        return {"base": torch.ones(3)}

    def _get_rewards(self):
        return self.base_rewards.clone()

    def _configure_training_reset(self, env_ids):
        self._training_estimator_valid[env_ids] = False
        self._training_policy_activated[env_ids] = False
        self._training_last_policy_observation_valid[env_ids] = False


def make_env() -> FakeTrainingEnv:
    return make_rj45_comparison_env_type(FakeTrainingEnv)()


def test_active_invalid_mixed_batch_never_calls_ground_truth_fallback() -> None:
    env = make_env()
    env._training_estimator_valid[:] = torch.tensor([True, False, True])

    relative = env._control_relative_object_pose()
    objects = env._control_object_poses()
    grasps = env._control_grasp_transforms("ignored")

    assert env.relative_super_calls == 0
    assert env.object_super_calls == 0
    assert env.grasp_super_calls == 0
    assert torch.equal(relative[0], env._training_estimated_relative_pos)
    assert torch.equal(objects[0], env._training_estimated_object_pos[:, 0])
    assert torch.equal(grasps[2], env._training_frozen_object_to_ee_pos[:, 1])

    terminated, truncated = env._get_dones()
    assert torch.equal(terminated, torch.tensor([False, True, False]))
    assert not truncated.any()
    assert torch.equal(
        env.extras["terminal_snapshot"]["comparison_observation_failure"],
        torch.tensor([False, True, False]),
    )
    aggregate = env.extras["rj45_comparison_aggregate"]
    assert aggregate["activated_episode_terminal_count"].tolist() == [0, 1, 0]
    assert aggregate["observation_failure_count"].tolist() == [0, 1, 0]
    rewards = env._get_rewards()
    assert rewards.tolist() == [1.0, -6.0, 1.0]
    assert env._get_rewards().tolist() == [1.0, 1.0, 1.0]


def test_active_invalid_is_action_and_observation_inactive_before_physics() -> None:
    env = make_env()
    env._training_estimator_valid[:] = torch.tensor([True, False, True])
    actions = torch.ones((3, 6))

    env._pre_physics_step(actions)

    assert torch.equal(
        env.executed_actions,
        torch.tensor([[1.0] * 6, [0.0] * 6, [1.0] * 6]),
    )
    assert env._policy_active_mask().tolist() == [True, False, True]
    assert env.relative_super_calls == 0
    assert torch.equal(
        env.prephysics_control_relative[0], env._training_estimated_relative_pos
    )
    assert env._comparison_observation_failure_pending.tolist() == [False, True, False]


def test_align_action_is_zero_when_base_advances_to_insert_before_physics() -> None:
    env = make_env()
    env.phase[:] = 0
    env.advance_to_insert[:] = True
    env._training_last_policy_observation_valid[:] = False
    actions = torch.full((3, 6), 0.75)

    env._pre_physics_step(actions)

    assert env.phase.tolist() == [1, 1, 1]
    assert not env.executed_actions.any()
    telemetry = env.extras["rj45_comparison_action_gate"]
    assert telemetry["zero_gated"].tolist() == [True, True, True]
    assert not telemetry["applied_action_6d"].any()


def test_first_valid_insert_observation_enables_the_next_action() -> None:
    env = make_env()
    env.phase[:] = 1
    env._training_last_policy_observation_valid[:] = True
    actions = torch.full((3, 6), 0.5)

    env._pre_physics_step(actions)

    assert torch.equal(env.executed_actions, actions)
    telemetry = env.extras["rj45_comparison_action_gate"]
    assert telemetry["last_policy_observation_valid"].all()
    assert torch.equal(telemetry["applied_action_6d"], actions)


def test_invalid_and_partial_reset_rows_are_isolated_by_action_gate() -> None:
    env = make_env()
    env._training_last_policy_observation_valid[:] = torch.tensor(
        [True, False, True]
    )
    env._training_estimator_valid[:] = torch.tensor([True, True, False])
    actions = torch.arange(18.0).reshape(3, 6)

    env._pre_physics_step(actions)

    assert torch.equal(env.executed_actions[0], actions[0])
    assert not env.executed_actions[1].any()
    assert not env.executed_actions[2].any()
    env._configure_training_reset(torch.tensor([0]))
    assert env._comparison_action_observation_gate.tolist() == [False, False, True]
    assert env._comparison_action_zero_gated.tolist() == [True, True, False]
    assert not env._comparison_applied_action_6d[0].any()
    assert not env._training_last_policy_observation_valid[0]


def test_prepolicy_invalid_uses_legacy_fallback_and_is_counted_separately() -> None:
    env = make_env()
    env._training_estimator_valid[:] = torch.tensor([True, False, True])
    env._training_policy_activated[:] = torch.tensor([True, False, True])

    relative = env._control_relative_object_pose()
    terminated, _ = env._get_dones()

    assert env.relative_super_calls == 1
    assert torch.equal(relative[0][1], torch.full((3,), 90.0))
    assert not terminated.any()
    aggregate = env.extras["rj45_comparison_aggregate"]
    assert aggregate["activated_episode_terminal_count"].sum().item() == 0
    assert aggregate["observation_failure_count"].sum().item() == 0
    assert aggregate["prepolicy_invalid_episode_count"].tolist() == [0, 1, 0]


def test_three_way_batch_selects_proxy_failure_proxy_and_prepolicy_truth() -> None:
    env = make_env()
    env._training_estimator_valid[:] = torch.tensor([True, False, False])
    env._training_policy_activated[:] = torch.tensor([True, True, False])

    relative = env._control_relative_object_pose()
    objects = env._control_object_poses()
    grasps = env._control_grasp_transforms()

    assert env.relative_super_calls == 1
    assert env.object_super_calls == 1
    assert env.grasp_super_calls == 1
    assert torch.equal(relative[0][0], env._training_estimated_relative_pos[0])
    assert torch.equal(relative[0][1], env._training_estimated_relative_pos[1])
    assert torch.equal(relative[0][2], torch.full((3,), 90.0))
    assert torch.equal(objects[0][1], env._training_estimated_object_pos[1, 0])
    assert torch.equal(objects[0][2], torch.full((3,), 92.0))
    assert torch.equal(grasps[2][1], env._training_frozen_object_to_ee_pos[1, 1])
    assert torch.equal(grasps[2][2], torch.full((3,), 98.0))
    assert env._comparison_observation_failure_pending.tolist() == [False, True, False]
    assert env._comparison_prepolicy_invalid_episode_count.tolist() == [0, 0, 1]


def test_valid_rows_preserve_proxy_outputs_and_base_done_masks() -> None:
    env = make_env()
    env.base_terminated[0] = True
    relative = env._control_relative_object_pose()
    terminated, truncated = env._get_dones()

    assert torch.equal(relative[0], env._training_estimated_relative_pos)
    assert env.relative_super_calls == 0
    assert torch.equal(terminated, torch.tensor([True, False, False]))
    assert not truncated.any()
    assert env.extras["rj45_comparison_aggregate"][
        "activated_episode_terminal_count"
    ].tolist() == [1, 0, 0]


def test_partial_reset_clears_only_selected_episode_state() -> None:
    env = make_env()
    env._comparison_observation_failure_pending[:] = True
    env._comparison_observation_failure_episode[:] = True
    env._comparison_prepolicy_invalid_episode_seen[:] = True
    env._comparison_terminal_counted[:] = True
    env._comparison_observation_failure_reward_emitted[:] = True
    env._comparison_activated_episode_terminal_count[:] = torch.tensor([2, 3, 4])

    env._configure_training_reset(torch.tensor([1]))

    assert env._comparison_observation_failure_pending.tolist() == [True, False, True]
    assert env._comparison_observation_failure_episode.tolist() == [True, False, True]
    assert env._comparison_prepolicy_invalid_episode_seen.tolist() == [True, False, True]
    assert env._comparison_terminal_counted.tolist() == [True, False, True]
    assert env._comparison_observation_failure_reward_emitted.tolist() == [True, False, True]
    assert env._comparison_activated_episode_terminal_count.tolist() == [2, 3, 4]
    assert env._training_estimator_valid.tolist() == [True, False, True]
    assert env._training_policy_activated.tolist() == [True, False, True]


def test_active_timeout_is_absorbing_and_reset_observation_is_not_bootstrapped() -> None:
    env = make_env()
    env.base_truncated[2] = True

    terminated, truncated = env._get_dones()

    assert torch.equal(terminated, torch.tensor([False, False, True]))
    assert not truncated.any()
    snapshot = env.extras["terminal_snapshot"]
    assert snapshot["comparison_timeout_absorbing_failure"].tolist() == [False, False, True]
    assert snapshot["comparison_absorbing_failure"].tolist() == [False, False, True]
    assert snapshot["comparison_final_observation_available"] is False
    assert snapshot["comparison_bootstrap_reset_observation"] is False
    assert env.extras["rj45_comparison_aggregate"]["timeout_failure_count"].tolist() == [0, 0, 1]


def test_metadata_is_explicit_about_simulation_proxy_and_camera_limit() -> None:
    metadata = comparison_observation_metadata()
    assert metadata["actor_observation_dimension"] == 79
    assert metadata["critic_observation_dimension"] == 79
    assert metadata["residual_action_dimension"] == 6
    assert metadata["actual_camera_measurement"] is False
    assert "simulator_ground_truth_pose" in metadata["proxy_initialization"]
    assert metadata["bootstrap_reset_observation"] is False
    assert metadata["external_requested_case_denominator_required"] is True
    assert "not the external requested-case" in metadata["aggregate_denominator_scope"]
