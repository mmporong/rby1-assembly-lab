from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import torch

from rby1_assembly.rj45_comparison_pilot_env import (
    PilotEnvironmentContractViolation,
    make_rj45_comparison_pilot_env_type,
)


def _conditions():
    biases = (
        ("tx-positive", (0.00012, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("tx-negative", (-0.00012, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ("tz-positive", (0.0, 0.0, 0.00012), (0.0, 0.0, 0.0)),
        ("tz-negative", (0.0, 0.0, -0.00012), (0.0, 0.0, 0.0)),
        ("rx-positive", (0.0, 0.0, 0.0), (0.003490658503988659, 0.0, 0.0)),
        ("rx-negative", (0.0, 0.0, 0.0), (-0.003490658503988659, 0.0, 0.0)),
    )
    return [
        {
            "condition_id": f"{grasp}-{case}",
            "grasp_id": grasp,
            "held_grasp_index": index,
            "held_artifact_sha256": digest,
            "translation_bias_entry_m": translation,
            "rotation_vector_bias_entry_rad": rotation,
        }
        for grasp, index, digest in (
            ("rj45-train-02", 1, "a" * 64),
            ("rj45-train-05", 4, "b" * 64),
        )
        for case, translation, rotation in biases
    ]


class FakeComparisonEnv:
    def __init__(self, num_envs=4):
        self.num_envs = num_envs
        self.device = "cpu"
        self.cfg = SimpleNamespace(seed=1234)
        self.extras = {}
        self._initialize_training_state()
        self._configure_training_reset(torch.arange(num_envs))

    def reset(self):
        self._phase[:] = 3
        self._configure_training_reset(torch.arange(self.num_envs))
        return {"wave": getattr(self, "_pilot_wave_sequence_index", 0)}

    def _initialize_training_state(self):
        self._training_translation_bias_entry = torch.zeros((self.num_envs, 3))
        self._training_rotation_bias_entry = torch.zeros((self.num_envs, 3))
        self._training_last_policy_observation_valid = torch.zeros(
            self.num_envs, dtype=torch.bool
        )
        self._comparison_applied_action_6d = torch.zeros((self.num_envs, 6))
        self._training_applied_tip_offset_entry = torch.zeros((self.num_envs, 6))
        self._phase = torch.full((self.num_envs,), 3, dtype=torch.long)
        self.transition_rows = torch.zeros(self.num_envs, dtype=torch.bool)

    def _sample_episode_bias(self, env_ids):
        self._training_translation_bias_entry[env_ids] = 99.0
        self._training_rotation_bias_entry[env_ids] = 99.0

    def _configure_training_reset(self, env_ids):
        self._sample_episode_bias(env_ids)
        self._training_last_policy_observation_valid[env_ids] = False

    def _advance_phases(self):
        self._phase[self.transition_rows] = 4

    def _pre_physics_step(self, actions):
        self._advance_phases()
        valid = self._training_last_policy_observation_valid[:, None]
        self._comparison_applied_action_6d[:] = torch.where(
            valid, actions, torch.zeros_like(actions)
        )
        self._training_applied_tip_offset_entry[:] = self._comparison_applied_action_6d

    def _training_step_snapshot(self):
        return {"base": True}


def _make_env():
    return make_rj45_comparison_pilot_env_type(
        FakeComparisonEnv, conditions=_conditions()
    )()


def test_fixed_12_rows_apply_exact_condition_biases_and_identity() -> None:
    env = _make_env()
    expected_translation = torch.tensor(
        [row["translation_bias_entry_m"] for row in _conditions()[:4]]
    )
    expected_rotation = torch.tensor(
        [row["rotation_vector_bias_entry_rad"] for row in _conditions()[:4]]
    )
    assert torch.allclose(env._training_translation_bias_entry, expected_translation)
    assert torch.allclose(env._training_rotation_bias_entry, expected_rotation)
    snapshot = env.extras["rj45_comparison_pilot"]
    assert snapshot["condition_catalog_ids"] == [
        row["condition_id"] for row in _conditions()
    ]
    assert snapshot["active_condition_ids"] == [
        row["condition_id"] for row in _conditions()[:4]
    ]
    assert snapshot["evaluator_gt_available_to_policy"] is False


def test_environment_rng_scope_is_paired_and_does_not_mutate_actor_rng() -> None:
    class RandomFakeComparisonEnv(FakeComparisonEnv):
        def reset(self):
            super().reset()
            observation = torch.tensor(
                [[random.random(), float(np.random.random()), torch.rand(()).item()]]
            )
            return {"policy": observation}, {"policy_observation_valid": torch.ones(4)}

        def step(self, actions):
            del actions
            observation = torch.tensor(
                [[random.random(), float(np.random.random()), torch.rand(()).item()]]
            )
            return {"policy": observation}, None, None, None, {}

    env_type = make_rj45_comparison_pilot_env_type(
        RandomFakeComparisonEnv, conditions=_conditions()
    )
    first = env_type()
    second = env_type()

    random.seed(91)
    np.random.seed(92)
    torch.manual_seed(93)
    outer_python = random.getstate()
    outer_numpy = np.random.get_state()
    outer_torch = torch.random.get_rng_state().clone()
    first_reset = first.reset()
    assert random.getstate() == outer_python
    assert np.array_equal(np.random.get_state()[1], outer_numpy[1])
    assert torch.equal(torch.random.get_rng_state(), outer_torch)

    random.seed(191)
    np.random.seed(192)
    torch.manual_seed(193)
    second_reset = second.reset()
    assert torch.equal(first_reset[0]["policy"], second_reset[0]["policy"])
    first_progress = first.comparison_pilot_progress()
    second_progress = second.comparison_pilot_progress()
    assert first_progress.reset_rng_fingerprints == second_progress.reset_rng_fingerprints
    assert first_progress.initial_observation_sha256 == second_progress.initial_observation_sha256
    before_step = first_progress.environment_rng_fingerprint
    first.step(torch.zeros((4, 6)))
    assert first.comparison_pilot_progress().environment_rng_fingerprint != before_step
    first.advance_comparison_pilot_wave(1)
    second.advance_comparison_pilot_wave(1)
    assert (
        first.comparison_pilot_progress().reset_rng_fingerprints
        == second.comparison_pilot_progress().reset_rng_fingerprints
    )
    assert "gpu_kernel_bit_exactness_not_claimed" in first_progress.rng_isolation_scope


def test_wrong_vector_count_fails_closed() -> None:
    env_type = make_rj45_comparison_pilot_env_type(
        FakeComparisonEnv, conditions=_conditions()
    )
    try:
        env_type(num_envs=3)
    except PilotEnvironmentContractViolation as error:
        assert "four vector rows" in str(error)
    else:
        raise AssertionError("unproven vector environment count was accepted")


def test_align_to_insert_transition_remains_zero_then_policy_action_applies() -> None:
    env = _make_env()
    env.transition_rows[[0, 3]] = True
    env._pre_physics_step(torch.full((4, 6), 0.75))
    snapshot = env.extras["rj45_comparison_pilot"]
    assert snapshot["transition_contract_checked"][[0, 3]].all()
    assert snapshot["transition_actual_action_is_exact_zero"][[0, 3]].all()

    env.transition_rows[:] = False
    env._phase[:] = 4
    env._training_last_policy_observation_valid[:] = True
    actions = torch.full((4, 6), 0.25)
    env._pre_physics_step(actions)
    assert torch.equal(env._comparison_applied_action_6d, actions)


def test_partial_reset_reapplies_only_fixed_rows_and_clears_transition_state() -> None:
    env = _make_env()
    env.transition_rows[:] = True
    env._pre_physics_step(torch.zeros((4, 6)))
    env._training_translation_bias_entry[2] = 5.0
    env._configure_training_reset(torch.tensor([2]))
    assert torch.allclose(
        env._training_translation_bias_entry[2],
        torch.tensor(_conditions()[2]["translation_bias_entry_m"]),
    )
    assert not env._pilot_transition_checked[2]
    assert env._pilot_transition_checked[3]
    progress = env.comparison_pilot_progress()
    assert progress.transition_ever_checked[2]
    assert progress.transition_ever_exact_zero[2]
    try:
        progress.transition_ever_checked = ()
    except AttributeError:
        pass
    else:
        raise AssertionError("pilot progress must be immutable")


def test_three_synchronous_waves_cover_all_12_conditions_with_public_audit() -> None:
    env = _make_env()
    for sequence_index in range(6):
        env.transition_rows[:] = True
        env._pre_physics_step(torch.zeros((4, 6)))
        if sequence_index < 5:
            reset_result = env.advance_comparison_pilot_wave(sequence_index + 1)
            assert reset_result == {"wave": sequence_index + 1}
    progress = env.comparison_pilot_progress()
    assert progress.wave_sequence_index == 5
    assert progress.transition_ever_checked == (True,) * 12
    assert progress.transition_ever_exact_zero == (True,) * 12
    assert progress.transition_check_counts == (2,) * 12
    assert progress.transition_exact_zero_counts == (2,) * 12
    try:
        env.advance_comparison_pilot_wave(7)
    except PilotEnvironmentContractViolation as error:
        assert "exactly one" in str(error)
    else:
        raise AssertionError("pilot wave sequence skipped an index")
