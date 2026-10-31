from pathlib import Path
import tempfile
import unittest

import torch
from skrl.agents.torch.ppo.ppo import compute_gae as skrl_compute_gae

from rby1_assembly.rj45_comparison_rl import (
    ACTION_DIM,
    OBSERVATION_DIM,
    absorbing_done,
    absorbing_gae_returns,
    collect_bounded,
    deterministic_action,
    finite_losses,
    load_inference_checkpoint,
    make_agent,
    parameter_vector,
    save_inference_checkpoint,
)


class _SyntheticEnv:
    def __init__(self, num_envs=4, horizon=4):
        self.num_envs = num_envs
        self.horizon = horizon
        self.count = torch.zeros(num_envs, dtype=torch.int64)
        self.obs = torch.zeros(num_envs, OBSERVATION_DIM)
        self.obs[:, :ACTION_DIM] = 0.35

    def reset(self):
        self.count.zero_()
        return self.obs.clone()

    def step(self, actions):
        self.count += 1
        reward = -((actions - 0.35) ** 2).mean(dim=-1, keepdim=True)
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        truncated = (self.count >= self.horizon).unsqueeze(1)
        self.count[truncated.squeeze(1)] = 0
        return self.obs.clone(), reward, terminated, truncated, {}


class _IsaacShapeSyntheticEnv(_SyntheticEnv):
    def reset(self):
        return {"policy": super().reset()}, {"reset": True}

    def step(self, actions):
        observation, reward, terminated, truncated, info = super().step(actions)
        return (
            {"policy": observation},
            reward.squeeze(-1),
            terminated.squeeze(-1),
            truncated.squeeze(-1),
            info,
        )


class _WarmupSyntheticEnv(_SyntheticEnv):
    def reset(self):
        return {"policy": super().reset()}, {
            "policy_observation_valid": torch.zeros(self.num_envs, dtype=torch.bool)
        }

    def step(self, actions):
        observation, reward, terminated, truncated, info = super().step(actions)
        info["policy_observation_valid"] = torch.ones(
            self.num_envs, dtype=torch.bool
        )
        return {"policy": observation}, reward, terminated, truncated, info


class _MixedValidityEnv(_SyntheticEnv):
    def reset(self):
        self._valid = torch.tensor([True, False])
        return {"policy": super().reset()}, {
            "policy_observation_valid": self._valid.clone()
        }

    def step(self, actions):
        observation, reward, terminated, truncated, info = super().step(actions)
        self._valid = ~self._valid
        info["policy_observation_valid"] = self._valid.clone()
        return {"policy": observation}, reward, terminated, truncated, info


class _FixedRewardEnv(_SyntheticEnv):
    def step(self, actions):
        self.count += 1
        reward = self.count.to(torch.float32).unsqueeze(1)
        terminated = torch.zeros(self.num_envs, 1, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        return self.obs.clone(), reward, terminated, truncated, {}


class _AlternatingSingleEnv(_SyntheticEnv):
    def reset(self):
        self._valid = True
        return {"policy": super().reset()}, {
            "policy_observation_valid": torch.tensor([self._valid])
        }

    def step(self, actions):
        self.count += 1
        reward = torch.zeros(1, 1)
        terminated = torch.zeros(1, 1, dtype=torch.bool)
        truncated = torch.zeros_like(terminated)
        self._valid = not self._valid
        return (
            {"policy": self.obs.clone()},
            reward,
            terminated,
            truncated,
            {"policy_observation_valid": torch.tensor([self._valid])},
        )


class ComparisonContractTests(unittest.TestCase):
    def test_zero_mean_initializer_is_exact_for_all_observations(self):
        observations = torch.randn(37, OBSERVATION_DIM)
        for algorithm in ("ppo", "sac"):
            agent = make_agent(
                algorithm,
                num_envs=2,
                rollout_steps=2,
                batch_size=4,
                seed=17,
                mean_initializer="zero",
                initial_log_std=-3.0,
            )
            self.assertEqual(torch.count_nonzero(agent.policy.mean.weight).item(), 0)
            self.assertEqual(torch.count_nonzero(agent.policy.mean.bias).item(), 0)
            self.assertTrue(torch.equal(deterministic_action(agent, observations), torch.zeros(37, ACTION_DIM)))
            self.assertEqual(agent._comparison_mean_initializer, "zero")
            self.assertEqual(agent._comparison_initial_log_std, -3.0)
            self.assertEqual(agent._comparison_log_std_bounds, (-5.0, 1.0))

    def test_default_initializer_preserves_seeded_legacy_construction(self):
        for algorithm in ("ppo", "sac"):
            legacy = make_agent(algorithm, num_envs=2, rollout_steps=2, batch_size=4, seed=23)
            explicit = make_agent(
                algorithm,
                num_envs=2,
                rollout_steps=2,
                batch_size=4,
                seed=23,
                mean_initializer="legacy_random",
                initial_log_std=-1.0,
            )
            self.assertEqual(legacy.models.keys(), explicit.models.keys())
            for name in legacy.models:
                for key, value in legacy.models[name].state_dict().items():
                    self.assertTrue(torch.equal(value, explicit.models[name].state_dict()[key]))

    def test_zero_initializer_changes_only_policy_mean_head(self):
        for algorithm in ("ppo", "sac"):
            legacy = make_agent(algorithm, num_envs=2, rollout_steps=2, batch_size=4, seed=31)
            zero = make_agent(
                algorithm,
                num_envs=2,
                rollout_steps=2,
                batch_size=4,
                seed=31,
                mean_initializer="zero",
            )
            for model_name in legacy.models:
                for key, value in legacy.models[model_name].state_dict().items():
                    if model_name == "policy" and key in {"mean.weight", "mean.bias"}:
                        self.assertEqual(torch.count_nonzero(zero.models[model_name].state_dict()[key]).item(), 0)
                    else:
                        self.assertTrue(torch.equal(value, zero.models[model_name].state_dict()[key]))

    def test_initializer_contract_rejects_invalid_values(self):
        with self.assertRaisesRegex(ValueError, "mean_initializer"):
            make_agent("ppo", num_envs=1, mean_initializer="unknown")
        for value in (float("nan"), float("inf"), -5.01, 1.01, None, "invalid"):
            with self.assertRaisesRegex(ValueError, "initial_log_std"):
                make_agent("sac", num_envs=1, initial_log_std=value)

    def test_absorbing_timeout_and_gae_do_not_bootstrap_reset_observation(self):
        rewards = torch.tensor([[[1.0]], [[2.0]]])
        values = torch.tensor([[[0.5]], [[0.7]]])
        terminated = torch.zeros_like(rewards, dtype=torch.bool)
        truncated = torch.tensor([[[False]], [[True]]])
        returns, advantages = absorbing_gae_returns(
            rewards,
            values,
            terminated,
            truncated,
            last_values=torch.tensor([[999.0]]),
            discount_factor=1.0,
            gae_lambda=1.0,
        )
        self.assertTrue(torch.equal(absorbing_done(terminated, truncated), truncated))
        self.assertTrue(torch.allclose(returns[:, 0, 0], torch.tensor([3.0, 2.0])))
        self.assertTrue(torch.allclose(advantages[:, 0, 0], torch.tensor([2.5, 1.3])))

        skrl_returns, _ = skrl_compute_gae(
            rewards=rewards,
            terminated=absorbing_done(terminated, truncated),
            truncated=torch.zeros_like(truncated),
            values=values,
            last_values=torch.tensor([[999.0]]),
            discount_factor=1.0,
            lambda_coefficient=1.0,
            time_limit_bootstrap=False,
        )
        leaked_returns, _ = skrl_compute_gae(
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            values=values,
            last_values=torch.tensor([[999.0]]),
            discount_factor=1.0,
            lambda_coefficient=1.0,
            time_limit_bootstrap=False,
        )
        self.assertTrue(torch.allclose(skrl_returns[:, 0, 0], torch.tensor([3.0, 2.0])))
        self.assertGreater(float(leaked_returns[-1, 0, 0]), 900.0)

        middle_timeout_rewards = torch.tensor([[[1.0]], [[2.0]], [[1000.0]]])
        middle_timeout = torch.tensor([[[False]], [[True]], [[False]]])
        isolated_returns, _ = skrl_compute_gae(
            rewards=middle_timeout_rewards,
            terminated=middle_timeout,
            truncated=torch.zeros_like(middle_timeout),
            values=torch.zeros_like(middle_timeout_rewards),
            last_values=torch.zeros(1, 1),
            discount_factor=1.0,
            lambda_coefficient=1.0,
            time_limit_bootstrap=False,
        )
        self.assertTrue(
            torch.equal(isolated_returns[:, 0, 0], torch.tensor([3.0, 2.0, 1000.0]))
        )

    def test_common_model_rights(self):
        for algorithm in ("ppo", "sac"):
            agent = make_agent(algorithm, num_envs=2, rollout_steps=4, batch_size=4)
            self.assertEqual(agent.observation_space.shape, (79,))
            self.assertEqual(agent.action_space.shape, (6,))
            self.assertIsNone(agent.cfg.observation_preprocessor)
            self.assertIsNone(agent.cfg.state_preprocessor)
            for model in agent.models.values():
                self.assertEqual(model.observation_space.shape, (79,))
            if algorithm == "sac":
                observations = torch.randn(2, OBSERVATION_DIM)
                torch.manual_seed(17)
                actions, outputs = agent.policy.act(
                    {"observations": observations}, role="policy"
                )
                raw_mean, log_std = agent.policy.compute(
                    {"observations": observations}, role="policy"
                )
                raw_action = torch.atanh(actions.clamp(-0.999999, 0.999999))
                expected_log_prob = (
                    torch.distributions.Normal(raw_mean, log_std.exp()).log_prob(
                        raw_action
                    )
                    - torch.log(1.0 - actions.square() + 1.0e-6)
                ).sum(dim=-1, keepdim=True)
                self.assertTrue(torch.all(actions.abs() < 1.0))
                self.assertTrue(torch.allclose(outputs["log_prob"], expected_log_prob))
                self.assertTrue(
                    torch.allclose(outputs["mean_actions"], torch.tanh(raw_mean))
                )

    def test_collector_accepts_isaac_reset_dict_and_vector_flags(self):
        agent = make_agent("ppo", num_envs=2, rollout_steps=4, batch_size=4)
        result = collect_bounded(
            agent,
            _IsaacShapeSyntheticEnv(num_envs=2, horizon=2),
            algorithm="ppo",
            vector_steps=4,
        )
        self.assertEqual(result.transitions, 8)
        self.assertEqual(result.truncated_transitions, 4)

    def test_collector_separates_warmup_from_learning_transitions(self):
        agent = make_agent("ppo", num_envs=2, rollout_steps=4, batch_size=4)
        result = collect_bounded(
            agent,
            _WarmupSyntheticEnv(num_envs=2, horizon=10),
            algorithm="ppo",
            vector_steps=5,
            require_policy_validity=True,
        )
        self.assertEqual(result.transitions, 10)
        self.assertEqual(result.valid_policy_transitions, 8)
        self.assertEqual(result.learning_transitions, 8)
        self.assertEqual(result.warmup_transitions, 2)
        self.assertEqual(result.skipped_valid_transitions, 0)
        self.assertEqual(result.mixed_validity_vector_steps, 0)

    def test_mixed_valid_rows_are_preserved_for_ppo(self):
        agent = make_agent("ppo", num_envs=2, rollout_steps=2, batch_size=4)
        result = collect_bounded(
            agent,
            _MixedValidityEnv(num_envs=2, horizon=20),
            algorithm="ppo",
            vector_steps=5,
            require_policy_validity=True,
        )
        self.assertEqual(result.transitions, 10)
        self.assertEqual(result.valid_policy_transitions, 5)
        self.assertEqual(result.learning_transitions, 5)
        self.assertEqual(result.skipped_valid_transitions, 0)
        self.assertEqual(result.mixed_validity_vector_steps, 5)
        self.assertEqual(result.algorithm_updates, 1)
        self.assertEqual(result.pending_policy_transitions, 1)

    def test_mixed_valid_rows_are_inserted_individually_for_sac(self):
        agent = make_agent(
            "sac",
            num_envs=2,
            rollout_steps=2,
            replay_capacity=32,
            batch_size=2,
            sac_utd_ratio=1.0,
        )
        result = collect_bounded(
            agent,
            _MixedValidityEnv(num_envs=2, horizon=20),
            algorithm="sac",
            vector_steps=5,
            require_policy_validity=True,
        )
        self.assertEqual(result.valid_policy_transitions, 5)
        self.assertEqual(result.replay_insertions, 5)
        self.assertEqual(result.replay_size, 5)
        self.assertEqual(result.skipped_valid_transitions, 0)
        self.assertEqual(result.mixed_validity_vector_steps, 5)
        self.assertEqual(result.algorithm_updates, 5)
        self.assertEqual(result.optimizer_steps, 15)

    def test_ragged_ppo_chunk_bootstrap_matches_per_env_gae(self):
        agent = make_agent("ppo", num_envs=2, rollout_steps=2, batch_size=4, seed=3)
        env = _FixedRewardEnv(num_envs=2, horizon=20)
        observations = env.obs.clone()
        with torch.no_grad():
            values, _ = agent.value.act({"observations": observations}, role="value")
        expected_returns, _ = absorbing_gae_returns(
            torch.tensor([[[1.0]], [[2.0]]]),
            values[:1].repeat(2, 1).reshape(2, 1, 1),
            torch.zeros(2, 1, 1, dtype=torch.bool),
            torch.zeros(2, 1, 1, dtype=torch.bool),
            last_values=values[:1],
            discount_factor=agent.cfg.discount_factor,
            gae_lambda=agent.cfg.gae_lambda,
        )
        result = collect_bounded(agent, env, algorithm="ppo", vector_steps=2)
        actual = agent.memory.get_tensor_by_name("returns")[:, 0, 0]
        expected = expected_returns[:, 0, 0].repeat(2)
        self.assertEqual(result.ppo_bootstrap_adjustments, 2)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-5))

    def test_two_collector_calls_close_reset_boundary_between_half_rollouts(self):
        agent = make_agent("ppo", num_envs=1, rollout_steps=4, batch_size=4, seed=9)
        with torch.no_grad():
            for parameter in agent.value.parameters():
                parameter.zero_()
            agent.value.net[-1].bias.fill_(10.0)
        env = _FixedRewardEnv(num_envs=1, horizon=20)
        first = collect_bounded(agent, env, algorithm="ppo", vector_steps=2)
        second = collect_bounded(agent, env, algorithm="ppo", vector_steps=2)
        segment_returns, _ = absorbing_gae_returns(
            torch.tensor([[[1.0]], [[2.0]]]),
            torch.full((2, 1, 1), 10.0),
            torch.zeros(2, 1, 1, dtype=torch.bool),
            torch.zeros(2, 1, 1, dtype=torch.bool),
            last_values=torch.tensor([[10.0]]),
            discount_factor=agent.cfg.discount_factor,
            gae_lambda=agent.cfg.gae_lambda,
        )
        expected = segment_returns[:, 0, 0].repeat(2)
        actual = agent.memory.get_tensor_by_name("returns")[:, 0, 0]
        self.assertEqual(first.pending_policy_transitions, 2)
        self.assertEqual(first.algorithm_updates, 0)
        self.assertEqual(second.algorithm_updates, 1)
        self.assertEqual(second.ppo_bootstrap_adjustments, 2)
        self.assertTrue(torch.allclose(actual, expected, atol=1.0e-5))

    def test_validity_gaps_preserve_raw_done_and_bootstrap_every_segment(self):
        agent = make_agent("ppo", num_envs=1, rollout_steps=4, batch_size=4, seed=4)
        with torch.no_grad():
            for parameter in agent.value.parameters():
                parameter.zero_()
            agent.value.net[-1].bias.fill_(10.0)
        result = collect_bounded(
            agent,
            _AlternatingSingleEnv(num_envs=1, horizon=20),
            algorithm="ppo",
            vector_steps=8,
            require_policy_validity=True,
        )
        returns = agent.memory.get_tensor_by_name("returns")[:, 0, 0]
        self.assertEqual(result.valid_policy_transitions, 4)
        self.assertEqual(result.warmup_transitions, 4)
        self.assertEqual(result.ppo_bootstrap_adjustments, 4)
        self.assertFalse(agent.memory.get_tensor_by_name("truncated").any())
        self.assertTrue(
            torch.allclose(returns, torch.full_like(returns, 9.9), atol=1.0e-5)
        )

    def test_ppo_and_sac_update_count_losses_and_checkpoint(self):
        for algorithm in ("ppo", "sac"):
            agent = make_agent(
                algorithm,
                num_envs=4,
                rollout_steps=8,
                replay_capacity=256,
                batch_size=16,
                seed=5,
            )
            before = parameter_vector(agent)
            result = collect_bounded(agent, _SyntheticEnv(), algorithm=algorithm, vector_steps=16)
            after = parameter_vector(agent)
            self.assertEqual(result.transitions, 64)
            self.assertEqual(result.transitions, result.num_envs * result.vector_steps)
            self.assertEqual(result.truncated_transitions, 16)
            self.assertEqual(result.absorbing_transitions, 16)
            self.assertTrue(torch.any(before != after))
            self.assertTrue(torch.isfinite(after).all())
            self.assertTrue(finite_losses(agent))
            if algorithm == "sac":
                stored_done = agent.memory.get_tensor_by_name("terminated")[:64]
                self.assertEqual(int(stored_done.sum().item()), 16)
                self.assertEqual(result.replay_insertions, 64)
                self.assertEqual(result.replay_size, 64)
                self.assertEqual(result.replay_first_ready_at, 16)
                self.assertEqual(result.random_warmup_transitions, 16)
                self.assertEqual(result.algorithm_updates, 64)
                self.assertEqual(result.optimizer_steps, 192)
                policy_steps = {
                    int(value["step"].item())
                    for value in agent.policy_optimizer.state.values()
                }
                critic_steps = {
                    int(value["step"].item())
                    for value in agent.critic_optimizer.state.values()
                }
                entropy_steps = {
                    int(value["step"].item())
                    for value in agent.entropy_optimizer.state.values()
                }
                self.assertEqual(policy_steps, {64})
                self.assertEqual(critic_steps, {64})
                self.assertEqual(entropy_steps, {64})
            else:
                self.assertFalse(agent.cfg.time_limit_bootstrap)
                self.assertEqual(
                    int(agent.memory.get_tensor_by_name("terminated").sum().item()), 8
                )
                self.assertFalse(agent.memory.get_tensor_by_name("truncated").any())
                self.assertEqual(result.algorithm_updates, 2)
                self.assertEqual(result.optimizer_steps, 8)
                optimizer_steps = {
                    int(value["step"].item()) for value in agent.optimizer.state.values()
                }
                self.assertEqual(optimizer_steps, {8})

            probe = torch.randn(3, OBSERVATION_DIM)
            expected = deterministic_action(agent, probe)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.pt"
                save_inference_checkpoint(agent, path, algorithm=algorithm)
                restored = make_agent(
                    algorithm,
                    num_envs=4,
                    rollout_steps=8,
                    replay_capacity=256,
                    batch_size=16,
                    seed=99,
                )
                payload = load_inference_checkpoint(restored, path)
                actual = deterministic_action(restored, probe)
                self.assertFalse(payload["resume_supported"])
                self.assertIsNone(payload["running_scaler"])
                self.assertTrue(torch.equal(expected, actual))
                with self.assertRaisesRegex(RuntimeError, "resume is disabled"):
                    load_inference_checkpoint(restored, path, resume=True)
                other_algorithm = "sac" if algorithm == "ppo" else "ppo"
                wrong_agent = make_agent(
                    other_algorithm,
                    num_envs=4,
                    rollout_steps=8,
                    replay_capacity=256,
                    batch_size=16,
                )
                with self.assertRaisesRegex(ValueError, "algorithm does not match"):
                    load_inference_checkpoint(wrong_agent, path)


if __name__ == "__main__":
    unittest.main()
