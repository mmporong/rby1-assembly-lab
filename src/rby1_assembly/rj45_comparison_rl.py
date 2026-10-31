"""Fair PPO/SAC comparison primitives for the 79D RJ45 residual task.

The module deliberately owns no Isaac Sim environment construction. It keeps the
algorithm boundary small: both agents receive the already unit-scaled 79D tensor
returned by :func:`rby1_assembly.rj45_training.build_policy_observation`, emit the
same bounded 6D residual action, and have no privileged critic input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
from typing import Any, Protocol

import gymnasium
import torch
from torch import nn

OBSERVATION_DIM = 79
ACTION_DIM = 6
MIN_LOG_STD = -5.0
MAX_LOG_STD = 1.0
CHECKPOINT_FORMAT = "rby1-rj45-comparison-policy-v1"


class VectorEnvironment(Protocol):
    """Minimum tensor interface accepted by :func:`collect_bounded`."""

    num_envs: int

    def reset(self) -> Any: ...

    def step(
        self, actions: torch.Tensor
    ) -> tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]: ...


@dataclass(frozen=True)
class CollectorResult:
    algorithm: str
    vector_steps: int
    num_envs: int
    transitions: int
    valid_policy_transitions: int
    learning_transitions: int
    warmup_transitions: int
    skipped_valid_transitions: int
    mixed_validity_vector_steps: int
    algorithm_updates: int
    optimizer_steps: int
    random_warmup_transitions: int
    replay_insertions: int
    replay_size: int
    replay_first_ready_at: int
    pending_policy_transitions: int
    ppo_bootstrap_adjustments: int
    ppo_rollout_target: int
    sac_utd_ratio: float
    terminated_transitions: int
    truncated_transitions: int
    absorbing_transitions: int
    sac_random_warmup_requested: int | None
    sac_random_warmup_effective: int
    sac_learning_starts_requested: int | None
    sac_learning_starts_effective: int
    sac_first_update_after_valid_insertions: int


def absorbing_done(terminated: torch.Tensor, truncated: torch.Tensor) -> torch.Tensor:
    """Treat failure and the 24 s finite-horizon timeout as absorbing endings."""

    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean tensors")
    if terminated.shape != truncated.shape:
        raise ValueError("terminated and truncated must have the same shape")
    return terminated | truncated


def _policy_tensor(observations: Any) -> torch.Tensor:
    if isinstance(observations, tuple):
        observations = observations[0]
    if isinstance(observations, dict):
        observations = observations.get("policy")
    if not isinstance(observations, torch.Tensor):
        raise TypeError("observations must be a tensor or a mapping with a policy tensor")
    return observations


def _reset_observation(reset_result: Any) -> tuple[torch.Tensor, dict[str, Any]]:
    if isinstance(reset_result, tuple):
        observations, infos = reset_result
    else:
        observations, infos = reset_result, {}
    return _policy_tensor(observations), infos


def _policy_validity(
    infos: dict[str, Any], *, num_envs: int, device: torch.device, required: bool
) -> torch.Tensor:
    validity = infos.get("policy_observation_valid")
    if validity is None:
        if required:
            raise RuntimeError("policy_observation_valid is required for task collection")
        return torch.ones(num_envs, dtype=torch.bool, device=device)
    if not isinstance(validity, torch.Tensor):
        raise TypeError("policy_observation_valid must be a tensor")
    validity = validity.to(device=device, dtype=torch.bool)
    if validity.shape != (num_envs,):
        raise ValueError("policy_observation_valid must have shape [num_envs]")
    return validity


def _column(tensor: torch.Tensor, *, num_envs: int, name: str) -> torch.Tensor:
    if tensor.shape == (num_envs,):
        return tensor.unsqueeze(-1)
    if tensor.shape == (num_envs, 1):
        return tensor
    raise ValueError(f"{name} must have shape [num_envs] or [num_envs, 1]")


def absorbing_gae_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    *,
    last_values: torch.Tensor,
    discount_factor: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without bootstrapping through auto-reset observations.

    This mirrors the comparison PPO contract (`time_limit_bootstrap=False`) while
    leaving advantages unnormalised so the boundary arithmetic is testable.
    """

    if rewards.shape != values.shape:
        raise ValueError("rewards and values must have the same shape")
    done = absorbing_done(terminated, truncated)
    if done.shape != rewards.shape:
        raise ValueError("terminal flags must match rewards")
    if last_values.shape != rewards[-1].shape:
        raise ValueError("last_values must match one vector step")
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(last_values)
    for index in reversed(range(rewards.shape[0])):
        next_values = values[index + 1] if index + 1 < rewards.shape[0] else last_values
        not_done = (~done[index]).to(rewards.dtype)
        delta = rewards[index] + discount_factor * not_done * next_values - values[index]
        running = delta + discount_factor * gae_lambda * not_done * running
        advantages[index] = running
    return advantages + values, advantages


def _spaces() -> tuple[gymnasium.spaces.Box, gymnasium.spaces.Box]:
    observation_space = gymnasium.spaces.Box(
        low=-float("inf"), high=float("inf"), shape=(OBSERVATION_DIM,), dtype="float32"
    )
    action_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_DIM,), dtype="float32")
    return observation_space, action_space


def _hidden() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(OBSERVATION_DIM, 128),
        nn.ELU(),
        nn.Linear(128, 128),
        nn.ELU(),
    )


def _model_types():
    try:
        from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
    except ImportError as error:  # pragma: no cover - exercised by dependency smoke
        raise RuntimeError(
            "skrl 2.1.0 is required; install configs/rj45-comparison-requirements.txt "
            "with --no-deps inside the pinned Isaac runtime"
        ) from error

    class GaussianActor(GaussianMixin, Model):
        def __init__(self, observation_space, action_space, device, initial_log_std=-1.0):
            Model.__init__(
                self,
                observation_space=observation_space,
                action_space=action_space,
                device=device,
            )
            GaussianMixin.__init__(
                self,
                clip_actions=True,
                clip_mean_actions=False,
                clip_log_std=True,
                min_log_std=MIN_LOG_STD,
                max_log_std=MAX_LOG_STD,
                reduction="sum",
            )
            self.net = _hidden()
            self.mean = nn.Linear(128, ACTION_DIM)
            self.log_std = nn.Parameter(torch.full((ACTION_DIM,), initial_log_std))

        def compute(self, inputs, role=""):
            mean = torch.tanh(self.mean(self.net(inputs["observations"])))
            return mean, {"log_std": self.log_std.expand_as(mean)}

    class SquashedGaussianActor(Model):
        """SAC policy with tanh reparameterisation and Jacobian correction."""

        def __init__(self, observation_space, action_space, device, initial_log_std=-1.0):
            Model.__init__(
                self,
                observation_space=observation_space,
                action_space=action_space,
                device=device,
            )
            self.net = _hidden()
            self.mean = nn.Linear(128, ACTION_DIM)
            self.log_std = nn.Parameter(torch.full((ACTION_DIM,), initial_log_std))

        def compute(self, inputs, role=""):
            raw_mean = self.mean(self.net(inputs["observations"]))
            log_std = torch.clamp(self.log_std, min=MIN_LOG_STD, max=MAX_LOG_STD).expand_as(raw_mean)
            return raw_mean, log_std

        def act(self, inputs, *, role=""):
            raw_mean, log_std = self.compute(inputs, role)
            distribution = torch.distributions.Normal(raw_mean, log_std.exp())
            raw_action = distribution.rsample()
            actions = torch.tanh(raw_action)
            log_prob = distribution.log_prob(raw_action) - torch.log(
                1.0 - actions.square() + 1.0e-6
            )
            return actions, {
                "log_prob": log_prob.sum(dim=-1, keepdim=True),
                "mean_actions": torch.tanh(raw_mean),
                "log_std": log_std,
            }

    class ValueModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device):
            Model.__init__(
                self,
                observation_space=observation_space,
                action_space=action_space,
                device=device,
            )
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = nn.Sequential(_hidden(), nn.Linear(128, 1))

        def compute(self, inputs, role=""):
            return self.net(inputs["observations"]), {}

    class CriticModel(DeterministicMixin, Model):
        def __init__(self, observation_space, action_space, device):
            Model.__init__(
                self,
                observation_space=observation_space,
                action_space=action_space,
                device=device,
            )
            DeterministicMixin.__init__(self, clip_actions=False)
            self.net = nn.Sequential(
                nn.Linear(OBSERVATION_DIM + ACTION_DIM, 128),
                nn.ELU(),
                nn.Linear(128, 128),
                nn.ELU(),
                nn.Linear(128, 1),
            )

        def compute(self, inputs, role=""):
            features = torch.cat((inputs["observations"], inputs["taken_actions"]), dim=-1)
            return self.net(features), {}

    return GaussianActor, SquashedGaussianActor, ValueModel, CriticModel


def make_agent(
    algorithm: str,
    *,
    num_envs: int,
    device: str | torch.device = "cpu",
    rollout_steps: int = 16,
    replay_capacity: int = 4096,
    batch_size: int = 32,
    sac_utd_ratio: float = 1.0,
    seed: int = 0,
    mean_initializer: str = "legacy_random",
    initial_log_std: float = -1.0,
    sac_random_warmup_valid_transitions: int | None = None,
    sac_learning_starts_valid_transitions: int | None = None,
):
    """Build PPO or SAC with identical 79D actor/critic and 6D action rights."""

    if algorithm not in {"ppo", "sac"}:
        raise ValueError("algorithm must be 'ppo' or 'sac'")
    if num_envs < 1 or rollout_steps < 1:
        raise ValueError("num_envs and rollout_steps must be positive")
    if isinstance(sac_utd_ratio, bool) or not isinstance(sac_utd_ratio, (int, float)) or not math.isfinite(sac_utd_ratio) or sac_utd_ratio <= 0:
        raise ValueError("sac_utd_ratio must be finite and positive")
    if mean_initializer not in {"legacy_random", "zero"}:
        raise ValueError("mean_initializer must be 'legacy_random' or 'zero'")
    try:
        initial_log_std = float(initial_log_std)
    except (TypeError, ValueError) as error:
        raise ValueError("initial_log_std must be a finite number") from error
    if not math.isfinite(initial_log_std) or not MIN_LOG_STD <= initial_log_std <= MAX_LOG_STD:
        raise ValueError(
            f"initial_log_std must be finite and within [{MIN_LOG_STD}, {MAX_LOG_STD}]"
        )
    for name, value in (
        ("sac_random_warmup_valid_transitions", sac_random_warmup_valid_transitions),
        ("sac_learning_starts_valid_transitions", sac_learning_starts_valid_transitions),
    ):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer or None")
    if algorithm != "sac" and (
        sac_random_warmup_valid_transitions is not None
        or sac_learning_starts_valid_transitions is not None
    ):
        raise ValueError("SAC warmup/start controls require algorithm='sac'")
    if algorithm == "sac":
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(replay_capacity, bool) or not isinstance(replay_capacity, int) or replay_capacity < batch_size:
            raise ValueError("replay_capacity must be an integer at least batch_size")
        if (
            sac_learning_starts_valid_transitions is not None
            and sac_learning_starts_valid_transitions < batch_size
        ):
            raise ValueError("sac_learning_starts_valid_transitions must be at least batch_size")
    torch.manual_seed(seed)
    observation_space, action_space = _spaces()
    GaussianActor, SquashedGaussianActor, ValueModel, CriticModel = _model_types()

    from skrl.memories.torch import RandomMemory

    common = {
        "observation_space": observation_space,
        "action_space": action_space,
        "device": device,
    }
    experiment = {
        "directory": str(Path(tempfile.gettempdir()) / "rby1-comparison-runs"),
        "write_interval": 1_000_000_000,
        "checkpoint_interval": 0,
    }
    if algorithm == "ppo":
        from skrl.agents.torch.ppo import PPO, PPO_CFG

        policy = GaussianActor(**common, initial_log_std=initial_log_std)
        if mean_initializer == "zero":
            nn.init.zeros_(policy.mean.weight)
            nn.init.zeros_(policy.mean.bias)
        value = ValueModel(**common)
        rollout_target = rollout_steps * num_envs
        memory = RandomMemory(memory_size=rollout_target, num_envs=1, device=device)
        config = PPO_CFG(
            rollouts=rollout_target,
            learning_epochs=2,
            mini_batches=2,
            learning_rate=3.0e-4,
            time_limit_bootstrap=False,
            observation_preprocessor=None,
            state_preprocessor=None,
            value_preprocessor=None,
            experiment=experiment,
        )
        agent = PPO(
            models={"policy": policy, "value": value}, memory=memory, cfg=config, **common
        )
    else:
        from skrl.agents.torch.sac import SAC, SAC_CFG

        policy = SquashedGaussianActor(**common, initial_log_std=initial_log_std)
        if mean_initializer == "zero":
            nn.init.zeros_(policy.mean.weight)
            nn.init.zeros_(policy.mean.bias)
        critic_1 = CriticModel(**common)
        critic_2 = CriticModel(**common)
        target_critic_1 = CriticModel(**common)
        target_critic_2 = CriticModel(**common)
        memory = RandomMemory(
            memory_size=replay_capacity, num_envs=1, device=device, replacement=True
        )
        config = SAC_CFG(
            gradient_steps=1,
            batch_size=batch_size,
            learning_rate=3.0e-4,
            learning_starts=0,
            random_timesteps=0,
            observation_preprocessor=None,
            state_preprocessor=None,
            experiment=experiment,
        )
        agent = SAC(
            models={
                "policy": policy,
                "critic_1": critic_1,
                "critic_2": critic_2,
                "target_critic_1": target_critic_1,
                "target_critic_2": target_critic_2,
            },
            memory=memory,
            cfg=config,
            **common,
        )
    agent.init()
    agent._comparison_algorithm = algorithm
    agent._comparison_num_envs = num_envs
    agent._comparison_ppo_rollout_target = rollout_steps * num_envs
    agent._comparison_sac_utd_ratio = float(sac_utd_ratio)
    agent._comparison_mean_initializer = mean_initializer
    agent._comparison_initial_log_std = float(initial_log_std)
    agent._comparison_log_std_bounds = (MIN_LOG_STD, MAX_LOG_STD)
    agent._comparison_random_warmup_requested = sac_random_warmup_valid_transitions
    agent._comparison_learning_starts_requested = sac_learning_starts_valid_transitions
    agent._comparison_random_warmup_target = (
        batch_size if algorithm == "sac" and sac_random_warmup_valid_transitions is None
        else int(sac_random_warmup_valid_transitions or 0)
    )
    agent._comparison_learning_starts_target = (
        batch_size if algorithm == "sac" and sac_learning_starts_valid_transitions is None
        else int(sac_learning_starts_valid_transitions or 0)
    )
    agent._comparison_random_warmup_effective = agent._comparison_random_warmup_target
    agent._comparison_learning_starts_effective = agent._comparison_learning_starts_target
    agent._comparison_state = {
        "trajectories": [[] for _ in range(num_envs)],
        "algorithm_updates": 0,
        "optimizer_steps": 0,
        "random_warmup_transitions": 0,
        "replay_insertions": 0,
        "valid_policy_transitions": 0,
        "ppo_bootstrap_adjustments": 0,
        "sac_update_credit": 0.0,
        "replay_first_ready_at": 0,
        "first_update_after_valid_insertions": 0,
    }
    return agent


def _ppo_update_upstream(agent) -> tuple[int, int]:
    """Pack ragged trajectories and call the unmodified skrl PPO update.

    Each environment is one contiguous segment. A nonterminal segment end gets
    ``gamma * V(next_observation)`` added to its learning-buffer reward, then a
    synthetic terminal bit prevents GAE from crossing into the next segment.
    Raw environment rewards and endings remain unchanged in the trajectory
    records until this adapter builds the temporary learning memory.
    """

    from skrl.memories.torch import RandomMemory

    trajectories = agent._comparison_state["trajectories"]
    packed: dict[str, list[torch.Tensor]] = {
        name: []
        for name in ("observations", "actions", "rewards", "terminated", "log_prob", "values")
    }
    bootstrap_adjustments = 0
    final_next_observation = None
    for trajectory in trajectories:
        if not trajectory:
            continue
        learning_rewards = [item["reward"].clone() for item in trajectory]
        learning_done = [item["done"].clone() for item in trajectory]
        for index, item in enumerate(trajectory):
            segment_end = bool(item["segment_end"]) or index == len(trajectory) - 1
            if not segment_end:
                continue
            if not bool(item["done"].item()):
                with torch.no_grad():
                    next_value, _ = agent.value.act(
                        {"observations": item["next_observation"].unsqueeze(0)},
                        role="value",
                    )
                learning_rewards[index] += (
                    agent.cfg.discount_factor * next_value.squeeze(0)
                )
                bootstrap_adjustments += 1
            learning_done[index] = torch.ones_like(learning_done[index])
        for index, item in enumerate(trajectory):
            packed["observations"].append(item["observation"])
            packed["actions"].append(item["action"])
            packed["rewards"].append(learning_rewards[index])
            packed["terminated"].append(learning_done[index])
            packed["log_prob"].append(item["log_prob"])
            packed["values"].append(item["value"])
        final_next_observation = trajectory[-1]["next_observation"]

    sample_count = len(packed["observations"])
    memory = RandomMemory(memory_size=sample_count, num_envs=1, device=agent.device)
    memory.create_tensor(name="observations", size=agent.observation_space, dtype=torch.float32)
    memory.create_tensor(name="states", size=agent.state_space, dtype=torch.float32)
    memory.create_tensor(name="actions", size=agent.action_space, dtype=torch.float32)
    memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
    memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
    memory.create_tensor(name="truncated", size=1, dtype=torch.bool)
    memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
    memory.create_tensor(name="values", size=1, dtype=torch.float32)
    memory.create_tensor(name="returns", size=1, dtype=torch.float32)
    memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
    for index in range(sample_count):
        memory.add_samples(
            observations=packed["observations"][index],
            states=None,
            actions=packed["actions"][index],
            rewards=packed["rewards"][index],
            terminated=packed["terminated"][index],
            truncated=torch.zeros_like(packed["terminated"][index]),
            log_prob=packed["log_prob"][index],
            values=packed["values"][index],
        )
    agent.memory = memory
    agent._current_next_observations = final_next_observation.unsqueeze(0)
    agent._current_next_states = None
    agent.enable_models_training_mode(True)
    agent.update(
        timestep=agent._comparison_state["algorithm_updates"],
        timesteps=sample_count,
    )
    agent.enable_models_training_mode(False)
    for trajectory in trajectories:
        trajectory.clear()
    optimizer_steps = agent.cfg.learning_epochs * agent.cfg.mini_batches
    return optimizer_steps, bootstrap_adjustments


def _append_sac_valid_rows(
    agent,
    *,
    observations: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_observations: torch.Tensor,
    done: torch.Tensor,
    valid: torch.Tensor,
) -> int:
    indexes = valid.nonzero(as_tuple=True)[0]
    if not indexes.numel():
        return 0
    # skrl 2.1.0's num_envs=1 multi-row insertion advances the memory index
    # once per tensor field. Insert one row at a time to preserve exact replay
    # cardinality until that upstream behavior changes.
    for index in indexes.tolist():
        agent.memory.add_samples(
            observations=observations[index],
            states=None,
            actions=actions[index],
            rewards=rewards[index],
            next_observations=next_observations[index],
            next_states=None,
            terminated=done[index],
        )
    return int(indexes.numel())


def collect_bounded(
    agent,
    env: VectorEnvironment,
    *,
    algorithm: str,
    vector_steps: int,
    require_policy_validity: bool = False,
) -> CollectorResult:
    """Observe a bounded vector rollout and preserve every policy-valid row.

    ``transitions`` always equals ``num_envs * vector_steps``. PPO keeps a
    ragged on-policy trajectory per environment and computes GAE before a
    masked update. SAC inserts only valid rows into a row-wise replay memory.
    The collector remains a G0 tensor-path implementation until an Isaac/PhysX
    smoke proves the environment integration.
    """

    if vector_steps < 1:
        raise ValueError("vector_steps must be positive")
    if algorithm not in {"ppo", "sac"}:
        raise ValueError("algorithm must be 'ppo' or 'sac'")
    if getattr(agent, "_comparison_algorithm", None) != algorithm:
        raise ValueError("collector algorithm does not match the agent")
    if getattr(agent, "_comparison_num_envs", None) != env.num_envs:
        raise ValueError("collector environment count does not match the agent")
    state = agent._comparison_state
    if algorithm == "ppo":
        # Each collector call performs an explicit environment reset. Close any
        # pending on-policy segment before that reset so GAE cannot cross calls.
        # Raw environment `done` remains unchanged; the packer will bootstrap
        # this nonterminal segment from its already-recorded next observation.
        for trajectory in state["trajectories"]:
            if trajectory and not bool(trajectory[-1]["segment_end"]):
                trajectory[-1]["segment_end"] = True
    observations, current_infos = _reset_observation(env.reset())
    if observations.shape != (env.num_envs, OBSERVATION_DIM):
        raise ValueError("environment reset must return [num_envs, 79]")
    agent.enable_training_mode(True, apply_to_models=False)
    before_updates = state["algorithm_updates"]
    before_optimizer_steps = state["optimizer_steps"]
    before_random = state["random_warmup_transitions"]
    before_replay = state["replay_insertions"]
    before_bootstrap_adjustments = state["ppo_bootstrap_adjustments"]
    total = env.num_envs * vector_steps
    terminated_count = truncated_count = absorbing_count = 0
    mixed_validity_vector_steps = 0
    valid_policy_transitions = warmup_transitions = 0
    for _environment_step in range(vector_steps):
        valid = _policy_validity(
            current_infos,
            num_envs=env.num_envs,
            device=observations.device,
            required=require_policy_validity,
        )
        valid_count = int(valid.sum().item())
        valid_policy_transitions += valid_count
        warmup_transitions += env.num_envs - valid_count
        with torch.no_grad():
            random_valid_count = 0
            if algorithm == "ppo":
                actions, outputs = agent.policy.act(
                    {"observations": observations}, role="policy"
                )
                values, _ = agent.value.act(
                    {"observations": observations}, role="value"
                )
            else:
                remaining_random = max(
                    0,
                    agent._comparison_random_warmup_target
                    - state["random_warmup_transitions"],
                )
                random_valid_count = min(remaining_random, valid_count)
                if random_valid_count == valid_count and valid_count > 0:
                    actions, outputs = agent.policy.random_act(
                        {"observations": observations}, role="policy"
                    )
                elif random_valid_count == 0:
                    actions, outputs = agent.policy.act(
                        {"observations": observations}, role="policy"
                    )
                else:
                    random_actions, _ = agent.policy.random_act(
                        {"observations": observations}, role="policy"
                    )
                    policy_actions, outputs = agent.policy.act(
                        {"observations": observations}, role="policy"
                    )
                    random_mask = torch.zeros_like(valid)
                    valid_indexes = valid.nonzero(as_tuple=True)[0]
                    random_mask[valid_indexes[:random_valid_count]] = True
                    actions = torch.where(
                        random_mask[:, None], random_actions, policy_actions
                    )
                values = None
            actions = torch.where(valid[:, None], actions, torch.zeros_like(actions))
        next_observation_result, rewards, terminated, truncated, infos = env.step(actions)
        next_observations = _policy_tensor(next_observation_result)
        rewards = _column(rewards, num_envs=env.num_envs, name="rewards")
        terminated = _column(terminated, num_envs=env.num_envs, name="terminated")
        truncated = _column(truncated, num_envs=env.num_envs, name="truncated")
        done = absorbing_done(terminated, truncated)
        if bool(valid.any()) and not bool(valid.all()):
            mixed_validity_vector_steps += 1
        if algorithm == "ppo":
            trajectories = state["trajectories"]
            for env_index in range(env.num_envs):
                if not bool(valid[env_index].item()):
                    if trajectories[env_index] and not bool(
                        trajectories[env_index][-1]["segment_end"]
                    ):
                        trajectories[env_index][-1]["segment_end"] = True
                    continue
                trajectories[env_index].append(
                    {
                        "observation": observations[env_index].detach().clone(),
                        "action": actions[env_index].detach().clone(),
                        "reward": rewards[env_index].detach().clone(),
                        "done": done[env_index].detach().clone(),
                        "segment_end": bool(done[env_index].item()),
                        "log_prob": outputs["log_prob"][env_index].detach().clone(),
                        "value": values[env_index].detach().clone(),
                        "next_observation": next_observations[env_index].detach().clone(),
                    }
                )
            pending = sum(len(trajectory) for trajectory in trajectories)
            update_threshold = agent._comparison_ppo_rollout_target
            if pending >= update_threshold:
                optimizer_steps, bootstrap_adjustments = _ppo_update_upstream(agent)
                state["algorithm_updates"] += 1
                state["optimizer_steps"] += optimizer_steps
                state["ppo_bootstrap_adjustments"] += bootstrap_adjustments
        else:
            inserted = _append_sac_valid_rows(
                agent,
                observations=observations,
                actions=actions,
                rewards=rewards,
                next_observations=next_observations,
                done=done,
                valid=valid,
            )
            state["replay_insertions"] += inserted
            state["random_warmup_transitions"] += random_valid_count
            if (
                not state["replay_first_ready_at"]
                and state["replay_insertions"] >= agent.cfg.batch_size
            ):
                state["replay_first_ready_at"] = state["replay_insertions"]
            state["sac_update_credit"] += inserted * agent._comparison_sac_utd_ratio
            if state["replay_insertions"] >= agent._comparison_learning_starts_target:
                update_calls = int(state["sac_update_credit"])
            else:
                update_calls = 0
            state["sac_update_credit"] -= update_calls
            if update_calls:
                if not state["first_update_after_valid_insertions"]:
                    state["first_update_after_valid_insertions"] = state[
                        "replay_insertions"
                    ]
                agent.enable_models_training_mode(True)
                for _ in range(update_calls):
                    agent.update(timestep=state["algorithm_updates"], timesteps=vector_steps)
                agent.enable_models_training_mode(False)
                state["algorithm_updates"] += update_calls
                optimizers_per_update = 3 if agent.cfg.learn_entropy else 2
                state["optimizer_steps"] += (
                    update_calls * agent.cfg.gradient_steps * optimizers_per_update
                )
        state["valid_policy_transitions"] += valid_count
        observations = next_observations
        current_infos = infos
        terminated_count += int(terminated.sum().item())
        truncated_count += int(truncated.sum().item())
        absorbing_count += int(done.sum().item())
    return CollectorResult(
        algorithm=algorithm,
        vector_steps=vector_steps,
        num_envs=env.num_envs,
        transitions=total,
        valid_policy_transitions=valid_policy_transitions,
        learning_transitions=valid_policy_transitions,
        warmup_transitions=warmup_transitions,
        skipped_valid_transitions=0,
        mixed_validity_vector_steps=mixed_validity_vector_steps,
        algorithm_updates=state["algorithm_updates"] - before_updates,
        optimizer_steps=state["optimizer_steps"] - before_optimizer_steps,
        random_warmup_transitions=(
            state["random_warmup_transitions"] - before_random
        ),
        replay_insertions=state["replay_insertions"] - before_replay,
        replay_size=len(agent.memory) if algorithm == "sac" else 0,
        replay_first_ready_at=(
            state["replay_first_ready_at"] if algorithm == "sac" else 0
        ),
        pending_policy_transitions=(
            sum(len(trajectory) for trajectory in state["trajectories"])
            if algorithm == "ppo"
            else 0
        ),
        ppo_bootstrap_adjustments=(
            state["ppo_bootstrap_adjustments"] - before_bootstrap_adjustments
            if algorithm == "ppo"
            else 0
        ),
        ppo_rollout_target=(
            agent._comparison_ppo_rollout_target if algorithm == "ppo" else 0
        ),
        sac_utd_ratio=(agent._comparison_sac_utd_ratio if algorithm == "sac" else 0.0),
        terminated_transitions=terminated_count,
        truncated_transitions=truncated_count,
        absorbing_transitions=absorbing_count,
        sac_random_warmup_requested=(
            agent._comparison_random_warmup_requested if algorithm == "sac" else None
        ),
        sac_random_warmup_effective=(
            agent._comparison_random_warmup_target if algorithm == "sac" else 0
        ),
        sac_learning_starts_requested=(
            agent._comparison_learning_starts_requested if algorithm == "sac" else None
        ),
        sac_learning_starts_effective=(
            agent._comparison_learning_starts_target if algorithm == "sac" else 0
        ),
        sac_first_update_after_valid_insertions=(
            state["first_update_after_valid_insertions"] if algorithm == "sac" else 0
        ),
    )


def deterministic_action(agent, observations: torch.Tensor) -> torch.Tensor:
    """Return the policy mean for reproducible inference and checkpoint checks."""

    agent.policy.eval()
    with torch.no_grad():
        _, outputs = agent.policy.act({"observations": observations}, role="policy")
    return outputs["mean_actions"]


def save_inference_checkpoint(agent, path: str | Path, *, algorithm: str) -> None:
    """Save policy-only inference state; replay/training resume is unsupported."""

    if getattr(agent, "_comparison_algorithm", None) != algorithm:
        raise ValueError("checkpoint algorithm does not match the agent")
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "algorithm": algorithm,
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "running_scaler": None,
            "resume_supported": False,
            "policy": agent.policy.state_dict(),
        },
        Path(path),
    )


def load_inference_checkpoint(
    agent, path: str | Path, *, resume: bool = False
) -> dict[str, Any]:
    """Load a policy for inference and fail closed on training-resume requests."""

    if resume:
        raise RuntimeError(
            "training resume is disabled: the replay/rollout buffer and optimizer "
            "state are not persisted"
        )
    payload = torch.load(Path(path), map_location=agent.device, weights_only=True)
    expected = (payload.get("format"), payload.get("observation_dim"), payload.get("action_dim"))
    if expected != (CHECKPOINT_FORMAT, OBSERVATION_DIM, ACTION_DIM):
        raise ValueError("incompatible comparison checkpoint")
    if payload.get("algorithm") != getattr(agent, "_comparison_algorithm", None):
        raise ValueError("checkpoint algorithm does not match the agent")
    agent.policy.load_state_dict(payload["policy"])
    agent.policy.eval()
    return payload


def finite_losses(agent) -> dict[str, float]:
    """Return tracked algorithm losses and reject missing/non-finite updates."""

    values: dict[str, float] = {}
    for key, samples in agent.tracking_data.items():
        if key.startswith("Loss / ") and samples:
            values[key] = float(samples[-1])
    if not values or not all(torch.isfinite(torch.tensor(value)) for value in values.values()):
        raise RuntimeError("algorithm did not report finite losses")
    return values


def parameter_vector(agent) -> torch.Tensor:
    """Flatten trainable policy parameters for update evidence."""

    return torch.cat(
        [parameter.detach().reshape(-1).cpu() for parameter in agent.policy.parameters()]
    )
