"""Fail-closed observation adapter for the held-grasp RJ45 comparison.

This wrapper is intentionally opt-in.  It leaves the legacy training
environment unchanged while preventing an active comparison episode from
silently replacing an invalid simulation proxy with simulator ground truth.
"""

from __future__ import annotations


COMPARISON_OBSERVATION_METADATA = {
    "schema_version": 1,
    "scope": "held_grasp_rj45_right_arm_residual_comparison",
    "actor_observation_dimension": 79,
    "critic_observation_dimension": 79,
    "observation_scaling": "fixed_unit_scale_from_legacy_79d_contract",
    "residual_action_dimension": 6,
    "residual_action_scope": "right_arm_only",
    "left_arm_control": "scripted_legacy_support_controller",
    "proxy_kind": "frozen_hand_object_transform_simulation_proxy",
    "proxy_initialization": (
        "simulator_ground_truth_pose_with_configured_bias_at_calibration_event"
    ),
    "actual_camera_measurement": False,
    "ground_truth_claim": (
        "policy-active invalid proxy rows never receive simulator-ground-truth "
        "control fallback; simulator ground truth still initializes the biased "
        "simulation proxy and remains available to the independent evaluator"
    ),
    "policy_active_invalid_handling": "absorbing_observation_failure",
    "prepolicy_invalid_handling": "legacy_fallback_with_separate_diagnostic",
    "finite_horizon_timeout_handling": "absorbing_failure_no_bootstrap",
    "autoreset_final_observation_available": False,
    "bootstrap_reset_observation": False,
    "aggregate_denominator_scope": (
        "activated policy episodes reaching a terminal transition only; this is "
        "not the external requested-case evaluation denominator"
    ),
    "external_requested_case_denominator_required": True,
    "prepolicy_preparation_failure_reporting": (
        "the external evaluator must retain every requested case and classify "
        "prepolicy preparation failures separately as failed or invalid"
    ),
    "action_observation_contract": (
        "a submitted residual is applied only when the last returned policy "
        "observation for that environment was valid"
    ),
}


def comparison_observation_metadata() -> dict[str, object]:
    """Return a copy safe for run manifests and terminal reports."""

    return dict(COMPARISON_OBSERVATION_METADATA)


def make_rj45_comparison_env_type(base_training_cls: type) -> type:
    """Add fail-closed proxy handling to an RJ45 held-grasp training type.

    The base type must already implement the legacy 79D observation and 6D
    right-arm residual contracts.  During reset and held-grasp warmup, policy
    input is inactive and legacy ground-truth fallback remains available so
    staging can complete.  Once policy input is active, an invalid estimator is
    a counted terminal failure.  Its stale proxy tensors are returned only to
    keep the batched tensor path finite until the environment is reset.
    """

    import torch

    class Rby1Rj45ComparisonEnv(base_training_cls):
        def _initialize_training_state(self) -> None:
            super()._initialize_training_state()
            config = self._training_config
            if config.get("base_observation_dim") != 79:
                raise ValueError("comparison requires the legacy fixed 79D observation")
            if config.get("diagnostic_observation_mode") != "standard":
                raise ValueError("comparison requires an unprivileged 79D observation")
            if config.get("action_mode") != "right_only_6d":
                raise ValueError("comparison requires a 6D right-arm residual action")

            shape = (self.num_envs,)
            self._comparison_observation_failure_pending = torch.zeros(
                shape, dtype=torch.bool, device=self.device
            )
            self._comparison_observation_failure_episode = torch.zeros_like(
                self._comparison_observation_failure_pending
            )
            self._comparison_prepolicy_invalid_episode_seen = torch.zeros_like(
                self._comparison_observation_failure_pending
            )
            self._comparison_terminal_counted = torch.zeros_like(
                self._comparison_observation_failure_pending
            )
            self._comparison_observation_failure_reward_emitted = torch.zeros_like(
                self._comparison_observation_failure_pending
            )
            self._comparison_activated_episode_terminal_count = torch.zeros(
                shape, dtype=torch.long, device=self.device
            )
            self._comparison_observation_failure_count = torch.zeros_like(
                self._comparison_activated_episode_terminal_count
            )
            self._comparison_timeout_failure_count = torch.zeros_like(
                self._comparison_activated_episode_terminal_count
            )
            self._comparison_prepolicy_invalid_episode_count = torch.zeros_like(
                self._comparison_activated_episode_terminal_count
            )
            self._comparison_action_observation_gate = torch.zeros_like(
                self._comparison_observation_failure_pending
            )
            self._comparison_action_zero_gated = torch.ones_like(
                self._comparison_observation_failure_pending
            )
            self._comparison_submitted_action_6d = torch.zeros(
                (self.num_envs, 6), device=self.device
            )
            self._comparison_applied_action_6d = torch.zeros_like(
                self._comparison_submitted_action_6d
            )

        def _comparison_active_invalid_mask(self):
            return self._training_policy_activated & ~self._training_estimator_valid

        def _comparison_note_prepolicy_fallback(self, mask) -> None:
            first = mask & ~self._comparison_prepolicy_invalid_episode_seen
            self._comparison_prepolicy_invalid_episode_count += first.to(torch.long)
            self._comparison_prepolicy_invalid_episode_seen |= mask

        def _comparison_control_fallback_mask(self):
            active_invalid = self._comparison_active_invalid_mask()
            self._comparison_observation_failure_pending |= active_invalid
            return ~self._training_estimator_valid & ~self._training_policy_activated

        def _policy_active_mask(self):
            active = super()._policy_active_mask()
            if not self._training_initialized:
                return active
            active_invalid = self._comparison_active_invalid_mask()
            self._comparison_observation_failure_pending |= active_invalid
            # This mask is consumed before the parent applies residual actions
            # and while it marks policy observations valid.  Invalid active
            # rows therefore receive neither an action nor a learnable input in
            # the physics step preceding their absorbing termination.
            return active & self._training_estimator_valid

        def _pre_physics_step(self, actions) -> None:
            if not self._training_initialized:
                super()._pre_physics_step(actions)
                return
            if actions.shape != (self.num_envs, 6):
                raise ValueError(
                    "comparison residual actions must have shape (num_envs, 6)"
                )
            observation_gate = self._training_last_policy_observation_valid.clone()
            gated_actions = torch.where(
                observation_gate[:, None], actions, torch.zeros_like(actions)
            )
            self._comparison_action_observation_gate[:] = observation_gate
            self._comparison_action_zero_gated[:] = ~observation_gate
            self._comparison_submitted_action_6d[:] = actions
            super()._pre_physics_step(gated_actions)
            self._comparison_applied_action_6d[:] = self._actions[:, 6:12]
            self.extras["rj45_comparison_action_gate"] = {
                "last_policy_observation_valid": observation_gate.clone(),
                "zero_gated": self._comparison_action_zero_gated.clone(),
                "submitted_action_6d": self._comparison_submitted_action_6d.clone(),
                "applied_action_6d": self._comparison_applied_action_6d.clone(),
            }

        def _control_relative_object_pose(self):
            if not self._training_initialized:
                return super()._control_relative_object_pose()
            fallback = self._comparison_control_fallback_mask()
            if not bool(torch.any(fallback)):
                return (
                    self._training_estimated_relative_pos,
                    self._training_estimated_relative_quat,
                )
            self._comparison_note_prepolicy_fallback(fallback)
            actual_pos, actual_quat = super()._control_relative_object_pose()
            use_actual = fallback[:, None]
            return (
                torch.where(use_actual, actual_pos, self._training_estimated_relative_pos),
                torch.where(use_actual, actual_quat, self._training_estimated_relative_quat),
            )

        def _control_object_poses(self):
            if not self._training_initialized:
                return super()._control_object_poses()
            estimated = (
                self._training_estimated_object_pos[:, 0],
                self._training_estimated_object_quat[:, 0],
                self._training_estimated_object_pos[:, 1],
                self._training_estimated_object_quat[:, 1],
            )
            fallback = self._comparison_control_fallback_mask()
            if not bool(torch.any(fallback)):
                return estimated
            self._comparison_note_prepolicy_fallback(fallback)
            actual = super()._control_object_poses()
            use_actual = fallback[:, None]
            return tuple(
                torch.where(use_actual, truth, estimate)
                for estimate, truth in zip(estimated, actual, strict=True)
            )

        def _control_grasp_transforms(self, *args):
            if not self._training_initialized:
                return super()._control_grasp_transforms(*args)
            estimated = (
                self._training_frozen_object_to_ee_pos[:, 0],
                self._training_frozen_object_to_ee_quat[:, 0],
                self._training_frozen_object_to_ee_pos[:, 1],
                self._training_frozen_object_to_ee_quat[:, 1],
            )
            fallback = self._comparison_control_fallback_mask()
            if not bool(torch.any(fallback)):
                return estimated
            self._comparison_note_prepolicy_fallback(fallback)
            actual = super()._control_grasp_transforms(*args)
            use_actual = fallback[:, None]
            return tuple(
                torch.where(use_actual, truth, estimate)
                for estimate, truth in zip(estimated, actual, strict=True)
            )

        def _get_dones(self):
            if not self._training_initialized:
                return super()._get_dones()

            self._comparison_observation_failure_pending |= (
                self._comparison_active_invalid_mask()
            )
            terminated, truncated = super()._get_dones()
            observation_failure = (
                self._comparison_observation_failure_pending
                & self._training_policy_activated
            )
            active_timeout = truncated & self._training_policy_activated

            # The 24 s task is finite horizon.  Both an observation failure and
            # a timeout are absorbing task failures, so an autoreset observation
            # must never be used to bootstrap either transition.
            terminated = terminated.clone()
            truncated = truncated.clone()
            terminated |= observation_failure | active_timeout
            truncated &= ~active_timeout
            self._comparison_observation_failure_episode |= observation_failure

            ended = (terminated | truncated) & self._training_policy_activated
            newly_counted = ended & ~self._comparison_terminal_counted
            self._comparison_activated_episode_terminal_count += newly_counted.to(
                torch.long
            )
            self._comparison_observation_failure_count += (
                newly_counted & self._comparison_observation_failure_episode
            ).to(torch.long)
            self._comparison_timeout_failure_count += (
                newly_counted & active_timeout
            ).to(torch.long)
            self._comparison_terminal_counted |= ended

            contract = comparison_observation_metadata()
            aggregate = {
                "activated_episode_terminal_count": (
                    self._comparison_activated_episode_terminal_count.clone()
                ),
                "observation_failure_count": (
                    self._comparison_observation_failure_count.clone()
                ),
                "timeout_failure_count": self._comparison_timeout_failure_count.clone(),
                "prepolicy_invalid_episode_count": (
                    self._comparison_prepolicy_invalid_episode_count.clone()
                ),
            }
            self.extras["rj45_comparison_contract"] = contract
            self.extras["rj45_comparison_aggregate"] = aggregate

            snapshot = self.extras.get("terminal_snapshot")
            if isinstance(snapshot, dict):
                snapshot["terminal_mask"] = terminated | truncated
                snapshot["comparison_observation_failure"] = (
                    self._comparison_observation_failure_episode.clone()
                )
                snapshot["comparison_timeout_absorbing_failure"] = active_timeout.clone()
                snapshot["comparison_absorbing_failure"] = (
                    self._comparison_observation_failure_episode | active_timeout
                ).clone()
                snapshot["comparison_contract"] = contract
                snapshot["comparison_aggregate"] = aggregate
                snapshot["comparison_final_observation_available"] = False
                snapshot["comparison_bootstrap_reset_observation"] = False
            return terminated, truncated

        def _get_rewards(self):
            rewards = super()._get_rewards()
            emit = (
                self._comparison_observation_failure_episode
                & self._training_policy_activated
                & ~self._comparison_observation_failure_reward_emitted
            )
            if bool(torch.any(emit)):
                failure_penalty = float(
                    self._training_config["reward"]["failure_penalty"]
                )
                rewards = rewards.clone()
                rewards[emit] -= failure_penalty
                self._comparison_observation_failure_reward_emitted |= emit
                if hasattr(self, "_training_terminal_reward_emitted"):
                    self._training_terminal_reward_emitted |= emit
                components = self.extras.get("rj45_reward_components")
                if isinstance(components, dict) and "terminal" in components:
                    terminal = components["terminal"].clone()
                    terminal[emit] -= failure_penalty
                    components["terminal"] = terminal
                    if "total" in components:
                        total = components["total"].clone()
                        total[emit] -= failure_penalty
                        components["total"] = total
                snapshot = self.extras.get("terminal_snapshot")
                if isinstance(snapshot, dict):
                    snapshot["comparison_observation_failure_penalty"] = (
                        emit.to(rewards.dtype) * -failure_penalty
                    )
            return rewards

        def _training_step_snapshot(self):
            snapshot = super()._training_step_snapshot()
            snapshot.update(
                {
                    "comparison_observation_failure": (
                        self._comparison_observation_failure_episode.clone()
                    ),
                    "comparison_prepolicy_invalid_episode_seen": (
                        self._comparison_prepolicy_invalid_episode_seen.clone()
                    ),
                    "comparison_action_observation_gate": (
                        self._comparison_action_observation_gate.clone()
                    ),
                    "comparison_action_zero_gated": (
                        self._comparison_action_zero_gated.clone()
                    ),
                    "comparison_applied_action_6d": (
                        self._comparison_applied_action_6d.clone()
                    ),
                }
            )
            return snapshot

        def _configure_training_reset(self, env_ids):
            super()._configure_training_reset(env_ids)
            self._comparison_observation_failure_pending[env_ids] = False
            self._comparison_observation_failure_episode[env_ids] = False
            self._comparison_prepolicy_invalid_episode_seen[env_ids] = False
            self._comparison_terminal_counted[env_ids] = False
            self._comparison_observation_failure_reward_emitted[env_ids] = False
            self._comparison_action_observation_gate[env_ids] = False
            self._comparison_action_zero_gated[env_ids] = True
            self._comparison_submitted_action_6d[env_ids] = 0.0
            self._comparison_applied_action_6d[env_ids] = 0.0

    Rby1Rj45ComparisonEnv.__name__ = "Rby1Rj45ComparisonEnv"
    return Rby1Rj45ComparisonEnv


__all__ = [
    "COMPARISON_OBSERVATION_METADATA",
    "comparison_observation_metadata",
    "make_rj45_comparison_env_type",
]
