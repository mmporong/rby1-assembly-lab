"""Passive one-environment monitor for physical RB-Y1 hand-to-hand contact.

The wrapper changes collision filtering only: it enables the nine audited
left-hand/right-hand rigid-body pairs and measures their PhysX contacts. It
does not alter commands, poses, forces, controller phases, or termination
reasons. The base environment's forbidden-contact evaluator consumes the
combined measured load in the usual way.
"""

from __future__ import annotations

from typing import Any

from rby1_assembly.hand_collision import (
    COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
    COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
    CROSS_HAND_CONTACT_LOAD_SOURCE,
    LEFT_HAND_BODY_PATHS,
    RIGHT_HAND_BODY_PATHS,
    configure_selective_hand_self_collisions,
)


def make_hand_contact_monitor_env_type(base_cls: type) -> type:
    """Wrap a nominal environment with passive one-env hand contact sensing.

    Multi-environment replication is intentionally rejected until selective
    filtered-pair authoring and contact-buffer isolation are validated for
    every cloned articulation.
    """

    if not isinstance(base_cls, type):
        raise TypeError("base_cls must be an environment type")

    import torch
    from isaacsim.core.utils.stage import get_current_stage
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from rby1_assembly.contact_loads import sum_contact_normal_loads

    class HandContactMonitorEnv(base_cls):
        """Nominal controller plus measured cross-hand forbidden-contact load."""

        def __init__(self, cfg, render_mode=None, **kwargs):
            if cfg.scene.num_envs != 1:
                raise ValueError("hand contact monitoring currently requires exactly one environment")
            super().__init__(cfg, render_mode, **kwargs)
            self._last_cross_hand_contact_load_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._episode_cross_hand_contact_load_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._last_external_forbidden_contact_load_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._episode_external_forbidden_contact_load_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )

        def _setup_scene(self):
            super()._setup_scene()
            if len(self.scene.env_prim_paths) != 1:
                raise RuntimeError("hand contact monitoring requires one composed environment")
            robot_root_path = f"{self.scene.env_prim_paths[0]}/Robot"
            self._hand_contact_filter_metadata = configure_selective_hand_self_collisions(
                get_current_stage(), robot_root_path
            )
            self.cfg.robot.spawn.articulation_props.enabled_self_collisions = True

            right_filters = [
                f"{robot_root_path}/{relative_path}"
                for relative_path in RIGHT_HAND_BODY_PATHS
            ]
            sensors = {}
            for relative_path in LEFT_HAND_BODY_PATHS:
                sensor_cfg = ContactSensorCfg(
                    prim_path=f"{robot_root_path}/{relative_path}",
                    max_contact_data_count_per_prim=128,
                    update_period=0.0,
                    history_length=0,
                    track_air_time=False,
                    filter_prim_paths_expr=right_filters,
                )
                sensor = ContactSensor(sensor_cfg)
                sensors[relative_path] = sensor
                self.scene.sensors[
                    f"nominal_cross_hand_{relative_path.replace('/', '_')}"
                ] = sensor
            self._hand_contact_sensors = sensors

        def _cross_hand_contact_load(self):
            loads = []
            for label, sensor in self._hand_contact_sensors.items():
                forces, _, _, _, counts, starts = (
                    sensor.contact_physx_view.get_contact_data(dt=self.cfg.sim.dt)
                )
                try:
                    load = sum_contact_normal_loads(
                        forces, counts, starts, self.num_envs
                    )
                except (TypeError, ValueError) as error:
                    raise RuntimeError(
                        f"invalid nominal cross-hand contact buffer for {label}: {error}"
                    ) from error
                loads.append(load)
            if len(loads) != len(LEFT_HAND_BODY_PATHS):
                raise RuntimeError("nominal cross-hand contact sensors are unavailable")
            total = torch.stack(loads, dim=1).sum(dim=1)
            if not torch.isfinite(total).all():
                raise RuntimeError("nominal cross-hand contact load is non-finite")
            return total

        def _forbidden_contact_load(self):
            external = super()._forbidden_contact_load()
            cross_hand = self._cross_hand_contact_load()
            self._last_external_forbidden_contact_load_n[:] = external
            self._episode_external_forbidden_contact_load_peak_n = torch.maximum(
                self._episode_external_forbidden_contact_load_peak_n, external
            )
            self._last_cross_hand_contact_load_n[:] = cross_hand
            self._episode_cross_hand_contact_load_peak_n = torch.maximum(
                self._episode_cross_hand_contact_load_peak_n, cross_hand
            )
            return torch.maximum(external, cross_hand)

        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            contact_fields = {
                "cross_hand_contact_load_n": self._last_cross_hand_contact_load_n.clone(),
                "cross_hand_contact_load_peak_n": (
                    self._episode_cross_hand_contact_load_peak_n.clone()
                ),
                "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
                "external_forbidden_contact_load_n": (
                    self._last_external_forbidden_contact_load_n.clone()
                ),
                "external_forbidden_contact_load_peak_n": (
                    self._episode_external_forbidden_contact_load_peak_n.clone()
                ),
                "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
                "forbidden_contact_load_definition": (
                    COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION
                ),
                "self_collision_enabled": True,
                "hand_contact_monitor_mode": "passive_nominal_one_environment",
                "selective_self_collision_filter": dict(
                    self._hand_contact_filter_metadata
                ),
            }
            self.extras.update(contact_fields)
            evaluation = self.extras.get("evaluation")
            if isinstance(evaluation, dict):
                evaluation.update(contact_fields)
            snapshot = self.extras.get("terminal_snapshot")
            if isinstance(snapshot, dict):
                snapshot.update(contact_fields)
                snapshot_evaluation = snapshot.get("evaluation")
                if isinstance(snapshot_evaluation, dict):
                    snapshot_evaluation.update(contact_fields)
            return terminated, truncated

        def _reset_idx(self, env_ids=None):
            super()._reset_idx(env_ids)
            if not hasattr(self, "_last_cross_hand_contact_load_n"):
                return
            if env_ids is None:
                env_ids = self.robot._ALL_INDICES
            self._last_cross_hand_contact_load_n[env_ids] = 0.0
            self._episode_cross_hand_contact_load_peak_n[env_ids] = 0.0
            self._last_external_forbidden_contact_load_n[env_ids] = 0.0
            self._episode_external_forbidden_contact_load_peak_n[env_ids] = 0.0

        def hand_contact_monitor_report(self) -> dict[str, Any]:
            return {
                "scope": (
                    "passive nominal one-environment hand-to-hand PhysX contact monitor"
                ),
                "command_pose_force_or_failure_injection": False,
                "multi_environment_validation_completed": False,
                "required_num_envs": 1,
                "physics_runtime_policy": (
                    "articulation self-collision enabled; only the 9 left-hand/right-hand "
                    "rigid-body pairs are unfiltered"
                ),
                "collision_filter": dict(self._hand_contact_filter_metadata),
                "cross_hand_contact_load_n": float(
                    self._last_cross_hand_contact_load_n[0].item()
                ),
                "cross_hand_contact_load_peak_n": float(
                    self._episode_cross_hand_contact_load_peak_n[0].item()
                ),
                "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
                "external_forbidden_contact_load_n": float(
                    self._last_external_forbidden_contact_load_n[0].item()
                ),
                "external_forbidden_contact_load_peak_n": float(
                    self._episode_external_forbidden_contact_load_peak_n[0].item()
                ),
                "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
                "forbidden_contact_load_definition": (
                    COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION
                ),
            }

    HandContactMonitorEnv.__name__ = f"HandContactMonitor{base_cls.__name__}"
    HandContactMonitorEnv.__qualname__ = HandContactMonitorEnv.__name__
    return HandContactMonitorEnv


__all__ = ["make_hand_contact_monitor_env_type"]
