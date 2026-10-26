"""Passive monitor for physical RB-Y1 hand-to-hand contact.

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
from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence


def _flatten_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        paths: list[str] = []
        for item in value:
            paths.extend(_flatten_paths(item))
        return paths
    raise TypeError(f"unexpected PhysX path container: {type(value).__name__}")


def make_hand_contact_monitor_env_type(base_cls: type) -> type:
    """Wrap a nominal environment with passive hand-to-hand contact sensing."""

    if not isinstance(base_cls, type):
        raise TypeError("base_cls must be an environment type")

    import torch
    from isaacsim.core.utils.stage import get_current_stage
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from rby1_assembly.contact_loads import sum_contact_normal_loads

    class HandContactMonitorEnv(base_cls):
        """Nominal controller plus measured cross-hand forbidden-contact load."""

        def __init__(self, cfg, render_mode=None, **kwargs):
            requested_num_envs = int(cfg.scene.num_envs)
            if requested_num_envs < 1:
                raise ValueError("hand contact monitoring requires at least one environment")
            self._parallel_contact_evidence = None
            if requested_num_envs > 1:
                if cfg.scene.replicate_physics:
                    raise ValueError(
                        "batched hand contact monitoring requires replicate_physics=False"
                    )
                specification = getattr(cfg, "rj45_parallel_contact_evidence", None)
                self._parallel_contact_evidence = load_parallel_contact_evidence(
                    specification
                )
            super().__init__(cfg, render_mode, **kwargs)
            self._validate_hand_contact_sensor_routing()
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
            if len(self.scene.env_prim_paths) != int(self.cfg.scene.num_envs):
                raise RuntimeError("hand contact environment roots do not match cfg.scene.num_envs")
            stage = get_current_stage()
            filter_metadata = []
            for env_path in self.scene.env_prim_paths:
                filter_metadata.append(
                    configure_selective_hand_self_collisions(stage, f"{env_path}/Robot")
                )
            self._hand_contact_filter_metadata = filter_metadata
            self.cfg.robot.spawn.articulation_props.enabled_self_collisions = True

            if self.cfg.scene.num_envs == 1:
                source_root = f"{self.scene.env_prim_paths[0]}/Robot"
            else:
                source_root = f"{self.scene.env_regex_ns}/Robot"
            right_filters = [f"{source_root}/{path}" for path in RIGHT_HAND_BODY_PATHS]
            sensors = {}
            for relative_path in LEFT_HAND_BODY_PATHS:
                sensor_cfg = ContactSensorCfg(
                    prim_path=f"{source_root}/{relative_path}",
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

        def _validate_hand_contact_sensor_routing(self) -> None:
            for relative_path, sensor in self._hand_contact_sensors.items():
                expected_sources = [
                    f"{env_path}/Robot/{relative_path}"
                    for env_path in self.scene.env_prim_paths
                ]
                backend = getattr(sensor.body_physx_view, "_backend", None)
                if backend is None or not hasattr(backend, "prim_paths"):
                    raise RuntimeError(
                        f"PhysX backend source paths unavailable for {relative_path}"
                    )
                actual_source_groups = {
                    "body": _flatten_paths(sensor.body_physx_view.prim_paths),
                    "backend": _flatten_paths(backend.prim_paths),
                    "contact": _flatten_paths(sensor.contact_physx_view.sensor_paths),
                }
                for label, actual in actual_source_groups.items():
                    if actual != expected_sources:
                        raise RuntimeError(
                            f"hand contact {label} source ordering mismatch for "
                            f"{relative_path}: expected={expected_sources}, actual={actual}"
                        )
                expected_filters = [
                    f"{env_path}/Robot/{filter_path}"
                    for env_path in self.scene.env_prim_paths
                    for filter_path in RIGHT_HAND_BODY_PATHS
                ]
                actual_filters = _flatten_paths(sensor.contact_physx_view.filter_paths)
                if (
                    int(sensor.body_physx_view.count) != self.num_envs
                    or int(sensor.contact_physx_view.sensor_count) != self.num_envs
                    or int(sensor.contact_physx_view.filter_count)
                    != len(RIGHT_HAND_BODY_PATHS)
                    or actual_filters != expected_filters
                ):
                    raise RuntimeError(
                        "hand contact filter routing mismatch for "
                        f"{relative_path}: expected={expected_filters}, actual={actual_filters}"
                    )

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
                "hand_contact_monitor_mode": self._hand_contact_monitor_mode(),
                "selective_self_collision_filter": self._collision_filter_report(),
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

        def _hand_contact_monitor_mode(self) -> str:
            if self.num_envs == 1:
                return "passive_nominal_one_environment"
            return "passive_batched_runtime_ordering_asserted"

        def _collision_filter_report(self) -> Any:
            if self.num_envs == 1:
                return dict(self._hand_contact_filter_metadata[0])
            return [dict(item) for item in self._hand_contact_filter_metadata]

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
            report = {
                "scope": (
                    "passive hand-to-hand PhysX contact monitor with runtime routing assertions"
                ),
                "command_pose_force_or_failure_injection": False,
                "hand_contact_monitor_mode": self._hand_contact_monitor_mode(),
                "num_envs": self.num_envs,
                "multi_environment_validation_completed": False,
                "multi_environment_routing_validation_completed": self.num_envs > 1,
                "required_num_envs": 1 if self.num_envs == 1 else None,
                "physics_runtime_policy": (
                    "articulation self-collision enabled; only the 9 left-hand/right-hand "
                    "rigid-body pairs are unfiltered"
                ),
                "collision_filter": self._collision_filter_report(),
                "cross_hand_contact_load_n": self._last_cross_hand_contact_load_n.detach()
                .cpu()
                .tolist(),
                "cross_hand_contact_load_peak_n": (
                    self._episode_cross_hand_contact_load_peak_n.detach().cpu().tolist()
                ),
                "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
                "external_forbidden_contact_load_n": (
                    self._last_external_forbidden_contact_load_n.detach().cpu().tolist()
                ),
                "external_forbidden_contact_load_peak_n": (
                    self._episode_external_forbidden_contact_load_peak_n.detach()
                    .cpu()
                    .tolist()
                ),
                "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
                "forbidden_contact_load_definition": (
                    COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION
                ),
            }
            if self.num_envs == 1:
                for key in (
                    "cross_hand_contact_load_n",
                    "cross_hand_contact_load_peak_n",
                    "external_forbidden_contact_load_n",
                    "external_forbidden_contact_load_peak_n",
                ):
                    report[key] = report[key][0]
            else:
                report["parallel_contact_evidence"] = dict(
                    self._parallel_contact_evidence
                )
            return report

    HandContactMonitorEnv.__name__ = f"HandContactMonitor{base_cls.__name__}"
    HandContactMonitorEnv.__qualname__ = HandContactMonitorEnv.__name__
    return HandContactMonitorEnv


__all__ = ["make_hand_contact_monitor_env_type"]
