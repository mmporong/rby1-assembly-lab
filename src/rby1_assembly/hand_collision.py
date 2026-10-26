"""Dedicated, physically measured RB-Y1 hand-collision failure environment.

The nominal assembly environment disables articulation self-collision.  This
module enables it only for a dedicated single-environment recording and authors
USD filtered pairs so that the three left-hand rigid bodies can collide with
the three right-hand rigid bodies while every other internal robot pair remains
filtered.  External robot/object/table/ground collision relationships are left
unchanged.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any


LEFT_HAND_BODY_PATHS = (
    "left_gripper/ee_left",
    "left_gripper/ee_finger_l1",
    "left_gripper/ee_finger_l2",
)
RIGHT_HAND_BODY_PATHS = (
    "right_gripper/ee_right",
    "right_gripper/ee_finger_r1",
    "right_gripper/ee_finger_r2",
)
EXPECTED_RIGID_BODY_COUNT = 66
CROSS_HAND_CONTACT_LOAD_SOURCE = (
    "physx_left_hand_sources_filtered_to_right_hand_sum_abs_normal_forces"
)
COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE = (
    "max_of_external_forbidden_and_cross_hand_physx_contact_loads"
)
COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION = (
    "max(external_forbidden_contact_load_n, cross_hand_contact_load_n)"
)


def _normalise_robot_path(robot_root_path: str) -> str:
    if not isinstance(robot_root_path, str) or not robot_root_path.startswith("/"):
        raise ValueError("robot_root_path must be an absolute USD path")
    return robot_root_path.rstrip("/")


def selective_hand_collision_pairs(robot_root_path: str) -> dict[str, tuple[tuple[str, str], ...]]:
    """Return the exact allowed hand pairs expected by the runtime authoring helper."""

    root = _normalise_robot_path(robot_root_path)
    left = tuple(f"{root}/{path}" for path in LEFT_HAND_BODY_PATHS)
    right = tuple(f"{root}/{path}" for path in RIGHT_HAND_BODY_PATHS)
    return {"allowed": tuple((left_path, right_path) for left_path in left for right_path in right)}


def configure_selective_hand_self_collisions(stage: Any, robot_root_path: str) -> dict[str, Any]:
    """Enable self-collision and filter all internal pairs except hand-to-hand.

    The helper fails closed if the composed robot differs from the audited
    66-body RB-Y1 asset or already contains body-level filtered-pair opinions.
    Each unordered filtered pair is authored once on the lexicographically
    first body, which makes the resulting pair count independently auditable.
    """

    from pxr import Usd, UsdPhysics

    if not isinstance(stage, Usd.Stage):
        raise TypeError("stage must be a pxr.Usd.Stage")
    root_path = _normalise_robot_path(robot_root_path)
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid() or not root.HasAPI(UsdPhysics.ArticulationRootAPI):
        raise RuntimeError(f"articulation root is missing: {root_path}")

    self_collision_attr = root.GetAttribute("physxArticulation:enabledSelfCollisions")
    if not self_collision_attr.IsValid():
        raise RuntimeError(
            f"authored physxArticulation:enabledSelfCollisions attribute is missing: {root_path}"
        )
    if self_collision_attr.Get() is not False:
        raise RuntimeError("selective hand collision setup requires self-collision initially disabled")

    rigid_body_paths = tuple(sorted(
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ))
    if len(rigid_body_paths) != EXPECTED_RIGID_BODY_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_RIGID_BODY_COUNT} robot rigid bodies; found {len(rigid_body_paths)}"
        )

    allowed_pairs = selective_hand_collision_pairs(root_path)["allowed"]
    rigid_body_set = set(rigid_body_paths)
    missing = sorted({path for pair in allowed_pairs for path in pair} - rigid_body_set)
    if missing:
        raise RuntimeError(f"required hand rigid bodies are missing: {missing}")

    for path in rigid_body_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            existing = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
            if existing:
                raise RuntimeError(f"pre-existing body filtered pairs are unsupported: {path}")

    allowed_unordered = {frozenset(pair) for pair in allowed_pairs}
    filtered_pairs = tuple(
        pair for pair in combinations(rigid_body_paths, 2)
        if frozenset(pair) not in allowed_unordered
    )
    expected_filtered_count = math.comb(EXPECTED_RIGID_BODY_COUNT, 2) - len(allowed_pairs)
    if len(allowed_unordered) != 9 or len(filtered_pairs) != expected_filtered_count:
        raise RuntimeError("internal collision-pair partition is inconsistent")

    targets_by_source: dict[str, list[str]] = {}
    for source, target in filtered_pairs:
        targets_by_source.setdefault(source, []).append(target)
    for source, targets in targets_by_source.items():
        api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(source))
        relationship = api.CreateFilteredPairsRel()
        relationship.SetTargets(targets)

    self_collision_attr.Set(True)
    if self_collision_attr.Get() is not True:
        raise RuntimeError("failed to enable articulation self-collision")

    authored_pairs: set[frozenset[str]] = set()
    for source in rigid_body_paths:
        prim = stage.GetPrimAtPath(source)
        if not prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            continue
        for target in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
            target_path = str(target)
            if target_path not in rigid_body_set or target_path == source:
                raise RuntimeError(f"invalid internal filtered-pair target: {source} -> {target_path}")
            pair = frozenset((source, target_path))
            if pair in authored_pairs:
                raise RuntimeError(f"filtered pair was authored more than once: {source}, {target_path}")
            authored_pairs.add(pair)
    if len(authored_pairs) != expected_filtered_count or authored_pairs & allowed_unordered:
        raise RuntimeError("authored filtered-pair validation failed")

    return {
        "robot_root_path": root_path,
        "self_collision_enabled": True,
        "rigid_body_count": len(rigid_body_paths),
        "allowed_cross_hand_pair_count": len(allowed_pairs),
        "filtered_internal_pair_count": len(authored_pairs),
        "allowed_cross_hand_pairs": allowed_pairs,
        "filtered_pair_authoring": "one_direction_per_unordered_rigid_body_pair",
        "external_collision_relationships_modified": False,
    }


def make_hand_collision_env_type(base_cls: type, parameters: dict[str, Any]) -> type:
    """Return a dedicated one-environment subclass for the hand collision recording."""

    target = parameters.get("target_wrist_position_world_m")
    max_error = parameters.get("max_cartesian_error_m")
    if (
        not isinstance(target, (list, tuple))
        or len(target) != 3
        or any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) for value in target)
    ):
        raise ValueError("target_wrist_position_world_m must contain three finite numbers")
    if (
        isinstance(max_error, bool)
        or not isinstance(max_error, (int, float))
        or not math.isfinite(max_error)
        or not 0.0 < max_error <= 0.010
    ):
        raise ValueError("max_cartesian_error_m must be in (0, 0.010]")
    frozen_parameters = {
        "target_wrist_position_world_m": [float(value) for value in target],
        "max_cartesian_error_m": float(max_error),
    }

    import torch
    from isaacsim.core.utils.stage import get_current_stage
    from isaaclab.sensors import ContactSensor, ContactSensorCfg
    from rby1_assembly.contact_loads import sum_contact_normal_loads

    class HandCollisionEnv(base_cls):
        """RB-Y1 failure case whose command makes the open hands physically collide."""

        def __init__(self, cfg, render_mode=None, **kwargs):
            if cfg.scene.num_envs != 1:
                raise ValueError("hand-collision recording requires exactly one environment")
            self._hand_collision_parameters = dict(frozen_parameters)
            self._hand_collision_events: list[dict[str, Any]] = []
            self._hand_collision_triggered = False
            self._hand_collision_command_active = False
            super().__init__(cfg, render_mode, **kwargs)
            self._last_cross_hand_contact_load_n = torch.zeros(self.num_envs, device=self.device)
            self._episode_cross_hand_contact_load_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._last_external_forbidden_contact_load_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._episode_external_forbidden_contact_load_peak_n = torch.zeros(
                self.num_envs, device=self.device
            )
            self._last_hand_separation_m = torch.linalg.vector_norm(
                self.robot.data.body_pos_w[:, self._left_ee_body_id]
                - self.robot.data.body_pos_w[:, self._right_ee_body_id],
                dim=1,
            )

        def _setup_scene(self):
            super()._setup_scene()
            stage = get_current_stage()
            if len(self.scene.env_prim_paths) != 1:
                raise RuntimeError("hand-collision setup requires exactly one composed environment")
            robot_root_path = f"{self.scene.env_prim_paths[0]}/Robot"
            self._hand_collision_filter_metadata = configure_selective_hand_self_collisions(
                stage, robot_root_path
            )
            # This is an instance configuration object.  The class-level nominal
            # configuration remains false and the base setup has already used it
            # to validate and deactivate the redundant wheel groups.
            self.cfg.robot.spawn.articulation_props.enabled_self_collisions = True

            sensors = {}
            right_filters = [f"{robot_root_path}/{path}" for path in RIGHT_HAND_BODY_PATHS]
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
                    f"cross_hand_{relative_path.replace('/', '_')}"
                ] = sensor
            self._cross_hand_contact_sensors = sensors

        def _desired_ee_twists(self):
            left_twist, right_twist = super()._desired_ee_twists()
            left_pos = self.robot.data.body_pos_w[:, self._left_ee_body_id]
            right_pos = self.robot.data.body_pos_w[:, self._right_ee_body_id]
            self._last_hand_separation_m = torch.linalg.vector_norm(
                left_pos - right_pos, dim=1
            )
            active = self._pregrasp_lift_reached & (
                self._phase == 0  # AssemblyPhase.PREGRASP without coupling to its module symbol.
            )
            if not torch.any(active):
                self._hand_collision_command_active = False
                return left_twist, right_twist

            target_w = torch.tensor(
                self._hand_collision_parameters["target_wrist_position_world_m"],
                dtype=left_pos.dtype,
                device=self.device,
            ).repeat(self.num_envs, 1)
            max_error = self._hand_collision_parameters["max_cartesian_error_m"]

            def bounded_error(position):
                error = target_w - position
                # Match the nominal controller's Cartesian contract: each XYZ
                # component is bounded independently before the shared DLS
                # joint-step limit is applied.
                return torch.clamp(error, -max_error, max_error)

            left_twist = left_twist.clone()
            right_twist = right_twist.clone()
            left_twist[active, :3] = bounded_error(left_pos)[active]
            right_twist[active, :3] = bounded_error(right_pos)[active]
            self._hand_collision_command_active = bool(active[0].item())
            if self._hand_collision_command_active and not self._hand_collision_triggered:
                self._hand_collision_triggered = True
                self._hand_collision_events.append({
                    "event": "both_hands_converge_command_started",
                    "issued_at_simulation_time_s": float(self._sim_time_s()),
                    "control_step_to_be_executed": int(self.common_step_counter) + 1,
                    "target_wrist_position_world_m": list(
                        self._hand_collision_parameters["target_wrist_position_world_m"]
                    ),
                    "cartesian_error_limit_definition": "per_axis_absolute_clamp",
                    "hand_separation_m": float(self._last_hand_separation_m[0].item()),
                })
            return left_twist, right_twist

        def _cross_hand_contact_load(self):
            loads = []
            for label, sensor in self._cross_hand_contact_sensors.items():
                forces, _, _, _, counts, starts = sensor.contact_physx_view.get_contact_data(
                    dt=self.cfg.sim.dt
                )
                try:
                    load = sum_contact_normal_loads(forces, counts, starts, self.num_envs)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        f"invalid cross-hand contact buffer for source {label}: {exc}"
                    ) from exc
                loads.append(load)
            if len(loads) != len(LEFT_HAND_BODY_PATHS):
                raise RuntimeError("cross-hand contact sensors are unavailable")
            total = torch.stack(loads, dim=1).sum(dim=1)
            if not torch.isfinite(total).all():
                raise RuntimeError("cross-hand contact aggregation produced a non-finite value")
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
            # Refresh after physics.  The command method records the pre-step
            # separation, which is intentionally not used as terminal evidence.
            self._last_hand_separation_m = torch.linalg.vector_norm(
                self.robot.data.body_pos_w[:, self._left_ee_body_id]
                - self.robot.data.body_pos_w[:, self._right_ee_body_id],
                dim=1,
            )
            revolute_joint_velocities = self.robot.data.joint_vel[
                :, self._torso_joint_ids + self._arm_joint_ids
            ]
            joint_velocity_finite = torch.isfinite(revolute_joint_velocities).all(dim=1)
            if not torch.all(joint_velocity_finite):
                invalid_envs = torch.nonzero(~joint_velocity_finite).flatten().tolist()
                raise RuntimeError(
                    f"non-finite revolute joint velocity in hand-collision environments: {invalid_envs}"
                )
            max_abs_joint_velocity = revolute_joint_velocities.abs().amax(dim=1)
            terminated, truncated = super()._get_dones()
            hand_fields = {
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
                "hand_wrist_separation_m": self._last_hand_separation_m.clone(),
                "hand_collision_joint_velocity_finite": joint_velocity_finite.clone(),
                "max_abs_revolute_joint_velocity_rad_s": max_abs_joint_velocity.clone(),
                "self_collision_enabled": True,
                "physics_runtime_policy": (
                    "articulation self-collision enabled; only the 9 left-hand/right-hand "
                    "rigid-body pairs are unfiltered"
                ),
                "selective_self_collision_filter": dict(
                    self._hand_collision_filter_metadata
                ),
            }
            self.extras.update(hand_fields)
            evaluation = self.extras.get("evaluation")
            if isinstance(evaluation, dict):
                evaluation.update(hand_fields)
            snapshot = self.extras.get("terminal_snapshot")
            if isinstance(snapshot, dict):
                snapshot.update(hand_fields)
                snapshot["hand_collision_command_state"] = self.hand_collision_state()
                snapshot_evaluation = snapshot.get("evaluation")
                if isinstance(snapshot_evaluation, dict):
                    snapshot_evaluation.update(hand_fields)
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
            self._last_hand_separation_m[env_ids] = torch.linalg.vector_norm(
                self.robot.data.body_pos_w[env_ids, self._left_ee_body_id]
                - self.robot.data.body_pos_w[env_ids, self._right_ee_body_id],
                dim=1,
            )
            self._hand_collision_command_active = False

        def hand_collision_state(self):
            return {
                "scenario": "hand_collision",
                "active": self._hand_collision_command_active,
                "triggered": self._hand_collision_triggered,
                "hand_wrist_separation_m": float(self._last_hand_separation_m[0].item()),
                "cross_hand_contact_load_n": float(
                    self._last_cross_hand_contact_load_n[0].item()
                ),
                "external_forbidden_contact_load_n": float(
                    self._last_external_forbidden_contact_load_n[0].item()
                ),
                "self_collision_enabled": True,
            }

        def hand_collision_report(self):
            return {
                "scope": (
                    "scripted command failure with measured PhysX hand-to-hand contact; "
                    "not a learned-policy failure or natural-failure frequency estimate"
                ),
                "scenario": "hand_collision",
                "parameters": dict(self._hand_collision_parameters),
                "triggered": self._hand_collision_triggered,
                "events": list(self._hand_collision_events),
                "modified_channel": (
                    "left and right end-effector position-error commands before DLS"
                ),
                "cartesian_error_limit_definition": "per_axis_absolute_clamp",
                "physical_state_force_sensor_or_termination_injection": False,
                "physics_runtime_policy": (
                    "articulation self-collision enabled; only the 9 left-hand/right-hand "
                    "rigid-body pairs are unfiltered"
                ),
                "collision_filter": dict(self._hand_collision_filter_metadata),
                "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
            }

    HandCollisionEnv.__name__ = f"HandCollision{base_cls.__name__}"
    HandCollisionEnv.__qualname__ = HandCollisionEnv.__name__
    return HandCollisionEnv
