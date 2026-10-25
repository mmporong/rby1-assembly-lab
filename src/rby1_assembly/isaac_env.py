"""Isaac Lab DirectRLEnv for friction-only RB-Y1 bimanual assembly.

This module is imported only after :class:`isaaclab.app.AppLauncher` starts
Isaac Sim.  It intentionally has no CPU fallback that can manufacture task
success without PhysX contacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum
import json
from pathlib import Path

import torch
from pxr import Usd, UsdPhysics

from isaacsim.core.utils.stage import get_current_stage
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    combine_frame_transforms,
    quat_apply,
    quat_apply_inverse,
    quat_box_minus,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_mul,
    subtract_frame_transforms,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "assets" / "generated"
TASK_CONFIG = json.loads((PROJECT_ROOT / "configs" / "task.json").read_text(encoding="utf-8"))
ASSET_MANIFEST = json.loads((PROJECT_ROOT / "configs" / "assets.json").read_text(encoding="utf-8"))
BUILD_REPORT = json.loads((ASSET_ROOT / "build-report.json").read_text(encoding="utf-8"))
SIMPLE_ROOM_URL = next(
    asset["verified_url"] for asset in ASSET_MANIFEST["assets"] if asset["id"] == "isaac_simple_room"
)
POLICY_OBSERVATION_SOURCE = "simulator_state_with_noise"
BASELINE_CONTROLLER_STATE_SOURCE = "simulator_ground_truth"
SOCKET_ENTRY_Y_M = -TASK_CONFIG["socket"]["length_m"] / 2.0
PLUG_TIP_Y_M = TASK_CONFIG["plug"]["grip_length_m"] / 2.0 + TASK_CONFIG["plug"]["shaft_length_m"]
# Generated from the referenced gripper mesh bound relative to the palm and
# the official wrist-to-gripper mount.  Missing provenance is a build error.
GRASP_CENTER_IN_WRIST_M = tuple(BUILD_REPORT["robot"]["grasp_center_in_wrist_m"])
GRIPPER_FINGER_SPAN_M = float(BUILD_REPORT["robot"]["gripper_finger_span_m"])
LEFT_FINGER_MIN_Z_IN_WRIST_M = float(
    BUILD_REPORT["robot"]["gripper_geometry"]["left"]["bounds_in_wrist_m"][0][2]
)
RIGHT_FINGER_MIN_Z_IN_WRIST_M = float(
    BUILD_REPORT["robot"]["gripper_geometry"]["right"]["bounds_in_wrist_m"][0][2]
)


class AssemblyPhase(IntEnum):
    PREGRASP = 0
    CLOSE = 1
    CHECK_LIFT = 2
    ALIGN = 3
    INSERT = 4
    RETRACT = 5
    DONE = 6
    ABORT = 7


class TerminationReason(IntEnum):
    NONE = 0
    INDEPENDENT_SUCCESS = 1
    HARD_FORCE_LIMIT = 2
    GRASP_LOST = 3
    PREGRASP_TIMEOUT = 4
    RETRY_LIMIT = 5
    EPISODE_TIMEOUT = 6
    FORBIDDEN_CONTACT = 7


@configclass
class Rby1AssemblyEnvCfg(DirectRLEnvCfg):
    """Simulation-design values; these are not calibrated hardware limits."""

    decimation = 4
    episode_length_s = TASK_CONFIG["evaluation"]["timeout_s"]
    action_space = 12
    observation_space = 59
    state_space = 0
    is_finite_horizon = True
    render_background = False
    background_usd_url = SIMPLE_ROOM_URL

    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=decimation,
        device="cuda:0",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=TASK_CONFIG["contact"]["static_friction"],
            dynamic_friction=TASK_CONFIG["contact"]["dynamic_friction"],
            restitution=TASK_CONFIG["contact"]["restitution"],
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_ROOT / "robot.usda"),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "torso_[0-5]": 0.0,
                "left_arm_[0-6]": 0.0,
                "right_arm_[0-6]": 0.0,
                "gripper_finger_[lr]1": -0.035,
                "gripper_finger_[lr]2": 0.035,
            },
        ),
        actuators={
            "torso_proximal_hold": ImplicitActuatorCfg(
                joint_names_expr=["torso_[0-2]"],
                effort_limit_sim=841.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "torso_distal_hold": ImplicitActuatorCfg(
                joint_names_expr=["torso_[3-5]"],
                effort_limit_sim=369.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_[0-2]", "right_arm_[0-2]"],
                effort_limit_sim=136.5217,
                stiffness=100.0,
                damping=10.0,
            ),
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_3", "right_arm_3"],
                effort_limit_sim=78.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "arm_wrist_pitch_yaw": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_[45]", "right_arm_[45]"],
                effort_limit_sim=70.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "arm_wrist_roll": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_6", "right_arm_6"],
                effort_limit_sim=44.0,
                stiffness=100.0,
                damping=10.0,
            ),
            "grippers": ImplicitActuatorCfg(
                joint_names_expr=["gripper_finger_[lr][12]"],
                effort_limit_sim=75.0,
                stiffness=1000.0,
                damping=50.0,
            ),
        },
    )
    socket: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Socket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_ROOT / "socket.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(TASK_CONFIG["socket"]["initial_position_m"])),
    )
    plug: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Plug",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(ASSET_ROOT / "plug.usda"),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(TASK_CONFIG["plug"]["initial_position_m"])),
    )
    table: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_ROOT / "table.usda")),
    )
    left_contact_l1: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper/ee_finger_l1",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Socket"],
    )
    left_contact_l2: ContactSensorCfg = left_contact_l1.replace(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper/ee_finger_l2"
    )
    right_contact_r1: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper/ee_finger_r1",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Plug"],
    )
    right_contact_r2: ContactSensorCfg = right_contact_r1.replace(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper/ee_finger_r2"
    )
    forbidden_robot_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link_.*",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
    )
    forbidden_left_palm_contact: ContactSensorCfg = forbidden_robot_contact.replace(
        prim_path="{ENV_REGEX_NS}/Robot/left_gripper/ee_left"
    )
    forbidden_right_palm_contact: ContactSensorCfg = forbidden_robot_contact.replace(
        prim_path="{ENV_REGEX_NS}/Robot/right_gripper/ee_right"
    )

    # Residual action and DLS controller parameters.
    residual_translation_bound_m = 0.003
    residual_rotation_bound_rad = 0.034906585
    dls_damping = 0.05
    max_joint_delta_rad = 0.04
    gripper_open_magnitude_m = 0.035
    left_gripper_grasp_magnitude_m = 0.028
    right_gripper_grasp_magnitude_m = 0.020
    grasp_table_clearance_m = 0.003
    pregrasp_lift_table_clearance_m = 0.030
    left_pregrasp_safe_wrist_z_m = (
        TASK_CONFIG["table"]["top_height_m"]
        - LEFT_FINGER_MIN_Z_IN_WRIST_M
        + pregrasp_lift_table_clearance_m
    )
    right_pregrasp_safe_wrist_z_m = (
        TASK_CONFIG["table"]["top_height_m"]
        - RIGHT_FINGER_MIN_Z_IN_WRIST_M
        + pregrasp_lift_table_clearance_m
    )
    left_grasp_center_above_object_m = (
        max(0.0, (GRIPPER_FINGER_SPAN_M - TASK_CONFIG["socket"]["outer_height_m"]) / 2.0)
        + grasp_table_clearance_m
    )
    right_grasp_center_above_object_m = (
        max(0.0, (GRIPPER_FINGER_SPAN_M - TASK_CONFIG["plug"]["grip_height_m"]) / 2.0)
        + grasp_table_clearance_m
    )

    # Phase gates use measured simulated state and contact-force proxies.
    pregrasp_distance_m = 0.025
    grip_contact_force_n = 1.0
    grip_lift_proof_m = 0.015
    phase_timeout_s = 5.0
    jam_contact_force_n = 20.0
    hard_contact_force_n = TASK_CONFIG["evaluation"]["max_contact_force_n"]
    assembled_depth_m = TASK_CONFIG["evaluation"]["assembled_depth_m"]
    max_lateral_error_m = TASK_CONFIG["evaluation"]["max_lateral_error_m"]
    max_orientation_error_rad = TASK_CONFIG["evaluation"]["max_orientation_error_rad"]
    success_dwell_s = TASK_CONFIG["evaluation"]["dwell_s"]
    max_retries = 2
    socket_entry_y_m = SOCKET_ENTRY_Y_M
    plug_tip_y_m = PLUG_TIP_Y_M
    assembly_socket_goal_m = (0.50, 0.05, 0.84)
    observation_position_noise_std_m = 0.0015
    observation_rotation_noise_std_rad = 0.017453293
    forbidden_contact_force_n = 1.0
    forbidden_contact_measurement_available = True


class Rby1AssemblyEnv(DirectRLEnv):
    """Batched bimanual insertion environment driven by real PhysX state."""

    cfg: Rby1AssemblyEnvCfg

    def __init__(self, cfg: Rby1AssemblyEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._arm_joint_ids = self._resolve_exact_joints(
            [*[f"left_arm_{i}" for i in range(7)], *[f"right_arm_{i}" for i in range(7)]]
        )
        self._left_arm_joint_ids = self._arm_joint_ids[:7]
        self._right_arm_joint_ids = self._arm_joint_ids[7:]
        self._torso_joint_ids = self._resolve_exact_joints([f"torso_{i}" for i in range(6)])
        self._gripper_joint_ids = self._resolve_exact_joints(
            ["gripper_finger_l1", "gripper_finger_l2", "gripper_finger_r1", "gripper_finger_r2"]
        )
        self._left_ee_body_id = self._resolve_exact_body("link_left_arm_6")
        self._right_ee_body_id = self._resolve_exact_body("link_right_arm_6")
        jacobian_offset = 1 if self.robot.is_fixed_base else 0
        self._left_ee_jacobian_id = self._left_ee_body_id - jacobian_offset
        self._right_ee_jacobian_id = self._right_ee_body_id - jacobian_offset

        self._actions = torch.zeros((self.num_envs, 12), device=self.device)
        self._phase = torch.full(
            (self.num_envs,), int(AssemblyPhase.PREGRASP), dtype=torch.long, device=self.device
        )
        self._phase_started_s = torch.zeros(self.num_envs, device=self.device)
        self._retry_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._success_since_s = torch.full((self.num_envs,), -1.0, device=self.device)
        self._evaluation_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluation_dwell_s = torch.zeros(self.num_envs, device=self.device)
        self._termination_reason_code = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._pregrasp_above_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._pregrasp_lift_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._grip_proven = torch.zeros((self.num_envs, 2), dtype=torch.bool, device=self.device)
        self._evaluation_grip_proven = torch.zeros(
            (self.num_envs, 2), dtype=torch.bool, device=self.device
        )
        self._object_start_z = torch.zeros((self.num_envs, 2), device=self.device)
        self._object_to_ee_pos = torch.zeros((self.num_envs, 2, 3), device=self.device)
        self._object_to_ee_quat = torch.zeros((self.num_envs, 2, 4), device=self.device)
        self._object_to_ee_quat[:, :, 0] = 1.0
        self._joint_targets = self.robot.data.default_joint_pos.clone()
        self._torso_hold_targets = self.robot.data.default_joint_pos[:, self._torso_joint_ids].clone()
        self._last_contact_proxy_n = torch.zeros((self.num_envs, 2), device=self.device)
        self._episode_force_peak_n = torch.zeros(self.num_envs, device=self.device)
        self._last_actual_relative_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self._forbidden_contact_seen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _setup_scene(self) -> None:
        robot_cfg = self._resolve_env_paths(self.cfg.robot)
        socket_cfg = self._resolve_env_paths(self.cfg.socket)
        plug_cfg = self._resolve_env_paths(self.cfg.plug)
        table_cfg = self._resolve_env_paths(self.cfg.table)
        left_l1_cfg = self._resolve_env_paths(self.cfg.left_contact_l1)
        left_l2_cfg = self._resolve_env_paths(self.cfg.left_contact_l2)
        right_r1_cfg = self._resolve_env_paths(self.cfg.right_contact_r1)
        right_r2_cfg = self._resolve_env_paths(self.cfg.right_contact_r2)
        forbidden_robot_cfg = self._resolve_env_paths(self.cfg.forbidden_robot_contact)
        forbidden_left_palm_cfg = self._resolve_env_paths(self.cfg.forbidden_left_palm_contact)
        forbidden_right_palm_cfg = self._resolve_env_paths(self.cfg.forbidden_right_palm_contact)

        self.robot = Articulation(robot_cfg)
        self.socket = RigidObject(socket_cfg)
        self.plug = RigidObject(plug_cfg)
        if table_cfg.spawn is None:
            raise RuntimeError("table spawn configuration is required")
        table_cfg.spawn.func(
            table_cfg.prim_path,
            table_cfg.spawn,
            translation=table_cfg.init_state.pos,
            orientation=table_cfg.init_state.rot,
        )
        self.left_contact_l1 = ContactSensor(left_l1_cfg)
        self.left_contact_l2 = ContactSensor(left_l2_cfg)
        self.right_contact_r1 = ContactSensor(right_r1_cfg)
        self.right_contact_r2 = ContactSensor(right_r2_cfg)
        self.forbidden_robot_contact = ContactSensor(forbidden_robot_cfg)
        self.forbidden_left_palm_contact = ContactSensor(forbidden_left_palm_cfg)
        self.forbidden_right_palm_contact = ContactSensor(forbidden_right_palm_cfg)

        spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())
        if self.cfg.render_background:
            background_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.background_usd_url)
            background_cfg.func("/World/Background", background_cfg)
            stage = get_current_stage()
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Background")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["socket"] = self.socket
        self.scene.rigid_objects["plug"] = self.plug
        self.scene.sensors["left_contact_l1"] = self.left_contact_l1
        self.scene.sensors["left_contact_l2"] = self.left_contact_l2
        self.scene.sensors["right_contact_r1"] = self.right_contact_r1
        self.scene.sensors["right_contact_r2"] = self.right_contact_r2
        self.scene.sensors["forbidden_robot_contact"] = self.forbidden_robot_contact
        self.scene.sensors["forbidden_left_palm_contact"] = self.forbidden_left_palm_contact
        self.scene.sensors["forbidden_right_palm_contact"] = self.forbidden_right_palm_contact
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.88, 0.92))
        light_cfg.func("/World/Light", light_cfg)

    def _resolve_env_paths(self, cfg):
        resolved = cfg.copy()
        resolved.prim_path = resolved.prim_path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
        if isinstance(resolved, ContactSensorCfg):
            resolved.filter_prim_paths_expr = [
                path.format(ENV_REGEX_NS=self.scene.env_regex_ns)
                for path in resolved.filter_prim_paths_expr
            ]
        return resolved

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._advance_phases()
        self._actions = self._bound_residual_actions(actions)
        left_twist, right_twist = self._desired_ee_twists()
        left_twist[:, :3] += self._actions[:, :3] * self.cfg.residual_translation_bound_m
        left_twist[:, 3:] += self._actions[:, 3:6] * self.cfg.residual_rotation_bound_rad
        right_twist[:, :3] += self._actions[:, 6:9] * self.cfg.residual_translation_bound_m
        right_twist[:, 3:] += self._actions[:, 9:12] * self.cfg.residual_rotation_bound_rad

        jacobians = self.robot.root_physx_view.get_jacobians()
        left_jacobian = jacobians[:, self._left_ee_jacobian_id, :, self._left_arm_joint_ids]
        right_jacobian = jacobians[:, self._right_ee_jacobian_id, :, self._right_arm_joint_ids]
        left_delta = self._damped_least_squares(left_jacobian, left_twist)
        right_delta = self._damped_least_squares(right_jacobian, right_twist)

        current = self.robot.data.joint_pos
        self._joint_targets[:] = current
        self._joint_targets[:, self._left_arm_joint_ids] = current[:, self._left_arm_joint_ids] + left_delta
        self._joint_targets[:, self._right_arm_joint_ids] = current[:, self._right_arm_joint_ids] + right_delta
        self._joint_targets[:, self._torso_joint_ids] = self._torso_hold_targets
        close_mask = self._phase >= int(AssemblyPhase.CLOSE)
        open_targets = torch.tensor(
            [-1.0, 1.0, -1.0, 1.0], device=self.device
        ) * self.cfg.gripper_open_magnitude_m
        grasp_targets = torch.tensor(
            [
                -self.cfg.left_gripper_grasp_magnitude_m,
                self.cfg.left_gripper_grasp_magnitude_m,
                -self.cfg.right_gripper_grasp_magnitude_m,
                self.cfg.right_gripper_grasp_magnitude_m,
            ],
            device=self.device,
        )
        self._joint_targets[:, self._gripper_joint_ids] = torch.where(
            close_mask.unsqueeze(1), grasp_targets.unsqueeze(0), open_targets.unsqueeze(0)
        )
        lower = self.robot.data.soft_joint_pos_limits[:, :, 0]
        upper = self.robot.data.soft_joint_pos_limits[:, :, 1]
        self._joint_targets = torch.maximum(torch.minimum(self._joint_targets, upper), lower)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        relative_pos, relative_quat = self._relative_object_pose()
        noisy_pos = relative_pos + torch.randn_like(relative_pos) * self.cfg.observation_position_noise_std_m
        rotation_vector_noise = (
            torch.randn((self.num_envs, 3), device=self.device)
            * self.cfg.observation_rotation_noise_std_rad
        )
        noise_angle = torch.linalg.vector_norm(rotation_vector_noise, dim=1)
        noise_axis = rotation_vector_noise / noise_angle.unsqueeze(1).clamp_min(1.0e-9)
        noise_quat = quat_from_angle_axis(noise_angle, noise_axis)
        noisy_quat = quat_mul(relative_quat, noise_quat)
        phase_one_hot = torch.nn.functional.one_hot(self._phase, num_classes=len(AssemblyPhase)).float()
        arm_pos = self.robot.data.joint_pos[:, self._arm_joint_ids]
        arm_vel = self.robot.data.joint_vel[:, self._arm_joint_ids]
        observations = torch.cat(
            (
                noisy_pos,
                noisy_quat,
                arm_pos,
                arm_vel,
                self._last_contact_proxy_n,
                self._grip_proven.float(),
                phase_one_hot,
                self._actions,
            ),
            dim=-1,
        )
        self.extras["observation_source"] = POLICY_OBSERVATION_SOURCE
        return {"policy": observations}

    def _get_rewards(self) -> torch.Tensor:
        relative_pos, relative_quat = self._relative_object_pose()
        depth = relative_pos[:, 1]
        lateral = torch.linalg.vector_norm(relative_pos[:, (0, 2)], dim=1)
        angle = quat_error_magnitude(relative_quat, self._identity_quat())
        force_penalty = torch.clamp(
            self._last_contact_proxy_n.max(dim=1).values / self.cfg.hard_contact_force_n, 0.0, 2.0
        )
        action_penalty = torch.sum(self._actions.square(), dim=1)
        proven = torch.all(self._grip_proven, dim=1).float()
        success = self._evaluation_success
        return (
            -4.0 * lateral
            - 0.4 * angle
            + 2.0 * torch.clamp(depth, min=0.0)
            + 0.2 * proven
            + 10.0 * success.float()
            - 0.05 * force_penalty
            - 0.002 * action_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_l1_force = self._filtered_force_magnitude(self.left_contact_l1)
        left_l2_force = self._filtered_force_magnitude(self.left_contact_l2)
        right_r1_force = self._filtered_force_magnitude(self.right_contact_r1)
        right_r2_force = self._filtered_force_magnitude(self.right_contact_r2)
        left_force = torch.maximum(left_l1_force, left_l2_force)
        right_force = torch.maximum(right_r1_force, right_r2_force)
        self._last_contact_proxy_n[:, 0] = left_force
        self._last_contact_proxy_n[:, 1] = right_force
        current_force = torch.maximum(left_force, right_force)
        self._episode_force_peak_n = torch.maximum(self._episode_force_peak_n, current_force)

        left_contact = (left_l1_force >= self.cfg.grip_contact_force_n) & (
            left_l2_force >= self.cfg.grip_contact_force_n
        )
        right_contact = (right_r1_force >= self.cfg.grip_contact_force_n) & (
            right_r2_force >= self.cfg.grip_contact_force_n
        )
        grip_live = torch.stack((left_contact, right_contact), dim=1)
        lift_left = self.socket.data.root_pos_w[:, 2] - (
            self.scene.env_origins[:, 2] + self._object_start_z[:, 0]
        )
        lift_right = self.plug.data.root_pos_w[:, 2] - (
            self.scene.env_origins[:, 2] + self._object_start_z[:, 1]
        )
        physical_grip_proof = grip_live & torch.stack(
            (
                lift_left >= self.cfg.grip_lift_proof_m,
                lift_right >= self.cfg.grip_lift_proof_m,
            ),
            dim=1,
        )
        self._evaluation_grip_proven |= physical_grip_proof
        relative_pos, relative_quat = self._relative_object_pose()
        depth = relative_pos[:, 1]
        lateral = torch.linalg.vector_norm(relative_pos[:, (0, 2)], dim=1)
        angle = quat_error_magnitude(relative_quat, self._identity_quat())
        forbidden_now = self._forbidden_contact_magnitude() > self.cfg.forbidden_contact_force_n
        self._forbidden_contact_seen |= forbidden_now
        hard_overload = self._episode_force_peak_n > self.cfg.hard_contact_force_n
        geometry_ok = (
            (depth >= self.cfg.assembled_depth_m)
            & (lateral <= self.cfg.max_lateral_error_m)
            & (angle <= self.cfg.max_orientation_error_rad)
        )
        handling_ok = (
            torch.all(self._evaluation_grip_proven, dim=1)
            & torch.all(grip_live, dim=1)
            & (lift_left >= self.cfg.grip_lift_proof_m)
            & (lift_right >= self.cfg.grip_lift_proof_m)
        )
        instantaneous_ok = geometry_ok & handling_ok & ~hard_overload & ~self._forbidden_contact_seen
        now = self._sim_time_s()
        self._advance_evaluation_dwell(instantaneous_ok, now)
        independent_eval_valid = torch.full(
            (self.num_envs,),
            bool(self.cfg.forbidden_contact_measurement_available),
            dtype=torch.bool,
            device=self.device,
        )
        self._evaluation_success = independent_eval_valid & (
            self._evaluation_dwell_s >= self.cfg.success_dwell_s
        )
        aborted = self._phase == int(AssemblyPhase.ABORT)
        evaluator_failure = self._forbidden_contact_seen | hard_overload
        terminated = self._evaluation_success | aborted | evaluator_failure
        truncated = self._episode_time_limit_reached()
        terminal_mask = terminated | truncated
        terminal_reason = self._termination_reason_code.clone()
        terminal_reason[self._evaluation_success] = int(TerminationReason.INDEPENDENT_SUCCESS)
        unclassified_hard = hard_overload & (terminal_reason == int(TerminationReason.NONE))
        terminal_reason[unclassified_hard] = int(TerminationReason.HARD_FORCE_LIMIT)
        unclassified_forbidden = self._forbidden_contact_seen & (
            terminal_reason == int(TerminationReason.NONE)
        )
        terminal_reason[unclassified_forbidden] = int(TerminationReason.FORBIDDEN_CONTACT)
        timed_out = truncated & (terminal_reason == int(TerminationReason.NONE))
        terminal_reason[timed_out] = int(TerminationReason.EPISODE_TIMEOUT)
        self._phase[self._evaluation_success] = int(AssemblyPhase.DONE)
        self._phase[evaluator_failure] = int(AssemblyPhase.ABORT)

        object_poses = torch.stack(
            (
                torch.cat((self.socket.data.root_pos_w, self.socket.data.root_quat_w), dim=1),
                torch.cat((self.plug.data.root_pos_w, self.plug.data.root_quat_w), dim=1),
            ),
            dim=1,
        )
        self.extras["success"] = self._evaluation_success.clone()
        self.extras["completed"] = bool(torch.any(self._evaluation_success).item())
        self.extras["contact_force_source"] = "contact_force_proxy"
        self.extras["baseline_controller_state_source"] = BASELINE_CONTROLLER_STATE_SOURCE
        self.extras["phase_gate_state_source"] = BASELINE_CONTROLLER_STATE_SOURCE
        self.extras["evaluation"] = {
            "valid": independent_eval_valid.clone(),
            "success": self._evaluation_success.clone(),
            "dwell_s": self._evaluation_dwell_s.clone(),
            "actual_depth_m": depth.clone(),
            "actual_lateral_error_m": lateral.clone(),
            "actual_orientation_error_rad": angle.clone(),
            "contact_force_proxy_peak_n": self._episode_force_peak_n.clone(),
            "forbidden_contact": self._forbidden_contact_seen.clone(),
            "grasp_both": handling_ok.clone(),
            "completion_source": "simulator_ground_truth_evaluator",
        }
        evaluation_snapshot = {
            "actual_depth_m": depth.clone(),
            "actual_lateral_error_m": lateral.clone(),
            "actual_orientation_error_rad": angle.clone(),
            "contact_force_proxy_peak_n": self._episode_force_peak_n.clone(),
            "grasp_both": handling_ok.clone(),
            "forbidden_contact": self._forbidden_contact_seen.clone(),
        }
        self.extras["terminal_snapshot"] = {
            "terminal_mask": terminal_mask.clone(),
            "timestamp_s": (self.episode_length_buf.float() * self.step_dt).clone(),
            "retry_count": self._retry_count.clone(),
            "revolute_joint_names": tuple(
                [*[f"torso_{i}" for i in range(6)], *[f"left_arm_{i}" for i in range(7)],
                 *[f"right_arm_{i}" for i in range(7)]]
            ),
            "revolute_joint_positions_rad": self.robot.data.joint_pos[
                :, self._torso_joint_ids + self._arm_joint_ids
            ].clone(),
            "revolute_joint_velocities_rad_s": self.robot.data.joint_vel[
                :, self._torso_joint_ids + self._arm_joint_ids
            ].clone(),
            "gripper_joint_names": (
                "gripper_finger_l1", "gripper_finger_l2",
                "gripper_finger_r1", "gripper_finger_r2",
            ),
            "gripper_positions_m": self.robot.data.joint_pos[:, self._gripper_joint_ids].clone(),
            "gripper_velocities_m_s": self.robot.data.joint_vel[:, self._gripper_joint_ids].clone(),
            "object_order": ("socket", "plug"),
            "object_poses_world_wxyz": object_poses.clone(),
            "contact_force_proxy_n": self._last_contact_proxy_n.clone(),
            "contact_force_proxy_format": "per_hand_peak_filtered_finger_magnitude_n",
            "phase": self._phase.clone(),
            "phase_name_by_code": {int(phase): phase.name.lower() for phase in AssemblyPhase},
            "grip_proven": self._evaluation_grip_proven.clone(),
            "controller_grip_proven": self._grip_proven.clone(),
            "grip_live": grip_live.clone(),
            "actual_depth_m": depth.clone(),
            "actual_lateral_error_m": lateral.clone(),
            "actual_orientation_error_rad": angle.clone(),
            "contact_force_proxy_peak_n": self._episode_force_peak_n.clone(),
            "forbidden_contact": self._forbidden_contact_seen.clone(),
            "forbidden_contact_measured": independent_eval_valid.clone(),
            "forbidden_contact_source": (
                "non_finger_body_net_force_and_finger_non_target_residual_proxy"
            ),
            "self_collision_enabled": False,
            "hard_overload": hard_overload.clone(),
            "independent_eval_valid": independent_eval_valid.clone(),
            "independent_eval_success": self._evaluation_success.clone(),
            "independent_eval_dwell_s": self._evaluation_dwell_s.clone(),
            "termination_reason_code": terminal_reason.clone(),
            "termination_reason_by_code": {
                int(reason): reason.name.lower() for reason in TerminationReason
            },
            "evaluation": evaluation_snapshot,
        }
        termination_reason_names = tuple(
            TerminationReason(code).name.lower() for code in terminal_reason.detach().cpu().tolist()
        )
        self.extras["terminal_evaluation"] = {
            "evaluator_source": "simulator_ground_truth_independent",
            "evaluator_version": "tip-entry-geometry-contact-v1",
            "valid": independent_eval_valid.clone(),
            "success": self._evaluation_success.clone(),
            "termination_reason": termination_reason_names,
            "criteria": {
                "depth": (depth >= self.cfg.assembled_depth_m).clone(),
                "lateral": (lateral <= self.cfg.max_lateral_error_m).clone(),
                "orientation": (angle <= self.cfg.max_orientation_error_rad).clone(),
                "grasp_live_and_proven": handling_ok.clone(),
                "dwell": (self._evaluation_dwell_s >= self.cfg.success_dwell_s).clone(),
                "force": (~hard_overload).clone(),
                "forbidden_contact": (~self._forbidden_contact_seen).clone(),
            },
        }
        return terminated, truncated

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None) -> None:
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        robot_root = self.robot.data.default_root_state[env_ids].clone()
        robot_root[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_root_pose_to_sim(robot_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._reset_rigid_object(self.socket, env_ids, y_noise=0.010)
        self._reset_rigid_object(self.plug, env_ids, y_noise=0.010)
        self._joint_targets[env_ids] = joint_pos
        self._torso_hold_targets[env_ids] = joint_pos[:, self._torso_joint_ids]
        self._phase[env_ids] = int(AssemblyPhase.PREGRASP)
        self._phase_started_s[env_ids] = self._sim_time_s()
        self._retry_count[env_ids] = 0
        self._success_since_s[env_ids] = -1.0
        self._evaluation_success[env_ids] = False
        self._evaluation_dwell_s[env_ids] = 0.0
        self._termination_reason_code[env_ids] = int(TerminationReason.NONE)
        self._pregrasp_above_reached[env_ids] = False
        self._pregrasp_lift_reached[env_ids] = False
        self._grip_proven[env_ids] = False
        self._evaluation_grip_proven[env_ids] = False
        self._object_to_ee_pos[env_ids] = 0.0
        self._object_to_ee_quat[env_ids] = 0.0
        self._object_to_ee_quat[env_ids, :, 0] = 1.0
        self._episode_force_peak_n[env_ids] = 0.0
        self._forbidden_contact_seen[env_ids] = False
        self._last_contact_proxy_n[env_ids] = 0.0
        self._last_actual_relative_pose[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._object_start_z[env_ids, 0] = self.socket.data.default_root_state[env_ids, 2]
        self._object_start_z[env_ids, 1] = self.plug.data.default_root_state[env_ids, 2]

    def _reset_rigid_object(self, obj: RigidObject, env_ids: Sequence[int] | torch.Tensor, y_noise: float) -> None:
        state = obj.data.default_root_state[env_ids].clone()
        state[:, :3] += self.scene.env_origins[env_ids]
        state[:, 0] += (torch.rand(len(env_ids), device=self.device) - 0.5) * 0.010
        state[:, 1] += (torch.rand(len(env_ids), device=self.device) - 0.5) * 2.0 * y_noise
        obj.write_root_pose_to_sim(state[:, :7], env_ids)
        obj.write_root_velocity_to_sim(state[:, 7:], env_ids)

    def _advance_phases(self) -> None:
        now = self._sim_time_s()
        left_l1_force = self._filtered_force_magnitude(self.left_contact_l1)
        left_l2_force = self._filtered_force_magnitude(self.left_contact_l2)
        right_r1_force = self._filtered_force_magnitude(self.right_contact_r1)
        right_r2_force = self._filtered_force_magnitude(self.right_contact_r2)
        left_force = torch.maximum(left_l1_force, left_l2_force)
        right_force = torch.maximum(right_r1_force, right_r2_force)
        left_contact = (left_l1_force >= self.cfg.grip_contact_force_n) & (
            left_l2_force >= self.cfg.grip_contact_force_n
        )
        right_contact = (right_r1_force >= self.cfg.grip_contact_force_n) & (
            right_r2_force >= self.cfg.grip_contact_force_n
        )
        self._last_contact_proxy_n[:, 0] = left_force
        self._last_contact_proxy_n[:, 1] = right_force

        left_ee = self.robot.data.body_pos_w[:, self._left_ee_body_id]
        right_ee = self.robot.data.body_pos_w[:, self._right_ee_body_id]
        grasp_offset = torch.tensor(GRASP_CENTER_IN_WRIST_M, device=self.device).repeat(self.num_envs, 1)
        left_grasp_center = left_ee + quat_apply(
            self.robot.data.body_quat_w[:, self._left_ee_body_id], grasp_offset
        )
        right_grasp_center = right_ee + quat_apply(
            self.robot.data.body_quat_w[:, self._right_ee_body_id], grasp_offset
        )
        lift_reached = self._pregrasp_lift_mask(left_ee, right_ee)
        self._pregrasp_lift_reached |= (
            (self._phase == int(AssemblyPhase.PREGRASP)) & lift_reached
        )
        left_grasp_target, right_grasp_target = self._grasp_target_positions()
        left_distance = torch.linalg.vector_norm(left_grasp_center - left_grasp_target, dim=1)
        right_distance = torch.linalg.vector_norm(right_grasp_center - right_grasp_target, dim=1)
        above_reached = self._pregrasp_above_mask(left_grasp_center, right_grasp_center)
        self._pregrasp_above_reached |= (
            (self._phase == int(AssemblyPhase.PREGRASP))
            & self._pregrasp_lift_reached
            & above_reached
        )

        self._transition_where(
            (self._phase == int(AssemblyPhase.PREGRASP))
            & self._pregrasp_above_reached
            & (left_distance < self.cfg.pregrasp_distance_m)
            & (right_distance < self.cfg.pregrasp_distance_m),
            AssemblyPhase.CLOSE,
            now,
        )
        self._transition_where(
            (self._phase == int(AssemblyPhase.CLOSE))
            & left_contact
            & right_contact,
            AssemblyPhase.CHECK_LIFT,
            now,
        )

        lift_left = self.socket.data.root_pos_w[:, 2] - (self.scene.env_origins[:, 2] + self._object_start_z[:, 0])
        lift_right = self.plug.data.root_pos_w[:, 2] - (self.scene.env_origins[:, 2] + self._object_start_z[:, 1])
        proof = (
            (self._phase == int(AssemblyPhase.CHECK_LIFT))
            & left_contact
            & right_contact
            & (lift_left >= self.cfg.grip_lift_proof_m)
            & (lift_right >= self.cfg.grip_lift_proof_m)
        )
        if torch.any(proof):
            self._grip_proven[proof] = True
            left_offset_pos, left_offset_quat = subtract_frame_transforms(
                self.socket.data.root_pos_w[proof],
                self.socket.data.root_quat_w[proof],
                left_ee[proof],
                self.robot.data.body_quat_w[proof, self._left_ee_body_id],
            )
            right_offset_pos, right_offset_quat = subtract_frame_transforms(
                self.plug.data.root_pos_w[proof],
                self.plug.data.root_quat_w[proof],
                right_ee[proof],
                self.robot.data.body_quat_w[proof, self._right_ee_body_id],
            )
            self._object_to_ee_pos[proof, 0] = left_offset_pos
            self._object_to_ee_quat[proof, 0] = left_offset_quat
            self._object_to_ee_pos[proof, 1] = right_offset_pos
            self._object_to_ee_quat[proof, 1] = right_offset_quat
        self._transition_where(proof, AssemblyPhase.ALIGN, now)

        relative_pos, relative_quat = self._relative_object_pose()
        lateral = torch.linalg.vector_norm(relative_pos[:, (0, 2)], dim=1)
        angle = quat_error_magnitude(relative_quat, self._identity_quat())
        aligned = (
            (self._phase == int(AssemblyPhase.ALIGN))
            & (relative_pos[:, 1] > -0.012)
            & (lateral < self.cfg.max_lateral_error_m * 1.5)
            & (angle < self.cfg.max_orientation_error_rad * 1.5)
        )
        self._transition_where(aligned, AssemblyPhase.INSERT, now)

        current_force = torch.maximum(left_force, right_force)
        continuous_grasp = left_contact & right_contact & (lift_left >= self.cfg.grip_lift_proof_m) & (
            lift_right >= self.cfg.grip_lift_proof_m
        )

        hard_overload = current_force > self.cfg.hard_contact_force_n
        self._transition_where(
            hard_overload, AssemblyPhase.ABORT, now, TerminationReason.HARD_FORCE_LIMIT
        )
        jam = (
            (self._phase == int(AssemblyPhase.INSERT))
            & (current_force > self.cfg.jam_contact_force_n)
            & ~hard_overload
        )
        self._start_recovery(jam, now)
        lost_grasp = (
            (self._phase >= int(AssemblyPhase.ALIGN))
            & (self._phase <= int(AssemblyPhase.RETRACT))
            & ~continuous_grasp
            & ((now - self._phase_started_s) > 0.20)
        )
        self._transition_where(lost_grasp, AssemblyPhase.ABORT, now, TerminationReason.GRASP_LOST)
        retract_done = (self._phase == int(AssemblyPhase.RETRACT)) & (relative_pos[:, 1] <= -0.015)
        self._transition_where(retract_done, AssemblyPhase.ALIGN, now)

        timed_out = (now - self._phase_started_s) >= self.cfg.phase_timeout_s
        recoverable = timed_out & (
            (self._phase == int(AssemblyPhase.ALIGN)) | (self._phase == int(AssemblyPhase.INSERT))
        )
        self._start_recovery(recoverable, now)
        pregrasp_failed = timed_out & (self._phase <= int(AssemblyPhase.CHECK_LIFT))
        self._transition_where(
            pregrasp_failed, AssemblyPhase.ABORT, now, TerminationReason.PREGRASP_TIMEOUT
        )

    def _start_recovery(self, mask: torch.Tensor, now: float) -> None:
        retryable = mask & (self._retry_count < self.cfg.max_retries)
        self._retry_count[retryable] += 1
        self._transition_where(retryable, AssemblyPhase.RETRACT, now)
        exhausted = mask & ~retryable
        self._transition_where(exhausted, AssemblyPhase.ABORT, now, TerminationReason.RETRY_LIMIT)

    def _desired_ee_twists(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_pos = self.robot.data.body_pos_w[:, self._left_ee_body_id]
        right_pos = self.robot.data.body_pos_w[:, self._right_ee_body_id]
        left_quat = self.robot.data.body_quat_w[:, self._left_ee_body_id]
        right_quat = self.robot.data.body_quat_w[:, self._right_ee_body_id]
        target_quat = self._identity_quat()
        grasp_center_offset = torch.tensor(GRASP_CENTER_IN_WRIST_M, device=self.device).repeat(self.num_envs, 1)
        left_grasp_target, right_grasp_target = self._grasp_target_positions()
        left_target = left_grasp_target - quat_apply(target_quat, grasp_center_offset)
        right_target = right_grasp_target - quat_apply(target_quat, grasp_center_offset)
        left_target_quat = target_quat.clone()
        right_target_quat = target_quat.clone()

        pregrasp_lift = (
            (self._phase == int(AssemblyPhase.PREGRASP)) & ~self._pregrasp_lift_reached
        )
        left_lift_target, right_lift_target = self._pregrasp_lift_targets(left_pos, right_pos)
        left_target[pregrasp_lift] = left_lift_target[pregrasp_lift]
        right_target[pregrasp_lift] = right_lift_target[pregrasp_lift]
        pregrasp_above = (
            (self._phase == int(AssemblyPhase.PREGRASP))
            & self._pregrasp_lift_reached
            & ~self._pregrasp_above_reached
        )
        left_target[pregrasp_above, 2] += 0.050
        right_target[pregrasp_above, 2] += 0.050
        check_lift = self._phase == int(AssemblyPhase.CHECK_LIFT)
        left_target[check_lift, 2] = (
            self.scene.env_origins[check_lift, 2]
            + self._object_start_z[check_lift, 0]
            + 0.050
            + self.cfg.left_grasp_center_above_object_m
            - GRASP_CENTER_IN_WRIST_M[2]
        )
        right_target[check_lift, 2] = (
            self.scene.env_origins[check_lift, 2]
            + self._object_start_z[check_lift, 1]
            + 0.050
            + self.cfg.right_grasp_center_above_object_m
            - GRASP_CENTER_IN_WRIST_M[2]
        )

        manipulation = (self._phase >= int(AssemblyPhase.ALIGN)) & (
            self._phase <= int(AssemblyPhase.RETRACT)
        )
        socket_goal = self.scene.env_origins + torch.tensor(self.cfg.assembly_socket_goal_m, device=self.device)
        plug_goal = self._plug_center_goal(socket_goal, -0.010)
        insert = self._phase == int(AssemblyPhase.INSERT)
        plug_goal[insert] = self._plug_center_goal(socket_goal[insert], self.cfg.assembled_depth_m)
        retract = self._phase == int(AssemblyPhase.RETRACT)
        plug_goal[retract] = self._plug_center_goal(socket_goal[retract], -0.030)
        left_goal, left_goal_quat = combine_frame_transforms(
            socket_goal,
            target_quat,
            self._object_to_ee_pos[:, 0],
            self._object_to_ee_quat[:, 0],
        )
        right_goal, right_goal_quat = combine_frame_transforms(
            plug_goal,
            target_quat,
            self._object_to_ee_pos[:, 1],
            self._object_to_ee_quat[:, 1],
        )
        left_target[manipulation] = left_goal[manipulation]
        right_target[manipulation] = right_goal[manipulation]
        left_target_quat[manipulation] = left_goal_quat[manipulation]
        right_target_quat[manipulation] = right_goal_quat[manipulation]

        left_twist = torch.zeros((self.num_envs, 6), device=self.device)
        right_twist = torch.zeros((self.num_envs, 6), device=self.device)
        left_twist[:, :3] = torch.clamp(left_target - left_pos, -0.010, 0.010)
        right_twist[:, :3] = torch.clamp(right_target - right_pos, -0.010, 0.010)
        left_twist[:, 3:] = torch.clamp(quat_box_minus(left_target_quat, left_quat), -0.08, 0.08)
        right_twist[:, 3:] = torch.clamp(quat_box_minus(right_target_quat, right_quat), -0.08, 0.08)
        terminal = (self._phase == int(AssemblyPhase.DONE)) | (self._phase == int(AssemblyPhase.ABORT))
        left_twist[terminal] = 0.0
        right_twist[terminal] = 0.0
        return left_twist, right_twist

    def _damped_least_squares(self, jacobian: torch.Tensor, twist: torch.Tensor) -> torch.Tensor:
        identity = torch.eye(6, device=self.device).unsqueeze(0).expand(self.num_envs, -1, -1)
        lhs = jacobian @ jacobian.transpose(1, 2) + (self.cfg.dls_damping**2) * identity
        delta = jacobian.transpose(1, 2) @ torch.linalg.solve(lhs, twist.unsqueeze(-1))
        return torch.clamp(delta.squeeze(-1), -self.cfg.max_joint_delta_rad, self.cfg.max_joint_delta_rad)

    def _relative_object_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        plug_tip_local = torch.tensor([0.0, self.cfg.plug_tip_y_m, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        socket_entry_local = torch.tensor([0.0, self.cfg.socket_entry_y_m, 0.0], device=self.device).repeat(
            self.num_envs, 1
        )
        plug_tip_w = self.plug.data.root_pos_w + quat_apply(self.plug.data.root_quat_w, plug_tip_local)
        socket_entry_w = self.socket.data.root_pos_w + quat_apply(
            self.socket.data.root_quat_w, socket_entry_local
        )
        relative_pos = quat_apply_inverse(self.socket.data.root_quat_w, plug_tip_w - socket_entry_w)
        _, relative_quat = subtract_frame_transforms(
            self.socket.data.root_pos_w,
            self.socket.data.root_quat_w,
            self.plug.data.root_pos_w,
            self.plug.data.root_quat_w,
        )
        self._last_actual_relative_pose[:, :3] = relative_pos
        self._last_actual_relative_pose[:, 3:] = relative_quat
        return relative_pos, relative_quat

    def _plug_center_goal(self, socket_center_w: torch.Tensor, depth_m: float) -> torch.Tensor:
        goal = socket_center_w.clone()
        goal[:, 1] += self.cfg.socket_entry_y_m + depth_m - self.cfg.plug_tip_y_m
        return goal

    def _pregrasp_above_mask(
        self, left_grasp_center: torch.Tensor, right_grasp_center: torch.Tensor
    ) -> torch.Tensor:
        left_above, right_above = self._grasp_target_positions()
        left_above[:, 2] += 0.050
        right_above[:, 2] += 0.050
        return (
            (torch.linalg.vector_norm(left_grasp_center - left_above, dim=1) < 0.015)
            & (torch.linalg.vector_norm(right_grasp_center - right_above, dim=1) < 0.015)
        )

    def _pregrasp_lift_targets(
        self, left_ee: torch.Tensor, right_ee: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        left_target = left_ee.clone()
        right_target = right_ee.clone()
        left_target[:, 2] = self.scene.env_origins[:, 2] + self.cfg.left_pregrasp_safe_wrist_z_m
        right_target[:, 2] = self.scene.env_origins[:, 2] + self.cfg.right_pregrasp_safe_wrist_z_m
        return left_target, right_target

    def _pregrasp_lift_mask(
        self, left_ee: torch.Tensor, right_ee: torch.Tensor
    ) -> torch.Tensor:
        left_safe_z = self.scene.env_origins[:, 2] + self.cfg.left_pregrasp_safe_wrist_z_m
        right_safe_z = self.scene.env_origins[:, 2] + self.cfg.right_pregrasp_safe_wrist_z_m
        return (left_ee[:, 2] >= left_safe_z - 0.005) & (
            right_ee[:, 2] >= right_safe_z - 0.005
        )

    def _grasp_target_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        left_target = self.socket.data.root_pos_w.clone()
        right_target = self.plug.data.root_pos_w.clone()
        left_target[:, 2] += self.cfg.left_grasp_center_above_object_m
        right_target[:, 2] += self.cfg.right_grasp_center_above_object_m
        return left_target, right_target

    def _advance_evaluation_dwell(self, instantaneous_ok: torch.Tensor, now: float) -> None:
        self._success_since_s = torch.where(
            instantaneous_ok & (self._success_since_s < 0.0),
            torch.full_like(self._success_since_s, now),
            torch.where(
                instantaneous_ok,
                self._success_since_s,
                torch.full_like(self._success_since_s, -1.0),
            ),
        )
        self._evaluation_dwell_s = torch.where(
            instantaneous_ok,
            torch.clamp(now - self._success_since_s, min=0.0),
            torch.zeros_like(self._evaluation_dwell_s),
        )

    def _bound_residual_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape != (self.num_envs, 12) or not torch.isfinite(actions).all():
            raise ValueError("Residual actions must be finite with shape (num_envs, 12)")
        bounded = torch.clamp(actions, -1.0, 1.0).reshape(self.num_envs, 4, 3)
        norms = torch.linalg.vector_norm(bounded, dim=2, keepdim=True).clamp_min(1.0)
        bounded = (bounded / norms).reshape(self.num_envs, 12)
        manipulation = (self._phase >= int(AssemblyPhase.ALIGN)) & (
            self._phase <= int(AssemblyPhase.RETRACT)
        )
        return torch.where(manipulation.unsqueeze(1), bounded, torch.zeros_like(bounded))

    def _filtered_force_magnitude(self, sensor: ContactSensor) -> torch.Tensor:
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError("filtered contact force data is unavailable; verify the sensor object filter")
        return torch.linalg.vector_norm(force_matrix, dim=-1).reshape(self.num_envs, -1).amax(dim=1)

    def _forbidden_contact_magnitude(self) -> torch.Tensor:
        maxima = []
        for sensor in (
            self.forbidden_robot_contact,
            self.forbidden_left_palm_contact,
            self.forbidden_right_palm_contact,
        ):
            net_forces = sensor.data.net_forces_w
            if net_forces is None:
                raise RuntimeError("non-finger robot contact data is unavailable")
            maxima.append(
                torch.linalg.vector_norm(net_forces, dim=-1).reshape(self.num_envs, -1).amax(dim=1)
            )
        for sensor in (
            self.left_contact_l1,
            self.left_contact_l2,
            self.right_contact_r1,
            self.right_contact_r2,
        ):
            net_forces = sensor.data.net_forces_w
            force_matrix = sensor.data.force_matrix_w
            if net_forces is None or force_matrix is None:
                raise RuntimeError("filtered finger contact data is unavailable")
            target_force = force_matrix.sum(dim=2)
            non_target_force = net_forces - target_force
            maxima.append(
                torch.linalg.vector_norm(non_target_force, dim=-1)
                .reshape(self.num_envs, -1)
                .amax(dim=1)
            )
        return torch.stack(maxima, dim=1).amax(dim=1)

    def _identity_quat(self) -> torch.Tensor:
        quat = torch.zeros((self.num_envs, 4), device=self.device)
        quat[:, 0] = 1.0
        return quat

    def _transition_where(
        self,
        mask: torch.Tensor,
        phase: AssemblyPhase,
        now: float,
        reason: TerminationReason = TerminationReason.NONE,
    ) -> None:
        if torch.any(mask):
            self._phase[mask] = int(phase)
            self._phase_started_s[mask] = now
            if phase == AssemblyPhase.ABORT and reason != TerminationReason.NONE:
                self._termination_reason_code[mask] = int(reason)

    def _sim_time_s(self) -> float:
        return float(self.common_step_counter * self.step_dt)

    def _episode_time_limit_reached(self) -> torch.Tensor:
        # Isaac Lab은 _get_dones 전에 완료된 step을 증가시킨다.
        return self.episode_length_buf >= self.max_episode_length

    def _resolve_exact_joints(self, names: list[str]) -> list[int]:
        ids = []
        for name in names:
            found, matched = self.robot.find_joints(name, preserve_order=True)
            if len(found) != 1 or matched != [name]:
                raise RuntimeError(f"required joint not found exactly once: {name}; matched={matched}")
            ids.append(found[0])
        return ids

    def _resolve_exact_body(self, name: str) -> int:
        found, matched = self.robot.find_bodies(name, preserve_order=True)
        if len(found) != 1 or matched != [name]:
            raise RuntimeError(f"required body not found exactly once: {name}; matched={matched}")
        return found[0]
