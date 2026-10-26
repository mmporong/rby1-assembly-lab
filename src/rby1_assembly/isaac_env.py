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
from rby1_assembly.contact_loads import sum_contact_normal_loads
from rby1_assembly.control_math import continuous_goal_dwell, continuous_grasp_loss, damped_joint_step
from rby1_assembly.contracts import ACTION_DIM, OBSERVATION_DIM
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
FORBIDDEN_CONTACT_LOAD_SOURCE = (
    "physx_rigid_contact_view_filtered_pairs_sum_abs_normal_forces"
)
FORBIDDEN_CONTACT_TRACE_SOURCE = (
    "non_finger_body_net_force_and_finger_non_target_residual_proxy"
)
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
    GRIP_CLOSE_TIMEOUT = 8
    LIFT_PROOF_TIMEOUT = 9


@configclass
class Rby1AssemblyEnvCfg(DirectRLEnvCfg):
    """Simulation-design values; these are not calibrated hardware limits."""

    decimation = 4
    episode_length_s = TASK_CONFIG["evaluation"]["timeout_s"]
    action_space = ACTION_DIM
    observation_space = OBSERVATION_DIM
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
        # Preserve the official wheel/roller CollisionGroups, unsupported by
        # PhysX's fast replication path. GPU vector simulation remains enabled.
        replicate_physics=False,
        clone_in_fabric=False,
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
                "gripper_finger_[lr]1": -0.048,
                "gripper_finger_[lr]2": 0.048,
            },
        ),
        # Preserve the authored USD angular drives. USD gains are per degree;
        # PhysX converts them to per-radian values. Overriding with 100/10
        # in Isaac Lab would reduce the gains by about 57.3 times.
        actuators={
            "torso_proximal_hold": ImplicitActuatorCfg(
                joint_names_expr=["torso_[0-2]"],
                effort_limit_sim=841.0,
                stiffness=None,
                damping=None,
            ),
            "torso_distal_hold": ImplicitActuatorCfg(
                joint_names_expr=["torso_[3-5]"],
                effort_limit_sim=369.0,
                stiffness=None,
                damping=None,
            ),
            "arm_shoulder": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_[0-2]", "right_arm_[0-2]"],
                effort_limit_sim=136.5217,
                stiffness=None,
                damping=None,
            ),
            "arm_elbow": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_3", "right_arm_3"],
                effort_limit_sim=78.0,
                stiffness=None,
                damping=None,
            ),
            "arm_wrist_pitch_yaw": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_[45]", "right_arm_[45]"],
                effort_limit_sim=70.0,
                stiffness=None,
                damping=None,
            ),
            "arm_wrist_roll": ImplicitActuatorCfg(
                joint_names_expr=["left_arm_6", "right_arm_6"],
                effort_limit_sim=44.0,
                stiffness=None,
                damping=None,
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
            activate_contact_sensors=True,
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
    insertion_contact: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Plug",
        max_contact_data_count_per_prim=128,
        update_period=0.0,
        history_length=2,
        track_air_time=False,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Socket"],
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
    insertion_integral_gain = 0.30
    gripper_open_magnitude_m = 0.048
    left_gripper_grasp_magnitude_m = 0.024
    right_gripper_grasp_magnitude_m = 0.0165
    grasp_table_clearance_m = 0.010
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
    pregrasp_distance_m = 0.004
    pregrasp_above_height_m = 0.050
    # A bent-elbow waypoint leaves the straight-arm singularity beside the
    # robot before either hand travels across the table edge.
    pregrasp_clearance_joint_pos_rad = (0.7, 0.0, 0.0, -2.1, 0.0, 1.4, 0.0)
    grip_contact_force_n = 1.0
    grip_lift_proof_m = 0.015
    grip_settle_s = 0.20
    grasp_loss_grace_s = 0.20
    phase_timeout_s = 5.0
    align_timeout_s = 10.0
    insert_timeout_s = 30.0
    pregrasp_timeout_s = 20.0
    jam_contact_force_n = 20.0
    hard_contact_force_n = TASK_CONFIG["evaluation"]["max_contact_force_n"]
    assembled_depth_m = TASK_CONFIG["evaluation"]["assembled_depth_m"]
    alignment_standoff_m = 0.008
    insertion_target_margin_m = 0.005
    insertion_speed_m_s = 0.004
    insertion_max_tracking_lead_m = 0.006
    insertion_advance_lateral_tolerance_m = 0.0015
    insertion_advance_orientation_tolerance_rad = 0.020
    insertion_resume_lateral_tolerance_m = 0.0010
    insertion_resume_orientation_tolerance_rad = 0.015
    max_lateral_error_m = TASK_CONFIG["evaluation"]["max_lateral_error_m"]
    max_orientation_error_rad = TASK_CONFIG["evaluation"]["max_orientation_error_rad"]
    success_dwell_s = TASK_CONFIG["evaluation"]["dwell_s"]
    max_retries = 2
    socket_entry_y_m = SOCKET_ENTRY_Y_M
    plug_tip_y_m = PLUG_TIP_Y_M
    assembly_socket_goal_m = (0.37, 0.05, 0.90)
    observation_position_noise_std_m = 0.0015
    observation_rotation_noise_std_rad = 0.017453293
    forbidden_contact_force_n = 1.0


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

        self._actions = torch.zeros((self.num_envs, ACTION_DIM), device=self.device)
        self._phase = torch.full(
            (self.num_envs,), int(AssemblyPhase.PREGRASP), dtype=torch.long, device=self.device
        )
        self._phase_started_s = torch.zeros(self.num_envs, device=self.device)
        self._retry_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._grip_contact_since_s = torch.full((self.num_envs,), -1.0, device=self.device)
        self._grasp_loss_since_s = torch.full((self.num_envs,), -1.0, device=self.device)
        self._insertion_command_depth_m = torch.zeros(self.num_envs, device=self.device)
        self._insertion_advance_enabled = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._success_since_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._evaluation_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._evaluation_dwell_s = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
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
        self._insert_joint_targets = self.robot.data.joint_pos[:, self._arm_joint_ids].clone()
        self._torso_hold_targets = self.robot.data.default_joint_pos[:, self._torso_joint_ids].clone()
        self._last_contact_proxy_n = torch.zeros((self.num_envs, 2), device=self.device)
        self._last_insertion_contact_force_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._last_insertion_contact_force_n = torch.zeros(self.num_envs, device=self.device)
        self._last_insertion_contact_load_n = torch.zeros(self.num_envs, device=self.device)
        self._episode_insertion_contact_load_peak_n = torch.zeros(self.num_envs, device=self.device)
        self._episode_grip_force_peak_n = torch.zeros(self.num_envs, device=self.device)
        self._episode_insertion_contact_force_peak_n = torch.zeros(
            self.num_envs, device=self.device
        )
        self._episode_force_peak_n = torch.zeros(self.num_envs, device=self.device)
        self._last_actual_relative_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self._last_forbidden_contact_load_n = torch.zeros(self.num_envs, device=self.device)
        self._episode_forbidden_contact_load_peak_n = torch.zeros(self.num_envs, device=self.device)
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
        insertion_contact_cfg = self._resolve_env_paths(self.cfg.insertion_contact)
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
        self.insertion_contact = ContactSensor(insertion_contact_cfg)
        self.forbidden_robot_contact = ContactSensor(forbidden_robot_cfg)
        self.forbidden_left_palm_contact = ContactSensor(forbidden_left_palm_cfg)
        self.forbidden_right_palm_contact = ContactSensor(forbidden_right_palm_cfg)

        spawn_ground_plane(prim_path="/World/Ground", cfg=GroundPlaneCfg())
        stage = get_current_stage()
        ground_collision_path = self._ground_collision_prim_path(stage, "/World/Ground")
        self._forbidden_load_sensors = self._create_forbidden_load_sensors(
            stage, ground_collision_path
        )
        if self.cfg.render_background:
            background_cfg = sim_utils.UsdFileCfg(usd_path=self.cfg.background_usd_url)
            background_cfg.func("/World/Background", background_cfg)
            for prim in Usd.PrimRange(stage.GetPrimAtPath("/World/Background")):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
        if self.cfg.scene.replicate_physics:
            self.scene.clone_environments(copy_from_source=False)
        self._disable_redundant_wheel_collision_groups(stage)
        self.scene.filter_collisions(global_prim_paths=["/World/Ground"])
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["socket"] = self.socket
        self.scene.rigid_objects["plug"] = self.plug
        self.scene.sensors["left_contact_l1"] = self.left_contact_l1
        self.scene.sensors["left_contact_l2"] = self.left_contact_l2
        self.scene.sensors["right_contact_r1"] = self.right_contact_r1
        self.scene.sensors["right_contact_r2"] = self.right_contact_r2
        self.scene.sensors["insertion_contact"] = self.insertion_contact
        self.scene.sensors["forbidden_robot_contact"] = self.forbidden_robot_contact
        self.scene.sensors["forbidden_left_palm_contact"] = self.forbidden_left_palm_contact
        self.scene.sensors["forbidden_right_palm_contact"] = self.forbidden_right_palm_contact
        for label, sensor in self._forbidden_load_sensors.items():
            self.scene.sensors[f"forbidden_load_{label.replace('/', '_')}"] = sensor
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.88, 0.92))
        light_cfg.func("/World/Light", light_cfg)

    def _disable_redundant_wheel_collision_groups(self, stage: Usd.Stage) -> None:
        """Keep one collision-group membership per shape in this fixed-base task.

        The official wheel groups filter only members of their own articulation.
        Articulation self-collision is already disabled here. Preserve the source
        USD and disable these redundant groups only in the composed runtime stage.
        """
        if self.cfg.robot.spawn.articulation_props.enabled_self_collisions is not False:
            raise RuntimeError("wheel group suppression requires articulation self-collision disabled")
        group_paths = []
        for env_path in self.scene.env_prim_paths:
            for wheel in ("fl", "fr", "rr", "rl"):
                wheel_path = f"{env_path}/Robot/wheel_{wheel}_link"
                group_path = f"{wheel_path}/WheelCollisionGroup"
                group = UsdPhysics.CollisionGroup.Get(stage, group_path)
                if not group or not group.GetPrim().IsActive():
                    raise RuntimeError(f"required active official wheel group missing: {group_path}")
                includes = group.GetCollidersCollectionAPI().GetIncludesRel().GetTargets()
                filtered = group.GetFilteredGroupsRel().GetTargets()
                if len(includes) != 49 or [str(path) for path in filtered] != [group_path]:
                    raise RuntimeError(f"unexpected official wheel group filter: {group_path}")
                if any(
                    not str(path).startswith(wheel_path + "/")
                    or not stage.GetPrimAtPath(path).HasAPI(UsdPhysics.CollisionAPI)
                    for path in includes
                ):
                    raise RuntimeError(f"wheel group contains unexpected collision members: {group_path}")
                group_paths.append(group_path)
        for group_path in group_paths:
            stage.GetPrimAtPath(group_path).SetActive(False)
        self.runtime_disabled_wheel_collision_groups = group_paths

    def _ground_collision_prim_path(self, stage: Usd.Stage, ground_root_path: str) -> str:
        ground_root = stage.GetPrimAtPath(ground_root_path)
        if not ground_root.IsValid():
            raise RuntimeError(f"ground prim was not spawned: {ground_root_path}")
        collision_planes = [
            prim
            for prim in Usd.PrimRange(ground_root)
            if prim.GetTypeName() == "Plane" and prim.HasAPI(UsdPhysics.CollisionAPI)
        ]
        if len(collision_planes) != 1:
            paths = [str(prim.GetPath()) for prim in collision_planes]
            raise RuntimeError(
                f"expected one collision Plane below {ground_root_path}; found={paths}"
            )
        return str(collision_planes[0].GetPath())

    def _create_forbidden_load_sensors(
        self, stage: Usd.Stage, ground_collision_path: str
    ) -> dict[str, ContactSensor]:
        env0_robot_path = f"{self.scene.env_prim_paths[0]}/Robot"
        robot_root = stage.GetPrimAtPath(env0_robot_path)
        if not robot_root.IsValid():
            raise RuntimeError(f"robot prim was not spawned: {env0_robot_path}")

        link_paths = [
            str(prim.GetPath())
            for prim in Usd.PrimRange(robot_root)
            if prim.GetName().startswith("link_") and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        if not link_paths:
            raise RuntimeError(f"no link_ rigid bodies found below {env0_robot_path}")

        source_specs: list[tuple[str, str, tuple[str, ...]]] = []
        common_filters = (
            "{ENV_REGEX_NS}/Table/top",
            ground_collision_path,
            *self._extra_forbidden_static_filters(),
        )
        robot_filters = (
            *common_filters,
            "{ENV_REGEX_NS}/Socket",
            "{ENV_REGEX_NS}/Plug",
        )
        for source_path in sorted(link_paths):
            relative_path = source_path.removeprefix(f"{env0_robot_path}/")
            source_specs.append(
                (relative_path, f"{{ENV_REGEX_NS}}/Robot/{relative_path}", robot_filters)
            )

        gripper_specs = (
            ("left_gripper/ee_left", robot_filters),
            ("right_gripper/ee_right", robot_filters),
            (
                "left_gripper/ee_finger_l1",
                (*common_filters, "{ENV_REGEX_NS}/Plug"),
            ),
            (
                "left_gripper/ee_finger_l2",
                (*common_filters, "{ENV_REGEX_NS}/Plug"),
            ),
            (
                "right_gripper/ee_finger_r1",
                (*common_filters, "{ENV_REGEX_NS}/Socket"),
            ),
            (
                "right_gripper/ee_finger_r2",
                (*common_filters, "{ENV_REGEX_NS}/Socket"),
            ),
        )
        for relative_path, filters in gripper_specs:
            source_path = f"{env0_robot_path}/{relative_path}"
            source_prim = stage.GetPrimAtPath(source_path)
            if not source_prim.IsValid() or not source_prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(f"forbidden-contact source is not a rigid body: {source_path}")
            source_specs.append(
                (relative_path, f"{{ENV_REGEX_NS}}/Robot/{relative_path}", filters)
            )

        sensors: dict[str, ContactSensor] = {}
        for label, prim_path, filters in source_specs:
            sensor_cfg = ContactSensorCfg(
                prim_path=prim_path,
                max_contact_data_count_per_prim=128,
                update_period=0.0,
                history_length=0,
                track_air_time=False,
                filter_prim_paths_expr=list(filters),
            )
            sensors[label] = ContactSensor(self._resolve_env_paths(sensor_cfg))
        return sensors

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
        residual_actions = self._residual_actions_world(self._actions)
        left_twist, right_twist = self._desired_ee_twists()
        baseline_left_twist, baseline_right_twist = left_twist.clone(), right_twist.clone()
        left_twist[:, :3] += residual_actions[:, :3] * self.cfg.residual_translation_bound_m
        left_twist[:, 3:] += residual_actions[:, 3:6] * self.cfg.residual_rotation_bound_rad
        right_twist[:, :3] += residual_actions[:, 6:9] * self.cfg.residual_translation_bound_m
        right_twist[:, 3:] += residual_actions[:, 9:12] * self.cfg.residual_rotation_bound_rad

        jacobians = self.robot.root_physx_view.get_jacobians()
        left_jacobian = jacobians[:, self._left_ee_jacobian_id, :, self._left_arm_joint_ids]
        right_jacobian = jacobians[:, self._right_ee_jacobian_id, :, self._right_arm_joint_ids]
        left_delta = self._damped_least_squares(left_jacobian, left_twist)
        right_delta = self._damped_least_squares(right_jacobian, right_twist)

        current = self.robot.data.joint_pos
        lifting_beside_base = (
            (self._phase == int(AssemblyPhase.PREGRASP)) & ~self._pregrasp_lift_reached
        )
        clearance_q = self._pregrasp_clearance_joint_targets()
        for joint_ids, delta in ((self._left_arm_joint_ids, left_delta),
                                 (self._right_arm_joint_ids, right_delta)):
            clearance_delta = torch.clamp(
                clearance_q - current[:, joint_ids],
                -self.cfg.max_joint_delta_rad, self.cfg.max_joint_delta_rad,
            )
            delta[lifting_beside_base] = clearance_delta[lifting_beside_base]
        self._joint_targets[:] = current
        self._joint_targets[:, self._left_arm_joint_ids] = current[:, self._left_arm_joint_ids] + left_delta
        self._joint_targets[:, self._right_arm_joint_ids] = current[:, self._right_arm_joint_ids] + right_delta
        insertion = self._phase == int(AssemblyPhase.INSERT)
        if torch.any(insertion):
            arm_positions = current[:, self._arm_joint_ids]
            stiffness = self.robot.data.joint_stiffness[:, self._arm_joint_ids]
            gravity = self.robot.root_physx_view.get_gravity_compensation_forces()[:, self._arm_joint_ids]
            effort_limits = self.robot.data.joint_effort_limits[:, self._arm_joint_ids]
            if (
                not torch.isfinite(stiffness).all() or torch.any(stiffness <= 0)
                or not torch.isfinite(gravity).all() or not torch.isfinite(effort_limits).all()
                or torch.any(effort_limits <= 0)
            ):
                raise RuntimeError("insertion integral requires finite positive drive limits and stiffness")
            baseline_delta = torch.cat(
                (
                    self._damped_least_squares(left_jacobian, baseline_left_twist),
                    self._damped_least_squares(right_jacobian, baseline_right_twist),
                ), dim=1,
            )
            residual_delta = torch.cat(
                (
                    self._damped_least_squares(left_jacobian, left_twist - baseline_left_twist),
                    self._damped_least_squares(right_jacobian, right_twist - baseline_right_twist),
                ), dim=1,
            )
            # Bound the stored position-drive demand after gravity bias. The
            # physical implicit drive still applies its own total force cap.
            position_headroom = ((effort_limits - gravity.abs()).clamp_min(0.0) / stiffness)
            position_headroom = position_headroom.clamp(max=self.cfg.max_joint_delta_rad)
            integral_targets = self._insert_joint_targets + self.cfg.insertion_integral_gain * baseline_delta
            integral_targets = torch.maximum(
                torch.minimum(integral_targets, arm_positions + position_headroom),
                arm_positions - position_headroom,
            )
            arm_limits = self.robot.data.soft_joint_pos_limits[:, self._arm_joint_ids]
            integral_targets = torch.maximum(
                torch.minimum(integral_targets, arm_limits[:, :, 1]), arm_limits[:, :, 0]
            )
            self._insert_joint_targets[insertion] = integral_targets[insertion]
            # Policy corrections are one-step offsets, never accumulated into
            # the baseline integral state.
            corrected_targets = torch.maximum(
                torch.minimum(integral_targets + residual_delta, arm_positions + position_headroom),
                arm_positions - position_headroom,
            )
            corrected_targets = torch.maximum(
                torch.minimum(corrected_targets, arm_limits[:, :, 1]), arm_limits[:, :, 0]
            )
            arm_targets = self._joint_targets[:, self._arm_joint_ids]
            arm_targets[insertion] = corrected_targets[insertion]
            self._joint_targets[:, self._arm_joint_ids] = arm_targets
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
        # Convert model gravity feedforward into an implicit-drive position
        # bias. There is no extra effort command outside the capped drive.
        controlled = self._torso_joint_ids + self._arm_joint_ids
        gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
        stiffness = self.robot.data.joint_stiffness[:, controlled]
        if torch.any(stiffness <= 0):
            raise RuntimeError("gravity position bias requires positive joint stiffness")
        targets = self._joint_targets.clone()
        targets[:, controlled] += gravity[:, controlled] / stiffness
        limits = self.robot.data.soft_joint_pos_limits
        targets = torch.maximum(torch.minimum(targets, limits[:, :, 1]), limits[:, :, 0])
        self.robot.set_joint_position_target(targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        relative_pos, relative_quat = self._control_relative_object_pose()
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
                self._last_insertion_contact_load_n.unsqueeze(1),
                self._grip_proven.float(),
                phase_one_hot,
                self._actions,
                self._insertion_command_depth_m.unsqueeze(1),
                self._insertion_advance_enabled.float().unsqueeze(1),
                torch.where(
                    (self._phase == int(AssemblyPhase.INSERT)).unsqueeze(1),
                    self._insert_joint_targets - arm_pos, torch.zeros_like(arm_pos),
                ),
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
        current_contact_force = torch.maximum(
            self._last_contact_proxy_n.max(dim=1).values,
            self._last_insertion_contact_load_n,
        )
        force_penalty = torch.clamp(
            current_contact_force / self.cfg.hard_contact_force_n, 0.0, 2.0
        )
        action_penalty = torch.sum(self._actions.square(), dim=1)
        success = self._evaluation_success
        failed = ((self._phase == int(AssemblyPhase.ABORT)) | self._episode_time_limit_reached()) & ~success
        return (
            -4.0 * lateral
            - 0.4 * angle
            + 2.0 * torch.clamp(
                depth, min=0.0,
                max=self.cfg.assembled_depth_m + self.cfg.insertion_target_margin_m,
            )
            + 10.0 * success.float()
            - 2.0 * failed.float()
            - 0.01
            - 0.05 * force_penalty
            - 0.002 * action_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        evaluation_eligible = self._evaluation_eligibility_mask()
        if evaluation_eligible.shape != (self.num_envs,) or evaluation_eligible.dtype != torch.bool:
            raise RuntimeError("evaluation eligibility must be a boolean mask per environment")
        left_l1_force = self._filtered_force_magnitude(self.left_contact_l1)
        left_l2_force = self._filtered_force_magnitude(self.left_contact_l2)
        right_r1_force = self._filtered_force_magnitude(self.right_contact_r1)
        right_r2_force = self._filtered_force_magnitude(self.right_contact_r2)
        left_force = torch.maximum(left_l1_force, left_l2_force)
        right_force = torch.maximum(right_r1_force, right_r2_force)
        self._last_contact_proxy_n[:, 0] = left_force
        self._last_contact_proxy_n[:, 1] = right_force
        insertion_force_w = self._filtered_force_vector(self.insertion_contact)
        insertion_force = torch.linalg.vector_norm(insertion_force_w, dim=1)
        self._last_insertion_contact_force_w[:] = insertion_force_w
        self._last_insertion_contact_force_n[:] = insertion_force
        insertion_load = self._insertion_contact_load()
        self._last_insertion_contact_load_n[:] = insertion_load
        current_grip_force = torch.maximum(left_force, right_force)
        self._episode_grip_force_peak_n = torch.maximum(
            self._episode_grip_force_peak_n, current_grip_force
        )
        self._episode_insertion_contact_force_peak_n = torch.maximum(
            self._episode_insertion_contact_force_peak_n, insertion_force
        )
        self._episode_insertion_contact_load_peak_n = torch.maximum(
            self._episode_insertion_contact_load_peak_n, insertion_load
        )
        self._episode_force_peak_n = torch.maximum(
            self._episode_grip_force_peak_n,
            self._episode_insertion_contact_load_peak_n,
        )

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
        forbidden_contact_load = self._forbidden_contact_load()
        self._last_forbidden_contact_load_n[:] = forbidden_contact_load
        self._episode_forbidden_contact_load_peak_n = torch.maximum(
            self._episode_forbidden_contact_load_peak_n, forbidden_contact_load
        )
        independent_eval_valid = torch.isfinite(forbidden_contact_load)
        forbidden_now = (~independent_eval_valid) | (
            forbidden_contact_load > self.cfg.forbidden_contact_force_n
        )
        self._forbidden_contact_seen |= forbidden_now
        grip_hard_overload = self._episode_grip_force_peak_n > self.cfg.hard_contact_force_n
        insertion_hard_overload = (
            self._episode_insertion_contact_load_peak_n > self.cfg.hard_contact_force_n
        )
        hard_overload = grip_hard_overload | insertion_hard_overload
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
        self._advance_evaluation_dwell(
            instantaneous_ok & independent_eval_valid & evaluation_eligible
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
        self.extras["contact_force_source"] = (
            "filtered_finger_object_and_plug_socket_contact_sensors"
        )
        self.extras["insertion_contact_force_source"] = (
            "plug_body_filtered_against_socket_net_normal_force_world_frame"
        )
        self.extras["insertion_contact_load_source"] = "physx_plug_socket_sum_abs_contact_normals"
        self.extras["forbidden_contact_load_source"] = FORBIDDEN_CONTACT_LOAD_SOURCE
        self.extras["forbidden_contact_load_n"] = forbidden_contact_load.clone()
        self.extras["forbidden_contact_load_peak_n"] = self._episode_forbidden_contact_load_peak_n.clone()
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
            "contact_force_proxy_peak_format": (
                "max_of_grip_and_plug_socket_sum_abs_contact_normals_n"
            ),
            "grip_contact_force_peak_n": self._episode_grip_force_peak_n.clone(),
            "insertion_contact_force_n": self._last_insertion_contact_force_n.clone(),
            "insertion_contact_load_n": self._last_insertion_contact_load_n.clone(),
            "insertion_contact_load_peak_n": self._episode_insertion_contact_load_peak_n.clone(),
            "insertion_contact_load_source": "physx_plug_socket_sum_abs_contact_normals",
            "insertion_contact_force_peak_n": (
                self._episode_insertion_contact_force_peak_n.clone()
            ),
            "forbidden_contact_load_n": forbidden_contact_load.clone(),
            "forbidden_contact_load_peak_n": self._episode_forbidden_contact_load_peak_n.clone(),
            "forbidden_contact_load_source": FORBIDDEN_CONTACT_LOAD_SOURCE,
            "forbidden_contact": self._forbidden_contact_seen.clone(),
            "grasp_both": handling_ok.clone(),
            "completion_source": "simulator_ground_truth_evaluator",
        }
        evaluation_snapshot = {
            "actual_depth_m": depth.clone(),
            "actual_lateral_error_m": lateral.clone(),
            "actual_orientation_error_rad": angle.clone(),
            "contact_force_proxy_peak_n": self._episode_force_peak_n.clone(),
            "contact_force_proxy_peak_format": (
                "max_of_grip_and_plug_socket_sum_abs_contact_normals_n"
            ),
            "grip_contact_force_peak_n": self._episode_grip_force_peak_n.clone(),
            "insertion_contact_force_n": self._last_insertion_contact_force_n.clone(),
            "insertion_contact_load_n": self._last_insertion_contact_load_n.clone(),
            "insertion_contact_load_peak_n": self._episode_insertion_contact_load_peak_n.clone(),
            "insertion_contact_load_source": "physx_plug_socket_sum_abs_contact_normals",
            "insertion_contact_force_peak_n": (
                self._episode_insertion_contact_force_peak_n.clone()
            ),
            "forbidden_contact_load_n": forbidden_contact_load.clone(),
            "forbidden_contact_load_peak_n": self._episode_forbidden_contact_load_peak_n.clone(),
            "forbidden_contact_load_source": FORBIDDEN_CONTACT_LOAD_SOURCE,
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
            "ee_poses_world_wxyz": torch.cat((
                self.robot.data.body_pos_w[:, [self._left_ee_body_id, self._right_ee_body_id]],
                self.robot.data.body_quat_w[:, [self._left_ee_body_id, self._right_ee_body_id]],
            ), dim=2).clone(),
            "forbidden_robot_body_names": tuple(self.forbidden_robot_contact.body_names),
            "forbidden_robot_forces_w": self.forbidden_robot_contact.data.net_forces_w.clone(),
            "forbidden_palm_forces_w": torch.stack((
                self.forbidden_left_palm_contact.data.net_forces_w[:, 0],
                self.forbidden_right_palm_contact.data.net_forces_w[:, 0],
            ), dim=1).clone(),
            "finger_contact_force_n": torch.stack(
                (left_l1_force, left_l2_force, right_r1_force, right_r2_force), dim=1
            ).clone(),
            "finger_non_target_force_w": torch.stack([
                (sensor.data.net_forces_w - sensor.data.force_matrix_w.sum(dim=2))[:, 0]
                for sensor in (self.left_contact_l1, self.left_contact_l2,
                               self.right_contact_r1, self.right_contact_r2)
            ], dim=1).clone(),
            "object_order": ("socket", "plug"),
            "object_poses_world_wxyz": object_poses.clone(),
            "contact_force_proxy_n": self._last_contact_proxy_n.clone(),
            "contact_force_proxy_format": "per_hand_peak_filtered_finger_magnitude_n",
            "grip_contact_force_peak_n": self._episode_grip_force_peak_n.clone(),
            "insertion_contact_force_n": self._last_insertion_contact_force_n.clone(),
            "insertion_contact_load_n": self._last_insertion_contact_load_n.clone(),
            "insertion_contact_load_peak_n": self._episode_insertion_contact_load_peak_n.clone(),
            "insertion_contact_load_source": "physx_plug_socket_sum_abs_contact_normals",
            "insertion_contact_force_w": self._last_insertion_contact_force_w.clone(),
            "insertion_contact_force_peak_n": (
                self._episode_insertion_contact_force_peak_n.clone()
            ),
            "insertion_contact_force_source": (
                "plug_body_filtered_against_socket_net_normal_force_world_frame"
            ),
            "phase": self._phase.clone(),
            "phase_name_by_code": {int(phase): phase.name.lower() for phase in AssemblyPhase},
            "grip_proven": self._evaluation_grip_proven.clone(),
            "controller_grip_proven": self._grip_proven.clone(),
            "continuous_grasp_loss_since_s": self._grasp_loss_since_s.clone(),
            "insertion_command_depth_m": self._insertion_command_depth_m.clone(),
            "insertion_advance_enabled": self._insertion_advance_enabled.clone(),
            "insert_joint_target_error_rad": (
                self._insert_joint_targets - self.robot.data.joint_pos[:, self._arm_joint_ids]
            ).clone(),
            "grip_live": grip_live.clone(),
            "actual_depth_m": depth.clone(),
            "actual_lateral_error_m": lateral.clone(),
            "actual_orientation_error_rad": angle.clone(),
            "contact_force_proxy_peak_n": self._episode_force_peak_n.clone(),
            "contact_force_proxy_peak_format": (
                "max_of_grip_and_plug_socket_sum_abs_contact_normals_n"
            ),
            "forbidden_contact": self._forbidden_contact_seen.clone(),
            "forbidden_contact_load_n": forbidden_contact_load.clone(),
            "forbidden_contact_load_peak_n": self._episode_forbidden_contact_load_peak_n.clone(),
            "forbidden_contact_load_source": FORBIDDEN_CONTACT_LOAD_SOURCE,
            "forbidden_contact_measured": independent_eval_valid.clone(),
            "forbidden_contact_source": FORBIDDEN_CONTACT_LOAD_SOURCE,
            "forbidden_contact_trace_source": FORBIDDEN_CONTACT_TRACE_SOURCE,
            "self_collision_enabled": False,
            "hard_overload": hard_overload.clone(),
            "grip_hard_overload": grip_hard_overload.clone(),
            "insertion_hard_overload": insertion_hard_overload.clone(),
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

        joint_pos, joint_vel = self._reset_physical_state(env_ids)

        self._joint_targets[env_ids] = joint_pos
        self._insert_joint_targets[env_ids] = joint_pos[:, self._arm_joint_ids]
        self._torso_hold_targets[env_ids] = joint_pos[:, self._torso_joint_ids]
        self._phase[env_ids] = int(AssemblyPhase.PREGRASP)
        self._phase_started_s[env_ids] = self._sim_time_s()
        self._retry_count[env_ids] = 0
        self._success_since_step[env_ids] = -1
        self._grip_contact_since_s[env_ids] = -1.0
        self._grasp_loss_since_s[env_ids] = -1.0
        self._insertion_command_depth_m[env_ids] = 0.0
        self._insertion_advance_enabled[env_ids] = False
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
        self._episode_grip_force_peak_n[env_ids] = 0.0
        self._episode_insertion_contact_force_peak_n[env_ids] = 0.0
        self._episode_force_peak_n[env_ids] = 0.0
        self._last_forbidden_contact_load_n[env_ids] = 0.0
        self._episode_forbidden_contact_load_peak_n[env_ids] = 0.0
        self._forbidden_contact_seen[env_ids] = False
        self._last_contact_proxy_n[env_ids] = 0.0
        self._last_insertion_contact_force_w[env_ids] = 0.0
        self._last_insertion_contact_force_n[env_ids] = 0.0
        self._last_insertion_contact_load_n[env_ids] = 0.0
        self._episode_insertion_contact_load_peak_n[env_ids] = 0.0
        self._last_actual_relative_pose[env_ids] = 0.0
        self._actions[env_ids] = 0.0
        self._object_start_z[env_ids, 0] = self.socket.data.default_root_state[env_ids, 2]
        self._object_start_z[env_ids, 1] = self.plug.data.default_root_state[env_ids, 2]

    def _reset_physical_state(self, env_ids):
        """Write one reset state and return the joint state used by controllers."""

        robot_root = self.robot.data.default_root_state[env_ids].clone()
        robot_root[:, :3] += self.scene.env_origins[env_ids]
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        self.robot.write_root_pose_to_sim(robot_root[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(robot_root[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._reset_rigid_object(self.socket, env_ids, y_noise=0.010)
        self._reset_rigid_object(self.plug, env_ids, y_noise=0.010)
        return joint_pos, joint_vel

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
        insertion_force_w = self._filtered_force_vector(self.insertion_contact)
        insertion_force = torch.linalg.vector_norm(insertion_force_w, dim=1)
        self._last_insertion_contact_force_w[:] = insertion_force_w
        self._last_insertion_contact_force_n[:] = insertion_force
        insertion_load = self._insertion_contact_load()
        self._last_insertion_contact_load_n[:] = insertion_load

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
            & (right_distance < self.cfg.pregrasp_distance_m)
            & self._pregrasp_ready_mask(),
            AssemblyPhase.CLOSE,
            now,
        )
        closed_contact = (self._phase == int(AssemblyPhase.CLOSE)) & left_contact & right_contact
        self._grip_contact_since_s = torch.where(
            closed_contact,
            torch.where(self._grip_contact_since_s < 0, now, self._grip_contact_since_s),
            -1.0,
        )
        self._transition_where(
            closed_contact & ((now - self._grip_contact_since_s) >= self.cfg.grip_settle_s),
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

        relative_pos, relative_quat = self._control_relative_object_pose()
        lateral = torch.linalg.vector_norm(relative_pos[:, (0, 2)], dim=1)
        angle = quat_error_magnitude(relative_quat, self._identity_quat())
        aligned = (
            (self._phase == int(AssemblyPhase.ALIGN))
            & (relative_pos[:, 1] > -0.012)
            & (lateral < self.cfg.max_lateral_error_m * 1.5)
            & (angle < self.cfg.max_orientation_error_rad * 1.5)
        )
        self._transition_where(aligned, AssemblyPhase.INSERT, now)

        current_grip_force = torch.maximum(left_force, right_force)
        continuous_grasp = left_contact & right_contact & (lift_left >= self.cfg.grip_lift_proof_m) & (
            lift_right >= self.cfg.grip_lift_proof_m
        )

        hard_overload = (current_grip_force > self.cfg.hard_contact_force_n) | (
            insertion_load > self.cfg.hard_contact_force_n
        )
        self._transition_where(
            hard_overload, AssemblyPhase.ABORT, now, TerminationReason.HARD_FORCE_LIMIT
        )
        jam = (
            (self._phase == int(AssemblyPhase.INSERT))
            & (insertion_load > self.cfg.jam_contact_force_n)
            & ~hard_overload
        )
        self._start_recovery(jam, now)
        grasp_monitor_active = (
            (self._phase >= int(AssemblyPhase.ALIGN))
            & (self._phase <= int(AssemblyPhase.RETRACT))
        )
        self._grasp_loss_since_s, lost_grasp = continuous_grasp_loss(
            grasp_monitor_active, continuous_grasp, self._grasp_loss_since_s,
            now, self.cfg.grasp_loss_grace_s,
        )
        self._transition_where(lost_grasp, AssemblyPhase.ABORT, now, TerminationReason.GRASP_LOST)
        retract_done = (self._phase == int(AssemblyPhase.RETRACT)) & (relative_pos[:, 1] <= -0.015)
        self._transition_where(retract_done, AssemblyPhase.ALIGN, now)

        phase_timeout = torch.where(
            self._phase == int(AssemblyPhase.ALIGN),
            self.cfg.align_timeout_s,
            torch.where(
                self._phase == int(AssemblyPhase.INSERT),
                self.cfg.insert_timeout_s, self.cfg.phase_timeout_s,
            ),
        )
        timed_out = (now - self._phase_started_s) >= phase_timeout
        recoverable = timed_out & (
            (self._phase == int(AssemblyPhase.ALIGN)) | (self._phase == int(AssemblyPhase.INSERT))
        )
        self._start_recovery(recoverable, now)
        preparation_timeout = torch.where(
            self._phase == int(AssemblyPhase.PREGRASP),
            self.cfg.pregrasp_timeout_s, self.cfg.phase_timeout_s,
        )
        pregrasp_failed = ((now - self._phase_started_s) >= preparation_timeout) & (
            self._phase <= int(AssemblyPhase.CHECK_LIFT)
        )
        for phase, reason in (
            (AssemblyPhase.PREGRASP, TerminationReason.PREGRASP_TIMEOUT),
            (AssemblyPhase.CLOSE, TerminationReason.GRIP_CLOSE_TIMEOUT),
            (AssemblyPhase.CHECK_LIFT, TerminationReason.LIFT_PROOF_TIMEOUT),
        ):
            self._transition_where(
                pregrasp_failed & (self._phase == int(phase)),
                AssemblyPhase.ABORT, now, reason,
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
        grasp_center_offset = torch.tensor(GRASP_CENTER_IN_WRIST_M, device=self.device).repeat(self.num_envs, 1)
        (
            left_grasp_target,
            left_grasp_quat,
            right_grasp_target,
            right_grasp_quat,
        ) = self._grasp_target_poses()
        left_target = left_grasp_target - quat_apply(left_grasp_quat, grasp_center_offset)
        right_target = right_grasp_target - quat_apply(right_grasp_quat, grasp_center_offset)
        left_target_quat = left_grasp_quat.clone()
        right_target_quat = right_grasp_quat.clone()

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
        left_target[pregrasp_above, 2] += self.cfg.pregrasp_above_height_m
        right_target[pregrasp_above, 2] += self.cfg.pregrasp_above_height_m
        check_lift = self._phase == int(AssemblyPhase.CHECK_LIFT)
        left_check_lift, right_check_lift = self._check_lift_target_positions(
            left_target,
            right_target,
            left_target_quat,
            right_target_quat,
        )
        left_target[check_lift] = left_check_lift[check_lift]
        right_target[check_lift] = right_check_lift[check_lift]

        manipulation = (self._phase >= int(AssemblyPhase.ALIGN)) & (
            self._phase <= int(AssemblyPhase.RETRACT)
        )
        socket_goal = self.scene.env_origins + torch.tensor(self.cfg.assembly_socket_goal_m, device=self.device)
        socket_goal_quat = torch.tensor(
            getattr(self.cfg, "assembly_socket_goal_quat_wxyz", (1.0, 0.0, 0.0, 0.0)),
            device=self.device,
        ).repeat(self.num_envs, 1)
        plug_goal, plug_goal_quat = self._plug_pose_goal(
            socket_goal, socket_goal_quat, -self.cfg.alignment_standoff_m
        )
        insert = self._phase == int(AssemblyPhase.INSERT)
        # The commanded depth includes tracking margin; the independent
        # evaluator still requires the original measured insertion depth.
        relative_position, relative_orientation = self._control_relative_object_pose()
        lateral_error = torch.linalg.vector_norm(relative_position[:, (0, 2)], dim=1)
        orientation_error = quat_error_magnitude(relative_orientation, self._identity_quat())
        pause_advance = (
            (lateral_error >= self.cfg.insertion_advance_lateral_tolerance_m)
            | (orientation_error >= self.cfg.insertion_advance_orientation_tolerance_rad)
        )
        resume_advance = (
            (lateral_error <= self.cfg.insertion_resume_lateral_tolerance_m)
            & (orientation_error <= self.cfg.insertion_resume_orientation_tolerance_rad)
        )
        self._insertion_advance_enabled = insert & (
            (self._insertion_advance_enabled & ~pause_advance) | resume_advance
        )
        # This reference ramp is a baseline servo heuristic. Policy residuals
        # are applied later; measured force/contact limits remain independent.
        next_depth = self._insertion_command_depth_m + (
            self._insertion_advance_enabled.float() * self.cfg.insertion_speed_m_s * self.step_dt
        )
        next_depth = torch.minimum(
            next_depth, relative_position[:, 1] + self.cfg.insertion_max_tracking_lead_m
        ).clamp(max=self.cfg.assembled_depth_m + self.cfg.insertion_target_margin_m)
        self._insertion_command_depth_m[insert] = next_depth[insert]
        # During insertion the measured support object defines the mating
        # frame, so support-arm tracking error is not added to the plug error.
        control_object_poses = self._control_object_poses()
        measured_mating_goal, measured_mating_quat = self._plug_pose_goal(
            control_object_poses[0], control_object_poses[1], next_depth
        )
        plug_goal[insert] = measured_mating_goal[insert]
        plug_goal_quat[insert] = measured_mating_quat[insert]
        retract = self._phase == int(AssemblyPhase.RETRACT)
        retract_goal, retract_goal_quat = self._plug_pose_goal(
            socket_goal, socket_goal_quat, -0.030
        )
        plug_goal[retract] = retract_goal[retract]
        plug_goal_quat[retract] = retract_goal_quat[retract]
        # Re-estimate the grasp transform from measured simulator poses so
        # millimetre-scale settling/slip is corrected at the object, rather
        # than treating the lift-proof transform as a rigid attachment.
        (
            left_grasp_pos,
            left_grasp_quat,
            right_grasp_pos,
            right_grasp_quat,
        ) = self._control_grasp_transforms(left_pos, left_quat, right_pos, right_quat)
        left_goal, left_goal_quat = combine_frame_transforms(
            socket_goal, socket_goal_quat, left_grasp_pos, left_grasp_quat,
        )
        right_goal, right_goal_quat = combine_frame_transforms(
            plug_goal, plug_goal_quat, right_grasp_pos, right_grasp_quat,
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
        return damped_joint_step(
            jacobian, twist, self.cfg.dls_damping, self.cfg.max_joint_delta_rad
        )

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

    def _control_relative_object_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Pose used by the controller; evaluation continues to use simulator ground truth."""

        return self._relative_object_pose()

    def _control_object_poses(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Object poses used by the controller, overridable by state estimators."""

        return (
            self.socket.data.root_pos_w,
            self.socket.data.root_quat_w,
            self.plug.data.root_pos_w,
            self.plug.data.root_quat_w,
        )

    def _control_grasp_transforms(
        self,
        left_ee_pos: torch.Tensor,
        left_ee_quat: torch.Tensor,
        right_ee_pos: torch.Tensor,
        right_ee_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return object-to-hand transforms used by the Cartesian controller."""

        socket_pos, socket_quat, plug_pos, plug_quat = self._control_object_poses()
        left_grasp_pos, left_grasp_quat = subtract_frame_transforms(
            socket_pos, socket_quat, left_ee_pos, left_ee_quat
        )
        right_grasp_pos, right_grasp_quat = subtract_frame_transforms(
            plug_pos, plug_quat, right_ee_pos, right_ee_quat
        )
        return left_grasp_pos, left_grasp_quat, right_grasp_pos, right_grasp_quat

    def _plug_center_goal(
        self, socket_center_w: torch.Tensor, depth_m: float | torch.Tensor
    ) -> torch.Tensor:
        socket_quat = self._identity_quat()[: socket_center_w.shape[0]]
        goal, _ = self._plug_pose_goal(socket_center_w, socket_quat, depth_m)
        return goal

    def _plug_pose_goal(
        self,
        socket_center_w: torch.Tensor,
        socket_quat_w: torch.Tensor,
        depth_m: float | torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offset = torch.zeros_like(socket_center_w)
        offset[:, 1] = self.cfg.socket_entry_y_m + depth_m - self.cfg.plug_tip_y_m
        return socket_center_w + quat_apply(socket_quat_w, offset), socket_quat_w.clone()

    def _pregrasp_above_mask(
        self, left_grasp_center: torch.Tensor, right_grasp_center: torch.Tensor
    ) -> torch.Tensor:
        left_above, right_above = self._grasp_target_positions()
        left_above[:, 2] += self.cfg.pregrasp_above_height_m
        right_above[:, 2] += self.cfg.pregrasp_above_height_m
        return (
            (torch.linalg.vector_norm(left_grasp_center - left_above, dim=1) < 0.005)
            & (torch.linalg.vector_norm(right_grasp_center - right_above, dim=1) < 0.005)
        )

    def _extra_forbidden_static_filters(self) -> tuple[str, ...]:
        return ()

    def _pregrasp_clearance_joint_targets(self) -> torch.Tensor:
        return torch.tensor(
            self.cfg.pregrasp_clearance_joint_pos_rad, device=self.device
        ).repeat(self.num_envs, 1)

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

    def _grasp_target_poses(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        left_target, right_target = self._grasp_target_positions()
        identity = self._identity_quat()
        return left_target, identity, right_target, identity.clone()

    def _check_lift_target_positions(
        self,
        left_target: torch.Tensor,
        right_target: torch.Tensor,
        left_target_quat: torch.Tensor,
        right_target_quat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del left_target_quat, right_target_quat
        left_lift = left_target.clone()
        right_lift = right_target.clone()
        left_lift[:, 2] = (
            self.scene.env_origins[:, 2]
            + self._object_start_z[:, 0]
            + 0.050
            + self.cfg.left_grasp_center_above_object_m
            - GRASP_CENTER_IN_WRIST_M[2]
        )
        right_lift[:, 2] = (
            self.scene.env_origins[:, 2]
            + self._object_start_z[:, 1]
            + 0.050
            + self.cfg.right_grasp_center_above_object_m
            - GRASP_CENTER_IN_WRIST_M[2]
        )
        return left_lift, right_lift

    def _pregrasp_ready_mask(self) -> torch.Tensor:
        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _residual_actions_world(self, actions: torch.Tensor) -> torch.Tensor:
        """Map task-frame residuals to world axes; the legacy task is already world-aligned."""

        return actions

    def _advance_evaluation_dwell(self, instantaneous_ok: torch.Tensor) -> None:
        (
            self._success_since_step,
            self._evaluation_dwell_s,
            self._evaluation_success,
        ) = continuous_goal_dwell(
            instantaneous_ok,
            self.episode_length_buf,
            self._success_since_step,
            step_dt_s=self.step_dt,
            required_dwell_s=self.cfg.success_dwell_s,
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

    def _insertion_contact_load(self) -> torch.Tensor:
        forces, _, _, _, counts, starts = self.insertion_contact.contact_physx_view.get_contact_data(
            dt=self.cfg.sim.dt
        )
        return sum_contact_normal_loads(forces, counts, starts, self.num_envs)

    def _filtered_force_magnitude(self, sensor: ContactSensor) -> torch.Tensor:
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError("filtered contact force data is unavailable; verify the sensor object filter")
        return torch.linalg.vector_norm(force_matrix, dim=-1).reshape(self.num_envs, -1).amax(dim=1)

    def _filtered_force_vector(self, sensor: ContactSensor) -> torch.Tensor:
        force_matrix = sensor.data.force_matrix_w
        if force_matrix is None:
            raise RuntimeError("filtered contact force data is unavailable; verify the sensor object filter")
        return force_matrix.reshape(self.num_envs, -1, 3).sum(dim=1)

    def _forbidden_contact_load(self) -> torch.Tensor:
        per_source_loads = []
        for label, sensor in self._forbidden_load_sensors.items():
            forces, _, _, _, counts, starts = sensor.contact_physx_view.get_contact_data(
                dt=self.cfg.sim.dt
            )
            try:
                load = sum_contact_normal_loads(forces, counts, starts, self.num_envs)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"invalid forbidden-contact buffer for source {label}: {exc}"
                ) from exc
            per_source_loads.append(load)
        if not per_source_loads:
            raise RuntimeError("forbidden-contact load sensors are unavailable")
        total_load = torch.stack(per_source_loads, dim=1).sum(dim=1)
        if not torch.isfinite(total_load).all():
            raise RuntimeError("forbidden-contact load aggregation produced a non-finite value")
        return total_load

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

    def _evaluation_eligibility_mask(self) -> torch.Tensor:
        """Select environments whose ground-truth completion evaluator may advance."""

        return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def _transition_where(
        self,
        mask: torch.Tensor,
        phase: AssemblyPhase,
        now: float,
        reason: TerminationReason = TerminationReason.NONE,
    ) -> None:
        if torch.any(mask):
            self._insert_joint_targets[mask] = self.robot.data.joint_pos[mask][:, self._arm_joint_ids]
            if phase == AssemblyPhase.INSERT:
                relative_position, _ = self._control_relative_object_pose()
                self._insertion_command_depth_m[mask] = relative_position[mask, 1]
                self._insertion_advance_enabled[mask] = False
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
