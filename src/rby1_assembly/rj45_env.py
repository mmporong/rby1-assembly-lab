"""Isaac runtime adapter for a validated RJ45 task configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rby1_assembly.rj45_task import (
    load_rj45_task,
    quaternion_rotate_wxyz,
    verify_generated_asset_hashes,
)


def _project_asset(project: Path, task: dict[str, Any], name: str) -> Path:
    project = project.resolve()
    path = (project / task["assets"][name]).resolve()
    if project not in path.parents or not path.is_file():
        raise FileNotFoundError(f"missing project-local RJ45 asset: {path}")
    return path


def advance_joint_reference(current, start, goal, progress, enabled, max_delta, tracking_tolerance):
    """Advance a stored joint-path reference only while tracking is bounded."""
    import torch

    if any(not torch.isfinite(value).all() for value in (current, start, goal, progress)):
        raise RuntimeError("joint-path state and references must be finite")
    travel = goal - start
    reference = start + progress[:, None] * travel
    tracking = (current - reference).abs().amax(1) <= tracking_tolerance
    step = max_delta / travel.abs().amax().clamp_min(1.0e-6)
    next_progress = (progress + (enabled & tracking).float() * step).clamp(max=1)
    return next_progress, start + next_progress[:, None] * travel


def nullspace_posture_step(jacobian, current, preferred, gain, max_step):
    """Hold the redundant arm posture without adding a first-order hand twist."""
    import torch

    if any(not torch.isfinite(value).all() for value in (jacobian, current, preferred)):
        raise RuntimeError("nullspace posture inputs must be finite")
    identity = torch.eye(current.shape[1], device=current.device, dtype=current.dtype)
    projector = identity - torch.linalg.pinv(jacobian, rtol=1.0e-4) @ jacobian
    delta = (projector @ (gain * (preferred - current)).unsqueeze(-1)).squeeze(-1)
    # A single scale preserves the nullspace direction under the step bound.
    return delta * (max_step / delta.abs().amax(1, keepdim=True).clamp_min(max_step)).clamp(max=1)


def bounded_joint_integral(reference, current, delta, stiffness, gravity, effort_limits,
                           position_limits, gain, max_delta):
    """Integrate pose error with drive-effort headroom and joint-limit anti-windup."""
    import torch

    values = (reference, current, delta, stiffness, gravity, effort_limits, position_limits)
    if any(not torch.isfinite(value).all() for value in values):
        raise RuntimeError("joint integral inputs must be finite")
    if torch.any(stiffness <= 0) or torch.any(effort_limits <= 0):
        raise RuntimeError("joint integral requires positive drive stiffness and effort limits")
    headroom = ((effort_limits - gravity.abs()).clamp_min(0) / stiffness).clamp(max=max_delta)
    lower = torch.maximum(current - headroom, position_limits[:, :, 0])
    upper = torch.minimum(current + headroom, position_limits[:, :, 1])
    return torch.maximum(torch.minimum(reference + gain * delta, upper), lower)


def bounded_residual_integral(reference, current, baseline, residual, stiffness, gravity,
                              effort_limits, position_limits, gain, max_delta):
    stored = bounded_joint_integral(reference, current, baseline, stiffness, gravity,
                                    effort_limits, position_limits, gain, max_delta)
    command = bounded_joint_integral(stored, current, residual, stiffness, gravity,
                                     effort_limits, position_limits, 1.0, max_delta)
    return stored, command


def _torch_quaternion_multiply_wxyz(left, right):
    import torch

    lw, lx, ly, lz = left.unbind(dim=-1)
    rw, rx, ry, rz = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def _torch_quaternion_rotate_wxyz(quaternion, vector):
    import torch

    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    return (
        vector * (2.0 * scalar.square() - 1.0)
        + 2.0 * scalar * torch.linalg.cross(xyz, vector, dim=-1)
        + 2.0 * xyz * torch.sum(xyz * vector, dim=-1, keepdim=True)
    )


def com_velocity_to_link_velocity(poses_world_wxyz, com_velocities_world, com_positions_body):
    """Translate native center-of-mass velocities to actor-link origins."""
    import torch

    if poses_world_wxyz.shape[:-1] != com_velocities_world.shape[:-1]:
        raise ValueError("pose and velocity sample dimensions must match")
    if poses_world_wxyz.shape[-1] != 7 or com_velocities_world.shape[-1] != 6:
        raise ValueError("poses must end in 7 and velocities must end in 6")
    if com_positions_body.shape != poses_world_wxyz.shape[:-1] + (3,):
        raise ValueError("body-frame COM offsets must match pose sample dimensions")
    angular = com_velocities_world[..., 3:]
    com_to_link_world = _torch_quaternion_rotate_wxyz(
        poses_world_wxyz[..., 3:], -com_positions_body
    )
    linear = com_velocities_world[..., :3] + torch.linalg.cross(
        angular, com_to_link_world, dim=-1
    )
    return torch.cat((linear, angular), dim=-1)


def evaluate_mating_stability_samples(
    poses_world_wxyz,
    link_velocities_world,
    socket_entry_position_body,
    socket_entry_quaternion_wxyz,
    plug_tip_position_body,
    plug_tip_quaternion_wxyz,
    *,
    min_depth_m,
    max_depth_m,
    max_lateral_error_m,
    max_orientation_error_rad,
    max_relative_linear_speed_m_s,
    max_relative_angular_speed_rad_s,
):
    """Evaluate full-pose RJ45 geometry and mating velocity at every physics sample."""
    import torch

    if poses_world_wxyz.ndim != 4 or poses_world_wxyz.shape[2:] != (2, 7):
        raise ValueError("poses must have shape (num_envs, samples, 2, 7)")
    if link_velocities_world.shape != poses_world_wxyz.shape[:-1] + (6,):
        raise ValueError("link velocities must match pose samples and end in 6")
    num_envs, sample_count = poses_world_wxyz.shape[:2]

    def expand_frame(value, width):
        tensor = torch.as_tensor(value, device=poses_world_wxyz.device, dtype=poses_world_wxyz.dtype)
        if tensor.shape == (width,):
            tensor = tensor.reshape(1, 1, width).expand(num_envs, sample_count, width)
        elif tensor.shape == (num_envs, width):
            tensor = tensor[:, None, :].expand(num_envs, sample_count, width)
        elif tensor.shape != (num_envs, sample_count, width):
            raise ValueError("mating frame tensors must be shared, per-environment, or per-sample")
        return tensor

    entry_position_body = expand_frame(socket_entry_position_body, 3)
    entry_quaternion_body = expand_frame(socket_entry_quaternion_wxyz, 4)
    tip_position_body = expand_frame(plug_tip_position_body, 3)
    tip_quaternion_body = expand_frame(plug_tip_quaternion_wxyz, 4)

    socket_position = poses_world_wxyz[:, :, 0, :3]
    socket_quaternion = poses_world_wxyz[:, :, 0, 3:]
    plug_position = poses_world_wxyz[:, :, 1, :3]
    plug_quaternion = poses_world_wxyz[:, :, 1, 3:]
    entry_offset_world = _torch_quaternion_rotate_wxyz(socket_quaternion, entry_position_body)
    tip_offset_world = _torch_quaternion_rotate_wxyz(plug_quaternion, tip_position_body)
    entry_position_world = socket_position + entry_offset_world
    tip_position_world = plug_position + tip_offset_world
    entry_quaternion_world = _torch_quaternion_multiply_wxyz(
        socket_quaternion, entry_quaternion_body
    )
    tip_quaternion_world = _torch_quaternion_multiply_wxyz(
        plug_quaternion, tip_quaternion_body
    )
    entry_inverse = entry_quaternion_world.clone()
    entry_inverse[..., 1:] *= -1.0
    relative_position = _torch_quaternion_rotate_wxyz(
        entry_inverse, tip_position_world - entry_position_world
    )
    relative_quaternion = _torch_quaternion_multiply_wxyz(entry_inverse, tip_quaternion_world)

    socket_linear = link_velocities_world[:, :, 0, :3]
    socket_angular = link_velocities_world[:, :, 0, 3:]
    plug_linear = link_velocities_world[:, :, 1, :3]
    plug_angular = link_velocities_world[:, :, 1, 3:]
    entry_velocity_world = socket_linear + torch.linalg.cross(
        socket_angular, entry_offset_world, dim=-1
    )
    tip_velocity_world = plug_linear + torch.linalg.cross(
        plug_angular, tip_offset_world, dim=-1
    )
    relative_linear_velocity_world = (
        tip_velocity_world
        - entry_velocity_world
        - torch.linalg.cross(
            socket_angular, tip_position_world - entry_position_world, dim=-1
        )
    )
    relative_angular_velocity_world = plug_angular - socket_angular

    depth = relative_position[..., 1]
    lateral = torch.linalg.vector_norm(relative_position[..., (0, 2)], dim=-1)
    orientation_angle = 2.0 * torch.atan2(
        torch.linalg.vector_norm(relative_quaternion[..., 1:], dim=-1),
        relative_quaternion[..., 0].abs(),
    )
    linear_speed = torch.linalg.vector_norm(relative_linear_velocity_world, dim=-1)
    angular_speed = torch.linalg.vector_norm(relative_angular_velocity_world, dim=-1)
    finite = (
        torch.isfinite(poses_world_wxyz).all(dim=(-1, -2))
        & torch.isfinite(link_velocities_world).all(dim=(-1, -2))
        & torch.isfinite(relative_position).all(dim=-1)
        & torch.isfinite(relative_quaternion).all(dim=-1)
        & torch.isfinite(linear_speed)
        & torch.isfinite(angular_speed)
    )
    quaternion_valid = (
        (torch.linalg.vector_norm(socket_quaternion, dim=-1) - 1.0).abs() <= 1.0e-5
    ) & ((torch.linalg.vector_norm(plug_quaternion, dim=-1) - 1.0).abs() <= 1.0e-5)
    stable = (
        finite
        & quaternion_valid
        & (depth >= min_depth_m)
        & (depth <= max_depth_m)
        & (lateral <= max_lateral_error_m)
        & (orientation_angle <= max_orientation_error_rad)
        & (linear_speed <= max_relative_linear_speed_m_s)
        & (angular_speed <= max_relative_angular_speed_rad_s)
    )
    return {
        "finite": finite,
        "depth_m": depth,
        "lateral_error_m": lateral,
        "orientation_error_rad": orientation_angle,
        "relative_linear_speed_m_s": linear_speed,
        "relative_angular_speed_rad_s": angular_speed,
        "stable": stable,
        "all_samples_stable": stable.all(dim=1),
    }


def physics_window_dwell_gate(is_insert, samples_complete, sample_stable):
    """Require a complete, stable physics window while the task is inserting."""
    if sample_stable.ndim != 2:
        raise ValueError("sample stability must have shape (num_envs, samples)")
    if is_insert.shape != sample_stable.shape[:1]:
        raise ValueError("insert phase mask must match the environment count")
    if samples_complete.shape != sample_stable.shape[:1]:
        raise ValueError("sample completeness mask must match the environment count")
    return (
        is_insert
        & samples_complete
        & sample_stable[:, -1]
        & sample_stable.all(dim=1)
    )


def configure_rj45_task(cfg, task: dict[str, Any], project: Path) -> type:
    """Apply RJ45 assets/limits to ``cfg`` and return an environment subclass.

    Isaac imports intentionally remain inside this function so config validation
    and frame-math tests run on CPU-only machines.
    """

    if cfg.scene.num_envs != 1:
        raise ValueError("parallel RJ45 collision validation pending; use exactly one environment")
    import torch
    from isaacsim.core.utils.stage import get_current_stage
    from isaaclab.utils.math import (
        combine_frame_transforms,
        quat_apply,
        quat_box_minus,
        quat_error_magnitude,
        subtract_frame_transforms,
    )
    from pxr import PhysxSchema

    from rby1_assembly.isaac_env import AssemblyPhase, GRASP_CENTER_IN_WRIST_M, Rby1AssemblyEnv

    project = Path(project).resolve()
    for name in ("robot", "socket", "plug", "table", "build_report"):
        _project_asset(project, task, name)
    build_report = json.loads(_project_asset(project, task, "build_report").read_text(encoding="utf-8"))
    if not isinstance(build_report, dict) or build_report.get("physics_executed") is not False:
        raise ValueError("RJ45 build report must explicitly identify an unexecuted static asset build")
    if build_report.get("config_sha256") != task.get("_config_sha256"):
        raise ValueError("RJ45 assets were not built from the selected task configuration")
    verify_generated_asset_hashes(project, task, build_report)

    cfg.robot.spawn.usd_path = str(_project_asset(project, task, "robot"))
    cfg.socket.spawn.usd_path = str(_project_asset(project, task, "socket"))
    cfg.plug.spawn.usd_path = str(_project_asset(project, task, "plug"))
    cfg.table.spawn.usd_path = str(_project_asset(project, task, "table"))
    cfg.socket.init_state.pos = tuple(task["socket"]["initial_position_m"])
    cfg.socket.init_state.rot = tuple(task["socket"]["initial_orientation_wxyz"])
    cfg.plug.init_state.pos = tuple(task["plug"]["initial_position_m"])
    cfg.plug.init_state.rot = tuple(task["plug"]["initial_orientation_wxyz"])

    contact = task["contact"]
    cfg.sim.physics_material.static_friction = contact["static_friction"]
    cfg.sim.physics_material.dynamic_friction = contact["dynamic_friction"]
    cfg.sim.physics_material.restitution = contact["restitution"]
    # Resolve light-part contact constraints before carrying velocity forward.
    # Keep geometric/dwell thresholds unchanged while testing solver convergence.
    for body_cfg in (cfg.socket, cfg.plug):
        body_cfg.spawn.rigid_props.solver_position_iteration_count = contact["solver_position_iteration_count"]
        body_cfg.spawn.rigid_props.solver_velocity_iteration_count = contact["solver_velocity_iteration_count"]
    # Island solver counts use the largest dynamic-actor request. Keep the
    # articulation and both light parts on the same explicit setting.
    cfg.robot.spawn.articulation_props.solver_position_iteration_count = contact["solver_position_iteration_count"]
    cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = contact["solver_velocity_iteration_count"]
    evaluation = task["evaluation"]
    cfg.episode_length_s = evaluation["timeout_s"]
    cfg.assembled_depth_m = evaluation["assembled_depth_m"]
    cfg.max_lateral_error_m = evaluation["max_lateral_error_m"]
    cfg.max_orientation_error_rad = evaluation["max_orientation_error_rad"]
    cfg.hard_contact_force_n = evaluation["max_contact_force_n"]
    cfg.success_dwell_s = evaluation["dwell_s"]
    cfg.max_depth_m = evaluation["max_depth_m"]
    cfg.max_relative_linear_speed_m_s = evaluation["max_relative_linear_speed_m_s"]
    cfg.max_relative_angular_speed_rad_s = evaluation["max_relative_angular_speed_rad_s"]
    controller = task["controller"]
    for name in (
        "left_gripper_grasp_magnitude_m",
        "right_gripper_grasp_magnitude_m",
        "gripper_open_magnitude_m",
        "residual_translation_bound_m",
        "residual_rotation_bound_rad",
        "alignment_standoff_m",
        "insertion_target_margin_m",
        "insertion_speed_m_s",
        "insertion_max_tracking_lead_m",
    ):
        setattr(cfg, name, controller[name])
    for name in (
        "align_timeout_s",
        "dls_damping",
        "grip_contact_force_n",
        "grip_settle_s",
        "insert_timeout_s",
        "insertion_advance_lateral_tolerance_m",
        "insertion_advance_orientation_tolerance_rad",
        "insertion_integral_gain",
        "insertion_resume_lateral_tolerance_m",
        "insertion_resume_orientation_tolerance_rad",
        "jam_contact_force_n",
        "max_joint_delta_rad",
        "max_retries",
        "phase_timeout_s",
        "pregrasp_distance_m",
        "pregrasp_timeout_s",
    ):
        if name in controller:
            setattr(cfg, name, controller[name])
    cfg.pregrasp_orientation_tolerance_rad = controller.get(
        "pregrasp_orientation_tolerance_rad", 0.05
    )
    cfg.pregrasp_above_height_m = controller.get("pregrasp_above_height_m", cfg.pregrasp_above_height_m)
    cfg.assembly_socket_goal_m = tuple(controller["assembly_socket_goal_m"])
    cfg.assembly_socket_goal_quat_wxyz = tuple(
        controller["assembly_socket_goal_quaternion_wxyz"]
    )
    cfg.pregrasp_clearance_joint_pos_rad = tuple(
        controller.get("pregrasp_clearance_joint_pos_rad", cfg.pregrasp_clearance_joint_pos_rad)
    )
    if "pregrasp_safe_wrist_height_m" in controller:
        cfg.left_pregrasp_safe_wrist_z_m = controller["pregrasp_safe_wrist_height_m"]
        cfg.right_pregrasp_safe_wrist_z_m = controller["pregrasp_safe_wrist_height_m"]
    left_grasp_offset = quaternion_rotate_wxyz(
        np.asarray(task["socket"]["initial_orientation_wxyz"]),
        np.asarray(task["frames"]["left_grasp"]["position_m"]),
    )
    right_grasp_offset = quaternion_rotate_wxyz(
        np.asarray(task["plug"]["initial_orientation_wxyz"]),
        np.asarray(task["frames"]["right_grasp"]["position_m"]),
    )
    cfg.left_grasp_center_above_object_m = float(left_grasp_offset[2])
    cfg.right_grasp_center_above_object_m = float(right_grasp_offset[2])
    cfg.rj45_task = task

    class Rby1Rj45Env(Rby1AssemblyEnv):
        """RB-Y1 RJ45 insertion using measured full-pose mating frames."""

        def _setup_scene(self) -> None:
            super()._setup_scene()
            enabled = self.cfg.rj45_task["contact"][
                "enable_external_forces_every_iteration"
            ]
            if type(enabled) is not bool:
                raise ValueError(
                    "contact.enable_external_forces_every_iteration must be boolean"
                )
            stage = get_current_stage()
            physics_scene_prim = stage.GetPrimAtPath(self.cfg.sim.physics_prim_path)
            if not physics_scene_prim.IsValid():
                raise RuntimeError(
                    f"missing physics scene prim: {self.cfg.sim.physics_prim_path}"
                )
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
            if not scene_api:
                raise RuntimeError("failed to apply PhysxSceneAPI to the physics scene")
            attribute = scene_api.CreateEnableExternalForcesEveryIterationAttr(enabled)
            if not attribute.IsValid() or attribute.Get() != enabled:
                raise RuntimeError(
                    "failed to author enableExternalForcesEveryIteration on the physics scene"
                )
            self._external_forces_every_iteration_enabled = enabled

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self._physics_pose_samples = []
            self._physics_velocity_samples = []
            self._physics_link_velocity_samples = []
            if not hasattr(self, "_verified_gravity_shape"):
                gravity = self.robot.root_physx_view.get_gravity_compensation_forces()
                if not self.robot.is_fixed_base or gravity.shape != self.robot.data.joint_pos.shape:
                    raise RuntimeError("RJ45 gravity feedforward requires fixed-base joint-order tensors")
                self._verified_gravity_shape = tuple(gravity.shape)
            controller = self.cfg.rj45_task["controller"]
            if "torso_work_joint_positions_rad" in controller:
                if not hasattr(self, "_initial_clearance_reached"):
                    self._initial_clearance_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self._work_clearance_reached = torch.zeros_like(self._initial_clearance_reached)
                    self._approach_progress = torch.zeros((self.num_envs, 2), device=self.device)
                    self._torso_progress = torch.zeros(self.num_envs, device=self.device)
                initial = torch.tensor(controller["initial_clearance_joint_positions_rad"], device=self.device)
                working = torch.tensor(self.cfg.pregrasp_clearance_joint_pos_rad, device=self.device)
                arms = torch.stack((
                    self.robot.data.joint_pos[:, self._left_arm_joint_ids],
                    self.robot.data.joint_pos[:, self._right_arm_joint_ids],
                ), dim=1)
                tolerance = controller["pregrasp_waypoint_tolerance_rad"]
                self._initial_clearance_reached |= (arms - initial).abs().amax((1, 2)) <= tolerance
                self._work_clearance_reached |= (
                    self._initial_clearance_reached & ((arms - working).abs().amax((1, 2)) <= tolerance)
                )
                work_pose = torch.tensor(
                    controller["torso_work_joint_positions_rad"], device=self.device
                )
                limits = self.robot.data.soft_joint_pos_limits[:, self._torso_joint_ids]
                if not torch.isfinite(limits).all() or torch.any(work_pose < limits[:, :, 0]) or torch.any(work_pose > limits[:, :, 1]):
                    raise ValueError("RJ45 torso work posture exceeds robot joint limits")
                current = self.robot.data.joint_pos[:, self._torso_joint_ids]
                max_delta = controller["torso_max_joint_delta_rad"]
                # Keep the Z-shaped torso joints on one shared trajectory.
                # Equal per-joint rates would pitch the upper body forward
                # while the middle joint travels twice as far.
                self._torso_progress, torso_reference = advance_joint_reference(
                    current, torch.zeros_like(work_pose), work_pose, self._torso_progress,
                    self._work_clearance_reached, max_delta,
                    controller["torso_reference_tracking_tolerance_rad"],
                )
                self._torso_hold_targets[:] = torso_reference
            super()._pre_physics_step(actions)
            if not hasattr(self, "_pregrasp_servo_targets"):
                self._pregrasp_servo_targets = torch.stack((
                    self.robot.data.joint_pos[:, self._left_arm_joint_ids],
                    self.robot.data.joint_pos[:, self._right_arm_joint_ids],
                ), dim=1).clone()
            for arm_index, (name, joint_ids) in enumerate((
                ("left_above_joint_positions_rad", self._left_arm_joint_ids),
                ("right_above_joint_positions_rad", self._right_arm_joint_ids),
            )):
                if name not in controller:
                    continue
                approaching = (self._phase == int(AssemblyPhase.PREGRASP)) & self._pregrasp_lift_reached & ~self._pregrasp_above_reached
                current = self.robot.data.joint_pos[:, joint_ids]
                waypoint = torch.tensor(controller[name], device=self.device)
                limits = self.robot.data.soft_joint_pos_limits[:, joint_ids]
                if not torch.isfinite(limits).all() or torch.any(waypoint < limits[:, :, 0]) or torch.any(waypoint > limits[:, :, 1]):
                    raise ValueError(f"RJ45 {name} exceeds robot joint limits")
                max_delta = controller["approach_waypoint_max_joint_delta_rad"]
                start = torch.tensor(self.cfg.pregrasp_clearance_joint_pos_rad, device=self.device)
                can_advance = approaching.clone()
                if name == "right_above_joint_positions_rad":
                    left_goal, left_quat, _, _ = self._grasp_target_poses()
                    left_goal[:, 2] += self.cfg.pregrasp_above_height_m
                    left_ee = self.robot.data.body_pos_w[:, self._left_ee_body_id]
                    left_actual_quat = self.robot.data.body_quat_w[:, self._left_ee_body_id]
                    offset = torch.tensor(GRASP_CENTER_IN_WRIST_M, device=self.device).expand(self.num_envs, -1)
                    left_center = left_ee + quat_apply(left_actual_quat, offset)
                    left_ready = (
                        (self._approach_progress[:, 0] >= 1)
                        & (torch.linalg.vector_norm(left_center - left_goal, dim=1) < 0.005)
                        & (quat_error_magnitude(left_quat, left_actual_quat) <= self.cfg.pregrasp_orientation_tolerance_rad)
                    )
                    can_advance &= left_ready
                self._approach_progress[:, arm_index], planned = advance_joint_reference(
                    current, start, waypoint, self._approach_progress[:, arm_index],
                    can_advance, max_delta, controller["approach_waypoint_tracking_tolerance_rad"],
                )
                # Joint waypoints choose a collision-screened IK branch. Once
                # reached, the Cartesian servo settles the measured hand pose.
                joint_path = approaching & (self._approach_progress[:, arm_index] < 1)
                targets = self._joint_targets[:, joint_ids]
                targets[joint_path] = planned[joint_path]
                cartesian = (
                    (self._approach_progress[:, arm_index] >= 1)
                    & (self._phase < int(AssemblyPhase.INSERT))
                )
                jacobian_id = self._left_ee_jacobian_id if arm_index == 0 else self._right_ee_jacobian_id
                jacobian = self.robot.root_physx_view.get_jacobians()[:, jacobian_id, :, joint_ids]
                posture = nullspace_posture_step(
                    jacobian, current, waypoint,
                    controller["nullspace_posture_gain"], controller["nullspace_max_joint_delta_rad"],
                )
                baseline_delta = self._damped_least_squares(jacobian, self._baseline_cartesian_twists[:, arm_index])
                residual_delta = targets - current - baseline_delta
                servoing = cartesian & (self._phase <= int(AssemblyPhase.ALIGN))
                gain = torch.where(
                    (self._phase == int(AssemblyPhase.ALIGN))[:, None],
                    controller["alignment_integral_gain"], controller["pregrasp_integral_gain"],
                )
                integrated, corrected = bounded_residual_integral(
                    self._pregrasp_servo_targets[:, arm_index], current,
                    baseline_delta + posture, residual_delta,
                    self.robot.data.joint_stiffness[:, joint_ids],
                    self.robot.root_physx_view.get_gravity_compensation_forces()[:, joint_ids],
                    self.robot.data.joint_effort_limits[:, joint_ids], limits,
                    gain, self.cfg.max_joint_delta_rad,
                )
                targets[servoing] = corrected[servoing]
                targets = torch.maximum(torch.minimum(targets, limits[:, :, 1]), limits[:, :, 0])
                self._pregrasp_servo_targets[:, arm_index] = torch.where(servoing[:, None], integrated, targets)
                self._joint_targets[:, joint_ids] = targets
            if "gripper_command_speed_m_s" in controller:
                if not hasattr(self, "_gripper_command"):
                    self._gripper_command = self.robot.data.joint_pos[:, self._gripper_joint_ids].clone()
                desired = self._joint_targets[:, self._gripper_joint_ids]
                max_step = controller["gripper_command_speed_m_s"] * self.step_dt
                self._gripper_command += (desired - self._gripper_command).clamp(-max_step, max_step)
                self._joint_targets[:, self._gripper_joint_ids] = self._gripper_command

        def _apply_action(self):
            super()._apply_action()
            poses = []
            velocities = []
            link_velocities = []
            for obj in (self.socket, self.plug):
                native = obj.root_physx_view.get_transforms().clone()
                pose = torch.cat((native[:, :3], native[:, 6:7], native[:, 3:6]), dim=1)
                velocity = obj.root_physx_view.get_velocities().clone()
                poses.append(pose)
                velocities.append(velocity)
                link_velocities.append(com_velocity_to_link_velocity(
                    pose,
                    velocity,
                    obj.data.body_com_pos_b[:, 0],
                ))
            self._physics_pose_samples.append(torch.stack(poses, dim=1))
            self._physics_velocity_samples.append(torch.stack(velocities, dim=1))
            self._physics_link_velocity_samples.append(torch.stack(link_velocities, dim=1))

        def _extra_forbidden_static_filters(self):
            return tuple("{ENV_REGEX_NS}/Table/" + name for name in (
                "pickup_nest", "pickup_nest_plate", "pickup_locator_left", "pickup_locator_right", "pickup_locator_front",
            ))

        def _transition_where(self, mask, phase, now, reason=None):
            from rby1_assembly.isaac_env import TerminationReason
            if phase == AssemblyPhase.CLOSE and torch.any(mask):
                poses = self._grasp_target_poses()
                if not hasattr(self, "_frozen_grasp_positions"):
                    self._frozen_grasp_positions = torch.zeros((self.num_envs, 2, 3), device=self.device)
                    self._frozen_grasp_quaternions = torch.zeros((self.num_envs, 2, 4), device=self.device)
                    self._grasp_reference_locked = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                self._frozen_grasp_positions[mask] = torch.stack((poses[0][mask], poses[2][mask]), dim=1)
                self._frozen_grasp_quaternions[mask] = torch.stack((poses[1][mask], poses[3][mask]), dim=1)
                self._grasp_reference_locked[mask] = True
            super()._transition_where(mask, phase, now, TerminationReason.NONE if reason is None else reason)
            if phase == AssemblyPhase.INSERT and hasattr(self, "_pregrasp_servo_targets"):
                self._insert_joint_targets[mask] = torch.cat((
                    self._pregrasp_servo_targets[mask, 0], self._pregrasp_servo_targets[mask, 1],
                ), dim=1)

        def _pregrasp_clearance_joint_targets(self):
            working = super()._pregrasp_clearance_joint_targets()
            if not hasattr(self, "_initial_clearance_reached"):
                return working
            initial = torch.tensor(
                self.cfg.rj45_task["controller"]["initial_clearance_joint_positions_rad"], device=self.device
            )
            return torch.where(self._initial_clearance_reached[:, None], working, initial)

        def _reset_idx(self, env_ids=None):
            super()._reset_idx(env_ids)
            if hasattr(self, "_initial_clearance_reached"):
                if env_ids is None:
                    env_ids = self.robot._ALL_INDICES
                self._initial_clearance_reached[env_ids] = False
                self._work_clearance_reached[env_ids] = False
                self._approach_progress[env_ids] = 0.0
                self._torso_progress[env_ids] = 0.0
            if hasattr(self, "_gripper_command"):
                if env_ids is None:
                    env_ids = self.robot._ALL_INDICES
                self._gripper_command[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self._gripper_joint_ids]
            if hasattr(self, "_pregrasp_servo_targets"):
                if env_ids is None:
                    env_ids = self.robot._ALL_INDICES
                self._pregrasp_servo_targets[env_ids] = torch.stack((
                    self.robot.data.default_joint_pos[env_ids][:, self._left_arm_joint_ids],
                    self.robot.data.default_joint_pos[env_ids][:, self._right_arm_joint_ids],
                ), dim=1)
            if hasattr(self, "_grasp_reference_locked"):
                self._grasp_reference_locked[env_ids] = False
            if hasattr(self, "_pose_difference_valid"):
                self._pose_difference_valid[env_ids] = False

        def _pregrasp_lift_mask(self, left_ee, right_ee) -> torch.Tensor:
            ready = super()._pregrasp_lift_mask(left_ee, right_ee)
            controller = self.cfg.rj45_task["controller"]
            if "torso_work_joint_positions_rad" not in controller:
                return ready
            work_pose = torch.tensor(
                controller["torso_work_joint_positions_rad"], device=self.device
            )
            clearance = torch.tensor(self.cfg.pregrasp_clearance_joint_pos_rad, device=self.device)
            torso_error = (self.robot.data.joint_pos[:, self._torso_joint_ids] - work_pose).abs().amax(1)
            left_error = (self.robot.data.joint_pos[:, self._left_arm_joint_ids] - clearance).abs().amax(1)
            right_error = (self.robot.data.joint_pos[:, self._right_arm_joint_ids] - clearance).abs().amax(1)
            return (
                ready
                & (torso_error <= controller["pregrasp_torso_tolerance_rad"])
                & (left_error <= controller["pregrasp_waypoint_tolerance_rad"])
                & (right_error <= controller["pregrasp_waypoint_tolerance_rad"])
            )

        def _pregrasp_above_mask(self, left_grasp_center, right_grasp_center):
            return super()._pregrasp_above_mask(
                left_grasp_center, right_grasp_center
            ) & self._pregrasp_ready_mask()

        def _desired_ee_twists(self):
            left_twist, right_twist = super()._desired_ee_twists()
            # Rotate the horizontal hand while it is still at the clearance
            # waypoint, before translating down towards the small plug.
            _, _, _, right_target_quat = self._grasp_target_poses()
            right_quat = self.robot.data.body_quat_w[:, self._right_ee_body_id]
            preparing = (
                (self._phase == int(AssemblyPhase.PREGRASP))
                & self._pregrasp_lift_reached
                & ~self._pregrasp_above_reached
                & (quat_error_magnitude(right_target_quat, right_quat) > self.cfg.pregrasp_orientation_tolerance_rad)
            )
            right_twist[preparing, :3] = 0.0
            self._baseline_cartesian_twists = torch.stack((left_twist, right_twist), dim=1).clone()
            return left_twist, right_twist

        def _frame_tensors(self, name: str) -> tuple[torch.Tensor, torch.Tensor]:
            frame = self.cfg.rj45_task["frames"][name]
            position = torch.tensor(frame["position_m"], device=self.device).repeat(self.num_envs, 1)
            quaternion = torch.tensor(
                frame["quaternion_wxyz"], device=self.device
            ).repeat(self.num_envs, 1)
            return position, quaternion

        def _relative_object_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
            entry_position, entry_quaternion = self._frame_tensors("socket_entry")
            tip_position, tip_quaternion = self._frame_tensors("plug_tip")
            entry_world_position, entry_world_quaternion = combine_frame_transforms(
                self.socket.data.root_pos_w,
                self.socket.data.root_quat_w,
                entry_position,
                entry_quaternion,
            )
            tip_world_position, tip_world_quaternion = combine_frame_transforms(
                self.plug.data.root_pos_w,
                self.plug.data.root_quat_w,
                tip_position,
                tip_quaternion,
            )
            relative_position, relative_quaternion = subtract_frame_transforms(
                entry_world_position,
                entry_world_quaternion,
                tip_world_position,
                tip_world_quaternion,
            )
            self._last_actual_relative_pose[:, :3] = relative_position
            self._last_actual_relative_pose[:, 3:] = relative_quaternion
            return relative_position, relative_quaternion

        def _grasp_target_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
            left_position, _, right_position, _ = self._grasp_target_poses()
            return left_position, right_position

        def _grasp_target_poses(self):
            """Full-pose hook consumed when the base controller exposes it."""

            left_position, left_quaternion = self._frame_tensors("left_grasp")
            right_position, right_quaternion = self._frame_tensors("right_grasp")
            left_world = combine_frame_transforms(
                self.socket.data.root_pos_w,
                self.socket.data.root_quat_w,
                left_position,
                left_quaternion,
            )
            right_world = combine_frame_transforms(
                self.plug.data.root_pos_w,
                self.plug.data.root_quat_w,
                right_position,
                right_quaternion,
            )
            if hasattr(self, "_grasp_reference_locked"):
                held = self._grasp_reference_locked & (
                    (self._phase == int(AssemblyPhase.CLOSE)) | (self._phase == int(AssemblyPhase.CHECK_LIFT))
                )
                left_world = (
                    torch.where(held[:, None], self._frozen_grasp_positions[:, 0], left_world[0]),
                    torch.where(held[:, None], self._frozen_grasp_quaternions[:, 0], left_world[1]),
                )
                right_world = (
                    torch.where(held[:, None], self._frozen_grasp_positions[:, 1], right_world[0]),
                    torch.where(held[:, None], self._frozen_grasp_quaternions[:, 1], right_world[1]),
                )
            return (*left_world, *right_world)

        def _check_lift_target_positions(
            self,
            left_target: torch.Tensor,
            right_target: torch.Tensor,
            left_target_quat: torch.Tensor,
            right_target_quat: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            grasp_offset = torch.tensor(
                GRASP_CENTER_IN_WRIST_M,
                device=self.device,
                dtype=left_target.dtype,
            ).repeat(self.num_envs, 1)
            left_grasp_center = left_target + quat_apply(left_target_quat, grasp_offset)
            right_grasp_center = right_target + quat_apply(right_target_quat, grasp_offset)
            left_grasp_center[:, 2] = (
                self.scene.env_origins[:, 2]
                + self._object_start_z[:, 0]
                + 0.050
                + self.cfg.left_grasp_center_above_object_m
            )
            right_grasp_center[:, 2] = (
                self.scene.env_origins[:, 2]
                + self._object_start_z[:, 1]
                + 0.050
                + self.cfg.right_grasp_center_above_object_m
            )
            return (
                left_grasp_center - quat_apply(left_target_quat, grasp_offset),
                right_grasp_center - quat_apply(right_target_quat, grasp_offset),
            )

        def _pregrasp_ready_mask(self) -> torch.Tensor:
            _, left_target_quat, _, right_target_quat = self._grasp_target_poses()
            left_quat = self.robot.data.body_quat_w[:, self._left_ee_body_id]
            right_quat = self.robot.data.body_quat_w[:, self._right_ee_body_id]
            tolerance = self.cfg.pregrasp_orientation_tolerance_rad
            return (quat_error_magnitude(left_target_quat, left_quat) <= tolerance) & (
                quat_error_magnitude(right_target_quat, right_quat) <= tolerance
            )

        def _reset_rigid_object(self, obj, env_ids, y_noise: float) -> None:
            del y_noise
            state = obj.data.default_root_state[env_ids].clone()
            state[:, :3] += self.scene.env_origins[env_ids]
            noise = torch.tensor(
                self.cfg.rj45_task["reset"]["position_noise_m"],
                device=self.device,
                dtype=state.dtype,
            )
            state[:, :3] += (torch.rand((len(env_ids), 3), device=self.device) * 2.0 - 1.0) * noise
            obj.write_root_pose_to_sim(state[:, :7], env_ids)
            obj.write_root_velocity_to_sim(state[:, 7:], env_ids)

        def _plug_center_goal(self, socket_center_w: torch.Tensor, depth_m) -> torch.Tensor:
            socket_quaternion = torch.tensor(
                self.cfg.assembly_socket_goal_quat_wxyz, device=self.device
            ).repeat(socket_center_w.shape[0], 1)
            goal_position, _ = self._plug_pose_goal(socket_center_w, socket_quaternion, depth_m)
            return goal_position

        def _plug_pose_goal(self, socket_position_w, socket_quaternion_w, depth_m):
            count = socket_position_w.shape[0]
            entry_position, entry_quaternion = self._frame_tensors("socket_entry")
            tip_position, tip_quaternion = self._frame_tensors("plug_tip")
            entry_position = entry_position[:count]
            entry_quaternion = entry_quaternion[:count]
            tip_position = tip_position[:count]
            tip_quaternion = tip_quaternion[:count]
            entry_world_position, entry_world_quaternion = combine_frame_transforms(
                socket_position_w,
                socket_quaternion_w,
                entry_position,
                entry_quaternion,
            )
            depth = torch.as_tensor(depth_m, device=self.device, dtype=entry_world_position.dtype)
            depth = depth.expand(count)
            target_tip_offset = torch.zeros((count, 3), device=self.device)
            target_tip_offset[:, 1] = depth
            target_tip_position = entry_world_position + quat_apply(
                entry_world_quaternion, target_tip_offset
            )
            inverse_tip_quaternion = tip_quaternion.clone()
            inverse_tip_quaternion[:, 1:] *= -1.0
            inverse_tip_position = -quat_apply(inverse_tip_quaternion, tip_position)
            goal_position, goal_quaternion = combine_frame_transforms(
                target_tip_position,
                entry_world_quaternion,
                inverse_tip_position,
                inverse_tip_quaternion,
            )
            return goal_position, goal_quaternion

        def _residual_actions_world(self, actions: torch.Tensor) -> torch.Tensor:
            bounded = actions.reshape(self.num_envs, 4, 3)
            entry_position, entry_quaternion = self._frame_tensors("socket_entry")
            del entry_position
            entry_world_quaternion = combine_frame_transforms(
                self.socket.data.root_pos_w,
                self.socket.data.root_quat_w,
                torch.zeros((self.num_envs, 3), device=self.device),
                entry_quaternion,
            )[1]
            rotated = torch.stack(
                [quat_apply(entry_world_quaternion, bounded[:, index]) for index in range(4)],
                dim=1,
            )
            return rotated.reshape(self.num_envs, 12)

        def _relative_mating_velocity(self) -> tuple[torch.Tensor, torch.Tensor]:
            entry_position, _ = self._frame_tensors("socket_entry")
            tip_position, _ = self._frame_tensors("plug_tip")
            entry_offset_world = quat_apply(self.socket.data.root_quat_w, entry_position)
            tip_offset_world = quat_apply(self.plug.data.root_quat_w, tip_position)
            entry_position_world = self.socket.data.root_pos_w + entry_offset_world
            tip_position_world = self.plug.data.root_pos_w + tip_offset_world
            entry_velocity_world = self.socket.data.root_link_lin_vel_w + torch.linalg.cross(
                self.socket.data.root_link_ang_vel_w, entry_offset_world, dim=1
            )
            tip_velocity_world = self.plug.data.root_link_lin_vel_w + torch.linalg.cross(
                self.plug.data.root_link_ang_vel_w, tip_offset_world, dim=1
            )
            relative_linear_velocity_world = (
                tip_velocity_world
                - entry_velocity_world
                - torch.linalg.cross(
                    self.socket.data.root_link_ang_vel_w,
                    tip_position_world - entry_position_world,
                    dim=1,
                )
            )
            relative_angular_velocity_world = (
                self.plug.data.root_link_ang_vel_w - self.socket.data.root_link_ang_vel_w
            )
            return relative_linear_velocity_world, relative_angular_velocity_world

        def _advance_evaluation_dwell(self, instantaneous_ok: torch.Tensor) -> None:
            expected_substeps = int(self.cfg.decimation)
            samples_complete = (
                len(self._physics_pose_samples) == expected_substeps
                and len(self._physics_link_velocity_samples) == expected_substeps
            )
            endpoint_poses = torch.stack((
                torch.cat((self.socket.data.root_pos_w, self.socket.data.root_quat_w), dim=1),
                torch.cat((self.plug.data.root_pos_w, self.plug.data.root_quat_w), dim=1),
            ), dim=1)[:, None]
            endpoint_link_velocities = torch.stack((
                self.socket.data.root_link_vel_w,
                self.plug.data.root_link_vel_w,
            ), dim=1)[:, None]
            if (
                self._physics_pose_samples
                and len(self._physics_pose_samples)
                == len(self._physics_link_velocity_samples)
            ):
                sampled_poses = torch.stack(self._physics_pose_samples, dim=1)
                sampled_link_velocities = torch.stack(
                    self._physics_link_velocity_samples, dim=1
                )
                all_poses = torch.cat((sampled_poses, endpoint_poses), dim=1)
                all_link_velocities = torch.cat(
                    (sampled_link_velocities, endpoint_link_velocities), dim=1
                )
            else:
                all_poses = endpoint_poses
                all_link_velocities = endpoint_link_velocities
            entry_position, entry_quaternion = self._frame_tensors("socket_entry")
            tip_position, tip_quaternion = self._frame_tensors("plug_tip")
            sampled_stability = evaluate_mating_stability_samples(
                all_poses,
                all_link_velocities,
                entry_position,
                entry_quaternion,
                tip_position,
                tip_quaternion,
                min_depth_m=self.cfg.assembled_depth_m,
                max_depth_m=self.cfg.max_depth_m,
                max_lateral_error_m=self.cfg.max_lateral_error_m,
                max_orientation_error_rad=self.cfg.max_orientation_error_rad,
                max_relative_linear_speed_m_s=self.cfg.max_relative_linear_speed_m_s,
                max_relative_angular_speed_rad_s=self.cfg.max_relative_angular_speed_rad_s,
            )
            samples_complete_tensor = torch.full(
                (self.num_envs,), samples_complete, dtype=torch.bool, device=self.device
            )
            all_samples_stable = (
                sampled_stability["all_samples_stable"] & samples_complete_tensor
            )
            inserting = self._phase == int(AssemblyPhase.INSERT)
            stability_ok = physics_window_dwell_gate(
                inserting, samples_complete_tensor, sampled_stability["stable"]
            )
            depth_within_window = (
                (sampled_stability["depth_m"][:, -1] >= self.cfg.assembled_depth_m)
                & (sampled_stability["depth_m"][:, -1] <= self.cfg.max_depth_m)
            )
            linear_speed_ok = (
                sampled_stability["relative_linear_speed_m_s"][:, -1]
                <= self.cfg.max_relative_linear_speed_m_s
            )
            angular_speed_ok = (
                sampled_stability["relative_angular_speed_rad_s"][:, -1]
                <= self.cfg.max_relative_angular_speed_rad_s
            )
            self._latest_physics_stability = sampled_stability
            self.extras["rj45_evaluation_stability"] = {
                "depth_within_window": depth_within_window.clone(),
                "relative_linear_speed_ok": linear_speed_ok.clone(),
                "relative_angular_speed_ok": angular_speed_ok.clone(),
                "relative_linear_speed_m_s": sampled_stability[
                    "relative_linear_speed_m_s"
                ][:, -1].clone(),
                "relative_angular_speed_rad_s": sampled_stability[
                    "relative_angular_speed_rad_s"
                ][:, -1].clone(),
                "physics_substeps_complete": samples_complete_tensor,
                "physics_all_samples_stable": all_samples_stable,
                "physics_sample_finite": sampled_stability["finite"],
                "physics_sample_stable": sampled_stability["stable"],
                "physics_sample_depth_m": sampled_stability["depth_m"],
                "physics_sample_lateral_error_m": sampled_stability["lateral_error_m"],
                "physics_sample_orientation_error_rad": sampled_stability[
                    "orientation_error_rad"
                ],
                "physics_sample_relative_linear_speed_m_s": sampled_stability[
                    "relative_linear_speed_m_s"
                ],
                "physics_sample_relative_angular_speed_rad_s": sampled_stability[
                    "relative_angular_speed_rad_s"
                ],
            }
            super()._advance_evaluation_dwell(instantaneous_ok & stability_ok)

        def _get_observations(self) -> dict[str, torch.Tensor]:
            observations = super()._get_observations()
            self.extras["task_variant"] = "rj45-v1"
            self.extras["mating_frame"] = "measured_socket_entry_full_pose"
            self.extras["residual_action_frame"] = "measured_socket_entry"
            return observations

        def _get_dones(self):
            terminated, truncated = super()._get_dones()
            snapshot = self.extras.get("terminal_snapshot")
            if isinstance(snapshot, dict):
                snapshot["object_substep_poses_world_wxyz"] = torch.stack(self._physics_pose_samples, dim=1)
                snapshot["object_substep_velocities_world"] = torch.stack(self._physics_velocity_samples, dim=1)
                snapshot["object_substep_link_velocities_world"] = torch.stack(
                    self._physics_link_velocity_samples, dim=1
                )
                object_poses = snapshot["object_poses_world_wxyz"]
                snapshot["object_com_velocities_world"] = torch.stack((
                    self.socket.data.root_com_vel_w.clone(), self.plug.data.root_com_vel_w.clone(),
                ), dim=1)
                snapshot["object_link_velocities_world"] = torch.stack((
                    self.socket.data.root_link_vel_w.clone(), self.plug.data.root_link_vel_w.clone(),
                ), dim=1)
                snapshot["object_physx_velocities_world"] = torch.stack((
                    self.socket.root_physx_view.get_velocities().clone(), self.plug.root_physx_view.get_velocities().clone(),
                ), dim=1)
                snapshot["object_data_cache_times_s"] = torch.tensor([
                    [obj.data._sim_timestamp, obj.data._root_com_vel_w.timestamp]
                    for obj in (self.socket, self.plug)
                ], device=self.device)
                if not hasattr(self, "_previous_recorded_object_poses"):
                    self._previous_recorded_object_poses = object_poses.clone()
                    self._pose_difference_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                linear_difference = (object_poses[:, :, :3] - self._previous_recorded_object_poses[:, :, :3]) / self.step_dt
                angular_difference = quat_box_minus(
                    object_poses[:, :, 3:].reshape(-1, 4),
                    self._previous_recorded_object_poses[:, :, 3:].reshape(-1, 4),
                ).reshape(self.num_envs, 2, 3) / self.step_dt
                snapshot["object_pose_difference_velocity_available"] = self._pose_difference_valid.clone()
                snapshot["object_pose_difference_linear_velocity_m_s"] = linear_difference.clone()
                snapshot["object_pose_difference_angular_velocity_rad_s"] = angular_difference.clone()
                self._previous_recorded_object_poses[:] = object_poses
                self._pose_difference_valid[:] = True
                controlled = self._torso_joint_ids + self._arm_joint_ids
                snapshot["raw_joint_reference_rad"] = self._joint_targets[:, controlled].clone()
                snapshot["applied_joint_position_target_rad"] = self.robot.data.joint_pos_target[:, controlled].clone()
                snapshot["model_gravity_torque_nm"] = self.robot.root_physx_view.get_gravity_compensation_forces()[:, controlled].clone()
                if not hasattr(self, "_finger_pose_ids"):
                    self._finger_pose_ids, _ = self.robot.find_bodies(
                        ["ee_finger_l1", "ee_finger_l2", "ee_finger_r1", "ee_finger_r2"], preserve_order=True
                    )
                snapshot["finger_poses_world_wxyz"] = torch.cat((
                    self.robot.data.body_pos_w[:, self._finger_pose_ids],
                    self.robot.data.body_quat_w[:, self._finger_pose_ids],
                ), dim=2).clone()
                snapshot["rj45_evaluation_stability"] = {
                    key: value.clone()
                    for key, value in self.extras["rj45_evaluation_stability"].items()
                }
                snapshot["pregrasp_clearance_reached"] = self._pregrasp_lift_reached.clone()
                snapshot["pregrasp_above_reached"] = self._pregrasp_above_reached.clone()
            return terminated, truncated

        def rj45_task_report(self) -> dict[str, Any]:
            return {
                "scope": self.cfg.rj45_task["scope"],
                "parameter_provenance": self.cfg.rj45_task["parameter_provenance"],
                "mating_frame": "measured_socket_entry_full_pose",
                "residual_action_frame": "measured_socket_entry",
                "observation_dimension": self.cfg.observation_space,
                "torso_work_joint_positions_rad": self.cfg.rj45_task["controller"].get(
                    "torso_work_joint_positions_rad"
                ),
                "cross_hand_collision_monitor": (
                    "enabled_single_environment_wrapper"
                ),
                "external_forces_every_iteration": (
                    self._external_forces_every_iteration_enabled
                ),
                "evaluation_stability_limits": {
                    "max_depth_m": self.cfg.max_depth_m,
                    "max_relative_linear_speed_m_s": (
                        self.cfg.max_relative_linear_speed_m_s
                    ),
                    "max_relative_angular_speed_rad_s": (
                        self.cfg.max_relative_angular_speed_rad_s
                    ),
                    "required_continuous_dwell_s": self.cfg.success_dwell_s,
                    "relative_linear_velocity_definition": (
                        "tip velocity minus entry velocity minus socket angular velocity "
                        "cross tip-to-entry displacement"
                    ),
                    "relative_angular_velocity_definition": (
                        "plug root angular velocity minus socket root angular velocity"
                    ),
                },
                "pose_or_force_injection": False,
                "fixed_base": bool(self.robot.is_fixed_base),
                "gravity_tensor_shape": list(self._verified_gravity_shape),
                "num_joints": int(self.robot.num_joints),
            }

    Rby1Rj45Env.__name__ = "Rby1Rj45Env"
    from rby1_assembly.hand_contact_monitor import make_hand_contact_monitor_env_type
    return make_hand_contact_monitor_env_type(Rby1Rj45Env)


__all__ = ["configure_rj45_task", "load_rj45_task"]
