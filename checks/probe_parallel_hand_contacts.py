#!/usr/bin/env python3
"""Probe four-environment routing for selective RB-Y1 cross-hand contacts.

This is a diagnostic joint-command and collision-routing experiment.  It does
not execute or evaluate the RJ45 assembly task, and its outputs must not be
used as task-success evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
from typing import Any

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Probe four-environment cross-hand contact routing")
parser.add_argument(
    "--report",
    type=Path,
    default=Path("artifacts/rj45-parallel-probe/integral/report.json"),
)
parser.add_argument(
    "--trace",
    type=Path,
    default=Path("artifacts/rj45-parallel-probe/integral/trace.jsonl"),
)
parser.add_argument("--ramp-steps", type=int, default=720)
parser.add_argument("--hold-steps", type=int, default=720)
parser.add_argument("--negative-control-steps", type=int, default=30)
parser.add_argument("--minimum-contact-load-n", type=float, default=1.0)
parser.add_argument("--idle-load-tolerance-n", type=float, default=1.0e-4)
parser.add_argument("--translation-step-m", type=float, default=0.001)
parser.add_argument("--integral-gain", type=float, default=0.25)
parser.add_argument("--integral-max-delta-rad", type=float, default=0.03)
parser.add_argument("--trace-every", type=int, default=5)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
if args.ramp_steps < 120:
    parser.error("--ramp-steps must be at least 120 physics steps")
if args.hold_steps < 120:
    parser.error("--hold-steps must be at least 120 physics steps")
if args.negative_control_steps < 1:
    parser.error("--negative-control-steps must be positive")
if args.trace_every < 1:
    parser.error("--trace-every must be positive")
for option, value in (
    ("--minimum-contact-load-n", args.minimum_contact_load_n),
    ("--idle-load-tolerance-n", args.idle_load_tolerance_n),
    ("--translation-step-m", args.translation_step_m),
    ("--integral-gain", args.integral_gain),
    ("--integral-max-delta-rad", args.integral_max_delta_rad),
):
    if not math.isfinite(value) or value <= 0.0:
        parser.error(f"{option} must be finite and positive")
if args.translation_step_m > 0.001:
    parser.error("--translation-step-m must not exceed 0.001 m per physics step")
if args.integral_gain > 1.0:
    parser.error("--integral-gain must not exceed 1.0")
if args.integral_max_delta_rad > 0.03:
    parser.error("--integral-max-delta-rad must not exceed 0.03 rad")

launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import omni.kit.app  # noqa: E402
import torch  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from rby1_assembly.contact_loads import sum_contact_normal_loads  # noqa: E402
from rby1_assembly.hand_collision import (  # noqa: E402
    LEFT_HAND_BODY_PATHS,
    RIGHT_HAND_BODY_PATHS,
    configure_selective_hand_self_collisions,
)
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402
from rby1_assembly.rj45_env import bounded_joint_integral  # noqa: E402


NUM_ENVS = 4
ACTIVE_ENVS = (0, 2)
IDLE_ENVS = (1, 3)
REVOLUTE_JOINT_NAMES = (
    *(f"torso_{index}" for index in range(6)),
    *(f"left_arm_{index}" for index in range(7)),
    *(f"right_arm_{index}" for index in range(7)),
)
# Recorded terminal joint state from the existing scripted hand-collision
# diagnostic.  It is used only as a bounded position target for this routing
# probe; no pose, force, sensor buffer, or terminal state is injected.
COLLISION_JOINT_TARGET_RAD = (
    -5.970539859845303e-06,
    0.001325518125668168,
    0.0030678685288876295,
    0.00233440101146698,
    1.4421443665924016e-05,
    1.032749651130871e-06,
    -0.5161737203598022,
    -0.015944158658385277,
    -0.486809641122818,
    -1.452203392982483,
    0.23558466136455536,
    1.8701485395431519,
    0.6979767680168152,
    -0.5124624371528625,
    0.015747707337141037,
    0.4818427860736847,
    -1.4568977355957031,
    -0.23136283457279205,
    1.8724753856658936,
    -0.6922694444656372,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(value: Any, *, indent: int | None = None) -> str:
    return json.dumps(
        value,
        indent=indent,
        separators=None if indent else (",", ":"),
        allow_nan=False,
    )


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def _clamp_vector_norm(value: torch.Tensor, maximum_norm: float) -> torch.Tensor:
    """Clamp each row without changing its direction."""
    norm = torch.linalg.vector_norm(value, dim=1, keepdim=True)
    scale = (maximum_norm / norm.clamp_min(maximum_norm)).clamp(max=1.0)
    return value * scale


def _flatten_paths(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (tuple, list)):
        paths: list[str] = []
        for item in value:
            paths.extend(_flatten_paths(item))
        return paths
    raise TypeError(f"unexpected PhysX path container: {type(value).__name__}")


class ParallelHandContactProbeEnv(Rby1AssemblyEnv):
    """Base environment plus independently auditable batched hand sensors."""

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if self.cfg.scene.replicate_physics:
            raise RuntimeError("parallel hand-contact probe requires replicate_physics=False")
        if len(self.scene.env_prim_paths) != NUM_ENVS:
            raise RuntimeError(
                f"parallel hand-contact probe requires {NUM_ENVS} environment roots"
            )

        stage = get_current_stage()
        metadata = []
        for env_path in self.scene.env_prim_paths:
            metadata.append(
                configure_selective_hand_self_collisions(stage, f"{env_path}/Robot")
            )
        self.cfg.robot.spawn.articulation_props.enabled_self_collisions = True
        self.parallel_hand_collision_filter_metadata = metadata

        sensors: dict[str, ContactSensor] = {}
        for source_relative_path in LEFT_HAND_BODY_PATHS:
            sensor_cfg = ContactSensorCfg(
                prim_path=f"{self.scene.env_regex_ns}/Robot/{source_relative_path}",
                max_contact_data_count_per_prim=128,
                update_period=0.0,
                history_length=0,
                track_air_time=False,
                filter_prim_paths_expr=[
                    f"{self.scene.env_regex_ns}/Robot/{filter_relative_path}"
                    for filter_relative_path in RIGHT_HAND_BODY_PATHS
                ],
            )
            sensor = ContactSensor(sensor_cfg)
            sensors[source_relative_path] = sensor
            self.scene.sensors[
                f"parallel_cross_hand_{source_relative_path.replace('/', '_')}"
            ] = sensor
        self.parallel_cross_hand_sensors = sensors


def _source_routing(env: ParallelHandContactProbeEnv) -> dict[str, dict[str, Any]]:
    routing: dict[str, dict[str, Any]] = {}
    for relative_path, sensor in env.parallel_cross_hand_sensors.items():
        expected = [
            f"{env_path}/Robot/{relative_path}" for env_path in env.scene.env_prim_paths
        ]
        body_paths = _flatten_paths(sensor.body_physx_view.prim_paths)
        backend = getattr(sensor.body_physx_view, "_backend", None)
        if backend is None or not hasattr(backend, "prim_paths"):
            raise RuntimeError(f"PhysX backend source paths unavailable for {relative_path}")
        backend_paths = _flatten_paths(backend.prim_paths)
        contact_source_paths = _flatten_paths(sensor.contact_physx_view.sensor_paths)
        if body_paths != expected or backend_paths != expected or contact_source_paths != expected:
            raise RuntimeError(
                "independent source ordering was not proven for "
                f"{relative_path}: expected={expected}, body={body_paths}, "
                f"backend={backend_paths}, contact={contact_source_paths}"
            )

        filter_paths = _flatten_paths(sensor.contact_physx_view.filter_paths)
        filter_count = int(sensor.contact_physx_view.filter_count)
        expected_filter_paths = [
            f"{env_path}/Robot/{filter_relative_path}"
            for env_path in env.scene.env_prim_paths
            for filter_relative_path in RIGHT_HAND_BODY_PATHS
        ]
        if (
            filter_count != len(RIGHT_HAND_BODY_PATHS)
            or filter_paths != expected_filter_paths
        ):
            raise RuntimeError(
                "contact filter paths do not resolve in env-major, three-body order for "
                f"{relative_path}: expected={expected_filter_paths}, actual={filter_paths}"
            )
        routing[relative_path] = {
            "expected_source_paths": expected,
            "body_physx_view_prim_paths": body_paths,
            "body_physx_backend_prim_paths": backend_paths,
            "contact_physx_view_sensor_paths": contact_source_paths,
            "configured_filter_regex_count": len(RIGHT_HAND_BODY_PATHS),
            "expected_contact_physx_view_filter_paths": expected_filter_paths,
            "contact_physx_view_filter_paths": filter_paths,
            "body_count": int(sensor.body_physx_view.count),
            "contact_sensor_count": int(sensor.contact_physx_view.sensor_count),
            "contact_filter_count": filter_count,
        }
    if len(routing) != len(LEFT_HAND_BODY_PATHS):
        raise RuntimeError("not all three independent hand-contact sources were initialized")
    return routing


def _raw_loads(env: ParallelHandContactProbeEnv) -> tuple[torch.Tensor, dict[str, Any]]:
    per_source = []
    raw_shapes: dict[str, Any] = {}
    for relative_path, sensor in env.parallel_cross_hand_sensors.items():
        forces, points, normals, distances, counts, starts = (
            sensor.contact_physx_view.get_contact_data(dt=env.cfg.sim.dt)
        )
        load = sum_contact_normal_loads(forces, counts, starts, NUM_ENVS)
        if tuple(load.shape) != (NUM_ENVS,):
            raise RuntimeError(
                f"raw load shape is not ({NUM_ENVS},) for {relative_path}: {load.shape}"
            )
        per_source.append(load)
        raw_shapes[relative_path] = {
            "forces": list(forces.shape),
            "points": list(points.shape),
            "normals": list(normals.shape),
            "distances": list(distances.shape),
            "counts": list(counts.shape),
            "starts": list(starts.shape),
            "aggregated_load": list(load.shape),
        }
    total = torch.stack(per_source, dim=1).sum(dim=1)
    if tuple(total.shape) != (NUM_ENVS,) or not torch.isfinite(total).all():
        raise RuntimeError(f"invalid combined cross-hand load: shape={total.shape}, value={total}")
    return total, raw_shapes


def _physics_step(env: ParallelHandContactProbeEnv, target: torch.Tensor) -> None:
    env.robot.set_joint_position_target(target)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(env.cfg.sim.dt)


def _gravity_biased_drive_target(
    env: ParallelHandContactProbeEnv,
    reference: torch.Tensor,
    controlled_joint_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = env.robot.root_physx_view.get_gravity_compensation_forces()
    stiffness = env.robot.data.joint_stiffness[:, controlled_joint_ids]
    if not torch.isfinite(gravity).all() or torch.any(stiffness <= 0.0):
        raise RuntimeError("gravity-biased drive target requires finite gravity and positive stiffness")
    drive_target = reference.clone()
    drive_target[:, controlled_joint_ids] += gravity[:, controlled_joint_ids] / stiffness
    limits = env.robot.data.soft_joint_pos_limits
    drive_target = torch.maximum(
        torch.minimum(drive_target, limits[:, :, 1]), limits[:, :, 0]
    )
    return drive_target, gravity


def _write_report(report: dict[str, Any]) -> None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(_json_dump(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)


def _failure_report(error: BaseException) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "passed": False,
        "experiment": "rb_y1_parallel_cross_hand_contact_routing_probe",
        "scope": "diagnostic contact routing only; not RJ45 assembly task-success evidence",
        "physics_executed": True,
        "num_envs": NUM_ENVS,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = NUM_ENVS
    cfg.scene.replicate_physics = False
    cfg.sim.device = args.device
    cfg.seed = 20261026
    cfg.render_background = False

    env: ParallelHandContactProbeEnv | None = None
    trace_file = None
    active_exception = False
    try:
        env = ParallelHandContactProbeEnv(cfg)
        env.reset(seed=cfg.seed)
        routing = _source_routing(env)
        q_reference = env.robot.data.default_joint_pos.clone()
        revolute_ids, resolved_names = env.robot.find_joints(
            list(REVOLUTE_JOINT_NAMES), preserve_order=True
        )
        if len(revolute_ids) != len(REVOLUTE_JOINT_NAMES) or resolved_names != list(
            REVOLUTE_JOINT_NAMES
        ):
            raise RuntimeError(
                f"failed to resolve ordered revolute joints: ids={revolute_ids}, names={resolved_names}"
            )
        collision_target = torch.tensor(
            COLLISION_JOINT_TARGET_RAD,
            device=env.device,
            dtype=q_reference.dtype,
        )
        if collision_target.numel() != len(revolute_ids):
            raise RuntimeError("collision joint target does not match resolved revolute joints")
        controlled_joint_ids = list(revolute_ids)

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        trace_file = args.trace.open("w", encoding="utf-8", buffering=1)
        global_step = 0

        reset_loads, reset_raw_shapes = _raw_loads(env)
        left_palm_ids, left_palm_names = env.robot.find_bodies("ee_left", preserve_order=True)
        right_palm_ids, right_palm_names = env.robot.find_bodies("ee_right", preserve_order=True)
        if (
            len(left_palm_ids) != 1
            or left_palm_names != ["ee_left"]
            or len(right_palm_ids) != 1
            or right_palm_names != ["ee_right"]
        ):
            raise RuntimeError(
                f"could not resolve exact palm bodies: left={left_palm_names}, right={right_palm_names}"
            )
        hand_body_names_expected = [
            "ee_left",
            "ee_finger_l1",
            "ee_finger_l2",
            "ee_right",
            "ee_finger_r1",
            "ee_finger_r2",
        ]
        hand_body_ids, hand_body_names = env.robot.find_bodies(
            hand_body_names_expected, preserve_order=True
        )
        if len(hand_body_ids) != 6 or hand_body_names != hand_body_names_expected:
            raise RuntimeError(
                f"could not resolve ordered six hand bodies: {hand_body_names}"
            )
        env0 = torch.tensor([0], dtype=torch.long, device=env.device)
        cross_environment_translation = (
            env.robot.data.body_pos_w[1, right_palm_ids[0]]
            - env.robot.data.body_pos_w[0, left_palm_ids[0]]
        )
        overlap_root_pose = torch.cat(
            (env.robot.data.root_pos_w[[0]], env.robot.data.root_quat_w[[0]]), dim=1
        ).clone()
        overlap_root_pose[:, :3] += cross_environment_translation
        env.robot.write_root_pose_to_sim(overlap_root_pose, env0)
        env.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env0)
        negative_peak = torch.zeros(NUM_ENVS, device=env.device)
        for negative_step in range(1, args.negative_control_steps + 1):
            negative_drive_target, _ = _gravity_biased_drive_target(
                env, q_reference, controlled_joint_ids
            )
            _physics_step(env, negative_drive_target)
            global_step += 1
            loads, _ = _raw_loads(env)
            negative_peak = torch.maximum(negative_peak, loads)
            if negative_step == 1 or negative_step == args.negative_control_steps:
                trace_file.write(
                    _json_dump(
                        {
                            "global_physics_step": global_step,
                            "phase": "cross_environment_root_overlap_negative_control",
                            "phase_step": negative_step,
                            "cross_hand_contact_load_n": _tensor_list(loads),
                        }
                    )
                    + "\n"
                )

        env.reset(seed=cfg.seed)
        q_reference = env.robot.data.default_joint_pos.clone()
        active_peak = torch.zeros(NUM_ENVS, device=env.device)
        active_indices = torch.tensor(ACTIVE_ENVS, dtype=torch.long, device=env.device)
        idle_indices = torch.tensor(IDLE_ENVS, dtype=torch.long, device=env.device)
        active_mask = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        active_mask[active_indices] = True
        contact_achieved = torch.zeros(NUM_ENVS, dtype=torch.bool, device=env.device)
        first_contact_step = torch.full(
            (NUM_ENVS,), -1, dtype=torch.long, device=env.device
        )
        integral_reference = env.robot.data.joint_pos[:, env._arm_joint_ids].clone()
        last_integral_current = integral_reference.clone()
        last_integral_headroom = torch.zeros_like(integral_reference)
        last_drive_target = q_reference.clone()
        last_raw_shapes = reset_raw_shapes
        total_command_steps = args.ramp_steps + args.hold_steps
        for command_step in range(1, total_command_steps + 1):
            alpha = min(command_step / args.ramp_steps, 1.0)
            target_reference = q_reference.clone()
            if command_step <= args.ramp_steps:
                for env_index in ACTIVE_ENVS:
                    target_reference[env_index, revolute_ids] = torch.lerp(
                        q_reference[env_index, revolute_ids], collision_target, alpha
                    )
                integral_reference[:] = target_reference[:, env._arm_joint_ids]
            else:
                current = env.robot.data.joint_pos
                for env_index in ACTIVE_ENVS:
                    target_reference[env_index, revolute_ids[:6]] = collision_target[:6]
                left_pos = env.robot.data.body_pos_w[:, env._left_ee_body_id]
                right_pos = env.robot.data.body_pos_w[:, env._right_ee_body_id]
                midpoint = 0.5 * (left_pos + right_pos)
                left_twist = torch.zeros((NUM_ENVS, 6), device=env.device)
                right_twist = torch.zeros((NUM_ENVS, 6), device=env.device)
                left_twist[:, :3] = _clamp_vector_norm(
                    midpoint - left_pos, args.translation_step_m
                )
                right_twist[:, :3] = _clamp_vector_norm(
                    midpoint - right_pos, args.translation_step_m
                )
                jacobians = env.robot.root_physx_view.get_jacobians()
                left_jacobian = jacobians[
                    :, env._left_ee_jacobian_id, :, env._left_arm_joint_ids
                ]
                right_jacobian = jacobians[
                    :, env._right_ee_jacobian_id, :, env._right_arm_joint_ids
                ]
                left_delta = env._damped_least_squares(left_jacobian, left_twist)
                right_delta = env._damped_least_squares(right_jacobian, right_twist)
                delta = torch.cat((left_delta, right_delta), dim=1)
                arm_current = current[:, env._arm_joint_ids]
                last_integral_current = arm_current.clone()
                arm_stiffness = env.robot.data.joint_stiffness[:, env._arm_joint_ids]
                arm_gravity = env.robot.root_physx_view.get_gravity_compensation_forces()[
                    :, env._arm_joint_ids
                ]
                arm_effort_limits = env.robot.data.joint_effort_limits[:, env._arm_joint_ids]
                arm_position_limits = env.robot.data.soft_joint_pos_limits[
                    :, env._arm_joint_ids
                ]
                candidate_reference = bounded_joint_integral(
                    integral_reference,
                    arm_current,
                    delta,
                    arm_stiffness,
                    arm_gravity,
                    arm_effort_limits,
                    arm_position_limits,
                    args.integral_gain,
                    args.integral_max_delta_rad,
                )
                servo_enabled = active_mask & ~contact_achieved
                integral_reference = torch.where(
                    servo_enabled[:, None], candidate_reference, integral_reference
                )
                last_integral_headroom = (
                    (arm_effort_limits - arm_gravity.abs()).clamp_min(0.0)
                    / arm_stiffness
                ).clamp(max=args.integral_max_delta_rad)
                for env_index in ACTIVE_ENVS:
                    target_reference[env_index, env._arm_joint_ids] = integral_reference[
                        env_index
                    ]
            last_drive_target, _ = _gravity_biased_drive_target(
                env, target_reference, controlled_joint_ids
            )
            _physics_step(env, last_drive_target)
            global_step += 1
            loads, last_raw_shapes = _raw_loads(env)
            active_peak = torch.maximum(active_peak, loads)
            new_contacts = active_mask & ~contact_achieved & (
                loads >= args.minimum_contact_load_n
            )
            first_contact_step[new_contacts] = command_step
            contact_achieved |= new_contacts
            if (
                command_step == 1
                or command_step == args.ramp_steps
                or command_step == total_command_steps
                or command_step % args.trace_every == 0
                or bool(torch.any(new_contacts).item())
            ):
                trace_file.write(
                    _json_dump(
                        {
                            "global_physics_step": global_step,
                            "phase": "joint_command_ramp"
                            if command_step <= args.ramp_steps
                            else "joint_command_hold",
                            "phase_step": command_step,
                            "command_alpha": alpha,
                            "cross_hand_contact_load_n": _tensor_list(loads),
                            "wrist_separation_m": _tensor_list(
                                torch.linalg.vector_norm(
                                    env.robot.data.body_pos_w[:, env._left_ee_body_id]
                                    - env.robot.data.body_pos_w[:, env._right_ee_body_id],
                                    dim=1,
                                )
                            ),
                            "revolute_joint_position_rad": _tensor_list(
                                env.robot.data.joint_pos[:, revolute_ids]
                            ),
                            "left_ee_pose_world_wxyz": _tensor_list(
                                torch.cat(
                                    (
                                        env.robot.data.body_pos_w[:, env._left_ee_body_id],
                                        env.robot.data.body_quat_w[:, env._left_ee_body_id],
                                    ),
                                    dim=1,
                                )
                            ),
                            "right_ee_pose_world_wxyz": _tensor_list(
                                torch.cat(
                                    (
                                        env.robot.data.body_pos_w[:, env._right_ee_body_id],
                                        env.robot.data.body_quat_w[:, env._right_ee_body_id],
                                    ),
                                    dim=1,
                                )
                            ),
                            "finger_joint_position_m": _tensor_list(
                                env.robot.data.joint_pos[:, env._gripper_joint_ids]
                            ),
                            "hand_body_names": hand_body_names,
                            "hand_body_pose_world_wxyz": _tensor_list(
                                torch.cat(
                                    (
                                        env.robot.data.body_pos_w[:, hand_body_ids],
                                        env.robot.data.body_quat_w[:, hand_body_ids],
                                    ),
                                    dim=2,
                                )
                            ),
                            "stored_integral_reference_arm_rad": _tensor_list(
                                integral_reference
                            ),
                            "integral_headroom_arm_rad": _tensor_list(
                                last_integral_headroom
                            ),
                            "drive_target_revolute_rad": _tensor_list(
                                last_drive_target[:, revolute_ids]
                            ),
                            "contact_achieved": _tensor_list(contact_achieved),
                        }
                    )
                    + "\n"
                )

        final_positions = env.robot.data.joint_pos[:, revolute_ids].clone()
        trace_file.flush()
        trace_file.close()
        trace_file = None
        active_motion = torch.linalg.vector_norm(
            final_positions[active_indices] - q_reference[active_indices][:, revolute_ids], dim=1
        )
        idle_motion = torch.linalg.vector_norm(
            final_positions[idle_indices] - q_reference[idle_indices][:, revolute_ids], dim=1
        )
        arm_limits = env.robot.data.soft_joint_pos_limits[:, env._arm_joint_ids]
        integral_offset = (integral_reference - last_integral_current).abs()

        checks = {
            "four_environments": env.num_envs == NUM_ENVS,
            "fast_physics_replication_disabled": not cfg.scene.replicate_physics,
            "three_independent_regex_sources": len(routing) == 3,
            "three_regex_filters_per_source": all(
                entry["configured_filter_regex_count"] == 3
                and entry["contact_filter_count"] == 3
                and len(entry["contact_physx_view_filter_paths"]) == NUM_ENVS * 3
                for entry in routing.values()
            ),
            "source_order_matches_scene_environment_order": True,
            "all_four_env_loads_are_finite": bool(torch.isfinite(active_peak).all().item()),
            "stored_integral_reference_is_finite": bool(
                torch.isfinite(integral_reference).all().item()
            ),
            "drive_target_is_finite": bool(torch.isfinite(last_drive_target).all().item()),
            "integral_headroom_is_bounded": bool(
                torch.all(last_integral_headroom >= 0.0).item()
                and torch.all(
                    last_integral_headroom <= args.integral_max_delta_rad + 1.0e-7
                ).item()
                and torch.all(
                    integral_offset[active_indices]
                    <= last_integral_headroom[active_indices] + 1.0e-5
                ).item()
            ),
            "stored_integral_reference_respects_soft_limits": bool(
                torch.all(integral_reference >= arm_limits[:, :, 0] - 1.0e-6).item()
                and torch.all(integral_reference <= arm_limits[:, :, 1] + 1.0e-6).item()
            ),
            "env0_and_env2_receive_real_joint_motion": bool(
                torch.all(active_motion > 0.1).item()
            ),
            "env1_and_env3_remain_near_reset_joint_command": bool(
                torch.all(idle_motion < 5.0e-2).item()
            ),
            "env0_and_env2_detect_cross_hand_contact": bool(
                torch.all(active_peak[active_indices] >= args.minimum_contact_load_n).item()
            ),
            "env1_and_env3_cross_hand_load_remains_zero": bool(
                torch.all(active_peak[idle_indices] <= args.idle_load_tolerance_n).item()
            ),
            "reset_cross_hand_load_is_zero": bool(
                torch.all(reset_loads <= args.idle_load_tolerance_n).item()
            ),
            "cross_environment_overlap_negative_control_is_zero": bool(
                torch.all(negative_peak <= args.idle_load_tolerance_n).item()
            ),
            "each_environment_has_nine_allowed_hand_pairs": all(
                item["allowed_cross_hand_pair_count"] == 9
                for item in env.parallel_hand_collision_filter_metadata
            ),
        }
        script_path = Path(__file__).resolve()
        source_paths = (
            script_path,
            PROJECT_ROOT / "src/rby1_assembly/isaac_env.py",
            PROJECT_ROOT / "src/rby1_assembly/hand_collision.py",
            PROJECT_ROOT / "src/rby1_assembly/contact_loads.py",
            PROJECT_ROOT / "src/rby1_assembly/rj45_env.py",
        )
        exemplar_sensor = next(iter(env.parallel_cross_hand_sensors.values()))
        contact_sensor_source = Path(
            sys.modules[type(exemplar_sensor).__module__].__file__  # type: ignore[arg-type]
        ).resolve()
        physics_tensor_source = Path(
            sys.modules[type(exemplar_sensor.body_physx_view).__module__].__file__  # type: ignore[arg-type]
        ).resolve()
        report = {
            "schema_version": 1,
            "passed": all(checks.values()),
            "experiment": "rb_y1_parallel_cross_hand_contact_routing_probe",
            "scope": (
                "Diagnostic joint-command and contact-sensor routing probe only; no RJ45 task "
                "execution, policy evaluation, task success, sensor-value injection, or force injection."
            ),
            "physics_executed": True,
            "device": args.device,
            "num_envs": NUM_ENVS,
            "scene": {
                "environment_prim_paths": list(env.scene.env_prim_paths),
                "environment_spacing_m": float(cfg.scene.env_spacing),
                "replicate_physics": bool(cfg.scene.replicate_physics),
                "collision_filter_prim": "/World/collisions",
                "collision_groups_preserved_from_base_environment": True,
            },
            "sensor_routing": routing,
            "raw_contact_buffer_shapes": last_raw_shapes,
            "post_run_validation_required": {
                "kind": "authored_collision_mesh_convex_sat",
                "input": "a recorded trace row at first contact or minimum hand separation",
                "required_result": "at least one of the nine allowed hand-body pairs overlaps in each active environment",
                "status": "not_run_by_this_gpu_probe",
            },
            "negative_control": {
                "kind": "test_only_env0_articulation_root_over_env1_reset_pose_then_reset",
                "steps": args.negative_control_steps,
                "env0_left_palm_to_env1_right_palm_translation_m": _tensor_list(
                    cross_environment_translation
                ),
                "env0_overlap_root_pose_world_wxyz": _tensor_list(overlap_root_pose[0]),
                "peak_cross_hand_contact_load_n": _tensor_list(negative_peak),
                "pose_force_sensor_or_terminal_state_injection": False,
                "root_pose_injection": True,
            },
            "joint_command": {
                "active_environment_indices": list(ACTIVE_ENVS),
                "idle_environment_indices": list(IDLE_ENVS),
                "joint_names": list(REVOLUTE_JOINT_NAMES),
                "target_rad": list(COLLISION_JOINT_TARGET_RAD),
                "target_provenance": "docs/evidence/failure-cases.json hand_collision terminal_state",
                "ramp_steps": args.ramp_steps,
                "hold_steps": args.hold_steps,
                "hold_command": (
                    "stored-reference bounded joint integral driven by midpoint DLS translation error"
                ),
                "translation_twist_norm_limit_m_per_physics_step": args.translation_step_m,
                "integral_gain": args.integral_gain,
                "integral_max_delta_from_current_rad": args.integral_max_delta_rad,
                "integral_anti_windup": (
                    "reference is clamped to current +/- min((effort_limit - "
                    "abs(gravity_compensation)) / stiffness, max_delta), then to soft limits"
                ),
                "gravity_biased_drive_target": (
                    "stored reference + gravity_compensation / reported joint stiffness, "
                    "clamped to soft limits"
                ),
                "active_joint_motion_norm_rad": _tensor_list(active_motion),
                "idle_joint_motion_norm_rad": _tensor_list(idle_motion),
                "peak_cross_hand_contact_load_n": _tensor_list(active_peak),
                "first_contact_command_step": _tensor_list(first_contact_step),
                "contact_achieved": _tensor_list(contact_achieved),
                "final_integral_reference_arm_rad": _tensor_list(integral_reference),
                "final_integral_headroom_arm_rad": _tensor_list(last_integral_headroom),
                "final_drive_target_revolute_rad": _tensor_list(
                    last_drive_target[:, revolute_ids]
                ),
                "hand_body_names": hand_body_names,
                "minimum_contact_load_n": args.minimum_contact_load_n,
                "idle_load_tolerance_n": args.idle_load_tolerance_n,
            },
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
            "provenance": {
                "source_sha256": {
                    str(path.relative_to(PROJECT_ROOT)): _sha256(path) for path in source_paths
                },
                "runtime_source_sha256": {
                    str(contact_sensor_source): _sha256(contact_sensor_source),
                    str(physics_tensor_source): _sha256(physics_tensor_source),
                },
                "trace_path": str(args.trace),
                "trace_sha256": _sha256(args.trace),
                "trace_bytes": args.trace.stat().st_size,
            },
            "interpretation_limits": [
                "The probe demonstrates tensor routing and isolation for this exact four-environment scene.",
                "The commanded collisions are scripted diagnostics, not learned-policy failures.",
                "Convex SAT overlap must be evaluated offline from a recorded trace row and authored hand collision meshes; contact load alone is not reported as a SAT result.",
                "Passing does not establish RJ45 insertion success or real-robot transfer performance.",
            ],
        }
        _write_report(report)
        output = _json_dump(report, indent=2)
        print(output, file=sys.stdout if report["passed"] else sys.stderr)
        return 0 if report["passed"] else 2
    except BaseException:
        active_exception = True
        raise
    finally:
        if trace_file is not None:
            trace_file.close()
        if env is not None:
            try:
                env.close()
            except Exception:
                print("Failed to close environment", file=sys.stderr, flush=True)
                traceback.print_exc()
                if not active_exception:
                    raise


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as error:
        traceback.print_exc()
        sys.stderr.flush()
        try:
            _write_report(_failure_report(error))
        except Exception:
            print("Failed to write probe failure report", file=sys.stderr, flush=True)
            traceback.print_exc()
        exit_code = 1
    finally:
        try:
            omni.kit.app.get_app().post_quit(exit_code)
        except Exception:
            print("Failed to set Isaac Kit exit code", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
        try:
            simulation_app.close()
        except Exception:
            print("Failed to close Isaac application", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
    raise SystemExit(exit_code)
