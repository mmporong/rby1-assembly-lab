# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch the smallest real-physics smoke run for the assembly environment."""

import argparse
from contextlib import nullcontext
import hashlib
from importlib import metadata
import json
import math
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.run_provenance import (  # noqa: E402
    InputMutationError,
    RunInputSnapshot,
    capture_run_inputs,
)


def _preimport_option(name: str, default: str) -> str:
    prefix = name + "="
    for index, value in enumerate(sys.argv[1:]):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 2 <= len(sys.argv[1:]):
            return sys.argv[1:][index + 1]
    return default


def _preimport_has_option(name: str) -> bool:
    prefix = name + "="
    return any(value == name or value.startswith(prefix) for value in sys.argv[1:])


_bootstrap_task = _preimport_option("--task", "toy")
_bootstrap_task_path = PROJECT_ROOT / (
    "configs/tasks/rj45-v1.json" if _bootstrap_task == "rj45" else "configs/task.json"
)
_bootstrap_input_snapshot = capture_run_inputs(
    PROJECT_ROOT,
    task_path=_bootstrap_task_path,
    build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
    config_paths=(
        PROJECT_ROOT / "configs/assets.json",
        PROJECT_ROOT / "configs/runtime.json",
        *((PROJECT_ROOT / "configs/rj45-assets.json",) if _bootstrap_task == "rj45" else ()),
        *((Path(_preimport_option("--command-fault-config", "configs/failure-scenarios.json")),)
          if _preimport_has_option("--command-fault") else ()),
    ),
    source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
    source_files=(Path(__file__).resolve(),),
)

from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="RB-Y1 assembly PhysX smoke run")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--task", choices=("toy", "rj45"), default="toy")
parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="control step 수. 생략하면 task timeout/control dt의 올림값을 사용",
)
parser.add_argument("--seed", type=int, help="Override configs/task.json data.train_seed.")
parser.add_argument("--report", type=Path, default=Path("artifacts/smoke_report.json"))
parser.add_argument("--trace", type=Path, help="Write sampled controller and physics state as JSONL.")
parser.add_argument("--trace-every", type=int, help="Trace every N control steps; default 10, or 1 for command-fault tests.")
parser.add_argument("--command-fault", choices=("right_gripper_open", "insertion_lateral_disturbance", "hand_collision"))
parser.add_argument("--command-fault-config", type=Path, default=Path("configs/failure-scenarios.json"))
parser.add_argument("--video", type=Path, help="Write the rendered single-environment run as a video.")
parser.add_argument("--video-fps", type=int, choices=(30, 60, 120), default=120,
                    help="Native rendered physics frames per simulation second; controller stays at 30 Hz.")
parser.add_argument(
    "--camera-view",
    choices=("overview", "assembly", "connector"),
    default="overview",
    help="Video camera preset.",
)
parser.add_argument("--require-completion", action="store_true")
parser.add_argument("--background", action="store_true", help="Load the visual-only Isaac Simple Room once.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.num_envs <= 0 or (args.steps is not None and args.steps <= 0):
    parser.error("--num-envs must be positive and --steps, when set, must be positive")
if args.video is not None and args.num_envs != 1:
    parser.error("--video requires --num-envs 1")
if args.video is not None and not getattr(args, "enable_cameras", False):
    parser.error("--video requires --enable_cameras")
if args.trace_every is not None and args.trace_every < 1:
    parser.error("--trace-every must be positive")
if args.command_fault is not None and (args.num_envs != 1 or args.trace is None or args.trace_every not in (None, 1)):
    parser.error("--command-fault requires one environment and --trace at every control step")
if args.task == "rj45" and args.command_fault is not None:
    parser.error("legacy command-fault demonstrations require --task toy")
trace_every = args.trace_every or (1 if args.command_fault or args.task == "rj45" else 10)
launcher = AppLauncher(args)
simulation_app = launcher.app

import torch  # noqa: E402
import omni.kit.app  # noqa: E402

from rby1_assembly.contracts import ACTION_DIM  # noqa: E402
from rby1_assembly.experiment_contracts import atomic_write_json  # noqa: E402
from rby1_assembly.video_capture import PhysicsFrameCapture  # noqa: E402
from rby1_assembly.isaac_env import (  # noqa: E402
    AssemblyPhase,
    FORBIDDEN_CONTACT_LOAD_SOURCE,
    Rby1AssemblyEnv,
    Rby1AssemblyEnvCfg,
)


_effective_steps_for_report: int | None = None
_input_snapshot_for_failure_report: RunInputSnapshot | None = None


def _tensor_list(value: torch.Tensor):
    return value.detach().cpu().tolist()


def _tensor_json_value(value: torch.Tensor):
    value = value.detach().cpu()
    return value.item() if value.numel() == 1 else value.tolist()


def _forbidden_contact_breakdown(
    body_names: tuple[str, ...] | list[str],
    robot_forces_w: torch.Tensor,
    palm_forces_w: torch.Tensor,
) -> dict:
    robot_magnitudes = torch.linalg.vector_norm(robot_forces_w, dim=-1)
    count = min(3, robot_magnitudes.numel())
    top_values, top_indices = torch.topk(robot_magnitudes, k=count)

    return {
        "robot_body_force_top3_n": [
            {
                "body_name": body_names[int(index.item())],
                "normal_force_magnitude_n": float(value.item()),
            }
            for value, index in zip(top_values, top_indices, strict=True)
        ],
        "left_palm_normal_force_magnitude_n": float(
            torch.linalg.vector_norm(palm_forces_w[0], dim=-1).max().item()
        ),
        "right_palm_normal_force_magnitude_n": float(
            torch.linalg.vector_norm(palm_forces_w[1], dim=-1).max().item()
        ),
        "force_definition": "net_normal_contact_force_world_frame",
    }


def _live_forbidden_contact_breakdown(env: Rby1AssemblyEnv, env_index: int) -> dict:
    robot_forces = env.forbidden_robot_contact.data.net_forces_w
    left_palm_forces = env.forbidden_left_palm_contact.data.net_forces_w
    right_palm_forces = env.forbidden_right_palm_contact.data.net_forces_w
    if robot_forces is None:
        raise RuntimeError("forbidden robot contact sensor has no net force data")
    if left_palm_forces is None or right_palm_forces is None:
        raise RuntimeError("forbidden palm contact sensor has no net force data")
    palm_forces = torch.stack(
        (left_palm_forces[env_index], right_palm_forces[env_index]),
        dim=0,
    )
    return _forbidden_contact_breakdown(
        env.forbidden_robot_contact.body_names,
        robot_forces[env_index],
        palm_forces,
    )


def _trace_record(
    env: Rby1AssemblyEnv,
    control_step: int,
    events: list[str],
    snapshot: dict | None = None,
) -> dict:
    if snapshot is None:
        phase = env._phase.clone()
        terminal_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        timestamp_s = env.episode_length_buf.float() * env.step_dt
        retry_count = env._retry_count.clone()
        joint_positions = env.robot.data.joint_pos[:, env._torso_joint_ids + env._arm_joint_ids]
        gripper_positions = env.robot.data.joint_pos[:, env._gripper_joint_ids]
        object_poses = torch.stack(
            (
                torch.cat((env.socket.data.root_pos_w, env.socket.data.root_quat_w), dim=1),
                torch.cat((env.plug.data.root_pos_w, env.plug.data.root_quat_w), dim=1),
            ),
            dim=1,
        )
        left_l1_force = env._filtered_force_magnitude(env.left_contact_l1)
        left_l2_force = env._filtered_force_magnitude(env.left_contact_l2)
        right_r1_force = env._filtered_force_magnitude(env.right_contact_r1)
        right_r2_force = env._filtered_force_magnitude(env.right_contact_r2)
        contact_force = torch.stack(
            (torch.maximum(left_l1_force, left_l2_force), torch.maximum(right_r1_force, right_r2_force)),
            dim=1,
        )
        grip_proven = env._evaluation_grip_proven.clone()
        grip_live = torch.stack(
            (
                (left_l1_force >= env.cfg.grip_contact_force_n)
                & (left_l2_force >= env.cfg.grip_contact_force_n),
                (right_r1_force >= env.cfg.grip_contact_force_n)
                & (right_r2_force >= env.cfg.grip_contact_force_n),
            ),
            dim=1,
        )
        relative_pos, relative_quat = env._relative_object_pose()
        depth = relative_pos[:, 1]
        lateral = torch.linalg.vector_norm(relative_pos[:, (0, 2)], dim=1)
        orientation = 2.0 * torch.acos(torch.clamp(torch.abs(relative_quat[:, 0]), max=1.0))
        reason_codes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        reason_names = {0: "none"}
        revolute_joint_names = tuple(
            [
                *[f"torso_{index}" for index in range(6)],
                *[f"left_arm_{index}" for index in range(7)],
                *[f"right_arm_{index}" for index in range(7)],
            ]
        )
        gripper_joint_names = (
            "gripper_finger_l1",
            "gripper_finger_l2",
            "gripper_finger_r1",
            "gripper_finger_r2",
        )
        snapshot_ee_poses = None
        snapshot_forbidden_body_names = None
        snapshot_forbidden_robot_forces = None
        snapshot_forbidden_palm_forces = None
        insertion_contact_force_n = getattr(env, "_last_insertion_contact_force_n", None)
        insertion_contact_force_w = getattr(env, "_last_insertion_contact_force_w", None)
        insertion_contact_force_peak_n = getattr(
            env, "_episode_insertion_contact_force_peak_n", None
        )
        insertion_contact_force_source = None
        insertion_contact_load_n = getattr(env, "_last_insertion_contact_load_n", None)
        insertion_contact_load_peak_n = getattr(env, "_episode_insertion_contact_load_peak_n", None)
        insertion_contact_load_source = None
        forbidden_contact_load_n = env._last_forbidden_contact_load_n
        forbidden_contact_load_peak_n = env._episode_forbidden_contact_load_peak_n
        forbidden_contact_load_source = FORBIDDEN_CONTACT_LOAD_SOURCE
        continuous_grasp_loss_since_s = env._grasp_loss_since_s
        insertion_command_depth_m = env._insertion_command_depth_m
        insertion_advance_enabled = env._insertion_advance_enabled
        evaluation_dwell_s = env._evaluation_dwell_s
        evaluation_success = env._evaluation_success
        insert_joint_target_error_rad = (
            env._insert_joint_targets - env.robot.data.joint_pos[:, env._arm_joint_ids]
        )
        grip_contact_force_peak_n = getattr(env, "_episode_grip_force_peak_n", None)
        finger_contact_force_n = None
        finger_non_target_force_w = None
    else:
        phase = snapshot["phase"]
        terminal_mask = snapshot["terminal_mask"]
        timestamp_s = snapshot["timestamp_s"]
        retry_count = snapshot["retry_count"]
        joint_positions = snapshot["revolute_joint_positions_rad"]
        gripper_positions = snapshot["gripper_positions_m"]
        object_poses = snapshot["object_poses_world_wxyz"]
        contact_force = snapshot["contact_force_proxy_n"]
        grip_proven = snapshot["grip_proven"]
        grip_live = snapshot["grip_live"]
        depth = snapshot["actual_depth_m"]
        lateral = snapshot["actual_lateral_error_m"]
        orientation = snapshot["actual_orientation_error_rad"]
        reason_codes = snapshot["termination_reason_code"]
        reason_names = snapshot["termination_reason_by_code"]
        revolute_joint_names = snapshot["revolute_joint_names"]
        gripper_joint_names = snapshot["gripper_joint_names"]
        snapshot_ee_poses = snapshot.get("ee_poses_world_wxyz")
        snapshot_forbidden_body_names = snapshot.get("forbidden_robot_body_names")
        snapshot_forbidden_robot_forces = snapshot.get("forbidden_robot_forces_w")
        snapshot_forbidden_palm_forces = snapshot.get("forbidden_palm_forces_w")
        insertion_contact_force_n = snapshot.get("insertion_contact_force_n")
        insertion_contact_force_w = snapshot.get("insertion_contact_force_w")
        insertion_contact_force_peak_n = snapshot.get("insertion_contact_force_peak_n")
        insertion_contact_force_source = snapshot.get("insertion_contact_force_source")
        insertion_contact_load_n = snapshot.get("insertion_contact_load_n")
        insertion_contact_load_peak_n = snapshot.get("insertion_contact_load_peak_n")
        insertion_contact_load_source = snapshot.get("insertion_contact_load_source")
        forbidden_contact_load_n = snapshot["forbidden_contact_load_n"]
        forbidden_contact_load_peak_n = snapshot["forbidden_contact_load_peak_n"]
        forbidden_contact_load_source = snapshot["forbidden_contact_load_source"]
        continuous_grasp_loss_since_s = snapshot["continuous_grasp_loss_since_s"]
        insertion_command_depth_m = snapshot["insertion_command_depth_m"]
        insertion_advance_enabled = snapshot["insertion_advance_enabled"]
        evaluation_dwell_s = snapshot["independent_eval_dwell_s"]
        evaluation_success = snapshot["independent_eval_success"]
        insert_joint_target_error_rad = snapshot["insert_joint_target_error_rad"]
        grip_contact_force_peak_n = snapshot.get("grip_contact_force_peak_n")
        finger_contact_force_n = snapshot.get("finger_contact_force_n")
        finger_non_target_force_w = snapshot.get("finger_non_target_force_w")

    left_ee_pos = env.robot.data.body_pos_w[:, env._left_ee_body_id]
    left_ee_quat = env.robot.data.body_quat_w[:, env._left_ee_body_id]
    right_ee_pos = env.robot.data.body_pos_w[:, env._right_ee_body_id]
    right_ee_quat = env.robot.data.body_quat_w[:, env._right_ee_body_id]
    phase_names = {int(item): item.name.lower() for item in AssemblyPhase}
    environments = []
    for env_index in range(env.num_envs):
        is_terminal = bool(terminal_mask[env_index].item())
        has_snapshot_forbidden = (
            snapshot_forbidden_body_names is not None
            and snapshot_forbidden_robot_forces is not None
            and snapshot_forbidden_palm_forces is not None
        )
        if snapshot_ee_poses is not None:
            ee_poses = {
                "left_world_wxyz": _tensor_list(snapshot_ee_poses[env_index, 0]),
                "right_world_wxyz": _tensor_list(snapshot_ee_poses[env_index, 1]),
            }
        elif is_terminal:
            ee_poses = None
        else:
            ee_poses = {
                "left_world_wxyz": _tensor_list(
                    torch.cat((left_ee_pos[env_index], left_ee_quat[env_index]))
                ),
                "right_world_wxyz": _tensor_list(
                    torch.cat((right_ee_pos[env_index], right_ee_quat[env_index]))
                ),
            }

        if has_snapshot_forbidden:
            forbidden_breakdown = _forbidden_contact_breakdown(
                snapshot_forbidden_body_names,
                snapshot_forbidden_robot_forces[env_index],
                snapshot_forbidden_palm_forces[env_index],
            )
        elif is_terminal:
            forbidden_breakdown = None
        else:
            forbidden_breakdown = _live_forbidden_contact_breakdown(env, env_index)

        if snapshot_ee_poses is not None and has_snapshot_forbidden:
            live_state_note = "pre_autoreset_snapshot_state"
        elif snapshot_ee_poses is not None and is_terminal:
            live_state_note = "pre_autoreset_ee_snapshot; forbidden detail absent from legacy snapshot"
        elif is_terminal:
            live_state_note = "omitted_after_autoreset; fields absent from legacy terminal_snapshot"
        else:
            live_state_note = (
                "initial_live_state" if snapshot is None else "live_state_matches_nonterminal_post_step"
            )
        reason_code = int(reason_codes[env_index].item())
        environments.append(
            {
                "env_index": env_index,
                "timestamp_s": float(timestamp_s[env_index].item()),
                "phase_code": int(phase[env_index].item()),
                "phase_name": phase_names[int(phase[env_index].item())],
                "terminal": is_terminal,
                "termination_reason_code": reason_code,
                "termination_reason": reason_names[reason_code],
                "retry_count": int(retry_count[env_index].item()),
                "continuous_grasp_loss_since_s": float(
                    continuous_grasp_loss_since_s[env_index].item()
                ),
                "insertion_command_depth_m": float(
                    insertion_command_depth_m[env_index].item()
                ),
                "insertion_advance_enabled": bool(
                    insertion_advance_enabled[env_index].item()
                ),
                "revolute_joint_positions_rad": _tensor_list(joint_positions[env_index]),
                "gripper_positions_m": _tensor_list(gripper_positions[env_index]),
                "ee_poses": ee_poses,
                "object_poses_world_wxyz": {
                    "socket": _tensor_list(object_poses[env_index, 0]),
                    "plug": _tensor_list(object_poses[env_index, 1]),
                },
                "grip_proven": _tensor_list(grip_proven[env_index]),
                "grip_live": _tensor_list(grip_live[env_index]),
                "independent_eval_dwell_s": float(evaluation_dwell_s[env_index].item()),
                "independent_eval_success": bool(evaluation_success[env_index].item()),
                "independent_eval_valid": (
                    None if snapshot is None else bool(snapshot["independent_eval_valid"][env_index].item())
                ),
                "insert_joint_target_error_active": int(phase[env_index].item()) in (
                    int(AssemblyPhase.INSERT), int(AssemblyPhase.DONE)
                ),
                "insert_joint_target_error_rad": (
                    _tensor_list(insert_joint_target_error_rad[env_index])
                    if int(phase[env_index].item()) in (int(AssemblyPhase.INSERT), int(AssemblyPhase.DONE))
                    else [0.0] * len(env._arm_joint_ids)
                ),
                "contact_force_proxy_n": _tensor_list(contact_force[env_index]),
                "grip_contact_force_peak_n": (
                    None
                    if grip_contact_force_peak_n is None
                    else float(grip_contact_force_peak_n[env_index].item())
                ),
                "grip_contact_force_definition": (
                    "per_hand_peak_filtered_finger_object_normal_contact_magnitude_n"
                ),
                "insertion_contact_force_n": (
                    None
                    if insertion_contact_force_n is None
                    else float(insertion_contact_force_n[env_index].item())
                ),
                "insertion_contact_force_w": (
                    None
                    if insertion_contact_force_w is None
                    else _tensor_list(insertion_contact_force_w[env_index])
                ),
                "insertion_contact_force_peak_n": (
                    None
                    if insertion_contact_force_peak_n is None
                    else float(insertion_contact_force_peak_n[env_index].item())
                ),
                "insertion_contact_force_source": insertion_contact_force_source,
                "insertion_contact_load_n": (
                    None
                    if insertion_contact_load_n is None
                    else float(insertion_contact_load_n[env_index].item())
                ),
                "insertion_contact_load_peak_n": (
                    None
                    if insertion_contact_load_peak_n is None
                    else float(insertion_contact_load_peak_n[env_index].item())
                ),
                "insertion_contact_load_source": insertion_contact_load_source,
                "forbidden_contact_load_n": (
                    None
                    if forbidden_contact_load_n is None
                    else float(forbidden_contact_load_n[env_index].item())
                ),
                "forbidden_contact_load_peak_n": (
                    None
                    if forbidden_contact_load_peak_n is None
                    else float(forbidden_contact_load_peak_n[env_index].item())
                ),
                "forbidden_contact_load_source": forbidden_contact_load_source,
                "finger_contact_force_n": (
                    None
                    if finger_contact_force_n is None
                    else _tensor_list(finger_contact_force_n[env_index])
                ),
                "finger_contact_order": (
                    "left_l1,left_l2,right_r1,right_r2"
                    if finger_contact_force_n is not None
                    else None
                ),
                "finger_non_target_force_w": (
                    None
                    if finger_non_target_force_w is None
                    else _tensor_list(finger_non_target_force_w[env_index])
                ),
                "finger_non_target_force_definition": (
                    "net_normal_force_world_frame_minus_filtered_target_force_world_frame"
                    if finger_non_target_force_w is not None
                    else None
                ),
                "actual_depth_m": None if depth is None else float(depth[env_index].item()),
                "actual_lateral_error_m": (
                    None if lateral is None else float(lateral[env_index].item())
                ),
                "actual_orientation_error_rad": (
                    None if orientation is None else float(orientation[env_index].item())
                ),
                "forbidden_contact_breakdown": forbidden_breakdown,
                "live_state_note": live_state_note,
            }
        )
        if snapshot is not None:
            criteria = env.extras["terminal_evaluation"]["criteria"]
            environments[-1]["independent_eval_criteria"] = {
                name: bool(value[env_index].item()) for name, value in criteria.items()
            }
        if hasattr(env, "hand_collision_state"):
            hand_state = env.hand_collision_state()
            for key in ("cross_hand_contact_load_n", "external_forbidden_contact_load_n", "hand_wrist_separation_m"):
                environments[-1][key] = (
                    float(snapshot[key][env_index].item()) if snapshot is not None else hand_state[key]
                )
            for key in ("cross_hand_contact_load_peak_n", "external_forbidden_contact_load_peak_n"):
                environments[-1][key] = (
                    float(snapshot[key][env_index].item()) if snapshot is not None else 0.0
                )
            if snapshot is not None:
                environments[-1]["hand_collision_joint_velocity_finite"] = bool(
                    snapshot["hand_collision_joint_velocity_finite"][env_index].item()
                )
                environments[-1]["max_abs_revolute_joint_velocity_rad_s"] = float(
                    snapshot["max_abs_revolute_joint_velocity_rad_s"][env_index].item()
                )
            environments[-1]["self_collision_enabled"] = True
        if hasattr(env.cfg, "rj45_task"):
            from rby1_assembly.rj45_task import verify_recorded_mating_geometry

            entry = environments[-1]
            if snapshot is not None:
                for key in ("raw_joint_reference_rad", "applied_joint_position_target_rad",
                            "model_gravity_torque_nm", "finger_poses_world_wxyz",
                            "object_com_velocities_world", "object_link_velocities_world", "object_physx_velocities_world",
                            "object_pose_difference_linear_velocity_m_s", "object_pose_difference_angular_velocity_rad_s",
                            "object_pose_difference_velocity_available",
                            "object_substep_poses_world_wxyz", "object_substep_velocities_world",
                            "object_substep_link_velocities_world"):
                    entry[key] = _tensor_list(snapshot[key][env_index])
                entry["object_data_cache_times_s"] = _tensor_list(snapshot["object_data_cache_times_s"])
                stability = snapshot.get("rj45_evaluation_stability", {})
                entry["mating_stability"] = {
                    key: _tensor_json_value(value[env_index]) for key, value in stability.items()
                }
                for key in ("pregrasp_clearance_reached", "pregrasp_above_reached"):
                    if key in snapshot:
                        entry[key] = bool(snapshot[key][env_index].item())
                for key in (
                    "cross_hand_contact_load_n", "cross_hand_contact_load_peak_n",
                    "external_forbidden_contact_load_n", "external_forbidden_contact_load_peak_n",
                ):
                    if key in snapshot:
                        entry[key] = float(snapshot[key][env_index].item())
                entry["self_collision_enabled"] = bool(snapshot.get("self_collision_enabled", False))
            frames = env.cfg.rj45_task["frames"]
            entry["independent_mating_geometry"] = verify_recorded_mating_geometry(
                entry["object_poses_world_wxyz"]["socket"],
                entry["object_poses_world_wxyz"]["plug"],
                frames["socket_entry"],
                frames["plug_tip"],
                entry["actual_depth_m"],
                entry["actual_lateral_error_m"],
                entry["actual_orientation_error_rad"],
            )
    return {
        "schema_version": 1,
        "control_step": control_step,
        "events": events,
        "state_source": "initial_live_state" if snapshot is None else "terminal_snapshot_plus_live_nonterminal",
        "revolute_joint_names": revolute_joint_names,
        "gripper_joint_names": gripper_joint_names,
        "pose_format": "position_xyz_m_then_quaternion_wxyz",
        "environments": environments,
    }


def _capture_actual_frame(env: Rby1AssemblyEnv):
    frame = env.render(recompute=True)
    if frame is not None and frame.size > 0 and bool(frame.any()):
        return frame
    raise RuntimeError("renderer did not produce a non-empty RGB frame")


def _write_runtime_failure_report(error: Exception) -> None:
    report = {
        "completed": False,
        "runtime_validated": False,
        "num_envs": args.num_envs,
        "steps_requested": _effective_steps_for_report,
        "steps_cli_override": args.steps,
        "trace_path": None if args.trace is None else str(args.trace),
        "video_path": None if args.video is None else str(args.video),
        "video_camera_preset": None if args.video is None else args.camera_view,
        "error": {
            "type": type(error).__name__,
            "message": str(error),
        },
    }
    immutability_error = error if isinstance(error, InputMutationError) else None
    if immutability_error is None and _input_snapshot_for_failure_report is not None:
        try:
            _input_snapshot_for_failure_report.assert_unchanged()
        except InputMutationError as detected:
            immutability_error = detected
    report["input_immutability"] = (
        {"verified_unchanged": True}
        if _input_snapshot_for_failure_report is not None and immutability_error is None
        else None if immutability_error is None
        else immutability_error.report()
    )
    atomic_write_json(args.report, report)


def main() -> int:
    global _effective_steps_for_report, _input_snapshot_for_failure_report
    task_path = PROJECT_ROOT / ("configs/tasks/rj45-v1.json" if args.task == "rj45" else "configs/task.json")
    assets_path = PROJECT_ROOT / "configs/assets.json"
    runtime_path = PROJECT_ROOT / "configs/runtime.json"
    connector_assets_path = PROJECT_ROOT / "configs/rj45-assets.json"
    input_snapshot = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(
            assets_path,
            runtime_path,
            *((connector_assets_path,) if args.task == "rj45" else ()),
            *((args.command_fault_config,) if args.command_fault is not None else ()),
        ),
        source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
        source_files=(PROJECT_ROOT / "scripts/run_sim.py",),
    )
    _input_snapshot_for_failure_report = input_snapshot
    _bootstrap_input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    task_config = input_snapshot.json_for(task_path)
    runtime_config = input_snapshot.json_for(runtime_path)
    build_report_path = input_snapshot.build_report_path
    runtime_versions = {name: metadata.version(name) for name in
                        ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py")}
    if args.task == "rj45":
        from rby1_assembly.rj45_task import verify_runtime_versions
        verify_runtime_versions(runtime_versions, runtime_config)
    run_seed = args.seed if args.seed is not None else int(task_config["data"]["train_seed"])
    cfg = Rby1AssemblyEnvCfg()
    cfg.seed = run_seed
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.render_background = args.background
    control_dt_s = float(cfg.sim.dt * cfg.decimation)
    configured_timeout_s = float(task_config["evaluation"]["timeout_s"])
    effective_steps = args.steps or math.ceil(configured_timeout_s / control_dt_s)
    _effective_steps_for_report = effective_steps
    if args.video is not None:
        render_stride = round(1.0 / (cfg.sim.dt * args.video_fps))
        if not math.isclose(render_stride * cfg.sim.dt * args.video_fps, 1.0):
            raise ValueError("video FPS must divide the physics frequency")
        cfg.sim.render_interval = render_stride
        if args.camera_view == "overview":
            cfg.viewer.eye = (2.2, -2.0, 1.8)
            cfg.viewer.lookat = (0.35, 0.0, 0.9)
            cfg.viewer.resolution = (960, 720)
        elif args.camera_view == "assembly":
            cfg.viewer.eye = (0.95, -0.60, 1.25)
            cfg.viewer.lookat = (0.36, 0.0, 0.87)
            cfg.viewer.resolution = (1280, 960)
        else:
            cfg.viewer.eye = (0.50, -0.16, 1.01)
            cfg.viewer.lookat = (0.37, 0.01, 0.90)
            cfg.viewer.resolution = (1920, 1080)
    fault_config = input_snapshot.json_for(args.command_fault_config) if args.command_fault else None
    env_type = Rby1AssemblyEnv
    if args.task == "rj45":
        from rby1_assembly.rj45_env import configure_rj45_task, load_rj45_task

        env_type = configure_rj45_task(cfg, load_rj45_task(task_path), PROJECT_ROOT)
    if args.command_fault == "hand_collision":
        from rby1_assembly.hand_collision import make_hand_collision_env_type

        assert fault_config is not None
        env_type = make_hand_collision_env_type(Rby1AssemblyEnv, fault_config["hand_collision"])
    input_snapshot.assert_unchanged()
    env = env_type(cfg, render_mode="rgb_array" if args.video is not None else None)
    completed = False
    max_contact_force_proxy_n = 0.0
    max_grip_contact_force_n = 0.0
    max_insertion_contact_force_n = 0.0
    max_insertion_contact_load_n = 0.0
    max_forbidden_contact_load_n = 0.0
    insertion_contact_force_source = None
    insertion_contact_load_source = None
    forbidden_contact_load_source = FORBIDDEN_CONTACT_LOAD_SOURCE
    grip_proof_peak_count = 0
    actual_steps = 0
    terminal_phase_counts = {phase.name.lower(): 0 for phase in AssemblyPhase}
    last_phase_counts = {phase.name.lower(): 0 for phase in AssemblyPhase}
    trace_file = None
    trace_record_count = 0
    video_writer = None
    video_capture = None
    command_fault = None
    previous_rtx_render_setting = None
    video_frame_count = 0
    video_terminal_step_frames_omitted = 0
    snapshot = None
    try:
        env.reset()
        if args.command_fault and args.command_fault != "hand_collision":
            from rby1_assembly.command_faults import CommandFaultExperiment

            assert fault_config is not None
            command_fault = CommandFaultExperiment(
                env, args.command_fault, fault_config[args.command_fault],
                {phase.name.lower(): int(phase) for phase in AssemblyPhase},
            )
            command_fault.install()
        if args.trace is not None:
            args.trace.parent.mkdir(parents=True, exist_ok=True)
            trace_file = args.trace.open("w", encoding="utf-8", buffering=1)
            trace_file.write(
                json.dumps(_trace_record(env, 0, ["initial"]), allow_nan=False) + "\n"
            )
            trace_record_count += 1
        if args.video is not None:
            import imageio.v2 as imageio

            args.video.parent.mkdir(parents=True, exist_ok=True)
            video_writer = imageio.get_writer(
                args.video, fps=args.video_fps, codec="libx264", quality=8,
                pixelformat="yuv420p", output_params=["-movflags", "+faststart"],
            )
            # Initialize the RGB annotator, then converge RTX history without
            # advancing physics or recording its noisy startup frames.
            env.render(recompute=True)
            for _ in range(16):
                env.sim.render()
            _capture_actual_frame(env)
            # The viewer render product is not a Lab Camera sensor, so register
            # its render scheduling requirement for DirectRLEnv's physics loop.
            previous_rtx_render_setting = bool(env.sim.get_setting("/isaaclab/render/rtx_sensors"))
            env.sim.set_setting("/isaaclab/render/rtx_sensors", True)
            video_capture = PhysicsFrameCapture(
                env.sim, lambda: env._sim_step_counter,
                lambda: _capture_actual_frame(env).copy(), video_writer.append_data,
                stride=render_stride,
            )
            video_capture.install()

        previous_phase = env._phase.clone()
        for control_step in range(1, effective_steps + 1):
            actions = torch.zeros((env.num_envs, ACTION_DIM), device=env.device)
            capture_context = (
                video_capture.control_step(cfg.decimation // render_stride)
                if video_capture is not None else nullcontext()
            )
            with capture_context:
                _, _, terminated, truncated, extras = env.step(actions)
            actual_steps += 1
            snapshot = extras["terminal_snapshot"]
            snapshot_phase = snapshot["phase"]
            terminal_mask = snapshot["terminal_mask"]
            phase_changed = snapshot_phase != previous_phase
            if trace_file is not None and (
                control_step % trace_every == 0 or bool(torch.any(phase_changed | terminal_mask).item())
            ):
                events = []
                if control_step % trace_every == 0:
                    events.append("periodic")
                if bool(torch.any(phase_changed).item()):
                    events.append("phase_change")
                if bool(torch.any(terminal_mask).item()):
                    events.append("terminal")
                record = _trace_record(env, control_step, events, snapshot)
                if command_fault is not None:
                    record["command_fault"] = command_fault.current_state()
                elif hasattr(env, "hand_collision_state"):
                    record["command_fault"] = snapshot["hand_collision_command_state"]
                trace_file.write(json.dumps(record, allow_nan=False) + "\n")
                trace_record_count += 1
            previous_phase = snapshot_phase.clone()
            if video_capture is not None:
                video_frame_count = video_capture.frame_count
            max_grip_contact_force_n = max(
                max_grip_contact_force_n,
                float(snapshot["grip_contact_force_peak_n"].max().item()),
            )
            max_insertion_contact_force_n = max(
                max_insertion_contact_force_n,
                float(snapshot["insertion_contact_force_peak_n"].max().item()),
            )
            insertion_load_peak = snapshot["insertion_contact_load_peak_n"]
            max_insertion_contact_load_n = max(
                max_insertion_contact_load_n,
                float(insertion_load_peak.max().item()),
            )
            max_forbidden_contact_load_n = max(
                max_forbidden_contact_load_n,
                float(snapshot["forbidden_contact_load_peak_n"].max().item()),
            )
            max_contact_force_proxy_n = max(
                max_grip_contact_force_n,
                max_insertion_contact_load_n,
            )
            insertion_contact_force_source = snapshot.get(
                "insertion_contact_force_source", insertion_contact_force_source
            )
            insertion_contact_load_source = snapshot.get(
                "insertion_contact_load_source", insertion_contact_load_source
            )
            forbidden_contact_load_source = snapshot["forbidden_contact_load_source"]
            grip_proof_peak_count = max(
                grip_proof_peak_count,
                int(torch.sum(torch.all(snapshot["grip_proven"], dim=1)).item()),
            )
            completed = completed or bool(torch.any(extras["success"]).item())
            last_phase_counts = {
                phase.name.lower(): int(torch.sum(snapshot_phase == int(phase)).item())
                for phase in AssemblyPhase
            }
            for phase in AssemblyPhase:
                terminal_phase_counts[phase.name.lower()] += int(
                    torch.sum(terminal_mask & (snapshot_phase == int(phase))).item()
                )
            if bool(torch.all(terminated | truncated).item()):
                break
        sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        if trace_file is not None:
            trace_file.flush()
        if snapshot is None:
            raise RuntimeError("simulation produced no control-step snapshot")
        final_record = _trace_record(env, actual_steps, ["final_report"], snapshot)
        if args.task == "rj45" and env.num_envs == 1 and completed:
            terminal_evidence = final_record["environments"][0]
            if not (terminal_evidence["terminal"] and terminal_evidence["phase_name"] == "done"
                    and terminal_evidence["independent_eval_valid"] is True
                    and terminal_evidence["independent_eval_success"] is True):
                raise RuntimeError("RJ45 completed flag disagrees with independent terminal evidence")
        experiment_report = (
            command_fault.report() if command_fault is not None else
            env.hand_collision_report() if hasattr(env, "hand_collision_report") else None
        )
        input_snapshot.assert_unchanged()
        report = {
            "completed": completed,
            "task_id": task_config.get("task_id", "toy"),
            "independent_evaluator": {
                "source": extras["terminal_evaluation"]["evaluator_source"],
                "version": extras["terminal_evaluation"]["evaluator_version"],
            },
            "task_model": env.rj45_task_report() if hasattr(env, "rj45_task_report") else None,
            "hand_contact_monitor": (
                env.hand_contact_monitor_report() if hasattr(env, "hand_contact_monitor_report") else None
            ),
            "command_fault_experiment": None if experiment_report is None else {
                **experiment_report, "config_path": str(args.command_fault_config),
                "config_sha256": input_snapshot.hash_for(args.command_fault_config),
            },
            "terminal_outcomes": [{
                key: entry[key] for key in (
                    "env_index", "timestamp_s", "phase_name", "terminal", "termination_reason",
                    "grip_proven", "grip_live", "actual_depth_m", "actual_lateral_error_m",
                    "actual_orientation_error_rad", "independent_eval_dwell_s",
                    "independent_eval_success", "independent_eval_valid", "mating_stability", "independent_mating_geometry",
                    "independent_eval_criteria",
                    "object_poses_world_wxyz", "insertion_contact_load_peak_n",
                    "grip_contact_force_peak_n", "forbidden_contact_load_peak_n",
                    "object_link_velocities_world", "object_substep_poses_world_wxyz",
                    "object_substep_velocities_world", "object_substep_link_velocities_world",
                    "pregrasp_clearance_reached", "pregrasp_above_reached",
                    "cross_hand_contact_load_n", "cross_hand_contact_load_peak_n",
                    "external_forbidden_contact_load_n", "external_forbidden_contact_load_peak_n",
                    "hand_wrist_separation_m", "self_collision_enabled",
                    "hand_collision_joint_velocity_finite", "max_abs_revolute_joint_velocity_rad_s",
                ) if key in entry
            } for entry in final_record["environments"]],
            "physics_backend": "PhysX via Isaac Sim",
            "observation_source": "simulator_state_with_noise",
            "baseline_controller_state_source": "simulator_ground_truth",
            "phase_gate_state_source": "simulator_ground_truth",
            "contact_force_source": "contact_force_proxy",
            "num_envs": env.num_envs,
            "steps_requested": effective_steps,
            "steps_cli_override": args.steps,
            "steps_default_source": (
                "cli" if args.steps is not None else "ceil(task_timeout_s/control_dt_s)"
            ),
            "actual_steps": actual_steps,
            "max_contact_force_proxy_n": max_contact_force_proxy_n,
            "contact_force_proxy_definition": "max_of_grip_and_insertion_contact_load_n",
            "max_grip_contact_force_n": max_grip_contact_force_n,
            "grip_contact_force_definition": (
                "per_hand_peak_filtered_finger_object_normal_contact_magnitude_n"
            ),
            "max_insertion_contact_force_n": max_insertion_contact_force_n,
            "insertion_contact_force_definition": (
                "norm_of_plug_socket_net_normal_contact_force_world_frame_n"
            ),
            "insertion_contact_force_source": insertion_contact_force_source,
            "max_insertion_contact_load_n": max_insertion_contact_load_n,
            "insertion_contact_load_definition": (
                "sum_of_absolute_plug_socket_contact_normal_scalars_n"
            ),
            "insertion_contact_load_source": insertion_contact_load_source,
            "max_forbidden_contact_load_n": max_forbidden_contact_load_n,
            "forbidden_contact_force_threshold_n": float(cfg.forbidden_contact_force_n),
            "forbidden_contact_load_definition": (
                snapshot.get("forbidden_contact_load_definition", "sum_of_absolute_filtered_pair_normal_forces_n")
            ),
            "self_collision_enabled": bool(snapshot.get("self_collision_enabled", False)),
            "forbidden_contact_load_source": forbidden_contact_load_source,
            "visual_background_enabled": args.background,
            "trace_path": None if args.trace is None else str(args.trace),
            "trace_sha256": None if args.trace is None else sha256(args.trace),
            "trace_record_count": trace_record_count,
            "trace_every_control_steps": trace_every,
            "video_path": None if args.video is None else str(args.video),
            "video_frame_count": video_frame_count,
            "video_fps": None if args.video is None else args.video_fps,
            "video_capture_source": None if args.video is None else "native_physics_substeps_without_interpolation",
            "video_render_stride_physics_steps": None if args.video is None else render_stride,
            "video_first_physics_step": None if video_capture is None else video_capture.first_step,
            "video_last_physics_step": None if video_capture is None else video_capture.last_step,
            "video_expected_frame_count": None if args.video is None else actual_steps * cfg.decimation // render_stride,
            "video_initial_frame_policy": None if args.video is None else "first_frame_after_first_capture_stride; no_initial_hold",
            "video_warmup_render_count": None if args.video is None else 16,
            "video_viewer_render_scheduling_enabled": args.video is not None,
            "video_terminal_step_frame_policy": (
                None
                if args.video is None
                else "captured_before_terminal_autoreset; duplicate_reset_render_omitted"
            ),
            "video_terminal_step_frames_omitted": video_terminal_step_frames_omitted,
            "video_camera": (
                None
                if args.video is None
                else {
                    "preset": args.camera_view,
                    "eye_world_m": list(cfg.viewer.eye),
                    "lookat_world_m": list(cfg.viewer.lookat),
                    "resolution_px": list(cfg.viewer.resolution),
                }
            ),
            "last_pre_autoreset_phase_counts": last_phase_counts,
            "terminal_phase_counts": terminal_phase_counts,
            "grip_proof_peak_count": grip_proof_peak_count,
            "runtime_validated": actual_steps > 0,
            "provenance": {
                "parameter_provenance": "simulation_design_assumptions_not_hardware_calibration",
                "runtime_versions": runtime_versions,
                "physics_dt_s": float(cfg.sim.dt),
                "control_decimation": int(cfg.decimation),
                "control_dt_s": control_dt_s,
                "configured_task_timeout_s": configured_timeout_s,
                "seed": run_seed,
                "seed_source": "cli" if args.seed is not None else f"{task_path.relative_to(PROJECT_ROOT)}:data.train_seed",
                "task_config_path": str(task_path.relative_to(PROJECT_ROOT)),
                "input_immutability_verified": True,
                "connector_assets_config_sha256": input_snapshot.hash_for(connector_assets_path) if args.task == "rj45" else None,
                "task_config_sha256": input_snapshot.hash_for(task_path),
                "assets_config_sha256": input_snapshot.hash_for(assets_path),
                "runtime_config_sha256": input_snapshot.hash_for(runtime_path),
                "build_report_sha256": input_snapshot.hash_for(build_report_path),
                "source_sha256": input_snapshot.source_sha256,
                "generated_asset_sha256": input_snapshot.generated_asset_sha256,
                "torch_version": torch.__version__,
            },
        }
        atomic_write_json(args.report, report)
        input_snapshot.assert_unchanged()
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)
        return 2 if args.require_completion and not completed else 0
    finally:
        active_exception = sys.exc_info()[0] is not None
        cleanup_error = None
        for resource_name, close_resource in (
            ("video_capture", None if video_capture is None else video_capture.close),
            ("command_fault", None if command_fault is None else command_fault.close),
            ("video_render_setting", None if previous_rtx_render_setting is None else
             lambda: env.sim.set_setting("/isaaclab/render/rtx_sensors", previous_rtx_render_setting)),
            ("video_writer", None if video_writer is None else video_writer.close),
            ("trace_file", None if trace_file is None else trace_file.close),
            ("environment", env.close),
        ):
            if close_resource is None:
                continue
            try:
                close_resource()
            except Exception as error:
                print(f"Failed to close {resource_name}: {error}", file=sys.stderr, flush=True)
                traceback.print_exc()
                if cleanup_error is None:
                    cleanup_error = error
        if cleanup_error is not None and not active_exception:
            raise cleanup_error


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as error:
        traceback.print_exc()
        sys.stderr.flush()
        try:
            _write_runtime_failure_report(error)
        except Exception:
            print("Failed to write runtime failure report", file=sys.stderr, flush=True)
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
