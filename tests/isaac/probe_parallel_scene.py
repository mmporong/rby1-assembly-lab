"""Verify four independent PhysX environments without fast physics replication.

The probe first advances one normal control step, then moves only env_0's plug
into its left-palm rigid body.  This is an explicit test-state injection; no
sensor buffers, forces, detector thresholds, or termination state are changed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RB-Y1 four-environment PhysX scene probe")
parser.add_argument("--fault-steps", type=int, default=30)
parser.add_argument(
    "--report",
    type=Path,
    default=Path("artifacts/cloud/parallel-scene-probe.json"),
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(headless=True)
args = parser.parse_args()
if args.fault_steps <= 0:
    parser.error("--fault-steps must be positive")

launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import omni.kit.app  # noqa: E402
import torch  # noqa: E402
from pxr import Usd, UsdPhysics  # noqa: E402

from rby1_assembly.contracts import ACTION_DIM, OBSERVATION_DIM  # noqa: E402
from rby1_assembly.isaac_env import (  # noqa: E402
    AssemblyPhase,
    FORBIDDEN_CONTACT_LOAD_SOURCE,
    Rby1AssemblyEnv,
    Rby1AssemblyEnvCfg,
    TASK_CONFIG,
    TerminationReason,
)


NUM_ENVS = 4
WHEEL_NAMES = ("wheel_fl_link", "wheel_fr_link", "wheel_rr_link", "wheel_rl_link")
ORIGINAL_ROBOT_USD = PROJECT_ROOT / "assets/cache/rby1/model_v_1_2_m.usd"
GENERATED_ROBOT_USD = PROJECT_ROOT / "assets/generated/robot.usda"
ASSET_CONFIG = PROJECT_ROOT / "configs/assets.json"
BUILD_REPORT = PROJECT_ROOT / "assets/generated/build-report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_provenance() -> dict:
    asset_config = json.loads(ASSET_CONFIG.read_text(encoding="utf-8"))
    build_report = json.loads(BUILD_REPORT.read_text(encoding="utf-8"))
    asset = next(item for item in asset_config["assets"] if item["id"] == "rby1_model_m_v1_2")
    source = asset_config["sources"][asset["source"]]
    return {
        "asset_id": asset["id"],
        "source_repository": source["repository"],
        "source_revision": source["revision"],
        "source_url": asset["url"],
        "original_usd_path": str(ORIGINAL_ROBOT_USD),
        "original_usd_expected_sha256": asset["sha256"],
        "original_usd_actual_sha256": _sha256(ORIGINAL_ROBOT_USD),
        "generated_robot_usd_path": str(GENERATED_ROBOT_USD),
        "generated_robot_expected_sha256": build_report["files"]["robot.usda"],
        "generated_robot_actual_sha256": _sha256(GENERATED_ROBOT_USD),
    }


def _wheel_collision_groups(
    stage: Usd.Stage, runtime: bool, expected_paths: set[str] | None = None
) -> dict:
    groups = []
    prims = (
        [stage.GetPrimAtPath(path) for path in sorted(expected_paths)]
        if expected_paths is not None
        else list(stage.Traverse())
    )
    for prim in prims:
        if not prim.IsValid():
            continue
        if prim.GetTypeName() != "PhysicsCollisionGroup" or prim.GetName() != "WheelCollisionGroup":
            continue
        path = str(prim.GetPath())
        collision_group = UsdPhysics.CollisionGroup(prim)
        collection = Usd.CollectionAPI.Get(prim, "colliders")
        includes = [str(target) for target in collection.GetIncludesRel().GetTargets()]
        filtered_groups = [
            str(target) for target in collision_group.GetFilteredGroupsRel().GetTargets()
        ]
        if runtime:
            path_parts = path.split("/")
            env_root = "/".join(path_parts[:4])
            env_local = all(target.startswith(f"{env_root}/") for target in includes)
        else:
            env_root = None
            env_local = None
        groups.append(
            {
                "path": path,
                "env_root": env_root,
                "active": prim.IsActive(),
                "include_count": len(includes),
                "includes_are_env_local": env_local,
                "self_filtered": filtered_groups == [path],
            }
        )
    groups.sort(key=lambda item: item["path"])
    return {"count": len(groups), "groups": groups}


def _write_report(report: dict) -> None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)


def _failure_report(error: BaseException) -> dict:
    report = {
        "passed": False,
        "probe": "parallel_scene_collision_isolation",
        "physics_backend": "PhysX via Isaac Sim",
        "num_envs": NUM_ENVS,
        "headless": bool(args.headless),
        "fault_injection": {
            "kind": "test_only_env0_plug_pose_into_left_palm",
            "sensor_value_injection": False,
            "force_injection": False,
        },
        "error": {"type": type(error).__name__, "message": str(error)},
    }
    try:
        report["asset_provenance"] = _source_provenance()
    except Exception as provenance_error:
        report["asset_provenance_error"] = {
            "type": type(provenance_error).__name__,
            "message": str(provenance_error),
        }
    return report


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.seed = int(TASK_CONFIG["data"]["evaluation_seed"])
    cfg.scene.num_envs = NUM_ENVS
    cfg.sim.device = args.device
    cfg.render_background = False

    env = None
    active_exception = False
    try:
        env = Rby1AssemblyEnv(cfg)
        observations, _ = env.reset()
        actions = torch.zeros((NUM_ENVS, ACTION_DIM), device=env.device)
        observations, _, normal_terminated, normal_truncated, normal_extras = env.step(actions)
        policy_observation = observations["policy"].clone()
        normal_terminated = normal_terminated.clone()
        normal_truncated = normal_truncated.clone()
        normal_loads = normal_extras["forbidden_contact_load_n"].clone()
        normal_episode_lengths = env.episode_length_buf.clone()

        expected_runtime_paths = {
            f"/World/envs/env_{env_index}/Robot/{wheel}/WheelCollisionGroup"
            for env_index in range(NUM_ENVS)
            for wheel in WHEEL_NAMES
        }
        runtime_groups = _wheel_collision_groups(
            env.scene.stage, runtime=True, expected_paths=expected_runtime_paths
        )
        source_stage = Usd.Stage.Open(str(ORIGINAL_ROBOT_USD))
        if source_stage is None:
            raise RuntimeError(f"failed to open original robot USD: {ORIGINAL_ROBOT_USD}")
        source_groups = _wheel_collision_groups(source_stage, runtime=False)
        runtime_paths = {item["path"] for item in runtime_groups["groups"]}
        expected_source_paths = {
            f"/RBY1_M_v1_2/{wheel}/WheelCollisionGroup" for wheel in WHEEL_NAMES
        }
        source_paths = {item["path"] for item in source_groups["groups"]}

        palm_ids, palm_names = env.robot.find_bodies("ee_left", preserve_order=True)
        if len(palm_ids) != 1 or palm_names != ["ee_left"]:
            raise RuntimeError(f"left palm body was not resolved exactly once: {palm_names}")
        left_palm_body_id = palm_ids[0]
        env0_id = torch.tensor([0], dtype=torch.long, device=env.device)
        cross_environment_pose = torch.cat(
            (
                env.robot.data.body_pos_w[[1], left_palm_body_id],
                env.robot.data.body_quat_w[[1], left_palm_body_id],
            ),
            dim=1,
        ).clone()
        env.plug.write_root_pose_to_sim(cross_environment_pose, env0_id)
        env.plug.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env0_id)
        _, _, cross_terminated, cross_truncated, cross_extras = env.step(actions)
        cross_terminated = cross_terminated.clone()
        cross_truncated = cross_truncated.clone()
        cross_loads = cross_extras["forbidden_contact_load_n"].clone()
        # Unfiltered palm forces also detect cross-env contact if the task's
        # target filters were accidentally excluding the foreign plug.
        cross_palm_net_forces = env.forbidden_left_palm_contact.data.net_forces_w.clone()
        env.reset(seed=cfg.seed)
        env.step(actions)

        injected_pose = torch.cat(
            (
                env.robot.data.body_pos_w[[0], left_palm_body_id],
                env.robot.data.body_quat_w[[0], left_palm_body_id],
            ),
            dim=1,
        ).clone()
        env.plug.write_root_pose_to_sim(injected_pose, env0_id)
        env.plug.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device), env0_id)

        terminal_snapshot = None
        terminal_evaluation = None
        terminal_step = None
        returned_terminated = None
        returned_truncated = None
        for fault_step in range(1, args.fault_steps + 1):
            _, _, terminated, truncated, extras = env.step(actions)
            snapshot = extras["terminal_snapshot"]
            if bool(snapshot["terminal_mask"][0].item()):
                terminal_snapshot = snapshot
                terminal_evaluation = extras["terminal_evaluation"]
                terminal_step = fault_step
                returned_terminated = terminated.clone()
                returned_truncated = truncated.clone()
                break

        if terminal_snapshot is None or terminal_evaluation is None or terminal_step is None:
            raise AssertionError(
                f"env_0 did not reach forbidden-contact termination within {args.fault_steps} steps"
            )
        if returned_terminated is None or returned_truncated is None:
            raise AssertionError("terminal return tensors were not captured")

        terminal_loads = terminal_snapshot["forbidden_contact_load_n"]
        terminal_mask = terminal_snapshot["terminal_mask"]
        reason_codes = terminal_snapshot["termination_reason_code"]
        phases = terminal_snapshot["phase"]
        expected_healthy_episode_length = normal_episode_lengths[1:] + terminal_step
        healthy_episode_lengths = env.episode_length_buf[1:].clone()
        provenance = _source_provenance()

        checks = {
            "headless": bool(args.headless),
            "four_environments": env.num_envs == NUM_ENVS,
            "fast_physics_replication_disabled": not cfg.scene.replicate_physics,
            "articulation_self_collision_disabled": (
                cfg.robot.spawn.articulation_props.enabled_self_collisions is False
            ),
            "cross_environment_filter_prim_exists": env.scene.stage.GetPrimAtPath(
                "/World/collisions"
            ).IsValid(),
            "normal_observation_shape_is_4x76": tuple(policy_observation.shape)
            == (NUM_ENVS, OBSERVATION_DIM),
            "normal_observation_is_finite": bool(torch.isfinite(policy_observation).all().item()),
            "normal_step_is_nonterminal": not bool(
                torch.any(normal_terminated | normal_truncated).item()
            ),
            "normal_forbidden_load_shape_is_4": tuple(normal_loads.shape) == (NUM_ENVS,),
            "normal_forbidden_load_is_finite": bool(torch.isfinite(normal_loads).all().item()),
            "normal_forbidden_load_is_zero": bool(torch.all(normal_loads == 0.0).item()),
            "normal_episode_length_is_one": bool(
                torch.all(normal_episode_lengths == 1).item()
            ),
            "cross_environment_overlap_is_nonterminal": not bool(
                torch.any(cross_terminated | cross_truncated).item()
            ),
            "cross_environment_overlap_filtered_load_is_zero": bool(
                torch.all(cross_loads == 0.0).item()
            ),
            "cross_environment_overlap_unfiltered_palm_contact_is_zero": bool(
                torch.isfinite(cross_palm_net_forces).all().item()
                and torch.all(cross_palm_net_forces == 0.0).item()
            ),
            "original_usd_hash_matches_manifest": (
                provenance["original_usd_actual_sha256"]
                == provenance["original_usd_expected_sha256"]
            ),
            "generated_robot_hash_matches_build_report": (
                provenance["generated_robot_actual_sha256"]
                == provenance["generated_robot_expected_sha256"]
            ),
            "original_has_exact_four_wheel_collision_groups": (
                source_groups["count"] == len(WHEEL_NAMES)
                and source_paths == expected_source_paths
            ),
            "original_groups_are_active": all(
                item["active"] for item in source_groups["groups"]
            ),
            "original_groups_have_49_includes": all(
                item["include_count"] == 49 for item in source_groups["groups"]
            ),
            "original_groups_are_self_filtered": all(
                item["self_filtered"] for item in source_groups["groups"]
            ),
            "runtime_has_exact_16_expected_groups": (
                runtime_groups["count"] == NUM_ENVS * len(WHEEL_NAMES)
                and runtime_paths == expected_runtime_paths
            ),
            "runtime_groups_have_49_includes": all(
                item["include_count"] == 49 for item in runtime_groups["groups"]
            ),
            "runtime_groups_are_self_filtered": all(
                item["self_filtered"] for item in runtime_groups["groups"]
            ),
            "runtime_groups_are_inactive_for_environment_filtering": all(
                not item["active"] for item in runtime_groups["groups"]
            ),
            "environment_reports_exact_disabled_wheel_groups": (
                set(env.runtime_disabled_wheel_collision_groups) == expected_runtime_paths
            ),
            "runtime_group_includes_are_env_local": all(
                item["includes_are_env_local"] for item in runtime_groups["groups"]
            ),
            "terminal_load_shape_is_4": tuple(terminal_loads.shape) == (NUM_ENVS,),
            "terminal_load_is_finite": bool(torch.isfinite(terminal_loads).all().item()),
            "env0_forbidden_load_is_positive": bool((terminal_loads[0] > 0.0).item()),
            "env0_forbidden_load_exceeds_threshold": bool(
                (terminal_loads[0] > cfg.forbidden_contact_force_n).item()
            ),
            "env0_terminates_only": bool(
                returned_terminated[0].item()
                and not returned_truncated[0].item()
                and not torch.any(returned_terminated[1:] | returned_truncated[1:]).item()
                and terminal_mask[0].item()
                and not torch.any(terminal_mask[1:]).item()
            ),
            "env0_reason_is_forbidden_contact": bool(
                reason_codes[0].item() == int(TerminationReason.FORBIDDEN_CONTACT)
                and terminal_evaluation["termination_reason"][0] == "forbidden_contact"
            ),
            "env0_phase_is_abort": bool(phases[0].item() == int(AssemblyPhase.ABORT)),
            "healthy_env_loads_remain_zero": bool(torch.all(terminal_loads[1:] == 0.0).item()),
            "healthy_env_episode_lengths_continue": bool(
                torch.equal(healthy_episode_lengths, expected_healthy_episode_length)
            ),
            "sensor_source_is_exact_filtered_pair_load": (
                terminal_snapshot["forbidden_contact_load_source"]
                == FORBIDDEN_CONTACT_LOAD_SOURCE
            ),
        }
        report = {
            "passed": all(checks.values()),
            "probe": "parallel_scene_collision_isolation",
            "physics_backend": "PhysX via Isaac Sim",
            "num_envs": NUM_ENVS,
            "headless": bool(args.headless),
            "scene": {
                "replicate_physics": bool(cfg.scene.replicate_physics),
                "clone_in_fabric": bool(cfg.scene.clone_in_fabric),
                "environment_spacing_m": float(cfg.scene.env_spacing),
                "cross_environment_filter_prim": "/World/collisions",
                "global_collision_path": "/World/Ground",
            },
            "asset_provenance": provenance,
            "source_sha256": {
                str(path.relative_to(PROJECT_ROOT)): _sha256(path)
                for path in sorted((PROJECT_ROOT / "src").rglob("*.py"))
                + [Path(__file__).resolve(), PROJECT_ROOT / "configs/task.json"]
            },
            "wheel_collision_groups": {
                "original_source": source_groups,
                "runtime": runtime_groups,
            },
            "normal_step": {
                "observation_shape": list(policy_observation.shape),
                "forbidden_contact_load_n": _tensor_list(normal_loads),
                "episode_length": _tensor_list(normal_episode_lengths),
            },
            "fault_injection": {
                "kind": "test_only_env0_plug_pose_into_left_palm",
                "target_env": 0,
                "target_body": "ee_left",
                "plug_pose_world_wxyz": _tensor_list(injected_pose[0]),
                "root_velocity_world": [0.0] * 6,
                "sensor_value_injection": False,
                "force_injection": False,
            },
            "cross_environment_negative_control": {
                "plug_env": 0,
                "palm_env": 1,
                "plug_pose_world_wxyz": _tensor_list(cross_environment_pose[0]),
                "forbidden_contact_load_n": _tensor_list(cross_loads),
                "unfiltered_left_palm_net_forces_w_n": _tensor_list(cross_palm_net_forces),
                "terminated": _tensor_list(cross_terminated),
                "truncated": _tensor_list(cross_truncated),
                "reset_before_same_environment_positive_control": True,
            },
            "fault_result": {
                "terminal_fault_step": terminal_step,
                "forbidden_contact_load_n": _tensor_list(terminal_loads),
                "forbidden_contact_threshold_n": float(cfg.forbidden_contact_force_n),
                "returned_terminated": _tensor_list(returned_terminated),
                "returned_truncated": _tensor_list(returned_truncated),
                "termination_reason_code": _tensor_list(reason_codes),
                "phase_code": _tensor_list(phases),
                "healthy_episode_length_expected": _tensor_list(
                    expected_healthy_episode_length
                ),
                "healthy_episode_length_actual": _tensor_list(healthy_episode_lengths),
            },
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        }
        _write_report(report)
        if not report["passed"]:
            print(json.dumps(report, indent=2), file=sys.stderr, flush=True)
            return 2
        print(json.dumps(report, indent=2), flush=True)
        return 0
    except BaseException:
        active_exception = True
        raise
    finally:
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
