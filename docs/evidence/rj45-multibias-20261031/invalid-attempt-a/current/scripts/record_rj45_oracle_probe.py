#!/usr/bin/env python3
"""Record a replay of one frozen RJ45 oracle-probe evaluation episode.

This helper films a new physics rollout.  It does not turn the recorded replay
into training evidence and does not replace the outcome stored by the probe.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping


FPS = 60
RENDER_STRIDE = 4
VIEWER_RESOLUTION = (1280, 960)
REPLAY_SCHEMA = "rj45_oracle_probe_replay_v1"


class ReplayError(RuntimeError):
    """The requested replay does not match the frozen probe lineage."""


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ReplayError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReplayError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise ReplayError(f"JSON root must be an object: {path}")
    return value


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_probe_lineage(
    *,
    mode: str,
    probe_report_path: Path,
    protocol_record: Mapping[str, Any],
    checkpoint_path: Path | None,
    file_sha256: Any,
) -> dict[str, Any]:
    """Bind a replay to one completed probe and its final policy artifact."""

    report = _load_json(probe_report_path)
    if report.get("status") != "completed" or report.get("mode") != mode:
        raise ReplayError("probe report is not a completed report for the selected mode")
    report_protocol = report.get("protocol")
    if not isinstance(report_protocol, dict):
        raise ReplayError("probe report has no protocol lineage")
    for field in ("file_sha256", "payload_sha256"):
        if report_protocol.get(field) != protocol_record.get(field):
            raise ReplayError(f"probe report protocol {field} mismatch")

    learning = report.get("learning")
    if not isinstance(learning, dict):
        raise ReplayError("probe report has no learning record")
    learned = mode in {"standard79", "oracle92"}
    if learned:
        if checkpoint_path is None:
            raise ReplayError(f"{mode} replay requires --checkpoint")
        checkpoint = learning.get("checkpoint")
        if not learning.get("executed") or not isinstance(checkpoint, dict):
            raise ReplayError("learning report has no final checkpoint")
        observed_sha = file_sha256(checkpoint_path)
        if checkpoint.get("sha256") != observed_sha:
            raise ReplayError("checkpoint bytes do not match the probe report")
        checkpoint_record: dict[str, Any] | None = {
            "path": str(checkpoint_path),
            "sha256": observed_sha,
        }
        training_outcome = report.get("final_evaluation")
    else:
        if checkpoint_path is not None:
            raise ReplayError(f"{mode} replay must not receive a checkpoint")
        if learning.get("executed") is not False:
            raise ReplayError("control-mode report unexpectedly claims learning")
        checkpoint_record = None
        training_outcome = report.get("evaluation")
    if not isinstance(training_outcome, dict):
        raise ReplayError("probe report has no recorded evaluation outcome")
    classification = training_outcome.get("classification")
    if classification not in {"valid_success", "valid_failure"}:
        raise ReplayError("probe outcome is missing or invalid")
    return {
        "mode": mode,
        "probe_report": {
            "path": str(probe_report_path),
            "sha256": file_sha256(probe_report_path),
        },
        "protocol": {
            "file_sha256": protocol_record["file_sha256"],
            "payload_sha256": protocol_record["payload_sha256"],
        },
        "checkpoint": checkpoint_record,
        "probe_outcome": {
            "classification": classification,
            "terminal": training_outcome.get("terminal"),
        },
    }


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _build_render_env(
    args: argparse.Namespace,
    mode: str,
    protocol: Mapping[str, Any],
    probe: Any,
) -> tuple[Any, dict[str, Any], dict[Path, str]]:
    """Repeat the probe's frozen builder, adding only RGB viewer configuration."""

    from rby1_assembly.experiment_contracts import apply_runtime_timing
    from rby1_assembly.held_grasp import (
        bank_source_contract_sha256,
        held_grasp_bank_dependency_paths,
        load_verified_held_grasp_bank,
    )
    from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg
    from rby1_assembly.rj45_env import configure_rj45_task
    from rby1_assembly.rj45_task import (
        load_rj45_task,
        verify_generated_asset_hashes,
        verify_runtime_versions,
    )
    from rby1_assembly.rj45_training import load_training_config
    from rby1_assembly.rj45_training_env import (
        _calibration_bias_case_seed,
        make_rj45_training_env_type,
    )

    root = args.runtime_root
    task_path = _resolve(root, args.task_config)
    training_path = _resolve(root, args.training_config)
    runtime_path = _resolve(root, args.runtime_config)
    bank_path = _resolve(root, args.held_bank)
    task = load_rj45_task(task_path)
    training = load_training_config(training_path)
    runtime = _load_json(runtime_path)
    if probe.file_sha256(task_path) != probe.EXPECTED_TASK_SHA256:
        raise ReplayError("frozen task SHA-256 mismatch")
    if probe.file_sha256(training_path) != probe.EXPECTED_TRAINING_SHA256:
        raise ReplayError("frozen training SHA-256 mismatch")
    if probe.file_sha256(runtime_path) != probe.EXPECTED_RUNTIME_CONFIG_SHA256:
        raise ReplayError("frozen runtime config SHA-256 mismatch")
    versions = probe._runtime_versions()
    verify_runtime_versions(versions, runtime)
    build_report_path = _resolve(root, task["assets"]["build_report"])
    build_report = _load_json(build_report_path)
    verify_generated_asset_hashes(root, task, build_report)
    bank, bank_proof = load_verified_held_grasp_bank(
        root, task, bank_path, "train", allow_final=False
    )
    if (
        bank_proof.get("manifest_sha256") != probe.EXPECTED_BANK_SHA256
        or bank_proof.get("selected_split") != "train"
        or bank_proof.get("selected_grasp_ids", [None] * 2)[probe.GRASP_INDEX]
        != probe.GRASP_ID
        or bank_source_contract_sha256(bank_proof)
        != probe.EXPECTED_SOURCE_CONTRACT_SHA256
    ):
        raise ReplayError("held-grasp bank identity mismatch")
    fixed = protocol["fixed_condition"]
    if (
        _calibration_bias_case_seed(
            fixed["base_seed"], probe.ENVIRONMENT_INDEX, probe.EPISODE_INDEX
        )
        != fixed["calibration_case_seed"]
    ):
        raise ReplayError("fixed calibration seed formula changed")

    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, runtime)
    if (
        not math.isclose(float(cfg.sim.dt), 1.0 / probe.PHYSICS_HZ)
        or int(cfg.decimation) != probe.CONTROL_DECIMATION
    ):
        raise ReplayError("replay timing differs from the probe")
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = fixed["base_seed"]
    cfg.sim.render_interval = RENDER_STRIDE
    cfg.viewer.resolution = VIEWER_RESOLUTION
    env_type = configure_rj45_task(cfg, task, root)
    env_type = make_rj45_training_env_type(
        env_type,
        training,
        held_bank=bank,
        held_bank_proof=bank_proof,
        calibration_bias_enabled=True,
        bias_level=training["calibration_bias"]["initial_level"],
    )
    if mode == "reference":
        from rby1_assembly.rj45_teacher_runtime import make_rj45_teacher_env_type

        env_type = make_rj45_teacher_env_type(env_type)
    elif mode in probe.LEARNING_MODES:
        adapter_path = args.current_root / "src/rby1_assembly/rj45_oracle_diagnostic.py"
        adapter = _load_module(adapter_path, "current_rj45_oracle_diagnostic_replay")
        env_type = adapter.make_rj45_oracle_diagnostic_env_type(
            env_type, actor_mode=mode
        )
    env_type = probe.make_fixed_condition_env_type(env_type)
    env = env_type(cfg, render_mode="rgb_array")
    if mode in probe.LEARNING_MODES:
        probe._rebind_observation_spaces(
            env, protocol["observation_groups"][mode]["dimensions"]
        )

    frozen_paths = {
        task_path,
        training_path,
        runtime_path,
        bank_path,
        args.protocol.resolve(),
        args.runtime_manifest.resolve(),
        args.probe_report.resolve(),
        Path(__file__).resolve(),
        (args.current_root / "scripts/rj45_evaluation_video.py").resolve(),
        (args.current_root / "scripts/run_rj45_oracle_ppo_probe.py").resolve(),
        build_report_path,
    }
    frozen_paths.update(
        held_grasp_bank_dependency_paths(root, bank_path, "train", allow_final=False)
    )
    frozen_paths.update((root / "src/rby1_assembly").glob("*.py"))
    frozen_paths.update(
        _resolve(root, task["assets"][name])
        for name in ("robot", "socket", "plug", "table")
    )
    frozen_paths.add(build_report_path.parent / "scene.usda")
    if mode in probe.LEARNING_MODES:
        frozen_paths.add(
            (
                args.current_root
                / "src/rby1_assembly/rj45_oracle_diagnostic.py"
            ).resolve()
        )
    if mode == "reference":
        frozen_paths.add(
            (root / "src/rby1_assembly/rj45_teacher_runtime.py").resolve()
        )
    if args.checkpoint is not None:
        frozen_paths.add(args.checkpoint.resolve())
    hashes = {path: probe.file_sha256(path) for path in frozen_paths}
    return env, {
        "task": task,
        "training": training,
        "versions": versions,
        "bank_proof": bank_proof,
        "physics_seed": fixed["base_seed"],
    }, hashes


def _policy_for_replay(
    env: Any,
    mode: str,
    checkpoint: Path | None,
    protocol: Mapping[str, Any],
    probe: Any,
) -> Any:
    if mode not in probe.LEARNING_MODES:
        return None
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(
        wrapped,
        copy.deepcopy(probe.runner_config(mode, protocol)),
        log_dir=None,
        device=env.device,
    )
    assert checkpoint is not None
    runner.load(str(checkpoint))
    return runner.get_inference_policy(device=env.device)


def _record_rollout(
    *,
    env: Any,
    mode: str,
    policy: Any,
    video_path: Path,
    lineage: Mapping[str, Any],
    frozen_hashes: Mapping[Path, str],
    physics_seed: int,
    probe: Any,
) -> dict[str, Any]:
    import torch
    from tensordict import TensorDict

    video_helper = _load_module(
        Path(__file__).resolve().with_name("rj45_evaluation_video.py"),
        "rj45_evaluation_video_replay",
    )
    rng = probe.reset_evaluation_rng(env.device, physics_seed=physics_seed)
    reset_result = env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    recorder = video_helper.EvaluationVideoRecorder(
        env,
        video_path=video_path,
        env_index=0,
        fps=FPS,
        max_seconds=probe.HORIZON_S,
        policy_lineage=dict(lineage),
        source_and_config_sha256={str(path): sha for path, sha in frozen_hashes.items()},
    )
    recorder.initialize()
    done = False
    steps = 0
    try:
        for steps in range(1, math.ceil((1.0 + probe.HORIZON_S) * probe.CONTROL_HZ) + 3):
            if mode in ("zero", "reference"):
                action = torch.zeros((1, probe.ACTION_DIM), device=env.device)
            else:
                td = TensorDict(
                    {key: value for key, value in observations.items()}, batch_size=[1]
                )
                with torch.inference_mode():
                    action = policy(td)
            if tuple(action.shape) != (1, probe.ACTION_DIM) or not bool(
                action.isfinite().all()
            ):
                raise ReplayError("replay policy returned an invalid action")
            with recorder.control_step():
                observations, _, terminated, truncated, extras = env.step(action)
            dones = terminated | truncated
            recorder.finalize_control_step(extras, dones)
            done = bool(dones[0].item())
            if done:
                break
    finally:
        evidence = recorder.close(rollout_completed=done)
    if not done or not isinstance(evidence, dict) or not evidence.get("completed"):
        raise ReplayError("replay did not complete a valid filmed episode")
    metadata = _load_json(Path(str(video_path) + ".metadata.json"))
    terminal = metadata.get("selected_terminal")
    if not isinstance(terminal, dict):
        raise ReplayError("replay metadata has no terminal outcome")
    valid = terminal.get("valid")
    success = terminal.get("success")
    if type(valid) is not bool or type(success) is not bool:
        raise ReplayError("replay terminal validity/success is not boolean")
    return {
        "evaluation_rng": rng,
        "steps": steps,
        "outcome": {
            "classification": (
                "valid_success"
                if valid and success
                else "valid_failure"
                if valid
                else "invalid"
            ),
            "success": success,
            "valid": valid,
            "termination_reason": terminal.get("termination_reason"),
            "actual_depth_m": terminal.get("actual_depth_m"),
            "actual_lateral_error_m": terminal.get("actual_lateral_error_m"),
            "actual_orientation_error_rad": terminal.get(
                "actual_orientation_error_rad"
            ),
        },
        "evidence": evidence,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", required=True, type=Path)
    parser.add_argument(
        "--current-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--probe-report", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("zero", "reference", "standard79", "oracle92"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--held-bank", type=Path, default=Path("assets/generated/rj45-held-grasps-v1.json"))
    parser.add_argument("--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/rj45-training-target-offset-v1.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--runtime-manifest", type=Path, default=Path("configs/rj45-oracle-runtime-104721a.json"))
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument(
        "--acknowledge-replay-only",
        action="store_true",
        help="confirm that the MP4 is a fresh replay and is not training evidence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.acknowledge_replay_only:
        parser.error("--acknowledge-replay-only is required")
    args.current_root = args.current_root.resolve()
    args.runtime_root = args.runtime_root.resolve()
    args.protocol = args.protocol.resolve()
    args.probe_report = args.probe_report.resolve()
    args.video = args.video.resolve()
    args.report = args.report.resolve()
    args.checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if not args.runtime_manifest.is_absolute():
        args.runtime_manifest = (args.current_root / args.runtime_manifest).resolve()
    if args.video.exists() or args.report.exists():
        parser.error("video/report output already exists")
    if args.video.suffix.lower() != ".mp4":
        parser.error("--video must use the .mp4 suffix")

    probe = _load_module(
        args.current_root / "scripts/run_rj45_oracle_ppo_probe.py",
        "rj45_oracle_probe_for_replay",
    )
    protocol_record = probe.validate_protocol(
        args.protocol, args.protocol_sha256, args.current_root
    )
    protocol = protocol_record["payload"]
    if args.mode not in protocol["diagnostic_scope"]["required_modes"]:
        raise ReplayError("mode is not preregistered")
    runtime_manifest = probe.validate_runtime_manifest(
        args.runtime_root, args.runtime_manifest
    )
    lineage = validate_probe_lineage(
        mode=args.mode,
        probe_report_path=args.probe_report,
        protocol_record=protocol_record,
        checkpoint_path=args.checkpoint,
        file_sha256=probe.file_sha256,
    )
    sys.path.insert(0, str(args.runtime_root / "src"))
    simulation_app = None
    env = None
    failure: BaseException | None = None
    failure_traceback = None
    exit_code = 1
    try:
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher(
            {"headless": True, "device": args.device, "enable_cameras": True}
        ).app
        env, context, frozen_hashes = _build_render_env(
            args, args.mode, protocol, probe
        )
        runtime_import_origins = probe.verify_runtime_import_origins(
            args.runtime_root
        )
        policy = _policy_for_replay(
            env, args.mode, args.checkpoint, protocol, probe
        )
        replay = _record_rollout(
            env=env,
            mode=args.mode,
            policy=policy,
            video_path=args.video,
            lineage=lineage,
            frozen_hashes=frozen_hashes,
            physics_seed=context["physics_seed"],
            probe=probe,
        )
        for path, digest in frozen_hashes.items():
            if probe.file_sha256(path) != digest:
                raise ReplayError(f"frozen input changed during replay: {path}")
        report = {
            "schema_name": REPLAY_SCHEMA,
            "status": "completed",
            "scope": "fresh physics replay filming; not training or benchmark evidence",
            "mode": args.mode,
            "fps": FPS,
            "physics_hz": probe.PHYSICS_HZ,
            "control_hz": probe.CONTROL_HZ,
            "render_stride_physics_steps": RENDER_STRIDE,
            "viewer_resolution": list(VIEWER_RESOLUTION),
            "lineage": lineage,
            "replay": replay,
            "outcome_comparison": {
                "probe_classification": lineage["probe_outcome"][
                    "classification"
                ],
                "replay_classification": replay["outcome"]["classification"],
                "classification_matches": (
                    lineage["probe_outcome"]["classification"]
                    == replay["outcome"]["classification"]
                ),
                "interpretation": (
                    "fresh replay comparison only; agreement does not reproduce "
                    "training or establish benchmark statistics"
                ),
            },
            "runtime_versions": context["versions"],
            "runtime_manifest": runtime_manifest,
            "runtime_import_origins": runtime_import_origins,
            "claims": {
                "training_result_reproduced": False,
                "benchmark_completed": False,
                "reinforcement_learning_effect_proven": False,
                "fresh_replay_filmed": True,
            },
        }
        _atomic_write_new(args.report, report)
        print(json.dumps({"status": "completed", "report": str(args.report)}, sort_keys=True))
        exit_code = 0
    except BaseException as error:
        failure = error
        failure_traceback = error.__traceback__
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                    failure_traceback = error.__traceback__
                exit_code = 1
        exit_code = probe._close_simulation_app(simulation_app, exit_code)
    if failure is not None:
        raise failure.with_traceback(failure_traceback)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
