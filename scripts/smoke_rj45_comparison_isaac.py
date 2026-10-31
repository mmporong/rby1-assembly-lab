#!/usr/bin/env python3
"""Run a bounded fail-closed PPO/SAC integration smoke in the frozen Isaac runtime.

CPU-only validation happens before AppLauncher is imported.  The runtime root
contains the archived 104721a physics code and assets; the current root supplies
only the comparison environment adapter and common PPO/SAC implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping


SCHEMA_NAME = "rj45_comparison_isaac_smoke_v1"
EXPECTED_RUNTIME_COMMIT = "104721acf2222c1cfe8ec86cd030f5116a4fef93"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "d672417f55a61a1345dd442d067813fffdfd30274b547c8c103430866dd7b0f2"
)
EXPECTED_TASK_SHA256 = "7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b"
EXPECTED_TRAINING_SHA256 = (
    "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179"
)
EXPECTED_RUNTIME_CONFIG_SHA256 = (
    "ebb7fa07ab966c1d3e02f24f60c4ef0ad49d01662f194d463d6414e2e003b67a"
)
EXPECTED_BANK_SHA256 = "c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5"
EXPECTED_BANK_SOURCE_CONTRACT_SHA256 = (
    "38cf66d4dea11fe4253021fa57e3c0b5347974dda7490ddb9b7890e6819098d0"
)
EXPECTED_PARALLEL_CONTACT_EVIDENCE_SHA256 = (
    "db1b1e6291490db85650764374dbbccf32d0f80273e3489d3b5966416d83a383"
)
OBSERVATION_DIM = 79
ACTION_DIM = 6
ALGORITHMS = ("ppo", "sac")


class ComparisonSmokeBlocked(RuntimeError):
    """The G0 integration contract failed; no smoke success may be claimed."""


class ProgressRecorder:
    """Emit flushed JSON checkpoints while retaining them for the final report."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self.events: list[dict[str, Any]] = []

    def record(self, stage: str, **details: Any) -> None:
        event = {
            "event": "g0_progress",
            "stage": stage,
            "pid": os.getpid(),
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self.events.append(event)
        print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def _arm_traceback_heartbeat(progress: ProgressRecorder) -> bool:
    """Schedule low-frequency all-thread dumps without changing process exit state."""

    try:
        faulthandler.enable()
        faulthandler.dump_traceback_later(60.0, repeat=True, exit=False)
    except (OSError, RuntimeError, ValueError) as error:
        progress.record(
            "traceback_heartbeat_unavailable",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        return False
    progress.record("traceback_heartbeat_armed", interval_seconds=60.0)
    return True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonSmokeBlocked(f"cannot load JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ComparisonSmokeBlocked(f"JSON root must be an object: {path}")
    return payload


def _load_current_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ComparisonSmokeBlocked(f"cannot import current helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resolve(root: Path, relative: Path | str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _parallel_evidence_dependency_paths(
    root: Path, specification: Mapping[str, Any]
) -> tuple[Path, ...]:
    dependencies = []
    for name in ("report_path", "trace_path", "sat_report_path"):
        value = specification.get(name)
        if not isinstance(value, str) or not value:
            raise ComparisonSmokeBlocked(
                f"parallel contact evidence {name} must be a non-empty string"
            )
        dependencies.append(_resolve(root, value).resolve())
    return tuple(dependencies)


def validate_runtime_snapshot(runtime_root: Path, manifest_path: Path) -> dict[str, Any]:
    runtime_root = runtime_root.resolve()
    manifest_path = manifest_path.resolve()
    if file_sha256(manifest_path) != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise ComparisonSmokeBlocked("runtime manifest SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    if manifest.get("git_commit") != EXPECTED_RUNTIME_COMMIT:
        raise ComparisonSmokeBlocked("archived runtime commit mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ComparisonSmokeBlocked("runtime manifest files are missing")
    verified: dict[str, str] = {}
    for relative, expected in sorted(files.items()):
        path = runtime_root / relative
        if not path.is_file():
            raise ComparisonSmokeBlocked(f"archived runtime file is missing: {relative}")
        observed = file_sha256(path)
        if observed != expected:
            raise ComparisonSmokeBlocked(f"archived runtime file drifted: {relative}")
        verified[relative] = observed
    return {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "git_commit": manifest["git_commit"],
        "verified_file_count": len(verified),
        "files": verified,
    }


def validate_prelaunch_inputs(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = args.runtime_root.resolve()
    current_root = args.current_root.resolve()
    if runtime_root == current_root:
        raise ComparisonSmokeBlocked("runtime-root and current-root must be separate")
    runtime_manifest = validate_runtime_snapshot(runtime_root, args.runtime_manifest)
    fixed = {
        "task_config": (
            _resolve(runtime_root, args.task_config),
            EXPECTED_TASK_SHA256,
        ),
        "training_config": (
            _resolve(runtime_root, args.training_config),
            EXPECTED_TRAINING_SHA256,
        ),
        "runtime_config": (
            _resolve(runtime_root, args.runtime_config),
            EXPECTED_RUNTIME_CONFIG_SHA256,
        ),
        "held_bank": (_resolve(runtime_root, args.held_bank), EXPECTED_BANK_SHA256),
        "parallel_contact_evidence": (
            _resolve(runtime_root, args.parallel_contact_evidence),
            EXPECTED_PARALLEL_CONTACT_EVIDENCE_SHA256,
        ),
    }
    frozen_inputs: dict[str, dict[str, str]] = {}
    for label, (path, expected) in fixed.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ComparisonSmokeBlocked(f"frozen input binding failed: {label}")
        frozen_inputs[label] = {"path": str(path.resolve()), "sha256": expected}
    current_paths = {
        "runner": Path(__file__).resolve(),
        "comparison_env": current_root / "src/rby1_assembly/rj45_comparison_env.py",
        "comparison_rl": current_root / "src/rby1_assembly/rj45_comparison_rl.py",
        "requirements": current_root / "configs/rj45-comparison-requirements.txt",
    }
    current_sources: dict[str, dict[str, str]] = {}
    for label, path in current_paths.items():
        if not path.is_file():
            raise ComparisonSmokeBlocked(f"current comparison source is missing: {label}")
        current_sources[label] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    return {
        "runtime_root": str(runtime_root),
        "current_root": str(current_root),
        "runtime_manifest": runtime_manifest,
        "frozen_inputs": frozen_inputs,
        "current_sources": current_sources,
        "snapshot_labels": {
            "physics_assets_and_legacy_environment": "archived_runtime_104721a",
            "comparison_adapter_and_algorithms": "current_root_uncommitted_bytes",
        },
    }


def validate_execution_args(args: argparse.Namespace) -> None:
    if not 4 <= args.num_envs <= 16:
        raise ComparisonSmokeBlocked("num-envs must be between 4 and 16")
    if args.device != "cuda:0":
        raise ComparisonSmokeBlocked("G0 Isaac smoke requires cuda:0")
    if args.num_envs > 1 and args.parallel_contact_evidence is None:
        raise ComparisonSmokeBlocked(
            "parallel G0 Isaac smoke requires --parallel-contact-evidence"
        )
    if args.ppo_rollout_steps < 2:
        raise ComparisonSmokeBlocked("ppo-rollout-steps must be at least 2")
    if args.sac_vector_steps < 1:
        raise ComparisonSmokeBlocked("sac-vector-steps must be positive")
    if args.sac_batch_size < 2:
        raise ComparisonSmokeBlocked("sac-batch-size must be at least 2")
    if not math.isfinite(args.sac_utd_ratio) or args.sac_utd_ratio != 1.0:
        raise ComparisonSmokeBlocked("G0 requires an explicit SAC UTD ratio of 1.0")
    if args.min_sac_updates < 2:
        raise ComparisonSmokeBlocked("min-sac-updates must be at least 2")
    if args.readiness_steps < 1:
        raise ComparisonSmokeBlocked("readiness-steps must be positive")


def _policy_observation(result: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ComparisonSmokeBlocked("reset/readiness result must be (observations, infos)")
    observations, infos = result
    if isinstance(observations, dict):
        observations = observations.get("policy")
    if not isinstance(infos, dict):
        raise ComparisonSmokeBlocked("environment infos must be a dictionary")
    return observations, infos


def wait_for_policy_readiness(
    env: Any,
    *,
    max_steps: int,
    progress: ProgressRecorder | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Advance zero-action warmup without exposing it as learning transitions."""

    import torch

    if progress is not None:
        progress.record("initial_reset_begin")
    observations, infos = _policy_observation(env.reset())
    if progress is not None:
        progress.record("initial_reset_complete")
    history: list[int] = []
    for readiness_step in range(max_steps + 1):
        validity = infos.get("policy_observation_valid")
        if not isinstance(validity, torch.Tensor) or validity.shape != (env.num_envs,):
            raise ComparisonSmokeBlocked("readiness requires per-row policy_observation_valid")
        validity = validity.to(dtype=torch.bool, device=observations.device)
        activated = getattr(env, "_training_policy_activated", None)
        estimator_valid = getattr(env, "_training_estimator_valid", None)
        if not isinstance(activated, torch.Tensor) or not isinstance(estimator_valid, torch.Tensor):
            raise ComparisonSmokeBlocked("comparison readiness tensors are unavailable")
        ready = validity & activated & estimator_valid
        ready_rows = int(ready.sum().item())
        history.append(ready_rows)
        if progress is not None and (
            readiness_step == 0
            or readiness_step % 30 == 0
            or readiness_step == max_steps
            or ready_rows == env.num_envs
        ):
            progress.record(
                "readiness_checkpoint",
                readiness_step=readiness_step,
                max_steps=max_steps,
                ready_rows=ready_rows,
                num_envs=env.num_envs,
            )
        if bool(ready.all()):
            if tuple(observations.shape) != (env.num_envs, OBSERVATION_DIM):
                raise ComparisonSmokeBlocked("ready observations must have shape [num_envs, 79]")
            if not bool(torch.isfinite(observations).all()):
                raise ComparisonSmokeBlocked("ready observations contain non-finite values")
            return observations, infos, {
                "warmup_vector_steps": readiness_step,
                "warmup_transitions_excluded_from_learning": readiness_step * env.num_envs,
                "ready_rows_history": history,
                "all_rows_policy_active": True,
                "all_rows_estimator_valid": True,
            }
        if readiness_step == max_steps:
            break
        step_result = env.step(
            torch.zeros((env.num_envs, ACTION_DIM), device=observations.device)
        )
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise ComparisonSmokeBlocked("Isaac step contract must have five values")
        next_observations, _rewards, terminated, truncated, infos = step_result
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            raise ComparisonSmokeBlocked("environment ended before all rows became policy-ready")
        observations = (
            next_observations["policy"]
            if isinstance(next_observations, dict)
            else next_observations
        )
    raise ComparisonSmokeBlocked("not all environments became policy-ready")


class ReadyVectorEnvironment:
    """Expose one captured ready state as collect_bounded's initial reset."""

    def __init__(self, env: Any, observations: Any, infos: dict[str, Any]):
        self._env = env
        self.num_envs = env.num_envs
        self._observations = observations
        self._infos = infos
        self._reset_consumed = False

    def reset(self):
        if self._reset_consumed:
            raise ComparisonSmokeBlocked("ready collector reset may only be consumed once")
        self._reset_consumed = True
        return {"policy": self._observations}, self._infos

    def step(self, actions):
        return self._env.step(actions)


def validate_collector_result(result: Any, *, expected_transitions: int) -> None:
    if result.transitions != expected_transitions:
        raise ComparisonSmokeBlocked("collector transition counter mismatch")
    if result.valid_policy_transitions != expected_transitions:
        raise ComparisonSmokeBlocked("collector observed a non-policy-ready transition")
    if result.learning_transitions != expected_transitions:
        raise ComparisonSmokeBlocked("collector omitted transitions from learning")
    if result.warmup_transitions != 0 or result.skipped_valid_transitions != 0:
        raise ComparisonSmokeBlocked("collector attempted warmup/sample deletion")
    if result.mixed_validity_vector_steps != 0:
        raise ComparisonSmokeBlocked("collector observed mixed-validity vector steps")


def algorithm_vector_steps(
    algorithm: str, *, ppo_rollout_steps: int, sac_vector_steps: int
) -> int:
    if algorithm == "ppo":
        return ppo_rollout_steps * 2
    if algorithm == "sac":
        return sac_vector_steps
    raise ValueError("algorithm must be ppo or sac")


def _runtime_versions() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in (
            "isaacsim",
            "isaaclab",
            "isaaclab-rl",
            "rsl-rl-lib",
            "torch",
            "numpy",
            "h5py",
        )
    }


def _comparison_dependency_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("skrl", "gymnasium")}


def _build_env(
    args: argparse.Namespace,
    comparison_env_module: Any,
    progress: ProgressRecorder,
) -> tuple[Any, dict[str, Any], dict[Path, str]]:
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    from rby1_assembly.held_grasp import (
        bank_source_contract_sha256,
        held_grasp_bank_dependency_paths,
        load_verified_held_grasp_bank,
    )
    from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg
    from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence
    from rby1_assembly.rj45_env import configure_rj45_task
    from rby1_assembly.rj45_task import (
        load_rj45_task,
        verify_generated_asset_hashes,
        verify_runtime_versions,
    )
    from rby1_assembly.rj45_training import load_training_config
    from rby1_assembly.rj45_training_env import make_rj45_training_env_type

    root = args.runtime_root.resolve()
    task_path = _resolve(root, args.task_config)
    training_path = _resolve(root, args.training_config)
    runtime_path = _resolve(root, args.runtime_config)
    bank_path = _resolve(root, args.held_bank)
    parallel_spec_path = _resolve(root, args.parallel_contact_evidence)
    task = load_rj45_task(task_path)
    training = load_training_config(training_path)
    runtime = _load_json(runtime_path)
    parallel_spec = _load_json(parallel_spec_path)
    if training["base_observation_dim"] != OBSERVATION_DIM:
        raise ComparisonSmokeBlocked("training config is not the fixed 79D contract")
    if training["diagnostic_observation_mode"] != "standard":
        raise ComparisonSmokeBlocked("training config exposes privileged diagnostics")
    if training["action_mode"] != "right_only_6d":
        raise ComparisonSmokeBlocked("training config is not right-arm 6D residual")
    versions = _runtime_versions()
    verify_runtime_versions(versions, runtime)
    build_report_path = _resolve(root, task["assets"]["build_report"])
    build_report = _load_json(build_report_path)
    asset_hashes = verify_generated_asset_hashes(root, task, build_report)
    bank, proof = load_verified_held_grasp_bank(
        root, task, bank_path, "train", allow_final=False
    )
    if proof.get("manifest_sha256") != EXPECTED_BANK_SHA256:
        raise ComparisonSmokeBlocked("held-bank manifest proof mismatch")
    if proof.get("selected_split") != "train":
        raise ComparisonSmokeBlocked("held-bank proof is not the train split")
    if bank_source_contract_sha256(proof) != EXPECTED_BANK_SOURCE_CONTRACT_SHA256:
        raise ComparisonSmokeBlocked("held-bank source contract mismatch")
    parallel_contact_proof = load_parallel_contact_evidence(
        parallel_spec, project_root=root
    )
    if args.num_envs != parallel_contact_proof.get("validated_probe_num_envs"):
        raise ComparisonSmokeBlocked(
            "parallel contact evidence does not cover the requested environment count"
        )

    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, runtime)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.rj45_parallel_contact_evidence = parallel_spec
    env_type = configure_rj45_task(cfg, task, root)
    env_type = make_rj45_training_env_type(
        env_type,
        training,
        held_bank=bank,
        held_bank_proof=proof,
        calibration_bias_enabled=True,
        bias_level=training["calibration_bias"]["initial_level"],
    )
    env_type = comparison_env_module.make_rj45_comparison_env_type(env_type)
    dependencies = held_grasp_bank_dependency_paths(
        root, bank_path, "train", allow_final=False
    )
    frozen_paths = {
        task_path,
        training_path,
        runtime_path,
        bank_path,
        parallel_spec_path,
        build_report_path,
        *_parallel_evidence_dependency_paths(root, parallel_spec),
        *(root / path for path in parallel_contact_proof["current_source_sha256"]),
        *dependencies,
        *(_resolve(root, task["assets"][name]) for name in ("robot", "socket", "plug", "table")),
        build_report_path.parent / "scene.usda",
    }
    frozen_paths.update((root / "src/rby1_assembly").glob("*.py"))
    hashes = {path.resolve(): file_sha256(path) for path in frozen_paths}
    progress.record("environment_constructor_begin", num_envs=args.num_envs)
    env = env_type(cfg)
    progress.record("environment_constructor_complete", num_envs=args.num_envs)
    return env, {
        "runtime_versions": versions,
        "comparison_dependency_versions": _comparison_dependency_versions(),
        "task": task,
        "training": training,
        "asset_sha256": asset_hashes,
        "held_bank_proof": proof,
        "held_bank_dependency_count": len(dependencies),
        "parallel_contact_evidence": parallel_contact_proof,
        "parallel_contact_evidence_dependency_sha256": {
            str(path): file_sha256(path)
            for path in _parallel_evidence_dependency_paths(root, parallel_spec)
        },
        "comparison_contract": comparison_env_module.comparison_observation_metadata(),
    }, hashes


def _run_algorithm(
    args: argparse.Namespace,
    algorithm: str,
    comparison_env_module: Any,
    comparison_rl_module: Any,
    progress: ProgressRecorder,
) -> dict[str, Any]:
    import torch

    progress.record("algorithm_begin", algorithm=algorithm)
    progress.record("environment_build_begin", algorithm=algorithm)
    env, context, frozen_hashes = _build_env(args, comparison_env_module, progress)
    progress.record("environment_build_complete", algorithm=algorithm)
    try:
        observations, infos, readiness = wait_for_policy_readiness(
            env, max_steps=args.readiness_steps, progress=progress
        )
        rollout_steps = args.ppo_rollout_steps
        progress.record("agent_creation_begin", algorithm=algorithm)
        agent = comparison_rl_module.make_agent(
            algorithm,
            num_envs=args.num_envs,
            device=args.device,
            rollout_steps=rollout_steps,
            replay_capacity=max(256, args.num_envs * args.sac_vector_steps * 2),
            batch_size=args.sac_batch_size,
            sac_utd_ratio=args.sac_utd_ratio,
            seed=args.seed,
        )
        progress.record("agent_creation_complete", algorithm=algorithm)
        before = comparison_rl_module.parameter_vector(agent)
        update_records: list[dict[str, Any]] = []
        original_update = agent.update

        def audited_update(*update_args, **update_kwargs):
            value = original_update(*update_args, **update_kwargs)
            update_records.append(
                {
                    "index": len(update_records) + 1,
                    "return_finite": _finite_structure(value),
                }
            )
            return value

        agent.update = audited_update
        vector_steps = algorithm_vector_steps(
            algorithm,
            ppo_rollout_steps=rollout_steps,
            sac_vector_steps=args.sac_vector_steps,
        )
        ready_env = ReadyVectorEnvironment(env, observations, infos)
        progress.record(
            "collection_begin",
            algorithm=algorithm,
            vector_steps=vector_steps,
        )
        collector = comparison_rl_module.collect_bounded(
            agent,
            ready_env,
            algorithm=algorithm,
            vector_steps=vector_steps,
            require_policy_validity=True,
        )
        progress.record(
            "collection_complete",
            algorithm=algorithm,
            vector_steps=vector_steps,
            observed_transitions=collector.transitions,
        )
        validate_collector_result(
            collector, expected_transitions=args.num_envs * vector_steps
        )
        if len(update_records) != collector.algorithm_updates:
            raise ComparisonSmokeBlocked(
                f"{algorithm} observed update calls differ from collector counters"
            )
        if not all(record["return_finite"] for record in update_records):
            raise ComparisonSmokeBlocked(f"{algorithm} update returned non-finite values")
        if algorithm == "ppo" and collector.algorithm_updates != 2:
            raise ComparisonSmokeBlocked(
                f"PPO update count is {collector.algorithm_updates}, expected 2"
            )
        if algorithm == "ppo" and collector.optimizer_steps != 8:
            raise ComparisonSmokeBlocked(
                f"PPO optimizer-step count is {collector.optimizer_steps}, expected 8"
            )
        if algorithm == "sac":
            if collector.algorithm_updates < args.min_sac_updates:
                raise ComparisonSmokeBlocked(
                    f"SAC update count is {collector.algorithm_updates}, below "
                    f"minimum {args.min_sac_updates}"
                )
            if collector.optimizer_steps != collector.algorithm_updates * 3:
                raise ComparisonSmokeBlocked("SAC optimizer-step counter mismatch")
            if collector.sac_utd_ratio != args.sac_utd_ratio:
                raise ComparisonSmokeBlocked("SAC UTD ratio counter mismatch")
            if collector.replay_first_ready_at != args.sac_batch_size:
                raise ComparisonSmokeBlocked("SAC replay readiness counter mismatch")
        after = comparison_rl_module.parameter_vector(agent)
        if before.shape != after.shape or not bool(torch.isfinite(after).all()):
            raise ComparisonSmokeBlocked(f"{algorithm} parameters are invalid")
        parameter_delta = after - before
        if not bool(torch.any(parameter_delta != 0)):
            raise ComparisonSmokeBlocked(f"{algorithm} parameters did not change")
        losses = comparison_rl_module.finite_losses(agent)
        aggregate = env.extras.get("rj45_comparison_aggregate")
        if not isinstance(aggregate, dict):
            raise ComparisonSmokeBlocked("comparison aggregate telemetry is missing")
        aggregate_json = {
            name: value.detach().cpu().tolist()
            for name, value in aggregate.items()
        }
        for path, digest in frozen_hashes.items():
            if file_sha256(path) != digest:
                raise ComparisonSmokeBlocked(f"frozen input changed during smoke: {path}")
        result = {
            "algorithm": algorithm,
            "status": "completed",
            "readiness": readiness,
            "collector": asdict(collector),
            "observed_agent_update_calls": len(update_records),
            "updates": update_records,
            "algorithm_settings": {
                "ppo_rollout_steps_per_environment": args.ppo_rollout_steps,
                "sac_batch_size": args.sac_batch_size,
                "sac_utd_ratio": args.sac_utd_ratio,
                "sac_vector_steps": args.sac_vector_steps,
                "minimum_sac_updates": args.min_sac_updates,
                "scope": (
                    "G0 tensor-path and optimizer execution smoke only; these "
                    "settings do not establish a fair PPO/SAC outcome comparison"
                ),
            },
            "finite_losses": losses,
            "parameter_delta": {
                "element_count": int(parameter_delta.numel()),
                "changed_element_count": int((parameter_delta != 0).sum().item()),
                "l2": float(torch.linalg.vector_norm(parameter_delta).item()),
                "finite": True,
            },
            "comparison_aggregate": aggregate_json,
            "context": context,
            "frozen_input_sha256": {
                str(path): digest for path, digest in frozen_hashes.items()
            },
        }
        progress.record(
            "algorithm_complete",
            algorithm=algorithm,
            algorithm_updates=collector.algorithm_updates,
            optimizer_steps=collector.optimizer_steps,
        )
        return result
    finally:
        progress.record("environment_close_begin", algorithm=algorithm)
        env.close()
        progress.record("environment_close_complete", algorithm=algorithm)
        torch.cuda.empty_cache()


def _finite_structure(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_structure(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_structure(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def prepare_next_environment(progress: ProgressRecorder) -> None:
    """Return Kit to a stopped, fresh-stage state before constructing another env."""

    import isaacsim.core.utils.stage as stage_utils
    import omni.kit.app
    import omni.timeline
    from isaaclab.sim import SimulationContext

    if SimulationContext.instance() is not None:
        raise ComparisonSmokeBlocked(
            "previous SimulationContext remains registered after environment close"
        )

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    progress.record(
        "environment_transition_begin",
        timeline_playing=bool(timeline.is_playing()),
        timeline_stopped=bool(timeline.is_stopped()),
    )
    if not timeline.is_stopped():
        progress.record("timeline_stop_requested")
        timeline.stop()
        app.update()
        progress.record(
            "timeline_stop_event_flushed",
            timeline_playing=bool(timeline.is_playing()),
            timeline_stopped=bool(timeline.is_stopped()),
        )
    if timeline.is_playing() or not timeline.is_stopped():
        raise ComparisonSmokeBlocked(
            "timeline did not reach stopped state before environment reconstruction"
        )

    progress.record("fresh_stage_creation_begin")
    stage = stage_utils.create_new_stage()
    stage_utils.update_stage()
    if stage is None or stage_utils.get_current_stage() is None:
        raise ComparisonSmokeBlocked(
            "fresh USD stage was not attached before environment reconstruction"
        )
    progress.record("fresh_stage_creation_complete")
    progress.record("environment_transition_complete")


def run_smoke(
    args: argparse.Namespace,
    prelaunch: Mapping[str, Any],
    progress: ProgressRecorder | None = None,
) -> dict[str, Any]:
    if progress is None:
        progress = ProgressRecorder()
    runtime_src = args.runtime_root.resolve() / "src"
    sys.path.insert(0, str(runtime_src))
    comparison_env_module = _load_current_module(
        args.current_root.resolve() / "src/rby1_assembly/rj45_comparison_env.py",
        "rby1_current_comparison_env",
    )
    comparison_rl_module = _load_current_module(
        args.current_root.resolve() / "src/rby1_assembly/rj45_comparison_rl.py",
        "rby1_current_comparison_rl",
    )
    results = []
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        results.append(
            _run_algorithm(
                args,
                algorithm,
                comparison_env_module,
                comparison_rl_module,
                progress,
            )
        )
        if algorithm_index + 1 < len(ALGORITHMS):
            prepare_next_environment(progress)
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name == "rby1_assembly" or name.startswith("rby1_assembly.")
        if getattr(module, "__file__", None)
    }
    escaped = {
        name: path
        for name, path in origins.items()
        if not Path(path).is_relative_to(runtime_src)
    }
    if escaped:
        raise ComparisonSmokeBlocked(f"archived runtime import escaped: {escaped}")
    return {
        "schema_name": SCHEMA_NAME,
        "status": "completed",
        "gpu_executed": True,
        "device": args.device,
        "num_envs": args.num_envs,
        "algorithms": results,
        "prelaunch": dict(prelaunch),
        "runtime_import_origins": origins,
        "progress_events": progress.events,
        "claims": {
            "g0_isaac_integration_smoke_passed": True,
            "algorithm_fairness_proven": False,
            "method_effect_comparison_completed": False,
            "reinforcement_learning_effect_proven": False,
            "generalization_proven": False,
            "portfolio_result_ready": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument(
        "--current-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=Path("configs/rj45-oracle-runtime-104721a.json"),
    )
    parser.add_argument(
        "--held-bank",
        type=Path,
        default=Path(
            "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json"
        ),
    )
    parser.add_argument(
        "--parallel-contact-evidence",
        type=Path,
        required=True,
        help="verified four-environment hand-contact evidence specification",
    )
    parser.add_argument(
        "--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/rj45-training-target-offset-v1.json"),
    )
    parser.add_argument(
        "--runtime-config", type=Path, default=Path("configs/runtime.json")
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--ppo-rollout-steps", type=int, default=8)
    parser.add_argument("--sac-vector-steps", type=int, default=16)
    parser.add_argument("--sac-batch-size", type=int, default=16)
    parser.add_argument("--sac-utd-ratio", type=float, default=1.0)
    parser.add_argument("--min-sac-updates", type=int, default=2)
    parser.add_argument("--readiness-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _close_simulation_app(simulation_app: Any, exit_code: int) -> int:
    """Close Kit without allowing SystemExit(0) to mask a smoke failure."""

    if simulation_app is None:
        return exit_code
    try:
        simulation_app.close()
    except SystemExit as error:
        close_code = error.code
        if close_code is None or close_code == 0:
            return exit_code
        return close_code if type(close_code) is int else 1
    except BaseException:
        traceback.print_exc()
        return 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.runtime_root = args.runtime_root.resolve()
    args.current_root = args.current_root.resolve()
    if not args.runtime_manifest.is_absolute():
        args.runtime_manifest = (args.current_root / args.runtime_manifest).resolve()
    args.report = args.report.resolve()
    if args.report.exists():
        parser.error("report already exists")
    simulation_app = None
    exit_code = 0
    failure_stage = "prelaunch_validation"
    progress = ProgressRecorder()
    heartbeat_armed = _arm_traceback_heartbeat(progress)
    try:
        progress.record("execution_argument_validation_begin")
        validate_execution_args(args)
        progress.record("execution_argument_validation_complete")
        progress.record("runtime_validation_begin")
        prelaunch = validate_prelaunch_inputs(args)
        progress.record("runtime_validation_complete")
        if args.validate_only:
            report = {
                "schema_name": SCHEMA_NAME,
                "status": "validated_not_executed",
                "gpu_executed": False,
                "prelaunch": prelaunch,
                "progress_events": progress.events,
                "claims": {"g0_isaac_integration_smoke_passed": False},
            }
        else:
            failure_stage = "application_launch"
            from isaaclab.app import AppLauncher

            progress.record("application_launch_begin")
            simulation_app = AppLauncher(
                {"headless": args.headless, "device": args.device}
            ).app
            progress.record("application_launch_complete")
            failure_stage = "isaac_comparison_smoke"
            progress.record("comparison_smoke_begin")
            report = run_smoke(args, prelaunch, progress)
            progress.record("comparison_smoke_complete")
        failure_stage = "report_write"
        progress.record("report_write_begin", report_status=report["status"])
        report["progress_events"] = progress.events
        _atomic_write_new(args.report, report)
        progress.record("report_write_complete", report_status=report["status"])
        print(json.dumps({"status": report["status"], "report": str(args.report)}))
    except Exception as error:
        traceback.print_exc()
        failure = {
            "schema_name": SCHEMA_NAME,
            "status": "blocked",
            "gpu_executed": simulation_app is not None,
            "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "requested_process_exit_code": 2,
            "external_supervisor_must_validate_report_status_and_claims": True,
            "progress_events": progress.events,
            "claims": {"g0_isaac_integration_smoke_passed": False},
        }
        if not args.report.exists():
            _atomic_write_new(args.report, failure)
        exit_code = 2
    finally:
        progress.record("simulation_app_close_begin", app_created=simulation_app is not None)
        exit_code = _close_simulation_app(simulation_app, exit_code)
        progress.record(
            "simulation_app_close_complete",
            app_created=simulation_app is not None,
            requested_exit_code=exit_code,
        )
        if heartbeat_armed:
            faulthandler.cancel_dump_traceback_later()
            progress.record("traceback_heartbeat_cancelled")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
