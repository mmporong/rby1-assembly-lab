#!/usr/bin/env python3
"""Run a frozen single-condition PPO learnability probe for RJ45 insertion.

Protocol generation and validation are CPU-only.  Isaac Sim, Isaac Lab, and
RSL-RL are imported only after the protocol and frozen inputs have passed their
checks.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from typing import Any, Mapping


SCHEMA_NAME = "rj45_oracle_ppo_probe_v1"
BASE_SEED = 20261120
POLICY_TRAINING_SEED = 20261121
CRITIC_INITIALIZATION_SEED = 20261122
ACTOR_INITIALIZATION_SEED = 20261123
EVALUATION_SEED = 20261124
ENVIRONMENT_INDEX = 0
EPISODE_INDEX = 1
CALIBRATION_CASE_SEED = 66384140
CALIBRATION_CASE_SEED_OFFSET = CALIBRATION_CASE_SEED - BASE_SEED
GRASP_INDEX = 1
GRASP_ID = "rj45-train-02"
PHYSICS_HZ = 240
CONTROL_DECIMATION = 8
CONTROL_HZ = 30
HORIZON_S = 24.0
ROLLOUT_STEPS = 128
DEFAULT_UPDATES = 32
MIN_UPDATES = 32
MAX_UPDATES = 96
ACTION_DIM = 6
BASE_OBSERVATION_DIM = 79
ORACLE_OBSERVATION_DIM = 92
EXPECTED_RUNTIME_COMMIT = "104721acf2222c1cfe8ec86cd030f5116a4fef93"
EXPECTED_BANK_SHA256 = "c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5"
EXPECTED_SOURCE_CONTRACT_SHA256 = "38cf66d4dea11fe4253021fa57e3c0b5347974dda7490ddb9b7890e6819098d0"
EXPECTED_TASK_SHA256 = "7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b"
EXPECTED_TRAINING_SHA256 = "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179"
EXPECTED_RUNTIME_CONFIG_SHA256 = "ebb7fa07ab966c1d3e02f24f60c4ef0ad49d01662f194d463d6414e2e003b67a"
EXPECTED_RUNTIME_MANIFEST_SHA256 = "d672417f55a61a1345dd442d067813fffdfd30274b547c8c103430866dd7b0f2"
EXPECTED_RUNTIME_SOURCE_FILE_COUNT = 35
CONTROL_MODE_WALL_S = 300.0
LEARNING_MODE_WALL_S = 1800.0
MODES = ("zero", "reference", "standard79", "oracle92")
LEARNING_MODES = ("standard79", "oracle92")


class OraclePpoProbeError(RuntimeError):
    """The preregistered diagnostic contract was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: Any) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise OraclePpoProbeError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise OraclePpoProbeError(f"JSON root must be an object: {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise OraclePpoProbeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _canonical_sha(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(temporary, path)


def append_jsonl_record(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError) as error:
        raise OraclePpoProbeError("JSONL update record is not finite JSON") from error
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def serialize_terminal_evaluation(terminal: Any) -> dict[str, Any]:
    """Serialize the frozen runtime's TerminalEvaluation through its public contract."""

    serializer = getattr(terminal, "as_dict", None)
    if not callable(serializer):
        raise OraclePpoProbeError("terminal evaluation does not expose as_dict()")
    payload = serializer()
    required = {
        "success",
        "valid",
        "termination_reason",
        "evaluator_source",
        "evaluator_version",
        "terminal_timestamp_s",
        "metrics",
        "criteria",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise OraclePpoProbeError("terminal evaluation serialization contract changed")
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise OraclePpoProbeError(
            "terminal evaluation serialization is not finite JSON"
        ) from error
    return payload


def _close_simulation_app(simulation_app: Any, exit_code: int) -> int:
    """Close Kit without allowing SystemExit(0) to mask an execution failure."""

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


def _normalize_deadline_utc(value: str | None) -> str | None:
    if value is None:
        return None
    if not value.endswith("Z"):
        raise OraclePpoProbeError("deadline must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise OraclePpoProbeError("deadline is not valid ISO-8601 UTC") from error
    if parsed.tzinfo != timezone.utc:
        raise OraclePpoProbeError("deadline must be UTC")
    return parsed.isoformat().replace("+00:00", "Z")


def _deadline_epoch(value: str | None) -> float | None:
    normalized = _normalize_deadline_utc(value)
    if normalized is None:
        return None
    return datetime.fromisoformat(normalized[:-1] + "+00:00").timestamp()


def validate_budget_admission(value: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "budget_cap_usd",
        "new_run_reservation_usd",
        "external_guard_minutes",
        "private_evidence_sha256",
        "locally_verified",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise OraclePpoProbeError("budget admission keys are invalid")

    def finite_nonnegative_number(name: str) -> float | int:
        candidate = value[name]
        if (
            isinstance(candidate, bool)
            or not isinstance(candidate, (int, float))
            or not math.isfinite(candidate)
            or candidate < 0
        ):
            raise OraclePpoProbeError(
                f"budget admission {name} must be a finite nonnegative number"
            )
        return candidate

    cap = finite_nonnegative_number("budget_cap_usd")
    reservation = finite_nonnegative_number("new_run_reservation_usd")
    guard_minutes = finite_nonnegative_number("external_guard_minutes")
    digest = value["private_evidence_sha256"]
    if cap != 700.0:
        raise OraclePpoProbeError("budget admission cap must equal 700 USD")
    if reservation > 12.0:
        raise OraclePpoProbeError("budget admission new-run reservation exceeds 12 USD")
    if guard_minutes > 75.0:
        raise OraclePpoProbeError("budget admission external guard exceeds 75 minutes")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise OraclePpoProbeError(
            "budget admission private evidence digest must be 64 lowercase hex characters"
        )
    if value["locally_verified"] is not True:
        raise OraclePpoProbeError("budget admission must be locally verified")
    return dict(value)


def protocol_payload(
    current_root: Path,
    *,
    updates: int = DEFAULT_UPDATES,
    deadline_utc: str | None = None,
    base_seed: int = BASE_SEED,
    budget_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not MIN_UPDATES <= updates <= MAX_UPDATES:
        raise OraclePpoProbeError(f"updates must be in [{MIN_UPDATES}, {MAX_UPDATES}]")
    if type(base_seed) is not int or base_seed < 0 or base_seed % 8 != 0:
        raise OraclePpoProbeError("base_seed must be a nonnegative exact int divisible by 8")
    calibration_case_seed = base_seed + CALIBRATION_CASE_SEED_OFFSET
    runner = Path(__file__).resolve()
    adapter = current_root / "src/rby1_assembly/rj45_oracle_diagnostic.py"
    g2_helper = current_root / "scripts/run_rj45_g2_physics_probe.py"
    runtime_manifest = current_root / "configs/rj45-oracle-runtime-104721a.json"
    paths = {
        "runner": runner,
        "adapter": adapter,
        "g2_physics_helper": g2_helper,
        "runtime_manifest": runtime_manifest,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise OraclePpoProbeError(f"protocol source missing: {missing}")
    if file_sha256(runtime_manifest) != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise OraclePpoProbeError("runtime source manifest SHA-256 mismatch")
    runtime_manifest_payload = _load_json(runtime_manifest)
    if (
        runtime_manifest_payload.get("schema_version") != 1
        or runtime_manifest_payload.get("git_commit") != EXPECTED_RUNTIME_COMMIT
        or len(runtime_manifest_payload.get("files", {})) != EXPECTED_RUNTIME_SOURCE_FILE_COUNT
    ):
        raise OraclePpoProbeError("runtime source manifest identity changed")
    deadline_utc = _normalize_deadline_utc(deadline_utc)
    mode_wall_time_limit_s = {
        "zero": CONTROL_MODE_WALL_S,
        "reference": CONTROL_MODE_WALL_S,
        "standard79": LEARNING_MODE_WALL_S,
        "oracle92": LEARNING_MODE_WALL_S,
    }
    if budget_admission is None:
        budget_and_stop = {
            "budget_cap_usd": 700.0,
            "prior_conservative_reservation_usd": 696.31822,
            "additional_reservation_usd": 3.0,
            "combined_conservative_reservation_usd": 699.31822,
            "conservative_rate_usd_per_hour": 8.0,
            "mode_wall_time_limit_s": mode_wall_time_limit_s,
            "requires_external_stop_guard": True,
            "absolute_deadline_utc": deadline_utc,
        }
    else:
        admission = validate_budget_admission(budget_admission)
        budget_and_stop = {
            "new_run_reservation_usd": admission["new_run_reservation_usd"],
            "external_guard_minutes": admission["external_guard_minutes"],
            "budget_admission": admission,
            "historical_reservation_not_invoice": True,
            "mode_wall_time_limit_s": mode_wall_time_limit_s,
            "absolute_deadline_utc": deadline_utc,
        }
    return {
        "schema_name": SCHEMA_NAME,
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "frozen_inputs": {
            "archived_runtime_commit": EXPECTED_RUNTIME_COMMIT,
            "bank_manifest_sha256": EXPECTED_BANK_SHA256,
            "bank_source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
            "task_config_sha256": EXPECTED_TASK_SHA256,
            "training_config_sha256": EXPECTED_TRAINING_SHA256,
            "runtime_config_sha256": EXPECTED_RUNTIME_CONFIG_SHA256,
            "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
            "runtime_source_file_sha256": runtime_manifest_payload["files"],
        },
        "diagnostic_scope": {
            "single_fixed_train_condition": True,
            "not_final_policy": True,
            "not_generalization_evidence": True,
            "dev_and_final_unopened": True,
            "required_modes": list(MODES),
            "promotion_rule": (
                "if zero succeeds, this condition cannot be promoted as evidence of a non-trivial RL contribution"
            ),
        },
        "fixed_condition": {
            "base_seed": base_seed,
            "environment_index": ENVIRONMENT_INDEX,
            "episode_index": EPISODE_INDEX,
            "calibration_case_seed": calibration_case_seed,
            "grasp_index": GRASP_INDEX,
            "grasp_id": GRASP_ID,
            "split": "train",
            "num_envs": 1,
            "reset_contract": "constructor_episode_0_then_every_explicit_reset_restores_episode_1",
        },
        "timing": {
            "physics_hz": PHYSICS_HZ,
            "control_hz": CONTROL_HZ,
            "control_decimation": CONTROL_DECIMATION,
            "active_horizon_s": HORIZON_S,
        },
        "observation_groups": {
            "standard79": {"policy": ["policy"], "critic": ["critic"], "dimensions": {"policy": 79, "critic": 92}},
            "oracle92": {"policy": ["policy"], "critic": ["critic"], "dimensions": {"policy": 92, "critic": 92}},
            "ground_truth_usage": "oracle13 is diagnostic actor input only in oracle92; both learning modes use critic92",
        },
        "ppo": {
            "implementation": "rsl-rl-lib==3.0.1 OnPolicyRunner/PPO unchanged",
            "policy_training_seed": POLICY_TRAINING_SEED,
            "critic_initialization_seed": CRITIC_INITIALIZATION_SEED,
            "actor_initialization_seed": ACTOR_INITIALIZATION_SEED,
            "evaluation_seed": EVALUATION_SEED,
            "training_rng_reset_seed": POLICY_TRAINING_SEED,
            "rollout_steps": ROLLOUT_STEPS,
            "updates": updates,
            "transitions": ROLLOUT_STEPS * updates,
            "learning_rate": 3.0e-4,
            "schedule": "fixed",
            "initial_noise_std": 0.1,
            "entropy_coefficient": 0.005,
            "gamma": 0.995,
            "lambda": 0.95,
            "num_learning_epochs": 5,
            "num_mini_batches": 1,
            "zero_actor_output": True,
            "checkpoint_selection": "preregistered final checkpoint only",
            "diagnostic_intermediate_checkpoint_updates": [8, 16, 24],
            "intermediate_checkpoint_policy": (
                "diagnostic recovery artifacts only; never eligible for promotion or selection"
            ),
            "interpretation": (
                f"{updates} updates/{ROLLOUT_STEPS * updates} transitions is a "
                "learnability diagnostic, not effect proof"
            ),
            "distinguished_from_prior_campaign": (
                f"single_env_{ROLLOUT_STEPS}x{updates}_fixed_condition_not_256_env_x32"
            ),
        },
        "physics_contract": {
            "execution_device": "cuda:0",
            "task_reward_action_termination": "unchanged archived runtime",
            "evaluation": "G2 independent geometry/contact/nine-substep verifier",
            "evaluations": "deterministic pre-training and final-checkpoint episodes; zero/reference each one episode",
        },
        "budget_and_stop": budget_and_stop,
    }


def write_protocol(
    path: Path,
    current_root: Path,
    *,
    updates: int,
    deadline_utc: str | None = None,
    base_seed: int = BASE_SEED,
    budget_admission: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = protocol_payload(
        current_root.resolve(), updates=updates, deadline_utc=deadline_utc,
        base_seed=base_seed,
        budget_admission=budget_admission,
    )
    _atomic_write_new(path.resolve(), payload)
    return {"path": str(path.resolve()), "file_sha256": file_sha256(path), "payload_sha256": _canonical_sha(payload)}


def validate_protocol(path: Path, expected_file_sha256: str, current_root: Path) -> dict[str, Any]:
    if file_sha256(path) != expected_file_sha256:
        raise OraclePpoProbeError("protocol file SHA-256 mismatch")
    observed = _load_json(path)
    updates = observed.get("ppo", {}).get("updates")
    if not isinstance(updates, int):
        raise OraclePpoProbeError("protocol update count is invalid")
    deadline_utc = observed.get("budget_and_stop", {}).get("absolute_deadline_utc")
    base_seed = observed.get("fixed_condition", {}).get("base_seed")
    budget_admission = observed.get("budget_and_stop", {}).get("budget_admission")
    expected = protocol_payload(
        current_root, updates=updates, deadline_utc=deadline_utc,
        base_seed=base_seed, budget_admission=budget_admission,
    )
    if observed != expected:
        raise OraclePpoProbeError("protocol payload changed or source hash drifted")
    return {"path": str(path), "file_sha256": expected_file_sha256, "payload_sha256": _canonical_sha(observed), "payload": observed}


def validate_runtime_manifest(runtime_root: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify every archived Python source byte before importing the runtime."""

    root = runtime_root.resolve()
    manifest_path = manifest_path.resolve()
    if file_sha256(manifest_path) != EXPECTED_RUNTIME_MANIFEST_SHA256:
        raise OraclePpoProbeError("runtime manifest file SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    files = manifest.get("files")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("git_commit") != EXPECTED_RUNTIME_COMMIT
        or not isinstance(files, dict)
        or len(files) != EXPECTED_RUNTIME_SOURCE_FILE_COUNT
    ):
        raise OraclePpoProbeError("runtime manifest identity/count mismatch")
    actual_relative = {
        str(path.relative_to(root))
        for path in (root / "src/rby1_assembly").glob("*.py")
    }
    if actual_relative != set(files):
        raise OraclePpoProbeError("runtime source file set differs from manifest")
    observed: dict[str, str] = {}
    for relative, expected_sha in sorted(files.items()):
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file():
            raise OraclePpoProbeError(f"runtime manifest path is invalid: {relative}")
        digest = file_sha256(path)
        if digest != expected_sha:
            raise OraclePpoProbeError(f"runtime source SHA-256 mismatch: {relative}")
        observed[relative] = digest
    return {
        "path": str(manifest_path),
        "file_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
        "git_commit": EXPECTED_RUNTIME_COMMIT,
        "source_file_count": len(observed),
        "source_file_sha256": observed,
    }


def verify_runtime_import_origins(runtime_root: Path) -> dict[str, str]:
    expected_root = (runtime_root / "src").resolve()
    origins: dict[str, str] = {}
    for name, module in sorted(sys.modules.items()):
        if name != "rby1_assembly" and not name.startswith("rby1_assembly."):
            continue
        source = getattr(module, "__file__", None)
        if source is None:
            continue
        path = Path(source).resolve()
        if expected_root not in path.parents:
            raise OraclePpoProbeError(f"runtime import escaped archived source root: {name}")
        origins[name] = str(path)
    if not origins:
        raise OraclePpoProbeError("no archived rby1_assembly modules were imported")
    return origins


def assert_execution_time(args: argparse.Namespace) -> None:
    if time.monotonic() - args.execution_started_monotonic > args.execution_mode_wall_s:
        raise TimeoutError("single-mode wall-time limit exceeded")
    if time.time() >= args.execution_deadline_epoch:
        raise TimeoutError("absolute UTC execution deadline exceeded")


def make_fixed_condition_env_type(base_cls: type) -> type:
    """Make every post-construction reset reproduce episode 1/grasp 1."""

    class FixedConditionEnv(base_cls):
        diagnostic_true_episode_count = 0

        def _reset_idx(self, env_ids=None):
            import torch

            training_ready = bool(getattr(self, "_training_initialized", False))
            if training_ready:
                all_indices = self.robot._ALL_INDICES
                if env_ids is None:
                    env_ids = all_indices
                elif not isinstance(env_ids, torch.Tensor):
                    env_ids = torch.as_tensor(
                        env_ids, dtype=torch.long, device=self.device
                    )
                else:
                    env_ids = env_ids.to(device=self.device, dtype=torch.long)
                if env_ids.ndim != 1 or not torch.equal(
                    torch.sort(env_ids).values, torch.sort(all_indices).values
                ):
                    raise OraclePpoProbeError(
                        "single-condition probe reset must cover robot._ALL_INDICES"
                    )
                self._training_episode_index[env_ids] = 0
            result = super()._reset_idx(env_ids)
            if training_ready:
                self.diagnostic_true_episode_count += int(len(env_ids))
                episode = self._training_episode_index[env_ids]
                grasp = self._training_grasp_index[env_ids]
                if not bool((episode == EPISODE_INDEX).all()):
                    raise OraclePpoProbeError("fixed reset did not restore episode index 1")
                if not bool((grasp == GRASP_INDEX).all()):
                    raise OraclePpoProbeError("fixed reset did not restore grasp index 1")
            return result

    FixedConditionEnv.__name__ = f"FixedCondition{base_cls.__name__}"
    return FixedConditionEnv


def runner_config(mode: str, protocol: Mapping[str, Any]) -> dict[str, Any]:
    if mode not in LEARNING_MODES:
        raise OraclePpoProbeError("PPO runner exists only for standard79/oracle92")
    ppo = protocol["ppo"]
    return {
        "seed": int(ppo["policy_training_seed"]),
        "device": "cuda:0",
        "class_name": "OnPolicyRunner",
        "num_steps_per_env": int(ppo["rollout_steps"]),
        "max_iterations": int(ppo["updates"]),
        "save_interval": int(ppo["updates"]),
        "experiment_name": f"rj45_oracle_probe_{mode}",
        "run_name": "",
        "logger": "tensorboard",
        "clip_actions": 1.0,
        "empirical_normalization": False,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": float(ppo["initial_noise_std"]),
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [256, 256, 128],
            "critic_hidden_dims": [256, 256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": float(ppo["entropy_coefficient"]),
            "num_learning_epochs": int(ppo["num_learning_epochs"]),
            "num_mini_batches": int(ppo["num_mini_batches"]),
            "learning_rate": float(ppo["learning_rate"]),
            "schedule": "fixed",
            "gamma": float(ppo["gamma"]),
            "lam": float(ppo["lambda"]),
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def finite_update_record(update_index: int, result: Any) -> dict[str, Any]:
    """Fail closed if RSL-RL reports a non-finite update metric."""

    import numpy as np

    if isinstance(result, Mapping):
        raw = dict(result)
    elif isinstance(result, (tuple, list)):
        raw = {f"value_{index}": value for index, value in enumerate(result)}
    else:
        raw = {"value": result}
    metrics: dict[str, float] = {}
    for name, value in raw.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.size != 1 or not np.issubdtype(array.dtype, np.number):
            continue
        number = float(array.reshape(-1)[0])
        if not math.isfinite(number):
            raise OraclePpoProbeError(f"non-finite PPO update metric: {name}")
        metrics[name] = number
    if not metrics:
        raise OraclePpoProbeError("PPO update returned no finite scalar metrics")
    return {"update": update_index, "metrics": metrics}


def assert_finite_rollout_targets(storage: Any) -> dict[str, int]:
    """Check PPO return/advantage tensors before the unmodified update clears them."""

    proof: dict[str, int] = {}
    for name in ("returns", "advantages"):
        value = getattr(storage, name, None)
        if value is None or not hasattr(value, "isfinite"):
            raise OraclePpoProbeError(f"PPO storage does not expose {name}")
        if not bool(value.isfinite().all()):
            raise OraclePpoProbeError(f"PPO rollout {name} contains non-finite values")
        proof[name] = int(value.numel())
    return proof


def assert_finite_policy_parameters(policy: Any) -> dict[str, int]:
    """Fail closed if an optimizer step leaves actor or critic parameters non-finite."""

    import torch

    proof: dict[str, int] = {}
    for group_name in ("actor", "critic"):
        module = getattr(policy, group_name, None)
        if module is None:
            raise OraclePpoProbeError(f"PPO policy does not expose {group_name}")
        scalar_count = 0
        parameter_count = 0
        for parameter_name, parameter in module.named_parameters():
            parameter_count += 1
            scalar_count += int(parameter.numel())
            if not bool(torch.isfinite(parameter).all()):
                raise OraclePpoProbeError(
                    f"non-finite PPO {group_name} parameter after update: {parameter_name}"
                )
        if parameter_count == 0:
            raise OraclePpoProbeError(f"PPO {group_name} has no parameters")
        proof[f"{group_name}_parameters"] = parameter_count
        proof[f"{group_name}_scalars"] = scalar_count
    return proof


def _parameter_sha256(module: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_state_sha256(*, include_cuda: bool = False, cuda_device: Any = None) -> dict[str, Any]:
    """Fingerprint RNG streams without consuming them."""

    import random

    import numpy as np
    import torch

    def digest_bytes(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    numpy_state = np.random.get_state()
    numpy_payload = json.dumps(
        {
            "kind": numpy_state[0],
            "keys": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    result = {
        "python": digest_bytes(repr(random.getstate()).encode("utf-8")),
        "numpy": digest_bytes(numpy_payload),
        "torch_cpu": digest_bytes(torch.random.get_rng_state().cpu().numpy().tobytes()),
        "torch_cuda": None,
    }
    if include_cuda:
        device = torch.device(cuda_device)
        result["torch_cuda"] = digest_bytes(
            torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
        )
    return result


def reinitialize_critic(policy: Any, *, seed: int, device: Any) -> dict[str, Any]:
    """Make the 92D critic initialization independent of actor input width."""

    import torch

    torch_device = torch.device(device)
    cuda_index = (
        torch_device.index
        if torch_device.type == "cuda" and torch_device.index is not None
        else torch.cuda.current_device() if torch_device.type == "cuda" else None
    )
    cuda_devices = [cuda_index] if cuda_index is not None else []
    reset_count = 0
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            with torch.cuda.device(cuda_index):
                torch.cuda.manual_seed(seed)
        for module in policy.critic.modules():
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()
                reset_count += 1
    if reset_count == 0:
        raise OraclePpoProbeError("critic exposes no resettable parameters")
    return {
        "seed": seed,
        "device": str(torch_device),
        "resettable_module_count": reset_count,
        "state_sha256": _parameter_sha256(policy.critic),
        "actor_input_width_independent": True,
    }


def paired_initialize_actor(
    policy: Any,
    mode: str,
    *,
    seed: int,
    device: Any,
    baseline_builder: Any,
) -> dict[str, Any]:
    """Share the complete 79D actor initialization and zero only oracle columns."""

    import torch

    torch_device = torch.device(device)
    cuda_index = (
        torch_device.index
        if torch_device.type == "cuda" and torch_device.index is not None
        else torch.cuda.current_device() if torch_device.type == "cuda" else None
    )
    cuda_devices = [cuda_index] if cuda_index is not None else []
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(seed)
        if torch_device.type == "cuda":
            with torch.cuda.device(cuda_index):
                torch.cuda.manual_seed(seed)
        baseline = baseline_builder().to(torch_device)
    target_linears = [module for module in policy.actor.modules() if isinstance(module, torch.nn.Linear)]
    baseline_linears = [module for module in baseline.modules() if isinstance(module, torch.nn.Linear)]
    if len(target_linears) != 4 or len(baseline_linears) != 4:
        raise OraclePpoProbeError("paired actor requires four linear layers")
    with torch.no_grad():
        for index, (target, source) in enumerate(zip(target_linears, baseline_linears, strict=True)):
            if target.bias.shape != source.bias.shape:
                raise OraclePpoProbeError("paired actor bias shape mismatch")
            if index == 0 and mode == "oracle92":
                if target.weight.shape != (source.weight.shape[0], ORACLE_OBSERVATION_DIM):
                    raise OraclePpoProbeError("oracle actor first-layer shape mismatch")
                target.weight[:, :BASE_OBSERVATION_DIM].copy_(source.weight)
                target.weight[:, BASE_OBSERVATION_DIM:].zero_()
            else:
                if target.weight.shape != source.weight.shape:
                    raise OraclePpoProbeError("paired actor weight shape mismatch")
                target.weight.copy_(source.weight)
            target.bias.copy_(source.bias)
        target_linears[-1].weight.zero_()
        target_linears[-1].bias.zero_()
    common_digest = hashlib.sha256()
    for index, layer in enumerate(target_linears):
        weight = (
            layer.weight[:, :BASE_OBSERVATION_DIM]
            if index == 0 and mode == "oracle92"
            else layer.weight
        )
        common_digest.update(weight.detach().cpu().contiguous().numpy().tobytes())
        common_digest.update(layer.bias.detach().cpu().contiguous().numpy().tobytes())
    extra_zero = (
        True
        if mode == "standard79"
        else not bool(torch.count_nonzero(target_linears[0].weight[:, BASE_OBSERVATION_DIM:]))
    )
    if not extra_zero or bool(torch.count_nonzero(target_linears[-1].weight)) or bool(
        torch.count_nonzero(target_linears[-1].bias)
    ):
        raise OraclePpoProbeError("paired actor zero-column/head contract failed")
    return {
        "mode": mode,
        "seed": seed,
        "state_sha256": _parameter_sha256(policy.actor),
        "common_79d_projection_sha256": common_digest.hexdigest(),
        "oracle_extra_13_columns_zero": extra_zero,
        "output_layer_zero": True,
        "extra_columns_trainable": mode == "oracle92",
    }


def verify_zero_actor_contract(
    policy: Any,
    mode: str,
    *,
    device: Any,
    tensor_dict_type: Any = None,
) -> dict[str, Any]:
    """Verify equal zero initial means and 0.1 standard deviation in both modes."""

    import torch
    if tensor_dict_type is None:
        from tensordict import TensorDict

        tensor_dict_type = TensorDict

    policy_width = BASE_OBSERVATION_DIM if mode == "standard79" else ORACLE_OBSERVATION_DIM
    observations = tensor_dict_type(
        {
            "policy": torch.zeros((3, policy_width), dtype=torch.float32, device=device),
            "critic": torch.zeros((3, ORACLE_OBSERVATION_DIM), dtype=torch.float32, device=device),
        },
        batch_size=[3],
    )
    with torch.inference_mode():
        actions = policy.act_inference(observations)
    if tuple(actions.shape) != (3, ACTION_DIM) or not bool(torch.isfinite(actions).all()):
        raise OraclePpoProbeError("initial actor output shape/finite contract failed")
    if bool(torch.count_nonzero(actions)):
        raise OraclePpoProbeError("zero-initialized actor has a non-zero initial mean")
    noise = _policy_noise_summary(policy)
    if not math.isclose(noise["minimum"], 0.1, rel_tol=0.0, abs_tol=1.0e-6) or not math.isclose(
        noise["maximum"], 0.1, rel_tol=0.0, abs_tol=1.0e-6
    ):
        raise OraclePpoProbeError("initial actor standard deviation differs from 0.1")
    return {
        "mode": mode,
        "sample_rows": 3,
        "action_shape": [3, ACTION_DIM],
        "mean_action_all_zero": True,
        "action_sha256": tensor_sha256(actions),
        "noise": noise,
    }


def seed_policy_rng(
    seed: int, *, seed_cuda: bool = False, physics_seed: int = BASE_SEED
) -> dict[str, Any]:
    """Seed policy initialization/exploration without changing the physics seed."""

    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if seed_cuda:
        torch.cuda.manual_seed_all(seed)
    return {
        "seed": seed,
        "python": True,
        "numpy": True,
        "torch_cpu": True,
        "torch_cuda_all_available_devices": seed_cuda,
        "physics_seed_unchanged": physics_seed,
    }


def reset_evaluation_rng(
    device: Any, *, physics_seed: int = BASE_SEED
) -> dict[str, Any]:
    """Reset every evaluation to one RNG state without changing the physics seed."""

    import torch

    torch_device = torch.device(device)
    include_cuda = torch_device.type == "cuda"
    seeded = seed_policy_rng(
        EVALUATION_SEED, seed_cuda=include_cuda, physics_seed=physics_seed
    )
    return {
        "seeded": seeded,
        "state_sha256_before_environment_reset": rng_state_sha256(
            include_cuda=include_cuda,
            cuda_device=torch_device,
        ),
    }


def _policy_noise_summary(policy: Any) -> dict[str, Any]:
    import numpy as np

    candidate = None
    for name in ("std", "action_std"):
        if hasattr(policy, name):
            candidate = getattr(policy, name)
            break
    if candidate is None and hasattr(policy, "distribution"):
        candidate = getattr(policy.distribution, "stddev", None)
    if candidate is None:
        raise OraclePpoProbeError("cannot audit PPO policy standard deviation")
    if hasattr(candidate, "detach"):
        candidate = candidate.detach().cpu().numpy()
    values = np.asarray(candidate, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or np.any(values <= 0.0):
        raise OraclePpoProbeError("PPO policy standard deviation is invalid")
    return {
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "mean": float(values.mean()),
        "shape": list(values.shape),
    }


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _rebind_observation_spaces(env: Any, dimensions: Mapping[str, int]) -> None:
    from gymnasium.spaces import Box, Dict
    from gymnasium.vector.utils import batch_space

    import numpy as np

    single = Dict({
        name: Box(-np.inf, np.inf, shape=(int(width),), dtype=np.float32)
        for name, width in dimensions.items()
    })
    env.single_observation_space = single
    env.observation_space = batch_space(single, env.num_envs)


def _observation_proof(observations: Mapping[str, Any], mode: str) -> dict[str, Any]:
    policy = observations["policy"]
    critic = observations["critic"]
    expected_policy = BASE_OBSERVATION_DIM if mode == "standard79" else ORACLE_OBSERVATION_DIM
    if tuple(policy.shape) != (1, expected_policy) or tuple(critic.shape) != (1, ORACLE_OBSERVATION_DIM):
        raise OraclePpoProbeError("actor/critic observation shape mismatch")
    if not bool(policy.isfinite().all()) or not bool(critic.isfinite().all()):
        raise OraclePpoProbeError("actor/critic observation contains non-finite values")
    base = critic[:, :BASE_OBSERVATION_DIM]
    oracle = critic[:, BASE_OBSERVATION_DIM:]
    if mode == "standard79" and not bool((policy == base).all()):
        raise OraclePpoProbeError("ground truth leaked into standard79 actor")
    if mode == "oracle92" and not bool((policy == critic).all()):
        raise OraclePpoProbeError("oracle92 actor and critic observations differ")
    return {
        "policy_sha256": tensor_sha256(policy),
        "critic_sha256": tensor_sha256(critic),
        "base79_sha256": tensor_sha256(base),
        "oracle13_sha256": tensor_sha256(oracle),
        "dimensions": {"policy": expected_policy, "critic": ORACLE_OBSERVATION_DIM},
        "actor_ground_truth_access": mode == "oracle92",
    }


def _runtime_versions() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py")
    }


def _build_env(args: argparse.Namespace, mode: str, protocol: Mapping[str, Any]) -> tuple[Any, dict[str, Any], dict[Path, str]]:
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    from rby1_assembly.held_grasp import (
        bank_source_contract_sha256,
        held_grasp_bank_dependency_paths,
        load_verified_held_grasp_bank,
    )
    from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg
    from rby1_assembly.rj45_env import configure_rj45_task
    from rby1_assembly.rj45_task import load_rj45_task, verify_generated_asset_hashes, verify_runtime_versions
    from rby1_assembly.rj45_training import load_training_config
    from rby1_assembly.rj45_training_env import _calibration_bias_case_seed, make_rj45_training_env_type

    root = args.runtime_root
    task_path = _resolve(root, args.task_config)
    training_path = _resolve(root, args.training_config)
    runtime_path = _resolve(root, args.runtime_config)
    bank_path = _resolve(root, args.held_bank)
    task = load_rj45_task(task_path)
    training = load_training_config(training_path)
    runtime = _load_json(runtime_path)
    if file_sha256(task_path) != EXPECTED_TASK_SHA256 or file_sha256(training_path) != EXPECTED_TRAINING_SHA256:
        raise OraclePpoProbeError("frozen task/training SHA-256 mismatch")
    if file_sha256(runtime_path) != EXPECTED_RUNTIME_CONFIG_SHA256:
        raise OraclePpoProbeError("frozen runtime config SHA-256 mismatch")
    versions = _runtime_versions()
    verify_runtime_versions(versions, runtime)
    build_report_path = _resolve(root, task["assets"]["build_report"])
    build_report = _load_json(build_report_path)
    verify_generated_asset_hashes(root, task, build_report)
    bank, proof = load_verified_held_grasp_bank(root, task, bank_path, "train", allow_final=False)
    if proof.get("manifest_sha256") != EXPECTED_BANK_SHA256 or proof.get("selected_split") != "train":
        raise OraclePpoProbeError("train-only held-bank identity mismatch")
    if proof.get("selected_grasp_ids", [None] * 2)[GRASP_INDEX] != GRASP_ID:
        raise OraclePpoProbeError("ordered fixed grasp identity mismatch")
    if bank_source_contract_sha256(proof) != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise OraclePpoProbeError("held-bank source contract mismatch")
    fixed_condition = protocol["fixed_condition"]
    base_seed = fixed_condition["base_seed"]
    calibration_case_seed = fixed_condition["calibration_case_seed"]
    if (
        _calibration_bias_case_seed(base_seed, ENVIRONMENT_INDEX, EPISODE_INDEX)
        != calibration_case_seed
    ):
        raise OraclePpoProbeError("fixed calibration seed formula changed")

    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, runtime)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = base_seed
    env_type = configure_rj45_task(cfg, task, root)
    env_type = make_rj45_training_env_type(
        env_type, training, held_bank=bank, held_bank_proof=proof,
        calibration_bias_enabled=True, bias_level=training["calibration_bias"]["initial_level"],
    )
    if mode == "reference":
        from rby1_assembly.rj45_teacher_runtime import make_rj45_teacher_env_type

        env_type = make_rj45_teacher_env_type(env_type)
    elif mode in LEARNING_MODES:
        adapter_path = args.current_root / "src/rby1_assembly/rj45_oracle_diagnostic.py"
        adapter = _load_module(adapter_path, "current_rj45_oracle_diagnostic")
        env_type = adapter.make_rj45_oracle_diagnostic_env_type(env_type, actor_mode=mode)
    env_type = make_fixed_condition_env_type(env_type)
    frozen_paths = {
        task_path,
        training_path,
        runtime_path,
        bank_path,
        args.protocol.resolve(),
        args.runtime_manifest.resolve(),
        Path(__file__).resolve(),
        (args.current_root / "scripts/run_rj45_g2_physics_probe.py").resolve(),
        build_report_path,
    }
    frozen_paths.update(held_grasp_bank_dependency_paths(root, bank_path, "train", allow_final=False))
    frozen_paths.update((root / "src/rby1_assembly").glob("*.py"))
    frozen_paths.update(
        _resolve(root, task["assets"][name])
        for name in ("robot", "socket", "plug", "table")
    )
    frozen_paths.add(build_report_path.parent / "scene.usda")
    if mode in LEARNING_MODES:
        frozen_paths.add((args.current_root / "src/rby1_assembly/rj45_oracle_diagnostic.py").resolve())
    if mode == "reference":
        frozen_paths.add((root / "src/rby1_assembly/rj45_teacher_runtime.py").resolve())
    hashes = {path: file_sha256(path) for path in frozen_paths}
    env = env_type(cfg)
    if mode in LEARNING_MODES:
        _rebind_observation_spaces(env, protocol["observation_groups"][mode]["dimensions"])
    return env, {
        "task": task,
        "training": training,
        "versions": versions,
        "bank_proof": proof,
        "physics_seed": base_seed,
    }, hashes


def _run_evaluation(
    env: Any,
    mode: str,
    policy: Any,
    g2: Any,
    task: Mapping[str, Any],
    training: Mapping[str, Any],
    trace_path: Path,
    time_guard: Any,
    physics_seed: int = BASE_SEED,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from rby1_assembly.evaluation import EvaluationResult, TaskEvaluator
    from rby1_assembly.recording import reconcile_terminal_evaluation

    evaluation_rng = reset_evaluation_rng(env.device, physics_seed=physics_seed)
    reset_result = env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    observation_proof = None
    evaluator = TaskEvaluator(g2._limits(task, training))
    independent: Any = EvaluationResult(False, False, 0.0, ())
    terminal = terminal_snapshot = None
    active_steps = 0
    action_boundary_steps = 0
    reference_active_steps = 0
    reference_nonzero_bounded_steps = 0
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("x", encoding="utf-8") as trace:
        for step in range(1, math.ceil((1.0 + HORIZON_S) * CONTROL_HZ) + 3):
            time_guard()
            if mode in ("zero", "reference"):
                action = torch.zeros((1, ACTION_DIM), device=env.device)
            else:
                from tensordict import TensorDict

                td = TensorDict({key: value for key, value in observations.items()}, batch_size=[1])
                with torch.inference_mode():
                    action = policy(td)
            if tuple(action.shape) != (1, ACTION_DIM) or not bool(action.isfinite().all()):
                raise OraclePpoProbeError("evaluation policy returned invalid action")
            submitted_observations = observations
            observations, _, terminated, truncated, extras = env.step(action)
            done = bool((terminated | truncated)[0].item())
            saved = extras.get("terminal_snapshot")
            if not isinstance(saved, Mapping):
                raise OraclePpoProbeError("terminal snapshot missing")
            snapshot = g2._probe_snapshot(env, saved, task["frames"])
            submitted_base_tensor = (
                submitted_observations["critic"][:, :BASE_OBSERVATION_DIM]
                if mode in LEARNING_MODES
                else submitted_observations["policy"]
            )
            transition = g2._transition_contract(saved, submitted_base_tensor, done)
            valid = bool(transition["valid_input"])
            active = bool(transition["active"])
            if active:
                if observation_proof is None and mode in LEARNING_MODES:
                    observation_proof = _observation_proof(submitted_observations, mode)
                active_steps += 1
                independent = evaluator.update(g2._evaluation_sample(snapshot, saved, active_steps / CONTROL_HZ, task["evaluation"]))
            submitted_base = submitted_base_tensor.detach().cpu().numpy()[0]
            submitted_actor = submitted_observations["policy"].detach().cpu().numpy()[0]
            submitted_oracle = (
                submitted_observations["critic"][:, BASE_OBSERVATION_DIM:]
                .detach()
                .cpu()
                .numpy()[0]
                if mode in LEARNING_MODES
                else None
            )
            requested_expanded = np.asarray(
                transition["requested_expanded"], dtype=np.float64
            )
            applied_expanded = np.asarray(
                transition["applied_expanded"], dtype=np.float64
            )
            if requested_expanded.shape != (12,) or applied_expanded.shape != (12,):
                raise OraclePpoProbeError("expanded action telemetry shape changed")
            if active and bool(np.any(np.abs(requested_expanded[6:]) >= 1.0)):
                action_boundary_steps += 1
            stability = saved["rj45_evaluation_stability"]
            physics_samples = {
                name: g2._scalar(stability[name])
                for name in (
                    "physics_sample_depth_m",
                    "physics_sample_lateral_error_m",
                    "physics_sample_orientation_error_rad",
                    "physics_sample_relative_linear_speed_m_s",
                    "physics_sample_relative_angular_speed_rad_s",
                    "physics_sample_finite",
                    "physics_sample_stable",
                )
            }
            teacher_step = None
            if mode == "reference":
                teacher_active = bool(g2._scalar(saved["teacher_active"]))
                teacher_bounded = np.asarray(
                    g2._scalar(saved["teacher_bounded_action_6d"]), dtype=np.float64
                )
                if teacher_bounded.shape != (6,) or not np.isfinite(teacher_bounded).all():
                    raise OraclePpoProbeError("reference bounded action telemetry is invalid")
                reference_active_steps += int(teacher_active)
                reference_nonzero_bounded_steps += int(
                    teacher_active and bool(np.any(teacher_bounded != 0.0))
                )
                if teacher_active and (
                    bool(np.any(applied_expanded[:6] != 0.0))
                    or not np.array_equal(applied_expanded[6:], teacher_bounded)
                ):
                    raise OraclePpoProbeError(
                        "reference bounded action differs from applied action"
                    )
                teacher_step = {
                    "active": teacher_active,
                    "requested_raw_action_6d": g2._scalar(
                        saved["teacher_requested_raw_action_6d"]
                    ),
                    "bounded_action_6d": teacher_bounded.tolist(),
                    "physical_offset_entry_6d": g2._scalar(
                        saved["teacher_physical_offset_entry_6d"]
                    ),
                    "bound_limited": bool(g2._scalar(saved["teacher_bound_limited"])),
                }
            trace.write(json.dumps({
                "step": step,
                "action": action.detach().cpu().numpy()[0].tolist(),
                "phase": snapshot.phase,
                "active": active,
                "policy_input_valid": valid,
                "environment_timestamp_s": float(transition["environment_timestamp_s"]),
                "observation": {
                    "actor": submitted_actor.tolist(),
                    "base79": submitted_base.tolist(),
                    "oracle13": (
                        submitted_oracle.tolist() if submitted_oracle is not None else None
                    ),
                },
                "requested_expanded_action": requested_expanded.tolist(),
                "applied_expanded_action": applied_expanded.tolist(),
                "applied_tip_target_offset_entry_6d": g2._scalar(saved["applied_tip_target_offset_entry_6d"]),
                "actual_depth_m": snapshot.ground_truth.actual_depth_m,
                "lateral_error_m": snapshot.ground_truth.lateral_error_m,
                "orientation_error_rad": snapshot.ground_truth.orientation_error_rad,
                "force_peak_n": snapshot.ground_truth.force_peak_n,
                "loads_n": {
                    "grip_peak": snapshot.grip_contact_force_peak_n,
                    "insertion_force": snapshot.insertion_contact_force_n,
                    "insertion_load": snapshot.insertion_contact_load_n,
                    "forbidden_load": snapshot.forbidden_contact_load_n,
                },
                "nine_physics_samples": physics_samples,
                "reference_controller": teacher_step,
            }, sort_keys=True) + "\n")
            if done:
                terminal_snapshot = snapshot
                terminal = reconcile_terminal_evaluation(
                    extras=extras, snapshot=snapshot,
                    terminal_timestamp_s=float(g2._scalar(saved["timestamp_s"])),
                    independent_result=independent, task_limits=g2._limits(task, training), env_index=0,
                )
                break
        if terminal is None or terminal_snapshot is None:
            raise OraclePpoProbeError("evaluation did not terminate inside the frozen horizon")
        if mode in LEARNING_MODES and observation_proof is None:
            raise OraclePpoProbeError("evaluation produced no valid post-warmup actor input")
        if active_steps <= 0:
            raise OraclePpoProbeError("evaluation produced no active policy rows")
        if mode == "reference" and reference_active_steps <= 0:
            raise OraclePpoProbeError("reference evaluation did not activate its hook")
        if mode == "reference" and reference_nonzero_bounded_steps <= 0:
            raise OraclePpoProbeError(
                "reference evaluation applied no non-zero recovery correction"
            )
        classification = g2.classify_terminal(terminal)
        trace.write(json.dumps({"event": "complete", "classification": classification}, sort_keys=True) + "\n")
    return {
        "classification": classification,
        "evaluation_rng": evaluation_rng,
        "terminal": serialize_terminal_evaluation(terminal),
        "terminal_phase": terminal_snapshot.phase,
        "active_rows": active_steps,
        "action_boundary_rows": action_boundary_steps,
        "action_boundary_fraction_over_active": (
            action_boundary_steps / active_steps if active_steps else 0.0
        ),
        "reference_controller": (
            {
                "active_rows": reference_active_steps,
                "nonzero_bounded_action_rows": reference_nonzero_bounded_steps,
                "recovery_correction_observed": reference_nonzero_bounded_steps > 0,
            }
            if mode == "reference"
            else None
        ),
        "observation": observation_proof,
        "trace": {"path": str(trace_path), "sha256": file_sha256(trace_path)},
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    protocol_record = validate_protocol(args.protocol.resolve(), args.protocol_sha256, args.current_root)
    protocol = protocol_record["payload"]
    if args.mode not in protocol["diagnostic_scope"]["required_modes"]:
        raise OraclePpoProbeError("mode is not preregistered")
    if args.device != protocol["physics_contract"]["execution_device"]:
        raise OraclePpoProbeError("execution device differs from the preregistered protocol")
    runtime_manifest_record = getattr(args, "runtime_manifest_record", None)
    if runtime_manifest_record is None:
        runtime_manifest_record = validate_runtime_manifest(
            args.runtime_root, args.runtime_manifest.resolve()
        )
    sys.path.insert(0, str(args.runtime_root / "src"))
    g2 = _load_module(args.current_root / "scripts/run_rj45_g2_physics_probe.py", "rj45_g2_probe_helper")
    env, context, frozen_hashes = _build_env(args, args.mode, protocol)
    runtime_import_origins = verify_runtime_import_origins(args.runtime_root)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    completed = False
    result: dict[str, Any] = {}
    try:
        if args.mode in ("zero", "reference"):
            evaluation = _run_evaluation(
                env, args.mode, None, g2, context["task"], context["training"],
                output / "evaluation.jsonl", lambda: assert_execution_time(args),
                physics_seed=context["physics_seed"],
            )
            if evaluation["classification"] == "invalid":
                raise OraclePpoProbeError(f"{args.mode} independent evaluation is invalid")
            result = {"evaluation": evaluation, "learning": {"executed": False}}
            if args.mode == "zero":
                result["condition_eligibility"] = {
                    "nontrivial_rl_contribution_test": evaluation["classification"] != "valid_success",
                    "reason": (
                        "zero-action baseline already succeeds"
                        if evaluation["classification"] == "valid_success"
                        else "zero-action baseline does not solve the fixed condition"
                    ),
                }
            else:
                result["condition_eligibility"] = {
                    "reference_recovery_correction_observed": evaluation[
                        "reference_controller"
                    ]["recovery_correction_observed"],
                    "eligible_for_recovery_learning_claim": evaluation[
                        "reference_controller"
                    ]["recovery_correction_observed"],
                }
        else:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner
            from rby1_assembly.rj45_bc import build_bc_actor, object_sha256

            config = runner_config(args.mode, protocol)
            rng = seed_policy_rng(
                int(protocol["ppo"]["policy_training_seed"]),
                seed_cuda=str(args.device).startswith("cuda"),
                physics_seed=context["physics_seed"],
            )
            pre_runner = OnPolicyRunner(
                RslRlVecEnvWrapper(env, clip_actions=1.0),
                copy.deepcopy(config),
                log_dir=str(output / "ppo_logs"),
                device=args.device,
            )
            initialization = paired_initialize_actor(
                pre_runner.alg.policy,
                args.mode,
                seed=int(protocol["ppo"]["actor_initialization_seed"]),
                device=args.device,
                baseline_builder=build_bc_actor,
            )
            critic_initialization = reinitialize_critic(
                pre_runner.alg.policy,
                seed=int(protocol["ppo"]["critic_initialization_seed"]),
                device=args.device,
            )
            zero_actor_contract = verify_zero_actor_contract(
                pre_runner.alg.policy, args.mode, device=args.device
            )
            optimizer_initial_sha256 = object_sha256(
                pre_runner.alg.optimizer.state_dict()
            )
            initialization_rng_reset = seed_policy_rng(
                int(protocol["ppo"]["training_rng_reset_seed"]),
                seed_cuda=str(args.device).startswith("cuda"),
                physics_seed=context["physics_seed"],
            )
            initialization_rng_sha256 = rng_state_sha256(
                include_cuda=str(args.device).startswith("cuda"),
                cuda_device=args.device,
            )
            actor_before = _parameter_sha256(pre_runner.alg.policy.actor)
            noise_before = _policy_noise_summary(pre_runner.alg.policy)
            pre_policy = pre_runner.get_inference_policy(device=args.device)
            pre = _run_evaluation(
                env, args.mode, pre_policy, g2, context["task"], context["training"],
                output / "pre_evaluation.jsonl", lambda: assert_execution_time(args),
                physics_seed=context["physics_seed"],
            )
            if pre["classification"] == "invalid":
                raise OraclePpoProbeError("pre-training independent evaluation is invalid")
            update_records: list[dict[str, Any]] = []
            updates_trace = output / "updates.jsonl"
            intermediate_updates = set(
                protocol["ppo"]["diagnostic_intermediate_checkpoint_updates"]
            )
            original_update = pre_runner.alg.update
            logging_writer_ready = False

            def audited_update(*update_args, **update_kwargs):
                assert_execution_time(args)
                rollout_proof = assert_finite_rollout_targets(pre_runner.alg.storage)
                value = original_update(*update_args, **update_kwargs)
                assert_execution_time(args)
                record = finite_update_record(len(update_records) + 1, value)
                record["rollout_targets"] = rollout_proof
                record["parameter_finiteness"] = assert_finite_policy_parameters(
                    pre_runner.alg.policy
                )
                update_index = int(record["update"])
                if update_index in intermediate_updates:
                    if not logging_writer_ready:
                        raise OraclePpoProbeError(
                            "diagnostic checkpoint requested before logger/writer preparation"
                        )
                    diagnostic_checkpoint = (
                        output / f"diagnostic_checkpoint_{update_index:04d}.pt"
                    )
                    if diagnostic_checkpoint.exists():
                        raise OraclePpoProbeError(
                            "diagnostic checkpoint path already exists"
                        )
                    pre_runner.save(str(diagnostic_checkpoint))
                    record["diagnostic_checkpoint"] = {
                        "path": str(diagnostic_checkpoint),
                        "sha256": file_sha256(diagnostic_checkpoint),
                        "eligible_for_selection": False,
                    }
                append_jsonl_record(updates_trace, record)
                update_records.append(record)
                return value

            pre_runner.alg.update = audited_update
            pre_runner._prepare_logging_writer()
            logging_writer_ready = True
            training_rng = seed_policy_rng(
                int(protocol["ppo"]["training_rng_reset_seed"]),
                seed_cuda=str(args.device).startswith("cuda"),
                physics_seed=context["physics_seed"],
            )
            training_rng["state_sha256"] = rng_state_sha256(
                include_cuda=str(args.device).startswith("cuda"),
                cuda_device=args.device,
            )
            pre_runner.learn(num_learning_iterations=config["max_iterations"], init_at_random_ep_len=False)
            if len(update_records) != config["max_iterations"]:
                raise OraclePpoProbeError("PPO update audit count mismatch")
            actor_after = _parameter_sha256(pre_runner.alg.policy.actor)
            noise_after = _policy_noise_summary(pre_runner.alg.policy)
            if actor_after == actor_before:
                raise OraclePpoProbeError("actor parameters did not change during PPO")
            checkpoint = output / "final_checkpoint.pt"
            pre_runner.save(str(checkpoint))
            final_policy = pre_runner.get_inference_policy(device=args.device)
            final = _run_evaluation(
                env, args.mode, final_policy, g2, context["task"], context["training"],
                output / "final_evaluation.jsonl", lambda: assert_execution_time(args),
                physics_seed=context["physics_seed"],
            )
            if final["classification"] == "invalid":
                raise OraclePpoProbeError("final independent evaluation is invalid")
            result = {
                "pre_evaluation": pre,
                "final_evaluation": final,
                "learning": {
                    "executed": True,
                    "rng": rng,
                    "training_start_rng": training_rng,
                    "updates": len(update_records),
                    "update_metrics": update_records,
                    "updates_trace": {
                        "path": str(updates_trace),
                        "sha256": file_sha256(updates_trace),
                    },
                    "initialization": initialization,
                    "critic_initialization": critic_initialization,
                    "zero_actor_contract": zero_actor_contract,
                    "optimizer_initial_state_sha256": optimizer_initial_sha256,
                    "initialization_rng_reset": initialization_rng_reset,
                    "initialization_rng_state_sha256": initialization_rng_sha256,
                    "actor_parameter_sha256_before": actor_before,
                    "actor_parameter_sha256_after": actor_after,
                    "actor_parameters_changed": True,
                    "policy_std_before": noise_before,
                    "policy_std_after": noise_after,
                    "post_constructor_fixed_condition_resets": int(env.diagnostic_true_episode_count),
                    "environment_training_report": env.rj45_training_report(),
                    "checkpoint": {"path": str(checkpoint), "sha256": file_sha256(checkpoint)},
                    "zero_actor_output": True,
                },
            }
        from rby1_assembly.held_grasp import (
            bank_source_contract_sha256,
            held_grasp_bank_dependency_paths,
            load_verified_held_grasp_bank,
        )
        from rby1_assembly.rj45_task import verify_generated_asset_hashes

        end_bank, end_proof = load_verified_held_grasp_bank(
            args.runtime_root,
            context["task"],
            _resolve(args.runtime_root, args.held_bank),
            "train",
            allow_final=False,
        )
        if (
            len(end_bank) != 8
            or end_proof.get("manifest_sha256") != EXPECTED_BANK_SHA256
            or end_proof.get("selected_grasp_ids") != context["bank_proof"].get("selected_grasp_ids")
            or bank_source_contract_sha256(end_proof) != EXPECTED_SOURCE_CONTRACT_SHA256
        ):
            raise OraclePpoProbeError("held bank identity changed during execution")
        end_dependencies = held_grasp_bank_dependency_paths(
            args.runtime_root,
            _resolve(args.runtime_root, args.held_bank),
            "train",
            allow_final=False,
        )
        build_report_path = _resolve(
            args.runtime_root, context["task"]["assets"]["build_report"]
        )
        end_asset_hashes = verify_generated_asset_hashes(
            args.runtime_root,
            context["task"],
            _load_json(build_report_path),
        )
        for path, digest in frozen_hashes.items():
            if file_sha256(path) != digest:
                raise OraclePpoProbeError(f"frozen input changed during execution: {path}")
        result["input_revalidation"] = {
            "held_bank_dependency_count": len(end_dependencies),
            "runtime_source_file_count": len(
                tuple((args.runtime_root / "src/rby1_assembly").glob("*.py"))
            ),
            "generated_asset_sha256": end_asset_hashes,
            "bank_manifest_sha256": end_proof["manifest_sha256"],
            "bank_source_contract_sha256": bank_source_contract_sha256(end_proof),
        }
        result.update({
            "schema_name": SCHEMA_NAME,
            "status": "completed",
            "mode": args.mode,
            "protocol": protocol_record,
            "runtime_versions": context["versions"],
            "runtime_manifest": runtime_manifest_record,
            "runtime_import_origins": runtime_import_origins,
            "execution_device": args.device,
            "frozen_input_sha256": {str(path): digest for path, digest in frozen_hashes.items()},
            "fixed_condition": protocol["fixed_condition"],
            "claims": {
                "single_fixed_condition_learnability_diagnostic": True,
                "reinforcement_learning_effect_proven": False,
                "heldout_generalization": False,
                "final_policy": False,
            },
            "inputs_unchanged": True,
            "execution_guard": {
                "mode_wall_time_limit_s": args.execution_mode_wall_s,
                "absolute_deadline_utc": protocol["budget_and_stop"][
                    "absolute_deadline_utc"
                ],
                "elapsed_wall_s_before_close": (
                    time.monotonic() - args.execution_started_monotonic
                ),
                "external_stop_guard_required": True,
            },
        })
        completed = True
        return result
    finally:
        env.close()
        if not completed:
            torch.cuda.empty_cache()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-protocol", type=Path)
    parser.add_argument("--deadline-utc")
    parser.add_argument("--updates", type=int, default=DEFAULT_UPDATES)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--budget-admission", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--held-bank", type=Path, default=Path("assets/generated/rj45-held-grasps-v1.json"))
    parser.add_argument("--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/rj45-training-target-offset-v1.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=Path("configs/rj45-oracle-runtime-104721a.json"),
    )
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.current_root = args.current_root.resolve()
    if not args.runtime_manifest.is_absolute():
        args.runtime_manifest = (args.current_root / args.runtime_manifest).resolve()
    if args.write_protocol is not None:
        budget_admission = (
            _load_json(args.budget_admission.resolve())
            if args.budget_admission is not None
            else None
        )
        record = write_protocol(
            args.write_protocol,
            args.current_root,
            updates=args.updates,
            deadline_utc=args.deadline_utc,
            base_seed=args.base_seed,
            budget_admission=budget_admission,
        )
        print(json.dumps(record, sort_keys=True))
        return 0
    required = ("runtime_root", "protocol", "protocol_sha256", "mode", "output_dir")
    missing = [f"--{name.replace('_', '-')}" for name in required if getattr(args, name) is None]
    if missing:
        parser.error(f"execution requires: {', '.join(missing)}")
    args.runtime_root = args.runtime_root.resolve()
    report_path = args.output_dir.resolve().with_name(args.output_dir.name + "-report.json")
    if report_path.exists() or args.output_dir.exists():
        parser.error("output directory/report already exists")
    simulation_app = None
    previous_alarm_handler = None
    failure_stage = "protocol_validation"
    exit_code = 1
    try:
        protocol_record = validate_protocol(
            args.protocol.resolve(), args.protocol_sha256, args.current_root
        )
        protocol_base_seed = protocol_record["payload"]["fixed_condition"]["base_seed"]
        if args.base_seed != protocol_base_seed:
            raise OraclePpoProbeError(
                "CLI base seed differs from the frozen protocol fixed condition"
            )
        deadline_utc = protocol_record["payload"]["budget_and_stop"][
            "absolute_deadline_utc"
        ]
        deadline_epoch = _deadline_epoch(deadline_utc)
        if deadline_epoch is None:
            raise OraclePpoProbeError(
                "execution refuses an unscheduled protocol without an absolute UTC deadline"
            )
        remaining_s = deadline_epoch - time.time()
        if remaining_s <= 0.0:
            raise OraclePpoProbeError("protocol absolute UTC deadline already expired")
        args.execution_started_monotonic = time.monotonic()
        args.execution_deadline_epoch = deadline_epoch
        args.execution_mode_wall_s = protocol_record["payload"]["budget_and_stop"][
            "mode_wall_time_limit_s"
        ][args.mode]
        args.runtime_manifest_record = validate_runtime_manifest(
            args.runtime_root, args.runtime_manifest
        )
        if hasattr(signal, "SIGALRM"):
            def alarm_handler(_signum, _frame):
                raise TimeoutError("outer single-mode execution timer expired")

            previous_alarm_handler = signal.signal(signal.SIGALRM, alarm_handler)
            signal.setitimer(
                signal.ITIMER_REAL,
                min(args.execution_mode_wall_s, remaining_s),
            )
        failure_stage = "application_launch"
        from isaaclab.app import AppLauncher

        simulation_app = AppLauncher({"headless": args.headless, "device": args.device}).app
        failure_stage = "physics_and_learning"
        result = run_probe(args)
        failure_stage = "report_write"
        _atomic_write_new(report_path, result)
        print(json.dumps({"status": "completed", "mode": args.mode, "report": str(report_path)}, sort_keys=True))
        exit_code = 0
    except Exception as error:
        failure = {
            "schema_name": SCHEMA_NAME,
            "status": "invalid",
            "mode": args.mode,
            "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "partial_artifacts": (
                [
                    {"path": str(path), "sha256": file_sha256(path)}
                    for path in sorted(args.output_dir.resolve().rglob("*"))
                    if path.is_file()
                ]
                if args.output_dir.resolve().is_dir()
                else []
            ),
            "claims": {"reinforcement_learning_effect_proven": False, "final_policy": False},
        }
        if not report_path.exists():
            _atomic_write_new(report_path, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        exit_code = 2
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            if previous_alarm_handler is not None:
                signal.signal(signal.SIGALRM, previous_alarm_handler)
    return _close_simulation_app(simulation_app, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
