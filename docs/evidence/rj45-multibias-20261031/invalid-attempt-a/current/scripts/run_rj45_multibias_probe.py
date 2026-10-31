#!/usr/bin/env python3
"""Run the preregistered single-bias versus multi-bias RJ45 PPO probe.

The four phases are intentionally separate processes.  Contract and source
validation happen before Isaac Sim is imported; every phase writes a durable
report before Kit is closed.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "rj45_multibias_probe_v1"
CONTRACT_SCHEMA = "rj45_multibias_contract_v1"
MODE = "standard79"
PHASES = ("preflight", "train-single", "train-multi", "evaluate")
METHODS = ("single64", "multi64", "zero")
ANCHOR_SEED = 20261128
SINGLE_POOL = (ANCHOR_SEED,)
MULTI_POOL = (ANCHOR_SEED, 20271208, 20271216, 20271224)
FRESH_EVAL_SEEDS = tuple(20281208 + 8 * index for index in range(8))
EVAL_SEEDS = (ANCHOR_SEED, *FRESH_EVAL_SEEDS)
CALIBRATION_CASE_SEED_OFFSET = 46123020
UPDATES = 64
ROLLOUT_STEPS = 128
TRANSITIONS = UPDATES * ROLLOUT_STEPS
DIAGNOSTIC_CHECKPOINT_UPDATE = 32
TRAIN_WALL_S = 3200
EVALUATE_WALL_S = 4000
PER_EPISODE_WALL_S = 300
EXTERNAL_STOP_MINUTES = 180
HELD_BANK_RELATIVE_PATH = Path("artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json")

EXPECTED_SOURCE_SHA256 = {
    "runner": "132918ca01fa831bb6f703b4cc2aa1d2d9bd73085459a8964495b37e3ef177f1",
    "replay_helper": "46bc89c708afb4ebb664a488e734b121364c9505462200aeddebc73186410bcd",
    "g2_helper": "128a48fabee86b239bedc3db70e244b2a08d052dfc4221a86e4180360294fa42",
    "adapter": "3523687acf7b78ee212f7472329cbc7a3248ec03f451d433fb2d271d6dc42100",
    "ablation_helper": "e23d6dfc2f4e31bb8c54f119d3655d18d63d3df4e5eea2f00812ce72ce39af1b",
    "runtime_manifest": "d672417f55a61a1345dd442d067813fffdfd30274b547c8c103430866dd7b0f2",
}


class MultibiasProbeError(RuntimeError):
    """The sealed multi-bias probe contract was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MultibiasProbeError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise MultibiasProbeError(f"JSON root must be an object: {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MultibiasProbeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise MultibiasProbeError(f"{field} must be 64 lowercase hex characters")
    return value


def _normalize_deadline(value: Any) -> tuple[str, float]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise MultibiasProbeError("deadline_utc must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise MultibiasProbeError("deadline_utc is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise MultibiasProbeError("deadline_utc must be UTC")
    return parsed.isoformat().replace("+00:00", "Z"), parsed.timestamp()


def _source_paths(current_root: Path) -> dict[str, Path]:
    return {
        "runner": current_root / "scripts/run_rj45_oracle_ppo_probe.py",
        "replay_helper": current_root / "scripts/record_rj45_oracle_probe.py",
        "g2_helper": current_root / "scripts/run_rj45_g2_physics_probe.py",
        "adapter": current_root / "src/rby1_assembly/rj45_oracle_diagnostic.py",
        "ablation_helper": current_root / "scripts/evaluate_rj45_frozen_ablation.py",
        "runtime_manifest": current_root / "configs/rj45-oracle-runtime-104721a.json",
    }


def _case(case_id: str, base_seed: int, *, source: str) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "base_seed": base_seed,
        "calibration_case_seed": base_seed + CALIBRATION_CASE_SEED_OFFSET,
        "split": "train",
        "grasp_id": "rj45-train-02",
        "episode_index": 1,
        "source": source,
    }


def contract_payload(
    current_root: Path,
    origin_protocol: Path,
    deadline_utc: str,
    budget_admission: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the exact contract payload that the execution phases validate."""

    current_root, origin_protocol = current_root.resolve(), origin_protocol.resolve()
    deadline_utc, _ = _normalize_deadline(deadline_utc)
    required_budget = {"budget_cap_usd", "new_run_reservation_usd", "external_guard_minutes", "private_evidence_sha256", "locally_verified"}
    if set(budget_admission) != required_budget:
        raise MultibiasProbeError("budget admission keys changed")
    if budget_admission["budget_cap_usd"] != 700 or budget_admission["external_guard_minutes"] != EXTERNAL_STOP_MINUTES:
        raise MultibiasProbeError("budget cap or external guard changed")
    reservation = budget_admission["new_run_reservation_usd"]
    if isinstance(reservation, bool) or not isinstance(reservation, (int, float)) or not math.isfinite(reservation) or not 0 <= reservation <= 700:
        raise MultibiasProbeError("new-run reservation must be finite and within the 700 USD cap")
    _sha256_hex(budget_admission["private_evidence_sha256"], "private_evidence_sha256")
    if budget_admission["locally_verified"] is not True:
        raise MultibiasProbeError("budget admission must be locally verified")
    paths = _source_paths(current_root)
    for name, path in paths.items():
        if not path.is_file() or file_sha256(path) != EXPECTED_SOURCE_SHA256[name]:
            raise MultibiasProbeError(f"source binding failed while building contract: {name}")
    if not origin_protocol.is_file():
        raise MultibiasProbeError("origin protocol is missing")
    cases = [_case("anchor", ANCHOR_SEED, source="training-anchor")]
    cases.extend(_case(f"fresh-{index + 1:02d}", seed, source="policy-unseen") for index, seed in enumerate(FRESH_EVAL_SEEDS))
    return {
        "schema_name": CONTRACT_SCHEMA,
        "helper_sha256": file_sha256(Path(__file__).resolve()),
        "source_sha256": {name: file_sha256(path) for name, path in paths.items()},
        "origin_protocol": {"path_name": origin_protocol.name, "sha256": file_sha256(origin_protocol), "template_only": True},
        "training": {
            "mode": MODE, "num_envs": 1, "actor_observation_dim": 79,
            "critic_observation_dim": 92, "action_dim": 6, "right_arm_only": True,
            "held_bank_relative_path": str(HELD_BANK_RELATIVE_PATH),
            "updates": UPDATES, "rollout_steps": ROLLOUT_STEPS, "transitions": TRANSITIONS,
            "single_pool": list(SINGLE_POOL), "multi_pool": list(MULTI_POOL),
            "policy_training_seed": 20261121, "critic_initialization_seed": 20261122,
            "actor_initialization_seed": 20261123, "evaluation_seed": 20261124,
            "final_checkpoint_update": 64, "eligible_checkpoint_updates": [64],
            "diagnostic_checkpoint_updates": [32], "checkpoint_selection": "final64_only_no_outcome_selection",
        },
        "reference_preflight": {
            "pool": list(MULTI_POOL), "required_successes": 4, "episode_count": 4,
            "failure_policy": "no_replacement_no_ranking",
        },
        "evaluation": {
            "cases": cases, "methods": list(METHODS), "episode_count": 27,
            "order": "case_outer_cyclic_method_offset_case_index_mod_3",
            "per_episode_wall_time_s": PER_EPISODE_WALL_S,
        },
        "budget_and_stop": {
            "admission": dict(budget_admission), "deadline_utc": deadline_utc,
            "train_phase_wall_time_s": TRAIN_WALL_S, "evaluate_phase_wall_time_s": EVALUATE_WALL_S,
            "external_stop_wall_time_s": EXTERNAL_STOP_MINUTES * 60,
        },
        "decision_rule": {
            "followup_multiseed_candidate": "fresh8_multi_successes_at_least_single_plus_2_and_greater_than_zero_and_zero_success_multi_failure_count_at_most_1",
            "effect_proven": False, "checkpoint_replacement_allowed": False,
        },
        "claims": {"final_split_sealed": True, "dev_split_sealed": True, "effect_proven": False, "generalization_proven": False},
    }


def validate_contract(path: Path, expected_sha256: str, *, current_root: Path, origin_protocol: Path) -> dict[str, Any]:
    if file_sha256(path) != _sha256_hex(expected_sha256, "contract SHA-256"):
        raise MultibiasProbeError("contract file SHA-256 mismatch")
    observed = _load_json(path)
    budget = observed.get("budget_and_stop", {}).get("admission")
    deadline = observed.get("budget_and_stop", {}).get("deadline_utc")
    expected = contract_payload(current_root, origin_protocol, deadline, budget)
    if observed != expected:
        raise MultibiasProbeError("contract payload changed or source binding drifted")
    return observed


def validate_preimport_sources(current_root: Path, origin_protocol: Path, contract: Mapping[str, Any]) -> dict[str, Any]:
    observed: dict[str, Any] = {}
    for name, path in _source_paths(current_root).items():
        digest = file_sha256(path)
        if digest != contract["source_sha256"][name] or digest != EXPECTED_SOURCE_SHA256[name]:
            raise MultibiasProbeError(f"pre-import source binding failed: {name}")
        observed[name] = {"path": str(path.resolve()), "sha256": digest}
    protocol_sha = file_sha256(origin_protocol)
    if protocol_sha != contract["origin_protocol"]["sha256"]:
        raise MultibiasProbeError("origin protocol source binding failed")
    observed["origin_protocol"] = {"path": str(origin_protocol.resolve()), "sha256": protocol_sha}
    return observed


def _atomic_write(path: Path, payload: Mapping[str, Any], *, exclusive: bool = True) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _append_jsonl(stream: Any, payload: Mapping[str, Any]) -> None:
    stream.write(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")


def _tensor_list(value: Any) -> list[Any]:
    return value.detach().cpu().tolist()


class ResetSchedule:
    """Phase-aware reset instrumentation around the environment's bound reset."""

    def __init__(self, env: Any, pool: Sequence[Mapping[str, Any]], expected_bias: Mapping[str, Mapping[str, Any]], log_path: Path):
        if not pool:
            raise MultibiasProbeError("reset pool cannot be empty")
        self.env = env
        self.pool = [dict(case) for case in pool]
        self.expected_bias = expected_bias
        self.log_path = log_path
        self.active = False
        self.next_index = 0
        self.reset_count = 0
        self.reset_count_by_case = {case["case_id"]: 0 for case in self.pool}
        self.original = env._reset_idx
        env._reset_idx = types.MethodType(self._wrapped, env)

    def activate(self) -> None:
        if self.active or self.reset_count:
            raise MultibiasProbeError("reset schedule can only be activated once")
        self.active = True
        self.next_index = 0

    def deactivate(self) -> None:
        self.active = False

    def _wrapped(self, _env: Any, env_ids: Any = None) -> Any:
        if not self.active:
            return self.original(env_ids)
        pool_index = self.next_index
        case = self.pool[pool_index]
        self.next_index = (pool_index + 1) % len(self.pool)
        self.env._training_bias_base_seed = int(case["base_seed"])
        result = self.original(env_ids)
        self._validate_and_record(case, pool_index)
        return result

    def _validate_and_record(self, case: Mapping[str, Any], pool_index: int) -> None:
        episode = _tensor_list(self.env._training_episode_index)
        grasp = _tensor_list(self.env._training_grasp_index)
        translation = _tensor_list(self.env._training_translation_bias_entry)
        rotation = _tensor_list(self.env._training_rotation_bias_entry)
        if episode != [1] or grasp != [1]:
            raise MultibiasProbeError("scheduled reset did not preserve episode1/grasp1")
        expected = self.expected_bias[case["case_id"]]
        if translation != [expected["translation_entry_m"]] or rotation != [expected["rotation_vector_entry_rad"]]:
            raise MultibiasProbeError("scheduled reset bias differs from independent CUDA sample")
        record = {
            "phase": "training", "ordinal": self.reset_count, "pool_index": pool_index,
            "case_id": case["case_id"], "base_seed": case["base_seed"],
            "calibration_case_seed": case["calibration_case_seed"],
            "actual_translation_entry_m": translation[0], "actual_rotation_vector_entry_rad": rotation[0],
            "episode_index": episode[0], "grasp_index": grasp[0],
        }
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            _append_jsonl(stream, record)
            stream.flush()
            os.fsync(stream.fileno())
        self.reset_count += 1
        self.reset_count_by_case[case["case_id"]] += 1


class TrainingStepAudit:
    """Record the case that produced each transition, including auto-reset steps."""

    def __init__(self, env: Any, schedule: ResetSchedule, g2: Any, path: Path, tensor_hash: Any = None):
        self.env, self.schedule, self.g2, self.path = env, schedule, g2, path
        self.original = env.step
        self.total = 0
        self.by_case = {case["case_id"]: {"transitions": 0, "active_valid": 0, "terminal": 0} for case in schedule.pool}
        self.first_active_base79: dict[str, dict[str, Any]] = {}
        self.tensor_hash = tensor_hash
        self.stream = path.open("x", encoding="utf-8")
        env.step = self.step

    def step(self, action: Any) -> Any:
        if not self.schedule.active:
            return self.original(action)
        case = self.schedule.pool[(self.schedule.next_index - 1) % len(self.schedule.pool)]
        observations = self.env.get_observations()
        base79 = observations["critic"][:, :79]
        result = self.original(action)
        if not isinstance(result, tuple) or len(result) != 5:
            raise MultibiasProbeError("environment step result contract changed")
        _next, reward, terminated, truncated, extras = result
        done = bool((terminated | truncated)[0].item())
        saved = extras.get("terminal_snapshot")
        if not isinstance(saved, Mapping):
            raise MultibiasProbeError("training terminal snapshot missing")
        transition = self.g2._transition_contract(saved, base79, done)
        case_id = case["case_id"]
        self.total += 1
        self.by_case[case_id]["transitions"] += 1
        self.by_case[case_id]["terminal"] += int(done)
        self.by_case[case_id]["active_valid"] += int(transition["active"] and transition["valid_input"])
        if transition["active"] and transition["valid_input"] and case_id not in self.first_active_base79:
            values = _tensor_list(base79)[0]
            if self.tensor_hash is None:
                raise MultibiasProbeError("training audit requires the frozen tensor hash function")
            self.first_active_base79[case_id] = {"values": values, "sha256": self.tensor_hash(base79)}
        _append_jsonl(self.stream, {
            "transition": self.total, "case_id": case_id,
            "reward": float(reward[0].item()), "done": done,
            "valid_input": bool(transition["valid_input"]), "active": bool(transition["active"]),
            "action": _tensor_list(action)[0], "applied_expanded_action": transition["applied_expanded"],
        })
        if self.total % ROLLOUT_STEPS == 0:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        return result

    def close(self) -> None:
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()

    def validate(self) -> dict[str, Any]:
        if self.total != TRANSITIONS or sum(item["transitions"] for item in self.by_case.values()) != TRANSITIONS:
            raise MultibiasProbeError("training transition total differs from 8192")
        missing = [case_id for case_id, item in self.by_case.items() if item["active_valid"] <= 0]
        if missing:
            raise MultibiasProbeError(f"training pool cases had no active valid exposure: {missing}")
        return {"total_transitions": self.total, "by_case": self.by_case, "first_active_base79": self.first_active_base79}


def _training_cases(pool: Sequence[int]) -> list[dict[str, Any]]:
    names = {ANCHOR_SEED: "anchor"}
    return [_case(names.get(seed, f"train-{index + 1:02d}"), seed, source="training-pool") for index, seed in enumerate(pool)]


def _time_guard(started: float, deadline_epoch: float, wall_s: int, episode_started: float | None = None) -> None:
    now = time.monotonic()
    if now - started > wall_s or time.time() >= deadline_epoch:
        raise TimeoutError("global phase guard exceeded")
    if episode_started is not None and now - episode_started > PER_EPISODE_WALL_S:
        raise TimeoutError("per-episode guard exceeded")


def _configure_probe_args(args: argparse.Namespace, probe: Any, protocol: Mapping[str, Any]) -> None:
    args.protocol = args.origin_protocol
    args.protocol_sha256 = args.origin_protocol_sha256
    args.runtime_manifest = args.current_root / "configs/rj45-oracle-runtime-104721a.json"
    args.runtime_manifest_record = probe.validate_runtime_manifest(args.runtime_root, args.runtime_manifest)
    args.held_bank = HELD_BANK_RELATIVE_PATH
    args.task_config = Path("configs/tasks/rj45-v1.json")
    args.training_config = Path("configs/rj45-training-target-offset-v1.json")
    args.runtime_config = Path("configs/runtime.json")
    if protocol["fixed_condition"]["base_seed"] != ANCHOR_SEED:
        raise MultibiasProbeError("origin protocol is not the anchor condition")


def _bias_cases(ablation: Any, env: Any, cases: Sequence[Mapping[str, Any]], torch: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    sampled = ablation.sample_bias_cases(env, cases, torch)
    return sampled, {record["case_id"]: record for record in sampled}


def validate_final_checkpoint_payload(payload: Any, label: str) -> dict[str, Any]:
    """Reject intermediate or structurally incomplete RSL checkpoints."""

    if not isinstance(payload, Mapping) or set(payload) != {"model_state_dict", "optimizer_state_dict", "iter", "infos"}:
        raise MultibiasProbeError(f"{label} checkpoint structure changed")
    if type(payload["iter"]) is not int or payload["iter"] != UPDATES - 1:
        raise MultibiasProbeError(f"{label} is not the final64 checkpoint")
    if not isinstance(payload["model_state_dict"], Mapping) or not payload["model_state_dict"]:
        raise MultibiasProbeError(f"{label} checkpoint model state is empty")
    if not isinstance(payload["optimizer_state_dict"], Mapping):
        raise MultibiasProbeError(f"{label} checkpoint optimizer state changed")
    return {"iteration_field": payload["iter"], "eligible_update": UPDATES, "eligible_for_evaluation": True}


def _run_preflight(args: argparse.Namespace, contract: Mapping[str, Any], probe: Any, ablation: Any, g2: Any, protocol: Mapping[str, Any], started: float, deadline: float) -> dict[str, Any]:
    import torch
    env, context, frozen = probe._build_env(args, "reference", protocol)
    try:
        cases = _training_cases(MULTI_POOL)
        sampled, expected = _bias_cases(ablation, env, cases, torch)
        episodes = []
        for index, case in enumerate(cases):
            episode_started = time.monotonic()
            env._training_bias_base_seed = case["base_seed"]
            evaluation = probe._run_evaluation(
                env, "reference", None, g2, context["task"], context["training"],
                args.output_dir / "traces" / f"{index + 1:02d}-{case['case_id']}-reference.jsonl",
                lambda: _time_guard(started, deadline, TRAIN_WALL_S, episode_started),
                physics_seed=ANCHOR_SEED,
            )
            observed = env.rj45_training_report()["calibration_bias"]
            wanted = expected[case["case_id"]]
            if observed["translation_entry_m"] != [wanted["translation_entry_m"]] or observed["rotation_vector_entry_rad"] != [wanted["rotation_vector_entry_rad"]]:
                raise MultibiasProbeError("reference preflight bias mismatch")
            episodes.append({"case": case, "evaluation": evaluation, "observed_bias": observed})
            if evaluation["classification"] != "valid_success":
                raise MultibiasProbeError("reference preflight did not succeed 4/4; ranking is disabled")
        return {"status": "completed", "phase": args.phase, "episodes": episodes, "bias_cases": sampled, "strict_successes": 4, "ranking_eligible": True, "frozen_input_count": len(frozen)}
    finally:
        env.close()


def _run_training(args: argparse.Namespace, contract: Mapping[str, Any], probe: Any, ablation: Any, g2: Any, protocol: Mapping[str, Any], started: float, deadline: float) -> dict[str, Any]:
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from rby1_assembly.rj45_bc import build_bc_actor, object_sha256

    pool = SINGLE_POOL if args.phase == "train-single" else MULTI_POOL
    cases = _training_cases(pool)
    env, context, frozen = probe._build_env(args, MODE, protocol)
    audit: TrainingStepAudit | None = None
    try:
        sampled, expected = _bias_cases(ablation, env, cases, torch)
        schedule = ResetSchedule(env, cases, expected, args.output_dir / "resets.jsonl")
        config = copy.deepcopy(probe.runner_config(MODE, protocol))
        config["max_iterations"] = UPDATES
        config["save_interval"] = UPDATES
        probe.seed_policy_rng(20261121, seed_cuda=True, physics_seed=ANCHOR_SEED)
        runner = OnPolicyRunner(RslRlVecEnvWrapper(env, clip_actions=1.0), config, log_dir=str(args.output_dir / "ppo_logs"), device=args.device)
        actor_init = probe.paired_initialize_actor(runner.alg.policy, MODE, seed=20261123, device=args.device, baseline_builder=build_bc_actor)
        critic_init = probe.reinitialize_critic(runner.alg.policy, seed=20261122, device=args.device)
        zero_contract = probe.verify_zero_actor_contract(runner.alg.policy, MODE, device=args.device)
        optimizer_initial = object_sha256(runner.alg.optimizer.state_dict())
        initialization_rng = probe.seed_policy_rng(20261121, seed_cuda=True, physics_seed=ANCHOR_SEED)
        initialization_rng["state_sha256"] = probe.rng_state_sha256(include_cuda=True, cuda_device=args.device)
        actor_before = probe._parameter_sha256(runner.alg.policy.actor)
        pre = probe._run_evaluation(
            env, MODE, runner.get_inference_policy(device=args.device), g2, context["task"], context["training"],
            args.output_dir / "pre-evaluation.jsonl", lambda: _time_guard(started, deadline, TRAIN_WALL_S), physics_seed=ANCHOR_SEED,
        )
        if pre["classification"] == "invalid":
            raise MultibiasProbeError("pre-training evaluation is invalid")
        schedule.activate()
        with torch.inference_mode():
            env.reset()
        if schedule.reset_count != 1 or schedule.next_index != (1 % len(cases)):
            raise MultibiasProbeError("explicit training reset did not consume pool[0] exactly once")
        training_rng = probe.seed_policy_rng(20261121, seed_cuda=True, physics_seed=ANCHOR_SEED)
        training_rng["state_sha256"] = probe.rng_state_sha256(include_cuda=True, cuda_device=args.device)
        audit = TrainingStepAudit(env, schedule, g2, args.output_dir / "transitions.jsonl", probe.tensor_sha256)
        update_records: list[dict[str, Any]] = []
        original_update = runner.alg.update
        logging_ready = False

        def audited_update(*positional: Any, **keywords: Any) -> Any:
            _time_guard(started, deadline, TRAIN_WALL_S)
            rollout = probe.assert_finite_rollout_targets(runner.alg.storage)
            value = original_update(*positional, **keywords)
            record = probe.finite_update_record(len(update_records) + 1, value)
            record["rollout_targets"] = rollout
            record["parameter_finiteness"] = probe.assert_finite_policy_parameters(runner.alg.policy)
            if record["update"] == DIAGNOSTIC_CHECKPOINT_UPDATE:
                if not logging_ready:
                    raise MultibiasProbeError("logging writer was not prepared before checkpoint32")
                path = args.output_dir / "diagnostic_checkpoint_0032.pt"
                runner.save(str(path))
                record["diagnostic_checkpoint"] = {"path": str(path), "sha256": file_sha256(path), "eligible_for_selection": False}
            update_records.append(record)
            with (args.output_dir / "updates.jsonl").open("a", encoding="utf-8") as stream:
                _append_jsonl(stream, record)
                stream.flush(); os.fsync(stream.fileno())
            return value

        runner.alg.update = audited_update
        runner._prepare_logging_writer()
        logging_ready = True
        runner.learn(num_learning_iterations=UPDATES, init_at_random_ep_len=False)
        schedule.deactivate()
        audit.close()
        audit_summary = audit.validate()
        if len(update_records) != UPDATES:
            raise MultibiasProbeError("PPO update count differs from 64")
        actor_after = probe._parameter_sha256(runner.alg.policy.actor)
        if actor_after == actor_before:
            raise MultibiasProbeError("actor parameters did not change")
        final_checkpoint = args.output_dir / "final_checkpoint.pt"
        runner.save(str(final_checkpoint))
        for path, digest in frozen.items():
            if file_sha256(path) != digest:
                raise MultibiasProbeError(f"frozen input changed during training: {path}")
        return {
            "status": "completed", "phase": args.phase, "learning_executed": True,
            "training_pool": cases, "bias_cases": sampled, "updates": len(update_records),
            "transitions": audit_summary, "reset_count": schedule.reset_count,
            "reset_count_by_case": schedule.reset_count_by_case,
            "initialization": {"actor": actor_init, "critic": critic_init, "zero_actor": zero_contract, "optimizer_sha256": optimizer_initial, "rng": initialization_rng},
            "training_rng": training_rng, "actor_parameter_sha256_before": actor_before,
            "actor_parameter_sha256_after": actor_after,
            "checkpoint": {"path": str(final_checkpoint), "sha256": file_sha256(final_checkpoint), "update": 64, "eligible_for_evaluation": True},
            "diagnostic_checkpoint": {"path": str(args.output_dir / "diagnostic_checkpoint_0032.pt"), "sha256": file_sha256(args.output_dir / "diagnostic_checkpoint_0032.pt"), "eligible_for_selection": False},
        }
    finally:
        if audit is not None and not audit.stream.closed:
            audit.close()
        env.close()


def _run_evaluate(args: argparse.Namespace, contract: Mapping[str, Any], probe: Any, ablation: Any, replay: Any, g2: Any, protocol: Mapping[str, Any], started: float, deadline: float) -> dict[str, Any]:
    import torch
    env, context, frozen = probe._build_env(args, MODE, protocol)
    try:
        checkpoint_records = {}
        for label, path in (("single64", args.single_checkpoint), ("multi64", args.multi_checkpoint)):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_records[label] = {
                "path": str(path), "sha256": file_sha256(path),
                **validate_final_checkpoint_payload(payload, label),
            }
        single = replay._policy_for_replay(env, MODE, args.single_checkpoint, protocol, probe)
        multi = replay._policy_for_replay(env, MODE, args.multi_checkpoint, protocol, probe)
        policies = {"single64": single, "multi64": multi, "zero": ablation.AblationPolicy(env, "zero", [[0.0] * 6], torch)}
        cases = contract["evaluation"]["cases"]
        sampled, expected = _bias_cases(ablation, env, cases, torch)
        episodes, first_by_case = [], {}
        for case_index, case in enumerate(cases):
            order = list(METHODS[case_index % 3:] + METHODS[:case_index % 3])
            for method in order:
                episode_started = time.monotonic()
                env._training_bias_base_seed = case["base_seed"]
                trace = args.output_dir / "traces" / f"{len(episodes) + 1:02d}-{case['case_id']}-{method}.jsonl"
                evaluation = probe._run_evaluation(
                    env, MODE, policies[method], g2, context["task"], context["training"], trace,
                    lambda: _time_guard(started, deadline, EVALUATE_WALL_S, episode_started), physics_seed=ANCHOR_SEED,
                )
                observed = env.rj45_training_report()["calibration_bias"]
                wanted = expected[case["case_id"]]
                if observed["translation_entry_m"] != [wanted["translation_entry_m"]] or observed["rotation_vector_entry_rad"] != [wanted["rotation_vector_entry_rad"]]:
                    raise MultibiasProbeError("evaluation bias differs from independent sample")
                base79 = ablation.first_active_base79(trace)
                base_hash = ablation.raw_base79_sha256(base79)
                if base_hash != evaluation["observation"]["base79_sha256"]:
                    raise MultibiasProbeError("evaluation raw79 hash mismatch")
                if case["case_id"] in first_by_case and first_by_case[case["case_id"]] != base79:
                    raise MultibiasProbeError("paired evaluation initial raw79 differs")
                first_by_case.setdefault(case["case_id"], base79)
                if evaluation["classification"] == "invalid":
                    raise MultibiasProbeError("evaluation classified invalid")
                episode = {"index": len(episodes) + 1, "case": case, "method": method, "evaluation": evaluation, "initial_base79_sha256": base_hash, "bias": observed}
                _atomic_write(args.output_dir / "episodes" / f"{len(episodes) + 1:02d}-{case['case_id']}-{method}.json", episode)
                episodes.append(episode)
                _atomic_write(args.output_dir / "partial-results.json", {"status": "running", "episodes": episodes}, exclusive=False)
        fresh = [episode for episode in episodes if episode["case"]["source"] == "policy-unseen"]
        successes = {method: sum(item["method"] == method and item["evaluation"]["classification"] == "valid_success" for item in fresh) for method in METHODS}
        zero_success_multi_failure = sum(
            any(item["case"]["case_id"] == case_id and item["method"] == "zero" and item["evaluation"]["classification"] == "valid_success" for item in fresh)
            and any(item["case"]["case_id"] == case_id and item["method"] == "multi64" and item["evaluation"]["classification"] != "valid_success" for item in fresh)
            for case_id in {item["case"]["case_id"] for item in fresh}
        )
        candidate = successes["multi64"] >= successes["single64"] + 2 and successes["multi64"] > successes["zero"] and zero_success_multi_failure <= 1
        for path, digest in frozen.items():
            if file_sha256(path) != digest:
                raise MultibiasProbeError(f"frozen input changed during evaluation: {path}")
        return {
            "status": "completed", "phase": args.phase, "episodes": episodes,
            "bias_cases": sampled, "fresh8_successes": successes,
            "zero_success_multi_failure_count": zero_success_multi_failure,
            "followup_multiseed_candidate": candidate, "effect_proven": False,
            "checkpoint_selection_changed": False,
            "checkpoints": checkpoint_records,
        }
    finally:
        env.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--origin-protocol", type=Path, required=True)
    parser.add_argument("--origin-protocol-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--single-checkpoint", type=Path)
    parser.add_argument("--multi-checkpoint", type=Path)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    return parser


def persist_report_before_app_close(report_path: Path, report: Mapping[str, Any], probe: Any, app: Any, exit_code: int) -> int:
    _atomic_write(report_path, report)
    print(json.dumps({"status": report.get("status"), "phase": report.get("phase"), "report": str(report_path)}, sort_keys=True), flush=True)
    return probe._close_simulation_app(app, exit_code)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.current_root, args.runtime_root = args.current_root.resolve(), args.runtime_root.resolve()
    args.origin_protocol, args.contract, args.output_dir = args.origin_protocol.resolve(), args.contract.resolve(), args.output_dir.resolve()
    report_path = args.output_dir.with_name(args.output_dir.name + "-report.json")
    if args.output_dir.exists() or report_path.exists():
        raise SystemExit("output directory/report already exists")
    contract = validate_contract(args.contract, args.contract_sha256, current_root=args.current_root, origin_protocol=args.origin_protocol)
    if file_sha256(args.origin_protocol) != args.origin_protocol_sha256 or args.origin_protocol_sha256 != contract["origin_protocol"]["sha256"]:
        raise MultibiasProbeError("CLI origin protocol SHA-256 mismatch")
    preimport = validate_preimport_sources(args.current_root, args.origin_protocol, contract)
    if args.phase == "evaluate":
        if args.single_checkpoint is None or args.multi_checkpoint is None:
            raise SystemExit("evaluate requires --single-checkpoint and --multi-checkpoint")
        args.single_checkpoint, args.multi_checkpoint = args.single_checkpoint.resolve(), args.multi_checkpoint.resolve()
        if not args.single_checkpoint.is_file() or not args.multi_checkpoint.is_file():
            raise MultibiasProbeError("evaluation checkpoint is missing")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    probe = _load_module(Path(preimport["runner"]["path"]), "rj45_multibias_origin_probe")
    ablation = _load_module(Path(preimport["ablation_helper"]["path"]), "rj45_multibias_ablation")
    replay = _load_module(Path(preimport["replay_helper"]["path"]), "rj45_multibias_replay")
    protocol_record = probe.validate_protocol(args.origin_protocol, args.origin_protocol_sha256, args.current_root)
    protocol = protocol_record["payload"]
    _configure_probe_args(args, probe, protocol)
    deadline_text, deadline_epoch = _normalize_deadline(contract["budget_and_stop"]["deadline_utc"])
    if time.time() >= deadline_epoch:
        raise MultibiasProbeError("contract deadline has expired")
    started = time.monotonic()
    wall_s = EVALUATE_WALL_S if args.phase == "evaluate" else TRAIN_WALL_S
    app = None
    previous_alarm = None
    report: dict[str, Any]
    exit_code = 1
    failure_stage = "application_launch"
    try:
        if hasattr(signal, "SIGALRM"):
            previous_alarm = signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("outer phase timer expired")))
            signal.setitimer(signal.ITIMER_REAL, min(wall_s, deadline_epoch - time.time()))
        from isaaclab.app import AppLauncher
        app = AppLauncher({"headless": args.headless, "device": args.device}).app
        sys.path.insert(0, str(args.runtime_root / "src"))
        g2 = _load_module(Path(preimport["g2_helper"]["path"]), "rj45_multibias_g2")
        failure_stage = args.phase
        if args.phase == "preflight":
            payload = _run_preflight(args, contract, probe, ablation, g2, protocol, started, deadline_epoch)
        elif args.phase in ("train-single", "train-multi"):
            payload = _run_training(args, contract, probe, ablation, g2, protocol, started, deadline_epoch)
        else:
            payload = _run_evaluate(args, contract, probe, ablation, replay, g2, protocol, started, deadline_epoch)
        runtime_origins = probe.verify_runtime_import_origins(args.runtime_root)
        for name, record in preimport.items():
            if file_sha256(record["path"]) != record["sha256"]:
                raise MultibiasProbeError(f"input changed during phase: {name}")
        report = {
            "schema_name": SCHEMA_NAME, **payload,
            "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
            "source_binding": preimport, "runtime_import_origins": runtime_origins,
            "origin_protocol": protocol_record,
            "execution": {"deadline_utc": deadline_text, "elapsed_wall_s_before_close": time.monotonic() - started},
            "claims": {"effect_proven": False, "generalization_proven": False, "final_policy": False},
        }
        exit_code = 0
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_name": SCHEMA_NAME,
            "status": "incomplete" if isinstance(error, TimeoutError) else "invalid",
            "phase": args.phase, "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "partial_artifacts": [{"path": str(path), "sha256": file_sha256(path)} for path in sorted(args.output_dir.rglob("*")) if path.is_file()],
            "claims": {"effect_proven": False, "generalization_proven": False, "ranking_allowed": False},
        }
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_alarm is not None:
                signal.signal(signal.SIGALRM, previous_alarm)
    return persist_report_before_app_close(report_path, report, probe, app, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
