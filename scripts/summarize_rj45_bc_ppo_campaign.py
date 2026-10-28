#!/usr/bin/env python3
"""Audit and summarize the frozen RJ45 PPO-only versus BC→PPO campaign."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any


SOURCE_COMMIT = "8512a5932547b5495325b5ff0b6894fb072db98e"
TEACHER_SOURCE_COMMIT = "104721acf2222c1cfe8ec86cd030f5116a4fef93"
SEEDS = (20261131, 20261132, 20261133)
CHECKPOINTS = {19: 20, 95: 96, 199: 200}
CONDITIONS = ("ppo_only", "bc_then_ppo")
ATTEMPTS = 64
DEV_SEED = 20261122
TRAIN_ENVS = 256
STEPS_PER_ENV = 32
TRAINING_UPDATES = 200
BC_REPORT_SHA256 = "32aa18808dfb9ae2ccc6bb4019f24e9c5782005ee69c99889af7ee2816467bbf"
FROZEN_CAMPAIGN_RUNNER_SHA256 = (
    "9828c2c9564aa25174466067d62a5e209ce9b29bf73e313c6b8f777aca91f3dc"
)
F_CONFIG_SHA256 = "c909830f5bce676b8372c272adad25796c9dee3a9c744546eba786eac3739519"
TRAINING_CONFIG_SHA256 = (
    "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179"
)
PAIR_KEYS = (
    "evaluation_start",
    "task_config_sha256",
    "base_evaluation_seed",
    "environment_index",
    "episode_index",
    "calibration_bias_level",
    "calibration_latent_case_seed",
    "calibration_latent_case_seed_formula",
    "grasp_index",
    "grasp_id",
    "bank_manifest_sha256",
    "bank_split",
)
PAIR_KEY_TEXT = ",".join(PAIR_KEYS)
LATENT_FORMULA = "base_seed + stream_offset + episode_index * episode_stride + env_id"
CALIBRATION_STREAM_OFFSET = 45_123_017
CALIBRATION_EPISODE_STRIDE = 1_000_003
DEV_GRASP_IDS = ("rj45-dev-01", "rj45-dev-02")


class CampaignAuditError(ValueError):
    """Raised when campaign evidence differs from the frozen contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CampaignAuditError(f"JSON root must be an object: {path}")
    return value


def load_verified_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CampaignAuditError(f"{label} SHA differs")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise CampaignAuditError(f"{label} JSON root must be an object")
    return value


def load_json_with_sha(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise CampaignAuditError(f"{label} JSON root must be an object")
    return value, hashlib.sha256(payload).hexdigest()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def checked_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignAuditError(f"{label} path is missing")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise CampaignAuditError(f"{label} path escapes snapshot root")
    return resolved


def recorded_snapshot_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CampaignAuditError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        resolved = (root / path).resolve()
    elif path.resolve().is_relative_to(root.resolve()):
        resolved = path.resolve()
    else:
        marker = "rby1-assembly-lab-rj45-data"
        parts = path.parts
        if marker not in parts:
            raise CampaignAuditError(f"{label} absolute path has no portable repository anchor")
        suffix = parts[parts.index(marker) + 1 :]
        resolved = root.joinpath(*suffix).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise CampaignAuditError(f"{label} path escapes snapshot root")
    return resolved


def finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CampaignAuditError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise CampaignAuditError(f"{label} must be finite")
    return result


def require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CampaignAuditError(f"{label} must be a lowercase SHA-256")
    return value


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {str(value): percentile(value) for value in (0.0, 0.25, 0.5, 0.75, 1.0)}


def verify_source_map(
    repo: Path, source_map: object, expected_source_map: dict[str, str]
) -> dict[str, str]:
    if (
        not isinstance(source_map, dict)
        or source_map != expected_source_map
        or len(expected_source_map) != 36
    ):
        raise CampaignAuditError("evaluation source map differs from the frozen 36-file map")
    verified: dict[str, str] = {}
    for relative, digest in source_map.items():
        if not isinstance(relative, str) or not isinstance(digest, str) or len(digest) != 64:
            raise CampaignAuditError("evaluation source map is malformed")
        payload = subprocess.check_output(
            ["git", "show", f"{SOURCE_COMMIT}:{relative}"], cwd=repo
        )
        if hashlib.sha256(payload).hexdigest() != digest:
            raise CampaignAuditError(f"source map differs from {SOURCE_COMMIT}: {relative}")
        verified[relative] = digest
    return verified


def validate_report(
    report: dict[str, Any],
    *,
    mode: str,
    expected_source: dict[str, str],
    expected_task_sha: str,
    expected_bank_sha: str,
    expected_config_sha: str | None = None,
    expected_checkpoint_sha: str | None = None,
    expected_bc_sha: str | None = None,
    expected_bias_by_id: dict[str, tuple[list[float], list[float]]] | None = None,
    expected_bank_proofs: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    outcomes = report.get("episode_outcomes")
    provenance = report.get("provenance")
    bank = report.get("evaluation_held_grasp_bank_proof")
    training_bank = report.get("training_held_grasp_bank_proof")
    training_config = report.get("rj45_training_config")
    expected_bank_proof = {
        "selected_split": "dev",
        "selected_count": 2,
        "selected_grasp_ids": list(DEV_GRASP_IDS),
        "manifest_split_counts": {"train": 8, "dev": 2, "final": 2},
        "final_opened": False,
        "no_final_opened": True,
    }
    if (
        report.get("rj45_evaluation_schema_version") != 2
        or report.get("baseline_mode") != mode
        or report.get("seed") != DEV_SEED
        or report.get("requested_attempts") != ATTEMPTS
        or report.get("attempts") != ATTEMPTS
        or report.get("evaluation_num_envs") != ATTEMPTS
        or report.get("per_environment_episode_quota") != [1] * ATTEMPTS
        or report.get("per_environment_accepted_attempt_count") != [1] * ATTEMPTS
        or not isinstance(outcomes, list)
        or len(outcomes) != ATTEMPTS
        or not isinstance(provenance, dict)
        or provenance.get("source_sha256") != expected_source
        or provenance.get("task_config_sha256") != expected_task_sha
        or provenance.get("evaluation_inputs_reverified_at_end") is not True
        or not isinstance(bank, dict)
        or bank.get("manifest_sha256") != expected_bank_sha
        or bank.get("selected_split") != "dev"
        or any(bank.get(key) != value for key, value in expected_bank_proof.items())
        or not isinstance(bank.get("selected_artifact_sha256"), list)
        or len(bank["selected_artifact_sha256"]) != 2
        or any(
            not isinstance(value, str) or len(value) != 64
            for value in bank["selected_artifact_sha256"]
        )
        or bank.get("provenance_source_task_match") is not True
        or not isinstance(bank.get("provenance_contract"), dict)
        or bank["provenance_contract"].get("task_config_sha256") != expected_task_sha
        or not isinstance(
            bank["provenance_contract"].get("source_and_config_sha256"), dict
        )
        or any(
            bank["provenance_contract"]["source_and_config_sha256"].get(path)
            != digest
            for path, digest in expected_source.items()
            if path.startswith("src/")
        )
        or not isinstance(training_config, dict)
        or training_config.get("sha256") != TRAINING_CONFIG_SHA256
        or training_config.get("action_mode") != "right_only_6d"
        or training_config.get("observation_dimension") != 79
        or training_config.get("calibration_bias_level") != 0
        or report.get("paired_condition_key") != PAIR_KEY_TEXT
    ):
        raise CampaignAuditError("evaluation report contract differs")
    if mode == "biased_zero":
        if training_bank is not None:
            raise CampaignAuditError("biased-zero training held-grasp proof must be null")
    elif (
        not isinstance(training_bank, dict)
        or training_bank.get("manifest_sha256") != expected_bank_sha
        or training_bank.get("selected_split") != "train"
        or training_bank.get("selected_count") != 8
        or training_bank.get("selected_grasp_ids")
        != [f"rj45-train-{index:02d}" for index in range(1, 9)]
        or training_bank.get("manifest_split_counts")
        != {"train": 8, "dev": 2, "final": 2}
        or training_bank.get("final_opened") is not False
        or training_bank.get("no_final_opened") is not True
        or training_bank.get("provenance_source_task_match") is not True
    ):
        raise CampaignAuditError("training held-grasp bank proof differs")
    config_sha = provenance.get("policy_config_sha256")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise CampaignAuditError("evaluation policy config SHA is missing")
    if expected_config_sha is not None and config_sha != expected_config_sha:
        raise CampaignAuditError("evaluation policy config SHA differs")
    if mode == "learned":
        if expected_checkpoint_sha is None or provenance.get("checkpoint_sha256") != expected_checkpoint_sha:
            raise CampaignAuditError("learned report checkpoint binding differs")
    elif provenance.get("checkpoint_sha256") is not None:
        raise CampaignAuditError("baseline report unexpectedly opened a checkpoint")
    if mode == "bc":
        evidence = provenance.get("bc_actor")
        if (
            expected_bc_sha is None
            or not isinstance(evidence, dict)
            or evidence.get("sha256") != expected_bc_sha
        ):
            raise CampaignAuditError("BC report actor binding differs")
    if expected_bank_proofs is not None and (bank, training_bank) != expected_bank_proofs:
        raise CampaignAuditError("held-grasp bank proofs differ across paired reports")

    identities: list[str] = []
    successes = 0
    invalid = 0
    forbidden_peak = 0.0
    lateral: list[float] = []
    successful_elapsed: list[float] = []
    failed_elapsed: list[float] = []
    bias_by_id: dict[str, tuple[list[float], list[float]]] = {}
    environments: set[int] = set()
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, dict):
            raise CampaignAuditError("episode outcome is not an object")
        valid = outcome.get("valid")
        success = outcome.get("success")
        if type(valid) is not bool or type(success) is not bool or success and not valid:
            raise CampaignAuditError("episode valid/success flags are invalid")
        if outcome.get("attempt_index") != index:
            raise CampaignAuditError("episode attempts are not complete and ordered")
        condition = outcome.get("paired_condition")
        if not isinstance(condition, dict) or set(condition) != set(PAIR_KEYS):
            raise CampaignAuditError("paired condition is missing")
        canonical = json.dumps(condition, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if (
            outcome.get("paired_condition_id") != canonical
            or condition.get("evaluation_start") != "held"
            or condition.get("base_evaluation_seed") != DEV_SEED
            or condition.get("task_config_sha256") != expected_task_sha
            or condition.get("bank_manifest_sha256") != expected_bank_sha
            or condition.get("bank_split") != "dev"
        ):
            raise CampaignAuditError("paired condition identity differs")
        environment = condition.get("environment_index")
        episode = condition.get("episode_index")
        grasp_index = condition.get("grasp_index")
        if (
            type(environment) is not int
            or not 0 <= environment < ATTEMPTS
            or type(episode) is not int
            or episode != 1
            or type(condition.get("calibration_bias_level")) is not int
            or condition["calibration_bias_level"] != 0
            or condition.get("calibration_latent_case_seed_formula") != LATENT_FORMULA
            or type(condition.get("calibration_latent_case_seed")) is not int
            or condition["calibration_latent_case_seed"]
            != DEV_SEED + CALIBRATION_STREAM_OFFSET + CALIBRATION_EPISODE_STRIDE + environment
            or type(grasp_index) is not int
            or grasp_index not in (0, 1)
            or condition.get("grasp_id") != DEV_GRASP_IDS[grasp_index]
            or type(outcome.get("environment_index")) is not int
            or outcome.get("environment_index") != environment
            or type(outcome.get("episode_index")) is not int
            or outcome.get("episode_index") != episode
        ):
            raise CampaignAuditError("paired condition environment, episode, bias, or grasp differs")
        environments.add(environment)
        identities.append(canonical)
        if valid and success:
            successes += 1
        if not valid:
            invalid += 1
        forbidden_peak = max(
            forbidden_peak,
            finite_number(outcome.get("forbidden_contact_load_peak_n"), "forbidden peak"),
        )
        lateral.append(finite_number(outcome.get("final_lateral_error_m"), "lateral error"))
        elapsed = finite_number(outcome.get("elapsed_s"), "elapsed time")
        (successful_elapsed if valid and success else failed_elapsed).append(elapsed)
        translation = outcome.get("calibration_bias_translation_entry_m")
        rotation = outcome.get("calibration_bias_rotation_vector_entry_rad")
        if (
            not isinstance(translation, list)
            or not isinstance(rotation, list)
            or len(translation) != 3
            or len(rotation) != 3
        ):
            raise CampaignAuditError("calibration bias vectors are missing")
        translation_values = [finite_number(value, "translation bias") for value in translation]
        rotation_values = [finite_number(value, "rotation bias") for value in rotation]
        bias_by_id[canonical] = (translation_values, rotation_values)
        if expected_bias_by_id is not None and expected_bias_by_id.get(canonical) != bias_by_id[canonical]:
            raise CampaignAuditError("actual calibration bias differs for a paired condition")
        if mode == "bc" and outcome.get("bc_actor_sha256") != expected_bc_sha:
            raise CampaignAuditError("BC outcome actor binding differs")
        if mode != "bc" and outcome.get("bc_actor_sha256") is not None:
            raise CampaignAuditError("non-BC outcome unexpectedly names a BC actor")
    if len(set(identities)) != ATTEMPTS:
        raise CampaignAuditError("paired condition identities are not unique")
    if environments != set(range(ATTEMPTS)):
        raise CampaignAuditError("paired conditions do not cover each environment exactly once")
    failures = ATTEMPTS - successes - invalid
    if (
        report.get("valid_successes") != successes
        or report.get("valid_failures") != failures
        or report.get("invalid_episodes") != invalid
        or report.get("valid_episodes") != ATTEMPTS - invalid
    ):
        raise CampaignAuditError("reported outcome aggregates differ from all attempts")
    force_proxy = finite_number(report.get("max_contact_force_proxy_n"), "force proxy maximum")
    horizon = finite_number(report.get("evaluation_horizon_s"), "evaluation horizon")
    return {
        "successes": successes,
        "failures": failures,
        "invalid": invalid,
        "forbidden_contact_peak_n": forbidden_peak,
        "force_proxy_max_n": force_proxy,
        "evaluation_horizon_s": horizon,
        "success_elapsed_s": quantiles(successful_elapsed),
        "failed_elapsed_s": quantiles(failed_elapsed),
        "final_lateral_error_m": quantiles(lateral),
        "paired_condition_ids": sorted(identities),
        "calibration_bias_by_paired_condition": bias_by_id,
        "evaluation_bank_proof": bank,
        "training_bank_proof": training_bank,
        "policy_config_sha256": config_sha,
    }


def promotion_gate(
    *,
    baseline: int,
    ppo_only: list[int],
    bc_then_ppo: list[int],
    invalid_counts: list[int],
    forbidden_peaks: list[float],
) -> dict[str, Any]:
    if len(ppo_only) != 3 or len(bc_then_ppo) != 3:
        raise CampaignAuditError("promotion gate requires three paired seeds")
    if any(type(value) is not int for value in [baseline, *ppo_only, *bc_then_ppo, *invalid_counts]):
        raise CampaignAuditError("promotion counts must be integers")
    if any(value < 0 or value > ATTEMPTS for value in [baseline, *ppo_only, *bc_then_ppo]):
        raise CampaignAuditError("promotion success count is outside the 64-case contract")
    if any(not math.isfinite(value) or value < 0.0 for value in forbidden_peaks):
        raise CampaignAuditError("promotion forbidden-contact peak is invalid")
    ppo_mean = sum(ppo_only) / 3.0
    bc_mean = sum(bc_then_ppo) / 3.0
    passed = (
        bc_mean >= baseline + 6.0
        and bc_mean >= ppo_mean + 6.0
        and all(value == 0 for value in invalid_counts)
        and all(value == 0.0 for value in forbidden_peaks)
    )
    return {
        "baseline_successes": baseline,
        "ppo_only_successes": ppo_only,
        "ppo_only_mean": ppo_mean,
        "bc_then_ppo_successes": bc_then_ppo,
        "bc_then_ppo_mean": bc_mean,
        "required_margin_successes_per_64": 6,
        "invalid_counts": invalid_counts,
        "forbidden_contact_peaks_n": forbidden_peaks,
        "pass": passed,
    }


def require_gate_match(summary: dict[str, Any], recomputed: dict[str, Any]) -> None:
    if (
        summary.get("promotion_gate") != recomputed
        or summary.get("promotion_allowed") != recomputed["pass"]
    ):
        raise CampaignAuditError("runner promotion gate differs from recomputation")


def public_report_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key
        not in {
            "paired_condition_ids",
            "calibration_bias_by_paired_condition",
            "evaluation_bank_proof",
            "training_bank_proof",
        }
    }


def validate_prelaunch(
    *,
    snapshot_root: Path,
    campaign_root: Path,
    frozen_plan: dict[str, Any],
    expected_source_map: dict[str, str],
) -> tuple[dict[int, dict[str, str]], str]:
    prelaunch_path = campaign_root / "prelaunch-manifest.json"
    prelaunch, prelaunch_sha = load_json_with_sha(prelaunch_path, "prelaunch manifest")
    generated = prelaunch.get("generated_configs")
    paired = prelaunch.get("paired_training")
    profile = prelaunch.get("profile")
    frozen_inputs = prelaunch.get("frozen_input_sha256")
    plan_inputs = frozen_plan.get("input_sha256")
    commands = prelaunch.get("commands")
    if (
        prelaunch.get("schema_version") != 1
        or prelaunch.get("source_commit") != SOURCE_COMMIT
        or prelaunch.get("final_seed_opened") is not False
        or not isinstance(generated, dict)
        or set(generated) != {str(seed) for seed in SEEDS}
        or paired
        != {
            "seeds": list(SEEDS),
            "num_envs": TRAIN_ENVS,
            "steps_per_env": STEPS_PER_ENV,
            "updates": TRAINING_UPDATES,
            "save_interval": 19,
        }
        or not isinstance(profile, dict)
        or profile.get("name") != "F"
        or profile.get("sha256") != F_CONFIG_SHA256
        or not isinstance(frozen_inputs, dict)
        or not isinstance(plan_inputs, dict)
        or not isinstance(commands, dict)
        or any(frozen_inputs.get(path) != digest for path, digest in plan_inputs.items())
        or any(frozen_inputs.get(path) != digest for path, digest in expected_source_map.items())
        or frozen_inputs.get("configs/rj45-training-target-offset-v1.json")
        != TRAINING_CONFIG_SHA256
    ):
        raise CampaignAuditError("prelaunch manifest differs from the frozen plan")
    base_config_path = checked_path(
        snapshot_root, profile.get("source_config"), "F source config"
    )
    if sha256(base_config_path) != F_CONFIG_SHA256:
        raise CampaignAuditError("F source config SHA differs")
    training_config_path = snapshot_root / "configs/rj45-training-target-offset-v1.json"
    if sha256(training_config_path) != TRAINING_CONFIG_SHA256:
        raise CampaignAuditError("RJ45 training config SHA differs")
    base_config = load_json(base_config_path)
    records: dict[int, dict[str, str]] = {}
    bc_actor_path = frozen_plan.get("bc_actor_path")
    bc_actor_sha = frozen_plan.get("bc_actor_sha256")
    if not isinstance(bc_actor_path, str) or require_sha256(
        bc_actor_sha, "frozen BC actor SHA"
    ) != bc_actor_sha:
        raise CampaignAuditError("frozen BC actor binding is missing")

    def option_value(command: list[str], option: str) -> str:
        if command.count(option) != 1:
            raise CampaignAuditError(f"prelaunch command {option} count differs")
        index = command.index(option)
        if index + 1 >= len(command) or command[index + 1].startswith("--"):
            raise CampaignAuditError(f"prelaunch command {option} value is missing")
        return command[index + 1]

    def common_command(command: list[str]) -> list[str]:
        result: list[str] = []
        index = 0
        value_options = {"--log-root", "--bc-actor-init", "--bc-actor-init-sha256"}
        while index < len(command):
            part = command[index]
            if part in value_options:
                if index + 1 >= len(command):
                    raise CampaignAuditError(f"prelaunch command {part} value is missing")
                index += 2
                continue
            if part == "--zero-init-actor-output":
                index += 1
                continue
            result.append(part)
            index += 1
        return result

    for seed in SEEDS:
        record = generated[str(seed)]
        if not isinstance(record, dict):
            raise CampaignAuditError("generated config record is malformed")
        path = checked_path(snapshot_root, record.get("path"), "generated policy config")
        digest = require_sha256(record.get("sha256"), "generated policy config SHA")
        derived = load_verified_json(path, digest, "generated policy config")
        expected = json.loads(json.dumps(base_config))
        expected["seed"] = seed
        expected["num_steps_per_env"] = STEPS_PER_ENV
        expected["max_iterations"] = TRAINING_UPDATES
        if (
            derived != expected
            or derived.get("save_interval") != 19
            or frozen_inputs.get(str(path.relative_to(snapshot_root))) != digest
        ):
            raise CampaignAuditError(f"generated policy config differs for seed {seed}")
        records[seed] = {"path": str(path.relative_to(snapshot_root)), "sha256": digest}
        condition_commands: dict[str, list[str]] = {}
        for condition in CONDITIONS:
            command = commands.get(f"{condition}-{seed}")
            if not isinstance(command, list) or not all(
                isinstance(part, str) for part in command
            ):
                raise CampaignAuditError("prelaunch training command is missing")
            forbidden = {"--resume", "--checkpoint", "--load-run"}
            if forbidden.intersection(command):
                raise CampaignAuditError("prelaunch training command is not fresh")
            if (
                option_value(command, "--config") != records[seed]["path"]
                or option_value(command, "--num-envs") != str(TRAIN_ENVS)
                or option_value(command, "--max-iterations") != str(TRAINING_UPDATES)
                or option_value(command, "--rj45-training-config")
                != "configs/rj45-training-target-offset-v1.json"
            ):
                raise CampaignAuditError("prelaunch training command fixed arguments differ")
            option_value(command, "--log-root")
            if condition == "ppo_only":
                if (
                    command.count("--zero-init-actor-output") != 1
                    or "--bc-actor-init" in command
                    or "--bc-actor-init-sha256" in command
                ):
                    raise CampaignAuditError("PPO-only initialization command differs")
            elif (
                "--zero-init-actor-output" in command
                or option_value(command, "--bc-actor-init") != bc_actor_path
                or option_value(command, "--bc-actor-init-sha256") != bc_actor_sha
            ):
                raise CampaignAuditError("BC→PPO initialization command differs")
            condition_commands[condition] = command
        if common_command(condition_commands["ppo_only"]) != common_command(
            condition_commands["bc_then_ppo"]
        ):
            raise CampaignAuditError("paired training common command differs")
    return records, prelaunch_sha


def validate_bank_manifest(
    snapshot_root: Path, frozen_plan: dict[str, Any], expected_sha256: str
) -> dict[str, list[str]]:
    path = checked_path(
        snapshot_root, frozen_plan.get("bank_manifest_path"), "held-grasp bank manifest"
    )
    manifest = load_verified_json(path, expected_sha256, "held-grasp bank manifest")
    entries = manifest.get("entries")
    if manifest.get("schema_version") != 1 or not isinstance(entries, list):
        raise CampaignAuditError("held-grasp bank manifest schema differs")
    result: dict[str, list[str]] = {}
    for split, count in (("train", 8), ("dev", 2), ("final", 2)):
        selected = [entry for entry in entries if isinstance(entry, dict) and entry.get("split") == split]
        if len(selected) != count:
            raise CampaignAuditError(f"held-grasp bank {split} count differs")
        ids: list[str] = []
        artifacts: list[str] = []
        for entry in selected:
            grasp_id = entry.get("grasp_id")
            artifact = entry.get("artifact")
            if not isinstance(grasp_id, str) or not isinstance(artifact, dict):
                raise CampaignAuditError("held-grasp bank entry is malformed")
            ids.append(grasp_id)
            artifacts.append(require_sha256(artifact.get("sha256"), "held-grasp artifact SHA"))
        result[f"{split}_ids"] = ids
        result[f"{split}_artifact_sha256"] = artifacts
    return result


def validate_initialization(
    initialization: object,
    *,
    condition: str,
    expected_bc_sha: str,
    expected_bc_metadata: dict[str, Any] | None = None,
) -> None:
    if not isinstance(initialization, dict):
        raise CampaignAuditError("training policy initialization is missing")
    if condition == "ppo_only":
        required = {
            "mode": "residual_actor_final_linear_zero",
            "output_dimension": 6,
            "zero_actor_output_layer": True,
            "weight_all_zero_after_initialization": True,
            "bias_all_zero_after_initialization": True,
            "deterministic_actor_output_all_zero": True,
            "deterministic_actor_output_max_abs": 0.0,
            "stochastic_policy_noise_modified": False,
        }
        if any(initialization.get(key) != value for key, value in required.items()):
            raise CampaignAuditError("PPO-only zero actor initialization differs")
        return
    before = initialization.get("hashes_before")
    after = initialization.get("hashes_after")
    if expected_bc_metadata is None:
        raise CampaignAuditError("BC initialization lineage expectation is missing")
    if (
        initialization.get("mode") != "bc_actor_warm_start"
        or initialization.get("output_dimension") != 6
        or initialization.get("artifact_file_sha256") != expected_bc_sha
        or initialization.get("bc_artifact_sha256") != expected_bc_sha
        or initialization.get("loaded_component") != "runner.alg.policy.actor"
        or initialization.get("artifact_actor_state_sha256")
        != expected_bc_metadata.get("actor_state_sha256")
        or initialization.get("bc_provenance")
        != expected_bc_metadata.get("provenance")
        or not isinstance(before, dict)
        or not isinstance(after, dict)
        or after.get("actor") != initialization.get("artifact_actor_state_sha256")
        or any(before.get(key) != after.get(key) for key in ("critic", "noise", "optimizer"))
    ):
        raise CampaignAuditError("BC actor-only initialization differs")


def validate_checkpoint_bundle(
    *,
    snapshot_root: Path,
    inspection: dict[str, Any],
    report: dict[str, Any],
    condition: str,
    seed: int,
    generated_config: dict[str, str],
    expected_source_map: dict[str, str],
    plan_inputs: dict[str, str],
    expected_bc_sha: str,
    expected_bc_metadata: dict[str, Any],
) -> dict[str, str]:
    checkpoint_record = inspection.get("checkpoint")
    manifest_record = inspection.get("checkpoint_manifest")
    provenance_record = inspection.get("training_provenance")
    runner_record = inspection.get("runner_config")
    if not isinstance(checkpoint_record, dict):
        raise CampaignAuditError("checkpoint inspection checkpoint reference is incomplete")
    if not isinstance(manifest_record, dict):
        raise CampaignAuditError("checkpoint inspection manifest reference is incomplete")
    if not isinstance(provenance_record, dict):
        raise CampaignAuditError("checkpoint inspection provenance reference is incomplete")
    if not isinstance(runner_record, dict):
        raise CampaignAuditError("checkpoint inspection bundle references are incomplete")
    records = {
        "checkpoint": checkpoint_record,
        "checkpoint_manifest": manifest_record,
        "training_provenance": provenance_record,
        "runner_config": runner_record,
    }
    paths = {
        name: recorded_snapshot_path(snapshot_root, record.get("path"), name)
        for name, record in records.items()
    }
    hashes: dict[str, str] = {}
    checkpoint_bytes = paths["checkpoint"].read_bytes()
    for name, record in records.items():
        digest = require_sha256(record.get("sha256"), f"{name} SHA")
        actual_digest = (
            hashlib.sha256(checkpoint_bytes).hexdigest()
            if name == "checkpoint"
            else sha256(paths[name])
        )
        if actual_digest != digest:
            raise CampaignAuditError(f"{name} bytes differ from checkpoint inspection")
        hashes[name] = digest
    manifest = load_json(paths["checkpoint_manifest"])
    provenance = load_json(paths["training_provenance"])
    runner = load_json(paths["runner_config"])
    checkpoint_name = paths["checkpoint"].name
    checkpoints = manifest.get("checkpoints")
    source_and_config = provenance.get("source_and_config_sha256")
    generated_path = generated_config["path"]
    runner_config_pins = (
        [
            digest
            for path, digest in source_and_config.items()
            if isinstance(path, str) and Path(path).name == paths["runner_config"].name
        ]
        if isinstance(source_and_config, dict)
        else []
    )
    if (
        manifest.get("schema_version") != 1
        or manifest.get("training_completed") is not True
        or manifest.get("training_provenance_sha256") != hashes["training_provenance"]
        or not isinstance(checkpoints, dict)
        or checkpoints.get(checkpoint_name) != hashes["checkpoint"]
        or provenance.get("seed") != seed
        or provenance.get("num_envs") != TRAIN_ENVS
        or provenance.get("checkpoint_scope") != "checkpoints created in this run directory"
        or provenance.get("runtime_versions", {}).get("rsl-rl-lib") != "3.0.1"
        or not isinstance(source_and_config, dict)
        or any(
            source_and_config.get(path) != digest
            for path, digest in expected_source_map.items()
            if path.startswith("src/rby1_assembly/")
        )
        or source_and_config.get(generated_path) != generated_config["sha256"]
        or source_and_config.get("scripts/train.py") != plan_inputs.get("scripts/train.py")
        or runner_config_pins != [hashes["runner_config"]]
        or runner.get("seed") != seed
        or runner.get("max_iterations") != TRAINING_UPDATES
        or runner.get("num_steps_per_env") != STEPS_PER_ENV
    ):
        raise CampaignAuditError("checkpoint manifest, provenance, or runner config differs")
    match = re.fullmatch(r"model_(\d+)\.pt", checkpoint_name)
    if match is None:
        raise CampaignAuditError("checkpoint filename does not encode an iteration")
    try:
        import torch

        checkpoint_payload = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise CampaignAuditError("checkpoint payload cannot be loaded safely") from error
    payload_iteration = (
        checkpoint_payload.get("iter") if isinstance(checkpoint_payload, dict) else None
    )
    filename_iteration = int(match.group(1))
    if (
        type(payload_iteration) is not int
        or payload_iteration != filename_iteration
        or checkpoint_record.get("payload_iteration") != payload_iteration
        or checkpoint_record.get("completed_updates") != payload_iteration + 1
    ):
        raise CampaignAuditError("checkpoint payload iteration differs from inspection")
    validate_initialization(
        provenance.get("policy_initialization"),
        condition=condition,
        expected_bc_sha=expected_bc_sha,
        expected_bc_metadata=expected_bc_metadata,
    )
    report_provenance = report.get("provenance")
    if not isinstance(report_provenance, dict):
        raise CampaignAuditError("evaluation report provenance is missing")
    evidence = report_provenance.get("verified_checkpoint_evidence")
    if (
        not isinstance(evidence, dict)
        or evidence.get("training_provenance_sha256") != hashes["training_provenance"]
        or evidence.get("checkpoint_manifest_sha256") != hashes["checkpoint_manifest"]
        or report_provenance.get("checkpoint_sha256") != hashes["checkpoint"]
    ):
        raise CampaignAuditError("evaluation report checkpoint evidence differs")
    return hashes


def summarize_campaign(
    *,
    repo: Path,
    snapshot_root: Path,
    campaign_summary: Path,
    campaign_summary_sha256: str,
    plan: Path,
    plan_sha256: str,
    bc_training_report: Path,
    bc_training_report_sha256: str,
) -> dict[str, Any]:
    snapshot_root = snapshot_root.resolve()
    for path, label in (
        (campaign_summary, "campaign summary"),
        (plan, "campaign plan"),
        (bc_training_report, "BC training report"),
    ):
        if not path.resolve().is_relative_to(snapshot_root):
            raise CampaignAuditError(f"{label} escapes snapshot root")
    summary = load_verified_json(
        campaign_summary, campaign_summary_sha256, "campaign summary"
    )
    frozen_plan = load_verified_json(plan, plan_sha256, "campaign plan")
    if bc_training_report_sha256 != BC_REPORT_SHA256:
        raise CampaignAuditError("BC training report SHA differs")
    bc_report = load_verified_json(
        bc_training_report, bc_training_report_sha256, "BC training report"
    )
    if (
        summary.get("status") != "measurement_complete"
        or summary.get("source_commit") != SOURCE_COMMIT
        or summary.get("diagnostic_checkpoints") != [20, 96]
        or summary.get("selection_checkpoint") != 200
        or summary.get("inputs_unchanged") is not True
        or summary.get("final_seed_opened") is not False
        or frozen_plan.get("source_commit") != SOURCE_COMMIT
        or frozen_plan.get("teacher_source_commit") != TEACHER_SOURCE_COMMIT
        or frozen_plan.get("runner_script_sha256")
        != FROZEN_CAMPAIGN_RUNNER_SHA256
        or frozen_plan.get("final_policy_seed_opened") is not False
    ):
        raise CampaignAuditError("campaign summary or plan contract differs")
    expected_bank_sha = require_sha256(
        frozen_plan.get("bank_manifest_sha256"), "bank manifest SHA"
    )
    input_sha256 = frozen_plan.get("input_sha256")
    if not isinstance(input_sha256, dict):
        raise CampaignAuditError("plan input SHA map is missing")
    expected_task_sha = require_sha256(
        input_sha256.get("configs/tasks/rj45-v1.json"), "task config SHA"
    )
    expected_bc_sha = require_sha256(
        frozen_plan.get("bc_actor_sha256"), "BC actor SHA"
    )
    bc_artifact = bc_report.get("artifact")
    bc_lineage = bc_report.get("lineage")
    if (
        not isinstance(bc_artifact, dict)
        or not isinstance(bc_lineage, dict)
        or bc_artifact.get("sha256") != expected_bc_sha
        or bc_lineage.get("task_sha256") != expected_task_sha
        or bc_lineage.get("training_sha256") != TRAINING_CONFIG_SHA256
        or bc_lineage.get("bank_sha256") != expected_bank_sha
    ):
        raise CampaignAuditError("BC training lineage differs from the campaign plan")
    expected_bc_metadata = {
        "actor_state_sha256": require_sha256(
            bc_artifact.get("actor_state_sha256"), "BC actor state SHA"
        ),
        "provenance": {
            "source_sha256": require_sha256(
                bc_lineage.get("source_sha256"), "BC source contract SHA"
            ),
            "task_sha256": expected_task_sha,
            "training_sha256": TRAINING_CONFIG_SHA256,
            "bank_sha256": expected_bank_sha,
            "dataset_sha256": require_sha256(
                frozen_plan.get("train_dataset_sha256"), "BC train dataset SHA"
            ),
        },
    }
    bank_manifest = validate_bank_manifest(
        snapshot_root, frozen_plan, expected_bank_sha
    )
    expected_source_map = {
        path: digest
        for path, digest in input_sha256.items()
        if path == "scripts/evaluate_policy.py"
        or path.startswith("src/rby1_assembly/") and path.endswith(".py")
    }
    if len(expected_source_map) != 36:
        raise CampaignAuditError("plan does not pin the exact 36-file evaluation source map")

    reports = summary.get("reports")
    baselines = summary.get("baseline_reports")
    expected_report_keys = {
        f"{condition}-{seed}-{iteration}"
        for condition in CONDITIONS
        for seed in SEEDS
        for iteration in CHECKPOINTS
    }
    if not isinstance(reports, dict) or set(reports) != expected_report_keys:
        raise CampaignAuditError("campaign must contain exactly 18 learned reports")
    if not isinstance(baselines, dict) or set(baselines) != {"biased_zero", "bc"}:
        raise CampaignAuditError("campaign must contain biased-zero and BC-only baselines")

    campaign_root = campaign_summary.parent
    generated_configs, prelaunch_sha = validate_prelaunch(
        snapshot_root=snapshot_root,
        campaign_root=campaign_root,
        frozen_plan=frozen_plan,
        expected_source_map=expected_source_map,
    )
    all_references = {
        **reports,
        **{f"baseline-{key}": value for key, value in baselines.items()},
    }
    report_paths: dict[str, Path] = {}
    report_payloads: dict[str, dict[str, Any]] = {}
    for key, reference in all_references.items():
        if not isinstance(reference, dict):
            raise CampaignAuditError("report reference is malformed")
        path = checked_path(snapshot_root, reference.get("path"), "report")
        expected = reference.get("sha256")
        if not isinstance(expected, str):
            raise CampaignAuditError("report SHA is missing")
        report_paths[key] = path
        report_payloads[key] = load_verified_json(
            path, expected, "report before data collection"
        )

    biased_payload = report_payloads["baseline-biased_zero"]
    source_map = verify_source_map(
        repo,
        biased_payload.get("provenance", {}).get("source_sha256"),
        expected_source_map,
    )
    biased = validate_report(
        biased_payload,
        mode="biased_zero",
        expected_source=source_map,
        expected_task_sha=expected_task_sha,
        expected_bank_sha=expected_bank_sha,
        expected_config_sha=generated_configs[SEEDS[0]]["sha256"],
    )
    if (
        biased["evaluation_bank_proof"].get("selected_grasp_ids")
        != bank_manifest["dev_ids"]
        or biased["evaluation_bank_proof"].get("selected_artifact_sha256")
        != bank_manifest["dev_artifact_sha256"]
    ):
        raise CampaignAuditError("evaluation held-grasp proofs differ from bank manifest")
    bc_only = validate_report(
        report_payloads["baseline-bc"],
        mode="bc",
        expected_source=source_map,
        expected_task_sha=expected_task_sha,
        expected_bank_sha=expected_bank_sha,
        expected_config_sha=generated_configs[SEEDS[0]]["sha256"],
        expected_bc_sha=expected_bc_sha,
        expected_bias_by_id=biased["calibration_bias_by_paired_condition"],
    )
    bc_training_proof = bc_only["training_bank_proof"]
    if (
        bc_only["evaluation_bank_proof"] != biased["evaluation_bank_proof"]
        or not isinstance(bc_training_proof, dict)
        or bc_training_proof.get("selected_grasp_ids")
        != bank_manifest["train_ids"]
        or bc_training_proof.get("selected_artifact_sha256")
        != bank_manifest["train_artifact_sha256"]
    ):
        raise CampaignAuditError("BC held-grasp proofs differ from bank manifest")
    reference_ids = biased["paired_condition_ids"]
    if bc_only["paired_condition_ids"] != reference_ids:
        raise CampaignAuditError("baseline paired IDs differ")
    identity_sha = hashlib.sha256("\n".join(reference_ids).encode()).hexdigest()
    if summary.get("paired_condition_ids_sha256") != identity_sha:
        raise CampaignAuditError("summary paired ID SHA differs")

    learned_rows: list[dict[str, Any]] = []
    inspection_hashes: dict[str, str] = {}
    per_seed_config: dict[int, str] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for iteration, completed_updates in CHECKPOINTS.items():
                key = f"{condition}-{seed}-{iteration}"
                inspection_path = campaign_root / "inspections" / f"{key}.json"
                inspection, inspection_sha = load_json_with_sha(
                    inspection_path, "checkpoint inspection"
                )
                details = inspection.get("checkpoint")
                if (
                    inspection.get("schema_version") != 2
                    or inspection.get("status")
                    != "checkpoint_payload_and_lineage_verified"
                    or inspection.get("planned_training_updates") != TRAINING_UPDATES
                    or inspection.get("loader")
                    != {"map_location": "cpu", "weights_only": True}
                    or not isinstance(details, dict)
                    or details.get("filename_iteration") != iteration
                    or details.get("payload_iteration") != iteration
                    or details.get("completed_updates") != completed_updates
                    or not isinstance(details.get("sha256"), str)
                ):
                    raise CampaignAuditError(f"checkpoint inspection differs: {key}")
                reference = reports[key]
                report_path = report_paths[key]
                bundle_hashes = validate_checkpoint_bundle(
                    snapshot_root=snapshot_root,
                    inspection=inspection,
                    report=report_payloads[key],
                    condition=condition,
                    seed=seed,
                    generated_config=generated_configs[seed],
                    expected_source_map=expected_source_map,
                    plan_inputs=input_sha256,
                    expected_bc_sha=expected_bc_sha,
                    expected_bc_metadata=expected_bc_metadata,
                )
                inspection_hashes[key] = inspection_sha
                metrics = validate_report(
                    report_payloads[key],
                    mode="learned",
                    expected_source=source_map,
                    expected_task_sha=expected_task_sha,
                    expected_bank_sha=expected_bank_sha,
                    expected_checkpoint_sha=details["sha256"],
                    expected_bias_by_id=biased[
                        "calibration_bias_by_paired_condition"
                    ],
                    expected_bank_proofs=(
                        biased["evaluation_bank_proof"],
                        bc_training_proof,
                    ),
                )
                if metrics["paired_condition_ids"] != reference_ids:
                    raise CampaignAuditError(f"learned paired IDs differ: {key}")
                prior_config = per_seed_config.setdefault(seed, metrics["policy_config_sha256"])
                if (
                    prior_config != metrics["policy_config_sha256"]
                    or metrics["policy_config_sha256"]
                    != generated_configs[seed]["sha256"]
                ):
                    raise CampaignAuditError(f"policy config changes within seed {seed}")
                learned_rows.append(
                    {
                        "condition": condition,
                        "training_seed": seed,
                        "checkpoint_iteration": iteration,
                        "completed_updates": completed_updates,
                        "report_path": str(report_path.relative_to(snapshot_root)),
                        "report_sha256": reference["sha256"],
                        "inspection_path": str(inspection_path.relative_to(snapshot_root)),
                        "inspection_sha256": inspection_hashes[key],
                        "checkpoint_bundle_sha256": bundle_hashes,
                        **{
                            metric_name: value
                            for metric_name, value in metrics.items()
                            if metric_name
                            not in {
                                "paired_condition_ids",
                                "calibration_bias_by_paired_condition",
                                "evaluation_bank_proof",
                                "training_bank_proof",
                            }
                        },
                    }
                )

    aggregates: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        aggregates[condition] = {}
        for completed in CHECKPOINTS.values():
            rows = [
                row
                for row in learned_rows
                if row["condition"] == condition and row["completed_updates"] == completed
            ]
            counts = [int(row["successes"]) for row in rows]
            aggregates[condition][str(completed)] = {
                "successes_by_training_seed": {
                    str(row["training_seed"]): row["successes"] for row in rows
                },
                "success_mean_over_three_training_seeds": sum(counts) / 3.0,
                "invalid_by_training_seed": {
                    str(row["training_seed"]): row["invalid"] for row in rows
                },
                "force_proxy_max_n_by_training_seed": {
                    str(row["training_seed"]): row["force_proxy_max_n"] for row in rows
                },
                "final_lateral_error_m_by_training_seed": {
                    str(row["training_seed"]): row["final_lateral_error_m"] for row in rows
                },
                "successful_episode_elapsed_s_by_training_seed": {
                    str(row["training_seed"]): row["success_elapsed_s"] for row in rows
                },
                "failed_episode_elapsed_s_by_training_seed": {
                    str(row["training_seed"]): row["failed_elapsed_s"] for row in rows
                },
            }

    final_rows = {
        condition: [
            row for row in learned_rows if row["condition"] == condition and row["completed_updates"] == 200
        ]
        for condition in CONDITIONS
    }
    gate_reports = [biased, bc_only, *final_rows["ppo_only"], *final_rows["bc_then_ppo"]]
    recomputed_gate = promotion_gate(
        baseline=biased["successes"],
        ppo_only=[row["successes"] for row in final_rows["ppo_only"]],
        bc_then_ppo=[row["successes"] for row in final_rows["bc_then_ppo"]],
        invalid_counts=[row["invalid"] for row in gate_reports],
        forbidden_peaks=[row["forbidden_contact_peak_n"] for row in gate_reports],
    )
    require_gate_match(summary, recomputed_gate)

    if (
        bc_report.get("status") != "completed"
        or bc_report.get("sample_visits") != 3_833_040
        or bc_report.get("total_bc_updates") != 3_760
    ):
        raise CampaignAuditError("BC compute accounting differs")
    ppo_transitions_per_run = TRAIN_ENVS * STEPS_PER_ENV * TRAINING_UPDATES
    return {
        "schema_version": 1,
        "status": "campaign_audit_complete",
        "source_commit": SOURCE_COMMIT,
        "teacher_source_commit": TEACHER_SOURCE_COMMIT,
        "inputs": {
            "campaign_summary": {"path": str(campaign_summary), "sha256": campaign_summary_sha256},
            "campaign_plan": {"path": str(plan), "sha256": plan_sha256},
            "prelaunch_manifest": {
                "path": str((campaign_root / "prelaunch-manifest.json").relative_to(snapshot_root)),
                "sha256": prelaunch_sha,
            },
            "bc_training_report": {"path": str(bc_training_report), "sha256": bc_training_report_sha256},
            "report_sha256": {key: value["sha256"] for key, value in sorted({**reports, **{f'baseline-{key}': value for key, value in baselines.items()}}.items())},
            "inspection_sha256": dict(sorted(inspection_hashes.items())),
        },
        "contract": {
            "training_seeds": list(SEEDS),
            "evaluation_seed": DEV_SEED,
            "attempts_per_report": ATTEMPTS,
            "paired_condition_ids_sha256": identity_sha,
            "bank_manifest_sha256": expected_bank_sha,
            "task_config_sha256": expected_task_sha,
            "source_sha256": source_map,
            "diagnostic_checkpoints": [20, 96],
            "selection_checkpoint": 200,
            "final_seed_opened": False,
        },
        "baselines": {
            "biased_zero": public_report_metrics(biased),
            "bc_only": public_report_metrics(bc_only),
        },
        "learned_reports": learned_rows,
        "aggregates": aggregates,
        "promotion_gate": recomputed_gate,
        "promotion_allowed": recomputed_gate["pass"],
        "compute_accounting": {
            "ppo": {
                "run_count": 6,
                "transitions_per_run": ppo_transitions_per_run,
                "total_transitions": ppo_transitions_per_run * 6,
                "formula_per_run": "256 environments × 32 steps × 200 updates",
            },
            "bc_pretraining_separate": {
                "sample_visits": bc_report["sample_visits"],
                "optimizer_updates": bc_report["total_bc_updates"],
                "elapsed_s": bc_report.get("elapsed_s"),
                "included_in_ppo_transition_count": False,
            },
        },
        "interpretation_limits": [
            "The three training seeds are reported as three raw points and one mean; 64 cases per seed are not pooled as 192 independent training seeds.",
            "Checkpoints 20 and 96 are diagnostic; only checkpoint 200 is used by the frozen promotion gate.",
            "Successful episode elapsed time and failed episode elapsed time are summarized separately; failures are not relabeled as 24-second successes.",
            "No confidence interval is inferred from three training seeds.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--campaign", "--campaign-summary", dest="campaign_summary", type=Path, required=True
    )
    parser.add_argument("--campaign-summary-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--bc-training-report", type=Path, required=True)
    parser.add_argument("--bc-training-report-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_campaign(
        repo=args.repo.resolve(),
        snapshot_root=args.snapshot_root.resolve(),
        campaign_summary=args.campaign_summary.resolve(),
        campaign_summary_sha256=args.campaign_summary_sha256,
        plan=args.plan.resolve(),
        plan_sha256=args.plan_sha256,
        bc_training_report=args.bc_training_report.resolve(),
        bc_training_report_sha256=args.bc_training_report_sha256,
    )
    write_new(args.output.resolve(), result)
    print(json.dumps({"output": str(args.output), "promotion_allowed": result["promotion_allowed"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
