#!/usr/bin/env python3
"""Run the sealed four-profile RJ45 20/96-checkpoint refinement campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from typing import Any, Callable, NoReturn, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rj45_long_campaign import StageResult, run_subprocess


PROFILE_LABELS = ("Z1", "F", "N", "R")
MINIMUM_FREE_BYTES = 2 * 1024**3
CANONICAL_PPO_SHA256 = "ccbdaddfa0be15406174c31a5ef9a2651e2e46e97f4ac71b4ec8fb86f027dffd"
CANONICAL_TRAINING_SHA256 = "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179"


class CampaignError(RuntimeError):
    pass


@dataclass(frozen=True)
class Profile:
    label: str
    ppo_config: Path
    training_config: Path
    zero_init_actor_output: bool


@dataclass(frozen=True)
class Checkpoint:
    filename: str
    iteration_index: int
    completed_updates: int


@dataclass(frozen=True)
class Contract:
    root: Path
    plan_path: Path
    plan_sha256: str
    inputs_path: Path
    inputs_sha256: str
    frozen_inputs: dict[Path, str]
    profiles: tuple[Profile, ...]
    execution_mode: str
    profile_devices: dict[str, str]
    cpu_threads_per_profile: int | None
    training_seed: int
    num_envs: int
    updates: int
    num_steps_per_env: int
    save_interval: int
    checkpoints: tuple[Checkpoint, ...]
    evaluation_seed: int
    evaluation_num_envs: int
    attempts: int
    nominal_evidence: Path
    held_state: Path
    parallel_contact_evidence: Path
    deadline_utc: datetime
    python: Path
    output_dir: Path


def _fail(message: str) -> NoReturn:
    raise CampaignError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise CampaignError(f"unable to hash {path}: {error}") from error
    return digest.hexdigest()


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _integer(value: Any, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer at least {minimum}")
    return value


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _load_json(path: Path, name: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        _fail(f"{name} contains non-finite JSON constant {value}")
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result
    def finite(value: Any, location: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            _fail(f"{location} contains a non-finite numeric value")
        if isinstance(value, dict):
            for key, item in value.items():
                finite(item, f"{location}.{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value):
                finite(item, f"{location}[{index}]")
    try:
        result = _object(json.loads(path.read_text(encoding="utf-8"), parse_constant=reject, object_pairs_hook=object_pairs), name)
        finite(result, name)
        return result
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid {name} {path}: {error}") from error


def _parse_utc(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        _fail(f"{name} must be an exact UTC timestamp")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise CampaignError(f"{name} must use YYYY-MM-DDTHH:MM:SSZ") from error
    return parsed


def _project_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        _fail(f"{name} must be a project-relative path without '..'")
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        _fail(f"{name} escapes project root")
    return path


def _referenced_path(root: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{name} must be a non-empty path")
    raw = Path(value)
    path = (raw if raw.is_absolute() else root / raw).resolve()
    if not path.is_relative_to(root):
        _fail(f"{name} escapes project root")
    return path


def _required_dependency_paths(root: Path, nominal: Path, held: Path, contact: Path) -> set[Path]:
    dependencies = {
        root / "configs/runtime.json",
        root / "configs/assets.json",
        root / "configs/rj45-assets.json",
        root / "configs/ppo-rj45-low-exploration-v1.json",
        root / "configs/rj45-training-target-offset-v1.json",
        root / "configs/tasks/rj45-v1.json",
        root / "scripts/run_rj45_long_campaign.py",
        *set((root / "src/rby1_assembly").glob("*.py")),
    }
    nominal_payload = _load_json(nominal, "nominal evidence")
    dependencies.add(_referenced_path(root, nominal_payload.get("trace_path"), "nominal trace_path"))
    held_payload = _load_json(held, "held state")
    held_provenance = _object(held_payload.get("provenance"), "held state provenance")
    for key in ("nominal_report_path", "nominal_trace_path"):
        dependencies.add(_referenced_path(root, held_provenance.get(key), f"held provenance {key}"))
    contact_payload = _load_json(contact, "parallel contact evidence")
    for key in ("report_path", "trace_path", "sat_report_path"):
        dependencies.add(_referenced_path(root, contact_payload.get(key), f"parallel contact {key}"))
    task = _load_json(root / "configs/tasks/rj45-v1.json", "RJ45 task config")
    assets = _object(task.get("assets"), "RJ45 task assets")
    for key in ("robot", "socket", "plug", "table", "build_report"):
        dependencies.add(_referenced_path(root, assets.get(key), f"RJ45 task assets.{key}"))
    build_report = _load_json(_referenced_path(root, assets.get("build_report"), "build report"), "build report")
    report_files = _object(build_report.get("files"), "build report files")
    asset_dir = _referenced_path(root, assets.get("build_report"), "build report").parent
    for filename in report_files:
        dependencies.add((asset_dir / filename).resolve())
    if not set((root / "src/rby1_assembly").glob("*.py")):
        _fail("src/rby1_assembly must contain Python sources")
    return dependencies


def _set_nested(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = _object(target.get(key), ".".join(path[:-1]))
    target[path[-1]] = value


def _require_profile_contrasts(root: Path, profiles: Sequence[Profile], *, seed: int, updates: int, steps: int, save_interval: int) -> None:
    ppo = [_load_json(profile.ppo_config, f"{profile.label} PPO config") for profile in profiles]
    training = [_load_json(profile.training_config, f"{profile.label} training config") for profile in profiles]
    canonical_ppo_path = root / "configs/ppo-rj45-low-exploration-v1.json"
    canonical_training_path = root / "configs/rj45-training-target-offset-v1.json"
    if _sha_file(canonical_ppo_path) != CANONICAL_PPO_SHA256:
        _fail("canonical low-exploration PPO SHA-256 mismatch")
    if _sha_file(canonical_training_path) != CANONICAL_TRAINING_SHA256:
        _fail("canonical target-offset-v1 training SHA-256 mismatch")
    expected_z1 = _load_json(canonical_ppo_path, "canonical PPO config")
    for key, value in (("seed", seed), ("max_iterations", updates), ("save_interval", save_interval), ("num_steps_per_env", steps)):
        expected_z1[key] = value
    if ppo[0] != expected_z1:
        _fail("Z1 PPO must equal the canonical low-exploration config with only planned rollout fields replaced")
    expected_f = copy.deepcopy(ppo[0]); _set_nested(expected_f, ("algorithm", "schedule"), "fixed")
    if ppo[1] != expected_f:
        _fail("F PPO must differ from Z1 only by algorithm.schedule adaptive->fixed")
    expected_n = copy.deepcopy(ppo[1]); _set_nested(expected_n, ("policy", "init_noise_std"), 0.02)
    if ppo[2] != expected_n:
        _fail("N PPO must differ from F only by policy.init_noise_std 0.1->0.02")
    if ppo[3] != ppo[2]:
        _fail("R PPO must exactly equal N PPO")
    canonical_training = _load_json(canonical_training_path, "canonical training config")
    if training[0] != canonical_training:
        _fail("Z1 training must exactly equal the canonical target-offset-v1 config")
    if training[1] != training[0] or training[2] != training[1]:
        _fail("Z1, F, and N training configs must be exactly equal")
    expected_r = copy.deepcopy(training[2]); _set_nested(expected_r, ("reward", "lateral_scale_m"), 0.00005)
    if _object(training[2].get("reward"), "N reward").get("lateral_scale_m") != 0.0005 or training[3] != expected_r:
        _fail("R training must differ from N only by reward.lateral_scale_m 0.0005->0.00005")


def _relative(contract: Contract, path: Path) -> str:
    return str(path.resolve().relative_to(contract.root))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _event(output_dir: Path, lock: threading.Lock | None = None, **payload: Any) -> None:
    record = {"utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), **payload}
    def write() -> None:
        with (output_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    if lock is None:
        write()
    else:
        with lock:
            write()


def load_contract(
    root: Path,
    plan_path: Path,
    plan_sha256: str,
    inputs_path: Path,
    inputs_sha256: str,
    python: Path,
    deadline_utc: str,
    output_dir: Path,
    *,
    now: datetime | None = None,
    disk_free: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free,
) -> Contract:
    root = root.resolve()
    plan_path, inputs_path, output_dir = plan_path.resolve(), inputs_path.resolve(), output_dir.resolve()
    if _sha_file(plan_path) != _sha(plan_sha256, "plan SHA-256"):
        _fail("plan SHA-256 mismatch")
    if _sha_file(inputs_path) != _sha(inputs_sha256, "inputs SHA-256"):
        _fail("inputs SHA-256 mismatch")
    plan = _load_json(plan_path, "plan")
    raw_inputs = _load_json(inputs_path, "inputs")
    frozen_inputs = {
        _project_path(root, relative, f"inputs[{relative!r}]"): _sha(digest, f"inputs[{relative!r}]")
        for relative, digest in raw_inputs.items()
    }
    if not frozen_inputs:
        _fail("inputs must not be empty")
    training = _object(plan.get("training"), "plan.training")
    evaluation = _object(plan.get("evaluation"), "plan.evaluation")
    seed = _integer(training.get("seed"), "training.seed")
    num_envs = _integer(training.get("num_envs"), "training.num_envs", 1)
    updates = _integer(training.get("updates"), "training.updates", 1)
    steps = _integer(training.get("num_steps_per_env"), "training.num_steps_per_env", 1)
    save_interval = _integer(training.get("save_interval"), "training.save_interval", 1)
    raw_checkpoints = training.get("checkpoints")
    if not isinstance(raw_checkpoints, list):
        _fail("training.checkpoints must be a list")
    checkpoints = tuple(
        Checkpoint(
            str(_object(item, f"training.checkpoints[{index}]").get("filename")),
            _integer(item.get("iteration_index"), f"training.checkpoints[{index}].iteration_index"),
            _integer(item.get("completed_updates"), f"training.checkpoints[{index}].completed_updates", 1),
        )
        for index, item in enumerate(raw_checkpoints)
    )
    if updates != 96 or save_interval != 19 or checkpoints != (
        Checkpoint("model_19.pt", 19, 20), Checkpoint("model_95.pt", 95, 96)
    ):
        _fail("refinement campaign requires 96 updates, save_interval 19, and checkpoints 19/95")
    if evaluation.get("final_seed_sealed") is not True:
        _fail("evaluation.final_seed_sealed must be true")
    eval_seed = _integer(evaluation.get("seed"), "evaluation.seed")
    forbidden_seeds = evaluation.get("forbidden_seeds")
    if forbidden_seeds != [20261027]:
        _fail("evaluation.forbidden_seeds must explicitly equal [20261027]")
    if seed in forbidden_seeds or eval_seed in forbidden_seeds or seed == eval_seed:
        _fail("training and development seeds must be distinct and exclude forbidden seeds")
    eval_envs = _integer(evaluation.get("num_envs"), "evaluation.num_envs", 1)
    attempts = _integer(evaluation.get("attempts"), "evaluation.attempts", 1)
    profiles_raw = plan.get("profiles")
    if not isinstance(profiles_raw, list) or len(profiles_raw) != 4:
        _fail("plan.profiles must contain four profiles")
    profiles: list[Profile] = []
    for index, raw in enumerate(profiles_raw):
        item = _object(raw, f"profiles[{index}]")
        label = item.get("label")
        if label != PROFILE_LABELS[index]:
            _fail(f"profiles must be ordered {PROFILE_LABELS}")
        zero = item.get("zero_init_actor_output")
        if type(zero) is not bool:
            _fail(f"profiles[{index}].zero_init_actor_output must be boolean")
        if zero is not True:
            _fail("all refinement profiles must zero-initialize actor output")
        ppo = _project_path(root, item.get("ppo_config"), f"profiles[{index}].ppo_config")
        training_config = _project_path(root, item.get("training_config"), f"profiles[{index}].training_config")
        ppo_payload = _load_json(ppo, f"{label} PPO config")
        if (ppo_payload.get("seed"), ppo_payload.get("max_iterations"), ppo_payload.get("num_steps_per_env"), ppo_payload.get("save_interval")) != (seed, updates, steps, save_interval):
            _fail(f"{label} PPO seed/update/rollout/save contract differs from plan")
        profiles.append(Profile(label, ppo, training_config, zero))
    _require_profile_contrasts(root, profiles, seed=seed, updates=updates, steps=steps, save_interval=save_interval)
    execution = _object(plan.get("execution"), "plan.execution")
    execution_mode = execution.get("mode")
    if execution_mode not in {"sequential", "parallel_profiles"}:
        _fail("execution.mode must be sequential or parallel_profiles")
    raw_devices = _object(execution.get("profile_devices"), "execution.profile_devices")
    if set(raw_devices) != set(PROFILE_LABELS) or any(not isinstance(value, str) or not value.startswith("cuda:") for value in raw_devices.values()):
        _fail("execution.profile_devices must map every profile to an explicit cuda device")
    expected_devices = {label: f"cuda:{index}" for index, label in enumerate(PROFILE_LABELS)}
    if execution_mode == "parallel_profiles" and raw_devices != expected_devices:
        _fail(f"parallel profile devices must equal {expected_devices}")
    if execution_mode == "sequential" and set(raw_devices.values()) != {"cuda:0"}:
        _fail("sequential execution must assign every profile to cuda:0")
    cpu_threads = execution.get("cpu_threads_per_profile")
    if cpu_threads is not None:
        cpu_threads = _integer(cpu_threads, "execution.cpu_threads_per_profile", 1)
    nominal = _project_path(root, plan.get("nominal_evidence"), "nominal_evidence")
    held = _project_path(root, plan.get("held_state"), "held_state")
    contact = _project_path(root, plan.get("parallel_contact_evidence"), "parallel_contact_evidence")
    required = [plan_path, nominal, held, contact, root / "scripts/train.py", root / "scripts/evaluate_policy.py", root / "scripts/inspect_rj45_checkpoint.py", root / "scripts/analyze_rj45_paired_v2.py", root / "scripts/run_rj45_refinement_campaign.py"]
    required += [path for profile in profiles for path in (profile.ppo_config, profile.training_config)]
    required += sorted(_required_dependency_paths(root, nominal, held, contact))
    for path in required:
        if frozen_inputs.get(path) != _sha_file(path):
            _fail(f"inputs do not pin current bytes for {path}")
    deadline = _parse_utc(deadline_utc, "deadline")
    created = _parse_utc(plan.get("created_utc"), "plan.created_utc")
    guard_started = _parse_utc(plan.get("guard_started_utc"), "plan.guard_started_utc")
    guard_deadline = _parse_utc(plan.get("guard_deadline_utc"), "plan.guard_deadline_utc")
    budget = _object(plan.get("budget"), "plan.budget")
    max_hours = budget.get("max_hours")
    if isinstance(max_hours, bool) or not isinstance(max_hours, (int, float)) or not 0 < float(max_hours) <= 6.0:
        _fail("budget.max_hours must be greater than zero and at most 6.0")
    current = now or datetime.now(timezone.utc)
    if created > current or guard_started > current or current >= deadline:
        _fail("time ordering must satisfy created <= now, guard_started <= now < deadline")
    if deadline - guard_started > timedelta(hours=float(max_hours)) or deadline > created + timedelta(hours=float(max_hours)) or deadline > guard_deadline:
        _fail("deadline exceeds campaign budget or external guard")
    if output_dir.exists():
        _fail("output directory already exists")
    if disk_free(output_dir.parent) < MINIMUM_FREE_BYTES:
        _fail("at least 2 GiB free disk space is required")
    executable = Path(os.path.abspath(python))
    if not executable.is_file():
        _fail(f"Python executable does not exist: {executable}")
    return Contract(root, plan_path, plan_sha256, inputs_path, inputs_sha256, frozen_inputs, tuple(profiles), execution_mode, dict(raw_devices), cpu_threads, seed, num_envs, updates, steps, save_interval, checkpoints, eval_seed, eval_envs, attempts, nominal, held, contact, deadline, executable, output_dir)


def _verify_inputs(contract: Contract) -> None:
    if _sha_file(contract.plan_path) != contract.plan_sha256:
        _fail("frozen plan changed")
    if _sha_file(contract.inputs_path) != contract.inputs_sha256:
        _fail("frozen inputs manifest changed")
    for path, digest in contract.frozen_inputs.items():
        if _sha_file(path) != digest:
            _fail(f"frozen input changed: {path}")


def _common_eval_args(contract: Contract, profile: Profile, report: Path) -> list[str]:
    return ["--task", "rj45", "--device", contract.profile_devices[profile.label], "--num-envs", str(contract.evaluation_num_envs), "--episodes", str(contract.attempts), "--seed", str(contract.evaluation_seed), "--config", _relative(contract, profile.ppo_config), "--report", str(report), "--rj45-nominal-evidence", _relative(contract, contract.nominal_evidence), "--rj45-training-config", _relative(contract, profile.training_config), "--rj45-held-state", _relative(contract, contract.held_state), "--evaluation-start", "held", "--parallel-contact-evidence", _relative(contract, contract.parallel_contact_evidence), "--headless"]


def _with_cpu_threads(contract: Contract, command: list[str]) -> list[str]:
    if contract.cpu_threads_per_profile is None:
        return command
    env = shutil.which("env")
    if env is None:
        _fail("execution.cpu_threads_per_profile requires the env executable")
    count = str(contract.cpu_threads_per_profile)
    return [env, f"OMP_NUM_THREADS={count}", f"MKL_NUM_THREADS={count}", *command]


def build_baseline_command(contract: Contract, report: Path) -> list[str]:
    return _with_cpu_threads(contract, [str(contract.python), "scripts/evaluate_policy.py", "--baseline-mode", "biased_zero", *_common_eval_args(contract, contract.profiles[0], report)])


def build_train_command(contract: Contract, profile: Profile) -> list[str]:
    command = [str(contract.python), "scripts/train.py", "--task", "rj45", "--device", contract.profile_devices[profile.label], "--num-envs", str(contract.num_envs), "--max-iterations", str(contract.updates), "--config", _relative(contract, profile.ppo_config), "--log-root", str(contract.output_dir / "runs" / profile.label), "--rj45-nominal-evidence", _relative(contract, contract.nominal_evidence), "--rj45-training-config", _relative(contract, profile.training_config), "--rj45-held-state", _relative(contract, contract.held_state), "--parallel-contact-evidence", _relative(contract, contract.parallel_contact_evidence), "--headless"]
    if profile.zero_init_actor_output:
        command.append("--zero-init-actor-output")
    return _with_cpu_threads(contract, command)


def _single_run_dir(contract: Contract, label: str) -> Path:
    base = contract.output_dir / "runs" / label / "rj45-v1"
    runs = sorted(path for path in base.iterdir() if path.is_dir()) if base.is_dir() else []
    if len(runs) != 1:
        _fail(f"{label} training must create exactly one run directory")
    return runs[0]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validate_report(analyzer: Any, report_path: Path, label: str, attempts: int) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _load_json(report_path, f"{label} report")
    indexed = analyzer._validate_report(report, label, attempts)
    invalid = sum(not outcome["valid"] for outcome in indexed.values())
    forbidden_values = [outcome.get("forbidden_contact_load_peak_n") for outcome in indexed.values()]
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in forbidden_values):
        _fail(f"{label} report does not provide finite forbidden-contact load for every attempt")
    maximum_forbidden = max(float(value) for value in forbidden_values)
    if invalid or maximum_forbidden != 0.0:
        _fail(f"{label} report failed invalid/safety gate: invalid={invalid}, forbidden={maximum_forbidden}")
    return report, indexed


def _validate_inspection(path: Path, checkpoint: Path, expected_updates: int, label: str) -> dict[str, Any]:
    inspected = _load_json(path, f"{label} inspection")
    details = _object(inspected.get("checkpoint"), f"{label} inspection checkpoint")
    if (
        inspected.get("status") != "checkpoint_payload_and_lineage_verified"
        or details.get("completed_updates") != expected_updates
        or details.get("sha256") != _sha_file(checkpoint)
    ):
        _fail(f"{label} checkpoint inspection failed")
    return inspected


def _run_stage(contract: Contract, stage: str, command: Sequence[str], runner: Callable[..., StageResult], disk_free: Callable[[Path], int], stop_event: threading.Event | None = None, event_lock: threading.Lock | None = None) -> None:
    if datetime.now(timezone.utc) >= contract.deadline_utc:
        _fail(f"deadline reached before {stage}")
    if disk_free(contract.output_dir) < MINIMUM_FREE_BYTES:
        _fail("free disk space fell below 2 GiB")
    _verify_inputs(contract)
    log = contract.output_dir / "logs" / f"{stage}.log"
    if stop_event is not None and stop_event.is_set():
        _fail(f"campaign stopped before {stage}")
    _event(contract.output_dir, lock=event_lock, stage=stage, status="started", command=list(command))
    result: StageResult | None = None
    try:
        def monitor() -> None:
            if stop_event is not None and stop_event.is_set():
                _fail("parallel profile failure requested process-group stop")
            _verify_inputs(contract)
            if disk_free(contract.output_dir) < MINIMUM_FREE_BYTES:
                _fail("free disk space fell below 2 GiB")
        result = runner(command, contract.root, log, contract.deadline_utc, monitor, 2.0, 900.0)
        text = log.read_text(encoding="utf-8", errors="replace")
        from run_rj45_long_campaign import _check_process_text
        _check_process_text(text, log)
        if result.return_code != 0 or not result.process_group_cleanup_confirmed:
            _fail(f"stage {stage} failed: return={result.return_code}, reason={result.failure_reason}")
        _verify_inputs(contract)
    except BaseException as error:
        _event(contract.output_dir, lock=event_lock, stage=stage, status="failed", return_code=result.return_code if result else None, error={"type": type(error).__name__, "message": str(error)})
        raise
    _event(contract.output_dir, lock=event_lock, stage=stage, status="completed", return_code=0)


def execute_campaign(contract: Contract, *, runner: Callable[..., StageResult] = run_subprocess, disk_free: Callable[[Path], int] = lambda path: shutil.disk_usage(path).free) -> dict[str, Any]:
    contract.output_dir.mkdir(parents=False)
    for name in ("logs", "reports", "inspections", "runs"):
        (contract.output_dir / name).mkdir()
    status_path = contract.output_dir / "campaign-status.json"
    _atomic_json(status_path, {"schema_version": 1, "status": "running", "execution_mode": contract.execution_mode, "profile_devices": contract.profile_devices, "completed_profiles": [], "final_seed_opened": False})
    analyzer = _load_module(contract.root / "scripts/analyze_rj45_paired_v2.py", "_rj45_refinement_analyzer")
    stop_event = threading.Event()
    write_lock = threading.Lock()
    try:
        baseline_path = contract.output_dir / "reports" / "baseline.json"
        _run_stage(contract, "evaluate-baseline", build_baseline_command(contract, baseline_path), runner, disk_free, stop_event, write_lock)
        baseline, baseline_index = _validate_report(analyzer, baseline_path, "baseline", contract.attempts)
        results: dict[str, Any] = {}
        def run_profile(profile: Profile) -> tuple[str, dict[str, Any]]:
            try:
                _run_stage(contract, f"train-{profile.label}", build_train_command(contract, profile), runner, disk_free, stop_event, write_lock)
                run_dir = _single_run_dir(contract, profile.label)
                profile_results: dict[str, Any] = {}
                for checkpoint_spec in contract.checkpoints:
                    checkpoint = run_dir / checkpoint_spec.filename
                    suffix = f"iter-{checkpoint_spec.iteration_index}"
                    inspection = contract.output_dir / "inspections" / f"{profile.label}-{suffix}.json"
                    inspect_command = _with_cpu_threads(contract, [str(contract.python), "scripts/inspect_rj45_checkpoint.py", "--checkpoint", str(checkpoint), "--manifest", str(run_dir / "checkpoint_manifest.json"), "--training-provenance", str(run_dir / "training_provenance.json"), "--runner-config", str(run_dir / "runner_config.json"), "--training-script", str(contract.root / "scripts/train.py"), "--output", str(inspection)])
                    _run_stage(contract, f"inspect-{profile.label}-{suffix}", inspect_command, runner, disk_free, stop_event, write_lock)
                    _validate_inspection(inspection, checkpoint, checkpoint_spec.completed_updates, f"{profile.label} {suffix}")
                    report_path = contract.output_dir / "reports" / f"{profile.label}-{suffix}.json"
                    eval_command = _with_cpu_threads(contract, [str(contract.python), "scripts/evaluate_policy.py", str(checkpoint), "--baseline-mode", "learned", *_common_eval_args(contract, profile, report_path)])
                    _run_stage(contract, f"evaluate-{profile.label}-{suffix}", eval_command, runner, disk_free, stop_event, write_lock)
                    label = f"{profile.label}-{suffix}"
                    report, indexed = _validate_report(analyzer, report_path, label, contract.attempts)
                    if set(indexed) != set(baseline_index):
                        _fail(f"{label} development conditions differ from baseline")
                    paired = analyzer.analyze_reports(baseline, {label: report}, expected_attempts=contract.attempts, bootstrap_resamples=1000, bootstrap_seed=contract.evaluation_seed, baseline_label="baseline")
                    bundle = analyzer._verify_training_bundle(label, report, checkpoint_path=checkpoint, manifest_path=run_dir / "checkpoint_manifest.json", training_provenance_path=run_dir / "training_provenance.json", runner_config_path=run_dir / "runner_config.json", checkpoint_inspection_path=inspection)
                    if bundle.get("status") != "checkpoint_manifest_provenance_runner_bytes_verified" or bundle.get("checkpoint_completed_updates") != checkpoint_spec.completed_updates:
                        _fail(f"{label} training bundle verification failed")
                    profile_results[str(checkpoint_spec.completed_updates)] = {"successes": sum(outcome["success"] for outcome in indexed.values()), "attempts": contract.attempts, "report_sha256": _sha_file(report_path), "checkpoint_sha256": _sha_file(checkpoint), "paired_result": paired["results"][label], "training_bundle": bundle}
                return profile.label, profile_results
            except BaseException:
                stop_event.set()
                raise

        if contract.execution_mode == "parallel_profiles":
            errors: list[BaseException] = []
            with ThreadPoolExecutor(max_workers=4, thread_name_prefix="rj45-profile") as pool:
                futures = [pool.submit(run_profile, profile) for profile in contract.profiles]
                for future in as_completed(futures):
                    try:
                        label, profile_result = future.result()
                        with write_lock:
                            results[label] = profile_result
                            _atomic_json(status_path, {"schema_version": 1, "status": "running", "execution_mode": contract.execution_mode, "profile_devices": contract.profile_devices, "completed_profiles": sorted(results), "final_seed_opened": False})
                    except BaseException as error:
                        stop_event.set()
                        errors.append(error)
            if errors:
                raise errors[0]
        else:
            for profile in contract.profiles:
                label, profile_result = run_profile(profile)
                results[label] = profile_result
                _atomic_json(status_path, {"schema_version": 1, "status": "running", "execution_mode": contract.execution_mode, "profile_devices": contract.profile_devices, "completed_profiles": list(results), "final_seed_opened": False})
        summary = {"schema_version": 1, "status": "completed", "execution": {"mode": contract.execution_mode, "profile_devices": contract.profile_devices, "cpu_threads_per_profile": contract.cpu_threads_per_profile}, "baseline": {"device": contract.profile_devices["Z1"], "successes": sum(outcome["success"] for outcome in baseline_index.values()), "attempts": contract.attempts, "report_sha256": _sha_file(baseline_path)}, "profiles": results, "training_transitions_per_profile": contract.num_envs * contract.num_steps_per_env * contract.updates, "training_transitions_total": len(contract.profiles) * contract.num_envs * contract.num_steps_per_env * contract.updates, "selection_performed": False, "final_seed_opened": False}
        _atomic_json(contract.output_dir / "summary.json", summary)
        _atomic_json(status_path, {"schema_version": 1, "status": "completed", "execution_mode": contract.execution_mode, "profile_devices": contract.profile_devices, "completed_profiles": list(results), "final_seed_opened": False})
        return summary
    except BaseException as error:
        _atomic_json(status_path, {"schema_version": 1, "status": "failed", "execution_mode": contract.execution_mode, "profile_devices": contract.profile_devices, "error": {"type": type(error).__name__, "message": str(error)}, "final_seed_opened": False})
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    contract = load_contract(args.project_root, args.plan, args.plan_sha256, args.inputs, args.inputs_sha256, args.python, args.deadline_utc, args.output_dir)
    execute_campaign(contract)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CampaignError as error:
        print(f"campaign error: {error}", file=sys.stderr)
        raise SystemExit(2)
