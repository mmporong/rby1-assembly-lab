#!/usr/bin/env python3
"""Evaluate one frozen Standard79 checkpoint against three action ablations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import struct
import sys
import time
import traceback
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "rj45_frozen_ablation_evaluation_v1"
CONTRACT_SCHEMA = "rj45_frozen_ablation_contract_v1"
MODE = "standard79"
ABLATION_MODES = ("standard79", "zero", "mean_constant", "open_loop_playback")
EXPECTED_UPDATES = 32
EXPECTED_CHECKPOINT_SHA256 = "935496bd8f6e8c0e53c151cbb497f7a0bcfb9768dd18e65de3874c6841583c02"
EXPECTED_PROTOCOL_SHA256 = "a1a00452c8c7a6151844e2b3bd56d9322a2baaa6f1c4b7e9c2bb925be07959ef"
EXPECTED_RUNNER_SHA256 = "132918ca01fa831bb6f703b4cc2aa1d2d9bd73085459a8964495b37e3ef177f1"
EXPECTED_UPDATES_TRACE_SHA256 = "11743fdef8ecdf98e7793cf1e3046c64d1333ead56f1f24f7f057570b387c807"
EXPECTED_SOURCE_JSONL_SHA256 = "1809f9092b53133ffece991da0c0daf0d07a083878d0bf1927b431d294fbe9e4"
EXPECTED_CASE_SEEDS = (20261128, 20270104, 20270112, 20270120, 20270128)
CALIBRATION_CASE_SEED_OFFSET = 46123020


class FrozenAblationError(RuntimeError):
    """The sealed ablation contract or evaluation was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise FrozenAblationError(f"{field} must be 64 lowercase hex characters")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FrozenAblationError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise FrozenAblationError(f"JSON root must be an object: {path}")
    return value


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise FrozenAblationError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _deadline(value: Any) -> tuple[str, float]:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise FrozenAblationError("deadline_utc must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise FrozenAblationError("deadline_utc is invalid") from error
    if parsed.tzinfo != timezone.utc:
        raise FrozenAblationError("deadline_utc must be UTC")
    return parsed.isoformat().replace("+00:00", "Z"), parsed.timestamp()


def validate_contract(path: Path, expected_sha256: str, sha: Any = file_sha256) -> dict[str, Any]:
    if sha(path) != _sha256_hex(expected_sha256, "contract SHA-256"):
        raise FrozenAblationError("contract file SHA-256 mismatch")
    contract = _load_json(path)
    required = {
        "schema_name", "helper_sha256", "source_sha256", "origin", "cases",
        "case_order", "mode_order", "budget_and_stop", "claims", "public_input_analysis",
    }
    if set(contract) != required or contract["schema_name"] != CONTRACT_SCHEMA:
        raise FrozenAblationError("contract schema or keys changed")
    if sha(Path(__file__).resolve()) != _sha256_hex(contract["helper_sha256"], "helper_sha256"):
        raise FrozenAblationError("helper bytes do not match contract")
    sources = contract["source_sha256"]
    if not isinstance(sources, dict) or set(sources) != {"runner", "replay_helper", "g2_helper", "runtime_manifest"}:
        raise FrozenAblationError("source_sha256 keys changed")
    for key, value in sources.items():
        _sha256_hex(value, f"source_sha256.{key}")
    if sources["runner"] != EXPECTED_RUNNER_SHA256:
        raise FrozenAblationError("runner SHA is not the preregistered source")
    origin = contract["origin"]
    origin_keys = {
        "mode", "protocol_sha256", "report_sha256", "checkpoint_sha256",
        "updates_trace_sha256", "source_trace_gzip_sha256",
        "source_trace_jsonl_sha256", "updates",
    }
    if not isinstance(origin, dict) or set(origin) != origin_keys:
        raise FrozenAblationError("origin keys changed")
    for key in origin_keys - {"mode", "updates"}:
        _sha256_hex(origin[key], f"origin.{key}")
    if origin["mode"] != MODE or type(origin["updates"]) is not int or origin["updates"] != EXPECTED_UPDATES:
        raise FrozenAblationError("origin mode/updates changed")
    expected = {
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "updates_trace_sha256": EXPECTED_UPDATES_TRACE_SHA256,
        "source_trace_jsonl_sha256": EXPECTED_SOURCE_JSONL_SHA256,
    }
    for key, value in expected.items():
        if origin[key] != value:
            raise FrozenAblationError(f"origin {key} changed")
    cases = contract["cases"]
    if not isinstance(cases, list) or len(cases) != 5:
        raise FrozenAblationError("contract must contain exactly five cases")
    case_ids: list[str] = []
    seeds: list[int] = []
    for case in cases:
        if not isinstance(case, dict) or set(case) != {"case_id", "base_seed", "calibration_case_seed", "split", "grasp_id", "episode_index"}:
            raise FrozenAblationError("case keys changed")
        case_id, seed = case["case_id"], case["base_seed"]
        if not isinstance(case_id, str) or not case_id or "/" in case_id or ".." in case_id:
            raise FrozenAblationError("case_id is unsafe")
        if type(seed) is not int or seed < 0 or seed % 8:
            raise FrozenAblationError("every base_seed must be a non-negative multiple of 8")
        if case["calibration_case_seed"] != seed + CALIBRATION_CASE_SEED_OFFSET:
            raise FrozenAblationError("calibration_case_seed formula changed")
        if case["split"] != "train" or case["grasp_id"] != "rj45-train-02" or case["episode_index"] != 1:
            raise FrozenAblationError("case fixed-condition lineage changed")
        case_ids.append(case_id)
        seeds.append(seed)
    if len(set(case_ids)) != 5 or len(set(seeds)) != 5:
        raise FrozenAblationError("case ids and seeds must be unique")
    if tuple(seeds) != EXPECTED_CASE_SEEDS:
        raise FrozenAblationError("case base seeds changed")
    if contract["case_order"] != case_ids:
        raise FrozenAblationError("case_order must exactly match the sealed cases")
    if contract["mode_order"] != list(ABLATION_MODES):
        raise FrozenAblationError("mode_order changed")
    analysis = contract["public_input_analysis"]
    if not isinstance(analysis, dict) or set(analysis) != {
        "source_actions_canonical_sha256", "source_action_count",
        "source_action_dimension", "source_action_mean", "source_action_last",
    }:
        raise FrozenAblationError("public_input_analysis keys changed")
    _sha256_hex(analysis["source_actions_canonical_sha256"], "source_actions_canonical_sha256")
    if type(analysis["source_action_count"]) is not int or analysis["source_action_count"] <= 0 or analysis["source_action_dimension"] != 6:
        raise FrozenAblationError("public input action shape changed")
    for field in ("source_action_mean", "source_action_last"):
        values = analysis[field]
        if not isinstance(values, list) or len(values) != 6 or not all(
            isinstance(value, (int, float)) and math.isfinite(value) for value in values
        ):
            raise FrozenAblationError(f"public input {field} is invalid")
    budget = contract["budget_and_stop"]
    if not isinstance(budget, dict) or set(budget) != {
        "budget_cap_usd", "reservation_usd", "deadline_utc", "helper_wall_time_s",
        "per_episode_wall_time_s", "external_stop_wall_time_s",
    }:
        raise FrozenAblationError("budget_and_stop keys changed")
    if (budget["budget_cap_usd"], budget["reservation_usd"], budget["helper_wall_time_s"],
            budget["per_episode_wall_time_s"], budget["external_stop_wall_time_s"]) != (700, 6, 3900, 300, 4500):
        raise FrozenAblationError("budget or wall-time guard changed")
    budget["deadline_utc"], _ = _deadline(budget["deadline_utc"])
    claims = contract["claims"]
    expected_claims = {
        "final_split_sealed": True, "training_executed": False,
        "dev_split_sealed": True,
        "checkpoint_selection_changed": False, "single_environment_serial": True,
        "episode_count": 20,
    }
    if claims != expected_claims:
        raise FrozenAblationError("sealed claims changed")
    if any(type(claims[key]) is not type(value) for key, value in expected_claims.items()):
        raise FrozenAblationError("sealed claim types changed")
    return contract


def validate_preimport_sources(
    *, current_root: Path, origin_protocol: Path, origin_report: Path,
    checkpoint: Path, source_trace: Path, contract: Mapping[str, Any], sha: Any = file_sha256,
) -> dict[str, Any]:
    origin = contract["origin"]
    paths = {
        "protocol": origin_protocol,
        "report": origin_report,
        "checkpoint": checkpoint,
        "source_trace_gzip": source_trace,
        "runner": current_root / "scripts/run_rj45_oracle_ppo_probe.py",
        "replay_helper": current_root / "scripts/record_rj45_oracle_probe.py",
        "g2_helper": current_root / "scripts/run_rj45_g2_physics_probe.py",
        "runtime_manifest": current_root / "configs/rj45-oracle-runtime-104721a.json",
    }
    expected = {
        "protocol": origin["protocol_sha256"], "report": origin["report_sha256"],
        "checkpoint": origin["checkpoint_sha256"], "source_trace_gzip": origin["source_trace_gzip_sha256"],
        **contract["source_sha256"],
    }
    observed: dict[str, Any] = {}
    for name, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file() or sha(resolved) != expected[name]:
            raise FrozenAblationError(f"pre-import source binding failed: {name}")
        observed[name] = {"path": str(resolved), "sha256": expected[name]}
    report = _load_json(origin_report)
    if report.get("status") != "completed" or report.get("mode") != MODE:
        raise FrozenAblationError("origin report is not completed standard79")
    learning = report.get("learning")
    if not isinstance(learning, dict) or learning.get("updates") != EXPECTED_UPDATES:
        raise FrozenAblationError("origin report does not bind 32 updates")
    if learning.get("updates_trace", {}).get("sha256") != origin["updates_trace_sha256"]:
        raise FrozenAblationError("origin updates trace SHA changed")
    checkpoint_record = learning.get("checkpoint")
    if not isinstance(checkpoint_record, dict) or checkpoint_record.get("sha256") != origin["checkpoint_sha256"]:
        raise FrozenAblationError("origin final checkpoint binding changed")
    return observed


def load_source_actions(path: Path, contract: Mapping[str, Any], sha: Any = file_sha256) -> list[list[float]]:
    if path.suffix != ".gz" or sha(path) != contract["origin"]["source_trace_gzip_sha256"]:
        raise FrozenAblationError("source trace gzip binding failed")
    try:
        raw = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as error:
        raise FrozenAblationError("source trace is not valid gzip") from error
    if hashlib.sha256(raw).hexdigest() != contract["origin"]["source_trace_jsonl_sha256"]:
        raise FrozenAblationError("uncompressed source trace SHA-256 mismatch")
    actions: list[list[float]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise FrozenAblationError(f"invalid source trace JSON row {number}") from error
        if not isinstance(row, dict):
            raise FrozenAblationError("source trace row is not an object")
        if "event" in row:
            continue
        if not isinstance(row.get("phase"), str):
            raise FrozenAblationError("source trace phase must be a string")
        if row.get("active") is True and row.get("policy_input_valid") is True:
            expanded = row.get("applied_expanded_action")
            if not isinstance(expanded, list) or len(expanded) != 12:
                raise FrozenAblationError("selected expanded action must have 12 values")
            values = [float(value) for value in expanded]
            if not all(math.isfinite(value) and -1.0 <= value <= 1.0 for value in values):
                raise FrozenAblationError("selected action is non-finite or out of bounds")
            if any(value != 0.0 for value in values[:6]):
                raise FrozenAblationError("selected source action has a non-zero left arm")
            if any(math.sqrt(sum(values[start + i] ** 2 for i in range(3))) > 1.0 + 1e-12 for start in (6, 9)):
                raise FrozenAblationError("selected source action violates the 3D norm bound")
            actions.append(values[6:])
    analysis = contract["public_input_analysis"]
    if len(actions) != analysis["source_action_count"]:
        raise FrozenAblationError("source trace selected-row count differs from provenance")
    means = [math.fsum(action[index] for action in actions) / len(actions) for index in range(6)]
    if any(not math.isclose(observed, expected, rel_tol=0.0, abs_tol=1e-15) for observed, expected in zip(means, analysis["source_action_mean"])):
        raise FrozenAblationError("source action mean differs from provenance")
    if actions[-1] != analysis["source_action_last"]:
        raise FrozenAblationError("source action last row differs from the sealed analysis")
    if canonical_source_actions_sha256(actions) != analysis["source_actions_canonical_sha256"]:
        raise FrozenAblationError("source actions canonical SHA-256 mismatch")
    return actions


def canonical_source_actions_payload(actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    rows = [[float(value) for value in row] for row in actions]
    return {
        "actions": rows,
        "last": rows[-1],
        "mean": [math.fsum(row[index] for row in rows) / len(rows) for index in range(6)],
    }


def canonical_source_actions_bytes(actions: Sequence[Sequence[float]]) -> bytes:
    return (json.dumps(canonical_source_actions_payload(actions), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def canonical_source_actions_sha256(actions: Sequence[Sequence[float]]) -> str:
    return hashlib.sha256(canonical_source_actions_bytes(actions)).hexdigest()


def sample_bias_cases(env: Any, cases: Sequence[Mapping[str, Any]], torch_module: Any) -> list[dict[str, Any]]:
    calibration = env._training_config["calibration_bias"]
    level = env._training_bias_level
    translation_max = float(calibration["translation_levels_m"][level])
    rotation_max = float(calibration["rotation_max_levels_rad"][level])
    result = []
    for case in cases:
        generator = torch_module.Generator(device=env.device)
        generator.manual_seed(case["calibration_case_seed"])
        translation = torch_module.randn((1, 3), device=env.device, generator=generator)
        translation /= torch_module.linalg.vector_norm(translation, dim=1, keepdim=True).clamp_min(1.0e-9)
        translation *= torch_module.rand((1, 1), device=env.device, generator=generator) * translation_max
        rotation = torch_module.randn((1, 3), device=env.device, generator=generator)
        rotation /= torch_module.linalg.vector_norm(rotation, dim=1, keepdim=True).clamp_min(1.0e-9)
        rotation *= torch_module.rand((1, 1), device=env.device, generator=generator) * rotation_max
        result.append({
            "case_id": case["case_id"], "base_seed": case["base_seed"],
            "calibration_case_seed": case["calibration_case_seed"],
            "translation_entry_m": translation[0].detach().cpu().tolist(),
            "rotation_vector_entry_rad": rotation[0].detach().cpu().tolist(),
        })
    return result


def first_active_base79(trace_path: Path) -> list[float]:
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("active") is True and row.get("policy_input_valid") is True:
            base = row.get("observation", {}).get("base79")
            if not isinstance(base, list) or len(base) != 79 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in base):
                raise FrozenAblationError("first active base79 observation is invalid")
            return [float(value) for value in base]
    raise FrozenAblationError("trace has no active valid base79 observation")


def raw_base79_sha256(values: Sequence[float]) -> str:
    if len(values) != 79 or not all(math.isfinite(float(value)) for value in values):
        raise FrozenAblationError("base79 hash input is invalid")
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(json.dumps([1, 79], separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack("<79f", *(float(value) for value in values)))
    return digest.hexdigest()


def _eligible_transition(env: Any) -> Any:
    """Admit the current valid input before its post-step counter increment.

    The frozen runtime increments the valid-input count in ``_get_dones``.
    Gating on its previous value would discard the first valid action.
    """

    return (
        env._policy_active_mask()
        & ~env._warmup_pending_mask()
        & env._training_last_policy_observation_valid
    )


class AblationPolicy:
    """A state-free constant/playback policy gated only by the supervisor clock."""

    def __init__(self, env: Any, mode: str, actions: Sequence[Sequence[float]], torch_module: Any):
        if mode not in ABLATION_MODES[1:]:
            raise ValueError("unsupported ablation policy")
        self.env, self.mode, self.actions, self.torch = env, mode, actions, torch_module
        self.index = 0
        self.eligible_call_count = 0
        self.source_rows_consumed = 0
        self.tail_hold_count = 0
        self.mean = [math.fsum(row[i] for row in actions) / len(actions) for i in range(6)]

    def __call__(self, _observations: Any) -> Any:
        zeros = self.torch.zeros((1, 6), dtype=self.torch.float32, device=self.env.device)
        if not bool(_eligible_transition(self.env)[0].item()):
            return zeros
        self.eligible_call_count += 1
        if self.mode == "zero":
            return zeros
        if self.mode == "mean_constant":
            value = self.mean
        else:
            if self.index < len(self.actions):
                value = self.actions[self.index]
                self.source_rows_consumed += 1
            else:
                value = self.actions[-1]
                self.tail_hold_count += 1
            self.index += 1
        return self.torch.tensor([value], dtype=self.torch.float32, device=self.env.device)

    def telemetry(self) -> dict[str, int]:
        return {
            "eligible_call_count": self.eligible_call_count,
            "source_rows_consumed": self.source_rows_consumed,
            "tail_hold_count": self.tail_hold_count,
        }


def _atomic_write(path: Path, payload: Mapping[str, Any], *, exclusive: bool = True) -> None:
    if exclusive and path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _fsync_existing(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def persist_report_before_app_close(report_path: Path, report: Mapping[str, Any], probe: Any, app: Any, exit_code: int) -> int:
    _atomic_write(report_path, report)
    print(json.dumps({"status": report.get("status"), "report": str(report_path)}, sort_keys=True), flush=True)
    return probe._close_simulation_app(app, exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--origin-protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--origin-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--held-bank", type=Path, default=Path("assets/generated/rj45-held-grasps-v1.json"))
    parser.add_argument("--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json"))
    parser.add_argument("--training-config", type=Path, default=Path("configs/rj45-training-target-offset-v1.json"))
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--runtime-manifest", type=Path, default=Path("configs/rj45-oracle-runtime-104721a.json"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.current_root, args.runtime_root = args.current_root.resolve(), args.runtime_root.resolve()
    args.origin_protocol, args.origin_report = args.origin_protocol.resolve(), args.origin_report.resolve()
    args.checkpoint, args.source_trace, args.contract = args.checkpoint.resolve(), args.source_trace.resolve(), args.contract.resolve()
    args.output_dir = args.output_dir.resolve()
    if not args.runtime_manifest.is_absolute():
        args.runtime_manifest = (args.current_root / args.runtime_manifest).resolve()
    report_path = args.output_dir.with_name(args.output_dir.name + "-report.json")
    if args.output_dir.exists() or report_path.exists():
        raise SystemExit("output directory/report already exists")
    contract = validate_contract(args.contract, args.contract_sha256)
    if args.protocol_sha256 != contract["origin"]["protocol_sha256"]:
        raise FrozenAblationError("CLI protocol SHA differs from contract")
    preimport = validate_preimport_sources(
        current_root=args.current_root, origin_protocol=args.origin_protocol,
        origin_report=args.origin_report, checkpoint=args.checkpoint,
        source_trace=args.source_trace, contract=contract,
    )
    source_actions = load_source_actions(args.source_trace, contract)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_actions_path = args.output_dir / "source-actions.json"
    source_actions_record = {
        **canonical_source_actions_payload(source_actions),
        "canonical_sha256": canonical_source_actions_sha256(source_actions),
        "source_trace": preimport["source_trace_gzip"],
    }
    _atomic_write(source_actions_path, source_actions_record)
    if _load_json(source_actions_path)["canonical_sha256"] != contract["public_input_analysis"]["source_actions_canonical_sha256"]:
        raise FrozenAblationError("persisted source action analysis changed")
    probe = _load_module(Path(preimport["runner"]["path"]), "rj45_frozen_ablation_probe")
    replay = _load_module(Path(preimport["replay_helper"]["path"]), "rj45_frozen_ablation_replay")
    protocol_record = probe.validate_protocol(args.origin_protocol, args.protocol_sha256, args.current_root)
    protocol = protocol_record["payload"]
    args.protocol, args.protocol_sha256 = args.origin_protocol, contract["origin"]["protocol_sha256"]
    args.runtime_manifest_record = probe.validate_runtime_manifest(args.runtime_root, args.runtime_manifest)
    _, deadline_epoch = _deadline(contract["budget_and_stop"]["deadline_utc"])
    if time.time() >= deadline_epoch:
        raise FrozenAblationError("ablation deadline already expired")
    started = time.monotonic()
    app = env = None
    failure_stage, exit_code = "application_launch", 1
    report: dict[str, Any]
    try:
        from isaaclab.app import AppLauncher
        app = AppLauncher({"headless": args.headless, "device": args.device}).app
        sys.path.insert(0, str(args.runtime_root / "src"))
        import torch
        g2 = _load_module(Path(preimport["g2_helper"]["path"]), "rj45_frozen_ablation_g2")
        env, context, frozen_hashes = probe._build_env(args, MODE, protocol)
        runtime_import_origins = probe.verify_runtime_import_origins(args.runtime_root)
        if int(env.num_envs) != 1:
            raise FrozenAblationError("ablation requires exactly one environment")
        ppo_policy = replay._policy_for_replay(env, MODE, args.checkpoint, protocol, probe)
        bias_cases = sample_bias_cases(env, contract["cases"], torch)
        runtime_source_sha256 = args.runtime_manifest_record["source_file_sha256"]
        if len(runtime_source_sha256) != 35:
            raise FrozenAblationError("runtime manifest no longer binds exactly 35 source files")
        runtime_source_canonical_sha256 = hashlib.sha256(
            (json.dumps(runtime_source_sha256, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        ).hexdigest()
        bias_cases_path = args.output_dir / "bias-cases.json"
        _atomic_write(bias_cases_path, {
            "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "sampling": "frozen CUDA generator: randn3/norm/clamp1e-9, rand1*max; translation then rotation",
            "runtime_source_file_count": len(runtime_source_sha256),
            "runtime_source_canonical_sha256": runtime_source_canonical_sha256,
            "runtime_manifest_sha256": contract["source_sha256"]["runtime_manifest"],
            "cases": bias_cases,
        })
        bias_by_case = {item["case_id"]: item for item in bias_cases}
        episodes: list[dict[str, Any]] = []
        initial_base79_by_case: dict[str, list[float]] = {}
        case_by_id = {case["case_id"]: case for case in contract["cases"]}
        execution_order: list[dict[str, Any]] = []
        for case_index, case_id in enumerate(contract["case_order"]):
            rotated_modes = contract["mode_order"][case_index:] + contract["mode_order"][:case_index]
            for ablation_mode in rotated_modes:
                failure_stage = f"evaluation:{ablation_mode}:{case_id}"
                case = case_by_id[case_id]
                case_started = time.monotonic()
                env._training_bias_base_seed = case["base_seed"]
                env.cfg.seed = case["base_seed"]
                if ablation_mode == MODE:
                    policy = ppo_policy
                else:
                    policy = AblationPolicy(env, ablation_mode, source_actions, torch)
                index = len(episodes) + 1
                trace_path = args.output_dir / "traces" / f"{index:02d}-{case_id}-{ablation_mode}.jsonl"
                def guard() -> None:
                    now = time.monotonic()
                    if now - started > 3900 or time.time() >= deadline_epoch:
                        raise TimeoutError("global ablation guard exceeded")
                    if now - case_started > 300:
                        raise TimeoutError("per-episode wall-time limit exceeded")
                evaluation = probe._run_evaluation(
                    env, MODE, policy, g2, context["task"], context["training"],
                    trace_path, guard, physics_seed=case["base_seed"],
                )
                _fsync_existing(trace_path)
                training_report = env.rj45_training_report()
                observed_bias = training_report["calibration_bias"]
                expected_bias = bias_by_case[case_id]
                if observed_bias["translation_entry_m"] != [expected_bias["translation_entry_m"]] or observed_bias["rotation_vector_entry_rad"] != [expected_bias["rotation_vector_entry_rad"]]:
                    raise FrozenAblationError("reset bias differs from the preflight CUDA sample")
                base79 = first_active_base79(trace_path)
                base79_hash = raw_base79_sha256(base79)
                if base79_hash != evaluation["observation"]["base79_sha256"]:
                    raise FrozenAblationError("raw base79 differs from the evaluation observation hash")
                if case_id in initial_base79_by_case and base79 != initial_base79_by_case[case_id]:
                    raise FrozenAblationError("paired modes started from different base79 observations")
                initial_base79_by_case.setdefault(case_id, base79)
                policy_telemetry = None
                if isinstance(policy, AblationPolicy):
                    policy_telemetry = policy.telemetry()
                    if policy_telemetry["eligible_call_count"] != evaluation["active_rows"]:
                        raise FrozenAblationError(
                            "policy eligibility count differs from evaluation active rows: "
                            f"calls={policy_telemetry['eligible_call_count']}, "
                            f"active_rows={evaluation['active_rows']}"
                        )
                    if ablation_mode == "open_loop_playback" and (
                        policy_telemetry["source_rows_consumed"] != min(evaluation["active_rows"], len(source_actions))
                        or policy_telemetry["tail_hold_count"] != max(evaluation["active_rows"] - len(source_actions), 0)
                    ):
                        raise FrozenAblationError("playback source/tail accounting changed")
                episode = {
                    "index": index, "case": case, "ablation_mode": ablation_mode,
                    "evaluation": evaluation, "policy_telemetry": policy_telemetry,
                    "fixed_condition": {
                        "base_seed": case["base_seed"],
                        "episode_index": training_report["calibration_bias"]["episode_index"],
                        "grasp_id": case["grasp_id"],
                        "grasp_index": 1,
                        "bias": observed_bias,
                        "evaluation_rng": evaluation["evaluation_rng"],
                        "initial_base79_sha256": base79_hash,
                        "initial_base79": base79,
                    },
                }
                _atomic_write(args.output_dir / "episodes" / f"{index:02d}-{case_id}-{ablation_mode}.json", episode)
                episodes.append(episode)
                execution_order.append({"index": index, "case_id": case_id, "ablation_mode": ablation_mode})
                _atomic_write(args.output_dir / "partial-results.json", {"status": "running", "episodes": episodes}, exclusive=False)
                if evaluation.get("classification") == "invalid":
                    raise FrozenAblationError("fresh physics evaluation classified invalid")
        for name, record in preimport.items():
            if file_sha256(record["path"]) != record["sha256"]:
                raise FrozenAblationError(f"frozen input changed during execution: {name}")
        for path, digest in frozen_hashes.items():
            if file_sha256(path) != digest:
                raise FrozenAblationError(f"runner frozen input changed: {path}")
        report = {
            "schema_name": SCHEMA_NAME, "status": "completed",
            "scope": "fresh single-environment physics ablation; no training or benchmark claim",
            "learning_executed": False, "checkpoint_selection_changed": False,
            "episodes": episodes, "execution_order": execution_order,
            "contract": {"path": str(args.contract), "sha256": args.contract_sha256},
            "source_binding": preimport, "protocol_record": protocol_record,
            "runtime_import_origins": runtime_import_origins,
            "source_action_analysis": {"path": str(source_actions_path), "sha256": file_sha256(source_actions_path), "canonical_sha256": source_actions_record["canonical_sha256"]},
            "bias_preflight": {"path": str(bias_cases_path), "sha256": file_sha256(bias_cases_path), "runtime_source_canonical_sha256": runtime_source_canonical_sha256},
            "runtime_manifest": args.runtime_manifest_record,
            "execution": {"elapsed_wall_s_before_close": time.monotonic() - started, "application_cleanup_verified_by_report": False},
        }
        exit_code = 0
    except BaseException as error:
        traceback.print_exc()
        terminal_status = (
            "incomplete"
            if isinstance(error, TimeoutError) and "global ablation guard" in str(error)
            else "invalid"
        )
        partial_artifacts = [
            {"path": str(path), "sha256": file_sha256(path)}
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file()
        ]
        report = {
            "schema_name": SCHEMA_NAME, "status": terminal_status, "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__)),
            "learning_executed": False, "ranking_permitted": False,
            "partial_artifacts": partial_artifacts,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException as close_error:
                if report.get("status") == "completed":
                    report = {"schema_name": SCHEMA_NAME, "status": "invalid", "failure_stage": "environment_close", "error": {"type": type(close_error).__name__, "message": str(close_error)}, "learning_executed": False}
                    exit_code = 1
                else:
                    traceback.print_exc()
    if app is None:
        _atomic_write(report_path, report)
        return exit_code
    return persist_report_before_app_close(report_path, report, probe, app, exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
