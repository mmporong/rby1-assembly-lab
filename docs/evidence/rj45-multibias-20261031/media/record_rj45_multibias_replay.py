#!/usr/bin/env python3
"""Film one fresh-05 multi64 or zero replay without changing benchmark evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import types
from typing import Any, Mapping


SCHEMA_NAME = "rj45_multibias_replay_v1"
EXPECTED_HELPER_SHA256 = "7035363ecbab9aa83196760a54b22ab00b2c163bb4a8b4151a825960291e0be5"
EXPECTED_REPLAY_HELPER_SHA256 = "46bc89c708afb4ebb664a488e734b121364c9505462200aeddebc73186410bcd"
DEADLINE_UTC = "2026-10-31T10:50:52Z"
CASE_ID = "fresh-05"
BASE_SEED = 20281240
CALIBRATION_CASE_SEED = 66404260
VALID_CLASSIFICATIONS = {"valid_success", "valid_failure"}


class MultibiasReplayError(RuntimeError):
    pass


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise MultibiasReplayError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
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


def _sha256_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise MultibiasReplayError(f"{field} must be 64 lowercase hex characters")
    return value


def _deadline_epoch() -> float:
    parsed = datetime.fromisoformat(DEADLINE_UTC[:-1] + "+00:00")
    if parsed.tzinfo != timezone.utc:
        raise MultibiasReplayError("replay deadline must be UTC")
    return parsed.timestamp()


def _require_deadline(now: float | None = None) -> None:
    if (time.time() if now is None else now) >= _deadline_epoch():
        raise MultibiasReplayError("replay deadline has expired")


def require_file_sha(path: Path, expected: str, file_sha256: Any, field: str) -> str:
    wanted = _sha256_hex(expected, f"{field} SHA")
    observed = file_sha256(path)
    if observed != wanted:
        raise MultibiasReplayError(f"{field} bytes changed")
    return observed


def read_verified_bytes(path: Path, expected: str, field: str) -> bytes:
    wanted = _sha256_hex(expected, f"{field} SHA")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != wanted:
        raise MultibiasReplayError(f"{field} bytes changed")
    return data


def load_verified_json(path: Path, expected: str, field: str) -> dict[str, Any]:
    data = read_verified_bytes(path, expected, field)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultibiasReplayError(f"{field} is not valid JSON") from error
    if not isinstance(value, dict):
        raise MultibiasReplayError(f"{field} JSON root must be an object")
    return value


def guarded_policy_load(checkpoint: Path, expected_sha256: str, file_sha256: Any, loader: Any) -> Any:
    require_file_sha(checkpoint, expected_sha256, file_sha256, "checkpoint before policy load")
    policy = loader()
    require_file_sha(checkpoint, expected_sha256, file_sha256, "checkpoint after policy load")
    return policy


def validate_checkpoint_payload(
    *, mode: str, payload: Any, checkpoint_sha256: str | None,
    expected_sha256: str, validator: Any,
) -> dict[str, Any] | None:
    if mode == "zero":
        if payload is not None or checkpoint_sha256 is not None:
            raise MultibiasReplayError("zero replay must not receive a checkpoint")
        return None
    if payload is None or checkpoint_sha256 is None:
        raise MultibiasReplayError("multi64 replay requires a checkpoint")
    if checkpoint_sha256 != expected_sha256:
        raise MultibiasReplayError("checkpoint bytes do not match primary and training reports")
    try:
        record = validator(payload, "multi64")
    except Exception as error:
        raise MultibiasReplayError("multi64 checkpoint is not final64") from error
    if record != {"iteration_field": 63, "eligible_update": 64, "eligible_for_evaluation": True}:
        raise MultibiasReplayError("multi64 checkpoint eligibility changed")
    return record


def _case_identity(value: Any) -> dict[str, Any]:
    expected = {
        "case_id": CASE_ID, "base_seed": BASE_SEED,
        "calibration_case_seed": CALIBRATION_CASE_SEED,
        "split": "train", "grasp_id": "rj45-train-02", "episode_index": 1,
        "source": "policy-unseen",
    }
    if value != expected:
        raise MultibiasReplayError("primary fresh-05 case identity changed")
    return expected


def _validate_evaluation_rng(value: Any) -> None:
    expected_seeded = {
        "seed": 20261124, "python": True, "numpy": True, "torch_cpu": True,
        "torch_cuda_all_available_devices": True, "physics_seed_unchanged": 20261128,
    }
    if not isinstance(value, dict) or value.get("seeded") != expected_seeded:
        raise MultibiasReplayError("primary evaluation RNG seed/flags changed")
    state = value.get("state_sha256_before_environment_reset")
    if not isinstance(state, dict) or set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise MultibiasReplayError("primary evaluation RNG state structure changed")
    for name, digest in state.items():
        _sha256_hex(digest, f"primary evaluation RNG {name}")


def validate_primary_lineage(
    *, mode: str, primary_report: Mapping[str, Any], primary_report_sha256: str,
    contract: Mapping[str, Any], contract_sha256: str, train_report: Mapping[str, Any],
    train_report_sha256: str, checkpoint_sha256: str | None,
) -> dict[str, Any]:
    """Select only the preregistered fresh-05 episode from a completed 27-run report."""

    if mode not in {"multi64", "zero"}:
        raise MultibiasReplayError("unsupported replay mode")
    for digest, field in ((primary_report_sha256, "primary report SHA"), (contract_sha256, "contract SHA"), (train_report_sha256, "train report SHA")):
        _sha256_hex(digest, field)
    if contract.get("helper_sha256") != EXPECTED_HELPER_SHA256:
        raise MultibiasReplayError("contract helper identity changed")
    if contract.get("budget_and_stop", {}).get("deadline_utc") != DEADLINE_UTC:
        raise MultibiasReplayError("contract deadline changed")
    if primary_report.get("schema_name") != "rj45_multibias_probe_v1" or primary_report.get("status") != "completed" or primary_report.get("phase") != "evaluate":
        raise MultibiasReplayError("primary evaluation report is not completed")
    binding = primary_report.get("contract")
    if not isinstance(binding, dict) or binding.get("sha256") != contract_sha256:
        raise MultibiasReplayError("primary report contract lineage mismatch")
    source_binding = primary_report.get("source_binding")
    contract_sources = contract.get("source_sha256")
    if not isinstance(source_binding, dict) or not isinstance(contract_sources, dict):
        raise MultibiasReplayError("primary source lineage is missing")
    for name, digest in contract_sources.items():
        record = source_binding.get(name)
        if not isinstance(record, dict) or record.get("sha256") != digest:
            raise MultibiasReplayError(f"primary source lineage mismatch: {name}")
    origin = source_binding.get("origin_protocol")
    contract_origin = contract.get("origin_protocol")
    if not isinstance(origin, dict) or not isinstance(contract_origin, dict) or origin.get("sha256") != contract_origin.get("sha256"):
        raise MultibiasReplayError("primary origin-protocol lineage mismatch")
    episodes = primary_report.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != 27:
        raise MultibiasReplayError("primary evaluation does not contain 27 episodes")
    matches = [episode for episode in episodes if isinstance(episode, dict) and episode.get("method") == mode and isinstance(episode.get("case"), dict) and episode["case"].get("case_id") == CASE_ID]
    if len(matches) != 1:
        raise MultibiasReplayError("primary report has no unique fresh-05 replay target")
    episode = matches[0]
    case = _case_identity(episode.get("case"))
    evaluation = episode.get("evaluation")
    if not isinstance(evaluation, dict) or evaluation.get("classification") not in VALID_CLASSIFICATIONS:
        raise MultibiasReplayError("primary fresh-05 classification is invalid")
    _validate_evaluation_rng(evaluation.get("evaluation_rng"))
    initial_hash = _sha256_hex(episode.get("initial_base79_sha256"), "primary fresh-05 initial79 SHA")
    bias = episode.get("bias")
    sampled = primary_report.get("bias_cases")
    if not isinstance(bias, dict) or bias.get("base_seed") != BASE_SEED or bias.get("episode_index") != [1]:
        raise MultibiasReplayError("primary fresh-05 observed bias identity changed")
    if not isinstance(sampled, list):
        raise MultibiasReplayError("primary bias cases are missing")
    sampled_matches = [item for item in sampled if isinstance(item, dict) and item.get("case_id") == CASE_ID]
    if len(sampled_matches) != 1 or sampled_matches[0].get("base_seed") != BASE_SEED or sampled_matches[0].get("calibration_case_seed") != CALIBRATION_CASE_SEED:
        raise MultibiasReplayError("primary fresh-05 sampled bias identity changed")
    sampled_case = sampled_matches[0]
    if bias.get("translation_entry_m") != [sampled_case.get("translation_entry_m")] or bias.get("rotation_vector_entry_rad") != [sampled_case.get("rotation_vector_entry_rad")]:
        raise MultibiasReplayError("primary fresh-05 observed bias differs from sampled bias")
    if train_report.get("schema_name") != "rj45_multibias_probe_v1" or train_report.get("status") != "completed" or train_report.get("phase") != "train-multi":
        raise MultibiasReplayError("multi64 training report is not completed")
    train_binding = train_report.get("contract")
    if not isinstance(train_binding, dict) or train_binding.get("sha256") != contract_sha256 or train_report.get("source_binding") != source_binding:
        raise MultibiasReplayError("multi64 training report lineage differs from primary")
    training_checkpoint = train_report.get("checkpoint")
    primary_checkpoint = primary_report.get("checkpoints", {}).get("multi64")
    if not isinstance(training_checkpoint, dict) or not isinstance(primary_checkpoint, dict):
        raise MultibiasReplayError("multi64 checkpoint lineage is missing")
    expected_checkpoint = _sha256_hex(training_checkpoint.get("sha256"), "training checkpoint SHA")
    if primary_checkpoint.get("sha256") != expected_checkpoint or primary_checkpoint.get("iteration_field") != 63 or primary_checkpoint.get("eligible_update") != 64 or primary_checkpoint.get("eligible_for_evaluation") is not True:
        raise MultibiasReplayError("primary multi64 checkpoint is not final64")
    if mode == "multi64":
        if checkpoint_sha256 != expected_checkpoint:
            raise MultibiasReplayError("checkpoint bytes do not match primary and training reports")
    elif checkpoint_sha256 is not None:
        raise MultibiasReplayError("zero replay must not receive a checkpoint")
    return {
        "case": case, "method": mode, "classification": evaluation["classification"],
        "initial_base79_sha256": initial_hash,
        "primary_report_sha256": primary_report_sha256,
        "train_report_sha256": train_report_sha256,
        "contract_sha256": contract_sha256,
        "checkpoint_sha256": expected_checkpoint if mode == "multi64" else None,
        "selection_note": "fresh-05 was selected after the sealed benchmark for visualization only",
    }


class FirstEligibleObservation:
    """Capture the already-cached actor input before the raw environment step."""

    def __init__(self, env: Any, ablation: Any, deadline_epoch: float):
        self.env, self.ablation, self.deadline_epoch = env, ablation, deadline_epoch
        self.original = env.step
        self.values: list[float] | None = None
        self.sha256: str | None = None
        env.step = types.MethodType(self.step, env)

    def step(self, _env: Any, action: Any) -> Any:
        if time.time() >= self.deadline_epoch:
            raise TimeoutError("replay deadline exceeded during rollout")
        if self.sha256 is None and bool(self.ablation._eligible_transition(self.env)[0].item()):
            cached = self.env._training_last_policy_observation.clone()
            values = cached.detach().cpu().tolist()[0]
            self.values = [float(value) for value in values]
            self.sha256 = self.ablation.raw_base79_sha256(self.values)
        return self.original(action)


def compare_replay(primary: Mapping[str, Any], observed_first79: str | None, replay_classification: str) -> tuple[str, dict[str, Any]]:
    first_match = observed_first79 == primary["initial_base79_sha256"]
    classification_match = replay_classification == primary["classification"]
    status = "completed" if first_match and classification_match else "invalid"
    return status, {
        "initial_base79_matches": first_match,
        "primary_classification": primary["classification"],
        "replay_classification": replay_classification,
        "classification_matches": classification_match,
        "performance_based_rerun_allowed": False,
    }


def persist_report_before_app_close(path: Path, report: Mapping[str, Any], multibias: Any, probe: Any, app: Any, exit_code: int) -> int:
    return multibias.persist_report_before_app_close(path, report, probe, app, exit_code)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("multi64", "zero"), required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--origin-protocol", type=Path, required=True)
    parser.add_argument("--origin-protocol-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--primary-report", type=Path, required=True)
    parser.add_argument("--primary-report-sha256", required=True)
    parser.add_argument("--train-report", type=Path, required=True)
    parser.add_argument("--train-report-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument("--acknowledge-replay-only", action="store_true", required=True)
    return parser


def _run_main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    for name in ("runtime_root", "current_root", "origin_protocol", "contract", "primary_report", "train_report", "video", "report"):
        setattr(args, name, getattr(args, name).resolve())
    args.checkpoint = args.checkpoint.resolve() if args.checkpoint else None
    if args.video.exists() or args.report.exists():
        raise SystemExit("video/report output already exists")
    if args.video.suffix.lower() != ".mp4":
        raise SystemExit("--video must use the .mp4 suffix")
    _require_deadline()
    multibias_path = args.current_root / "scripts/run_rj45_multibias_probe.py"
    replay_path = args.current_root / "scripts/record_rj45_oracle_probe.py"
    multibias = _load_module(multibias_path, "rj45_multibias_replay_contract")
    if multibias.file_sha256(multibias_path) != EXPECTED_HELPER_SHA256 or multibias.file_sha256(replay_path) != EXPECTED_REPLAY_HELPER_SHA256:
        raise MultibiasReplayError("replay helper source binding changed")
    contract = multibias.validate_contract(args.contract, args.contract_sha256, current_root=args.current_root, origin_protocol=args.origin_protocol)
    preimport = multibias.validate_preimport_sources(args.current_root, args.origin_protocol, contract)
    if multibias.file_sha256(args.origin_protocol) != args.origin_protocol_sha256:
        raise MultibiasReplayError("origin protocol SHA mismatch")
    primary_report = load_verified_json(args.primary_report, args.primary_report_sha256, "primary report")
    train_report = load_verified_json(args.train_report, args.train_report_sha256, "train report")
    checkpoint_bytes = args.checkpoint.read_bytes() if args.checkpoint is not None else None
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest() if checkpoint_bytes is not None else None
    primary = validate_primary_lineage(
        mode=args.mode, primary_report=primary_report, primary_report_sha256=args.primary_report_sha256,
        contract=contract, contract_sha256=args.contract_sha256, train_report=train_report,
        train_report_sha256=args.train_report_sha256, checkpoint_sha256=checkpoint_sha,
    )
    checkpoint_payload = None
    if checkpoint_bytes is not None:
        import torch

        checkpoint_payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False)
    validate_checkpoint_payload(
        mode=args.mode, payload=checkpoint_payload, checkpoint_sha256=checkpoint_sha,
        expected_sha256=primary["checkpoint_sha256"] or "", validator=multibias.validate_final_checkpoint_payload,
    )
    origin_probe = _load_module(Path(preimport["runner"]["path"]), "rj45_multibias_replay_origin_probe")
    protocol_record = origin_probe.validate_protocol(args.origin_protocol, args.origin_protocol_sha256, args.current_root)
    multibias._configure_probe_args(args, origin_probe, protocol_record["payload"])
    args.protocol = args.origin_protocol
    args.probe_report = args.primary_report
    protocol = protocol_record["payload"]
    replay_helper = _load_module(replay_path, "rj45_multibias_replay_renderer")
    ablation = _load_module(Path(preimport["ablation_helper"]["path"]), "rj45_multibias_replay_ablation")
    deadline_epoch = _deadline_epoch()
    app = env = None
    report: dict[str, Any] = {
        "schema_name": SCHEMA_NAME, "status": "invalid", "mode": args.mode,
        "error": {"type": "UnstartedReplay", "message": "replay did not enter the application phase"},
        "claims": {"primary_result_changed": False, "training_evidence": False, "benchmark_statistics": False, "effect_proven": False},
    }
    exit_code = 1
    previous_alarm = None
    try:
        if hasattr(signal, "SIGALRM"):
            previous_alarm = signal.signal(signal.SIGALRM, lambda _s, _f: (_ for _ in ()).throw(TimeoutError("replay deadline alarm expired")))
            signal.setitimer(signal.ITIMER_REAL, deadline_epoch - time.time())
        from isaaclab.app import AppLauncher

        app = AppLauncher({"headless": True, "device": args.device, "enable_cameras": True}).app
        sys.path.insert(0, str(args.runtime_root / "src"))
        env, context, frozen = replay_helper._build_render_env(args, "standard79", protocol, origin_probe)
        env._training_bias_base_seed = BASE_SEED
        runtime_import_origins = origin_probe.verify_runtime_import_origins(args.runtime_root)
        frozen.update({
            Path(__file__).resolve(): multibias.file_sha256(Path(__file__).resolve()),
            multibias_path.resolve(): EXPECTED_HELPER_SHA256,
            args.contract: args.contract_sha256, args.primary_report: args.primary_report_sha256,
            args.train_report: args.train_report_sha256,
        })
        if args.checkpoint is not None:
            frozen[args.checkpoint] = primary["checkpoint_sha256"]
        observer = FirstEligibleObservation(env, ablation, deadline_epoch)
        policy = (
            guarded_policy_load(
                args.checkpoint, primary["checkpoint_sha256"], multibias.file_sha256,
                lambda: replay_helper._policy_for_replay(env, "standard79", args.checkpoint, protocol, origin_probe),
            )
            if args.mode == "multi64" and args.checkpoint is not None else None
        )
        replay = replay_helper._record_rollout(
            env=env, mode="standard79" if args.mode == "multi64" else "zero", policy=policy,
            video_path=args.video, lineage=primary, frozen_hashes=frozen,
            physics_seed=20261128, probe=origin_probe,
        )
        _require_deadline()
        for path, digest in frozen.items():
            if multibias.file_sha256(path) != digest:
                raise MultibiasReplayError(f"frozen input changed during replay: {path}")
        status, comparison = compare_replay(primary, observer.sha256, replay["outcome"]["classification"])
        report = {
            "schema_name": SCHEMA_NAME, "status": status, "mode": args.mode,
            "scope": "fresh-05 visualization replay only; not training or benchmark statistical evidence",
            "fresh_physics_replay": True, "lineage": primary, "replay": replay,
            "comparison": comparison, "runtime_versions": context["versions"],
            "runtime_manifest": args.runtime_manifest_record,
            "runtime_import_origins": runtime_import_origins,
            "claims": {"primary_result_changed": False, "training_evidence": False, "benchmark_statistics": False, "effect_proven": False},
        }
        exit_code = 0 if status == "completed" else 1
    except BaseException as error:
        traceback.print_exc()
        report = {
            "schema_name": SCHEMA_NAME, "status": "invalid", "mode": args.mode,
            "error": {"type": type(error).__name__, "message": str(error)},
            "video_preserved_if_created": args.video.exists(),
            "claims": {"primary_result_changed": False, "training_evidence": False, "benchmark_statistics": False, "effect_proven": False},
        }
        exit_code = 1
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            if previous_alarm is not None:
                signal.signal(signal.SIGALRM, previous_alarm)
        if env is not None:
            try:
                env.close()
            except BaseException:
                traceback.print_exc()
                report = {
                    **report, "status": "invalid",
                    "error": {"type": "EnvironmentCloseError", "message": "environment close failed"},
                }
                exit_code = 1
        exit_code = persist_report_before_app_close(args.report, report, multibias, origin_probe, app, exit_code)
    return exit_code


def main(argv: list[str] | None = None) -> int:
    try:
        return _run_main(argv)
    except Exception as error:
        traceback.print_exc()
        args = _parser().parse_args(argv)
        report_path = args.report.resolve()
        if not report_path.exists():
            _atomic_write_new(report_path, {
                "schema_name": SCHEMA_NAME, "status": "invalid", "mode": args.mode,
                "error": {"type": type(error).__name__, "message": str(error)},
                "video_preserved_if_created": args.video.resolve().exists(),
                "claims": {"primary_result_changed": False, "training_evidence": False, "benchmark_statistics": False, "effect_proven": False},
            })
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
