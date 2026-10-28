#!/usr/bin/env python3
"""Validate RJ45 scale benchmark evidence and select the frozen rollout topology."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, NoReturn


CANDIDATES = {64: 128, 128: 64, 256: 32}
MEASURED_STEPS = tuple(range(3, 8))
CANONICAL_PPO_SHA256 = "ccbdaddfa0be15406174c31a5ef9a2651e2e46e97f4ac71b4ec8fb86f027dffd"
COLLECTION_TAG = "Perf/collection_time"
LEARNING_TAG = "Perf/learning_time"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class BenchmarkError(RuntimeError):
    pass


def _fail(message: str) -> NoReturn:
    raise BenchmarkError(message)


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        _fail(f"{name} contains non-finite constant {value}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise BenchmarkError(f"invalid {name} {path}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _ref(root: Path, value: Any, name: str) -> tuple[Path, str]:
    if not isinstance(value, dict):
        _fail(f"{name} must be a file reference")
    raw, digest = value.get("path"), value.get("sha256")
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
        _fail(f"{name}.path must be project-relative")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        _fail(f"{name}.sha256 must be lowercase SHA-256")
    path = (root / raw).resolve()
    if not path.is_relative_to(root) or _sha_file(path) != digest:
        _fail(f"{name} bytes do not match SHA-256")
    return path, digest


def _run_dir(root: Path, value: Any, num_envs: int) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts:
        _fail(f"candidate {num_envs} run_dir must be project-relative")
    path = (root / value).resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        _fail(f"candidate {num_envs} run_dir is missing or outside project root")
    return path


def _default_scalar_loader(event_file: Path, tag: str) -> list[tuple[int, float]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as error:
        raise BenchmarkError("TensorBoard is required to read benchmark event files") from error
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    if tag not in accumulator.Tags().get("scalars", []):
        _fail(f"TensorBoard event file is missing scalar tag {tag}")
    return [(int(item.step), float(item.value)) for item in accumulator.Scalars(tag)]


def _default_checkpoint_verifier(
    checkpoint: Path,
    manifest: Path,
    provenance: Path,
    runner: Path,
    training_script: Path,
) -> dict[str, Any]:
    from inspect_rj45_checkpoint import inspect_checkpoint
    import torch

    result = inspect_checkpoint(
        checkpoint_path=checkpoint,
        manifest_path=manifest,
        training_provenance_path=provenance,
        runner_config_path=runner,
        training_script_path=training_script,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        _fail("checkpoint payload must be an object")
    counts: dict[str, int] = {}
    for key in ("model_state_dict", "optimizer_state_dict"):
        if not isinstance(payload.get(key), dict):
            _fail(f"checkpoint is missing {key}")
        count = 0
        stack = [payload[key]]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                stack.extend(value.values())
            elif isinstance(value, (list, tuple)):
                stack.extend(value)
            elif isinstance(value, torch.Tensor):
                count += 1
                if not bool(torch.isfinite(value).all().item()):
                    _fail(f"checkpoint {key} contains non-finite tensor values")
            elif isinstance(value, float) and not math.isfinite(value):
                _fail(f"checkpoint {key} contains non-finite scalar values")
        if count == 0:
            _fail(f"checkpoint {key} contains no tensors")
        counts[key] = count
    result["finite_tensor_audit"] = {"status": "all_finite", "tensor_counts": counts}
    return result


def _default_evidence_verifier(root: Path, nominal: Path, held: Path, contact: Path) -> dict[str, Any]:
    from rby1_assembly.experiment_contracts import load_finite_json, verify_rj45_nominal_evidence
    from rby1_assembly.held_grasp import load_verified_held_grasp_artifact
    from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence
    from rby1_assembly.rj45_task import load_rj45_task

    task = load_rj45_task(root / "configs/tasks/rj45-v1.json")
    nominal_result = verify_rj45_nominal_evidence(root, task, nominal)
    _artifact, held_result = load_verified_held_grasp_artifact(root, task, held)
    contact_result = load_parallel_contact_evidence(load_finite_json(contact), project_root=root)
    return {"nominal": nominal_result, "held": held_result, "parallel_contact": contact_result}


def _timings(event_file: Path, loader: Callable[[Path, str], list[tuple[int, float]]]) -> list[dict[str, float | int]]:
    by_tag: dict[str, dict[int, float]] = {}
    for tag in (COLLECTION_TAG, LEARNING_TAG):
        rows = loader(event_file, tag)
        values: dict[int, float] = {}
        for step, value in rows:
            if step in values:
                _fail(f"duplicate TensorBoard step {step} for {tag}")
            if not math.isfinite(value) or value <= 0:
                _fail(f"{tag} step {step} must be finite and positive")
            values[step] = value
        if set(values) != set(range(8)):
            _fail(f"{tag} must contain exactly updates 0..7")
        by_tag[tag] = values
    return [
        {"step": step, "collection_time_s": by_tag[COLLECTION_TAG][step], "learning_time_s": by_tag[LEARNING_TAG][step]}
        for step in MEASURED_STEPS
    ]


def _validate_protocol(protocol: dict[str, Any]) -> None:
    scale = protocol.get("scale_benchmark")
    if not isinstance(scale, dict):
        _fail("protocol.scale_benchmark is missing")
    expected = {
        "candidate_num_envs": [64, 128, 256],
        "steps_per_env": [128, 64, 32],
        "transitions_per_update": 8192,
        "warmup_updates": 3,
        "measured_updates": 5,
        "seed": 20261110,
        "profile": "Z1",
    }
    for key, value in expected.items():
        if scale.get(key) != value:
            _fail(f"protocol scale benchmark {key} differs from the frozen contract")


def _validate_success(root: Path, candidate: dict[str, Any], num_envs: int, steps: int, scalar_loader: Callable[[Path, str], list[tuple[int, float]]], checkpoint_verifier: Callable[..., dict[str, Any]], evidence_verifier: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    run_dir = _run_dir(root, candidate.get("run_dir"), num_envs)
    event_file, event_sha = _ref(root, candidate.get("event_file"), f"candidate {num_envs} event_file")
    checkpoint, checkpoint_sha = _ref(root, candidate.get("checkpoint"), f"candidate {num_envs} checkpoint")
    inspection_path, inspection_sha = _ref(root, candidate.get("checkpoint_inspection"), f"candidate {num_envs} inspection")
    runner_path, runner_sha = _ref(root, candidate.get("runner_config"), f"candidate {num_envs} runner_config")
    manifest_path, manifest_sha = _ref(root, candidate.get("checkpoint_manifest"), f"candidate {num_envs} checkpoint_manifest")
    provenance_path, provenance_sha = _ref(root, candidate.get("training_provenance"), f"candidate {num_envs} training_provenance")
    training_script, training_script_sha = _ref(root, candidate.get("training_script"), f"candidate {num_envs} training_script")
    nominal, nominal_sha = _ref(root, candidate.get("nominal_evidence"), f"candidate {num_envs} nominal_evidence")
    held, held_sha = _ref(root, candidate.get("held_state"), f"candidate {num_envs} held_state")
    contact, contact_sha = _ref(root, candidate.get("parallel_contact_evidence"), f"candidate {num_envs} parallel_contact_evidence")
    raw_log, raw_log_sha = _ref(root, candidate.get("raw_log"), f"candidate {num_envs} raw_log")
    if any(not path.is_relative_to(run_dir) for path in (event_file, checkpoint, inspection_path, runner_path, manifest_path, provenance_path, raw_log)):
        _fail(f"candidate {num_envs} run artifacts must be inside run_dir")
    inspection = _load_json(inspection_path, "checkpoint inspection")
    details = inspection.get("checkpoint")
    loader_proof = inspection.get("loader")
    if (
        inspection.get("status") != "checkpoint_payload_and_lineage_verified"
        or inspection.get("planned_training_updates") != 8
        or not isinstance(details, dict)
        or details.get("filename_iteration") != 7
        or details.get("payload_iteration") != 7
        or details.get("completed_updates") != 8
        or details.get("sha256") != checkpoint_sha
        or loader_proof != {"map_location": "cpu", "weights_only": True}
    ):
        _fail(f"candidate {num_envs} checkpoint inspection is not a finite completed 8-update proof")
    recomputed = checkpoint_verifier(checkpoint, manifest_path, provenance_path, runner_path, training_script)
    finite_audit = recomputed.get("finite_tensor_audit") if isinstance(recomputed, dict) else None
    recomputed_base = dict(recomputed) if isinstance(recomputed, dict) else {}
    recomputed_base.pop("finite_tensor_audit", None)
    if recomputed_base != inspection:
        _fail(f"candidate {num_envs} checkpoint inspection differs from actual checkpoint recomputation")
    if not isinstance(finite_audit, dict) or finite_audit.get("status") != "all_finite":
        _fail(f"candidate {num_envs} checkpoint model and optimizer finite audit failed")
    evidence_result = evidence_verifier(root, nominal, held, contact)
    if not isinstance(evidence_result, dict) or set(evidence_result) != {"nominal", "held", "parallel_contact"}:
        _fail(f"candidate {num_envs} physical evidence verification is incomplete")
    runner = _load_json(runner_path, "runner config")
    canonical_path = root / "configs/ppo-rj45-low-exploration-v1.json"
    if _sha_file(canonical_path) != CANONICAL_PPO_SHA256:
        _fail("canonical Z1 PPO bytes changed")
    expected_runner = _load_json(canonical_path, "canonical Z1 PPO")
    expected_runner.update({"seed": 20261110, "num_steps_per_env": steps, "max_iterations": 8, "save_interval": 7})
    if runner != expected_runner:
        _fail(f"candidate {num_envs} runner config differs from canonical Z1 benchmark config")
    from run_rj45_long_campaign import _check_process_text
    _check_process_text(raw_log.read_text(encoding="utf-8", errors="replace"), raw_log)
    validity = candidate.get("validity")
    if not isinstance(validity, dict) or validity.get("valid") is not True:
        _fail(f"candidate {num_envs} is not marked valid")
    for field in ("parallel_contact_verified", "no_nonfinite_or_gpu_memory_errors", "finite_checkpoint_and_optimizer_state"):
        if validity.get(field) is not True:
            _fail(f"candidate {num_envs} validity gate {field} did not pass")
    proof_refs = validity.get("physical_proof_refs")
    if not isinstance(proof_refs, list) or not proof_refs:
        _fail(f"candidate {num_envs} must pin physical proof references")
    proof_hashes = []
    for index, value in enumerate(proof_refs):
        path, digest = _ref(root, value, f"candidate {num_envs} proof {index}")
        proof_hashes.append({"path": str(path.relative_to(root)), "sha256": digest})
    timings = _timings(event_file, scalar_loader)
    total_time = sum(float(row["collection_time_s"]) + float(row["learning_time_s"]) for row in timings)
    throughput = 8192 * 5 / total_time
    return {
        "status": "valid",
        "num_envs": num_envs,
        "num_steps_per_env": steps,
        "gpu_device": candidate.get("gpu_device"),
        "event_file_sha256": event_sha,
        "checkpoint_sha256": checkpoint_sha,
        "inspection_sha256": inspection_sha,
        "runner_config_sha256": runner_sha,
        "checkpoint_manifest_sha256": manifest_sha,
        "training_provenance_sha256": provenance_sha,
        "training_script_sha256": training_script_sha,
        "nominal_evidence_sha256": nominal_sha,
        "held_state_sha256": held_sha,
        "parallel_contact_evidence_sha256": contact_sha,
        "raw_log_sha256": raw_log_sha,
        "physical_proof_refs": proof_hashes,
        "measured_timings": timings,
        "measured_total_time_s": total_time,
        "throughput_transitions_per_s": throughput,
    }


def summarize(root: Path, protocol_path: Path, protocol_sha: str, spec_path: Path, spec_sha: str, *, scalar_loader: Callable[[Path, str], list[tuple[int, float]]] = _default_scalar_loader, checkpoint_verifier: Callable[..., dict[str, Any]] = _default_checkpoint_verifier, evidence_verifier: Callable[..., dict[str, Any]] = _default_evidence_verifier) -> dict[str, Any]:
    root = root.resolve()
    if _sha_file(protocol_path) != protocol_sha or _sha_file(spec_path) != spec_sha:
        _fail("protocol or benchmark spec SHA-256 mismatch")
    protocol, spec = _load_json(protocol_path, "protocol"), _load_json(spec_path, "benchmark spec")
    forbidden_keys = {"success", "successes", "success_rate", "reward", "rewards", "episode_outcomes"}
    if any(key in forbidden_keys for candidate in spec.get("candidates", []) if isinstance(candidate, dict) for key in candidate):
        _fail("benchmark spec must not contain success or reward outcomes")
    _validate_protocol(protocol)
    raw = spec.get("candidates")
    if not isinstance(raw, list) or len(raw) != 3:
        _fail("benchmark spec must contain exactly three candidates")
    by_num = {entry.get("num_envs"): entry for entry in raw if isinstance(entry, dict)}
    if set(by_num) != set(CANDIDATES) or len(by_num) != 3:
        _fail("benchmark candidates must be unique 64, 128, and 256")
    results: dict[str, Any] = {}
    for num_envs, steps in CANDIDATES.items():
        candidate = by_num[num_envs]
        if candidate.get("num_steps_per_env") != steps or candidate.get("transitions_per_update") != 8192:
            _fail(f"candidate {num_envs} topology differs from protocol")
        gpu = candidate.get("gpu_device")
        if not isinstance(gpu, str) or not gpu.startswith("cuda:"):
            _fail(f"candidate {num_envs} must record actual GPU device")
        exit_code = candidate.get("exit_code")
        if type(exit_code) is not int:
            _fail(f"candidate {num_envs} exit_code must be integer")
        if exit_code != 0:
            run_dir = _run_dir(root, candidate.get("run_dir"), num_envs)
            raw_log, raw_sha = _ref(root, candidate.get("raw_log"), f"candidate {num_envs} raw_log")
            if not raw_log.is_relative_to(run_dir):
                _fail(f"candidate {num_envs} failure raw log must be inside run_dir")
            failure = candidate.get("failure")
            if not isinstance(failure, dict) or failure.get("kind") not in {"oom", "runtime_error"} or not isinstance(failure.get("message"), str) or not failure["message"]:
                _fail(f"candidate {num_envs} failed run lacks explicit runtime error evidence")
            raw_text = raw_log.read_text(encoding="utf-8", errors="replace")
            if failure["kind"] == "oom" and not any(token in raw_text.lower() for token in ("out of memory", "outofmemory", "oom")):
                _fail(f"candidate {num_envs} OOM classification is not supported by raw log")
            if failure["kind"] == "runtime_error" and failure["message"] not in raw_text:
                _fail(f"candidate {num_envs} runtime error message is not present in raw log")
            results[str(num_envs)] = {"status": "runtime_failure", "num_envs": num_envs, "num_steps_per_env": steps, "gpu_device": gpu, "exit_code": exit_code, "failure": failure, "raw_log": {"path": str(raw_log.relative_to(root)), "sha256": raw_sha}}
        else:
            results[str(num_envs)] = _validate_success(root, candidate, num_envs, steps, scalar_loader, checkpoint_verifier, evidence_verifier)
    valid = [entry for entry in results.values() if entry["status"] == "valid"]
    if not valid:
        _fail("no valid scale benchmark candidate remains")
    highest = max(entry["throughput_transitions_per_s"] for entry in valid)
    eligible = [entry for entry in valid if entry["throughput_transitions_per_s"] >= highest * 0.9]
    selected = min(eligible, key=lambda entry: entry["num_envs"])
    result = {
        "schema_version": 1,
        "selection_basis": "throughput_and_correctness_only_no_success_or_reward_outcomes",
        "protocol": {"path": str(protocol_path.resolve().relative_to(root)), "sha256": protocol_sha},
        "benchmark_spec": {"path": str(spec_path.resolve().relative_to(root)), "sha256": spec_sha},
        "candidates": results,
        "selected_topology": {"num_envs": selected["num_envs"], "num_steps_per_env": selected["num_steps_per_env"], "transitions_per_update": 8192, "selection_rule": "smallest_valid_candidate_at_least_90_percent_of_highest_throughput"},
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["result_content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        _fail("output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--benchmark-spec", required=True, type=Path)
    parser.add_argument("--benchmark-spec-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = summarize(args.project_root, args.protocol, args.protocol_sha256, args.benchmark_spec, args.benchmark_spec_sha256)
    _write_new(args.output, result)
    print(json.dumps({"output": str(args.output), "selected_topology": result["selected_topology"], "result_content_sha256": result["result_content_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
