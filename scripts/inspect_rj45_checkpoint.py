#!/usr/bin/env python3
"""Inspect an RJ45 training checkpoint and its immutable lineage bundle."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, NoReturn


CHECKPOINT_NAME = re.compile(r"^model_(\d+)\.pt$")


def _fail(message: str) -> NoReturn:
    raise ValueError(message)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        _fail(f"{name} must be an object")
    return value


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{name} must be an integer at least {minimum}")
    return value


def _sha(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        _fail(f"{name} must be a lowercase SHA-256")
    return value


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> tuple[dict[str, Any], str]:
    def reject(constant: str) -> None:
        _fail(f"{name} contains non-finite JSON constant {constant}")

    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_constant=reject)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {name} JSON {path}: {error}") from error
    return _object(value, name), hashlib.sha256(raw).hexdigest()


def inspect_checkpoint(
    *,
    checkpoint_path: Path,
    manifest_path: Path,
    training_provenance_path: Path,
    runner_config_path: Path,
    training_script_path: Path,
) -> dict[str, Any]:
    match = CHECKPOINT_NAME.fullmatch(checkpoint_path.name)
    if match is None:
        _fail("checkpoint filename must use model_N.pt")
    filename_iteration = int(match.group(1))

    try:
        checkpoint_bytes = checkpoint_path.read_bytes()
    except OSError as error:
        raise ValueError(f"checkpoint is unreadable: {checkpoint_path}: {error}") from error
    checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    manifest, manifest_sha = _load_json(manifest_path, "checkpoint manifest")
    provenance, provenance_sha = _load_json(training_provenance_path, "training provenance")
    runner, runner_sha = _load_json(runner_config_path, "runner config")
    training_script_sha = _sha_file(training_script_path)

    if manifest.get("schema_version") != 1:
        _fail("checkpoint manifest schema_version must be 1")
    if manifest.get("training_completed") is not True:
        _fail("checkpoint manifest does not mark training as completed")
    if manifest.get("training_provenance_sha256") != provenance_sha:
        _fail("checkpoint manifest does not reference the supplied training provenance")
    checkpoints = _object(manifest.get("checkpoints"), "checkpoint manifest checkpoints")
    for filename, digest in checkpoints.items():
        if not isinstance(filename, str) or not filename:
            _fail("checkpoint manifest contains an invalid checkpoint filename")
        _sha(digest, f"checkpoint manifest checkpoints[{filename!r}]")
    if checkpoints.get(checkpoint_path.name) != checkpoint_sha:
        _fail("completed checkpoint manifest does not match the supplied checkpoint")

    source_and_config = _object(
        provenance.get("source_and_config_sha256"),
        "training provenance source_and_config_sha256",
    )
    if provenance.get("checkpoint_scope") != "checkpoints created in this run directory":
        _fail("training provenance does not prove a fresh run-local checkpoint scope")
    runtime_versions = _object(
        provenance.get("runtime_versions"), "training provenance runtime_versions"
    )
    if runtime_versions.get("rsl-rl-lib") != "3.0.1":
        _fail("checkpoint completion derivation requires rsl-rl-lib 3.0.1")
    if source_and_config.get("scripts/train.py") != training_script_sha:
        _fail("training provenance does not match the supplied training script bytes")
    runner_matches = [
        digest
        for path, digest in source_and_config.items()
        if isinstance(path, str) and Path(path).name == runner_config_path.name
    ]
    if runner_matches != [runner_sha]:
        _fail("runner config bytes are not uniquely pinned by training provenance")
    provenance_seed = _integer(provenance.get("seed"), "training provenance seed")
    runner_seed = _integer(runner.get("seed"), "runner seed")
    if runner_seed != provenance_seed:
        _fail("runner seed differs from training provenance")
    planned_updates = _integer(
        runner.get("max_iterations"), "runner max_iterations", minimum=1
    )

    try:
        import torch
    except ImportError as error:
        raise RuntimeError("torch is required to inspect checkpoint payloads") from error
    try:
        payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError(f"checkpoint payload could not be loaded safely: {error}") from error
    payload = _object(payload, "checkpoint payload")
    checkpoint_iteration = _integer(payload.get("iter"), "checkpoint payload iter")
    if checkpoint_iteration != filename_iteration:
        _fail("checkpoint payload iter differs from model_N.pt filename")
    completed_updates = checkpoint_iteration + 1
    if completed_updates > planned_updates:
        _fail("checkpoint completed updates exceed runner max_iterations")

    return {
        "schema_version": 2,
        "status": "checkpoint_payload_and_lineage_verified",
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": checkpoint_sha,
            "filename_iteration": filename_iteration,
            "payload_iteration": checkpoint_iteration,
            "completed_updates": completed_updates,
        },
        "planned_training_updates": planned_updates,
        "checkpoint_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
            "training_completed": True,
        },
        "training_provenance": {
            "path": str(training_provenance_path),
            "sha256": provenance_sha,
            "seed": provenance_seed,
        },
        "runner_config": {
            "path": str(runner_config_path),
            "sha256": runner_sha,
            "seed": runner_seed,
        },
        "loader": {"map_location": "cpu", "weights_only": True},
        "completion_derivation": {
            "formula": "completed_updates = checkpoint payload iter + 1",
            "training_mode_basis": (
                "assumed_fresh_no_resume_requires_review_of_pinned_training_script"
            ),
            "checkpoint_scope": "checkpoints created in this run directory",
            "rsl_rl_version": "3.0.1",
            "training_script": {
                "path": str(training_script_path),
                "sha256": training_script_sha,
            },
        },
    }


def _verify_unchanged(expected: dict[Path, str]) -> None:
    for path, digest in expected.items():
        try:
            current = _sha_file(path)
        except OSError as error:
            raise ValueError(f"input changed or became unreadable during inspection: {path}") from error
        if current != digest:
            _fail(f"input changed during inspection: {path}")


def _write_new(path: Path, text: str) -> None:
    if path.exists():
        _fail(f"output already exists and will not be overwritten: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError(f"output already exists and will not be overwritten: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--training-provenance", required=True, type=Path)
    parser.add_argument("--runner-config", required=True, type=Path)
    parser.add_argument("--training-script", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = (
        args.checkpoint,
        args.manifest,
        args.training_provenance,
        args.runner_config,
        args.training_script,
    )
    resolved_inputs = {path.resolve() for path in inputs}
    if len(resolved_inputs) != len(inputs):
        _fail("checkpoint inspection inputs must be distinct files")
    if args.output.resolve() in resolved_inputs:
        _fail("output must not overwrite an inspection input")
    input_hashes = {path: _sha_file(path) for path in inputs}
    result = inspect_checkpoint(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        training_provenance_path=args.training_provenance,
        runner_config_path=args.runner_config,
        training_script_path=args.training_script,
    )
    _verify_unchanged(input_hashes)
    _write_new(args.output, json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), **result}, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
