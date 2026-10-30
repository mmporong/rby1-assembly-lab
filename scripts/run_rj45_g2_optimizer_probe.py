#!/usr/bin/env python3
"""Run the frozen two-recipe G2 optimizer diagnostic on one train trajectory."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_NAME = "rj45_g2_optimizer_probe_v1"
EXPECTED_NPZ_SHA256 = "482215e4507810e04d9a3a1168d0dde6007c9015fe0f3720101543b8e41a3633"
EXPECTED_RUNTIME = {
    "python": "3.11.15",
    "numpy": "1.26.0",
    "torch": "2.7.0+cpu",
}
SEED = 20261030
STEPS = 2000
THREADS = 2
REPORT_INTERVAL = 200
OBSERVATION_DIM = 79
ACTION_DIM = 6
RECIPES = (
    {"id": "e0", "learning_rate": 0.001},
    {"id": "e1", "learning_rate": 0.0001},
)


class OptimizerProbeError(ValueError):
    """The frozen optimizer-probe contract was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_helper() -> Any:
    path = Path(__file__).resolve().with_name("probe_rj45_train_condition.py")
    spec = importlib.util.spec_from_file_location("rj45_fixed_condition_probe", path)
    if spec is None or spec.loader is None:
        raise OptimizerProbeError(f"cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runtime() -> dict[str, str]:
    import numpy as np
    import torch

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }


def _require_runtime(runtime: Mapping[str, str]) -> None:
    if dict(runtime) != EXPECTED_RUNTIME:
        raise OptimizerProbeError(
            f"runtime differs from frozen contract: expected {EXPECTED_RUNTIME}, got {dict(runtime)}"
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise OptimizerProbeError(f"JSON root must be an object: {path}")
    return payload


def _input_identity(manifest_path: Path, helper: Any) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    npz_entry = manifest.get("npz")
    if not isinstance(npz_entry, Mapping):
        raise OptimizerProbeError("input manifest has no NPZ object")
    npz_path = Path(str(npz_entry.get("path", ""))).resolve()
    manifest_npz_sha = str(npz_entry.get("sha256", ""))
    actual_npz_sha = file_sha256(npz_path)
    if manifest_npz_sha != EXPECTED_NPZ_SHA256 or actual_npz_sha != EXPECTED_NPZ_SHA256:
        raise OptimizerProbeError("input NPZ differs from the frozen optimizer-probe input")
    helper_path = Path(helper.__file__).resolve()
    return {
        "manifest": {
            "path": str(manifest_path),
            "sha256": file_sha256(manifest_path),
        },
        "npz": {"path": str(npz_path), "sha256": actual_npz_sha},
        "helper": {"path": str(helper_path), "sha256": file_sha256(helper_path)},
        "runner": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
    }


def _protocol_payload(manifest_path: Path) -> dict[str, Any]:
    helper = _load_helper()
    runtime = _runtime()
    _require_runtime(runtime)
    return {
        "schema_name": SCHEMA_NAME,
        "status": "frozen_before_execution",
        "frozen_at_utc": _utc_now(),
        "purpose": (
            "Compare only the existing learning rate and a 10x lower learning rate on the "
            "same fixed train trajectory under one current CPU runtime."
        ),
        "claims": {
            "same_trajectory_optimizer_diagnostic": True,
            "state_dependent_feedback_proven": False,
            "closed_loop_physics_proven": False,
            "reinforcement_learning_proven": False,
            "heldout_generalization_proven": False,
        },
        "input_identity": _input_identity(manifest_path, helper),
        "runtime": runtime,
        "model_contract": {
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "architecture": "79->256->256->128->6 MLP from build_bc_actor",
            "final_layer_initialization": "zero weight and bias",
        },
        "shared_recipe": {
            "device": "cpu",
            "optimizer": "Adam",
            "full_batch_optimizer_steps": STEPS,
            "seed": SEED,
            "threads": THREADS,
            "report_interval_steps": REPORT_INTERVAL,
            "selection": "final checkpoint only",
            "input": "identical 395 valid rows from the frozen 79D/6D train condition",
        },
        "recipes": [dict(recipe) for recipe in RECIPES],
        "diagnostic_criterion": {
            "definition": (
                "E1 final normalized MSE <= 90% of the constant train-mean MSE, "
                "E1 final MSE < E0 final MSE, and E1's last five reported MSE values "
                "are monotonically non-increasing"
            ),
            "all_conditions_required": True,
            "controls_rl_or_generalization_gate": False,
        },
        "stop_rule": "Run E0 and E1 once; do not launch 20k-step or label-normalization follow-ups.",
    }


def freeze_protocol(manifest_path: str | Path, protocol_path: str | Path) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    destination = Path(protocol_path).resolve()
    if destination.exists():
        raise FileExistsError(f"protocol already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = _protocol_payload(manifest)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(temporary, destination)
    return payload


def _validate_protocol(protocol_path: Path, manifest_path: Path) -> Mapping[str, Any]:
    protocol = _load_json(protocol_path)
    expected = _protocol_payload(manifest_path)
    for volatile in ("frozen_at_utc",):
        expected.pop(volatile, None)
    observed = dict(protocol)
    observed.pop("frozen_at_utc", None)
    if observed != expected:
        raise OptimizerProbeError("protocol identity or frozen contract changed after freezing")
    if protocol.get("status") != "frozen_before_execution":
        raise OptimizerProbeError("protocol was not frozen before execution")
    return protocol


@contextmanager
def _capture_initial_actor_fingerprint() -> Iterator[list[str]]:
    import torch
    import rby1_assembly.rj45_bc as bc

    original = bc.build_bc_actor
    fingerprints: list[str] = []

    def instrumented_build() -> Any:
        actor = original()
        with torch.no_grad():
            actor[-1].weight.zero_()
            actor[-1].bias.zero_()
        state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        fingerprints.append(bc.actor_state_sha256(state))
        return actor

    bc.build_bc_actor = instrumented_build
    try:
        yield fingerprints
    finally:
        bc.build_bc_actor = original


def _criterion(run_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    e0 = run_results["e0"]
    e1 = run_results["e1"]
    e0_final = float(e0["metrics"]["final_actor"]["overall_normalized_mse"])
    e1_final = float(e1["metrics"]["final_actor"]["overall_normalized_mse"])
    constant_mse = float(e1["metrics"]["constant_train_mean_action"]["overall_normalized_mse"])
    last_five = [float(item["train_normalized_mse"]) for item in e1["diagnostics"][-5:]]
    if len(last_five) != 5:
        raise OptimizerProbeError("E1 must contain at least five reported MSE checkpoints")
    checks = {
        "e1_at_most_90_percent_of_constant_mean": e1_final <= 0.9 * constant_mse,
        "e1_lower_than_e0": e1_final < e0_final,
        "e1_last_five_monotonic_nonincrease": all(
            current <= previous for previous, current in zip(last_five, last_five[1:])
        ),
    }
    return {
        "met": all(checks.values()),
        "checks": checks,
        "values": {
            "e0_final_normalized_mse": e0_final,
            "e1_final_normalized_mse": e1_final,
            "constant_train_mean_normalized_mse": constant_mse,
            "e1_to_constant_mean_ratio": e1_final / constant_mse,
            "e1_last_five_reported_mse": last_five,
        },
    }


def _validate_run_pair(
    run_results: Mapping[str, Mapping[str, Any]], initial_fingerprints: Mapping[str, str]
) -> None:
    if set(run_results) != {"e0", "e1"} or set(initial_fingerprints) != {"e0", "e1"}:
        raise OptimizerProbeError("exactly E0 and E1 are required")
    if initial_fingerprints["e0"] != initial_fingerprints["e1"]:
        raise OptimizerProbeError("E0 and E1 did not start from the same actor state")
    for recipe in RECIPES:
        result = run_results[recipe["id"]]
        observed = result["recipe"]
        expected = {
            "seed": SEED,
            "threads": THREADS,
            "learning_rate": recipe["learning_rate"],
            "full_batch_optimizer_steps": STEPS,
            "report_interval_steps": REPORT_INTERVAL,
        }
        if any(observed.get(key) != value for key, value in expected.items()):
            raise OptimizerProbeError(f"{recipe['id']} recipe differs from the frozen contract")


def _rewrite_helper_artifact_paths(run_dir: Path, final_dir: Path) -> Mapping[str, Any]:
    result_path = run_dir / "fit-result.json"
    payload = dict(_load_json(result_path))
    artifact = dict(payload["artifact"])
    artifact["path"] = str((final_dir / "final-diagnostic-actor.pt").resolve())
    payload["artifact"] = artifact
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def run_probe(
    manifest_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    result_path: str | Path,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    destination = Path(output_dir).resolve()
    public_result = Path(result_path).resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if public_result.exists():
        raise FileExistsError(f"result already exists: {public_result}")
    protocol = _validate_protocol(protocol_file, manifest)
    protocol_sha = file_sha256(protocol_file)
    helper = _load_helper()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    run_results: dict[str, Mapping[str, Any]] = {}
    fingerprints: dict[str, str] = {}
    try:
        for recipe in RECIPES:
            run_id = str(recipe["id"])
            with _capture_initial_actor_fingerprint() as captured:
                helper.fit_condition(
                    manifest,
                    temporary / run_id,
                    steps=STEPS,
                    seed=SEED,
                    learning_rate=float(recipe["learning_rate"]),
                    threads=THREADS,
                    report_interval=REPORT_INTERVAL,
                )
            if len(captured) != 1:
                raise OptimizerProbeError(f"{run_id} initial actor fingerprint was not captured once")
            fingerprints[run_id] = captured[0]
            run_results[run_id] = _rewrite_helper_artifact_paths(
                temporary / run_id, destination / run_id
            )
        _validate_run_pair(run_results, fingerprints)
        criterion = _criterion(run_results)
        result = {
            "schema_name": SCHEMA_NAME,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "protocol": {
                "path": str(protocol_file),
                "sha256": protocol_sha,
                "frozen_at_utc": protocol["frozen_at_utc"],
            },
            "input_identity": protocol["input_identity"],
            "runtime": _runtime(),
            "initial_actor": {
                "same_state_for_both_runs": fingerprints["e0"] == fingerprints["e1"],
                "actor_state_sha256": fingerprints["e0"],
                "per_run": fingerprints,
            },
            "runs": {
                run_id: {
                    "recipe": run_results[run_id]["recipe"],
                    "metrics": run_results[run_id]["metrics"],
                    "diagnostics": run_results[run_id]["diagnostics"],
                    "artifact": run_results[run_id]["artifact"],
                }
                for run_id in ("e0", "e1")
            },
            "diagnostic_criterion": criterion,
            "claims": protocol["claims"],
            "interpretation_limit": (
                "This is a same-trajectory offline regression diagnostic. It does not establish "
                "state-dependent feedback, closed-loop physics success, RL contribution, or "
                "held-out generalization."
            ),
            "stop_rule_observed": True,
        }
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if file_sha256(protocol_file) != protocol_sha:
            raise OptimizerProbeError("protocol changed during execution")
        os.rename(temporary, destination)
        public_result.parent.mkdir(parents=True, exist_ok=True)
        public_temporary = public_result.with_name(f".{public_result.name}.{os.getpid()}.tmp")
        public_temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.rename(public_temporary, public_result)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze the protocol before running training")
    freeze.add_argument("--manifest", required=True, type=Path)
    freeze.add_argument("--protocol", required=True, type=Path)
    run = commands.add_parser("run", help="run exactly E0 and E1 under the frozen protocol")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--result", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "freeze":
        payload = freeze_protocol(arguments.manifest, arguments.protocol)
    else:
        payload = run_probe(
            arguments.manifest,
            arguments.protocol,
            arguments.output_dir,
            arguments.result,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
