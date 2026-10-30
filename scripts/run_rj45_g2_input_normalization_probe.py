#!/usr/bin/env python3
"""Compare frozen axis-scaled BC fitting with and without input normalization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "rj45_g2_input_normalization_probe_v1"
EXPECTED_RUNTIME = {"python": "3.11.15", "numpy": "1.26.0", "torch": "2.7.0+cpu"}
PRIOR_RESULT = Path("docs/evidence/rj45-g2-axis-scaling-20261030/result.json")
PRIOR_RUNNER = Path("scripts/run_rj45_g2_axis_scaling_probe.py")
EXPECTED_PRIOR_RESULT_SHA256 = "e0b588869c9bb7ec8df1b69d92457e1aa3729016dc1e53239d3bb000c99b32f0"
EXPECTED_PRIOR_RUNNER_SHA256 = "729d81a0da7643c32361b559d6f1548aba9f7d60502e7096f4ba65902b6f750d"
OBSERVATION_DIM = 79
ACTION_DIM = 6
INPUT_STD_FLOOR = 1e-4
ABS_MSE_REPRODUCTION_TOLERANCE = 1e-12
PREVIOUS_COMMAND_COLUMNS = tuple(range(50, 62))
ESTIMATED_RELATIVE_POSE_COLUMNS = tuple(range(0, 7))


class InputNormalizationProbeError(ValueError):
    """The frozen input-normalization diagnostic contract was violated."""


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise InputNormalizationProbeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _axis_runner() -> Any:
    path = Path(__file__).resolve().with_name("run_rj45_g2_axis_scaling_probe.py")
    return _load_module(path, "rj45_g2_axis_scaling_reuse")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputNormalizationProbeError(f"cannot load JSON object: {path}") from error
    if not isinstance(payload, Mapping):
        raise InputNormalizationProbeError(f"JSON root must be an object: {path}")
    return payload


def _array_sha256(value: Any) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    return __import__("hashlib").sha256(array.tobytes(order="C")).hexdigest()


def _normalizer(observation: Any, zero_columns: Sequence[int] = ()) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(observation)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != OBSERVATION_DIM:
        raise InputNormalizationProbeError("observation must have shape (N, 79), N > 0")
    values64 = values.astype(np.float64, copy=False)
    if not bool(np.isfinite(values64).all()):
        raise InputNormalizationProbeError("observation must contain only finite values")
    mean64 = np.mean(values64, axis=0, dtype=np.float64)
    std64 = np.std(values64, axis=0, ddof=0, dtype=np.float64)
    scale64 = np.maximum(std64, INPUT_STD_FLOOR)
    constant = std64 == 0.0
    mask = np.zeros(OBSERVATION_DIM, dtype=np.bool_)
    for column in zero_columns:
        if not 0 <= int(column) < OBSERVATION_DIM:
            raise InputNormalizationProbeError(f"feature mask column is out of range: {column}")
        mask[int(column)] = True
    return {
        "mean64": mean64,
        "std64": std64,
        "scale64": scale64,
        "mean32": mean64.astype(np.float32),
        "scale32": scale64.astype(np.float32),
        "constant_mask": constant,
        "feature_zero_mask": mask,
    }


def _normalizer_record(normalizer: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np

    record: dict[str, Any] = {
        "estimator": "train_rows_float64_population_std_ddof_0",
        "std_floor": INPUT_STD_FLOOR,
        "maximum_inverse_scale_from_floor": 1.0 / INPUT_STD_FLOOR,
        "normalization": "(float32_observation - mean32) / scale32, then feature mask to zero",
    }
    for name in ("mean64", "std64", "scale64", "mean32", "scale32"):
        array = np.asarray(normalizer[name])
        record[name] = array.tolist()
        record[f"{name}_dtype"] = str(array.dtype)
        record[f"{name}_sha256"] = _array_sha256(array)
    for name in ("constant_mask", "feature_zero_mask"):
        array = np.asarray(normalizer[name], dtype=np.bool_)
        record[name] = array.tolist()
        record[f"{name}_sha256"] = _array_sha256(array)
        record[f"{name}_columns"] = np.flatnonzero(array).astype(int).tolist()
    return record


def _apply_normalizer(observation: Any, normalizer: Mapping[str, Any]) -> Any:
    import numpy as np

    values = np.asarray(observation, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != OBSERVATION_DIM:
        raise InputNormalizationProbeError("inference observation must have shape (N, 79)")
    if not bool(np.isfinite(values).all()):
        raise InputNormalizationProbeError("inference observation must contain only finite values")
    mean = np.asarray(normalizer["mean32"], dtype=np.float32)
    scale = np.asarray(normalizer["scale32"], dtype=np.float32)
    constant = np.asarray(normalizer["constant_mask"], dtype=np.bool_)
    feature_mask = np.asarray(normalizer["feature_zero_mask"], dtype=np.bool_)
    if any(array.shape != (OBSERVATION_DIM,) for array in (mean, scale, constant, feature_mask)):
        raise InputNormalizationProbeError("normalizer arrays must have shape (79,)")
    if not bool(np.isfinite(mean).all()) or not bool(np.isfinite(scale).all()) or bool((scale <= 0).any()):
        raise InputNormalizationProbeError("normalizer mean/scale must be finite and scale positive")
    normalized = (values - mean) / scale
    normalized[:, constant | feature_mask] = 0.0
    if not bool(np.isfinite(normalized).all()):
        raise InputNormalizationProbeError("normalized observation became non-finite")
    return normalized


def _source_identity(manifest_path: Path, axis: Any) -> dict[str, Any]:
    import rby1_assembly.rj45_bc as bc

    repo = Path(__file__).resolve().parents[1]
    prior_result = (repo / PRIOR_RESULT).resolve()
    prior_runner = (repo / PRIOR_RUNNER).resolve()
    if axis.file_sha256(prior_result) != EXPECTED_PRIOR_RESULT_SHA256:
        raise InputNormalizationProbeError("prior axis-scaling result differs from frozen input")
    if axis.file_sha256(prior_runner) != EXPECTED_PRIOR_RUNNER_SHA256:
        raise InputNormalizationProbeError("axis-scaling runner differs from frozen input")
    helper = axis._helper()
    observation, action, manifest = axis._load_input(manifest_path, helper)
    npz_path = Path(str(manifest["npz"]["path"])).resolve()
    return {
        "runner": {"path": str(Path(__file__).resolve()), "sha256": axis.file_sha256(__file__)},
        "axis_scaling_runner": {"path": str(prior_runner), "sha256": axis.file_sha256(prior_runner)},
        "axis_scaling_result": {"path": str(prior_result), "sha256": axis.file_sha256(prior_result)},
        "input_helper": {"path": str(Path(helper.__file__).resolve()), "sha256": axis.file_sha256(helper.__file__)},
        "bc_source": {"path": str(Path(bc.__file__).resolve()), "sha256": axis.file_sha256(bc.__file__)},
        "manifest": {"path": str(manifest_path), "sha256": axis.file_sha256(manifest_path)},
        "npz": {"path": str(npz_path), "sha256": axis.file_sha256(npz_path)},
        "observation_shape": list(observation.shape),
        "action_shape": list(action.shape),
    }


def _prior_contract(axis: Any) -> dict[str, Any]:
    prior_path = (Path(__file__).resolve().parents[1] / PRIOR_RESULT).resolve()
    prior = _load_json(prior_path)
    try:
        r1 = prior["runs"]["r1_axis_scaled_mse"]
        return {
            "initial_actor_state_sha256": str(prior["initial_identity"]["actor_state_sha256"]),
            "initial_rng_state_sha256": str(prior["initial_identity"]["rng_state_sha256"]),
            "final_actor_state_sha256": str(r1["artifact"]["actor_state_sha256"]),
            "final_original_coordinate_mse": float(r1["metrics"]["overall_original_coordinate_mse"]),
            "absolute_mse_tolerance": ABS_MSE_REPRODUCTION_TOLERANCE,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise InputNormalizationProbeError("prior axis-scaling result schema is invalid") from error


def _protocol_payload(manifest_path: Path) -> dict[str, Any]:
    axis = _axis_runner()
    runtime = axis._runtime()
    if runtime != EXPECTED_RUNTIME:
        raise InputNormalizationProbeError(f"runtime differs from frozen contract: {runtime}")
    helper = axis._helper()
    observation, action, _ = axis._load_input(manifest_path, helper)
    base_normalizer = _normalizer(observation)
    return {
        "schema_name": SCHEMA_NAME,
        "status": "frozen_before_execution",
        "frozen_at_utc": _utc_now(),
        "purpose": "Test whether train-only input normalization improves the frozen axis-scaled BC fit.",
        "source_identity": _source_identity(manifest_path, axis),
        "runtime": runtime,
        "normalizer": _normalizer_record(base_normalizer),
        "recipe": {
            "architecture": "79->256->256->128->6 ELU MLP from build_bc_actor",
            "device": "cpu",
            "optimizer": "Adam",
            "learning_rate": axis.LEARNING_RATE,
            "full_batch_optimizer_steps": axis.STEPS,
            "seed": axis.SEED,
            "threads": axis.THREADS,
            "report_interval_steps": axis.REPORT_INTERVAL,
            "final_layer_initialization": "zero weight and bias",
            "loss": "axis_scaled_mse using the frozen 6D action-axis population standard deviations",
            "selection": "final checkpoint only",
            "output": "original normalized 6D command; no output unscale or clipping",
        },
        "runs": {
            "a_axis_scaled_standard_input": {"input_transform": "identity", "required": True},
            "b_axis_scaled_train_normalized_input": {"input_transform": "frozen train-only normalizer", "required": True},
            "c_zero_previous_command": {
                "input_transform": "B normalizer then columns 50:62 zero",
                "required_only_if_b_passes": True,
            },
            "d_zero_estimated_relative_pose": {
                "input_transform": "B normalizer then columns 0:7 zero",
                "required_only_if_b_passes": True,
            },
        },
        "a_reproduction_gate": _prior_contract(axis),
        "b_pass_criterion": {
            "all_values_finite": True,
            "raw_mse_at_most_90_percent_of_constant_mean": True,
            "axis_balanced_mse_at_most": 0.90,
            "raw_mse_to_a_at_most": 0.95,
            "raw_mse_lower_than_a": True,
            "each_axis_mse_to_constant_axis_mse_at_most": 1.10,
            "last_five_reports_monotonic_nonincreasing": "reported_only_not_a_gate",
        },
        "conditional_feature_contrasts": {
            "c_to_b_raw_mse_ratio_at_most_1_05": "previous-command variation had little fit value on this trajectory",
            "c_to_b_raw_mse_ratio_at_least_1_10": "previous-command columns may contain useful fit information",
            "d_to_b_raw_mse_ratio_at_least_1_10": "removing estimated relative pose worsened this fit",
            "intermediate_values": "inconclusive",
        },
        "claims": {
            "same_trajectory_regression_only": True,
            "state_feedback_proven": False,
            "causal_failure_reason_proven": False,
            "closed_loop_physics_proven": False,
            "reinforcement_learning_proven": False,
            "heldout_generalization_proven": False,
        },
        "artifact_contract": "diagnostic_not_main_bc; normalizer is mandatory at inference; main PPO injection forbidden",
        "budget_and_stop_rule": "A and B once; C and D once only if B passes; no longer run, floor/LR search, or rerun after source/protocol change.",
        "input_contract": {
            "rows": int(observation.shape[0]),
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "current_observation_only": True,
            "ground_truth_input_added": False,
            "schema_shape_changed": False,
            "action_statistics_sha256": _array_sha256(action.astype("float32")),
        },
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
    observed = dict(_load_json(protocol_path))
    expected = _protocol_payload(manifest_path)
    observed.pop("frozen_at_utc", None)
    expected.pop("frozen_at_utc", None)
    if observed != expected:
        raise InputNormalizationProbeError("protocol identity or frozen contract changed after freezing")
    return _load_json(protocol_path)


def _identity_normalizer() -> dict[str, Any]:
    import numpy as np

    return {
        "mean64": np.zeros(OBSERVATION_DIM, dtype=np.float64),
        "std64": np.ones(OBSERVATION_DIM, dtype=np.float64),
        "scale64": np.ones(OBSERVATION_DIM, dtype=np.float64),
        "mean32": np.zeros(OBSERVATION_DIM, dtype=np.float32),
        "scale32": np.ones(OBSERVATION_DIM, dtype=np.float32),
        "constant_mask": np.zeros(OBSERVATION_DIM, dtype=np.bool_),
        "feature_zero_mask": np.zeros(OBSERVATION_DIM, dtype=np.bool_),
    }


def _checkpoint_payload(run: Mapping[str, Any], normalizer: Mapping[str, Any], transform_kind: str) -> dict[str, Any]:
    import numpy as np

    transform = {
        "kind": transform_kind,
        "estimator": "train_rows_float64_population_std_ddof_0",
        "std_floor": INPUT_STD_FLOOR,
        "mean64": np.asarray(normalizer["mean64"], dtype=np.float64),
        "std64": np.asarray(normalizer["std64"], dtype=np.float64),
        "scale64": np.asarray(normalizer["scale64"], dtype=np.float64),
        "mean32": np.asarray(normalizer["mean32"], dtype=np.float32),
        "scale32": np.asarray(normalizer["scale32"], dtype=np.float32),
        "constant_mask": np.asarray(normalizer["constant_mask"], dtype=np.bool_),
        "feature_zero_mask": np.asarray(normalizer["feature_zero_mask"], dtype=np.bool_),
    }
    transform["array_sha256"] = {
        key: _array_sha256(value)
        for key, value in transform.items()
        if key not in {"kind", "estimator", "std_floor"}
    }
    return {
        "artifact_kind": "diagnostic_not_main_bc",
        "must_not_be_injected_into_main_experiment": True,
        "actor_state_dict": run["state"],
        "actor_state_sha256": run["final_actor_state_sha256"],
        "input_transform": transform,
        "inference_output_coordinates": "original_normalized_6d_command",
        "output_unscale_or_clip": False,
    }


def _load_checkpoint(path: str | Path) -> dict[str, Any]:
    import numpy as np
    import torch
    import rby1_assembly.rj45_bc as bc

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("artifact_kind") != "diagnostic_not_main_bc":
        raise InputNormalizationProbeError("checkpoint is not a diagnostic BC artifact")
    if "input_transform" not in payload:
        raise InputNormalizationProbeError("checkpoint normalizer is missing")
    transform = payload["input_transform"]
    if not isinstance(transform, Mapping):
        raise InputNormalizationProbeError("checkpoint normalizer is invalid")
    arrays: dict[str, Any] = {}
    fields = (
        ("mean64", np.float64),
        ("std64", np.float64),
        ("scale64", np.float64),
        ("mean32", np.float32),
        ("scale32", np.float32),
        ("constant_mask", np.bool_),
        ("feature_zero_mask", np.bool_),
    )
    for key, dtype in fields:
        if key not in transform:
            raise InputNormalizationProbeError(f"checkpoint normalizer field is missing: {key}")
        arrays[key] = np.asarray(transform[key], dtype=dtype)
        if arrays[key].shape != (OBSERVATION_DIM,):
            raise InputNormalizationProbeError("checkpoint normalizer arrays must have shape (79,)")
        expected_sha = transform.get("array_sha256", {}).get(key)
        if expected_sha != _array_sha256(arrays[key]):
            raise InputNormalizationProbeError(f"checkpoint normalizer hash mismatch: {key}")
    numeric = (arrays["mean64"], arrays["std64"], arrays["scale64"], arrays["mean32"], arrays["scale32"])
    if not all(bool(np.isfinite(array).all()) for array in numeric) or bool((arrays["scale64"] <= 0).any()) or bool((arrays["scale32"] <= 0).any()):
        raise InputNormalizationProbeError("checkpoint normalizer has non-finite or non-positive values")
    state = payload.get("actor_state_dict")
    if not isinstance(state, Mapping) or bc.actor_state_sha256(state) != payload.get("actor_state_sha256"):
        raise InputNormalizationProbeError("checkpoint actor state hash mismatch")
    return {"payload": dict(payload), "normalizer": arrays}


def predict_checkpoint(path: str | Path, observation: Any) -> Any:
    import torch
    import rby1_assembly.rj45_bc as bc

    loaded = _load_checkpoint(path)
    normalized = _apply_normalizer(observation, loaded["normalizer"])
    actor = bc.build_bc_actor().cpu()
    actor.load_state_dict(loaded["payload"]["actor_state_dict"], strict=True)
    actor.eval()
    with torch.no_grad():
        prediction = actor(torch.as_tensor(normalized, dtype=torch.float32)).cpu().numpy()
    if not bool(__import__("numpy").isfinite(prediction).all()):
        raise InputNormalizationProbeError("checkpoint prediction became non-finite")
    return prediction


def _predict_state(state: Mapping[str, Any], normalizer: Mapping[str, Any], observation: Any) -> Any:
    import torch
    import rby1_assembly.rj45_bc as bc

    transformed = _apply_normalizer(observation, normalizer)
    actor = bc.build_bc_actor().cpu()
    actor.load_state_dict(state, strict=True)
    actor.eval()
    with torch.no_grad():
        return actor(torch.as_tensor(transformed, dtype=torch.float32)).cpu().numpy()


def _reproduction_gate(run: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = protocol["a_reproduction_gate"]
    mse = float(run["metrics"]["overall_original_coordinate_mse"])
    checks = {
        "initial_actor_state_sha256": run["initial_actor_state_sha256"] == expected["initial_actor_state_sha256"],
        "initial_rng_state_sha256": run["initial_rng_state_sha256"] == expected["initial_rng_state_sha256"],
        "final_actor_state_sha256": run["final_actor_state_sha256"] == expected["final_actor_state_sha256"],
        "final_original_coordinate_mse": abs(mse - float(expected["final_original_coordinate_mse"])) <= float(expected["absolute_mse_tolerance"]),
    }
    return {"met": all(checks.values()), "checks": checks, "actual_mse": mse}


def _last_five_monotonic(run: Mapping[str, Any]) -> bool:
    values = [float(item["axis_balanced_mse"]) for item in run["diagnostics"][-5:]]
    return len(values) == 5 and all(next_value <= value for value, next_value in zip(values, values[1:]))


def _b_criterion(a: Mapping[str, Any], b: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    a_mse = float(a["metrics"]["overall_original_coordinate_mse"])
    b_mse = float(b["metrics"]["overall_original_coordinate_mse"])
    base_mse = float(baseline["overall_original_coordinate_mse"])
    axis_ratios = {
        name: float(b["metrics"]["axis_original_coordinate_mse"][name]) / float(baseline["axis_original_coordinate_mse"][name])
        for name in b["metrics"]["axis_original_coordinate_mse"]
    }
    numbers = [a_mse, b_mse, base_mse, float(b["metrics"]["axis_balanced_mse"]), *axis_ratios.values()]
    checks = {
        "all_values_finite": all(math.isfinite(value) for value in numbers),
        "b_at_most_90_percent_of_constant_mean": b_mse <= 0.90 * base_mse,
        "b_axis_balanced_mse_at_most_0_90": float(b["metrics"]["axis_balanced_mse"]) <= 0.90,
        "b_to_a_raw_mse_at_most_0_95": b_mse <= 0.95 * a_mse,
        "b_lower_than_a": b_mse < a_mse,
        "all_axis_ratios_at_most_1_10": all(value <= 1.10 for value in axis_ratios.values()),
    }
    return {
        "met": all(checks.values()),
        "checks": checks,
        "reported_only": {"last_five_reports_monotonic_nonincreasing": _last_five_monotonic(b)},
        "values": {
            "a_raw_mse": a_mse,
            "b_raw_mse": b_mse,
            "constant_mean_raw_mse": base_mse,
            "b_to_a_raw_mse_ratio": b_mse / a_mse,
            "b_to_constant_mean_raw_mse_ratio": b_mse / base_mse,
            "b_axis_balanced_mse": float(b["metrics"]["axis_balanced_mse"]),
            "b_axis_mse_to_constant_axis_mse": axis_ratios,
        },
    }


def _run_record(run: Mapping[str, Any], artifact: Mapping[str, Any], transform: str) -> dict[str, Any]:
    return {
        "input_transform": transform,
        "metrics": run["metrics"],
        "diagnostics": run["diagnostics"],
        "parameters_finite": run["parameters_finite"],
        "initial_actor_state_sha256": run["initial_actor_state_sha256"],
        "initial_rng_state_sha256": run["initial_rng_state_sha256"],
        "artifact": artifact,
    }


def _run_feature_contrasts(
    axis: Any,
    observation: Any,
    action: Any,
    training_axis_std: Any,
    metric_axis_std: Any,
    initial_identity: tuple[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    runs: dict[str, Any] = {}
    normalizers: dict[str, Any] = {}
    transforms: dict[str, str] = {}
    for run_id, columns, kind in (
        ("c_zero_previous_command", PREVIOUS_COMMAND_COLUMNS, "train_standardize_zero_previous_command_50_62"),
        ("d_zero_estimated_relative_pose", ESTIMATED_RELATIVE_POSE_COLUMNS, "train_standardize_zero_estimated_relative_pose_0_7"),
    ):
        variant = _normalizer(observation, columns)
        transformed = _apply_normalizer(observation, variant)
        run = axis._train_once(
            transformed, action, training_axis_std, metric_axis_std, "axis_scaled_mse"
        )
        if (run["initial_actor_state_sha256"], run["initial_rng_state_sha256"]) != initial_identity:
            raise InputNormalizationProbeError(f"{run_id} initial actor/RNG identity differs")
        runs[run_id] = run
        normalizers[run_id] = variant
        transforms[run_id] = kind
    return runs, normalizers, transforms


def _maybe_run_feature_contrasts(
    criterion: Mapping[str, Any],
    axis: Any,
    observation: Any,
    action: Any,
    training_axis_std: Any,
    metric_axis_std: Any,
    initial_identity: tuple[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    if not bool(criterion.get("met")):
        return {}, {}, {}
    return _run_feature_contrasts(
        axis,
        observation,
        action,
        training_axis_std,
        metric_axis_std,
        initial_identity,
    )


def run_probe(manifest_path: str | Path, protocol_path: str | Path, output_dir: str | Path, result_path: str | Path) -> dict[str, Any]:
    import numpy as np
    import torch

    manifest = Path(manifest_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    destination = Path(output_dir).resolve()
    public_result = Path(result_path).resolve()
    if destination.exists() or public_result.exists():
        raise FileExistsError("output directory and public result must not already exist")
    protocol = _validate_protocol(protocol_file, manifest)
    axis = _axis_runner()
    helper = axis._helper()
    observation, action, _ = axis._load_input(manifest, helper)
    action_stats = axis._axis_statistics(action)
    metric_axis_std = np.asarray([action_stats["effective_std"][name] for name in axis.AXIS_NAMES], dtype=np.float64)
    training_axis_std = metric_axis_std.astype(np.float32)
    normalizer = _normalizer(observation)
    normalized = _apply_normalizer(observation, normalizer)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    protocol_sha = axis.file_sha256(protocol_file)
    try:
        a = axis._train_once(observation, action, training_axis_std, metric_axis_std, "axis_scaled_mse")
        gate = _reproduction_gate(a, protocol)
        if not gate["met"]:
            raise InputNormalizationProbeError("A did not reproduce frozen R1; B was not started")
        b = axis._train_once(normalized, action, training_axis_std, metric_axis_std, "axis_scaled_mse")
        if (a["initial_actor_state_sha256"], a["initial_rng_state_sha256"]) != (b["initial_actor_state_sha256"], b["initial_rng_state_sha256"]):
            raise InputNormalizationProbeError("A and B initial actor/RNG identities differ")
        baseline = axis._constant_baseline(action, metric_axis_std)
        criterion = _b_criterion(a, b, baseline)
        runs: dict[str, Any] = {"a_axis_scaled_standard_input": a, "b_axis_scaled_train_normalized_input": b}
        run_normalizers: dict[str, Any] = {"a_axis_scaled_standard_input": _identity_normalizer(), "b_axis_scaled_train_normalized_input": normalizer}
        transforms = {"a_axis_scaled_standard_input": "identity", "b_axis_scaled_train_normalized_input": "train_standardize"}
        contrasts: dict[str, Any] = {"executed": False, "reason": "B failed the predeclared gate"}
        contrast_runs, contrast_normalizers, contrast_transforms = _maybe_run_feature_contrasts(
            criterion,
            axis,
            observation,
            action,
            training_axis_std,
            metric_axis_std,
            (a["initial_actor_state_sha256"], a["initial_rng_state_sha256"]),
        )
        if contrast_runs:
            runs.update(contrast_runs)
            run_normalizers.update(contrast_normalizers)
            transforms.update(contrast_transforms)
            b_mse = float(b["metrics"]["overall_original_coordinate_mse"])
            c_ratio = float(runs["c_zero_previous_command"]["metrics"]["overall_original_coordinate_mse"]) / b_mse
            d_ratio = float(runs["d_zero_estimated_relative_pose"]["metrics"]["overall_original_coordinate_mse"]) / b_mse
            contrasts = {
                "executed": True,
                "c_to_b_raw_mse_ratio": c_ratio,
                "c_interpretation": "little_fit_value_on_this_trajectory" if c_ratio <= 1.05 else ("possibly_useful_fit_information" if c_ratio >= 1.10 else "inconclusive"),
                "d_to_b_raw_mse_ratio": d_ratio,
                "d_interpretation": "fit_worsened_when_pose_removed" if d_ratio >= 1.10 else "inconclusive",
            }
        artifacts: dict[str, Any] = {}
        for run_id, run in runs.items():
            run_dir = temporary / run_id
            run_dir.mkdir()
            checkpoint = run_dir / "final-diagnostic-actor.pt"
            torch.save(_checkpoint_payload(run, run_normalizers[run_id], transforms[run_id]), checkpoint)
            loaded_prediction = predict_checkpoint(checkpoint, observation)
            expected_prediction = _predict_state(run["state"], run_normalizers[run_id], observation)
            if loaded_prediction.shape != (395, ACTION_DIM) or not np.array_equal(loaded_prediction, expected_prediction):
                raise InputNormalizationProbeError(f"checkpoint inference differs from training evaluation: {run_id}")
            artifacts[run_id] = {
                "path": str(destination / run_id / checkpoint.name),
                "sha256": axis.file_sha256(checkpoint),
                "actor_state_sha256": run["final_actor_state_sha256"],
                "input_transform_array_sha256": _checkpoint_payload(run, run_normalizers[run_id], transforms[run_id])["input_transform"]["array_sha256"],
            }
        result = {
            "schema_name": SCHEMA_NAME,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "protocol": {"path": str(protocol_file), "sha256": protocol_sha, "frozen_at_utc": protocol["frozen_at_utc"]},
            "source_identity": protocol["source_identity"],
            "runtime": axis._runtime(),
            "a_reproduction_gate": gate,
            "initial_identity": {
                "actor_state_sha256": a["initial_actor_state_sha256"],
                "rng_state_sha256": a["initial_rng_state_sha256"],
                "all_executed_runs_match": all(run["initial_actor_state_sha256"] == a["initial_actor_state_sha256"] and run["initial_rng_state_sha256"] == a["initial_rng_state_sha256"] for run in runs.values()),
            },
            "constant_train_mean_baseline": baseline,
            "runs": {run_id: _run_record(run, artifacts[run_id], transforms[run_id]) for run_id, run in runs.items()},
            "b_pass_criterion": criterion,
            "conditional_feature_contrasts": contrasts,
            "claims": protocol["claims"],
            "interpretation_limit": "Same-train-trajectory regression only; no state-feedback, causal, physics, RL, or held-out-generalization claim.",
            "stop_rule_observed": True,
        }
        (temporary / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if axis.file_sha256(protocol_file) != protocol_sha or _source_identity(manifest, axis) != protocol["source_identity"]:
            raise InputNormalizationProbeError("protocol or frozen source changed during execution")
        os.rename(temporary, destination)
        public_result.parent.mkdir(parents=True, exist_ok=True)
        result_tmp = public_result.with_name(f".{public_result.name}.{os.getpid()}.tmp")
        result_tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.rename(result_tmp, public_result)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--manifest", required=True, type=Path)
    freeze.add_argument("--protocol", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--result", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    payload = freeze_protocol(arguments.manifest, arguments.protocol) if arguments.command == "freeze" else run_probe(arguments.manifest, arguments.protocol, arguments.output_dir, arguments.result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
