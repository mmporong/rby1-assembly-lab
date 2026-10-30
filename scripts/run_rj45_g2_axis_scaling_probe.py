#!/usr/bin/env python3
"""Compare raw and per-axis-scaled BC losses on one frozen train trajectory."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "rj45_g2_axis_scaling_probe_v1"
EXPECTED_NPZ_SHA256 = "482215e4507810e04d9a3a1168d0dde6007c9015fe0f3720101543b8e41a3633"
EXPECTED_RUNTIME = {"python": "3.11.15", "numpy": "1.26.0", "torch": "2.7.0+cpu"}
HISTORICAL_RESULT_PATH = Path(
    "docs/evidence/rj45-g2-optimizer-probe-20261030/result.json"
)
HISTORICAL_RESULT_SHA256 = "49fd48f379d77c3ff6bafc955a35a40b48584ae3550941d8f9809a72ffcaaf03"
SEED = 20261030
STEPS = 2000
THREADS = 2
REPORT_INTERVAL = 200
LEARNING_RATE = 0.001
OBSERVATION_DIM = 79
ACTION_DIM = 6
AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
STD_FLOOR = 1e-8
ABS_MSE_REPRODUCTION_TOLERANCE = 1e-12
RUNS = (
    {"id": "r0_raw_mse", "loss": "raw_mse"},
    {"id": "r1_axis_scaled_mse", "loss": "axis_scaled_mse"},
)


class AxisScalingProbeError(ValueError):
    """The frozen axis-scaling diagnostic contract was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(value: Any) -> str:
    return hashlib.sha256(value).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AxisScalingProbeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _helper() -> Any:
    return _load_module(
        Path(__file__).resolve().with_name("probe_rj45_train_condition.py"),
        "rj45_axis_scaling_input_helper",
    )


def _runtime() -> dict[str, str]:
    import numpy as np
    import torch

    return {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}


def _require_runtime(runtime: Mapping[str, str]) -> None:
    if dict(runtime) != EXPECTED_RUNTIME:
        raise AxisScalingProbeError(
            f"runtime differs from frozen contract: expected {EXPECTED_RUNTIME}, got {dict(runtime)}"
        )


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AxisScalingProbeError(f"cannot load JSON object: {path}") from error
    if not isinstance(payload, Mapping):
        raise AxisScalingProbeError(f"JSON root must be an object: {path}")
    return payload


def _load_input(manifest_path: Path, helper: Any) -> tuple[Any, Any, Mapping[str, Any]]:
    import numpy as np

    manifest, observation, action = helper._load_probe_input(manifest_path)
    observation = np.asarray(observation, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    if observation.shape != (395, OBSERVATION_DIM) or action.shape != (395, ACTION_DIM):
        raise AxisScalingProbeError("frozen input must contain 395 valid 79D/6D rows")
    if not bool(np.isfinite(observation).all()) or not bool(np.isfinite(action).all()):
        raise AxisScalingProbeError("frozen input must contain only finite values")
    npz_path = Path(str(manifest["npz"]["path"])).resolve()
    if file_sha256(npz_path) != EXPECTED_NPZ_SHA256:
        raise AxisScalingProbeError("input NPZ differs from the frozen axis-scaling input")
    return observation, action, manifest


def _axis_statistics(action: Any) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(action, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != ACTION_DIM:
        raise AxisScalingProbeError("action must have shape (N, 6), N > 0")
    if not bool(np.isfinite(values).all()):
        raise AxisScalingProbeError("action must contain only finite values")
    mean = np.mean(values, axis=0, dtype=np.float64)
    raw_std = np.std(values, axis=0, ddof=0, dtype=np.float64)
    if not bool(np.isfinite(raw_std).all()):
        raise AxisScalingProbeError("axis standard deviation must be finite")
    floored = raw_std < STD_FLOOR
    effective = np.maximum(raw_std, STD_FLOOR)
    return {
        "axis_order": list(AXIS_NAMES),
        "estimator": "numpy_float64_population_std_ddof_0",
        "row_count": int(values.shape[0]),
        "mean": {name: float(value) for name, value in zip(AXIS_NAMES, mean, strict=True)},
        "raw_std": {name: float(value) for name, value in zip(AXIS_NAMES, raw_std, strict=True)},
        "effective_std": {
            name: float(value) for name, value in zip(AXIS_NAMES, effective, strict=True)
        },
        "std_floor": STD_FLOOR,
        "floor_applied": {
            name: bool(value) for name, value in zip(AXIS_NAMES, floored, strict=True)
        },
        "any_floor_applied": bool(floored.any()),
    }


def _source_identity(manifest_path: Path, helper: Any, action: Any) -> dict[str, Any]:
    import rby1_assembly.rj45_bc as bc

    historical_path = (Path(__file__).resolve().parents[1] / HISTORICAL_RESULT_PATH).resolve()
    historical_sha = file_sha256(historical_path)
    if historical_sha != HISTORICAL_RESULT_SHA256:
        raise AxisScalingProbeError("historical optimizer result differs from the frozen source")
    manifest = _load_json(manifest_path)
    npz_path = Path(str(manifest["npz"]["path"])).resolve()
    helper_path = Path(helper.__file__).resolve()
    reused_runner = Path(__file__).resolve().with_name("run_rj45_g2_optimizer_probe.py")
    bc_path = Path(bc.__file__).resolve()
    return {
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "npz": {"path": str(npz_path), "sha256": file_sha256(npz_path)},
        "input_helper": {"path": str(helper_path), "sha256": file_sha256(helper_path)},
        "runner": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(__file__)},
        "prior_optimizer_runner": {
            "path": str(reused_runner),
            "sha256": file_sha256(reused_runner),
        },
        "bc_source": {"path": str(bc_path), "sha256": file_sha256(bc_path)},
        "historical_result": {"path": str(historical_path), "sha256": historical_sha},
        "axis_statistics": _axis_statistics(action),
    }


def _protocol_payload(manifest_path: Path) -> dict[str, Any]:
    helper = _helper()
    runtime = _runtime()
    _require_runtime(runtime)
    _, action, _ = _load_input(manifest_path, helper)
    historical_path = (Path(__file__).resolve().parents[1] / HISTORICAL_RESULT_PATH).resolve()
    historical = _load_json(historical_path)
    try:
        historical_initial_sha = str(historical["initial_actor"]["actor_state_sha256"])
        historical_e0 = historical["runs"]["e0"]
        historical_final_sha = str(historical_e0["artifact"]["actor_state_sha256"])
        historical_final_mse = float(
            historical_e0["metrics"]["final_actor"]["overall_normalized_mse"]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AxisScalingProbeError("historical optimizer result schema is invalid") from error
    if not all(
        len(value) == 64 and all(character in "0123456789abcdef" for character in value)
        for value in (historical_initial_sha, historical_final_sha)
    ) or not math.isfinite(historical_final_mse):
        raise AxisScalingProbeError("historical optimizer result values are invalid")
    return {
        "schema_name": SCHEMA_NAME,
        "status": "frozen_before_execution",
        "frozen_at_utc": _utc_now(),
        "purpose": "Compare raw MSE and per-axis population-standard-deviation-scaled MSE.",
        "claims": {
            "same_trajectory_fit_diagnostic": True,
            "state_feedback_proven": False,
            "closed_loop_physics_proven": False,
            "reinforcement_learning_proven": False,
            "heldout_generalization_proven": False,
        },
        "source_identity": _source_identity(manifest_path, helper, action),
        "runtime": runtime,
        "model_contract": {
            "observation_dim": OBSERVATION_DIM,
            "action_dim": ACTION_DIM,
            "architecture": "79->256->256->128->6 ELU MLP from build_bc_actor",
            "final_layer_initialization": "zero weight and bias",
            "inference_output": "original normalized 6D command coordinates",
            "inference_rescaling_or_clipping": False,
        },
        "shared_recipe": {
            "device": "cpu",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "full_batch_optimizer_steps": STEPS,
            "seed": SEED,
            "threads": THREADS,
            "report_interval_steps": REPORT_INTERVAL,
            "selection": "final checkpoint only",
            "rows": 395,
        },
        "runs": [dict(run) for run in RUNS],
        "loss_contract": {
            "r0_raw_mse": "mean((prediction - target)^2)",
            "r1_axis_scaled_mse": "mean(((prediction - target) / effective_axis_std)^2)",
            "axis_std_derivation": (
                "395 train rows, float64, ddof=0; retain float64 for evaluation and "
                "cast to float32 only for the training loss tensor"
            ),
            "centering": "omitted because subtracting the same axis mean from both sides cancels",
        },
        "historical_reproduction_gate": {
            "required_before_r1": True,
            "source": "SHA-256-verified historical_result JSON",
            "final_actor_state_sha256": historical_final_sha,
            "initial_actor_state_sha256": historical_initial_sha,
            "final_original_coordinate_mse": historical_final_mse,
            "absolute_mse_tolerance": ABS_MSE_REPRODUCTION_TOLERANCE,
        },
        "pass_criterion": {
            "all_conditions_required": True,
            "r1_original_mse_at_most_90_percent_of_constant_mean": True,
            "r1_original_mse_lower_than_r0": True,
            "r1_axis_balanced_mse_at_most": 0.90,
            "each_axis_mse_to_constant_axis_mse_at_most": 1.10,
            "all_values_finite": True,
        },
        "stop_rule": "Run R0 and R1 once; no 20k-step extension or hyperparameter search.",
        "artifact_contract": "diagnostic_not_main_bc; must not be injected into the main experiment",
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
    for key in ("frozen_at_utc",):
        observed.pop(key, None)
        expected.pop(key, None)
    if observed != expected:
        raise AxisScalingProbeError("protocol identity or frozen contract changed after freezing")
    return _load_json(protocol_path)


def _axis_metrics(prediction: Any, target: Any, axis_std: Any) -> dict[str, Any]:
    import numpy as np

    prediction64 = np.asarray(prediction, dtype=np.float64)
    target64 = np.asarray(target, dtype=np.float64)
    std64 = np.asarray(axis_std, dtype=np.float64)
    if prediction64.shape != target64.shape or target64.ndim != 2 or target64.shape[1] != ACTION_DIM:
        raise AxisScalingProbeError("prediction and target must have aligned shape (N, 6)")
    if std64.shape != (ACTION_DIM,) or not bool(np.isfinite(std64).all()) or bool((std64 <= 0).any()):
        raise AxisScalingProbeError("effective axis standard deviation must be finite and positive")
    if not bool(np.isfinite(prediction64).all()) or not bool(np.isfinite(target64).all()):
        raise AxisScalingProbeError("metric inputs must contain only finite values")
    error = prediction64 - target64
    axis_mse = np.mean(np.square(error), axis=0, dtype=np.float64)
    balanced = float(np.mean(np.square(error / std64), dtype=np.float64))
    correlations: dict[str, float | None] = {}
    prediction_std = np.std(prediction64, axis=0, ddof=0, dtype=np.float64)
    target_std = np.std(target64, axis=0, ddof=0, dtype=np.float64)
    correlation_floor = np.finfo(np.float64).eps
    for index, name in enumerate(AXIS_NAMES):
        prediction_constant = bool(np.all(prediction64[:, index] == prediction64[0, index]))
        target_constant = bool(np.all(target64[:, index] == target64[0, index]))
        if (
            prediction_constant
            or target_constant
            or prediction_std[index] <= correlation_floor
            or target_std[index] <= correlation_floor
        ):
            correlations[name] = None
        else:
            value = float(np.corrcoef(prediction64[:, index], target64[:, index])[0, 1])
            correlations[name] = value if math.isfinite(value) else None
    return {
        "overall_original_coordinate_mse": float(np.mean(np.square(error), dtype=np.float64)),
        "axis_original_coordinate_mse": {
            name: float(value) for name, value in zip(AXIS_NAMES, axis_mse, strict=True)
        },
        "axis_balanced_mse": balanced,
        "axis_correlation": correlations,
        "output_std_to_target_std": {
            name: (float(prediction_std[i] / target_std[i]) if target_std[i] > 0 else None)
            for i, name in enumerate(AXIS_NAMES)
        },
    }


def _constant_baseline(target: Any, axis_std: Any) -> dict[str, Any]:
    import numpy as np

    target64 = np.asarray(target, dtype=np.float64)
    prediction = np.broadcast_to(np.mean(target64, axis=0, dtype=np.float64), target64.shape)
    return _axis_metrics(prediction, target64, axis_std)


def _training_loss(prediction: Any, target: Any, axis_std: Any, loss_kind: str) -> Any:
    import torch.nn.functional as functional

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != ACTION_DIM:
        raise AxisScalingProbeError("training tensors must have aligned shape (N, 6)")
    if axis_std.shape != (ACTION_DIM,) or not bool(axis_std.isfinite().all()) or bool((axis_std <= 0).any()):
        raise AxisScalingProbeError("training axis standard deviation must be finite and positive")
    if not bool(prediction.isfinite().all()) or not bool(target.isfinite().all()):
        raise AxisScalingProbeError("training tensors must contain only finite values")
    if loss_kind == "raw_mse":
        return functional.mse_loss(prediction, target)
    if loss_kind == "axis_scaled_mse":
        return ((prediction - target) / axis_std).square().mean()
    raise AxisScalingProbeError(f"unknown loss kind: {loss_kind}")


def _train_once(
    observation_np: Any,
    action_np: Any,
    training_axis_std_np: Any,
    metric_axis_std_np: Any,
    loss_kind: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as functional
    import rby1_assembly.rj45_bc as bc

    torch.set_num_threads(THREADS)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    observation = torch.as_tensor(observation_np, dtype=torch.float32, device="cpu")
    action = torch.as_tensor(action_np, dtype=torch.float32, device="cpu")
    axis_std = torch.as_tensor(training_axis_std_np, dtype=torch.float32, device="cpu")
    actor = bc.build_bc_actor().cpu()
    with torch.no_grad():
        actor[-1].weight.zero_()
        actor[-1].bias.zero_()
    initial_state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    initial_sha = bc.actor_state_sha256(initial_state)
    rng_sha = _bytes_sha256(torch.get_rng_state().cpu().numpy().tobytes())
    optimizer = torch.optim.Adam(actor.parameters(), lr=LEARNING_RATE)
    diagnostics: list[dict[str, Any]] = []
    actor.train()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = actor(observation)
        loss = _training_loss(prediction, action, axis_std, loss_kind)
        if not bool(torch.isfinite(loss)):
            raise AxisScalingProbeError("training loss became non-finite")
        loss.backward()
        optimizer.step()
        if step % REPORT_INTERVAL == 0 or step == STEPS:
            actor.eval()
            with torch.no_grad():
                current = actor(observation)
                original_mse = float(functional.mse_loss(current, action))
                balanced_mse = float(torch.mean(torch.square((current - action) / axis_std)))
            if not math.isfinite(original_mse) or not math.isfinite(balanced_mse):
                raise AxisScalingProbeError("diagnostic metric became non-finite")
            diagnostics.append(
                {
                    "optimizer_step": step,
                    "training_loss": balanced_mse if loss_kind == "axis_scaled_mse" else original_mse,
                    "training_loss_coordinate": loss_kind,
                    "original_coordinate_mse": original_mse,
                    "axis_balanced_mse": balanced_mse,
                }
            )
            actor.train()
    actor.eval()
    with torch.no_grad():
        prediction_np = actor(observation).cpu().numpy()
    state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
    parameters_finite = all(bool(torch.isfinite(value).all()) for value in state.values())
    if not parameters_finite:
        raise AxisScalingProbeError("trained actor contains non-finite parameters")
    return {
        "initial_actor_state_sha256": initial_sha,
        "initial_rng_state_sha256": rng_sha,
        "final_actor_state_sha256": bc.actor_state_sha256(state),
        "state": state,
        "metrics": _axis_metrics(prediction_np, action_np, metric_axis_std_np),
        "diagnostics": diagnostics,
        "parameters_finite": parameters_finite,
        "inference_output_sample": prediction_np[:2].astype(np.float32).tolist(),
    }


def _historical_gate(r0: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected = protocol["historical_reproduction_gate"]
    actual_mse = float(r0["metrics"]["overall_original_coordinate_mse"])
    checks = {
        "initial_actor_state_sha256": r0["initial_actor_state_sha256"]
        == expected["initial_actor_state_sha256"],
        "final_actor_state_sha256": r0["final_actor_state_sha256"]
        == expected["final_actor_state_sha256"],
        "final_original_coordinate_mse": abs(actual_mse - expected["final_original_coordinate_mse"])
        <= expected["absolute_mse_tolerance"],
    }
    return {"met": all(checks.values()), "checks": checks, "actual_mse": actual_mse}


def _criterion(r0: Mapping[str, Any], r1: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, Any]:
    r0_mse = float(r0["metrics"]["overall_original_coordinate_mse"])
    r1_mse = float(r1["metrics"]["overall_original_coordinate_mse"])
    constant_mse = float(baseline["overall_original_coordinate_mse"])
    axis_ratios = {
        name: float(r1["metrics"]["axis_original_coordinate_mse"][name])
        / float(baseline["axis_original_coordinate_mse"][name])
        for name in AXIS_NAMES
    }
    numeric_values = [r0_mse, r1_mse, constant_mse, float(r1["metrics"]["axis_balanced_mse"]), *axis_ratios.values()]
    checks = {
        "r1_at_most_90_percent_of_constant_mean": r1_mse <= 0.9 * constant_mse,
        "r1_lower_than_r0": r1_mse < r0_mse,
        "r1_axis_balanced_mse_at_most_0_90": float(r1["metrics"]["axis_balanced_mse"]) <= 0.90,
        "all_axis_ratios_at_most_1_10": all(value <= 1.10 for value in axis_ratios.values()),
        "all_values_finite": all(math.isfinite(value) for value in numeric_values),
    }
    return {
        "met": all(checks.values()),
        "checks": checks,
        "values": {
            "r0_original_coordinate_mse": r0_mse,
            "r1_original_coordinate_mse": r1_mse,
            "constant_train_mean_original_coordinate_mse": constant_mse,
            "r1_to_constant_mean_mse_ratio": r1_mse / constant_mse,
            "r1_axis_balanced_mse": float(r1["metrics"]["axis_balanced_mse"]),
            "r1_axis_mse_to_constant_axis_mse": axis_ratios,
        },
    }


def _save_actor(path: Path, run: Mapping[str, Any]) -> dict[str, str]:
    import torch

    torch.save(
        {
            "artifact_kind": "diagnostic_not_main_bc",
            "must_not_be_injected_into_main_experiment": True,
            "actor_state_dict": run["state"],
            "actor_state_sha256": run["final_actor_state_sha256"],
            "inference_output_coordinates": "original_normalized_6d_command",
        },
        path,
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "actor_state_sha256": str(run["final_actor_state_sha256"]),
    }


def run_probe(
    manifest_path: str | Path,
    protocol_path: str | Path,
    output_dir: str | Path,
    result_path: str | Path,
) -> dict[str, Any]:
    import numpy as np

    manifest = Path(manifest_path).resolve()
    protocol_file = Path(protocol_path).resolve()
    destination = Path(output_dir).resolve()
    public_result = Path(result_path).resolve()
    if destination.exists() or public_result.exists():
        raise FileExistsError("output directory and public result must not already exist")
    protocol = _validate_protocol(protocol_file, manifest)
    protocol_sha = file_sha256(protocol_file)
    helper = _helper()
    observation, action, _ = _load_input(manifest, helper)
    stats = protocol["source_identity"]["axis_statistics"]
    metric_axis_std = np.asarray(
        [stats["effective_std"][name] for name in AXIS_NAMES], dtype=np.float64
    )
    training_axis_std = metric_axis_std.astype(np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        r0 = _train_once(observation, action, training_axis_std, metric_axis_std, "raw_mse")
        reproduction = _historical_gate(r0, protocol)
        if not reproduction["met"]:
            raise AxisScalingProbeError("R0 did not reproduce the historical run; R1 was not started")
        r1 = _train_once(
            observation, action, training_axis_std, metric_axis_std, "axis_scaled_mse"
        )
        if r0["initial_actor_state_sha256"] != r1["initial_actor_state_sha256"]:
            raise AxisScalingProbeError("R0 and R1 initial actor states differ")
        if r0["initial_rng_state_sha256"] != r1["initial_rng_state_sha256"]:
            raise AxisScalingProbeError("R0 and R1 initial RNG states differ")
        baseline = _constant_baseline(action, metric_axis_std)
        if abs(float(baseline["axis_balanced_mse"]) - 1.0) > 1e-12:
            raise AxisScalingProbeError("constant baseline axis-balanced MSE sanity invariant failed")
        artifacts: dict[str, Any] = {}
        for run_id, run in (("r0_raw_mse", r0), ("r1_axis_scaled_mse", r1)):
            run_dir = temporary / run_id
            run_dir.mkdir()
            artifacts[run_id] = _save_actor(run_dir / "final-diagnostic-actor.pt", run)
        criterion = _criterion(r0, r1, baseline)
        result = {
            "schema_name": SCHEMA_NAME,
            "status": "completed",
            "completed_at_utc": _utc_now(),
            "protocol": {
                "path": str(protocol_file),
                "sha256": protocol_sha,
                "frozen_at_utc": protocol["frozen_at_utc"],
            },
            "source_identity": protocol["source_identity"],
            "runtime": _runtime(),
            "initial_identity": {
                "same_actor_state": r0["initial_actor_state_sha256"] == r1["initial_actor_state_sha256"],
                "same_rng_state": r0["initial_rng_state_sha256"] == r1["initial_rng_state_sha256"],
                "actor_state_sha256": r0["initial_actor_state_sha256"],
                "rng_state_sha256": r0["initial_rng_state_sha256"],
            },
            "historical_reproduction_gate": reproduction,
            "constant_train_mean_baseline": baseline,
            "runs": {
                run_id: {
                    "loss": loss_kind,
                    "metrics": run["metrics"],
                    "diagnostics": run["diagnostics"],
                    "parameters_finite": run["parameters_finite"],
                    "artifact": artifacts[run_id],
                    "inference_output_sample_original_coordinates": run["inference_output_sample"],
                }
                for run_id, loss_kind, run in (
                    ("r0_raw_mse", "raw_mse", r0),
                    ("r1_axis_scaled_mse", "axis_scaled_mse", r1),
                )
            },
            "pass_criterion": criterion,
            "claims": protocol["claims"],
            "interpretation_limit": (
                "Same-trajectory offline fit diagnostic only; it does not establish state feedback, "
                "closed-loop physics success, reinforcement-learning contribution, or held-out generalization."
            ),
            "stop_rule_observed": True,
        }
        for artifact in artifacts.values():
            artifact["path"] = str(destination / Path(artifact["path"]).relative_to(temporary))
        (temporary / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if file_sha256(protocol_file) != protocol_sha:
            raise AxisScalingProbeError("protocol changed during execution")
        if _runtime() != protocol["runtime"]:
            raise AxisScalingProbeError("runtime changed during execution")
        if _source_identity(manifest, helper, action) != protocol["source_identity"]:
            raise AxisScalingProbeError("input or source identity changed during execution")
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
    if arguments.command == "freeze":
        payload = freeze_protocol(arguments.manifest, arguments.protocol)
    else:
        payload = run_probe(arguments.manifest, arguments.protocol, arguments.output_dir, arguments.result)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
