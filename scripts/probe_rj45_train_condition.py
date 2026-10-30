#!/usr/bin/env python3
"""Fixed-condition, train-only RJ45 behavior-cloning overfit diagnostic.

The export command depends on h5py through the approved student loader.  The fit
command depends on torch through the existing BC actor module.  Imports stay
separate so the two commands can run in the repository's two local Python
environments without installing another dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Mapping, Sequence


SCHEMA_NAME = "rj45_train_condition_probe_v1"
EXPECTED_DATASET_SHA256 = "ae2884f3635d6177b278156f8cfadf1d3d29db54115cc99a938927ea3d6b1efc"
EXPECTED_GRASP_ID = "rj45-train-02"
EXPECTED_BIAS_CASE_ID = "env-0-episode-1"
EXPECTED_SEED = 66384140
FIT_SEED = 20261030
FIT_STEPS = 2000
FIT_LR = 0.001
FIT_THREADS = 2
REPORT_INTERVAL = 200
OBSERVATION_DIM = 79
ACTION_DIM = 6
AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")
LINEAGE_KEYS = ("source_sha256", "task_sha256", "training_sha256", "bank_sha256")
SHA256_LENGTH = 64


class ProbeContractError(ValueError):
    """The fixed-condition diagnostic contract was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    import numpy as np

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _valid_sha256(value):
        raise ProbeContractError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProbeContractError(f"{label} must be an object")
    return value


def _script_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def _atomic_directory(destination: Path) -> tuple[Path, Path]:
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    return destination, temporary


def _publish_directory(temporary: Path, destination: Path) -> None:
    try:
        os.rename(temporary, destination)
    except FileExistsError:
        raise FileExistsError(f"output directory already exists: {destination}") from None


def _validate_selected_arrays(observation: Any, action: Any, valid_mask: Any) -> None:
    import numpy as np

    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM or observation.shape[0] == 0:
        raise ProbeContractError("observation must have shape (N, 79), N > 0")
    if action.shape != (observation.shape[0], ACTION_DIM):
        raise ProbeContractError("action must have shape (N, 6) aligned with observation")
    if valid_mask.shape != (observation.shape[0],) or valid_mask.dtype != np.bool_:
        raise ProbeContractError("valid_mask must be boolean with shape (N,)")
    if not bool(valid_mask.any()):
        raise ProbeContractError("selected condition has no valid action rows")
    if not bool(np.isfinite(observation).all()) or not bool(np.isfinite(action).all()):
        raise ProbeContractError("observation and action must contain only finite values")
    valid_action = action[valid_mask]
    if bool((np.abs(valid_action) > 1.0 + 1e-6).any()):
        raise ProbeContractError("valid normalized BC actions exceed [-1, 1]")
    grouped = valid_action.reshape(-1, 2, 3)
    if bool((np.linalg.norm(grouped, axis=2) > 1.0 + 1e-6).any()):
        raise ProbeContractError("valid normalized BC action triplets exceed unit norm")


def export_condition(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    expected_dataset_sha256: str = EXPECTED_DATASET_SHA256,
    expected_grasp_id: str = EXPECTED_GRASP_ID,
    expected_bias_case_id: str = EXPECTED_BIAS_CASE_ID,
    expected_seed: int = EXPECTED_SEED,
) -> dict[str, Any]:
    """Export exactly one successful, valid train episode through the student loader."""

    import h5py
    import numpy as np

    from rby1_assembly.rj45_teacher_dataset import load_student_episodes

    dataset = Path(dataset_path).resolve()
    expected_digest = _require_sha256(expected_dataset_sha256, "expected dataset SHA-256")
    initial_digest = file_sha256(dataset)
    if initial_digest != expected_digest:
        raise ProbeContractError("teacher dataset SHA-256 does not match the fixed input")

    all_train = load_student_episodes(
        dataset, split="train", successful_only=False, valid_only=False
    )
    eligible = load_student_episodes(
        dataset, split="train", successful_only=True, valid_only=True
    )
    matching = [
        episode
        for episode in all_train
        if episode["metadata"]["grasp_id"] == expected_grasp_id
        and episode["metadata"]["bias_case_id"] == expected_bias_case_id
    ]
    if len(matching) != 1:
        raise ProbeContractError("fixed grasp/bias condition must identify exactly one train episode")
    episode = matching[0]
    metadata = _require_mapping(episode.get("metadata"), "episode metadata")
    expected_metadata_keys = {
        "grasp_id", "bias_case_id", "split", "seed", *LINEAGE_KEYS
    }
    if set(metadata) != expected_metadata_keys:
        raise ProbeContractError("episode metadata keys differ from the student allowlist")
    if metadata["split"] != "train":
        raise ProbeContractError("selected episode is not train-only")
    if metadata["seed"] != expected_seed:
        raise ProbeContractError("selected episode seed differs from the fixed condition")
    episode_id = str(episode["episode_id"])
    eligible_ids = {str(item["episode_id"]) for item in eligible}
    if episode_id not in eligible_ids:
        raise ProbeContractError("selected episode is not both successful and terminal-valid")
    lineage = {key: _require_sha256(metadata[key], f"metadata {key}") for key in LINEAGE_KEYS}

    observation = np.asarray(episode["observation"], dtype=np.float32)
    action = np.asarray(episode["action"], dtype=np.float32)
    valid_mask = np.asarray(episode["valid"], dtype=np.bool_)
    _validate_selected_arrays(observation, action, valid_mask)
    valid_rows = int(valid_mask.sum())

    destination, temporary = _atomic_directory(Path(output_dir))
    try:
        npz_path = temporary / "train-condition.npz"
        np.savez_compressed(
            npz_path,
            observation=observation,
            action=action,
            valid_mask=valid_mask,
        )
        array_hashes = {
            "observation": array_sha256(observation),
            "action": array_sha256(action),
            "valid_mask": array_sha256(valid_mask),
        }
        manifest = {
            "schema_name": SCHEMA_NAME,
            "artifact_kind": "train_only_fixed_condition_offline_bc_probe_input",
            "claims": {
                "train_only": True,
                "heldout_generalization": False,
                "reinforcement_learning": False,
                "closed_loop_success": False,
                "main_bc_actor_reusable": False,
            },
            "condition": {
                "episode_ids": [episode_id],
                "grasp_id": expected_grasp_id,
                "bias_case_id": expected_bias_case_id,
                "seed": expected_seed,
                "split": "train",
                "successful_and_terminal_valid": True,
            },
            "contract": {
                "observation": "standard79_without_additional_normalization",
                "action": (
                    "normalized teacher_applied 6D BC command; translation and rotation "
                    "triplets are separately unit-ball bounded, not physical micrometre errors"
                ),
                "fit_rows": "valid_mask_true_only",
            },
            "row_counts": {
                "episode_total": int(observation.shape[0]),
                "valid_action": valid_rows,
            },
            "shapes": {
                "observation": list(observation.shape),
                "action": list(action.shape),
                "valid_mask": list(valid_mask.shape),
            },
            "lineage": lineage,
            "inputs": {
                "teacher_dataset": {"path": str(dataset), "sha256": initial_digest},
                "script": {"path": str(Path(__file__).resolve()), "sha256": _script_sha256()},
            },
            "arrays": {
                name: {"sha256": digest} for name, digest in array_hashes.items()
            },
            "npz": {
                "path": str((destination / npz_path.name).resolve()),
                "sha256": file_sha256(npz_path),
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "h5py": h5py.__version__,
            },
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if file_sha256(dataset) != initial_digest:
            raise ProbeContractError("teacher dataset changed during export")
        _publish_directory(temporary, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _load_probe_input(manifest_path: Path) -> tuple[dict[str, Any], Any, Any]:
    import numpy as np

    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ProbeContractError("probe manifest is not valid JSON") from error
    expected_top_level = {
        "schema_name",
        "artifact_kind",
        "claims",
        "condition",
        "contract",
        "row_counts",
        "shapes",
        "lineage",
        "inputs",
        "arrays",
        "npz",
        "runtime",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_top_level
        or manifest.get("schema_name") != SCHEMA_NAME
        or manifest.get("artifact_kind")
        != "train_only_fixed_condition_offline_bc_probe_input"
    ):
        raise ProbeContractError("probe manifest schema is invalid")
    expected_claims = {
        "train_only": True,
        "heldout_generalization": False,
        "reinforcement_learning": False,
        "closed_loop_success": False,
        "main_bc_actor_reusable": False,
    }
    if manifest.get("claims") != expected_claims:
        raise ProbeContractError("probe manifest claims are invalid")
    expected_contract = {
        "observation": "standard79_without_additional_normalization",
        "action": (
            "normalized teacher_applied 6D BC command; translation and rotation "
            "triplets are separately unit-ball bounded, not physical micrometre errors"
        ),
        "fit_rows": "valid_mask_true_only",
    }
    if manifest.get("contract") != expected_contract:
        raise ProbeContractError("probe observation/action unit contract is invalid")
    condition = _require_mapping(manifest.get("condition"), "condition")
    if set(condition) != {
        "episode_ids",
        "grasp_id",
        "bias_case_id",
        "seed",
        "split",
        "successful_and_terminal_valid",
    } or any(
        (
            condition.get("grasp_id") != EXPECTED_GRASP_ID,
            condition.get("bias_case_id") != EXPECTED_BIAS_CASE_ID,
            condition.get("seed") != EXPECTED_SEED,
            condition.get("split") != "train",
            condition.get("successful_and_terminal_valid") is not True,
        )
    ):
        raise ProbeContractError("probe manifest condition differs from the fixed train condition")
    episode_ids = condition.get("episode_ids")
    if not isinstance(episode_ids, list) or len(episode_ids) != 1 or not isinstance(episode_ids[0], str):
        raise ProbeContractError("probe manifest must contain exactly one episode ID")
    lineage = _require_mapping(manifest.get("lineage"), "lineage")
    if set(lineage) != set(LINEAGE_KEYS):
        raise ProbeContractError("probe lineage keys are invalid")
    for key in LINEAGE_KEYS:
        _require_sha256(lineage[key], f"lineage {key}")
    inputs = _require_mapping(manifest.get("inputs"), "inputs")
    script_record = _require_mapping(inputs.get("script"), "script input")
    if _require_sha256(script_record.get("sha256"), "script SHA-256") != _script_sha256():
        raise ProbeContractError("probe script differs from the exporting script")
    dataset_record = _require_mapping(inputs.get("teacher_dataset"), "teacher dataset input")
    if _require_sha256(dataset_record.get("sha256"), "dataset SHA-256") != EXPECTED_DATASET_SHA256:
        raise ProbeContractError("probe manifest teacher dataset SHA-256 is not approved")
    npz_record = _require_mapping(manifest.get("npz"), "npz")
    npz_path = Path(str(npz_record.get("path"))).resolve()
    if file_sha256(npz_path) != _require_sha256(npz_record.get("sha256"), "NPZ SHA-256"):
        raise ProbeContractError("probe NPZ SHA-256 does not match")
    with np.load(npz_path, allow_pickle=False) as payload:
        if set(payload.files) != {"observation", "action", "valid_mask"}:
            raise ProbeContractError("probe NPZ keys are invalid")
        observation = np.asarray(payload["observation"], dtype=np.float32)
        action = np.asarray(payload["action"], dtype=np.float32)
        valid_mask = np.asarray(payload["valid_mask"])
    _validate_selected_arrays(observation, action, valid_mask)
    arrays = _require_mapping(manifest.get("arrays"), "arrays")
    if set(arrays) != {"observation", "action", "valid_mask"}:
        raise ProbeContractError("probe array hash keys are invalid")
    for name, value in (
        ("observation", observation), ("action", action), ("valid_mask", valid_mask)
    ):
        record = _require_mapping(arrays[name], f"array {name}")
        if array_sha256(value) != _require_sha256(record.get("sha256"), f"array {name} SHA-256"):
            raise ProbeContractError(f"probe {name} array SHA-256 does not match")
    rows = _require_mapping(manifest.get("row_counts"), "row_counts")
    if rows.get("episode_total") != observation.shape[0] or rows.get("valid_action") != int(valid_mask.sum()):
        raise ProbeContractError("probe row counts do not match the NPZ")
    shapes = _require_mapping(manifest.get("shapes"), "shapes")
    if shapes != {
        "observation": list(observation.shape),
        "action": list(action.shape),
        "valid_mask": list(valid_mask.shape),
    }:
        raise ProbeContractError("probe shapes do not match the NPZ")
    return manifest, observation[valid_mask], action[valid_mask]


def _metric_payload(prediction: Any, target: Any) -> dict[str, Any]:
    import numpy as np

    error = prediction - target
    axis_mse = np.mean(np.square(error), axis=0, dtype=np.float64)
    return {
        "overall_normalized_mse": float(np.mean(np.square(error), dtype=np.float64)),
        "axis_normalized_mse": {
            name: float(value) for name, value in zip(AXIS_NAMES, axis_mse, strict=True)
        },
        "axis_normalized_rmse": {
            name: float(math.sqrt(value))
            for name, value in zip(AXIS_NAMES, axis_mse, strict=True)
        },
    }


def _constant_train_mean_action_baseline(target: Any) -> dict[str, Any]:
    """Evaluate the post-hoc constant predictor formed from this train trajectory."""

    import numpy as np

    target64 = np.asarray(target, dtype=np.float64)
    if target64.ndim != 2 or target64.shape[0] == 0 or target64.shape[1] != ACTION_DIM:
        raise ProbeContractError("constant baseline target must have shape (N, 6), N > 0")
    if not bool(np.isfinite(target64).all()):
        raise ProbeContractError("constant baseline target must contain only finite values")
    mean_action = np.mean(target64, axis=0, dtype=np.float64)
    prediction = np.broadcast_to(mean_action, target64.shape)
    return {
        "predictor": "per_axis_train_mean_action_repeated_for_every_row",
        "mean_normalized_action": {
            name: float(value) for name, value in zip(AXIS_NAMES, mean_action, strict=True)
        },
        **_metric_payload(prediction, target64),
    }


def fit_condition(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    steps: int = FIT_STEPS,
    seed: int = FIT_SEED,
    learning_rate: float = FIT_LR,
    threads: int = FIT_THREADS,
    report_interval: int = REPORT_INTERVAL,
) -> dict[str, Any]:
    """Fit the final actor on the same fixed trajectory; no checkpoint selection."""

    import numpy as np
    import torch

    from rby1_assembly.rj45_bc import actor_state_sha256, build_bc_actor

    if isinstance(steps, bool) or steps <= 0:
        raise ProbeContractError("fit steps must be positive")
    if isinstance(seed, bool) or seed < 0:
        raise ProbeContractError("fit seed must be non-negative")
    if isinstance(threads, bool) or threads <= 0:
        raise ProbeContractError("fit threads must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ProbeContractError("fit learning rate must be finite and positive")
    if isinstance(report_interval, bool) or report_interval <= 0:
        raise ProbeContractError("report interval must be positive")

    source_manifest_path = Path(manifest_path).resolve()
    source_manifest_sha256 = file_sha256(source_manifest_path)
    manifest, observation_np, action_np = _load_probe_input(source_manifest_path)
    if observation_np.shape[0] == 0:
        raise ProbeContractError("probe input has no valid training rows")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    observation = torch.as_tensor(observation_np, dtype=torch.float32, device="cpu")
    action = torch.as_tensor(action_np, dtype=torch.float32, device="cpu")
    actor = build_bc_actor().cpu()
    with torch.no_grad():
        actor[-1].weight.zero_()
        actor[-1].bias.zero_()
        initial_prediction = actor(observation)
    if not bool(torch.equal(initial_prediction, torch.zeros_like(initial_prediction))):
        raise ProbeContractError("zero-output baseline initialization failed")
    initial_metrics = _metric_payload(initial_prediction.numpy(), action.numpy())
    initial_mse = initial_metrics["overall_normalized_mse"]
    if not math.isfinite(initial_mse) or initial_mse <= 0.0:
        raise ProbeContractError("zero-output baseline MSE must be finite and positive")
    constant_baseline = _constant_train_mean_action_baseline(action.numpy())
    constant_mse = constant_baseline["overall_normalized_mse"]
    if not math.isfinite(constant_mse) or constant_mse <= 0.0:
        raise ProbeContractError("constant train-mean baseline MSE must be finite and positive")

    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    diagnostics: list[dict[str, float | int]] = []
    actor.train()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = actor(observation)
        loss = torch.nn.functional.mse_loss(prediction, action)
        if not bool(torch.isfinite(loss)):
            raise ProbeContractError("fit loss became non-finite")
        loss.backward()
        optimizer.step()
        if step % report_interval == 0 or step == steps:
            actor.eval()
            with torch.no_grad():
                diagnostic_mse = float(torch.nn.functional.mse_loss(actor(observation), action))
            if not math.isfinite(diagnostic_mse):
                raise ProbeContractError("diagnostic fit MSE became non-finite")
            diagnostics.append({"optimizer_step": step, "train_normalized_mse": diagnostic_mse})
            actor.train()

    actor.eval()
    with torch.no_grad():
        final_prediction = actor(observation).numpy()
    final_metrics = _metric_payload(final_prediction, action.numpy())
    final_mse = final_metrics["overall_normalized_mse"]
    ratio = final_mse / initial_mse
    final_to_constant_ratio = final_mse / constant_mse
    final_to_constant_axis_ratios = {
        name: (
            final_metrics["axis_normalized_mse"][name]
            / constant_baseline["axis_normalized_mse"][name]
        )
        for name in AXIS_NAMES
    }
    parameters_finite = all(bool(torch.isfinite(parameter).all()) for parameter in actor.parameters())
    criterion_met = parameters_finite and math.isfinite(final_mse) and ratio <= 0.01

    destination, temporary = _atomic_directory(Path(output_dir))
    try:
        actor_path = temporary / "final-diagnostic-actor.pt"
        state = {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
        state_digest = actor_state_sha256(state)
        torch.save(
            {
                "artifact_kind": "offline_same_trajectory_overfit_diagnostic_not_main_bc_actor",
                "actor_state_dict": state,
                "actor_state_sha256": state_digest,
            },
            actor_path,
        )
        result = {
            "schema_name": SCHEMA_NAME,
            "status": "completed",
            "artifact_kind": "train_only_fixed_condition_offline_bc_overfit_result",
            "claims": {
                "same_trajectory_regression_check": True,
                "heldout_generalization": False,
                "reinforcement_learning": False,
                "closed_loop_success": False,
                "physics_learning_progress_gate": False,
                "main_bc_actor_reusable": False,
            },
            "interpretation": (
                "This only tests whether the existing BC actor can regress one fixed train "
                "trajectory in normalized action coordinates. It is not held-out, RL, or "
                "closed-loop evidence and does not replace the physics-learning gate. The "
                "zero-relative pre-registered fit criterion alone does not demonstrate "
                "state-dependent trajectory tracking because a constant train-mean action "
                "predictor nearly matches the final actor."
            ),
            "recipe": {
                "device": "cpu",
                "seed": seed,
                "threads": threads,
                "optimizer": "Adam",
                "learning_rate": learning_rate,
                "full_batch_optimizer_steps": steps,
                "selection": "final_actor_only_no_intermediate_checkpoint_selection",
                "final_layer_initialization": "zero_weight_and_bias",
                "report_interval_steps": report_interval,
            },
            "input": {
                "manifest_path": str(source_manifest_path),
                "manifest_sha256": source_manifest_sha256,
                "episode_ids": manifest["condition"]["episode_ids"],
                "valid_rows": int(observation.shape[0]),
                "lineage": manifest["lineage"],
            },
            "metrics": {
                "initial_zero_output": initial_metrics,
                "constant_train_mean_action": constant_baseline,
                "final_actor": final_metrics,
                "final_to_initial_mse_ratio": ratio,
                "final_to_constant_mean_mse_ratio": final_to_constant_ratio,
                "final_to_constant_mean_axis_mse_ratios": final_to_constant_axis_ratios,
                "parameters_finite": parameters_finite,
            },
            "pre_registered_offline_fit_criterion": {
                "definition": "final normalized MSE <= 1% of initial zero-output MSE and finite",
                "met": criterion_met,
            },
            "post_hoc_constant_baseline_diagnostic": {
                "added_after_initial_zero_relative_result": True,
                "pre_registered_pass_criterion": False,
                "changes_original_fit_criterion_or_result": False,
                "purpose": (
                    "Check whether the zero-relative improvement could be explained by "
                    "learning the fixed trajectory's mean normalized action."
                ),
                "conclusion": (
                    "The zero-relative PASS is insufficient evidence of state-dependent "
                    "tracking; no new retrospective pass threshold is defined."
                ),
            },
            "diagnostics": diagnostics,
            "artifact": {
                "path": str((destination / actor_path.name).resolve()),
                "sha256": file_sha256(actor_path),
                "actor_state_sha256": state_digest,
            },
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
        }
        (temporary / "fit-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if file_sha256(source_manifest_path) != source_manifest_sha256:
            raise ProbeContractError("probe manifest changed during fit")
        _publish_directory(temporary, destination)
        return result
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="export one fixed train condition to NPZ")
    export.add_argument("--dataset", required=True, type=Path)
    export.add_argument("--output-dir", required=True, type=Path)
    export.add_argument("--expected-sha256", default=EXPECTED_DATASET_SHA256)
    fit = commands.add_parser("fit", help="run the fixed 2000-step CPU overfit probe")
    fit.add_argument("--manifest", required=True, type=Path)
    fit.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "export":
        payload = export_condition(
            arguments.dataset,
            arguments.output_dir,
            expected_dataset_sha256=arguments.expected_sha256,
        )
    else:
        payload = fit_condition(arguments.manifest, arguments.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
