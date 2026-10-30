#!/usr/bin/env python3
"""Train-only, offline baselines for the frozen global RJ45 BC actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform

import h5py
import numpy as np
import torch

from rby1_assembly.rj45_bc import (
    build_bc_actor,
    load_bc_actor_artifact,
    prepare_train_samples,
)
from rby1_assembly.rj45_teacher_dataset import load_student_episodes


EXPECTED_DATASET_SHA256 = "ae2884f3635d6177b278156f8cfadf1d3d29db54115cc99a938927ea3d6b1efc"
EXPECTED_ACTOR_SHA256 = "f0fd65b56691748dce5bee378152524bf84f3f9daed2f1f36dd43e0e189233b5"
AXES = ("tx", "ty", "tz", "rx", "ry", "rz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deployment_projection(action: np.ndarray) -> np.ndarray:
    clipped = np.clip(action, -1.0, 1.0).reshape(-1, 2, 3)
    norms = np.maximum(np.linalg.norm(clipped, axis=2, keepdims=True), 1.0)
    return (clipped / norms).reshape(-1, 6)


def metrics(prediction: np.ndarray, label: np.ndarray, zero_mse: float) -> dict[str, object]:
    error = np.asarray(prediction, dtype=np.float64) - np.asarray(label, dtype=np.float64)
    per_axis = np.mean(error * error, axis=0)
    mse = float(np.mean(per_axis))
    return {
        "normalized_action_mse": mse,
        "normalized_action_mse_per_axis": dict(zip(AXES, map(float, per_axis), strict=True)),
        "mse_as_fraction_of_zero": mse / zero_mse,
        "mse_reduction_vs_zero_fraction": 1.0 - mse / zero_mse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset_path = args.train_dataset.resolve()
    actor_path = args.actor.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if sha256(dataset_path) != EXPECTED_DATASET_SHA256:
        raise ValueError("frozen train dataset SHA-256 differs")
    if sha256(actor_path) != EXPECTED_ACTOR_SHA256:
        raise ValueError("frozen global BC actor SHA-256 differs")

    prepared = prepare_train_samples(dataset_path, EXPECTED_DATASET_SHA256)
    observations = prepared.observation.numpy()
    labels = prepared.action.numpy()
    episodes = load_student_episodes(
        dataset_path, split="train", successful_only=True, valid_only=False
    )
    if len(episodes) != 256 or len(observations) != 95826:
        raise ValueError("frozen train episode/row count differs")

    episode_mean_predictions: list[np.ndarray] = []
    episode_ids: list[str] = []
    episode_seeds: list[int] = []
    eligible_counts: list[int] = []
    previous_alignment_errors: list[np.ndarray] = []
    previous_alignment_rows = 0
    observation_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    lineage_rows: set[tuple[str, str, str, str]] = set()
    for episode in episodes:
        observation_all = np.asarray(episode["observation"], dtype=np.float32)
        action_all = np.asarray(episode["action"], dtype=np.float32)
        valid = np.asarray(episode["valid"], dtype=np.bool_)
        observation = observation_all[valid]
        action = action_all[valid]
        if not len(action):
            raise ValueError("eligible train episode has no valid rows")
        observation_parts.append(observation)
        label_parts.append(action)
        episode_mean_predictions.append(np.broadcast_to(action.mean(axis=0), action.shape))
        episode_ids.append(str(episode["episode_id"]))
        metadata = episode["metadata"]
        episode_seeds.append(int(metadata["seed"]))
        eligible_counts.append(int(valid.sum()))
        lineage_rows.add(
            tuple(
                str(metadata[key])
                for key in ("source_sha256", "task_sha256", "training_sha256", "bank_sha256")
            )
        )

        prior_action = np.vstack(
            (np.zeros((1, 6), dtype=np.float32), action_all[:-1])
        )
        previous_alignment_errors.append(
            np.abs(observation_all[valid, 56:62] - prior_action[valid])
        )
        previous_alignment_rows += int(valid.sum())

    loader_observations = np.concatenate(observation_parts)
    loader_labels = np.concatenate(label_parts)
    if not np.array_equal(loader_observations, observations) or not np.array_equal(
        loader_labels, labels
    ):
        raise ValueError("prepared training samples differ from grouped student-loader rows")
    if len(lineage_rows) != 1:
        raise ValueError("eligible train episodes do not share one lineage")

    artifact = load_bc_actor_artifact(actor_path, expected_sha256=EXPECTED_ACTOR_SHA256)
    if artifact["provenance"] != prepared.provenance:
        raise ValueError("frozen actor and train dataset provenance differ")
    actor = build_bc_actor()
    actor.load_state_dict(artifact["actor_state_dict"], strict=True)
    actor.eval()
    with torch.inference_mode():
        actor_raw = actor(torch.from_numpy(observations)).numpy()
    actor_deployment = deployment_projection(actor_raw)

    zeros = np.zeros_like(labels)
    global_mean = np.broadcast_to(labels.mean(axis=0), labels.shape)
    episode_mean = np.concatenate(episode_mean_predictions)
    previous_action = observations[:, 56:62]
    zero_mse = float(np.mean(np.asarray(labels, dtype=np.float64) ** 2))
    baselines = {
        "zero": metrics(zeros, labels, zero_mse),
        "train_global_mean_fixed_action": metrics(global_mean, labels, zero_mse),
        "per_episode_mean_fixed_action_oracle": metrics(episode_mean, labels, zero_mse),
        "previous_right_action_teacher_forced": metrics(previous_action, labels, zero_mse),
        "frozen_global_bc_actor_raw": metrics(actor_raw, labels, zero_mse),
        "frozen_global_bc_actor_deployment_bounded": metrics(
            actor_deployment, labels, zero_mse
        ),
    }
    global_mean_mse = float(
        baselines["train_global_mean_fixed_action"]["normalized_action_mse"]
    )
    for value in baselines.values():
        mse = float(value["normalized_action_mse"])
        value["mse_as_fraction_of_global_mean"] = mse / global_mean_mse
        value["mse_reduction_vs_global_mean_fraction"] = 1.0 - mse / global_mean_mse

    actor_axis_mse = baselines["frozen_global_bc_actor_raw"][
        "normalized_action_mse_per_axis"
    ]
    previous_axis_mse = baselines["previous_right_action_teacher_forced"][
        "normalized_action_mse_per_axis"
    ]
    actor_mse = float(baselines["frozen_global_bc_actor_raw"]["normalized_action_mse"])
    previous_mse = float(
        baselines["previous_right_action_teacher_forced"]["normalized_action_mse"]
    )
    oracle_episode_mean_mse = float(
        baselines["per_episode_mean_fixed_action_oracle"]["normalized_action_mse"]
    )
    pairwise = {
        "global_bc_mse_reduction_vs_previous_action_fraction": 1.0
        - actor_mse / previous_mse,
        "global_bc_mse_as_fraction_of_previous_action": actor_mse / previous_mse,
        "global_bc_mse_as_multiple_of_oracle_episode_mean": actor_mse
        / oracle_episode_mean_mse,
        "previous_action_mse_as_multiple_of_oracle_episode_mean": previous_mse
        / oracle_episode_mean_mse,
        "global_bc_mse_reduction_vs_previous_action_per_axis_fraction": {
            axis: 1.0 - float(actor_axis_mse[axis]) / float(previous_axis_mse[axis])
            for axis in AXES
        },
    }

    label_triplet_norm = np.linalg.norm(labels.reshape(-1, 2, 3), axis=2)
    actor_clipped = np.clip(actor_raw, -1.0, 1.0)
    actor_triplet_norm_after_component_clip = np.linalg.norm(
        actor_clipped.reshape(-1, 2, 3), axis=2
    )
    alignment = np.concatenate(previous_alignment_errors)
    source, task, training, bank = next(iter(lineage_rows))
    result = {
        "schema_version": 1,
        "scope": "post-hoc offline train-only normalized-action regression diagnostic",
        "claims": {
            "closed_loop_performance": False,
            "heldout_generalization": False,
            "reinforcement_learning": False,
            "shortcut_causality": False,
            "dev_or_final_opened": False,
        },
        "inputs": {
            "teacher_train_h5": {"path": str(dataset_path), "sha256": sha256(dataset_path)},
            "global_bc_actor": {
                "path": str(actor_path),
                "file_sha256": sha256(actor_path),
                "actor_state_sha256": artifact["actor_state_sha256"],
            },
            "lineage": {
                "source_sha256": source,
                "task_sha256": task,
                "training_sha256": training,
                "bank_sha256": bank,
                "dataset_sha256": prepared.provenance.dataset_sha256,
            },
        },
        "selection": {
            "split": "train",
            "successful_only": True,
            "terminal_valid_only": True,
            "row_valid_only": True,
            "episode_count": len(episodes),
            "eligible_row_count": len(labels),
            "episode_ids": episode_ids,
            "episode_seed_min": min(episode_seeds),
            "episode_seed_max": max(episode_seeds),
            "episode_seed_unique_count": len(set(episode_seeds)),
            "episode_eligible_row_count_min": min(eligible_counts),
            "episode_eligible_row_count_max": max(eligible_counts),
        },
        "action_contract": {
            "label": "teacher_applied normalized dimensionless 6D action",
            "axis_order": list(AXES),
            "translation_and_rotation_triplets_separately_unit_ball_bounded": True,
            "teacher_label_max_abs_component": float(np.abs(labels).max()),
            "teacher_label_max_triplet_norm": float(label_triplet_norm.max()),
            "teacher_label_triplet_norm_over_one_rows": int(
                np.any(label_triplet_norm > 1.0 + 1.0e-6, axis=1).sum()
            ),
            "actor_raw_max_abs_component": float(np.abs(actor_raw).max()),
            "actor_raw_component_clip_row_count": int(
                np.any(np.abs(actor_raw) > 1.0, axis=1).sum()
            ),
            "actor_raw_norm_projection_row_count": int(
                np.any(actor_triplet_norm_after_component_clip > 1.0, axis=1).sum()
            ),
            "actor_raw_to_deployment_bounded_rms": float(
                np.sqrt(np.mean((actor_raw - actor_deployment) ** 2))
            ),
            "deployment_projection": (
                "component clip to [-1,1], then separate translation/rotation 3D unit-ball projection"
            ),
        },
        "previous_action_timing_check": {
            "feature_indices_half_open": [56, 62],
            "meaning": "prior control step's applied right 6D action at the current pre-step observation",
            "checked_valid_rows": previous_alignment_rows,
            "max_abs_difference_from_immediately_prior_episode_row_teacher_applied": float(
                alignment.max()
            ),
            "mismatch_rows_at_1e_6": int(np.any(alignment > 1.0e-6, axis=1).sum()),
            "interpretation_limit": (
                "teacher-forced previous action is an offline baseline; low error does not establish a causal shortcut"
            ),
        },
        "metrics": {
            "definition": "mean squared error in the dataset's dimensionless normalized 6D action coordinates",
            "baselines": baselines,
            "pairwise": pairwise,
        },
        "limitations": [
            "Per-episode mean uses the episode identity and all labels from that episode, so it is an oracle condition mean rather than a deployable policy.",
            "Every comparison uses frozen train rows; no dev or final split is opened and no held-out claim is supported.",
            "Previous-action input is teacher-forced from the recorded trajectory and cannot establish shortcut causality.",
            "Offline action MSE does not establish closed-loop success, stability, recovery, or RL contribution.",
        ],
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "h5py": h5py.__version__,
            "torch": torch.__version__,
            "torch_cpu_threads": torch.get_num_threads(),
        },
        "producer": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "implementation_sha256": {
                name: sha256(Path(__file__).resolve().parents[1] / "src/rby1_assembly" / name)
                for name in ("rj45_bc.py", "rj45_teacher_dataset.py")
            },
        },
    }
    if sha256(dataset_path) != EXPECTED_DATASET_SHA256 or sha256(actor_path) != EXPECTED_ACTOR_SHA256:
        raise ValueError("frozen analysis inputs changed during execution")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "sha256": sha256(output), "baselines": baselines}))


if __name__ == "__main__":
    main()
