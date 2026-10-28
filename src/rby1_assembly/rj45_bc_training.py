"""Deterministic CPU behavior-cloning training for the RJ45 teacher datasets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping

import torch

from .rj45_bc import (
    BcContractError,
    build_bc_actor,
    file_sha256,
    prepare_train_samples,
    save_bc_actor_artifact,
    train_bc_batch,
)
from .rj45_teacher_dataset import load_student_episodes


FINAL_HOLDOUT_SEED = 20261027
LINEAGE_KEYS = ("source_sha256", "task_sha256", "training_sha256", "bank_sha256")


@dataclass(frozen=True)
class BcRecipe:
    epochs: int
    batch_size: int
    learning_rate: float
    seed: int
    threads: int

    @classmethod
    def load(cls, path: str | Path) -> "BcRecipe":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {"schema_version", "epochs", "batch_size", "learning_rate", "seed", "threads"}
        if not isinstance(payload, dict) or set(payload) != expected or payload["schema_version"] != 1:
            raise BcContractError("BC recipe schema is invalid")
        for name in ("epochs", "batch_size", "seed", "threads"):
            if isinstance(payload[name], bool) or not isinstance(payload[name], int):
                raise BcContractError(f"recipe {name} must be an integer")
        if payload["epochs"] <= 0 or payload["batch_size"] <= 0 or payload["threads"] <= 0:
            raise BcContractError("recipe epochs, batch_size, and threads must be positive")
        if payload["seed"] < 0 or payload["seed"] == FINAL_HOLDOUT_SEED:
            raise BcContractError("recipe seed is invalid or opens the final holdout seed")
        learning_rate = payload["learning_rate"]
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, (int, float)):
            raise BcContractError("recipe learning_rate must be numeric")
        if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0.0:
            raise BcContractError("recipe learning_rate must be finite and positive")
        return cls(
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            learning_rate=float(learning_rate),
            seed=payload["seed"],
            threads=payload["threads"],
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
            "threads": self.threads,
        }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(episodes: list[dict[str, object]], split: str) -> tuple[set[str], set[tuple[str, int]], dict[str, str]]:
    grasps: set[str] = set()
    bias_seeds: set[tuple[str, int]] = set()
    lineage: dict[str, str] | None = None
    for episode in episodes:
        metadata = episode.get("metadata")
        if not isinstance(metadata, Mapping) or metadata.get("split") != split:
            raise BcContractError(f"{split} dataset metadata is invalid")
        if metadata.get("seed") == FINAL_HOLDOUT_SEED:
            raise BcContractError("teacher dataset opens the final holdout seed")
        grasps.add(str(metadata["grasp_id"]))
        bias_seeds.add((str(metadata["bias_case_id"]), int(metadata["seed"])))
        current = {name: str(metadata[name]) for name in LINEAGE_KEYS}
        if lineage is None:
            lineage = current
        elif current != lineage:
            raise BcContractError(f"{split} dataset has mixed provenance lineage")
    if lineage is None:
        raise BcContractError(f"{split} dataset has no episodes")
    return grasps, bias_seeds, lineage


def _eligible_rows(path: Path, split: str) -> tuple[torch.Tensor, torch.Tensor, list[str], list[dict[str, object]]]:
    all_episodes = load_student_episodes(path, split=split, successful_only=False, valid_only=False)
    eligible = load_student_episodes(path, split=split, successful_only=True, valid_only=True)
    if not eligible:
        raise BcContractError(f"{split} dataset has no successful valid rows")
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    row_episode_ids: list[str] = []
    for episode in eligible:
        observation = torch.as_tensor(episode["observation"], dtype=torch.float32)
        action = torch.as_tensor(episode["action"], dtype=torch.float32)
        observations.append(observation)
        actions.append(action)
        row_episode_ids.extend([str(episode["episode_id"])] * observation.shape[0])
    return torch.cat(observations), torch.cat(actions), row_episode_ids, all_episodes


def _mse(actor: torch.nn.Module, observation: torch.Tensor, action: torch.Tensor) -> float:
    actor.eval()
    with torch.no_grad():
        value = torch.nn.functional.mse_loss(actor(observation), action)
    if not bool(torch.isfinite(value)):
        raise BcContractError("BC evaluation MSE is non-finite")
    return float(value)


def observability_diagnostic(
    observation: torch.Tensor,
    action: torch.Tensor,
    row_episode_ids: list[str],
    *,
    seed: int,
    limit: int = 1024,
) -> dict[str, Any]:
    count = min(limit, observation.shape[0])
    generator = torch.Generator(device="cpu").manual_seed(seed)
    indices = torch.randperm(observation.shape[0], generator=generator)[:count]
    observation_distances: list[float] = []
    label_distances: list[float] = []
    skipped = 0
    for row_index in indices.tolist():
        candidates = torch.tensor(
            [episode_id != row_episode_ids[row_index] for episode_id in row_episode_ids],
            dtype=torch.bool,
        )
        if not bool(candidates.any()):
            skipped += 1
            continue
        delta = observation[candidates] - observation[row_index]
        distances = torch.sqrt(torch.mean(delta * delta, dim=1))
        candidate_indices = torch.nonzero(candidates, as_tuple=False).flatten()
        nearest = int(candidate_indices[int(torch.argmin(distances))])
        label_delta = action[nearest] - action[row_index]
        observation_distances.append(float(torch.min(distances)))
        label_distances.append(float(torch.sqrt(torch.mean(label_delta * label_delta))))
    return {
        "scope": "fixed-seed sample of at most 1024 eligible train rows; nearest row must be from another episode",
        "interpretation": "label collision is an observability diagnostic, not a proof of impossibility",
        "seed": seed,
        "requested_rows": count,
        "evaluated_rows": len(observation_distances),
        "skipped_without_other_episode": skipped,
        "mean_nearest_observation_rms": sum(observation_distances) / len(observation_distances) if observation_distances else None,
        "mean_corresponding_label_rms": sum(label_distances) / len(label_distances) if label_distances else None,
        "max_corresponding_label_rms": max(label_distances) if label_distances else None,
    }


def train_bc(
    *,
    train_dataset: Path,
    train_sha256: str,
    dev_dataset: Path,
    dev_sha256: str,
    recipe_path: Path,
    output_dir: Path,
    enable_observability_diagnostic: bool = False,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    recipe_sha256 = _sha256(recipe_path)
    recipe = BcRecipe.load(recipe_path)
    if file_sha256(train_dataset) != train_sha256 or file_sha256(dev_dataset) != dev_sha256:
        raise BcContractError("teacher dataset SHA-256 does not match")
    prepared = prepare_train_samples(train_dataset, train_sha256)
    train_obs, train_action, row_episode_ids, train_all = _eligible_rows(train_dataset, "train")
    dev_obs, dev_action, _dev_row_ids, dev_all = _eligible_rows(dev_dataset, "dev")
    if not torch.equal(prepared.observation, train_obs) or not torch.equal(prepared.action, train_action):
        raise BcContractError("train sample preparation differs from verified student loader")
    train_grasps, train_bias_seeds, train_lineage = _metadata(train_all, "train")
    dev_grasps, dev_bias_seeds, dev_lineage = _metadata(dev_all, "dev")
    if train_grasps & dev_grasps:
        raise BcContractError("grasp_id crosses train/dev dataset files")
    if train_bias_seeds & dev_bias_seeds:
        raise BcContractError("bias_case_id/seed crosses train/dev dataset files")
    if train_lineage != dev_lineage:
        raise BcContractError("train/dev source, task, training, or bank lineage differs")
    prepared_lineage = prepared.provenance.as_dict()
    if prepared_lineage["dataset_sha256"] != train_sha256 or any(
        prepared_lineage[name] != value for name, value in train_lineage.items()
    ):
        raise BcContractError("prepared train provenance differs from validated dataset lineage")

    torch.set_num_threads(recipe.threads)
    torch.manual_seed(recipe.seed)
    torch.use_deterministic_algorithms(True)
    actor = build_bc_actor().cpu()
    with torch.no_grad():
        actor[-1].weight.zero_()
        actor[-1].bias.zero_()
    optimizer = torch.optim.Adam(actor.parameters(), lr=recipe.learning_rate)
    initial_dev_mse = _mse(actor, dev_obs, dev_action)
    best_dev_mse = math.inf
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_metrics: list[dict[str, Any]] = []
    updates = 0
    started = time.monotonic()
    for epoch in range(1, recipe.epochs + 1):
        generator = torch.Generator(device="cpu").manual_seed(recipe.seed + epoch)
        permutation = torch.randperm(train_obs.shape[0], generator=generator)
        batch_losses: list[float] = []
        for start in range(0, train_obs.shape[0], recipe.batch_size):
            rows = permutation[start : start + recipe.batch_size]
            result = train_bc_batch(actor, train_obs[rows], train_action[rows], optimizer)
            batch_losses.append(result["loss_before"])
            updates += 1
        train_mse = _mse(actor, train_obs, train_action)
        dev_mse = _mse(actor, dev_obs, dev_action)
        epoch_metrics.append({
            "epoch": epoch,
            "optimizer_updates": len(batch_losses),
            "mean_batch_loss_before": sum(batch_losses) / len(batch_losses),
            "train_mse": train_mse,
            "dev_mse": dev_mse,
        })
        if dev_mse < best_dev_mse:
            best_dev_mse = dev_mse
            best_epoch = epoch
            best_state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
    elapsed_s = time.monotonic() - started
    if best_state is None or best_epoch < 1 or not math.isfinite(best_dev_mse):
        raise AssertionError("no BC epoch was eligible for dev selection")
    actor.load_state_dict(best_state, strict=True)

    if (
        file_sha256(train_dataset) != train_sha256
        or file_sha256(dev_dataset) != dev_sha256
        or _sha256(recipe_path) != recipe_sha256
    ):
        raise BcContractError("BC inputs changed during training")
    output_dir.mkdir(parents=True, exist_ok=False)
    artifact_path = output_dir / "bc_actor_init.pt"
    artifact = save_bc_actor_artifact(artifact_path, actor, prepared.provenance)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "artifact_kind": "bc_actor_initialization_not_ppo_checkpoint",
        "student_observation": "standard79_without_additional_normalization",
        "selection": "lowest_dev_mse_with_earliest_epoch_tie_break",
        "recipe": recipe.as_dict(),
        "inputs": {
            "train_dataset": {"path": str(train_dataset), "sha256": train_sha256},
            "dev_dataset": {"path": str(dev_dataset), "sha256": dev_sha256},
            "recipe": {"path": str(recipe_path), "sha256": recipe_sha256},
        },
        "lineage": train_lineage,
        "row_counts": {
            "train_eligible": train_obs.shape[0],
            "dev_eligible": dev_obs.shape[0],
            "train_episodes_all": len(train_all),
            "dev_episodes_all": len(dev_all),
        },
        "initial_dev_mse": initial_dev_mse,
        "best_dev_mse": best_dev_mse,
        "best_epoch": best_epoch,
        "epoch_metrics": epoch_metrics,
        "total_bc_updates": updates,
        "sample_visits": train_obs.shape[0] * recipe.epochs,
        "elapsed_s": elapsed_s,
        "artifact": artifact,
        "observability_diagnostic": (
            observability_diagnostic(train_obs, train_action, row_episode_ids, seed=recipe.seed)
            if enable_observability_diagnostic
            else None
        ),
        "claims": {
            "ppo_checkpoint": False,
            "critic_or_std_trained": False,
            "privileged_teacher_ground_truth_used_as_student_input": False,
        },
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
