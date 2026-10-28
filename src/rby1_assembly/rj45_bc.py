"""CPU behavior cloning actor and actor-only RSL-RL initialization artifact."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping
import uuid

import torch
from torch import nn


OBSERVATION_DIM = 79
ACTION_DIM = 6
HIDDEN_DIMS = (256, 256, 128)
SCHEMA_NAME = "rj45_bc_actor_init_v1"
ARTIFACT_KIND = "bc_actor_initialization_not_ppo_checkpoint"
RSL_RL_VERSION = "3.0.1"
RSL_RL_SOURCE_COMMIT = "2fc1f78bc1d796ffa8f07ce6b09898227db284bb"
EXPECTED_ACTOR_KEYS = (
    "0.weight",
    "0.bias",
    "2.weight",
    "2.bias",
    "4.weight",
    "4.bias",
    "6.weight",
    "6.bias",
)
TEACHER_METADATA_KEYS = frozenset(
    {
        "grasp_id",
        "bias_case_id",
        "split",
        "seed",
        "source_sha256",
        "task_sha256",
        "training_sha256",
        "bank_sha256",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_KEYS = frozenset(
    {"schema_name", "artifact_kind", "architecture", "provenance", "actor_state_dict", "actor_state_sha256"}
)


class BcContractError(ValueError):
    """BC 학습 또는 actor 초기화 계약 위반."""


@dataclass(frozen=True)
class BcProvenance:
    source_sha256: str
    task_sha256: str
    training_sha256: str
    bank_sha256: str
    dataset_sha256: str

    def validate(self) -> None:
        for name, value in self.as_dict().items():
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise BcContractError(f"{name} must be a lowercase SHA-256 digest")

    def as_dict(self) -> dict[str, str]:
        return {
            "source_sha256": self.source_sha256,
            "task_sha256": self.task_sha256,
            "training_sha256": self.training_sha256,
            "bank_sha256": self.bank_sha256,
            "dataset_sha256": self.dataset_sha256,
        }


@dataclass(frozen=True)
class PreparedTrainingSamples:
    observation: torch.Tensor
    action: torch.Tensor
    episode_ids: tuple[str, ...]
    provenance: BcProvenance


def build_bc_actor() -> nn.Sequential:
    """RSL-RL 3.0.1 MLP와 같은 module 순서와 state_dict key를 만든다."""

    return nn.Sequential(
        nn.Linear(OBSERVATION_DIM, 256),
        nn.ELU(),
        nn.Linear(256, 256),
        nn.ELU(),
        nn.Linear(256, 128),
        nn.ELU(),
        nn.Linear(128, ACTION_DIM),
    )


def train_bc_batch(
    actor: nn.Module,
    observation: torch.Tensor,
    action: torch.Tensor,
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """유한한 standard79/6D 한 batch로 MSE optimizer update를 한 번 수행한다."""

    _validate_runtime_actor(actor)
    _validate_batch(observation, action)
    if observation.device.type != "cpu" or action.device.type != "cpu":
        raise BcContractError("BC batch training is CPU-only")
    if any(parameter.device.type != "cpu" for parameter in actor.parameters()):
        raise BcContractError("BC actor training is CPU-only")
    _validate_optimizer_owns_actor(optimizer, actor)
    actor.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = actor(observation)
    loss = torch.nn.functional.mse_loss(prediction, action)
    if not bool(torch.isfinite(loss)):
        raise BcContractError("BC loss must be finite")
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        after = torch.nn.functional.mse_loss(actor(observation), action)
    if not bool(torch.isfinite(after)) or not _module_finite(actor):
        raise BcContractError("BC update produced non-finite values")
    return {"loss_before": float(loss.detach()), "loss_after": float(after)}


def prepare_train_samples(
    dataset_path: str | Path, expected_dataset_sha256: str
) -> PreparedTrainingSamples:
    """teacher loader를 지연 import하고 train split의 student 관측·label만 결합한다."""

    path = Path(dataset_path)
    if not _valid_digest(expected_dataset_sha256):
        raise BcContractError("expected_dataset_sha256 must be a lowercase SHA-256 digest")
    if file_sha256(path) != expected_dataset_sha256:
        raise BcContractError("teacher dataset SHA-256 does not match")
    from .rj45_teacher_dataset import load_student_episodes

    episodes = load_student_episodes(path, split="train", successful_only=True, valid_only=True)
    if not episodes:
        raise BcContractError("teacher dataset has no eligible train samples")
    observations: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    episode_ids: list[str] = []
    lineage: dict[str, str] | None = None
    for episode in episodes:
        metadata = episode.get("metadata")
        if not isinstance(metadata, Mapping) or set(metadata) != TEACHER_METADATA_KEYS:
            raise BcContractError("teacher metadata keys do not match the student allowlist")
        if metadata["split"] != "train":
            raise BcContractError("BC preparation accepts train split only")
        current = {name: metadata[name] for name in BcProvenance.__dataclass_fields__ if name != "dataset_sha256"}
        if not all(_valid_digest(value) for value in current.values()):
            raise BcContractError("teacher metadata provenance must contain SHA-256 digests")
        if lineage is None:
            lineage = current
        elif current != lineage:
            raise BcContractError("eligible teacher episodes must share one provenance lineage")
        observation = torch.as_tensor(episode["observation"], dtype=torch.float32, device="cpu")
        action = torch.as_tensor(episode["action"], dtype=torch.float32, device="cpu")
        _validate_batch(observation, action)
        observations.append(observation)
        actions.append(action)
        episode_ids.append(str(episode["episode_id"]))
    assert lineage is not None
    provenance = BcProvenance(dataset_sha256=expected_dataset_sha256, **lineage)
    provenance.validate()
    return PreparedTrainingSamples(
        observation=torch.cat(observations, dim=0),
        action=torch.cat(actions, dim=0),
        episode_ids=tuple(episode_ids),
        provenance=provenance,
    )


def save_bc_actor_artifact(
    path: str | Path, actor: nn.Module, provenance: BcProvenance
) -> dict[str, str]:
    """PPO checkpoint와 구분되는 actor-only 초기화 artifact를 원자적으로 저장한다."""

    _validate_runtime_actor(actor)
    provenance.validate()
    if not _module_finite(actor):
        raise BcContractError("actor parameters must be finite")
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: tensor.detach().cpu().clone() for name, tensor in actor.state_dict().items()}
    state_sha256 = actor_state_sha256(state)
    payload: dict[str, Any] = {
        "schema_name": SCHEMA_NAME,
        "artifact_kind": ARTIFACT_KIND,
        "architecture": _architecture(),
        "provenance": provenance.as_dict(),
        "actor_state_dict": state,
        "actor_state_sha256": state_sha256,
    }
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    torch.save(payload, temporary)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        raise FileExistsError(destination) from None
    finally:
        temporary.unlink(missing_ok=True)
    return {"path": str(destination), "sha256": file_sha256(destination), "actor_state_sha256": state_sha256}


def load_bc_actor_artifact(
    path: str | Path, *, expected_sha256: str | None = None
) -> dict[str, Any]:
    """weights_only 모드로 actor artifact를 읽고 schema·lineage·weights를 검증한다."""

    source = Path(path)
    artifact_bytes = source.read_bytes()
    file_digest = hashlib.sha256(artifact_bytes).hexdigest()
    if expected_sha256 is not None:
        if not _valid_digest(expected_sha256):
            raise BcContractError("expected_sha256 must be a lowercase SHA-256 digest")
        if file_digest != expected_sha256:
            raise BcContractError("BC actor artifact differs from expected SHA-256")
    payload = torch.load(io.BytesIO(artifact_bytes), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or set(payload) != _ARTIFACT_KEYS:
        raise BcContractError("BC actor artifact top-level schema is invalid")
    if payload["schema_name"] != SCHEMA_NAME or payload["artifact_kind"] != ARTIFACT_KIND:
        raise BcContractError("BC actor artifact schema or kind is invalid")
    if payload["architecture"] != _architecture():
        raise BcContractError("BC actor architecture is incompatible")
    if not isinstance(payload["provenance"], dict):
        raise BcContractError("BC artifact provenance is invalid")
    try:
        provenance = BcProvenance(**payload["provenance"])
    except TypeError as error:
        raise BcContractError("BC artifact provenance keys are invalid") from error
    provenance.validate()
    state = payload["actor_state_dict"]
    _validate_actor_state(state)
    if payload["actor_state_sha256"] != actor_state_sha256(state):
        raise BcContractError("BC actor state SHA-256 does not match")
    return {
        **payload,
        "provenance": provenance,
        "file_sha256": file_digest,
    }


def initialize_runner_actor(
    runner: object,
    artifact_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """runner.alg.policy.actor만 strict load하고 PPO의 나머지 상태 보존을 검증한다."""

    try:
        policy = runner.alg.policy  # type: ignore[attr-defined]
        actor = policy.actor
        critic = policy.critic
        optimizer = runner.alg.optimizer  # type: ignore[attr-defined]
    except AttributeError as error:
        raise BcContractError("runner must expose alg.policy.actor/critic and alg.optimizer") from error
    _validate_runtime_actor(actor)
    artifact = load_bc_actor_artifact(
        artifact_path, expected_sha256=expected_sha256
    )
    before = {
        "actor": module_state_sha256(actor),
        "critic": module_state_sha256(critic),
        "noise": _policy_noise_sha256(policy),
        "optimizer": object_sha256(optimizer.state_dict()),
    }
    actor.load_state_dict(artifact["actor_state_dict"], strict=True)
    after = {
        "actor": module_state_sha256(actor),
        "critic": module_state_sha256(critic),
        "noise": _policy_noise_sha256(policy),
        "optimizer": object_sha256(optimizer.state_dict()),
    }
    if after["actor"] != artifact["actor_state_sha256"]:
        raise BcContractError("loaded actor state does not match the BC artifact state SHA-256")
    for name in ("critic", "noise", "optimizer"):
        if before[name] != after[name]:
            raise BcContractError(f"actor initialization changed PPO {name}")
    return {
        "artifact_file_sha256": artifact["file_sha256"],
        "artifact_actor_state_sha256": artifact["actor_state_sha256"],
        "loaded_component": "runner.alg.policy.actor",
        "hashes_before": before,
        "hashes_after": after,
        "actor_changed": before["actor"] != after["actor"],
    }


def _architecture() -> dict[str, Any]:
    return {
        "observation_dim": OBSERVATION_DIM,
        "hidden_dims": list(HIDDEN_DIMS),
        "action_dim": ACTION_DIM,
        "activation": "elu",
        "state_dict_keys": list(EXPECTED_ACTOR_KEYS),
        "rsl_rl_version": RSL_RL_VERSION,
        "rsl_rl_source_commit": RSL_RL_SOURCE_COMMIT,
    }


def _validate_runtime_actor(actor: nn.Module) -> None:
    reference = build_bc_actor()
    if not isinstance(actor, nn.Sequential) or len(actor) != len(reference):
        raise BcContractError("actor must be the standard79 256-256-128 ELU 6D architecture")
    for actual, expected in zip(actor, reference, strict=True):
        if type(actual) is not type(expected):
            raise BcContractError("actor layer types do not match RSL-RL 3.0.1")
        if isinstance(actual, nn.Linear) and (
            actual.in_features != expected.in_features or actual.out_features != expected.out_features
        ):
            raise BcContractError("actor layer dimensions do not match RSL-RL 3.0.1")
    if tuple(actor.state_dict()) != EXPECTED_ACTOR_KEYS:
        raise BcContractError("actor state_dict keys do not match RSL-RL 3.0.1")


def _validate_batch(observation: torch.Tensor, action: torch.Tensor) -> None:
    if observation.ndim != 2 or observation.shape[1] != OBSERVATION_DIM or observation.shape[0] == 0:
        raise BcContractError("observation must have shape (N, 79), N > 0")
    if action.shape != (observation.shape[0], ACTION_DIM):
        raise BcContractError("action must have shape (N, 6) aligned with observation")
    if not observation.is_floating_point() or not action.is_floating_point():
        raise BcContractError("observation and action must be floating-point tensors")
    if not bool(torch.isfinite(observation).all()) or not bool(torch.isfinite(action).all()):
        raise BcContractError("observation and action must contain only finite values")


def _validate_optimizer_owns_actor(
    optimizer: torch.optim.Optimizer, actor: nn.Module
) -> None:
    actor_parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group.get("params", [])
    ]
    actor_ids = {id(parameter) for parameter in actor_parameters}
    optimizer_ids = {id(parameter) for parameter in optimizer_parameters}
    if len(optimizer_ids) != len(optimizer_parameters):
        raise BcContractError("optimizer must not contain duplicate actor parameters")
    missing = actor_ids - optimizer_ids
    extra = optimizer_ids - actor_ids
    if missing or extra:
        raise BcContractError(
            "optimizer parameters must exactly match all trainable actor parameters"
        )


def _validate_actor_state(state: object) -> None:
    if not isinstance(state, dict) or tuple(state) != EXPECTED_ACTOR_KEYS:
        raise BcContractError("actor_state_dict keys are invalid")
    reference = build_bc_actor().state_dict()
    for name, tensor in state.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape != reference[name].shape:
            raise BcContractError(f"actor tensor {name} has an invalid shape")
        if tensor.dtype != reference[name].dtype or not bool(torch.isfinite(tensor).all()):
            raise BcContractError(f"actor tensor {name} has an invalid dtype or value")


def _module_finite(module: nn.Module) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in module.state_dict().values())


def actor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def module_state_sha256(module: nn.Module) -> str:
    return actor_state_sha256(module.state_dict())


def object_sha256(value: object) -> str:
    stream = io.BytesIO()
    torch.save(value, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _policy_noise_sha256(policy: object) -> str:
    if hasattr(policy, "std"):
        std = getattr(policy, "std")
        if not isinstance(std, torch.Tensor):
            raise BcContractError("policy std must be a tensor")
        return actor_state_sha256({"std": std.detach()})
    if hasattr(policy, "log_std"):
        log_std = getattr(policy, "log_std")
        if not isinstance(log_std, torch.Tensor):
            raise BcContractError("policy log_std must be a tensor")
        return actor_state_sha256({"log_std": log_std.detach()})
    raise BcContractError("policy must expose std or log_std")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None

