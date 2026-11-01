"""Fixed 12-condition, four-row wave adapter for the G2 comparison pilot."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import hashlib
import math
import random
from typing import Any, Mapping, Sequence

import numpy as np


ALIGN_PHASE_CODE = 3
INSERT_PHASE_CODE = 4
PILOT_ENV_COUNT = 4
PILOT_CONDITION_COUNT = 12
PILOT_WAVE_COUNT = 3
BIAS_ABSOLUTE_TOLERANCE = 1.0e-9


class PilotEnvironmentContractViolation(RuntimeError):
    """A frozen pilot row or transition invariant was violated."""


@dataclass(frozen=True)
class PilotEnvironmentProgress:
    """Cumulative public transition evidence for the 12-condition catalog."""

    condition_ids: tuple[str, ...]
    transition_ever_checked: tuple[bool, ...]
    transition_ever_exact_zero: tuple[bool, ...]
    transition_check_counts: tuple[int, ...]
    transition_exact_zero_counts: tuple[int, ...]
    wave_sequence_index: int
    environment_rng_fingerprint: str
    reset_rng_fingerprints: tuple[str, ...]
    initial_observation_sha256: str
    rng_isolation_scope: str


RNG_ISOLATION_SCOPE = (
    "python_numpy_torch_cpu_and_cuda0_generator_state_isolated_at_reset_step_and_wave_reset;"
    "gpu_kernel_bit_exactness_not_claimed"
)


def _tensor_bytes(value: Any) -> bytes:
    return value.detach().contiguous().cpu().numpy().tobytes()


def _observation_sha256(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if hasattr(item, "detach") and hasattr(item, "shape"):
            digest.update(b"tensor\0")
            digest.update(str(item.dtype).encode("ascii"))
            digest.update(repr(tuple(item.shape)).encode("ascii"))
            digest.update(_tensor_bytes(item))
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=str):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                update(child)
        else:
            digest.update(type(item).__name__.encode("ascii"))
            digest.update(b"\0")
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must be finite")
    return result  # type: ignore[return-value]


def _sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _normalize_conditions(conditions: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if len(conditions) != PILOT_CONDITION_COUNT:
        raise ValueError("pilot catalog must contain exactly 12 conditions")
    normalized: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, row in enumerate(conditions):
        if not isinstance(row, Mapping):
            raise ValueError(f"pilot condition {index} must be an object")
        condition_id = row.get("condition_id")
        grasp_id = row.get("grasp_id")
        grasp_index = row.get("held_grasp_index")
        if not isinstance(condition_id, str) or not condition_id or condition_id in identifiers:
            raise ValueError("pilot condition_id values must be non-empty and unique")
        if not isinstance(grasp_id, str) or not grasp_id:
            raise ValueError("pilot grasp_id must be non-empty")
        if isinstance(grasp_index, bool) or not isinstance(grasp_index, int) or grasp_index < 0:
            raise ValueError("pilot held_grasp_index must be a non-negative integer")
        identifiers.add(condition_id)
        normalized.append(
            {
                "condition_id": condition_id,
                "grasp_id": grasp_id,
                "held_grasp_index": grasp_index,
                "held_artifact_sha256": _sha256(
                    row.get("held_artifact_sha256"), "held_artifact_sha256"
                ),
                "translation_bias_entry_m": _vector3(
                    row.get("translation_bias_entry_m"), "translation_bias_entry_m"
                ),
                "rotation_vector_bias_entry_rad": _vector3(
                    row.get("rotation_vector_bias_entry_rad"),
                    "rotation_vector_bias_entry_rad",
                ),
            }
        )
    return tuple(normalized)


def make_rj45_comparison_pilot_env_type(
    base_comparison_cls: type,
    *,
    conditions: Sequence[Mapping[str, Any]],
) -> type:
    """Bind a 12-condition catalog to three fixed waves of four vector rows."""

    import torch

    rows = _normalize_conditions(conditions)
    translations = torch.tensor(
        [row["translation_bias_entry_m"] for row in rows], dtype=torch.float64
    ).reshape(PILOT_WAVE_COUNT, PILOT_ENV_COUNT, 3)
    rotations = torch.tensor(
        [row["rotation_vector_bias_entry_rad"] for row in rows], dtype=torch.float64
    ).reshape(PILOT_WAVE_COUNT, PILOT_ENV_COUNT, 3)
    grasp_order = list(dict.fromkeys(row["grasp_id"] for row in rows))
    bank_slots = torch.tensor(
        [grasp_order.index(row["grasp_id"]) for row in rows], dtype=torch.long
    ).reshape(PILOT_WAVE_COUNT, PILOT_ENV_COUNT)

    class Rby1Rj45ComparisonPilotEnv(base_comparison_cls):
        def _initialize_training_state(self) -> None:
            super()._initialize_training_state()
            if self.num_envs != PILOT_ENV_COUNT:
                raise PilotEnvironmentContractViolation(
                    "G2 pilot environment requires the proven four vector rows"
                )
            self._pilot_wave_sequence_index = 0
            self._pilot_translation_bias_entry_m = translations.to(self.device)
            self._pilot_rotation_bias_entry_rad = rotations.to(self.device)
            self._pilot_held_bank_slot = bank_slots.to(self.device)
            self._pilot_transition_checked = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._pilot_transition_actual_action_is_exact_zero = torch.zeros_like(
                self._pilot_transition_checked
            )
            self._pilot_transition_ever_checked = torch.zeros_like(
                self._pilot_transition_checked
            )
            self._pilot_transition_ever_exact_zero = torch.zeros_like(
                self._pilot_transition_checked
            ).repeat(PILOT_WAVE_COUNT)
            self._pilot_transition_ever_checked = self._pilot_transition_ever_checked.repeat(
                PILOT_WAVE_COUNT
            )
            self._pilot_transition_check_counts = torch.zeros(
                PILOT_CONDITION_COUNT, dtype=torch.long, device=self.device
            )
            self._pilot_transition_exact_zero_counts = torch.zeros_like(
                self._pilot_transition_check_counts
            )
            seed = int(self.cfg.seed)
            self._pilot_environment_seed = seed
            self._pilot_torch_cuda0_rng_state = None
            self._pilot_set_rng_seed(seed)
            self._pilot_rng_scope_depth = 0
            self._pilot_reset_rng_fingerprints: list[str] = []
            self._pilot_initial_observation_sha256 = ""

        def _pilot_set_rng_seed(self, seed: int) -> None:
            python_rng = random.Random(seed)
            numpy_rng = np.random.RandomState(seed % (2**32))
            cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
            self._pilot_python_rng_state = python_rng.getstate()
            self._pilot_numpy_rng_state = numpy_rng.get_state()
            self._pilot_torch_cpu_rng_state = cpu_generator.get_state()
            if torch.cuda.is_available():
                cuda_generator = torch.Generator(device="cuda:0").manual_seed(seed)
                self._pilot_torch_cuda0_rng_state = cuda_generator.get_state()

        def _pilot_wave_rng_seed(self, sequence_index: int) -> int:
            material = (
                f"rj45-g2-environment:{self._pilot_environment_seed}:wave:{sequence_index}"
            ).encode("ascii")
            return int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % (2**63)

        def _pilot_rng_fingerprint(self) -> str:
            digest = hashlib.sha256()
            digest.update(repr(self._pilot_python_rng_state).encode("ascii"))
            numpy_state = self._pilot_numpy_rng_state
            digest.update(str(numpy_state[0]).encode("ascii"))
            digest.update(numpy_state[1].tobytes())
            digest.update(repr(numpy_state[2:]).encode("ascii"))
            digest.update(_tensor_bytes(self._pilot_torch_cpu_rng_state))
            if self._pilot_torch_cuda0_rng_state is not None:
                digest.update(_tensor_bytes(self._pilot_torch_cuda0_rng_state))
            return digest.hexdigest()

        @contextmanager
        def _pilot_environment_rng_scope(self):
            if self._pilot_rng_scope_depth:
                self._pilot_rng_scope_depth += 1
                try:
                    yield
                finally:
                    self._pilot_rng_scope_depth -= 1
                return
            outer_python = random.getstate()
            outer_numpy = np.random.get_state()
            outer_cpu = torch.random.get_rng_state()
            outer_cuda0 = (
                torch.cuda.get_rng_state(0) if self._pilot_torch_cuda0_rng_state is not None else None
            )
            self._pilot_rng_scope_depth = 1
            random.setstate(self._pilot_python_rng_state)
            np.random.set_state(self._pilot_numpy_rng_state)
            torch.random.set_rng_state(self._pilot_torch_cpu_rng_state)
            if self._pilot_torch_cuda0_rng_state is not None:
                torch.cuda.set_rng_state(self._pilot_torch_cuda0_rng_state, 0)
            try:
                yield
            finally:
                self._pilot_python_rng_state = random.getstate()
                self._pilot_numpy_rng_state = np.random.get_state()
                self._pilot_torch_cpu_rng_state = torch.random.get_rng_state()
                if self._pilot_torch_cuda0_rng_state is not None:
                    self._pilot_torch_cuda0_rng_state = torch.cuda.get_rng_state(0)
                random.setstate(outer_python)
                np.random.set_state(outer_numpy)
                torch.random.set_rng_state(outer_cpu)
                if outer_cuda0 is not None:
                    torch.cuda.set_rng_state(outer_cuda0, 0)
                self._pilot_rng_scope_depth = 0

        def reset(self, *args, **kwargs):
            self._pilot_reset_rng_fingerprints.append(self._pilot_rng_fingerprint())
            with self._pilot_environment_rng_scope():
                result = super().reset(*args, **kwargs)
            if not self._pilot_initial_observation_sha256:
                observation = result[0] if isinstance(result, tuple) else result
                self._pilot_initial_observation_sha256 = _observation_sha256(observation)
            return result

        def step(self, actions):
            with self._pilot_environment_rng_scope():
                return super().step(actions)

        def _pilot_catalog_wave_index(self) -> int:
            return self._pilot_wave_sequence_index % PILOT_WAVE_COUNT

        def _select_held_bank_rows(self, env_ids, episode_indices) -> None:
            del episode_indices
            if self._training_held_bank is None:
                raise PilotEnvironmentContractViolation(
                    "G2 fixed grasp rows require a verified held bank"
                )
            self._ensure_held_bank_tensors()
            if len(self._training_held_bank) != len(grasp_order):
                raise PilotEnvironmentContractViolation(
                    "held bank must match the two configured pilot grasps"
                )
            wave = self._pilot_catalog_wave_index()
            self._training_grasp_index[env_ids] = self._pilot_held_bank_slot[wave, env_ids]

        @staticmethod
        def _bound_right6(actions):
            clipped = torch.clamp(actions, -1.0, 1.0).reshape(-1, 2, 3)
            norms = torch.linalg.vector_norm(clipped, dim=2, keepdim=True)
            return (clipped / torch.maximum(norms, torch.ones_like(norms))).reshape(-1, 6)

        def _sample_episode_bias(self, env_ids) -> None:
            wave = self._pilot_catalog_wave_index()
            self._training_translation_bias_entry[env_ids] = (
                self._pilot_translation_bias_entry_m[wave, env_ids].to(
                    dtype=self._training_translation_bias_entry.dtype
                )
            )
            self._training_rotation_bias_entry[env_ids] = (
                self._pilot_rotation_bias_entry_rad[wave, env_ids].to(
                    dtype=self._training_rotation_bias_entry.dtype
                )
            )
            translation_error = torch.abs(
                self._training_translation_bias_entry[env_ids].to(torch.float64)
                - self._pilot_translation_bias_entry_m[wave, env_ids]
            )
            rotation_error = torch.abs(
                self._training_rotation_bias_entry[env_ids].to(torch.float64)
                - self._pilot_rotation_bias_entry_rad[wave, env_ids]
            )
            if bool(torch.any(translation_error > BIAS_ABSOLUTE_TOLERANCE)) or bool(
                torch.any(rotation_error > BIAS_ABSOLUTE_TOLERANCE)
            ):
                raise PilotEnvironmentContractViolation(
                    "fixed pilot calibration bias was not applied within 1e-9"
                )
            self.extras["rj45_comparison_pilot"] = self._pilot_snapshot()

        def _pre_physics_step(self, actions) -> None:
            if not isinstance(actions, torch.Tensor) or actions.shape != (self.num_envs, 6):
                raise ValueError("pilot action must be a (4, 6) active-wave tensor")
            if not bool(torch.isfinite(actions).all()):
                raise ValueError("pilot action must be finite")
            before = self._phase.clone()
            requested = actions.clone()
            bounded = self._bound_right6(requested)
            super()._pre_physics_step(actions)
            transitioned = (before == ALIGN_PHASE_CODE) & (self._phase == INSERT_PHASE_CODE)
            if bool(torch.any(transitioned)):
                actual = self._comparison_applied_action_6d[transitioned]
                tip_offset = self._training_applied_tip_offset_entry[transitioned]
                exact_zero = (actual == 0.0).all(dim=1) & (tip_offset == 0.0).all(dim=1)
                self._pilot_transition_checked[transitioned] = True
                self._pilot_transition_actual_action_is_exact_zero[transitioned] = exact_zero
                local_indexes = transitioned.nonzero(as_tuple=True)[0]
                catalog_indexes = (
                    self._pilot_catalog_wave_index() * PILOT_ENV_COUNT + local_indexes
                )
                self._pilot_transition_ever_checked[catalog_indexes] = True
                self._pilot_transition_ever_exact_zero[catalog_indexes] = exact_zero
                self._pilot_transition_check_counts[catalog_indexes] += 1
                self._pilot_transition_exact_zero_counts[catalog_indexes] += exact_zero.to(
                    dtype=torch.long
                )
                if not bool(exact_zero.all()):
                    raise PilotEnvironmentContractViolation(
                        "ALIGN-to-INSERT transition applied a nonzero pilot residual"
                    )
            self.extras["rj45_comparison_pilot"] = {
                **self._pilot_snapshot(),
                "requested_action_6d": requested.clone(),
                "bounded_action_6d": bounded.clone(),
            }

        def _pilot_snapshot(self) -> dict[str, Any]:
            wave = self._pilot_catalog_wave_index()
            start = wave * PILOT_ENV_COUNT
            active_rows = rows[start : start + PILOT_ENV_COUNT]
            return {
                "condition_catalog_ids": [row["condition_id"] for row in rows],
                "active_condition_ids": [row["condition_id"] for row in active_rows],
                "active_grasp_ids": [row["grasp_id"] for row in active_rows],
                "active_held_grasp_indices": [
                    row["held_grasp_index"] for row in active_rows
                ],
                "active_held_bank_slots": self._pilot_held_bank_slot[wave].clone(),
                "active_held_artifact_sha256": [
                    row["held_artifact_sha256"] for row in active_rows
                ],
                "wave_sequence_index": self._pilot_wave_sequence_index,
                "catalog_wave_index": wave,
                "requested_translation_bias_entry_m": (
                    self._pilot_translation_bias_entry_m[wave].clone()
                ),
                "requested_rotation_bias_entry_rad": (
                    self._pilot_rotation_bias_entry_rad[wave].clone()
                ),
                "transition_contract_checked": self._pilot_transition_checked.clone(),
                "transition_actual_action_is_exact_zero": (
                    self._pilot_transition_actual_action_is_exact_zero.clone()
                ),
                "transition_ever_checked": self._pilot_transition_ever_checked.clone(),
                "transition_ever_exact_zero": (
                    self._pilot_transition_ever_exact_zero.clone()
                ),
                "transition_check_counts": self._pilot_transition_check_counts.clone(),
                "transition_exact_zero_counts": (
                    self._pilot_transition_exact_zero_counts.clone()
                ),
                "actor_observation_dim": 79,
                "actor_action_dim": 6,
                "evaluator_gt_available_to_policy": False,
            }

        def _training_step_snapshot(self):
            snapshot = super()._training_step_snapshot()
            snapshot["comparison_pilot"] = self._pilot_snapshot()
            return snapshot

        def comparison_pilot_progress(self) -> PilotEnvironmentProgress:
            """Expose cumulative condition coverage without mutable environment state."""

            return PilotEnvironmentProgress(
                condition_ids=tuple(row["condition_id"] for row in rows),
                transition_ever_checked=tuple(
                    bool(value)
                    for value in self._pilot_transition_ever_checked.detach().cpu().tolist()
                ),
                transition_ever_exact_zero=tuple(
                    bool(value)
                    for value in self._pilot_transition_ever_exact_zero.detach().cpu().tolist()
                ),
                transition_check_counts=tuple(
                    int(value)
                    for value in self._pilot_transition_check_counts.detach().cpu().tolist()
                ),
                transition_exact_zero_counts=tuple(
                    int(value)
                    for value in self._pilot_transition_exact_zero_counts.detach().cpu().tolist()
                ),
                wave_sequence_index=self._pilot_wave_sequence_index,
                environment_rng_fingerprint=self._pilot_rng_fingerprint(),
                reset_rng_fingerprints=tuple(self._pilot_reset_rng_fingerprints),
                initial_observation_sha256=self._pilot_initial_observation_sha256,
                rng_isolation_scope=RNG_ISOLATION_SCOPE,
            )

        def advance_comparison_pilot_wave(self, sequence_index: int):
            """Advance exactly one preregistered wave and synchronously reset all rows."""

            if (
                isinstance(sequence_index, bool)
                or not isinstance(sequence_index, int)
                or sequence_index != self._pilot_wave_sequence_index + 1
            ):
                raise PilotEnvironmentContractViolation(
                    "pilot wave sequence must advance by exactly one"
                )
            previous = self._pilot_wave_sequence_index
            previous_rng = (
                self._pilot_python_rng_state,
                self._pilot_numpy_rng_state,
                self._pilot_torch_cpu_rng_state,
                self._pilot_torch_cuda0_rng_state,
            )
            previous_reset_count = len(self._pilot_reset_rng_fingerprints)
            self._pilot_wave_sequence_index = sequence_index
            self._pilot_set_rng_seed(self._pilot_wave_rng_seed(sequence_index))
            try:
                return self.reset()
            except Exception:
                self._pilot_wave_sequence_index = previous
                (
                    self._pilot_python_rng_state,
                    self._pilot_numpy_rng_state,
                    self._pilot_torch_cpu_rng_state,
                    self._pilot_torch_cuda0_rng_state,
                ) = previous_rng
                del self._pilot_reset_rng_fingerprints[previous_reset_count:]
                raise

        def _configure_training_reset(self, env_ids) -> None:
            super()._configure_training_reset(env_ids)
            self._pilot_transition_checked[env_ids] = False
            self._pilot_transition_actual_action_is_exact_zero[env_ids] = False

    Rby1Rj45ComparisonPilotEnv.__name__ = "Rby1Rj45ComparisonPilotEnv"
    return Rby1Rj45ComparisonPilotEnv


__all__ = [
    "BIAS_ABSOLUTE_TOLERANCE",
    "PILOT_ENV_COUNT",
    "PILOT_CONDITION_COUNT",
    "PILOT_WAVE_COUNT",
    "RNG_ISOLATION_SCOPE",
    "PilotEnvironmentContractViolation",
    "PilotEnvironmentProgress",
    "make_rj45_comparison_pilot_env_type",
]
