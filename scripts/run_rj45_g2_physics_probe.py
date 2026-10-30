#!/usr/bin/env python3
"""Run one fail-closed G2 diagnostic actor in the fixed RJ45 PhysX condition.

The module is safe to import without Isaac Sim.  Isaac Lab and Kit are loaded only
by :func:`main` after command-line parsing and application launch.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping


SCHEMA_NAME = "rj45_g2_physics_probe_v1"
OBSERVATION_DIM = 79
ACTION_DIM = 6
BASE_SEED = 20261120
ENVIRONMENT_INDEX = 0
EPISODE_INDEX = 1
CALIBRATION_CASE_SEED = 66384140
GRASP_INDEX = 1
GRASP_ID = "rj45-train-02"
PHYSICS_HZ = 240
CONTROL_DECIMATION = 8
CONTROL_HZ = 30
HORIZON_S = 24.0
RJ45_GEOMETRY_ABSOLUTE_TOLERANCE = 2.0e-6
EXPECTED_BANK_SHA256 = "c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5"
EXPECTED_NPZ_SHA256 = "482215e4507810e04d9a3a1168d0dde6007c9015fe0f3720101543b8e41a3633"
EXPECTED_SOURCE_CONTRACT_SHA256 = "38cf66d4dea11fe4253021fa57e3c0b5347974dda7490ddb9b7890e6819098d0"
EXPECTED_TASK_SHA256 = "7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b"
EXPECTED_TRAINING_SHA256 = "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179"
EXPECTED_INPUT_NORMALIZATION_RUNNER_SHA256 = (
    "26b631cfc873ba87de011fa32f71af1868bafe1885d255b4d11d346652f1f928"
)
EXPECTED_POLICIES = {
    "a_standard_input": {
        "checkpoint_file_sha256": "b36c7e0e9ec09389286d1d6c6f61bc60b97f6f5d8ccc81effa7a28a47d1329be",
        "actor_state_sha256": "d901b02c36654df9cf83474fe66036582f54b9347abc5db505f16833788795a3",
        "input_transform_kind": "identity",
        "offline_mse": 2.1611929828961087e-6,
        "prediction_sha256": "d57ed256237a6bfa85eef92af49048e0cefeb0f6f9b34f2a9d09fa3ae0d105ff",
    },
    "b_normalized_input": {
        "checkpoint_file_sha256": "a4bd4eb498e27c09899608591480287a45016beb466a2f8d0a74879ef400ee65",
        "actor_state_sha256": "d3bf262d7033772828ce53c2e4b4d6a583db830b99a0f47b9e1d6643bed3f0c5",
        "input_transform_kind": "train_standardize",
        "offline_mse": 3.4301454586823394e-7,
        "prediction_sha256": "6d8eb037b95653c0095e2b684bf39b2b69c280dede230800623b88053e2cb9b6",
    },
}
OFFLINE_MSE_ABSOLUTE_TOLERANCE = 1e-12


class PhysicsProbeError(RuntimeError):
    """The frozen one-episode diagnostic contract was violated."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: Any) -> str:
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def manifest_array_sha256(value: Any) -> str:
    """Match the fixed-condition exporter hash including dtype and shape."""

    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PhysicsProbeError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise PhysicsProbeError(f"JSON root must be an object: {path}")
    return value


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PhysicsProbeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _diagnostic_loader(current_root: Path) -> Any:
    path = current_root / "scripts/run_rj45_g2_input_normalization_probe.py"
    if file_sha256(path) != EXPECTED_INPUT_NORMALIZATION_RUNNER_SHA256:
        raise PhysicsProbeError("input-normalization runner SHA-256 mismatch")
    return _load_module(
        path,
        "rj45_g2_input_normalization_runtime_loader",
    )


def _validated_offline_rows(
    manifest_path: Path,
    npz_path: Path,
    *,
    expected_npz_sha256: str = EXPECTED_NPZ_SHA256,
) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np

    manifest = _load_json(manifest_path)
    if manifest.get("artifact_kind") != "train_only_fixed_condition_offline_bc_probe_input":
        raise PhysicsProbeError("offline manifest artifact kind is invalid")
    if file_sha256(npz_path) != expected_npz_sha256:
        raise PhysicsProbeError("offline NPZ SHA-256 mismatch")
    if manifest.get("npz", {}).get("sha256") != expected_npz_sha256:
        raise PhysicsProbeError("offline manifest NPZ SHA-256 mismatch")
    condition = manifest.get("condition")
    expected_condition = {
        "bias_case_id": "env-0-episode-1",
        "grasp_id": GRASP_ID,
        "seed": CALIBRATION_CASE_SEED,
        "split": "train",
    }
    if not isinstance(condition, Mapping) or any(condition.get(k) != v for k, v in expected_condition.items()):
        raise PhysicsProbeError("offline fixed-condition identity mismatch")
    lineage = manifest.get("lineage")
    expected_lineage = {
        "bank_sha256": EXPECTED_BANK_SHA256,
        "source_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "task_sha256": EXPECTED_TASK_SHA256,
        "training_sha256": EXPECTED_TRAINING_SHA256,
    }
    if not isinstance(lineage, Mapping) or any(
        lineage.get(key) != value for key, value in expected_lineage.items()
    ):
        raise PhysicsProbeError("offline lineage mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"observation", "action", "valid_mask"}:
            raise PhysicsProbeError("offline NPZ array names changed")
        observation = np.asarray(arrays["observation"], dtype=np.float32)
        action = np.asarray(arrays["action"], dtype=np.float32)
        valid_mask = np.asarray(arrays["valid_mask"], dtype=np.bool_)
    if observation.shape != (418, OBSERVATION_DIM) or action.shape != (418, ACTION_DIM) or valid_mask.shape != (418,):
        raise PhysicsProbeError("offline NPZ shapes changed")
    if int(valid_mask.sum()) != 395:
        raise PhysicsProbeError("offline NPZ must contain exactly 395 valid rows")
    if not np.isfinite(observation).all() or not np.isfinite(action).all():
        raise PhysicsProbeError("offline NPZ contains non-finite values")
    manifest_arrays = manifest.get("arrays", {})
    for name, value in (("observation", observation), ("action", action), ("valid_mask", valid_mask)):
        expected = manifest_arrays.get(name, {}).get("sha256")
        if expected != manifest_array_sha256(value):
            raise PhysicsProbeError(f"offline array SHA-256 mismatch: {name}")
    return observation[valid_mask], action[valid_mask], manifest


class CpuDiagnosticPolicy:
    """Verified CPU actor and its mandatory input transform."""

    def __init__(self, checkpoint: Path, expected_sha256: str, current_root: Path) -> None:
        import torch
        import rby1_assembly.rj45_bc as bc

        observed_sha = file_sha256(checkpoint)
        if observed_sha != expected_sha256:
            raise PhysicsProbeError("diagnostic checkpoint SHA-256 mismatch")
        loader = _diagnostic_loader(current_root)
        loaded = loader._load_checkpoint(checkpoint)
        payload = loaded["payload"]
        if payload.get("must_not_be_injected_into_main_experiment") is not True:
            raise PhysicsProbeError("diagnostic-only checkpoint marker is missing")
        if payload.get("output_unscale_or_clip") is not False:
            raise PhysicsProbeError("checkpoint output contract changed")
        actor = bc.build_bc_actor().cpu()
        actor.load_state_dict(payload["actor_state_dict"], strict=True)
        actor.eval()
        self._torch = torch
        self._loader = loader
        self._actor = actor
        self._normalizer = loaded["normalizer"]
        self.identity = {
            "path": str(checkpoint),
            "file_sha256": observed_sha,
            "actor_state_sha256": payload["actor_state_sha256"],
            "normalizer_array_sha256": dict(payload["input_transform"]["array_sha256"]),
            "input_transform_kind": payload["input_transform"].get("kind"),
        }

    def predict(self, observation: Any) -> Any:
        import numpy as np

        values = np.asarray(observation, dtype=np.float32)
        if values.shape == (OBSERVATION_DIM,):
            values = values[None, :]
        if values.shape != (1, OBSERVATION_DIM) or not np.isfinite(values).all():
            raise PhysicsProbeError("policy observation must be one finite 79D row")
        transformed = self._loader._apply_normalizer(values, self._normalizer)
        with self._torch.inference_mode():
            output = self._actor(self._torch.as_tensor(transformed, dtype=self._torch.float32)).cpu().numpy()
        if output.shape != (1, ACTION_DIM) or not np.isfinite(output).all():
            raise PhysicsProbeError("diagnostic actor produced an invalid 6D action")
        return output.astype(np.float32, copy=False)


def offline_parity_preflight(policy: CpuDiagnosticPolicy, observation: Any, action: Any) -> dict[str, Any]:
    import numpy as np

    predictions = np.concatenate([policy.predict(row) for row in observation], axis=0)
    if predictions.shape != (395, ACTION_DIM) or not np.isfinite(predictions).all():
        raise PhysicsProbeError("offline checkpoint preflight prediction is invalid")
    errors = predictions.astype(np.float64) - np.asarray(action, dtype=np.float64)
    return {
        "rows": 395,
        "prediction_sha256": array_sha256(predictions),
        "target_sha256": array_sha256(np.asarray(action, dtype=np.float32)),
        "mse": float(np.mean(errors * errors)),
        "maximum_absolute_error": float(np.max(np.abs(errors))),
        "all_finite": True,
    }


def validate_policy_preflight(
    policy_label: str, policy_identity: Mapping[str, Any], preflight: Mapping[str, Any]
) -> None:
    expected = EXPECTED_POLICIES[policy_label]
    if policy_identity.get("actor_state_sha256") != expected["actor_state_sha256"]:
        raise PhysicsProbeError("policy label and actor state identity do not match")
    if policy_identity.get("input_transform_kind") != expected["input_transform_kind"]:
        raise PhysicsProbeError("policy label and input transform do not match")
    if preflight.get("prediction_sha256") != expected["prediction_sha256"]:
        raise PhysicsProbeError("offline checkpoint prediction SHA-256 changed")
    if (
        abs(float(preflight["mse"]) - float(expected["offline_mse"]))
        > OFFLINE_MSE_ABSOLUTE_TOLERANCE
    ):
        raise PhysicsProbeError("offline checkpoint parity MSE changed")


def fixed_condition_identity(bank_proof: Mapping[str, Any]) -> dict[str, Any]:
    ids = bank_proof.get("selected_grasp_ids")
    if bank_proof.get("manifest_sha256") != EXPECTED_BANK_SHA256:
        raise PhysicsProbeError("held bank manifest SHA-256 mismatch")
    if bank_proof.get("selected_split") != "train":
        raise PhysicsProbeError("held bank must select the train split")
    if not isinstance(ids, list) or len(ids) != 8 or ids[GRASP_INDEX] != GRASP_ID:
        raise PhysicsProbeError("held bank ordered train identity mismatch")
    return {
        "base_seed": BASE_SEED,
        "environment_index": ENVIRONMENT_INDEX,
        "episode_index": EPISODE_INDEX,
        "calibration_case_seed": CALIBRATION_CASE_SEED,
        "grasp_index": GRASP_INDEX,
        "grasp_id": GRASP_ID,
        "bank_manifest_sha256": bank_proof["manifest_sha256"],
        "bank_split": bank_proof["selected_split"],
    }


def verify_calibration_case_seed(seed_function: Any) -> None:
    actual = seed_function(BASE_SEED, ENVIRONMENT_INDEX, EPISODE_INDEX)
    if actual != CALIBRATION_CASE_SEED:
        raise PhysicsProbeError("calibration case seed formula changed")


def advance_to_fixed_episode(env: Any) -> Any:
    """Advance constructor-created episode 0 to fixed episode 1 exactly once."""

    if int(_scalar(env._training_episode_index)) != 0:
        raise PhysicsProbeError("constructor did not establish episode index 0")
    if int(_scalar(env._training_grasp_index)) != 0:
        raise PhysicsProbeError("constructor did not select held-bank grasp index 0")
    reset_result = env.reset()
    observations = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    actual_episode = int(_scalar(env._training_episode_index))
    if actual_episode != EPISODE_INDEX:
        raise PhysicsProbeError(f"unexpected reset episode index: {actual_episode}")
    if int(_scalar(env._training_grasp_index)) != GRASP_INDEX:
        raise PhysicsProbeError("fixed held-bank grasp index mismatch after reset")
    return observations


def classify_terminal(terminal: Any) -> str:
    if not bool(terminal.valid):
        return "invalid"
    return "valid_success" if bool(terminal.success) else "valid_failure"


def frozen_protocol_record(runner_sha256: str) -> dict[str, Any]:
    payload = {
        "schema_name": SCHEMA_NAME,
        "runner_sha256": runner_sha256,
        "bank_manifest_sha256": EXPECTED_BANK_SHA256,
        "offline_npz_sha256": EXPECTED_NPZ_SHA256,
        "source_contract_sha256": EXPECTED_SOURCE_CONTRACT_SHA256,
        "task_config_sha256": EXPECTED_TASK_SHA256,
        "training_config_sha256": EXPECTED_TRAINING_SHA256,
        "input_normalization_runner_sha256": (
            EXPECTED_INPUT_NORMALIZATION_RUNNER_SHA256
        ),
        "fixed_condition": {
            "base_seed": BASE_SEED,
            "environment_index": ENVIRONMENT_INDEX,
            "episode_index": EPISODE_INDEX,
            "calibration_case_seed": CALIBRATION_CASE_SEED,
            "grasp_index": GRASP_INDEX,
            "grasp_id": GRASP_ID,
        },
        "timing": {
            "physics_hz": PHYSICS_HZ,
            "control_decimation": CONTROL_DECIMATION,
            "control_hz": CONTROL_HZ,
            "horizon_s": HORIZON_S,
        },
        "policies": EXPECTED_POLICIES,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"payload": payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_protocol(path: Path, expected_file_sha256: str, runner_sha256: str) -> dict[str, Any]:
    if file_sha256(path) != expected_file_sha256:
        raise PhysicsProbeError("shared protocol file SHA-256 mismatch")
    observed = _load_json(path)
    expected = frozen_protocol_record(runner_sha256)
    if observed != expected["payload"]:
        raise PhysicsProbeError("shared protocol payload changed")
    return {"path": str(path), "file_sha256": expected_file_sha256, **expected}


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.rename(temporary, path)


def _scalar(value: Any, env_id: int = 0) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    import numpy as np

    array = np.asarray(value)
    if array.ndim:
        array = array[env_id]
    return array.item() if np.asarray(array).ndim == 0 else array.tolist()


def _observation_tensor(observations: Any) -> Any:
    try:
        value = observations["policy"]
    except (KeyError, TypeError) as error:
        raise PhysicsProbeError("environment observation must contain policy") from error
    if tuple(value.shape) != (1, OBSERVATION_DIM) or not bool(value.isfinite().all().item()):
        raise PhysicsProbeError("environment policy observation is not finite shape (1, 79)")
    return value


def _limits(task: Mapping[str, Any], training: Mapping[str, Any]) -> Any:
    from rby1_assembly.contracts import TaskLimits

    evaluation = task["evaluation"]
    controller = task["controller"]
    return TaskLimits(
        assembled_depth_m=float(evaluation["assembled_depth_m"]),
        max_lateral_error_m=float(evaluation["max_lateral_error_m"]),
        max_orientation_error_rad=float(evaluation["max_orientation_error_rad"]),
        max_contact_force_n=float(evaluation["max_contact_force_n"]),
        dwell_s=float(evaluation["dwell_s"]),
        timeout_s=float(training["active_horizon_s"]),
        max_retries=int(controller.get("max_retries", 2)),
        residual_translation_bound_m=float(controller["residual_translation_bound_m"]),
        residual_rotation_bound_rad=float(controller["residual_rotation_bound_rad"]),
    )


def _evaluation_sample(
    snapshot: Any,
    transition: Mapping[str, Any],
    timestamp_s: float,
    evaluation_contract: Mapping[str, Any],
) -> Any:
    import numpy as np
    from rby1_assembly.contracts import EvaluationSample

    metrics = snapshot.ground_truth
    stability = transition.get("rj45_evaluation_stability")
    if not isinstance(stability, Mapping):
        raise PhysicsProbeError("RJ45 substep stability snapshot is missing")
    names = (
        "physics_sample_depth_m",
        "physics_sample_lateral_error_m",
        "physics_sample_orientation_error_rad",
        "physics_sample_relative_linear_speed_m_s",
        "physics_sample_relative_angular_speed_rad_s",
    )
    samples = {
        name: np.asarray(_scalar(stability[name]), dtype=np.float64)
        for name in names
    }
    expected_shape = (CONTROL_DECIMATION + 1,)
    if any(value.shape != expected_shape for value in samples.values()):
        raise PhysicsProbeError("RJ45 stability snapshot must contain nine physics samples")
    endpoint = (
        samples["physics_sample_depth_m"][-1],
        samples["physics_sample_lateral_error_m"][-1],
        samples["physics_sample_orientation_error_rad"][-1],
    )
    if (
        abs(endpoint[0] - metrics.actual_depth_m)
        > RJ45_GEOMETRY_ABSOLUTE_TOLERANCE
        or abs(endpoint[1] - metrics.lateral_error_m)
        > RJ45_GEOMETRY_ABSOLUTE_TOLERANCE
        or abs(endpoint[2] - metrics.orientation_error_rad)
        > RJ45_GEOMETRY_ABSOLUTE_TOLERANCE
    ):
        raise PhysicsProbeError("RJ45 substep endpoint differs from object-pose metrics")
    finite = np.logical_and.reduce(tuple(np.isfinite(value) for value in samples.values()))
    recomputed = (
        finite
        & (samples["physics_sample_depth_m"] >= evaluation_contract["assembled_depth_m"])
        & (samples["physics_sample_depth_m"] <= evaluation_contract["max_depth_m"])
        & (
            samples["physics_sample_lateral_error_m"]
            <= evaluation_contract["max_lateral_error_m"]
        )
        & (
            samples["physics_sample_orientation_error_rad"]
            <= evaluation_contract["max_orientation_error_rad"]
        )
        & (
            samples["physics_sample_relative_linear_speed_m_s"]
            <= evaluation_contract["max_relative_linear_speed_m_s"]
        )
        & (
            samples["physics_sample_relative_angular_speed_rad_s"]
            <= evaluation_contract["max_relative_angular_speed_rad_s"]
        )
    )
    reported_finite = np.asarray(
        _scalar(stability["physics_sample_finite"]), dtype=np.bool_
    )
    reported_stable = np.asarray(
        _scalar(stability["physics_sample_stable"]), dtype=np.bool_
    )
    if not np.array_equal(reported_finite, finite) or not np.array_equal(
        reported_stable, recomputed
    ):
        raise PhysicsProbeError("RJ45 raw substep stability proof changed")
    samples_complete = bool(_scalar(stability["physics_substeps_complete"]))
    all_samples_stable = bool(_scalar(stability["physics_all_samples_stable"]))
    if all_samples_stable != bool(recomputed.all() and samples_complete):
        raise PhysicsProbeError("RJ45 all-substep stability proof changed")
    strict_grasp_gate = (
        metrics.grasp_both
        and snapshot.phase in {"insert", "done"}
        and samples_complete
        and all_samples_stable
    )
    return EvaluationSample(
        actual_depth_m=metrics.actual_depth_m,
        lateral_error_m=metrics.lateral_error_m,
        orientation_error_rad=metrics.orientation_error_rad,
        force_peak_n=metrics.force_peak_n,
        grasp_both=strict_grasp_gate,
        forbidden_contact=metrics.forbidden_contact,
        sim_time_s=timestamp_s,
    )


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(_scalar(value))
    if not math.isfinite(number) or number < 0.0:
        raise PhysicsProbeError(f"measured contact field is invalid: {name}")
    return number


def _verified_contact_metadata(
    env: Any, payload: Mapping[str, Any], evaluation: Mapping[str, Any]
) -> dict[str, Any]:
    """Verify the passive combined-contact source and its component evidence."""

    from rby1_assembly.hand_collision import (
        COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
        COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
        CROSS_HAND_CONTACT_LOAD_SOURCE,
        selective_hand_collision_pairs,
    )

    report = env.hand_contact_monitor_report()
    expected_report = {
        "command_pose_force_or_failure_injection": False,
        "hand_contact_monitor_mode": "passive_nominal_one_environment",
        "num_envs": 1,
        "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
        "forbidden_contact_load_definition": COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
        "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
    }
    if any(report.get(key) != value for key, value in expected_report.items()):
        raise PhysicsProbeError("hand-contact monitor runtime proof is incomplete")
    collision_filter = report.get("collision_filter")
    if not isinstance(collision_filter, Mapping):
        raise PhysicsProbeError("hand-contact collision-filter proof is missing")
    expected_filter = {
        "self_collision_enabled": True,
        "rigid_body_count": 66,
        "allowed_cross_hand_pair_count": 9,
        "filtered_internal_pair_count": 2136,
        "filtered_pair_authoring": "one_direction_per_unordered_rigid_body_pair",
        "external_collision_relationships_modified": False,
    }
    if any(collision_filter.get(key) != value for key, value in expected_filter.items()):
        raise PhysicsProbeError("hand-contact collision-filter proof changed")
    env_prim_paths = env.scene.env_prim_paths
    if not isinstance(env_prim_paths, (list, tuple)) or len(env_prim_paths) != 1:
        raise PhysicsProbeError("one-environment scene path proof changed")
    robot_root_path = f"{env_prim_paths[0]}/Robot"
    if collision_filter.get("robot_root_path") != robot_root_path:
        raise PhysicsProbeError("hand-contact robot-root proof changed")
    allowed_pairs = collision_filter.get("allowed_cross_hand_pairs")
    if not isinstance(allowed_pairs, (list, tuple)):
        raise PhysicsProbeError("hand-contact allowed-pair proof changed")
    try:
        observed_pairs = tuple(tuple(pair) for pair in allowed_pairs)
    except TypeError as exc:
        raise PhysicsProbeError("hand-contact allowed-pair proof changed") from exc
    expected_pairs = selective_hand_collision_pairs(robot_root_path)["allowed"]
    if (
        any(
            len(pair) != 2 or not all(isinstance(path, str) for path in pair)
            for pair in observed_pairs
        )
        or len(set(observed_pairs)) != len(expected_pairs)
        or set(observed_pairs) != set(expected_pairs)
    ):
        raise PhysicsProbeError("hand-contact allowed-pair proof changed")

    payload_expected = {
        "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
        "forbidden_contact_load_definition": COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
        "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
        "self_collision_enabled": True,
        "hand_contact_monitor_mode": "passive_nominal_one_environment",
    }
    for container_name, container in (("payload", payload), ("evaluation", evaluation)):
        if any(container.get(key) != value for key, value in payload_expected.items()):
            raise PhysicsProbeError(
                f"{container_name} combined-contact metadata differs from runtime proof"
            )
        if container.get("selective_self_collision_filter") != collision_filter:
            raise PhysicsProbeError(
                f"{container_name} collision-filter metadata differs from runtime proof"
            )

    component_names = (
        "cross_hand_contact_load_n",
        "cross_hand_contact_load_peak_n",
        "external_forbidden_contact_load_n",
        "external_forbidden_contact_load_peak_n",
    )
    components = {name: _finite_nonnegative(payload[name], name) for name in component_names}
    for name, value in components.items():
        if not math.isclose(
            _finite_nonnegative(evaluation[name], f"evaluation.{name}"),
            value,
            abs_tol=1e-9,
            rel_tol=1e-6,
        ):
            raise PhysicsProbeError(f"evaluation component differs from payload: {name}")
    return {"collision_filter": dict(collision_filter), **components}


def _probe_snapshot(env: Any, payload: Mapping[str, Any], geometry: Any) -> Any:
    """Build one immutable post-physics transition from the pre-reset snapshot."""

    import numpy as np
    from rby1_assembly.dataset import GroundTruthMetrics
    from rby1_assembly.recording import PHASE_NAMES
    from rby1_assembly.rj45_task import verify_recorded_mating_geometry

    evaluation = payload.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise PhysicsProbeError("physics snapshot evaluation mapping is missing")
    contact_metadata = _verified_contact_metadata(env, payload, evaluation)

    poses = np.asarray(_scalar(payload["object_poses_world_wxyz"]), dtype=np.float64)
    phase_index = int(_scalar(payload["phase"]))
    if poses.shape != (2, 7) or not np.isfinite(poses).all():
        raise PhysicsProbeError("physics object poses are invalid")
    if not 0 <= phase_index < len(PHASE_NAMES):
        raise PhysicsProbeError("physics phase index is invalid")
    reported = (
        float(_scalar(evaluation["actual_depth_m"])),
        float(_scalar(evaluation["actual_lateral_error_m"])),
        float(_scalar(evaluation["actual_orientation_error_rad"])),
    )
    try:
        verified_geometry = verify_recorded_mating_geometry(
            poses[0],
            poses[1],
            geometry["socket_entry"],
            geometry["plug_tip"],
            *reported,
            absolute_tolerance=RJ45_GEOMETRY_ABSOLUTE_TOLERANCE,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PhysicsProbeError("RJ45 full-frame geometry proof changed") from exc
    depth = verified_geometry["depth_m"]
    lateral = verified_geometry["lateral_error_m"]
    orientation = verified_geometry["orientation_error_rad"]

    from rby1_assembly.recording import INSERTION_FORCE_SOURCE, INSERTION_LOAD_SOURCE

    if payload.get("insertion_contact_force_source") != INSERTION_FORCE_SOURCE:
        raise PhysicsProbeError("insertion contact-force source changed")
    if payload.get("insertion_contact_load_source") != INSERTION_LOAD_SOURCE:
        raise PhysicsProbeError("insertion contact-load source changed")
    grip_peak = _finite_nonnegative(payload["grip_contact_force_peak_n"], "grip peak")
    insertion_force = _finite_nonnegative(payload["insertion_contact_force_n"], "insertion force")
    insertion_force_w = np.asarray(_scalar(payload["insertion_contact_force_w"]), dtype=np.float64)
    insertion_force_peak = _finite_nonnegative(
        payload["insertion_contact_force_peak_n"], "insertion force peak"
    )
    insertion_load = _finite_nonnegative(payload["insertion_contact_load_n"], "insertion load")
    insertion_peak = _finite_nonnegative(payload["insertion_contact_load_peak_n"], "insertion load peak")
    forbidden_load = _finite_nonnegative(payload["forbidden_contact_load_n"], "forbidden load")
    forbidden_peak = _finite_nonnegative(payload["forbidden_contact_load_peak_n"], "forbidden peak")
    if insertion_force_w.shape != (3,) or not np.isfinite(insertion_force_w).all():
        raise PhysicsProbeError("insertion contact-force vector is invalid")
    if not math.isclose(
        float(np.linalg.norm(insertion_force_w)), insertion_force, abs_tol=1e-5, rel_tol=1e-5
    ):
        raise PhysicsProbeError("insertion contact-force norm is inconsistent")
    measured_values = {
        "grip_contact_force_peak_n": grip_peak,
        "insertion_contact_force_n": insertion_force,
        "insertion_contact_force_peak_n": insertion_force_peak,
        "insertion_contact_load_n": insertion_load,
        "insertion_contact_load_peak_n": insertion_peak,
        "forbidden_contact_load_n": forbidden_load,
        "forbidden_contact_load_peak_n": forbidden_peak,
    }
    for name, value in measured_values.items():
        observed = _finite_nonnegative(evaluation[name], f"evaluation.{name}")
        if not math.isclose(observed, value, abs_tol=1e-4, rel_tol=1e-5):
            raise PhysicsProbeError(f"evaluation contact field differs from measurement: {name}")
    force_peak = float(_scalar(evaluation["contact_force_proxy_peak_n"]))
    if (
        insertion_load + 1e-4 < insertion_force
        or insertion_peak + 1e-9 < insertion_load
        or insertion_peak + 1e-4 < insertion_force_peak
    ):
        raise PhysicsProbeError("insertion contact-load ordering is invalid")
    if forbidden_peak + 1e-9 < forbidden_load:
        raise PhysicsProbeError("forbidden contact-load ordering is invalid")
    if not math.isclose(force_peak, max(grip_peak, insertion_peak), abs_tol=1e-4, rel_tol=1e-5):
        raise PhysicsProbeError("contact force peak differs from measured component peaks")
    combined_current = max(
        contact_metadata["cross_hand_contact_load_n"],
        contact_metadata["external_forbidden_contact_load_n"],
    )
    combined_peak = max(
        contact_metadata["cross_hand_contact_load_peak_n"],
        contact_metadata["external_forbidden_contact_load_peak_n"],
    )
    if not math.isclose(forbidden_load, combined_current, abs_tol=1e-9, rel_tol=1e-6):
        raise PhysicsProbeError("combined forbidden load differs from its components")
    if not math.isclose(forbidden_peak, combined_peak, abs_tol=1e-9, rel_tol=1e-6):
        raise PhysicsProbeError("combined forbidden peak differs from its components")
    return SimpleNamespace(
        phase=PHASE_NAMES[phase_index],
        ground_truth=GroundTruthMetrics(
            actual_depth_m=depth,
            lateral_error_m=lateral,
            orientation_error_rad=orientation,
            force_peak_n=force_peak,
            grasp_both=bool(_scalar(evaluation["grasp_both"])),
            forbidden_contact=bool(_scalar(evaluation["forbidden_contact"])),
        ),
        reported_geometry_consistent=True,
        grip_contact_force_peak_n=grip_peak,
        insertion_contact_force_n=insertion_force,
        insertion_contact_load_n=insertion_load,
        forbidden_contact_load_n=forbidden_load,
        forbidden_contact_load_source=payload["forbidden_contact_load_source"],
        forbidden_contact_load_definition=payload["forbidden_contact_load_definition"],
        collision_filter=contact_metadata["collision_filter"],
    )


def action_provenance(
    submitted_6d: Any, requested_expanded_12d: Any, applied_expanded_12d: Any
) -> dict[str, Any]:
    import numpy as np

    submitted = np.asarray(submitted_6d, dtype=np.float32).reshape(-1)
    requested = np.asarray(requested_expanded_12d, dtype=np.float32).reshape(-1)
    applied = np.asarray(applied_expanded_12d, dtype=np.float32).reshape(-1)
    if submitted.shape != (6,) or requested.shape != (12,) or applied.shape != (12,):
        raise PhysicsProbeError("action provenance shapes must be 6D, 12D, and 12D")
    if not all(np.isfinite(value).all() for value in (submitted, requested, applied)):
        raise PhysicsProbeError("action provenance contains non-finite values")
    if not np.array_equal(requested[:6], np.zeros(6, dtype=np.float32)):
        raise PhysicsProbeError("requested expanded action changed the held left arm")
    if not np.array_equal(applied[:6], np.zeros(6, dtype=np.float32)):
        raise PhysicsProbeError("applied expanded action changed the held left arm")
    active = bool(np.any(requested[6:] != 0.0))
    if active and not np.array_equal(requested[6:], submitted):
        raise PhysicsProbeError("submitted 6D action does not match requested right-arm action")
    return {
        "submitted_action_6d": submitted.tolist(),
        "requested_expanded_action_12d": requested.tolist(),
        "applied_bounded_action_6d": applied[6:].tolist(),
        "applied_bounded_action_12d": applied.tolist(),
        "policy_action_active": active,
    }


def _trace_record(
    step: int,
    observation: Any,
    submitted: Any,
    requested_expanded: Any,
    applied_expanded: Any,
    offset: Any,
    snapshot: Any,
    control_elapsed_s: float,
    environment_episode_timestamp_s: float,
    valid_input: bool,
) -> dict[str, Any]:
    import numpy as np

    raw = observation.detach().cpu().numpy().astype(np.float32, copy=False)[0]
    offset_6d = np.asarray(offset, dtype=np.float32).reshape(-1)
    if offset_6d.shape != (6,) or not np.isfinite(offset_6d).all():
        raise PhysicsProbeError("applied tip-target offset must be finite 6D")
    record = {
        "control_step": step,
        "timestamp_s": control_elapsed_s,
        "environment_episode_timestamp_s": environment_episode_timestamp_s,
        "phase": snapshot.phase,
        "policy_input_valid": valid_input,
        "observation_raw79": raw.tolist(),
        "observation_raw79_sha256": array_sha256(raw),
        "applied_tip_target_offset_entry_6d": offset_6d.tolist(),
        "actual_depth_m": snapshot.ground_truth.actual_depth_m,
        "actual_lateral_error_m": snapshot.ground_truth.lateral_error_m,
        "actual_orientation_error_rad": snapshot.ground_truth.orientation_error_rad,
        "grasp_both": snapshot.ground_truth.grasp_both,
        "forbidden_contact": snapshot.ground_truth.forbidden_contact,
        "grip_contact_force_peak_n": snapshot.grip_contact_force_peak_n,
        "insertion_contact_force_n": snapshot.insertion_contact_force_n,
        "insertion_contact_load_n": snapshot.insertion_contact_load_n,
        "forbidden_contact_load_n": snapshot.forbidden_contact_load_n,
        "forbidden_contact_load_source": snapshot.forbidden_contact_load_source,
        "forbidden_contact_load_definition": (
            snapshot.forbidden_contact_load_definition
        ),
    }
    record.update(action_provenance(submitted, requested_expanded, applied_expanded))
    return record


def _transition_contract(
    saved: Mapping[str, Any], observation: Any, done: bool
) -> dict[str, Any]:
    import numpy as np

    if bool(_scalar(saved["terminal_mask"])) != done:
        raise PhysicsProbeError("transition terminal mask differs from step result")
    saved_observation = np.asarray(
        _scalar(saved["policy_input_observation"]), dtype=np.float32
    )
    current_observation = observation.detach().cpu().numpy()[0]
    if saved_observation.shape != (OBSERVATION_DIM,) or not np.array_equal(
        saved_observation, current_observation
    ):
        raise PhysicsProbeError(
            "transition policy input differs from submitted observation"
        )
    valid_input = bool(_scalar(saved["policy_input_valid"]))
    return {
        "environment_timestamp_s": float(_scalar(saved["timestamp_s"])),
        "requested_expanded": _scalar(saved["requested_expanded_residual_action"]),
        "applied_expanded": _scalar(saved["applied_bounded_residual_action"]),
        "offset": _scalar(saved["applied_tip_target_offset_entry_6d"]),
        "valid_input": valid_input,
        "active": (
            bool(_scalar(saved["training_policy_activated"]))
            and int(_scalar(saved["valid_policy_input_step_count"])) > 0
            and valid_input
        ),
    }


def run_physics(args: argparse.Namespace, policy: CpuDiagnosticPolicy, offline: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    import torch
    from rby1_assembly.evaluation import EvaluationResult, TaskEvaluator
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    from rby1_assembly.held_grasp import (
        bank_source_contract_sha256,
        load_verified_held_grasp_bank,
    )
    from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg
    from rby1_assembly.recording import reconcile_terminal_evaluation
    from rby1_assembly.rj45_env import configure_rj45_task
    from rby1_assembly.rj45_task import load_rj45_task, verify_generated_asset_hashes, verify_runtime_versions
    from rby1_assembly.rj45_training import load_training_config
    from rby1_assembly.rj45_training_env import _calibration_bias_case_seed, make_rj45_training_env_type

    root = args.runtime_root
    task_path = _resolve(root, args.task_config)
    training_path = _resolve(root, args.training_config)
    runtime_path = _resolve(root, args.runtime_config)
    bank_path = _resolve(root, args.held_bank)
    task = load_rj45_task(task_path)
    training = load_training_config(training_path)
    runtime = _load_json(runtime_path)
    if file_sha256(training_path) != EXPECTED_TRAINING_SHA256:
        raise PhysicsProbeError("fixed target-offset training config SHA-256 mismatch")
    if training["base_observation_dim"] != OBSERVATION_DIM or training["action_mode"] != "right_only_6d":
        raise PhysicsProbeError("training observation/action contract changed")
    if not math.isclose(float(training["active_horizon_s"]), HORIZON_S, abs_tol=1e-12):
        raise PhysicsProbeError("training horizon must remain 24 seconds")
    current_versions = {
        name: metadata.version(name)
        for name in ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py")
    }
    verify_runtime_versions(current_versions, runtime)
    build_report_path = _resolve(root, task["assets"]["build_report"])
    verify_generated_asset_hashes(root, task, _load_json(build_report_path))
    bank, bank_proof = load_verified_held_grasp_bank(root, task, bank_path, "train", allow_final=False)
    identity = fixed_condition_identity(bank_proof)
    source_contract_sha = bank_source_contract_sha256(bank_proof)
    if source_contract_sha != EXPECTED_SOURCE_CONTRACT_SHA256:
        raise PhysicsProbeError("held bank source contract SHA-256 mismatch")
    verify_calibration_case_seed(_calibration_bias_case_seed)

    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, runtime)
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = BASE_SEED
    env_type = configure_rj45_task(cfg, task, root)
    env_type = make_rj45_training_env_type(
        env_type,
        training,
        held_bank=bank,
        held_bank_proof=bank_proof,
        calibration_bias_enabled=True,
        bias_level=training["calibration_bias"]["initial_level"],
    )
    geometry = task["frames"]
    limits = _limits(task, training)
    evaluator = TaskEvaluator(limits)
    trace_path = args.trace.resolve()
    if trace_path.exists():
        raise FileExistsError(f"trace already exists: {trace_path}")
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_paths = {
        task_path,
        training_path,
        runtime_path,
        bank_path,
        args.checkpoint.resolve(),
        args.offline_manifest.resolve(),
        args.offline_npz.resolve(),
        args.protocol.resolve(),
        Path(__file__).resolve(),
        (args.current_root / "scripts/run_rj45_g2_input_normalization_probe.py").resolve(),
    }
    source_hashes = {path: file_sha256(path) for path in frozen_paths}
    env = env_type(cfg)
    try:
        observations = advance_to_fixed_episode(env)
        if not math.isclose(float(env.cfg.sim.dt), 1.0 / PHYSICS_HZ, abs_tol=1e-12):
            raise PhysicsProbeError("environment physics rate changed")
        if int(env.cfg.decimation) != CONTROL_DECIMATION or not math.isclose(float(env.step_dt), 1.0 / CONTROL_HZ, abs_tol=1e-12):
            raise PhysicsProbeError("environment control rate changed")
        terminal = None
        terminal_snapshot = None
        independent_result: Any = EvaluationResult(False, False, 0.0, ())
        active_step_count = 0
        with trace_path.open("x", encoding="utf-8") as trace:
            warmup_step_limit = math.ceil(1.0 * CONTROL_HZ)
            active_step_limit = math.ceil(HORIZON_S * CONTROL_HZ)
            for step in range(1, warmup_step_limit + active_step_limit + 3):
                observation = _observation_tensor(observations)
                requested = policy.predict(observation.detach().cpu().numpy()[0])
                action_tensor = torch.as_tensor(requested, dtype=torch.float32, device=env.device)
                observations, _, terminated, truncated, extras = env.step(action_tensor)
                done = bool((terminated | truncated)[0].item())
                saved = extras.get("terminal_snapshot")
                if not isinstance(saved, Mapping):
                    raise PhysicsProbeError("per-step transition snapshot mapping is missing")
                transition = _transition_contract(saved, observation, done)
                snapshot = _probe_snapshot(env, saved, geometry)
                environment_timestamp_s = transition["environment_timestamp_s"]
                if transition["active"]:
                    active_step_count += 1
                    independent_result = evaluator.update(
                        _evaluation_sample(
                            snapshot,
                            saved,
                            active_step_count / CONTROL_HZ,
                            task["evaluation"],
                        )
                    )
                if done:
                    terminal_snapshot = snapshot
                record = _trace_record(
                    step,
                    observation,
                    requested,
                    transition["requested_expanded"],
                    transition["applied_expanded"],
                    transition["offset"],
                    snapshot,
                    step / CONTROL_HZ,
                    environment_timestamp_s,
                    transition["valid_input"],
                )
                trace.write(json.dumps(record, sort_keys=True) + "\n")
                trace.flush()
                if done:
                    if terminal_snapshot is None:
                        raise AssertionError("terminal snapshot unexpectedly missing")
                    terminal = reconcile_terminal_evaluation(
                        extras=extras,
                        snapshot=terminal_snapshot,
                        terminal_timestamp_s=environment_timestamp_s,
                        independent_result=independent_result,
                        task_limits=limits,
                        env_index=0,
                    )
                    break
                if independent_result.terminal:
                    raise PhysicsProbeError("independent evaluator terminated before environment evaluator")
            if terminal is None:
                raise PhysicsProbeError("environment did not terminate inside the frozen 24 second horizon")
            trace.write(json.dumps({"event": "complete", "classification": classify_terminal(terminal)}, sort_keys=True) + "\n")
    finally:
        env.close()
    end_bank, end_proof = load_verified_held_grasp_bank(root, task, bank_path, "train", allow_final=False)
    if len(end_bank) != len(bank) or fixed_condition_identity(end_proof) != identity:
        raise PhysicsProbeError("held bank identity changed during physics execution")
    for path, digest in source_hashes.items():
        if file_sha256(path) != digest:
            raise PhysicsProbeError(f"frozen input changed during execution: {path}")
    assert terminal is not None and terminal_snapshot is not None
    result = {
        "schema_name": SCHEMA_NAME,
        "status": classify_terminal(terminal),
        "policy_label": args.policy_label,
        "runtime_root": str(root),
        "runtime_versions": current_versions,
        "physics": {"physics_hz": PHYSICS_HZ, "control_decimation": CONTROL_DECIMATION, "control_hz": CONTROL_HZ, "horizon_s": HORIZON_S},
        "fixed_condition": identity,
        "policy": policy.identity,
        "offline_preflight": offline,
        "frozen_protocol": args.protocol_record,
        "frozen_input_sha256": {str(path): digest for path, digest in source_hashes.items()},
        "bank_provenance": {
            "source_contract_sha256": source_contract_sha,
            "selected_artifact_sha256": bank_proof.get("selected_artifact_sha256"),
        },
        "terminal_evaluation": asdict(terminal),
        "terminal_phase": terminal_snapshot.phase,
        "contact_source_proof": {
            "forbidden_contact_load_source": (
                terminal_snapshot.forbidden_contact_load_source
            ),
            "forbidden_contact_load_definition": (
                terminal_snapshot.forbidden_contact_load_definition
            ),
            "collision_filter": terminal_snapshot.collision_filter,
        },
        "trace": {"path": str(trace_path), "sha256": file_sha256(trace_path)},
        "claims": {"closed_loop_physics": True, "reinforcement_learning": False, "heldout_generalization": False, "single_fixed_condition_diagnostic": True},
        "inputs_unchanged": True,
    }
    return result


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--current-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--policy-label", choices=("a_standard_input", "b_normalized_input"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--offline-manifest", type=Path, required=True)
    parser.add_argument("--offline-npz", type=Path, required=True)
    parser.add_argument("--held-bank", type=Path, required=True)
    parser.add_argument("--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json"))
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/rj45-training-target-offset-v1.json"),
    )
    parser.add_argument("--runtime-config", type=Path, default=Path("configs/runtime.json"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _base_parser()
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    report_path = args.report.resolve()
    if report_path.exists():
        parser.error(f"report already exists: {report_path}")
    simulation_app = None
    policy = None
    offline = None
    failure_stage = "input_validation"
    try:
        observed_runner_sha = file_sha256(Path(__file__).resolve())
        if observed_runner_sha != args.runner_sha256:
            raise PhysicsProbeError("runner SHA-256 mismatch")
        args.protocol_record = validate_protocol(
            args.protocol.resolve(), args.protocol_sha256, args.runner_sha256
        )
        args.runtime_root = args.runtime_root.resolve()
        args.current_root = args.current_root.resolve()
        if not args.runtime_root.is_dir():
            raise PhysicsProbeError("--runtime-root must be a directory")
        sys.path.insert(0, str(args.runtime_root / "src"))
        observation, action, _ = _validated_offline_rows(
            args.offline_manifest.resolve(),
            args.offline_npz.resolve(),
            expected_npz_sha256=EXPECTED_NPZ_SHA256,
        )
        expected_checkpoint_sha = EXPECTED_POLICIES[args.policy_label][
            "checkpoint_file_sha256"
        ]
        if args.checkpoint_sha256 != expected_checkpoint_sha:
            raise PhysicsProbeError("policy label and checkpoint file SHA-256 do not match")
        if file_sha256(args.checkpoint.resolve()) != expected_checkpoint_sha:
            raise PhysicsProbeError("canonical diagnostic checkpoint SHA-256 mismatch")
        policy = CpuDiagnosticPolicy(
            args.checkpoint.resolve(), args.checkpoint_sha256, args.current_root
        )
        offline = offline_parity_preflight(policy, observation, action)
        validate_policy_preflight(args.policy_label, policy.identity, offline)
        failure_stage = "application_launch"
        launcher = AppLauncher(args, multi_gpu=False)
        simulation_app = launcher.app
        failure_stage = "closed_loop_physics"
        result = run_physics(args, policy, offline)
        failure_stage = "report_write"
        _atomic_write_new(report_path, result)
        print(json.dumps({"status": result["status"], "report": str(report_path)}, sort_keys=True))
        return 2 if result["status"] == "invalid" else 0
    except Exception as error:
        failure = {
            "schema_name": SCHEMA_NAME,
            "status": "invalid",
            "policy_label": args.policy_label,
            "runtime_root": str(args.runtime_root),
            "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "runner": {
                "expected_sha256": args.runner_sha256,
                "observed_sha256": file_sha256(Path(__file__).resolve()),
            },
            "frozen_protocol": getattr(
                args, "protocol_record", frozen_protocol_record(args.runner_sha256)
            ),
            "fixed_condition": {
                "base_seed": BASE_SEED,
                "environment_index": ENVIRONMENT_INDEX,
                "episode_index": EPISODE_INDEX,
                "calibration_case_seed": CALIBRATION_CASE_SEED,
                "grasp_index": GRASP_INDEX,
                "grasp_id": GRASP_ID,
                "bank_manifest_sha256": EXPECTED_BANK_SHA256,
            },
            "policy": policy.identity if policy is not None else None,
            "offline_preflight": offline,
            "partial_trace": (
                {
                    "path": str(args.trace.resolve()),
                    "sha256": file_sha256(args.trace.resolve()),
                }
                if args.trace.resolve().is_file()
                else None
            ),
            "claims": {
                "closed_loop_physics": False,
                "reinforcement_learning": False,
                "heldout_generalization": False,
                "single_fixed_condition_diagnostic": True,
            },
        }
        if not report_path.exists():
            _atomic_write_new(report_path, failure)
        print(json.dumps({"status": "invalid", "report": str(report_path), "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    finally:
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
