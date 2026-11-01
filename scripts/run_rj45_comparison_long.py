#!/usr/bin/env python3
"""Fail-closed G2 long PPO/SAC driver for the development-only RJ45 pilot."""

from __future__ import annotations

import argparse
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from types import SimpleNamespace
from typing import Any, Mapping, NoReturn


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ALGORITHMS = ("ppo", "sac")
EXECUTION_MODES = ("train", "evaluate-dev")
CONDITION_CASES = (
    "tx-positive",
    "tx-negative",
    "tz-positive",
    "tz-negative",
    "rx-positive",
    "rx-negative",
)
EXPECTED_GRASPS = {
    "rj45-train-02": (1, "0c0a21e53cb944460f3402b46598871933b8a89a1f61eebf4c15947bedbe5ff5"),
    "rj45-train-05": (4, "8ccca951e5deb7b515e394047b2cf3c645bac6b66eb46c2b531155c274d852d7"),
}
EXPECTED_DEV_GRASPS = {
    "rj45-dev-01": (0, "7fc4d78e011c4267fc1968a3e81b3d0281bf4c8d647ffa3823e7197836bec9b5"),
    "rj45-dev-02": (1, "0a7520631b095167764ed2443847c786ebabcfa9dd26a3ab666fd1cd4b4d5358"),
}
EXPECTED_BIASES = {
    "tx-positive": ([0.00012, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "tx-negative": ([-0.00012, 0.0, 0.0], [0.0, 0.0, 0.0]),
    "tz-positive": ([0.0, 0.0, 0.00012], [0.0, 0.0, 0.0]),
    "tz-negative": ([0.0, 0.0, -0.00012], [0.0, 0.0, 0.0]),
    "rx-positive": ([0.0, 0.0, 0.0], [0.003490658503988659, 0.0, 0.0]),
    "rx-negative": ([0.0, 0.0, 0.0], [-0.003490658503988659, 0.0, 0.0]),
}
RUNTIME_PYTHON_INPUT_COUNT = 35
PHYSICS_INPUT_COUNT = 13
ACTIVE_IMPORT_INPUT_COUNT = 2
EVIDENCE_SOURCE_INPUT_COUNT = 42
PARALLEL_PROOF_EXTRA_INPUT_COUNT = 5
ENVIRONMENT_RNG_ISOLATION_SCOPE = (
    "python_numpy_torch_cpu_and_cuda0_generator_state_isolated_at_reset_step_and_wave_reset;"
    "gpu_kernel_bit_exactness_not_claimed"
)


class LongDriverBlocked(RuntimeError):
    """The preregistered pilot contract or immutable evidence was not satisfied."""


def _load_module_path(path: Path, module_name: str):
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        _fail(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_current_module(name: str):
    path = Path(__file__).resolve().parents[1] / f"src/rby1_assembly/{name}.py"
    return _load_module_path(path, f"rby1_g2_current_{name}")


def resolve_environment_module_paths(
    *, runtime_root: Path, current_root: Path
) -> dict[str, Path]:
    """Resolve active base modules and current-only adapters without fallback."""

    runtime_package = runtime_root.resolve() / "src/rby1_assembly"
    current_package = current_root.resolve() / "src/rby1_assembly"
    required_runtime = {
        name: runtime_package / f"{name}.py"
        for name in (
            "rj45_task",
            "rj45_env",
            "rj45_training",
            "rj45_training_env",
            "parallel_contact_evidence",
        )
    }
    current_adapters = {
        name: current_package / f"{name}.py"
        for name in ("rj45_comparison_env", "rj45_comparison_pilot_env")
    }
    for label, path in {**required_runtime, **current_adapters}.items():
        if not path.is_file():
            _fail(f"required environment module is missing: {label}")
    if (runtime_package / "rj45_comparison_env.py").exists():
        _fail("comparison adapter must be loaded from the reviewed current root")
    return {**required_runtime, **current_adapters}


def _fail(message: str) -> NoReturn:
    raise LongDriverBlocked(message)


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: str | Path, label: str) -> dict[str, Any]:
    def reject(value: str) -> None:
        _fail(f"{label} contains non-finite constant {value}")

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)
        json.dumps(value, allow_nan=False)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise LongDriverBlocked(f"cannot load {label}: {error}") from error
    if not isinstance(value, dict):
        _fail(f"{label} root must be an object")
    return value


def _verified_hash_map(
    root: Path, expected: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, digest in sorted(expected.items()):
        expected_sha = _sha(digest, f"{label}.{relative}")
        path = _resolve_under(root, relative, f"{label}.{relative}")
        if not path.is_file() or file_sha256(path) != expected_sha:
            _fail(f"{label} SHA-256 mismatch: {relative}")
        observed[str(relative)] = expected_sha
    return observed


def validate_source_closure(
    contract: Mapping[str, Any], *, current_root: Path
) -> dict[str, dict[str, str]]:
    """Verify the complete imported runtime, physics, and evidence source closure."""

    inputs = contract["inputs"]
    runtime_root = Path(inputs["runtime_root"]).resolve()
    evidence_root = Path(inputs["evidence_root"]).resolve()
    runtime_manifest_path = _resolve_under(
        current_root, inputs.get("runtime_manifest_path"), "runtime manifest"
    )
    physics_manifest_path = _resolve_under(
        current_root, inputs.get("physics_input_manifest_path"), "physics input manifest"
    )
    evidence_manifest_path = _resolve_under(
        evidence_root, inputs.get("evidence_manifest_path"), "evidence manifest"
    )
    for path, field, label in (
        (runtime_manifest_path, "runtime_manifest_sha256", "runtime manifest"),
        (physics_manifest_path, "physics_input_manifest_sha256", "physics input manifest"),
        (evidence_manifest_path, "evidence_manifest_sha256", "evidence manifest"),
    ):
        if not path.is_file() or file_sha256(path) != _sha(inputs.get(field), label):
            _fail(f"{label} SHA-256 mismatch")

    runtime_manifest = load_json(runtime_manifest_path, "runtime manifest")
    evidence_manifest = load_json(evidence_manifest_path, "evidence manifest")
    runtime_python = runtime_manifest.get("files")
    evidence_python = evidence_manifest.get("files")
    if not isinstance(runtime_python, Mapping) or len(runtime_python) != RUNTIME_PYTHON_INPUT_COUNT:
        _fail("runtime manifest must bind exactly 35 Python files")
    if not isinstance(evidence_python, Mapping) or len(evidence_python) != RUNTIME_PYTHON_INPUT_COUNT:
        _fail("evidence manifest must bind exactly 35 Python files")
    actual_runtime_python = {
        str(path.relative_to(runtime_root))
        for path in (runtime_root / "src/rby1_assembly").rglob("*.py")
        if path.is_file()
    }
    actual_evidence_python = {
        str(path.relative_to(evidence_root))
        for path in (evidence_root / "src/rby1_assembly").rglob("*.py")
        if path.is_file()
    }
    if actual_runtime_python != set(runtime_python):
        _fail("runtime Python set differs from the exact 35-file manifest")
    if actual_evidence_python != set(evidence_python):
        _fail("evidence Python set differs from the exact 35-file manifest")

    physics_overlay = load_json(physics_manifest_path, "physics input manifest")
    base_spec = physics_overlay.get("base_physics_manifest")
    if not isinstance(base_spec, Mapping):
        _fail("physics input manifest lacks its base manifest binding")
    base_path = _resolve_under(
        current_root, base_spec.get("relative_path"), "base physics manifest"
    )
    if not base_path.is_file() or file_sha256(base_path) != _sha(
        base_spec.get("sha256"), "base physics manifest SHA"
    ):
        _fail("base physics manifest SHA-256 mismatch")
    physics = {**load_json(base_path, "base physics manifest"), **physics_overlay}
    core_spec = physics.get("runtime_core_manifest")
    evidence_spec = physics.get("evidence_core_manifest")
    if (
        not isinstance(core_spec, Mapping)
        or not isinstance(evidence_spec, Mapping)
        or core_spec.get("relative_path") != inputs.get("runtime_manifest_path")
        or core_spec.get("sha256") != inputs.get("runtime_manifest_sha256")
        or evidence_spec.get("relative_path") != inputs.get("evidence_manifest_path")
        or evidence_spec.get("sha256") != inputs.get("evidence_manifest_sha256")
    ):
        _fail("physics manifest source-manifest bindings differ from the contract")
    physics_inputs = physics.get("physics_inputs")
    active_imports = physics.get("active_import_inputs")
    evidence_configs = physics.get("evidence_source_config_inputs")
    if not isinstance(physics_inputs, Mapping) or len(physics_inputs) != PHYSICS_INPUT_COUNT:
        _fail("physics manifest must bind exactly 13 inputs")
    if not isinstance(active_imports, Mapping) or len(active_imports) != ACTIVE_IMPORT_INPUT_COUNT:
        _fail("physics manifest must bind exactly two active import inputs")
    if not isinstance(evidence_configs, Mapping) or len(evidence_configs) != 7:
        _fail("physics manifest must bind exactly seven evidence config inputs")

    runtime_map = {
        **_verified_hash_map(runtime_root, runtime_python, label="runtime Python input"),
        **_verified_hash_map(runtime_root, physics_inputs, label="runtime physics input"),
        **_verified_hash_map(runtime_root, active_imports, label="runtime import input"),
    }
    evidence_source = {**evidence_python, **evidence_configs}
    if len(evidence_source) != EVIDENCE_SOURCE_INPUT_COUNT:
        _fail("evidence source union must contain exactly 42 files")
    shared_catalog = physics.get("shared_bank_catalog_relative")
    if shared_catalog not in evidence_source:
        _fail("physics manifest omits the shared bank catalog binding")
    evidence_map = {
        **_verified_hash_map(
            evidence_root,
            {key: value for key, value in evidence_source.items() if key != shared_catalog},
            label="evidence source input",
        ),
        **_verified_hash_map(evidence_root, physics_inputs, label="evidence physics input"),
    }
    if len(runtime_map) != 50 or len(evidence_map) != 50:
        _fail("runtime/evidence normalized source closure must contain 50 files each")
    return {"runtime": runtime_map, "evidence": evidence_map}


def validate_import_origins(modules: Mapping[str, Any], runtime_root: Path) -> dict[str, str]:
    package_root = (runtime_root / "src/rby1_assembly").resolve()
    observed: dict[str, str] = {}
    for name, module in modules.items():
        raw = getattr(module, "__file__", None)
        if not isinstance(raw, str):
            _fail(f"runtime import has no file origin: {name}")
        path = Path(raw).resolve()
        if not path.is_relative_to(package_root) or path.name != f"{name}.py":
            _fail(f"runtime import escaped the active root: {name}")
        observed[name] = str(path)
    return observed


def validate_split_environment_inputs(
    *,
    runtime_root: Path,
    evidence_root: Path,
    development_evidence_root: Path,
    task_relative: str,
    training_relative: str,
    training_loader: Any,
) -> dict[str, Any]:
    """Require train and dev to differ only in bank selection and condition grid."""

    roots = (runtime_root, evidence_root, development_evidence_root)
    records: dict[str, Any] = {}
    for relative, label in (
        ("configs/runtime.json", "runtime config"),
        (task_relative, "task config"),
        (training_relative, "training config"),
    ):
        paths = [root / relative for root in roots]
        raw = [path.read_bytes() for path in paths]
        digests = [file_sha256(path) for path in paths]
        parsed = [load_json(path, label) for path in paths]
        if len(set(raw)) != 1 or len(set(digests)) != 1 or parsed[1:] != parsed[:1] * 2:
            _fail(f"train/dev {label} bytes, SHA, and parsed values must be identical")
        records[label.replace(" ", "_")] = {
            "sha256": digests[0],
            "relative_path": relative,
        }
    normalized_training = [training_loader(root / training_relative) for root in roots]
    if normalized_training[1:] != normalized_training[:1] * 2:
        _fail("train/dev normalized training configuration must be identical")
    records["normalized_training_sha256"] = canonical_sha256(normalized_training[0])
    return records


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA_RE.fullmatch(value) is None:
        _fail(f"{label} must be a lowercase SHA-256")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(f"{label} must be an integer >= {minimum}")
    return value


def _resolve_under(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        _fail(f"{label} must be a non-empty path")
    candidate = Path(value)
    path = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        _fail(f"{label} escapes its immutable root")
    return path


def build_fixed_conditions(
    contract: Mapping[str, Any], *, split: str = "train"
) -> list[dict[str, Any]]:
    if split == "train":
        source = contract
        expected_grasps = EXPECTED_GRASPS
        expected_cases = list(CONDITION_CASES)
    elif split == "dev":
        source = contract.get("development_evaluation")
        if not isinstance(source, Mapping):
            _fail("development evaluation subcontract is missing")
        expected_grasps = EXPECTED_DEV_GRASPS
        expected_cases = [f"dev-{name}" for name in CONDITION_CASES]
    else:
        _fail("condition split must be train or dev")
    grasps = source.get("held_grasps")
    cases = source.get("bias_cases")
    if not isinstance(grasps, list) or [row.get("grasp_id") for row in grasps] != list(
        expected_grasps
    ):
        _fail(f"held_grasps must be the ordered reviewed {split} pair")
    if not isinstance(cases, list) or [row.get("case_id") for row in cases] != expected_cases:
        _fail(f"bias_cases must be the exact ordered signed-six {split} cases")
    conditions: list[dict[str, Any]] = []
    for grasp in grasps:
        grasp_index = _integer(grasp.get("held_grasp_index"), "held_grasp_index")
        grasp_sha = _sha(grasp.get("held_artifact_sha256"), "held_artifact_sha256")
        if (grasp_index, grasp_sha) != expected_grasps[grasp["grasp_id"]]:
            _fail(f"held grasp index/SHA differs from the reviewed {split} pair")
        for case in cases:
            for key in ("translation_bias_entry_m", "rotation_vector_bias_entry_rad"):
                vector = case.get(key)
                if not isinstance(vector, list) or len(vector) != 3:
                    _fail(f"{key} must contain exactly three values")
                if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in vector):
                    _fail(f"{key} must be numeric")
            base_case_id = case["case_id"].removeprefix("dev-")
            expected_translation, expected_rotation = EXPECTED_BIASES[base_case_id]
            if (
                case["translation_bias_entry_m"] != expected_translation
                or case["rotation_vector_bias_entry_rad"] != expected_rotation
            ):
                _fail("pilot bias array differs from the reviewed development cases")
            conditions.append(
                {
                    "condition_id": f"{grasp['grasp_id']}:{case['case_id']}",
                    "grasp_id": grasp["grasp_id"],
                    "held_grasp_index": grasp["held_grasp_index"],
                    "held_artifact_sha256": grasp["held_artifact_sha256"],
                    "translation_bias_entry_m": case["translation_bias_entry_m"],
                    "rotation_vector_bias_entry_rad": case[
                        "rotation_vector_bias_entry_rad"
                    ],
                }
            )
    if len(conditions) != 12:
        _fail("pilot schedule must contain exactly 12 conditions")
    return conditions


def _namespace_seed(seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"rj45-g2:{seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_paired_schedule(contract: Mapping[str, Any]) -> dict[str, Any]:
    seeds = contract.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        _fail("frozen pilot must contain exactly three seeds")
    normalized = [_integer(seed, "seed") for seed in seeds]
    if len(set(normalized)) != len(normalized):
        _fail("pilot seeds must be unique")
    conditions = build_fixed_conditions(contract, split="train")
    dev_conditions = build_fixed_conditions(contract, split="dev")
    if {row["grasp_id"] for row in conditions} & {
        row["grasp_id"] for row in dev_conditions
    }:
        _fail("training and development grasp IDs must be disjoint")
    rows = []
    for seed in normalized:
        namespaces = {
            name: _namespace_seed(seed, name)
            for name in (
                "model",
                "schedule",
                "environment",
                "development_environment",
            )
        }
        condition_order = [row["condition_id"] for row in conditions]
        condition_hash = canonical_sha256(condition_order)
        dev_condition_order = [row["condition_id"] for row in dev_conditions]
        dev_condition_hash = canonical_sha256(dev_condition_order)
        for algorithm in ALGORITHMS:
            rows.append(
                {
                    "algorithm": algorithm,
                    "seed": seed,
                    "model_seed": _namespace_seed(namespaces["model"], algorithm),
                    "schedule_seed": namespaces["schedule"],
                    "environment_seed": namespaces["environment"],
                    "development_environment_seed": namespaces[
                        "development_environment"
                    ],
                    "condition_order": condition_order,
                    "condition_schedule_sha256": condition_hash,
                    "development_condition_order": dev_condition_order,
                    "development_condition_schedule_sha256": dev_condition_hash,
                }
            )
    return {
        "conditions": conditions,
        "development_conditions": dev_conditions,
        "runs": rows,
        "sha256": canonical_sha256(rows),
    }


def _expected_g1_formal_plan(development: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = [row["case_id"] for row in development["bias_cases"]]
    tuning = [
        ("tuning", "rj45-train-03", case, "classical_force_search", 1, candidate["id"])
        for candidate in development["classical_tuning"]["candidates"]
        for case in cases
    ]
    methods = [
        (method["id"], int(method["slots_per_case"]))
        for method in development["methods"]
    ]
    admission = [
        ("admission", grasp["grasp_id"], case, method, slot, None)
        for grasp in development["held_grasps"]
        for case in cases
        for method, slots in methods
        for slot in range(1, slots + 1)
    ]
    if len(tuning) != 18 or len(admission) != 72:
        _fail("authenticated G1 config does not define tuning18 plus admission72")
    return [
        {
            "ordinal": ordinal,
            "stage": row[0],
            "grasp_id": row[1],
            "case_id": row[2],
            "method_id": row[3],
            "slot": row[4],
            "candidate_id": row[5],
        }
        for ordinal, row in enumerate(tuning + admission, start=1)
    ]


def _expected_classical_controller_sha256(
    development: Mapping[str, Any], candidate_id: str
) -> str:
    candidates = {
        row["id"]: row for row in development["classical_tuning"]["candidates"]
    }
    candidate = candidates.get(candidate_id)
    if not isinstance(candidate, Mapping):
        _fail("G1 selected classical candidate is absent from the frozen config")
    payload = {
        "schema": "rj45-classical-force-search-v1",
        "candidate_id": candidate_id,
        "position_gain": candidate["position_gain"],
        "rotation_gain": candidate["rotation_gain"],
        "force_trigger_n": candidate["force_trigger_n"],
        "search_step_m": candidate["search_step_m"],
        "stall_ticks": 6,
        "stall_depth_delta_per_tick_m": 0.00002,
        "controlled_translation_axes": ["x", "z"],
        "controlled_rotation_axes": ["x", "z"],
        "lateral_force_direction_threshold_n": 0.25,
        "ticks_per_search_direction": 6,
        "search_direction_order": [
            "initial", "rotate_plus_90", "opposite", "rotate_minus_90"
        ],
        "search_radius_max_m": 0.0006,
        "advance_recovery_ticks": 3,
        "zero_action_entry_force_norm_n": 20.0,
        "translation_bound_m": 0.0015,
        "hard_stop_evaluator_force_n": 25.0,
        "rotation_bound_rad": 0.026179938779914945,
        "final_component_clip": [-1.0, 1.0],
        "known_bias_access": False,
        "observation_dim": 79,
        "action_dim": 6,
        "entry_force_slice": [37, 40],
        "entry_force_scale_n": 8.0,
        "command_depth_index": 62,
        "command_depth_scale_m": 0.012,
        "advance_index": 63,
    }
    return canonical_sha256(payload)


def validate_g1_gate(
    contract: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    current_root: Path | None = None,
    development: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gate = contract.get("g1_gate")
    if not isinstance(gate, Mapping):
        _fail("g1_gate is missing")
    required = gate.get("required_claims")
    if (
        gate.get("required_status") != "admit_rl_pilot"
        or gate.get("required_smoke_only") is not False
        or required
        != {
            "task_admission_completed": True,
            "rl_pilot_admitted": True,
        }
    ):
        _fail("G1 gate target is not the immutable RL admission contract")
    if report.get("status") != "admit_rl_pilot":
        _fail("G1 report status does not admit the RL pilot")
    schema = report.get("schema_name")
    if schema != "rj45-task-admission-isolated-report-v1":
        _fail("unsupported or missing G1 report schema")
    if current_root is None or development is None:
        _fail("isolated G1 report requires its frozen campaign contract")
    if schema == "rj45-task-admission-isolated-report-v1":
        campaign_path = _resolve_under(
            current_root, gate.get("campaign_path"), "G1 isolated campaign"
        )
        campaign_sha = _sha(gate.get("campaign_sha256"), "G1 campaign SHA")
        if not campaign_path.is_file() or file_sha256(campaign_path) != campaign_sha:
            _fail("G1 campaign SHA-256 mismatch")
        identity = report.get("campaign_identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("sha256") != campaign_sha
            or not isinstance(identity.get("path"), str)
            or not identity.get("path")
        ):
            _fail("G1 report campaign identity mismatch")
        campaign = load_json(campaign_path, "G1 isolated campaign")
        expected_plan = _expected_g1_formal_plan(development)
        expected_partitions = [
            list(range(start, end + 1))
            for start, end in (
                (1, 6), (7, 12), (13, 18), (19, 30), (31, 42),
                (43, 54), (55, 66), (67, 78), (79, 90),
            )
        ]
        if (
            campaign.get("schema_name")
            != "rj45-task-admission-isolated-campaign-v1"
            or campaign.get("purpose") != "formal_campaign"
            or campaign.get("environment_seed") != 20261121
            or campaign.get("execution_plan") != expected_plan
            or campaign.get("formal_partitions") != expected_partitions
        ):
            _fail("G1 campaign is not the exact formal tuning18 plus admission72 plan")
        campaign_config = campaign.get("config")
        if (
            not isinstance(campaign_config, Mapping)
            or campaign_config.get("sha256")
            != file_sha256(current_root / contract["inputs"]["development_config_path"])
        ):
            _fail("G1 campaign config identity differs from the frozen development config")
        episodes = report.get("episodes")
        if not isinstance(episodes, list) or len(episodes) != 90:
            _fail("G1 isolated formal report must contain exactly 90 episodes")
        raw_selection = report.get("tuning_selection")
        selected_candidate = (
            raw_selection.get("candidate_id")
            if isinstance(raw_selection, Mapping)
            else None
        )
        identity_keys = ("stage", "grasp_id", "case_id", "method_id", "slot")
        for expected, episode in zip(expected_plan, episodes, strict=True):
            if not isinstance(episode, Mapping) or any(
                episode.get(key) != expected[key] for key in identity_keys
            ):
                _fail(f"G1 episode identity mismatch at ordinal {expected['ordinal']}")
            expected_candidate = expected["candidate_id"]
            if expected["stage"] == "admission" and expected["method_id"] == "classical_force_search":
                expected_candidate = selected_candidate
            if episode.get("candidate_id") != expected_candidate:
                _fail(f"G1 episode candidate identity mismatch at ordinal {expected['ordinal']}")
        producer_claims = report.get("claims")
        if producer_claims != {
            "task_admission_completed": True,
            "rl_pilot_admitted": True,
            "native_crash_cause_fixed": False,
        }:
            _fail("G1 isolated formal report claims do not admit the RL pilot")
        selection = raw_selection
        seal = report.get("classical_seal")
        selection_keys = {
            "candidate_id",
            "success_count",
            "evaluator_force_p95_n",
            "completion_time_median_s",
            "met_target",
        }
        if (
            not isinstance(selection, Mapping)
            or not isinstance(seal, Mapping)
            or set(selection) != selection_keys
            or set(seal)
            != {
                "selected_candidate_id",
                "canonical_controller_config_sha256",
                "tuning_report_sha256",
                "selection",
            }
            or selection.get("candidate_id") != seal.get("selected_candidate_id")
            or selection.get("candidate_id") not in {"A", "B", "C"}
            or seal.get("selection") != selection
            or not SHA_RE.fullmatch(
                str(seal.get("canonical_controller_config_sha256", ""))
            )
            or seal.get("canonical_controller_config_sha256")
            != _expected_classical_controller_sha256(
                development, str(selection.get("candidate_id", ""))
            )
            or not SHA_RE.fullmatch(str(seal.get("tuning_report_sha256", "")))
            or type(selection.get("success_count")) is not int
            or not 0 <= selection["success_count"] <= 6
            or isinstance(selection.get("evaluator_force_p95_n"), bool)
            or not isinstance(selection.get("evaluator_force_p95_n"), (int, float))
            or not math.isfinite(float(selection["evaluator_force_p95_n"]))
            or isinstance(selection.get("completion_time_median_s"), bool)
            or not isinstance(selection.get("completion_time_median_s"), (int, float))
            or not math.isfinite(float(selection["completion_time_median_s"]))
            or type(selection.get("met_target")) is not bool
        ):
            _fail("G1 isolated formal tuning seal is invalid")
        decision = report.get("decision")
        if not isinstance(decision, Mapping) or decision.get("state") != report["status"]:
            _fail("G1 isolated decision differs from report status")
        normalized_claims = dict(required)
        return {
            "schema_name": "rj45_g1_gate_normalized_v1",
            "source_report_schema": schema,
            "status": report["status"],
            "non_smoke": True,
            "claims": normalized_claims,
            "scientific_scope_claims": {
                "physical_initial_pose_error_sweep": False,
                "success_range_curve_proven": False,
            },
            "producer_claims": dict(producer_claims),
            "campaign": {
                "frozen_path": str(campaign_path),
                "reported_source_path": identity["path"],
                "sha256": campaign_sha,
                "purpose": campaign["purpose"],
                "environment_seed": campaign["environment_seed"],
                "episode_count": 90,
                "tuning_episode_count": 18,
                "admission_episode_count": 72,
            },
        }
    raise AssertionError("unreachable G1 schema branch")


def validate_development_config_binding(
    contract: Mapping[str, Any], development: Mapping[str, Any]
) -> None:
    expected_grasps = [
        {
            "grasp_id": row["grasp_id"],
            "verified_index": row["held_grasp_index"],
            "artifact_sha256": row["held_artifact_sha256"],
        }
        for row in contract["held_grasps"]
    ]
    if development.get("held_grasps") != expected_grasps:
        _fail("pilot train grasp contract differs from the authenticated G1 config")
    expected_cases = [
        {
            "case_id": row["case_id"],
            "translation_bias_entry_m": row["translation_bias_entry_m"],
            "rotation_vector_bias_entry_rad": row["rotation_vector_bias_entry_rad"],
        }
        for row in contract["bias_cases"]
    ]
    actual_cases = [
        {
            "case_id": row.get("case_id"),
            "translation_bias_entry_m": row.get("translation_bias_entry_m"),
            "rotation_vector_bias_entry_rad": row.get(
                "rotation_vector_bias_entry_rad"
            ),
        }
        for row in development.get("bias_cases", [])
    ]
    if actual_cases != expected_cases:
        _fail("pilot train bias grid differs from the authenticated G1 config")


def validate_parallel_contact_proof(proof: Mapping[str, Any], num_envs: int) -> None:
    if proof.get("validated_probe_num_envs") != num_envs:
        _fail("parallel-contact proof does not validate the exact num_envs")
    if num_envs > 16:
        _fail("pilot num_envs exceeds the reviewed parallel-contact ceiling of 16")


def validate_contract(
    contract: Mapping[str, Any], *, execution_mode: str
) -> dict[str, Any]:
    if contract.get("schema_name") != "rj45_comparison_pilot_v1":
        _fail("unexpected comparison pilot schema")
    if execution_mode not in EXECUTION_MODES:
        _fail("execution mode must be train or evaluate-dev")
    if execution_mode in set(contract.get("forbidden_execution_modes", [])):
        _fail("final evaluation mode is sealed and forbidden in G2")
    if contract.get("final_split_allowed") is not False:
        _fail("G2 contract must keep the final split sealed")
    if contract.get("frozen") is not True:
        _fail("scientific pilot contract is not frozen")
    if contract.get("resume_supported") is not False:
        _fail("training resume must remain explicitly unsupported")
    if contract.get("retry_policy") != {
        "maximum_full_seed_retries": 1,
        "allowed_reasons": ["infrastructure_failure", "provenance_failure"],
        "low_success_rate_retry_allowed": False,
        "resume_partial_seed_allowed": False,
    }:
        _fail("retry policy must allow only one full-seed infrastructure/provenance retry")
    initialization = contract.get("policy_initialization")
    if not isinstance(initialization, Mapping) or initialization.get(
        "mean_initializer"
    ) != "zero":
        _fail("policy mean initializer must preserve the reviewed zero baseline")
    initial_log_std = initialization.get("initial_log_std")
    if (
        isinstance(initial_log_std, bool)
        or not isinstance(initial_log_std, (int, float))
        or not -5.0 <= float(initial_log_std) <= 1.0
    ):
        _fail("initial_log_std must be frozen within the policy bounds")
    scope = contract.get("scientific_scope")
    if not isinstance(scope, Mapping) or (
        scope.get("actor_observation_dim"),
        scope.get("critic_observation_dim"),
        scope.get("residual_action_dim"),
        scope.get("left_arm"),
        scope.get("environment"),
        scope.get("ground_truth_policy_fallback"),
    ) != (79, 79, 6, "scripted_support", "simulation_proxy", False):
        _fail("scientific scope differs from 79D/6D simulation-only policy rights")
    fairness = contract.get("fairness")
    if not isinstance(fairness, Mapping) or (
        fairness.get("primary_budget"),
        fairness.get("primary_budget_formula"),
        fairness.get("raw_equals_learnable_claim_allowed"),
        fairness.get("same_valid_learning_efficiency_claim_allowed"),
        fairness.get("success_contingent_stopping"),
        fairness.get("paired_algorithm_condition_schedule"),
        fairness.get("paired_environment_seeds"),
        fairness.get("separate_rng_namespaces"),
        fairness.get("environment_rng_isolation_scope"),
    ) != (
        "exact_raw_environment_control_transitions",
        "num_envs_times_wave_count_times_cycles_times_wave_vector_steps",
        False,
        False,
        False,
        True,
        True,
        ["model", "schedule", "environment", "development_environment"],
        ENVIRONMENT_RNG_ISOLATION_SCOPE,
    ):
        _fail("fairness and RNG namespace contract differs from preregistration")
    budget = contract.get("budget")
    if not isinstance(budget, Mapping):
        _fail("budget is missing")
    num_envs = _integer(budget.get("num_envs"), "num_envs", minimum=1)
    raw_budget = _integer(
        budget.get("raw_transition_budget"), "raw_transition_budget", minimum=1
    )
    vector_steps = _integer(budget.get("vector_steps"), "vector_steps", minimum=1)
    wave_count = _integer(budget.get("wave_count"), "wave_count", minimum=1)
    cycles = _integer(budget.get("cycles"), "cycles", minimum=1)
    wave_vector_steps = _integer(
        budget.get("wave_vector_steps"), "wave_vector_steps", minimum=1
    )
    if (
        num_envs != 4
        or wave_count != 3
        or vector_steps != wave_count * cycles * wave_vector_steps
        or num_envs * vector_steps != raw_budget
    ):
        _fail("raw transition budget must equal exact 4 env * 3 waves * cycles * wave steps")
    raw_upper_bound = _integer(
        budget.get("raw_transition_budget_upper_bound"),
        "raw_transition_budget_upper_bound",
        minimum=1,
    )
    if raw_budget > raw_upper_bound:
        _fail("raw transition budget exceeds its preregistered upper bound")
    rollout_target = _integer(
        budget.get("ppo_rollout_target_valid_transitions"),
        "ppo_rollout_target_valid_transitions",
        minimum=1,
    )
    if rollout_target > raw_budget or rollout_target % num_envs:
        _fail("PPO rollout target must be preregistered, <= raw budget, and divisible by 4")
    maximum_tail_fraction = budget.get("maximum_ppo_wave_tail_fraction")
    if (
        isinstance(maximum_tail_fraction, bool)
        or not isinstance(maximum_tail_fraction, (int, float))
        or not 0.0 < float(maximum_tail_fraction) <= 1.0
        or rollout_target / (num_envs * wave_vector_steps)
        > float(maximum_tail_fraction)
    ):
        _fail("PPO wave-tail upper bound is missing or exceeded")
    replay_capacity = _integer(
        budget.get("sac_replay_capacity"), "sac_replay_capacity", minimum=1
    )
    batch_size = _integer(budget.get("sac_batch_size"), "sac_batch_size", minimum=1)
    _integer(
        budget.get("sac_random_warmup_valid_transitions"),
        "sac_random_warmup_valid_transitions",
        minimum=0,
    )
    learning_starts = _integer(
        budget.get("sac_learning_starts_valid_transitions"),
        "sac_learning_starts_valid_transitions",
        minimum=0,
    )
    if replay_capacity < batch_size or learning_starts < batch_size:
        _fail("SAC replay/learning-start contract is smaller than its batch size")
    utd = budget.get("sac_utd_ratio")
    if isinstance(utd, bool) or not isinstance(utd, (int, float)) or float(utd) <= 0:
        _fail("sac_utd_ratio must be frozen and positive")
    if list(contract.get("algorithms", [])) != list(ALGORITHMS):
        _fail("algorithm order must be exactly PPO then SAC")
    schedule = build_paired_schedule(contract)
    minimums = contract.get("minimum_outcome_contract")
    if not isinstance(minimums, Mapping):
        _fail("minimum outcome contract is missing")
    for key in (
        "minimum_valid_policy_transitions",
        "minimum_algorithm_updates",
        "minimum_actual_update_input_transitions",
    ):
        _integer(minimums.get(key), key, minimum=1)
    minimum_condition_count = _integer(
        minimums.get("minimum_condition_transition_count_per_condition"),
        "minimum_condition_transition_count_per_condition",
        minimum=1,
    )
    if minimum_condition_count != cycles:
        _fail("each training condition must reach transition once per frozen cycle")
    if (
        minimums.get("require_all_12_conditions_reach_align_to_insert") is not True
        or minimums.get("require_all_transition_actions_exact_zero") is not True
        or minimums.get("low_reward_or_success_rate_is_not_an_invalidity") is not True
    ):
        _fail("minimum outcome gates must cover fixed conditions without reward retry")
    dev = contract.get("development_evaluation")
    if not isinstance(dev, Mapping) or dev.get("split") != "dev" or dev.get(
        "allow_final"
    ) is not False:
        _fail("development evaluation must use only the dev split")
    dev_raw = _integer(dev.get("raw_transition_budget"), "dev raw budget", minimum=1)
    dev_steps = _integer(dev.get("vector_steps"), "dev vector_steps", minimum=1)
    dev_cycles = _integer(dev.get("cycles"), "dev cycles", minimum=1)
    dev_wave_steps = _integer(
        dev.get("wave_vector_steps"), "dev wave_vector_steps", minimum=1
    )
    if (
        dev_steps != wave_count * dev_cycles * dev_wave_steps
        or dev_raw != num_envs * dev_steps
    ):
        _fail("development raw budget must equal exact fixed-wave quota")
    bindings = contract.get("source_and_config_sha256")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "runtime",
        "evidence",
        "development_evidence",
        "current",
    }:
        _fail("frozen contract must declare runtime/evidence/dev/current source inputs")
    inputs = contract.get("inputs")
    if not isinstance(inputs, Mapping):
        _fail("input path contract is missing")
    required_bindings = {
        "runtime": {
            "configs/runtime.json",
            inputs.get("task_config_path"),
            inputs.get("parallel_contact_proof_path"),
            "src/rby1_assembly/experiment_contracts.py",
            "src/rby1_assembly/held_grasp.py",
            "src/rby1_assembly/isaac_env.py",
            "src/rby1_assembly/rj45_env.py",
            "src/rby1_assembly/rj45_task.py",
            "src/rby1_assembly/rj45_training.py",
            "src/rby1_assembly/rj45_training_env.py",
            "src/rby1_assembly/parallel_contact_evidence.py",
        },
        "evidence": {
            inputs.get("task_config_path"),
            inputs.get("training_config_path"),
            inputs.get("held_bank_path"),
        },
        "development_evidence": {
            inputs.get("task_config_path"),
            inputs.get("training_config_path"),
            dev.get("held_bank_path"),
        },
        "current": {
            contract.get("g1_gate", {}).get("report_path"),
            inputs.get("development_config_path"),
            inputs.get("runtime_manifest_path"),
            inputs.get("physics_input_manifest_path"),
            "scripts/run_rj45_comparison_long.py",
            "src/rby1_assembly/rj45_comparison_rl.py",
            "src/rby1_assembly/rj45_comparison_env.py",
            "src/rby1_assembly/rj45_comparison_pilot_env.py",
        },
    }
    for namespace, required in required_bindings.items():
        if None in required or not required.issubset(bindings[namespace]):
            _fail(f"{namespace} source bindings omit a required execution input")
    for field in (
        "runtime_manifest_sha256",
        "physics_input_manifest_sha256",
        "evidence_manifest_sha256",
    ):
        _sha(inputs.get(field), field)
    return {
        "budget": dict(budget),
        "development_budget": {
            "raw_transition_budget": dev_raw,
            "vector_steps": dev_steps,
            "cycles": dev_cycles,
            "wave_vector_steps": dev_wave_steps,
            "wave_vector_steps_plan": tuple(
                dev_wave_steps for _ in range(wave_count * dev_cycles)
            ),
        },
        "policy_initialization": dict(initialization),
        "minimums": dict(minimums),
        "wave_vector_steps_plan": tuple(
            wave_vector_steps for _ in range(wave_count * cycles)
        ),
        "schedule": schedule,
    }


def capture_input_snapshot(
    contract: Mapping[str, Any], *, current_root: Path
) -> dict[str, Any]:
    inputs = contract.get("inputs")
    bindings = contract.get("source_and_config_sha256")
    if not isinstance(inputs, Mapping) or not isinstance(bindings, Mapping):
        _fail("input roots or bindings are missing")
    roots = {
        "runtime": Path(str(inputs.get("runtime_root", ""))).resolve(),
        "evidence": Path(str(inputs.get("evidence_root", ""))).resolve(),
        "development_evidence": Path(
            str(inputs.get("development_evidence_root", ""))
        ).resolve(),
        "current": current_root.resolve(),
    }
    if len(set(roots.values())) != 4:
        _fail("runtime, evidence, development evidence, and current roots must be distinct")
    normalized: dict[str, dict[str, str]] = {}
    for namespace, root in roots.items():
        declared = bindings.get(namespace)
        if not isinstance(declared, Mapping) or not declared:
            _fail(f"{namespace} source binding must be a non-empty object")
        actual: dict[str, str] = {}
        for relative, expected in declared.items():
            expected_sha = _sha(expected, f"{namespace}.{relative}")
            path = _resolve_under(root, relative, f"{namespace}.{relative}")
            if not path.is_file() or file_sha256(path) != expected_sha:
                _fail(f"source input mismatch: {namespace}.{relative}")
            actual[str(relative)] = expected_sha
        normalized[namespace] = actual
    source_manifests = {
        "runtime_core": {
            "namespace": "current",
            "relative_path": inputs["runtime_manifest_path"],
            "sha256": _sha(inputs["runtime_manifest_sha256"], "runtime manifest SHA"),
        },
        "physics_inputs": {
            "namespace": "current",
            "relative_path": inputs["physics_input_manifest_path"],
            "sha256": _sha(
                inputs["physics_input_manifest_sha256"], "physics manifest SHA"
            ),
        },
        "evidence_core": {
            "namespace": "evidence",
            "relative_path": inputs["evidence_manifest_path"],
            "sha256": _sha(inputs["evidence_manifest_sha256"], "evidence manifest SHA"),
        },
    }
    physics_manifest_path = _resolve_under(
        roots["current"], inputs["physics_input_manifest_path"], "physics input manifest"
    )
    base_spec = load_json(physics_manifest_path, "physics input manifest").get(
        "base_physics_manifest"
    )
    if not isinstance(base_spec, Mapping):
        _fail("physics input manifest lacks its base manifest binding")
    source_manifests["base_physics"] = {
        "namespace": "current",
        "relative_path": base_spec.get("relative_path"),
        "sha256": _sha(base_spec.get("sha256"), "base physics manifest SHA"),
    }
    for binding in source_manifests.values():
        path = _resolve_under(
            roots[binding["namespace"]],
            binding["relative_path"],
            "source manifest",
        )
        if not path.is_file() or file_sha256(path) != binding["sha256"]:
            _fail("source manifest binding mismatch")
    return {
        "roots": {key: str(value) for key, value in roots.items()},
        "normalized_sha256": normalized,
        "source_manifests": source_manifests,
        "canonical_sha256": canonical_sha256(
            {"normalized_sha256": normalized, "source_manifests": source_manifests}
        ),
    }


def verify_input_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    roots = snapshot.get("roots")
    normalized = snapshot.get("normalized_sha256")
    if not isinstance(roots, Mapping) or not isinstance(normalized, Mapping):
        _fail("start input snapshot is malformed")
    errors: list[str] = []
    for namespace, declared in normalized.items():
        root = Path(str(roots.get(namespace, ""))).resolve()
        if not isinstance(declared, Mapping):
            _fail("start input namespace is malformed")
        for relative, expected in declared.items():
            try:
                path = _resolve_under(root, relative, f"{namespace}.{relative}")
                if not path.is_file() or file_sha256(path) != expected:
                    errors.append(f"{namespace}.{relative}")
            except LongDriverBlocked:
                errors.append(f"{namespace}.{relative}")
    source_manifests = snapshot.get("source_manifests")
    if not isinstance(source_manifests, Mapping) or set(source_manifests) != {
        "runtime_core",
        "physics_inputs",
        "evidence_core",
        "base_physics",
    }:
        _fail("start input snapshot source manifests are malformed")
    for role, binding in source_manifests.items():
        if not isinstance(binding, Mapping):
            _fail("start input snapshot source manifest binding is malformed")
        namespace = binding.get("namespace")
        path = _resolve_under(
            Path(str(roots.get(namespace, ""))).resolve(),
            binding.get("relative_path"),
            f"source manifest {role}",
        )
        if not path.is_file() or file_sha256(path) != binding.get("sha256"):
            errors.append(f"source_manifest.{role}")
    if errors:
        _fail(f"source inputs changed after start: {sorted(errors)}")
    canonical = canonical_sha256(
        {"normalized_sha256": normalized, "source_manifests": source_manifests}
    )
    if canonical != snapshot.get("canonical_sha256"):
        _fail("start input snapshot canonical SHA mismatch")
    return {
        "matched": True,
        "start_canonical_sha256": snapshot.get("canonical_sha256"),
        "end_canonical_sha256": canonical,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    _atomic_write_bytes(path, payload)


def write_checkpoint_bundle(
    agent,
    checkpoint_path: Path,
    *,
    algorithm: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    save_inference_checkpoint = _load_current_module(
        "rj45_comparison_rl"
    ).save_inference_checkpoint

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=checkpoint_path.parent, prefix=f".{checkpoint_path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        save_inference_checkpoint(agent, temporary, algorithm=algorithm)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        checkpoint_sha = file_sha256(temporary)
        os.replace(temporary, checkpoint_path)
        temporary = None
        directory_fd = os.open(checkpoint_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        if file_sha256(checkpoint_path) != checkpoint_sha:
            _fail("checkpoint SHA changed during atomic rename")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    sidecar = {
        "schema_name": "rj45_comparison_checkpoint_sidecar_v1",
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": checkpoint_sha,
        "resume_supported": False,
        "algorithm": algorithm,
        "observation_dim": 79,
        "action_dim": 6,
        **dict(metadata),
    }
    sidecar_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".json")
    atomic_write_json(sidecar_path, sidecar)
    return {"checkpoint": checkpoint_path, "sidecar": sidecar_path, "metadata": sidecar}


def verify_checkpoint_bundle(
    checkpoint_path: Path, sidecar_path: Path, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    sidecar = load_json(sidecar_path, "checkpoint sidecar")
    if sidecar.get("checkpoint_path") != checkpoint_path.name:
        _fail("checkpoint sidecar path mismatch")
    if sidecar.get("checkpoint_sha256") != file_sha256(checkpoint_path):
        _fail("checkpoint binary SHA-256 mismatch")
    for key, value in expected.items():
        if sidecar.get(key) != value:
            _fail(f"checkpoint sidecar mismatch: {key}")
    if sidecar.get("resume_supported") is not False:
        _fail("checkpoint sidecar must reject resume")
    return sidecar


def write_run_manifest(
    path: Path,
    *,
    prepared: Mapping[str, Any],
    run: Mapping[str, Any],
    checkpoint_path: Path,
    sidecar_path: Path,
    source_input_end_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema_name": "rj45_comparison_frozen_run_manifest_v1",
        "expected": _checkpoint_metadata(prepared, run),
        "source_input_end_sha256": source_input_end_sha256,
        "checkpoint_path": checkpoint_path.name,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "sidecar_path": sidecar_path.name,
        "sidecar_sha256": file_sha256(sidecar_path),
    }
    atomic_write_json(path, payload)
    return payload


def verify_run_manifest(
    path: Path,
    manifest_sha256: str,
    *,
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    if file_sha256(path) != _sha(manifest_sha256, "run manifest SHA"):
        _fail("run manifest SHA-256 mismatch")
    manifest = load_json(path, "frozen run manifest")
    if manifest.get("schema_name") != "rj45_comparison_frozen_run_manifest_v1":
        _fail("unexpected frozen run manifest schema")
    expected = manifest.get("expected")
    if not isinstance(expected, Mapping):
        _fail("run manifest expected identity is missing")
    matches = [
        run
        for run in prepared["validated"]["schedule"]["runs"]
        if run["algorithm"] == expected.get("algorithm")
        and run["seed"] == expected.get("seed")
    ]
    if len(matches) != 1 or dict(expected) != _checkpoint_metadata(prepared, matches[0]):
        _fail("run manifest identity differs from the frozen contract")
    if manifest.get("source_input_end_sha256") != prepared["input_snapshot"][
        "canonical_sha256"
    ]:
        _fail("run manifest source end identity differs from current frozen inputs")
    checkpoint_path = path.parent / str(manifest.get("checkpoint_path", ""))
    sidecar_path = path.parent / str(manifest.get("sidecar_path", ""))
    if (
        not checkpoint_path.is_file()
        or file_sha256(checkpoint_path) != manifest.get("checkpoint_sha256")
        or not sidecar_path.is_file()
        or file_sha256(sidecar_path) != manifest.get("sidecar_sha256")
    ):
        _fail("run manifest artifact hashes do not match")
    sidecar_expected = {
        **dict(expected),
        "source_input_end_sha256": manifest["source_input_end_sha256"],
        "terminal_checkpoint_policy": "last_completed_update",
        "diagnostic_best_promoted": False,
    }
    sidecar = verify_checkpoint_bundle(
        checkpoint_path, sidecar_path, expected=sidecar_expected
    )
    return {
        "manifest": manifest,
        "run": matches[0],
        "checkpoint": checkpoint_path,
        "sidecar": sidecar_path,
        "sidecar_payload": sidecar,
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    contract_path = args.contract.resolve()
    expected_sha = _sha(args.contract_sha256, "--contract-sha256")
    actual_sha = file_sha256(contract_path)
    if actual_sha != expected_sha:
        _fail("contract SHA-256 mismatch")
    contract = load_json(contract_path, "pilot contract")
    validated = validate_contract(contract, execution_mode=args.execution_mode)
    gate = contract["g1_gate"]
    g1_path = _resolve_under(root, gate.get("report_path"), "g1 report")
    if file_sha256(g1_path) != _sha(gate.get("report_sha256"), "g1 report SHA"):
        _fail("G1 report SHA-256 mismatch")
    g1_report = load_json(g1_path, "G1 report")
    inputs = contract["inputs"]
    development_config_path = _resolve_under(
        root, inputs.get("development_config_path"), "G1 development config"
    )
    development_config = load_json(development_config_path, "G1 development config")
    validate_development_config_binding(contract, development_config)
    g1_scope = validate_g1_gate(
        contract,
        g1_report,
        current_root=root,
        development=development_config,
    )
    if g1_scope["campaign"] is not None and gate.get("campaign_path") not in contract[
        "source_and_config_sha256"
    ]["current"]:
        _fail("isolated G1 campaign is absent from the start/end source binding")
    runtime_root = Path(inputs["runtime_root"]).resolve()
    evidence_root = Path(inputs["evidence_root"]).resolve()
    development_evidence_root = Path(inputs["development_evidence_root"]).resolve()
    proof_path = _resolve_under(
        runtime_root, inputs.get("parallel_contact_proof_path"), "parallel contact proof"
    )
    if file_sha256(proof_path) != _sha(
        inputs.get("parallel_contact_proof_sha256"), "parallel contact proof SHA"
    ):
        _fail("parallel contact proof SHA-256 mismatch")
    module_paths = resolve_environment_module_paths(
        runtime_root=runtime_root, current_root=root
    )
    source_closure = validate_source_closure(contract, current_root=root)
    training_module = _load_module_path(
        module_paths["rj45_training"], "rby1_g2_active_rj45_training_contract"
    )
    split_environment_inputs = validate_split_environment_inputs(
        runtime_root=runtime_root,
        evidence_root=evidence_root,
        development_evidence_root=development_evidence_root,
        task_relative=inputs["task_config_path"],
        training_relative=inputs["training_config_path"],
        training_loader=training_module.load_training_config,
    )
    parallel_module = _load_module_path(
        module_paths["parallel_contact_evidence"],
        "rby1_g2_active_parallel_contact_evidence",
    )
    parallel_spec = load_json(proof_path, "parallel contact evidence specification")
    parallel_runtime_paths = {
        inputs["parallel_contact_proof_path"],
        parallel_spec.get("report_path"),
        parallel_spec.get("trace_path"),
        parallel_spec.get("sat_report_path"),
        "checks/probe_parallel_hand_contacts.py",
    }
    if None in parallel_runtime_paths or len(parallel_runtime_paths) != PARALLEL_PROOF_EXTRA_INPUT_COUNT:
        _fail("parallel proof closure must contain exactly five extra inputs")
    parallel_map = _verified_hash_map(
        runtime_root,
        {
            path: contract["source_and_config_sha256"]["runtime"].get(path)
            for path in parallel_runtime_paths
        },
        label="parallel proof input",
    )
    expected_runtime_map = {**source_closure["runtime"], **parallel_map}
    if contract["source_and_config_sha256"]["runtime"] != expected_runtime_map:
        _fail("runtime source binding differs from the full 35+13+2+5 closure")
    if contract["source_and_config_sha256"]["evidence"] != source_closure["evidence"]:
        _fail("evidence source binding differs from the authenticated 50-file closure")
    parallel_proof = parallel_module.load_parallel_contact_evidence(
        parallel_spec, project_root=runtime_root
    )
    validate_parallel_contact_proof(parallel_proof, validated["budget"]["num_envs"])
    snapshot = capture_input_snapshot(contract, current_root=root)
    train_bank_path = evidence_root / inputs["held_bank_path"]
    dev_bank_path = (
        development_evidence_root
        / contract["development_evaluation"]["held_bank_path"]
    )
    return {
        "root": root,
        "contract": contract,
        "contract_path": contract_path,
        "contract_sha256": actual_sha,
        "g1_path": g1_path,
        "g1_sha256": file_sha256(g1_path),
        "g1_report": g1_report,
        "g1_scope": g1_scope,
        "development_config_path": development_config_path,
        "development_config_sha256": file_sha256(development_config_path),
        "held_bank_sha256": file_sha256(train_bank_path),
        "development_held_bank_sha256": file_sha256(dev_bank_path),
        "parallel_contact_specification": parallel_spec,
        "parallel_contact_proof": parallel_proof,
        "source_closure": source_closure,
        "split_environment_inputs": split_environment_inputs,
        "validated": validated,
        "input_snapshot": snapshot,
    }


def _build_env(
    prepared: Mapping[str, Any], run: Mapping[str, Any], *, split: str = "train"
):
    """Build base -> training -> comparison -> fixed 12-condition wave pilot."""

    contract = prepared["contract"]
    inputs = contract["inputs"]
    runtime_root = Path(inputs["runtime_root"]).resolve()
    evidence_root = Path(
        inputs["evidence_root"]
        if split == "train"
        else inputs["development_evidence_root"]
    ).resolve()
    module_paths = resolve_environment_module_paths(
        runtime_root=runtime_root, current_root=prepared["root"]
    )
    if str(runtime_root / "src") not in sys.path:
        sys.path.insert(0, str(runtime_root / "src"))
    from rby1_assembly import experiment_contracts
    from rby1_assembly import held_grasp
    from rby1_assembly import isaac_env
    from rby1_assembly import rj45_env
    from rby1_assembly import rj45_task
    from rby1_assembly import rj45_training
    from rby1_assembly import rj45_training_env

    import_origins = validate_import_origins(
        {
            "experiment_contracts": experiment_contracts,
            "held_grasp": held_grasp,
            "isaac_env": isaac_env,
            "rj45_env": rj45_env,
            "rj45_task": rj45_task,
            "rj45_training": rj45_training,
            "rj45_training_env": rj45_training_env,
        },
        runtime_root,
    )
    apply_runtime_timing = experiment_contracts.apply_runtime_timing
    load_verified_held_grasp_bank = held_grasp.load_verified_held_grasp_bank
    Rby1AssemblyEnvCfg = isaac_env.Rby1AssemblyEnvCfg
    configure_rj45_task = rj45_env.configure_rj45_task
    load_rj45_task = rj45_task.load_rj45_task
    load_training_config = rj45_training.load_training_config
    make_rj45_training_env_type = rj45_training_env.make_rj45_training_env_type

    make_rj45_comparison_env_type = _load_module_path(
        module_paths["rj45_comparison_env"], "rby1_g2_current_rj45_comparison_env"
    ).make_rj45_comparison_env_type
    make_rj45_comparison_pilot_env_type = _load_current_module(
        "rj45_comparison_pilot_env"
    ).make_rj45_comparison_pilot_env_type

    active_task_path = runtime_root / inputs["task_config_path"]
    evidence_task_path = evidence_root / inputs["task_config_path"]
    task = load_rj45_task(active_task_path)
    evidence_task = load_rj45_task(evidence_task_path)
    if (
        active_task_path.read_bytes() != evidence_task_path.read_bytes()
        or task != evidence_task
        or task.get("_config_sha256") != file_sha256(active_task_path)
    ):
        _fail("active and selected-split task configs differ or lack authenticated SHA")
    training_path = evidence_root / inputs["training_config_path"]
    training = load_training_config(training_path)
    expected_training_sha = prepared["split_environment_inputs"]["training_config"][
        "sha256"
    ]
    if (
        file_sha256(training_path) != expected_training_sha
        or training.get("_config_sha256") != expected_training_sha
        or canonical_sha256(training)
        != prepared["split_environment_inputs"]["normalized_training_sha256"]
    ):
        _fail("selected split training config differs from the frozen common config")
    split_contract = (
        contract if split == "train" else contract["development_evaluation"]
    )
    bank_path = evidence_root / split_contract["held_bank_path"]
    bank, proof = load_verified_held_grasp_bank(
        evidence_root, evidence_task, bank_path, split, allow_final=False
    )
    selected_indices = [
        row["held_grasp_index"] for row in split_contract["held_grasps"]
    ]
    selected_bank = [bank[index] for index in selected_indices]
    selected_sha = [proof["selected_artifact_sha256"][index] for index in selected_indices]
    selected_ids = [proof["selected_grasp_ids"][index] for index in selected_indices]
    expected_sha = [
        row["held_artifact_sha256"] for row in split_contract["held_grasps"]
    ]
    if selected_sha != expected_sha:
        _fail("held bank selected artifact SHA mismatch")
    selected_proof = dict(proof)
    selected_proof.update(
        selected_count=2,
        selected_grasp_ids=selected_ids,
        selected_artifact_sha256=selected_sha,
    )
    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, load_json(runtime_root / "configs/runtime.json", "runtime config"))
    cfg.scene.num_envs = prepared["validated"]["budget"]["num_envs"]
    cfg.sim.device = "cuda:0"
    cfg.seed = run[
        "environment_seed" if split == "train" else "development_environment_seed"
    ]
    cfg.rj45_parallel_contact_evidence = prepared["parallel_contact_specification"]
    env_type = configure_rj45_task(cfg, task, runtime_root)
    env_type = make_rj45_training_env_type(
        env_type,
        training,
        held_bank=selected_bank,
        held_bank_proof=selected_proof,
        calibration_bias_enabled=True,
        bias_level=training["calibration_bias"]["initial_level"],
    )
    env_type = make_rj45_comparison_env_type(env_type)
    env_type = make_rj45_comparison_pilot_env_type(
        env_type,
        conditions=prepared["validated"]["schedule"][
            "conditions" if split == "train" else "development_conditions"
        ],
    )
    env = env_type(cfg)
    env.rj45_comparison_runtime_import_origins = dict(import_origins)
    return env


def _checkpoint_metadata(prepared: Mapping[str, Any], run: Mapping[str, Any]) -> dict[str, Any]:
    contract = prepared["contract"]
    inputs = contract["inputs"]
    return {
        "contract_path": str(prepared["contract_path"]),
        "contract_sha256": prepared["contract_sha256"],
        "g1_report_path": str(prepared["g1_path"]),
        "g1_report_sha256": prepared["g1_sha256"],
        "g1_status": prepared["g1_scope"]["status"],
        "g1_non_smoke": prepared["g1_scope"]["non_smoke"],
        "g1_claims": dict(prepared["g1_scope"]["claims"]),
        "g1_gate_scope": dict(prepared["g1_scope"]),
        "g1_source_report_schema": prepared["g1_scope"]["source_report_schema"],
        "g1_development_config_path": str(prepared["development_config_path"]),
        "g1_development_config_sha256": prepared["development_config_sha256"],
        "source_input_start_sha256": prepared["input_snapshot"]["canonical_sha256"],
        "source_and_config_sha256": prepared["input_snapshot"]["normalized_sha256"],
        "runtime_root": inputs["runtime_root"],
        "evidence_root": inputs["evidence_root"],
        "development_evidence_root": inputs["development_evidence_root"],
        "held_bank_path": inputs["held_bank_path"],
        "held_bank_sha256": prepared["held_bank_sha256"],
        "development_held_bank_path": contract["development_evaluation"][
            "held_bank_path"
        ],
        "development_held_bank_sha256": prepared[
            "development_held_bank_sha256"
        ],
        "condition_schedule_sha256": run["condition_schedule_sha256"],
        "development_condition_schedule_sha256": run[
            "development_condition_schedule_sha256"
        ],
        "paired_schedule_sha256": prepared["validated"]["schedule"]["sha256"],
        "algorithm": run["algorithm"],
        "seed": run["seed"],
        "model_seed": run["model_seed"],
        "schedule_seed": run["schedule_seed"],
        "environment_seed": run["environment_seed"],
        "development_environment_seed": run["development_environment_seed"],
        "skrl_version": "2.1.0",
        "sidecar_schema_version": 1,
        "mean_initializer": prepared["validated"]["policy_initialization"][
            "mean_initializer"
        ],
        "initial_log_std": prepared["validated"]["policy_initialization"][
            "initial_log_std"
        ],
    }


def validate_run_outcome(progress: Any, pilot_progress: Any, minimums: Mapping[str, Any]) -> list[str]:
    """Return preregistered scientific invalidities without using reward outcomes."""

    failures: list[str] = []
    if progress.valid_policy_transitions < minimums["minimum_valid_policy_transitions"]:
        failures.append("minimum_valid_policy_transitions")
    if progress.algorithm_updates < minimums["minimum_algorithm_updates"]:
        failures.append("minimum_algorithm_updates")
    if (
        progress.actual_update_input_transitions
        < minimums["minimum_actual_update_input_transitions"]
    ):
        failures.append("minimum_actual_update_input_transitions")
    if len(pilot_progress.condition_ids) != 12 or not all(
        pilot_progress.transition_ever_checked
    ):
        failures.append("all_12_conditions_reach_align_to_insert")
    if not all(pilot_progress.transition_ever_exact_zero):
        failures.append("all_transition_actions_exact_zero")
    if any(
        count < minimums["minimum_condition_transition_count_per_condition"]
        for count in pilot_progress.transition_check_counts
    ):
        failures.append("minimum_condition_transition_count_per_condition")
    return failures


def environment_rng_evidence(progress: Any) -> dict[str, Any]:
    fingerprints = tuple(progress.reset_rng_fingerprints)
    if not fingerprints or not SHA_RE.fullmatch(progress.initial_observation_sha256):
        _fail("pilot environment lacks initial RNG/observation evidence")
    if not SHA_RE.fullmatch(progress.environment_rng_fingerprint):
        _fail("pilot environment RNG fingerprint is invalid")
    return {
        "initial_reset_rng_fingerprint": fingerprints[0],
        "initial_observation_sha256": progress.initial_observation_sha256,
        "rng_isolation_scope": progress.rng_isolation_scope,
    }


def ppo_rollout_steps(budget: Mapping[str, Any]) -> int:
    return int(budget["ppo_rollout_target_valid_transitions"]) // int(
        budget["num_envs"]
    )


def build_wave_accounting(
    result: Any,
    final_progress: Any,
    *,
    wave_vector_steps_plan: tuple[int, ...],
    num_envs: int,
) -> list[dict[str, Any]]:
    """Reconcile exact raw quotas and cumulative learning counters per wave."""

    points = list(result.wave_boundaries) + [
        SimpleNamespace(
            raw_transitions=final_progress.raw_transitions,
            valid_policy_transitions=final_progress.valid_policy_transitions,
            algorithm_updates=final_progress.algorithm_updates,
            actual_update_input_transitions=final_progress.actual_update_input_transitions,
            pending_policy_transitions=final_progress.pending_policy_transitions,
        )
    ]
    if len(points) != len(wave_vector_steps_plan):
        _fail("collector wave-boundary count differs from the frozen plan")
    previous = {
        "raw_transitions": 0,
        "valid_policy_transitions": 0,
        "algorithm_updates": 0,
        "actual_update_input_transitions": 0,
    }
    rows: list[dict[str, Any]] = []
    for index, (point, planned_steps) in enumerate(
        zip(points, wave_vector_steps_plan, strict=True)
    ):
        row = {
            "sequence_index": index,
            "cycle_index": index // 3,
            "catalog_wave_index": index % 3,
            "planned_vector_steps": planned_steps,
            "raw_transitions": point.raw_transitions - previous["raw_transitions"],
            "valid_policy_transitions": (
                point.valid_policy_transitions
                - previous["valid_policy_transitions"]
            ),
            "algorithm_updates": (
                point.algorithm_updates - previous["algorithm_updates"]
            ),
            "actual_update_input_transitions": (
                point.actual_update_input_transitions
                - previous["actual_update_input_transitions"]
            ),
            "cumulative_pending_policy_transitions": (
                point.pending_policy_transitions
            ),
        }
        if row["raw_transitions"] != planned_steps * num_envs:
            _fail("collector wave raw quota differs from preregistration")
        rows.append(row)
        for key in previous:
            previous[key] = getattr(point, key)
    return rows


def _train(prepared: Mapping[str, Any], output_dir: Path) -> dict[str, Any]:
    comparison_rl = _load_current_module("rj45_comparison_rl")
    collect_bounded = comparison_rl.collect_bounded
    collector_progress = comparison_rl.collector_progress
    discard_pending_ppo_tail = comparison_rl.discard_pending_ppo_tail
    make_agent = comparison_rl.make_agent

    budget = prepared["validated"]["budget"]
    results: list[dict[str, Any]] = []
    paired_environment_evidence: dict[int, dict[str, Any]] = {}
    paired_wave_rng_evidence: dict[int, tuple[str, ...]] = {}
    for run in prepared["validated"]["schedule"]["runs"]:
        env = _build_env(prepared, run)
        try:
            algorithm = run["algorithm"]
            kwargs = {
                "algorithm": algorithm,
                "num_envs": budget["num_envs"],
                "device": "cuda:0",
                "rollout_steps": ppo_rollout_steps(budget),
                "batch_size": budget["sac_batch_size"],
                "seed": run["model_seed"],
                "mean_initializer": prepared["validated"]["policy_initialization"][
                    "mean_initializer"
                ],
                "initial_log_std": prepared["validated"]["policy_initialization"][
                    "initial_log_std"
                ],
            }
            if algorithm == "sac":
                kwargs.update(
                    replay_capacity=budget["sac_replay_capacity"],
                    sac_utd_ratio=budget["sac_utd_ratio"],
                    sac_random_warmup_valid_transitions=budget[
                        "sac_random_warmup_valid_transitions"
                    ],
                    sac_learning_starts_valid_transitions=budget[
                        "sac_learning_starts_valid_transitions"
                    ],
                )
            agent = make_agent(**kwargs)
            initial = env.reset()
            initial_environment_evidence = environment_rng_evidence(
                env.comparison_pilot_progress()
            )
            prior_environment_evidence = paired_environment_evidence.setdefault(
                run["seed"], initial_environment_evidence
            )
            if prior_environment_evidence != initial_environment_evidence:
                _fail("paired PPO/SAC initial environment RNG evidence differs")
            started = time.monotonic()
            last_progress = None
            progress_callbacks = 0

            def record_progress(progress):
                nonlocal last_progress, progress_callbacks
                if last_progress is not None and progress.raw_transitions <= last_progress.raw_transitions:
                    _fail("collector progress raw transitions are not monotonic")
                last_progress = progress
                progress_callbacks += 1

            if comparison_rl.torch.cuda.is_available():
                comparison_rl.torch.cuda.reset_peak_memory_stats()
            result = collect_bounded(
                agent,
                env,
                algorithm=algorithm,
                vector_steps=budget["vector_steps"],
                require_policy_validity=True,
                initial_reset_result=initial,
                progress_callback=record_progress,
                wave_vector_steps=prepared["validated"]["wave_vector_steps_plan"],
                wave_reset_callback=env.advance_comparison_pilot_wave,
            )
            if result.transitions != budget["raw_transition_budget"]:
                _fail("collector did not consume the exact primary raw budget")
            if algorithm == "ppo":
                discard_pending_ppo_tail(agent)
            progress = collector_progress(agent)
            if last_progress is None or last_progress.raw_transitions != progress.raw_transitions:
                _fail("collector progress callback did not reach the terminal raw budget")
            end_gate = verify_input_snapshot(prepared["input_snapshot"])
            name = f"{algorithm}-seed-{run['seed']}"
            pilot_progress = env.comparison_pilot_progress()
            wave_rng_evidence = tuple(pilot_progress.reset_rng_fingerprints)
            if len(wave_rng_evidence) != len(
                prepared["validated"]["wave_vector_steps_plan"]
            ):
                _fail("pilot reset RNG evidence count differs from the wave plan")
            prior_wave_rng_evidence = paired_wave_rng_evidence.setdefault(
                run["seed"], wave_rng_evidence
            )
            if prior_wave_rng_evidence != wave_rng_evidence:
                _fail("paired PPO/SAC wave reset RNG fingerprints differ")
            wave_accounting = build_wave_accounting(
                result,
                progress,
                wave_vector_steps_plan=prepared["validated"][
                    "wave_vector_steps_plan"
                ],
                num_envs=budget["num_envs"],
            )
            invalidities = validate_run_outcome(
                progress, pilot_progress, prepared["validated"]["minimums"]
            )
            collector_telemetry = dict(result.__dict__)
            collector_telemetry.pop("learning_transitions", None)
            collector_telemetry["wave_boundaries"] = [
                row.__dict__ for row in result.wave_boundaries
            ]
            common_result = {
                "run": dict(run),
                "collector_result": collector_telemetry,
                "collector_counter_semantics": {
                    "valid_policy_transitions": "valid_exposure_not_learned_input",
                    "actual_update_input_transitions": "inputs_consumed_by_completed_updates",
                    "warmup_transitions": "invalid_or_prepolicy_exposure",
                    "pending_policy_transitions": "unupdated_ppo_tail_before_discard",
                    "discarded_tail_valid_transitions": "terminal_ppo_tail_not_updated",
                    "sac_update_credit": "fractional_cumulative_update_credit",
                },
                "collector_progress": progress.__dict__,
                "pilot_environment_progress": pilot_progress.__dict__,
                "initial_environment_rng_evidence": initial_environment_evidence,
                "wave_reset_rng_fingerprints": wave_rng_evidence,
                "wave_accounting": wave_accounting,
                "wall_seconds": time.monotonic() - started,
                "gpu_peak_memory_bytes": (
                    comparison_rl.torch.cuda.max_memory_allocated()
                    if comparison_rl.torch.cuda.is_available()
                    else None
                ),
                "progress_callback_count": progress_callbacks,
                "retry_count": 0,
            }
            if invalidities:
                results.append(
                    {
                        **common_result,
                        "status": "invalid_minimum_outcome_contract",
                        "invalidities": invalidities,
                        "checkpoint_promoted": False,
                        "checkpoint": None,
                        "sidecar": None,
                        "run_manifest": None,
                    }
                )
                continue
            bundle = write_checkpoint_bundle(
                agent,
                output_dir / f"{name}.pt",
                algorithm=algorithm,
                metadata={
                    **_checkpoint_metadata(prepared, run),
                    "source_input_end_sha256": end_gate["end_canonical_sha256"],
                    "terminal_checkpoint_policy": "last_completed_update",
                    "diagnostic_best_promoted": False,
                },
            )
            manifest_path = output_dir / f"{name}.run.json"
            write_run_manifest(
                manifest_path,
                prepared=prepared,
                run=run,
                checkpoint_path=bundle["checkpoint"],
                sidecar_path=bundle["sidecar"],
                source_input_end_sha256=end_gate["end_canonical_sha256"],
            )
            results.append(
                {
                    **common_result,
                    "status": "valid_training_run",
                    "invalidities": [],
                    "checkpoint_promoted": True,
                    "checkpoint": str(bundle["checkpoint"]),
                    "sidecar": str(bundle["sidecar"]),
                    "run_manifest": str(manifest_path),
                    "run_manifest_sha256": file_sha256(manifest_path),
                }
            )
        finally:
            env.close()
    return {
        "mode": "train",
        "status": (
            "valid" if all(row["status"] == "valid_training_run" for row in results) else "invalid"
        ),
        "runs": results,
    }


def _evaluate_dev(
    prepared: Mapping[str, Any], run_manifest: Path, run_manifest_sha256: str
) -> dict[str, Any]:
    comparison_rl = _load_current_module("rj45_comparison_rl")
    deterministic_action = comparison_rl.deterministic_action
    load_inference_checkpoint = comparison_rl.load_inference_checkpoint
    make_agent = comparison_rl.make_agent

    verified_manifest = verify_run_manifest(
        run_manifest,
        run_manifest_sha256,
        prepared=prepared,
    )
    metadata = verified_manifest["sidecar_payload"]
    run = verified_manifest["run"]
    checkpoint = verified_manifest["checkpoint"]
    budget = prepared["validated"]["budget"]
    dev_budget = prepared["validated"]["development_budget"]
    agent = make_agent(
        metadata["algorithm"],
        num_envs=budget["num_envs"],
        device="cuda:0",
        rollout_steps=ppo_rollout_steps(budget),
        batch_size=budget["sac_batch_size"],
        seed=run["model_seed"],
        mean_initializer=prepared["validated"]["policy_initialization"][
            "mean_initializer"
        ],
        initial_log_std=prepared["validated"]["policy_initialization"][
            "initial_log_std"
        ],
    )
    load_inference_checkpoint(agent, checkpoint, resume=False)
    env = _build_env(prepared, run, split="dev")
    try:
        observation, infos = env.reset()
        initial_environment_evidence = environment_rng_evidence(
            env.comparison_pilot_progress()
        )
        policy = observation["policy"] if isinstance(observation, dict) else observation
        raw_transitions = valid_transitions = 0
        wave_accounting = []
        for wave_index, wave_steps in enumerate(dev_budget["wave_vector_steps_plan"]):
            wave_valid = 0
            for _ in range(wave_steps):
                validity = infos.get("policy_observation_valid")
                if validity is None:
                    _fail("development evaluation requires policy validity telemetry")
                wave_valid += int(validity.sum().item())
                actions = deterministic_action(agent, policy)
                next_observation, _reward, _terminated, _truncated, infos = env.step(
                    actions
                )
                policy = (
                    next_observation["policy"]
                    if isinstance(next_observation, dict)
                    else next_observation
                )
            wave_raw = wave_steps * env.num_envs
            raw_transitions += wave_raw
            valid_transitions += wave_valid
            wave_accounting.append(
                {
                    "sequence_index": wave_index,
                    "cycle_index": wave_index // 3,
                    "catalog_wave_index": wave_index % 3,
                    "planned_vector_steps": wave_steps,
                    "raw_transitions": wave_raw,
                    "valid_policy_transitions": wave_valid,
                }
            )
            if wave_index + 1 < len(dev_budget["wave_vector_steps_plan"]):
                observation, infos = env.advance_comparison_pilot_wave(wave_index + 1)
                policy = (
                    observation["policy"]
                    if isinstance(observation, dict)
                    else observation
                )
        if raw_transitions != dev_budget["raw_transition_budget"]:
            _fail("development evaluation did not consume the exact raw budget")
        pilot_progress = env.comparison_pilot_progress()
        if not all(pilot_progress.transition_ever_checked) or not all(
            pilot_progress.transition_ever_exact_zero
        ) or any(
            count < dev_budget["cycles"]
            for count in pilot_progress.transition_check_counts
        ):
            _fail("development evaluation did not satisfy all fixed transition gates")
        end_gate = verify_input_snapshot(prepared["input_snapshot"])
        return {
            "mode": "evaluate-dev",
            "algorithm": metadata["algorithm"],
            "seed": metadata["seed"],
            "raw_transitions": raw_transitions,
            "valid_policy_transitions": valid_transitions,
            "algorithm_updates": 0,
            "optimizer_steps": 0,
            "pilot_environment_progress": pilot_progress.__dict__,
            "initial_environment_rng_evidence": initial_environment_evidence,
            "wave_accounting": wave_accounting,
            "development_condition_schedule_sha256": run[
                "development_condition_schedule_sha256"
            ],
            "source_input_end_gate": end_gate,
        }
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--run-manifest-sha256")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.resume:
            _fail("training resume is unsupported")
        prepared = _prepare(args)
        if args.output_dir.exists():
            _fail("output directory must not already exist")
        if args.execution_mode == "train":
            if args.run_manifest is not None or args.run_manifest_sha256 is not None:
                _fail("train mode does not accept a run manifest")
            report = _train(prepared, args.output_dir.resolve())
        else:
            if args.run_manifest is None or args.run_manifest_sha256 is None:
                _fail("evaluate-dev requires a frozen run manifest and its SHA")
            report = _evaluate_dev(
                prepared,
                args.run_manifest.resolve(),
                args.run_manifest_sha256,
            )
        report.update(
            {
                "schema_name": "rj45_comparison_long_report_v1",
                "contract_sha256": prepared["contract_sha256"],
                "paired_schedule_sha256": prepared["validated"]["schedule"]["sha256"],
                "final_split_used": False,
                "full_native_success_claimed": False,
            }
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.output_dir / f"{args.execution_mode}-report.json", report)
        return 2 if report.get("status") == "invalid" else 0
    except LongDriverBlocked as error:
        print(f"BLOCKED: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
