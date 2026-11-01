#!/usr/bin/env python3
"""Run a bounded fail-closed PPO/SAC integration smoke in the frozen Isaac runtime.

CPU-only validation happens before AppLauncher is imported.  The runtime root
contains the archived 104721a physics code and assets; the current root supplies
only the comparison environment adapter and common PPO/SAC implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import faulthandler
import hashlib
import importlib.util
from importlib import metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any, Mapping


SCHEMA_NAME = "rj45_comparison_isaac_smoke_v1"
EXPECTED_RUNTIME_COMMIT = "1517fbf4af11a296e3e89fcbf2aa25c49a8d017d"
EXPECTED_EVIDENCE_COMMIT = "104721acf2222c1cfe8ec86cd030f5116a4fef93"
EXPECTED_EVIDENCE_MANIFEST_SHA256 = (
    "d672417f55a61a1345dd442d067813fffdfd30274b547c8c103430866dd7b0f2"
)
EXPECTED_ACTIVE_MANIFEST_SHA256 = (
    "79a8cc65b6a72ab1d9fa2bb7824c9e6c0cf37d096585a8f4cce7f2464d134184"
)
EXPECTED_RUNTIME_MANIFEST_SHA256 = EXPECTED_EVIDENCE_MANIFEST_SHA256
EXPECTED_TASK_SHA256 = "7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b"
EXPECTED_TRAINING_SHA256 = (
    "0d5f5a07dcdfe74dfbee58852af0ad1076480cc4f91eaeb990f5e535fc61a179"
)
EXPECTED_RUNTIME_CONFIG_SHA256 = (
    "ebb7fa07ab966c1d3e02f24f60c4ef0ad49d01662f194d463d6414e2e003b67a"
)
EXPECTED_BANK_SHA256 = "c2fbcfea9528f4a9cefb8c4c63aba6cc561fe2f3095a0fa16b7456dda8da54d5"
EXPECTED_BANK_SOURCE_CONTRACT_SHA256 = (
    "38cf66d4dea11fe4253021fa57e3c0b5347974dda7490ddb9b7890e6819098d0"
)
EXPECTED_PARALLEL_CONTACT_EVIDENCE_SHA256 = (
    "db1b1e6291490db85650764374dbbccf32d0f80273e3489d3b5966416d83a383"
)
OBSERVATION_DIM = 79
ACTION_DIM = 6
ALGORITHMS = ("ppo", "sac")
PHYSICS_INPUT_SCHEMA = "rj45_task_admission_physics_inputs_v2"
PHYSICS_INPUT_COUNT = 13
EXPECTED_CACHE_ASSET_PATHS = {
    "rby1_model_m_v1_2": "assets/cache/rby1/model_v_1_2_m.usd",
    "rb_gripper_left": "assets/cache/rby1/gripper/rb_gripper_left.usd",
    "rb_gripper_right": "assets/cache/rby1/gripper/rb_gripper_right.usd",
}
EXPECTED_CURRENT_SOURCE_PATHS = {
    "scripts/smoke_rj45_comparison_isaac.py",
    "scripts/run_rj45_task_admission.py",
    "src/rby1_assembly/rj45_comparison_env.py",
    "src/rby1_assembly/rj45_task_admission_env.py",
    "src/rby1_assembly/rj45_comparison_providers.py",
    "src/rby1_assembly/rj45_task_admission.py",
}
EXPECTED_EVIDENCE_SOURCE_CONFIG_INPUTS = {
    "assets/generated/tasks/rj45-v1/build-report.json": "25b3dec084e2f1becb47268a65ec76fba4880241cfacaf43e34eae56d70cd383",
    "configs/assets.json": "92b41a462db0562d9599dff5bfdce3cbb0d38af774fd9fc77052bf9071c0bd08",
    "configs/rj45-assets.json": "30ebef7d228a2d02642bf468ae7c60e92d2e740c623e65536342a487c7bb1313",
    "configs/rj45-grasp-generation-v2.json": "7b1150212f26bec2129444123dfb8d7fa2f145ecdbbec85d05b931f5458a3d86",
    "configs/runtime.json": "ebb7fa07ab966c1d3e02f24f60c4ef0ad49d01662f194d463d6414e2e003b67a",
    "configs/tasks/rj45-v1.json": "7ac32092e6e723be066c80c53aecbbd873792e849ce42b742dbafd92cb3af88b",
    "scripts/run_sim.py": "d20444cac6af9b8ef11903249ea4b0f43ffa50f5bf0808352059b058638694ed",
}


class ComparisonSmokeBlocked(RuntimeError):
    """The G0 integration contract failed; no smoke success may be claimed."""


class ProgressRecorder:
    """Emit flushed JSON checkpoints while retaining them for the final report."""

    def __init__(self) -> None:
        self._started = time.monotonic()
        self.events: list[dict[str, Any]] = []

    def record(self, stage: str, **details: Any) -> None:
        event = {
            "event": "g0_progress",
            "stage": stage,
            "pid": os.getpid(),
            "elapsed_seconds": round(time.monotonic() - self._started, 3),
            "utc": datetime.now(timezone.utc).isoformat(),
            **details,
        }
        self.events.append(event)
        print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)


def _arm_traceback_heartbeat(progress: ProgressRecorder) -> bool:
    """Schedule low-frequency all-thread dumps without changing process exit state."""

    try:
        faulthandler.enable()
        faulthandler.dump_traceback_later(60.0, repeat=True, exit=False)
    except (OSError, RuntimeError, ValueError) as error:
        progress.record(
            "traceback_heartbeat_unavailable",
            error_type=type(error).__name__,
            error_message=str(error),
        )
        return False
    progress.record("traceback_heartbeat_armed", interval_seconds=60.0)
    return True


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComparisonSmokeBlocked(f"cannot load JSON object: {path}") from error
    if not isinstance(payload, dict):
        raise ComparisonSmokeBlocked(f"JSON root must be an object: {path}")
    return payload


def _load_current_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ComparisonSmokeBlocked(f"cannot import current helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _atomic_write_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resolve(root: Path, relative: Path | str) -> Path:
    path = Path(relative)
    return path if path.is_absolute() else root / path


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _safe_relative(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ComparisonSmokeBlocked(f"{label} must be a non-empty relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not path.parts
        or value != path.as_posix()
        or "\\" in value
    ):
        raise ComparisonSmokeBlocked(f"{label} must stay project-relative: {value}")
    return value


def _exact_hash_map(
    root: Path, expected: Mapping[str, Any], *, label: str
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for raw_relative, raw_digest in sorted(expected.items()):
        relative = _safe_relative(raw_relative, label=label)
        if (
            not isinstance(raw_digest, str)
            or len(raw_digest) != 64
            or any(character not in "0123456789abcdef" for character in raw_digest)
        ):
            raise ComparisonSmokeBlocked(f"{label} has an invalid SHA-256: {relative}")
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ComparisonSmokeBlocked(f"{label} file is missing: {relative}")
        digest = file_sha256(path)
        if digest != raw_digest:
            raise ComparisonSmokeBlocked(f"{label} SHA-256 mismatch: {relative}")
        observed[relative] = digest
    return observed


def _runtime_python_map(runtime_root: Path, runtime_manifest: Mapping[str, Any]) -> dict[str, str]:
    files = runtime_manifest.get("files")
    if not isinstance(files, dict):
        raise ComparisonSmokeBlocked("runtime manifest files are missing")
    expected = {
        path: digest
        for path, digest in files.items()
        if isinstance(path, str) and path.endswith(".py")
    }
    actual = {
        str(path.relative_to(runtime_root))
        for path in (runtime_root / "src/rby1_assembly").rglob("*.py")
        if path.is_file()
    }
    if len(expected) != 35 or actual != set(expected):
        raise ComparisonSmokeBlocked(
            "frozen runtime Python set differs from the exact 35-file manifest"
        )
    return _exact_hash_map(runtime_root, expected, label="runtime Python input")


def _selected_cache_records(
    runtime_root: Path, manifest: Mapping[str, Any], physics_map: Mapping[str, str]
) -> tuple[list[dict[str, Any]], str]:
    assets_path = runtime_root / "configs/assets.json"
    assets = _load_json(assets_path)
    rows = assets.get("assets")
    if not isinstance(rows, list):
        raise ComparisonSmokeBlocked("assets catalog has no asset rows")
    indexed = {row.get("id"): row for row in rows if isinstance(row, dict)}
    expected_rows = manifest.get("cache_assets")
    if not isinstance(expected_rows, list):
        raise ComparisonSmokeBlocked("cache asset records must be a list")
    declared = [
        (row.get("asset_id"), row.get("relative_path"))
        for row in expected_rows
        if isinstance(row, dict)
    ]
    if (
        len(declared) != len(expected_rows)
        or len(declared) != len(set(declared))
        or dict(declared) != EXPECTED_CACHE_ASSET_PATHS
    ):
        raise ComparisonSmokeBlocked("selected cache asset ID/path set differs")
    records = []
    for expected in expected_rows:
        if not isinstance(expected, dict):
            raise ComparisonSmokeBlocked("cache asset record must be an object")
        asset_id = expected.get("asset_id")
        catalog = indexed.get(asset_id)
        if not isinstance(catalog, dict):
            raise ComparisonSmokeBlocked(f"selected cache asset is absent: {asset_id}")
        relative = _safe_relative(expected.get("relative_path"), label="cache asset")
        catalog_path = catalog.get("path")
        cache_relative = f"assets/cache/{catalog_path}"
        record = {
            "asset_id": asset_id,
            "relative_path": relative,
            "size_bytes": expected.get("size_bytes"),
            "sha256": expected.get("sha256"),
        }
        if relative != cache_relative:
            raise ComparisonSmokeBlocked(f"cache asset path mismatch: {asset_id}")
        if catalog.get("size_bytes") != record["size_bytes"]:
            raise ComparisonSmokeBlocked(f"cache asset size declaration mismatch: {asset_id}")
        if catalog.get("sha256") != record["sha256"]:
            raise ComparisonSmokeBlocked(f"cache asset catalog SHA mismatch: {asset_id}")
        path = runtime_root / relative
        if path.stat().st_size != record["size_bytes"]:
            raise ComparisonSmokeBlocked(f"cache asset actual size mismatch: {asset_id}")
        if physics_map.get(relative) != record["sha256"]:
            raise ComparisonSmokeBlocked(f"cache asset actual SHA mismatch: {asset_id}")
        records.append(record)
    if len(records) != 3:
        raise ComparisonSmokeBlocked("selected cache graph must contain exactly three assets")
    graph = {
        "robot_usda_sha256": physics_map["assets/generated/tasks/rj45-v1/robot.usda"],
        "selected_cache_assets": records,
    }
    graph_sha = _canonical_json_sha256(graph)
    if graph_sha != manifest.get("selected_cache_graph_sha256"):
        raise ComparisonSmokeBlocked("selected cache graph SHA mismatch")
    return records, graph_sha


def validate_physics_input_snapshot(
    runtime_root: Path,
    evidence_root: Path,
    current_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Validate and snapshot every physics input before AppLauncher exists."""

    runtime_root = runtime_root.resolve()
    evidence_root = evidence_root.resolve()
    current_root = current_root.resolve()
    manifest_path = manifest_path.resolve()
    if len({runtime_root, evidence_root, current_root}) != 3:
        raise ComparisonSmokeBlocked("runtime, evidence, and current roots must be distinct")
    manifest = _load_json(manifest_path)
    if manifest.get("schema_name") != PHYSICS_INPUT_SCHEMA:
        raise ComparisonSmokeBlocked("unexpected physics input manifest schema")
    if manifest.get("runtime_commit") != EXPECTED_RUNTIME_COMMIT:
        raise ComparisonSmokeBlocked("physics manifest runtime commit mismatch")
    base_spec = manifest.get("base_physics_manifest")
    if not isinstance(base_spec, dict):
        raise ComparisonSmokeBlocked("base physics manifest binding is missing")
    base_relative = _safe_relative(base_spec.get("relative_path"), label="base physics manifest")
    base_path = (current_root / base_relative).resolve()
    if not base_path.is_relative_to(current_root) or file_sha256(base_path) != base_spec.get("sha256"):
        raise ComparisonSmokeBlocked("base physics manifest binding mismatch")
    manifest = {**_load_json(base_path), **manifest}
    physics_expected = manifest.get("physics_inputs")
    if not isinstance(physics_expected, dict) or len(physics_expected) != PHYSICS_INPUT_COUNT:
        raise ComparisonSmokeBlocked("physics input manifest must contain exactly 13 files")
    required = set(manifest.get("required_physics_relative_paths", ()))
    if required != set(physics_expected):
        raise ComparisonSmokeBlocked("physics input required-path set mismatch")
    physics_map = _exact_hash_map(runtime_root, physics_expected, label="active physics input")
    evidence_physics_map = _exact_hash_map(evidence_root, physics_expected, label="evidence physics input")
    if physics_map != evidence_physics_map:
        raise ComparisonSmokeBlocked("active and evidence physics inputs differ")
    active_import_expected = manifest.get("active_import_inputs", {})
    if set(active_import_expected) != {
            "configs/task.json",
            "assets/generated/build-report.json",
    }:
        raise ComparisonSmokeBlocked("active import input set must contain exactly two files")
    active_import_map = _exact_hash_map(runtime_root, active_import_expected, label="active import input")
    generic_task_sha = active_import_map["configs/task.json"]
    generic_report = _load_json(runtime_root / "assets/generated/build-report.json")
    if generic_report.get("config_sha256") != generic_task_sha:
        raise ComparisonSmokeBlocked("generic build report task binding mismatch")

    core_spec = manifest.get("runtime_core_manifest")
    if not isinstance(core_spec, dict):
        raise ComparisonSmokeBlocked("runtime core manifest binding is missing")
    core_relative = _safe_relative(core_spec.get("relative_path"), label="core manifest")
    core_path = (current_root / core_relative).resolve()
    if not core_path.is_relative_to(current_root) or file_sha256(core_path) != core_spec.get("sha256"):
        raise ComparisonSmokeBlocked("runtime core manifest binding mismatch")
    runtime_manifest = _load_json(core_path)
    runtime_map = _runtime_python_map(runtime_root, runtime_manifest)

    evidence_spec = manifest.get("evidence_core_manifest")
    if not isinstance(evidence_spec, dict):
        raise ComparisonSmokeBlocked("evidence core manifest binding is missing")
    evidence_relative = _safe_relative(
        evidence_spec.get("relative_path"), label="evidence core manifest"
    )
    evidence_core_path = (evidence_root / evidence_relative).resolve()
    if (
        not evidence_core_path.is_relative_to(evidence_root)
        or file_sha256(evidence_core_path) != evidence_spec.get("sha256")
        or evidence_spec.get("sha256") != EXPECTED_EVIDENCE_MANIFEST_SHA256
    ):
        raise ComparisonSmokeBlocked("evidence core manifest binding mismatch")
    evidence_manifest = _load_json(evidence_core_path)
    if evidence_manifest.get("git_commit") != EXPECTED_EVIDENCE_COMMIT:
        raise ComparisonSmokeBlocked("evidence core commit mismatch")
    evidence_configs = manifest.get("evidence_source_config_inputs")
    if evidence_configs != EXPECTED_EVIDENCE_SOURCE_CONFIG_INPUTS:
        raise ComparisonSmokeBlocked("evidence source config map must contain exactly 7 files")
    evidence_sources = {**evidence_manifest.get("files", {}), **evidence_configs}
    if len(evidence_sources) != 42:
        raise ComparisonSmokeBlocked("evidence source input union must contain exactly 42 files")
    evidence_source_map = _exact_hash_map(
        evidence_root, evidence_sources, label="evidence source input"
    )
    if set(evidence_manifest.get("files", {})) - set(evidence_source_map):
        raise ComparisonSmokeBlocked("evidence source map omits core Python files")

    if physics_map.get("configs/rj45-training-target-offset-v1.json") != EXPECTED_TRAINING_SHA256:
        raise ComparisonSmokeBlocked("trusted training config SHA mismatch")
    if physics_map.get("configs/rj45-training-target-offset-v1.json") != manifest.get(
        "trusted_training_sha256"
    ):
        raise ComparisonSmokeBlocked("physics manifest training binding mismatch")

    active_task_path = runtime_root / "configs/tasks/rj45-v1.json"
    evidence_task_path = evidence_root / "configs/tasks/rj45-v1.json"
    if active_task_path.read_bytes() != evidence_task_path.read_bytes():
        raise ComparisonSmokeBlocked("active and evidence task bytes differ")
    task = _load_json(active_task_path)
    if task != _load_json(evidence_task_path):
        raise ComparisonSmokeBlocked("active and evidence parsed tasks differ")
    expected_generated = manifest.get("generated_asset_sha256")
    if not isinstance(expected_generated, dict) or len(expected_generated) != 5:
        raise ComparisonSmokeBlocked("generated asset map must contain five files")
    task_assets = task.get("assets")
    if not isinstance(task_assets, dict):
        raise ComparisonSmokeBlocked("task assets mapping is missing")
    expected_task_assets = {
        key: f"assets/generated/tasks/rj45-v1/{filename}"
        for key, filename in {
            "robot": "robot.usda",
            "socket": "socket.usda",
            "plug": "plug.usda",
            "table": "table.usda",
            "build_report": "build-report.json",
        }.items()
    }
    if task_assets != expected_task_assets:
        raise ComparisonSmokeBlocked("task generated-asset paths differ")

    build = _load_json(runtime_root / task_assets["build_report"])
    if build.get("config_sha256") != physics_map["configs/tasks/rj45-v1.json"]:
        raise ComparisonSmokeBlocked("build report task SHA mismatch")
    build_files = build.get("files")
    if build_files != expected_generated:
        raise ComparisonSmokeBlocked("build report generated-file map mismatch")
    for filename, digest in expected_generated.items():
        relative = f"assets/generated/tasks/rj45-v1/{filename}"
        if physics_map.get(relative) != digest:
            raise ComparisonSmokeBlocked(f"generated asset SHA mismatch: {filename}")
    generic_robot = generic_report.get("robot")
    rj45_robot = build.get("robot")
    if not isinstance(generic_robot, dict) or not isinstance(rj45_robot, dict):
        raise ComparisonSmokeBlocked("build report robot projection is missing")
    projection = {
        "grasp_center_in_wrist_m": rj45_robot.get("grasp_center_in_wrist_m"),
        "gripper_finger_span_m": rj45_robot.get("gripper_finger_span_m"),
        "gripper_geometry": {
            side: {"bounds_in_wrist_m": rj45_robot.get("gripper_geometry", {}).get(side, {}).get("bounds_in_wrist_m")}
            for side in ("left", "right")
        },
    }
    generic_projection = {
        "grasp_center_in_wrist_m": generic_robot.get("grasp_center_in_wrist_m"),
        "gripper_finger_span_m": generic_robot.get("gripper_finger_span_m"),
        "gripper_geometry": {
            side: {"bounds_in_wrist_m": generic_robot.get("gripper_geometry", {}).get(side, {}).get("bounds_in_wrist_m")}
            for side in ("left", "right")
        },
    }
    if generic_projection != projection:
        raise ComparisonSmokeBlocked("generic and RJ45 robot geometry projections differ")

    cache_records, cache_graph_sha = _selected_cache_records(
        runtime_root, manifest, physics_map
    )
    robot_text = (runtime_root / task_assets["robot"]).read_text(encoding="utf-8")
    references = manifest.get("generated_robot_references")
    expected_references = set(EXPECTED_CACHE_ASSET_PATHS.values())
    if (
        not isinstance(references, list)
        or len(references) != len(expected_references)
        or len(references) != len(set(references))
        or set(references) != expected_references
    ):
        raise ComparisonSmokeBlocked("robot reference list must be the exact three assets")
    if {row["relative_path"] for row in cache_records} != expected_references:
        raise ComparisonSmokeBlocked("robot reference and selected cache sets differ")
    expected_authored = {
        "../../../" + relative.removeprefix("assets/")
        for relative in expected_references
    }
    authored_references = re.findall(r"@([^@\r\n]+)@", robot_text)
    if (
        len(authored_references) != len(expected_authored)
        or len(authored_references) != len(set(authored_references))
        or set(authored_references) != expected_authored
    ):
        raise ComparisonSmokeBlocked("robot USD authored reference set differs")

    closure = manifest.get("headless_physics_geometry_closure")
    if closure != {
        "cache_roots": 3,
        "self_layers_per_root": 1,
        "extra_resolved_assets": 0,
        "unresolved_authored_asset": "OmniPBR.mdl",
        "authored_property": "Shader.info:mdl:sourceAsset",
        "scope": "headless_physics_geometry_only",
        "full_render_shader_closure_proven": False,
    }:
        raise ComparisonSmokeBlocked("headless physics geometry closure contract differs")

    current_paths = manifest.get("current_source_paths")
    if (
        not isinstance(current_paths, list)
        or len(current_paths) != len(set(current_paths))
        or set(current_paths) != EXPECTED_CURRENT_SOURCE_PATHS
    ):
        raise ComparisonSmokeBlocked("current source path list must be the exact six files")
    current_map = {}
    for value in current_paths:
        relative = _safe_relative(value, label="current source")
        path = (current_root / relative).resolve()
        if not path.is_relative_to(current_root) or not path.is_file():
            raise ComparisonSmokeBlocked(f"current source is missing: {relative}")
        current_map[relative] = file_sha256(path)
    validator_relative = "scripts/smoke_rj45_comparison_isaac.py"
    if Path(__file__).resolve() != (current_root / validator_relative).resolve():
        raise ComparisonSmokeBlocked("validator source path differs from current root")

    shared_catalog = "configs/rj45-grasp-generation-v2.json"
    if shared_catalog not in evidence_source_map:
        raise ComparisonSmokeBlocked("evidence source map omits the shared bank catalog")
    normalized = {
        "runtime": runtime_map | physics_map | active_import_map,
        "evidence": {
            **{k: v for k, v in evidence_source_map.items() if k != shared_catalog},
            **evidence_physics_map,
        },
        "current": current_map,
    }
    if (
        len(normalized["runtime"]) != 50 or len(normalized["evidence"]) != 50
    ):
        raise ComparisonSmokeBlocked("dual-root normalized namespace count mismatch")
    return {
        "schema_name": "rj45_task_admission_physics_input_snapshot_v2",
        "runtime_root": str(runtime_root),
        "evidence_root": str(evidence_root),
        "current_root": str(current_root),
        "manifest": {"path": str(manifest_path), "sha256": file_sha256(manifest_path)},
        "source_manifests": [
            {
                "role": "active_core",
                "path": str(core_path),
                "sha256": file_sha256(core_path),
            },
            {
                "role": "base_physics",
                "path": str(base_path),
                "sha256": file_sha256(base_path),
            },
            {
                "role": "evidence_core",
                "path": str(evidence_core_path),
                "sha256": file_sha256(evidence_core_path),
            },
        ],
        "normalized_sha256": normalized,
        "normalized_sha256_canonical_sha256": _canonical_json_sha256(normalized),
        "selected_cache_graph_sha256": cache_graph_sha,
        "headless_physics_geometry_closure": closure,
        "verification_contract": manifest.get("verification_contract"),
    }


def verify_physics_input_end(
    start_snapshot: Mapping[str, Any],
    runtime_root: Path,
    evidence_root: Path,
    current_root: Path,
) -> dict[str, Any]:
    """Compare current bytes with an immutable prelaunch physics snapshot."""

    expected = start_snapshot.get("normalized_sha256")
    if not isinstance(expected, dict):
        raise ComparisonSmokeBlocked("start snapshot normalized map is missing")
    if {name: len(values) if isinstance(values, dict) else -1 for name, values in expected.items()} != {
        "runtime": 50,
        "evidence": 50,
        "current": 6,
    }:
        raise ComparisonSmokeBlocked("start snapshot namespace counts differ")
    changed: list[str] = []
    missing: list[str] = []
    added: list[str] = []
    expected_canonical = start_snapshot.get("normalized_sha256_canonical_sha256")
    if _canonical_json_sha256(expected) != expected_canonical:
        raise ComparisonSmokeBlocked("start snapshot canonical SHA mismatch")
    roots = {
        "runtime": runtime_root.resolve(),
        "current": current_root.resolve(),
    }
    roots["evidence"] = evidence_root.resolve()
    if set(expected) != {"runtime", "evidence", "current"} or len(set(roots.values())) != 3:
        raise ComparisonSmokeBlocked("runtime, evidence, and current roots must be distinct")
    sealed_roots = {
        "runtime": start_snapshot.get("runtime_root"),
        "evidence": start_snapshot.get("evidence_root"),
        "current": start_snapshot.get("current_root"),
    }
    if sealed_roots != {name: str(root) for name, root in roots.items()}:
        raise ComparisonSmokeBlocked("end-check roots differ from the sealed snapshot roots")
    for namespace, root in roots.items():
        mapping = expected.get(namespace)
        if not isinstance(mapping, dict):
            raise ComparisonSmokeBlocked(f"start snapshot {namespace} map is missing")
        for relative, digest in sorted(mapping.items()):
            path = root / _safe_relative(relative, label=f"{namespace} end input")
            if not path.is_file():
                missing.append(f"{namespace}/{relative}")
            elif file_sha256(path) != digest:
                changed.append(f"{namespace}/{relative}")
    expected_runtime_python = {
        relative
        for relative in expected["runtime"]
        if relative.startswith("src/rby1_assembly/") and relative.endswith(".py")
    }
    actual_runtime_python = {
        str(path.relative_to(runtime_root.resolve()))
        for path in (runtime_root.resolve() / "src/rby1_assembly").rglob("*.py")
        if path.is_file()
    }
    added.extend(
        f"runtime/{relative}"
        for relative in sorted(actual_runtime_python - expected_runtime_python)
    )
    manifest = start_snapshot.get("manifest")
    if not isinstance(manifest, dict):
        raise ComparisonSmokeBlocked("start snapshot manifest binding is missing")
    manifest_path = Path(str(manifest.get("path", "")))
    if not manifest_path.is_file():
        missing.append("current/physics-input-manifest")
    elif file_sha256(manifest_path) != manifest.get("sha256"):
        changed.append("current/physics-input-manifest")
    source_manifests = start_snapshot.get("source_manifests", [])
    if not isinstance(source_manifests, list):
        raise ComparisonSmokeBlocked("start snapshot source manifests are invalid")
    if len(source_manifests) != 3:
        raise ComparisonSmokeBlocked("start snapshot must bind exactly three source manifests")
    expected_source_paths = {
        "active_core": roots["current"] / "configs/rj45-oracle-runtime-validation-ulp2-v1.json",
        "base_physics": roots["current"] / "configs/rj45-task-admission-physics-inputs-104721a.json",
        "evidence_core": roots["evidence"] / "configs/rj45-oracle-runtime-104721a.json",
    }
    seen_roles: set[str] = set()
    for index, binding in enumerate(source_manifests):
        if not isinstance(binding, dict):
            raise ComparisonSmokeBlocked("start snapshot source manifest binding is invalid")
        role = binding.get("role")
        if role not in expected_source_paths or role in seen_roles:
            raise ComparisonSmokeBlocked("start snapshot source manifest role is invalid")
        seen_roles.add(role)
        raw_path = binding.get("path")
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_absolute()
            or str(Path(raw_path).resolve()) != raw_path
        ):
            raise ComparisonSmokeBlocked("start snapshot source manifest path is invalid")
        path = Path(raw_path).resolve()
        if path != expected_source_paths[role].resolve():
            raise ComparisonSmokeBlocked("start snapshot source manifest path differs")
        label = f"source-manifest/{index}"
        if not path.is_file():
            missing.append(label)
        elif file_sha256(path) != binding.get("sha256"):
            changed.append(label)
    if seen_roles != set(expected_source_paths):
        raise ComparisonSmokeBlocked("start snapshot source manifest roles differ")
    matched = not added and not changed and not missing
    return {
        "schema_name": "rj45_task_admission_physics_input_end_verification_v1",
        "status": "matched" if matched else "input_mutation_blocked",
        "matched": matched,
        "changed": changed,
        "missing": missing,
        "added": added,
        "start_normalized_sha256_canonical_sha256": start_snapshot.get(
            "normalized_sha256_canonical_sha256"
        ),
        "selected_cache_graph_sha256": start_snapshot.get(
            "selected_cache_graph_sha256"
        ),
    }


def validate_held_proof_physics_binding(
    proof: Mapping[str, Any], snapshot: Mapping[str, Any], manifest_path: Path
) -> None:
    manifest = _load_json(manifest_path)
    base_spec = manifest.get("base_physics_manifest")
    if isinstance(base_spec, dict):
        relative = _safe_relative(
            base_spec.get("relative_path"), label="base physics manifest"
        )
        base_path = manifest_path.resolve().parents[1] / relative
        if file_sha256(base_path) != base_spec.get("sha256"):
            raise ComparisonSmokeBlocked("base physics manifest binding mismatch")
        manifest = {**_load_json(base_path), **manifest}
    expected = manifest.get("held_proof_required_source_and_config_sha256")
    actual = proof.get("provenance_contract", {}).get("source_and_config_sha256")
    if not isinstance(actual, dict) or not isinstance(expected, dict):
        raise ComparisonSmokeBlocked("held proof source/config binding is missing")
    if {path: actual.get(path) for path in expected} != expected:
        raise ComparisonSmokeBlocked("held proof source/config binding mismatch")
    runtime_map = snapshot["normalized_sha256"]["runtime"]
    if any(runtime_map.get(path) != digest for path, digest in expected.items()):
        raise ComparisonSmokeBlocked("held proof binding differs from start snapshot")


def _parallel_evidence_dependency_paths(
    root: Path, specification: Mapping[str, Any]
) -> tuple[Path, ...]:
    dependencies = []
    for name in ("report_path", "trace_path", "sat_report_path"):
        value = specification.get(name)
        if not isinstance(value, str) or not value:
            raise ComparisonSmokeBlocked(
                f"parallel contact evidence {name} must be a non-empty string"
            )
        dependencies.append(_resolve(root, value).resolve())
    return tuple(dependencies)


def validate_runtime_snapshot(runtime_root: Path, manifest_path: Path) -> dict[str, Any]:
    runtime_root = runtime_root.resolve()
    manifest_path = manifest_path.resolve()
    if (
        manifest_path.name != "rj45-oracle-runtime-validation-ulp2-v1.json"
        or file_sha256(manifest_path) != EXPECTED_ACTIVE_MANIFEST_SHA256
    ):
        raise ComparisonSmokeBlocked("active runtime manifest path or SHA-256 mismatch")
    manifest = _load_json(manifest_path)
    if manifest.get("git_commit") != EXPECTED_RUNTIME_COMMIT:
        raise ComparisonSmokeBlocked("archived runtime commit mismatch")
    if (
        manifest.get("schema_version") != 2
        or manifest.get("base_commit") != EXPECTED_EVIDENCE_COMMIT
        or manifest.get("allowed_delta_path")
        != "src/rby1_assembly/experiment_contracts.py"
    ):
        raise ComparisonSmokeBlocked("active runtime manifest provenance mismatch")
    verified = _runtime_python_map(runtime_root, manifest)
    return {
        "path": str(manifest_path),
        "sha256": file_sha256(manifest_path),
        "git_commit": manifest["git_commit"],
        "verified_file_count": len(verified),
        "files": verified,
    }


def validate_prelaunch_inputs(args: argparse.Namespace) -> dict[str, Any]:
    runtime_root = args.runtime_root.resolve()
    evidence_root = args.evidence_root.resolve()
    current_root = args.current_root.resolve()
    if len({runtime_root, evidence_root, current_root}) != 3:
        raise ComparisonSmokeBlocked("runtime, evidence, and current roots must be distinct")
    runtime_manifest = validate_runtime_snapshot(runtime_root, args.runtime_manifest)
    fixed = {
        "task_config": (
            _resolve(evidence_root, args.task_config),
            EXPECTED_TASK_SHA256,
        ),
        "training_config": (
            _resolve(evidence_root, args.training_config),
            EXPECTED_TRAINING_SHA256,
        ),
        "runtime_config": (
            _resolve(evidence_root, args.runtime_config),
            EXPECTED_RUNTIME_CONFIG_SHA256,
        ),
        "held_bank": (_resolve(evidence_root, args.held_bank), EXPECTED_BANK_SHA256),
        "parallel_contact_evidence": (
            _resolve(runtime_root, args.parallel_contact_evidence),
            EXPECTED_PARALLEL_CONTACT_EVIDENCE_SHA256,
        ),
    }
    frozen_inputs: dict[str, dict[str, str]] = {}
    for label, (path, expected) in fixed.items():
        if not path.is_file() or file_sha256(path) != expected:
            raise ComparisonSmokeBlocked(f"frozen input binding failed: {label}")
        frozen_inputs[label] = {"path": str(path.resolve()), "sha256": expected}
    current_paths = {
        "runner": Path(__file__).resolve(),
        "comparison_env": current_root / "src/rby1_assembly/rj45_comparison_env.py",
        "comparison_rl": current_root / "src/rby1_assembly/rj45_comparison_rl.py",
        "requirements": current_root / "configs/rj45-comparison-requirements.txt",
    }
    current_sources: dict[str, dict[str, str]] = {}
    for label, path in current_paths.items():
        if not path.is_file():
            raise ComparisonSmokeBlocked(f"current comparison source is missing: {label}")
        current_sources[label] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    return {
        "runtime_root": str(runtime_root),
        "evidence_root": str(evidence_root),
        "current_root": str(current_root),
        "runtime_manifest": runtime_manifest,
        "frozen_inputs": frozen_inputs,
        "current_sources": current_sources,
        "snapshot_labels": {
            "physics_assets_and_legacy_environment": "archived_runtime_104721a",
            "comparison_adapter_and_algorithms": "current_root_uncommitted_bytes",
        },
    }


def validate_execution_args(args: argparse.Namespace) -> None:
    if not 4 <= args.num_envs <= 16:
        raise ComparisonSmokeBlocked("num-envs must be between 4 and 16")
    if args.device != "cuda:0":
        raise ComparisonSmokeBlocked("G0 Isaac smoke requires cuda:0")
    if args.num_envs > 1 and args.parallel_contact_evidence is None:
        raise ComparisonSmokeBlocked(
            "parallel G0 Isaac smoke requires --parallel-contact-evidence"
        )
    if args.ppo_rollout_steps < 2:
        raise ComparisonSmokeBlocked("ppo-rollout-steps must be at least 2")
    if args.sac_vector_steps < 1:
        raise ComparisonSmokeBlocked("sac-vector-steps must be positive")
    if args.sac_batch_size < 2:
        raise ComparisonSmokeBlocked("sac-batch-size must be at least 2")
    if not math.isfinite(args.sac_utd_ratio) or args.sac_utd_ratio != 1.0:
        raise ComparisonSmokeBlocked("G0 requires an explicit SAC UTD ratio of 1.0")
    if args.min_sac_updates < 2:
        raise ComparisonSmokeBlocked("min-sac-updates must be at least 2")
    if args.readiness_steps < 1:
        raise ComparisonSmokeBlocked("readiness-steps must be positive")


def _policy_observation(result: Any) -> tuple[Any, dict[str, Any]]:
    if not isinstance(result, tuple) or len(result) != 2:
        raise ComparisonSmokeBlocked("reset/readiness result must be (observations, infos)")
    observations, infos = result
    if isinstance(observations, dict):
        observations = observations.get("policy")
    if not isinstance(infos, dict):
        raise ComparisonSmokeBlocked("environment infos must be a dictionary")
    return observations, infos


def wait_for_policy_readiness(
    env: Any,
    *,
    max_steps: int,
    progress: ProgressRecorder | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Advance zero-action warmup without exposing it as learning transitions."""

    import torch

    if progress is not None:
        progress.record("initial_reset_begin")
    observations, infos = _policy_observation(env.reset())
    if progress is not None:
        progress.record("initial_reset_complete")
    history: list[int] = []
    for readiness_step in range(max_steps + 1):
        validity = infos.get("policy_observation_valid")
        if not isinstance(validity, torch.Tensor) or validity.shape != (env.num_envs,):
            raise ComparisonSmokeBlocked("readiness requires per-row policy_observation_valid")
        validity = validity.to(dtype=torch.bool, device=observations.device)
        activated = getattr(env, "_training_policy_activated", None)
        estimator_valid = getattr(env, "_training_estimator_valid", None)
        if not isinstance(activated, torch.Tensor) or not isinstance(estimator_valid, torch.Tensor):
            raise ComparisonSmokeBlocked("comparison readiness tensors are unavailable")
        ready = validity & activated & estimator_valid
        ready_rows = int(ready.sum().item())
        history.append(ready_rows)
        if progress is not None and (
            readiness_step == 0
            or readiness_step % 30 == 0
            or readiness_step == max_steps
            or ready_rows == env.num_envs
        ):
            progress.record(
                "readiness_checkpoint",
                readiness_step=readiness_step,
                max_steps=max_steps,
                ready_rows=ready_rows,
                num_envs=env.num_envs,
            )
        if bool(ready.all()):
            if tuple(observations.shape) != (env.num_envs, OBSERVATION_DIM):
                raise ComparisonSmokeBlocked("ready observations must have shape [num_envs, 79]")
            if not bool(torch.isfinite(observations).all()):
                raise ComparisonSmokeBlocked("ready observations contain non-finite values")
            return observations, infos, {
                "warmup_vector_steps": readiness_step,
                "warmup_transitions_excluded_from_learning": readiness_step * env.num_envs,
                "ready_rows_history": history,
                "all_rows_policy_active": True,
                "all_rows_estimator_valid": True,
            }
        if readiness_step == max_steps:
            break
        step_result = env.step(
            torch.zeros((env.num_envs, ACTION_DIM), device=observations.device)
        )
        if not isinstance(step_result, tuple) or len(step_result) != 5:
            raise ComparisonSmokeBlocked("Isaac step contract must have five values")
        next_observations, _rewards, terminated, truncated, infos = step_result
        if bool(torch.as_tensor(terminated).any()) or bool(torch.as_tensor(truncated).any()):
            raise ComparisonSmokeBlocked("environment ended before all rows became policy-ready")
        observations = (
            next_observations["policy"]
            if isinstance(next_observations, dict)
            else next_observations
        )
    raise ComparisonSmokeBlocked("not all environments became policy-ready")


class ReadyVectorEnvironment:
    """Expose one captured ready state as collect_bounded's initial reset."""

    def __init__(self, env: Any, observations: Any, infos: dict[str, Any]):
        self._env = env
        self.num_envs = env.num_envs
        self._observations = observations
        self._infos = infos
        self._reset_consumed = False

    def reset(self):
        if self._reset_consumed:
            raise ComparisonSmokeBlocked("ready collector reset may only be consumed once")
        self._reset_consumed = True
        return {"policy": self._observations}, self._infos

    def step(self, actions):
        return self._env.step(actions)


def validate_collector_result(result: Any, *, expected_transitions: int) -> None:
    if result.transitions != expected_transitions:
        raise ComparisonSmokeBlocked("collector transition counter mismatch")
    if result.valid_policy_transitions != expected_transitions:
        raise ComparisonSmokeBlocked("collector observed a non-policy-ready transition")
    if result.learning_transitions != expected_transitions:
        raise ComparisonSmokeBlocked("collector omitted transitions from learning")
    if result.warmup_transitions != 0 or result.skipped_valid_transitions != 0:
        raise ComparisonSmokeBlocked("collector attempted warmup/sample deletion")
    if result.mixed_validity_vector_steps != 0:
        raise ComparisonSmokeBlocked("collector observed mixed-validity vector steps")


def algorithm_vector_steps(
    algorithm: str, *, ppo_rollout_steps: int, sac_vector_steps: int
) -> int:
    if algorithm == "ppo":
        return ppo_rollout_steps * 2
    if algorithm == "sac":
        return sac_vector_steps
    raise ValueError("algorithm must be ppo or sac")


def _runtime_versions() -> dict[str, str]:
    return {
        name: metadata.version(name)
        for name in (
            "isaacsim",
            "isaaclab",
            "isaaclab-rl",
            "rsl-rl-lib",
            "torch",
            "numpy",
            "h5py",
        )
    }


def _comparison_dependency_versions() -> dict[str, str]:
    return {name: metadata.version(name) for name in ("skrl", "gymnasium")}


def _build_env(
    args: argparse.Namespace,
    comparison_env_module: Any,
    progress: ProgressRecorder,
) -> tuple[Any, dict[str, Any], dict[Path, str]]:
    from rby1_assembly.experiment_contracts import apply_runtime_timing
    from rby1_assembly.held_grasp import (
        bank_source_contract_sha256,
        held_grasp_bank_dependency_paths,
        load_verified_held_grasp_bank,
    )
    from rby1_assembly.isaac_env import Rby1AssemblyEnvCfg
    from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence
    from rby1_assembly.rj45_env import configure_rj45_task
    from rby1_assembly.rj45_task import (
        load_rj45_task,
        verify_generated_asset_hashes,
        verify_runtime_versions,
    )
    from rby1_assembly.rj45_training import load_training_config
    from rby1_assembly.rj45_training_env import make_rj45_training_env_type

    root = args.runtime_root.resolve()
    evidence_root = args.evidence_root.resolve()
    task_path = _resolve(root, args.task_config)
    evidence_task_path = _resolve(evidence_root, args.task_config)
    training_path = _resolve(evidence_root, args.training_config)
    runtime_path = _resolve(evidence_root, args.runtime_config)
    bank_path = _resolve(evidence_root, args.held_bank)
    parallel_spec_path = _resolve(root, args.parallel_contact_evidence)
    task = load_rj45_task(task_path)
    evidence_task = load_rj45_task(evidence_task_path)
    if task_path.read_bytes() != evidence_task_path.read_bytes() or task != evidence_task:
        raise ComparisonSmokeBlocked("active and evidence tasks differ at runtime")
    training = load_training_config(training_path)
    runtime = _load_json(runtime_path)
    parallel_spec = _load_json(parallel_spec_path)
    if training["base_observation_dim"] != OBSERVATION_DIM:
        raise ComparisonSmokeBlocked("training config is not the fixed 79D contract")
    if training["diagnostic_observation_mode"] != "standard":
        raise ComparisonSmokeBlocked("training config exposes privileged diagnostics")
    if training["action_mode"] != "right_only_6d":
        raise ComparisonSmokeBlocked("training config is not right-arm 6D residual")
    versions = _runtime_versions()
    verify_runtime_versions(versions, runtime)
    build_report_path = _resolve(root, task["assets"]["build_report"])
    build_report = _load_json(build_report_path)
    asset_hashes = verify_generated_asset_hashes(root, task, build_report)
    bank, proof = load_verified_held_grasp_bank(
        evidence_root, evidence_task, bank_path, "train", allow_final=False
    )
    if proof.get("manifest_sha256") != EXPECTED_BANK_SHA256:
        raise ComparisonSmokeBlocked("held-bank manifest proof mismatch")
    if proof.get("selected_split") != "train":
        raise ComparisonSmokeBlocked("held-bank proof is not the train split")
    if bank_source_contract_sha256(proof) != EXPECTED_BANK_SOURCE_CONTRACT_SHA256:
        raise ComparisonSmokeBlocked("held-bank source contract mismatch")
    parallel_contact_proof = load_parallel_contact_evidence(
        parallel_spec, project_root=root
    )
    if args.num_envs != parallel_contact_proof.get("validated_probe_num_envs"):
        raise ComparisonSmokeBlocked(
            "parallel contact evidence does not cover the requested environment count"
        )

    cfg = Rby1AssemblyEnvCfg()
    apply_runtime_timing(cfg, runtime)
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.rj45_parallel_contact_evidence = parallel_spec
    env_type = configure_rj45_task(cfg, task, root)
    env_type = make_rj45_training_env_type(
        env_type,
        training,
        held_bank=bank,
        held_bank_proof=proof,
        calibration_bias_enabled=True,
        bias_level=training["calibration_bias"]["initial_level"],
    )
    env_type = comparison_env_module.make_rj45_comparison_env_type(env_type)
    dependencies = held_grasp_bank_dependency_paths(
        evidence_root, bank_path, "train", allow_final=False
    )
    frozen_paths = {
        task_path,
        training_path,
        runtime_path,
        bank_path,
        parallel_spec_path,
        build_report_path,
        *_parallel_evidence_dependency_paths(root, parallel_spec),
        *(root / path for path in parallel_contact_proof["current_source_sha256"]),
        *dependencies,
        *(_resolve(root, task["assets"][name]) for name in ("robot", "socket", "plug", "table")),
        build_report_path.parent / "scene.usda",
    }
    frozen_paths.update((root / "src/rby1_assembly").glob("*.py"))
    hashes = {path.resolve(): file_sha256(path) for path in frozen_paths}
    progress.record("environment_constructor_begin", num_envs=args.num_envs)
    env = env_type(cfg)
    progress.record("environment_constructor_complete", num_envs=args.num_envs)
    return env, {
        "runtime_versions": versions,
        "comparison_dependency_versions": _comparison_dependency_versions(),
        "task": task,
        "training": training,
        "asset_sha256": asset_hashes,
        "held_bank_proof": proof,
        "held_bank_dependency_count": len(dependencies),
        "parallel_contact_evidence": parallel_contact_proof,
        "parallel_contact_evidence_dependency_sha256": {
            str(path): file_sha256(path)
            for path in _parallel_evidence_dependency_paths(root, parallel_spec)
        },
        "comparison_contract": comparison_env_module.comparison_observation_metadata(),
    }, hashes


def _run_algorithm(
    args: argparse.Namespace,
    algorithm: str,
    comparison_env_module: Any,
    comparison_rl_module: Any,
    progress: ProgressRecorder,
) -> dict[str, Any]:
    import torch

    progress.record("algorithm_begin", algorithm=algorithm)
    progress.record("environment_build_begin", algorithm=algorithm)
    env, context, frozen_hashes = _build_env(args, comparison_env_module, progress)
    progress.record("environment_build_complete", algorithm=algorithm)
    try:
        observations, infos, readiness = wait_for_policy_readiness(
            env, max_steps=args.readiness_steps, progress=progress
        )
        rollout_steps = args.ppo_rollout_steps
        progress.record("agent_creation_begin", algorithm=algorithm)
        agent = comparison_rl_module.make_agent(
            algorithm,
            num_envs=args.num_envs,
            device=args.device,
            rollout_steps=rollout_steps,
            replay_capacity=max(256, args.num_envs * args.sac_vector_steps * 2),
            batch_size=args.sac_batch_size,
            sac_utd_ratio=args.sac_utd_ratio,
            seed=args.seed,
        )
        progress.record("agent_creation_complete", algorithm=algorithm)
        before = comparison_rl_module.parameter_vector(agent)
        update_records: list[dict[str, Any]] = []
        original_update = agent.update

        def audited_update(*update_args, **update_kwargs):
            value = original_update(*update_args, **update_kwargs)
            update_records.append(
                {
                    "index": len(update_records) + 1,
                    "return_finite": _finite_structure(value),
                }
            )
            return value

        agent.update = audited_update
        vector_steps = algorithm_vector_steps(
            algorithm,
            ppo_rollout_steps=rollout_steps,
            sac_vector_steps=args.sac_vector_steps,
        )
        ready_env = ReadyVectorEnvironment(env, observations, infos)
        progress.record(
            "collection_begin",
            algorithm=algorithm,
            vector_steps=vector_steps,
        )
        collector = comparison_rl_module.collect_bounded(
            agent,
            ready_env,
            algorithm=algorithm,
            vector_steps=vector_steps,
            require_policy_validity=True,
        )
        progress.record(
            "collection_complete",
            algorithm=algorithm,
            vector_steps=vector_steps,
            observed_transitions=collector.transitions,
        )
        validate_collector_result(
            collector, expected_transitions=args.num_envs * vector_steps
        )
        if len(update_records) != collector.algorithm_updates:
            raise ComparisonSmokeBlocked(
                f"{algorithm} observed update calls differ from collector counters"
            )
        if not all(record["return_finite"] for record in update_records):
            raise ComparisonSmokeBlocked(f"{algorithm} update returned non-finite values")
        if algorithm == "ppo" and collector.algorithm_updates != 2:
            raise ComparisonSmokeBlocked(
                f"PPO update count is {collector.algorithm_updates}, expected 2"
            )
        if algorithm == "ppo" and collector.optimizer_steps != 8:
            raise ComparisonSmokeBlocked(
                f"PPO optimizer-step count is {collector.optimizer_steps}, expected 8"
            )
        if algorithm == "sac":
            if collector.algorithm_updates < args.min_sac_updates:
                raise ComparisonSmokeBlocked(
                    f"SAC update count is {collector.algorithm_updates}, below "
                    f"minimum {args.min_sac_updates}"
                )
            if collector.optimizer_steps != collector.algorithm_updates * 3:
                raise ComparisonSmokeBlocked("SAC optimizer-step counter mismatch")
            if collector.sac_utd_ratio != args.sac_utd_ratio:
                raise ComparisonSmokeBlocked("SAC UTD ratio counter mismatch")
            if collector.replay_first_ready_at != args.sac_batch_size:
                raise ComparisonSmokeBlocked("SAC replay readiness counter mismatch")
        after = comparison_rl_module.parameter_vector(agent)
        if before.shape != after.shape or not bool(torch.isfinite(after).all()):
            raise ComparisonSmokeBlocked(f"{algorithm} parameters are invalid")
        parameter_delta = after - before
        if not bool(torch.any(parameter_delta != 0)):
            raise ComparisonSmokeBlocked(f"{algorithm} parameters did not change")
        losses = comparison_rl_module.finite_losses(agent)
        aggregate = env.extras.get("rj45_comparison_aggregate")
        if not isinstance(aggregate, dict):
            raise ComparisonSmokeBlocked("comparison aggregate telemetry is missing")
        aggregate_json = {
            name: value.detach().cpu().tolist()
            for name, value in aggregate.items()
        }
        for path, digest in frozen_hashes.items():
            if file_sha256(path) != digest:
                raise ComparisonSmokeBlocked(f"frozen input changed during smoke: {path}")
        result = {
            "algorithm": algorithm,
            "status": "completed",
            "readiness": readiness,
            "collector": asdict(collector),
            "observed_agent_update_calls": len(update_records),
            "updates": update_records,
            "algorithm_settings": {
                "ppo_rollout_steps_per_environment": args.ppo_rollout_steps,
                "sac_batch_size": args.sac_batch_size,
                "sac_utd_ratio": args.sac_utd_ratio,
                "sac_vector_steps": args.sac_vector_steps,
                "minimum_sac_updates": args.min_sac_updates,
                "scope": (
                    "G0 tensor-path and optimizer execution smoke only; these "
                    "settings do not establish a fair PPO/SAC outcome comparison"
                ),
            },
            "finite_losses": losses,
            "parameter_delta": {
                "element_count": int(parameter_delta.numel()),
                "changed_element_count": int((parameter_delta != 0).sum().item()),
                "l2": float(torch.linalg.vector_norm(parameter_delta).item()),
                "finite": True,
            },
            "comparison_aggregate": aggregate_json,
            "context": context,
            "frozen_input_sha256": {
                str(path): digest for path, digest in frozen_hashes.items()
            },
        }
        progress.record(
            "algorithm_complete",
            algorithm=algorithm,
            algorithm_updates=collector.algorithm_updates,
            optimizer_steps=collector.optimizer_steps,
        )
        return result
    finally:
        progress.record("environment_close_begin", algorithm=algorithm)
        env.close()
        progress.record("environment_close_complete", algorithm=algorithm)
        torch.cuda.empty_cache()


def _finite_structure(value: Any) -> bool:
    import torch

    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_finite_structure(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite_structure(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(float(value))
    return True


def prepare_next_environment(progress: ProgressRecorder) -> None:
    """Return Kit to a stopped, fresh-stage state before constructing another env."""

    import isaacsim.core.utils.stage as stage_utils
    import omni.kit.app
    import omni.timeline
    from isaaclab.sim import SimulationContext

    if SimulationContext.instance() is not None:
        raise ComparisonSmokeBlocked(
            "previous SimulationContext remains registered after environment close"
        )

    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    progress.record(
        "environment_transition_begin",
        timeline_playing=bool(timeline.is_playing()),
        timeline_stopped=bool(timeline.is_stopped()),
    )
    if not timeline.is_stopped():
        progress.record("timeline_stop_requested")
        timeline.stop()
        app.update()
        progress.record(
            "timeline_stop_event_flushed",
            timeline_playing=bool(timeline.is_playing()),
            timeline_stopped=bool(timeline.is_stopped()),
        )
    if timeline.is_playing() or not timeline.is_stopped():
        raise ComparisonSmokeBlocked(
            "timeline did not reach stopped state before environment reconstruction"
        )

    progress.record("fresh_stage_creation_begin")
    stage = stage_utils.create_new_stage()
    stage_utils.update_stage()
    if stage is None or stage_utils.get_current_stage() is None:
        raise ComparisonSmokeBlocked(
            "fresh USD stage was not attached before environment reconstruction"
        )
    progress.record("fresh_stage_creation_complete")
    progress.record("environment_transition_complete")


def run_smoke(
    args: argparse.Namespace,
    prelaunch: Mapping[str, Any],
    progress: ProgressRecorder | None = None,
) -> dict[str, Any]:
    if progress is None:
        progress = ProgressRecorder()
    runtime_src = args.runtime_root.resolve() / "src"
    sys.path.insert(0, str(runtime_src))
    comparison_env_module = _load_current_module(
        args.current_root.resolve() / "src/rby1_assembly/rj45_comparison_env.py",
        "rby1_current_comparison_env",
    )
    comparison_rl_module = _load_current_module(
        args.current_root.resolve() / "src/rby1_assembly/rj45_comparison_rl.py",
        "rby1_current_comparison_rl",
    )
    results = []
    for algorithm_index, algorithm in enumerate(ALGORITHMS):
        results.append(
            _run_algorithm(
                args,
                algorithm,
                comparison_env_module,
                comparison_rl_module,
                progress,
            )
        )
        if algorithm_index + 1 < len(ALGORITHMS):
            prepare_next_environment(progress)
    origins = {
        name: str(Path(module.__file__).resolve())
        for name, module in sys.modules.items()
        if name == "rby1_assembly" or name.startswith("rby1_assembly.")
        if getattr(module, "__file__", None)
    }
    escaped = {
        name: path
        for name, path in origins.items()
        if not Path(path).is_relative_to(runtime_src)
    }
    if escaped:
        raise ComparisonSmokeBlocked(f"archived runtime import escaped: {escaped}")
    return {
        "schema_name": SCHEMA_NAME,
        "status": "completed",
        "gpu_executed": True,
        "device": args.device,
        "num_envs": args.num_envs,
        "algorithms": results,
        "prelaunch": dict(prelaunch),
        "runtime_import_origins": origins,
        "progress_events": progress.events,
        "claims": {
            "g0_isaac_integration_smoke_passed": True,
            "algorithm_fairness_proven": False,
            "method_effect_comparison_completed": False,
            "reinforcement_learning_effect_proven": False,
            "generalization_proven": False,
            "portfolio_result_ready": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--current-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--runtime-manifest",
        type=Path,
        default=Path("configs/rj45-oracle-runtime-validation-ulp2-v1.json"),
    )
    parser.add_argument(
        "--held-bank",
        type=Path,
        default=Path(
            "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json"
        ),
    )
    parser.add_argument(
        "--parallel-contact-evidence",
        type=Path,
        required=True,
        help="verified four-environment hand-contact evidence specification",
    )
    parser.add_argument(
        "--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json")
    )
    parser.add_argument(
        "--training-config",
        type=Path,
        default=Path("configs/rj45-training-target-offset-v1.json"),
    )
    parser.add_argument(
        "--runtime-config", type=Path, default=Path("configs/runtime.json")
    )
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--ppo-rollout-steps", type=int, default=8)
    parser.add_argument("--sac-vector-steps", type=int, default=16)
    parser.add_argument("--sac-batch-size", type=int, default=16)
    parser.add_argument("--sac-utd-ratio", type=float, default=1.0)
    parser.add_argument("--min-sac-updates", type=int, default=2)
    parser.add_argument("--readiness-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--device", choices=("cuda:0",), default="cuda:0")
    parser.add_argument(
        "--headless", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _close_simulation_app(simulation_app: Any, exit_code: int) -> int:
    """Close Kit without allowing SystemExit(0) to mask a smoke failure."""

    if simulation_app is None:
        return exit_code
    try:
        simulation_app.close()
    except SystemExit as error:
        close_code = error.code
        if close_code is None or close_code == 0:
            return exit_code
        return close_code if type(close_code) is int else 1
    except BaseException:
        traceback.print_exc()
        return 1
    return exit_code


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    args.runtime_root = args.runtime_root.resolve()
    args.evidence_root = args.evidence_root.resolve()
    args.current_root = args.current_root.resolve()
    if not args.runtime_manifest.is_absolute():
        args.runtime_manifest = (args.current_root / args.runtime_manifest).resolve()
    args.report = args.report.resolve()
    if args.report.exists():
        parser.error("report already exists")
    simulation_app = None
    exit_code = 0
    failure_stage = "prelaunch_validation"
    progress = ProgressRecorder()
    heartbeat_armed = _arm_traceback_heartbeat(progress)
    try:
        progress.record("execution_argument_validation_begin")
        validate_execution_args(args)
        progress.record("execution_argument_validation_complete")
        progress.record("runtime_validation_begin")
        prelaunch = validate_prelaunch_inputs(args)
        progress.record("runtime_validation_complete")
        if args.validate_only:
            report = {
                "schema_name": SCHEMA_NAME,
                "status": "validated_not_executed",
                "gpu_executed": False,
                "prelaunch": prelaunch,
                "progress_events": progress.events,
                "claims": {"g0_isaac_integration_smoke_passed": False},
            }
        else:
            failure_stage = "application_launch"
            from isaaclab.app import AppLauncher

            progress.record("application_launch_begin")
            simulation_app = AppLauncher(
                {"headless": args.headless, "device": args.device}
            ).app
            progress.record("application_launch_complete")
            failure_stage = "isaac_comparison_smoke"
            progress.record("comparison_smoke_begin")
            report = run_smoke(args, prelaunch, progress)
            progress.record("comparison_smoke_complete")
        failure_stage = "report_write"
        progress.record("report_write_begin", report_status=report["status"])
        report["progress_events"] = progress.events
        _atomic_write_new(args.report, report)
        progress.record("report_write_complete", report_status=report["status"])
        print(json.dumps({"status": report["status"], "report": str(args.report)}))
    except Exception as error:
        traceback.print_exc()
        failure = {
            "schema_name": SCHEMA_NAME,
            "status": "blocked",
            "gpu_executed": simulation_app is not None,
            "failure_stage": failure_stage,
            "error": {"type": type(error).__name__, "message": str(error)},
            "requested_process_exit_code": 2,
            "external_supervisor_must_validate_report_status_and_claims": True,
            "progress_events": progress.events,
            "claims": {"g0_isaac_integration_smoke_passed": False},
        }
        if not args.report.exists():
            _atomic_write_new(args.report, failure)
        exit_code = 2
    finally:
        progress.record("simulation_app_close_begin", app_created=simulation_app is not None)
        exit_code = _close_simulation_app(simulation_app, exit_code)
        progress.record(
            "simulation_app_close_complete",
            app_created=simulation_app is not None,
            requested_exit_code=exit_code,
        )
        if heartbeat_armed:
            faulthandler.cancel_dump_traceback_later()
            progress.record("traceback_heartbeat_cancelled")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
