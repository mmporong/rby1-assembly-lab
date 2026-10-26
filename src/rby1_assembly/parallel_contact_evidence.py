"""Fail-closed loader for batched cross-hand contact routing evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


PROVEN_NUM_ENVS = 4
REQUIRED_PROBE_CHECKS = frozenset(
    {
        "four_environments",
        "fast_physics_replication_disabled",
        "three_independent_regex_sources",
        "three_regex_filters_per_source",
        "source_order_matches_scene_environment_order",
        "all_four_env_loads_are_finite",
        "stored_integral_reference_is_finite",
        "drive_target_is_finite",
        "integral_headroom_is_bounded",
        "stored_integral_reference_respects_soft_limits",
        "env0_and_env2_receive_real_joint_motion",
        "env1_and_env3_remain_near_reset_joint_command",
        "env0_and_env2_detect_cross_hand_contact",
        "env1_and_env3_cross_hand_load_remains_zero",
        "reset_cross_hand_load_is_zero",
        "cross_environment_overlap_negative_control_is_zero",
        "each_environment_has_nine_allowed_hand_pairs",
    }
)
REQUIRED_SAT_CHECKS = frozenset(
    {
        "actual_body_pose_quaternions_are_unit",
        "each_active_environment_has_physical_contact_at_or_above_1N",
        "each_active_environment_has_authored_convex_sat_intersection_at_first_contact",
        "each_active_environment_has_authored_convex_sat_intersection_at_peak_contact",
    }
)
REQUIRED_CURRENT_SOURCE_PATHS = (
    "checks/probe_parallel_hand_contacts.py",
    "src/rby1_assembly/isaac_env.py",
    "src/rby1_assembly/hand_collision.py",
    "src/rby1_assembly/contact_loads.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _load_json(path: Path) -> Any:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_non_finite)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid parallel contact evidence JSON {path}: {error}") from error
    _validate_finite(value, str(path))
    return value


def _validate_json_lines(path: Path) -> int:
    row_count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL row at {path}:{line_number}")
                try:
                    value = json.loads(line, parse_constant=_reject_non_finite)
                except (json.JSONDecodeError, ValueError) as error:
                    raise ValueError(
                        f"invalid parallel contact trace row {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise ValueError(
                        f"parallel contact trace row must be an object at {path}:{line_number}"
                    )
                _validate_finite(value, f"{path}:{line_number}")
                row_count += 1
    except (OSError, UnicodeError) as error:
        raise ValueError(f"invalid parallel contact trace {path}: {error}") from error
    if row_count == 0:
        raise ValueError("parallel contact trace must contain at least one row")
    return row_count


def _validate_finite(value: Any, location: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string JSON key at {location}")
            _validate_finite(item, f"{location}.{key}")
        return
    raise ValueError(f"unsupported JSON value at {location}: {type(value).__name__}")


def _required_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"parallel contact evidence requires non-empty {key}")
    return value


def _required_sha(mapping: Mapping[str, Any], key: str) -> str:
    value = _required_string(mapping, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"parallel contact evidence {key} must be a lowercase SHA-256")
    return value


def _resolve_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _validate_true_checks(
    payload: Mapping[str, Any], required: frozenset[str], label: str
) -> None:
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ValueError(f"{label} checks are missing")
    missing = sorted(required - checks.keys())
    false_checks = sorted(name for name in required if checks.get(name) is not True)
    extra_false = sorted(name for name, passed in checks.items() if passed is not True)
    if missing or false_checks or extra_false:
        raise ValueError(
            f"{label} checks are not all true: missing={missing}, false={false_checks}, "
            f"other_non_true={extra_false}"
        )
    if payload.get("failed_checks") != []:
        raise ValueError(f"{label} failed_checks must be empty")


def _validate_report_routing(report: Mapping[str, Any]) -> None:
    routing = report.get("sensor_routing")
    scene = report.get("scene")
    if not isinstance(routing, dict) or len(routing) != 3 or not isinstance(scene, dict):
        raise ValueError("probe report routing or scene metadata is incomplete")
    env_paths = scene.get("environment_prim_paths")
    if not isinstance(env_paths, list) or len(env_paths) != PROVEN_NUM_ENVS:
        raise ValueError("probe report must contain four environment prim paths")
    if scene.get("replicate_physics") is not False:
        raise ValueError("probe report must have replicate_physics=false")
    for source, entry in routing.items():
        if not isinstance(source, str) or not isinstance(entry, dict):
            raise ValueError("probe report sensor routing entry is invalid")
        expected_sources = [f"{env_path}/Robot/{source}" for env_path in env_paths]
        for key in (
            "expected_source_paths",
            "body_physx_view_prim_paths",
            "body_physx_backend_prim_paths",
            "contact_physx_view_sensor_paths",
        ):
            if entry.get(key) != expected_sources:
                raise ValueError(f"probe report source ordering mismatch for {source}.{key}")
        if entry.get("body_count") != PROVEN_NUM_ENVS:
            raise ValueError(f"probe report body count mismatch for {source}")
        if entry.get("contact_sensor_count") != PROVEN_NUM_ENVS:
            raise ValueError(f"probe report contact sensor count mismatch for {source}")
        if entry.get("configured_filter_regex_count") != 3 or entry.get(
            "contact_filter_count"
        ) != 3:
            raise ValueError(f"probe report filter count mismatch for {source}")
        expected_filters = entry.get("expected_contact_physx_view_filter_paths")
        if not isinstance(expected_filters, list) or len(expected_filters) != 12:
            raise ValueError(f"probe report filter ordering metadata is incomplete for {source}")
        if entry.get("contact_physx_view_filter_paths") != expected_filters:
            raise ValueError(f"probe report resolved filter ordering mismatch for {source}")


def _validate_sat_report(sat: Mapping[str, Any]) -> None:
    if sat.get("schema_version") != 1 or sat.get("passed") is not True:
        raise ValueError("SAT report is not a passing schema-version-1 report")
    _validate_true_checks(sat, REQUIRED_SAT_CHECKS, "SAT report")
    environments = sat.get("environments")
    if not isinstance(environments, dict) or not {"0", "2"}.issubset(environments):
        raise ValueError("SAT report is missing active environments 0 and 2")
    for env_index in ("0", "2"):
        result = environments[env_index]
        if not isinstance(result, dict):
            raise ValueError(f"SAT environment {env_index} result is invalid")
        for sample in ("first_threshold_contact", "maximum_recorded_contact_load"):
            frame = result.get(sample)
            if not isinstance(frame, dict) or frame.get("intersection_count", 0) < 1:
                raise ValueError(f"SAT environment {env_index} lacks overlap at {sample}")


def load_parallel_contact_evidence(
    specification: Mapping[str, Any], *, project_root: Path | None = None
) -> dict[str, Any]:
    """Validate pinned four-env probe evidence and current routing sources.

    The returned mapping is safe to attach to runtime metadata. The evidence
    proves the four-environment routing probe only; every actual runtime still
    has its own native PhysX source/filter path ordering checked by the monitor.
    """

    if not isinstance(specification, Mapping):
        raise TypeError("parallel contact evidence must be a mapping")
    root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    paths = {
        "report": _resolve_path(_required_string(specification, "report_path"), root),
        "trace": _resolve_path(_required_string(specification, "trace_path"), root),
        "sat_report": _resolve_path(
            _required_string(specification, "sat_report_path"), root
        ),
    }
    expected_hashes = {
        name: _required_sha(specification, f"{name}_sha256") for name in paths
    }
    actual_hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise ValueError(f"parallel contact evidence file is missing: {path}")
        actual_hashes[name] = _sha256(path)
        if actual_hashes[name] != expected_hashes[name]:
            raise ValueError(
                f"parallel contact evidence {name} SHA-256 mismatch: "
                f"expected={expected_hashes[name]}, actual={actual_hashes[name]}"
            )

    report = _load_json(paths["report"])
    sat = _load_json(paths["sat_report"])
    trace_row_count = _validate_json_lines(paths["trace"])
    if not isinstance(report, dict) or not isinstance(sat, dict):
        raise ValueError("parallel contact evidence reports must be JSON objects")
    if (
        report.get("schema_version") != 1
        or report.get("passed") is not True
        or report.get("physics_executed") is not True
        or report.get("num_envs") != PROVEN_NUM_ENVS
        or report.get("experiment") != "rb_y1_parallel_cross_hand_contact_routing_probe"
    ):
        raise ValueError("probe report is not the required passing four-environment experiment")
    _validate_true_checks(report, REQUIRED_PROBE_CHECKS, "probe report")
    _validate_report_routing(report)
    _validate_sat_report(sat)

    report_provenance = report.get("provenance")
    sat_provenance = sat.get("provenance")
    if not isinstance(report_provenance, dict) or not isinstance(sat_provenance, dict):
        raise ValueError("parallel contact evidence provenance is missing")
    if report_provenance.get("trace_sha256") != actual_hashes["trace"]:
        raise ValueError("probe report trace SHA-256 does not match the pinned trace")
    if sat_provenance.get("trace_sha256") != actual_hashes["trace"]:
        raise ValueError("SAT report trace SHA-256 does not match the pinned trace")
    source_hashes = report_provenance.get("source_sha256")
    if not isinstance(source_hashes, dict):
        raise ValueError("probe report source hashes are missing")
    current_source_hashes = {}
    for relative_path in REQUIRED_CURRENT_SOURCE_PATHS:
        current_path = root / relative_path
        current_hash = _sha256(current_path)
        current_source_hashes[relative_path] = current_hash
        if source_hashes.get(relative_path) != current_hash:
            raise ValueError(
                f"parallel contact evidence source mismatch for {relative_path}: "
                f"evidence={source_hashes.get(relative_path)}, current={current_hash}"
            )

    return {
        "schema_version": 1,
        "validated_probe_num_envs": PROVEN_NUM_ENVS,
        "runtime_num_envs_requires_native_ordering_check": True,
        "scope": (
            "4env routing proof; target num_envs runtime asserted; "
            "full task validation pending"
        ),
        "report_sha256": actual_hashes["report"],
        "trace_sha256": actual_hashes["trace"],
        "sat_report_sha256": actual_hashes["sat_report"],
        "trace_row_count": trace_row_count,
        "current_source_sha256": current_source_hashes,
    }


__all__ = ["PROVEN_NUM_ENVS", "load_parallel_contact_evidence"]
