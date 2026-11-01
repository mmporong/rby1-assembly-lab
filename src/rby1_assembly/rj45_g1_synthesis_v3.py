"""Pure-JSON synthesis for independent RJ45 G1-v3 review readiness."""

from __future__ import annotations

import hashlib
import json
import math
from statistics import median
from typing import Any, Mapping, Sequence


class G1SynthesisV3Error(ValueError):
    """The bundle is not finite canonical JSON or has no usable outer shape."""


FALSE_CLAIMS = {
    "task_admission_completed": False,
    "rl_pilot_admitted": False,
    "training_authorized": False,
}
GRASPS = ("rj45-train-02", "rj45-train-05")
CASES = (
    "tx-positive", "tx-negative", "tz-positive", "tz-negative",
    "rx-positive", "rx-negative",
)
RAW_METHOD_SLOTS = {
    "zero": (1, 2),
    "mean_constant": (1,),
    "open_loop_playback": (1,),
    "classical_force_search": (1,),
    "oracle_gt_reference": (1,),
}
SHARED_CURRENT_ALLOWLIST = (
    "scripts/smoke_rj45_comparison_isaac.py",
    "src/rby1_assembly/rj45_comparison_env.py",
    "src/rby1_assembly/rj45_comparison_providers.py",
    "src/rby1_assembly/rj45_task_admission.py",
    "src/rby1_assembly/rj45_task_admission_env.py",
)
TASK_CONTRACT = [
    0.00005,
    0.008726646259971648,
    0.012,
    0.013,
    0.4,
    25.0,
    0.005,
    0.05,
]
RAW_SCOPE = {
    "source_closed_full_se3_evaluator_terminal_scalar": True,
    "per_frame_geometry_replay": False,
    "external_verification_required": True,
}
RAW_ENVELOPE_KEYS = {
    "schema_name", "report_sha256", "archive_manifest_sha256",
    "archive_payload_sha256", "reviewed_source_freeze_sha256",
    "physics_compatibility_sha256", "raw_replay_verified",
    "row_integrity_verified", "archive_verified", "runtime_closure_verified",
    "identity_verified", "schema_verified", "terminal_verified",
    "execution_error_free", "bias_contract_verified",
    "transition_contract_verified", "verification_scope",
}
CLASSICAL_ROW_KEYS = {
    "schema_name", "report_sha256", "trace_sha256",
    "archive_payload_sha256", "reviewed_source_freeze_sha256",
    "physics_compatibility_sha256", "verified_tuning_rows_sha256",
    "raw_replay_verified", "archive_verified", "runtime_closure_verified",
    "pre_end_physics_matched", "wrapper",
}


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G1SynthesisV3Error("bundle must be finite canonical JSON") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _is_sha(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _mapping(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _exact_keys(value: Any, keys: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == keys


def _finite_nonnegative(value: Any) -> bool:
    return (
        type(value) in (int, float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _add(errors: list[str], condition: bool, code: str) -> bool:
    if not condition:
        errors.append(code)
        return False
    return True


def _raw_expected_identities() -> set[tuple[Any, ...]]:
    tuning = {
        ("tuning", "rj45-train-03", case, "classical_force_search", 1, candidate)
        for candidate in ("A", "B", "C") for case in CASES
    }
    admission = {
        ("admission", grasp, case, method, slot, None)
        for grasp in GRASPS for case in CASES
        for method, slots in RAW_METHOD_SLOTS.items() for slot in slots
    }
    return tuning | admission


def _validate_raw90(raw90: Any, errors: list[str]) -> dict[str, Any]:
    result = {
        "complete": False,
        "zero_slot1_safe_successes": 0,
        "oracle_safe_successes": 0,
        "oracle_safe_successes_by_grasp": {grasp: 0 for grasp in GRASPS},
        "oracle_unsafe_episodes": 0,
        "headroom_cases": 0,
        "rx_control_pairs": 0,
        "method_hard_or_forbidden_episodes": {
            method: 0 for method in RAW_METHOD_SLOTS
        },
    }
    if not _exact_keys(raw90, {"envelope", "rows"}):
        errors.append("raw90_shape")
        return result
    envelope = raw90["envelope"]
    if not _exact_keys(envelope, RAW_ENVELOPE_KEYS):
        errors.append("raw90_envelope_shape")
    else:
        _add(errors, envelope["schema_name"] == "rj45-g1-raw90-envelope-v3", "raw90_envelope_schema")
        for key in (
            "report_sha256", "archive_manifest_sha256", "archive_payload_sha256",
            "reviewed_source_freeze_sha256", "physics_compatibility_sha256",
        ):
            _add(errors, _is_sha(envelope[key]), f"raw90_{key}")
        for key in (
            "raw_replay_verified", "row_integrity_verified", "archive_verified",
            "runtime_closure_verified", "identity_verified", "schema_verified",
            "terminal_verified", "execution_error_free", "bias_contract_verified",
            "transition_contract_verified",
        ):
            _add(errors, envelope[key] is True, f"raw90_{key}")
        _add(errors, envelope["verification_scope"] == RAW_SCOPE, "raw90_verification_scope")
    rows = raw90["rows"]
    if type(rows) is not list or len(rows) != 90:
        errors.append("raw90_exact_episode_count")
        return result
    identities: list[tuple[Any, ...]] = []
    indexed: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    row_ok = True
    required = {
        "schema_name", "ordinal", "stage", "grasp_id", "case_id", "method_id",
        "slot", "candidate_id", "success", "hard_contact", "forbidden_contact",
        "execution_error", "terminal_complete", "requested_applied_bias_valid",
        "transition_contract_valid",
    }
    for index, row in enumerate(rows, 1):
        if not _exact_keys(row, required):
            errors.append(f"raw90_row_shape:{index}")
            row_ok = False
            continue
        identity = (
            row["stage"], row["grasp_id"], row["case_id"], row["method_id"],
            row["slot"], row["candidate_id"],
        )
        identities.append(identity)
        checks = (
            row["schema_name"] == "rj45-g1-raw90-verified-row-v3",
            row["ordinal"] == index,
            type(row["success"]) is bool,
            type(row["hard_contact"]) is bool,
            type(row["forbidden_contact"]) is bool,
            row["execution_error"] is None,
            row["terminal_complete"] is True,
            row["requested_applied_bias_valid"] is True,
            row["transition_contract_valid"] is True,
        )
        if not all(checks):
            errors.append(f"raw90_row_contract:{index}")
            row_ok = False
        if row["stage"] == "admission":
            indexed[(row["grasp_id"], row["case_id"], row["method_id"], row["slot"])] = row
            if row["method_id"] in result["method_hard_or_forbidden_episodes"]:
                result["method_hard_or_forbidden_episodes"][row["method_id"]] += int(
                    row["hard_contact"] or row["forbidden_contact"]
                )
    if len(set(identities)) != 90 or set(identities) != _raw_expected_identities():
        errors.append("raw90_identity_set")
        row_ok = False
    if not row_ok:
        return result
    for grasp in GRASPS:
        for case in CASES:
            zero = indexed[(grasp, case, "zero", 1)]
            oracle = indexed[(grasp, case, "oracle_gt_reference", 1)]
            zero_safe = zero["success"] and not zero["hard_contact"] and not zero["forbidden_contact"]
            oracle_safe = oracle["success"] and not oracle["hard_contact"] and not oracle["forbidden_contact"]
            result["zero_slot1_safe_successes"] += int(zero_safe)
            result["oracle_safe_successes"] += int(oracle_safe)
            result["oracle_safe_successes_by_grasp"][grasp] += int(oracle_safe)
            result["oracle_unsafe_episodes"] += int(oracle["hard_contact"] or oracle["forbidden_contact"])
            result["headroom_cases"] += int(oracle_safe and not zero_safe)
        for sign in ("positive", "negative"):
            zero = indexed[(grasp, f"rx-{sign}", "zero", 1)]
            oracle = indexed[(grasp, f"rx-{sign}", "oracle_gt_reference", 1)]
            if oracle["success"] and not oracle["hard_contact"] and not oracle["forbidden_contact"] and not zero["success"]:
                result["rx_control_pairs"] += 1
    result["complete"] = True
    return result


def _validate_aggregate(
    name: str,
    value: Any,
    config: Mapping[str, Any],
    expected_count: int,
    errors: list[str],
) -> Mapping[str, Any] | None:
    if not _exact_keys(value, {"aggregate", "recomputed"}):
        errors.append(f"observability_{name}_shape")
        return None
    aggregate = value["aggregate"]
    if not isinstance(aggregate, Mapping) or aggregate != value["recomputed"]:
        errors.append(f"observability_{name}_recomputation")
        return None
    expected_status = f"{name}_pass"
    results = aggregate.get("results")
    checks = (
        aggregate.get("schema_name") == "rj45-force-slope-stage-adjudication-v3",
        aggregate.get("stage") == name,
        aggregate.get("status") == expected_status,
        aggregate.get("pass") is True,
        aggregate.get("episode_count") == expected_count,
        aggregate.get("config") == config,
        aggregate.get("config_sha256") == _sha(config),
        type(results) is list,
        type(results) is list and len(results) == expected_count,
        _is_sha(aggregate.get("expected_sources_sha256")),
        aggregate.get("claims") == {
            "task_admission_completed": False, "rl_pilot_admitted": False,
        },
    )
    if not all(checks):
        errors.append(f"observability_{name}_aggregate_contract")
        return None
    return aggregate


def _validate_observability(value: Any, errors: list[str]) -> dict[str, Any]:
    result = {"development_complete": False, "confirmation_complete": False}
    keys = {"config", "development", "confirmation", "current_controller_sources"}
    if not _exact_keys(value, keys):
        errors.append("observability_shape")
        return result
    config = value["config"]
    if not isinstance(config, Mapping):
        errors.append("observability_config_shape")
        return result
    if not (
        config.get("schema") == "rj45-force-slope-observability-contract-v3"
        and config.get("frozen") is True
        and config.get("development_episode_count") == 6
        and config.get("confirmation_episode_count") == 24
        and config.get("held_out_access") is False
    ):
        errors.append("observability_frozen_config")
    development = _validate_aggregate("development", value["development"], config, 6, errors)
    confirmation = _validate_aggregate("confirmation", value["confirmation"], config, 24, errors)
    result["development_complete"] = development is not None
    result["confirmation_complete"] = confirmation is not None
    if development is None or confirmation is None:
        return result
    current = value["current_controller_sources"]
    if not isinstance(current, Mapping) or set(current) != {"force_slope_probe", "active_axis_probe"}:
        errors.append("observability_current_controller_sources")
        return result
    for provider_id, binding in current.items():
        if not _exact_keys(binding, {"controller_sha256", "provider_source_sha256", "provider_import_origin"}):
            errors.append(f"observability_current_binding:{provider_id}")
        elif not (
            _is_sha(binding["controller_sha256"])
            and _is_sha(binding["provider_source_sha256"])
            and type(binding["provider_import_origin"]) is str
            and bool(binding["provider_import_origin"])
        ):
            errors.append(f"observability_current_binding_values:{provider_id}")
    development_sources = development.get("controller_sources")
    confirmation_sources = confirmation.get("controller_sources")
    if not (
        isinstance(development_sources, Mapping)
        and isinstance(confirmation_sources, Mapping)
        and development_sources.get("force_slope_probe") == current.get("force_slope_probe")
        and confirmation_sources == current
    ):
        errors.append("observability_controller_source_origin_binding")
    if development["expected_sources_sha256"] != confirmation["expected_sources_sha256"]:
        errors.append("observability_source_freeze_mismatch")
    return result


def _percentile95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower))


def _recompute_classical_selection(
    config: Mapping[str, Any], wrappers: Any, errors: list[str],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    if type(wrappers) is not list or len(wrappers) != 24:
        errors.append("classical_tuning_exact_count")
        return None, None, None
    candidates = config.get("candidates")
    cases = config.get("signed_cases")
    if not (
        type(candidates) is list and len(candidates) == 4
        and type(cases) is list and tuple(cases) == CASES
    ):
        errors.append("classical_config_candidates")
        return None, None, None
    candidate_ids = [candidate.get("id") for candidate in candidates if isinstance(candidate, Mapping)]
    if len(candidate_ids) != 4 or len(set(candidate_ids)) != 4:
        errors.append("classical_candidate_ids")
        return None, None, None
    expected = [(candidate_id, case) for candidate_id in candidate_ids for case in CASES]
    metrics_by_candidate: dict[str, list[Mapping[str, Any]]] = {item: [] for item in candidate_ids}
    controller_by_candidate: dict[str, str] = {}
    registered_config_sha = _sha(config)
    classical_source_sha: str | None = None
    for index, (wrapper, identity) in enumerate(zip(wrappers, expected, strict=True), 1):
        if not _exact_keys(wrapper, {"slot", "verification", "metrics"}):
            errors.append(f"classical_tuning_wrapper_shape:{index}")
            continue
        slot, verification, metrics = wrapper["slot"], wrapper["verification"], wrapper["metrics"]
        candidate_id, case = identity
        if not (
            isinstance(slot, Mapping)
            and slot.get("ordinal") == index
            and slot.get("stage") == "classical_v2_tuning"
            and slot.get("candidate_id") == candidate_id
            and slot.get("grasp_id") == "rj45-train-03"
            and slot.get("case_id") == case
        ):
            errors.append(f"classical_tuning_slot:{index}")
            continue
        if not _exact_keys(
            verification,
            {"verified", "registered_config_sha256", "classical_source_sha256", "controller_sha256"},
        ) or not (
            verification["verified"] is True
            and verification["registered_config_sha256"] == registered_config_sha
            and _is_sha(verification["classical_source_sha256"])
            and _is_sha(verification["controller_sha256"])
            and slot.get("registered_config_sha256") == registered_config_sha
            and slot.get("controller_sha256") == verification["controller_sha256"]
        ):
            errors.append(f"classical_tuning_verification:{index}")
            continue
        if classical_source_sha is None:
            classical_source_sha = verification["classical_source_sha256"]
        elif classical_source_sha != verification["classical_source_sha256"]:
            errors.append("classical_tuning_source_mismatch")
        previous = controller_by_candidate.setdefault(candidate_id, verification["controller_sha256"])
        if previous != verification["controller_sha256"]:
            errors.append(f"classical_tuning_controller_mismatch:{candidate_id}")
        if not _exact_keys(
            metrics,
            {"success", "hard_contact", "forbidden_contact", "evaluator_force_p95_n", "completion_time_s"},
        ) or not (
            type(metrics["success"]) is bool
            and type(metrics["hard_contact"]) is bool
            and type(metrics["forbidden_contact"]) is bool
            and _finite_nonnegative(metrics["evaluator_force_p95_n"])
            and _finite_nonnegative(metrics["completion_time_s"])
        ):
            errors.append(f"classical_tuning_metrics:{index}")
            continue
        metrics_by_candidate[candidate_id].append(metrics)
    if any(len(rows) != 6 for rows in metrics_by_candidate.values()):
        errors.append("classical_tuning_incomplete")
        return None, classical_source_sha, None
    eligible = []
    for candidate_id in candidate_ids:
        rows = metrics_by_candidate[candidate_id]
        if any(row["hard_contact"] or row["forbidden_contact"] for row in rows):
            continue
        successes = sum(row["success"] for row in rows)
        eligible.append({
            "candidate_id": candidate_id,
            "success_count": successes,
            "evaluator_force_p95_n": _percentile95(
                [float(row["evaluator_force_p95_n"]) for row in rows]
            ),
            "completion_time_median_s": float(median(
                [float(row["completion_time_s"]) for row in rows]
            )),
            "strong_baseline_qualifier_met": successes >= 4,
        })
    if not eligible:
        errors.append("classical_no_safe_tuning_candidate")
        return None, classical_source_sha, None
    selected = min(eligible, key=lambda row: (
        -row["success_count"], row["evaluator_force_p95_n"],
        row["completion_time_median_s"], row["candidate_id"],
    ))
    selected = {
        **selected,
        "safe": True,
        "new_admission_baseline_episode_count": 12,
        "strong_qualifier_does_not_gate_baseline": True,
    }
    return selected, classical_source_sha, controller_by_candidate[selected["candidate_id"]]


def _bias(case_id: str) -> tuple[list[float], list[float]]:
    sign = 1.0 if case_id.endswith("positive") else -1.0
    if case_id.startswith("tx"):
        return [sign * 0.00012, 0.0, 0.0], [0.0, 0.0, 0.0]
    if case_id.startswith("tz"):
        return [0.0, 0.0, sign * 0.00012], [0.0, 0.0, 0.0]
    return [0.0, 0.0, 0.0], [sign * 0.003490658503988659, 0.0, 0.0]


def _validate_classical(value: Any, errors: list[str]) -> dict[str, Any]:
    result = {
        "tuning_complete": False,
        "admission_complete": False,
        "selected_candidate_id": None,
        "tuning_successes": 0,
        "strong_baseline_qualifier_met": False,
        "fresh_safe_successes": 0,
        "fresh_unsafe_episodes": 0,
    }
    keys = {"config", "verified_tuning_rows", "selection", "admission_rows", "source_bindings"}
    if not _exact_keys(value, keys):
        errors.append("classical_shape")
        return result
    config = value["config"]
    shared_contract = config.get("shared_controller_contract") if isinstance(config, Mapping) else None
    if not isinstance(config, Mapping) or not isinstance(shared_contract, Mapping) or not (
        config.get("schema") == "rj45-classical-tuning-v2"
        and config.get("frozen") is True
        and config.get("tuning_episode_count") == 24
        and shared_contract.get("known_bias_access") is False
        and shared_contract.get("case_grasp_sign_ground_truth_access") is False
    ):
        errors.append("classical_frozen_config")
        return result
    selected, classical_source_sha, selected_controller_sha = _recompute_classical_selection(
        config, value["verified_tuning_rows"], errors
    )
    if selected is None:
        return result
    result.update({
        "tuning_complete": True,
        "selected_candidate_id": selected["candidate_id"],
        "tuning_successes": selected["success_count"],
        "strong_baseline_qualifier_met": selected["strong_baseline_qualifier_met"],
    })
    if value["selection"] != selected:
        errors.append("classical_selection_recomputation")
    tuning_sha = _sha(value["verified_tuning_rows"])
    source_bindings = value["source_bindings"]
    if not _exact_keys(source_bindings, {"tuning", "admission"}):
        errors.append("classical_source_bindings_shape")
    else:
        tuning_binding, admission_binding = source_bindings["tuning"], source_bindings["admission"]
        if not isinstance(tuning_binding, Mapping) or not tuning_binding or not all(
            type(key) is str and bool(key) and _is_sha(item)
            for key, item in tuning_binding.items()
        ):
            errors.append("classical_tuning_source_binding")
        elif admission_binding != {**dict(tuning_binding), "verified_tuning_rows_sha256": tuning_sha}:
            errors.append("classical_admission_source_relation")
    rows = value["admission_rows"]
    if type(rows) is not list or len(rows) != 12:
        errors.append("classical_admission_exact_count")
        return result
    expected = [(grasp, case) for grasp in GRASPS for case in CASES]
    for index, (row, (grasp, case)) in enumerate(zip(rows, expected, strict=True), 1):
        if not _exact_keys(row, CLASSICAL_ROW_KEYS):
            errors.append(f"classical_admission_row_shape:{index}")
            continue
        for key in (
            "report_sha256", "trace_sha256", "archive_payload_sha256",
            "reviewed_source_freeze_sha256", "physics_compatibility_sha256",
        ):
            _add(errors, _is_sha(row[key]), f"classical_admission_{key}:{index}")
        checks = (
            row["schema_name"] == "rj45-g1-classical-verified-row-v3",
            row["verified_tuning_rows_sha256"] == tuning_sha,
            row["raw_replay_verified"] is True,
            row["archive_verified"] is True,
            row["runtime_closure_verified"] is True,
            row["pre_end_physics_matched"] is True,
        )
        if not all(checks):
            errors.append(f"classical_admission_envelope:{index}")
        wrapper = row["wrapper"]
        if not _exact_keys(wrapper, {"slot", "verification", "metrics"}):
            errors.append(f"classical_admission_wrapper_shape:{index}")
            continue
        translation, rotation = _bias(case)
        expected_slot = {
            "ordinal": index,
            "stage": "classical_v2_new_admission_baseline",
            "candidate_id": selected["candidate_id"],
            "controller_sha256": selected_controller_sha,
            "grasp_id": grasp,
            "case_id": case,
            "environment_seed": 20261121,
            "requested_translation_bias_entry_m": translation,
            "requested_rotation_bias_entry_rad": rotation,
            "requested_applied_bias_absolute_tolerance": 1.0e-9,
        }
        verification = wrapper["verification"]
        expected_verification = {
            "verified": True,
            "registered_config_sha256": _sha(config),
            "classical_source_sha256": classical_source_sha,
            "controller_sha256": selected_controller_sha,
        }
        metrics = wrapper["metrics"]
        if wrapper["slot"] != expected_slot or verification != expected_verification:
            errors.append(f"classical_admission_slot_or_verification:{index}")
        if not _exact_keys(
            metrics,
            {"success", "hard_contact", "forbidden_contact", "evaluator_force_p95_n", "completion_time_s"},
        ) or not (
            type(metrics["success"]) is bool
            and type(metrics["hard_contact"]) is bool
            and type(metrics["forbidden_contact"]) is bool
            and _finite_nonnegative(metrics["evaluator_force_p95_n"])
            and _finite_nonnegative(metrics["completion_time_s"])
        ):
            errors.append(f"classical_admission_metrics:{index}")
        else:
            unsafe = metrics["hard_contact"] or metrics["forbidden_contact"]
            result["fresh_unsafe_episodes"] += int(unsafe)
            result["fresh_safe_successes"] += int(metrics["success"] and not unsafe)
    result["admission_complete"] = not any(
        error.startswith("classical_admission") for error in errors
    )
    return result


def _validate_physical_seals(value: Any, errors: list[str]) -> bool:
    if not isinstance(value, Mapping) or set(value) != {"raw90", "observability", "classical"}:
        errors.append("physical_seals_shape")
        return False
    comparable = None
    for family in ("raw90", "observability", "classical"):
        seal = value[family]
        keys = {
            "evidence_normalized_sha256", "runtime_normalized_sha256",
            "runtime_versions", "held_bank_manifest_sha256",
            "held_artifact_sha256", "task_contract", "shared_current_sha256",
            "family_source_sha256", "full_commit",
        }
        if not _exact_keys(seal, keys):
            errors.append(f"physical_seal_shape:{family}")
            continue
        evidence = seal["evidence_normalized_sha256"]
        runtime = seal["runtime_normalized_sha256"]
        held = seal["held_artifact_sha256"]
        shared = seal["shared_current_sha256"]
        valid = (
            isinstance(evidence, Mapping) and len(evidence) == 50
            and isinstance(runtime, Mapping) and len(runtime) == 50
            and all(type(key) is str and _is_sha(item) for key, item in evidence.items())
            and all(type(key) is str and _is_sha(item) for key, item in runtime.items())
            and isinstance(seal["runtime_versions"], Mapping) and bool(seal["runtime_versions"])
            and _is_sha(seal["held_bank_manifest_sha256"])
            and isinstance(held, Mapping) and set(held) == set(GRASPS)
            and all(_is_sha(item) for item in held.values())
            and seal["task_contract"] == TASK_CONTRACT
            and isinstance(shared, Mapping) and set(shared) == set(SHARED_CURRENT_ALLOWLIST)
            and all(_is_sha(item) for item in shared.values())
            and _is_sha(seal["family_source_sha256"])
            and type(seal["full_commit"]) is str and bool(seal["full_commit"])
        )
        if not valid:
            errors.append(f"physical_seal_values:{family}")
            continue
        current = {
            "evidence_normalized_sha256": dict(evidence),
            "runtime_normalized_sha256": dict(runtime),
            "runtime_versions": dict(seal["runtime_versions"]),
            "held_bank_manifest_sha256": seal["held_bank_manifest_sha256"],
            "held_artifact_sha256": dict(held),
            "task_contract": list(seal["task_contract"]),
            "shared_current_sha256": dict(shared),
        }
        if comparable is None:
            comparable = current
        elif current != comparable:
            errors.append(f"physical_seal_family_mismatch:{family}")
    return comparable is not None and not any(error.startswith("physical_seal") for error in errors)


def _validate_framework(value: Any, errors: list[str]) -> bool:
    keys = {
        "archive_closure_verified", "source_closure_verified",
        "heldout_closure_verified", "probe_safety_closure_verified",
        "guard_closure_verified", "held_out_access_detected",
    }
    if not _exact_keys(value, keys):
        errors.append("framework_shape")
        return False
    for key in keys - {"held_out_access_detected"}:
        if value[key] is not True:
            errors.append(f"framework_{key}")
    if value["held_out_access_detected"] is not False:
        errors.append("framework_heldout_leakage")
    return not any(error.startswith("framework_") for error in errors)


def synthesize_g1_v3(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Validate sealed family evidence and synthesize non-authorizing G1-v3 state."""

    _canonical(bundle)
    if not isinstance(bundle, Mapping):
        raise G1SynthesisV3Error("bundle must be an object")
    outer_keys = {
        "schema_name", "raw90", "observability", "classical",
        "physical_seals", "framework",
    }
    contract_errors: list[str] = []
    outer_valid = set(bundle) == outer_keys
    if not outer_valid:
        contract_errors.append("bundle_shape")
    if bundle.get("schema_name") != "rj45-g1-synthesis-bundle-v3":
        contract_errors.append("bundle_schema")
    raw = _validate_raw90(bundle.get("raw90"), contract_errors)
    observability = _validate_observability(bundle.get("observability"), contract_errors)
    classical = _validate_classical(bundle.get("classical"), contract_errors)
    physical_complete = _validate_physical_seals(bundle.get("physical_seals"), contract_errors)
    framework_errors: list[str] = []
    framework_complete = _validate_framework(bundle.get("framework"), framework_errors)

    oracle_reasons = []
    if not raw["complete"]:
        oracle_reasons.append("raw90_incomplete")
    if raw["oracle_safe_successes"] < 10:
        oracle_reasons.append("oracle_safe_successes_below_10_of_12")
    if min(raw["oracle_safe_successes_by_grasp"].values()) < 5:
        oracle_reasons.append("oracle_safe_successes_below_5_of_6_per_grasp")
    if raw["oracle_unsafe_episodes"] != 0:
        oracle_reasons.append("oracle_unsafe_episode_present")

    controlled_reasons = []
    if not observability["development_complete"]:
        controlled_reasons.append("development_6_not_recomputed_pass")
    if not observability["confirmation_complete"]:
        controlled_reasons.append("confirmation_24_not_recomputed_pass")

    fatal_gates = {
        "contract_and_seal": {
            "pass": not contract_errors and physical_complete,
            "reasons": list(dict.fromkeys(contract_errors)),
        },
        "oracle_feasibility": {
            "pass": not oracle_reasons,
            "reasons": oracle_reasons,
        },
        "controlled_confirmation": {
            "pass": not controlled_reasons,
            "reasons": controlled_reasons,
        },
        "framework_and_probe_safety": {
            "pass": framework_complete,
            "reasons": list(dict.fromkeys(framework_errors)),
        },
    }
    fatal = any(not gate["pass"] for gate in fatal_gates.values())
    return {
        "schema_name": "rj45-g1-synthesis-v3",
        "state": "fatal_gate_failed_v3" if fatal else "eligible_for_independent_g1_review_v3",
        "diagnostic_only": True,
        "stage_completion": {
            "raw90": raw["complete"],
            "observability_development_6": observability["development_complete"],
            "observability_confirmation_24": observability["confirmation_complete"],
            "classical_tuning_24": classical["tuning_complete"],
            "classical_fresh_admission_12": classical["admission_complete"],
            "physical_seals": physical_complete,
            "framework_closure": framework_complete,
        },
        "fatal_gates": fatal_gates,
        "evidence_counts": {
            "raw90_episode_count": 90 if raw["complete"] else 0,
            "oracle_safe_successes": raw["oracle_safe_successes"],
            "oracle_safe_successes_by_grasp": raw["oracle_safe_successes_by_grasp"],
            "oracle_unsafe_episodes": raw["oracle_unsafe_episodes"],
            "controlled_development_episode_count": 6 if observability["development_complete"] else 0,
            "controlled_confirmation_episode_count": 24 if observability["confirmation_complete"] else 0,
            "classical_tuning_episode_count": 24 if classical["tuning_complete"] else 0,
            "classical_fresh_admission_episode_count": 12 if classical["admission_complete"] else 0,
        },
        "qualifiers": {
            "selected_classical_candidate_id": classical["selected_candidate_id"],
            "selected_tuning_successes_out_of_6": classical["tuning_successes"],
            "strong_baseline_4_of_6": classical["strong_baseline_qualifier_met"],
            "classical_fresh_safe_successes_out_of_12": classical["fresh_safe_successes"],
            "classical_fresh_unsafe_episodes": classical["fresh_unsafe_episodes"],
            "zero_slot1_safe_successes_out_of_12": raw["zero_slot1_safe_successes"],
            "oracle_zero_headroom_cases": raw["headroom_cases"],
            "rx_control_cases": raw["rx_control_pairs"],
            "raw_method_hard_or_forbidden_episodes": raw[
                "method_hard_or_forbidden_episodes"
            ],
            "qualifiers_do_not_change_fatal_gate_state": True,
        },
        "historical_authority": {
            "source_closed_full_se3_evaluator_terminal_scalar": True,
            "per_frame_geometry_replay": False,
            "old_v1_v2_success_retroactively_authoritative": False,
            "external_verification_required": True,
        },
        "claims": dict(FALSE_CLAIMS),
    }


__all__ = ["FALSE_CLAIMS", "G1SynthesisV3Error", "synthesize_g1_v3"]
