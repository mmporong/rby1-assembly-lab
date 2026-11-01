"""Pure-CPU packaging validation for sealed RJ45 evaluation results.

It verifies canonical JSON packaging and byte-level evidence linkage. It does not prove
physical validity, tensor semantics, freeze timing/authority, held-out independence,
actual GPU activity, provider billing, or scientific CI validity.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from statistics import fmean
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "rj45-sealed-results-v2"
METHODS = ("zero", "classical", "ppo", "sac")
BASELINE_METHODS = ("zero", "classical")
LEARNED_METHODS = ("ppo", "sac")
REQUIRED_ARTIFACT_KINDS = frozenset({"checkpoint", "controller", "source", "evaluator", "schedule", "g2_selection", "run_manifest", "episode", "evaluation_report"})
COMMON_REF_KEYS = frozenset({"source", "selection", "schedule", "evaluator", "observation_contract", "action_contract", "provider_authority", "control_contract"})
METRIC_SPECS = {
    "success": {"metric_id": "success_rate", "unit": "proportion", "source": "evaluation_report", "population": "fixed_grid_attempts", "aggregation": "all_attempts", "missingness": "invalid_as_failure"},
    "force": {"metric_id": "peak_force", "unit": "N", "source": "evaluation_report", "population": "fixed_grid_attempts", "aggregation": "all_attempts", "missingness": "preserve_missing"},
    "time": {"metric_id": "completion_time", "unit": "s", "source": "evaluation_report", "population": "fixed_grid_attempts", "aggregation": "outcome_stratified", "missingness": "preserve_outcome"},
    "raw_transitions": {"metric_id": "raw_transitions", "unit": "count", "source": "training_report", "population": "training_seed", "aggregation": "per_seed", "missingness": "not_applicable"},
    "resource_accounting": {"metric_id": "resource_accounting", "unit": "typed_fields", "source": "resource_evidence", "population": "policy_instance", "aggregation": "per_policy", "missingness": "not_available_without_evidence"},
}
ASSURANCE_NOT_ESTABLISHED = (
    "physical validity, tensor/file-format semantics, freeze timing, and evaluator authority",
    "held-out independence, actual GPU activity, and provider billing truth",
    "scientific confidence-interval validity or generalization beyond the fixed grid",
)


class ResultContractError(ValueError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    except (TypeError, ValueError) as exc:
        raise ResultContractError("payload must be canonical-JSON serializable") from exc


def _decode_canonical(data: bytes, label: str) -> Any:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultContractError(f"{label} must be UTF-8 JSON") from exc
    if _canonical_bytes(value) != data:
        raise ResultContractError(f"{label} must use canonical JSON bytes")
    return value


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _require_digest(value: Any, label: str) -> str:
    if not _valid_digest(value):
        raise ResultContractError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _exact(value: Mapping[str, Any], fields: set[str] | frozenset[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        actual = set(value) if isinstance(value, Mapping) else set()
        raise ResultContractError(f"{label} fields mismatch; missing={sorted(set(fields)-actual)}, extra={sorted(actual-set(fields))}")


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ResultContractError(f"{label} must be a sequence")
    return value


def _number(value: Any, label: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
        raise ResultContractError(f"{label} must be finite and >= {minimum}")
    return float(value)


@dataclass(frozen=True)
class ValidatedPreregistration:
    _manifest_bytes: bytes
    canonical_sha256: str

    @property
    def manifest(self) -> Mapping[str, Any]:
        return json.loads(self._manifest_bytes)

    @property
    def policy_instances(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest["policy_instances"])

    @property
    def evaluation_units(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.manifest["evaluation_units"])


@dataclass(frozen=True)
class ValidatedResultBundle:
    preregistration: ValidatedPreregistration
    _bundle_bytes: bytes
    canonical_sha256: str

    @property
    def bundle(self) -> Mapping[str, Any]:
        return json.loads(self._bundle_bytes)

    @property
    def episodes(self) -> tuple[Mapping[str, Any], ...]:
        return tuple(self.bundle["episodes"])


def _assert_preregistration(value: ValidatedPreregistration) -> Mapping[str, Any]:
    if not isinstance(value, ValidatedPreregistration):
        raise ResultContractError("validate_preregistration output is required")
    if _sha(value._manifest_bytes) != value.canonical_sha256:
        raise ResultContractError("stored preregistration digest mismatch")
    return _decode_canonical(value._manifest_bytes, "stored preregistration")


def _assert_bundle(value: ValidatedResultBundle) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(value, ValidatedResultBundle):
        raise ResultContractError("validate_result_bundle output is required")
    manifest = _assert_preregistration(value.preregistration)
    if _sha(value._bundle_bytes) != value.canonical_sha256:
        raise ResultContractError("stored result bundle digest mismatch")
    bundle = _decode_canonical(value._bundle_bytes, "stored result bundle")
    if bundle["manifest_sha256"] != value.preregistration.canonical_sha256:
        raise ResultContractError("stored bundle manifest digest mismatch")
    return manifest, bundle


def canonical_evaluation_unit_id(unit: Mapping[str, Any]) -> str:
    fields = ("condition", "grasp", "eval_seed", "reset_slot", "initial_snapshot_artifact_id", "initial_snapshot_sha256", "stratum", "magnitude", "magnitude_unit")
    return f"eu-{_sha(_canonical_bytes({field: unit.get(field) for field in fields}))}"


def canonical_result_row_sha256(row: Mapping[str, Any]) -> str:
    fields = ("policy_instance_id", "unit_id", "initial_snapshot_sha256", "stratum", "valid", "success", "force_peak_n", "actual_terminal_elapsed_seconds", "success_time_seconds", "time_outcome", "jammed")
    return _sha(_canonical_bytes({field: row.get(field) for field in fields}))


def _artifact_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for item in _sequence(manifest["artifacts"], "artifacts"):
        _exact(item, {"artifact_id", "kind", "sha256"}, "artifact declaration")
        artifact_id = item["artifact_id"]
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in result:
            raise ResultContractError("artifact IDs must be nonempty and unique")
        if not isinstance(item["kind"], str) or not item["kind"]:
            raise ResultContractError("artifact kind must be nonempty")
        _require_digest(item["sha256"], "artifact sha256")
        result[artifact_id] = item
    missing = REQUIRED_ARTIFACT_KINDS - {item["kind"] for item in result.values()}
    if missing:
        raise ResultContractError(f"missing artifact kinds: {sorted(missing)}")
    return result


def validate_preregistration(manifest: Mapping[str, Any], expected_sha256: str) -> ValidatedPreregistration:
    if not isinstance(manifest, Mapping):
        raise ResultContractError("manifest must be a mapping")
    data = _canonical_bytes(manifest)
    if _sha(data) != _require_digest(expected_sha256, "expected_sha256"):
        raise ResultContractError("manifest canonical SHA-256 mismatch")
    _exact(manifest, {"schema_version", "method_definitions", "observation_dimension", "right_residual_dimension", "common_contracts", "policy_instances", "evaluation_scope", "evaluation_units", "frozen_horizon_seconds", "uncertainty", "metrics", "artifacts"}, "manifest")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ResultContractError(f"schema_version must be {SCHEMA_VERSION}")
    if manifest["method_definitions"] != list(METHODS):
        raise ResultContractError("method_definitions must be exactly zero,classical,ppo,sac")
    if manifest["observation_dimension"] != 79 or manifest["right_residual_dimension"] != 6:
        raise ResultContractError("all methods must use 79D observations and 6D right residuals")
    _number(manifest["frozen_horizon_seconds"], "frozen_horizon_seconds", 1e-12)
    _exact(manifest["evaluation_scope"], {"kind", "population_contract_artifact_id"}, "evaluation_scope")
    if manifest["evaluation_scope"] != {"kind": "fixed_grid", "population_contract_artifact_id": None}:
        raise ResultContractError("only fixed_grid is supported; sampled populations require a future contract")
    metrics = manifest["metrics"]
    if not isinstance(metrics, Mapping) or "success" not in metrics or not set(metrics) <= set(METRIC_SPECS):
        raise ResultContractError("metrics must contain typed supported definitions including success")
    if "jam" in metrics or "jam_rate" in metrics:
        raise ResultContractError("jam measurement is not implemented")
    for name, definition in metrics.items():
        if definition != METRIC_SPECS[name]:
            raise ResultContractError(f"metric {name} does not match the typed v2 contract")
    artifacts = _artifact_map(manifest)
    common = manifest["common_contracts"]
    _exact(common, COMMON_REF_KEYS, "common_contracts")
    common_kinds = {"source": "source", "selection": "g2_selection", "schedule": "schedule", "evaluator": "evaluator", "observation_contract": "observation_contract", "action_contract": "action_contract", "provider_authority": "provider_authority", "control_contract": "control_contract"}
    for key, kind in common_kinds.items():
        if common[key] not in artifacts or artifacts[common[key]]["kind"] != kind:
            raise ResultContractError(f"common {key} must reference a declared {kind}")
    policies = tuple(_sequence(manifest["policy_instances"], "policy_instances"))
    policy_ids, seeds = set(), {method: [] for method in METHODS}
    pfields = {"policy_instance_id", "method", "training_seed", "policy_artifact_id", "run_manifest_artifact_id", "training_report_artifact_id", "common_contracts"}
    for policy in policies:
        _exact(policy, pfields, "policy instance")
        policy_id, method, seed = policy["policy_instance_id"], policy["method"], policy["training_seed"]
        if not isinstance(policy_id, str) or not policy_id or policy_id in policy_ids or method not in METHODS:
            raise ResultContractError("policy IDs must be unique and methods known")
        policy_ids.add(policy_id)
        if policy["common_contracts"] != common:
            raise ResultContractError("all policies must share sealed common contracts")
        expected_kind = "controller" if method in BASELINE_METHODS else "checkpoint"
        refs = ((policy["policy_artifact_id"], expected_kind), (policy["run_manifest_artifact_id"], "run_manifest"))
        if any(ref not in artifacts or artifacts[ref]["kind"] != kind for ref, kind in refs):
            raise ResultContractError("policy artifact/run_manifest kind mismatch")
        if method in BASELINE_METHODS:
            if seed is not None or policy["training_report_artifact_id"] is not None:
                raise ResultContractError("baseline seed and training report must be null")
        else:
            if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
                raise ResultContractError("learned training seeds must be nonnegative integers")
            report_id = policy["training_report_artifact_id"]
            if report_id not in artifacts or artifacts[report_id]["kind"] != "training_report":
                raise ResultContractError("learned policy requires a training report")
            seeds[method].append(seed)
    for method in BASELINE_METHODS:
        if sum(policy["method"] == method for policy in policies) != 1:
            raise ResultContractError(f"{method} must have one unreplicated policy")
    if any(len(seeds[m]) < 5 or len(set(seeds[m])) != len(seeds[m]) for m in LEARNED_METHODS) or sorted(seeds["ppo"]) != sorted(seeds["sac"]):
        raise ResultContractError("PPO/SAC require the same roster of at least five unique seeds")
    units = tuple(_sequence(manifest["evaluation_units"], "evaluation_units"))
    unit_ids = set()
    ufields = {"unit_id", "condition", "grasp", "eval_seed", "reset_slot", "initial_snapshot_artifact_id", "initial_snapshot_sha256", "stratum", "magnitude", "magnitude_unit", "episode_artifact_ids", "evaluation_report_artifact_ids"}
    for unit in units:
        _exact(unit, ufields, "evaluation unit")
        snapshot = unit["initial_snapshot_artifact_id"]
        if snapshot not in artifacts or artifacts[snapshot]["kind"] != "initial_snapshot" or not _valid_digest(unit["initial_snapshot_sha256"]):
            raise ResultContractError("evaluation unit requires a declared snapshot and digest")
        if isinstance(unit["eval_seed"], bool) or not isinstance(unit["eval_seed"], int) or isinstance(unit["reset_slot"], bool) or not isinstance(unit["reset_slot"], int):
            raise ResultContractError("eval_seed and reset_slot must be integers")
        _number(unit["magnitude"], "magnitude")
        if not isinstance(unit["magnitude_unit"], str) or not unit["magnitude_unit"] or unit["unit_id"] != canonical_evaluation_unit_id(unit) or unit["unit_id"] in unit_ids:
            raise ResultContractError("evaluation unit is noncanonical or duplicated")
        unit_ids.add(unit["unit_id"])
        for field, kind in (("episode_artifact_ids", "episode"), ("evaluation_report_artifact_ids", "evaluation_report")):
            refs = unit[field]
            if not isinstance(refs, Mapping) or set(refs) != policy_ids or any(ref not in artifacts or artifacts[ref]["kind"] != kind for ref in refs.values()):
                raise ResultContractError(f"{field} must cover every policy with {kind} artifacts")
    if not units:
        raise ResultContractError("at least one evaluation unit is required")
    uncertainty = manifest["uncertainty"]
    _exact(uncertainty, {"confidence_level", "iterations", "rng_seed", "cluster_unit"}, "uncertainty")
    confidence = _number(uncertainty["confidence_level"], "confidence_level")
    if not 0 < confidence < 1 or isinstance(uncertainty["iterations"], bool) or not isinstance(uncertainty["iterations"], int) or uncertainty["iterations"] < 1 or isinstance(uncertainty["rng_seed"], bool) or not isinstance(uncertainty["rng_seed"], int) or uncertainty["cluster_unit"] != "training_seed":
        raise ResultContractError("invalid fixed-grid uncertainty contract")
    return ValidatedPreregistration(bytes(data), expected_sha256)


def _resolver_bytes(resolver: Mapping[str, bytes] | Callable[[str], bytes], artifact_id: str) -> bytes:
    try:
        value = resolver[artifact_id] if isinstance(resolver, Mapping) else resolver(artifact_id)
    except (KeyError, OSError) as exc:
        raise ResultContractError(f"artifact bytes unavailable: {artifact_id}") from exc
    if not isinstance(value, (bytes, bytearray, memoryview)):
        raise ResultContractError("artifact resolver values must be bytes-like")
    return bytes(value)


def _load_artifacts(manifest: Mapping[str, Any], resolver: Mapping[str, bytes] | Callable[[str], bytes]) -> dict[str, Mapping[str, Any]]:
    result = {}
    for artifact_id, declaration in _artifact_map(manifest).items():
        data = _resolver_bytes(resolver, artifact_id)
        if _sha(data) != declaration["sha256"]:
            raise ResultContractError(f"artifact SHA-256 mismatch: {artifact_id}")
        payload = _decode_canonical(data, f"artifact {artifact_id}")
        if not isinstance(payload, Mapping) or payload.get("artifact_id") != artifact_id or payload.get("kind") != declaration["kind"]:
            raise ResultContractError(f"artifact identity/kind mismatch: {artifact_id}")
        result[artifact_id] = payload
    return result


def _validate_semantics(manifest: Mapping[str, Any], payloads: Mapping[str, Mapping[str, Any]]) -> None:
    common = manifest["common_contracts"]
    if payloads[common["observation_contract"]].get("dimension") != 79 or payloads[common["action_contract"]].get("right_residual_dimension") != 6:
        raise ResultContractError("observation/action contract semantics mismatch")
    evaluator_sha = next(item["sha256"] for item in manifest["artifacts"] if item["artifact_id"] == common["evaluator"])
    if payloads[common["provider_authority"]].get("evaluator_sha256") != evaluator_sha:
        raise ResultContractError("provider authority does not bind evaluator bytes")
    schedule = payloads[common["schedule"]]
    if not _valid_digest(schedule.get("environment_seed_roster_sha256")):
        raise ResultContractError("schedule must seal environment seeds")
    for policy in manifest["policy_instances"]:
        method, seed = policy["method"], policy["training_seed"]
        if payloads[policy["policy_artifact_id"]].get("method") != method or payloads[policy["policy_artifact_id"]].get("training_seed") != seed:
            raise ResultContractError("policy artifact method/seed mismatch")
        run = payloads[policy["run_manifest_artifact_id"]]
        expected = {"method": method, "training_seed": seed, "policy_artifact_id": policy["policy_artifact_id"], "source_artifact_id": common["source"], "selection_artifact_id": common["selection"], "schedule_artifact_id": common["schedule"], "observation_contract_artifact_id": common["observation_contract"], "action_contract_artifact_id": common["action_contract"], "provider_authority_artifact_id": common["provider_authority"], "control_contract_artifact_id": common["control_contract"], "evaluator_artifact_id": common["evaluator"]}
        if any(run.get(key) != value for key, value in expected.items()):
            raise ResultContractError("run_manifest semantic references mismatch")
    declarations = {item["artifact_id"]: item for item in manifest["artifacts"]}
    for unit in manifest["evaluation_units"]:
        if declarations[unit["initial_snapshot_artifact_id"]]["sha256"] != unit["initial_snapshot_sha256"]:
            raise ResultContractError("snapshot digest is not linked to bytes")


def validate_result_bundle(preregistration: ValidatedPreregistration, bundle: Mapping[str, Any], artifact_resolver: Mapping[str, bytes] | Callable[[str], bytes]) -> ValidatedResultBundle:
    manifest = _assert_preregistration(preregistration)
    _exact(bundle, {"manifest_sha256", "episodes", "training_records", "resource_accounting"}, "bundle")
    if bundle["manifest_sha256"] != preregistration.canonical_sha256:
        raise ResultContractError("bundle manifest SHA-256 mismatch")
    payloads = _load_artifacts(manifest, artifact_resolver)
    _validate_semantics(manifest, payloads)
    policy_by_id = {p["policy_instance_id"]: p for p in manifest["policy_instances"]}
    unit_by_id = {u["unit_id"]: u for u in manifest["evaluation_units"]}
    expected_grid = {(p, u) for p in policy_by_id for u in unit_by_id}
    observed, record_ids = set(), set()
    fields = {"record_id", "row_sha256", "episode_artifact_id", "evaluation_report_artifact_id", "policy_instance_id", "unit_id", "initial_snapshot_sha256", "stratum", "valid", "success", "force_peak_n", "actual_terminal_elapsed_seconds", "success_time_seconds", "time_outcome", "jammed"}
    horizon, declared = float(manifest["frozen_horizon_seconds"]), set(manifest["metrics"])
    for row in _sequence(bundle["episodes"], "episodes"):
        _exact(row, fields, "episode row")
        key = (row["policy_instance_id"], row["unit_id"])
        if key in observed:
            raise ResultContractError("duplicate policy/evaluation-unit row")
        observed.add(key)
        if not isinstance(row["record_id"], str) or not row["record_id"] or row["record_id"] in record_ids:
            raise ResultContractError("result record IDs must be nonempty and unique")
        record_ids.add(row["record_id"])
        if key not in expected_grid:
            raise ResultContractError("episode roster mismatch")
        policy, unit = policy_by_id[key[0]], unit_by_id[key[1]]
        if row["episode_artifact_id"] != unit["episode_artifact_ids"][key[0]] or row["evaluation_report_artifact_id"] != unit["evaluation_report_artifact_ids"][key[0]]:
            raise ResultContractError("row evidence references differ from preregistration")
        digest = canonical_result_row_sha256(row)
        if row["row_sha256"] != digest:
            raise ResultContractError("episode row canonical digest mismatch")
        evidence = {"record_id": row["record_id"], "row_sha256": digest, "policy_instance_id": key[0], "unit_id": key[1], "initial_snapshot_artifact_id": unit["initial_snapshot_artifact_id"], "evaluator_artifact_id": manifest["common_contracts"]["evaluator"], "run_manifest_artifact_id": policy["run_manifest_artifact_id"]}
        for artifact_id in (row["episode_artifact_id"], row["evaluation_report_artifact_id"]):
            if any(payloads[artifact_id].get(field) != value for field, value in evidence.items()):
                raise ResultContractError("episode/report semantics differ from row")
        report_values = {field: row[field] for field in ("valid", "success", "force_peak_n", "actual_terminal_elapsed_seconds", "success_time_seconds", "time_outcome", "jammed")}
        if payloads[row["evaluation_report_artifact_id"]].get("metric_values") != report_values:
            raise ResultContractError("evaluation report metrics differ from row")
        if row["initial_snapshot_sha256"] != unit["initial_snapshot_sha256"] or row["stratum"] != unit["stratum"]:
            raise ResultContractError("row snapshot/stratum differs from preregistration")
        if type(row["valid"]) is not bool or type(row["success"]) is not bool or (not row["valid"] and row["success"]):
            raise ResultContractError("invalid outcomes are failures")
        actual = _number(row["actual_terminal_elapsed_seconds"], "actual elapsed")
        if actual > horizon:
            raise ResultContractError("actual elapsed exceeds horizon")
        outcome = "invalid" if not row["valid"] else "success" if row["success"] else "horizon_timeout" if actual == horizon else "competing_failure"
        if row["time_outcome"] != outcome or (row["success"] and row["success_time_seconds"] != row["actual_terminal_elapsed_seconds"]) or (not row["success"] and row["success_time_seconds"] is not None):
            raise ResultContractError("time outcome/success time mismatch")
        if row["success"]:
            _number(row["success_time_seconds"], "success_time_seconds")
        if row["force_peak_n"] is not None:
            if "force" not in declared:
                raise ResultContractError("undeclared force value")
            _number(row["force_peak_n"], "force_peak_n")
        if row["jammed"] is not None:
            raise ResultContractError("jam is unimplemented and must be null")
    if observed != expected_grid:
        raise ResultContractError(f"incomplete episode grid: missing={len(expected_grid-observed)}")
    learned_ids = {p["policy_instance_id"] for p in policy_by_id.values() if p["method"] in LEARNED_METHODS}
    records = {}
    tfields = {"policy_instance_id", "training_report_artifact_id", "raw_transitions", "valid_transitions", "update_inputs", "replay_warmup", "schedule_artifact_id", "environment_seed_roster_sha256"}
    for row in _sequence(bundle["training_records"], "training_records"):
        _exact(row, tfields, "training record")
        policy_id = row["policy_instance_id"]
        if policy_id in records or policy_id not in learned_ids:
            raise ResultContractError("training roster mismatch or duplicate")
        records[policy_id] = row
        policy = policy_by_id[policy_id]
        if row["training_report_artifact_id"] != policy["training_report_artifact_id"] or row["schedule_artifact_id"] != manifest["common_contracts"]["schedule"]:
            raise ResultContractError("training evidence references mismatch")
        for field in ("raw_transitions", "valid_transitions", "update_inputs", "replay_warmup"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0:
                raise ResultContractError("training counts must be nonnegative integers")
        if row["valid_transitions"] > row["raw_transitions"]:
            raise ResultContractError("valid transitions cannot exceed raw transitions")
        schedule_environment_sha = payloads[manifest["common_contracts"]["schedule"]]["environment_seed_roster_sha256"]
        if row["environment_seed_roster_sha256"] != schedule_environment_sha:
            raise ResultContractError("training environment-seed roster differs from the common schedule")
        report = payloads[row["training_report_artifact_id"]]
        if any(report.get(field) != row[field] for field in tfields - {"training_report_artifact_id"}):
            raise ResultContractError("training report differs from row")
    if set(records) != learned_ids:
        raise ResultContractError("training roster must exactly match learned policies")
    for seed in {policy_by_id[p]["training_seed"] for p in learned_ids}:
        paired = [records[p] for p in learned_ids if policy_by_id[p]["training_seed"] == seed]
        if len({r["raw_transitions"] for r in paired}) != 1 or len({r["schedule_artifact_id"] for r in paired}) != 1 or len({r["environment_seed_roster_sha256"] for r in paired}) != 1:
            raise ResultContractError("paired seeds require equal raw counts and schedule/environment evidence")
    rfields = {"policy_instance_id", "evidence_artifact_id", "wall_seconds", "gpu_allocation_seconds", "measured_gpu_active_seconds", "peak_vram_bytes", "provider_amount", "provider_currency"}
    accounting = {}
    accounting_rows = _sequence(bundle["resource_accounting"], "resource_accounting")
    if "resource_accounting" not in declared and accounting_rows:
        raise ResultContractError("undeclared resource accounting rows are forbidden")
    for row in accounting_rows:
        _exact(row, rfields, "resource accounting")
        policy_id = row["policy_instance_id"]
        if policy_id in accounting or policy_id not in policy_by_id:
            raise ResultContractError("resource roster mismatch or duplicate")
        accounting[policy_id] = row
        _number(row["wall_seconds"], "wall_seconds")
        evidence_id = row["evidence_artifact_id"]
        optional = ("gpu_allocation_seconds", "measured_gpu_active_seconds", "peak_vram_bytes", "provider_amount", "provider_currency")
        if evidence_id is None:
            if any(row[field] is not None for field in optional):
                raise ResultContractError("GPU/provider values require typed evidence")
        else:
            if evidence_id not in payloads or payloads[evidence_id].get("kind") != "resource_evidence" or any(payloads[evidence_id].get(field) != row[field] for field in rfields - {"evidence_artifact_id"}):
                raise ResultContractError("resource evidence differs from row")
            for field in ("gpu_allocation_seconds", "measured_gpu_active_seconds", "peak_vram_bytes", "provider_amount"):
                if row[field] is not None:
                    _number(row[field], field)
            amount, currency = row["provider_amount"], row["provider_currency"]
            if amount is None:
                if currency is not None:
                    raise ResultContractError("provider amount and currency must both be unavailable")
            elif not isinstance(currency, str) or not currency:
                raise ResultContractError("provider amount requires a nonempty string currency")
    if "resource_accounting" in declared and set(accounting) != set(policy_by_id):
        raise ResultContractError("resource roster must exactly match policies")
    bundle_bytes = _canonical_bytes(bundle)
    return ValidatedResultBundle(preregistration, bundle_bytes, _sha(bundle_bytes))


def _interval(values: Sequence[float], confidence: float) -> list[float]:
    ordered = sorted(values)
    def q(probability: float) -> float:
        position = (len(ordered)-1)*probability
        low, high = math.floor(position), math.ceil(position)
        return ordered[low] if low == high else ordered[low]+(ordered[high]-ordered[low])*(position-low)
    tail = (1-confidence)/2
    return [q(tail), q(1-tail)]


def summarize_validated_results(validated: ValidatedResultBundle) -> Mapping[str, Any]:
    manifest, bundle = _assert_bundle(validated)
    policy_by_id = {p["policy_instance_id"]: p for p in manifest["policy_instances"]}
    unit_ids = [u["unit_id"] for u in manifest["evaluation_units"]]
    unit_by_id = {u["unit_id"]: u for u in manifest["evaluation_units"]}
    rows = {(r["policy_instance_id"], r["unit_id"]): r for r in bundle["episodes"]}
    policies = {m: sorted((p for p in policy_by_id.values() if p["method"] == m), key=lambda p: -1 if p["training_seed"] is None else p["training_seed"]) for m in METHODS}
    summary = {"schema_version": "rj45-sealed-summary-v2", "manifest_sha256": validated.preregistration.canonical_sha256, "assurance": {"packaging_validation_complete": True, "canonical_json_row_cross_checks_complete": True, "row_semantics_verified": False, "scientific_validation_complete": False, "not_established": list(ASSURANCE_NOT_ESTABLISHED)}, "evaluation_scope": {"kind": "fixed_grid", "generalizes_to_sampled_population": False}, "uncertainty_scope": {**manifest["uncertainty"], "fixed_grid_baseline_confidence_interval": None, "learned_intervals_are_conditional_on_fixed_grid": True, "automatic_positive_effect_decision": False}, "metrics": {}, "jam": {"status": "not_measured", "reason": "jam detector contract is not implemented"}}
    success = {}
    for method in METHODS:
        per_policy = []
        for policy in policies[method]:
            selected = [rows[(policy["policy_instance_id"], unit)] for unit in unit_ids]
            per_policy.append({"policy_instance_id": policy["policy_instance_id"], "training_seed": policy["training_seed"], "success_rate": sum(r["success"] for r in selected)/len(selected)})
        strata = []
        for key in sorted({_canonical_bytes(unit_by_id[u]["stratum"]) for u in unit_ids}):
            selected_units = [u for u in unit_ids if _canonical_bytes(unit_by_id[u]["stratum"]) == key]
            denominator = len(selected_units)*len(policies[method])
            strata.append({"stratum": json.loads(key), "denominator": denominator, "magnitude_unit": sorted({unit_by_id[u]["magnitude_unit"] for u in selected_units}), "magnitudes": sorted({unit_by_id[u]["magnitude"] for u in selected_units}), "invalid_count": sum(not rows[(p["policy_instance_id"], u)]["valid"] for p in policies[method] for u in selected_units), "success_rate": sum(rows[(p["policy_instance_id"], u)]["success"] for p in policies[method] for u in selected_units)/denominator})
        success[method] = {"denominator": len(unit_ids)*len(policies[method]), "invalid_count": sum(not rows[(p["policy_instance_id"], u)]["valid"] for p in policies[method] for u in unit_ids), "per_policy": per_policy, "method_mean": fmean(x["success_rate"] for x in per_policy), "preregistered_strata": strata}
    uncertainty = manifest["uncertainty"]
    rng = random.Random(uncertainty["rng_seed"])
    seeds = [p["training_seed"] for p in policies["ppo"]]
    draws = [[rng.randrange(len(seeds)) for _ in seeds] for _ in range(uncertainty["iterations"])]
    effects = {}
    for learned in LEARNED_METHODS:
        for baseline in BASELINE_METHODS:
            baseline_id = policies[baseline][0]["policy_instance_id"]
            per_seed = [fmean(float(rows[(p["policy_instance_id"], u)]["success"])-float(rows[(baseline_id, u)]["success"]) for u in unit_ids) for p in policies[learned]]
            effects[f"{learned}_vs_{baseline}"] = {"per_training_seed": per_seed, "mean": fmean(per_seed), "conditional_confidence_interval": _interval([fmean(per_seed[i] for i in draw) for draw in draws], uncertainty["confidence_level"])}
    ppo, sac = ({p["training_seed"]: p["policy_instance_id"] for p in policies[m]} for m in LEARNED_METHODS)
    paired = [fmean(float(rows[(ppo[s], u)]["success"])-float(rows[(sac[s], u)]["success"]) for u in unit_ids) for s in seeds]
    effects["ppo_vs_sac"] = {"per_paired_training_seed": paired, "mean": fmean(paired), "conditional_confidence_interval": _interval([fmean(paired[i] for i in draw) for draw in draws], uncertainty["confidence_level"])}
    summary["metrics"]["success"] = {"methods": success, "paired_effects": effects, "baseline_confidence_intervals": None}
    declared = set(manifest["metrics"])
    if "force" in declared:
        force = {}
        for method in METHODS:
            by_policy, values, success_values = [], [], []
            for policy in policies[method]:
                policy_values = [rows[(policy["policy_instance_id"], u)]["force_peak_n"] for u in unit_ids]
                values.extend(v for v in policy_values if v is not None)
                success_values.extend(rows[(policy["policy_instance_id"], u)]["force_peak_n"] for u in unit_ids if rows[(policy["policy_instance_id"], u)]["success"] and rows[(policy["policy_instance_id"], u)]["force_peak_n"] is not None)
                by_policy.append({"policy_instance_id": policy["policy_instance_id"], "training_seed": policy["training_seed"], "values_n": policy_values})
            denominator = len(unit_ids)*len(policies[method])
            force[method] = {"all_attempt_mean_n": fmean(values) if values else None, "observed": len(values), "missing": denominator-len(values), "all_attempt_force_coverage_complete": len(values) == denominator, "conditional_descriptive_success_only_mean_n": fmean(success_values) if success_values else None, "scientific_safety_decision_allowed": False, "per_training_seed_or_policy": by_policy}
        summary["metrics"]["force"] = force
    if "time" in declared:
        time = {}
        horizon = manifest["frozen_horizon_seconds"]
        for method in METHODS:
            method_rows = [rows[(p["policy_instance_id"], u)] for p in policies[method] for u in unit_ids]
            success_times = [r["success_time_seconds"] for r in method_rows if r["success_time_seconds"] is not None]
            time[method] = {"outcome_records": [{"time_outcome": r["time_outcome"], "actual_terminal_elapsed_seconds": r["actual_terminal_elapsed_seconds"], "success_time_seconds": r["success_time_seconds"]} for r in method_rows], "conditional_descriptive_success_time_mean_seconds": fmean(success_times) if success_times else None, "capped_time_with_horizon_penalty_mean_seconds": fmean(r["success_time_seconds"] if r["success"] else horizon for r in method_rows), "survival_or_km_estimate": None}
        summary["metrics"]["time"] = time
    if "raw_transitions" in declared:
        summary["metrics"]["raw_transitions"] = list(bundle["training_records"])
    if "resource_accounting" in declared:
        summary["metrics"]["resource_accounting"] = [{**r, "evidence_status": "available" if r["evidence_artifact_id"] else "not_available"} for r in bundle["resource_accounting"]]
    return summary
