import copy
import hashlib
import json

import pytest

from rby1_assembly.rj45_sealed_results import (
    METRIC_SPECS,
    METHODS,
    ResultContractError,
    canonical_evaluation_unit_id,
    canonical_result_row_sha256,
    summarize_validated_results,
    validate_preregistration,
    validate_result_bundle,
)


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(encoded(value) if not isinstance(value, bytes) else value).hexdigest()


def fixture(*, metric_names=("success", "force", "time", "raw_transitions", "resource_accounting"), units=3):
    blobs = {}

    def artifact(artifact_id, kind, **values):
        payload = {"artifact_id": artifact_id, "kind": kind, **values}
        blobs[artifact_id] = encoded(payload)
        return artifact_id

    evaluator = artifact("evaluator", "evaluator", version="synthetic-v1")
    common = {
        "source": artifact("source", "source", revision="synthetic"),
        "selection": artifact("selection", "g2_selection", frozen=False),
        "schedule": artifact("schedule", "schedule", environment_seed_roster_sha256="9" * 64),
        "evaluator": evaluator,
        "observation_contract": artifact("observation-contract", "observation_contract", dimension=79),
        "action_contract": artifact("action-contract", "action_contract", right_residual_dimension=6),
        "provider_authority": artifact("provider-authority", "provider_authority", evaluator_sha256=digest(blobs[evaluator])),
        "control_contract": artifact("control-contract", "control_contract", right_residual_dimension=6),
    }
    policies = []
    training = []
    specs = [("zero", None), ("classical", None)] + [(method, seed) for method in ("ppo", "sac") for seed in range(5)]
    for method, seed in specs:
        policy_id = method if seed is None else f"{method}-{seed}"
        policy_artifact = artifact(f"policy-{policy_id}", "controller" if seed is None else "checkpoint", method=method, training_seed=seed, tensor_format_claim="unverified")
        run_id = artifact(
            f"run-{policy_id}", "run_manifest", method=method, training_seed=seed,
            policy_artifact_id=policy_artifact, source_artifact_id=common["source"],
            selection_artifact_id=common["selection"], schedule_artifact_id=common["schedule"],
            observation_contract_artifact_id=common["observation_contract"], action_contract_artifact_id=common["action_contract"],
            provider_authority_artifact_id=common["provider_authority"], control_contract_artifact_id=common["control_contract"],
            evaluator_artifact_id=common["evaluator"],
        )
        report_id = None
        if seed is not None:
            record = {
                "policy_instance_id": policy_id, "raw_transitions": 1000 + seed, "valid_transitions": 900,
                "update_inputs": 800, "replay_warmup": 100, "schedule_artifact_id": common["schedule"],
                "environment_seed_roster_sha256": "9" * 64,
            }
            report_id = artifact(f"training-{policy_id}", "training_report", **record)
            training.append({"training_report_artifact_id": report_id, **record})
        policies.append({
            "policy_instance_id": policy_id, "method": method, "training_seed": seed,
            "policy_artifact_id": policy_artifact, "run_manifest_artifact_id": run_id,
            "training_report_artifact_id": report_id, "common_contracts": copy.deepcopy(common),
        })
    evaluation_units, episodes = [], []
    for unit_index in range(units):
        snapshot_id = artifact(f"snapshot-{unit_index}", "initial_snapshot", state=[unit_index, 0, 1])
        unit = {
            "condition": f"condition-{unit_index % 2}", "grasp": f"grasp-{unit_index}", "eval_seed": 100 + unit_index,
            "reset_slot": unit_index, "initial_snapshot_artifact_id": snapshot_id,
            "initial_snapshot_sha256": digest(blobs[snapshot_id]), "stratum": {"difficulty": "nominal" if unit_index % 2 == 0 else "hard"},
            "magnitude": 0.1 + 0.1 * unit_index, "magnitude_unit": "mm",
        }
        unit["unit_id"] = canonical_evaluation_unit_id(unit)
        episode_refs, report_refs = {}, {}
        for policy_index, policy in enumerate(policies):
            success = (policy_index + unit_index) % 3 != 0
            actual = float(1 + unit_index) if success else 30.0 if unit_index % 2 == 0 else 7.0
            outcome = "success" if success else "horizon_timeout" if actual == 30.0 else "competing_failure"
            policy_id = policy["policy_instance_id"]
            row = {
                "record_id": f"record-{policy_id}-{unit_index}", "policy_instance_id": policy_id, "unit_id": unit["unit_id"],
                "initial_snapshot_sha256": unit["initial_snapshot_sha256"], "stratum": copy.deepcopy(unit["stratum"]),
                "valid": True, "success": success, "force_peak_n": float(3 + policy_index + unit_index) if "force" in metric_names else None,
                "actual_terminal_elapsed_seconds": actual, "success_time_seconds": actual if success else None,
                "time_outcome": outcome, "jammed": None,
            }
            row["row_sha256"] = canonical_result_row_sha256(row)
            evidence = {
                "record_id": row["record_id"], "row_sha256": row["row_sha256"], "policy_instance_id": policy_id,
                "unit_id": unit["unit_id"], "initial_snapshot_artifact_id": snapshot_id,
                "evaluator_artifact_id": common["evaluator"], "run_manifest_artifact_id": policy["run_manifest_artifact_id"],
            }
            episode_id = artifact(f"episode-{policy_id}-{unit_index}", "episode", **evidence)
            metric_values = {key: row[key] for key in ("valid", "success", "force_peak_n", "actual_terminal_elapsed_seconds", "success_time_seconds", "time_outcome", "jammed")}
            report_id = artifact(f"report-{policy_id}-{unit_index}", "evaluation_report", **evidence, metric_values=metric_values)
            row.update(episode_artifact_id=episode_id, evaluation_report_artifact_id=report_id)
            episode_refs[policy_id], report_refs[policy_id] = episode_id, report_id
            episodes.append(row)
        unit.update(episode_artifact_ids=episode_refs, evaluation_report_artifact_ids=report_refs)
        evaluation_units.append(unit)
    accounting = [{
        "policy_instance_id": policy["policy_instance_id"], "evidence_artifact_id": None, "wall_seconds": 120.0,
        "gpu_allocation_seconds": None, "measured_gpu_active_seconds": None, "peak_vram_bytes": None,
        "provider_amount": None, "provider_currency": None,
    } for policy in policies] if "resource_accounting" in metric_names else []
    manifest = {
        "schema_version": "rj45-sealed-results-v2", "method_definitions": list(METHODS),
        "observation_dimension": 79, "right_residual_dimension": 6, "common_contracts": common,
        "policy_instances": policies, "evaluation_scope": {"kind": "fixed_grid", "population_contract_artifact_id": None},
        "evaluation_units": evaluation_units, "frozen_horizon_seconds": 30.0,
        "uncertainty": {"confidence_level": 0.9, "iterations": 30, "rng_seed": 123, "cluster_unit": "training_seed"},
        "metrics": {name: copy.deepcopy(METRIC_SPECS[name]) for name in metric_names},
        "artifacts": [{"artifact_id": artifact_id, "kind": json.loads(data)["kind"], "sha256": digest(data)} for artifact_id, data in blobs.items()],
    }
    bundle = {"manifest_sha256": digest(manifest), "episodes": episodes, "training_records": training, "resource_accounting": accounting}
    return manifest, bundle, blobs


def validated(manifest, bundle, blobs):
    prereg = validate_preregistration(manifest, digest(manifest))
    return validate_result_bundle(prereg, bundle, blobs)


def reseal_row(manifest, bundle, blobs, row):
    row["row_sha256"] = canonical_result_row_sha256(row)
    unit = next(unit for unit in manifest["evaluation_units"] if unit["unit_id"] == row["unit_id"])
    policy = next(policy for policy in manifest["policy_instances"] if policy["policy_instance_id"] == row["policy_instance_id"])
    evidence = {"record_id": row["record_id"], "row_sha256": row["row_sha256"], "policy_instance_id": row["policy_instance_id"], "unit_id": row["unit_id"], "initial_snapshot_artifact_id": unit["initial_snapshot_artifact_id"], "evaluator_artifact_id": manifest["common_contracts"]["evaluator"], "run_manifest_artifact_id": policy["run_manifest_artifact_id"]}
    episode_payload = {"artifact_id": row["episode_artifact_id"], "kind": "episode", **evidence}
    metric_values = {key: row[key] for key in ("valid", "success", "force_peak_n", "actual_terminal_elapsed_seconds", "success_time_seconds", "time_outcome", "jammed")}
    report_payload = {"artifact_id": row["evaluation_report_artifact_id"], "kind": "evaluation_report", **evidence, "metric_values": metric_values}
    for payload in (episode_payload, report_payload):
        blobs[payload["artifact_id"]] = encoded(payload)
        next(item for item in manifest["artifacts"] if item["artifact_id"] == payload["artifact_id"])["sha256"] = digest(blobs[payload["artifact_id"]])
    bundle["manifest_sha256"] = digest(manifest)


def reseal_training_report(manifest, bundle, blobs, row):
    report_id = row["training_report_artifact_id"]
    payload = {"artifact_id": report_id, "kind": "training_report", **{key: value for key, value in row.items() if key != "training_report_artifact_id"}}
    blobs[report_id] = encoded(payload)
    next(item for item in manifest["artifacts"] if item["artifact_id"] == report_id)["sha256"] = digest(blobs[report_id])
    bundle["manifest_sha256"] = digest(manifest)


def add_resource_evidence(manifest, bundle, blobs, row, evidence_id="resource-ppo-0"):
    row["evidence_artifact_id"] = evidence_id
    payload = {"artifact_id": evidence_id, "kind": "resource_evidence", **{key: value for key, value in row.items() if key != "evidence_artifact_id"}}
    blobs[evidence_id] = encoded(payload)
    manifest["artifacts"].append({"artifact_id": evidence_id, "kind": "resource_evidence", "sha256": digest(blobs[evidence_id])})
    bundle["manifest_sha256"] = digest(manifest)


def test_normal_grid_and_byte_identical_summary():
    manifest, bundle, blobs = fixture(units=4)
    result = validated(manifest, bundle, blobs)
    assert len(manifest["policy_instances"]) == 12 and len(result.episodes) == 48
    first, second = summarize_validated_results(result), summarize_validated_results(validated(copy.deepcopy(manifest), copy.deepcopy(bundle), dict(blobs)))
    assert encoded(first) == encoded(second)
    assert first["assurance"]["canonical_json_row_cross_checks_complete"] is True
    assert first["assurance"]["row_semantics_verified"] is False
    assert first["assurance"]["scientific_validation_complete"] is False
    assert first["evaluation_scope"]["generalizes_to_sampled_population"] is False
    assert first["metrics"]["success"]["baseline_confidence_intervals"] is None


def test_preregistration_detaches_original_and_returned_nested_objects():
    manifest, bundle, blobs = fixture()
    prereg = validate_preregistration(manifest, digest(manifest))
    original_sha = prereg.manifest["artifacts"][0]["sha256"]
    manifest["artifacts"][0]["sha256"] = "0" * 64
    exposed = prereg.manifest
    exposed["artifacts"][0]["sha256"] = "1" * 64
    assert prereg.manifest["artifacts"][0]["sha256"] == original_sha
    validate_result_bundle(prereg, bundle, blobs)
    altered = dict(blobs)
    altered[next(iter(altered))] += b"x"
    with pytest.raises(ResultContractError, match="SHA-256"):
        validate_result_bundle(prereg, bundle, altered)


def test_validated_bundle_detaches_original_and_returned_rows():
    manifest, bundle, blobs = fixture()
    result = validated(manifest, bundle, blobs)
    expected = encoded(summarize_validated_results(result))
    bundle["episodes"][0]["success"] = not bundle["episodes"][0]["success"]
    exposed = result.bundle
    exposed["episodes"][0]["force_peak_n"] = 999.0
    assert encoded(summarize_validated_results(result)) == expected
    object.__setattr__(result, "_bundle_bytes", encoded(exposed))
    with pytest.raises(ResultContractError, match="stored result bundle digest"):
        summarize_validated_results(result)


def test_opposite_outcome_with_same_evidence_is_rejected():
    manifest, bundle, blobs = fixture()
    row = bundle["episodes"][0]
    row.update(success=not row["success"], success_time_seconds=None, time_outcome="competing_failure", actual_terminal_elapsed_seconds=7.0)
    row["row_sha256"] = canonical_result_row_sha256(row)
    with pytest.raises(ResultContractError, match="semantics|metrics"):
        validated(manifest, bundle, blobs)


def test_unique_policy_artifacts_and_method_seed_swaps_are_rejected():
    manifest, bundle, blobs = fixture()
    learned = [p for p in manifest["policy_instances"] if p["method"] == "ppo"]
    assert len({p["policy_artifact_id"] for p in learned}) == 5
    learned[0]["policy_artifact_id"], learned[1]["policy_artifact_id"] = learned[1]["policy_artifact_id"], learned[0]["policy_artifact_id"]
    bundle["manifest_sha256"] = digest(manifest)
    with pytest.raises(ResultContractError, match="method/seed"):
        validated(manifest, bundle, blobs)


@pytest.mark.parametrize("mutation", ["baseline_copy", "baseline_seed", "few", "mismatch"])
def test_policy_seed_roster_contract(mutation):
    manifest, _, _ = fixture()
    if mutation == "baseline_copy":
        manifest["policy_instances"].append(copy.deepcopy(manifest["policy_instances"][0]))
        manifest["policy_instances"][-1]["policy_instance_id"] = "zero-copy"
    elif mutation == "baseline_seed":
        manifest["policy_instances"][0]["training_seed"] = 1
    elif mutation == "few":
        manifest["policy_instances"] = [p for p in manifest["policy_instances"] if p["policy_instance_id"] != "ppo-4"]
    else:
        next(p for p in manifest["policy_instances"] if p["policy_instance_id"] == "sac-4")["training_seed"] = 99
    with pytest.raises(ResultContractError):
        validate_preregistration(manifest, digest(manifest))


def test_missing_duplicate_snapshot_and_stratum_mutation_rejected():
    manifest, bundle, blobs = fixture()
    for mutation, match in (("missing", "incomplete"), ("duplicate", "duplicate"), ("snapshot", "digest|snapshot"), ("stratum", "digest|stratum")):
        candidate = copy.deepcopy(bundle)
        if mutation == "missing": candidate["episodes"].pop()
        elif mutation == "duplicate": candidate["episodes"].append(copy.deepcopy(candidate["episodes"][0]))
        elif mutation == "snapshot": candidate["episodes"][0]["initial_snapshot_sha256"] = "0" * 64
        else: candidate["episodes"][0]["stratum"] = {"difficulty": "posthoc"}
        with pytest.raises(ResultContractError, match=match):
            validated(manifest, candidate, blobs)


def test_invalid_stays_in_denominator_with_typed_time_outcome():
    manifest, bundle, blobs = fixture()
    row = bundle["episodes"][0]
    row.update(valid=False, success=False, actual_terminal_elapsed_seconds=4.0, success_time_seconds=None, time_outcome="invalid")
    reseal_row(manifest, bundle, blobs, row)
    summary = summarize_validated_results(validated(manifest, bundle, blobs))
    assert summary["metrics"]["success"]["methods"]["zero"]["denominator"] == 3
    assert summary["metrics"]["success"]["methods"]["zero"]["invalid_count"] == 1


def test_metric_declarations_control_inputs_and_outputs():
    manifest, bundle, blobs = fixture(metric_names=("success",))
    summary = summarize_validated_results(validated(manifest, bundle, blobs))
    assert set(summary["metrics"]) == {"success"}
    assert summary["jam"]["status"] == "not_measured"
    row = bundle["episodes"][0]
    row["force_peak_n"] = 2.0
    reseal_row(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match="undeclared force"):
        validated(manifest, bundle, blobs)


def test_schema_population_scope_and_metric_contract_are_strict():
    manifest, _, _ = fixture()
    for mutate in ("schema", "population", "metric"):
        candidate = copy.deepcopy(manifest)
        if mutate == "schema": candidate["schema_version"] = "v3"
        elif mutate == "population": candidate["evaluation_scope"] = {"kind": "sampled_population", "population_contract_artifact_id": "source"}
        else: candidate["metrics"]["force"]["unit"] = "kg"
        with pytest.raises(ResultContractError):
            validate_preregistration(candidate, digest(candidate))


def test_time_outcomes_preserve_early_failure_and_horizon_timeout():
    manifest, bundle, blobs = fixture()
    summary = summarize_validated_results(validated(manifest, bundle, blobs))
    records = [record for method in summary["metrics"]["time"].values() for record in method["outcome_records"]]
    assert any(r["time_outcome"] == "competing_failure" and r["actual_terminal_elapsed_seconds"] == 7.0 for r in records)
    assert any(r["time_outcome"] == "horizon_timeout" and r["actual_terminal_elapsed_seconds"] == 30.0 for r in records)
    assert all("censored" not in r for r in records)
    assert summary["metrics"]["time"]["zero"]["survival_or_km_estimate"] is None


def test_success_time_rejects_bool_even_when_true_equals_one():
    manifest, bundle, blobs = fixture()
    row = next(r for r in bundle["episodes"] if r["success"] and r["actual_terminal_elapsed_seconds"] == 1.0)
    row["success_time_seconds"] = True
    reseal_row(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match="success_time_seconds"):
        validated(manifest, bundle, blobs)


def test_force_coverage_seed_values_and_safety_decision_remain_explicit():
    manifest, bundle, blobs = fixture()
    row = next(r for r in bundle["episodes"] if r["policy_instance_id"] == "ppo-0")
    row["force_peak_n"] = None
    reseal_row(manifest, bundle, blobs, row)
    force = summarize_validated_results(validated(manifest, bundle, blobs))["metrics"]["force"]["ppo"]
    assert force["missing"] == 1 and force["all_attempt_force_coverage_complete"] is False
    assert force["scientific_safety_decision_allowed"] is False
    assert len(force["per_training_seed_or_policy"]) == 5


def test_raw_counts_equal_but_policy_experience_bytes_may_differ():
    manifest, bundle, blobs = fixture()
    assert blobs["policy-ppo-0"] != blobs["policy-sac-0"]
    validated(manifest, bundle, blobs)
    row = next(r for r in bundle["training_records"] if r["policy_instance_id"] == "sac-0")
    row["raw_transitions"] += 1
    reseal_training_report(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match="equal raw counts"):
        validated(manifest, bundle, blobs)


def test_training_environment_roster_must_match_common_schedule_payload():
    manifest, bundle, blobs = fixture()
    for policy_id in ("ppo-0", "sac-0"):
        row = next(r for r in bundle["training_records"] if r["policy_instance_id"] == policy_id)
        row["environment_seed_roster_sha256"] = "8" * 64
        reseal_training_report(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match="common schedule"):
        validated(manifest, bundle, blobs)


def test_valid_transitions_cannot_exceed_raw_transitions():
    manifest, bundle, blobs = fixture()
    row = next(r for r in bundle["training_records"] if r["policy_instance_id"] == "ppo-0")
    row.update(raw_transitions=1000, valid_transitions=1001)
    reseal_training_report(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match="valid transitions"):
        validated(manifest, bundle, blobs)


def test_gpu_and_provider_values_require_typed_matching_artifact():
    manifest, bundle, blobs = fixture()
    row = next(r for r in bundle["resource_accounting"] if r["policy_instance_id"] == "ppo-0")
    row.update(provider_amount=999.0, provider_currency="USD")
    with pytest.raises(ResultContractError, match="typed evidence"):
        validated(manifest, bundle, blobs)
    evidence_id = "resource-ppo-0"
    row.update(evidence_artifact_id=evidence_id, gpu_allocation_seconds=100.0, measured_gpu_active_seconds=75.0, peak_vram_bytes=1024.0)
    add_resource_evidence(manifest, bundle, blobs, row, evidence_id)
    output = summarize_validated_results(validated(manifest, bundle, blobs))["metrics"]["resource_accounting"]
    assert next(r for r in output if r["policy_instance_id"] == "ppo-0")["evidence_status"] == "available"


@pytest.mark.parametrize(
    ("amount", "currency", "match"),
    ((None, "USD", "both be unavailable"), (True, "USD", "provider_amount"), (1.0, {"code": "USD"}, "nonempty string")),
)
def test_provider_amount_currency_pair_is_typed(amount, currency, match):
    manifest, bundle, blobs = fixture()
    row = next(r for r in bundle["resource_accounting"] if r["policy_instance_id"] == "ppo-0")
    row.update(provider_amount=amount, provider_currency=currency)
    add_resource_evidence(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match=match):
        validated(manifest, bundle, blobs)


def test_jam_values_and_jam_metric_are_rejected_not_coerced_to_zero():
    manifest, bundle, blobs = fixture()
    row = bundle["episodes"][0]
    row["jammed"] = False
    reseal_row(manifest, bundle, blobs, row)
    with pytest.raises(ResultContractError, match="jam"):
        validated(manifest, bundle, blobs)
    manifest, _, _ = fixture()
    manifest["metrics"]["jam_rate"] = {"metric_id": "jam_rate"}
    with pytest.raises(ResultContractError):
        validate_preregistration(manifest, digest(manifest))


def test_snapshot_bytes_and_common_authority_are_semantically_bound():
    manifest, bundle, blobs = fixture()
    snapshot_id = manifest["evaluation_units"][0]["initial_snapshot_artifact_id"]
    altered = dict(blobs)
    altered[snapshot_id] = encoded({"artifact_id": snapshot_id, "kind": "initial_snapshot", "state": [9]})
    with pytest.raises(ResultContractError, match="SHA-256"):
        validated(manifest, bundle, altered)
    authority = json.loads(blobs["provider-authority"])
    authority["evaluator_sha256"] = "0" * 64
    altered = dict(blobs); altered["provider-authority"] = encoded(authority)
    next(item for item in manifest["artifacts"] if item["artifact_id"] == "provider-authority")["sha256"] = digest(altered["provider-authority"])
    bundle["manifest_sha256"] = digest(manifest)
    with pytest.raises(ResultContractError, match="authority"):
        validated(manifest, bundle, altered)
