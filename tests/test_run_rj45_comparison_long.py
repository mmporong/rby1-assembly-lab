from __future__ import annotations

import importlib.util
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_rj45_comparison_long", ROOT / "scripts/run_rj45_comparison_long.py"
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def _draft():
    return json.loads(
        (ROOT / "configs/rj45-comparison-pilot-v1.json").read_text(encoding="utf-8")
    )


def _frozen():
    value = _draft()
    value["frozen"] = True
    value["status"] = "frozen"
    value["seeds"] = [101, 202, 303]
    value["policy_initialization"]["initial_log_std"] = -3.0
    value["minimum_outcome_contract"].update(
        minimum_valid_policy_transitions=12,
        minimum_algorithm_updates=1,
        minimum_actual_update_input_transitions=12,
        minimum_condition_transition_count_per_condition=1,
    )
    value["g1_gate"].update(
        report_path="g1-job/output/report.json",
        report_sha256="4" * 64,
        campaign_path="g1-job/output/campaign-identity.json",
        campaign_sha256="5" * 64,
        finished_path="g1-job/finished.json",
        finished_sha256="6" * 64,
        supervisor_validation_path="g1-job/supervisor-validation.json",
        supervisor_validation_sha256="7" * 64,
    )
    value["budget"].update(
        cycles=1,
        wave_vector_steps=10,
        raw_transition_budget=120,
        raw_transition_budget_upper_bound=120,
        vector_steps=30,
        ppo_rollout_target_valid_transitions=4,
        maximum_ppo_wave_tail_fraction=0.1,
        sac_replay_capacity=256,
        sac_batch_size=8,
        sac_random_warmup_valid_transitions=8,
        sac_learning_starts_valid_transitions=8,
        sac_utd_ratio=1.0,
    )
    value["development_evaluation"].update(
        raw_transition_budget=120,
        vector_steps=30,
        cycles=1,
        wave_vector_steps=10,
    )
    value["inputs"]["development_evidence_root"] = "/frozen/dev-evidence"
    value["inputs"]["runtime_manifest_path"] = "configs/runtime-manifest.json"
    value["inputs"]["runtime_manifest_sha256"] = "1" * 64
    value["inputs"]["physics_input_manifest_path"] = "configs/physics-manifest.json"
    value["inputs"]["physics_input_manifest_sha256"] = "2" * 64
    value["inputs"]["evidence_manifest_path"] = "configs/evidence-manifest.json"
    value["inputs"]["evidence_manifest_sha256"] = "3" * 64
    value["inputs"]["parallel_contact_proof_path"] = "parallel.json"
    runtime_paths = {
        "configs/runtime.json",
        value["inputs"]["task_config_path"],
        "parallel.json",
        "src/rby1_assembly/rj45_task.py",
        "src/rby1_assembly/rj45_env.py",
        "src/rby1_assembly/rj45_training.py",
        "src/rby1_assembly/rj45_training_env.py",
        "src/rby1_assembly/parallel_contact_evidence.py",
        "src/rby1_assembly/experiment_contracts.py",
        "src/rby1_assembly/held_grasp.py",
        "src/rby1_assembly/isaac_env.py",
    }
    evidence_paths = {
        value["inputs"]["task_config_path"],
        value["inputs"]["training_config_path"],
        value["inputs"]["held_bank_path"],
    }
    current_paths = {
        value["g1_gate"]["report_path"],
        value["g1_gate"]["campaign_path"],
        value["g1_gate"]["finished_path"],
        value["g1_gate"]["supervisor_validation_path"],
        value["inputs"]["development_config_path"],
        value["inputs"]["runtime_manifest_path"],
        value["inputs"]["physics_input_manifest_path"],
        "scripts/run_rj45_comparison_long.py",
        *runner.G1_CPU_SOURCE_PATHS,
        "src/rby1_assembly/rj45_comparison_rl.py",
        "src/rby1_assembly/rj45_comparison_env.py",
        "src/rby1_assembly/rj45_comparison_pilot_env.py",
    }
    value["source_and_config_sha256"] = {
        "runtime": {path: "a" * 64 for path in runtime_paths},
        "evidence": {path: "b" * 64 for path in evidence_paths},
        "development_evidence": {path: "d" * 64 for path in evidence_paths},
        "current": {path: "c" * 64 for path in current_paths},
    }
    return value


def test_draft_config_is_intentionally_rejected_before_native_work() -> None:
    draft = _draft()
    budget = draft["budget"]
    assert budget["raw_transition_budget_candidate"] == (
        budget["num_envs"]
        * budget["wave_count"]
        * budget["cycles_candidate"]
        * budget["wave_vector_steps_candidate"]
    ) == 196608
    assert budget["ppo_rollout_target_valid_transitions_candidate"] == 128
    assert budget["maximum_ppo_wave_tail_fraction_candidate"] == 0.03125
    with pytest.raises(runner.LongDriverBlocked, match="not frozen"):
        runner.validate_contract(draft, execution_mode="train")


def test_final_split_and_resume_surface_are_rejected() -> None:
    with pytest.raises(runner.LongDriverBlocked, match="execution mode"):
        runner.validate_contract(_frozen(), execution_mode="evaluate-final")
    with pytest.raises(SystemExit):
        runner.parse_args(
            [
                "--contract",
                "contract.json",
                "--contract-sha256",
                "a" * 64,
                "--execution-mode",
                "train",
                "--output-dir",
                "out",
                "--seed",
                "7",
            ]
        )


def test_g1_gate_target_and_report_claims_cannot_be_weakened(tmp_path: Path) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    assert runner.validate_g1_gate(
        contract, report, current_root=tmp_path, development=development
    )["non_smoke"] is True

    blocked_contract = json.loads(json.dumps(contract))
    blocked_contract["g1_gate"]["required_status"] = "blocked"
    blocked_report = {**report, "status": "blocked", "decision": {"state": "blocked"}}
    mutations = (
        (blocked_contract, blocked_report),
        (
            {**contract, "g1_gate": {**contract["g1_gate"], "required_smoke_only": True}},
            report,
        ),
        (
            {**contract, "g1_gate": {**contract["g1_gate"], "required_claims": {}}},
            report,
        ),
        (
            {
                **contract,
                "g1_gate": {
                    **contract["g1_gate"],
                    "required_claims": {
                        "task_admission_completed": False,
                        "rl_pilot_admitted": True,
                    },
                },
            },
            report,
        ),
        (
            contract,
            {
                **report,
                "claims": {
                    **report["claims"],
                    "rl_pilot_admitted": False,
                },
            },
        ),
        (
            contract,
            {
                **report,
                "classical_seal": {
                    **report["classical_seal"],
                    "canonical_controller_config_sha256": "0" * 64,
                },
            },
        ),
        (
            contract,
            {
                **report,
                "claims": {
                    **report["claims"],
                    "physical_initial_pose_error_sweep": True,
                },
            },
        ),
        (
            contract,
            {
                **report,
                "classical_seal": {
                    key: value
                    for key, value in report["classical_seal"].items()
                    if key != "canonical_controller_config_sha256"
                },
            },
        ),
        (
            contract,
            {
                **report,
                "classical_seal": {
                    **report["classical_seal"],
                    "selection": {
                        **report["tuning_selection"],
                        "success_count": 3,
                    },
                },
            },
        ),
    )
    for mutated_contract, mutated_report in mutations:
        with pytest.raises(runner.LongDriverBlocked):
            runner.validate_g1_gate(
                mutated_contract,
                mutated_report,
                current_root=tmp_path,
                development=development,
            )


def _isolated_formal_gate_fixture(tmp_path: Path):
    contract = _frozen()
    development = json.loads(
        (ROOT / "configs/rj45-comparison-development.json").read_text(encoding="utf-8")
    )
    development_path = tmp_path / contract["inputs"]["development_config_path"]
    development_path.parent.mkdir(parents=True, exist_ok=True)
    development_path.write_bytes(
        (ROOT / contract["inputs"]["development_config_path"]).read_bytes()
    )
    plan = runner._expected_g1_formal_plan(development)
    for relative in runner.G1_CPU_SOURCE_PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        contract["source_and_config_sha256"]["current"][relative] = runner.file_sha256(
            destination
        )
    campaign = {
        "schema_name": "rj45-task-admission-isolated-campaign-v1",
        "purpose": "formal_campaign",
        "environment_seed": 20261121,
        "execution_plan": plan,
        "formal_partitions": [
            list(range(start, end + 1))
            for start, end in (
                (1, 6), (7, 12), (13, 18), (19, 30), (31, 42),
                (43, 54), (55, 66), (67, 78), (79, 90),
            )
        ],
        "config": {
            "path": "/producer/configs/rj45-comparison-development.json",
            "sha256": runner.file_sha256(development_path),
        },
        "physics_input_start": {
            "normalized_sha256": {
                "current": {
                    runner.G1_CPU_SOURCE_PATHS[0]: contract["source_and_config_sha256"]["current"][runner.G1_CPU_SOURCE_PATHS[0]],
                    runner.G1_CPU_SOURCE_PATHS[2]: contract["source_and_config_sha256"]["current"][runner.G1_CPU_SOURCE_PATHS[2]],
                }
            }
        },
        "source_sha256": {
            "runner": contract["source_and_config_sha256"]["current"][runner.G1_CPU_SOURCE_PATHS[0]],
            "coordinator": contract["source_and_config_sha256"]["current"][runner.G1_CPU_SOURCE_PATHS[1]],
        },
    }
    job_root = tmp_path / "job"
    output = job_root / "output"
    output.mkdir(parents=True)
    campaign_path = output / "campaign-identity.json"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    campaign_sha = runner.file_sha256(campaign_path)
    contract["g1_gate"].update(
        campaign_path=str(campaign_path.relative_to(tmp_path)),
        campaign_sha256=campaign_sha,
        report_path="job/output/report.json",
        finished_path="job/finished.json",
        supervisor_validation_path="job/supervisor-validation.json",
    )
    g1_runner = runner._load_module_path(
        tmp_path / runner.G1_CPU_SOURCE_PATHS[0], "synthetic_g1_runner"
    )
    coordinator = runner._load_module_path(
        tmp_path / runner.G1_CPU_SOURCE_PATHS[1], "synthetic_g1_coordinator"
    )
    admission = runner._load_module_path(
        tmp_path / runner.G1_CPU_SOURCE_PATHS[2], "synthetic_g1_protocol"
    )
    candidates = {
        row["id"]: row for row in development["classical_tuning"]["candidates"]
    }
    held = {row["grasp_id"]: row for row in development["held_grasps"]}
    biases = {row["case_id"]: row for row in development["bias_cases"]}
    baseline = admission.make_baseline_artifact_seal(
        source_trace_gzip_sha256=development["methods"][2]["source_trace_gzip_sha256"],
        mean_constant_action=development["methods"][1]["normalized_right6"],
        playback_selected_rows=development["methods"][2]["selected_rows"],
    )

    def snapshot(row):
        bias = biases[row["case_id"]]
        grasp = held.get(row["grasp_id"])
        return {
            "applied_held_grasp_id": row["grasp_id"],
            "applied_held_grasp_index": 2 if grasp is None else grasp["verified_index"],
            "applied_held_artifact_sha256": "8" * 64 if grasp is None else grasp["artifact_sha256"],
            "requested_translation_bias_entry_m": [list(bias["translation_bias_entry_m"])],
            "applied_translation_bias_entry_m": [list(bias["translation_bias_entry_m"])],
            "requested_rotation_bias_entry_rad": [list(bias["rotation_vector_bias_entry_rad"])],
            "applied_rotation_bias_entry_rad": [list(bias["rotation_vector_bias_entry_rad"])],
            "prefreeze_translation_socket_entry_m_xyz": [[0.0, 0.0, 0.0]],
            "prefreeze_shortest_rotation_vector_socket_entry_rad_xyz": [[0.0, 0.0, 0.0]],
            "insert_transition_translation_socket_entry_m_xyz": [[0.0, 0.0, 0.0]],
            "insert_transition_shortest_rotation_vector_socket_entry_rad_xyz": [[0.0, 0.0, 0.0]],
            "prefreeze_snapshot_valid": [True],
            "insert_transition_snapshot_valid": [True],
            "transition_contract_checked": [True],
            "actual_applied_action_is_exact_zero": [True],
        }

    def observations(case_id):
        value = 0.2 if case_id.endswith("positive") else -0.2
        sample = [0.0] * 79
        sample[7] = value
        sample[37] = 0.1
        return [list(sample) for _ in range(13)]

    episodes = []
    for row in plan:
        episode = {key: value for key, value in row.items() if key != "ordinal"}
        method = row["method_id"]
        if row["stage"] == "tuning":
            candidate = candidates[row["candidate_id"]]
            payload = g1_runner._classical_payload(
                SimpleNamespace(
                    candidate_id=row["candidate_id"],
                    position_gain=candidate["position_gain"],
                    rotation_gain=candidate["rotation_gain"],
                    force_trigger_n=candidate["force_trigger_n"],
                    search_step_m=candidate["search_step_m"],
                )
            )
            success_limit = {"A": 4, "B": 3, "C": 2}[row["candidate_id"]]
            success = development["bias_cases"].index(biases[row["case_id"]]) < success_limit
            artifact_sha = None
            controller_sha = runner.canonical_sha256(payload)
            actual_payload = payload
        else:
            success = method == "oracle_gt_reference" or (
                method == "zero" and row["case_id"] == "tz-positive" and row["slot"] == 1
            )
            episode["candidate_id"] = "A" if method == "classical_force_search" else None
            controller_sha = (
                runner._expected_classical_controller_sha256(development, "A")
                if method == "classical_force_search"
                else None
            )
            artifact_sha = (
                baseline.mean_constant_artifact_sha256
                if method == "mean_constant"
                else baseline.playback_artifact_sha256
                if method == "open_loop_playback"
                else None
            )
            actual_payload = None
        episode.update(
            success=success,
            hard_contact=False,
            forbidden_contact=False,
            evaluator_force_p95_n=5.0,
            completion_time_s=3.0,
            provider_inputs=["observation79", "provider_owned_state"],
            applied_controller_sha256=controller_sha,
            applied_artifact_sha256=artifact_sha,
            observation_samples=(
                observations(row["case_id"])
                if row["stage"] == "admission" and method == "zero"
                else []
            ),
            admission_snapshot=snapshot(row),
            terminal_complete=True,
            execution_error=None,
            actual_classical_controller_payload=actual_payload,
        )
        episodes.append(episode)
    protocol, seal = coordinator._seal_tuning(
        episodes[:18], development, g1_runner, admission
    )
    protocol, recomputed_baseline, evidence = coordinator._adjudicate_exact90(
        episodes,
        {"config": development, "playback_actions": [None] * 364},
        g1_runner,
        admission,
        protocol,
    )
    assert protocol.decision.state == "admit_rl_pilot"
    report = {
        "schema_name": "rj45-task-admission-isolated-report-v1",
        "status": "admit_rl_pilot",
        "campaign_identity": {
            "path": "/producer/output/formal-campaign.json",
            "sha256": campaign_sha,
        },
        "tuning_selection": asdict(protocol.selection),
        "classical_seal": seal["classical_seal"],
        "baseline_artifact_seal": asdict(recomputed_baseline),
        "contract_evidence": asdict(evidence),
        "decision": asdict(protocol.decision),
        "episodes": episodes,
        "claims": {
            "task_admission_completed": True,
            "rl_pilot_admitted": True,
            "native_crash_cause_fixed": False,
        },
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    report_sha = runner.file_sha256(report_path)
    supervisor = {
        "execution_contract_passed": True,
        "problems": [],
        "proof": {
            "formal_report_sha256": report_sha,
            "campaign_sha256": campaign_sha,
            "episode_count": 90,
            "decision": "admit_rl_pilot",
            "rl_pilot_admitted": True,
        },
    }
    supervisor_path = job_root / "supervisor-validation.json"
    supervisor_path.write_text(json.dumps(supervisor), encoding="utf-8")
    supervisor_sha = runner.file_sha256(supervisor_path)
    finished = {
        "exit_code": 0,
        "child_exit_code": 0,
        "external_verifier_exit_code": 0,
        "execution_contract_passed": True,
        "task_admission_decision": "admit_rl_pilot",
        "rl_pilot_admitted": True,
        "supervisor_validation_sha256": supervisor_sha,
        "formal_report_sha256": report_sha,
        "campaign_sha256": campaign_sha,
    }
    finished_path = job_root / "finished.json"
    finished_path.write_text(json.dumps(finished), encoding="utf-8")
    contract["g1_gate"].update(
        report_sha256=report_sha,
        finished_sha256=runner.file_sha256(finished_path),
        supervisor_validation_sha256=supervisor_sha,
    )
    for path in (campaign_path, report_path, finished_path, supervisor_path):
        contract["source_and_config_sha256"]["current"][
            str(path.relative_to(tmp_path))
        ] = runner.file_sha256(path)
    return contract, development, campaign_path, campaign, report


def _reseal_report(
    tmp_path: Path, contract: dict, report: dict, *, supervisor: dict | None = None
) -> None:
    report_path = tmp_path / contract["g1_gate"]["report_path"]
    report_path.write_text(json.dumps(report, allow_nan=True), encoding="utf-8")
    report_sha = runner.file_sha256(report_path)
    contract["g1_gate"]["report_sha256"] = report_sha
    contract["source_and_config_sha256"]["current"][
        contract["g1_gate"]["report_path"]
    ] = report_sha
    supervisor_path = tmp_path / contract["g1_gate"]["supervisor_validation_path"]
    value = (
        json.loads(supervisor_path.read_text(encoding="utf-8"))
        if supervisor is None
        else supervisor
    )
    if supervisor is None and isinstance(value.get("proof"), dict):
        value["proof"]["formal_report_sha256"] = report_sha
    supervisor_path.write_text(json.dumps(value), encoding="utf-8")
    supervisor_sha = runner.file_sha256(supervisor_path)
    contract["g1_gate"]["supervisor_validation_sha256"] = supervisor_sha
    contract["source_and_config_sha256"]["current"][
        contract["g1_gate"]["supervisor_validation_path"]
    ] = supervisor_sha
    finished_path = tmp_path / contract["g1_gate"]["finished_path"]
    finished = json.loads(finished_path.read_text(encoding="utf-8"))
    finished.update(
        supervisor_validation_sha256=supervisor_sha,
        formal_report_sha256=report_sha,
        campaign_sha256=contract["g1_gate"]["campaign_sha256"],
    )
    _reseal_finished(tmp_path, contract, finished)


def _reseal_finished(tmp_path: Path, contract: dict, finished: dict) -> None:
    path = tmp_path / contract["g1_gate"]["finished_path"]
    path.write_text(json.dumps(finished), encoding="utf-8")
    digest = runner.file_sha256(path)
    contract["g1_gate"]["finished_sha256"] = digest
    contract["source_and_config_sha256"]["current"][
        contract["g1_gate"]["finished_path"]
    ] = digest


def test_isolated_formal90_report_derives_authenticated_non_smoke_scope(
    tmp_path: Path,
) -> None:
    """Exercise a synthetic positive fixture; this is not an actual G1 admit claim."""

    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    normalized = runner.validate_g1_gate(
        contract, report, current_root=tmp_path, development=development
    )
    assert normalized["non_smoke"] is True
    assert normalized["campaign"]["purpose"] == "formal_campaign"
    assert normalized["campaign"]["episode_count"] == 90
    assert normalized["campaign"]["tuning_episode_count"] == 18
    assert normalized["campaign"]["admission_episode_count"] == 72
    assert normalized["source_report_schema"] == (
        "rj45-task-admission-isolated-report-v1"
    )
    assert normalized["producer_completion"][
        "required_future_producer_finished_fields"
    ] == [
        "supervisor_validation_sha256",
        "formal_report_sha256",
        "campaign_sha256",
    ]


def test_isolated_diagnostic_and_partial_reports_cannot_open_g1_gate(
    tmp_path: Path,
) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    diagnostic = {
        "schema_name": "rj45-task-admission-isolated-diagnostic-v1",
        "status": "admit_rl_pilot",
        "purpose": "diagnostic_nonadjudicative",
        "ordinal": 33,
        "episodes": [report["episodes"][32]],
        "claims": {"task_admission_completed": False, "rl_pilot_admitted": False},
    }
    with pytest.raises(runner.LongDriverBlocked, match="unsupported"):
        runner.validate_g1_gate(
            contract, diagnostic, current_root=tmp_path, development=development
        )
    errata = {**report, "schema_name": "rj45-task-admission-isolated-errata-v1"}
    with pytest.raises(runner.LongDriverBlocked, match="errata"):
        runner.validate_g1_gate(
            contract, errata, current_root=tmp_path, development=development
        )
    partial = {**report, "episodes": report["episodes"][:32]}
    with pytest.raises(runner.LongDriverBlocked, match="authenticated artifact"):
        runner.validate_g1_gate(
            contract, partial, current_root=tmp_path, development=development
        )


def test_isolated_campaign_hash_and_formal_purpose_are_fail_closed(
    tmp_path: Path,
) -> None:
    contract, development, path, campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    path.write_text(json.dumps({**campaign, "mutated": True}), encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="SHA-256 mismatch"):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )
    campaign.pop("purpose")
    path.write_text(json.dumps(campaign), encoding="utf-8")
    mutated_sha = runner.file_sha256(path)
    contract["g1_gate"]["campaign_sha256"] = mutated_sha
    report["campaign_identity"]["sha256"] = mutated_sha
    with pytest.raises(runner.LongDriverBlocked):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "identity_only",
        "89_rows",
        "91_rows",
        "terminal_false",
        "execution_error",
        "nonfinite",
        "boolean_as_integer",
        "snapshot_bias",
        "zero_action_false",
        "provider_authority",
        "controller_seal",
        "artifact_seal",
    ),
)
def test_raw90_cpu_recomputation_rejects_identity_and_contract_tamper(
    tmp_path: Path, mutation: str
) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    rows = report["episodes"]
    if mutation == "identity_only":
        report["episodes"] = [
            {
                key: row[key]
                for key in ("stage", "grasp_id", "case_id", "method_id", "slot", "candidate_id")
            }
            for row in rows
        ]
    elif mutation == "89_rows":
        report["episodes"] = rows[:-1]
    elif mutation == "91_rows":
        report["episodes"] = [*rows, dict(rows[-1])]
    elif mutation == "terminal_false":
        rows[18]["terminal_complete"] = False
    elif mutation == "execution_error":
        rows[18]["execution_error"] = {"type": "RuntimeError"}
    elif mutation == "nonfinite":
        rows[18]["completion_time_s"] = float("nan")
    elif mutation == "boolean_as_integer":
        rows[18]["hard_contact"] = 0
    elif mutation == "snapshot_bias":
        rows[18]["admission_snapshot"]["applied_translation_bias_entry_m"][0][0] += 0.01
    elif mutation == "zero_action_false":
        rows[18]["admission_snapshot"]["actual_applied_action_is_exact_zero"] = [False]
    elif mutation == "provider_authority":
        rows[18]["provider_inputs"] = ["known_calibration_bias"]
    elif mutation == "controller_seal":
        classical = next(row for row in rows[18:] if row["method_id"] == "classical_force_search")
        classical["applied_controller_sha256"] = "0" * 64
    elif mutation == "artifact_seal":
        mean = next(row for row in rows[18:] if row["method_id"] == "mean_constant")
        mean["applied_artifact_sha256"] = "0" * 64
    _reseal_report(tmp_path, contract, report)
    with pytest.raises(runner.LongDriverBlocked):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )


@pytest.mark.parametrize(
    "field,value",
    (
        ("exit_code", 1),
        ("child_exit_code", 1),
        ("external_verifier_exit_code", 1),
        ("execution_contract_passed", 1),
        ("task_admission_decision", "blocked"),
        ("rl_pilot_admitted", False),
    ),
)
def test_finished_record_requires_typed_success_proof(
    tmp_path: Path, field: str, value: object
) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    path = tmp_path / contract["g1_gate"]["finished_path"]
    finished = json.loads(path.read_text(encoding="utf-8"))
    finished[field] = value
    _reseal_finished(tmp_path, contract, finished)
    with pytest.raises(runner.LongDriverBlocked):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )


@pytest.mark.parametrize(
    "field,drop",
    (
        ("supervisor_validation_sha256", True),
        ("supervisor_validation_sha256", False),
        ("formal_report_sha256", True),
        ("formal_report_sha256", False),
        ("campaign_sha256", True),
        ("campaign_sha256", False),
    ),
)
def test_finished_record_closes_every_artifact_sha_edge(
    tmp_path: Path, field: str, drop: bool
) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    path = tmp_path / contract["g1_gate"]["finished_path"]
    finished = json.loads(path.read_text(encoding="utf-8"))
    if drop:
        finished.pop(field)
    else:
        finished[field] = "0" * 64
    _reseal_finished(tmp_path, contract, finished)
    with pytest.raises(runner.LongDriverBlocked, match="proof DAG"):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "execution_false",
        "problems",
        "proof_missing",
        "report_hash",
        "campaign_hash",
        "row_count",
        "decision",
        "rl_bool_as_int",
    ),
)
def test_supervisor_validation_requires_exact_recomputed_proof(
    tmp_path: Path, mutation: str
) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    path = tmp_path / contract["g1_gate"]["supervisor_validation_path"]
    supervisor = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "execution_false":
        supervisor["execution_contract_passed"] = False
    elif mutation == "problems":
        supervisor["problems"] = [{"stage": "validation"}]
    elif mutation == "proof_missing":
        supervisor["proof"] = None
    elif mutation == "report_hash":
        supervisor["proof"]["formal_report_sha256"] = "0" * 64
    elif mutation == "campaign_hash":
        supervisor["proof"]["campaign_sha256"] = "0" * 64
    elif mutation == "row_count":
        supervisor["proof"]["episode_count"] = 89
    elif mutation == "decision":
        supervisor["proof"]["decision"] = "blocked"
    elif mutation == "rl_bool_as_int":
        supervisor["proof"]["rl_pilot_admitted"] = 1
    _reseal_report(tmp_path, contract, report, supervisor=supervisor)
    with pytest.raises(runner.LongDriverBlocked):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )


def test_producer_abort_path_is_derived_and_must_remain_absent(tmp_path: Path) -> None:
    contract, development, _path, _campaign, report = _isolated_formal_gate_fixture(
        tmp_path
    )
    abort = tmp_path / "job/output/campaign-aborted.json"
    abort.write_text("{}", encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="abort artifact"):
        runner.validate_g1_gate(
            contract, report, current_root=tmp_path, development=development
        )


def test_fresh_isolated_python_recomputes_positive_gate_without_native_imports(
    tmp_path: Path,
) -> None:
    """Synthetic CPU evidence only; this does not claim an actual G1 admission."""

    contract, _development, _path, _campaign, _report = (
        _isolated_formal_gate_fixture(tmp_path)
    )
    contract_path = tmp_path / "synthetic-contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    script_path = ROOT / "scripts/run_rj45_comparison_long.py"
    child = f"""
import importlib.util
import json
from pathlib import Path
import sys

forbidden = ("torch", "isaac", "omni")
before = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
spec = importlib.util.spec_from_file_location("fresh_g2_gate", {str(script_path)!r})
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
root = Path({str(tmp_path)!r})
contract = module.load_json(root / "synthetic-contract.json", "synthetic contract")
development = module.load_json(
    root / contract["inputs"]["development_config_path"], "development"
)
report = module.load_json(root / contract["g1_gate"]["report_path"], "G1 report")
normalized = module.validate_g1_gate(
    contract, report, current_root=root, development=development
)
after = sorted(name for name in sys.modules if name.split(".")[0] in forbidden)
print(json.dumps({{"before": before, "after": after, "status": normalized["status"]}}))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", child],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    evidence = json.loads(result.stdout)
    assert evidence == {
        "before": [],
        "after": [],
        "status": "admit_rl_pilot",
    }


def test_budget_requires_exact_num_env_product_and_reviewed_contact_count() -> None:
    runner.validate_contract(_frozen(), execution_mode="train")
    wrong = _frozen()
    wrong["budget"]["raw_transition_budget"] = 121
    with pytest.raises(runner.LongDriverBlocked, match="4 env"):
        runner.validate_contract(wrong, execution_mode="train")
    runner.validate_parallel_contact_proof({"validated_probe_num_envs": 4}, 4)
    with pytest.raises(runner.LongDriverBlocked, match="exact num_envs"):
        runner.validate_parallel_contact_proof({"validated_probe_num_envs": 12}, 4)


@pytest.mark.parametrize(
    "relative",
    (
        "g1-job/output/report.json",
        "g1-job/output/campaign-identity.json",
        "g1-job/finished.json",
        "g1-job/supervisor-validation.json",
        *runner.G1_CPU_SOURCE_PATHS,
    ),
)
def test_contract_requires_all_g1_artifact_and_cpu_source_bindings(
    relative: str,
) -> None:
    contract = _frozen()
    contract["source_and_config_sha256"]["current"].pop(relative)
    with pytest.raises(runner.LongDriverBlocked, match="current source bindings"):
        runner.validate_contract(contract, execution_mode="train")


def test_paired_schedule_has_identical_conditions_and_environment_seeds() -> None:
    schedule = runner.build_paired_schedule(_frozen())
    assert len(schedule["conditions"]) == 12
    assert len(schedule["development_conditions"]) == 12
    assert len(schedule["runs"]) == 6
    for seed in (101, 202, 303):
        pair = [row for row in schedule["runs"] if row["seed"] == seed]
        assert [row["algorithm"] for row in pair] == ["ppo", "sac"]
        assert pair[0]["condition_order"] == pair[1]["condition_order"]
        assert pair[0]["environment_seed"] == pair[1]["environment_seed"]
        assert pair[0]["development_environment_seed"] == pair[1][
            "development_environment_seed"
        ]
        assert pair[0]["environment_seed"] != pair[0]["development_environment_seed"]
        assert pair[0]["schedule_seed"] == pair[1]["schedule_seed"]
        assert pair[0]["model_seed"] != pair[1]["model_seed"]


def test_reviewed_fixed_condition_values_cannot_be_rewritten() -> None:
    contract = _frozen()
    contract["bias_cases"][0]["translation_bias_entry_m"][0] = 0.00013
    with pytest.raises(runner.LongDriverBlocked, match="bias array"):
        runner.build_fixed_conditions(contract)


def test_g1_development_config_and_dev_split_are_distinct_frozen_inputs() -> None:
    contract = _frozen()
    development = json.loads(
        (ROOT / "configs/rj45-comparison-development.json").read_text(encoding="utf-8")
    )
    runner.validate_development_config_binding(contract, development)
    train_ids = {
        row["grasp_id"] for row in runner.build_fixed_conditions(contract, split="train")
    }
    dev_conditions = runner.build_fixed_conditions(contract, split="dev")
    assert train_ids.isdisjoint({row["grasp_id"] for row in dev_conditions})
    assert {row["grasp_id"] for row in dev_conditions} == {
        "rj45-dev-01",
        "rj45-dev-02",
    }
    assert contract["development_evaluation"]["allow_final"] is False


def test_input_start_end_gate_detects_hash_tamper(tmp_path: Path) -> None:
    roots = {
        name: tmp_path / name
        for name in ("runtime", "evidence", "development_evidence", "current")
    }
    for name, root in roots.items():
        root.mkdir()
    contract = _frozen()
    contract["inputs"]["runtime_root"] = str(roots["runtime"])
    contract["inputs"]["evidence_root"] = str(roots["evidence"])
    contract["inputs"]["development_evidence_root"] = str(
        roots["development_evidence"]
    )
    for namespace, declared in contract["source_and_config_sha256"].items():
        root = roots[namespace]
        for relative in declared:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{namespace}:{relative}", encoding="utf-8")
            declared[relative] = runner.file_sha256(path)
    for path_field, sha_field, namespace in (
        ("runtime_manifest_path", "runtime_manifest_sha256", "current"),
        ("physics_input_manifest_path", "physics_input_manifest_sha256", "current"),
        ("evidence_manifest_path", "evidence_manifest_sha256", "evidence"),
    ):
        path = roots[namespace] / contract["inputs"][path_field]
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(f"manifest:{path_field}", encoding="utf-8")
        if path_field == "physics_input_manifest_path":
            base = roots["current"] / "configs/base-physics.json"
            base.write_text("{}", encoding="utf-8")
            path.write_text(
                json.dumps(
                    {
                        "base_physics_manifest": {
                            "relative_path": "configs/base-physics.json",
                            "sha256": runner.file_sha256(base),
                        }
                    }
                ),
                encoding="utf-8",
            )
            contract["source_and_config_sha256"]["current"][
                contract["inputs"][path_field]
            ] = runner.file_sha256(path)
        contract["inputs"][sha_field] = runner.file_sha256(path)
    snapshot = runner.capture_input_snapshot(contract, current_root=roots["current"])
    assert runner.verify_input_snapshot(snapshot)["matched"] is True
    abort_path = Path(snapshot["must_remain_absent"][0])
    abort_path.parent.mkdir(parents=True, exist_ok=True)
    abort_path.write_text("{}", encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="G1_campaign_abort"):
        runner.verify_input_snapshot(snapshot)
    abort_path.unlink()
    tampered = roots["evidence"] / contract["inputs"]["training_config_path"]
    tampered.write_text("tampered", encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="changed after start"):
        runner.verify_input_snapshot(snapshot)


def test_full_runtime_physics_import_and_evidence_source_closure(tmp_path: Path) -> None:
    runtime = tmp_path / "runtime"
    evidence = tmp_path / "evidence"
    current = tmp_path / "current"
    for root in (runtime, evidence, current):
        root.mkdir()

    def write(root: Path, relative: str, content: str) -> str:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return runner.file_sha256(path)

    python_paths = [f"src/rby1_assembly/module_{index:02d}.py" for index in range(35)]
    runtime_python = {
        path: write(runtime, path, f"runtime-python-{index}")
        for index, path in enumerate(python_paths)
    }
    evidence_python = {
        path: write(evidence, path, f"evidence-python-{index}")
        for index, path in enumerate(python_paths)
    }
    physics_paths = [f"physics/input_{index:02d}.bin" for index in range(13)]
    physics = {}
    for index, path in enumerate(physics_paths):
        content = f"shared-physics-{index}"
        physics[path] = write(runtime, path, content)
        assert write(evidence, path, content) == physics[path]
    active_imports = {
        path: write(runtime, path, f"active-import-{index}")
        for index, path in enumerate(("configs/task.json", "assets/build.json"))
    }
    evidence_configs = {path: physics[path] for path in physics_paths[:5]}
    for index, path in enumerate(("configs/evidence-a.json", "configs/evidence-b.json")):
        evidence_configs[path] = write(evidence, path, f"evidence-config-{index}")

    runtime_manifest_relative = "configs/runtime-manifest.json"
    evidence_manifest_relative = "configs/evidence-manifest.json"
    physics_manifest_relative = "configs/physics-manifest.json"
    base_relative = "configs/base-physics.json"
    runtime_manifest_sha = write(
        current, runtime_manifest_relative, json.dumps({"files": runtime_python})
    )
    evidence_manifest_sha = write(
        evidence, evidence_manifest_relative, json.dumps({"files": evidence_python})
    )
    base_sha = write(current, base_relative, json.dumps({"physics_inputs": physics}))
    physics_manifest = {
        "base_physics_manifest": {"relative_path": base_relative, "sha256": base_sha},
        "runtime_core_manifest": {
            "relative_path": runtime_manifest_relative,
            "sha256": runtime_manifest_sha,
        },
        "evidence_core_manifest": {
            "relative_path": evidence_manifest_relative,
            "sha256": evidence_manifest_sha,
        },
        "active_import_inputs": active_imports,
        "evidence_source_config_inputs": evidence_configs,
        "shared_bank_catalog_relative": physics_paths[0],
    }
    physics_manifest_sha = write(
        current, physics_manifest_relative, json.dumps(physics_manifest)
    )
    contract = {
        "inputs": {
            "runtime_root": str(runtime),
            "evidence_root": str(evidence),
            "runtime_manifest_path": runtime_manifest_relative,
            "runtime_manifest_sha256": runtime_manifest_sha,
            "evidence_manifest_path": evidence_manifest_relative,
            "evidence_manifest_sha256": evidence_manifest_sha,
            "physics_input_manifest_path": physics_manifest_relative,
            "physics_input_manifest_sha256": physics_manifest_sha,
        }
    }
    closure = runner.validate_source_closure(contract, current_root=current)
    assert {name: len(values) for name, values in closure.items()} == {
        "runtime": 50,
        "evidence": 50,
    }
    (runtime / python_paths[0]).write_text("tampered", encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="SHA-256 mismatch"):
        runner.validate_source_closure(contract, current_root=current)


def test_runtime_import_origins_are_bound_to_active_root(tmp_path: Path) -> None:
    package = tmp_path / "src/rby1_assembly"
    package.mkdir(parents=True)
    names = (
        "experiment_contracts",
        "held_grasp",
        "isaac_env",
        "rj45_env",
        "rj45_task",
        "rj45_training",
        "rj45_training_env",
    )
    modules = {}
    for name in names:
        path = package / f"{name}.py"
        path.write_text("", encoding="utf-8")
        modules[name] = SimpleNamespace(__file__=str(path))
    observed = runner.validate_import_origins(modules, tmp_path)
    assert set(observed) == set(names)
    assert observed["rj45_env"] == str((package / "rj45_env.py").resolve())
    with pytest.raises(runner.LongDriverBlocked, match="escaped"):
        runner.validate_import_origins(
            {"rj45_env": SimpleNamespace(__file__=str(tmp_path / "rj45_env.py"))},
            tmp_path,
        )


def test_train_and_dev_environment_configs_must_be_byte_and_normalized_equal(
    tmp_path: Path,
) -> None:
    roots = [tmp_path / name for name in ("runtime", "train", "dev")]
    for root in roots:
        for relative, payload in (
            ("configs/runtime.json", {"control": 30}),
            ("configs/task.json", {"observation_dim": 79, "horizon": 12}),
            ("configs/training.json", {"warmup": 4}),
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload), encoding="utf-8")

    def load_training(path: Path):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["_config_sha256"] = runner.file_sha256(path)
        return value

    record = runner.validate_split_environment_inputs(
        runtime_root=roots[0],
        evidence_root=roots[1],
        development_evidence_root=roots[2],
        task_relative="configs/task.json",
        training_relative="configs/training.json",
        training_loader=load_training,
    )
    assert record["task_config"]["sha256"] == runner.file_sha256(
        roots[0] / "configs/task.json"
    )
    (roots[2] / "configs/training.json").write_text(
        json.dumps({"warmup": 5}), encoding="utf-8"
    )
    with pytest.raises(runner.LongDriverBlocked, match="training config bytes"):
        runner.validate_split_environment_inputs(
            runtime_root=roots[0],
            evidence_root=roots[1],
            development_evidence_root=roots[2],
            task_relative="configs/task.json",
            training_relative="configs/training.json",
            training_loader=load_training,
        )


def test_ppo_rollout_steps_use_proven_four_env_rows() -> None:
    assert runner.ppo_rollout_steps(
        {"ppo_rollout_target_valid_transitions": 4, "num_envs": 4}
    ) == 1
    assert runner.ppo_rollout_steps(
        {"ppo_rollout_target_valid_transitions": 8, "num_envs": 4}
    ) == 2


def test_policy_initialization_is_frozen_and_zero_mean() -> None:
    contract = _frozen()
    validated = runner.validate_contract(contract, execution_mode="train")
    assert validated["policy_initialization"] == {
        "mean_initializer": "zero",
        "initial_log_std": -3.0,
        "selection_source": "frozen_prelaunch_contract_not_runtime_outcome",
    }
    contract["policy_initialization"]["mean_initializer"] = "legacy_random"
    with pytest.raises(runner.LongDriverBlocked, match="zero baseline"):
        runner.validate_contract(contract, execution_mode="train")


def test_outcome_gate_rejects_zero_valid_updates_and_incomplete_conditions() -> None:
    minimums = _frozen()["minimum_outcome_contract"]
    progress = SimpleNamespace(
        valid_policy_transitions=0,
        algorithm_updates=0,
        actual_update_input_transitions=0,
    )
    pilot = SimpleNamespace(
        condition_ids=tuple(f"condition-{index}" for index in range(12)),
        transition_ever_checked=(False,) * 12,
        transition_ever_exact_zero=(False,) * 12,
        transition_check_counts=(0,) * 12,
        environment_rng_fingerprint="e" * 64,
        reset_rng_fingerprints=("f" * 64,) * 3,
        initial_observation_sha256="d" * 64,
        rng_isolation_scope="cpu-test",
    )
    assert runner.validate_run_outcome(progress, pilot, minimums) == [
        "minimum_valid_policy_transitions",
        "minimum_algorithm_updates",
        "minimum_actual_update_input_transitions",
        "all_12_conditions_reach_align_to_insert",
        "all_transition_actions_exact_zero",
        "minimum_condition_transition_count_per_condition",
    ]


def test_invalid_training_outcome_never_promotes_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = runner.build_paired_schedule(_frozen())["runs"][0]
    progress = SimpleNamespace(
        raw_transitions=120,
        valid_policy_transitions=0,
        warmup_transitions=120,
        algorithm_updates=0,
        optimizer_steps=0,
        pending_policy_transitions=0,
        actual_update_input_transitions=0,
        discarded_tail_valid_transitions=0,
        random_warmup_transitions=0,
        replay_insertions=0,
        replay_size=0,
        replay_first_ready_at=0,
        sac_update_credit=0.0,
        sac_first_update_after_valid_insertions=0,
    )
    result = SimpleNamespace(
        transitions=120,
        learning_transitions=0,
        valid_policy_transitions=0,
        warmup_transitions=120,
        wave_boundaries=(
            SimpleNamespace(
                raw_transitions=40,
                valid_policy_transitions=0,
                algorithm_updates=0,
                actual_update_input_transitions=0,
                pending_policy_transitions=0,
            ),
            SimpleNamespace(
                raw_transitions=80,
                valid_policy_transitions=0,
                algorithm_updates=0,
                actual_update_input_transitions=0,
                pending_policy_transitions=0,
            ),
        ),
    )
    pilot = SimpleNamespace(
        condition_ids=tuple(f"condition-{index}" for index in range(12)),
        transition_ever_checked=(False,) * 12,
        transition_ever_exact_zero=(False,) * 12,
        transition_check_counts=(0,) * 12,
        environment_rng_fingerprint="e" * 64,
        reset_rng_fingerprints=("f" * 64,) * 3,
        initial_observation_sha256="d" * 64,
        rng_isolation_scope="cpu-test",
    )

    class FakeEnv:
        def reset(self):
            return object()

        def comparison_pilot_progress(self):
            return pilot

        def advance_comparison_pilot_wave(self, _index):
            return object()

        def close(self):
            return None

    fake_module = SimpleNamespace(
        torch=torch,
        make_agent=lambda **_kwargs: object(),
        collect_bounded=lambda *_args, progress_callback=None, **_kwargs: (
            progress_callback(progress) if progress_callback is not None else None
        )
        or result,
        collector_progress=lambda _agent: progress,
        discard_pending_ppo_tail=lambda _agent: 0,
    )
    prepared = {
        "validated": {
            "budget": _frozen()["budget"],
            "minimums": _frozen()["minimum_outcome_contract"],
            "policy_initialization": _frozen()["policy_initialization"],
            "schedule": {"runs": [run]},
            "wave_vector_steps_plan": (10, 10, 10),
        },
        "input_snapshot": {"canonical_sha256": "a" * 64},
    }
    monkeypatch.setattr(runner, "_load_current_module", lambda _name: fake_module)
    monkeypatch.setattr(runner, "_build_env", lambda *_args, **_kwargs: FakeEnv())
    monkeypatch.setattr(
        runner,
        "verify_input_snapshot",
        lambda _snapshot: {"end_canonical_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        runner,
        "write_checkpoint_bundle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid outcome promoted a checkpoint")
        ),
    )
    report = runner._train(prepared, tmp_path)
    assert report["status"] == "invalid"
    assert report["runs"][0]["checkpoint_promoted"] is False
    assert report["runs"][0]["checkpoint"] is None
    assert "learning_transitions" not in report["runs"][0]["collector_result"]


def test_wave_accounting_reconciles_repeated_three_wave_cycles() -> None:
    boundaries = tuple(
        SimpleNamespace(
            raw_transitions=40 * index,
            valid_policy_transitions=32 * index,
            algorithm_updates=index,
            actual_update_input_transitions=16 * index,
            pending_policy_transitions=index,
        )
        for index in range(1, 6)
    )
    result = SimpleNamespace(wave_boundaries=boundaries)
    final = SimpleNamespace(
        raw_transitions=240,
        valid_policy_transitions=192,
        algorithm_updates=6,
        actual_update_input_transitions=96,
        pending_policy_transitions=0,
    )
    rows = runner.build_wave_accounting(
        result,
        final,
        wave_vector_steps_plan=(10,) * 6,
        num_envs=4,
    )
    assert [row["catalog_wave_index"] for row in rows] == [0, 1, 2, 0, 1, 2]
    assert [row["cycle_index"] for row in rows] == [0, 0, 0, 1, 1, 1]
    assert [row["raw_transitions"] for row in rows] == [40] * 6


def test_active_runtime_and_current_adapter_paths_are_resolved_separately(
    tmp_path: Path,
) -> None:
    actual = runner.resolve_environment_module_paths(
        runtime_root=Path(
            "/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf"
        ),
        current_root=ROOT,
    )
    assert actual["rj45_task"].is_file()
    assert actual["rj45_comparison_env"] == ROOT / "src/rby1_assembly/rj45_comparison_env.py"
    loaded_current = runner._load_module_path(
        actual["rj45_comparison_env"], "g2_test_current_comparison_env"
    )
    assert callable(loaded_current.make_rj45_comparison_env_type)
    active_root = Path(
        "/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf"
    )
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                "from rby1_assembly.rj45_task import load_rj45_task; "
                f"p=Path({str(active_root / 'configs/tasks/rj45-v1.json')!r}); "
                "t=load_rj45_task(p); "
                "assert len(t['_config_sha256']) == 64; "
                "import importlib.util; "
                "assert importlib.util.find_spec('rby1_assembly.rj45_comparison_env') is None"
            ),
        ],
        env={"PYTHONPATH": str(active_root / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert check.returncode == 0, check.stderr

    runtime = tmp_path / "runtime"
    current = tmp_path / "current"
    (runtime / "src/rby1_assembly").mkdir(parents=True)
    (current / "src/rby1_assembly").mkdir(parents=True)
    for name in (
        "rj45_task",
        "rj45_env",
        "rj45_training",
        "rj45_training_env",
        "parallel_contact_evidence",
    ):
        (runtime / f"src/rby1_assembly/{name}.py").write_text("", encoding="utf-8")
    for name in ("rj45_comparison_env", "rj45_comparison_pilot_env"):
        (current / f"src/rby1_assembly/{name}.py").write_text("", encoding="utf-8")
    (runtime / "src/rby1_assembly/rj45_comparison_env.py").write_text("", encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="reviewed current root"):
        runner.resolve_environment_module_paths(
            runtime_root=runtime, current_root=current
        )


def test_checkpoint_sidecar_roundtrip_and_binary_tamper(tmp_path: Path) -> None:
    comparison_rl = runner._load_current_module("rj45_comparison_rl")
    agent = comparison_rl.make_agent("ppo", num_envs=1, rollout_steps=2, batch_size=2)
    checkpoint = tmp_path / "ppo.pt"
    bundle = runner.write_checkpoint_bundle(
        agent,
        checkpoint,
        algorithm="ppo",
        metadata={
            "contract_sha256": "d" * 64,
            "condition_schedule_sha256": "e" * 64,
            "seed": 101,
        },
    )
    sidecar = runner.verify_checkpoint_bundle(
        checkpoint,
        bundle["sidecar"],
        expected={"contract_sha256": "d" * 64, "algorithm": "ppo", "seed": 101},
    )
    restored = comparison_rl.make_agent(
        "ppo", num_envs=1, rollout_steps=2, batch_size=2, seed=99
    )
    comparison_rl.load_inference_checkpoint(restored, checkpoint)
    observations = torch.randn(3, 79)
    assert torch.equal(
        comparison_rl.deterministic_action(agent, observations),
        comparison_rl.deterministic_action(restored, observations),
    )
    assert sidecar["resume_supported"] is False
    checkpoint.write_bytes(checkpoint.read_bytes() + b"tamper")
    with pytest.raises(runner.LongDriverBlocked, match="SHA-256"):
        runner.verify_checkpoint_bundle(checkpoint, bundle["sidecar"], expected={})


def test_frozen_run_manifest_binds_sidecar_seed_and_full_run_identity(
    tmp_path: Path,
) -> None:
    contract = _frozen()
    validated = runner.validate_contract(contract, execution_mode="evaluate-dev")
    run = validated["schedule"]["runs"][0]
    prepared = {
        "contract": contract,
        "contract_path": tmp_path / "contract.json",
        "contract_sha256": "1" * 64,
        "g1_path": tmp_path / "g1.json",
        "g1_sha256": "2" * 64,
        "g1_report": {
            "schema_name": "rj45-task-admission-isolated-report-v1",
            "status": "admit_rl_pilot",
            "claims": {
                **contract["g1_gate"]["required_claims"],
                "native_crash_cause_fixed": False,
            },
        },
        "g1_scope": {
            "schema_name": "rj45_g1_gate_normalized_v1",
            "source_report_schema": "rj45-task-admission-isolated-report-v1",
            "status": "admit_rl_pilot",
            "non_smoke": True,
            "claims": contract["g1_gate"]["required_claims"],
            "scientific_scope_claims": {
                "physical_initial_pose_error_sweep": False,
                "success_range_curve_proven": False,
            },
            "producer_claims": {
                **contract["g1_gate"]["required_claims"],
                "native_crash_cause_fixed": False,
            },
            "campaign": {
                "frozen_path": "/frozen/campaign.json",
                "reported_source_path": "/producer/campaign.json",
                "sha256": "7" * 64,
                "purpose": "formal_campaign",
                "environment_seed": 20261121,
                "episode_count": 90,
                "tuning_episode_count": 18,
                "admission_episode_count": 72,
            },
        },
        "development_config_path": tmp_path / "development.json",
        "development_config_sha256": "3" * 64,
        "held_bank_sha256": "4" * 64,
        "development_held_bank_sha256": "6" * 64,
        "validated": validated,
        "input_snapshot": {
            "canonical_sha256": "5" * 64,
            "normalized_sha256": contract["source_and_config_sha256"],
        },
    }
    comparison_rl = runner._load_current_module("rj45_comparison_rl")
    agent = comparison_rl.make_agent(
        "ppo",
        num_envs=1,
        rollout_steps=2,
        batch_size=2,
        mean_initializer="zero",
        initial_log_std=-3.0,
    )
    metadata = {
        **runner._checkpoint_metadata(prepared, run),
        "source_input_end_sha256": "5" * 64,
        "terminal_checkpoint_policy": "last_completed_update",
        "diagnostic_best_promoted": False,
    }
    assert metadata["g1_non_smoke"] is True
    assert metadata["g1_gate_scope"] == prepared["g1_scope"]
    assert metadata["g1_source_report_schema"] == (
        "rj45-task-admission-isolated-report-v1"
    )
    assert "g1_smoke_only" not in metadata
    bundle = runner.write_checkpoint_bundle(
        agent,
        tmp_path / "ppo.pt",
        algorithm="ppo",
        metadata=metadata,
    )
    manifest_path = tmp_path / "ppo.run.json"
    runner.write_run_manifest(
        manifest_path,
        prepared=prepared,
        run=run,
        checkpoint_path=bundle["checkpoint"],
        sidecar_path=bundle["sidecar"],
        source_input_end_sha256="5" * 64,
    )
    manifest_sha = runner.file_sha256(manifest_path)
    assert runner.verify_run_manifest(
        manifest_path, manifest_sha, prepared=prepared
    )["run"] == run

    sidecar = json.loads(bundle["sidecar"].read_text(encoding="utf-8"))
    sidecar["seed"] = 303
    bundle["sidecar"].write_text(json.dumps(sidecar), encoding="utf-8")
    with pytest.raises(runner.LongDriverBlocked, match="artifact hashes"):
        runner.verify_run_manifest(manifest_path, manifest_sha, prepared=prepared)


def test_contract_file_sha_is_required_exactly(tmp_path: Path) -> None:
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(_draft()), encoding="utf-8")
    args = runner.parse_args(
        [
            "--contract",
            str(path),
            "--contract-sha256",
            "0" * 64,
            "--execution-mode",
            "train",
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    with pytest.raises(runner.LongDriverBlocked, match="contract SHA-256 mismatch"):
        runner._prepare(args)
