from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
COORDINATOR_PATH = ROOT / "scripts/run_rj45_task_admission_isolated.py"
RUNNER_PATH = ROOT / "scripts/run_rj45_task_admission.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COORDINATOR = _load(COORDINATOR_PATH, "test_isolated_coordinator")
RUNNER = _load(RUNNER_PATH, "test_isolated_runner")
ADMISSION = _load(
    ROOT / "src/rby1_assembly/rj45_task_admission.py", "test_isolated_admission"
)
CONFIG = json.loads(
    (ROOT / "configs/rj45-comparison-development.json").read_text(encoding="utf-8")
)
PLAN = RUNNER.build_execution_plan(CONFIG, smoke_only=False)
PLAYBACK_PATH = ROOT / "docs/evidence/rj45-frozen-ablation-20261031/source-actions.json"
PLAYBACK = json.loads(PLAYBACK_PATH.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _snapshot() -> dict:
    normalized = {
        "runtime": {"input.usd": "1" * 64},
        "current": {relative: "2" * 64 for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS},
    }
    return {
        "schema_name": RUNNER.PHYSICS_INPUT_SNAPSHOT_SCHEMA,
        "normalized_sha256": normalized,
        "normalized_sha256_canonical_sha256": RUNNER._canonical_json_sha256(normalized),
        "manifest": {"path": "/tmp/physics.json", "sha256": "3" * 64},
    }


def _campaign(
    tmp_path: Path, *, purpose: str = "formal_campaign"
) -> tuple[Path, dict, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": COORDINATOR.CAMPAIGN_SCHEMA,
        "purpose": purpose,
        "environment_seed": RUNNER.TASK_ADMISSION_ENVIRONMENT_SEED,
        "execution_plan": [
            COORDINATOR._plan_record(index, row)
            for index, row in enumerate(PLAN, start=1)
        ],
        "config": {
            "path": str(ROOT / "configs/rj45-comparison-development.json"),
            "sha256": _sha(ROOT / "configs/rj45-comparison-development.json"),
        },
        "playback_actions": {
            "path": str(PLAYBACK_PATH),
            "sha256": _sha(PLAYBACK_PATH),
            "canonical_sha256": PLAYBACK["canonical_sha256"],
        },
        "physics_input_start": _snapshot(),
    }
    path = tmp_path / "campaign.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload, _sha(path)


def _prepared() -> dict:
    return {
        "plan": PLAN,
        "physics_input_snapshot": _snapshot(),
        "config_path": ROOT / "configs/rj45-comparison-development.json",
        "playback_path": PLAYBACK_PATH,
        "playback_canonical_sha256": PLAYBACK["canonical_sha256"],
    }


def _valid_worker_spec(tmp_path: Path, start: int, end: int, *, purpose="formal_campaign"):
    campaign_path, campaign, campaign_sha = _campaign(tmp_path, purpose=purpose)
    selected = "B" if any(
        PLAN[index - 1][0] == "admission"
        and PLAN[index - 1][3] == "classical_force_search"
        for index in range(start, end + 1)
    ) else None
    tuning_binding = None
    if start >= 19 and purpose == "formal_campaign":
        tuning_path = tmp_path / "tuning-seal.json"
        tuning_path.write_text(
            json.dumps(
                {
                    "episode_count": 18,
                    "campaign_sha256": campaign_sha,
                    "selection": {"candidate_id": selected},
                    "classical_seal": {"selected_candidate_id": selected},
                }
            ),
            encoding="utf-8",
        )
        tuning_binding = {"path": str(tuning_path), "sha256": _sha(tuning_path)}
    spec = COORDINATOR._worker_spec(
        campaign_path,
        campaign_sha,
        campaign,
        tuple(range(start, end + 1)),
        purpose=purpose,
        selected_classical_candidate=selected,
        tuning_seal=tuning_binding,
    )
    path = tmp_path / "worker-spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    args = SimpleNamespace(
        worker_spec=path,
        expected_worker_spec_sha256=_sha(path),
        smoke_only=False,
        validate_only=False,
        verify_after_child_exit=False,
    )
    return args, spec


def test_exact_partitions_cover_90_and_never_exceed_12() -> None:
    partitions = COORDINATOR._fixed_partitions(PLAN)
    assert tuple(value for chunk in partitions for value in chunk) == tuple(range(1, 91))
    assert max(map(len, partitions)) == 12
    assert partitions[:3] == (tuple(range(1, 7)), tuple(range(7, 13)), tuple(range(13, 19)))


def test_coordinator_source_has_no_native_imports() -> None:
    source = COORDINATOR_PATH.read_text(encoding="utf-8")
    for forbidden in ("import torch", "import isaac", "import omni"):
        assert forbidden not in source


@pytest.mark.parametrize("start,end", COORDINATOR.FORMAL_CHUNKS)
def test_runner_accepts_only_fixed_formal_chunks(tmp_path: Path, start: int, end: int) -> None:
    args, _spec = _valid_worker_spec(tmp_path, start, end)
    worker = RUNNER._validate_worker_spec(args, _prepared())
    assert worker is not None
    assert [row["ordinal"] for row in worker["rows"]] == list(range(start, end + 1))


def test_runner_rejects_worker_spec_sha_tampering(tmp_path: Path) -> None:
    args, spec = _valid_worker_spec(tmp_path, 1, 6)
    args.expected_worker_spec_sha256 = "0" * 64
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="spec SHA"):
        RUNNER._validate_worker_spec(args, _prepared())


def test_worker_preflight_and_remaining_bindings_fail_closed(tmp_path: Path) -> None:
    current = tmp_path / "current"
    for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS:
        destination = current / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    normalized = {
        "runtime": {"input.usd": "1" * 64},
        "current": {
            relative: _sha(current / relative)
            for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS
        },
    }
    snapshot = {
        "schema_name": RUNNER.PHYSICS_INPUT_SNAPSHOT_SCHEMA,
        "normalized_sha256": normalized,
        "normalized_sha256_canonical_sha256": RUNNER._canonical_json_sha256(normalized),
        "manifest": {"path": "/tmp/physics.json", "sha256": "3" * 64},
    }
    campaign_path = tmp_path / "campaign.json"
    campaign_path.write_text(
        json.dumps(
            {
                "schema_name": RUNNER.ISOLATED_CAMPAIGN_SCHEMA,
                "physics_input_start": snapshot,
            }
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "campaign_identity_path": str(campaign_path),
                "campaign_sha256": _sha(campaign_path),
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "validator-imported"
    validator = current / RUNNER.END_VALIDATOR_RELATIVE
    validator.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n",
        encoding="utf-8",
    )
    args = SimpleNamespace(
        worker_spec=spec_path,
        expected_worker_spec_sha256=_sha(spec_path),
    )
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="current source changed"):
        RUNNER._prevalidate_worker_validator(args, current)
    assert not marker.exists()

    args, spec = _valid_worker_spec(tmp_path / "campaign", 1, 6)
    campaign_path = Path(spec["campaign_identity_path"])
    campaign = json.loads(campaign_path.read_text())
    campaign["execution_plan"][0]["method_id"] = "mean_constant"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    spec["campaign_sha256"] = _sha(campaign_path)
    args.worker_spec.write_text(json.dumps(spec), encoding="utf-8")
    args.expected_worker_spec_sha256 = _sha(args.worker_spec)
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="execution plan"):
        RUNNER._validate_worker_spec(args, _prepared())

    args, _spec = _valid_worker_spec(tmp_path / "inputs", 1, 6)
    prepared = _prepared()
    prepared["physics_input_snapshot"]["normalized_sha256_canonical_sha256"] = "9" * 64
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="physics inputs"):
        RUNNER._validate_worker_spec(args, prepared)

    args, spec = _valid_worker_spec(tmp_path / "row", 1, 6)
    spec["rows"][1] = dict(spec["rows"][0])
    args.worker_spec.write_text(json.dumps(spec), encoding="utf-8")
    args.expected_worker_spec_sha256 = _sha(args.worker_spec)
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="contiguous"):
        RUNNER._validate_worker_spec(args, _prepared())

    args, spec = _valid_worker_spec(tmp_path / "candidate", 19, 30)
    tuning_path = Path(spec["tuning_seal"]["path"])
    tuning = json.loads(tuning_path.read_text())
    tuning["classical_seal"]["selected_candidate_id"] = "A"
    tuning_path.write_text(json.dumps(tuning), encoding="utf-8")
    spec["tuning_seal"]["sha256"] = _sha(tuning_path)
    args.worker_spec.write_text(json.dumps(spec), encoding="utf-8")
    args.expected_worker_spec_sha256 = _sha(args.worker_spec)
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="candidate differs"):
        RUNNER._validate_worker_spec(args, _prepared())


def test_diagnostic_is_only_ordinal33_and_nonadjudicative(tmp_path: Path) -> None:
    args, spec = _valid_worker_spec(
        tmp_path, 33, 33, purpose="diagnostic_nonadjudicative"
    )
    worker = RUNNER._validate_worker_spec(args, _prepared())
    assert worker["purpose"] == "diagnostic_nonadjudicative"
    assert worker["rows"][0] == COORDINATOR._plan_record(33, PLAN[32])
    assert PLAN[32][1:5] == (
        "rj45-train-02",
        "tz-positive",
        "mean_constant",
        1,
    )

    spec["rows"] = [COORDINATOR._plan_record(34, PLAN[33])]
    args.worker_spec.write_text(json.dumps(spec), encoding="utf-8")
    args.expected_worker_spec_sha256 = _sha(args.worker_spec)
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="ordinal 33"):
        RUNNER._validate_worker_spec(args, _prepared())


def test_worker_identity_references_single_global_physics_map(tmp_path: Path) -> None:
    campaign_path, campaign, campaign_sha = _campaign(tmp_path)
    worker = {
        "purpose": "formal_campaign",
        "campaign_path": campaign_path,
        "campaign_sha256": campaign_sha,
        "path": tmp_path / "spec.json",
        "sha256": "4" * 64,
        "rows": campaign["execution_plan"][:6],
    }
    worker["path"].write_text("{}")
    prepared = {"physics_input_snapshot": campaign["physics_input_start"]}
    identity_ref = RUNNER._write_worker_start_identity(tmp_path / "worker", prepared, worker)
    identity = json.loads(Path(identity_ref["path"]).read_text())
    assert "physics_input_start" not in identity
    assert RUNNER._physics_start_from_identity(identity) == campaign["physics_input_start"]
    spec = COORDINATOR._worker_spec(
        campaign_path,
        campaign_sha,
        campaign,
        (1,),
        purpose="formal_campaign",
        selected_classical_candidate=None,
    )
    assert "physics_input_start" not in spec
    campaign["purpose"] = "diagnostic_nonadjudicative"
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="campaign identity SHA"):
        RUNNER._physics_start_from_identity(identity)


def _actual_shaped_attestation_inputs(tmp_path: Path):
    runtime_root = tmp_path / "runtime"
    (runtime_root / "configs").mkdir(parents=True)
    shutil.copyfile(ROOT / "configs/runtime.json", runtime_root / "configs/runtime.json")
    bank_root = runtime_root / "artifacts/rj45-4c-bank-generation-v2-20261028"
    bank_root.mkdir(parents=True)
    catalog_path = bank_root / "catalog.json"
    catalog_path.write_text('{"catalog": 1}\n', encoding="utf-8")
    entries = []
    for index in range(1, 9):
        grasp_id = f"rj45-train-{index:02d}"
        report = bank_root / f"report-{index:02d}.json"
        trace = bank_root / f"trace-{index:02d}.jsonl"
        artifact = bank_root / f"artifact-{index:02d}.json"
        report.write_text(f'{{"report": {index}}}\n', encoding="utf-8")
        trace.write_text(f'{{"trace": {index}}}\n', encoding="utf-8")
        artifact.write_text(
            json.dumps(
                {
                    "provenance": {
                        "nominal_report_path": str(report.relative_to(runtime_root)),
                        "nominal_report_sha256": _sha(report),
                        "nominal_trace_path": str(trace.relative_to(runtime_root)),
                        "nominal_trace_sha256": _sha(trace),
                        "grasp_generation": {
                            "catalog_path": str(catalog_path.relative_to(runtime_root)),
                            "catalog_sha256": _sha(catalog_path),
                        },
                    }
                }
            ),
            encoding="utf-8",
        )
        entries.append(
            {
                "grasp_id": grasp_id,
                "split": "train",
                "artifact": {
                    "path": str(artifact.relative_to(runtime_root)),
                    "sha256": _sha(artifact),
                },
            }
        )
    manifest_path = bank_root / "held-grasp-bank.json"
    manifest_path.write_text(
        json.dumps({"schema_version": 1, "task_id": "rj45-v1", "entries": entries}),
        encoding="utf-8",
    )
    physics_runtime = {
        **{f"src/rby1_assembly/runtime_{index:02d}.py": f"{index + 1:064x}" for index in range(35)},
        **{f"assets/physics_{index:02d}.usd": f"{index + 101:064x}" for index in range(13)},
    }
    snapshot = {
        "normalized_sha256": {
            "runtime": physics_runtime,
            "current": {
                relative: _sha(ROOT / relative)
                for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS
            },
        }
    }
    prepared = {"physics_input_snapshot": snapshot}
    runner = SimpleNamespace(
        EXPECTED_BANK_SHA256=_sha(manifest_path),
        EXPECTED_BANK_SOURCE_SHA256="9" * 64,
        EXPECTED_IDS=RUNNER.EXPECTED_IDS,
    )
    args = SimpleNamespace(runtime_root=runtime_root, current_root=ROOT)
    return args, prepared, runner, report, manifest_path


def test_trusted_bank_builder_accepts_actual_physics_shape_and_rehashes_bank26(
    tmp_path: Path,
) -> None:
    args, prepared, runner, report, _manifest = _actual_shaped_attestation_inputs(tmp_path)
    trusted = COORDINATOR._trusted_worker_attestation(args, prepared, runner)
    frozen = trusted["frozen_input_sha256"]
    physics = prepared["physics_input_snapshot"]["normalized_sha256"]["runtime"]
    assert len(physics) == 48
    assert len(frozen) == 26
    frozen_relative = {
        str(Path(path).relative_to(args.runtime_root)) for path in frozen
    }
    assert frozen_relative.isdisjoint(physics)
    campaign = {"trusted_worker_attestation": trusted}
    COORDINATOR._verify_trusted_bank_inputs(campaign, args, runner)

    report.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="frozen bank input changed"):
        COORDINATOR._verify_trusted_bank_inputs(campaign, args, runner)
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="nominal report.*hash differs"):
        COORDINATOR._trusted_worker_attestation(args, prepared, runner)


def _worker_output(tmp_path: Path, *, start: int = 1, end: int = 1):
    campaign_path, campaign, campaign_sha = _campaign(tmp_path)
    runtime_root = tmp_path / "runtime"
    runtime_origin = runtime_root / "src/rby1_assembly/__init__.py"
    runtime_origin.parent.mkdir(parents=True)
    runtime_origin.write_text("# frozen runtime\n", encoding="utf-8")
    held_grasp_origin = runtime_origin.parent / "held_grasp.py"
    held_grasp_origin.write_text("# frozen held grasp\n", encoding="utf-8")
    rj45_task_origin = runtime_origin.parent / "rj45_task.py"
    rj45_task_origin.write_text("# frozen task\n", encoding="utf-8")
    frozen_input = runtime_root / "artifacts/frozen.json"
    frozen_input.parent.mkdir(parents=True)
    frozen_input.write_text("{}\n", encoding="utf-8")
    campaign["physics_input_start"]["normalized_sha256"] = {
        "runtime": {
            str(runtime_origin.relative_to(runtime_root)): _sha(runtime_origin),
            str(held_grasp_origin.relative_to(runtime_root)): _sha(held_grasp_origin),
            str(rj45_task_origin.relative_to(runtime_root)): _sha(rj45_task_origin),
            str(frozen_input.relative_to(runtime_root)): _sha(frozen_input),
        },
        "current": {
            relative: _sha(ROOT / relative)
            for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS
        },
    }
    campaign["physics_input_start"]["normalized_sha256_canonical_sha256"] = (
        RUNNER._canonical_json_sha256(
            campaign["physics_input_start"]["normalized_sha256"]
        )
    )
    trusted_attestation = {
        "runtime_versions": {
            "isaacsim": "5.1.0.0",
            "isaaclab": "0.47.2",
            "isaaclab-rl": "0.4.4",
            "rsl-rl-lib": "3.0.1",
            "torch": "2.7.0+cu128",
            "numpy": "1.26.0",
            "h5py": "3.14.0",
        },
        "frozen_input_sha256": {str(frozen_input): _sha(frozen_input)},
        "held_bank_identity": {
            "manifest_sha256": RUNNER.EXPECTED_BANK_SHA256,
            "selected_grasp_ids": list(RUNNER.EXPECTED_IDS),
            "selected_artifact_sha256": [f"{index:064x}" for index in range(1, 9)],
            "source_contract_sha256": RUNNER.EXPECTED_BANK_SOURCE_SHA256,
        },
        "current_module_origins": {
            name: str(ROOT / f"src/rby1_assembly/{name}.py")
            for name in COORDINATOR.CURRENT_WORKER_MODULES
        },
    }
    campaign["trusted_worker_attestation"] = trusted_attestation
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    campaign_sha = _sha(campaign_path)
    spec = COORDINATOR._worker_spec(
        campaign_path,
        campaign_sha,
        campaign,
        tuple(range(start, end + 1)),
        purpose="formal_campaign",
        selected_classical_candidate=None,
    )
    spec_path = tmp_path / "worker-spec.json"
    spec_path.write_text(json.dumps(spec), encoding="utf-8")
    spec_sha = _sha(spec_path)
    attestation = {
        "schema_name": "rj45-task-admission-isolated-worker-attestation-v1",
        "campaign_sha256": campaign_sha,
        "worker_spec_sha256": spec_sha,
        "runtime_import_origins": {
            "rby1_assembly": str(runtime_origin),
            "rby1_assembly.held_grasp": str(held_grasp_origin),
            "rby1_assembly.rj45_task": str(rj45_task_origin),
        },
        **trusted_attestation,
    }
    attestation_path = tmp_path / "worker-attestation.json"
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    identity = {
        "schema_name": RUNNER.ISOLATED_WORKER_IDENTITY_SCHEMA,
        "environment_seed": RUNNER.TASK_ADMISSION_ENVIRONMENT_SEED,
        "purpose": "formal_campaign",
        "campaign_identity": {"path": str(campaign_path), "sha256": campaign_sha},
        "worker_spec": {"path": str(spec_path.resolve()), "sha256": spec_sha},
        "execution_plan": spec["rows"],
    }
    identity_path = tmp_path / "run-identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    identity_sha = _sha(identity_path)
    refs = []
    for ordinal in range(start, end + 1):
        expected = campaign["execution_plan"][ordinal - 1]
        source = {
            key: expected[key]
            for key in ("stage", "grasp_id", "case_id", "method_id", "slot", "candidate_id")
        }
        episode = {**source, "execution_error": None, "terminal_complete": True}
        envelope = {
            "schema_name": "rj45-task-admission-episode-row-v1",
            "ordinal": ordinal,
            "campaign_sha256": campaign_sha,
            "run_identity_sha256": identity_sha,
            "source_identity": source,
            "episode": episode,
        }
        path = tmp_path / "episode-rows" / f"{ordinal:03d}.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(envelope), encoding="utf-8")
        refs.append({"ordinal": ordinal, "path": str(path), "sha256": _sha(path)})
    (tmp_path / "physics-input-end-verification.json").write_text(
        json.dumps({"matched": True, "scientific_completion_allowed": True}), encoding="utf-8"
    )
    (tmp_path / "worker-exit.json").write_text(
        json.dumps({
            "return_code": 0,
            "timed_out": False,
            "worker_process_group_cleanup": {
                "pgid": 1001, "signals": [], "group_absent_after_cleanup": True,
            },
            "post_exit_verifier_return_code": 0,
            "post_exit_verifier_timed_out": False,
            "post_exit_verifier_process_group_cleanup": {
                "pgid": 1002, "signals": [], "group_absent_after_cleanup": True,
            },
        }),
        encoding="utf-8",
    )
    report = {
        "schema_name": COORDINATOR.CHUNK_REPORT_SCHEMA,
        "status": "worker_completed",
        "purpose": "formal_campaign",
        "campaign_sha256": campaign_sha,
        "worker_spec_sha256": spec_sha,
        "run_identity": {"path": str(identity_path), "sha256": identity_sha},
        "worker_attestation": {"path": str(attestation_path), "sha256": _sha(attestation_path)},
        "episode_rows": refs,
        "claims": {"task_admission_completed": False, "physics_success_proven": False},
    }
    (tmp_path / "chunk-report.json").write_text(json.dumps(report), encoding="utf-8")
    args = SimpleNamespace(runtime_root=runtime_root, current_root=ROOT)
    return spec, spec_sha, campaign, args


def test_worker_output_requires_exit_end_attestation_and_row_hashes(tmp_path: Path) -> None:
    spec, spec_sha, campaign, args = _worker_output(tmp_path)
    assert len(COORDINATOR._validate_worker_output(
        tmp_path, spec, spec_sha, args=args, campaign=campaign
    )) == 1
    (tmp_path / "worker-exit.json").unlink()
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="exit record"):
        COORDINATOR._validate_worker_output(
            tmp_path, spec, spec_sha, args=args, campaign=campaign
        )


@pytest.mark.parametrize(
    "missing,message",
    [("chunk-report.json", "cannot load"), ("physics-input-end-verification.json", "cannot load")],
)
def test_worker_output_rejects_missing_chunk_or_end_record(
    tmp_path: Path, missing: str, message: str
) -> None:
    spec, spec_sha, campaign, args = _worker_output(tmp_path)
    (tmp_path / missing).unlink()
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match=message):
        COORDINATOR._validate_worker_output(
            tmp_path, spec, spec_sha, args=args, campaign=campaign
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("duplicate", "row reference count"),
        ("swapped", "hash binding"),
        ("wrong_method", "source identity"),
        ("terminal_incomplete", "incomplete"),
        ("execution_error", "incomplete"),
        ("old_campaign", "campaign binding"),
        ("source_origin", "source-origin"),
    ],
)
def test_worker_output_rejects_mixed_or_incomplete_evidence(
    tmp_path: Path, mutation: str, message: str
) -> None:
    spec, spec_sha, campaign, args = _worker_output(tmp_path, start=1, end=2)
    report_path = tmp_path / "chunk-report.json"
    report = json.loads(report_path.read_text())
    first_path = tmp_path / "episode-rows/001.json"
    first = json.loads(first_path.read_text())
    if mutation == "duplicate":
        report["episode_rows"].append(dict(report["episode_rows"][0]))
        report_path.write_text(json.dumps(report))
    elif mutation == "swapped":
        report["episode_rows"][0], report["episode_rows"][1] = report["episode_rows"][1], report["episode_rows"][0]
        report_path.write_text(json.dumps(report))
    elif mutation == "wrong_method":
        first["episode"]["method_id"] = "zero"
        first_path.write_text(json.dumps(first))
        report["episode_rows"][0]["sha256"] = _sha(first_path)
        report_path.write_text(json.dumps(report))
    elif mutation == "terminal_incomplete":
        first["episode"]["terminal_complete"] = False
        first_path.write_text(json.dumps(first))
        report["episode_rows"][0]["sha256"] = _sha(first_path)
        report_path.write_text(json.dumps(report))
    elif mutation == "execution_error":
        first["episode"]["execution_error"] = {"type": "RuntimeError"}
        first_path.write_text(json.dumps(first))
        report["episode_rows"][0]["sha256"] = _sha(first_path)
        report_path.write_text(json.dumps(report))
    elif mutation == "old_campaign":
        first["campaign_sha256"] = "0" * 64
        first_path.write_text(json.dumps(first))
        report["episode_rows"][0]["sha256"] = _sha(first_path)
        report_path.write_text(json.dumps(report))
    else:
        attestation_path = tmp_path / "worker-attestation.json"
        attestation = json.loads(attestation_path.read_text())
        attestation["current_module_origins"].pop("rj45_task_admission")
        attestation_path.write_text(json.dumps(attestation))
        report["worker_attestation"]["sha256"] = _sha(attestation_path)
        report_path.write_text(json.dumps(report))
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match=message):
        COORDINATOR._validate_worker_output(
            tmp_path, spec, spec_sha, args=args, campaign=campaign
        )


@pytest.mark.parametrize(
    "mutation,message",
    [
        ("missing_report_ref", "run identity binding"),
        ("identity_content", "run identity content"),
        ("envelope_replay", "campaign binding"),
    ],
)
def test_worker_output_binds_exact_run_identity_and_every_envelope(
    tmp_path: Path, mutation: str, message: str
) -> None:
    spec, spec_sha, campaign, args = _worker_output(tmp_path)
    report_path = tmp_path / "chunk-report.json"
    report = json.loads(report_path.read_text())
    identity_path = tmp_path / "run-identity.json"
    row_path = tmp_path / "episode-rows/001.json"
    if mutation == "missing_report_ref":
        report.pop("run_identity")
    elif mutation == "identity_content":
        identity = json.loads(identity_path.read_text())
        identity["purpose"] = "diagnostic_nonadjudicative"
        identity_path.write_text(json.dumps(identity))
        forged_sha = _sha(identity_path)
        report["run_identity"]["sha256"] = forged_sha
        envelope = json.loads(row_path.read_text())
        envelope["run_identity_sha256"] = forged_sha
        row_path.write_text(json.dumps(envelope))
        report["episode_rows"][0]["sha256"] = _sha(row_path)
    else:
        envelope = json.loads(row_path.read_text())
        envelope["run_identity_sha256"] = "0" * 64
        row_path.write_text(json.dumps(envelope))
        report["episode_rows"][0]["sha256"] = _sha(row_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match=message):
        COORDINATOR._validate_worker_output(
            tmp_path, spec, spec_sha, args=args, campaign=campaign
        )


@pytest.mark.parametrize(
    "mutation",
    ("runtime_versions", "current_origin", "frozen_inputs", "held_bank", "runtime_origin"),
)
def test_worker_output_rejects_repeated_forged_attestation_values(
    tmp_path: Path, mutation: str
) -> None:
    spec, spec_sha, campaign, args = _worker_output(tmp_path)
    attestation_path = tmp_path / "worker-attestation.json"
    attestation = json.loads(attestation_path.read_text())
    if mutation == "runtime_versions":
        attestation["runtime_versions"] = {"forged": "0"}
    elif mutation == "current_origin":
        attestation["current_module_origins"]["rj45_task_admission"] = "/tmp/evil.py"
    elif mutation == "frozen_inputs":
        attestation["frozen_input_sha256"] = {"/tmp/evil": "0" * 64}
    elif mutation == "held_bank":
        attestation["held_bank_identity"]["selected_artifact_sha256"] = ["0" * 64] * 8
    else:
        evil = tmp_path / "evil.py"
        evil.write_text("# evil\n")
        attestation["runtime_import_origins"] = {"rby1_assembly": str(evil)}
    attestation_path.write_text(json.dumps(attestation))
    report_path = tmp_path / "chunk-report.json"
    report = json.loads(report_path.read_text())
    report["worker_attestation"]["sha256"] = _sha(attestation_path)
    report_path.write_text(json.dumps(report))
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="attestation|origin"):
        COORDINATOR._validate_worker_output(
            tmp_path, spec, spec_sha, args=args, campaign=campaign
        )


def test_post_exit_verifier_timeout_is_recorded_and_fail_closed(tmp_path: Path) -> None:
    result = COORDINATOR._run_post_exit_verifier(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_seconds=0.05,
        cleanup_grace_seconds=0.2,
    )
    assert result["timed_out"] is True
    assert result["process_group_cleanup"]["group_absent_after_cleanup"] is True
    exit_path = tmp_path / "worker-exit.json"
    exit_path.write_text(json.dumps({
        "return_code": 0,
        "timed_out": False,
        "post_exit_verifier_return_code": result["return_code"],
        "post_exit_verifier_timed_out": True,
    }))
    spec, spec_sha, campaign, args = _worker_output(tmp_path / "evidence")
    shutil.copyfile(exit_path, tmp_path / "evidence/worker-exit.json")
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="verifier failed"):
        COORDINATOR._validate_worker_output(
            tmp_path / "evidence", spec, spec_sha, args=args, campaign=campaign
        )


def test_cleanup_targets_only_spawned_process_group_and_descendants(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    script = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid)); "
        "time.sleep(30)"
    )
    process = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    deadline = time.monotonic() + 2.0
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    child_pid = int(child_pid_path.read_text())
    parent_pgrp = os.getpgrp()
    proof = COORDINATOR._cleanup_owned_process_group(process, process.pid, 0.5)
    assert proof["pgid"] == process.pid
    assert proof["pgid"] != parent_pgrp
    assert proof["group_absent_after_cleanup"] is True
    child_stat = Path(f"/proc/{child_pid}/stat")
    if child_stat.exists():
        assert child_stat.read_text().split()[2] == "Z"


def _rc0_parent_with_sleeping_child_command(child_pid_path: Path) -> list[str]:
    script = (
        "import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        f"pathlib.Path({str(child_pid_path)!r}).write_text(str(p.pid))"
    )
    return [sys.executable, "-c", script]


def test_worker_rc0_parent_cleans_residual_owned_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    child_pid_path = tmp_path / "worker-child.pid"
    original_popen = subprocess.Popen

    def spawn_test_worker(_command, **kwargs):
        return original_popen(
            _rc0_parent_with_sleeping_child_command(child_pid_path),
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            start_new_session=kwargs["start_new_session"],
        )

    monkeypatch.setattr(COORDINATOR.subprocess, "Popen", spawn_test_worker)
    monkeypatch.setattr(
        COORDINATOR,
        "_run_post_exit_verifier",
        lambda *_args, **_kwargs: {
            "return_code": 0,
            "timed_out": False,
            "stdout": "",
            "stderr": "",
            "process_group_cleanup": {
                "pgid": 2002,
                "signals": [],
                "group_absent_after_cleanup": True,
            },
        },
    )
    spec_path = tmp_path / "worker-spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    args = SimpleNamespace(
        current_root=ROOT,
        runtime_root=tmp_path / "runtime",
        config=ROOT / "configs/rj45-comparison-development.json",
        playback_actions=PLAYBACK_PATH,
        worker_timeout_seconds=2.0,
        post_exit_verifier_timeout_seconds=2.0,
        process_group_cleanup_grace_seconds=0.5,
    )
    result = COORDINATOR._launch_worker(
        args, spec_path, _sha(spec_path), tmp_path / "worker"
    )
    proof = result["worker_process_group_cleanup"]
    assert result["return_code"] == 0
    assert proof["signals"]
    assert proof["group_absent_after_cleanup"] is True
    assert proof["pgid"] != os.getpgrp()


def test_verifier_rc0_parent_cleans_residual_owned_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "verifier-child.pid"
    result = COORDINATOR._run_post_exit_verifier(
        _rc0_parent_with_sleeping_child_command(child_pid_path),
        timeout_seconds=2.0,
        cleanup_grace_seconds=0.5,
    )
    assert result["return_code"] == 0
    assert result["timed_out"] is False
    assert result["process_group_cleanup"]["signals"]
    assert result["process_group_cleanup"]["group_absent_after_cleanup"] is True


def test_worker_output_rejects_incomplete_process_group_cleanup(tmp_path: Path) -> None:
    spec, spec_sha, campaign, args = _worker_output(tmp_path)
    exit_path = tmp_path / "worker-exit.json"
    exit_record = json.loads(exit_path.read_text())
    exit_record["worker_process_group_cleanup"]["group_absent_after_cleanup"] = False
    exit_path.write_text(json.dumps(exit_record))
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="verifier failed"):
        COORDINATOR._validate_worker_output(
            tmp_path, spec, spec_sha, args=args, campaign=campaign
        )


def test_native_signal_still_invokes_post_exit_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class CrashedProcess:
        pid = 99123
        returncode = -11

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(COORDINATOR.subprocess, "Popen", lambda *args, **kwargs: CrashedProcess())
    monkeypatch.setattr(
        COORDINATOR,
        "_cleanup_owned_process_group",
        lambda *args: {"pgid": 99123, "signals": [], "group_absent_after_cleanup": True},
    )
    verifier_calls = []
    monkeypatch.setattr(
        COORDINATOR,
        "_run_post_exit_verifier",
        lambda command, **kwargs: verifier_calls.append(command) or {
            "return_code": 2,
            "timed_out": False,
            "stdout": "blocked",
            "stderr": "",
            "process_group_cleanup": None,
        },
    )
    args = SimpleNamespace(
        current_root=ROOT,
        runtime_root=tmp_path / "runtime",
        config=ROOT / "configs/rj45-comparison-development.json",
        playback_actions=ROOT / "docs/evidence/rj45-frozen-ablation-20261031/source-actions.json",
        worker_timeout_seconds=1.0,
        post_exit_verifier_timeout_seconds=1.0,
        process_group_cleanup_grace_seconds=0.1,
    )
    spec = tmp_path / "spec.json"
    spec.write_text("{}")
    output = tmp_path / "worker"
    result = COORDINATOR._launch_worker(args, spec, _sha(spec), output)
    assert result["return_code"] == -11
    assert result["post_exit_verifier_return_code"] == 2
    assert len(verifier_calls) == 1


def test_first_failure_stops_following_chunks_and_identity_precedes_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = {
        "schema_name": COORDINATOR.CAMPAIGN_SCHEMA,
        "execution_plan": [
            COORDINATOR._plan_record(index, row)
            for index, row in enumerate(PLAN, start=1)
        ],
        "physics_input_start": _snapshot(),
    }
    prepared = {"plan": PLAN, "config": CONFIG, "playback_actions": [0] * 364}
    monkeypatch.setattr(COORDINATOR, "_prepare_campaign", lambda _args, _runner: (prepared, campaign))
    monkeypatch.setattr(COORDINATOR, "_verify_campaign_inputs", lambda *args: None)
    launches = []

    def launcher(args, spec_path, spec_sha, worker_dir):
        assert (args.output_dir / "campaign-identity.json").is_file()
        launches.append(worker_dir)
        return {}

    monkeypatch.setattr(
        COORDINATOR,
        "_validate_worker_output",
        lambda *args, **kwargs: (_ for _ in ()).throw(COORDINATOR.IsolatedCampaignBlocked("-11")),
    )
    args = SimpleNamespace(
        current_root=ROOT,
        runtime_root=tmp_path / "runtime",
        config=ROOT / "configs/rj45-comparison-development.json",
        playback_actions=ROOT / "docs/evidence/rj45-frozen-ablation-20261031/source-actions.json",
        output_dir=tmp_path / "campaign-output",
        purpose="formal_campaign",
    )
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="-11"):
        COORDINATOR.run_campaign(args, worker_launcher=launcher)
    assert len(launches) == 1
    assert not (args.output_dir / "report.json").exists()


def test_last_chunk_or_adjudication_mutation_blocks_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign = {
        "schema_name": COORDINATOR.CAMPAIGN_SCHEMA,
        "execution_plan": [
            COORDINATOR._plan_record(index, row)
            for index, row in enumerate(PLAN, start=1)
        ],
        "physics_input_start": _snapshot(),
    }
    prepared = {"plan": PLAN, "config": CONFIG, "playback_actions": [0] * 364}
    monkeypatch.setattr(COORDINATOR, "_prepare_campaign", lambda *_args: (prepared, campaign))
    state = {"mutated": False, "checks": 0}

    def verify(*_args):
        state["checks"] += 1
        if state["mutated"]:
            raise COORDINATOR.IsolatedCampaignBlocked("source changed after final chunk")

    monkeypatch.setattr(COORDINATOR, "_verify_campaign_inputs", verify)
    monkeypatch.setattr(
        COORDINATOR,
        "_validate_worker_output",
        lambda _worker, spec, _sha, **_kwargs: [dict(row) for row in spec["rows"]],
    )
    monkeypatch.setattr(COORDINATOR, "_attestation_contract", lambda _worker: {"fixed": True})
    protocol = SimpleNamespace(selection=SimpleNamespace(candidate_id="B"))
    monkeypatch.setattr(
        COORDINATOR,
        "_seal_tuning",
        lambda *_args: (
            protocol,
            {
                "selection": {"candidate_id": "B"},
                "classical_seal": {"selected_candidate_id": "B"},
                "episode_count": 18,
            },
        ),
    )

    def mutate_during_adjudication(*_args):
        state["mutated"] = True
        return protocol, {}, {}

    monkeypatch.setattr(COORDINATOR, "_adjudicate_exact90", mutate_during_adjudication)
    args = SimpleNamespace(
        current_root=ROOT,
        runtime_root=tmp_path / "runtime",
        config=ROOT / "configs/rj45-comparison-development.json",
        playback_actions=PLAYBACK_PATH,
        output_dir=tmp_path / "campaign-output",
        purpose="formal_campaign",
    )
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="after final chunk"):
        COORDINATOR.run_campaign(args, worker_launcher=lambda *_args: {})
    assert state["checks"] == 20
    assert not (args.output_dir / "report.json").exists()


def _tuning_rows() -> list[dict]:
    rows = []
    for candidate in CONFIG["classical_tuning"]["candidates"]:
        payload = RUNNER._classical_payload(
            SimpleNamespace(
                candidate_id=candidate["id"],
                position_gain=candidate["position_gain"],
                rotation_gain=candidate["rotation_gain"],
                force_trigger_n=candidate["force_trigger_n"],
                search_step_m=candidate["search_step_m"],
            )
        )
        for case in CONFIG["bias_cases"]:
            rows.append(
                {
                    "candidate_id": candidate["id"],
                    "case_id": case["case_id"],
                    "success": candidate["id"] == "B",
                    "hard_contact": False,
                    "forbidden_contact": False,
                    "evaluator_force_p95_n": {"A": 4.0, "B": 3.0, "C": 5.0}[candidate["id"]],
                    "completion_time_s": 10.0,
                    "execution_error": None,
                    "terminal_complete": True,
                    "admission_snapshot": {
                        "prefreeze_snapshot_valid": [True],
                        "insert_transition_snapshot_valid": [True],
                        "transition_contract_checked": [True],
                        "actual_applied_action_is_exact_zero": [True],
                    },
                    "actual_classical_controller_payload": payload,
                    "applied_controller_sha256": RUNNER._canonical_json_sha256(payload),
                }
            )
    return rows


def test_tuning_requires_exact18_before_seal_and_binds_candidate() -> None:
    rows = _tuning_rows()
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="exact tuning18"):
        COORDINATOR._seal_tuning(rows[:-1], CONFIG, RUNNER, ADMISSION)
    protocol, seal = COORDINATOR._seal_tuning(rows, CONFIG, RUNNER, ADMISSION)
    assert protocol.selection.candidate_id == "B"
    assert seal["episode_count"] == 18
    assert seal["classical_seal"]["selected_candidate_id"] == "B"


def test_campaign_self_mutation_is_rejected_before_validator_import(tmp_path: Path) -> None:
    campaign_path, campaign, campaign_sha = _campaign(tmp_path)
    campaign["source_sha256"] = {"coordinator": "0" * 64, "runner": "0" * 64}
    campaign_path.write_text(json.dumps(campaign), encoding="utf-8")
    campaign_sha = _sha(campaign_path)
    args = SimpleNamespace(current_root=ROOT, runtime_root=tmp_path / "runtime")
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="source changed"):
        COORDINATOR._verify_campaign_inputs(
            campaign_path, campaign_sha, campaign, args, RUNNER
        )


def test_adjudication_is_not_called_before_exact90() -> None:
    protocol = SimpleNamespace(validate_admission=lambda *args: pytest.fail("called early"))
    with pytest.raises(COORDINATOR.IsolatedCampaignBlocked, match="exact 90"):
        COORDINATOR._adjudicate_exact90(
            [{}] * 89,
            {"config": {}, "playback_actions": []},
            SimpleNamespace(),
            SimpleNamespace(),
            protocol,
        )


def test_adjudication_calls_protocol_only_with_exact90() -> None:
    calls = []

    class Protocol:
        def validate_admission(self, episodes, evidence, baseline):
            calls.append(("validate", len(episodes)))
            return self

        def adjudicate(self, episodes, evidence):
            calls.append(("adjudicate", len(episodes)))
            return self

    class Admission:
        EpisodeResult = staticmethod(lambda **values: values)
        make_baseline_artifact_seal = staticmethod(lambda **values: values)
        validate_baseline_artifact_seal = staticmethod(lambda _seal: True)

    admission_row = {
        "grasp_id": "g",
        "case_id": "c",
        "method_id": "m",
        "slot": 1,
        "success": False,
        "hard_contact": False,
        "forbidden_contact": False,
        "evaluator_force_p95_n": 1.0,
        "completion_time_s": 1.0,
        "provider_inputs": [],
        "applied_controller_sha256": None,
        "applied_artifact_sha256": None,
        "observation_samples": [],
    }
    runner = SimpleNamespace(_evidence=lambda *args, **kwargs: {"valid": True})
    prepared = {"config": CONFIG, "playback_actions": [None] * 364}
    result, _baseline, _evidence = COORDINATOR._adjudicate_exact90(
        ([{}] * 18) + ([admission_row] * 72),
        prepared,
        runner,
        Admission,
        Protocol(),
    )
    assert isinstance(result, Protocol)
    assert calls == [("validate", 72), ("adjudicate", 72)]
