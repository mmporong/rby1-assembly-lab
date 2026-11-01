from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_rj45_task_admission.py"
SPEC = importlib.util.spec_from_file_location("run_rj45_task_admission", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)
CONFIG = json.loads(
    (SCRIPT.parents[1] / "configs/rj45-comparison-development.json").read_text(
        encoding="utf-8"
    )
)


def test_episode_optional_observer_brackets_the_same_single_step(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    step_extras = {"terminal_snapshot": {"source": "pre-reset"}}

    class Env:
        device = "cpu"
        extras = {"terminal_snapshot": object()}
        _last_contact_proxy_n = torch.zeros((1, 2))
        _last_insertion_contact_load_n = torch.zeros(1)

        def reset(self):
            return {"policy": torch.zeros((1, 79))}

        def step(self, action):
            assert action.shape == (1, 6)
            return (
                {"policy": torch.ones((1, 79))},
                torch.zeros(1),
                torch.tensor([True]),
                torch.tensor([False]),
                step_extras,
            )

        def close(self):
            return None

    class Provider:
        def reset(self):
            return object()

        def act(self, observation, state):
            return [0.0] * 6, state

    calls = []

    class Observer:
        def capture_pre(self, env, policy, action):
            calls.append(("pre", env, policy.clone(), action.clone()))

        def capture_post(self, env, extras, terminal):
            calls.append(("post", env, extras, terminal))

    terminal = {
        "independent_eval_success": [False],
        "hard_overload": [False],
        "forbidden_contact": [False],
        "timestamp_s": [0.1],
        "task_admission": {},
    }
    monkeypatch.setattr(RUNNER, "_make_env", lambda *_args: (Env(), torch))
    monkeypatch.setattr(RUNNER, "_validate_terminal_snapshot", lambda _value: terminal)
    monkeypatch.setattr(RUNNER, "_reset_isaac_episode_lifecycle", lambda: None)
    runtime = {
        "modules": {
            "rj45_comparison_providers": SimpleNamespace(
                make_fair_provider=lambda *_args, **_kwargs: Provider()
            )
        }
    }
    prepared = {"config": CONFIG}
    result = RUNNER._episode(
        prepared,
        runtime,
        ("diagnostic", "rj45-train-02", "tx-positive", "zero", 1, None),
        None,
        step_observer=Observer(),
    )
    assert [call[0] for call in calls] == ["pre", "post"]
    assert calls[1][2] is step_extras
    assert calls[1][3] is True
    assert len(result["observation_samples"]) == 1


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_source_hashes(root: Path) -> dict[str, str]:
    return {
        relative: _file_sha(root / relative)
        for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS
    }


def _end_verifier_fixture(
    tmp_path: Path, current: Path
) -> tuple[Path, Path, dict, dict[str, str]]:
    runtime = tmp_path / "runtime"
    evidence = tmp_path / "evidence"
    runtime_map = {}
    evidence_map = {}
    for index in range(50):
        runtime_path = runtime / f"inputs/runtime-{index:02d}.bin"
        evidence_path = evidence / f"inputs/evidence-{index:02d}.bin"
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        runtime_path.write_bytes(f"runtime-{index}\n".encode())
        evidence_path.write_bytes(f"evidence-{index}\n".encode())
        runtime_map[str(runtime_path.relative_to(runtime))] = _file_sha(runtime_path)
        evidence_map[str(evidence_path.relative_to(evidence))] = _file_sha(evidence_path)
    normalized = {
        "runtime": runtime_map,
        "evidence": evidence_map,
        "current": _current_source_hashes(current),
    }
    manifest_path = tmp_path / "physics-manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    snapshot = {
        "schema_name": RUNNER.PHYSICS_INPUT_SNAPSHOT_SCHEMA,
        "runtime_root": str(runtime.resolve()),
        "evidence_root": str(evidence.resolve()),
        "current_root": str(current.resolve()),
        "normalized_sha256": normalized,
        "normalized_sha256_canonical_sha256": RUNNER._canonical_json_sha256(
            normalized
        ),
        "manifest": {"path": str(manifest_path), "sha256": _file_sha(manifest_path)},
        "source_manifests": [
            {
                "role": "active_core",
                "path": str(
                    current / "configs/rj45-oracle-runtime-validation-ulp2-v1.json"
                ),
                "sha256": _file_sha(
                    current / "configs/rj45-oracle-runtime-validation-ulp2-v1.json"
                ),
            },
            {
                "role": "base_physics",
                "path": str(
                    current
                    / "configs/rj45-task-admission-physics-inputs-104721a.json"
                ),
                "sha256": _file_sha(
                    current
                    / "configs/rj45-task-admission-physics-inputs-104721a.json"
                ),
            },
            {
                "role": "evidence_core",
                "path": str(evidence / "configs/rj45-oracle-runtime-104721a.json"),
                "sha256": "",
            },
        ],
        "selected_cache_graph_sha256": "b" * 64,
    }
    frozen = {
        str(evidence / relative): digest
        for relative, digest in list(evidence_map.items())[:26]
    }
    evidence_manifest = evidence / "configs/rj45-oracle-runtime-104721a.json"
    evidence_manifest.parent.mkdir(parents=True, exist_ok=True)
    evidence_manifest.write_bytes(
        Path(
            "/opt/homedata/rby1-storage/immutable-inputs/rj45-evidence-104721a/"
            "configs/rj45-oracle-runtime-104721a.json"
        ).read_bytes()
    )
    snapshot["source_manifests"][2]["sha256"] = _file_sha(evidence_manifest)
    return runtime, evidence, snapshot, frozen


def test_full_plan_runs_tuning18_before_admission72() -> None:
    plan = RUNNER.build_execution_plan(CONFIG, smoke_only=False)
    assert len(plan) == 90
    assert all(row[0] == "tuning" for row in plan[:18])
    assert all(row[0] == "admission" for row in plan[18:])
    assert len({(row[1], row[2], row[3], row[4]) for row in plan[18:]}) == 72


def test_smoke_plan_is_fixed_and_cannot_claim_full_admission() -> None:
    assert RUNNER.build_execution_plan(CONFIG, smoke_only=True) == RUNNER.SMOKE_PLAN
    assert len(RUNNER.SMOKE_PLAN) == 3
    assert {row[3] for row in RUNNER.SMOKE_PLAN} == {
        "classical_force_search",
        "zero",
        "oracle_gt_reference",
    }


def test_make_env_forwards_fixed_nonnegative_seed_before_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class Config:
        def __init__(self) -> None:
            self.scene = SimpleNamespace(num_envs=None)
            self.sim = SimpleNamespace(device=None)

    class Environment:
        def __init__(self, cfg) -> None:
            observed["seed_at_construction"] = cfg.seed

    def module(name: str, **attributes: object) -> ModuleType:
        value = ModuleType(name)
        for attribute, item in attributes.items():
            setattr(value, attribute, item)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    module(
        "rby1_assembly.experiment_contracts",
        apply_runtime_timing=lambda _cfg, _runtime: None,
    )
    module("rby1_assembly.isaac_env", Rby1AssemblyEnvCfg=Config)

    def configure_task(cfg, _task, _root):
        observed["seed_at_task_configuration"] = cfg.seed
        return Environment

    module(
        "rby1_assembly.rj45_env",
        configure_rj45_task=configure_task,
    )
    module(
        "rby1_assembly.rj45_training",
        load_training_config=lambda _path: {
            "calibration_bias": {"initial_level": 0}
        },
    )
    module(
        "rby1_assembly.rj45_training_env",
        make_rj45_training_env_type=lambda env_type, *_args, **_kwargs: env_type,
    )

    runtime_root = tmp_path / "runtime"
    (runtime_root / "configs").mkdir(parents=True)
    (runtime_root / "configs/runtime.json").write_text("{}\n", encoding="utf-8")
    (runtime_root / "configs/rj45-training-target-offset-v1.json").write_text(
        "{}\n", encoding="utf-8"
    )
    passthrough = SimpleNamespace(
        make_rj45_comparison_env_type=lambda env_type: env_type,
        make_rj45_task_admission_env_type=lambda env_type, **_kwargs: env_type,
    )
    prepared = {
        "runtime_root": runtime_root,
        "config": {
            "bias_cases": [
                {
                    "case_id": "tx-positive",
                    "translation_bias_entry_m": [0.00012, 0.0, 0.0],
                    "rotation_vector_bias_entry_rad": [0.0, 0.0, 0.0],
                }
            ]
        },
    }
    runtime = {
        "task": {},
        "bank": [object()],
        "proof": {"selected_artifact_sha256": ["a" * 64]},
        "indices": {"rj45-train-02": 0},
        "modules": {
            "rj45_comparison_env": passthrough,
            "rj45_task_admission_env": passthrough,
        },
    }

    env, _torch = RUNNER._make_env(
        prepared,
        runtime,
        ("admission", "rj45-train-02", "tx-positive", "zero", 1, None),
    )

    assert isinstance(env, Environment)
    assert type(RUNNER.TASK_ADMISSION_ENVIRONMENT_SEED) is int
    assert RUNNER.TASK_ADMISSION_ENVIRONMENT_SEED >= 0
    assert observed == {
        "seed_at_task_configuration": RUNNER.TASK_ADMISSION_ENVIRONMENT_SEED,
        "seed_at_construction": RUNNER.TASK_ADMISSION_ENVIRONMENT_SEED,
    }


def test_playback_actions_are_the_fixed_364_row_artifact() -> None:
    path = SCRIPT.parents[1] / RUNNER._parser().get_default("playback_actions")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert RUNNER._sha256(path) == RUNNER.EXPECTED_PLAYBACK_ACTIONS_FILE_SHA256
    assert len(payload["actions"]) == 364
    assert payload["source_trace"]["sha256"] == CONFIG["methods"][2][
        "source_trace_gzip_sha256"
    ]


def test_bank_contract_checks_all_eight_ids_indices_and_hashes() -> None:
    hashes = [f"{index:064x}" for index in range(1, 9)]
    configured = json.loads(json.dumps(CONFIG))
    configured["held_grasps"][0]["artifact_sha256"] = hashes[1]
    configured["held_grasps"][1]["artifact_sha256"] = hashes[4]
    proof = {
        "manifest_sha256": RUNNER.EXPECTED_BANK_SHA256,
        "selected_grasp_ids": list(RUNNER.EXPECTED_IDS),
        "selected_artifact_sha256": hashes,
    }
    assert RUNNER.validate_bank_contract(proof, configured) == {
        "rj45-train-02": 1,
        "rj45-train-05": 4,
        "rj45-train-03": 2,
    }
    proof["selected_grasp_ids"][1], proof["selected_grasp_ids"][2] = (
        proof["selected_grasp_ids"][2],
        proof["selected_grasp_ids"][1],
    )
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="ordered"):
        RUNNER.validate_bank_contract(proof, configured)


def _snapshot(translation=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0)):
    return {
        "prefreeze_snapshot_valid": [True],
        "prefreeze_translation_socket_entry_m_xyz": [list(translation)],
        "prefreeze_shortest_rotation_vector_socket_entry_rad_xyz": [list(rotation)],
        "insert_transition_snapshot_valid": [True],
        "insert_transition_translation_socket_entry_m_xyz": [list(translation)],
        "insert_transition_shortest_rotation_vector_socket_entry_rad_xyz": [list(rotation)],
    }


def test_equivalence_groups_are_fail_closed_at_preregistered_limits() -> None:
    rows = [
        {"grasp_id": "g", "case_id": "c", "admission_snapshot": _snapshot()},
        {
            "grasp_id": "g",
            "case_id": "c",
            "admission_snapshot": _snapshot(translation=(9.0e-6, 0.0, 0.0)),
        },
    ]
    assert RUNNER._group_equivalent(rows, ("grasp_id",), "prefreeze")
    rows[1]["admission_snapshot"] = _snapshot(translation=(11.0e-6, 0.0, 0.0))
    assert not RUNNER._group_equivalent(rows, ("grasp_id",), "prefreeze")


def test_validate_only_writes_nonclaiming_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        RUNNER,
        "_prepare",
        lambda _args: {"plan": RUNNER.SMOKE_PLAN},
    )
    output = tmp_path / "output"
    exit_code = RUNNER.main(
        [
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(output),
            "--validate-only",
            "--smoke-only",
        ]
    )
    assert exit_code == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "validated_not_executed"
    assert report["episode_count"] == 3
    assert report["claims"]["task_admission_completed"] is False


def test_terminal_snapshot_is_strict_and_has_no_missing_value_defaults() -> None:
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="terminal_snapshot"):
        RUNNER._validate_terminal_snapshot(None)
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="independent_eval_valid"):
        RUNNER._validate_terminal_snapshot({"terminal_mask": [True]})


def _complete_row() -> dict[str, object]:
    return {
        "execution_error": None,
        "terminal_complete": True,
        "admission_snapshot": {
            "prefreeze_snapshot_valid": [True],
            "insert_transition_snapshot_valid": [True],
            "transition_contract_checked": [True],
            "actual_applied_action_is_exact_zero": [True],
        },
    }


def _complete_tuning_rows() -> list[dict[str, object]]:
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
        for _case in CONFIG["bias_cases"]:
            rows.append(
                {
                    **_complete_row(),
                    "candidate_id": candidate["id"],
                    "actual_classical_controller_payload": payload,
                    "applied_controller_sha256": RUNNER._canonical_json_sha256(
                        payload
                    ),
                }
            )
    return rows


def test_tuning_error_blocks_before_seal() -> None:
    rows = _complete_tuning_rows()
    rows[7] = {
        **rows[7],
        "execution_error": {"type": "RuntimeError", "message": "failed"},
        "terminal_complete": False,
    }
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="tuning episode 8"):
        RUNNER._validate_tuning_rows(rows, CONFIG)


def test_tuning_rejects_actual_candidate_or_sha_tampering_before_seal() -> None:
    rows = _complete_tuning_rows()
    rows[6] = {
        **rows[6],
        "actual_classical_controller_payload": rows[0][
            "actual_classical_controller_payload"
        ],
        "applied_controller_sha256": "0" * 64,
    }
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="provider identity mismatch"):
        RUNNER._validate_tuning_rows(rows, CONFIG)


def test_tampered_actual_playback_provider_is_rejected() -> None:
    path = SCRIPT.parents[1] / RUNNER._parser().get_default("playback_actions")
    payload = json.loads(path.read_text(encoding="utf-8"))
    tampered = [list(row) for row in payload["actions"]]
    tampered[0][0] += 1.0e-6
    provider = SimpleNamespace(trace=tuple(tuple(row) for row in tampered))
    prepared = {
        "config": CONFIG,
        "playback_actions": payload["actions"],
        "playback_canonical_sha256": payload["canonical_sha256"],
    }
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="differs"):
        RUNNER._attest_provider(provider, "open_loop_playback", prepared)


def test_failing_smoke_cannot_report_smoke_completed() -> None:
    rows = [_complete_row() for _ in RUNNER.SMOKE_PLAN]
    rows[2] = {**rows[2], "terminal_complete": False}
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="smoke episode 3"):
        RUNNER._validate_smoke_rows(rows)


def test_failing_smoke_main_writes_blocked_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = {"plan": RUNNER.SMOKE_PLAN}
    monkeypatch.setattr(RUNNER, "_prepare", lambda _args: prepared)
    monkeypatch.setattr(RUNNER, "_load_runtime", lambda _prepared: {})
    monkeypatch.setattr(RUNNER, "_provenance", lambda _prepared, _runtime: {"source": "test"})
    calls = 0

    def episode(_prepared, _runtime, row, candidate_id, *, heartbeat):
        nonlocal calls
        calls += 1
        complete = calls != 3
        return {
            **_complete_row(),
            "stage": row[0],
            "grasp_id": row[1],
            "case_id": row[2],
            "method_id": row[3],
            "slot": row[4],
            "candidate_id": candidate_id,
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
            "terminal_complete": complete,
        }

    monkeypatch.setattr(RUNNER, "_episode", episode)
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda _options: SimpleNamespace(  # type: ignore[attr-defined]
        app=SimpleNamespace(close=lambda: None)
    )
    package = ModuleType("isaaclab")
    package.app = app_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)
    output = tmp_path / "smoke-failure"
    exit_code = RUNNER.main(
        [
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(output),
            "--smoke-only",
        ]
    )
    assert exit_code == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["claims"]["task_admission_completed"] is False


def test_partial_interrupt_preserves_full_immutable_episode_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = {"plan": RUNNER.SMOKE_PLAN}
    monkeypatch.setattr(RUNNER, "_prepare", lambda _args: prepared)
    monkeypatch.setattr(RUNNER, "_load_runtime", lambda _prepared: {})
    monkeypatch.setattr(RUNNER, "_provenance", lambda _prepared, _runtime: {"source": "test"})
    events: list[str] = []
    original_link = RUNNER.os.link
    original_fsync_directory = RUNNER._fsync_directory
    original_append_jsonl = RUNNER._append_jsonl

    def link(source, destination):
        events.append("publish")
        return original_link(source, destination)

    def fsync_directory(path):
        events.append("directory_fsync")
        return original_fsync_directory(path)

    def append_jsonl(path, payload):
        events.append("progress_jsonl")
        return original_append_jsonl(path, payload)

    monkeypatch.setattr(RUNNER.os, "link", link)
    monkeypatch.setattr(RUNNER, "_fsync_directory", fsync_directory)
    monkeypatch.setattr(RUNNER, "_append_jsonl", append_jsonl)
    calls = 0
    first_row: dict[str, object] = {}

    def episode(_prepared, _runtime, row, candidate_id, *, heartbeat):
        nonlocal calls, first_row
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        events.clear()
        first_row = {
            **_complete_row(),
            "stage": row[0],
            "grasp_id": row[1],
            "case_id": row[2],
            "method_id": row[3],
            "slot": row[4],
            "candidate_id": candidate_id,
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
            "evaluator_force_p95_n": 4.0,
            "completion_time_s": 1.5,
            "provider_inputs": ["observation79", "provider_owned_state"],
            "applied_controller_sha256": "a" * 64,
            "applied_artifact_sha256": None,
            "observation_samples": [[0.0] * 79],
        }
        return first_row

    monkeypatch.setattr(RUNNER, "_episode", episode)
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda _options: SimpleNamespace(  # type: ignore[attr-defined]
        app=SimpleNamespace(close=lambda: None)
    )
    package = ModuleType("isaaclab")
    package.app = app_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    output = tmp_path / "partial-interrupt"
    exit_code = RUNNER.main(
        [
            "--runtime-root", str(tmp_path / "runtime"),
            "--evidence-root", str(tmp_path / "evidence"),
            "--config", str(tmp_path / "config.json"),
            "--output-dir", str(output),
            "--smoke-only",
        ]
    )

    assert exit_code == 2
    persisted_path = output / "episode-rows" / "001.json"
    persisted = json.loads(persisted_path.read_text(encoding="utf-8"))
    assert persisted["ordinal"] == 1
    identity_sha = RUNNER._sha256(output / "run-identity.json")
    assert persisted["run_identity_sha256"] == identity_sha
    assert persisted["source_identity"] == {
        "stage": RUNNER.SMOKE_PLAN[0][0],
        "grasp_id": RUNNER.SMOKE_PLAN[0][1],
        "case_id": RUNNER.SMOKE_PLAN[0][2],
        "method_id": RUNNER.SMOKE_PLAN[0][3],
        "slot": RUNNER.SMOKE_PLAN[0][4],
        "candidate_id": RUNNER.SMOKE_PLAN[0][5],
    }
    assert persisted["episode"] == first_row
    assert not (output / "episode-rows" / "002.json").exists()
    progress_rows = [
        json.loads(line)
        for line in (output / "episodes-progress.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert len(progress_rows) == 1
    assert set(progress_rows[0]) == {
        "stage", "grasp_id", "case_id", "method_id", "slot", "candidate_id",
        "success", "hard_contact", "forbidden_contact", "terminal_complete",
        "execution_error",
    }
    assert events.index("publish") < events.index("directory_fsync") < events.index("progress_jsonl")
    failure_report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert failure_report["run_identity"]["sha256"] == identity_sha
    with pytest.raises(FileExistsError):
        RUNNER._persist_episode_row(
            output,
            1,
            RUNNER.SMOKE_PLAN[0],
            RUNNER.SMOKE_PLAN[0][5],
            first_row,
            failure_report["run_identity"],
        )


def test_episode_persist_failure_is_not_acknowledged_as_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RUNNER, "_prepare", lambda _args: {"plan": RUNNER.SMOKE_PLAN})
    monkeypatch.setattr(RUNNER, "_load_runtime", lambda _prepared: {})
    monkeypatch.setattr(RUNNER, "_provenance", lambda _prepared, _runtime: {"source": "test"})

    def episode(_prepared, _runtime, row, candidate_id, *, heartbeat):
        return {
            **_complete_row(),
            "stage": row[0], "grasp_id": row[1], "case_id": row[2],
            "method_id": row[3], "slot": row[4], "candidate_id": candidate_id,
            "success": False, "hard_contact": False, "forbidden_contact": False,
        }

    monkeypatch.setattr(RUNNER, "_episode", episode)
    monkeypatch.setattr(
        RUNNER,
        "_persist_episode_row",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk write failed")),
    )
    app_module = ModuleType("isaaclab.app")
    app_module.AppLauncher = lambda _options: SimpleNamespace(  # type: ignore[attr-defined]
        app=SimpleNamespace(close=lambda: None)
    )
    package = ModuleType("isaaclab")
    package.app = app_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "isaaclab", package)
    monkeypatch.setitem(sys.modules, "isaaclab.app", app_module)

    output = tmp_path / "persist-failure"
    exit_code = RUNNER.main(
        [
            "--runtime-root", str(tmp_path / "runtime"),
            "--evidence-root", str(tmp_path / "evidence"),
            "--config", str(tmp_path / "config.json"),
            "--output-dir", str(output),
            "--smoke-only",
        ]
    )

    assert exit_code == 2
    assert not (output / "episodes-progress.jsonl").exists()
    progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
    assert progress["completed_episodes"] == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["run_identity"]["sha256"] == RUNNER._sha256(
        output / "run-identity.json"
    )


def test_preflight_error_writes_blocked_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_args):
        raise RUNNER.AdmissionRunBlocked("preflight failed")

    monkeypatch.setattr(RUNNER, "_prepare", fail)
    output = tmp_path / "preflight"
    exit_code = RUNNER.main(
        [
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(output),
            "--validate-only",
        ]
    )
    assert exit_code == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "blocked"
    assert report["requested_process_exit_code"] == 2
    assert report["external_supervisor_contract"]["enforce_wall_clock_timeout"] is True
    assert report["run_identity"] is None


def test_sigint_writes_interrupted_report_and_preserves_exit_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def interrupt(_args):
        raise KeyboardInterrupt()

    monkeypatch.setattr(RUNNER, "_prepare", interrupt)
    output = tmp_path / "interrupted"
    exit_code = RUNNER.main(
        [
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--config",
            str(tmp_path / "config.json"),
            "--output-dir",
            str(output),
            "--validate-only",
        ]
    )
    assert exit_code == 2
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "interrupted"
    assert report["requested_process_exit_code"] == 2
    assert report["claims"]["task_admission_completed"] is False
    assert report["run_identity"] is None


def test_cpu_end_verifier_marks_missing_identity_unverifiable(tmp_path: Path) -> None:
    output = tmp_path / "missing-identity"
    runtime = tmp_path / "runtime"
    evidence = tmp_path / "evidence"
    result = RUNNER.verify_after_child_exit(
        output_dir=output,
        runtime_root=runtime,
        evidence_root=evidence,
        current_root=SCRIPT.parents[1],
    )
    assert result["status"] == "unverifiable_start"
    assert result["scientific_completion_allowed"] is False
    assert (output / "physics-input-end-verification.json").is_file()


def test_cpu_end_verifier_survives_reportless_child_crash_and_blocks_mutation(
    tmp_path: Path,
) -> None:
    current = SCRIPT.parents[1]
    runtime, evidence, snapshot, frozen = _end_verifier_fixture(tmp_path, current)
    matched_output = tmp_path / "matched"
    matched_output.mkdir()
    (matched_output / "run-identity.json").write_text(
        json.dumps(
            {
                "physics_input_start": snapshot,
                "provenance": {"frozen_input_sha256": frozen},
            }
        ),
        encoding="utf-8",
    )
    matched = RUNNER.verify_after_child_exit(
        output_dir=matched_output,
        runtime_root=runtime,
        evidence_root=evidence,
        current_root=current,
    )
    assert matched["status"] == "matched"
    assert matched["scientific_completion_allowed"] is True
    assert not (matched_output / "report.json").exists()
    assert RUNNER.verify_after_child_exit(
        output_dir=matched_output,
        runtime_root=runtime,
        evidence_root=evidence,
        current_root=current,
    )["sha256"] == matched["sha256"]

    runtime_input = runtime / next(iter(snapshot["normalized_sha256"]["runtime"]))
    runtime_input.write_text("mutated\n", encoding="utf-8")
    blocked_output = tmp_path / "blocked"
    blocked_output.mkdir()
    (blocked_output / "run-identity.json").write_text(
        json.dumps(
            {
                "physics_input_start": snapshot,
                "provenance": {"frozen_input_sha256": frozen},
            }
        ),
        encoding="utf-8",
    )
    blocked = RUNNER.verify_after_child_exit(
        output_dir=blocked_output,
        runtime_root=runtime,
        evidence_root=evidence,
        current_root=current,
    )
    assert blocked["status"] == "input_mutation_blocked"
    assert blocked["scientific_completion_allowed"] is False
    assert blocked["changed"] == ["runtime/inputs/runtime-00.bin"]
    assert not (blocked_output / "report.json").exists()


def test_cpu_end_verifier_rejects_changed_validator_before_import(tmp_path: Path) -> None:
    source_root = SCRIPT.parents[1]
    current = tmp_path / "current"
    for relative in RUNNER.EXPECTED_CURRENT_SOURCE_PATHS:
        destination = current / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    for relative in (
        "configs/rj45-oracle-runtime-validation-ulp2-v1.json",
        "configs/rj45-task-admission-physics-inputs-104721a.json",
    ):
        destination = current / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    runtime, evidence, snapshot, frozen = _end_verifier_fixture(tmp_path, current)
    output = tmp_path / "malicious-validator"
    output.mkdir()
    (output / "run-identity.json").write_text(
        json.dumps(
            {
                "physics_input_start": snapshot,
                "provenance": {"frozen_input_sha256": frozen},
            }
        ),
        encoding="utf-8",
    )
    marker = tmp_path / "validator-imported"
    validator = current / RUNNER.END_VALIDATOR_RELATIVE
    validator.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\n"
        "def verify_physics_input_end(*args):\n"
        "    return {'status': 'matched', 'matched': True}\n",
        encoding="utf-8",
    )
    result = RUNNER.verify_after_child_exit(
        output_dir=output,
        runtime_root=runtime,
        evidence_root=evidence,
        current_root=current,
    )
    assert result["status"] == "input_mutation_blocked"
    assert result["scientific_completion_allowed"] is False
    assert "current source changed" in result["error"]["message"]
    assert not marker.exists()


def test_cpu_end_verifier_rejects_copied_base_physics_manifest_tamper(
    tmp_path: Path,
) -> None:
    source_root = SCRIPT.parents[1]
    current = tmp_path / "current"
    for relative in (
        *RUNNER.EXPECTED_CURRENT_SOURCE_PATHS,
        "configs/rj45-oracle-runtime-validation-ulp2-v1.json",
        "configs/rj45-task-admission-physics-inputs-104721a.json",
    ):
        destination = current / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / relative).read_bytes())
    runtime, evidence, snapshot, frozen = _end_verifier_fixture(tmp_path, current)
    base_manifest = current / "configs/rj45-task-admission-physics-inputs-104721a.json"
    base_manifest.write_bytes(base_manifest.read_bytes() + b"\n")
    output = tmp_path / "tampered-base-manifest"
    output.mkdir()
    (output / "run-identity.json").write_text(
        json.dumps(
            {
                "physics_input_start": snapshot,
                "provenance": {"frozen_input_sha256": frozen},
            }
        ),
        encoding="utf-8",
    )
    result = RUNNER.verify_after_child_exit(
        output_dir=output,
        runtime_root=runtime,
        evidence_root=evidence,
        current_root=current,
    )
    assert result["status"] == "input_mutation_blocked"
    assert result["error"]["message"] == (
        "physics input source manifest changed before end check"
    )
    assert result["scientific_completion_allowed"] is False


def test_dual_root_cli_requires_evidence_root(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        RUNNER._parser().parse_args(
            [
                "--runtime-root", str(tmp_path / "runtime"),
                "--config", str(tmp_path / "config.json"),
                "--output-dir", str(tmp_path / "output"),
            ]
        )


def test_actual_active_validator_proves_historical_bank8_without_native_imports() -> None:
    runtime = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf")
    evidence = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-evidence-104721a")
    code = f'''\
import importlib.util, json
from pathlib import Path
p = Path({str(Path(__file__).resolve().parents[1] / "scripts/run_rj45_task_admission.py")!r})
s = importlib.util.spec_from_file_location("fresh_runner", p)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)
print(json.dumps(m._semantic_bank_preflight(Path({str(runtime)!r}), Path({str(evidence)!r}))))
'''
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    proof = json.loads(completed.stdout)
    assert proof["selected_grasp_ids"] == list(RUNNER.EXPECTED_IDS)
    assert proof["manifest_sha256"] == RUNNER.EXPECTED_BANK_SHA256
    assert Path(proof["verifier_origin"]).is_relative_to(runtime / "src")


@pytest.mark.parametrize(
    "value",
    [Path("../physics.json"), Path("configs/other.json"), Path("/tmp/physics.json")],
)
def test_physics_manifest_path_rejects_override_and_escape(
    tmp_path: Path, value: Path
) -> None:
    with pytest.raises(RUNNER.AdmissionRunBlocked, match="fixed project-relative"):
        RUNNER._physics_input_manifest_path(value, tmp_path)


def test_physics_manifest_path_accepts_only_fixed_relative_path(tmp_path: Path) -> None:
    assert RUNNER._physics_input_manifest_path(
        RUNNER.PHYSICS_INPUT_MANIFEST_RELATIVE, tmp_path
    ) == (tmp_path / RUNNER.PHYSICS_INPUT_MANIFEST_RELATIVE).resolve()


def test_episode_lifecycle_stops_timeline_then_creates_fresh_stage() -> None:
    calls: list[str] = []

    class Context:
        @staticmethod
        def instance():
            return None

    class Timeline:
        stopped = False

        def is_stopped(self):
            return self.stopped

        def is_playing(self):
            return not self.stopped

        def stop(self):
            calls.append("timeline.stop")
            self.stopped = True

    class App:
        def update(self):
            calls.append("app.update")

    class Stage:
        current = None

        @classmethod
        def create_new_stage(cls):
            calls.append("stage.create")
            cls.current = object()

        @classmethod
        def update_stage(cls):
            calls.append("stage.update")

        @classmethod
        def get_current_stage(cls):
            return cls.current

    RUNNER._reset_isaac_episode_lifecycle(
        simulation_context_cls=Context,
        timeline=Timeline(),
        app=App(),
        stage_utils=Stage,
    )
    assert calls == [
        "timeline.stop",
        "app.update",
        "stage.create",
        "stage.update",
    ]
