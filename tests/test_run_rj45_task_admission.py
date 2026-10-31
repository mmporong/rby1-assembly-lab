from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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
