from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rj45_oracle_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("evaluate_rj45_oracle_checkpoint_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RECOVERY)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _protocol(runner_sha: str, deadline: str, seed: int = 20261128) -> dict:
    return {
        "source_sha256": {"runner": runner_sha, "adapter": "a" * 64},
        "fixed_condition": {
            "base_seed": seed,
            "split": "train",
            "grasp_id": "rj45-train-02",
        },
        "ppo": {"updates": 32, "rollout_steps": 128},
        "budget_and_stop": {"absolute_deadline_utc": deadline},
    }


def _write_updates(path: Path) -> None:
    path.write_text(
        "".join(
            json.dumps({"update": index, "metrics": {"loss": 1.0 / index}}) + "\n"
            for index in range(1, 33)
        ),
        encoding="utf-8",
    )


def _contract(**overrides: object) -> dict:
    value = {
        "schema_name": RECOVERY.CONTRACT_SCHEMA,
        "mode": "oracle92",
        "base_seed": 20261128,
        "updates": 32,
        "deadline_utc": "2099-10-31T01:30:00Z",
        "invalid_report_sha256": "a" * 64,
        "original_protocol_sha256": "b" * 64,
        "original_runner_sha256": "c" * 64,
        "checkpoint_sha256": "d" * 64,
        "updates_sha256": "e" * 64,
        "current_protocol_sha256": "f" * 64,
        "recovery_helper_sha256": _sha(SCRIPT),
        "replay_helper_sha256": "1" * 64,
    }
    value.update(overrides)
    return value


def test_contract_is_cpu_only_and_binds_its_file_and_helper_bytes() -> None:
    assert "isaaclab" not in __import__("sys").modules
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "contract.json"
        _write_json(path, _contract())
        observed = RECOVERY.validate_recovery_contract(path, _sha(path), _sha)
        assert observed["updates"] == 32

        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="contract file"):
            RECOVERY.validate_recovery_contract(path, "0" * 64, _sha)

        _write_json(path, _contract(base_seed=20261136))
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="20261128"):
            RECOVERY.validate_recovery_contract(path, _sha(path), _sha)


def test_updates_trace_requires_exact_finite_sequence_1_through_32() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "updates.jsonl"
        _write_updates(path)
        rows = RECOVERY.validate_updates_trace(path, _sha(path), _sha)
        assert [row["update"] for row in rows] == list(range(1, 33))

        lines = path.read_text(encoding="utf-8").splitlines()
        lines[7] = json.dumps({"update": 9, "metrics": {"loss": 0.1}})
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="sequence"):
            RECOVERY.validate_updates_trace(path, _sha(path), _sha)

        _write_updates(path)
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[-1] = json.dumps({"update": 32, "metrics": {"loss": math.nan}})
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="non-finite"):
            RECOVERY.validate_updates_trace(path, _sha(path), _sha)


def test_invalid_origin_binds_only_final_checkpoint_and_full_update_trace() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "final_checkpoint.pt"
        checkpoint.write_bytes(b"final weights")
        updates = root / "updates.jsonl"
        _write_updates(updates)
        report = root / "invalid-report.json"
        payload = {
            "status": "invalid",
            "mode": "oracle92",
            "error": {
                "type": "RuntimeError",
                "message": "Inplace update to inference tensor outside InferenceMode is not allowed",
            },
            "partial_artifacts": [
                {"path": "/remote/final_checkpoint.pt", "sha256": _sha(checkpoint)},
                {"path": "/remote/updates.jsonl", "sha256": _sha(updates)},
            ],
        }
        _write_json(report, payload)
        contract = _contract(
            invalid_report_sha256=_sha(report),
            checkpoint_sha256=_sha(checkpoint),
            updates_sha256=_sha(updates),
        )
        observed = RECOVERY.validate_invalid_origin(
            report_path=report,
            checkpoint_path=checkpoint,
            updates_path=updates,
            contract=contract,
            file_sha256=_sha,
        )
        assert observed["status"] == "invalid"

        middle = root / "diagnostic_checkpoint_0024.pt"
        middle.write_bytes(checkpoint.read_bytes())
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="final_checkpoint"):
            RECOVERY.validate_invalid_origin(
                report_path=report,
                checkpoint_path=middle,
                updates_path=updates,
                contract=contract,
                file_sha256=_sha,
            )


def test_protocol_lineage_allows_only_runner_hash_and_deadline_change() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        original_runner = root / "old_runner.py"
        original_runner.write_bytes(b"old runner")
        old = _protocol(_sha(original_runner), "2026-10-31T01:03:24Z")
        new = _protocol("9" * 64, "2099-10-31T01:30:00Z")
        original_protocol = root / "old-protocol.json"
        _write_json(original_protocol, old)
        contract = _contract(
            original_protocol_sha256=_sha(original_protocol),
            original_runner_sha256=_sha(original_runner),
        )
        current_record = {
            "file_sha256": contract["current_protocol_sha256"],
            "payload_sha256": "1" * 64,
            "payload": new,
        }
        lineage = RECOVERY.validate_protocol_lineage(
            original_protocol_path=original_protocol,
            current_protocol_record=current_record,
            original_runner_source=original_runner,
            contract=contract,
            file_sha256=_sha,
        )
        assert lineage["original"]["runner_source"]["sha256"] == _sha(original_runner)

        changed = copy.deepcopy(new)
        changed["ppo"]["rollout_steps"] = 256
        current_record["payload"] = changed
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="differ outside"):
            RECOVERY.validate_protocol_lineage(
                original_protocol_path=original_protocol,
                current_protocol_record=current_record,
                original_runner_source=original_runner,
                contract=contract,
                file_sha256=_sha,
            )


def test_preimport_source_binding_rejects_replay_helper_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scripts = root / "scripts"
        scripts.mkdir()
        runner = scripts / "run_rj45_oracle_ppo_probe.py"
        replay = scripts / "record_rj45_oracle_probe.py"
        runner.write_bytes(b"runner")
        replay.write_bytes(b"replay")
        protocol = root / "protocol.json"
        _write_json(protocol, {"source_sha256": {"runner": _sha(runner)}})
        contract = _contract(
            current_protocol_sha256=_sha(protocol),
            replay_helper_sha256=_sha(replay),
        )
        binding = RECOVERY.validate_preimport_sources(
            current_root=root,
            protocol_path=protocol,
            contract=contract,
        )
        assert binding["replay_helper"]["sha256"] == _sha(replay)

        replay.write_bytes(b"tampered replay")
        with pytest.raises(RECOVERY.CheckpointRecoveryError, match="replay helper"):
            RECOVERY.validate_preimport_sources(
                current_root=root,
                protocol_path=protocol,
                contract=contract,
            )


def test_report_survives_native_app_close_that_calls_os_exit() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "recovery-report.json"
        program = r'''
import importlib.util
import os
from pathlib import Path
import sys

spec = importlib.util.spec_from_file_location("recovery_subprocess", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class NativeExitApp:
    def close(self):
        os._exit(0)

class Probe:
    @staticmethod
    def _close_simulation_app(app, exit_code):
        app.close()
        return exit_code

module._persist_report_before_app_close(
    report_path=Path(sys.argv[2]),
    report={"status": "completed", "scope": "fresh physics evaluation"},
    probe=Probe,
    simulation_app=NativeExitApp(),
    exit_code=0,
)
raise AssertionError("native close unexpectedly returned")
'''
        completed = subprocess.run(
            [sys.executable, "-c", program, str(SCRIPT), str(report)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert json.loads(report.read_text(encoding="utf-8"))["status"] == "completed"


def test_nonzero_app_close_return_does_not_rewrite_persisted_report() -> None:
    class Probe:
        @staticmethod
        def _close_simulation_app(_app, _exit_code):
            return 7

    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "recovery-report.json"
        returned = RECOVERY._persist_report_before_app_close(
            report_path=report,
            report={"status": "completed"},
            probe=Probe,
            simulation_app=object(),
            exit_code=0,
        )
        assert returned == 7
        assert json.loads(report.read_text(encoding="utf-8")) == {"status": "completed"}


def test_environment_close_does_not_replace_an_existing_failure_stage() -> None:
    class BrokenEnv:
        def close(self):
            raise OSError("secondary close failure")

    original = RuntimeError("physics failure")
    observed = RECOVERY._close_environment_preserving_failure(
        env=BrokenEnv(),
        failure=original,
        failure_traceback=None,
        failure_text="original traceback",
        failure_stage="fresh_physics_evaluation",
        exit_code=1,
    )
    assert observed[0] is original
    assert observed[2] == "original traceback"
    assert observed[3] == "fresh_physics_evaluation"
    assert observed[4] == 1


def test_environment_close_becomes_primary_failure_only_when_none_exists() -> None:
    class BrokenEnv:
        def close(self):
            raise OSError("primary close failure")

    observed = RECOVERY._close_environment_preserving_failure(
        env=BrokenEnv(),
        failure=None,
        failure_traceback=None,
        failure_text="",
        failure_stage="fresh_physics_evaluation",
        exit_code=0,
    )
    assert isinstance(observed[0], OSError)
    assert observed[3] == "environment_close"
    assert observed[4] == 1
