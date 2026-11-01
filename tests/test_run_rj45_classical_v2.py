from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rby1_assembly import rj45_classical_v2_contract as REAL_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_rj45_classical_v2.py"
SPEC = importlib.util.spec_from_file_location("classical_v2_runner_test", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _args(tmp_path, stage="tuning"):
    return [
        "--stage", stage,
        "--runtime-root", str(tmp_path / "absent-runtime"),
        "--evidence-root", str(tmp_path / "absent-evidence"),
        "--output-dir", str(tmp_path / "no-output"),
    ]


def _config(frozen=True):
    value = json.loads(
        (ROOT / "configs/rj45-classical-tuning-v2.json").read_text()
    )
    value["frozen"] = frozen
    return value


def _verified_rows(config):
    binding = REAL_CONTRACT.validate_classical_v2_config(
        config, require_frozen=True
    )
    rows = []
    for slot in REAL_CONTRACT.tuning_execution_plan(config, require_frozen=True):
        rows.append({
            "slot": slot,
            "verification": {
                "verified": True,
                "registered_config_sha256": binding["registered_config_sha256"],
                "classical_source_sha256": binding["classical_source_sha256"],
                "controller_sha256": slot["controller_sha256"],
            },
            "metrics": {
                "success": slot["candidate_id"] == "V2A",
                "hard_contact": False,
                "forbidden_contact": False,
                "evaluator_force_p95_n": float(ord(slot["candidate_id"][-1]) - 64),
                "completion_time_s": 2.0,
            },
        })
    return rows


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, allow_nan=False))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validate_only_tuning_is_cpu_only_and_does_not_write(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *_args(tmp_path), "--validate-only"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["stage"] == "tuning"
    assert payload["frozen"] is False
    assert len(payload["plan"]) == 24
    assert not (tmp_path / "no-output").exists()


def test_unfrozen_draft_cannot_execute(tmp_path):
    draft = tmp_path / "draft.json"
    _write_json(draft, _config(frozen=False))
    with pytest.raises(ValueError, match="not frozen"):
        RUNNER.main([
            *_args(tmp_path), "--config", str(draft), "--ordinal", "1",
            "--expected-config-sha256", "0" * 64,
            "--expected-sources-sha256", "0" * 64,
        ])
    assert not (tmp_path / "no-output").exists()


def test_wrong_registered_config_digest_is_rejected_during_cpu_validation(tmp_path):
    with pytest.raises(ValueError, match="registered config digest differs"):
        RUNNER.main([
            *_args(tmp_path), "--validate-only",
            "--expected-config-sha256", "0" * 64,
        ])


def test_tuning_and_admission_plans_are_exact_and_source_bound(tmp_path):
    config = _config()
    binding, tuning = RUNNER._plan(
        REAL_CONTRACT, config, "tuning", None, None, require_frozen=True
    )
    assert len(tuning) == 24
    assert {row["stage"] for row in tuning} == {"classical_v2_tuning"}
    assert {row["grasp_id"] for row in tuning} == {"rj45-train-03"}
    rows_path = tmp_path / "verified.json"
    rows_sha = _write_json(rows_path, _verified_rows(config))
    admission_binding, admission = RUNNER._plan(
        REAL_CONTRACT, config, "admission", rows_path, rows_sha,
        require_frozen=True,
    )
    assert binding == admission_binding
    assert len(admission) == 12
    assert {row["stage"] for row in admission} == {
        "classical_v2_new_admission_baseline"
    }
    assert {row["candidate_id"] for row in admission} == {"V2A"}
    assert {row["grasp_id"] for row in admission} == {
        "rj45-train-02", "rj45-train-05"
    }


def test_admission_requires_pinned_verified_rows_and_rejects_wrong_digest(tmp_path):
    config = _config()
    with pytest.raises(ValueError, match="requires verified"):
        RUNNER._plan(
            REAL_CONTRACT, config, "admission", None, None,
            require_frozen=True,
        )
    rows_path = tmp_path / "verified.json"
    _write_json(rows_path, _verified_rows(config))
    with pytest.raises(ValueError, match="file SHA differs"):
        RUNNER._plan(
            REAL_CONTRACT, config, "admission", rows_path, "0" * 64,
            require_frozen=True,
        )


def test_source_freeze_digest_detects_change():
    bindings = {"classical_v2_provider": "a" * 64, "verified_tuning_rows": "b" * 64}
    digest = RUNNER._source_digest(bindings)
    assert RUNNER._check_source_freeze(bindings, digest) == digest
    bindings["verified_tuning_rows"] = "c" * 64
    with pytest.raises(ValueError, match="reviewed source freeze"):
        RUNNER._check_source_freeze(bindings, digest)


def _install_cpu_execution(monkeypatch, tmp_path, *, failure=None):
    config = _config()
    config_path = tmp_path / "frozen-classical.json"
    _write_json(config_path, config)
    config_sha = REAL_CONTRACT.validate_classical_v2_config(
        config, require_frozen=True
    )["registered_config_sha256"]
    stats = {
        "launcher_calls": 0,
        "close_calls": 0,
        "report_failures": 0,
        "source_calls": 0,
        "episode_row": None,
    }

    class AdmissionRunBlocked(RuntimeError):
        pass

    class Validator:
        def __init__(self):
            self.calls = 0

        def verify_physics_input_end(self, *_args):
            self.calls += 1
            return {"matched": not (failure == "end_physics" and self.calls == 2)}

    prepared = {
        "runtime_validator": Validator(),
        "physics_input_snapshot": {"sha256": "physics"},
        "runtime_root": tmp_path / "runtime",
        "evidence_root": tmp_path / "evidence",
        "current_root": ROOT,
    }

    def atomic_json(path, payload):
        if failure == "report_write":
            raise OSError("report write failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=False))

    def episode(_prepared, _runtime, row, candidate, **kwargs):
        stats["episode_row"] = row
        if failure == "episode":
            raise AdmissionRunBlocked("episode failed")
        assert "diagnostic_provider_factory" not in kwargs
        assert "diagnostic_environment_seed" not in kwargs
        provider = kwargs["comparison_provider_factory"]()
        attestation = kwargs["comparison_provider_attestor"](provider)
        assert attestation["comparison_only"] is True
        result = {
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
            "evaluator_force_p95_n": 0.5,
            "completion_time_s": 1.0,
            "environment_seed": True if failure == "wrong_seed_type" else 20261121,
            "comparison_only": True,
            "validated_terminal_snapshot": {"independent_eval_success": [False]},
            "controller_payload": attestation["controller_payload"],
            "applied_controller_sha256": attestation["applied_controller_sha256"],
            "provider_source_sha256": attestation["provider_source_sha256"],
            "provider_import_origin": attestation["provider_import_origin"],
            "comparison_provider_steps": [],
            "candidate_id": candidate,
        }
        mutations = {
            "wrong_candidate": lambda: result.update(candidate_id="V2D"),
            "wrong_controller": lambda: result.update(applied_controller_sha256="c" * 64),
            "wrong_source": lambda: result.update(provider_source_sha256="0" * 64),
            "wrong_origin": lambda: result.update(provider_import_origin="/wrong/provider.py"),
            "wrong_payload_candidate": lambda: result["controller_payload"].update(candidate_id="V2D"),
            "wrong_base_helper": lambda: result["controller_payload"].update(base_helper_source_sha256="b" * 64),
        }
        if failure in mutations:
            mutations[failure]()
        return result


    fake_runner = SimpleNamespace(
        AdmissionRunBlocked=AdmissionRunBlocked,
        EXPECTED_RUNTIME_COMMIT="frozen-runtime",
        _prepare=lambda _args: dict(prepared),
        _load_runtime=lambda _prepared: {"task": object()},
        _episode=episode,
        _atomic_json=atomic_json,
    )

    class FakeWriter:
        def __init__(self, *_args):
            pass

    class FakeObserver:
        def __init__(self, *_args, **_kwargs):
            pass

        def finalize(self):
            if failure == "finalize":
                raise RuntimeError("finalize failed")
            return {"schema_name": "fake-sidecar", "row_count": 1}

    fake_trace = SimpleNamespace(
        JsonlTraceWriter=FakeWriter,
        ContactDiagnosticObserver=FakeObserver,
    )

    class FakeProvider:
        def __init__(self, candidate_id):
            self.candidate_id = candidate_id
            self._real = REAL_CONTRACT._classical.ClassicalForceSearchV2Provider(candidate_id)

        def canonical_payload(self):
            return self._real.canonical_payload()

        def controller_sha256(self):
            return self._real.controller_sha256()

    fake_provider = SimpleNamespace(
        ClassicalForceSearchV2Provider=FakeProvider,
        __file__=str(ROOT / "src/rby1_assembly/rj45_classical_v2.py"),
    )
    fake_diagnostic = SimpleNamespace(
        _prepare_args=lambda value: value,
        _source_bindings=lambda _prepared: {"diagnostic": "d" * 64},
        _capture_runtime_start_attestation=lambda *_args: {"started": True},
        _geometry=lambda *_args: object(),
        _diagnostic_snapshot=lambda *_args, **_kwargs: object(),
        _verify_frozen_bank_end=lambda *_args: {"matched": True},
        _verify_runtime_import_origins_end=lambda *_args: {"matched": True},
    )

    def report_failure(_runner, report_path, error):
        stats["report_failures"] += 1
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps({"status": "failed", "error": str(error)}))

    fake_diagnostic._report_failure = report_failure
    stable_bindings = RUNNER._source_bindings(
        prepared, fake_diagnostic, config_path, None
    )
    expected_sources_sha = RUNNER._source_digest(stable_bindings)

    def load_module(name, _path):
        return {
            "rby1_classical_v2_diagnostic_helpers": fake_diagnostic,
            "rby1_classical_v2_contract": REAL_CONTRACT,
            "rby1_classical_v2_admission_runner": fake_runner,
            "rby1_classical_v2_trace": fake_trace,
            "rby1_classical_v2_episode_provider": fake_provider,
        }[name]

    def source_bindings(_prepared, _diagnostic, _config_path, _rows_path):
        stats["source_calls"] += 1
        if failure == "prelaunch_source":
            return {**stable_bindings, "classical_v2_provider": "9" * 64}
        if failure == "end_source" and stats["source_calls"] == 2:
            return {**stable_bindings, "classical_v2_provider": "9" * 64}
        return dict(stable_bindings)

    class FakeApp:
        def close(self):
            stats["close_calls"] += 1

    class FakeAppLauncher:
        def __init__(self, _options):
            stats["launcher_calls"] += 1
            self.app = FakeApp()

    isaaclab = ModuleType("isaaclab")
    isaaclab_app = ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = FakeAppLauncher
    recording = ModuleType("rby1_assembly.recording")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setitem(sys.modules, "rby1_assembly.recording", recording)
    monkeypatch.setattr(RUNNER, "_load_module", load_module)
    monkeypatch.setattr(RUNNER, "_source_bindings", source_bindings)
    monkeypatch.setattr(RUNNER.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(
        RUNNER.faulthandler, "dump_traceback_later", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        RUNNER.faulthandler, "cancel_dump_traceback_later", lambda: None
    )
    output_dir = tmp_path / "classical-output"
    argv = [
        "--stage", "tuning",
        "--runtime-root", str(tmp_path / "runtime"),
        "--evidence-root", str(tmp_path / "evidence"),
        "--current-root", str(ROOT),
        "--output-dir", str(output_dir),
        "--config", str(config_path),
        "--expected-config-sha256", config_sha,
        "--expected-sources-sha256", expected_sources_sha,
        "--ordinal", "1",
    ]
    return argv, output_dir, stats


def test_cpu_lifecycle_success_report_contract_and_close_once(monkeypatch, tmp_path):
    argv, output_dir, stats = _install_cpu_execution(monkeypatch, tmp_path)
    assert RUNNER.main(argv) == 0
    assert stats["launcher_calls"] == 1
    assert stats["close_calls"] == 1
    assert stats["episode_row"][0:4] == (
        "classical_v2_tuning", "rj45-train-03", "tx-positive",
        "classical_force_search_v2",
    )
    report = json.loads((output_dir / "report.json").read_text())
    assert report["schema_name"] == "rj45-classical-v2-report-v1"
    assert report["comparison_only"] is True
    assert report["actual_environment_seed"] == 20261121
    assert report["validated_terminal_snapshot"] == {
        "independent_eval_success": [False]
    }
    assert report["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }
    assert report["classical_v2"]["base_helper_source_sha256"] == REAL_CONTRACT.validate_classical_v2_config(_config())["base_helper_source_sha256"]


@pytest.mark.parametrize(
    "failure", (
        "episode", "finalize", "end_source", "end_physics", "report_write",
        "wrong_seed_type", "wrong_candidate", "wrong_controller", "wrong_source",
        "wrong_origin", "wrong_payload_candidate", "wrong_base_helper",
    )
)
def test_cpu_lifecycle_failures_close_app_once(monkeypatch, tmp_path, failure):
    argv, output_dir, stats = _install_cpu_execution(
        monkeypatch, tmp_path, failure=failure
    )
    assert RUNNER.main(argv) == 2
    assert stats["launcher_calls"] == 1
    assert stats["close_calls"] == 1
    assert stats["report_failures"] == 1
    assert json.loads((output_dir / "report.json").read_text())["status"] == "failed"


def test_prelaunch_source_drift_fails_before_app_launcher(monkeypatch, tmp_path):
    argv, output_dir, stats = _install_cpu_execution(
        monkeypatch, tmp_path, failure="prelaunch_source"
    )
    assert RUNNER.main(argv) == 2
    assert stats["launcher_calls"] == 0
    assert stats["close_calls"] == 0
    assert stats["report_failures"] == 1
    assert json.loads((output_dir / "report.json").read_text())["status"] == "failed"


def test_preexisting_output_is_untouched_and_never_launches(monkeypatch, tmp_path):
    argv, output_dir, stats = _install_cpu_execution(monkeypatch, tmp_path)
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("unchanged")
    assert RUNNER.main(argv) == 2
    assert stats["launcher_calls"] == 0
    assert stats["close_calls"] == 0
    assert stats["report_failures"] == 0
    assert marker.read_text() == "unchanged"
    assert sorted(path.name for path in output_dir.iterdir()) == ["keep.txt"]
