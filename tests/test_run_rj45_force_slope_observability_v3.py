from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rby1_assembly import rj45_force_slope_observability as REAL_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_rj45_force_slope_observability_v3.py"
SPEC = importlib.util.spec_from_file_location("force_slope_runner_test", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _args(tmp_path):
    return [
        "--runtime-root", str(tmp_path / "absent-runtime"),
        "--evidence-root", str(tmp_path / "absent-evidence"),
        "--output-dir", str(tmp_path / "no-output"),
    ]


def _draft_config():
    return json.loads(
        (ROOT / "configs/rj45-force-slope-observability-v3.json").read_text()
    )


def test_validate_only_is_cpu_only_for_both_stages(tmp_path):
    for stage, count in (("development", 6), ("confirmation", 24)):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), *_args(tmp_path), "--stage", stage, "--validate-only"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload["stage"] == stage
        assert payload["frozen"] is False
        assert len(payload["plan"]) == count
    assert not (tmp_path / "no-output").exists()
    assert not any(name.startswith("isaaclab") for name in sys.modules)


def test_unfrozen_execution_fails_before_simulator_import(tmp_path):
    with pytest.raises(ValueError, match="not frozen"):
        RUNNER.main([*_args(tmp_path), "--stage", "development", "--ordinal", "1"])
    assert not (tmp_path / "no-output").exists()


def test_development_rejects_gate_inputs(tmp_path):
    gate = tmp_path / "gate.json"
    gate.write_text("{}")
    with pytest.raises(ValueError, match="cannot receive"):
        RUNNER.main([
            *_args(tmp_path), "--stage", "development", "--validate-only",
            "--verified-development-results", str(gate),
            "--expected-development-results-sha256", hashlib.sha256(b"{}").hexdigest(),
        ])


def test_confirmation_gate_recomputes_exact_aggregate(tmp_path):
    aggregate = {
        "schema_name": "rj45-force-slope-stage-adjudication-v3",
        "status": "development_pass",
        "expected_sources_sha256": "e" * 64,
        "results": [{"verified": True}],
        "claims": {"task_admission_completed": False, "rl_pilot_admitted": False},
    }
    path = tmp_path / "development.json"
    path.write_text(json.dumps(aggregate))
    analysis = SimpleNamespace(adjudicate_stage=lambda *_args: dict(aggregate))
    result = RUNNER._validate_development_gate(
        analysis, {}, path, hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert result == {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "status": "development_pass",
        "expected_sources_sha256": "e" * 64,
    }


@pytest.mark.parametrize("mutation", ["sha", "recompute", "status", "claims", "shape", "source"])
def test_confirmation_gate_fails_closed(tmp_path, mutation):
    aggregate = {
        "status": "development_pass",
        "expected_sources_sha256": "e" * 64,
        "results": [{"verified": True}],
        "claims": {"task_admission_completed": False, "rl_pilot_admitted": False},
    }
    path = tmp_path / "development.json"
    path.write_text(json.dumps(aggregate))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    recomputed = dict(aggregate)
    if mutation == "sha":
        expected = "0" * 64
    elif mutation == "recompute":
        recomputed = {**aggregate, "episode_count": 6}
    elif mutation == "status":
        aggregate["status"] = "development_fail"
        path.write_text(json.dumps(aggregate))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        recomputed = dict(aggregate)
    elif mutation == "claims":
        aggregate["claims"] = {"task_admission_completed": True, "rl_pilot_admitted": False}
        path.write_text(json.dumps(aggregate))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        recomputed = dict(aggregate)
    elif mutation == "shape":
        aggregate.pop("results")
        path.write_text(json.dumps(aggregate))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
    elif mutation == "source":
        aggregate["expected_sources_sha256"] = "not-a-sha"
        path.write_text(json.dumps(aggregate))
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        recomputed = dict(aggregate)
    analysis = SimpleNamespace(adjudicate_stage=lambda *_args: recomputed)
    with pytest.raises(ValueError):
        RUNNER._validate_development_gate(analysis, {}, path, expected)


def test_source_freeze_detects_any_changed_binding():
    bindings = {"force_slope_runner": "a" * 64, "force_slope_provider": "b" * 64}
    expected = RUNNER._source_digest(bindings)
    assert RUNNER._check_source_freeze(bindings, expected) == expected
    bindings["force_slope_provider"] = "c" * 64
    with pytest.raises(ValueError, match="reviewed source freeze"):
        RUNNER._check_source_freeze(bindings, expected)


def test_source_binding_inventory_includes_all_runtime_consumers(tmp_path):
    prepared = {
        "current_root": ROOT,
        "runtime_root": tmp_path,
        "physics_input_snapshot": {"normalized_sha256": {"current": {}}},
    }
    diagnostic = SimpleNamespace(_source_bindings=lambda _prepared: {"base": "d" * 64})
    bindings = RUNNER._source_bindings(
        prepared,
        diagnostic,
        ROOT / "configs/rj45-force-slope-observability-v3.json",
    )
    assert {
        "force_slope_runner", "task_admission_runner", "diagnostic_helpers",
        "diagnostic_trace", "force_slope_contract", "force_slope_analysis",
        "force_slope_provider", "active_axis_provider", "active_z_base_provider",
        "controlled_v2_contract", "force_slope_config",
    } <= set(bindings)


class _FakeProvider:
    def __init__(self, axis="tx"):
        self.axis = axis

    def canonical_payload(self):
        return {"schema": "fake-force-slope", "axis": self.axis}

    def controller_sha256(self):
        return "c" * 64


def _install_execution(monkeypatch, tmp_path, *, stage="development", failure=None):
    config = _draft_config()
    config["frozen"] = True
    config_path = tmp_path / "frozen-v3.json"
    config_path.write_text(json.dumps(config))
    config_sha256 = REAL_CONTRACT.canonical_sha256(config)
    stats = {"launcher": 0, "close": 0, "source": 0, "failure_report": 0}

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
        "physics_input_snapshot": {"normalized_sha256": {"current": {}}},
        "runtime_root": tmp_path / "runtime",
        "evidence_root": tmp_path / "evidence",
        "current_root": ROOT,
    }

    def atomic_json(path, payload):
        if failure == "report_write":
            raise OSError("report write failed")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, allow_nan=False))

    gate_path_holder = {}

    def episode(_prepared, _runtime, _row, _candidate, **kwargs):
        if failure == "episode":
            raise AdmissionRunBlocked("episode failed")
        provider = kwargs["diagnostic_provider_factory"]()
        attestation = kwargs["diagnostic_provider_attestor"](provider)
        values = {
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
            "completion_time_s": 1.0,
            "environment_seed": kwargs["diagnostic_environment_seed"],
            "controller_payload": attestation["controller_payload"],
            "applied_controller_sha256": attestation["applied_controller_sha256"],
            "provider_source_sha256": attestation["provider_source_sha256"],
            "provider_import_origin": attestation["provider_import_origin"],
            "diagnostic_provider_steps": [],
        }
        if failure in {"payload", "controller", "provider_source", "origin", "seed"}:
            key = {
                "payload": "controller_payload",
                "controller": "applied_controller_sha256",
                "provider_source": "provider_source_sha256",
                "origin": "provider_import_origin",
                "seed": "environment_seed",
            }[failure]
            values[key] = 0 if failure == "seed" else "forged"
        if failure == "gate_mutation":
            gate_path_holder["path"].write_text("{}")
        return values

    fake_runner = SimpleNamespace(
        AdmissionRunBlocked=AdmissionRunBlocked,
        EXPECTED_RUNTIME_COMMIT="runtime-commit",
        _prepare=lambda _args: dict(prepared),
        _load_runtime=lambda _prepared: {"task": object()},
        _episode=episode,
        _atomic_json=atomic_json,
    )

    class Writer:
        def __init__(self, *_args):
            pass

    class Observer:
        def __init__(self, *_args, **_kwargs):
            pass

        def finalize(self):
            if failure == "finalize":
                raise RuntimeError("finalize failed")
            return {"schema_name": "sidecar", "row_count": 1}

    fake_trace = SimpleNamespace(JsonlTraceWriter=Writer, ContactDiagnosticObserver=Observer)
    provider_path = ROOT / "src/rby1_assembly/rj45_force_slope_probe.py"
    fake_force = SimpleNamespace(
        ForceSlopeProbeProvider=_FakeProvider,
        __file__=str(provider_path),
    )
    fake_axis = SimpleNamespace(
        ActiveAxisProbeProvider=lambda _axis: _FakeProvider("tz"),
        __file__=str(ROOT / "src/rby1_assembly/rj45_active_axis_probe.py"),
    )
    fake_diagnostic = SimpleNamespace(
        _prepare_args=lambda value: value,
        _source_bindings=lambda _prepared: {"base": "d" * 64},
        _capture_runtime_start_attestation=lambda *_args: {"started": True},
        _geometry=lambda *_args: object(),
        _diagnostic_snapshot=lambda *_args, **_kwargs: object(),
        _verify_frozen_bank_end=lambda *_args: {"matched": True},
        _verify_runtime_import_origins_end=lambda *_args: {"matched": True},
    )

    def report_failure(_runner, path, error):
        stats["failure_report"] += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"status": "failed", "error": str(error)}))

    fake_diagnostic._report_failure = report_failure
    stable = RUNNER._source_bindings(prepared, fake_diagnostic, config_path)
    sources_sha256 = RUNNER._source_digest(stable)

    gate_aggregate = {
        "schema_name": "rj45-force-slope-stage-adjudication-v3",
        "status": "development_pass",
        "expected_sources_sha256": (
            "f" * 64 if failure == "gate_source" else sources_sha256
        ),
        "results": [{"verified": True}],
        "claims": {"task_admission_completed": False, "rl_pilot_admitted": False},
    }
    fake_analysis = SimpleNamespace(
        adjudicate_stage=lambda *_args: dict(gate_aggregate)
    )

    def load_module(name, _path):
        return {
            "rby1_force_slope_diagnostic_helpers": fake_diagnostic,
            "rby1_force_slope_contract": REAL_CONTRACT,
            "rby1_force_slope_analysis": fake_analysis,
            "rby1_force_slope_task_admission_runner": fake_runner,
            "rby1_force_slope_trace": fake_trace,
            "rby1_force_slope_provider": fake_force,
            "rby1_force_slope_active_axis_provider": fake_axis,
        }[name]

    def source_bindings(*_args):
        stats["source"] += 1
        if failure == "pre_source" or (failure == "end_source" and stats["source"] == 2):
            return {**stable, "force_slope_provider": "9" * 64}
        return dict(stable)

    class App:
        def close(self):
            stats["close"] += 1

    class Launcher:
        def __init__(self, _options):
            stats["launcher"] += 1
            self.app = App()

    isaaclab = ModuleType("isaaclab")
    isaaclab_app = ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = Launcher
    recording = ModuleType("rby1_assembly.recording")
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setitem(sys.modules, "rby1_assembly.recording", recording)
    monkeypatch.setattr(RUNNER, "_load_module", load_module)
    monkeypatch.setattr(RUNNER, "_source_bindings", source_bindings)
    monkeypatch.setattr(RUNNER.faulthandler, "enable", lambda: None)
    monkeypatch.setattr(RUNNER.faulthandler, "dump_traceback_later", lambda *_a, **_k: None)
    monkeypatch.setattr(RUNNER.faulthandler, "cancel_dump_traceback_later", lambda: None)

    output = tmp_path / "output"
    argv = [
        "--stage", stage,
        "--runtime-root", str(tmp_path / "runtime"),
        "--evidence-root", str(tmp_path / "evidence"),
        "--current-root", str(ROOT),
        "--output-dir", str(output),
        "--config", str(config_path),
        "--expected-config-sha256", config_sha256,
        "--expected-sources-sha256", sources_sha256,
        "--ordinal", "1",
    ]
    if stage == "confirmation":
        gate_path = tmp_path / "development-results.json"
        gate_path.write_text(json.dumps(gate_aggregate))
        gate_path_holder["path"] = gate_path
        argv.extend([
            "--verified-development-results", str(gate_path),
            "--expected-development-results-sha256",
            hashlib.sha256(gate_path.read_bytes()).hexdigest(),
        ])
    return argv, output, stats


def test_development_lifecycle_writes_bound_report_and_closes(monkeypatch, tmp_path):
    argv, output, stats = _install_execution(monkeypatch, tmp_path)
    assert RUNNER.main(argv) == 0
    assert stats == {"launcher": 1, "close": 1, "source": 2, "failure_report": 0}
    report = json.loads((output / "report.json").read_text())
    assert report["schema_name"] == "rj45-force-slope-observability-report-v3"
    assert report["stage"] == "development"
    assert report["case"]["grasp_id"] == "rj45-train-03"
    assert report["actual_environment_seed"] == report["case"]["environment_seed"]
    assert report["provenance"]["verified_development_results_sha256"] is None
    assert report["claims"] == {
        "task_admission_completed": False,
        "rl_pilot_admitted": False,
    }


def test_confirmation_report_pins_external_development_gate(monkeypatch, tmp_path):
    argv, output, stats = _install_execution(
        monkeypatch, tmp_path, stage="confirmation"
    )
    assert RUNNER.main(argv) == 0
    report = json.loads((output / "report.json").read_text())
    gate_path = tmp_path / "development-results.json"
    assert report["stage"] == "confirmation"
    assert report["case"]["grasp_id"] == "rj45-train-02"
    assert report["provenance"]["verified_development_results_sha256"] == hashlib.sha256(
        gate_path.read_bytes()
    ).hexdigest()
    assert stats["launcher"] == stats["close"] == 1


def test_confirmation_rejects_development_with_different_source_freeze_before_launch(monkeypatch, tmp_path):
    argv, output, stats = _install_execution(
        monkeypatch, tmp_path, stage="confirmation", failure="gate_source"
    )
    assert RUNNER.main(argv) == 2
    assert stats["launcher"] == stats["close"] == 0
    assert json.loads((output / "report.json").read_text())["status"] == "failed"


def test_confirmation_detects_gate_file_changed_during_execution(monkeypatch, tmp_path):
    argv, output, stats = _install_execution(
        monkeypatch, tmp_path, stage="confirmation", failure="gate_mutation"
    )
    assert RUNNER.main(argv) == 2
    assert stats["launcher"] == stats["close"] == 1
    assert json.loads((output / "report.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "failure",
    [
        "episode", "finalize", "end_source", "end_physics", "report_write",
        "payload", "controller", "provider_source", "origin", "seed",
    ],
)
def test_failures_and_provider_forgeries_close_app(monkeypatch, tmp_path, failure):
    argv, output, stats = _install_execution(monkeypatch, tmp_path, failure=failure)
    assert RUNNER.main(argv) == 2
    assert stats["launcher"] == 1
    assert stats["close"] == 1
    assert stats["failure_report"] == 1
    assert json.loads((output / "report.json").read_text())["status"] == "failed"


def test_prelaunch_source_mismatch_never_launches(monkeypatch, tmp_path):
    argv, output, stats = _install_execution(monkeypatch, tmp_path, failure="pre_source")
    assert RUNNER.main(argv) == 2
    assert stats["launcher"] == 0
    assert stats["close"] == 0
    assert json.loads((output / "report.json").read_text())["status"] == "failed"


def test_provider_attestation_rejects_mutated_payload_before_episode():
    binding = {
        "path": ROOT / "src/rby1_assembly/rj45_force_slope_probe.py",
        "module": SimpleNamespace(
            __file__=str(ROOT / "src/rby1_assembly/rj45_force_slope_probe.py")
        ),
        "controller_payload": {"schema": "expected"},
        "controller_sha256": "c" * 64,
        "provider_source_sha256": hashlib.sha256(
            (ROOT / "src/rby1_assembly/rj45_force_slope_probe.py").read_bytes()
        ).hexdigest(),
        "provider_import_origin": str(
            (ROOT / "src/rby1_assembly/rj45_force_slope_probe.py").resolve()
        ),
    }
    with pytest.raises(ValueError, match="reviewed binding"):
        RUNNER._attest_provider(_FakeProvider(), binding)
