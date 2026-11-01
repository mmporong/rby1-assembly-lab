import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from rby1_assembly import rj45_controlled_observability as REAL_CONTRACT


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_rj45_controlled_observability_v2.py"
SPEC = importlib.util.spec_from_file_location("controlled_runner_test", SCRIPT)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def args(tmp_path):
    return ["--runtime-root", str(tmp_path / "absent-runtime"),
            "--evidence-root", str(tmp_path / "absent-evidence"),
            "--output-dir", str(tmp_path / "no-output")]


def test_validate_only_is_cpu_only_and_does_not_write(tmp_path):
    result = subprocess.run([sys.executable, str(SCRIPT), *args(tmp_path), "--validate-only"],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["frozen"] == json.loads((ROOT / "configs/rj45-controlled-observability-v2.json").read_text())["frozen"]
    assert len(payload["plan"]) == 24
    assert not (tmp_path / "no-output").exists()


def test_unfrozen_config_cannot_execute_or_import_simulator(tmp_path):
    config = json.loads((ROOT / "configs/rj45-controlled-observability-v2.json").read_text())
    config["frozen"] = False
    draft = tmp_path / "draft.json"
    draft.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="not frozen"):
        RUNNER.main([*args(tmp_path), "--ordinal", "1", "--config", str(draft)])
    assert not (tmp_path / "no-output").exists()


def test_wrong_digest_is_rejected_even_for_preflight(tmp_path):
    with pytest.raises(ValueError, match="digest"):
        RUNNER.main([*args(tmp_path), "--validate-only", "--expected-config-sha256", "0" * 64])


def test_frozen_execution_requires_explicit_digest_and_ordinal(tmp_path):
    config = json.loads((ROOT / "configs/rj45-controlled-observability-v2.json").read_text())
    config["frozen"] = True
    config_path = tmp_path / "frozen.json"
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="expected config SHA and ordinal"):
        RUNNER.main([*args(tmp_path), "--config", str(config_path)])
    assert not (tmp_path / "no-output").exists()


def test_source_freeze_rejects_changed_provider_before_simulation():
    bindings = {"active_axis_provider": "a" * 64, "controlled_contract": "b" * 64}
    digest = RUNNER._source_digest(bindings)
    assert RUNNER._check_source_freeze(bindings, digest) == digest
    bindings["active_axis_provider"] = "c" * 64
    with pytest.raises(ValueError, match="reviewed source freeze"):
        RUNNER._check_source_freeze(bindings, digest)


def _install_cpu_execution(monkeypatch, tmp_path, *, failure=None):
    config = json.loads((ROOT / "configs/rj45-controlled-observability-v2.json").read_text())
    config["frozen"] = True
    config_path = tmp_path / "frozen-controlled.json"
    config_path.write_text(json.dumps(config))
    config_sha256 = REAL_CONTRACT.canonical_sha256(config)
    stats = {
        "launcher_calls": 0,
        "close_calls": 0,
        "report_failures": 0,
        "source_calls": 0,
    }

    class AdmissionRunBlocked(RuntimeError):
        pass

    class Validator:
        def __init__(self):
            self.calls = 0

        def verify_physics_input_end(self, *_args):
            self.calls += 1
            return {"matched": not (failure == "end_physics" and self.calls == 2)}

    validator = Validator()
    prepared = {
        "runtime_validator": validator,
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

    def episode(_prepared, _runtime, _row, _candidate, **kwargs):
        if failure == "episode":
            raise AdmissionRunBlocked("episode failed")
        seed = kwargs["diagnostic_environment_seed"]
        provider = kwargs["diagnostic_provider_factory"]()
        attestation = kwargs["diagnostic_provider_attestor"](provider)
        return {
            "success": False,
            "hard_contact": False,
            "forbidden_contact": False,
            "completion_time_s": 1.0,
            "environment_seed": seed,
            "controller_payload": attestation["controller_payload"],
            "applied_controller_sha256": attestation["applied_controller_sha256"],
            "provider_source_sha256": attestation["provider_source_sha256"],
            "provider_import_origin": attestation["provider_import_origin"],
            "diagnostic_provider_steps": [],
        }

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
        def __init__(self, axis):
            self.axis = axis

        def canonical_payload(self):
            return {"schema": "fake-axis-v2", "axis": self.axis}

        def controller_sha256(self):
            return "c" * 64

    fake_provider_module = SimpleNamespace(
        ActiveAxisProbeProvider=FakeProvider,
        __file__=str(ROOT / "src/rby1_assembly/rj45_active_axis_probe.py"),
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
    stable_bindings = RUNNER._source_bindings(prepared, fake_diagnostic, config_path)
    expected_sources_sha256 = RUNNER._source_digest(stable_bindings)

    def load_module(name, _path):
        return {
            "rby1_controlled_diagnostic_helpers": fake_diagnostic,
            "rby1_controlled_contract": REAL_CONTRACT,
            "rby1_contact_diagnostic_admission_runner": fake_runner,
            "rby1_contact_diagnostic_trace": fake_trace,
            "rby1_controlled_axis_provider": fake_provider_module,
        }[name]

    def source_bindings(_prepared, _diagnostic, _config_path):
        stats["source_calls"] += 1
        if failure == "prelaunch_source":
            return {**stable_bindings, "active_axis_provider": "9" * 64}
        if failure == "end_source" and stats["source_calls"] == 2:
            return {**stable_bindings, "active_axis_provider": "9" * 64}
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
    monkeypatch.setattr(RUNNER.faulthandler, "dump_traceback_later", lambda *_a, **_k: None)
    monkeypatch.setattr(RUNNER.faulthandler, "cancel_dump_traceback_later", lambda: None)

    output_dir = tmp_path / "controlled-output"
    argv = [
        "--runtime-root", str(tmp_path / "runtime"),
        "--evidence-root", str(tmp_path / "evidence"),
        "--current-root", str(ROOT),
        "--output-dir", str(output_dir),
        "--config", str(config_path),
        "--expected-config-sha256", config_sha256,
        "--expected-sources-sha256", expected_sources_sha256,
        "--ordinal", "1",
    ]
    return argv, output_dir, stats


def test_cpu_lifecycle_success_closes_app_exactly_once(monkeypatch, tmp_path):
    argv, output_dir, stats = _install_cpu_execution(monkeypatch, tmp_path)
    assert RUNNER.main(argv) == 0
    assert stats["launcher_calls"] == 1
    assert stats["close_calls"] == 1
    assert stats["report_failures"] == 0
    report = json.loads((output_dir / "report.json").read_text())
    assert report["actual_environment_seed"] == report["case"]["environment_seed"]


@pytest.mark.parametrize(
    "failure",
    ("episode", "finalize", "end_source", "end_physics", "report_write"),
)
def test_cpu_lifecycle_failures_close_app_exactly_once(monkeypatch, tmp_path, failure):
    argv, output_dir, stats = _install_cpu_execution(
        monkeypatch, tmp_path, failure=failure
    )
    assert RUNNER.main(argv) == 2
    assert stats["launcher_calls"] == 1
    assert stats["close_calls"] == 1
    assert stats["report_failures"] == 1
    assert json.loads((output_dir / "report.json").read_text())["status"] == "failed"


def test_prelaunch_source_drift_fails_on_cpu_before_app_launcher(monkeypatch, tmp_path):
    argv, output_dir, stats = _install_cpu_execution(
        monkeypatch, tmp_path, failure="prelaunch_source"
    )
    assert RUNNER.main(argv) == 2
    assert stats["launcher_calls"] == 0
    assert stats["close_calls"] == 0
    assert stats["report_failures"] == 1
    assert json.loads((output_dir / "report.json").read_text())["status"] == "failed"


def test_preexisting_output_is_untouched_and_never_launches_app(monkeypatch, tmp_path):
    argv, output_dir, stats = _install_cpu_execution(monkeypatch, tmp_path)
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("unchanged")
    before = sorted(path.name for path in output_dir.iterdir())
    assert RUNNER.main(argv) == 2
    assert stats["launcher_calls"] == 0
    assert stats["close_calls"] == 0
    assert stats["report_failures"] == 0
    assert marker.read_text() == "unchanged"
    assert sorted(path.name for path in output_dir.iterdir()) == before
