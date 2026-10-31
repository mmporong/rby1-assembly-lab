from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import types

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/record_rj45_oracle_probe.py"
SPEC = importlib.util.spec_from_file_location("record_rj45_oracle_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _report(mode: str, protocol: dict, checkpoint_sha: str | None = None) -> dict:
    learned = mode in {"standard79", "oracle92"}
    outcome = {"classification": "valid_failure", "terminal": {"success": False}}
    report = {
        "status": "completed",
        "mode": mode,
        "protocol": protocol,
        "learning": {"executed": learned},
    }
    if learned:
        report["learning"]["checkpoint"] = {"sha256": checkpoint_sha}
        report["final_evaluation"] = outcome
    else:
        report["evaluation"] = outcome
    return report


def test_learning_lineage_binds_protocol_report_and_checkpoint_bytes() -> None:
    protocol = {"file_sha256": "a" * 64, "payload_sha256": "b" * 64}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "final_checkpoint.pt"
        checkpoint.write_bytes(b"policy")
        checkpoint_sha = REPLAY._load_module(
            Path(__file__).resolve().parents[1] / "scripts/run_rj45_oracle_ppo_probe.py",
            "probe_hash_for_replay_test",
        ).file_sha256(checkpoint)
        report = root / "oracle92-report.json"
        _write(report, _report("oracle92", protocol, checkpoint_sha))

        lineage = REPLAY.validate_probe_lineage(
            mode="oracle92",
            probe_report_path=report,
            protocol_record=protocol,
            checkpoint_path=checkpoint,
            file_sha256=lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest(),
        )

        assert lineage["checkpoint"]["sha256"] == checkpoint_sha
        assert lineage["probe_outcome"]["classification"] == "valid_failure"


def test_learning_lineage_rejects_checkpoint_replacement() -> None:
    protocol = {"file_sha256": "a" * 64, "payload_sha256": "b" * 64}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        checkpoint = root / "final_checkpoint.pt"
        checkpoint.write_bytes(b"replacement")
        report = root / "standard79-report.json"
        _write(report, _report("standard79", protocol, "c" * 64))
        with pytest.raises(REPLAY.ReplayError, match="checkpoint bytes"):
            REPLAY.validate_probe_lineage(
                mode="standard79",
                probe_report_path=report,
                protocol_record=protocol,
                checkpoint_path=checkpoint,
                file_sha256=lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest(),
            )


@pytest.mark.parametrize("mode", ("zero", "reference"))
def test_control_lineage_forbids_checkpoint_and_keeps_probe_outcome(mode: str) -> None:
    protocol = {"file_sha256": "a" * 64, "payload_sha256": "b" * 64}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        report = root / f"{mode}-report.json"
        _write(report, _report(mode, protocol))
        lineage = REPLAY.validate_probe_lineage(
            mode=mode,
            probe_report_path=report,
            protocol_record=protocol,
            checkpoint_path=None,
            file_sha256=lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest(),
        )
        assert lineage["checkpoint"] is None
        assert lineage["probe_outcome"]["terminal"] == {"success": False}

        checkpoint = root / "not-allowed.pt"
        checkpoint.write_bytes(b"unused")
        with pytest.raises(REPLAY.ReplayError, match="must not receive"):
            REPLAY.validate_probe_lineage(
                mode=mode,
                probe_report_path=report,
                protocol_record=protocol,
                checkpoint_path=checkpoint,
                file_sha256=lambda path: __import__("hashlib").sha256(Path(path).read_bytes()).hexdigest(),
            )


def test_cli_requires_explicit_replay_only_acknowledgement() -> None:
    with pytest.raises(SystemExit) as error:
        REPLAY.main(
            [
                "--runtime-root", "/tmp/runtime",
                "--protocol", "/tmp/protocol.json",
                "--protocol-sha256", "a" * 64,
                "--probe-report", "/tmp/report.json",
                "--mode", "zero",
                "--video", "/tmp/replay.mp4",
                "--report", "/tmp/replay.json",
            ]
        )
    assert error.value.code == 2


def test_main_preserves_physics_failure_when_app_close_raises_system_exit_zero(
    monkeypatch,
) -> None:
    closed = []

    class FakeApp:
        def close(self) -> None:
            closed.append(True)
            raise SystemExit(0)

    class FakeAppLauncher:
        def __init__(self, settings) -> None:
            assert settings["enable_cameras"] is True
            self.app = FakeApp()

    fake_probe = types.SimpleNamespace(
        validate_protocol=lambda *_args: {
            "file_sha256": "a" * 64,
            "payload_sha256": "b" * 64,
            "payload": {"diagnostic_scope": {"required_modes": ["zero"]}},
        },
        validate_runtime_manifest=lambda *_args: {},
        file_sha256=lambda _path: "c" * 64,
        _close_simulation_app=lambda app, code: (
            code if _close_with_system_exit_zero(app) else code
        ),
    )

    def _close_with_system_exit_zero(app) -> bool:
        try:
            app.close()
        except SystemExit as error:
            assert error.code == 0
        return True

    isaaclab = types.ModuleType("isaaclab")
    isaaclab_app = types.ModuleType("isaaclab.app")
    isaaclab_app.AppLauncher = FakeAppLauncher
    monkeypatch.setitem(__import__("sys").modules, "isaaclab", isaaclab)
    monkeypatch.setitem(__import__("sys").modules, "isaaclab.app", isaaclab_app)
    monkeypatch.setattr(REPLAY, "_load_module", lambda *_args: fake_probe)
    monkeypatch.setattr(REPLAY, "validate_probe_lineage", lambda **_kwargs: {})
    monkeypatch.setattr(
        REPLAY,
        "_build_render_env",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("physics failed")),
    )

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with pytest.raises(RuntimeError, match="physics failed"):
            REPLAY.main(
                [
                    "--runtime-root", str(root),
                    "--current-root", str(root),
                    "--protocol", str(root / "protocol.json"),
                    "--protocol-sha256", "a" * 64,
                    "--probe-report", str(root / "probe.json"),
                    "--mode", "zero",
                    "--video", str(root / "replay.mp4"),
                    "--report", str(root / "replay.json"),
                    "--acknowledge-replay-only",
                ]
            )
    assert closed == [True]


def test_source_binds_real_step_recorder_and_rgb_render_contract() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert 'env_type(cfg, render_mode="rgb_array")' in source
    assert "cfg.sim.render_interval = RENDER_STRIDE" in source
    assert "cfg.viewer.resolution = VIEWER_RESOLUTION" in source
    assert "with recorder.control_step():" in source
    assert "recorder.finalize_control_step(extras, dones)" in source
    assert '"enable_cameras": True' in source
    assert '"training_result_reproduced": False' in source
    assert '"classification_matches"' in source
