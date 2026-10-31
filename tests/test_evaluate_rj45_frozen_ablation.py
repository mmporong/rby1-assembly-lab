from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_rj45_frozen_ablation.py"
SPEC = importlib.util.spec_from_file_location("frozen_ablation_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
ABLATION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _contract(**changes: object) -> dict:
    cases = [
        {"case_id": f"case-{index}", "base_seed": seed,
         "calibration_case_seed": seed + ABLATION.CALIBRATION_CASE_SEED_OFFSET,
         "split": "train", "grasp_id": "rj45-train-02", "episode_index": 1}
        for index, seed in enumerate(ABLATION.EXPECTED_CASE_SEEDS)
    ]
    source_actions = [[0.1, -0.2, 0.3, -0.1, 0.2, -0.3] for _ in range(4)]
    value = {
        "schema_name": ABLATION.CONTRACT_SCHEMA,
        "helper_sha256": _sha(SCRIPT),
        "source_sha256": {"runner": ABLATION.EXPECTED_RUNNER_SHA256, "replay_helper": "1" * 64, "g2_helper": "2" * 64, "runtime_manifest": "3" * 64},
        "origin": {
            "mode": "standard79", "protocol_sha256": ABLATION.EXPECTED_PROTOCOL_SHA256,
            "report_sha256": "4" * 64, "checkpoint_sha256": ABLATION.EXPECTED_CHECKPOINT_SHA256,
            "updates_trace_sha256": ABLATION.EXPECTED_UPDATES_TRACE_SHA256,
            "source_trace_gzip_sha256": "5" * 64,
            "source_trace_jsonl_sha256": ABLATION.EXPECTED_SOURCE_JSONL_SHA256,
            "updates": 32,
        },
        "cases": cases, "case_order": [case["case_id"] for case in cases],
        "mode_order": list(ABLATION.ABLATION_MODES),
        "public_input_analysis": {
            "source_actions_canonical_sha256": ABLATION.canonical_source_actions_sha256(source_actions),
            "source_action_count": len(source_actions), "source_action_dimension": 6,
            "source_action_mean": source_actions[0].copy(),
            "source_action_last": source_actions[-1].copy(),
        },
        "budget_and_stop": {"budget_cap_usd": 700, "reservation_usd": 6, "deadline_utc": "2099-10-31T01:30:00Z", "helper_wall_time_s": 3900, "per_episode_wall_time_s": 300, "external_stop_wall_time_s": 4500},
        "claims": {"final_split_sealed": True, "dev_split_sealed": True,
                   "training_executed": False,
                   "checkpoint_selection_changed": False,
                   "single_environment_serial": True, "episode_count": 20},
    }
    value.update(changes)
    return value


def test_contract_rejects_seed_and_lineage_tampering() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "contract.json"
        contract = _contract()
        _write_json(path, contract)
        assert ABLATION.validate_contract(path, _sha(path))["claims"]["episode_count"] == 20
        contract["cases"][2]["base_seed"] += 1
        _write_json(path, contract)
        with pytest.raises(ABLATION.FrozenAblationError, match="multiple of 8"):
            ABLATION.validate_contract(path, _sha(path))
        contract = _contract()
        contract["origin"]["checkpoint_sha256"] = "9" * 64
        _write_json(path, contract)
        with pytest.raises(ABLATION.FrozenAblationError, match="checkpoint"):
            ABLATION.validate_contract(path, _sha(path))


def test_source_trace_selects_string_phase_and_active_done_row() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.jsonl.gz"
        rows = []
        contract = _contract()
        source_action = contract["public_input_analysis"]["source_action_mean"]
        source_count = contract["public_input_analysis"]["source_action_count"]
        for index in range(source_count):
            action = [0.0] * 6 + source_action
            rows.append({"phase": "done" if index == source_count - 1 else "insert", "active": True, "policy_input_valid": True, "applied_expanded_action": action})
        rows.append({"event": "complete", "classification": "valid_success"})
        raw = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
        path.write_bytes(gzip.compress(raw, mtime=0))
        contract["origin"]["source_trace_gzip_sha256"] = _sha(path)
        contract["origin"]["source_trace_jsonl_sha256"] = hashlib.sha256(raw).hexdigest()
        actions = ABLATION.load_source_actions(path, contract)
        assert len(actions) == source_count
        assert actions[-1] == source_action

        contract["public_input_analysis"]["source_action_last"][0] += 0.001
        with pytest.raises(ABLATION.FrozenAblationError, match="last row"):
            ABLATION.load_source_actions(path, contract)


def test_canonical_source_actions_binds_order_mean_and_tail() -> None:
    actions = [[0.0] * 6, [1.0] * 6]
    payload = ABLATION.canonical_source_actions_payload(actions)
    assert payload == {"actions": actions, "last": [1.0] * 6, "mean": [0.5] * 6}
    assert ABLATION.canonical_source_actions_sha256(actions) != ABLATION.canonical_source_actions_sha256(list(reversed(actions)))


class _Scalar:
    def __init__(self, value: bool): self.value = value
    def __getitem__(self, _index: int): return self
    def item(self) -> bool: return self.value
    def __and__(self, other): return _Scalar(self.value and other.value)
    def __invert__(self): return _Scalar(not self.value)
    def __gt__(self, value): return _Scalar(int(self.value) > value)


class _Torch:
    float32 = "float32"
    @staticmethod
    def zeros(shape, **_kwargs): return [[0.0] * shape[1]]
    @staticmethod
    def tensor(value, **_kwargs): return value


class _Env:
    device = "cpu"
    _training_config = {}
    def __init__(self):
        self.active = _Scalar(False)
        self.warmup = _Scalar(False)
        self._training_last_policy_observation_valid = _Scalar(True)
        self._training_valid_policy_input_step_count = _Scalar(True)
    def _policy_active_mask(self): return self.active
    def _warmup_pending_mask(self): return self.warmup


def test_constant_and_playback_use_activation_clock_and_hold_tail() -> None:
    env = _Env()
    actions = [[float(index)] * 6 for index in range(3)]
    playback = ABLATION.AblationPolicy(env, "open_loop_playback", actions, _Torch)
    assert playback(None) == [[0.0] * 6]
    assert playback.index == 0
    env.active = _Scalar(True)
    assert playback(None) == [actions[0]]
    assert playback(None) == [actions[1]]
    assert playback(None) == [actions[2]]
    assert playback(None) == [actions[2]]
    assert playback.telemetry() == {"eligible_call_count": 4, "source_rows_consumed": 3, "tail_hold_count": 1}
    constant = ABLATION.AblationPolicy(env, "mean_constant", actions, _Torch)
    assert constant(None) == [[1.0] * 6]

    env._training_last_policy_observation_valid = _Scalar(False)
    assert playback(None) == [[0.0] * 6]
    assert playback.telemetry()["eligible_call_count"] == 4


def test_first_valid_input_is_consumed_before_post_step_counter_increment() -> None:
    env = _Env()
    env.active = _Scalar(True)
    env._training_valid_policy_input_step_count = _Scalar(False)
    playback = ABLATION.AblationPolicy(env, "open_loop_playback", [[0.25] * 6], _Torch)
    assert playback(None) == [[0.25] * 6]
    assert playback.telemetry()["eligible_call_count"] == 1


def test_raw_base79_hash_matches_probe_tensor_contract_and_rejects_mutation() -> None:
    values = [index / 100.0 for index in range(79)]
    expected = hashlib.sha256(
        b"float32\0[1,79]\0" + __import__("struct").pack("<79f", *values)
    ).hexdigest()
    assert ABLATION.raw_base79_sha256(values) == expected
    changed = values.copy()
    changed[-1] += 0.01
    assert ABLATION.raw_base79_sha256(changed) != expected


def test_preimport_checks_run_before_heavy_imports_and_reject_tampering() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("validate_preimport_sources(") < source.index("from isaaclab.app import AppLauncher")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "scripts").mkdir()
        (root / "configs").mkdir()
        paths = {
            "runner": root / "scripts/run_rj45_oracle_ppo_probe.py",
            "replay_helper": root / "scripts/record_rj45_oracle_probe.py",
            "g2_helper": root / "scripts/run_rj45_g2_physics_probe.py",
            "runtime_manifest": root / "configs/rj45-oracle-runtime-104721a.json",
        }
        for name, path in paths.items(): path.write_bytes(name.encode())
        protocol, report, checkpoint, trace = root / "protocol.json", root / "report.json", root / "checkpoint.pt", root / "trace.gz"
        _write_json(protocol, {})
        checkpoint.write_bytes(b"checkpoint")
        trace.write_bytes(b"trace")
        contract = _contract()
        contract["source_sha256"] = {name: _sha(path) for name, path in paths.items()}
        contract["source_sha256"]["runner"] = ABLATION.EXPECTED_RUNNER_SHA256
        # Use injected SHA to preserve the frozen runner identity while exercising all paths.
        runner_path = str(paths["runner"].resolve())
        contract["origin"].update({"protocol_sha256": _sha(protocol), "checkpoint_sha256": _sha(checkpoint), "source_trace_gzip_sha256": _sha(trace)})
        report_payload = {"status": "completed", "mode": "standard79", "learning": {"updates": 32, "updates_trace": {"sha256": ABLATION.EXPECTED_UPDATES_TRACE_SHA256}, "checkpoint": {"sha256": _sha(checkpoint)}}}
        _write_json(report, report_payload)
        contract["origin"]["report_sha256"] = _sha(report)
        fake_sha = lambda path: (ABLATION.EXPECTED_RUNNER_SHA256 if str(Path(path).resolve()) == runner_path else _sha(Path(path)))
        ABLATION.validate_preimport_sources(current_root=root, origin_protocol=protocol, origin_report=report, checkpoint=checkpoint, source_trace=trace, contract=contract, sha=fake_sha)
        paths["replay_helper"].write_bytes(b"tampered")
        with pytest.raises(ABLATION.FrozenAblationError, match="replay_helper"):
            ABLATION.validate_preimport_sources(current_root=root, origin_protocol=protocol, origin_report=report, checkpoint=checkpoint, source_trace=trace, contract=contract, sha=fake_sha)


def test_report_survives_native_app_close() -> None:
    with tempfile.TemporaryDirectory() as directory:
        report = Path(directory) / "report.json"
        program = r'''
import importlib.util, os, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("ablation", sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class Probe:
 @staticmethod
 def _close_simulation_app(app, code): app.close(); return code
class App:
 def close(self): os._exit(0)
m.persist_report_before_app_close(Path(sys.argv[2]), {"status":"completed"}, Probe, App(), 0)
raise AssertionError("unreachable")
'''
        completed = subprocess.run([sys.executable, "-c", program, str(SCRIPT), str(report)], capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr
        assert json.loads(report.read_text())["status"] == "completed"
