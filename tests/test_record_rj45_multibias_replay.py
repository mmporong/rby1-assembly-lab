from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/record_rj45_multibias_replay.py"
SPEC = importlib.util.spec_from_file_location("record_rj45_multibias_replay_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _rng() -> dict:
    return {
        "seeded": {
            "seed": 20261124, "python": True, "numpy": True, "torch_cpu": True,
            "torch_cuda_all_available_devices": True, "physics_seed_unchanged": 20261128,
        },
        "state_sha256_before_environment_reset": {
            "python": "1" * 64, "numpy": "2" * 64,
            "torch_cpu": "3" * 64, "torch_cuda": "4" * 64,
        },
    }


def _case(case_id: str, base_seed: int, source: str) -> dict:
    return {
        "case_id": case_id, "base_seed": base_seed,
        "calibration_case_seed": base_seed + 46123020,
        "split": "train", "grasp_id": "rj45-train-02", "episode_index": 1,
        "source": source,
    }


def _fixture() -> tuple[dict, dict, dict]:
    cases = [_case("anchor", 20261128, "training-anchor")]
    cases.extend(_case(f"fresh-{index + 1:02d}", 20281208 + 8 * index, "policy-unseen") for index in range(8))
    episodes = []
    for case in cases:
        for method in ("single64", "multi64", "zero"):
            episodes.append({
                "index": len(episodes) + 1, "case": case, "method": method,
                "initial_base79_sha256": "a" * 64,
                "bias": {
                    "base_seed": case["base_seed"], "episode_index": [1],
                    "translation_entry_m": [[0.001, 0.0, 0.0]],
                    "rotation_vector_entry_rad": [[0.0, 0.001, 0.0]],
                },
                "evaluation": {
                    "classification": "valid_success" if method == "multi64" else "valid_failure",
                    "evaluation_rng": _rng(),
                },
            })
    source_hashes = {"runner": "1" * 64, "runtime_manifest": "2" * 64}
    source_binding = {
        **{name: {"path": f"/remote/{name}", "sha256": digest} for name, digest in source_hashes.items()},
        "origin_protocol": {"path": "/remote/origin.json", "sha256": "3" * 64},
    }
    primary = {
        "schema_name": "rj45_multibias_probe_v1", "status": "completed", "phase": "evaluate",
        "contract": {"sha256": "c" * 64}, "episodes": episodes,
        "source_binding": source_binding,
        "bias_cases": [
            {
                "case_id": case["case_id"], "base_seed": case["base_seed"],
                "calibration_case_seed": case["calibration_case_seed"],
                "translation_entry_m": [0.001, 0.0, 0.0],
                "rotation_vector_entry_rad": [0.0, 0.001, 0.0],
            }
            for case in cases
        ],
        "checkpoints": {
            "multi64": {
                "sha256": "d" * 64, "iteration_field": 63, "eligible_update": 64,
                "eligible_for_evaluation": True,
            },
        },
    }
    contract = {
        "helper_sha256": REPLAY.EXPECTED_HELPER_SHA256,
        "budget_and_stop": {"deadline_utc": REPLAY.DEADLINE_UTC},
        "source_sha256": source_hashes,
        "origin_protocol": {"sha256": "3" * 64},
    }
    train = {
        "schema_name": "rj45_multibias_probe_v1", "status": "completed", "phase": "train-multi",
        "contract": {"sha256": "c" * 64}, "source_binding": source_binding,
        "checkpoint": {"sha256": "d" * 64},
    }
    return primary, contract, train


def _validate(mode: str = "multi64", **overrides):
    primary, contract, train = _fixture()
    values = {
        "mode": mode, "primary_report": primary, "primary_report_sha256": "b" * 64,
        "contract": contract, "contract_sha256": "c" * 64,
        "train_report": train, "train_report_sha256": "e" * 64,
        "checkpoint_sha256": "d" * 64 if mode == "multi64" else None,
    }
    values.update(overrides)
    return REPLAY.validate_primary_lineage(**values)


def test_primary_fresh05_lineage_accepts_multi64_and_zero() -> None:
    multi = _validate("multi64")
    zero = _validate("zero")
    assert multi["case"]["base_seed"] == 20281240
    assert multi["classification"] == "valid_success"
    assert zero["classification"] == "valid_failure"
    assert zero["checkpoint_sha256"] is None


def test_wrong_report_bytes_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_bytes(b"changed")
    with pytest.raises(REPLAY.MultibiasReplayError, match="primary report bytes changed"):
        REPLAY.require_file_sha(path, "0" * 64, lambda value: _sha(value.read_bytes()), "primary report")


def test_verified_json_parses_the_same_single_read_bytes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "primary.json"
    approved = json.dumps({"status": "completed"}).encode()
    changed = json.dumps({"status": "invalid"}).encode()
    calls = 0

    def changing_read_bytes(self: Path) -> bytes:
        nonlocal calls
        assert self == path
        calls += 1
        return approved if calls == 1 else changed

    monkeypatch.setattr(Path, "read_bytes", changing_read_bytes)
    result = REPLAY.load_verified_json(path, _sha(approved), "primary report")
    assert result == {"status": "completed"}
    assert calls == 1


def test_checkpoint_is_revalidated_after_policy_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    approved = b"approved checkpoint"
    checkpoint.write_bytes(approved)

    def mutate_during_load() -> str:
        checkpoint.write_bytes(b"changed checkpoint")
        return "policy"

    with pytest.raises(REPLAY.MultibiasReplayError, match="checkpoint after policy load bytes changed"):
        REPLAY.guarded_policy_load(
            checkpoint, _sha(approved), lambda path: _sha(path.read_bytes()), mutate_during_load,
        )


def test_wrong_case_and_incomplete_primary_are_rejected() -> None:
    primary, contract, train = _fixture()
    primary["episodes"] = primary["episodes"][:-1]
    with pytest.raises(REPLAY.MultibiasReplayError, match="27 episodes"):
        _validate(primary_report=primary, contract=contract, train_report=train)
    primary, contract, train = _fixture()
    for episode in primary["episodes"]:
        if episode["method"] == "multi64" and episode["case"]["case_id"] == "fresh-05":
            episode["case"] = _case("fresh-05", 20281248, "policy-unseen")
    with pytest.raises(REPLAY.MultibiasReplayError, match="case identity changed"):
        _validate(primary_report=primary, contract=contract, train_report=train)


def test_checkpoint_hash_and_iter32_are_rejected() -> None:
    with pytest.raises(REPLAY.MultibiasReplayError, match="checkpoint bytes"):
        _validate(checkpoint_sha256="f" * 64)
    payload = {"model_state_dict": {"w": 1}, "optimizer_state_dict": {}, "iter": 31, "infos": None}

    def validator(value, _label):
        if value["iter"] != 63:
            raise RuntimeError("not final")

    with pytest.raises(REPLAY.MultibiasReplayError, match="not final64"):
        REPLAY.validate_checkpoint_payload(
            mode="multi64", payload=payload, checkpoint_sha256="d" * 64,
            expected_sha256="d" * 64, validator=validator,
        )


def test_zero_checkpoint_is_rejected() -> None:
    with pytest.raises(REPLAY.MultibiasReplayError, match="must not receive"):
        _validate("zero", checkpoint_sha256="d" * 64)
    with pytest.raises(REPLAY.MultibiasReplayError, match="must not receive"):
        REPLAY.validate_checkpoint_payload(
            mode="zero", payload={}, checkpoint_sha256="d" * 64,
            expected_sha256="", validator=lambda *_: None,
        )


@pytest.mark.parametrize(
    "first79, classification, expected",
    [("f" * 64, "valid_success", "invalid"), ("a" * 64, "valid_failure", "invalid"), ("a" * 64, "valid_success", "completed")],
)
def test_first79_and_classification_must_match(first79: str, classification: str, expected: str) -> None:
    primary = _validate("multi64")
    status, comparison = REPLAY.compare_replay(primary, first79, classification)
    assert status == expected
    assert comparison["performance_based_rerun_allowed"] is False


def test_expired_deadline_is_rejected() -> None:
    with pytest.raises(REPLAY.MultibiasReplayError, match="expired"):
        REPLAY._require_deadline(REPLAY._deadline_epoch())


def test_first_eligible_observer_uses_cached79_and_calls_step_once() -> None:
    torch = pytest.importorskip("torch")

    class Env:
        def __init__(self):
            self.calls = 0
            self._training_last_policy_observation = torch.zeros((1, 79))

        def step(self, action):
            self.calls += 1
            self._training_last_policy_observation.fill_(1.0)
            return action

    class Ablation:
        @staticmethod
        def _eligible_transition(env):
            return torch.tensor([True])

        @staticmethod
        def raw_base79_sha256(values):
            assert values == [0.0] * 79
            return "a" * 64

    env = Env()
    observer = REPLAY.FirstEligibleObservation(env, Ablation, time.time() + 1000)
    result = env.step("action")
    assert result == "action" and env.calls == 1
    assert observer.sha256 == "a" * 64


def test_preapp_failure_still_writes_invalid_report(tmp_path: Path) -> None:
    report = tmp_path / "invalid-report.json"
    missing = tmp_path / "missing"
    code = REPLAY.main([
        "--mode", "zero", "--runtime-root", str(tmp_path), "--current-root", str(missing),
        "--origin-protocol", str(missing / "origin.json"), "--origin-protocol-sha256", "a" * 64,
        "--contract", str(missing / "contract.json"), "--contract-sha256", "b" * 64,
        "--primary-report", str(missing / "primary.json"), "--primary-report-sha256", "c" * 64,
        "--train-report", str(missing / "train.json"), "--train-report-sha256", "d" * 64,
        "--video", str(tmp_path / "video.mp4"), "--report", str(report),
        "--acknowledge-replay-only",
    ])
    assert code == 1
    payload = json.loads(report.read_text())
    assert payload["status"] == "invalid" and payload["claims"]["benchmark_statistics"] is False


def test_report_is_persisted_before_app_close(tmp_path: Path) -> None:
    report = tmp_path / "replay-report.json"
    program = r'''
import importlib.util, json, os, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("r",sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Multi:
 @staticmethod
 def persist_report_before_app_close(path, report, probe, app, code):
  path.write_text(json.dumps(report));app.close();return code
class App:
 def close(self): os._exit(0)
m.persist_report_before_app_close(Path(sys.argv[2]),{"status":"invalid"},Multi(),object(),App(),1)
raise AssertionError("unreachable")
'''
    completed = subprocess.run([sys.executable, "-c", program, str(SCRIPT), str(report)], capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(report.read_text())["status"] == "invalid"


def test_import_has_no_side_effects(tmp_path: Path) -> None:
    before = set(tmp_path.iterdir())
    spec = importlib.util.spec_from_file_location("record_replay_side_effect", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert set(tmp_path.iterdir()) == before
