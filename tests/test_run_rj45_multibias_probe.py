from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_rj45_multibias_probe.py"
SPEC = importlib.util.spec_from_file_location("run_rj45_multibias_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)
ORIGIN = ROOT / "docs/evidence/rj45-oracle-condition-20261031/protocol-20261128.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _budget() -> dict:
    return {
        "budget_cap_usd": 700,
        "new_run_reservation_usd": 15,
        "external_guard_minutes": 180,
        "private_evidence_sha256": "a" * 64,
        "locally_verified": True,
    }


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_contract_binds_sources_schedule_and_final_only_rule() -> None:
    payload = PROBE.contract_payload(ROOT, ORIGIN, "2099-10-31T01:00:00Z", _budget())
    assert payload["training"]["single_pool"] == [20261128]
    assert payload["training"]["multi_pool"] == [20261128, 20271208, 20271216, 20271224]
    assert payload["training"]["transitions"] == 8192
    assert payload["training"]["eligible_checkpoint_updates"] == [64]
    assert payload["training"]["diagnostic_checkpoint_updates"] == [32]
    assert len(payload["evaluation"]["cases"]) == 9
    assert payload["evaluation"]["episode_count"] == 27
    assert str(PROBE.HELD_BANK_RELATIVE_PATH) == "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json"


def test_contract_rejects_tampering_and_source_binding_is_exact() -> None:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "contract.json"
        payload = PROBE.contract_payload(ROOT, ORIGIN, "2099-10-31T01:00:00Z", _budget())
        _write(path, payload)
        assert PROBE.validate_contract(path, _sha(path), current_root=ROOT, origin_protocol=ORIGIN)["training"]["updates"] == 64
        payload["training"]["updates"] = 63
        _write(path, payload)
        with pytest.raises(PROBE.MultibiasProbeError, match="contract payload changed"):
            PROBE.validate_contract(path, _sha(path), current_root=ROOT, origin_protocol=ORIGIN)
    observed = PROBE.validate_preimport_sources(ROOT, ORIGIN, PROBE.contract_payload(ROOT, ORIGIN, "2099-10-31T01:00:00Z", _budget()))
    assert observed["runner"]["sha256"] == PROBE.EXPECTED_SOURCE_SHA256["runner"]
    assert len(observed) == 7


class _Tensor:
    def __init__(self, value): self.value = value
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.value


class _ResetEnv:
    step_calls: int
    step: Any
    _training_last_policy_observation: Any

    def __init__(self, expected):
        self.calls = 0
        self.step_calls = 0
        self.step = None
        self._training_last_policy_observation = None
        self.expected = expected
        self._training_bias_base_seed = 0
        self._training_episode_index = _Tensor([0])
        self._training_grasp_index = _Tensor([0])
        self._training_translation_bias_entry = _Tensor([[0.0, 0.0, 0.0]])
        self._training_rotation_bias_entry = _Tensor([[0.0, 0.0, 0.0]])

    def _reset_idx(self, env_ids=None):
        self.calls += 1
        case = self.expected[self._training_bias_base_seed]
        self._training_episode_index = _Tensor([1])
        self._training_grasp_index = _Tensor([1])
        self._training_translation_bias_entry = _Tensor([case["translation_entry_m"]])
        self._training_rotation_bias_entry = _Tensor([case["rotation_vector_entry_rad"]])
        return env_ids


def _schedule_fixture(tmp_path: Path, pool_size: int = 2):
    cases = [PROBE._case(f"c{i}", 8 * (i + 1), source="test") for i in range(pool_size)]
    expected = {
        case["case_id"]: {
            "translation_entry_m": [case["base_seed"] / 100.0, 0.0, 0.0],
            "rotation_vector_entry_rad": [0.0, case["base_seed"] / 100.0, 0.0],
        }
        for case in cases
    }
    by_seed = {case["base_seed"]: expected[case["case_id"]] for case in cases}
    by_seed[0] = {"translation_entry_m": [0.0] * 3, "rotation_vector_entry_rad": [0.0] * 3}
    env = _ResetEnv(by_seed)
    schedule = PROBE.ResetSchedule(env, cases, expected, tmp_path / "resets.jsonl")
    return env, schedule, cases


def test_reset_schedule_passes_through_inactive_constructor_reset(tmp_path: Path) -> None:
    env, schedule, _ = _schedule_fixture(tmp_path)
    env._reset_idx(None)
    assert env.calls == 1
    assert schedule.reset_count == 0
    assert schedule.next_index == 0


def test_explicit_first_training_reset_consumes_anchor_then_rolls_over(tmp_path: Path) -> None:
    env, schedule, cases = _schedule_fixture(tmp_path)
    schedule.activate()
    env._reset_idx(None)
    assert env._training_bias_base_seed == cases[0]["base_seed"]
    assert schedule.reset_count == 1 and schedule.next_index == 1
    env._reset_idx(None)
    assert env._training_bias_base_seed == cases[1]["base_seed"]
    env._reset_idx(None)
    assert env._training_bias_base_seed == cases[0]["base_seed"]
    records = [json.loads(line) for line in (tmp_path / "resets.jsonl").read_text().splitlines()]
    assert [row["pool_index"] for row in records] == [0, 1, 0]


def test_schedule_deactivation_restores_complete_passthrough(tmp_path: Path) -> None:
    env, schedule, _ = _schedule_fixture(tmp_path)
    schedule.activate(); env._reset_idx(None); schedule.deactivate()
    calls, resets = env.calls, schedule.reset_count
    env._reset_idx(None)
    assert env.calls == calls + 1
    assert schedule.reset_count == resets


def test_reset_bias_mismatch_fails_closed(tmp_path: Path) -> None:
    env, schedule, cases = _schedule_fixture(tmp_path)
    env.expected[cases[0]["base_seed"]] = {
        **env.expected[cases[0]["base_seed"]],
        "translation_entry_m": [9.0, 0.0, 0.0],
    }
    schedule.activate()
    with pytest.raises(PROBE.MultibiasProbeError, match="bias differs"):
        env._reset_idx(None)


def test_training_step_audit_preserves_previous_case_across_auto_reset_and_calls_once(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    env, schedule, cases = _schedule_fixture(tmp_path)
    schedule.activate(); env._reset_idx(None)
    env._training_last_policy_observation = torch.zeros((1, 79))
    assert not hasattr(env, "get_observations")
    env.step_calls = 0

    def original_step(action):
        env.step_calls += 1
        previous_seed = env._training_bias_base_seed
        env._training_last_policy_observation.fill_(1.0)
        env._reset_idx(None)
        saved = {"previous_seed": previous_seed, "policy_input_observation": torch.zeros((1, 79))}
        return ({"policy": torch.zeros((1, 79)), "critic": torch.zeros((1, 92))}, torch.tensor([1.0]), torch.tensor([True]), torch.tensor([False]), {"terminal_snapshot": saved})

    env.step = original_step

    class G2:
        @staticmethod
        def _transition_contract(saved, observation, done):
            assert done and tuple(observation.shape) == (1, 79)
            assert torch.equal(observation, saved["policy_input_observation"])
            return {"valid_input": True, "active": True, "applied_expanded": [0.0] * 12}

    audit = PROBE.TrainingStepAudit(env, schedule, G2, tmp_path / "transitions.jsonl", lambda _value: "a" * 64)
    audit.step(torch.zeros((1, 6)))
    audit.close()
    assert env.step_calls == 1
    assert audit.by_case[cases[0]["case_id"]]["transitions"] == 1
    assert env._training_bias_base_seed == cases[1]["base_seed"]


def test_invalid_input_is_counted_as_transition_but_not_active_exposure(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    env, schedule, cases = _schedule_fixture(tmp_path, pool_size=1)
    schedule.activate(); env._reset_idx(None)
    env._training_last_policy_observation = torch.zeros((1, 79))
    env.step = lambda action: ({}, torch.tensor([0.0]), torch.tensor([False]), torch.tensor([False]), {"terminal_snapshot": {}})

    class G2:
        @staticmethod
        def _transition_contract(saved, observation, done):
            return {"valid_input": False, "active": False, "applied_expanded": [0.0] * 12}

    audit = PROBE.TrainingStepAudit(env, schedule, G2, tmp_path / "transitions.jsonl", lambda _value: "a" * 64)
    audit.step(torch.zeros((1, 6))); audit.close()
    assert audit.total == 1
    assert audit.by_case[cases[0]["case_id"]]["active_valid"] == 0


@pytest.mark.parametrize(
    "cached, message",
    [
        (lambda torch: torch.zeros((1, 78)), "finite shape"),
        (lambda torch: torch.full((1, 79), float("nan")), "finite shape"),
    ],
)
def test_training_step_audit_rejects_invalid_cached_policy_input_before_step(
    tmp_path: Path, cached: Any, message: str
) -> None:
    torch = pytest.importorskip("torch")
    env, schedule, _ = _schedule_fixture(tmp_path, pool_size=1)
    schedule.activate(); env._reset_idx(None)
    env._training_last_policy_observation = cached(torch)
    env.step = lambda action: pytest.fail("underlying step must not run")
    audit = PROBE.TrainingStepAudit(env, schedule, object(), tmp_path / "transitions.jsonl")
    with pytest.raises(PROBE.MultibiasProbeError, match=message):
        audit.step(torch.zeros((1, 6)))
    audit.close()


def test_training_total_and_pool_exposure_gate(tmp_path: Path) -> None:
    env, schedule, cases = _schedule_fixture(tmp_path)
    audit = object.__new__(PROBE.TrainingStepAudit)
    audit.total = PROBE.TRANSITIONS
    audit.by_case = {case["case_id"]: {"transitions": PROBE.TRANSITIONS // 2, "active_valid": 1, "terminal": 1} for case in cases}
    audit.first_active_base79 = {}
    assert audit.validate()["total_transitions"] == 8192
    audit.by_case[cases[1]["case_id"]]["active_valid"] = 0
    with pytest.raises(PROBE.MultibiasProbeError, match="no active valid exposure"):
        audit.validate()


def test_only_final64_checkpoint_payload_is_eligible() -> None:
    payload = {
        "model_state_dict": {"actor.weight": object()},
        "optimizer_state_dict": {}, "iter": 63, "infos": None,
    }
    assert PROBE.validate_final_checkpoint_payload(payload, "single64")["eligible_update"] == 64
    payload["iter"] = 31
    with pytest.raises(PROBE.MultibiasProbeError, match="not the final64"):
        PROBE.validate_final_checkpoint_payload(payload, "checkpoint32")


def test_preimport_validation_precedes_isaac_import_and_existing_output_is_rejected() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert source.index("validate_preimport_sources(") < source.index("from isaaclab.app import AppLauncher")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--phase", "preflight", "--runtime-root", str(ROOT), "--current-root", str(ROOT), "--origin-protocol", str(ORIGIN), "--origin-protocol-sha256", _sha(ORIGIN), "--contract", str(ORIGIN), "--contract-sha256", _sha(ORIGIN), "--output-dir", str(ROOT)],
        capture_output=True, text=True,
    )
    assert completed.returncode != 0
    assert "output directory/report already exists" in completed.stderr


def test_report_is_persisted_before_native_close(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    program = r'''
import importlib.util, os, sys
from pathlib import Path
spec=importlib.util.spec_from_file_location("p", sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class Probe:
 @staticmethod
 def _close_simulation_app(app, code): app.close(); return code
class App:
 def close(self): os._exit(0)
m.persist_report_before_app_close(Path(sys.argv[2]), {"status":"completed","phase":"evaluate"}, Probe, App(), 0)
raise AssertionError("unreachable")
'''
    completed = subprocess.run([sys.executable, "-c", program, str(SCRIPT), str(report)], capture_output=True, text=True)
    assert completed.returncode == 0
    assert json.loads(report.read_text())["status"] == "completed"
