from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rby1_assembly.rj45_diagnostic_trace import (
    ContactDiagnosticObserver,
    DiagnosticTraceError,
    JsonlTraceWriter,
)


def _training_snapshot(observation: np.ndarray, timestamp: float = 1.0) -> dict:
    return {
        "policy_input_observation": observation,
        "policy_input_timestamp_s": np.asarray([timestamp]),
        "policy_input_valid": np.asarray([True]),
        "requested_expanded_residual_action": np.full((1, 6), 0.1),
        "applied_bounded_residual_action": np.full((1, 6), 0.2),
        "applied_tip_target_offset_entry_6d": np.full((1, 6), 0.3),
    }


class FakeEnv:
    def __init__(self, source: dict, *, poison_post_reset: bool = False) -> None:
        self.source = source
        self._phase = np.asarray([4])
        self._episode_length_buf = np.asarray([11])
        self._step_dt = 0.1
        self.poison_post_reset = poison_post_reset

    def _training_step_snapshot(self):
        raise AssertionError("observer read the stale pre-step teacher cache")

    @property
    def episode_length_buf(self):
        if self.poison_post_reset:
            raise AssertionError("terminal capture read reset live env")
        return self._episode_length_buf

    @property
    def step_dt(self):
        if self.poison_post_reset:
            raise AssertionError("terminal capture read reset live env")
        return self._step_dt


def _physics(*, force=(3.0, 4.0, 0.0), load=6.0):
    return SimpleNamespace(
        insertion_contact_force_w=np.asarray(force),
        insertion_contact_force_n=5.0,
        insertion_contact_load_n=load,
        contact_force_proxy_n=np.asarray([2.0, 3.0]),
        phase="insert",
        ground_truth=SimpleNamespace(
            actual_depth_m=0.02,
            lateral_error_m=0.001,
            orientation_error_rad=0.002,
        ),
    )


def _observer(tmp_path: Path, *, terminal_snapshot=None, live_snapshot=None):
    writer = JsonlTraceWriter(tmp_path / "ticks.jsonl", tmp_path / "ticks.sidecar.json")
    return ContactDiagnosticObserver(
        writer,
        object(),
        live_snapshot=live_snapshot or (lambda *_args: _physics()),
        terminal_snapshot_from_extras=terminal_snapshot or (lambda *_args: _physics()),
    )


def _post_snapshot(
    observation: np.ndarray,
    *,
    policy_timestamp: float = 1.0,
    post_timestamp: float = 1.1,
    action: float = 0.7,
) -> dict:
    return {
        "timestamp_s": np.asarray([post_timestamp]),
        "policy_input_observation": observation.copy(),
        "policy_input_timestamp_s": np.asarray([policy_timestamp]),
        "policy_input_valid": np.asarray([True]),
        "requested_expanded_residual_action": np.full((1, 6), action),
        "applied_bounded_residual_action": np.full((1, 6), action),
        "applied_tip_target_offset_entry_6d": np.full((1, 6), action),
    }


def _capture_pre(
    observer: ContactDiagnosticObserver,
    env: FakeEnv,
    observation: np.ndarray,
    *,
    action: float = 0.0,
    timestamp: float = 1.0,
) -> None:
    observer.capture_pre(
        env,
        observation,
        np.full((1, 6), action),
        np.asarray([True]),
        np.asarray([timestamp]),
    )


def test_terminal_uses_pre_reset_extras_and_never_reads_mutated_live_env(tmp_path: Path) -> None:
    observation = np.arange(79, dtype=float)[None, :]
    env = FakeEnv(_training_snapshot(observation))
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation)
    env.poison_post_reset = True
    observer.capture_post(
        env,
        {"terminal_snapshot": _post_snapshot(observation)},
        True,
    )
    sidecar = observer.finalize()
    row = json.loads((tmp_path / "ticks.jsonl").read_text().strip())
    assert row["post_state_source"] == "pre_autoreset_terminal_snapshot"
    assert row["observer_pre_step_index"] == 0
    assert row["observer_post_step_index"] == 1
    assert sidecar["terminal_count"] == 1


def test_teacher_cache_refresh_and_current_actions_are_captured_post_step(
    tmp_path: Path,
) -> None:
    observation = np.arange(79, dtype=float)[None, :]
    source = _training_snapshot(np.zeros((1, 79)), timestamp=0.5)
    env = FakeEnv(source)
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation, action=0.7)
    current = _post_snapshot(observation, action=0.7)
    observer.capture_post(
        env,
        {"terminal_snapshot": current},
        True,
    )
    current["policy_input_observation"][:] = -999.0
    current["applied_bounded_residual_action"][:] = -999.0
    observer.finalize()
    row = json.loads((tmp_path / "ticks.jsonl").read_text().strip())
    assert row["policy_command_depth"] == 62.0
    assert row["policy_advance"] == 63.0
    assert row["commanded_normalized_action6"] == [0.7] * 6
    assert row["current_requested_expanded_residual_action6"] == [0.7] * 6
    assert row["current_applied_bounded_residual_action6"] == [0.7] * 6
    assert row["current_applied_tip_target_offset_entry6"] == [0.7] * 6


def test_reset_source_timestamps_are_preserved_without_fake_monotonic_fallback(tmp_path: Path) -> None:
    observation = np.zeros((1, 79))
    env = FakeEnv(_training_snapshot(observation, timestamp=2.0))
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation, timestamp=2.0)
    observer.capture_post(
        env,
        {
            "terminal_snapshot": _post_snapshot(
                observation, policy_timestamp=2.0, post_timestamp=1.9
            )
        },
        True,
    )
    observer.finalize()
    row = json.loads((tmp_path / "ticks.jsonl").read_text().strip())
    assert row["pre_policy_timestamp_s"] == 2.0
    assert row["post_source_timestamp_s"] == 1.9
    assert row["observer_pre_control_elapsed_s"] == 0.0
    assert row["observer_post_control_elapsed_s"] == 0.1


def test_missing_post_step_source_field_fails_without_fallback(tmp_path: Path) -> None:
    observation = np.zeros((1, 79))
    source = _post_snapshot(observation)
    del source["policy_input_timestamp_s"]
    observer = _observer(tmp_path)
    env = FakeEnv({})
    _capture_pre(observer, env, observation)
    with pytest.raises(DiagnosticTraceError, match="missing fields"):
        observer.capture_post(env, {"terminal_snapshot": source}, True)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda snapshot: snapshot["policy_input_observation"].fill(9.0), "caller policy"),
        (
            lambda snapshot: snapshot.__setitem__(
                "policy_input_timestamp_s", np.asarray([9.0])
            ),
            "caller timestamp",
        ),
        (
            lambda snapshot: snapshot.__setitem__(
                "policy_input_valid", np.asarray([False])
            ),
            "caller validity",
        ),
    ),
)
def test_post_step_snapshot_must_be_fresh_for_the_caller_input(
    tmp_path: Path, mutation, message: str
) -> None:
    observation = np.zeros((1, 79))
    env = FakeEnv({})
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation)
    snapshot = _post_snapshot(observation)
    mutation(snapshot)
    with pytest.raises(DiagnosticTraceError, match=message):
        observer.capture_post(env, {"terminal_snapshot": snapshot}, True)


def test_terminal_timestamp_source_is_required_without_live_env_fallback(tmp_path: Path) -> None:
    observation = np.zeros((1, 79))
    env = FakeEnv(_training_snapshot(observation))
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation)
    env.poison_post_reset = True
    with pytest.raises(DiagnosticTraceError, match="terminal_snapshot.timestamp_s"):
        observer.capture_post(
            env,
            {"terminal_snapshot": _post_snapshot(observation) | {"timestamp_s": None}},
            True,
        )


def test_force_vector_and_normal_load_contracts_are_checked(tmp_path: Path) -> None:
    observation = np.zeros((1, 79))
    env = FakeEnv(_training_snapshot(observation))
    observer = _observer(
        tmp_path,
        terminal_snapshot=lambda *_args: _physics(load=4.0),
    )
    _capture_pre(observer, env, observation)
    with pytest.raises(DiagnosticTraceError, match="normal-load sum"):
        observer.capture_post(
            env,
            {"terminal_snapshot": _post_snapshot(observation)},
            True,
        )


def test_jsonl_hash_chain_has_one_row_per_step_and_one_terminal(tmp_path: Path) -> None:
    observation = np.zeros((1, 79))
    env = FakeEnv(_training_snapshot(observation))
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation)
    observer.capture_post(
        env, {"terminal_snapshot": _post_snapshot(observation)}, False
    )
    env.source = _training_snapshot(observation, timestamp=1.1)
    _capture_pre(observer, env, observation, timestamp=1.1)
    observer.capture_post(
        env,
        {
            "terminal_snapshot": _post_snapshot(
                observation, policy_timestamp=1.1, post_timestamp=1.2
            )
        },
        True,
    )
    sidecar = observer.finalize()
    rows = [json.loads(line) for line in (tmp_path / "ticks.jsonl").read_text().splitlines()]
    assert len(rows) == sidecar["row_count"] == 2
    assert rows[1]["previous_row_sha256"] == rows[0]["row_sha256"]
    assert [row["terminal"] for row in rows] == [False, True]
    with pytest.raises(DiagnosticTraceError, match="order"):
        _capture_pre(observer, env, observation)


def test_finalize_revalidates_stream_hashes_before_writing_sidecar(tmp_path: Path) -> None:
    observation = np.zeros((1, 79))
    env = FakeEnv(_training_snapshot(observation))
    observer = _observer(tmp_path)
    _capture_pre(observer, env, observation)
    observer.capture_post(
        env,
        {"terminal_snapshot": _post_snapshot(observation)},
        True,
    )
    path = tmp_path / "ticks.jsonl"
    row = json.loads(path.read_text())
    row["post_physical_gt"]["actual_depth_m"] = 999.0
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(DiagnosticTraceError, match="hash chain"):
        observer.finalize()
    assert not (tmp_path / "ticks.sidecar.json").exists()
