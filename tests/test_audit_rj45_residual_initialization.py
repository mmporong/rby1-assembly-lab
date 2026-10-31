from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/audit_rj45_residual_initialization.py"
SPEC = importlib.util.spec_from_file_location("audit_rj45_residual_initialization", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
CONFIG = SCRIPT.parents[1] / "configs/rj45-comparison-development.json"


def test_bound_residual_actions_matches_component_clip_and_block_projection() -> None:
    actions = torch.tensor([[2.0, 2.0, 0.0, 0.5, 0.5, 0.5]])
    bounded, rates = AUDIT.bound_residual_actions(actions)
    torch.testing.assert_close(
        bounded[0, :3], torch.tensor([2.0**-0.5, 2.0**-0.5, 0.0])
    )
    torch.testing.assert_close(bounded[0, 3:], torch.tensor([0.5, 0.5, 0.5]))
    assert rates["component_clip_rate"] == pytest.approx(2.0 / 6.0)
    assert rates["translation_projection_rate"] == 1.0
    assert rates["rotation_projection_rate"] == 0.0
    assert rates["environment_sample_clip_boundary_rate"] == 1.0
    assert rates["environment_sample_any_projection_rate"] == 1.0


def test_cpu_audit_is_finite_reproducible_and_distinguishes_policy_transforms() -> None:
    first = AUDIT.run_audit(
        config_path=CONFIG,
        seed=41,
        sample_count=10_000,
        initial_log_std=-1.0,
    )
    second = AUDIT.run_audit(
        config_path=CONFIG,
        seed=41,
        sample_count=10_000,
        initial_log_std=-1.0,
    )
    assert first["results"]["ppo"]["sample_sha256"] == second["results"]["ppo"]["sample_sha256"]
    assert first["results"]["sac"]["sample_sha256"] == second["results"]["sac"]["sample_sha256"]
    assert first["results"]["ppo"]["sample_sha256"] != first["results"]["sac"]["sample_sha256"]
    assert first["results"]["ppo"]["policy_transform_component_clip_rate"] > 0.0
    assert first["results"]["sac"]["policy_transform_component_clip_rate"] == 0.0
    assert first["claims"]["physics_executed"] is False
    assert first["claims"]["training_executed"] is False
    assert first["claims"]["g2_parameter_frozen"] is False
    assert first["provenance"]["sample_count"] == 10_000
    assert first["provenance"]["actor_observation_contract"] == "79D_simulation_proxy_only"
    assert first["provenance"]["skrl_version"] == "2.1.0"
    assert first["provenance"]["task_config_sha256"] == hashlib.sha256(
        (SCRIPT.parents[1] / "configs/tasks/rj45-v1.json").read_bytes()
    ).hexdigest()
    assert first["provenance"]["source_sha256"]["audit_script"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert set(first["provenance"]["source_sha256"]) == {
        "audit_script", "comparison_rl", "rj45_env", "rj45_training",
        "rj45_training_env", "isaac_env",
    }
    json.dumps(first, allow_nan=False)


def test_physical_unit_statistics_use_independent_translation_and_rotation_bounds() -> None:
    report = AUDIT.run_audit(
        config_path=CONFIG,
        seed=7,
        sample_count=512,
        initial_log_std=-3.0,
    )
    ppo = report["results"]["ppo"]
    translation = ppo["translation_block_norm"]
    rotation = ppo["rotation_block_norm"]
    assert translation["mm"]["p95"] == pytest.approx(translation["m"]["p95"] * 1_000.0)
    assert rotation["deg"]["p95"] == pytest.approx(rotation["rad"]["p95"] * 180.0 / math.pi)
    assert translation["m"]["p95"] == pytest.approx(
        translation["normalized"]["p95"] * report["provenance"]["translation_bound_m"]
    )
    assert rotation["rad"]["p95"] == pytest.approx(
        rotation["normalized"]["p95"] * report["provenance"]["rotation_bound_rad"]
    )


def test_cli_requires_explicit_log_std_and_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "audit.json"
    assert AUDIT.main([
        "--config", str(CONFIG), "--output", str(output),
        "--sample-count", "64", "--initial-log-std", "-4.5",
    ]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["provenance"]["initial_log_std"] == -4.5
    with pytest.raises(FileExistsError):
        AUDIT.main([
            "--config", str(CONFIG), "--output", str(output),
            "--sample-count", "64", "--initial-log-std", "-4.5",
        ])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -5.1, 1.1])
def test_audit_rejects_invalid_log_std(value: float) -> None:
    with pytest.raises(ValueError, match="initial_log_std"):
        AUDIT.run_audit(
            config_path=CONFIG,
            seed=1,
            sample_count=10,
            initial_log_std=value,
        )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -0.1])
def test_audit_rejects_nonfinite_or_nonpositive_action_bounds(
    tmp_path: Path, value: float
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["shared_control_contract"]["translation_bound_m"] = value
    config = tmp_path / "invalid-bounds.json"
    config.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="finite and positive"):
        AUDIT.run_audit(
            config_path=config,
            seed=1,
            sample_count=10,
            initial_log_std=-1.0,
        )


def test_audit_rejects_task_controller_bound_mismatch(tmp_path: Path) -> None:
    task_path = SCRIPT.parents[1] / "configs/tasks/rj45-v1.json"
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["controller"]["residual_rotation_bound_rad"] *= 0.5
    modified = tmp_path / "task.json"
    modified.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="must match"):
        AUDIT.run_audit(
            config_path=CONFIG,
            task_config_path=modified,
            seed=1,
            sample_count=10,
            initial_log_std=-1.0,
        )


def test_audit_fails_if_actual_actor_sampler_drifts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_make_agent = AUDIT.make_agent

    def drifting_make_agent(algorithm, **kwargs):
        agent = original_make_agent(algorithm, **kwargs)
        if algorithm == "ppo":
            original_act = agent.policy.act

            def drift(inputs, *, role=""):
                actions, outputs = original_act(inputs, role=role)
                return actions * 0.5, outputs

            agent.policy.act = drift
        return agent

    monkeypatch.setattr(AUDIT, "make_agent", drifting_make_agent)
    with pytest.raises(RuntimeError, match="actual policy sampler"):
        AUDIT.run_audit(
            config_path=CONFIG,
            seed=3,
            sample_count=32,
            initial_log_std=-1.0,
        )
