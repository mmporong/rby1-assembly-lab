from __future__ import annotations

from enum import IntEnum
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import tempfile

import pytest
import torch

from rby1_assembly.rj45_grasp_generation import (
    load_grasp_generation_catalog,
    make_rj45_grasp_generation_env_type,
    select_grasp_generation_entry,
    validate_grasp_generation_catalog,
    validate_grasp_generation_launch,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "configs/rj45-grasp-generation-v2.json"
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "docs/evidence/rj45-demonstrations-20261028/grasp-generation-v2-protocol.json"
)


class _Phase(IntEnum):
    PREGRASP = 0
    CLOSE = 1
    CHECK_LIFT = 2
    ALIGN = 3


class _BaseEnvironment:
    def _grasp_target_poses(self):
        return (
            self.base_left.clone(),
            self.base_left_quaternion.clone(),
            self.base_right.clone(),
            self.base_right_quaternion.clone(),
        )


def _environment(env_type: type, *, count: int = 1):
    env = env_type.__new__(env_type)
    env.device = torch.device("cpu")
    env.num_envs = count
    env.base_left = torch.zeros((count, 3))
    env.base_right = torch.ones((count, 3))
    env.base_left_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(
        count, 1
    )
    env.base_right_quaternion = env.base_left_quaternion.clone()
    env.socket = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=env.base_left_quaternion.clone())
    )
    env.plug = SimpleNamespace(
        data=SimpleNamespace(root_quat_w=env.base_right_quaternion.clone())
    )
    return env


def test_catalog_is_balanced_and_uses_the_declared_twelve_rows() -> None:
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    assert len(catalog["entries"]) == 12
    for coordinate in (1, 2, 4, 5):
        values = []
        for entry in catalog["entries"]:
            values.append(
                (
                    entry["left_object_local_offset_m"]
                    + entry["right_object_local_offset_m"]
                )[coordinate]
            )
        assert values.count(0.00015) == 6
        assert values.count(-0.00015) == 6


def test_seed_catalog_assignments_are_fixed_before_execution() -> None:
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    pilot = [
        select_grasp_generation_entry(catalog, seed)
        for seed in range(20261301, 20261305)
    ]
    assert [entry["id"] for _, entry in pilot] == [
        "grasp-00",
        "grasp-01",
        "grasp-02",
        "grasp-03",
    ]
    assert select_grasp_generation_entry(catalog, 20261219)[1]["id"] == "grasp-00"
    assert select_grasp_generation_entry(catalog, 20261233)[1]["id"] == "grasp-08"
    assert select_grasp_generation_entry(catalog, 20261244)[1]["id"] == "grasp-11"
    with pytest.raises(ValueError, match="not uniquely assigned"):
        select_grasp_generation_entry(catalog, 20261027)


def test_noncanonical_catalog_row_is_rejected() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    payload["entries"][0]["left_object_local_offset_m"][0] = 0.00015
    with pytest.raises(ValueError, match="not canonical"):
        validate_grasp_generation_catalog(payload)


def test_catalog_rejects_boolean_disguised_as_numeric_zero() -> None:
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    payload["entries"][0]["left_object_local_offset_m"][0] = False
    with pytest.raises(ValueError, match="not canonical"):
        validate_grasp_generation_catalog(payload)


def test_wrapper_rotates_object_local_offsets_into_world() -> None:
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    env_type = make_rj45_grasp_generation_env_type(
        _BaseEnvironment, catalog, seed=20261301
    )
    env = _environment(env_type)
    half_sqrt_two = 2.0**-0.5
    env.socket.data.root_quat_w[:] = torch.tensor(
        [half_sqrt_two, 0.0, 0.0, half_sqrt_two]
    )
    left, _, right, _ = env._grasp_target_poses()
    entry = select_grasp_generation_entry(catalog, 20261301)[1]
    local_left = entry["left_object_local_offset_m"]
    assert torch.allclose(
        left[0], torch.tensor([-local_left[1], 0.0, local_left[2]]), atol=1.0e-8
    )
    assert torch.allclose(
        right[0], env.base_right[0] + torch.tensor(entry["right_object_local_offset_m"])
    )


@pytest.mark.parametrize("phase", (_Phase.CLOSE, _Phase.CHECK_LIFT))
def test_frozen_grasp_reference_does_not_receive_offset_twice(
    monkeypatch, phase: _Phase
) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "rby1_assembly.isaac_env",
        SimpleNamespace(AssemblyPhase=_Phase),
    )
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    env_type = make_rj45_grasp_generation_env_type(
        _BaseEnvironment, catalog, seed=20261302
    )
    env = _environment(env_type, count=2)
    env._phase = torch.tensor([int(phase), int(phase)])
    env._grasp_reference_locked = torch.tensor([True, False])
    left, _, right, _ = env._grasp_target_poses()
    assert torch.equal(left[0], env.base_left[0])
    assert torch.equal(right[0], env.base_right[0])
    assert not torch.equal(left[1], env.base_left[1])
    assert not torch.equal(right[1], env.base_right[1])


def test_report_binds_seed_catalog_entry_and_offset_method() -> None:
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    env_type = make_rj45_grasp_generation_env_type(
        _BaseEnvironment, catalog, seed=20261241
    )
    report = _environment(env_type).rj45_grasp_generation_report()
    assert report["selected_catalog_entry_id"] == "grasp-10"
    assert report["seed"] == 20261241
    assert report["assignment_group"] == "production_final"
    assert report["offset_method"] == catalog["offset_method"]


def test_launch_contract_rejects_nonisolated_generation_modes() -> None:
    valid = {
        "catalog_path": CATALOG_PATH,
        "task": "rj45",
        "num_envs": 1,
        "seed": 20261301,
        "command_fault": None,
        "training_config": None,
        "calibration_bias": False,
        "bias_level": 0,
    }
    validate_grasp_generation_launch(**valid)
    for override in (
        {"task": "toy"},
        {"num_envs": 2},
        {"seed": None},
        {"command_fault": "hand_collision"},
        {"training_config": Path("training.json")},
        {"calibration_bias": True},
        {"bias_level": 1},
    ):
        with pytest.raises(ValueError, match="isolated|requires"):
            validate_grasp_generation_launch(**(valid | override))


def test_run_sim_keeps_generation_wrapper_opt_in() -> None:
    source = (PROJECT_ROOT / "scripts/run_sim.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--rj45-grasp-generation-catalog", type=Path)' in source
    assert "if args.rj45_grasp_generation_catalog is not None:" in source
    assert "grasp_generation_catalog_sha256" in source


def test_protocol_preserves_failed_pilot_gate_and_staged_freezes() -> None:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    diversity = protocol["motivation"]["corrected_time_posthoc_diversity"]
    assert diversity["maximum_pairwise_distinct_count"] == 2
    assert diversity["gate_pass"] is False
    assert protocol["physical_pilot"]["seeds"] == list(range(20261301, 20261305))
    assert protocol["static_cad_audit"][
        "gripper_convex_hull_vs_locator_dynamic_collision_verified"
    ] is False
    freezes = protocol["stage_freeze_contract"]
    assert "teacher dataset SHA" not in freezes["before_teacher_collection"]
    assert "teacher dataset SHA in addition to the teacher-stage inputs" in freezes[
        "before_bc_training"
    ]


def test_pilot_audit_cli_preserves_all_failed_attempts_and_returns_gate_failure() -> None:
    catalog = load_grasp_generation_catalog(CATALOG_PATH)
    attempts = []
    for seed in range(20261301, 20261305):
        entry = select_grasp_generation_entry(catalog, seed)[1]
        attempts.append(
            {
                "seed": seed,
                "catalog_entry_id": entry["id"],
                "status": "failed",
                "failure_reason": f"pilot-{seed}-failed",
            }
        )
    artifacts = PROJECT_ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=artifacts) as directory:
        directory = Path(directory)
        manifest = directory / "attempts.json"
        output = directory / "audit.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "catalog_sha256": hashlib.sha256(CATALOG_PATH.read_bytes()).hexdigest(),
                    "attempts": attempts,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts/audit_rj45_grasp_generation_pilot.py"),
                "--catalog",
                str(CATALOG_PATH),
                "--attempts",
                str(manifest),
                "--output",
                str(output),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        assert result.returncode == 2, result.stderr
        audit = json.loads(output.read_text(encoding="utf-8"))
        assert len(audit["attempts"]) == 4
        assert all(attempt["status"] == "failed" for attempt in audit["attempts"])
        assert audit["maximum_pairwise_distinct_count"] == 0
        assert audit["gate_pass"] is False
