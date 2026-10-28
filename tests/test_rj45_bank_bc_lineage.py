from __future__ import annotations

import copy
import unittest

from rby1_assembly.held_grasp import bank_source_contract_sha256
from rby1_assembly.rj45_task import (
    _validate_policy_initialization_metadata,
    _validate_rj45_training_contract,
    verify_checkpoint_lineage,
)


def bank_proof() -> dict:
    return {
        "manifest_sha256": "5" * 64,
        "selected_split": "train",
        "selected_count": 2,
        "selected_grasp_ids": ["grasp-a", "grasp-b"],
        "selected_artifact_sha256": ["6" * 64, "7" * 64],
        "manifest_split_counts": {"train": 2, "dev": 1, "final": 1},
        "provenance_contract": {
            "task_config_sha256": "d" * 64,
            "source_and_config_sha256": {
                "configs/runtime.json": "a" * 64,
                "configs/tasks/rj45-v1.json": "b" * 64,
                "src/rby1_assembly/rj45_training_env.py": "c" * 64,
            },
        },
        "provenance_source_task_match": True,
        "final_opened": False,
        "no_final_opened": True,
    }


def training_contract() -> dict:
    return {
        "schema_version": 1,
        "policy_observation_schema_version": 1,
        "policy_observation_dimension": 79,
        "policy_action_dimension": 6,
        "action_mode": "right_only_6d",
        "episode_scope": "held_grasp_alignment_insertion",
        "warmup_s": 0.2,
        "active_horizon_s": 24.0,
        "estimator": "frozen_kinematic_grasp_estimator",
        "calibration_bias": "episode_fixed_one_shot_se3",
        "calibration_bias_level": 0,
        "policy_pose_source": "joint_encoder_and_frozen_grasp_transform",
        "ground_truth_usage": "scripted_setup_control_and_safety_then_reward_and_independent_evaluation_only",
        "held_grasp_reset_proof": bank_proof(),
        "training_config_sha256": "4" * 64,
        "parallel_contact_evidence": None,
        "learned_recovery": False,
        "retract_controller": "baseline_only",
    }


def bc_initialization() -> dict:
    before = {
        "actor": "1" * 64,
        "critic": "2" * 64,
        "noise": "3" * 64,
        "optimizer": "8" * 64,
    }
    after = {**before, "actor": "9" * 64}
    return {
        "mode": "bc_actor_warm_start",
        "output_dimension": 6,
        "artifact_file_sha256": "e" * 64,
        "artifact_actor_state_sha256": "9" * 64,
        "loaded_component": "runner.alg.policy.actor",
        "hashes_before": before,
        "hashes_after": after,
        "actor_changed": True,
        "bc_provenance": {
            "source_sha256": bank_source_contract_sha256(bank_proof()),
            "task_sha256": "d" * 64,
            "training_sha256": "4" * 64,
            "bank_sha256": "5" * 64,
            "dataset_sha256": "f" * 64,
        },
        "bc_artifact_sha256": "e" * 64,
    }


def runtime_contract() -> dict:
    return {
        "runtime_versions": {"isaacsim": "5.1.0.0"},
        "physics_dt_s": 1 / 120,
        "control_decimation": 4,
        "control_dt_s": 1 / 30,
        "observation_noise": {"position_std_m": 0.0005},
        "baseline_controller_state_source": (
            "joint_encoder_frozen_grasp_estimator_with_episode_fixed_calibration_bias"
        ),
        "task_config_path": "configs/tasks/rj45-v1.json",
        "policy_config_path": "configs/ppo-rj45-v1.json",
        "collision_monitor_mode": "passive_parallel_native_ordering_verified",
        "num_envs": 64,
        "rj45_training_contract": training_contract(),
    }


def checkpoint_lineage() -> tuple[dict, dict, dict, dict]:
    runtime = runtime_contract()
    evidence = {
        "report_sha256": "a" * 64,
        "trace_sha256": "b" * 64,
        "trace_records": 12,
    }
    sources = {"src/rby1_assembly/rj45_task.py": "c" * 64}
    lineage = {
        "schema_version": 1,
        "task_id": "rj45-v1",
        "task_config_sha256": "d" * 64,
        "source_and_config_sha256": sources,
        "rj45_nominal_evidence": evidence,
        "policy_initialization": bc_initialization(),
        **copy.deepcopy(runtime),
    }
    return lineage, runtime, evidence, sources


class Rj45BankBcLineageTests(unittest.TestCase):
    def test_training_contract_accepts_strict_train_bank_proof(self) -> None:
        _validate_rj45_training_contract(training_contract())

    def test_training_contract_rejects_final_or_malformed_bank_proof(self) -> None:
        final = training_contract()
        final["held_grasp_reset_proof"].update(
            {"selected_split": "final", "final_opened": True, "no_final_opened": False}
        )
        with self.assertRaisesRegex(ValueError, "sealed train split"):
            _validate_rj45_training_contract(final)
        malformed = training_contract()
        malformed["held_grasp_reset_proof"]["selected_count"] = 3
        with self.assertRaisesRegex(ValueError, "count, IDs, or hashes"):
            _validate_rj45_training_contract(malformed)

    def test_bc_metadata_accepts_actor_only_warm_start(self) -> None:
        _validate_policy_initialization_metadata(bc_initialization())

    def test_bc_metadata_rejects_critic_change(self) -> None:
        damaged = bc_initialization()
        damaged["hashes_after"]["critic"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "changed PPO critic"):
            _validate_policy_initialization_metadata(damaged)

    def test_bc_metadata_rejects_float_dimension_and_unbound_actor_hash(self) -> None:
        float_dimension = bc_initialization()
        float_dimension["output_dimension"] = 6.0
        with self.assertRaisesRegex(ValueError, "dimension"):
            _validate_policy_initialization_metadata(float_dimension)
        unbound = bc_initialization()
        unbound["artifact_actor_state_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "after hash"):
            _validate_policy_initialization_metadata(unbound)

    def test_checkpoint_lineage_binds_bc_to_bank_task_and_training(self) -> None:
        lineage, runtime, evidence, sources = checkpoint_lineage()
        verify_checkpoint_lineage(
            lineage,
            task_id="rj45-v1",
            task_config_sha256="d" * 64,
            source_sha256=sources,
            runtime_contract=runtime,
            rj45_nominal_evidence=evidence,
        )
        for name in ("source_sha256", "task_sha256", "training_sha256", "bank_sha256"):
            damaged = copy.deepcopy(lineage)
            damaged["policy_initialization"]["bc_provenance"][name] = "0" * 64
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "lineage differs"):
                verify_checkpoint_lineage(
                    damaged,
                    task_id="rj45-v1",
                    task_config_sha256="d" * 64,
                    source_sha256=sources,
                    runtime_contract=runtime,
                    rj45_nominal_evidence=evidence,
                )
        wrong_bank_task_runtime = copy.deepcopy(runtime)
        wrong_bank_task_runtime["rj45_training_contract"]["held_grasp_reset_proof"][
            "provenance_contract"
        ]["task_config_sha256"] = "0" * 64
        wrong_bank_task_lineage = copy.deepcopy(lineage)
        wrong_bank_task_lineage["rj45_training_contract"] = copy.deepcopy(
            wrong_bank_task_runtime["rj45_training_contract"]
        )
        with self.assertRaisesRegex(ValueError, "bank task lineage"):
            verify_checkpoint_lineage(
                wrong_bank_task_lineage,
                task_id="rj45-v1",
                task_config_sha256="d" * 64,
                source_sha256=sources,
                runtime_contract=wrong_bank_task_runtime,
                rj45_nominal_evidence=evidence,
            )


if __name__ == "__main__":
    unittest.main()
