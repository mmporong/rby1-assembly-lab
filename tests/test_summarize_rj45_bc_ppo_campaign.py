from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_rj45_bc_ppo_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_rj45_bc_ppo_campaign", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


SOURCE = {"scripts/evaluate_policy.py": "a" * 64}
TASK_SHA = "b" * 64
BANK_SHA = "c" * 64
CONFIG_SHA = "d" * 64


def report(*, successes: int = 8) -> dict:
    outcomes = []
    for index in range(64):
        condition = {
            "evaluation_start": "held",
            "task_config_sha256": TASK_SHA,
            "base_evaluation_seed": 20261122,
            "environment_index": index,
            "episode_index": 1,
            "calibration_bias_level": 0,
            "calibration_latent_case_seed": (
                20261122 + 45_123_017 + 1_000_003 + index
            ),
            "calibration_latent_case_seed_formula": MODULE.LATENT_FORMULA,
            "grasp_index": index % 2,
            "grasp_id": MODULE.DEV_GRASP_IDS[index % 2],
            "bank_manifest_sha256": BANK_SHA,
            "bank_split": "dev",
        }
        success = index < successes
        outcomes.append(
            {
                "attempt_index": index,
                "valid": True,
                "success": success,
                "environment_index": index,
                "episode_index": 1,
                "paired_condition": condition,
                "paired_condition_id": json.dumps(
                    condition, sort_keys=True, separators=(",", ":")
                ),
                "forbidden_contact_load_peak_n": 0.0,
                "final_lateral_error_m": 4.0e-5 if success else 8.0e-5,
                "elapsed_s": 12.0 if success else 24.0,
                "calibration_bias_translation_entry_m": [index * 1.0e-7, 0.0, 0.0],
                "calibration_bias_rotation_vector_entry_rad": [0.0, index * 1.0e-6, 0.0],
                "bc_actor_sha256": None,
            }
        )
    return {
        "rj45_evaluation_schema_version": 2,
        "baseline_mode": "biased_zero",
        "seed": 20261122,
        "requested_attempts": 64,
        "attempts": 64,
        "evaluation_num_envs": 64,
        "per_environment_episode_quota": [1] * 64,
        "per_environment_accepted_attempt_count": [1] * 64,
        "episode_outcomes": outcomes,
        "valid_successes": successes,
        "valid_failures": 64 - successes,
        "invalid_episodes": 0,
        "valid_episodes": 64,
        "max_contact_force_proxy_n": 4.0,
        "evaluation_horizon_s": 24.0,
        "paired_condition_key": MODULE.PAIR_KEY_TEXT,
        "rj45_training_config": {
            "sha256": MODULE.TRAINING_CONFIG_SHA256,
            "action_mode": "right_only_6d",
            "observation_dimension": 79,
            "calibration_bias_level": 0,
        },
        "provenance": {
            "source_sha256": SOURCE,
            "task_config_sha256": TASK_SHA,
            "policy_config_sha256": CONFIG_SHA,
            "checkpoint_sha256": None,
            "evaluation_inputs_reverified_at_end": True,
        },
        "evaluation_held_grasp_bank_proof": {
            "manifest_sha256": BANK_SHA,
            "selected_split": "dev",
            "selected_count": 2,
            "selected_grasp_ids": list(MODULE.DEV_GRASP_IDS),
            "selected_artifact_sha256": ["e" * 64, "f" * 64],
            "manifest_split_counts": {"train": 8, "dev": 2, "final": 2},
            "provenance_contract": {
                "task_config_sha256": TASK_SHA,
                "source_and_config_sha256": {},
            },
            "provenance_source_task_match": True,
            "final_opened": False,
            "no_final_opened": True,
        },
        "training_held_grasp_bank_proof": None,
    }


class CampaignSummaryTests(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, value: object) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _checkpoint_bundle(self, root: Path) -> tuple[dict, dict, dict[str, str], dict[str, str]]:
        checkpoint = root / "runs/ppo_only/20261131/model_19.pt"
        checkpoint.parent.mkdir(parents=True)
        torch.save({"iter": 19}, checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        generated = {
            "path": "campaign/configs/f-seed-20261131.json",
            "sha256": "6" * 64,
        }
        plan_inputs = {"scripts/train.py": "7" * 64}
        initialization = {
            "mode": "residual_actor_final_linear_zero",
            "output_dimension": 6,
            "zero_actor_output_layer": True,
            "weight_all_zero_after_initialization": True,
            "bias_all_zero_after_initialization": True,
            "deterministic_actor_output_all_zero": True,
            "deterministic_actor_output_max_abs": 0.0,
            "stochastic_policy_noise_modified": False,
        }
        runner_path = checkpoint.parent / "runner_config.json"
        runner_sha = self._write_json(
            runner_path,
            {"seed": 20261131, "max_iterations": 200, "num_steps_per_env": 32},
        )
        provenance_path = checkpoint.parent / "training_provenance.json"
        provenance = {
            "seed": 20261131,
            "num_envs": 256,
            "checkpoint_scope": "checkpoints created in this run directory",
            "runtime_versions": {"rsl-rl-lib": "3.0.1"},
            "source_and_config_sha256": {
                **SOURCE,
                generated["path"]: generated["sha256"],
                "scripts/train.py": plan_inputs["scripts/train.py"],
                str(runner_path): runner_sha,
            },
            "policy_initialization": initialization,
        }
        provenance_sha = self._write_json(provenance_path, provenance)
        manifest_path = checkpoint.parent / "checkpoint_manifest.json"
        manifest_sha = self._write_json(
            manifest_path,
            {
                "schema_version": 1,
                "training_completed": True,
                "training_provenance_sha256": provenance_sha,
                "checkpoints": {"model_19.pt": checkpoint_sha},
            },
        )
        inspection = {
            "status": "checkpoint_payload_and_lineage_verified",
            "planned_training_updates": 200,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha,
                "payload_iteration": 19,
                "completed_updates": 20,
            },
            "checkpoint_manifest": {
                "path": str(manifest_path),
                "sha256": manifest_sha,
            },
            "training_provenance": {
                "path": str(provenance_path),
                "sha256": provenance_sha,
            },
            "runner_config": {"path": str(runner_path), "sha256": runner_sha},
        }
        evaluation_report = {
            "provenance": {
                "checkpoint_sha256": checkpoint_sha,
                "verified_checkpoint_evidence": {
                    "training_provenance_sha256": provenance_sha,
                    "checkpoint_manifest_sha256": manifest_sha,
                },
            }
        }
        return inspection, evaluation_report, generated, plan_inputs

    def test_report_uses_all_attempts_and_preserves_success_failure_time(self) -> None:
        metrics = MODULE.validate_report(
            report(),
            mode="biased_zero",
            expected_source=SOURCE,
            expected_task_sha=TASK_SHA,
            expected_bank_sha=BANK_SHA,
        )
        self.assertEqual(metrics["successes"], 8)
        self.assertEqual(metrics["failures"], 56)
        self.assertEqual(metrics["success_elapsed_s"]["0.5"], 12.0)
        self.assertEqual(metrics["failed_elapsed_s"]["0.5"], 24.0)
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "policy config"):
            MODULE.validate_report(
                report(),
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
                expected_config_sha="0" * 64,
            )

    def test_count_and_pairing_tampering_fail_closed(self) -> None:
        count_tamper = report()
        count_tamper["valid_successes"] += 1
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "aggregates"):
            MODULE.validate_report(
                count_tamper,
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
            )

    def test_training_config_bank_and_bias_tampering_fail_closed(self) -> None:
        missing_config = report()
        del missing_config["rj45_training_config"]
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "contract differs"):
            MODULE.validate_report(
                missing_config,
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
            )

        biased_with_training_proof = report()
        biased_with_training_proof["training_held_grasp_bank_proof"] = {}
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "must be null"):
            MODULE.validate_report(
                biased_with_training_proof,
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
            )

        bc_report = report()
        bc_report["baseline_mode"] = "bc"
        bc_report["provenance"]["bc_actor"] = {"sha256": "8" * 64}
        bc_report["training_held_grasp_bank_proof"] = {
            "manifest_sha256": BANK_SHA,
            "selected_split": "train",
            "selected_count": 8,
            "selected_grasp_ids": [f"rj45-train-{index:02d}" for index in range(1, 9)],
            "selected_artifact_sha256": [str(index) * 64 for index in range(1, 9)],
            "manifest_split_counts": {"train": 8, "dev": 2, "final": 2},
            "provenance_source_task_match": True,
            "final_opened": False,
            "no_final_opened": True,
        }
        for outcome in bc_report["episode_outcomes"]:
            outcome["bc_actor_sha256"] = "8" * 64
        bc_metrics = MODULE.validate_report(
            bc_report,
            mode="bc",
            expected_source=SOURCE,
            expected_task_sha=TASK_SHA,
            expected_bank_sha=BANK_SHA,
            expected_bc_sha="8" * 64,
        )
        self.assertEqual(bc_metrics["training_bank_proof"]["selected_count"], 8)

        final_grasp = report()
        final_grasp["evaluation_held_grasp_bank_proof"]["selected_grasp_ids"] = [
            "rj45-final-01",
            "rj45-final-02",
        ]
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "contract differs"):
            MODULE.validate_report(
                final_grasp,
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
            )

        original = report()
        original_metrics = MODULE.validate_report(
            original,
            mode="biased_zero",
            expected_source=SOURCE,
            expected_task_sha=TASK_SHA,
            expected_bank_sha=BANK_SHA,
        )
        changed_bias = report()
        changed_bias["episode_outcomes"][0]["calibration_bias_translation_entry_m"][0] = 1.0
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "actual calibration bias"):
            MODULE.validate_report(
                changed_bias,
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
                expected_bias_by_id=original_metrics[
                    "calibration_bias_by_paired_condition"
                ],
            )

        pairing_tamper = report()
        pairing_tamper["episode_outcomes"][1]["paired_condition"] = pairing_tamper[
            "episode_outcomes"
        ][0]["paired_condition"]
        pairing_tamper["episode_outcomes"][1]["paired_condition_id"] = pairing_tamper[
            "episode_outcomes"
        ][0]["paired_condition_id"]
        pairing_tamper["episode_outcomes"][1]["environment_index"] = 0
        pairing_tamper["episode_outcomes"][1][
            "calibration_bias_translation_entry_m"
        ] = [0.0, 0.0, 0.0]
        pairing_tamper["episode_outcomes"][1][
            "calibration_bias_rotation_vector_entry_rad"
        ] = [0.0, 0.0, 0.0]
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "not unique"):
            MODULE.validate_report(
                pairing_tamper,
                mode="biased_zero",
                expected_source=SOURCE,
                expected_task_sha=TASK_SHA,
                expected_bank_sha=BANK_SHA,
            )

    def test_summary_sha_and_gate_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "summary.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(MODULE.CampaignAuditError, "summary SHA"):
                MODULE.summarize_campaign(
                    repo=SCRIPT.parents[1],
                    snapshot_root=Path(temporary),
                    campaign_summary=path,
                    campaign_summary_sha256="0" * 64,
                    plan=path,
                    plan_sha256=MODULE.sha256(path),
                    bc_training_report=path,
                    bc_training_report_sha256=MODULE.BC_REPORT_SHA256,
                )

        gate = MODULE.promotion_gate(
            baseline=12,
            ppo_only=[10, 10, 10],
            bc_then_ppo=[18, 18, 18],
            invalid_counts=[0] * 8,
            forbidden_peaks=[0.0] * 8,
        )
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "promotion gate"):
            MODULE.require_gate_match(
                {"promotion_gate": {**gate, "pass": False}, "promotion_allowed": False},
                gate,
            )

    def test_promotion_gate_keeps_three_seed_means_separate(self) -> None:
        gate = MODULE.promotion_gate(
            baseline=12,
            ppo_only=[10, 11, 12],
            bc_then_ppo=[18, 19, 20],
            invalid_counts=[0] * 8,
            forbidden_peaks=[0.0] * 8,
        )
        self.assertEqual(gate["ppo_only_successes"], [10, 11, 12])
        self.assertEqual(gate["ppo_only_mean"], 11.0)
        self.assertEqual(gate["bc_then_ppo_mean"], 19.0)
        self.assertTrue(gate["pass"])

    def test_checkpoint_status_cannot_replace_bundle_bytes_and_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection, evaluation, generated, plan_inputs = self._checkpoint_bundle(root)
            hashes = MODULE.validate_checkpoint_bundle(
                snapshot_root=root,
                inspection=inspection,
                report=evaluation,
                condition="ppo_only",
                seed=20261131,
                generated_config=generated,
                expected_source_map=SOURCE,
                plan_inputs=plan_inputs,
                expected_bc_sha="8" * 64,
                expected_bc_metadata={},
            )
            self.assertEqual(hashes["checkpoint"], inspection["checkpoint"]["sha256"])

            manifest = Path(inspection["checkpoint_manifest"]["path"])
            payload = json.loads(manifest.read_text())
            payload["training_completed"] = False
            manifest.write_text(json.dumps(payload) + "\n")
            with self.assertRaisesRegex(MODULE.CampaignAuditError, "bytes differ"):
                MODULE.validate_checkpoint_bundle(
                    snapshot_root=root,
                    inspection=inspection,
                    report=evaluation,
                    condition="ppo_only",
                    seed=20261131,
                    generated_config=generated,
                    expected_source_map=SOURCE,
                    plan_inputs=plan_inputs,
                    expected_bc_sha="8" * 64,
                    expected_bc_metadata={},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection, evaluation, generated, plan_inputs = self._checkpoint_bundle(root)
            provenance_path = Path(inspection["training_provenance"]["path"])
            provenance = json.loads(provenance_path.read_text())
            runner_name = Path(inspection["runner_config"]["path"]).name
            runner_keys = [
                key
                for key in provenance["source_and_config_sha256"]
                if Path(key).name == runner_name
            ]
            self.assertEqual(len(runner_keys), 1)
            del provenance["source_and_config_sha256"][runner_keys[0]]
            provenance_sha = self._write_json(provenance_path, provenance)
            manifest_path = Path(inspection["checkpoint_manifest"]["path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["training_provenance_sha256"] = provenance_sha
            manifest_sha = self._write_json(manifest_path, manifest)
            inspection["training_provenance"]["sha256"] = provenance_sha
            inspection["checkpoint_manifest"]["sha256"] = manifest_sha
            evaluation["provenance"]["verified_checkpoint_evidence"][
                "training_provenance_sha256"
            ] = provenance_sha
            evaluation["provenance"]["verified_checkpoint_evidence"][
                "checkpoint_manifest_sha256"
            ] = manifest_sha
            with self.assertRaisesRegex(MODULE.CampaignAuditError, "runner config"):
                MODULE.validate_checkpoint_bundle(
                    snapshot_root=root,
                    inspection=inspection,
                    report=evaluation,
                    condition="ppo_only",
                    seed=20261131,
                    generated_config=generated,
                    expected_source_map=SOURCE,
                    plan_inputs=plan_inputs,
                    expected_bc_sha="8" * 64,
                    expected_bc_metadata={},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection, evaluation, generated, plan_inputs = self._checkpoint_bundle(root)
            checkpoint_path = Path(inspection["checkpoint"]["path"])
            torch.save({"iter": 18}, checkpoint_path)
            checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
            inspection["checkpoint"]["sha256"] = checkpoint_sha
            inspection["checkpoint"]["payload_iteration"] = 18
            inspection["checkpoint"]["completed_updates"] = 19
            manifest_path = Path(inspection["checkpoint_manifest"]["path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["checkpoints"][checkpoint_path.name] = checkpoint_sha
            inspection["checkpoint_manifest"]["sha256"] = self._write_json(
                manifest_path, manifest
            )
            evaluation["provenance"]["checkpoint_sha256"] = checkpoint_sha
            evaluation["provenance"]["verified_checkpoint_evidence"][
                "checkpoint_manifest_sha256"
            ] = inspection["checkpoint_manifest"]["sha256"]
            with self.assertRaisesRegex(MODULE.CampaignAuditError, "payload iteration"):
                MODULE.validate_checkpoint_bundle(
                    snapshot_root=root,
                    inspection=inspection,
                    report=evaluation,
                    condition="ppo_only",
                    seed=20261131,
                    generated_config=generated,
                    expected_source_map=SOURCE,
                    plan_inputs=plan_inputs,
                    expected_bc_sha="8" * 64,
                    expected_bc_metadata={},
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inspection, evaluation, generated, plan_inputs = self._checkpoint_bundle(root)
            provenance_path = Path(inspection["training_provenance"]["path"])
            provenance = json.loads(provenance_path.read_text())
            del provenance["policy_initialization"]
            provenance_sha = self._write_json(provenance_path, provenance)
            manifest_path = Path(inspection["checkpoint_manifest"]["path"])
            manifest = json.loads(manifest_path.read_text())
            manifest["training_provenance_sha256"] = provenance_sha
            manifest_sha = self._write_json(manifest_path, manifest)
            inspection["training_provenance"]["sha256"] = provenance_sha
            inspection["checkpoint_manifest"]["sha256"] = manifest_sha
            evaluation["provenance"]["verified_checkpoint_evidence"][
                "training_provenance_sha256"
            ] = provenance_sha
            evaluation["provenance"]["verified_checkpoint_evidence"][
                "checkpoint_manifest_sha256"
            ] = manifest_sha
            with self.assertRaisesRegex(MODULE.CampaignAuditError, "initialization"):
                MODULE.validate_checkpoint_bundle(
                    snapshot_root=root,
                    inspection=inspection,
                    report=evaluation,
                    condition="ppo_only",
                    seed=20261131,
                    generated_config=generated,
                    expected_source_map=SOURCE,
                    plan_inputs=plan_inputs,
                    expected_bc_sha="8" * 64,
                    expected_bc_metadata={},
                )

    def test_prelaunch_requires_training_config_and_generated_config_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / "campaign"
            base_path = root / "configs/ppo-rj45-refinement-f.json"
            base = {"seed": 1, "num_steps_per_env": 16, "max_iterations": 96, "save_interval": 19}
            base_sha = self._write_json(base_path, base)
            training_sha = self._write_json(
                root / "configs/rj45-training-target-offset-v1.json",
                {"schema_version": 1},
            )
            generated: dict[str, dict[str, str]] = {}
            commands: dict[str, list[str]] = {}
            actor_path = "artifacts/bc/bc_actor_init.pt"
            actor_sha = "8" * 64
            frozen = {
                "scripts/evaluate_policy.py": SOURCE["scripts/evaluate_policy.py"],
                "configs/rj45-training-target-offset-v1.json": training_sha,
            }
            for seed in MODULE.SEEDS:
                path = campaign / f"configs/f-seed-{seed}.json"
                payload = {**base, "seed": seed, "num_steps_per_env": 32, "max_iterations": 200}
                digest = self._write_json(path, payload)
                relative = str(path.relative_to(root))
                generated[str(seed)] = {"path": relative, "sha256": digest}
                frozen[relative] = digest
                commands[f"ppo_only-{seed}"] = [
                    "train.py", "--num-envs", "256", "--max-iterations", "200",
                    "--config", relative, "--log-root", f"runs/ppo_only/{seed}",
                    "--rj45-training-config", "configs/rj45-training-target-offset-v1.json",
                    "--zero-init-actor-output",
                ]
                commands[f"bc_then_ppo-{seed}"] = [
                    "train.py", "--num-envs", "256", "--max-iterations", "200",
                    "--config", relative, "--log-root", f"runs/bc_then_ppo/{seed}",
                    "--rj45-training-config", "configs/rj45-training-target-offset-v1.json",
                    "--bc-actor-init", actor_path, "--bc-actor-init-sha256", actor_sha,
                ]
            prelaunch = {
                "schema_version": 1,
                "source_commit": MODULE.SOURCE_COMMIT,
                "final_seed_opened": False,
                "profile": {"name": "F", "source_config": str(base_path.relative_to(root)), "sha256": base_sha},
                "generated_configs": generated,
                "paired_training": {
                    "seeds": list(MODULE.SEEDS),
                    "num_envs": 256,
                    "steps_per_env": 32,
                    "updates": 200,
                    "save_interval": 19,
                },
                "commands": commands,
                "frozen_input_sha256": frozen,
            }
            self._write_json(campaign / "prelaunch-manifest.json", prelaunch)
            plan = {
                "input_sha256": {"scripts/evaluate_policy.py": SOURCE["scripts/evaluate_policy.py"]},
                "bc_actor_path": actor_path,
                "bc_actor_sha256": actor_sha,
            }
            with mock.patch.object(MODULE, "F_CONFIG_SHA256", base_sha), mock.patch.object(
                MODULE, "TRAINING_CONFIG_SHA256", training_sha
            ):
                records, _ = MODULE.validate_prelaunch(
                    snapshot_root=root,
                    campaign_root=campaign,
                    frozen_plan=plan,
                    expected_source_map=SOURCE,
                )
                self.assertEqual(set(records), set(MODULE.SEEDS))
                del prelaunch["frozen_input_sha256"][
                    "configs/rj45-training-target-offset-v1.json"
                ]
                self._write_json(
                    campaign / "prelaunch-manifest-tampered.json", prelaunch
                )
                (campaign / "prelaunch-manifest.json").write_text(
                    json.dumps(prelaunch) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(MODULE.CampaignAuditError, "prelaunch manifest"):
                    MODULE.validate_prelaunch(
                        snapshot_root=root,
                        campaign_root=campaign,
                        frozen_plan=plan,
                        expected_source_map=SOURCE,
                    )

                prelaunch["frozen_input_sha256"][
                    "configs/rj45-training-target-offset-v1.json"
                ] = training_sha
                prelaunch["commands"][f"bc_then_ppo-{MODULE.SEEDS[0]}"][2] = "128"
                self._write_json(campaign / "prelaunch-manifest.json", prelaunch)
                with self.assertRaisesRegex(MODULE.CampaignAuditError, "fixed arguments|common command"):
                    MODULE.validate_prelaunch(
                        snapshot_root=root,
                        campaign_root=campaign,
                        frozen_plan=plan,
                        expected_source_map=SOURCE,
                    )

    def test_bc_initialization_requires_actor_only_load(self) -> None:
        before = {
            "actor": "1" * 64,
            "critic": "2" * 64,
            "noise": "3" * 64,
            "optimizer": "4" * 64,
        }
        after = {**before, "actor": "5" * 64}
        initialization = {
            "mode": "bc_actor_warm_start",
            "output_dimension": 6,
            "artifact_file_sha256": "8" * 64,
            "bc_artifact_sha256": "8" * 64,
            "artifact_actor_state_sha256": "5" * 64,
            "loaded_component": "runner.alg.policy.actor",
            "hashes_before": before,
            "hashes_after": after,
            "bc_provenance": {"dataset_sha256": "6" * 64},
        }
        expected_metadata = {
            "actor_state_sha256": "5" * 64,
            "provenance": {"dataset_sha256": "6" * 64},
        }
        MODULE.validate_initialization(
            initialization,
            condition="bc_then_ppo",
            expected_bc_sha="8" * 64,
            expected_bc_metadata=expected_metadata,
        )
        initialization["hashes_after"]["critic"] = "9" * 64
        with self.assertRaisesRegex(MODULE.CampaignAuditError, "actor-only"):
            MODULE.validate_initialization(
                initialization,
                condition="bc_then_ppo",
                expected_bc_sha="8" * 64,
                expected_bc_metadata=expected_metadata,
            )


if __name__ == "__main__":
    unittest.main()
