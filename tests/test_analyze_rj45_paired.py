from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_rj45_paired.py"
SPEC = importlib.util.spec_from_file_location("analyze_rj45_paired", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
TRANSLATION_UM = (0.0, 50.0, 100.0, 150.0)
ROTATION_DEG = (0.0, 1.0 / 12.0, 1.0 / 6.0, 0.25)


def _nominal_proof() -> dict:
    return {"report_sha256": SHA_A, "trace_sha256": SHA_B, "trace_records": 10}


def _held_proof() -> dict:
    return {
        "artifact_sha256": SHA_C,
        "nominal_report_sha256": SHA_A,
        "nominal_trace_sha256": SHA_B,
        "selection_control_step": 100,
    }


def _policy_initialization(*, zero: bool = False) -> dict:
    return {
        "mode": "residual_actor_final_linear_zero" if zero else "rsl_rl_default",
        "zero_actor_output_layer": zero,
        "actor_module_path": "runner.alg.policy.actor",
        "final_layer_index": 6,
        "final_layer_class": "torch.nn.Linear",
        "input_dimension": 128,
        "output_dimension": 6,
        "pre_weight_sha256": SHA_A,
        "pre_bias_sha256": SHA_B,
        "post_weight_sha256": SHA_C,
        "post_bias_sha256": SHA_A,
        "weight_all_zero_after_initialization": zero,
        "bias_all_zero_after_initialization": zero,
        "deterministic_actor_test_input_count": 3,
        "deterministic_actor_output_all_zero": zero,
        "deterministic_actor_output_max_abs": 0.0 if zero else 0.2,
        "stochastic_policy_noise_modified": False,
        "method_reference": {
            "title": "Residual Policy Learning",
            "url": "https://arxiv.org/pdf/1812.06298",
            "section": "IV-A",
            "adaptation": "zero initialization of final residual actor mean layer only",
        } if zero else None,
    }


def _condition(environment_index: int, episode_index: int) -> dict:
    return {
        "evaluation_start": "held",
        "task_config_sha256": SHA_A,
        "base_evaluation_seed": 42,
        "environment_index": environment_index,
        "episode_index": episode_index,
        "calibration_bias_level": 0,
        "calibration_latent_case_seed": 1000 + episode_index * 2 + environment_index,
        "calibration_latent_case_seed_formula": "test formula",
    }


def _outcome(attempt: int, *, success: bool, valid: bool = True) -> dict:
    environment_index = attempt % 2
    episode_index = attempt // 2
    condition = _condition(environment_index, episode_index)
    translation_m = TRANSLATION_UM[attempt] * 1e-6
    rotation_rad = math.radians(ROTATION_DEG[attempt])
    return {
        "attempt_index": attempt,
        "environment_index": environment_index,
        "episode_index": episode_index,
        "paired_condition": condition,
        "paired_condition_id": json.dumps(
            condition, sort_keys=True, separators=(",", ":"), allow_nan=False
        ),
        "calibration_bias_translation_entry_m": [translation_m, 0.0, 0.0],
        "calibration_bias_rotation_vector_entry_rad": [rotation_rad, 0.0, 0.0],
        "valid": valid,
        "success": success,
        "termination_reason": "independent_success" if success else "episode_timeout",
        "invalid_reason": None if valid else "test_invalid",
        "held_initialization_passed": True,
        "valid_policy_input_step_count": 10,
        "final_depth_m": 0.012 if valid else None,
        "final_lateral_error_m": 0.00004 if valid else None,
        "final_orientation_error_rad": 0.001 if valid else None,
        "final_dwell_s": 0.4 if success else 0.0,
        "grip_contact_force_peak_n": 4.0 if valid else None,
        "insertion_contact_force_peak_n": 1.0 if valid else None,
        "insertion_contact_load_peak_n": 1.1 if valid else None,
        "forbidden_contact_load_peak_n": 0.0 if valid else None,
    }


def _report(mode: str, successes: set[int], *, invalid: set[int] | None = None) -> dict:
    invalid = invalid or set()
    outcomes = [
        _outcome(index, success=index in successes and index not in invalid, valid=index not in invalid)
        for index in range(4)
    ]
    provenance = {
        "input_immutability_verified": True,
        "source_sha256": {"scripts/evaluate_policy.py": SHA_B},
        "task_config_sha256": SHA_A,
        "runtime_config_sha256": SHA_A,
        "assets_config_sha256": SHA_A,
        "build_report_sha256": SHA_A,
        "physics_dt_s": 1.0 / 240.0,
        "control_decimation": 8,
        "control_dt_s": 1.0 / 30.0,
        "observation_noise": {"position_std_m": 0.0, "rotation_std_rad": 0.0},
        "generated_asset_sha256": {"scene.usd": SHA_A},
        "runtime_versions": {"isaac_sim": "5.1.0", "torch": "2.7.0"},
        "policy_config": "configs/ppo.json",
        "policy_config_sha256": SHA_B,
        "rj45_nominal_evidence": _nominal_proof(),
        "held_grasp_reset_proof": _held_proof(),
        "evaluation_inputs_reverified_at_end": True,
    }
    if mode == "learned":
        provenance.update({
            "checkpoint": "checkpoint.pt",
            "checkpoint_sha256": SHA_B,
            "verified_checkpoint_evidence": {
                "training_provenance_sha256": SHA_A,
                "checkpoint_manifest_sha256": SHA_B,
                "runtime_contract_verified_at_start": True,
                "rj45_nominal_evidence": _nominal_proof(),
                "held_grasp_reset_proof": _held_proof(),
                "training_num_envs": 64,
                "policy_initialization": _policy_initialization(),
            },
        })
    return {
        "rj45_evaluation_schema_version": 2,
        "baseline_mode": mode,
        "requested_attempts": 4,
        "attempts": 4,
        "episode_outcomes": outcomes,
        "evaluation_start": "held",
        "evaluation_horizon_s": 24.0,
        "evaluation_num_envs": 2,
        "per_environment_episode_quota": [2, 2],
        "per_environment_accepted_attempt_count": [2, 2],
        "seed": 42,
        "task_id": "rj45_insert",
        "task_model": {
            "scope": "test",
            "fixed_base": True,
            "num_joints": 62,
            "mating_frame": "measured_socket_entry_full_pose",
            "residual_action_frame": "measured_socket_entry",
            "pose_or_force_injection": False,
            "evaluation_stability_limits": {
                "assembled_depth_m": 0.012,
                "max_lateral_error_m": 0.00005,
                "max_orientation_error_rad": math.radians(0.5),
                "required_continuous_dwell_s": 0.4,
            },
        },
        "training_environment": {"diagnostic": "test"},
        "observation_source": "test estimator",
        "parallel_contact_evidence": {"report_sha256": SHA_A, "trace_sha256": SHA_B},
        "invalid_episodes": len(invalid),
        "invalid_reasons": {"test_invalid": len(invalid)} if invalid else {},
        "valid_episodes": 4 - len(invalid),
        "valid_successes": len(successes - invalid),
        "valid_failures": 4 - len(invalid) - len(successes - invalid),
        "success_rate_valid_episodes": (
            len(successes - invalid) / (4 - len(invalid)) if len(invalid) < 4 else None
        ),
        "paired_condition_key": ",".join(MODULE.PAIR_KEYS),
        "rj45_training_config": {
            "path": "configs/test.json",
            "sha256": SHA_A,
            "calibration_bias_level": 0,
            "action_mode": "right_only_6d",
            "observation_dimension": 79,
            "action_application_mode": "absolute_tip_pose_target_insert_only",
            "policy_active_phases": ["insert"],
        },
        "provenance": provenance,
    }


def _analyze(
    baseline: dict, learned: dict, *, seed: int = 9, allow_legacy: bool = False
) -> dict:
    return MODULE.analyze_reports(
        baseline,
        {"learned": learned},
        expected_attempts=4,
        bootstrap_resamples=200,
        bootstrap_seed=seed,
        allow_legacy_unproven_held=allow_legacy,
    )


class AnalyzeRj45PairedTests(unittest.TestCase):
    def test_e1_z1_policy_initialization_lineage_is_strict(self) -> None:
        e1 = _policy_initialization(zero=False)
        z1 = _policy_initialization(zero=True)
        MODULE._validate_policy_initialization_lineage(e1, copy.deepcopy(e1), "E1")
        MODULE._validate_policy_initialization_lineage(z1, copy.deepcopy(z1), "Z1")
        with self.assertRaisesRegex(ValueError, "differs from training provenance"):
            MODULE._validate_policy_initialization_lineage(z1, e1, "Z1")
        damaged = copy.deepcopy(z1)
        damaged["deterministic_actor_output_all_zero"] = False
        with self.assertRaisesRegex(ValueError, "does not prove zero residual actor output"):
            MODULE._validate_policy_initialization(damaged, "Z1")

        learned = _report("learned", {0})
        learned["provenance"]["verified_checkpoint_evidence"][
            "policy_initialization"
        ] = z1
        result = _analyze(_report("biased_zero", {0}), learned)
        self.assertTrue(
            result["experiment"]["reported_configuration_identifiers"]["learned"]
            ["policy_initialization"]["zero_actor_output_layer"]
        )

    def test_current_nominal_and_held_proofs_are_cross_checked(self) -> None:
        result = _analyze(_report("biased_zero", {0}), _report("learned", {0}))
        self.assertEqual(
            result["scope"]["verification_status"],
            "report_proof_hashes_consistent_inputs_not_rehashed",
        )

        missing_schema = _report("biased_zero", {0})
        del missing_schema["rj45_evaluation_schema_version"]
        with self.assertRaisesRegex(ValueError, "rj45_evaluation_schema_version=2"):
            _analyze(missing_schema, _report("learned", {0}))

        not_reverified = _report("learned", {0})
        not_reverified["provenance"]["evaluation_inputs_reverified_at_end"] = False
        with self.assertRaisesRegex(ValueError, "not reverified at end"):
            _analyze(_report("biased_zero", {0}), not_reverified)

        learned = _report("learned", {0})
        learned["provenance"]["held_grasp_reset_proof"]["artifact_sha256"] = SHA_A
        with self.assertRaisesRegex(ValueError, "proof hashes differ"):
            _analyze(_report("biased_zero", {0}), learned)

        learned = _report("learned", {0})
        learned["provenance"]["verified_checkpoint_evidence"][
            "held_grasp_reset_proof"
        ]["artifact_sha256"] = SHA_A
        with self.assertRaisesRegex(ValueError, "checkpoint evidence does not match"):
            _analyze(_report("biased_zero", {0}), learned)

    def test_proof_missing_mixed_and_legacy_modes_fail_closed(self) -> None:
        baseline = _report("biased_zero", {0})
        del baseline["provenance"]["held_grasp_reset_proof"]
        with self.assertRaisesRegex(ValueError, "partial RJ45 nominal/held proof"):
            _analyze(baseline, _report("learned", {0}))

        legacy_baseline = _report("biased_zero", {0})
        legacy_learned = _report("learned", {0})
        for report in (legacy_baseline, legacy_learned):
            del report["rj45_evaluation_schema_version"]
            del report["provenance"]["rj45_nominal_evidence"]
            del report["provenance"]["held_grasp_reset_proof"]
            del report["provenance"]["evaluation_inputs_reverified_at_end"]
            if report["baseline_mode"] == "learned":
                del report["provenance"]["verified_checkpoint_evidence"][
                    "policy_initialization"
                ]
        with self.assertRaisesRegex(ValueError, "--allow-legacy-unproven-held"):
            _analyze(copy.deepcopy(legacy_baseline), copy.deepcopy(legacy_learned))
        result = _analyze(legacy_baseline, legacy_learned, allow_legacy=True)
        self.assertEqual(result["scope"]["verification_status"], "legacy_unproven_from_reports")
        self.assertNotIn("verified held", result["scope"]["description"])

        empty_checkpoint_proof = copy.deepcopy(legacy_learned)
        empty_checkpoint_proof["provenance"]["verified_checkpoint_evidence"][
            "rj45_nominal_evidence"
        ] = {}
        with self.assertRaisesRegex(ValueError, "must be a lowercase SHA-256"):
            _analyze(
                copy.deepcopy(legacy_baseline),
                empty_checkpoint_proof,
                allow_legacy=True,
            )

        mixed_baseline = _report("biased_zero", {0})
        del mixed_baseline["rj45_evaluation_schema_version"]
        del mixed_baseline["provenance"]["rj45_nominal_evidence"]
        del mixed_baseline["provenance"]["held_grasp_reset_proof"]
        del mixed_baseline["provenance"]["evaluation_inputs_reverified_at_end"]
        with self.assertRaisesRegex(ValueError, "cannot be mixed"):
            _analyze(mixed_baseline, _report("learned", {0}), allow_legacy=True)

    def test_pair_order_does_not_change_result(self) -> None:
        baseline = _report("biased_zero", {0, 1})
        learned = _report("learned", {1, 2})
        expected = _analyze(copy.deepcopy(baseline), copy.deepcopy(learned))
        baseline["episode_outcomes"].reverse()
        learned["episode_outcomes"].reverse()
        actual = _analyze(baseline, learned)
        self.assertEqual(expected["paired_contrasts"], actual["paired_contrasts"])
        self.assertEqual(expected["paired_cases"], actual["paired_cases"])

    def test_all_condition_pairs_receive_a_contrast(self) -> None:
        result = MODULE.analyze_reports(
            _report("biased_zero", {0, 1}),
            {
                "A1": _report("learned", {1, 2}),
                "A2": _report("learned", {2, 3}),
            },
            expected_attempts=4,
            bootstrap_resamples=50,
            baseline_label="A0",
        )
        self.assertEqual(
            set(result["paired_contrasts"]),
            {"A1_minus_A0", "A2_minus_A0", "A2_minus_A1"},
        )

    def test_rejects_bias_mismatch_for_same_pair(self) -> None:
        baseline = _report("biased_zero", {0})
        learned = _report("learned", {0})
        learned["episode_outcomes"][0]["calibration_bias_translation_entry_m"][0] += 1e-9
        with self.assertRaisesRegex(ValueError, "calibration bias differs"):
            _analyze(baseline, learned)

    def test_rejects_missing_and_duplicate_pairs(self) -> None:
        baseline = _report("biased_zero", {0})
        learned = _report("learned", {0})
        learned["episode_outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "every requested episode outcome"):
            _analyze(copy.deepcopy(baseline), learned)

        learned = _report("learned", {0})
        learned["episode_outcomes"][3] = copy.deepcopy(learned["episode_outcomes"][0])
        learned["episode_outcomes"][3]["attempt_index"] = 3
        with self.assertRaisesRegex(ValueError, "duplicate paired condition"):
            _analyze(baseline, learned)

    def test_invalid_episode_stays_in_denominator(self) -> None:
        result = _analyze(
            _report("biased_zero", {0, 1}, invalid={3}),
            _report("learned", {0, 1, 2}, invalid={3}),
        )
        self.assertEqual(result["results"]["learned"]["valid_episodes"], 3)
        self.assertEqual(result["results"]["learned"]["successes_all_attempts"], 3)
        self.assertEqual(result["results"]["learned"]["success_rate_all_attempts"], 0.75)

    def test_zero_successes_and_bootstrap_are_supported_and_deterministic(self) -> None:
        first = _analyze(_report("biased_zero", set()), _report("learned", set()), seed=17)
        second = _analyze(_report("biased_zero", set()), _report("learned", set()), seed=17)
        comparison = first["paired_contrasts"]["learned_minus_baseline"]
        self.assertEqual(comparison["paired_success_rate_difference"], 0.0)
        self.assertEqual(comparison["paired_bootstrap_95_percentile_interval"], [0.0, 0.0])
        self.assertEqual(first["bootstrap"], second["bootstrap"])
        self.assertEqual(first["paired_contrasts"], second["paired_contrasts"])

        nonzero_first = _analyze(_report("biased_zero", {0, 1}), _report("learned", {1, 2}), seed=21)
        nonzero_second = _analyze(_report("biased_zero", {0, 1}), _report("learned", {1, 2}), seed=21)
        self.assertEqual(
            nonzero_first["paired_contrasts"],
            nonzero_second["paired_contrasts"],
        )

    def test_records_bounded_training_and_policy_config_differences(self) -> None:
        baseline = _report("biased_zero", {0})
        learned = _report("learned", {0})
        learned["rj45_training_config"]["path"] = "configs/reward-gamma.json"
        learned["rj45_training_config"]["sha256"] = SHA_C
        learned["provenance"]["policy_config"] = "configs/ppo-low-noise.json"
        learned["provenance"]["policy_config_sha256"] = SHA_C
        result = _analyze(baseline, learned)
        differences = result["experiment"]["reported_configuration_identifiers"]
        self.assertEqual(differences["learned"]["training_config_sha256"], SHA_C)
        self.assertEqual(differences["learned"]["policy_config_sha256"], SHA_C)

        learned["rj45_training_config"]["observation_dimension"] = 85
        with self.assertRaisesRegex(ValueError, "policy input/control contract differs"):
            _analyze(_report("biased_zero", {0}), learned)

    def test_rejects_environment_and_success_aggregate_tampering(self) -> None:
        learned = _report("learned", {0})
        learned["provenance"]["generated_asset_sha256"]["scene.usd"] = SHA_B
        with self.assertRaisesRegex(ValueError, "generated_asset_sha256"):
            _analyze(_report("biased_zero", {0}), learned)

        learned = _report("learned", {0})
        learned["task_model"]["pose_or_force_injection"] = True
        with self.assertRaisesRegex(ValueError, "pose_or_force_injection"):
            _analyze(_report("biased_zero", {0}), learned)

        learned = _report("learned", {0})
        learned["parallel_contact_evidence"]["report_sha256"] = SHA_C
        with self.assertRaisesRegex(ValueError, "parallel_contact_evidence"):
            _analyze(_report("biased_zero", {0}), learned)

        learned = _report("learned", set())
        learned["task_model"]["evaluation_stability_limits"]["required_continuous_dwell_s"] = 0.5
        with self.assertRaisesRegex(ValueError, "stability limits"):
            _analyze(_report("biased_zero", set()), learned)

        learned = _report("learned", set())
        learned["episode_outcomes"][0]["success"] = True
        with self.assertRaisesRegex(ValueError, "non-success termination reason"):
            _analyze(_report("biased_zero", set()), learned)

        learned = _report("learned", {0})
        learned["valid_successes"] = 0
        with self.assertRaisesRegex(ValueError, "valid_successes aggregate"):
            _analyze(_report("biased_zero", {0}), learned)

    def test_boundary_values_enter_next_half_open_bin(self) -> None:
        result = _analyze(_report("biased_zero", {0}), _report("learned", {0}))
        translation = result["evaluation_bias_coverage"]["translation"]
        rotation = result["evaluation_bias_coverage"]["rotation"]
        self.assertEqual(sum(item["sample_count"] for item in translation["bins"]), 4)
        self.assertEqual(sum(item["sample_count"] for item in rotation["bins"]), 4)
        self.assertEqual(translation["unit"], "um")
        self.assertEqual(rotation["unit"], "deg")
        self.assertEqual(MODULE._bin_index(50.0 - 1e-10, MODULE.DEFAULT_TRANSLATION_EDGES_UM), 0)
        self.assertEqual(MODULE._bin_index(50.0, MODULE.DEFAULT_TRANSLATION_EDGES_UM), 1)

    def test_historical_training_bundle_bytes_are_verified(self) -> None:
        root = Path(__file__).resolve().parents[1]
        run = root / "artifacts/rj45-target-offset-pilot/logs/rj45-v1/2026-10-26_15-55-36"
        report = json.loads(
            (root / "artifacts/rj45-target-offset-pilot/validation-learned42.json").read_text()
        )
        result = MODULE._verify_training_bundle(
            "standard79",
            report,
            checkpoint_path=run / "model_19.pt",
            manifest_path=run / "checkpoint_manifest.json",
            training_provenance_path=run / "training_provenance.json",
            runner_config_path=run / "runner_config.json",
        )
        self.assertEqual(
            result["status"],
            "checkpoint_manifest_provenance_runner_bytes_verified",
        )
        self.assertEqual(result["training_seed"], 20261026)
        self.assertEqual(result["training_updates"], 20)
        with tempfile.TemporaryDirectory() as directory:
            tampered = Path(directory) / "model_19.pt"
            tampered.write_bytes((run / "model_19.pt").read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes do not match"):
                MODULE._verify_training_bundle(
                    "standard79",
                    report,
                    checkpoint_path=tampered,
                    manifest_path=run / "checkpoint_manifest.json",
                    training_provenance_path=run / "training_provenance.json",
                    runner_config_path=run / "runner_config.json",
                )

    def test_training_source_lineage_requires_full_package_and_train_entrypoint(self) -> None:
        evaluation = {
            "src/rby1_assembly/a.py": SHA_A,
            "src/rby1_assembly/b.py": SHA_B,
            "scripts/evaluate_policy.py": SHA_C,
        }
        training = {
            "src/rby1_assembly/a.py": SHA_A,
            "src/rby1_assembly/b.py": SHA_B,
            "scripts/train.py": SHA_C,
        }
        MODULE._validate_training_source_lineage(evaluation, training, "test")
        missing = dict(training)
        del missing["src/rby1_assembly/b.py"]
        with self.assertRaisesRegex(ValueError, "source path sets differ"):
            MODULE._validate_training_source_lineage(evaluation, missing, "test")
        no_train = dict(training)
        del no_train["scripts/train.py"]
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            MODULE._validate_training_source_lineage(evaluation, no_train, "test")

    def test_output_cannot_overwrite_bundle_and_inputs_are_rehashed(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pilot = root / "artifacts/rj45-target-offset-pilot"
        run = pilot / "logs/rj45-v1/2026-10-26_15-55-36"
        with self.assertRaisesRegex(ValueError, "must not overwrite"):
            MODULE.main([
                "--baseline",
                str(pilot / "validation-biased-zero42.json"),
                "--condition",
                f"A1={pilot / 'validation-learned42.json'}",
                "--checkpoint-artifact",
                f"A1={run / 'model_19.pt'}",
                "--output",
                str(run / "model_19.pt"),
            ])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text("before")
            expected = {path: MODULE._sha_file(path)}
            path.write_text("after")
            with self.assertRaisesRegex(ValueError, "changed during analysis"):
                MODULE._verify_path_hashes(expected)


if __name__ == "__main__":
    unittest.main()
