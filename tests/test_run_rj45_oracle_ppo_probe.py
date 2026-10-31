from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_rj45_oracle_ppo_probe.py"
ISAAC_WAS_LOADED = "isaaclab" in sys.modules
SPEC = importlib.util.spec_from_file_location("run_rj45_oracle_ppo_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
ROOT = Path(__file__).resolve().parents[1]


def valid_budget_admission() -> dict[str, object]:
    return {
        "budget_cap_usd": 700.0,
        "new_run_reservation_usd": 12.0,
        "external_guard_minutes": 75.0,
        "private_evidence_sha256": "a" * 64,
        "locally_verified": True,
    }


class OraclePpoProbeTests(unittest.TestCase):
    def test_actual_terminal_evaluation_serializes_through_dataclass_contract(self) -> None:
        try:
            import h5py  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["h5py"] = types.ModuleType("h5py")
        from rby1_assembly.dataset import GroundTruthMetrics, TerminalEvaluation

        terminal = TerminalEvaluation(
            success=False,
            valid=True,
            termination_reason="timeout",
            evaluator_source="simulator_ground_truth_independent",
            evaluator_version="task-evaluator-v1",
            terminal_timestamp_s=24.0,
            metrics=GroundTruthMetrics(
                actual_depth_m=0.001,
                lateral_error_m=0.002,
                orientation_error_rad=0.003,
                force_peak_n=4.0,
                grasp_both=True,
                forbidden_contact=False,
            ),
            criteria={"depth": False, "force": True},
        )
        payload = RUNNER.serialize_terminal_evaluation(terminal)
        self.assertEqual(payload, terminal.as_dict())
        self.assertEqual(payload["terminal_timestamp_s"], 24.0)
        self.assertEqual(payload["termination_reason"], "timeout")
        self.assertNotIn("completion_time_s", payload)

    def test_close_system_exit_zero_preserves_main_failure_exit_code(self) -> None:
        class FakeApp:
            def close(self) -> None:
                raise SystemExit(0)

        self.assertEqual(RUNNER._close_simulation_app(FakeApp(), 2), 2)

    def test_main_failure_exit_survives_fake_app_close_system_exit_zero(self) -> None:
        class FakeApp:
            def close(self) -> None:
                raise SystemExit(0)

        class FakeAppLauncher:
            def __init__(self, _settings) -> None:
                self.app = FakeApp()

        isaaclab_module = types.ModuleType("isaaclab")
        isaaclab_app_module = types.ModuleType("isaaclab.app")
        isaaclab_app_module.AppLauncher = FakeAppLauncher
        protocol_record = {
            "payload": {
                "fixed_condition": {"base_seed": RUNNER.BASE_SEED},
                "budget_and_stop": {
                    "absolute_deadline_utc": "2099-01-01T00:00:00Z",
                    "mode_wall_time_limit_s": {
                        "zero": RUNNER.CONTROL_MODE_WALL_S
                    },
                }
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            with (
                mock.patch.object(RUNNER, "validate_protocol", return_value=protocol_record),
                mock.patch.object(RUNNER, "validate_runtime_manifest", return_value={}),
                mock.patch.object(RUNNER, "run_probe", side_effect=RuntimeError("probe failed")),
                mock.patch.dict(
                    sys.modules,
                    {"isaaclab": isaaclab_module, "isaaclab.app": isaaclab_app_module},
                ),
            ):
                result = RUNNER.main(
                    [
                        "--runtime-root", str(root),
                        "--protocol", str(root / "protocol.json"),
                        "--protocol-sha256", "0" * 64,
                        "--mode", "zero",
                        "--output-dir", str(output),
                    ]
                )
            self.assertEqual(result, 2)
            failure = json.loads(
                (root / "output-report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["status"], "invalid")
            self.assertEqual(failure["failure_stage"], "physics_and_learning")

    def test_import_and_protocol_write_are_cpu_only(self) -> None:
        self.assertEqual("isaaclab" in sys.modules, ISAAC_WAS_LOADED)
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.json"
            result = RUNNER.write_protocol(protocol, ROOT, updates=32)
            self.assertEqual(result["file_sha256"], RUNNER.file_sha256(protocol))
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            self.assertEqual(payload["ppo"]["transitions"], 4096)
            self.assertEqual(payload["diagnostic_scope"]["required_modes"], list(RUNNER.MODES))
            self.assertTrue(payload["diagnostic_scope"]["dev_and_final_unopened"])
            self.assertEqual(payload["observation_groups"]["standard79"]["dimensions"], {"policy": 79, "critic": 92})
            self.assertEqual(payload["observation_groups"]["oracle92"]["dimensions"], {"policy": 92, "critic": 92})
            self.assertEqual(payload["fixed_condition"]["calibration_case_seed"], 66384140)
            self.assertEqual(payload["fixed_condition"]["base_seed"], RUNNER.BASE_SEED)
            self.assertEqual(payload["fixed_condition"]["split"], "train")
            self.assertEqual(payload["fixed_condition"]["grasp_id"], "rj45-train-02")
            self.assertEqual(payload["frozen_inputs"]["runtime_manifest_sha256"], RUNNER.EXPECTED_RUNTIME_MANIFEST_SHA256)
            self.assertEqual(len(payload["frozen_inputs"]["runtime_source_file_sha256"]), 35)
            self.assertEqual(payload["budget_and_stop"]["combined_conservative_reservation_usd"], 699.31822)
            self.assertIsNone(payload["budget_and_stop"]["absolute_deadline_utc"])
            self.assertEqual(payload["physics_contract"]["execution_device"], "cuda:0")
            self.assertEqual(
                payload["ppo"]["diagnostic_intermediate_checkpoint_updates"],
                [8, 16, 24],
            )
            self.assertIn(
                "never eligible",
                payload["ppo"]["intermediate_checkpoint_policy"],
            )
            self.assertEqual(
                payload["budget_and_stop"]["mode_wall_time_limit_s"],
                {
                    "zero": 300.0,
                    "reference": 300.0,
                    "standard79": 1800.0,
                    "oracle92": 1800.0,
                },
            )
            self.assertIn("32 updates/4096 transitions", payload["ppo"]["interpretation"])

    def test_update_jsonl_is_durable_before_a_later_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "updates.jsonl"
            first = {
                "update": 8,
                "metrics": {"loss": 0.25},
                "diagnostic_checkpoint": {
                    "path": "diagnostic_checkpoint_0008.pt",
                    "sha256": "b" * 64,
                    "eligible_for_selection": False,
                },
            }
            RUNNER.append_jsonl_record(trace, first)
            try:
                raise TimeoutError("later training timeout")
            except TimeoutError:
                pass
            self.assertTrue(trace.is_file())
            rows = [
                json.loads(line)
                for line in trace.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(rows, [first])

            with self.assertRaisesRegex(
                RUNNER.OraclePpoProbeError, "not finite JSON"
            ):
                RUNNER.append_jsonl_record(
                    trace, {"update": 9, "loss": float("nan")}
                )
            self.assertEqual(
                len(trace.read_text(encoding="utf-8").splitlines()), 1
            )

    def test_alternate_train_seed_roundtrips_and_tampering_fails_closed(self) -> None:
        alternate_seed = RUNNER.BASE_SEED + 8
        expected_calibration = alternate_seed + (
            RUNNER.CALIBRATION_CASE_SEED - RUNNER.BASE_SEED
        )
        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.json"
            RUNNER.write_protocol(
                protocol,
                ROOT,
                updates=32,
                base_seed=alternate_seed,
                budget_admission=valid_budget_admission(),
            )
            digest = RUNNER.file_sha256(protocol)
            validated = RUNNER.validate_protocol(protocol, digest, ROOT)
            fixed = validated["payload"]["fixed_condition"]
            self.assertEqual(fixed["base_seed"], alternate_seed)
            self.assertEqual(fixed["calibration_case_seed"], expected_calibration)
            self.assertEqual(fixed["environment_index"], 0)
            self.assertEqual(fixed["episode_index"], 1)
            self.assertEqual(fixed["grasp_id"], "rj45-train-02")
            budget = validated["payload"]["budget_and_stop"]
            self.assertEqual(budget["new_run_reservation_usd"], 12.0)
            self.assertEqual(budget["external_guard_minutes"], 75.0)
            self.assertEqual(budget["budget_admission"], valid_budget_admission())
            self.assertTrue(budget["historical_reservation_not_invoice"])
            self.assertNotIn("prior_conservative_reservation_usd", budget)
            self.assertNotIn("combined_conservative_reservation_usd", budget)

            tampered = validated["payload"]
            tampered["fixed_condition"]["grasp_id"] = "rj45-train-03"
            protocol.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "payload changed"):
                RUNNER.validate_protocol(
                    protocol, RUNNER.file_sha256(protocol), ROOT
                )

            calibration_protocol = Path(directory) / "calibration-protocol.json"
            RUNNER.write_protocol(
                calibration_protocol,
                ROOT,
                updates=32,
                base_seed=alternate_seed,
                budget_admission=valid_budget_admission(),
            )
            calibration_payload = json.loads(
                calibration_protocol.read_text(encoding="utf-8")
            )
            calibration_payload["fixed_condition"]["calibration_case_seed"] += 1
            calibration_protocol.write_text(
                json.dumps(calibration_payload), encoding="utf-8"
            )
            with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "payload changed"):
                RUNNER.validate_protocol(
                    calibration_protocol,
                    RUNNER.file_sha256(calibration_protocol),
                    ROOT,
                )

            self.assertEqual(
                RUNNER.seed_policy_rng(
                    RUNNER.POLICY_TRAINING_SEED, physics_seed=alternate_seed
                )["physics_seed_unchanged"],
                alternate_seed,
            )
            self.assertEqual(
                RUNNER.reset_evaluation_rng(
                    "cpu", physics_seed=alternate_seed
                )["seeded"]["physics_seed_unchanged"],
                alternate_seed,
            )

    def test_budget_admission_rejects_tampering_excess_and_invalid_types(self) -> None:
        cases = []
        extra = valid_budget_admission()
        extra["unexpected"] = 1
        cases.append((extra, "keys"))
        over_cap = valid_budget_admission()
        over_cap["budget_cap_usd"] = 701.0
        cases.append((over_cap, "cap"))
        over_reservation = valid_budget_admission()
        over_reservation["new_run_reservation_usd"] = 12.01
        cases.append((over_reservation, "reservation"))
        over_guard = valid_budget_admission()
        over_guard["external_guard_minutes"] = 75.01
        cases.append((over_guard, "guard"))
        bad_digest = valid_budget_admission()
        bad_digest["private_evidence_sha256"] = "G" * 64
        cases.append((bad_digest, "digest"))
        not_verified = valid_budget_admission()
        not_verified["locally_verified"] = 1
        cases.append((not_verified, "locally verified"))
        nonfinite = valid_budget_admission()
        nonfinite["new_run_reservation_usd"] = float("nan")
        cases.append((nonfinite, "finite"))
        for admission, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, message):
                    RUNNER.protocol_payload(ROOT, budget_admission=admission)

        with tempfile.TemporaryDirectory() as directory:
            protocol = Path(directory) / "protocol.json"
            RUNNER.write_protocol(
                protocol,
                ROOT,
                updates=32,
                budget_admission=valid_budget_admission(),
            )
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["budget_and_stop"]["new_run_reservation_usd"] = 11.0
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "payload changed"):
                RUNNER.validate_protocol(
                    protocol, RUNNER.file_sha256(protocol), ROOT
                )

    def test_base_seed_validation_rejects_bool_negative_and_non_bank_stride(self) -> None:
        for invalid in (True, -8, RUNNER.BASE_SEED + 1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(
                    RUNNER.OraclePpoProbeError, "nonnegative exact int divisible by 8"
                ):
                    RUNNER.protocol_payload(ROOT, base_seed=invalid)

    def test_mode_specific_time_guard_uses_control_and_learning_limits(self) -> None:
        protocol = RUNNER.protocol_payload(ROOT)
        limits = protocol["budget_and_stop"]["mode_wall_time_limit_s"]
        self.assertEqual(limits["zero"], RUNNER.CONTROL_MODE_WALL_S)
        self.assertEqual(limits["reference"], RUNNER.CONTROL_MODE_WALL_S)
        self.assertEqual(limits["standard79"], RUNNER.LEARNING_MODE_WALL_S)
        self.assertEqual(limits["oracle92"], RUNNER.LEARNING_MODE_WALL_S)
        for mode in RUNNER.MODES:
            args = types.SimpleNamespace(
                execution_started_monotonic=10.0,
                execution_mode_wall_s=limits[mode],
                execution_deadline_epoch=10_000.0,
            )
            with (
                mock.patch.object(RUNNER.time, "monotonic", return_value=10.0 + limits[mode] + 0.1),
                mock.patch.object(RUNNER.time, "time", return_value=0.0),
            ):
                with self.assertRaisesRegex(TimeoutError, "wall-time"):
                    RUNNER.assert_execution_time(args)

    def test_protocol_tamper_and_source_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            RUNNER.write_protocol(protocol, ROOT, updates=32)
            digest = RUNNER.file_sha256(protocol)
            RUNNER.validate_protocol(protocol, digest, ROOT)
            payload = json.loads(protocol.read_text(encoding="utf-8"))
            payload["ppo"]["gamma"] = 0.9
            protocol.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "SHA-256"):
                RUNNER.validate_protocol(protocol, digest, ROOT)
            with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "updates"):
                RUNNER.protocol_payload(ROOT, updates=31)
            scheduled = RUNNER.protocol_payload(
                ROOT, updates=32, deadline_utc="2026-11-01T00:00:00Z"
            )
            self.assertEqual(
                scheduled["budget_and_stop"]["absolute_deadline_utc"],
                "2026-11-01T00:00:00Z",
            )
            with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "UTC Z"):
                RUNNER.protocol_payload(
                    ROOT, updates=32, deadline_utc="2026-11-01T00:00:00+00:00"
                )
            extended = RUNNER.protocol_payload(ROOT, updates=96)
            self.assertEqual(extended["ppo"]["transitions"], 12288)
            self.assertIn(
                "96 updates/12288 transitions", extended["ppo"]["interpretation"]
            )
            self.assertIn(
                "single_env_128x96", extended["ppo"]["distinguished_from_prior_campaign"]
            )

    def test_execution_rejects_unscheduled_protocol_before_isaac_import(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = root / "protocol.json"
            RUNNER.write_protocol(protocol, ROOT, updates=32)
            result = RUNNER.main(
                [
                    "--runtime-root", str(ROOT),
                    "--protocol", str(protocol),
                    "--protocol-sha256", RUNNER.file_sha256(protocol),
                    "--mode", "zero",
                    "--output-dir", str(root / "output"),
                ]
            )
            self.assertEqual(result, 2)
            report = root / "output-report.json"
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(payload["failure_stage"], "protocol_validation")
            self.assertIn("unscheduled", payload["error"]["message"])

    def test_runner_config_is_fixed_one_env_diagnostic_contract(self) -> None:
        protocol = RUNNER.protocol_payload(ROOT, updates=32)
        for mode in RUNNER.LEARNING_MODES:
            config = RUNNER.runner_config(mode, protocol)
            self.assertEqual(config["num_steps_per_env"], 128)
            self.assertEqual(config["max_iterations"], 32)
            self.assertEqual(config["obs_groups"], {"policy": ["policy"], "critic": ["critic"]})
            self.assertEqual(config["algorithm"]["schedule"], "fixed")
            self.assertEqual(config["algorithm"]["num_mini_batches"], 1)
            self.assertEqual(config["policy"]["init_noise_std"], 0.1)
        with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "only"):
            RUNNER.runner_config("zero", protocol)

    def test_policy_rng_is_reproducible_and_separate_from_physics_seed(self) -> None:
        first = RUNNER.seed_policy_rng(RUNNER.POLICY_TRAINING_SEED)
        values_a = (np.random.rand(3), torch.rand(3))
        second = RUNNER.seed_policy_rng(RUNNER.POLICY_TRAINING_SEED)
        values_b = (np.random.rand(3), torch.rand(3))
        np.testing.assert_array_equal(values_a[0], values_b[0])
        torch.testing.assert_close(values_a[1], values_b[1], rtol=0.0, atol=0.0)
        self.assertEqual(first, second)
        self.assertEqual(first["physics_seed_unchanged"], RUNNER.BASE_SEED)

    def test_evaluation_rng_is_identical_after_different_constructor_consumption(self) -> None:
        fingerprints = []
        for random_draws in (79, 92):
            RUNNER.seed_policy_rng(RUNNER.POLICY_TRAINING_SEED)
            np.random.rand(random_draws)
            torch.rand(random_draws)
            record = RUNNER.reset_evaluation_rng("cpu")
            fingerprints.append(record["state_sha256_before_environment_reset"])
            self.assertEqual(record["seeded"]["seed"], RUNNER.EVALUATION_SEED)
            self.assertEqual(
                record["seeded"]["physics_seed_unchanged"], RUNNER.BASE_SEED
            )
        self.assertEqual(fingerprints[0], fingerprints[1])

    def test_critic_init_is_independent_of_actor_width_and_rng_is_restored(self) -> None:
        class Policy(torch.nn.Module):
            def __init__(self, actor_width: int) -> None:
                super().__init__()
                self.actor = torch.nn.Sequential(
                    torch.nn.Linear(actor_width, 8), torch.nn.ELU(), torch.nn.Linear(8, 6)
                )
                self.critic = torch.nn.Sequential(
                    torch.nn.Linear(92, 8), torch.nn.ELU(), torch.nn.Linear(8, 1)
                )

        hashes = []
        post_rng = []
        for width in (79, 92):
            RUNNER.seed_policy_rng(RUNNER.POLICY_TRAINING_SEED)
            policy = Policy(width)
            before = RUNNER.rng_state_sha256()
            proof = RUNNER.reinitialize_critic(
                policy, seed=RUNNER.CRITIC_INITIALIZATION_SEED, device="cpu"
            )
            after = RUNNER.rng_state_sha256()
            self.assertEqual(before, after)
            hashes.append(proof["state_sha256"])
            RUNNER.seed_policy_rng(RUNNER.POLICY_TRAINING_SEED)
            post_rng.append(RUNNER.rng_state_sha256())
        self.assertEqual(hashes[0], hashes[1])
        self.assertEqual(post_rng[0], post_rng[1])

    def test_paired_actor_shares_79d_projection_and_zeros_oracle_columns(self) -> None:
        def actor(width: int) -> torch.nn.Sequential:
            return torch.nn.Sequential(
                torch.nn.Linear(width, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 256),
                torch.nn.ELU(),
                torch.nn.Linear(256, 128),
                torch.nn.ELU(),
                torch.nn.Linear(128, 6),
            )

        class Policy(torch.nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.actor = actor(width)

        proofs = []
        policies = []
        for mode, width in (("standard79", 79), ("oracle92", 92)):
            policy = Policy(width)
            proof = RUNNER.paired_initialize_actor(
                policy,
                mode,
                seed=RUNNER.ACTOR_INITIALIZATION_SEED,
                device="cpu",
                baseline_builder=lambda: actor(79),
            )
            proofs.append(proof)
            policies.append(policy)
        self.assertEqual(
            proofs[0]["common_79d_projection_sha256"],
            proofs[1]["common_79d_projection_sha256"],
        )
        oracle_first = next(
            module for module in policies[1].actor.modules() if isinstance(module, torch.nn.Linear)
        )
        self.assertEqual(
            int(torch.count_nonzero(oracle_first.weight[:, 79:])), 0
        )
        self.assertTrue(oracle_first.weight.requires_grad)

    def test_zero_actor_contract_checks_mean_and_std(self) -> None:
        class FakeTensorDict(dict):
            def __init__(self, values, batch_size):
                super().__init__(values)
                self.batch_size = batch_size

        class Policy(torch.nn.Module):
            def __init__(self, width: int) -> None:
                super().__init__()
                self.actor = torch.nn.Linear(width, 6)
                torch.nn.init.zeros_(self.actor.weight)
                torch.nn.init.zeros_(self.actor.bias)
                self.action_std = torch.full((6,), 0.1)

            def act_inference(self, observations: FakeTensorDict) -> torch.Tensor:
                return self.actor(observations["policy"])

        for mode, width in (("standard79", 79), ("oracle92", 92)):
            proof = RUNNER.verify_zero_actor_contract(
                Policy(width), mode, device="cpu", tensor_dict_type=FakeTensorDict
            )
            self.assertTrue(proof["mean_action_all_zero"])
            self.assertAlmostEqual(proof["noise"]["mean"], 0.1)

    def test_fixed_reset_forces_episode_one_and_keeps_true_count(self) -> None:
        class Base:
            def __init__(self) -> None:
                self._training_initialized = False
                self.device = "cpu"
                self.robot = type(
                    "Robot", (), {"_ALL_INDICES": torch.tensor([0], dtype=torch.long)}
                )()
                self._training_episode_index = torch.tensor([0], dtype=torch.long)
                self._training_grasp_index = torch.tensor([0], dtype=torch.long)

            def _reset_idx(self, env_ids):
                if env_ids is None:
                    env_ids = self.robot._ALL_INDICES
                self._training_episode_index[env_ids] += 1
                self._training_grasp_index[env_ids] = self._training_episode_index[env_ids] % 8

        fixed = RUNNER.make_fixed_condition_env_type(Base)()
        ids = torch.tensor([0], dtype=torch.long)
        fixed._reset_idx([0])
        self.assertEqual(fixed._training_episode_index.tolist(), [1])
        self.assertEqual(fixed.diagnostic_true_episode_count, 0)
        fixed._training_initialized = True
        fixed._training_episode_index[:] = 17
        fixed._reset_idx(ids)
        self.assertEqual(fixed._training_episode_index.tolist(), [1])
        self.assertEqual(fixed._training_grasp_index.tolist(), [1])
        self.assertEqual(fixed.diagnostic_true_episode_count, 1)
        fixed._reset_idx(None)
        self.assertEqual(fixed._training_episode_index.tolist(), [1])
        self.assertEqual(fixed.diagnostic_true_episode_count, 2)
        with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "_ALL_INDICES"):
            fixed._reset_idx([])

    def test_observation_proof_enforces_no_gt_leak_and_shapes(self) -> None:
        base = torch.arange(79, dtype=torch.float32).reshape(1, 79)
        oracle = torch.arange(13, dtype=torch.float32).reshape(1, 13)
        critic = torch.cat((base, oracle), dim=1)
        standard = RUNNER._observation_proof({"policy": base.clone(), "critic": critic}, "standard79")
        self.assertFalse(standard["actor_ground_truth_access"])
        privileged = RUNNER._observation_proof({"policy": critic.clone(), "critic": critic}, "oracle92")
        self.assertTrue(privileged["actor_ground_truth_access"])
        leaked = base.clone()
        leaked[0, 0] += 1
        with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "leaked"):
            RUNNER._observation_proof({"policy": leaked, "critic": critic}, "standard79")

    def test_update_audit_rejects_nonfinite_and_actor_hash_detects_change(self) -> None:
        record = RUNNER.finite_update_record(3, (torch.tensor(1.25), 0.5))
        self.assertEqual(record["update"], 3)
        self.assertEqual(record["metrics"]["value_0"], 1.25)
        with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "non-finite"):
            RUNNER.finite_update_record(1, {"loss": float("nan")})
        storage = type(
            "Storage",
            (),
            {"returns": torch.ones((2, 1)), "advantages": torch.zeros((2, 1))},
        )()
        self.assertEqual(
            RUNNER.assert_finite_rollout_targets(storage),
            {"returns": 2, "advantages": 2},
        )
        storage.advantages[0] = float("inf")
        with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "advantages"):
            RUNNER.assert_finite_rollout_targets(storage)
        actor = torch.nn.Linear(3, 2)
        before = RUNNER._parameter_sha256(actor)
        with torch.no_grad():
            actor.weight.add_(1.0)
        self.assertNotEqual(before, RUNNER._parameter_sha256(actor))

        class Policy:
            std = torch.full((6,), 0.1)

        self.assertAlmostEqual(RUNNER._policy_noise_summary(Policy())["mean"], 0.1)
        Policy.std = torch.tensor([0.1, float("nan")])
        with self.assertRaisesRegex(RUNNER.OraclePpoProbeError, "standard deviation"):
            RUNNER._policy_noise_summary(Policy())

        finite_policy = type(
            "FinitePolicy",
            (),
            {
                "actor": torch.nn.Linear(3, 2),
                "critic": torch.nn.Linear(4, 1),
            },
        )()
        parameter_proof = RUNNER.assert_finite_policy_parameters(finite_policy)
        self.assertEqual(parameter_proof["actor_parameters"], 2)
        self.assertEqual(parameter_proof["critic_parameters"], 2)
        with torch.no_grad():
            finite_policy.critic.weight[0, 0] = float("nan")
        with self.assertRaisesRegex(
            RUNNER.OraclePpoProbeError, "critic parameter"
        ):
            RUNNER.assert_finite_policy_parameters(finite_policy)


if __name__ == "__main__":
    unittest.main()
