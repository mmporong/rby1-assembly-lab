from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_rj45_g2_physics_probe.py"
ISAAC_WAS_LOADED = "isaaclab" in sys.modules
SPEC = importlib.util.spec_from_file_location("rj45_g2_physics_probe_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class G2PhysicsProbeTests(unittest.TestCase):
    def test_import_has_no_isaac_or_evaluate_policy_side_effect(self) -> None:
        self.assertEqual("isaaclab" in sys.modules, ISAAC_WAS_LOADED)
        self.assertNotIn("evaluate_policy", sys.modules)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("import evaluate_policy", source)
        self.assertNotIn("from evaluate_policy", source)
        self.assertNotIn("AssemblyGeometry", source)
        self.assertNotIn("geometry_metrics_from_object_poses", source)

    def _offline_input(self, root: Path) -> tuple[Path, Path, str]:
        observation = np.arange(418 * 79, dtype=np.float32).reshape(418, 79) / 1000.0
        action = np.arange(418 * 6, dtype=np.float32).reshape(418, 6) / 10000.0
        valid_mask = np.zeros(418, dtype=np.bool_)
        valid_mask[:395] = True
        npz = root / "input.npz"
        np.savez(npz, observation=observation, action=action, valid_mask=valid_mask)
        npz_sha = RUNNER.file_sha256(npz)
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "artifact_kind": "train_only_fixed_condition_offline_bc_probe_input",
                    "condition": {
                        "bias_case_id": "env-0-episode-1",
                        "grasp_id": RUNNER.GRASP_ID,
                        "seed": RUNNER.CALIBRATION_CASE_SEED,
                        "split": "train",
                    },
                    "lineage": {
                        "bank_sha256": RUNNER.EXPECTED_BANK_SHA256,
                        "source_sha256": RUNNER.EXPECTED_SOURCE_CONTRACT_SHA256,
                        "task_sha256": RUNNER.EXPECTED_TASK_SHA256,
                        "training_sha256": RUNNER.EXPECTED_TRAINING_SHA256,
                    },
                    "npz": {"sha256": npz_sha},
                    "arrays": {
                        "observation": {"sha256": RUNNER.manifest_array_sha256(observation)},
                        "action": {"sha256": RUNNER.manifest_array_sha256(action)},
                        "valid_mask": {"sha256": RUNNER.manifest_array_sha256(valid_mask)},
                    },
                }
            ),
            encoding="utf-8",
        )
        return manifest, npz, npz_sha

    def test_offline_input_requires_hashes_fixed_condition_and_395_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, npz, npz_sha = self._offline_input(root)
            observation, action, _ = RUNNER._validated_offline_rows(
                manifest, npz, expected_npz_sha256=npz_sha
            )
            self.assertEqual(observation.shape, (395, 79))
            self.assertEqual(action.shape, (395, 6))
            with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "NPZ SHA"):
                RUNNER._validated_offline_rows(manifest, npz, expected_npz_sha256="0" * 64)
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["condition"]["grasp_id"] = "wrong"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "fixed-condition"):
                RUNNER._validated_offline_rows(manifest, npz, expected_npz_sha256=npz_sha)

    def _checkpoint(self, path: Path) -> None:
        import rby1_assembly.rj45_bc as bc

        helper = RUNNER._diagnostic_loader(Path(__file__).resolve().parents[1])
        actor = bc.build_bc_actor().cpu()
        with torch.no_grad():
            for parameter in actor.parameters():
                parameter.zero_()
            actor[-1].bias.fill_(0.125)
        state = {name: value.detach().clone() for name, value in actor.state_dict().items()}
        normalizer = helper._identity_normalizer()
        payload = helper._checkpoint_payload(
            {"state": state, "final_actor_state_sha256": bc.actor_state_sha256(state)},
            normalizer,
            "identity",
        )
        torch.save(payload, path)

    def test_cpu_policy_validates_checkpoint_and_runs_offline_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "actor.pt"
            self._checkpoint(checkpoint)
            policy = RUNNER.CpuDiagnosticPolicy(
                checkpoint,
                RUNNER.file_sha256(checkpoint),
                Path(__file__).resolve().parents[1],
            )
            prediction = policy.predict(np.zeros(79, dtype=np.float32))
            np.testing.assert_array_equal(prediction, np.full((1, 6), 0.125, dtype=np.float32))
            preflight = RUNNER.offline_parity_preflight(
                policy,
                np.zeros((395, 79), dtype=np.float32),
                np.zeros((395, 6), dtype=np.float32),
            )
            self.assertEqual(preflight["rows"], 395)
            self.assertAlmostEqual(preflight["mse"], 0.125**2)
            with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "checkpoint SHA"):
                RUNNER.CpuDiagnosticPolicy(checkpoint, "0" * 64, Path(__file__).resolve().parents[1])

    def test_diagnostic_loader_rejects_tamper_before_module_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "scripts"
            scripts.mkdir()
            marker = root / "executed.txt"
            loader = scripts / "run_rj45_g2_input_normalization_probe.py"
            loader.write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                RUNNER.PhysicsProbeError, "input-normalization runner SHA"
            ):
                RUNNER._diagnostic_loader(root)
            self.assertFalse(marker.exists())

    def test_fixed_bank_identity_is_ordered_and_strict(self) -> None:
        proof = {
            "manifest_sha256": RUNNER.EXPECTED_BANK_SHA256,
            "selected_split": "train",
            "selected_grasp_ids": [f"rj45-train-{index:02d}" for index in range(1, 9)],
        }
        identity = RUNNER.fixed_condition_identity(proof)
        self.assertEqual(identity["grasp_index"], 1)
        self.assertEqual(identity["grasp_id"], "rj45-train-02")
        self.assertEqual(identity["episode_index"], 1)
        damaged = dict(proof)
        damaged["selected_grasp_ids"] = list(reversed(proof["selected_grasp_ids"]))
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "ordered train identity"):
            RUNNER.fixed_condition_identity(damaged)

    def test_calibration_seed_uses_base_env_episode_argument_order(self) -> None:
        observed = []

        def seed_function(base_seed: int, env_id: int, episode_index: int) -> int:
            observed.append((base_seed, env_id, episode_index))
            return RUNNER.CALIBRATION_CASE_SEED

        RUNNER.verify_calibration_case_seed(seed_function)
        self.assertEqual(
            observed,
            [(RUNNER.BASE_SEED, RUNNER.ENVIRONMENT_INDEX, RUNNER.EPISODE_INDEX)],
        )

    def test_fixed_episode_uses_constructor_episode_zero_and_one_reset(self) -> None:
        class FakeEnv:
            def __init__(self) -> None:
                self._training_episode_index = np.asarray([0])
                self._training_grasp_index = np.asarray([0])
                self.reset_count = 0

            def reset(self):
                self.reset_count += 1
                self._training_episode_index[:] = 1
                self._training_grasp_index[:] = 1
                return {"policy": "episode-1"}, {"reset": True}

        env = FakeEnv()
        observation = RUNNER.advance_to_fixed_episode(env)
        self.assertEqual(observation, {"policy": "episode-1"})
        self.assertEqual(env.reset_count, 1)
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "episode index 0"):
            RUNNER.advance_to_fixed_episode(env)

    def test_terminal_classification_is_fail_closed(self) -> None:
        class Terminal:
            def __init__(self, valid: bool, success: bool) -> None:
                self.valid = valid
                self.success = success

        self.assertEqual(RUNNER.classify_terminal(Terminal(True, True)), "valid_success")
        self.assertEqual(RUNNER.classify_terminal(Terminal(True, False)), "valid_failure")
        self.assertEqual(RUNNER.classify_terminal(Terminal(False, True)), "invalid")

    def test_action_provenance_extracts_right_6d_and_rejects_left_motion(self) -> None:
        submitted = np.arange(6, dtype=np.float32) / 10.0
        requested = np.concatenate((np.zeros(6, dtype=np.float32), submitted))
        applied = np.concatenate((np.zeros(6, dtype=np.float32), submitted / 2.0))
        record = RUNNER.action_provenance(submitted, requested, applied)
        np.testing.assert_array_equal(record["applied_bounded_action_6d"], applied[6:])
        self.assertEqual(len(record["applied_bounded_action_12d"]), 12)
        damaged = applied.copy()
        damaged[0] = 0.01
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "held left arm"):
            RUNNER.action_provenance(submitted, requested, damaged)

    def test_probe_snapshot_uses_runtime_contact_source_proof(self) -> None:
        from rby1_assembly.hand_collision import (
            COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
            COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
            CROSS_HAND_CONTACT_LOAD_SOURCE,
            selective_hand_collision_pairs,
        )
        from rby1_assembly.recording import (
            INSERTION_FORCE_SOURCE,
            INSERTION_LOAD_SOURCE,
        )
        from rby1_assembly.rj45_task import verify_recorded_mating_geometry

        socket_position = np.asarray([[0.0, 0.0, 0.0]])
        plug_position = np.asarray([[0.0, 0.0, 0.0]])
        quaternion = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        poses = np.stack(
            (
                np.concatenate((socket_position[0], quaternion[0])),
                np.concatenate((plug_position[0], quaternion[0])),
            )
        )
        geometry = {
            "socket_entry": {
                "position_m": [0.0, -0.033, 0.0005075],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
            "plug_tip": {
                "position_m": [0.0, 0.01137, 0.0],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            },
        }
        verified_geometry = verify_recorded_mating_geometry(
            poses[0], poses[1], geometry["socket_entry"], geometry["plug_tip"],
            0.04437, 0.0005075, 0.0,
        )
        depth = verified_geometry["depth_m"]
        lateral = verified_geometry["lateral_error_m"]
        orientation = verified_geometry["orientation_error_rad"]
        robot_root_path = "/World/envs/env_0/Robot"
        collision_filter = {
            "robot_root_path": robot_root_path,
            "self_collision_enabled": True,
            "rigid_body_count": 66,
            "allowed_cross_hand_pair_count": 9,
            "filtered_internal_pair_count": 2136,
            "allowed_cross_hand_pairs": [
                list(pair)
                for pair in selective_hand_collision_pairs(robot_root_path)["allowed"]
            ],
            "filtered_pair_authoring": "one_direction_per_unordered_rigid_body_pair",
            "external_collision_relationships_modified": False,
        }
        contact_metadata = {
            "cross_hand_contact_load_n": np.asarray([0.25]),
            "cross_hand_contact_load_peak_n": np.asarray([0.5]),
            "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
            "external_forbidden_contact_load_n": np.asarray([0.1]),
            "external_forbidden_contact_load_peak_n": np.asarray([0.4]),
            "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
            "forbidden_contact_load_definition": COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
            "self_collision_enabled": True,
            "hand_contact_monitor_mode": "passive_nominal_one_environment",
            "selective_self_collision_filter": collision_filter,
        }
        evaluation = {
            "actual_depth_m": np.asarray([depth]),
            "actual_lateral_error_m": np.asarray([lateral]),
            "actual_orientation_error_rad": np.asarray([orientation]),
            "contact_force_proxy_peak_n": np.asarray([6.0]),
            "grip_contact_force_peak_n": np.asarray([4.0]),
            "insertion_contact_force_n": np.asarray([3.0]),
            "insertion_contact_force_peak_n": np.asarray([3.5]),
            "insertion_contact_load_n": np.asarray([5.0]),
            "insertion_contact_load_peak_n": np.asarray([6.0]),
            "forbidden_contact_load_n": np.asarray([0.25]),
            "forbidden_contact_load_peak_n": np.asarray([0.5]),
            "grasp_both": np.asarray([True]),
            "forbidden_contact": np.asarray([False]),
            **contact_metadata,
        }

        class FakeEnv:
            scene = SimpleNamespace(env_prim_paths=["/World/envs/env_0"])
            socket = SimpleNamespace(
                data=SimpleNamespace(root_pos_w=socket_position, root_quat_w=quaternion)
            )
            plug = SimpleNamespace(
                data=SimpleNamespace(root_pos_w=plug_position, root_quat_w=quaternion)
            )
            _phase = np.asarray([4])

            @staticmethod
            def hand_contact_monitor_report():
                return {
                    "command_pose_force_or_failure_injection": False,
                    "hand_contact_monitor_mode": "passive_nominal_one_environment",
                    "num_envs": 1,
                    "forbidden_contact_load_source": COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
                    "forbidden_contact_load_definition": COMBINED_FORBIDDEN_CONTACT_LOAD_DEFINITION,
                    "cross_hand_contact_load_source": CROSS_HAND_CONTACT_LOAD_SOURCE,
                    "collision_filter": collision_filter,
                }

        payload = {
            "evaluation": evaluation,
            "object_poses_world_wxyz": poses[None, ...],
            "phase": np.asarray([4]),
            "grip_contact_force_peak_n": np.asarray([4.0]),
            "insertion_contact_force_n": np.asarray([3.0]),
            "insertion_contact_force_w": np.asarray([[3.0, 0.0, 0.0]]),
            "insertion_contact_force_peak_n": np.asarray([3.5]),
            "insertion_contact_load_n": np.asarray([5.0]),
            "insertion_contact_load_peak_n": np.asarray([6.0]),
            "forbidden_contact_load_n": np.asarray([0.25]),
            "forbidden_contact_load_peak_n": np.asarray([0.5]),
            "insertion_contact_force_source": INSERTION_FORCE_SOURCE,
            "insertion_contact_load_source": INSERTION_LOAD_SOURCE,
            **contact_metadata,
        }
        snapshot = RUNNER._probe_snapshot(FakeEnv(), payload, geometry)
        self.assertAlmostEqual(snapshot.ground_truth.lateral_error_m, 0.0005075)
        self.assertEqual(
            snapshot.forbidden_contact_load_source,
            COMBINED_FORBIDDEN_CONTACT_LOAD_SOURCE,
        )
        self.assertEqual(snapshot.ground_truth.force_peak_n, 6.0)
        damaged = dict(payload)
        damaged["forbidden_contact_load_source"] = "unproven-source"
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "metadata"):
            RUNNER._probe_snapshot(FakeEnv(), damaged, geometry)
        z_tampered_geometry = {
            **geometry,
            "socket_entry": {
                **geometry["socket_entry"],
                "position_m": [0.0, -0.033, 0.0],
            },
        }
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "full-frame"):
            RUNNER._probe_snapshot(FakeEnv(), payload, z_tampered_geometry)
        quaternion_tampered_geometry = {
            **geometry,
            "socket_entry": {
                **geometry["socket_entry"],
                "quaternion_wxyz": [0.9, 0.0, 0.0, 0.0],
            },
        }
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "full-frame"):
            RUNNER._probe_snapshot(FakeEnv(), payload, quaternion_tampered_geometry)
        tampered_filter = {
            **collision_filter,
            "allowed_cross_hand_pairs": [
                *collision_filter["allowed_cross_hand_pairs"][:-1],
                collision_filter["allowed_cross_hand_pairs"][0],
            ],
        }
        tampered_evaluation = {
            **evaluation,
            "selective_self_collision_filter": tampered_filter,
        }
        tampered_pairs = {
            **payload,
            "evaluation": tampered_evaluation,
            "selective_self_collision_filter": tampered_filter,
        }

        class TamperedPairEnv(FakeEnv):
            @staticmethod
            def hand_contact_monitor_report():
                report = FakeEnv.hand_contact_monitor_report()
                return {**report, "collision_filter": tampered_filter}

        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "allowed-pair"):
            RUNNER._probe_snapshot(
                TamperedPairEnv(), tampered_pairs, geometry
            )
        missing = dict(payload)
        missing.pop("cross_hand_contact_load_n")
        with self.assertRaises(KeyError):
            RUNNER._probe_snapshot(FakeEnv(), missing, geometry)
        terminal_payload = payload
        terminal_snapshot = RUNNER._probe_snapshot(FakeEnv(), terminal_payload, geometry)
        self.assertEqual(terminal_snapshot.forbidden_contact_load_n, 0.25)
        tampered_components = dict(terminal_payload)
        tampered_components["cross_hand_contact_load_n"] = np.asarray([0.3])
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "component"):
            RUNNER._probe_snapshot(FakeEnv(), tampered_components, geometry)

    def test_evaluation_sample_recomputes_all_rj45_physics_substeps(self) -> None:
        metrics = SimpleNamespace(
            actual_depth_m=0.0125,
            lateral_error_m=0.0,
            orientation_error_rad=0.0,
            force_peak_n=5.0,
            grasp_both=True,
            forbidden_contact=False,
        )
        snapshot = SimpleNamespace(ground_truth=metrics, phase="insert")
        stable = np.ones(RUNNER.CONTROL_DECIMATION + 1, dtype=bool)
        transition = {
            "rj45_evaluation_stability": {
                "physics_sample_depth_m": np.full((1, 9), 0.0125),
                "physics_sample_lateral_error_m": np.zeros((1, 9)),
                "physics_sample_orientation_error_rad": np.zeros((1, 9)),
                "physics_sample_relative_linear_speed_m_s": np.zeros((1, 9)),
                "physics_sample_relative_angular_speed_rad_s": np.zeros((1, 9)),
                "physics_sample_finite": stable[None, :],
                "physics_sample_stable": stable[None, :],
                "physics_substeps_complete": np.asarray([True]),
                "physics_all_samples_stable": np.asarray([True]),
            }
        }
        contract = {
            "assembled_depth_m": 0.012,
            "max_depth_m": 0.013,
            "max_lateral_error_m": 5e-5,
            "max_orientation_error_rad": 0.008726646259971648,
            "max_relative_linear_speed_m_s": 0.005,
            "max_relative_angular_speed_rad_s": 0.05,
        }
        sample = RUNNER._evaluation_sample(snapshot, transition, 1.0, contract)
        self.assertTrue(sample.grasp_both)
        damaged = {
            **transition,
            "rj45_evaluation_stability": {
                **transition["rj45_evaluation_stability"],
                "physics_sample_relative_linear_speed_m_s": np.asarray(
                    [[0.006, *([0.0] * 8)]]
                ),
            },
        }
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "stability proof"):
            RUNNER._evaluation_sample(snapshot, damaged, 1.0, contract)
        endpoint_mismatch = {
            **transition,
            "rj45_evaluation_stability": {
                **transition["rj45_evaluation_stability"],
                "physics_sample_depth_m": np.full((1, 9), 0.0126),
            },
        }
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "endpoint"):
            RUNNER._evaluation_sample(
                snapshot, endpoint_mismatch, 1.0, contract
            )

    def test_transition_contract_rejects_mismatch_and_excludes_warmup(self) -> None:
        observation = torch.zeros((1, RUNNER.OBSERVATION_DIM), dtype=torch.float32)
        saved = {
            "terminal_mask": np.asarray([False]),
            "policy_input_observation": np.zeros((1, RUNNER.OBSERVATION_DIM)),
            "timestamp_s": np.asarray([0.2]),
            "requested_expanded_residual_action": np.zeros((1, 12)),
            "applied_bounded_residual_action": np.zeros((1, 12)),
            "applied_tip_target_offset_entry_6d": np.zeros((1, 6)),
            "policy_input_valid": np.asarray([False]),
            "training_policy_activated": np.asarray([True]),
            "valid_policy_input_step_count": np.asarray([0]),
        }
        self.assertFalse(RUNNER._transition_contract(saved, observation, False)["active"])
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "terminal mask"):
            RUNNER._transition_contract(saved, observation, True)
        damaged = {
            **saved,
            "policy_input_observation": np.ones((1, RUNNER.OBSERVATION_DIM)),
        }
        with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "policy input"):
            RUNNER._transition_contract(damaged, observation, False)

    def test_frozen_protocol_binds_runner_and_both_policies(self) -> None:
        record = RUNNER.frozen_protocol_record("a" * 64)
        self.assertEqual(record["payload"]["runner_sha256"], "a" * 64)
        self.assertEqual(set(record["payload"]["policies"]), set(RUNNER.EXPECTED_POLICIES))
        self.assertEqual(
            record["payload"]["input_normalization_runner_sha256"],
            RUNNER.EXPECTED_INPUT_NORMALIZATION_RUNNER_SHA256,
        )
        self.assertEqual(len(record["sha256"]), 64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            path.write_text(
                json.dumps(record["payload"], indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            verified = RUNNER.validate_protocol(
                path, RUNNER.file_sha256(path), "a" * 64
            )
            self.assertEqual(verified["sha256"], record["sha256"])
            with self.assertRaisesRegex(RUNNER.PhysicsProbeError, "file SHA"):
                RUNNER.validate_protocol(path, "0" * 64, "a" * 64)

    def test_output_is_new_only_and_atomic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            RUNNER._atomic_write_new(path, {"status": "valid_failure"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "valid_failure")
            with self.assertRaises(FileExistsError):
                RUNNER._atomic_write_new(path, {"status": "valid_success"})


if __name__ == "__main__":
    unittest.main()
