from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import numpy as np

from rby1_assembly.dataset import (
    EpisodeDatasetWriter,
    EpisodeProvenance,
    ProvenanceKind,
)
from rby1_assembly.contracts import TaskLimits
from rby1_assembly.evaluation import EvaluationResult
from rby1_assembly.recording import (
    AssemblyGeometry,
    RecordingContractError,
    geometry_metrics_from_object_poses,
    live_snapshot,
    reconcile_terminal_evaluation,
    record_baseline_rollouts,
    select_failure_reports,
    terminal_evaluation_from_extras,
    terminal_snapshot_from_extras,
    wxyz_to_xyzw,
)


GEOMETRY = AssemblyGeometry(socket_entry_y_m=-0.03, plug_tip_y_m=0.065)


def evaluation() -> dict[str, np.ndarray]:
    return {
        "actual_depth_m": np.asarray([0.031]),
        "actual_lateral_error_m": np.asarray([0.001]),
        "actual_orientation_error_rad": np.asarray([0.0]),
        "contact_force_proxy_peak_n": np.asarray([12.0]),
        "grasp_both": np.asarray([True]),
        "forbidden_contact": np.asarray([False]),
    }


def terminal_snapshot() -> dict[str, object]:
    return {
        "revolute_joint_positions_rad": np.arange(20, dtype=float)[None, :],
        "revolute_joint_velocities_rad_s": np.ones((1, 20)),
        "gripper_positions_m": np.full((1, 4), 0.02),
        "gripper_velocities_m_s": np.zeros((1, 4)),
        "object_poses_world_wxyz": np.asarray(
            [[[0.0, 0.0, 0.75, 1.0, 0.0, 0.0, 0.0], [0.001, -0.064, 0.75, 1.0, 0.0, 0.0, 0.0]]]
        ),
        "contact_force_proxy_n": np.asarray([[3.0, 4.0]]),
        "phase": np.asarray([6]),
        "evaluation": evaluation(),
    }


class RecordingTests(unittest.TestCase):
    def test_quaternion_conversion_preserves_rotation_order(self) -> None:
        converted = wxyz_to_xyzw(np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
        np.testing.assert_allclose(converted, [[0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]])
        with self.assertRaises(RecordingContractError):
            wxyz_to_xyzw([2.0, 0.0, 0.0, 0.0])
        half = np.sqrt(0.5)
        rotated = np.asarray(
            [
                [0.0, 0.0, 0.0, half, 0.0, 0.0, half],
                [0.064, 0.001, 0.0, half, 0.0, 0.0, half],
            ]
        )
        depth, lateral, angle = geometry_metrics_from_object_poses(rotated, GEOMETRY)
        self.assertAlmostEqual(depth, 0.031)
        self.assertAlmostEqual(lateral, 0.001)
        self.assertAlmostEqual(angle, 0.0)

    def test_terminal_snapshot_is_required_after_autoreset(self) -> None:
        with self.assertRaisesRegex(RecordingContractError, "resets before step returns"):
            terminal_snapshot_from_extras({"evaluation": evaluation()}, GEOMETRY)

    def test_terminal_snapshot_uses_saved_state_and_converts_quaternion(self) -> None:
        extras = {"terminal_snapshot": terminal_snapshot()}
        snapshot = terminal_snapshot_from_extras(extras, GEOMETRY)
        np.testing.assert_allclose(snapshot.joint_position_rad, np.arange(20))
        np.testing.assert_allclose(snapshot.gripper_position_m, np.full(4, 0.02))
        np.testing.assert_allclose(snapshot.object_poses_world_xyzw[0, 3:], [0.0, 0.0, 0.0, 1.0])
        np.testing.assert_allclose(snapshot.object_poses_world_xyzw[1, 3:], [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(snapshot.reported_geometry_consistent)
        self.assertEqual(snapshot.phase, "done")

    def test_live_snapshot_reads_physics_state(self) -> None:
        identity = np.asarray([[1.0, 0.0, 0.0, 0.0]])
        env = SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=np.zeros((1, 24)), joint_vel=np.ones((1, 24)))),
            socket=SimpleNamespace(data=SimpleNamespace(root_pos_w=np.zeros((1, 3)), root_quat_w=identity)),
            plug=SimpleNamespace(data=SimpleNamespace(root_pos_w=np.asarray([[0.001, -0.064, 0.0]]), root_quat_w=identity)),
            _last_contact_proxy_n=np.asarray([[2.0, 3.0]]),
            _phase=np.asarray([4]),
            _grip_proven=np.asarray([[True, True]]),
            _torso_joint_ids=np.arange(6),
            _arm_joint_ids=np.arange(6, 20),
            _gripper_joint_ids=np.arange(20, 24),
            cfg=SimpleNamespace(hard_contact_force_n=40.0),
        )
        snapshot = live_snapshot(env, {"evaluation": evaluation()}, GEOMETRY)
        self.assertEqual(snapshot.phase, "insert")
        self.assertTrue(snapshot.ground_truth.grasp_both)
        self.assertFalse(snapshot.ground_truth.forbidden_contact)
        self.assertEqual(snapshot.object_poses_world_xyzw.shape, (2, 7))

    def test_live_snapshot_does_not_infer_forbidden_contact_from_force(self) -> None:
        env = SimpleNamespace(
            robot=SimpleNamespace(data=SimpleNamespace(joint_pos=np.zeros((1, 24)), joint_vel=np.zeros((1, 24)))),
            socket=SimpleNamespace(data=SimpleNamespace(root_pos_w=np.zeros((1, 3)), root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]))),
            plug=SimpleNamespace(data=SimpleNamespace(root_pos_w=np.zeros((1, 3)), root_quat_w=np.asarray([[1.0, 0.0, 0.0, 0.0]]))),
            _last_contact_proxy_n=np.zeros((1, 2)),
            _phase=np.asarray([0]),
            _grip_proven=np.asarray([[False, False]]),
            _torso_joint_ids=np.arange(6),
            _arm_joint_ids=np.arange(6, 20),
            _gripper_joint_ids=np.arange(20, 24),
        )
        incomplete = evaluation()
        del incomplete["forbidden_contact"]
        with self.assertRaisesRegex(RecordingContractError, "forbidden_contact"):
            live_snapshot(env, {"evaluation": incomplete}, GEOMETRY)

    def test_terminal_evaluation_fails_closed_without_independent_evaluator(self) -> None:
        snapshot = terminal_snapshot_from_extras({"terminal_snapshot": terminal_snapshot()}, GEOMETRY)
        result = terminal_evaluation_from_extras({}, snapshot, 1.0)
        self.assertFalse(result.valid)
        self.assertFalse(result.success)
        self.assertEqual(result.termination_reason, "independent_evaluator_missing")

    def test_terminal_evaluation_accepts_complete_independent_result(self) -> None:
        snapshot = terminal_snapshot_from_extras({"terminal_snapshot": terminal_snapshot()}, GEOMETRY)
        extras = {
            "terminal_evaluation": {
                "evaluator_source": "simulator_ground_truth_independent",
                "evaluator_version": "geometry-v1",
                "valid": np.asarray([True]),
                "success": np.asarray([True]),
                "termination_reason": np.asarray(["success"]),
                "criteria": {"depth": np.asarray([True]), "force": np.asarray([True])},
            }
        }
        result = terminal_evaluation_from_extras(extras, snapshot, 1.0)
        self.assertTrue(result.valid)
        self.assertTrue(result.success)
        self.assertEqual(result.evaluator_version, "geometry-v1")

    def test_cross_check_rejects_disagreement_and_accepts_matching_success(self) -> None:
        snapshot = terminal_snapshot_from_extras({"terminal_snapshot": terminal_snapshot()}, GEOMETRY)
        extras = {
            "terminal_evaluation": {
                "evaluator_source": "simulator_ground_truth_independent",
                "evaluator_version": "geometry-v1",
                "valid": np.asarray([True]),
                "success": np.asarray([True]),
                "termination_reason": np.asarray(["success"]),
                "criteria": {"depth": np.asarray([True]), "dwell": np.asarray([True])},
            }
        }
        accepted = reconcile_terminal_evaluation(
            extras=extras,
            snapshot=snapshot,
            terminal_timestamp_s=1.0,
            independent_result=EvaluationResult(True, True, 0.4),
            task_limits=TaskLimits(),
        )
        self.assertTrue(accepted.valid)
        self.assertTrue(accepted.success)
        rejected = reconcile_terminal_evaluation(
            extras=extras,
            snapshot=snapshot,
            terminal_timestamp_s=1.0,
            independent_result=EvaluationResult(True, False, 0.0, ("timeout",)),
            task_limits=TaskLimits(),
        )
        self.assertFalse(rejected.valid)
        self.assertFalse(rejected.success)
        malformed = dict(extras)
        malformed["terminal_evaluation"] = dict(extras["terminal_evaluation"])
        malformed["terminal_evaluation"]["criteria"] = {}
        failed_closed = reconcile_terminal_evaluation(
            extras=malformed,
            snapshot=snapshot,
            terminal_timestamp_s=1.0,
            independent_result=EvaluationResult(True, True, 0.4),
            task_limits=TaskLimits(),
        )
        self.assertFalse(failed_closed.valid)
        self.assertEqual(failed_closed.termination_reason, "environment_evaluator_malformed")
        inconsistent_saved = terminal_snapshot()
        inconsistent_saved["evaluation"] = dict(evaluation())
        inconsistent_saved["evaluation"]["actual_depth_m"] = np.asarray([0.5])
        inconsistent = terminal_snapshot_from_extras(
            {"terminal_snapshot": inconsistent_saved}, GEOMETRY
        )
        geometry_rejected = reconcile_terminal_evaluation(
            extras=extras,
            snapshot=inconsistent,
            terminal_timestamp_s=1.0,
            independent_result=EvaluationResult(True, True, 0.4),
            task_limits=TaskLimits(),
        )
        self.assertFalse(geometry_rejected.valid)
        self.assertEqual(geometry_rejected.termination_reason, "geometry_metric_mismatch")

        timeout_extras = {
            "terminal_evaluation": {
                "evaluator_source": "simulator_ground_truth_independent",
                "evaluator_version": "geometry-v1",
                "valid": np.asarray([True]),
                "success": np.asarray([False]),
                "termination_reason": np.asarray(["episode_timeout"]),
                "criteria": {"depth": np.asarray([False]), "dwell": np.asarray([False])},
            }
        }
        early_timeout = reconcile_terminal_evaluation(
            extras=timeout_extras,
            snapshot=snapshot,
            terminal_timestamp_s=0.9,
            independent_result=EvaluationResult(False, False, 0.0),
            task_limits=TaskLimits(timeout_s=1.0),
        )
        exact_timeout = reconcile_terminal_evaluation(
            extras=timeout_extras,
            snapshot=snapshot,
            terminal_timestamp_s=1.0,
            independent_result=EvaluationResult(False, False, 0.0),
            task_limits=TaskLimits(timeout_s=1.0),
        )
        self.assertFalse(early_timeout.valid)
        self.assertTrue(exact_timeout.valid)
        self.assertFalse(exact_timeout.success)

    def test_rollout_uses_pre_action_observation_and_terminal_snapshot(self) -> None:
        class AutoResetEnv:
            num_envs = 1
            step_dt = 0.1

            def __init__(self) -> None:
                self.episode = -1

            def reset(self, *, seed: int):
                self.episode += 1
                return {"policy": np.asarray([[10.0 + self.episode]])}, {}

            def step(self, action):
                marker = 100.0 + self.episode
                saved = terminal_snapshot()
                saved["revolute_joint_positions_rad"] = np.full((1, 20), marker)
                terminal = {
                    "evaluator_source": "simulator_ground_truth_independent",
                    "evaluator_version": "geometry-v1",
                    "valid": np.asarray([True]),
                    "success": np.asarray([False]),
                    "termination_reason": np.asarray(["pregrasp_timeout"]),
                    "criteria": {"depth": np.asarray([False]), "force": np.asarray([True])},
                }
                # 반환 시점의 env 속성은 이미 reset됐다고 가정한다.
                self.reset_state_marker = -1.0
                return (
                    {"policy": np.asarray([[-999.0]])},
                    np.asarray([0.0]),
                    np.asarray([True]),
                    np.asarray([False]),
                    {"terminal_snapshot": saved, "terminal_evaluation": terminal},
                )

        provenance = EpisodeProvenance(
            kind=ProvenanceKind.SYNTHETIC_TEST_FIXTURE,
            source_hashes={"fixture": "a" * 64},
            task_config_sha256="b" * 64,
            runtime_versions={"fixture": "1"},
            observation_source="fixture",
            contact_force_source="fixture",
            owns_demonstration=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "recorded.h5"
            with EpisodeDatasetWriter(path) as writer:
                report = record_baseline_rollouts(
                    env=AutoResetEnv(),
                    writer=writer,
                    provenance=provenance,
                    episode_count=2,
                    max_steps=3,
                    base_seed=50,
                    action_factory=lambda env: np.zeros((1, 12)),
                    task_limits=TaskLimits(),
                    geometry=GEOMETRY,
                )
            self.assertEqual(len(report), 2)
            self.assertTrue(all(item["evaluation_valid"] for item in report))
            with h5py.File(path, "r") as file:
                first = file["episodes/baseline-seed-50-episode-0000"]
                second = file["episodes/baseline-seed-51-episode-0001"]
                self.assertEqual(first["observations"][0, 0], 10.0)
                self.assertEqual(second["observations"][0, 0], 11.0)
                self.assertEqual(first["joint_positions_rad"][0, 0], 100.0)
                self.assertEqual(second["joint_positions_rad"][0, 0], 101.0)

    def test_failure_selection_is_bounded_and_deterministic(self) -> None:
        reports = [
            {"episode_id": "invalid", "success": False, "evaluation_valid": False},
            {"episode_id": "unknown", "success": False},
            {"episode_id": "a", "success": False, "evaluation_valid": True},
            {"episode_id": "b", "success": True, "evaluation_valid": True},
            {"episode_id": "c", "success": False, "evaluation_valid": True},
            {"episode_id": "d", "success": False, "evaluation_valid": True},
        ]
        self.assertEqual(
            [item["episode_id"] for item in select_failure_reports(reports)], ["a", "c"]
        )


if __name__ == "__main__":
    unittest.main()
