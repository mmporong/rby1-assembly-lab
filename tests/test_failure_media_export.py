import unittest

from scripts.export_failure_media import (
    EvidenceError,
    frame_advance,
    held_trace_index,
    scenario_metrics,
    source_frame_range,
    source_frame_time_s,
    validate_report,
)


def row(timestamp, *, terminal=False, reason="none", right=(3.0, 2.0), z=0.86,
        load=0.0, depth=0.004, lateral=0.001, retry=0, fault=None, loss_since=-1.0):
    value = {
        "schema_version": 1,
        "environments": [{
            "env_index": 0,
            "timestamp_s": timestamp,
            "terminal": terminal,
            "termination_reason": reason,
            "finger_contact_force_n": [4.0, 4.0, *right],
            "object_poses_world_wxyz": {"plug": [0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]},
            "insertion_contact_load_n": load,
            "actual_depth_m": depth,
            "actual_lateral_error_m": lateral,
            "retry_count": retry,
            "continuous_grasp_loss_since_s": loss_since,
        }],
    }
    if fault is not None:
        value["command_fault"] = fault
    return value


class TimestampAlignmentTests(unittest.TestCase):
    def test_hold_uses_previous_sample_without_future_leakage(self):
        timestamps = [0.0, 1 / 30, 2 / 30]
        self.assertEqual(held_trace_index(timestamps, 0.049), 1)
        self.assertEqual(held_trace_index(timestamps, 2 / 30), 2)
        with self.assertRaises(EvidenceError):
            held_trace_index(timestamps, -0.001)

    def test_native_video_timebase_and_fault_clip(self):
        self.assertAlmostEqual(source_frame_time_s(119, 1, 1 / 120, 120), 1.0)
        start, end = source_frame_range(
            fault_time_s=10.0, terminal_time_s=12.0, video_first_physics_step=1,
            physics_dt_s=1 / 120, source_fps=120, source_frame_count=2000,
        )
        self.assertEqual(start, 1019)
        self.assertEqual(end, 1439)

    def test_export_cadences_only_consume_native_frames(self):
        self.assertEqual(frame_advance(120, 60, 0.5), 1)
        self.assertEqual(frame_advance(60, 60, 2.0), 2)
        with self.assertRaises(EvidenceError):
            frame_advance(30, 60, 1.0)

    def test_hand_collision_motion_window_preserves_actual_fault_time(self):
        start, end = source_frame_range(
            fault_time_s=5.566666666666666,
            terminal_time_s=18.666667938232422,
            video_first_physics_step=1,
            physics_dt_s=1 / 120,
            source_fps=120,
            source_frame_count=2240,
            max_motion_window_s=3.0,
        )
        self.assertEqual(start, 1880)
        self.assertEqual(end, 2239)
        self.assertAlmostEqual(source_frame_time_s(start, 1, 1 / 120, 120), 15.675)


class EvidenceMetricTests(unittest.TestCase):
    def test_gripper_metrics_are_measured_from_trace(self):
        rows = [
            row(1.0, right=(3.0, 2.0), z=0.86),
            row(1.1, right=(0.8, 0.2), z=0.85, loss_since=1.02,
                fault={"right_gripper_open_command": True}),
        ]
        metrics = scenario_metrics("right_gripper_open", rows, 1, 0)
        self.assertAlmostEqual(metrics["right_min_finger_contact_n"], 0.2)
        self.assertAlmostEqual(metrics["drop_from_peak_mm"], 10.0)
        self.assertAlmostEqual(metrics["continuous_grasp_loss_s"], 0.08)
        self.assertTrue(metrics["command_active"])

    def test_insertion_metrics_distinguish_command_from_actual_lateral_error(self):
        rows = [row(2.0, load=31.5, depth=0.006, lateral=0.0022, retry=1,
                    fault={"active": True, "lateral_position_error_command_m": 0.0035})]
        metrics = scenario_metrics("insertion_lateral_disturbance", rows, 0, 0)
        self.assertAlmostEqual(metrics["insertion_contact_load_n"], 31.5)
        self.assertAlmostEqual(metrics["actual_depth_mm"], 6.0)
        self.assertAlmostEqual(metrics["actual_lateral_error_mm"], 2.2)
        self.assertAlmostEqual(metrics["command_error_mm"], 3.5)
        self.assertEqual(metrics["retry_count"], 1)

    def test_hand_collision_metrics_require_actual_self_collision_measurements(self):
        collision = row(4.0, fault={"active": True})
        collision["environments"][0].update({
            "self_collision_enabled": True,
            "cross_hand_contact_load_n": 12.5,
            "external_forbidden_contact_load_n": 0.0,
            "hand_wrist_separation_m": 0.083,
        })
        metrics = scenario_metrics("hand_collision", [collision], 0, 0)
        self.assertAlmostEqual(metrics["cross_hand_contact_load_n"], 12.5)
        self.assertAlmostEqual(metrics["hand_wrist_separation_mm"], 83.0)
        collision["environments"][0]["self_collision_enabled"] = False
        with self.assertRaises(EvidenceError):
            scenario_metrics("hand_collision", [collision], 0, 0)

    def test_report_reason_must_match_terminal_trace(self):
        rows = [row(1.0), row(2.0, terminal=True, reason="grasp_lost")]
        report = {
            "command_fault_experiment": {
                "scenario": "right_gripper_open",
                "parameters": {"trigger_min_plug_height_m": 0.86},
                "triggered": True,
                "events": [{
                    "event": "right_gripper_open_command_started",
                    "issued_at_simulation_time_s": 1.5,
                }],
                "physical_state_force_sensor_or_termination_injection": False,
            },
            "terminal_outcomes": [{
                "terminal": True, "timestamp_s": 2.0,
                "termination_reason": "grasp_lost",
            }],
            "trace_every_control_steps": 1,
            "provenance": {"physics_dt_s": 1 / 120, "control_dt_s": 1 / 30},
        }
        result = validate_report(report, rows)
        self.assertEqual(result["actual_reason"], "grasp_lost")
        report["terminal_outcomes"][0]["termination_reason"] = "hard_force_limit"
        with self.assertRaises(EvidenceError):
            validate_report(report, rows)

    def test_retry_may_record_multiple_lateral_command_events(self):
        rows = [row(1.0), row(3.0, terminal=True, reason="retry_limit")]
        report = {
            "command_fault_experiment": {
                "scenario": "insertion_lateral_disturbance",
                "parameters": {
                    "trigger_insertion_depth_m": 0.008,
                    "max_position_error_command_m": 0.0075,
                    "ramp_duration_s": 1.0,
                },
                "triggered": True,
                "events": [
                    {"event": "lateral_command_started", "issued_at_simulation_time_s": 1.5},
                    {"event": "lateral_command_stopped", "issued_at_simulation_time_s": 2.0},
                    {"event": "lateral_command_started", "issued_at_simulation_time_s": 2.5},
                ],
                "physical_state_force_sensor_or_termination_injection": False,
            },
            "terminal_outcomes": [{
                "terminal": True, "timestamp_s": 3.0, "termination_reason": "retry_limit",
            }],
            "trace_every_control_steps": 1,
            "provenance": {"physics_dt_s": 1 / 120, "control_dt_s": 1 / 30},
        }
        result = validate_report(report, rows)
        self.assertEqual(result["fault_time_s"], 1.5)

    def test_hand_collision_requires_actual_forbidden_contact_reason(self):
        rows = [row(1.0), row(2.0, terminal=True, reason="forbidden_contact")]
        rows[0]["environments"][0].update({
            "self_collision_enabled": True,
            "cross_hand_contact_load_n": 0.0,
            "external_forbidden_contact_load_n": 0.0,
        })
        rows[1]["environments"][0].update({
            "self_collision_enabled": True,
            "cross_hand_contact_load_n": 12.5,
            "external_forbidden_contact_load_n": 0.0,
        })
        report = {
            "command_fault_experiment": {
                "scenario": "hand_collision",
                "parameters": {
                    "target_wrist_position_world_m": [0.4, 0.0, 1.0],
                    "max_cartesian_error_m": 0.04,
                },
                "triggered": True,
                "events": [{
                    "event": "both_hands_converge_command_started",
                    "issued_at_simulation_time_s": 1.5,
                }],
                "physical_state_force_sensor_or_termination_injection": False,
            },
            "terminal_outcomes": [{
                "terminal": True, "timestamp_s": 2.0,
                "termination_reason": "forbidden_contact",
            }],
            "trace_every_control_steps": 1,
            "forbidden_contact_force_threshold_n": 1.0,
            "self_collision_enabled": True,
            "provenance": {"physics_dt_s": 1 / 120, "control_dt_s": 1 / 30},
        }
        self.assertEqual(validate_report(report, rows)["scenario"], "hand_collision")
        report["terminal_outcomes"][0]["termination_reason"] = "episode_timeout"
        rows[-1]["environments"][0]["termination_reason"] = "episode_timeout"
        with self.assertRaises(EvidenceError):
            validate_report(report, rows)

    def test_hand_collision_rejects_external_or_preexisting_contact(self):
        rows = [row(1.0), row(2.0, terminal=True, reason="forbidden_contact")]
        for item, cross, external in ((rows[0], 0.0, 0.0), (rows[1], 5.0, 1.1)):
            item["environments"][0].update({
                "self_collision_enabled": True,
                "cross_hand_contact_load_n": cross,
                "external_forbidden_contact_load_n": external,
            })
        report = {
            "command_fault_experiment": {
                "scenario": "hand_collision",
                "parameters": {
                    "target_wrist_position_world_m": [0.4, 0.0, 1.0],
                    "max_cartesian_error_m": 0.01,
                },
                "triggered": True,
                "events": [{
                    "event": "both_hands_converge_command_started",
                    "issued_at_simulation_time_s": 1.5,
                }],
                "physical_state_force_sensor_or_termination_injection": False,
            },
            "terminal_outcomes": [{
                "terminal": True, "timestamp_s": 2.0,
                "termination_reason": "forbidden_contact",
            }],
            "trace_every_control_steps": 1,
            "forbidden_contact_force_threshold_n": 1.0,
            "self_collision_enabled": True,
            "provenance": {"physics_dt_s": 1 / 120, "control_dt_s": 1 / 30},
        }
        with self.assertRaises(EvidenceError):
            validate_report(report, rows)
        rows[1]["environments"][0]["external_forbidden_contact_load_n"] = 0.0
        rows[0]["environments"][0]["cross_hand_contact_load_n"] = 0.01
        with self.assertRaises(EvidenceError):
            validate_report(report, rows)


if __name__ == "__main__":
    unittest.main()
