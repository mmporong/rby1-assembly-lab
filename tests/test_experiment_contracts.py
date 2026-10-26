from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from rby1_assembly.experiment_contracts import (
    load_finite_json,
    apply_runtime_timing,
    validate_ppo_config,
    verify_rj45_nominal_evidence,
    verify_physics_sample_stability,
)
from rby1_assembly.rj45_task import verify_recorded_mating_geometry


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _synthetic_contract_fixture(project: Path) -> tuple[dict, Path, Path]:
    """Create CPU-only contract data; this is not a training demonstration."""
    source = project / "src/rby1_assembly/fixture.py"
    run_sim = project / "scripts/run_sim.py"
    source.parent.mkdir(parents=True)
    run_sim.parent.mkdir(parents=True)
    source.write_text("VALUE = 1\n", encoding="utf-8")
    run_sim.write_text("VALUE = 2\n", encoding="utf-8")

    configs = project / "configs"
    configs.mkdir()
    runtime_versions = {
        "isaacsim": "5.1.0.0",
        "isaaclab": "0.47.2",
        "isaaclab-rl": "0.4.4",
        "rsl-rl-lib": "3.0.1",
        "torch": "2.7.0+cu128",
        "numpy": "1.26.0",
        "h5py": "3.14.0",
    }
    runtime = {
        "isaac_sim_distribution": runtime_versions["isaacsim"],
        "isaac_lab_distribution": runtime_versions["isaaclab"],
        "isaac_lab_rl": runtime_versions["isaaclab-rl"],
        "rsl_rl": runtime_versions["rsl-rl-lib"],
        "torch": "2.7.0",
        "torch_cuda_build": "cu128",
        "numpy": runtime_versions["numpy"],
        "h5py": runtime_versions["h5py"],
        "physics_dt_s": 1 / 120,
        "control_decimation": 4,
        "control_dt_s": 1 / 30,
    }
    runtime_path = configs / "runtime.json"
    assets_config_path = configs / "assets.json"
    connector_assets_path = configs / "rj45-assets.json"
    _write_json(runtime_path, runtime)
    _write_json(assets_config_path, {"schema_version": 1, "scope": "robot"})
    _write_json(
        connector_assets_path, {"schema_version": 1, "scope": "connector"}
    )

    generated = project / "assets/generated/tasks/rj45-v1"
    generated.mkdir(parents=True)
    asset_paths = {
        name: generated / f"{name}.usda"
        for name in ("robot", "socket", "plug", "table")
    }
    for name, path in asset_paths.items():
        path.write_text(f"#{name}\n", encoding="utf-8")
    scene_path = generated / "scene.usda"
    scene_path.write_text("#scene\n", encoding="utf-8")
    asset_hashes = {
        path.name: _sha256(path) for path in (*asset_paths.values(), scene_path)
    }
    build_path = generated / "build-report.json"
    _write_json(build_path, {"files": asset_hashes})

    identity_frame = {
        "position_m": [0.0, 0.0, 0.0],
        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
    }
    task = {
        "_config_sha256": "a" * 64,
        "assets": {
            **{name: str(path.relative_to(project)) for name, path in asset_paths.items()},
            "build_report": str(build_path.relative_to(project)),
        },
        "frames": {
            "socket_entry": copy.deepcopy(identity_frame),
            "plug_tip": copy.deepcopy(identity_frame),
        },
        "evaluation": {
            "assembled_depth_m": 0.012,
            "max_depth_m": 0.013,
            "max_lateral_error_m": 0.00005,
            "max_orientation_error_rad": 0.008726646259971648,
            "max_contact_force_n": 25.0,
            "max_relative_linear_speed_m_s": 0.005,
            "max_relative_angular_speed_rad_s": 0.05,
            "dwell_s": 0.4,
        },
    }
    socket_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    plug_pose = [0.0, 0.012, 0.0, 1.0, 0.0, 0.0, 0.0]
    geometry = verify_recorded_mating_geometry(
        socket_pose,
        plug_pose,
        task["frames"]["socket_entry"],
        task["frames"]["plug_tip"],
        0.012,
        0.0,
        0.0,
    )

    def environment(index: int, timestamp: float) -> dict:
        terminal = index == 13
        entry = {
            "env_index": 0,
            "timestamp_s": timestamp,
            "phase_name": "done" if terminal else "insert",
            "terminal": terminal,
            "termination_reason": "independent_success" if terminal else "none",
            "object_poses_world_wxyz": {"socket": socket_pose, "plug": plug_pose},
            "object_substep_poses_world_wxyz": [[socket_pose, plug_pose] for _ in range(4)],
            "object_substep_link_velocities_world": [[[0.0] * 6, [0.0] * 6] for _ in range(4)],
            "object_link_velocities_world": [[0.0] * 6, [0.0] * 6],
            "actual_depth_m": 0.012,
            "actual_lateral_error_m": 0.0,
            "actual_orientation_error_rad": 0.0,
            "independent_mating_geometry": geometry,
            "grip_live": [True, True],
            "grip_proven": [True, True],
            "independent_eval_valid": True,
            "independent_eval_success": terminal,
            "independent_eval_dwell_s": 0.4 if terminal else 0.0,
            "independent_eval_criteria": {
                "depth": True,
                "lateral": True,
                "orientation": True,
                "grasp_live_and_proven": True,
                "dwell": terminal,
                "force": True,
                "forbidden_contact": True,
            },
            "mating_stability": {
                "depth_within_window": True,
                "physics_substeps_complete": True,
                "physics_all_samples_stable": True,
                "relative_linear_speed_ok": True,
                "relative_angular_speed_ok": True,
                "relative_linear_speed_m_s": 0.0,
                "relative_angular_speed_rad_s": 0.0,
            },
            "grip_contact_force_peak_n": 3.0,
            "insertion_contact_load_peak_n": 4.0,
            "forbidden_contact_load_peak_n": 0.0,
            "pregrasp_clearance_reached": True,
            "pregrasp_above_reached": True,
        }
        if index == 0:
            entry["independent_eval_valid"] = None
            entry.pop("independent_eval_criteria")
            entry.pop("mating_stability")
        return entry

    trace_path = project / "evidence/nominal.jsonl"
    trace_path.parent.mkdir(parents=True)
    records = [
        {
            "schema_version": 1,
            "control_step": index,
            "environments": [environment(index, timestamp)],
        }
        for index, timestamp in enumerate(index / 30 for index in range(14))
    ]
    trace_path.write_text(
        "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
        encoding="utf-8",
    )
    report = {
        "task_id": "rj45-v1",
        "completed": True,
        "runtime_validated": True,
        "num_envs": 1,
        "command_fault_experiment": None,
        "self_collision_enabled": True,
        "independent_evaluator": {
            "source": "simulator_ground_truth_independent",
            "version": "tip-entry-geometry-contact-v1",
        },
        "trace_every_control_steps": 1,
        "trace_path": str(trace_path.relative_to(project)),
        "trace_sha256": _sha256(trace_path),
        "trace_record_count": len(records),
        "actual_steps": len(records) - 1,
        "forbidden_contact_force_threshold_n": 1.0,
        "terminal_outcomes": [records[-1]["environments"][0]],
        "provenance": {
            "input_immutability_verified": True,
            "runtime_config_sha256": _sha256(runtime_path),
            "assets_config_sha256": _sha256(assets_config_path),
            "connector_assets_config_sha256": _sha256(connector_assets_path),
            "runtime_versions": runtime_versions,
            "physics_dt_s": runtime["physics_dt_s"],
            "control_decimation": runtime["control_decimation"],
            "control_dt_s": runtime["control_dt_s"],
            "task_config_sha256": task["_config_sha256"],
            "build_report_sha256": _sha256(build_path),
            "generated_asset_sha256": asset_hashes,
            "source_sha256": {
                str(source.relative_to(project)): _sha256(source),
                str(run_sim.relative_to(project)): _sha256(run_sim),
            },
        },
    }
    report_path = project / "evidence/nominal-report.json"
    _write_json(report_path, report)
    return task, report_path, trace_path


class RuntimeTimingTests(unittest.TestCase):
    def test_refinement_keeps_control_rate_fixed(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(sim=SimpleNamespace())
        apply_runtime_timing(cfg, {"physics_dt_s": 1 / 240, "control_decimation": 8, "control_dt_s": 1 / 30})
        self.assertEqual(cfg.sim.dt, 1 / 240)
        self.assertEqual(cfg.decimation, 8)
        self.assertEqual(cfg.sim.render_interval, 8)

    def test_invalid_timing_is_rejected(self):
        from types import SimpleNamespace
        baseline = {"physics_dt_s": 1 / 120, "control_decimation": 4, "control_dt_s": 1 / 30}
        for key, value in (("physics_dt_s", 0), ("physics_dt_s", float("nan")),
                           ("control_decimation", True), ("control_decimation", 0),
                           ("control_decimation", 4.5), ("control_dt_s", 0.02)):
            runtime = dict(baseline, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                apply_runtime_timing(SimpleNamespace(sim=SimpleNamespace()), runtime)


class FiniteJsonTests(unittest.TestCase):
    def test_named_nonfinite_constant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nan.json"
            path.write_text('{"value": NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                load_finite_json(path)

    def test_overflowing_number_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overflow.json"
            path.write_text('{"value": 1e999}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                load_finite_json(path)

    def test_ppo_config_rejects_nested_nonfinite_value(self) -> None:
        config = {
            "max_iterations": 1,
            "num_steps_per_env": 1,
            "save_interval": 1,
            "seed": 0,
            "clip_actions": 1.0,
            "policy": {
                "init_noise_std": 0.1,
                "actor_hidden_dims": [32],
                "critic_hidden_dims": [32],
                "unexpected": float("nan"),
            },
            "algorithm": {
                "learning_rate": 1e-3,
                "clip_param": 0.2,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
                "gamma": 0.99,
                "lam": 0.95,
                "entropy_coef": 0.0,
                "value_loss_coef": 1.0,
                "num_learning_epochs": 1,
                "num_mini_batches": 1,
            },
        }
        with self.assertRaises(ValueError):
            validate_ppo_config(config)


class NominalEvidenceTests(unittest.TestCase):
    def test_complete_current_bundle_passes_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            result = verify_rj45_nominal_evidence(project, task, report_path)
            self.assertEqual(result["report_sha256"], _sha256(report_path))
            self.assertEqual(result["trace_sha256"], _sha256(trace_path))
            self.assertEqual(result["trace_records"], 14)

    def test_fake_completed_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            report = load_finite_json(report_path)
            report["terminal_outcomes"][0]["independent_eval_success"] = False
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "continuous independent"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_failed_run_report_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            report = load_finite_json(report_path)
            report["completed"] = False
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "completed single-environment"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_incomplete_terminal_criteria_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            del records[-1]["environments"][0]["independent_eval_criteria"]["force"]
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            report["terminal_outcomes"] = [records[-1]["environments"][0]]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "continuous independent"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_stale_task_configuration_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            task["_config_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "task hash is stale"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_unverified_input_immutability_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            report = load_finite_json(report_path)
            del report["provenance"]["input_immutability_verified"]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "input immutability"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_missing_runtime_versions_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            report = load_finite_json(report_path)
            del report["provenance"]["runtime_versions"]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "runtime versions"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_swapped_runtime_distribution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            report = load_finite_json(report_path)
            report["provenance"]["runtime_versions"]["numpy"] = "2.0.0"
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "pinned runtime"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_swapped_config_hashes_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            report = load_finite_json(report_path)
            provenance = report["provenance"]
            provenance["assets_config_sha256"], provenance[
                "connector_assets_config_sha256"
            ] = (
                provenance["connector_assets_config_sha256"],
                provenance["assets_config_sha256"],
            )
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "config_sha256 is stale"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_overflowing_runtime_input_is_rejected_with_current_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            runtime_path = project / "configs/runtime.json"
            runtime_path.write_text('{"physics_dt_s": 1e999}\n', encoding="utf-8")
            report = load_finite_json(report_path)
            report["provenance"]["runtime_config_sha256"] = _sha256(runtime_path)
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_changed_runtime_timing_is_rejected_even_with_current_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, _ = _synthetic_contract_fixture(project)
            runtime_path = project / "configs/runtime.json"
            runtime = load_finite_json(runtime_path)
            runtime["physics_dt_s"] = 0.01
            _write_json(runtime_path, runtime)
            report = load_finite_json(report_path)
            report["provenance"]["runtime_config_sha256"] = _sha256(runtime_path)
            report["provenance"]["physics_dt_s"] = runtime["physics_dt_s"]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "physics and control timing"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_trace_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            trace_path.write_text(trace_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "trace hash mismatch"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_unstable_physics_substep_cannot_hide_behind_stable_endpoint(self) -> None:
        for component, speed in ((0, 0.006), (3, 0.06)):
            with self.subTest(component=component), tempfile.TemporaryDirectory() as directory:
                project = Path(directory)
                task, report_path, trace_path = _synthetic_contract_fixture(project)
                records = [json.loads(line) for line in trace_path.read_text().splitlines()]
                entry = records[-1]["environments"][0]
                entry["object_substep_link_velocities_world"][1][1][component] = speed
                self.assertFalse(verify_physics_sample_stability(entry, task, 4))
                trace_path.write_text("".join(json.dumps(r) + "\n" for r in records))
                report = load_finite_json(report_path)
                report["trace_sha256"] = _sha256(trace_path)
                report["terminal_outcomes"] = [entry]
                _write_json(report_path, report)
                with self.assertRaisesRegex(ValueError, "continuous independent"):
                    verify_rj45_nominal_evidence(project, task, report_path)

    def test_malformed_physics_samples_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, _, trace_path = _synthetic_contract_fixture(project)
            baseline = json.loads(trace_path.read_text().splitlines()[-1])["environments"][0]
            variants = []
            missing = copy.deepcopy(baseline)
            del missing["object_substep_link_velocities_world"]
            variants.append(missing)
            short = copy.deepcopy(baseline)
            short["object_substep_poses_world_wxyz"].pop()
            variants.append(short)
            nonunit = copy.deepcopy(baseline)
            nonunit["object_substep_poses_world_wxyz"][2][1][3] = 2.0
            variants.append(nonunit)
            nonfinite = copy.deepcopy(baseline)
            nonfinite["object_substep_link_velocities_world"][2][1][3] = float("inf")
            variants.append(nonfinite)
            for entry in variants:
                with self.assertRaisesRegex(ValueError, "physics sample"):
                    verify_physics_sample_stability(entry, task, 4)

    def test_force_limit_cannot_be_overridden_by_success_flags(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            records[-1]["environments"][0]["insertion_contact_load_peak_n"] = 25.1
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            report["terminal_outcomes"] = [records[-1]["environments"][0]]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "continuous independent"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_claimed_dwell_requires_contiguous_trace_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            for record in records[1:6]:
                entry = record["environments"][0]
                entry["grip_live"] = [False, False]
                entry["independent_eval_criteria"]["grasp_live_and_proven"] = False
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            report["terminal_outcomes"] = [records[-1]["environments"][0]]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "continuous independent"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_missing_trace_step_is_rejected_even_with_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            del records[6]
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            report["trace_record_count"] = len(records)
            report["actual_steps"] = len(records) - 1
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "contiguous one-environment"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_inflated_trace_timestamp_is_rejected_with_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            records[6]["environments"][0]["timestamp_s"] += 0.005
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "contiguous control_dt"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_raw_pose_and_reported_geometry_disagreement_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            records = [json.loads(line) for line in trace_path.read_text().splitlines()]
            records[-1]["environments"][0]["object_poses_world_wxyz"]["plug"][1] = 0.0125
            trace_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
            )
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            report["terminal_outcomes"] = [records[-1]["environments"][0]]
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "geometry disagree"):
                verify_rj45_nominal_evidence(project, task, report_path)

    def test_nonfinite_trace_value_is_rejected_even_with_matching_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            task, report_path, trace_path = _synthetic_contract_fixture(project)
            lines = trace_path.read_text(encoding="utf-8").splitlines()
            record = json.loads(lines[1])
            record["environments"][0]["timestamp_s"] = "OVERFLOW"
            lines[1] = json.dumps(record).replace(
                '"timestamp_s": "OVERFLOW"', '"timestamp_s": 1e999'
            )
            trace_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            report = load_finite_json(report_path)
            report["trace_sha256"] = _sha256(trace_path)
            _write_json(report_path, report)
            with self.assertRaisesRegex(ValueError, "finite JSON"):
                verify_rj45_nominal_evidence(project, task, report_path)


class TrainingPreflightTests(unittest.TestCase):
    def test_rj45_missing_nominal_gate_stops_before_launcher(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            fake_modules = Path(directory)
            isaaclab = fake_modules / "isaaclab"
            isaaclab.mkdir()
            (isaaclab / "__init__.py").write_text("", encoding="utf-8")
            (isaaclab / "app.py").write_text(
                "class AppLauncher:\n"
                "    @staticmethod\n"
                "    def add_app_launcher_args(parser):\n"
                "        parser.add_argument('--device', default='cpu')\n"
                "    def __init__(self, args):\n"
                "        raise AssertionError('launcher started')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(fake_modules), str(root / "src"))
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(root / "scripts/train.py"),
                    "--task",
                    "rj45",
                    "--num-envs",
                    "1",
                ],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--rj45-nominal-evidence is required", completed.stderr)
        self.assertNotIn("launcher started", completed.stderr)

    def test_missing_rj45_physical_evidence_stops_before_launcher(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            fake_modules = Path(directory)
            isaaclab = fake_modules / "isaaclab"
            isaaclab.mkdir()
            (isaaclab / "__init__.py").write_text("", encoding="utf-8")
            (isaaclab / "app.py").write_text(
                "class AppLauncher:\n"
                "    @staticmethod\n"
                "    def add_app_launcher_args(parser):\n"
                "        parser.add_argument('--device', default='cpu')\n"
                "    def __init__(self, args):\n"
                "        raise AssertionError('launcher started')\n",
                encoding="utf-8",
            )
            environment = dict(os.environ)
            environment["PYTHONPATH"] = os.pathsep.join(
                (str(fake_modules), str(root / "src"))
            )
            completed = subprocess.run(
                [sys.executable, str(root / "scripts/train.py"), "--task", "rj45"],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--rj45-nominal-evidence", completed.stderr)
        self.assertNotIn("launcher started", completed.stderr)


if __name__ == "__main__":
    unittest.main()
