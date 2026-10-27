from __future__ import annotations

import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKER_PATH = PROJECT_ROOT / "scripts/check_rj45_reset.py"


class _Tensor:
    def __init__(self, value) -> None:
        self.value = np.asarray(value)

    def __getitem__(self, index):
        if isinstance(index, _Tensor):
            index = index.value
        return _Tensor(self.value[index])

    def clone(self):
        return _Tensor(self.value.copy())


class _TorchShim:
    Tensor = _Tensor

    @staticmethod
    def cat(values, dim=-1):
        return _Tensor(np.concatenate([value.value for value in values], axis=dim))

    @staticmethod
    def equal(left, right):
        return np.array_equal(left.value, right.value)


class _PhysxView:
    def __init__(
        self, offset: float, *, articulated: bool = False, num_envs: int = 4
    ) -> None:
        self.transforms = _Tensor(
            np.arange(num_envs * 7, dtype=np.float64).reshape(num_envs, 7) + offset
        )
        self.velocities = _Tensor(
            np.arange(num_envs * 6, dtype=np.float64).reshape(num_envs, 6) + offset
        )
        if articulated:
            self.dof_positions = _Tensor(
                np.arange(num_envs * 5, dtype=np.float64).reshape(num_envs, 5) + offset
            )
            self.dof_velocities = _Tensor(
                np.arange(num_envs * 5, dtype=np.float64).reshape(num_envs, 5)
                + offset
                + 100.0
            )

    def get_dof_positions(self):
        return self.dof_positions

    def get_dof_velocities(self):
        return self.dof_velocities

    def get_root_transforms(self):
        return self.transforms

    def get_root_velocities(self):
        return self.velocities

    def get_transforms(self):
        return self.transforms

    def get_velocities(self):
        return self.velocities


def _load_native_physx_helpers():
    tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
    names = {"_native_physx_state", "_assert_native_physx_untouched_equal"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in functions} != names:
        raise AssertionError("native PhysX reset helpers are missing")
    namespace = {"torch": _TorchShim}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), CHECKER_PATH, "exec"),
        namespace,
    )
    return namespace


def _load_parallel_requirement_helper():
    tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_requires_parallel_contact_evidence"
    )
    namespace = {}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), CHECKER_PATH, "exec"),
        namespace,
    )
    return namespace["_requires_parallel_contact_evidence"]


def _fake_environment(num_envs: int = 4):
    return types.SimpleNamespace(
        robot=types.SimpleNamespace(
            root_physx_view=_PhysxView(0.0, articulated=True, num_envs=num_envs)
        ),
        socket=types.SimpleNamespace(
            root_physx_view=_PhysxView(1000.0, num_envs=num_envs)
        ),
        plug=types.SimpleNamespace(
            root_physx_view=_PhysxView(2000.0, num_envs=num_envs)
        ),
    )


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def _make_preimport_repository(directory: str) -> tuple[Path, Path, Path, Path]:
    root = Path(directory) / "project"
    checker = root / "scripts/check_rj45_reset.py"
    provenance = root / "src/rby1_assembly/run_provenance.py"
    checker.parent.mkdir(parents=True)
    provenance.parent.mkdir(parents=True)
    shutil.copyfile(CHECKER_PATH, checker)
    shutil.copyfile(PROJECT_ROOT / "src/rby1_assembly/run_provenance.py", provenance)
    (provenance.parent / "__init__.py").write_text("", encoding="utf-8")

    artifacts = root / "artifacts"
    artifacts.mkdir()
    trace = artifacts / "nominal-trace.json"
    trace.write_bytes(b'{"locked":"trace-input"}\n')
    nominal = artifacts / "nominal.json"
    _write_json(nominal, {"trace_path": "artifacts/nominal-trace.json"})
    held = artifacts / "held.json"
    _write_json(
        held,
        {
            "provenance": {
                "nominal_report_path": "artifacts/nominal.json",
                "nominal_trace_path": "artifacts/nominal-trace.json",
            }
        },
    )
    return root, nominal, held, trace


def _run_preimport_checker(
    root: Path, nominal: Path, held: Path, report: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/check_rj45_reset.py"),
            "--num-envs",
            "1",
            "--nominal",
            str(nominal),
            "--held",
            str(held),
            "--report",
            str(report),
        ],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


class NativePhysxIsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        helpers = _load_native_physx_helpers()
        cls.snapshot = staticmethod(helpers["_native_physx_state"])
        cls.assert_untouched = staticmethod(
            helpers["_assert_native_physx_untouched_equal"]
        )

    def test_every_native_state_field_mutation_is_detected(self) -> None:
        mutations = {
            "robot_joint_position": (
                "robot_joint_position",
                lambda env: env.robot.root_physx_view.dof_positions.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "robot_joint_velocity": (
                "robot_joint_velocity",
                lambda env: env.robot.root_physx_view.dof_velocities.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "robot_root_transform": (
                "robot_root_state",
                lambda env: env.robot.root_physx_view.transforms.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "robot_root_velocity": (
                "robot_root_state",
                lambda env: env.robot.root_physx_view.velocities.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "socket_root_transform": (
                "socket_root_state",
                lambda env: env.socket.root_physx_view.transforms.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "socket_root_velocity": (
                "socket_root_state",
                lambda env: env.socket.root_physx_view.velocities.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "plug_root_transform": (
                "plug_root_state",
                lambda env: env.plug.root_physx_view.transforms.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
            "plug_root_velocity": (
                "plug_root_state",
                lambda env: env.plug.root_physx_view.velocities.value.__setitem__(
                    (2, 0), -1.0
                ),
            ),
        }
        untouched = np.asarray([1, 2, 3])
        for mutation, (field, mutate) in mutations.items():
            with self.subTest(mutation=mutation):
                env = _fake_environment()
                before = self.snapshot(env, untouched)
                mutate(env)
                with self.assertRaisesRegex(RuntimeError, field):
                    self.assert_untouched(before, env, untouched)

    def test_target_environment_change_does_not_fail_untouched_comparison(self) -> None:
        env = _fake_environment()
        untouched = np.asarray([1, 2, 3])
        before = self.snapshot(env, untouched)
        env.robot.root_physx_view.dof_positions.value[0, 0] = -1.0
        env.socket.root_physx_view.transforms.value[0, 0] = -1.0
        env.plug.root_physx_view.velocities.value[0, 0] = -1.0
        self.assert_untouched(before, env, untouched)

    def test_n64_single_target_reset_preserves_63_native_physx_rows(self) -> None:
        env = _fake_environment(num_envs=64)
        untouched = np.arange(1, 64)
        before = self.snapshot(env, untouched)
        env.robot.root_physx_view.dof_positions.value[0, 0] = -1.0
        env.socket.root_physx_view.transforms.value[0, 0] = -1.0
        env.plug.root_physx_view.velocities.value[0, 0] = -1.0
        self.assertEqual(len(untouched), 63)
        self.assert_untouched(before, env, untouched)


class ParallelCliContractTests(unittest.TestCase):
    def test_num_env_choices_include_n64_and_parallel_gate_applies_above_one(self) -> None:
        tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
        num_env_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--num-envs"
        )
        choices = next(
            keyword.value
            for keyword in num_env_call.keywords
            if keyword.arg == "choices"
        )
        self.assertEqual(ast.literal_eval(choices), (1, 4, 64))

        required = _load_parallel_requirement_helper()
        self.assertFalse(required(1))
        self.assertTrue(required(4))
        self.assertTrue(required(64))
        for invalid in (0, 2, 65, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                required(invalid)

        source = CHECKER_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            source.count("_requires_parallel_contact_evidence(args.num_envs)"), 2
        )
        self.assertNotIn("args.num_envs == 4", source)

    def test_report_separates_n4_contact_proof_from_n64_reset_measurement(self) -> None:
        source = CHECKER_PATH.read_text(encoding="utf-8")
        self.assertIn("preexisting N4 contact-routing evidence", source)
        self.assertIn('"measurement_num_envs": args.num_envs', source)
        self.assertIn('"untouched_environments_per_cycle": args.num_envs - 1', source)

    def test_report_limits_state_claims_and_exposes_per_cycle_warmup_counts(self) -> None:
        source = CHECKER_PATH.read_text(encoding="utf-8")
        self.assertIn('"warmup_control_steps_per_physical_cycle"', source)
        for field in (
            "episode_index",
            "episode_length",
            "phase_started",
            "joint_targets",
            "command_depth",
            "previous_action",
        ):
            self.assertIn(f'"{field}"', source)
        for excluded_group in (
            "calibration_bias",
            "held_reset_and_warmup",
            "reward_components_and_action_saturation",
            "policy_observation_cache",
        ):
            self.assertIn(f'"{excluded_group}"', source)
        self.assertIn("does not assert preservation of unlisted training telemetry", source)
        for native_group in (
            "robot_joint_position",
            "robot_joint_velocity",
            "robot_root_state",
            "socket_root_state",
            "plug_root_state",
        ):
            self.assertIn(f'"{native_group}"', source)


class ReportPathPreimportTests(unittest.TestCase):
    def test_invalid_report_paths_fail_before_isaac_import_and_preserve_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, nominal, held, trace = _make_preimport_repository(directory)
            original = {
                nominal: nominal.read_bytes(),
                held: held.read_bytes(),
                trace: trace.read_bytes(),
            }
            cases = {
                "outside": (root.parent / "outside.json", "must stay below"),
                "non_json": (root / "artifacts/report.txt", "must be a JSON file"),
                "nominal": (nominal, "must not overwrite"),
                "held": (held, "must not overwrite"),
            }
            for label, (report, message) in cases.items():
                with self.subTest(label=label):
                    completed = _run_preimport_checker(root, nominal, held, report)
                    self.assertNotEqual(completed.returncode, 0)
                    self.assertIn(message, completed.stderr)
                    self.assertNotIn("isaaclab", completed.stderr)
                    for path, payload in original.items():
                        self.assertEqual(path.read_bytes(), payload)

    def test_report_symlink_to_trace_is_rejected_without_writing_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, nominal, held, trace = _make_preimport_repository(directory)
            trace_before = trace.read_bytes()
            report_link = root / "artifacts/report.json"
            report_link.symlink_to(trace.name)
            completed = _run_preimport_checker(root, nominal, held, report_link)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("must not overwrite an input or source file", completed.stderr)
            self.assertNotIn("isaaclab", completed.stderr)
            self.assertEqual(trace.read_bytes(), trace_before)


if __name__ == "__main__":
    unittest.main()
