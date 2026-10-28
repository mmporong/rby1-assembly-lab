from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
from typing import Any
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
    function_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in function_defs} != names:
        raise AssertionError("native PhysX reset helpers are missing")
    functions: list[ast.stmt] = [*function_defs]
    namespace: dict[str, Any] = {"torch": _TorchShim}
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


def _load_bank_contract_helpers():
    tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
    names = {
        "_bank_cycle_indices",
        "_validate_bank_cli_contract",
        "_resolve_bank_grasp_ids",
        "_assert_bank_identity_unchanged",
        "_assert_origin_applied_once",
        "_assert_bank_check_coverage",
    }
    function_defs = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in function_defs} != names:
        raise AssertionError("held-bank checker contract helpers are missing")
    functions: list[ast.stmt] = [*function_defs]
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), CHECKER_PATH, "exec"),
        namespace,
    )
    return namespace


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

    def test_large_single_target_reset_preserves_all_untouched_physx_rows(self) -> None:
        for num_envs in (64, 128, 256):
            with self.subTest(num_envs=num_envs):
                env = _fake_environment(num_envs=num_envs)
                untouched = np.arange(1, num_envs)
                before = self.snapshot(env, untouched)
                env.robot.root_physx_view.dof_positions.value[0, 0] = -1.0
                env.socket.root_physx_view.transforms.value[0, 0] = -1.0
                env.plug.root_physx_view.velocities.value[0, 0] = -1.0
                self.assertEqual(len(untouched), num_envs - 1)
                self.assert_untouched(before, env, untouched)


class ParallelCliContractTests(unittest.TestCase):
    def test_num_env_choices_include_scale_targets_and_parallel_gate_applies(self) -> None:
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
        self.assertEqual(ast.literal_eval(choices), (1, 4, 64, 128, 256))

        required = _load_parallel_requirement_helper()
        self.assertFalse(required(1))
        for num_envs in (4, 64, 128, 256):
            with self.subTest(num_envs=num_envs):
                self.assertTrue(required(num_envs))
        for invalid in (0, 2, 65, 127, 129, 255, 257, True):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                required(invalid)

        source = CHECKER_PATH.read_text(encoding="utf-8")
        self.assertEqual(
            source.count("_requires_parallel_contact_evidence(args.num_envs)"), 2
        )
        self.assertNotIn("args.num_envs == 4", source)

    def test_app_launcher_disables_multi_gpu_for_selected_device(self) -> None:
        tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
        launcher_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AppLauncher"
        )
        multi_gpu = next(
            keyword.value
            for keyword in launcher_call.keywords
            if keyword.arg == "multi_gpu"
        )
        self.assertIs(ast.literal_eval(multi_gpu), False)

    def test_report_separates_n4_contact_proof_from_scaled_reset_measurement(self) -> None:
        source = CHECKER_PATH.read_text(encoding="utf-8")
        self.assertIn("preexisting N4 contact-routing evidence", source)
        self.assertIn('"measurement_num_envs": args.num_envs', source)
        self.assertIn('"untouched_environments_per_cycle"', source)
        self.assertIn("'quarter-bank' if bank_mode else 'one-environment'", source)

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


class HeldBankCheckerContractTests(unittest.TestCase):
    def test_bank_mode_requires_n64_or_n256_and_exactly_four_cycles(self) -> None:
        validate = _load_bank_contract_helpers()["_validate_bank_cli_contract"]
        self.assertFalse(validate("held.json", None, num_envs=1, reset_cycles=1000))
        for num_envs in (64, 256):
            self.assertTrue(
                validate(None, "bank.json", num_envs=num_envs, reset_cycles=4)
            )
        for invalid in (1, 4, 128):
            with self.subTest(num_envs=invalid), self.assertRaisesRegex(
                ValueError, "64 or 256"
            ):
                validate(None, "bank.json", num_envs=invalid, reset_cycles=4)
        with self.assertRaisesRegex(ValueError, "reset-cycles 4"):
            validate(None, "bank.json", num_envs=64, reset_cycles=3)
        with self.assertRaisesRegex(ValueError, "physical-warmup-cycles"):
            validate(
                None,
                "bank.json",
                num_envs=64,
                reset_cycles=4,
                mode="reset-api-stress",
            )
        for held, bank in ((None, None), ("held.json", "bank.json")):
            with self.assertRaisesRegex(ValueError, "exactly one"):
                validate(held, bank)

    def test_four_partial_cycles_cover_every_environment_once(self) -> None:
        indices = _load_bank_contract_helpers()["_bank_cycle_indices"]
        for num_envs in (64, 256):
            groups = [indices(num_envs, cycle) for cycle in range(4)]
            self.assertTrue(all(0 < len(group) < num_envs for group in groups))
            self.assertEqual(
                sorted(value for group in groups for value in group),
                list(range(num_envs)),
            )

    def test_stale_grasp_identity_is_rejected(self) -> None:
        resolve = _load_bank_contract_helpers()["_resolve_bank_grasp_ids"]
        self.assertEqual(
            resolve([1, 0], ["grasp-a", "grasp-b"]),
            ["grasp-b", "grasp-a"],
        )
        with self.assertRaisesRegex(ValueError, "stale"):
            resolve(
                [1, 0],
                ["grasp-a", "grasp-b"],
                ["grasp-a", "grasp-b"],
            )

    def test_artifact_hash_drift_and_double_origin_are_rejected(self) -> None:
        helpers = _load_bank_contract_helpers()
        unchanged = helpers["_assert_bank_identity_unchanged"]
        start = {
            "manifest_sha256": "a" * 64,
            "source_contract_sha256": "b" * 64,
            "artifact_sha256": ["c" * 64],
        }
        unchanged(start, dict(start), 1, 1)
        changed = {**start, "artifact_sha256": ["d" * 64]}
        with self.assertRaisesRegex(ValueError, "identity changed"):
            unchanged(start, changed, 1, 1)

        check_origin = helpers["_assert_origin_applied_once"]
        local = [[[0.1, 0.2, 0.3]]]
        origins = [[10.0, 20.0, 30.0]]
        check_origin([[[10.1, 20.2, 30.3]]], local, origins)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            check_origin([[[20.1, 40.2, 60.3]]], local, origins)

    def test_restoration_records_require_exact_groups_and_physics_boundary(self) -> None:
        helpers = _load_bank_contract_helpers()
        checks = []
        for cycle, ids in [(-1, list(range(64)))] + [
            (cycle, list(helpers["_bank_cycle_indices"](64, cycle))) for cycle in range(4)
        ]:
            checks.append({
                "environment_ids": ids, "reset_cycle": cycle,
                "control_step": (cycle + 2) * 10,
                "physics_step_before_restore": (cycle + 2) * 80,
                "physics_step_after_check": (cycle + 2) * 80,
                "required_staging_control_steps": 2,
                "staging_before_restore": {"queued": [True] * len(ids), "restored": [False] * len(ids), "waited_control_steps": [2] * len(ids)},
                "staging_after_restore": {"queued": [False] * len(ids), "restored": [True] * len(ids), "waited_control_steps": [2] * len(ids)},
            })
        validate = helpers["_assert_bank_check_coverage"]
        validate(checks, 64)
        for key, value in (
            ("environment_ids", checks[1]["environment_ids"]),
            ("reset_cycle", 0),
            ("control_step", 0),
            ("physics_step_after_check", 999),
            ("staging_before_restore", {"queued": [True] * 16, "restored": [False] * 16, "waited_control_steps": [1] * 16}),
            ("staging_after_restore", {"queued": [False] * 16, "restored": [False] * 16, "waited_control_steps": [2] * 16}),
        ):
            with self.subTest(key=key):
                corrupted = copy.deepcopy(checks)
                corrupted[2][key] = value
                with self.assertRaises(RuntimeError):
                    validate(corrupted, 64)


@unittest.skipUnless(importlib.util.find_spec("torch"), "requires torch CPU")
class HeldBankIndependentSourceTests(unittest.TestCase):
    def _fixture(self):
        import torch

        bank = [{
            "joint_names": ["b", "a", "d", "c"],
            "state": {
                "joint_positions_si": [0.2 + i, 0.1 + i, 0.4 + i, 0.3 + i],
                "joint_velocities_si": [0.02, 0.01, 0.04, 0.03],
                "joint_reference_si": [0.22 + i, 0.11 + i, 0.44 + i, 0.33 + i],
                "object_poses_world_wxyz": [[0.1 + i, 0.2, 0.3, 1, 0, 0, 0], [0.4, 0.5 + i, 0.6, 1, 0, 0, 0]],
                "object_com_velocities_world": [[0.01] * 6, [0.02] * 6],
            },
        } for i in range(2)]
        proof = {"selected_grasp_ids": ["g0", "g1"], "selected_artifact_sha256": ["a" * 64, "b" * 64]}
        cache = {
            name: torch.tensor([[row["state"][field][i] for i in (1, 0, 3, 2)] for row in bank])
            for name, field in (("position", "joint_positions_si"), ("velocity", "joint_velocities_si"), ("reference", "joint_reference_si"))
        }
        cache["local_pose"] = torch.tensor([row["state"]["object_poses_world_wxyz"] for row in bank])
        cache["object_velocity"] = torch.tensor([row["state"]["object_com_velocities_world"] for row in bank])
        origins = torch.tensor([[0.0, 0, 0], [3.0, 4.0, 0]])
        def object_state(ids):
            poses = cache["local_pose"][ids].clone()
            poses[:, :, :3] += origins[ids, None, :]
            return poses, cache["object_velocity"][ids]
        env = types.SimpleNamespace(
            device="cpu", robot=types.SimpleNamespace(joint_names=["a", "b", "c", "d"]),
            scene=types.SimpleNamespace(env_origins=origins),
            _training_grasp_index=torch.tensor([0, 1]),
            _training_episode_index=torch.tensor([0, 0]),
            _training_held_bank_proof=copy.deepcopy(proof),
            _training_held_bank_object_pose=cache["local_pose"],
            _held_artifact_joint_state=lambda ids: tuple(cache[key][ids] for key in ("position", "velocity", "reference")),
            _held_artifact_object_state=object_state,
            _joint_targets=cache["reference"].clone(),
            _insert_joint_targets=cache["reference"][:, :2].clone(),
            _torso_hold_targets=cache["reference"][:, 2:3].clone(),
            _gripper_command=cache["reference"][:, 3:4].clone(),
            _arm_joint_ids=[0, 1], _torso_joint_ids=[2], _gripper_joint_ids=[3],
        )
        def native_state(_env, ids):
            # Deliberately share corrupted cache/native values: independent source must reject them.
            poses, velocities = object_state(ids)
            transform = poses[:, :, (0, 1, 2, 4, 5, 6, 3)]
            return {
                "robot_joint_position": cache["position"][ids].clone(),
                "robot_joint_velocity": cache["velocity"][ids].clone(),
                "socket_root_state": torch.cat((transform[:, 0], velocities[:, 0]), dim=1),
                "plug_root_state": torch.cat((transform[:, 1], velocities[:, 1]), dim=1),
            }
        tree = ast.parse(CHECKER_PATH.read_text(encoding="utf-8"))
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_resolve_bank_grasp_ids", "_assert_selected_bank_initial_state"}]
        namespace = {
            "torch": torch, "bank_mode": True, "bank_proof": proof,
            "verified_bank_reference": copy.deepcopy(bank),
            "task": {"data": {"evaluation_seed": 0}}, "_native_physx_state": native_state,
        }
        exec(compile(ast.Module(body=definitions, type_ignores=[]), CHECKER_PATH, "exec"), namespace)
        return namespace["_assert_selected_bank_initial_state"], env, cache, torch.tensor([0, 1])

    def test_verified_source_reorders_joints_and_origin_once(self) -> None:
        check, env, _, ids = self._fixture()
        result = check(env, ids)
        self.assertEqual(result["selected_grasp_ids"], ["g0", "g1"])
        self.assertEqual(result["selected_artifact_sha256"], ["a" * 64, "b" * 64])

    def test_mutating_cache_and_native_together_does_not_hide_wrong_bank_row(self) -> None:
        for field in ("position", "velocity", "reference", "local_pose", "object_velocity"):
            with self.subTest(field=field):
                check, env, cache, ids = self._fixture()
                cache[field][0].reshape(-1)[0] += 0.1
                with self.assertRaisesRegex(RuntimeError, "verified source artifact"):
                    check(env, ids)

    def test_stale_schedule_identity_and_actuator_targets_fail(self) -> None:
        for kind in ("selection", "identity", "target"):
            with self.subTest(kind=kind):
                check, env, _, ids = self._fixture()
                if kind == "selection":
                    env._training_grasp_index[:] = 0
                elif kind == "identity":
                    env._training_held_bank_proof["selected_grasp_ids"].reverse()
                else:
                    env._joint_targets[0, 0] += 0.1
                with self.assertRaises((ValueError, RuntimeError)):
                    check(env, ids)


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
