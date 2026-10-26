# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Train the 12-D residual policy with the real RSL-RL PPO runner."""

import argparse
from datetime import datetime
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.run_provenance import InputMutationError, capture_run_inputs  # noqa: E402


def _preimport_option(name: str, default: str | None) -> str | None:
    prefix = name + "="
    values = sys.argv[1:]
    for index, value in enumerate(values):
        if value.startswith(prefix):
            return value[len(prefix):]
        if value == name and index + 1 < len(values):
            return values[index + 1]
    return default


_bootstrap_task = _preimport_option("--task", "toy")
_bootstrap_config_default = (
    "configs/ppo-rj45-v1.json" if _bootstrap_task == "rj45" else "configs/ppo.json"
)
_bootstrap_task_path = PROJECT_ROOT / (
    "configs/tasks/rj45-v1.json" if _bootstrap_task == "rj45" else "configs/task.json"
)
_bootstrap_nominal_report = _preimport_option("--rj45-nominal-evidence", None)
_bootstrap_nominal_paths: tuple[Path, ...] = ()
if _bootstrap_task == "rj45" and _bootstrap_nominal_report is not None:
    nominal_report_path = Path(_bootstrap_nominal_report).resolve()
    nominal_report = json.loads(nominal_report_path.read_text(encoding="utf-8"))
    nominal_trace = nominal_report.get("trace_path")
    if not isinstance(nominal_trace, str) or not nominal_trace:
        raise ValueError("RJ45 nominal evidence report must identify its trace_path")
    nominal_trace_path = Path(nominal_trace)
    if not nominal_trace_path.is_absolute():
        nominal_trace_path = PROJECT_ROOT / nominal_trace_path
    _bootstrap_nominal_paths = (nominal_report_path, nominal_trace_path)
_bootstrap_input_snapshot = capture_run_inputs(
    PROJECT_ROOT,
    task_path=_bootstrap_task_path,
    build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
    config_paths=(
        Path(_preimport_option("--config", _bootstrap_config_default)),
        *sorted((PROJECT_ROOT / "configs").glob("*.json")),
        *((PROJECT_ROOT / "configs/rj45-assets.json",) if _bootstrap_task == "rj45" else ()),
    ),
    source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
    source_files=(Path(__file__).resolve(),),
    extra_paths=_bootstrap_nominal_paths,
)

from rby1_assembly.experiment_contracts import (  # noqa: E402
    atomic_write_json,
    load_finite_json,
    validate_ppo_config,
    verify_rj45_nominal_evidence,
)
from rby1_assembly.rj45_task import (  # noqa: E402
    load_rj45_task,
    verify_runtime_versions,
)
from isaaclab.app import AppLauncher  # noqa: E402


parser = argparse.ArgumentParser(description="Train RB-Y1 assembly residual policy")
parser.add_argument("--num-envs", type=int, default=64)
parser.add_argument("--task", choices=("toy", "rj45"), default="toy")
parser.add_argument("--max-iterations", type=int, default=None)
parser.add_argument("--config", type=Path, default=Path("configs/ppo.json"))
parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/rby1_assembly"))
parser.add_argument(
    "--rj45-nominal-evidence",
    type=Path,
    help="completed single-environment nominal RJ45 run report required for RJ45 training",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.task == "rj45" and args.config == Path("configs/ppo.json"):
    args.config = Path("configs/ppo-rj45-v1.json")
if args.num_envs <= 0:
    parser.error("--num-envs must be positive")
if args.task == "rj45" and args.num_envs > 1:
    parser.error("parallel RJ45 collision validation pending; use --num-envs 1")
if args.task == "rj45" and args.rj45_nominal_evidence is None:
    parser.error("--rj45-nominal-evidence is required for --task rj45")
try:
    task_path = PROJECT_ROOT / (
        "configs/tasks/rj45-v1.json" if args.task == "rj45" else "configs/task.json"
    )
    nominal_trace_paths: tuple[Path, ...] = ()
    if args.task == "rj45":
        nominal_report = load_finite_json(args.rj45_nominal_evidence)
        nominal_trace = nominal_report.get("trace_path")
        if not isinstance(nominal_trace, str) or not nominal_trace:
            raise ValueError("RJ45 nominal evidence report must identify its trace_path")
        nominal_trace_path = Path(nominal_trace)
        if not nominal_trace_path.is_absolute():
            nominal_trace_path = PROJECT_ROOT / nominal_trace_path
        nominal_trace_paths = (args.rj45_nominal_evidence, nominal_trace_path)
    input_snapshot = capture_run_inputs(
        PROJECT_ROOT,
        task_path=task_path,
        build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
        config_paths=(args.config, *sorted((PROJECT_ROOT / "configs").glob("*.json"))),
        source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
        source_files=(Path(__file__).resolve(),),
        extra_paths=nominal_trace_paths,
    )
    _bootstrap_input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    if args.task == "rj45":
        captured_nominal_report = input_snapshot.json_for(args.rj45_nominal_evidence)
        if captured_nominal_report.get("trace_path") != nominal_trace:
            raise ValueError("RJ45 nominal evidence changed while its trace was discovered")
        captured_trace_path = Path(captured_nominal_report["trace_path"])
        if not captured_trace_path.is_absolute():
            captured_trace_path = PROJECT_ROOT / captured_trace_path
        if captured_trace_path.resolve() != nominal_trace_path.resolve():
            raise ValueError("RJ45 nominal evidence trace path changed during startup")
    runner_cfg = input_snapshot.json_for(args.config)
    if args.max_iterations is not None:
        runner_cfg["max_iterations"] = args.max_iterations
    validate_ppo_config(runner_cfg)
    runtime_versions = {
        name: metadata.version(name)
        for name in (
            "isaacsim",
            "isaaclab",
            "isaaclab-rl",
            "rsl-rl-lib",
            "torch",
            "numpy",
            "h5py",
        )
    }
    verify_runtime_versions(
        runtime_versions, input_snapshot.json_for(PROJECT_ROOT / "configs/runtime.json")
    )
    selected_task = (
        load_rj45_task(task_path)
        if args.task == "rj45"
        else input_snapshot.json_for(task_path)
    )
    nominal_evidence = (
        verify_rj45_nominal_evidence(
            PROJECT_ROOT, selected_task, args.rj45_nominal_evidence
        )
        if args.task == "rj45"
        else None
    )
except (OSError, ValueError, TypeError, KeyError, metadata.PackageNotFoundError) as error:
    parser.error(f"invalid training configuration: {error}")
launcher = AppLauncher(args)
simulation_app = launcher.app

from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import omni.kit.app  # noqa: E402
from rby1_assembly.isaac_env import Rby1AssemblyEnv, Rby1AssemblyEnvCfg  # noqa: E402


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = runner_cfg["seed"]
    env_type = Rby1AssemblyEnv
    if args.task == "rj45":
        from rby1_assembly.rj45_env import configure_rj45_task

        env_type = configure_rj45_task(cfg, selected_task, PROJECT_ROOT)
    _bootstrap_input_snapshot.assert_unchanged()
    input_snapshot.assert_unchanged()
    _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
    env = env_type(cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=runner_cfg.get("clip_actions", 1.0))
    selected_log_root = (
        args.log_root / "rj45-v1" if args.task == "rj45" else args.log_root
    )
    log_dir = selected_log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir.mkdir(parents=True, exist_ok=False)
    atomic_write_json(log_dir / "runner_config.json", runner_cfg)
    try:
        source_paths = sorted((PROJECT_ROOT / "src/rby1_assembly").glob("*.py"))
        evidence_paths = source_paths + [Path(__file__).resolve(), args.config.resolve()]
        evidence_paths += sorted((PROJECT_ROOT / "configs").glob("*.json"))
        generated_dir = (
            (PROJECT_ROOT / selected_task["assets"]["build_report"]).parent
            if args.task == "rj45"
            else PROJECT_ROOT / "assets/generated"
        )
        evidence_paths += sorted(generated_dir.glob("*.usda"))
        evidence_paths += [
            generated_dir / "build-report.json",
            task_path,
            log_dir / "runner_config.json",
        ]
        lineage = {
            "schema_version": 1,
            "source_and_config_sha256": {
                str(path.resolve().relative_to(PROJECT_ROOT))
                if path.resolve().is_relative_to(PROJECT_ROOT)
                else str(path.resolve()): (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    if path == log_dir / "runner_config.json"
                    else input_snapshot.hash_for(path)
                )
                for path in evidence_paths
            },
            "runtime_versions": runtime_versions,
            "seed": cfg.seed,
            "num_envs": cfg.scene.num_envs,
            "task_id": "rj45-v1" if args.task == "rj45" else "toy",
            "collision_monitor_mode": "passive_nominal_one_environment" if args.task == "rj45" else "disabled",
            "task_config_path": str(task_path.relative_to(PROJECT_ROOT)),
            "policy_config_path": (
                str(args.config.resolve().relative_to(PROJECT_ROOT))
                if args.config.resolve().is_relative_to(PROJECT_ROOT)
                else str(args.config.resolve())
            ),
            "task_config_sha256": input_snapshot.hash_for(task_path),
            "physics_dt_s": cfg.sim.dt,
            "control_decimation": cfg.decimation,
            "control_dt_s": cfg.sim.dt * cfg.decimation,
            "observation_noise": {
                "distribution": "independent_zero_mean_gaussian_position_and_rotation_vector",
                "position_std_m": cfg.observation_position_noise_std_m,
                "rotation_vector_std_rad": cfg.observation_rotation_noise_std_rad,
            },
            "baseline_controller_state_source": "simulator_ground_truth",
            "checkpoint_scope": "checkpoints created in this run directory",
            "rj45_nominal_evidence": nominal_evidence,
        }
        provenance_path = log_dir / "training_provenance.json"
        atomic_write_json(provenance_path, lineage)
        runner = OnPolicyRunner(wrapped, runner_cfg, log_dir=str(log_dir), device=env.device)
        runner.add_git_repo_to_log(__file__)
        runner.learn(
            num_learning_iterations=runner_cfg["max_iterations"],
            init_at_random_ep_len=False,
        )
        if load_finite_json(provenance_path) != lineage:
            raise RuntimeError("training provenance changed during the experiment")
        for recorded_path, expected_sha256 in lineage["source_and_config_sha256"].items():
            evidence_path = Path(recorded_path)
            if not evidence_path.is_absolute():
                evidence_path = PROJECT_ROOT / evidence_path
            if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != expected_sha256:
                raise RuntimeError(
                    f"training evidence changed during the experiment: {recorded_path}"
                )
        if args.task == "rj45":
            final_nominal_evidence = verify_rj45_nominal_evidence(
                PROJECT_ROOT, selected_task, args.rj45_nominal_evidence
            )
            if final_nominal_evidence != nominal_evidence:
                raise RuntimeError("RJ45 nominal evidence changed during the experiment")
        _bootstrap_input_snapshot.assert_unchanged()
        input_snapshot.assert_unchanged()
        _bootstrap_input_snapshot.assert_preserved_in(input_snapshot)
        checkpoint_manifest = {
            "schema_version": 1,
            "training_completed": True,
            "training_provenance_sha256": hashlib.sha256(
                (log_dir / "training_provenance.json").read_bytes()
            ).hexdigest(),
            "checkpoints": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(log_dir.glob("model_*.pt"))
            },
        }
        if not checkpoint_manifest["checkpoints"]:
            raise RuntimeError("training finished without a model checkpoint")
        checkpoint_manifest_path = log_dir / "checkpoint_manifest.json"
        atomic_write_json(checkpoint_manifest_path, checkpoint_manifest)
        try:
            _bootstrap_input_snapshot.assert_unchanged()
            input_snapshot.assert_unchanged()
        except InputMutationError as error:
            atomic_write_json(
                checkpoint_manifest_path,
                {
                    "schema_version": 1,
                    "training_completed": False,
                    "input_immutability": error.report(),
                },
            )
            raise
        return 0
    finally:
        wrapped.close()


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        exit_code = 1
    finally:
        try:
            omni.kit.app.get_app().post_quit(exit_code)
        except Exception:
            print("Failed to set Isaac Kit exit code", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
        try:
            simulation_app.close()
        except Exception:
            print("Failed to close Isaac application", file=sys.stderr, flush=True)
            traceback.print_exc()
            exit_code = 1
    raise SystemExit(exit_code)
