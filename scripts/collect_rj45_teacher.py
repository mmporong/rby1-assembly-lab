#!/usr/bin/env python3
"""Collect verified train/dev RJ45 teacher demonstrations in Isaac Lab."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rby1_assembly.held_grasp import (  # noqa: E402
    bank_source_contract_sha256,
    held_grasp_bank_dependency_paths,
)
from rby1_assembly.run_provenance import capture_run_inputs  # noqa: E402


def _project_path(value: Path) -> Path:
    return (value if value.is_absolute() else PROJECT_ROOT / value).resolve()


def _json_dependencies(path: Path, keys: tuple[str, ...]) -> tuple[Path, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON input must be an object: {path}")
    result = [path]
    for key in keys:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"JSON input {path} must identify {key}")
        result.append(_project_path(Path(value)))
    return tuple(result)


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--summary", type=Path)
parser.add_argument("--split", choices=("train", "dev"), required=True)
parser.add_argument("--rj45-nominal-evidence", type=Path, required=True)
parser.add_argument("--rj45-held-bank", type=Path, required=True)
parser.add_argument("--rj45-training-config", type=Path, required=True)
parser.add_argument("--parallel-contact-evidence", type=Path)
parser.add_argument("--task-config", type=Path, default=Path("configs/tasks/rj45-v1.json"))
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes", type=int, required=True)
parser.add_argument("--seed", type=int, default=20261120)
bootstrap_args, _unknown = parser.parse_known_args()
if bootstrap_args.num_envs <= 0 or bootstrap_args.episodes <= 0 or bootstrap_args.seed < 0:
    parser.error("num-envs/episodes must be positive and seed must be non-negative")
if bootstrap_args.seed == 20261027:
    parser.error("seed 20261027 is reserved for sealed final evaluation and cannot collect data")
if bootstrap_args.num_envs > 1 and bootstrap_args.parallel_contact_evidence is None:
    parser.error("parallel teacher collection requires --parallel-contact-evidence")

task_path = _project_path(bootstrap_args.task_config)
training_path = _project_path(bootstrap_args.rj45_training_config)
bank_path = _project_path(bootstrap_args.rj45_held_bank)
nominal_path = _project_path(bootstrap_args.rj45_nominal_evidence)
nominal_payload = json.loads(nominal_path.read_text(encoding="utf-8"))
nominal_trace = nominal_payload.get("trace_path") if isinstance(nominal_payload, dict) else None
if not isinstance(nominal_trace, str) or not nominal_trace:
    parser.error("nominal evidence must identify trace_path")
nominal_trace_path = _project_path(Path(nominal_trace))
bank_dependencies = held_grasp_bank_dependency_paths(
    PROJECT_ROOT, bank_path, bootstrap_args.split
)
contact_spec = None
contact_dependencies: tuple[Path, ...] = ()
if bootstrap_args.parallel_contact_evidence is not None:
    contact_path = _project_path(bootstrap_args.parallel_contact_evidence)
    contact_dependencies = _json_dependencies(
        contact_path, ("report_path", "trace_path", "sat_report_path")
    )
    contact_spec = json.loads(contact_path.read_text(encoding="utf-8"))

input_snapshot = capture_run_inputs(
    PROJECT_ROOT,
    task_path=task_path,
    build_report_fallback=PROJECT_ROOT / "assets/generated/build-report.json",
    config_paths=(
        training_path,
        PROJECT_ROOT / "configs/assets.json",
        PROJECT_ROOT / "configs/runtime.json",
        PROJECT_ROOT / "configs/rj45-assets.json",
    ),
    source_roots=((PROJECT_ROOT / "src/rby1_assembly", False),),
    source_files=(Path(__file__).resolve(),),
    extra_paths=(nominal_path, nominal_trace_path, *bank_dependencies, *contact_dependencies),
)

from isaaclab.app import AppLauncher  # noqa: E402

AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, multi_gpu=False)
simulation_app = launcher.app

import torch  # noqa: E402

from rby1_assembly.experiment_contracts import (  # noqa: E402
    apply_runtime_timing,
    verify_rj45_nominal_evidence,
)
from rby1_assembly.held_grasp import load_verified_held_grasp_bank  # noqa: E402
from rby1_assembly.isaac_env import AssemblyPhase, Rby1AssemblyEnvCfg  # noqa: E402
from rby1_assembly.parallel_contact_evidence import (  # noqa: E402
    load_parallel_contact_evidence,
)
from rby1_assembly.rj45_env import configure_rj45_task, load_rj45_task  # noqa: E402
from rby1_assembly.rj45_task import verify_runtime_versions  # noqa: E402
from rby1_assembly.rj45_teacher_collection import (  # noqa: E402
    ParallelTeacherCollector,
    write_collected_episodes,
)
from rby1_assembly.rj45_teacher_runtime import make_rj45_teacher_env_type  # noqa: E402
from rby1_assembly.rj45_training import load_training_config  # noqa: E402
from rby1_assembly.rj45_training_env import (  # noqa: E402
    _calibration_bias_case_seed,
    make_rj45_training_env_type,
)


def main() -> int:
    summary_path = _project_path(args.summary) if args.summary else _project_path(args.output).with_suffix(".summary.json")
    output_path = _project_path(args.output)
    collector = None
    env = None
    try:
        task = load_rj45_task(task_path)
        training = load_training_config(training_path)
        if training["diagnostic_observation_mode"] != "standard":
            raise ValueError("teacher collection requires the standard 79D observation")
        nominal_proof = verify_rj45_nominal_evidence(PROJECT_ROOT, task, nominal_path)
        bank, bank_proof = load_verified_held_grasp_bank(
            PROJECT_ROOT, task, bank_path, args.split
        )
        parallel_proof = (
            load_parallel_contact_evidence(contact_spec, project_root=PROJECT_ROOT)
            if contact_spec is not None
            else None
        )
        verify_runtime_versions(
            {
                name: metadata.version(name)
                for name in ("isaacsim", "isaaclab", "isaaclab-rl", "torch", "numpy", "h5py")
            },
            input_snapshot.json_for(PROJECT_ROOT / "configs/runtime.json"),
        )
        cfg = Rby1AssemblyEnvCfg()
        apply_runtime_timing(cfg, input_snapshot.json_for(PROJECT_ROOT / "configs/runtime.json"))
        cfg.scene.num_envs = args.num_envs
        cfg.sim.device = args.device
        cfg.seed = args.seed
        if contact_spec is not None:
            if parallel_proof is None:
                raise ValueError("parallel contact evidence was not verified")
            cfg.rj45_parallel_contact_evidence = contact_spec
        env_type = configure_rj45_task(cfg, task, PROJECT_ROOT)
        env_type = make_rj45_training_env_type(
            env_type,
            training,
            held_bank=bank,
            held_bank_proof=bank_proof,
            calibration_bias_enabled=True,
            bias_level=training["calibration_bias"]["initial_level"],
        )
        env_type = make_rj45_teacher_env_type(env_type)
        input_snapshot.assert_unchanged()
        env = env_type(cfg)
        collector = ParallelTeacherCollector(
            num_envs=args.num_envs,
            total_episodes=args.episodes,
            grasp_ids=tuple(bank_proof["selected_grasp_ids"]),
            episode_seed=lambda env_id, episode_index: _calibration_bias_case_seed(
                args.seed, env_id, episode_index
            ),
        )
        env.reset()
        step_index = 0
        while not collector.complete:
            pre_step_time_s = step_index * float(env.step_dt)
            actions = torch.zeros((args.num_envs, 6), device=env.device)
            _observations, _reward, terminated, truncated, extras = env.step(actions)
            ended = terminated | truncated
            live = env._training_step_snapshot()
            live.update(
                {
                    "phase": env._phase.clone(),
                    "phase_name_by_code": {
                        int(phase): phase.name.lower() for phase in AssemblyPhase
                    },
                    "episode_index": env._training_episode_index.clone(),
                    "held_grasp_index": env._training_grasp_index.clone(),
                }
            )
            collector.consume_step(
                live_snapshot=live,
                terminal_snapshot=extras.get("terminal_snapshot"),
                ended=ended,
                pre_step_time_s=pre_step_time_s,
            )
            step_index += 1

        input_snapshot.assert_unchanged()
        if verify_rj45_nominal_evidence(PROJECT_ROOT, task, nominal_path) != nominal_proof:
            raise RuntimeError("nominal evidence changed during teacher collection")
        _, final_bank_proof = load_verified_held_grasp_bank(
            PROJECT_ROOT, task, bank_path, args.split
        )
        if final_bank_proof != bank_proof:
            raise RuntimeError("held bank changed during teacher collection")
        if contact_spec is not None:
            final_parallel = load_parallel_contact_evidence(contact_spec, project_root=PROJECT_ROOT)
            if final_parallel != parallel_proof:
                raise RuntimeError("parallel contact evidence changed during teacher collection")
        write_collected_episodes(
            output_path,
            collector.episodes,
            split=args.split,
            source_sha256=bank_source_contract_sha256(bank_proof),
            task_sha256=task["_config_sha256"],
            training_sha256=training["_config_sha256"],
            bank_sha256=bank_proof["manifest_sha256"],
            translation_bound_m=float(cfg.residual_translation_bound_m),
            rotation_bound_rad=float(cfg.residual_rotation_bound_rad),
        )
        _atomic_json(
            summary_path,
            {
                "status": "complete",
                "dataset": str(output_path),
                "split": args.split,
                "seed": args.seed,
                "episodes": len(collector.episodes),
                "accepted_per_environment": list(collector.accepted_per_environment),
                "independent_successes": sum(
                    episode.independent_success for episode in collector.episodes
                ),
                "terminal_valid": sum(episode.terminal_valid for episode in collector.episodes),
                "inputs_reverified_at_end": True,
            },
        )
        return 0
    except Exception as error:
        _atomic_json(
            summary_path,
            {
                "status": "incomplete",
                "dataset_published": False,
                "seed": args.seed,
                "completed_attempts": 0 if collector is None else len(collector.episodes),
                "accepted_per_environment": (
                    [] if collector is None else list(collector.accepted_per_environment)
                ),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        simulation_app.close()
