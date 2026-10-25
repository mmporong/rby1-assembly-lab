"""Run a real-PhysX negative scene against the forbidden-contact detector.

The generated table is translated so its top overlaps the robot's initial
forward finger workspace.  No sensor values or forces are injected: the probe
passes only when PhysX reports filtered pair contacts through the environment's
normal pre-autoreset terminal snapshot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="RB-Y1 forbidden-contact PhysX probe")
parser.add_argument("--steps", type=int, default=90)
parser.add_argument(
    "--report",
    type=Path,
    default=Path("artifacts/cloud/forbidden-contact-probe.json"),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= 0:
    parser.error("--steps must be positive")

launcher = AppLauncher(args)
simulation_app = launcher.app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import omni.kit.app  # noqa: E402
import torch  # noqa: E402

from rby1_assembly.contracts import ACTION_DIM  # noqa: E402
from rby1_assembly.isaac_env import (  # noqa: E402
    AssemblyPhase,
    FORBIDDEN_CONTACT_LOAD_SOURCE,
    Rby1AssemblyEnv,
    Rby1AssemblyEnvCfg,
    TASK_CONFIG,
    TerminationReason,
)


TABLE_TRANSLATION_M = (-0.30, 0.0, -0.14)


def _write_report(report: dict) -> None:
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.report)


def _failure_report(error: BaseException) -> dict:
    return {
        "passed": False,
        "probe": "forbidden_contact_negative_scene",
        "physics_backend": "PhysX via Isaac Sim",
        "force_injection": False,
        "table_translation_world_m": list(TABLE_TRANSLATION_M),
        "error": {"type": type(error).__name__, "message": str(error)},
    }


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu().reshape(-1)[0].item())


def _boolean(value: torch.Tensor) -> bool:
    return bool(value.detach().cpu().reshape(-1)[0].item())


def main() -> int:
    cfg = Rby1AssemblyEnvCfg()
    cfg.seed = int(TASK_CONFIG["data"]["evaluation_seed"])
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.table.init_state.pos = TABLE_TRANSLATION_M
    env = Rby1AssemblyEnv(cfg)
    terminal_snapshot = None
    terminal_evaluation = None
    terminal_step = None
    active_exception = False

    try:
        env.reset()
        for control_step in range(1, args.steps + 1):
            actions = torch.zeros((1, ACTION_DIM), device=env.device)
            _, _, terminated, truncated, extras = env.step(actions)
            snapshot = extras["terminal_snapshot"]
            if _boolean(snapshot["terminal_mask"]):
                terminal_snapshot = snapshot
                terminal_evaluation = extras["terminal_evaluation"]
                terminal_step = control_step
                if not _boolean(terminated | truncated):
                    raise AssertionError("terminal snapshot was set without a terminal return")
                break

        if terminal_snapshot is None or terminal_evaluation is None or terminal_step is None:
            raise AssertionError(f"no forbidden-contact termination within {args.steps} steps")

        source = terminal_snapshot["forbidden_contact_load_source"]
        load_n = _scalar(terminal_snapshot["forbidden_contact_load_n"])
        peak_n = _scalar(terminal_snapshot["forbidden_contact_load_peak_n"])
        threshold_n = float(cfg.forbidden_contact_force_n)
        forbidden_seen = _boolean(terminal_snapshot["forbidden_contact"])
        measured = _boolean(terminal_snapshot["forbidden_contact_measured"])
        phase_code = int(_scalar(terminal_snapshot["phase"]))
        reason_code = int(_scalar(terminal_snapshot["termination_reason_code"]))
        reason_name = terminal_snapshot["termination_reason_by_code"][reason_code]
        evaluator_reason = terminal_evaluation["termination_reason"][0]
        evaluator_valid = _boolean(terminal_evaluation["valid"])
        evaluator_criterion_ok = _boolean(
            terminal_evaluation["criteria"]["forbidden_contact"]
        )

        checks = {
            "source_is_exact_filtered_pair_load": source == FORBIDDEN_CONTACT_LOAD_SOURCE,
            "load_is_finite": bool(
                torch.isfinite(terminal_snapshot["forbidden_contact_load_n"]).all()
            ),
            "load_exceeds_threshold": load_n > threshold_n,
            "peak_covers_terminal_load": peak_n >= load_n,
            "forbidden_contact_seen": forbidden_seen,
            "forbidden_contact_measured": measured,
            "independent_evaluator_is_valid": evaluator_valid,
            "independent_evaluator_source_is_ground_truth": (
                terminal_evaluation["evaluator_source"]
                == "simulator_ground_truth_independent"
            ),
            "terminal_phase_is_abort": phase_code == int(AssemblyPhase.ABORT),
            "termination_reason_is_forbidden_contact": (
                reason_code == int(TerminationReason.FORBIDDEN_CONTACT)
                and reason_name == "forbidden_contact"
                and evaluator_reason == "forbidden_contact"
            ),
            "evaluator_rejects_forbidden_contact": not evaluator_criterion_ok,
        }
        table_size = TASK_CONFIG["table"]["size_m"]
        original_center = [
            *TASK_CONFIG["table"]["center_xy_m"],
            TASK_CONFIG["table"]["top_height_m"] - table_size[2] / 2.0,
        ]
        translated_center = [
            original_center[index] + TABLE_TRANSLATION_M[index] for index in range(3)
        ]
        report = {
            "passed": all(checks.values()),
            "probe": "forbidden_contact_negative_scene",
            "physics_backend": "PhysX via Isaac Sim",
            "force_injection": False,
            "control_action": "zero_residual_action",
            "terminal_state_source": "pre_autoreset_terminal_snapshot",
            "table_translation_world_m": list(TABLE_TRANSLATION_M),
            "table_top_size_m": table_size,
            "table_top_center_after_translation_world_m": translated_center,
            "table_top_surface_after_translation_world_z_m": (
                TASK_CONFIG["table"]["top_height_m"] + TABLE_TRANSLATION_M[2]
            ),
            "scene_rationale": (
                "move the 0.65 x 0.90 x 0.05 m table top into the initialized "
                "robot's forward finger workspace without changing detector thresholds"
            ),
            "steps_requested": args.steps,
            "terminal_control_step": terminal_step,
            "forbidden_contact_load_n": load_n,
            "forbidden_contact_load_peak_n": peak_n,
            "forbidden_contact_threshold_n": threshold_n,
            "forbidden_contact_load_source": source,
            "phase_code": phase_code,
            "phase_name": terminal_snapshot["phase_name_by_code"][phase_code],
            "termination_reason_code": reason_code,
            "termination_reason": reason_name,
            "checks": checks,
            "failed_checks": [name for name, passed in checks.items() if not passed],
        }
        _write_report(report)
        if not report["passed"]:
            print(json.dumps(report, indent=2), file=sys.stderr, flush=True)
            return 2
        print(json.dumps(report, indent=2), flush=True)
        return 0
    except BaseException:
        active_exception = True
        raise
    finally:
        try:
            env.close()
        except Exception:
            print("Failed to close environment", file=sys.stderr, flush=True)
            traceback.print_exc()
            if not active_exception:
                raise


if __name__ == "__main__":
    exit_code = 1
    try:
        exit_code = main()
    except Exception as error:
        traceback.print_exc()
        sys.stderr.flush()
        try:
            _write_report(_failure_report(error))
        except Exception:
            print("Failed to write probe failure report", file=sys.stderr, flush=True)
            traceback.print_exc()
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
