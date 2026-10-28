from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import h5py
import numpy as np
import torch

from rby1_assembly.rj45_bc import build_bc_actor, load_bc_actor_artifact

ZERO_COMMIT = "104721acf2222c1cfe8ec86cd030f5116a4fef93"
BC_COMMIT = "8512a5932547b5495325b5ff0b6894fb072db98e"
EXPECTED_INPUT_SHA256 = {
    "train_h5": "ae2884f3635d6177b278156f8cfadf1d3d29db54115cc99a938927ea3d6b1efc",
    "dev_h5": "680d18be628c2eb9bb6adecf0ff719f48d99b6128ee80ca58af5485235f60830",
    "actor": "f0fd65b56691748dce5bee378152524bf84f3f9daed2f1f36dd43e0e189233b5",
    "training_report": "32aa18808dfb9ae2ccc6bb4019f24e9c5782005ee69c99889af7ee2816467bbf",
    "bc_eval": "5c61a1ca92cda3f473f61888d70f308eb6a920f3d26acae47d4413cfbc90413f",
    "zero_eval": "ea6c92c0cd2ac70f04429842244ffadd65a1bf12af455c965efbeaa2c3b3aea5",
}
CODE_PATHS = (
    "src/rby1_assembly/rj45_bc.py",
    "src/rby1_assembly/rj45_bc_training.py",
    "src/rby1_assembly/rj45_teacher.py",
    "src/rby1_assembly/rj45_teacher_runtime.py",
    "src/rby1_assembly/rj45_teacher_dataset.py",
    "src/rby1_assembly/rj45_training.py",
    "src/rby1_assembly/rj45_training_env.py",
    "scripts/evaluate_policy.py",
    "scripts/train.py",
    "scripts/analyze_rj45_bc_closed_loop.py",
)
LAYOUT = {
    "relative_position": (0, 3),
    "relative_quaternion": (3, 7),
    "arm_joint_position": (7, 21),
    "arm_joint_velocity": (21, 35),
    "grip_force": (35, 37),
    "entry_force": (37, 40),
    "grip_proven": (40, 42),
    "phase_one_hot": (42, 50),
    "previous_action": (50, 62),
    "command_depth": (62, 63),
    "insertion_advance": (63, 64),
    "joint_reference_error": (64, 78),
    "policy_active": (78, 79),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quantiles(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        str(value): float(np.quantile(array, value))
        for value in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def compact_quantiles(values: Any) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        str(value): float(np.quantile(array, value))
        for value in (0.0, 0.25, 0.5, 0.75, 1.0)
    }


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1)
    return {
        "mean_per_dim": array.mean(0).tolist(),
        "std_per_dim": array.std(0).tolist(),
        "abs_mean_per_dim": np.abs(array).mean(0).tolist(),
        "norm_quantiles": quantiles(norms),
        "max_abs": float(np.abs(array).max()),
        "zero_row_fraction": float(np.all(np.abs(array) < 1.0e-8, axis=1).mean()),
    }


def mean_squared_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean((left - right) ** 2))


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def verify_report_source_map(
    report: dict[str, Any], commit: str, *, repo: Path
) -> dict[str, Any]:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("evaluation report lacks provenance")
    source_map = provenance.get("source_sha256")
    if not isinstance(source_map, dict) or not source_map:
        raise ValueError("evaluation report lacks a source SHA map")
    for relative, expected in source_map.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("evaluation source map is malformed")
        payload = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=repo)
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"evaluation source does not match {commit}: {relative}")
    return {
        "commit": commit,
        "source_file_count": len(source_map),
        "source_sha256": source_map,
        "evaluate_policy_sha256": source_map["scripts/evaluate_policy.py"],
        "git_show_all_sources_verified": True,
    }


def require_h5_group(
    value: h5py.Group | h5py.Dataset | h5py.Datatype, label: str
) -> h5py.Group:
    if not isinstance(value, h5py.Group):
        raise TypeError(f"{label} must be an HDF5 group")
    return value


def require_h5_dataset(
    value: h5py.Group | h5py.Dataset | h5py.Datatype, label: str
) -> h5py.Dataset:
    if not isinstance(value, h5py.Dataset):
        raise TypeError(f"{label} must be an HDF5 dataset")
    return value


def load_split(path: Path) -> dict[str, Any]:
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    previous_actions: list[np.ndarray] = []
    phases: list[str] = []
    lags: list[float] = []
    grasps: list[str] = []
    gt_poses: list[np.ndarray] = []
    temporal_deltas: list[np.ndarray] = []
    episodes: list[dict[str, Any]] = []
    with h5py.File(path, "r") as handle:
        episodes_group = require_h5_group(handle["episodes"], "episodes")
        for episode_id in episodes_group:
            group = require_h5_group(episodes_group[episode_id], f"episode {episode_id}")
            valid_dataset = require_h5_dataset(group["valid"], "valid")
            observation_dataset = require_h5_dataset(group["observation"], "observation")
            action_dataset = require_h5_dataset(group["teacher_applied"], "teacher_applied")
            policy_time_dataset = require_h5_dataset(
                group["policy_observation_timestamp"], "policy_observation_timestamp"
            )
            observation_time_dataset = require_h5_dataset(
                group["observation_timestamp"], "observation_timestamp"
            )
            phase_dataset = require_h5_dataset(group["phase"], "phase")
            valid = np.asarray(valid_dataset[:], dtype=np.bool_)
            success = bool(group.attrs["independent_success"])
            terminal_valid = bool(group.attrs["valid"])
            eligible = valid & success & terminal_valid
            observation = np.asarray(observation_dataset[:], dtype=np.float32)[eligible]
            action = np.asarray(action_dataset[:], dtype=np.float32)[eligible]
            policy_time = np.asarray(policy_time_dataset[:], dtype=np.float64)[eligible]
            observation_time = np.asarray(observation_time_dataset[:], dtype=np.float64)[eligible]
            phase = np.asarray(phase_dataset.asstr()[:])[eligible]
            observations.append(observation)
            actions.append(action)
            previous_actions.append(observation[:, 56:62])
            phases.extend(str(value) for value in phase.tolist())
            lags.extend((observation_time - policy_time).tolist())
            grasps.append(str(group.attrs["grasp_id"]))
            privileged = group.get("privileged")
            if isinstance(privileged, h5py.Group) and "teacher_gt_pose" in privileged:
                teacher_gt_pose = require_h5_dataset(
                    privileged["teacher_gt_pose"], "teacher_gt_pose"
                )
                gt_poses.append(
                    np.asarray(teacher_gt_pose[:], dtype=np.float32)[eligible]
                )
            if len(action) > 1:
                temporal_deltas.append(np.diff(action, axis=0))
            episodes.append(
                {
                    "id": episode_id,
                    "rows": int(len(valid)),
                    "eligible": int(eligible.sum()),
                    "grasp_id": str(group.attrs["grasp_id"]),
                    "success": success,
                    "terminal_valid": terminal_valid,
                }
            )
    return {
        "episodes": episodes,
        "obs": np.concatenate(observations),
        "act": np.concatenate(actions),
        "prev": np.concatenate(previous_actions),
        "phases": phases,
        "lags": np.asarray(lags),
        "grasps": grasps,
        "gtpose": np.concatenate(gt_poses),
        "temporal": np.concatenate(temporal_deltas),
    }


def actor_prediction(actor: torch.nn.Module, observation: np.ndarray) -> np.ndarray:
    with torch.inference_mode():
        return actor(torch.from_numpy(observation)).numpy()


def evaluation_stats(report: dict[str, Any]) -> dict[str, Any]:
    outcomes = report["episode_outcomes"]
    if not isinstance(outcomes, list):
        raise ValueError("episode_outcomes must be a list")
    successes = np.asarray([bool(row["success"]) for row in outcomes])
    valid = np.asarray([bool(row["valid"]) for row in outcomes])
    fields = (
        "final_depth_m",
        "final_lateral_error_m",
        "final_orientation_error_rad",
        "final_dwell_s",
        "grip_contact_force_peak_n",
        "insertion_contact_force_peak_n",
        "insertion_contact_load_peak_n",
        "elapsed_s",
        "policy_active_step_count",
        "action_saturation_rate_active_steps",
    )
    result: dict[str, Any] = {
        "successes": int(successes.sum()),
        "valid": int(valid.sum()),
        "termination_reasons": dict(
            collections.Counter(str(row["termination_reason"]) for row in outcomes)
        ),
        "by_all": {},
        "by_success": {},
        "by_failure": {},
    }
    for field in fields:
        values = np.asarray([float(row[field]) for row in outcomes])
        result["by_all"][field] = quantiles(values)
        result["by_success"][field] = quantiles(values[successes])
        result["by_failure"][field] = quantiles(values[~successes])
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute the reviewed RJ45 BC offline and closed-loop diagnostic."
    )
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--dev-h5", type=Path, required=True)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--bc-eval", type=Path, required=True)
    parser.add_argument("--zero-eval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    started = time.time()
    args = parse_args()
    repo = args.repo.resolve()
    paths = {
        "train_h5": args.train_h5.resolve(),
        "dev_h5": args.dev_h5.resolve(),
        "actor": args.actor.resolve(),
        "training_report": args.training_report.resolve(),
        "bc_eval": args.bc_eval.resolve(),
        "zero_eval": args.zero_eval.resolve(),
    }
    output = args.output.resolve()
    code_files = [repo / relative for relative in CODE_PATHS]
    actual_hashes = {name: sha256(path) for name, path in paths.items()}
    if actual_hashes != EXPECTED_INPUT_SHA256:
        raise ValueError("diagnostic input SHA-256 differs from the reviewed six-file set")

    bc_report = load_json(paths["bc_eval"])
    zero_report = load_json(paths["zero_eval"])
    zero_provenance = verify_report_source_map(zero_report, ZERO_COMMIT, repo=repo)
    bc_provenance = verify_report_source_map(bc_report, BC_COMMIT, repo=repo)
    zero_sources = zero_provenance["source_sha256"]
    bc_sources = bc_provenance["source_sha256"]
    differing_runtime_sources = sorted(
        key for key in zero_sources if zero_sources[key] != bc_sources.get(key)
    )
    if differing_runtime_sources != ["scripts/evaluate_policy.py"]:
        raise ValueError("actual evaluation runtime sources differ beyond the BC input fix")
    commit_diff = subprocess.check_output(
        ["git", "diff", "--name-only", f"{ZERO_COMMIT}..{BC_COMMIT}"],
        cwd=repo,
        text=True,
    ).splitlines()
    if commit_diff != ["scripts/evaluate_policy.py", "tests/test_rj45_bank_eval_contract.py"]:
        raise ValueError("evaluation fix commit differs beyond its reviewed two files")

    train = load_split(paths["train_h5"])
    dev = load_split(paths["dev_h5"])
    artifact = load_bc_actor_artifact(
        paths["actor"], expected_sha256=EXPECTED_INPUT_SHA256["actor"]
    )
    actor = build_bc_actor()
    actor.load_state_dict(artifact["actor_state_dict"])
    actor.eval()
    train_prediction = actor_prediction(actor, train["obs"])
    dev_prediction = actor_prediction(actor, dev["obs"])

    observation_shift: dict[str, Any] = {}
    for name, (start, end) in LAYOUT.items():
        delta = dev["obs"][:, start:end].mean(0) - train["obs"][:, start:end].mean(0)
        scale = np.maximum(train["obs"][:, start:end].std(0), 1.0e-3)
        observation_shift[name] = {
            "raw_mean_shift_rms": float(np.sqrt(np.mean(delta**2))),
            "standardized_mean_shift_rms": float(np.sqrt(np.mean((delta / scale) ** 2))),
            "max_standardized_dim": float(np.max(np.abs(delta / scale))),
        }

    bc_by_id = {row["paired_condition_id"]: row for row in bc_report["episode_outcomes"]}
    zero_by_id = {
        row["paired_condition_id"]: row for row in zero_report["episode_outcomes"]
    }
    paired_ids = sorted(set(bc_by_id) & set(zero_by_id))
    transitions = collections.Counter(
        (bool(zero_by_id[key]["success"]), bool(bc_by_id[key]["success"]))
        for key in paired_ids
    )
    paired_delta_fields = (
        "final_depth_m",
        "final_lateral_error_m",
        "final_orientation_error_rad",
        "final_dwell_s",
        "grip_contact_force_peak_n",
        "insertion_contact_load_peak_n",
        "elapsed_s",
    )
    paired_deltas = {
        field: quantiles(
            [float(bc_by_id[key][field]) - float(zero_by_id[key][field]) for key in paired_ids]
        )
        for field in paired_delta_fields
    }
    transition_bias: dict[str, Any] = {}
    for zero_success, bc_success in ((0, 0), (0, 1), (1, 0), (1, 1)):
        rows = [
            bc_by_id[key]
            for key in paired_ids
            if (
                int(bool(zero_by_id[key]["success"])),
                int(bool(bc_by_id[key]["success"])),
            )
            == (zero_success, bc_success)
        ]
        transition_bias[f"{zero_success}->{bc_success}"] = {
            "count": len(rows),
            "translation_bias_norm_um": compact_quantiles(
                [
                    np.linalg.norm(row["calibration_bias_translation_entry_m"]) * 1.0e6
                    for row in rows
                ]
            ),
            "rotation_bias_norm_mrad": compact_quantiles(
                [
                    np.linalg.norm(row["calibration_bias_rotation_vector_entry_rad"]) * 1.0e3
                    for row in rows
                ]
            ),
        }

    training_report = load_json(paths["training_report"])
    dev_mse = mean_squared_error(dev_prediction, dev["act"])
    zero_mse = mean_squared_error(np.zeros_like(dev["act"]), dev["act"])
    previous_mse = mean_squared_error(dev["prev"], dev["act"])
    dev_error_stats = vector_stats(dev_prediction - dev["act"])
    absolute_error = dev_error_stats["abs_mean_per_dim"]
    result: dict[str, Any] = {
        "schema_version": 2,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "execution_seconds": None,
        "evidence_sha256": actual_hashes,
        "actual_evaluation_provenance": {
            "biased_zero": zero_provenance,
            "bc_only": bc_provenance,
            "runtime_source_differences": differing_runtime_sources,
            "commit_diff": commit_diff,
            "same_physics_and_src_sources": all(
                zero_sources[key] == bc_sources[key]
                for key in zero_sources
                if key.startswith("src/")
            ),
        },
        "analysis_environment": {
            "repo_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip(),
            "current_code_sha256": {
                str(path.relative_to(repo)): sha256(path) for path in code_files
            },
            "note": (
                "Current analysis code is recorded separately and is not attributed to either "
                "actual evaluation run."
            ),
        },
        "contracts": {
            "observation": (
                "standard79_without_additional_normalization; identical "
                "build_policy_observation in collection/evaluation"
            ),
            "action": (
                "teacher_applied bounded dimensionless 6D; absolute tip target; "
                "translation scale 0.0015 m and rotation scale 0.02617993878 rad"
            ),
            "causal_alignment": (
                "teacher caches last policy observation before pre_physics_step, then labels "
                "action applied in that step; dataset timestamps preserve this pairing"
            ),
            "previous_action_indices_right_6d": [56, 62],
        },
        "dataset": {
            "train": {
                "episodes": len(train["episodes"]),
                "eligible_rows": len(train["obs"]),
                "grasp_ids": sorted(set(train["grasps"])),
                "phase_counts": dict(collections.Counter(train["phases"])),
                "timestamp_lag_s_quantiles": quantiles(train["lags"]),
                "label": vector_stats(train["act"]),
                "previous_action_to_label_mse": mean_squared_error(
                    train["prev"], train["act"]
                ),
                "label_temporal_delta": vector_stats(train["temporal"]),
            },
            "dev": {
                "episodes": len(dev["episodes"]),
                "eligible_rows": len(dev["obs"]),
                "grasp_ids": sorted(set(dev["grasps"])),
                "phase_counts": dict(collections.Counter(dev["phases"])),
                "timestamp_lag_s_quantiles": quantiles(dev["lags"]),
                "label": vector_stats(dev["act"]),
                "previous_action_to_label_mse": previous_mse,
                "label_temporal_delta": vector_stats(dev["temporal"]),
            },
            "train_dev_observation_shift_by_field": observation_shift,
            "privileged_teacher_gt_pose_shape_train": list(train["gtpose"].shape),
            "privileged_teacher_gt_pose_shape_dev": list(dev["gtpose"].shape),
        },
        "actor": {
            "artifact_actor_state_sha256": artifact["actor_state_sha256"],
            "training_report_best_epoch": training_report["best_epoch"],
            "training_report_best_dev_mse": training_report["best_dev_mse"],
            "training_report_zero_actor_dev_mse": training_report["initial_dev_mse"],
            "observability_diagnostic": training_report["observability_diagnostic"],
            "train_mse_recomputed": mean_squared_error(train_prediction, train["act"]),
            "dev_mse_recomputed": dev_mse,
            "dev_zero_mse_recomputed": zero_mse,
            "dev_previous_action_mse": previous_mse,
            "train_prediction": vector_stats(train_prediction),
            "dev_prediction": vector_stats(dev_prediction),
            "dev_prediction_error": dev_error_stats,
            "dev_label_prediction_corr_per_dim": [
                float(np.corrcoef(dev_prediction[:, index], dev["act"][:, index])[0, 1])
                for index in range(6)
            ],
            "dev_improvement_vs_zero_fraction": 1.0 - dev_mse / zero_mse,
            "dev_improvement_vs_previous_action_fraction": 1.0 - dev_mse / previous_mse,
            "dev_mean_absolute_physical_error": {
                "translation_um": [float(value) * 1500.0 for value in absolute_error[:3]],
                "rotation_mrad": [
                    float(value) * 0.026179938779914945 * 1000.0
                    for value in absolute_error[3:]
                ],
            },
            "closed_loop_shortcut_evidence": {
                "teacher_label_changes_slowly_per_step": True,
                "previous_action_is_student_observation_indices_56_to_61": True,
                "dev_previous_action_mse": previous_mse,
                "dev_actor_mse": dev_mse,
                "relative_mse_improvement_over_previous_action": 1.0 - dev_mse / previous_mse,
            },
        },
        "closed_loop": {
            "bc": evaluation_stats(bc_report),
            "biased_zero": evaluation_stats(zero_report),
            "paired_case_count": len(paired_ids),
            "paired_success_transition_zero_to_bc": {
                f"{int(left)}->{int(right)}": count
                for (left, right), count in sorted(transitions.items())
            },
            "paired_bc_minus_zero_quantiles": paired_deltas,
            "paired_transition_bias": transition_bias,
            "success_thresholds": {
                "assembled_depth_m": 0.012,
                "max_lateral_error_m": 5.0e-5,
                "max_orientation_error_rad": 0.008726646259971648,
                "dwell_s": 0.4,
                "max_contact_force_n": 25.0,
            },
        },
        "findings": {
            "confirmed_contract_bug": False,
            "confirmed_contract_checks": [
                "79D ordering is produced by the same build_policy_observation path",
                "no extra normalization in BC train or eval",
                "6D labels and actor outputs are dimensionless bounded actions and the environment applies the same physical scale",
                "teacher rows pair cached policy observation t with transmitted action t",
                "valid dataset policy timestamps are nonnegative and causal",
            ],
            "confirmed_risk": (
                "teacher target uses privileged true socket/plug/right-EE geometry and frozen "
                "estimated plug-to-EE transform, while standard79 excludes the sampled "
                "persistent calibration bias and teacher target pose"
            ),
            "interpretation": (
                "The 7.28% improvement over previous-action copying supports prioritizing a "
                "closed-loop distribution-shift test. It does not prove an autoregressive "
                "shortcut, drift, or partial observability."
            ),
            "confirmed_dominant_failure_signature": (
                "56/64 BC episodes time out; failed BC median lateral error is 66.8 um, "
                "above the 50 um gate, while median depth 12.362 mm and orientation "
                "4.449 mrad meet their scalar limits. Dwell median is 0 because all "
                "conditions do not remain jointly true."
            ),
            "confirmed_paired_effect": (
                "BC retains 4 of 38 zero-baseline successes, breaks 34, rescues 4 of 26 "
                "zero-baseline failures, and leaves 22 failed."
            ),
            "unverified_hypotheses": [
                "Self-induced observation shift through the previous-action feature may cause autoregressive rollout drift or a biased fixed point. Episode action time series were not recorded, so this is not proven.",
                "The standard79 observation may be insufficient to infer the privileged correction for unseen calibration-bias cases. The nearest-neighbor label-collision statistic is compatible with this hypothesis but is not an impossibility proof.",
                "Dev held-grasp distribution shift may contribute; only two dev grasps are used versus eight train grasps, but offline dev MSE remains low and does not isolate grasp from hidden-bias shift.",
            ],
        },
    }
    result["execution_seconds"] = time.time() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
