"""Offline diagnostic only: previous-action sensitivity on frozen teacher rows."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--teacher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    helper = repo / "scripts/analyze_rj45_bc_closed_loop.py"
    if digest(helper) != "d090f00eb48afeb79ff3326c2e1f67933c01d976350bc2b9a4aed5e17a56c98a":
        raise ValueError("reviewed diagnostic loader changed")
    actor_source = repo / "src/rby1_assembly/rj45_bc.py"
    if digest(actor_source) != "19cd2fa7df24bf3b7dd7896a468b340aaf49b290e3f28d5c6abeddc436bd39f4":
        raise ValueError("reviewed actor implementation changed")
    sys.path.insert(0, str(repo / "src"))
    spec = importlib.util.spec_from_file_location("bc_diagnostic", helper)
    if spec is None or spec.loader is None:
        raise ValueError("diagnostic loader unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.set_num_threads(4)
    actor_path = args.teacher_root / "bc/bc_actor_init.pt"
    artifact = module.load_bc_actor_artifact(
        actor_path, expected_sha256=module.EXPECTED_INPUT_SHA256["actor"]
    )
    actor = module.build_bc_actor()
    actor.load_state_dict(artifact["actor_state_dict"])
    actor.eval()
    result = {
        "schema_version": 1,
        "scope": "offline teacher-distribution sensitivity; no physical rollout",
        "final_opened": False,
        "selection_use": False,
        "input_sha256": {
            "actor": digest(actor_path),
            "loader": digest(helper),
            "actor_source": digest(actor_source),
        },
        "analysis_script_sha256": digest(Path(__file__)),
        "torch_version": torch.__version__,
        "protocol": {
            "feature_indices_half_open": [56, 62],
            "selection": "all valid successful INSERT rows, then deterministic stride to at most 1024 rows per split",
            "central_difference_epsilon": 0.001,
            "epsilon_units": "dimensionless previous right-arm action",
            "other_features": "fixed at the recorded teacher observation",
            "jacobian": "six actor outputs by six previous right-arm action inputs",
            "interpretation_limit": "A local gain above one is not evidence of unstable robot dynamics; zeroing features is out-of-distribution.",
        },
        "splits": {},
    }
    for split in ("train", "dev"):
        path = args.teacher_root / f"teacher-{split}.h5"
        expected = module.EXPECTED_INPUT_SHA256[f"{split}_h5"]
        if digest(path) != expected:
            raise ValueError("teacher dataset bytes changed")
        result["input_sha256"][split] = expected
        data = module.load_split(path)
        mask = np.asarray([str(p).upper() == "INSERT" for p in data["phases"]])
        obs = data["obs"][mask]
        labels = data["act"][mask]
        if not len(obs) or obs.shape[1] != 79 or labels.shape[1] != 6:
            raise ValueError("INSERT row contract differs")
        pred = module.actor_prediction(actor, obs)
        modified = obs.copy()
        modified[:, 56:62] = 0.0
        zeroed = module.actor_prediction(actor, modified)
        stride = max(1, int(np.ceil(len(obs) / 1024)))
        selected = obs[::stride]
        epsilon = 0.001
        columns = []
        for index in range(56, 62):
            plus = selected.copy()
            minus = selected.copy()
            plus[:, index] += epsilon
            minus[:, index] -= epsilon
            columns.append(
                (module.actor_prediction(actor, plus) - module.actor_prediction(actor, minus))
                / (2.0 * epsilon)
            )
        jacobian = np.stack(columns, axis=-1).astype(np.float64)
        if not np.isfinite(jacobian).all():
            raise ValueError("nonfinite local derivative")
        spectral = np.linalg.svd(jacobian, compute_uv=False)[:, 0]
        radius = np.max(np.abs(np.linalg.eigvals(jacobian)), axis=1)
        result["splits"][split] = {
            "insert_rows": len(obs),
            "selected_rows": len(selected),
            "stride": stride,
            "actor_label_mse": module.mean_squared_error(pred, labels),
            "previous_action_copy_mse": module.mean_squared_error(obs[:, 56:62], labels),
            "zero_previous_action_mse": module.mean_squared_error(zeroed, labels),
            "zero_previous_action_prediction_change_rms": float(np.sqrt(np.mean((zeroed - pred) ** 2))),
            "jacobian_spectral_norm_quantiles": module.quantiles(spectral),
            "jacobian_spectral_radius_quantiles": module.quantiles(radius),
            "spectral_norm_above_one_fraction": float(np.mean(spectral > 1.0)),
            "spectral_radius_above_one_fraction": float(np.mean(radius > 1.0)),
            "mean_jacobian": jacobian.mean(axis=0).tolist(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "sha256": digest(args.output), "splits": result["splits"]}))


if __name__ == "__main__":
    main()
