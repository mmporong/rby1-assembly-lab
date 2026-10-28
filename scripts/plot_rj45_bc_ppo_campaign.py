#!/usr/bin/env python3
"""Render white-background PNG and SVG figures from a verified campaign audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_audit(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("status") != "campaign_audit_complete"
        or value.get("contract", {}).get("diagnostic_checkpoints") != [20, 96]
        or value.get("contract", {}).get("selection_checkpoint") != 200
        or value.get("contract", {}).get("attempts_per_report") != 64
    ):
        raise ValueError("campaign audit contract differs")
    rows = value.get("learned_reports")
    if not isinstance(rows, list) or len(rows) != 18:
        raise ValueError("campaign audit must contain 18 learned report rows")
    return value


def median(summary: object) -> float:
    if not isinstance(summary, dict) or not isinstance(summary.get("0.5"), (int, float)):
        raise ValueError("metric summary lacks a median")
    return float(summary["0.5"])


def render(audit_path: Path, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    audit = load_audit(audit_path)
    rows = audit["learned_reports"]
    conditions = ("ppo_only", "bc_then_ppo")
    labels = {"ppo_only": "PPO-only", "bc_then_ppo": "BC → PPO"}
    colors = {"ppo_only": "#2563EB", "bc_then_ppo": "#EA580C"}
    checkpoints = (20, 96, 200)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    figure = cast(Any, figure)

    success_axis = axes[0, 0]
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        seeds = sorted({row["training_seed"] for row in condition_rows})
        for seed in seeds:
            values = [
                next(
                    row["successes"]
                    for row in condition_rows
                    if row["training_seed"] == seed and row["completed_updates"] == update
                )
                for update in checkpoints
            ]
            success_axis.plot(checkpoints, values, color=colors[condition], alpha=0.28, linewidth=1)
            success_axis.scatter(checkpoints, values, color=colors[condition], alpha=0.55, s=18)
        means = [
            audit["aggregates"][condition][str(update)]["success_mean_over_three_training_seeds"]
            for update in checkpoints
        ]
        success_axis.plot(checkpoints, means, color=colors[condition], marker="o", linewidth=2.5, label=labels[condition])
    success_axis.axhline(audit["baselines"]["biased_zero"]["successes"], color="#475569", linestyle="--", label="biased-zero")
    success_axis.axhline(audit["baselines"]["bc_only"]["successes"], color="#7C3AED", linestyle=":", label="BC-only")
    success_axis.set(title="Successes out of 64", xlabel="Completed PPO updates", ylabel="Success count", xticks=checkpoints, ylim=(0, 64))
    success_axis.legend(frameon=False, fontsize=8)
    success_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)

    final_rows = [row for row in rows if row["completed_updates"] == 200]
    names = ["biased-zero", "BC-only", "PPO-only", "BC → PPO"]
    groups = [
        [audit["baselines"]["biased_zero"]],
        [audit["baselines"]["bc_only"]],
        [row for row in final_rows if row["condition"] == "ppo_only"],
        [row for row in final_rows if row["condition"] == "bc_then_ppo"],
    ]

    lateral_axis = axes[0, 1]
    for index, group in enumerate(groups):
        values = [median(row["final_lateral_error_m"]) * 1.0e6 for row in group]
        lateral_axis.scatter([index] * len(values), values, color="#0F766E", s=38, zorder=3)
        lateral_axis.hlines(sum(values) / len(values), index - 0.22, index + 0.22, color="#0F172A", linewidth=2)
    lateral_axis.axhline(50.0, color="#DC2626", linestyle="--", linewidth=1, label="50 µm success gate")
    lateral_axis.set(title="Final lateral error median", ylabel="µm", xticks=range(4), xticklabels=names)
    lateral_axis.tick_params(axis="x", rotation=18)
    lateral_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    lateral_axis.legend(frameon=False, fontsize=8)

    force_axis = axes[1, 0]
    for index, group in enumerate(groups):
        values = [float(row["force_proxy_max_n"]) for row in group]
        force_axis.scatter([index] * len(values), values, color="#B45309", s=38, zorder=3)
        force_axis.hlines(sum(values) / len(values), index - 0.22, index + 0.22, color="#0F172A", linewidth=2)
    force_axis.set(title="Reported contact-force proxy maximum", ylabel="N", xticks=range(4), xticklabels=names)
    force_axis.tick_params(axis="x", rotation=18)
    force_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)

    time_axis = axes[1, 1]
    width = 0.34
    for index, group in enumerate(groups):
        success_values = [median(row["success_elapsed_s"]) for row in group if row["success_elapsed_s"] is not None]
        failure_values = [median(row["failed_elapsed_s"]) for row in group if row["failed_elapsed_s"] is not None]
        if success_values:
            time_axis.scatter([index - width / 2] * len(success_values), success_values, color="#16A34A", s=34, label="successful episodes" if index == 0 else None)
        if failure_values:
            time_axis.scatter([index + width / 2] * len(failure_values), failure_values, color="#64748B", s=34, label="failed episodes" if index == 0 else None)
    time_axis.axhline(24.0, color="#94A3B8", linestyle="--", linewidth=1, label="24 s horizon")
    time_axis.set(title="Episode elapsed-time median", ylabel="s", xticks=range(4), xticklabels=names)
    time_axis.tick_params(axis="x", rotation=18)
    time_axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    time_axis.legend(frameon=False, fontsize=8)

    figure.suptitle("RJ45 paired PPO campaign · raw seed points and three-seed means", fontsize=14)
    outputs = [output_dir / "rj45-bc-ppo-campaign.png", output_dir / "rj45-bc-ppo-campaign.svg"]
    for path in outputs:
        if path.exists():
            raise FileExistsError(path)
        figure.savefig(path, dpi=180 if path.suffix == ".png" else None)
    plt.close(figure)
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    audit = args.audit.resolve()
    if sha256(audit) != args.audit_sha256:
        raise ValueError("campaign audit SHA differs")
    outputs = render(audit, args.output_dir.resolve())
    print(json.dumps({str(path): sha256(path) for path in outputs}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
