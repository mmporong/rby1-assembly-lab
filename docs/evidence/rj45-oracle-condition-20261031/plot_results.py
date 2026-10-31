"""Plot recorded insertion episodes; no rollout or outcome synthesis."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--controls-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    task_path = root.parents[2] / "configs/tasks/rj45-v1.json"
    task_sha = hashlib.sha256(task_path.read_bytes()).hexdigest()
    protocol = json.loads((root / "protocol-20261128.json").read_text())
    if task_sha != protocol["frozen_inputs"]["task_config_sha256"]:
        raise ValueError("plot threshold source differs from frozen task")
    lateral_limit_um = json.loads(task_path.read_text())["evaluation"]["max_lateral_error_m"] * 1e6
    modes = ["zero", "reference"]
    if not args.controls_only:
        modes += ["oracle92", "standard79"]
    colors = {"zero": "#667085", "reference": "#157f57", "oracle92": "#215bc8", "standard79": "#bf5a17"}
    labels = {"zero": "Zero residual", "reference": "GT reference", "oracle92": "PPO oracle92 (32 updates)", "standard79": "PPO standard79 (32 updates)"}
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    manifest = {}
    for mode in modes:
        filename = f"{mode}-trace.jsonl.gz"
        path = root / filename
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            records = [json.loads(line) for line in stream]
        rows = [row for row in records if row.get("active") is True]
        if not rows:
            raise ValueError(f"no active physics records: {mode}")
        times = [row["environment_timestamp_s"] for row in rows]
        if not all(b > a for a, b in zip(times, times[1:])):
            raise ValueError(f"timestamps do not increase: {mode}")
        terminal = records[-1]
        if terminal.get("event") != "complete" or terminal.get("classification") not in ("valid_success", "valid_failure"):
            raise ValueError(f"episode did not complete validly: {mode}")
        label = labels[mode] + (" / pass" if terminal["classification"] == "valid_success" else " / fail")
        axes[0].plot(times, [row["lateral_error_m"] * 1e6 for row in rows], color=colors[mode], label=label, linewidth=1.5)
        axes[1].plot(times, [row["actual_depth_m"] * 1e3 for row in rows], color=colors[mode], linewidth=1.5)
        manifest[mode] = {"trace": filename, "compressed_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "raw_sha256": hashlib.sha256(gzip.decompress(path.read_bytes())).hexdigest(), "active_rows": len(rows), "classification": terminal["classification"], "last_lateral_error_um": rows[-1]["lateral_error_m"] * 1e6, "last_depth_mm": rows[-1]["actual_depth_m"] * 1e3}
    axes[0].axhline(lateral_limit_um, color="#a32e40", linestyle="--", linewidth=1, label=f"Project lateral limit: {lateral_limit_um:g} um")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Lateral error [um, log scale]")
    axes[1].set_ylabel("Insertion depth [mm]")
    for ax in axes:
        ax.set_xlabel("Episode simulation time [s]")
        ax.grid(True, alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=7.5, loc="best")
    figure.suptitle("Train-02: one selected insertion condition / restored held grasp", fontsize=12)
    name = "controls" if args.controls_only else "comparison"
    figure.savefig(root / f"{name}.png", dpi=180, facecolor="white")
    plt.close(figure)
    (root / f"{name}-plot-data.json").write_text(json.dumps({"scope": "one train condition; no generalization or effect estimate", "warmup_rows": "excluded; original episode timestamps preserved", "threshold_source_sha256": task_sha, "lateral_limit_um": lateral_limit_um, "traces": manifest}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
