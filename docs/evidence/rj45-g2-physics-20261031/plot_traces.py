"""Plot the archived fixed-condition traces without importing the simulator."""

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULT = json.loads((ROOT / "result.json").read_text())
fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True, layout="constrained")
for label, color, short in (
    ("a_standard_input", "#2563eb", "A: standard input"),
    ("b_normalized_input", "#df593d", "B: normalized input"),
):
    metadata = RESULT["policies"][label]["trace"]
    payload = (ROOT / metadata["path"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == metadata["gzip_sha256"]
    raw = gzip.decompress(payload)
    assert hashlib.sha256(raw).hexdigest() == metadata["uncompressed_sha256"]
    records = [json.loads(line) for line in raw.splitlines()]
    rows = [row for row in records if row.get("policy_input_valid") is True]
    time_s = np.array([row["timestamp_s"] for row in rows])
    axes[0].plot(time_s, [row["actual_depth_m"] * 1000 for row in rows], color=color, label=short)
    axes[1].plot(time_s, [max(row["actual_lateral_error_m"] * 1e6, 0.01) for row in rows], color=color)
    axes[2].plot(time_s, [max(np.degrees(row["actual_orientation_error_rad"]), 1e-5) for row in rows], color=color)
    action = np.array([row["applied_bounded_action_6d"] for row in rows])
    axes[3].plot(time_s, np.linalg.norm(action[:, :3], axis=1), color=color, label=f"{short}, translation")
    axes[3].plot(time_s, np.linalg.norm(action[:, 3:], axis=1), color=color, linestyle="--", label=f"{short}, rotation")

axes[0].axhspan(12, 13, color="#21a179", alpha=0.13)
axes[0].set_ylabel("Insertion depth (mm)")
axes[0].legend(loc="lower right", frameon=False)
axes[1].axhline(50, color="#475569", linestyle=":", label="50 um criterion")
axes[1].set_yscale("log")
axes[1].set_ylabel("Lateral error (um)")
axes[1].legend(loc="lower right", frameon=False)
axes[2].axhline(0.5, color="#475569", linestyle=":", label="0.5 deg criterion")
axes[2].set_yscale("log")
axes[2].set_ylabel("Orientation error (deg)")
axes[2].legend(loc="lower right", frameon=False)
axes[3].set_ylabel("Bounded command norm\n(unitless; each triplet)")
axes[3].set_ylim(-0.03, 1.05)
axes[3].legend(loc="upper left", framealpha=0.95, ncol=2, fontsize=8)
axes[3].set_xlabel("Control time since reset (s; valid-policy-input rows only)")
for axis in axes:
    axis.grid(alpha=0.15)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_facecolor("white")
fig.set_facecolor("white")
fig.suptitle("Fixed training condition: one completed episode per policy", fontsize=13)
fig.savefig(ROOT / "physics-comparison.png", dpi=160)
fig.savefig(ROOT / "physics-comparison.svg")
svg_path = ROOT / "physics-comparison.svg"
svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
