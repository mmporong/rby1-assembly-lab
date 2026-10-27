#!/usr/bin/env python3
"""Validate real GPU smoke report invariants before starting development runs."""
import argparse
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("report", type=Path)
parser.add_argument("--expected-profile", required=True)
parser.add_argument("--require-zero-init", action="store_true")
args = parser.parse_args()
data = json.loads(args.report.read_text())
if data.get("rj45_evaluation_schema_version") != 2:
    raise SystemExit("evaluation schema 2 is required")
proof = data.get("provenance", {})
if proof.get("evaluation_inputs_reverified_at_end") is not True:
    raise SystemExit("evaluation input reverification is required")
outcomes = data.get("episode_outcomes", [])
if len(outcomes) != data.get("requested_attempts") or not outcomes:
    raise SystemExit("requested episode count mismatch")
for index, item in enumerate(outcomes):
    if not item.get("valid") or not item.get("held_initialization_passed"):
        raise SystemExit(f"episode {index}: invalid held initialization or policy input")
    profile = item.get("reward_profile", {})
    if profile.get("reward_profile_id") != args.expected_profile:
        raise SystemExit(f"episode {index}: unexpected reward profile")
    terms = item.get("reward_component_episode_sums", {})
    names = ("potential_progress", "force_cost", "action_cost", "terminal", "total")
    if any(name not in terms or not math.isfinite(terms[name]) for name in names):
        raise SystemExit(f"episode {index}: invalid reward components")
    if not math.isclose(sum(terms[name] for name in names[:-1]), terms["total"], rel_tol=1e-5, abs_tol=1e-3):
        raise SystemExit(f"episode {index}: component sum does not match total")
    if item.get("policy_active_step_count", 0) <= 0:
        raise SystemExit(f"episode {index}: no policy-active steps")
    if item.get("termination_reason") == "episode_timeout" and item.get("reward_truncated") is not True:
        raise SystemExit(f"episode {index}: timeout lacks truncation telemetry")
    if item.get("success") and item.get("reward_true_terminated") is not True:
        raise SystemExit(f"episode {index}: success lacks terminal telemetry")
if args.require_zero_init:
    init = proof.get("verified_checkpoint_evidence", {}).get("policy_initialization", {})
    if init.get("mode") != "residual_actor_final_linear_zero" or not init.get("deterministic_actor_output_all_zero") or init.get("stochastic_policy_noise_modified") is not False:
        raise SystemExit("Z1 actor initialization evidence mismatch")
print(json.dumps({"passed": True, "report": str(args.report), "episodes": len(outcomes), "profile": args.expected_profile, "terminations": sorted({item.get("termination_reason") for item in outcomes})}))
