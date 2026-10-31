"""Audit the two preregistered, fully completed patched-source PPO runs."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROTOCOLS = {
    "standard79": "a1a00452c8c7a6151844e2b3bd56d9322a2baaa6f1c4b7e9c2bb925be07959ef",
    "oracle92": "6f4dab69036c9e0cd30a9cfaf813a2445616b71864ae0faf776a4d947640a5e1",
}


def audit(root: Path) -> dict:
    reports = {}
    checks = {}
    summaries = {}
    for mode, protocol_sha in PROTOCOLS.items():
        path = root / f"{mode}-report.json"
        report = json.loads(path.read_text())
        reports[mode] = report
        checks[f"{mode}_completed_full_run"] = (
            report["status"] == "completed"
            and report["mode"] == mode
            and report["learning"]["executed"] is True
            and report["learning"]["updates"] == 32
            and report["inputs_unchanged"] is True
            and report["protocol"]["file_sha256"] == protocol_sha
            and report["pre_evaluation"]["classification"] == "valid_failure"
            and report["final_evaluation"]["classification"] in ("valid_success", "valid_failure")
        )
        raw_updates = gzip.decompress((root / f"{mode}-updates.jsonl.gz").read_bytes())
        updates = [json.loads(line) for line in raw_updates.splitlines()]
        checks[f"{mode}_durable_updates"] = (
            hashlib.sha256(raw_updates).hexdigest() == report["learning"]["updates_trace"]["sha256"]
            and [row["update"] for row in updates] == list(range(1, 33))
            and updates == report["learning"]["update_metrics"]
        )
        raw_trace = gzip.decompress((root / f"{mode}-trace.jsonl.gz").read_bytes())
        checks[f"{mode}_final_trace"] = hashlib.sha256(raw_trace).hexdigest() == report["final_evaluation"]["trace"]["sha256"]
        checks[f"{mode}_actor_updated"] = report["learning"]["actor_parameters_changed"] is True
        summaries[mode] = {
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "updates": 32,
            "transitions": report["protocol"]["payload"]["ppo"]["transitions"],
            "before": report["pre_evaluation"]["classification"],
            "after": report["final_evaluation"]["classification"],
            "terminal": report["final_evaluation"]["terminal"],
            "checkpoint_sha256": report["learning"]["checkpoint"]["sha256"],
        }
    standard, oracle = (reports[mode] for mode in ("standard79", "oracle92"))
    left, right = standard["learning"], oracle["learning"]
    checks["common79_actor_initialization"] = left["initialization"]["common_79d_projection_sha256"] == right["initialization"]["common_79d_projection_sha256"]
    checks["critic92_initialization"] = left["critic_initialization"]["state_sha256"] == right["critic_initialization"]["state_sha256"]
    checks["optimizer_initialization"] = left["optimizer_initial_state_sha256"] == right["optimizer_initial_state_sha256"]
    checks["initialization_rng"] = left["initialization_rng_state_sha256"] == right["initialization_rng_state_sha256"]
    checks["training_rng"] = left["training_start_rng"]["state_sha256"] == right["training_start_rng"]["state_sha256"]
    checks["zero_initial_actor"] = left["zero_actor_contract"]["action_sha256"] == right["zero_actor_contract"]["action_sha256"]
    checks["fixed_condition"] = standard["fixed_condition"] == oracle["fixed_condition"]
    checks["runtime_versions"] = standard["runtime_versions"] == oracle["runtime_versions"]
    checks["end_input_revalidation"] = standard["input_revalidation"] == oracle["input_revalidation"]
    payloads = [copy.deepcopy(report["protocol"]["payload"]) for report in (standard, oracle)]
    for payload in payloads:
        payload["budget_and_stop"]["absolute_deadline_utc"] = "<preregistered runtime window>"
    checks["protocol_only_deadline_differs"] = payloads[0] == payloads[1]
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "primary_pair_verified" if not failures else "primary_pair_incomplete",
        "checks": checks,
        "failures": failures,
        "runs": summaries,
        "scope": "one seed, one selected train condition; invalid ancestor and checkpoint recovery excluded",
        "reinforcement_learning_effect_proven": False,
        "heldout_generalization": False,
        "no_substitution_or_rerun_on_audit_failure": True,
    }


if __name__ == "__main__":
    try:
        result = audit(ROOT)
    except (OSError, ValueError, KeyError, TypeError) as error:
        result = {"status": "primary_pair_incomplete", "error": str(error), "reinforcement_learning_effect_proven": False}
    (ROOT / "primary-pair-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "failures": result.get("failures", [])}))
    raise SystemExit(0 if result["status"] == "primary_pair_verified" else 2)
