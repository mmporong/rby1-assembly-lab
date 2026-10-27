from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_rj45_long_campaign.py"
SPEC = importlib.util.spec_from_file_location("prepare_rj45_long_campaign", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)
RUNNER_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_rj45_long_campaign.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("campaign_contract_for_prepare_test", RUNNER_SCRIPT)
assert RUNNER_SPEC and RUNNER_SPEC.loader
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC); sys.modules[RUNNER_SPEC.name] = RUNNER; RUNNER_SPEC.loader.exec_module(RUNNER)
GROUPS = ("A0", "A1", "A2", "E1", "Z1")


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path, data: bytes) -> Path: path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data); return path
def write_json(path: Path, value: dict) -> Path: return write(path, (json.dumps(value, sort_keys=True) + "\n").encode())
def ref(root: Path, path: Path) -> dict: return {"path": str(path.relative_to(root)), "sha256": sha(path)}


def fixture(root: Path) -> dict:
    (root / "artifacts").mkdir(parents=True)
    source = write(root / "src/rby1_assembly/core.py", b"VALUE=1\n")
    evaluate = write(root / "scripts/evaluate_policy.py", b"# evaluate\n")
    analyzer = write(root / "scripts/analyze_rj45_paired.py", b"# v1 analyzer\n")
    runner = write(root / "scripts/run_rj45_long_campaign.py", b"# runner\n")
    inspector = write(root / "scripts/inspect_rj45_checkpoint.py", b"# inspector\n")
    training_script = write(root / "scripts/train.py", b"# train\n")
    training = write_json(root / "configs/training.json", {"schema_version": 2})
    ppo = write_json(root / "configs/ppo.json", {"seed": 7, "max_iterations": 20, "save_interval": 50, "num_steps_per_env": 128, "policy": {"init_noise_std": 0.1}})
    task = write_json(root / "configs/tasks/rj45-v1.json", {"task": "rj45"})
    support = [write(root / f"frozen/support-{index}.bin", f"support-{index}".encode()) for index in range(50)]
    parent_frozen_paths = (source, evaluate, analyzer, training, ppo, task, training_script, *support)
    parent = write_json(root / "evidence/parent.json", {
        "schema_version": 1,
        "groups": [{"id": group, "training_config": "configs/training.json", "ppo_config": "configs/ppo.json", "training_flags": ["--zero-init-actor-output"] if group == "Z1" else []} for group in GROUPS],
        "frozen_inputs": {"sha256": {str(path.relative_to(root)): sha(path) for path in parent_frozen_paths}},
    })
    plan = write_json(root / "docs/long-plan.json", {
        "schema_version": 1, "parent_pilot_plan_sha256": sha(parent),
        "pilot_candidate_selection": {"eligible_groups": list(GROUPS), "primary_rank": "Largest strict-v1 success count on the predeclared 32 pilot development cases.", "tie_break_order": ["Z1", "E1", "A2", "A1", "A0"]},
        "training": {"training_seeds": [101, 102, 103], "num_envs": 64, "steps_per_env_per_update": 128, "updates_per_seed": 200, "checkpoint_save_interval_iterations": 19, "assessed_checkpoints": [{"filename": f"model_{i}.pt", "iteration_index": i, "completed_updates": i + 1} for i in (19,95,199)]},
        "development_evaluation": {"seed": 20261028, "num_envs": 16, "attempts_per_policy": 32},
        "final_evaluation": {"seed": 20261027, "status": "sealed_until_candidate_and_common_checkpoint_and_final_protocol_are_frozen"},
        "budget": {"bounded_guard_max_hours_from_follow_up_start": 9, "minimum_free_bytes_local_and_remote": 1},
    })
    trace = write(root / "runtime/nominal.jsonl", b"{}\n")
    nominal = write_json(root / "runtime/nominal.json", {"trace_path": str(trace.relative_to(root)), "trace_sha256": sha(trace)})
    held_dependency = write_json(root / "runtime/build.json", {"ok": True})
    held = write_json(root / "runtime/held.json", {"provenance": {"source_and_config_sha256": {str(held_dependency.relative_to(root)): sha(held_dependency)}}})
    contact_report = write_json(root / "runtime/contact-report.json", {"ok": True})
    contact_trace = write(root / "runtime/contact-trace.jsonl", b"{}\n")
    contact_sat = write_json(root / "runtime/contact-sat.json", {"ok": True})
    contact = write_json(root / "runtime/contact-spec.json", {
        "report_path": str(contact_report.relative_to(root)), "report_sha256": sha(contact_report),
        "trace_path": str(contact_trace.relative_to(root)), "trace_sha256": sha(contact_trace),
        "sat_report_path": str(contact_sat.relative_to(root)), "sat_report_sha256": sha(contact_sat),
    })
    reports = {label: write_json(root / f"pilot/{label}.json", {
        "label": label,
        "rj45_training_config": {"path": "configs/training.json", "sha256": sha(training)},
        "provenance": {
            "policy_config": "configs/ppo.json", "policy_config_sha256": sha(ppo),
            "rj45_nominal_evidence": {"report_sha256": sha(nominal), "trace_sha256": sha(trace)},
            "held_grasp_reset_proof": {"artifact_sha256": sha(held), "nominal_report_sha256": sha(nominal), "nominal_trace_sha256": sha(trace)},
        },
        "parallel_contact_evidence": {"report_sha256": sha(contact_report), "trace_sha256": sha(contact_trace), "sat_report_sha256": sha(contact_sat)},
    }) for label in ("baseline", *GROUPS)}
    bundles = {}
    for group in GROUPS:
        bundles[group] = {}
        for field in ("checkpoint", "checkpoint_manifest", "training_provenance", "runner_config"):
            path = write(root / f"pilot/{group}/{field}.bin", f"{group}-{field}".encode())
            bundles[group][field] = ref(root, path)
    raw_files = [*reports.values(), *(root / item["path"] for bundle in bundles.values() for item in bundle.values())]
    audit = write_json(root / "evidence/audit.json", {"schema_version": 1, "results": {group: {"attempts": 32, "successes_all_attempts": 1 if group == "Z1" else 0, "invalid_episodes": 0} for group in GROUPS}, "input_reports": {label: ref(root, path) for label, path in reports.items()}, "training_artifact_verification": {group: {"status": "checkpoint_manifest_provenance_runner_bytes_verified", **bundles[group]} for group in GROUPS}})
    immutable = {str(path.relative_to(root)): sha(path) for path in (source, evaluate, analyzer, training, ppo, task)}
    candidate = write_json(root / "evidence/candidate.json", {
        "schema_version": 2,
        "decision": {"selected_group": "Z1", "selected_successes_all_attempts": 1, "attempts": 32, "selection_metric": "strict-v1 successes_all_attempts", "tie_break_order": ["Z1", "E1", "A2", "A1", "A0"], "reward_sums_used": False},
        "ranking": [{"group": group, "eligible": True, "successes_all_attempts": 1 if group == "Z1" else 0, "attempts": 32, "invalid_episodes": 0, "maximum_forbidden_contact_load_peak_n": 0.0, "checkpoint_bundle_status": "checkpoint_manifest_provenance_runner_bytes_verified", "tie_break_position": ["Z1", "E1", "A2", "A1", "A0"].index(group)} for group in GROUPS],
        "verification_scope": {"status": "raw_reports_and_training_bundles_rehashed_and_fresh_analyzer_replayed"},
        "selected_profile": {"training_config": ref(root, training), "ppo_config": ref(root, ppo), "zero_init_actor_output": True},
        "candidate_evidence": {group: {"raw_report": ref(root, reports[group]), "checkpoint_bundle": bundles[group]} for group in GROUPS},
        "inputs": {"long_run_plan": ref(root, plan), "parent_pilot_plan": ref(root, parent), "stored_paired_audit": ref(root, audit), "frozen_analyzer": ref(root, analyzer), "raw_evidence_sha256": {str(path.relative_to(root)): sha(path) for path in raw_files}, "immutable_inputs_sha256": immutable},
    })
    guard = write_json(root / "evidence/guard-record.json", {
        "schema_version": 1, "candidate_selection_sha256": sha(candidate), "long_run_plan_sha256": sha(plan),
        "guard_started_utc": "2026-10-27T00:00:00Z", "guard_deadline_utc": "2026-10-27T09:00:00Z",
        "brev_instance": {"id": "inst-1", "name": "rj45", "organization": "org"},
        "timer_target": "rby1-rj45-stop.timer", "observation_scope": "operator_recorded_not_provider_authenticated",
    })
    return locals()


def kwargs(f: dict) -> dict:
    root = f["root"]
    return dict(project_root=root, candidate_path=f["candidate"], candidate_sha256=sha(f["candidate"]), long_plan_path=f["plan"], long_plan_sha256=sha(f["plan"]), runner_path=f["runner"], runner_sha256=sha(f["runner"]), inspector_path=f["inspector"], inspector_sha256=sha(f["inspector"]), training_script_path=f["training_script"], training_script_sha256=sha(f["training_script"]), analyzer_path=f["analyzer"], analyzer_sha256=sha(f["analyzer"]), guard_record_path=f["guard"], guard_record_sha256=sha(f["guard"]), nominal_path=f["nominal"], nominal_sha256=sha(f["nominal"]), held_path=f["held"], held_sha256=sha(f["held"]), contact_path=f["contact"], contact_sha256=sha(f["contact"]), guard_started_utc="2026-10-27T00:00:00Z", guard_deadline_utc="2026-10-27T09:00:00Z", timer_target="rby1-rj45-stop.timer", instance_id="inst-1", instance_name="rj45", instance_organization="org", output_dir=root / "artifacts/prepared", poll_interval_s=30.0, hang_timeout_s=900.0)


def refresh_guard(f: dict) -> None:
    guard = json.loads(f["guard"].read_text())
    guard["candidate_selection_sha256"] = sha(f["candidate"])
    write_json(f["guard"], guard)


class PrepareTests(unittest.TestCase):
    def test_success_generates_contract_and_seed_configs_with_only_three_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); result = MODULE.prepare(**kwargs(f))
            launch = json.loads(result["launch_manifest"].read_text()); manifest = json.loads(result["input_manifest"].read_text())
            self.assertEqual(launch["guard_started_utc"], "2026-10-27T00:00:00Z")
            self.assertEqual(launch["guard_deadline_utc"], "2026-10-27T09:00:00Z")
            self.assertEqual(launch["timer_target"], "rby1-rj45-stop.timer")
            self.assertEqual(launch["brev_instance"], {"id": "inst-1", "name": "rj45", "organization": "org"})
            self.assertEqual(launch["input_manifest"]["sha256"], sha(result["input_manifest"]))
            self.assertEqual(len(manifest["inputs"]), len(set(manifest["inputs"])))
            self.assertEqual(len(f["parent_frozen_paths"]), 57)
            for frozen_path in f["parent_frozen_paths"]:
                self.assertEqual(manifest["inputs"][str(frozen_path.relative_to(f["root"]))], sha(frozen_path))
            self.assertEqual(manifest["inputs"][str(f["audit"].relative_to(f["root"]))], sha(f["audit"]))
            for required in (f["trace"], f["held_dependency"], f["contact_report"], f["contact_trace"], f["contact_sat"], f["runner"], f["inspector"], f["training_script"], f["guard"]):
                self.assertEqual(manifest["inputs"][str(required.relative_to(f["root"]))], sha(required))
            self.assertTrue(result["guard_anchor"].is_file())
            base = json.loads(f["ppo"].read_text())
            for seed in (101, 102, 103):
                generated = json.loads(result[f"seed_config_{seed}"].read_text())
                expected = copy.deepcopy(base); expected.update(seed=seed, max_iterations=200, save_interval=19)
                self.assertEqual(generated, expected)
                self.assertTrue(str(result[f"seed_config_{seed}"].relative_to(f["root"])).startswith("artifacts/"))
                self.assertEqual(launch["seed_ppo_configs"][str(seed)]["sha256"], sha(result[f"seed_config_{seed}"]))
            summary = lambda _root, _report, label, _attempts, _analyzer: {"attempts": 32, "successes_all_attempts": 1 if label == "Z1" else 0, "invalid_episodes": 0, "maximum_forbidden_contact_load_peak_n": 0.0}
            guard = datetime(2026, 10, 27, tzinfo=timezone.utc)
            with mock.patch.object(RUNNER, "_revalidate_pilot_report", side_effect=summary), mock.patch.object(RUNNER, "_revalidate_pilot_bundle", return_value=None):
                contract = RUNNER.load_contract(project_root=f["root"], plan_path=f["plan"], expected_plan_sha256=sha(f["plan"]), selection_path=result["launch_manifest"], expected_selection_sha256=sha(result["launch_manifest"]), expected_instance_id="inst-1", expected_instance_name="rj45", expected_instance_organization="org", output_dir=f["root"] / "artifacts/run-output", deadline_utc=launch["guard_deadline_utc"], python=Path(sys.executable), now=lambda: guard, disk_free=lambda _p: 10**12)
            self.assertEqual(contract.profile_label, "Z1")
            self.assertEqual(contract.training_seeds, (101, 102, 103))
            values = kwargs(f); values["output_dir"] = f["root"] / "artifacts/prepared-again"
            recovered = MODULE.prepare(**values)
            self.assertTrue(recovered["launch_manifest"].is_file())

    def test_runtime_evidence_and_guard_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); nominal = json.loads(f["nominal"].read_text()); nominal["changed"] = True
            write_json(f["nominal"], nominal); values = kwargs(f)
            with self.assertRaisesRegex(MODULE.PreparationError, "nominal report SHA differs"):
                MODULE.prepare(**values)
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); write_json(f["held"], {"substituted": True}); values = kwargs(f)
            with self.assertRaisesRegex(MODULE.PreparationError, "held artifact SHA differs"):
                MODULE.prepare(**values)
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); replacement = write_json(f["root"] / "runtime/replacement.json", {"wrong": True})
            contact = json.loads(f["contact"].read_text()); contact["report_path"] = str(replacement.relative_to(f["root"])); contact["report_sha256"] = sha(replacement)
            write_json(f["contact"], contact); values = kwargs(f)
            with self.assertRaisesRegex(MODULE.PreparationError, "contact report differs"):
                MODULE.prepare(**values)
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); values = kwargs(f); values["output_dir"] = f["root"] / "artifacts/first-failure"
            original_verify = MODULE._verify
            def mutate_on_final(path, digest, name):
                if name == "frozen prepared input" and path == f["nominal"]:
                    path.write_text("{}\n")
                original_verify(path, digest, name)
            with mock.patch.object(MODULE, "_verify", side_effect=mutate_on_final):
                with self.assertRaisesRegex(MODULE.PreparationError, "frozen prepared input SHA-256 mismatch"):
                    MODULE.prepare(**values)
            self.assertFalse((values["output_dir"] / "launch-manifest.json").exists())
            self.assertTrue((values["output_dir"] / "input-manifest.json").exists())
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); MODULE.prepare(**kwargs(f))
            guard = json.loads(f["guard"].read_text()); guard["guard_deadline_utc"] = "2026-10-27T08:00:00Z"; write_json(f["guard"], guard)
            values = kwargs(f); values["guard_deadline_utc"] = "2026-10-27T08:00:00Z"; values["output_dir"] = f["root"] / "artifacts/different-guard"
            with self.assertRaisesRegex(MODULE.PreparationError, "guard anchor already exists with different bytes"):
                MODULE.prepare(**values)

    def test_bad_hash_parent_mismatch_missing_dependency_and_overwrite_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); values = kwargs(f); values["candidate_sha256"] = "0" * 64
            with self.assertRaisesRegex(MODULE.PreparationError, "candidate selection SHA"):
                MODULE.prepare(**values)
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); f["training"].write_text('{"changed":true}\n')
            candidate = json.loads(f["candidate"].read_text()); changed_sha = sha(f["training"])
            candidate["selected_profile"]["training_config"]["sha256"] = changed_sha
            candidate["inputs"]["immutable_inputs_sha256"]["configs/training.json"] = changed_sha
            write_json(f["candidate"], candidate); refresh_guard(f); values = kwargs(f)
            with self.assertRaisesRegex(MODULE.PreparationError, "parent frozen input SHA-256 mismatch"):
                MODULE.prepare(**values)
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); f["trace"].unlink()
            with self.assertRaisesRegex(MODULE.PreparationError, "unable to hash.*nominal.jsonl"):
                MODULE.prepare(**kwargs(f))
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); values = kwargs(f); values["output_dir"].mkdir()
            with self.assertRaisesRegex(MODULE.PreparationError, "already exists"):
                MODULE.prepare(**values)
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); candidate = json.loads(f["candidate"].read_text())
            candidate["inputs"]["raw_evidence_sha256"].pop("pilot/A0.json")
            write_json(f["candidate"], candidate); refresh_guard(f); values = kwargs(f)
            with self.assertRaisesRegex(MODULE.PreparationError, "raw evidence map differs"):
                MODULE.prepare(**values)

    def test_source_path_addition_and_invalid_guard_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); write(f["root"] / "src/rby1_assembly/added.py", b"x=1\n")
            with self.assertRaisesRegex(MODULE.PreparationError, "source path set"):
                MODULE.prepare(**kwargs(f))
        with tempfile.TemporaryDirectory() as directory:
            f = fixture(Path(directory)); values = kwargs(f); values["guard_started_utc"] = "today"
            with self.assertRaisesRegex(MODULE.PreparationError, "YYYY"):
                MODULE.prepare(**values)


if __name__ == "__main__": unittest.main()
