from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import types
import unittest
import venv
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_rj45_long_campaign.py"
SPEC = importlib.util.spec_from_file_location("run_rj45_long_campaign", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
GROUPS = ("A0", "A1", "A2", "E1", "Z1")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def _ref(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": _sha(path)}


def _fixture(root: Path) -> dict:
    guard_started = datetime.now(timezone.utc).replace(microsecond=0)
    guard_deadline = guard_started + timedelta(hours=8)
    scripts = ("train.py", "evaluate_policy.py", "inspect_rj45_checkpoint.py", "analyze_rj45_paired.py", "run_rj45_long_campaign.py")
    for relative in [*(f"scripts/{name}" for name in scripts), "src/rby1_assembly/__init__.py", "src/rby1_assembly/task.py"]:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {relative}\n", encoding="utf-8")
    base_ppo = {"seed": 11, "num_steps_per_env": 128, "max_iterations": 20, "save_interval": 50, "policy": {"init_noise_std": 0.1}}
    ppo = root / "configs/ppo.json"
    training = root / "configs/training.json"
    _write_json(ppo, base_ppo)
    _write_json(training, {"schema_version": 2})
    evidence = {}
    for name in ("nominal", "held", "parallel"):
        evidence[name] = root / f"evidence/{name}.json"
        _write_json(evidence[name], {"kind": name})
    parent = root / "evidence/pilot-plan.json"
    _write_json(parent, {"schema_version": 1, "groups": [
        {"id": group, "training_config": str(training.relative_to(root)), "ppo_config": str(ppo.relative_to(root)), "training_flags": ["--zero-init-actor-output"] if group == "Z1" else []}
        for group in GROUPS
    ], "frozen_inputs": {"sha256": {str(training.relative_to(root)): _sha(training), str(ppo.relative_to(root)): _sha(ppo)}}})
    seeds = (20261101, 20261102, 20261103)
    seed_configs = {}
    for seed in seeds:
        payload = json.loads(json.dumps(base_ppo)); payload.update(seed=seed, max_iterations=200, save_interval=19)
        seed_configs[seed] = root / f"configs/seed-{seed}.json"; _write_json(seed_configs[seed], payload)
    plan = root / "docs/long-run-plan.json"
    _write_json(plan, {
        "schema_version": 1, "parent_pilot_plan_sha256": _sha(parent),
        "pilot_candidate_selection": {"eligible_groups": list(GROUPS), "primary_rank": "Largest strict-v1 success count on the predeclared 32 pilot development cases.", "tie_break_order": ["Z1", "E1", "A2", "A1", "A0"]},
        "training": {"training_seeds": list(seeds), "num_envs": 64, "steps_per_env_per_update": 128, "updates_per_seed": 200, "checkpoint_save_interval_iterations": 19,
            "assessed_checkpoints": [{"filename": f"model_{i}.pt", "iteration_index": i, "completed_updates": i + 1} for i in (19, 95, 199)]},
        "development_evaluation": {"seed": 20261028, "num_envs": 16, "attempts_per_policy": 32},
        "final_evaluation": {"seed": 20261027, "status": "sealed_until_candidate_and_common_checkpoint_and_final_protocol_are_frozen"},
        "budget": {"minimum_free_bytes_local_and_remote": 1024, "bounded_guard_max_hours_from_follow_up_start": 9},
    })
    reports = {}; bundles = {}; raw_map = {}
    for label in ("baseline", *GROUPS):
        reports[label] = root / f"evidence/{label}.json"; _write_json(reports[label], {"label": label, "rj45_training_config": {"path": str(training.relative_to(root)), "sha256": _sha(training)}, "provenance": {"policy_config": str(ppo.relative_to(root)), "policy_config_sha256": _sha(ppo)}})
        raw_map[str(reports[label].relative_to(root))] = _sha(reports[label])
    for group in GROUPS:
        group_dir = root / f"evidence/{group}-run"; group_dir.mkdir(parents=True)
        checkpoint = group_dir / "model_19.pt"; checkpoint.write_bytes(group.encode())
        manifest = group_dir / "checkpoint_manifest.json"; _write_json(manifest, {"group": group})
        provenance = group_dir / "training_provenance.json"; _write_json(provenance, {"group": group})
        runner = group_dir / "runner_config.json"; _write_json(runner, {"group": group})
        bundles[group] = {name: _ref(root, path) for name, path in {"checkpoint": checkpoint, "checkpoint_manifest": manifest, "training_provenance": provenance, "runner_config": runner}.items()}
        for path in (checkpoint, manifest, provenance, runner): raw_map[str(path.relative_to(root))] = _sha(path)
    audit = root / "evidence/paired-audit.json"
    _write_json(audit, {"schema_version": 1,
        "scope": {}, "experiment": {}, "bootstrap": {}, "paired_contrasts": {},
        "evaluation_bias_coverage": {}, "paired_cases": [],
        "results": {g: {"attempts": 32, "successes_all_attempts": 1 if g == "Z1" else 0, "invalid_episodes": 0} for g in GROUPS},
        "input_reports": {label: _ref(root, path) for label, path in reports.items()},
        "training_artifact_verification": {g: {"status": "checkpoint_manifest_provenance_runner_bytes_verified", **bundles[g]} for g in GROUPS},
    })
    immutable_paths = [root / "scripts/analyze_rj45_paired.py", root / "scripts/evaluate_policy.py", training, ppo, *(root / "src/rby1_assembly").glob("*.py")]
    candidate = root / "evidence/candidate.json"
    _write_json(candidate, {
        "schema_version": 2,
        "decision": {"selected_group": "Z1", "selected_successes_all_attempts": 1, "attempts": 32, "selection_metric": "strict-v1 successes_all_attempts", "tie_break_order": ["Z1", "E1", "A2", "A1", "A0"], "reward_sums_used": False},
        "ranking": [{"group": g, "eligible": True, "successes_all_attempts": 1 if g == "Z1" else 0, "attempts": 32, "invalid_episodes": 0, "maximum_forbidden_contact_load_peak_n": 0.0, "checkpoint_bundle_status": "checkpoint_manifest_provenance_runner_bytes_verified", "tie_break_position": ["Z1", "E1", "A2", "A1", "A0"].index(g), "exclusion_reasons": []} for g in GROUPS],
        "verification_scope": {"status": "raw_reports_and_training_bundles_rehashed_and_fresh_analyzer_replayed", "success_interpretation": "validated", "unverified": []},
        "selected_profile": {"training_config": _ref(root, training), "ppo_config": _ref(root, ppo), "zero_init_actor_output": True},
        "candidate_evidence": {g: {"raw_report": _ref(root, reports[g]), "checkpoint_bundle": bundles[g]} for g in GROUPS},
        "inputs": {"stored_paired_audit": _ref(root, audit), "long_run_plan": _ref(root, plan), "parent_pilot_plan": _ref(root, parent), "frozen_analyzer": _ref(root, root / "scripts/analyze_rj45_paired.py"), "raw_evidence_sha256": raw_map, "immutable_inputs_sha256": {str(path.relative_to(root)): _sha(path) for path in immutable_paths}},
    })
    identity = {"id": "inst-123", "name": "rj45-long", "organization": "org-abc"}
    timer_target = "rby1-rj45-stop.timer"
    guard_record = root / "evidence/guard-record.json"
    _write_json(guard_record, {
        "schema_version": 1,
        "candidate_selection_sha256": _sha(candidate),
        "long_run_plan_sha256": _sha(plan),
        "guard_started_utc": guard_started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guard_deadline_utc": guard_deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "brev_instance": identity,
        "timer_target": timer_target,
        "observation_scope": "operator_recorded_not_provider_authenticated",
    })
    anchor = root / "artifacts/rj45-long-campaign-guards" / f"{_sha(candidate)}-{_sha(plan)}.anchor.json"
    _write_json(anchor, {
        "schema_version": 1,
        "candidate_selection_sha256": _sha(candidate),
        "long_run_plan_sha256": _sha(plan),
        "guard_record": _ref(root, guard_record),
        "guard_started_utc": guard_started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "guard_deadline_utc": guard_deadline.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "brev_instance": identity,
        "timer_target": timer_target,
        "scope": "local_no_clobber_reissuance_anchor_not_provider_authenticated",
    })
    manifest_paths = [plan, parent, candidate, audit, guard_record, ppo, training, *reports.values(), *evidence.values(), *seed_configs.values(), *(root / "scripts").glob("*.py"), *(root / "src/rby1_assembly").glob("*.py")]
    for bundle in bundles.values():
        manifest_paths.extend(root / ref["path"] for ref in bundle.values())
    manifest = root / "evidence/inputs.json"
    _write_json(manifest, {"schema_version": 1, "inputs": {str(p.relative_to(root)): _sha(p) for p in manifest_paths}})
    launch = root / "evidence/launch.json"
    _write_json(launch, {"schema_version": 1, "status": "ready_for_long_campaign", "long_run_plan_sha256": _sha(plan), "guard_started_utc": guard_started.strftime("%Y-%m-%dT%H:%M:%SZ"), "guard_deadline_utc": guard_deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), "guard_record": _ref(root, guard_record), "timer_target": timer_target, "guard_verification_boundary": "record_and_anchor_are_local_integrity_evidence;_control_plane_and_timer_must_be_observed_by_operator_immediately_before_launch", "brev_instance": identity, "candidate_selection": _ref(root, candidate), "input_manifest": _ref(root, manifest), "selected_profile": {"training_config": _ref(root, training), "ppo_config": _ref(root, ppo), "zero_init_actor_output": True},
        "process_monitoring": {"poll_interval_s": 0.01, "hang_no_progress_timeout_s": 0.05},
        "runtime_inputs": {"nominal_evidence": _ref(root, evidence["nominal"]), "held_state": _ref(root, evidence["held"]), "parallel_contact_evidence": _ref(root, evidence["parallel"])},
        "seed_ppo_configs": {str(seed): _ref(root, path) for seed, path in seed_configs.items()}})
    return {"root": root, "plan": plan, "selection": launch, "candidate": candidate, "output": root / "out", "guard_started": guard_started, "guard_deadline": guard_deadline, "guard_record": guard_record, "anchor": anchor, "manifest": manifest}


def _raw_summary(_root, _ref, label, _attempts, _analyzer):
    return {"attempts": 32, "successes_all_attempts": 1 if label == "Z1" else 0, "invalid_episodes": 0, "maximum_forbidden_contact_load_peak_n": 0.0}


def _contract(f, *, python=None):
    deadline = f["guard_deadline"]
    with mock.patch.object(MODULE, "_revalidate_pilot_report", side_effect=_raw_summary), mock.patch.object(MODULE, "_revalidate_pilot_bundle", return_value=None):
        return MODULE.load_contract(project_root=f["root"], plan_path=f["plan"], expected_plan_sha256=_sha(f["plan"]), selection_path=f["selection"], expected_selection_sha256=_sha(f["selection"]), expected_instance_id="inst-123", expected_instance_name="rj45-long", expected_instance_organization="org-abc", output_dir=f["output"], deadline_utc=deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), python=python or Path(sys.executable), now=lambda: f["guard_started"], disk_free=lambda _p: 10**12)


def _arg(command, name): return command[list(command).index(name) + 1]


def _refresh_manifest_ref(f):
    launch = json.loads(f["selection"].read_text())
    launch["input_manifest"] = _ref(f["root"], f["manifest"])
    _write_json(f["selection"], launch)


class FakeRunner:
    def __init__(self, contract, fail_call=None, log_text="loss 1.0\n", corrupt_training_lineage=False):
        self.contract = contract; self.fail_call = fail_call; self.log_text = log_text; self.corrupt_training_lineage = corrupt_training_lineage; self.commands = []
    def __call__(self, command, cwd, log_path, deadline, monitor, interval, hang):
        self.commands.append(list(command)); monitor(); log_path.write_text(self.log_text, encoding="utf-8")
        if len(self.commands) == self.fail_call: return MODULE.StageResult(7, "dummy_failure")
        exe = Path(command[1]).name
        if exe == "train.py":
            seed = json.loads((cwd / _arg(command, "--config")).read_text())["seed"]
            run = Path(_arg(command, "--log-root")) / "rj45-v1" / f"run-{seed}"; run.mkdir(parents=True)
            seed_ref = self.contract.seed_configs[seed]; (run / "runner_config.json").write_bytes(seed_ref.path.read_bytes())
            lineage = {MODULE._relative(cwd, p): d for p, d in self.contract.frozen_inputs.items() if p.parent == cwd / "src/rby1_assembly" and p.suffix == ".py"}
            lineage.update({"scripts/train.py": self.contract.frozen_inputs[cwd / "scripts/train.py"], MODULE._relative(cwd, seed_ref.path): seed_ref.sha256, MODULE._relative(cwd, self.contract.training_config.path): self.contract.training_config.sha256})
            if self.corrupt_training_lineage: lineage["scripts/train.py"] = "0" * 64
            _write_json(run / "training_provenance.json", {"schema_version": 1, "seed": seed, "num_envs": 64, "checkpoint_scope": "checkpoints created in this run directory", "runtime_versions": {"rsl-rl-lib": "3.0.1"}, "source_and_config_sha256": lineage})
            hashes = {}
            for cp in self.contract.checkpoints:
                p = run / cp.filename; p.write_bytes(f"{seed}:{cp.iteration_index}".encode()); hashes[cp.filename] = _sha(p)
            _write_json(run / "checkpoint_manifest.json", {"schema_version": 1, "training_completed": True, "training_provenance_sha256": _sha(run / "training_provenance.json"), "checkpoints": hashes})
        elif exe == "inspect_rj45_checkpoint.py":
            cp = Path(_arg(command, "--checkpoint")); i = int(cp.stem.split("_")[1]); out = Path(_arg(command, "--output"))
            _write_json(out, {"schema_version": 2, "status": "checkpoint_payload_and_lineage_verified", "checkpoint": {"filename_iteration": i, "payload_iteration": i, "completed_updates": i + 1, "sha256": _sha(cp)}, "planned_training_updates": 200})
        elif exe == "evaluate_policy.py":
            cp = Path(command[2]); report = Path(_arg(command, "--report")); seed = json.loads((cwd / _arg(command, "--config")).read_text())["seed"]; run = cp.parent
            source = {MODULE._relative(cwd, p): d for p, d in self.contract.frozen_inputs.items() if p.parent == cwd / "src/rby1_assembly" and p.suffix == ".py"}; source["scripts/evaluate_policy.py"] = self.contract.frozen_inputs[cwd / "scripts/evaluate_policy.py"]
            _write_json(report, {"rj45_evaluation_schema_version": 2, "baseline_mode": "learned", "seed": self.contract.evaluation_seed, "requested_attempts": 32, "attempts": 32, "evaluation_num_envs": 16,
                "rj45_training_config": {"sha256": self.contract.training_config.sha256},
                "provenance": {"checkpoint_sha256": _sha(cp), "input_immutability_verified": True, "evaluation_inputs_reverified_at_end": True, "policy_config_sha256": self.contract.seed_configs[seed].sha256, "runtime_versions": {"rsl_rl": "3.0.1"}, "source_sha256": source, "verified_checkpoint_evidence": {"training_provenance_sha256": _sha(run / "training_provenance.json"), "checkpoint_manifest_sha256": _sha(run / "checkpoint_manifest.json")}}})
        return MODULE.StageResult(0)


class Tests(unittest.TestCase):
    def test_campaign_python_preserves_virtual_environment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            f = _fixture(root / "repo")
            environment = root / "venv"
            venv.create(environment, symlinks=True, with_pip=False)
            executable = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            contract = _contract(f, python=executable)
            prefix = subprocess.check_output(
                [str(contract.python), "-c", "import sys; print(sys.prefix)"], text=True
            ).strip()
            self.assertEqual(Path(prefix), environment)

    def test_two_stage_contract_and_commands(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f)
            self.assertEqual(c.candidate_selection.path, f["candidate"])
            self.assertEqual(c.guard_record.path, f["guard_record"])
            self.assertEqual(c.guard_deadline_utc, f["guard_deadline"])
            self.assertEqual(c.timer_target, "rby1-rj45-stop.timer")
            self.assertEqual(c.training_seeds, (20261101, 20261102, 20261103))
            self.assertIn("--zero-init-actor-output", MODULE.build_train_command(c, 20261101))
            self.assertNotIn("20261027", MODULE.build_evaluation_command(c, 20261101, Path("m.pt"), Path("r.json")))
            self.assertEqual(json.loads((f["root"] / "evidence/paired-audit.json").read_text())["schema_version"], 1)

    def test_campaign_nine_reports_and_partial_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f); runner = FakeRunner(c)
            fake_helper = types.SimpleNamespace(inspect_checkpoint=lambda **kw: json.loads(Path(kw["checkpoint_path"]).parents[3].joinpath("missing").read_text()))
            def modules(path, _name):
                if path.name == "inspect_rj45_checkpoint.py":
                    return types.SimpleNamespace(inspect_checkpoint=lambda **kw: json.loads((c.output_dir / "inspections" / f"seed-{json.loads((kw['runner_config_path']).read_text())['seed']}-{kw['checkpoint_path'].stem}.json").read_text()))
                return types.SimpleNamespace(_validate_report=lambda *args: {})
            with mock.patch.object(MODULE, "_load_local_module", side_effect=modules):
                reports = MODULE.execute_campaign(c, runner=runner, disk_free=lambda _p: 10**12)
            self.assertEqual(len(reports), 9); self.assertEqual(len(runner.commands), 21)
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f); runner = FakeRunner(c, fail_call=2)
            with self.assertRaisesRegex(MODULE.CampaignError, "return code 7"):
                MODULE.execute_campaign(c, runner=runner, disk_free=lambda _p: 10**12)
            self.assertEqual(json.loads((c.output_dir / "campaign-status.json").read_text())["status"], "failed")

    def test_deadline_disk_selection_and_label_swap_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); guard = f["guard_started"]; base = dict(project_root=f["root"], plan_path=f["plan"], expected_plan_sha256=_sha(f["plan"]), selection_path=f["selection"], expected_selection_sha256=_sha(f["selection"]), expected_instance_id="inst-123", expected_instance_name="rj45-long", expected_instance_organization="org-abc", output_dir=f["output"], python=Path(sys.executable), disk_free=lambda _p: 10**12)
            with self.assertRaisesRegex(MODULE.CampaignError, "already passed"):
                MODULE.load_contract(**base, deadline_utc=(guard - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), now=lambda: guard)
            with self.assertRaisesRegex(MODULE.CampaignError, "differs from the launch guard deadline"):
                MODULE.load_contract(**base, deadline_utc=(guard + timedelta(hours=9, seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), now=lambda: guard)
            candidate = json.loads(f["candidate"].read_text()); candidate["selected_profile"]["zero_init_actor_output"] = False; _write_json(f["candidate"], candidate)
            with self.assertRaisesRegex(MODULE.CampaignError, "candidate selection SHA-256 mismatch"):
                with mock.patch.object(MODULE, "_revalidate_pilot_report", side_effect=_raw_summary), mock.patch.object(MODULE, "_revalidate_pilot_bundle", return_value=None):
                    MODULE.load_contract(**base, deadline_utc=(guard + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"), now=lambda: guard)

    def test_candidate_label_cannot_be_swapped_onto_other_profile(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f)
            candidate = json.loads(f["candidate"].read_text())
            candidate["decision"]["selected_group"] = "A0"
            candidate["decision"]["selected_successes_all_attempts"] = 0
            _write_json(f["candidate"], candidate)
            candidate_ref = MODULE.FileRef(f["candidate"], _sha(f["candidate"]))
            frozen = dict(c.frozen_inputs); frozen[f["candidate"]] = candidate_ref.sha256
            with mock.patch.object(MODULE, "_revalidate_pilot_report", side_effect=_raw_summary), mock.patch.object(MODULE, "_revalidate_pilot_bundle", return_value=None):
                with self.assertRaisesRegex(MODULE.CampaignError, "profile settings.*group label"):
                    MODULE._validate_candidate_selection(project_root=f["root"], plan=json.loads(f["plan"].read_text()), plan_ref=c.plan, candidate_ref=candidate_ref, frozen_inputs=frozen)

    def test_selected_configs_must_be_pinned_by_candidate_and_launch(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f)
            candidate = json.loads(f["candidate"].read_text())
            candidate["inputs"]["immutable_inputs_sha256"].pop("configs/training.json")
            _write_json(f["candidate"], candidate)
            candidate_ref = MODULE.FileRef(f["candidate"], _sha(f["candidate"]))
            frozen = dict(c.frozen_inputs); frozen[f["candidate"]] = candidate_ref.sha256
            with mock.patch.object(MODULE, "_revalidate_pilot_report", side_effect=_raw_summary), mock.patch.object(MODULE, "_revalidate_pilot_bundle", return_value=None):
                with self.assertRaisesRegex(MODULE.CampaignError, "immutable inputs do not pin selected training"):
                    MODULE._validate_candidate_selection(project_root=f["root"], plan=json.loads(f["plan"].read_text()), plan_ref=c.plan, candidate_ref=candidate_ref, frozen_inputs=frozen)
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); launch = json.loads(f["selection"].read_text()); launch["selected_profile"]["zero_init_actor_output"] = False; _write_json(f["selection"], launch)
            with self.assertRaisesRegex(MODULE.CampaignError, "launch selected profile differs"):
                _contract(f)
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f); training = f["root"] / "configs/training.json"
            _write_json(training, {"schema_version": 2, "tampered": True})
            candidate = json.loads(f["candidate"].read_text())
            candidate["selected_profile"]["training_config"]["sha256"] = _sha(training)
            candidate["inputs"]["immutable_inputs_sha256"]["configs/training.json"] = _sha(training)
            _write_json(f["candidate"], candidate)
            candidate_ref = MODULE.FileRef(f["candidate"], _sha(f["candidate"]))
            frozen = dict(c.frozen_inputs); frozen[training] = _sha(training); frozen[f["candidate"]] = candidate_ref.sha256
            with mock.patch.object(MODULE, "_revalidate_pilot_report", side_effect=_raw_summary), mock.patch.object(MODULE, "_revalidate_pilot_bundle", return_value=None):
                with self.assertRaisesRegex(MODULE.CampaignError, "parent plan frozen input"):
                    MODULE._validate_candidate_selection(project_root=f["root"], plan=json.loads(f["plan"].read_text()), plan_ref=c.plan, candidate_ref=candidate_ref, frozen_inputs=frozen)

    def test_guard_window_cannot_reset_and_instance_identity_is_exact(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); guard = f["guard_started"]
            base = dict(project_root=f["root"], plan_path=f["plan"], expected_plan_sha256=_sha(f["plan"]), selection_path=f["selection"], expected_selection_sha256=_sha(f["selection"]), expected_instance_id="inst-123", expected_instance_name="rj45-long", expected_instance_organization="org-abc", output_dir=f["output"], python=Path(sys.executable), disk_free=lambda _p: 10**12)
            with self.assertRaisesRegex(MODULE.CampaignError, "differs from the launch guard deadline"):
                MODULE.load_contract(**base, deadline_utc=(guard + timedelta(hours=16)).strftime("%Y-%m-%dT%H:%M:%SZ"), now=lambda: guard + timedelta(hours=8))
            base["expected_instance_id"] = "other-instance"
            with self.assertRaisesRegex(MODULE.CampaignError, "identity differs"):
                MODULE.load_contract(**base, deadline_utc=(guard + timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ"), now=lambda: guard)

    def test_guard_record_manifest_identity_and_anchor_are_verified(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); manifest = json.loads(f["manifest"].read_text())
            manifest["inputs"].pop(str(f["guard_record"].relative_to(f["root"])))
            _write_json(f["manifest"], manifest); _refresh_manifest_ref(f)
            with self.assertRaisesRegex(MODULE.CampaignError, "does not pin the launch guard record"):
                _contract(f)
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); record = json.loads(f["guard_record"].read_text())
            record["brev_instance"]["id"] = "substituted-instance"; _write_json(f["guard_record"], record)
            manifest = json.loads(f["manifest"].read_text())
            manifest["inputs"][str(f["guard_record"].relative_to(f["root"]))] = _sha(f["guard_record"])
            _write_json(f["manifest"], manifest)
            launch = json.loads(f["selection"].read_text()); launch["guard_record"] = _ref(f["root"], f["guard_record"]); launch["input_manifest"] = _ref(f["root"], f["manifest"]); _write_json(f["selection"], launch)
            with self.assertRaisesRegex(MODULE.CampaignError, "guard record Brev identity differs"):
                _contract(f)
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); anchor = json.loads(f["anchor"].read_text()); anchor["timer_target"] = "other.timer"; _write_json(f["anchor"], anchor)
            with self.assertRaisesRegex(MODULE.CampaignError, "local guard anchor differs"):
                _contract(f)

    def test_nonfinite_json_oom_and_new_source_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"; p.write_text('{"x":1e400}')
            with self.assertRaisesRegex(MODULE.CampaignError, "non-finite"):
                MODULE._load_json(p, "bad")
            log = Path(d) / "oom.log"; log.write_text("CUDA out of memory")
            with self.assertRaisesRegex(MODULE.CampaignError, "out-of-memory"):
                MODULE._check_process_log(log)
            f = _fixture(Path(d) / "repo"); c = _contract(f); (f["root"] / "src/rby1_assembly/new.py").write_text("x=1")
            with self.assertRaisesRegex(MODULE.CampaignError, "path set changed"):
                MODULE._verify_frozen(c)

    def test_training_snapshot_provenance_tampering_stops_before_inspection(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f); runner = FakeRunner(c, corrupt_training_lineage=True)
            with self.assertRaisesRegex(MODULE.CampaignError, "training lineage differs"):
                MODULE.execute_campaign(c, runner=runner, disk_free=lambda _p: 10**12)
            self.assertEqual(len(runner.commands), 1)

    def test_subprocess_periodic_monitor_stopped_hang_and_descendant_cleanup(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); calls = []
            marker = root / "monitor-called-twice"
            def monitor():
                calls.append(1)
                if len(calls) == 2:
                    marker.write_text("ready\n", encoding="utf-8")
            code = (
                "import pathlib,time; "
                f"marker=pathlib.Path({str(marker)!r}); "
                "deadline=time.monotonic()+3.0; "
                "\nwhile not marker.exists():\n"
                "    if time.monotonic() >= deadline: raise SystemExit(3)\n"
                "    time.sleep(0.005)\n"
            )
            result = MODULE.run_subprocess([sys.executable, "-c", code], root, root / "quiet.log", datetime.now(timezone.utc)+timedelta(seconds=5), monitor, .01, .03)
            self.assertEqual(result.return_code, 0); self.assertGreaterEqual(len(calls), 2)
            def disk_failure():
                raise MODULE.CampaignError("disk below minimum")
            result = MODULE.run_subprocess([sys.executable, "-c", "import time; time.sleep(10)"], root, root / "disk.log", datetime.now(timezone.utc)+timedelta(seconds=2), disk_failure, .01, .5)
            self.assertEqual(result.failure_reason, "monitor_failed: disk below minimum")
            result = MODULE.run_subprocess([sys.executable, "-c", "import os,signal; os.kill(os.getpid(), signal.SIGSTOP)"], root, root / "hang.log", datetime.now(timezone.utc)+timedelta(seconds=2), lambda: None, .01, .04)
            self.assertEqual(result.failure_reason, "confirmed_stopped_process_group_hang")
            child_file = root / "child.pid"
            code = f"import subprocess,sys; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']); open({str(child_file)!r},'w').write(str(p.pid)); sys.exit(7)"
            result = MODULE.run_subprocess([sys.executable, "-c", code], root, root / "child.log", datetime.now(timezone.utc)+timedelta(seconds=2), lambda: None, .01, .5)
            self.assertNotEqual(result.return_code, 0)
            child_pid = int(child_file.read_text())
            state_path = Path("/proc") / str(child_pid) / "stat"
            if state_path.exists():
                state = state_path.read_text()[state_path.read_text().rfind(")") + 2]
                self.assertEqual(state, "Z")

    def test_run_subprocess_allows_four_real_worker_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def launch(index):
                return MODULE.run_subprocess(
                    [sys.executable, "-c", "import time; time.sleep(0.05)"],
                    root,
                    root / f"worker-{index}.log",
                    datetime.now(timezone.utc) + timedelta(seconds=5),
                    lambda: None,
                    0.01,
                    1.0,
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(launch, range(4)))
            self.assertEqual([result.return_code for result in results], [0, 0, 0, 0])

    def test_live_oom_log_stops_stage_before_process_exit(self):
        for index, (token, error) in enumerate((("CUDA out of memory", "out-of-memory evidence"), ("loss nan", "non-finite numeric token"))):
            with self.subTest(token=token), tempfile.TemporaryDirectory() as d:
                f = _fixture(Path(d)); c = _contract(f)
                c.output_dir.mkdir(); (c.output_dir / "logs").mkdir()
                started = time.monotonic()
                with self.assertRaisesRegex(MODULE.CampaignError, error):
                    MODULE._run_stage(c, stage=f"live-log-{index}", command=[sys.executable, "-c", f"import time; print({token!r}, flush=True); time.sleep(10)"], runner=MODULE.run_subprocess, disk_free=lambda _p: 10**12)
                self.assertLess(time.monotonic() - started, 2.0)

    def test_baseexception_cleanup_and_startup_failure_preserve_original(self):
        class FakeProcess:
            pid = 43210
            def poll(self): raise KeyboardInterrupt()
            def wait(self, timeout): return -signal.SIGTERM
        with tempfile.TemporaryDirectory() as d, mock.patch.object(MODULE.subprocess, "Popen", return_value=FakeProcess()), mock.patch.object(MODULE, "_process_group_snapshot", return_value=(0,0,("S",))), mock.patch.object(MODULE, "_terminate_process_group") as cleanup:
            with self.assertRaises(KeyboardInterrupt):
                MODULE.run_subprocess(["x"], Path(d), Path(d)/"log", datetime.now(timezone.utc)+timedelta(seconds=1), lambda: None, .01, .1)
            cleanup.assert_called_once_with(43210)
        class WaitErrorProcess(FakeProcess):
            def poll(self): raise OSError("wait failed")
        with tempfile.TemporaryDirectory() as d, mock.patch.object(MODULE.subprocess, "Popen", return_value=WaitErrorProcess()), mock.patch.object(MODULE, "_process_group_snapshot", return_value=(0,0,("S",))), mock.patch.object(MODULE, "_terminate_process_group"):
            with self.assertRaisesRegex(OSError, "wait failed"):
                MODULE.run_subprocess(["x"], Path(d), Path(d)/"wait-error.log", datetime.now(timezone.utc)+timedelta(seconds=1), lambda: None, .01, .1)
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f)
            with mock.patch.object(MODULE, "_atomic_json", side_effect=KeyboardInterrupt("stop")):
                with self.assertRaisesRegex(KeyboardInterrupt, "stop"):
                    MODULE.execute_campaign(c)
            self.assertTrue(c.output_dir.is_dir())

    def test_minimal_fake_inspection_and_evaluation_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            f = _fixture(Path(d)); c = _contract(f); cp = Path(d)/"model_19.pt"; cp.write_bytes(b"x"); inspection = Path(d)/"i.json"
            _write_json(inspection, {"schema_version":2,"status":"checkpoint_payload_and_lineage_verified","checkpoint":{"filename_iteration":19,"payload_iteration":19,"completed_updates":20,"sha256":_sha(cp)},"planned_training_updates":200})
            run = Path(d)/"run"; run.mkdir(); (run/"checkpoint_manifest.json").write_text("{}"); (run/"training_provenance.json").write_text("{}"); (run/"runner_config.json").write_text("{}")
            with mock.patch.object(MODULE, "_load_local_module", return_value=types.SimpleNamespace(inspect_checkpoint=lambda **kw: {"different": True})):
                with self.assertRaisesRegex(MODULE.CampaignError, "differs from helper recomputation"):
                    MODULE._validate_inspection(c, run, inspection, c.checkpoints[0], cp)
            report = Path(d)/"report.json"; _write_json(report, {"rj45_evaluation_schema_version":2,"baseline_mode":"learned","seed":c.evaluation_seed,"requested_attempts":32,"attempts":32,"evaluation_num_envs":16,"provenance":{"checkpoint_sha256":_sha(cp)}})
            with self.assertRaises(MODULE.CampaignError):
                MODULE._validate_evaluation_report(report, c, 20261101, run, cp)

    def test_frozen_v1_analyzer_links_raw_report_config_to_training_bundle(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); analyzer = root / "frozen_v1.py"
            analyzer.write_text(
                "import json\n"
                "def _verify_training_bundle(label, report, *, checkpoint_path, manifest_path, training_provenance_path, runner_config_path):\n"
                "    provenance=json.loads(training_provenance_path.read_text())\n"
                "    if report['training_config_sha256'] != provenance['training_config_sha256']:\n"
                "        raise ValueError('training config lineage mismatch')\n"
                "    return {'status':'checkpoint_manifest_provenance_runner_bytes_verified','training_updates':20}\n",
                encoding="utf-8",
            )
            report = root / "report.json"; _write_json(report, {"training_config_sha256": "a" * 64})
            files = {}
            for name in ("checkpoint", "checkpoint_manifest", "runner_config"):
                path = root / name; path.write_text(name); files[name] = MODULE.FileRef(path, _sha(path))
            provenance = root / "training_provenance"; _write_json(provenance, {"training_config_sha256": "a" * 64}); files["training_provenance"] = MODULE.FileRef(provenance, _sha(provenance))
            MODULE._revalidate_pilot_bundle(root, files, MODULE.FileRef(report, _sha(report)), "A0", analyzer)
            _write_json(provenance, {"training_config_sha256": "b" * 64})
            with self.assertRaisesRegex(MODULE.CampaignError, "training config lineage mismatch"):
                MODULE._revalidate_pilot_bundle(root, files, MODULE.FileRef(report, _sha(report)), "A0", analyzer)


if __name__ == "__main__": unittest.main()
