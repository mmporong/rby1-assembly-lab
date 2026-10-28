from __future__ import annotations

from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_rj45_refinement_campaign", ROOT / "scripts/run_rj45_refinement_campaign.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RefinementCampaignTests(unittest.TestCase):
    def _refresh_inputs(self, root: Path, inputs: Path) -> None:
        frozen = json.loads(inputs.read_text(encoding="utf-8"))
        for relative in list(frozen):
            frozen[relative] = _sha(root / relative)
        _write_json(inputs, frozen)

    def _project(self, parent: Path):
        root = parent / "project"
        scripts = root / "scripts"
        configs = root / "configs"
        evidence = root / "evidence"
        sources = root / "src/rby1_assembly"
        generated = root / "assets/generated/tasks/rj45-v1"
        scripts.mkdir(parents=True)
        configs.mkdir()
        evidence.mkdir()
        sources.mkdir(parents=True)
        generated.mkdir(parents=True)
        for name in (
            "train.py",
            "evaluate_policy.py",
            "inspect_rj45_checkpoint.py",
            "analyze_rj45_paired_v2.py",
            "run_rj45_refinement_campaign.py",
            "run_rj45_long_campaign.py",
        ):
            (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
        (sources / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
        for name in ("runtime.json", "assets.json", "rj45-assets.json"):
            _write_json(configs / name, {"name": name})
        shutil.copy2(ROOT / "configs/ppo-rj45-low-exploration-v1.json", configs / "ppo-rj45-low-exploration-v1.json")
        shutil.copy2(ROOT / "configs/rj45-training-target-offset-v1.json", configs / "rj45-training-target-offset-v1.json")
        for name in ("robot.usda", "socket.usda", "plug.usda", "table.usda", "scene.usda"):
            (generated / name).write_text(f"#{name}\n", encoding="utf-8")
        _write_json(generated / "build-report.json", {"files": {name: _sha(generated / name) for name in ("robot.usda", "socket.usda", "plug.usda", "table.usda", "scene.usda")}})
        _write_json(configs / "tasks/rj45-v1.json", {"assets": {"robot": "assets/generated/tasks/rj45-v1/robot.usda", "socket": "assets/generated/tasks/rj45-v1/socket.usda", "plug": "assets/generated/tasks/rj45-v1/plug.usda", "table": "assets/generated/tasks/rj45-v1/table.usda", "build_report": "assets/generated/tasks/rj45-v1/build-report.json"}})
        (evidence / "nominal.jsonl").write_text('{}\n', encoding="utf-8")
        _write_json(evidence / "nominal.json", {"trace_path": "evidence/nominal.jsonl"})
        _write_json(evidence / "held.json", {"provenance": {"nominal_report_path": "evidence/nominal.json", "nominal_trace_path": "evidence/nominal.jsonl"}})
        for name in ("contact-report.json", "contact-trace.jsonl", "contact-sat.json"):
            (evidence / name).write_text('{}\n', encoding="utf-8")
        _write_json(evidence / "contact.json", {"report_path": "evidence/contact-report.json", "trace_path": "evidence/contact-trace.jsonl", "sat_report_path": "evidence/contact-sat.json"})
        profiles = []
        canonical_ppo = json.loads((configs / "ppo-rj45-low-exploration-v1.json").read_text())
        canonical_training = json.loads((configs / "rj45-training-target-offset-v1.json").read_text())
        for label in MODULE.PROFILE_LABELS:
            ppo = configs / f"ppo-{label}.json"
            training = configs / f"training-{label}.json"
            schedule = "adaptive" if label == "Z1" else "fixed"
            noise = 0.1 if label in {"Z1", "F"} else 0.02
            ppo_payload = json.loads(json.dumps(canonical_ppo)); ppo_payload.update({"seed": 41, "max_iterations": 96, "num_steps_per_env": 128, "save_interval": 19}); ppo_payload["policy"]["init_noise_std"] = noise; ppo_payload["algorithm"]["schedule"] = schedule
            _write_json(ppo, ppo_payload)
            lateral = 0.00005 if label == "R" else 0.0005
            training_payload = json.loads(json.dumps(canonical_training)); training_payload["reward"]["lateral_scale_m"] = lateral
            _write_json(training, training_payload)
            profiles.append(
                {
                    "label": label,
                    "ppo_config": str(ppo.relative_to(root)),
                    "training_config": str(training.relative_to(root)),
                    "zero_init_actor_output": True,
                }
            )
        created = datetime.now(timezone.utc).replace(microsecond=0)
        plan = root / "plan.json"
        _write_json(
            plan,
            {
                "created_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "guard_started_utc": created.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "guard_deadline_utc": (created + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "budget": {"max_hours": 6.0},
                "execution": {"mode": "sequential", "profile_devices": {label: "cuda:0" for label in MODULE.PROFILE_LABELS}},
                "training": {"seed": 41, "num_envs": 64, "updates": 96, "num_steps_per_env": 128, "save_interval": 19, "checkpoints": [{"filename": "model_19.pt", "iteration_index": 19, "completed_updates": 20}, {"filename": "model_95.pt", "iteration_index": 95, "completed_updates": 96}]},
                "evaluation": {"seed": 51, "num_envs": 32, "attempts": 32, "final_seed_sealed": True, "forbidden_seeds": [20261027]},
                "profiles": profiles,
                "nominal_evidence": "evidence/nominal.json",
                "held_state": "evidence/held.json",
                "parallel_contact_evidence": "evidence/contact.json",
            },
        )
        pinned = [plan, *root.rglob("*.json"), *root.rglob("*.jsonl"), *root.rglob("*.py"), *root.rglob("*.usda")]
        inputs = root / "inputs.json"
        _write_json(inputs, {str(path.relative_to(root)): _sha(path) for path in pinned})
        python = root / "base-python"
        python.write_text("python\n", encoding="utf-8")
        venv_python = root / "venv-python"
        venv_python.symlink_to(python)
        return root, plan, inputs, venv_python, created

    def test_contract_preserves_venv_symlink_and_builds_four_profile_commands(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            contract = MODULE.load_contract(
                root,
                plan,
                _sha(plan),
                inputs,
                _sha(inputs),
                python,
                (created + timedelta(hours=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                root / "output",
                now=created,
                disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES,
            )
            self.assertEqual(contract.python, python.absolute())
            self.assertTrue(contract.python.is_symlink())
            self.assertEqual(tuple(profile.label for profile in contract.profiles), MODULE.PROFILE_LABELS)
            baseline = MODULE.build_baseline_command(contract, root / "baseline.json")
            self.assertIn("biased_zero", baseline)
            self.assertNotIn("model_19.pt", baseline)
            for profile in contract.profiles:
                command = MODULE.build_train_command(contract, profile)
                self.assertEqual(command[0], str(python.absolute()))
                self.assertIn(str(root / "output" / "runs" / profile.label), command)
                self.assertEqual("--zero-init-actor-output" in command, profile.zero_init_actor_output)

    def test_contract_rejects_open_final_seed_and_deadline_over_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["evaluation"]["final_seed_sealed"] = False
            _write_json(plan, payload)
            frozen = json.loads(inputs.read_text(encoding="utf-8"))
            frozen["plan.json"] = _sha(plan)
            _write_json(inputs, frozen)
            with self.assertRaisesRegex(MODULE.CampaignError, "final_seed_sealed"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            payload["evaluation"]["final_seed_sealed"] = True
            _write_json(plan, payload)
            frozen["plan.json"] = _sha(plan)
            _write_json(inputs, frozen)
            with self.assertRaisesRegex(MODULE.CampaignError, "campaign budget"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

    def test_profile_contrasts_devices_and_time_order_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["execution"] = {"mode": "parallel_profiles", "profile_devices": {label: f"cuda:{index}" for index, label in enumerate(MODULE.PROFILE_LABELS)}, "cpu_threads_per_profile": 3}
            _write_json(plan, payload); self._refresh_inputs(root, inputs)
            contract = MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            for index, profile in enumerate(contract.profiles):
                command = MODULE.build_train_command(contract, profile)
                self.assertEqual(command[command.index("--device") + 1], f"cuda:{index}")
                self.assertIn("OMP_NUM_THREADS=3", command)
                self.assertNotIn("CUDA_VISIBLE_DEVICES", " ".join(command))

            f_config = root / "configs/ppo-F.json"
            broken = json.loads(f_config.read_text()); broken["algorithm"]["learning_rate"] = 0.0002
            _write_json(f_config, broken); self._refresh_inputs(root, inputs)
            with self.assertRaisesRegex(MODULE.CampaignError, "F PPO must differ"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            payload = json.loads(plan.read_text()); payload["guard_started_utc"] = (created + timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
            _write_json(plan, payload); self._refresh_inputs(root, inputs)
            with self.assertRaisesRegex(MODULE.CampaignError, "time ordering"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

    def test_identical_profile_configs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            (root / "configs/ppo-F.json").write_bytes((root / "configs/ppo-Z1.json").read_bytes())
            self._refresh_inputs(root, inputs)
            with self.assertRaisesRegex(MODULE.CampaignError, "F PPO must differ"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

    def test_common_ppo_mutation_and_canonical_source_mutation_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            for label in MODULE.PROFILE_LABELS:
                path = root / f"configs/ppo-{label}.json"
                payload = json.loads(path.read_text()); payload["algorithm"]["entropy_coef"] = 99
                _write_json(path, payload)
            self._refresh_inputs(root, inputs)
            with self.assertRaisesRegex(MODULE.CampaignError, "Z1 PPO must equal"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            anchor = root / "configs/ppo-rj45-low-exploration-v1.json"
            payload = json.loads(anchor.read_text()); payload["algorithm"]["entropy_coef"] = 99
            _write_json(anchor, payload); self._refresh_inputs(root, inputs)
            with self.assertRaisesRegex(MODULE.CampaignError, "canonical low-exploration PPO SHA-256 mismatch"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

    def test_plan_created_after_guard_start_is_valid_when_both_precede_now(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, guard_started = self._project(Path(directory))
            created = guard_started + timedelta(minutes=5)
            current = created + timedelta(minutes=1)
            payload = json.loads(plan.read_text()); payload["created_utc"] = created.strftime("%Y-%m-%dT%H:%M:%SZ")
            _write_json(plan, payload); self._refresh_inputs(root, inputs)
            contract = MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=current, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            self.assertEqual(contract.deadline_utc, created + timedelta(hours=1))

    def test_contract_rejects_forbidden_seed_changed_dependency_and_malformed_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            payload = json.loads(plan.read_text(encoding="utf-8"))
            payload["evaluation"]["seed"] = 20261027
            _write_json(plan, payload)
            frozen = json.loads(inputs.read_text(encoding="utf-8")); frozen["plan.json"] = _sha(plan); _write_json(inputs, frozen)
            with self.assertRaisesRegex(MODULE.CampaignError, "forbidden seeds"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            payload["evaluation"]["seed"] = 51
            _write_json(plan, payload); frozen["plan.json"] = _sha(plan); _write_json(inputs, frozen)
            with self.assertRaisesRegex(MODULE.CampaignError, "YYYY-MM-DD"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, "tomorrow", root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            (root / "evidence/nominal.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
            with self.assertRaisesRegex(MODULE.CampaignError, "do not pin current bytes"):
                MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)

    def test_json_loader_rejects_duplicate_keys_and_overflow(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text('{"x":1,"x":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(MODULE.CampaignError, "duplicate JSON key"):
                MODULE._load_json(path, "bad")
            path.write_text('{"x":1e309}\n', encoding="utf-8")
            with self.assertRaisesRegex(MODULE.CampaignError, "non-finite numeric"):
                MODULE._load_json(path, "bad")

    def test_checkpoint_inspection_rejects_payload_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model_19.pt"; checkpoint.write_bytes(b"checkpoint")
            inspection = root / "inspection.json"
            _write_json(inspection, {"status": "checkpoint_payload_and_lineage_verified", "checkpoint": {"completed_updates": 20, "sha256": "0" * 64}})
            with self.assertRaisesRegex(MODULE.CampaignError, "inspection failed"):
                MODULE._validate_inspection(inspection, checkpoint, 20, "Z1")

    def test_stage_runner_is_injectable_and_rechecks_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            payload = json.loads(plan.read_text()); payload["execution"] = {"mode": "parallel_profiles", "profile_devices": {label: f"cuda:{index}" for index, label in enumerate(MODULE.PROFILE_LABELS)}}
            _write_json(plan, payload); self._refresh_inputs(root, inputs)
            contract = MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            contract.output_dir.mkdir()
            (contract.output_dir / "logs").mkdir()

            calls = []
            def good_runner(command, cwd, log, deadline, monitor, interval, hang):
                calls.append((list(command), cwd, deadline, interval, hang))
                log.write_text("completed\n", encoding="utf-8")
                monitor()
                return MODULE.StageResult(0)

            MODULE._run_stage(contract, "mock", ["demo"], good_runner, lambda _path: MODULE.MINIMUM_FREE_BYTES)
            self.assertEqual(calls[0][0], ["demo"])
            self.assertTrue((contract.output_dir / "events.jsonl").is_file())

            mutable = next(iter(contract.frozen_inputs))
            def mutating_runner(command, cwd, log, deadline, monitor, interval, hang):
                log.write_text("completed\n", encoding="utf-8")
                mutable.write_text("changed\n", encoding="utf-8")
                return MODULE.StageResult(0)

            with self.assertRaisesRegex(MODULE.CampaignError, "frozen .* changed"):
                MODULE._run_stage(contract, "mutating", ["demo"], mutating_runner, lambda _path: MODULE.MINIMUM_FREE_BYTES)

    def test_parallel_stop_signal_reaches_and_cleans_all_owned_runners(self):
        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            contract = MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            contract.output_dir.mkdir(); (contract.output_dir / "logs").mkdir()
            stop = threading.Event(); write_lock = threading.Lock(); barrier = threading.Barrier(4); stopped = threading.Barrier(4)
            cleaned: list[str] = []

            def runner(command, cwd, log, deadline, monitor, interval, hang):
                label = command[-1]; log.write_text("running\n", encoding="utf-8")
                try:
                    barrier.wait(timeout=2)
                    if label == "F":
                        stop.set()
                    stopped.wait(timeout=2)
                    if label == "F":
                        raise MODULE.CampaignError("profile F failed")
                    monitor()
                    return MODULE.StageResult(0)
                finally:
                    with write_lock:
                        cleaned.append(label)

            def launch(label):
                try:
                    MODULE._run_stage(contract, f"parallel-{label}", ["profile", label], runner, lambda _path: MODULE.MINIMUM_FREE_BYTES, stop, write_lock)
                except MODULE.CampaignError:
                    return "failed"
                return "completed"

            with ThreadPoolExecutor(max_workers=4) as pool:
                outcomes = list(pool.map(launch, MODULE.PROFILE_LABELS))
            self.assertEqual(sorted(cleaned), sorted(MODULE.PROFILE_LABELS))
            self.assertEqual(outcomes, ["failed"] * 4)

    def test_main_sigterm_relay_cleans_four_real_process_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            child_dir = root / "children"
            child_dir.mkdir()
            script = f"""
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

sys.path.insert(0, {str(ROOT / 'scripts')!r})
import run_rj45_refinement_campaign as refinement
from run_rj45_long_campaign import run_subprocess

stop_event = threading.Event()
original_sigint = signal.getsignal(signal.SIGINT)
original_sigterm = signal.getsignal(signal.SIGTERM)
previous, received = refinement._install_parallel_signal_relay(stop_event)

child_dir = Path({str(child_dir)!r})
def launch(index):
    child_pid = child_dir / f"{{index}}.pid"
    code = "import os,pathlib,time; pathlib.Path(" + repr(str(child_pid)) + ").write_text(str(os.getpid())); time.sleep(30)"
    return run_subprocess(
        [sys.executable, "-c", code], Path({str(root)!r}),
        Path({str(root)!r}) / f"worker-{{index}}.log",
        datetime.now(timezone.utc) + timedelta(seconds=20),
        lambda: (_ for _ in ()).throw(RuntimeError("stop requested")) if stop_event.is_set() else None,
        0.01, 5.0,
    )

with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(launch, index) for index in range(4)]
    deadline = time.monotonic() + 5
    while len(list(child_dir.glob("*.pid"))) != 4:
        if time.monotonic() >= deadline:
            raise RuntimeError("children did not start")
        time.sleep(0.01)
    os.kill(os.getpid(), signal.SIGTERM)
    results = [future.result() for future in futures]
refinement._restore_signal_handlers(previous)
Path({str(result_path)!r}).write_text(json.dumps({{
    "received": received,
    "return_codes": [result.return_code for result in results],
    "failure_reasons": [result.failure_reason for result in results],
    "handlers_restored": (
        signal.getsignal(signal.SIGINT) is original_sigint
        and signal.getsignal(signal.SIGTERM) is original_sigterm
    ),
}}))
"""
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=ROOT,
                text=True,
                capture_output=True,
                timeout=10,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            evidence = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(evidence["received"], [signal.SIGTERM])
            self.assertEqual(evidence["return_codes"], [125, 125, 125, 125])
            self.assertTrue(all("stop requested" in reason for reason in evidence["failure_reasons"]))
            self.assertTrue(evidence["handlers_restored"])
            for pid_path in child_dir.glob("*.pid"):
                pid = int(pid_path.read_text(encoding="utf-8"))
                self.assertFalse((Path("/proc") / str(pid)).exists(), f"child {pid} remained")

    def test_report_gate_allows_valid_failure_but_rejects_invalid_or_contact(self):
        class Analyzer:
            @staticmethod
            def _validate_report(report, label, attempts):
                return {str(index): row for index, row in enumerate(report["rows"])}

        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            rows = [{"valid": True, "success": False, "forbidden_contact_load_peak_n": 0.0} for _ in range(2)]
            _write_json(report, {"rows": rows})
            _payload, indexed = MODULE._validate_report(Analyzer, report, "R", 2)
            self.assertFalse(any(row["success"] for row in indexed.values()))
            rows[0]["valid"] = False
            _write_json(report, {"rows": rows})
            with self.assertRaisesRegex(MODULE.CampaignError, "invalid/safety gate"):
                MODULE._validate_report(Analyzer, report, "R", 2)

    def test_mock_campaign_completes_all_stages_and_stops_after_first_failure(self):
        class Analyzer:
            @staticmethod
            def _validate_report(report, label, attempts):
                return {row["id"]: row for row in report["rows"]}
            @staticmethod
            def analyze_reports(baseline, conditions, **kwargs):
                label = next(iter(conditions)); return {"results": {label: {"successes": 0}}}
            @staticmethod
            def _verify_training_bundle(label, report, **kwargs):
                updates = 20 if kwargs["checkpoint_path"].name == "model_19.pt" else 96
                return {"status": "checkpoint_manifest_provenance_runner_bytes_verified", "checkpoint_completed_updates": updates}

        def prepare_stage(contract, calls, fail_at=None):
            def fake_stage(_contract, stage, command, runner, disk_free, *args):
                calls.append(stage)
                if stage == fail_at:
                    raise MODULE.CampaignError("injected stage failure")
                if stage == "evaluate-baseline" or stage.startswith("evaluate-"):
                    report = Path(command[command.index("--report") + 1])
                    _write_json(report, {"rows": [{"id": "same", "valid": True, "success": False, "forbidden_contact_load_peak_n": 0.0}]})
                elif stage.startswith("train-"):
                    label = stage.removeprefix("train-")
                    run = contract.output_dir / "runs" / label / "rj45-v1" / "one"
                    run.mkdir(parents=True)
                    for filename in ("model_19.pt", "model_95.pt"):
                        (run / filename).write_bytes(filename.encode())
                    for filename in ("checkpoint_manifest.json", "training_provenance.json", "runner_config.json"):
                        _write_json(run / filename, {})
                elif stage.startswith("inspect-"):
                    checkpoint = Path(command[command.index("--checkpoint") + 1])
                    output = Path(command[command.index("--output") + 1])
                    updates = 20 if checkpoint.name == "model_19.pt" else 96
                    _write_json(output, {"status": "checkpoint_payload_and_lineage_verified", "checkpoint": {"completed_updates": updates, "sha256": _sha(checkpoint)}})
            return fake_stage

        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            payload = json.loads(plan.read_text()); payload["execution"] = {"mode": "parallel_profiles", "profile_devices": {label: f"cuda:{index}" for index, label in enumerate(MODULE.PROFILE_LABELS)}}
            _write_json(plan, payload); self._refresh_inputs(root, inputs)
            contract = MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            calls = []
            with mock.patch.object(MODULE, "_load_module", return_value=Analyzer), mock.patch.object(MODULE, "_run_stage", side_effect=prepare_stage(contract, calls)):
                summary = MODULE.execute_campaign(contract, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(len(calls), 21)
            self.assertEqual(calls[0], "evaluate-baseline")
            self.assertEqual({stage for stage in calls if stage.startswith("train-")}, {f"train-{label}" for label in MODULE.PROFILE_LABELS})
            self.assertIn("evaluate-R-iter-95", calls)

        with tempfile.TemporaryDirectory() as directory:
            root, plan, inputs, python, created = self._project(Path(directory))
            contract = MODULE.load_contract(root, plan, _sha(plan), inputs, _sha(inputs), python, (created + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"), root / "output", now=created, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            calls = []
            with mock.patch.object(MODULE, "_load_module", return_value=Analyzer), mock.patch.object(MODULE, "_run_stage", side_effect=prepare_stage(contract, calls, "train-F")):
                with self.assertRaisesRegex(MODULE.CampaignError, "injected"):
                    MODULE.execute_campaign(contract, disk_free=lambda _path: MODULE.MINIMUM_FREE_BYTES)
            self.assertEqual(calls[-1], "train-F")
            self.assertFalse(any(stage.startswith("train-N") or stage.startswith("train-R") for stage in calls))
            status = json.loads((contract.output_dir / "campaign-status.json").read_text())
            self.assertEqual(status["status"], "failed")


if __name__ == "__main__":
    unittest.main()
