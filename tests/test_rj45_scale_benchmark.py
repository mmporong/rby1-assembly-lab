from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("summarize_rj45_scale_benchmark", ROOT / "scripts/summarize_rj45_scale_benchmark.py")
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC); sys.modules[SPEC.name] = MODULE; SPEC.loader.exec_module(MODULE)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref(root: Path, path: Path) -> dict[str, str]:
    return {"path": str(path.relative_to(root)), "sha256": sha(path)}


class ScaleBenchmarkTests(unittest.TestCase):
    def make_case(self, parent: Path):
        root = parent / "project"; (root / "configs").mkdir(parents=True)
        shutil.copy2(ROOT / "configs/ppo-rj45-low-exploration-v1.json", root / "configs/ppo-rj45-low-exploration-v1.json")
        protocol = root / "protocol.json"
        write_json(protocol, {"scale_benchmark": {"candidate_num_envs": [64, 128, 256], "steps_per_env": [128, 64, 32], "transitions_per_update": 8192, "warmup_updates": 3, "measured_updates": 5, "seed": 20261110, "profile": "Z1"}})
        proof = root / "proof.json"; write_json(proof, {"passed": True})
        training_script = root / "train.py"; training_script.write_text("# train\n")
        nominal = root / "nominal.json"; write_json(nominal, {"proof": "nominal"})
        held = root / "held.json"; write_json(held, {"proof": "held"})
        contact = root / "contact.json"; write_json(contact, {"proof": "contact"})
        candidates = []
        for index, (num_envs, steps) in enumerate(MODULE.CANDIDATES.items()):
            run = root / f"run-{num_envs}"; run.mkdir()
            event = run / "events.out.tfevents.mock"; event.write_bytes(f"event-{num_envs}".encode())
            checkpoint = run / "model_7.pt"; checkpoint.write_bytes(f"checkpoint-{num_envs}".encode())
            inspection = run / "inspection.json"
            write_json(inspection, {"status": "checkpoint_payload_and_lineage_verified", "planned_training_updates": 8, "loader": {"map_location": "cpu", "weights_only": True}, "checkpoint": {"filename_iteration": 7, "payload_iteration": 7, "completed_updates": 8, "sha256": sha(checkpoint)}})
            runner = json.loads((root / "configs/ppo-rj45-low-exploration-v1.json").read_text())
            runner.update({"seed": 20261110, "num_steps_per_env": steps, "max_iterations": 8, "save_interval": 7})
            runner_path = run / "runner.json"; write_json(runner_path, runner)
            manifest = run / "manifest.json"; write_json(manifest, {"checkpoint": sha(checkpoint)})
            provenance = run / "provenance.json"; write_json(provenance, {"seed": 20261110})
            raw_log = run / "raw.log"; raw_log.write_text("benchmark completed\n")
            candidates.append({"num_envs": num_envs, "num_steps_per_env": steps, "transitions_per_update": 8192, "run_dir": str(run.relative_to(root)), "gpu_device": f"cuda:{index}", "exit_code": 0, "event_file": ref(root, event), "checkpoint": ref(root, checkpoint), "checkpoint_inspection": ref(root, inspection), "checkpoint_manifest": ref(root, manifest), "training_provenance": ref(root, provenance), "training_script": ref(root, training_script), "runner_config": ref(root, runner_path), "nominal_evidence": ref(root, nominal), "held_state": ref(root, held), "parallel_contact_evidence": ref(root, contact), "raw_log": ref(root, raw_log), "validity": {"valid": True, "parallel_contact_verified": True, "no_nonfinite_or_gpu_memory_errors": True, "finite_checkpoint_and_optimizer_state": True, "physical_proof_refs": [ref(root, proof)]}})
        spec = root / "spec.json"; write_json(spec, {"candidates": candidates})
        return root, protocol, spec

    @staticmethod
    def loader_for(times):
        def loader(path, tag):
            values = times[int(path.parent.name.split("-")[-1])]
            factor = 0.6 if tag == MODULE.COLLECTION_TAG else 0.4
            return [(step, value * factor) for step, value in enumerate(values)]
        return loader

    @staticmethod
    def checkpoint_verifier(checkpoint, manifest, provenance, runner, training_script):
        result = json.loads((checkpoint.parent / "inspection.json").read_text())
        result["finite_tensor_audit"] = {"status": "all_finite", "tensor_counts": {"model_state_dict": 1, "optimizer_state_dict": 1}}
        return result

    @staticmethod
    def evidence_verifier(root, nominal, held, contact):
        return {"nominal": {"passed": True}, "held": {"passed": True}, "parallel_contact": {"passed": True}}

    def summarize(self, root, protocol, spec, loader):
        return MODULE.summarize(root, protocol, sha(protocol), spec, sha(spec), scalar_loader=loader, checkpoint_verifier=self.checkpoint_verifier, evidence_verifier=self.evidence_verifier)

    def test_selects_smallest_candidate_within_ninety_percent(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory))
            times = {64: [1.0] * 8, 128: [0.92] * 8, 256: [0.90] * 8}
            result = self.summarize(root, protocol, spec, self.loader_for(times))
            self.assertEqual(result["selected_topology"], {"num_envs": 64, "num_steps_per_env": 128, "transitions_per_update": 8192, "selection_rule": "smallest_valid_candidate_at_least_90_percent_of_highest_throughput"})
            self.assertEqual([row["step"] for row in result["candidates"]["64"]["measured_timings"]], [3, 4, 5, 6, 7])
            self.assertEqual(len(result["result_content_sha256"]), 64)

    def test_missing_duplicate_and_nonfinite_steps_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory))
            base = {64: [1.0] * 8, 128: [1.0] * 8, 256: [1.0] * 8}
            def missing(path, tag):
                rows = self.loader_for(base)(path, tag); return rows[:-1] if path.parent.name == "run-64" and tag == MODULE.COLLECTION_TAG else rows
            with self.assertRaisesRegex(MODULE.BenchmarkError, "exactly updates 0..7"):
                self.summarize(root, protocol, spec, missing)
            def duplicate(path, tag):
                rows = self.loader_for(base)(path, tag); return rows + [rows[-1]] if path.parent.name == "run-64" and tag == MODULE.COLLECTION_TAG else rows
            with self.assertRaisesRegex(MODULE.BenchmarkError, "duplicate TensorBoard step"):
                self.summarize(root, protocol, spec, duplicate)
            base[64][4] = float("nan")
            with self.assertRaisesRegex(MODULE.BenchmarkError, "finite and positive"):
                self.summarize(root, protocol, spec, self.loader_for(base))

    def test_threshold_excludes_slow_small_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory))
            times = {64: [1.2] * 8, 128: [1.0] * 8, 256: [0.95] * 8}
            result = self.summarize(root, protocol, spec, self.loader_for(times))
            self.assertEqual(result["selected_topology"]["num_envs"], 128)

    def test_runtime_failure_is_evidence_backed_and_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory))
            payload = json.loads(spec.read_text()); failed = payload["candidates"][0]
            log = root / "run-64/raw.log"; log.write_text("CUDA out of memory\n")
            failed.update({"exit_code": 1, "raw_log": ref(root, log), "failure": {"kind": "oom", "message": "CUDA out of memory"}})
            for key in ("event_file", "checkpoint", "checkpoint_inspection", "runner_config", "validity"):
                failed.pop(key)
            write_json(spec, payload)
            times = {64: [1.0] * 8, 128: [1.0] * 8, 256: [1.0] * 8}
            result = self.summarize(root, protocol, spec, self.loader_for(times))
            self.assertEqual(result["candidates"]["64"]["status"], "runtime_failure")
            self.assertEqual(result["selected_topology"]["num_envs"], 128)

    def test_no_valid_candidate_and_invalid_failure_evidence_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory)); payload = json.loads(spec.read_text())
            for candidate in payload["candidates"]:
                log = root / f"run-{candidate['num_envs']}/raw.log"; log.write_text("runtime failure\n")
                candidate.update({"exit_code": 2, "raw_log": ref(root, log), "failure": {"kind": "runtime_error", "message": "runtime failure"}})
                for key in ("event_file", "checkpoint", "checkpoint_inspection", "runner_config", "validity"):
                    candidate.pop(key)
            write_json(spec, payload)
            with self.assertRaisesRegex(MODULE.BenchmarkError, "no valid"):
                self.summarize(root, protocol, spec, lambda *_: [])
            payload["candidates"][0]["failure"] = {"kind": "low_reward", "message": "not allowed"}; write_json(spec, payload)
            with self.assertRaisesRegex(MODULE.BenchmarkError, "explicit runtime error evidence"):
                self.summarize(root, protocol, spec, lambda *_: [])

    def test_success_or_reward_outcomes_cannot_influence_scale_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory)); payload = json.loads(spec.read_text())
            payload["candidates"][0]["reward"] = 999.0; write_json(spec, payload)
            with self.assertRaisesRegex(MODULE.BenchmarkError, "must not contain success or reward"):
                self.summarize(root, protocol, spec, self.loader_for({64: [1.0] * 8, 128: [1.0] * 8, 256: [1.0] * 8}))

    def test_exit_zero_oom_log_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory)); payload = json.loads(spec.read_text())
            log = root / "run-64/raw.log"; log.write_text("CUDA out of memory\n")
            payload["candidates"][0]["raw_log"] = ref(root, log); write_json(spec, payload)
            with self.assertRaisesRegex(Exception, "out-of-memory evidence"):
                self.summarize(root, protocol, spec, self.loader_for({64: [1.0] * 8, 128: [1.0] * 8, 256: [1.0] * 8}))

    def test_failure_log_must_be_inside_its_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root, protocol, spec = self.make_case(Path(directory)); payload = json.loads(spec.read_text())
            outside = root / "other-run.log"; outside.write_text("CUDA out of memory\n")
            candidate = payload["candidates"][0]
            candidate.update({"exit_code": 1, "raw_log": ref(root, outside), "failure": {"kind": "oom", "message": "CUDA out of memory"}})
            write_json(spec, payload)
            with self.assertRaisesRegex(MODULE.BenchmarkError, "must be inside run_dir"):
                self.summarize(root, protocol, spec, self.loader_for({64: [1.0] * 8, 128: [1.0] * 8, 256: [1.0] * 8}))

    def test_default_checkpoint_verifier_rejects_plaintext_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / "model_7.pt"; checkpoint.write_bytes(b"checkpoint-64")
            manifest = root / "manifest.json"; provenance = root / "provenance.json"; runner = root / "runner.json"; training = root / "train.py"
            write_json(manifest, {}); write_json(provenance, {}); write_json(runner, {}); training.write_text("# train\n")
            with self.assertRaises(Exception):
                MODULE._default_checkpoint_verifier(checkpoint, manifest, provenance, runner, training)


if __name__ == "__main__":
    unittest.main()
