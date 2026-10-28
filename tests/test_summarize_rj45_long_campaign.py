import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_rj45_long_campaign.py"
SPEC = importlib.util.spec_from_file_location("summarize_rj45_long_campaign", SCRIPT)
assert SPEC is not None
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SummarizeRj45LongCampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / "snapshot"
        self.snapshot.mkdir()
        self.files = {}
        for index in range(93):
            relative = f"files/{index:02d}.txt"
            path = self.snapshot / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"evidence-{index}".encode())
            self.files[relative] = MODULE._sha_file(path)
        payload = {"terminal_status": "completed", "files_sha256": self.files}
        (self.snapshot / ".retrieval.json").write_text(json.dumps(payload))
        self.manifest = self.root / "retrieval.json"
        self.manifest.write_text(json.dumps({
            "schema_version": 1,
            "terminal_status": "completed",
            "snapshot": str(self.snapshot),
            "files_sha256": self.files,
        }))

    def tearDown(self):
        self.temporary.cleanup()

    def test_retrieval_checks_remote_payload_and_local_bytes(self):
        _retrieval, snapshot, hashes, _payload_sha = MODULE.verify_retrieval_manifest(self.manifest)
        self.assertEqual(snapshot, self.snapshot)
        self.assertEqual(hashes, self.files)

    def test_retrieval_rejects_local_byte_hash_mismatch(self):
        (self.snapshot / "files/00.txt").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "byte hash mismatch"):
            MODULE.verify_retrieval_manifest(self.manifest)

    def test_retrieval_rejects_missing_evidence(self):
        (self.snapshot / "files/00.txt").unlink()
        with self.assertRaisesRegex(ValueError, "cannot read evidence file"):
            MODULE.verify_retrieval_manifest(self.manifest)

    def test_consumed_report_or_checkpoint_must_be_listed(self):
        for name in ("reports/unlisted.json", "runs/seed/model_19.pt"):
            path = self.snapshot / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"unlisted")
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError, "absent from retrieval manifest"
            ):
                MODULE.verify_consumed_artifacts(
                    self.snapshot, self.files, {name: path}
                )

    def test_campaign_contract_rejects_tampering_and_plan_sha_mismatch(self):
        input_manifest = self.root / "input-manifest.json"
        input_manifest.write_text("{}")
        plan = {
            "training": {
                "assessed_checkpoints": [
                    {"filename": "model_19.pt", "iteration_index": 19, "completed_updates": 20}
                ]
            },
            "development_evaluation": {"seed": 9},
        }
        contract = {
            "schema_version": 1,
            "plan": {"sha256": "a" * 64},
            "profile_label": "Z1",
            "training_seeds": [1, 2, 3],
            "assessed_checkpoints": plan["training"]["assessed_checkpoints"],
            "development_evaluation_seed": 9,
            "final_holdout_opened": False,
            "input_manifest": {
                "path": str(input_manifest),
                "sha256": MODULE._sha_file(input_manifest),
            },
            "seed_ppo_configs": {str(seed): {} for seed in (1, 2, 3)},
        }
        with self.assertRaisesRegex(ValueError, "plan SHA differs"):
            MODULE.validate_campaign_contract(
                contract,
                repo_root=self.root,
                plan=plan,
                plan_sha="b" * 64,
                baseline_sha="c" * 64,
                seeds=[1, 2, 3],
                checkpoints=[{"iteration_index": 19, "completed_updates": 20}],
            )
        contract["plan"]["sha256"] = "a" * 64
        contract["profile_label"] = "A0"
        with self.assertRaisesRegex(ValueError, "profile must be Z1"):
            MODULE.validate_campaign_contract(
                contract,
                repo_root=self.root,
                plan=plan,
                plan_sha="a" * 64,
                baseline_sha="c" * 64,
                seeds=[1, 2, 3],
                checkpoints=[{"iteration_index": 19, "completed_updates": 20}],
            )

    def test_plan_rejects_duplicate_seed_and_checkpoint_ids(self):
        def plan(seeds, checkpoints):
            return {
                "training": {
                    "training_seeds": seeds,
                    "assessed_checkpoints": checkpoints,
                },
                "development_evaluation": {
                    "promotion_rule": MODULE.EXPECTED_PROMOTION_RULE,
                    "attempts_per_policy": 32,
                },
            }

        checkpoints = [
            {"filename": "model_19.pt", "iteration_index": 19, "completed_updates": 20},
            {"filename": "model_95.pt", "iteration_index": 95, "completed_updates": 96},
            {"filename": "model_199.pt", "iteration_index": 199, "completed_updates": 200},
        ]
        with self.assertRaisesRegex(ValueError, "three integer training seeds"):
            MODULE._campaign_contract(plan([1, 1, 3], checkpoints))
        duplicate = [checkpoints[0], checkpoints[0], checkpoints[2]]
        with self.assertRaisesRegex(ValueError, "checkpoints must be unique"):
            MODULE._campaign_contract(plan([1, 2, 3], duplicate))

    def test_consumed_artifact_coverage_rejects_omission(self):
        with self.assertRaisesRegex(ValueError, "coverage differs"):
            MODULE._assert_consumed_artifact_coverage(
                {"report.json": "a" * 64}, {"report.json", "checkpoint.pt"}
            )

    def test_report_identity_rejects_mismatched_seed(self):
        with self.assertRaisesRegex(ValueError, "seed differs"):
            MODULE._validate_report_identity(
                {"requested_attempts": 32, "attempts": 32},
                {"training_seed": 2, "checkpoint_completed_updates": 20},
                seed=1,
                completed_updates=20,
            )

    def test_training_provenance_frozen_map_rejects_mismatch_and_omission(self):
        repo = SCRIPT.parents[1]
        retrieval = repo / "artifacts/cloud/rj45-long-retry-retrieval-20261027.json"
        if not retrieval.exists():
            self.skipTest("ignored retrieved snapshot is unavailable")
        snapshot = Path(json.loads(retrieval.read_text())["snapshot"])
        run = next((snapshot / "runs/20261101/rj45-v1").iterdir())
        original = json.loads((run / "training_provenance.json").read_text())
        anchored = json.loads(
            (repo / "artifacts/rj45-long-inputs-retry-20261027/input-manifest.json").read_text()
        )["inputs"]
        source = original["source_and_config_sha256"]
        frozen_path = next(path for path in source if path.startswith("src/rby1_assembly/"))
        for case in ("mismatch", "missing"):
            changed = json.loads(json.dumps(original))
            if case == "mismatch":
                changed["source_and_config_sha256"][frozen_path] = "0" * 64
                expected = "differs from frozen pre-run map"
            else:
                del changed["source_and_config_sha256"][frozen_path]
                expected = "source/config set differs"
            path = self.root / f"provenance-{case}.json"
            path.write_text(json.dumps(changed))
            with self.subTest(case=case), self.assertRaisesRegex(ValueError, expected):
                MODULE.verify_training_provenance_frozen_map(
                    provenance_path=path,
                    runner_path=run / "runner_config.json",
                    snapshot=snapshot,
                    anchored_inputs=anchored,
                    seed=20261101,
                )

    def test_pinned_v1_analyzer_tamper_and_v2_semantic_change_fail_closed(self):
        analyzer = self.root / "analyzer.py"
        analyzer.write_text("VALUE = 1\n")
        expected = hashlib.sha256(analyzer.read_bytes()).hexdigest()
        analyzer.write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "pinned analyzer bytes differ"):
            MODULE._load_pinned_analyzer(analyzer, expected, "tampered")

        frozen = {
            "results": {"policy": {"successes_all_attempts": 10}},
            "paired_contrasts": {},
            "paired_cases": [],
        }
        posthoc = json.loads(json.dumps(frozen))
        posthoc["results"]["policy"]["successes_all_attempts"] = 11
        with self.assertRaisesRegex(ValueError, "changes frozen-v1 gate semantics"):
            MODULE._assert_gate_semantics_match(frozen, posthoc)

    def test_remote_reference_cannot_escape_repository(self):
        escaped = self.root.parent / "outside.json"
        with self.assertRaisesRegex(ValueError, "escapes the repository"):
            MODULE._local_path_from_remote_ref(
                self.root,
                {"path": f"/remote/{self.root.name}/../{escaped.name}"},
                "malicious",
            )

    def test_common_checkpoint_uses_total_then_earliest_tie(self):
        rows = []
        for updates, successes in ((20, (10, 10, 7)), (96, (10, 12, 11)), (200, (11, 11, 9))):
            rows.extend(
                {"completed_updates": updates, "training_seed": seed, "successes": success}
                for seed, success in zip((20261101, 20261102, 20261103), successes)
            )
        result = MODULE.select_common_checkpoint(rows)
        self.assertEqual(result["selected"]["completed_updates"], 96)
        self.assertEqual(result["selected"]["total_successes"], 33)

        for row in rows:
            if row["completed_updates"] == 200:
                row["successes"] += 1 if row["training_seed"] in (20261101, 20261102) else 0
        self.assertEqual(
            MODULE.select_common_checkpoint(rows)["selected"]["completed_updates"], 96
        )

    def test_actual_snapshot_reproduces_common_96_and_failed_gate(self):
        repo = SCRIPT.parents[1]
        retrieval = repo / "artifacts/cloud/rj45-long-retry-retrieval-20261027.json"
        if not retrieval.exists():
            self.skipTest("ignored retrieved snapshot is unavailable")
        output = self.root / "actual-summary.json"
        result = MODULE.summarize_campaign(
            repo_root=repo,
            retrieval_manifest_path=retrieval,
            plan_path=repo / "docs/evidence/rj45-paper-ablation/long-run-plan.json",
            baseline_path=repo / "docs/evidence/rj45-paper-ablation/pilot-results/baseline.json",
            output_path=output,
        )
        selected = result["common_checkpoint_selection"]["selected"]
        self.assertEqual(selected["completed_updates"], 96)
        self.assertEqual(
            selected["successes_by_training_seed"],
            {"20261101": 10, "20261102": 12, "20261103": 11},
        )
        self.assertEqual(result["promotion_gate"]["baseline_successes"], 12)
        self.assertFalse(result["promotion_gate"]["passed"])
        self.assertEqual(
            result["integrity"]["consumed_snapshot_artifact_check"]["file_count"],
            37,
        )
        anchors = result["inputs"]["campaign_contract"]["verified_input_anchors"]
        self.assertEqual(
            anchors["artifacts/rj45-paper-ablation/baseline.json"],
            "b1c82b3704dae971abae41d91e88f6e6e2c94f625132fc06cea8a8d2dc47da61",
        )
        self.assertIn("docs/evidence/rj45-paper-ablation/long-run-plan.json", anchors)
        self.assertIn("artifacts/rj45-paper-ablation/long-candidate.json", anchors)
        analyzers = result["inputs"]["analyzers"]
        self.assertEqual(
            analyzers["gate"]["sha256"],
            "cdd087abb1298178cdf57b21a98c98c098da28d4730dd77b22429e72b37699a7",
        )
        self.assertIn("cannot change gate counts", analyzers["posthoc_bundle_verifier"]["role"])
        lineage = result["integrity"]["training_source_and_config_check"]
        self.assertEqual(lineage["pre_run_input_manifest_entry_count"], 96)
        self.assertEqual(lineage["training_relevant_union_count"], 50)
        self.assertEqual(lineage["non_training_provenance_input_count"], 46)
        for seed in ("20261101", "20261102", "20261103"):
            self.assertEqual(
                lineage["by_training_seed"][seed]["source_and_config_entry_count"], 49
            )
            self.assertEqual(
                lineage["by_training_seed"][seed]["frozen_pre_run_digest_match_count"],
                48,
            )
        first_audit = result["paired_audits"]["seed-20261101-updates-20"]
        self.assertEqual(first_audit["gate_analyzer"], "frozen_v1")
        self.assertTrue(first_audit["posthoc_v2_semantics_match"])
        bundle = first_audit["training_artifact_verification"]
        self.assertNotIn("unverified", bundle)
        self.assertIn("summary_wrapper_cross_checks", bundle)
        self.assertEqual(len(bundle["remaining_unverified"]), 1)

    def test_consistent_zero_success_baseline_tamper_fails_closed(self):
        repo = SCRIPT.parents[1]
        retrieval = repo / "artifacts/cloud/rj45-long-retry-retrieval-20261027.json"
        if not retrieval.exists():
            self.skipTest("ignored retrieved snapshot is unavailable")
        baseline = json.loads(
            (repo / "docs/evidence/rj45-paper-ablation/pilot-results/baseline.json").read_text()
        )
        for outcome in baseline["episode_outcomes"]:
            outcome["success"] = False
        baseline["valid_successes"] = 0
        baseline["valid_failures"] = 32
        baseline["success_rate_valid_episodes"] = 0.0
        tampered = self.root / "tampered-baseline.json"
        tampered.write_text(json.dumps(baseline, indent=2) + "\n")
        with self.assertRaisesRegex(ValueError, "verified input manifest.*baseline.json"):
            MODULE.summarize_campaign(
                repo_root=repo,
                retrieval_manifest_path=retrieval,
                plan_path=repo / "docs/evidence/rj45-paper-ablation/long-run-plan.json",
                baseline_path=tampered,
                output_path=self.root / "must-not-exist.json",
            )


if __name__ == "__main__":
    unittest.main()
