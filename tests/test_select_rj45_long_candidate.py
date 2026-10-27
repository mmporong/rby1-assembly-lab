import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "select_rj45_long_candidate.py"
SPEC = importlib.util.spec_from_file_location("select_rj45_long_candidate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

GROUPS = list(MODULE.EXPECTED_GROUPS)
LABELS = ["baseline", *GROUPS]
TRAINING_CONFIG = {
    "A0": "configs/rj45-training-target-offset-v1.json",
    "A1": "configs/rj45-training-target-offset-a1-discounted-pose-v2.json",
    "A2": "configs/rj45-training-target-offset-a2-keypoint-shape-v2.json",
    "E1": "configs/rj45-training-target-offset-v1.json",
    "Z1": "configs/rj45-training-target-offset-v1.json",
}
PPO_CONFIG = {
    "A0": "configs/ppo-rj45-v1.json", "A1": "configs/ppo-rj45-v1.json",
    "A2": "configs/ppo-rj45-v1.json", "E1": "configs/ppo-rj45-low-exploration-v1.json",
    "Z1": "configs/ppo-rj45-low-exploration-v1.json",
}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    data = (json.dumps(value, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return sha_bytes(data)


def long_plan(parent_sha="a" * 64):
    return {
        "schema_version": 1,
        "parent_pilot_plan_sha256": parent_sha,
        "selection_preregistered_before_any_pilot_32_case_policy_evaluation_result": True,
        "pilot_candidate_selection": {
            "eligible_groups": GROUPS,
            "tie_break_order": ["Z1", "E1", "A2", "A1", "A0"],
            "primary_rank": "Largest strict-v1 success count on the predeclared 32 pilot development cases.",
            "not_a_selection_metric": "Do not rank different reward profiles by their reward sums, whose scales differ.",
        },
    }


def parent_plan():
    groups = []
    frozen = {"scripts/analyze_rj45_paired.py": "a" * 64}
    for label in GROUPS:
        zero = label == "Z1"
        groups.append({
            "id": label, "training_config": TRAINING_CONFIG[label],
            "ppo_config": PPO_CONFIG[label],
            "training_flags": ["--zero-init-actor-output"] if zero else [],
        })
        frozen[TRAINING_CONFIG[label]] = "b" * 64
        frozen[PPO_CONFIG[label]] = "c" * 64
    return {
        "schema_version": 1, "status": "predeclared_before_training",
        "development_evaluation": {
            "baseline": "biased_zero", "episodes": 32, "num_envs": 16,
            "seed": 20261028, "matching_condition_ids_and_bias_required": True,
            "all_requested_attempts_in_denominator": True,
            "primary": "strict-v1 success (12–13 mm depth, <=50 µm lateral)",
            "bootstrap_seed": 20261029, "bootstrap_resamples": 100000,
        },
        "training": {
            "environments": 64, "updates": 20, "rollout_steps_per_env": 128,
            "seed": 20261026, "checkpoint": "final_only_model_19.pt",
        },
        "groups": groups, "frozen_inputs": {"sha256": frozen},
    }


def report(label):
    training = TRAINING_CONFIG.get(label, TRAINING_CONFIG["A0"])
    ppo = PPO_CONFIG.get(label, PPO_CONFIG["A0"])
    return {
        "seed": 20261028, "evaluation_start": "held", "evaluation_num_envs": 16,
        "requested_attempts": 32,
        "task_model": {"evaluation_stability_limits": {
            "max_depth_m": 0.013, "required_continuous_dwell_s": 0.4,
            "max_relative_linear_speed_m_s": 0.005,
            "max_relative_angular_speed_rad_s": 0.05,
        }},
        "episode_outcomes": [],
        "rj45_training_config": {"path": training, "sha256": "b" * 64},
        "provenance": {
            "task_config_sha256": "d" * 64,
            "source_sha256": {},
            "policy_config": ppo, "policy_config_sha256": "c" * 64,
            "verified_checkpoint_evidence": {"policy_initialization": {"zero_actor_output_layer": label == "Z1"}},
        },
    }


def fresh_audit(reports, successes=None):
    successes = successes or {"baseline": 12, "A0": 1, "A1": 2, "A2": 3, "E1": 4, "Z1": 5}
    cases = []
    for index in range(32):
        cases.append({
            "paired_condition_id": f"canonical-{index}",
            "results": {
                label: {"valid": True, "success": index < successes[label], "forbidden_contact_load_peak_n": 0.0}
                for label in LABELS
            },
        })
    identifiers = {}
    for label, value in reports.items():
        identifiers[label] = {
            "training_config_path": value["rj45_training_config"]["path"],
            "training_config_sha256": value["rj45_training_config"]["sha256"],
            "policy_config": value["provenance"]["policy_config"],
            "policy_config_sha256": value["provenance"]["policy_config_sha256"],
            "policy_initialization": value["provenance"].get("verified_checkpoint_evidence", {}).get("policy_initialization"),
        }
    return {
        "schema_version": 1,
        "scope": {},
        "experiment": {"reported_configuration_identifiers": identifiers},
        "bootstrap": {},
        "results": {
            label: {"attempts": 32, "invalid_episodes": 0, "successes_all_attempts": successes[label]}
            for label in LABELS
        },
        "paired_contrasts": {}, "evaluation_bias_coverage": {}, "paired_cases": cases,
    }


class FakeAnalyzer:
    @staticmethod
    def _load(path):
        raw = path.read_bytes()
        return json.loads(raw), raw, sha_bytes(raw)

    @staticmethod
    def analyze_reports(baseline, conditions, **_kwargs):
        reports = {"baseline": baseline, **conditions}
        if any(value.get("bad_case") for value in reports.values()):
            raise ValueError("paired_condition_id is not canonical")
        return fresh_audit(reports)

    @staticmethod
    def _verify_training_bundle(label, _report, *, checkpoint_path, manifest_path, training_provenance_path, runner_config_path):
        noise = 0.1 if label in {"E1", "Z1"} else 0.6
        return {
            "status": MODULE.VERIFIED_BUNDLE_STATUS,
            "checkpoint": {"path": str(checkpoint_path), "sha256": MODULE._sha_file(checkpoint_path)},
            "checkpoint_manifest": {"path": str(manifest_path), "sha256": MODULE._sha_file(manifest_path)},
            "training_provenance": {"path": str(training_provenance_path), "sha256": MODULE._sha_file(training_provenance_path)},
            "runner_config": {"path": str(runner_config_path), "sha256": MODULE._sha_file(runner_config_path)},
            "training_seed": 20261026, "training_num_envs": 64,
            "rollout_steps_per_env": 128, "training_updates": 20,
            "initial_action_noise_std": noise,
        }


class SelectCandidateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.parent = parent_plan()
        task_config = {"evaluation": {
            "assembled_depth_m": 0.012, "max_depth_m": 0.013,
            "max_lateral_error_m": 0.00005,
            "max_orientation_error_rad": 0.008726646259971648,
            "dwell_s": 0.4, "max_contact_force_n": 25.0,
            "max_relative_linear_speed_m_s": 0.005,
            "max_relative_angular_speed_rad_s": 0.05,
        }}
        self.task_sha = write_json(self.root / "configs/tasks/rj45-v1.json", task_config)
        self.parent["frozen_inputs"]["sha256"]["configs/tasks/rj45-v1.json"] = self.task_sha
        for relative, content in (
            ("src/rby1_assembly/test.py", b"SOURCE = 'frozen'\n"),
            ("scripts/evaluate_policy.py", b"# frozen evaluator\n"),
        ):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.parent["frozen_inputs"]["sha256"][relative] = sha_bytes(content)
        self.mapping = MODULE._parent_contract(self.parent)
        self.strict, loaded_sha = MODULE._load_strict_contract(self.root, self.parent)
        self.assertEqual(loaded_sha, self.task_sha)
        for relative in set(PPO_CONFIG.values()):
            write_json(self.root / relative, {"policy": {"init_noise_std": 0.1 if "low" in relative else 0.6}})

    def tearDown(self):
        self.temp.cleanup()

    def build_evidence(self, mutate_reports=None):
        reports = {label: report(label) for label in LABELS}
        frozen_source = {
            path: digest
            for path, digest in self.parent["frozen_inputs"]["sha256"].items()
            if (path.startswith("src/rby1_assembly/") and path.endswith(".py"))
            or path == "scripts/evaluate_policy.py"
        }
        for value in reports.values():
            value["provenance"]["task_config_sha256"] = self.task_sha
            value["provenance"]["source_sha256"] = frozen_source
        if mutate_reports:
            mutate_reports(reports)
        audit = fresh_audit(reports)
        audit["input_reports"] = {}
        audit["training_artifact_verification"] = {}
        for label, value in reports.items():
            relative = f"raw/{label}.json"
            digest = write_json(self.root / relative, value)
            audit["input_reports"][label] = {"path": relative, "sha256": digest}
        for label in GROUPS:
            paths = {}
            for key, filename in (
                ("checkpoint", "model_19.pt"), ("checkpoint_manifest", "checkpoint_manifest.json"),
                ("training_provenance", "training_provenance.json"), ("runner_config", "runner_config.json"),
            ):
                relative = f"bundle/{label}/{filename}"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"{label}-{key}".encode())
                paths[key] = path
            verified = FakeAnalyzer._verify_training_bundle(
                label, reports[label], checkpoint_path=paths["checkpoint"],
                manifest_path=paths["checkpoint_manifest"], training_provenance_path=paths["training_provenance"],
                runner_config_path=paths["runner_config"],
            )
            verified["causal_interpretation"] = "prohibited_without_a_separately_verified_preregistered_experiment_plan"
            for value in verified.values():
                if isinstance(value, dict) and "path" in value:
                    value["path"] = str(Path(value["path"]).relative_to(self.root))
            audit["training_artifact_verification"][label] = verified
        return audit

    def test_plan_chain_rejects_plan_and_sidecar_style_cotamper(self):
        parent_path = self.root / "parent.json"
        parent_sha = write_json(parent_path, self.parent)
        plan_path = self.root / "long.json"
        original_sha = write_json(plan_path, long_plan(parent_sha))
        modified = long_plan(parent_sha)
        modified["pilot_candidate_selection"]["tie_break_order"].reverse()
        write_json(plan_path, modified)
        with self.assertRaisesRegex(ValueError, "remotely preregistered"):
            MODULE._verify_plan_chain(plan_path, parent_path, original_sha)

    def test_parent_rejects_wrong_seed_and_config_mapping(self):
        wrong_seed = copy.deepcopy(self.parent)
        wrong_seed["development_evaluation"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "seed"):
            MODULE._parent_contract(wrong_seed)
        wrong_config = copy.deepcopy(self.parent)
        wrong_config["groups"][0]["training_config"] = TRAINING_CONFIG["A1"]
        with self.assertRaisesRegex(ValueError, "A0 configuration"):
            MODULE._parent_contract(wrong_config)

    def test_missing_raw_report_and_status_only_bundle_fail(self):
        audit = self.build_evidence()
        (self.root / audit["input_reports"]["A2"]["path"]).unlink()
        with self.assertRaisesRegex(ValueError, "missing or unreadable"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)
        audit = self.build_evidence()
        audit["training_artifact_verification"]["A0"] = {"status": MODULE.VERIFIED_BUNDLE_STATUS}
        with self.assertRaisesRegex((ValueError, TypeError), "checkpoint"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

    def test_modified_report_and_checkpoint_fail_hash_verification(self):
        audit = self.build_evidence()
        report_path = self.root / audit["input_reports"]["A0"]["path"]
        report_path.write_bytes(report_path.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)
        audit = self.build_evidence()
        checkpoint = self.root / audit["training_artifact_verification"]["A1"]["checkpoint"]["path"]
        checkpoint.write_bytes(b"modified")
        with self.assertRaisesRegex(ValueError, "bytes differ"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

    def test_wrong_seed_start_label_swap_and_arbitrary_case_fail(self):
        for field, value in (("seed", 7), ("evaluation_start", "reset")):
            audit = self.build_evidence(lambda reports: reports["A0"].__setitem__(field, value))
            with self.assertRaisesRegex(ValueError, field):
                MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)
        audit = self.build_evidence(lambda reports: reports["A0"].__setitem__("rj45_training_config", reports["A1"]["rj45_training_config"]))
        with self.assertRaisesRegex(ValueError, "A0 report uses the wrong"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)
        audit = self.build_evidence(lambda reports: reports["A2"].__setitem__("bad_case", True))
        with self.assertRaisesRegex(ValueError, "not canonical"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

    def test_loosened_report_limit_and_missing_required_report_limit_fail(self):
        def loosen(reports):
            reports["A1"]["task_model"]["evaluation_stability_limits"]["assembled_depth_m"] = 0.001

        audit = self.build_evidence(loosen)
        with self.assertRaisesRegex(ValueError, "loosens or changes"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

        def omit(reports):
            del reports["A2"]["task_model"]["evaluation_stability_limits"]["max_relative_linear_speed_m_s"]

        audit = self.build_evidence(omit)
        with self.assertRaisesRegex(ValueError, "omits reported"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

    def test_frozen_task_config_cannot_omit_analyzer_fallback_limit(self):
        path = self.root / "configs/tasks/rj45-v1.json"
        payload = json.loads(path.read_text())
        del payload["evaluation"]["max_lateral_error_m"]
        modified_sha = write_json(path, payload)
        parent = copy.deepcopy(self.parent)
        parent["frozen_inputs"]["sha256"]["configs/tasks/rj45-v1.json"] = modified_sha
        with self.assertRaisesRegex(ValueError, "max_lateral_error_m"):
            MODULE._load_strict_contract(self.root, parent)

    def test_report_source_map_and_actual_source_bytes_are_frozen(self):
        def alter_source_map(reports):
            reports["E1"]["provenance"]["source_sha256"].pop("scripts/evaluate_policy.py")

        audit = self.build_evidence(alter_source_map)
        with self.assertRaisesRegex(ValueError, "source map differs"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

        verified = MODULE._verify_parent_source_files(self.root, self.parent)
        self.assertEqual(len(verified), 2)
        (self.root / "src/rby1_assembly/test.py").write_text("SOURCE = 'changed'\n")
        with self.assertRaisesRegex(ValueError, "source bytes differ"):
            MODULE._verify_parent_source_files(self.root, self.parent)

    def test_success_requires_bounded_grip_force_and_insertion_contact_load(self):
        def success_outcome(**overrides):
            outcome = {
                "success": True,
                "final_depth_m": 0.0125,
                "final_lateral_error_m": 0.00004,
                "final_orientation_error_rad": 0.004,
                "final_dwell_s": 0.4,
                "grip_contact_force_peak_n": 4.0,
                "insertion_contact_force_peak_n": 1.0,
                "insertion_contact_load_peak_n": 5.0,
            }
            outcome.update(overrides)
            return outcome

        def missing_grip(reports):
            outcome = success_outcome()
            del outcome["grip_contact_force_peak_n"]
            reports["A0"]["episode_outcomes"] = [outcome]

        audit = self.build_evidence(missing_grip)
        with self.assertRaisesRegex(ValueError, "invalid grip_contact_force_peak_n"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

        audit = self.build_evidence(
            lambda reports: reports["A1"].__setitem__(
                "episode_outcomes", [success_outcome(insertion_contact_load_peak_n=float("nan"))]
            )
        )
        with self.assertRaisesRegex(ValueError, "invalid insertion_contact_load_peak_n"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

        audit = self.build_evidence(
            lambda reports: reports["A2"].__setitem__(
                "episode_outcomes",
                [success_outcome(
                    grip_contact_force_peak_n=100.0,
                    insertion_contact_load_peak_n=100.0,
                    insertion_contact_force_peak_n=1.0,
                )],
            )
        )
        with self.assertRaisesRegex(ValueError, "grip contact force.*insertion contact load"):
            MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)

    def test_baseline_other_than_twelve_fails(self):
        audit = self.build_evidence()
        original = FakeAnalyzer.analyze_reports
        try:
            FakeAnalyzer.analyze_reports = staticmethod(lambda baseline, conditions, **kwargs: fresh_audit({"baseline": baseline, **conditions}, {"baseline": 11, "A0": 1, "A1": 2, "A2": 3, "E1": 4, "Z1": 5}))
            altered = FakeAnalyzer.analyze_reports(
                json.loads((self.root / audit["input_reports"]["baseline"]["path"]).read_text()),
                {label: json.loads((self.root / audit["input_reports"][label]["path"]).read_text()) for label in GROUPS},
            )
            for field in MODULE.FRESH_AUDIT_FIELDS:
                audit[field] = altered[field]
            with self.assertRaisesRegex(ValueError, "12/32"):
                MODULE._verify_raw_audit(self.root, self.parent, audit, FakeAnalyzer, self.mapping, self.strict, self.task_sha)
        finally:
            FakeAnalyzer.analyze_reports = original

    def test_verified_ranking_uses_success_then_tie_order(self):
        reports = {label: report(label) for label in LABELS}
        successes = {label: 12 for label in LABELS}
        verified = fresh_audit(reports, successes)
        verified["training_artifact_verification"] = {
            label: {"status": MODULE.VERIFIED_BUNDLE_STATUS} for label in GROUPS
        }
        result = MODULE._rank_verified_audit(long_plan(), verified)
        self.assertEqual(result["decision"]["selected_group"], "Z1")
        self.assertFalse(result["decision"]["reward_sums_used"])
        self.assertIn("relative-velocity", result["verification_scope"]["unverified"][0])


if __name__ == "__main__":
    unittest.main()
