import unittest
from dataclasses import replace

import numpy as np

from rby1_assembly.rj45_task_admission import (
    CASES, EXPECTED_MEAN_CONSTANT_ACTION, EXPECTED_SOURCE_TRACE_GZIP_SHA256,
    AdmissionProtocol, ClassicalSeal, ContractEvidence, EpisodeResult, ProtocolStage,
    TuningResult, adjudicate_admission, canonical_tuning_report_bytes, compute_observability,
    make_baseline_artifact_seal, required_admission_slots, seal_classical_selection,
    select_classical_candidate, validate_admission_contract,
)


def evidence(**changes):
    values = {name: True for name in ContractEvidence.__dataclass_fields__}
    values.update(changes)
    return ContractEvidence(**values)


def baseline_seal():
    return make_baseline_artifact_seal(
        source_trace_gzip_sha256=EXPECTED_SOURCE_TRACE_GZIP_SHA256,
        mean_constant_action=EXPECTED_MEAN_CONSTANT_ACTION,
        playback_selected_rows=364,
    )


def complete_tuning():
    rows = []
    for candidate in ("A", "B", "C"):
        successes = {"A": 4, "B": 4, "C": 3}[candidate]
        force = {"A": 5.0, "B": 4.0, "C": 3.0}[candidate]
        for index, case in enumerate(CASES):
            rows.append(TuningResult(candidate, case, index < successes, False, False, force, 2.0))
    return rows


def classical_seal():
    rows = complete_tuning()
    return seal_classical_selection(select_classical_candidate(rows), canonical_tuning_report_bytes(rows), rows)


def observation_window(case_id, slot, *, contact=True, length=13):
    rows = np.zeros((length, 79), dtype=np.float64)
    rows[:, 3] = 1.0
    if contact:
        rows[:, 37] = 0.25 / 8.0
    sign = 1.0 if case_id.endswith("positive") else -1.0
    rows[:, 7] = sign * {"tx": 0.10, "tz": 0.12, "rx": 0.14}[case_id[:2]] + (0.001 if slot == 2 else 0.0)
    return tuple(tuple(row) for row in rows)


def complete_episodes():
    base, classical = baseline_seal(), classical_seal()
    zero_success = {("rj45-train-02", "rx-positive"), ("rj45-train-05", "tx-positive")}
    rows = []
    for grasp, case, method, slot in required_admission_slots():
        success = method == "oracle_gt_reference" or (method == "zero" and slot == 1 and (grasp, case) in zero_success)
        artifact = base.mean_constant_artifact_sha256 if method == "mean_constant" else base.playback_artifact_sha256 if method == "open_loop_playback" else None
        rows.append(EpisodeResult(
            grasp, case, method, slot, success, False, False, 0.0, 0.0,
            frozenset(("observation79", "provider_owned_state")) if method != "oracle_gt_reference" else None,
            classical.canonical_controller_config_sha256 if method == "classical_force_search" else None,
            artifact,
            observation_window(case, slot) if method == "zero" else (),
        ))
    return rows


def validate(rows):
    return validate_admission_contract(rows, evidence(), classical_seal=classical_seal(), baseline_seal=baseline_seal())


def adjudicate(rows, *, seal=None, baseline=None, contract=None):
    return adjudicate_admission(rows, contract or evidence(), classical_seal=classical_seal() if seal is None else seal, baseline_seal=baseline_seal() if baseline is None else baseline)


class TaskAdmissionTests(unittest.TestCase):
    def test_required_records_have_no_safe_defaults(self):
        with self.assertRaises(TypeError):
            ContractEvidence()
        with self.assertRaises(TypeError):
            EpisodeResult("g", "c", "zero", 1, False)

    def test_exact_72_denominator_and_missing_invalid(self):
        rows = complete_episodes()
        self.assertEqual(len(required_admission_slots()), 72)
        self.assertEqual(validate(rows), ())
        errors = validate(rows[:-1])
        self.assertIn("admission_slot_count_must_be_72", errors)
        self.assertIn("missing_requested_slots:1", errors)

    def test_all_contract_evidence_must_be_literal_true(self):
        rows = complete_episodes()
        for field in ContractEvidence.__dataclass_fields__:
            self.assertIn(field, validate_admission_contract(rows, evidence(**{field: None}), classical_seal=classical_seal(), baseline_seal=baseline_seal()))

    def test_missing_safety_metrics_and_authority_are_invalid(self):
        rows = complete_episodes()
        rows[0] = replace(rows[0], hard_contact=None, evaluator_force_p95_n=None, provider_inputs=None)
        errors = validate(rows)
        self.assertTrue(any(error.startswith("missing_or_invalid_safety_result") for error in errors))
        self.assertTrue(any(error.startswith("nonfinite_episode") for error in errors))
        self.assertTrue(any(error.startswith("provider_authority") for error in errors))

    def test_structured_seals_reject_tampering_and_episode_mismatch(self):
        rows = complete_episodes()
        seal = classical_seal()
        tampered = replace(seal, tuning_report_bytes=b"changed")
        self.assertEqual(adjudicate(rows, seal=tampered).state, "invalid_contract")
        index = next(i for i, row in enumerate(rows) if row.method_id == "classical_force_search")
        rows[index] = replace(rows[index], applied_controller_sha256="0" * 64)
        self.assertTrue(any(error.startswith("classical_controller_seal") for error in validate(rows)))
        with self.assertRaises(ValueError):
            make_baseline_artifact_seal(source_trace_gzip_sha256="0" * 64, mean_constant_action=EXPECTED_MEAN_CONSTANT_ACTION, playback_selected_rows=364)

    def test_observability_13_samples_signed_distance_repeat_noise(self):
        rows = complete_episodes(); axes = compute_observability(rows)
        self.assertEqual(len(axes), 6); self.assertTrue(all(axis.observable for axis in axes))
        index = next(i for i, row in enumerate(rows) if row.key == ("rj45-train-02", "tx-positive", "zero", 1))
        rows[index] = replace(rows[index], observation_samples=observation_window("tx-positive", 1, length=12))
        self.assertFalse(next(axis for axis in compute_observability(rows) if axis.grasp_id == "rj45-train-02" and axis.pair == "tx").observable)

    def test_tuning_selection_eligibility_order_and_grasp(self):
        self.assertEqual(select_classical_candidate(complete_tuning()).candidate_id, "B")
        rows = complete_tuning(); rows[6] = replace(rows[6], hard_contact=True)
        self.assertEqual(select_classical_candidate(rows).candidate_id, "A")
        with self.assertRaises(ValueError): select_classical_candidate(rows[:-1])
        rows = complete_tuning(); rows[0] = replace(rows[0], grasp_id="rj45-train-02")
        with self.assertRaises(ValueError): select_classical_candidate(rows)

    def test_state_precedence_and_protocol_order(self):
        protocol = AdmissionProtocol()
        with self.assertRaises(RuntimeError): protocol.validate_admission(complete_episodes(), evidence(), baseline_seal())
        tuning = complete_tuning()
        protocol = protocol.validate_tuning(tuning).seal_classical(canonical_tuning_report_bytes(tuning), tuning)
        protocol = protocol.validate_admission(complete_episodes(), evidence(), baseline_seal())
        protocol = protocol.adjudicate(complete_episodes(), evidence())
        self.assertEqual(protocol.stage, ProtocolStage.ADJUDICATED)
        self.assertEqual(protocol.decision.state, "admit_rl_pilot")
        self.assertEqual(adjudicate(complete_episodes()[:-1]).state, "invalid_contract")

    def test_unsafe_success_never_counts_as_safe_no_headroom(self):
        rows = complete_episodes()
        zero = [i for i, row in enumerate(rows) if row.method_id == "zero" and row.slot == 1]
        for index in zero[:10]: rows[index] = replace(rows[index], success=True)
        self.assertEqual(adjudicate(rows).state, "valid_stop_no_headroom")
        rows[zero[0]] = replace(rows[zero[0]], hard_contact=True)
        decision = adjudicate(rows)
        self.assertNotEqual(decision.state, "valid_stop_no_headroom")
        self.assertEqual(decision.success_counts["zero"], 9)

    def test_safe_ten_plus_separate_unsafe_episode_cannot_admit(self):
        for unsafe_field in ("hard_contact", "forbidden_contact"):
            rows = complete_episodes()
            constant = [i for i, row in enumerate(rows) if row.method_id == "mean_constant"]
            for index in constant[:11]:
                rows[index] = replace(rows[index], success=True)
            rows[constant[0]] = replace(rows[constant[0]], **{unsafe_field: True})
            decision = adjudicate(rows)
            self.assertEqual(decision.success_counts["mean_constant"], 10)
            self.assertEqual(decision.state, "valid_stop_insufficient_structured_headroom")

    def test_sparse_and_structured_headroom_terminal_states(self):
        sparse = [replace(row, success=False) if row.method_id == "zero" and row.slot == 1 else row for row in complete_episodes()]
        self.assertEqual(adjudicate(sparse).state, "valid_hard_sparse")
        rows = complete_episodes()
        # Break orientation coverage while retaining two successes and valid feasibility.
        for i, row in enumerate(rows):
            if row.method_id == "zero" and row.slot == 1 and row.case_id.startswith("rx-"):
                rows[i] = replace(row, success=True)
        self.assertEqual(adjudicate(rows).state, "valid_stop_insufficient_structured_headroom")


if __name__ == "__main__":
    unittest.main()
