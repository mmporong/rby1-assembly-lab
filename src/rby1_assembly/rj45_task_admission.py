"""Pure CPU admission accounting for the RJ45 comparison task.

Safety and provenance fields are required. Missing evidence is a contract
failure rather than an implicit safe value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
import json
from statistics import median
from typing import Mapping, Sequence

import numpy as np


GRASPS = ("rj45-train-02", "rj45-train-05")
CASES = ("tx-positive", "tx-negative", "tz-positive", "tz-negative", "rx-positive", "rx-negative")
METHOD_SLOTS: Mapping[str, tuple[int, ...]] = {
    "zero": (1, 2), "mean_constant": (1,), "open_loop_playback": (1,),
    "classical_force_search": (1,), "oracle_gt_reference": (1,),
}
FAIR_PROVIDER_INPUTS = frozenset(("observation79", "provider_owned_state"))
FEATURE_INDEX = np.r_[7:40, 63:78]
EXPECTED_SOURCE_TRACE_GZIP_SHA256 = "3ef8a4456bdfedec12996d1f78a6dc2736347a771be53137cab4263353920f59"
EXPECTED_MEAN_CONSTANT_ACTION = (0.034892102401539844, -0.010848934130210961, -0.041849699608244736, 0.03485121803133042, -0.07816294091497804, -0.02233053562947764)
EXPECTED_PLAYBACK_ROWS = 364
CLASSICAL_CANDIDATE_VALUES = {"A": (0.8, 0.8, 8.0, 0.00015), "B": (1.0, 0.8, 6.0, 0.00010), "C": (0.6, 1.0, 8.0, 0.00020)}


@dataclass(frozen=True)
class EpisodeResult:
    grasp_id: str
    case_id: str
    method_id: str
    slot: int
    success: bool
    hard_contact: bool | None
    forbidden_contact: bool | None
    evaluator_force_p95_n: float | None
    completion_time_s: float | None
    provider_inputs: frozenset[str] | None
    applied_controller_sha256: str | None
    applied_artifact_sha256: str | None
    observation_samples: tuple[tuple[float, ...], ...] = ()

    @property
    def key(self) -> tuple[str, str, str, int]:
        return self.grasp_id, self.case_id, self.method_id, self.slot


@dataclass(frozen=True)
class ContractEvidence:
    archive_hash_valid: bool
    held_identity_valid: bool
    requested_applied_bias_valid: bool
    prefreeze_reset_equivalent: bool
    insert_transition_equivalent: bool
    first_insert_physical_zero: bool
    telemetry_complete: bool
    snapshots_complete: bool
    finite_values: bool
    baseline_artifacts_verified: bool

    def failures(self) -> tuple[str, ...]:
        return tuple(name for name, value in vars(self).items() if value is not True)


@dataclass(frozen=True)
class TuningResult:
    candidate_id: str
    case_id: str
    success: bool
    hard_contact: bool
    forbidden_contact: bool
    evaluator_force_p95_n: float
    completion_time_s: float
    grasp_id: str = "rj45-train-03"


@dataclass(frozen=True)
class TuningSelection:
    candidate_id: str
    success_count: int
    evaluator_force_p95_n: float
    completion_time_median_s: float
    met_target: bool


@dataclass(frozen=True)
class ClassicalSeal:
    selected_candidate_id: str
    canonical_controller_config_sha256: str
    tuning_report_sha256: str
    tuning_report_bytes: bytes
    selection: TuningSelection
    tuning_results: tuple[TuningResult, ...]


@dataclass(frozen=True)
class BaselineArtifactSeal:
    source_trace_gzip_sha256: str
    mean_constant_action: tuple[float, ...]
    playback_selected_rows: int
    mean_constant_artifact_sha256: str
    playback_artifact_sha256: str


@dataclass(frozen=True)
class ObservabilityAxis:
    grasp_id: str
    pair: str
    signed_distance: float
    repeat_noise: float
    ratio: float
    observable: bool


@dataclass(frozen=True)
class AdmissionDecision:
    state: str
    reasons: tuple[str, ...]
    success_counts: Mapping[str, int]
    observability: tuple[ObservabilityAxis, ...]


def required_admission_slots() -> tuple[tuple[str, str, str, int], ...]:
    return tuple((grasp, case, method, slot) for grasp in GRASPS for case in CASES for method, slots in METHOD_SLOTS.items() for slot in slots)


def _canonical_json_sha(payload: Mapping[str, object]) -> str:
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _classical_controller_payload(candidate_id: str) -> Mapping[str, object]:
    position_gain, rotation_gain, force_trigger_n, search_step_m = CLASSICAL_CANDIDATE_VALUES[candidate_id]
    return {
        "schema": "rj45-classical-force-search-v1", "candidate_id": candidate_id,
        "position_gain": position_gain, "rotation_gain": rotation_gain,
        "force_trigger_n": force_trigger_n, "search_step_m": search_step_m,
        "stall_ticks": 6, "stall_depth_delta_per_tick_m": 0.00002,
        "controlled_translation_axes": ["x", "z"], "controlled_rotation_axes": ["x", "z"],
        "lateral_force_direction_threshold_n": 0.25, "ticks_per_search_direction": 6,
        "search_direction_order": ["initial", "rotate_plus_90", "opposite", "rotate_minus_90"],
        "search_radius_max_m": 0.0006, "advance_recovery_ticks": 3,
        "zero_action_entry_force_norm_n": 20.0, "translation_bound_m": 0.0015,
        "hard_stop_evaluator_force_n": 25.0, "rotation_bound_rad": 0.026179938779914945,
        "final_component_clip": [-1.0, 1.0], "known_bias_access": False,
        "observation_dim": 79, "action_dim": 6,
        "entry_force_slice": [37, 40], "entry_force_scale_n": 8.0,
        "command_depth_index": 62, "command_depth_scale_m": 0.012, "advance_index": 63,
    }


def select_classical_candidate(results: Sequence[TuningResult]) -> TuningSelection:
    expected = {(candidate, case) for candidate in ("A", "B", "C") for case in CASES}
    keys = [(result.candidate_id, result.case_id) for result in results]
    if len(results) != 18 or len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("tuning must contain exactly the sealed 18 candidate/case slots")
    if any(result.grasp_id != "rj45-train-03" for result in results):
        raise ValueError("tuning is restricted to independent grasp rj45-train-03")
    eligible = []
    for candidate in ("A", "B", "C"):
        rows = [row for row in results if row.candidate_id == candidate]
        if any(row.hard_contact or row.forbidden_contact for row in rows):
            continue
        forces = np.asarray([row.evaluator_force_p95_n for row in rows], dtype=np.float64)
        times = np.asarray([row.completion_time_s for row in rows], dtype=np.float64)
        if not np.isfinite(forces).all() or not np.isfinite(times).all():
            raise ValueError("tuning metrics must be finite")
        successes = sum(row.success for row in rows)
        eligible.append(TuningSelection(candidate, successes, float(np.percentile(forces, 95, method="linear")), float(median(times)), successes >= 4))
    if not eligible:
        raise ValueError("no tuning candidate has zero hard and forbidden contacts")
    return min(eligible, key=lambda item: (-item.success_count, item.evaluator_force_p95_n, item.completion_time_median_s, item.candidate_id))


def canonical_tuning_report_bytes(results: Sequence[TuningResult]) -> bytes:
    selection = select_classical_candidate(results)
    ordered = sorted(results, key=lambda row: (row.candidate_id, CASES.index(row.case_id)))
    payload = {
        "schema": "rj45-classical-tuning-report-v1",
        "selected_candidate_id": selection.candidate_id,
        "selection": {
            "success_count": selection.success_count,
            "evaluator_force_p95_n": selection.evaluator_force_p95_n,
            "completion_time_median_s": selection.completion_time_median_s,
            "met_target": selection.met_target,
        },
        "episodes": [vars(row) for row in ordered],
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def seal_classical_selection(selection: TuningSelection, tuning_report: bytes, tuning_results: Sequence[TuningResult]) -> ClassicalSeal:
    selected_from_results = select_classical_candidate(tuning_results)
    expected_report = canonical_tuning_report_bytes(tuning_results)
    if selection != selected_from_results or tuning_report != expected_report:
        raise ValueError("selection and report bytes must match the exact 18 tuning results")
    return ClassicalSeal(selection.candidate_id, _canonical_json_sha(_classical_controller_payload(selection.candidate_id)), sha256(tuning_report).hexdigest(), bytes(tuning_report), selection, tuple(tuning_results))


def validate_classical_seal(seal: ClassicalSeal) -> bool:
    try:
        selected = select_classical_candidate(seal.tuning_results)
        expected_report = canonical_tuning_report_bytes(seal.tuning_results)
    except (TypeError, ValueError):
        return False
    return seal.selected_candidate_id in CLASSICAL_CANDIDATE_VALUES and seal.selection == selected and selected.candidate_id == seal.selected_candidate_id and seal.canonical_controller_config_sha256 == _canonical_json_sha(_classical_controller_payload(seal.selected_candidate_id)) and seal.tuning_report_bytes == expected_report and seal.tuning_report_sha256 == sha256(expected_report).hexdigest()


def make_baseline_artifact_seal(*, source_trace_gzip_sha256: str, mean_constant_action: Sequence[float], playback_selected_rows: int) -> BaselineArtifactSeal:
    action = tuple(float(value) for value in mean_constant_action)
    if source_trace_gzip_sha256 != EXPECTED_SOURCE_TRACE_GZIP_SHA256 or action != EXPECTED_MEAN_CONSTANT_ACTION or playback_selected_rows != EXPECTED_PLAYBACK_ROWS:
        raise ValueError("baseline evidence does not match the fixed development artifact contract")
    common = {"source_trace_gzip_sha256": source_trace_gzip_sha256}
    mean_sha = _canonical_json_sha({**common, "method": "mean_constant", "normalized_right6": action})
    playback_sha = _canonical_json_sha({**common, "method": "open_loop_playback", "selected_rows": playback_selected_rows, "tail": "hold_final_action"})
    return BaselineArtifactSeal(source_trace_gzip_sha256, action, playback_selected_rows, mean_sha, playback_sha)


def validate_baseline_artifact_seal(seal: BaselineArtifactSeal) -> bool:
    try:
        return seal == make_baseline_artifact_seal(source_trace_gzip_sha256=seal.source_trace_gzip_sha256, mean_constant_action=seal.mean_constant_action, playback_selected_rows=seal.playback_selected_rows)
    except ValueError:
        return False


def validate_admission_contract(episodes: Sequence[EpisodeResult], evidence: ContractEvidence, *, classical_seal: ClassicalSeal | None, baseline_seal: BaselineArtifactSeal | None) -> tuple[str, ...]:
    errors = list(evidence.failures())
    expected, actual = set(required_admission_slots()), [episode.key for episode in episodes]
    if len(episodes) != 72: errors.append("admission_slot_count_must_be_72")
    if len(actual) != len(set(actual)): errors.append("duplicate_admission_slot")
    if expected - set(actual): errors.append(f"missing_requested_slots:{len(expected - set(actual))}")
    if set(actual) - expected: errors.append(f"unexpected_slots:{len(set(actual) - expected)}")
    if classical_seal is None or not validate_classical_seal(classical_seal): errors.append("classical_tuning_not_sealed")
    if baseline_seal is None or not validate_baseline_artifact_seal(baseline_seal): errors.append("baseline_artifacts_not_sealed")
    for episode in episodes:
        if not isinstance(episode.success, bool) or not isinstance(episode.hard_contact, bool) or not isinstance(episode.forbidden_contact, bool): errors.append(f"missing_or_invalid_safety_result:{episode.key}")
        metrics = (episode.evaluator_force_p95_n, episode.completion_time_s)
        if any(value is None for value in metrics) or not np.isfinite(np.asarray(metrics, dtype=np.float64)).all(): errors.append(f"nonfinite_episode:{episode.key}")
        if episode.method_id != "oracle_gt_reference" and episode.provider_inputs != FAIR_PROVIDER_INPUTS: errors.append(f"provider_authority:{episode.key}")
        if episode.method_id == "classical_force_search" and (classical_seal is None or episode.applied_controller_sha256 != classical_seal.canonical_controller_config_sha256): errors.append(f"classical_controller_seal:{episode.key}")
        if episode.method_id == "mean_constant" and (baseline_seal is None or episode.applied_artifact_sha256 != baseline_seal.mean_constant_artifact_sha256): errors.append(f"mean_constant_artifact_seal:{episode.key}")
        if episode.method_id == "open_loop_playback" and (baseline_seal is None or episode.applied_artifact_sha256 != baseline_seal.playback_artifact_sha256): errors.append(f"playback_artifact_seal:{episode.key}")
    return tuple(dict.fromkeys(errors))


def _aligned_feature_median(samples: Sequence[Sequence[float]]) -> np.ndarray | None:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 79 or not np.isfinite(array).all(): return None
    contacts = np.flatnonzero(np.linalg.norm(array[:, 37:40] * 8.0, axis=1) >= 0.25)
    if not contacts.size: return None
    window = array[int(contacts[0]):int(contacts[0]) + 13]
    return np.median(window[:, FEATURE_INDEX], axis=0) if window.shape[0] == 13 else None


def compute_observability(episodes: Sequence[EpisodeResult]) -> tuple[ObservabilityAxis, ...]:
    indexed, results = {episode.key: episode for episode in episodes}, []
    for grasp in GRASPS:
        for pair in ("tx", "tz", "rx"):
            summaries = {}
            for sign in ("positive", "negative"):
                for slot in (1, 2):
                    episode = indexed.get((grasp, f"{pair}-{sign}", "zero", slot))
                    summaries[(sign, slot)] = None if episode is None else _aligned_feature_median(episode.observation_samples)
            if any(value is None for value in summaries.values()):
                results.append(ObservabilityAxis(grasp, pair, 0.0, 1e-6, 0.0, False)); continue
            p1, p2, n1, n2 = summaries[("positive", 1)], summaries[("positive", 2)], summaries[("negative", 1)], summaries[("negative", 2)]
            distance = float(np.linalg.norm(p1 - n1)); noise = max(float(np.linalg.norm(p1 - p2)), float(np.linalg.norm(n1 - n2)), 1e-6); ratio = distance / noise
            results.append(ObservabilityAxis(grasp, pair, distance, noise, ratio, distance >= 0.05 and ratio >= 2.0))
    return tuple(results)


def adjudicate_admission(episodes: Sequence[EpisodeResult], evidence: ContractEvidence, *, classical_seal: ClassicalSeal | None, baseline_seal: BaselineArtifactSeal | None) -> AdmissionDecision:
    errors = validate_admission_contract(episodes, evidence, classical_seal=classical_seal, baseline_seal=baseline_seal)
    observability = compute_observability(episodes)
    if errors: return AdmissionDecision("invalid_contract", errors, {}, observability)
    indexed = {episode.key: episode for episode in episodes}
    safe_counts = {method: sum(indexed[(grasp, case, method, 1)].success and indexed[(grasp, case, method, 1)].hard_contact is False and indexed[(grasp, case, method, 1)].forbidden_contact is False for grasp in GRASPS for case in CASES) for method in METHOD_SLOTS}
    reference_rows = [row for row in episodes if row.method_id == "oracle_gt_reference"]
    per_grasp = {grasp: sum(row.success and row.hard_contact is False and row.forbidden_contact is False for row in reference_rows if row.grasp_id == grasp) for grasp in GRASPS}
    observable = all(sum(row.observable for row in observability if row.grasp_id == grasp) >= 2 for grasp in GRASPS)
    if safe_counts["oracle_gt_reference"] < 10 or min(per_grasp.values()) < 5 or any(row.hard_contact or row.forbidden_contact for row in reference_rows) or not observable: return AdmissionDecision("infeasible_or_unobservable", ("reference_or_observability_gate_failed",), safe_counts, observability)
    nonlearning = ("zero", "mean_constant", "open_loop_playback", "classical_force_search")
    if any(safe_counts[method] >= 10 and not any(row.hard_contact or row.forbidden_contact for row in episodes if row.method_id == method and row.slot == 1) for method in nonlearning): return AdmissionDecision("valid_stop_no_headroom", ("safe_nonlearning_method_reached_10_of_12",), safe_counts, observability)
    if safe_counts["zero"] <= 1: return AdmissionDecision("valid_hard_sparse", ("zero_slot1_successes_below_2",), safe_counts, observability)
    headroom, pairs = 0, []
    for grasp in GRASPS:
        for case in CASES:
            ref = indexed[(grasp, case, "oracle_gt_reference", 1)]
            headroom += int(ref.success and not ref.hard_contact and not ref.forbidden_contact and all(not indexed[(grasp, case, method, 1)].success for method in nonlearning))
        for pair in ("tx", "tz", "rx"):
            if all(not indexed[(grasp, f"{pair}-{sign}", "zero", 1)].success and indexed[(grasp, f"{pair}-{sign}", "oracle_gt_reference", 1)].success for sign in ("positive", "negative")): pairs.append((grasp, pair))
    types, grasps = {"orientation" if pair == "rx" else "position" for _, pair in pairs}, {grasp for grasp, _ in pairs}
    if all(safe_counts[method] < 10 for method in nonlearning) and headroom >= 2 and len(pairs) >= 2 and types == {"position", "orientation"} and grasps == set(GRASPS): return AdmissionDecision("admit_rl_pilot", (), safe_counts, observability)
    return AdmissionDecision("valid_stop_insufficient_structured_headroom", ("insufficient_structured_headroom",), safe_counts, observability)


class ProtocolStage(str, Enum):
    CREATED = "created"; TUNING_VALIDATED = "tuning_validated"; CLASSICAL_SEALED = "classical_sealed"; ADMISSION_VALIDATED = "admission_validated"; ADJUDICATED = "adjudicated"


@dataclass(frozen=True)
class AdmissionProtocol:
    stage: ProtocolStage = ProtocolStage.CREATED
    selection: TuningSelection | None = None
    classical_seal: ClassicalSeal | None = None
    baseline_seal: BaselineArtifactSeal | None = None
    contract_errors: tuple[str, ...] = ()
    decision: AdmissionDecision | None = None

    def validate_tuning(self, results: Sequence[TuningResult]) -> "AdmissionProtocol":
        self._require(ProtocolStage.CREATED); return replace(self, stage=ProtocolStage.TUNING_VALIDATED, selection=select_classical_candidate(results))
    def seal_classical(self, tuning_report: bytes, tuning_results: Sequence[TuningResult]) -> "AdmissionProtocol":
        self._require(ProtocolStage.TUNING_VALIDATED); assert self.selection is not None
        return replace(self, stage=ProtocolStage.CLASSICAL_SEALED, classical_seal=seal_classical_selection(self.selection, tuning_report, tuning_results))
    def validate_admission(self, episodes: Sequence[EpisodeResult], evidence: ContractEvidence, baseline_seal: BaselineArtifactSeal) -> "AdmissionProtocol":
        self._require(ProtocolStage.CLASSICAL_SEALED)
        errors = validate_admission_contract(episodes, evidence, classical_seal=self.classical_seal, baseline_seal=baseline_seal)
        return replace(self, stage=ProtocolStage.ADMISSION_VALIDATED, baseline_seal=baseline_seal, contract_errors=errors)
    def adjudicate(self, episodes: Sequence[EpisodeResult], evidence: ContractEvidence) -> "AdmissionProtocol":
        self._require(ProtocolStage.ADMISSION_VALIDATED)
        return replace(self, stage=ProtocolStage.ADJUDICATED, decision=adjudicate_admission(episodes, evidence, classical_seal=self.classical_seal, baseline_seal=self.baseline_seal))
    def _require(self, expected: ProtocolStage) -> None:
        if self.stage is not expected: raise RuntimeError(f"protocol stage must be {expected.value}, got {self.stage.value}")
