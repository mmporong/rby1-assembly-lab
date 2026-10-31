#!/usr/bin/env python3
"""Audit the recovered frozen-ablation evidence using only the standard library."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import sys
from typing import Any, Mapping, Sequence


VALID_CLASSIFICATIONS = {"valid_success", "valid_failure"}
MODES = ("standard79", "zero", "mean_constant", "open_loop_playback")


class AuditError(RuntimeError):
    pass


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"cannot load JSON object: {path}") from error
    if not isinstance(value, dict):
        raise AuditError(f"JSON root must be an object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def finite_vector(value: Any, length: int, field: str) -> list[float]:
    require(isinstance(value, list) and len(value) == length, f"{field} must have {length} values")
    vector = [float(item) for item in value]
    require(all(math.isfinite(item) for item in vector), f"{field} contains non-finite values")
    return vector


def close_vectors(left: Sequence[float], right: Sequence[float], tolerance: float = 1.0e-6) -> bool:
    return len(left) == len(right) and all(
        math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
        for a, b in zip(left, right)
    )


def raw_base79_sha256(values: Sequence[float]) -> str:
    require(len(values) == 79, "base79 must contain 79 values")
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(json.dumps([1, 79], separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(struct.pack("<79f", *(float(value) for value in values)))
    return digest.hexdigest()


def canonical_actions_payload(actions: Sequence[Sequence[float]]) -> dict[str, Any]:
    rows = [[float(value) for value in row] for row in actions]
    require(bool(rows), "source action sequence is empty")
    return {
        "actions": rows,
        "last": rows[-1],
        "mean": [math.fsum(row[index] for row in rows) / len(rows) for index in range(6)],
    }


def canonical_actions_sha256(actions: Sequence[Sequence[float]]) -> str:
    encoded = (
        json.dumps(
            canonical_actions_payload(actions),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_trace(path: Path) -> tuple[list[dict[str, Any]], str]:
    try:
        raw = gzip.decompress(path.read_bytes())
    except (OSError, EOFError) as error:
        raise AuditError(f"invalid gzip trace: {path}") from error
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuditError(f"invalid trace JSON at {path}:{number}") from error
        require(isinstance(row, dict), f"trace row is not an object: {path}:{number}")
        rows.append(row)
    require(bool(rows), f"trace is empty: {path}")
    events = [index for index, row in enumerate(rows) if "event" in row]
    require(events == [len(rows) - 1] and rows[-1].get("event") == "complete", f"trace must have one final complete event: {path}")
    for number, row in enumerate(rows[:-1], 1):
        require("event" not in row, f"unexpected event at {path}:{number}")
        require(isinstance(row.get("phase"), str), f"phase must be a string at {path}:{number}")
        require(type(row.get("active")) is bool, f"active must be boolean at {path}:{number}")
        require(type(row.get("policy_input_valid")) is bool, f"policy_input_valid must be boolean at {path}:{number}")
    return rows, hashlib.sha256(raw).hexdigest()


def expected_execution_order(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    modes = contract.get("mode_order")
    case_order = contract.get("case_order")
    require(modes == list(MODES), "contract mode order changed")
    require(isinstance(case_order, list) and len(case_order) == 5, "contract must contain five ordered cases")
    order = []
    for case_index, case_id in enumerate(case_order):
        rotated = modes[case_index:] + modes[:case_index]
        for mode in rotated:
            order.append({"index": len(order) + 1, "case_id": case_id, "ablation_mode": mode})
    return order


def indexed_json_files(directory: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            index = int(path.name.split("-", 1)[0])
        except (ValueError, IndexError) as error:
            raise AuditError(f"episode filename has no numeric prefix: {path.name}") from error
        require(index not in files, f"duplicate episode index: {index}")
        files[index] = path
    return files


def indexed_trace_files(directory: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(directory.glob("*.jsonl.gz")):
        try:
            index = int(path.name.split("-", 1)[0])
        except (ValueError, IndexError) as error:
            raise AuditError(f"trace filename has no numeric prefix: {path.name}") from error
        require(index not in files, f"duplicate trace index: {index}")
        files[index] = path
    return files


def audit_exit_files(root: Path) -> dict[str, int]:
    exits = {}
    for name in ("batch.raw.exit", "batch.validated.exit"):
        text = (root / name).read_text(encoding="utf-8").strip()
        require(text == "0", f"{name} is not zero")
        exits[name] = int(text)
    return exits


def audit_source_actions(
    contract: Mapping[str, Any], source_record: Mapping[str, Any]
) -> tuple[list[list[float]], dict[str, Any]]:
    analysis = contract.get("public_input_analysis")
    require(isinstance(analysis, dict), "contract has no public_input_analysis")
    actions_raw = source_record.get("actions")
    require(isinstance(actions_raw, list), "source-actions.json has no actions")
    actions = [finite_vector(row, 6, f"source action {index}") for index, row in enumerate(actions_raw)]
    payload = canonical_actions_payload(actions)
    canonical_sha = canonical_actions_sha256(actions)
    require(len(actions) == analysis.get("source_action_count"), "source action count differs from contract")
    require(analysis.get("source_action_dimension") == 6, "source action dimension differs from contract")
    require(close_vectors(payload["mean"], finite_vector(analysis.get("source_action_mean"), 6, "contract source mean"), 1.0e-15), "source action mean differs from contract")
    require(payload["last"] == finite_vector(analysis.get("source_action_last"), 6, "contract source last"), "source action tail differs from contract")
    require(canonical_sha == analysis.get("source_actions_canonical_sha256"), "canonical source action SHA differs from contract")
    require(source_record.get("canonical_sha256") == canonical_sha, "source-actions record canonical SHA differs")
    require(source_record.get("mean") == payload["mean"] and source_record.get("last") == payload["last"], "source-actions summary differs from its rows")
    source_trace = source_record.get("source_trace")
    require(isinstance(source_trace, dict) and source_trace.get("sha256") == contract.get("origin", {}).get("source_trace_gzip_sha256"), "source-actions origin trace binding mismatch")
    return actions, {"count": len(actions), "canonical_sha256": canonical_sha}


def trace_terminal(rows: Sequence[Mapping[str, Any]], path: Path) -> Mapping[str, Any]:
    terminal = [row for row in rows if row.get("event") == "complete"]
    require(len(terminal) == 1 and terminal[0] is rows[-1], f"trace terminal event is invalid: {path}")
    return terminal[0]


def audit_actions(
    *, mode: str, rows: Sequence[Mapping[str, Any]], source_actions: Sequence[Sequence[float]],
    source_mean: Sequence[float], telemetry: Any,
) -> dict[str, int]:
    active_rows = 0
    playback_index = 0
    tail_holds = 0
    for row_number, row in enumerate(rows, 1):
        if "event" in row:
            continue
        applied = finite_vector(row.get("applied_expanded_action"), 12, f"trace row {row_number} applied action")
        require(applied[:6] == [0.0] * 6, f"left arm action is non-zero at trace row {row_number}")
        eligible = row.get("active") is True and row.get("policy_input_valid") is True
        if eligible:
            active_rows += 1
        if mode == "zero":
            require(applied[6:] == [0.0] * 6, f"zero mode applied a non-zero right action at row {row_number}")
        elif mode == "mean_constant":
            expected = source_mean if eligible else [0.0] * 6
            require(close_vectors(applied[6:], expected), f"mean-constant applied action mismatch at row {row_number}")
        elif mode == "open_loop_playback":
            if eligible:
                expected = source_actions[min(playback_index, len(source_actions) - 1)]
                if playback_index >= len(source_actions):
                    tail_holds += 1
                playback_index += 1
            else:
                expected = [0.0] * 6
            require(close_vectors(applied[6:], expected), f"playback applied action mismatch at row {row_number}")
        elif not eligible:
            require(applied == [0.0] * 12, f"inactive PPO row applied a non-zero action at row {row_number}")
    if mode != "standard79":
        require(isinstance(telemetry, dict), f"{mode} episode has no policy telemetry")
        require(telemetry.get("eligible_call_count") == active_rows, f"{mode} eligibility counter mismatch")
    if mode == "open_loop_playback":
        require(telemetry.get("source_rows_consumed") == min(active_rows, len(source_actions)), "playback consumed-row counter mismatch")
        require(telemetry.get("tail_hold_count") == tail_holds, "playback tail counter mismatch")
    return {"active_rows": active_rows, "playback_tail_holds": tail_holds}


def audit_episode(
    *, expected: Mapping[str, Any], episode_path: Path, trace_path: Path,
    report_episode: Mapping[str, Any], case: Mapping[str, Any], source_actions: Sequence[Sequence[float]],
    source_mean: Sequence[float], bias_case: Mapping[str, Any], paired: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    episode = load_json(episode_path)
    index, mode, case_id = expected["index"], expected["ablation_mode"], expected["case_id"]
    require(episode.get("index") == index, f"episode {index} index mismatch")
    require(episode.get("ablation_mode") == mode, f"episode {index} mode mismatch")
    require(episode.get("case") == case, f"episode {index} case lineage mismatch")
    require(report_episode.get("index") == index and report_episode.get("ablation_mode") == mode, f"report episode {index} identity mismatch")
    require(report_episode.get("case") == case, f"report episode {index} case mismatch")
    require(report_episode == episode, f"report episode {index} differs from the recovered episode JSON")
    evaluation = episode.get("evaluation")
    require(isinstance(evaluation, dict), f"episode {index} has no evaluation")
    classification = evaluation.get("classification")
    require(classification in VALID_CLASSIFICATIONS, f"episode {index} classification is not valid")
    require(report_episode.get("evaluation", {}).get("classification") == classification, f"report episode {index} classification mismatch")
    rows, uncompressed_sha = load_trace(trace_path)
    require(evaluation.get("trace", {}).get("sha256") == uncompressed_sha, f"episode {index} original trace SHA mismatch")
    require(trace_terminal(rows, trace_path).get("classification") == classification, f"episode {index} terminal classification mismatch")
    terminal = evaluation.get("terminal")
    require(isinstance(terminal, dict) and terminal.get("valid") is True, f"episode {index} terminal is not valid")
    expected_success = classification == "valid_success"
    require(terminal.get("success") is expected_success, f"episode {index} terminal success differs from classification")
    terminal_phase = evaluation.get("terminal_phase")
    require(rows[-2].get("phase") == terminal_phase, f"episode {index} final physical phase differs from terminal_phase")
    if expected_success:
        require(terminal_phase == "done", f"successful episode {index} did not finish in done phase")
    action_result = audit_actions(
        mode=mode, rows=rows, source_actions=source_actions,
        source_mean=source_mean, telemetry=episode.get("policy_telemetry"),
    )
    require(evaluation.get("active_rows") == action_result["active_rows"], f"episode {index} active-row count mismatch")
    fixed = episode.get("fixed_condition")
    require(isinstance(fixed, dict), f"episode {index} has no fixed-condition record")
    require(fixed.get("base_seed") == case.get("base_seed"), f"episode {index} base seed mismatch")
    require(fixed.get("grasp_id") == case.get("grasp_id") and fixed.get("grasp_index") == 1, f"episode {index} grasp mismatch")
    bias = fixed.get("bias")
    require(isinstance(bias, dict), f"episode {index} has no bias record")
    require(bias.get("episode_index") == [case.get("episode_index")], f"episode {index} episode index mismatch")
    require(bias.get("base_seed") == case.get("base_seed"), f"episode {index} bias seed mismatch")
    require(bias.get("translation_entry_m") == [bias_case.get("translation_entry_m")], f"episode {index} translation bias mismatch")
    require(bias.get("rotation_vector_entry_rad") == [bias_case.get("rotation_vector_entry_rad")], f"episode {index} rotation bias mismatch")
    first_rows = [row for row in rows if row.get("active") is True and row.get("policy_input_valid") is True]
    require(bool(first_rows), f"episode {index} has no active valid observation")
    base79 = finite_vector(first_rows[0].get("observation", {}).get("base79"), 79, f"episode {index} first base79")
    require(base79 == finite_vector(fixed.get("initial_base79"), 79, f"episode {index} recorded base79"), f"episode {index} raw base79 mismatch")
    require(raw_base79_sha256(base79) == fixed.get("initial_base79_sha256"), f"episode {index} base79 SHA mismatch")
    observation = evaluation.get("observation")
    require(isinstance(observation, dict), f"episode {index} observation proof is missing")
    dimensions = observation.get("dimensions")
    require(dimensions == {"policy": 79, "critic": 92}, f"episode {index} observation dimensions changed")
    require(observation.get("actor_ground_truth_access") is False, f"episode {index} actor ground-truth access is not false")
    require(observation.get("base79_sha256") == fixed.get("initial_base79_sha256"), f"episode {index} observation/base79 SHA mismatch")
    require(observation.get("policy_sha256") == observation.get("base79_sha256"), f"episode {index} policy/base79 SHA mismatch")
    require(fixed.get("evaluation_rng") == evaluation.get("evaluation_rng"), f"episode {index} RNG record mismatch")
    paired_value = {"base79": base79, "base79_sha256": fixed.get("initial_base79_sha256"), "rng": fixed.get("evaluation_rng"), "bias": bias, "grasp_id": fixed.get("grasp_id")}
    if case_id in paired:
        require(paired[case_id] == paired_value, f"case {case_id} is not paired identically across modes")
    else:
        paired[case_id] = paired_value
    return {
        "index": index, "case_id": case_id, "mode": mode,
        "classification": classification, "trace_sha256": uncompressed_sha,
        **action_result,
    }


def audit_report_lineage(
    *, root: Path, contract: Mapping[str, Any], report: Mapping[str, Any],
    source_record: Mapping[str, Any], bias_record: Mapping[str, Any],
) -> dict[str, Any]:
    require(report.get("status") == "completed", "results report is not completed")
    require(report.get("learning_executed") is False, "results report claims learning execution")
    require(report.get("checkpoint_selection_changed") is False, "checkpoint selection changed")
    contract_binding = report.get("contract")
    require(isinstance(contract_binding, dict) and contract_binding.get("sha256") == file_sha256(root / "contract.json"), "report contract SHA mismatch")
    source_binding = report.get("source_binding")
    require(isinstance(source_binding, dict), "report has no source binding")
    origin = contract.get("origin")
    sources = contract.get("source_sha256")
    require(source_binding.get("checkpoint", {}).get("sha256") == origin.get("checkpoint_sha256"), "checkpoint lineage mismatch")
    require(source_binding.get("protocol", {}).get("sha256") == origin.get("protocol_sha256"), "protocol lineage mismatch")
    require(source_binding.get("report", {}).get("sha256") == origin.get("report_sha256"), "origin report lineage mismatch")
    require(source_binding.get("source_trace_gzip", {}).get("sha256") == origin.get("source_trace_gzip_sha256"), "source trace lineage mismatch")
    for name in ("runner", "replay_helper", "g2_helper", "runtime_manifest"):
        require(source_binding.get(name, {}).get("sha256") == sources.get(name), f"{name} source lineage mismatch")
    runtime_manifest = report.get("runtime_manifest")
    require(isinstance(runtime_manifest, dict) and runtime_manifest.get("file_sha256") == sources.get("runtime_manifest"), "runtime manifest record mismatch")
    require(runtime_manifest.get("source_file_count") == 35, "runtime manifest source count changed")
    imports = report.get("runtime_import_origins")
    require(isinstance(imports, dict) and bool(imports), "runtime import origins are missing")
    action_binding = report.get("source_action_analysis")
    require(isinstance(action_binding, dict) and action_binding.get("sha256") == file_sha256(root / "source-actions.json"), "source-actions artifact SHA mismatch")
    require(action_binding.get("canonical_sha256") == source_record.get("canonical_sha256"), "source-actions canonical report binding mismatch")
    bias_binding = report.get("bias_preflight")
    require(isinstance(bias_binding, dict) and bias_binding.get("sha256") == file_sha256(root / "bias-cases.json"), "bias artifact SHA mismatch")
    require(bias_binding.get("runtime_source_canonical_sha256") == bias_record.get("runtime_source_canonical_sha256"), "bias runtime-source binding mismatch")
    return {
        "checkpoint_sha256": origin.get("checkpoint_sha256"),
        "protocol_sha256": origin.get("protocol_sha256"),
        "runtime_manifest_sha256": sources.get("runtime_manifest"),
        "runtime_source_file_count": runtime_manifest.get("source_file_count"),
    }


def run_audit(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = load_json(root / "contract.json")
    admission = load_json(root / "execution-admission.json")
    report = load_json(root / "results-report.json")
    source_record = load_json(root / "source-actions.json")
    bias_record = load_json(root / "bias-cases.json")
    exits = audit_exit_files(root)
    contract_sha = file_sha256(root / "contract.json")
    require(admission.get("contract_sha256") == contract_sha, "execution admission contract SHA mismatch")
    require(admission.get("helper_sha256") == contract.get("helper_sha256"), "execution admission helper SHA mismatch")
    require(admission.get("cap_usd") == contract.get("budget_and_stop", {}).get("budget_cap_usd"), "execution admission budget cap mismatch")
    require(admission.get("reservation_usd") == contract.get("budget_and_stop", {}).get("reservation_usd"), "execution admission reservation mismatch")
    require(admission.get("deadline_utc") == contract.get("budget_and_stop", {}).get("deadline_utc"), "execution admission deadline mismatch")
    helper_path = root.parents[2] / "scripts/evaluate_rj45_frozen_ablation.py"
    require(helper_path.is_file() and file_sha256(helper_path) == contract.get("helper_sha256"), "local frozen helper SHA mismatch")
    source_actions, source_summary = audit_source_actions(contract, source_record)
    source_mean = canonical_actions_payload(source_actions)["mean"]
    expected_order = expected_execution_order(contract)
    require(report.get("execution_order") == expected_order, "report execution order differs from cyclic contract order")
    report_episodes = report.get("episodes")
    require(isinstance(report_episodes, list) and len(report_episodes) == 20, "report must contain 20 episodes")
    episode_files = indexed_json_files(root / "episodes")
    trace_files = indexed_trace_files(root / "traces")
    require(set(episode_files) == set(range(1, 21)), "episode file set is not exactly 1..20")
    require(set(trace_files) == set(range(1, 21)), "trace file set is not exactly 1..20")
    cases = contract.get("cases")
    require(isinstance(cases, list) and len(cases) == 5, "contract cases are invalid")
    case_by_id = {case.get("case_id"): case for case in cases if isinstance(case, dict)}
    require(set(case_by_id) == set(contract.get("case_order", [])), "contract case inventory differs from case order")
    bias_cases = bias_record.get("cases")
    require(isinstance(bias_cases, list) and len(bias_cases) == 5, "bias preflight must contain five cases")
    bias_by_id = {case.get("case_id"): case for case in bias_cases if isinstance(case, dict)}
    require(set(bias_by_id) == set(case_by_id), "bias case inventory differs from contract")
    paired: dict[str, dict[str, Any]] = {}
    episode_results = []
    for expected in expected_order:
        index, case_id = expected["index"], expected["case_id"]
        episode_results.append(audit_episode(
            expected=expected, episode_path=episode_files[index], trace_path=trace_files[index],
            report_episode=report_episodes[index - 1], case=case_by_id[case_id],
            source_actions=source_actions, source_mean=source_mean,
            bias_case=bias_by_id[case_id], paired=paired,
        ))
    require(len(paired) == 5, "paired-condition inventory is incomplete")
    lineage = audit_report_lineage(
        root=root, contract=contract, report=report,
        source_record=source_record, bias_record=bias_record,
    )
    classifications = {name: 0 for name in sorted(VALID_CLASSIFICATIONS)}
    for episode in episode_results:
        classifications[episode["classification"]] += 1
    anchor_id = contract["case_order"][0]
    by_mode: dict[str, Any] = {}
    for mode in MODES:
        selected = [episode for episode in episode_results if episode["mode"] == mode]
        anchor = [episode for episode in selected if episode["case_id"] == anchor_id]
        transfer = [episode for episode in selected if episode["case_id"] != anchor_id]
        require(len(anchor) == 1 and len(transfer) == 4 and len(selected) == 5, f"{mode} anchor/transfer inventory changed")
        counts = lambda rows: {
            "success": sum(row["classification"] == "valid_success" for row in rows),
            "failure": sum(row["classification"] == "valid_failure" for row in rows),
            "total": len(rows),
        }
        by_mode[mode] = {"anchor": counts(anchor), "transfer": counts(transfer), "overall": counts(selected)}
    audit = {
        "schema_name": "rj45_frozen_ablation_independent_audit_v1",
        "status": "completed",
        "root": str(root),
        "execution_order": expected_order,
        "episodes": episode_results,
        "paired_case_count": len(paired),
        "source_actions": source_summary,
        "lineage": lineage,
        "exit_codes": exits,
        "checks": {
            "all_twenty_episodes_present": True,
            "cyclic_order_verified": True,
            "all_classifications_valid": True,
            "paired_conditions_identical": True,
            "applied_actions_verified": True,
            "source_and_checkpoint_lineage_verified": True,
        },
    }
    summary = {
        "status": "completed",
        "episode_count": len(episode_results),
        "case_count": len(paired),
        "classification_counts": classifications,
        "mode_results": by_mode,
        "source_actions_canonical_sha256": source_summary["canonical_sha256"],
        "checkpoint_sha256": lineage["checkpoint_sha256"],
        "exploratory_only": True,
        "effect_proven": False,
        "promotion_permitted": False,
        "audit_sha256": None,
    }
    return audit, summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--evidence-dir", type=Path, default=Path(__file__).resolve().parent)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.evidence_dir.resolve()
    audit_path, summary_path = root / "audit.json", root / "summary.json"
    try:
        audit, summary = run_audit(root)
        atomic_json(audit_path, audit)
        summary["audit_sha256"] = file_sha256(audit_path)
        atomic_json(summary_path, summary)
        print(json.dumps({"status": "completed", "audit": str(audit_path), "summary": str(summary_path)}, sort_keys=True))
        return 0
    except BaseException as error:
        failure = {
            "schema_name": "rj45_frozen_ablation_independent_audit_v1",
            "status": "invalid",
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        atomic_json(audit_path, failure)
        atomic_json(summary_path, {"status": "invalid", "audit_sha256": file_sha256(audit_path)})
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
