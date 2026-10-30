#!/usr/bin/env python3
"""Summarize stability evidence from the three approved RJ45 recordings."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
from typing import Any, Callable


STEMS = ("v3-biased-zero-env54", "v5-bc-env54", "v5-bc-env42")
EXPECTED_ENV = {"v3-biased-zero-env54": 54, "v5-bc-env54": 54, "v5-bc-env42": 42}
RECORDER_SOURCE = "scripts/rj45_evaluation_video.py"
OBSERVATION_LAYOUT = {
    "estimated_relative_position_entry_scaled": [0, 3],
    "estimated_relative_quaternion_entry_wxyz": [3, 7],
    "arm_joint_position_scaled": [7, 21],
    "arm_joint_velocity_scaled": [21, 35],
    "grip_force_scaled": [35, 37],
    "entry_contact_force_scaled": [37, 40],
    "grip_proven": [40, 42],
    "phase_one_hot": [42, 50],
    "previous_expanded_action": [50, 62],
    "command_depth_scaled": [62, 63],
    "insertion_advance": [63, 64],
    "insertion_joint_reference_error_scaled": [64, 78],
    "policy_active": [78, 79],
}


class AnalysisError(ValueError):
    """Raised when an input violates the frozen evidence contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_export_module(script_dir: Path) -> Any:
    path = script_dir / "export_rj45_policy_media.py"
    spec = importlib.util.spec_from_file_location("rj45_policy_media_contract", path)
    if spec is None or spec.loader is None:
        raise AnalysisError(f"cannot load media contract: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _paths(input_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "video": input_dir / f"{stem}.mp4",
        "frames": input_dir / f"{stem}.mp4.frames.jsonl",
        "metadata": input_dir / f"{stem}.mp4.metadata.json",
        "evidence": input_dir / f"{stem}.mp4.evidence.json",
        "report": input_dir / f"{stem}.report.json",
    }


def verify_public_manifest(
    manifest: dict[str, Any], stem: str, paths: dict[str, Path]
) -> None:
    runs = {run.get("stem"): run for run in manifest.get("runs", [])}
    if set(runs) != set(STEMS):
        raise AnalysisError("media manifest must contain exactly the three approved recordings")
    expected = runs[stem].get("input")
    if not isinstance(expected, dict):
        raise AnalysisError(f"missing media-manifest input contract for {stem}")
    for label, path in paths.items():
        contract = expected.get(label)
        if not isinstance(contract, dict) or contract.get("sha256") != sha256(path):
            raise AnalysisError(f"{stem} {label} hash does not match media manifest")
        if contract.get("bytes") != path.stat().st_size:
            raise AnalysisError(f"{stem} {label} size does not match media manifest")


def control_endpoints(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for row in rows:
        control_time = float(row["control_time_s"])
        policy_time = float(row["control_step_snapshot"]["policy_input_timestamp_s"])
        if control_time < 0.0 or policy_time < 0.0:
            raise AnalysisError("control and policy timestamps must be nonnegative")
        if policy_time > control_time + 1e-9:
            raise AnalysisError("policy timestamp cannot be later than its control endpoint")
        step = int(row["control_step"])
        previous = selected.get(step)
        if previous is None or (row["physics_step"], row["frame_index"]) > (
            previous["physics_step"], previous["frame_index"]
        ):
            selected[step] = row
    endpoints = [selected[key] for key in sorted(selected)]
    times = [float(row["control_time_s"]) for row in endpoints]
    if any(right <= left for left, right in zip(times, times[1:])):
        raise AnalysisError("control endpoint timestamps must increase")
    return endpoints


def pose_ok(row: dict[str, Any]) -> bool:
    criteria = row["control_step_snapshot"]["terminal_evaluation_criteria"]
    return bool(criteria["depth"] and criteria["lateral"] and criteria["orientation"])


def velocity_ok(row: dict[str, Any]) -> bool:
    stability = row["control_step_snapshot"]["rj45_evaluation_stability"]
    return bool(stability["relative_linear_speed_ok"] and stability["relative_angular_speed_ok"])


def all_nine_stable(row: dict[str, Any]) -> bool:
    stability = row["control_step_snapshot"]["rj45_evaluation_stability"]
    samples = stability["physics_sample_stable"]
    return bool(
        stability["physics_substeps_complete"]
        and stability["physics_sample_count"] == 9
        and len(samples) == 9
        and all(samples)
        and stability["physics_all_samples_stable"]
    )


def vector_stats(vectors: list[list[float]]) -> dict[str, Any]:
    if not vectors:
        raise AnalysisError("cannot summarize an empty vector series")
    columns = list(zip(*vectors))
    return {
        "minimum": [min(column) for column in columns],
        "maximum": [max(column) for column in columns],
        "range": [max(column) - min(column) for column in columns],
        "population_stddev": [statistics.pstdev(column) for column in columns],
    }


def longest_run(
    rows: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool], control_dt_s: float
) -> dict[str, Any]:
    best = current = 0
    best_start: float | None = None
    best_end: float | None = None
    current_start: float | None = None
    for row in rows:
        if predicate(row):
            if current == 0:
                current_start = float(row["control_time_s"])
            current += 1
            if current > best:
                best = current
                best_start = current_start
                best_end = float(row["control_time_s"])
        else:
            current = 0
            current_start = None
    return {
        "control_steps": best,
        "duration_s": best * control_dt_s,
        "first_endpoint_control_time_s": best_start,
        "last_endpoint_control_time_s": best_end,
    }


def endpoint_record(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    snapshot = row["control_step_snapshot"]
    stability = snapshot["rj45_evaluation_stability"]
    return {
        "control_step": int(row["control_step"]),
        "control_time_s": float(row["control_time_s"]),
        "physics_time_s": float(row["physics_time_s"]),
        "depth_mm": float(row["actual_depth_m"]) * 1000.0,
        "lateral_error_um": float(row["actual_lateral_error_m"]) * 1e6,
        "orientation_error_deg": math.degrees(float(row["actual_orientation_error_rad"])),
        "relative_linear_speed_mm_s": float(stability["relative_linear_speed_m_s"]) * 1000.0,
        "relative_angular_speed_rad_s": float(stability["relative_angular_speed_rad_s"]),
        "dwell_s": float(snapshot["independent_eval_dwell_s"]),
        "pose_gate_pass": pose_ok(row),
        "velocity_gate_pass": velocity_ok(row),
        "all_nine_physics_samples_stable": all_nine_stable(row),
    }


def selected_outcome(
    report: dict[str, Any], metadata: dict[str, Any]
) -> dict[str, Any]:
    environment = metadata.get("selected_environment_index")
    episode = metadata.get("selected_episode_index")
    matches = [
        outcome for outcome in report.get("episode_outcomes", [])
        if outcome.get("environment_index") == environment
        and outcome.get("episode_index") == episode
    ]
    if len(matches) != 1:
        raise AnalysisError("selected recording must map to exactly one report outcome")
    outcome = matches[0]
    canonical = json.dumps(outcome.get("paired_condition"), sort_keys=True, separators=(",", ":"))
    if outcome.get("paired_condition_id") != canonical:
        raise AnalysisError("paired condition ID is not the canonical selected case identity")
    if outcome.get("valid") is not True:
        raise AnalysisError("selected report outcome is not valid")
    terminal = metadata["selected_terminal"]
    if outcome.get("success") != terminal.get("success"):
        raise AnalysisError("selected outcome success disagrees with capture metadata")
    return outcome


def summarize_run(
    stem: str,
    metadata: dict[str, Any],
    rows: list[dict[str, Any]],
    report: dict[str, Any],
) -> dict[str, Any]:
    if metadata.get("selected_environment_index") != EXPECTED_ENV[stem]:
        raise AnalysisError(f"unexpected selected environment for {stem}")
    endpoints = control_endpoints(rows)
    valid_insert = [
        row for row in endpoints
        if row.get("phase_name") == "insert" and row.get("policy_input_valid") is True
    ]
    if not valid_insert:
        raise AnalysisError(f"{stem} has no valid INSERT control endpoints")
    control_dt_s = float(report["provenance"]["control_dt_s"])
    physics_dt_s = float(report["provenance"]["physics_dt_s"])
    decimation = int(report["provenance"]["control_decimation"])
    if not math.isclose(control_dt_s, physics_dt_s * decimation, rel_tol=0.0, abs_tol=1e-12):
        raise AnalysisError("physics and control clocks disagree with decimation")
    outcome = selected_outcome(report, metadata)
    pose_rows = [row for row in valid_insert if pose_ok(row)]
    velocity_fail = [row for row in pose_rows if not velocity_ok(row)]
    all_nine_fail = [row for row in pose_rows if not all_nine_stable(row)]
    final_time = float(valid_insert[-1]["control_time_s"])
    last_second = [row for row in valid_insert if float(row["control_time_s"]) >= final_time - 1.0 - 1e-9]
    requested = [row["control_step_snapshot"]["requested_expanded_residual_action"] for row in valid_insert]
    applied = [row["control_step_snapshot"]["applied_bounded_residual_action"] for row in valid_insert]
    differences = [abs(float(a) - float(b)) for left, right in zip(requested, applied) for a, b in zip(left, right)]
    last_requested = [list(map(float, row["control_step_snapshot"]["requested_expanded_residual_action"][6:12])) for row in last_second]
    last_applied = [list(map(float, row["control_step_snapshot"]["applied_bounded_residual_action"][6:12])) for row in last_second]
    last_offset = [list(map(float, row["control_step_snapshot"]["applied_tip_target_offset_entry_6d"])) for row in last_second]
    delta = [
        math.sqrt(sum((right[index] - left[index]) ** 2 for index in range(6)))
        for left, right in zip(last_requested, last_requested[1:])
    ]
    return {
        "stem": stem,
        "selected_case": {
            "environment_index": outcome["environment_index"],
            "episode_index": outcome["episode_index"],
            "paired_condition_id": outcome["paired_condition_id"],
            "paired_condition": outcome["paired_condition"],
        },
        "policy_mode": outcome["policy_mode"],
        "outcome": {"valid": outcome["valid"], "success": outcome["success"], "termination_reason": outcome["termination_reason"]},
        "clock_contract": {
            "raw_frames": len(rows),
            "unique_control_endpoints": len(endpoints),
            "control_dt_s": control_dt_s,
            "physics_dt_s": physics_dt_s,
            "physics_samples_per_control_step": decimation + 1,
            "endpoint_selection": "maximum (physics_step, frame_index) per control_step",
            "measurement_scope": "pose and endpoint velocity use the selected control endpoint; all-nine stability uses the recorded nine physics samples for that control step",
        },
        "valid_insert_control_endpoints": len(valid_insert),
        "pose_gate_pass": {
            "count": len(pose_rows),
            "velocity_gate_fail_count": len(velocity_fail),
            "all_nine_stability_fail_count": len(all_nine_fail),
            "first_pass": endpoint_record(pose_rows[0] if pose_rows else None),
            "first_velocity_fail": endpoint_record(velocity_fail[0] if velocity_fail else None),
        },
        "stability": {
            "maximum_recorded_dwell_s": max(float(row["control_step_snapshot"]["independent_eval_dwell_s"]) for row in valid_insert),
            "longest_pose_and_endpoint_velocity_pass": longest_run(valid_insert, lambda row: pose_ok(row) and velocity_ok(row), control_dt_s),
            "longest_all_nine_physics_samples_stable": longest_run(valid_insert, all_nine_stable, control_dt_s),
            "terminal_endpoint": endpoint_record(endpoints[-1]),
        },
        "action_application": {
            "requested_vs_applied_all_valid_insert": {
                "maximum_absolute_difference": max(differences),
                "different_elements_over_1e_9": sum(value > 1e-9 for value in differences),
                "total_elements": len(differences),
            },
            "last_one_second": {
                "control_endpoints": len(last_second),
                "start_control_time_s": float(last_second[0]["control_time_s"]),
                "end_control_time_s": float(last_second[-1]["control_time_s"]),
                "requested_right_action_dimensionless": vector_stats(last_requested),
                "applied_right_action_dimensionless": vector_stats(last_applied),
                "applied_tip_target_offset_entry_6d_si": vector_stats(last_offset),
                "requested_action_step_delta_l2": {
                    "mean": statistics.fmean(delta) if delta else 0.0,
                    "maximum": max(delta, default=0.0),
                },
            },
        },
    }


def render_readme(summary: dict[str, Any]) -> str:
    rows = []
    for run in summary["runs"]:
        gate = run["pose_gate_pass"]
        stability = run["stability"]
        rows.append(
            f"| `{run['stem']}` | {run['outcome']['success']} | {gate['count']} | "
            f"{gate['velocity_gate_fail_count']} | {gate['all_nine_stability_fail_count']} | "
            f"{stability['maximum_recorded_dwell_s']:.3f} s |"
        )
    return """# RJ45 안정화 추적 진단

이 문서는 공개 미디어에 사용한 세 실행의 원시 60 FPS 기록을 제어 스텝 단위로 다시 읽은 결과다. 같은 제어 스텝이 두 프레임에 기록된 경우 `(physics_step, frame_index)`가 가장 큰 끝점 하나만 남겼다. 자세 조건과 상대 속도는 그 끝점 값이며, `all-nine`은 해당 제어 스텝에 저장된 물리 샘플 9개가 모두 안정 기준을 통과했는지를 뜻한다.

| 실행 | 성공 | 자세 조건 통과 스텝 | 그중 속도 실패 | 그중 all-nine 실패 | 최대 안정 유지 |
|---|---:|---:|---:|---:|---:|
""" + "\n".join(rows) + """

## 확인된 사실

- 기준 제어 env54는 성공 판정까지 0.4초의 안정 유지 시간을 채웠다.
- BC env54와 env42는 타임아웃으로 끝났고, 자세 조건을 통과한 제어 끝점에서도 속도 또는 9개 물리 샘플 안정 조건을 계속 놓쳤다.
- 요청 action과 제한 후 적용 action의 차이, 마지막 1초의 action·절대 tip offset 범위는 `summary.json`에 보존했다.
- 제어 끝점 시각과 렌더 물리 시각은 구분해서 기록했다. 이 결과는 선택된 세 캡처에만 해당한다.

## 해석 범위와 다음 검증

이 분석은 BC 실패의 관측 신호를 좁히지만, 학습량 부족·데이터 분포 이동·보상 설계 가운데 하나를 원인으로 확정하지 않는다. 다음 A/B 실험은 같은 초기 조건과 학습량에서 회복 데이터 또는 안정화 보상 하나만 바꾸고, 미사용 조건으로 평가해야 한다. 시뮬레이터 정답 자세나 성공 gate는 평가기와 진단에만 사용하고 배포 정책 관측으로 넣지 않는다.

79차원 관측의 인덱스 범위는 `src/rby1_assembly/rj45_training.py`의 `build_policy_observation` 호출 계약과 테스트를 근거로 `summary.json`에 기록했다. 모든 위치·속도·action 단위도 같은 파일에 명시했다.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--media-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    export = load_export_module(Path(__file__).resolve().parent)
    media_manifest = json.loads(args.media_manifest.read_text(encoding="utf-8"))
    runs: list[dict[str, Any]] = []
    input_records: dict[str, Any] = {}
    source_maps: list[dict[str, str]] = []
    physics_contracts: list[tuple[float, float, int]] = []
    for stem in STEMS:
        paths = _paths(args.input_dir, stem)
        verify_public_manifest(media_manifest, stem, paths)
        metadata, rows, details = export.validate_inputs(stem, paths)
        report = details["report"]
        runs.append(summarize_run(stem, metadata, rows, report))
        input_records[stem] = {
            label: {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
            for label, path in paths.items()
        }
        source_maps.append(report["provenance"]["source_sha256"])
        physics_contracts.append((report["provenance"]["physics_dt_s"], report["provenance"]["control_dt_s"], report["provenance"]["control_decimation"]))
    if len(source_maps[0]) != 37:
        raise AnalysisError("unexpected frozen source-map size")
    if any(set(source_map) != set(source_maps[0]) for source_map in source_maps[1:]):
        raise AnalysisError("the three selected reports do not share one source-file set")
    common_source = {
        path: digest for path, digest in source_maps[0].items() if path != RECORDER_SOURCE
    }
    for source_map in source_maps[1:]:
        if any(source_map[path] != digest for path, digest in common_source.items()):
            raise AnalysisError("physics or policy source differs between selected reports")
    if any(contract != physics_contracts[0] for contract in physics_contracts[1:]):
        raise AnalysisError("the three selected reports do not share one physics contract")
    if runs[0]["selected_case"]["paired_condition_id"] != runs[1]["selected_case"]["paired_condition_id"]:
        raise AnalysisError("env54 reference and BC recordings are not the same paired case")
    summary = {
        "schema_version": 1,
        "scope": "three preselected policy-media recordings only; not a population estimate",
        "runs": runs,
        "observation_79d_layout_provenance": {
            "layout": OBSERVATION_LAYOUT,
            "source_contract": "src/rby1_assembly/rj45_training.py::build_policy_observation",
            "indexing": "zero-based half-open ranges",
            "policy_previous_right_action_range": [56, 62],
        },
        "common_provenance": {
            "common_physics_and_policy_source_file_count": len(common_source),
            "common_physics_and_policy_source_sha256": common_source,
            "per_run_recorder_source_sha256": {
                stem: source_map[RECORDER_SOURCE] for stem, source_map in zip(STEMS, source_maps)
            },
            "recorder_difference_scope": "only scripts/rj45_evaluation_video.py differs; the 36 physics, policy, configuration, and remaining source entries match",
            "physics_dt_s": physics_contracts[0][0],
            "control_dt_s": physics_contracts[0][1],
            "control_decimation": physics_contracts[0][2],
        },
        "inputs": input_records,
        "analysis": {
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "lineage_validator_path": str(Path(export.__file__).resolve()),
            "lineage_validator_sha256": sha256(Path(export.__file__)),
            "media_manifest_path": str(args.media_manifest.resolve()),
            "media_manifest_sha256": sha256(args.media_manifest),
        },
        "interpretation_limit": "observed stability signatures only; no claim that insufficient learning, data distribution shift, or reward design is the confirmed cause",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    readme_path = args.output_dir / "README.md"
    manifest_path = args.output_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    readme_path.write_text(render_readme(summary), encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "scope": summary["scope"],
        "inputs": input_records,
        "analysis_script": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())},
        "lineage_validator": {"path": str(Path(export.__file__).resolve()), "sha256": sha256(Path(export.__file__))},
        "media_manifest": {"path": str(args.media_manifest.resolve()), "sha256": sha256(args.media_manifest)},
        "outputs": {
            "summary.json": {"sha256": sha256(summary_path), "bytes": summary_path.stat().st_size},
            "README.md": {"sha256": sha256(readme_path), "bytes": readme_path.stat().st_size},
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "manifest_sha256": sha256(manifest_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
