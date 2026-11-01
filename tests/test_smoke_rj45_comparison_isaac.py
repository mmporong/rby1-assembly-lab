from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import shutil
import subprocess

import pytest
import torch


SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/smoke_rj45_comparison_isaac.py"
)
SPEC = importlib.util.spec_from_file_location("smoke_rj45_comparison_isaac", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SMOKE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SMOKE
SPEC.loader.exec_module(SMOKE)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


ACTUAL_RUNTIME = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf")
ACTUAL_EVIDENCE = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-evidence-104721a")


def _copy_snapshot_namespace(
    source_root: Path, destination_root: Path, mapping: dict[str, str]
) -> None:
    for relative in mapping:
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)


def _physics_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, dict]:
    current = SCRIPT.parents[1]
    runtime = tmp_path / "runtime"
    evidence = tmp_path / "evidence"
    actual_manifest_path = current / "configs/rj45-task-admission-physics-inputs-validation-ulp2-v1.json"
    actual_snapshot = SMOKE.validate_physics_input_snapshot(
        ACTUAL_RUNTIME, ACTUAL_EVIDENCE, current, actual_manifest_path
    )
    _copy_snapshot_namespace(
        ACTUAL_RUNTIME, runtime, actual_snapshot["normalized_sha256"]["runtime"]
    )
    _copy_snapshot_namespace(
        ACTUAL_EVIDENCE, evidence, actual_snapshot["normalized_sha256"]["evidence"]
    )
    shared_catalog = Path("configs/rj45-grasp-generation-v2.json")
    (evidence / shared_catalog).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ACTUAL_EVIDENCE / shared_catalog, evidence / shared_catalog)
    evidence_core_relative = "configs/rj45-oracle-runtime-104721a.json"
    evidence_core = evidence / evidence_core_relative
    evidence_core.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ACTUAL_EVIDENCE / evidence_core_relative, evidence_core)
    overlay = json.loads(actual_manifest_path.read_text(encoding="utf-8"))
    base = json.loads(
        (current / overlay["base_physics_manifest"]["relative_path"]).read_text(
            encoding="utf-8"
        )
    )
    manifest = {**base, **overlay}
    manifest_path = tmp_path / "physics.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return runtime, evidence, current, manifest_path, manifest


def test_parser_exposes_exact_frozen_runtime_and_gpu_arguments(tmp_path: Path) -> None:
    args = SMOKE._parser().parse_args(
        [
            "--runtime-root",
            str(tmp_path / "runtime-104721a"),
            "--evidence-root",
            str(tmp_path / "evidence-104721a"),
            "--current-root",
            str(tmp_path / "current"),
            "--report",
            str(tmp_path / "report.json"),
            "--parallel-contact-evidence",
            str(tmp_path / "runtime-104721a/parallel-evidence.json"),
        ]
    )
    assert args.device == "cuda:0"
    assert args.num_envs == 4
    assert args.ppo_rollout_steps == 8
    assert args.sac_vector_steps == 16
    assert args.sac_batch_size == 16
    assert args.sac_utd_ratio == 1.0
    assert str(args.held_bank).endswith(
        "artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json"
    )


def test_actual_dual_root_snapshot_has_exact_namespaces_and_rejects_equal_roots() -> None:
    runtime = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf")
    evidence = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-evidence-104721a")
    current = SCRIPT.parents[1]
    manifest = current / "configs/rj45-task-admission-physics-inputs-validation-ulp2-v1.json"
    snapshot = SMOKE.validate_physics_input_snapshot(
        runtime, evidence, current, manifest
    )
    assert {name: len(values) for name, values in snapshot["normalized_sha256"].items()} == {
        "runtime": 50,
        "evidence": 50,
        "current": 6,
    }
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="must be distinct"):
        SMOKE.validate_physics_input_snapshot(runtime, runtime, current, manifest)


def test_active_import_geometry_report_tamper_is_rejected(tmp_path: Path) -> None:
    source = Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-runtime-1517fbf")
    runtime = tmp_path / "runtime"
    shutil.copytree(source, runtime, copy_function=shutil.copyfile)
    report = runtime / "assets/generated/build-report.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["robot"]["gripper_finger_span_m"] += 1e-9
    report.write_text(json.dumps(payload), encoding="utf-8")
    current = SCRIPT.parents[1]
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="active import input"):
        SMOKE.validate_physics_input_snapshot(
            runtime,
            Path("/opt/homedata/rby1-storage/immutable-inputs/rj45-evidence-104721a"),
            current,
            current / "configs/rj45-task-admission-physics-inputs-validation-ulp2-v1.json",
        )


@pytest.mark.parametrize("num_envs", [3, 17])
def test_execution_rejects_environment_count_outside_short_smoke(num_envs: int) -> None:
    args = argparse.Namespace(
        num_envs=num_envs,
        device="cuda:0",
        ppo_rollout_steps=2,
        sac_vector_steps=4,
        sac_batch_size=16,
        sac_utd_ratio=1.0,
        min_sac_updates=2,
        readiness_steps=10,
        parallel_contact_evidence=Path("parallel-evidence.json"),
    )
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="num-envs"):
        SMOKE.validate_execution_args(args)


def test_runtime_snapshot_validation_is_exact_and_fail_closed(
    tmp_path: Path,
) -> None:
    runtime, _evidence, current, _physics_manifest, _manifest = _physics_fixture(
        tmp_path
    )
    manifest = current / "configs/rj45-oracle-runtime-validation-ulp2-v1.json"
    proof = SMOKE.validate_runtime_snapshot(runtime, manifest)
    assert proof["verified_file_count"] == 35
    source = runtime / "src/rby1_assembly/experiment_contracts.py"
    source.write_text("value = 2\n", encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="SHA-256 mismatch"):
        SMOKE.validate_runtime_snapshot(runtime, manifest)


def test_physics_snapshot_binds_all_inputs_and_validator_source(tmp_path: Path) -> None:
    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path)
    snapshot = SMOKE.validate_physics_input_snapshot(
        runtime, evidence, current, manifest_path
    )
    assert len(snapshot["normalized_sha256"]["runtime"]) == 50
    assert len(snapshot["normalized_sha256"]["evidence"]) == 50
    assert snapshot["normalized_sha256"]["current"] == {
        relative: _sha(current / relative)
        for relative in SMOKE.EXPECTED_CURRENT_SOURCE_PATHS
    }
    assert snapshot["manifest"]["sha256"] == _sha(manifest_path)
    assert snapshot["selected_cache_graph_sha256"] == manifest[
        "selected_cache_graph_sha256"
    ]
    assert snapshot["headless_physics_geometry_closure"][
        "full_render_shader_closure_proven"
    ] is False
    end = SMOKE.verify_physics_input_end(snapshot, runtime, evidence, current)
    assert end["matched"] is True
    assert end["selected_cache_graph_sha256"] == snapshot[
        "selected_cache_graph_sha256"
    ]
    (runtime / "src/rby1_assembly/late.py").write_text("value = 1\n")
    added = SMOKE.verify_physics_input_end(snapshot, runtime, evidence, current)
    assert added["status"] == "input_mutation_blocked"
    assert added["added"] == ["runtime/src/rby1_assembly/late.py"]


@pytest.mark.parametrize("mutation", ["missing", "additional", "absolute", "escape"])
def test_physics_manifest_rejects_incomplete_or_escaping_paths(
    tmp_path: Path, mutation: str
) -> None:
    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path)
    if mutation == "missing":
        manifest["physics_inputs"].pop("assets/generated/tasks/rj45-v1/scene.usda")
    elif mutation == "additional":
        manifest["physics_inputs"]["extra.usd"] = "0" * 64
    else:
        old = "assets/generated/tasks/rj45-v1/scene.usda"
        replacement = "/tmp/scene.usda" if mutation == "absolute" else "../scene.usda"
        manifest["physics_inputs"][replacement] = manifest["physics_inputs"].pop(old)
        manifest["required_physics_relative_paths"] = sorted(manifest["physics_inputs"])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)


def test_physics_snapshot_rejects_unlisted_runtime_python(tmp_path: Path) -> None:
    runtime, evidence, current, manifest_path, _manifest = _physics_fixture(tmp_path)
    (runtime / "src/rby1_assembly/unlisted.py").write_text("value = 1\n")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="exact 35-file"):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)


def test_physics_snapshot_rejects_training_generated_and_cache_drift(tmp_path: Path) -> None:
    for target, message in (
        ("configs/rj45-training-target-offset-v1.json", "physics input SHA"),
        ("assets/generated/tasks/rj45-v1/plug.usda", "physics input SHA"),
        ("assets/cache/rby1/model_v_1_2_m.usd", "physics input SHA"),
    ):
        case = tmp_path / target.replace("/", "-")
        runtime, evidence, current, manifest_path, _manifest = _physics_fixture(case)
        with (runtime / target).open("ab") as stream:
            stream.write(b"tamper")
        with pytest.raises(SMOKE.ComparisonSmokeBlocked, match=message):
            SMOKE.validate_physics_input_snapshot(
                runtime, evidence, current, manifest_path
            )


def test_generated_build_and_cache_catalog_mismatches_are_rejected(tmp_path: Path) -> None:
    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path / "build")
    manifest["generated_asset_sha256"]["plug.usda"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="build report"):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)

    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path / "cache")
    manifest["cache_assets"][0]["size_bytes"] += 1
    graph = {
        "robot_usda_sha256": manifest["physics_inputs"][
            "assets/generated/tasks/rj45-v1/robot.usda"
        ],
        "selected_cache_assets": manifest["cache_assets"],
    }
    manifest["selected_cache_graph_sha256"] = SMOKE._canonical_json_sha256(graph)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="size declaration"):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)


def test_cache_and_robot_reference_sets_reject_duplicate_substitution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path / "cache-set")
    repeated = dict(manifest["cache_assets"][1])
    manifest["cache_assets"] = [dict(repeated) for _ in range(3)]
    manifest["generated_robot_references"] = [repeated["relative_path"]] * 3
    graph = {
        "robot_usda_sha256": manifest["physics_inputs"][
            "assets/generated/tasks/rj45-v1/robot.usda"
        ],
        "selected_cache_assets": manifest["cache_assets"],
    }
    manifest["selected_cache_graph_sha256"] = SMOKE._canonical_json_sha256(graph)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="ID/path set"):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)

    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path / "robot-set")
    robot = runtime / "assets/generated/tasks/rj45-v1/robot.usda"
    robot.write_text(
        "#usda 1.0\n" + "@../../../cache/rby1/gripper/rb_gripper_left.usd@\n" * 3,
        encoding="utf-8",
    )
    robot_sha = _sha(robot)
    build_path = runtime / "assets/generated/tasks/rj45-v1/build-report.json"
    build = json.loads(build_path.read_text(encoding="utf-8"))
    build["files"]["robot.usda"] = robot_sha
    build_path.write_text(json.dumps(build), encoding="utf-8")
    evidence_robot = evidence / "assets/generated/tasks/rj45-v1/robot.usda"
    evidence_build = evidence / "assets/generated/tasks/rj45-v1/build-report.json"
    shutil.copyfile(robot, evidence_robot)
    shutil.copyfile(build_path, evidence_build)
    manifest["generated_asset_sha256"]["robot.usda"] = robot_sha
    manifest["physics_inputs"]["assets/generated/tasks/rj45-v1/robot.usda"] = robot_sha
    manifest["physics_inputs"]["assets/generated/tasks/rj45-v1/build-report.json"] = _sha(
        build_path
    )
    evidence_sources = dict(SMOKE.EXPECTED_EVIDENCE_SOURCE_CONFIG_INPUTS)
    evidence_sources["assets/generated/tasks/rj45-v1/build-report.json"] = _sha(
        evidence_build
    )
    manifest["evidence_source_config_inputs"] = evidence_sources
    monkeypatch.setattr(
        SMOKE, "EXPECTED_EVIDENCE_SOURCE_CONFIG_INPUTS", evidence_sources
    )
    graph = {
        "robot_usda_sha256": robot_sha,
        "selected_cache_assets": manifest["cache_assets"],
    }
    manifest["selected_cache_graph_sha256"] = SMOKE._canonical_json_sha256(graph)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="authored reference set"):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)


@pytest.mark.parametrize(
    "value",
    ["./scripts/tool.py", "scripts//tool.py", "scripts/tool.py/", "scripts\\tool.py"],
)
def test_safe_relative_rejects_noncanonical_spelling(value: str) -> None:
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="project-relative"):
        SMOKE._safe_relative(value, label="test path")


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_current_source_paths_require_exact_six_files(
    tmp_path: Path, mutation: str
) -> None:
    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path)
    if mutation == "missing":
        manifest["current_source_paths"].pop()
    else:
        manifest["current_source_paths"].append("README.md")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="exact six"):
        SMOKE.validate_physics_input_snapshot(runtime, evidence, current, manifest_path)


def test_held_proof_binding_and_end_mutation_fail_closed(tmp_path: Path) -> None:
    runtime, evidence, current, manifest_path, manifest = _physics_fixture(tmp_path)
    snapshot = SMOKE.validate_physics_input_snapshot(
        runtime, evidence, current, manifest_path
    )
    proof = {
        "provenance_contract": {
            "source_and_config_sha256": dict(
                manifest["held_proof_required_source_and_config_sha256"]
            )
        }
    }
    bound_manifest = current / "configs/rj45-task-admission-physics-inputs-validation-ulp2-v1.json"
    SMOKE.validate_held_proof_physics_binding(proof, snapshot, bound_manifest)
    proof["provenance_contract"]["source_and_config_sha256"][
        "configs/assets.json"
    ] = "0" * 64
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="held proof"):
        SMOKE.validate_held_proof_physics_binding(proof, snapshot, bound_manifest)
    with (runtime / "assets/generated/tasks/rj45-v1/scene.usda").open("ab") as stream:
        stream.write(b"mutation")
    end = SMOKE.verify_physics_input_end(snapshot, runtime, evidence, current)
    assert end["status"] == "input_mutation_blocked"
    assert end["matched"] is False
    assert end["changed"] == [
        "runtime/assets/generated/tasks/rj45-v1/scene.usda"
    ]


def test_end_verifier_rejects_source_manifest_role_path_swap(tmp_path: Path) -> None:
    runtime, evidence, current, manifest_path, _manifest = _physics_fixture(tmp_path)
    snapshot = SMOKE.validate_physics_input_snapshot(
        runtime, evidence, current, manifest_path
    )
    swapped = json.loads(json.dumps(snapshot))
    swapped["source_manifests"][0]["role"], swapped["source_manifests"][1]["role"] = (
        swapped["source_manifests"][1]["role"],
        swapped["source_manifests"][0]["role"],
    )
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="source manifest path differs"):
        SMOKE.verify_physics_input_end(swapped, runtime, evidence, current)


@pytest.mark.parametrize(
    "left,right",
    [("runtime", "evidence"), ("runtime", "current"), ("evidence", "current")],
)
def test_end_verifier_rejects_sealed_root_swap(
    tmp_path: Path, left: str, right: str
) -> None:
    runtime, evidence, current, manifest_path, _manifest = _physics_fixture(tmp_path)
    snapshot = SMOKE.validate_physics_input_snapshot(
        runtime, evidence, current, manifest_path
    )
    swapped = json.loads(json.dumps(snapshot))
    swapped[f"{left}_root"], swapped[f"{right}_root"] = (
        swapped[f"{right}_root"],
        swapped[f"{left}_root"],
    )
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="sealed snapshot roots"):
        SMOKE.verify_physics_input_end(swapped, runtime, evidence, current)


def test_end_verifier_rejects_identical_bytes_at_different_runtime_root(
    tmp_path: Path,
) -> None:
    runtime, evidence, current, manifest_path, _manifest = _physics_fixture(tmp_path)
    snapshot = SMOKE.validate_physics_input_snapshot(
        runtime, evidence, current, manifest_path
    )
    copied_runtime = tmp_path / "copied-runtime"
    shutil.copytree(runtime, copied_runtime, copy_function=shutil.copyfile)
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="sealed snapshot roots"):
        SMOKE.verify_physics_input_end(snapshot, copied_runtime, evidence, current)


def test_resolve_accepts_path_and_string_asset_boundaries(tmp_path: Path) -> None:
    task = json.loads(
        (SCRIPT.parents[1] / "configs/tasks/rj45-v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert isinstance(task["assets"]["build_report"], str)
    relative = task["assets"]["build_report"]
    assert SMOKE._resolve(tmp_path, relative) == tmp_path / relative
    assert SMOKE._resolve(tmp_path, Path(relative)) == tmp_path / relative

    absolute = (tmp_path / "absolute.json").resolve()
    assert SMOKE._resolve(tmp_path / "ignored", str(absolute)) == absolute
    assert SMOKE._resolve(tmp_path / "ignored", absolute) == absolute


def test_runtime_versions_match_the_actual_pinned_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rby1_assembly.rj45_task import verify_runtime_versions

    runtime = json.loads(
        (SCRIPT.parents[1] / "configs/runtime.json").read_text(encoding="utf-8")
    )
    installed = {
        "isaacsim": runtime["isaac_sim_distribution"],
        "isaaclab": runtime["isaac_lab_distribution"],
        "isaaclab-rl": runtime["isaac_lab_rl"],
        "rsl-rl-lib": runtime["rsl_rl"],
        "torch": runtime["torch"] + "+" + runtime["torch_cuda_build"],
        "numpy": runtime["numpy"],
        "h5py": runtime["h5py"],
    }
    monkeypatch.setattr(SMOKE.metadata, "version", installed.__getitem__)
    versions = SMOKE._runtime_versions()
    assert versions == installed
    verify_runtime_versions(versions, runtime)


def test_parallel_contact_evidence_matches_frozen_four_env_contract() -> None:
    from rby1_assembly.parallel_contact_evidence import load_parallel_contact_evidence

    root = SCRIPT.parents[1]
    relative = Path(
        "artifacts/rj45-parallel-probe/after-persistent-target-hook/evidence-spec.json"
    )
    path = root / relative
    assert _sha(path) == SMOKE.EXPECTED_PARALLEL_CONTACT_EVIDENCE_SHA256
    specification = json.loads(path.read_text(encoding="utf-8"))
    dependencies = SMOKE._parallel_evidence_dependency_paths(root, specification)
    before = {dependency: _sha(dependency) for dependency in dependencies}
    proof = load_parallel_contact_evidence(specification, project_root=root)
    after = {dependency: _sha(dependency) for dependency in dependencies}
    assert before == after
    assert proof["validated_probe_num_envs"] == 4
    assert proof["report_sha256"] == specification["report_sha256"]
    assert proof["trace_sha256"] == specification["trace_sha256"]
    assert proof["sat_report_sha256"] == specification["sat_report_sha256"]


def test_main_failure_exit_survives_app_close_system_exit_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeApp:
        def close(self) -> None:
            raise SystemExit(0)

    class FakeAppLauncher:
        def __init__(self, _settings) -> None:
            self.app = FakeApp()

    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_app_module = types.ModuleType("isaaclab.app")
    isaaclab_app_module.AppLauncher = FakeAppLauncher
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.app", isaaclab_app_module)
    monkeypatch.setattr(SMOKE, "validate_execution_args", lambda _args: None)
    monkeypatch.setattr(SMOKE, "validate_prelaunch_inputs", lambda _args: {})
    monkeypatch.setattr(
        SMOKE,
        "run_smoke",
        lambda _args, _prelaunch, _progress: (_ for _ in ()).throw(
            RuntimeError("smoke failed")
        ),
    )
    report = tmp_path / "blocked.json"
    exit_code = SMOKE.main(
        [
            "--runtime-root",
            str(tmp_path / "runtime"),
            "--evidence-root",
            str(tmp_path / "evidence"),
            "--current-root",
            str(tmp_path / "current"),
            "--report",
            str(report),
            "--parallel-contact-evidence",
            str(tmp_path / "runtime/parallel-evidence.json"),
        ]
    )
    assert exit_code == 2
    failure = json.loads(report.read_text(encoding="utf-8"))
    assert failure["status"] == "blocked"
    assert failure["requested_process_exit_code"] == 2
    assert failure["external_supervisor_must_validate_report_status_and_claims"] is True
    stages = [event["stage"] for event in failure["progress_events"]]
    assert "runtime_validation_complete" in stages
    assert "comparison_smoke_begin" in stages


def test_progress_recorder_emits_flushed_json_and_retains_event(
    capsys: pytest.CaptureFixture[str],
) -> None:
    progress = SMOKE.ProgressRecorder()
    progress.record("unit_checkpoint", item_count=4)
    line = capsys.readouterr().out.strip()
    emitted = json.loads(line)
    assert emitted["event"] == "g0_progress"
    assert emitted["stage"] == "unit_checkpoint"
    assert emitted["item_count"] == 4
    assert progress.events == [emitted]


def test_traceback_heartbeat_uses_repeat_without_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(SMOKE.faulthandler, "enable", lambda: calls.append(("enable",)))
    monkeypatch.setattr(
        SMOKE.faulthandler,
        "dump_traceback_later",
        lambda timeout, *, repeat, exit: calls.append(
            ("dump", timeout, repeat, exit)
        ),
    )
    progress = SMOKE.ProgressRecorder()
    assert SMOKE._arm_traceback_heartbeat(progress) is True
    assert calls == [("enable",), ("dump", 60.0, True, False)]
    assert progress.events[-1]["stage"] == "traceback_heartbeat_armed"


def test_next_environment_lifecycle_stops_flushes_and_replaces_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeTimeline:
        def __init__(self) -> None:
            self.playing = True
            self.stopped = False
            self.stop_pending = False

        def is_playing(self) -> bool:
            return self.playing

        def is_stopped(self) -> bool:
            return self.stopped

        def stop(self) -> None:
            calls.append("timeline.stop")
            self.stop_pending = True

    timeline = FakeTimeline()

    class FakeApp:
        def update(self) -> None:
            calls.append("app.update")
            assert timeline.stop_pending
            timeline.playing = False
            timeline.stopped = True

    old_stage = object()
    new_stage = object()
    current_stage = [old_stage]

    stage_module = types.ModuleType("isaacsim.core.utils.stage")

    def create_new_stage():
        calls.append("stage.create")
        assert timeline.stopped
        current_stage[0] = new_stage
        return new_stage

    def update_stage() -> None:
        calls.append("stage.update")

    stage_module.create_new_stage = create_new_stage
    stage_module.update_stage = update_stage
    stage_module.get_current_stage = lambda: current_stage[0]

    timeline_module = types.ModuleType("omni.timeline")
    timeline_module.get_timeline_interface = lambda: timeline
    kit_app_module = types.ModuleType("omni.kit.app")
    kit_app_module.get_app = lambda: FakeApp()
    simulation_context = type(
        "SimulationContext", (), {"instance": classmethod(lambda cls: None)}
    )
    isaaclab_sim_module = types.ModuleType("isaaclab.sim")
    isaaclab_sim_module.SimulationContext = simulation_context

    isaacsim_module = types.ModuleType("isaacsim")
    isaacsim_core_module = types.ModuleType("isaacsim.core")
    isaacsim_utils_module = types.ModuleType("isaacsim.core.utils")
    isaacsim_module.core = isaacsim_core_module
    isaacsim_core_module.utils = isaacsim_utils_module
    isaacsim_utils_module.stage = stage_module
    omni_module = types.ModuleType("omni")
    omni_kit_module = types.ModuleType("omni.kit")
    omni_module.kit = omni_kit_module
    omni_module.timeline = timeline_module
    omni_kit_module.app = kit_app_module
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_module.sim = isaaclab_sim_module
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", isaacsim_core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", isaacsim_utils_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage", stage_module)
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit_module)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", kit_app_module)
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim_module)

    progress = SMOKE.ProgressRecorder()
    SMOKE.prepare_next_environment(progress)

    assert calls == [
        "timeline.stop",
        "app.update",
        "stage.create",
        "stage.update",
    ]
    assert [event["stage"] for event in progress.events] == [
        "environment_transition_begin",
        "timeline_stop_requested",
        "timeline_stop_event_flushed",
        "fresh_stage_creation_begin",
        "fresh_stage_creation_complete",
        "environment_transition_complete",
    ]


def test_next_environment_lifecycle_rejects_registered_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    simulation_context = type(
        "SimulationContext", (), {"instance": classmethod(lambda cls: object())}
    )
    isaaclab_sim_module = types.ModuleType("isaaclab.sim")
    isaaclab_sim_module.SimulationContext = simulation_context
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_module.sim = isaaclab_sim_module
    isaacsim_module = types.ModuleType("isaacsim")
    isaacsim_core_module = types.ModuleType("isaacsim.core")
    isaacsim_utils_module = types.ModuleType("isaacsim.core.utils")
    stage_module = types.ModuleType("isaacsim.core.utils.stage")
    isaacsim_module.core = isaacsim_core_module
    isaacsim_core_module.utils = isaacsim_utils_module
    isaacsim_utils_module.stage = stage_module
    omni_module = types.ModuleType("omni")
    omni_kit_module = types.ModuleType("omni.kit")
    timeline_module = types.ModuleType("omni.timeline")
    kit_app_module = types.ModuleType("omni.kit.app")
    omni_module.kit = omni_kit_module
    omni_module.timeline = timeline_module
    omni_kit_module.app = kit_app_module
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab_module)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", isaaclab_sim_module)
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core", isaacsim_core_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils", isaacsim_utils_module)
    monkeypatch.setitem(sys.modules, "isaacsim.core.utils.stage", stage_module)
    monkeypatch.setitem(sys.modules, "omni", omni_module)
    monkeypatch.setitem(sys.modules, "omni.kit", omni_kit_module)
    monkeypatch.setitem(sys.modules, "omni.timeline", timeline_module)
    monkeypatch.setitem(sys.modules, "omni.kit.app", kit_app_module)

    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="remains registered"):
        SMOKE.prepare_next_environment(SMOKE.ProgressRecorder())


class _ReadinessEnv:
    def __init__(self, *, mixed_forever: bool = False):
        self.num_envs = 4
        self.device = torch.device("cpu")
        self.steps = 0
        self.mixed_forever = mixed_forever
        self._training_policy_activated = torch.zeros(4, dtype=torch.bool)
        self._training_estimator_valid = torch.zeros(4, dtype=torch.bool)

    def _state(self):
        valid = self._training_policy_activated & self._training_estimator_valid
        return (
            {"policy": torch.full((4, 79), float(self.steps))},
            {"policy_observation_valid": valid.clone()},
        )

    def reset(self):
        return self._state()

    def step(self, actions):
        assert torch.equal(actions, torch.zeros((4, 6)))
        self.steps += 1
        self._training_policy_activated[:] = True
        self._training_estimator_valid[:] = True
        if self.mixed_forever:
            self._training_estimator_valid[2] = False
        observations, infos = self._state()
        return (
            observations,
            torch.zeros(4),
            torch.zeros(4, dtype=torch.bool),
            torch.zeros(4, dtype=torch.bool),
            infos,
        )


def test_readiness_excludes_warmup_and_requires_all_rows() -> None:
    env = _ReadinessEnv()
    progress = SMOKE.ProgressRecorder()
    observations, infos, proof = SMOKE.wait_for_policy_readiness(
        env, max_steps=2, progress=progress
    )
    assert observations.shape == (4, 79)
    assert infos["policy_observation_valid"].all()
    assert proof["warmup_vector_steps"] == 1
    assert proof["warmup_transitions_excluded_from_learning"] == 4
    assert proof["ready_rows_history"] == [0, 4]
    stages = [event["stage"] for event in progress.events]
    assert stages == [
        "initial_reset_begin",
        "initial_reset_complete",
        "readiness_checkpoint",
        "readiness_checkpoint",
    ]
    assert progress.events[-1]["ready_rows"] == 4

    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="not all"):
        SMOKE.wait_for_policy_readiness(_ReadinessEnv(mixed_forever=True), max_steps=2)


def test_ready_vector_reset_is_one_shot_and_preserves_captured_infos() -> None:
    env = _ReadinessEnv()
    observations = torch.ones((4, 79))
    infos = {"policy_observation_valid": torch.ones(4, dtype=torch.bool)}
    ready = SMOKE.ReadyVectorEnvironment(env, observations, infos)
    reset_observations, reset_infos = ready.reset()
    assert torch.equal(reset_observations["policy"], observations)
    assert reset_infos is infos
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="only be consumed once"):
        ready.reset()


def test_collector_validation_rejects_mixed_or_deleted_samples() -> None:
    valid = argparse.Namespace(
        transitions=16,
        valid_policy_transitions=16,
        learning_transitions=16,
        warmup_transitions=0,
        skipped_valid_transitions=0,
        mixed_validity_vector_steps=0,
    )
    SMOKE.validate_collector_result(valid, expected_transitions=16)
    mixed = argparse.Namespace(**vars(valid))
    mixed.learning_transitions = 12
    mixed.warmup_transitions = 4
    with pytest.raises(SMOKE.ComparisonSmokeBlocked, match="omitted transitions"):
        SMOKE.validate_collector_result(mixed, expected_transitions=16)


def test_finite_structure_rejects_nonfinite_update_values() -> None:
    assert SMOKE._finite_structure({"loss": torch.tensor([1.0]), "count": 2})
    assert not SMOKE._finite_structure({"loss": torch.tensor([float("nan")])})


def test_selected_vector_steps_report_exact_common_agent_counters() -> None:
    from rby1_assembly.rj45_comparison_rl import collect_bounded, make_agent

    class ReadyEnv:
        num_envs = 4

        def __init__(self):
            self.observations = torch.zeros((4, 79))

        def reset(self):
            return {"policy": self.observations.clone()}, {
                "policy_observation_valid": torch.ones(4, dtype=torch.bool)
            }

        def step(self, _actions):
            return (
                {"policy": self.observations.clone()},
                torch.zeros(4),
                torch.zeros(4, dtype=torch.bool),
                torch.zeros(4, dtype=torch.bool),
                {"policy_observation_valid": torch.ones(4, dtype=torch.bool)},
            )

    for algorithm in SMOKE.ALGORITHMS:
        agent = make_agent(
            algorithm,
            num_envs=4,
            rollout_steps=8,
            replay_capacity=256,
            batch_size=16,
            sac_utd_ratio=1.0,
            seed=3,
        )
        updates = 0
        original_update = agent.update

        def audited_update(*args, _original=original_update, **kwargs):
            nonlocal updates
            updates += 1
            return _original(*args, **kwargs)

        agent.update = audited_update
        vector_steps = SMOKE.algorithm_vector_steps(
            algorithm, ppo_rollout_steps=8, sac_vector_steps=16
        )
        result = collect_bounded(
            agent,
            ReadyEnv(),
            algorithm=algorithm,
            vector_steps=vector_steps,
            require_policy_validity=True,
        )
        assert updates == result.algorithm_updates
        if algorithm == "ppo":
            assert result.algorithm_updates == 2
            assert result.optimizer_steps == 8
        else:
            assert result.algorithm_updates == 64
            assert result.optimizer_steps == 192
            assert result.replay_first_ready_at == 16
            assert result.sac_utd_ratio == 1.0
        SMOKE.validate_collector_result(
            result, expected_transitions=4 * vector_steps
        )
