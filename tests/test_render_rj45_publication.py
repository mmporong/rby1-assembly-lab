from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/render_rj45_publication.py"
SPEC = importlib.util.spec_from_file_location("render_rj45_publication_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RENDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RENDER)


def test_deadline_is_future_and_bounded() -> None:
    now = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc).timestamp()
    assert RENDER._parse_utc_deadline("2026-11-01T13:00:00Z", now=now) == now + 3600
    with pytest.raises(RENDER.PublicationRenderError, match="at least 60 seconds"):
        RENDER._parse_utc_deadline("2026-11-01T12:00:30Z", now=now)
    with pytest.raises(RENDER.PublicationRenderError, match="four hours"):
        RENDER._parse_utc_deadline("2026-11-01T17:00:00Z", now=now)


class _Array:
    def __init__(self, row):
        self._row = row

    def __getitem__(self, index):
        assert index == 0
        return self

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._row


def test_camera_tracks_live_connection_midpoint() -> None:
    env = SimpleNamespace(
        socket=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=_Array([1.0, 2.0, 3.0]),
            root_quat_w=_Array([1.0, 0.0, 0.0, 0.0]),
        )),
        plug=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=_Array([1.0, 2.2, 3.0]),
            root_quat_w=_Array([1.0, 0.0, 0.0, 0.0]),
        )),
        cfg=SimpleNamespace(rj45_task={"frames": {
            "socket_entry": {"position_m": [0.0, -0.03, 0.0]},
            "plug_tip": {"position_m": [0.0, 0.01, 0.0]},
        }}),
    )
    center = RENDER.connection_center(env)
    assert center == pytest.approx((1.0, 2.09, 3.0))
    camera = RENDER.camera_from_connection(env, (0.4, 0.0, 0.2), (0.0, 0.0, 0.0))
    assert camera["eye_world_m"] == pytest.approx((1.4, 2.09, 3.2))
    assert camera["lookat_world_m"] == pytest.approx(center)


def test_explicit_camera_center_does_not_use_staged_object_pose() -> None:
    camera = RENDER.camera_from_center(
        (0.37, 0.0145, 0.9005),
        RENDER.DEFAULT_EYE_OFFSET,
        RENDER.DEFAULT_LOOKAT_OFFSET,
    )
    assert camera["eye_world_m"] == pytest.approx((0.79, 0.1145, 1.1805))
    assert camera["lookat_world_m"] == pytest.approx((0.37, 0.0145, 0.9005))


def test_preview_gate_requires_capture_and_warmup() -> None:
    env = SimpleNamespace(
        _training_warmup_complete=[SimpleNamespace(item=lambda: True)]
    )
    assert RENDER._warmup_complete(env) is True
    env._training_warmup_complete = [SimpleNamespace(item=lambda: False)]
    assert RENDER._warmup_complete(env) is False


def test_preview_hook_waits_for_first_warm_capture_and_restores_camera(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    preview_calls: list[tuple[float, float, float]] = []
    camera_calls: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def fake_previews(env, preview_dir, center, lookat_offset):
        preview_calls.append(center)
        return [{"path": str(preview_dir / "camera-1.png")}]

    monkeypatch.setattr(RENDER, "_write_previews", fake_previews)

    class Recorder:
        def _read_frame(self):
            return "original-frame"

    video_module = SimpleNamespace(EvaluationVideoRecorder=Recorder)
    renderer = SimpleNamespace(
        _build_render_env=lambda *args, **kwargs: None,
        _load_module=lambda path, name: video_module,
    )
    state: dict = {}
    center = (0.37, 0.0145, 0.9005)
    RENDER._patch_renderer(
        renderer,
        eye_offset=RENDER.DEFAULT_EYE_OFFSET,
        lookat_offset=RENDER.DEFAULT_LOOKAT_OFFSET,
        center_world=center,
        preview_dir=tmp_path,
        state=state,
    )
    renderer._load_module(Path("rj45_evaluation_video.py"), "video")
    env = SimpleNamespace(
        socket=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=_Array([0.37, 0.05, 0.90]),
            root_quat_w=_Array([1.0, 0.0, 0.0, 0.0]),
        )),
        plug=SimpleNamespace(data=SimpleNamespace(
            root_pos_w=_Array([0.37, 0.0006, 0.9005]),
            root_quat_w=_Array([1.0, 0.0, 0.0, 0.0]),
        )),
        cfg=SimpleNamespace(rj45_task={"frames": {
            "socket_entry": {"position_m": [0.0, -0.033, 0.0]},
            "plug_tip": {"position_m": [0.0, 0.01137, 0.0]},
        }}),
        _training_warmup_complete=[SimpleNamespace(item=lambda: True)],
        sim=SimpleNamespace(
            set_camera_view=lambda eye, lookat: camera_calls.append((eye, lookat))
        ),
    )
    recorder = Recorder()
    recorder.env = env
    recorder._capture_active = False
    assert recorder._read_frame() == "original-frame"
    assert preview_calls == []

    recorder._capture_active = True
    assert recorder._read_frame() == "original-frame"
    assert preview_calls == [center]
    assert len(camera_calls) == 1
    assert state["connection_positions_at_preview"]["socket_entry_world_m"] == pytest.approx(
        [0.37, 0.017, 0.90]
    )

    assert recorder._read_frame() == "original-frame"
    assert preview_calls == [center]


def test_quaternion_rotation_uses_wxyz_order() -> None:
    half = math.sqrt(0.5)
    assert RENDER._rotate_wxyz((half, 0.0, 0.0, half), (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))


def test_patch_preserves_historical_deadline_constant() -> None:
    renderer = SimpleNamespace(
        _build_render_env=lambda *args, **kwargs: None,
        _load_module=lambda path, name: SimpleNamespace(),
    )
    replay = SimpleNamespace(
        DEADLINE_UTC="2026-10-31T10:50:52Z",
        _load_module=lambda path, name: renderer,
        _deadline_epoch=lambda: 0.0,
        persist_report_before_app_close=lambda *args: 0,
    )
    RENDER._patch_sealed_replay(
        replay,
        deadline_epoch=1234.0,
        eye_offset=RENDER.DEFAULT_EYE_OFFSET,
        lookat_offset=RENDER.DEFAULT_LOOKAT_OFFSET,
        center_world=None,
        preview_dir=None,
        publication_report=Path("publication.json"),
        state={},
    )
    assert replay._deadline_epoch() == 1234.0
    assert replay.DEADLINE_UTC == "2026-10-31T10:50:52Z"


def test_sealed_source_hash_matches_repository_copy() -> None:
    sealed = ROOT / "scripts/record_rj45_multibias_replay.py"
    assert RENDER._file_sha256(sealed) == RENDER.SEALED_REPLAY_SHA256


def test_appearance_overrides_are_authored_only_in_session_layer() -> None:
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    table = UsdGeom.Cube.Define(stage, "/World/envs/env_0/Table/top")
    table.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.75, 0.78)])
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2500.0)
    root_layer = stage.GetRootLayer()

    report = RENDER.apply_publication_appearance(stage)

    assert report["edit_layer"] == stage.GetSessionLayer().identifier
    assert stage.GetEditTarget().GetLayer() == root_layer
    assert tuple(table.GetDisplayColorAttr().Get()[0]) == pytest.approx(RENDER.TABLE_COLOR)
    light_api = UsdLux.LightAPI(stage.GetPrimAtPath("/World/Light"))
    assert light_api.GetIntensityAttr().Get() == RENDER.LIGHT_INTENSITY
    assert light_api.GetExposureAttr().Get() == pytest.approx(RENDER.LIGHT_EXPOSURE)
    assert table.GetDisplayColorAttr().GetPropertyStack()[0].layer == stage.GetSessionLayer()


def test_publication_sidecar_is_written_before_app_close(tmp_path: Path) -> None:
    calls: list[str] = []
    renderer = SimpleNamespace(
        _build_render_env=lambda *args, **kwargs: None,
        _load_module=lambda path, name: SimpleNamespace(),
    )

    def persist(*args):
        assert (tmp_path / "publication.json").is_file()
        before_close = json.loads((tmp_path / "publication.json").read_text())
        assert before_close["status"] == "pending_app_close"
        report_path = args[0]
        report_path.write_text(
            json.dumps(args[1], indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        calls.append("sealed-persist-and-close")
        return args[-1]

    replay = SimpleNamespace(
        _load_module=lambda path, name: renderer,
        _deadline_epoch=lambda: 0.0,
        persist_report_before_app_close=persist,
    )
    video = tmp_path / "sealed.mp4"
    video.write_bytes(b"video")
    state = {
        "wrapper": {"path": "/wrapper.py", "sha256": "a" * 64},
        "sealed_replay_source": {"path": "/sealed.py", "sha256": "b" * 64},
        "sealed_run_arguments": ["--mode", "zero"],
        "sealed_video_path": str(video),
        "publication_deadline_utc": "2026-11-01T13:00:00Z",
    }
    RENDER._patch_sealed_replay(
        replay,
        deadline_epoch=1234.0,
        eye_offset=RENDER.DEFAULT_EYE_OFFSET,
        lookat_offset=RENDER.DEFAULT_LOOKAT_OFFSET,
        center_world=None,
        preview_dir=None,
        publication_report=tmp_path / "publication.json",
        state=state,
    )
    result = replay.persist_report_before_app_close(
        tmp_path / "sealed.json",
        {"status": "completed", "lineage": {"primary_report_sha256": "c" * 64}},
        object(),
        object(),
        object(),
        0,
    )
    payload = json.loads((tmp_path / "publication.json").read_text())
    assert result == 0
    assert calls == ["sealed-persist-and-close"]
    assert payload["status"] == "completed"
    assert payload["app_close_exit_code"] == 0
    assert payload["sealed_report"]["sha256"] == RENDER._file_sha256(
        tmp_path / "sealed.json"
    )
    assert payload["sealed_report"]["lineage"]["primary_report_sha256"] == "c" * 64


def test_close_failure_keeps_sidecar_invalid(tmp_path: Path) -> None:
    video = tmp_path / "sealed.mp4"
    video.write_bytes(b"video")
    renderer = SimpleNamespace(
        _build_render_env=lambda *args, **kwargs: None,
        _load_module=lambda path, name: SimpleNamespace(),
    )

    def failed_close(report_path, report, *_args):
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 9

    replay = SimpleNamespace(
        _load_module=lambda path, name: renderer,
        _deadline_epoch=lambda: 0.0,
        persist_report_before_app_close=failed_close,
    )
    state = {
        "wrapper": {"path": "/wrapper.py", "sha256": "a" * 64},
        "sealed_replay_source": {"path": "/sealed.py", "sha256": "b" * 64},
        "sealed_run_arguments": ["--mode", "zero"],
        "sealed_video_path": str(video),
        "publication_deadline_utc": "2026-11-01T13:00:00Z",
    }
    publication_report = tmp_path / "publication.json"
    RENDER._patch_sealed_replay(
        replay,
        deadline_epoch=1234.0,
        eye_offset=RENDER.DEFAULT_EYE_OFFSET,
        lookat_offset=RENDER.DEFAULT_LOOKAT_OFFSET,
        center_world=None,
        preview_dir=None,
        publication_report=publication_report,
        state=state,
    )
    result = replay.persist_report_before_app_close(
        tmp_path / "sealed.json",
        {"status": "completed"},
        object(),
        object(),
        object(),
        0,
    )
    payload = json.loads(publication_report.read_text())
    assert result == 9
    assert payload["status"] == "invalid"
    assert payload["app_close_exit_code"] == 9


def pending_publication_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    replay_report = tmp_path / "replay-report.json"
    replay_report.write_text(json.dumps({"status": "completed"}, sort_keys=True) + "\n")
    video = tmp_path / "replay.mp4"
    video.write_bytes(b"video-bytes")
    exit_record = tmp_path / "native-exit.json"
    exit_record.write_text('{"exit_code": 0}\n')
    publication_report = tmp_path / "publication-report.json"
    publication_report.write_text(json.dumps({
        "schema_name": RENDER.SCHEMA_NAME,
        "status": "pending_app_close",
        "wrapper": {
            "path": str(SCRIPT),
            "sha256": RENDER._file_sha256(SCRIPT),
        },
        "sealed_report": {
            "path": str(replay_report),
            "expected_sha256": RENDER._file_sha256(replay_report),
            "status": "completed",
        },
        "sealed_video": {
            "path": str(video),
            "exists": True,
            "sha256": RENDER._file_sha256(video),
        },
        "app_close_exit_code": None,
    }, indent=2, sort_keys=True) + "\n")
    return publication_report, exit_record, replay_report, video


def test_native_exit_finalizer_completes_verified_pending_sidecar(tmp_path: Path) -> None:
    publication, exit_record, replay_report, _video = pending_publication_fixture(tmp_path)
    result = RENDER.finalize_from_native_exit(publication, exit_record)
    persisted = json.loads(publication.read_text())
    assert result == persisted
    assert persisted["status"] == "completed"
    assert persisted["app_close_exit_code"] is None
    assert persisted["native_process_exit_code"] == 0
    assert persisted["completion_witness"] == "outer_runner_native_exit"
    assert persisted["sealed_report"]["sha256"] == RENDER._file_sha256(replay_report)
    assert persisted["native_exit_record"]["sha256"] == RENDER._file_sha256(exit_record)


def test_native_exit_finalizer_rejects_bad_exit_without_mutation(tmp_path: Path) -> None:
    publication, exit_record, _report, _video = pending_publication_fixture(tmp_path)
    exit_record.write_text('{"exit_code": 7}\n')
    before = publication.read_bytes()
    with pytest.raises(RENDER.PublicationRenderError, match="exit code is not zero"):
        RENDER.finalize_from_native_exit(publication, exit_record)
    assert publication.read_bytes() == before


@pytest.mark.parametrize("artifact", ["report", "video"])
def test_native_exit_finalizer_rejects_changed_artifact(
    tmp_path: Path, artifact: str
) -> None:
    publication, exit_record, report, video = pending_publication_fixture(tmp_path)
    (report if artifact == "report" else video).write_bytes(b"changed")
    before = publication.read_bytes()
    with pytest.raises(RENDER.PublicationRenderError, match=f"sealed {artifact} hash"):
        RENDER.finalize_from_native_exit(publication, exit_record)
    assert publication.read_bytes() == before
