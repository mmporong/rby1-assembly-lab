#!/usr/bin/env python3
"""Render the sealed fresh-05 replay with publication-only visual overrides.

The sealed replay remains responsible for all lineage, checkpoint, initial
observation, and outcome checks.  This wrapper changes only the composed USD
stage's appearance and the RGB camera used by the video recorder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence


SEALED_REPLAY_SHA256 = "49bf950a0b466820d82b5dfec1aae6bee498c8ac0b0428f24908b387d04a4e2f"
SCHEMA_NAME = "rj45_publication_render_v1"
TABLE_COLOR = (0.10, 0.12, 0.14)
LIGHT_INTENSITY = 700.0
LIGHT_EXPOSURE = -0.4
DEFAULT_EYE_OFFSET = (0.42, 0.10, 0.28)
DEFAULT_LOOKAT_OFFSET = (0.0, 0.0, 0.0)
PREVIEW_EYE_OFFSETS = (
    DEFAULT_EYE_OFFSET,
    (0.36, 0.20, 0.24),
    (0.38, -0.08, 0.22),
)


class PublicationRenderError(RuntimeError):
    """The publication renderer cannot preserve the sealed replay contract."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PublicationRenderError(f"cannot import sealed replay: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_utc_deadline(value: str, *, now: float | None = None) -> float:
    if not value.endswith("Z"):
        raise PublicationRenderError("--deadline-utc must end in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise PublicationRenderError("--deadline-utc is not valid ISO 8601") from error
    if parsed.tzinfo != timezone.utc:
        raise PublicationRenderError("--deadline-utc must be UTC")
    current = time.time() if now is None else now
    deadline = parsed.timestamp()
    if deadline <= current + 60.0:
        raise PublicationRenderError("--deadline-utc must leave at least 60 seconds")
    if deadline > current + 4.0 * 60.0 * 60.0:
        raise PublicationRenderError("--deadline-utc may be at most four hours ahead")
    return deadline


def _finite_vector(values: Sequence[float], field: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise PublicationRenderError(f"{field} must contain three numbers")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise PublicationRenderError(f"{field} must be finite")
    return vector  # type: ignore[return-value]


def _tensor_row(value: Any, index: int = 0) -> tuple[float, ...]:
    selected = value[index]
    if hasattr(selected, "detach"):
        selected = selected.detach().cpu()
    if hasattr(selected, "tolist"):
        selected = selected.tolist()
    return tuple(float(item) for item in selected)


def _rotate_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    w, x, y, z = (float(value) for value in quaternion)
    vx, vy, vz = (float(value) for value in vector)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not math.isfinite(norm) or norm <= 0.0:
        raise PublicationRenderError("object quaternion is invalid")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def _world_frame_position(env: Any, body_name: str, frame_name: str) -> tuple[float, float, float]:
    body = getattr(env, body_name)
    root_position = _tensor_row(body.data.root_pos_w)
    root_quaternion = _tensor_row(body.data.root_quat_w)
    frame = env.cfg.rj45_task["frames"][frame_name]
    offset = _rotate_wxyz(root_quaternion, frame["position_m"])
    return tuple(root_position[index] + offset[index] for index in range(3))  # type: ignore[return-value]


def connection_center(env: Any) -> tuple[float, float, float]:
    """Return the midpoint between the live socket entry and plug tip."""

    entry = _world_frame_position(env, "socket", "socket_entry")
    tip = _world_frame_position(env, "plug", "plug_tip")
    return tuple((entry[index] + tip[index]) / 2.0 for index in range(3))  # type: ignore[return-value]


def connection_positions(env: Any) -> dict[str, list[float]]:
    entry = _world_frame_position(env, "socket", "socket_entry")
    tip = _world_frame_position(env, "plug", "plug_tip")
    center = tuple((entry[index] + tip[index]) / 2.0 for index in range(3))
    return {
        "socket_entry_world_m": list(entry),
        "plug_tip_world_m": list(tip),
        "connection_center_world_m": list(center),
    }


def camera_from_connection(
    env: Any,
    eye_offset: Sequence[float],
    lookat_offset: Sequence[float],
) -> dict[str, tuple[float, float, float]]:
    return camera_from_center(connection_center(env), eye_offset, lookat_offset)


def camera_from_center(
    center: Sequence[float],
    eye_offset: Sequence[float],
    lookat_offset: Sequence[float],
) -> dict[str, tuple[float, float, float]]:
    eye = tuple(center[index] + float(eye_offset[index]) for index in range(3))
    lookat = tuple(center[index] + float(lookat_offset[index]) for index in range(3))
    return {"eye_world_m": eye, "lookat_world_m": lookat}


def _camera_center(
    env: Any, center_world: tuple[float, float, float] | None
) -> tuple[float, float, float]:
    return connection_center(env) if center_world is None else center_world


def _warmup_complete(env: Any) -> bool:
    value = env._training_warmup_complete[0]
    if hasattr(value, "item"):
        value = value.item()
    return value is True


def apply_publication_appearance(stage: Any) -> dict[str, Any]:
    """Apply render-only USD attributes to the in-memory composed stage."""

    from pxr import Gf, Usd, UsdGeom, UsdLux

    table_paths: list[str] = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.endswith("/Table/top") and prim.IsA(UsdGeom.Gprim):
            table_paths.append(path)
    if not table_paths:
        raise PublicationRenderError("composed stage has no Table/top visual prim")

    light_prim = stage.GetPrimAtPath("/World/Light")
    if not light_prim.IsValid():
        raise PublicationRenderError("composed stage has no /World/Light")
    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, session_layer):
        for path in table_paths:
            UsdGeom.Gprim(stage.GetPrimAtPath(path)).CreateDisplayColorAttr().Set(
                [Gf.Vec3f(*TABLE_COLOR)]
            )
        light = UsdLux.LightAPI(stage.GetPrimAtPath("/World/Light"))
        light.CreateIntensityAttr(LIGHT_INTENSITY)
        light.CreateExposureAttr(LIGHT_EXPOSURE)
    return {
        "edit_layer": session_layer.identifier,
        "table_paths": table_paths,
        "table_display_color_rgb": list(TABLE_COLOR),
        "light_path": "/World/Light",
        "light_intensity": LIGHT_INTENSITY,
        "light_exposure": LIGHT_EXPOSURE,
        "physics_or_action_contract_modified": False,
    }


def _write_previews(
    env: Any,
    preview_dir: Path,
    center_world: tuple[float, float, float],
    lookat_offset: Sequence[float],
) -> list[dict[str, Any]]:
    import imageio.v2 as imageio

    preview_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, eye_offset in enumerate(PREVIEW_EYE_OFFSETS, 1):
        camera = camera_from_center(center_world, eye_offset, lookat_offset)
        env.sim.set_camera_view(camera["eye_world_m"], camera["lookat_world_m"])
        env.render(recompute=True)
        for _ in range(4):
            env.sim.render()
        frame = env.render(recompute=True)
        if frame is None or frame.size == 0:
            raise PublicationRenderError("preview renderer produced an empty frame")
        output = preview_dir / f"camera-{index}.png"
        if output.exists():
            raise PublicationRenderError(f"preview output already exists: {output}")
        imageio.imwrite(output, frame)
        records.append({
            "path": str(output),
            "camera": camera,
            "sha256": _file_sha256(output),
        })
    return records


def _patch_renderer(
    renderer: Any,
    *,
    eye_offset: tuple[float, float, float],
    lookat_offset: tuple[float, float, float],
    center_world: tuple[float, float, float] | None,
    preview_dir: Path | None,
    state: dict[str, Any],
) -> None:
    original_build = renderer._build_render_env
    original_renderer_load = renderer._load_module

    def build_render_env(*args: Any, **kwargs: Any) -> Any:
        env, context, frozen = original_build(*args, **kwargs)
        from isaacsim.core.utils.stage import get_current_stage

        state["env"] = env
        state["appearance"] = apply_publication_appearance(get_current_stage())
        return env, context, frozen

    def renderer_load(path: Path, name: str) -> Any:
        module = original_renderer_load(path, name)
        if Path(path).name != "rj45_evaluation_video.py":
            return module
        original_read_frame = module.EvaluationVideoRecorder._read_frame

        def selected_environment_camera(
            _origins: Any, _env_index: int
        ) -> dict[str, tuple[float, float, float]]:
            env = state.get("env")
            if env is None:
                raise PublicationRenderError("render environment is unavailable")
            initial_positions = connection_positions(env)
            state["connection_positions_at_camera_initialization"] = initial_positions
            center = _camera_center(env, center_world)
            state["camera_center_world_m"] = list(center)
            camera = camera_from_center(center, eye_offset, lookat_offset)
            state["selected_camera"] = camera
            return camera

        def read_frame(recorder: Any) -> Any:
            should_preview = (
                preview_dir is not None
                and "previews" not in state
                and recorder._capture_active is True
                and _warmup_complete(recorder.env)
            )
            if should_preview:
                preview_positions = connection_positions(recorder.env)
                state["connection_positions_at_preview"] = preview_positions
                center = _camera_center(recorder.env, center_world)
                state["previews"] = _write_previews(
                    recorder.env,
                    preview_dir,
                    center,
                    lookat_offset,
                )
                selected_camera = camera_from_center(center, eye_offset, lookat_offset)
                recorder.env.sim.set_camera_view(
                    selected_camera["eye_world_m"],
                    selected_camera["lookat_world_m"],
                )
                state["selected_camera"] = selected_camera
            return original_read_frame(recorder)

        module.selected_environment_camera = selected_environment_camera
        module.EvaluationVideoRecorder._read_frame = read_frame
        return module

    renderer._build_render_env = build_render_env
    renderer._load_module = renderer_load


def _patch_sealed_replay(
    replay: Any,
    *,
    deadline_epoch: float,
    eye_offset: tuple[float, float, float],
    lookat_offset: tuple[float, float, float],
    center_world: tuple[float, float, float] | None,
    preview_dir: Path | None,
    publication_report: Path,
    state: dict[str, Any],
) -> None:
    original_load = replay._load_module
    original_persist = replay.persist_report_before_app_close

    def replay_load(path: Path, name: str) -> Any:
        module = original_load(path, name)
        if Path(path).name == "record_rj45_oracle_probe.py":
            _patch_renderer(
                module,
                eye_offset=eye_offset,
                lookat_offset=lookat_offset,
                center_world=center_world,
                preview_dir=preview_dir,
                state=state,
            )
        return module

    replay._deadline_epoch = lambda: deadline_epoch
    replay._load_module = replay_load

    def persist_report_before_app_close(
        report_path: Path,
        sealed_report: dict[str, Any],
        multibias: Any,
        probe: Any,
        app: Any,
        exit_code: int,
    ) -> int:
        sealed_bytes = (
            json.dumps(sealed_report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        state["sealed_report_link"] = {
            "path": str(report_path),
            "expected_sha256": hashlib.sha256(sealed_bytes).hexdigest(),
            "status": sealed_report.get("status"),
            "lineage": sealed_report.get("lineage"),
            "comparison": sealed_report.get("comparison"),
        }
        state["sealed_exit_code_before_app_close"] = exit_code
        try:
            _write_publication_report(publication_report, state)
        except BaseException as error:
            state["publication_report_error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
            return original_persist(
                report_path, sealed_report, multibias, probe, app, 1
            )
        closed_exit_code = original_persist(
            report_path, sealed_report, multibias, probe, app, exit_code
        )
        state["app_close_exit_code"] = closed_exit_code
        if report_path.is_file():
            state["sealed_report_link"]["sha256"] = _file_sha256(report_path)
            state["sealed_report_link"]["bytes"] = report_path.stat().st_size
        _replace_publication_report(publication_report, state)
        return closed_exit_code

    replay.persist_report_before_app_close = persist_report_before_app_close


def _argument_value(arguments: Sequence[str], flag: str) -> str | None:
    try:
        index = arguments.index(flag)
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def _artifact_record(path_text: str | None) -> dict[str, Any] | None:
    if path_text is None:
        return None
    path = Path(path_text).resolve()
    record: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        record.update({"bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    return record


def _preview_evidence_valid(state: Mapping[str, Any]) -> bool:
    if state.get("preview_required") is not True:
        return True
    records = state.get("previews")
    if not isinstance(records, list) or len(records) != len(PREVIEW_EYE_OFFSETS):
        return False
    for record in records:
        if not isinstance(record, dict):
            return False
        path = Path(str(record.get("path")))
        if not path.is_file() or record.get("sha256") != _file_sha256(path):
            return False
    return True


def _publication_payload(state: dict[str, Any]) -> dict[str, Any]:
    sealed_link = state.get("sealed_report_link")
    sealed_status = sealed_link.get("status") if isinstance(sealed_link, dict) else None
    exit_code = state.get(
        "app_close_exit_code", state.get("sealed_exit_code_before_app_close")
    )
    video_record = _artifact_record(state.get("sealed_video_path"))
    close_recorded = "app_close_exit_code" in state
    report_hash_matches = bool(
        isinstance(sealed_link, dict)
        and sealed_link.get("sha256") == sealed_link.get("expected_sha256")
    )
    video_exists = bool(isinstance(video_record, dict) and video_record.get("exists"))
    previews_valid = _preview_evidence_valid(state)
    if (
        not close_recorded
        and sealed_status == "completed"
        and exit_code == 0
        and previews_valid
    ):
        status = "pending_app_close"
    elif (
        close_recorded
        and exit_code == 0
        and sealed_status == "completed"
        and report_hash_matches
        and video_exists
        and previews_valid
    ):
        status = "completed"
    else:
        status = "invalid"
    return {
        "schema_name": SCHEMA_NAME,
        "status": status,
        "scope": "publication visualization replay only",
        "wrapper": state["wrapper"],
        "sealed_replay_source": state["sealed_replay_source"],
        "sealed_run_arguments": state["sealed_run_arguments"],
        "publication_deadline_utc": state["publication_deadline_utc"],
        "appearance": state.get("appearance"),
        "camera_center": state.get("camera_center"),
        "camera_center_world_m": state.get("camera_center_world_m"),
        "connection_positions_at_camera_initialization": state.get(
            "connection_positions_at_camera_initialization"
        ),
        "connection_positions_at_preview": state.get("connection_positions_at_preview"),
        "selected_camera": state.get("selected_camera"),
        "preview_images": state.get("previews", []),
        "preview_required": state.get("preview_required") is True,
        "sealed_report": sealed_link,
        "sealed_video": video_record,
        "app_close_exit_code": state.get("app_close_exit_code"),
        "claims": {
            "benchmark_result_replaced": False,
            "training_evidence": False,
            "physics_or_action_contract_modified": False,
        },
    }


def _write_publication_report(path: Path, state: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"publication report already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_publication_payload(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _replace_publication_report(path: Path, state: dict[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"publication report is missing: {path}")
    temporary = path.with_suffix(path.suffix + ".post-close.tmp")
    temporary.write_text(
        json.dumps(_publication_payload(state), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationRenderError(f"{field} is not valid UTF-8 JSON: {path}") from error
    if not isinstance(value, dict):
        raise PublicationRenderError(f"{field} root must be an object")
    return value


def finalize_from_native_exit(
    publication_report: Path, exit_record: Path
) -> dict[str, Any]:
    """Finalize a pre-close sidecar using the outer process exit witness."""

    publication_report = publication_report.resolve()
    exit_record = exit_record.resolve()
    publication = _read_json_object(publication_report, "publication report")
    if publication.get("schema_name") != SCHEMA_NAME:
        raise PublicationRenderError("publication report schema changed")
    if publication.get("status") != "pending_app_close":
        raise PublicationRenderError("publication report is not pending app close")
    wrapper = publication.get("wrapper")
    current_wrapper = Path(__file__).resolve()
    if not isinstance(wrapper, dict) or wrapper.get("sha256") != _file_sha256(
        current_wrapper
    ):
        raise PublicationRenderError("publication wrapper source hash mismatch")
    witness = _read_json_object(exit_record, "native exit record")
    if type(witness.get("exit_code")) is not int or witness["exit_code"] != 0:
        raise PublicationRenderError("native process exit code is not zero")

    sealed_report = publication.get("sealed_report")
    if not isinstance(sealed_report, dict):
        raise PublicationRenderError("sealed report binding is missing")
    report_path = Path(str(sealed_report.get("path"))).resolve()
    if not report_path.is_file():
        raise PublicationRenderError("sealed report file is missing")
    report_sha256 = _file_sha256(report_path)
    if report_sha256 != sealed_report.get("expected_sha256"):
        raise PublicationRenderError("sealed report hash differs from pre-close expectation")
    report = _read_json_object(report_path, "sealed replay report")
    if report.get("status") != "completed":
        raise PublicationRenderError("sealed replay report is not completed")

    sealed_video = publication.get("sealed_video")
    if not isinstance(sealed_video, dict):
        raise PublicationRenderError("sealed video binding is missing")
    video_path = Path(str(sealed_video.get("path"))).resolve()
    if not video_path.is_file():
        raise PublicationRenderError("sealed video file is missing")
    video_sha256 = _file_sha256(video_path)
    if video_sha256 != sealed_video.get("sha256"):
        raise PublicationRenderError("sealed video hash differs from pre-close record")
    if publication.get("preview_required") is True:
        previews = publication.get("preview_images")
        if not isinstance(previews, list) or len(previews) != len(PREVIEW_EYE_OFFSETS):
            raise PublicationRenderError("publication previews are incomplete")
        for preview in previews:
            if not isinstance(preview, dict):
                raise PublicationRenderError("publication preview record is invalid")
            preview_path = Path(str(preview.get("path"))).resolve()
            if not preview_path.is_file() or preview.get("sha256") != _file_sha256(
                preview_path
            ):
                raise PublicationRenderError("publication preview hash mismatch")

    publication["status"] = "completed"
    publication["native_process_exit_code"] = 0
    publication["completion_witness"] = "outer_runner_native_exit"
    publication["native_exit_record"] = {
        "path": str(exit_record),
        "bytes": exit_record.stat().st_size,
        "sha256": _file_sha256(exit_record),
    }
    sealed_report["sha256"] = report_sha256
    sealed_report["bytes"] = report_path.stat().st_size
    sealed_video["exists"] = True
    sealed_video["bytes"] = video_path.stat().st_size
    temporary = publication_report.with_suffix(publication_report.suffix + ".finalize.tmp")
    temporary.write_text(
        json.dumps(publication, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(publication_report)
    return publication


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=True)
    parser.add_argument("--sealed-replay", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    parser.add_argument("--publication-report", type=Path, required=True)
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--eye-offset", nargs=3, type=float, default=DEFAULT_EYE_OFFSET)
    parser.add_argument("--lookat-offset", nargs=3, type=float, default=DEFAULT_LOOKAT_OFFSET)
    parser.add_argument("--center-world", nargs=3, type=float)
    parser.add_argument("--camera-center-source", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    wrapper_args, sealed_args = _parser().parse_known_args(argv)
    sealed_replay = wrapper_args.sealed_replay.resolve()
    publication_report = wrapper_args.publication_report.resolve()
    preview_dir = wrapper_args.preview_dir.resolve() if wrapper_args.preview_dir else None
    if publication_report.exists():
        raise PublicationRenderError(f"publication report already exists: {publication_report}")
    observed_sha = _file_sha256(sealed_replay)
    if observed_sha != SEALED_REPLAY_SHA256:
        raise PublicationRenderError("sealed replay source SHA-256 mismatch")
    deadline_epoch = _parse_utc_deadline(wrapper_args.deadline_utc)
    eye_offset = _finite_vector(wrapper_args.eye_offset, "--eye-offset")
    lookat_offset = _finite_vector(wrapper_args.lookat_offset, "--lookat-offset")
    center_world = (
        _finite_vector(wrapper_args.center_world, "--center-world")
        if wrapper_args.center_world is not None
        else None
    )
    camera_center_source = (
        wrapper_args.camera_center_source.resolve()
        if wrapper_args.camera_center_source is not None
        else None
    )
    if camera_center_source is not None and center_world is None:
        raise PublicationRenderError("--camera-center-source requires --center-world")
    if camera_center_source is not None and not camera_center_source.is_file():
        raise PublicationRenderError(
            f"camera center source does not exist: {camera_center_source}"
        )

    wrapper_path = Path(__file__).resolve()
    replay = _load_module(sealed_replay, "rj45_publication_sealed_replay")
    state: dict[str, Any] = {
        "wrapper": {"path": str(wrapper_path), "sha256": _file_sha256(wrapper_path)},
        "sealed_replay_source": {"path": str(sealed_replay), "sha256": observed_sha},
        "sealed_run_arguments": list(sealed_args),
        "sealed_video_path": _argument_value(sealed_args, "--video"),
        "publication_deadline_utc": wrapper_args.deadline_utc,
        "preview_required": preview_dir is not None,
        "camera_center": {
            "mode": "explicit_world" if center_world is not None else "live_connection",
            "world_m": list(center_world) if center_world is not None else None,
            "source": (
                {
                    "path": str(camera_center_source),
                    "sha256": _file_sha256(camera_center_source),
                    "bytes": camera_center_source.stat().st_size,
                }
                if camera_center_source is not None
                else None
            ),
        },
    }
    _patch_sealed_replay(
        replay,
        deadline_epoch=deadline_epoch,
        eye_offset=eye_offset,
        lookat_offset=lookat_offset,
        center_world=center_world,
        preview_dir=preview_dir,
        publication_report=publication_report,
        state=state,
    )
    exit_code = replay.main(sealed_args)
    if not publication_report.exists():
        state["sealed_exit_code_before_app_close"] = exit_code
        state["sealed_report_link"] = _artifact_record(
            _argument_value(sealed_args, "--report")
        )
        _write_publication_report(publication_report, state)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
