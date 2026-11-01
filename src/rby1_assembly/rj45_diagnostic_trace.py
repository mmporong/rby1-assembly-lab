"""Fail-closed, audit-only RJ45 control-step diagnostic tracing."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


class DiagnosticTraceError(RuntimeError):
    """The diagnostic trace cannot preserve its timing or source contract."""


def _array(value: Any, name: str, shape: tuple[int, ...]) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    result = np.asarray(value)
    if result.shape != shape:
        raise DiagnosticTraceError(f"{name} must have shape {shape}, got {result.shape}")
    if not np.issubdtype(result.dtype, np.number) or not np.isfinite(result).all():
        raise DiagnosticTraceError(f"{name} must contain only finite numeric values")
    return result.copy()


def _boolean(value: Any, name: str) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    result = np.asarray(value)
    if result.shape != (1,) or result.dtype.kind != "b":
        raise DiagnosticTraceError(f"{name} must have boolean shape (1,)")
    return bool(result[0])


def _scalar(value: Any, name: str, *, integer: bool = False) -> float | int:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    result = np.asarray(value)
    if result.shape not in ((), (1,)):
        raise DiagnosticTraceError(f"{name} must be scalar or shape (1,)")
    item = result.reshape(-1)[0] if result.shape else result.item()
    if integer:
        try:
            converted = int(item)
        except (TypeError, ValueError, OverflowError) as error:
            raise DiagnosticTraceError(f"{name} must be an integer") from error
        if isinstance(item, (bool, np.bool_)) or converted != item:
            raise DiagnosticTraceError(f"{name} must be an integer")
        return converted
    try:
        value_float = float(item)
    except (TypeError, ValueError, OverflowError) as error:
        raise DiagnosticTraceError(f"{name} must be finite") from error
    if not math.isfinite(value_float):
        raise DiagnosticTraceError(f"{name} must be finite")
    return value_float


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


class JsonlTraceWriter:
    """Append an authenticated one-row-per-step stream and immutable sidecar."""

    def __init__(self, path: Path, sidecar_path: Path) -> None:
        self.path = Path(path)
        self.sidecar_path = Path(sidecar_path)
        if self.path.exists() or self.sidecar_path.exists():
            raise FileExistsError("diagnostic trace output already exists")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._previous_sha256: str | None = None
        self._row_count = 0
        self._terminal_count = 0

    def append(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "sequence": self._row_count,
            "previous_row_sha256": self._previous_sha256,
            **payload,
        }
        row_sha256 = hashlib.sha256(_canonical_bytes(row)).hexdigest()
        sealed = {**row, "row_sha256": row_sha256}
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(sealed, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._row_count += 1
        self._previous_sha256 = row_sha256
        self._terminal_count += int(bool(payload.get("terminal")))
        return sealed

    def finalize(self) -> dict[str, Any]:
        if self._row_count < 1:
            raise DiagnosticTraceError("diagnostic trace must contain at least one row")
        if self._terminal_count != 1:
            raise DiagnosticTraceError("diagnostic trace must contain exactly one terminal row")
        rows = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DiagnosticTraceError(
                    f"diagnostic trace row {line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise DiagnosticTraceError("diagnostic trace row must be an object")
            rows.append(row)
        if len(rows) != self._row_count:
            raise DiagnosticTraceError("diagnostic trace row count differs from writer state")
        previous = None
        terminal_rows = []
        for sequence, row in enumerate(rows):
            claimed = row.get("row_sha256")
            unhashed = {key: value for key, value in row.items() if key != "row_sha256"}
            actual = hashlib.sha256(_canonical_bytes(unhashed)).hexdigest()
            if (
                row.get("sequence") != sequence
                or row.get("previous_row_sha256") != previous
                or claimed != actual
            ):
                raise DiagnosticTraceError("diagnostic trace hash chain is invalid")
            if row.get("terminal") is True:
                terminal_rows.append(row)
            previous = claimed
        if (
            len(terminal_rows) != 1
            or terminal_rows[0] is not rows[-1]
            or terminal_rows[0].get("post_state_source")
            != "pre_autoreset_terminal_snapshot"
            or previous != self._previous_sha256
        ):
            raise DiagnosticTraceError("diagnostic terminal row or final digest is invalid")
        payload = {
            "schema_name": "rj45-contact-diagnostic-trace-sidecar-v1",
            "trace_path": str(self.path.resolve()),
            "trace_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest(),
            "row_count": self._row_count,
            "last_row_sha256": self._previous_sha256,
            "terminal_count": self._terminal_count,
            "terminal_source": "pre_autoreset_terminal_snapshot",
        }
        temporary = self.sidecar_path.with_name(f".{self.sidecar_path.name}.{os.getpid()}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, self.sidecar_path)
        temporary.unlink()
        return payload


class ContactDiagnosticObserver:
    """Capture one immutable pre-action/post-physics pair for every env step."""

    _POST_STEP_FIELDS = (
        ("policy_input_observation", (1, 79)),
        ("policy_input_timestamp_s", (1,)),
        ("requested_expanded_residual_action", (1, 12)),
        ("applied_bounded_residual_action", (1, 12)),
        ("applied_tip_target_offset_entry_6d", (1, 6)),
    )

    def __init__(
        self,
        writer: JsonlTraceWriter,
        geometry: Any,
        *,
        live_snapshot: Callable[..., Any],
        terminal_snapshot_from_extras: Callable[..., Any],
    ) -> None:
        self.writer = writer
        self.geometry = geometry
        self._live_snapshot = live_snapshot
        self._terminal_snapshot_from_extras = terminal_snapshot_from_extras
        self._pending: dict[str, Any] | None = None
        self._step_index = 0
        self._terminal_seen = False
        self._control_step_dt_s: float | None = None

    def capture_pre(
        self,
        env: Any,
        policy: Any,
        commanded_action: Any,
        policy_input_valid: Any,
        policy_input_timestamp_s: Any,
    ) -> None:
        if self._terminal_seen or self._pending is not None:
            raise DiagnosticTraceError("pre-step capture order is invalid")
        observation = _array(policy, "policy", (1, 79))
        timestamp_s = float(
            _scalar(policy_input_timestamp_s, "policy_input_timestamp_s")
        )
        phase = _scalar(getattr(env, "_phase", None), "pre_phase", integer=True)
        step_dt_s = float(_scalar(getattr(env, "step_dt", None), "step_dt"))
        if step_dt_s <= 0.0:
            raise DiagnosticTraceError("step_dt must be positive")
        if self._control_step_dt_s is None:
            self._control_step_dt_s = step_dt_s
        elif step_dt_s != self._control_step_dt_s:
            raise DiagnosticTraceError("step_dt changed during diagnostic trace")
        self._pending = {
            "observer_pre_step_index": self._step_index,
            "observer_pre_control_elapsed_s": self._step_index * step_dt_s,
            "control_step_dt_s": step_dt_s,
            "pre_policy_timestamp_s": timestamp_s,
            "pre_phase_code": phase,
            "pre_policy_input_valid": _boolean(
                policy_input_valid, "policy_input_valid"
            ),
            "policy_observation79": observation.tolist()[0],
            "policy_command_depth": float(observation[0, 62]),
            "policy_advance": float(observation[0, 63]),
            "commanded_normalized_action6": _array(
                commanded_action, "commanded_action", (1, 6)
            ).tolist()[0],
        }

    def capture_post(self, env: Any, extras: Mapping[str, Any], terminal: bool) -> None:
        if self._pending is None or self._terminal_seen:
            raise DiagnosticTraceError("post-step capture order is invalid")
        if not isinstance(extras, Mapping):
            raise DiagnosticTraceError("step extras must be a mapping")
        raw_step = extras.get("terminal_snapshot")
        if not isinstance(raw_step, Mapping):
            raise DiagnosticTraceError(
                "every post-step capture requires the pre-reset terminal_snapshot clone"
            )
        required = {name for name, _ in self._POST_STEP_FIELDS} | {
            "policy_input_valid"
        }
        missing = sorted(required - set(raw_step))
        if missing:
            raise DiagnosticTraceError(
                "post-step training snapshot is missing fields: " + ", ".join(missing)
            )
        current = {
            name: _array(raw_step[name], name, shape)
            for name, shape in self._POST_STEP_FIELDS
        }
        if not np.array_equal(
            current["policy_input_observation"][0],
            np.asarray(self._pending["policy_observation79"]),
        ):
            raise DiagnosticTraceError(
                "post-step snapshot policy input differs from the caller policy"
            )
        if (
            float(current["policy_input_timestamp_s"][0])
            != self._pending["pre_policy_timestamp_s"]
        ):
            raise DiagnosticTraceError(
                "post-step snapshot policy timestamp differs from the caller timestamp"
            )
        if (
            _boolean(raw_step["policy_input_valid"], "policy_input_valid")
            != self._pending["pre_policy_input_valid"]
        ):
            raise DiagnosticTraceError(
                "post-step snapshot policy validity differs from the caller validity"
            )
        if terminal:
            snapshot = self._terminal_snapshot_from_extras(extras, self.geometry)
            post_timestamp_s = _scalar(
                raw_step.get("timestamp_s"), "terminal_snapshot.timestamp_s"
            )
            source = "pre_autoreset_terminal_snapshot"
        else:
            snapshot = self._live_snapshot(env, extras, self.geometry)
            post_timestamp_s = _scalar(
                getattr(env, "episode_length_buf", None), "episode_length_buf"
            ) * _scalar(getattr(env, "step_dt", None), "step_dt")
            source = "post_step_live_snapshot"
        force_vector = _array(
            snapshot.insertion_contact_force_w,
            "insertion_contact_force_w",
            (3,),
        )
        force_norm = float(np.linalg.norm(force_vector))
        if not math.isclose(
            force_norm,
            float(snapshot.insertion_contact_force_n),
            rel_tol=1e-5,
            abs_tol=1e-5,
        ):
            raise DiagnosticTraceError("insertion net-force norm differs from its vector")
        if float(snapshot.insertion_contact_load_n) + 1e-4 < force_norm:
            raise DiagnosticTraceError("normal-load sum is below insertion net-force norm")
        reported = snapshot.reported_ground_truth
        reported_values = {
            "actual_depth_m": float(reported.actual_depth_m),
            "lateral_error_m": float(reported.lateral_error_m),
            "orientation_error_rad": float(reported.orientation_error_rad),
        }
        recomputed_values = {
            "actual_depth_m": float(snapshot.ground_truth.actual_depth_m),
            "lateral_error_m": float(snapshot.ground_truth.lateral_error_m),
            "orientation_error_rad": float(snapshot.ground_truth.orientation_error_rad),
        }
        geometry_consistent = snapshot.reported_geometry_consistent
        if type(geometry_consistent) is not bool:
            raise DiagnosticTraceError("reported geometry consistency must be boolean")
        legacy_values = {
            "actual_depth_m": float(snapshot.legacy_y_only_ground_truth.actual_depth_m),
            "lateral_error_m": float(
                snapshot.legacy_y_only_ground_truth.lateral_error_m
            ),
            "orientation_error_rad": float(
                snapshot.legacy_y_only_ground_truth.orientation_error_rad
            ),
        }
        legacy_consistent = snapshot.legacy_y_only_geometry_consistent
        if type(legacy_consistent) is not bool:
            raise DiagnosticTraceError("legacy geometry consistency must be boolean")
        metric_tolerance_m = float(snapshot.geometry_metric_tolerance_m)
        angular_tolerance_rad = float(snapshot.geometry_angular_tolerance_rad)
        if (
            not math.isfinite(metric_tolerance_m)
            or metric_tolerance_m <= 0.0
            or not math.isfinite(angular_tolerance_rad)
            or angular_tolerance_rad <= 0.0
        ):
            raise DiagnosticTraceError("geometry tolerances must be finite and positive")
        row = {
            "schema_name": "rj45-contact-diagnostic-tick-v2",
            **self._pending,
            "observer_post_step_index": self._step_index + 1,
            "observer_post_control_elapsed_s": (
                (self._step_index + 1) * self._pending["control_step_dt_s"]
            ),
            "post_source_timestamp_s": post_timestamp_s,
            "source_timestamp_delta_s": (
                post_timestamp_s - self._pending["pre_policy_timestamp_s"]
            ),
            "current_requested_expanded_residual_action12": current[
                "requested_expanded_residual_action"
            ][0].tolist(),
            "current_applied_bounded_expanded_action12": current[
                "applied_bounded_residual_action"
            ][0].tolist(),
            "current_applied_tip_target_offset_entry6": current[
                "applied_tip_target_offset_entry_6d"
            ][0].tolist(),
            "post_phase_name": snapshot.phase,
            "post_state_source": source,
            "terminal": bool(terminal),
            "post_physical_gt": recomputed_values,
            "reported_physical_gt": reported_values,
            "reported_geometry_consistent": geometry_consistent,
            "legacy_y_only_physical_gt": legacy_values,
            "legacy_y_only_geometry_consistent": legacy_consistent,
            "geometry_metric_tolerance_m": metric_tolerance_m,
            "geometry_angular_tolerance_rad": angular_tolerance_rad,
            "reported_minus_recomputed_gt": {
                key: reported_values[key] - recomputed_values[key]
                for key in recomputed_values
            },
            "insertion_net_force_world_n": force_vector.tolist(),
            "insertion_net_force_norm_n": force_norm,
            "insertion_normal_load_sum_n": float(snapshot.insertion_contact_load_n),
            "grip_contact_proxy_n": _array(
                snapshot.contact_force_proxy_n,
                "contact_force_proxy_n",
                (2,),
            ).tolist(),
        }
        self.writer.append(row)
        self._pending = None
        self._step_index += 1
        self._terminal_seen = bool(terminal)

    def finalize(self) -> dict[str, Any]:
        if self._pending is not None:
            raise DiagnosticTraceError("trace ended with an unmatched pre-step capture")
        if not self._terminal_seen:
            raise DiagnosticTraceError("trace ended without a terminal snapshot")
        return self.writer.finalize()


__all__ = [
    "ContactDiagnosticObserver",
    "DiagnosticTraceError",
    "JsonlTraceWriter",
]
