"""Case-blind active translation-axis probe built on the sealed Z state machine."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


_BASE_PROVIDER_PATH = Path(__file__).with_name("rj45_active_z_probe.py").resolve()
_BASE_PROVIDER_SOURCE_SHA256 = hashlib.sha256(_BASE_PROVIDER_PATH.read_bytes()).hexdigest()
_BASE_MODULE_NAME = f"rby1_active_axis_base_{_BASE_PROVIDER_SOURCE_SHA256[:16]}"
_base_spec = importlib.util.spec_from_file_location(_BASE_MODULE_NAME, _BASE_PROVIDER_PATH)
if _base_spec is None or _base_spec.loader is None:
    raise RuntimeError(f"cannot load active-Z base provider: {_BASE_PROVIDER_PATH}")
_base_module = importlib.util.module_from_spec(_base_spec)
sys.modules[_BASE_MODULE_NAME] = _base_module
_base_spec.loader.exec_module(_base_module)

ACTION_DIM = _base_module.ACTION_DIM
BASE_ACTION_INDEX = _base_module.ACTION_Z_INDEX
_AXIS_TO_ACTION_INDEX = {"tx": 0, "tz": 2}
_AXIS_TO_DIRECTION_SUFFIX = {"tx": "x", "tz": "z"}


class ActiveAxisProbeProvider:
    """Expose the sealed probe on X or Z without accepting experiment identity."""

    def __init__(self, axis: str) -> None:
        if not isinstance(axis, str) or axis not in _AXIS_TO_ACTION_INDEX:
            raise ValueError("axis must be exactly 'tx' or 'tz'")
        self._axis = axis
        self._action_index = _AXIS_TO_ACTION_INDEX[axis]
        self._base = _base_module.ActiveZProbeProvider()

    @property
    def axis(self) -> str:
        return self._axis

    def reset(self) -> Any:
        return self._base.reset()

    def act(self, observation79: Sequence[float], state: Any) -> tuple[np.ndarray, Any]:
        base_action, next_state = self._base.act(observation79, state)
        if self._action_index == BASE_ACTION_INDEX:
            return base_action.copy(), next_state
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[self._action_index] = base_action[BASE_ACTION_INDEX]
        return action, next_state

    def state_telemetry(self, state: Any) -> dict[str, Any]:
        return self._base.state_telemetry(state)

    def canonical_payload(self) -> dict[str, Any]:
        payload = self._base.canonical_payload()
        payload.pop("diagnostic_episode_repeat_count", None)
        payload.pop("action_z_index", None)
        suffix = _AXIS_TO_DIRECTION_SUFFIX[self._axis]
        payload.update(
            {
                "schema": "rj45-active-axis-probe-controller-v2",
                "probe_axis": self._axis,
                "action_translation_index": self._action_index,
                "direction_order": [f"positive_{suffix}", f"negative_{suffix}"],
                "waveform_repeat_count": 1,
                "base_provider_schema": "rj45-active-z-probe-controller-v1",
                "base_provider_controller_sha256": self._base.controller_sha256(),
                "base_provider_source_sha256": _BASE_PROVIDER_SOURCE_SHA256,
            }
        )
        return payload

    def controller_sha256(self) -> str:
        encoded = json.dumps(
            self.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["ActiveAxisProbeProvider"]
