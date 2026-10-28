"""Generation-only RJ45 grasp-target catalog and environment wrapper."""

from __future__ import annotations

from copy import deepcopy
import itertools
import math
from pathlib import Path
from typing import Any, Mapping

from rby1_assembly.experiment_contracts import load_finite_json


CATALOG_SCHEMA_VERSION = 2
OFFSET_MAGNITUDE_M = 0.00015
EXCLUDED_SIGN_PATTERNS = ("----", "++++", "-+-+", "+-+-")
OFFSET_METHOD = "object_local_offset_rotated_to_world_by_object_quaternion"


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _strict_equal(lhs, rhs) for lhs, rhs in zip(left, right, strict=True)
        )
    return bool(left == right)


def _expected_offsets() -> list[tuple[int, int, int, int]]:
    return [
        (left_y, left_z, right_y, right_z)
        for left_y, left_z, right_y, right_z in itertools.product((-1, 1), repeat=4)
        if "".join(
            "+" if value > 0 else "-"
            for value in (left_y, left_z, right_y, right_z)
        )
        not in EXCLUDED_SIGN_PATTERNS
    ]


def _expected_seed_assignments() -> dict[str, dict[str, str]]:
    return {
        "pilot": {
            str(seed): f"grasp-{index:02d}"
            for index, seed in enumerate(range(20261301, 20261305))
        },
        "production_train": {
            str(seed): f"grasp-{index if index < 8 else index - 8:02d}"
            for index, seed in enumerate(range(20261211, 20261223))
        },
        "production_dev": {
            str(seed): f"grasp-{8 + (index % 2):02d}"
            for index, seed in enumerate(range(20261231, 20261235))
        },
        "production_final": {
            str(seed): f"grasp-{10 + (index % 2):02d}"
            for index, seed in enumerate(range(20261241, 20261245))
        },
    }


def validate_grasp_generation_catalog(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fixed 12-row physical grasp-generation contract."""

    catalog = deepcopy(dict(value))
    required = {
        "schema_version",
        "catalog_id",
        "offset_method",
        "offset_magnitude_m",
        "coordinate_sign_order",
        "excluded_sign_patterns",
        "entries",
        "seed_assignments",
        "static_cad_audit",
    }
    if set(catalog) != required:
        raise ValueError(f"grasp generation catalog must contain exactly {sorted(required)}")
    if (
        type(catalog["schema_version"]) is not int
        or catalog["schema_version"] != CATALOG_SCHEMA_VERSION
    ):
        raise ValueError("unsupported grasp generation catalog schema")
    if catalog["catalog_id"] != "rj45-grasp-generation-v2":
        raise ValueError("unexpected grasp generation catalog ID")
    if catalog["offset_method"] != OFFSET_METHOD:
        raise ValueError("unexpected grasp generation offset method")
    magnitude = catalog["offset_magnitude_m"]
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise ValueError("grasp generation offset magnitude must be numeric")
    if not math.isfinite(float(magnitude)) or float(magnitude) != OFFSET_MAGNITUDE_M:
        raise ValueError("grasp generation offset magnitude must be 0.15 mm")
    if catalog["coordinate_sign_order"] != [
        "left_y", "left_z", "right_y", "right_z"
    ]:
        raise ValueError("unexpected grasp generation sign order")
    if catalog["excluded_sign_patterns"] != list(EXCLUDED_SIGN_PATTERNS):
        raise ValueError("unexpected excluded grasp-generation sign patterns")

    entries = catalog["entries"]
    expected = _expected_offsets()
    if not isinstance(entries, list) or len(entries) != len(expected):
        raise ValueError("grasp generation catalog must contain 12 rows")
    for index, (entry, signs) in enumerate(zip(entries, expected, strict=True)):
        expected_entry = {
            "id": f"grasp-{index:02d}",
            "left_object_local_offset_m": [
                0.0,
                signs[0] * OFFSET_MAGNITUDE_M,
                signs[1] * OFFSET_MAGNITUDE_M,
            ],
            "right_object_local_offset_m": [
                0.0,
                signs[2] * OFFSET_MAGNITUDE_M,
                signs[3] * OFFSET_MAGNITUDE_M,
            ],
        }
        if not _strict_equal(entry, expected_entry):
            raise ValueError(f"grasp generation catalog row {index} is not canonical")

    if not _strict_equal(catalog["seed_assignments"], _expected_seed_assignments()):
        raise ValueError("grasp generation seed assignments are not canonical")
    audit = catalog["static_cad_audit"]
    expected_audit = {
        "plug_insertion_corridor_baseline_m": 0.001136,
        "plug_insertion_corridor_minimum_m": 0.000986,
        "socket_port_baseline_m": 0.022,
        "socket_port_minimum_m": 0.02185,
        "gripper_convex_hull_vs_locator_dynamic_collision_verified": False,
        "physical_pilot_required": True,
    }
    if not _strict_equal(audit, expected_audit):
        raise ValueError("grasp generation static CAD audit is not canonical")
    return catalog


def load_grasp_generation_catalog(path: Path) -> dict[str, Any]:
    return validate_grasp_generation_catalog(load_finite_json(Path(path)))


def validate_grasp_generation_launch(
    *,
    catalog_path: Path | None,
    task: str,
    num_envs: int,
    seed: int | None,
    command_fault: str | None,
    training_config: Path | None,
    calibration_bias: bool,
    bias_level: int,
) -> None:
    """Fail closed unless the generation wrapper is an isolated scripted run."""

    if catalog_path is None:
        return
    if (
        task != "rj45"
        or num_envs != 1
        or type(seed) is not int
        or command_fault is not None
        or training_config is not None
        or calibration_bias
        or bias_level != 0
    ):
        raise ValueError(
            "grasp generation requires --task rj45, --num-envs 1, an explicit "
            "--seed, and no training, bias, or command-fault wrapper"
        )


def select_grasp_generation_entry(
    catalog: Mapping[str, Any], seed: int
) -> tuple[str, dict[str, Any]]:
    catalog = validate_grasp_generation_catalog(catalog)
    if type(seed) is not int:
        raise ValueError("grasp generation seed must be an explicit integer")
    matches = [
        (group, assignments[str(seed)])
        for group, assignments in catalog["seed_assignments"].items()
        if str(seed) in assignments
    ]
    if len(matches) != 1:
        raise ValueError("seed is not uniquely assigned in the grasp generation catalog")
    group, entry_id = matches[0]
    entry = next(entry for entry in catalog["entries"] if entry["id"] == entry_id)
    return group, deepcopy(entry)


def _rotate_wxyz(quaternion, vector):
    import torch

    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    return (
        vector * (2.0 * scalar.square() - 1.0)
        + 2.0 * scalar * torch.linalg.cross(xyz, vector, dim=-1)
        + 2.0 * xyz * torch.sum(xyz * vector, dim=-1, keepdim=True)
    )


def make_rj45_grasp_generation_env_type(
    base_env_type: type, catalog: Mapping[str, Any], *, seed: int
) -> type:
    """Wrap one scripted RJ45 environment with one predeclared grasp target."""

    catalog = validate_grasp_generation_catalog(catalog)
    assignment_group, entry = select_grasp_generation_entry(catalog, seed)

    class Rj45GraspGenerationEnv(base_env_type):
        def _grasp_target_poses(self):
            import torch

            left_position, left_quaternion, right_position, right_quaternion = (
                super()._grasp_target_poses()
            )
            left_offset = torch.tensor(
                entry["left_object_local_offset_m"],
                device=self.device,
                dtype=left_position.dtype,
            ).expand(self.num_envs, -1)
            right_offset = torch.tensor(
                entry["right_object_local_offset_m"],
                device=self.device,
                dtype=right_position.dtype,
            ).expand(self.num_envs, -1)
            apply_offset = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            if hasattr(self, "_grasp_reference_locked"):
                from rby1_assembly.isaac_env import AssemblyPhase

                frozen = self._grasp_reference_locked & (
                    (self._phase == int(AssemblyPhase.CLOSE))
                    | (self._phase == int(AssemblyPhase.CHECK_LIFT))
                )
                apply_offset &= ~frozen
            left_shifted = left_position + _rotate_wxyz(
                self.socket.data.root_quat_w, left_offset
            )
            right_shifted = right_position + _rotate_wxyz(
                self.plug.data.root_quat_w, right_offset
            )
            return (
                torch.where(apply_offset[:, None], left_shifted, left_position),
                left_quaternion,
                torch.where(apply_offset[:, None], right_shifted, right_position),
                right_quaternion,
            )

        def rj45_grasp_generation_report(self) -> dict[str, Any]:
            return {
                "catalog_id": catalog["catalog_id"],
                "selected_catalog_entry_id": entry["id"],
                "seed": seed,
                "assignment_group": assignment_group,
                "offset_method": catalog["offset_method"],
                "left_object_local_offset_m": deepcopy(
                    entry["left_object_local_offset_m"]
                ),
                "right_object_local_offset_m": deepcopy(
                    entry["right_object_local_offset_m"]
                ),
            }

    Rj45GraspGenerationEnv.__name__ = (
        f"Rj45GraspGeneration{base_env_type.__name__}"
    )
    return Rj45GraspGenerationEnv


__all__ = [
    "load_grasp_generation_catalog",
    "make_rj45_grasp_generation_env_type",
    "select_grasp_generation_entry",
    "validate_grasp_generation_catalog",
    "validate_grasp_generation_launch",
]
