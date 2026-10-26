"""고정된 출처의 시뮬레이션 자산 다운로드와 USD 물리 메타데이터 감사."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Iterable, Mapping, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen


ALLOWED_DOWNLOAD_HOSTS = frozenset(
    {
        "raw.githubusercontent.com",
        "omniverse-content-production.s3-us-west-2.amazonaws.com",
        "www.te.com.cn",
        "www.kroton.com.pe",
    }
)
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_BYTES = 64 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
PHYSICAL_ROLES = frozenset(
    {
        "robot_articulation",
        "end_effector",
        "configuration",
        "visual_background",
        "physical_interaction_static",
        "physical_interaction_dynamic",
        "training_data",
        "candidate_data",
    }
)


class AssetError(RuntimeError):
    """자산 manifest, 다운로드 또는 감사 오류."""


@dataclass(frozen=True)
class AssetRecord:
    id: str
    kind: str
    physical_role: str
    source: str
    url: Optional[str] = None
    path: Optional[str] = None
    size_bytes: Optional[int] = None
    sha256: Optional[str] = None
    redistributable: Optional[bool] = None
    usd: bool = False
    asset_root_path: Optional[str] = None
    verified_url: Optional[str] = None
    verified_size_bytes: Optional[int] = None
    verified_sha256: Optional[str] = None
    verified_on: Optional[str] = None
    used: Optional[bool] = None
    reason: Optional[str] = None
    runtime_mass_kg: Optional[float] = None


@dataclass(frozen=True)
class AssetManifest:
    path: Path
    cache_root: str
    assets: tuple[AssetRecord, ...]
    sources: Mapping[str, Mapping[str, Any]]

    def by_id(self) -> dict[str, AssetRecord]:
        return {asset.id: asset for asset in self.assets}


@dataclass(frozen=True)
class FetchResult:
    asset_id: str
    path: Path
    status: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class UsdAudit:
    path: str
    physical_role: str
    opened: bool
    default_prim: Optional[str] = None
    up_axis: Optional[str] = None
    meters_per_unit: Optional[float] = None
    prim_count: int = 0
    articulation_roots: tuple[str, ...] = ()
    joint_counts: Mapping[str, int] = field(default_factory=dict)
    joints_with_limits: int = 0
    joints_without_limits: tuple[str, ...] = ()
    collision_prim_count: int = 0
    rigid_body_count: int = 0
    mass_or_density_prim_count: int = 0
    runtime_mass_kg: Optional[float] = None
    physx_metadata_prim_count: int = 0
    unresolved_dependencies: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.opened and not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


@dataclass(frozen=True)
class AutoMateDataAudit:
    assembly_id: str
    grasp_vector_length: int
    disassembly_distance_m: Optional[float]
    episode_count: int
    trajectory_step_counts: tuple[int, ...]
    required_episode_fields: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


@dataclass(frozen=True)
class GeneratedAssetAudit:
    robot_path: str
    scene_path: str
    source_hashes: Mapping[str, Mapping[str, Any]]
    source_references: tuple[str, ...]
    gripper_collider_paths: tuple[str, ...]
    convex_hull_collider_count: int
    tool_mounts: Mapping[str, Mapping[str, Any]]
    free_rigid_objects: tuple[str, ...]
    socket_hole_axis: Optional[str]
    socket_hole_width_m: Optional[float]
    socket_hole_height_m: Optional[float]
    socket_collision_boxes: int
    socket_hole_clear: bool
    robot_usd: Mapping[str, Any]
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.errors and bool(self.robot_usd.get("passed"))

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["passed"] = self.passed
        return result


def _require_string(data: Mapping[str, Any], key: str, asset_id: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise AssetError(f"asset {asset_id!r}: {key} must be a non-empty string")
    return value


def _validate_relative_path(value: str, asset_id: str) -> str:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise AssetError(f"asset {asset_id!r}: path must stay inside the cache root")
    return value


def _validate_download_url(value: str, asset_id: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_DOWNLOAD_HOSTS:
        raise AssetError(
            f"asset {asset_id!r}: URL must use HTTPS on an allowlisted host"
        )
    if parsed.username or parsed.password or parsed.port:
        raise AssetError(f"asset {asset_id!r}: URL credentials and ports are forbidden")
    return value


def load_manifest(path: Path | str) -> AssetManifest:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AssetError(f"cannot read asset manifest {manifest_path}: {exc}") from exc

    if payload.get("schema_version") != 1:
        raise AssetError("asset manifest schema_version must be 1")
    cache_root = payload.get("cache_root")
    if not isinstance(cache_root, str) or not cache_root:
        raise AssetError("asset manifest cache_root must be a non-empty string")
    _validate_relative_path(cache_root, "manifest")

    sources = payload.get("sources")
    if not isinstance(sources, dict):
        raise AssetError("asset manifest sources must be an object")

    records: list[AssetRecord] = []
    seen: set[str] = set()
    raw_assets = payload.get("assets")
    if not isinstance(raw_assets, list):
        raise AssetError("asset manifest assets must be an array")
    for raw in raw_assets:
        if not isinstance(raw, dict):
            raise AssetError("each asset entry must be an object")
        asset_id = _require_string(raw, "id", "<unknown>")
        if asset_id in seen:
            raise AssetError(f"duplicate asset id: {asset_id}")
        seen.add(asset_id)
        kind = _require_string(raw, "kind", asset_id)
        role = _require_string(raw, "physical_role", asset_id)
        source = _require_string(raw, "source", asset_id)
        if source not in sources:
            raise AssetError(f"asset {asset_id!r}: unknown source {source!r}")
        if role not in PHYSICAL_ROLES:
            raise AssetError(f"asset {asset_id!r}: unsupported physical_role {role!r}")
        if kind not in {"download", "external_runtime", "candidate"}:
            raise AssetError(f"asset {asset_id!r}: unsupported kind {kind!r}")

        values = dict(raw)
        if kind == "download":
            values["url"] = _validate_download_url(
                _require_string(raw, "url", asset_id), asset_id
            )
            values["path"] = _validate_relative_path(
                _require_string(raw, "path", asset_id), asset_id
            )
            size = raw.get("size_bytes")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise AssetError(f"asset {asset_id!r}: size_bytes must be positive")
            sha = _require_string(raw, "sha256", asset_id).lower()
            if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
                raise AssetError(f"asset {asset_id!r}: sha256 must be 64 lowercase hex chars")
            values["sha256"] = sha
            runtime_mass = raw.get("runtime_mass_kg")
            if runtime_mass is not None:
                if (isinstance(runtime_mass, bool) or
                        not isinstance(runtime_mass, (int, float)) or runtime_mass <= 0):
                    raise AssetError(
                        f"asset {asset_id!r}: runtime_mass_kg must be positive"
                    )
        elif kind == "external_runtime":
            root_path = _require_string(raw, "asset_root_path", asset_id)
            if not root_path.startswith("/Isaac/") or ".." in PurePosixPath(root_path).parts:
                raise AssetError(
                    f"asset {asset_id!r}: asset_root_path must be under /Isaac"
                )
            verified_sha = raw.get("verified_sha256")
            if verified_sha is not None and (
                not isinstance(verified_sha, str) or len(verified_sha) != 64 or
                any(char not in "0123456789abcdef" for char in verified_sha)
            ):
                raise AssetError(
                    f"asset {asset_id!r}: verified_sha256 must be 64 lowercase hex chars"
                )
        else:
            if raw.get("used") is not False:
                raise AssetError(f"candidate asset {asset_id!r} must set used=false")
            _require_string(raw, "reason", asset_id)

        allowed = {field.name for field in AssetRecord.__dataclass_fields__.values()}
        records.append(AssetRecord(**{key: value for key, value in values.items() if key in allowed}))

    return AssetManifest(manifest_path, cache_root, tuple(records), sources)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cache_destination(cache_root: Path, relative_path: str) -> Path:
    destination = (cache_root / relative_path).resolve()
    root = cache_root.resolve()
    if destination != root and root not in destination.parents:
        raise AssetError("resolved asset path escaped the cache root")
    return destination


def fetch_asset(
    asset: AssetRecord,
    cache_root: Path | str,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> FetchResult:
    if asset.kind != "download" or not asset.url or not asset.path:
        raise AssetError(f"asset {asset.id!r} is not downloadable")
    if not asset.sha256 or asset.size_bytes is None:
        raise AssetError(f"asset {asset.id!r} is missing integrity metadata")
    if asset.size_bytes > max_bytes:
        raise AssetError(
            f"asset {asset.id!r} exceeds size cap: {asset.size_bytes} > {max_bytes}"
        )

    destination = _cache_destination(Path(cache_root), asset.path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        actual_sha = sha256_file(destination)
        actual_size = destination.stat().st_size
        if actual_sha == asset.sha256 and actual_size == asset.size_bytes:
            return FetchResult(asset.id, destination, "cached", actual_size, actual_sha)

    request = Request(asset.url, headers={"User-Agent": "rby1-assembly-lab/asset-fetcher"})
    temp_path: Optional[Path] = None
    try:
        with urlopen(request, timeout=timeout_s) as response:
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > max_bytes:
                raise AssetError(f"asset {asset.id!r}: server content length exceeds cap")
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{destination.name}.", suffix=".tmp",
                dir=destination.parent, delete=False
            ) as output:
                temp_path = Path(output.name)
                digest = hashlib.sha256()
                total = 0
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise AssetError(f"asset {asset.id!r}: download exceeded size cap")
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())

        actual_sha = digest.hexdigest()
        if total != asset.size_bytes:
            raise AssetError(
                f"asset {asset.id!r}: size mismatch, expected {asset.size_bytes}, got {total}"
            )
        if actual_sha != asset.sha256:
            raise AssetError(
                f"asset {asset.id!r}: SHA-256 mismatch, expected {asset.sha256}, got {actual_sha}"
            )
        os.replace(temp_path, destination)
        temp_path = None
        return FetchResult(asset.id, destination, "downloaded", total, actual_sha)
    except AssetError:
        raise
    except Exception as exc:
        raise AssetError(f"asset {asset.id!r}: download failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def fetch_manifest_assets(
    manifest: AssetManifest,
    cache_root: Path | str,
    asset_ids: Optional[Sequence[str]] = None,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> list[FetchResult]:
    selected = set(asset_ids or ())
    known = manifest.by_id()
    unknown = selected.difference(known)
    if unknown:
        raise AssetError(f"unknown asset ids: {', '.join(sorted(unknown))}")
    assets = [
        asset for asset in manifest.assets
        if asset.kind == "download" and (not selected or asset.id in selected)
    ]
    return [
        fetch_asset(asset, cache_root, timeout_s=timeout_s, max_bytes=max_bytes)
        for asset in assets
    ]


def _has_applied_schema(prim: Any, schema_prefix: str) -> bool:
    return any(str(schema).startswith(schema_prefix) for schema in prim.GetAppliedSchemas())


def audit_usd(
    path: Path | str,
    physical_role: str,
    *,
    runtime_mass_kg: Optional[float] = None,
) -> UsdAudit:
    if physical_role not in PHYSICAL_ROLES:
        raise AssetError(f"unsupported physical_role {physical_role!r}")
    usd_path = Path(path).resolve()
    try:
        from pxr import Ar, Sdf, Usd, UsdGeom, UsdPhysics
    except ImportError as exc:
        return UsdAudit(
            str(usd_path), physical_role, False,
            errors=("pxr USD bindings are unavailable; install usd-core",),
        )

    stage = Usd.Stage.Open(str(usd_path), load=Usd.Stage.LoadAll)
    if not stage:
        return UsdAudit(
            str(usd_path), physical_role, False,
            errors=("USD stage could not be opened",),
        )

    joint_counts: dict[str, int] = {}
    articulation_roots: list[str] = []
    joints_without_limits: list[str] = []
    joints_with_limits = 0
    collision_count = 0
    rigid_body_count = 0
    mass_count = 0
    physx_count = 0
    prim_count = 0

    for prim in stage.Traverse():
        prim_count += 1
        path_text = str(prim.GetPath())
        schemas = tuple(str(schema) for schema in prim.GetAppliedSchemas())
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI) or _has_applied_schema(
            prim, "PhysicsArticulationRootAPI"
        ):
            articulation_roots.append(path_text)
        if prim.IsA(UsdPhysics.Joint):
            joint_type = prim.GetTypeName() or "PhysicsJoint"
            joint_counts[joint_type] = joint_counts.get(joint_type, 0) + 1
            if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                lower = prim.GetAttribute("physics:lowerLimit")
                upper = prim.GetAttribute("physics:upperLimit")
                if lower and upper and lower.HasAuthoredValueOpinion() and upper.HasAuthoredValueOpinion():
                    joints_with_limits += 1
                else:
                    joints_without_limits.append(path_text)
        if prim.HasAPI(UsdPhysics.CollisionAPI) or _has_applied_schema(
            prim, "PhysicsCollisionAPI"
        ):
            collision_count += 1
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or _has_applied_schema(
            prim, "PhysicsRigidBodyAPI"
        ):
            rigid_body_count += 1
        has_mass_api = prim.HasAPI(UsdPhysics.MassAPI) or _has_applied_schema(
            prim, "PhysicsMassAPI"
        )
        if has_mass_api:
            mass_attr = prim.GetAttribute("physics:mass")
            density_attr = prim.GetAttribute("physics:density")
            if ((mass_attr and mass_attr.HasAuthoredValueOpinion()) or
                    (density_attr and density_attr.HasAuthoredValueOpinion())):
                mass_count += 1
        has_physx_schema = any(schema.startswith("Physx") for schema in schemas)
        has_physx_attribute = any(
            attribute.GetName().startswith("physx") for attribute in prim.GetAttributes()
        )
        if has_physx_schema or has_physx_attribute:
            physx_count += 1

    unresolved: set[str] = set()
    try:
        resolver = Ar.GetResolver()
        for layer in stage.GetUsedLayers():
            for reference in layer.GetExternalReferences():
                if not reference:
                    continue
                computed = Sdf.ComputeAssetPathRelativeToLayer(layer, reference)
                if not resolver.Resolve(computed):
                    unresolved.add(str(reference))
        for prim in stage.Traverse():
            for attribute in prim.GetAttributes():
                value = attribute.Get()
                asset_values = []
                if isinstance(value, Sdf.AssetPath):
                    asset_values = [value]
                elif isinstance(value, (list, tuple)):
                    asset_values = [item for item in value if isinstance(item, Sdf.AssetPath)]
                for asset_value in asset_values:
                    if (asset_value.path and not asset_value.resolvedPath and
                            not resolver.Resolve(asset_value.path)):
                        unresolved.add(asset_value.path)
        unresolved_dependencies = tuple(sorted(unresolved))
        dependency_warning = ""
    except Exception as exc:
        unresolved_dependencies = ()
        dependency_warning = f"dependency scan failed: {exc}"

    errors: list[str] = []
    warnings: list[str] = []
    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        errors.append("stage has no default prim")
    if not stage.HasAuthoredMetadata("upAxis"):
        warnings.append("upAxis is not authored; USD fallback applies")
    if not stage.HasAuthoredMetadata("metersPerUnit"):
        warnings.append("metersPerUnit is not authored; USD fallback applies")

    if physical_role == "robot_articulation":
        if not articulation_roots:
            errors.append("robot asset has no articulation root")
        if sum(joint_counts.values()) == 0:
            errors.append("robot asset has no joints")
        if collision_count == 0:
            errors.append("robot asset has no collision prims")
    elif physical_role == "end_effector":
        if sum(joint_counts.values()) == 0:
            errors.append("end-effector asset has no joints")
        if collision_count == 0:
            errors.append("end-effector asset has no collision prims")
    elif physical_role.startswith("physical_interaction"):
        if collision_count == 0:
            errors.append("interaction asset has no collision prims")
        if (physical_role.endswith("dynamic") and mass_count == 0 and
                runtime_mass_kg is None):
            errors.append("dynamic interaction asset has no authored mass or density")
        elif physical_role.endswith("dynamic") and mass_count == 0:
            warnings.append(
                f"dynamic mass is supplied by runtime configuration: {runtime_mass_kg} kg"
            )
    elif physical_role == "visual_background" and collision_count == 0:
        warnings.append("background has no collision prims; do not use it as a support surface")

    if joints_without_limits:
        warnings.append(
            f"{len(joints_without_limits)} revolute/prismatic joints lack authored limits"
        )
    if unresolved_dependencies:
        warnings.append(
            "unresolved dependencies require the Isaac Sim resolver/runtime: "
            + ", ".join(unresolved_dependencies[:5])
        )
    if dependency_warning:
        warnings.append(dependency_warning)

    return UsdAudit(
        path=str(usd_path),
        physical_role=physical_role,
        opened=True,
        default_prim=str(default_prim.GetPath()) if default_prim else None,
        up_axis=UsdGeom.GetStageUpAxis(stage),
        meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        prim_count=prim_count,
        articulation_roots=tuple(articulation_roots),
        joint_counts=dict(sorted(joint_counts.items())),
        joints_with_limits=joints_with_limits,
        joints_without_limits=tuple(joints_without_limits),
        collision_prim_count=collision_count,
        rigid_body_count=rigid_body_count,
        mass_or_density_prim_count=mass_count,
        runtime_mass_kg=runtime_mass_kg,
        physx_metadata_prim_count=physx_count,
        unresolved_dependencies=unresolved_dependencies,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


def audit_manifest_assets(
    manifest: AssetManifest,
    cache_root: Path | str,
    asset_ids: Optional[Iterable[str]] = None,
) -> list[UsdAudit]:
    selected = set(asset_ids or ())
    known = manifest.by_id()
    unknown = selected.difference(known)
    if unknown:
        raise AssetError(f"unknown asset ids: {', '.join(sorted(unknown))}")
    reports: list[UsdAudit] = []
    for asset in manifest.assets:
        if selected and asset.id not in selected:
            continue
        if asset.kind != "download" or not asset.usd or not asset.path:
            continue
        reports.append(
            audit_usd(
                _cache_destination(Path(cache_root), asset.path),
                asset.physical_role,
                runtime_mass_kg=asset.runtime_mass_kg,
            )
        )
    return reports


def audit_automate_data(
    cache_root: Path | str, assembly_id: str = "00015"
) -> AutoMateDataAudit:
    root = Path(cache_root) / "automate"
    required_fields = (
        "fingertip_centered_pos",
        "fingertip_centered_quat",
        "arm_dof_pos",
        "plug_grasp_pos",
        "plug_grasp_quat",
        "init_plug_pos",
        "init_plug_quat",
        "plug_pos",
        "plug_quat",
    )
    errors: list[str] = []

    def read_json(path: Path) -> Any:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {path}: {exc}")
            return None

    grasps = read_json(root / "plug_grasps.json")
    distances = read_json(root / "disassembly_dist.json")
    trajectories = read_json(root / assembly_id / "disassemble_traj.json")
    asset_key = f"asset_{assembly_id}"

    grasp = grasps.get(asset_key) if isinstance(grasps, dict) else None
    if not isinstance(grasp, list) or len(grasp) != 7:
        errors.append(f"{asset_key} grasp must be a 7-value pose")
        grasp_length = len(grasp) if isinstance(grasp, list) else 0
    else:
        grasp_length = len(grasp)

    distance = distances.get(asset_key) if isinstance(distances, dict) else None
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or distance <= 0:
        errors.append(f"{asset_key} disassembly distance must be positive")
        distance_value = None
    else:
        distance_value = float(distance)

    step_counts: set[int] = set()
    if not isinstance(trajectories, list) or not trajectories:
        errors.append("disassembly trajectories must be a non-empty array")
        episode_count = 0
    else:
        episode_count = len(trajectories)
        for index, episode in enumerate(trajectories):
            if not isinstance(episode, dict):
                errors.append(f"episode {index} must be an object")
                continue
            missing = [field for field in required_fields if field not in episode]
            if missing:
                errors.append(f"episode {index} missing fields: {', '.join(missing)}")
                continue
            plug_positions = episode["plug_pos"]
            plug_rotations = episode["plug_quat"]
            if not isinstance(plug_positions, list) or not isinstance(plug_rotations, list):
                errors.append(f"episode {index} trajectory fields must be arrays")
                continue
            if len(plug_positions) != len(plug_rotations) or not plug_positions:
                errors.append(f"episode {index} has inconsistent trajectory lengths")
                continue
            step_counts.add(len(plug_positions))

    return AutoMateDataAudit(
        assembly_id=assembly_id,
        grasp_vector_length=grasp_length,
        disassembly_distance_m=distance_value,
        episode_count=episode_count,
        trajectory_step_counts=tuple(sorted(step_counts)),
        required_episode_fields=required_fields,
        errors=tuple(errors),
    )


def audit_generated_assets(
    project_root: Path | str,
    manifest: AssetManifest,
) -> GeneratedAssetAudit:
    project = Path(project_root).resolve()
    generated = project / "assets" / "generated"
    cache = project / "assets" / "cache"
    robot_path = generated / "robot.usda"
    scene_path = generated / "scene.usda"
    socket_path = generated / "socket.usda"
    errors: list[str] = []
    warnings: list[str] = []

    source_hashes: dict[str, dict[str, Any]] = {}
    for asset_id in (
        "rby1_model_m_v1_2",
        "rb_gripper_left",
        "rb_gripper_right",
        "rb_gripper_mount_config",
    ):
        asset = manifest.by_id()[asset_id]
        assert asset.path is not None and asset.sha256 is not None
        source_path = _cache_destination(cache, asset.path)
        actual = sha256_file(source_path) if source_path.is_file() else None
        matches = actual == asset.sha256
        source_hashes[asset_id] = {
            "path": str(source_path),
            "expected_sha256": asset.sha256,
            "actual_sha256": actual,
            "matches": matches,
        }
        if not matches:
            errors.append(f"generated source hash mismatch: {asset_id}")

    robot_report = audit_usd(robot_path, "robot_articulation")
    try:
        from pxr import Usd, UsdGeom, UsdPhysics
    except ImportError:
        return GeneratedAssetAudit(
            str(robot_path), str(scene_path), source_hashes, (), (), 0, {}, (),
            None, None, None, 0, False, robot_report.to_dict(),
            errors=tuple(errors + ["pxr USD bindings are unavailable"]),
        )

    robot_stage = Usd.Stage.Open(str(robot_path), load=Usd.Stage.LoadAll)
    source_references: tuple[str, ...] = ()
    collider_paths: list[str] = []
    convex_count = 0
    tool_mounts: dict[str, dict[str, Any]] = {}
    if not robot_stage:
        errors.append("generated robot stage could not be opened")
    else:
        source_references = tuple(
            sorted(
                reference for reference in robot_stage.GetRootLayer().GetExternalReferences()
                if reference
            )
        )
        for prim in robot_stage.Traverse():
            path_text = str(prim.GetPath())
            if (prim.IsA(UsdGeom.Mesh) and
                    ("/left_gripper/" in path_text or "/right_gripper/" in path_text) and
                    prim.HasAPI(UsdPhysics.CollisionAPI)):
                collider_paths.append(path_text)
                approximation = prim.GetAttribute("physics:approximation")
                if approximation and approximation.Get() == "convexHull":
                    convex_count += 1
        if len(collider_paths) != 6:
            errors.append(
                f"generated robot must have 6 gripper mesh colliders, got {len(collider_paths)}"
            )
        if convex_count != 6:
            errors.append(
                f"all gripper colliders must use convexHull, got {convex_count}/6"
            )
        if len(source_references) != 3:
            errors.append(
                f"generated robot must retain 3 source USD references, got {len(source_references)}"
            )
        for side in ("left", "right"):
            joint_path = f"/Robot/joints/tool_{side}"
            joint_prim = robot_stage.GetPrimAtPath(joint_path)
            joint = UsdPhysics.FixedJoint(joint_prim)
            body0 = tuple(str(path) for path in joint.GetBody0Rel().GetTargets()) if joint else ()
            body1 = tuple(str(path) for path in joint.GetBody1Rel().GetTargets()) if joint else ()
            pos0 = tuple(float(value) for value in joint.GetLocalPos0Attr().Get()) if joint else ()
            pos1 = tuple(float(value) for value in joint.GetLocalPos1Attr().Get()) if joint else ()
            rot0_value = joint.GetLocalRot0Attr().Get() if joint else None
            rot1_value = joint.GetLocalRot1Attr().Get() if joint else None
            rot0 = ((float(rot0_value.GetReal()),) +
                    tuple(float(value) for value in rot0_value.GetImaginary())) if rot0_value else ()
            rot1 = ((float(rot1_value.GetReal()),) +
                    tuple(float(value) for value in rot1_value.GetImaginary())) if rot1_value else ()
            expected_body0 = (f"/Robot/link_{side}_arm_6",)
            expected_body1 = (f"/Robot/{side}_gripper/ee_{side}",)
            expected_pos0 = (0.0, 0.0, -0.1261)
            expected_pos1 = (0.0, 0.0, 0.0)
            expected_rotation = (1.0, 0.0, 0.0, 0.0)

            def close_tuple(actual: tuple[float, ...], expected: tuple[float, ...]) -> bool:
                return len(actual) == len(expected) and all(
                    abs(left - right) <= 1e-6 for left, right in zip(actual, expected)
                )

            consistent = (
                body0 == expected_body0 and body1 == expected_body1 and
                close_tuple(pos0, expected_pos0) and close_tuple(pos1, expected_pos1) and
                close_tuple(rot0, expected_rotation) and
                close_tuple(rot1, expected_rotation)
            )
            tool_mounts[side] = {
                "joint_path": joint_path,
                "body0": body0,
                "body1": body1,
                "local_pos0_m": pos0,
                "local_pos1_m": pos1,
                "local_rot0_wxyz": rot0,
                "local_rot1_wxyz": rot1,
                "consistent": consistent,
            }
            if not consistent:
                errors.append(f"{side} tool mount frame is inconsistent")

    scene_stage = Usd.Stage.Open(str(scene_path), load=Usd.Stage.LoadAll)
    free_objects: list[str] = []
    if not scene_stage:
        errors.append("generated scene stage could not be opened")
    else:
        joint_targets: set[str] = set()
        for prim in scene_stage.Traverse():
            if prim.IsA(UsdPhysics.Joint):
                for relation_name in ("physics:body0", "physics:body1"):
                    relation = prim.GetRelationship(relation_name)
                    if relation:
                        joint_targets.update(str(path) for path in relation.GetTargets())
        for path_text in ("/World/Plug", "/World/Socket"):
            prim = scene_stage.GetPrimAtPath(path_text)
            if (prim and prim.HasAPI(UsdPhysics.RigidBodyAPI) and
                    not any(target == path_text or target.startswith(path_text + "/")
                            for target in joint_targets)):
                free_objects.append(path_text)
        if len(free_objects) != 2:
            errors.append(
                f"scene must have 2 free rigid assembly objects, got {len(free_objects)}"
            )

    socket_stage = Usd.Stage.Open(str(socket_path), load=Usd.Stage.LoadAll)
    hole_axis: Optional[str] = None
    hole_width: Optional[float] = None
    hole_height: Optional[float] = None
    socket_boxes = 0
    hole_clear = False
    if not socket_stage:
        errors.append("generated socket stage could not be opened")
    else:
        root = socket_stage.GetDefaultPrim()
        hole_axis = root.GetCustomDataByKey("assembly:hole_axis")
        width_value = root.GetCustomDataByKey("assembly:hole_width_m")
        height_value = root.GetCustomDataByKey("assembly:hole_height_m")
        hole_width = float(width_value) if isinstance(width_value, (int, float)) else None
        hole_height = float(height_value) if isinstance(height_value, (int, float)) else None
        collider_ranges = []
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
        for prim in socket_stage.Traverse():
            if prim.IsA(UsdGeom.Cube) and prim.HasAPI(UsdPhysics.CollisionAPI):
                socket_boxes += 1
                collider_ranges.append(bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange())
        if hole_axis != "+Y" or hole_width is None or hole_height is None:
            errors.append("socket hole metadata is incomplete")
        elif socket_boxes != 4:
            errors.append(f"socket hole requires 4 wall colliders, got {socket_boxes}")
        else:
            half_width = hole_width / 2.0
            half_height = hole_height / 2.0
            tolerance = 1e-9
            hole_clear = all(
                (box.GetMax()[0] <= -half_width + tolerance or
                 box.GetMin()[0] >= half_width - tolerance or
                 box.GetMax()[2] <= -half_height + tolerance or
                 box.GetMin()[2] >= half_height - tolerance)
                for box in collider_ranges
            )
            if not hole_clear:
                errors.append("socket collision geometry blocks the declared central hole")

    return GeneratedAssetAudit(
        robot_path=str(robot_path),
        scene_path=str(scene_path),
        source_hashes=source_hashes,
        source_references=source_references,
        gripper_collider_paths=tuple(sorted(collider_paths)),
        convex_hull_collider_count=convex_count,
        tool_mounts=tool_mounts,
        free_rigid_objects=tuple(free_objects),
        socket_hole_axis=hole_axis,
        socket_hole_width_m=hole_width,
        socket_hole_height_m=hole_height,
        socket_collision_boxes=socket_boxes,
        socket_hole_clear=hole_clear,
        robot_usd=robot_report.to_dict(),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
