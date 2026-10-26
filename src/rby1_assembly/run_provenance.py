"""Fail-closed input snapshots for long-running simulation experiments."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


class InputMutationError(RuntimeError):
    """Raised when an experiment input differs from its start snapshot."""

    def __init__(self, *, added: list[str], removed: list[str], changed: list[str]):
        self.added = added
        self.removed = removed
        self.changed = changed
        details = []
        if added:
            details.append(f"added={added}")
        if removed:
            details.append(f"removed={removed}")
        if changed:
            details.append(f"changed={changed}")
        super().__init__("experiment inputs changed after start: " + ", ".join(details))

    def report(self) -> dict[str, Any]:
        return {
            "verified_unchanged": False,
            "added_paths": self.added,
            "removed_paths": self.removed,
            "changed_paths": self.changed,
        }


@dataclass(frozen=True)
class _CaptureSpec:
    project_root: Path
    task_path: Path
    build_report_fallback: Path | None
    config_paths: tuple[Path, ...]
    source_roots: tuple[tuple[Path, bool], ...]
    source_files: tuple[Path, ...]
    extra_paths: tuple[Path, ...]


@dataclass(frozen=True)
class RunInputSnapshot:
    """Hashes and parsed JSON captured before an environment is constructed."""

    _spec: _CaptureSpec
    sha256: dict[str, str]
    source_sha256: dict[str, str]
    generated_asset_sha256: dict[str, str]
    _json_documents: dict[str, Any]

    def key_for(self, path: Path) -> str:
        return _path_key(self._spec.project_root, path)

    def hash_for(self, path: Path) -> str:
        key = self.key_for(path)
        try:
            return self.sha256[key]
        except KeyError as error:
            raise KeyError(f"path was not part of the start snapshot: {key}") from error

    def json_for(self, path: Path) -> Any:
        key = self.key_for(path)
        try:
            return deepcopy(self._json_documents[key])
        except KeyError as error:
            raise KeyError(f"JSON path was not part of the start snapshot: {key}") from error

    @property
    def build_report_path(self) -> Path:
        task = self.json_for(self._spec.task_path)
        configured = task.get("assets", {}).get("build_report")
        if configured is not None:
            return (self._spec.project_root / configured).resolve()
        if self._spec.build_report_fallback is None:
            raise ValueError("task does not declare assets.build_report")
        return self._spec.build_report_fallback

    def assert_unchanged(self) -> None:
        removed = []
        changed = []
        for key, expected in self.sha256.items():
            path = Path(key)
            if not path.is_absolute():
                path = self._spec.project_root / path
            try:
                current = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                removed.append(key)
                continue
            if current != expected:
                changed.append(key)
        current_source_keys = set()
        for source_root, recursive in self._spec.source_roots:
            iterator = source_root.rglob("*.py") if recursive else source_root.glob("*.py")
            current_source_keys.update(
                _path_key(self._spec.project_root, path)
                for path in iterator if path.is_file()
            )
        current_source_keys.update(
            _path_key(self._spec.project_root, path)
            for path in self._spec.source_files if path.is_file()
        )
        added = sorted(current_source_keys - set(self.source_sha256))
        if added or removed or changed:
            raise InputMutationError(
                added=added,
                removed=sorted(removed),
                changed=sorted(changed),
            )

        current = _capture(self._spec)
        start_keys = set(self.sha256)
        current_keys = set(current.sha256)
        added = sorted(current_keys - start_keys)
        removed = sorted(start_keys - current_keys)
        changed = sorted(
            key for key in start_keys & current_keys
            if self.sha256[key] != current.sha256[key]
        )
        if added or removed or changed:
            raise InputMutationError(added=added, removed=removed, changed=changed)

    def assert_preserved_in(self, later: "RunInputSnapshot") -> None:
        """Require every bootstrap input to appear unchanged in a later snapshot."""

        start_keys = set(self.sha256)
        later_keys = set(later.sha256)
        removed = sorted(start_keys - later_keys)
        changed = sorted(
            key for key in start_keys & later_keys
            if self.sha256[key] != later.sha256[key]
        )
        if removed or changed:
            raise InputMutationError(added=[], removed=removed, changed=changed)


def _path_key(project_root: Path, path: Path) -> str:
    resolved = Path(path).resolve()
    if resolved.is_relative_to(project_root):
        return str(resolved.relative_to(project_root))
    return str(resolved)


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValueError(f"unable to snapshot experiment input {path}: {error}") from error


def _decode_json(path: Path, payload: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(payload.decode("utf-8"), parse_constant=reject_constant)
        json.dumps(value, allow_nan=False)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise ValueError(f"invalid finite JSON experiment input {path}: {error}") from error
    return value


def _project_local(project_root: Path, path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved != project_root and not resolved.is_relative_to(project_root):
        raise ValueError(f"{label} must stay inside the project: {resolved}")
    return resolved


def _capture(spec: _CaptureSpec) -> RunInputSnapshot:
    project_root = spec.project_root
    task_payload = _read(spec.task_path)
    task = _decode_json(spec.task_path, task_payload)
    configured_build_report = task.get("assets", {}).get("build_report")
    if configured_build_report is None:
        if spec.build_report_fallback is None:
            raise ValueError("task does not declare assets.build_report")
        build_report_path = spec.build_report_fallback
    else:
        build_report_path = _project_local(
            project_root, project_root / configured_build_report, "build report"
        )
    build_payload = _read(build_report_path)
    build_report = _decode_json(build_report_path, build_payload)
    files = build_report.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError(f"build report has no generated file hashes: {build_report_path}")

    explicit_paths = {
        spec.task_path,
        build_report_path,
        *spec.config_paths,
        *spec.source_files,
        *spec.extra_paths,
    }
    source_paths = set(spec.source_files)
    for source_root, recursive in spec.source_roots:
        source_root = _project_local(project_root, source_root, "source root")
        iterator = source_root.rglob("*.py") if recursive else source_root.glob("*.py")
        discovered = {path.resolve() for path in iterator if path.is_file()}
        explicit_paths.update(discovered)
        source_paths.update(discovered)

    generated_paths: dict[str, Path] = {}
    for filename, expected_hash in files.items():
        if not isinstance(filename, str) or not isinstance(expected_hash, str):
            raise ValueError(f"invalid generated file hash entry in {build_report_path}")
        path = _project_local(project_root, build_report_path.parent / filename, "generated asset")
        generated_paths[filename] = path
        explicit_paths.add(path)

    hashes: dict[str, str] = {}
    documents: dict[str, Any] = {}
    payloads = {
        spec.task_path: task_payload,
        build_report_path: build_payload,
    }
    for path in sorted(explicit_paths, key=lambda item: _path_key(project_root, item)):
        payload = payloads.get(path)
        if payload is None:
            payload = _read(path)
        key = _path_key(project_root, path)
        hashes[key] = hashlib.sha256(payload).hexdigest()
        if path.suffix.lower() == ".json":
            documents[key] = _decode_json(path, payload)

    generated_hashes = {
        filename: hashes[_path_key(project_root, path)]
        for filename, path in sorted(generated_paths.items())
    }
    expected_hashes = {str(name): str(digest) for name, digest in files.items()}
    if generated_hashes != expected_hashes:
        mismatched = sorted(
            name for name in set(generated_hashes) | set(expected_hashes)
            if generated_hashes.get(name) != expected_hashes.get(name)
        )
        raise ValueError(f"generated asset hashes disagree with build report: {mismatched}")

    return RunInputSnapshot(
        _spec=spec,
        sha256=hashes,
        source_sha256={
            _path_key(project_root, path): hashes[_path_key(project_root, path)]
            for path in sorted(source_paths, key=lambda item: _path_key(project_root, item))
        },
        generated_asset_sha256=generated_hashes,
        _json_documents=documents,
    )


def capture_run_inputs(
    project_root: Path,
    *,
    task_path: Path,
    build_report_fallback: Path | None = None,
    config_paths: tuple[Path, ...] = (),
    source_roots: tuple[tuple[Path, bool], ...] = (),
    source_files: tuple[Path, ...] = (),
    extra_paths: tuple[Path, ...] = (),
) -> RunInputSnapshot:
    """Capture every declared input and verify the capture is internally stable."""

    project_root = Path(project_root).resolve()
    spec = _CaptureSpec(
        project_root=project_root,
        task_path=Path(task_path).resolve(),
        build_report_fallback=(
            None if build_report_fallback is None else Path(build_report_fallback).resolve()
        ),
        config_paths=tuple(Path(path).resolve() for path in config_paths),
        source_roots=tuple((Path(path).resolve(), recursive) for path, recursive in source_roots),
        source_files=tuple(Path(path).resolve() for path in source_files),
        extra_paths=tuple(Path(path).resolve() for path in extra_paths),
    )
    snapshot = _capture(spec)
    snapshot.assert_unchanged()
    return snapshot
