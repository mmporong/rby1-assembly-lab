#!/usr/bin/env python3
"""Verify the preregistered plan and input bytes before reproducing the run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify(root: Path, plan_path: Path) -> int:
    root = root.resolve()
    checksum_path = plan_path.with_name("plan.sha256")
    checksum_parts = checksum_path.read_text(encoding="utf-8").split()
    if (
        len(checksum_parts) != 2
        or not re.fullmatch(r"[0-9a-f]{64}", checksum_parts[0])
        or checksum_parts[1] != plan_path.name
        or digest(plan_path) != checksum_parts[0]
    ):
        raise ValueError("실험 계획의 SHA-256 또는 파일명이 일치하지 않습니다")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    inputs = plan["frozen_inputs"]["sha256"]
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("고정 입력 목록이 비어 있거나 잘못되었습니다")

    errors: list[str] = []
    for relative, expected in inputs.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError("입력 경로와 SHA-256은 문자열이어야 합니다")
        path = PurePosixPath(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or relative != path.as_posix()
            or not re.fullmatch(r"[0-9a-f]{64}", expected)
        ):
            raise ValueError(f"잘못된 입력 경로 또는 SHA-256: {relative}")
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"저장소 밖을 가리키는 입력: {relative}")
        if not target.is_file():
            errors.append(f"파일 없음: {relative}")
        elif digest(target) != expected:
            errors.append(f"해시 불일치: {relative}")

    for directory, pattern in (("src/rby1_assembly", "*.py"), ("configs", "*.json")):
        actual = {
            item.relative_to(root).as_posix()
            for item in (root / directory).rglob(pattern)
            if item.is_file()
        }
        expected_paths = {
            item for item in inputs
            if PurePosixPath(item).is_relative_to(PurePosixPath(directory))
            and PurePosixPath(item).match(pattern)
        }
        if actual != expected_paths:
            errors.append(
                f"{directory} 파일 목록 불일치: "
                f"추가={sorted(actual - expected_paths)}, 누락={sorted(expected_paths - actual)}"
            )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"고정 입력 {len(inputs)}개와 소스·설정 파일 목록이 일치합니다")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--plan", type=Path, default=Path(__file__).with_name("experiment-plan.json"))
    args = parser.parse_args()
    try:
        return verify(args.root, args.plan)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"고정 입력 검증 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
