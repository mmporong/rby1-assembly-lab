#!/usr/bin/env python3
"""Stop one Brev instance only when its ID and name match exactly."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from typing import Any


COMMAND_TIMEOUT_S = 20
MAX_STATUS_POLLS = 8
POLL_INTERVAL_S = 5


class BrevCommandError(RuntimeError):
    """A bounded Brev CLI command failed without exposing its output."""


def _run_brev(org: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        result = subprocess.run(
            ["brev", "--org", org, *arguments],
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise BrevCommandError(type(error).__name__) from error
    if result.returncode != 0:
        raise BrevCommandError(f"exit_{result.returncode}")
    return result


def _list_workspaces(org: str) -> list[dict[str, Any]]:
    result = _run_brev(org, ["ls", "--json"])
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as error:
        raise BrevCommandError("invalid_json") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("workspaces"), list):
        raise BrevCommandError("invalid_workspaces")
    if not all(isinstance(workspace, dict) for workspace in payload["workspaces"]):
        raise BrevCommandError("invalid_workspace_entry")
    return payload["workspaces"]


def _resolve_exact_workspace(
    workspaces: list[dict[str, Any]], instance_id: str, instance_name: str
) -> tuple[dict[str, Any] | None, str | None]:
    id_matches = [item for item in workspaces if item.get("id") == instance_id]
    name_matches = [item for item in workspaces if item.get("name") == instance_name]

    if not id_matches and not name_matches:
        return None, "absent"
    if len(id_matches) != 1 or len(name_matches) != 1 or id_matches[0] is not name_matches[0]:
        return None, "identity_mismatch"
    return id_matches[0], None


def _status(workspace: dict[str, Any]) -> str:
    value = workspace.get("status")
    return value.upper() if isinstance(value, str) else "UNKNOWN"


def _verify_stopped(
    org: str,
    instance_id: str,
    instance_name: str,
    *,
    prior_error: str | None = None,
) -> tuple[dict[str, Any], int]:
    for poll_number in range(1, MAX_STATUS_POLLS + 1):
        if poll_number > 1:
            time.sleep(POLL_INTERVAL_S)
        try:
            workspace, resolution_error = _resolve_exact_workspace(
                _list_workspaces(org), instance_id, instance_name
            )
        except BrevCommandError as error:
            report: dict[str, Any] = {
                "ok": False,
                "status": "verification_failed",
                "error": prior_error if prior_error is not None else str(error),
            }
            return report, 4
        if resolution_error is not None:
            report = {"ok": False, "status": f"verification_{resolution_error}"}
            if prior_error is not None:
                report["error"] = prior_error
            return report, 4
        assert workspace is not None
        if _status(workspace) == "STOPPED":
            report = {
                "ok": True,
                "status": "stopped",
                "instance_id": instance_id,
                "instance_name": instance_name,
                "polls": poll_number,
            }
            return report, 0

    report = {
        "ok": False,
        "status": "stop_not_confirmed",
        "polls": MAX_STATUS_POLLS,
    }
    if prior_error is not None:
        report["error"] = prior_error
    return report, 4


def stop_exact_instance(
    org: str, instance_id: str, instance_name: str
) -> tuple[dict[str, Any], int]:
    try:
        workspace, resolution_error = _resolve_exact_workspace(
            _list_workspaces(org), instance_id, instance_name
        )
    except BrevCommandError as error:
        return {"ok": False, "status": "list_failed", "error": str(error)}, 3

    if resolution_error is not None:
        return {"ok": False, "status": resolution_error}, 2
    assert workspace is not None

    if _status(workspace) == "STOPPED":
        return {
            "ok": True,
            "status": "already_stopped",
            "instance_id": instance_id,
            "instance_name": instance_name,
            "polls": 0,
        }, 0

    initial_status = _status(workspace)
    stop_error: str | None = None
    if initial_status != "STOPPING":
        try:
            _run_brev(org, ["stop", "--", instance_name])
        except BrevCommandError as error:
            stop_error = str(error)

    return _verify_stopped(
        org,
        instance_id,
        instance_name,
        prior_error=stop_error,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--org", required=True)
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--instance-name", required=True)
    args = parser.parse_args()
    report, exit_code = stop_exact_instance(args.org, args.instance_id, args.instance_name)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
