from __future__ import annotations

import importlib.util
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "stop_brev_instance.py"
SPEC = importlib.util.spec_from_file_location("stop_brev_instance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
stop_brev_instance = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stop_brev_instance)


INSTANCE_ID = "instance-123"
INSTANCE_NAME = "rby1-assembly-20261025"
ORG = "mmporong-test-org"


def completed(stdout: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="hidden detail")


def listing(*workspaces: dict[str, str]) -> str:
    import json

    return json.dumps({"workspaces": list(workspaces)})


class StopBrevInstanceTests(unittest.TestCase):
    @patch.object(stop_brev_instance.subprocess, "run")
    def test_absent_instance_fails_without_stop(self, run) -> None:
        run.return_value = completed(listing())

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(2, exit_code)
        self.assertEqual("absent", report["status"])
        run.assert_called_once()
        self.assertEqual(["brev", "--org", ORG, "ls", "--json"], run.call_args.args[0])

    @patch.object(stop_brev_instance.subprocess, "run")
    def test_identity_mismatch_fails_without_stop(self, run) -> None:
        run.return_value = completed(
            listing(
                {"id": INSTANCE_ID, "name": "different-name", "status": "RUNNING"},
                {"id": "different-id", "name": INSTANCE_NAME, "status": "RUNNING"},
            )
        )

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(2, exit_code)
        self.assertEqual("identity_mismatch", report["status"])
        run.assert_called_once()

    @patch.object(stop_brev_instance.subprocess, "run")
    def test_already_stopped_is_success_without_stop(self, run) -> None:
        run.return_value = completed(
            listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPED"})
        )

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(0, exit_code)
        self.assertEqual("already_stopped", report["status"])
        self.assertEqual(0, report["polls"])
        run.assert_called_once()

    @patch.object(stop_brev_instance.time, "sleep")
    @patch.object(stop_brev_instance.subprocess, "run")
    def test_stop_success_is_rechecked(self, run, sleep) -> None:
        run.side_effect = [
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})),
            completed(),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPED"})),
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(0, exit_code)
        self.assertEqual("stopped", report["status"])
        self.assertEqual(1, report["polls"])
        self.assertEqual(
            ["brev", "--org", ORG, "stop", "--", INSTANCE_NAME],
            run.call_args_list[1].args[0],
        )
        sleep.assert_not_called()

    @patch.object(stop_brev_instance.subprocess, "run")
    def test_stop_command_failure_is_verified_and_does_not_expose_stderr(self, run) -> None:
        run.side_effect = [
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})),
            completed(returncode=9),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPED"})),
        ]

        with patch.object(stop_brev_instance.time, "sleep"):
            report, exit_code = stop_brev_instance.stop_exact_instance(
                ORG, INSTANCE_ID, INSTANCE_NAME
            )

        self.assertEqual(0, exit_code)
        self.assertEqual("stopped", report["status"])
        self.assertNotIn("error", report)
        self.assertNotIn("hidden detail", str(report))

    @patch.object(stop_brev_instance.subprocess, "run")
    def test_stop_timeout_then_stopped_is_success(self, run) -> None:
        run.side_effect = [
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})),
            subprocess.TimeoutExpired(["brev", "stop"], 20),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPED"})),
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(0, exit_code)
        self.assertEqual("stopped", report["status"])
        self.assertNotIn("error", report)

    @patch.object(stop_brev_instance.time, "sleep")
    @patch.object(stop_brev_instance.subprocess, "run")
    def test_stopping_instance_is_only_polled(self, run, sleep) -> None:
        run.side_effect = [
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPING"})),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPING"})),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPED"})),
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(0, exit_code)
        self.assertEqual("stopped", report["status"])
        self.assertEqual(3, run.call_count)
        self.assertNotIn("stop", [call.args[0][3] for call in run.call_args_list])
        sleep.assert_called_once_with(stop_brev_instance.POLL_INTERVAL_S)

    @patch.object(stop_brev_instance.time, "sleep")
    @patch.object(stop_brev_instance.subprocess, "run")
    def test_polling_is_bounded_when_stop_is_not_confirmed(self, run, sleep) -> None:
        running = completed(
            listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})
        )
        run.side_effect = [running, completed()] + [
            running for _ in range(stop_brev_instance.MAX_STATUS_POLLS)
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(4, exit_code)
        self.assertEqual("stop_not_confirmed", report["status"])
        self.assertEqual(stop_brev_instance.MAX_STATUS_POLLS + 2, run.call_count)
        self.assertEqual(stop_brev_instance.MAX_STATUS_POLLS - 1, sleep.call_count)

    @patch.object(stop_brev_instance.time, "sleep")
    @patch.object(stop_brev_instance.subprocess, "run")
    def test_stop_failure_unconfirmed_reports_only_error_code(self, run, sleep) -> None:
        running = completed(
            listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})
        )
        run.side_effect = [running, completed(returncode=9)] + [
            running for _ in range(stop_brev_instance.MAX_STATUS_POLLS)
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(4, exit_code)
        self.assertEqual("stop_not_confirmed", report["status"])
        self.assertEqual("exit_9", report["error"])
        self.assertNotIn("hidden detail", str(report))
        self.assertEqual(stop_brev_instance.MAX_STATUS_POLLS - 1, sleep.call_count)

    @patch.object(stop_brev_instance.subprocess, "run")
    def test_every_brev_call_uses_fixed_org(self, run) -> None:
        run.side_effect = [
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "RUNNING"})),
            completed(),
            completed(listing({"id": INSTANCE_ID, "name": INSTANCE_NAME, "status": "STOPPED"})),
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(ORG, INSTANCE_ID, INSTANCE_NAME)

        self.assertEqual(0, exit_code)
        self.assertTrue(report["ok"])
        for call in run.call_args_list:
            self.assertEqual(["brev", "--org", ORG], call.args[0][:3])

    @patch.object(stop_brev_instance.subprocess, "run")
    def test_instance_name_is_separated_from_stop_flags(self, run) -> None:
        option_like_name = "--help"
        run.side_effect = [
            completed(listing({"id": INSTANCE_ID, "name": option_like_name, "status": "RUNNING"})),
            completed(),
            completed(listing({"id": INSTANCE_ID, "name": option_like_name, "status": "STOPPED"})),
        ]

        report, exit_code = stop_brev_instance.stop_exact_instance(
            ORG, INSTANCE_ID, option_like_name
        )

        self.assertEqual(0, exit_code)
        self.assertTrue(report["ok"])
        self.assertEqual(
            ["brev", "--org", ORG, "stop", "--", option_like_name],
            run.call_args_list[1].args[0],
        )


if __name__ == "__main__":
    unittest.main()
