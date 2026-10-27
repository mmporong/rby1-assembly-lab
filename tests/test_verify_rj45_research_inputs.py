from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CHECKER = (
    Path(__file__).resolve().parents[1]
    / "docs/evidence/rj45-paper-ablation/verify_frozen_inputs.py"
)


class FrozenInputVerifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root / "experiment-plan.json"
        self.files = {
            "src/rby1_assembly/example.py": b"source\n",
            "configs/example.json": b"{}\n",
            "assets/example.usda": b"asset\n",
        }
        for relative, content in self.files.items():
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.write_plan()

    def write_plan(self, inputs: dict[str, str] | None = None) -> None:
        if inputs is None:
            inputs = {name: hashlib.sha256(value).hexdigest() for name, value in self.files.items()}
        self.plan.write_text(json.dumps({"frozen_inputs": {"sha256": inputs}}), encoding="utf-8")
        checksum = hashlib.sha256(self.plan.read_bytes()).hexdigest()
        self.plan.with_name("plan.sha256").write_text(
            f"{checksum}  {self.plan.name}\n", encoding="utf-8"
        )

    def run_checker(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(CHECKER), "--root", str(self.root), "--plan", str(self.plan)],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_matches_plan_and_bytes(self) -> None:
        result = self.run_checker()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3개", result.stdout)

    def test_changed_or_missing_frozen_bytes_fail(self) -> None:
        (self.root / "assets/example.usda").write_bytes(b"changed")
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("해시 불일치", result.stderr)
        (self.root / "assets/example.usda").unlink()
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("파일 없음", result.stderr)

    def test_extra_source_and_config_are_rejected(self) -> None:
        (self.root / "src/rby1_assembly/unplanned.py").write_text("pass\n", encoding="utf-8")
        (self.root / "configs/unplanned.json").write_text("{}\n", encoding="utf-8")
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("src/rby1_assembly 파일 목록 불일치", result.stderr)
        self.assertIn("configs 파일 목록 불일치", result.stderr)

    def test_changed_plan_is_rejected(self) -> None:
        self.plan.write_text("{}", encoding="utf-8")
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("실험 계획의 SHA-256", result.stderr)

    def test_nested_extra_source_and_config_are_rejected(self) -> None:
        source = self.root / "src/rby1_assembly/submodule/unplanned.py"
        config = self.root / "configs/tasks/unplanned.json"
        source.parent.mkdir(parents=True)
        config.parent.mkdir(parents=True)
        source.write_text("pass\n", encoding="utf-8")
        config.write_text("{}\n", encoding="utf-8")
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("src/rby1_assembly/submodule/unplanned.py", result.stderr)
        self.assertIn("configs/tasks/unplanned.json", result.stderr)

    def test_path_outside_root_is_rejected(self) -> None:
        self.write_plan({"../outside": "0" * 64})
        result = self.run_checker()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("잘못된 입력 경로", result.stderr)


if __name__ == "__main__":
    unittest.main()
