"""Release labels must not be confused with installed distribution versions."""

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("preflight", ROOT / "scripts/preflight.py")
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightVersionTests(unittest.TestCase):
    def inspect_with(self, versions):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "assets/generated").mkdir(parents=True)
            (root / "assets/generated/scene.usda").touch()
            (root / "configs/runtime.json").write_text((ROOT / "configs/runtime.json").read_text())
            with patch.object(preflight, "ROOT", root), patch.object(
                preflight.importlib.metadata, "version", side_effect=versions.__getitem__
            ), patch.object(preflight, "command", return_value={"returncode": 0, "stdout": "", "stderr": ""}):
                return preflight.inspect()

    def test_release_and_distribution_versions_are_distinct(self):
        versions = {"isaacsim": "5.1.0.0", "isaaclab": "0.47.2", "isaaclab-rl": "0.4.4",
                    "rsl-rl-lib": "3.0.1", "torch": "2.7.0+cu128", "numpy": "1.26.0", "h5py": "3.14.0",
                    "setuptools": "80.9.0", "wheel": "0.45.1"}
        report = self.inspect_with(versions)
        package_checks = {k: v for k, v in report["checks"].items() if k.endswith("_version")}
        self.assertTrue(all(package_checks.values()), package_checks)
        self.assertFalse(report["physics_executed"])

    def test_different_lab_distribution_is_rejected(self):
        versions = {"isaacsim": "5.1.0.0", "isaaclab": "0.48.0", "isaaclab-rl": "0.4.4",
                    "rsl-rl-lib": "3.0.1", "torch": "2.7.0+cu128", "numpy": "1.26.0", "h5py": "3.14.0",
                    "setuptools": "80.9.0", "wheel": "0.45.1"}
        report = self.inspect_with(versions)
        self.assertFalse(report["checks"]["isaaclab_version"])
        self.assertFalse(report["prerequisites_passed"])


if __name__ == "__main__":
    unittest.main()
