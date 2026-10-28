from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest import mock

from rby1_assembly.held_grasp import (
    bank_source_contract_sha256,
    held_grasp_bank_dependency_paths,
    load_verified_held_grasp_bank,
)

JsonObject = dict[str, Any]


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _entry(grasp_id: str, split: str, suffix: str) -> JsonObject:
    descriptor = {"lateral_offset_m": 0.0001 * len(grasp_id), "label": grasp_id}
    return {
        "grasp_id": grasp_id,
        "split": split,
        "generation_seed": len(grasp_id),
        "artifact": {
            "path": f"artifacts/held/{grasp_id}.json",
            "sha256": suffix * 64,
        },
        "descriptor": descriptor,
        "descriptor_sha256": _canonical_sha256(descriptor),
    }


class HeldGraspBankTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.project = Path(self.temporary.name)
        self.manifest_path = self.project / "artifacts/held/bank.json"
        self.task = {"task_id": "rj45-v1", "_config_sha256": "d" * 64}
        self.entries: list[JsonObject] = [
            _entry("train-a", "train", "a"),
            _entry("dev-a", "dev", "b"),
            _entry("final-a", "final", "c"),
        ]
        self._write_manifest(self.entries)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_manifest(self, entries: list[JsonObject]) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(
                {"schema_version": 1, "task_id": "rj45-v1", "entries": entries},
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _artifact(grasp_id: str, *, source: str = "e") -> JsonObject:
        return {
            "schema_version": 1,
            "task_id": "rj45-v1",
            "provenance": {
                "task_config_sha256": "d" * 64,
                "source_and_config_sha256": {"src/example.py": source * 64},
            },
            "fixture_id": grasp_id,
        }

    def _loader(self, project: Path, task: dict, path: Path):
        del project, task
        grasp_id = path.stem
        entry = next(item for item in self.entries if item["grasp_id"] == grasp_id)
        return self._artifact(grasp_id), {
            "artifact_sha256": entry["artifact"]["sha256"]
        }

    def _write_dependency_artifact(self, grasp_id: str = "train-a") -> tuple[Path, Path, Path]:
        report = self.project / f"artifacts/evidence/{grasp_id}-report.json"
        trace = self.project / f"artifacts/evidence/{grasp_id}-trace.jsonl"
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("{}\n", encoding="utf-8")
        trace.write_text('{"step":0}\n', encoding="utf-8")
        artifact = self.project / f"artifacts/held/{grasp_id}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "schema_version": 1,
            "task_id": "rj45-v1",
            "provenance": {
                "nominal_report_path": str(report.relative_to(self.project)),
                "nominal_report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
                "nominal_trace_path": str(trace.relative_to(self.project)),
                "nominal_trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
            },
        }
        artifact.write_text(json.dumps(value) + "\n", encoding="utf-8")
        entry = next(item for item in self.entries if item["grasp_id"] == grasp_id)
        entry["artifact"]["sha256"] = hashlib.sha256(artifact.read_bytes()).hexdigest()
        self._write_manifest(self.entries)
        return artifact, report, trace

    def test_dependency_paths_return_selected_artifact_report_and_trace(self) -> None:
        artifact, report, trace = self._write_dependency_artifact()
        dependencies = held_grasp_bank_dependency_paths(
            self.project, self.manifest_path, "train"
        )
        self.assertEqual(
            dependencies,
            (self.manifest_path, artifact, report, trace),
        )

    def test_dependency_paths_do_not_open_unselected_final_artifact(self) -> None:
        artifact, report, trace = self._write_dependency_artifact()
        final_path = self.project / "artifacts/held/final-a.json"
        final_path.write_text("not-json", encoding="utf-8")
        dependencies = held_grasp_bank_dependency_paths(
            self.project, self.manifest_path, "train"
        )
        self.assertEqual(dependencies, (self.manifest_path, artifact, report, trace))
        self.assertNotIn(final_path, dependencies)

    def test_dependency_paths_reject_manifest_escape_before_artifact_open(self) -> None:
        self._write_dependency_artifact()
        self.entries[1]["artifact"]["path"] = "../../outside.json"
        self._write_manifest(self.entries)
        with mock.patch(
            "rby1_assembly.held_grasp.load_finite_json",
            wraps=__import__(
                "rby1_assembly.held_grasp", fromlist=["load_finite_json"]
            ).load_finite_json,
        ) as loader:
            with self.assertRaisesRegex(ValueError, "must stay inside the project"):
                held_grasp_bank_dependency_paths(
                    self.project, self.manifest_path, "train"
                )
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(Path(loader.call_args.args[0]), self.manifest_path)

    def test_dependency_paths_reject_selected_artifact_sha_mismatch(self) -> None:
        artifact, _report, _trace = self._write_dependency_artifact()
        artifact.write_text("{}\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
            held_grasp_bank_dependency_paths(
                self.project, self.manifest_path, "train"
            )

    def test_bank_source_contract_hash_is_order_stable_and_strict(self) -> None:
        first = {
            "provenance_contract": {
                "source_and_config_sha256": {
                    "src/b.py": "b" * 64,
                    "src/a.py": "a" * 64,
                }
            }
        }
        second = {
            "provenance_contract": {
                "source_and_config_sha256": {
                    "src/a.py": "a" * 64,
                    "src/b.py": "b" * 64,
                }
            }
        }
        expected = _canonical_sha256(second["provenance_contract"]["source_and_config_sha256"])
        self.assertEqual(bank_source_contract_sha256(first), expected)
        self.assertEqual(bank_source_contract_sha256(second), expected)
        for source_map, message in (({}, "must not be empty"), ({"src/a.py": "bad"}, "lowercase SHA-256")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                bank_source_contract_sha256(
                    {"provenance_contract": {"source_and_config_sha256": source_map}}
                )

    def test_loads_only_selected_split_and_keeps_final_sealed(self) -> None:
        with mock.patch(
            "rby1_assembly.held_grasp.load_verified_held_grasp_artifact",
            side_effect=self._loader,
        ) as loader:
            artifacts, proof = load_verified_held_grasp_bank(
                self.project, self.task, self.manifest_path, "train"
            )

        self.assertEqual([item["fixture_id"] for item in artifacts], ["train-a"])
        self.assertEqual(loader.call_count, 1)
        self.assertEqual(loader.call_args.args[2].name, "train-a.json")
        self.assertEqual(proof["selected_grasp_ids"], ["train-a"])
        self.assertEqual(proof["manifest_split_counts"], {"train": 1, "dev": 1, "final": 1})
        self.assertTrue(proof["no_final_opened"])
        self.assertFalse(proof["final_opened"])

    def test_rejects_final_without_opening_any_artifact(self) -> None:
        with mock.patch(
            "rby1_assembly.held_grasp.load_verified_held_grasp_artifact"
        ) as loader:
            with self.assertRaisesRegex(ValueError, "requires allow_final"):
                load_verified_held_grasp_bank(
                    self.project, self.task, self.manifest_path, "final"
                )
        loader.assert_not_called()

    def test_rejects_path_escape_in_unselected_entry(self) -> None:
        self.entries[1]["artifact"]["path"] = "../../outside.json"
        self._write_manifest(self.entries)
        with mock.patch(
            "rby1_assembly.held_grasp.load_verified_held_grasp_artifact"
        ) as loader:
            with self.assertRaisesRegex(ValueError, "must stay inside the project"):
                load_verified_held_grasp_bank(
                    self.project, self.task, self.manifest_path, "train"
                )
        loader.assert_not_called()

    def test_rejects_invalid_split_and_hash_metadata(self) -> None:
        for mutate, message in (
            (lambda entries: entries[1].__setitem__("split", "test"), "split must be"),
            (
                lambda entries: entries[1]["artifact"].__setitem__("sha256", "ABC"),
                "lowercase SHA-256",
            ),
            (
                lambda entries: entries[1].__setitem__("descriptor_sha256", "f" * 64),
                "descriptor SHA-256 mismatch",
            ),
        ):
            with self.subTest(message=message):
                entries = json.loads(json.dumps(self.entries))
                mutate(entries)
                self._write_manifest(entries)
                with self.assertRaisesRegex(ValueError, message):
                    load_verified_held_grasp_bank(
                        self.project, self.task, self.manifest_path, "train"
                    )

    def test_rejects_duplicate_id_and_artifact_hash(self) -> None:
        for field, message in (
            ("grasp_id", "duplicate held-grasp grasp_id"),
            ("artifact_sha256", "duplicate held-grasp artifact SHA-256"),
        ):
            with self.subTest(field=field):
                entries = json.loads(json.dumps(self.entries))
                if field == "grasp_id":
                    entries[1]["grasp_id"] = entries[0]["grasp_id"]
                else:
                    entries[1]["artifact"]["sha256"] = entries[0]["artifact"]["sha256"]
                self._write_manifest(entries)
                with self.assertRaisesRegex(ValueError, message):
                    load_verified_held_grasp_bank(
                        self.project, self.task, self.manifest_path, "train"
                    )

    def test_rejects_selected_artifact_hash_mismatch(self) -> None:
        with mock.patch(
            "rby1_assembly.held_grasp.load_verified_held_grasp_artifact",
            return_value=(self._artifact("train-a"), {"artifact_sha256": "f" * 64}),
        ):
            with self.assertRaisesRegex(ValueError, "artifact SHA-256 mismatch"):
                load_verified_held_grasp_bank(
                    self.project, self.task, self.manifest_path, "train"
                )

    def test_rejects_mixed_selected_provenance_contracts(self) -> None:
        self.entries.insert(1, _entry("train-b", "train", "f"))
        self._write_manifest(self.entries)

        def mixed_loader(project: Path, task: dict, path: Path):
            del project, task
            grasp_id = path.stem
            source = "e" if grasp_id == "train-a" else "9"
            entry = next(item for item in self.entries if item["grasp_id"] == grasp_id)
            return self._artifact(grasp_id, source=source), {
                "artifact_sha256": entry["artifact"]["sha256"]
            }

        with mock.patch(
            "rby1_assembly.held_grasp.load_verified_held_grasp_artifact",
            side_effect=mixed_loader,
        ):
            with self.assertRaisesRegex(ValueError, "mixed provenance contracts"):
                load_verified_held_grasp_bank(
                    self.project, self.task, self.manifest_path, "train"
                )


if __name__ == "__main__":
    unittest.main()
