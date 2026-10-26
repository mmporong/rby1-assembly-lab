import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rby1_assembly.run_provenance import InputMutationError, capture_run_inputs


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    (tmp_path / "src/pkg").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "assets/generated").mkdir(parents=True)
    generated = tmp_path / "assets/generated/scene.usda"
    generated.write_text("#usda 1.0\n", encoding="utf-8")
    report = tmp_path / "assets/generated/build-report.json"
    report.write_text(
        json.dumps({"files": {"scene.usda": _sha(generated)}}), encoding="utf-8"
    )
    task = tmp_path / "configs/task.json"
    task.write_text(
        json.dumps({"assets": {"build_report": "assets/generated/build-report.json"}}),
        encoding="utf-8",
    )
    runtime = tmp_path / "configs/runtime.json"
    runtime.write_text("{}", encoding="utf-8")
    source = tmp_path / "src/pkg/controller.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    script = tmp_path / "scripts/run.py"
    script.write_text("pass\n", encoding="utf-8")
    return task, runtime, source, script, generated


def _capture(tmp_path: Path):
    task, runtime, source, script, generated = _fixture(tmp_path)
    snapshot = capture_run_inputs(
        tmp_path,
        task_path=task,
        config_paths=(runtime,),
        source_roots=((tmp_path / "src/pkg", False),),
        source_files=(script,),
    )
    return snapshot, task, runtime, source, script, generated


class RunProvenanceTests(unittest.TestCase):
    def test_snapshot_reports_start_hashes_and_accepts_unchanged_inputs(self):
        with TemporaryDirectory() as directory:
            snapshot, task, runtime, source, script, generated = _capture(Path(directory))
            snapshot.assert_unchanged()
            self.assertEqual(snapshot.hash_for(task), _sha(task))
            self.assertEqual(snapshot.json_for(runtime), {})
            self.assertEqual(
                snapshot.source_sha256,
                {
                    "scripts/run.py": _sha(script),
                    "src/pkg/controller.py": _sha(source),
                },
            )
            self.assertEqual(
                snapshot.generated_asset_sha256, {"scene.usda": _sha(generated)}
            )

    def test_snapshot_rejects_changed_input(self):
        for target_name in ("runtime", "source", "generated"):
            with self.subTest(target_name=target_name), TemporaryDirectory() as directory:
                snapshot, _task, runtime, source, _script, generated = _capture(
                    Path(directory)
                )
                target = {
                    "runtime": runtime,
                    "source": source,
                    "generated": generated,
                }[target_name]
                target.write_bytes(target.read_bytes() + b"changed")
                with self.assertRaises(InputMutationError) as caught:
                    snapshot.assert_unchanged()
                self.assertIn(snapshot.key_for(target), caught.exception.changed)

    def test_snapshot_rejects_deleted_input(self):
        with TemporaryDirectory() as directory:
            snapshot, _task, _runtime, source, _script, _generated = _capture(
                Path(directory)
            )
            source.unlink()
            with self.assertRaises(InputMutationError) as caught:
                snapshot.assert_unchanged()
            self.assertIn("src/pkg/controller.py", caught.exception.removed)

    def test_snapshot_rejects_source_added_after_start(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            snapshot, *_ = _capture(tmp_path)
            added = tmp_path / "src/pkg/late_edit.py"
            added.write_text("VALUE = 2\n", encoding="utf-8")
            with self.assertRaises(InputMutationError) as caught:
                snapshot.assert_unchanged()
            self.assertIn("src/pkg/late_edit.py", caught.exception.added)

    def test_bootstrap_snapshot_must_match_later_snapshot(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            bootstrap, _task, runtime, _source, _script, _generated = _capture(tmp_path)
            runtime.write_text('{"changed": true}', encoding="utf-8")
            later = capture_run_inputs(
                tmp_path,
                task_path=tmp_path / "configs/task.json",
                config_paths=(runtime,),
                source_roots=((tmp_path / "src/pkg", False),),
                source_files=(tmp_path / "scripts/run.py",),
            )
            with self.assertRaises(InputMutationError) as caught:
                bootstrap.assert_preserved_in(later)
            self.assertIn("configs/runtime.json", caught.exception.changed)

    def test_capture_rejects_generated_asset_that_disagrees_with_report(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            task, runtime, _source, script, generated = _fixture(tmp_path)
            generated.write_text("different\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "generated asset hashes disagree"):
                capture_run_inputs(
                    tmp_path,
                    task_path=task,
                    config_paths=(runtime,),
                    source_roots=((tmp_path / "src/pkg", False),),
                    source_files=(script,),
                )


if __name__ == "__main__":
    unittest.main()
