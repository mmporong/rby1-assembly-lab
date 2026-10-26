#!/usr/bin/env python3
"""Fetch pinned RJ45 sources and convert supplier STEP with installed KiCad 10."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rby1_assembly.assets import fetch_manifest_assets, load_manifest, sha256_file


def extract_step(archive: Path, destination: Path, member: str, expected_sha: str) -> None:
    """Read one pinned member without allowing archive-supplied filesystem paths."""
    with zipfile.ZipFile(archive) as zipped:
        info = zipped.getinfo(member)
        if info.file_size > 8 * 1024 * 1024:
            raise ValueError("STEP member exceeds extraction size cap")
        payload = zipped.read(info)
    if hashlib.sha256(payload).hexdigest() != expected_sha:
        raise ValueError("extracted STEP SHA-256 mismatch")
    destination.write_bytes(payload)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/rj45-assets.json")
    parser.add_argument("--conversion-python", default="/usr/bin/python3",
                        help="Python with pcbnew, numpy and trimesh; USD Python may be separate")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    raw = json.loads(args.manifest.read_text())
    cache = ROOT / manifest.cache_root
    results = fetch_manifest_assets(manifest, cache)
    by_id = {result.asset_id: result for result in results}
    step = cache / "te-5555153-1.stp"
    extract_step(by_id["te_5555153_1_step_archive"].path, step,
                 raw["step_member"], raw["step_sha256"])
    subprocess.run([
        args.conversion_python, str(ROOT / "scripts/convert_step_model.py"),
        str(step), "--output-dir", str(cache), "--stem", "te-5555153-1",
    ], check=True)
    report = {
        "schema_version": 1,
        "manifest_sha256": sha256_file(args.manifest),
        "downloads": [{**asdict(result), "path": str(result.path)} for result in results],
        "step": {"path": str(step), "sha256": sha256_file(step)},
        "mesh": {"path": str(cache / "te-5555153-1-mesh.json"),
                 "sha256": sha256_file(cache / "te-5555153-1-mesh.json")},
        "scope": "Vendor jack visual CAD; plug derived from drawing. No electrical/latch/cable validation.",
    }
    (cache / "preparation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
