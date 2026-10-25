#!/usr/bin/env python3
"""manifest에 고정된 런타임 자산을 무결성 검사와 함께 받는다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from rby1_assembly.assets import (  # noqa: E402
    AssetError,
    DEFAULT_MAX_BYTES,
    DEFAULT_TIMEOUT_S,
    fetch_manifest_assets,
    load_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=REPO_ROOT / "configs" / "assets.json"
    )
    parser.add_argument("--cache-root", type=Path, default=REPO_ROOT / "assets" / "cache")
    parser.add_argument("--asset", action="append", dest="asset_ids", help="manifest asset id")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    args = parser.parse_args()
    if args.timeout <= 0 or args.max_bytes <= 0:
        parser.error("--timeout and --max-bytes must be positive")

    try:
        manifest = load_manifest(args.manifest)
        results = fetch_manifest_assets(
            manifest,
            args.cache_root,
            args.asset_ids,
            timeout_s=args.timeout,
            max_bytes=args.max_bytes,
        )
    except AssetError as exc:
        print(f"asset fetch failed: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            [
                {
                    "asset_id": item.asset_id,
                    "path": str(item.path),
                    "status": item.status,
                    "size_bytes": item.size_bytes,
                    "sha256": item.sha256,
                }
                for item in results
            ],
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
