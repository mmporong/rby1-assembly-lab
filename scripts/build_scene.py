#!/usr/bin/env python3
"""캐시한 공식 자산을 상대 참조하는 조립 USD를 만든다."""

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rby1_assembly.scene import build_scene

if __name__ == "__main__":
    print(json.dumps(build_scene(ROOT), ensure_ascii=False, indent=2))
