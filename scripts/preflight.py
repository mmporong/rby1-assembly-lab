#!/usr/bin/env python3
"""서버와 Python 패키지를 읽기 전용으로 확인한다. 시뮬레이션은 시작하지 않는다."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"returncode": -1, "error": str(error)}


def inspect() -> dict:
    expected = json.loads((ROOT / "configs/runtime.json").read_text())
    versions = {}
    for name in ("isaacsim", "isaaclab", "isaaclab-rl", "rsl-rl-lib", "torch", "numpy", "h5py"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpu = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu", "--format=csv,noheader"])
    processes = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"])
    checks = {
        "python_3_11": sys.version_info[:2] == (3, 11),
        "gpu_driver_query": gpu["returncode"] == 0,
        "generated_scene_exists": (ROOT / "assets/generated/scene.usda").is_file(),
    }
    required_versions = {
        "isaacsim": expected["isaac_sim"],
        "isaaclab": expected["isaac_lab"],
        "isaaclab-rl": expected["isaac_lab_rl"],
        "rsl-rl-lib": expected["rsl_rl"],
        "torch": expected["torch"] + "+" + expected["torch_cuda_build"],
        "numpy": expected["numpy"],
        "h5py": expected["h5py"],
    }
    checks.update({name + "_version": versions[name] == version
                   for name, version in required_versions.items()})
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "python": platform.python_version(),
        "packages": versions, "required_packages": required_versions,
        "gpu": gpu, "gpu_processes": processes,
        "free_disk_gib": round(shutil.disk_usage(ROOT).free / 1024 ** 3, 2),
        "git_head": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "checks": checks, "prerequisites_passed": all(checks.values()),
        "physics_executed": False,
        "note": "패키지와 드라이버 사전 확인이다. RTX/Vulkan 호환성·장면 로딩·물리 실행 성공은 별도 실행으로 검증한다.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/preflight.json")
    args = parser.parse_args()
    report = inspect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["prerequisites_passed"] else 2)
