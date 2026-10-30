"""One-time, ledger-gated archival; no training data or checkpoint deletion."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent
PLAN = json.loads((BASE / "plan.json").read_text())
LEDGER_PATH = BASE / "ledger-before.json"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def note(event: str, **fields: object) -> None:
    with (BASE / "progress.jsonl").open("a") as stream:
        stream.write(json.dumps({"event": event, **fields}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def physical_path(logical: Path) -> Path:
    # Resolve the parent only: the file may later itself become a symlink.
    return logical.parent.resolve(strict=True) / logical.name


def verify_membership(items: dict, *, archived: bool) -> None:
    def kind(mode: int) -> str:
        return "directory" if stat.S_ISDIR(mode) else "regular" if stat.S_ISREG(mode) else "symlink" if stat.S_ISLNK(mode) else "other"

    for move in PLAN["moves"]:
        old_prefix = move["old_path"]
        expected = {(path[len(old_prefix):].lstrip("/"), item["type"])
                    for path, item in items.items()
                    if path == old_prefix or path.startswith(old_prefix + "/")}
        base = ROOT / move["new_path" if archived else "old_path"]
        actual = {(str(path.relative_to(base)) if path != base else "", kind(path.lstat().st_mode))
                  for path in [base, *base.rglob("*")]}
        if expected != actual:
            raise RuntimeError(f"Subtree membership changed: {old_prefix}")


def main() -> None:
    if not __debug__:
        raise RuntimeError("Safety checks require a non-optimized Python invocation")
    assert sha(LEDGER_PATH) == PLAN["ledger_before_sha256"]
    ledger = json.loads(LEDGER_PATH.read_text())
    assert ledger["root"] == str(ROOT)
    assert not ledger["open_writer_check"]["writers"]
    items = {item["path"]: item for item in ledger["items"]}
    assert not (BASE / "progress.jsonl").exists(), "Inspect recovery ledger before retry"
    # Revalidate every byte before the first mutation.
    for item in items.values():
        path = ROOT / item["path"]
        info = path.lstat()
        assert info.st_ino == item["inode"] and info.st_dev == item["device"]
        assert info.st_mtime_ns == item["mtime_ns"]
        assert stat.S_IMODE(info.st_mode) == item["mode"]
        assert (info.st_uid, info.st_gid) == (item["uid"], item["gid"])
        if item["type"] == "regular":
            assert sha(path) == item["sha256"]
    before_available = shutil.disk_usage(ROOT).free
    before_blocks = sum((ROOT / i["path"]).lstat().st_blocks * 512
                        for i in items.values() if i["type"] == "regular")
    note("validated", ledger_sha256=PLAN["ledger_before_sha256"])
    verify_membership(items, archived=False)
    for move in PLAN["moves"]:
        old, new = ROOT / move["old_path"], ROOT / move["new_path"]
        assert old.is_dir() and not old.is_symlink() and not new.exists()
        new.parent.mkdir(parents=True, exist_ok=True)
        assert old.stat().st_dev == new.parent.stat().st_dev
        note("move_intent", old=move["old_path"], new=move["new_path"])
        old.rename(new)
        old.symlink_to(os.path.relpath(new, old.parent), target_is_directory=True)
        assert old.resolve(strict=True) == new
        note("moved", old=move["old_path"], target=os.readlink(old))
    verify_membership(items, archived=True)
    verified = 0
    for item in items.values():
        if item["type"] == "regular":
            path = ROOT / item["path"]
            assert path.stat().st_size == item["size"]
            assert sha(path) == item["sha256"]
            info = path.stat()
            assert (info.st_dev, info.st_ino) == (item["device"], item["inode"])
            assert stat.S_IMODE(info.st_mode) == item["mode"]
            assert info.st_mtime_ns == item["mtime_ns"]
            assert (info.st_uid, info.st_gid) == (item["uid"], item["gid"])
            verified += 1
    for move in PLAN["moves"]:
        assert (ROOT / move["old_path"]).resolve(strict=True) == ROOT / move["new_path"]
    after_blocks = sum(physical_path(ROOT / i["path"]).lstat().st_blocks * 512
                       for i in items.values() if i["type"] == "regular")
    receipt = {"status": "completed", "ledger_before_sha256": PLAN["ledger_before_sha256"],
               "verified_regular_files": verified, "archived_directory_count": len(PLAN["moves"]),
               "archived_logical_bytes": sum(m["logical_bytes"] for m in PLAN["moves"]),
               "dedup_group_count": 0, "dedup_path_count": 0,
               "logical_duplicate_bytes_saved": 0,
               "allocated_bytes_saved": before_blocks - after_blocks,
               "df_available_before_bytes": before_available,
               "df_available_after_bytes": shutil.disk_usage(ROOT).free,
               "all_original_paths_readable": True, "all_original_byte_hashes_match": True,
               "all_original_file_inodes_modes_mtimes_match": True,
               "training_data_and_checkpoints_deleted": False,
               "progress_sha256": sha(BASE / "progress.jsonl")}
    (BASE / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
