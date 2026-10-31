#!/usr/bin/env bash
set -euo pipefail
job="$HOME/workspace/rby1-frozen-ablation-20261031b"
cd "$job"
python3 - <<'VERIFY'
from pathlib import Path
import hashlib,tarfile
p=Path('payload.tar.gz')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='16273faf72b9384539de02fd52019c5c19c18c319640582c64950ffceac942cc'
with tarfile.open(p) as t:
 members=t.getmembers()
 assert len(members)==15
 for x in members:
  z=Path(x.name)
  assert x.isfile() and not z.is_absolute() and '..' not in z.parts
  assert not z.exists(),str(z)
 t.extractall(filter='data')
assert hashlib.sha256(Path('launch.sh').read_bytes()).hexdigest()=='66e3fd0596ae305c13b75e1f6a9a1a5ef2f6b02d247ec2d29ef09f724a153103'
VERIFY
exec bash launch.sh
