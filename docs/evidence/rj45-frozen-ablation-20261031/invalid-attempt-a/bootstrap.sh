#!/usr/bin/env bash
set -euo pipefail
job="$HOME/workspace/rby1-frozen-ablation-20261031a"
cd "$job"
python3 - <<'VERIFY'
from pathlib import Path
import hashlib,tarfile
p=Path('payload.tar.gz')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='aa8f57e276e74cd8bfbd9698da4882438f791606041bd535e1506859daf8b8e4'
with tarfile.open(p) as t:
 members=t.getmembers()
 assert len(members)==14
 for x in members:
  z=Path(x.name)
  assert x.isfile() and not z.is_absolute() and '..' not in z.parts
  assert not z.exists(),str(z)
 t.extractall(filter='data')
assert hashlib.sha256(Path('launch.sh').read_bytes()).hexdigest()=='87079069f79fe780a84eea1562a27d923cd0edb4d85c88a45aa8db7723f4bf19'
VERIFY
exec bash launch.sh
