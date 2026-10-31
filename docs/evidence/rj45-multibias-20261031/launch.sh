#!/usr/bin/env bash
set -euo pipefail
job="$HOME/workspace/rby1-multibias-20261031b"
runtime="$HOME/workspace/rby1-g2-20261031"
cd "$job"
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_CORETYPE=Zen
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
export PYTHONPATH="$runtime/runtime-104721a/src"
python3 - <<'VERIFY'
from pathlib import Path
import hashlib,json,time,datetime
r=Path.cwd()
assert hashlib.sha256((r/'packet-members.json').read_bytes()).hexdigest()=='0d2a20a0154a93cca5807ae56283184d1249868c32a7796ea96ba002c532d7f8'
m=json.load(open('packet-members.json'))
for name,digest in m.items():
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
for phase in ['preflight','train-single','train-multi','evaluate']:
 assert not (r/phase).exists() and not (r/(phase+'-report.json')).exists()
a=json.load(open('contract.json'));deadline=datetime.datetime.fromisoformat(a['budget_and_stop']['deadline_utc'].replace('Z','+00:00')).timestamp();assert time.time()<deadline
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if args and Path(args[0].decode(errors='replace')).name.startswith('python'):
  assert not any(x.endswith((b'run_rj45_multibias_probe.py',b'evaluate_rj45_frozen_ablation.py',b'run_rj45_oracle_ppo_probe.py',b'evaluate_rj45_oracle_checkpoint.py',b'record_rj45_oracle_probe.py')) for x in args),p
(r/'packet-verified.json').write_text(json.dumps({'status':'pass','members':len(m),'deadline_utc':a['budget_and_stop']['deadline_utc']},indent=2)+'\n')
VERIFY
printf '%s\n' "$$" > launch.pid
set +e
timeout --signal=TERM --kill-after=15s 10500s python3 -u batch.py > batch.log 2>&1
code=$?
set -e
printf '%s\n' "$code" > batch.validated.exit
date -u +%FT%TZ > batch.finished
exit "$code"
