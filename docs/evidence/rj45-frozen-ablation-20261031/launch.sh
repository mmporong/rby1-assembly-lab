#!/usr/bin/env bash
set -euo pipefail
job="$HOME/workspace/rby1-frozen-ablation-20261031b"
runtime="$HOME/workspace/rby1-g2-20261031"
cd "$job"
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_CORETYPE=Zen
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
export PYTHONPATH="$runtime/runtime-104721a/src"
python3 - <<'VERIFY'
from pathlib import Path
import hashlib,json
root=Path.cwd();assert hashlib.sha256((root/'packet-members.json').read_bytes()).hexdigest()=='909927eb9991815be98b99935e050ced071c4c72f4554f9ef349d530cb3b68da', 'packet manifest SHA mismatch'
m=json.load(open('packet-members.json'))
for name,sha in m.items():
 p=Path(name)
 assert not p.is_absolute() and '..' not in p.parts
 assert hashlib.sha256(p.read_bytes()).hexdigest()==sha,name
assert not (root/'results').exists() and not (root/'results-report.json').exists()
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if a and Path(a[0].decode(errors='replace')).name.startswith('python'):
  assert not any(x.endswith((b'evaluate_rj45_frozen_ablation.py',b'run_rj45_oracle_ppo_probe.py',b'evaluate_rj45_oracle_checkpoint.py',b'record_rj45_oracle_probe.py')) for x in a),p
ckpt=Path.home()/'workspace/rby1-oracle-condition-20261031/revision4/results/20261128/standard79/final_checkpoint.pt'
assert hashlib.sha256(ckpt.read_bytes()).hexdigest()=='935496bd8f6e8c0e53c151cbb497f7a0bcfb9768dd18e65de3874c6841583c02'
(root/'packet-verified.json').write_text(json.dumps({'status':'pass','members':len(m),'checkpoint_sha256':hashlib.sha256(ckpt.read_bytes()).hexdigest()},indent=2)+'\n')
VERIFY
contract_sha="$(sha256sum contract.json | cut -d' ' -f1)"
printf '%s\n' "$$" > launch.pid
date -u +%FT%TZ > batch.started
set +e
timeout --signal=TERM --kill-after=10s 4000s "$runtime/.venv-isaac/bin/python" -u current/scripts/evaluate_rj45_frozen_ablation.py --runtime-root "$runtime/runtime-104721a" --current-root "$job/current" --origin-protocol "$job/origin-protocol.json" --protocol-sha256 a1a00452c8c7a6151844e2b3bd56d9322a2baaa6f1c4b7e9c2bb925be07959ef --origin-report "$job/standard79-origin-report.json" --checkpoint "$HOME/workspace/rby1-oracle-condition-20261031/revision4/results/20261128/standard79/final_checkpoint.pt" --source-trace "$job/standard79-source-trace.jsonl.gz" --contract "$job/contract.json" --contract-sha256 "$contract_sha" --output-dir "$job/results" --held-bank artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json --headless --device cuda:0 > run.log 2>&1
code=$?
set -e
printf '%s\n' "$code" > batch.raw.exit
date -u +%FT%TZ > batch.finished
if [[ "$code" != 0 ]]; then printf '%s\n' "$code" > batch.validated.exit; exit "$code"; fi
set +e
python3 - <<'VERIFY'
import json
r=json.load(open('results-report.json'))
assert r['status']=='completed' and r['learning_executed'] is False
assert len(r['episodes'])==20 and len(r['execution_order'])==20
assert all(x['evaluation']['classification'] in ('valid_success','valid_failure') for x in r['episodes'])
VERIFY
status=$?
set -e
printf '%s\n' "$status" > batch.validated.exit
exit "$status"
