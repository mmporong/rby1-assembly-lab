#!/usr/bin/env bash
set -euo pipefail
job="$HOME/workspace/rby1-oracle-condition-20261031"
runtime="$HOME/workspace/rby1-g2-20261031"
media="$job/media"
export OMNI_KIT_ACCEPT_EULA=YES OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 OPENBLAS_CORETYPE=Zen
export NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL
export PYTHONPATH="$runtime/runtime-104721a/src"
py="$runtime/.venv-isaac/bin/python"
exec 9>"$media/queue.lock"
flock -n 9
while [[ ! -f "$job/revision5/oracle.validated.exit" ]]; do sleep 2; done
test "$(cat "$job/revision5/oracle.validated.exit")" = 0
python3 - <<'VERIFY'
import time,datetime,pathlib
assert datetime.datetime(2026,10,31,1,45,tzinfo=datetime.timezone.utc).timestamp()-time.time()>1170, 'insufficient guarded recording time'
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if a and pathlib.Path(a[0].decode(errors='replace')).name.startswith('python'):
  assert not any(v.endswith((b'run_rj45_oracle_ppo_probe.py',b'evaluate_rj45_oracle_checkpoint.py',b'record_rj45_oracle_probe.py')) for v in a),p
VERIFY
for mode in zero standard79; do
 rev=revision3
 protocol="$job/$rev/protocols/20261128.json"
 digest=5c6dff5a338994ae67cd64fb36ee702656371b9fbfa392e0edf66053741bd0d0
 extra=()
 if [[ "$mode" == standard79 ]]; then
  rev=revision4
  protocol="$job/$rev/protocol.json"
  digest=a1a00452c8c7a6151844e2b3bd56d9322a2baaa6f1c4b7e9c2bb925be07959ef
  extra=(--checkpoint "$job/$rev/results/20261128/standard79/final_checkpoint.pt")
 fi
 python3 - "$job/$rev/current" <<'VERIFY'
import hashlib,pathlib,sys
root=pathlib.Path(sys.argv[1])/'scripts'
for name,sha in {'record_rj45_oracle_probe.py':'46bc89c708afb4ebb664a488e734b121364c9505462200aeddebc73186410bcd','rj45_evaluation_video.py':'9aecdc29894dbc0ce5411663c3759bb4f5358b6ad768d44b437b9c0701d69034'}.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==sha,name
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:a=(p/'cmdline').read_bytes().split(b'\0')
 except OSError:continue
 if a and pathlib.Path(a[0].decode(errors='replace')).name.startswith('python'):
  assert not any(v.endswith((b'run_rj45_oracle_ppo_probe.py',b'evaluate_rj45_oracle_checkpoint.py',b'record_rj45_oracle_probe.py')) for v in a),p
VERIFY
 date -u +%FT%TZ > "$media/$mode.started"
 set +e
 timeout --signal=TERM --kill-after=10s 480s "$py" -u "$job/$rev/current/scripts/record_rj45_oracle_probe.py" --runtime-root "$runtime/runtime-104721a" --current-root "$job/$rev/current" --protocol "$protocol" --protocol-sha256 "$digest" --probe-report "$job/$rev/results/20261128/$mode-report.json" --mode "$mode" "${extra[@]}" --video "$media/$mode.mp4" --report "$media/$mode-report.json" --held-bank artifacts/rj45-4c-bank-generation-v2-20261028/held-grasp-bank.json --device cuda:0 --acknowledge-replay-only > "$media/$mode.log" 2>&1
 code=$?
 set -e
 printf '%s\n' "$code" > "$media/$mode.raw.exit"
 date -u +%FT%TZ > "$media/$mode.finished"
 if [[ "$code" != 0 ]]; then printf '%s\n' "$code" > "$media/$mode.validated.exit"; exit "$code"; fi
 set +e
 python3 - "$media" "$mode" "$digest" <<'VERIFY'
import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);m=sys.argv[2];r=json.loads((p/f'{m}-report.json').read_text())
assert r['status']=='completed' and r['mode']==m and r['claims']['fresh_replay_filmed'] is True
assert r['replay']['outcome']['classification'] in ('valid_success','valid_failure')
assert (p/f'{m}.mp4').stat().st_size>0
assert r['lineage']['protocol']['file_sha256']==sys.argv[3]
assert r['outcome_comparison']['classification_matches'] is True
if m=='zero':assert r['lineage']['checkpoint'] is None
else:assert r['lineage']['checkpoint']['sha256']=='935496bd8f6e8c0e53c151cbb497f7a0bcfb9768dd18e65de3874c6841583c02'
VERIFY
 status=$?
 set -e
 printf '%s\n' "$status" > "$media/$mode.validated.exit"
 test "$status" = 0
done
date -u +%FT%TZ > "$media/batch.finished"
