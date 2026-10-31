from pathlib import Path
import json, shutil, hashlib, subprocess, tarfile
ROOT = Path.cwd()
P = ROOT / 'artifacts/cloud/rj45-multibias-20261031'
R = P / 'completed'
S = P / 'media-packet'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
a = json.loads((ROOT / 'docs/evidence/rj45-multibias-20261031/audit.json').read_text())
if not (a['status'] == 'pass' and a['evaluation']['episode_count'] == 27):
    raise RuntimeError("operational gate failed: a['status'] == 'pass' and a['evaluation']['episode_count'] == 27")
if not sha(ROOT / 'scripts/record_rj45_multibias_replay.py') == '49bf950a0b466820d82b5dfec1aae6bee498c8ac0b0428f24908b387d04a4e2f':
    raise RuntimeError("operational gate failed: sha(ROOT / 'scripts/record_rj45_multibias_replay.py') == '49bf950a0b466820d82b5dfec1aae6bee498c8ac0b0428f24908b387d04a4e2f'")
S.mkdir(exist_ok=False)
shutil.copytree(R / 'current', S / 'current', ignore=shutil.ignore_patterns('__pycache__'))
shutil.copyfile(ROOT / 'scripts/rj45_evaluation_video.py', S / 'current/scripts/rj45_evaluation_video.py')
shutil.copyfile(ROOT / 'scripts/record_rj45_multibias_replay.py', S / 'record_rj45_multibias_replay.py')
shutil.copyfile(P / 'media-runner.py', S / 'media-runner.py')
inputs = ['contract.json', 'origin-protocol.json', 'evaluate-report.json', 'train-multi-report.json', 'train-multi/final_checkpoint.pt']
for f in sorted((R / 'current').rglob('*')):
    if f.is_file() and '__pycache__' not in f.parts:
        inputs.append(str(f.relative_to(R)))
m = {'schema': 'rj45_multibias_media_sealed_v1', 'deadline_utc': '2026-10-31T10:50:52Z', 'primary_inputs': {rel: sha(R / rel) for rel in inputs}, 'media_inputs': {str(f.relative_to(S)): sha(f) for f in sorted(S.rglob('*')) if f.is_file()}, 'primary_audit_sha256': sha(ROOT / 'docs/evidence/rj45-multibias-20261031/audit.json')}
(S / 'sealed-inputs.json').write_text(json.dumps(m, indent=2, sort_keys=True) + '\n')
archive = P / 'media-packet.tar.gz'
with tarfile.open(archive, 'w:gz') as t:
    for f in sorted(S.rglob('*')):
        if f.is_file():
            t.add(f, arcname=str(f.relative_to(S)), recursive=False)
subprocess.run(['scp', '-q', str(archive), 'rby1-oracle-crusoe-20261031a:workspace/rby1-multibias-media-packet-20261031a.tar.gz'], check=True, timeout=60)
remote = 'from pathlib import Path\nimport tarfile, hashlib, json, subprocess, os\np = Path.home() / \'workspace/rby1-multibias-media-20261031a\'\np.mkdir(exist_ok=False)\na = Path.home() / \'workspace/rby1-multibias-media-packet-20261031a.tar.gz\'\nif not hashlib.sha256(a.read_bytes()).hexdigest() == EXPECTED:\n    raise RuntimeError(\'operational gate failed: hashlib.sha256(a.read_bytes()).hexdigest() == EXPECTED\')\nwith tarfile.open(a) as t:\n    for m in t.getmembers():\n        z = Path(m.name)\n        if not (m.isfile() and (not z.is_absolute()) and (\'..\' not in z.parts)):\n            raise RuntimeError("operational gate failed: m.isfile() and (not z.is_absolute()) and (\'..\' not in z.parts)")\n    t.extractall(p, filter=\'data\')\nmanifest = json.loads((p / \'sealed-inputs.json\').read_text())\nfor rel, sha in manifest[\'media_inputs\'].items():\n    if not hashlib.sha256((p / rel).read_bytes()).hexdigest() == sha:\n        raise RuntimeError(rel)\nenv = os.environ.copy()\nenv.pop(\'PYTHONOPTIMIZE\', None)\nenv.update({\'OMNI_KIT_ACCEPT_EULA\': \'YES\', \'OPENBLAS_NUM_THREADS\': \'1\', \'OMP_NUM_THREADS\': \'4\', \'OPENBLAS_CORETYPE\': \'Zen\', \'NPY_DISABLE_CPU_FEATURES\': \'AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL\', \'PYTHONPATH\': str(Path.home() / \'workspace/rby1-g2-20261031/runtime-104721a/src\')})\nwith (p / \'bootstrap.log\').open(\'x\') as f:\n    child = subprocess.Popen([\'python3\', str(p / \'media-runner.py\')], cwd=p, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)\n(p / \'bootstrap-pid.json\').write_text(json.dumps({\'pid\': child.pid, \'pgid\': child.pid}) + \'\\n\')\nprint(json.dumps({\'pid\': child.pid, \'remote_job\': str(p), \'sealed_inputs_sha256\': hashlib.sha256((p / \'sealed-inputs.json\').read_bytes()).hexdigest()}))\n'
remote = 'EXPECTED=' + repr(sha(archive)) + '\n' + remote
r = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'rby1-oracle-crusoe-20261031a', 'python3 -'], input=remote, capture_output=True, text=True, timeout=45, check=True)
(P / 'media-launch.json').write_text(r.stdout)
print(r.stdout)
