from pathlib import Path
import json, hashlib, subprocess, os, time, datetime, traceback, signal
P = Path(__file__).resolve().parent
PRIMARY = Path.home() / 'workspace/rby1-multibias-20261031b'
RUNTIME = Path.home() / 'workspace/rby1-g2-20261031'
DEADLINE = datetime.datetime.fromisoformat('2026-10-31T10:50:52+00:00').timestamp()
manifest = json.loads((P / 'sealed-inputs.json').read_text())

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def write(name, x):
    q = P / name
    t = q.with_suffix(q.suffix + '.tmp')
    t.write_text(json.dumps(x, indent=2, sort_keys=True) + '\n')
    t.replace(q)
start = time.time()
exit_code = 0
try:
    if not (PRIMARY / 'batch.validated.exit').read_text().strip() == '0':
        raise RuntimeError("operational gate failed: (PRIMARY / 'batch.validated.exit').read_text().strip() == '0'")
    if not (PRIMARY / 'batch.finished').exists():
        raise RuntimeError("operational gate failed: (PRIMARY / 'batch.finished').exists()")
    for rel, sha in manifest['primary_inputs'].items():
        if not digest(PRIMARY / rel) == sha:
            raise RuntimeError(rel)
    for rel, sha in manifest['media_inputs'].items():
        if not digest(P / rel) == sha:
            raise RuntimeError(rel)
    if not not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip():
        raise RuntimeError("operational gate failed: not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()")
    if not time.time() < DEADLINE - 300:
        raise RuntimeError('operational gate failed: time.time() < DEADLINE - 300')
    write('started.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'deadline_utc': '2026-10-31T10:50:52Z', 'primary_results_changed': False, 'selected_case': 'fresh-05', 'selection': 'after outcomes for illustration, not statistical evidence'})
    for mode in ['multi64', 'zero']:
        for rel, sha in manifest['primary_inputs'].items():
            if not digest(PRIMARY / rel) == sha:
                raise RuntimeError(rel)
        for rel, sha in manifest['media_inputs'].items():
            if not digest(P / rel) == sha:
                raise RuntimeError(rel)
        remain = DEADLINE - time.time() - 90
        if not remain > 120:
            raise RuntimeError('operational gate failed: remain > 120')
        write('progress.json', {'mode': mode, 'state': 'recording', 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        args = [str(RUNTIME / '.venv-isaac/bin/python'), '-u', str(P / 'record_rj45_multibias_replay.py'), '--mode', mode, '--runtime-root', str(RUNTIME / 'runtime-104721a'), '--current-root', str(P / 'current'), '--origin-protocol', str(PRIMARY / 'origin-protocol.json'), '--origin-protocol-sha256', manifest['primary_inputs']['origin-protocol.json'], '--contract', str(PRIMARY / 'contract.json'), '--contract-sha256', manifest['primary_inputs']['contract.json'], '--primary-report', str(PRIMARY / 'evaluate-report.json'), '--primary-report-sha256', manifest['primary_inputs']['evaluate-report.json'], '--train-report', str(PRIMARY / 'train-multi-report.json'), '--train-report-sha256', manifest['primary_inputs']['train-multi-report.json'], '--video', str(P / ('fresh05-' + mode + '.mp4')), '--report', str(P / ('fresh05-' + mode + '-report.json')), '--device', 'cuda:0', '--acknowledge-replay-only']
        if mode == 'multi64':
            args += ['--checkpoint', str(PRIMARY / 'train-multi/final_checkpoint.pt')]
        with (P / (mode + '.log')).open('x') as f:
            child = subprocess.Popen(args, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            write(mode + '-process.json', {'pid': child.pid, 'pgid': child.pid})
            try:
                child.wait(timeout=min(1200, remain))
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=10)
                raise
            r = child
        report = json.loads((P / ('fresh05-' + mode + '-report.json')).read_text())
        write(mode + '-exit.json', {'native_returncode': r.returncode, 'validated_report_status': report['status']})
        if not (r.returncode == 0 and report['status'] == 'completed'):
            raise RuntimeError((mode, report.get('error'), report.get('comparison')))
    write('final.json', {'status': 'completed', 'elapsed_s': time.time() - start, 'performance_based_reruns': 0, 'primary_results_changed': False})
except BaseException as e:
    exit_code = 1
    traceback.print_exc()
    write('final.json', {'status': 'invalid', 'elapsed_s': time.time() - start, 'error': {'type': type(e).__name__, 'message': str(e)}, 'performance_based_reruns': 0, 'primary_results_changed': False})
finally:
    write('finished.json', {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
raise SystemExit(exit_code)
