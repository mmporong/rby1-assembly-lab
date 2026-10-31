#!/usr/bin/env python3
"""Execute the sealed four-phase experiment and reject incomplete reports."""
from pathlib import Path
import json,subprocess,sys,time,traceback,datetime,hashlib
JOB=Path(__file__).resolve().parent
RUNTIME=Path.home()/'workspace/rby1-g2-20261031'
PYTHON=RUNTIME/'.venv-isaac/bin/python'

def write(name,value):
 p=JOB/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n');q.replace(p)

def read(phase):return json.loads((JOB/f'{phase}-report.json').read_text())

def run(phase):
 write('batch-progress.json',{'phase':phase,'state':'starting','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 contract_sha=hashlib.sha256((JOB/'contract.json').read_bytes()).hexdigest()
 protocol_sha=hashlib.sha256((JOB/'origin-protocol.json').read_bytes()).hexdigest()
 args=[str(PYTHON),'-u',str(JOB/'current/scripts/run_rj45_multibias_probe.py'),'--phase',phase,'--runtime-root',str(RUNTIME/'runtime-104721a'),'--current-root',str(JOB/'current'),'--origin-protocol',str(JOB/'origin-protocol.json'),'--origin-protocol-sha256',protocol_sha,'--contract',str(JOB/'contract.json'),'--contract-sha256',contract_sha,'--output-dir',str(JOB/phase),'--headless','--device','cuda:0']
 if phase=='evaluate':
  for label,train in [('single64','train-single'),('multi64','train-multi')]:
   path=JOB/train/'final_checkpoint.pt'
   assert hashlib.sha256(path.read_bytes()).hexdigest()==read(train)['checkpoint']['sha256'],label
 if phase=='evaluate':args+=['--single-checkpoint',str(JOB/'train-single/final_checkpoint.pt'),'--multi-checkpoint',str(JOB/'train-multi/final_checkpoint.pt')]
 with (JOB/f'{phase}.log').open('x') as f:
  p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=4100 if phase=='evaluate' else 3300)
 write(f'{phase}-raw-exit.json',{'returncode':p.returncode})
 if p.returncode:raise RuntimeError(f'{phase}: nonzero native process exit {p.returncode}')
 report=read(phase)
 assert report['status']=='completed' and report['phase']==phase,(phase,report.get('status'))
 if phase=='preflight':
  assert report['strict_successes']==4 and len(report['episodes'])==4
  assert all(x['evaluation']['classification']=='valid_success' for x in report['episodes'])
 elif phase.startswith('train-'):
  assert report['learning_executed'] is True and report['updates']==64
  assert report['transitions']['total_transitions']==8192
  assert all(x['active_valid']>0 for x in report['transitions']['by_case'].values())
  assert report['checkpoint']['update']==64 and report['checkpoint']['eligible_for_evaluation'] is True
  assert report['diagnostic_checkpoint']['eligible_for_selection'] is False
 else:
  assert len(report['episodes'])==27
  assert all(x['evaluation']['classification'] in ('valid_success','valid_failure') for x in report['episodes'])
  assert report['effect_proven'] is False and report['checkpoint_selection_changed'] is False
  for label,train in [('single64','train-single'),('multi64','train-multi')]:
   actual=hashlib.sha256((JOB/train/'final_checkpoint.pt').read_bytes()).hexdigest()
   assert report['checkpoints'][label]['sha256']==read(train)['checkpoint']['sha256']==actual,label
 write('batch-progress.json',{'phase':phase,'state':'validated','utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 return report

def compare_training():
 a,b=read('train-single'),read('train-multi')
 checks={}
 for path in [('initialization','actor'),('initialization','critic'),('initialization','optimizer_sha256'),('initialization','zero_actor'),('initialization','rng'),('training_rng',),('actor_parameter_sha256_before',)]:
  x,y=a,b
  for key in path:x,y=x[key],y[key]
  checks['.'.join(path)]=x==y
 checks['first_active_anchor_base79']=a['transitions']['first_active_base79']['anchor']==b['transitions']['first_active_base79']['anchor']
 checks['same_transition_budget']=a['transitions']['total_transitions']==b['transitions']['total_transitions']==8192
 write('paired-training-audit.json',{'status':'pass' if all(checks.values()) else 'invalid','checks':checks})
 assert all(checks.values()),checks

start=time.monotonic()
write('batch-started.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
try:
 run('preflight')
 run('train-single')
 run('train-multi')
 compare_training()
 result=run('evaluate')
 write('batch-final.json',{'status':'completed','elapsed_s':time.monotonic()-start,'fresh8_successes':result['fresh8_successes'],'followup_multiseed_candidate':result['followup_multiseed_candidate'],'effect_proven':False})
except BaseException as e:
 traceback.print_exc()
 write('batch-final.json',{'status':'incomplete','elapsed_s':time.monotonic()-start,'error':{'type':type(e).__name__,'message':str(e)},'ranking_allowed':False})
 sys.exit(1)
