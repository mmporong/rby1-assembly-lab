"""Derive action-scale and episode-length diagnostics from audited recordings."""
from __future__ import annotations
import gzip,hashlib,json,statistics
from pathlib import Path
from audit_results import audit

def observed_training(path: Path, translation_bound: float, rotation_bound: float, expected_sha256: str | None = None) -> dict:
 raw=path.read_bytes()
 if expected_sha256 is not None and hashlib.sha256(raw).hexdigest()!=expected_sha256:raise ValueError("audited transition bytes changed")
 rows=[json.loads(line) for line in gzip.decompress(raw).decode().splitlines()]
 active=[r for r in rows if r['active'] and r['valid_input']]
 actions=[r['applied_expanded_action'][6:] for r in active]
 if not actions:raise ValueError('no active action records')
 if any(any(x!=0 for x in r['applied_expanded_action'][:6]) for r in rows):raise ValueError('left-arm residual changed')
 axes=[]
 for axis in range(6):
  values=[r[axis] for r in actions];bound=translation_bound if axis<3 else rotation_bound
  axes.append({'axis':axis,'normalized_mean':statistics.fmean(values),'normalized_population_std':statistics.pstdev(values),'physical_mean':statistics.fmean(values)*bound,'physical_population_std':statistics.pstdev(values)*bound,'physical_unit':'m' if axis<3 else 'rad'})
 lengths=[];last=0;terminal_rewards=[]
 for row in rows:
  if row['done']:
   lengths.append(row['transition']-last);last=row['transition'];terminal_rewards.append(row['reward'])
 return {'rows':len(rows),'active_valid_rows':len(active),'applied_action_axes':axes,'completed_episode_transition_lengths':lengths,'terminal_step_rewards':terminal_rewards,'unfinished_tail_transitions':len(rows)-last,'measured_action_std_is_not_pure_exploration_std':True,'training_success_flags_not_persisted_in_transition_log':True}

def main():
 root=Path(__file__).resolve().parent
 checked=audit(root)
 origin_bytes=(root/'origin-protocol.json').read_bytes()
 if hashlib.sha256(origin_bytes).hexdigest()!=checked['source_lineage']['source_binding']['origin_protocol']['sha256']:raise ValueError('bound origin protocol changed')
 origin=json.loads(origin_bytes)
 configs=root/'frozen-configs';task_path=configs/'rj45-v1.json';training_path=configs/'rj45-training-target-offset-v1.json'
 parsed={}
 for p,key in [(task_path,'task_config_sha256'),(training_path,'training_config_sha256'),(configs/'runtime.json','runtime_config_sha256')]:
  raw=p.read_bytes()
  if hashlib.sha256(raw).hexdigest()!=origin['frozen_inputs'][key]:raise ValueError('frozen config changed: '+str(p))
  parsed[p.name]=json.loads(raw)
 task=parsed[task_path.name];training=parsed[training_path.name]
 translation=float(task['controller']['residual_translation_bound_m']);rotation=float(task['controller']['residual_rotation_bound_rad']);sigma=float(origin['ppo']['initial_noise_std']);tolerance=float(task['evaluation']['max_lateral_error_m'])
 result={'schema':'rj45_multibias_training_diagnostics_v1','nominal_initial_gaussian_sigma_normalized':sigma,'absolute_tip_pose_offset_not_integrated_delta':True,'translation_bound_m':translation,'rotation_bound_rad':rotation,'nominal_unclipped_translation_sigma_m':sigma*translation,'nominal_unclipped_rotation_sigma_rad':sigma*rotation,'strict_lateral_tolerance_m':tolerance,'strict_lateral_tolerance_provenance':'project-defined evaluation criterion; not a manufacturer specification','nominal_translation_sigma_over_lateral_tolerance':sigma*translation/tolerance,'dense_lateral_scale_m':training['reward']['lateral_scale_m'],'dense_lateral_scale_over_strict_tolerance':float(training['reward']['lateral_scale_m'])/tolerance,'dense_progress_pose_source':'GT relative object pose; calibration bias is applied to estimator/control, not the reward potential','reward_parameters':training['reward'],'training':{}}
 for phase,label in [('train-single','single64'),('train-multi','multi64')]:
  expected=checked['training'][label]['transitions']['integrity']['compressed_sha256']
  result['training'][phase]=observed_training(root/phase/'transitions.jsonl.gz',translation,rotation,expected)
 result['interpretation']={'estimator_reward_coordinate_bug_confirmed':False,'exploration_scale_as_cause_proven':False,'training_distribution_effect_proven':False,'next_test_hypothesis':'compare initial exploration scale while holding initialization, conditions, reward, action bounds, data budget and final checkpoint selection fixed','no_hyperparameter_changed_in_this_run':True,'nominal_sigma_is_a_command_distribution_scale_not_measured_end_effector_error':True}
 (root/'training-diagnostics.json').write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
 print(json.dumps({'action_sigma_um':result['nominal_unclipped_translation_sigma_m']*1e6,'strict_tolerance_um':tolerance*1e6,'causal_claim':False}))
if __name__=='__main__':main()
