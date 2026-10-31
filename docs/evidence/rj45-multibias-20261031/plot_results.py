"""Draw the audited paired outcomes and the recorded training exposure."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from audit_results import audit

def save(fig,root,name):
 fig.savefig(root/(name+'.png'),dpi=180,facecolor='white')
 fig.savefig(root/(name+'.svg'),facecolor='white')
 p=root/(name+'.svg');p.write_text('\n'.join(x.rstrip() for x in p.read_text().splitlines())+'\n')
 plt.close(fig)

def main():
 root=Path(__file__).resolve().parent
 names=['evaluate-report.json','contract.json','train-multi-report.json']
 captured={name:(root/name).read_bytes() for name in names}
 checked=audit(root)
 for name,raw in captured.items():
  if (root/name).read_bytes()!=raw:raise ValueError('audited plot input changed: '+name)
 report=json.loads(captured['evaluate-report.json'])
 contract=json.loads(captured['contract.json'])
 font=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
 if font.is_file():
  font_manager.fontManager.addfont(str(font));plt.rcParams['font.family']=font_manager.FontProperties(fname=font).get_name()
 plt.rcParams['axes.unicode_minus']=False
 methods=contract['evaluation']['methods'];cases=contract['evaluation']['cases']
 labels={'single64':'단일 조건 PPO','multi64':'다중 조건 PPO','zero':'무보정 기준'}
 lookup={(x['case']['case_id'],x['method']):x for x in report['episodes']}
 success=checked['evaluation']['fresh8_successes']
 fig,ax=plt.subplots(figsize=(10.5,9),layout='constrained')
 data=[]
 for row,case in enumerate(cases):
  for col,method in enumerate(methods):
   e=lookup[case['case_id'],method];terminal=e['evaluation']['terminal'];ok=terminal['success'];failed=[k for k,v in terminal['criteria'].items() if not v]
   reason={'depth':'깊이','lateral':'정렬','orientation':'방향','force':'하중','forbidden_contact':'금지 접촉','grasp_live_and_proven':'파지','dwell':'안정 유지','independent_evaluator_terminal':'독립 판정'}
   short='성공' if ok else '실패 · '+'·'.join(reason[k] for k in failed if k not in ['dwell','independent_evaluator_terminal'])
   if short=='실패 · ':short='실패 · 안정 유지'
   error=terminal['metrics']['lateral_error_m']*1e6
   ax.add_patch(Rectangle((col,row),1,1,facecolor='#e5f4eb' if ok else '#f9e9e7',edgecolor='white',linewidth=3))
   ax.text(col+.5,row+.39,short,ha='center',va='center',fontsize=11,weight='bold',color='#16613d' if ok else '#a53b30')
   ax.text(col+.5,row+.72,f'{error:.2f} µm',ha='center',va='center',fontsize=10,color='#344054')
   data.append({'case':case['case_id'],'method':method,'success':ok,'terminal_lateral_error_um':error,'failed_criteria':failed,'trace_sha256':e['evaluation']['trace']['sha256']})
 ax.axhline(1,color='#667085',linewidth=1.4)
 ax.set_xlim(0,3);ax.set_ylim(9,0)
 ax.set_xticks([x+.5 for x in range(3)],[f'{labels[m]}\n새 8조건 {success[m]}/8' for m in methods],fontsize=12);ax.xaxis.tick_top()
 ax.set_yticks([x+.5 for x in range(9)],['학습 anchor (별도)']+[f'새 보정 오차 {i}' for i in range(1,9)],fontsize=10)
 ax.tick_params(length=0,pad=10)
 for spine in ax.spines.values():spine.set_visible(False)
 fig.suptitle('학습 조건을 넓혔을 때의 삽입 성공 비교',fontsize=17,weight='bold')
 fig.supxlabel('각 PPO 64 updates · 8,192 transitions · 동일 train 파지와 학습 seed 1개\n숫자는 종료 횡오차이며 성공은 깊이·방향·파지·하중·안정 유지까지 함께 판정',fontsize=10,color='#475467')
 save(fig,root,'comparison')
 multi=json.loads(captured['train-multi-report.json'])
 counts=multi['transitions']['by_case'];names=list(counts)
 fig,ax=plt.subplots(figsize=(8.8,4.2),layout='constrained')
 totals=[counts[x]['transitions'] for x in names];active=[counts[x]['active_valid'] for x in names]
 ax.bar(names,totals,color='#dbe7f4',label='전체 전이')
 ax.bar(names,active,color='#426fa6',label='활성·유효 정책 입력 전이')
 for index,(t,a) in enumerate(zip(totals,active,strict=True)):ax.text(index,t+35,f'{a:,} / {t:,}',ha='center',fontsize=10)
 ax.set_ylabel('전이 수');ax.set_ylim(0,max(totals)*1.2);ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False)
 ax.set_title('다중 조건 학습에서 실제로 수집한 데이터',fontsize=15,weight='bold')
 fig.supxlabel('조건은 episode마다 순환한다. 마지막 episode 도중 전이 예산이 끝나 조건별 전이 수가 다르다.',fontsize=10,color='#475467')
 save(fig,root,'training-exposure')
 (root/'comparison-data.json').write_text(json.dumps({'evaluation':data,'training_exposure':counts,'fresh8_successes':success,'paired':checked['evaluation']['paired'],'effect_proven':False},indent=2,sort_keys=True)+'\n')
if __name__=='__main__':main()
