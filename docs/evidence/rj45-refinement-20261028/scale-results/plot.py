from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
benchmark = json.loads((root / 'benchmark-summary.json').read_text())
refinement = json.loads((root / 'refinement-summary.json').read_text())
followup = json.loads((root / 'f-followup-summary.json').read_text())
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout='constrained')
fig.set_facecolor('white')
for ax in axes:
    ax.set_facecolor('white')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax.set_axisbelow(True)
sizes = [64, 128, 256]
rates = [benchmark['candidates'][str(n)]['throughput_transitions_per_s'] for n in sizes]
bars = axes[0].bar([str(n) for n in sizes], rates, color=['#a5b4fc', '#818cf8', '#4f46e5'], width=0.6)
axes[0].bar_label(bars, fmt='%.1f', padding=4)
axes[0].set(ylim=(0, max(rates)*1.17), ylabel='Transitions / second', xlabel='Parallel environments', title='Measured collection + learning throughput')
axes[0].text(0.5, -0.25, '8,192 transitions/update; steps/env 128 / 64 / 32', transform=axes[0].transAxes, ha='center', fontsize=8, color='#475569')
profiles = ['Z1', 'F', 'N', 'R']
counts = [refinement['profiles'][p]['96']['successes'] for p in profiles]
bars = axes[1].bar(profiles, counts, color=['#94a3b8', '#4f46e5', '#a5b4fc', '#cbd5e1'], width=0.6)
axes[1].bar_label(bars, fmt='%d / 32', padding=4)
axes[1].axhline(7, color='#475569', linestyle='--', label='Paired baseline: 7 / 32')
axes[1].set(ylim=(0, 16), ylabel='Successful development cases / 32', title='96-update development comparison')
axes[1].legend(frameon=False, loc='upper right', fontsize=8)
axes[1].text(0.5, -0.25, 'One training seed/profile; F chosen by the fixed rule', transform=axes[1].transAxes, ha='center', fontsize=8, color='#475569')
for seed, color in zip(sorted(followup['seeds']), ['#4f46e5', '#0891b2', '#d97706']):
    values = [followup['seeds'][seed][str(k)]['successes'] for k in [20, 96, 200]]
    axes[2].plot([20, 96, 200], values, 'o-', label=seed, color=color)
axes[2].axhline(7, color='#475569', linestyle='--', label='Baseline')
axes[2].axhline(10, color='#b91c1c', linestyle=':', label='Required 200-update mean')
axes[2].set(xlim=(10, 215), ylim=(0, 16), xticks=[20,96,200], xlabel='Completed PPO updates', ylabel='Successful development cases / 32', title='F replication: 3 fresh training seeds')
axes[2].legend(frameon=False, fontsize=7, ncol=2, loc='upper left')
axes[2].text(0.5, -0.25, '200-update mean 8.33 / 32: promotion gate FAILED', transform=axes[2].transAxes, ha='center', fontsize=8, color='#b91c1c')
fig.suptitle('RJ45 parallel PPO experiments — development results, final holdout sealed', fontsize=14)
fig.savefig(root / 'scale-and-learning.png', dpi=180, facecolor='white')
fig.savefig(root / 'scale-and-learning.svg', facecolor='white')
svg_path = root / 'scale-and-learning.svg'
svg_path.write_text('\n'.join(line.rstrip() for line in svg_path.read_text().splitlines()) + '\n')
manifest = {
    'source_commit': '63f96e5951c7de4a4a5e804c0584c70721d05e8c',
    'inputs': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['benchmark-summary.json', 'refinement-summary.json', 'f-followup-summary.json', 'independent-audit.json']},
    'selection_note': 'refinement summary selection_performed=false; F derived by applying the preregistered rule after measurement',
    'promotion_passed': False,
    'final_holdout_opened': False,
    'outputs': {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['scale-and-learning.png','scale-and-learning.svg']},
}
(root/'figure-manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
print(root/'scale-and-learning.png')
