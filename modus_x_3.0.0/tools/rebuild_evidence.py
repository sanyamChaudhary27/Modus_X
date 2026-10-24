"""Rebuild tables, statistics and figures from the packaged result records."""
from pathlib import Path
import json
import statistics
PACKAGE = Path(__file__).resolve().parents[1]

def figures_and_tables():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.bbox': 'tight'})
    paper = PACKAGE / 'paper'
    figures = paper / 'figures'; figures.mkdir(exist_ok=True)
    result = json.loads((PACKAGE / 'research/v3/05_local_kv_filtering/RESULT.json').read_text())
    rows = result['seeds']
    gains = [r['validation_control'] - r['validation_candidate'] for r in rows]
    test_gains = [r['test_control'] - r['test_candidate'] for r in rows]
    stats = {'validation_gains': gains, 'mean_validation_gain': statistics.mean(gains),
             'median_validation_gain': statistics.median(gains),
             'sample_stdev_validation_gain': statistics.stdev(gains),
             'mean_test_gain_report_only': statistics.mean(test_gains),
             'validation_wins': sum(g > 0 for g in gains),
             'mean_runtime_ratio': statistics.mean(r['runtime_ratio'] for r in rows),
             'relative_parameter_overhead_percent': 100 * 24576 / 47437768,
             'control_state_fp32_bytes': (2 * 512**2 + 512) * 12 * 4,
             'candidate_state_fp32_bytes': (2 * 512**2 + 512 + 4 * 512) * 12 * 4,
             'max_audit_reproduction_delta': max(abs(r[f'audit_validation_{k}'] - r[f'validation_{k}']) for r in rows for k in ('control', 'candidate')),
             'interpretation': 'Three seed-level differences, not independent token-level samples; no significance claim'}
    (PACKAGE / 'docs/AGGREGATE.json').write_text(json.dumps(stats, indent=2) + '\n')
    lines = ['\\begin{tabular}{crrrrr}', '\\toprule',
             'Seed & Control val. & MX3 val. & Gain & Control test & MX3 test \\\\', '\\midrule']
    for r, gain in zip(rows, gains):
        lines.append(f"{r['seed']} & {r['validation_control']:.6f} & {r['validation_candidate']:.6f} & {gain:.6f} & {r['test_control']:.6f} & {r['test_candidate']:.6f} \\\\")
    lines.extend(['\\bottomrule', '\\end{tabular}'])
    (paper / 'endpoint_table.tex').write_text('\n'.join(lines) + '\n')
    macros = (f"\\newcommand{{\\MeanGain}}{{{stats['mean_validation_gain']:.6f}}}\n"
              f"\\newcommand{{\\MedianGain}}{{{stats['median_validation_gain']:.6f}}}\n"
              f"\\newcommand{{\\MeanTestGain}}{{{stats['mean_test_gain_report_only']:.6f}}}\n"
              f"\\newcommand{{\\GainSD}}{{{stats['sample_stdev_validation_gain']:.6f}}}\n"
              f"\\newcommand{{\\RuntimeOverhead}}{{{100*(stats['mean_runtime_ratio']-1):.3f}}}\n")
    (paper / 'numbers.tex').write_text(macros)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), gridspec_kw={'width_ratios': [1.1, 1]})
    for r, color in zip(rows, ['#146b61', '#9b4b1f', '#4859a6']):
        axes[0].plot([0, 1], [r['validation_control'], r['validation_candidate']], 'o-', color=color, label=f"Seed {r['seed']}", lw=1.7)
    axes[0].set_xticks([0, 1], ['Matched control', 'MX3 + local K/V'])
    axes[0].set_ylabel('Dense validation BPC (lower is better)')
    axes[0].set_xlim(-.2, 1.2); axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis='y', alpha=.18)
    x = [1, 2, 3]
    axes[1].bar([v-.17 for v in x], [result['screen_validation_gains'][str(v)] for v in x], .32, color='#a9afb3', label='5k screen')
    axes[1].bar([v+.17 for v in x], gains, .32, color='#146b61', label='25k endpoint')
    axes[1].axhline(0, color='black', lw=.6)
    axes[1].set_xticks(x, ['Seed 1', 'Seed 2', 'Seed 3'])
    axes[1].set_ylabel('Control minus candidate BPC')
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=2.0); fig.savefig(figures / 'endpoints.pdf'); fig.savefig(figures / 'endpoints.png', dpi=180); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 2.0))
    labels = ['Seed 2: 500 updates', 'Seed 1: 2,000 updates']
    y = [0, 1]
    ax.barh([v-.16 for v in y], [.8968588375, .8970010105], .3, color='#a9afb3', label='Canonical token loop')
    ax.barh([v+.16 for v in y], [.306832406, .306875592], .3, color='#146b61', label='Two-stage chunk32')
    ax.set_yticks(y, labels); ax.invert_yaxis(); ax.set_xlabel('Median update time (s); TPU v5e-8, 47M, batch 8, length 512')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(.5, 1.24), ncol=2, fontsize=8)
    ax.set_xlim(0, 1.05); ax.grid(axis='x', alpha=.15)
    fig.tight_layout(); fig.savefig(figures / 'execution.pdf'); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.8)); ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')
    def box(x, y, w, h, text, color):
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor='#475057', lw=.75))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=6.8)
    def arrow(a, b, color='#475057'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=10, lw=1, color=color))
    box(.1, 2.25, 1.35, .7, 'Input x\nnormalize', '#f4f5f5')
    box(1.95, 3.35, 1.8, .85, 'Project K and V\ncausal two-lag filter', '#e7efec')
    box(4.25, 3.1, 2.05, 1.35, 'Current C / archive A\nsequential delta writes\nthen query reads\narchive scale 512', '#e7efec')
    box(4.25, .65, 2.05, .9, 'Low-rank feedback\nadd gated read to x\nthen normalize', '#edf0f7')
    box(6.8, .65, 1.65, .9, 'Vector recurrence\nupdate state s', '#edf0f7')
    box(6.8, 3.35, 1.65, .85, 'Direct matrix\noutput projection', '#e7efec')
    box(8.9, 2.25, 1.0, .7, 'Mix\nresidual', '#f4f5f5')
    arrow((1.45, 2.8), (1.95, 3.75)); arrow((3.75, 3.75), (4.25, 3.75))
    arrow((5.28, 3.1), (5.28, 1.55)); ax.text(5.45, 2.4, 'retrieved\ncontext c', fontsize=7, va='center')
    arrow((6.3, 1.1), (6.8, 1.1)); arrow((6.3, 3.75), (6.8, 3.75))
    arrow((8.45, 3.75), (9.4, 2.95)); arrow((8.45, 1.1), (9.4, 2.25))
    ax.text(.1, .08, 'State: C, A, vector s, two prior projected keys and values.\nQ, gates and router use the layer input. Matrix writes do not use the within-layer vector state.', fontsize=7.2)
    ax.text(.1, 4.75, 'Modus_X 3: bounded state with memory-conditioned recurrence', fontsize=9, fontweight='bold')
    fig.tight_layout(); fig.savefig(figures / 'architecture.pdf'); fig.savefig(figures / 'architecture.png', dpi=200); plt.close(fig)

if __name__ == '__main__': figures_and_tables()
