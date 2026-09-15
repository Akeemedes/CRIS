"""Print-size layouts for transport assembly, solver work and Bratu accuracy."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from matplotlib.text import Text

ROOT = Path(__file__).resolve().parents[1]
COLORS = ['#0072B2', '#B2182B', '#009E73', '#CC79A7']


def source(path):
    path = ROOT / path
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def records(path, evidence):
    path, digest = source(path)
    evidence['sources'][path.relative_to(ROOT).as_posix()] = digest
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def fonts(fig):
    """Apply native physical font sizes; the manuscript includes at native width."""
    for text in fig.findobj(Text):
        text.set_fontsize(min(text.get_fontsize(), 10.5))
    for ax in fig.axes:
        ax.tick_params(axis='both', which='both', labelsize=9.5)
        ax.xaxis.label.set_fontsize(10)
        ax.yaxis.label.set_fontsize(10)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(9.5)


def panel_title(ax, letter, text):
    ax.set_title(r'$\mathbf{' + letter + r'}$  ' + text, loc='left', pad=6, fontsize=10.5)


def build_embedding():
    fig = plt.figure(figsize=(165 / 25.4, 37 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    # Assembly, local inputs and inverse evaluation form the three stages.
    for x in (.02, .363, .706):
        ax.add_patch(FancyBboxPatch(
            (x, .27), .274, .69, boxstyle='round,pad=.005,rounding_size=.018',
            facecolor='#F3F6F8', edgecolor='#B5C4CE', lw=.8))
    for x, title in zip((.157, .5, .843),
                        ('Transport assembly', 'Three local inputs', 'Selected inverse')):
        ax.text(x, .84, title, ha='center', va='center', fontsize=10, fontweight='bold')
    ax.text(.157, .54, 'Fluxes and sources\nStorage\nPrevious state',
            ha='center', va='center', fontsize=9.5, linespacing=1.2)
    ax.text(.5, .62, r'$a_i=\dfrac{mA_i}{mB_i+\epsilon}$',
            ha='center', va='center', fontsize=10)
    ax.text(.5, .36, r'$\beta_i=mB_i\Delta t,\quad u_i^n$',
            ha='center', va='center', fontsize=10)
    ax.text(.843, .55, r'$\hat u_i=\widehat\Phi(a_i,\beta_i,u_i^n)$',
            ha='center', va='center', fontsize=10)
    for x in (.328, .671):
        ax.annotate('', xy=(x + .024, .59), xytext=(x - .024, .59),
                    arrowprops=dict(arrowstyle='->', lw=1.2, color=COLORS[0]))
    ax.text(.5, .10, r'$u_i+\beta_i[f(u_i)-a_i]=u_i^n$',
            ha='center', va='center', fontsize=10.5)
    path, digest = source('Figures/build_supplementary_methods.py')
    return fig, dict(sources={path.relative_to(ROOT).as_posix(): digest},
                     content='Transport assembly, normalized inputs and local inverse equation.')


def build_timestep():
    evidence = dict(sources={}, findings={})
    fig, grid = plt.subplots(2, 2, figsize=(183 / 25.4, 103 / 25.4))
    fig.subplots_adjust(left=.105, right=.98, bottom=.13, top=.85, wspace=.39, hspace=.70)
    axes = grid.ravel()
    base = Path('training/twophasetransport/analysis/regular_v3/paired_benchmarks/EDFM')
    for name, color in [('CRIS', COLORS[0]), ('SRDM', COLORS[1])]:
        path, digest = source(base / name / 'result.json')
        meta = json.loads(path.read_text())
        evidence['sources'][path.relative_to(ROOT).as_posix()] = digest
        assert meta['completed']
        report = base / name / 'output/solver_report.csv'
        rows = records(report, evidence)
        assert evidence['sources'][report.as_posix()] == meta['output_sha256']['output/solver_report.csv']
        good = np.array([int(row['converged']) == 1 for row in rows])
        dt = np.array([float(row['DT']) for row in rows])
        time = 0.
        start, end = [], []
        for delta, accepted in zip(dt, good):
            start.append(time)
            time += delta if accepted else 0
            end.append(time)
        config_path, config_digest = source(base / name / 'sim.txt')
        evidence['sources'][config_path.relative_to(ROOT).as_posix()] = config_digest
        config = dict(line.split(None, 1) for line in config_path.read_text().splitlines()
                      if len(line.split()) >= 2 and not line.lstrip().startswith('#'))
        final_time, initial_dt = float(config['T_END']), float(config['DT_INIT'])
        assert np.isclose(time, final_time)
        start, end = np.array(start) / final_time, np.array(end) / final_time
        axes[0].plot(start[good], dt[good] / initial_dt, 'o-', ms=3, lw=.8, color=color, label=name)
        axes[0].scatter(start[~good], dt[~good] / initial_dt, marker='x', s=20, color='#C98A00')
        for ax, key in zip(axes[1:3], ['NLNSTEPS', 'LINSTEPS']):
            values = np.array([float(row[key]) for row in rows])
            ax.step(np.r_[0, end], np.r_[np.nan, np.cumsum(values)],
                    where='post', color=color, label=name)
            if name == 'SRDM':
                ax.step(np.r_[0, end], np.r_[np.nan, np.cumsum(values * good)],
                        where='post', color=color, ls='--', label='SRDM accepted')
        discarded = []
        for key in ['NLNSTEPS', 'LINSTEPS', 'NFEVAL']:
            values = np.array([float(row[key]) for row in rows])
            discarded.append(100 * sum(values[~good]) / sum(values))
        evidence['findings'][name] = dict(accepted=int(sum(good)), rejected=int(sum(~good)),
                                          discarded_percent=discarded)
    axes[0].set(yscale='log', xlabel=r'$\hat t$',
                ylabel=r'$\Delta t/\Delta t_{\rm init}$')
    for ax, ylabel in zip(axes[1:3], ['Newton updates', 'Krylov iterations']):
        ax.set(yscale='log', xlabel=r'$\hat t$', ylabel=ylabel)
    y = np.arange(3)
    discarded = np.array(evidence['findings']['SRDM']['discarded_percent'])
    axes[3].barh(y, 100 - discarded, color=COLORS[1], label='Accepted')
    axes[3].barh(y, discarded, left=100 - discarded, color='#E69F00', label='Discarded')
    axes[3].set(yticks=y, yticklabels=['Newton', 'Krylov', 'Residual'],
                xlim=(0, 100), ylim=(-.6, 3.25), xlabel='SRDM work (%)')
    axes[3].spines['left'].set_bounds(-.4, 2.4)
    axes[3].legend(frameon=False, loc='upper left', bbox_to_anchor=(0, 1.04),
                   ncol=2, fontsize=9.5, handlelength=1, columnspacing=.9, borderaxespad=0)
    for j, value in enumerate(discarded):
        axes[3].text(53, j, f'{value:.1f}%', ha='center', va='center', fontsize=10)
    for ax, letter, title in zip(axes, 'abcd',
            ['Timestep attempts', 'Total nonlinear work', 'Total linear work', 'Cost of rejected steps']):
        panel_title(ax, letter, title)
    fig.legend(handles=[
        Line2D([], [], color=COLORS[0], label='CRIS'),
        Line2D([], [], color=COLORS[1], label='SRDM'),
        Line2D([], [], color=COLORS[1], ls='--', label='Accepted only'),
        Line2D([], [], color='#C98A00', marker='x', ls='none', label='Rejected')],
        loc='upper center', bbox_to_anchor=(.52, 1.0), ncol=4, frameon=False,
        fontsize=9.5, handlelength=1.5, columnspacing=1)
    fonts(fig)
    return fig, evidence


def build_budget():
    # Reuse the budget plot's data assembly with a print-size export canvas.
    import build_publication_completion as original
    captured = []
    previous_save = original.save
    original.SOURCES.clear()
    original.save = lambda fig, folder, name: captured.append(fig)
    try:
        original.solver_sensitivity()
    finally:
        original.save = previous_save
    assert len(captured) == 1
    fig = captured[0]
    fig.set_size_inches(183 / 25.4, 95 / 25.4)
    fonts(fig)
    fig.get_layout_engine().set(w_pad=4 / 72, h_pad=3 / 72, wspace=.08, hspace=.08)
    for ax, letter in zip(fig.axes, 'abcd'):
        for text in list(ax.texts):
            if text.get_text() == letter:
                text.remove()
        title = ax.get_title(loc='left')
        panel_title(ax, letter, title)
    evidence = dict(sources={path.relative_to(ROOT).as_posix():
                             hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sorted(original.SOURCES)})
    return fig, evidence


def build_continuation():
    evidence = dict(sources={})
    rows = records('Figures/fig3_bratu/data/continuation_validation.csv', evidence)
    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 65 / 25.4))
    fig.subplots_adjust(left=.105, right=.98, bottom=.235, top=.755, wspace=.58)
    for i, schedule in enumerate(['alpha1_only', 'admissible_curved']):
        selected = [row for row in rows if row['schedule'] == schedule]
        rho = sorted(set(float(row['rho']) for row in selected))
        x = 1 - np.array(rho)
        for ax, key, reduction in [(axes[0], 'abs_label_error', max),
                                    (axes[2], 'min_derivative', min)]:
            values = [reduction(float(row[key]) for row in selected if float(row['rho']) == v)
                      for v in rho]
            ax.loglog(x, np.maximum(values, 1e-16), color=COLORS[i], ls=['-', '--'][i],
                      label=[r'Fixed $\alpha_2$', r'Curved, $\rho<1$'][i])
    selected = [row for row in rows if row['schedule'] == 'alpha1_only']
    values = [max(float(row['path_disagreement']) for row in selected if float(row['rho']) == v)
              for v in rho]
    axes[1].loglog(x, np.maximum(values, 1e-16), color=COLORS[2])
    for ax, letter, title, ylabel in zip(axes, 'abc',
            ['Label error', 'Path difference', 'Regularity'],
            ['Maximum error', 'Maximum difference', r'Minimum $|r_u|$']):
        panel_title(ax, letter, title)
        ax.set(ylabel=ylabel, xticks=[1e-3, 1e-1, 1])
    fig.supxlabel(r'Fold deficit $1-\rho$', y=.018, fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               bbox_to_anchor=(.53, 1.025), ncol=2, frameon=False, fontsize=9.5)
    fonts(fig)
    evidence['content'] = 'Maximum label/path errors and minimum derivative by fold deficit; display floor 1e-16.'
    return fig, evidence


def build_bratu_accuracy():
    """Plot field RMSE over successful lower-branch initializations."""
    evidence = dict(sources={})
    rows = records('training/bratu/analysis/three_dimensional_initialization.csv', evidence)
    arms = ['mse', 'lambda_0p01', 'lambda_0p1', 'lambda_1']
    selected = [[row for row in rows if row['method'] == method
                 and row['lower_capture'] == 'True'] for method in arms]
    amplitudes = [[float(row['initial_amplitude']) for row in group] for group in selected]
    if not all(group and sorted(group) == sorted(amplitudes[0]) for group in amplitudes):
        raise ValueError('All models must be compared over the same captured initial amplitudes')
    values = np.array([[float(row['lower_rmse']) for row in group] for group in selected])
    centers = np.median(values, axis=1)
    minimum, maximum = values.min(axis=1), values.max(axis=1)
    with plt.rc_context({'font.size': 8.5, 'axes.labelsize': 9, 'axes.linewidth': .65}):
        fig, ax = plt.subplots(figsize=(86/25.4, 48/25.4))
        fig.subplots_adjust(left=.19, right=.98, bottom=.28, top=.92)
        ax.errorbar(np.arange(4), centers*1e5,
                    yerr=np.array([centers-minimum, maximum-centers])*1e5,
                    fmt='o', ms=4.5, color='#007E87', capsize=3, elinewidth=.8)
        ax.set_xticks(range(4), ['0', '0.01', '0.1', '1'])
        ax.set(xlabel=r'Residual weight $\lambda$', ylabel=r'Field RMSE ($10^{-5}$)',
               ylim=(0, 3.3), xlim=(-.35, 3.35))
        ax.set_yticks([0, 1, 2, 3])
        ax.spines[['top', 'right']].set_visible(False)
        ax.yaxis.grid(True, color='#DFE4E8', linewidth=.6)
        ax.set_axisbelow(True)
    evidence['findings'] = dict(methods=arms, captured_amplitudes=amplitudes[0],
                                median=centers.tolist(), minimum=minimum.tolist(),
                                maximum=maximum.tolist())
    return fig, evidence
