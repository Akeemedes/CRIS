"""Plot total error and inverse-error crossover for the shock refinement study."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT/'training/twophasetransport/analysis/shock_refinement/errors.csv'
DEFAULT_OUTPUT = ROOT/'Figures/fig6_refinement/shock_refinement'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT,
                        help='Output stem, without an extension')
    parser.add_argument('--formats', nargs='+', choices=['png','svg','pdf'], default=['pdf','png','svg'])
    parser.add_argument('--time', type=float, default=200)
    args = parser.parse_args()
    with args.data.open(newline='', encoding='utf-8-sig') as stream:
        rows = [r for r in csv.DictReader(stream) if float(r['time']) == args.time]
    if not rows:
        raise ValueError('No records at the requested time')
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'axes.labelsize':9, 'xtick.labelsize':8, 'ytick.labelsize':8,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none', 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), layout='constrained')
    colors = {1:'#777777', 10:'#0072B2', 100:'#D55E00', 1000:'#009E73'}
    for beta in sorted({float(r['beta']) for r in rows}):
        color = colors.get(beta, '#CC79A7')
        pairs = {}
        for method in ('SRDM','CRIS'):
            selected = sorted((r for r in rows if float(r['beta']) == beta and r['method'] == method),
                              key=lambda r:int(r['N']))
            pairs[method] = selected
            axes[0].loglog([int(r['N']) for r in selected], [float(r['total_L2']) for r in selected],
                           color=color, ls='--' if method == 'SRDM' else '-', lw=1.35,
                           marker=None if method == 'SRDM' else 'o', ms=3.1)
        if [r['N'] for r in pairs['CRIS']] != [r['N'] for r in pairs['SRDM']]:
            raise ValueError('Unpaired mesh results')
        selected = pairs['CRIS']
        axes[1].loglog([int(r['N']) for r in selected],
                       [float(r['vs_discrete_over_discretization_L2']) for r in selected],
                       '-o', color=color, lw=1.35, ms=3.1, label=rf'$\Delta t/h={beta:g}$')
    axes[0].set(ylabel='RMSE against analytic solution')
    axes[1].set(ylabel=r'$E_{\rm inv}/E_{\rm disc}$')
    axes[1].axhline(1, color='.25', ls=':', lw=1, zorder=0)
    axes[0].legend(handles=[Line2D([],[],color='.25',ls='--',label=r'SRDM ($E_{\rm disc}$)'),
                            Line2D([],[],color='.25',marker='o',ms=3.1,label=r'CRIS ($E_{\rm tot}$)')],
                   frameon=False, fontsize=8, loc='lower left', handlelength=2)
    axes[1].legend(frameon=False, fontsize=7.5, loc='lower right', handlelength=1.6,
                   borderaxespad=.1, labelspacing=.28)
    for label, ax in zip(('a  Total solution error','b  Inverse-error contribution'), axes):
        ax.set_title(label, loc='left', fontsize=10, pad=9)
        ax.set_xlabel('Number of cells, $N_h$')
        ax.grid(which='major', color='.9', lw=.5)
        ax.tick_params(which='both', direction='out')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in args.formats:
        fig.savefig(args.output.with_suffix('.'+extension), dpi=240,
                    metadata={'Creator':'CRIS mesh-refinement figure builder'} if extension != 'svg' else None)
    plt.close(fig)
    print(f'Plotted {len(rows)//2} paired meshes at T={args.time:g}: {args.output}')


if __name__ == '__main__':
    main()
