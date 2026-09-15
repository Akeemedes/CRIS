"""Recreate the print-size figure layouts used in the manuscript.

Run ``python Figures/build_manuscript_compact.py`` to export all layouts, or
select individual figures with ``--only``. Inputs are supplied numerical results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.mplconfig'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text

SPECS = {
    'transport-inputs': ('compact_methods_layouts', 'build_embedding'),
    'transport-contours': ('compact_transport_contours', 'build'),
    'timestep-work': ('compact_methods_layouts', 'build_timestep'),
    'linear-budget': ('compact_methods_layouts', 'build_budget'),
    'training-histories': ('compact_training_layouts', 'build_histories'),
    'bratu-continuation': ('compact_methods_layouts', 'build_continuation'),
    'regular-distributions': ('compact_training_layouts', 'build_distributions'),
    'bratu-3d-accuracy': ('compact_methods_layouts', 'build_bratu_accuracy'),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_bounds(fig):
    """Check labels, legends and displayed tick text against the native canvas."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    artists = list(fig.texts) + list(fig.legends)
    for ax in fig.axes:
        artists += list(ax.texts)
        if not ax.axison:
            continue
        artists += [ax.xaxis.label, ax.yaxis.label, ax.title, ax._left_title, ax._right_title]
        if ax.get_legend() is not None:
            artists.append(ax.get_legend())
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lower, upper = sorted(limits)
            for location, label in zip(axis.get_majorticklocs(), axis.get_majorticklabels()):
                if lower - 1e-12 <= location <= upper + 1e-12:
                    artists.append(label)
    for artist in artists:
        if not artist.get_visible() or (isinstance(artist, Text) and not artist.get_text()):
            continue
        box = artist.get_window_extent(renderer)
        assert box.x0 >= -2 and box.y0 >= -2 and box.x1 <= bounds.width + 2 and box.y1 <= bounds.height + 2, (
            getattr(artist, 'get_text', lambda: type(artist).__name__)(), box.bounds, bounds.bounds)


def main():
    import importlib
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', choices=list(SPECS), default=list(SPECS))
    args = parser.parse_args()
    destination = HERE / 'manuscript_compact'
    destination.mkdir(exist_ok=True)
    for name in args.only:
        module_name, function_name = SPECS[name]
        module = importlib.import_module(module_name)
        with plt.rc_context({
            'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.labelsize': 10,
            'axes.titlesize': 10.5, 'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
            'legend.fontsize': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
            'axes.linewidth': .7, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
            'savefig.dpi': 300,
        }):
            fig, evidence = getattr(module, function_name)()
            check_bounds(fig)
            # Keep the intended physical canvas; no bbox_inches='tight' rescaling.
            for extension in ('pdf', 'png', 'svg'):
                fig.savefig(destination / f'{name}.{extension}', dpi=300)
            evidence.update(
                size_mm=(fig.get_size_inches() * 25.4).round(3).tolist(),
                builder=Path(module.__file__).relative_to(ROOT).as_posix(),
                builder_sha256=digest(Path(module.__file__)),
                export_builder_sha256=digest(Path(__file__)),
                files={f'{name}.{extension}': digest(destination / f'{name}.{extension}')
                       for extension in ('pdf', 'png', 'svg')},
            )
            (destination / f'{name}_manifest.json').write_text(
                json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
            print(f'{name}: {evidence["size_mm"]} mm')
            plt.close(fig)


if __name__ == '__main__':
    main()
