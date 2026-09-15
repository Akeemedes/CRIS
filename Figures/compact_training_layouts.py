"""Print-size training-history and local-error figures.

The data assembly is shared with ``build_training_supplement``. These functions
return the assembled figure and its input checksums for export by the caller.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np

import build_training_supplement as original


ROOT = Path(__file__).resolve().parents[1]


def _capture(composer):
    figures = []
    old_save = original.save
    old_sources = set(original.SOURCES)

    def capture(fig, folder, name):
        figures.append((fig, name))

    original.save = capture
    original.SOURCES.clear()
    try:
        composer()
        if len(figures) != 1:
            raise RuntimeError("Expected one figure from the saved-data composer")
        sources = set(original.SOURCES)
        sources.update((Path(original.__file__).resolve(), Path(__file__).resolve(),
                        ROOT / "Figures/build_main_results.py",
                        ROOT / "Figures/artifact_paths.py"))
    finally:
        original.save = old_save
        original.SOURCES.clear()
        original.SOURCES.update(old_sources)
    fig, canonical_name = figures[0]
    fig.canvas.draw()
    return fig, canonical_name, sources


def _compact_type(fig):
    """Set native print typography without changing the shared style module."""
    for text in fig.findobj(Text):
        text.set_fontsize(10)
    for ax in fig.axes:
        ax.tick_params(axis="both", which="both", labelsize=9.5, pad=2.5)
        for label in (ax.xaxis.label, ax.yaxis.label):
            label.set_fontsize(10)
        ax.xaxis.labelpad = 3
        ax.yaxis.labelpad = 3
        for title in (ax.title, ax._left_title, ax._right_title):
            title.set_fontsize(10.5)
        for text in ax.texts:
            if len(text.get_text()) == 1 and text.get_text() in "abcdefghij":
                text.set_fontsize(10.5)
    for legend in fig.legends:
        for text in legend.get_texts():
            text.set_fontsize(9.5)


def _finish(fig, canonical_name, sources):
    evidence = {
        "canonical_name": canonical_name,
        "size_mm": (fig.get_size_inches() * 25.4).round(6).tolist(),
        "axes_count": len(fig.axes),
        "line_count": sum(len(ax.lines) for ax in fig.axes),
        "patch_count": sum(len(ax.patches) for ax in fig.axes),
        "native_font_sizes_pt": {"ticks": 9.5, "axis_labels": 10,
                                  "titles": 10.5, "legends": 9.5},
        "sources": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(sources)},
    }
    return fig, evidence


def build_histories():
    """Return the ten-panel training plate, compacted to 183 x 208 mm."""
    fig, name, sources = _capture(original.combined)
    fig.set_size_inches(183 / 25.4, 208 / 25.4, forward=True)
    _compact_type(fig)
    # Distinguish constitutive laws from the loss used during optimization.
    fig.axes[0].set_title("Quadratic exponents", loc="left", pad=7, fontsize=10.5)
    fig.axes[2].set_title("Singular endpoint", loc="left", pad=7, fontsize=10.5)
    fig.axes[7].set_title("Quadratic-exponent optimization", loc="left", pad=7,
                          fontsize=10.5)
    fig.get_layout_engine().set(h_pad=.018, w_pad=.035, hspace=.018, wspace=.025)
    return _finish(fig, name, sources)


def build_distributions():
    """Return the six-panel error-distribution figure at 183 x 118 mm."""
    fig, name, sources = _capture(original.distributions)
    fig.set_size_inches(183 / 25.4, 118 / 25.4, forward=True)
    _compact_type(fig)
    for ax in fig.axes:
        ax.set_title(ax.get_title(loc="left").replace("\n", " "), loc="left",
                     pad=7, fontsize=10.5)
    fig.get_layout_engine().set(h_pad=.025, w_pad=.035, hspace=.025, wspace=.035)
    return _finish(fig, name, sources)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    args.preview_dir.mkdir(parents=True, exist_ok=True)
    for name, builder in (("training-histories-compact", build_histories),
                          ("regular-distributions-compact", build_distributions)):
        figure, evidence = builder()
        figure.savefig(args.preview_dir / (name + ".png"), dpi=180)
        print(json.dumps({"name": name, **evidence}, indent=2))
        plt.close(figure)
