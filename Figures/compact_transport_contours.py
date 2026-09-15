"""Plot transport residual contours with the supplied Newton trajectories.

``build()`` returns a Matplotlib figure and source metadata for a 160 x 81 mm
layout. The background evaluates each residual on a common two-cell grid.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import numpy as np
import two_cell_contours as residuals


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAJECTORIES = HERE / "figS2_transport_geometry/two_cell_residual_contours.json"
ORIGINAL_BUILDER = HERE / "two_cell_contours.py"
WIDTH_MM = 160.0
HEIGHT_MM = 81.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_residual_definitions():
    """Use the same residual definitions as the two-cell simulation driver."""
    return residuals


def _draw_saved_paths(ax, records, colors):
    """Draw the saved iteration path, terminal root and direction arrows."""
    limits = (0.0, 1.5)
    for index, record in enumerate(records):
        path = np.asarray(record["trajectory"], dtype=float)
        start = np.asarray(record["start"], dtype=float)
        if path.ndim != 2 or path.shape[1] != 2 or not np.array_equal(path[0], start):
            raise ValueError("Trajectory/start mismatch")
        color = colors[index]
        ax.plot(path[:, 0], path[:, 1], color=color, lw=1.15, zorder=4)
        ax.plot(*path[0], "o", color=color, mec="white", mew=0.65,
                ms=4.5, zorder=6)
        if record["status"] == "converged":
            admissible = bool(np.all((path[-1] >= 0) & (path[-1] <= 1)))
            classification = "physical" if admissible else "nonphysical"
            if record["terminal_class"] != classification:
                raise ValueError("Terminal classification mismatch")
            ax.plot(*path[-1], marker="*" if admissible else "D",
                    color="#009E73" if admissible else "#D55E00",
                    mec="black", mew=0.6, ms=10.5 if admissible else 6.25,
                    zorder=7)
        for first, second in zip(path[:-1], path[1:]):
            # Add arrows to visible steps that are long enough to show direction.
            if (np.linalg.norm(second - first) > 0.025
                    and np.all((first >= limits[0]) & (first <= limits[1]))
                    and np.all((second >= limits[0]) & (second <= limits[1]))):
                ax.annotate(
                    "", xy=first + 0.70 * (second - first),
                    xytext=first + 0.40 * (second - first),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1,
                                    mutation_scale=9), zorder=5,
                )


def build():
    """Return the figure and metadata; the caller controls export paths."""
    source_bytes = TRAJECTORIES.read_bytes()
    retained = json.loads(source_bytes)
    if retained["kind"] != "transport" or len(retained["trajectories"]) != 10:
        raise ValueError("Expected two methods with five initial states each")
    original = _load_residual_definitions()
    physical, learned, model = original.transport_functions()
    if _sha256(model) != retained["model_sha256"]:
        raise ValueError("Residual-grid network differs from the trajectory model")

    methods = ["SRDM + line search", r"Learned CRIS: $\lambda=0.01$"]
    records = [
        [record for record in retained["trajectories"] if record["method"] == method]
        for method in methods
    ]
    starts = [[record["start"] for record in group] for group in records]
    if any(len(group) != 5 for group in records) or starts[0] != starts[1]:
        raise ValueError("Expected five identical starts for both methods")

    grid = np.linspace(0.0, 1.5, 301)
    xx, yy = np.meshgrid(grid, grid)
    grid_hashes = {}
    normalizers = {}
    style = {
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.labelsize": 10, "axes.titlesize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9, "pdf.fonttype": 42, "svg.fonttype": "none",
        "axes.linewidth": 0.7, "axes.spines.top": True,
        "axes.spines.right": True,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
        # Physical panel size is exactly 52 by 52 mm on both axes.
        axes = [fig.add_axes([left / WIDTH_MM, 11 / HEIGHT_MM,
                              52 / WIDTH_MM, 52 / HEIGHT_MM])
                for left in (13.6, 77.6)]
        cmap = LinearSegmentedColormap.from_list(
            "compact_residual_landscape",
            plt.get_cmap("Greys")(np.linspace(0.02, 0.9, 256)),
        )
        images = []
        for ax, residual, method, group in zip(
                axes, (physical, learned), methods, records):
            values = np.array([
                np.linalg.norm(residual(np.array([first, second]))[0])
                for first, second in zip(xx.ravel(), yy.ravel())
            ]).reshape(xx.shape)
            base = np.linalg.norm(residual(np.asarray(group[0]["start"]))[0])
            z = np.log10(np.maximum(values / base, 1e-8))
            normalizers[method] = float(base)
            grid_hashes[method] = hashlib.sha256(z.tobytes()).hexdigest()
            filled = ax.contourf(
                xx, yy, z, levels=np.linspace(-4, 2, 61), cmap=cmap,
                extend="both", antialiased=False, zorder=0,
            )
            # Rasterize only the filled background. This avoids contour seams
            # while retaining vector contour lines, paths, labels and markers.
            ax.set_rasterization_zorder(1)
            images.append(filled)
            ax.contour(
                xx, yy, z, levels=np.arange(-4, 2.01, 0.25),
                colors="#555555", linewidths=0.65,
                negative_linestyles="solid",
            )
            _draw_saved_paths(ax, group, original.COLORS)
            ax.set(xlim=(0, 1.5), ylim=(0, 1.5), xlabel="$u_1$",
                   ylabel="$u_2$", aspect="equal")
            ax.set_title(method, fontsize=10, pad=5)
            ax.set_xticks([0, 0.5, 1, 1.5])
            ax.set_yticks([0, 0.5, 1, 1.5])
            ax.tick_params(labelsize=9, length=3, pad=2)
            ax.xaxis.labelpad = 2
            ax.yaxis.labelpad = 2
        axes[1].set_ylabel("")
        colorbar = fig.colorbar(
            images[0], cax=fig.add_axes([135.2 / WIDTH_MM, 16 / HEIGHT_MM,
                                       2.2 / WIDTH_MM, 42 / HEIGHT_MM]),
            orientation="vertical", ticks=[-4, -2, 0, 2],
        )
        colorbar.ax.tick_params(labelsize=9, length=3, pad=2)
        colorbar.set_label("Log normalized residual", fontsize=10, labelpad=4)
        handles = [
            Line2D([], [], marker="o", color="none", mfc="#555555",
                   ms=4.5, label="Initial state"),
            Line2D([], [], marker="*", ls="none", mfc="#009E73", mec="black",
                   ms=10.5, label="Physical root"),
            Line2D([], [], marker="D", ls="none", mfc="#D55E00", mec="black",
                   ms=6.25, label="Nonphysical root"),
        ]
        fig.legend(handles=handles, loc="upper center",
                   bbox_to_anchor=(0.50, 0.985), ncol=3, frameon=False,
                   fontsize=9, columnspacing=1.05, handletextpad=0.35,
                   borderaxespad=0.25)

    if TRAJECTORIES.read_bytes() != source_bytes:
        raise RuntimeError("Trajectory input changed during the figure build")
    evidence = {key: copy.deepcopy(retained[key]) for key in ('kind', 'normalization', 'model_sha256')}
    evidence.update({
        "layout_mm": [WIDTH_MM, HEIGHT_MM],
        "panel_mm": [52.0, 52.0],
        "grid_shape": [301, 301],
        "grid_domain": [0.0, 1.5],
        "filled_levels": [-4.0, 2.0, 61],
        "line_level_step": 0.25,
        "normalizers": normalizers,
        "residual_grid_sha256": grid_hashes,
        "trajectory_source": TRAJECTORIES.relative_to(ROOT).as_posix(),
        "sources": {
            path.relative_to(ROOT).as_posix(): _sha256(path)
            for path in (TRAJECTORIES, ORIGINAL_BUILDER, model, Path(__file__))
        },
    })
    return fig, evidence
