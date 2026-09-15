#!/usr/bin/env python3
"""Map final regular-law surrogate error over a controlled local-input slice.

This complements distributional test metrics with a deterministic topology:
for each fixed old saturation, every point in the (a, beta) grid is evaluated
against a monotone bracketed root of the same regular recurrence.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from evaluate_checkpoints import load_weights, network, sha256


def parse_model(value: str) -> tuple[str, Path]:
    label, separator, checkpoint = value.rpartition("=")
    if not separator or not label or not checkpoint:
        raise argparse.ArgumentTypeError("--model must be LABEL=CHECKPOINT")
    return label, Path(checkpoint)


def fractional_flow(u: np.ndarray) -> np.ndarray:
    return u * u / (u * u + 0.4 * (1.0 - u) * (1.0 - u))


def exact_root(a: np.ndarray, beta: np.ndarray, u_old: np.ndarray) -> np.ndarray:
    lo = np.zeros_like(a)
    hi = np.ones_like(a)
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        residual = mid - u_old + beta * (fractional_flow(mid) - a)
        lo = np.where(residual <= 0.0, mid, lo)
        hi = np.where(residual >= 0.0, mid, hi)
    return 0.5 * (lo + hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", type=parse_model, required=True,
                        help="LABEL=CHECKPOINT; two to four final models, in column order")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--beta-max", type=float, default=1.0e6)
    parser.add_argument("--beta-coordinate", choices=("raw", "log1p"), default="log1p")
    args = parser.parse_args()
    if not 2 <= len(args.model) <= 4:
        raise ValueError("This publication figure requires two to four models")
    if len({label for label, _ in args.model}) != len(args.model):
        raise ValueError("Model labels must be unique")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.labelsize": 13,
        "axes.titlesize": 13, "xtick.labelsize": 11, "ytick.labelsize": 11,
        "figure.dpi": 160, "savefig.dpi": 300, "pdf.fonttype": 42,
    })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Zero cannot be represented on a logarithmic axis. This figure samples
    # strictly positive beta; exact beta=0 anchors belong to separate tests.
    beta_axis = np.geomspace(1.0e-8, args.beta_max, 200)
    a_axis = np.linspace(0.0, 1.0, 181)
    beta_mesh, a_mesh = np.meshgrid(beta_axis, a_axis)
    old_values = (0.0, 0.5, 1.0)
    model_data: list[tuple[str, Path, np.ndarray]] = []
    for label, checkpoint in args.model:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model_data.append((label, checkpoint, load_weights(checkpoint)))

    figure, axes = plt.subplots(len(old_values), len(model_data), figsize=(7.2, 7.6),
                                sharex=True, sharey=True)
    # Reserve an explicit, independent colour-bar axis. Adjusting subplot
    # geometry after an automatic colour bar can otherwise overlap the maps.
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.22, top=0.945,
                           hspace=0.17, wspace=0.15)
    colour_axis = figure.add_axes([0.21, 0.085, 0.70, 0.022])
    figure.supxlabel(r"Local CFL coefficient $\beta$", x=0.55, y=0.15, fontsize=13)
    all_errors: list[np.ndarray] = []
    computed: list[tuple[float, str, Path, np.ndarray, np.ndarray]] = []
    summary: list[dict[str, object]] = []
    for u_old_value in old_values:
        u_old = np.full(beta_mesh.size, u_old_value)
        rows = np.column_stack((a_mesh.ravel(), beta_mesh.ravel(), u_old,
                                np.zeros(beta_mesh.size), np.zeros(beta_mesh.size),
                                np.zeros(beta_mesh.size)))
        root = exact_root(rows[:, 0], rows[:, 1], rows[:, 2])
        for label, checkpoint, weights in model_data:
            prediction = network(weights, rows, args.beta_max, args.beta_coordinate)
            if not np.all(np.isfinite(prediction)):
                raise ValueError(f"Nonfinite predictions for {label}, u_old={u_old_value}")
            error = np.abs(prediction - root).reshape(a_mesh.shape)
            all_errors.append(error)
            computed.append((u_old_value, label, checkpoint, error, prediction.reshape(a_mesh.shape)))
            summary.append({
                "model": label,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
                "u_old": u_old_value,
                "grid_points": int(error.size),
                "root_abs_error_rmse": float(np.sqrt(np.mean(error * error))),
                "root_abs_error_q99": float(np.quantile(error, 0.99)),
                "root_abs_error_max": float(np.max(error)),
                "out_of_bounds_fraction": float(np.mean((prediction < 0.0) | (prediction > 1.0) | ~np.isfinite(prediction))),
                "prediction_min": float(np.min(prediction)),
                "prediction_max": float(np.max(prediction)),
                "maximum_undershoot": float(np.max(np.maximum(-prediction, 0.0))),
                "maximum_overshoot": float(np.max(np.maximum(prediction - 1.0, 0.0))),
            })
    positive = np.concatenate([item[item > 0.0] for item in all_errors])
    # Full decades are readable, and the upper limit includes every error.
    # Values below the displayed floor use the colour-bar's lower extension.
    norm = colors.LogNorm(vmin=1.0e-9,
                          vmax=10.0 ** np.ceil(np.log10(np.max(positive))))
    image = None
    for u_old_value, label, _, error, prediction in computed:
        row_index = old_values.index(u_old_value)
        col_index = [item[0] for item in model_data].index(label)
        axis = axes[row_index, col_index]
        image = axis.pcolormesh(beta_axis, a_axis, np.maximum(error, np.finfo(float).tiny),
                                shading="auto", cmap="magma", norm=norm, rasterized=True)
        oob = (prediction < 0.0) | (prediction > 1.0) | ~np.isfinite(prediction)
        if np.any(oob) and not np.all(oob):
            axis.contour(beta_mesh, a_mesh, oob.astype(float), levels=(0.5,), colors="#00ffff", linewidths=0.75)
        axis.set_xscale("log")
        axis.set_xlim(beta_axis[0], beta_axis[-1])
        axis.set_ylim(0.0, 1.0)
        axis.set_xticks([1.0e-8, 1.0, args.beta_max])
        axis.tick_params(axis="x", labelsize=11)
        if row_index + 1 == len(old_values):
            # Keep endpoint labels inside their panel in dense column layouts.
            tick_labels = axis.get_xticklabels()
            tick_labels[0].set_horizontalalignment("left")
            tick_labels[-1].set_horizontalalignment("right")
        axis.grid(False)
        if row_index == 0:
            axis.set_title(label.replace('Residual ', ''))
        if col_index == 0:
            axis.set_ylabel(fr"$a$  ($u_{{old}}={u_old_value:g}$)")
    colour_bar = figure.colorbar(image, cax=colour_axis, extend="min", orientation="horizontal",
                    ticks=10.0 ** np.arange(-9, int(np.log10(norm.vmax)) + 1),
                    label=r"Absolute saturation error $|\hat u-u^*|$")
    colour_bar.ax.minorticks_off()
    destination = args.output_dir / "figure_r3_structured_error_topology"
    figure.savefig(destination.with_suffix(".pdf"))
    figure.savefig(destination.with_suffix(".png"))
    figure.savefig(destination.with_suffix(".svg"))
    plt.close(figure)

    with (args.output_dir / "structured_error_topology.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    print(f"wrote {destination.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
