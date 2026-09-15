"""Plot paired transport accuracy, water cuts and solver work for seven problems."""

import os
import glob
import csv
import json
import hashlib
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from publication_style import enforce_print_fonts


# ============================================================
# Global style
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],

    # Larger, print-readable fonts
    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.0,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "legend.fontsize": 9.0,

    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "axes.grid": True,
    "grid.alpha": 0.16,
    "grid.linewidth": 0.45,
    "grid.linestyle": "--",

    "figure.dpi": 300,
    "savefig.dpi": 600,
})

COL_SRDM = "#D62728"
COL_CRIS = "#1F77B4"
HERE=Path(__file__).resolve().parent
BENCH=HERE.parents[1]/'training/twophasetransport/analysis/regular_v3/benchmarks'
SOURCE_MANIFEST=None

def accuracy(case):
    path=Path(next(r['accuracy'] for r in SOURCE_MANIFEST['cases'] if r['slug']==case)) if SOURCE_MANIFEST else (BENCH/'accuracy.csv' if case=='1D' else BENCH/case/'reference_accuracy.csv')
    with path.open() as f:
        return sorted([r for r in csv.DictReader(f) if r['case']==case and r['model']=='lambda_0p01'],key=lambda r:float(r['time']))

def verify_pair(case):
    def config(path):
        return {p[0].upper():p[1] for line in path.read_text().splitlines() if len(p:=line.split(maxsplit=1))==2 and not line.startswith('#')}
    c,s=Path(case['cris']),Path(case['srdm']);cc,sc=config(c.parent/'sim.txt'),config(s.parent/'sim.txt')
    for key in ['RTOL','DUTOL','LINMAX','LINTOL','MAX_STEPS_NEWTON','REFRESH_TRANSPORT_PRECONDITIONER','DT_INIT','DT_MIN','DT_MAX','DT_INCR','DT_DECR','T_END','REPORT_FREQ','MU_W','MU_NW','N_W','N_NW','NG']:
        assert cc.get(key)==sc.get(key),(case['slug'],key,cc.get(key),sc.get(key))
    assert cc['REFRESH_TRANSPORT_PRECONDITIONER']=='1'
    if SOURCE_MANIFEST:
        assert cc['MODE']=='CRIS' and sc['MODE']=='SRDM'
        assert cc['NEWTONUPDATER']=='STANDARD' and sc['NEWTONUPDATER']=='LINESEARCH'
        if case.get('srdm_provenance')!='separate_completed_benchmark':
            for key in ('PRESSURE_LINMAX','LINTOL_MIN','LINEAR_SOLVE_POLICY'):
                assert cc[key]==sc[key],(case['slug'],key)
        else:assert case['slug']=='3DSPE10_5SpotBase'
    if case.get('image'):
        for cfg in (cc,sc):
            assert cfg['INIT_FIELD'].upper()=='HOMOGENEOUS' and float(cfg['INIT_VALUE'])==0
    model=(c.parent/cc['CRIS_MODEL']).resolve()
    expected=BENCH.parent/'packages/residual_lambda_0p01_final.pt'
    assert hashlib.sha256(model.read_bytes()).digest()==hashlib.sha256(expected.read_bytes()).digest()


def matched_case_sources():
    """Resolve field views, accuracy references and paired solver outputs."""
    if SOURCE_MANIFEST:
        cases=SOURCE_MANIFEST['cases']
        for case in cases:verify_pair(case)
        return cases[0],cases[1:4],cases[4:]
    root = Path(__file__).resolve().parent.parent
    srdm_root = root / "benchmark_spe10_srdm" / "cases"
    cris_root = BENCH

    def source(label, slug, image_folder):
        return {
            "label": label,
            "slug": slug,
            "cris": cris_root / slug / "lambda_0p01" / "output",
            "srdm": srdm_root / slug / "SRDM" / "output",
            "image": root / image_folder / "CRIS" / "output" / "output.jpg",
        }

    one_d = {
        "slug": "1D",
        "cris": BENCH / "1D/lambda_0p01/output",
        "srdm": root / "1D" / "NX_10000_full_fresh_ilu_linmax7_dt150" / "SRDM" / "output",
    }
    two_d = [
        source("Layer 50", "2DSPE10_Layer50", "2DSPE10/Layer50"),
        source("Layer 75", "2DSPE10_Layer75", "2DSPE10/Layer75"),
        source("Layer 84", "2DSPE10_Layer84", "2DSPE10/Layer84"),
    ]
    three_d = [
        source("Norne", "Norne_5SpotBase", "Norne/5SpotBase"),
        source("SPE10", "3DSPE10_5SpotBase", "3DSPE10/5SpotBase"),
        source("EDFM", "EDFM", "EDFM"),
    ]
    for case in [one_d,*two_d,*three_d]:verify_pair(case)
    return one_d, two_d, three_d


def case_outputs(case):
    """Resolve CRIS/SRDM output paths and the optional field-view image."""
    if isinstance(case, dict):
        return str(case["cris"]), str(case["srdm"]), str(case.get("image", ""))
    return os.path.join(case, "CRIS/output/"), os.path.join(case, "SRDM/output/"), os.path.join(case, "CRIS/output/output.jpg")


# ============================================================
# IO and metrics
# ============================================================
def read_state_binary(filepath):
    with open(filepath, "rb") as f:
        n = np.frombuffer(f.read(8), dtype=np.uint64)[0]
        return np.frombuffer(f.read(int(n) * 8), dtype=np.float64).copy()


def read_csv_rows(filepath):
    with open(filepath, encoding="utf-8", newline="") as stream:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]


def _bin_series_index(output_folder, prop="saturation"):
    d = {}
    for fp in glob.glob(str(Path(output_folder) / f"{prop}_*.bin")):
        t = int(Path(fp).stem.split("_")[1])
        d[t] = fp
    return d


def state_errors_from_bins(cris_out, srdm_out, prop="saturation", clip=True):
    C = _bin_series_index(cris_out, prop=prop)
    S = _bin_series_index(srdm_out, prop=prop)

    common_times = sorted(set(C).intersection(S))
    if not common_times:
        raise FileNotFoundError(
            f"No common {prop}_*.bin times between:\n"
            f"  {cris_out}\n"
            f"  {srdm_out}"
        )

    rows = []
    sum_abs = 0.0
    sum_sq = 0.0
    n_tot = 0

    for t in common_times:
        a = read_state_binary(C[t])
        b = read_state_binary(S[t])

        if clip:
            a = np.clip(a, 0.0, 1.0)
            b = np.clip(b, 0.0, 1.0)

        if a.size != b.size:
            raise ValueError(
                f"Size mismatch at time {t}: CRIS={a.size}, SRDM={b.size}"
            )

        d = a - b
        n = d.size

        mae_t = float(np.mean(np.abs(d)))
        mse_t = float(np.mean(d * d))

        rows.append({
            "time": t,
            "MAE": mae_t,
            "MSE": mse_t,
            "RMSE": float(np.sqrt(mse_t)),
            "n": int(n),
        })

        sum_abs += float(np.sum(np.abs(d)))
        sum_sq += float(np.sum(d * d))
        n_tot += int(n)

    per_time = rows

    global_metrics = {
        "prop": prop,
        "MAE": sum_abs / n_tot,
        "RMSE": float(np.sqrt(sum_sq / n_tot)),
        "n_times": int(len(common_times)),
        "n_vals_total": int(n_tot),
    }

    return global_metrics, per_time


def perf_speedups(perf_CRIS, perf_SRDM):
    keys = ('NLNSTEPS', 'LINSTEPS', 'NFEVAL')
    cris = np.array([sum(row[key] for row in perf_CRIS) for key in keys], dtype=float)
    srdm = np.array([sum(row[key] for row in perf_SRDM) for key in keys], dtype=float)
    if not (np.isfinite(cris).all() and np.isfinite(srdm).all()
            and np.all(cris > 0) and np.all(srdm > 0)):
        raise ValueError('Logarithmic work ratios require finite, positive component counts')
    return {
        "Component work ratios (SRDM/CRIS)": srdm / cris,
        # One Jacobian build per Newton update; equal weights define a work proxy.
        "Equal-weight work speedup":
            sum(row[key] for row in perf_SRDM for key in ('NLNSTEPS','LINSTEPS','NFEVAL')) /
            sum(row[key] for row in perf_CRIS for key in ('NLNSTEPS','LINSTEPS','NFEVAL')),
        "Nonlinear steps speedup (CRIS/SRDM)":
            sum(row["NLNSTEPS"] for row in perf_CRIS) / sum(row["NLNSTEPS"] for row in perf_SRDM),

        "Linear steps speedup (CRIS/SRDM)":
            sum(row["LINSTEPS"] for row in perf_CRIS) / sum(row["LINSTEPS"] for row in perf_SRDM),

        "Function evals speedup (CRIS/SRDM)":
            sum(row["NFEVAL"] for row in perf_CRIS) / sum(row["NFEVAL"] for row in perf_SRDM),
    }


# ============================================================
# Axes-level plot functions
# ============================================================
def plot_saturation_evolution_ax(ax, output_folder, cax=None):
    """
    Saturation evolution plot adapted from the standalone version:

      - multiple saturation profiles colored by viridis
      - qualitative vertical colorbar as the time legend
      - no colorbar ticks
      - colorbar label is exactly the intended "Time  →" style

    The only adaptation is that this draws into supplied axes and
    optionally supplied colorbar axes for full-page layout control.
    """
    files = sorted(
        glob.glob(f"{output_folder}/saturation_*.bin"),
        key=lambda x: int(Path(x).stem.split("_")[1])
    )

    if not files:
        raise FileNotFoundError(f"No saturation files found in {output_folder}")

    states = []
    time_steps = []

    for fp in files:
        states.append(read_state_binary(fp))
        time_steps.append(int(Path(fp).stem.split("_")[1]))

    states = np.array(states)
    time_steps = np.array(time_steps)

    nsteps = len(states)
    colors = plt.cm.viridis(np.linspace(0, 1, nsteps))

    for sat, color in zip(states, colors):
        ax.plot(sat, color=color, lw=1.15, alpha=0.88)

    # Colorbar as qualitative time legend, matching the standalone plot.
    sm = plt.cm.ScalarMappable(
        cmap="viridis",
        norm=plt.Normalize(time_steps[0], time_steps[-1])
    )
    sm.set_array([])

    if cax is not None:
        cbar = ax.figure.colorbar(sm, cax=cax)
    else:
        cbar = ax.figure.colorbar(sm, ax=ax, pad=0.02)

    cbar.set_label(
        "Time  →",
        fontsize=9.0,
        fontweight="bold",
        labelpad=2.0,
    )
    cbar.ax.set_yticks([])          # qualitative: no tick values
    cbar.ax.tick_params(length=0)
    cbar.ax.grid(False)
    cbar.outline.set_linewidth(0.6)

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.6)

    ax.set_xlim(0, states.shape[1] - 1)
    ax.set_ylim(-0.02, 1.05)

    ax.set_xlabel("x  →", fontweight="bold", fontsize=10.0, labelpad=2)
    ax.set_ylabel("Water saturation", fontweight="bold", fontsize=10.0, labelpad=3)

    ax.set_xticks([])               # qualitative x-axis
    ax.tick_params(axis="both", labelsize=9.0, pad=1.5)


def plot_speedup_ax(ax, speed_tbl):
    ratios = speed_tbl['Component work ratios (SRDM/CRIS)']
    total = speed_tbl['Equal-weight work speedup']
    # Positions, not bar lengths, encode multiplicative reductions. All seven
    # panels share the same axis, including an explicit unity reference.
    if not all(.7 <= value <= 15000 for value in [*ratios, total]):
        raise ValueError('Work ratio lies outside the common axis limits')
    ax.set_xscale('log')
    ax.set_xlim(.7, 15000)
    ax.set_ylim(-.55, 3.55)
    ax.grid(False)
    ax.hlines([0, 1, 2, 3], 1, 10000, color='.9', lw=.65, zorder=1)
    ax.axvline(1, color='.5', lw=.8, zorder=2)
    ax.plot(ratios, [3, 2, 1], ls='none', marker='o', ms=5,
            color=COL_CRIS, zorder=3)
    ax.plot([total], [0], ls='none', marker='D', ms=5.5,
            color='.12', zorder=3)
    ax.spines['left'].set_visible(False)
    summary=float(f"{speed_tbl['Equal-weight work speedup']:.2g}")
    ax.set_title(rf"Work speedup: $\mathbf{{{summary:g}}}\times$",
                 fontsize=11, fontweight='normal', math_fontfamily='stix', pad=5)
    ax.set_yticks([3, 2, 1, 0], ['Jacobian', 'Krylov', 'Residual', 'Total'])
    ax.set_xticks([1, 10, 100, 1000, 10000], [rf'$10^{power}$' for power in range(5)])
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.tick_params(axis="x", pad=1.0)
    ax.tick_params(axis="y", labelsize=9.0, pad=4, length=0)

    ax.set_xlabel('')
    ax.set_ylabel('')

def plot_sat_error_ax(ax, sat_per_time, metric="MAE"):
    y = np.array([row[metric] for row in sat_per_time])
    y = np.insert(y, 0, 0.0)
    t = np.r_[0,[float(row['time']) for row in sat_per_time]]
    t=t/t[-1]

    ax.plot(t, y, color=COL_CRIS, lw=1.65)

    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(0, None)

    ax.set_xlabel(r"$t$", fontsize=11.0, labelpad=2)
    ax.set_ylabel(metric, fontweight="bold", fontsize=10.5, labelpad=3)

    ax.set_xticks([0,.5,1])
    ax.tick_params(axis="both", labelsize=9.0, pad=1.5)

    ax.yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.get_offset_text().set_fontsize(9.0)
    ax.yaxis.get_offset_text().set_fontweight("bold")


def plot_well_series_ax(ax, well_CRIS, well_SRDM, col=None, reference_rmse=None):
    if col is None:
        col = next(key for key in well_CRIS[0] if key != "Time")

    if col not in well_CRIS[0] or col not in well_SRDM[0]:
        raise KeyError(
            f"well_col='{col}' not found in both CSVs.\n"
            f"CRIS cols: {list(well_CRIS[0])}\n"
            f"SRDM cols: {list(well_SRDM[0])}"
        )

    tC = np.array([row["Time"] for row in well_CRIS])
    tS = np.array([row["Time"] for row in well_SRDM])
    yC = np.array([row[col] for row in well_CRIS])
    yS = np.array([row[col] for row in well_SRDM])

    # Zero initial saturation is checked in verify_pair. Reports start after
    # accepted steps; prepend the initial state without modifying report rows.
    if tC[0]>0:tC=np.r_[0,tC];yC=np.r_[0,yC]
    if tS[0]>0:tS=np.r_[0,tS];yS=np.r_[0,yS]

    end=max(tS[-1],tC[-1]);tS=tS/end;tC=tC/end
    ax.plot(tS, yS, color=COL_SRDM, lw=1.55, zorder=2)
    ax.plot(tC, yC, color=COL_CRIS, lw=1.55, ls="--", dashes=(4, 2), zorder=3)

    ax.set_xlim(min(tC[0], tS[0]), max(tC[-1], tS[-1]))
    ax.set_ylim(0.0, 1.0)

    ax.set_xlabel(r"$t$", fontsize=11.0, labelpad=2)
    ax.set_ylabel(
        "Produced\nwater cut" if SOURCE_MANIFEST else "Producer\nsaturation",
        fontweight="bold",
        fontsize=10.0,
        labelpad=3,
    )

    ax.set_xticks([0,.5,1])
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.5))
    ax.tick_params(axis="both", labelsize=9.0, pad=1.5)

    # Full-field saturation error; its reference is defined in the caption.
    # The lower-right area is clear of these increasing production curves.
    if reference_rmse is not None:
        label = HPacker(children=[
            TextArea('RMSE = ', textprops=dict(fontsize=11, fontweight='normal')),
            TextArea(f'{reference_rmse:.1e}', textprops=dict(fontsize=11, fontweight='bold')),
        ], align='baseline', pad=0, sep=0)
        ax.add_artist(AnnotationBbox(
            label, (0.97, 0.035), xycoords=ax.transAxes,
            box_alignment=(1, 0), frameon=False, pad=0, zorder=10))


def show_case_image_ax(ax, jpg_path):
    if not os.path.exists(jpg_path):
        raise FileNotFoundError(jpg_path)

    img = mpimg.imread(jpg_path)

    ax.imshow(img, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)


# ============================================================
# Full figure builder
# ============================================================
def build_nature_fullpage(
    outpath="Figure_fullpage.pdf",
    well_col=None,
    fig_width_in=7.2,
    fig_height_in=10.2,
    matched=False,
):
    """
    Layout:
      panel a = approximately one row height
      panel b = three equal-height rows
      panel c = three equal-height rows

    Image path:
      case_folder/CRIS/output/output.jpg
    """

    if matched:
        case_1d, cases_2d, cases_3d = matched_case_sources()
    else:
        case_1d = "./1D/NX_10000/"
        cases_2d = ["./2DSPE10/Layer50/", "./2DSPE10/Layer75/", "./2DSPE10/Layer84/"]
        cases_3d = ["./Norne/5SpotBase/", "./3DSPE10/5SpotBase/", "./EDFM/"]

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        top=0.958,
        bottom=0.080,
    )

    # Panel a is about one row.
    # Panels b/c are each three rows.
    outer = fig.add_gridspec(
        3, 1,
        height_ratios=[1.08, 3.0, 3.0],
        hspace=0.28,
    )

    # ------------------------------------------------------------
    # Panel label helpers
    # ------------------------------------------------------------
    def add_left_panel_label(ax_list, letter, xpad=0.004, ypad=0.005):
        """
        Small panel label toward the left edge of a panel.
        """
        boxes = [ax.get_position() for ax in ax_list]
        x0 = min(b.x0 for b in boxes)
        y1 = max(b.y1 for b in boxes)

        fig.text(
            x0 - xpad,
            y1 + ypad,
            letter,
            ha="right",
            va="bottom",
            fontsize=13,
            fontweight="black",
        )

    def add_between_rows_legend(upper_axes, lower_axes):
        """
        Add SRDM/CRIS legend centered between row 1 image axes
        and row 2 speedup axes of panels b/c.
        """
        upper_boxes = [ax.get_position() for ax in upper_axes]
        lower_boxes = [ax.get_position() for ax in lower_axes]

        x0 = min(b.x0 for b in upper_boxes + lower_boxes)
        x1 = max(b.x1 for b in upper_boxes + lower_boxes)

        y_upper_bottom = min(b.y0 for b in upper_boxes)
        y_lower_top = max(b.y1 for b in lower_boxes)

        x_mid = 0.5 * (x0 + x1)
        y_mid = 0.5 * (y_upper_bottom + y_lower_top) + .011

        # Centered compact SRDM/CRIS key
        fig.text(
            x_mid - 0.075,
            y_mid,
            "■",
            color=COL_SRDM,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            x_mid - 0.055,
            y_mid,
            "SRDM",
            color="black",
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="center",
        )

        fig.text(
            x_mid + 0.020,
            y_mid,
            "■",
            color=COL_CRIS,
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="center",
        )
        fig.text(
            x_mid + 0.040,
            y_mid,
            "CRIS",
            color="black",
            fontsize=9.5,
            fontweight="bold",
            ha="left",
            va="center",
        )

    # ------------------------------------------------------------
    # Compact SRDM/CRIS key, top-centre for panel a
    # ------------------------------------------------------------
    fig.text(
        0.455, 0.995,
        "■",
        color=COL_SRDM,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.text(
        0.475, 0.995,
        "SRDM",
        color="black",
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.text(
        0.555, 0.995,
        "■",
        color=COL_CRIS,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="top",
    )

    fig.text(
        0.575, 0.995,
        "CRIS",
        color="black",
        fontsize=9.5,
        fontweight="bold",
        ha="left",
        va="top",
    )

    # ============================================================
    # Panel a
    # ============================================================
    # Keep the current row spacing that works, while reserving room
    # inside column 1 for the colorbar label.
    gs_a = outer[0].subgridspec(
        1, 3,
        wspace=0.58,
    )

    # Panel a, column 1:
    #   saturation axis | colorbar | invisible spacer for colorbar label
    #
    # The spacer is important because cbar.set_label("Time  →") draws
    # the label to the right of the colorbar. Reserving this space keeps
    # the desired visual and prevents overlap with column 2.
    gs_a1 = gs_a[0, 0].subgridspec(
        1, 3,
        width_ratios=[1.0, 0.045, 0.145],
        wspace=0.045,
    )

    ax_a1 = fig.add_subplot(gs_a1[0, 0])
    cax_a1 = fig.add_subplot(gs_a1[0, 1])
    spacer_a1 = fig.add_subplot(gs_a1[0, 2])
    spacer_a1.set_axis_off()
    spacer_a1.grid(False)

    ax_a2 = fig.add_subplot(gs_a[0, 1])
    ax_a3 = fig.add_subplot(gs_a[0, 2])

    CRIS_1d, SRDM_1d, _ = case_outputs(case_1d)
    plot_saturation_evolution_ax(ax_a1, CRIS_1d, cax=cax_a1)

    perf_CRIS = read_csv_rows(os.path.join(CRIS_1d, "solver_report.csv"))
    perf_SRDM = read_csv_rows(os.path.join(SRDM_1d, "solver_report.csv"))

    speed_tbl = perf_speedups(perf_CRIS, perf_SRDM)

    sat_pt=[dict(time=r['time'],RMSE=float(r['rmse'])) for r in accuracy('1D')]

    plot_speedup_ax(ax_a2, speed_tbl)
    plot_sat_error_ax(ax_a3, sat_pt, metric="RMSE")

    add_left_panel_label([ax_a1, ax_a2, ax_a3], "a")

    # ============================================================
    # Panels b and c
    # ============================================================
    def draw_case_panel(gs_slot, letter, case_folders):
        gs = gs_slot.subgridspec(
            3, 3,
            height_ratios=[1.0, 1.0, 1.0],
            hspace=0.49,
            wspace=0.28,
        )

        image_axes = []
        speed_axes = []
        well_axes = []

        for j, case_folder in enumerate(case_folders):
            CRIS, SRDM, image_path = case_outputs(case_folder)

            perfC = read_csv_rows(os.path.join(CRIS, "solver_report.csv"))
            perfS = read_csv_rows(os.path.join(SRDM, "solver_report.csv"))
            spd_tbl = perf_speedups(perfC, perfS)

            wellC = read_csv_rows(case_folder['cris_watercut'] if SOURCE_MANIFEST else os.path.join(CRIS, 'well_report.csv'))
            wellS = read_csv_rows(case_folder['srdm_watercut'] if SOURCE_MANIFEST else os.path.join(SRDM, 'well_report.csv'))

            # Row 1: image
            ax_img = fig.add_subplot(gs[0, j])
            image_axes.append(ax_img)

            show_case_image_ax(
                ax_img,
                image_path,
            )
            label = case_folder.get("label", f"case {j + 1}") if isinstance(case_folder, dict) else f"case {j + 1}"
            ax_img.set_title(label, loc="left", fontsize=9.3, fontweight="bold", pad=3)

            # Row 2: speedup
            ax_spd = fig.add_subplot(gs[1, j])
            speed_axes.append(ax_spd)

            plot_speedup_ax(ax_spd, spd_tbl)
            ax_spd.set_xlabel("")
            ax_spd.tick_params(axis="x", labelsize=9, pad=1)
            ax_spd.tick_params(axis="y", labelsize=9.0, pad=1.5)

            # Row 3: well
            ax_w = fig.add_subplot(gs[2, j])
            well_axes.append(ax_w)

            plot_well_series_ax(ax_w, wellC, wellS, col=well_col,reference_rmse=float(accuracy(case_folder['slug'])[-1]['rmse']))
            ax_w.tick_params(axis="x", labelsize=9.0, pad=1.5)
            ax_w.tick_params(axis="y", labelsize=9.0, pad=1.5)

            # Remove repeated y labels/ticks
            if j != 0:
                for ax in (ax_spd, ax_w):
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelleft=False)

        # Small left panel label
        add_left_panel_label(image_axes, letter)

        # Repeated centered SRDM/CRIS legend between row 1 and row 2
        add_between_rows_legend(image_axes, speed_axes)

    draw_case_panel(outer[1], "b", cases_2d)
    draw_case_panel(outer[2], "c", cases_3d)

    if matched:
        write_cost_table([case_1d,*cases_2d,*cases_3d])
    fig.text(.55, .014, 'Work ratios: SRDM / CRIS', ha='center', va='center', fontsize=11)
    # Fixed geometry. Avoid bbox_inches="tight" because it can distort spacing.
    enforce_print_fonts(fig)
    fig.savefig(outpath)
    fig.savefig(Path(outpath).with_suffix(".png"), dpi=300)
    fig.savefig(Path(outpath).with_suffix('.svg'))
    plt.close(fig)

    print(f"Saved: {outpath}")
    print(f"Saved: {Path(outpath).with_suffix('.png')}")
    plt.close(fig)
    return outpath


# ============================================================
# Companion table uses exactly the plotted runs, including failed attempts.
# ============================================================
def write_cost_table(cases):
    def records(path):
        with path.open(newline='') as stream:return list(csv.DictReader(stream))
    cris_summary=records(BENCH/'summary.csv') if not SOURCE_MANIFEST else []
    srdm_summary=records(HERE.parent/'benchmark_spe10_srdm/summary.csv') if not SOURCE_MANIFEST else []
    one_d=records(HERE.parent/'1D/NX_10000_full_fresh_ilu_linmax7_dt150/summary.csv') if not SOURCE_MANIFEST else []
    aliases={'2DSPE10_Layer50':('2DSPE10','Layer50'),'2DSPE10_Layer75':('2DSPE10','Layer75'),
             '2DSPE10_Layer84':('2DSPE10','Layer84'),'Norne_5SpotBase':('Norne','5SpotBase'),
             '3DSPE10_5SpotBase':('3DSPE10','5SpotBase'),'EDFM':('EDFM','EDFM')}
    lines=['# Transport computational cost','',
           'CRIS with residual weight lambda=0.01 and paired SRDM results.',
           'All factors are SRDM/CRIS. Equal-weight work sums Jacobian builds, Krylov',
           'iterations and residual evaluations before taking the ratio. Counts include',
           'failed attempts; one Jacobian build is represented by each Newton update.',
           '', '| Case | Jacobian factor | Krylov factor | Residual factor | Work factor | CRIS (s) | SRDM (s) | Time factor |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    counts=[];numeric=[]
    for case in cases:
        slug=case['slug']
        if not SOURCE_MANIFEST:
            c=next(r for r in cris_summary if r['case']==slug and r['model']=='lambda_0p01')
            s=next(r for r in one_d if r['method']=='SRDM') if slug=='1D' else next(r for r in srdm_summary if (r['family'],r['case'])==aliases[slug])
        cc=[sum(r[k] for r in read_csv_rows(Path(case['cris'])/'solver_report.csv')) for k in ('NLNSTEPS','LINSTEPS','NFEVAL')]
        ss=[sum(r[k] for r in read_csv_rows(Path(case['srdm'])/'solver_report.csv')) for k in ('NLNSTEPS','LINSTEPS','NFEVAL')]
        if SOURCE_MANIFEST:
            c=dict(zip(('newton','krylov','residual_evaluations'),cc));c['wall_seconds']=case['cris_seconds']
            s=dict(zip(('newton_steps','linear_steps','residual_evaluations'),ss));s['wall_seconds']=case['srdm_seconds']
        assert cc==[float(c[k]) for k in ('newton','krylov','residual_evaluations')]
        assert ss==[float(s[k]) for k in ('newton_steps','linear_steps','residual_evaluations')]
        cw,sw=float(c['wall_seconds']),float(s['wall_seconds'])
        name=case.get('label','1D')
        vals=[*(b/a for a,b in zip(cc,ss)),sum(ss)/sum(cc),cw,sw,sw/cw]
        lines.append('| '+name+' | '+' | '.join(f'{v:.3g}' for v in vals)+' |')
        counts.append('| '+name+' | '+' | '.join(f'{int(b):,} / {int(a):,}' for a,b in zip(cc,ss))+' |')
        if SOURCE_MANIFEST:
            reports={k:read_csv_rows(Path(case[k])/'solver_report.csv') for k in ('cris','srdm')}
            numeric.append(dict(case=slug,label=name,cris_seconds=cw,srdm_seconds=sw,time_factor=sw/cw,
                work_factor=sum(ss)/sum(cc),jacobian_factor=ss[0]/cc[0],krylov_factor=ss[1]/cc[1],residual_factor=ss[2]/cc[2],
                cris_newton=cc[0],srdm_newton=ss[0],cris_krylov=cc[1],srdm_krylov=ss[1],cris_residual=cc[2],srdm_residual=ss[2],
                **{k+'_accepted':sum(r['converged']==1 for r in rr) for k,rr in reports.items()},
                **{k+'_failed':sum(r['converged']!=1 for r in rr) for k,rr in reports.items()},
                **{k+'_linear_seconds':sum(r['LNSOLVE TIME (S)'] for r in rr) for k,rr in reports.items()},
                final_rmse=float(accuracy(slug)[-1]['rmse']),reference_residual_inf=float(accuracy(slug)[-1]['reference_residual_inf']),
                cris_provenance=case['cris_provenance'],srdm_provenance=case['srdm_provenance']))
    lines+=['','Times include process startup, pressure solution and I/O.',
            'Each time is a single observation. Concurrent reference computations may',
            'affect the three-dimensional timings; these measurements do not provide',
            'repeated-trial confidence intervals.',
            '', '## Underlying counts (SRDM / CRIS)','',
            '| Case | Jacobian builds | Krylov iterations | Residual evaluations |',
            '|---|---:|---:|---:|',*counts,'','## Timing sources','',
            '- CRIS: `training/twophasetransport/analysis/regular_v3/benchmarks/summary.csv`',
            '- SRDM, 1D: `Figures/1D/NX_10000_full_fresh_ilu_linmax7_dt150/summary.csv`',
            '- SRDM, other cases: `Figures/benchmark_spe10_srdm/summary.csv`',
            '', 'Paths above are relative to the repository root. Timing-summary work counts',
            'are asserted to equal the plotted native solver-report counts on every export.','']
    if SOURCE_MANIFEST:
        start=lines.index('Times include process startup, pressure solution and I/O.')
        end=lines.index('## Underlying counts (SRDM / CRIS)')
        lines[start:end]=['Times are measured sequential single-process observations including startup,',
            'pressure and I/O. Reference generation is excluded. These are single realizations,',
            'not repeated-timing confidence intervals. The manifest records threads and host.', '']
        lines=lines[:lines.index('## Timing sources')]+['## Timing sources','',
            '`publication_sources.json` identifies the output files and process time for each paired run.',
            'All cases completed; failed attempts are included. Independent references use CRIS time grids.',
            'Producer curves use physical fractional-flow water cuts reconstructed from completion rates.']
        lines+=['', '## Accuracy and timestep accounting','',
                '| Case | Final saturation RMSE | CRIS accepted / failed | SRDM accepted / failed |',
                '|---|---:|---:|---:|']
        lines += [f"| {r['label']} | {r['final_rmse']:.4g} | {r['cris_accepted']} / {r['cris_failed']} | {r['srdm_accepted']} / {r['srdm_failed']} |" for r in numeric]
        with (HERE/'data/current_model_cost_table.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(numeric[0]));writer.writeheader();writer.writerows(numeric)
    (HERE/'data/current_model_cost_table.md').write_text('\n'.join(lines),encoding='utf-8')

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,default=HERE/'data/publication_sources.json',
                        help='Manifest of the seven paired benchmark runs')
    args=parser.parse_args()
    if args.sources:
        SOURCE_MANIFEST=json.loads(args.sources.read_text())
        assert SOURCE_MANIFEST['all_completed'] and len(SOURCE_MANIFEST['cases'])==7
        for case in SOURCE_MANIFEST['cases']:
            for key in ('cris','srdm','accuracy','image','cris_watercut','srdm_watercut'):
                if key in case:case[key]=str((HERE.parents[1]/case[key]).resolve())
    build_nature_fullpage(
        outpath=Path(__file__).resolve().parent / "fig4_transport.pdf",
        well_col=None,        # None = automatically use first well column, e.g. P1
        fig_width_in=7.2,     # Approx. Nature double-column width, 183 mm
        fig_height_in=10.2,
        matched=True,
    )
