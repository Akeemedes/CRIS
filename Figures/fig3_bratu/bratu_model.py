"""Bratu inverse branches, three-dimensional reference solve and solution plots."""

import os
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.special import lambertw
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================
# Global style
# ============================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],

    "font.size": 9.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "xtick.labelsize": 9.0,
    "ytick.labelsize": 9.0,
    "legend.fontsize": 8.8,

    "axes.linewidth": 0.75,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "figure.dpi": 300,
    "savefig.dpi": 600,
})


# ============================================================
# Constants / colours
# ============================================================
COL_LOWER = "0.12"
COL_UPPER = "0.48"
COL_CRIT = "#C7352E"

CACHE_FILE = str(
    Path(__file__).resolve().parent / "data" / "bratu3d_regenerated" / "bratu3d_lam2_dx001_dz002.npz"
)


# ============================================================
# Lambert branch utilities
# ============================================================
def SSLambert(alpha, beta):
    """
    Solve reduced scalar equation

        alpha - u + beta exp(u) = 0

    using Lambert W branches.

    Lower branch:
        u = alpha - W_0(-beta exp(alpha))

    Upper branch:
        u = alpha - W_{-1}(-beta exp(alpha))
    """
    z = -beta * np.exp(alpha)

    lower = alpha - lambertw(z, 0).real
    upper = alpha - lambertw(z, -1).real

    return lower, upper


def critical_alpha(beta):
    """
    Critical alpha for fixed beta.

    Singularity condition:
        -beta exp(alpha_c) = -1/e

    hence:
        alpha_c = log(1 / beta) - 1
        u_c     = alpha_c + 1 = log(1 / beta)
    """
    alpha_c = np.log(1.0 / beta) - 1.0
    u_c = alpha_c + 1.0
    return alpha_c, u_c


def critical_beta(alpha):
    """
    Critical beta for fixed alpha.

    Singularity condition:
        -beta_c exp(alpha) = -1/e

    hence:
        beta_c = exp(-(alpha + 1))
        u_c    = alpha + 1
    """
    beta_c = np.exp(-(alpha + 1.0))
    u_c = alpha + 1.0
    return beta_c, u_c


# ============================================================
# Bratu 3D solver utilities
# ============================================================
def build_grid(Lx, Ly, Lz, dx, dy, dz):
    Nx = int(round(Lx / dx)) + 1
    Ny = int(round(Ly / dy)) + 1
    Nz = int(round(Lz / dz)) + 1

    x = np.linspace(0.0, Lx, Nx)
    y = np.linspace(0.0, Ly, Ny)
    z = np.linspace(0.0, Lz, Nz)

    dx = x[1] - x[0] if Nx > 1 else 1.0
    dy = y[1] - y[0] if Ny > 1 else 1.0
    dz = z[1] - z[0] if Nz > 1 else 1.0

    return x, y, z, dx, dy, dz


def interior_indexer(Nx, Ny, Nz):
    """
    Map interior full-grid indices

        i = 1, ..., Nx - 2
        j = 1, ..., Ny - 2
        k = 1, ..., Nz - 2

    to a flat index p in [0, Nint - 1].
    """
    nxi = Nx - 2
    nyi = Ny - 2
    nzi = Nz - 2
    Nint = nxi * nyi * nzi

    def idx(i, j, k):
        return (k - 1) * (nxi * nyi) + (j - 1) * nxi + (i - 1)

    return idx, Nint


def bc2(X, Y, Z):
    """
    Nonzero Dirichlet condition on z = 0 face:

        u(x, y, 0) = sin(pi x)

    and zero elsewhere.
    """
    if np.isclose(Z, 0.0):
        return np.sin(np.pi * X)
    return 0.0


def init2(X, Y, Z):
    """
    Mild interior initial guess.
    """
    return (
        1.0
        * np.sin(np.pi * X)
        * np.sin(np.pi * Y)
        * np.sin(np.pi * Z)
    )


def assemble_residual_and_jacobian(
    u_int,
    x,
    y,
    z,
    dx,
    dy,
    dz,
    lam,
    bc_func,
):
    """
    Assemble residual and Jacobian for

        -Delta u - lambda exp(u) = 0

    written with the positive diagonal Laplacian convention:

        lap_neg = diag_lap * u_i - sum_neighbors coeff * u_neighbor

        F_i = lap_neg - lambda exp(u_i)
    """
    Nx, Ny, Nz = len(x), len(y), len(z)
    idx, Nint = interior_indexer(Nx, Ny, Nz)

    ax = 1.0 / dx**2 if Nx > 3 else 0.0
    ay = 1.0 / dy**2 if Ny > 3 else 0.0
    az = 1.0 / dz**2 if Nz > 3 else 0.0

    diag_lap = 2.0 * (ax + ay + az)

    F = np.zeros(Nint, dtype=float)

    rows = []
    cols = []
    data = []

    def u_full(i, j, k):
        if 1 <= i <= Nx - 2 and 1 <= j <= Ny - 2 and 1 <= k <= Nz - 2:
            return u_int[idx(i, j, k)]
        return bc_func(x[i], y[j], z[k])

    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                p = idx(i, j, k)
                u0 = u_int[p]

                ux_p = u_full(i + 1, j, k)
                ux_m = u_full(i - 1, j, k)
                uy_p = u_full(i, j + 1, k)
                uy_m = u_full(i, j - 1, k)
                uz_p = u_full(i, j, k + 1)
                uz_m = u_full(i, j, k - 1)

                lap_neg = (
                    diag_lap * u0
                    - ax * (ux_p + ux_m)
                    - ay * (uy_p + uy_m)
                    - az * (uz_p + uz_m)
                )

                F[p] = lap_neg - lam * np.exp(u0)

                rows.append(p)
                cols.append(p)
                data.append(diag_lap - lam * np.exp(u0))

                if i + 1 <= Nx - 2:
                    rows.append(p)
                    cols.append(idx(i + 1, j, k))
                    data.append(-ax)

                if i - 1 >= 1:
                    rows.append(p)
                    cols.append(idx(i - 1, j, k))
                    data.append(-ax)

                if j + 1 <= Ny - 2:
                    rows.append(p)
                    cols.append(idx(i, j + 1, k))
                    data.append(-ay)

                if j - 1 >= 1:
                    rows.append(p)
                    cols.append(idx(i, j - 1, k))
                    data.append(-ay)

                if k + 1 <= Nz - 2:
                    rows.append(p)
                    cols.append(idx(i, j, k + 1))
                    data.append(-az)

                if k - 1 >= 1:
                    rows.append(p)
                    cols.append(idx(i, j, k - 1))
                    data.append(-az)

    J = sp.coo_matrix((data, (rows, cols)), shape=(Nint, Nint)).tocsr()

    return F, J


def _gmres_solve(J, rhs, rtol=1e-4, restart=50, maxiter=500):
    """
    SciPy-version-tolerant GMRES wrapper.
    """
    try:
        du, info = spla.gmres(
            J,
            rhs,
            atol=0.0,
            rtol=rtol,
            restart=restart,
            maxiter=maxiter,
        )
    except TypeError:
        du, info = spla.gmres(
            J,
            rhs,
            tol=rtol,
            restart=restart,
            maxiter=maxiter,
        )

    if info != 0:
        raise RuntimeError(f"GMRES failed, info={info}")

    return du


def newton_solve_bratu_3d(
    Lx=1.0,
    Ly=1.0,
    Lz=1.0,
    dx=0.01,
    dy=0.01,
    dz=0.02,
    lam=2.0,
    bc_func=bc2,
    init_func=init2,
    tol=1e-8,
    max_newton=100,
    linear_solver="gmres",
    damping=True,
    verbose=True,
):
    x, y, z, dx, dy, dz = build_grid(Lx, Ly, Lz, dx, dy, dz)

    Nx, Ny, Nz = len(x), len(y), len(z)

    if Nx < 3 or Ny < 3 or Nz < 3:
        raise ValueError("Need at least 3 grid points in each direction.")

    idx, Nint = interior_indexer(Nx, Ny, Nz)

    if verbose:
        print(f"Grid: Nx={Nx}, Ny={Ny}, Nz={Nz}")
        print(f"Interior unknowns: {Nint}")

    u = np.zeros(Nint, dtype=float)

    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                u[idx(i, j, k)] = init_func(x[i], y[j], z[k])

    def norm_inf(v):
        return np.linalg.norm(v, ord=np.inf)

    def merit(F):
        return 0.5 * float(F @ F)

    for it in range(1, max_newton + 1):
        F, J = assemble_residual_and_jacobian(
            u,
            x,
            y,
            z,
            dx,
            dy,
            dz,
            lam,
            bc_func,
        )

        Fn = norm_inf(F)

        if verbose:
            print(f"Newton {it:02d}: ||F||_inf = {Fn:.3e}")

        if Fn < tol:
            if verbose:
                print("Converged.")
            break

        rhs = -F

        if linear_solver == "spsolve":
            du = spla.spsolve(J, rhs)
        elif linear_solver == "gmres":
            du = _gmres_solve(J, rhs, rtol=1e-4, restart=50, maxiter=500)
        else:
            raise ValueError("linear_solver must be 'spsolve' or 'gmres'.")

        if damping:
            alpha = 1.0
            phi0 = merit(F)
            c = 1e-4

            for _ in range(20):
                u_trial = u + alpha * du

                F_trial, _ = assemble_residual_and_jacobian(
                    u_trial,
                    x,
                    y,
                    z,
                    dx,
                    dy,
                    dz,
                    lam,
                    bc_func,
                )

                if merit(F_trial) <= (1.0 - c * alpha) * phi0:
                    u = u_trial
                    break

                alpha *= 0.5
            else:
                u = u + alpha * du
        else:
            alpha = 1.0
            u = u + du

        if verbose:
            print(
                f"           alpha = {alpha:.3e}, "
                f"||du||_inf = {norm_inf(du):.3e}"
            )

    else:
        raise RuntimeError("Newton did not converge within max_newton iterations.")

    U = np.zeros((Nx, Ny, Nz), dtype=float)

    for k in range(Nz):
        for j in range(Ny):
            for i in range(Nx):
                if i in (0, Nx - 1) or j in (0, Ny - 1) or k in (0, Nz - 1):
                    U[i, j, k] = bc_func(x[i], y[j], z[k])

    for k in range(1, Nz - 1):
        for j in range(1, Ny - 1):
            for i in range(1, Nx - 1):
                U[i, j, k] = u[idx(i, j, k)]

    return x, y, z, U


def compute_or_load_bratu_solution(
    cache_file=CACHE_FILE,
    force_recompute=False,
    dx=0.01,
    dy=0.01,
    dz=0.02,
    lam=2.0,
):
    """
    Load cached Bratu solution if available; otherwise solve and cache.
    """
    cache_path = Path(cache_file)

    if cache_path.exists() and not force_recompute:
        print(f"Loading cached Bratu solution from {cache_file}")
        data = np.load(cache_file)
        return data["x"], data["y"], data["z"], data["U"]

    print("Solving Bratu 3D problem...")
    x, y, z, U = newton_solve_bratu_3d(
        dx=dx,
        dy=dy,
        dz=dz,
        lam=lam,
        linear_solver="gmres",
        bc_func=bc2,
        init_func=init2,
        damping=True,
        verbose=True,
    )

    print(f"Saving Bratu solution cache to {cache_file}")
    np.savez_compressed(cache_file, x=x, y=y, z=z, U=U)

    return x, y, z, U


# ============================================================
# Axes-level plotting functions
# ============================================================
def style_branch_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="both", labelsize=9.0, pad=1.5)
    ax.grid(False)


def plot_alpha_branch_ax(ax, beta=0.1):
    """
    Panel a:
    Branches versus alpha_2 at fixed alpha_1 = beta.
    """
    alpha_c, u_c = critical_alpha(beta)

    eps = 1e-7
    alpha = np.linspace(0.0, alpha_c * (1.0 - eps), 1000)

    lower, upper = SSLambert(alpha, beta)

    ax.plot(
        alpha,
        lower,
        color=COL_LOWER,
        lw=1.35,
        label="Lower branch",
    )

    ax.plot(
        alpha,
        upper,
        color=COL_UPPER,
        lw=1.35,
        ls="--",
        label="Upper branch",
    )

    ax.scatter(
        alpha_c,
        u_c,
        color=COL_CRIT,
        s=20,
        zorder=5,
        label=r"$u_c$",
    )

    ax.set_xlabel(r"$\alpha_2$", fontsize=10.5, fontweight="bold", labelpad=2)
    ax.set_ylabel(r"$u$", fontsize=10.5, fontweight="bold", labelpad=2)

    ax.text(
        0.18,
        0.84,
        rf"$\alpha_1 = {beta:.1f}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.02, 0.43),
        handlelength=2.0,
        labelspacing=0.35,
        borderpad=0.1,
        fontsize=8.8,
    )

    style_branch_axis(ax)


def plot_beta_branch_ax(ax, alpha=1.0):
    """
    Panel b:
    Branches versus alpha_1 = beta at fixed alpha_2 = alpha.
    """
    beta_c, u_c = critical_beta(alpha)

    beta = np.linspace(1e-8, beta_c * (1.0 - 1e-7), 1000)

    lower, upper = SSLambert(alpha, beta)

    ax.plot(
        beta,
        lower,
        color=COL_LOWER,
        lw=1.35,
        label="Lower branch",
    )

    ax.plot(
        beta,
        upper,
        color=COL_UPPER,
        lw=1.35,
        ls="--",
        label="Upper branch",
    )

    ax.scatter(
        beta_c,
        u_c,
        color=COL_CRIT,
        s=20,
        zorder=5,
        label=r"$u_c$",
    )

    ax.set_xlabel(r"$\alpha_1$", fontsize=10.5, fontweight="bold", labelpad=2)
    ax.set_ylabel(r"$u$", fontsize=10.5, fontweight="bold", labelpad=2)

    ax.text(
        0.18,
        0.84,
        rf"$\alpha_2 = {alpha:.1f}$",
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=9.5,
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        loc="center left",
        bbox_to_anchor=(0.34, 0.43),
        handlelength=2.0,
        labelspacing=0.35,
        borderpad=0.1,
        fontsize=8.8,
    )

    style_branch_axis(ax)


def plot_slice_ax(ax, cax, x, y, z, U, zloc=0.5, cmap="viridis"):
    """Plot a constant-z slice, with color limits set by that slice's range."""
    iz = int(np.argmin(np.abs(z - zloc)))
    z_actual = z[iz]

    data = U[:, :, iz]

    im = ax.imshow(
        data.T,
        origin="lower",
        extent=[x[0], x[-1], y[0], y[-1]],
        aspect="auto",
        cmap=cmap,
        vmin=float(data.min()),
        vmax=float(data.max()),
        interpolation="bilinear",
    )

    ax.set_xlabel(r"$x$", fontsize=10.5, fontweight="bold", labelpad=2)
    ax.set_ylabel(r"$y$", fontsize=10.5, fontweight="bold", labelpad=2)

    ax.set_title(
        rf"$z = {z_actual:.1f}$",
        fontsize=10.5,
        fontweight="bold",
        pad=2,
    )

    ax.tick_params(axis="both", labelsize=9.0, pad=1.5)

    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.65)

    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.set_label(r"$u$", fontsize=10.5, fontweight="bold", labelpad=3)
    cbar.ax.tick_params(labelsize=9.0, pad=1.5)
    cbar.outline.set_linewidth(0.6)

    for spine in cbar.ax.spines.values():
        spine.set_linewidth(0.6)


def add_panel_label(fig, ax, label, xpad=0.006, ypad=0.012):
    box = ax.get_position()

    fig.text(
        box.x0 - xpad,
        box.y1 + ypad,
        label,
        ha="left",
        va="bottom",
        fontsize=12,
        fontweight="black",
    )


# ============================================================
# Full merged figure
# ============================================================
def build_merged_bratu_figure(
    outpath="bratu_merged_figure.pdf",
    force_recompute=False,
    fig_width_in=7.2,
    fig_height_in=2.25,
):
    """
    Build one compact double-column figure:

      a) Lambert branches vs alpha_2
      b) Lambert branches vs alpha_1
      c) Bratu solution slice at z = 0.5
    """
    x, y, z, U = compute_or_load_bratu_solution(
        cache_file=CACHE_FILE,
        force_recompute=force_recompute,
        dx=0.01,
        dy=0.01,
        dz=0.02,
        lam=2.0,
    )

    fig = plt.figure(figsize=(fig_width_in, fig_height_in))

    # Reserve room on the right for the colorbar and its tick labels.
    fig.subplots_adjust(
        left=0.055,
        right=0.945,
        top=0.865,
        bottom=0.235,
    )

    outer = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.10],
        wspace=0.34,
    )

    ax_a = fig.add_subplot(outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])

    gs_c = outer[0, 2].subgridspec(
        1,
        2,
        width_ratios=[1.0, 0.050],
        wspace=0.075,
    )

    ax_c = fig.add_subplot(gs_c[0, 0])
    cax_c = fig.add_subplot(gs_c[0, 1])

    plot_alpha_branch_ax(ax_a, beta=0.1)
    plot_beta_branch_ax(ax_b, alpha=1.0)
    plot_slice_ax(ax_c, cax_c, x, y, z, U, zloc=0.5, cmap="viridis")

    add_panel_label(fig, ax_a, "a)")
    add_panel_label(fig, ax_b, "b)")
    add_panel_label(fig, ax_c, "c)")

    fig.savefig(outpath)
    fig.savefig(Path(outpath).with_suffix(".png"), dpi=600)
    plt.close(fig)

    print(f"Saved → {outpath}")
    print(f"Saved → {Path(outpath).with_suffix('.png')}")

    return outpath


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    build_merged_bratu_figure(
        outpath=Path(__file__).resolve().parent / "bratu_branches_and_slice.pdf",
        force_recompute=False,
        fig_width_in=7.2,
        fig_height_in=2.25,
    )
