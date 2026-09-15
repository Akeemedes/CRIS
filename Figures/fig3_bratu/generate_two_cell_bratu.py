"""Generate two-cell Bratu branch and convergence data.

The local generator is r(u; a, beta) = a - u + beta exp(u).  Two cells are coupled
through a = alpha0 + coupling * u_neighbour.  The CRIS map uses the exact
lower Lambert-W branch. The resulting basins characterize exact local inversion
independently of network approximation error.
"""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.optimize import root
from scipy.special import lambertw


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"

ALPHA0 = 0.65
BETA = 0.10
COUPLING = 0.25
TOL = 1.0e-10
MAX_ITER = 80
GRID_MIN = 0.0
GRID_MAX = 3.0
LANDSCAPE_N = 181
BASIN_N = 91


@dataclass
class SolveResult:
    status: str
    state: np.ndarray
    trajectory: list[tuple[int, float, float, float, float]]


def residual(u: np.ndarray) -> np.ndarray:
    return np.array(
        [
            ALPHA0 + COUPLING * u[1] - u[0] + BETA * np.exp(u[0]),
            ALPHA0 + COUPLING * u[0] - u[1] + BETA * np.exp(u[1]),
        ],
        dtype=float,
    )


def jacobian(u: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [-1.0 + BETA * np.exp(u[0]), COUPLING],
            [COUPLING, -1.0 + BETA * np.exp(u[1])],
        ],
        dtype=float,
    )


def lower_inverse(a: np.ndarray) -> np.ndarray | None:
    z = -BETA * np.exp(a)
    if np.any(z < (-1.0 / np.e)):
        return None
    return a - lambertw(z, 0).real


def cris_residual(u: np.ndarray) -> np.ndarray | None:
    phi = lower_inverse(ALPHA0 + COUPLING * np.array([u[1], u[0]]))
    if phi is None:
        return None
    return u - phi


def norm2(value: np.ndarray) -> float:
    return float(np.linalg.norm(value, ord=2))


def solve_srdm(start: np.ndarray) -> SolveResult:
    u = start.astype(float).copy()
    trace: list[tuple[int, float, float, float, float]] = []
    for iteration in range(MAX_ITER + 1):
        r = residual(u)
        rnorm = norm2(r)
        trace.append((iteration, float(u[0]), float(u[1]), rnorm, 1.0))
        if rnorm < TOL:
            return SolveResult("converged", u, trace)
        try:
            delta = np.linalg.solve(jacobian(u), -r)
        except np.linalg.LinAlgError:
            return SolveResult("singular_jacobian", u, trace)

        merit = 0.5 * rnorm**2
        step = 1.0
        accepted = False
        for _ in range(30):
            trial = u + step * delta
            if np.all(np.isfinite(trial)) and np.all(np.abs(trial) < 30.0):
                trial_norm = norm2(residual(trial))
                if 0.5 * trial_norm**2 <= (1.0 - 1.0e-4 * step) * merit:
                    u = trial
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return SolveResult("line_search_failed", u, trace)
        trace[-1] = (iteration, float(trace[-1][1]), float(trace[-1][2]), rnorm, step)
    return SolveResult("iteration_limit", u, trace)


def solve_exact_cris(start: np.ndarray) -> SolveResult:
    u = start.astype(float).copy()
    trace: list[tuple[int, float, float, float, float]] = []
    for iteration in range(MAX_ITER + 1):
        g = cris_residual(u)
        if g is None:
            return SolveResult("outside_selected_branch", u, trace)
        rnorm = norm2(residual(u))
        trace.append((iteration, float(u[0]), float(u[1]), rnorm, 1.0))
        if rnorm < TOL:
            return SolveResult("converged", u, trace)
        u = u - g
    return SolveResult("iteration_limit", u, trace)


def discover_roots() -> list[np.ndarray]:
    candidates: list[np.ndarray] = []
    for x in np.linspace(0.0, 5.0, 7):
        for y in np.linspace(0.0, 5.0, 7):
            result = root(residual, np.array([x, y]), jac=jacobian, method="hybr")
            if result.success and norm2(residual(result.x)) < TOL and np.all(np.isfinite(result.x)):
                if not any(norm2(result.x - item) < 1.0e-7 for item in candidates):
                    candidates.append(result.x)
    return sorted(candidates, key=lambda value: (float(value.sum()), float(value[0])))


def root_label(state: np.ndarray, roots: list[np.ndarray]) -> str:
    distances = [norm2(state - item) for item in roots]
    index = int(np.argmin(distances))
    return f"root_{index}" if distances[index] < 1.0e-6 else "unclassified"


def write_csv(path: Path, header: list[str], rows: list[tuple[object, ...]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    roots = discover_roots()
    write_csv(
        DATA_DIR / "roots.csv",
        ["root_id", "u1", "u2", "residual_norm"],
        [(f"root_{i}", value[0], value[1], norm2(residual(value))) for i, value in enumerate(roots)],
    )

    landscape_rows: list[tuple[object, ...]] = []
    values = np.linspace(GRID_MIN, GRID_MAX, LANDSCAPE_N)
    for u1 in values:
        for u2 in values:
            state = np.array([u1, u2])
            srdm_norm = norm2(residual(state))
            g = cris_residual(state)
            cris_norm = float("nan") if g is None else norm2(g)
            condition = float(np.linalg.cond(jacobian(state)))
            landscape_rows.append((u1, u2, srdm_norm, cris_norm, int(g is not None), condition))
    write_csv(
        DATA_DIR / "landscape.csv",
        ["u1", "u2", "srdm_residual_norm", "cris_residual_norm", "cris_admissible", "srdm_jacobian_condition"],
        landscape_rows,
    )

    starts = [
        (0.10, 0.10),
        (0.20, 2.20),
        (1.20, 1.20),
        (2.20, 0.20),
        (2.20, 2.20),
        (2.80, 1.00),
    ]
    trajectory_rows: list[tuple[object, ...]] = []
    for start_id, start in enumerate(starts):
        for method, solver in (("SRDM", solve_srdm), ("CRIS_exact", solve_exact_cris)):
            solved = solver(np.array(start))
            for iteration, u1, u2, rnorm, step in solved.trajectory:
                trajectory_rows.append((start_id, method, solved.status, iteration, u1, u2, rnorm, step))
    write_csv(
        DATA_DIR / "trajectories.csv",
        ["start_id", "method", "status", "iteration", "u1", "u2", "physical_residual_norm", "step_length"],
        trajectory_rows,
    )

    basin_rows: list[tuple[object, ...]] = []
    basin_values = np.linspace(GRID_MIN, GRID_MAX, BASIN_N)
    for u1 in basin_values:
        for u2 in basin_values:
            start = np.array([u1, u2])
            for method, solver in (("SRDM", solve_srdm), ("CRIS_exact", solve_exact_cris)):
                solved = solver(start)
                label = root_label(solved.state, roots) if solved.status == "converged" else solved.status
                iterations = len(solved.trajectory) - 1 if solved.trajectory else 0
                basin_rows.append((method, u1, u2, solved.status, label, iterations, solved.state[0], solved.state[1]))
    write_csv(
        DATA_DIR / "basins.csv",
        ["method", "u1_initial", "u2_initial", "status", "outcome", "iterations", "u1_final", "u2_final"],
        basin_rows,
    )
    outcomes = {(str(method), float(u1), float(u2)): str(outcome) for method, u1, u2, _status, outcome, _iterations, _f1, _f2 in basin_rows}
    admissible_starts = [
        (u1, u2)
        for method, u1, u2, _status, outcome, _iterations, _f1, _f2 in basin_rows
        if method == "CRIS_exact" and outcome == "root_0"
    ]
    srdm_selected = sum(outcomes[("SRDM", u1, u2)] == "root_0" for u1, u2 in admissible_starts)
    summary_rows = [
        ("selected_branch_admissible_starts", "CRIS_exact", len(admissible_starts), len(admissible_starts), 1.0),
        ("selected_branch_admissible_starts", "SRDM", len(admissible_starts), srdm_selected, srdm_selected / len(admissible_starts)),
    ]
    write_csv(
        DATA_DIR / "basin_summary.csv",
        ["domain", "method", "starts", "selected_lower_root", "selected_lower_root_fraction"],
        summary_rows,
    )

    manifest = {
        "artifact": "bratu_two_cell_exact_cris",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "controlled_exact_inverse_mechanism_diagnostic",
        "equations": {
            "local": "r(u; a, beta) = a - u + beta exp(u)",
            "coupled": "a_i = alpha0 + coupling * u_j",
            "cris_branch": "Lambert-W principal (lower) branch",
        },
        "parameters": {
            "alpha0": ALPHA0,
            "beta": BETA,
            "coupling": COUPLING,
            "tolerance": TOL,
            "max_iterations": MAX_ITER,
            "landscape_grid": LANDSCAPE_N,
            "basin_grid": BASIN_N,
            "state_domain": [GRID_MIN, GRID_MAX],
        },
        "notes": [
            "This uses the exact selected inverse, not a learned Bratu network.",
            "It supports Fig. 3 mechanism panels and does not establish trained-model accuracy.",
        ],
        "selected_branch_admissible_capture": {
            "CRIS_exact": 1.0,
            "SRDM": srdm_selected / len(admissible_starts),
        },
    }
    (DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote two-cell Bratu data to {DATA_DIR}")


if __name__ == "__main__":
    main()
