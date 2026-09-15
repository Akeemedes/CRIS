"""Generate conservative two-cell transport roots and convergence basins.

The system is a periodic two-cell upwind finite-volume ring.  With positive
velocity at both faces, each cell receives the other cell's fractional-flow
flux, giving the local equation

    u_i - u_old,i + beta * (f(u_i) - a_i) = 0,

where a_i=f(u_j).  SRDM uses the physical coupled residual. CRIS uses the
supplied transport local inverse and its exact network derivative. The small
system allows roots and solver trajectories to be examined directly.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
MODEL_DIR = ROOT / "build" / "TwoPhaseTransport" / "cases" / "models" / "TransportNetRegular2"
EVALUATOR = ROOT / "Figures" / "fig5_local_inverse" / "evaluate_saved_transport_models.py"

BETA = float(os.environ.get("TWO_CELL_BETA", "100.0"))
MAX_TRAINED_BETA = 200.0
SYSTEM = os.environ.get("TWO_CELL_SYSTEM", "open_chain").strip().lower()
if SYSTEM not in {"open_chain", "periodic_ring"}:
    raise ValueError("TWO_CELL_SYSTEM must be open_chain or periodic_ring")
U_OLD = np.array([0.0, 0.0]) if SYSTEM == "open_chain" else np.array([0.05, 0.95])
GRID_MIN = float(os.environ.get("TWO_CELL_GRID_MIN", "-0.5"))
GRID_MAX = float(os.environ.get("TWO_CELL_GRID_MAX", "1.5"))
GRID_N = int(os.environ.get("TWO_CELL_GRID_N", "121"))
LANDSCAPE_N = int(os.environ.get("TWO_CELL_LANDSCAPE_N", str(GRID_N)))
DOMAIN_TAG = f"domain_{GRID_MIN:g}_{GRID_MAX:g}".replace("-", "m").replace(".", "p")
DATA = OUT / "data" / SYSTEM / f"beta_{BETA:.0e}".replace("+", "p").replace("-", "m") / DOMAIN_TAG
MAX_ITER = 100
TOL = 1.0e-8
ARMIJO = 1.0e-4
STARTS = np.array([[-0.25, 1.25], [0.05, 0.95], [0.20, 0.80], [0.50, 0.50], [0.80, 0.20], [0.95, 0.05], [1.25, -0.25]])
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]


def load_evaluator():
    spec = importlib.util.spec_from_file_location("transport_model_evaluator", EVALUATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {EVALUATOR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def network_value_and_input_gradient(network, x: np.ndarray) -> tuple[float, np.ndarray]:
    """Mirror CRIS.hpp forward scaling and return dy/d(a,beta,u_old)."""
    value = 2.0 * (x - network.input_min) / (network.input_max - network.input_min) - 1.0
    jacobian = np.diag(2.0 / (network.input_max - network.input_min))
    for layer in network.layers:
        pre = layer.weight @ value
        if layer.bias is not None:
            pre = pre + layer.bias
        jacobian = layer.weight @ jacobian
        if layer.activation == 1:
            active = pre > 0.0
            value = np.maximum(pre, 0.0)
            jacobian = jacobian * active[:, None]
        elif layer.activation == 2:
            value = np.tanh(pre)
            jacobian = jacobian * (1.0 - value**2)[:, None]
        else:
            value = pre
    scale = 0.5 * (network.output_max - network.output_min)
    return float(network.output_min[0] + scale[0] * (value[0] + 1.0)), scale[0] * jacobian[0]


def fractional_flow_raw(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unclipped C++ regular fractional-flow algebra, valid beyond [0, 1]."""
    krw = u**2
    krnw = (1.0 - u)**2
    denominator = krw + 0.4 * krnw
    flux = krw / denominator
    derivative = 0.8 * u * (1.0 - u) / denominator**2
    return flux, derivative


def physical_residual(u: np.ndarray) -> np.ndarray:
    flux, _ = fractional_flow_raw(u)
    if SYSTEM == "open_chain":
        # Left boundary supplies f(S_in)=1; the right boundary carries f(u_2)
        # out of the domain. This is the conventional two-cell upwind chain.
        return np.array([u[0] - U_OLD[0] + BETA * (flux[0] - 1.0),
                         u[1] - U_OLD[1] + BETA * (flux[1] - flux[0])])
    return u - U_OLD + BETA * np.array([flux[0] - flux[1], flux[1] - flux[0]])


def physical_jacobian(u: np.ndarray) -> np.ndarray:
    _, derivative = fractional_flow_raw(u)
    if SYSTEM == "open_chain":
        return np.array([[1.0 + BETA * derivative[0], 0.0],
                         [-BETA * derivative[0], 1.0 + BETA * derivative[1]]], dtype=float)
    return np.array(
        [[1.0 + BETA * derivative[0], -BETA * derivative[1]],
         [-BETA * derivative[0], 1.0 + BETA * derivative[1]]],
        dtype=float,
    )


def cris_residual_and_jacobian(u: np.ndarray, network) -> tuple[np.ndarray, np.ndarray]:
    flux, derivative = fractional_flow_raw(u)
    if SYSTEM == "open_chain":
        # Inlet a_1=f(S_in)=1; second-cell inflow a_2=f(u_1).
        y0, _dy0 = network_value_and_input_gradient(network, np.array([1.0, BETA, U_OLD[0]]))
        y1, dy1 = network_value_and_input_gradient(network, np.array([flux[0], BETA, U_OLD[1]]))
        return u - np.array([y0, y1]), np.array([[1.0, 0.0], [-dy1[0] * derivative[0], 1.0]], dtype=float)
    # Periodic ring: a_1=f(u_2), a_2=f(u_1).
    y0, dy0 = network_value_and_input_gradient(network, np.array([flux[1], BETA, U_OLD[0]]))
    y1, dy1 = network_value_and_input_gradient(network, np.array([flux[0], BETA, U_OLD[1]]))
    residual = u - np.array([y0, y1])
    jacobian = np.array(
        [[1.0, -dy0[0] * derivative[1]],
         [-dy1[0] * derivative[0], 1.0]],
        dtype=float,
    )
    return residual, jacobian


@dataclass
class Solve:
    status: str
    state: np.ndarray
    iterations: int
    trace: list[tuple[int, float, float, float, float]]


def solve(start: np.ndarray, method: str, network) -> Solve:
    u = start.astype(float).copy()
    trace: list[tuple[int, float, float, float, float]] = []
    for iteration in range(MAX_ITER + 1):
        if method == "SRDM":
            residual = physical_residual(u)
            jacobian = physical_jacobian(u)
        else:
            residual, jacobian = cris_residual_and_jacobian(u, network)
        norm = float(np.linalg.norm(residual))
        trace.append((iteration, float(u[0]), float(u[1]), norm, 1.0))
        if norm < TOL:
            return Solve("converged", u, iteration, trace)
        try:
            delta = np.linalg.solve(jacobian, -residual)
        except np.linalg.LinAlgError:
            return Solve("singular_jacobian", u, iteration, trace)
        merit = 0.5 * norm**2
        step = 1.0
        accepted = False
        for _ in range(35):
            trial = u + step * delta
            if np.all(np.isfinite(trial)) and np.max(np.abs(trial)) < 5.0:
                trial_residual = physical_residual(trial) if method == "SRDM" else cris_residual_and_jacobian(trial, network)[0]
                trial_merit = 0.5 * float(np.dot(trial_residual, trial_residual))
                if trial_merit <= (1.0 - ARMIJO * step) * merit:
                    u = trial
                    trace[-1] = (iteration, float(trace[-1][1]), float(trace[-1][2]), norm, step)
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            return Solve("line_search_failed", u, iteration, trace)
    return Solve("iteration_limit", u, MAX_ITER, trace)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if BETA > MAX_TRAINED_BETA:
        raise ValueError(f"beta={BETA:g} exceeds the empirical training limit {MAX_TRAINED_BETA:g}")
    DATA.mkdir(parents=True, exist_ok=True)
    evaluator = load_evaluator()
    network = evaluator.parse_network(MODEL_DIR)
    values = np.linspace(GRID_MIN, GRID_MAX, GRID_N)
    landscape_values = np.linspace(GRID_MIN, GRID_MAX, LANDSCAPE_N)
    common_start = np.array([0.20, 0.80])
    physical_base = float(np.linalg.norm(physical_residual(common_start)))
    cris_base = float(np.linalg.norm(cris_residual_and_jacobian(common_start, network)[0]))

    landscape: list[tuple[object, ...]] = []
    for u1 in landscape_values:
        for u2 in landscape_values:
            state = np.array([u1, u2])
            physical = physical_residual(state)
            cris, _ = cris_residual_and_jacobian(state, network)
            physical_norm = float(np.linalg.norm(physical))
            cris_norm = float(np.linalg.norm(cris))
            landscape.append((u1, u2, physical_norm, cris_norm, physical_norm / physical_base, cris_norm / cris_base,
                              physical[0], physical[1], cris[0], cris[1]))
    with (DATA / "landscape.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["u1", "u2", "srdm_residual_norm", "cris_residual_norm", "srdm_normalized", "cris_normalized",
                         "srdm_r1", "srdm_r2", "cris_r1", "cris_r2"])
        writer.writerows(landscape)

    trajectory_rows: list[tuple[object, ...]] = []
    basin_rows: list[tuple[object, ...]] = []
    for start_id, start in enumerate(STARTS):
        for method in ("SRDM", "CRIS"):
            result = solve(start, method, network)
            for iteration, u1, u2, norm, step in result.trace:
                trajectory_rows.append((method, start_id, result.status, iteration, u1, u2, norm, step))
    for u1 in values:
        for u2 in values:
            start = np.array([u1, u2])
            for method in ("SRDM", "CRIS"):
                result = solve(start, method, network)
                physical_norm = float(np.linalg.norm(physical_residual(result.state)))
                basin_rows.append((method, u1, u2, result.status, result.iterations, result.state[0], result.state[1], physical_norm))
    with (DATA / "trajectories.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "start_id", "status", "iteration", "u1", "u2", "residual_norm", "line_search_step"])
        writer.writerows(trajectory_rows)
    # Cluster terminal states to distinguish physically bounded and unbounded
    # SRDM roots rather than treating every converged state as equivalent.
    labeled_rows: list[tuple[object, ...]] = []
    root_counts: dict[str, int] = {}
    for method in ("SRDM", "CRIS"):
        roots: list[np.ndarray] = []
        method_rows = [row for row in basin_rows if row[0] == method]
        for row in method_rows:
            if row[3] != "converged":
                labeled_rows.append((*row, row[3]))
                continue
            state = np.array([row[5], row[6]])
            index = next((i for i, root in enumerate(roots) if np.linalg.norm(state - root) < 1.0e-5), None)
            if index is None:
                roots.append(state)
                index = len(roots) - 1
            bounded = bool(np.all((state >= 0.0) & (state <= 1.0)))
            if bounded:
                terminal_class = "bounded_root"
            elif state[0] > 1.0 and state[1] > 1.0:
                terminal_class = "u1_u2_above_1"
            elif state[0] > 1.0:
                terminal_class = "u1_above_1"
            elif state[1] > 1.0:
                terminal_class = "u2_above_1"
            else:
                terminal_class = "outside_bounds"
            labeled_rows.append((*row, terminal_class))
        root_counts[method] = len(roots)
    with (DATA / "basins.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["method", "u1_initial", "u2_initial", "status", "iterations", "u1_final", "u2_final", "physical_residual_norm_final", "terminal_root"])
        writer.writerows(labeled_rows)

    summary: list[dict[str, object]] = []
    for method in ("SRDM", "CRIS"):
        rows = [row for row in basin_rows if row[0] == method]
        successful = [row for row in rows if row[3] == "converged"]
        iterations = np.array([int(row[4]) for row in successful])
        summary.append({
            "method": method,
            "starts": len(rows),
            "converged": len(successful),
            "convergence_fraction": len(successful) / len(rows),
            "iteration_median": float(np.median(iterations)) if len(iterations) else np.nan,
            "iteration_max": int(iterations.max()) if len(iterations) else -1,
            "median_final_physical_residual": float(np.median([row[7] for row in successful])) if successful else np.nan,
            "distinct_terminal_roots": root_counts[method],
        })
    with (DATA / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    manifest = {
        "artifact": "two_cell_transport_geometry",
        "status": "controlled_mechanism_diagnostic_with_deployed_regular2_weights",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "physical_system": "two-cell open upwind chain with saturated inlet" if SYSTEM == "open_chain" else "two-cell periodic upwind finite-volume ring",
        "residual": "R_1=u_1-u_old,1+beta*(f(u_1)-1); R_2=u_2-u_old,2+beta*(f(u_2)-f(u_1))" if SYSTEM == "open_chain" else "R_i=u_i-u_old_i+beta*(f(u_i)-f(u_j))",
        "beta": BETA,
        "u_old": U_OLD.tolist(),
        "fractional_flow": {"M": 0.4, "n_w": 2.0, "n_nw": 2.0, "Ng": 0.0},
        "network": {"path": str(MODEL_DIR.resolve()), "config_sha256": sha256(MODEL_DIR / "config.txt"), "vars_sha256": sha256(MODEL_DIR / "vars.bin")},
        "grid": {"basin_n": GRID_N, "landscape_n": LANDSCAPE_N, "domain": [GRID_MIN, GRID_MAX]},
        "empirical_training_limit_beta": MAX_TRAINED_BETA,
        "solver": {"line_search": "Armijo", "max_iterations": MAX_ITER, "tolerance": TOL},
        "normalization_start": common_start.tolist(),
        "summary": summary,
    }
    (DATA / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
