"""Validate safeguarded Bratu continuation labels against the Lambert-W0 inverse.

The manuscript's local generator is r(U; alpha1, alpha2) = U + alpha1 exp(U) +
alpha2.  At alpha1=0 the regular reference root is U=-alpha2.  This script compares
two continuation paths from regular roots to the same target and verifies that both
remain on the W0 branch without crossing the critical state Uc.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.special import lambertw


OUT_DIR = Path(__file__).resolve().parent
DATA_DIR = OUT_DIR / "data"
N_ALPHA2 = 49
N_RHO = 55
N_STEPS = 80
TOL = 1.0e-13
MAX_NEWTON = 30
GUARD = 1.0e-11


@dataclass
class ContinuationResult:
    root: float
    residual: float
    min_derivative: float
    min_branch_gap: float
    converged: bool
    crossed_critical: bool


def residual(u: float, alpha1: float, alpha2: float) -> float:
    return u + alpha1 * np.exp(u) + alpha2


def derivative(u: float, alpha1: float) -> float:
    return 1.0 + alpha1 * np.exp(u)


def exact_w0(alpha1: float, alpha2: float) -> float:
    # Let y=U+alpha2. Then -y exp(-y)=alpha1 exp(-alpha2), hence
    # U=-alpha2-W0(alpha1 exp(-alpha2)).
    return float((-alpha2 - lambertw(alpha1 * np.exp(-alpha2), 0)).real)


def critical_state(alpha1: float) -> float:
    return np.inf if alpha1 == 0.0 else float(np.log(-1.0 / alpha1))


def safeguarded_newton(u0: float, alpha1: float, alpha2: float) -> tuple[float, bool, bool]:
    """Newton solve restricted to the lower branch U < Uc when alpha1 < 0."""
    u = float(u0)
    uc = critical_state(alpha1)
    crossed = False
    for _ in range(MAX_NEWTON):
        r = residual(u, alpha1, alpha2)
        if abs(r) < TOL:
            return u, True, crossed
        d = derivative(u, alpha1)
        if not np.isfinite(d) or d <= 0.0:
            return u, False, True
        delta = -r / d
        step = 1.0
        accepted = False
        for _ in range(50):
            trial = u + step * delta
            if alpha1 < 0.0 and trial >= uc - GUARD:
                crossed = True
                step *= 0.5
                continue
            if np.isfinite(trial) and abs(residual(trial, alpha1, alpha2)) <= (1.0 - 1.0e-4 * step) * abs(r):
                u = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            return u, False, crossed
    return u, False, crossed


def continue_path(alpha1_target: float, alpha2_target: float, schedule: str) -> ContinuationResult:
    if schedule == "alpha1_only":
        alpha1_values = np.linspace(0.0, alpha1_target, N_STEPS + 1)
        alpha2_values = np.full(N_STEPS + 1, alpha2_target)
        u = -alpha2_target
    elif schedule == "admissible_curved":
        # Interpolate alpha2 and the branch-distance coordinate rho rather than
        # alpha1 directly. This remains on the real/admissible side rho<1 at every
        # continuation level, unlike an arbitrary straight line in (alpha1,alpha2).
        t = np.linspace(0.0, 1.0, N_STEPS + 1)
        alpha2_values = t * alpha2_target
        rho_target = -np.e * alpha1_target * np.exp(-alpha2_target)
        alpha1_values = -(t * rho_target) * np.exp(alpha2_values - 1.0)
        u = 0.0
    else:
        raise ValueError(schedule)

    min_derivative = np.inf
    min_gap = np.inf
    crossed = False
    for alpha1, alpha2 in zip(alpha1_values[1:], alpha2_values[1:], strict=True):
        u, converged, crossed_step = safeguarded_newton(u, float(alpha1), float(alpha2))
        crossed = crossed or crossed_step
        if not converged:
            return ContinuationResult(u, abs(residual(u, float(alpha1), float(alpha2))), min_derivative, min_gap, False, crossed)
        min_derivative = min(min_derivative, derivative(u, float(alpha1)))
        min_gap = min(min_gap, critical_state(float(alpha1)) - u)
    return ContinuationResult(u, abs(residual(u, alpha1_target, alpha2_target)), min_derivative, min_gap, True, crossed)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[object, ...]] = []
    # rho=-e*alpha1*exp(-alpha2) parametrizes distance to the W branch point;
    # rho<1 is precisely the real two-branch admissible side of the fold.
    alpha2_values = np.linspace(-1.5, 1.5, N_ALPHA2)
    rho_values = np.linspace(0.02, 0.999, N_RHO)
    for alpha2 in alpha2_values:
        for rho in rho_values:
            alpha1 = -rho * np.exp(alpha2 - 1.0)
            exact = exact_w0(alpha1, alpha2)
            completed: dict[str, ContinuationResult] = {}
            for schedule in ("alpha1_only", "admissible_curved"):
                result = continue_path(float(alpha1), float(alpha2), schedule)
                completed[schedule] = result
            disagreement = abs(completed["alpha1_only"].root - completed["admissible_curved"].root)
            for schedule, result in completed.items():
                rows.append(
                    (
                        schedule,
                        alpha1,
                        alpha2,
                        rho,
                        exact,
                        result.root,
                        abs(result.root - exact),
                        disagreement,
                        result.residual,
                        result.min_derivative,
                        result.min_branch_gap,
                        int(result.converged),
                        int(result.crossed_critical),
                    )
                )

    header = [
        "schedule", "alpha1", "alpha2", "rho", "lambert_w0", "continuation_label",
        "abs_label_error", "path_disagreement", "residual_abs", "min_derivative",
        "min_branch_gap", "converged", "critical_guard_activated",
    ]
    with (DATA_DIR / "continuation_validation.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    errors = np.array([float(row[6]) for row in rows])
    disagreements = np.array([float(row[7]) for row in rows])
    summary = {
        "artifact": "bratu_safeguarded_continuation_validation",
        "status": "semi_analytical_w0_branch_validation",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "equation": "U + alpha1 exp(U) + alpha2 = 0",
        "reference": "U = -alpha2 - W0(alpha1 exp(-alpha2))",
        "parameterization": "rho = -e alpha1 exp(-alpha2), with rho < 1 on the real two-branch side",
        "grid": {"n_alpha2": N_ALPHA2, "n_rho": N_RHO, "alpha2_range": [-1.5, 1.5], "rho_range": [0.02, 0.999]},
        "continuation_steps": N_STEPS,
        "paths": ["alpha1_only", "admissible_curved"],
        "max_label_error": float(errors.max()),
        "max_path_disagreement": float(disagreements.max()),
        "all_converged": bool(all(bool(row[11]) for row in rows)),
        "any_critical_guard_activated": bool(any(bool(row[12]) for row in rows)),
    }
    (DATA_DIR / "continuation_validation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {DATA_DIR / 'continuation_validation.csv'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
