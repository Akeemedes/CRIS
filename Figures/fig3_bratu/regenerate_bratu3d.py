"""Regenerate the documented 3D Bratu reference used by Fig. 3."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "bratu3d_regenerated"
import bratu_model as bratu


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "bratu3d_lam2_dx001_dz002.npz"
    x, y, z, solution = bratu.compute_or_load_bratu_solution(
        cache_file=cache, force_recompute=False, dx=.01, dy=.01, dz=.02, lam=2.0
    )
    manifest = {
        "artifact": "bratu3d_reference",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "equation": "-Delta u = lambda exp(u)",
        "lambda": 2.0,
        "domain": [0.0, 1.0],
        "grid": {"nx": len(x), "ny": len(y), "nz": len(z), "dx": .01, "dy": .01, "dz": .02},
        "solver": {"nonlinear": "damped Newton", "linear": "GMRES", "nonlinear_tolerance": 1e-8},
        "solution_min": float(solution.min()),
        "solution_max": float(solution.max()),
        "cache": cache.name,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {cache}")


if __name__ == "__main__":
    main()
