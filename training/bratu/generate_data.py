"""Branch-consistent Bratu data; bisection labels independently checked by W0."""
import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
import numpy as np
from scipy.special import lambertw

BASE = Path(__file__).resolve().parent
DATA = BASE / "datasets/lower_branch"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(n, seed, fold_test=False):
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2, 6, n)
    cap = np.minimum(1.0, np.exp(-a - 1) * (1 - 1e-8))
    beta = np.exp(rng.uniform(np.log(1e-8), np.log(cap)))
    family = np.arange(n) % 100
    beta[family < 8] = 0
    fold = (family >= 60) & (family < 90)
    deficit = 10 ** rng.uniform(-8, -1, n)
    # At a<-1 the beta<=1 range truncates the fold; cap is declared in metadata.
    beta[fold] = np.minimum(1.0, np.exp(-a[fold] - 1) * (1 - deficit[fold]))
    uniform = family >= 90
    beta[uniform] = rng.uniform(size=uniform.sum()) * cap[uniform]
    if fold_test:
        a = rng.uniform(-1, 6, n)
        deficit = 10 ** rng.uniform(-10, -8, n)
        beta = np.exp(-a - 1) * (1 - deficit)
    lo, hi = a.copy(), a + 1
    for _ in range(80):
        mid = (lo + hi) / 2
        r = a - mid + beta * np.exp(mid)
        lo = np.where(r > 0, mid, lo)
        hi = np.where(r > 0, hi, mid)
    root = np.where(beta == 0, a, (lo + hi) / 2)
    analytic = a - lambertw(-beta * np.exp(a), 0).real
    delta = float(np.max(np.abs(root - analytic)))
    residual = a - root + beta * np.exp(root)
    slope = 1 - beta * np.exp(root)
    if delta > 1e-9 or not np.all(slope > 0) or np.max(np.abs(residual)) > 2e-14:
        raise ValueError(f"Label validation failed: discrepancy={delta}")
    return np.column_stack([a, beta, root, slope]).astype("<f8"), {
        "rows": n, "seed": seed, "max_W0_discrepancy": delta,
        "max_physical_residual": float(np.max(np.abs(residual))),
        "minimum_root_slope": float(slope.min()), "beta_zero_rows": int((beta == 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DATA,
                        help="New output directory for the four datasets and manifest")
    args = parser.parse_args()
    output = args.output.resolve()
    started = time.perf_counter()
    try:
        output.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"Output directory already exists: {output}; choose a new directory")
    products = {}
    for name, n, seed, fold in [("train", 62400, 20260919, False),
                                ("validation", 100000, 20260931, False),
                                ("test", 20000, 20261001, False),
                                ("test_near_fold", 20000, 20261002, True)]:
        values, report = generate(n, seed, fold)
        path = output / f"{name}.bin"
        path.write_bytes(b"BRATUD01" + struct.pack("<Q", n) + values.tobytes())
        products[name] = dict(report, path=path.name, sha256=digest(path))
    manifest = {
        "physics": "a-u+beta*exp(u)=0", "branch": "lower, continued from beta=0; Lambert W0",
        "root_formula": "u=a-W0(-beta*exp(a))", "a_range": [-2, 6], "beta_range": [0, 1],
        "branch_domain": "beta*exp(a)<=1/e", "train_fold_deficit_min": 1e-8,
        "near_fold_test_deficit": [1e-10, 1e-8],
        "sampling": {"beta_zero": .08, "log_beta": .52, "near_fold": .30, "uniform_beta": .10},
        "input_transform": ["(a+2)/4-1", "2*log1p(beta/1e-6)/log1p(1e6)-1"],
        "output_transform": "normalized=(u+2)/4.5-1; physical=4.5*(normalized+1)-2",
        "physical_MSE_multiplier": 20.25,
        "generation_seconds": time.perf_counter() - started,
        "generator_sha256": digest(Path(__file__)), "products": products,
        "product_path_base": "Directory containing this manifest",
        "branch_domain_note": "The selected real branch ends at the fold; points beyond it have no real-valued training labels.",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(products, indent=2))


if __name__ == "__main__":
    main()
