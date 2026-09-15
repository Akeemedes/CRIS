#!/usr/bin/env python3
"""Generate immutable train/validation/test label bundles from the native oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_oracle(oracle: Path, law: str, split: str, design: str, samples: int, seed: int, beta_max: float, output: Path) -> dict[str, object]:
    command = [
        str(oracle), "--law", law, "--split", split, "--design", design,
        "--samples", str(samples), "--seed", str(seed), "--beta-max", repr(beta_max),
        "--output", str(output),
    ]
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "dataset": str(output),
        "dataset_sha256": sha256(output),
        "dataset_manifest": str(Path(str(output) + ".manifest.json")),
    }


def pack_dataset(packer: Path, input_csv: Path, output: Path, max_records: int | None = None) -> dict[str, object]:
    command = [str(packer), "--input", str(input_csv), "--output", str(output)]
    if max_records is not None:
        command += ["--max-records", str(max_records)]
    started = time.perf_counter()
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    return {
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "stdout": completed.stdout,
        "dataset": str(output),
        "dataset_sha256": sha256(output),
        "rows": max_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--law", choices=("regular", "imp"), required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=1_000_000)
    parser.add_argument("--validation", type=int, default=100_000)
    parser.add_argument("--test-per-design", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--training-design", choices=("stratified", "stratified_target_tail", "balanced_transport"), default="balanced_transport")
    parser.add_argument("--beta-max", type=float, default=1_000_000.0)
    parser.add_argument("--train-prefix", action="append", type=int,
                        help="Nested packed training-set size; repeatable. Defaults to 62.4k, 125k, 250k, 500k, and 1M.")
    parser.add_argument("--specification", type=Path,
                        default=Path("training/twophasetransport/specification/active_advection_v2_balanced_beta1e6.json"))
    args = parser.parse_args()
    if not args.oracle.is_file():
        raise SystemExit(f"Oracle executable not found: {args.oracle}")
    if not args.packer.is_file():
        raise SystemExit(f"Dataset packer executable not found: {args.packer}")
    if min(args.train, args.validation, args.test_per_design) <= 0:
        raise SystemExit("All sample counts must be positive")
    prefixes = args.train_prefix or [62_400, 125_000, 250_000, 500_000, 1_000_000]
    if any(prefix <= 0 or prefix > args.train or prefix % 200 != 0 for prefix in prefixes):
        raise SystemExit("Every --train-prefix must be positive, at most --train, and divisible by the 200-row sampling cycle")
    if args.training_design == "balanced_transport" and args.beta_max <= 100_000.0:
        raise SystemExit("balanced_transport requires --beta-max greater than 1e5")
    if not args.specification.is_file():
        raise SystemExit(f"Training specification not found: {args.specification}")

    args.outdir.mkdir(parents=True, exist_ok=False)
    jobs = [
        ("train", args.training_design, args.train, args.seed + 11),
        ("validation", args.training_design, args.validation, args.seed + 23),
        ("test_uniform", "uniform", args.test_per_design, args.seed + 37),
        ("test_edge", "edge", args.test_per_design, args.seed + 41),
        ("test_log_beta", "log_beta", args.test_per_design, args.seed + 43),
        ("test_target_tail", "target_tail", args.test_per_design, args.seed + 47),
        ("test_balanced_transport", "balanced_transport", args.test_per_design, args.seed + 53),
    ]
    products: dict[str, object] = {}
    for split, design, samples, seed in jobs:
        output = args.outdir / f"{split}.csv"
        products[split] = run_oracle(args.oracle, args.law, split, design, samples, seed, args.beta_max, output)

    packed_products: dict[str, object] = {}
    for prefix in prefixes:
        name = f"train_{prefix // 1_000}k.tptbin" if prefix < 1_000_000 else "train_1m.tptbin"
        packed_products[f"train_{prefix}"] = pack_dataset(args.packer, args.outdir / "train.csv", args.outdir / name, prefix)
    for split, _, _, _ in jobs[1:]:
        packed_products[split] = pack_dataset(args.packer, args.outdir / f"{split}.csv", args.outdir / f"{split}.tptbin")

    specification = args.specification.resolve()
    manifest = {
        "artifact": "twophasetransport_active_advection_label_bundle",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "law": args.law,
        "oracle": str(args.oracle.resolve()),
        "oracle_sha256": sha256(args.oracle),
        "specification": str(specification),
        "specification_sha256": sha256(specification),
        "beta_max": args.beta_max,
        "seed_root": args.seed,
        "training_design": args.training_design,
        "nested_train_prefixes": prefixes,
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "products": products,
        "packed_products": packed_products,
    }
    (args.outdir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote_bundle={args.outdir}")


if __name__ == "__main__":
    main()
