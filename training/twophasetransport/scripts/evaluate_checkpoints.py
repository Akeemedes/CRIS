#!/usr/bin/env python3
"""Evaluate native LM checkpoints on independent packed TwoPhaseTransport data partitions.

This offline evidence tool is deliberately separate from the C++ optimizer.
It evaluates every saved checkpoint in vectorized NumPy, keeping snapshot
diagnostics cheap enough to be part of a normal reproducible run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

MAGIC = b"TPTBIN01"
HEADER = struct.Struct("<8sQII")
ARCHITECTURES = {
    941: (3, 20, 20, 20, 1),
    1361: (3, 20, 20, 20, 20, 1),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        magic, count, fields, _ = HEADER.unpack(handle.read(HEADER.size))
    if magic != MAGIC or fields != 6:
        raise ValueError(f"Not a TwoPhaseTransport packed data file: {path}")
    if path.stat().st_size != HEADER.size + count * 6 * 8:
        raise ValueError(f"Unexpected packed data size: {path}")
    return np.fromfile(path, dtype="<f8", offset=HEADER.size).reshape(count, 6)


def load_weights(path: Path) -> np.ndarray:
    weights = np.fromfile(path, dtype="<f8")
    if weights.size not in ARCHITECTURES:
        supported = ", ".join(str(count) for count in ARCHITECTURES)
        raise ValueError(f"{path} has {weights.size} doubles; supported checkpoint sizes are {supported}")
    return weights


def network(weights: np.ndarray, rows: np.ndarray, beta_max: float, beta_coordinate: str) -> np.ndarray:
    beta = (2.0 * np.log1p(rows[:, 1]) / np.log1p(beta_max) - 1.0
            if beta_coordinate == "log1p" else 2.0 * rows[:, 1] / beta_max - 1.0)
    inputs = np.column_stack((2.0 * rows[:, 0] - 1.0, beta, 2.0 * rows[:, 2] - 1.0))
    widths = ARCHITECTURES[weights.size]
    values = inputs
    offset = 0
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        matrix_count = input_width * output_width
        matrix = weights[offset:offset + matrix_count].reshape(output_width, input_width)
        offset += matrix_count
        bias = weights[offset:offset + output_width]
        offset += output_width
        values = values @ matrix.T + bias
        if index + 1 < len(widths) - 1:
            values = np.tanh(values)
    if offset != weights.size:
        raise RuntimeError("Checkpoint layout did not consume every parameter")
    return 0.5 * (values[:, 0] + 1.0)


def fractional_flow(u: np.ndarray, law: str) -> np.ndarray:
    if law == "regular":
        m, nw, nnw = 0.4, 2.0, 2.0
    elif law == "imp":
        m, nw, nnw = 0.4, 2.0, 0.2
    else:
        raise ValueError(f"Unknown law {law}")
    numerator = u ** nw
    return numerator / (numerator + m * (1.0 - u) ** nnw)


def partitions(rows: np.ndarray, attribute: str):
    column = 1 if attribute == "beta" else 4
    values = rows[:, column]
    if attribute == "beta":
        edges = (0.0, 1.0, 10.0, 100.0, 1_000.0, 10_000.0, 100_000.0, 1_000_000.0, math.inf)
    elif attribute == "dr_du":
        edges = (*np.quantile(values, (.0, .2, .4, .6, .8)).tolist(), math.inf)
    elif attribute == "root_edge_distance":
        values = np.minimum(rows[:, 3], 1.0 - rows[:, 3])
        edges = (0.0, 1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 5.0e-2, 0.2, math.inf)
    else:
        raise ValueError(attribute)
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        mask = (values >= lower) & (values < upper)
        yield str(index + 1), lower, upper, rows[mask]


def clipped_fractional_flow(u: np.ndarray, law: str) -> np.ndarray:
    """Physical extension used by the native recurrence term outside [0, 1]."""
    return fractional_flow(np.clip(u, 0.0, 1.0), law)


def metrics(weights: np.ndarray, rows: np.ndarray, law: str, beta_max: float,
            beta_coordinate: str, recurrence_weight: float, recurrence_cap: float | None) -> dict[str, float | int]:
    count = int(rows.shape[0])
    if count == 0:
        return {"count": 0}
    prediction = network(weights, rows, beta_max, beta_coordinate)
    error = prediction - rows[:, 3]
    finite = np.isfinite(prediction)
    in_bounds = finite & (prediction >= 0.0) & (prediction <= 1.0)
    residual = prediction[in_bounds] - rows[in_bounds, 2] + rows[in_bounds, 1] * (fractional_flow(prediction[in_bounds], law) - rows[in_bounds, 0])
    valid = int(np.count_nonzero(in_bounds))
    result: dict[str, float | int] = {
        "count": count,
        "representation_limited_count": int(np.count_nonzero(rows[:, 5])),
        "physical_mse": float(np.mean(error * error)),
        "physical_mae": float(np.mean(np.abs(error))),
        "physical_rmse": float(np.sqrt(np.mean(error * error))),
        "physical_abs_error_q99": float(np.quantile(np.abs(error), 0.99)),
        "physical_max_abs_error": float(np.max(np.abs(error))),
        "prediction_min": float(np.nanmin(prediction)),
        "prediction_max": float(np.nanmax(prediction)),
        "nonfinite_prediction_count": count - int(np.count_nonzero(finite)),
        "out_of_bounds_count": count - valid,
        "in_bounds_fraction": valid / count,
        "physics_residual_rmse_in_bounds": float(np.sqrt(np.mean(residual * residual))) if valid else float("nan"),
        "physics_residual_max_abs_in_bounds": float(np.max(np.abs(residual))) if valid else float("nan"),
    }
    # This mirrors the optional native regular-law objective.  The exact root
    # derivative in column 4 converts recurrence error to the normalized-output
    # scale locally; its MSE can therefore be compared directly with normalized
    # supervised MSE.  Retain raw physical residual above for model-agnostic
    # diagnostics and avoid claiming an objective for the distinct IMP law.
    if law == "regular" and np.all(finite):
        normalized_error = 2.0 * error
        normalized_recurrence = 2.0 * (
            prediction - rows[:, 2] + rows[:, 1] *
            (clipped_fractional_flow(prediction, law) - rows[:, 0])) / rows[:, 4]
        normalized_mse = float(np.mean(normalized_error * normalized_error))
        normalized_recurrence_mse = float(np.mean(normalized_recurrence * normalized_recurrence))
        objective_recurrence = (recurrence_cap * np.tanh(normalized_recurrence / recurrence_cap)
                                if recurrence_cap is not None else normalized_recurrence)
        objective_recurrence_mse = float(np.mean(objective_recurrence * objective_recurrence))
        result["normalized_supervised_mse"] = normalized_mse
        result["root_jacobian_normalized_recurrence_mse"] = normalized_recurrence_mse
        result["objective_recurrence_mse"] = objective_recurrence_mse
        result["composite_objective_mse_equivalent"] = normalized_mse + recurrence_weight * objective_recurrence_mse
    else:
        result["normalized_supervised_mse"] = float("nan")
        result["root_jacobian_normalized_recurrence_mse"] = float("nan")
        result["objective_recurrence_mse"] = float("nan")
        result["composite_objective_mse_equivalent"] = float("nan")
    return result


def checkpoint_paths(run_dir: Path, explicit: list[Path] | None = None) -> list[Path]:
    if explicit:
        missing = [path for path in explicit if not path.is_file()]
        if missing:
            raise ValueError(f"Checkpoint paths do not exist: {missing}")
        return explicit
    paths = sorted(run_dir.glob("checkpoint_*.bin"))
    final = run_dir / "final_checkpoint.bin"
    if final.exists():
        paths.append(final)
    if not paths:
        raise ValueError(f"No checkpoints in {run_dir}")
    return paths


def parse_dataset(argument: str) -> tuple[str, Path]:
    name, separator, value = argument.partition("=")
    if not separator or not name or not value:
        raise argparse.ArgumentTypeError("Dataset must be NAME=PATH")
    return name, Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path,
                        help="Evaluate only this checkpoint; repeatable. Defaults to every checkpoint in --run-dir.")
    parser.add_argument("--dataset", action="append", type=parse_dataset, required=True, help="NAME=DATA.tptbin; repeatable")
    parser.add_argument("--law", choices=("regular", "imp"), required=True)
    parser.add_argument("--beta-max", type=float, default=150_000.0)
    parser.add_argument("--beta-coordinate", choices=("raw", "log1p"), default="raw")
    parser.add_argument("--recurrence-weight", type=float, default=0.0,
                        help="Report MSE + weight * normalized recurrence MSE for the regular-law objective.")
    parser.add_argument("--recurrence-cap", type=float,
                        help="Apply c*tanh(r/c) before the weighted recurrence MSE; omit for unbounded r.")
    args = parser.parse_args()
    if args.beta_max <= 0.0:
        raise ValueError("--beta-max must be positive")
    if args.recurrence_weight < 0.0 or not math.isfinite(args.recurrence_weight):
        raise ValueError("--recurrence-weight must be finite and nonnegative")
    if args.recurrence_cap is not None and (args.recurrence_cap <= 0.0 or not math.isfinite(args.recurrence_cap)):
        raise ValueError("--recurrence-cap must be finite and positive")

    data = [(name, path, load_dataset(path), sha256(path)) for name, path in args.dataset]
    summary_path = args.run_dir / "checkpoint_metrics.csv"
    conditional_path = args.run_dir / "conditional_metrics.csv"
    metric_fields = ["count", "representation_limited_count", "physical_mse", "physical_mae", "physical_rmse", "physical_abs_error_q99", "physical_max_abs_error", "prediction_min", "prediction_max", "nonfinite_prediction_count", "out_of_bounds_count", "in_bounds_fraction", "physics_residual_rmse_in_bounds", "physics_residual_max_abs_in_bounds", "normalized_supervised_mse", "root_jacobian_normalized_recurrence_mse", "objective_recurrence_mse", "composite_objective_mse_equivalent"]
    fields = ["checkpoint", "dataset", "checkpoint_sha256", "dataset_sha256", *metric_fields]
    conditional_fields = ["checkpoint", "dataset", "condition", "bin", "lower", "upper", *metric_fields]
    with summary_path.open("w", newline="") as summary, conditional_path.open("w", newline="") as conditional:
        summary_writer = csv.DictWriter(summary, fieldnames=fields)
        conditional_writer = csv.DictWriter(conditional, fieldnames=conditional_fields)
        summary_writer.writeheader()
        conditional_writer.writeheader()
        for checkpoint in checkpoint_paths(args.run_dir, args.checkpoint):
            weights = load_weights(checkpoint)
            checkpoint_hash = sha256(checkpoint)
            for name, data_path, rows, data_hash in data:
                common = {"checkpoint": checkpoint.name, "dataset": name, "checkpoint_sha256": checkpoint_hash, "dataset_sha256": data_hash}
                summary_writer.writerow(common | metrics(weights, rows, args.law, args.beta_max, args.beta_coordinate,
                                                         args.recurrence_weight, args.recurrence_cap))
                for condition in ("beta", "dr_du", "root_edge_distance"):
                    for label, lower, upper, subset in partitions(rows, condition):
                        conditional_writer.writerow({"checkpoint": checkpoint.name, "dataset": name, "condition": condition, "bin": label, "lower": lower, "upper": upper, **metrics(weights, subset, args.law, args.beta_max, args.beta_coordinate, args.recurrence_weight, args.recurrence_cap)})
            summary.flush()
            conditional.flush()

    manifest = {
        "law": args.law,
        "beta_max": args.beta_max,
        "beta_coordinate": args.beta_coordinate,
        "recurrence_weight": args.recurrence_weight,
        "recurrence_cap": args.recurrence_cap,
        "supported_architectures": {str(count): list(widths) for count, widths in ARCHITECTURES.items()},
        "checkpoint_format": "raw little-endian float64; layout inferred from parameter count",
        "evaluator": {"numpy_version": np.__version__},
        "datasets": [{"name": name, "path": str(path), "sha256": data_hash} for name, path, _, data_hash in data],
        "reports": [summary_path.name, conditional_path.name],
    }
    (args.run_dir / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {summary_path} and {conditional_path}")


if __name__ == "__main__":
    main()
