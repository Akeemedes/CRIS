"""Experiment configuration, read-only checks/plans and explicit fresh training.

Only the explicit train command launches a process and writes a new run.
Native executables and runtime directories are explicit user inputs, not
workstation paths embedded in scientific configuration.
"""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path, PureWindowsPath
import struct

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = Path(__file__).with_suffix(".json")
ROW_NAMES = {
    62400: "62k",
    125000: "125k",
    250000: "250k",
    500000: "500k",
    1000000: "1m",
}


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def repository_path(root, value):
    relative = Path(value.replace("\\", "/"))
    if relative.is_absolute() or PureWindowsPath(value).drive or ".." in relative.parts:
        raise ValueError("Dataset paths must be repository-relative without '..'")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("Dataset path escapes the repository")
    return path


def validate_config(config):
    expected = {
        "id",
        "adapter",
        "physics",
        "dataset_root",
        "architecture",
        "beta_coordinate",
        "beta_max",
        "physical_mse_multiplier",
        "seed",
        "default_rows",
        "validation_rows",
        "residual",
        "limits",
    }
    if set(config) != expected:
        raise ValueError(
            "Missing or unknown experiment fields: " + str(set(config) ^ expected)
        )
    adapter = config["adapter"]
    if adapter not in ("transport", "bratu"):
        raise ValueError("Unsupported native adapter")
    inputs = 3 if adapter == "transport" else 2
    if config["architecture"] != [inputs, 20, 20, 20, 20, 1]:
        raise ValueError(
            "Architecture is not supported by the current compiled adapter"
        )
    if config["seed"] != 20260904 or config["beta_coordinate"] != "log1p":
        raise ValueError(
            "This validated cold-start profile uses seed 20260904 and log1p"
        )
    limits = config["limits"]
    if set(limits) != {
        "accepted",
        "proposed",
        "validation_patience",
        "threads",
        "block_size",
    }:
        raise ValueError("Missing or unknown optimizer limits")
    for value in [*limits.values(), config["default_rows"], config["validation_rows"]]:
        if type(value) is not int or value < 1:
            raise ValueError(
                "Row counts and optimizer limits must be positive integers"
            )
    if limits["accepted"] > limits["proposed"]:
        raise ValueError(
            "Proposed-step budget cannot be smaller than accepted-step budget"
        )
    if adapter == "transport":
        law = config["physics"].get("law")
        expected_physics = {
            "law": law,
            "M": 0.4,
            "n_w": 2,
            "n_nw": 0.2 if law == "imp" else 2,
            "Ng": 0,
            "dispersion": False,
        }
        if law not in ("regular", "imp") or config["physics"] != expected_physics:
            raise ValueError("Unsupported transport physics for this native adapter")
        if config["beta_max"] != 1e6 or config["physical_mse_multiplier"] != 0.25:
            raise ValueError("Transport data/normalization profile mismatch")
        if config["default_rows"] not in ROW_NAMES:
            raise ValueError("Unknown frozen training prefix")
        if config["residual"] != "tanh(2R / R_u(label)) with cap 1":
            raise ValueError("Transport residual contract mismatch")
    else:
        if config["physics"] != {
            "equation": "a-u+beta*exp(u)=0",
            "branch": "lower",
            "a_range": [-2, 6],
            "beta_range": [0, 1],
        }:
            raise ValueError("Bratu native adapter requires the lower-branch profile")
        if config["beta_max"] != 1 or config["physical_mse_multiplier"] != 20.25:
            raise ValueError("Bratu normalization profile mismatch")
        if config["residual"] != "tanh(R / 4.5) with cap 1":
            raise ValueError("Bratu residual contract mismatch")
    return config


def load_catalogue(path=CATALOGUE):
    data = json.loads(path.read_text())
    if set(data) != {"schema_version", "experiments"} or data["schema_version"] != 1:
        raise ValueError("Unsupported catalogue schema")
    result = {}
    for config in data["experiments"]:
        validate_config(config)
        if (
            not isinstance(config["id"], str)
            or not config["id"]
            or config["id"] in result
        ):
            raise ValueError("Invalid or duplicate experiment ID")
        result[config["id"]] = config
    return result


def check_binary(path, adapter, rows, expected_hash):
    header_bytes, fields, magic = (
        (24, 6, b"TPTBIN01") if adapter == "transport" else (16, 4, b"BRATUD01")
    )
    with path.open("rb") as stream:
        header = stream.read(header_bytes)
    if (
        len(header) != header_bytes
        or header[:8] != magic
        or struct.unpack_from("<Q", header, 8)[0] != rows
    ):
        raise ValueError("Dataset header/row count mismatch: " + str(path))
    if adapter == "transport" and struct.unpack_from("<II", header, 16) != (fields, 0):
        raise ValueError("Unsupported packed transport layout")
    if path.stat().st_size != header_bytes + rows * fields * 8:
        raise ValueError("Dataset length mismatch: " + str(path))
    actual = sha(path)
    if actual != expected_hash:
        raise ValueError("Dataset checksum mismatch: " + str(path))
    return {"path": str(path), "rows": rows, "sha256": actual}


def check_data(config, root=ROOT, rows=None):
    validate_config(config)
    rows = config["default_rows"] if rows is None else rows
    directory = repository_path(root, config["dataset_root"])
    adapter = config["adapter"]
    if adapter == "transport":
        if rows not in ROW_NAMES:
            raise ValueError("Select one of the frozen transport prefixes")
        manifest_path = directory / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        labels_path = directory / "train.csv.manifest.json"
        labels = json.loads(labels_path.read_text())
        physics = config["physics"]
        expected = {k: physics[k] for k in ("M", "n_w", "n_nw", "Ng")}
        if (
            manifest["law"] != physics["law"]
            or labels["law"] != physics["law"]
            or manifest["beta_max"] != config["beta_max"]
            or labels["beta_max"] != config["beta_max"]
            or labels["fractional_flow"] != expected
            or labels["recurrence"] != "u - u_old + beta * (f(u) - a) = 0"
        ):
            raise ValueError("Dataset physics does not match the selected experiment")
        train = directory / f"train_{ROW_NAMES[rows]}.tptbin"
        validation = directory / "validation.tptbin"
        products = manifest["packed_products"]
        hashes = [
            products[f"train_{rows}"]["dataset_sha256"],
            products["validation"]["dataset_sha256"],
        ]
        extra = {"label_manifest_sha256": sha(labels_path)}
    else:
        if rows != config["default_rows"]:
            raise ValueError("This Bratu bundle has a fixed training split")
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest["physics"] != config["physics"]["equation"]
            or manifest["branch"] != "lower, continued from beta=0; Lambert W0"
            or manifest["a_range"] != [-2, 6]
            or manifest["beta_range"] != [0, 1]
            or manifest["physical_MSE_multiplier"] != 20.25
        ):
            raise ValueError("Bratu dataset branch/normalization mismatch")
        train, validation = directory / "train.bin", directory / "validation.bin"
        hashes = [
            manifest["products"][key]["sha256"] for key in ("train", "validation")
        ]
        extra = {}
    if train.resolve() == validation.resolve():
        raise ValueError("Training and validation files must be distinct")
    return {
        "train": check_binary(train, adapter, rows, hashes[0]),
        "validation": check_binary(
            validation, adapter, config["validation_rows"], hashes[1]
        ),
        "dataset_manifest_sha256": sha(manifest_path),
        **extra,
    }


def command_arguments(config, data, executable, run_dir, weight=0.0, threads=None):
    validate_config(config)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("Residual weight must be finite and nonnegative")
    limits = config["limits"]
    threads = limits["threads"] if threads is None else threads
    if type(threads) is not int or threads < 1:
        raise ValueError("Threads must be a positive integer")
    command = [
        str(executable),
        "--train",
        data["train"]["path"],
        "--validation",
        data["validation"]["path"],
        "--run-dir",
        str(run_dir),
        "--threads",
        str(threads),
        "--block-size",
        str(limits["block_size"]),
        "--portable-recovery",
        "1",
    ]
    if config["adapter"] == "bratu":
        command += [
            "--weight",
            str(weight),
            "--max-accepted",
            str(limits["accepted"]),
            "--max-proposals",
            str(limits["proposed"]),
            "--patience",
            str(limits["validation_patience"]),
        ]
    else:
        command += [
            "--beta-max",
            str(config["beta_max"]),
            "--beta-coordinate",
            config["beta_coordinate"],
            "--objective",
            "mse+recurrence" if weight else "mse",
            "--seed",
            str(config["seed"]),
            "--normal-equation-kernel",
            "mkl-syrk",
            "--device",
            "cpu",
            "--damping",
            "identity",
            "--initialization",
            "nguyen-widrow",
            "--lambda",
            "0.001",
            "--lambda-decrease",
            "2",
            "--lambda-increase",
            "10",
            "--lambda-max",
            "1e10",
            "--max-iterations",
            str(limits["proposed"]),
            "--max-accepted-updates",
            str(limits["accepted"]),
            "--max-validation-fail",
            str(limits["validation_patience"]),
            "--export-package",
            str(run_dir / "final_model.pt"),
        ]
        if weight:
            command += [
                "--recurrence-weight",
                str(weight),
                "--recurrence-cap",
                "1",
                "--nonwetting-exponent",
                str(config["physics"]["n_nw"]),
            ]
    return command


def make_plan(
    config,
    executable,
    run_dir,
    weight=None,
    threads=None,
    rows=None,
    runtime_dirs=(),
    root=ROOT,
    accepted=None,
    proposed=None,
    resume_from=None,
    continue_from=None,
    parent_stopped=False,
):
    if resume_from is not None and continue_from is not None:
        raise ValueError("Select resume or dataset continuation, not both")
    config = copy.deepcopy(config)
    parent = resume_from if resume_from is not None else continue_from
    mode = "resume" if resume_from is not None else ("continuation" if continue_from is not None else "cold-start")
    old = json.loads((parent.resolve() / "execution_plan.json").read_text()) if parent is not None else None
    if weight is None:
        weight = old["objective_weight"] if old else 0.0
    if threads is None and old:
        threads = old["resolved_settings"]["threads"]
    if rows is None and old and mode == "resume":
        rows = old["resolved_settings"]["training_rows"]
    for key, value in (("accepted", accepted), ("proposed", proposed)):
        if value is not None:
            config["limits"][key] = value
        elif old and mode == "resume":
            config["limits"][key] = old["config"]["limits"][key]
    validate_config(config)
    if parent_stopped and parent is None:
        raise ValueError("--parent-stopped requires a parent run")
    executable, run_dir = executable.resolve(), run_dir.resolve()
    if not executable.is_file():
        raise ValueError("Provide an existing native executable explicitly")
    if run_dir.exists():
        raise ValueError(
            "Cold-start plan requires a new run directory; never reuse campaign outputs"
        )
    data = check_data(config, root, rows)
    for directory in runtime_dirs:
        if not directory.is_dir():
            raise ValueError("Missing runtime library directory: " + str(directory))
    command = command_arguments(config, data, executable, run_dir, weight, threads)
    lineage = None
    if parent is not None:
        from restart import inspect_parent
        lineage = inspect_parent(parent, mode, config, data, executable, weight,
                                 config["limits"]["threads"] if threads is None else threads, root, parent_stopped)
        command += (["--resume-state", str(run_dir / "resume_input.bin")] if mode == "resume" else
                    ["--initial-checkpoint", str(run_dir / "initial_checkpoint.bin")])
    result = {
        "schema_version": 1,
        "experiment": config["id"],
        "mode": "cold-start plan only",
        "launch_supported": True,
        "command_argv": command,
        "runtime_path_prepend": [str(p.resolve()) for p in runtime_dirs],
        "executable_sha256": sha(executable),
        "config": config,
        "inputs": data,
        "resolved_settings": {
            "training_rows": data["train"]["rows"],
            "threads": config["limits"]["threads"] if threads is None else threads,
            "precision": "float64",
            "device": "cpu",
        },
        "objective_weight": weight,
        "physical_mse_multiplier": config["physical_mse_multiplier"],
        "notes": [
            "No process launched and no output directory created",
            "Executable hash records identity, not validation of a different build",
            "No resume, data-size continuation or scheduling is implied",
            "Runtime directories are explicit; no environment is modified",
        ],
    }
    if lineage:
        result.update(schema_version=2, mode="restart plan only", start_mode=mode, lineage=lineage)
        result["notes"] = ["No process launched; parent run is read-only",
                           "Resume restores full state; continuation resets optimizer with selected best weights",
                           "Inclusive wall cost includes prior attempts; native phase timings must not be added to it"]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("list")
    for action in ("show", "check", "plan", "train"):
        child = commands.add_parser(action)
        child.add_argument("experiment")
        if action != "show":
            child.add_argument("--rows", type=int)
        if action in ("plan", "train"):
            child.add_argument("--executable", type=Path, required=True)
            child.add_argument("--run-dir", type=Path, required=True)
            child.add_argument("--weight", type=float, default=None)
            child.add_argument("--threads", type=int)
            child.add_argument("--runtime-dir", type=Path, action="append", default=[])
            child.add_argument("--max-accepted", type=int, help="Total accepted budget for resume; new-stage budget otherwise")
            child.add_argument("--max-proposed", type=int, help="Separate total proposal safeguard")
            start = child.add_mutually_exclusive_group()
            start.add_argument("--resume-from", type=Path, help="Recorded parent run; restore complete optimizer state")
            start.add_argument("--continue-from", type=Path, help="Completed parent run; best weights on a strictly larger dataset")
            child.add_argument("--parent-stopped", action="store_true", help="Explicit assertion that a parent with a stale running record has stopped")
        if action == "train":
            child.add_argument(
                "--timeout-seconds",
                type=float,
                help="Optional wall-time limit; expiry stops the child and records failure",
            )
    args = parser.parse_args()
    try:
        catalogue = load_catalogue(args.catalogue)
        if args.action == "list":
            for key, config in catalogue.items():
                print(
                    f"{key}: {config['adapter']}, {config['default_rows']} training / {config['validation_rows']} validation rows"
                )
            return
        if args.experiment not in catalogue:
            raise ValueError("Unknown experiment: " + args.experiment)
        config = catalogue[args.experiment]
        if args.action == "show":
            result = config
        elif args.action == "check":
            result = {
                "experiment": args.experiment,
                "passed": True,
                "inputs": check_data(config, rows=args.rows),
            }
        else:
            result = make_plan(
                config,
                args.executable,
                args.run_dir,
                args.weight,
                args.threads,
                args.rows,
                args.runtime_dir,
                accepted=args.max_accepted,
                proposed=args.max_proposed,
                resume_from=args.resume_from,
                continue_from=args.continue_from,
                parent_stopped=args.parent_stopped,
            )
            if args.action == "train":
                from execution import execute_plan

                result = execute_plan(result, timeout=args.timeout_seconds)
                print(json.dumps(result, indent=2, allow_nan=False))
                if result["state"] != "completed":
                    parser.exit(
                        1, "Native training failed; inspect execution.json and stderr\n"
                    )
                return
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
