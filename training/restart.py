"""Validate checkpoints and prepare training continuation or optimizer recovery."""
import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
import struct

import experiments


def checksum(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return value


def read_state(path, adapter):
    data = path.read_bytes()
    count = 1361 if adapter == "transport" else 1341
    if len(data) != 104 + 16 * count or data[:8] != b"NLMR0001":
        raise ValueError("Complete recovery length/magic mismatch: " + str(path))
    if struct.unpack_from("<Q", data, 8)[0] != count or struct.unpack_from("<Q", data, len(data)-8)[0] != checksum(data[:-8]):
        raise ValueError("Complete recovery architecture/checksum mismatch: " + str(path))
    values = struct.unpack_from("<10d", data, 16)
    weights = struct.unpack_from(f"<{count * 2}d", data, 96)
    if not all(math.isfinite(x) and x >= 0 for x in values) or not all(math.isfinite(x) for x in weights):
        raise ValueError("Nonfinite/negative complete recovery state")
    if any(x != int(x) or x > 2147483647 for x in values[:3]) or not 0 <= values[2] <= values[1] <= values[0] or values[3] <= 0:
        raise ValueError("Invalid complete recovery counters/damping")
    keys = ("iteration", "accepted_updates", "validation_failures", "lambda", "objective_loss",
            "training_mse", "validation_mse", "best_validation_mse", "gradient_norm", "phase_seconds")
    state = dict(zip(keys, values))
    for key in keys[:3]:
        state[key] = int(state[key])
    return state


def trace_prefix(path, state):
    """Copy only committed history; preserve parent bytes, including any tail."""
    lines = path.read_bytes().splitlines(keepends=True)
    if not lines:
        raise ValueError("Empty parent trace")
    header = next(csv.reader([lines[0].decode("utf-8")]))
    iteration_col, accepted_col = header.index("iteration"), header.index("accepted_updates")
    keep, previous, accepted = [lines[0]], None, 0
    for line in lines[1:]:
        if previous == state["iteration"]:
            break
        row = next(csv.reader([line.decode("utf-8")]))
        iteration = int(row[iteration_col])
        if iteration > state["iteration"]:
            break
        if previous is not None and iteration != previous + 1:
            raise ValueError("Gap in committed parent optimizer history")
        if previous is None and iteration not in (0, 1):
            raise ValueError("Parent history must start at proposal 0 or 1")
        accepted = int(row[accepted_col])
        if not 0 <= accepted <= iteration:
            raise ValueError("Invalid accepted counter in parent trace")
        previous = iteration
        keep.append(line)
    if (previous is None and state["iteration"] != 0) or (previous is not None and previous != state["iteration"]) or accepted != state["accepted_updates"]:
        raise ValueError("Parent history does not reach the committed recovery state")
    if not keep[-1].endswith(b"\n"):
        raise ValueError("Committed parent trace has a partial final row")
    return b"".join(keep)


def signature(config):
    # Budgets may be extended; dataset identity is checked by bytes, not paths.
    return {k: v for k, v in config.items() if k not in ("limits", "dataset_root", "default_rows")}


def prefix_equal(old, new, adapter, rows):
    header, stride = (24, 48) if adapter == "transport" else (16, 32)
    with old.open("rb") as a, new.open("rb") as b:
        a.seek(header); b.seek(header)
        left = rows * stride
        while left:
            n = min(left, 1024 * 1024)
            x, y = a.read(n), b.read(n)
            if len(x) != n or x != y:
                return False
            left -= n
    return True


def inspect_parent(parent, mode, config, data, executable, weight, threads, root, parent_stopped=False):
    parent = parent.resolve()
    plan_path, record_path = parent / "execution_plan.json", parent / "execution.json"
    old, record = json.loads(plan_path.read_text()), json.loads(record_path.read_text())
    if old.get("schema_version") not in (1, 2) or old.get("experiment") != config["id"]:
        raise ValueError("Parent must be a recorded run of the same experiment")
    terminal = record.get("state") in ("completed", "failed", "interrupted")
    if not terminal and not (parent_stopped and record.get("state") in ("preparing", "running")):
        raise ValueError("Parent is not terminal; stop it first, then explicitly assert --parent-stopped for a stale record")
    if signature(old["config"]) != signature(config) or old["objective_weight"] != weight:
        raise ValueError("Parent physics, architecture, transform, seed or objective differs")
    if old["inputs"]["validation"]["sha256"] != data["validation"]["sha256"]:
        raise ValueError("Validation data must stay identical across resume/continuation")
    if old["config"]["limits"]["block_size"] != config["limits"]["block_size"] or old["resolved_settings"]["threads"] != threads:
        raise ValueError("Restart requires the parent's block size and thread count")
    if mode == "resume":
        if old["executable_sha256"] != experiments.sha(executable):
            raise ValueError("Resume requires the identical native executable")
        if old["inputs"]["train"]["sha256"] != data["train"]["sha256"] or old["config"]["limits"]["validation_patience"] != config["limits"]["validation_patience"]:
            raise ValueError("Resume requires identical training data and validation patience")
    else:
        if record.get("state") != "completed":
            raise ValueError("Dataset continuation requires a completed parent with a selected best model")
        old_data = experiments.check_data(old["config"], root, old["resolved_settings"]["training_rows"])
        if old_data["train"]["sha256"] != old["inputs"]["train"]["sha256"] or data["train"]["rows"] <= old_data["train"]["rows"]:
            raise ValueError("Continuation requires a strictly larger training dataset and an unchanged parent prefix")
        if not prefix_equal(Path(old_data["train"]["path"]), Path(data["train"]["path"]), config["adapter"], old_data["train"]["rows"]):
            raise ValueError("Larger training dataset does not preserve the parent prefix")
    candidates, rejected = [], []
    for path in sorted(parent.glob("optimizer_state_[01].bin")):
        try:
            state = read_state(path, config["adapter"])
            prefix = trace_prefix(parent / "optimizer_trace.csv", state)
            candidates.append((state["iteration"], path, state, prefix))
        except (ValueError, OSError, IndexError, UnicodeError) as error:
            rejected.append({"path": path.name, "reason": str(error)})
    if not candidates:
        raise ValueError("No optimizer-state checkpoint matches the recorded history; weights alone cannot resume the optimizer")
    _, snapshot, state, prefix = max(candidates, key=lambda x: x[0])
    retained = {"checkpoint_000000.bin"}
    for row in csv.DictReader(io.StringIO(prefix.decode("utf-8"))):
        filename = row.get("checkpoint", "")
        if filename:
            if not re.fullmatch(r"(?:checkpoint_\d+|milestone_\d+)\.bin", filename):
                raise ValueError("Unexpected retained checkpoint filename in parent history")
            retained.add(filename)
    retained_checkpoints = {name: experiments.sha(parent / name) for name in sorted(retained)}
    if mode == "resume" and (config["limits"]["accepted"] < state["accepted_updates"] or config["limits"]["proposed"] < state["iteration"]):
        raise ValueError("Resume budgets are totals and cannot precede committed counters")
    model = parent / "final_checkpoint.bin"
    if mode == "continuation":
        expected = record.get("outputs", {}).get(model.name, {}).get("sha256")
        if not expected or experiments.sha(model) != expected:
            raise ValueError("Parent best model differs from its completed execution record")
        expected_weights = snapshot.read_bytes()[96 + (1361 if config["adapter"] == "transport" else 1341) * 8:-8]
        if model.read_bytes() != expected_weights:
            raise ValueError("Final model does not match the committed best-validation weights")
    cost = record.get("cost_ledger", {}).get("cumulative_child_wall_seconds", record.get("child_wall_seconds"))
    if cost is not None and (not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost < 0):
        raise ValueError("Invalid parent cumulative wall cost")
    old_lineage = old.get("lineage") or {}
    stage = old_lineage.get("dataset_stage", 0) + (mode == "continuation")
    return dict(parent_run=str(parent), parent_plan_sha256=experiments.sha(plan_path),
        parent_execution_sha256=experiments.sha(record_path), parent_stopped_assertion=parent_stopped,
        dataset_stage=int(stage), recovery_file=str(snapshot), recovery_sha256=experiments.sha(snapshot),
        recovery_state=state, rejected_recovery_slots=rejected,
        trace_sha256=experiments.sha(parent / "optimizer_trace.csv"),
        trace_prefix_sha256=hashlib.sha256(prefix).hexdigest(), trace_prefix_bytes=len(prefix),
        retained_checkpoints=retained_checkpoints,
        selected_model=str(model) if mode == "continuation" else None,
        selected_model_sha256=experiments.sha(model) if mode == "continuation" else None,
        prior_child_wall_seconds=cost,
        accepted_before_dataset_stage=old_lineage.get("accepted_before_dataset_stage", 0) +
            (state["accepted_updates"] if mode == "continuation" else 0))


def prepare_inputs(plan, directory):
    """Create immutable local restart inputs after the exclusive directory claim."""
    lineage = plan.get("lineage")
    if not lineage:
        return
    if plan["start_mode"] == "resume":
        source = Path(lineage["recovery_file"])
        data = source.read_bytes()
        if hashlib.sha256(data).hexdigest() != lineage["recovery_sha256"]:
            raise ValueError("Recovery changed during preparation")
        with (directory / "resume_input.bin").open("xb") as stream:
            stream.write(data)
        prefix = trace_prefix(Path(lineage["parent_run"]) / "optimizer_trace.csv", lineage["recovery_state"])
        if hashlib.sha256(prefix).hexdigest() != lineage["trace_prefix_sha256"]:
            raise ValueError("Committed parent history changed during preparation")
        with (directory / "optimizer_trace.csv").open("xb") as stream:
            stream.write(prefix)
        for filename, expected in lineage["retained_checkpoints"].items():
            data = (Path(lineage["parent_run"]) / filename).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("Retained checkpoint changed during preparation")
            with (directory / filename).open("xb") as stream:
                stream.write(data)
    else:
        data = Path(lineage["selected_model"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != lineage["selected_model_sha256"]:
            raise ValueError("Selected parent model changed during preparation")
        with (directory / "initial_checkpoint.bin").open("xb") as stream:
            stream.write(data)
