"""Explicit, synchronous training execution; independent of campaign queues.

No shell, automatic retry or scheduling. Restart requires a validated parent.
This module does not import or modify either live campaign controller.
"""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import experiments


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def host_observation():
    # Avoid platform.platform()/uname() on Windows: they may spawn a shell or WMI.
    if sys.platform == "win32":
        version = str(sys.getwindowsversion())
        machine = os.environ.get("PROCESSOR_ARCHITECTURE")
    else:
        version = os.uname().release
        machine = os.uname().machine
    return {
        "system": sys.platform,
        "os_version": version,
        "machine": machine,
        "python": platform.python_version(),
        "logical_cpus": os.cpu_count(),
    }


def write_record(path, value):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def child_environment(directories, inherited=None, system=None):
    """Change only the child's loader search path; never dump the environment."""
    env = dict(os.environ if inherited is None else inherited)
    system = sys.platform if system is None else system
    key = (
        "PATH"
        if system == "win32"
        else ("DYLD_LIBRARY_PATH" if system == "darwin" else "LD_LIBRARY_PATH")
    )
    if directories:
        separator = ";" if system == "win32" else ":"
        env[key] = separator.join([*directories, *([env[key]] if env.get(key) else [])])
    return env, key


def source_observation(root):
    """Observe available source; do not imply it produced the supplied binary."""
    paths = set((root / "training/native_lm/include").rglob("*.hpp"))
    for adapter in ("bratu", "twophasetransport"):
        directory = root / "training" / adapter
        paths.update((directory / "src").glob("*.cpp"))
        paths.update((directory / "src").glob("*.hpp"))
        if (directory / "CMakeLists.txt").is_file():
            paths.add(directory / "CMakeLists.txt")
    paths.update(
        root / "training" / name
        for name in ("experiments.py", "experiments.json", "execution.py", "restart.py")
        if (root / "training" / name).is_file()
    )
    files = {p.relative_to(root).as_posix(): experiments.sha(p) for p in sorted(paths)}
    revision = None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
            **(
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if os.name == "nt"
                else {}
            ),
        )
        if result.returncode == 0:
            revision = result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {
        "git_revision": revision,
        "file_sha256": files,
        "scope": "Checkout observation, not binary build attestation; file hashes include local edits",
    }


def revalidate(plan, root):
    """Reconstruct argv from the supported contract; never execute arbitrary plan argv."""
    cold = plan.get("schema_version") == 1 and plan.get("mode") == "cold-start plan only"
    restart = plan.get("schema_version") == 2 and plan.get("mode") == "restart plan only" and plan.get("start_mode") in ("resume", "continuation")
    if not cold and not restart:
        raise ValueError(
            "Unsupported plan; unvalidated resume/continuation are disabled"
        )
    command = plan["command_argv"]
    if not command or len(command) % 2 != 1:
        raise ValueError("Malformed native argument array")
    flags = dict(zip(command[1::2], command[2::2]))
    fresh = experiments.make_plan(
        plan["config"],
        Path(command[0]),
        Path(flags["--run-dir"]),
        weight=plan["objective_weight"],
        threads=plan["resolved_settings"]["threads"],
        rows=plan["resolved_settings"]["training_rows"],
        runtime_dirs=[Path(p) for p in plan["runtime_path_prepend"]],
        root=root,
        accepted=plan["config"]["limits"]["accepted"],
        proposed=plan["config"]["limits"]["proposed"],
        resume_from=Path(plan["lineage"]["parent_run"]) if restart and plan["start_mode"] == "resume" else None,
        continue_from=Path(plan["lineage"]["parent_run"]) if restart and plan["start_mode"] == "continuation" else None,
        parent_stopped=plan["lineage"]["parent_stopped_assertion"] if restart else False,
    )
    if fresh != plan:
        raise ValueError(
            "Plan changed or an input/executable no longer matches its hash"
        )
    return Path(flags["--run-dir"])


def collect_outputs(directory, adapter, successful):
    required = ["final_checkpoint.bin", "optimizer_trace.csv"]
    required += ["result.json"] if adapter == "bratu" else ["final_model.pt"]
    if successful:
        for name in required:
            path = directory / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(
                    "Trainer exited zero but required output is missing/empty: " + name
                )
        parameters = 1341 if adapter == "bratu" else 1361
        if (directory / "final_checkpoint.bin").stat().st_size != parameters * 8:
            raise ValueError(
                "Final checkpoint size does not match the compiled architecture"
            )
        from restart import read_state
        snapshots = list(directory.glob("optimizer_state_[01].bin"))
        if not snapshots:
            raise ValueError("Trainer exited zero without complete-state recovery; rebuild the current native adapter")
        snapshot = max(snapshots, key=lambda p: read_state(p, adapter)["iteration"])
        if snapshot.read_bytes()[96 + parameters*8:-8] != (directory / "final_checkpoint.bin").read_bytes():
            raise ValueError("Final checkpoint differs from the complete-state best weights")
    names = [*required, "trainer_stdout.log", "trainer_stderr.log",
             *(p.name for p in directory.glob("optimizer_state_[01].bin"))]
    return {
        name: {
            "sha256": experiments.sha(directory / name),
            "bytes": (directory / name).stat().st_size,
        }
        for name in names
        if (directory / name).is_file()
    }


def execute_plan(plan, root=experiments.ROOT, timeout=None):
    """Execute one new run. Existing run directories are never opened for writing."""
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("Timeout must be positive and finite")
    started = time.perf_counter()
    started_utc = utc_now()
    directory = revalidate(plan, root)
    observation = source_observation(root)
    # Exclusive mkdir is the run claim: even two concurrent callers cannot reuse it.
    # Parent creation is deliberately explicit, so typos do not create directory trees.
    directory.mkdir(parents=False, exist_ok=False)
    record_path = directory / "execution.json"
    environment, loader_key = child_environment(plan["runtime_path_prepend"])
    record = {
        "schema_version": 1,
        "mode": plan.get("start_mode", "cold-start"),
        "state": "preparing",
        "started_utc": started_utc,
        "controller_pid": os.getpid(),
        "command_argv": plan["command_argv"],
        "lineage": plan.get("lineage"),
        "host": host_observation(),
        "runtime": {
            "loader_variable": loader_key,
            "prepended_directories": plan["runtime_path_prepend"],
            "inherited_thread_settings": {
                k: environment[k]
                for k in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "MKL_DYNAMIC",
                    "OMP_DYNAMIC",
                )
                if k in environment
            },
        },
        "source_observation": observation,
        "timing_scope": "execution_total_seconds includes revalidation, provenance and output hashing; child_wall_seconds includes native startup, training, validation, checkpoint I/O and export; neither includes label generation or prior runs",
        "accuracy_verification": "Not performed; process/output checks are not numerical validation",
    }
    process = None
    child_started = None
    child_seconds = None
    error = None
    try:
        write_record(directory / "execution_plan.json", plan)
        write_record(record_path, record)
        from restart import prepare_inputs
        prepare_inputs(plan, directory)
        with (directory / "trainer_stdout.log").open("wb") as out, (
            directory / "trainer_stderr.log"
        ).open("wb") as err:
            child_started = time.perf_counter()
            process = subprocess.Popen(
                plan["command_argv"],
                cwd=root,
                env=environment,
                stdout=out,
                stderr=err,
                stdin=subprocess.DEVNULL,
                shell=False,
                **(
                    {"creationflags": subprocess.CREATE_NO_WINDOW}
                    if os.name == "nt"
                    else {}
                ),
            )
            record.update(state="running", child_pid=process.pid)
            write_record(record_path, record)
            returncode = process.wait(timeout=timeout)
            child_seconds = time.perf_counter() - child_started
            record["returncode"] = returncode
            record["state"] = "completed" if returncode == 0 else "failed"
        record["outputs"] = collect_outputs(
            directory, plan["config"]["adapter"], returncode == 0
        )
    except BaseException as caught:
        error = caught
        record["state"] = (
            "interrupted" if isinstance(caught, KeyboardInterrupt) else "failed"
        )
        record["error"] = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if process is not None:
            record["returncode"] = process.returncode
        if child_started is not None:
            record["child_wall_seconds"] = (
                child_seconds
                if child_seconds is not None
                else time.perf_counter() - child_started
            )
        if "outputs" not in record:
            try:
                record["outputs"] = collect_outputs(
                    directory, plan["config"]["adapter"], False
                )
            except OSError as output_error:
                record["output_inventory_error"] = str(output_error)
        record.update(
            finished_utc=utc_now(),
            execution_total_seconds=time.perf_counter() - started,
        )
        prior = (plan.get("lineage") or {}).get("prior_child_wall_seconds", 0.0)
        child = record.get("child_wall_seconds", 0.0)
        record["cost_ledger"] = {
            "prior_child_wall_seconds": prior,
            "this_child_wall_seconds": child,
            "cumulative_child_wall_seconds": prior + child if prior is not None else None,
            "complete": prior is not None,
            "scope": "Inclusive native child wall time across linked attempts/stages; includes replayed/lost work when measured; excludes labels, compilation and launcher overhead",
        }
        write_record(record_path, record)
    if error is not None:
        raise error
    return record
