"""Persistent fuzz worker: one JSON job per stdin line, one JSON outcome per protocol line.

The protocol uses a duplicate of the original stdout and fd 1 is redirected to stderr, so nothing an importer
prints can corrupt it. A job is a ``translate`` (the source is written to a scratch file with the job's
extension, because importers infer the stage from it) or a ``preprocess`` of a unit in a memory file system.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

from .outcomes import classify, signature

SCRATCH_ROOT = "/fuzz"


def _translate(job, scratch):
    from crosstl import translate

    path = os.path.join(scratch, "input" + job["extension"])
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(job["source"])
    translate(
        path,
        backend=job["backend"],
        source_backend=job["importer"],
        format_output=False,
    )


def _preprocess(job):
    from crosstl.translator.preprocessor import MemoryFileSystem, preprocess

    unit = job["unit"]
    files = {f"{SCRATCH_ROOT}/{name}": text for name, text in unit["headers"].items()}
    main = f"{SCRATCH_ROOT}/main.src"
    files[main] = unit["main"]
    preprocess(
        unit["main"],
        unit["dialect"],
        path=main,
        include_paths=(SCRATCH_ROOT,),
        system_paths=(SCRATCH_ROOT,),
        file_system=MemoryFileSystem(files),
    )


def run_job(job, scratch):
    """Run one job and describe its outcome; never raises for an importer failure."""
    started = time.perf_counter()
    error = None
    try:
        if job["kind"] == "preprocess":
            _preprocess(job)
        else:
            _translate(job, scratch)
    except (KeyboardInterrupt, GeneratorExit):
        raise
    except (
        BaseException
    ) as caught:  # noqa: BLE001 - every importer failure is classified, never raised
        error = caught
    status = classify(error)
    outcome = {"status": status, "seconds": round(time.perf_counter() - started, 4)}
    if error is not None:
        outcome.update(
            exception=type(error).__name__,
            message=str(error)[:300],
            code=getattr(error, "project_diagnostic_code", None) or None,
        )
    if status == "crash":
        outcome["signature"] = signature(job["importer"], job["backend"], status, error)
    return outcome


def main():
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    import crosstl  # noqa: F401 - import before reporting ready, so start-up is not timed per job

    scratch = tempfile.mkdtemp(prefix="crosstl-fuzz-")
    try:
        protocol.write(json.dumps({"ready": True}) + "\n")
        protocol.flush()
        for line in sys.stdin:
            job = json.loads(line)
            protocol.write(json.dumps(run_job(job, scratch)) + "\n")
            protocol.flush()
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
