"""Persistent translation worker: one JSON job per stdin line, one JSON result per protocol line.

The protocol uses a duplicate of the original stdout, and fd 1 is redirected to stderr, so nothing the
translator prints can corrupt it. Only ``crosstl.translate_artifacts`` is timed; process start-up and imports are not
part of a unit's latency.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

# Text an emitter leaves where it lost a construct; counted on the raw (unformatted) output.
FALLBACK_MARKERS = (
    r"^\s*; WARNING",
    r"/\*\s*unsupported",
    r"\bplaceholder",
    r"// WARNING",
    r"#\s*CrossGL .*placeholder",
    r"compile-only placeholder",
    r"using default value",
)


def markers(text):
    return sum(
        len(re.findall(pattern, text, re.M | re.I)) for pattern in FALLBACK_MARKERS
    )


def translate_job(job, translate_artifacts):
    """Translate one job; ``seconds`` is wall time and ``cpuSeconds`` the worker's CPU time, which a loaded
    machine inflates far less."""
    return _translate_job(
        job, translate_artifacts, time.perf_counter(), time.process_time()
    )


def _translate_job(job, translate_artifacts, started, cpu_started):
    """``crosstl.translate_artifacts`` writes a target whose compilation unit holds one entry per stage (GLSL) one
    artifact per entry, importing the source once; the cell is translated only when every entry is, and an entry's
    failure is the cell's."""
    options = {
        "include_paths": job.get("includeDirs") or None,
        "defines": job.get("defines") or None,
    }
    options = {name: value for name, value in options.items() if value}
    try:
        artifacts = translate_artifacts(
            job["source"],
            job["backend"],
            format_output=False,
            source_backend=job["sourceBackend"],
            **options,
        )
        seconds = time.perf_counter() - started
        if len(artifacts) == 1 and artifacts[0].entry_point is None:
            code = artifacts[0].code or ""
            Path(job["artifact"]).parent.mkdir(parents=True, exist_ok=True)
            Path(job["artifact"]).write_text(code, encoding="utf-8")
            return {
                "status": "translated",
                "seconds": seconds,
                "cpuSeconds": time.process_time() - cpu_started,
                "markers": markers(code),
            }
        entries, total = [], 0
        for index, artifact in enumerate(artifacts):
            path = entry_artifact(job["artifact"], index, artifact.entry_point)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(artifact.code or "", encoding="utf-8")
            total += markers(artifact.code or "")
            entries.append(
                {
                    "entry": artifact.entry_point,
                    "stage": artifact.stage,
                    "artifact": path,
                }
            )
        return {
            "status": "translated",
            "seconds": seconds,
            "cpuSeconds": time.process_time() - cpu_started,
            "markers": total,
            "entryArtifacts": entries,
        }
    except RecursionError as error:
        return _failure(error, "translator-error", None, started, cpu_started)
    except (
        Exception
    ) as error:  # noqa: BLE001 - every translator failure is classified, never raised
        # Only a stable diagnostic code makes a failure a typed refusal; anything else is a translator error.
        code = getattr(error, "project_diagnostic_code", None)
        typed = isinstance(code, str) and bool(code)
        status = "refused" if typed else "translator-error"
        return _failure(error, status, code if typed else None, started, cpu_started)


def _failure(error, status, code, started, cpu_started):
    return {
        "status": status,
        "diagnosticCode": code,
        "exception": type(error).__name__,
        "message": str(error)[-1000:],
        "seconds": time.perf_counter() - started,
        "cpuSeconds": time.process_time() - cpu_started,
    }


def entry_artifact(artifact, index, entry):
    """Where entry ``index`` of a unit that the target translates one entry at a time is written."""
    path = Path(artifact)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", entry)
    return str(path.with_name(f"{path.stem}.{index}.{safe}{path.suffix}"))


def main():
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.setrecursionlimit(20000)
    from crosstl import translate_artifacts

    protocol.write(json.dumps({"ready": True}) + "\n")
    protocol.flush()
    for line in sys.stdin:
        job = json.loads(line)
        result = {"id": job["id"], **translate_job(job, translate_artifacts)}
        protocol.write(json.dumps(result) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
