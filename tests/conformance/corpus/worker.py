"""Persistent translation worker: one JSON job per stdin line, one JSON result per protocol line.

The protocol uses a duplicate of the original stdout, and fd 1 is redirected to stderr, so nothing the
translator prints can corrupt it. Only ``crosstl.translate`` is timed; process start-up and imports are not part
of a unit's latency.
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


def translate_job(job, translate):
    """Translate one job; ``seconds`` is wall time and ``cpuSeconds`` the worker's CPU time, which a loaded
    machine inflates far less."""
    return _translate_job(job, translate, time.perf_counter(), time.process_time())


def _translate_job(job, translate, started, cpu_started):
    try:
        options = {
            "include_paths": job.get("includeDirs") or None,
            "defines": job.get("defines") or None,
        }
        code = translate(
            job["source"],
            backend=job["backend"],
            format_output=False,
            source_backend=job["sourceBackend"],
            **{name: value for name, value in options.items() if value},
        )
        seconds = time.perf_counter() - started
        code = code or ""
        Path(job["artifact"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["artifact"]).write_text(code, encoding="utf-8")
        return {
            "status": "translated",
            "seconds": seconds,
            "cpuSeconds": time.process_time() - cpu_started,
            "markers": markers(code),
        }
    except RecursionError as error:
        return {
            "status": "translator-error",
            "exception": type(error).__name__,
            "message": str(error)[-1000:],
            "seconds": time.perf_counter() - started,
            "cpuSeconds": time.process_time() - cpu_started,
        }
    except (
        Exception
    ) as error:  # noqa: BLE001 - every translator failure is classified, never raised
        # Only a stable diagnostic code makes a failure a typed refusal; anything else is a translator error.
        code = getattr(error, "project_diagnostic_code", None)
        typed = isinstance(code, str) and bool(code)
        return {
            "status": "refused" if typed else "translator-error",
            "diagnosticCode": code if typed else None,
            "exception": type(error).__name__,
            "message": str(error)[-1000:],
            "seconds": time.perf_counter() - started,
            "cpuSeconds": time.process_time() - cpu_started,
        }


def main():
    protocol = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    sys.setrecursionlimit(20000)
    from crosstl import translate

    protocol.write(json.dumps({"ready": True}) + "\n")
    protocol.flush()
    for line in sys.stdin:
        job = json.loads(line)
        result = {"id": job["id"], **translate_job(job, translate)}
        protocol.write(json.dumps(result) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
