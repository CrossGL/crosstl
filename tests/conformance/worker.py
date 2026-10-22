"""One isolated translation or native execution, with a machine-readable outcome.

The execution half imports only the standard library and the runtime's own bindings, so it also runs inside the
Mesa container without CrossTL installed.
"""

import json
import sys
import time
from pathlib import Path

from . import native


def translate_job(job):
    from crosstl import translate

    from .validators import BACKENDS, SOURCE_BACKENDS

    started = time.perf_counter()
    try:
        result = translate(
            job["source"],
            backend=BACKENDS[job["target"]],
            source_backend=SOURCE_BACKENDS.get(job["language"], job["language"]),
            format_output=False,
            source_options=job.get("source_options"),
        )
        Path(job["artifact"]).write_text(result)
        return {"status": "translated", "seconds": time.perf_counter() - started}
    except (
        Exception
    ) as error:  # noqa: BLE001 - every translator failure is classified, never raised
        # Only a stable diagnostic code makes a failure a typed refusal; anything else is a translator error.
        code = getattr(error, "project_diagnostic_code", None)
        return {
            "status": (
                "refused" if isinstance(code, str) and code else "translator-error"
            ),
            "diagnosticCode": code if isinstance(code, str) else None,
            "exception": type(error).__name__,
            "message": str(error)[-3000:],
            "seconds": time.perf_counter() - started,
        }


def execute_job(job):
    started = time.perf_counter()
    try:
        result = {
            "status": "executed",
            **getattr(native, "execute_" + job["adapter"])(job),
        }
    except ImportError as error:
        result = {"status": "unavailable", "message": str(error)}
    except native.DriverRejected as error:
        result = {"status": "driver-rejected", "message": str(error)[-3000:]}
    except native.ExecutionTimeout as error:
        result = {"status": "timeout", "message": str(error)}
    except (
        Exception
    ) as error:  # noqa: BLE001 - driver failures are outcomes of the cell
        result = {
            "status": "runtime-error",
            "exception": type(error).__name__,
            "message": str(error)[-3000:],
        }
    result["seconds"] = time.perf_counter() - started
    return result


def main():
    job_path = Path(sys.argv[1])
    job = json.loads(job_path.read_text())
    result = translate_job(job) if job["operation"] == "translate" else execute_job(job)
    job_path.with_suffix(".result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
