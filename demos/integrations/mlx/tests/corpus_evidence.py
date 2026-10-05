from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

KEEP_EVIDENCE_ENV = "CROSTL_KEEP_CORPUS_EVIDENCE"
EVIDENCE_DIRECTORY = ".crosstl-corpus-evidence"


def assert_deferred_metal_compiler_diagnostics(payload: dict) -> None:
    toolchains = payload["validation"]["toolchains"]
    assert len(toolchains) == 1
    assert toolchains[0]["target"] == "metal"
    assert toolchains[0]["status"] in {"available", "unavailable"}
    expected_count = int(toolchains[0]["status"] == "unavailable")
    assert payload["summary"]["diagnosticCounts"] == {
        "note": 0,
        "warning": expected_count,
        "error": 0,
    }
    assert len(payload["diagnostics"]) == expected_count
    for diagnostic in payload["diagnostics"]:
        assert {
            field: diagnostic.get(field)
            for field in ("severity", "code", "target", "missingCapabilities")
        } == {
            "severity": "warning",
            "code": "project.validate.toolchain-unavailable",
            "target": "metal",
            "missingCapabilities": ["toolchain.validation"],
        }


@contextmanager
def corpus_workspace(
    root: Path, *, family: str, target: str, entry_point: str
) -> Iterator[Path]:
    prefix = f"{family}-{target}-"
    if os.environ.get(KEEP_EVIDENCE_ENV) == "1":
        evidence_root = root / EVIDENCE_DIRECTORY
        evidence_root.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=prefix, dir=evidence_root))
        (work_dir / "case.json").write_text(
            json.dumps(
                {"family": family, "target": target, "entryPoint": entry_point},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        yield work_dir
    else:
        with tempfile.TemporaryDirectory(prefix=f".crosstl-{prefix}", dir=root) as path:
            yield Path(path)


def run_compiler(
    command: Sequence[str], *, work_dir: Path, timeout: int
) -> subprocess.CompletedProcess:
    record = {"command": list(command), "timeoutSeconds": timeout, "status": "running"}
    record_path = work_dir / "compiler.json"

    def write_record() -> None:
        record_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    write_record()
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as exc:
        # Timeout output can be bytes even when subprocess text mode is enabled.
        def decoded(value):
            return (
                value.decode("utf-8", errors="replace")
                if isinstance(value, bytes)
                else value
            )

        record.update(
            status="timed-out", stdout=decoded(exc.stdout), stderr=decoded(exc.stderr)
        )
        write_record()
        raise
    except OSError as exc:
        record.update(status="launch-failed", error=str(exc))
        write_record()
        raise
    record.update(
        status="completed",
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
    )
    write_record()
    return result


def native_compiler_runner(work_dir: Path):
    def run(command, *, input_text=None):
        assert input_text is None
        result = run_compiler(command, work_dir=work_dir, timeout=120)
        files = [Path(command[-1])]
        if "-Fo" in command:
            files.append(Path(command[command.index("-Fo") + 1]))
        identities = []
        retained = work_dir / "native-compiler"
        retained.mkdir(exist_ok=True)
        for path in files:
            if path.is_file():
                destination = retained / path.name
                shutil.copyfile(path, destination)
                raw = destination.read_bytes()
                identities.append(
                    {
                        "path": destination.relative_to(work_dir).as_posix(),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "sizeBytes": len(raw),
                    }
                )
        (work_dir / "compiler-artifacts.json").write_text(
            json.dumps({"files": identities}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return result

    return run


def record_native_request(
    work_dir: Path, request, *, commit: str, workload_id: str
) -> None:
    (work_dir / "request.json").write_text(
        json.dumps(
            {
                "commit": commit,
                "workload": workload_id,
                "fixture": request.fixture.to_json(),
                "executionPlan": request.execution_plan.to_json(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def record_native_result(work_dir: Path, result) -> None:
    (work_dir / "result.json").write_text(
        json.dumps(
            {
                "status": result.status,
                "outputs": result.outputs,
                "message": result.message,
                "details": result.details,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
