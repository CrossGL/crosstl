from __future__ import annotations

import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

KEEP_EVIDENCE_ENV = "CROSTL_KEEP_CORPUS_EVIDENCE"
EVIDENCE_DIRECTORY = ".crosstl-corpus-evidence"


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
