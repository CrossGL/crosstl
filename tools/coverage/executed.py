"""Execution verdicts kept from one coverage run to the next.

Running a compute probe and its translation (:mod:`.execution`) takes most of a coverage leg's time, and from one
commit to the next CrossTL writes almost every translation the same. A run that ``CROSSTL_COVERAGE_EXECUTIONS`` names a
directory for files each verdict it judges under a digest of everything the run depends on, and a later run with the
same digest takes the verdict instead of running the two probes again:

- both languages and both sources, the native probe and the translation;
- the feature, whose types choose the inputs and the exact words a result may also match;
- the runtimes the run selects, with the identity each one's driver reports (:func:`tests.conformance.runtimes.provision`),
  so another Mesa build or another device runs every probe again;
- the versions of the compilers that lower either language and SPIR-V;
- the code of the coverage engine (``tools/coverage``) and of the conformance harness (``tests/conformance``), the
  Python version, the version of every installed distribution (NumPy draws the inputs, and the OpenGL and WebGPU
  runtimes are Python packages) and the operating system's release, which ships the GPU drivers on macOS and
  Windows, so a change to any of them runs every probe again.

Only a result the runtime judged is kept: one that ran gives the same verdict for the same inputs, while a probe the
harness recorded no run of may run next time. The directory keeps only the verdicts the latest run used, as the
validation verdicts do (:mod:`.runner`), and a verdict is written aside and renamed, so a reader never sees half of it.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import platform
import sys
import time
from dataclasses import replace
from importlib import metadata
from pathlib import Path

from . import validators
from .model import Result

EXECUTIONS = "CROSSTL_COVERAGE_EXECUTIONS"
ROOT = Path(__file__).resolve().parents[2]
# The code a verdict depends on besides the sources: the coverage engine and the conformance harness it runs.
CODE = (ROOT / "tools" / "coverage", ROOT / "tests" / "conformance")
# The fields of a result that :func:`.execution.run` sets.
FIELDS = ("outcome", "detail", "runtime", "unrun")


def distributions() -> str:
    """Every installed distribution with its version, one per line, sorted."""
    return "\n".join(
        sorted(
            {
                f"{(dist.metadata['Name'] or '').lower()}=={dist.version}"
                for dist in metadata.distributions()
            }
        )
    )


@functools.lru_cache(maxsize=None)
def code_digest() -> str:
    """A digest of every file under :data:`CODE`, the Python version, :func:`distributions` and the platform."""
    digest = hashlib.sha256(
        sys.version.encode() + b"\0" + platform.platform().encode() + b"\0"
    )
    digest.update(distributions().encode() + b"\0")
    for folder in CODE:
        for path in sorted(folder.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0")
                digest.update(path.read_bytes() + b"\0")
    return digest.hexdigest()


@functools.lru_cache(maxsize=None)
def _compilers(language: str) -> str:
    tools = dict.fromkeys(
        [*validators.toolchain(language), *validators.toolchain("spirv")]
    )
    return "\n".join(f"{tool}={validators.version(tool)}" for tool in tools)


class Kept:
    """The verdicts in ``folder`` for a run on the ``runtimes`` the harness selects, with driver ``identities``."""

    def __init__(self, folder: Path, runtimes: str, identities: str):
        self.folder = folder
        self.context = "\0".join((runtimes, identities, code_digest()))
        self.started = time.time()

    def _path(self, result: Result, languages: tuple[str, str]) -> Path:
        parts = (
            self.context,
            *languages,
            _compilers(languages[0]),
            _compilers(languages[1]),
            repr(result.feature),
            result.direction,
            *result.sources,
        )
        return self.folder / hashlib.sha256("\0".join(parts).encode()).hexdigest()

    def lookup(self, result: Result, languages: tuple[str, str]) -> Result | None:
        """``result`` with the verdict a run kept for it, or ``None``."""
        path = self._path(result, languages)
        try:
            kept = json.loads(path.read_text())
        except (FileNotFoundError, ValueError):
            return None
        # A verdict this run used survives forget_unused.
        os.utime(path)
        return replace(result, **{name: kept[name] for name in FIELDS})

    def store(self, result: Result, languages: tuple[str, str], judged: Result) -> None:
        """Keep ``judged``, the verdict of ``result``'s run, when the runtime gave one."""
        if not judged.runtime:
            return
        path = self._path(result, languages)
        partial = path.with_suffix(f".{os.getpid()}")
        partial.write_text(json.dumps({name: getattr(judged, name) for name in FIELDS}))
        partial.replace(path)

    def forget_unused(self) -> None:
        """Delete the verdicts this run neither used nor wrote."""
        for path in self.folder.iterdir():
            # One second of slack covers file systems that store whole-second times.
            if path.stat().st_mtime < self.started - 1:
                path.unlink()


def open_kept(runtimes: str, identities: str) -> Kept | None:
    """The kept verdicts for this run, or ``None`` when ``CROSSTL_COVERAGE_EXECUTIONS`` is unset."""
    folder = os.environ.get(EXECUTIONS)
    if not folder:
        return None
    Path(folder).mkdir(parents=True, exist_ok=True)
    return Kept(Path(folder), runtimes, identities)
