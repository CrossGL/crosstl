"""Known defects of the pinned Direct3D WARP libraries, keyed by library digest.

The Windows native jobs run the same tests on two pinned WARP libraries. A test
listed here is expected to fail only while its defective library is the one the
test process loads. The marker is strict: an unexpected pass fails the run, so a
corrected runtime forces the entry to be removed. The same test must pass,
unmarked, on the conformant library named by its entry, so every assertion keeps
running against a runtime that implements the behaviour it checks.

Entries carry no tolerance or expectation changes. They cite the issue that
records the independent runtime evidence.
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pytest

# SHA-256 of build/native/bin/x64/d3d10warp.dll in the pinned NuGet packages.
WARP_1_0_21 = "e79c10550449365adf0a9393d97a0df69941e671ab6e952a78d92da066517ca3"
WARP_1_0_13 = "0621056518e047fd2fa9f75f03a49fed42d7158dee77c11c441cd34d32e93638"
PINNED_WARP_LIBRARIES = {WARP_1_0_21: "1.0.21", WARP_1_0_13: "1.0.13"}
WARP_LIBRARY_NAME = "d3d10warp.dll"

_MATRIX = "tests/test_translator/test_directx_cooperative_matrix.py"
_NARROW = "tests/test_translator/test_directx_narrow_buffer_extents.py"
_SHUFFLES = "tests/test_translator/test_integer_subgroup_shuffles.py"
_BINARY16_ALIASES = "tests/test_translator/test_binary16_alias_aggregates.py"
_FLOATING_BINARY = (
    "demos/integrations/mlx/tests/kernels/test_current_floating_binary.py"
)


@dataclass(frozen=True)
class RuntimeDefect:
    """One runtime defect, the library that has it and the tests it breaks."""

    issue: int
    summary: str
    defective: str
    conformant: str
    tests: tuple[str, ...]

    @property
    def reason(self) -> str:
        version = PINNED_WARP_LIBRARIES[self.defective]
        return (
            f"WARP {version} runtime defect #{self.issue}: {self.summary}. "
            f"WARP {PINNED_WARP_LIBRARIES[self.conformant]} must pass this test."
        )


DEFECTS = (
    RuntimeDefect(
        issue=2193,
        summary="precise multiply and add are contracted into one rounding",
        defective=WARP_1_0_21,
        conformant=WARP_1_0_13,
        tests=(
            *(
                f"{_MATRIX}::test_precise_scalar_control_executes[{view}-{flags}]"
                for view in ("direct", "native-views")
                for flags in ("precise", "strict", "unoptimized")
            ),
            f"{_MATRIX}::test_software_matrix_executes[shape1-True]",
        ),
    ),
    RuntimeDefect(
        issue=2197,
        summary="the final element of an odd-length 16-bit buffer view reads zero",
        defective=WARP_1_0_13,
        conformant=WARP_1_0_21,
        tests=(
            *(
                f"{_NARROW}::test_narrow_buffer_extent_execute[{mode}-105-{kind}]"
                for mode in ("exact", "padded-allocation")
                for kind in ("short", "ushort")
            ),
            *(
                f"{_SHUFFLES}::test_integer_shuffles_execute[shape1-{kind}]"
                for kind in ("short", "ushort")
            ),
            f"{_BINARY16_ALIASES}::test_binary16_alias_aggregates_execute",
        ),
    ),
    RuntimeDefect(
        issue=2259,
        summary=(
            "binary16 results in [65520, 65536) round to the largest finite "
            "value instead of infinity"
        ),
        defective=WARP_1_0_13,
        conformant=WARP_1_0_21,
        tests=(
            f"{_FLOATING_BINARY}::test_current_floating_binary_native_parity[half]",
        ),
    ),
)


@lru_cache(maxsize=1)
def loaded_warp_sha256() -> str | None:
    """Digest of the WARP library that Direct3D loads for this interpreter.

    The Windows jobs install the pinned library beside ``python.exe``. The
    application directory precedes the system directory in the library search
    order, and the native worker copies the same file beside its executable.
    """

    if sys.platform != "win32":
        return None
    library = Path(sys.executable).with_name(WARP_LIBRARY_NAME)
    if not library.is_file():
        return None
    return hashlib.sha256(library.read_bytes()).hexdigest()


def expected_failures(digest: str | None) -> dict[str, RuntimeDefect]:
    """Map each test node expected to fail on ``digest`` to its defect."""

    failures: dict[str, RuntimeDefect] = {}
    for defect in DEFECTS:
        if defect.defective != digest:
            continue
        for node in defect.tests:
            if node in failures:
                raise ValueError(f"{node} is registered for two runtime defects")
            failures[node] = defect
    return failures


def mark_known_runtime_defects(items, digest: str | None = None) -> None:
    """Mark collected items that the loaded WARP library is known to fail."""

    failures = expected_failures(loaded_warp_sha256() if digest is None else digest)
    if not failures:
        return
    for item in items:
        defect = failures.get(item.nodeid)
        if defect is not None:
            item.add_marker(
                pytest.mark.xfail(
                    reason=defect.reason, raises=AssertionError, strict=True
                )
            )
