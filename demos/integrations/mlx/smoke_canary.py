"""Run a few workloads of every complete MLX kernel family in the smoke tier.

The complete family proofs translate every unary, binary, copy, reduce and
quantized kernel of MLX ``CORPUS_COMMIT``. They run only in the full tier, so a
pull request or merge-queue group that changes the translator but not the MLX
demo reaches ``main`` after the smoke tier alone. That tier translated two
arange kernels, and a change to library-call resolution that refused every
bfloat16 unary kernel landed unnoticed (#4065, fixed by #4085).

The canary runs a handful of workloads of each family through the same tests the
full tier runs, for every target the family proves, so a change of that kind
fails on its own pull request. The workloads cover the type paths that share the
most translator code: bfloat16 library math in several namespaces, float16,
complex64, narrowing casts, a looped row reduction and a quantized matrix-vector
product.

Usage, from the repository root, with ``CROSTL_MLX_ROOT`` naming a checkout of
``CORPUS_COMMIT`` that holds ``mlx/backend/metal/kernels``::

    python -m demos.integrations.mlx.smoke_canary [pytest arguments]
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import pytest

KERNEL_TESTS = "demos/integrations/mlx/tests/kernels"
CORPUS_COMMIT = "846d176227a0ac13d2667e58d2bb68b322109ab0"
ROOT_ENV = "CROSTL_MLX_ROOT"
PIN_MODE_ENV = "CROSSTL_MLX_PIN_MODE"
WORKERS = "4"
TARGETS = ("metal", "directx", "opengl")


@dataclass(frozen=True)
class Family:
    """One complete family: its workloads and, per target, the test that proves it."""

    name: str
    workloads: tuple[str, ...]
    targets: tuple[str, ...] = TARGETS

    def test(self, target: str) -> str:
        """Return the ``module::function`` that proves this family on ``target``."""
        if target == "metal":
            module = (
                "test_unary_native_loader"
                if self.name == "unary"
                else f"test_{self.name}_complete_metal_roundtrip"
            )
            function = (
                f"test_current_mlx_{self.name}_family_exports_native_compilation_bundle"
            )
        else:
            module = f"test_{self.name}_complete_{target}"
            function = f"test_current_mlx_{self.name}_family_translates_to_{target}"
        return f"{KERNEL_TESTS}/{module}.py::{function}"

    def require_environment(self) -> dict[str, str]:
        """Name the variables that make a missing corpus or tool fail, not skip."""
        kinds = {
            "metal": "METAL_SOURCE",
            "directx": "DIRECTX_TRANSLATION",
            "opengl": "OPENGL_TRANSLATION",
        }
        return {
            f"CROSTL_REQUIRE_MLX_{self.name.upper()}_{kinds[target]}": "1"
            for target in self.targets
        }


FAMILIES = (
    Family(
        "unary",
        (
            "vn_Sinbfloat16bfloat16",
            "v_Expfloat16float16",
            "v_Logcomplex64complex64",
        ),
    ),
    Family("binary", ("gn2_ArcTan2bfloat16", "vv_Dividecomplex64")),
    Family("copy", ("g3_copybfloat16int8", "g2_copycomplex64uint16")),
    Family(
        "reduce",
        ("all_reduce_sumbfloat16", "row_reduce_looped_1_reduce_sumbfloat16"),
    ),
    # The complete quantized proof covers the Metal round trip only.
    Family(
        "quantized",
        ("affine_qmv_bfloat16_t_gs_128_b_3_batch_0",),
        targets=("metal",),
    ),
)


def node_ids() -> tuple[str, ...]:
    """Return every canary test, family by family and target by target."""
    return tuple(
        f"{family.test(target)}[{workload}]"
        for family in FAMILIES
        for target in family.targets
        for workload in family.workloads
    )


def environment() -> dict[str, str]:
    """Return the variables the canary sets for its tests.

    Generated-artifact pins are reported rather than enforced, as in merge-queue
    groups: the canary runs on translator changes the pins were not reviewed
    against, and the full tier owns them. Translation, diagnostics and the DXC,
    glslang and SPIR-V validation stay enforced.
    """
    values = {PIN_MODE_ENV: "report"}
    for family in FAMILIES:
        values.update(family.require_environment())
    return values


def problems(expected, passed, other) -> list[str]:
    """Describe why a run is not a pass of exactly the ``expected`` tests.

    ``passed`` holds the node ids that passed and ``other`` maps the remaining
    reported node ids to their outcome. A skipped canary proves nothing, so a
    skip is a problem like a failure.
    """
    expected, passed = set(expected), set(passed)
    found = [f"{node}: {other[node]}" for node in sorted(other)]
    found += [f"{node}: did not run" for node in sorted(expected - passed - set(other))]
    found += [f"{node}: not a canary test" for node in sorted(passed - expected)]
    return found


class _Outcomes:
    """Collect test outcomes, including those xdist workers report."""

    def __init__(self) -> None:
        self.passed: set[str] = set()
        self.other: dict[str, str] = {}

    def pytest_runtest_logreport(self, report) -> None:
        if report.failed or report.skipped:
            self.other[report.nodeid] = f"{report.outcome} in {report.when}"
        elif report.when == "call" and report.passed:
            self.passed.add(report.nodeid)


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not os.environ.get(ROOT_ENV):
        print(
            f"{ROOT_ENV} must name a checkout of MLX {CORPUS_COMMIT}", file=sys.stderr
        )
        return 2
    os.environ.update(environment())
    expected = node_ids()
    outcomes = _Outcomes()
    status = pytest.main(
        ["-q", "-n", WORKERS, "--tb=short", "-rfEs", *arguments, *expected],
        plugins=[outcomes],
    )
    if status != 0:
        return int(status)
    # Pytest passed: make sure it passed exactly the canary, with no skips.
    found = problems(expected, outcomes.passed, outcomes.other)
    for problem in found:
        print(f"MLX complete-family canary: {problem}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
