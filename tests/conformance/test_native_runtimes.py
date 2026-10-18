"""Native execution of one source fixture per runtime; opt in with CROSSTL_CONFORMANCE_NATIVE=1.

Each available runtime executes a source kernel of its own language unchanged and must reproduce the source-rule
oracle bit for bit, so a broken adapter, binding or driver identity is caught before translated results are trusted.
With CROSSTL_CONFORMANCE_REQUIRE naming runtimes, an unavailable required runtime fails instead of skipping.
"""

import json
import os

import pytest

from . import runner
from .runtimes import RUNTIMES, provision

REFERENCE = {
    "metal": "m_ptr_arith",
    "hlsl": "h_precise",
    "glsl": "g_std430_vec3_float",
    "spirv": "s_nocontraction",
}
REQUIRED = set(
    filter(None, os.environ.get("CROSSTL_CONFORMANCE_REQUIRE", "").split(","))
)

pytestmark = pytest.mark.skipif(
    os.environ.get("CROSSTL_CONFORMANCE_NATIVE") != "1",
    reason="set CROSSTL_CONFORMANCE_NATIVE=1 to execute on native runtimes",
)


def test_required_runtimes_are_known():
    assert REQUIRED <= set(RUNTIMES)


@pytest.mark.parametrize("name", sorted(RUNTIMES))
def test_reference_kernel_reproduces_its_oracle(name, tmp_path):
    runtime = RUNTIMES[name]
    provided = provision(runtime)
    if not provided.available:
        if name in REQUIRED:
            pytest.fail(f"{name} is required but unavailable: {provided.reason}")
        pytest.skip(provided.reason)
    output = tmp_path / "run"
    assert (
        runner.main(
            [
                "--output",
                str(output),
                "--cases",
                REFERENCE[runtime.target],
                "--targets",
                runtime.target,
                "--runtimes",
                name,
                "--reference",
            ]
        )
        == 0
    )
    records = [
        json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()
    ]
    # Name the failing stage and its message, so a CI log explains a runtime failure without an artifact.
    reasons = [
        {
            "outcome": record["outcome"],
            "validation": [
                (
                    os.path.basename((item.get("command") or ("?",))[0]),
                    item.get("status"),
                    item.get("log", "")[-600:],
                )
                for item in record.get("validation") or ()
                if item.get("status") != "accepted"
            ],
            "execution": {
                key: value
                for key, value in (record.get("execution") or {}).items()
                if key in {"status", "exception", "message", "device"}
            },
            "comparison": record.get("comparison"),
        }
        for record in records
    ]
    assert [(record["runtime"], record["outcome"]) for record in records] == [
        (name, "preserved")
    ], json.dumps(reasons, indent=2, default=str)
    assert runtime.expect in records[0]["execution"]["device"]
