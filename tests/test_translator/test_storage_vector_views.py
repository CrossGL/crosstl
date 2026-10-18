"""Indexed storage addresses retain scalar and vector offset units."""

import os
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_STORAGE_VECTOR_VIEWS"
CASES = [
    (kind, width, indexed)
    for kind in ("uint", "int", "float")
    for width in (2, 4)
    for indexed in (False, True)
]
GUARD = 0x5A1B2C3D
WORDS = [0, 0x3F800000, 0x80000000, 0x3F012345, 0x7F7FFFFF, 0xBF000000]
INPUTS = [WORDS[i % len(WORDS)] for i in range(52)]


def _source(kind, width, indexed):
    pointer = f"reinterpret_cast<const device {kind}{width}*>(&inputs[base])"
    read = f"{pointer}[1]" if indexed else f"*({pointer} + 1)"
    stores = "\n".join(
        f"results[4u + tid * {width}u + {lane}u] = as_type<uint>(value.{component});"
        for lane, component in enumerate("xyzw"[:width])
    )
    return f"""#include <metal_stdlib>
using namespace metal;
kernel void views(const device uint* inputs [[buffer(0)]],
                  device uint* results [[buffer(1)]], uint tid [[thread_position_in_grid]]) {{
    uint base = 4u + 8u * tid;
    {kind}{width} value = {read};
    {stores}
}}
"""


def _request(root, target, case):
    source, descriptor, package = _package(
        root, target, "uint", (1, 1, 1), source=_source(*case), software_subgroups=False
    )
    width = case[1]
    words = [GUARD] * 4
    for tid in range(5):
        offset = 4 + 8 * tid + width
        words.extend(INPUTS[offset : offset + width])
    words.extend([GUARD] * 4)
    inputs = _bound_values(
        descriptor,
        {
            "inputs": {"dtype": "uint32", "shape": [len(INPUTS)], "values": INPUTS},
            "results": {
                "dtype": "uint32",
                "shape": [len(words)],
                "values": [GUARD] * 4 + [0xDEADBEEF] * (5 * width) + [GUARD] * 4,
            },
        },
    )
    expected = _bound_values(
        descriptor,
        {"results": {"dtype": "uint32", "shape": [len(words)], "values": words}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        expected,
        {"workgroupCount": [5, 1, 1], "workgroupSize": [1, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, expected


@pytest.mark.parametrize("target", ["directx", "opengl"])
@pytest.mark.parametrize("case", CASES)
def test_indexed_storage_vector_view_compiles(tmp_path, target, case):
    _, request, _ = _request(tmp_path, target, case)
    _compile(request.artifact_path.read_text(), target, tmp_path)


@pytest.mark.parametrize("case", CASES)
def test_indexed_storage_vector_view_executes(tmp_path, case):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required storage vector execution")
    if sys.platform == "darwin":
        pytest.skip("Metal vector pointer round-trip lowering is tracked separately")
    target = {"win32": "directx", "linux": "opengl"}[sys.platform]
    _, request, expected = _request(tmp_path, target, case)
    _execute(request, expected, tmp_path)


def test_storage_vector_view_execution_is_required():
    from tools import ci_coverage

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = ci_coverage.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_storage_vector_views.py" in step
    assert "if:" not in step and "continue-on-error" not in step
    assert "pytest -q -n auto" in step
    assert "--basetemp=" in step and "--junitxml=" in step
