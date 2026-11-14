"""Member and swizzle stores through lowered storage pointers stay in place.

Direct3D and OpenGL reach pointers held in private aggregates through access
helpers whose loads are values, so ``row[i].x = value`` must not become an
assignment to a helper's result. The store helper writes the member of the
element in place: the element's other members keep their stored values, as
the source's member store leaves them.
"""

import os
import re
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_resource_aggregates import REQUIRE_ENV
from tests.test_translator.test_resource_aggregates_opengl import (
    REQUIRE_ENV as REQUIRE_OPENGL_ENV,
)
from tests.test_translator.test_software_subgroup_product import _package

SOURCE = """
#include <metal_stdlib>
using namespace metal;

struct Rows {
  const device float* in;
  threadgroup float2* buf;
  device float2* out;
  int n;

  void fill(int i) const thread {
    threadgroup float2* row = buf + n;
    row[i].x = in[i];
    row[i].y = -row[i].x;
  }

  void emit(int i) const thread {
    device float2* dst = out + n;
    dst[i].y = buf[n + i].x * 2.0f;
    dst[i].y += buf[n + i].y;
  }
};

kernel void member_stores(
    const device float* in [[buffer(0)]],
    device float2* out [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  threadgroup float2 shared_values[8];
  thread Rows rows = {in, &shared_values[0], out, 4};
  rows.fill(int(index));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  rows.emit(int(index));
}
"""
INPUT = [1.5, 2.5, -3.0, 4.25]
GUARD = -9.0
# Each thread stores only the y member of its output row; x keeps the guard.
EXPECTED = [GUARD] * 8 + [part for value in INPUT for part in (GUARD, value)]
# Every subscript stays inside the eight-element threadgroup array and the
# eight-element output; OpenGL needs them proven before narrowing to int.
ASSERTIONS = tuple(
    {"source": "products.metal", "expression": expression, "minimum": 0, "maximum": 7}
    for expression in (
        "reference.offset + index",
        "crosstl_resource + reference.offset + index",
        "crosstl_resource_2 + reference.offset + index",
    )
)
WORKGROUP_ASSERTIONS = (
    {
        "source": "products.metal",
        "entry_point": "member_stores",
        "function": "*",
        "parameter": "*",
        "minimum": 0,
        "maximum": 7,
    },
)


def _translated(tmp_path, target):
    return _package(
        tmp_path,
        target,
        "float",
        (4, 1, 1),
        source=SOURCE,
        software_subgroups=False,
        index_range_assertions=ASSERTIONS if target == "opengl" else (),
        workgroup_access_assertions=WORKGROUP_ASSERTIONS if target == "opengl" else (),
    )


def _workload():
    vector = {"dtype": "float32", "shape": [8, 2]}
    return (
        {
            "in_": {"dtype": "float32", "shape": [4], "values": INPUT},
            "out_": dict(vector, values=[GUARD] * 16),
        },
        {"out_": dict(vector, values=EXPECTED)},
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_member_stores_write_the_member_in_place(tmp_path, target):
    _source, descriptor, package = _translated(tmp_path, target)
    generated = (package / descriptor["artifact"]["packagePath"]).read_text()
    assert not re.search(r"\)\s*\.\s*[xy]\s*[+]?=", generated)
    assert re.search(r"\]\s*\.x = value;", generated)
    assert re.search(r"\]\s*\.y = value;", generated)
    assert re.search(r"\]\s*\.y \+= value;", generated)
    validation = tmp_path / "validation"
    validation.mkdir()
    _compile(generated, target, validation)


def test_member_stores_execute_natively(tmp_path):
    required = {
        "darwin": ("metal", REQUIRE_ENV),
        "win32": ("directx", REQUIRE_ENV),
        "linux": ("opengl", REQUIRE_OPENGL_ENV),
    }.get(sys.platform)
    if required is None or os.environ.get(required[1]) != "1":
        pytest.skip("set the resource-aggregate runtime variable for native execution")
    target = required[0]
    source, descriptor, package = _translated(tmp_path, target)
    inputs, outputs = _workload()
    expected = _bound_values(descriptor, outputs)
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        _bound_values(descriptor, inputs),
        expected,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source if target == "metal" else None,
        original_entry="member_stores" if target == "metal" else None,
    )
