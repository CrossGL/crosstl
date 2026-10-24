"""Template member overloads that bind the same template arguments keep distinct names.

Two member function templates with the same name and template parameters are distinct overloads when their
function parameter types differ ([over.load], [temp.over.link]). A call from one to the other is an ordinary call,
not recursion. MLX's ``BaseMMAFrag::mma`` has a fragment overload that copies its operands into
``simdgroup_matrix`` values and calls the matrix overload with the same ``Atype, Btype, Ctype``. Lowering both to
one free-function name made the forwarding call read as a self call, which Metal refuses.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_METAL_MEMBER_OVERLOAD_NAMES"
ENTRY = "forwarded_scale"
WORKGROUP = (32, 1, 1)
GROUPS = 2
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(GROUPS * 32 * 2)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

template <typename T>
struct Box {
  T first;
  T second;
};

template <typename T>
struct Ops {
  template <typename A, typename B>
  static void scale(thread vec<T, 2>& out, thread vec<A, 2>& a, thread vec<B, 2>& b) {
    Box<T> result;
    Box<A> lhs;
    Box<B> rhs;
    lhs.first = a.x;
    lhs.second = a.y;
    rhs.first = b.x;
    rhs.second = b.y;
    scale(result, lhs, rhs);
    out = vec<T, 2>(result.first, result.second);
  }

  template <typename A, typename B>
  static void scale(thread Box<T>& out, thread Box<A>& a, thread Box<B>& b) {
    out.first = T(a.first * b.first + 3);
    out.second = T(a.second - b.second);
  }
};

[[kernel]] void forwarded_scale(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  int2 a = int2(values[2 * index], values[2 * index + 1]);
  int2 b = int2(int(index % 5) + 1, 7);
  int2 out;
  Ops<int>::scale(out, a, b);
  results[index] = out.x * 4096 + out.y;
}
"""


def _expected():
    results = []
    for index in range(GROUPS * 32):
        a = VALUES[2 * index : 2 * index + 2]
        b = [index % 5 + 1, 7]
        results.append((a[0] * b[0] + 3) * 4096 + (a[1] - b[1]))
    return results + GUARD


def _request(root, target):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=SOURCE, software_subgroups=False
    )
    expected = _expected()
    initial = [-0x21524111] * (len(expected) - len(GUARD)) + GUARD
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "int32", "shape": [len(VALUES)], "values": VALUES},
            "results": {"dtype": "int32", "shape": [len(initial)], "values": initial},
        },
    )
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "int32", "shape": [len(expected)], "values": expected}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [GROUPS, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


@pytest.mark.parametrize("target", ("metal", "directx", "opengl"))
def test_forwarding_overload_keeps_its_own_name(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "overload_1" in generated
    _compile(generated, target, tmp_path, metal_compile_flags=("-Wall", "-Wextra"))


def test_forwarding_overload_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native member overload names")
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=ENTRY,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_member_overload_names_are_required_on_all_native_platforms():
    from pathlib import Path

    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_metal_member_overload_names.py" in step
    assert "if:" not in step and "continue-on-error" not in step
