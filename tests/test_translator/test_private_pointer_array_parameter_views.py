"""Private pointer views into a fixed array parameter are backed by that array.

A Metal function may take ``thread T vals[N]`` and pass ``&vals[i]`` or ``vals + k`` to a ``thread T*`` parameter.
In C++ the array parameter is a pointer to the caller's array ([dcl.fct]/5), so the callee writes the caller's
elements. DirectX and OpenGL have no private pointers: the view must be lowered to the backing array plus an
offset. Both targets refused such views with ``missing-view-backing`` because only function-local arrays were
accepted as backing objects. MLX's ``MMATile::row_reduce`` passes ``&vals[i * kElemRows]`` from its array
parameter to ``BaseMMAFrag::row_reduce``.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_PRIVATE_POINTER_ARRAY_PARAMETER_VIEWS"
ENTRY = "array_parameter_views"
WORKGROUP = (32, 1, 1)
GROUPS = 2
THREADS = GROUPS * WORKGROUP[0]
ROWS = 3
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(THREADS * ROWS * 2)]
INITIAL = (-5, 0, 7)

SOURCE = """#include <metal_stdlib>
using namespace metal;

static inline void accumulate_max(thread const int2& inp, thread int* reduced) {
  reduced[0] = metal::max(reduced[0], metal::max(inp.x, inp.y));
}

static inline void scale_first(thread int* tail, int factor) {
  tail[0] = tail[0] * factor + 1;
}

static inline void row_reduce(thread const int2* frags, thread int vals[3]) {
  for (short i = 0; i < 3; ++i) {
    accumulate_max(frags[i], &vals[i]);
  }
  scale_first(vals + 2, 3);
}

[[kernel]] void array_parameter_views(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint index [[thread_position_in_grid]]) {
  int2 frags[3];
  for (int i = 0; i < 3; ++i) {
    frags[i] = int2(values[6 * index + 2 * i], values[6 * index + 2 * i + 1]);
  }
  int m[3] = {-5, 0, 7};
  row_reduce(frags, m);
  for (int i = 0; i < 3; ++i) {
    results[3 * index + i] = m[i];
  }
}
"""


def _expected():
    results = []
    for index in range(THREADS):
        row = []
        for i in range(ROWS):
            pair = VALUES[6 * index + 2 * i : 6 * index + 2 * i + 2]
            row.append(max(INITIAL[i], *pair))
        row[2] = row[2] * 3 + 1
        results.extend(row)
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


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_array_parameter_views_lower_to_backing_array_offsets(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    # The callee receives the caller's array and an element offset, never a copy of one element.
    assert "reduced_base" in generated and "tail_base" in generated
    _compile(generated, target, tmp_path)


def test_array_parameter_views_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_array_parameter_views_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native array parameter views")
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


def test_array_parameter_views_are_required_on_all_native_platforms():
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
    assert "test_private_pointer_array_parameter_views.py" in step
    assert "if:" not in step and "continue-on-error" not in step
