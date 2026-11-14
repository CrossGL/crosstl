"""Accessor branches for distinct threadgroup allocations follow the view's proven identity.

Resource-aggregate lowering reads every ``threadgroup int*`` held in a struct through one accessor that
selects the allocation by the view's identity: ``if (reference.identity == 2) return wide[offset];
return narrow[offset];``. OpenGL must prove each subscript fits the allocation it indexes. The identity
literals were built with an untyped literal, so the range proof could not compare them with the typed
identity member and checked the ``narrow`` fallthrough against offsets that only the ``wide`` view
reaches. A view 64 elements into a 96-element tile was refused because a 40-element tile shares the
accessor. The literals are now typed ``int``, and the view's exact identity decides the branch.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_WORKGROUP_VIEW_IDENTITY_BRANCHES"
ENTRY = "two_tiles"
THREADS = 32
WORKGROUP = (THREADS, 1, 1)
WIDE, NARROW = 96, 40
WIDE_STEP, NARROW_STEP = 64, 8
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 4001) - 2000 for index in range(WIDE)]

SOURCE = f"""#include <metal_stdlib>
using namespace metal;

struct Reader {{
  const threadgroup int* src;
  Reader(const threadgroup int* src_, ushort lid) : src(src_ + lid) {{}}
  int read(short step) const {{
    return src[0] + src[step];
  }}
}};

[[kernel]] void {ENTRY}(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]]) {{
  threadgroup int wide[{WIDE}];
  threadgroup int narrow[{NARROW}];
  for (uint row = 0; row < {WIDE // THREADS}; row++) {{
    wide[lid + row * {THREADS}] = values[lid + row * {THREADS}];
  }}
  narrow[lid] = values[lid] * 3;
  if (lid < {NARROW - THREADS}) {{
    narrow[lid + {THREADS}] = values[lid + {THREADS}] * 3;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Reader high(wide, ushort(lid));
  Reader low(narrow, ushort(lid));
  results[lid] = high.read({WIDE_STEP}) - low.read({NARROW_STEP});
}}
"""


def _expected():
    results = [
        VALUES[lane]
        + VALUES[lane + WIDE_STEP]
        - 3 * (VALUES[lane] + VALUES[lane + NARROW_STEP])
        for lane in range(THREADS)
    ]
    return results + GUARD


def _request(root, target):
    # No index range assertions: every threadgroup subscript must be proven.
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=SOURCE, software_subgroups=False
    )
    expected = _expected()
    initial = [-0x21524111] * THREADS + GUARD
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
        {"workgroupCount": [1, 1, 1], "workgroupSize": list(WORKGROUP)},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return source, request, outputs


def test_wide_view_reaches_past_the_narrow_allocation():
    # The case is only meaningful if the wide view's offsets overrun the narrow tile.
    assert THREADS - 1 + WIDE_STEP >= NARROW
    assert THREADS - 1 + WIDE_STEP < WIDE
    assert THREADS - 1 + NARROW_STEP < NARROW


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_identity_branches_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    if target == "opengl":
        assert "crosstl_workgroup_wide[" in generated
        assert "crosstl_workgroup_narrow[" in generated
    _compile(generated, target, tmp_path)


def test_identity_branches_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_identity_branches_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native identity branches")
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


def test_identity_branches_are_required_on_all_native_platforms():
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
    assert "test_workgroup_view_identity_branches.py" in step
    assert "if:" not in step and "continue-on-error" not in step
