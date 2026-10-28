"""Workgroup pointer reinterpretations that keep the element type are plain views.

Resource-aggregate lowering turns every ``threadgroup`` pointer into an allocation identity
and an offset. A ``reinterpret_cast`` between pointers to the same element type in the same
address space changes no byte, so it must lower to the reinterpreted pointer itself, narrowed
to the access the cast keeps. Such casts are common when a kernel reuses a tile whose element
type is a template parameter that happens to equal the view's type. They were refused as a
pointer-contract mismatch on DirectX and OpenGL. A reinterpretation that changes the element
type, or adds write access, still fails closed.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_WORKGROUP_IDENTITY_REINTERPRET"
ENTRY = "exchange_halves"
THREADS = 32
WORKGROUP = (THREADS, 1, 1)
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 4001) - 2000 for index in range(2 * THREADS)]


def _source(view="threadgroup int*", peer="const threadgroup int*"):
    return f"""#include <metal_stdlib>
using namespace metal;

struct Reader {{
  const threadgroup int* src;
  int read(short offset) const {{
    return src[offset];
  }}
}};

[[kernel]] void {ENTRY}(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]]) {{
  threadgroup int tile[{2 * THREADS}];
  tile[lid] = values[lid];
  tile[lid + {THREADS}] = values[lid + {THREADS}];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  Reader reader{{tile}};
  int own = reader.read(short(lid)) + reader.read(short(lid + {THREADS}));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Reuse the tile as an exchange buffer through same-typed views.
  {view} slots = reinterpret_cast<{view}>(tile);
  slots[lid] = own;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {peer} other = reinterpret_cast<{peer}>(tile) + ({THREADS - 1} - lid);
  results[lid] = own - 2 * other[0];
}}
"""


def _expected():
    own = [VALUES[lane] + VALUES[lane + THREADS] for lane in range(THREADS)]
    return [own[lane] - 2 * own[THREADS - 1 - lane] for lane in range(THREADS)] + GUARD


def _request(root, target, source=None):
    # No index range assertions: every threadgroup subscript must be proven.
    source, descriptor, package = _package(
        root,
        target,
        "int",
        WORKGROUP,
        source=source or _source(),
        software_subgroups=False,
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


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_identity_reinterpret_compiles(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    assert "reinterpret" not in generated
    assert "crosstl_workgroup_tile" in generated
    _compile(generated, target, tmp_path)


def test_identity_reinterpret_compiles_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize("target", ("directx", "opengl"))
@pytest.mark.parametrize(
    "view, peer",
    (
        # A different element type would need byte-level storage.
        ("threadgroup uint*", "const threadgroup uint*"),
        ("threadgroup float*", "const threadgroup float*"),
    ),
)
def test_reinterpret_to_another_element_fails_closed(tmp_path, target, view, peer):
    with pytest.raises(AssertionError, match="pointer-contract-mismatch"):
        _request(tmp_path, target, _source(view, peer))


def test_identity_reinterpret_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 for required native identity reinterpretation"
        )
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


def test_identity_reinterpret_is_required_on_all_native_platforms():
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
    assert "test_workgroup_identity_reinterpret.py" in step
    assert "if:" not in step and "continue-on-error" not in step
