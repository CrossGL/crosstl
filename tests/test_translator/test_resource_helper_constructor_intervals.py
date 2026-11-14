"""Struct constructor intervals see through arguments built from read-only resource helper calls.

A Metal tile loader is constructed from a buffer, a stride and a threadgroup array:
``Loader<...> loader(src, strides[0] & 63, tile, lid)``. The constructor stores ``dst_ + bi * dst_ld + bj``
and ``src_ + bi * src_ld + bj``, and a member function then indexes ``dst[i * dst_ld + j]``. OpenGL has no
pointers, so the threadgroup view becomes an offset into the shared array, and the 64-bit pointer offset must be
proven to fit a 32-bit subscript. The interval summary of the constructor refused every argument that contained
a call. The stride argument masks a buffer load, which resource-aggregate lowering emits as a helper that only
reads, so no bound reached ``dst.offset`` and OpenGL refused the store with ``index range unproven``. Such
arguments are now summarized as opaque values; an unmasked runtime stride is still refused. MLX's
``BlockLoader`` in the steel GEMM and attention kernels is constructed the same way.
"""

import os
import re
import sys

import pytest

from crosstl.project import (
    ProjectConfig,
    build_native_loader_dispatch_request,
    translate_project,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_RESOURCE_HELPER_CONSTRUCTOR_INTERVALS"
ENTRY = "tile_loader"
WORKGROUP = (32, 1, 1)
GROUPS = 1
ROWS = 8
COLUMNS = 16
STRIDE_WORD = 64 + COLUMNS
STRIDE = STRIDE_WORD & 63
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(ROWS * STRIDE)]

SOURCE_TEMPLATE = """#include <metal_stdlib>
using namespace metal;

template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size>
struct Loader {
  static constant constexpr const short n_reads = (BCOLS * BROWS) / tgp_size;
  static constant constexpr const short TCOLS = BCOLS / n_reads;
  static constant constexpr const short TROWS = tgp_size / TCOLS;

  const int src_ld;
  const short thread_idx;
  const short bi;
  const short bj;
  threadgroup T* dst;
  const device T* src;

  Loader(const device T* src_, const int src_ld_, threadgroup T* dst_, ushort thread_idx_)
      : src_ld(src_ld_),
        thread_idx(thread_idx_),
        bi(thread_idx / TCOLS),
        bj(n_reads * (thread_idx % TCOLS)),
        dst(dst_ + bi * dst_ld + bj),
        src(src_ + bi * src_ld + bj) {}

  void load_unsafe() const {
    for (short i = 0; i < BROWS; i += TROWS) {
      for (short j = 0; j < n_reads; j++) {
        dst[i * dst_ld + j] = src[i * src_ld + j];
      }
    }
  }
};

[[kernel]] void tile_loader(
    const device int* values [[buffer(0)]],
    const constant int* strides [[buffer(1)]],
    device int* results [[buffer(2)]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup int tile[8 * 20];
  Loader<int, 8, 16, 20, 32> loader(values, STRIDE_ARGUMENT, tile, ushort(lid));
  loader.load_unsafe();
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint k = lid; k < 128; k += 32) {
    results[k] = tile[(k / 16) * 20 + k % 16];
  }
}
"""

SOURCE = SOURCE_TEMPLATE.replace("STRIDE_ARGUMENT", "strides[0] & 63")


def _expected():
    tile = [
        VALUES[row * STRIDE + column]
        for row in range(ROWS)
        for column in range(COLUMNS)
    ]
    return tile * GROUPS + GUARD


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
            "strides": {"dtype": "int32", "shape": [1], "values": [STRIDE_WORD]},
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
def test_masked_stride_loader_compiles(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    if target == "opengl":
        # The threadgroup view is an offset into the shared tile, subscripted with a 32-bit index.
        assert "shared int crosstl_workgroup_tile[160]" in generated
        subscripts = re.findall(r"crosstl_workgroup_tile\[(int\(.*?\))\]", generated)
        assert subscripts and all("int64_t" not in item for item in subscripts)
    _compile(generated, target, tmp_path)


def test_masked_stride_loader_compiles_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_unmasked_runtime_stride_is_refused_on_opengl(tmp_path):
    filename = "loader.metal"
    (tmp_path / filename).write_text(
        SOURCE_TEMPLATE.replace("STRIDE_ARGUMENT", "strides[0]"), encoding="utf-8"
    )
    report = translate_project(
        ProjectConfig(
            root=tmp_path,
            include_patterns=(filename,),
            targets=("opengl",),
            output_dir="out",
            workgroup_size=WORKGROUP,
        ),
        format_output=False,
    ).to_json()
    assert report["summary"]["translatedCount"] == 0
    errors = [
        diagnostic
        for diagnostic in report["diagnostics"]
        if diagnostic.get("severity") == "error"
    ]
    assert errors and "index range unproven" in errors[0]["message"], errors


def test_masked_stride_loader_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 for required native resource helper intervals"
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


def test_resource_helper_constructor_intervals_are_required_on_all_native_platforms():
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
    assert "test_resource_helper_constructor_intervals.py" in step
    assert "if:" not in step and "continue-on-error" not in step
