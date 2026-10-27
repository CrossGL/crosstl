"""Workgroup reads strided by an ``integral_constant`` are bounded on OpenGL.

A Metal helper may take its stride as a stateless struct whose conversion operator returns a
compile-time value: ``src[i * str_x]`` with ``str_x`` an ``integral_constant<int, 8>`` (``Int<8>{}``).
Member lowering turns the conversion into ``integral_constant_int_8__operator_value_type(str_x)``, a
helper that returns 8. OpenGL subscripts its shared arrays with 32-bit indices and must bound such a read
before it emits it. Three gaps hid the bound and the read was refused:

* the helper's argument is a struct without instance members, which the return analysis did not model;
* a workgroup view helper's body and its call arguments were analyzed without their own lexical types;
* resource-aggregate lowering appends hidden buffer arguments to the call, which the analysis could not
  bind, although every read through such a handle is unknown and none is needed for the result.

MLX's ``BaseMMAFrag::load`` reads threadgroup tiles with ``Int<LD>{}`` strides in the steel GEMM and
attention kernels.
"""

import os
import re
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_INTEGRAL_CONSTANT_STRIDES"
WORKGROUP = (32, 1, 1)
THREADS = WORKGROUP[0]
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(2 * THREADS)]

PRELUDE = """#include <metal_stdlib>
using namespace metal;

namespace tiles {
template <typename T, T v>
struct integral_constant {
  static constexpr constant T value = v;
  using value_type = T;
  constexpr operator value_type() const thread noexcept {
    return value;
  }
};

template <int val>
using Int = integral_constant<int, val>;
}

template <typename StrX>
inline void load_pair(thread int* dst, const threadgroup int* src, StrX str_x) {
  for (short i = 0; i < 2; i++) {
    dst[i] = src[i * str_x];
  }
}
"""

# The threadgroup view is specialized to its backing array and passed as an offset.
SPECIALIZED = PRELUDE + """
[[kernel]] void strided_pairs(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup int tile[64];
  tile[lid] = values[lid];
  tile[lid + 32] = values[lid + 32];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int pair[2];
  load_pair(pair, &tile[lid % 8], tiles::Int<8>{});
  results[lid] = pair[0] - 3 * pair[1];
}
"""

# A struct member pointer into the tile makes every view a resource aggregate reference.
AGGREGATE = PRELUDE + """
struct Writer {
  threadgroup int* dst;
  Writer(threadgroup int* dst_, ushort lid) : dst(dst_ + lid) {}
  void store(int low, int high) const {
    dst[0] = low;
    dst[32] = high;
  }
};

[[kernel]] void strided_view_pairs(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup int tile[64];
  Writer writer(tile, ushort(lid));
  writer.store(values[lid], values[lid + 32]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  int pair[2];
  load_pair(pair, &tile[lid % 8], tiles::Int<8>{});
  results[lid] = pair[0] - 3 * pair[1];
}
"""

CASES = {"strided_pairs": SPECIALIZED, "strided_view_pairs": AGGREGATE}


def _expected():
    return [VALUES[lid % 8] - 3 * VALUES[lid % 8 + 8] for lid in range(THREADS)] + GUARD


def _request(root, target, entry):
    source, descriptor, package = _package(
        root, target, "int", WORKGROUP, source=CASES[entry], software_subgroups=False
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


@pytest.mark.parametrize("entry", sorted(CASES))
@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_integral_constant_strides_compile(tmp_path, target, entry):
    _, request, _ = _request(tmp_path, target, entry)
    generated = request.artifact_path.read_text()
    if target == "opengl":
        # The stride stays a call of the conversion helper; only its bound was missing.
        assert "integral_constant_int_8_operator_value_type" in generated
        # Every subscript of the shared tile is a 32-bit index.
        subscripts = re.findall(r"\w*tile\[([^\]]*)\]", generated)
        assert len(subscripts) > 1
        assert all("int64_t" not in item for item in subscripts), subscripts
    _compile(generated, target, tmp_path)


@pytest.mark.parametrize("entry", sorted(CASES))
def test_integral_constant_strides_compile_on_metal(tmp_path, entry):
    _, request, _ = _request(tmp_path, "metal", entry)
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


@pytest.mark.parametrize("entry", sorted(CASES))
def test_integral_constant_strides_execute(tmp_path, entry):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(
            f"set {REQUIRE_ENV}=1 for required native integral-constant strides"
        )
    from tests.test_translator.test_loop_updates import _execute

    target = {"darwin": "metal", "linux": "opengl", "win32": "directx"}[sys.platform]
    source, request, expected = _request(tmp_path, target, entry)
    _execute(
        request,
        expected,
        tmp_path,
        original_source=source,
        original_entry=entry,
        metal_compile_flags=("-Wall", "-Wextra"),
        compare_with_original=target == "metal",
    )


def test_integral_constant_strides_are_required_on_all_native_platforms():
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
    assert "test_integral_constant_strides.py" in step
    assert "if:" not in step and "continue-on-error" not in step
