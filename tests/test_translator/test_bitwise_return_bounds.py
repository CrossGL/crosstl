"""Helper returns built with bitwise operators bound threadgroup view offsets on OpenGL.

Fragment layouts often place a lane with masks: ``qid = lane / 4; fm = (qid & 4) + (lane / 2) % 4;
fn = (qid & 2) * 2 + (lane % 2) * 2; return short2{fn, fm};``. A kernel then views its threadgroup tile at
``&tile[fm * LD + fn]``. OpenGL lowers that view to an offset into the shared array and must prove the
offset fits a 32-bit subscript. The helper-return analysis modeled only ``+ - * / %``, so a single ``&``
left the returned components, and every offset built from them, unbounded and the view was refused.
``&``, ``|`` and ``^`` are now bounded for two's-complement operands: ``x & y`` lies in ``[0, x]`` when
``x >= 0``; for non-negative operands ``x | y`` and ``x ^ y`` set no bit above the wider operand's highest
bit. MLX's ``BaseMMAFrag::get_coord`` places lanes this way in the steel GEMM and attention kernels.
"""

import os
import sys

import pytest

from crosstl.project import build_native_loader_dispatch_request
from crosstl.translator.codegen.integer_intervals import (
    bitwise_interval,
    integer_arithmetic_intervals,
)
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_BITWISE_RETURN_BOUNDS"
ENTRY = "lane_coordinates"
WORKGROUP = (32, 1, 1)
THREADS = WORKGROUP[0]
LEADING = 8
GUARD = [0x6D2B79F5, -7, 0x1234567, -0x5A5A5A5]
VALUES = [((index * 2654435761) % 2003) - 1001 for index in range(8 * LEADING)]

SOURCE = """#include <metal_stdlib>
using namespace metal;

struct Frag {
  static short2 get_coord(ushort lane) {
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    return short2{fn, fm};
  }
};

struct Writer {
  threadgroup int* dst;
  Writer(threadgroup int* dst_, ushort lid) : dst(dst_ + lid) {}
  void store(const device int* values, ushort lid) const {
    dst[0] = values[lid];
    dst[32] = values[lid + 32];
  }
};

inline int read_pair(const threadgroup int* src) {
  return src[0] - 3 * src[1];
}

[[kernel]] void lane_coordinates(
    const device int* values [[buffer(0)]],
    device int* results [[buffer(1)]],
    uint lid [[thread_index_in_threadgroup]]) {
  threadgroup int tile[8 * 8];
  Writer writer(tile, ushort(lid));
  writer.store(values, ushort(lid));
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const short2 coord = Frag::get_coord(ushort(lid));
  const short offset = coord.y * 8 + coord.x;
  results[lid] = read_pair(&tile[offset]);
}
"""


def _coordinates(lane):
    qid = lane // 4
    return (qid & 2) * 2 + (lane % 2) * 2, (qid & 4) + (lane // 2) % 4


def _expected():
    results = []
    for lane in range(THREADS):
        column, row = _coordinates(lane)
        offset = row * LEADING + column
        results.append(VALUES[offset] - 3 * VALUES[offset + 1])
    return results + GUARD


def _request(root, target):
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


@pytest.mark.parametrize("operator", ("&", "|", "^"))
def test_bitwise_intervals_contain_every_result(operator):
    for a in range(-9, 10):
        for b in range(a, 10):
            for c in range(-9, 10):
                for d in range(c, 10):
                    bounds = bitwise_interval(operator, (a, b), (c, d))
                    if bounds is None:
                        # Only sign mixes the docstring excludes stay unbounded.
                        assert (
                            (a < 0 and c < 0) if operator == "&" else (a < 0 or c < 0)
                        )
                        continue
                    values = [
                        eval(f"{x} {operator} {y}")
                        for x in range(a, b + 1)
                        for y in range(c, d + 1)
                    ]
                    assert bounds[0] <= min(values) and max(values) <= bounds[1]


def test_bitwise_intervals_are_typed():
    assert integer_arithmetic_intervals("&", [(0, 7)], [(4, 4)], 16, True) == [(0, 4)]
    assert integer_arithmetic_intervals("|", [(0, 3)], [(4, 4)], 32, False) == [(4, 7)]
    assert integer_arithmetic_intervals("^", [(0, 5)], [(2, 2)], 8, True) == [(0, 7)]
    # A negative operand on both sides of & has no non-negative mask to bound it.
    assert integer_arithmetic_intervals("&", [(-4, -1)], [(-2, 3)], 32, True) is None
    assert integer_arithmetic_intervals("|", [(-1, 0)], [(0, 1)], 32, True) is None


@pytest.mark.parametrize("target", ("directx", "opengl"))
def test_lane_coordinate_views_compile(tmp_path, target):
    _, request, _ = _request(tmp_path, target)
    _compile(request.artifact_path.read_text(), target, tmp_path)


def test_lane_coordinate_views_compile_on_metal(tmp_path):
    _, request, _ = _request(tmp_path, "metal")
    _compile(
        request.artifact_path.read_text(),
        "metal",
        tmp_path,
        metal_compile_flags=("-Wall", "-Wextra"),
    )


def test_lane_coordinate_views_execute(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required native bitwise return bounds")
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


def test_bitwise_return_bounds_are_required_on_all_native_platforms():
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
    assert "test_bitwise_return_bounds.py" in step
    assert "if:" not in step and "continue-on-error" not in step
