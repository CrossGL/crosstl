"""Subscripted vector components keep their scalar type in DirectX assignments."""

import os
import re
import struct
import sys
from pathlib import Path

import pytest

from crosstl.project import build_native_loader_dispatch_request
from tests.test_translator.test_boolean_buffer_runtime import _bound_values
from tests.test_translator.test_loop_updates import _execute
from tests.test_translator.test_metal_builtin_ownership import _compile
from tests.test_translator.test_software_subgroup_product import _package

REQUIRE_ENV = "CROSTL_REQUIRE_VECTOR_COMPONENT_NARROWING"
NATIVE_TARGET = {"darwin": "metal", "win32": "directx", "linux": "opengl"}.get(
    sys.platform
)
SOURCE = """#include <metal_stdlib>
using namespace metal;
typedef bfloat bfloat16_t;
struct Tile {
  float2 frags[2];
  thread float* elems() { return reinterpret_cast<thread float*>(frags); }
};
struct Narrow {
  static bfloat16_t apply(float x) { return static_cast<bfloat16_t>(x); }
};
struct Block {
  Tile ctile;
  void finish(device float* results, uint lane) {
    for (short i = 0; i < 4; i++) { ctile.elems()[i] = Narrow::apply(ctile.elems()[i]); }
    for (short i = 0; i < 4; i++) { results[lane * 4 + i] = ctile.elems()[i]; }
  }
};
kernel void components(device float* results [[buffer(0)]],
                       const device float* values [[buffer(1)]],
                       uint lane [[thread_position_in_grid]]) {
  Block b;
  b.ctile.frags[0] = float2(values[lane], 2.0f);
  b.ctile.frags[1] = float2(3.0f, values[lane + 1]);
  b.finish(results, lane);
}
"""
# Values one bfloat16 step apart from their neighbours, so rounding shows.
INPUTS = (1.0 + 2.0**-10, 1.0 + 3 * 2.0**-8, -2.0 - 3 * 2.0**-8, 0.1, 255.5)


def _bfloat16(value):
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _expected():
    values = []
    for lane in range(4):
        values.extend(
            _bfloat16(value) for value in (INPUTS[lane], 2.0, 3.0, INPUTS[lane + 1])
        )
    return values


def _request(root, target):
    _, descriptor, package = _package(
        root, target, "float", (4, 1, 1), source=SOURCE, software_subgroups=False
    )
    inputs = _bound_values(
        descriptor,
        {
            "values": {"dtype": "float32", "shape": [5], "values": list(INPUTS)},
            "results": {"dtype": "float32", "shape": [16], "values": [0.0] * 16},
        },
    )
    outputs = _bound_values(
        descriptor,
        {"results": {"dtype": "float32", "shape": [16], "values": _expected()}},
    )
    request = build_native_loader_dispatch_request(
        descriptor,
        package,
        inputs,
        outputs,
        {"workgroupCount": [1, 1, 1], "workgroupSize": [4, 1, 1]},
        expected_target=target,
    )
    assert not request.execution_plan.diagnostics
    return request, outputs


def test_expected_values_round_to_bfloat16():
    assert _bfloat16(INPUTS[0]) == 1.0
    assert _bfloat16(INPUTS[1]) == 1.015625
    assert _bfloat16(INPUTS[2]) == -2.015625
    assert _expected()[:4] == [1.0, 2.0, 3.0, 1.015625]


@pytest.mark.parametrize("target", ["metal", "directx", "opengl"])
def test_vector_component_narrowing_compiles(tmp_path, target):
    request, _ = _request(tmp_path, target)
    generated = request.artifact_path.read_text()
    if target == "directx":
        # The component receives a decoded scalar, never a vector conversion.
        assert re.search(
            r"frags\[[^\n]*\]\[[^\n]*\] = __crossgl_bfloat16_to_float\(", generated
        )
    _compile(generated, target, tmp_path)


def test_vector_component_narrowing_executes(tmp_path):
    if os.environ.get(REQUIRE_ENV) != "1":
        pytest.skip(f"set {REQUIRE_ENV}=1 for required component execution")
    request, outputs = _request(tmp_path, NATIVE_TARGET)
    _execute(
        request,
        outputs,
        tmp_path,
        original_source=SOURCE if NATIVE_TARGET == "metal" else None,
        original_entry="components",
        metal_compile_flags=("-Wall", "-Wextra", "-fno-fast-math"),
    )


def test_vector_component_narrowing_execution_is_required():
    from tests import workflow_text

    workflow = (
        Path(__file__).resolve().parents[2]
        / ".github/workflows/demo-project-testing.yml"
    ).read_text()
    step = workflow_text.workflow_job_step_section(
        workflow, "portable-host", "Validate collective helper arguments"
    )
    assert f'{REQUIRE_ENV}: "1"' in step
    assert "test_directx_vector_component_narrowing.py" in step
    assert "if:" not in step and "continue-on-error" not in step
