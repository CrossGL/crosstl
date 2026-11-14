"""HLSL: a pointer alias's offset applies to accesses through the alias, never to its root buffer."""

import re

import pytest

import crosstl

TWO_ALIASES = """#include <metal_stdlib>
using namespace metal;
kernel void k(device float* data [[buffer(0)]], device float* out [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
  device float* first = data + 4;
  device float* second = data + 8;
  out[tid] = data[tid] + first[tid] * 10.0f + second[tid] * 100.0f;
  data[tid] = 1.0f;
  second[tid] = 2.0f;
}
"""

VECTOR_ROWS = """#include <metal_stdlib>
using namespace metal;
kernel void k(device const float2* partials [[buffer(0)]], device float* out [[buffer(1)]],
              uint tid [[thread_position_in_grid]]) {
  device const float2* row_partials = partials + tid * 4;
  float total = 0.0f;
  for (uint i = 0; i < 4; i++) {
    total += partials[i].x + row_partials[i].y;
  }
  out[tid] = total;
}
"""


def _hlsl(tmp_path, source):
    path = tmp_path / "aliases.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend="directx", format_output=False)


def test_root_buffer_accesses_take_no_alias_offset(tmp_path):
    output = _hlsl(tmp_path, TWO_ALIASES)
    # Reads and writes of `data` itself index it directly.
    assert "data.Load(tid)" in output
    assert re.search(r"\bdata\[tid\] = 1\.0", output)
    # Each alias keeps its own offset.
    assert "data[uint((first_offset + tid))] * 10.0" in output
    assert "data[uint((second_offset + tid))] * 100.0" in output
    assert re.search(r"data\[uint\(\(second_offset \+ tid\)\)\] = 2\.0", output)
    assert "first_offset + tid))] = 1.0" not in output


def test_vector_root_and_alias_components(tmp_path):
    output = _hlsl(tmp_path, VECTOR_ROWS)
    assert "partials.Load(i).x" in output
    assert "partials[uint((row_partials_offset + i))].y" in output


@pytest.mark.parametrize("backend", ("opengl", "vulkan"))
def test_other_targets_translate_the_same_source(tmp_path, backend):
    path = tmp_path / "aliases.metal"
    path.write_text(TWO_ALIASES)
    assert crosstl.translate(str(path), backend=backend, format_output=False)
