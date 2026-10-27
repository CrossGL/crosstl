"""OpenGL lowering of ``++`` and ``--`` on Metal storage pointers.

OpenGL lowers a storage pointer to its backing buffer plus an ``int`` element
offset. A step whose value is discarded (an expression statement or a loop
update) moves that offset by one element, exactly as ``p += 1`` does, for
local pointers, entry buffers and helper parameters alike.
"""

import re

import pytest

import crosstl
from tests.test_backend.test_metal.test_codegen import (
    assert_opengl_compute_validates_if_available,
)

HEADER = """
#include <metal_stdlib>
using namespace metal;
"""


def translate(tmp_path, body):
    path = tmp_path / "steps.metal"
    path.write_text(HEADER + body)
    return crosstl.translate(
        str(path),
        backend="opengl",
        format_output=False,
        source_options={"threadgroup_size": (32, 1, 1)},
    )


@pytest.mark.parametrize(
    "step, expected",
    [("sl++;", "+="), ("++sl;", "+="), ("sl--;", "-="), ("--sl;", "-=")],
)
def test_local_storage_pointer_step_moves_its_offset(tmp_path, step, expected):
    generated = translate(
        tmp_path,
        f"""
[[kernel]] void steps(
    const device uint8_t* scales [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {{
  float acc = 0.0f;
  for (int row = 0; row < 4; row++) {{
    const device uint8_t* sl = scales + 8 + row * 8;
    for (int k = 0; k < 2; ++k) {{
      acc += float(sl[0]);
      {step}
    }}
  }}
  out[lid] = acc;
}}
""",
    )
    assert "int sl_offset = int((8 + (row * 8)));" in generated
    assert f"(sl_offset {expected} 1);" in generated
    assert "scales[sl_offset]" in generated
    assert_opengl_compute_validates_if_available(generated, tmp_path, "local")


def test_parameter_storage_pointer_steps_move_their_offsets(tmp_path):
    generated = translate(
        tmp_path,
        """
METAL_FUNC float walk(const device uint8_t* s, int n) {
  float acc = 0.0f;
  for (int k = 0; k < n; ++k) {
    acc += float(s[0]);
    s++;
  }
  return acc;
}

[[kernel]] void steps(
    const device uint8_t* scales [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {
  scales += lid;
  ++scales;
  scales--;
  out[lid] = walk(scales, 3);
}
""",
    )
    helper = re.search(
        r"float walk\w*\(int n, int s_offset\) \{.*?\n\}", generated, re.S
    )
    assert helper is not None
    assert "(s_offset += 1);" in helper.group(0)
    assert "acc += float(scales[s_offset]);" in helper.group(0)
    main = generated[generated.index("void main()") :]
    assert re.search(r"\(scales_offset \+= 1\);\s*\(scales_offset -= 1\);", main), main
    assert re.search(r"walk\w*\(3, int\(scales_offset\)\)", main)
    assert_opengl_compute_validates_if_available(generated, tmp_path, "parameter")


def test_storage_pointer_loop_update_step_moves_its_offset(tmp_path):
    generated = translate(
        tmp_path,
        """
[[kernel]] void steps(
    const device float* values [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {
  float acc = 0.0f;
  const device float* p = values + lid;
  for (int k = 0; k < 4; k++, p++) {
    acc += p[0];
  }
  out[lid] = acc;
}
""",
    )
    assert "(p_offset += 1)" in generated
    assert "acc += values[p_offset];" in generated
    assert_opengl_compute_validates_if_available(generated, tmp_path, "loop")


def test_storage_pointer_step_whose_value_is_used_reads_the_old_element(tmp_path):
    generated = translate(
        tmp_path,
        """
[[kernel]] void steps(
    const device uint8_t* scales [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {
  const device uint8_t* sl = scales + lid;
  float first = float(*(sl++));
  out[lid] = first + float(sl[0]);
}
""",
    )
    assert "scales[(sl_offset++)]" in generated
    assert "scales[sl_offset]" in generated
    assert_opengl_compute_validates_if_available(generated, tmp_path, "used")
