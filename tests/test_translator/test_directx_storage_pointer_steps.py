"""DirectX lowering of ``++`` and ``--`` on Metal storage pointers.

DirectX lowers a storage pointer to its backing buffer plus an element offset.
A step whose value is discarded (an expression statement or a loop update)
moves that offset by one element, exactly as ``p += 1`` does, for local
pointers and helper parameters alike. A step whose value is used is refused
with a typed diagnostic instead of naming the lowered pointer.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.directx_codegen import (
    DirectXResourcePointerParameterError,
)
from tests.test_translator.test_project_translation import (
    assert_directx_compute_validates_if_available,
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
        backend="directx",
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
    const device float* scales [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {{
  float acc = 0.0f;
  for (int row = 0; row < 4; row++) {{
    const device float* sl = scales + 8 + row * 8;
    for (int k = 0; k < 2; ++k) {{
      acc += sl[0];
      {step}
    }}
  }}
  out[lid] = acc;
}}
""",
    )
    assert re.search(
        r"int64_t sl_offset = int64_t\(\(?8 \+ \(row \* 8\)\)?\);", generated
    )
    assert f"sl_offset {expected} 1;" in generated
    assert not re.search(r"(\+\+|--)\s*sl\b|\bsl\s*(\+\+|--)", generated)
    assert_directx_compute_validates_if_available(generated, tmp_path)


def test_parameter_storage_pointer_steps_move_their_offsets(tmp_path):
    generated = translate(
        tmp_path,
        """
METAL_FUNC float walk(const device float* s, int n) {
  float acc = 0.0f;
  for (int k = 0; k < n; ++k) {
    acc += s[0];
    s++;
  }
  return acc;
}

[[kernel]] void steps(
    const device float* scales [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {
  const device float* p = scales + lid;
  ++p;
  p--;
  out[lid] = walk(p, 3);
}
""",
    )
    helper = re.search(r"float walk\w*\([^)]*\) \{.*?\n\}", generated, re.S)
    assert helper is not None
    assert "s_offset += 1;" in helper.group(0)
    assert re.search(r"p_offset \+= 1;\s*p_offset -= 1;", generated), generated
    assert not re.search(r"\b[sp]\s*(\+\+|--)|(\+\+|--)\s*[sp]\b", generated)
    assert_directx_compute_validates_if_available(generated, tmp_path)


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
    assert re.search(r"for \([^;]*;[^;]*; k\+\+, p_offset \+= 1\)", generated)
    assert "p++" not in generated
    assert_directx_compute_validates_if_available(generated, tmp_path)


def test_storage_pointer_step_whose_value_is_used_is_refused(tmp_path):
    with pytest.raises(DirectXResourcePointerParameterError, match="step"):
        translate(
            tmp_path,
            """
[[kernel]] void steps(
    const device float* values [[buffer(0)]],
    device float* out [[buffer(1)]],
    uint lid [[thread_position_in_grid]]) {
  const device float* p = values + lid;
  const device float* q = p++;
  out[lid] = q[0] + p[0];
}
""",
        )
