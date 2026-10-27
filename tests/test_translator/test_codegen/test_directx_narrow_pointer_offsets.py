"""DirectX storage roots whose offsets a pointer-reference helper advances.

A narrow source element (8 or 16 bits) is widened into a 32-bit carrier, one
carrier element per source element. Advancing the root through a ``T*&``
helper must keep the root's logical element type, so later pointer helpers
still accept it.
"""

import re

import pytest

import crosstl
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from tests.test_translator.test_project_translation import (
    assert_directx_compute_validates_if_available,
)

SHADER = """
#include <metal_stdlib>
using namespace metal;

METAL_FUNC void advance(
    const device uint32_t*& words,
    const device {scalar}*& values,
    const constant int64_t* strides,
    uint3 tid) {{
  words += tid.z * 4;
  values += tid.z * strides[0];
}}

METAL_FUNC float read_value(const device {scalar}* values, int i) {{
  return float(values[i]);
}}

template <bool batched>
[[kernel]] void advanced_reads(
    const device uint32_t* words [[buffer(0)]],
    const device {scalar}* values [[buffer(1)]],
    device float* out [[buffer(2)]],
    const constant int64_t* strides [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_simdgroup]]) {{
  if (batched) {{
    advance(words, values, strides, tid);
  }}
  out[lid] = read_value(values, int(lid)) + float(words[0]);
}}

template [[host_name("advanced_reads_{batched}")]] [[kernel]] void
advanced_reads<{flag}>(
    const device uint32_t*, const device {scalar}*, device float*,
    const constant int64_t*, uint3, uint);
"""


def translate(tmp_path, scalar, batched):
    path = tmp_path / "advanced_reads.metal"
    path.write_text(
        SHADER.format(scalar=scalar, batched=int(batched), flag=str(batched).lower())
    )
    return crosstl.translate(
        str(path),
        backend="directx",
        format_output=False,
        source_options={"threadgroup_size": (32, 1, 1)},
    )


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize(
    "scalar",
    [
        "uint8_t",
        "uchar",
        "char",
        "int8_t",
        "ushort",
        "short",
        "uint16_t",
        "half",
        "bfloat",
        "float",
        "uint",
    ],
)
def test_advanced_storage_roots_keep_their_logical_element_type(
    tmp_path, scalar, batched
):
    generated = translate(tmp_path, scalar, batched)
    # One carrier element per source element: the advanced offset indexes the
    # carrier directly, with no byte packing.
    assert re.search(r"StructuredBuffer<\w+> values : register\(t1\)", generated)
    assert "int64_t values_offset = int64_t(0);" in generated
    assert re.search(r"read_value\w*\(values, int64_t\(values_offset\)", generated)
    assert "/ 4" not in generated and "% 4" not in generated
    if batched:
        assert re.search(r"inout int64_t values_offset", generated)
    HLSLParser(HLSLLexer(generated).tokenize()).parse()
    assert_directx_compute_validates_if_available(generated, tmp_path)
