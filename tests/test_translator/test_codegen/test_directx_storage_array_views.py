"""DirectX storage array parameters that receive views into a buffer.

A Metal array parameter in a storage address space is a pointer to its first
element, so a caller may pass ``strides + 2``. HLSL passes buffers only whole,
so such a parameter takes the storage-pointer base-offset ABI, while a
parameter that only receives whole buffers keeps the plain buffer parameter.
"""

import re

import crosstl
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from tests.test_translator.test_project_translation import (
    assert_directx_compute_validates_if_available,
)

VIEWS = """
#include <metal_stdlib>
using namespace metal;

int64_t inner(constant const int64_t s[2]) { return s[0] + 2 * s[1]; }

int64_t outer(constant const int64_t s[2]) { return inner(s) + inner(s + 1); }

void bump(device int v[2], int amount) { v[1] += amount; }

int64_t through_local(constant const int64_t* base) {
  constant const int64_t* view = base + 1;
  return inner(view);
}

kernel void views(
    device int* out [[buffer(0)]],
    constant const int64_t* strides [[buffer(1)]],
    uint pos [[thread_position_in_grid]]) {
  bump(out + 4 * pos, int(outer(strides + 2)));
  out[0] += int(through_local(strides));
}
"""

WHOLE = """
#include <metal_stdlib>
using namespace metal;

int64_t whole(constant const int64_t s[2]) { return s[0] + 2 * s[1]; }

kernel void whole_buffers(
    device int* out [[buffer(0)]],
    constant const int64_t strides[2] [[buffer(1)]],
    uint pos [[thread_position_in_grid]]) {
  out[pos] = int(whole(strides));
}
"""


def translate(tmp_path, source):
    path = tmp_path / "views.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def test_storage_array_parameters_receiving_views_take_offsets(tmp_path):
    generated = translate(tmp_path, VIEWS)

    # Prototypes and definitions share the offset ABI.
    for signature in (
        "int64_t inner(StructuredBuffer<int64_t> s, int64_t s_offset)",
        "int64_t outer(StructuredBuffer<int64_t> s, int64_t s_offset)",
        "void bump(RWStructuredBuffer<int> v, int64_t v_offset, int amount)",
    ):
        assert generated.count(signature) == 2, signature
    assert "s[uint(s_offset)]" in generated
    assert "s[uint((s_offset + 1))]" in generated
    assert "v[uint((v_offset + 1))] += amount;" in generated
    # A forwarded view adds its own offset; a local pointer passes its offset.
    assert "inner(s, int64_t((s_offset + 1)))" in generated
    assert "inner(base, int64_t(view_offset))" in generated
    assert "outer(strides, int64_t(2))" in generated
    assert "bump(out_, int64_t((4 * pos))," in generated
    # The view is passed by value even though its elements are written.
    assert not re.search(r"inout int64_t v_offset", generated)
    # No buffer is offset as if it were a pointer.
    assert not re.search(r"\b(strides|out_|s|base) [+-] ", generated)
    HLSLParser(HLSLLexer(generated).tokenize()).parse()
    assert_directx_compute_validates_if_available(generated, tmp_path)


def test_storage_array_parameters_receiving_whole_buffers_stay_plain(tmp_path):
    generated = translate(tmp_path, WHOLE)

    assert "int64_t whole(StructuredBuffer<int64_t> s)" in generated
    assert "s_offset" not in generated
    assert "whole(strides)" in generated
    HLSLParser(HLSLLexer(generated).tokenize()).parse()
    assert_directx_compute_validates_if_available(generated, tmp_path)
