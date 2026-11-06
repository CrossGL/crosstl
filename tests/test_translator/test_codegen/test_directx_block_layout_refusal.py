"""HLSL refuses, with a typed diagnostic, a block whose source layout its declarations cannot reproduce."""

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_refusals import DirectXBufferLayoutError

SIXTEEN_BIT_PUSH_CONSTANTS = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
layout(local_size_x = 8) in;
layout(std430, binding = 0) buffer Out { uint values[]; } outBuf;
layout(push_constant) uniform Registers { uint16_t count; uint scale; } registers;
void main() {
    outBuf.values[gl_GlobalInvocationID.x] = uint(registers.count) * registers.scale;
}
"""


def test_block_member_without_a_buffer_layout_is_a_typed_refusal(tmp_path):
    path = tmp_path / "registers.comp"
    path.write_text(SIXTEEN_BIT_PUSH_CONSTANTS, encoding="utf-8")
    with pytest.raises(DirectXBufferLayoutError) as refused:
        crosstl.translate(str(path), backend="directx", format_output=False)
    error = refused.value
    assert error.project_diagnostic_code == (
        "project.translate.directx-buffer-layout-unsupported"
    )
    assert error.block == "Registers"
    assert error.layout == "std430"
    assert "uint16_t" in str(error)
