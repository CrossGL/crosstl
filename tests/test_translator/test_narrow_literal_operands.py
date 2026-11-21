"""A literal operand of a narrow integer or half value takes the type the source language gives it.

HLSL gives an unsuffixed int literal the other operand's type and MSL converts a scalar to a vector operand's
element type, so ``int16_t a; a * 2`` stays 16-bit in HLSL. GLSL and Slang never convert an int implicitly to a
narrower type, so ``int8_t a; a * 2`` and ``i8vec2 v; v * 2`` are ``int`` operations there (glslc and slangc write
``OpIMul %int`` and ``OpIMul %v2int``). A floating-point literal is a ``float`` operand of an integer operation in
every source, and a Slang float literal is a ``float`` beside a ``half`` (slangc writes ``OpFMul %float``).
"""

import pytest

import crosstl

SLANG_PRELUDE = (
    "RWStructuredBuffer<int> iin;\n"
    "RWStructuredBuffer<float> outp;\n"
    '[shader("compute")]\n'
    "[numthreads(1, 1, 1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) {\n"
)
GLSL_PRELUDE = (
    "#version 450\n"
    "#extension GL_EXT_shader_explicit_arithmetic_types : require\n"
    "layout(local_size_x = 1) in;\n"
    "layout(std430, binding = 0) buffer Inp { int a[]; };\n"
    "layout(std430, binding = 1) buffer Outp { float o[]; };\n"
    "void main() {\n"
)
HLSL_PRELUDE = (
    "RWStructuredBuffer<int> iin;\n"
    "RWStructuredBuffer<float> outp;\n"
    "[numthreads(1, 1, 1)]\n"
    "void main(uint3 id : SV_DispatchThreadID) {\n"
)


def translate(tmp_path, suffix, prelude, body, backend):
    path = tmp_path / f"kernel{suffix}"
    path.write_text(prelude + body + "}\n")
    options = {"enable_16bit_types": True} if suffix == ".hlsl" else None
    return crosstl.translate(
        str(path), backend=backend, format_output=False, source_options=options
    )


def test_slang_scalar_operation_with_an_int_literal_is_an_int_operation(tmp_path):
    body = (
        "    int8_t a = int8_t(iin[id.x]);\n"
        "    outp[0] = float(a * 2);\n"
        "    outp[1] = float(a + 1);\n"
        "    outp[2] = float(a * int8_t(3));\n"
    )
    metal = translate(tmp_path, ".slang", SLANG_PRELUDE, body, "metal")
    # ``a = 100`` gives 200 and 101, not the 8-bit -56 and 101; a narrow operand pair still wraps at 8 bits.
    assert "outp[0] = float(a * 2);" in metal
    assert "outp[1] = float(a + 1);" in metal
    assert "char(int(uint(a) * uint(int(char(int(3))))))" in metal


@pytest.mark.parametrize("backend", ["metal", "directx"])
def test_slang_vector_operation_with_an_int_literal_widens_the_vector(
    tmp_path, backend
):
    body = (
        "    vector<uint8_t, 2> u = vector<uint8_t, 2>(uint8_t(iin[0]), uint8_t(iin[1]));\n"
        "    vector<int16_t, 3> s = vector<int16_t, 3>(int16_t(iin[0]));\n"
        "    outp[0] = float((u * 2).x);\n"
        "    outp[1] = float((u * 2u).x);\n"
        "    outp[2] = float((s & 255).x);\n"
        "    outp[3] = float((2 - s).x);\n"
    )
    output = translate(tmp_path, ".slang", SLANG_PRELUDE, body, backend)
    assert "(int2(u) * 2)" in output
    assert "(uint2(u) * 2u)" in output
    assert "(int3(s) & 255)" in output
    assert "(2 - int3(s))" in output


def test_glsl_vector_operation_with_an_int_literal_widens_the_vector(tmp_path):
    body = (
        "    int8_t s = int8_t(a[0]);\n"
        "    i8vec2 v = i8vec2(s, s);\n"
        "    o[0] = float((v * 2).x);\n"
        "    o[1] = float((v * s).x);\n"
    )
    metal = translate(tmp_path, ".comp", GLSL_PRELUDE, body, "metal")
    assert "char2(v * 2)" not in metal
    assert "as_type<uint2>(int2(v)) * 2u" in metal
    # Two narrow operands keep the 8-bit operation of GLSL.
    assert "int2(char2(v * s))" in metal


def test_float_literal_operand_of_a_narrow_integer_is_a_float_operation(tmp_path):
    body = (
        "    int16_t a = int16_t(iin[id.x]);\n"
        "    outp[0] = a * 1.5;\n"
        "    outp[1] = a * 2.0;\n"
    )
    spirv = translate(tmp_path, ".hlsl", HLSL_PRELUDE, body, "vulkan")
    # ``a = 3`` stores 4.5 and 6.0; reducing the product to int16_t would store 4.
    assert "OpFMul" in spirv
    assert "OpConvertFToS" not in spirv


def test_hlsl_int_literal_operand_keeps_the_narrow_width(tmp_path):
    body = "    int16_t a = int16_t(iin[id.x]);\n    outp[0] = float(a * 2);\n"
    metal = translate(tmp_path, ".hlsl", HLSL_PRELUDE, body, "metal")
    assert "short(int(uint(a) * uint(2)))" in metal


def test_slang_float_literal_beside_half_is_a_float_operation(tmp_path):
    body = (
        "    half h = half(iin[id.x]);\n"
        "    outp[0] = float(h * 1.1);\n"
        "    outp[1] = float(h * 3);\n"
    )
    wgsl = translate(tmp_path, ".slang", SLANG_PRELUDE, body, "wgsl")
    # ``h * 1.1`` is a binary32 product; ``h * 3`` converts the int to half and rounds the half product.
    assert "outp[0] = f32((h * 1.1));" in wgsl
    assert "outp[1] = f32(cgl_half_round((h * 3.0)));" in wgsl
