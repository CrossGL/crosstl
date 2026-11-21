"""Slang output promotes 8- and 16-bit integer scalar operands of promoting sources to ``int``.

CrossGL, MSL, CUDA, HIP and OpenCL C compute ``v * v`` on ``int8_t v = 12`` as the ``int`` 144 (C17 6.3.1.1);
Slang computes it at 8 bits and wraps to -112, which was the ``integer-promotion`` wrong row of the Slang output
coverage. Sources without promotion (GLSL, HLSL, SPIR-V, Slang) keep Slang's narrow arithmetic. Every output is
compiled with slangc and validated with spirv-val when they are installed, and the compute case runs on a host
Vulkan runtime when one is available.
"""

from __future__ import annotations

import pytest

import crosstl
from crosstl.translator.codegen.slang_refusals import SlangLoweringRefused
from tests.test_translator.test_slang_value_resizes import compile_slang, run_compute

PROMOTION = """shader P {
    int8_t square(int8_t a) { return a * a; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer int* products;
        buffer int* stored;
        buffer int* complements;
        buffer int* negations;
        buffer int* shifted;
        buffer int* returned;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int8_t v = int8_t(iin[i]);
            uint8_t u = uint8_t(iin[i]);
            products[i] = v * v;
            int8_t c = v * v + 1;
            stored[i] = int(c);
            complements[i] = ~u;
            negations[i] = -v;
            uint8_t s = u;
            s <<= 9;
            shifted[i] = int(s);
            returned[i] = int(square(v));
        }
    }
}
"""

INPUTS = [0, 1, 12, 15, -128, 127, 100, -7]


def int8(value):
    value &= 0xFF
    return value - 0x100 if value >= 0x80 else value


def translate(tmp_path, source, suffix="cgl"):
    path = tmp_path / f"probe.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend="slang")


def test_narrow_scalar_operands_are_promoted_to_int(tmp_path):
    code = translate(tmp_path, PROMOTION)
    assert "products[i] = int(v) * int(v);" in code
    assert "complements[i] = ~int(u);" in code
    assert "negations[i] = -int(v);" in code
    compile_slang(tmp_path, code)


def test_a_promoted_result_stored_into_a_narrow_scalar_is_converted(tmp_path):
    code = translate(tmp_path, PROMOTION)
    assert "int8_t c = int8_t(int(v) * int(v) + 1);" in code
    assert "return int8_t(int(a) * int(a));" in code


def test_a_narrow_shift_assignment_shifts_the_promoted_value(tmp_path):
    code = translate(tmp_path, PROMOTION)
    assert "s = uint8_t(int(s) << 9);" in code


def test_promoted_arithmetic_computes_the_c_result(tmp_path):
    binary = compile_slang(tmp_path, translate(tmp_path, PROMOTION))
    zeros = [0] * len(INPUTS)
    results = run_compute(
        tmp_path, binary, [("i", INPUTS)] + [("i", list(zeros)) for _ in range(6)]
    )
    narrow = [int8(x) for x in INPUTS]
    unsigned = [x & 0xFF for x in INPUTS]
    assert results[1] == [v * v for v in narrow]
    assert results[2] == [int8(v * v + 1) for v in narrow]
    assert results[3] == [-1 - u for u in unsigned]
    assert results[4] == [-v for v in narrow]
    assert results[5] == zeros
    assert results[6] == [int8(v * v) for v in narrow]


def test_narrow_vector_arithmetic_is_not_promoted(tmp_path):
    source = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int8_t v = int8_t(iin[i]);
            i8vec2 w = i8vec2(v, v);
            i8vec2 p = w * w;
            outp[i] = int(p.x);
        }
    }
}
"""
    code = translate(tmp_path, source)
    assert "p = w * w;" in code
    compile_slang(tmp_path, code)


def test_a_metal_char_product_is_promoted(tmp_path):
    source = """#include <metal_stdlib>
using namespace metal;
kernel void k(device const int* iin [[buffer(0)]], device float* outp [[buffer(1)]],
              uint i [[thread_position_in_grid]]) {
    char v = char(iin[i]);
    outp[i] = float(v * v);
}
"""
    code = translate(tmp_path, source, "metal")
    assert "float(int(v) * int(v))" in code
    compile_slang(tmp_path, code)


def test_a_glsl_source_keeps_narrow_arithmetic(tmp_path):
    source = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer I { int iin[]; };
layout(std430, binding = 1) buffer O { int outp[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
    int8_t v = int8_t(iin[i]);
    int8_t c = v * v;
    outp[i] = int(c);
}
"""
    code = translate(tmp_path, source, "comp")
    assert "int(v)" not in code
    compile_slang(tmp_path, code)


def test_a_narrow_division_assignment_with_side_effects_is_refused(tmp_path):
    source = """shader P {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer int* iin;
        buffer int* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            int8_t v = int8_t(iin[i]);
            int8_t values[2];
            values[0] = v;
            values[1] = v;
            uint k = 0u;
            values[k++] /= v;
            outp[i] = int(values[0]);
        }
    }
}
"""
    with pytest.raises(SlangLoweringRefused, match="compound assignment /="):
        translate(tmp_path, source)
