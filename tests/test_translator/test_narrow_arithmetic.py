"""Narrow-integer arithmetic of GLSL and HLSL sources wraps at its width on Metal, DirectX, CUDA and HIP.

GLSL (``GL_EXT_shader_explicit_arithmetic_types``) and HLSL with ``-enable-16bit-types`` compute ``uint16_t``
``a * b`` in 16 bits. CrossGL promotes narrow scalar operands to ``int`` (C++ [conv.prom]), so those emitters
compute in 32 bits and the result never wrapped: ``uint((a * b) >> 8)`` kept the high product bits. DXC wraps the
HLSL original at 16 bits (``add i32`` then ``and 65535``). DirectX also keeps 8-bit vectors in ``int2`` carriers,
which never wrapped for any source.
"""

import re
import shutil
import struct
import subprocess

import pytest

import crosstl

GLSL = """#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Inputs { uint v[]; };
layout(std430, binding = 1) buffer Results { uint r[]; };

uint16_t square(uint16_t x) {
    return x * x;
}

void main() {
    uint16_t a = uint16_t(v[0]);
    uint16_t b = uint16_t(v[1]);
    int16_t s = int16_t(int(v[2]));
    int16_t m = int16_t(int(v[3]));
    int8_t c = int8_t(int(v[4]));
    i8vec2 cv = i8vec2(c, c);
    uint16_t p = a * b;
    r[0] = uint((a * b) >> 8);
    r[1] = uint(a + b);
    r[2] = uint(int(s * s));
    r[3] = uint(int(-m));
    r[4] = uint(~a);
    r[5] = uint(int((cv + cv).x));
    r[6] = uint(a << 12);
    r[7] = uint(int(c * c));
    r[8] = uint((a & b) + a);
    r[9] = uint(-a);
    r[10] = uint(p);
    r[11] = uint(square(a));
    r[12] = (a + b) < a ? 1u : 0u;
    r[13] = uint(int(c + c) >> 1);
}
"""

HLSL = """RWStructuredBuffer<uint> v : register(u0);
RWStructuredBuffer<uint> r : register(u1);

uint16_t square(uint16_t x) {
    return x * x;
}

[numthreads(1, 1, 1)]
void CSMain() {
    uint16_t a = (uint16_t)v[0];
    uint16_t b = (uint16_t)v[1];
    int16_t s = (int16_t)(int)v[2];
    int16_t m = (int16_t)(int)v[3];
    uint16_t p = a * b;
    r[0] = (uint)((a * b) >> 8);
    r[1] = (uint)(a + b);
    r[2] = (uint)(int)(s * s);
    r[3] = (uint)(int)(-m);
    r[4] = (uint)(~a);
    r[5] = (uint)(a + 1);
    r[6] = (uint)(a << 12);
    r[7] = (uint)p;
    r[8] = (uint)((a & b) + a);
    r[9] = (uint)(-a);
    r[10] = (uint)square(a);
    r[11] = (a + b) < a ? 1u : 0u;
}
"""

METAL = """#include <metal_stdlib>
using namespace metal;

kernel void k(device const uint* v [[buffer(0)]], device uint* r [[buffer(1)]]) {
    ushort a = ushort(v[0]);
    ushort b = ushort(v[1]);
    char c = char(v[2]);
    char2 cv = char2(c, c);
    r[0] = uint((a * b) >> 8);
    r[1] = uint(int(c * c));
    r[2] = uint(int((cv + cv).x));
    r[3] = uint(int((-cv).y));
}
"""

A, B, S, M, C = 0xFFF3, 0x0123, -300, -32768, 100


def _u16(value):
    return value & 0xFFFF


def _s8(value):
    value &= 0xFF
    return value - 0x100 if value >= 0x80 else value


def _u32(value):
    return value & 0xFFFFFFFF


def _s16(value):
    value &= 0xFFFF
    return value - 0x10000 if value >= 0x8000 else value


def _expected_16_bit():
    """Results of the 16-bit operations both sources share, computed at their width."""
    return {
        0: _u16(A * B) >> 8,
        1: _u16(A + B),
        2: _u32(_s16(S * S)),
        3: _u32(_s16(-M)),
        4: _u16(~A),
        6: _u16(A << 12),
        8: _u16((A & B) + A),
        9: _u16(-A),
    }


def _translate(tmp_path, source, name, backend):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _lines(code):
    return [line.strip() for line in code.splitlines()]


def _compact(code):
    return re.sub(r"\s+", "", code)


@pytest.mark.parametrize(
    ("backend", "product", "negation"),
    [
        ("directx", "uint16_t((uint(a) * uint(b)))", "int16_t(-int(m))"),
        ("cuda", "uint16_t((uint(a) * uint(b)))", "int16_t(-m)"),
        (
            "hip",
            "uint16_t((static_cast<unsigned int>(a) * static_cast<unsigned int>(b)))",
            "int16_t(-m)",
        ),
        ("metal", "ushort(uint(uint(a) * uint(b)))", "short(int(-m))"),
    ],
)
def test_glsl_narrow_scalar_results_are_converted_back(
    tmp_path, backend, product, negation
):
    code = _compact(_translate(tmp_path, GLSL, "shader.comp", backend))
    assert _compact(product) in code
    assert _compact(negation) in code


def test_stores_and_in_range_operations_keep_their_form(tmp_path):
    code = _translate(tmp_path, GLSL, "shader.comp", "directx")
    lines = _lines(code)
    # A store of the operation's own type reduces it already.
    assert "uint16_t p = uint16_t((int(a) * int(b)));" in lines
    assert "return uint16_t((int(x) * int(x)));" in lines
    # ``&`` stays in range; the enclosing ``+`` still wraps.
    assert any("uint16_t((uint((int(a) & int(b))) + uint(a)))" in l for l in lines)
    # A comparison of a wrapped sum compares the 16-bit value.
    assert any("(int(uint16_t((uint(a) + uint(b)))) < int(a))" in l for l in lines)


@pytest.mark.parametrize(
    ("declaration", "expression", "wrapped"),
    [
        ("int16_t", "x / y", True),
        ("uint16_t", "x / y", False),
        ("int16_t", "x % y", False),
        ("uint16_t", "x >> y", False),
        ("uint16_t", "x ^ y", False),
        ("int16_t", "~x", False),
        ("uint16_t", "~x", True),
        ("int16_t", "x - y", True),
    ],
)
def test_only_range_changing_operations_are_converted(
    tmp_path, declaration, expression, wrapped
):
    source = f"""#version 450
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) buffer Results {{ int r[]; }};
void main() {{
    {declaration} x = {declaration}(r[0]);
    {declaration} y = {declaration}(r[1]);
    r[2] = int({expression});
}}
"""
    code = _translate(tmp_path, source, "shader.comp", "directx")
    store = next(l for l in _lines(code) if "Store(8," in l)
    assert (f"{declaration}((" in store or f"{declaration}(~" in store) is wrapped


def test_glsl_byte_vectors_wrap_in_directx_carriers(tmp_path):
    code = _translate(tmp_path, GLSL, "shader.comp", "directx")
    assert "(int2(uint2(int2((cv + cv))) << 24u) >> 24).x" in code


def test_metal_source_keeps_scalar_promotion(tmp_path):
    hlsl = _translate(tmp_path, METAL, "shader.metal", "directx")
    lines = _lines(hlsl)
    # MSL promotes ushort and char operands, so the product keeps all 32 bits.
    assert "r[0] = uint(((int(a) * int(b)) >> 8));" in lines
    assert "r[1] = uint(int((c * c)));" in lines
    # MSL does not promote vectors: char2 arithmetic wraps at 8 bits.
    assert "r[2] = uint(int((int2(uint2(int2((cv + cv))) << 24u) >> 24).x));" in lines
    assert "r[3] = uint(int((int2(uint2(int2(-cv)) << 24u) >> 24).y));" in lines
    metal = _translate(tmp_path, METAL, "shader.metal", "metal")
    assert "uint(a) * uint(b)" not in metal
    assert "uint(c) * uint(c)" not in metal


def _dxc():
    return shutil.which("dxc")


@pytest.mark.parametrize(("source", "name"), [(GLSL, "shader.comp"), (HLSL, "a.hlsl")])
def test_directx_output_compiles_with_dxc(tmp_path, source, name):
    compiler = _dxc()
    if compiler is None:
        pytest.skip("DXC is not installed")
    artifact = tmp_path / "out.hlsl"
    artifact.write_text(_translate(tmp_path, source, name, "directx"))
    result = subprocess.run(
        [compiler, "-T", "cs_6_2", "-E", "CSMain", "-enable-16bit-types", "-WX"]
        + [str(artifact)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_metal(source, words, count):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    entry = re.search(r"kernel void (\w+)\(", source).group(1)
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_(entry), None
    )
    assert pipeline is not None, error
    shared = Metal.MTLResourceStorageModeShared
    data = [
        struct.pack(f"<{len(words)}I", *(_u32(word) for word in words)),
        bytes(4 * count),
    ]
    buffers = [
        device.newBufferWithBytes_length_options_(item, len(item), shared)
        for item in data
    ]
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    for index, buffer in enumerate(buffers):
        encoder.setBuffer_offset_atIndex_(buffer, 0, index)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        Metal.MTLSizeMake(1, 1, 1), Metal.MTLSizeMake(1, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    assert command.error() is None, command.error()
    return struct.unpack(
        f"<{count}I", bytes(buffers[1].contents().as_buffer(4 * count))
    )


def test_glsl_narrow_arithmetic_wraps_on_the_metal_gpu(tmp_path):
    metal = _translate(tmp_path, GLSL, "shader.comp", "metal")
    results = _run_metal(metal, [A, B, S, M, C], 14)
    expected = _expected_16_bit()
    expected.update(
        {
            5: _u32(_s8(C + C)),
            7: _u32(_s8(C * C)),
            10: _u16(A * B),
            11: _u16(A * A),
            12: int(_u16(A + B) < A),
            13: _u32(_s8(C + C) >> 1),
        }
    )
    assert {index: results[index] for index in expected} == expected


def test_hlsl_narrow_arithmetic_wraps_on_the_metal_gpu(tmp_path):
    metal = _translate(tmp_path, HLSL, "a.hlsl", "metal")
    results = _run_metal(metal, [A, B, S, M], 12)
    expected = _expected_16_bit()
    expected.update(
        {
            5: _u16(A + 1),
            7: _u16(A * B),
            10: _u16(A * A),
            11: int(_u16(A + B) < A),
        }
    )
    assert {index: results[index] for index in expected} == expected


def test_glsl_source_is_valid(tmp_path):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    path = tmp_path / "shader.comp"
    path.write_text(GLSL)
    result = subprocess.run(
        [validator, "-V", "--target-env", "vulkan1.1", str(path), "-o", "/dev/null"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
