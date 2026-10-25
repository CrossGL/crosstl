"""GLSL 8- and 16-bit integer arithmetic wraps at the narrow width.

The GLSL emitters keep 8- and 16-bit integer scalars and vectors in 32-bit carriers (``int``/``uint`` and
``ivecN``/``uvecN``) and compute on promoted 32-bit operands. Same-type narrow binary operations keep their narrow
width, so ``float(v * v)`` on ``int8_t v = 13`` wraps at 8 bits to ``-87.0`` and ``(n + m).x`` on ``u16vec2`` wraps
modulo 2^16. A 32-bit operand promotes the narrow value first, and an integer literal adopts a narrow vector's type
while promoting a narrow scalar to 32 bits.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.glsl_narrow_vectors import (
    NarrowShape,
    narrow_vector_binary_result,
    narrow_vector_unary_result,
)
from crosstl.translator.codegen.webgl_codegen import WebGLCodeGen

U16X2 = NarrowShape(False, 16, 2)
I16X2 = NarrowShape(True, 16, 2)
U16 = NarrowShape(False, 16, 1)
I8 = NarrowShape(True, 8, 1)
U8 = NarrowShape(False, 8, 1)
U8X2 = NarrowShape(False, 8, 2)

WRAP_PROBE = """
shader NarrowWrap {
    StructuredBuffer<u16vec2> vecs;
    StructuredBuffer<i16vec2> signedVecs;
    RWStructuredBuffer<uint> output;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            u16vec2 n = vecs[0];
            u16vec2 m = vecs[1];
            i16vec2 s = signedVecs[0];
            uvec2 w = uvec2(1u);
            output[0] = (n + m).x;
            output[1] = (n * m).y;
            output[2] = (n + w).x;
            output[3] = (~n).y;
            output[4] = (vecs[0] + vecs[1]).x;
            output[5] = uint((s + s).x + 40000);
            output[6] = ((n + m) * m).y;
            output[7] = (n << 4).x;
            output[8] = uint((-s).x + 40000);
            u16vec2 c = n;
            c += m;
            output[9] = c.x;
            u16vec2 e = n;
            e++;
            output[10] = e.x;
        }
    }
}
"""

# vecs = [(65535, 300), (1, 300)] and signedVecs = [(20000, 0)], stored in the 32-bit carriers the GLSL buffers
# declare. A 32-bit operand promotes the narrow vector first, so (n + w).x is 65536.
WRAP_EXPECTED = (
    0,
    90000 % 65536,
    65536,
    0xFFFF & ~300,
    0,
    40000 - 25536,
    (600 * 300) % 65536,
    (65535 << 4) % 65536,
    40000 - 20000,
    0,
    0,
)

FRAGMENT = """
shader NarrowFragment {
    struct FragmentOutput {
        vec4 color;
    }

    fragment {
        uniform u16vec2 packed;
        uniform i16vec2 signedPacked;

        FragmentOutput main() {
            FragmentOutput output;
            u16vec2 sum = packed + packed;
            float wrapped = float((packed * packed).x);
            float negated = float((-signedPacked).y);
            output.color = vec4(float(sum.x), wrapped, negated, 1.0);
            return output;
        }
    }
}
"""


SCALAR_WRAP_PROBE = """
shader NarrowScalarWrap {
    StructuredBuffer<int> iin;
    RWStructuredBuffer<float> outp;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            int8_t b = int8_t(iin[0]);
            uint8_t ub = uint8_t(iin[1]);
            int16_t s = int16_t(iin[2]);
            uint16_t us = uint16_t(iin[3]);
            int8_t k = int8_t(iin[4]);
            outp[0] = float(b * b);
            outp[1] = float(ub * ub);
            outp[2] = float(s * s);
            outp[3] = float(us * us);
            outp[4] = float((b * b) + b);
            int8_t stored = b * b;
            outp[5] = float(stored);
            outp[6] = float(b + 100);
            outp[7] = float(-ub);
            outp[8] = float(~ub);
            outp[9] = float(b << k);
        }
    }
}
"""

# iin = [13, 20, 200, 300, 4]: 13*13 = 169 -> -87, 20*20 = 400 -> 144, 200*200 = 40000 -> -25536,
# 300*300 = 90000 -> 24464, (-87) + 13 = -74, stored = -87, int8_t(13) + 100 promotes to int (113),
# -20 -> 236 and ~20 -> 235 as uint8_t, and 13 << 4 = 208 -> -48.
SCALAR_WRAP_EXPECTED = (
    -87.0,
    144.0,
    -25536.0,
    24464.0,
    -74.0,
    -87.0,
    113.0,
    236.0,
    235.0,
    -48.0,
)

SCALAR_FRAGMENT = """
shader NarrowScalarFragment {
    struct FragmentOutput {
        vec4 color;
    }

    fragment {
        uniform int8_t signedByte;
        uniform uint8_t unsignedByte;

        FragmentOutput main() {
            FragmentOutput output;
            int8_t stored = signedByte * signedByte;
            float s = float(signedByte * signedByte);
            float u = float(unsignedByte * unsignedByte);
            output.color = vec4(float(stored), s, u, 1.0);
            return output;
        }
    }
}
"""


def _compact(code):
    return "".join(code.split())


def _generate(source, generator=GLSLCodeGen):
    return generator().generate(crosstl.translator.parse(source))


def test_vector_operations_keep_their_narrow_type():
    rule = narrow_vector_binary_result
    assert rule("+", U16X2, U16X2, left_literal=False, right_literal=False) == U16X2
    assert rule("&", I16X2, I16X2, left_literal=False, right_literal=False) == I16X2
    assert rule("*", U16X2, U16, left_literal=False, right_literal=False) == U16X2
    assert rule("+", U16, U16, left_literal=False, right_literal=False) == U16
    assert rule("*", I8, I8, left_literal=False, right_literal=False) == I8
    assert rule("*", U8, U8, left_literal=False, right_literal=False) == U8
    assert rule("+", U16X2, None, left_literal=False, right_literal=True) == U16X2
    assert rule("<<", U16X2, None, left_literal=False, right_literal=False) == U16X2
    assert rule("<<", I8, I8, left_literal=False, right_literal=False) == I8
    assert narrow_vector_unary_result("~", U8X2) == U8X2
    assert narrow_vector_unary_result("-", U8) == U8


def test_promoting_operations_have_no_narrow_vector_type():
    rule = narrow_vector_binary_result
    # A 32-bit operand that is not a literal promotes the narrow value, and a 32-bit literal promotes a narrow scalar.
    assert rule("+", U16X2, None, left_literal=False, right_literal=False) is None
    assert rule("+", U16, None, left_literal=False, right_literal=True) is None
    assert rule("<<", U16, None, left_literal=False, right_literal=True) is None
    # Signedness and width mismatches are not narrow either.
    assert rule("+", U16X2, I16X2, left_literal=False, right_literal=False) is None
    assert rule("+", U16X2, U8X2, left_literal=False, right_literal=False) is None
    assert (
        rule(
            "+", U16, NarrowShape(True, 16, 1), left_literal=False, right_literal=False
        )
        is None
    )
    assert rule("+", U16, U8, left_literal=False, right_literal=False) is None
    # A shift takes the type of its left operand; comparisons are never narrow.
    assert rule(">>", None, U16X2, left_literal=False, right_literal=False) is None
    assert rule("<", U16X2, U16X2, left_literal=False, right_literal=False) is None
    assert rule("<", I8, I8, left_literal=False, right_literal=False) is None
    assert rule("<<", I8, U8, left_literal=False, right_literal=False) is None
    assert narrow_vector_unary_result("!", U16X2) is None


def test_promoting_sources_promote_narrow_scalars_only():
    def rule(operator, left, right):
        return narrow_vector_binary_result(
            operator,
            left,
            right,
            left_literal=False,
            right_literal=False,
            scalars_promote=True,
        )

    assert rule("*", I8, I8) is None
    assert rule("<<", U8, U8) is None
    assert rule("*", U16X2, U16) == U16X2
    assert rule("<<", U16X2, U16X2) == U16X2
    assert narrow_vector_unary_result("-", U16, scalars_promote=True) is None
    assert narrow_vector_unary_result("~", U8X2, scalars_promote=True) == U8X2


def test_narrow_vector_arithmetic_wraps_in_glsl():
    code = _compact(_generate(WRAP_PROBE))

    assert "ivec2(bitfieldExtract(uvec2((ivec2(n)+ivec2(m))),0,16)).x" in code
    assert "ivec2(bitfieldExtract(uvec2((ivec2(n)*ivec2(m))),0,16)).y" in code
    assert "ivec2(bitfieldExtract(uvec2((~ivec2(n))),0,16)).y" in code
    assert "bitfieldExtract((s+s),0,16).x" in code
    assert "bitfieldExtract((-s),0,16).x" in code
    assert "ivec2(bitfieldExtract(uvec2((ivec2(n)<<4)),0,16)).x" in code
    # Nested narrow results wrap at every step.
    assert (
        "ivec2(bitfieldExtract(uvec2((ivec2(bitfieldExtract(uvec2((ivec2(n)+ivec2(m))),0,16))"
        "*ivec2(m))),0,16)).y" in code
    )
    # A 32-bit operand promotes, so the result does not wrap.
    assert "output_[2]=(n+w).x;" in code


def test_narrow_vector_stores_wrap_once():
    code = _compact(_generate(WRAP_PROBE))

    assert "c=bitfieldExtract(uvec2((ivec2(c)+ivec2(m))),0,16);" in code
    assert "(e=bitfieldExtract(uvec2((ivec2(e)+1)),0,16));" in code
    assert code.count("bitfieldExtract(uvec2(ivec2(bitfieldExtract(") == 0


def test_narrow_vector_arithmetic_wraps_with_masks_in_webgl():
    code = _compact(_generate(FRAGMENT, WebGLCodeGen))

    assert "uvec2sum=(uvec2((ivec2(packed_)+ivec2(packed_)))&uvec2(65535u));" in code
    assert "ivec2((uvec2((ivec2(packed_)*ivec2(packed_)))&uvec2(65535u))).x" in code
    assert "(((-signedPacked)<<16)>>16).y" in code
    assert "bitfieldExtract" not in code


def test_narrow_scalar_arithmetic_wraps_in_glsl():
    code = _compact(_generate(SCALAR_WRAP_PROBE))

    assert "outp[0]=float(bitfieldExtract((b*b),0,8));" in code
    assert "outp[1]=float(int(bitfieldExtract(uint((int(ub)*int(ub))),0,8)));" in code
    assert "outp[2]=float(bitfieldExtract((s*s),0,16));" in code
    assert "outp[3]=float(int(bitfieldExtract(uint((int(us)*int(us))),0,16)));" in code
    assert "outp[4]=float(bitfieldExtract((bitfieldExtract((b*b),0,8)+b),0,8));" in code
    assert "intstored=bitfieldExtract((b*b),0,8);" in code
    assert "bitfieldExtract(int(bitfieldExtract(" not in code
    assert "outp[6]=float((b+100));" in code
    assert "outp[7]=float(int(bitfieldExtract(uint((-int(ub))),0,8)));" in code
    assert "outp[8]=float(int(bitfieldExtract(uint((~int(ub))),0,8)));" in code
    assert "outp[9]=float(bitfieldExtract((b<<k),0,8));" in code


def test_narrow_scalar_arithmetic_wraps_with_masks_in_webgl():
    code = _compact(_generate(SCALAR_FRAGMENT, WebGLCodeGen))

    assert "intstored=(((signedByte*signedByte)<<24)>>24);" in code
    assert "floats=float((((signedByte*signedByte)<<24)>>24));" in code
    assert (
        "floatu=float(int((uint((int(unsignedByte)*int(unsignedByte)))&255u)));" in code
    )
    assert "bitfieldExtract" not in code


def test_glsl_input_roundtrip_wraps_narrow_scalar_product(tmp_path):
    shader = tmp_path / "probe.comp"
    shader.write_text(
        "#version 460\n"
        "#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require\n"
        "layout(local_size_x = 1) in;\n"
        "layout(std430, binding = 0) buffer B0 { int iin[]; };\n"
        "layout(std430, binding = 1) buffer B1 { float outp[]; };\n"
        "void main() {\n"
        "    uint i = gl_GlobalInvocationID.x;\n"
        "    int8_t v = int8_t(iin[i]);\n"
        "    outp[i] = float(v * v);\n"
        "}\n",
        encoding="utf-8",
    )

    cgl = crosstl.translate(str(shader), backend="cgl", format_output=False)
    cgl_path = tmp_path / "probe.cgl"
    cgl_path.write_text(cgl, encoding="utf-8")
    translated = crosstl.translate(str(cgl_path), backend="opengl", format_output=False)

    assert "float(bitfieldExtract((v * v), 0, 8))" in translated


@pytest.mark.parametrize(
    ("suffix", "source"),
    [
        (
            ".metal",
            "#include <metal_stdlib>\n"
            "using namespace metal;\n"
            "kernel void k(device const uchar* a [[buffer(0)]], device float* o [[buffer(1)]],\n"
            "              uint i [[thread_position_in_grid]]) {\n"
            "    uchar x = a[i];\n"
            "    o[i] = float(x * x) + float(-x) + float(x << x);\n"
            "}\n",
        ),
        (
            ".cl",
            "__kernel void k(__global const uchar* a, __global float* o) {\n"
            "    int i = get_global_id(0);\n"
            "    uchar x = a[i];\n"
            "    o[i] = (float)(x * x) + (float)(-x) + (float)(x << x);\n"
            "}\n",
        ),
    ],
    ids=["metal", "opencl"],
)
def test_promoting_sources_keep_narrow_scalar_results_in_32_bits(
    tmp_path, suffix, source
):
    # C++ [conv.prom] and C99 6.3.1.1 promote uchar operands, so x * x for x = 200 is 40000, not 64.
    shader = tmp_path / f"k{suffix}"
    shader.write_text(source, encoding="utf-8")

    translated = crosstl.translate(str(shader), backend="opengl", format_output=False)

    assert "(int(x) * int(x))" in translated
    assert "bitfieldExtract(uint((int(x)" not in translated


def _glslang():
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    return validator


@pytest.mark.parametrize(
    ("source", "generator", "stage"),
    [
        (WRAP_PROBE, GLSLCodeGen, "comp"),
        (SCALAR_WRAP_PROBE, GLSLCodeGen, "comp"),
        (FRAGMENT, GLSLCodeGen, "frag"),
        (FRAGMENT, WebGLCodeGen, "frag"),
        (SCALAR_FRAGMENT, GLSLCodeGen, "frag"),
        (SCALAR_FRAGMENT, WebGLCodeGen, "frag"),
    ],
    ids=[
        "glsl-compute",
        "glsl-scalar-compute",
        "glsl-fragment",
        "webgl-fragment",
        "glsl-scalar-fragment",
        "webgl-scalar-fragment",
    ],
)
def test_narrow_vector_wrapping_compiles_with_glslang(
    tmp_path, source, generator, stage
):
    validator = _glslang()
    shader = tmp_path / f"narrow.{stage}"
    shader.write_text(_generate(source, generator), encoding="utf-8")

    completed = subprocess.run(
        [validator, "-S", stage, str(shader)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def _metal_source_from_glsl(tmp_path, glsl):
    validator = _glslang()
    spirv_cross = shutil.which("spirv-cross")
    if spirv_cross is None:
        pytest.skip("spirv-cross is not installed")
    shader = tmp_path / "narrow.comp"
    shader.write_text(glsl, encoding="utf-8")
    spirv = tmp_path / "narrow.spv"
    compiled = subprocess.run(
        [validator, "-V", "-S", "comp", str(shader), "-o", str(spirv)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    converted = subprocess.run(
        [spirv_cross, "--msl", str(spirv)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert converted.returncode == 0, converted.stderr
    return converted.stdout


def _run_metal_kernel(metal_source, inputs, result_count):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    function = library.newFunctionWithName_("main0")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error

    shared = Metal.MTLResourceStorageModeShared
    buffers = [
        device.newBufferWithBytes_length_options_(data, len(data), shared)
        for data in inputs
    ]
    result_size = 4 * result_count
    buffers.append(device.newBufferWithLength_options_(result_size, shared))
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
    data = bytes(buffers[-1].contents().as_buffer(result_size))
    return struct.unpack(f"<{result_count}I", data)


def test_narrow_vector_arithmetic_wraps_on_the_gpu(tmp_path):
    glsl = _generate(WRAP_PROBE)
    assert "readonly buffer vecsBuffer { uvec2 vecs[]; }" in glsl
    metal = _metal_source_from_glsl(tmp_path, glsl)
    inputs = [struct.pack("<4I", 65535, 300, 1, 300), struct.pack("<2i", 20000, 0)]

    assert _run_metal_kernel(metal, inputs, len(WRAP_EXPECTED)) == WRAP_EXPECTED


def test_narrow_scalar_arithmetic_wraps_on_the_gpu(tmp_path):
    metal = _metal_source_from_glsl(tmp_path, _generate(SCALAR_WRAP_PROBE))
    inputs = [struct.pack("<5i", 13, 20, 200, 300, 4)]
    count = len(SCALAR_WRAP_EXPECTED)

    words = _run_metal_kernel(metal, inputs, count)

    assert struct.unpack(f"<{count}f", struct.pack(f"<{count}I", *words)) == (
        SCALAR_WRAP_EXPECTED
    )


def _opengl_context():
    moderngl = pytest.importorskip("moderngl")
    for kwargs in ({"require": 430, "backend": "egl"}, {"require": 430}):
        try:
            return moderngl.create_standalone_context(**kwargs)
        except Exception:
            continue
    pytest.skip("no standalone OpenGL 4.3+ context")


def test_narrow_scalar_arithmetic_wraps_on_opengl():
    glsl = _generate(SCALAR_WRAP_PROBE)
    context = _opengl_context()
    try:
        shader = context.compute_shader(glsl)
        in_buffer = context.buffer(struct.pack("<5i", 13, 20, 200, 300, 4))
        out_buffer = context.buffer(reserve=4 * len(SCALAR_WRAP_EXPECTED))
        in_buffer.bind_to_storage_buffer(0)
        out_buffer.bind_to_storage_buffer(1)
        shader.run(1, 1, 1)
        actual = struct.unpack(f"<{len(SCALAR_WRAP_EXPECTED)}f", out_buffer.read())
    finally:
        context.release()

    assert actual == SCALAR_WRAP_EXPECTED
