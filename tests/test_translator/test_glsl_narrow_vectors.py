"""GLSL 8- and 16-bit integer vector arithmetic wraps at the narrow width.

The GLSL emitters keep a ``u16vec2`` value in a 32-bit ``uvec2`` carrier and compute on promoted ``ivec2``
operands. CrossGL promotes scalars only: vector arithmetic keeps its narrow type, so ``(n + m).x`` is reduced
modulo 2^16. A 32-bit operand promotes the narrow vector first, and only an integer literal adopts the narrow type.
"""

import shutil
import struct
import subprocess

import pytest

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


def _compact(code):
    return "".join(code.split())


def _generate(source, generator=GLSLCodeGen):
    return generator().generate(crosstl.translator.parse(source))


def test_vector_operations_keep_their_narrow_type():
    rule = narrow_vector_binary_result
    assert rule("+", U16X2, U16X2, left_literal=False, right_literal=False) == U16X2
    assert rule("&", I16X2, I16X2, left_literal=False, right_literal=False) == I16X2
    assert rule("*", U16X2, U16, left_literal=False, right_literal=False) == U16X2
    assert rule("+", U16X2, None, left_literal=False, right_literal=True) == U16X2
    assert rule("<<", U16X2, None, left_literal=False, right_literal=False) == U16X2
    assert narrow_vector_unary_result("~", U8X2) == U8X2


def test_promoting_operations_have_no_narrow_vector_type():
    rule = narrow_vector_binary_result
    # A 32-bit operand that is not a literal promotes the narrow vector.
    assert rule("+", U16X2, None, left_literal=False, right_literal=False) is None
    # Scalars promote to int; signedness and width mismatches are not narrow either.
    assert rule("+", U16, U16, left_literal=False, right_literal=False) is None
    assert rule("+", U16X2, I16X2, left_literal=False, right_literal=False) is None
    assert rule("+", U16X2, U8X2, left_literal=False, right_literal=False) is None
    # A shift takes the type of its left operand; comparisons are never narrow.
    assert rule(">>", None, U16X2, left_literal=False, right_literal=False) is None
    assert rule("<", U16X2, U16X2, left_literal=False, right_literal=False) is None
    assert narrow_vector_unary_result("~", U16) is None
    assert narrow_vector_unary_result("!", U16X2) is None


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


def _glslang():
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    return validator


@pytest.mark.parametrize(
    ("source", "generator", "stage"),
    [
        (WRAP_PROBE, GLSLCodeGen, "comp"),
        (FRAGMENT, GLSLCodeGen, "frag"),
        (FRAGMENT, WebGLCodeGen, "frag"),
    ],
    ids=["glsl-compute", "glsl-fragment", "webgl-fragment"],
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
