"""Integer bit functions: HLSL spellings on every target, and 8- and 16-bit operands at their own width.

``semantics/bit_functions.py`` spells ``countbits``, ``reversebits``, ``firstbithigh`` and ``firstbitlow`` with the
GLSL functions and HLSL's ``uint`` result for targets other than HLSL and Slang, and computes ``bitCount``,
``bitfieldReverse`` and signed ``abs`` of narrow integers on 32-bit operands, so 32-bit carriers give the narrow
result.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.semantics.bit_functions import BitFunctionArgumentUntypedError

NARROW_PROGRAM = """
shader NarrowBits {
    StructuredBuffer<int16_t> shorts;
    StructuredBuffer<uint16_t> ushorts;
    RWStructuredBuffer<int> output;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            int16_t s = shorts[0];
            int16_t one = shorts[1];
            int16_t lo = shorts[2];
            uint16_t u = ushorts[0];
            i16vec2 v = i16vec2(shorts[0], shorts[1]);
            i8vec2 c = i8vec2(int8_t(-2), int8_t(64));
            u8vec2 d = u8vec2(uint8_t(1), uint8_t(255));
            output[0] = int(bitCount(s));
            output[1] = int(bitfieldReverse(one));
            output[2] = int(abs(lo));
            output[3] = int(bitfieldReverse(u));
            output[4] = int(bitCount(v).x);
            output[5] = int(bitfieldReverse(v).y);
            output[6] = int(countbits(s));
            output[7] = int(reversebits(one));
            output[8] = abs(lo) < int16_t(0) ? 1 : 0;
            output[9] = int(bitCount(c).x);
            output[10] = int(bitfieldReverse(c).y);
            output[11] = int(bitfieldReverse(c).x);
            output[12] = int(bitfieldReverse(d).x);
            output[13] = int(bitCount(d).y);
            output[14] = int(bitCount(one + one));
            output[15] = int(bitCount(s) + bitCount(u));
        }
    }
}
"""

# shorts = (-1, 1, -32768, 0), ushorts = (0x00F1, 0). Counts and reversals take the value's own width:
# reverse16(1) = 0x8000, reverse16(0x00F1) = 0x8F00, reverse8(0x40) = 0x02, reverse8(0xFE) = 0x7F,
# reverse8(1) = 0x80. abs(int16_t(-32768)) is -32768. one + one promotes to int, a 32-bit count of 2.
NARROW_EXPECTED = (
    16,
    -32768,
    -32768,
    0x8F00,
    16,
    -32768,
    16,
    -32768,
    1,
    7,
    2,
    127,
    128,
    8,
    1,
    16 + 5,
)
NARROW_INPUTS = (struct.pack("<4h", -1, 1, -32768, 0), struct.pack("<2H", 0x00F1, 0))

HLSL_PROGRAM = """
shader HlslBits {
    StructuredBuffer<int> inp;
    RWStructuredBuffer<uint> output;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            int v = inp[0];
            int zero = inp[1];
            int negative = inp[2];
            ivec2 w = ivec2(v, zero);
            output[0] = countbits(v);
            output[1] = reversebits(uint(v));
            output[2] = firstbithigh(v);
            output[3] = firstbitlow(v);
            output[4] = firstbithigh(zero) > 5 ? 1u : 0u;
            output[5] = firstbitlow(zero);
            output[6] = firstbithigh(negative);
            uvec2 c = countbits(w) + firstbithigh(w);
            output[7] = c.x;
            output[8] = c.y;
            output[9] = firstbitlow(zero) > 5 ? 1u : 0u;
        }
    }
}
"""

# inp = (0xF0, 0, -8). HLSL returns uint: firstbithigh(0) and firstbitlow(0) are 0xFFFFFFFF, which compares
# greater than 5. firstbithigh(-8) is the highest bit that differs from the sign bit.
HLSL_EXPECTED = (4, 0x0F000000, 7, 4, 1, 0xFFFFFFFF, 2, 11, 0xFFFFFFFF, 1)
HLSL_INPUTS = (struct.pack("<3i", 0xF0, 0, -8),)

HLSL_NAMES = ("countbits(", "reversebits(", "firstbithigh(", "firstbitlow(")


def _translate(source, backend, tmp_path):
    path = tmp_path / "program.cgl"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _run_metal(metal_source, inputs, count, result_format):
    Metal = pytest.importorskip("Metal")
    device = Metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(
        metal_source, Metal.MTLCompileOptions.new(), None
    )
    assert library is not None, error
    function = library.newFunctionWithName_("kernel_main")
    pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
    assert pipeline is not None, error
    shared = Metal.MTLResourceStorageModeShared
    buffers = [
        device.newBufferWithBytes_length_options_(data, len(data), shared)
        for data in inputs
    ]
    buffers.append(device.newBufferWithLength_options_(4 * count, shared))
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
    data = bytes(buffers[-1].contents().as_buffer(4 * count))
    return struct.unpack(f"<{count}{result_format}", data)


def test_narrow_bit_functions_take_the_value_width_on_the_metal_gpu(tmp_path):
    metal = _translate(NARROW_PROGRAM, "metal", tmp_path)

    assert (
        _run_metal(metal, NARROW_INPUTS, len(NARROW_EXPECTED), "i") == NARROW_EXPECTED
    )


def test_hlsl_bit_functions_keep_unsigned_results_on_the_metal_gpu(tmp_path):
    metal = _translate(HLSL_PROGRAM, "metal", tmp_path)

    assert _run_metal(metal, HLSL_INPUTS, len(HLSL_EXPECTED), "I") == HLSL_EXPECTED


@pytest.mark.parametrize("backend", ["opengl", "metal", "wgsl", "vulkan", "mojo"])
def test_hlsl_bit_functions_are_spelled_for_targets_without_them(backend, tmp_path):
    output = _translate(HLSL_PROGRAM, backend, tmp_path)

    for name in HLSL_NAMES:
        assert name not in output


@pytest.mark.parametrize("backend", ["directx", "slang"])
def test_hlsl_and_slang_keep_their_bit_functions(backend, tmp_path):
    output = _translate(HLSL_PROGRAM, backend, tmp_path)

    for name in HLSL_NAMES:
        assert name in output


def test_glsl_spells_hlsl_bit_functions_with_unsigned_results(tmp_path):
    glsl = _translate(HLSL_PROGRAM, "opengl", tmp_path)

    assert "output_[0] = uint(bitCount(v));" in glsl
    assert "output_[1] = bitfieldReverse(uint(v));" in glsl
    assert "((uint(findMSB(zero)) > 5) ? 1u : 0u)" in glsl
    assert "uvec2(bitCount(w)) + uvec2(findMSB(w))" in glsl


def test_narrow_counts_and_reversals_take_32_bit_operands(tmp_path):
    glsl = _translate(NARROW_PROGRAM, "opengl", tmp_path)

    assert "bitCount((uint(s) & 65535u))" in glsl
    assert "(bitfieldReverse(uint(one)) >> 16u)" in glsl
    assert "bitCount((uvec2(c) & uvec2(255u)))" in glsl
    assert "(bitfieldReverse(uvec2(v)) >> uvec2(16u))" in glsl


def test_32_bit_bit_functions_are_unchanged(tmp_path):
    source = HLSL_PROGRAM.replace(
        "output[0] = countbits(v);", "output[0] = bitCount(v);"
    )
    glsl = _translate(source, "opengl", tmp_path)

    # The store into a uint element converts the int count; the operand keeps its 32-bit type.
    assert "output_[0] = uint(bitCount(v));" in glsl


def test_hlsl_output_keeps_native_16_bit_counts_at_their_width(tmp_path):
    hlsl = _translate(NARROW_PROGRAM, "directx", tmp_path)

    assert "countbits((uint(s) & 65535u))" in hlsl
    assert "(reversebits(uint(one)) >> 16u)" in hlsl


def test_program_functions_named_like_bit_functions_are_not_rewritten(tmp_path):
    source = """
shader UserBits {
    RWStructuredBuffer<uint> output;
    uint countbits(int value) { return uint(value) + 1u; }
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            output[0] = countbits(int(output[1]));
        }
    }
}
"""
    glsl = _translate(source, "opengl", tmp_path)

    assert "bitCount" not in glsl


def test_untyped_hlsl_bit_function_argument_is_refused(tmp_path):
    source = """
shader Untyped {
    RWStructuredBuffer<uint> counter;
    RWStructuredBuffer<uint> output;
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            output[0] = firstbithigh(atomicAdd(counter[0], 1u));
        }
    }
}
"""
    with pytest.raises(BitFunctionArgumentUntypedError) as refusal:
        _translate(source, "opengl", tmp_path)

    assert refusal.value.code == "project.translate.hlsl-bit-function-untyped"
    assert "firstbithigh" in str(refusal.value)


def _check(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "program", [NARROW_PROGRAM, HLSL_PROGRAM], ids=["narrow", "hlsl"]
)
def test_bit_function_output_compiles_natively(program, tmp_path):
    checked = 0
    if shutil.which("glslangValidator"):
        glsl = tmp_path / "program.comp"
        glsl.write_text(_translate(program, "opengl", tmp_path), encoding="utf-8")
        _check(["glslangValidator", "-S", "comp", str(glsl)])
        checked += 1
    if shutil.which("spirv-as") and shutil.which("spirv-val"):
        assembly = tmp_path / "program.spvasm"
        assembly.write_text(_translate(program, "vulkan", tmp_path), encoding="utf-8")
        binary = tmp_path / "program.spv"
        _check(
            ["spirv-as", str(assembly), "-o", str(binary), "--target-env", "vulkan1.1"]
        )
        _check(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
        checked += 1
    if shutil.which("naga"):
        wgsl = tmp_path / "program.wgsl"
        wgsl.write_text(_translate(program, "wgsl", tmp_path), encoding="utf-8")
        _check(["naga", str(wgsl)])
        checked += 1
    if shutil.which("xcrun"):
        metal = tmp_path / "program.metal"
        metal.write_text(_translate(program, "metal", tmp_path), encoding="utf-8")
        _check(
            [
                "xcrun",
                "-sdk",
                "macosx",
                "metal",
                "-c",
                str(metal),
                "-o",
                str(tmp_path / "program.air"),
            ]
        )
        checked += 1
    if checked == 0:
        pytest.skip("no native shader compiler is installed")
