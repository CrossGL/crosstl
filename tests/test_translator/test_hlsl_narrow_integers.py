"""HLSL native 16-bit integers follow the CrossGL (MSL) promotion rules.

CrossGL promotes an 8- or 16-bit integer scalar to ``int`` before arithmetic and never promotes a vector (MSL
section 2.12, C++ [conv.prom] and [expr.shift]). HLSL with ``-enable-16bit-types`` promotes nothing, so ``-a`` of a
``uint16_t`` must be emitted as ``-int(a)``. A vector shift keeps the narrow type of its left operand in CrossGL,
MSL, HLSL and GLSL alike, so ``n << 4`` of a ``u16vec2`` must not be computed on ``int2`` operands, which never
wrapped at 16 bits.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen

PROBE = """
shader NarrowPromotion {
    StructuredBuffer<u16vec2> vecs;
    StructuredBuffer<i16vec2> signedVecs;
    StructuredBuffer<uint> words;
    RWStructuredBuffer<uint> output;

    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        void main() {
            u16vec2 n = vecs[0];
            u16vec2 m = vecs[1];
            i16vec2 s = signedVecs[0];
            i16vec2 t = signedVecs[1];
            uint k = words[0];
            uint16_t a = m.x;
            int16_t b = t.x;
            output[0] = (n << 4).x;
            output[1] = (n << k).x;
            output[2] = ((n << 4) >> 4).x;
            output[3] = uint(-a);
            output[4] = uint(~a);
            output[5] = uint(int((s << 2).x));
            output[6] = uint(-b);
            output[7] = (n + m).x;
            output[8] = uint(a << 4);
            output[9] = (-n).y;
        }
    }
}
"""

# vecs = [(65535, 300), (1, 300)], signedVecs = [(20000, 0), (-32768, 0)] and words = [4].
EXPECTED = (
    (65535 << 4) % 65536,
    (65535 << 4) % 65536,
    ((65535 << 4) % 65536) >> 4,
    2**32 - 1,
    2**32 - 2,
    (20000 << 2) - 65536,
    32768,
    0,
    1 << 4,
    65536 - 300,
)


def _generate(source=PROBE):
    return HLSLCodeGen(target_profile="dx12").generate(crosstl.translator.parse(source))


def test_narrow_vector_shifts_keep_their_narrow_type():
    code = _generate()

    assert "output[0] = (n << 4).x;" in code
    assert "output[1] = (n << k).x;" in code
    assert "output[2] = ((n << 4) >> 4).x;" in code
    assert "output[5] = uint(int((s << 2).x));" in code
    assert "int2(n)" not in code
    assert "int2(s)" not in code


def test_narrow_scalar_negation_and_complement_promote_to_int():
    code = _generate()

    assert "output[3] = uint(-int(a));" in code
    assert "output[4] = uint(~int(a));" in code
    assert "output[6] = uint(-int(b));" in code
    # Scalar shifts already promote; vector negation stays narrow and wraps natively.
    assert "output[8] = uint((int(a) << 4));" in code
    assert "output[9] = -n.y;" in code


def test_narrow_scalar_unary_result_feeds_int_arithmetic():
    shader = """
    shader NarrowUnaryType {
        int negateWide(uint16_t value) {
            return -value * 70000;
        }
    }
    """

    code = _generate(shader)

    assert "(-int(value) * 70000)" in code


def test_increments_of_narrow_scalars_are_not_promoted():
    shader = """
    shader NarrowIncrement {
        uint16_t bump(uint16_t value) {
            value++;
            return value;
        }
    }
    """

    code = _generate(shader)

    assert "value++;" in code
    assert "int(value)++" not in code


def _dxc():
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    return dxc


def test_narrow_promotion_compiles_with_dxc(tmp_path):
    dxc = _dxc()
    shader = tmp_path / "narrow.hlsl"
    shader.write_text(_generate(), encoding="utf-8")

    completed = subprocess.run(
        [dxc, "-T", "cs_6_2", "-E", "CSMain", "-enable-16bit-types", str(shader)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def _metal_source_from_hlsl(tmp_path, hlsl):
    dxc = _dxc()
    spirv_cross = shutil.which("spirv-cross")
    if spirv_cross is None:
        pytest.skip("spirv-cross is not installed")
    shader = tmp_path / "narrow.hlsl"
    shader.write_text(hlsl, encoding="utf-8")
    spirv = tmp_path / "narrow.spv"
    compiled = subprocess.run(
        [
            dxc,
            "-T",
            "cs_6_2",
            "-E",
            "CSMain",
            "-enable-16bit-types",
            "-spirv",
            "-fspv-target-env=vulkan1.1",
            # Keep u0 apart from t0..t2 so every buffer has its own binding.
            "-fvk-u-shift",
            "3",
            "0",
            str(shader),
            "-Fo",
            str(spirv),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if compiled.returncode != 0 and "SPIR-V CodeGen not available" in (
        compiled.stdout + compiled.stderr
    ):
        pytest.skip("dxc was built without SPIR-V code generation")
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    converted = subprocess.run(
        [spirv_cross, "--msl", "--msl-version", "20100", str(spirv)],
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
    function = library.newFunctionWithName_("CSMain")
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


def test_narrow_promotion_matches_crossgl_on_the_gpu(tmp_path):
    metal = _metal_source_from_hlsl(tmp_path, _generate())
    inputs = [
        struct.pack("<4H", 65535, 300, 1, 300),
        struct.pack("<4h", 20000, 0, -32768, 0),
        struct.pack("<I", 4),
    ]

    assert _run_metal_kernel(metal, inputs, len(EXPECTED)) == EXPECTED
