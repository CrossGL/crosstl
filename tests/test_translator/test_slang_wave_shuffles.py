"""Slang lowers CrossGL's relative wave shuffles through ``WaveReadLaneAt`` helpers.

Slang has no relative shuffles, so ``WaveShuffleDown``/``Up``/``Xor``/``AndFillUp`` were refused, and with them any
Metal kernel whose headers define shuffle overloads. A source lane outside the SIMD-group reads the calling lane, as
MSL documents. When a slangc and a Metal device are available, the Slang output is compiled for Metal and run
against the original Metal kernel.
"""

from __future__ import annotations

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.slang_codegen import SlangCodeGen
from crosstl.translator.codegen.slang_wave_shuffles import (
    render_slang_wave_shuffle_helper,
    slang_wave_shuffle_helper_name,
)

CROSSGL_SOURCE = """
shader Shuffles {
    bool shuffle_bool(bool data, uint lane) {
        return WaveReadLaneAt(uint(data), lane);
    }

    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(RWStructuredBuffer<float> values @ binding(0),
                  RWStructuredBuffer<uint> flags @ binding(1),
                  uvec3 id @ gl_GlobalInvocationID) {
            uint i = id.x;
            float v = values[i];
            vec4 w = vec4(v, v + 1.0, v + 2.0, v + 3.0);
            float down = WaveShuffleDown(v, 1u);
            float up = WaveShuffleUp(v, 2u);
            float x = WaveShuffleXor(v, 3u);
            vec4 f = WaveShuffleAndFillUp(w, w * 2.0, 1u);
            float g = WaveShuffleAndFillUp(v, -v, 1u, 8u);
            bool b = shuffle_bool(flags[i] != 0u, i % 4u);
            values[i] = down + up + x + f.x + g + (b ? 1.0 : 0.0);
        }
    }
}
"""

METAL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
inline bool shuffle_bool(bool data, ushort lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}
kernel void k(device float* out [[buffer(0)]], device uint* flags [[buffer(1)]],
              uint i [[thread_position_in_grid]],
              uint lane [[thread_index_in_simdgroup]],
              uint width [[threads_per_simdgroup]]) {
  float v = out[i];
  float4 w = float4(v, v + 1.0f, v + 2.0f, v + 3.0f);
  float down = simd_shuffle_down(v, 1);
  float up = simd_shuffle_up(v, 2);
  float x = simd_shuffle_xor(v, 3);
  float4 f = simd_shuffle_and_fill_up(w, w * 2.0f, 1);
  float g = simd_shuffle_and_fill_up(v, -v, 1, 8);
  bool b = shuffle_bool(flags[i] != 0, ushort(i % 4));
  out[i] = down + up + x + f.x + g + (b ? 1.0f : 0.0f) + float(lane) * 100.0f
      + float(width) * 10000.0f;
}
"""


def _generate(source=CROSSGL_SOURCE):
    return SlangCodeGen().generate(crosstl.translator.parse(source))


def test_shuffles_call_one_helper_per_operation_and_type():
    generated = _generate()

    assert "float down = __crossgl_wave_shuffle_down_float(v, uint(1u));" in generated
    assert "float up = __crossgl_wave_shuffle_up_float(v, uint(2u));" in generated
    assert "float x = __crossgl_wave_shuffle_xor_float(v, uint(3u));" in generated
    assert (
        "__crossgl_wave_shuffle_and_fill_up_float4(w, w * 2.0, uint(1u), WaveGetLaneCount())"
        in generated
    )
    assert (
        "__crossgl_wave_shuffle_and_fill_up_float(v, -v, uint(1u), uint(8u))"
        in generated
    )
    for operation, value_type in (
        ("WaveShuffleDown", "float"),
        ("WaveShuffleUp", "float"),
        ("WaveShuffleXor", "float"),
        ("WaveShuffleAndFillUp", "float4"),
        ("WaveShuffleAndFillUp", "float"),
    ):
        helper = render_slang_wave_shuffle_helper(operation, value_type)
        assert generated.count(helper) == 1
    assert "WaveShuffleDown(" not in generated


def test_helpers_read_the_calling_lane_outside_the_simd_group():
    down = render_slang_wave_shuffle_helper("WaveShuffleDown", "float")
    up = render_slang_wave_shuffle_helper("WaveShuffleUp", "int2")
    xor = render_slang_wave_shuffle_helper("WaveShuffleXor", "uint")
    fill = render_slang_wave_shuffle_helper("WaveShuffleAndFillUp", "half4")

    assert "uint source = (delta < laneCount - lane) ? lane + delta : lane;" in down
    assert "uint source = (delta <= lane) ? lane - delta : lane;" in up
    assert "uint source = ((lane ^ delta) < laneCount) ? lane ^ delta : lane;" in xor
    assert "fillSource = fillSource < laneCount ? fillSource : lane;" in fill
    assert fill.startswith("half4 __crossgl_wave_shuffle_and_fill_up_half4(half4 value")
    assert slang_wave_shuffle_helper_name("WaveShuffleUp", "int2") in up


def test_a_wave_read_converts_to_the_type_its_use_expects():
    generated = _generate()

    # Metal converts the uint result to the bool return type implicitly.
    assert "return bool(WaveReadLaneAt(uint(data), lane));" in generated


def test_a_fill_of_another_type_is_refused():
    source = CROSSGL_SOURCE.replace(
        "WaveShuffleAndFillUp(v, -v, 1u, 8u)", "WaveShuffleAndFillUp(v, w, 1u, 8u)"
    )

    with pytest.raises(Exception, match="fill must be float, got float4"):
        _generate(source)


BIT_CAST_SHUFFLE = """
shader WideShuffle {
    uint64 shuffle_down_wide(uint64 data, uint delta) {
        return as_type<uint64>(WaveShuffleDown(as_type<uvec2>(data), delta));
    }

    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(RWStructuredBuffer<uint> output @ binding(0)) {
            output[0] = uint(shuffle_down_wide(uint64(output[1]), 1u));
        }
    }
}
"""


def test_bit_reinterpretations_become_bit_casts_and_type_shuffles():
    generated = _generate(BIT_CAST_SHUFFLE)

    # as_type<T> was emitted verbatim, which slangc rejects, and its type was unknown.
    assert (
        "return bit_cast<uint64_t>(__crossgl_wave_shuffle_down_uint2("
        "bit_cast<uint2>(data), uint(delta)));" in generated
    )
    assert "as_type" not in generated


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_bit_cast_shuffle_compiles_to_spirv(tmp_path):
    path = tmp_path / "wide.slang"
    path.write_text(_generate(BIT_CAST_SHUFFLE), encoding="utf-8")

    result = subprocess.run(
        [
            shutil.which("slangc"),
            str(path),
            "-target",
            "spirv",
            "-o",
            str(tmp_path / "w.spv"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


LANE_PARAMETERS = """
shader LaneParameters {
    compute {
        layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;

        void main(RWStructuredBuffer<uint> output @ binding(0),
                  uint lane @ gl_SubgroupInvocationID,
                  uint width @ gl_SubgroupSize) {
            output[lane] = width;
        }
    }
}
"""


def test_lane_index_and_width_parameters_read_wave_intrinsics():
    generated = _generate(LANE_PARAMETERS)

    assert "uint lane = WaveGetLaneIndex();" in generated
    assert "uint width = WaveGetLaneCount();" in generated
    assert "gl_Subgroup" not in generated


def test_a_non_integer_lane_parameter_is_refused():
    source = LANE_PARAMETERS.replace("uint lane @", "float lane @").replace(
        "output[lane]", "output[uint(lane)]"
    )

    with pytest.raises(Exception, match="must be an integer scalar, not float"):
        _generate(source)


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_slang_output_compiles_to_spirv_shuffles(tmp_path):
    path = tmp_path / "shuffles.slang"
    path.write_text(_generate(), encoding="utf-8")
    spirv = tmp_path / "shuffles.spv"

    result = subprocess.run(
        [shutil.which("slangc"), str(path), "-target", "spirv", "-o", str(spirv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    disassembler = shutil.which("spirv-dis")
    if disassembler is not None:
        listing = subprocess.run(
            [disassembler, str(spirv)], check=True, capture_output=True, text=True
        ).stdout
        assert "OpGroupNonUniformShuffle " in listing


def _metal_device():
    try:
        import Metal
    except ImportError:
        return None, None
    return Metal, Metal.MTLCreateSystemDefaultDevice()


def _run_metal(metal, device, source, values, flags):
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_("k"), None
    )
    assert pipeline is not None, error
    count = len(values)
    out = device.newBufferWithBytes_length_options_(
        struct.pack(f"{count}f", *values), 4 * count, 0
    )
    flag_buffer = device.newBufferWithBytes_length_options_(
        struct.pack(f"{count}I", *flags), 4 * count, 0
    )
    command = device.newCommandQueue().commandBuffer()
    encoder = command.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(out, 0, 0)
    encoder.setBuffer_offset_atIndex_(flag_buffer, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(32, 1, 1)
    )
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    return struct.unpack(f"{count}f", bytes(out.contents().as_buffer(4 * count)))


@pytest.mark.skipif(shutil.which("slangc") is None, reason="slangc is not available")
def test_metal_shuffles_through_slang_match_metal_on_the_gpu(tmp_path):
    metal, device = _metal_device()
    if device is None:
        pytest.skip("no Metal device")
    source = tmp_path / "shuffles.metal"
    source.write_text(METAL_SOURCE, encoding="utf-8")
    slang = crosstl.translate(
        str(source),
        backend="slang",
        source_backend="metal",
        source_options={"threadgroup_size": (32, 1, 1)},
    )
    slang_path = tmp_path / "shuffles.slang"
    slang_path.write_text(slang, encoding="utf-8")
    metal_path = tmp_path / "shuffles_from_slang.metal"
    result = subprocess.run(
        [
            shutil.which("slangc"),
            str(slang_path),
            "-target",
            "metal",
            "-o",
            str(metal_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    values = [float((i * 7) % 23) - 5.0 for i in range(256)]
    flags = [1 if (i * 5) % 3 == 0 else 0 for i in range(256)]
    expected = _run_metal(metal, device, METAL_SOURCE, values, flags)
    actual = _run_metal(
        metal, device, metal_path.read_text(encoding="utf-8"), values, flags
    )

    assert actual == expected
