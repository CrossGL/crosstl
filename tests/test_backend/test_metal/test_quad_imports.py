"""Metal quad-group functions import as CrossGL quad reads and clustered reductions (MSL 4.1, section 6.10.3).

A Metal quad-group is four consecutive SIMD lanes, the quad of a CrossGL compute stage without a derivative group,
so every target that writes CrossGL's quad operations can translate them: the lane reads become ``QuadReadLaneAt``
or ``QuadReadAcross*``, the reductions ``WaveClustered*(value, 4u)``, and the relative permutes read the lane they
compute from ``WaveGetLaneIndex() & 3u``. On an Apple GPU the round trip computes what the native kernel computes.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl

HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# name -> (Metal call on the uint ``v``, the CrossGL that call imports as).
CASES = {
    "broadcast": ("quad_broadcast(v, 2)", "QuadReadLaneAt(v, 2u)"),
    "shuffle": (
        "quad_shuffle(v, in_[tid] & 3u)",
        "QuadReadLaneAt(v, (((0x3210u | ((WaveGetLaneIndex() & 3u) << 16u)) >> "
        "(min((uint(buffer_load(in_, tid) & 3u) & 65535u), 4u) << 2u)) & 3u))",
    ),
    # A lane or mask of 4 or more reads the caller's own value on Apple GPUs.
    "shuffle_high": ("quad_shuffle(v, in_[tid] & 15u)", None),
    "shuffle_constant_high": (
        "quad_shuffle(v, 9u)",
        "QuadReadLaneAt(v, (WaveGetLaneIndex() & 3u))",
    ),
    "broadcast_high": ("quad_broadcast(v, in_[tid & ~3u] & 7u)", None),
    "broadcast_constant_high": (
        "quad_broadcast(v, 5)",
        "QuadReadLaneAt(v, (WaveGetLaneIndex() & 3u))",
    ),
    "xor_high": ("quad_shuffle_xor(v, in_[tid & ~3u] & 7u)", None),
    "xor_constant_high": (
        "quad_shuffle_xor(v, ushort(6))",
        "QuadReadLaneAt(v, (WaveGetLaneIndex() & 3u))",
    ),
    "up_high": ("quad_shuffle_up(v, in_[tid & ~3u] & 7u)", None),
    "down_high": ("quad_shuffle_down(v, in_[tid & ~3u] & 7u)", None),
    "rotate_up_high": ("quad_shuffle_rotate_up(v, in_[tid & ~3u] & 7u)", None),
    "xor1": ("quad_shuffle_xor(v, 1)", "QuadReadAcrossX(v)"),
    "xor2": ("quad_shuffle_xor(v, 2u)", "QuadReadAcrossY(v)"),
    "xor3": ("quad_shuffle_xor(v, 3)", "QuadReadAcrossDiagonal(v)"),
    "xor_run_time": (
        "quad_shuffle_xor(v, in_[0] & 3u)",
        "QuadReadLaneAt(v, ((WaveGetLaneIndex() & 3u) ^ ((0x3210u >> "
        "(min((uint(buffer_load(in_, 0) & 3u) & 65535u), 4u) << 2u)) & 3u)))",
    ),
    "up": (
        "quad_shuffle_up(v, 1)",
        "QuadReadLaneAt(v, min((WaveGetLaneIndex() & 3u) - 1u, (WaveGetLaneIndex() & 3u)))",
    ),
    "up_run_time": ("quad_shuffle_up(v, (in_[0] & 3u) | 2u)", None),
    "down": (
        "quad_shuffle_down(v, 1)",
        "QuadReadLaneAt(v, (3u - min((3u - (WaveGetLaneIndex() & 3u)) - 1u, "
        "3u - (WaveGetLaneIndex() & 3u))))",
    ),
    "down_run_time": ("quad_shuffle_down(v, (in_[0] & 3u) | 2u)", None),
    "rotate_up": (
        "quad_shuffle_rotate_up(v, 1)",
        "QuadReadLaneAt(v, (((WaveGetLaneIndex() & 3u) - 1u) & 3u))",
    ),
    "rotate_down": (
        "quad_shuffle_rotate_down(v, 3u)",
        "QuadReadLaneAt(v, (((WaveGetLaneIndex() & 3u) + 3u) & 3u))",
    ),
    "min": ("quad_min(v)", "WaveClusteredMin(v, 4u)"),
    "max": ("quad_max(v)", "WaveClusteredMax(v, 4u)"),
    "and": ("quad_and(v)", "WaveClusteredBitAnd(v, 4u)"),
    "or": ("quad_or(v)", "WaveClusteredBitOr(v, 4u)"),
    "xor": ("quad_xor(v)", "WaveClusteredBitXor(v, 4u)"),
    "product": ("quad_product(v)", "WaveClusteredProduct(v, 4u)"),
    "all": ("uint(quad_all((v & 1u) != 0u))", "QuadAll((v & 1u) != 0u)"),
    "any": ("uint(quad_any((v & 5u) == 5u))", "QuadAny((v & 5u) == 5u)"),
}


def _kernel(call):
    return HEADER + (
        "kernel void k(device uint* out [[buffer(0)]], device const uint* in_ [[buffer(1)]],\n"
        "              uint tid [[thread_position_in_grid]]) {\n"
        "    uint v = in_[tid] ^ (tid * 7u);\n"
        f"    out[tid] = {call};\n"
        "}\n"
    )


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    "name", sorted(name for name, case in CASES.items() if case[1])
)
def test_quad_function_imports_as_crossgl_quad_operation(tmp_path, name):
    call, expected = CASES[name]
    crossgl = _translate(tmp_path, _kernel(call), "cgl")
    assert expected in crossgl
    assert "quad_" not in crossgl


@pytest.mark.parametrize(
    ("call", "expected"),
    [
        ("quad_shuffle_xor(v, ushort(2))", "QuadReadAcrossY(v)"),
        ("quad_shuffle_xor(v, (ushort)3)", "QuadReadAcrossDiagonal(v)"),
        # uchar(257) is 1.
        ("quad_shuffle_xor(v, uchar(257))", "QuadReadAcrossX(v)"),
        ("quad_broadcast(v, ushort(1))", "QuadReadLaneAt(v, 1u)"),
        ("quad_shuffle(v, uint(65538))", "QuadReadLaneAt(v, 2u)"),
    ],
)
def test_a_converted_constant_lane_stays_a_constant(tmp_path, call, expected):
    crossgl = _translate(tmp_path, _kernel(call), "cgl")
    assert expected in crossgl


def test_user_function_named_like_a_quad_function_is_called(tmp_path):
    source = HEADER + """
uint quad_shuffle_xor(uint value, uint mask) { return value + mask; }
kernel void k(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = quad_shuffle_xor(tid, 1u);
}
"""
    crossgl = _translate(tmp_path, source, "cgl")
    assert "QuadReadAcrossX" not in crossgl
    assert "quad_shuffle_xor(tid, 1u)" in crossgl


def test_quad_functions_without_crossgl_operation_keep_their_metal_names(tmp_path):
    crossgl = _translate(tmp_path, _kernel("quad_prefix_inclusive_sum(v)"), "cgl")
    assert "quad_prefix_inclusive_sum(v)" in crossgl


@pytest.mark.parametrize("name", sorted(CASES))
def test_imported_quad_function_validates_as_glsl_and_spirv(tmp_path, name):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "unit.comp"
    shader.write_text(_translate(tmp_path, _kernel(CASES[name][0]), "opengl"))
    sizes = [f"-DCROSSGL_WORKGROUP_SIZE_{axis}=64" for axis in "X"] + [
        f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "YZ"
    ]
    for environment in ([], ["--target-env", "vulkan1.1"]):
        result = subprocess.run(
            [validator, "-l", *environment, "-S", "comp", *sizes, str(shader)],
            capture_output=True,
            text=True,
            # A Vulkan target writes comp.spv to the working directory.
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def _run_metal_kernel(source, count, group):
    metal = pytest.importorskip("Metal")
    device = metal.MTLCreateSystemDefaultDevice()
    if device is None:
        pytest.skip("no Metal device")
    library, error = device.newLibraryWithSource_options_error_(source, None, None)
    assert library is not None, error
    pipeline, error = device.newComputePipelineStateWithFunction_error_(
        library.newFunctionWithName_("k"), None
    )
    assert pipeline is not None, error
    words = [((i * 2654435761) & 0xFFFFFFFF) >> 7 for i in range(count)]
    data = struct.pack(f"<{count}I", *words)
    output = device.newBufferWithLength_options_(4 * count, 0)
    inputs = device.newBufferWithBytes_length_options_(data, len(data), 0)
    commands = device.newCommandQueue().commandBuffer()
    encoder = commands.computeCommandEncoder()
    encoder.setComputePipelineState_(pipeline)
    encoder.setBuffer_offset_atIndex_(output, 0, 0)
    encoder.setBuffer_offset_atIndex_(inputs, 0, 1)
    encoder.dispatchThreads_threadsPerThreadgroup_(
        metal.MTLSizeMake(count, 1, 1), metal.MTLSizeMake(group, 1, 1)
    )
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    return struct.unpack(f"<{count}I", output.contents().as_buffer(4 * count))


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_computes_what_the_native_quad_function_computes(tmp_path, name):
    native = _kernel(CASES[name][0])
    translated = _translate(tmp_path, native, "metal")
    expected = _run_metal_kernel(native, 256, 64)
    assert _run_metal_kernel(translated, 256, 64) == expected
    if name not in {"all", "any"}:
        # The values differ between lanes, so a read of the wrong lane changes the result.
        assert len(set(expected)) > 4


CROSSGL_LANE_READS = """
shader S {
    compute {
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        buffer uint* out_;
        buffer uint* lanes;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            out_[i] = QuadReadLaneAt(i, lanes[i]) + QuadReadLaneAt(i, 2u);
        }
    }
}
"""


def test_metal_output_reads_a_run_time_quad_lane_with_quad_shuffle(tmp_path):
    # quad_broadcast's lane must be the same in every thread of the quad-group; quad_shuffle's may differ.
    path = tmp_path / "unit.cgl"
    path.write_text(CROSSGL_LANE_READS, encoding="utf-8")
    metal = crosstl.translate(str(path), backend="metal")
    assert "quad_shuffle(i, ushort((lanes[i]) & 3u))" in metal
    assert "quad_broadcast(i, ushort(2u))" in metal
