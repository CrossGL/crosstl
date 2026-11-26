"""Metal library functions that CrossGL spells differently import as the CrossGL operations every target writes.

``dfdx``/``dfdy`` are ``dFdx``/``dFdy``, ``rint`` is ``roundEven``, ``max3``/``min3`` (and ``fmax3``/``fmin3``) are two
nested ``max``/``min`` calls, ``median3``/``fmedian3`` call a helper that evaluates each argument once,
``simd_is_first`` is ``WaveIsFirstLane``, the SIMD rotations are ``WaveRotate`` and ``simd_is_helper_thread`` is
``gl_HelperInvocation`` in a fragment function and ``false`` elsewhere (MSL 4.1, sections 6.4, 6.5, 6.8 and 6.9.2).
The imported kernels validate as GLSL and SPIR-V, and on an Apple GPU the round trip computes what the native kernel
computes.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused

HEADER = """
#include <metal_stdlib>
using namespace metal;
"""

# name -> (Metal expression of the uint ``v`` and the uniform ``d``, the CrossGL it imports as).
CASES = {
    "rint": ("as_type<uint>(rint(f * 0.37))", "roundEven(f * 0.37)"),
    "qualified_rint": (
        "as_type<uint>(metal::rint(f) + fast::rint(f * 1.5) + metal::precise::rint(g))",
        "roundEven(f) + roundEven(f * 1.5) + roundEven(g)",
    ),
    "max3": ("as_type<uint>(max3(f, g, h))", "max(f, max(g, h))"),
    "min3": ("as_type<uint>(min3(f, g, h))", "min(f, min(g, h))"),
    "fmax3": ("as_type<uint>(metal::fmax3(h, f, g))", "fmax(h, fmax(f, g))"),
    "fmin3": ("as_type<uint>(fmin3(g, h, f))", "fmin(g, fmin(h, f))"),
    "int_max3": ("uint(max3(x, y, z) - min3(z, y, x))", "max(x, max(y, z))"),
    "median3": (
        "as_type<uint>(median3(f, g, h) + 2.0 * fmedian3(h, g, f))",
        "__crossgl_metal_median3_float(f, g, h)",
    ),
    "int_median3": ("uint(median3(x, y, z) * 5 + median3(x + 1, y - 1, z))", None),
    "vector_median3": (
        "as_type<uint>(dot(median3(float3(f, g, h), float3(h, f, g), float3(g, h, f)), float3(1.0, 2.0, 3.0)))",
        "__crossgl_metal_median3_vec3(",
    ),
    "rotate_down": (
        "simd_shuffle_rotate_down(v, ushort(d))",
        "WaveRotate(v, (uint(uint16(d)) & 65535u))",
    ),
    "rotate_up": (
        "simd_shuffle_rotate_up(v * 3u, ushort(d + 37u))",
        "WaveRotate(v * 3u, 0u - (uint(uint16(d + 37u)) & 65535u))",
    ),
    "rotate_constant": (
        "simd_shuffle_rotate_up(v, 3) ^ simd_shuffle_rotate_down(v, ushort(33))",
        "WaveRotate(v, 0u - (uint(3) & 65535u))",
    ),
    "is_first": (
        "v * 4u + ((v & 1u) != 0u ? (simd_is_first() ? 100u : 1u) : 5u)",
        "WaveIsFirstLane()",
    ),
    "helper_thread": ("simd_is_helper_thread() ? 1000u : v", "false ? 1000u : v"),
}


def _kernel(expression):
    return HEADER + (
        "kernel void k(device uint* out [[buffer(0)]], device const uint* in_ [[buffer(1)]],\n"
        "              uint tid [[thread_position_in_grid]]) {\n"
        "    uint v = in_[tid] ^ (tid * 7u);\n"
        "    uint d = in_[5] & 63u;\n"
        "    float f = float(v % 1000u) * 0.25 - 100.0, g = float(d) - 7.5, h = float(tid) * 0.5;\n"
        "    int x = int(v % 77u) - 30, y = int(d) - 40, z = int(tid) - 20;\n"
        f"    out[tid] = {expression};\n"
        "}\n"
    )


FRAGMENT = HEADER + """
struct VertexOut { float4 position [[position]]; float2 uv; };
bool shaded() { return !simd_is_helper_thread(); }
fragment float4 f(VertexOut in [[stage_in]]) {
    float2 r = dfdx(in.uv) + metal::dfdy(in.uv) * 2.0;
    bool helper = quad_is_helper_thread();
    return float4(r, helper ? 1.0 : 0.0, shaded() ? fwidth(in.uv.x) : 0.0);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize(
    "name", sorted(name for name, case in CASES.items() if case[1])
)
def test_metal_function_imports_as_crossgl_operation(tmp_path, name):
    expression, expected = CASES[name]
    crossgl = _translate(tmp_path, _kernel(expression), "cgl")
    assert expected in crossgl
    for metal_name in (
        "rint(",
        "max3(",
        "min3(",
        "median3(",
        "simd_",
        "is_helper_thread",
    ):
        assert metal_name not in crossgl.replace("__crossgl_metal_median3_", "")


def test_median3_helper_is_declared_once_per_type(tmp_path):
    crossgl = _translate(tmp_path, _kernel(CASES["int_median3"][0]), "cgl")
    assert crossgl.count("int __crossgl_metal_median3_int(int x, int y, int z)") == 1
    assert "return max(min(x, y), min(max(x, y), z));" in crossgl
    assert crossgl.count("__crossgl_metal_median3_int(") == 3


def test_median3_of_mixed_argument_types_is_refused(tmp_path):
    with pytest.raises(MetalImportRefused, match="median3 needs one argument type"):
        _translate(tmp_path, _kernel("uint(median3(x, z, 2u))"), "cgl")


def test_fragment_derivatives_and_helper_threads_import_as_crossgl(tmp_path):
    crossgl = _translate(tmp_path, FRAGMENT, "cgl")
    assert "dFdx(in.uv)" in crossgl.replace("input.", "in.") or "dFdx(" in crossgl
    assert "dFdy(" in crossgl
    assert "dfdx" not in crossgl and "dfdy" not in crossgl
    # The helper function of a program with a fragment function reads the built-in too.
    assert crossgl.count("gl_HelperInvocation") == 2
    assert "is_helper_thread" not in crossgl


def test_user_functions_named_like_the_library_functions_are_called(tmp_path):
    source = HEADER + """
float max3(float a, float b, float c) { return a + b + c; }
uint simd_is_helper_thread() { return 7u; }
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    out[tid] = max3(1.0, 2.0, float(tid)) + float(simd_is_helper_thread());
}
"""
    crossgl = _translate(tmp_path, source, "cgl")
    assert "max3(1.0, 2.0, float(tid))" in crossgl
    assert "simd_is_helper_thread()" in crossgl
    assert "max(1.0" not in crossgl and "false" not in crossgl


def _validate(tmp_path, source, stage, extra=()):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / f"unit.{stage}"
    shader.write_text(_translate(tmp_path, source, "opengl"))
    for environment in ([], ["--target-env", "vulkan1.1"]):
        result = subprocess.run(
            [validator, "-l", *environment, "-S", stage, *extra, str(shader)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", sorted(CASES))
def test_imported_kernel_validates_as_glsl(tmp_path, name):
    sizes = ["-DCROSSGL_WORKGROUP_SIZE_X=64"] + [
        f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "YZ"
    ]
    _validate(tmp_path, _kernel(CASES[name][0]), "comp", sizes)


def test_imported_fragment_validates_as_glsl(tmp_path):
    _validate(tmp_path, FRAGMENT, "frag")


@pytest.mark.parametrize("name", sorted(CASES))
def test_imported_kernel_assembles_as_spirv(tmp_path, name):
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        pytest.skip("spirv-tools are not installed")
    (tmp_path / "unit.spvasm").write_text(
        _translate(tmp_path, _kernel(CASES[name][0]), "vulkan")
    )
    for command in (
        [assembler, "--target-env", "vulkan1.1", "unit.spvasm", "-o", "unit.spv"],
        [validator, "--target-env", "vulkan1.1", "unit.spv"],
    ):
        result = subprocess.run(command, capture_output=True, text=True, cwd=tmp_path)
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


@pytest.mark.parametrize("group", [64, 13])
@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_computes_what_the_native_kernel_computes(tmp_path, name, group):
    native = _kernel(CASES[name][0])
    translated = _translate(tmp_path, native, "metal")
    expected = _run_metal_kernel(native, 256, group)
    assert _run_metal_kernel(translated, 256, group) == expected
    assert len(set(expected)) > 4
