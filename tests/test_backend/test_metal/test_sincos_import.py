"""Metal ``sincos`` imports as a CrossGL helper that every target writes.

MSL's ``T sincos(T x, thread T& cosval)`` returns the sine of ``x`` and stores its cosine in ``cosval`` (MSL 4.1,
section 6.5). CrossGL's ``sincos`` is HLSL's three-operand statement, so the importer writes one helper per namespace
and value type, ``T helper(T value, out T cosine)``, and calls it where the source called ``sincos``. The imported
kernels validate as GLSL and SPIR-V, and on an Apple GPU the round trip computes what the native kernel computes.
"""

import shutil
import struct
import subprocess

import pytest

import crosstl
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalSincosLoweringError

HEADER = """
#include <metal_stdlib>
using namespace metal;
struct Pair { float c; float2 v; };
"""

# name -> (Metal statements that set the float ``r`` from the float ``x`` and the uint ``v``, the CrossGL they import as).
CASES = {
    "scalar": (
        "float c; float s = sincos(x, c); r = s * 3.0 + c;",
        "float s = __crossgl_metal_sincos_default_float(x, c);",
    ),
    "qualified": (
        "float c; float s = metal::sincos(x * 0.5, c); r = s - c * 2.0;",
        "__crossgl_metal_sincos_default_float(x * 0.5, c)",
    ),
    "vector": (
        "float2 c; float2 s = sincos(float2(x, x * 0.25), c); r = s.x + s.y * 2.0 + c.x * 4.0 + c.y * 8.0;",
        "vec2 s = __crossgl_metal_sincos_default_vec2(vec2(x, x * 0.25), c);",
    ),
    "float4": (
        "float4 c; float4 s = sincos(float4(x, -x, x * 3.0, 0.5), c); r = dot(s, float4(1, 2, 3, 4)) + dot(c, float4(5, 6, 7, 8));",
        "__crossgl_metal_sincos_default_vec4(",
    ),
    "half": (
        "half c; half s = sincos(half(x * 0.1), c); r = float(s) * 3.0 + float(c);",
        "float16 s = __crossgl_metal_sincos_default_float16(float16(x * 0.1), c);",
    ),
    "half_to_float_overload": (
        "half h = half(x * 0.1); float c; r = sincos(h, c) + c * 2.0;",
        "__crossgl_metal_sincos_default_float(float(h), c)",
    ),
    "fast": (
        "float c; float s = fast::sincos(x, c); r = s * 3.0 + c;",
        "__crossgl_metal_sincos_fast_float(x, c)",
    ),
    "precise": (
        "float c; float s = metal::precise::sincos(x, c); r = s * 3.0 + c;",
        "__crossgl_metal_sincos_precise_float(x, c)",
    ),
    "array_and_member_destinations": (
        "float arr[3] = {0.0, 0.0, 0.0}; Pair p = {0.0, float2(0.0)}; uint j = v % 3u;"
        " float s1 = sincos(x * 2.0, arr[j]); float s2 = sincos(x * 3.0, p.c); float2 s3 = sincos(float2(x, -x), p.v);"
        " r = s1 + arr[j] * 2.0 + s2 * 4.0 + p.c * 8.0 + s3.x + s3.y * 3.0 + p.v.x * 5.0 + p.v.y * 7.0;",
        "__crossgl_metal_sincos_default_float(x * 3.0, p.c)",
    ),
    "argument_evaluated_once": (
        "uint n = v & 7u; float c; float s = sincos(float(n++) + x, c); r = s + c * 2.0 + float(n) * 4.0;",
        "__crossgl_metal_sincos_default_float(float(n++) + x, c)",
    ),
    "nested_call": (
        "float c; float c2; r = sincos(sincos(x, c) + 1.0, c2) + c + c2 * 2.0;",
        "__crossgl_metal_sincos_default_float(__crossgl_metal_sincos_default_float(x, c) + 1.0, c2)",
    ),
}

# Cases WGSL refuses with a typed diagnostic: WGSL cannot evaluate precise arithmetic.
WGSL_REFUSED = {"precise"}


def _kernel(statements):
    return HEADER + (
        "kernel void k(device uint* out [[buffer(0)]], device const uint* in_ [[buffer(1)]],\n"
        "              uint tid [[thread_position_in_grid]]) {\n"
        "    uint v = in_[tid] ^ (tid * 7u);\n"
        "    float x = float(v & 65535u) * 0.001 - 30.0;\n"
        "    float r = 0.0;\n"
        f"    {statements}\n"
        "    out[tid] = as_type<uint>(r);\n"
        "}\n"
    )


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


@pytest.mark.parametrize("name", sorted(CASES))
def test_sincos_imports_as_a_helper_call(tmp_path, name):
    statements, expected = CASES[name]
    crossgl = _translate(tmp_path, _kernel(statements), "cgl")
    assert expected in crossgl
    # No two-operand Metal sincos call is left for a target to copy.
    assert "sincos(" not in crossgl.replace("__crossgl_metal_sincos_", "")


def test_helper_is_declared_once_per_namespace_and_type(tmp_path):
    statements = (
        "float c; float d; float2 e; float s = sincos(x, c) + sincos(x * 2.0, d);"
        " float2 t = sincos(float2(x), e); float f; s += fast::sincos(x, f);"
        " r = s + c + d + t.x + e.y + f;"
    )
    crossgl = _translate(tmp_path, _kernel(statements), "cgl")
    declaration = (
        "float __crossgl_metal_sincos_default_float(float value, out float cosine) {"
    )
    assert crossgl.count(declaration) == 1
    assert crossgl.count("__crossgl_metal_sincos_default_float(") == 3
    assert (
        crossgl.count(
            "vec2 __crossgl_metal_sincos_default_vec2(vec2 value, out vec2 cosine) {"
        )
        == 1
    )
    assert "cosine = fast::cos(value);" in crossgl
    assert "return fast::sin(value);" in crossgl


def test_precise_helper_calls_the_portable_precise_trigonometry(tmp_path):
    crossgl = _translate(tmp_path, _kernel(CASES["precise"][0]), "cgl")
    assert "cosine = __crossgl_metal_precise_cos_float(value);" in crossgl
    assert "return __crossgl_metal_precise_sin_float(value);" in crossgl


def test_argument_with_a_side_effect_is_evaluated_once(tmp_path):
    crossgl = _translate(tmp_path, _kernel(CASES["argument_evaluated_once"][0]), "cgl")
    assert crossgl.count("n++") == 1


def test_user_function_named_sincos_is_called(tmp_path):
    source = HEADER + """
float sincos(float a, thread float& b) { b = a * 2.0; return a + 1.0; }
kernel void k(device float* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    float c;
    out[tid] = sincos(float(tid), c) + c;
}
"""
    crossgl = _translate(tmp_path, source, "cgl")
    assert "__crossgl_metal_sincos_" not in crossgl
    assert "sincos(float(tid), c)" in crossgl


def test_double_sincos_is_refused(tmp_path):
    source = HEADER + "double f(double x) { double c; return sincos(x, c) + c; }"
    with pytest.raises(MetalSincosLoweringError) as error:
        _translate(tmp_path, source, "cgl")
    assert (
        error.value.project_diagnostic_code
        == "project.translate.metal-sincos-unsupported"
    )


def _sizes():
    return ["-DCROSSGL_WORKGROUP_SIZE_X=64"] + [
        f"-DCROSSGL_WORKGROUP_SIZE_{axis}=1" for axis in "YZ"
    ]


@pytest.mark.parametrize("name", sorted(CASES))
def test_imported_kernel_validates_as_glsl(tmp_path, name):
    validator = shutil.which("glslangValidator")
    if validator is None:
        pytest.skip("glslangValidator is not installed")
    shader = tmp_path / "unit.comp"
    shader.write_text(_translate(tmp_path, _kernel(CASES[name][0]), "opengl"))
    for environment in ([], ["--target-env", "vulkan1.1"]):
        result = subprocess.run(
            [validator, "-l", *environment, "-S", "comp", *_sizes(), str(shader)],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.parametrize("name", sorted(set(CASES) - WGSL_REFUSED))
def test_imported_kernel_validates_as_wgsl(tmp_path, name):
    validator = shutil.which("naga")
    if validator is None:
        pytest.skip("naga is not installed")
    shader = tmp_path / "unit.wgsl"
    shader.write_text(_translate(tmp_path, _kernel(CASES[name][0]), "wgsl"))
    result = subprocess.run(
        [validator, str(shader)], capture_output=True, text=True, cwd=tmp_path
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", sorted(CASES))
def test_imported_kernel_compiles_as_hlsl(tmp_path, name):
    compiler = shutil.which("dxc")
    if compiler is None:
        pytest.skip("dxc is not installed")
    shader = tmp_path / "unit.hlsl"
    shader.write_text(_translate(tmp_path, _kernel(CASES[name][0]), "directx"))
    result = subprocess.run(
        [compiler, "-T", "cs_6_2", "-enable-16bit-types", "-E", "k", str(shader)],
        capture_output=True,
        text=True,
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


def _as_float(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


@pytest.mark.parametrize("name", sorted(CASES))
def test_round_trip_computes_what_the_native_kernel_computes(tmp_path, name):
    native = _kernel(CASES[name][0])
    translated = _translate(tmp_path, native, "metal")
    expected = _run_metal_kernel(native, 512, 64)
    got = _run_metal_kernel(translated, 512, 64)
    assert len(set(expected)) > 4
    if name == "precise":
        # CrossGL lowers precise::sin and precise::cos to portable binary32 routines, which can differ from
        # Apple's in the last bits (precise::sin and precise::cos import the same way). r = 3 sin(x) + cos(x)
        # lies in [-4, 4], so compare it with an absolute bound of a few binary32 ulps of 1.
        assert (
            max(abs(_as_float(a) - _as_float(b)) for a, b in zip(got, expected)) < 1e-6
        )
    else:
        assert got == expected
