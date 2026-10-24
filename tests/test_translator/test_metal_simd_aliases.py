"""Metal's ``<simd/simd.h>`` aliases as constructors, and the header itself on other targets.

Apple's SIMD library names Metal's vector and matrix types with aliases that Metal and host code share
(``vector_float2`` is ``float2``). The importer mapped the aliases as declared types but kept them verbatim as
constructors, so ``vector_float2(viewport)`` reached HLSL and GLSL unchanged, and SPIR-V refused the call. HLSL and
GLSL also kept ``#include <simd/simd.h>``, which neither compiler can resolve.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """
#include <metal_stdlib>
#include <simd/simd.h>
using namespace metal;

kernel void shade(device float *out [[buffer(0)]],
                  uint tid [[thread_position_in_grid]]) {
    vector_float2 half_size = vector_float2(float(tid), 2.0) / 2.0;
    simd_int3 offsets = simd_int3(1, 2, 3);
    simd_uint2 counts = simd_uint2(tid, 4u);
    matrix_float4x4 scale = matrix_float4x4(vector_float4(2.0, 0.0, 0.0, 0.0),
                                            vector_float4(0.0, 2.0, 0.0, 0.0),
                                            vector_float4(0.0, 0.0, 2.0, 0.0),
                                            vector_float4(0.0, 0.0, 0.0, 1.0));
    vector_float4 point = scale * vector_float4(half_size, 1.0, 1.0);
    matrix_float2x2 rotation = matrix_float2x2(vector_float2(0.0, 1.0), vector_float2(-1.0, 0.0));
    vector_uint2 lanes = vector_uint2(tid, 1u);
    vector_float2 turned = rotation * point.xy;
    out[tid] = turned.x + float(offsets.y) + float(counts.y) + float(lanes.y);
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "shade.metal"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(
        str(path), backend=backend, source_options={"threadgroup_size": (64, 1, 1)}
    )


def _run(command, **kwargs):
    completed = subprocess.run(command, capture_output=True, text=True, **kwargs)
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]


def test_alias_constructors_import_as_crossgl_constructors(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "vec2 half_size = vec2(float(tid), 2.0) / 2.0;" in crossgl
    assert "ivec3 offsets = ivec3(1, 2, 3);" in crossgl
    assert "uvec2 counts = uvec2(tid, 4u);" in crossgl
    assert "mat4 scale = mat4(vec4(2.0, 0.0, 0.0, 0.0)" in crossgl
    assert "mat2 rotation = mat2(vec2(0.0, 1.0), vec2((-1.0), 0.0));" in crossgl
    assert "uvec2 lanes = uvec2(tid, 1u);" in crossgl
    for alias in ("vector_float", "simd_int", "simd_uint", "matrix_float"):
        assert f"{alias}" not in crossgl.split("using namespace", 1)[-1]


def test_hlsl_drops_the_simd_header_and_compiles(tmp_path):
    hlsl = _translate(tmp_path, "directx")
    assert "#include" not in hlsl
    assert "float2 half_size = " in hlsl
    assert "float4x4 scale = " in hlsl
    assert "float2x2 rotation = " in hlsl
    for alias in ("vector_float", "simd_int", "simd_uint", "matrix_float"):
        assert alias not in hlsl
    dxc = shutil.which("dxc")
    if dxc:
        path = tmp_path / "shade.hlsl"
        path.write_text(hlsl, encoding="utf-8")
        entry = re.search(r"\[numthreads\([^)]*\)\]\s*void\s+(\w+)", hlsl).group(1)
        _run([dxc, "-T", "cs_6_0", "-E", entry, str(path)], timeout=180)


def test_glsl_drops_the_simd_header_and_validates(tmp_path):
    glsl = _translate(tmp_path, "opengl")
    assert "#include" not in glsl
    assert "vector_float" not in glsl
    validator = shutil.which("glslangValidator")
    if validator:
        path = tmp_path / "shade.comp"
        path.write_text(glsl, encoding="utf-8")
        _run([validator, str(path)])


def test_spirv_lowers_the_alias_constructors(tmp_path):
    spirv = _translate(tmp_path, "vulkan")
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if spirv_as is None or spirv_val is None:
        pytest.skip("spirv-as and spirv-val are not installed")
    assembly, module = tmp_path / "shade.spvasm", tmp_path / "shade.spv"
    assembly.write_text(spirv, encoding="utf-8")
    _run([spirv_as, "--target-env", "vulkan1.1", str(assembly), "-o", str(module)])
    _run([spirv_val, "--target-env", "vulkan1.1", str(module)])


def test_metal_keeps_the_simd_header(tmp_path):
    assert "#include <simd/simd.h>" in _translate(tmp_path, "metal")
