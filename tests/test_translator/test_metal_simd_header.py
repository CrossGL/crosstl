"""Metal shaders that share simd.h types with their host translate without Apple's SDK headers."""

import re

import crosstl

SHARED_HEADER = """#ifndef SHARED_TYPES_H
#define SHARED_TYPES_H
#include <TargetConditionals.h>
#include <simd/simd.h>
typedef struct {
    vector_float2 offset;
    vector_float4 tint;
    matrix_float4x4 transform;
    vector_uint2 size;
} Uniforms;
#endif
"""

SHADER = """#include <metal_stdlib>
#include "SharedTypes.h"
using namespace metal;
struct VOut { float4 position [[position]]; float4 color; };
fragment float4 fmain(VOut in [[stage_in]], constant Uniforms &u [[buffer(0)]]) {
    vector_float2 scaled = vector_float2(u.size) * u.offset;
    vector_int3 cell = vector_int3(1, 2, 3);
    matrix_float2x2 m = matrix_float2x2(1.0);
    return u.transform * vector_float4(scaled * (m * float2(1.0)), float(cell.x), 1.0) * u.tint;
}
"""


def _translate(tmp_path, backend):
    (tmp_path / "SharedTypes.h").write_text(SHARED_HEADER)
    path = tmp_path / "shader.metal"
    path.write_text(SHADER)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_import_reads_simd_typedefs_as_crossgl_types(tmp_path):
    crossgl = _translate(tmp_path, "cgl")

    assert not re.search(r"\b(vector|matrix)_(float|int|uint)\d", crossgl), crossgl
    assert "vec2(u.size)" in crossgl or re.search(r"vec2\(\(?u\.size\)?\)", crossgl)
    assert re.search(r"ivec3\(1, 2, 3\)", crossgl), crossgl
    assert re.search(r"mat2\(1\.0\)", crossgl), crossgl


def test_hlsl_and_glsl_drop_apple_sdk_headers(tmp_path):
    for backend in ("directx", "opengl"):
        code = _translate(tmp_path, backend)

        assert "simd/simd.h" not in code, code
        assert "TargetConditionals.h" not in code, code
        assert "metal_stdlib" not in code, code
        assert not re.search(r"\b(vector|matrix)_(float|int|uint)\d", code), code
