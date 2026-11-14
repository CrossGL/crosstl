"""Metal ``make_filled_matrix<T, Cols, Rows>(value)`` imports as a matrix whose every element is ``T(value)``.

MSL 4.1 section 6.7 declares ``template <typename T, int Cols, int Rows = Cols, typename U> matrix<T, Cols, Rows>
make_filled_matrix(U value)``. CrossGL has no such function, and the importer passed the call through as a call to an
undeclared function, so the Metal round trip failed with ``use of undeclared identifier
'make_filled_matrix_u3cfloat_u2c3_u3e'`` and every other target was wrong too.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.import_refusals import MetalImportRefused

SOURCE = """
#include <metal_stdlib>
using namespace metal;
typedef half Element;

kernel void probe(device float* fin [[buffer(0)]],
                  device float3* outp [[buffer(1)]],
                  device float2* outs [[buffer(2)]],
                  device half2* outh [[buffer(3)]],
                  uint3 id [[thread_position_in_grid]]) {
    uint i = id.x;
    float a0 = fin[i + 0u];
    float3x3 r = make_filled_matrix<float, 3>(a0);
    metal::half2x4 h = metal::make_filled_matrix<Element, 2, 4>(fin[i]);
    float4x2 s = make_filled_matrix<float, 4, 2>(fin[i++]);
    outp[i] = r[0] + make_filled_matrix<float, 3>(1)[2];
    outs[i] = s[3];
    outh[i] = h[1].xy;
}
"""


def _translate(tmp_path, backend):
    path = tmp_path / "probe.metal"
    path.write_text(SOURCE)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_make_filled_matrix_imports_as_column_splats(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert "make_filled_matrix" not in crossgl
    assert "mat3 r = mat3(vec3(a0), vec3(a0), vec3(a0));" in crossgl
    # Rows defaults to Cols, a typedef element type resolves, and Rows sets each column's width.
    assert (
        "f16mat2x4 h = f16mat2x4(f16vec4(buffer_load(fin, i)), f16vec4(buffer_load(fin, i)));"
        in crossgl
    )
    assert "mat3(vec3(1), vec3(1), vec3(1))[2]" in crossgl


def test_make_filled_matrix_evaluates_a_side_effecting_value_once(tmp_path):
    crossgl = _translate(tmp_path, "cgl")
    assert crossgl.count("i++") == 1
    assert (
        "mat4x2 s = (mat4x2(vec2(1.0), vec2(1.0), vec2(1.0), vec2(1.0)) * float(buffer_load(fin, i++)));"
        in crossgl
    )


def test_user_make_filled_matrix_keeps_its_meaning(tmp_path):
    path = tmp_path / "user.metal"
    path.write_text("""
#include <metal_stdlib>
using namespace metal;
template <typename T, int N>
float3x3 make_filled_matrix(T value) { return float3x3(value); }
kernel void probe(device float* fin [[buffer(0)]], device float3* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {
    outp[id.x] = make_filled_matrix<float, 3>(fin[id.x])[0];
}
""")
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "make_filled_matrix_float_3(buffer_load(fin, id.x))[0]" in crossgl
    assert "return mat3(value);" in crossgl


@pytest.mark.parametrize(
    "buffer_type, call",
    [
        ("bfloat", "make_filled_matrix<bfloat, 2>(fin[id.x])[1][0]"),
        ("float", "make_filled_matrix<float, 5>(fin[id.x])[4][4]"),
        ("float", "make_filled_matrix<float, N>(fin[id.x])[1][0]"),
    ],
)
def test_matrix_crossgl_cannot_spell_is_refused(tmp_path, buffer_type, call):
    path = tmp_path / "refused.metal"
    path.write_text(f"""
#include <metal_stdlib>
using namespace metal;
constant int N = 3;
kernel void probe(device {buffer_type}* fin [[buffer(0)]], device float* outp [[buffer(1)]],
                  uint3 id [[thread_position_in_grid]]) {{
    outp[id.x] = float({call});
}}
""")
    with pytest.raises(MetalImportRefused, match="make_filled_matrix<"):
        crosstl.translate(str(path), backend="cgl", format_output=False)


def test_metal_round_trip_has_no_make_filled_matrix_call(tmp_path):
    metal = _translate(tmp_path, "metal")
    assert "make_filled_matrix" not in metal
    assert "float3x3 r = float3x3(float3(a0), float3(a0), float3(a0));" in metal


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator not installed"
)
def test_glsl_output_validates(tmp_path):
    glsl = _translate(tmp_path, "glsl")
    shader = tmp_path / "probe.comp"
    shader.write_text(glsl)
    result = subprocess.run(
        ["glslangValidator", str(shader)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("backend", ["directx", "opengl"])
def test_other_targets_translate(tmp_path, backend):
    output = _translate(tmp_path, backend)
    assert "make_filled_matrix" not in output
