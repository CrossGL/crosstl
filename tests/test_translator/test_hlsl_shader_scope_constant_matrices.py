"""Shader-scope constant matrices in HLSL output and HLSL import.

HLSL declares a constant matrix directly (``static const float2x2 R = ...;``), so a CrossGL ``static const``
matrix must not be refused, and a product with a shader-scope ``const`` matrix must be written with ``mul``
because ``*`` is a component-wise product in HLSL. The HLSL importer also has to read an array of matrices whose
brace initializer lists one matrix constructor per element.
"""

import shutil
import subprocess

import pytest

import crosstl

STATIC_CONST = """
shader S {
    static const mat2 ROTATE = mat2(0.5, -1.0, 2.0, 1.5);
    buffer float values[];
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            values[i] = (ROTATE * vec2(values[i], 1.0)).x;
        }
    }
}
"""

PLAIN_CONST = """
shader S {
    const mat2 ROTATE = mat2(0.5, -1.0, 2.0, 1.5);
    buffer float values[];
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec2 v = ROTATE * vec2(values[i], 1.0);
            vec2 u = vec2(values[i], 2.0) * ROTATE;
            mat2 m = ROTATE;
            vec2 w = m * vec2(values[i], 1.0);
            values[i] = v.x + w.y + u.x;
        }
    }
}
"""

AGGREGATES = """
shader S {
    struct Frame {
        mat2 basis;
        float scale;
    }
    static const mat3 STACK[2] = {mat3(1.0), mat3(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0), vec3(7.0, 8.0, 9.0))};
    static const Frame FRAME = {mat2(1.0, 2.0, 3.0, 4.0), 0.5};
    buffer float values[];
    compute {
        layout(local_size_x = 1) in;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            vec3 a = STACK[1] * vec3(values[i], 1.0, 0.0);
            vec2 b = FRAME.basis * a.xy * FRAME.scale;
            values[i] = a.z + b.y;
        }
    }
}
"""

HLSL_SOURCE = """
static const float2x2 R = float2x2(0.5, -1.0, 2.0, 1.5);
static const float3x3 STACK[2] = {
    float3x3(1, 0, 0, 0, 1, 0, 0, 0, 1),
    float3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)
};
RWStructuredBuffer<float> values : register(u0);
[numthreads(1, 1, 1)]
void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    float2 v = mul(R, float2(values[i], 1.0));
    float3 w = mul(STACK[1], float3(v, 1.0));
    values[i] = v.x + w.z;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend)


def _compile(tmp_path, hlsl):
    dxc = shutil.which("dxc")
    if dxc is None:
        return
    path = tmp_path / "out.hlsl"
    path.write_text(hlsl, encoding="utf-8")
    completed = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]


def test_static_const_matrix_is_declared_and_multiplied_with_mul(tmp_path):
    hlsl = _translate(tmp_path, STATIC_CONST, ".cgl", "directx")
    assert (
        "static const float2x2 ROTATE = transpose(float2x2(0.5, -1.0, 2.0, 1.5));"
        in hlsl
    )
    assert "mul(ROTATE, float2(values[i], 1.0)).x" in hlsl
    _compile(tmp_path, hlsl)


def test_plain_const_matrix_product_uses_mul(tmp_path):
    hlsl = _translate(tmp_path, PLAIN_CONST, ".cgl", "directx")
    assert "float2 v = mul(ROTATE, float2(values[i], 1.0));" in hlsl
    assert "float2 u = mul(float2(values[i], 2.0), ROTATE);" in hlsl
    assert "ROTATE *" not in hlsl and "* ROTATE" not in hlsl
    _compile(tmp_path, hlsl)


def test_matrix_elements_and_fields_of_compile_time_aggregates(tmp_path):
    hlsl = _translate(tmp_path, AGGREGATES, ".cgl", "directx")
    assert "static const float3x3 STACK[2] = {" in hlsl
    assert "transpose(float3x3(float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0)" in hlsl
    assert (
        "static const Frame FRAME = {transpose(float2x2(1.0, 2.0, 3.0, 4.0)), 0.5};"
        in (hlsl)
    )
    assert "mul(STACK[1], float3(values[i], 1.0, 0.0))" in hlsl
    assert "mul(FRAME.basis, a.xy)" in hlsl
    _compile(tmp_path, hlsl)


def test_hlsl_import_reads_one_matrix_constructor_per_array_element(tmp_path):
    crossgl = _translate(tmp_path, HLSL_SOURCE, ".hlsl", "cgl")
    assert "static const mat2 R = mat2(0.5, 2.0, -1.0, 1.5);" in crossgl
    assert (
        "static const mat3 STACK[2] = {mat3(1, 0, 0, 0, 1, 0, 0, 0, 1), "
        "mat3(1, 4, 7, 2, 5, 8, 3, 6, 9)};"
    ) in crossgl


def test_hlsl_constant_matrices_round_trip_to_hlsl(tmp_path):
    crossgl = _translate(tmp_path, HLSL_SOURCE, ".hlsl", "cgl")
    hlsl = _translate(tmp_path, crossgl, ".cgl", "directx")
    assert "static const float2x2 R = transpose(float2x2(0.5, 2.0, -1.0, 1.5));" in hlsl
    assert "transpose(float3x3(1, 4, 7, 2, 5, 8, 3, 6, 9))" in hlsl
    assert "mul(R, float2(values[i], 1.0))" in hlsl
    assert "mul(STACK[1], float3(v, 1.0))" in hlsl
    _compile(tmp_path, hlsl)


def test_hlsl_import_refuses_an_array_element_of_another_matrix_shape(tmp_path):
    source = HLSL_SOURCE.replace(
        "float3x3(1, 2, 3, 4, 5, 6, 7, 8, 9)", "float2x2(1, 2, 3, 4)"
    )
    with pytest.raises(Exception, match="cannot match to the matrix"):
        _translate(tmp_path, source, ".hlsl", "cgl")
