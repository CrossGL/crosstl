"""CrossGL matrices follow GLSL: ``matCxR`` has C columns and R rows (ADR 0004).

The HLSL emitter declares ``matCxR`` as ``floatRxC`` and lowers the places where HLSL disagrees: ``m[i]`` (a row in
HLSL, a column in CrossGL), constructors and ``{...}`` initializers (rows in HLSL, columns in CrossGL), ``matN(s)``
(a splat in HLSL, a diagonal in CrossGL) and storage-block loads and stores. MSL spells ``matCxR`` as ``floatCxR``.
"""

import shutil
import subprocess

import pytest

from crosstl.translator.ast import MatrixType
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.metal_codegen import MetalCodeGen
from crosstl.translator.lexer import Lexer
from crosstl.translator.matrix_shape import (
    crossgl_linear_algebra_product_type,
    crossgl_matrix_shape,
)
from crosstl.translator.parser import Parser

SHADER = """
shader Orientation {
    compute {
        layout(std430, binding = 0) buffer Out { float data[]; } outBuf;
        layout(std430, binding = 1) buffer Mats { mat2x3 value; } mats;
        void main() {
            int k = int(outBuf.data[0]);
            mat2x3 m = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
            mat3 d = mat3(2.0);
            vec3 c1 = m[1];
            vec3 ck = m[k];
            m[k] = vec3(7.0, 8.0, 9.0);
            m[0][2] = 10.0;
            mat2x3 g = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
            mat2x3 loaded = mats.value;
            mats.value = g;
            vec3 mv = m * vec2(1.0, 10.0);
            outBuf.data[1] = c1.x + ck.y + mv.z + d[1][1] + float(m.length()) + loaded[1][0];
        }
    }
}
"""


def parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def lines(code, needle):
    return [line.strip() for line in code.splitlines() if needle in line]


def test_matrix_type_records_columns_and_rows():
    variable = parse(SHADER).stages
    statements = next(iter(variable.values())).entry_point.body.statements
    matrix = statements[1].var_type
    assert isinstance(matrix, MatrixType)
    assert (matrix.cols, matrix.rows) == (2, 3)
    assert crossgl_matrix_shape("mat2x3") == ("float", 2, 3)
    assert crossgl_matrix_shape("dmat4") == ("double", 4, 4)


@pytest.mark.parametrize(
    ("left", "right", "product"),
    [
        ("mat2x3", "vec2", "vec3"),
        ("vec3", "mat2x3", "vec2"),
        ("mat2x3", "mat4x2", "mat4x3"),
        ("mat3", "float", "mat3"),
        ("mat2x3", "vec3", None),
        ("vec2", "vec2", None),
    ],
)
def test_linear_algebra_product_types(left, right, product):
    assert crossgl_linear_algebra_product_type(left, right) == product


def test_hlsl_declares_rows_by_columns_and_lowers_columns():
    hlsl = HLSLCodeGen().generate(parse(SHADER))

    assert "float3x2 m = transpose(float2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));" in hlsl
    assert "float3x3 d = float3x3(2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0);" in hlsl
    # A constant column is a matrix swizzle, a dynamic one goes through transpose() or a setter.
    assert "float3 c1 = m._m01_m11_m21;" in hlsl
    assert "float3 ck = transpose(m)[k];" in hlsl
    assert "m = __crossgl_set_column_float3x2(m, k, float3(7.0, 8.0, 9.0));" in hlsl
    assert "m[2][0] = 10.0;" in hlsl
    assert "float3x2 g = transpose(float2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));" in hlsl
    assert "float3 mv = mul(m, float2(1.0, 10.0));" in hlsl
    assert "int(2)" in hlsl
    assert "loaded[0][1]" in hlsl


def test_hlsl_storage_block_matrices_are_stored_by_column():
    hlsl = HLSLCodeGen().generate(parse(SHADER))

    assert (
        "float3x2 loaded = transpose(float2x3(asfloat(mats.Load3(0)), "
        "asfloat(mats.Load3(16))));" in hlsl
    )
    assert "mats.Store3(0, asuint(g._m00_m10_m20));" in hlsl
    assert "mats.Store3(16, asuint(g._m01_m11_m21));" in hlsl


def test_metal_and_glsl_keep_the_column_major_spelling():
    metal = MetalCodeGen().generate(parse(SHADER))
    glsl = GLSLCodeGen().generate(parse(SHADER))

    assert lines(metal, " m = float2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);")
    assert lines(metal, "float3 c1 = m[1];")
    assert "mat2x3 m = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);" in glsl
    assert "mat2x3 g = mat2x3(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0));" in glsl


def test_hlsl_refuses_reordering_two_side_effecting_subscripts():
    source = SHADER.replace("m[0][2] = 10.0;", "m[k++][k++] = 10.0;")
    with pytest.raises(Exception, match="column and row subscripts"):
        HLSLCodeGen().generate(parse(source))


COMPONENTWISE_SHADER = """
shader MatrixColumns {
    compute {
        layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
        buffer float* fin;
        buffer float* outp;
        void main() {
            uint i = gl_GlobalInvocationID.x;
            mat2 m = mat2(fin[i], 2.0, 3.0, 4.0);
            mat2 n = mat2(1.0, 2.0, 3.0, fin[i]);
            vec2 v = vec2(fin[i], 1.0);
            vec3 w = vec3(1.0, 2.0, fin[i]);
            mat2x3 p = mat2x3(1.0, 2.0, 3.0, 4.0, 5.0, fin[i]);
            vec2 a = (m / 2.0)[1];
            vec2 b = (m + 1.0)[1];
            vec2 c = (1.0 - m)[1];
            vec2 d = matrixCompMult(m, n)[1];
            vec2 e = outerProduct(v, v)[1];
            vec2 g = (m - n)[i];
            vec2 h = outerProduct(v, w)[2];
            vec3 j = matrixCompMult(p, p)[1];
            outp[i] = a.x + b.x + c.x + d.x + e.x + g.x + h.x + j.x;
        }
    }
}
"""


def test_hlsl_subscripts_componentwise_matrix_results_by_column():
    hlsl = HLSLCodeGen().generate(parse(COMPONENTWISE_SHADER))

    assert "float2 a = ((m / 2.0))._m01_m11;" in hlsl
    assert "float2 b = ((m + 1.0))._m01_m11;" in hlsl
    assert "float2 c = ((1.0 - m))._m01_m11;" in hlsl
    assert "float2 d = ((m * n))._m01_m11;" in hlsl
    assert "float2 e = (mul(float2x1(v), float1x2(v)))._m01_m11;" in hlsl
    assert "float2 g = transpose(((m - n)))[i];" in hlsl
    # outerProduct(vec2, vec3) is mat3x2: three columns of two rows.
    assert "float2 h = (mul(float2x1(v), float1x3(w)))._m02_m12;" in hlsl
    # mat2x3 is float3x2 in HLSL: column 1 is three rows of HLSL column 1.
    assert "float3 j = ((p * p))._m01_m11_m21;" in hlsl


def test_oriented_hlsl_compiles_with_dxc(tmp_path):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not installed")
    path = tmp_path / "orientation.hlsl"
    path.write_text(HLSLCodeGen().generate(parse(SHADER)))
    result = subprocess.run(
        [dxc, "-T", "cs_6_0", "-E", "CSMain", str(path), "-Fo", str(tmp_path / "o")],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr


METAL_CONSTRUCTORS = """
shader Constructors {
    cbuffer Frame {
        mat4 world;
    }
    compute {
        layout(std430, binding = 0) buffer Out { float data[]; } outBuf;
        void main() {
            vec4 flat = vec4(1.0, 2.0, 3.0, outBuf.data[0]);
            mat2 fromVector = mat2(flat);
            mat2x3 mixed = mat2x3(flat.xy, 3.0, flat.w, flat.zw);
            mat2 columns = mat2(flat.xy, flat.zw);
            mat3x4 resized = mat3x4(world);
            outBuf.data[1] = fromVector[1][0] + mixed[1][2] + columns[0][1] + resized[2][3];
        }
    }
}
"""


def test_metal_spreads_mixed_constructor_components_over_the_columns():
    metal = MetalCodeGen().generate(parse(METAL_CONSTRUCTORS))

    assert (
        "inline float2x2 __crossgl_float2x2_from_float4(float4 p0) {\n"
        "    return float2x2(float2(p0.x, p0.y), float2(p0.z, p0.w));\n}\n"
    ) in metal
    assert (
        "inline float2x3 __crossgl_float2x3_from_float2_float_float_float2("
        "float2 p0, float p1, float p2, float2 p3) {\n"
        "    return float2x3(float3(p0.x, p0.y, p1), float3(p2, p3.x, p3.y));\n}\n"
    ) in metal
    assert lines(metal, "fromVector = __crossgl_float2x2_from_float4(flat);")
    assert lines(metal, "columns = float2x2(flat.xy, flat.zw);")


def test_metal_resizes_constant_buffer_matrices_column_by_column():
    metal = MetalCodeGen().generate(parse(METAL_CONSTRUCTORS))

    # The helper takes the matrix by value and keeps its first three columns.
    assert (
        "inline float3x4 __crossgl_resize_float4x4_to_float3x4(float4x4 m) {\n"
        "    return float3x4(m[0], m[1], m[2]);\n}\n"
    ) in metal
    assert lines(metal, "resized = __crossgl_resize_float4x4_to_float3x4(frame.world);")
