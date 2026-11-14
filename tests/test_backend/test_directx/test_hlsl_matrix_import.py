"""HLSL floatRxC imports as CrossGL matCxR with rows, row-filling constructors and _mRC swizzles lowered (ADR 0004)."""

import textwrap

import pytest

from crosstl.backend.DirectX.DirectxCrossGLCodeGen import HLSLToCrossGLConverter
from crosstl.backend.DirectX.DirectxLexer import HLSLLexer
from crosstl.backend.DirectX.DirectxParser import HLSLParser
from crosstl.backend.import_refusals import DirectXImportRefused
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def to_crossgl(body, globals_=""):
    source = textwrap.dedent(globals_) + textwrap.dedent("""
        RWStructuredBuffer<float> o : register(u0);
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
        %s
        }
        """) % textwrap.indent(textwrap.dedent(body), "    ")
    output = HLSLToCrossGLConverter().generate(
        HLSLParser(HLSLLexer(source).tokenize()).parse()
    )
    Parser(Lexer(output).get_tokens()).parse()
    return output


@pytest.mark.parametrize(
    "hlsl, crossgl",
    [
        ("float2x3", "mat3x2"),
        ("float3x2", "mat2x3"),
        ("float4x3", "mat3x4"),
        ("float3x3", "mat3"),
        ("half2x4", "f16mat4x2"),
        ("min16float4x2", "f16mat2x4"),
        ("double3x4", "dmat4x3"),
        ("matrix<float, 2, 4>", "mat4x2"),
        ("float64_t2x3", "dmat3x2"),
    ],
)
def test_matrix_types_swap_rows_and_columns(hlsl, crossgl):
    assert f"{crossgl} m;" in to_crossgl(f"{hlsl} m;")


def test_constructors_and_initializer_lists_fill_rows():
    output = to_crossgl("""
        float2x3 a = float2x3(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
        float3x3 b = float3x3(float3(1.0, 2.0, 3.0), float3(4.0, 5.0, 6.0), float3(7.0, 8.0, 9.0));
        float2x2 c = {1.0, 2.0, 3.0, 4.0};
        float2x2 d = {{1.0, 2.0}, {3.0, 4.0}};
        float2x2 e[2] = {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}};
        float2x2 f = float2x2(float4(1.0, 2.0, 3.0, 4.0));
        o[0] = a[0][0] + b[0][0] + c[0][0] + d[0][0] + e[1][0][0] + f[0][0];
    """)
    # Side-effect-free scalars are re-listed column by column, so constant matrices stay constant expressions.
    assert "mat3x2 a = mat3x2(1.0, 4.0, 2.0, 5.0, 3.0, 6.0);" in output
    assert (
        "mat3 b = transpose(mat3(vec3(1.0, 2.0, 3.0), vec3(4.0, 5.0, 6.0), vec3(7.0, 8.0, 9.0)));"
        in output
    )
    assert "mat2 c = mat2(1.0, 3.0, 2.0, 4.0);" in output
    assert "mat2 d = mat2(1.0, 3.0, 2.0, 4.0);" in output
    assert "mat2 e[2] = {mat2(1.0, 3.0, 2.0, 4.0), mat2(5.0, 7.0, 6.0, 8.0)};" in output
    assert "mat2 f = transpose(mat2(vec4(1.0, 2.0, 3.0, 4.0)));" in output
    assert "e[1][0][0]" in output


def test_casts_truncate_matrices_and_splat_scalars():
    output = to_crossgl(
        """
        float3x3 a = (float3x3)world;
        float4x3 b = (float4x3)world;
        float3x3 c = (float3x3)0;
        float2x2 d = (float2x2)o[0];
        o[1] = a[0][0] + b[0][0] + c[0][0] + d[0][0];
    """,
        "cbuffer Frame : register(b0) { float4x4 world; };\n",
    )
    assert "mat3 a = mat3(world);" in output
    assert "mat3x4 b = mat3x4(world);" in output
    assert "mat3 c = mat3(0);" in output
    assert "mat2 d = mat2(vec2(o[0]), vec2(o[0]));" in output


def test_scalar_stored_to_a_matrix_fills_every_component():
    output = to_crossgl(
        """
        float x = o[0];
        float3x3 a = 1.0;
        float3x3 b = x;
        float3x3 m;
        m = 2.0;
        o[1] = a[0][0] + b[1][1] + m[2][2] + total(4.0);
    """,
        """
        float total(float3x3 v) { return v[0][0]; }
        float3x3 filled(float s) { return s; }
        """,
    )
    assert "mat3 a = mat3(vec3(1.0), vec3(1.0), vec3(1.0));" in output
    assert "mat3 b = mat3(vec3(x), vec3(x), vec3(x));" in output
    assert "m = mat3(vec3(2.0), vec3(2.0), vec3(2.0));" in output
    assert "total(mat3(vec3(4.0), vec3(4.0), vec3(4.0)))" in output
    assert "return mat3(vec3(s), vec3(s), vec3(s));" in output


def test_scalar_expressions_are_splatted_like_single_operands():
    output = to_crossgl("""
        float x = o[0];
        float3x3 r = x * 2.0 + 1.0;
        float3x3 t = id.x > 0 ? x : 1.0;
        float3x3 c = (float)id.x;
        float3x3 n = -x;
        float3x3 s = sin(x);
        o[1] = r[0][0] + t[1][1] + c[2][2] + n[0][0] + s[1][2];
    """)
    assert "mat3 r = mat3(vec3((x * 2.0) + 1.0)," in output
    assert "mat3 t = mat3(vec3(id.x > 0 ? x : 1.0)," in output
    assert (
        "mat3 c = mat3(vec3(float(id.x)), vec3(float(id.x)), vec3(float(id.x)));"
        in output
    )
    assert "mat3 n = mat3(vec3(-x), vec3(-x), vec3(-x));" in output
    assert "mat3 s = mat3(vec3(sin(x)), vec3(sin(x)), vec3(sin(x)));" in output


def test_scalar_results_of_known_intrinsics_are_splatted():
    output = to_crossgl("""
        float3 v = float3(o[0], 1.0, 2.0);
        float3x3 a = length(v);
        float3x3 b = dot(v, v);
        float3x3 c = distance(v, v);
        float3x3 d = asfloat(id.x) * 1.0;
        float3x3 e = true;
        o[1] = a[0][0] + b[0][0] + c[0][0] + d[0][0] + e[0][0];
    """)
    assert "mat3 a = mat3(vec3(length(v)), vec3(length(v)), vec3(length(v)));" in output
    assert "mat3 b = mat3(vec3(dot(v, v)), vec3(dot(v, v)), vec3(dot(v, v)));" in output
    assert "mat3 c = mat3(vec3(distance(v, v))," in output
    assert "mat3 d = mat3(vec3(uintBitsToFloat(id.x) * 1.0)," in output
    assert (
        "mat3 e = mat3(vec3(float(true)), vec3(float(true)), vec3(float(true)));"
        in output
    )


def test_scalar_mul_and_select_are_splatted():
    output = to_crossgl("""
        float x = o[0];
        float3x3 a = mul(x, 2.0);
        float3x3 b = select(x > 1.0, 1.0, 2.0);
        o[1] = a[0][0] + b[1][1];
    """)
    assert "mat3 a = mat3(vec3((x * 2.0)), vec3((x * 2.0)), vec3((x * 2.0)));" in output
    assert "mat3 b = mat3(vec3(select(2.0, 1.0, x > 1.0))," in output


def test_bool_and_integer_scalars_are_converted_to_float_before_the_splat():
    # WGSL's vec3<f32>(true) is a type error, and an integer dot is an integer.
    output = to_crossgl("""
        float x = o[0];
        int3 iv = int3(1, 2, 3);
        float3x3 a = true;
        float3x3 b = x > 1.0;
        float3x3 c = (bool)id.x;
        float3x3 d = dot(iv, iv);
        float3x3 e = id.x * 2 + 1;
        float3x3 f = x * 2.0;
        o[1] = a[0][0] + b[0][0] + c[0][0] + d[0][0] + e[0][0] + f[0][0];
    """)
    assert (
        "mat3 a = mat3(vec3(float(true)), vec3(float(true)), vec3(float(true)));"
        in output
    )
    assert "mat3 b = mat3(vec3(float(x > 1.0)), vec3(float(x > 1.0))," in output
    assert "mat3 c = mat3(vec3(float(bool(id.x))), vec3(float(bool(id.x)))," in output
    assert "mat3 d = mat3(vec3(float(dot(iv, iv))), vec3(float(dot(iv, iv)))," in output
    assert (
        "mat3 e = mat3(vec3(float((id.x * 2) + 1)), vec3(float((id.x * 2) + 1)),"
        in output
    )
    # A float stays as it is.
    assert "mat3 f = mat3(vec3(x * 2.0), vec3(x * 2.0), vec3(x * 2.0));" in output


def test_values_of_unknown_type_are_not_written_unchanged():
    with pytest.raises(DirectXImportRefused):
        to_crossgl("float3x3 m = mystery;\no[1] = m[0][0];")


@pytest.mark.parametrize(
    "statement",
    ["float3x3 m = x++;", "float3x3 m = f(x);", "float3x3 m = (float)(x++);"],
)
def test_scalar_stores_that_cannot_be_repeated_are_refused(statement):
    with pytest.raises(DirectXImportRefused):
        to_crossgl(
            "float x = o[0];\n" + statement + "\no[1] = m[0][0];",
            "float f(float v) { return v; }\n",
        )


METHOD_STRUCT = """
    struct S {
        float3x3 m;
        float3x3 get() { return m; }
        float sc() { return m[0][0]; }
        float two() { float t = m[0][0]; return t * 2.0; }
        float3x3 bump() { m[0][0] += 1.0; return m; }
        float sbump() { m[0][0] += 1.0; return m[0][0]; }
        float tw(inout float q) { q = 1.0; return q; }
    };
"""


def test_matrix_returned_by_a_method_is_stored_as_it_is():
    output = to_crossgl(
        """
        float x = o[0];
        S s;
        s.m = (float3x3)x;
        float3x3 a = s.get();
        float3x3 b = s.bump();
        o[1] = a[0][0] + b[1][1];
    """,
        METHOD_STRUCT,
    )
    assert "mat3 a = S_get(s);" in output
    assert "mat3 b = S_bump(s);" in output


def test_scalar_returned_by_a_pure_method_is_splatted():
    output = to_crossgl(
        """
        float x = o[0];
        S s;
        s.m = (float3x3)x;
        float3x3 a = s.get()[0][0];
        float3x3 b = s.sc();
        float3x3 c = s.two();
        o[1] = a[0][0] + b[1][1] + c[2][2];
    """,
        METHOD_STRUCT,
    )
    assert "mat3 a = mat3(vec3(S_get(s)[0][0]), vec3(S_get(s)[0][0])," in output
    assert "mat3 b = mat3(vec3(S_sc(s)), vec3(S_sc(s)), vec3(S_sc(s)));" in output
    assert "mat3 c = mat3(vec3(S_two(s)), vec3(S_two(s)), vec3(S_two(s)));" in output


@pytest.mark.parametrize(
    "statement",
    [
        "float3x3 a = s.sbump();",
        "float3x3 a = s.bump()[0][0];",
        "float3x3 a = s.tw(x);",
    ],
)
def test_scalar_returned_by_a_method_that_changes_state_is_refused(statement):
    with pytest.raises(DirectXImportRefused):
        to_crossgl(
            "float x = o[0];\nS s;\ns.m = (float3x3)x;\n"
            + statement
            + "\no[1] = a[0][0];",
            METHOD_STRUCT,
        )


def test_matrix_values_are_not_splatted():
    output = to_crossgl("""
        float3x3 a = (float3x3)1.0;
        float3x3 b = a;
        float3x3 c;
        c = b;
        c *= 2.0;
        c._m00 = 3.0;
        o[0] = c[0][0];
    """)
    assert "mat3 b = a;" in output
    assert "c = b;" in output
    assert "c[0][0] = 3.0;" in output


def test_rows_elements_and_swizzles():
    output = to_crossgl("""
        float2x3 m = (float2x3)0;
        float3 r = m[1];
        uint k = id.x;
        float3 rk = m[k];
        float e = m[1][2] + m[0].y + m._m12 + m._11;
        float2 s = m._m02_m10;
        o[0] = r.x + rk.x + e + s.x;
    """)
    assert "vec3 r = transpose(m)[1];" in output
    assert "vec3 rk = transpose(m)[k];" in output
    assert "float e = ((m[2][1] + m[1][0]) + m[2][1]) + m[0][0];" in output
    assert "vec2 s = vec2(m[2][0], m[0][1]);" in output


def test_row_and_element_writes():
    output = to_crossgl("""
        float2x3 m = (float2x3)0;
        int k = int(id.x);
        m[1] = float3(1.0, 2.0, 3.0);
        m[k] += float3(1.0, 2.0, 3.0);
        m[0] = 2.0;
        m[0][2] = 4.0;
        m[1].y = 5.0;
        m._m01 = 6.0;
        o[0] = m[0][0];
    """)
    assert "cgl_hlsl_set_row_mat3x2(m, int(1), vec3(1.0, 2.0, 3.0));" in output
    assert (
        "cgl_hlsl_set_row_mat3x2(m, int(k), vec3(transpose(m)[k] + (vec3(1.0, 2.0, 3.0))));"
        in output
    )
    assert "cgl_hlsl_set_row_mat3x2(m, int(0), vec3(2.0));" in output
    assert "m[2][0] = 4.0;" in output
    assert "m[1][1] = 5.0;" in output
    assert "m[1][0] = 6.0;" in output
    assert (
        "    vec3 cgl_hlsl_set_row_mat3x2(inout mat3x2 target, int row, vec3 value) {\n"
        "        target[0][row] = value.x;\n"
        "        target[1][row] = value.y;\n"
        "        target[2][row] = value.z;\n"
        "        return value;\n"
        "    }\n"
    ) in output


def test_componentwise_star_and_linear_algebra_mul():
    output = to_crossgl("""
        float2x3 a = (float2x3)1;
        float2x3 b = (float2x3)2;
        float2x3 c = a * b;
        a *= b;
        a *= 3.0;
        float2x2 p = mul(a, transpose(b));
        float2 v = mul(a, float3(1.0, 2.0, 3.0));
        float3 w = mul(float2(1.0, 2.0), a);
        o[0] = c[0][0] + p[0][0] + v.x + w.x;
    """)
    assert "mat3x2 c = matrixCompMult(a, b);" in output
    assert "a = matrixCompMult(a, b);" in output
    assert "a *= 3.0;" in output
    assert "mat2 p = (a * transpose(b));" in output
    assert "vec2 v = (a * vec3(1.0, 2.0, 3.0));" in output
    assert "vec3 w = (vec2(1.0, 2.0) * a);" in output


def test_mul_and_componentwise_intrinsics_truncate_wider_vectors_like_dxc():
    output = to_crossgl(
        """
        float3x3 n = (float3x3)world;
        float4 t = float4(1.0, 2.0, 3.0, 4.0);
        float3 a = mul(n, t);
        float3 b = mul(t, n);
        float d = mul(t, float3(1.0, 0.0, 0.0));
        float e = dot(t, a);
        float3 f = max(t, a);
        o[0] = a.x + b.x + d + e + f.x;
    """,
        "float4x4 world;\n",
    )
    assert "vec3 a = (n * t.xyz);" in output
    assert "vec3 b = (t.xyz * n);" in output
    assert "float d = dot(t.xyz, vec3(1.0, 0.0, 0.0));" in output
    assert "float e = dot(t.xyz, a);" in output
    assert "vec3 f = max(t.xyz, a);" in output


def test_constant_matrix_constructor_stays_a_constant_expression():
    output = to_crossgl(
        "o[0] = bias[3][0];",
        """
        static const float4x4 bias = float4x4(
            0.5, 0.0, 0.0, 0.5,
            0.0, 0.5, 0.0, 0.5,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0);
        """,
    )
    assert (
        "mat4 bias = mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);"
        in output
    )
    assert "transpose" not in output


def test_arrays_of_matrices_and_constant_buffer_members():
    output = to_crossgl(
        """
        float4 r = bones[1][2];
        float e = bones[0][1][2] + local[1][0][1];
        o[0] = r.x + e;
    """,
        """
        cbuffer Skin : register(b0) { float3x4 bones[4]; };
        static float2x2 local[2];
        """,
    )
    assert "mat4x3 bones[4];" in output
    assert "vec4 r = transpose(bones[1])[2];" in output
    assert "float e = bones[0][2][1] + local[1][1][0];" in output


@pytest.mark.parametrize(
    "statement, reason",
    [
        ("m._m00_m11 = float2(1.0, 2.0);", "multi-component matrix swizzle"),
        ("m[0].xy = float2(1.0, 2.0);", "multi-component matrix swizzle"),
        ("m[next()] = float2(1.0, 2.0);", "side effects"),
        ("float2x2 n = (float2x2)next();", "side effects"),
    ],
)
def test_unrepresentable_matrix_forms_are_refused(statement, reason):
    with pytest.raises(DirectXImportRefused, match=reason):
        to_crossgl(
            f"""
            float2x2 m = (float2x2)0;
            {statement}
            o[0] = m[0][0];
        """,
            "int next() { return 1; }\n",
        )
