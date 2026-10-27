"""Slang matrices import as the same CrossGL matrix: ``floatRxC`` is ``matCxR`` (ADR 0004)."""

import pytest

import crosstl.translator as cgl_translator
from crosstl import translate
from crosstl.backend.import_refusals import SlangImportRefused
from crosstl.backend.slang import SlangCrossGLCodeGen, SlangLexer, SlangParser


def to_crossgl(code):
    ast = SlangParser(SlangLexer(code).tokenize()).parse()
    return SlangCrossGLCodeGen.SlangToCrossGLConverter().generate(ast)


def kernel(body, prelude=""):
    return f"""
    {prelude}
    RWStructuredBuffer<float> data;

    [shader("compute")]
    [numthreads(1, 1, 1)]
    void main(uint3 id : SV_DispatchThreadID) {{
        float s = data[id.x];
        {body}
    }}
    """


def test_mul_keeps_operand_order_with_transposed_constructor_and_subscripts():
    # The reproducer of the issue: mul(m, n) is CrossGL m * n once m and n are the same matrices.
    code = to_crossgl(kernel("""
        float2x2 m = float2x2(s, 2.0, 1.0, 3.0);
        float2x2 n = float2x2(1.0, 0.0, 2.0, 1.0);
        float2x2 p = mul(m, n);
        float2 v = mul(float2(1.0, 1.0), p);
        data[id.x] = p[0][1] + v.y;
        """))
    assert "mat2 m = mat2(s, 1.0, 2.0, 3.0);" in code
    assert "mat2 n = mat2(1.0, 2.0, 0.0, 1.0);" in code
    assert "mat2 p = (m * n);" in code
    assert "vec2 v = (vec2(1.0, 1.0) * p);" in code
    assert "data[id.x] = p[1][0] + v.y;" in code
    cgl_translator.parse(code)


@pytest.mark.parametrize(
    ("slang_type", "crossgl_type"),
    [
        ("float2x3", "mat3x2"),
        ("float3x2", "mat2x3"),
        ("float4x3", "mat3x4"),
        ("float3x3", "mat3"),
        ("double2x4", "dmat4x2"),
        ("matrix<float, 2, 3>", "mat3x2"),
        ("matrix<float, 4, 4>", "mat4"),
        ("int2x3", "int2x3"),
    ],
)
def test_matrix_types_are_spelled_column_by_row(slang_type, crossgl_type):
    code = to_crossgl(f"""
        struct Holder {{ {slang_type} member; }};
        void main() {{ {slang_type} held; }}
        """)
    assert f"{crossgl_type} member;" in code
    assert f"{crossgl_type} held;" in code


def test_constructors_fill_rows():
    code = to_crossgl(kernel("""
        float2x3 a = float2x3(s, 1.0, 2.0, 3.0, 4.0, 5.0);
        float2x3 b = float2x3(float3(1, 2, 3), float3(4, 5, 6));
        float2x2 c = float2x2(s);
        float3x2 d = (float3x2)a;
        float2x2 e = float2x2(float4(1.0, 2.0, 3.0, 4.0));
        data[id.x] = a[0][0] + b[0][0] + c[0][0] + d[0][0] + e[0][0];
        """))
    assert "mat3x2 a = mat3x2(s, 3.0, 1.0, 4.0, 2.0, 5.0);" in code
    assert "mat3x2 b = transpose(mat2x3(vec3(1, 2, 3), vec3(4, 5, 6)));" in code
    # A scalar fills every element, as in HLSL; CrossGL mat2(s) would be a diagonal.
    assert "mat2 c = mat2(vec2(s), vec2(s));" in code
    assert "mat2x3 d = mat2x3(a);" in code
    assert "mat2 e = transpose(mat2(vec4(1.0, 2.0, 3.0, 4.0)));" in code


def test_subscripts_and_swizzles_name_the_same_elements():
    code = to_crossgl(kernel("""
        float2x3 a = float2x3(s, 1.0, 2.0, 3.0, 4.0, 5.0);
        float3 row = a[1];
        float x = a[0].z + a._m12 + a._21;
        float2 y = a._m00_m11;
        data[id.x] = row.x + x + y.y + a[id.x % 2][1];
        """))
    assert "vec3 row = transpose(a)[1];" in code
    assert "float x = a[2][0] + a[2][1] + a[0][1];" in code
    assert "vec2 y = vec2(a[0][0], a[1][1]);" in code
    assert "a[1][id.x % 2]" in code


def test_row_writes_call_a_generated_setter():
    code = to_crossgl(kernel("""
        float2x3 a = float2x3(s, 1.0, 2.0, 3.0, 4.0, 5.0);
        a[0] = float3(s, s, s);
        a[1] += float3(1.0, 2.0, 3.0);
        a[1].y = 9.0;
        a[0][2] = 7.0;
        data[id.x] = a[1][1];
        """))
    assert (
        "vec3 cgl_slang_set_row_mat3x2(inout mat3x2 target, int row, vec3 value) {"
        in code
    )
    assert "target[2][row] = value.z;" in code
    assert "cgl_slang_set_row_mat3x2(a, int(0), vec3(s, s, s));" in code
    assert (
        "cgl_slang_set_row_mat3x2(a, int(1), vec3(transpose(a)[1] + (vec3(1.0, 2.0, 3.0))));"
        in code
    )
    assert "a[1][1] = 9.0;" in code
    assert "a[2][0] = 7.0;" in code
    # The setter is a shader-scope function ahead of every stage.
    assert code.index("cgl_slang_set_row_mat3x2(inout") < code.index("compute {")
    cgl_translator.parse(code)


def test_matrix_times_matrix_is_componentwise():
    code = to_crossgl(kernel("""
        float2x2 m = float2x2(s, 2.0, 1.0, 3.0);
        float2x2 n = float2x2(1.0, 0.0, 2.0, 1.0);
        float2x2 c = m * n;
        m *= n;
        m *= 2.0;
        float2x2 d = m * 2.0;
        data[id.x] = c[0][1] + d[0][0];
        """))
    assert "mat2 c = matrixCompMult(m, n);" in code
    assert "m = matrixCompMult(m, n);" in code
    assert "m *= 2.0;" in code
    assert "mat2 d = m * 2.0;" in code


def test_initializer_lists_fill_rows():
    code = to_crossgl(
        kernel(
            """
        float2x2 d = { 1.0, s, 3.0, 4.0 };
        float2x2 pair[2] = { { 1.0, 2.0, 3.0, 4.0 }, { 5.0, 6.0, 7.0, 8.0 } };
        float2x2 flat[2] = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
        data[id.x] = d[0][1] + pair[1][0][1] + flat[0][1][0];
        """,
            prelude="static const float2x3 K = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };",
        )
    )
    assert "mat2 d = mat2(1.0, 3.0, s, 4.0);" in code
    assert (
        "mat2 pair[2] = {mat2(1.0, 3.0, 2.0, 4.0), mat2(5.0, 7.0, 6.0, 8.0)};" in code
    )
    assert (
        "mat2 flat[2] = {mat2(1.0, 3.0, 2.0, 4.0), mat2(5.0, 7.0, 6.0, 8.0)};" in code
    )
    assert "static const mat3x2 K = mat3x2(1.0, 4.0, 2.0, 5.0, 3.0, 6.0);" in code
    assert "d[1][0] + pair[1][1][0] + flat[0][0][1]" in code


def test_members_buffers_constants_and_layout_qualifiers():
    code = to_crossgl("""
        struct S { float2x3 a; row_major float2x3 b; float3x3 c[2]; };
        cbuffer Params { float3x2 transform; };
        RWStructuredBuffer<float> data;
        StructuredBuffer<S> sb;

        float2x3 build(float s) { return float2x3(s, 0, 0, 0, s, 0); }

        [shader("compute")]
        [numthreads(1, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            S s = sb[0];
            data[id.x] = s.a[0][1] + sb[0].b[1][2] + s.c[1][0][2]
                + transform[2][1] + build(1.0)[0][1] + S(sb[0]).a[1][0];
        }
        """)
    assert "@ row_major\n        mat3x2 b;" in code
    assert "mat3 c[2];" in code
    assert "mat2x3 transform;" in code
    assert "s.a[1][0]" in code
    assert "sb[0].b[2][1]" in code
    assert "s.c[1][2][0]" in code
    assert "transform[1][2]" in code
    assert "build(1.0)[1][0]" in code


def test_let_declarations_take_the_matrix_type_of_their_initializer():
    code = to_crossgl(kernel("""
        let m = float2x3(s, 1.0, 2.0, 3.0, 4.0, 5.0);
        data[id.x] = m[0][2];
        """))
    assert "m[2][0]" in code


def test_translated_matrix_kernel_reaches_every_graphics_target(tmp_path):
    source = tmp_path / "matrices.slang"
    source.write_text(kernel("""
        float2x3 a = float2x3(s, 1.0, 2.0, 3.0, 4.0, 5.0);
        float3x2 b = float3x2(float2(1.0, s), float2(2.0, 3.0), float2(4.0, 5.0));
        float2x2 ab = mul(a, b);
        a[0] = float3(s, s, s);
        float2x2 c = ab * ab;
        data[id.x] = ab[1][0] + c._m01 + mul(float2(1.0, 2.0), a).z;
        """))
    for backend in ("opengl", "directx", "metal", "vulkan"):
        assert translate(str(source), backend=backend)


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            "float2x3 a = float2x3(s, 1, 2, 3, 4, 5); a[0].xy = float2(1, 2);",
            "multi-component matrix swizzle .xy",
        ),
        (
            "float2x3 a = float2x3(s, 1, 2, 3, 4, 5); a._m00_m01 = float2(1, 2);",
            "multi-component matrix swizzle ._m00_m01",
        ),
        (
            "float2x3 a = float2x3(s, 1, 2, 3, 4, 5); data[id.x] = a._m22;",
            "swizzle ._m22 is outside float2x3",
        ),
        (
            "float2x2 m = float2x2(unknown());",
            "value whose type the importer cannot determine",
        ),
        (
            "float2x2 m = float2x2(f(s));",
            "conversion of an expression with side effects",
        ),
        (
            "float2x2 m = float2x2(s, 1, 2, 3); m[0]++;",
            "increment or decrement of a matrix row",
        ),
        (
            "float2x2 m = { 1.0, 2.0, 3.0 };",
            "component count the importer cannot match",
        ),
        (
            "Holder h = { { 1.0, 2.0, 3.0, 4.0 } };",
            "initializer list for Holder, which has a matrix member",
        ),
    ],
)
def test_matrix_constructs_without_a_faithful_form_are_refused(body, reason):
    prelude = "struct Holder { float2x2 m; }; float f(float x) { return x; }"
    with pytest.raises(SlangImportRefused, match=reason):
        to_crossgl(kernel(body, prelude=prelude))
