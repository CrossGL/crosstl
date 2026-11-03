"""WGSL output writes a CrossGL array constructor as a WGSL array value constructor wherever it stands."""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.wgsl_struct_initializers import (
    WGSLStructInitializerError,
)

PROGRAM = """shader Probe {{
    const int COUNT = 3;
    struct Pair {{ float a; float b; }};
    buffer float values[];
{functions}
    compute {{
        layout(local_size_x = 1) in;
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            float x = values[i];
{body}
        }}
    }}
}}
"""


def translate(tmp_path, body, functions=""):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM.format(functions=functions, body=body))
    output = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(output)
        checked = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr
    return output


def test_returned_array_constructor(tmp_path):
    output = translate(
        tmp_path,
        "float a[3] = spread(x);\n values[i] = a[0] + a[1] + a[2];",
        functions="""
    float[3] spread(float x) {
        return float[3](x, x * 2.0, x + 1.0);
    }""",
    )
    assert "return array<f32, 3>(x, (x * 2.0), (x + 1.0));" in output
    assert "float[3]" not in output


def test_subscripted_array_constructor(tmp_path):
    output = translate(tmp_path, "values[i] = float[3](x, 1.0, 2.0)[1];")
    assert "values[i] = array<f32, 3>(x, 1.0, 2.0)[1];" in output


def test_assigned_unsized_array_constructor(tmp_path):
    output = translate(
        tmp_path,
        "float b[3] = float[3](x, x, x);\n b = float[](x, 3.0, 4.0);\n values[i] = b[2];",
    )
    assert "b = array<f32, 3>(x, 3.0, 4.0);" in output
    assert "float[]" not in output


def test_element_type_is_spelled_out(tmp_path):
    # An inferred array(1u, 2u) works, but array(1, 2) would be array<i32, 2>; the element type is always written.
    output = translate(tmp_path, "values[i] = float(uint[2](1u, 2u)[1]);")
    assert "array<u32, 2>(1u, 2u)[1]" in output


def test_vector_and_struct_elements(tmp_path):
    output = translate(
        tmp_path,
        "vec3 v = vec3[2](vec3(x), vec3(1.0))[0];\n"
        " Pair p = Pair[1](Pair(x, 2.0))[0];\n"
        " values[i] = v.x + p.b;",
    )
    assert "array<vec3<f32>, 2>(vec3<f32>(x), vec3<f32>(1.0))[0]" in output
    assert "array<Pair, 1>(Pair(x, 2.0))[0]" in output


def test_constant_size(tmp_path):
    output = translate(tmp_path, "values[i] = float[COUNT](x, 1.0, 2.0)[2];")
    assert "array<f32, COUNT>(x, 1.0, 2.0)[2]" in output


def test_array_constructor_argument(tmp_path):
    output = translate(
        tmp_path,
        "values[i] = first(float[2](x, 1.0));",
        functions="""
    float first(float a[2]) {
        return a[0];
    }""",
    )
    assert "first(array<f32, 2>(x, 1.0))" in output


def test_size_mismatch_is_refused(tmp_path):
    with pytest.raises(Exception, match="needs exactly 3"):
        translate(tmp_path, "values[i] = float[3](x, 1.0)[1];")


def test_one_argument_constructor_in_declaration_zero_fills(tmp_path):
    # WGSL array(0.0) is a one-element array, which a declared array<f32, 3> rejects (#3563).
    output = translate(
        tmp_path,
        "float a[3] = float[3](0.0);\n"
        " vec4 v[2] = vec4[2](0.0);\n"
        " values[i] = a[2] + v[1].x;",
    )
    assert "var a: array<f32, 3> = array<f32, 3>(0.0, f32(), f32());" in output
    assert (
        "var v: array<vec4<f32>, 2> = "
        "array<vec4<f32>, 2>(vec4<f32>(0.0), vec4<f32>());"
    ) in output
    assert "array(0.0)" not in output


def test_short_matrix_array_initializer_converts_scalars_to_diagonal_matrices(
    tmp_path,
):
    output = translate(
        tmp_path,
        "mat3 m[2] = mat3[2](0.0);\n"
        " mat4 n[3] = mat4[3](1.0);\n"
        " values[i] = m[1][2][0] + n[0][0][0];",
    )
    assert (
        "array<mat3x3<f32>, 2>("
        "mat3x3<f32>(vec3<f32>(0.0, 0.0, 0.0), "
        "vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.0)), "
        "mat3x3<f32>())"
    ) in output
    assert (
        "array<mat4x4<f32>, 3>("
        "mat4x4<f32>(vec4<f32>(1.0, 0.0, 0.0, 0.0), "
        "vec4<f32>(0.0, 1.0, 0.0, 0.0), vec4<f32>(0.0, 0.0, 1.0, 0.0), "
        "vec4<f32>(0.0, 0.0, 0.0, 1.0)), mat4x4<f32>(), mat4x4<f32>())"
    ) in output


def test_matrix_array_value_converts_scalar_elements(tmp_path):
    output = translate(
        tmp_path,
        "mat2 m = mat2[2](0.0, 1.0)[1];\n values[i] = m[0][0];",
    )
    assert (
        "array<mat2x2<f32>, 2>("
        "mat2x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0)), "
        "mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0)))"
    ) in output


def test_full_matrix_array_initializer_converts_scalar_elements(tmp_path):
    output = translate(
        tmp_path,
        "mat2 m[2] = mat2[2](x, 1.0);\n"
        " mat2 n[] = mat2[](2.0, mat2(x));\n"
        " mat2 p[2] = {x, 0.0};\n"
        " values[i] = m[0][1][1] + n[1][0][0] + p[0][1][1];",
    )
    diagonal_x = "mat2x2<f32>(vec2<f32>(x, 0.0), vec2<f32>(0.0, x))"
    assert (
        f"var m: array<mat2x2<f32>, 2> = array({diagonal_x}, "
        "mat2x2<f32>(vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0)));"
    ) in output
    assert (
        "array(mat2x2<f32>(vec2<f32>(2.0, 0.0), vec2<f32>(0.0, 2.0)), "
        f"{diagonal_x});"
    ) in output
    assert (
        f"var p: array<mat2x2<f32>, 2> = array({diagonal_x}, "
        "mat2x2<f32>(vec2<f32>(0.0, 0.0), vec2<f32>(0.0, 0.0)));"
    ) in output


def test_scalar_matrix_operand_outside_an_array_is_kept(tmp_path):
    # Only an array element reads a scalar as mat3(s); m *= s scales every component and stays as written.
    output = translate(
        tmp_path,
        "mat3 m = mat3(x);\n m *= 2.0;\n m *= twice(x);\n values[i] = m[0][0];",
        functions="""
    float twice(float v) {
        return v * 2.0;
    }""",
    )
    assert "m *= 2.0;" in output
    assert "m *= twice(x);" in output


def test_short_braced_initializer_zero_fills_the_rest(tmp_path):
    # The elements a braced initializer leaves out are zero, as in GLSL output.
    output = translate(
        tmp_path, "float a[3] = {x, 1.0};\n values[i] = a[0] + a[1] + a[2];"
    )
    assert "var a: array<f32, 3> = array<f32, 3>(x, 1.0, f32());" in output


def test_struct_braced_initializer_is_the_struct_constructor(tmp_path):
    output = translate(tmp_path, "Pair p = {x, 2.0};\n values[i] = p.a + p.b;")
    assert "var p: Pair = Pair(x, 2.0);" in output
    assert "array(x, 2.0)" not in output


def test_struct_braced_initializer_with_an_array_member(tmp_path):
    output = translate(
        tmp_path,
        "Frame f = {mat3[2](1.0, x), 2.0};\n values[i] = f.w;",
        functions="""
    struct Frame { mat3 m[2]; float w; };""",
    )
    assert "var f: Frame = Frame(array<mat3x3<f32>, 2>(" in output
    assert "), 2.0);" in output
    assert "= array(array" not in output


def test_struct_braced_initializer_converts_values_to_member_types(tmp_path):
    # WGSL has no implicit conversions, so each value is converted as it would be in Pair(x, k).
    output = translate(
        tmp_path,
        "ivec3 n = ivec3(1);\n Q q = {1, x, 2, n};\n values[i] = q.f + q.v.x;",
        functions="""
    struct Q { uint u; int s; float f; vec3 v; };""",
    )
    assert "var q: Q = Q(u32(1), i32(x), f32(2), vec3<f32>(n));" in output


def test_short_struct_braced_initializer_zero_fills_the_rest(tmp_path):
    # The members a braced initializer leaves out are zero, as in GLSL output.
    output = translate(
        tmp_path, "Pair p = {x};\n Pair z = {};\n values[i] = p.a + z.b;"
    )
    assert "var p: Pair = Pair(x, f32());" in output
    assert "var z: Pair = Pair(f32(), f32());" in output


def test_nested_and_assigned_struct_braced_initializers(tmp_path):
    output = translate(
        tmp_path,
        "Outer o = {{x}, 3};\n Pair p;\n p = {x, 2.0};\n values[i] = o.p.a + p.b;",
        functions="""
    struct Outer { Pair p; int n; };""",
    )
    assert "var o: Outer = Outer(Pair(x, f32()), 3);" in output
    assert "p = Pair(x, 2.0);" in output


def test_array_of_structs_braced_initializer(tmp_path):
    output = translate(
        tmp_path,
        "Pair ps[2] = {{x, 1}, {2.0, x}};\n"
        " Pair qs[COUNT] = {{x, 1.0}};\n"
        " values[i] = ps[1].a + qs[0].b;",
    )
    assert (
        "var ps: array<Pair, 2> = array<Pair, 2>(Pair(x, f32(1)), Pair(2.0, x));"
        in output
    )
    assert (
        "var qs: array<Pair, COUNT> = array<Pair, 3>(Pair(x, 1.0), Pair(), Pair());"
        in output
    )


def test_empty_array_of_structs_braced_initializer_is_zero(tmp_path):
    output = translate(tmp_path, "Pair ps[2] = {};\n values[i] = ps[1].a + x;")
    assert "var ps: array<Pair, 2> = array<Pair, 2>(Pair(), Pair());" in output


def test_struct_named_like_an_atomic_is_constructed(tmp_path):
    output = translate(
        tmp_path,
        "atomic_stats s = {x};\n values[i] = s.w;",
        functions="""
    struct atomic_stats { float w; };""",
    )
    assert "var s: atomic_stats = atomic_stats(x);" in output


def test_struct_braced_initializer_with_too_many_values_is_refused(tmp_path):
    with pytest.raises(WGSLStructInitializerError, match="gives 3 values") as raised:
        translate(tmp_path, "Pair p = {x, 1.0, 2.0};\n values[i] = p.a;")
    assert raised.value.code == "project.translate.wgsl-struct-initializer-unsupported"


def test_module_scope_struct_braced_initializers(tmp_path):
    output = translate(
        tmp_path,
        "values[i] = G.a + GS[2].b + g.a + x;",
        functions="""
    const Pair G = {1.0, 2};
    const Pair GS[3] = {{1.0, 2.0}};
    Pair g = {x0, 2.0};
    const float x0 = 1.0;""",
    )
    assert "const G: Pair = Pair(1.0, f32(2));" in output
    assert (
        "const GS: array<Pair, 3> = array<Pair, 3>(Pair(1.0, 2.0), Pair(), Pair());"
        in output
    )
    assert "var<private> g: Pair = Pair(x0, 2.0);" in output


@pytest.mark.parametrize(
    "struct, reason",
    [
        ("struct A { atomic_uint c; float w; };", "holds an atomic"),
        ("struct A { float w; texture2D t; };", "holds a resource"),
    ],
)
def test_struct_braced_initializer_of_an_unconstructible_struct_is_refused(
    tmp_path, struct, reason
):
    with pytest.raises(WGSLStructInitializerError, match=reason) as raised:
        translate(
            tmp_path, "A s = {x};\n values[i] = s.w;", functions=f"\n    {struct}"
        )
    assert raised.value.code == "project.translate.wgsl-struct-initializer-unsupported"
