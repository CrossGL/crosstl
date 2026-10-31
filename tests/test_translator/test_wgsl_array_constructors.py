"""WGSL output writes a CrossGL array constructor as a WGSL array value constructor wherever it stands."""

import shutil
import subprocess

import pytest

import crosstl

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


def test_short_braced_initializer_zero_fills_the_rest(tmp_path):
    # The elements a braced initializer leaves out are zero, as in GLSL output.
    output = translate(
        tmp_path, "float a[3] = {x, 1.0};\n values[i] = a[0] + a[1] + a[2];"
    )
    assert "var a: array<f32, 3> = array<f32, 3>(x, 1.0, f32());" in output
