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
