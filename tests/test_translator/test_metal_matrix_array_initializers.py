"""Metal output initializes an array of matrices from a one-argument array constructor.

``mat3[2](0.0)`` sets the first element to ``mat3(0.0)`` and zero-fills the rest, as
GLSL output writes it. Metal used to write ``{0.0}``, which the Metal compiler rejects
because a scalar cannot initialize a ``float3x3`` element (#3567).
"""

import textwrap

import crosstl

PROGRAM = textwrap.dedent("""
    shader P {
        compute {
            buffer float* outp;
            void main() {
                %s
                outp[0] = 0.0;
            }
        }
    }
""")


def translate(tmp_path, declaration):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % declaration)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def test_scalar_matrix_array_constructor_writes_matrix_element(tmp_path):
    output = translate(tmp_path, "mat3 a[2] = mat3[2](0.0);")
    assert "float3x3 a[2] = {float3x3(0.0)};" in output


def test_non_zero_scalar_builds_a_diagonal_matrix_element(tmp_path):
    output = translate(tmp_path, "mat2 a[3] = mat2[3](1.0);")
    assert "float2x2 a[3] = {float2x2(1.0)};" in output


def test_non_square_matrix_keeps_its_columns_and_rows(tmp_path):
    output = translate(tmp_path, "mat2x3 a[2] = mat2x3[2](1.0);")
    assert "float2x3 a[2] = {float2x3(1.0)};" in output


def test_scalar_array_constructor_is_unchanged(tmp_path):
    output = translate(tmp_path, "float a[3] = float[3](0.0); float b = a[1];")
    assert "float a[3] = {0.0};" in output
