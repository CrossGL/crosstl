"""GLSL output declares a double-precision matrix with its ``dmat`` type (#3729).

``dmat3 m = dmat3(double(x));`` used to be written ``mat3 m = dmat3(double(x));``,
which glslang rejects: a double matrix cannot initialize a float one.
"""

import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader P {
        buffer float* outp;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                double d = double(outp[i]);
                %s
            }
        }
    }
""")


def translate(tmp_path, body):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % body)
    return crosstl.translate(str(path), backend="glsl", format_output=False)


@pytest.mark.parametrize(
    "matrix_type", ["dmat2", "dmat3", "dmat4", "dmat2x3", "dmat4x2"]
)
def test_double_matrix_local_keeps_its_double_type(tmp_path, matrix_type):
    output = translate(
        tmp_path,
        f"{matrix_type} m = {matrix_type}(d); outp[i] = float(m[1][1]);",
    )
    assert f"{matrix_type} m = {matrix_type}(d);" in output


def test_float_matrix_and_double_vector_locals_are_unchanged(tmp_path):
    output = translate(
        tmp_path,
        "mat3 f = mat3(outp[i]); mat2x3 g = mat2x3(outp[i]); dvec3 v = dvec3(d); "
        "outp[i] = f[0][0] + g[1][2] + float(v.x);",
    )
    assert "mat3 f = mat3(outp[i]);" in output
    assert "mat2x3 g = mat2x3(outp[i]);" in output
    assert "dvec3 v = dvec3(d);" in output
