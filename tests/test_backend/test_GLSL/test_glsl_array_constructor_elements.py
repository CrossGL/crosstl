# SPDX-License-Identifier: Apache-2.0
"""GLSL array constructors write each scalar as the element type's constructor.

An array constructor takes one value of the element type per element
(GLSL 4.60 section 5.4.4), and a scalar is not a vector or matrix, so
``mat2[2](1.0, 2.0)`` and ``vec3[2](1.0, 2.0)`` were rejected with
"not enough data provided for construction". Now each scalar is written as
the element type's constructor: ``mat2[2](mat2(1.0), mat2(2.0))`` and
``vec3[2](vec3(1.0), vec3(2.0))`` (#3636).
"""

import textwrap

import crosstl

PROGRAM = textwrap.dedent("""
    shader Mod {
        compute {
            layout(local_size_x = 8, local_size_y = 1, local_size_z = 1) in;
            buffer float* outp;
            void main() {
                %s
                outp[0] = 0.0;
            }
        }
    }
""")


def translate(tmp_path, statement):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % statement)
    return crosstl.translate(str(path), backend="glsl", format_output=False)


def test_matrix_elements_come_from_scalars(tmp_path):
    output = translate(tmp_path, "mat4 a[3] = mat4[3](1.0);")
    assert "mat4 a[3] = mat4[3](mat4(1.0), mat4(0.0), mat4(0.0));" in output


def test_full_matrix_constructor_converts_each_scalar(tmp_path):
    output = translate(tmp_path, "mat2 b[2] = mat2[2](1.0, 2.0);")
    assert "mat2 b[2] = mat2[2](mat2(1.0), mat2(2.0));" in output


def test_vector_elements_come_from_scalars(tmp_path):
    output = translate(tmp_path, "vec3 w[2] = vec3[2](1.0, 2.0);")
    assert "vec3 w[2] = vec3[2](vec3(1.0), vec3(2.0));" in output


def test_assignment_converts_scalar_elements_too(tmp_path):
    output = translate(tmp_path, "vec3 v[2]; v = vec3[2](1.0, 2.0);")
    assert "v = vec3[2](vec3(1.0), vec3(2.0));" in output


def test_assignment_pads_short_matrix_constructor(tmp_path):
    output = translate(tmp_path, "mat3 m[2]; m = mat3[2](0.0);")
    assert "m = mat3[2](mat3(0.0), mat3(0.0));" in output
