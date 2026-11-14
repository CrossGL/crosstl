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


STRUCT_PROGRAM = textwrap.dedent("""
    shader P {
        buffer float* outp;
        struct W { mat3 m[2]; float s; };
        struct V { W w; float t; };
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                W w = {mat3[2](1.0), 2.0};
                V v = {{mat3[2](0.0), 3.0}, 4.0};
                outp[i] = w.m[0][0][0] + w.s + v.w.s + v.t;
            }
        }
    }
""")


def test_struct_brace_initializer_builds_matrix_array_member(tmp_path):
    """A matrix array member of a structure's brace list gets matrices (#3634)."""
    path = tmp_path / "struct.cgl"
    path.write_text(STRUCT_PROGRAM)
    output = crosstl.translate(str(path), backend="metal", format_output=False)
    assert "W w = {{float3x3(1.0)}, 2.0};" in output
    assert "V v = {{{float3x3(0.0)}, 3.0}, 4.0};" in output


STRUCT_ARRAY_PROGRAM = textwrap.dedent("""
    shader P {
        buffer float* outp;
        struct W { mat3 m[2]; float s; };
        struct U { W ws[2]; float t; };
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                W ws[2] = {{mat3[2](1.0), 2.0}, {mat3[2](mat3(2.0), mat3(3.0)), 4.0}};
                U u = {{{mat3[2](1.0), 2.0}, {mat3[2](0.0), 3.0}}, 5.0};
                outp[i] = ws[1].m[1][0][0] + u.ws[0].m[0][0][0] + u.t;
            }
        }
    }
""")


def test_array_of_structs_brace_initializer_builds_matrix_array_member(tmp_path):
    """Each structure of an array of structures' brace list gets its matrices too."""
    path = tmp_path / "structs.cgl"
    path.write_text(STRUCT_ARRAY_PROGRAM)
    output = crosstl.translate(str(path), backend="metal", format_output=False)
    assert (
        "W ws[2] = {{{float3x3(1.0)}, 2.0}, {{float3x3(2.0), float3x3(3.0)}, 4.0}};"
        in output
    )
    assert "U u = {{{{float3x3(1.0)}, 2.0}, {{float3x3(0.0)}, 3.0}}, 5.0};" in output
