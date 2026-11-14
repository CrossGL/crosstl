"""A matrix or array type argument of a generic call is specialized like any other (#3643).

The parser keeps a call's explicit type arguments in the callee name, as in ``pick<mat4>``. It used to write a
matrix argument as the Python repr of the parsed type and an array argument as ``vec`` followed by the size node, so
Rust output carried the repr and the other targets refused the call as one they could not specialize.
"""

import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader P {
        buffer float* outp;
        T pick<T>(T x) { return x; }
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                float x = outp[i];
                %s
            }
        }
    }
""")

MATRIX = "mat4 m = mat4(x); mat4 r = pick<mat4>(m); outp[i] = r[1][1];"
ARRAY = (
    "float a[2]; a[0] = x; a[1] = x; float c[2] = pick<float[2]>(a); outp[i] = c[1];"
)
NESTED_ARRAY = (
    "float a[2][3]; a[1][2] = x; "
    "float c[2][3] = pick<float[2][3]>(a); outp[i] = c[1][2];"
)


def translate(tmp_path, body, backend):
    path = tmp_path / "pick.cgl"
    path.write_text(PROGRAM % body)
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("backend", "specialization"),
    [
        ("directx", "float4x4 pick_mat4(float4x4 x)"),
        ("metal", "float4x4 pick_mat4(float4x4 x)"),
        ("opengl", "mat4 pick_mat4(mat4 x)"),
        ("slang", "float4x4 pick_mat4(float4x4 x)"),
        ("wgsl", "fn pick_mat4x4_f32(x: mat4x4<f32>) -> mat4x4<f32>"),
    ],
)
def test_matrix_type_argument_specializes_the_call(tmp_path, backend, specialization):
    assert specialization in translate(tmp_path, MATRIX, backend)


@pytest.mark.parametrize(
    ("backend", "specialization"),
    [
        ("opengl", "float[2] pick_float_2(float x[2])"),
        ("wgsl", "fn pick_array_f32_2(x: array<f32, 2>) -> array<f32, 2>"),
    ],
)
def test_array_type_argument_specializes_the_call(tmp_path, backend, specialization):
    assert specialization in translate(tmp_path, ARRAY, backend)


@pytest.mark.parametrize(
    ("body", "call"),
    [
        (MATRIX, "pick::<Mat4<f32>>("),
        (ARRAY, "pick::<[f32; 2]>("),
        (NESTED_ARRAY, "pick::<[[f32; 3]; 2]>("),
    ],
)
def test_rust_spells_the_type_argument_as_a_rust_type(tmp_path, body, call):
    assert call in translate(tmp_path, body, "rust")
