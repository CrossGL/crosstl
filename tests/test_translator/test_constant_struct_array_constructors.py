"""An array constructor inside a shader-scope constant structure's brace list becomes a list (#3678).

Metal and HLSL build arrays only in initializer lists. ``const W C = {float[2](1.0, 2.0), 3.0};``
used to reach Metal output as ``{float[2](1.0, 2.0), 3.0}`` and HLSL output as a
``__crossgl_construct_W(float[2](1.0, 2.0), 3.0)`` call, which both compilers reject.
"""

import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader Probe {
        buffer float* outp;
        struct W { %s m[2]; float s; };
        const W C = {%s, 3.0};
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = C.s;
            }
        }
    }
""")


def translate(tmp_path, backend, member, constructor):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % (member, constructor))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [("metal", "constant W C = "), ("directx", "static const W C = ")],
)
def test_float_array_member_is_a_list(tmp_path, backend, declaration):
    output = translate(tmp_path, backend, "float", "float[2](1.0, 2.0)")
    assert f"{declaration}{{{{1.0, 2.0}}, 3.0}};" in output


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [("metal", "constant W C = "), ("directx", "static const W C = ")],
)
def test_matrix_array_member_is_a_list(tmp_path, backend, declaration):
    output = translate(tmp_path, backend, "mat3", "mat3[2](mat3(1.0), mat3(2.0))")
    assert f"{declaration}{{{{float3x3(" in output
    assert "mat3[2](" not in output
    assert "__crossgl_construct_W" not in output


def test_metal_one_argument_matrix_array_member_builds_a_matrix(tmp_path):
    output = translate(tmp_path, "metal", "mat3", "mat3[2](1.0)")
    assert "constant W C = {{float3x3(1.0)}, 3.0};" in output
