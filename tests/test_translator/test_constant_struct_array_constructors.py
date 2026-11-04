"""An array constructor inside a shader-scope constant structure's brace list becomes a list (#3678).

Metal and HLSL build arrays only in initializer lists. ``const W C = {float[2](1.0, 2.0), 3.0};``
used to reach Metal output as ``{float[2](1.0, 2.0), 3.0}`` and HLSL output as a
``__crossgl_construct_W(float[2](1.0, 2.0), 3.0)`` call, which both compilers reject.
"""

import textwrap

import pytest

import crosstl
from crosstl.translator.semantics.array_values import ArrayConstructorUnsupportedError

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


SHADER = textwrap.dedent("""
    shader Probe {
        buffer float* outp;
        struct W { float m[2]; float s; };
        struct I { float a[2]; };
        struct O { I i; float s; };
        %s
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = %s;
            }
        }
    }
""")


def translate_constant(tmp_path, backend, declaration, use):
    path = tmp_path / "probe.cgl"
    path.write_text(SHADER % (declaration, use))
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("backend", "declaration", "initializer"),
    [
        ("metal", "constant O C = ", "{{{1.0, 2.0}}, 3.0}"),
        ("directx", "static const O C = ", "{{{1.0, 2.0}}, 3.0}"),
    ],
)
def test_nested_structure_array_member_is_a_list(
    tmp_path, backend, declaration, initializer
):
    output = translate_constant(
        tmp_path, backend, "const O C = {{float[2](1.0, 2.0)}, 3.0};", "C.i.a[1]"
    )
    assert declaration + initializer + ";" in output


@pytest.mark.parametrize(
    ("backend", "declaration", "initializer"),
    [
        ("metal", "constant W C = ", "W{{1.0, 2.0}, 3.0}"),
        ("directx", "static const W C = ", "{{1.0, 2.0}, 3.0}"),
    ],
)
def test_structure_constructor_array_member_is_a_list(
    tmp_path, backend, declaration, initializer
):
    output = translate_constant(
        tmp_path, backend, "const W C = W(float[2](1.0, 2.0), 3.0);", "C.m[1]"
    )
    assert declaration + initializer + ";" in output


@pytest.mark.parametrize(
    ("backend", "declaration"),
    [("metal", "constant W A[2] = "), ("directx", "static const W A[2] = ")],
)
def test_each_structure_of_a_constant_array_gets_lists(tmp_path, backend, declaration):
    output = translate_constant(
        tmp_path,
        backend,
        "const W A[2] = {{float[2](1.0, 2.0), 3.0}, {float[2](4.0, 5.0), 6.0}};",
        "A[1].m[1]",
    )
    assert declaration + "{{{1.0, 2.0}, 3.0}, {{4.0, 5.0}, 6.0}};" in output


@pytest.mark.parametrize("backend", ["metal", "directx", "vulkan"])
def test_array_constructor_that_is_not_a_member_is_refused(tmp_path, backend):
    # A shader-scope initializer has no statement to declare a local before, so an
    # array constructor that is not itself a member of the list is refused.
    with pytest.raises(ArrayConstructorUnsupportedError, match="constant 'C'"):
        translate_constant(
            tmp_path, backend, "const W C = {{float[2](1.0, 2.0)[1], 2.0}, 3.0};", "C.s"
        )
