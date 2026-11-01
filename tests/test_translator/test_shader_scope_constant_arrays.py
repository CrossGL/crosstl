"""A shader-scope constant array initialized with an array constructor gets an initializer list.

A local declaration's array constructor becomes its initializer list, but a shader-scope
``const float G[3] = float[3](1.0, 2.0, 3.0);`` reached Metal and HLSL output unchanged, which
both compilers reject (#3633).
"""

import textwrap

import pytest

import crosstl

PROGRAM = textwrap.dedent("""
    shader Probe {
        buffer float* outp;
        const float G[3] = %s;
        const mat3 H[2] = %s;
        compute {
            layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
            void main() {
                uint i = gl_GlobalInvocationID.x;
                outp[i] = G[2] + H[1][1][1];
            }
        }
    }
""")


def translate(tmp_path, backend, g, h):
    path = tmp_path / "probe.cgl"
    path.write_text(PROGRAM % (g, h))
    return crosstl.translate(str(path), backend=backend, format_output=False)


FULL = ("float[3](1.0, 2.0, 3.0)", "mat3[2](mat3(1.0), mat3(2.0))")


def test_metal_constant_array_constructor_is_an_initializer_list(tmp_path):
    output = translate(tmp_path, "metal", *FULL)
    assert "constant float G[3] = {1.0, 2.0, 3.0};" in output
    assert "constant float3x3 H[2] = {float3x3(1.0), float3x3(2.0)};" in output


def test_hlsl_constant_array_constructor_is_an_initializer_list(tmp_path):
    output = translate(tmp_path, "directx", *FULL)
    assert "static const float G[3] = {1.0, 2.0, 3.0};" in output
    assert "static const float3x3 H[2] = {float3x3(" in output
    assert "mat3[2](" not in output


def test_metal_short_constant_array_constructor_is_zero_filled(tmp_path):
    output = translate(tmp_path, "metal", "float[3](1.0)", "mat3[2](1.0)")
    assert "constant float G[3] = {1.0};" in output
    assert "constant float3x3 H[2] = {float3x3(1.0)};" in output


def test_wgsl_short_constant_array_constructor_is_zero_filled(tmp_path):
    output = translate(tmp_path, "wgsl", "float[3](1.0)", "mat3[2](mat3(1.0))")
    assert "const G: array<f32, 3> = array<f32, 3>(1.0, f32(), f32());" in output
    assert "const H: array<mat3x3<f32>, 2> = array<mat3x3<f32>, 2>(" in output


@pytest.mark.parametrize("backend", ["opengl", "wgsl"])
def test_constant_array_constructor_still_translates(tmp_path, backend):
    output = translate(tmp_path, backend, *FULL)
    assert "G" in output and "H" in output
