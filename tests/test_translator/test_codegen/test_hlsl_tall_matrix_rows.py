"""Dynamic row reads of tall HLSL matrices go through a parameter copy.

DXC lowers a dynamically indexed row of a matrix with more rows than columns (``float3x2``, ``float4x2``,
``float4x3``) held in a constant buffer, an array element or a structure member through a temporary sized by the
column count, and DXIL validation rejects the out-of-bounds store ("Access to out-of-bounds memory is
disallowed"). Such reads now take the matrix from ``crossgl_matrix_value_<type>(m)``, which returns its argument.
Written rows keep the direct form.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

SOURCE = """#version 450
layout(local_size_x = 64) in;
layout(std430, binding = 1) buffer Outp { float o[]; };
layout(std140, binding = 2) uniform Params { mat2x3 one; mat2x3 m[2]; mat3x4 b[2]; mat3 sq[2]; } params;
struct S { vec4 pad; mat2x3 q; };
void put(out float v) { v = 3.0; }
void main() {
    uint i = gl_GlobalInvocationID.x;
    mat2x3 l[2] = params.m;
    mat2x3 single = params.one;
    S s = S(vec4(1.0), params.one);
    l[i % 2u][(i / 2u) % 2u][i % 3u] = 7.0;
    l[(i + 1u) % 2u][1][i % 3u] += 2.0;
    l[0][0][i % 3u]++;
    put(l[1][(i / 4u) % 2u][i % 3u]);
    o[8u * i + 0u] = params.one[(i / 2u) % 2u][i % 3u];
    o[8u * i + 1u] = params.m[i % 2u][(i / 2u) % 2u][i % 3u];
    o[8u * i + 2u] = l[i % 2u][(i / 2u) % 2u][i % 3u] + l[0][0][(i + 1u) % 3u];
    o[8u * i + 3u] = s.q[(i / 2u) % 2u][i % 3u];
    o[8u * i + 4u] = params.b[i % 2u][i % 3u][i % 4u];
    o[8u * i + 5u] = params.m[i % 2u][1][2] + params.sq[i % 2u][1][i % 3u];
    o[8u * i + 6u] = single[1][i % 3u];
    o[8u * i + 7u] = params.m[1][i % 2u].z;
}
"""


@pytest.fixture(scope="module")
def hlsl(tmp_path_factory):
    path = tmp_path_factory.mktemp("tall") / "rows.comp"
    path.write_text(SOURCE, encoding="utf-8")
    return crosstl.translate(str(path), backend="directx")


def _statement(code, slot):
    return next(
        line.strip() for line in code.splitlines() if f"+ {slot}u) * 4)" in line
    )


def test_dynamic_row_reads_take_a_parameter_copy(hlsl):
    assert "float3x2 crossgl_matrix_value_float3x2(float3x2 value)" in hlsl
    assert "float4x3 crossgl_matrix_value_float4x3(float4x3 value)" in hlsl
    assert "crossgl_matrix_value_float3x2(one)[(i % 3u)]" in _statement(hlsl, 0)
    assert "crossgl_matrix_value_float3x2(m[(i % 2u)])[(i % 3u)]" in _statement(hlsl, 1)
    assert "crossgl_matrix_value_float3x2(l[(i % 2u)])[(i % 3u)]" in _statement(hlsl, 2)
    assert "crossgl_matrix_value_float3x2(s.q)[(i % 3u)]" in _statement(hlsl, 3)
    assert "crossgl_matrix_value_float4x3(b[(i % 2u)])[(i % 4u)]" in _statement(hlsl, 4)


def test_constant_rows_square_matrices_and_locals_keep_the_direct_form(hlsl):
    slot5 = _statement(hlsl, 5)
    assert "m[(i % 2u)][2][1]" in slot5 and "sq[(i % 2u)]" in slot5
    assert "crossgl_matrix_value_float3x3" not in hlsl
    assert "single[(i % 3u)][1]" in _statement(hlsl, 6)


def test_written_rows_keep_the_direct_form(hlsl):
    body = hlsl[hlsl.index("void CSMain") :]
    writes = body[: body.index("+ 0u) * 4)")]
    assert "crossgl_matrix_value" not in writes
    assert "l[(i % 2u)][(i % 3u)][((i / 2u) % 2u)] = 7.0;" in writes


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
def test_dxil_validation_accepts_the_reads(hlsl, tmp_path):
    shader = tmp_path / "rows.hlsl"
    shader.write_text(hlsl, encoding="utf-8")
    result = subprocess.run(
        [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)]
        + ["-Fo", str(tmp_path / "rows.dxil")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
