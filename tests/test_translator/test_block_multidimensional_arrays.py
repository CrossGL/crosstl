"""Multidimensional arrays in std140 uniform blocks and HLSL constant buffers.

The block layout planners refused every array of arrays, so a GLSL uniform block with ``float a[2][3]`` could not
reach HLSL or even GLSL output, and an HLSL ``cbuffer`` with one could not reach GLSL or SPIR-V output. The layout
rules already nest (GLSL 4.60 section 7.6.2.2 rules 4 and 10; HLSL packing starts every array element on a
register), so the planners now accept them. The spelling ``float[2][3]`` lists its outermost dimension first:
two arrays of three floats.

Metal declares such a member with every extent, ``float4 a[2][3]``, because each layout's outer strides are the
inner extent times the innermost stride. The innermost element keeps its one-dimensional storage, so an element
read takes every subscript before its suffix or conversion applies (``params.a[i][j].x``), and register words
shared with the next member are read through the flattened index ``(i) * 3 + (j)``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.array_utils import split_outer_array_dimension
from crosstl.translator.codegen.glsl_block_layout import crossgl_layout_type
from tests.spirv_friendly_names import friendly_spirv

GLSL_SOURCE = """#version 450
layout(local_size_x = 64) in;
struct S { vec3 p; float w; vec2 q; };
layout(std430, binding = 1) writeonly buffer Outp { float o[]; };
layout(std140, binding = 2) uniform Params {
    float a[2][3];
    float t0;
    S s[2][2];
    mat3 m[2][2];
    ivec3 iv[3][1];
    float t1;
} params;
void main() {
    uint i = gl_GlobalInvocationID.x;
    o[4u * i + 0u] = params.a[i % 2u][i % 3u] + params.t0;
    o[4u * i + 1u] = params.s[i % 2u][(i / 2u) % 2u].p[i % 3u] + params.s[1][0].w;
    o[4u * i + 2u] = params.m[i % 2u][1][(i / 4u) % 3u][i % 3u] + params.t1;
    o[4u * i + 3u] = float(params.iv[i % 3u][0][i % 3u]);
}
"""

HLSL_SOURCE = """cbuffer Params : register(b2) {
    float a[2][3];
    float t0;
    float2 v[3][2];
    float t1;
};
RWStructuredBuffer<float> o : register(u1);
[numthreads(64, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x;
    o[4 * i + 0] = a[i % 2u][i % 3u] + t0;
    o[4 * i + 1] = v[i % 3u][(i / 3u) % 2u][i % 2u] + t1;
}
"""


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"blocks{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("float[2][3]", ("float[3]", "2")),
        ("vec2 [3][2]", ("vec2[2]", "3")),
        ("S[2][ N ][3]", ("S[N][3]", "2")),
        ("float[4]", ("float", "4")),
        ("float[]", ("float", "")),
        ("float", None),
    ],
)
def test_the_outermost_dimension_is_written_first(spelling, expected):
    assert split_outer_array_dimension(spelling) == expected


def test_std140_nests_array_strides():
    layout_type = crossgl_layout_type("float[2][3]", lambda _name: None)
    assert (layout_type.count, layout_type.element.count) == (2, 3)
    assert layout_type.element.element.kind == "scalar"


def test_glsl_block_reaches_hlsl_with_register_offsets(tmp_path):
    code = _translate(tmp_path, GLSL_SOURCE, ".comp", "directx")
    # std140: a takes 6 slots, t0 one, s[2][2] two per element, mat3 three columns per matrix, iv three slots.
    for declaration in (
        "float a[2][3] : packoffset(c0);",
        "float t0 : packoffset(c6);",
        "S s[2][2] : packoffset(c7);",
        "float3x3 m[2][2] : packoffset(c15);",
        "int3 iv[3][1] : packoffset(c27);",
        "float t1 : packoffset(c30);",
    ):
        assert declaration in code, code


def test_glsl_block_reaches_glsl_unchanged(tmp_path):
    code = _translate(tmp_path, GLSL_SOURCE, ".comp", "opengl")
    block = code[code.index("uniform Params") :]
    block = block[: block.index("}")]
    for member in ("float a[2][3];", "S s[2][2];", "mat3 m[2][2];", "ivec3 iv[3][1];"):
        assert member in block, block


def test_hlsl_cbuffer_reaches_glsl_as_registers(tmp_path):
    code = _translate(tmp_path, HLSL_SOURCE, ".hlsl", "opengl")
    # HLSL packing: a in c0-c5 with t0 in c5.y, v in c6-c11 with t1 in c11.z.
    assert "uvec4 crossgl_cbuffer_registers[12];" in code, code


def test_hlsl_cbuffer_reaches_spirv_with_nested_strides(tmp_path):
    path = tmp_path / "blocks.hlsl"
    path.write_text(HLSL_SOURCE, encoding="utf-8")
    code = friendly_spirv(
        crosstl.translate(str(path), backend="vulkan", format_output=False)
    )
    strides = set(re.findall(r"OpDecorate %(\w+) ArrayStride (\d+)", code))
    assert {stride for _name, stride in strides} >= {"16", "32", "48"}
    offsets = re.findall(r"OpMemberDecorate %Params (\d) Offset (\d+)", code)
    assert offsets == [("0", "0"), ("1", "84"), ("2", "96"), ("3", "184")], offsets


ROW_SOURCE = """#version 450
layout(local_size_x = 1) in;
layout(std140, binding = 0) uniform Params {
    vec3 a[2][3];
    float f[2][2];
    mat2 m[2][2];
    bool b[2][2];
    vec2 g;
};
layout(std430, binding = 1) buffer Data { float o[]; };
float pick(vec3 v[3], int k) { return v[k].y; }
void main() {
    int i = int(g.x) % 2;
    o[0] = a[i][1].z + f[1][i] + m[1][i][1].y + (b[i][1] ? 1.0 : 0.0);
    vec3 row[3] = a[i];
    float all[2][2] = f;
    o[1] = row[2].x + pick(a[1], 2) + all[1][0];
}
"""

STRUCTURED_SOURCE = """struct E {
    float3 v[2][2];
    bool b[2][3];
};
RWStructuredBuffer<E> e : register(u0);
RWStructuredBuffer<float> o : register(u1);
[numthreads(1, 1, 1)] void CSMain(uint3 id : SV_DispatchThreadID) {
    uint i = id.x % 2u;
    o[0] = e[0].v[1][i].y + (e[0].b[i][2] ? 1.0 : 0.0);
    e[1].b[1][i] = o[0] > 0.0;
}
"""


def test_metal_declares_std140_arrays_with_every_extent(tmp_path):
    code = _translate(tmp_path, GLSL_SOURCE, ".comp", "metal")
    params = code[code.index("struct Params {") :]
    params = params[: params.index("};")]
    for member in (
        "float4 a[2][3];",
        "S s[2][2];",
        "float3x3 m[2][2];",
        "int4 iv[3][1];",
    ):
        assert member in params, params
    # The element suffix and the read conversion follow the last subscript.
    assert "params.a[i % 2u][i % 3u].x + params.t0" in code, code
    assert "float3(params.s[i % 2u][i / 2u % 2u].p)[i % 3u]" in code, code
    assert "float(params.iv[i % 3u][0].xyz[i % 3u])" in code, code


def test_metal_reads_shared_hlsl_registers_by_flattened_index(tmp_path):
    code = _translate(tmp_path, HLSL_SOURCE, ".hlsl", "metal")
    # t0 shares the last register of a, and t1 that of v, so both arrays are held as register words.
    assert "crosstl_hlsl_cbuffer_float_array6 a;" in code, code
    assert "crosstl_hlsl_cbuffer_float2_array6 v;" in code, code
    assert (
        "crosstl_hlsl_cbuffer_float_array6_element(params.a, (i % 2u) * 3 + (i % 3u))"
        in code
    ), code
    assert "params.v, (i % 3u) * 2 + (i / 3u % 2u))[i % 2u]" in code, code


def test_metal_copies_inner_arrays_element_by_element(tmp_path):
    code = _translate(tmp_path, ROW_SOURCE, ".comp", "metal")
    assert "float4 a[2][3];" in code and "uint4 b[2][2];" in code, code
    assert "crosstl_hlsl_cbuffer_float2x2_padded m[2][2];" in code, code
    assert "params.a[i][1].xyz.z + params.f[1][i].x" in code, code
    assert "crosstl_hlsl_cbuffer_float2x2_padded_load(params.m[1][i])[1].y" in code
    assert "bool(params.b[i][1].x)" in code, code
    assert (
        "float3 row[3] = {params.a[i][0].xyz, params.a[i][1].xyz, params.a[i][2].xyz};"
        in code
    ), code
    assert "float all[2][2] = {{params.f[0][0].x, params.f[0][1].x}," in code, code
    assert "crosstl_array_value<float3, 3>{" in code, code
    assert "{params.a[1][0].xyz, params.a[1][1].xyz, params.a[1][2].xyz}}" in code
    # No inner array of storage elements is used as a value.
    assert not re.search(r"params\.[afb]\[[^\]]+\][^\[]", code), code


def test_metal_converts_nested_arrays_of_structured_buffer_elements(tmp_path):
    code = _translate(tmp_path, STRUCTURED_SOURCE, ".hlsl", "metal")
    assert "packed_float3 v[2][2];" in code and "uint b[2][3];" in code, code
    assert "float3(e[0].v[1][i]).y + (bool(e[0].b[i][2]) ? 1.0 : 0.0)" in code, code
    assert "e[1].b[1][i] = o[0] > 0.0;" in code, code


@pytest.mark.parametrize(
    ("source", "suffix"),
    [
        (GLSL_SOURCE, ".comp"),
        (HLSL_SOURCE, ".hlsl"),
        (ROW_SOURCE, ".comp"),
        (STRUCTURED_SOURCE, ".hlsl"),
    ],
)
def test_xcrun_metal_compiles_nested_buffer_arrays(tmp_path, source, suffix):
    xcrun = shutil.which("xcrun")
    if xcrun is None:
        pytest.skip("xcrun is not available")
    probe = subprocess.run(
        [xcrun, "-sdk", "macosx", "-f", "metal"], capture_output=True, timeout=30
    )
    if probe.returncode != 0:
        pytest.skip("xcrun cannot locate metal")
    shader = tmp_path / "blocks.metal"
    shader.write_text(_translate(tmp_path, source, suffix, "metal"), encoding="utf-8")
    result = subprocess.run(
        [xcrun, "-sdk", "macosx", "metal", "-c", str(shader)]
        + ["-o", str(tmp_path / "blocks.air")],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("dxc") is None, reason="dxc is not available")
def test_dxc_compiles_the_hlsl_block(tmp_path):
    shader = tmp_path / "blocks.hlsl"
    shader.write_text(
        _translate(tmp_path, GLSL_SOURCE, ".comp", "directx"), encoding="utf-8"
    )
    result = subprocess.run(
        [shutil.which("dxc"), "-T", "cs_6_0", "-E", "CSMain", str(shader)]
        + ["-Fo", str(tmp_path / "blocks.dxil")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
