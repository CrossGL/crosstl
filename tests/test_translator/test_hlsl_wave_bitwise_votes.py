"""HLSL output of subgroup bitwise reductions on signed integers and of whole-value equality votes.

DXC declares ``WaveActiveBitAnd``, ``WaveActiveBitOr`` and ``WaveActiveBitXor`` for ``uint`` and ``uint64_t``
scalars and vectors only, so a signed or 16-bit integer operand reduces as an unsigned integer and converts back.
GLSL's ``subgroupAllEqual`` compares a whole vector and returns one ``bool``, while HLSL's ``WaveActiveAllEqual``
compares each component, so a vector vote in a scalar ``bool`` context is ``all(WaveActiveAllEqual(...))``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

HEADER = """#version 450
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_vote : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
layout(local_size_x = 64) in;
layout(std430, binding = 0) buffer B { int fin[]; };
layout(std430, binding = 1) buffer O { int fout[]; };
void main() {
    uint i = gl_GlobalInvocationID.x;
"""


def _hlsl(tmp_path, body):
    path = tmp_path / "probe.comp"
    path.write_text(HEADER + body + "\n}\n", encoding="utf-8")
    return crosstl.translate(str(path), backend="directx", format_output=False)


def _flat(code):
    return re.sub(r"\s+", " ", code)


def _dxc(tmp_path, code):
    dxc = shutil.which("dxc")
    if dxc is None:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "probe.hlsl"
    path.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [dxc, "-T", "cs_6_6", "-E", "CSMain", "-enable-16bit-types", str(path)]
        + ["-Fo", str(tmp_path / "probe.dxil")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "declaration, expected",
    [
        (
            "int r = subgroupAnd(fin[i]);",
            "((int)(WaveActiveBitAnd((uint)(",
        ),
        (
            "ivec3 r = subgroupXor(ivec3(fin[i]));",
            "((int3)(WaveActiveBitXor((uint3)(",
        ),
        (
            "int64_t r = subgroupOr(int64_t(fin[i]));",
            "((int64_t)(WaveActiveBitOr((uint64_t)(",
        ),
        (
            "i64vec2 r = subgroupAnd(i64vec2(fin[i]));",
            "((int64_t2)(WaveActiveBitAnd((uint64_t2)(",
        ),
        (
            "int16_t r = subgroupXor(int16_t(fin[i]));",
            "((int16_t)(WaveActiveBitXor((uint)(",
        ),
        (
            "u16vec2 r = subgroupOr(u16vec2(fin[i]));",
            "((uint16_t2)(WaveActiveBitOr((uint2)(",
        ),
    ],
)
def test_signed_and_16_bit_bitwise_reductions_use_the_unsigned_overload(
    tmp_path, declaration, expected
):
    read = "r" if declaration.split()[0] in {"int", "int64_t", "int16_t"} else "r.x"
    code = _hlsl(tmp_path, f"    {declaration}\n    fout[i] = int({read});")
    assert expected in _flat(code)
    _dxc(tmp_path, code)


def test_vector_bitwise_reduction_result_component_reads_compile(tmp_path):
    code = _hlsl(
        tmp_path,
        "    ivec2 r = subgroupOr(ivec2(fin[i], 1));\n    fout[i] = r.x + r.y;",
    )
    assert "((int2)(WaveActiveBitOr((uint2)(" in _flat(code)
    _dxc(tmp_path, code)


def test_unsigned_and_bool_bitwise_reductions_are_unchanged(tmp_path):
    code = _flat(
        _hlsl(
            tmp_path,
            "    uint u = subgroupAnd(uint(fin[i]));\n"
            "    bool x = subgroupXor(fin[i] > 2);\n"
            "    fout[i] = int(u) + (x ? 1 : 0);",
        )
    )
    assert "uint u = WaveActiveBitAnd(uint(" in code
    assert "WaveActiveCountBits(" in code
    assert "(uint)(" not in code


@pytest.mark.parametrize(
    "statement, expected",
    [
        (
            "bool e = subgroupAllEqual(vec3(fin[i]));",
            "bool e = all(WaveActiveAllEqual(float3(",
        ),
        (
            "bool e = !subgroupAllEqual(vec2(fin[i]));",
            "bool e = !all(WaveActiveAllEqual(float2(",
        ),
        (
            "bool e = subgroupAllEqual(uvec4(i)) && fin[i] > 0;",
            "bool e = (all(WaveActiveAllEqual(uint4(",
        ),
        (
            "bool e = fin[i] > 0 || subgroupAllEqual(ivec2(fin[i], 2));",
            "|| all(WaveActiveAllEqual(int2(",
        ),
        (
            "bool e = subgroupAllEqual(vec2(fin[i])) ? true : false;",
            "bool e = (all(WaveActiveAllEqual(float2(",
        ),
    ],
)
def test_vector_equality_vote_in_a_bool_context_compares_the_whole_value(
    tmp_path, statement, expected
):
    code = _hlsl(tmp_path, f"    {statement}\n    fout[i] = e ? 1 : 0;")
    assert expected in _flat(code)
    _dxc(tmp_path, code)


def test_vector_equality_vote_as_a_condition_compares_the_whole_value(tmp_path):
    code = _hlsl(
        tmp_path,
        "    int r = 0;\n"
        "    if (subgroupAllEqual(ivec2(fin[i], 2))) { r = 1; }\n"
        "    fout[i] = r;",
    )
    assert "if (all(WaveActiveAllEqual(int2(" in _flat(code)
    _dxc(tmp_path, code)


def test_scalar_equality_vote_keeps_the_bare_intrinsic(tmp_path):
    code = _flat(
        _hlsl(
            tmp_path, "    bool e = subgroupAllEqual(fin[i]);\n    fout[i] = e ? 1 : 0;"
        )
    )
    assert "bool e = WaveActiveAllEqual(" in code
    assert "all(" not in code


def test_hlsl_per_component_equality_vote_keeps_its_vector_result(tmp_path):
    path = tmp_path / "probe.hlsl"
    path.write_text(
        "RWStructuredBuffer<float3> b : register(u0);\n"
        "RWStructuredBuffer<uint> o : register(u1);\n"
        "[numthreads(64, 1, 1)] void main(uint3 id : SV_DispatchThreadID) {\n"
        "    bool3 r = WaveActiveAllEqual(b[id.x]);\n"
        "    o[id.x] = r.y ? 1u : 0u;\n"
        "}\n",
        encoding="utf-8",
    )
    code = _flat(crosstl.translate(str(path), backend="directx", format_output=False))
    assert "bool3 r = WaveActiveAllEqual(" in code
    assert "all(WaveActiveAllEqual" not in code
