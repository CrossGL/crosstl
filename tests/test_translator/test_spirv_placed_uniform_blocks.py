"""SPIR-V uniform blocks keep HLSL constant-buffer packing and explicit member offsets.

HLSL packs a constant buffer into 16-byte registers, and ``packoffset`` and GLSL ``layout(offset = N)`` place a
member at the byte they name. The SPIR-V emitter decorates such blocks with the planned offsets and strides and
declares ``OpSourceExtension "GL_EXT_scalar_block_layout"``, which the host must honour by enabling
``scalarBlockLayout``. The expected HLSL offsets follow the constant-buffer packing rules and are the offsets
DXC reflects for the same source compiled to DXIL; the GLSL ones are glslang's.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

HLSL_PACKED = """cbuffer Params : register(b0) {
    float a;
    float2 b;
    float c[2];
    float d;
    float3 e;
    float4x4 m;
    row_major float3x3 r;
    float t;
};
RWStructuredBuffer<float> o : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    float3 v = mul(r, e) + mul(m, float4(e, 1.0)).xyz;
    o[0] = a + b.y + c[1] + d + v.x + t;
}
"""
# a | b at 4 packs into the first register | c[0] and c[1] start registers 1 and 2 | d packs after c[1] |
# e would straddle 40..52, so it moves to 48 | m takes four registers from 64 | the three rows of r from 128 |
# t packs into the last row's fourth component.
HLSL_PACKED_OFFSETS = [0, 4, 16, 36, 48, 64, 128, 172]

HLSL_PACKOFFSET = """cbuffer Params : register(b0) {
    float4 tint : packoffset(c2);
    float scale : packoffset(c0.y);
    float2 uv : packoffset(c1.z);
};
RWStructuredBuffer<float> o : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[0] = tint.w * scale + uv.y;
}
"""
HLSL_PACKOFFSET_OFFSETS = [32, 4, 24]

HLSL_STD140_COMPATIBLE = """cbuffer Params : register(b0) {
    float4 tint;
    float4x4 m;
    float4 extra[2];
};
RWStructuredBuffer<float> o : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[0] = mul(m, tint).x + extra[1].y;
}
"""

HLSL_MATRIX_TAIL = """cbuffer Params : register(b0) {
    float3x3 rot;
    float k;
};
RWStructuredBuffer<float> o : register(u1);
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    o[0] = mul(rot, float3(1.0, 2.0, 3.0)).x + k;
}
"""

GLSL_EXPLICIT_OFFSETS = """#version 450
layout(local_size_x = 1) in;
layout(std140, binding = 0) uniform Params {
    layout(offset = 0) float a;
    layout(offset = 20) float b;
    layout(offset = 32) vec4 c;
    layout(align = 32) vec2 d;
} params;
layout(std430, binding = 1) buffer Out { float o[]; };
void main() {
    o[0] = params.a + params.b + params.c.w + params.d.y;
}
"""
GLSL_EXPLICIT_OFFSETS_OFFSETS = [0, 20, 32, 64]


def _translate(tmp_path, source, name, source_backend):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend="vulkan", source_backend=source_backend
    )


def _member_offsets(code, struct):
    offsets = {
        int(match.group(1)): int(match.group(2))
        for match in re.finditer(rf"OpMemberDecorate %{struct} (\d+) Offset (\d+)", code)
    }
    return [offsets[index] for index in sorted(offsets)]


def _member_decorations(code, struct, index):
    return re.findall(rf"OpMemberDecorate %{struct} {index} (\w+(?: \d+)?)", code)


def _validate(tmp_path, code, *flags):
    """``spirv-val``'s verdict on ``code`` with ``flags``, or ``None`` without the tools."""
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if assembler is None or validator is None:
        return None
    asm = tmp_path / "shader.spvasm"
    asm.write_text(code, encoding="utf-8")
    binary = tmp_path / "shader.spv"
    assembled = subprocess.run(
        [assembler, "--target-env", "vulkan1.1", str(asm), "-o", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert assembled.returncode == 0, assembled.stderr[-2000:]
    return subprocess.run(
        [validator, "--target-env", "vulkan1.1", *flags, str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_hlsl_cbuffer_packing_reaches_spirv(tmp_path):
    code = _translate(tmp_path, HLSL_PACKED, "shader.hlsl", "directx")
    assert _member_offsets(code, "Params") == HLSL_PACKED_OFFSETS
    # Both arrays and matrices keep the constant-buffer register stride.
    assert re.findall(r"OpDecorate %\w+ ArrayStride (\d+)", code).count("16") >= 1
    assert _member_decorations(code, "Params", 5)[1:] == ["ColMajor", "MatrixStride 16"]
    assert _member_decorations(code, "Params", 6)[1:] == ["RowMajor", "MatrixStride 16"]
    assert 'OpSourceExtension "GL_EXT_scalar_block_layout"' in code
    result = _validate(tmp_path, code, "--scalar-block-layout")
    if result is None:
        return
    assert result.returncode == 0, result.stderr[-2000:]
    # The packing breaks the default (extended) uniform-buffer rules, which is why the requirement is declared.
    assert _validate(tmp_path, code).returncode != 0


def test_hlsl_packoffset_reaches_spirv(tmp_path):
    code = _translate(tmp_path, HLSL_PACKOFFSET, "shader.hlsl", "directx")
    assert _member_offsets(code, "Params") == HLSL_PACKOFFSET_OFFSETS
    result = _validate(tmp_path, code, "--scalar-block-layout")
    if result is not None:
        assert result.returncode == 0, result.stderr[-2000:]


def test_hlsl_cbuffer_laid_out_as_std140_keeps_the_default_rules(tmp_path):
    code = _translate(tmp_path, HLSL_STD140_COMPATIBLE, "shader.hlsl", "directx")
    assert _member_offsets(code, "Params") == [0, 16, 80]
    assert "OpSourceExtension" not in code
    result = _validate(tmp_path, code)
    if result is not None:
        assert result.returncode == 0, result.stderr[-2000:]


def test_member_in_column_major_matrix_padding_is_refused(tmp_path):
    with pytest.raises(Exception, match=r"padding SPIR-V counts as part of 'rot'"):
        _translate(tmp_path, HLSL_MATRIX_TAIL, "shader.hlsl", "directx")


def test_glsl_offset_and_align_qualifiers_reach_spirv(tmp_path):
    code = _translate(tmp_path, GLSL_EXPLICIT_OFFSETS, "shader.comp", "opengl")
    assert _member_offsets(code, "Params") == GLSL_EXPLICIT_OFFSETS_OFFSETS
    validator, disassembler = shutil.which("glslangValidator"), shutil.which(
        "spirv-dis"
    )
    if validator is not None and disassembler is not None:
        source = tmp_path / "reflect.comp"
        source.write_text(GLSL_EXPLICIT_OFFSETS, encoding="utf-8")
        binary = tmp_path / "reflect.spv"
        subprocess.run(
            [validator, "-V", str(source), "-o", str(binary)],
            capture_output=True,
            check=True,
        )
        listing = subprocess.run(
            [disassembler, str(binary)], capture_output=True, text=True, check=True
        ).stdout
        assert _member_offsets(listing, "Params") == GLSL_EXPLICIT_OFFSETS_OFFSETS
    result = _validate(tmp_path, code, "--scalar-block-layout")
    if result is not None:
        assert result.returncode == 0, result.stderr[-2000:]
