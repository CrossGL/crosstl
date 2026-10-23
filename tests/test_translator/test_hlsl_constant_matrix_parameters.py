"""Metal ``constant floatCxR &m [[buffer(n)]]`` entry parameters become HLSL constant buffers.

Scalars and vectors passed this way already became constant buffers, but matrices did not: a vertex function's
``constant float4x4 &transform`` became a per-vertex input (``float4x4 transform : TEXCOORD0``), and a kernel's became
an entry parameter without a semantic. See :mod:`crosstl.translator.codegen.hlsl_constant_matrices`.
"""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.hlsl_constant_matrices import (
    DirectXConstantMatrixLayoutError,
)

VERTEX = """
#include <metal_stdlib>
using namespace metal;
struct VertexIn { float3 position [[attribute(0)]]; };
struct VertexOut { float4 position [[position]]; };
vertex VertexOut shade(VertexIn in [[stage_in]],
                       constant float4x4 &transform [[buffer(1)]],
                       constant float3x3 &normals [[buffer(2)]],
                       constant float4x3 &skew [[buffer(3)]]) {
    VertexOut out;
    float3 bent = normals * in.position + skew * float4(in.position, 1.0);
    out.position = transform * float4(bent, 1.0);
    return out;
}
"""

KERNEL = """
#include <metal_stdlib>
using namespace metal;
kernel void apply(constant float4x4 &m [[buffer(0)]],
                  device float4 *values [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {
    values[tid] = m * values[tid];
}
"""


def _translate(tmp_path, source, **options):
    path = tmp_path / "shader.metal"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(
        str(path), backend="directx", source_options=options or None
    )


def _compile(tmp_path, hlsl, profile, entry):
    dxc = shutil.which("dxc")
    if dxc is None:
        return None
    path, listing = tmp_path / "shader.hlsl", tmp_path / "shader.asm"
    path.write_text(hlsl, encoding="utf-8")
    completed = subprocess.run(
        [dxc, "-T", profile, "-E", entry, str(path), "-Fc", str(listing)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, (completed.stdout + completed.stderr)[-2000:]
    return listing.read_text(encoding="utf-8", errors="replace")


def test_vertex_matrix_constants_become_constant_buffers(tmp_path):
    hlsl = _translate(tmp_path, VERTEX)
    entry = re.search(r"VertexOut VSMain\(([^)]*)\)", hlsl)
    assert entry and entry.group(1).strip() == "VertexIn in_", hlsl
    for name, kind, slot in (
        ("transform", "float4x4", "b1"),
        ("normals", "float3x3", "b2"),
        ("skew", "float3x4", "b3"),
    ):
        assert re.search(
            rf"cbuffer shade_{name}_Constants : register\({slot}\)\s*\{{\s*{kind} shade_{name};",
            hlsl,
        ), name
    assert "TEXCOORD" not in hlsl.split("VSMain", 1)[1].split(")", 1)[0]
    listing = _compile(tmp_path, hlsl, "vs_6_0", "VSMain")
    if listing is not None:
        # Every column starts a 16-byte register, as in MSL for 3- and 4-row float columns.
        assert re.search(
            r"column_major float3x3 shade_normals;\s*; Offset:\s+0", listing
        )


def test_kernel_matrix_constant_becomes_a_constant_buffer(tmp_path):
    hlsl = _translate(tmp_path, KERNEL, threadgroup_size=(64, 1, 1))
    assert re.search(r"cbuffer apply_m_Constants : register\(b0\)", hlsl)
    # The formatter may wrap the entry signature.
    assert re.search(
        r"void CSMain\(\s*uint3 \w+\s*:\s*SV_DispatchThreadID\)", hlsl
    ), hlsl
    _compile(tmp_path, hlsl, "cs_6_0", "CSMain")


@pytest.mark.parametrize(
    ("declaration", "reason"),
    [
        ("constant float2x2 &m", "2-row float columns 8 bytes apart"),
        ("constant float3x2 &m", "2-row float columns 8 bytes apart"),
        ("constant half3x3 &m", "half"),
    ],
)
def test_matrices_with_narrower_columns_are_refused(tmp_path, declaration, reason):
    source = f"""
#include <metal_stdlib>
using namespace metal;
kernel void apply({declaration} [[buffer(0)]],
                  device float *values [[buffer(1)]],
                  uint tid [[thread_position_in_grid]]) {{
    values[tid] = float(m[0][0]) + values[tid];
}}
"""
    with pytest.raises(DirectXConstantMatrixLayoutError) as raised:
        _translate(tmp_path, source, threadgroup_size=(64, 1, 1))
    assert (
        raised.value.project_diagnostic_code
        == "project.translate.directx-constant-matrix-layout-unsupported"
    )
    assert reason in str(raised.value) or "float16_t" in str(raised.value)
