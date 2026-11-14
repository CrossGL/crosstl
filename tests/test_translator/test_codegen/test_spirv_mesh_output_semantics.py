"""System-value semantics on HLSL mesh output members in generated SPIR-V.

A mesh output member whose semantic names a system value is a SPIR-V BuiltIn; every other member is a user
output at a Location. ``SV_ShadingRate`` on a primitive output used to fall through to a user output at Location
0, so the per-primitive shading rate never reached the rasterizer, and a system value with no mesh BuiltIn
(``SV_ClipDistance``, or a per-primitive value on a vertex) became a user output the same way. The shading rate
is now ``PrimitiveShadingRateKHR`` (D3D12_SHADING_RATE and the Vulkan encoding agree bit for bit), and the others
are refused.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_exact_lookups import (
    spirv_mesh_output_builtin,
    spirv_mesh_output_refusal,
)
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

MESH = """struct V {{ float4 pos : SV_Position; {vertex_member} }};
struct P {{ {primitive_member} }};
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void main(out vertices V verts[3], out indices uint3 tris[1], out primitives P prims[1]) {{
    SetMeshOutputCounts(3, 1);
    verts[0].pos = float4(0, 0, 0, 1);
    verts[1].pos = float4(1, 0, 0, 1);
    verts[2].pos = float4(0, 1, 0, 1);
    {vertex_store}
    tris[0] = uint3(0, 1, 2);
    {primitive_store}
}}
"""


def _translate(
    tmp_path,
    vertex_member="",
    primitive_member="uint pid : SV_PrimitiveID;",
    vertex_store="",
    primitive_store="prims[0].pid = 3;",
):
    path = tmp_path / "mesh.hlsl"
    path.write_text(
        MESH.format(
            vertex_member=vertex_member,
            primitive_member=primitive_member,
            vertex_store=vertex_store,
            primitive_store=primitive_store,
        ),
        encoding="utf-8",
    )
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _validate(tmp_path, assembly):
    missing = [name for name in ("spirv-as", "spirv-val") if shutil.which(name) is None]
    if missing:
        pytest.skip(f"{', '.join(missing)} not on PATH")
    source = tmp_path / "out.spvasm"
    binary = tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        ["spirv-as", "--target-env", "vulkan1.3", str(source), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.3", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("role", "semantic", "builtin"),
    [
        ("vertices", "SV_Position", "Position"),
        ("vertices", "sv_position", "Position"),
        ("vertices", "gl_PointSize", "PointSize"),
        ("primitives", "SV_ShadingRate", "PrimitiveShadingRateKHR"),
        ("primitives", "gl_PrimitiveShadingRateEXT", "PrimitiveShadingRateKHR"),
        ("primitives", "SV_RenderTargetArrayIndex", "Layer"),
        ("primitives", "SV_ViewportArrayIndex", "ViewportIndex"),
        ("primitives", "SV_CullPrimitive", "CullPrimitiveEXT"),
        ("primitives", "SV_PrimitiveID", "PrimitiveId"),
        ("vertices", "COLOR0", None),
        ("vertices", "POSITION", None),
        ("primitives", "SV_Position", None),
    ],
)
def test_mesh_output_builtin_table(role, semantic, builtin):
    assert spirv_mesh_output_builtin(role, semantic) == builtin


@pytest.mark.parametrize(
    ("role", "semantic", "reason"),
    [
        ("vertices", "SV_ClipDistance", "no SPIR-V mesh output BuiltIn"),
        ("vertices", "gl_CullDistance", "no SPIR-V mesh output BuiltIn"),
        ("vertices", "SV_PrimitiveID", "per-primitive system value"),
        ("primitives", "SV_Position", "per-vertex system value"),
        ("primitives", "SV_Target0", "no SPIR-V mesh output BuiltIn"),
    ],
)
def test_mesh_output_system_value_without_builtin_is_refused(role, semantic, reason):
    assert reason in spirv_mesh_output_refusal(role, semantic)


@pytest.mark.parametrize(
    "semantic", [None, "COLOR0", "TEXCOORD3", "POSITION", "SV_Position"]
)
def test_mesh_output_user_semantics_and_builtins_are_not_refused(semantic):
    assert spirv_mesh_output_refusal("vertices", semantic) is None


def test_primitive_shading_rate_is_the_primitive_shading_rate_builtin(tmp_path):
    assembly = _translate(
        tmp_path,
        primitive_member="uint rate : SV_ShadingRate; uint layer : SV_RenderTargetArrayIndex;",
        primitive_store="prims[0].rate = 5; prims[0].layer = 1;",
    )
    builtins = re.findall(r"OpDecorate %\d+ BuiltIn (\w+)", assembly)
    assert "PrimitiveShadingRateKHR" in builtins
    assert "Layer" in builtins
    assert not re.search(r"OpDecorate %\d+ Location", assembly)
    assert "OpCapability FragmentShadingRateKHR" in assembly
    _validate(tmp_path, assembly)


@pytest.mark.parametrize(
    ("vertex_member", "vertex_store", "message"),
    [
        (
            "float clip : SV_ClipDistance0;",
            "verts[0].clip = 1.0;",
            "mesh vertices output semantic 'gl_ClipDistance' names a system value",
        ),
        (
            "uint pid : SV_PrimitiveID;",
            "verts[0].pid = 1;",
            "is a per-primitive system value",
        ),
    ],
)
def test_vertex_output_system_value_without_builtin_is_refused(
    tmp_path, vertex_member, vertex_store, message
):
    with pytest.raises(SpirvLoweringRefused, match=re.escape(message)):
        _translate(tmp_path, vertex_member=vertex_member, vertex_store=vertex_store)
