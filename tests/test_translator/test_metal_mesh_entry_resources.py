"""Metal mesh and object functions receive the program's resources and groupshared values.

HLSL mesh and amplification shaders read constant buffers and structured buffers and keep the
amplification payload in a module-scope ``groupshared`` variable. The Metal generator bound
resources only for vertex, fragment and compute entries, so a mesh function read ``uBO`` without
a ``constant UBO &uBO [[buffer(0)]]`` argument, and an object function could not find the type of
a module-scope payload. ``DispatchMesh`` implies ``GroupMemoryBarrierWithGroupSync`` (Direct3D
mesh shader specification, "DispatchMesh intrinsic"), so the payload copy follows a threadgroup
barrier.
"""

import shutil
import subprocess

import pytest

import crosstl

HLSL_MESH = """cbuffer UBO : register(b0) { float4x4 mvp; };
struct VertexOutput { float4 position : SV_Position; float4 color : COLOR0; };
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(uint gi : SV_GroupIndex, out vertices VertexOutput verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    verts[0].position = mul(mvp, float4(0.0, -1.0, 0.0, 1.0));
    verts[0].color = float4(0, 1, 0, 1);
    verts[1].position = mul(mvp, float4(-1.0, 1.0, 0.0, 1.0));
    verts[1].color = float4(0, 0, 1, 1);
    verts[2].position = mul(mvp, float4(1.0, 1.0, 0.0, 1.0));
    verts[2].color = float4(1, 0, 0, 1);
    tris[gi] = uint3(0, 1, 2);
}
"""

HLSL_AMPLIFICATION = """StructuredBuffer<uint> visible : register(t0);
cbuffer Cull : register(b0) { uint count; };
struct Payload { uint ids[32]; };
groupshared Payload payload;
[numthreads(32, 1, 1)]
void main(uint gi : SV_GroupIndex) {
    payload.ids[gi] = visible[gi] + count;
    DispatchMesh(1, 1, 1, payload);
}
"""


def _translate(tmp_path, name, source):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="metal", format_output=False)


def _compile(tmp_path, code):
    if shutil.which("xcrun") is None:
        pytest.skip("the Metal toolchain is not installed")
    found = subprocess.run(
        ["xcrun", "-sdk", "macosx", "--find", "metal"], capture_output=True
    )
    if found.returncode != 0:
        pytest.skip("the Metal toolchain is not installed")
    artifact = tmp_path / "out.metal"
    artifact.write_text(code)
    result = subprocess.run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(artifact)]
        + ["-o", str(tmp_path / "out.air")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_mesh_functions_bind_constant_buffers(tmp_path):
    code = _translate(tmp_path, "shader.hlsl", HLSL_MESH)
    signature = code[
        code.index("void mesh_main(") : code.index("{", code.index("void mesh_main("))
    ]
    assert "constant UBO& uBO [[buffer(0)]]" in signature.replace(" &", "& ")
    assert "uBO.mvp * float4(0.0, -1.0, 0.0, 1.0)" in code
    _compile(tmp_path, code)


def test_object_functions_bind_resources_and_module_scope_payloads(tmp_path):
    code = _translate(tmp_path, "shader.hlsl", HLSL_AMPLIFICATION)
    start = code.index("void object_main(")
    signature = code[start : code.index("{", start)]
    assert "constant Cull& cull [[buffer(0)]]" in signature.replace(" &", "& ")
    assert "const device uint* visible [[buffer(1)]]" in signature.replace(" *", "* ")
    assert "object_data Payload& _crossglMeshPayload [[payload]]" in signature.replace(
        " &", "& "
    )
    body = code[code.index("{", start) :]
    assert "threadgroup Payload payload;" in body
    assert "payload.ids[gi] = visible[gi] + cull.count;" in body
    barrier = body.index("threadgroup_barrier(mem_flags::mem_threadgroup);")
    assert barrier < body.index("_crossglMeshPayload = payload;")
    assert "threadgroup Payload payload;" not in code[:start]
    _compile(tmp_path, code)
