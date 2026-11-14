"""Slang input keeps the stage roles of mesh, task and ray tracing entry points.

The Slang importer could not parse ``struct [raypayload] Payload``, read ``in payload Payload payload`` as a parameter
of type ``payload``, dropped the ``vertices``, ``indices`` and ``primitives`` roles and the mesh stage's
``[numthreads]`` and ``[outputtopology]``, and wrote a miss or hit stage's payload as an ordinary ``inout``
parameter. It now writes the HLSL importer's spellings, so each stage translates to every target that has it.
"""

import re

import pytest

import crosstl

MISS = """struct [raypayload] Payload {
    float4 color : read(caller) : write(closesthit, miss);
};

[shader("miss")]
void main(inout Payload payload) {
    payload.color = float4(1.0, 0.0, 0.0, 1.0);
}
"""

HIT = """struct [raypayload] Payload {
    float4 color : read(caller) : write(closesthit, miss);
};

[shader("closesthit")]
void main(inout Payload payload, in BuiltInTriangleIntersectionAttributes attr) {
    payload.color = float4(attr.barycentrics, 0.0, 1.0);
}
"""

MESH = """struct Vertex {
    float4 position : SV_Position;
};

struct Primitive {
    uint id : SV_PrimitiveID;
};

struct Payload {
    uint value;
};

[shader("mesh")]
[numthreads(2, 1, 1)]
[outputtopology("triangle")]
void main(out vertices Vertex verts[3], out indices uint3 tris[1], out primitives Primitive prims[1],
          in payload Payload payload) {
    SetMeshOutputCounts(3, 1);
    verts[0].position = float4(0.0, 0.0, 0.0, 1.0);
    verts[1].position = float4(float(payload.value), 0.0, 0.0, 1.0);
    verts[2].position = float4(0.0, 1.0, 0.0, 1.0);
    tris[0] = uint3(0, 1, 2);
    prims[0].id = 5;
}
"""

TASK = """struct Payload { uint value; };
groupshared Payload payload;
[shader("amplification")]
[numthreads(4, 1, 1)]
void main() {
    payload.value = 7;
    DispatchMesh(1, 1, 1, payload);
}
"""


def _translate(tmp_path, source, backend):
    path = tmp_path / "shader.slang"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def _entry(crossgl):
    return " ".join(
        re.search(r"void main\((.*?)\)\s*\{", crossgl, flags=re.S).group(1).split()
    )


def test_ray_payload_structure_and_miss_payload(tmp_path):
    crossgl = _translate(tmp_path, MISS, "cgl")
    assert re.search(
        r"vec4 color @ ray_payload_read\(caller\) @ ray_payload_write\(closesthit, ?miss\);",
        crossgl,
    ), crossgl
    assert _entry(crossgl) == "inout Payload payload @ payload"
    spirv = _translate(tmp_path, MISS, "vulkan")
    assert re.search(r"OpVariable %\w+ IncomingRayPayloadKHR", spirv), spirv


def test_hit_stage_payload_and_hit_attributes(tmp_path):
    crossgl = _translate(tmp_path, HIT, "cgl")
    assert _entry(crossgl) == (
        "inout Payload payload @ payload, "
        "in BuiltInTriangleIntersectionAttributes attr @ hit_attribute"
    )
    spirv = _translate(tmp_path, HIT, "vulkan")
    assert re.search(r"OpVariable %\w+ HitAttributeKHR", spirv), spirv


def test_mesh_entry_roles_size_and_topology(tmp_path):
    crossgl = _translate(tmp_path, MESH, "cgl")
    assert _entry(crossgl) == (
        "@ vertices out Vertex verts[3], @ indices out uvec3 tris[1], "
        "@ primitives out Primitive prims[1], @ mesh_payload in Payload payload"
    )
    assert re.search(
        r"@ numthreads\(2, 1, 1\)\s*@ outputtopology\(\"triangle\"\)", crossgl
    )
    hlsl = _translate(tmp_path, MESH, "directx")
    assert re.search(r"\[numthreads\(2,\s*1,\s*1\)\]", hlsl), hlsl
    assert re.search(r"\[outputtopology\(\"triangle\"\)\]", hlsl), hlsl
    assert re.search(r"\bin\s+payload\s+Payload\s+payload\b", hlsl), hlsl
    assert re.search(r"\bout\s+primitives\s+Primitive\s+prims\[1\]", hlsl), hlsl
    spirv = _translate(tmp_path, MESH, "vulkan")
    assert "OutputTrianglesEXT" in spirv and "LocalSize 2 1 1" in spirv, spirv
    assert re.search(r"OpVariable %\w+ TaskPayloadWorkgroupEXT", spirv), spirv


@pytest.mark.parametrize("backend", ["directx", "vulkan"])
def test_task_entry_keeps_its_thread_group_size(tmp_path, backend):
    code = _translate(tmp_path, TASK, backend)
    if backend == "directx":
        assert re.search(r"\[numthreads\(4,\s*1,\s*1\)\]", code), code
    else:
        assert "LocalSize 4 1 1" in code, code
