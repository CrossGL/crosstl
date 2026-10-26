"""GLSL-style mesh and ray stage interfaces reach HLSL rewritten or refused with a typed diagnostic.

See :mod:`crosstl.translator.semantics.hlsl_stage_interfaces`.
"""

from __future__ import annotations

import pytest

import crosstl

TASK_WITHOUT_PAYLOAD = """#version 450
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
void main()
{
    EmitMeshTasksEXT(3, 1, 1);
}
"""

TASK_WITH_PAYLOAD = """#version 450
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
struct Task { uint count; };
taskPayloadSharedEXT Task task;
void main()
{
    task.count = 3u;
    EmitMeshTasksEXT(3, 1, 1);
}
"""

MESH_GLOBAL_OUTPUTS = """#version 450
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main()
{
    SetMeshOutputsEXT(3, 1);
    gl_MeshVerticesEXT[0].gl_Position = vec4(0.0);
    gl_MeshVerticesEXT[1].gl_Position = vec4(1.0);
    gl_MeshVerticesEXT[2].gl_Position = vec4(2.0);
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
"""

HLSL_TASK = """struct Payload { uint count; };
groupshared Payload payload;
[numthreads(1, 1, 1)]
[shader("amplification")]
void ASMain()
{
    payload.count = 3;
    DispatchMesh(3, 1, 1, payload);
}
"""


def _translate(tmp_path, name, source, backend="directx"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def test_task_without_payload_dispatches_with_a_groupshared_placeholder(tmp_path):
    hlsl = _translate(tmp_path, "shader.task", TASK_WITHOUT_PAYLOAD)
    assert "struct CrossGLMeshPayload {" in hlsl
    assert "groupshared CrossGLMeshPayload crossglMeshPayload;" in hlsl
    assert "DispatchMesh(3, 1, 1, crossglMeshPayload);" in hlsl


def test_task_payload_placeholder_is_hlsl_only(tmp_path):
    glsl = _translate(tmp_path, "shader.task", TASK_WITHOUT_PAYLOAD, backend="glsl")
    assert "CrossGLMeshPayload" not in glsl
    assert "EmitMeshTasksEXT(3, 1, 1);" in glsl


def test_hlsl_task_payload_is_kept(tmp_path):
    hlsl = _translate(tmp_path, "shader.hlsl", HLSL_TASK)
    assert "DispatchMesh(3, 1, 1, payload);" in hlsl
    assert "CrossGLMeshPayload" not in hlsl


@pytest.mark.parametrize(
    ("name", "source", "expected"),
    [
        ("shader.task", TASK_WITH_PAYLOAD, "DispatchMesh(3, 1, 1, task);"),
        ("shader.mesh", MESH_GLOBAL_OUTPUTS, "out vertices MeshVertex meshVertices[3]"),
    ],
)
def test_glsl_mesh_and_task_interfaces_become_entry_parameters(
    tmp_path, name, source, expected
):
    # semantics.glsl_mesh_stages raises the GLSL form before this pass would refuse it.
    hlsl = _translate(tmp_path, name, source)
    assert expected in hlsl
    assert "CrossGLMeshPayload" not in hlsl
