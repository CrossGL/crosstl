"""Slang output of GLSL-style mesh and task stages.

Slang declares mesh and task stages as HLSL does, but only the DirectX and Metal targets raised a GLSL-style stage
to the entry-parameter form, so Slang output copied ``gl_MeshVerticesEXT``, ``gl_PrimitiveTriangleIndicesEXT`` and
``EmitMeshTasksEXT`` verbatim (slangc: undefined identifier). A task stage built alone was also refused, because the
check of its ``DispatchMesh`` payload looked for a mesh stage in the same program.
"""

import re
import shutil
import subprocess

import crosstl

MESH = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
layout(location = 0) out vec3 color[];
void main() {
    SetMeshOutputsEXT(3, 1);
    gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_MeshVerticesEXT[1].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);
    gl_MeshVerticesEXT[2].gl_Position = vec4(0.0, 1.0, 0.0, 1.0);
    color[0] = vec3(1.0, 0.0, 0.0);
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
"""

TASK = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
struct Payload { uint value; };
taskPayloadSharedEXT Payload payload;
void main() {
    payload.value = 7u;
    EmitMeshTasksEXT(1, 1, 1);
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="slang")


def _compile(tmp_path, code, stage):
    slangc = shutil.which("slangc")
    if slangc is None:
        return None
    entry = re.search(r"\bvoid\s+(\w+)\s*\(", code).group(1)
    source, output = tmp_path / "out.slang", tmp_path / "out.spvasm"
    source.write_text(code)
    result = subprocess.run(
        [slangc, str(source), "-entry", entry, "-stage", stage]
        + ["-target", "spirv-asm", "-o", str(output)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    return output.read_text()


def test_glsl_mesh_stage_takes_entry_outputs(tmp_path):
    slang = _translate(tmp_path, MESH, ".mesh")
    assert not re.search(r"\bgl_(MeshVerticesEXT|PrimitiveTriangleIndicesEXT)\b", slang)
    assert re.search(r"\bout\s+vertices\s+\w+\s+\w+\[3\]", slang), slang
    assert re.search(r"\bout\s+indices\s+uint3\s+\w+\[1\]", slang), slang
    assert re.search(r"\[outputtopology\(\"triangle\"\)\]", slang), slang
    spirv = _compile(tmp_path, slang, "mesh")
    if spirv is not None:
        assert "OpSetMeshOutputsEXT" in spirv


def test_glsl_task_stage_built_alone_dispatches_its_payload(tmp_path):
    slang = _translate(tmp_path, TASK, ".task")
    assert re.search(r"\bgroupshared\s+Payload\s+payload\s*;", slang), slang
    assert re.search(r"DispatchMesh\(1,\s*1,\s*1,\s*payload\)", slang), slang
    spirv = _compile(tmp_path, slang, "amplification")
    if spirv is not None:
        assert "OpEmitMeshTasksEXT" in spirv
