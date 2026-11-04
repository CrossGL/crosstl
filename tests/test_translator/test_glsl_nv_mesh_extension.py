"""GLSL output keeps a GL_NV_mesh_shader program on GL_NV_mesh_shader.

The emitter added ``#extension GL_EXT_mesh_shader : require`` to every mesh and task stage, above the program's own
``GL_NV_mesh_shader`` line. glslang refuses the pair ("GL_EXT_mesh_shader is already turned on, and not allowed with
GL_NV_mesh_shader") and, under OpenGL rules, refuses GL_EXT_mesh_shader itself.
"""

import re
import shutil
import subprocess

import pytest

import crosstl

NV_MESH = """#version 450
#extension GL_NV_mesh_shader : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main() {
    gl_PrimitiveCountNV = 1;
    gl_MeshVerticesNV[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_MeshVerticesNV[1].gl_Position = vec4(0.0, 1.0, 0.0, 1.0);
    gl_MeshVerticesNV[2].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);
    gl_PrimitiveIndicesNV[0] = 0u;
    gl_PrimitiveIndicesNV[1] = 1u;
    gl_PrimitiveIndicesNV[2] = 2u;
}
"""

NV_TASK = """#version 450
#extension GL_NV_mesh_shader : require
layout(local_size_x = 1) in;
void main() {
    gl_TaskCountNV = 1;
}
"""

EXT_MESH = """#version 460
#extension GL_EXT_mesh_shader : require
layout(local_size_x = 1) in;
layout(triangles, max_vertices = 3, max_primitives = 1) out;
void main() {
    SetMeshOutputsEXT(3, 1);
    gl_MeshVerticesEXT[0].gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    gl_MeshVerticesEXT[1].gl_Position = vec4(0.0, 1.0, 0.0, 1.0);
    gl_MeshVerticesEXT[2].gl_Position = vec4(1.0, 0.0, 0.0, 1.0);
    gl_PrimitiveTriangleIndicesEXT[0] = uvec3(0, 1, 2);
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend="opengl")


def _extension_lines(glsl, name):
    return re.findall(rf"^\s*#extension\s+{name}\s*:", glsl, flags=re.M)


def _validate(tmp_path, glsl, stage, *environment):
    validator = shutil.which("glslangValidator")
    if validator is None:
        return
    source = tmp_path / f"out.{stage}"
    source.write_text(glsl)
    result = subprocess.run(
        [validator, *environment, "-S", stage, str(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + glsl


@pytest.mark.parametrize(
    "source, stage", [(NV_MESH, "mesh"), (NV_TASK, "task")], ids=["mesh", "task"]
)
def test_nv_program_keeps_only_the_nv_extension(tmp_path, source, stage):
    glsl = _translate(tmp_path, source, f".{stage}")
    assert len(_extension_lines(glsl, "GL_NV_mesh_shader")) == 1, glsl
    assert not _extension_lines(glsl, "GL_EXT_mesh_shader"), glsl
    _validate(tmp_path, glsl, stage)
    _validate(tmp_path, glsl, stage, "--target-env", "vulkan1.3")


def test_ext_program_keeps_the_ext_extension(tmp_path):
    glsl = _translate(tmp_path, EXT_MESH, ".mesh")
    assert len(_extension_lines(glsl, "GL_EXT_mesh_shader")) == 1, glsl
    assert not _extension_lines(glsl, "GL_NV_mesh_shader"), glsl
    _validate(tmp_path, glsl, "mesh", "--target-env", "vulkan1.3")
