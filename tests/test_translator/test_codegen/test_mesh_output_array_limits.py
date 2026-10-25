"""Mesh stages whose output counts are computed at run time.

``SetMeshOutputCounts`` bounds GLSL's ``max_vertices``/``max_primitives`` and SPIR-V's ``OutputVertices``/
``OutputPrimitivesEXT`` only when its arguments are constants. Without ``@max_vertices``/``@max_primitives``, the
GLSL generator used to omit the limits (glslang rejects the shader) and the SPIR-V generator declared one primitive
while the entry wrote up to the size of its index array. Both now take the limits from the ``@vertices``,
``@primitives`` and ``@indices`` array sizes, as HLSL does.

Mesh and task stages marked ``@ host_workgroup_size`` take the host's size the same way compute stages do.
"""

import shutil
import subprocess

import pytest

import crosstl.translator
from crosstl.translator.codegen.directx_codegen import HLSLCodeGen
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen
from crosstl.translator.host_workgroup_size import HostDefinedWorkgroupSizeError

MESH = """
shader DynamicMeshCounts {
    struct MeshVertex {
        vec4 position @gl_Position;
    }

    mesh {
        @ host_workgroup_size(4, 1, 1)
        void main(out MeshVertex verts[12] @vertices, out uvec3 tris[6] @indices,
                  uvec3 group @gl_WorkGroupID, uint lid @gl_LocalInvocationIndex)
            @outputtopology(triangle) {
            uint count = group.x % 7u;
            SetMeshOutputCounts(12u, count);
            if (lid < count) {
                verts[lid].position = vec4(float(lid), 0.0, 0.0, 1.0);
                tris[lid] = uvec3(lid, lid, lid);
            }
        }
    }
}
"""


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def test_glsl_mesh_layout_takes_limits_from_the_output_arrays(tmp_path):
    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(MESH), "mesh")

    assert "layout(triangles, max_vertices = 12, max_primitives = 6) out;" in code
    assert "layout(local_size_x = 4, local_size_y = 1, local_size_z = 1) in;" in code

    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not available")
    shader = tmp_path / "dynamic.mesh"
    shader.write_text(code, encoding="utf-8")
    result = _run(
        [
            glslang,
            "-S",
            "mesh",
            "--target-env",
            "vulkan1.3",
            "-o",
            str(tmp_path / "dynamic.spv"),
            str(shader),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_mesh_execution_modes_take_limits_from_the_output_arrays(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(crosstl.translator.parse(MESH))

    assert "LocalSize 4 1 1" in assembly
    assert "OutputVertices 12" in assembly
    assert "OutputPrimitivesEXT 6" in assembly

    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        pytest.skip("spirv-as and spirv-val are required")
    source = tmp_path / "dynamic.spvasm"
    binary = tmp_path / "dynamic.spv"
    source.write_text(assembly, encoding="utf-8")
    result = _run(
        [spirv_as, "--target-env", "vulkan1.3", str(source), "-o", str(binary)]
    )
    assert result.returncode == 0, result.stderr
    result = _run([spirv_val, "--target-env", "vulkan1.3", str(binary)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_mesh_stage_takes_the_host_workgroup_size():
    code = HLSLCodeGen().generate(crosstl.translator.parse(MESH))
    assert "[numthreads(4, 1, 1)]" in code


def test_size_dependent_mesh_stage_without_a_host_size_is_refused():
    shader = MESH.replace("@ host_workgroup_size(4, 1, 1)", "@ host_workgroup_size")
    with pytest.raises(HostDefinedWorkgroupSizeError, match="Entry 'main'"):
        HLSLCodeGen().generate(crosstl.translator.parse(shader))


def test_size_independent_mesh_stage_drops_the_host_mark():
    shader = (
        MESH.replace("@ host_workgroup_size(4, 1, 1)", "@ host_workgroup_size")
        .replace(
            ",\n                  uvec3 group @gl_WorkGroupID, uint lid @gl_LocalInvocationIndex",
            "",
        )
        .replace("group.x % 7u", "6u")
        .replace("lid", "0u")
    )
    code = HLSLCodeGen().generate(crosstl.translator.parse(shader))
    assert "[numthreads(1, 1, 1)]" in code
