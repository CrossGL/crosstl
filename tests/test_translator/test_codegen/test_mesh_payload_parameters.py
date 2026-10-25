"""A mesh entry's task payload parameter reaches GLSL and SPIR-V as task payload storage.

HLSL's ``in payload`` mesh parameter (CrossGL ``@mesh_payload``) is the data the task stage passed to
``DispatchMesh``. The GLSL generator used to drop the parameter, so the entry read an undeclared name, and the
SPIR-V generator flattened it into per-vertex ``Input`` variables with locations, which no stage writes. GLSL now
declares it ``taskPayloadSharedEXT`` and SPIR-V as a ``TaskPayloadWorkgroupEXT`` variable on the entry interface
(GL_EXT_mesh_shader, SPV_EXT_mesh_shader).
"""

import re
import shutil
import subprocess

import pytest

import crosstl
import crosstl.translator
from crosstl.translator.codegen.GLSL_codegen import GLSLCodeGen
from crosstl.translator.codegen.SPIRV_codegen import VulkanSPIRVCodeGen

MESH = """
shader MeshPayloadParameter {
    struct MeshVertex {
        vec4 position @gl_Position;
        vec3 color @TEXCOORD0;
    }

    struct Payload {
        uint count;
        vec3 color;
    }

    mesh {
        void main(in Payload payload @mesh_payload, out MeshVertex verts[3] @vertices,
                  out uvec3 tris[1] @indices) @numthreads(1, 1, 1) @outputtopology(triangle) {
            SetMeshOutputCounts(3u, min(payload.count, 1u));
            for (uint i = 0u; i < 3u; i++) {
                verts[i].position = vec4(float(i), 0.0, 0.0, 1.0);
                verts[i].color = payload.color;
            }
            tris[0] = uvec3(0u, 1u, 2u);
        }
    }
}
"""

HLSL = """
struct MeshVertex { float4 position : SV_Position; float3 color : TEXCOORD0; };
struct Payload { uint count; float3 color; };
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void main(in payload Payload payload, out vertices MeshVertex verts[3], out indices uint3 tris[1]) {
    SetMeshOutputCounts(3, 1);
    for (uint i = 0; i < 3; i++) {
        verts[i].position = float4(float(i), 0.0, 0.0, 1.0);
        verts[i].color = payload.color;
    }
    tris[0] = uint3(0, 1, 2);
}
"""


def _run(command):
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _validate_spirv(tmp_path, assembly):
    spirv_as = shutil.which("spirv-as")
    spirv_val = shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        pytest.skip("spirv-as and spirv-val are required")
    source = tmp_path / "payload.spvasm"
    binary = tmp_path / "payload.spv"
    source.write_text(assembly, encoding="utf-8")
    result = _run(
        [spirv_as, "--target-env", "vulkan1.3", str(source), "-o", str(binary)]
    )
    assert result.returncode == 0, result.stderr
    result = _run([spirv_val, "--target-env", "vulkan1.3", str(binary)])
    assert result.returncode == 0, result.stdout + result.stderr


def test_glsl_mesh_payload_parameter_is_task_payload_shared(tmp_path):
    code = GLSLCodeGen().generate_stage(crosstl.translator.parse(MESH), "mesh")

    assert "taskPayloadSharedEXT Payload payload;" in code
    assert "payload.color" in code

    glslang = shutil.which("glslangValidator")
    if glslang is None:
        pytest.skip("glslangValidator is not available")
    shader = tmp_path / "payload.mesh"
    shader.write_text(code, encoding="utf-8")
    result = _run(
        [
            glslang,
            "-S",
            "mesh",
            "--target-env",
            "vulkan1.3",
            "-o",
            str(tmp_path / "glsl.spv"),
            str(shader),
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_spirv_mesh_payload_parameter_is_task_payload_workgroup_storage(tmp_path):
    assembly = VulkanSPIRVCodeGen().generate(crosstl.translator.parse(MESH))

    payload_id = re.search(r'OpName (%\w+) "payload"', assembly).group(1)
    assert re.search(
        rf"{payload_id} = OpVariable %\w+ TaskPayloadWorkgroupEXT", assembly
    ), assembly
    entry = next(
        line for line in assembly.splitlines() if "OpEntryPoint MeshEXT" in line
    )
    assert re.search(rf"{payload_id}\b", entry)
    assert " Input" not in assembly
    assert not re.search(rf"OpDecorate {payload_id} Location", assembly)
    _validate_spirv(tmp_path, assembly)


def test_hlsl_mesh_payload_parameter_reaches_spirv_as_task_payload(tmp_path):
    source = tmp_path / "payload.hlsl"
    source.write_text(HLSL, encoding="utf-8")
    assembly = crosstl.translate(str(source), backend="vulkan", format_output=False)

    assert "TaskPayloadWorkgroupEXT" in assembly
    _validate_spirv(tmp_path, assembly)
