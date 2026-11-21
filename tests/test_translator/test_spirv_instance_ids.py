"""SPIR-V's ``InstanceIndex`` includes the draw's first instance; CrossGL's ``gl_InstanceID`` does not.

The SPIR-V generator decorated ``gl_InstanceID`` as ``BuiltIn InstanceIndex`` and subtracted nothing, so a draw
with ``firstInstance != 0`` read ``firstInstance`` more than every other target. It now reads
``InstanceIndex - BaseInstance`` with ``DrawParameters``, as slangc and DXC lower ``SV_InstanceID``. The SPIR-V
importer reads ``InstanceIndex`` as ``gl_InstanceIndex``, so a SPIR-V round trip keeps the bare built-in. See
:mod:`crosstl.translator.semantics.spirv_instance_ids`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

CROSSGL_READ = """shader InstanceRead {
    vertex {
        void main() {
            gl_Position = vec4(float(gl_InstanceID), 0.0, 0.0, 1.0);
        }
    }
}
"""

CROSSGL_HELPER = """shader InstanceRead {
    float instance() { return float(gl_InstanceID); }
    vertex {
        void main() {
            gl_Position = vec4(instance(), 0.0, 0.0, 1.0);
        }
    }
}
"""

# The form the coverage template uses: a stage input structure member.
CROSSGL_MEMBER = """shader Probe {
    struct VertexInput {
        vec4 position;
        int iid @ gl_InstanceID;
    }
    struct VertexOutput {
        vec4 position @ gl_Position;
    }
    vertex {
        VertexOutput main(VertexInput input) {
            VertexOutput output;
            output.position = input.position * 2.0 + vec4(float(input.iid), 0.0, 0.0, 0.0);
            return output;
        }
    }
}
"""

CROSSGL_STAGE_INPUT = """shader main {
    vertex {
        int gl_InstanceID @input @gl_InstanceID;
        vec4 gl_Position @output @gl_Position;
        void main() {
            gl_Position = vec4(float(gl_InstanceID), 0.0, 0.0, 1.0);
        }
    }
}
"""

HLSL = """float4 main(uint iid : SV_InstanceID) : SV_Position {
    return float4(float(iid), 0, 0, 1);
}
"""

GLSL = """#version 450
void main() { gl_Position = vec4(float(gl_InstanceID), 0.0, 0.0, 1.0); }
"""

INSTANCE_INDEX = """shader InstanceIndexRead {
    vertex {
        void main() {
            gl_Position = vec4(float(gl_InstanceIndex), 0.0, 0.0, 1.0);
        }
    }
}
"""

# glslang 16 output for `gl_Position = vec4(float(gl_InstanceIndex), 0.0, 0.0, 1.0);` (Vulkan 1.0).
GLSLANG_INSTANCE_INDEX = """; SPIR-V
; Version: 1.0
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 27
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Vertex %main "main" %_ %gl_InstanceIndex
               OpSource GLSL 450
               OpName %main "main"
               OpName %gl_PerVertex "gl_PerVertex"
               OpMemberName %gl_PerVertex 0 "gl_Position"
               OpName %_ ""
               OpName %gl_InstanceIndex "gl_InstanceIndex"
               OpMemberDecorate %gl_PerVertex 0 BuiltIn Position
               OpDecorate %gl_PerVertex Block
               OpDecorate %gl_InstanceIndex BuiltIn InstanceIndex
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%gl_PerVertex = OpTypeStruct %v4float
%_ptr_Output_gl_PerVertex = OpTypePointer Output %gl_PerVertex
          %_ = OpVariable %_ptr_Output_gl_PerVertex Output
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
%_ptr_Input_int = OpTypePointer Input %int
%gl_InstanceIndex = OpVariable %_ptr_Input_int Input
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
%_ptr_Output_v4float = OpTypePointer Output %v4float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %18 = OpLoad %int %gl_InstanceIndex
         %19 = OpConvertSToF %float %18
         %22 = OpCompositeConstruct %v4float %19 %float_0 %float_0 %float_1
         %24 = OpAccessChain %_ptr_Output_v4float %_ %int_0
               OpStore %24 %22
               OpReturn
               OpFunctionEnd
"""


def translate(tmp_path, source, backend, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _validate(tmp_path, assembly):
    spirv_as, spirv_val = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (spirv_as and spirv_val):
        return
    (tmp_path / "out.spvasm").write_text(assembly)
    for command in (
        [spirv_as, "--target-env", "vulkan1.0", "out.spvasm", "-o", "out.spv"],
        [spirv_val, "--target-env", "vulkan1.0", "out.spv"],
    ):
        result = subprocess.run(
            command, cwd=tmp_path, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("source", "name"),
    [
        (CROSSGL_READ, "shader.cgl"),
        (CROSSGL_HELPER, "shader.cgl"),
        (CROSSGL_MEMBER, "shader.cgl"),
        (CROSSGL_STAGE_INPUT, "shader.cgl"),
        (HLSL, "shader.hlsl"),
        (GLSL, "shader.vert"),
    ],
    ids=["read", "helper", "member", "stage-input", "hlsl", "glsl"],
)
def test_spirv_subtracts_the_base_instance(tmp_path, source, name):
    assembly = translate(tmp_path, source, "vulkan", name)
    assert "OpCapability DrawParameters" in assembly
    assert 'OpExtension "SPV_KHR_shader_draw_parameters"' in assembly
    assert re.search(r"OpDecorate %\w+ BuiltIn InstanceIndex", assembly)
    assert re.search(r"OpDecorate %\w+ BuiltIn BaseInstance", assembly)
    assert "OpISub" in assembly
    _validate(tmp_path, assembly)


@pytest.mark.parametrize(
    ("source", "name"),
    [(INSTANCE_INDEX, "shader.cgl"), (GLSLANG_INSTANCE_INDEX, "shader.spvasm")],
    ids=["crossgl", "glslang"],
)
def test_spirv_instance_index_stays_bare(tmp_path, source, name):
    assembly = translate(tmp_path, source, "vulkan", name)
    assert re.search(r"OpDecorate %\w+ BuiltIn InstanceIndex", assembly)
    assert "BaseInstance" not in assembly
    assert "DrawParameters" not in assembly
    assert "OpISub" not in assembly
    _validate(tmp_path, assembly)


def test_spirv_importer_reads_instance_index_as_gl_instance_index(tmp_path):
    crossgl = translate(tmp_path, GLSLANG_INSTANCE_INDEX, "cgl", "shader.spvasm")
    assert "@gl_InstanceIndex" in crossgl
    assert "@gl_InstanceID" not in crossgl


def test_spirv_instance_index_keeps_the_base_on_slang_and_metal(tmp_path):
    slang = translate(tmp_path, GLSLANG_INSTANCE_INDEX, "slang", "shader.spvasm")
    assert re.search(r"uint \w+ : SV_InstanceID", slang)
    assert re.search(r"uint \w+ : SV_StartInstanceLocation", slang)
    metal = translate(tmp_path, GLSLANG_INSTANCE_INDEX, "metal", "shader.spvasm")
    assert "[[instance_id]]" in metal
    assert "[[base_instance]]" not in metal
