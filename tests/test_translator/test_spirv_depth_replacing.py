"""SPIR-V fragment entries that write ``gl_FragDepth`` declare ``DepthReplacing`` whatever form the output takes."""

import re
import shutil
import subprocess

import pytest

import crosstl

STAGE_VARIABLE = """shader Probe {
    fragment {
        vec4 color @output @location(0);
        vec2 uv @input @location(0);
        float depth @output @gl_FragDepth;
        void main() {
            color = vec4(uv, 0.0, 1.0);
            depth = uv.x * 0.5 + 0.25;
        }
    }
}
"""

STRUCT_MEMBER = """shader Probe {
    struct FragmentInput {
        vec2 uv @ location(0);
    }
    struct FragmentOutput {
        vec4 color @ location(0);
        float depth @ gl_FragDepth;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, 0.0, 1.0);
            output.depth = input.uv.x * 0.5 + 0.25;
            return output;
        }
    }
}
"""

NO_DEPTH = """shader Probe {
    fragment {
        vec4 color @output @location(0);
        vec2 uv @input @location(0);
        void main() {
            color = vec4(uv, 0.0, 1.0);
        }
    }
}
"""

GLSL_SOURCE = """#version 450 core
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(uv, 0.0, 1.0);
    gl_FragDepth = uv.x * 0.5 + 0.25;
}
"""

# glslang's module for GLSL_SOURCE, which the SPIR-V importer reads back as a stage-level gl_FragDepth output.
SPIRV_SOURCE = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %main "main" %c %uv %gl_FragDepth
OpExecutionMode %main OriginUpperLeft
OpExecutionMode %main DepthReplacing
OpDecorate %c Location 0
OpDecorate %uv Location 0
OpDecorate %gl_FragDepth BuiltIn FragDepth
%void = OpTypeVoid
%3 = OpTypeFunction %void
%float = OpTypeFloat 32
%v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
%c = OpVariable %_ptr_Output_v4float Output
%v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
%uv = OpVariable %_ptr_Input_v2float Input
%float_0 = OpConstant %float 0
%float_1 = OpConstant %float 1
%_ptr_Output_float = OpTypePointer Output %float
%gl_FragDepth = OpVariable %_ptr_Output_float Output
%uint = OpTypeInt 32 0
%uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
%float_0_5 = OpConstant %float 0.5
%float_0_25 = OpConstant %float 0.25
%main = OpFunction %void None %3
%5 = OpLabel
%13 = OpLoad %v2float %uv
%16 = OpCompositeExtract %float %13 0
%17 = OpCompositeExtract %float %13 1
%18 = OpCompositeConstruct %v4float %16 %17 %float_0 %float_1
OpStore %c %18
%24 = OpAccessChain %_ptr_Input_float %uv %uint_0
%25 = OpLoad %float %24
%27 = OpFMul %float %25 %float_0_5
%29 = OpFAdd %float %27 %float_0_25
OpStore %gl_FragDepth %29
OpReturn
OpFunctionEnd
"""


def _translate(tmp_path, source, name="shader.cgl"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend="vulkan", format_output=False)


def _depth_replacing_modes(code):
    return re.findall(r"OpExecutionMode %\w+ DepthReplacing\b", code)


@pytest.mark.parametrize(
    "source", [STAGE_VARIABLE, STRUCT_MEMBER], ids=["stage-variable", "member"]
)
def test_fragment_depth_output_declares_depth_replacing(tmp_path, source):
    code = _translate(tmp_path, source)
    assert re.search(r"OpDecorate %\w+ BuiltIn FragDepth\b", code)
    assert len(_depth_replacing_modes(code)) == 1


def test_fragment_without_depth_output_keeps_depth(tmp_path):
    assert _depth_replacing_modes(_translate(tmp_path, NO_DEPTH)) == []


@pytest.mark.parametrize(
    "source, name",
    [(GLSL_SOURCE, "shader.frag"), (SPIRV_SOURCE, "shader.spvasm")],
    ids=["glsl", "spirv"],
)
def test_imported_fragment_depth_write_declares_depth_replacing(tmp_path, source, name):
    code = _translate(tmp_path, source, name)
    assert len(_depth_replacing_modes(code)) == 1


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "source", [STAGE_VARIABLE, STRUCT_MEMBER], ids=["stage-variable", "member"]
)
def test_fragment_depth_output_validates_for_vulkan(tmp_path, source):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not on PATH")
    assembly = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    assembly.write_text(_translate(tmp_path, source))
    _run(["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
