"""SPIR-V output and input carry CrossGL's conservative depth layouts as depth execution modes."""

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.spirv_refusals import SpirvLoweringRefused

STAGE_LEVEL = """shader Probe {
    struct FragmentInput {
        vec2 uv;
    }
    out vec4 color @location(0);
    out float gl_FragDepth @LAYOUT;
    fragment {
        void main(FragmentInput input) {
            color = vec4(input.uv, 0.0, 1.0);
            gl_FragDepth = input.uv.x * 0.5 + 0.25;
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
        float depth @ gl_FragDepth @ LAYOUT;
    }
    fragment {
        FragmentOutput main(FragmentInput input) {
            FragmentOutput output;
            output.color = vec4(input.uv, 0.0, 1.0);
            output.depth = input.uv.x * 0.5;
            return output;
        }
    }
}
"""

COLOR_LAYOUT = """shader Probe {
    fragment {
        vec4 color @output @location(0) @depth_greater;
        vec2 uv @input @location(0);
        void main() {
            color = vec4(uv, 0.0, 1.0);
        }
    }
}
"""

TWO_LAYOUTS = """shader Probe {
    fragment {
        vec4 color @output @location(0);
        float depth @output @gl_FragDepth @depth_greater @depth_less;
        vec2 uv @input @location(0);
        void main() {
            color = vec4(uv, 0.0, 1.0);
            depth = uv.x;
        }
    }
}
"""

GLSL_SOURCE = """#version 450
layout(depth_unchanged) out float gl_FragDepth;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
void main() {
    color = vec4(uv, 0.0, 1.0);
    gl_FragDepth = uv.x;
}
"""

# glslang's module for GLSL_SOURCE with layout(depth_greater).
SPIRV_SOURCE = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint Fragment %4 "main" %9 %12 %20
OpExecutionMode %4 OriginUpperLeft
OpExecutionMode %4 DepthReplacing
OpExecutionMode %4 DepthGreater
OpDecorate %9 Location 0
OpDecorate %12 Location 0
OpDecorate %20 BuiltIn FragDepth
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeFloat 32
%7 = OpTypeVector %6 4
%8 = OpTypePointer Output %7
%9 = OpVariable %8 Output
%10 = OpTypeVector %6 2
%11 = OpTypePointer Input %10
%12 = OpVariable %11 Input
%14 = OpConstant %6 0
%15 = OpConstant %6 1
%19 = OpTypePointer Output %6
%20 = OpVariable %19 Output
%21 = OpTypeInt 32 0
%22 = OpConstant %21 0
%23 = OpTypePointer Input %6
%4 = OpFunction %2 None %3
%5 = OpLabel
%13 = OpLoad %10 %12
%16 = OpCompositeExtract %6 %13 0
%17 = OpCompositeExtract %6 %13 1
%18 = OpCompositeConstruct %7 %16 %17 %14 %15
OpStore %9 %18
%24 = OpAccessChain %23 %12 %22
%25 = OpLoad %6 %24
OpStore %20 %25
OpReturn
OpFunctionEnd
"""

DEPTH_MODES = ("DepthGreater", "DepthLess", "DepthUnchanged")
LAYOUTS = {
    "depth_greater": "DepthGreater",
    "depth_less": "DepthLess",
    "depth_unchanged": "DepthUnchanged",
    "depth_any": None,
}


def _translate(tmp_path, source, name="shader.cgl", backend="vulkan"):
    path = tmp_path / name
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _execution_modes(code):
    return re.findall(r"OpExecutionMode %\w+ (\w+)", code)


def _depth_modes(code):
    return [mode for mode in _execution_modes(code) if mode in DEPTH_MODES]


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
@pytest.mark.parametrize(
    "template", [STAGE_LEVEL, STRUCT_MEMBER], ids=["stage-level", "member"]
)
def test_depth_layout_becomes_one_execution_mode(tmp_path, template, layout):
    code = _translate(tmp_path, template.replace("LAYOUT", layout))
    modes = _execution_modes(code)
    expected = LAYOUTS[layout]
    assert _depth_modes(code) == ([expected] if expected else [])
    assert "DepthReplacing" in modes
    # The layout decorates the depth output; it never turns into a location or a second built-in.
    assert len(re.findall(r"BuiltIn FragDepth\b", code)) == 1


def test_glsl_depth_layout_reaches_spirv(tmp_path):
    code = _translate(tmp_path, GLSL_SOURCE, "shader.frag")
    assert _depth_modes(code) == ["DepthUnchanged"]


def test_layout_on_a_colour_output_is_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="only the fragment depth output"):
        _translate(tmp_path, COLOR_LAYOUT)


def test_two_layouts_on_one_output_are_refused(tmp_path):
    with pytest.raises(SpirvLoweringRefused, match="depth_greater, depth_less"):
        _translate(tmp_path, TWO_LAYOUTS)


def test_importer_writes_the_depth_layout(tmp_path):
    crossgl = _translate(tmp_path, SPIRV_SOURCE, "shader.spvasm", backend="cgl")
    assert re.search(
        r"float gl_FragDepth @output @gl_FragDepth @depth_greater;", crossgl
    )


def test_imported_depth_layout_round_trips(tmp_path):
    code = _translate(tmp_path, SPIRV_SOURCE, "shader.spvasm")
    assert _depth_modes(code) == ["DepthGreater"]


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "template", [STAGE_LEVEL, STRUCT_MEMBER], ids=["stage-level", "member"]
)
def test_depth_layout_validates_for_vulkan(tmp_path, template):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("spirv-as and spirv-val are not on PATH")
    assembly = tmp_path / "shader.spvasm"
    binary = tmp_path / "shader.spv"
    assembly.write_text(_translate(tmp_path, template.replace("LAYOUT", "depth_less")))
    _run(["spirv-as", "--target-env", "vulkan1.1", str(assembly), "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
