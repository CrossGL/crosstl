"""Shader-scope functions that name a stage's declarations become functions of the stages that call them.

The SPIR-V importer writes an entry point's helpers at shader scope and its interface variables in the stage block.
A helper that read a stage input became ``input.inUV`` in an HLSL or Metal function with no ``input`` (DXC and the
Metal compiler reject it), and the SPIR-V generator refused the unknown variable.
"""

import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser
from crosstl.translator.semantics.stage_bound_functions import bind_stage_functions

SHARED_HELPER = """
shader S {
    fragment {
        float2 inUV @input @location(0);
        float4 outColor @output @location(0);
        void main() {
            outColor = float4(outer(), inUV.y, 0.0, 1.0);
        }
    }
    float scale(float value) {
        return value * 0.5;
    }
    float inner(float k) {
        return inUV.x * k;
    }
    float outer() {
        return scale(inner(2.0));
    }
}
"""

TWO_STAGES = """
shader S {
    vertex {
        float3 position @input @location(0);
        float2 uv @input @location(1);
        float2 vUV @output @location(0);
        void main() {
            vUV = flip();
            gl_Position = float4(position, 1.0);
        }
    }
    fragment {
        float2 uv @input @location(0);
        float4 outColor @output @location(0);
        void main() {
            outColor = float4(flip(), 0.0, 1.0);
        }
    }
    float2 flip() {
        return uv.yx;
    }
}
"""

STAGE_LOCAL_TWINS = """
shader S {
    vertex {
        float3 position @input @location(0);
        float2 uv @input @location(1);
        float2 vUV @output @location(0);
        float2 flip() { return uv.yx; }
        void main() {
            vUV = flip();
            gl_Position = float4(position, 1.0);
        }
    }
    fragment {
        float2 uv @input @location(0);
        float4 outColor @output @location(0);
        float2 flip() { return uv.yx; }
        void main() {
            outColor = float4(flip(), 0.0, 1.0);
        }
    }
}
"""

SHADOWED = """
shader S {
    float bias = 0.25;
    fragment {
        float bias @input @location(0);
        float4 outColor @output @location(0);
        void main() {
            outColor = float4(shifted(1.0), 0.0, 0.0, 1.0);
        }
    }
    float shifted(float value) {
        return value + bias;
    }
}
"""

PARTLY_RESOLVED = """
shader S {
    vertex {
        float3 position @input @location(0);
        float vTint @output @location(0);
        void main() {
            vTint = weight();
            gl_Position = float4(position, 1.0);
        }
    }
    fragment {
        float tint @input @location(0);
        float4 outColor @output @location(0);
        void main() {
            outColor = float4(weight(), 0.0, 0.0, 1.0);
        }
    }
    float weight() {
        return tint;
    }
}
"""

# glslangValidator -V of:
#   layout(location = 0) in vec2 inUV; layout(location = 0) out vec4 outColor;
#   float shade() { return inUV.x * 2.0; }
#   void main() { outColor = vec4(shade(), inUV.y, 0.0, 1.0); }
SPIRV_HELPER = """
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %inUV %outColor
               OpExecutionMode %main OriginUpperLeft
               OpSource GLSL 450
               OpName %main "main"
               OpName %shade_ "shade("
               OpName %inUV "inUV"
               OpName %outColor "outColor"
               OpDecorate %inUV Location 0
               OpDecorate %outColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
          %7 = OpTypeFunction %float
    %v2float = OpTypeVector %float 2
%_ptr_Input_v2float = OpTypePointer Input %v2float
       %inUV = OpVariable %_ptr_Input_v2float Input
       %uint = OpTypeInt 32 0
     %uint_0 = OpConstant %uint 0
%_ptr_Input_float = OpTypePointer Input %float
    %float_2 = OpConstant %float 2
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
     %uint_1 = OpConstant %uint 1
    %float_0 = OpConstant %float 0
    %float_1 = OpConstant %float 1
       %main = OpFunction %void None %3
          %5 = OpLabel
         %25 = OpFunctionCall %float %shade_
         %27 = OpAccessChain %_ptr_Input_float %inUV %uint_1
         %28 = OpLoad %float %27
         %31 = OpCompositeConstruct %v4float %25 %28 %float_0 %float_1
               OpStore %outColor %31
               OpReturn
               OpFunctionEnd
     %shade_ = OpFunction %float None %7
          %9 = OpLabel
         %16 = OpAccessChain %_ptr_Input_float %inUV %uint_0
         %17 = OpLoad %float %16
         %19 = OpFMul %float %17 %float_2
               OpReturnValue %19
               OpFunctionEnd
"""


def _parse(source):
    return Parser(Lexer(source).get_tokens()).parse()


def _translate(tmp_path, source, backend, suffix=".cgl"):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _flat(code):
    return " ".join(code.split())


def _stage(program, name):
    return next(
        stage
        for stage in program.stages.values()
        if str(getattr(stage.stage, "value", stage.stage)).lower() == name
    )


def _local_names(stage):
    return [function.name for function in stage.local_functions]


def _run(command, cwd):
    result = subprocess.run(
        command, capture_output=True, text=True, cwd=cwd, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _validate(tmp_path, code, backend):
    """Compile the output with the native tool when it is installed."""
    if backend == "directx" and shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "ps_6_0", "-E", "PSMain", "out.hlsl"], tmp_path)
    elif backend == "metal" and shutil.which("xcrun"):
        (tmp_path / "out.metal").write_text(code)
        _run(
            ["xcrun", "-sdk", "macosx", "metal", "-c", "out.metal", "-o", "out.air"],
            tmp_path,
        )
    elif backend == "vulkan" and shutil.which("spirv-as") and shutil.which("spirv-val"):
        (tmp_path / "out.spvasm").write_text(code)
        _run(["spirv-as", "out.spvasm", "-o", "out.spv"], tmp_path)
        _run(["spirv-val", "out.spv"], tmp_path)
    elif backend == "opengl" and shutil.which("glslangValidator"):
        (tmp_path / "out.frag").write_text(code)
        _run(["glslangValidator", "-S", "frag", "out.frag"], tmp_path)


def test_helpers_that_reach_a_stage_input_move_into_the_stage():
    program = _parse(SHARED_HELPER)
    fragment = _stage(program, "fragment")
    assert bind_stage_functions(program) == 2
    assert _local_names(fragment) == ["inner", "outer"]
    # ``scale`` names nothing of the stage, so every stage can still call it at shader scope.
    assert [function.name for function in program.functions] == ["scale"]
    assert all(function.parent is fragment for function in fragment.local_functions)


def test_a_program_without_stage_names_in_its_helpers_is_unchanged():
    source = SHARED_HELPER.replace("return inUV.x * k;", "return k;")
    program = _parse(source)
    assert bind_stage_functions(program) == 0
    assert [f.name for f in program.functions] == ["scale", "inner", "outer"]


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_hlsl_and_metal_pass_the_input_structure_to_the_helpers(tmp_path, backend):
    code = _translate(tmp_path, SHARED_HELPER, backend)
    flat = _flat(code)
    assert "float inner(float k, FragmentInput input)" in flat
    assert "float outer(FragmentInput input)" in flat
    assert "inner(2.0, input)" in flat
    assert "outer(input)" in flat
    _validate(tmp_path, code, backend)


@pytest.mark.parametrize("backend", ["vulkan", "opengl"])
def test_spirv_and_glsl_resolve_the_helper_names(tmp_path, backend):
    code = _translate(tmp_path, SHARED_HELPER, backend)
    assert ("OpFunctionCall" if backend == "vulkan" else "inner(2.0)") in code
    _validate(tmp_path, code, backend)


def test_each_calling_stage_gets_its_own_copy():
    program = _parse(TWO_STAGES)
    assert bind_stage_functions(program) == 2
    assert _local_names(_stage(program, "vertex")) == ["flip"]
    assert _local_names(_stage(program, "fragment")) == ["flip"]
    assert program.functions == []
    vertex_copy = _stage(program, "vertex").local_functions[0]
    fragment_copy = _stage(program, "fragment").local_functions[0]
    assert vertex_copy is not fragment_copy
    assert vertex_copy.body is not fragment_copy.body


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_each_stage_copy_reads_its_own_inputs(tmp_path, backend):
    flat = _flat(_translate(tmp_path, TWO_STAGES, backend))
    assert "float2 flip(VertexInput input)" in flat
    assert "float2 flip(FragmentInput input)" in flat


@pytest.mark.parametrize("backend", ["directx", "metal"])
def test_same_named_stage_helpers_take_their_own_stage_input(tmp_path, backend):
    code = _translate(tmp_path, STAGE_LOCAL_TWINS, backend)
    flat = _flat(code)
    assert "float2 flip(VertexInput input)" in flat
    assert "float2 flip(FragmentInput input)" in flat
    if backend == "directx" and shutil.which("dxc"):
        (tmp_path / "out.hlsl").write_text(code)
        _run(["dxc", "-T", "vs_6_0", "-E", "VSMain", "out.hlsl"], tmp_path)
    _validate(tmp_path, code, backend)


def test_a_shader_scope_declaration_of_the_name_wins():
    program = _parse(SHADOWED)
    assert bind_stage_functions(program) == 0
    assert [f.name for f in program.functions] == ["shifted"]


def test_a_helper_stays_for_a_stage_that_does_not_declare_its_names():
    program = _parse(PARTLY_RESOLVED)
    assert bind_stage_functions(program) == 1
    assert _local_names(_stage(program, "fragment")) == ["weight"]
    assert _local_names(_stage(program, "vertex")) == []
    assert [f.name for f in program.functions] == ["weight"]


@pytest.mark.parametrize("backend", ["directx", "metal", "vulkan"])
def test_spirv_helpers_that_read_interface_variables(tmp_path, backend):
    code = _translate(tmp_path, SPIRV_HELPER, backend, suffix=".spvasm")
    if backend != "vulkan":
        flat = _flat(code)
        assert "float shade(FragmentInput input)" in flat
        assert "shade(input)" in flat
    _validate(tmp_path, code, backend)
