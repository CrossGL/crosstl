"""WGSL entry points take and return the stage-level inputs and outputs a shader declares.

The SPIR-V importer declares a vertex or fragment stage's interface one variable per input or output
(``float4 vColor @input @location(0);``), and CrossGL written the GLSL way declares it at module scope
(``in vec2 uv @location(0);``). WGSL has no module-scope stage variables (WGSL section 13.3.1): an entry point takes
its inputs as parameters and returns its outputs. The WGSL generator declared each as ``var<private>``, which nothing
writes or returns, so a fragment shader translated from SPIR-V read zeros and wrote nothing, and one whose helper read
``gl_FragCoord`` was refused. The stage-interface lowering the HLSL, MSL and Slang outputs use now runs for WGSL too:
the inputs arrive in a ``FragmentInput``/``VertexInput`` structure, the outputs leave in the returned
``FragmentOutput``/``VertexOutput``, and a helper that writes an output takes ``ptr<function, FragmentOutput>``.

SPIR-V's unnamed blocks import as ``_`` and ``__0``; WGSL reserves identifiers that start with two underscores, and
the escaped name of ``__0`` must not start with a digit.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

FRAGMENT_SPIRV = """
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %outColor %vColor %gl_FragCoord
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %outColor "outColor"
               OpName %vColor "vColor"
               OpName %gl_FragCoord "gl_FragCoord"
               OpDecorate %outColor Location 0
               OpDecorate %vColor Location 0
               OpDecorate %gl_FragCoord BuiltIn FragCoord
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
%_ptr_Input_v4float = OpTypePointer Input %v4float
     %vColor = OpVariable %_ptr_Input_v4float Input
  %float_0_5 = OpConstant %float 0.5
%gl_FragCoord = OpVariable %_ptr_Input_v4float Input
       %main = OpFunction %void None %3
          %5 = OpLabel
         %12 = OpLoad %v4float %vColor
         %14 = OpVectorTimesScalar %v4float %12 %float_0_5
         %16 = OpLoad %v4float %gl_FragCoord
         %17 = OpFAdd %v4float %14 %16
               OpStore %outColor %17
               OpReturn
               OpFunctionEnd
"""

# glslang -V of: vec4 shade(float k) { return vColor * k; } void writeOut(vec4 c) { outColor = c; }
# void main() { writeOut(shade(0.5)); }
HELPER_FRAGMENT_SPIRV = """
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint Fragment %main "main" %vColor %outColor
               OpExecutionMode %main OriginUpperLeft
               OpName %main "main"
               OpName %shade_f1_ "shade(f1;"
               OpName %writeOut_vf4_ "writeOut(vf4;"
               OpName %vColor "vColor"
               OpName %outColor "outColor"
               OpDecorate %vColor Location 0
               OpDecorate %outColor Location 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
%_ptr_Function_float = OpTypePointer Function %float
    %v4float = OpTypeVector %float 4
          %9 = OpTypeFunction %v4float %_ptr_Function_float
%_ptr_Function_v4float = OpTypePointer Function %v4float
         %14 = OpTypeFunction %void %_ptr_Function_v4float
%_ptr_Input_v4float = OpTypePointer Input %v4float
     %vColor = OpVariable %_ptr_Input_v4float Input
%_ptr_Output_v4float = OpTypePointer Output %v4float
   %outColor = OpVariable %_ptr_Output_v4float Output
  %float_0_5 = OpConstant %float 0.5
       %main = OpFunction %void None %3
          %5 = OpLabel
      %param = OpVariable %_ptr_Function_float Function
    %param_0 = OpVariable %_ptr_Function_v4float Function
               OpStore %param %float_0_5
         %30 = OpFunctionCall %v4float %shade_f1_ %param
               OpStore %param_0 %30
         %32 = OpFunctionCall %void %writeOut_vf4_ %param_0
               OpReturn
               OpFunctionEnd
  %shade_f1_ = OpFunction %v4float None %9
          %k = OpFunctionParameter %_ptr_Function_float
         %12 = OpLabel
         %20 = OpLoad %v4float %vColor
         %21 = OpLoad %float %k
         %22 = OpVectorTimesScalar %v4float %20 %21
               OpReturnValue %22
               OpFunctionEnd
%writeOut_vf4_ = OpFunction %void None %14
          %c = OpFunctionParameter %_ptr_Function_v4float
         %17 = OpLabel
         %27 = OpLoad %v4float %c
               OpStore %outColor %27
               OpReturn
               OpFunctionEnd
"""

# glslang -V of a compute shader whose two storage blocks have no instance name, which spirv-dis spells %_ and
# %__0.
UNNAMED_BLOCKS_SPIRV = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
               OpName %B0 "B0"
               OpMemberName %B0 0 "iin"
               OpName %_ ""
               OpName %B1 "B1"
               OpMemberName %B1 0 "outp"
               OpName %__0 ""
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %_runtimearr_int ArrayStride 4
               OpDecorate %B0 Block
               OpMemberDecorate %B0 0 Offset 0
               OpDecorate %_ Binding 0
               OpDecorate %_ DescriptorSet 0
               OpDecorate %_runtimearr_float ArrayStride 4
               OpDecorate %B1 Block
               OpMemberDecorate %B1 0 Offset 0
               OpDecorate %__0 Binding 1
               OpDecorate %__0 DescriptorSet 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
      %float = OpTypeFloat 32
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
      %int_0 = OpConstant %int 0
%_ptr_Input_uint = OpTypePointer Input %uint
%_runtimearr_int = OpTypeRuntimeArray %int
         %B0 = OpTypeStruct %_runtimearr_int
%_ptr_StorageBuffer_B0 = OpTypePointer StorageBuffer %B0
          %_ = OpVariable %_ptr_StorageBuffer_B0 StorageBuffer
%_runtimearr_float = OpTypeRuntimeArray %float
         %B1 = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer_B1 = OpTypePointer StorageBuffer %B1
        %__0 = OpVariable %_ptr_StorageBuffer_B1 StorageBuffer
%_ptr_StorageBuffer_int = OpTypePointer StorageBuffer %int
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %12 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
         %13 = OpLoad %uint %12
         %14 = OpAccessChain %_ptr_StorageBuffer_int %_ %int_0 %13
         %15 = OpLoad %int %14
         %16 = OpConvertSToF %float %15
         %17 = OpAccessChain %_ptr_StorageBuffer_float %__0 %int_0 %13
               OpStore %17 %16
               OpReturn
               OpFunctionEnd
"""

MODULE_FRAGMENT = """
shader main {
    in vec3 n @location(1) @flat;
    in vec2 uv @location(0);
    out vec4 a @location(0);
    vec4 shade() {
        return vec4(n * uv.x, 1.0);
    }
    fragment {
        void main() {
            a = shade();
        }
    }
}
"""

MODULE_VERTEX = """
shader main {
    in vec3 pos @location(0);
    out vec2 uv @location(1);
    vertex {
        void main() {
            uv = pos.xy;
            gl_Position = vec4(pos, 1.0);
        }
    }
}
"""


def _translate(tmp_path, source, suffix):
    path = tmp_path / f"shader{suffix}"
    path.write_text(source, encoding="utf-8")
    code = crosstl.translate(str(path), backend="wgsl", format_output=False)
    if shutil.which("naga"):
        (tmp_path / "out.wgsl").write_text(code, encoding="utf-8")
        result = subprocess.run(
            ["naga", "out.wgsl"], cwd=tmp_path, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stdout + result.stderr + code
    return re.sub(r"\s+", " ", code)


def test_spirv_fragment_interface_is_the_entry_points(tmp_path):
    code = _translate(tmp_path, FRAGMENT_SPIRV, ".spvasm")
    assert "var<private>" not in code, code
    assert "struct FragmentInput { @location(0) vColor: vec4<f32>, };" in code, code
    assert "struct FragmentOutput { @location(0) outColor: vec4<f32>, };" in code, code
    assert re.search(
        r"fn fragment_main\(input: FragmentInput, @builtin\(position\) (\w+): vec4<f32>\) -> FragmentOutput",
        code,
    ), code
    assert "output.outColor = ((input.vColor * 0.5) + " in code, code
    assert "return output;" in code, code


def test_spirv_fragment_helpers_take_the_input_and_output_structures(tmp_path):
    code = _translate(tmp_path, HELPER_FRAGMENT_SPIRV, ".spvasm")
    assert "var<private>" not in code, code
    assert "fn shade(k: f32, input: FragmentInput) -> vec4<f32>" in code, code
    assert "return (input.vColor * k);" in code, code
    assert (
        "fn writeOut(c: vec4<f32>, output: ptr<function, FragmentOutput>)" in code
    ), code
    assert "(*output).outColor = c;" in code, code
    assert re.search(r"writeOut\(param_\w+, &output\);", code), code
    assert "-> FragmentOutput" in code and "return output;" in code, code


def test_module_scope_fragment_interface_is_the_entry_points(tmp_path):
    code = _translate(tmp_path, MODULE_FRAGMENT, ".cgl")
    assert "var<private>" not in code, code
    assert "@location(1) @interpolate(flat) n: vec3<f32>," in code, code
    assert "fn shade(input: FragmentInput) -> vec4<f32>" in code, code
    assert "fn fragment_main(input: FragmentInput) -> FragmentOutput" in code, code
    assert "output.a = shade(input);" in code, code


def test_module_scope_vertex_writes_its_position_output(tmp_path):
    code = _translate(tmp_path, MODULE_VERTEX, ".cgl")
    assert "var<private>" not in code, code
    assert "@builtin(position) position: vec4<f32>," in code, code
    assert "@location(1) uv: vec2<f32>," in code, code
    assert "fn vertex_main(input: VertexInput) -> VertexOutput" in code, code
    assert "output.position = vec4<f32>(input.pos, 1.0);" in code, code


def test_unnamed_spirv_blocks_get_valid_wgsl_names(tmp_path):
    code = _translate(tmp_path, UNNAMED_BLOCKS_SPIRV, ".spvasm")
    assert "var<storage, read_write> identifier_: array<i32>;" in code, code
    assert "var<storage, read_write> identifier_0: array<f32>;" in code, code
    assert "identifier_0[" in code, code
    assert not re.search(r"[^\w]\d+_\b", code), code


@pytest.mark.parametrize(
    "name, escaped",
    [("_", "identifier_"), ("__0", "identifier_0"), ("__12", "identifier_12")],
)
def test_escaped_double_underscore_names_start_like_identifiers(name, escaped):
    from crosstl.translator.codegen.wgsl_codegen import WGSLCodeGen

    assert WGSLCodeGen().escaped_wgsl_identifier_base(name) == escaped
    assert WGSLCodeGen().escaped_wgsl_identifier_base("__value") == "value_"
