"""GLSL output of DXC-shaped SPIR-V tessellation stages declares unsized per-vertex inputs.

DXC sizes a hull or domain shader's per-vertex inputs by the patch, ``OpTypeArray %v4float %uint_3`` for an
``InputPatch<T, 3>``. GLSL 4.60 section 4.3.4 allows a tessellation input array only unsized or sized
``gl_MaxPatchVertices``, so the GLSL output drops the size; the control stage's output keeps it, since
``layout(vertices = 3) out`` must match it.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

import crosstl

CONTROL = """\
               OpCapability Tessellation
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationControl %main "main" %in_pos %out_pos %invocation %outer
               OpExecutionMode %main Triangles
               OpExecutionMode %main SpacingFractionalOdd
               OpExecutionMode %main VertexOrderCw
               OpExecutionMode %main OutputVertices 3
               OpName %main "main"
               OpName %in_pos "in_var_POSITION"
               OpName %out_pos "out_var_POSITION"
               OpDecorate %in_pos Location 0
               OpDecorate %out_pos Location 0
               OpDecorate %invocation BuiltIn InvocationId
               OpDecorate %outer BuiltIn TessLevelOuter
               OpDecorate %outer Patch
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %uint_3 = OpConstant %uint 3
     %uint_4 = OpConstant %uint 4
      %int_0 = OpConstant %int 0
    %float_4 = OpConstant %float 4
    %arr_v4 = OpTypeArray %v4float %uint_3
  %arr_outer = OpTypeArray %float %uint_4
 %ptr_in_arr = OpTypePointer Input %arr_v4
%ptr_out_arr = OpTypePointer Output %arr_v4
 %ptr_in_int = OpTypePointer Input %int
%ptr_out_outer = OpTypePointer Output %arr_outer
  %ptr_in_v4 = OpTypePointer Input %v4float
 %ptr_out_v4 = OpTypePointer Output %v4float
%ptr_out_float = OpTypePointer Output %float
     %in_pos = OpVariable %ptr_in_arr Input
    %out_pos = OpVariable %ptr_out_arr Output
 %invocation = OpVariable %ptr_in_int Input
      %outer = OpVariable %ptr_out_outer Output
       %main = OpFunction %void None %fn
      %label = OpLabel
         %id = OpLoad %int %invocation
        %src = OpAccessChain %ptr_in_v4 %in_pos %id
      %value = OpLoad %v4float %src
        %dst = OpAccessChain %ptr_out_v4 %out_pos %id
               OpStore %dst %value
     %level0 = OpAccessChain %ptr_out_float %outer %int_0
               OpStore %level0 %float_4
               OpReturn
               OpFunctionEnd
"""

EVALUATION = """\
               OpCapability Tessellation
               OpMemoryModel Logical GLSL450
               OpEntryPoint TessellationEvaluation %main "main" %in_pos %coord %out_pos
               OpExecutionMode %main Triangles
               OpExecutionMode %main SpacingEqual
               OpExecutionMode %main VertexOrderCcw
               OpName %main "main"
               OpName %in_pos "in_var_POSITION"
               OpName %out_pos "out_var_POSITION"
               OpDecorate %in_pos Location 0
               OpDecorate %coord BuiltIn TessCoord
               OpDecorate %out_pos Location 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v3float = OpTypeVector %float 3
    %v4float = OpTypeVector %float 4
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %uint_3 = OpConstant %uint 3
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
     %arr_v4 = OpTypeArray %v4float %uint_3
 %ptr_in_arr = OpTypePointer Input %arr_v4
  %ptr_in_v3 = OpTypePointer Input %v3float
  %ptr_in_v4 = OpTypePointer Input %v4float
 %ptr_out_v4 = OpTypePointer Output %v4float
     %in_pos = OpVariable %ptr_in_arr Input
      %coord = OpVariable %ptr_in_v3 Input
    %out_pos = OpVariable %ptr_out_v4 Output
       %main = OpFunction %void None %fn
      %label = OpLabel
         %uvw = OpLoad %v3float %coord
           %u = OpCompositeExtract %float %uvw 0
           %v = OpCompositeExtract %float %uvw 1
         %p0p = OpAccessChain %ptr_in_v4 %in_pos %int_0
          %p0 = OpLoad %v4float %p0p
         %p1p = OpAccessChain %ptr_in_v4 %in_pos %int_1
          %p1 = OpLoad %v4float %p1p
          %a = OpVectorTimesScalar %v4float %p0 %u
          %b = OpVectorTimesScalar %v4float %p1 %v
        %sum = OpFAdd %v4float %a %b
               OpStore %out_pos %sum
               OpReturn
               OpFunctionEnd
"""


def _translate(tmp_path, source, backend="opengl"):
    path = tmp_path / "stage.spvasm"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend, format_output=False)


def _glslang(tmp_path, code, suffix):
    if shutil.which("glslangValidator") is None:
        return
    path = tmp_path / f"checked.{suffix}"
    path.write_text(code)
    result = subprocess.run(
        ["glslangValidator", str(path)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_control_stage_input_is_unsized_and_its_output_keeps_the_patch_size(
    tmp_path,
):
    code = _translate(tmp_path, CONTROL)

    assert "layout(location = 0) in vec4 in_var_POSITION[];" in code
    assert "layout(location = 0) out vec4 out_var_POSITION[3];" in code
    assert "layout(vertices = 3) out;" in code
    _glslang(tmp_path, code, "tesc")


def test_evaluation_stage_input_is_unsized(tmp_path):
    code = _translate(tmp_path, EVALUATION)

    assert "layout(location = 0) in vec4 in_var_POSITION[];" in code
    assert "in_var_POSITION[3]" not in code
    _glslang(tmp_path, code, "tese")


@pytest.mark.parametrize("source", [CONTROL, EVALUATION], ids=["control", "evaluation"])
def test_crossgl_keeps_the_patch_size(tmp_path, source):
    code = _translate(tmp_path, source, backend="cgl")

    assert "in_var_POSITION[3] @input @location(0);" in code
