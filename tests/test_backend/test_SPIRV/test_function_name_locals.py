"""SPIR-V input gives a local or parameter that shares its name with a module function a name of its own.

glslang names a ``Function``-storage ``OpVariable`` after its source variable, and the variable may share its name
with a function of the module. C-family scoping lets a local hide the function for the rest of its scope, so the
imported ``float twice; ... twice(x)`` failed in GLSL ("can't use function syntax on variable"), HLSL and Metal.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

SOURCE = """; SPIR-V
; Version: 1.3
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 64
; Schema: 0
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 1 1 1
               OpName %main "main"
               OpName %twice_f1_ "twice(f1;"
               OpName %x "x"
               OpName %apply_f1_ "apply(f1;"
               OpName %scale "scale"
               OpName %twice "twice"
               OpName %param "param"
               OpName %twice_0 "twice"
               OpName %apply "apply"
               OpName %gid "gl_GlobalInvocationID"
               OpName %Out "Out"
               OpName %out ""
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %arr ArrayStride 4
               OpMemberDecorate %Out 0 Offset 0
               OpDecorate %Out Block
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 0
       %void = OpTypeVoid
     %fvoid = OpTypeFunction %void
      %float = OpTypeFloat 32
  %ptr_float = OpTypePointer Function %float
    %ffloat = OpTypeFunction %float %ptr_float
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
   %ptr_gid = OpTypePointer Input %v3uint
     %uint_0 = OpConstant %uint 0
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
   %ptr_uint = OpTypePointer Input %uint
        %arr = OpTypeRuntimeArray %float
        %Out = OpTypeStruct %arr
    %ptr_out = OpTypePointer StorageBuffer %Out
%ptr_sb_float = OpTypePointer StorageBuffer %float
    %float_1 = OpConstant %float 1
    %float_2 = OpConstant %float 2
    %float_4 = OpConstant %float 4
        %gid = OpVariable %ptr_gid Input
        %out = OpVariable %ptr_out StorageBuffer
       %main = OpFunction %void None %fvoid
         %10 = OpLabel
      %param = OpVariable %ptr_float Function
    %twice_0 = OpVariable %ptr_float Function
      %apply = OpVariable %ptr_float Function
         %11 = OpAccessChain %ptr_uint %gid %uint_0
          %i = OpLoad %uint %11
         %12 = OpConvertUToF %float %i
               OpStore %param %12
         %13 = OpFunctionCall %float %twice_f1_ %param
               OpStore %twice_0 %13
         %14 = OpFunctionCall %float %apply_f1_ %twice_0
               OpStore %apply %14
         %15 = OpLoad %float %apply
         %16 = OpAccessChain %ptr_sb_float %out %int_0 %i
               OpStore %16 %15
               OpReturn
               OpFunctionEnd
  %twice_f1_ = OpFunction %float None %ffloat
          %x = OpFunctionParameter %ptr_float
         %20 = OpLabel
         %21 = OpLoad %float %x
         %22 = OpFMul %float %21 %float_2
               OpReturnValue %22
               OpFunctionEnd
  %apply_f1_ = OpFunction %float None %ffloat
      %scale = OpFunctionParameter %ptr_float
         %30 = OpLabel
      %twice = OpVariable %ptr_float Function
         %31 = OpLoad %float %scale
         %32 = OpFAdd %float %31 %float_1
               OpStore %twice %32
         %33 = OpLoad %float %twice
         %34 = OpFMul %float %33 %float_4
               OpReturnValue %34
               OpFunctionEnd
"""


@pytest.fixture
def source_path(tmp_path):
    path = tmp_path / "probe.spvasm"
    path.write_text(SOURCE, encoding="utf-8")
    return path


def test_locals_named_like_a_module_function_are_renamed(source_path):
    crossgl = crosstl.translate(str(source_path), backend="cgl")
    # The functions keep their names; the locals that shared them take the first free name_N.
    assert re.search(r"float twice\(float x\)", crossgl)
    assert re.search(r"float apply\(float scale\)", crossgl)
    assert not re.search(r"float (twice|apply);", crossgl)
    assert "float twice_1;" in crossgl
    assert "float apply_1;" in crossgl
    assert re.search(r"twice_1 = twice\(param\);", crossgl)
    assert re.search(r"apply_1 = apply\(twice_1\);", crossgl)


@pytest.mark.skipif(
    shutil.which("glslangValidator") is None, reason="glslangValidator is not installed"
)
def test_renamed_locals_compile_as_glsl(source_path, tmp_path):
    glsl = crosstl.translate(str(source_path), backend="opengl")
    path = tmp_path / "probe.comp"
    path.write_text(glsl, encoding="utf-8")
    result = subprocess.run(
        ["glslangValidator", "-S", "comp", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(
    shutil.which("spirv-as") is None or shutil.which("spirv-val") is None,
    reason="SPIRV-Tools is not installed",
)
def test_renamed_locals_round_trip_validates(source_path, tmp_path):
    assembly = crosstl.translate(str(source_path), backend="vulkan")
    path = tmp_path / "round_trip.spvasm"
    path.write_text(assembly, encoding="utf-8")
    binary = tmp_path / "round_trip.spv"
    for command in (
        ["spirv-as", "--target-env", "vulkan1.1", str(path), "-o", str(binary)],
        ["spirv-val", "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
