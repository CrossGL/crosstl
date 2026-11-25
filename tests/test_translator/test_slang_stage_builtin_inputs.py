"""Slang binds a compute stage's declared built-in inputs as entry parameters only, not also as globals.

The SPIR-V importer declares a compute stage's built-in inputs in the stage (``uint3 gl_GlobalInvocationID @input
@gl_GlobalInvocationID;``) and the entry point reads them by that name. The Slang generator binds such a read as an
entry parameter (``uint3 gl_GlobalInvocationID : SV_DispatchThreadID``) or an intrinsic (``WorkgroupCount()``), and
also wrote the declaration as a global. Slang makes a global without ``static`` an implicit uniform shader parameter,
so slangc gave the artifact a constants block (``GlobalParams``) that the host does not bind.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import crosstl

COMPUTE_SPIRV = """
               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_GlobalInvocationID %gl_LocalInvocationID %gl_NumWorkGroups
               OpExecutionMode %main LocalSize 4 1 1
               OpName %main "main"
               OpName %gl_GlobalInvocationID "gl_GlobalInvocationID"
               OpName %gl_LocalInvocationID "gl_LocalInvocationID"
               OpName %gl_NumWorkGroups "gl_NumWorkGroups"
               OpName %B "B"
               OpMemberName %B 0 "v"
               OpName %out "out_"
               OpDecorate %gl_GlobalInvocationID BuiltIn GlobalInvocationId
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %gl_NumWorkGroups BuiltIn NumWorkgroups
               OpDecorate %_runtimearr_uint ArrayStride 4
               OpDecorate %B Block
               OpMemberDecorate %B 0 Offset 0
               OpDecorate %out Binding 0
               OpDecorate %out DescriptorSet 0
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_GlobalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
%gl_NumWorkGroups = OpVariable %_ptr_Input_v3uint Input
     %uint_0 = OpConstant %uint 0
      %int_0 = OpConstant %int 0
%_ptr_Input_uint = OpTypePointer Input %uint
%_runtimearr_uint = OpTypeRuntimeArray %uint
          %B = OpTypeStruct %_runtimearr_uint
%_ptr_StorageBuffer_B = OpTypePointer StorageBuffer %B
        %out = OpVariable %_ptr_StorageBuffer_B StorageBuffer
%_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
       %main = OpFunction %void None %3
          %5 = OpLabel
         %10 = OpAccessChain %_ptr_Input_uint %gl_GlobalInvocationID %uint_0
         %11 = OpLoad %uint %10
         %12 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
         %13 = OpLoad %uint %12
         %14 = OpAccessChain %_ptr_Input_uint %gl_NumWorkGroups %uint_0
         %15 = OpLoad %uint %14
         %16 = OpIAdd %uint %13 %15
         %17 = OpAccessChain %_ptr_StorageBuffer_uint %out %int_0 %11
               OpStore %17 %16
               OpReturn
               OpFunctionEnd
"""


def test_declared_compute_builtins_are_entry_parameters_only(tmp_path):
    path = tmp_path / "probe.spvasm"
    path.write_text(COMPUTE_SPIRV, encoding="utf-8")
    code = crosstl.translate(str(path), backend="slang", format_output=False)
    flat = re.sub(r"\s+", " ", code)
    # No global declaration of a built-in the entry point binds.
    for name in ("gl_GlobalInvocationID", "gl_LocalInvocationID", "gl_NumWorkGroups"):
        assert not re.search(rf"(^|;|\}}) ?uint3 {name};", flat), code
    assert "uint3 gl_GlobalInvocationID : SV_DispatchThreadID" in flat, code
    assert "uint3 gl_LocalInvocationID : SV_GroupThreadID" in flat, code
    assert "WorkgroupCount()" in flat, code
    slangc = shutil.which("slangc")
    if slangc is None:
        return
    source = tmp_path / "probe.slang"
    source.write_text(code, encoding="utf-8")
    # -preserve-params keeps every declared parameter, as the coverage harness compiles; an implicit global would
    # appear as a Uniform GlobalParams block.
    result = subprocess.run(
        [
            slangc,
            str(source),
            "-target",
            "spirv-asm",
            "-entry",
            "main",
            "-stage",
            "compute",
            "-O0",
            "-preserve-params",
            "-o",
            str(tmp_path / "probe.spvasm.out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr + code
    assembly = (tmp_path / "probe.spvasm.out").read_text(encoding="utf-8")
    assert "Uniform" not in assembly, assembly
    assert "implicit global shader parameter" not in result.stderr, result.stderr
