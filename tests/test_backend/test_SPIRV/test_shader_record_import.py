"""A SPIR-V ``ShaderRecordBufferKHR`` variable imports as a CrossGL shader record buffer.

The importer declared nothing for the storage class, so the stage body read an undeclared ``_``. See
:mod:`crosstl.translator.semantics.shader_records`.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

MISS = """; SPIR-V
; Version: 1.4
               OpCapability RayTracingKHR
               OpExtension "SPV_KHR_ray_tracing"
               OpMemoryModel Logical GLSL450
               OpEntryPoint MissKHR %main "main" %hitValue %_
               OpName %main "main"
               OpName %hitValue "hitValue"
               OpName %SBT "SBT"
               OpMemberName %SBT 0 "r"
               OpMemberName %SBT 1 "g"
               OpName %_ ""
               OpDecorate %hitValue Location 0
               OpMemberDecorate %SBT 0 Offset 0
               OpMemberDecorate %SBT 1 Offset 4
               OpDecorate %SBT Block
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
      %float = OpTypeFloat 32
    %v2float = OpTypeVector %float 2
        %int = OpTypeInt 32 1
%_ptr_IncomingRayPayloadKHR_v2float = OpTypePointer IncomingRayPayloadKHR %v2float
   %hitValue = OpVariable %_ptr_IncomingRayPayloadKHR_v2float IncomingRayPayloadKHR
        %SBT = OpTypeStruct %float %float
%_ptr_ShaderRecordBufferKHR_SBT = OpTypePointer ShaderRecordBufferKHR %SBT
          %_ = OpVariable %_ptr_ShaderRecordBufferKHR_SBT ShaderRecordBufferKHR
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
%_ptr_ShaderRecordBufferKHR_float = OpTypePointer ShaderRecordBufferKHR %float
       %main = OpFunction %void None %3
          %5 = OpLabel
         %10 = OpAccessChain %_ptr_ShaderRecordBufferKHR_float %_ %int_0
         %11 = OpLoad %float %10
         %12 = OpAccessChain %_ptr_ShaderRecordBufferKHR_float %_ %int_1
         %13 = OpLoad %float %12
         %14 = OpCompositeConstruct %v2float %11 %13
               OpStore %hitValue %14
               OpReturn
               OpFunctionEnd
"""


def _write(tmp_path, source):
    path = tmp_path / "miss.spvasm"
    path.write_text(source)
    return str(path)


def test_shader_record_variable_is_declared(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, MISS), backend="cgl", format_output=False
    )
    assert re.search(
        r"\bSBT\s+(\w+)\s+@glsl_buffer_block\(shaderRecordEXT,\s*std430\)", crossgl
    )
    assert re.search(r"\b_\.r\b", crossgl) and re.search(r"\b_\.g\b", crossgl)


def test_shader_record_round_trips_to_spirv(tmp_path):
    spirv = crosstl.translate(
        _write(tmp_path, MISS), backend="vulkan", format_output=False
    )
    assert re.search(r"OpVariable %\w+ ShaderRecordBufferKHR", spirv)
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    assembly = tmp_path / "out.spvasm"
    assembly.write_text(spirv)
    binary = str(tmp_path / "out.spv")
    subprocess.run(
        ["spirv-as", "--target-env", "vulkan1.2", str(assembly), "-o", binary],
        check=True,
    )
    subprocess.run(["spirv-val", "--target-env", "vulkan1.2", binary], check=True)
