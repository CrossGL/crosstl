"""A SPIR-V ``Restrict`` decoration on a storage buffer's variable survives import.

The coverage probe for ``restrict-buffer`` (SPIR-V input, ``StorageClass.StorageBuffer``
``Decoration.Restrict``) compiles GLSL ``restrict`` storage buffers with glslang, which
decorates both the variable and each block member with ``Restrict``, and checks that the
round trip through CrossGL still carries the decoration. The importer used to drop it
(#3844). These tests cover the decoration on the variable, alone and in glslang's form;
``test_buffer_memory_qualifiers.py`` covers members alone and a partly decorated block.
"""

from __future__ import annotations

import re

import pytest

import crosstl

_ASSEMBLY = """; SPIR-V
; Version: 1.3
; Generator: Khronos Glslang Reference Front End; 11
; Bound: 40
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %4 "main" %11
               OpExecutionMode %4 LocalSize 1 1 1
               OpSource GLSL 460
               OpName %4 "main"
               OpName %11 "gl_GlobalInvocationID"
               OpName %20 "B0"
               OpMemberName %20 0 "fin"
               OpName %22 ""
               OpName %30 "B1"
               OpMemberName %30 0 "outp"
               OpName %32 ""
               OpDecorate %11 BuiltIn GlobalInvocationId
               OpDecorate %19 ArrayStride 4
               OpDecorate %20 Block
               OpMemberDecorate %20 0 Offset 0
{member0}
               OpDecorate %22 Binding 0
               OpDecorate %22 DescriptorSet 0
{variable0}
               OpDecorate %29 ArrayStride 4
               OpDecorate %30 Block
               OpMemberDecorate %30 0 Offset 0
{member1}
               OpDecorate %32 Binding 1
               OpDecorate %32 DescriptorSet 0
{variable1}
               OpDecorate %39 BuiltIn WorkgroupSize
          %2 = OpTypeVoid
          %3 = OpTypeFunction %2
          %6 = OpTypeInt 32 0
          %9 = OpTypeVector %6 3
         %10 = OpTypePointer Input %9
         %11 = OpVariable %10 Input
         %12 = OpConstant %6 0
         %13 = OpTypePointer Input %6
         %16 = OpTypeFloat 32
         %19 = OpTypeRuntimeArray %16
         %20 = OpTypeStruct %19
         %21 = OpTypePointer StorageBuffer %20
         %22 = OpVariable %21 StorageBuffer
         %24 = OpConstant %16 2
         %25 = OpTypePointer StorageBuffer %16
         %29 = OpTypeRuntimeArray %16
         %30 = OpTypeStruct %29
         %31 = OpTypePointer StorageBuffer %30
         %32 = OpVariable %31 StorageBuffer
         %38 = OpConstant %6 1
         %39 = OpConstantComposite %9 %38 %38 %38
          %4 = OpFunction %2 None %3
          %5 = OpLabel
         %14 = OpAccessChain %13 %11 %12
         %15 = OpLoad %6 %14
         %26 = OpAccessChain %25 %22 %12 %15
         %27 = OpLoad %16 %26
         %28 = OpFMul %16 %27 %24
         %33 = OpAccessChain %25 %32 %12 %15
               OpStore %33 %28
               OpReturn
               OpFunctionEnd
"""

# glslang writes the decoration on the variable and on every member; the
# variable's decoration on its own is also valid SPIR-V for the same declaration.
_DECORATIONS = {
    "variable": ("               OpDecorate %{var} Restrict", ""),
    "both": (
        "               OpDecorate %{var} Restrict",
        "               OpMemberDecorate %{block} 0 Restrict",
    ),
}


def _assembly(style):
    variable, member = _DECORATIONS[style]
    return _ASSEMBLY.format(
        variable0=variable.format(var=22),
        member0=member.format(block=20),
        variable1=variable.format(var=32),
        member1=member.format(block=30),
    )


def _translate(tmp_path, style, backend):
    path = tmp_path / "restrict_buffer.spvasm"
    path.write_text(_assembly(style), encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize("style", ["variable", "both"])
def test_restrict_buffers_import_as_restrict_crossgl(tmp_path, style):
    crossgl = _translate(tmp_path, style, "cgl")
    assert crossgl.count("@restrict") == 2, crossgl


@pytest.mark.parametrize("style", ["variable", "both"])
def test_spirv_round_trip_keeps_restrict_decoration(tmp_path, style):
    assembly = _translate(tmp_path, style, "vulkan")
    decorations = re.findall(
        r"^\s*Op(?:Member)?Decorate %\w+(?: \d+)? Restrict\b", assembly, re.M
    )
    assert len(decorations) == 2, assembly


def test_buffers_without_restrict_stay_unqualified(tmp_path):
    path = tmp_path / "plain_buffer.spvasm"
    path.write_text(
        _ASSEMBLY.format(member0="", variable0="", member1="", variable1=""),
        encoding="utf-8",
    )
    crossgl = crosstl.translate(str(path), backend="cgl", format_output=False)
    assert "@restrict" not in crossgl
    assembly = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert "Restrict" not in assembly
