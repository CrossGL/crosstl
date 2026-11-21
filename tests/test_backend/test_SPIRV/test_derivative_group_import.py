"""SPIR-V input keeps the derivative group of a compute entry (#4840).

``DerivativeGroupQuadsKHR`` and ``DerivativeGroupLinearKHR`` (SPV_KHR_compute_shader_derivatives; the NV names are
aliases) decide which invocations form the quads of a compute, mesh or task entry's derivatives and quad operations.
The importer dropped them, so the CrossGL stage meant subgroup quads, and an 8x8 entry that declared the linear group
came back from SPIR-V output with the 2x2 group. They now import as ``layout(derivative_group_quadsKHR) in;`` and
``layout(derivative_group_linearKHR) in;``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

MODULE = """; SPIR-V
; Version: 1.3
               OpCapability Shader
               OpCapability {capability}
               OpExtension "SPV_KHR_compute_shader_derivatives"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationIndex
               OpExecutionMode %main LocalSize 8 8 1
               OpExecutionMode %main {mode}
               OpDecorate %gl_LocalInvocationIndex BuiltIn LocalInvocationIndex
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %Out 0 Offset 0
               OpDecorate %Out Block
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
      %float = OpTypeFloat 32
 %_ptr_Input_uint = OpTypePointer Input %uint
 %gl_LocalInvocationIndex = OpVariable %_ptr_Input_uint Input
 %_runtimearr_float = OpTypeRuntimeArray %float
        %Out = OpTypeStruct %_runtimearr_float
 %_ptr_StorageBuffer_Out = OpTypePointer StorageBuffer %Out
        %out = OpVariable %_ptr_StorageBuffer_Out StorageBuffer
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
 %_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %gi = OpLoad %uint %gl_LocalInvocationIndex
          %v = OpConvertUToF %float %gi
         %dy = OpDPdy %float %v
        %ptr = OpAccessChain %_ptr_StorageBuffer_float %out %int_0 %gi
               OpStore %ptr %dy
               OpReturn
               OpFunctionEnd
"""

GROUPS = [
    ("DerivativeGroupQuadsKHR", "ComputeDerivativeGroupQuadsKHR", "quads"),
    ("DerivativeGroupLinearKHR", "ComputeDerivativeGroupLinearKHR", "linear"),
    ("DerivativeGroupQuadsNV", "ComputeDerivativeGroupQuadsNV", "quads"),
    ("DerivativeGroupLinearNV", "ComputeDerivativeGroupLinearNV", "linear"),
]


def _translate(tmp_path, source, suffix, backend):
    path = tmp_path / f"probe.{suffix}"
    path.write_text(source, encoding="utf-8")
    return crosstl.translate(str(path), backend=backend, format_output=False)


@pytest.mark.parametrize(("mode", "capability", "group"), GROUPS)
def test_the_derivative_group_imports_as_the_stage_layout(
    tmp_path, mode, capability, group
):
    crossgl = _translate(
        tmp_path, MODULE.format(mode=mode, capability=capability), "spvasm", "cgl"
    )

    assert re.search(
        rf"compute \{{\s*layout\(local_size_x = 8, local_size_y = 8, local_size_z = 1\) in;\s*"
        rf"layout\(derivative_group_{group}KHR\) in;",
        crossgl,
    ), crossgl


@pytest.mark.parametrize(("mode", "capability", "group"), GROUPS[:2])
def test_the_group_survives_a_round_trip(tmp_path, mode, capability, group):
    crossgl = _translate(
        tmp_path, MODULE.format(mode=mode, capability=capability), "spvasm", "cgl"
    )
    assembly = _translate(tmp_path, crossgl, "cgl", "vulkan")

    expected = "Quads" if group == "quads" else "Linear"
    assert re.findall(r"OpExecutionMode %\w+ (DerivativeGroup\w+)", assembly) == [
        f"DerivativeGroup{expected}KHR"
    ]
    assert "OpDPdy" in assembly
    assembler, validator = shutil.which("spirv-as"), shutil.which("spirv-val")
    if not (assembler and validator):
        pytest.skip("SPIRV-Tools is not installed")
    source, binary = tmp_path / "out.spvasm", tmp_path / "out.spv"
    source.write_text(assembly, encoding="utf-8")
    for command in (
        [assembler, "--target-env", "vulkan1.1", str(source), "-o", str(binary)],
        [validator, "--target-env", "vulkan1.1", str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr or result.stdout


def test_an_entry_without_a_group_imports_no_layout(tmp_path):
    source = MODULE.format(mode="LocalSize 8 8 1", capability="Shader")
    crossgl = _translate(tmp_path, source, "spvasm", "cgl")

    assert "derivative_group" not in crossgl
