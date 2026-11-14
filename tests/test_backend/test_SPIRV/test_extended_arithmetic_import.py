"""SPIR-V extended integer arithmetic imports as GLSL's carry, borrow and extended-multiply built-ins.

``OpIAddCarry``, ``OpISubBorrow``, ``OpUMulExtended`` and ``OpSMulExtended`` (SPIR-V 1.6 section 3.52.13) return a
two-member struct. The importer used to write ``spirvIAddCarry(x, y)[1]``, which every emitter printed verbatim. They
now import as ``uaddCarry``, ``usubBorrow``, ``umulExtended`` and ``imulExtended`` with out operands, and
``OpCompositeExtract`` reads the locals. Execution is covered by the ``s_extended_arithmetic`` conformance case.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.translator.codegen.SPIRV_codegen import SPIRVBuiltinLoweringError
from crosstl.translator.diagnostics import TranslationDiagnostic

COMPUTE = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 64 1 1
               OpName %main "main"
               OpName %Buf "Buf"
               OpMemberName %Buf 0 "v"
               OpName %buf "buf"
               OpName %Wide "ResType"
               OpName %Pair "ResType"
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %rt ArrayStride 4
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
    %v2uint = OpTypeVector %uint 2
     %v3uint = OpTypeVector %uint 3
       %Wide = OpTypeStruct %uint %uint
       %Pair = OpTypeStruct %v2uint %v2uint
         %rt = OpTypeRuntimeArray %uint
        %Buf = OpTypeStruct %rt
    %ptr_buf = OpTypePointer StorageBuffer %Buf
   %ptr_uint = OpTypePointer StorageBuffer %uint
    %ptr_gid = OpTypePointer Input %v3uint
     %ptr_in = OpTypePointer Input %uint
  %ptr_local = OpTypePointer Function %Wide
      %int_0 = OpConstant %int 0
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
     %uint_4 = OpConstant %uint 4
     %uint_5 = OpConstant %uint 5
        %buf = OpVariable %ptr_buf StorageBuffer
        %gid = OpVariable %ptr_gid Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %local = OpVariable %ptr_local Function
         %10 = OpAccessChain %ptr_in %gid %uint_0
          %i = OpLoad %uint %10
         %11 = OpAccessChain %ptr_uint %buf %int_0 %i
          %x = OpLoad %uint %11
         %12 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
          %y = OpLoad %uint %12
         %20 = OpIAddCarry %Wide %x %y
         %21 = OpCompositeExtract %uint %20 0
         %22 = OpCompositeExtract %uint %20 1
         %23 = OpIAdd %uint %21 %22
         %24 = OpAccessChain %ptr_uint %buf %int_0 %uint_0
               OpStore %24 %23
         %30 = OpSMulExtended %Wide %x %y
         %31 = OpCompositeExtract %uint %30 1
         %32 = OpAccessChain %ptr_uint %buf %int_0 %uint_2
               OpStore %32 %31
         %33 = OpCompositeExtract %uint %30 0
         %34 = OpAccessChain %ptr_uint %buf %int_0 %uint_3
               OpStore %34 %33
         %40 = OpCompositeConstruct %v2uint %x %y
         %41 = OpISubBorrow %Pair %40 %40
         %42 = OpCompositeExtract %uint %41 1 0
         %43 = OpAccessChain %ptr_uint %buf %int_0 %uint_4
               OpStore %43 %42
         %50 = OpUMulExtended %Wide %x %y
         %51 = OpCompositeExtract %uint %50 1
         %52 = OpAccessChain %ptr_uint %buf %int_0 %uint_5
               OpStore %52 %51
               OpReturn
               OpFunctionEnd
"""

# The same kernel, but the carry struct is stored whole into a local before its members are read.
WHOLE_STRUCT = COMPUTE.replace(
    "         %21 = OpCompositeExtract %uint %20 0\n",
    "               OpStore %local %20\n"
    "         %25 = OpLoad %Wide %local\n"
    "         %21 = OpCompositeExtract %uint %25 0\n",
)


def _write(tmp_path, source, name="shader.spvasm"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("source", [COMPUTE, WHOLE_STRUCT])
def test_fixtures_are_valid_spirv(tmp_path, source):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    binary = tmp_path / "shader.spv"
    path = _write(tmp_path, source)
    _run(["spirv-as", "--target-env", "vulkan1.1", path, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_members_become_the_result_and_the_out_operands(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    assert re.search(r"\buint value_20_carry;", crossgl)
    assert re.search(
        r"\buint value_20_sum = uaddCarry\([^;]*, value_20_carry\);", crossgl
    )
    # The multiplications write the high word, then the low word.
    assert re.search(r"\buint value_50_msb;\s*uint value_50_lsb;", crossgl)
    assert re.search(r"umulExtended\([^;]*, value_50_msb, value_50_lsb\);", crossgl)
    # Indices after the member index still apply to a vector member.
    assert re.search(r"\buint2 value_41_borrow;", crossgl)
    assert "value_41_borrow[0]" in crossgl
    assert "spirv" not in crossgl.replace("spirv_", "")


def test_signed_multiplication_with_uint_members_is_reinterpreted(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    # imulExtended takes ints: the uint operands and members are reinterpreted with same-width constructors.
    assert re.search(r"\bint value_30_msb;\s*int value_30_lsb;", crossgl)
    assert re.search(
        r"imulExtended\(int\([^;]*\), int\([^;]*\), value_30_msb, value_30_lsb\);",
        crossgl,
    )
    assert "uint(value_30_msb)" in crossgl and "uint(value_30_lsb)" in crossgl


def test_glsl_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="opengl", format_output=False
    )
    for name in ("uaddCarry", "usubBorrow", "umulExtended", "imulExtended"):
        assert f"{name}(" in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(code, encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_metal_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="metal", format_output=False
    )
    assert "mulhi(" in code
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not on PATH")
    path = tmp_path / "shader.metal"
    path.write_text(code, encoding="utf-8")
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(path) + ".air"]
    )


def test_hlsl_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="directx", format_output=False
    )
    assert "__crossgl_uaddCarry_uint(" in code
    assert "__crossgl_imulExtended_int(" in code
    dxc = shutil.which("dxc")
    if not dxc:
        pytest.skip("dxc is not on PATH")
    path = tmp_path / "shader.hlsl"
    path.write_text(code, encoding="utf-8")
    entry = re.search(r"\]\s*void\s+(\w+)\s*\(", code).group(1)
    _run([dxc, "-T", "cs_6_0", "-E", entry, str(path), "-Fo", str(tmp_path / "o")])


def test_spirv_output_validates(tmp_path):
    assembly = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="vulkan", format_output=False
    )
    for opcode in ("OpIAddCarry", "OpISubBorrow", "OpUMulExtended", "OpSMulExtended"):
        assert opcode in assembly
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = _write(tmp_path, assembly, "round_trip.spvasm")
    binary = tmp_path / "round_trip.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", path, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


@pytest.mark.parametrize("backend", ["metal", "directx", "opengl", "vulkan"])
def test_a_struct_result_used_whole_is_refused(tmp_path, backend):
    refusal = (
        SPIRVBuiltinLoweringError if backend == "vulkan" else TranslationDiagnostic
    )
    with pytest.raises(refusal, match="uaddCarry"):
        crosstl.translate(
            _write(tmp_path, WHOLE_STRUCT), backend=backend, format_output=False
        )
