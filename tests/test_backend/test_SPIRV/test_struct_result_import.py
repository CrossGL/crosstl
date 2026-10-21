"""GLSL.std.450 ``ModfStruct`` and ``FrexpStruct`` import as CrossGL ``modf`` and ``frexp`` with an out operand.

The importer used to write ``modf(x)[1]``, an index into a one-operand call that no emitter accepts, and declared
both of glslang's ``ResType`` structs under one name. Execution on every runtime is covered by the
``s_frexp_ldexp_modf`` conformance case; these tests check the imported CrossGL, native compiles and the refusal of
a struct result used as a whole.
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
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 64 1 1
               OpName %main "main"
               OpName %Buf "Buf"
               OpMemberName %Buf 0 "v"
               OpName %buf "buf"
               OpName %ModfResult "ResType"
               OpName %FrexpResult "ResType"
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %rt ArrayStride 8
               OpMemberDecorate %Buf 0 Offset 0
               OpDecorate %Buf Block
               OpDecorate %buf DescriptorSet 0
               OpDecorate %buf Binding 0
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
    %v2float = OpTypeVector %float 2
      %v2int = OpTypeVector %int 2
     %v3uint = OpTypeVector %uint 3
 %ModfResult = OpTypeStruct %v2float %v2float
%FrexpResult = OpTypeStruct %v2float %v2int
         %rt = OpTypeRuntimeArray %v2float
        %Buf = OpTypeStruct %rt
    %ptr_buf = OpTypePointer StorageBuffer %Buf
    %ptr_v2f = OpTypePointer StorageBuffer %v2float
    %ptr_gid = OpTypePointer Input %v3uint
     %ptr_in = OpTypePointer Input %uint
 %ptr_local = OpTypePointer Function %ModfResult
      %int_0 = OpConstant %int 0
     %uint_0 = OpConstant %uint 0
        %buf = OpVariable %ptr_buf StorageBuffer
        %gid = OpVariable %ptr_gid Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
      %local = OpVariable %ptr_local Function
         %10 = OpAccessChain %ptr_in %gid %uint_0
          %i = OpLoad %uint %10
         %11 = OpAccessChain %ptr_v2f %buf %int_0 %i
          %x = OpLoad %v2float %11
         %20 = OpExtInst %ModfResult %1 ModfStruct %x
         %21 = OpCompositeExtract %float %20 1 1
         %22 = OpCompositeExtract %v2float %20 0
         %30 = OpExtInst %FrexpResult %1 FrexpStruct %x
         %31 = OpCompositeExtract %v2float %30 0
         %32 = OpCompositeExtract %int %30 1 0
         %33 = OpConvertSToF %float %32
         %34 = OpVectorTimesScalar %v2float %31 %33
         %35 = OpVectorTimesScalar %v2float %22 %21
         %36 = OpFAdd %v2float %34 %35
               OpStore %11 %36
               OpReturn
               OpFunctionEnd
"""

# The same kernel, but the modf struct is stored whole into a local before its members are read.
WHOLE_STRUCT = COMPUTE.replace(
    "         %21 = OpCompositeExtract %float %20 1 1\n",
    "               OpStore %local %20\n"
    "         %23 = OpLoad %ModfResult %local\n"
    "         %21 = OpCompositeExtract %float %23 1 1\n",
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


def test_struct_members_become_the_result_and_the_out_operand(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    assert re.search(r"\bfloat2 value_20_whole;", crossgl)
    assert re.search(
        r"\bfloat2 value_20_fract = modf\([^;]*, value_20_whole\);", crossgl
    )
    assert re.search(r"\bint2 value_30_exponent;", crossgl)
    assert re.search(
        r"\bfloat2 value_30_significand = frexp\([^;]*, value_30_exponent\);",
        crossgl,
    )
    # Indices after the member index still apply to the member.
    assert "value_20_whole[1]" in crossgl
    assert "value_30_exponent[0]" in crossgl
    assert not re.search(r"\b(modf|frexp)\([^,;]*\)\s*\[", crossgl)


def test_struct_types_sharing_a_debug_name_are_declared_apart(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    names = re.findall(r"\bstruct\s+(\w+)", crossgl)
    assert "ResType" in names and "ResType_1" in names
    assert len(names) == len(set(names))


def test_metal_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="metal", format_output=False
    )
    if not shutil.which("xcrun"):
        pytest.skip("xcrun is not on PATH")
    path = tmp_path / "shader.metal"
    path.write_text(code, encoding="utf-8")
    _run(
        ["xcrun", "-sdk", "macosx", "metal", "-c", str(path), "-o", str(path) + ".air"]
    )


def test_glsl_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="opengl", format_output=False
    )
    assert re.search(r"modf\([^;]*, value_20_whole\)", code)
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(code, encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_hlsl_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="directx", format_output=False
    )
    assert re.search(r"modf\([^;]*, value_20_whole\)", code)
    assert re.search(r"frexp\([^;]*, value_30_exponent\)", code)
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
    assert "ModfStruct" in assembly and "FrexpStruct" in assembly
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
    with pytest.raises(refusal, match="modf"):
        crosstl.translate(
            _write(tmp_path, WHOLE_STRUCT), backend=backend, format_output=False
        )
