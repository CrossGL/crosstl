"""SPIR-V integer instructions keep their opcode's signedness when imported.

SPIR-V takes the signedness of an integer operation from the opcode, not from the operand types (SPIR-V 1.6 section
3.3.6), while CrossGL takes it from the operand types. The importer used to emit ``OpShiftRightLogical`` of an ``int``
as ``>>`` (an arithmetic shift), ``OpULessThan`` of ``int`` values as a signed ``<``, ``OpSMod`` as the truncated
``%`` and an ``OpSelect`` with a vector condition as ``?:``. Execution on every runtime is covered by the
``s_integer_signedness`` conformance case; these tests check the imported CrossGL and native compiles.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl

COMPUTE = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gid
               OpExecutionMode %main LocalSize 64 1 1
               OpName %main "main"
               OpName %Buf "Buf"
               OpMemberName %Buf 0 "v"
               OpName %buf "buf"
               OpName %x "x"
               OpName %xi "xi"
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
      %float = OpTypeFloat 32
       %bool = OpTypeBool
     %v2int = OpTypeVector %int 2
    %v2uint = OpTypeVector %uint 2
    %v2bool = OpTypeVector %bool 2
     %v3uint = OpTypeVector %uint 3
         %rt = OpTypeRuntimeArray %uint
        %Buf = OpTypeStruct %rt
    %ptr_buf = OpTypePointer StorageBuffer %Buf
   %ptr_uint = OpTypePointer StorageBuffer %uint
    %ptr_gid = OpTypePointer Input %v3uint
     %ptr_in = OpTypePointer Input %uint
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
      %int_3 = OpConstant %int 3
     %uint_0 = OpConstant %uint 0
     %uint_1 = OpConstant %uint 1
     %uint_2 = OpConstant %uint 2
     %uint_3 = OpConstant %uint 3
     %uint_4 = OpConstant %uint 4
     %uint_5 = OpConstant %uint 5
     %uint_6 = OpConstant %uint 6
     %uint_7 = OpConstant %uint 7
  %uint_high = OpConstant %uint 4294967280
   %v2uint_0 = OpConstantComposite %v2uint %uint_0 %uint_0
   %v2uint_1 = OpConstantComposite %v2uint %uint_1 %uint_1
        %buf = OpVariable %ptr_buf StorageBuffer
        %gid = OpVariable %ptr_gid Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %10 = OpAccessChain %ptr_in %gid %uint_0
          %i = OpLoad %uint %10
         %11 = OpAccessChain %ptr_uint %buf %int_0 %i
          %x = OpLoad %uint %11
         %xi = OpBitcast %int %x
         %20 = OpShiftRightLogical %int %xi %int_3
         %21 = OpBitcast %uint %20
         %22 = OpAccessChain %ptr_uint %buf %int_0 %uint_0
               OpStore %22 %21
         %23 = OpShiftRightArithmetic %uint %x %uint_3
         %24 = OpAccessChain %ptr_uint %buf %int_0 %uint_1
               OpStore %24 %23
         %25 = OpULessThan %bool %xi %int_3
         %26 = OpSelect %uint %25 %uint_1 %uint_0
         %27 = OpAccessChain %ptr_uint %buf %int_0 %uint_2
               OpStore %27 %26
         %28 = OpBitwiseOr %uint %x %uint_high
         %29 = OpSDiv %uint %28 %uint_5
         %30 = OpAccessChain %ptr_uint %buf %int_0 %uint_3
               OpStore %30 %29
         %31 = OpSMod %int %xi %int_3
         %32 = OpBitcast %uint %31
         %33 = OpAccessChain %ptr_uint %buf %int_0 %uint_4
               OpStore %33 %32
         %34 = OpConvertSToF %float %x
         %35 = OpConvertFToS %uint %34
         %36 = OpAccessChain %ptr_uint %buf %int_0 %uint_5
               OpStore %36 %35
         %40 = OpCompositeConstruct %v2int %xi %int_1
         %41 = OpCompositeConstruct %v2int %int_1 %xi
         %42 = OpULessThan %v2bool %40 %41
         %43 = OpSelect %v2uint %42 %v2uint_1 %v2uint_0
         %44 = OpCompositeExtract %uint %43 1
         %45 = OpAccessChain %ptr_uint %buf %int_0 %uint_6
               OpStore %45 %44
         %46 = OpIAdd %uint %xi %x
         %47 = OpAccessChain %ptr_uint %buf %int_0 %uint_7
               OpStore %47 %46
               OpReturn
               OpFunctionEnd
"""


def _write(tmp_path, source, name="shader.spvasm"):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _run(command):
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def _crossgl(tmp_path):
    crossgl = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="cgl", format_output=False
    )
    crossgl = re.sub(r"\s+", " ", crossgl)
    # The stores to buf.v[0..7] may alias buf.v[gid.x], so the loaded word is read once, into x, before them.
    snapshot = "uint x = buf[0].v[gl_GlobalInvocationID[0]];"
    assert snapshot in crossgl
    assert crossgl.index(snapshot) < crossgl.index("buf[0].v[0] =")
    # X is the loaded word.
    return re.sub(r"\bx\b", "X", crossgl.replace(snapshot, ""))


def test_fixture_is_valid_spirv(tmp_path):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    binary = tmp_path / "shader.spv"
    path = _write(tmp_path, COMPUTE)
    _run(["spirv-as", "--target-env", "vulkan1.1", path, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])


def test_shift_operands_are_reinterpreted_to_the_opcode_signedness(tmp_path):
    crossgl = _crossgl(tmp_path)
    # A logical shift of an int shifts the uint bits; an arithmetic shift of a uint shifts the int bits.
    assert "buf[0].v[0] = asuint(int((uint(asint(X)) >> 3)));" in crossgl
    assert "buf[0].v[1] = uint((int(X) >> 3));" in crossgl


def test_comparison_and_division_operands_follow_the_opcode(tmp_path):
    crossgl = _crossgl(tmp_path)
    assert "((uint(asint(X)) < uint(3)) ? 1 : 0)" in crossgl
    assert "uint((int((X | 4294967280u)) / int(5)))" in crossgl


def test_a_uint_constant_above_int_max_keeps_its_type(tmp_path):
    crossgl = _crossgl(tmp_path)
    assert "4294967280u" in crossgl
    assert not re.search(r"\b4294967280\b(?!u)", crossgl)


def test_signed_modulus_takes_the_sign_of_the_divisor(tmp_path):
    crossgl = _crossgl(tmp_path)
    remainder = "(asint(X) % 3)"
    mask = f"((({remainder} ^ 3) & ({remainder} | -{remainder})) >> 31)"
    assert f"buf[0].v[4] = asuint(({remainder} + (3 & {mask})));" in crossgl


def test_float_to_signed_conversion_into_a_uint_converts_to_int(tmp_path):
    crossgl = _crossgl(tmp_path)
    assert "buf[0].v[5] = uint(int(float(int(X))));" in crossgl


def test_mixed_signedness_addition_is_well_typed(tmp_path):
    crossgl = _crossgl(tmp_path)
    assert "buf[0].v[7] = (uint(asint(X)) + X);" in crossgl


def test_select_with_a_vector_condition_selects_per_component(tmp_path):
    crossgl = _crossgl(tmp_path)
    assert (
        "mix(uint2(0, 0), uint2(1, 1), "
        "(uint2(int2(asint(X), 1)) < uint2(int2(1, asint(X)))))[1]"
    ) in crossgl
    assert "? uint2(" not in crossgl


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
    assert "int64" not in code
    if not shutil.which("glslangValidator"):
        pytest.skip("glslangValidator is not on PATH")
    path = tmp_path / "shader.comp"
    path.write_text(code, encoding="utf-8")
    _run(["glslangValidator", "-S", "comp", str(path)])


def test_hlsl_output_compiles(tmp_path):
    code = crosstl.translate(
        _write(tmp_path, COMPUTE), backend="directx", format_output=False
    )
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
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = _write(tmp_path, assembly, "round_trip.spvasm")
    binary = tmp_path / "round_trip.spv"
    _run(["spirv-as", "--target-env", "vulkan1.1", path, "-o", str(binary)])
    _run(["spirv-val", "--target-env", "vulkan1.1", str(binary)])
