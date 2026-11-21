"""SPIR-V pointer access chains, built-in-named block types and array sample masks.

- ``OpPtrAccessChain``'s first index, ``Element``, moves ``Base`` within the array it points into (SPIR-V
  specification, ``OpPtrAccessChain``). The importer used to read it as one more subscript, so ``p = &a[0]`` moved by
  ``i`` became ``a[0][i]``, which no longer type-checks. It is now ``a[0 + i]``, written ``a[i]``.
- slangc names the block of a ``RWStructuredBuffer<float>`` ``RWStructuredBuffer``. Declared under that name, the
  CrossGL struct shadowed the built-in type; it now takes the name ``RWStructuredBuffer_``.
- A fragment shader's ``int gl_SampleMask[1]`` output (the GLSL declaration, which is what glslang writes) was
  refused by the SPIR-V emitter's type check, which read the SPIR-V array type's internal name instead of
  ``int[1]``.
"""

from __future__ import annotations

import re
import shutil
import subprocess

import pytest

import crosstl
from crosstl.backend.SPIRV.ray_tracing_import import SpirvInstructionUnsupportedError
from crosstl.backend.SPIRV.type_declaration_names import declarable_struct_name

POINTER_ARITHMETIC = """; SPIR-V
               OpCapability VariablePointersStorageBuffer
               OpCapability Shader
               OpExtension "SPV_KHR_storage_buffer_storage_class"
               OpExtension "SPV_KHR_variable_pointers"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %outp %fin %gid
               OpExecutionMode %main LocalSize 1 1 1
               OpName %RWStructuredBuffer "RWStructuredBuffer"
               OpName %outp "outp"
               OpName %fin "fin"
               OpName %main "main"
               OpDecorate %arr ArrayStride 4
               OpDecorate %RWStructuredBuffer Block
               OpMemberDecorate %RWStructuredBuffer 0 Offset 0
               OpDecorate %outp Binding 1
               OpDecorate %outp DescriptorSet 0
               OpDecorate %fin Binding 0
               OpDecorate %fin DescriptorSet 0
               OpDecorate %gid BuiltIn GlobalInvocationId
               OpDecorate %ptr_float ArrayStride STRIDE
      %float = OpTypeFloat 32
        %arr = OpTypeRuntimeArray %float
%RWStructuredBuffer = OpTypeStruct %arr
    %ptr_buf = OpTypePointer StorageBuffer %RWStructuredBuffer
       %uint = OpTypeInt 32 0
     %v3uint = OpTypeVector %uint 3
     %ptr_in = OpTypePointer Input %v3uint
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
        %int = OpTypeInt 32 1
      %int_0 = OpConstant %int 0
  %ptr_float = OpTypePointer StorageBuffer %float
    %float_4 = OpConstant %float 4
    %float_1 = OpConstant %float 1
       %outp = OpVariable %ptr_buf StorageBuffer
        %fin = OpVariable %ptr_buf StorageBuffer
        %gid = OpVariable %ptr_in Input
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %15 = OpLoad %v3uint %gid
          %i = OpCompositeExtract %uint %15 0
         %20 = OpAccessChain %ptr_float %outp %int_0 %int_0
          %p = OpPtrAccessChain %ptr_float %20 %i
         %22 = OpAccessChain %ptr_float %fin %int_0 %i
         %23 = OpLoad %float %22
               OpStore %p %23
         %25 = OpSNegate %uint %i
         %26 = OpPtrAccessChain %ptr_float %p %25
         %27 = OpPtrAccessChain %ptr_float %26 %i
         %zero = OpPtrAccessChain %ptr_float %27 %int_0
         %28 = OpLoad %float %zero
         %29 = OpFMul %float %28 %float_4
         %31 = OpFAdd %float %29 %float_1
               OpStore %27 %31
               OpReturn
               OpFunctionEnd
"""

SAMPLE_MASK = """shader Probe {
    fragment {
        float4 color @output @location(0);
        float2 uv @input @location(0);
        int gl_SampleMask[1] @output @gl_SampleMask;
        void main() {
            color = float4(uv[0], uv[1], 0.0, 1.0);
            gl_SampleMask[0] = 1;
        }
    }
}
"""


def _write(tmp_path, source, name):
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return str(path)


def _validate(tmp_path, assembly, env="vulkan1.1"):
    if not (shutil.which("spirv-as") and shutil.which("spirv-val")):
        pytest.skip("SPIRV-Tools are not on PATH")
    path = _write(tmp_path, assembly, "out.spvasm")
    binary = tmp_path / "out.spv"
    for command in (
        ["spirv-as", "--target-env", env, path, "-o", str(binary)],
        ["spirv-val", "--target-env", env, str(binary)],
    ):
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr


def _import(tmp_path, source):
    path = _write(tmp_path, source, "probe.spvasm")
    return re.sub(r"\s+", " ", crosstl.translate(path, backend="cgl"))


def test_fixture_is_valid_spirv(tmp_path):
    # The entry point lists its StorageBuffer variables, as SPIR-V 1.4 and later require.
    _validate(tmp_path, POINTER_ARITHMETIC.replace("STRIDE", "4"), "vulkan1.2")


def test_pointer_access_chain_element_moves_the_array_index(tmp_path):
    crossgl = _import(tmp_path, POINTER_ARITHMETIC.replace("STRIDE", "4"))
    i = "gl_GlobalInvocationID[0]"
    # &member0[0] moved by i is &member0[i]; moved by -i and then i again it is member0[(i + -i) + i].
    assert f"outp[0].member0[{i}] = fin[0].member0[{i}];" in crossgl
    moved = f"outp[0].member0[(({i} + -{i}) + {i})]"
    assert f"{moved} = (({moved} * 4.0) + 1.0);" in crossgl
    assert "member0[0][" not in crossgl


def test_block_named_for_a_builtin_type_is_renamed(tmp_path):
    crossgl = _import(tmp_path, POINTER_ARITHMETIC.replace("STRIDE", "4"))
    assert "struct RWStructuredBuffer_ {" in crossgl
    assert "RWStructuredBuffer<RWStructuredBuffer_> outp" in crossgl
    assert declarable_struct_name("Buf") == "Buf"
    assert declarable_struct_name("float4") == "float4_"
    assert declarable_struct_name("Texture2D", {"Texture2D_"}) == "Texture2D__"


def test_pointer_arithmetic_round_trip_validates(tmp_path):
    path = _write(tmp_path, POINTER_ARITHMETIC.replace("STRIDE", "4"), "probe.spvasm")
    assembly = crosstl.translate(path, backend="vulkan")
    assert "OpPtrAccessChain" not in assembly
    _validate(tmp_path, assembly)


def test_pointer_stride_other_than_the_array_stride_is_refused(tmp_path):
    path = _write(tmp_path, POINTER_ARITHMETIC.replace("STRIDE", "8"), "probe.spvasm")
    with pytest.raises(SpirvInstructionUnsupportedError, match="ArrayStride 8"):
        crosstl.translate(path, backend="cgl")


def test_pointer_to_a_whole_variable_cannot_move(tmp_path):
    source = POINTER_ARITHMETIC.replace("STRIDE", "4").replace(
        "%p = OpPtrAccessChain %ptr_float %20 %i",
        "%whole = OpPtrAccessChain %ptr_buf %outp %i\n"
        "          %p = OpAccessChain %ptr_float %whole %int_0 %i",
    )
    path = _write(tmp_path, source, "probe.spvasm")
    with pytest.raises(SpirvInstructionUnsupportedError, match="array element"):
        crosstl.translate(path, backend="cgl")


def test_array_sample_mask_output_compiles_to_spirv(tmp_path):
    assembly = crosstl.translate(
        _write(tmp_path, SAMPLE_MASK, "probe.cgl"), backend="vulkan"
    )
    assert "BuiltIn SampleMask" in assembly
    assert re.search(r"OpTypeArray %int %(int|uint)_1\b", assembly)
    _validate(tmp_path, assembly)
