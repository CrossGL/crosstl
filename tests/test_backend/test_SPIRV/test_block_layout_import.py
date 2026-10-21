"""The SPIR-V importer keeps the explicit layout of uniform and push-constant blocks as a CrossGL buffer layout."""

import re

import pytest

import crosstl
from crosstl.backend.SPIRV.block_layout_import import SpirvBlockLayoutUnsupportedError

# struct { float a[2]; float b; } in a Uniform (or PushConstant) block. The rule sets place it differently:
#   std140:       ArrayStride 16, b at 32;   std430 and scalar: ArrayStride 4, b at 8;
#   HLSL cbuffer: ArrayStride 16, b at 20 (the last element of an array is not padded).
MODULE = """               OpCapability Shader
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main"
               OpExecutionMode %main LocalSize 1 1 1
               OpName %Block "Params"
               OpMemberName %Block 0 "a"
               OpMemberName %Block 1 "b"
               OpName %params "params"
               OpName %Out "Out"
               OpMemberName %Out 0 "o"
               OpName %out "result"
               OpDecorate %Block Block
               OpMemberDecorate %Block 0 Offset 0
               OpMemberDecorate %Block 1 Offset {offset}
               OpDecorate %arr ArrayStride {stride}
{binding}               OpDecorate %Out Block
               OpMemberDecorate %Out 0 Offset 0
               OpDecorate %rt ArrayStride 4
               OpDecorate %out DescriptorSet 0
               OpDecorate %out Binding 1
       %void = OpTypeVoid
         %fn = OpTypeFunction %void
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
        %int = OpTypeInt 32 1
     %uint_2 = OpConstant %uint 2
      %int_0 = OpConstant %int 0
      %int_1 = OpConstant %int 1
        %arr = OpTypeArray %float %uint_2
      %Block = OpTypeStruct %arr %float
    %ptr_blk = OpTypePointer {storage} %Block
     %params = OpVariable %ptr_blk {storage}
         %rt = OpTypeRuntimeArray %float
        %Out = OpTypeStruct %rt
    %ptr_out = OpTypePointer StorageBuffer %Out
        %out = OpVariable %ptr_out StorageBuffer
    %ptr_flu = OpTypePointer {storage} %float
    %ptr_fls = OpTypePointer StorageBuffer %float
       %main = OpFunction %void None %fn
      %entry = OpLabel
         %pb = OpAccessChain %ptr_flu %params %int_1
          %b = OpLoad %float %pb
         %po = OpAccessChain %ptr_fls %out %int_0 %int_0
               OpStore %po %b
               OpReturn
               OpFunctionEnd
"""
UNIFORM_BINDING = (
    "               OpDecorate %params DescriptorSet 0\n"
    "               OpDecorate %params Binding 2\n"
)


def _module(tmp_path, stride, offset, storage="Uniform"):
    path = tmp_path / "block.spvasm"
    path.write_text(
        MODULE.format(
            stride=stride,
            offset=offset,
            storage=storage,
            binding=UNIFORM_BINDING if storage == "Uniform" else "",
        )
    )
    return path


def _crossgl(path):
    return crosstl.translate(str(path), backend="cgl")


@pytest.mark.parametrize(
    ("stride", "offset", "storage", "layout"),
    [
        (16, 32, "Uniform", "std140"),
        (4, 8, "Uniform", "std430"),
        (16, 20, "Uniform", "hlsl_cbuffer"),
        (4, 8, "PushConstant", "std430"),
        (16, 32, "PushConstant", "std140"),
    ],
)
def test_block_decorations_select_the_rule_set_that_reproduces_them(
    tmp_path, stride, offset, storage, layout
):
    code = _crossgl(_module(tmp_path, stride, offset, storage))
    assert re.search(
        rf"cbuffer Params [^{{]*@buffer_layout\({layout}\) \{{", code
    ), code


# An 8-byte float stride belongs to no rule set, and b at byte 8 or 4 lies inside a under every rule set.
@pytest.mark.parametrize(("stride", "offset"), [(8, 16), (16, 8), (4, 4)])
def test_block_decorations_no_rule_set_reproduces_are_refused(tmp_path, stride, offset):
    with pytest.raises(SpirvBlockLayoutUnsupportedError, match="Params") as raised:
        _crossgl(_module(tmp_path, stride, offset))
    assert raised.value.code.endswith("spirv-block-layout-unsupported")


@pytest.mark.parametrize(
    ("stride", "offset", "layout", "packoffset"),
    [(16, 24, "hlsl_cbuffer", "c1.z"), (4, 12, "std430", None)],
)
def test_block_offsets_past_the_rule_set_import_as_explicit_offsets(
    tmp_path, stride, offset, layout, packoffset
):
    """A member placed past the byte its rule set gives it (GLSL ``layout(offset = N)``, HLSL ``packoffset``)
    keeps that offset as ``@ offset(N)`` and lands there on every target."""
    path = _module(tmp_path, stride, offset)

    code = _crossgl(path)
    assert re.search(
        rf"cbuffer Params [^{{]*@buffer_layout\({layout}\) \{{", code
    ), code
    assert f"float b @offset({offset});" in code

    spirv = crosstl.translate(str(path), backend="vulkan", format_output=False)
    assert re.search(rf"OpMemberDecorate %\w+ 1 Offset {offset}\b", spirv)
    assert f"ArrayStride {stride}" in spirv

    hlsl = crosstl.translate(str(path), backend="directx", format_output=False)
    if packoffset is not None:
        assert f"float b : packoffset({packoffset});" in hlsl
    else:
        # A 4-byte float stride has no packoffset form: the block is read through its registers.
        assert "b = asfloat(__crossgl_Params_registers[0].w);" in hlsl


def test_blocks_without_offsets_keep_the_default_layout(tmp_path):
    path = _module(tmp_path, 16, 32)
    path.write_text(
        re.sub(
            r"\s+OpMemberDecorate %Block \d Offset \d+|\s+OpDecorate %arr ArrayStride \d+",
            "",
            path.read_text(),
        )
    )
    code = _crossgl(path)
    assert "cbuffer Params @set(0) @binding(2) {" in code, code


@pytest.mark.parametrize(("stride", "offset"), [(16, 32), (4, 8)])
def test_spirv_output_reproduces_the_imported_offsets(tmp_path, stride, offset):
    """SPIR-V to SPIR-V keeps the array stride and the member offset of the source block."""
    for storage in ("Uniform", "PushConstant"):
        code = crosstl.translate(
            str(_module(tmp_path, stride, offset, storage)), backend="vulkan"
        )
        block = re.search(
            r"OpDecorate %(\w+) Block\n(?:.*\n)*?.*OpMemberDecorate %\1 1 Offset (\d+)",
            code,
        )
        assert block is not None and int(block.group(2)) == offset, code
        assert re.search(rf"OpDecorate %\w+ ArrayStride {stride}\b", code), code
