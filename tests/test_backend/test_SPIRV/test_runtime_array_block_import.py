"""SPIR-V storage blocks that end in a runtime array, imported as CrossGL and translated on.

The importer writes such a block as ``struct B { ...; T a[]; }`` plus ``RWStructuredBuffer<B> v`` read through
``v[0]``. Translating that CrossGL used to fail for GLSL, MSL and HLSL, because only translations from SPIR-V
rewrote it into a storage buffer; and a module without debug names gave the struct and the variable one name.
"""

import re

import pytest

import crosstl
from crosstl.translator.codegen.buffer_length import BufferLengthUnsupportedError
from crosstl.translator.semantics.storage_blocks import (
    SpirvArrayLengthUnsupportedError,
)

TARGETS = ("opengl", "metal", "directx")

# glslangValidator -V of `layout(std430, binding = 0) buffer Out { float outp[]; };`
# `void main() { outp[gl_GlobalInvocationID.x] = 1.0; }`, disassembled with raw ids.
FLAT_BLOCK = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %4 "main" %16
OpExecutionMode %4 LocalSize 1 1 1
NAMES
OpDecorate %7 ArrayStride 4
OpDecorate %8 BufferBlock
OpMemberDecorate %8 0 Offset 0
OpDecorate %10 Binding 0
OpDecorate %10 DescriptorSet 0
OpDecorate %16 BuiltIn GlobalInvocationId
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeFloat 32
%7 = OpTypeRuntimeArray %6
%8 = OpTypeStruct %7
%9 = OpTypePointer Uniform %8
%10 = OpVariable %9 Uniform
%11 = OpTypeInt 32 1
%12 = OpConstant %11 0
%13 = OpTypeInt 32 0
%14 = OpTypeVector %13 3
%15 = OpTypePointer Input %14
%16 = OpVariable %15 Input
%17 = OpConstant %13 0
%18 = OpTypePointer Input %13
%21 = OpConstant %6 1
%22 = OpTypePointer Uniform %6
%4 = OpFunction %2 None %3
%5 = OpLabel
%19 = OpAccessChain %18 %16 %17
%20 = OpLoad %13 %19
%23 = OpAccessChain %22 %10 %12 %20
OpStore %23 %21
OpReturn
OpFunctionEnd
"""

FLAT_NAMES = """OpName %4 "main"
OpName %8 "Out"
OpMemberName %8 0 "outp"
OpName %10 ""
OpName %16 "gl_GlobalInvocationID\""""

# `buffer Data { uint count; vec4 items[]; };` with `count = uint(items.length());`, debug names stripped.
LENGTH_BLOCK = """OpCapability Shader
%1 = OpExtInstImport "GLSL.std.450"
OpMemoryModel Logical GLSL450
OpEntryPoint GLCompute %4 "main" %23
OpExecutionMode %4 LocalSize 1 1 1
OpDecorate %9 ArrayStride 16
OpDecorate %10 BufferBlock
OpMemberDecorate %10 0 Offset 0
OpMemberDecorate %10 1 Offset 16
OpDecorate %12 Binding 1
OpDecorate %12 DescriptorSet 0
OpDecorate %23 BuiltIn GlobalInvocationId
%2 = OpTypeVoid
%3 = OpTypeFunction %2
%6 = OpTypeInt 32 0
%7 = OpTypeFloat 32
%8 = OpTypeVector %7 4
%9 = OpTypeRuntimeArray %8
%10 = OpTypeStruct %6 %9
%11 = OpTypePointer Uniform %10
%12 = OpVariable %11 Uniform
%13 = OpTypeInt 32 1
%14 = OpConstant %13 0
%18 = OpTypePointer Uniform %6
%20 = OpConstant %13 1
%21 = OpTypeVector %6 3
%22 = OpTypePointer Input %21
%23 = OpVariable %22 Input
%24 = OpConstant %6 0
%25 = OpTypePointer Input %6
%28 = OpConstant %7 1
%29 = OpConstantComposite %8 %28 %28 %28 %28
%30 = OpTypePointer Uniform %8
%4 = OpFunction %2 None %3
%5 = OpLabel
%15 = OpArrayLength %6 %12 1
%16 = OpBitcast %13 %15
%17 = OpBitcast %6 %16
%19 = OpAccessChain %18 %12 %14
OpStore %19 %17
%26 = OpAccessChain %25 %23 %24
%27 = OpLoad %6 %26
%31 = OpAccessChain %30 %12 %20 %27
OpStore %31 %29
OpReturn
OpFunctionEnd
"""

MODULES = {
    "named": FLAT_BLOCK.replace("NAMES", FLAT_NAMES),
    "stripped": FLAT_BLOCK.replace("NAMES\n", ""),
}


def _import(tmp_path, assembly):
    """The module's path and the path of the CrossGL the importer writes for it."""
    module = tmp_path / "module.spvasm"
    module.write_text(assembly)
    imported = tmp_path / "imported.cgl"
    imported.write_text(crosstl.translate(str(module), backend="cgl"))
    return str(module), str(imported)


def _translate(path, target):
    return crosstl.translate(path, backend=target, format_output=False)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("names", sorted(MODULES))
def test_imported_crossgl_translates_like_the_module(tmp_path, names, target):
    module, imported = _import(tmp_path, MODULES[names])

    assert _translate(imported, target) == _translate(module, target)


def test_unnamed_block_type_does_not_take_the_variable_name(tmp_path):
    _, imported = _import(tmp_path, MODULES["stripped"])
    code = open(imported).read()

    block_type, variable = re.search(r"RWStructuredBuffer<(\w+)> (\w+)", code).groups()
    assert block_type != variable
    assert re.search(rf"struct {block_type}\b", code)


def test_flat_block_becomes_a_glsl_storage_buffer(tmp_path):
    _, imported = _import(tmp_path, MODULES["named"])
    code = _translate(imported, "opengl")

    assert re.search(r"buffer \w+ \{ float \w+\[\]; \};", code)
    assert "[0]." not in code


@pytest.mark.parametrize("target", ["opengl", "directx"])
def test_array_length_translates_like_the_module(tmp_path, target):
    module, imported = _import(tmp_path, LENGTH_BLOCK)

    assert "spirvArrayLength" in open(imported).read()
    code = _translate(imported, target)
    assert code == _translate(module, target)
    assert "spirvArrayLength" not in code


def test_array_length_is_refused_alike_for_metal(tmp_path):
    module, imported = _import(tmp_path, LENGTH_BLOCK)

    for path in (module, imported):
        with pytest.raises(BufferLengthUnsupportedError):
            _translate(path, "metal")


HAND_WRITTEN = """shader Probe {
    struct Samples {
        uint count;
        float values[];
    };
    RWStructuredBuffer<Samples> samples @set(0) @binding(0);
    compute {
        layout(local_size_x = 1) in;
        void main() {
            samples[0].values[gl_GlobalInvocationID.x] = 2.0;
            samples[0].count = 1u;
        }
    }
}
"""


@pytest.mark.parametrize("target", TARGETS)
def test_written_runtime_array_block_translates(tmp_path, target):
    path = tmp_path / "probe.cgl"
    path.write_text(HAND_WRITTEN)

    code = _translate(str(path), target)
    assert "samples[0]" not in code


def test_fixed_size_block_stays_a_structured_buffer(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(
        HAND_WRITTEN.replace("float values[];", "float values[4];").replace(
            "gl_GlobalInvocationID.x", "0"
        )
    )

    assert "RWStructuredBuffer<Samples>" in _translate(str(path), "directx")


def test_leftover_array_length_is_refused(tmp_path):
    path = tmp_path / "probe.cgl"
    path.write_text(
        HAND_WRITTEN.replace("float values[];", "float values[4];").replace(
            "samples[0].count = 1u;", "samples[0].count = spirvArrayLength(samples, 1);"
        )
    )

    with pytest.raises(SpirvArrayLengthUnsupportedError):
        _translate(str(path), "opengl")
