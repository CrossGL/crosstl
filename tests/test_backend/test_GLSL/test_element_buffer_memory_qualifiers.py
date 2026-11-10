"""GLSL input keeps the memory qualifiers of a block lowered to an element buffer.

A named storage block whose only member is a runtime array, such as ``buffer B { float d[]; } b;``, imports as
``RWStructuredBuffer<float> b``. Its ``coherent``, ``volatile``, ``restrict`` and ``writeonly`` qualifiers, and the
Vulkan memory-model qualifiers, have to survive that lowering as the attributes the block form writes; ``readonly``
is the ``StructuredBuffer`` resource type itself.
"""

import re

import pytest

import crosstl
from crosstl.backend.GLSL.openglCrossglCodegen import GLSLToCrossGLConverter
from crosstl.backend.GLSL.OpenglLexer import GLSLLexer
from crosstl.backend.GLSL.OpenglParser import GLSLParser
from crosstl.translator.codegen.hlsl_refusals import DirectXLoweringRefused

QUALIFIED_ELEMENT_BUFFERS = """
#version 460
layout(local_size_x = 1) in;
layout(std430, binding = 0) restrict readonly buffer B0 { float d[]; } b0;
layout(std430, binding = 1) coherent buffer B1 { float d[]; } b1;
layout(std430, binding = 2) volatile writeonly buffer B2 { float d[]; } b2;
void main() {
    uint i = gl_GlobalInvocationID.x;
    b1.d[i] = b0.d[i] * 2.0;
    b2.d[i] = b0.d[i];
}
"""


def import_glsl(code: str) -> str:
    ast = GLSLParser(GLSLLexer(code).tokenize(), "compute").parse()
    return GLSLToCrossGLConverter(shader_type="compute").generate(ast)


def buffer_declarations(crossgl: str) -> list[str]:
    return [
        line.strip()
        for line in crossgl.splitlines()
        if re.match(r"\s*(RW)?StructuredBuffer<", line)
    ]


def test_element_buffers_keep_memory_qualifier_attributes():
    assert buffer_declarations(import_glsl(QUALIFIED_ELEMENT_BUFFERS)) == [
        "StructuredBuffer<float> b0 @binding(0) @restrict;",
        "RWStructuredBuffer<float> b1 @binding(1) @coherent;",
        "RWStructuredBuffer<float> b2 @binding(2) @volatile @writeonly;",
    ]


def test_unqualified_element_buffer_gains_no_attributes():
    code = """
#version 450
layout(local_size_x = 1) in;
layout(std430, binding = 3) buffer B { uint d[]; } b;
void main() { b.d[gl_GlobalInvocationID.x] = 1u; }
"""
    assert buffer_declarations(import_glsl(code)) == [
        "RWStructuredBuffer<uint> b @binding(3);"
    ]


def test_vulkan_memory_model_qualifier_survives_element_buffer_lowering():
    code = """
#version 450
#extension GL_KHR_memory_scope_semantics : require
layout(local_size_x = 1) in;
layout(std430, binding = 0) workgroupcoherent nonprivate buffer B { uint d[]; } b;
void main() { b.d[gl_LocalInvocationID.x] = 1u; }
"""
    assert buffer_declarations(import_glsl(code)) == [
        "RWStructuredBuffer<uint> b @binding(0) @workgroupcoherent @nonprivate;"
    ]


def translate_qualified_buffers(tmp_path, backend: str) -> str:
    source = tmp_path / "qualified_element_buffers.comp"
    source.write_text(QUALIFIED_ELEMENT_BUFFERS)
    return crosstl.translate(str(source), backend=backend, format_output=False)


def test_glsl_round_trip_keeps_storage_block_qualifiers(tmp_path):
    output = translate_qualified_buffers(tmp_path, "opengl")
    assert re.search(
        r"binding = 0\) restrict readonly buffer \w+ \{ float b0\[\]; \}", output
    )
    assert re.search(r"binding = 1\) coherent buffer \w+ \{ float b1\[\]; \}", output)
    assert re.search(
        r"binding = 2\) volatile writeonly buffer \w+ \{ float b2\[\]; \}", output
    )


def test_spirv_output_decorates_imported_element_buffers(tmp_path):
    output = translate_qualified_buffers(tmp_path, "vulkan")
    decorations = {}
    for block, decoration in re.findall(r"OpMemberDecorate %(\w+) 0 (\w+)", output):
        if decoration != "Offset":
            decorations.setdefault(block, set()).add(decoration)
    assert sorted(sorted(values) for values in decorations.values()) == [
        ["Coherent"],
        ["NonReadable", "Volatile"],
        ["NonWritable", "Restrict"],
    ]


@pytest.mark.parametrize(
    "parameter",
    [
        r"const device float\s*\*\s*__restrict b0 \[\[buffer\(0\)\]\]",
        r"coherent device float\s*\*\s*b1 \[\[buffer\(1\)\]\]",
        r"volatile device float\s*\*\s*b2 \[\[buffer\(2\)\]\]",
    ],
)
def test_metal_output_keeps_imported_element_buffer_qualifiers(tmp_path, parameter):
    assert re.search(parameter, translate_qualified_buffers(tmp_path, "metal"))


def test_hlsl_output_refuses_imported_volatile_element_buffer(tmp_path):
    with pytest.raises(DirectXLoweringRefused, match=r"resource 'b2' is volatile"):
        translate_qualified_buffers(tmp_path, "directx")
