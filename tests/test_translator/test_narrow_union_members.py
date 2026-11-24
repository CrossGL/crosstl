"""Narrow-integer legalization leaves the members of a ``@union_layout`` struct at their declared width.

A union member overlaps the other members' bytes, so widening ``u8vec4[2]`` to ``ivec4[2]`` would make the union
32 bytes instead of 8 and move every byte lane. The SPIR-V and WGSL union-storage lowering reads the declared width:
it lowers a union whose byte view is never used to word storage, and refuses one whose byte view is used. The same
holds for a ``u16vec2[2]`` 16-bit lane view: widening it to ``ivec2[2]`` would move every lane.
"""

import pytest

import crosstl
from crosstl.translator import ast as A
from crosstl.translator.semantics import legalize_narrow_integers
from crosstl.translator.semantics.narrow import NARROW_TARGETS, narrow_of
from crosstl.translator.source_registry import SOURCE_REGISTRY, register_default_sources

TARGETS = sorted(NARROW_TARGETS)

CGL_UNION = """shader s {
    @union_layout(8, 8, little_endian, metal)
    struct Bits {
        @union_member_layout(0, 8, 8) uvec2 words;
        @union_member_layout(0, 8, 4) u8vec4[2] bytes;
    }
    struct Plain {
        u8vec4 lanes;
    }
    compute {
        void main() {
            Bits b;
            Plain p;
            b.words = uvec2(1u, 2u);
            p.lanes = u8vec4(1, 2, 3, 4);
        }
    }
}
"""

METAL_UNION = """#include <metal_stdlib>
using namespace metal;
union Bits {{ uint2 words; uchar4 bytes[2]; }};
kernel void k(device uint* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
    Bits b;
    b.words = uint2(id, id + 1u);
    out[id] = {value};
}}
"""

METAL_U16_UNION = """#include <metal_stdlib>
using namespace metal;
union Bits {{ uint2 words; ushort2 halves[2]; }};
kernel void k(device uint* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {{
    Bits b;
    b.words = uint2(id, id + 1u);
    out[id] = {value};
}}
"""


def members(ast, name):
    struct = next(s for s in ast.structs if s.name == name)
    return {member.name: member.member_type for member in struct.members}


def element(type_node):
    return type_node.element_type if isinstance(type_node, A.ArrayType) else type_node


@pytest.mark.parametrize("target", TARGETS)
def test_union_members_keep_their_width_and_struct_members_widen(target):
    register_default_sources()
    ast = SOURCE_REGISTRY.get("cgl").parse(CGL_UNION)
    assert legalize_narrow_integers(ast, "metal", target) is True

    union = members(ast, "Bits")
    assert isinstance(union["bytes"], A.ArrayType)
    lanes = narrow_of(element(union["bytes"]))
    assert (lanes.kind, lanes.bits, lanes.lanes) == ("u", 8, 4)
    assert narrow_of(members(ast, "Plain")["lanes"]) is None


def translate(tmp_path, value, backend):
    path = tmp_path / "bits.metal"
    path.write_text(METAL_UNION.format(value=value))
    return crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_options={"threadgroup_size": 1},
    )


def translate_u16(tmp_path, value, backend):
    path = tmp_path / "bits.metal"
    path.write_text(METAL_U16_UNION.format(value=value))
    return crosstl.translate(
        str(path),
        backend=backend,
        format_output=False,
        source_options={"threadgroup_size": 1},
    )


@pytest.mark.parametrize("target", TARGETS)
def test_unused_byte_view_lowers_to_word_storage(tmp_path, target):
    output = translate(tmp_path, "b.words.y + b.words.x", target)
    if target == "wgsl":
        assert "CrossGLUnionStorage: vec2<u32>" in output
        assert "vec4<i32>" not in output
    else:
        assert spirv_struct_members(output, "Bits") == [("OpTypeVector", "uint", "2")]


def spirv_struct_members(output, name):
    """The member types of the SPIR-V struct named ``name``, as (opcode, component, count) tuples."""
    definitions = {}
    for line in output.splitlines():
        words = line.split()
        if len(words) >= 3 and words[1] == "=":
            definitions[words[0]] = words[2:]
    struct_id = next(
        line.split()[1]
        for line in output.splitlines()
        if line.split()[:1] == ["OpName"] and line.split()[2] == f'"{name}"'
    )

    def describe(type_id):
        opcode, *operands = definitions[type_id]
        if opcode == "OpTypeVector":
            component = definitions[operands[0]]
            kind = "uint" if component[:3] == ["OpTypeInt", "32", "0"] else component
            return (opcode, kind, operands[1])
        return (opcode, *operands)

    opcode, *member_ids = definitions[struct_id]
    assert opcode == "OpTypeStruct"
    return [describe(member) for member in member_ids]


@pytest.mark.parametrize("target", TARGETS)
def test_used_byte_view_lowers_at_its_declared_width(tmp_path, target):
    # Each u8vec4 of the view is one storage word, so the union keeps its two words (not ivec4[2]'s eight).
    output = translate(tmp_path, "uint(b.bytes[1].x) + b.words.x", target)
    if target == "wgsl":
        assert "CrossGLUnionStorage: vec2<u32>" in output
        assert (
            "(vec4<u32>(b.CrossGLUnionStorage[1]) >> vec4<u32>(0u, 8u, 16u, 24u))"
            in output
        )
    else:
        assert spirv_struct_members(output, "Bits") == [("OpTypeVector", "uint", "2")]


@pytest.mark.parametrize("target", TARGETS)
def test_used_u16_view_lowers_at_its_declared_width(tmp_path, target):
    # Each u16vec2 of the view is one storage word, so the union keeps its two words (not ivec2[2]'s eight).
    output = translate_u16(tmp_path, "uint(b.halves[1].x) + b.words.x", target)
    if target == "wgsl":
        assert "CrossGLUnionStorage: vec2<u32>" in output
        assert "(vec2<u32>(b.CrossGLUnionStorage[1]) >> vec2<u32>(0u, 16u))" in output
    else:
        assert spirv_struct_members(output, "Bits") == [("OpTypeVector", "uint", "2")]
