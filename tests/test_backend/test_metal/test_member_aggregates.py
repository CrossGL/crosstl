"""Unnamed aggregates declared as struct members, and anonymous member unions.

The Metal parser used to skip an unnamed member aggregate with everything it declared, so ``union { ... } u;``
lost the member ``u`` and an anonymous ``union { float4 data; };`` emptied the enclosing struct. Both changed the
struct's layout without a diagnostic, and every access to the dropped members produced invalid output.
"""

import re

import pytest

import crosstl
from crosstl.backend.Metal.member_aggregates import MetalAnonymousMemberRefusal
from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser

KERNEL = """
#include <metal_stdlib>
using namespace metal;
{declaration}
kernel void read_record(device const Record* records [[buffer(0)]],
                        device float* out [[buffer(1)]],
                        uint i [[thread_position_in_grid]]) {{
    out[i] = {expression};
}}
"""


def _parse(source):
    return MetalParser(MetalLexer(source).tokenize()).parse()


def _translate(tmp_path, source, backend):
    path = tmp_path / "unit.metal"
    path.write_text(source)
    return crosstl.translate(str(path), backend=backend)


def test_unnamed_member_aggregates_keep_every_declarator():
    record = _parse("""
        struct Record {
            float4 position;
            struct { float mass; half charge; } props, spare[2];
            union { uint bits; float value; } tag;
        };
        """).structs[-1]

    assert [member.name for member in record.members] == [
        "position",
        "props",
        "spare",
        "tag",
    ]
    nested = {aggregate.name: aggregate for aggregate in record.nested_structs}
    assert len(nested) == 2
    props, spare, tag = record.members[1:]
    assert props.vtype == spare.vtype and spare.array_sizes
    assert [member.name for member in nested[props.vtype].members] == [
        "mass",
        "charge",
    ]
    assert nested[tag.vtype].aggregate_kind == "union"


@pytest.mark.parametrize("backend", ("cgl", "metal", "directx", "vulkan"))
def test_unnamed_member_aggregates_translate_like_named_ones(tmp_path, backend):
    unnamed = KERNEL.format(
        declaration="""
        struct Record {
            float4 position;
            struct { float mass; half charge; } props;
            union { uint bits; float value; } tag;
        };""",
        expression="records[i].position.x + records[i].props.mass + records[i].tag.value",
    )
    named = unnamed.replace("struct {", "struct Props {").replace(
        "union {", "union Tag {"
    )

    unnamed_output = _translate(tmp_path, unnamed, backend)
    named_output = _translate(tmp_path, named, backend)

    assert re.search(r"__crossgl_unnamed_(struct|union)_\d+", unnamed_output)
    renamed = re.sub(r"__crossgl_unnamed_struct_\d+", "Props", unnamed_output)
    renamed = re.sub(r"__crossgl_unnamed_union_\d+", "Tag", renamed)
    # Longer names change only where the formatter wraps lines.
    assert "".join(renamed.split()) == "".join(named_output.split())


@pytest.mark.parametrize(
    ("member", "promoted"),
    (
        ("union { float4 data; float3 center; };", ("data", "center")),
        ("struct { float x; float y; };", ("x", "y")),
        (
            "union { float4 data; struct { float xyz[3]; float radius; }; };",
            ("data", "xyz", "radius"),
        ),
    ),
)
def test_anonymous_member_aggregate_is_held_by_a_reserved_member(member, promoted):
    source = f"struct Record {{ float w; {member} }};"
    record = _parse(source)
    outer = next(struct for struct in record.structs if struct.name == "Record")

    assert [member.name for member in outer.members] == [
        "w",
        "__crossgl_unnamed_member_1",
    ]
    assert outer.members[1].anonymous_aggregate_member
    crossgl = MetalToCrossGLConverter().generate(record)
    assert "__crossgl_unnamed_member_1;" in crossgl
    for name in promoted:
        assert re.search(rf"\b{name}\b", crossgl)


def test_anonymous_member_union_translates_through_its_reserved_member(tmp_path):
    source = KERNEL.format(
        declaration="struct Record { union { float4 data; float3 center; }; };",
        expression="records[i].data.w",
    )
    crossgl = _translate(tmp_path, source, "cgl")
    metal = _translate(tmp_path, source, "metal")

    assert "buffer_load(records, i).__crossgl_unnamed_member_1.data.w" in crossgl
    assert "records[i].__crossgl_unnamed_member_1.data.w" in metal
    assert re.search(r"^union \w+ \{$", metal, re.M)


GGML_BLOCK_Q4_1 = """
#define GGML_EXTENSION __extension__
typedef half ggml_half;
typedef half2 ggml_half2;
typedef struct {
    GGML_EXTENSION union {
        struct {
            ggml_half d;
            ggml_half m;
        };
        ggml_half2 dm;
    };
    uint8_t qs[16];
} block_q4_1;
"""


def test_ggml_block_with_an_anonymous_union_keeps_its_layout():
    # Reduced from ggml/src/ggml-common.h (llama.cpp e3471b3e), included by ggml-metal.metal. The importer used
    # to drop the union, so block_q4_1 lost d, m and dm and shrank from 20 to 16 bytes, and then refused it.
    crossgl = MetalToCrossGLConverter().generate(_parse(GGML_BLOCK_Q4_1))

    union = re.search(
        r"@union_layout\((\d+), (\d+), little_endian, metal\)\s*// Structs\s*struct (\w+) \{(.*?)\}",
        crossgl,
        re.S,
    )
    assert union is not None and union.group(1, 2) == ("4", "4")
    assert "@union_member_layout(0, 4, 2)" in union.group(4)
    assert "@union_member_layout(0, 4, 4) f16vec2 dm;" in union.group(4)
    block = re.search(r"struct block_q4_1 \{(.*?)\}", crossgl, re.S).group(1)
    assert block.split() == [
        union.group(3),
        "__crossgl_unnamed_member_1;",
        "uint8[16]",
        "qs;",
    ]


def test_gnu_extension_prefix_on_a_named_member_union_is_consumed():
    named = GGML_BLOCK_Q4_1.replace("        };\n", "        } parts;\n").replace(
        "    };\n    uint8_t", "    } scale;\n    uint8_t"
    )
    record = _parse(named)
    block = next(struct for struct in record.structs if struct.members[-1].name == "qs")

    assert [member.name for member in block.members] == ["scale", "qs"]
    crossgl = MetalToCrossGLConverter().generate(record)
    assert "__extension__" not in crossgl and "GGML_EXTENSION" not in crossgl


def test_unnamed_member_class_is_refused():
    with pytest.raises(MetalAnonymousMemberRefusal):
        _parse("struct Record { class { public: float x; } inner; };")
