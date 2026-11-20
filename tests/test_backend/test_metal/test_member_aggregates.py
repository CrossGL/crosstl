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
    "member",
    (
        "union { float4 data; float3 center; };",
        "struct { float x; float y; };",
        "union { float4 data; struct { float xyz[3]; float radius; }; };",
    ),
)
def test_anonymous_member_aggregate_is_refused_with_a_code(member):
    source = f"struct Record {{ float w; {member} }};"
    with pytest.raises(MetalAnonymousMemberRefusal) as refusal:
        MetalToCrossGLConverter().generate(_parse(source))

    assert refusal.value.code == "project.translate.metal-anonymous-member-unsupported"
    assert "struct Record" in str(refusal.value)
    assert refusal.value.span is not None


def test_anonymous_member_union_refuses_translation(tmp_path):
    source = KERNEL.format(
        declaration="struct Record { union { float4 data; float3 center; }; };",
        expression="records[i].data.w",
    )
    for backend in ("cgl", "metal", "directx", "opengl", "vulkan"):
        with pytest.raises(MetalAnonymousMemberRefusal):
            _translate(tmp_path, source, backend)


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


def test_ggml_block_with_an_anonymous_union_is_refused_not_shrunk():
    # Reduced from ggml/src/ggml-common.h (llama.cpp e3471b3e), included by ggml-metal.metal. The importer used
    # to drop the union, so block_q4_1 lost d, m and dm and shrank from 20 to 16 bytes.
    with pytest.raises(MetalAnonymousMemberRefusal) as refusal:
        MetalToCrossGLConverter().generate(_parse(GGML_BLOCK_Q4_1))

    assert "anonymous union member" in str(refusal.value)


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
