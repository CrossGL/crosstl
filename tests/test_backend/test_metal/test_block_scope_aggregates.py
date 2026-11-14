"""Block-scope Metal aggregates that declare objects, and block-scope unions."""

import pytest

from crosstl.backend.Metal.local_aggregate_declarators import (
    is_unnamed_aggregate_name,
    union_layout,
)
from crosstl.backend.Metal.MetalAst import StructNode, VariableNode
from crosstl.backend.Metal.MetalCrossGLCodeGen import (
    MetalFunctionLocalTypeResolutionError,
    MetalToCrossGLConverter,
)
from crosstl.backend.Metal.MetalLexer import MetalLexer
from crosstl.backend.Metal.MetalParser import MetalParser
from crosstl.translator.lexer import Lexer as CrossGLLexer
from crosstl.translator.parser import Parser as CrossGLParser


def _parse(source):
    return MetalParser(MetalLexer(source).tokenize()).parse()


def _convert(source):
    crossgl = MetalToCrossGLConverter().generate(_parse(source))
    assert CrossGLParser(CrossGLLexer(crossgl).get_tokens()).parse() is not None
    return crossgl


def _body(source):
    return _parse(source).functions[0].body


def test_definition_with_objects_splits_into_aggregate_and_declarations():
    body = _body("""
        uint read_word() {
            struct Pair { uint first; uint second; } left, right;
            return left.first + right.second;
        }
        """)
    assert [type(statement) for statement in body[:3]] == [
        StructNode,
        VariableNode,
        VariableNode,
    ]
    assert body[0].name == "Pair"
    assert not body[0].trailing_declarations
    assert [(body[1].vtype, body[1].name), (body[2].vtype, body[2].name)] == [
        ("Pair", "left"),
        ("Pair", "right"),
    ]


@pytest.mark.parametrize("kind", ("union", "struct"))
def test_unnamed_block_aggregate_receives_a_reserved_name(kind):
    body = _body(f"""
        uint read_word() {{
            {kind} {{ uint word; uint other; }} value;
            return value.word;
        }}
        """)
    aggregate, declaration = body[:2]
    assert is_unnamed_aggregate_name(aggregate.name)
    assert "__" in aggregate.name
    assert aggregate.unnamed_aggregate_kind == kind
    assert declaration.vtype == aggregate.name
    assert declaration.name == "value"


def test_anonymous_block_union_without_objects_is_refused():
    with pytest.raises(SyntaxError, match="unnamed block-scope union must declare"):
        _parse("""
            uint read_word() {
                union { uint word; float value; };
                return word;
            }
            """)


def test_program_scope_union_still_requires_a_name():
    with pytest.raises(SyntaxError, match="Expected union name"):
        _parse("union { uint word; float value; } shared_value;")


def test_block_union_is_hoisted_with_overlapping_layout():
    crossgl = _convert("""
        uint read_word() {
            union Bits { uint word; half halves[2]; uchar bytes[3]; } value;
            value.word = 0u;
            return value.word;
        }
        """)
    name = "MetalLocal_read_word_Bits"
    assert "@union_layout(4, 4, little_endian, metal)" in crossgl
    assert f"struct {name} {{" in crossgl
    assert "@union_member_layout(0, 4, 4) uint word;" in crossgl
    assert "@union_member_layout(0, 4, 2) float16[2] halves;" in crossgl
    assert "@union_member_layout(0, 3, 1) uint8[3] bytes;" in crossgl
    assert f"{name} value;" in crossgl


def test_unnamed_block_union_uses_a_readable_hoisted_name():
    crossgl = _convert("""
        uint read_word() {
            union { uint word; ushort halves[2]; } value;
            value.halves[0] = ushort(1u);
            return value.word;
        }
        """)
    assert "struct MetalLocal_read_word_UnnamedUnion {" in crossgl
    assert "MetalLocal_read_word_UnnamedUnion value;" in crossgl
    assert "__crossgl_unnamed" not in crossgl


def test_typedef_block_union_is_hoisted_with_overlapping_layout():
    crossgl = _convert("""
        uint read_word() {
            typedef union { uint word; float value; } Word;
            Word bits;
            bits.value = 1.0f;
            return bits.word;
        }
        """)
    assert "@union_layout(4, 4, little_endian, metal)" in crossgl
    assert "struct MetalLocal_read_word_Word {" in crossgl
    assert "@union_member_layout(0, 4, 4) float value;" in crossgl


def test_block_union_member_without_a_known_layout_is_refused():
    with pytest.raises(MetalFunctionLocalTypeResolutionError) as error:
        _convert("""
            uint read_word() {
                union { uint word; atomic_uint counter; } value;
                value.word = 0u;
                return value.word;
            }
            """)
    assert error.value.type_name == "<unnamed union>"
    assert "union storage layout cannot be proven for atomic_uint" in str(error.value)


def test_union_layout_rounds_the_largest_member_to_the_strictest_alignment():
    assert union_layout([(3, 1), (2, 2)]) == (4, 2)
    assert union_layout([(12, 4), (16, 16)]) == (16, 16)
    assert union_layout([(1, 1)]) == (1, 1)
