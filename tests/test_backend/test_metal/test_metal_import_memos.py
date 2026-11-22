"""Analyses the Metal importer computes once per source text must answer as a fresh computation would."""

from crosstl.backend.Metal.MetalCrossGLCodeGen import MetalToCrossGLConverter
from crosstl.backend.Metal.preprocessor import MetalPreprocessor

SOURCE = """
struct Inner { float x; float scale() const { return x * 2.0f; } };
namespace detail { struct Outer { Inner inner; int count; }; }
template <typename T> struct Box { T value; };
union Bits { uint u; float f; };
"""


def test_struct_definitions_match_a_fresh_scan_on_every_call():
    preprocessor = MetalPreprocessor()
    fresh = preprocessor._scan_concrete_struct_definitions(SOURCE)

    first = preprocessor._find_concrete_struct_definitions(SOURCE)
    second = preprocessor._find_concrete_struct_definitions(SOURCE)

    assert [definition.qualified_name for definition in first] == [
        "Inner",
        "detail::Outer",
        "Bits",
    ]
    assert first == fresh
    assert second == fresh
    assert first is not second


def test_rebinding_a_returned_definition_leaves_later_results_unchanged():
    preprocessor = MetalPreprocessor()
    first = preprocessor._find_concrete_struct_definitions(SOURCE)
    expected = [definition.span for definition in first]

    # Template probes rebase a synthetic definition onto the template's span.
    first[0].span = (1, 2)
    first.clear()

    later = preprocessor._find_concrete_struct_definitions(SOURCE)
    assert [definition.span for definition in later] == expected


def test_struct_definitions_follow_the_source_text():
    preprocessor = MetalPreprocessor()
    preprocessor._find_concrete_struct_definitions(SOURCE)
    changed = SOURCE.replace("struct Inner", "struct Renamed")

    names = [
        definition.name
        for definition in preprocessor._find_concrete_struct_definitions(changed)
    ]

    assert names == ["Renamed", "Outer", "Bits"]


def test_sanitized_template_names_keep_the_first_spelling_per_spelling_table():
    converter = MetalToCrossGLConverter()
    converter.template_name_spellings = {}

    first = converter.sanitize_identifier("Box<uint>")
    assert converter.sanitize_identifier("Box<uint32_t>") == first
    assert converter.sanitize_identifier("Box<uint>") == first
    assert converter.sanitize_identifier("a::b") == "a_u3a_u3ab"
    assert converter.sanitize_identifier("plain") == "plain"

    # A new program starts a new spelling table, so the other spelling may name the specialization now.
    converter.template_name_spellings = {}
    assert converter.sanitize_identifier("Box<uint32_t>") != first
    assert converter.sanitize_identifier("Box<uint>") == converter.sanitize_identifier(
        "Box<uint32_t>"
    )
