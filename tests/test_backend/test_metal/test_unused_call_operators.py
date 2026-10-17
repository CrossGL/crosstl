"""Unreferenced out-of-line bodies must not block concrete kernel lowering."""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor

SOURCE = """struct Root {
    float operator()(float value) const thread { return value * 2.0f; }
};
struct Offset {
    template <typename T> T operator()(T value) const thread { return -value; }
    float operator()(float value) const thread;
};
float Offset::operator()(float value) const thread {
    return value + Root{}(value);
}
kernel void calculate(device float* results [[buffer(0)]]) {
    results[0] = Root{}(3.0f);
}
"""


def test_unreferenced_call_operator_preserves_source_offsets():
    processor = MetalPreprocessor()
    output = processor._prune_unreferenced_call_operator_definitions(SOURCE)
    start = SOURCE.index("float Offset::")
    end = SOURCE.index("\n}", start) + 2
    assert output[:start] == SOURCE[:start]
    assert output[end:] == SOURCE[end:]
    assert not output[start:end].strip()
    assert len(output) == len(SOURCE)
    assert [i for i, ch in enumerate(output) if ch == "\n"] == [
        i for i, ch in enumerate(SOURCE) if ch == "\n"
    ]
    assert processor._prune_unreferenced_call_operator_definitions(output) == output


@pytest.mark.parametrize(
    "reference",
    (
        "using Alias = Offset;",
        "typedef Offset Alias;",
        "Offset get_offset();",
        "template <typename T> T invoke(T value) { return Offset{}(value); }",
        "struct Wrapper { float call(float value) { return Offset{}(value); } };",
        "struct Child : Offset {};",
        "float elsewhere(float x) { Offset operation; return operation(x); }",
        "float explicit_call(Offset operation) { return operation.operator()(1.0f); }",
    ),
)
def test_any_owner_reference_retains_qualified_body(reference):
    source = SOURCE + reference
    assert (
        MetalPreprocessor()._prune_unreferenced_call_operator_definitions(source)
        == source
    )


@pytest.mark.parametrize("newline", ("\n", "\r\n"))
def test_comments_and_literals_are_not_owner_references(newline):
    source = SOURCE.replace("\n", newline)
    source += '\n// Offset is unused.\n/* Offset */\nconstant char* label = "Offset";'
    output = MetalPreprocessor()._prune_unreferenced_call_operator_definitions(source)
    assert "float Offset::" not in output
    assert output.count("\n") == source.count("\n")
    assert output.count("\r") == source.count("\r")
    assert output.endswith('constant char* label = "Offset";')


@pytest.mark.parametrize(
    "source",
    (
        SOURCE[: SOURCE.index("kernel void")],
        SOURCE.replace("float Offset::", "extern float Offset::"),
        SOURCE.replace("float Offset::", "[[visible]] float Offset::"),
        SOURCE.replace("};\nfloat Offset::", "} instance;\nfloat Offset::"),
        SOURCE.replace("struct Offset", "typedef struct Offset").replace(
            "};\nfloat Offset::", "} Alias;\nfloat Offset::"
        ),
        SOURCE.replace("struct Offset", "namespace nested { struct Offset").replace(
            "kernel void", "}\nkernel void"
        ),
        SOURCE + "\nnamespace other { struct Offset {}; }",
    ),
)
def test_uncertain_or_exported_owner_is_not_pruned(source):
    assert (
        MetalPreprocessor()._prune_unreferenced_call_operator_definitions(source)
        == source
    )


def test_full_preprocessing_keeps_selected_out_of_line_implementation():
    source = SOURCE.replace("results[0] = Root{}", "results[0] = Offset{}")
    output = MetalPreprocessor().preprocess(source)
    assert "float Offset::operator()" in output
    assert "return value + Root__operator_call__temporary(value);" in output
    assert "Offset__operator_call__float__temporary(3.0f)" in output


def test_full_preprocessing_drops_only_unreferenced_owner():
    output = MetalPreprocessor().preprocess(SOURCE)
    assert "float Offset::operator()" not in output
    assert "results[0] = Root__operator_call__temporary(3.0f);" in output
    assert "return value * 2.0f;" in output
