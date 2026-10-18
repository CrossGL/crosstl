"""Delimiter scans preserve source positions without visiting plain text runs."""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor


@pytest.mark.parametrize(
    "statement",
    [
        "float value;",
        "\n\tfloat value = (first(1, 2) + second[3]);",
        "Box<Pair<int, float>> value{};",
        'template [[host_name("entry;name")]] void apply<float>(float);',
        "char marker = '\\'';",
        'const char* text = "escaped\\";still literal";',
        "float value /* ; < ( [ { */ = 1;",
        "float value // ; < ( [ {\n = 1;",
        "struct Record { float field; int other[2]; };",
        "int values[] = {1, 2, (3 + 4)};",
        "}}]])>float value;",
        "float value = numerator / denominator;",
    ],
)
def test_statement_end_preserves_nested_and_quoted_delimiters(statement):
    preprocessor = MetalPreprocessor()
    assert preprocessor._statement_end(statement + " float next;", 0) == len(statement)


@pytest.mark.parametrize(
    "source,expected",
    [
        ("", None),
        ("float value", None),
        ("float value // trailing comment", 31),
        ("float value /* unfinished", None),
        ('const char* text = "unfinished;', None),
        ("float value = (1;", None),
        ("int values[3;", None),
        ("Box<float value;", None),
        ("struct Record { int value;", None),
    ],
)
def test_statement_end_retains_incomplete_source_behavior(source, expected):
    assert MetalPreprocessor()._statement_end(source, 0) == expected


def test_statement_end_respects_start_and_end_of_source():
    source = "float first; float second;"
    preprocessor = MetalPreprocessor()
    assert preprocessor._statement_end(source, 0) == len("float first;")
    assert preprocessor._statement_end(source, len("float first;")) == len(source)
    assert preprocessor._statement_end(source, len(source)) is None
    assert preprocessor._statement_end(source, len(source) + 1) is None


def test_namespace_scan_preserves_nested_anonymous_and_non_namespace_braces():
    source = """
    int namespace_suffix;
    int prefixnamespace;
    const char* text = "namespace fake { }";
    // namespace commented { }
    /* namespace hidden { } */
    namespace outer /* gap */ :: inner {
        struct Record { int value; };
        namespace nested { int value; }
        namespace { int anonymous; }
    }
    namespace alias = outer::inner;
    namespace end { int value; }
    """
    preprocessor = MetalPreprocessor()
    spans = preprocessor._find_namespace_spans(source)
    assert [name for _, _, name in spans] == [
        "outer::inner::nested",
        "outer::inner",
        "outer::inner",
        "end",
    ]
    assert all(
        source[start - 1] == "{" and source[end] == "}" for start, end, _ in spans
    )
    assert [source[start:end].strip() for start, end, _ in spans[:2]] == [
        "int value;",
        "int anonymous;",
    ]
    assert preprocessor._source_analysis(source).anonymous_namespace_spans == [
        spans[1][:2]
    ]


@pytest.mark.parametrize(
    "suffix", ["// unfinished {", "/* unfinished {", '"unfinished {']
)
def test_namespace_scan_retains_completed_spans_before_unfinished_trivia(suffix):
    source = "namespace complete { int value; } " + suffix
    spans = MetalPreprocessor()._find_namespace_spans(source)
    assert spans == [(source.index("{") + 1, source.index("}"), "complete")]


class _CountedSource(str):
    def __new__(cls, value):
        result = super().__new__(cls, value)
        result.character_reads = 0
        return result

    def __getitem__(self, index):
        self.character_reads += 1
        return super().__getitem__(index)


@pytest.mark.parametrize("kind", ["statement", "namespace"])
def test_delimiter_scans_skip_plain_source_runs(kind):
    source = _CountedSource("float " + "long_name_" * 10000 + ";")
    preprocessor = MetalPreprocessor()
    if kind == "statement":
        assert preprocessor._statement_end(source, 0) == len(source)
    else:
        assert preprocessor._find_namespace_spans(source) == []
    assert source.character_reads < 10
