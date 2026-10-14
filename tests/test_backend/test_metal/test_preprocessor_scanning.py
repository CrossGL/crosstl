"""Delimiter scans preserve source offsets while skipping unrelated text."""

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor


@pytest.mark.parametrize(
    "source",
    [
        "declaration name;",
        "type<other<int>, value> name;",
        'call(";", nested(1; 2));',
        "array[index; other];",
        "/* ; */ declaration // ;\n;",
        "int operator<(int rhs);",
        "int operator /* < ; */ <<=(int rhs);",
        "int operator // < ;\n<=>(int rhs);",
        "type<other<int>> operator>>(int rhs);",
        "name / other;",
        r"""call("escaped\";", '\'');""",
    ],
)
def test_top_level_delimiter_preserves_nested_and_quoted_offsets(source):
    assert MetalPreprocessor()._find_next_top_level_char(source, 0, ";") == (
        len(source) - 1
    )


@pytest.mark.parametrize("target", ["{", ";", ":", "[", "]", ".", "-", "^", "\\"])
def test_top_level_search_escapes_target_and_honors_start(target):
    source = target + " ignored text " + target
    assert MetalPreprocessor()._find_next_top_level_char(source, 1, target) == (
        len(source) - 1
    )


@pytest.mark.parametrize("track_angles", [False, True])
def test_top_level_search_preserves_angle_tracking_policy(track_angles):
    source = "type<value; other>;"
    result = MetalPreprocessor()._find_next_top_level_char(
        source, 0, ";", track_angles=track_angles
    )
    assert result == (source.rfind(";") if track_angles else source.find(";"))


@pytest.mark.parametrize("target,stop", [("{", ";"), (";", "{")])
@pytest.mark.parametrize(
    "header",
    [
        "type name",
        "name<other<int; 2>> function()",
        '[[attribute("{;}")]] type name',
        "type name[sizeof(value; other)]",
        "type name /* { ; */ // { ;\n",
        "type operator /* { ; */ <<=(int value)",
    ],
)
def test_declaration_search_stops_only_at_top_level_terminators(target, stop, header):
    processor = MetalPreprocessor()
    source = header + stop + " unrelated declaration " + target
    assert processor._find_next_top_level_char(source, 0, target) == len(source) - 1
    assert processor._find_next_top_level_char(source, 0, target, stop_at=stop) is None
    source = header + target + " unrelated declaration " + stop
    assert processor._find_next_top_level_char(source, 0, target, stop_at=stop) == len(
        header
    )


@pytest.mark.parametrize("target,stop", [("{", ";"), (";", "{")])
def test_declaration_terminator_does_not_scan_later_declarations(target, stop):
    class BoundedSource(str):
        def __getitem__(self, key):
            if isinstance(key, int):
                assert key <= self.index(stop), "Read beyond the declaration terminator"
            return super().__getitem__(key)

    source = BoundedSource("name" + stop + " unrelated_identifier " * 4096 + target)
    assert (
        MetalPreprocessor()._find_next_top_level_char(source, 0, target, stop_at=stop)
        is None
    )


@pytest.mark.parametrize(
    "kind", ["struct", "function", "trait", "span", "specialization"]
)
def test_forward_declarations_preserve_later_template_bodies(kind):
    source = """
template<class T> struct Forward;
template<class T> T forward(T value);
template<class T> struct Defined { using type = T; T value; };
template<class T> T defined(T value) { return value; }
template<> int forward<int>(int value);
template<> int defined<int>(int value) { return value + 1; }
"""
    processor = MetalPreprocessor()
    if kind == "struct":
        records = processor._find_template_structs(
            source, include_forward_declarations=True
        )
        assert [(item.name, item.is_forward_declaration) for item in records] == [
            ("Forward", True),
            ("Defined", False),
        ]
    elif kind == "function":
        assert [item.name for item in processor._find_template_functions(source)] == [
            "defined"
        ]
    elif kind == "trait":
        assert processor._find_template_type_traits(source)["Defined"]["default"] == "T"
    elif kind == "span":
        assert len(processor._find_template_declaration_spans(source)) == 6
    else:
        assert processor._find_explicit_template_specialization_keys(source) == {
            ("forward", ("int",)),
            ("defined", ("int",)),
        }


@pytest.mark.parametrize("suffix", ["", "// ;", "/* ;", '"; tail', "'; tail"])
def test_top_level_search_does_not_use_unterminated_comment_or_literal(suffix):
    assert (
        MetalPreprocessor()._find_next_top_level_char("name " + suffix, 0, ";") is None
    )


@pytest.mark.parametrize("opener,closer", [("(", ")"), ("{", "}"), ("[", "]")])
@pytest.mark.parametrize(
    "body",
    [
        "identifier text",
        '"{}[]()"',
        "'(' + ')'",
        "/* {}[]() */ name",
        "// {}[]()\n name",
        "a / b",
        r'"escaped\"()"',
    ],
)
def test_matching_delimiter_preserves_nested_and_quoted_offsets(opener, closer, body):
    source = "prefix " + opener + body + opener + body + closer + closer + " suffix"
    assert MetalPreprocessor()._find_matching_delimiter(source, 7, opener, closer) == (
        len(source) - len(" suffix") - 1
    )


@pytest.mark.parametrize("source", ["(text", "(// )", "(/* )", '(" )', "(' )"])
def test_matching_delimiter_rejects_unclosed_source(source):
    assert MetalPreprocessor()._find_matching_delimiter(source, 0, "(", ")") is None


@pytest.mark.parametrize(
    "argument",
    [
        "float",
        "nested<float, other<int>>",
        "(1 > 2)",
        "(1 < 2)",
        "(1 << 2)",
        "array[(1 > 2)]",
        "value{(1 > 2)}",
        '"quoted>text"',
        "/* > */ float",
        "// >\n float",
    ],
)
def test_template_delimiter_preserves_expression_contexts(argument):
    source = "type<" + argument + "> suffix"
    assert MetalPreprocessor()._find_matching_angle(source, 4) == len(source) - 8


class _IndexedSource(str):
    accesses = 0

    def __getitem__(self, key):
        self.accesses += 1
        return super().__getitem__(key)


@pytest.mark.parametrize(
    "kind", ["top-level", "matching", "template", "comments", "scopes"]
)
def test_delimiter_search_skips_long_unrelated_runs(kind):
    text = "ordinary_identifier " * 4096
    preprocessor = MetalPreprocessor()
    if kind == "top-level":
        source = _IndexedSource(text + ";")
        result = preprocessor._find_next_top_level_char(source, 0, ";")
    elif kind == "matching":
        source = _IndexedSource("(" + text + ")")
        result = preprocessor._find_matching_delimiter(source, 0, "(", ")")
    elif kind == "template":
        source = _IndexedSource("<" + text + ">")
        result = preprocessor._find_matching_angle(source, 0)
    elif kind == "comments":
        source = _IndexedSource(text)
        assert preprocessor._scan_comment_and_literal_spans(source) == []
        result = len(source) - 1
    else:
        source = _IndexedSource("{" + text + "}")
        assert preprocessor._scan_lexical_brace_scopes(source) == [
            (0, len(source)),
            (1, len(source) - 1),
        ]
        result = len(source) - 1
    assert result == len(source) - 1
    assert source.accesses < 20


@pytest.mark.parametrize("span", ['"{}"', "'{'", "/* {} */", "// {}\n"])
def test_lexical_scans_ignore_braces_in_comments_and_literals(span):
    source = "void f() {" + span + " { body } }"
    end = 10 + len(span.rstrip("\n"))
    assert MetalPreprocessor()._scan_comment_and_literal_spans(source) == [(10, end)]
    assert MetalPreprocessor()._scan_lexical_brace_scopes(source) == [
        (0, len(source)),
        (source.rfind("{") + 1, source.index("}", end)),
        (10, len(source) - 1),
    ]


@pytest.mark.parametrize("suffix", ['" }', "' }", "/* }", "// }"])
def test_lexical_scans_preserve_unterminated_trivia(suffix):
    source = "void f() { " + suffix
    preprocessor = MetalPreprocessor()
    assert preprocessor._scan_comment_and_literal_spans(source) == [(11, len(source))]
    assert preprocessor._scan_lexical_brace_scopes(source) == [(0, len(source))]
