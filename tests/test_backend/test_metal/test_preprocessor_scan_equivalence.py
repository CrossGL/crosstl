"""Faster Metal source scans return exactly what the scans they replaced returned.

Each reference below is the previous implementation, kept verbatim, and is
compared with the current one on generated C++-like text full of the characters
the scanners treat specially.
"""

import random
import re

import pytest

from crosstl.backend.Metal.preprocessing.support import (
    _keyword_before,
    _MetalFreeOperatorDefinition,
)
from crosstl.backend.Metal.preprocessor import SCAN_STOP_RE, MetalPreprocessor

FRAGMENTS = [
    "int",
    "for",
    "xfor",
    "for_",
    "typename",
    "template",
    "x",
    "_y",
    "\u00e9t\u00e9",
    "\u00b2",
    "12",
    " ",
    "  ",
    "\t",
    "\n",
    "(",
    ")",
    "{",
    "}",
    "[",
    "]",
    ";",
    ",",
    "=",
    "<",
    ">",
    "::",
    "/",
    "*",
    '"a;{"',
    '"esc\\";"',
    "'b'",
    "// c;{\n",
    "/* d; } */",
    "#pragma once\n",
]


def generated_sources(seed, count=300, length=40):
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(FRAGMENTS) for _ in range(rng.randint(0, length)))


def reference_iter_simple_declarations(preprocessor, body):
    statements = []
    i = 0
    n = len(body)
    segment_start = 0
    paren_depth = 0
    while i < n:
        ch = body[i]
        if ch in "\"'":
            _literal, consumed = preprocessor._read_string(body, i)
            i += consumed
            continue
        if body.startswith("//", i):
            end = body.find("\n", i)
            i = n if end == -1 else end + 1
            segment_start = i
            continue
        if body.startswith("/*", i):
            end = body.find("*/", i + 2)
            i = n if end == -1 else end + 2
            segment_start = i
            continue
        if ch == "(":
            preceding = body[segment_start:i]
            if re.search(r"\bfor\s*$", preceding):
                close = preprocessor._find_matching_delimiter(body, i, "(", ")")
                if close is not None:
                    header = body[i + 1 : close]
                    first_clause = header.split(";", 1)[0]
                    statements.append(first_clause)
                    i = close + 1
                    segment_start = i
                    continue
            paren_depth += 1
            i += 1
            continue
        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue
        if paren_depth == 0 and ch == "{":
            _declarator, initializer = preprocessor._split_top_level_assignment(
                body[segment_start:i]
            )
            if initializer is not None:
                close = preprocessor._find_matching_delimiter(body, i, "{", "}")
                if close is not None:
                    i = close + 1
                    continue
        if paren_depth == 0 and ch in ";{}":
            statements.append(body[segment_start:i])
            i += 1
            segment_start = i
            continue
        i += 1
    if segment_start < n:
        statements.append(body[segment_start:n])
    return statements


def reference_read_identifier(text, start):
    i = start
    while i < len(text) and (text[i].isalnum() or text[i] == "_"):
        i += 1
    return text[start:i], i - start


@pytest.mark.parametrize("seed", range(4))
def test_simple_declaration_scan_matches_reference(seed):
    preprocessor = MetalPreprocessor()
    for source in generated_sources(seed):
        assert preprocessor._iter_simple_declarations(
            source
        ) == reference_iter_simple_declarations(preprocessor, source), source


@pytest.mark.parametrize("keyword", ["for", "typename"])
def test_trailing_keyword_search_matches_regex(keyword):
    pattern = re.compile(rf"\b{keyword}\s*$")
    for source in generated_sources(len(keyword), count=200, length=12):
        for start in range(len(source) + 1):
            for end in range(start, len(source) + 1):
                match = pattern.search(source[start:end])
                expected = None if match is None else start + match.start()
                assert _keyword_before(source, start, end, keyword) == expected, (
                    source,
                    start,
                    end,
                )


def test_identifier_read_matches_character_loop():
    preprocessor = MetalPreprocessor()
    for source in generated_sources(7, count=200):
        for start in range(-min(2, len(source)), len(source) + 2):
            assert preprocessor._read_identifier(
                source, start
            ) == reference_read_identifier(source, start)


@pytest.mark.parametrize("seed", range(3))
def test_scan_stop_is_the_first_position_a_scanner_acts_on(seed):
    rng = random.Random(seed)
    preprocessor = MetalPreprocessor()
    for source in generated_sources(seed + 11, count=150):
        span_lists = []
        for _ in range(2):
            spans = []
            for _ in range(rng.randint(0, 4)):
                start = rng.randint(0, len(source))
                spans.append((start, rng.randint(start, len(source))))
            if rng.random() < 0.5:
                spans.sort()
            span_lists.append(spans)
        boundaries = {
            position for spans in span_lists for span in spans for position in span
        }
        for start in range(len(source) + 1):
            candidates = [
                position
                for position in range(start, len(source))
                if SCAN_STOP_RE.match(source[position])
            ]
            candidates += [boundary for boundary in boundaries if boundary >= start]
            expected = min(candidates, default=len(source))
            assert preprocessor._next_scan_stop(source, start, *span_lists) == expected


def test_scan_stop_accepts_any_character_a_scanner_can_read():
    for character in map(chr, range(0x110000)):
        if character in "\"'/" or character.isalpha() or character == "_":
            assert SCAN_STOP_RE.match(character), repr(character)


def test_constexpr_index_is_shared_but_its_containers_are_not():
    preprocessor = MetalPreprocessor()
    source = (
        "constexpr int twice(int value) { return value * 2; }\n"
        "template <int N> constexpr int scaled() { return N * 3; }\n"
    )
    first = preprocessor._static_constexpr_function_index(source)
    assert sorted(first) == ["scaled", "twice"]
    first["twice"].clear()
    first.pop("scaled")
    second = preprocessor._static_constexpr_function_index(source)
    assert sorted(second) == ["scaled", "twice"]
    assert len(second["twice"]) == 1


def reference_find_free_operator_definitions(preprocessor, code):
    definitions = []
    ignored = preprocessor._find_comment_and_literal_spans(code)
    preprocessor._find_namespace_spans(code)
    anonymous_namespace_spans = preprocessor._source_analysis(
        code
    ).anonymous_namespace_spans
    operator_re = re.compile(
        r"\boperator\s*(?P<operator>==|!=|<=|>=|[+\-*/%]=|[+\-*/%<>])\s*\("
    )
    cursor = 0
    while True:
        template_match = re.search(r"\btemplate\s*<", code[cursor:])
        if template_match is None:
            break
        template_start = cursor + template_match.start()
        cursor = template_start + len("template")
        if preprocessor._containing_span(template_start, ignored) is not None:
            continue
        angle_start = code.find("<", template_start, cursor + 4)
        angle_end = preprocessor._find_matching_template_param_angle(code, angle_start)
        if angle_end is None:
            continue
        operator_match = operator_re.search(code, angle_end + 1)
        if operator_match is None:
            cursor = angle_end + 1
            continue
        declaration_prefix = code[angle_end + 1 : operator_match.start()]
        if any(token in declaration_prefix for token in (";", "{", "}")):
            cursor = angle_end + 1
            continue
        parameter_open = code.find("(", operator_match.start(), operator_match.end())
        parameter_close = preprocessor._find_matching_delimiter(
            code, parameter_open, "(", ")"
        )
        if parameter_close is None:
            cursor = operator_match.end()
            continue
        body_open = code.find("{", parameter_close + 1)
        declaration_end = code.find(";", parameter_close + 1)
        if body_open == -1 or (declaration_end != -1 and declaration_end < body_open):
            cursor = parameter_close + 1
            continue
        body_end = preprocessor._find_matching_brace(code, body_open)
        if body_end is None:
            cursor = body_open + 1
            continue
        parameters = code[parameter_open + 1 : parameter_close]
        parameter_types = tuple(preprocessor._parameter_declared_types(parameters))
        if len(parameter_types) not in {1, 2} or any(
            parameter_type is None for parameter_type in parameter_types
        ):
            cursor = body_end
            continue
        template_text = code[angle_start + 1 : angle_end]
        template_records = preprocessor._parse_template_parameter_list(template_text)
        template_parameters = tuple(
            record.name for record in template_records if record.name
        )
        constraints = tuple(
            record.constraint_text
            for record in template_records
            if record.constraint_text
        )
        return_type = re.sub(
            r"\b(?:constexpr|consteval|inline|static|friend)\b",
            " ",
            declaration_prefix,
        )
        return_type = preprocessor._normalize_template_argument_text(return_type)
        if not return_type or not template_parameters:
            cursor = body_end
            continue
        definitions.append(
            _MetalFreeOperatorDefinition(
                operator=operator_match.group("operator"),
                template_text=template_text,
                template_parameters=template_parameters,
                template_constraints=constraints,
                return_type=return_type,
                parameters=parameters,
                parameter_types=parameter_types,
                body=code[body_open + 1 : body_end - 1],
                span=(template_start, body_end),
                internal_linkage=bool(
                    re.search(
                        r"\bstatic\b",
                        preprocessor._mask_comments_and_literals(declaration_prefix),
                    )
                )
                or any(
                    start <= template_start < end
                    for start, end in anonymous_namespace_spans
                ),
            )
        )
        cursor = body_end
    return definitions


def reference_reject_unresolved_temporary_functor_calls(
    preprocessor, code, structs_by_name, template_declaration_spans
):
    ignored_spans = sorted(
        preprocessor._find_comment_and_literal_spans(code) + template_declaration_spans
    )
    cursor = 0
    while cursor < len(code):
        ignored = preprocessor._containing_span(cursor, ignored_spans)
        if ignored is not None:
            cursor = ignored[1]
            continue
        if not (code[cursor].isalpha() or code[cursor] == "_"):
            cursor += 1
            continue
        ident, consumed = preprocessor._read_identifier(code, cursor)
        ident_end = cursor + consumed
        if ident == "operator":
            cursor = ident_end
            continue
        constructor_start = ident_end
        while constructor_start < len(code) and code[constructor_start].isspace():
            constructor_start += 1
        temporary = preprocessor._temporary_functor_call(code, constructor_start)
        struct = structs_by_name.get(ident)
        if (
            temporary is not None
            and preprocessor._temporary_functor_is_candidate(code, cursor, temporary)
            and (
                struct is None
                or not (
                    struct.has_operator_call
                    or "operator()" in struct.member_prototype_names
                )
            )
        ):
            preprocessor._raise_temporary_functor_error(
                code,
                ident,
                cursor,
                temporary,
                reason="temporary-functor-type-unresolved",
                detail=(
                    "the temporary type does not resolve to a concrete struct "
                    "with operator()"
                ),
            )
        cursor = ident_end


OPERATOR_FRAGMENTS = [
    "template <typename T> ",
    "template<class U>",
    "template < typename T, int N > ",
    "T ",
    "U ",
    "static ",
    "inline ",
    "bool ",
    "operator+",
    "operator ==",
    "operator<",
    "operator*=",
    "operator",
    "(T a, T b) ",
    "(U a) ",
    "(thread T& a, int b)",
    "()",
    "(",
    ")",
    "{ return a; }",
    "{ return T(a + b); }",
    "{",
    "}",
    ";",
    " ",
    "\n",
    "<",
    ">",
    "x",
    "// operator+(T a) {\n",
    "/* template <typename T> */",
    '"template <"',
    "namespace { ",
]

FUNCTOR_FRAGMENTS = [
    "F",
    "Fn",
    "_g",
    "return",
    "operator",
    "template",
    "\u00e9t\u00e9",
    "\u00b2",
    "1e5f",
    "12",
    "{}",
    "{1}",
    "()",
    "(x)",
    "(a, b)",
    "(*Fn)",
    " ",
    "\n",
    "::",
    ";",
    "(",
    ")",
    "{",
    "}",
    "// F{}(x)\n",
    "/* F()(y) */",
    '"F{}(z)"',
    "'c'",
]


def generated_from(fragments, seed, count, length):
    rng = random.Random(seed)
    for _ in range(count):
        yield "".join(rng.choice(fragments) for _ in range(rng.randint(0, length)))


def outcome(call):
    try:
        return "returned", call()
    except Exception as error:  # the refusal text is part of the result
        return type(error).__name__, str(error)


@pytest.mark.parametrize("seed", range(4))
def test_free_operator_scan_matches_reference(seed):
    found = 0
    for source in generated_from(OPERATOR_FRAGMENTS, seed, count=300, length=24):
        expected = outcome(
            lambda: reference_find_free_operator_definitions(
                MetalPreprocessor(), source
            )
        )
        actual = outcome(
            lambda: MetalPreprocessor()._find_free_operator_definitions(source)
        )
        assert actual == expected, source
        found += expected[0] == "returned" and len(expected[1])
    assert found


def test_free_operator_scan_stops_at_the_declaration_break():
    source = (
        "template <typename T> T twice(T a);\n"
        "template <typename T> T operator+(T a, T b) { return a; }\n"
        "template <typename T> struct S { int x; };\n"
        "bool operator==(int a, int b) { return true; }\n"
        "template <class U> static U operator*=(U a, U b) { return a; }\n"
    )
    expected = reference_find_free_operator_definitions(MetalPreprocessor(), source)
    actual = MetalPreprocessor()._find_free_operator_definitions(source)
    assert actual == expected
    assert [definition.operator for definition in actual] == ["+", "*="]


@pytest.mark.parametrize("seed", range(4))
def test_temporary_functor_scan_matches_reference(seed):
    rng = random.Random(seed)
    refused = 0
    for source in generated_from(FUNCTOR_FRAGMENTS, seed + 21, count=300, length=30):
        spans = []
        for _ in range(rng.randint(0, 3)):
            start = rng.randint(0, len(source))
            spans.append((start, rng.randint(start, len(source))))
        spans.sort()
        expected = outcome(
            lambda: reference_reject_unresolved_temporary_functor_calls(
                MetalPreprocessor(), source, {}, list(spans)
            )
        )
        actual = outcome(
            lambda: MetalPreprocessor()._reject_unresolved_temporary_functor_calls(
                source, {}, list(spans)
            )
        )
        assert actual == expected, (source, spans)
        refused += expected[0] != "returned"
    assert refused
