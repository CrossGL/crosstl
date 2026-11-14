"""Faster Metal source scans return exactly what the scans they replaced returned.

Each reference below is the previous implementation, kept verbatim, and is
compared with the current one on generated C++-like text full of the characters
the scanners treat specially.
"""

import random
import re

import pytest

from crosstl.backend.Metal.preprocessor import (
    SCAN_STOP_RE,
    MetalPreprocessor,
    _keyword_before,
)

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
