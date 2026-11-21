"""The skip-ahead scans of Metal project materialization reach the same results as a scan of every character."""

import sys

import pytest

from crosstl.backend.Metal.preprocessor import MetalPreprocessor
from crosstl.project import pipeline
from crosstl.project.metal_source_scans import (
    next_identifier_scan_stop,
    next_statement_scan_stop,
)

SOURCE = """
#include <metal_stdlib>
using namespace metal;
// A comment with a call helper(x); and a "quote
/* A block comment ; { } helper(y) */
template <typename T> T helper(T value) { return value + T(1); }
struct Pair { float a; int b; };
kernel void run(device float* out [[buffer(0)]], uint id [[thread_position_in_grid]]) {
    constexpr int n = 4;
    float values[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    const char* label = "a ; string { with } helper(z)";
    char quote = '\\'';
    Pair pair{1.0f, 2};
    for (int i = 0; i < n; ++i) {
        float scaled = helper(values[i]) * 2.0f;
        if (scaled > 3.0f) { out[id] += scaled; } else { out[id] -= helper<float>(scaled); }
    }
    auto lambda = [&](float x) { return helper(x); };
    out[id] = lambda(pair.a) + float(pair.b) + 1e5f;
}
"""


def _naive_stop(source, start, end, acts):
    for position in range(start, min(end, len(source))):
        if acts(source[position]):
            return position
    return end


@pytest.mark.parametrize(
    ("stop", "acts"),
    [
        (
            next_identifier_scan_stop,
            lambda ch: ch in "\"'/" or ch.isalpha() or ch == "_",
        ),
        (next_statement_scan_stop, lambda ch: ch in "\"'/;()[]<>{}"),
    ],
    ids=["identifier", "statement"],
)
def test_stop_is_the_first_character_the_scan_acts_on(stop, acts):
    for end in (len(SOURCE), len(SOURCE) // 2):
        for start in range(len(SOURCE)):
            assert stop(SOURCE, start, end) == _naive_stop(SOURCE, start, end, acts)


def test_identifier_stops_cover_every_alphabetic_character():
    # A stop pattern may match more than a scan acts on, never less.
    missed = [
        chr(code)
        for code in range(min(sys.maxunicode, 0x2FFFF) + 1)
        if (chr(code).isalpha() or chr(code) == "_")
        and next_identifier_scan_stop(chr(code), 0, 1) != 0
    ]
    assert missed == []


def test_scoped_statement_spans_from_a_kept_scan_match_a_scan_to_each_position():
    preprocessor = MetalPreprocessor()
    body_start = SOURCE.index("{", SOURCE.index("kernel void run")) + 1
    body_end = SOURCE.rindex("}")
    scopes = preprocessor._find_lexical_brace_scopes(SOURCE)
    scans = {}
    for position in range(body_start, body_end + 1):
        direct = pipeline._metal_scoped_statement_spans(
            preprocessor, SOURCE, body_start, body_end, position, scopes
        )
        kept = pipeline._metal_scoped_statement_spans(
            preprocessor, SOURCE, body_start, body_end, position, scopes, scans
        )
        assert kept == direct, position
    assert list(scans) == [(body_start, body_end)]


def test_statement_spans_split_at_top_level_semicolons_only():
    body_start = SOURCE.index("{", SOURCE.index("kernel void run")) + 1
    body_end = SOURCE.rindex("}")
    statements = [
        SOURCE[start:end]
        for start, end in pipeline._metal_statement_spans(SOURCE, body_start, body_end)
    ]
    assert statements[0] == "constexpr int n = 4;"
    assert statements[2] == 'const char* label = "a ; string { with } helper(z)";'
    assert statements[3] == "char quote = '\\'';"


def test_template_reference_names_skip_comments_strings_and_excluded_spans():
    preprocessor = MetalPreprocessor()
    template_start = SOURCE.index("template <typename T>")
    template_end = SOURCE.index("}", template_start) + 1
    body_start = SOURCE.index("{", SOURCE.index("kernel void run")) + 1
    body_end = SOURCE.rindex("}")
    lambda_start = SOURCE.index("auto lambda")
    lambda_end = SOURCE.index(";", lambda_start) + 1
    names = pipeline._metal_template_reference_names(
        preprocessor,
        SOURCE,
        {"helper"},
        [(template_start, template_end)],
        [(0, len(SOURCE))],
    )
    assert names == {"helper"}
    sites = pipeline._plain_template_helper_call_sites(
        preprocessor,
        SOURCE,
        {"helper": object()},
        [(template_start, template_end), (lambda_start, lambda_end)],
        [(body_start, body_end)],
    )
    called = [SOURCE[span[0] : span[1]] for _name, _args, span, _explicit in sites]
    assert called == ["helper", "helper<float>"]
    assert [arguments for _name, arguments, _span, _explicit in sites] == [
        ["values[i]"],
        ["scaled"],
    ]
    assert [explicit for _name, _args, _span, explicit in sites] == [[], ["float"]]
