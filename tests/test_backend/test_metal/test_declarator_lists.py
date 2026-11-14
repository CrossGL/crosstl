"""Declarator lists of concrete struct types become one declaration per name."""

import pytest

from crosstl.backend.Metal.declarator_lists import split_struct_declarator_lists
from crosstl.backend.Metal.preprocessor import MetalPreprocessor


def _split(code, names=("S",)):
    spans = MetalPreprocessor()._find_comment_and_literal_spans(code)
    return split_struct_declarator_lists(code, names, spans)


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("void f() { S a, b, c; }", "void f() { S a; S b; S c; }"),
        (
            "void f() { const thread S a = {1, 2}, *b = &a, c[2]; }",
            "void f() { const thread S a = {1, 2}; const thread S *b = &a; "
            "const thread S c[2]; }",
        ),
        ("void f() { S a(1, 2), b{3, 4}; }", "void f() { S a(1, 2); S b{3, 4}; }"),
        (
            "void f() {\n  // a = x ; b = (y)\n  S a, b;\n}",
            "void f() {\n  // a = x ; b = (y)\n  S a; S b;\n}",
        ),
    ],
)
def test_struct_declarator_lists_split_into_single_declarations(code, expected):
    assert _split(code) == expected


@pytest.mark.parametrize(
    "code",
    [
        # A for init-statement must stay one statement.
        "void f() { for (S a, b; ;) {} }",
        # Expressions and declarations that only look like lists.
        "void f() { S::g(a, b); }",
        "void f() { S(x), y; }",
        "S make(int a, int b);",
        # A comma inside template arguments cannot be split exactly.
        "void f() { S a = cast<int, 2>(x), b; }",
        # Other types and single declarators are left alone.
        "void f() { T a, b; S c; }",
        "void f() { /* S a, b; */ }",
    ],
)
def test_lists_that_cannot_be_split_exactly_stay_unchanged(code):
    assert _split(code) == code
