"""A name spelled like a CrossGL keyword can follow a C-style cast.

Imported code keeps source names such as ``from``, which CrossGL lexes as a
keyword token. The expression parser already reads such tokens as names.
The C-style cast gate did not, so ``(uint8)from`` and ``(uint)from`` failed
to parse. Keywords that continue an expression (``as``, ``in``, ``where``,
``match``) still do not start a cast operand.
"""

import re

import pytest

import crosstl
from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser

PROGRAM = """
shader S {
    uint8 f(float from, float x) { %s }
    compute {
        buffer uint data[];
        layout(local_size_x = 1) in;
        void main() { data[0] = uint(f(1.0, 2.0)); }
    }
}
"""


def _translate(tmp_path, body, backend="opengl"):
    source = tmp_path / "casts.cgl"
    source.write_text(PROGRAM % body, encoding="utf-8")
    return crosstl.translate(str(source), backend=backend)


@pytest.mark.parametrize(
    "body",
    ["return (uint8)from;", "return (uint)from;", "return (uint8)(from + x);"],
)
def test_keyword_spelled_names_are_cast_operands(tmp_path, body):
    code = _translate(tmp_path, body)
    assert re.search(r"uint\(\(?from", code)


@pytest.mark.parametrize(
    ("text", "operand"),
    [("from", True), ("local", True), ("in", False), ("as", False), ("where", False)],
)
def test_only_keywords_that_cannot_continue_an_expression_are_operands(text, operand):
    lexer = Lexer(text)
    parser = Parser(lexer.tokens)
    assert parser.current_token_is_keyword_spelled_operand() is operand
