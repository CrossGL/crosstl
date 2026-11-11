"""Explicit generic arguments keep their values in the callee spelling the parser records.

The parser keeps a call's explicit generic arguments in the callee name (``f<uint, 5u>``), and every generator reads
the value parameters back from it. It dropped an unsigned literal's suffix (``5u`` became ``5``, so a ``uint``
parameter compared as ``int``) and the grouping of nested operations (``(N + 1) * 2`` became ``N + 1 * 2``).
"""

from crosstl.translator.lexer import Lexer
from crosstl.translator.parser import Parser


def _callee(arguments):
    source = (
        "shader main { compute { void main() { " f"int a = f<{arguments}>(x); " "} } }"
    )
    program = Parser(Lexer(source).get_tokens()).parse()
    entry = next(iter(program.stages.values())).entry_point
    return entry.body.statements[0].initial_value.function.name


def test_unsigned_literals_keep_their_suffix():
    assert _callee("uint, 5u") == "f<uint, 5u>"
    assert _callee("int, 5") == "f<int, 5>"


def test_nested_operations_keep_their_grouping():
    assert _callee("(N + 1) * 2") == "f<(N + 1) * 2>"
    assert _callee("N - (M - 1)") == "f<N - (M - 1)>"
    assert _callee("N + M * 2") == "f<N + (M * 2)>"


def test_boolean_values_are_spelled_in_lowercase():
    assert _callee("float, true") == "f<float, true>"
