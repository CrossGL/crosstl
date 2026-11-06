"""Syntax of CrossGL ``static_assert`` declarations.

``static_assert ( constant-expression [, string-literal...] ) ;`` may appear at shader, stage and structure scope and
as a statement, as in C++17 [dcl.dcl]/1 and MSL. Adjacent string literals form one message ([lex.string]/13).
:mod:`.semantics.static_assertions` gives the declaration its meaning.
"""

from __future__ import annotations

from .ast import StaticAssertNode

__all__ = ["parse_static_assertion", "starts_static_assertion"]


def starts_static_assertion(parser) -> bool:
    """Whether ``parser`` stands at ``static_assert (``."""
    return parser.current_token == ("IDENTIFIER", "static_assert") and (
        parser.peek()[0] == "LPAREN"
    )


def _string_text(spelling):
    text = str(spelling)
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


def parse_static_assertion(parser) -> StaticAssertNode:
    """Parse one ``static_assert`` declaration, its semicolon included."""
    parser.eat("IDENTIFIER")
    parser.eat("LPAREN")
    condition = parser.parse_expression()
    message = None
    if parser.current_token[0] == "COMMA":
        parser.eat("COMMA")
        if parser.current_token[0] != "STRING_LITERAL":
            raise SyntaxError(
                "CrossGL static_assert takes a string literal as its message, got "
                f"{parser.current_token[0]} {parser.current_token[1]!r}"
            )
        pieces = []
        while parser.current_token[0] == "STRING_LITERAL":
            pieces.append(_string_text(parser.current_token[1]))
            parser.eat("STRING_LITERAL")
        message = "".join(pieces)
    parser.eat("RPAREN")
    parser.eat("SEMICOLON")
    return StaticAssertNode(condition, message)
