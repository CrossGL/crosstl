"""Anonymous enumerations declared inside a structure.

``struct traits { enum { qk = 32, b1 = qk / 4 }; };`` declares the constants ``traits::qk`` and ``traits::b1``,
which a template reads through its type parameter (``T::qk``). The structure parser read the enumeration as a member
declaration without a name, so the CrossGL output had a member ``enum;`` that no parser accepts, and ``traits::qk``
named nothing. Each enumerator becomes a ``static constant constexpr`` member of the structure instead; the importer
already folds reads of such members to their values on every target.

An enumerator has the enumeration's type, which converts to its underlying type wherever it is read as a number. The
underlying type is the declared one, else ``int`` ([dcl.enum]/7): an enumerator whose literal value does not fit
``int`` is refused, since the source's type for it would be wider.
"""

from __future__ import annotations

import re

from .MetalAst import BinaryOpNode, VariableNode

__all__ = ["is_anonymous_member_enum_start", "parse_anonymous_member_enum"]

_INT_MAX = (1 << 31) - 1
_LITERAL = re.compile(r"(?P<digits>0[xX][0-9A-Fa-f]+|0[bB][01]+|\d+)[uUlL]*")


def _closing_brace(tokens, start):
    depth = 0
    for index in range(start, len(tokens)):
        kind = tokens[index][0]
        if kind == "LBRACE":
            depth += 1
        elif kind == "RBRACE":
            depth -= 1
            if depth == 0:
                return index
        elif kind == "EOF":
            return None
    return None


def is_anonymous_member_enum_start(tokens, pos):
    """Whether ``enum [: type] { ... };`` without a name or declarator starts at ``pos``."""
    if pos >= len(tokens) or tokens[pos][0] != "ENUM":
        return False
    index = pos + 1
    if index < len(tokens) and tokens[index][0] == "COLON":
        while index < len(tokens) and tokens[index][0] not in {
            "LBRACE",
            "SEMICOLON",
            "EOF",
        }:
            index += 1
    if index >= len(tokens) or tokens[index][0] != "LBRACE":
        return False
    close = _closing_brace(tokens, index)
    return (
        close is not None
        and close + 1 < len(tokens)
        and tokens[close + 1][0] == "SEMICOLON"
    )


def _literal_value(value):
    if not isinstance(value, str):
        return None
    match = _LITERAL.fullmatch(value.strip())
    if match is None:
        return None
    digits = match.group("digits")
    if digits[:2] in {"0x", "0X"}:
        return int(digits, 16)
    if digits[:2] in {"0b", "0B"}:
        return int(digits[2:], 2)
    return int(digits, 8) if len(digits) > 1 and digits[0] == "0" else int(digits)


def parse_anonymous_member_enum(parser):
    """Parse the enumeration at the cursor and return its enumerators as ``static constant constexpr`` members."""
    start = parser.current_token
    enum = parser.parse_enum()
    underlying = getattr(enum, "underlying_type", None)
    member_type = underlying if underlying is not None else "int"
    qualifiers = ["static", "constant", "constexpr"]
    members, previous = [], None
    for name, value in enum.members:
        if value is None:
            # An enumerator without a value is one more than the previous one, and the first is zero.
            value = (
                "0"
                if previous is None
                else BinaryOpNode(VariableNode("", previous), "+", "1")
            )
        literal = _literal_value(value)
        if underlying is None and literal is not None and literal > _INT_MAX:
            from .MetalParser import MetalParserError

            raise MetalParserError(
                f"Metal anonymous member enumeration value {literal} of '{name}' does not fit int; "
                "declare the enumeration's underlying type",
                token=start,
            )
        member = VariableNode(
            member_type, name, qualifiers=list(qualifiers), attributes=[]
        )
        member.pointee_qualifiers = list(qualifiers)
        member.indirection_qualifiers = []
        member.array_sizes = []
        member.alignas = []
        member.default_value = value
        member.source_location = getattr(enum, "source_location", None)
        members.append(member)
        previous = name
    return members
