"""Unnamed aggregates declared inside a struct, class or union.

A member declaration may define an unnamed class and declare members of it at once, as in
``struct S { union { float a; int b; } u; };`` ([class.mem], [dcl.pre]). The unnamed class is parsed like a named
nested aggregate under a reserved name, and its declarators become members of the enclosing class.

An unnamed union that declares no member is an anonymous union: its members are members of the enclosing class and
share its storage ([class.union.anon]). CrossGL has no anonymous members, and accesses to them are written without
the name of the union, so the importer cannot carry them. It refuses them with a typed diagnostic instead of
dropping the members, which would change the layout of the class and leave every access to them undeclared. An
unnamed struct that declares no member is not standard C++ ([class.mem] requires a declarator), but Clang accepts
it as an extension with the same meaning, so it is refused in the same way.
"""

from __future__ import annotations

from typing import Any

from crosstl.translator.diagnostics import CODE_PREFIX
from crosstl.translator.refusals import MetalSourceRefusal

from .local_aggregate_declarators import is_unnamed_aggregate_name

__all__ = [
    "MetalAnonymousMemberRefusal",
    "parse_unnamed_member_aggregate",
    "unnamed_member_aggregate_kind",
]


class MetalAnonymousMemberRefusal(MetalSourceRefusal):
    """An anonymous union or struct among the members of a class, which CrossGL cannot represent."""

    code = f"{CODE_PREFIX}metal-anonymous-member-unsupported"


def unnamed_member_aggregate_kind(parser: Any) -> str | None:
    """``"struct"``, ``"class"`` or ``"union"`` when the parser stands on an unnamed aggregate definition."""

    index = parser.skip_gnu_extension_prefix_tokens_at(parser.pos)
    if index >= len(parser.tokens):
        return None
    token = parser.tokens[index]
    if token[0] == "STRUCT":
        kind = "struct"
    elif token[0] == "CLASS":
        kind = "class"
    elif token == ("IDENTIFIER", "union"):
        kind = "union"
    else:
        return None
    body = _body_index(parser, index + 1)
    if body >= len(parser.tokens) or parser.tokens[body][0] != "LBRACE":
        return None
    return kind


def parse_unnamed_member_aggregate(parser: Any, pre_alignas=None):
    """Parse an unnamed member aggregate and the members it declares, or refuse an anonymous one."""

    kind = unnamed_member_aggregate_kind(parser)
    start = parser.skip_gnu_extension_prefix_tokens_at(parser.pos)
    body = _body_index(parser, start + 1)
    after = parser.skip_balanced_tokens_at(body, "LBRACE", "RBRACE")
    owner = parser.current_declaration_context()
    if after >= len(parser.tokens) or parser.tokens[after][0] == "SEMICOLON":
        raise MetalAnonymousMemberRefusal(
            f"anonymous {kind} member{_owner_text(owner)}: its members join the enclosing class "
            "([class.union.anon]), which CrossGL cannot represent; name the member",
            span=parser.source_span_from_tokens(
                parser.tokens[start], parser.tokens[max(after - 1, start)]
            ),
        )
    if kind == "class":
        raise MetalAnonymousMemberRefusal(
            f"unnamed class member{_owner_text(owner)} is not supported; declare it with a name or as a struct",
            span=parser.source_span_from_tokens(
                parser.tokens[start], parser.tokens[max(after - 1, start)]
            ),
        )
    if kind == "struct":
        return parser.parse_struct(pre_alignas, allow_unnamed=True)
    nested = parser.parse_union(allow_unnamed=True)
    if nested is not None and pre_alignas:
        nested.alignas = list(pre_alignas)
    return nested


def _body_index(parser: Any, index: int) -> int:
    index = parser.skip_leading_attribute_tokens_at(index)
    return parser.skip_alignas_specifier_tokens_at(index)


def _owner_text(owner) -> str:
    if not owner:
        return ""
    kind, _, name = str(owner).partition(" ")
    if is_unnamed_aggregate_name(name):
        return f" of an unnamed {kind}"
    return f" of {owner}"
