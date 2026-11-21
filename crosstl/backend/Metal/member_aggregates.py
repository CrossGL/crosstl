"""Unnamed aggregates declared inside a struct, class or union.

A member declaration may define an unnamed class and declare members of it at once, as in
``struct S { union { float a; int b; } u; };`` ([class.mem], [dcl.pre]). The unnamed class is parsed like a named
nested aggregate under a reserved name, and its declarators become members of the enclosing class.

An unnamed union that declares no member is an anonymous union: its members are members of the enclosing class and
share its storage ([class.union.anon]). An unnamed struct that declares no member is not standard C++ ([class.mem]
requires a declarator), but Clang accepts it as an extension with the same meaning: its members follow each other
in the enclosing class, or inside the storage of an enclosing union. CrossGL has no anonymous members, so the
importer gives each one a reserved member name (:func:`anonymous_member_declaration`), which keeps the layout of the
class, and the CrossGL writer reaches the promoted members through it: ``b.d`` of
``struct B { union { struct { half d; half m; }; half2 dm; }; }`` becomes ``b.<union>.<struct>.d``
(:func:`anonymous_member_paths`). An unnamed class that declares no member is still refused: its members would be
private ([class.access]/1), which a promoted access cannot honour.
"""

from __future__ import annotations

from typing import Any

from crosstl.translator.diagnostics import CODE_PREFIX
from crosstl.translator.refusals import MetalSourceRefusal

from .local_aggregate_declarators import (
    UNNAMED_AGGREGATE_PREFIX,
    is_unnamed_aggregate_name,
)
from .MetalAst import MemberAccessNode, VariableNode

__all__ = [
    "MetalAnonymousMemberRefusal",
    "anonymous_member_declaration",
    "anonymous_member_paths",
    "promoted_member_access",
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
    anonymous = after >= len(parser.tokens) or parser.tokens[after][0] == "SEMICOLON"
    if anonymous and kind == "class":
        raise MetalAnonymousMemberRefusal(
            f"anonymous class member{_owner_text(owner)}: its members join the enclosing class as private "
            "members ([class.access]), which CrossGL cannot represent; name the member or declare it as a struct",
            span=parser.source_span_from_tokens(
                parser.tokens[start], parser.tokens[max(after - 1, start)]
            ),
        )
    if anonymous:
        if kind == "struct":
            return parser.parse_struct(pre_alignas, allow_unnamed=True, anonymous=True)
        nested = parser.parse_union(allow_unnamed=True, anonymous=True)
        if nested is not None and pre_alignas:
            nested.alignas = list(pre_alignas)
        return nested
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


def anonymous_member_declaration(parser: Any, aggregate: Any) -> VariableNode:
    """The member that holds an anonymous union or struct of a class, under a reserved name."""

    ordinal = str(getattr(aggregate, "name", "")).rsplit("_", 1)[-1]
    member = VariableNode(aggregate.name, f"{UNNAMED_AGGREGATE_PREFIX}member_{ordinal}")
    member.array_sizes = []
    member.anonymous_aggregate_member = True
    aggregate.anonymous_member = True
    parser.register_local_variable_name(member.name)
    return member


def anonymous_member_paths(struct_node: Any, aggregates: dict) -> dict[str, list[str]]:
    """Each member an anonymous member promotes into ``struct_node``, with the member names that reach it.

    ``aggregates`` maps the reserved names of the unnamed aggregates to their nodes. A promoted member of a nested
    anonymous member is reached through both reserved names, outermost first.
    """

    paths: dict[str, list[str]] = {}
    for member in getattr(struct_node, "members", None) or []:
        if not getattr(member, "anonymous_aggregate_member", False):
            continue
        nested = aggregates.get(getattr(member, "vtype", None))
        if nested is None:
            continue
        for inner in getattr(nested, "members", None) or []:
            name = getattr(inner, "name", None)
            if not isinstance(inner, VariableNode) or not name:
                continue
            if getattr(inner, "anonymous_aggregate_member", False):
                for promoted, path in anonymous_member_paths(
                    nested, aggregates
                ).items():
                    paths.setdefault(promoted, [member.name, *path])
                continue
            paths.setdefault(name, [member.name, name])
    return paths


def promoted_member_access(codegen: Any, expression: Any) -> MemberAccessNode | None:
    """The access through reserved member names that a member access of a promoted member stands for, or ``None``.

    ``b->d`` with ``d`` promoted by an anonymous union of ``b``'s class becomes ``b-><union>.d``, which the writer and
    the type inference then handle like any access of a named member.
    """

    paths = getattr(codegen, "anonymous_member_paths", None)
    if not paths or not isinstance(expression, MemberAccessNode):
        return None
    cached = getattr(expression, "promoted_member_access", None)
    if cached is not None:
        return cached
    member = str(expression.member)
    try:
        object_type = codegen.expression_metal_type(expression.object)
    except (TypeError, ValueError):
        return None
    if not object_type:
        return None
    owner = codegen.normalized_metal_type(codegen.resolve_type_alias(object_type))
    path = paths.get(owner, {}).get(member)
    if not path:
        return None
    access = MemberAccessNode(
        expression.object, path[0], getattr(expression, "is_pointer", False)
    )
    for name in path[1:]:
        access = MemberAccessNode(access, name)
    for node in _access_chain(access):
        node.source_location = getattr(expression, "source_location", None)
    expression.promoted_member_access = access
    return access


def _access_chain(access: MemberAccessNode):
    while isinstance(access, MemberAccessNode):
        yield access
        access = access.object
