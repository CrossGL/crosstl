"""Undeclared type names: a CrossGL program that names a type it never declares is refused.

A type written as a plain name (``Foo v;``, ``Foo(1.0)``, ``(Foo)x``) is a CrossGL built-in or a structure,
enumeration, type alias, constant block or generic parameter the program declares. Any other name has no meaning, and
the generators would copy it into the target verbatim, where it is at best a compile error and at worst a different
type of the same spelling. :func:`refuse_undeclared_types` raises :class:`UndeclaredTypeError` for the first such
name, so the translation fails with a typed diagnostic instead.

Only plain names in code a stage entry can reach are checked (every function of a program without entries), and
trait bounds of generic parameters are not types. Names with generic arguments (``Texture2D<float4>``), namespace or member qualifiers
(``metal::texture2d``, ``Outer.Inner``) and parenthesised parts are resource or library spellings the generators
resolve themselves.
"""

from __future__ import annotations

import re

from .. import ast as A
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic
from .builtin_type_names import BUILTIN_TYPE_NAMES

__all__ = [
    "UndeclaredTypeError",
    "declared_type_names",
    "refuse_undeclared_types",
    "undeclared_type_uses",
]

_PLAIN_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class UndeclaredTypeError(TranslationDiagnostic):
    """A type name the program neither declares nor CrossGL defines."""

    def __init__(self, target, name, *, source_location=None):
        self.source_location = source_location
        self.type_name = name
        super().__init__(
            f"type '{name}' is not declared: it is neither a CrossGL built-in type nor a structure, "
            "enumeration, type alias or generic parameter of the program",
            target=target,
            code=f"{CODE_PREFIX}undeclared-type",
        )


def _children(node):
    for field, value in vars(node).items():
        if field in ("parent", "annotations", "source_location"):
            continue
        if isinstance(value, A.StageMap):
            yield from value.values()
        elif isinstance(value, A.ASTNode):
            yield value
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, A.ASTNode):
                    yield item
        elif isinstance(value, dict):
            for item in value.values():
                if isinstance(item, A.ASTNode):
                    yield item


def _nodes(program, skip=frozenset()):
    """Every node reachable from ``program``, type nodes included, each once, in a stable order.

    The nodes whose ids are in ``skip`` are left out with everything below them.
    """
    seen = set(skip)
    stack = [program]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        stack.extend(reversed(list(_children(node))))


def _short(name):
    return str(name).rsplit("::", 1)[-1].rsplit(".", 1)[-1]


def declared_type_names(program) -> frozenset:
    """The type names ``program`` declares: structures, constant blocks, enumerations, aliases, generic parameters."""
    names = set()
    for node in _nodes(program):
        name = getattr(node, "name", None)
        if not isinstance(name, str) or not name:
            continue
        if isinstance(
            node, (A.StructNode, A.EnumNode, A.GenericParameterNode, A.BufferNode)
        ):
            names.update((name, _short(name)))
        elif isinstance(node, A.GenericType):
            names.add(name)
        elif isinstance(node, A.VariableNode) and getattr(node, "is_type_alias", False):
            names.update((name, _short(name)))
    return frozenset(names)


def _function_lists(program):
    yield getattr(program, "functions", None) or []
    for stage in (getattr(program, "stages", None) or {}).values():
        yield getattr(stage, "local_functions", None) or []


def _referenced_names(node, skip):
    """The identifier and callee names used under ``node``, not descending into the nodes in ``skip``."""
    names = set()
    for child in _nodes(node, skip):
        if isinstance(child, A.IdentifierNode) and isinstance(child.name, str):
            names.add(child.name)
        elif isinstance(child, A.FunctionCallNode) and isinstance(child.function, str):
            names.add(child.function)
    return names


def _dead_functions(program):
    """Ids of the functions no stage entry reaches, which the generators drop; none when there is no entry.

    A function is live when it is a stage's entry point, or when a live function, a stage or a module-level
    declaration names it. A program without a stage entry is a library, so every function is checked.
    """
    helpers = {}
    for functions in _function_lists(program):
        for function in functions:
            if isinstance(function, A.FunctionNode) and isinstance(function.name, str):
                helpers.setdefault(function.name, []).append(function)
    entries = {
        id(stage.entry_point)
        for stage in (getattr(program, "stages", None) or {}).values()
        if isinstance(getattr(stage, "entry_point", None), A.FunctionNode)
    }
    if not entries:
        return set()
    every = {id(function) for group in helpers.values() for function in group}
    skip = every - entries
    live = set(entries)
    pending = sorted(_referenced_names(program, skip))
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        for function in helpers.get(name, []):
            if id(function) in live:
                continue
            live.add(id(function))
            pending.extend(sorted(_referenced_names(function, skip - {id(function)})))
    return every - live


def undeclared_type_uses(program):
    """``(name, node)`` for each plain type name in ``program`` that is neither declared nor built in.

    Functions no stage entry reaches are not read: they cannot change what the program computes, and translation
    for a selected entry point drops them (a project's explicit specialization that was materialized under another
    name keeps its unresolved parameter types, for example).
    """
    declared = declared_type_names(program)
    # Trait bounds (``generic<T: Numeric + Copy>``) constrain a parameter; they are not types any generator emits.
    bounds = {
        id(bound)
        for node in _nodes(program)
        if isinstance(node, A.GenericParameterNode)
        for bound in node.constraints
    }
    uses = []
    for node in _nodes(program, _dead_functions(program) | bounds):
        if not isinstance(node, A.NamedType) or node.generic_args:
            continue
        name = node.name
        if not isinstance(name, str) or not _PLAIN_NAME.fullmatch(name):
            continue
        if name in declared or name in BUILTIN_TYPE_NAMES:
            continue
        uses.append((name, node))
    return uses


def refuse_undeclared_types(program, target):
    """Raise :class:`UndeclaredTypeError` for the first undeclared plain type name in ``program``."""
    for name, node in undeclared_type_uses(program):
        raise UndeclaredTypeError(
            target, name, source_location=getattr(node, "source_location", None)
        )
