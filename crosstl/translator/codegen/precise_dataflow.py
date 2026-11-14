"""Variables whose values flow into a precise result, for generators that decorate each operation.

GLSL 4.60 section 4.7.1 and HLSL ``precise`` apply to every operation that contributes to the value of a precise
variable, not only to the expression written in its declaration: in

    float t = a * b;
    precise float r = t - c;

``a * b`` contributes to ``r`` and must not be fused with the subtraction either. glslang and DXC propagate the
property backwards through the data flow of the function. A generator that emits each operation itself, such as the
SPIR-V generator with its ``NoContraction`` decorations, has to do the same.

:func:`precise_contributors` computes, for one function body, the names of the variables whose stored values reach
a precise variable through assignments and initializers. The analysis is by name and ignores control flow and
scopes, so it over-approximates: evaluating an operation without contraction is always a valid evaluation of it,
because contraction is only ever permitted, never required.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from ..ast import (
    ArrayAccessNode,
    AssignmentNode,
    BinaryOpNode,
    IdentifierNode,
    MemberAccessNode,
    PointerAccessNode,
    SwizzleNode,
    VariableNode,
)

__all__ = ["declaration_is_precise", "precise_contributors"]

_ASSIGNMENT_OPERATORS = frozenset(
    {"=", "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>="}
)


def declaration_is_precise(node: Any) -> bool:
    """Return whether a declaration carries ``precise`` as a qualifier or as an attribute."""
    qualifiers = {str(item).lower() for item in getattr(node, "qualifiers", None) or ()}
    if "precise" in qualifiers:
        return True
    return any(
        str(getattr(attribute, "name", "")).lower() == "precise"
        for attribute in getattr(node, "attributes", None) or ()
    )


def precise_contributors(
    body: Any,
    seeds: Iterable[str] = (),
    *,
    is_precise: Callable[[Any], bool] = declaration_is_precise,
) -> frozenset:
    """Return the precise variables of ``body`` and every variable whose value flows into one.

    ``seeds`` names variables declared outside ``body`` that are precise, such as precise globals and parameters.
    """
    definitions: list[tuple[str, list]] = []
    precise = set(seeds)
    for node in _walk(body):
        if isinstance(node, VariableNode):
            name = getattr(node, "name", None)
            if not isinstance(name, str):
                continue
            if is_precise(node):
                precise.add(name)
            if getattr(node, "initial_value", None) is not None:
                definitions.append((name, [node.initial_value]))
        elif isinstance(node, AssignmentNode) or (
            isinstance(node, BinaryOpNode)
            and str(node.operator) in _ASSIGNMENT_OPERATORS
        ):
            # An assignment statement, or an assignment used as an expression (``a = b = c``).
            target = node.target if isinstance(node, AssignmentNode) else node.left
            value = node.value if isinstance(node, AssignmentNode) else node.right
            name = _root_name(target)
            if name is None:
                continue
            reads = [value]
            if str(getattr(node, "operator", "=")) != "=":
                reads.append(target)
            definitions.append((name, reads))
    if not precise:
        return frozenset()
    changed = True
    while changed:
        changed = False
        for name, reads in definitions:
            if name not in precise:
                continue
            for read in reads:
                for identifier in _read_names(read):
                    if identifier not in precise:
                        precise.add(identifier)
                        changed = True
    return frozenset(precise)


def _root_name(target: Any) -> str | None:
    """Return the variable an assignment target stores into, if it names one."""
    while True:
        if isinstance(target, str):
            return target
        if isinstance(target, IdentifierNode):
            return target.name
        if isinstance(target, MemberAccessNode):
            target = target.object_expr
        elif isinstance(target, PointerAccessNode):
            target = target.pointer_expr
        elif isinstance(target, ArrayAccessNode):
            target = target.array_expr
        elif isinstance(target, SwizzleNode):
            target = target.vector_expr
        else:
            return None


def _read_names(expression: Any) -> set:
    """Return the identifiers ``expression`` reads."""
    names = set()
    for node in _walk(expression):
        if isinstance(node, IdentifierNode) and isinstance(node.name, str):
            names.add(node.name)
    return names


def _walk(root: Any):
    """Yield every AST node under ``root`` once, in a deterministic order, ignoring parent links."""
    stack = [root]
    seen = set()
    while stack:
        node = stack.pop()
        if node is None or isinstance(node, (str, bytes, int, float, bool)):
            continue
        if isinstance(node, (list, tuple)):
            stack.extend(reversed(node))
            continue
        if isinstance(node, dict):
            stack.extend(reversed(list(node.values())))
            continue
        if id(node) in seen:
            continue
        seen.add(id(node))
        fields = getattr(node, "__dict__", None)
        if fields is None:
            continue
        yield node
        for key, value in fields.items():
            if key in {"parent", "parent_node", "source_location"}:
                continue
            stack.append(value)
