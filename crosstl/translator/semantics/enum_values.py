"""CrossGL enum values where a target needs an integer: array extents and conversions to an enum type.

Every generator declares a CrossGL enumerator ``E::A`` as an integer constant of its own (``E_A``) and an enum-typed
variable as its underlying integer type, but two uses of an enum still reached the targets as written:

* an array extent that names an enumerator, ``float weights[Mode::Count]``, which HLSL and GLSL emitted verbatim and
  SPIR-V could not size;
* a conversion to the enum type, ``Mode(value)`` (C++ functional cast), which HLSL and Metal emitted as a call to an
  undeclared function.

:func:`lower_enum_values` folds each enumerator extent to its integer, following C's rules (an enumerator without an
initializer is one more than the previous one, C++17 section 10.2), and rewrites each conversion to a conversion to
the underlying type (``int`` unless the enum names one). An extent whose value cannot be computed keeps its spelling.
"""

from __future__ import annotations

from .. import ast as A
from .rewrite import child_slots, get_child, set_child

__all__ = ["enum_values", "lower_enum_values"]

_CARRYING = frozenset({"cgl", "crossgl", ""})
_OPERATORS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "<<": lambda a, b: a << b,
    "|": lambda a, b: a | b,
    "&": lambda a, b: a & b,
}


def _enums(program):
    return [
        node
        for node in getattr(program, "structs", None) or []
        if isinstance(node, A.EnumNode)
    ]


def _integer(node, values):
    """The integer ``node`` denotes, or ``None``: literals, enumerators and their sums, products and shifts."""
    if isinstance(node, A.LiteralNode):
        text = str(node.value).rstrip("uUlL")
        try:
            return int(text, 0)
        except ValueError:
            return None
    if isinstance(node, A.IdentifierNode):
        return values.get(node.name)
    if isinstance(node, A.BinaryOpNode):
        operation = _OPERATORS.get(getattr(node, "operator", getattr(node, "op", None)))
        left = _integer(getattr(node, "left", None), values)
        right = _integer(getattr(node, "right", None), values)
        if operation is None or left is None or right is None:
            return None
        return operation(left, right)
    if (
        isinstance(node, A.UnaryOpNode)
        and getattr(node, "operator", getattr(node, "op", None)) == "-"
    ):
        operand = _integer(getattr(node, "operand", None), values)
        return None if operand is None else -operand
    return None


def enum_values(program) -> dict:
    """``{"E::A": value}`` for every enumerator whose value is an integer expression of earlier ones."""
    values = {}
    for enum in _enums(program):
        next_value = 0
        for variant in getattr(enum, "variants", None) or []:
            value = next_value
            if getattr(variant, "value", None) is not None:
                value = _integer(variant.value, values)
            if value is None:
                next_value = None
                continue
            values[f"{enum.name}::{variant.name}"] = value
            # An initializer may also name an earlier enumerator of its own enum unqualified.
            values.setdefault(variant.name, value)
            next_value = value + 1
    return values


def _underlying(enum):
    underlying = getattr(enum, "underlying_type", None)
    name = getattr(underlying, "name", underlying)
    return str(name) if name else "int"


def lower_enum_values(program, target) -> int:
    """Fold enumerator array extents and rewrite conversions to enum types for ``target``; returns the count."""
    if str(target or "").lower() in _CARRYING or not isinstance(program, A.ShaderNode):
        return 0
    enums = {enum.name: enum for enum in _enums(program) if enum.name}
    if not enums:
        return 0
    values = enum_values(program)
    changed = 0

    def visit(node):
        nonlocal changed
        for field, key in list(child_slots(node)):
            child = get_child(node, field, key)
            visit(child)
            if isinstance(child, A.FunctionCallNode):
                function = getattr(child, "function", None)
                name = function.name if isinstance(function, A.IdentifierNode) else None
                arguments = list(getattr(child, "arguments", None) or [])
                if name in enums and len(arguments) == 1:
                    set_child(
                        node,
                        field,
                        key,
                        A.FunctionCallNode(
                            A.IdentifierNode(_underlying(enums[name])), arguments
                        ),
                    )
                    changed += 1
        for type_field in ("var_type", "vtype", "param_type", "member_type"):
            changed += _fold_extent(getattr(node, type_field, None), values)

    visit(program)
    return changed


def _fold_extent(type_node, values) -> int:
    changed = 0
    while isinstance(type_node, A.ArrayType):
        size = getattr(type_node, "size", None)
        value = _integer(size, values) if isinstance(size, A.ASTNode) else None
        if value is not None and not isinstance(size, A.LiteralNode):
            type_node.size = A.LiteralNode(value, A.PrimitiveType("int"))
            changed += 1
        type_node = getattr(type_node, "element_type", None)
    return changed
