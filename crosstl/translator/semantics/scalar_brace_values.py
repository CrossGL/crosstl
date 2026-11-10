"""Scalar brace initializers: ``float one = {};`` becomes ``float one = 0.0;``.

C and C++ value-initialize a scalar from an empty brace list, so ``float one = {};`` and ``int n = {};`` are zero.
CUDA and HIP input import them as ``var one: f32 = {};``. Only Metal reads an empty brace list as a scalar's zero:
HLSL copied it, which DXC rejects, WGSL wrote ``array()``, and GLSL refused it as an aggregate.
:func:`zero_empty_scalar_initializers` writes the scalar's zero literal in place of the empty list, for every target,
in variable and constant declarations whose type is ``float``, ``double``, ``half``, ``int``, ``uint``, ``bool``, an
8-, 16- or 64-bit integer (``i16 s = {};`` becomes ``i16 s = 0;``, as written by hand), or an alias of one
(``typedef float real_t; real_t one = {};``).

A singleton scalar list becomes its sole value, converted to the destination type.
Nested singleton lists are unwrapped too; an empty inner list is zero. More than one
value is refused with :class:`ScalarBraceInitializerError` before code generation.

An enumeration is value-initialized to zero as well. For a plain enumeration, whose variants carry no data,
``Mode m = {};`` becomes ``Mode m = 0;``: the generators write a variable of a plain enumeration as ``int`` (``i32``
in WGSL), whatever type the enumeration names (``enum Mode : uint``), so the zero is ``int``'s. Two targets are left
alone: CUDA output declares the variable with the enumeration's own type, for which ``Mode m = {};`` is already zero
and ``Mode m = 0;`` is an error, and a Rust enum has no integer value. An enumeration whose variants carry data is a
structure on most targets and keeps its braces.

A singleton list of a plain enumeration becomes its sole value on every target, unconverted, as if
``Mode m = Mode::B;`` were written; nested singleton lists are unwrapped too. An empty inner list is zero where the
enumeration is written as ``int``, and stays an empty list on CUDA and Rust, which value-initialize it themselves.
More than one value is refused with :class:`ScalarBraceInitializerError`.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.enum_utils import collect_plain_enums
from ..diagnostics import CODE_PREFIX, TranslationDiagnostic

__all__ = ["ScalarBraceInitializerError", "zero_empty_scalar_initializers"]

_SKIPPED_FIELDS = frozenset({"parent", "annotations", "source_location"})
# CrossGL output keeps the program as written.
_CARRYING = frozenset({"cgl", "crossgl", ""})
# Targets whose generators declare an enumeration-typed variable with the enumeration's own type.
_ENUM_TYPED_TARGETS = frozenset({"cuda", "rust"})
# Scalar spellings, CrossGL and WGSL, by the CrossGL name of their zero's type. An 8-, 16- or 64-bit integer's zero is
# the 32-bit literal of its signedness, as in ``i64 a = 0;`` and ``u8 c = 0u;``.
_SCALAR_NAMES = {
    "float": "float",
    "f32": "float",
    "double": "double",
    "f64": "double",
    "half": "half",
    "f16": "half",
    "int": "int",
    "i32": "int",
    "i8": "int",
    "i16": "int",
    "i64": "int",
    "uint": "uint",
    "u32": "uint",
    "u8": "uint",
    "u16": "uint",
    "u64": "uint",
    "bool": "bool",
}
_ZERO_LITERALS = {
    "float": 0.0,
    "double": 0.0,
    "half": 0.0,
    "int": 0,
    "uint": 0,
    "bool": False,
}


class ScalarBraceInitializerError(TranslationDiagnostic):
    """A scalar declaration has more than one brace-initializer value."""

    # The program is invalid C and C++ whatever the target, so the code names no target.
    code = CODE_PREFIX + "scalar-brace-initializer-invalid"

    def __init__(self, target, name, count, *, source_location=None):
        self.source_location = source_location
        self.missing_capabilities = ()
        super().__init__(
            f"scalar brace initializer for '{name}' requires at most one value, got {count}",
            target=target,
        )


def zero_empty_scalar_initializers(program, target) -> int:
    """Normalize scalar declaration brace lists to zero or one converted value; return how many."""
    target = str(target or "").lower()
    if target in _CARRYING:
        return 0
    nodes = list(_nodes(program))
    aliases = {
        node.name: node.var_type
        for node in nodes
        if isinstance(node, A.VariableNode)
        and getattr(node, "is_type_alias", False)
        and isinstance(node.name, str)
    }
    enums = {
        enum.name for enum in collect_plain_enums(nodes) if isinstance(enum.name, str)
    }
    empty_aliases = dict(aliases)
    if target not in _ENUM_TYPED_TARGETS:
        for name in enums:
            empty_aliases.setdefault(name, "int")
    count = 0
    for node in nodes:
        if isinstance(node, A.VariableNode) and not getattr(
            node, "is_type_alias", False
        ):
            type_field, value_field = "var_type", "initial_value"
        elif isinstance(node, A.ConstantNode):
            type_field, value_field = "const_type", "value"
        else:
            continue
        value = getattr(node, value_field, None)
        if not isinstance(value, A.ArrayLiteralNode):
            continue
        declared = getattr(node, type_field, None)
        if value.elements and _enum_name(declared, aliases, enums) is not None:
            setattr(node, value_field, _enum_value(target, node, value, enums))
            count += 1
            continue
        # Only empty enum lists are integers on enum-erasing targets.
        scalar_type = _scalar_type(
            declared, aliases if value.elements else empty_aliases
        )
        if scalar_type is None:
            continue
        scalar = _SCALAR_NAMES[scalar_type.name]
        while isinstance(value, A.ArrayLiteralNode):
            if len(value.elements) > 1:
                raise ScalarBraceInitializerError(
                    target,
                    node.name,
                    len(value.elements),
                    source_location=getattr(value, "source_location", None),
                )
            if not value.elements:
                value = A.LiteralNode(_ZERO_LITERALS[scalar], A.PrimitiveType(scalar))
                break
            value = value.elements[0]
        else:
            # A constructor preserves the destination's width and signedness;
            # later target legalization handles half and narrow integer types.
            value_type = _scalar_type(getattr(value, "literal_type", None), {})
            if (
                not isinstance(value_type, A.PrimitiveType)
                or value_type.name != scalar_type.name
            ):
                value = A.FunctionCallNode(A.IdentifierNode(scalar_type.name), [value])
        setattr(node, value_field, value)
        count += 1
    return count


def _enum_value(target, node, value, enums):
    """Unwrap a plain enumeration's singleton list to its value, which keeps its own type."""
    while isinstance(value, A.ArrayLiteralNode):
        if len(value.elements) > 1:
            raise ScalarBraceInitializerError(
                target,
                node.name,
                len(value.elements),
                source_location=getattr(value, "source_location", None),
            )
        if not value.elements:
            # CUDA and Rust value-initialize the enumeration from ``{}``; elsewhere it is ``int``.
            if target in _ENUM_TYPED_TARGETS:
                return value
            return A.LiteralNode(0, A.PrimitiveType("int"))
        value = value.elements[0]
    return value


def _enum_name(type_node, aliases, enums):
    """Resolve a plain enumeration's name through aliases."""
    seen = set()
    while True:
        if isinstance(type_node, (A.PrimitiveType, A.NamedType)) and not getattr(
            type_node, "generic_args", None
        ):
            type_node = type_node.name
        if not isinstance(type_node, str) or type_node in seen:
            return None
        if type_node in enums:
            return type_node
        seen.add(type_node)
        type_node = aliases.get(type_node)


def _scalar_type(type_node, aliases):
    """Resolve a scalar through aliases without losing its width or signedness."""
    seen = set()
    while True:
        if isinstance(type_node, str):
            name = type_node
        elif isinstance(type_node, (A.PrimitiveType, A.NamedType)) and not getattr(
            type_node, "generic_args", None
        ):
            name = type_node.name
        else:
            return None
        if not isinstance(name, str):
            return None
        if name in _SCALAR_NAMES:
            return A.PrimitiveType(
                name
                if name in {"i8", "i16", "i64", "u8", "u16", "u64"}
                else _SCALAR_NAMES[name]
            )
        if name in seen or name not in aliases:
            return None
        seen.add(name)
        type_node = aliases[name]


def _nodes(program):
    seen = set()
    stack = [program]
    while stack:
        node = stack.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for field, value in vars(node).items():
            if field in _SKIPPED_FIELDS:
                continue
            if isinstance(value, A.StageMap):
                stack.extend(value.values())
            elif isinstance(value, A.ASTNode):
                stack.append(value)
            elif isinstance(value, (list, tuple)):
                stack.extend(item for item in value if isinstance(item, A.ASTNode))
            elif isinstance(value, dict):
                stack.extend(
                    item for item in value.values() if isinstance(item, A.ASTNode)
                )
