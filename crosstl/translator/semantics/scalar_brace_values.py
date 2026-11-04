"""Empty brace initializers of scalars: ``float one = {};`` becomes ``float one = 0.0;``.

C and C++ value-initialize a scalar from an empty brace list, so ``float one = {};`` and ``int n = {};`` are zero.
CUDA and HIP input import them as ``var one: f32 = {};``. Only Metal reads an empty brace list as a scalar's zero:
HLSL copied it, which DXC rejects, WGSL wrote ``array()``, and GLSL refused it as an aggregate.
:func:`zero_empty_scalar_initializers` writes the scalar's zero literal in place of the empty list, for every target,
in variable and constant declarations whose type is ``float``, ``double``, ``half``, ``int``, ``uint``, ``bool``, an
8-, 16- or 64-bit integer (``i16 s = {};`` becomes ``i16 s = 0;``, as written by hand), or an alias of one
(``typedef float real_t; real_t one = {};``).

An enumeration is value-initialized to zero as well. For a plain enumeration, whose variants carry no data,
``Mode m = {};`` becomes ``Mode m = 0;``: the generators write a variable of a plain enumeration as ``int`` (``i32``
in WGSL), whatever type the enumeration names (``enum Mode : uint``), so the zero is ``int``'s. Two targets are left
alone: CUDA output declares the variable with the enumeration's own type, for which ``Mode m = {};`` is already zero
and ``Mode m = 0;`` is an error, and a Rust enum has no integer value. An enumeration whose variants carry data is a
structure on most targets and keeps its braces.
"""

from __future__ import annotations

from .. import ast as A
from ..codegen.enum_utils import collect_plain_enums

__all__ = ["zero_empty_scalar_initializers"]

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


def zero_empty_scalar_initializers(program, target) -> int:
    """Replace each empty brace initializer of a scalar declaration with the scalar's zero; return how many."""
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
    if target not in _ENUM_TYPED_TARGETS:
        for enum in collect_plain_enums(nodes):
            if isinstance(enum.name, str):
                aliases.setdefault(enum.name, "int")
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
        if not isinstance(value, A.ArrayLiteralNode) or value.elements:
            continue
        scalar = _scalar_name(getattr(node, type_field, None), aliases)
        if scalar is None:
            continue
        zero = A.LiteralNode(_ZERO_LITERALS[scalar], A.PrimitiveType(scalar))
        setattr(node, value_field, zero)
        count += 1
    return count


def _scalar_name(type_node, aliases):
    """The CrossGL scalar ``type_node`` names, through aliases, or None when it is not a scalar."""
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
            return _SCALAR_NAMES[name]
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
